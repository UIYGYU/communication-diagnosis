import argparse
import asyncio
import json
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from mcp import Client, StdioServerParameters
from pydantic import ValidationError

from src.tools.contracts import ARGUMENT_MODELS, TOOLS
from src.tools.local import LocalTools
import os

TIMEOUT_SECONDS = 15
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"

def tool_error(code: str, message: str) -> dict:
    return {"ok": False, "error": code, "message": message}


def without_annotations(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}

        for key, item in value.items():
            if key in ("title", "description"):
                continue

            result[key] = without_annotations(item)
        return result
    if isinstance(value, list):
        result = []

        for item in value:
            result.append(without_annotations(item))

        return result
    return value


def validate_discovered_tools(listed: Any) -> None:
    remote = {tool.name: tool.input_schema for tool in listed.tools}
    expected = {tool["name"]: tool["parameters"] for tool in TOOLS}
    if listed.next_cursor is not None:
        raise ValueError("工具列表存在下一页，当前未取得完整清单。")

    if len(remote) != len(listed.tools):
        raise ValueError("服务端返回了重名工具。")

    if set(remote) != set(expected):
        raise ValueError("服务端工具名称与本地约定不一致。")

    for name, schema in expected.items():
        actual = remote[name]

        if actual.get("type") != schema.get("type"):
            raise ValueError(f"工具 {name} 的参数根类型不一致。")

        if set(actual.get("required", [])) != set(schema.get("required", [])):
            raise ValueError(f"工具 {name} 的必填参数不一致。")

        actual_properties = without_annotations(actual.get("properties", {}))

        expected_properties = without_annotations(schema.get("properties", {}))

        if actual_properties != expected_properties:
            raise ValueError(f"工具 {name} 的参数约束不一致。")


class MCPTools:
    def __init__(self, client: Client, *, transport: str, manifest: dict):
        self.client = client
        self.transport = transport
        self.tools = deepcopy(TOOLS)
        self.expected_files = manifest["files"]
        self._remote_clients: dict[str, Client] = {}

    async def add_remote_tools(self, client: Client) -> None:
        names = {tool["name"] for tool in self.tools}
        cursor = None
        while True:
            listed = await client.list_tools(cursor=cursor)
            for tool in listed.tools:
                if tool.name in names:
                    raise ValueError(f"MCP 工具重名：{tool.name}")
                names.add(tool.name)
                self.tools.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                        "strict": False,
                    }
                )
                self._remote_clients[tool.name] = client

            cursor = listed.next_cursor
            if cursor is None:
                return

    async def _call_remote(self, name:str, arguments:str) -> dict:
        try:
            params = json.loads(arguments)
        except json.JSONDecodeError:
            return tool_error("invalid_arguments", "工具参数需要是 JSON 对象。")
        if not isinstance(params, dict):
            return tool_error("invalid_arguments", "工具参数需要是 JSON 对象。")

        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                result = await self._remote_clients[name].call_tool(name, params)
        except TimeoutError:
            return tool_error("mcp_timeout", "MCP 工具调用超时。")
        except Exception:
            return tool_error("mcp_failed", "MCP 工具通信失败。")

        data = result.structured_content
        if not result.is_error and isinstance(data, dict):
            return {"ok": True, **data}

        payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        if result.is_error:
            return {
                **tool_error("mcp_tool_error", "MCP 服务端报告工具错误。"),
                "response": payload,
            }
        return {"ok": True, "response": payload}

    async def call(self, name: str, arguments: str) -> dict:
        if name in self._remote_clients:
            return await self._call_remote(name, arguments)        
        model = ARGUMENT_MODELS.get(name)
        if not model:
            return tool_error("unknown_tool", "未知工具")
        try:
            args = model.model_validate_json(arguments)
        except (ValidationError, TypeError):
            return tool_error(
                "invalid_arguments", "工具参数不符合要求，未发送 MCP 请求。"
            )

        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                result = await self.client.call_tool(name, args.model_dump(mode="json"))
        except TimeoutError:
            return tool_error("mcp_timeout", "MCP 工具调用超时。")
        except Exception:
            return tool_error("mcp_failed", "MCP 工具通信失败。")

        if result.is_error:
            return tool_error("mcp_failed", "MCP 服务端拒绝本次调用。")

        data = result.structured_content
        if not isinstance(data, dict) or type(data.get("ok")) is not bool:
            return tool_error("invalid_result", "工具结果必须包含布尔类型的 ok。")

        if not data["ok"]:
            return data
        if name == "read_log":
            string_fields = ("role", "file", "input_sha256")

            if (
                not all(isinstance(data.get(key), str) for key in string_fields)
                or type(data.get("truncated")) is not bool
                or not isinstance(data.get("lines"), list)
            ):
                return tool_error("invalid_result", "日志结果的字段或类型不正确。")

            for row in data["lines"]:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("evidence_id"), str)
                    or not row["evidence_id"].strip()
                    or type(row.get("line")) is not int
                    or row["line"] <= 0
                    or not isinstance(row.get("text"), str)
                ):
                    return tool_error(
                        "invalid_result", "日志行的引用、行号或文本不正确。"
                    )
            if data["role"] != args.role:
                return tool_error("invalid_result", "返回的日志角色与请求不一致。")

            expected = self.expected_files.get(args.role)
            if (
                not isinstance(expected, dict)
                or data["file"] != expected["file"]
                or data["input_sha256"] != expected["sha256"]
            ):
                return tool_error(
                    "case_changed",
                    "案例文件与连接时不一致，请核对输入并重新连接。",
                )
        else:
            if not isinstance(data.get("clauses"), list):
                return tool_error("invalid_result", "协议结果的 clauses 必须是列表。")

            string_fields = (
                "section",
                "title",
                "text",
                "protocol",
                "version",
                "content_kind",
                "source_url",
            )

            for clause in data["clauses"]:
                if (
                    not isinstance(clause, dict)
                    or not isinstance(clause.get("clause_id"), str)
                    or not clause["clause_id"].strip()
                    or not all(
                        isinstance(clause.get(key), str) for key in string_fields
                    )
                ):
                    return tool_error(
                        "invalid_result", "协议条目缺少字段或类型不正确。"
                    )

        return data


@asynccontextmanager
async def _open_mcp_client(
    target: str | StdioServerParameters,
    *,
    read_timeout_seconds: float = TIMEOUT_SECONDS,
) -> AsyncGenerator[Client, None]:
    async with Client(target, read_timeout_seconds=read_timeout_seconds) as client:
        yield client


@asynccontextmanager
async def open_toolbox(
    case_dir: Path,
    *,
    transport: Literal["mcp-stdio", "mcp-http"] = "mcp-http",
    mcp_url: str = DEFAULT_MCP_URL,
) -> AsyncGenerator[MCPTools, None]:
    root = case_dir.resolve()
    core = LocalTools(root)
    expected_manifest = await asyncio.to_thread(core.case_manifest)
    if transport == "mcp-stdio":
        target = StdioServerParameters(
            command=sys.executable,
            args=[
                "-B",
                "-X",
                "utf8",
                "-m",
                "src.tools.server",
                "--case-dir",
                str(root),
                "--transport",
                "stdio",
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            env={},
        )
    elif transport == "mcp-http":
        address = urlsplit(mcp_url)
        port = address.port
        if (
            address.scheme != "http"
            or address.hostname != "127.0.0.1"
            or port is None
            or not 1 <= port <= 65535
            or address.path != "/mcp"
            or address.username is not None
            or address.password is not None
            or address.query
            or address.fragment
        ):
            raise ValueError(
                "本练习地址须为 http://127.0.0.1:端口/mcp，"
                "且不能包含账号、查询参数或片段。"
            )
        target = mcp_url
    else:
        raise ValueError("不支持的 MCP 连接方式。")

    failure: Exception | None = None
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                client = await stack.enter_async_context(_open_mcp_client(target))
                listed = await client.list_tools()
                validate_discovered_tools(listed)
                resource = await client.read_resource("diagnosis://case-manifest")
                if len(resource.contents) != 1:
                    raise ValueError("案例清单资源必须恰好包含一项内容。")

                text = getattr(resource.contents[0], "text", None)

                if not isinstance(text, str):
                    raise ValueError("案例清单资源必须是文本。")

                actual_manifest = json.loads(text)
                if actual_manifest != expected_manifest:
                    raise ValueError("服务端案例文件与本地输入不一致。")

                toolbox = MCPTools(
                    client, transport=transport, manifest=expected_manifest
                )
                for url in os.getenv("DIAGNOSIS_EXTRA_MCP_URLS", "").split():
                    remote = await stack.enter_async_context(_open_mcp_client(url))
                    await toolbox.add_remote_tools(remote)
        except NotImplementedError as error:
            failure = error
        except ValueError as error:
            failure = RuntimeError(f"MCP 连接检查失败：{error}")
        except Exception:
            failure = RuntimeError("MCP 连接或发现失败，请检查服务、地址和案例。")
        if failure is None:
            try:
                yield toolbox
            except Exception as error:
                failure = error
    if failure is not None:
        raise failure


# 不走mcp的场景
class AsyncLocalTools:
    def __init__(self, case_dir: Path):
        self.core = LocalTools(case_dir)
        self.tools = self.core.tools
        self.transport = "local"

    async def call(self, name: str, arguments: str) -> dict:
        return await asyncio.to_thread(self.core.call, name, arguments)


@asynccontextmanager
async def open_local_toolbox(case_dir: Path) -> AsyncGenerator[AsyncLocalTools, None]:
    yield AsyncLocalTools(case_dir)


async def check_connection(
    case_dir: Path,
    *,
    transport: Literal["mcp-stdio", "mcp-http"] = "mcp-http",
    mcp_url: str = DEFAULT_MCP_URL,
) -> None:
    local = LocalTools(case_dir)
    checks = [
        ("read_log", {"role": "broker"}),
        ("search_protocol", {"query": "PUBACK"}),
    ]

    async with open_toolbox(case_dir, transport=transport, mcp_url=mcp_url) as toolbox:
        for name, parameters in checks:
            arguments = json.dumps(parameters, ensure_ascii=False)
            expected = local.call(name, arguments)
            actual = await toolbox.call(name, arguments)
            if expected.get("ok") is not True:
                raise RuntimeError(
                    f"本地工具 {name} 调用失败：{expected.get('message')}"
                )

            if actual.get("ok") is not True:
                raise RuntimeError(f"MCP 工具 {name} 调用失败：{actual.get('message')}")

            if actual != expected:
                raise RuntimeError(f"工具 {name} 的本地与 MCP 结果不一致。")
        test = await toolbox.call("test", "test")
        if test.get("ok") is not False or test.get("error") != "unknown_tool":
            raise RuntimeError("未知工具未按预期被拒绝。")

    print("连通验证完成；HTTP 服务会继续运行。本步骤未请求模型。")

def main() -> int:
    parser = argparse.ArgumentParser(description="验证 MCP 工具连接")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    parser.add_argument(
        "--transport", choices=("mcp-stdio", "mcp-http"), default="mcp-http"
    )
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    args = parser.parse_args()
    try:
        asyncio.run(
            check_connection(
                args.case_dir, transport=args.transport, mcp_url=args.mcp_url
            )
        )
    except (ValueError, RuntimeError) as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
