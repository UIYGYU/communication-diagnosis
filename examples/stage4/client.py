"""可复用的 MCP 客户端适配；main 另作不调用模型的连通检查。"""

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from pydantic import ValidationError

from examples.stage3.tools import (
    TOOLS,
    CaseTools,
    ReadLogArguments,
    SearchArguments,
)

TIMEOUT_SECONDS = 15
ARGUMENT_MODELS = {"read_log": ReadLogArguments, "search_protocol": SearchArguments}


def without_annotations(value: Any) -> Any:
    """只忽略 schema 的展示标题和说明，保留类型、枚举、长度等约束。"""
    if isinstance(value, dict):
        return {
            key: without_annotations(item)
            for key, item in value.items()
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [without_annotations(item) for item in value]
    return value


def validation_tool_contracts(listed: Any) -> None:
    remote = {tool.name: tool.input_schema for tool in listed.tools}
    expected = {tool["name"]: tool["parameters"] for tool in TOOLS}
    if (
        listed.next_cursor is not None
        or len(remote) != len(listed.tools)
        or set(remote) != set(expected)
    ):
        raise ValueError("MCP 工具列表不符合本项目约定，请检查是否启动了整合版服务端。")
    for name, schema in expected.items():
        actual = remote[name]
        if (
            actual.get("type") != schema["type"]
            or set(actual.get("required", [])) != set(schema["required"])
            or without_annotations(actual.get("properties"))
            != without_annotations(schema["properties"])
        ):
            raise ValueError(f"MCP 工具 {name} 的参数约束与本地声明不一致。")
    # SDK 从函数签名生成的根 schema 不声明 additionalProperties:false。
    # 因此这里只比较已登记字段的约束，额外字段由下面的 Pydantic 校验拒绝。


class MCPTools:
    transport = "mcp_stdio"

    def __init__(self, client: Client) -> None:
        self.client = client
        self.tools = TOOLS
        self.evidence: dict[str, dict] = {}
        self.clauses: dict[str, dict] = {}

    async def call(self, name: str, arguments: str) -> dict:
        model = ARGUMENT_MODELS.get(name)
        if model is None:
            return {
                "ok": False,
                "error": "未知工具；仅支持 read_log 和 search_protocol",
            }
        try:
            args = model.model_validate_json(arguments)
        except (ValidationError, TypeError):
            return {"ok": False, "error": "工具参数不符合要求，未发送 MCP 请求"}

        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                result = await self.client.call_tool(name, args.model_dump(mode="json"))
        except TimeoutError:
            return {"ok": False, "error": "MCP 工具调用超时，本次未自动重试"}
        except Exception:
            # 不把 SDK 的原始异常或服务端原始错误正文放进模型上下文。
            return {"ok": False, "error": "MCP 工具调用失败，请查看本机服务端状态"}

        if result.is_error:
            return {"ok": False, "error": "MCP 服务端拒绝本次工具调用"}
        data = result.structured_content
        if not isinstance(data, dict) or type(data.get("ok")) is not bool:
            return {"ok": False, "error": "MCP 工具没有返回约定的结构化结果"}
        if not data["ok"]:
            # 本项目两项工具的失败只含业务提示；不接收任意原始异常正文。
            return {
                "ok": False,
                "error": "工具未取得有效结果，请检查日志格式、角色或查询",
            }

        field, id_field, registry = (
            ("records", "evidence_id", self.evidence)
            if name == "read_log"
            else ("clauses", "clause_id", self.clauses)
        )
        rows = data.get(field)
        if not isinstance(rows, list):
            return {"ok": False, "error": "MCP 工具返回的证据或条款列表格式错误"}
        pending = {}
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get(id_field), str)
                or not row[id_field]
            ):
                return {"ok": False, "error": "MCP 工具返回了缺少标识的证据或条款"}
            identity = row[id_field]
            if (identity in pending and pending[identity] != row) or (
                identity in registry and registry[identity] != row
            ):
                return {"ok": False, "error": "同一引用标识对应了不同内容"}
            pending[identity] = row
        # 全部记录校验完才写入，失败结果不污染本次引用集合。
        registry.update(pending)
        return data


@asynccontextmanager
async def open_toolbox(case_dir: Path) -> AsyncGenerator[MCPTools]:
    root = case_dir.resolve()
    if not root.is_dir():
        raise ValueError("案例目录不存在或不是文件夹")
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-B",
            "-X",
            "utf8",
            "-m",
            "examples.stage4.server",
            "--case-dir",
            str(root),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        # SDK 仅继承标准运行环境；不转发应用的模型密钥和代理配置。
        env={},
    )
    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                client = await stack.enter_async_context(
                    Client(params, read_timeout_seconds=TIMEOUT_SECONDS)
                )
                listed = await client.list_tools()
                validation_tool_contracts(listed)
        except Exception:
            raise RuntimeError(
                "MCP 连接或工具发现失败，请检查整合版服务端、依赖和案例目录。"
            ) from None
        yield MCPTools(client)


async def check_connection(case_dir: Path) -> None:
    """固定调用仅用于此处的连通检查；Agent 使用 MCPTools.call 动态调用。"""
    local = CaseTools(case_dir)
    async with open_toolbox(case_dir) as toolbox:
        print("已发现工具：", [tool["name"] for tool in toolbox.tools])
        for role in local.snapshots:
            arguments = json.dumps({"role": role})
            remote = await toolbox.call("read_log", arguments)
            expected = local.call("read_log", arguments)
            if remote.get("ok") is not True or remote != expected:
                raise RuntimeError("日志工具未成功，或与本地结果不一致。")
        arguments = json.dumps({"query": "PUBACK"})
        if await toolbox.call("search_protocol", arguments) != local.call(
            "search_protocol", arguments
        ):
            raise RuntimeError("协议检索与本地结果不一致。")
        rejected = await toolbox.call("not_registered", "{}")
        if rejected.get("ok") is not False:
            raise RuntimeError("未知工具没有被拒绝。")
    print("MCP 连通检查通过；与本地工具结果一致，子进程已关闭。此步骤未调用模型。")


def main() -> int:
    parser = argparse.ArgumentParser(description="验证整合版 MCP 工具，不请求模型")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    args = parser.parse_args()
    try:
        asyncio.run(check_connection(args.case_dir))
    except Exception:
        print("MCP 连通检查失败，请检查案例文件及客户端、服务端是否均已更新。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
