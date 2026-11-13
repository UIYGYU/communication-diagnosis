"""LangGraph 支线入口：图保存会话状态，每问重新建立 MCP 工具连接。"""

import argparse
import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from openai import APIError, OpenAI

from examples.stage3.llm_client import api_error_message, create_client
from examples.stage3.tools import CLAUSES, FILES, TOOLS, CaseTools
from examples.stage4.agent_demo import INSTRUCTIONS
from examples.stage4.client import open_toolbox
from examples.stage4.service import (
    DiagnosisRequest,
    UploadedLog,
    input_manifest,
    validate_log_contents,
)
from experiments.langgraph_lab.graph_agent import RunContext, build_graph

DEFAULT_QUESTION = (
    "请查看可用的发布端和订阅端日志，检索 PUBACK 教学摘要，"
    "说明能确认的收发情况与证据缺口。"
)


def make_case_id(request: DiagnosisRequest) -> str:
    """绑定日志、知识和提示词；追问文字与本问预算不属于案例身份。"""
    identity = {
        "files": input_manifest(request),
        "protocol_version": "MQTT-5.0",
        "clauses": CLAUSES,
        "instructions": INSTRUCTIONS,
        "tools": TOOLS,
    }
    raw = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def read_from_terminal(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def show_turn(turn: dict) -> None:
    result = turn["result"]
    print(f"第 {turn['turn']} 问：{result['status']}；{result['reason']}")
    for item in result["trace"]:
        if item.get("type") in {"node", "node_enter"}:
            print("  节点：", item.get("name", item.get("node", "未知")))
        elif item.get("type") == "model_request":
            print(
                f"  请求前：{item['history_items']} 项历史，"
                f"{item['context_bytes']} 字节，上限 {item['limit']} 字节"
            )
        elif item.get("type") == "model_response":
            usage = item.get("usage")
            print(
                "  模型返回的 usage："
                + (json.dumps(usage, ensure_ascii=False) if usage else "接口未提供")
            )
    if result.get("report"):
        print(result["report"]["summary"])


def export_result(state: dict, transport: str) -> dict:
    """下载文件保留依据与轨迹，不导出完整历史或运行资源。"""
    return deepcopy(
        {
            "mode": "langgraph_learning_agent",
            "transport": transport,
            "status": state["status"],
            "reason": state["reason"],
            "report": state.get("report"),
            "trace": state.get("trace", []),
            "evidence": state.get("evidence", {}),
            "clauses": state.get("clauses", {}),
            "limits": state.get("limits", {}),
            "history_items": len(state.get("history", [])),
        }
    )


async def run_lab(
    case_dir: Path,
    question: str,
    *,
    thread_id: str | None = None,
    max_questions: int = 3,
    max_rounds: int = 6,
    max_tool_calls: int = 8,
    max_context_bytes: int = 128_000,
    client_factory: Callable[[], tuple[OpenAI, str]] | None = None,
    toolbox_factory: Callable[[Path], AbstractAsyncContextManager[Any]] | None = None,
    read_question: Callable[[str], Awaitable[str]] | None = None,
    observer: Callable[[dict], None] | None = None,
) -> dict:
    """一个内存检查点管理器跨问保留状态；连接每问创建并关闭。"""
    if type(max_questions) is not int or not 1 <= max_questions <= 3:
        raise ValueError("这个练习最多连续处理 1～3 个问题。")
    if type(max_context_bytes) is not int or max_context_bytes < 1:
        raise ValueError("上下文字节上限必须是正整数。")
    if thread_id is not None and (
        not isinstance(thread_id, str) or not thread_id.strip() or len(thread_id) > 128
    ):
        raise ValueError("thread_id 须为 1～128 个字符，不能全为空白。")
    selected_thread = thread_id.strip() if thread_id is not None else uuid4().hex

    # 只读三个受控角色文件，复用现有目录、链接、10 MiB 与格式校验。
    snapshots = CaseTools(case_dir).snapshots
    if not snapshots:
        raise ValueError("案例目录没有提供任何受支持的日志文件。")
    request = DiagnosisRequest(
        question=question,
        files=[
            UploadedLog(role=role, file_name=FILES[role], content=raw)
            for role, raw in snapshots.items()
        ],
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
    )
    validate_log_contents(request)
    question = request.question
    case_id = make_case_id(request)

    # 只创建一次。InMemorySaver 退出进程后丢失；相同 thread_id 不会自动恢复。
    graph = build_graph(checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": selected_thread},
        "recursion_limit": 50,
    }
    result = {
        "mode": "langgraph_lab",
        "thread_id": selected_thread,
        "case_id": case_id,
        "checkpoint_backend": "in_memory",
        "status": "completed",
        "reason": "本次终端练习已结束；内存检查点不会跨进程保留。",
        "input_manifest": input_manifest(request),
        "turns": [],
        "limits": {
            "max_questions": max_questions,
            "max_rounds_per_question": max_rounds,
            "max_tool_calls_per_question": max_tool_calls,
            "max_context_bytes": max_context_bytes,
        },
    }
    client, model = (client_factory or create_client)()
    try:
        with client:
            with TemporaryDirectory(prefix="communication-langgraph-") as temporary:
                root = Path(temporary)
                for role, raw in snapshots.items():
                    (root / FILES[role]).write_bytes(raw)
                for number in range(1, max_questions + 1):
                    # 此处刻意每问重连：会话数据属于图状态，不属于 MCPTools。
                    async with (toolbox_factory or open_toolbox)(root) as toolbox:
                        state = await graph.ainvoke(
                            {"question": question},
                            config=config,
                            context=RunContext(
                                client=client,
                                model=model,
                                toolbox=toolbox,
                                case_id=case_id,
                                max_rounds=max_rounds,
                                max_tool_calls=max_tool_calls,
                                max_context_bytes=max_context_bytes,
                            ),
                        )
                        answer = export_result(state, toolbox.transport)
                    # 已退出本问连接，再展示、等待输入；下问会获得新 MCPTools。
                    turn = {"turn": number, "question": question, "result": answer}
                    result["turns"].append(turn)
                    if observer is not None:
                        observer(turn)
                    if answer["status"] != "completed":
                        result.update(status=answer["status"], reason=answer["reason"])
                        break
                    if number == max_questions:
                        break
                    try:
                        question = (
                            await (read_question or read_from_terminal)(
                                "继续问同一案例（/exit 或直接回车退出）："
                            )
                        ).strip()
                    except EOFError:
                        break
                    if not question or question == "/exit":
                        break
                    if len(question) > 4000:
                        raise ValueError("追问不能超过 4000 个字符。")
    except APIError as error:
        result.update(status="failed", reason=api_error_message(error))
    except ValueError as error:
        result.update(status="failed", reason=str(error))
    except Exception as error:
        # 不把模型、MCP 或图执行的原始异常正文带入报告。
        result.update(
            status="failed",
            reason=f"图或工具连接执行失败（{type(error).__name__}）；本次会话已停止。",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="LangGraph 编排与会话对照练习")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--thread-id")
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-context-bytes", type=int, default=128_000)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/langgraph_lab/session-report.json")
    )
    args = parser.parse_args()
    try:
        output_root = Path("outputs").resolve()
        output = args.out.resolve()
        if not output.is_relative_to(output_root) or output == output_root:
            raise ValueError("--out 必须位于项目 outputs 文件夹内。")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"status": "not_started", "turns": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        result = asyncio.run(
            run_lab(
                args.case_dir,
                args.question,
                thread_id=args.thread_id,
                max_questions=args.max_questions,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                max_context_bytes=args.max_context_bytes,
                observer=show_turn,
            )
        )
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"会话状态：{result['status']}；报告与轨迹：{output}")
        return 0 if result["status"] == "completed" else 1
    except ValueError as error:
        print(f"练习未完成：{error}")
        return 1
    except APIError as error:
        print(api_error_message(error))
        return 1
    except OSError:
        print("练习未完成：文件读写失败，请检查案例目录与 outputs 路径。")
        return 1
    except KeyboardInterrupt:
        print("练习已停止，工具连接与临时案例已清理。")
        return 130
    except Exception:
        print("练习未完成：图或工具连接异常，请检查本机依赖与 MCP 服务。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
