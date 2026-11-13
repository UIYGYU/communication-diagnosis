"""同案例短期记忆练习：一个模型客户端、一个 MCP 连接，最多三个问题。"""

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from openai import APIError, OpenAI

from examples.stage3.llm_client import api_error_message, create_client
from examples.stage3.tools import FILES, CaseTools
from examples.stage4.agent_demo import AgentSession, run_agent_async
from examples.stage4.client import open_toolbox
from examples.stage4.service import (
    DiagnosisRequest,
    UploadedLog,
    input_manifest,
    validate_log_contents,
)

DEFAULT_QUESTION = (
    "请查看发布者和订阅者日志，结合协议摘要说明能确认的收发情况与证据缺口。"
)


async def read_from_terminal(prompt: str) -> str:
    # 等待键盘时不占住 MCP 连接使用的事件循环。
    return await asyncio.to_thread(input, prompt)


def show_turn(turn: dict) -> None:
    result = turn["result"]
    print(f"第 {turn['turn']} 问：{result['status']}；{result['reason']}")
    for item in result["trace"]:
        if item["type"] == "model_request":
            print(
                f"  请求前：{item['history_items']} 项历史，"
                f"{item['context_bytes']} 字节，上限 {item['limit']} 字节"
            )
        elif item["type"] == "model_response":
            usage = item.get("usage")
            print(
                "  模型返回的 usage："
                + (json.dumps(usage, ensure_ascii=False) if usage else "接口未提供")
            )
    if result["report"]:
        print(result["report"]["summary"])


async def run_context_session(
    case_dir: Path,
    question: str,
    *,
    max_questions: int = 3,
    max_context_bytes: int = 128_000,
    max_rounds: int = 6,
    max_tool_calls: int = 8,
    client_factory: Callable[[], tuple[OpenAI, str]] | None = None,
    toolbox_factory: Callable[[Path], AbstractAsyncContextManager[Any]] | None = None,
    read_question: Callable[[str], Awaitable[str]] | None = None,
    observer: Callable[[dict], None] | None = None,
) -> dict:
    """工厂和输入函数只供离线验证；普通 CLI 使用真实连接与终端输入。"""
    if type(max_questions) is not int or not 1 <= max_questions <= 3:
        raise ValueError("这个练习最多连续处理 1～3 个问题。")
    if type(max_context_bytes) is not int or max_context_bytes < 1:
        raise ValueError("上下文字节上限必须是正整数。")

    # CaseTools 只读取受控文件名，检查目录、链接和合计大小；这里不执行日志命令。
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
    session = AgentSession()
    result = {
        "mode": "context_session",
        "session_id": uuid4().hex,
        "status": "completed",
        "reason": "短期会话已结束；再次运行会建立新的案例与历史。",
        "input_manifest": input_manifest(request),
        "turns": [],
        "limits": {
            "max_questions": max_questions,
            "max_context_bytes": max_context_bytes,
            "max_rounds_per_question": max_rounds,
            "max_tool_calls_per_question": max_tool_calls,
        },
    }
    client, model = (client_factory or create_client)()
    try:
        with client:
            with TemporaryDirectory(prefix="communication-memory-") as temporary:
                root = Path(temporary)
                for role, raw in snapshots.items():
                    (root / FILES[role]).write_bytes(raw)
                # 连接跨越全部问题；退出后临时案例、连接、子进程一起清理。
                async with (toolbox_factory or open_toolbox)(root) as toolbox:
                    for number in range(1, max_questions + 1):
                        answer = await run_agent_async(
                            client,
                            model,
                            question,
                            toolbox,
                            max_rounds=max_rounds,
                            max_tool_calls=max_tool_calls,
                            session=session,
                            max_context_bytes=max_context_bytes,
                        )
                        turn = {"turn": number, "question": question, "result": answer}
                        result["turns"].append(turn)
                        if observer is not None:
                            observer(turn)
                        if answer["status"] != "completed":
                            result.update(
                                status=answer["status"], reason=answer["reason"]
                            )
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
    finally:
        # 即使调用方仍保留对象，也不能在连接关闭后继续这个会话。
        session.closed = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="同案例上下文与短期记忆练习")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--max-context-bytes", type=int, default=128_000)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/stage4/context-session.json")
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
            run_context_session(
                args.case_dir,
                args.question,
                max_questions=args.max_questions,
                max_context_bytes=args.max_context_bytes,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                observer=show_turn,
            )
        )
        # 只保存报告、工具轨迹和计量摘要；不导出完整 session.history。
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
        print("练习未完成：工具连接或执行异常，请检查本机 MCP 服务及依赖。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
