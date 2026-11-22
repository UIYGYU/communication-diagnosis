import asyncio
import json
from uuid import uuid4
import sqlite3

from langgraph.checkpoint.memory import InMemorySaver
from openai import APIError

from src.application.cases import make_case_id
from src.application.session import DiagnosisSession
from src.infrastructure.llm_client import create_client
from src.infrastructure.storage import save_result, save_history_segment
from src.models.request import CaseConfig, TurnRequest
from src.orchestration.graph import RunContext, build_graph
from src.orchestration.context import ContextPolicy
from src.tools.client import open_local_toolbox, open_toolbox
from src.tools.local import LocalTools
from pathlib import Path
from src.memory.store import MemoryScope, find_memories


def create_session(config: CaseConfig) -> DiagnosisSession:
    root = config.case_dir.resolve()
    manifest = LocalTools(root).case_manifest()
    if all(value is None for value in manifest["files"].values()):
        raise ValueError("不能没有文件")
    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    return DiagnosisSession(
        thread_id=uuid4().hex,
        case_id=make_case_id(manifest),
        case_dir=root,
        manifest=manifest,
        transport=config.transport,
        mcp_url=config.mcp_url,
        saver=saver,
        graph=graph,
        memory_scope=MemoryScope(
            user_id="local-user",
            project_id="mqtt-lab",
            protocol="MQTT",
            protocol_version=config.mqtt_version,
        ),
        memory_db_path=Path("outputs/memory/memories.sqlite3").resolve(),
    )


def collect_calls(history: list[dict], start: int) -> list[dict]:
    calls: dict[str, dict] = {}
    result = []
    for output in history[start:]:
        if output.get("type") == "function_call":
            calls[output.get("call_id")] = {
                "name": output.get("name"),
                "arguments": output.get("arguments"),
                "call_id": output.get("call_id"),
            }
    for output in history[start:]:
        if output.get("type") == "function_call_output":
            call = calls.pop(output.get("call_id"), None)
            if call is not None:
                tool_result = json.loads(output["output"])
                result.append(
                    {
                        **call,
                        "result": tool_result,
                        "ok": tool_result["ok"],
                        "error": tool_result.get("error"),
                    }
                )
    for call in calls.values():
        result.append({**call, "result": None, "ok": None, "error": "结果未记录"})
    return result


def public_result(
    session: DiagnosisSession,
    request: TurnRequest,
    state: dict,
    *,
    run_id: str,
    history_start: int,
) -> dict:
    return {
        "run_id": run_id,
        "thread_id": session.thread_id,
        "case_id": session.case_id,
        "question": request.question,
        "status": state["status"],
        "reason": state["reason"],
        "report": state["report"],
        "evidence": dict(state["evidence"]),
        "clauses": dict(state["clauses"]),
        "model_rounds": state["model_rounds"],
        "tool_count": state["tool_count"],
        "call_trace": collect_calls(state["history"], history_start),
        "summary": state.get("summary"),
        "context_trace": state.get("context_trace", []),
        "summary_calls": state.get("summary_calls", 0),
        "visible_evidence_ids": state.get("visible_evidence_ids", []),
        "visible_clause_ids": state.get("visible_clause_ids", []),
    }


async def analyze_async(session: DiagnosisSession, request: TurnRequest) -> dict:
    if session.busy:
        raise ValueError("上一问题仍在运行")
    if not session.usable:
        raise ValueError("请新建会话")

    try:
        current_manifest = LocalTools(session.case_dir).case_manifest()
    except (ValueError, OSError) as error:
        session.usable = False
        raise ValueError("案例读取失败，请新建会话") from error

    if session.manifest != current_manifest:
        session.usable = False
        raise ValueError("案例内容已改变，请新建会话")
    run_id = uuid4().hex
    config = {
        "configurable": {"thread_id": session.thread_id},
        "recursion_limit": 3 * request.max_rounds + 6,
    }
    previous = session.graph.get_state(config).values
    history_start = len(previous.get("history", []))
    session.busy = True
    memory_candidates = []
    memory_warning = None
    try:
        try:
            if session.memory_db_path.is_file():
                memory_candidates = await asyncio.to_thread(
                    find_memories,
                    session.memory_db_path,
                    session.memory_scope,
                    request.question,
                )
        except (OSError, sqlite3.Error):
            memory_warning = "长期记忆读取失败，本问使用已有对话和工具资料。"
        client, model = create_client()
        with client:
            if not session.model_id:
                session.model_id = model
            if session.model_id != model:
                raise ValueError("模型已改变，请新建会话")
            if session.transport == "local":
                toolbox_context = open_local_toolbox(session.case_dir)
            else:
                toolbox_context = open_toolbox(
                    case_dir=session.case_dir,
                    transport=session.transport,
                    mcp_url=session.mcp_url,
                )
            context_policy = ContextPolicy(max_input_bytes=request.max_context_bytes)
            async with toolbox_context as toolbox:
                state = await session.graph.ainvoke(
                    {"question": request.question},
                    context=RunContext(
                        client=client,
                        model=model,
                        toolbox=toolbox,
                        max_rounds=request.max_rounds,
                        max_tool_calls=request.max_tool_calls,
                        context_policy=context_policy,
                        memory_candidates=memory_candidates
                    ),
                    config=config,
                )
                latest_state = state
        result = public_result(
            session, request, state, run_id=run_id, history_start=history_start
        )
    except (APIError, ValueError, RuntimeError) as error:
        session.usable = False
        if isinstance(error, APIError):
            reason = f"模型请求失败：{type(error).__name__}"
            status_code = getattr(error, "status_code", None)
            if status_code is not None:
                reason += f"，HTTP 状态码：{status_code}"
        else:
            reason = str(error)

        reason += (
            "；轨迹来自最新检查点；失败节点内尚未提交的操作可能未记录，计数可能不完整。"
        )

        snapshot = session.graph.get_state(config).values
        latest_state = snapshot

        if len(snapshot.get("history", [])) > history_start:
            failed_state = dict(snapshot)
            failed_state.update(
                {
                    "status": "failed",
                    "reason": reason,
                    "report": None,
                }
            )

            result = public_result(
                session,
                request,
                failed_state,
                run_id=run_id,
                history_start=history_start,
            )
        else:
            result = {
                "run_id": run_id,
                "thread_id": session.thread_id,
                "case_id": session.case_id,
                "question": request.question,
                "status": "failed",
                "reason": reason,
                "report": None,
                "evidence": {},
                "clauses": {},
                "model_rounds": None,
                "tool_count": None,
                "call_trace": [],
                "summary": None,
                "summary_calls": 0,
                "context_trace": [],
                "visible_evidence_ids": [],
                "visible_clause_ids": [],
            }
    except Exception:
        session.usable = False
        raise
    finally:
        session.busy = False

    if result["status"] != "completed":
        session.usable = False

    result["history_archive"] = None
    new_history = latest_state.get("history", [])[history_start:]
    if new_history:
        archive = {
            "status": "failed",
            "path": None,
            "history_range": [history_start, history_start + len(new_history)],
        }
        result["history_archive"] = archive
        try:
            path = save_history_segment(
                thread_id=session.thread_id,
                case_id=session.case_id,
                run_id=run_id,
                history_start=history_start,
                history=new_history,
                summary=latest_state.get("summary"),
            )
        except OSError:
            result["storage_warning"] = "本问新增历史归档失败"
        else:
            archive.update(status="saved", path=path.as_posix())
    answered_rounds = result["model_rounds"] or 0

    result["memory_inputs"] = [
        {"model_round": row["model_round"], "items": row["memory_selected"]}
        for row in result["context_trace"]
        if row.get("kind") == "context"
        and 0 < row.get("model_round", 0) <= answered_rounds
        and row.get("memory_selected")
    ]
    result["memory_warning"] = memory_warning
    try:
        save_result(result)
    except OSError:
        warning = "结果文件保存失败，仍可从界面下载"
        if result.get("storage_warning"):
            warning = f"{result['storage_warning']}; {warning}"
        result["storage_warning"] = warning
    session.turns.append(result)
    return result


def analyze(session: DiagnosisSession, request: TurnRequest) -> dict:
    return asyncio.run(analyze_async(session, request))
