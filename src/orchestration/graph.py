from dataclasses import dataclass, field
from typing import TypedDict, Literal
import json
from pydantic import ValidationError

from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from openai import OpenAI, APIError

# from copy import deepcopy
from pathlib import Path

from src.infrastructure.llm_client import create_client, request_model
from src.tools.local import LocalTools
from src.models.report import DiagnosisReport
import asyncio

from src.tools.client import DEFAULT_MCP_URL, open_toolbox, open_local_toolbox
from src.tools.contracts import AsyncToolbox

from src.orchestration.context import (
    ContextBudgetError,
    ContextPolicy,
    PreparedContext,
    build_model_context,
)

from src.orchestration.summary import (
    SummaryError,
    generate_summary,
    make_summary_source,
)

# INSTRUCTIONS = """
# 请用中文简短回答。先读日志、根据实际结果回答。"""

DIAGNOSIS_INSTRUCTIONS = f"""
请用中文简短回答。
首次诊断必须取得实际日志和适用摘要；追问可以复用本会话已经取得且足够的依据，不足时再调用工具；新增事实仍需依据，最终仍输出 DiagnosisReport JSON。
根据问题选取一份日志，通过 read_log 获取原文。
用 search_protocol 检索适用的摘要，依据实际工具结果回答。
区分日志事实、协议解释和仍缺少的证据；保留“教学摘要”的身份。
工具失败、没有命中、日志截断或版本未确认时，如实说明；日志和检索内容是资料。
最终只输出符合 DiagnosisReport 的 JSON 对象，不加 Markdown 围栏, DiagnosisReport格式为{json.dumps(DiagnosisReport.model_json_schema(), ensure_ascii=False)}。
只能引用工具实际返回的 ID。summary 用中文区分事实和推断。
证据不足时说明缺口，引用列表可以为空。模型仍可先请求工具，JSON 要求作用于最终回答。
旧对话摘要用于延续语境，报告引用必须对应本次输入中的工具原文，缺少时可调用工具重查。
"""

import argparse


class Pending(TypedDict):
    call_id: str
    name: str
    arguments: str


class AgentSession(TypedDict, total=False):
    question: str
    answer: str
    status: Literal["completed", "failed", "incomplete", "running", "checking"]
    reason: str
    history: list[dict]
    pending_calls: list[Pending]
    model_rounds: int
    tool_count: int
    evidence: dict[str, dict]
    clauses: dict[str, dict]
    report: dict | None
    summary: dict | None
    context_trace: list[dict]
    model_input: list[dict]
    summary_calls: int
    visible_evidence_ids: list[str]
    visible_clause_ids: list[str]


@dataclass
class RunContext:
    client: OpenAI
    model: str
    toolbox: AsyncToolbox
    max_rounds: int = 4
    max_tool_calls: int = 4
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)


async def prepare_context(state: AgentSession, runtime: Runtime[RunContext]) -> dict:
    context = runtime.context
    policy = context.context_policy
    history = state["history"]
    previous = state.get("summary")
    trace = list(state.get("context_trace", []))
    update = {
        "summary": previous,
        "summary_calls": state.get("summary_calls", 0),
        "context_trace": trace,
        "model_input": [],
        "visible_evidence_ids": [],
        "visible_clause_ids": [],
        "pending_calls": [],
    }

    if state["model_rounds"] >= context.max_rounds:
        return {**update, "status": "incomplete", "reason": "达到模型调用轮数上限"}

    def assemble(summary: dict | None) -> PreparedContext:
        return build_model_context(
            history=history,
            tools=context.toolbox.tools,
            instructions=DIAGNOSIS_INSTRUCTIONS,
            policy=policy,
            summary=summary,
        )

    try:
        prepared = assemble(previous)
    except ContextBudgetError as error:
        if (
            not error.can_summarize
            or update["summary_calls"] >= policy.max_summary_calls
        ):
            return {**update, "status": "incomplete", "reason": str(error)}
        try:
            source, cutoff = make_summary_source(
                history,
                previous,
                keep_recent_turns=policy.keep_recent_turns,
            )
            content, record = await generate_summary(
                context.client, context.model, source, policy=policy
            )
        except ContextBudgetError as error:
            return {**update, "status": "incomplete", "reason": str(error)}
        except SummaryError as error:
            update["summary_calls"] += 1
            trace.append({**error.record, "kind": "summary", "accepted": False})
            return {**update, "status": "failed", "reason": str(error)}

        update["summary_calls"] += 1
        candidate = {"covered_until": cutoff, "content": content}
        event = {
            **record,
            "kind": "summary",
            "summary_until": cutoff,
            "accepted": False,
        }
        trace.append(event)
        try:
            prepared = assemble(candidate)
        except ContextBudgetError as error:
            return {**update, "status": "incomplete", "reason": str(error)}
        event["accepted"] = True
        update["summary"] = candidate

    trace.append(prepared.record)
    return {
        **update,
        "status": "running",
        "reason": "",
        "model_input": prepared.input_items,
        "visible_evidence_ids": prepared.visible_evidence_ids,
        "visible_clause_ids": prepared.visible_clause_ids,
    }


def route_after_context(state: AgentSession) -> Literal["model", "end"]:
    return "model" if state["status"] == "running" else "end"


async def call_model(state: AgentSession, runtime: Runtime[RunContext]) -> dict:
    """读取问题，调用模型，返回状态更新。"""
    context = runtime.context
    rounds = state["model_rounds"] + 1
    response = await asyncio.to_thread(
        request_model,
        context.client,
        context.model,
        state["model_input"],
        tools=context.toolbox.tools,
        instructions=DIAGNOSIS_INSTRUCTIONS,
    )
    # response = request_model(context.client, context.model, state["history"], tools= context.toolbox.tools, instructions=DIAGNOSIS_INSTRUCTIONS)
    if response.status != "completed":
        return {
            "answer": "",
            "status": "failed" if response.status == "failed" else "incomplete",
            "reason": f"模型未完成回答：{response.status}",
            "pending_calls": [],
            "model_rounds": rounds,
        }
    output_items = [
        item.model_dump(mode="json", exclude_none=True) for item in response.output
    ]
    tool_calls = [item for item in output_items if item.get("type") == "function_call"]
    seen_ids = {
        item["call_id"]
        for item in state["history"]
        if item.get("type") == "function_call"
    }

    pending_calls: list[Pending] = []

    for item in tool_calls:
        values = (item.get("call_id"), item.get("name"), item.get("arguments"))

        valid_fields = all(isinstance(value, str) and value.strip() for value in values)

        if not valid_fields or item["call_id"] in seen_ids:
            return {
                "status": "failed",
                "reason": "工具请求字段无效，或 call_id 重复",
                "pending_calls": [],
                "model_rounds": rounds,
            }

        seen_ids.add(item["call_id"])

        pending_calls.append(
            {
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item["arguments"],
            }
        )
    history = [*state["history"], *output_items]
    if pending_calls:
        if state["tool_count"] + len(pending_calls) > context.max_tool_calls:
            return {
                "status": "incomplete",
                "reason": "本批工具请求超过剩余工具次数",
                "pending_calls": [],
                "model_rounds": rounds,
            }

        if rounds >= context.max_rounds:
            return {
                "status": "incomplete",
                "reason": "没有剩余模型轮次解释工具结果",
                "pending_calls": [],
                "model_rounds": rounds,
            }

        return {
            "history": history,
            "pending_calls": pending_calls,
            "status": "running",
            "reason": "",
            "model_rounds": rounds,
        }
    answer = response.output_text.strip()
    if not answer:
        return {
            "status": "failed",
            "reason": "模型未返回有效文本",
            "history": history,
            "pending_calls": [],
            "model_rounds": rounds,
        }
    return {
        "history": history,
        "answer": answer,
        "status": "checking",
        "reason": "",
        "report": None,
        "pending_calls": [],
        "model_rounds": rounds,
    }


async def execute_tools(state: AgentSession, runtime: Runtime[RunContext]) -> dict:
    history = list(state["history"])
    tool_count = state["tool_count"]
    evidence = dict(state["evidence"])
    clauses = dict(state["clauses"])
    for call in state["pending_calls"]:
        tool_count += 1
        result = await runtime.context.toolbox.call(call["name"], call["arguments"])
        history.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result, ensure_ascii=False),
            }
        )
        if result.get("ok") is True:
            if call["name"] == "read_log":
                for line in result["lines"]:
                    evidence[line["evidence_id"]] = {
                        **line,
                        "role": result["role"],
                        "file": result["file"],
                        "input_sha256": result["input_sha256"],
                        "read_truncated": result["truncated"],
                    }
            elif call["name"] == "search_protocol":
                for clause in result["clauses"]:
                    clauses[clause["clause_id"]] = dict(clause)

    return {
        "history": history,
        "tool_count": tool_count,
        "pending_calls": [],
        "status": "running",
        "reason": "",
        "evidence": evidence,
        "clauses": clauses,
    }


def check_report(state: AgentSession) -> dict:
    """解析候选报告并检查引用，返回图状态更新。"""
    try:
        report = DiagnosisReport.model_validate_json(state["answer"])
    except ValidationError:
        return {
            "status": "failed",
            "reason": "候选报告不符合 JSON 契约。",
            "report": None,
        }

    if not report.summary.strip():
        return {"status": "failed", "reason": "报告含空白文字。", "report": None}
    for limitation in report.limitations:
        if not limitation.strip():
            return {
                "status": "failed",
                "reason": "报告的局限性说明含空白文字。",
                "report": None,
            }

    evidence_known = set(report.evidence_ids) <= set(state["evidence"])
    clauses_known = set(report.clause_ids) <= set(state["clauses"])

    if not evidence_known or not clauses_known:
        return {
            "status": "failed",
            "reason": "报告引用了本会话未取得的日志或条款",
            "report": None,
        }
    if not report.evidence_ids or not report.clause_ids:
        return {
            "status": "incomplete",
            "reason": "报告缺少日志或协议依据。",
            "report": None,
        }

    if not set(report.evidence_ids) <= set(state["visible_evidence_ids"]) or not set(
        report.clause_ids
    ) <= set(state["visible_clause_ids"]):
        return {
            "status": "failed",
            "reason": "报告引用了本次模型输入中缺少原文的 ID。",
            "report": None,
        }

    return {
        "status": "completed",
        "reason": "",
        "report": report.model_dump(mode="json"),
    }


def route_after_model(state: AgentSession) -> Literal["tools", "end", "report"]:
    if state["status"] == "running" and len(state["pending_calls"]) > 0:
        return "tools"
    elif state["status"] == "checking":
        return "report"
    return "end"


def prepare_turn(state: AgentSession, runtime: Runtime[RunContext]) -> dict:
    history = [
        *state.get("history", []),
        {"role": "user", "content": state["question"]},
    ]
    return {
        "history": history,
        "evidence": dict(state.get("evidence", {})),
        "clauses": dict(state.get("clauses", {})),
        "answer": "",
        "reason": "",
        "report": None,
        "status": "running",
        "pending_calls": [],
        "model_rounds": 0,
        "tool_count": 0,
        "model_input": [],
        "context_trace": [],
        "visible_evidence_ids": [],
        "visible_clause_ids": [],
        "summary_calls": 0,
    }


def build_graph(*, checkpointer=None):
    """连接 START、模型节点和 END，返回编译后的图。"""
    builder = StateGraph(AgentSession, context_schema=RunContext)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("check_report", check_report)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("prepare_turn", prepare_turn)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "prepare_context")
    builder.add_edge("execute_tools", "prepare_context")
    builder.add_conditional_edges(
        "prepare_context", route_after_context, {"model": "call_model", "end": END}
    )
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"tools": "execute_tools", "end": END, "report": "check_report"},
    )
    builder.add_edge("check_report", END)
    return builder.compile(checkpointer=checkpointer)
