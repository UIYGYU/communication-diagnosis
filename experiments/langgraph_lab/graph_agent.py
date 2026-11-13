"""LangGraph 编排对照：会话数据进 State，活动连接放 Runtime.context。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from examples.stage3.llm_client import api_error_message
from examples.stage4.agent_demo import (
    INSTRUCTIONS,
    LearningReport,
    context_size_bytes,
)


class Toolbox(Protocol):
    tools: list[dict]
    transport: str

    async def call(self, name: str, arguments: str) -> dict: ...


class LabState(TypedDict, total=False):
    question: str
    case_id: str
    model: str
    history: list[dict]
    evidence: dict[str, dict]
    clauses: dict[str, dict]
    successful_tools: list[str]
    pending_calls: list[dict]
    final_text: str
    round_number: int
    tool_count: int
    status: str
    reason: str
    report: dict | None
    trace: list[dict]
    limits: dict[str, int]
    transport: str
    mode: str


@dataclass
class RunContext:
    """只在本次运行使用；模型和 MCP 客户端不会进入检查点。"""

    client: OpenAI
    model: str
    toolbox: Toolbox
    case_id: str
    max_rounds: int = 6
    max_tool_calls: int = 8
    max_context_bytes: int = 128000


# 执行准备开始一次循环，一次任务
def prepare_turn(state: LabState, runtime: Runtime[RunContext]) -> dict:
    """恢复同一 thread 的资料，并为本次问题重新分配执行预算。"""
    context = runtime.context
    if context is None:
        raise ValueError("运行图时必须提供 RunContext。")
    question = state.get("question", "")
    if not isinstance(question, str):
        raise ValueError("问题必须是字符串。")
    question = question.strip()
    if not question or len(question) > 4000:
        raise ValueError("问题须为 1～4000 个非空白字符。")
    if not context.case_id or not context.model:
        raise ValueError("案例身份和模型名不能为空。")
    for value, lower, upper in (
        (context.max_rounds, 1, 12),
        (context.max_tool_calls, 1, 16),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("模型轮数须为 1～12，工具调用上限须为 1～16。")
    if type(context.max_context_bytes) is not int or context.max_context_bytes < 1:
        raise ValueError("上下文载荷字节上限必须是正整数。")
    if state.get("case_id") not in (None, context.case_id):
        raise ValueError("这个 thread 已绑定另一份案例；请使用新的 thread_id。")
    if state.get("model") not in (None, context.model):
        raise ValueError("同一 thread 不更换模型；请使用新的 thread_id。")
    if state.get("status") not in (None, "completed"):
        raise ValueError("这个 thread 上次未完整结束；请建立新的会话。")
    return {
        "question": question,
        "case_id": context.case_id,
        "model": context.model,
        "history": [
            *deepcopy(state.get("history", [])),
            {"role": "user", "content": question},
        ],
        "evidence": deepcopy(state.get("evidence", {})),
        "clauses": deepcopy(state.get("clauses", {})),
        "successful_tools": list(state.get("successful_tools", [])),
        "pending_calls": [],
        "final_text": "",
        "round_number": 0,
        "tool_count": 0,
        "status": "running",
        "reason": "正在执行本次问题。",
        "report": None,
        "trace": [{"type": "node_enter", "node": "prepare_turn"}],
        "limits": {
            "max_rounds": context.max_rounds,
            "max_tool_calls": context.max_tool_calls,
            "max_context_bytes": context.max_context_bytes,
        },
        "transport": context.toolbox.transport,
        "mode": "langgraph_learning_agent",
    }


def stop(status: str, reason: str, trace: list[dict], **updates: Any) -> dict:
    return {
        "status": status,
        "reason": reason,
        "report": None,
        "trace": trace,
        **updates,
    }


# 单次调用model
async def call_model(state: LabState, runtime: Runtime[RunContext]) -> dict:
    """仍然调用原来的 Responses SDK；LangGraph 负责节点之间的调度。"""
    context = runtime.context
    number = state["round_number"] + 1
    trace = [*state["trace"], {"type": "node_enter", "node": "call_model"}]
    size = context_size_bytes(state["history"], context.toolbox.tools)
    trace.append(
        {
            "type": "model_request",
            "round": number,
            "history_items": len(state["history"]),
            "context_bytes": size,
            "limit": context.max_context_bytes,
        }
    )
    if number > context.max_rounds or size > context.max_context_bytes:
        return stop(
            "incomplete", "达到模型轮数或上下文载荷字节上限；未发送本次请求。", trace
        )
    try:
        response = await asyncio.to_thread(
            context.client.responses.create,
            model=context.model,
            instructions=INSTRUCTIONS,
            input=state["history"],
            tools=context.toolbox.tools,
            tool_choice="auto",
            store=False,
            max_output_tokens=4096,
        )
    except APIError as error:
        reason = api_error_message(error)
        trace.append({"type": "api_error", "round": number, "message": reason})
        return stop("failed", reason, trace, round_number=number)
    trace.append(
        {
            "type": "model_response",
            "round": number,
            "response_id": response.id,
            "status": response.status,
            "usage": response.usage.model_dump() if response.usage else None,
        }
    )
    if response.status != "completed" or response.output is None:
        return stop(
            "incomplete",
            "模型响应未完成，未执行其中的工具请求。",
            trace,
            round_number=number,
        )

    # 只将 SDK 对象转成 JSON 数据，不筛掉 reasoning、phase 或其他已返回字段。
    output = [
        item.model_dump(mode="json", exclude_none=True)
        if isinstance(item, BaseModel)
        else deepcopy(item)
        for item in response.output
    ]
    if any(not isinstance(item, dict) for item in output):
        return stop(
            "failed", "模型输出条目不符合 Responses 格式。", trace, round_number=number
        )

    history = [*state["history"], *output]
    calls = [item for item in output if item.get("type") == "function_call"]
    update = {"history": history, "pending_calls": calls, "round_number": number}
    if calls:
        identities = [item.get("call_id") for item in calls]
        if any(not isinstance(value, str) or not value for value in identities) or len(
            set(identities)
        ) != len(identities):
            return stop("failed", "模型返回缺失或重复的工具调用标识。", trace, **update)
        if any(
            not isinstance(item.get("name"), str)
            or not isinstance(item.get("arguments"), str)
            for item in calls
        ):
            return stop(
                "failed",
                "工具名称或 arguments 不符合 Responses 格式。",
                trace,
                **update,
            )
        if (
            number == context.max_rounds
            or state["tool_count"] + len(calls) > context.max_tool_calls
        ):
            trace.append(
                {"type": "budget_stop", "round": number, "pending_calls": len(calls)}
            )
            return stop("incomplete", "达到调用上限；本轮工具未执行。", trace, **update)
    return {**update, "final_text": response.output_text.strip(), "trace": trace}


def register_result(name: str, result: dict, evidence: dict, clauses: dict) -> dict:
    """先验证整份结果，再登记到图状态；新连接不能覆盖之前的引用。"""
    if type(result.get("ok")) is not bool:
        return {"ok": False, "error": "工具结果缺少布尔类型 ok。"}
    if not result["ok"]:
        return result
    if name == "read_log":
        field, id_field, registry = "records", "evidence_id", evidence
    elif name == "search_protocol":
        field, id_field, registry = "clauses", "clause_id", clauses
    else:
        return {"ok": False, "error": "未知工具结果。"}
    rows = result.get(field)
    if not isinstance(rows, list):
        return {"ok": False, "error": "工具没有返回约定的引用列表。"}
    pending = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get(id_field), str)
            or not row[id_field]
        ):
            return {"ok": False, "error": "工具返回了缺少标识的引用。"}
        identity = row[id_field]
        if (identity in registry and registry[identity] != row) or (
            identity in pending and pending[identity] != row
        ):
            return {"ok": False, "error": "同一引用标识对应不同内容；本次结果未登记。"}
        pending[identity] = deepcopy(row)
    registry.update(pending)
    return result


# 单执行一次工具
async def execute_tools(state: LabState, runtime: Runtime[RunContext]):
    context = runtime.context
    history = deepcopy(state["history"])
    evidence = deepcopy(state["evidence"])
    clauses = deepcopy(state["clauses"])
    successful = set(state["successful_tools"])
    trace = [*state["trace"], {"type": "node_enter", "node": "execute_tools"}]
    count = state["tool_count"]
    for call in state["pending_calls"]:
        count += 1
        name, arguments = call["name"], call["arguments"]
        try:
            if name not in {"read_log", "search_protocol"}:
                result = {"ok": False, "error": "未知工具；没有发送 MCP 请求。"}
            else:
                result = await context.toolbox.call(name, arguments)
            if not isinstance(result, dict):
                raise TypeError("工具结果不是对象")
            result = register_result(name, result, evidence, clauses)
        except Exception:
            trace.append(
                {"type": "tool_error", "name": name, "call_id": call["call_id"]}
            )
            return stop(
                "failed",
                "工具执行出现异常；本会话停止，未生成完成报告。",
                trace,
                history=history,
                evidence=evidence,
                clauses=clauses,
                successful_tools=sorted(successful),
                tool_count=count,
            )
        if result.get("ok") is True:
            successful.add(name)
        trace.append(
            {
                "type": "tool_result",
                "round": state["round_number"],
                "transport": context.toolbox.transport,
                "call_id": call["call_id"],
                "name": name,
                "arguments": arguments,
                "result": deepcopy(result),
            }
        )
        history.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result, ensure_ascii=False),
            }
        )
    return {
        "history": history,
        "evidence": evidence,
        "clauses": clauses,
        "successful_tools": sorted(successful),
        "pending_calls": [],
        "tool_count": count,
        "trace": trace,
    }


def check_report(state: LabState) -> dict:
    trace = [*state["trace"], {"type": "node_enter", "node": "check_report"}]
    if not state["final_text"]:
        return stop("incomplete", "模型没有返回最终文本。", trace)
    if not {"read_log", "search_protocol"} <= set(state["successful_tools"]):
        return stop("incomplete", "尚未成功取得日志与协议检索结果。", trace)
    try:
        report = LearningReport.model_validate_json(state["final_text"])
    except ValidationError:
        return stop("incomplete", "模型最终输出不符合报告 JSON 结构。", trace)
    if not set(report.evidence_ids) <= state["evidence"].keys():
        return stop("incomplete", "报告引用了本 thread 未取得的日志证据。", trace)
    if not set(report.clause_ids) <= state["clauses"].keys():
        return stop("incomplete", "报告引用了本 thread 未取得的协议摘要。", trace)
    return {
        "status": "completed",
        "reason": "LangGraph 学习闭环与引用存在性检查通过；结论仍需对照原文。",
        "report": report.model_dump(),
        "trace": trace,
    }


def after_model(state: LabState) -> Literal["execute_tools", "check_report", "__end__"]:
    if state["status"] != "running":
        return END
    return "execute_tools" if state["pending_calls"] else "check_report"


def after_tools(state: LabState) -> Literal["call_model", "__end__"]:
    return "call_model" if state["status"] == "running" else END


def build_graph(checkpointer=None):
    builder = StateGraph(LabState, context_schema=RunContext)
    builder.add_node("prepare_turn", prepare_turn)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("check_report", check_report)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "call_model")
    builder.add_conditional_edges("call_model", after_model)
    builder.add_conditional_edges("execute_tools", after_tools)
    builder.add_edge("check_report", END)
    return builder.compile(checkpointer=checkpointer)
