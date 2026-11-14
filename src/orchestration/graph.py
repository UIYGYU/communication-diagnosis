from dataclasses import dataclass
from typing import TypedDict, Literal
import json

from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from openai import OpenAI, APIError
from copy import deepcopy
from pathlib import Path

from src.infrastructure.llm_client import INSTRUCTIONS,create_client, request_model
from src.tools.local import LocalTools

import argparse

class Pending(TypedDict):
    call_id:str
    name:str
    arguments:str

class AgentSession(TypedDict, total=False):
    question: str
    answer: str
    status: Literal["completed", "failed", "incomplete", "running"]
    reason: str
    history: list[dict]
    pending_calls:list[Pending]
    model_rounds:int
    tool_count:int
    successful_read:bool


@dataclass
class RunContext:
    client: OpenAI
    model: str
    toolbox: LocalTools
    max_rounds: int = 4
    max_tool_calls:int = 4
    max_context_bytes: int = 128_000


def call_model(state: AgentSession, runtime: Runtime[RunContext]) -> dict:
    """读取问题，调用模型，返回状态更新。"""
    context = runtime.context
    if state["model_rounds"] >= context.max_rounds:
        return {
        "status": "incomplete",
        "reason": "达到模型调用轮数上限",
        "pending_calls": [],
    }
    payload = {
        "history": state["history"],
        "tools": context.toolbox.tools,
        "instructions": INSTRUCTIONS,
    }
    json_str = json.dumps(payload, ensure_ascii=False)
    size_bytes = len(json_str.encode("utf-8"))
    if size_bytes > context.max_context_bytes:
        return {
        "status": "incomplete",
        "reason": f"请求输入超过字节预算：{size_bytes}",
        "pending_calls": [],
    }
    rounds = state["model_rounds"] + 1
    response = request_model(context.client, context.model, state["history"], tools= context.toolbox.tools, instructions=INSTRUCTIONS)
    if response.status != "completed":
        return {
            "answer": "",
            "status": "failed" if response.status == "failed" else "incomplete",
            "reason": f"模型未完成回答：{response.status}",
            "pending_calls":[],
            "model_rounds":rounds
        }
    output_items = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
    tool_calls = [item for item in output_items if item.get("type") == "function_call"]
    seen_ids = {item["call_id"] for item in state["history"] if item.get("type") == "function_call"}

    pending_calls: list[Pending] = []

    for item in tool_calls:
        values=(
            item.get("call_id"),
            item.get("name"),
            item.get("arguments")
        )

        valid_fields = all(
            isinstance(value, str) and value.strip()
            for value in values
        )

        if not valid_fields or item["call_id"] in seen_ids:
            return {
            "status": "failed",
            "reason": "工具请求字段无效，或 call_id 重复",
            "pending_calls": [],
            "model_rounds": rounds,
        }

        seen_ids.add(item["call_id"])

        pending_calls.append({
            "call_id": item["call_id"],
            "name": item["name"],
            "arguments": item["arguments"],
        })
    history = [*state["history"], *output_items]
    if pending_calls:
        if state["tool_count"] + len(pending_calls) > context.max_tool_calls:
            return{
                "status":"incomplete",
                "reason":"本批工具请求超过剩余工具次数",
                "pending_calls":[],
                "model_rounds":rounds
            }

        if rounds >= context.max_rounds:
            return {
                "status": "incomplete",
                "reason": "没有剩余模型轮次解释工具结果",
                "pending_calls": [],
                "model_rounds": rounds,
            }

        return{
            "history":history,
            "pending_calls":pending_calls,
            "status":"running",
            "reason":"",
            "model_rounds":rounds
        }
    answer = response.output_text.strip()
    if not answer:
        return {
                "status": "failed",
                "reason": "模型未返回有效文本",
                "history":history,
                "pending_calls": [],
                "model_rounds":rounds
            }
    if not state["successful_read"]:
        return{
            "status": "incomplete",
            "reason": "工具闭环还没取到日志",
            "history": history,
            "pending_calls": [],
            "model_rounds":rounds
        }
    return {
        "history":history,
        "answer": response.output_text,
        "status": "completed",
        "reason": "",
        "pending_calls":[],
        "model_rounds":rounds
    }

def execute_tools(state: AgentSession, runtime:Runtime[RunContext]) -> dict:
    history = deepcopy(state["history"])
    tool_count = state["tool_count"]
    successful_read = state["successful_read"]
    for call in state["pending_calls"]:
        tool_count+=1
        result = runtime.context.toolbox.call(call["name"],call["arguments"])
        history.append(
            {
                "type":"function_call_output",
                "call_id":call["call_id"],
                "output":json.dumps(result, ensure_ascii=False)
            }
        )
        if call["name"] == "read_log" and result.get("ok") is True:
            successful_read = True

    return{
        "history":history,
        "tool_count":tool_count,
        "successful_read":successful_read,
        "pending_calls":[],
        "status":"running",
        "reason":""
    }

def route_after_model(state:AgentSession) -> Literal["tools", "end"]:
    if state["status"]=="running" and len(state["pending_calls"]) > 0:
        return "tools"
    return "end"



def build_graph():
    """连接 START、模型节点和 END，返回编译后的图。"""
    builder = StateGraph(AgentSession, context_schema=RunContext)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model, {"tools":"execute_tools", "end":END})
    builder.add_edge("execute_tools", "call_model")
    return builder.compile()


def run_once(question: str, case_dir: Path) -> AgentSession:
    """校验输入，管理客户端，执行一次图。"""
    question = question.strip()
    if not question:
        raise ValueError("缺少问题")
    toolbox = LocalTools(case_dir)
    client, model = create_client()
    initial_state:AgentSession={
        "question":question,
        "answer":"",
        "history":[{"role":"user", "content":question}],
        "successful_read":False,
        "model_rounds":0,
        "pending_calls":[],
        "reason":"",
        "tool_count":0,
        "status":"running"
    }
    with client:
        graph = build_graph()
        # print(graph.get_graph().draw_mermaid())
        result = graph.invoke(
            initial_state,
            context=RunContext(client, model, toolbox=toolbox),
        )
    return result


def main() -> int:
    """终端入口：完整回答返回 0，失败返回 1。"""
    parser = argparse.ArgumentParser(description="第三步练习")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    args = parser.parse_args()
    try:
        question = input("请输入问题：")
        agentsession = run_once(question, args.case_dir)
    except APIError as error:
        print(f"模型请求失败：{type(error).__name__}")
        return 1
    except ValueError as error:
        print(f"输入或配置错误：{error}")
        return 1
    if agentsession["status"] == "completed":
        print(f"报告:{agentsession['answer']}")
        return 0

    print(f"执行失败： {agentsession['reason']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
