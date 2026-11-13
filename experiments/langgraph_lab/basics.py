"""两个普通节点的图 API 热身；不调用模型，不生成诊断报告。"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PracticeState(TypedDict, total=False):
    question: str
    normalized: str
    note: str


def normalize_question(state: PracticeState) -> dict:
    question = state["question"].strip()
    if not question:
        raise ValueError("问题不能为空。")
    return {"normalized": question}


def describe_input(state: PracticeState) -> dict:
    return {"note": f"待分析的问题共有 {len(state['normalized'])} 个字符。"}


def build_basics_graph():
    builder = StateGraph(PracticeState)
    builder.add_node("normalize_question", normalize_question)
    builder.add_node("describe_input", describe_input)
    builder.add_edge(START, "normalize_question")
    builder.add_edge("normalize_question", "describe_input")
    builder.add_edge("describe_input", END)
    return builder.compile()


def main() -> None:
    graph = build_basics_graph()
    result = graph.invoke({"question": "  发布端的日志说明了什么？  "})
    print(result)
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
