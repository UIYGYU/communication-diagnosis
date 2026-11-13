"""真实 Responses SDK + 内存检查点；不加载 .env，不请求真实模型。"""

import asyncio
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from examples.stage4.agent_demo import LocalTools, context_size_bytes
from examples.stage4.test_integration import (
    final_message,
    function_call,
    model_factory,
    publisher_bytes,
)
from experiments.langgraph_lab.graph_agent import RunContext, build_graph


def initial_calls():
    return [
        {
            "type": "reasoning",
            "id": "rs_first",
            "summary": [],
            "encrypted_content": "OPAQUE-REASONING-ITEM",
        },
        function_call("read_log", {"role": "publisher"}, "log_1"),
        function_call("search_protocol", {"query": "PUBACK"}, "search_1"),
    ]


def config(thread_id="thread-a"):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


@pytest.fixture
def case_dir(tmp_path):
    (tmp_path / "publisher.json").write_bytes(publisher_bytes())
    return tmp_path


def test_graph_replays_complete_history_after_replacing_toolbox(case_dir):
    requests = []
    first_final = final_message()
    first_final["phase"] = "final_answer"
    client, model = model_factory(
        [initial_calls(), [first_final], [final_message()]], requests
    )()
    graph = build_graph(InMemorySaver())

    async def exercise():
        first_tools = LocalTools(case_dir)
        first = await graph.ainvoke(
            {"question": "请诊断这个案例。"},
            config(),
            context=RunContext(client, model, first_tools, "case-a"),
        )
        second_tools = LocalTools(case_dir)
        assert second_tools.evidence == second_tools.clauses == {}
        second = await graph.ainvoke(
            {"question": "为什么刚才的确认不能证明订阅者收到？"},
            config(),
            context=RunContext(client, model, second_tools, "case-a"),
        )
        return first, second, second_tools

    with client:
        first, second, second_tools = asyncio.run(exercise())
    assert first["status"] == second["status"] == "completed"
    assert len(requests) == 3
    follow_up = requests[-1]["input"]
    assert follow_up[-1]["role"] == "user"
    assert "为什么刚才" in follow_up[-1]["content"]
    assert any(
        item.get("type") == "reasoning"
        and item.get("encrypted_content") == "OPAQUE-REASONING-ITEM"
        for item in follow_up
    )
    assert any(item.get("phase") == "final_answer" for item in follow_up)
    outputs = {
        item["call_id"]: json.loads(item["output"])
        for item in follow_up
        if item.get("type") == "function_call_output"
    }
    assert set(outputs) == {"log_1", "search_1"}
    assert outputs["log_1"]["records"][0]["evidence_id"] == "E-publisher-stdout-1"
    assert any(row["clause_id"] == "K-PUBACK" for row in outputs["search_1"]["clauses"])
    # 新工具没有读取过日志；引用校验使用了图状态中的真实证据。
    assert second_tools.evidence == second_tools.clauses == {}
    assert "E-publisher-stdout-1" in second["evidence"]
    assert "K-PUBACK" in second["clauses"]
    assert not any(item["type"] == "tool_result" for item in second["trace"])
    serialized = json.dumps(second)
    assert "TEST-KEY" not in serialized and "PRIVATE-COMMAND" not in serialized


def test_two_threads_keep_histories_and_case_evidence_separate(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "publisher.json").write_bytes(publisher_bytes("lab/alpha"))
    (right / "publisher.json").write_bytes(publisher_bytes("lab/beta"))
    requests = []
    client, model = model_factory(
        [initial_calls(), [final_message()], initial_calls(), [final_message()]],
        requests,
    )()
    graph = build_graph(InMemorySaver())

    async def exercise():
        first = await graph.ainvoke(
            {"question": "检查 alpha。"},
            config("alpha"),
            context=RunContext(client, model, LocalTools(left), "case-alpha"),
        )
        second = await graph.ainvoke(
            {"question": "检查 beta。"},
            config("beta"),
            context=RunContext(client, model, LocalTools(right), "case-beta"),
        )
        return first, second

    with client:
        first, second = asyncio.run(exercise())
    assert first["status"] == second["status"] == "completed"
    assert "lab/alpha" in first["evidence"]["E-publisher-stdout-1"]["raw"]
    assert "lab/beta" in second["evidence"]["E-publisher-stdout-1"]["raw"]
    assert len(requests[2]["input"]) == 1
    assert "lab/alpha" not in json.dumps(requests[2:])


@pytest.mark.parametrize("change", ["case", "model"])
def test_existing_thread_rejects_different_case_or_model(case_dir, change):
    requests = []
    client, model = model_factory([initial_calls(), [final_message()]], requests)()
    graph = build_graph(InMemorySaver())

    async def exercise():
        first = await graph.ainvoke(
            {"question": "请诊断。"},
            config(),
            context=RunContext(client, model, LocalTools(case_dir), "case-a"),
        )
        assert first["status"] == "completed"
        with pytest.raises(ValueError):
            await graph.ainvoke(
                {"question": "请继续。"},
                config(),
                context=RunContext(
                    client,
                    "other-model" if change == "model" else model,
                    LocalTools(case_dir),
                    "case-b" if change == "case" else "case-a",
                ),
            )

    with client:
        asyncio.run(exercise())
    assert len(requests) == 2


@pytest.mark.parametrize("kind", ["bad_reference", "api_error", "pending_calls"])
def test_failed_thread_cannot_replay_unfinished_history(case_dir, kind):
    requests = []
    if kind == "bad_reference":
        replies, rounds = [initial_calls(), [final_message(["E-OTHER-CASE"])]], 6
    elif kind == "api_error":
        replies, rounds = [(401, {"error": {"message": "PRIVATE-UPSTREAM-ERROR"}})], 6
    else:
        replies, rounds = [initial_calls()], 1
    client, model = model_factory(replies, requests)()
    graph = build_graph(InMemorySaver())

    async def exercise():
        context = RunContext(
            client, model, LocalTools(case_dir), "case-a", max_rounds=rounds
        )
        first = await graph.ainvoke({"question": "检查。"}, config(), context=context)
        assert first["status"] != "completed" and first["report"] is None
        assert "PRIVATE-UPSTREAM-ERROR" not in json.dumps(first)
        before = len(requests)
        with pytest.raises(ValueError):
            await graph.ainvoke({"question": "继续。"}, config(), context=context)
        assert len(requests) == before

    with client:
        asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["bad_arguments", "unknown_tool", "no_match"])
def test_invalid_tools_or_empty_retrieval_never_make_a_valid_report(case_dir, kind):
    calls = initial_calls()
    if kind == "bad_arguments":
        calls[1]["arguments"] = '{"role":"publisher","path":"private.txt"}'
    elif kind == "unknown_tool":
        calls[1]["name"] = "delete_log"
    else:
        calls[2]["arguments"] = '{"query":"zzzznotaprotocoltermzzzz"}'
    requests = []
    client, model = model_factory([calls, [final_message()]], requests)()
    graph = build_graph(InMemorySaver())
    with client:
        result = asyncio.run(
            graph.ainvoke(
                {"question": "请检查。"},
                config(),
                context=RunContext(client, model, LocalTools(case_dir), "case-a"),
            )
        )
    assert result["status"] != "completed" and result["report"] is None
    assert (case_dir / "publisher.json").read_bytes() == publisher_bytes()


def test_call_budget_stops_before_executing_any_excess_tools(case_dir):
    requests = []
    client, model = model_factory([initial_calls()], requests)()
    graph = build_graph(InMemorySaver())
    toolbox = LocalTools(case_dir)
    with client:
        result = asyncio.run(
            graph.ainvoke(
                {"question": "检查。"},
                config(),
                context=RunContext(client, model, toolbox, "case-a", max_tool_calls=1),
            )
        )
    assert result["status"] == "incomplete"
    assert len(requests) == 1
    assert toolbox.evidence == toolbox.clauses == {}
    assert result["evidence"] == result["clauses"] == {}


def test_context_budget_stops_before_first_http_request(case_dir):
    requests = []
    client, model = model_factory([], requests)()
    graph = build_graph(InMemorySaver())
    with client:
        result = asyncio.run(
            graph.ainvoke(
                {"question": "检查。"},
                config(),
                context=RunContext(
                    client, model, LocalTools(case_dir), "case-a", max_context_bytes=1
                ),
            )
        )
    assert result["status"] == "incomplete" and result["report"] is None
    assert requests == []


def test_context_budget_checks_tool_outputs_before_next_http_request(case_dir):
    question = "检查。"
    toolbox = LocalTools(case_dir)
    limit = context_size_bytes([{"role": "user", "content": question}], toolbox.tools)
    requests = []
    client, model = model_factory([initial_calls()], requests)()
    graph = build_graph(InMemorySaver())
    with client:
        result = asyncio.run(
            graph.ainvoke(
                {"question": question},
                config(),
                context=RunContext(
                    client, model, toolbox, "case-a", max_context_bytes=limit
                ),
            )
        )
    assert result["status"] == "incomplete"
    assert len(requests) == 1
    assert result["evidence"] and result["clauses"]


def test_previous_result_does_not_gain_later_retrieval(case_dir):
    requests = []
    client, model = model_factory(
        [
            initial_calls(),
            [final_message()],
            [
                function_call(
                    "search_protocol", {"query": "Packet Identifier"}, "search_2"
                )
            ],
            [final_message()],
        ],
        requests,
    )()
    graph = build_graph(InMemorySaver())

    async def exercise():
        first = await graph.ainvoke(
            {"question": "先检查确认范围。"},
            config(),
            context=RunContext(client, model, LocalTools(case_dir), "case-a"),
        )
        original_ids = set(first["clauses"])
        second = await graph.ainvoke(
            {"question": "再查编号关联。"},
            config(),
            context=RunContext(client, model, LocalTools(case_dir), "case-a"),
        )
        assert second["status"] == "completed"
        assert "K-IDENTIFIER" in second["clauses"]
        assert set(first["clauses"]) == original_ids
        assert "K-IDENTIFIER" not in first["clauses"]

    with client:
        asyncio.run(exercise())


def test_incomplete_model_output_never_executes_its_pending_tools(case_dir):
    requests = []
    response = {
        "id": "resp_incomplete",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": "incomplete",
        "output": initial_calls(),
        "usage": None,
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    client, model = model_factory([(200, response)], requests)()
    graph = build_graph(InMemorySaver())
    toolbox = LocalTools(case_dir)
    with client:
        result = asyncio.run(
            graph.ainvoke(
                {"question": "请检查。"},
                config(),
                context=RunContext(client, model, toolbox, "case-a"),
            )
        )
    assert result["status"] == "incomplete" and result["report"] is None
    assert toolbox.evidence == toolbox.clauses == {}
    assert not any(item["type"] == "tool_result" for item in result["trace"])


def test_conflicting_reference_from_new_toolbox_is_rejected_atomically(case_dir):
    requests = []
    client, model = model_factory(
        [
            initial_calls(),
            [final_message()],
            [function_call("read_log", {"role": "publisher"}, "log_again")],
            [final_message()],
        ],
        requests,
    )()
    graph = build_graph(InMemorySaver())

    class ConflictingTools:
        tools = LocalTools.tools
        transport = "test_conflict"

        async def call(self, name, arguments):
            return {
                "ok": True,
                "records": [
                    {"evidence_id": "E-new", "raw": "SHOULD-NOT-BE-REGISTERED"},
                    {"evidence_id": "E-publisher-stdout-1", "raw": "CHANGED-CONTENT"},
                ],
            }

    async def exercise():
        first = await graph.ainvoke(
            {"question": "请先检查。"},
            config(),
            context=RunContext(client, model, LocalTools(case_dir), "case-a"),
        )
        second = await graph.ainvoke(
            {"question": "再核对一次原文。"},
            config(),
            context=RunContext(client, model, ConflictingTools(), "case-a"),
        )
        assert second["evidence"] == first["evidence"]
        assert "E-new" not in second["evidence"]
        tool_results = [
            item for item in second["trace"] if item["type"] == "tool_result"
        ]
        assert len(tool_results) == 1 and tool_results[0]["result"]["ok"] is False
        replayed = [
            json.loads(item["output"])
            for item in requests[-1]["input"]
            if item.get("type") == "function_call_output"
            and item["call_id"] == "log_again"
        ]
        assert replayed and replayed[0]["ok"] is False

    with client:
        asyncio.run(exercise())
