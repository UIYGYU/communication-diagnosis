"""上下文与短期记忆：合成案例、本地 HTTP 替身，不读取本机模型配置。"""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from examples.stage4.agent_demo import (
    AgentSession,
    LocalTools,
    context_size_bytes,
    run_agent_async,
)
from examples.stage4.client import open_toolbox
from examples.stage4.context_demo import run_context_session
from examples.stage4.test_integration import (
    final_message,
    function_call,
    model_factory,
    publisher_bytes,
)


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


@pytest.fixture
def case_dir(tmp_path):
    (tmp_path / "publisher.json").write_bytes(publisher_bytes())
    return tmp_path


def test_follow_up_receives_complete_prior_output_and_tools(case_dir):
    requests = []
    client, model = model_factory(
        [initial_calls(), [final_message()], [final_message()]], requests
    )()
    session = AgentSession()
    toolbox = LocalTools(case_dir)

    async def exercise():
        first = await run_agent_async(
            client, model, "请先诊断这个案例。", toolbox, session=session
        )
        second = await run_agent_async(
            client,
            model,
            "为什么确认不能证明最终订阅者收到？",
            toolbox,
            session=session,
        )
        return first, second

    with client:
        first, second = asyncio.run(exercise())
    assert first["status"] == second["status"] == "completed"
    assert len(requests) == 3
    follow_up = requests[2]["input"]
    assert follow_up[-1]["role"] == "user"
    assert "为什么确认" in follow_up[-1]["content"]
    assert any(
        item.get("type") == "reasoning"
        and item.get("encrypted_content") == "OPAQUE-REASONING-ITEM"
        for item in follow_up
    )
    returned = {
        item["call_id"]: json.loads(item["output"])
        for item in follow_up
        if item.get("type") == "function_call_output"
    }
    assert set(returned) == {"log_1", "search_1"}
    assert returned["log_1"]["records"][0]["evidence_id"] == "E-publisher-stdout-1"
    assert any(
        row["clause_id"] == "K-PUBACK" for row in returned["search_1"]["clauses"]
    )
    # 第二问没有再调用工具，但仍经过 JSON 与真实引用集合校验。
    assert not any(item["type"] == "tool_result" for item in second["trace"])
    assert "history" not in second
    assert "TEST-KEY" not in json.dumps(second)


def test_session_rejects_different_toolbox_and_model(case_dir):
    requests = []
    client, model = model_factory([initial_calls(), [final_message()]], requests)()
    session = AgentSession()
    toolbox = LocalTools(case_dir)

    async def exercise():
        assert (
            await run_agent_async(client, model, "诊断。", toolbox, session=session)
        )["status"] == "completed"
        with pytest.raises(ValueError, match="另一个案例或工具连接"):
            await run_agent_async(
                client, model, "追问。", LocalTools(case_dir), session=session
            )
        with pytest.raises(ValueError, match="更换模型"):
            await run_agent_async(
                client, "other-model", "追问。", toolbox, session=session
            )

    with client:
        asyncio.run(exercise())
    assert len(requests) == 2


@pytest.mark.parametrize("kind", ["pending_call", "bad_reference", "api_error"])
def test_failed_or_incomplete_session_cannot_continue(case_dir, kind):
    requests = []
    if kind == "pending_call":
        replies, rounds = [initial_calls()], 1
    elif kind == "bad_reference":
        replies, rounds = [initial_calls(), [final_message(["E-OTHER-CASE"])]], 6
    else:
        replies, rounds = [(401, {"error": {"message": "PRIVATE-ERROR"}})], 6
    client, model = model_factory(replies, requests)()
    toolbox, session = LocalTools(case_dir), AgentSession()

    async def exercise():
        result = await run_agent_async(
            client, model, "诊断。", toolbox, max_rounds=rounds, session=session
        )
        assert result["status"] != "completed"
        assert session.closed
        before = len(requests)
        with pytest.raises(ValueError, match="会话已经结束"):
            await run_agent_async(client, model, "继续。", toolbox, session=session)
        assert len(requests) == before
        assert "PRIVATE-ERROR" not in json.dumps(result)

    with client:
        asyncio.run(exercise())


def test_context_budget_stops_before_first_network_request(case_dir):
    requests = []
    client, model = model_factory([], requests)()
    session = AgentSession()
    with client:
        result = asyncio.run(
            run_agent_async(
                client,
                model,
                "诊断。",
                LocalTools(case_dir),
                session=session,
                max_context_bytes=1,
            )
        )
    assert result["status"] == "incomplete"
    assert not requests
    assert session.closed
    request_trace = result["trace"][0]
    assert request_trace["type"] == "model_request"
    assert request_trace["context_bytes"] > request_trace["limit"]
    assert request_trace["history_items"] == 1


def test_context_budget_checks_again_after_tool_outputs(case_dir):
    requests = []
    client, model = model_factory([initial_calls()], requests)()
    toolbox = LocalTools(case_dir)
    question = "诊断。"
    limit = context_size_bytes([{"role": "user", "content": question}], toolbox.tools)
    with client:
        result = asyncio.run(
            run_agent_async(
                client,
                model,
                question,
                toolbox,
                max_context_bytes=limit,
            )
        )
    assert result["status"] == "incomplete"
    assert len(requests) == 1
    checks = [item for item in result["trace"] if item["type"] == "model_request"]
    assert checks[0]["context_bytes"] == limit
    assert checks[1]["context_bytes"] > limit
    assert result["evidence"] and result["clauses"]


def test_completed_result_keeps_its_own_evidence_snapshot(case_dir):
    requests = []
    client, model = model_factory([initial_calls(), [final_message()]], requests)()
    toolbox = LocalTools(case_dir)

    async def exercise():
        first = await run_agent_async(client, model, "诊断。", toolbox)
        old_clauses = set(first["clauses"])
        await toolbox.call("search_protocol", '{"query":"Packet Identifier"}')
        assert "K-IDENTIFIER" in toolbox.clauses
        assert set(first["clauses"]) == old_clauses
        toolbox.evidence["E-publisher-stdout-1"]["raw"] = "后续修改"
        assert first["evidence"]["E-publisher-stdout-1"]["raw"] != "后续修改"

    with client:
        asyncio.run(exercise())


def test_session_driver_reuses_one_real_mcp_connection_and_cleans_up(case_dir):
    requests, roots, prompts = [], [], []
    client_factory = model_factory(
        [initial_calls(), [final_message()], [final_message()]], requests
    )

    @asynccontextmanager
    async def toolbox_factory(root):
        roots.append(root)
        assert (root / "publisher.json").read_bytes() == publisher_bytes()
        async with open_toolbox(root) as toolbox:
            yield toolbox

    async def follow_up(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            # 原目录后续变化不会改变当前会话已经固定的临时案例。
            (case_dir / "publisher.json").write_bytes(publisher_bytes("lab/changed"))
            return "请解释刚才的确认范围。"
        return "/exit"

    result = asyncio.run(
        run_context_session(
            case_dir,
            "请诊断。",
            client_factory=client_factory,
            toolbox_factory=toolbox_factory,
            read_question=follow_up,
        )
    )
    assert result["status"] == "completed"
    assert len(result["turns"]) == 2
    assert len(roots) == 1 and not roots[0].exists()
    assert len(requests) == 3
    assert "lab/changed" not in json.dumps(requests)
    exported = json.dumps(result)
    assert "OPAQUE-REASONING-ITEM" not in exported
    assert "TEST-KEY" not in exported and "PRIVATE-COMMAND" not in exported


def test_session_driver_stops_on_failure_and_cleans_up(case_dir):
    roots, requests = [], []

    @asynccontextmanager
    async def toolbox_factory(root):
        roots.append(root)
        async with open_toolbox(root) as toolbox:
            yield toolbox

    async def must_not_ask(prompt):
        raise AssertionError("失败后不能继续追问")

    result = asyncio.run(
        run_context_session(
            case_dir,
            "请诊断。",
            client_factory=model_factory([initial_calls()], requests),
            toolbox_factory=toolbox_factory,
            read_question=must_not_ask,
            max_rounds=1,
        )
    )
    assert result["status"] == "incomplete"
    assert len(result["turns"]) == len(requests) == 1
    assert roots and all(not root.exists() for root in roots)
