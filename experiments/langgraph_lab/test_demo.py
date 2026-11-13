"""临时合成日志、本地模型 HTTP 替身、每问重建真实 stdio MCP 连接。"""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from examples.stage4.client import open_toolbox
from examples.stage4.test_integration import (
    final_message,
    function_call,
    model_factory,
    publisher_bytes,
)
from experiments.langgraph_lab.demo import run_lab


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
    (tmp_path / "publisher.json").write_bytes(publisher_bytes("lab/original"))
    return tmp_path


def test_two_questions_reconnect_mcp_and_keep_frozen_case_memory(case_dir):
    requests, roots, toolboxes, closed, prompts, clients = [], [], [], [], [], []
    factory = model_factory(
        [initial_calls(), [final_message()], [final_message()]], requests
    )

    def client_factory():
        client, model = factory()
        clients.append(client)
        return client, model

    @asynccontextmanager
    async def toolbox_factory(root):
        roots.append(root)
        assert (root / "publisher.json").read_bytes() == publisher_bytes("lab/original")
        async with open_toolbox(root) as toolbox:
            toolboxes.append(toolbox)
            assert toolbox.evidence == toolbox.clauses == {}
            yield toolbox
        closed.append(root)

    async def read_question(prompt):
        prompts.append(prompt)
        # 驱动等待下一问前，上一条 MCP 连接已经退出。
        assert len(closed) == len(roots)
        if len(prompts) == 1:
            (case_dir / "publisher.json").write_bytes(publisher_bytes("lab/changed"))
            return "请解释刚才的确认范围。"
        return "/exit"

    result = asyncio.run(
        run_lab(
            case_dir,
            "请先诊断。",
            thread_id="test-thread",
            client_factory=client_factory,
            toolbox_factory=toolbox_factory,
            read_question=read_question,
        )
    )
    assert result["status"] == "completed"
    assert len(result["turns"]) == 2
    assert len(toolboxes) == len(closed) == 2 and toolboxes[0] is not toolboxes[1]
    assert len(set(roots)) == 1 and all(not root.exists() for root in roots)
    assert len(requests) == 3
    follow_up = requests[-1]["input"]
    assert any(item.get("type") == "function_call_output" for item in follow_up)
    assert "lab/original" in json.dumps(follow_up)
    assert "lab/changed" not in json.dumps(requests)
    assert toolboxes[1].evidence == toolboxes[1].clauses == {}
    exported = json.dumps(result)
    assert "OPAQUE-REASONING-ITEM" not in exported
    assert "TEST-KEY" not in exported and "PRIVATE-COMMAND" not in exported
    assert clients and all(client.is_closed() for client in clients)


def test_failed_first_question_stops_without_prompt_and_cleans_up(case_dir):
    roots, closed, requests = [], [], []

    @asynccontextmanager
    async def toolbox_factory(root):
        roots.append(root)
        async with open_toolbox(root) as toolbox:
            yield toolbox
        closed.append(root)

    async def forbidden_input(prompt):
        raise AssertionError("首问未完成时不应继续收取下一问")

    result = asyncio.run(
        run_lab(
            case_dir,
            "请检查。",
            max_rounds=1,
            client_factory=model_factory([initial_calls()], requests),
            toolbox_factory=toolbox_factory,
            read_question=forbidden_input,
        )
    )
    assert result["status"] == "incomplete"
    assert len(result["turns"]) == len(requests) == len(roots) == len(closed) == 1
    assert all(not root.exists() for root in roots)


def test_model_failure_keeps_raw_upstream_details_out_and_closes_mcp(case_dir):
    roots, closed, requests = [], [], []

    @asynccontextmanager
    async def toolbox_factory(root):
        roots.append(root)
        try:
            async with open_toolbox(root) as toolbox:
                yield toolbox
        finally:
            closed.append(root)

    result = asyncio.run(
        run_lab(
            case_dir,
            "请检查。",
            client_factory=model_factory(
                [(401, {"error": {"message": "PRIVATE-UPSTREAM-ERROR"}})], requests
            ),
            toolbox_factory=toolbox_factory,
        )
    )
    assert result["status"] == "failed" and len(result["turns"]) == 1
    assert "PRIVATE-UPSTREAM-ERROR" not in json.dumps(result)
    assert roots == closed and all(not root.exists() for root in roots)


@pytest.mark.parametrize("content", [b"\xff", b"not json", b"[]"])
def test_invalid_case_is_rejected_before_creating_model_or_tools(case_dir, content):
    (case_dir / "publisher.json").write_bytes(content)

    def forbidden_client():
        raise AssertionError("无效案例不能创建模型连接")

    with pytest.raises(ValueError):
        asyncio.run(run_lab(case_dir, "请检查。", client_factory=forbidden_client))
