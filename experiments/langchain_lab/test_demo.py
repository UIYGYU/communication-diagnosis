"""真实 LangChain + Responses SDK 的离线验证；只使用合成日志和本地 HTTP。"""

import json
import sys

import httpx
import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from examples.stage4.test_integration import function_call, publisher_bytes
from experiments.langchain_lab import demo

TEST_MODEL = "private-proxy-model-name"


@pytest.fixture(autouse=True)
def fake_configuration(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://model.test.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "TEST-KEY-NOT-A-CREDENTIAL")
    monkeypatch.setenv("LLM_MODEL", TEST_MODEL)
    # 测试不启用任何追踪服务，即使运行终端曾设置过全局开关。
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")


@pytest.fixture
def case_dir(tmp_path):
    (tmp_path / "publisher.json").write_bytes(publisher_bytes())
    return tmp_path


def text_message(text="已看到发布记录；现有证据不能证明最终订阅者收到。"):
    return {
        "type": "message",
        "id": "msg_final",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def initial_calls(role="publisher", query="PUBACK"):
    return [
        function_call("read_log", {"role": role}, "log_1"),
        function_call("search_protocol", {"query": query}, "search_1"),
    ]


@pytest.fixture
def local_model():
    clients = []

    def create(replies, requests):
        def handler(request):
            assert request.url.path == "/v1/responses"
            payload = json.loads(request.content)
            requests.append(payload)
            assert payload["model"] == TEST_MODEL
            assert payload["store"] is False
            assert "previous_response_id" not in payload
            assert len(requests) <= len(replies), "出现了未预期的额外模型请求"
            reply = replies[len(requests) - 1]
            if isinstance(reply, tuple):
                status_code, body = reply
                return httpx.Response(status_code, json=body)
            response_status = "completed"
            incomplete = None
            if isinstance(reply, dict):
                response_status = reply["status"]
                incomplete = {"reason": "max_output_tokens"}
                reply = reply["output"]
            return httpx.Response(
                200,
                json={
                    "id": f"resp_{len(requests)}",
                    "object": "response",
                    "created_at": 1,
                    "model": TEST_MODEL,
                    "status": response_status,
                    "error": None,
                    "incomplete_details": incomplete,
                    "output": reply,
                    "usage": None,
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return demo.create_chat_model(http_client=client)

    yield create
    for client in clients:
        client.close()


def returned_tools(payload):
    return {
        item["call_id"]: json.loads(item["output"])
        for item in payload["input"]
        if item.get("type") == "function_call_output"
    }


def test_native_agent_calls_tools_and_replays_their_actual_results(
    case_dir, local_model
):
    requests = []
    model = local_model([initial_calls(), [text_message()]], requests)
    result = demo.run_demo(case_dir, "检查发布日志和确认范围。", model=model)
    assert result["status"] == "completed"
    assert len(requests) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_messages} == {"log_1", "search_1"}
    assert result["messages"][-1].text
    assert "E-publisher-stdout-1" in result["evidence"]
    assert "K-PUBACK" in result["clauses"]
    assert {item["name"] for item in result["calls"]} == {
        "read_log",
        "search_protocol",
    }
    tools = {item["name"]: item for item in requests[0]["tools"]}
    assert set(tools) == {"read_log", "search_protocol"}
    assert set(tools["read_log"]["parameters"]["properties"]["role"]["enum"]) == {
        "broker",
        "publisher",
        "subscriber",
    }
    assert tools["search_protocol"]["description"]
    returned = returned_tools(requests[1])
    assert returned["log_1"]["records"][0]["evidence_id"] == "E-publisher-stdout-1"
    assert any(c["clause_id"] == "K-PUBACK" for c in returned["search_1"]["clauses"])
    assert "PRIVATE-COMMAND" not in json.dumps(requests)


def test_model_only_cli_uses_the_same_responses_adapter(
    local_model, monkeypatch, capsys
):
    requests = []
    model = local_model([[text_message("LangChain 模型接口已连通。")]], requests)
    monkeypatch.setattr(demo, "create_chat_model", lambda: model)
    monkeypatch.setattr(sys, "argv", ["demo", "--model-only"])
    assert demo.main() == 0
    output = capsys.readouterr().out
    assert "LangChain 模型接口已连通" in output
    assert len(requests) == 1
    assert not requests[0].get("tools")


@pytest.mark.parametrize("failure", ["no_calls", "missing_log"])
def test_text_or_failed_tools_do_not_become_a_completed_loop(
    case_dir, local_model, failure
):
    replies = (
        [[text_message()]]
        if failure == "no_calls"
        else [initial_calls(role="subscriber"), [text_message("订阅端日志缺失。")]]
    )
    result = demo.run_demo(case_dir, "检查日志。", model=local_model(replies, []))
    assert result["status"] == "incomplete"
    assert result["evidence"] == {}


def test_empty_retrieval_is_returned_without_fabricated_clauses(case_dir, local_model):
    requests = []
    model = local_model(
        [
            initial_calls(query="zzzznotaprotocoltermzzzz"),
            [text_message("检索没有命中，无法引用协议摘要。")],
        ],
        requests,
    )
    result = demo.run_demo(case_dir, "读取日志并检查资料缺口。", model=model)
    # completed 只表示本入门例子的工具回路完成，不代表协议结论通过验收。
    assert result["status"] == "completed"
    assert result["clauses"] == {}
    returned = returned_tools(requests[1])["search_1"]
    assert returned["ok"] is True
    assert returned["clauses"] == []


def test_missing_configuration_is_rejected_before_creating_a_model(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(demo.LabInputError, match="LLM_API_KEY"):
        demo.create_chat_model()


@pytest.mark.parametrize(
    "address",
    ["ftp://model.test.invalid/v1", "http://model.test.invalid/v1/responses"],
)
def test_invalid_api_root_is_rejected(monkeypatch, address):
    monkeypatch.setenv("LLM_BASE_URL", address)
    with pytest.raises(demo.LabInputError):
        demo.create_chat_model()


def test_cli_hides_raw_upstream_error_and_credentials(local_model, monkeypatch, capsys):
    model = local_model(
        [
            (
                401,
                {"error": {"message": "PRIVATE-UPSTREAM-DETAIL", "type": "auth_error"}},
            )
        ],
        [],
    )
    monkeypatch.setattr(demo, "create_chat_model", lambda: model)
    monkeypatch.setattr(sys, "argv", ["demo", "--model-only"])
    assert demo.main() == 1
    output = capsys.readouterr().out
    assert "401" in output
    assert "PRIVATE-UPSTREAM-DETAIL" not in output
    assert "TEST-KEY-NOT-A-CREDENTIAL" not in output
    assert "model.test.invalid" not in output


def test_agent_cannot_run_an_unbounded_tool_loop(case_dir, local_model):
    replies = [
        [function_call("read_log", {"role": "publisher"}, f"log_{number}")]
        for number in range(20)
    ]
    requests = []
    with pytest.raises(GraphRecursionError):
        demo.run_demo(case_dir, "检查日志。", model=local_model(replies, requests))
    assert 1 < len(requests) < len(replies)


def test_partial_response_is_not_a_completed_answer(case_dir, local_model):
    model = local_model(
        [
            initial_calls(),
            {"status": "incomplete", "output": [text_message("尚未完成的部分结论。")]},
        ],
        [],
    )
    result = demo.run_demo(case_dir, "检查日志。", model=model)
    assert result["messages"][-1].response_metadata["status"] == "incomplete"
    assert result["status"] == "incomplete"
