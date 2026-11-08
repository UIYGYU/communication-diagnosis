"""真实 SDK + 本地 HTTP 替身：检查请求与执行循环，不连接代理或模型服务。"""

import json

import httpx
import pytest
from openai import OpenAI

from examples.stage3.agent_demo import run_agent


@pytest.fixture
def case_dir(tmp_path):
    record = {
        "command": ["PRIVATE-COMMAND-MUST-NOT-REACH-MODEL"],
        "exit_code": 0,
        "stdout": "Client demo-pub sending PUBLISH "
        "(d0, q1, r0, m1, 'lab/demo', ... (20 bytes))\n",
        "stderr": "",
    }
    (tmp_path / "publisher.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def function_call(name, arguments, call_id):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def tool_calls():
    return [
        function_call("read_log", {"role": "publisher"}, "log_1"),
        function_call("search_protocol", {"query": "PUBACK"}, "search_1"),
    ]


def final_message(**changes):
    report = {
        "summary": "发布端日志包含 PUBLISH；不能据此确定订阅者收到。",
        "evidence_ids": ["E-publisher-stdout-1"],
        "clause_ids": ["K-PUBACK"],
        "limitations": ["仅使用 MQTT 5 教学摘要，缺少订阅者证据。"],
    }
    report.update(changes)
    return message(json.dumps(report, ensure_ascii=False))


def message(text):
    return {
        "id": "msg_result",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def response(output, status="completed"):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": status,
        "output": output,
        "usage": None,
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
    }


def execute(case_dir, replies, **options):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        assert len(requests) <= len(replies), "发生了未预期的额外模型请求"
        reply = replies[len(requests) - 1]
        if isinstance(reply, tuple):
            code, body = reply
            return httpx.Response(code, json=body)
        return httpx.Response(200, json=reply)

    with OpenAI(
        base_url="http://model.test.invalid/v1",
        api_key="TEST-KEY-NOT-A-REAL-CREDENTIAL",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        result = run_agent(
            client, "test-model", case_dir, "请核查日志和确认范围", **options
        )
    return result, requests


def test_sdk_round_trip_preserves_context_and_returns_each_call_result(case_dir):
    reasoning = {
        "id": "reasoning_1",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "opaque-test-context",
    }
    first_output = [reasoning, *tool_calls()]
    result, requests = execute(
        case_dir, [response(first_output), response([final_message()])]
    )

    assert result["status"] == "completed"
    assert len(requests) == 2
    assert requests[0]["store"] is False
    assert requests[0]["tool_choice"] == "auto"
    assert {tool["name"] for tool in requests[0]["tools"]} == {
        "read_log",
        "search_protocol",
    }
    history = requests[1]["input"]
    assert (
        next(item for item in history if item.get("type") == "reasoning")[
            "encrypted_content"
        ]
        == "opaque-test-context"
    )
    returned = {
        item["call_id"]: json.loads(item["output"])
        for item in history
        if item.get("type") == "function_call_output"
    }
    assert set(returned) == {"log_1", "search_1"}
    assert returned["log_1"]["records"][0]["evidence_id"] == "E-publisher-stdout-1"
    assert any(
        item["clause_id"] == "K-PUBACK" for item in returned["search_1"]["clauses"]
    )
    assert "PRIVATE-COMMAND" not in json.dumps(requests)
    assert "opaque-test-context" not in json.dumps(result)
    assert "TEST-KEY" not in json.dumps(result)


def test_answer_without_tool_execution_is_not_completed(case_dir):
    result, requests = execute(case_dir, [response([final_message()])])
    assert result["status"] == "incomplete"
    assert result["report"] is None
    assert result["evidence"] == {}
    assert len(requests) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence_ids": ["E-other-task-1"]},
        {"clause_ids": ["K-NOT-RETRIEVED"]},
        {"evidence_ids": []},
        {"limitations": []},
        {"unexpected": "extra field"},
    ],
)
def test_invalid_or_unavailable_report_references_are_rejected(case_dir, changes):
    result, _ = execute(
        case_dir, [response(tool_calls()), response([final_message(**changes)])]
    )
    assert result["status"] == "incomplete"
    assert result["report"] is None
    assert result["evidence"]


@pytest.mark.parametrize("text", ["这不是 JSON", "```json\n{}\n```", ""])
def test_invalid_final_output_retains_tool_trace(case_dir, text):
    result, _ = execute(case_dir, [response(tool_calls()), response([message(text)])])
    assert result["status"] == "incomplete"
    assert result["report"] is None
    assert len([item for item in result["trace"] if item["type"] == "tool_result"]) == 2


@pytest.mark.parametrize("options", [{"max_rounds": 1}, {"max_tool_calls": 1}])
def test_budget_stop_does_not_execute_tools_without_room_to_continue(case_dir, options):
    result, requests = execute(case_dir, [response(tool_calls())], **options)
    assert result["status"] == "incomplete"
    assert result["evidence"] == {}
    assert result["clauses"] == {}
    assert result["trace"][-1]["type"] == "budget_stop"
    assert len(requests) == 1


def test_tool_error_is_returned_to_model_and_can_be_recovered(case_dir):
    result, requests = execute(
        case_dir,
        [
            response([function_call("unregistered", {}, "bad_call")]),
            response(tool_calls()),
            response([final_message()]),
        ],
    )
    assert result["status"] == "completed"
    error_return = next(
        item
        for item in requests[1]["input"]
        if item.get("type") == "function_call_output"
    )
    assert error_return["call_id"] == "bad_call"
    assert json.loads(error_return["output"])["ok"] is False
    assert result["trace"][1]["result"]["ok"] is False


def test_incomplete_model_output_does_not_execute_embedded_calls(case_dir):
    result, _ = execute(case_dir, [response(tool_calls(), status="incomplete")])
    assert result["status"] == "incomplete"
    assert result["evidence"] == {}
    assert not any(item["type"] == "tool_result" for item in result["trace"])


def test_duplicate_call_ids_fail_before_execution(case_dir):
    calls = tool_calls()
    calls[1]["call_id"] = calls[0]["call_id"]
    result, _ = execute(case_dir, [response(calls)])
    assert result["status"] == "failed"
    assert result["evidence"] == {}


def test_api_error_does_not_copy_private_response_body_into_report(case_dir):
    result, requests = execute(
        case_dir,
        [(401, {"error": {"message": "PRIVATE-PROXY-DETAIL", "type": "auth_error"}})],
    )
    assert result["status"] == "failed"
    assert "401" in result["reason"]
    assert "PRIVATE-PROXY-DETAIL" not in json.dumps(result)
    assert len(requests) == 1


def test_empty_log_cannot_support_a_successful_report(case_dir):
    (case_dir / "publisher.json").write_text(
        json.dumps({"command": [], "exit_code": 0, "stdout": "", "stderr": ""}),
        encoding="utf-8",
    )
    result, _ = execute(case_dir, [response(tool_calls()), response([final_message()])])
    assert result["status"] == "incomplete"
    assert result["evidence"] == {}
