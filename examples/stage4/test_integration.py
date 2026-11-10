"""本地 HTTP 替身 + 真实 stdio MCP：不读取 .env，不请求实际模型。"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError

from examples.stage4.agent_demo import run_agent
from examples.stage4.client import MCPTools, open_toolbox
from examples.stage4.service import (
    DiagnosisRequest,
    UploadedLog,
    analyze,
    request_identity,
)


def publisher_bytes(topic="lab/demo"):
    return json.dumps(
        {
            "command": ["PRIVATE-COMMAND-MUST-NOT-REACH-MODEL"],
            "exit_code": 0,
            "stdout": "Client demo-pub sending PUBLISH "
            f"(d0, q1, r0, m1, '{topic}', ... (20 bytes))\n",
            "stderr": "",
        }
    ).encode("utf-8")


def request_for(content=None, **options):
    return DiagnosisRequest(
        question="请核查发布者日志与确认范围。",
        files=[
            UploadedLog(
                role="publisher",
                file_name="../../untrusted-upload-name.json",
                content=publisher_bytes() if content is None else content,
            )
        ],
        **options,
    )


def function_call(name, arguments, call_id):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def final_message(evidence_ids=None):
    report = {
        "summary": "已观察到发布；不能由此确定最终订阅者收到。",
        "evidence_ids": evidence_ids or ["E-publisher-stdout-1"],
        "clause_ids": ["K-PUBACK"],
        "limitations": ["缺少订阅者证据；检索内容为教学摘要。"],
    }
    return {
        "type": "message",
        "id": "msg_final",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": json.dumps(report, ensure_ascii=False),
                "annotations": [],
            }
        ],
    }


def model_factory(replies, requests):
    """使用实际 OpenAI SDK，只将 HTTP 传输替换为本地函数。"""

    def handler(request):
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        assert len(requests) <= len(replies), "发生了未预期的额外模型请求"
        reply = replies[len(requests) - 1]
        if isinstance(reply, tuple):
            status_code, body = reply
            return httpx.Response(status_code, json=body)
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(requests)}",
                "object": "response",
                "created_at": 1,
                "model": "test-model",
                "status": "completed",
                "output": reply,
                "usage": None,
            },
        )

    def create():
        return (
            OpenAI(
                base_url="http://model.test.invalid/v1",
                api_key="TEST-KEY-NOT-A-CREDENTIAL",
                timeout=5,
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            ),
            "test-model",
        )

    return create


@pytest.fixture
def observed_toolbox():
    paths = []

    @asynccontextmanager
    async def factory(case_dir):
        paths.append(case_dir)
        # 上传名称不能用作落盘路径，必须转换为受控的角色文件名。
        assert {path.name for path in case_dir.iterdir()} == {"publisher.json"}
        async with open_toolbox(case_dir) as toolbox:
            yield toolbox

    return factory, paths


def test_stage4_local_adapter_round_trip(tmp_path):
    """第四阶段自己的 Agent 也保留本地工具对比入口。"""
    (tmp_path / "publisher.json").write_bytes(publisher_bytes())
    requests = []
    client, model = model_factory(
        [
            [
                function_call("read_log", {"role": "publisher"}, "log_1"),
                function_call("search_protocol", {"query": "PUBACK"}, "search_1"),
            ],
            [final_message()],
        ],
        requests,
    )()
    with client:
        result = run_agent(client, model, tmp_path, "请核查日志与确认范围。")
    assert result["status"] == "completed"
    assert result["transport"] == "local"
    assert "E-publisher-stdout-1" in result["evidence"]
    assert "K-PUBACK" in result["clauses"]
    assert len(requests) == 2
    assert "PRIVATE-COMMAND" not in json.dumps(requests)
    tool_trace = [item for item in result["trace"] if item["type"] == "tool_result"]
    assert len(tool_trace) == 2
    assert all(item["transport"] == "local" for item in tool_trace)


def test_application_model_mcp_round_trip_and_cleanup(observed_toolbox):
    requests = []
    # 先检索，再根据已有结果读取日志；不是脚本中写死“读日志再检索”。
    replies = [
        [function_call("search_protocol", {"query": "PUBACK"}, "search_1")],
        [function_call("read_log", {"role": "publisher"}, "log_1")],
        [final_message()],
    ]
    factory, paths = observed_toolbox
    result = analyze(
        request_for(),
        client_factory=model_factory(replies, requests),
        toolbox_factory=factory,
    )
    assert result["status"] == "completed"
    assert result["transport"] == "mcp_stdio"
    assert len(requests) == 3
    assert paths and all(not path.exists() for path in paths)
    returned = {
        item["call_id"]: json.loads(item["output"])
        for item in requests[-1]["input"]
        if item.get("type") == "function_call_output"
    }
    assert set(returned) == {"search_1", "log_1"}
    assert returned["log_1"]["records"][0]["evidence_id"] == "E-publisher-stdout-1"
    assert any(
        row["clause_id"] == "K-PUBACK" for row in returned["search_1"]["clauses"]
    )
    tool_trace = [item for item in result["trace"] if item["type"] == "tool_result"]
    assert [item["name"] for item in tool_trace] == ["search_protocol", "read_log"]
    assert all(item["transport"] == "mcp_stdio" for item in tool_trace)
    assert "PRIVATE-COMMAND" not in json.dumps(requests)
    assert "TEST-KEY" not in json.dumps(result)
    assert result["input_manifest"][0]["case_file"] == "publisher.json"


def test_unavailable_evidence_does_not_become_a_completed_report(observed_toolbox):
    factory, paths = observed_toolbox
    requests = []
    result = analyze(
        request_for(),
        client_factory=model_factory(
            [
                [
                    function_call("read_log", {"role": "publisher"}, "log_1"),
                    function_call("search_protocol", {"query": "PUBACK"}, "search_1"),
                ],
                [final_message(["E-other-case-1"])],
            ],
            requests,
        ),
        toolbox_factory=factory,
    )
    assert result["status"] == "incomplete"
    assert result["report"] is None
    assert result["evidence"]
    assert all(not path.exists() for path in paths)


def test_budget_stops_before_executing_excess_tools(observed_toolbox):
    factory, paths = observed_toolbox
    requests = []
    result = analyze(
        request_for(max_tool_calls=1),
        client_factory=model_factory(
            [
                [
                    function_call("read_log", {"role": "publisher"}, "log_1"),
                    function_call("search_protocol", {"query": "PUBACK"}, "search_1"),
                ]
            ],
            requests,
        ),
        toolbox_factory=factory,
    )
    assert result["status"] == "incomplete"
    assert result["evidence"] == result["clauses"] == {}
    assert result["trace"][-1]["type"] == "budget_stop"
    assert len(requests) == 1
    assert all(not path.exists() for path in paths)


def test_model_failure_closes_tools_and_removes_uploads(observed_toolbox):
    factory, paths = observed_toolbox
    result = analyze(
        request_for(),
        client_factory=model_factory(
            [
                (
                    401,
                    {
                        "error": {
                            "message": "PRIVATE-PROXY-DETAIL",
                            "type": "auth_error",
                        }
                    },
                )
            ],
            [],
        ),
        toolbox_factory=factory,
    )
    assert result["status"] == "failed"
    assert result["report"] is None
    assert "401" in result["reason"]
    assert "PRIVATE-PROXY-DETAIL" not in json.dumps(result)
    assert paths and all(not path.exists() for path in paths)


def test_mcp_startup_failure_closes_model_client_and_removes_uploads():
    paths = []
    created_clients = []
    requests = []
    original_factory = model_factory([], requests)

    def client_factory():
        client, model = original_factory()
        created_clients.append(client)
        return client, model

    @asynccontextmanager
    async def unavailable(case_dir):
        paths.append(case_dir)
        if case_dir.exists():
            raise RuntimeError("PRIVATE-SERVER-STARTUP-DETAIL")
        yield None

    result = analyze(
        request_for(),
        client_factory=client_factory,
        toolbox_factory=unavailable,
    )
    assert result["status"] == "failed"
    assert result["report"] is None
    assert requests == []
    assert paths and all(not path.exists() for path in paths)
    assert all(client.is_closed() for client in created_clients)
    assert "PRIVATE-SERVER-STARTUP-DETAIL" not in json.dumps(result)


@pytest.mark.parametrize("content", [b"\xff", b"not json", b"[]", b'{"exit_code":"0"}'])
def test_invalid_upload_is_rejected_before_model_or_mcp(content):
    def forbidden():
        raise AssertionError("无效日志不应创建模型客户端")

    result = analyze(request_for(content), client_factory=forbidden)
    assert result["status"] == "failed"
    assert result["report"] is None
    assert result["evidence"] == {}


def test_request_constraints_and_identity():
    first = request_for()
    changed_question = first.model_copy(update={"question": "另一个问题"})
    changed_file = request_for(publisher_bytes("lab/other"))
    changed_budget = request_for(max_rounds=3)
    assert (
        len(
            {
                request_identity(item)
                for item in [first, changed_question, changed_file, changed_budget]
            }
        )
        == 4
    )
    with pytest.raises(ValidationError):
        DiagnosisRequest(question="检查", files=first.files * 2)
    with pytest.raises(ValidationError):
        DiagnosisRequest(question="   ", files=first.files)
    with pytest.raises(ValidationError):
        request_for(b"x" * (10 * 1024 * 1024 + 1))


def test_two_real_mcp_sessions_keep_case_evidence_separate(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "publisher.json").write_bytes(publisher_bytes("lab/first"))
    (second / "publisher.json").write_bytes(publisher_bytes("lab/second"))

    async def run():
        async with open_toolbox(first) as left:
            await left.call("search_protocol", '{"query":"PUBACK"}')
            left_result = await left.call("read_log", '{"role":"publisher"}')
            async with open_toolbox(second) as right:
                right_result = await right.call("read_log", '{"role":"publisher"}')
                assert right.clauses == {}
                assert left.clauses
                assert left_result["input_sha256"] != right_result["input_sha256"]
                assert "lab/first" in left.evidence["E-publisher-stdout-1"]["raw"]
                assert "lab/second" in right.evidence["E-publisher-stdout-1"]["raw"]

    asyncio.run(run())


def test_bad_tool_requests_are_not_sent_to_mcp():
    class ForbiddenClient:
        async def call_tool(self, *args):
            raise AssertionError("非法工具请求不应发往 MCP")

    async def run():
        toolbox = MCPTools(ForbiddenClient())
        for name, arguments in [
            ("not_registered", "{}"),
            ("read_log", "not json"),
            ("read_log", '{"role":"publisher","path":"private.txt"}'),
        ]:
            result = await toolbox.call(name, arguments)
            assert result["ok"] is False
            assert "未发送" in result["error"] or "未知工具" in result["error"]
        assert toolbox.evidence == toolbox.clauses == {}

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["disconnect", "malformed", "partial_records"])
def test_transport_or_result_failure_never_registers_evidence(failure):
    class FaultyClient:
        async def call_tool(self, *args):
            if failure == "disconnect":
                raise RuntimeError("PRIVATE-TRANSPORT-DETAIL")
            content = (
                {"ok": "true"}
                if failure == "malformed"
                else {
                    "ok": True,
                    "records": [{"evidence_id": "E-first"}, {"raw": "missing id"}],
                }
            )
            return SimpleNamespace(is_error=False, structured_content=content)

    async def run():
        toolbox = MCPTools(FaultyClient())
        result = await toolbox.call("read_log", '{"role":"publisher"}')
        assert result["ok"] is False
        assert toolbox.evidence == {}
        assert "PRIVATE-TRANSPORT-DETAIL" not in json.dumps(result)

    asyncio.run(run())
