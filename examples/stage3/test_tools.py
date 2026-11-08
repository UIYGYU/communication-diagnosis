import json
from pathlib import Path

import pytest

from examples.stage3.tools import MAX_LOG_LINES, CaseTools

PUBLISH = "Client pub-1 sending PUBLISH (d0, q1, r0, m1, 'lab/status', ... (20 bytes))"


def write_client(directory: Path, stdout: str, stderr: str = "") -> Path:
    path = directory / "publisher.json"
    path.write_text(
        json.dumps(
            {
                "command": ["PRIVATE_COMMAND_MARKER"],
                "exit_code": 0,
                "stdout": stdout,
                "stderr": stderr,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_read_log_keeps_evidence_but_not_command(tmp_path):
    write_client(tmp_path, PUBLISH + "\nreceived PUBACK", "Timed out")
    tools = CaseTools(tmp_path)
    result = tools.call("read_log", '{"role":"publisher"}')
    assert result["ok"]
    assert result["parsed_events"] == 1
    assert result["unknown_lines"] == 2
    assert result["records"][0]["event"]["topic"] == "lab/status"
    assert result["records"][1]["unparsed"]
    assert "E-publisher-stderr-1" in tools.evidence
    assert "PRIVATE_COMMAND_MARKER" not in json.dumps(result)


def test_log_bytes_are_snapshotted(tmp_path):
    path = write_client(tmp_path, PUBLISH)
    tools = CaseTools(tmp_path)
    path.write_text("changed", encoding="utf-8")
    assert tools.call("read_log", '{"role":"publisher"}')["parsed_events"] == 1


def test_missing_file_is_not_normal_communication(tmp_path):
    result = CaseTools(tmp_path).call("read_log", '{"role":"subscriber"}')
    assert not result["ok"]
    assert "未提供" in result["error"]


def test_broker_direction_and_source_are_preserved(tmp_path):
    (tmp_path / "broker.log").write_text(
        "2026-11-08T00:00:00Z 1794096000: Sending PUBLISH to sub-1 "
        "(d0, q1, r0, m2, 'lab/status', ... (20 bytes))",
        encoding="utf-8",
    )
    result = CaseTools(tmp_path).call("read_log", '{"role":"broker"}')
    record = result["records"][0]
    assert record["evidence_id"] == "E-broker-text-1"
    assert record["source_file"] == "broker.log"
    assert record["event"]["direction"] == "send"


def test_input_budget_covers_all_role_files(tmp_path, monkeypatch):
    monkeypatch.setattr("examples.stage3.tools.MAX_INPUT_BYTES", 10)
    (tmp_path / "broker.log").write_bytes(b"123456")
    (tmp_path / "publisher.json").write_bytes(b"12345")
    with pytest.raises(ValueError, match="总大小"):
        CaseTools(tmp_path)


@pytest.mark.parametrize(
    "arguments",
    [
        '{"role":"../../secret"}',
        '{"role":1}',
        '{"role":"broker","command":"x"}',
        "[]",
        "bad json",
    ],
)
def test_invalid_arguments_are_rejected(tmp_path, arguments):
    assert not CaseTools(tmp_path).call("read_log", arguments)["ok"]


def test_unknown_tool_is_not_executed(tmp_path):
    assert not CaseTools(tmp_path).call("run_shell", '{"command":"anything"}')["ok"]


def test_malformed_log_does_not_echo_input(tmp_path):
    (tmp_path / "publisher.json").write_text(
        '{"command":"PRIVATE_COMMAND_MARKER"}', encoding="utf-8"
    )
    result = CaseTools(tmp_path).call("read_log", '{"role":"publisher"}')
    assert not result["ok"]
    assert "PRIVATE_COMMAND_MARKER" not in json.dumps(result)


def test_returned_lines_and_characters_have_explicit_limits(tmp_path):
    write_client(tmp_path, "\n".join(["x" * 600] * (MAX_LOG_LINES + 1)))
    tools = CaseTools(tmp_path)
    result = tools.call("read_log", '{"role":"publisher"}')
    assert result["truncated"]
    assert len(result["records"]) == MAX_LOG_LINES
    assert len(result["records"][0]["raw"]) == 500
    assert len(tools.evidence) == MAX_LOG_LINES
    assert f"E-publisher-stdout-{MAX_LOG_LINES + 1}" not in tools.evidence


def test_search_registers_only_returned_teaching_clauses(tmp_path):
    tools = CaseTools(tmp_path)
    result = tools.call("search_protocol", '{"query":"确认范围"}')
    assert result["ok"]
    assert "K-PUBACK" in tools.clauses
    assert all(item["version"] == "5.0" for item in result["clauses"])
    assert all(item["content_kind"] == "tutorial_summary" for item in result["clauses"])
    assert set(tools.clauses) == {item["clause_id"] for item in result["clauses"]}


@pytest.mark.parametrize("query", ['" OR *', "不存在的中文词", "notarealprotocolterm"])
def test_query_is_not_a_raw_fts_expression(tmp_path, query):
    result = CaseTools(tmp_path).call("search_protocol", json.dumps({"query": query}))
    assert result["ok"]
    assert result["clauses"] == []


def test_outside_link_is_rejected_before_reading(tmp_path, monkeypatch):
    # 模拟 resolve 发现文件链接到案例目录之外，避免依赖 Windows 符号链接权限。
    write_client(tmp_path, PUBLISH)
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path.name == "publisher.json":
            return tmp_path.parent / "outside.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="不能链接到目录外"):
        CaseTools(tmp_path)
