import json

import pytest
from pydantic import ValidationError

from examples.stage2.models import ClientLog, Evidence, ParseResult
from examples.stage2.parser import (
    parse_broker_text,
    parse_client_text,
    read_client,
)

BODY = "(d0, q1, r0, m7, 'lab/demo/status', ... (20 bytes))"
CLIENT_LINE = f"Client demo-pub sending PUBLISH {BODY}"
BROKER_LINE = (
    f"2026-11-01T15:41:27.852587635Z 1793547687: Received PUBLISH from demo-pub {BODY}"
)


def client_json(stdout: str, stderr: str = "") -> str:
    return json.dumps(
        {
            "command": ["example-command-not-to-run"],
            "exit_code": 0,
            "stdout": stdout,
            "stderr": stderr,
        }
    )


def test_client_event_and_original_location():
    text = client_json(f"unrecognized first line\n{CLIENT_LINE}\n")
    result = parse_client_text(text, "publisher.json", "publisher")
    assert len(result.events) == 1
    event = result.events[0]
    assert (event.direction, event.client_id) == ("send", "demo-pub")
    assert (event.qos, event.packet_id, event.topic) == (1, 7, "lab/demo/status")
    assert event.payload_size == 20
    assert event.dup is False
    assert event.retain is False
    assert event.broker_time_raw is None
    assert event.container_time_raw is None
    assert event.evidence.source_file == "publisher.json"
    assert event.evidence.field == "stdout"
    assert event.evidence.line_number == 2
    assert event.evidence.raw == CLIENT_LINE
    assert result.unknown_lines[0].evidence.line_number == 1


def test_broker_preserves_both_time_strings():
    result = parse_broker_text(BROKER_LINE, "broker.log")
    event = result.events[0]
    assert event.direction == "receive"
    assert event.broker_time_raw == "1793547687"
    assert event.container_time_raw == "2026-11-01T15:41:27.852587635Z"
    assert event.evidence.field == "text"
    assert event.evidence.line_number == 1
    assert event.evidence.raw == BROKER_LINE


def test_unknown_and_duplicate_lines_are_retained():
    text = client_json(f"{CLIENT_LINE}\nnew format line\n{CLIENT_LINE}\n")
    result = parse_client_text(text, "publisher.json", "publisher")
    assert [event.evidence.line_number for event in result.events] == [1, 3]
    assert len(result.unknown_lines) == 1
    assert result.unknown_lines[0].evidence.raw == "new format line"


def test_streams_keep_their_own_line_numbers():
    result = parse_client_text(
        client_json(CLIENT_LINE, CLIENT_LINE), "publisher.json", "publisher"
    )
    locations = {
        (event.evidence.field, event.evidence.line_number) for event in result.events
    }
    assert locations == {("stdout", 1), ("stderr", 1)}
    assert all(event.broker_time_raw is None for event in result.events)


@pytest.mark.parametrize("packet_id", [0, 65536])
def test_invalid_packet_id_becomes_unknown(packet_id):
    line = CLIENT_LINE.replace("m7,", f"m{packet_id},")
    result = parse_client_text(client_json(line), "publisher.json", "publisher")
    assert result.events == []
    assert len(result.unknown_lines) == 1
    assert result.unknown_lines[0].evidence.raw == line
    assert "模型校验" in result.unknown_lines[0].reason


@pytest.mark.parametrize("value", ["0", None, True])
def test_exit_code_does_not_accept_wrong_types(value):
    data = json.loads(client_json(CLIENT_LINE))
    data["exit_code"] = value
    with pytest.raises(ValidationError):
        ClientLog.model_validate(data)


def test_missing_field_is_an_error():
    data = json.loads(client_json(CLIENT_LINE))
    del data["stderr"]
    with pytest.raises(ValidationError):
        ClientLog.model_validate(data)


def test_broken_json_is_an_error():
    with pytest.raises(json.JSONDecodeError):
        parse_client_text("{", "publisher.json", "publisher")


def test_line_number_must_start_at_one():
    with pytest.raises(ValidationError):
        Evidence(
            source_file="broker.log",
            role="broker",
            field="text",
            line_number=0,
            raw="any line",
        )


def test_utf8_bom_file(tmp_path):
    path = tmp_path / "publisher.json"
    path.write_text(client_json(CLIENT_LINE), encoding="utf-8-sig")
    result = read_client(path, "publisher")
    assert len(result.events) == 1


def test_result_json_round_trip_excludes_command():
    result = parse_client_text(client_json(CLIENT_LINE), "publisher.json", "publisher")
    encoded = result.model_dump_json()
    assert "example-command-not-to-run" not in encoded
    restored = ParseResult.model_validate_json(encoded)
    assert restored == result


def test_empty_streams_produce_no_events():
    result = parse_client_text(client_json(""), "publisher.json", "publisher")
    assert result.events == []
    assert result.unknown_lines == []


# A：检查另外两个方向，样例直接在测试中构造。
def test_client_receive_direction_and_evidence():
    line = f"Client demo-sub received PUBLISH {BODY}"
    result = parse_client_text(client_json(line), "subscriber.json", "subscriber")

    assert len(result.events) == 1
    assert result.unknown_lines == []
    event = result.events[0]
    assert event.direction == "receive"
    assert event.client_id == "demo-sub"
    assert event.evidence == Evidence(
        source_file="subscriber.json",
        role="subscriber",
        field="stdout",
        line_number=1,
        raw=line,
    )


def test_broker_send_direction_and_evidence():
    line = (
        f"2026-11-01T15:41:27.852587635Z 1793547687: Sending PUBLISH to demo-sub {BODY}"
    )
    result = parse_broker_text(line, "broker.log")

    assert len(result.events) == 1
    assert result.unknown_lines == []
    event = result.events[0]
    assert event.direction == "send"
    assert event.client_id == "demo-sub"
    assert event.evidence == Evidence(
        source_file="broker.log",
        role="broker",
        field="text",
        line_number=1,
        raw=line,
    )


# B：正常事件与 stderr 中不支持的内容同时保留。
def test_timeout_line_and_publish_are_both_preserved():
    text = client_json(f"{CLIENT_LINE}\n", stderr="Timed out\n")
    result = parse_client_text(text, "publisher.json", "publisher")

    assert len(result.events) == 1
    assert result.events[0].evidence == Evidence(
        source_file="publisher.json",
        role="publisher",
        field="stdout",
        line_number=1,
        raw=CLIENT_LINE,
    )
    assert len(result.unknown_lines) == 1
    assert result.unknown_lines[0].evidence == Evidence(
        source_file="publisher.json",
        role="publisher",
        field="stderr",
        line_number=1,
        raw="Timed out",
    )


# C：坏输入应明确报错，文件样例只写入 pytest 的临时目录。
def test_stdout_array_is_a_structure_error():
    data = json.loads(client_json(CLIENT_LINE))
    data["stdout"] = [CLIENT_LINE]

    with pytest.raises(ValidationError):
        parse_client_text(json.dumps(data), "publisher.json", "publisher")


def test_root_array_is_a_structure_error():
    with pytest.raises(ValidationError):
        parse_client_text("[]", "publisher.json", "publisher")


def test_unsupported_field_is_a_structure_error():
    data = json.loads(client_json(CLIENT_LINE))
    data["unsupported_field"] = "example"

    with pytest.raises(ValidationError):
        parse_client_text(json.dumps(data), "publisher.json", "publisher")


def test_empty_json_file_is_a_syntax_error(tmp_path):
    path = tmp_path / "publisher.json"
    path.write_text("", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_client(path, "publisher")


def test_invalid_utf8_file_is_an_encoding_error(tmp_path):
    path = tmp_path / "publisher.json"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        read_client(path, "publisher")


# 命令行入口只显示校验位置和说明，不回显原始 command。
def test_cli_structure_error_omits_command(tmp_path, monkeypatch, capsys):
    from examples.stage2.demo import main

    command_marker = "private-command-marker"
    data = json.loads(client_json(CLIENT_LINE))
    data["command"] = [command_marker]
    data["stdout"] = [CLIENT_LINE]
    path = tmp_path / "publisher.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["demo", "--case-dir", str(tmp_path)])

    assert main() == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "输入结构校验失败" in output
    assert "stdout" in output
    assert command_marker not in output
