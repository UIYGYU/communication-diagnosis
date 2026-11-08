import json
import re
from pathlib import Path

from pydantic import ValidationError

from examples.stage2.models import (
    ClientLog,
    ClientRole,
    Event,
    Evidence,
    ParseResult,
    Role,
    Stream,
    UnknownLine,
)

PAYLOAD = (
    r"\(d(?P<dup>[01]), q1, r(?P<retain>[01]), "
    r"m(?P<packet_id>\d+), '(?P<topic>[^']*)', "
    r"\.\.\. \((?P<payload_size>\d+) bytes\)\)"
)
CLIENT_PUBLISH = re.compile(
    r"Client (?P<client_id>\S+) (?P<action>sending|received) PUBLISH " + PAYLOAD
)
BROKER_PUBLISH = re.compile(
    r"(?P<container_time>\S+) (?P<broker_time>\d+): "
    r"(?P<action>Received PUBLISH from|Sending PUBLISH to) "
    r"(?P<client_id>\S+) " + PAYLOAD
)


def _parse_lines(
    text: str,
    source_file: str,
    role: Role,
    field: Stream,
    pattern: re.Pattern[str],
    result: ParseResult,
) -> None:
    for line_number, raw in enumerate(text.splitlines(), start=1):
        evidence = Evidence(
            source_file=source_file,
            role=role,
            field=field,
            line_number=line_number,
            raw=raw,
        )
        match = pattern.fullmatch(raw)
        if match is None:
            result.unknown_lines.append(
                UnknownLine(
                    evidence=evidence,
                    reason="当前练习仅识别指定格式的 QoS 1 PUBLISH",
                )
            )
            continue

        values = match.groupdict()
        receiving = values["action"] in {"received", "Received PUBLISH from"}
        try:
            event = Event(
                direction="receive" if receiving else "send",
                client_id=values["client_id"],
                topic=values["topic"],
                packet_id=int(values["packet_id"]),
                dup=values["dup"] == "1",
                retain=values["retain"] == "1",
                payload_size=int(values["payload_size"]),
                container_time_raw=values.get("container_time"),
                broker_time_raw=values.get("broker_time"),
                evidence=evidence,
            )
        except ValidationError:
            result.unknown_lines.append(
                UnknownLine(evidence=evidence, reason="PUBLISH 字段未通过模型校验")
            )
        else:
            result.events.append(event)


def parse_client_text(text: str, source_file: str, role: ClientRole) -> ParseResult:
    if role not in {"publisher", "subscriber"}:
        raise ValueError("客户端角色必须是 publisher 或 subscriber")
    log = ClientLog.model_validate(json.loads(text))
    result = ParseResult(source_file=source_file, role=role, exit_code=log.exit_code)
    for field in ("stdout", "stderr"):
        _parse_lines(
            getattr(log, field), source_file, role, field, CLIENT_PUBLISH, result
        )
    return result


def parse_broker_text(text: str, source_file: str) -> ParseResult:
    result = ParseResult(source_file=source_file, role="broker")
    _parse_lines(text, source_file, "broker", "text", BROKER_PUBLISH, result)
    return result


def read_client(path: Path, role: ClientRole) -> ParseResult:
    text = path.read_text(encoding="utf-8-sig")
    return parse_client_text(text, str(path), role)


def read_broker(path: Path) -> ParseResult:
    text = path.read_text(encoding="utf-8-sig")
    return parse_broker_text(text, str(path))
