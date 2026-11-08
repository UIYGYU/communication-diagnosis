"""给模型使用的只读教学工具；不运行 MQTT，也不执行日志中的命令。"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from examples.stage2.parser import parse_broker_text, parse_client_text

MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_LOG_LINES = 80
MAX_LINE_CHARS = 500
FILES = {
    "broker": "broker.log",
    "publisher": "publisher.json",
    "subscriber": "subscriber.json",
}

SOURCE_URL = "https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html"


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadLogArguments(Arguments):
    role: Literal["broker", "publisher", "subscriber"]


class SearchArguments(Arguments):
    query: str = Field(min_length=1, max_length=256)


TOOLS = [
    {
        "type": "function",
        "name": "read_log",
        "description": "读取本案例指定角色日志，返回原文位置及可识别的 QoS 1 PUBLISH。其他行仍返回，但未结构化解析。",
        "parameters": ReadLogArguments.model_json_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_protocol",
        "description": "检索 MQTT 5.0 教学摘要，支持 PUBACK、PUBLISH、Packet Identifier 等术语。返回的不是正式规范原文。",
        "parameters": SearchArguments.model_json_schema(),
        "strict": True,
    },
]

# 作者编写的教学摘要；正式首版需要换成经核对的规范快照与引用。
CLAUSES = [
    {
        "clause_id": "K-PUBACK",
        "section": "4.3.2",
        "title": "QoS 1 and PUBACK",
        "text": "教学摘要：QoS 1 的 PUBACK 确认发生在一个发送者与接收者之间。发布端收到确认不能代替最终订阅者接收或业务处理证据。",
    },
    {
        "clause_id": "K-IDENTIFIER",
        "section": "2.2.1",
        "title": "Packet Identifier",
        "text": "教学摘要：报文标识符需要结合连接、方向和交互理解。不同连接中的相同编号不能作为跨端消息的唯一关联依据。",
    },
    {
        "clause_id": "K-PUBLISH",
        "section": "3.3; 4.3.2",
        "title": "PUBLISH QoS 1",
        "text": "教学摘要：PUBLISH 携带应用消息，QoS 1 交付涉及 PUBLISH 与 PUBACK。应区分日志中的发送、对端接收和业务处理，未记录不能直接推断未执行。",
    },
]
TERMS = {"确认": "PUBACK", "发布": "PUBLISH", "标识符": "Packet Identifier"}


class CaseTools:
    def __init__(self, case_dir: Path) -> None:
        root = case_dir.resolve()
        if not root.is_dir():
            raise ValueError("案例目录不存在或不是文件夹")
        self.snapshots: dict[str, bytes] = {}
        self.evidence: dict[str, dict] = {}
        self.clauses: dict[str, dict] = {}
        total = 0
        for role, filename in FILES.items():
            path = root / filename
            if not path.exists():
                continue
            if not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError("日志必须是案例目录内的普通文件，不能链接到目录外")
            with path.open("rb") as stream:
                raw = stream.read(MAX_INPUT_BYTES - total + 1)
            total += len(raw)
            if total > MAX_INPUT_BYTES:
                raise ValueError("三个日志的总大小不能超过 10 MiB")
            self.snapshots[role] = raw

    def call(self, name: str, arguments: str) -> dict:
        try:
            if name == "read_log":
                args = ReadLogArguments.model_validate_json(arguments)
                return self._read_log(args.role)
            if name == "search_protocol":
                args = SearchArguments.model_validate_json(arguments)
                return self._search_protocol(args.query)
            return {
                "ok": False,
                "error": "未知工具；仅支持 read_log 和 search_protocol",
            }
        except ValidationError:
            return {
                "ok": False,
                "error": "参数或日志字段不符合要求，请核对工具参数与输入格式",
            }
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {
                "ok": False,
                "error": "日志不是有效的 UTF-8 文本或 JSON 对象",
            }
        except sqlite3.Error:
            return {"ok": False, "error": "协议检索失败，请检查 SQLite 是否支持 FTS5"}

    def _read_log(self, role: str) -> dict:
        if role not in self.snapshots:
            return {
                "ok": False,
                "error": f"本案例未提供 {FILES[role]}，不能推断该端行为",
            }
        raw = self.snapshots[role]
        text = raw.decode("utf-8-sig")
        parsed = (
            parse_broker_text(text, FILES[role])
            if role == "broker"
            else parse_client_text(text, FILES[role], role)
        )
        rows = []
        for event in parsed.events:
            rows.append((event.evidence, event.model_dump(exclude={"evidence"})))
        for unknown in parsed.unknown_lines:
            rows.append((unknown.evidence, None))
        order = {"text": 0, "stdout": 1, "stderr": 2}
        rows.sort(key=lambda row: (order[row[0].field], row[0].line_number))
        records = []
        truncated = len(rows) > MAX_LOG_LINES
        for source, event in rows[:MAX_LOG_LINES]:
            clipped = len(source.raw) > MAX_LINE_CHARS
            if event:
                for key, value in event.items():
                    if isinstance(value, str) and len(value) > MAX_LINE_CHARS:
                        event[key] = value[:MAX_LINE_CHARS]
                        clipped = True
            evidence_id = f"E-{role}-{source.field}-{source.line_number}"
            record = {
                "evidence_id": evidence_id,
                "source_file": FILES[role],
                "field": source.field,
                "line_number": source.line_number,
                "raw": source.raw[:MAX_LINE_CHARS],
                "event": event,
                "unparsed": event is None,
                "truncated": clipped,
            }
            records.append(record)
            self.evidence[evidence_id] = record
            truncated = truncated or clipped
        return {
            "ok": True,
            "role": role,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "exit_code": parsed.exit_code,
            "total_lines": len(rows),
            "parsed_events": len(parsed.events),
            "unknown_lines": len(parsed.unknown_lines),
            "records": records,
            "truncated": truncated,
            "limits": "仅结构化识别指定格式 QoS 1 PUBLISH；最多返回 80 行，每行 500 字符。stdout/stderr 各自有序，不能合成跨流时间线。未见记录不等于未执行。",
        }

    def _search_protocol(self, query: str) -> dict:
        expanded = (
            query
            + " "
            + " ".join(value for term, value in TERMS.items() if term in query)
        )
        words = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", expanded.lower())))[:16]
        if not words:
            return {
                "ok": True,
                "clauses": [],
                "note": "未命中，请使用 PUBACK、PUBLISH 或 Packet Identifier 等术语",
            }
        # 每个词都加引号；用户输入不会直接变成 FTS 查询运算符或 SQL。
        match_query = " OR ".join(f'"{word}"' for word in words)
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE clauses USING fts5(clause_id UNINDEXED, title, text)"
            )
            connection.executemany(
                "INSERT INTO clauses VALUES (?, ?, ?)",
                [(item["clause_id"], item["title"], item["text"]) for item in CLAUSES],
            )
            ids = [
                row[0]
                for row in connection.execute(
                    "SELECT clause_id FROM clauses WHERE clauses MATCH ? ORDER BY rank LIMIT 3",
                    (match_query,),
                )
            ]
        finally:
            connection.close()
        found = []
        for clause_id in ids:
            item = next(row for row in CLAUSES if row["clause_id"] == clause_id)
            clause = dict(
                item,
                protocol="MQTT",
                version="5.0",
                source_url=SOURCE_URL,
                content_kind="tutorial_summary",
            )
            found.append(clause)
            self.clauses[clause_id] = clause
        return {
            "ok": True,
            "clauses": found,
            "note": "仅教学摘要，未命中不代表协议没有相关要求；正式验收需规范原文库",
        }
