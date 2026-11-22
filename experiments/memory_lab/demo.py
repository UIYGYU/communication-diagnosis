"""用临时 SQLite 演示记忆确认、按范围取回和模型输入组装。"""

import json
import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from src.orchestration.context import ContextPolicy, build_model_context, payload_bytes


@dataclass(frozen=True)
class MemoryScope:
    user_id: str
    project_id: str
    protocol: str | None = None
    protocol_version: str | None = None


@contextmanager
def open_database(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    # connection 负责提交或回滚事务，closing 负责关闭连接。
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.row_factory = sqlite3.Row
        yield connection


def initialize_database(db_path: Path) -> None:
    with open_database(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('preference', 'experience')),
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                protocol TEXT,
                protocol_version TEXT,
                content TEXT NOT NULL,
                keywords TEXT NOT NULL,
                source TEXT NOT NULL,
                confirmation TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                CHECK (
                    (kind = 'preference' AND protocol IS NULL AND protocol_version IS NULL)
                    OR
                    (kind = 'experience' AND protocol IS NOT NULL AND protocol_version IS NOT NULL)
                )
            )
            """
        )


def save_confirmed_memory(
    db_path: Path,
    scope: MemoryScope,
    *,
    kind: str,
    content: str,
    keywords: list[str],
    source: dict,
) -> str:
    """演示里的确认动作在调用处发生，这个函数只保存已确认的记录。"""
    memory_id = uuid4().hex
    protocol, version = (
        (None, None)
        if kind == "preference"
        else (scope.protocol, scope.protocol_version)
    )
    with open_database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO memories (
                memory_id, kind, user_id, project_id, protocol, protocol_version,
                content, keywords, source, confirmation, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed', ?)
            """,
            (
                memory_id,
                kind,
                scope.user_id,
                scope.project_id,
                protocol,
                version,
                content,
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(source, ensure_ascii=False),
                datetime.now(UTC).isoformat(),
            ),
        )
    return memory_id


def find_memories(db_path: Path, scope: MemoryScope, question: str) -> list[dict]:
    """先让 SQLite 筛范围，再用问题里的关键词排序。"""
    with open_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND project_id = ?
              AND confirmation = 'confirmed' AND enabled = 1
              AND (kind = 'preference' OR (protocol = ? AND protocol_version = ?))
            ORDER BY updated_at DESC, memory_id
            """,
            (scope.user_id, scope.project_id, scope.protocol, scope.protocol_version),
        ).fetchall()

    preferences = []
    experience_matches = []
    question = question.casefold()
    for row in rows:
        memory = dict(row)
        memory["source"] = json.loads(memory["source"])
        memory["keywords"] = json.loads(memory["keywords"])
        if memory["kind"] == "preference":
            if not preferences:
                preferences.append(memory)
            continue
        score = sum(
            keyword.casefold() in question for keyword in memory["keywords"] if keyword
        )
        if score:
            experience_matches.append((score, memory))

    # 同分时，沿用数据库给出的更新时间顺序。
    experience_matches.sort(key=lambda match: match[0], reverse=True)
    return [*preferences, *(memory for _, memory in experience_matches)][:3]


def update_memory(
    db_path: Path, scope: MemoryScope, memory_id: str, *, content: str, enabled: bool
) -> None:
    """保留 ID 和来源，递增修订号。"""
    with open_database(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE memories
            SET content = ?, enabled = ?, revision = revision + 1, updated_at = ?
            WHERE memory_id = ? AND user_id = ? AND project_id = ?
            """,
            (
                content,
                enabled,
                datetime.now(UTC).isoformat(),
                memory_id,
                scope.user_id,
                scope.project_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("当前用户和项目中找不到要修改的记忆")


def add_memories_to_input(
    base_input: list[dict],
    candidates: list[dict],
    *,
    tools: list[dict],
    instructions: str,
    limit_bytes: int,
) -> tuple[list[dict], list[dict]]:
    """基础输入已经通过预算检查，再逐条尝试加记忆。"""
    model_input = base_input
    selected = []
    for record in candidates:
        snapshot = {
            key: deepcopy(record[key])
            for key in ("memory_id", "kind", "revision", "content", "source")
        }
        trial_items = [*selected, snapshot]
        message = {
            "role": "user",
            "content": json.dumps(
                {"kind": "long_term_memory", "items": trial_items}, ensure_ascii=False
            ),
        }
        trial_input = [message, *base_input]
        if payload_bytes(trial_input, tools, instructions) <= limit_bytes:
            selected = trial_items
            model_input = trial_input
    return model_input, selected


def show(label: str, records: list[dict]) -> None:
    entries = [
        f"{record['kind']} v{record['revision']} {record['content']}"
        for record in records
    ]
    print(label, entries)


def main() -> None:
    scope = MemoryScope("local-user", "mqtt-lab", "MQTT", "5.0")
    question = "订阅端为什么收不到消息？"
    tools = []
    instructions = "偏好用于报告表达，历史经验用于选择检查方向。"
    draft = {
        "content": "订阅端收不到消息时，先核对发布主题和订阅过滤器。",
        "keywords": ["收不到", "订阅", "主题"],
        "source": {
            "case_id": "case-A",
            "run_id": "run-A",
            "evidence_ids": ["E-A-01"],
            "clause_ids": ["K-A-01"],
        },
    }

    with TemporaryDirectory(prefix="memory-lab-") as temporary:
        db_path = Path(temporary) / "memories.sqlite3"
        initialize_database(db_path)
        save_confirmed_memory(
            db_path,
            scope,
            kind="preference",
            content="先给结论，再列证据。",
            keywords=[],
            source={"kind": "user_input"},
        )
        # 新的读库调用会打开新的连接。此时经验草稿仍在内存中。
        show("重新打开数据库，确认前", find_memories(db_path, scope, question))

        # 模拟用户在案例 A 的结果区确认草稿。
        experience_id = save_confirmed_memory(
            db_path, scope, kind="experience", **draft
        )
        candidates = find_memories(db_path, scope, question)
        show("案例 B 在同范围读到", candidates)
        show(
            "换到其他 MQTT 版本",
            find_memories(db_path, replace(scope, protocol_version="3.1.1"), question),
        )

        policy = ContextPolicy()
        base = build_model_context(
            history=[{"role": "user", "content": question}],
            tools=tools,
            instructions=instructions,
            policy=policy,
        )
        model_input, snapshot = add_memories_to_input(
            base.input_items,
            candidates,
            tools=tools,
            instructions=instructions,
            limit_bytes=policy.max_input_bytes,
        )
        show(f"加入模型输入（共 {len(model_input)} 条消息）", snapshot)
        _, skipped_when_full = add_memories_to_input(
            base.input_items,
            candidates,
            tools=tools,
            instructions=instructions,
            limit_bytes=base.record["input_bytes"],
        )
        show("预算只够基础输入时选中", skipped_when_full)

        new_text = "订阅端收不到时，先从两端日志取出主题和过滤器再核对。"
        update_memory(db_path, scope, experience_id, content=new_text, enabled=True)
        show("修正后的下一问", find_memories(db_path, scope, question))
        update_memory(db_path, scope, experience_id, content=new_text, enabled=False)
        show("停用后的下一问", find_memories(db_path, scope, question))
        show("此前请求的记忆快照", snapshot)


if __name__ == "__main__":
    main()
