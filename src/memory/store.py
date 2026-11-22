import json
import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

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
    memory_id: str | None = None
) -> str:
    """演示里的确认动作在调用处发生，这个函数只保存已确认的记录。"""
    memory_id = memory_id or uuid4().hex
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
            ON CONFLICT(memory_id) DO NOTHING
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