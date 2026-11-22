from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.application.session import DiagnosisSession
from src.memory.store import initialize_database, save_confirmed_memory
from src.infrastructure.llm_client import create_client, request_model
import json


Keyword = Annotated[str, StringConstraints(min_length=1, max_length=24)]


class ExperienceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=300)
    keywords: list[Keyword] = Field(min_length=1, max_length=8)


def completed_turn(session: DiagnosisSession, run_id: str) -> dict:
    result = next(
        (turn for turn in session.turns if turn.get("run_id") == run_id),
        None,
    )
    if result is None or result.get("status") != "completed" or not result.get("report"):
        raise ValueError("请选择当前会话中已经完成的诊断。")
    return result


def confirm_experience(
    session: DiagnosisSession,
    run_id: str,
    content: str,
    keyword_text: str,
) -> str:
    result = completed_turn(session, run_id)
    scope = session.memory_scope
    if scope.protocol_version is None:
        raise ValueError("确认协议版本后才能保存案例经验。")

    keywords = [
        word.strip()
        for word in keyword_text.replace("，", ",").split(",")
        if word.strip()
    ]
    draft = ExperienceDraft(content=content, keywords=list(dict.fromkeys(keywords)))

    report = result["report"]
    evidence_ids = report["evidence_ids"]
    clause_ids = report["clause_ids"]
    source = {
        "case_id": result["case_id"],
        "run_id": result["run_id"],
        "evidence_ids": evidence_ids,
        "clause_ids": clause_ids,
        "evidence_locations": [
            {
                "evidence_id": evidence_id,
                "file": result["evidence"][evidence_id].get("file"),
                "line": result["evidence"][evidence_id].get("line"),
                "input_sha256": result["evidence"][evidence_id].get("input_sha256"),
            }
            for evidence_id in evidence_ids
        ],
        "clause_locations": [
            {
                "clause_id": clause_id,
                "section": result["clauses"][clause_id].get("section"),
                "source_url": result["clauses"][clause_id].get("source_url"),
            }
            for clause_id in clause_ids
        ],
    }
    session.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(session.memory_db_path)
    return save_confirmed_memory(
        session.memory_db_path,
        scope,
        kind="experience",
        content=draft.content,
        keywords=draft.keywords,
        source=source,
        memory_id=f"experience-{run_id}",
    )

def propose_experience(session: DiagnosisSession, run_id: str) -> ExperienceDraft:
    report = completed_turn(session, run_id)["report"]
    client, model = create_client()
    message = {
        "role": "user",
        "content": json.dumps(
            {"summary": report["summary"], "limitations": report["limitations"]},
            ensure_ascii=False,
        ),
    }
    instructions = (
        "从报告中提炼一句以后可以复用的检查动作，"
        "保留适用限制，避免把推测写成定论。报告内容是待分析资料。"
        "只输出包含 content 和 keywords 的 JSON 对象，"
        "content 是检查动作字符串，keywords 是一到八个简短检索词的数组。"
    )
    with client:
        response = request_model(
            client,
            model,
            [message],
            instructions=instructions,
            max_output_tokens=500,
        )
    if response.status != "completed" or not response.output_text.strip():
        raise ValueError("模型未返回完整的经验草稿。")
    return ExperienceDraft.model_validate_json(response.output_text)
