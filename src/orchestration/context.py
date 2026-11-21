from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any


class ContextBudgetError(ValueError):
    """完整输入在当前策略下无法放入预算。"""

    def __init__(self, message: str, *, can_summarize: bool = False):
        super().__init__(message)
        self.can_summarize = can_summarize


@dataclass(frozen=True)
class ContextPolicy:
    max_input_bytes: int = 128_000
    keep_recent_turns: int = 2
    max_material_bytes: int = 16_000
    max_summary_input_bytes: int = 96_000
    max_summary_bytes: int = 8_000
    max_summary_output_tokens: int = 1_200
    max_summary_calls: int = 1


@dataclass(frozen=True)
class TurnSpan:
    number: int
    start: int
    end: int


@dataclass
class PreparedContext:
    input_items: list[dict]
    record: dict
    visible_evidence_ids: list[str]
    visible_clause_ids: list[str]


def payload_bytes(
    input_items: list[dict],
    tools: list[dict] | None,
    instructions: str,
):
    result: dict[str, Any] = {}
    result["input"] = input_items
    result["instructions"] = instructions
    if tools is not None:
        result["tools"] = tools
    return len(json.dumps(result, ensure_ascii=False).encode("utf-8"))


def split_turns(history: list[dict]) -> list[TurnSpan]:
    if not history:
        return []
    starts = [index for index, item in enumerate(history) if item.get("role") == "user"]
    if not starts or starts[0] != 0:
        raise ValueError("history 必须从用户问题开始")
    ends = starts[1:] + [len(history)]
    spans = [
        TurnSpan(number=number, start=start, end=end)
        for number, (start, end) in enumerate(zip(starts, ends), start=1)
    ]
    seen_ids: set[str] = set()
    for span in spans:
        pending_ids: set[str] = set()

        for item in history[span.start : span.end]:
            kind = item.get("type")
            if kind not in ("function_call", "function_call_output"):
                continue

            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f"第 {span.number} 轮有无效的 call_id")

            if kind == "function_call":
                if call_id in seen_ids:
                    raise ValueError(f"工具请求 ID 重复：{call_id}")
                seen_ids.add(call_id)
                pending_ids.add(call_id)

            if kind == "function_call_output":
                if call_id not in pending_ids:
                    raise ValueError(
                        f"第 {span.number} 轮有未配对或重复的结果：{call_id}"
                    )
                pending_ids.remove(call_id)

        if pending_ids:
            missing = ",".join(sorted(pending_ids))
            raise ValueError(f"第 {span.number} 轮缺少工具结果：{missing}")
    return spans


def select_materials(
    question: str,
    evidence: dict[str, dict],
    clauses: dict[str, dict],
    *,
    max_bytes: int,
    fits_input: Callable[[dict], bool],
) -> dict:
    question = question.casefold()
    mentioned_ids = set(re.findall(r"[a-z0-9_-]+", question))
    # C3a：先找点名的条目。
    candidates = [
        (kind, item_id, item, "explicit_id")
        for kind, items in (("evidence", evidence), ("clauses", clauses))
        for item_id, item in items.items()
        if item_id.casefold() in mentioned_ids
    ]

    if not candidates:
        roles = {
            role
            for word, role in (
                ("发布", "publisher"),
                ("订阅", "subscriber"),
                ("服务端", "broker"),
            )
            if word in question
        }

        recent_logs = list(reversed(list(evidence.items())))
        matching_logs = [
            (item_id, item)
            for item_id, item in recent_logs
            if item.get("role") in roles
        ]
        logs = matching_logs or recent_logs
        reason = "role" if matching_logs else "recent"

        for item_id, item in logs[:6]:
            candidates.append(("evidence", item_id, item, reason))
        for item_id, item in list(reversed(list(clauses.items())))[:3]:
            candidates.append(("clauses", item_id, item, "recent"))

    # C3b：每次只试放一条。
    selected = {"evidence": {}, "clauses": {}, "selection": []}

    for kind, item_id, item, reason in candidates:
        trial = deepcopy(selected)
        trial[kind][item_id] = deepcopy(item)

        originals = {"evidence": trial["evidence"], "clauses": trial["clauses"]}
        size = len(json.dumps(originals, ensure_ascii=False).encode("utf-8"))

        if size <= max_bytes and fits_input(trial):
            trial["selection"].append({"id": item_id, "reason": reason})
            selected = trial
        elif reason == "explicit_id":
            raise ContextBudgetError(f"点名材料 {item_id} 超过预算")

    return selected


def source_ids(history_suffix: list[dict]) -> tuple[set[str], set[str]]:
    tool_names = {}
    for item in history_suffix:
        if item.get("type") == "function_call":
            tool_names[item["call_id"]] = item["name"]

    evidence_ids = set()
    clause_ids = set()

    for item in history_suffix:
        if item.get("type") != "function_call_output":
            continue

        name = tool_names.get(item["call_id"])
        if name not in ("read_log", "search_protocol"):
            continue

        result = json.loads(item["output"])
        if result.get("ok") is not True:
            continue

        if name == "read_log":
            for line in result["lines"]:
                evidence_ids.add(line["evidence_id"])
        else:
            for clause in result["clauses"]:
                clause_ids.add(clause["clause_id"])

    return evidence_ids, clause_ids


def build_model_context(
    *,
    history: list[dict],
    tools: list[dict],
    instructions: str,
    policy: ContextPolicy,
    summary: dict | None = None,
) -> PreparedContext:
    spans = split_turns(history)
    if not spans or policy.keep_recent_turns < 1:
        raise ValueError("至少需要一个用户轮次，保留轮数至少为 1")

    recent_start = spans[-policy.keep_recent_turns :][0].start
    start = 0 if summary is None else summary["covered_until"]
    input_items = deepcopy(history[start:])
    if summary is not None:
        text = json.dumps({"summary": summary["content"]}, ensure_ascii=False)
        input_items.insert(0, {"role": "user", "content": text})

    limit = policy.max_input_bytes
    input_bytes = payload_bytes(input_items, tools, instructions)
    if input_bytes > limit:
        can_summarize = start < recent_start and (
            payload_bytes(history[recent_start:], tools, instructions) <= limit
        )
        raise ContextBudgetError(
            f"上下文共 {input_bytes} 字节，超过上限 {limit}",
            can_summarize=can_summarize,
        )

    evidence_ids, clause_ids = source_ids(input_items)
    return PreparedContext(
        input_items=input_items,
        record={
            "kind": "context",
            "input_bytes": input_bytes,
            "limit_bytes": limit,
            "history_range": [start, len(history)],
            "summary_until": start,
        },
        visible_clause_ids=sorted(clause_ids),
        visible_evidence_ids=sorted(evidence_ids),
    )
