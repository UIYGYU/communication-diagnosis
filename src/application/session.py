from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.models.request import Transport


@dataclass
class DiagnosisSession:
    thread_id: str
    case_id: str
    case_dir: Path
    manifest: dict
    transport: Transport
    mcp_url: str
    graph: CompiledStateGraph
    saver: InMemorySaver
    model_id: str | None = None
    busy: bool = False
    usable: bool = True
    turns: list[dict] = field(default_factory=list)
