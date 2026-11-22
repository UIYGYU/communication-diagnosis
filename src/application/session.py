from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.models.request import Transport
from src.memory.store import MemoryScope


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
    memory_scope: MemoryScope = field(
        default_factory=lambda: MemoryScope("local-user", "mqtt-lab", "MQTT")
    )
    memory_db_path: Path = Path("outputs/memory/memories.sqlite3")
