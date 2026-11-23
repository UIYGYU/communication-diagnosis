from pathlib import Path
from typing import Literal

Scenario = Literal["normal", "topic-mismatch"]
RUNS_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "active-experiments"