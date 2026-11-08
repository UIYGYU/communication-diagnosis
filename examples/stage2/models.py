from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["publisher", "subscriber", "broker"]
ClientRole = Literal["publisher", "subscriber"]
Stream = Literal["text", "stdout", "stderr"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClientLog(StrictModel):
    command: list[str] = Field(repr=False, exclude=True)
    exit_code: int
    stdout: str
    stderr: str


class Evidence(StrictModel):
    source_file: str
    role: Role
    field: Stream
    line_number: int = Field(ge=1)
    raw: str


class Event(StrictModel):
    event: Literal["PUBLISH"] = "PUBLISH"
    direction: Literal["send", "receive"]
    client_id: str
    topic: str
    qos: Literal[1] = 1
    packet_id: int = Field(ge=1, le=65535)
    dup: bool
    retain: bool
    payload_size: int = Field(ge=0)
    container_time_raw: str | None = None
    broker_time_raw: str | None = None
    evidence: Evidence


class UnknownLine(StrictModel):
    evidence: Evidence
    reason: str


class ParseResult(StrictModel):
    source_file: str
    role: Role
    exit_code: int | None = None
    events: list[Event] = Field(default_factory=list)
    unknown_lines: list[UnknownLine] = Field(default_factory=list)
