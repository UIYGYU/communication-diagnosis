from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_LOG_BYTES = 1024* 1024
Transport = Literal["local", "mcp-stdio", "mcp-http"]
LogRole = Literal["broker", "publisher", "subscriber"]

class CaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_dir: Path
    transport: Transport = "mcp-http"
    mcp_url:str = "http://127.0.0.1:8000/mcp"

class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    max_rounds: int = Field(default=6, ge=1,le=12)
    max_tool_calls:int = Field(default=8, ge=1,le=16)

class UploadedLog(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role:LogRole
    file_name: str = Field(min_length=1, max_length=255)
    content: bytes = Field(max_length=MAX_LOG_BYTES, repr=False, exclude=True)

class CaseUpload(BaseModel):
    model_config=ConfigDict(extra="forbid", strict = True)

    files: list[UploadedLog] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_roles(self) -> Self:
        roles = [item.role for item in self.files]
        if len(roles) != len(set(roles)):
            raise ValueError("每种日志角色最多提供一个文件。")
        return self