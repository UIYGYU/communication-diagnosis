#共享工具参数、模型工具声明及图使用的异步工具接口
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

class ReadLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role:Literal["broker", "publisher", "subscriber"]

class SearchProtocolArgs(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    query:str = Field(min_length=1, max_length=256)

TOOLS = [
    {
        "type":"function",
        "name":"read_log",
        "description":"读取本次案例中某一角色的日志原文片段，返回物理行号。",
        "parameters":ReadLogArgs.model_json_schema(),
        "strict":True
    },
    {
        "type": "function",
        "name": "search_protocol",
        "description": "检索 MQTT 5.0 教学摘要，支持 PUBACK、PUBLISH、Packet Identifier 等术语；结果不是正式规范原文。",
        "parameters": SearchProtocolArgs.model_json_schema(),
        "strict": True,
    },    
]

ARGUMENT_MODELS={
    "read_log": ReadLogArgs,
    "search_protocol":SearchProtocolArgs
}

class AsyncToolbox(Protocol):
    tools: list[dict]
    transport:str

    async def call(self, name:str, arguments:str) -> dict:...