import json
from pathlib import Path
from typing import Literal
import hashlib
import sqlite3

from pydantic import BaseModel, ConfigDict, ValidationError, Field

from src.knowledge.protocol import search_protocol

class ReadLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["broker", "publisher", "subscriber"]

class SearchProtocolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=256)

class LocalTools:
    def __init__(self, case_dir:Path):
        """绑定案例根目录，并建立工具声明。"""
        root = case_dir.resolve()
        if not root.is_dir():
            raise ValueError(f"案例目录不存在或不是文件夹：{root}")
        self.root = root
        self.tools = [
            {
                "type":"function",
                "name":"read_log",
                "description":"读取本次案例中某一角色的日志原文片段，返回物理行号。",
                "parameters":ReadLogArgs.model_json_schema(),
                "strict":True,
            },
            {
                "type":"function",
                "name":"search_protocol",
                "description":"检索 MQTT 5.0 教学摘要，支持 PUBACK、PUBLISH、Packet Identifier 等术语；结果不是正式规范原文。",
                "parameters":SearchProtocolArgs.model_json_schema(),
                "strict":True,
            }
        ]

    def read_log(self, role:str) -> dict:
        """读取固定角色的限量原文，返回文件名和物理行号。"""
        filenames = {
            "broker":"broker.log",
            "publisher":"publisher.json",
            "subscriber":"subscriber.json",
        }
        if role not in filenames:
            raise ValueError("不支持的日志角色")
        filename = filenames[role]
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError(f"目标路径无权限:{filename}")

        max_bytes = 1024 * 1024
        with path.open("rb") as file:
            raw = file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("日志文件超过 1 MiB")

        digest = hashlib.sha256(raw).hexdigest()

        text = raw.decode("utf-8-sig")

        all_lines = text.splitlines()

        max_lines = 80
        max_chars = 6000
        lines=[]
        used_chars = 0
        truncated = False
        for number, line in enumerate(all_lines, 1):
            evidence_id = f"E-{role}-{digest[:12]}-L{number}"
            if number > max_lines:
                truncated = True
                break
            remaing = max_chars - used_chars
            used_chars += len(line)

            if remaing < len(line):
                if remaing > 0:
                    lines.append({"line": number, "text": line[:remaing], "evidence_id": evidence_id})
                truncated = True
                break
            lines.append({"line": number, "text": line, "evidence_id": evidence_id})
        return{
            "ok": True,
            "role": role,
            "file": filename,
            "lines":lines,
            "input_sha256":digest,
            "total_lines":len(all_lines),
            "truncated":truncated
        }

    def call(self, name:str, arguments:str)->dict:
        """校验工具名和 JSON 参数，执行或返回预期错误。"""
        if any(tool["name"] == name for tool in self.tools):
            if name=="read_log":
                try:
                    args = ReadLogArgs.model_validate_json(arguments)
                    return self.read_log(args.role)
                except ValidationError:
                    return {
                        "ok": False, 
                        "error":"invalid_arguments", 
                        "message":"JSON 无效、参数不符或带额外字段"
                    }
                except (OSError, ValueError) as error:
                    return{
                        "ok": False, 
                        "error":"read_failed", 
                        "message":f"日志读取失败: {error}"
                    }
            if name=="search_protocol":
                try:
                    args = SearchProtocolArgs.model_validate_json(arguments)
                    return search_protocol(args.query)
                except sqlite3.Error as error:
                    return {
                        "ok":False,
                        "error":"search_failed",
                        "message":f"协议检索失败：{error}"
                    }
                except ValidationError:
                    return {
                        "ok":False,
                        "error":"invalid_arguments",
                        "message":"JSON 无效、参数不符或带额外字段"
                    }
        else:
            return {"ok": False, "error":"unknown_tool", "message":"未登记的工具名"}

# if __name__=="__main__":
#     localtools = LocalTools(Path("samples/case-01"))
#     print(
#         localtools.call("read_log", json.dumps({"role": "broker"}))
#     )
