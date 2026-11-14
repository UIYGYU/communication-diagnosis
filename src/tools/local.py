import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel,ConfigDict, ValidationError

class ReadLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["broker", "publisher", "subscriber"]

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

        text = raw.decode("utf-8-sig")

        all_lines = text.splitlines()

        max_lines = 80
        max_chars = 6000
        lines=[]
        used_chars = 0
        truncated = False
        for number, line in enumerate(all_lines, 1):
            if number > max_lines:
                truncated = True
                break
            remaing = max_chars - used_chars
            used_chars += len(line)

            if remaing < len(line):
                if remaing > 0:
                    lines.append({"line": number, "text": line[:remaing]})
                truncated = True
                break
            lines.append({"line": number, "text": line})
        return{
            "ok": True,
            "role": role,
            "file": filename,
            "lines":lines,
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
        else:
            return {"ok": False, "error":"unknown_tool", "message":"未登记的工具名"}

# if __name__=="__main__":
#     localtools = LocalTools(Path("samples/case-01"))
#     print(
#         localtools.call("read_log", json.dumps({"role": "broker"}))
#     )