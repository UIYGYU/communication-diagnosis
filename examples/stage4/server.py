"""MCP 服务：每次任务启动一个进程，复用第三阶段的两项工具。"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import Field

from examples.stage3.tools import CaseTools, ReadLogArguments, SearchArguments


def create_server(case_dir: Path) -> MCPServer:
    # 只在启动时选择案例。模型的工具参数中没有任意文件路径。
    toolbox = CaseTools(case_dir)
    server = MCPServer("通信诊断助手工具", version="0.1.0")

    @server.tool()
    def read_log(role: Literal["broker", "publisher", "subscriber"]) -> dict[str, Any]:
        """读取本案例指定角色日志，返回已识别事件、未知行与原文位置。"""
        arguments = ReadLogArguments(role=role)
        return toolbox.call("read_log", arguments.model_dump_json())

    @server.tool()
    def search_protocol(
        query: Annotated[str, Field(min_length=1, max_length=256)],
    ) -> dict[str, Any]:
        """检索 MQTT 5 教学摘要；返回内容不是正式规范原文。"""
        arguments = SearchArguments(query=query)
        return toolbox.call("search_protocol", arguments.model_dump_json())

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="供本项目客户端启动的 MCP 工具服务")
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    # stdio 的 stdout 只留给 MCP 协议消息。
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        create_server(args.case_dir).run(transport="stdio")
    except (OSError, ValueError):
        print("MCP 服务启动失败，请检查案例目录及日志大小。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
