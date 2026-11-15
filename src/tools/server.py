import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import Field

from src.tools.contracts import ReadLogArgs,SearchProtocolArgs
from src.tools.local import LocalTools

def create_server(case_dir:Path) -> MCPServer:
    core = LocalTools(case_dir)
    server = MCPServer("通信诊断助手工具", version="0.2.0")

    @server.resource("diagnosis://case-manifest", mime_type="application/json")
    def case_manifest() -> str:
        """描述当前服务绑定的案例输入，供客户端核对。"""
        return json.dumps(core.case_manifest(), ensure_ascii=False)

    @server.tool()
    def read_log(role:Literal["broker", "publisher", "subscriber"]) -> dict[str,Any]:
        args = ReadLogArgs(role=role)
        return core.call("read_log", args.model_dump_json())

    @server.tool()
    def search_protocol(query:Annotated[str, Field(min_length=1, max_length=256)]) -> dict[str, Any]:
        args = SearchProtocolArgs(query=query)
        return core.call("search_protocol", args.model_dump_json())

    return server

def main() -> int:
    parser = argparse.ArgumentParser(description="服务器启动")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        server = create_server(args.case_dir)
        if args.transport=="http":
            server.run(transport="streamable-http", host="127.0.0.1", port=args.port, streamable_http_path="/mcp")
            return 0
        else:
            server.run(transport="stdio")
            return 0
    except (OSError, ValueError) as error:
        print(f"服务启动失败：{error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())