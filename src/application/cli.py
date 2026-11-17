import argparse
import json
from pathlib import Path

from src.application.service import analyze, create_session
from src.models.request import CaseConfig, TurnRequest

def main() -> int:
    parser = argparse.ArgumentParser(description="同案例连续诊断")
    parser.add_argument("--case-dir", type=Path, default=Path("samples/case-01"))
    parser.add_argument(
        "--transport", choices=("local", "mcp-stdio", "mcp-http"),
        default="mcp-http"
    )
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8000/mcp")
    args = parser.parse_args()
    try:
        session = create_session(CaseConfig(case_dir=args.case_dir, transport=args.transport, mcp_url=args.mcp_url))
    except ValueError:
        print("创建会话失败")
        return 1
    while True:
        question = input("问题（空行退出）：").strip()
        if not question:
            return 0
        try:
            request = TurnRequest(question=question)
        except ValueError as error:
            print(f"问题输入有误：{error}")
            continue
        try:
            result = analyze(session, request)
        except ValueError as error:
            print(f"无法执行诊断：{error}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

        if not session.usable:
            print("当前会话已失效，请重新运行程序。")
            return 1

if __name__=="__main__":
    raise SystemExit(main())