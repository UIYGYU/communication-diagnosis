import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from examples.stage2.models import ParseResult
from examples.stage2.parser import (
    parse_broker_text,
    parse_client_text,
    read_broker,
    read_client,
)


def show(result: ParseResult) -> None:
    print(
        f"{Path(result.source_file).name}: events={len(result.events)}, "
        f"unknown={len(result.unknown_lines)}, exit_code={result.exit_code}"
    )
    for event in result.events:
        evidence = event.evidence
        print(
            f"  PUBLISH {event.direction} client={event.client_id} "
            f"qos={event.qos} mid={event.packet_id} topic={event.topic} "
            f"@ {evidence.field}:{evidence.line_number}"
        )


def run_demo() -> None:
    body = "(d0, q1, r0, m1, 'lab/demo/status', ... (20 bytes))"
    client = {
        "command": ["example-command-not-to-run"],
        "exit_code": 0,
        "stdout": f"Client demo-pub sending PUBLISH {body}\nnew format line\n",
        "stderr": "",
    }
    broker = (
        "2026-11-01T15:41:27.852587635Z 1793547687: "
        f"Received PUBLISH from demo-pub {body}\nnew format line\n"
    )
    show(parse_client_text(json.dumps(client), "publisher.json", "publisher"))
    show(parse_broker_text(broker, "broker.log"))


def replay(case_dir: Path) -> None:
    selected = (
        ("broker.log", "broker"),
        ("publisher.json", "publisher"),
        ("subscriber.json", "subscriber"),
    )

    for filename, role in selected:
        path = case_dir / filename
        if not path.is_file():
            print(f"缺失输入：{filename}")
            continue
        if role == "broker":
            result = read_broker(path)
        else:
            result = read_client(path, role)
        show(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.case_dir is None:
            run_demo()
        else:
            replay(args.case_dir)
    except ValidationError as exc:
        print("输入结构校验失败：")
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(item) for item in error["loc"])
            print(f"  {location or '根对象'}: {error['msg']}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"JSON 语法错误：第 {exc.lineno} 行，第 {exc.colno} 列")
        return 1
    except UnicodeError:
        print("输入编码不合法：本练习要求 UTF-8 或 UTF-8 BOM")
        return 1
    except OSError as exc:
        print(f"无法读取指定文件：{exc.__class__.__name__}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
