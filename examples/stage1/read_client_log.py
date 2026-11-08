import argparse
import hashlib
import json
from pathlib import Path


def inspect_client(path: Path) -> dict:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"文件{path}为空，没有可读取的客户端记录")
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"文件{path}不是有效的客户端记录")

    for field in ("stdout", "stderr"):
        if field not in data or not isinstance(data[field], str):
            raise ValueError(f"{field} 必须存在且为字符串")
    if type(data.get("exit_code")) is not int:
        raise ValueError("exit_code 必须存在且为整数")

    streams = {}
    for field in {"stdout", "stderr"}:
        streams[field] = [
            {"line": line, "text": text}
            for line, text in enumerate(data[field].splitlines(), start=1)
        ]

    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "exit_code": data["exit_code"],
        "streams": streams,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="查看一份客户端日志")
    parser.add_argument("input", type=Path, help="客户端日志文件路径")
    parser.add_argument(
        "--output",
        type=Path,
        help="输出文件路径",
        default=Path("outputs/input-summary.json"),
    )

    args = parser.parse_args()
    try:
        summary = inspect_client(args.input)
        print(f"文件: {summary['file']}")
        print(f"退出码: {summary['exit_code']}")
        for field, lines in summary["streams"].items():
            print(f"{field}: {len(lines)} 行")
            for line in lines:
                print(f"  {field}:{line['line']} {line['text']}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"摘要已写入: {args.output}")
    except (OSError, UnicodeError, ValueError) as e:
        print(f"读取失败: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
