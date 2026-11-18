import json
from pathlib import Path
from tempfile import TemporaryDirectory


def save_result(result: dict) -> Path:
    output_dir = Path("outputs/runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{result['run_id']}.json"
    content = json.dumps(result, ensure_ascii=False, indent=2)
    output_file.write_text(content, encoding="utf-8")

    return output_file


def write_case_snapshot(
    case_id: str,
    files: dict[str, bytes],
    output_root: Path = Path("outputs/cases"),
) -> Path:
    try:
        base = output_root.resolve()
        base.mkdir(parents=True, exist_ok=True)
        root = base / case_id

        if not root.resolve().is_relative_to(base):
            raise ValueError("案例目录不能指向保存目录外。")

        for file_name in files:
            if (
                file_name in ("", ".", "..")
                or "/" in file_name
                or "\\" in file_name
                or ":" in file_name
            ):
                raise ValueError(f"不允许的文件名：{file_name}")

        if root.exists():
            if not root.is_dir():
                raise ValueError("案例路径已存在，但不是目录。")

            actual_name = {path.name for path in root.iterdir()}
            if actual_name != set(files):
                raise ValueError("已保存案例的文件集合发生了变化。")
            for file_name, content in files.items():
                path = root / file_name

                if not path.resolve().is_relative_to(root):
                    raise ValueError("案例文件不能指向案例目录外。")

                if not path.is_file() or path.read_bytes() != content:
                    raise ValueError(f"已保存的文件内容不一致：{file_name}")
            return root

        with TemporaryDirectory(prefix=".upload-", dir=base) as temporary:
            staged = Path(temporary) / "case"
            staged.mkdir()

            for file_name, content in files.items():
                path = staged / file_name
                path.write_bytes(content)

            for file_name, content in files.items():
                path = staged / file_name
                if path.read_bytes() != content:
                    raise ValueError(f"文件写入后核对失败：{file_name}")

            staged.rename(root)

        return root
    except OSError as error:
        raise ValueError(f"案例文件保存或读取失败：{error}") from error
