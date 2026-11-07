import sqlite3
import sys
from importlib.metadata import version

def main()->None:
    if sys.version_info[:2] < (3,13):
        raise SystemExit("本教程要求 Python 3.13,请检查 uv 的版本选择。")

    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    for package in ["pydantic", "mcp", "streamlit", "pytest", "ruff"]:
        print(f"{package}: {version(package)}")

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        connection.execute("INSERT INTO probe(text) VALUES (?)", ("MQTT PUBACK",))
        rows = connection.execute("SELECT text FROM probe WHERE probe MATCH ?", ("PUBACK",)).fetchall()
        assert rows == [("MQTT PUBACK",)]
        print(f"SQLite: {sqlite3.sqlite_version}; FTS5: OK")
    finally:
        connection.close()

if __name__ == "__main__":
    main()