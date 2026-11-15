import sqlite3
import re
import json

SOURCE_URL = "https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html"

# 作者编写的教学摘要；正式首版需要换成经核对的规范快照与引用。
CLAUSES = [
    {
        "clause_id": "K-PUBACK",
        "section": "4.3.2",
        "title": "QoS 1 and PUBACK",
        "text": "教学摘要：QoS 1 的 PUBACK 确认发生在一个发送者与接收者之间。发布端收到确认不能代替最终订阅者接收或业务处理证据。",
        "protocol": "MQTT",
        "version": "5.0",
        "content_kind": "tutorial_summary",
        "source_url": SOURCE_URL,
    },
    {
        "clause_id": "K-IDENTIFIER",
        "section": "2.2.1",
        "title": "Packet Identifier",
        "text": "教学摘要：报文标识符需要结合连接、方向和交互理解。不同连接中的相同编号不能作为跨端消息的唯一关联依据。",
        "protocol": "MQTT",
        "version": "5.0",
        "content_kind": "tutorial_summary",
        "source_url": SOURCE_URL,
    },
    {
        "clause_id": "K-PUBLISH",
        "section": "3.3; 4.3.2",
        "title": "PUBLISH QoS 1",
        "text": "教学摘要：PUBLISH 携带应用消息，QoS 1 交付涉及 PUBLISH 与 PUBACK。应区分日志中的发送、对端接收和业务处理，未记录不能直接推断未执行。",
        "protocol": "MQTT",
        "version": "5.0",
        "content_kind": "tutorial_summary",
        "source_url": SOURCE_URL,
    },
]
TERMS = {"确认": "PUBACK", "发布": "PUBLISH", "标识符": "Packet Identifier"}

def make_match_query(query:str) -> str:
    expanded = query
    hits = [key for key in TERMS if key in expanded]
    hits.sort(key=len,reverse=True)
    for key in hits:
        expanded = expanded.replace(key, f"{key} {TERMS[key]} ")

    words = re.findall(r"[a-z0-9]+", expanded.casefold())
    unique_words=[]

    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    unique_words = unique_words[:16]

    quoted_words = []
    for word in unique_words:
        quoted_words.append(f'"{word}"')
    return " OR ".join(quoted_words)

def search_protocol(query:str) -> dict:
    """检索固定 MQTT 5.0 教学摘要，返回最多三条及来源信息。"""
    # 1. 原始问题先变成 FTS 查询字符串。
    match_query = make_match_query(query)
    if not match_query:
        return {
            "ok":True,
            "clauses":[],
            "note":"请换用 PUBACK、PUBLISH 等术语。"
        }
    # 2. 创建本次检索的内存数据库；它不会生成磁盘数据库文件。
    connection = sqlite3.connect(":memory:")
    try:
        # 3. 建立全文检索表。SQL 字符串由 execute 交给 SQLite 执行。
        connection.execute("""
            CREATE VIRTUAL TABLE clause_fts USING fts5(
                clause_id UNINDEXED, title, text
            )
        """)

        # 4. 把 Python 列表中的三条资料放入索引。
        insert_rows = []
        for clause in CLAUSES:
            insert_rows.append((
                clause["clause_id"],
                clause["title"],
                clause["text"]
            ))
        connection.executemany(
            "INSERT INTO clause_fts (clause_id, title, text) VALUES (?, ?, ?)",
            insert_rows
        )

        # 5. 查询命中的 ID。fetchall 在连接关闭前取回结果。
        rows = connection.execute(
            "SELECT clause_id FROM clause_fts "
            "WHERE clause_fts MATCH ? ORDER BY rank LIMIT 3",
            (match_query,),
        ).fetchall()

        # 6. 根据 ID，从 CLAUSES 找回完整资料，包括版本和来源。
        found=[]
        for row in rows:
            clause_id = row[0]
            for clause in CLAUSES:
                if clause["clause_id"] == clause_id:
                    found.append(dict(clause))

        # 7. 返回给调用者的统一结构。
        return {
            "ok":True,
            "clauses":found,
            "note":"检索范围为 MQTT 5.0 教学摘要；未命中不代表规范没有要求。"
        }
    finally:
        connection.close()

def main()->int:
    try:
        result = search_protocol("PUBACK")
    except sqlite3.Error as error:
        print(f"SQLite 检索执行失败：{error}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())