"""使用只读 TShark 容器解码 MQTT 抓包文件并生成帧级证据。"""

import hashlib
import subprocess
from pathlib import Path

CAPTURE_IMAGE = "communication-diagnosis-capture:local"
DOCKER = ["docker", "--context", "desktop-linux"]
MAX_PCAP_BYTES = 2 * 1024 * 1024
MAX_FRAMES_LIMIT = 80
CAPTURE_INTERFACE = "lo"
CAPTURE_FILTER = "tcp port 1883"

MQTT_MSGTYPES = {
    "1": "CONNECT",
    "2": "CONNACK",
    "3": "PUBLISH",
    "4": "PUBACK",
    "5": "PUBREC",
    "6": "PUBREL",
    "7": "PUBCOMP",
    "8": "SUBSCRIBE",
    "9": "SUBACK",
    "10": "UNSUBSCRIBE",
    "11": "UNSUBACK",
    "12": "PINGREQ",
    "13": "PINGRESP",
    "14": "DISCONNECT",
    "15": "AUTH",
}


def _describe_msgtype(raw: str) -> str:
    if not raw:
        return ""
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    return ",".join(MQTT_MSGTYPES.get(code, f"UNKNOWN({code})") for code in parts)


def read_pcap_frames(pcap_path: Path, *, max_frames: int = 40) -> dict:
    """读取并解码单个 PCAPNG 文件中的 MQTT 帧，返回可引用的结构化证据。"""
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES_LIMIT:
        raise ValueError(f"max_frames 须在 1 到 {MAX_FRAMES_LIMIT} 之间。")
    path = pcap_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"抓包文件不存在：{path.name}")

    with path.open("rb") as stream:
        raw = stream.read(MAX_PCAP_BYTES + 1)
    if not raw:
        raise ValueError("抓包文件为空。")
    if len(raw) > MAX_PCAP_BYTES:
        raise ValueError("抓包文件超过 2 MiB 上限。")

    digest = hashlib.sha256(raw).hexdigest()
    command = [
        *DOCKER,
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--mount",
        f"type=bind,src={path.parent},dst=/capture,readonly",
        CAPTURE_IMAGE,
        "tshark",
        "-n",
        "-r",
        f"/capture/{path.name}",
        "-Y",
        "mqtt",
        "-T",
        "fields",
        "-E",
        "header=y",
        "-e",
        "frame.number",
        "-e",
        "frame.time_epoch",
        "-e",
        "tcp.stream",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "mqtt.msgtype",
        "-e",
        "mqtt.qos",
        "-e",
        "mqtt.topic",
        "-e",
        "mqtt.msgid",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        timeout=12,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"TShark 解码失败（退出码 {result.returncode}）：{detail}")

    rows = [line for line in result.stdout.splitlines() if line.strip()]
    data_rows = rows[1:] if rows and rows[0].startswith("frame.number") else rows

    frames: list[dict] = []
    truncated = len(data_rows) > max_frames
    for row in data_rows[:max_frames]:
        cols = row.split("\t")
        while len(cols) < 9:
            cols.append("")
        frame_no = int(cols[0])
        frames.append(
            {
                "evidence_id": f"F-{digest[:12]}-N{frame_no}",
                "frame_number": frame_no,
                "time_epoch": cols[1],
                "tcp_stream": cols[2],
                "src_port": cols[3],
                "dst_port": cols[4],
                "msgtype": cols[5],
                "msgtype_name": _describe_msgtype(cols[5]),
                "qos": cols[6],
                "topic": cols[7],
                "msgid": cols[8],
            }
        )

    payload = {
        "ok": True,
        "file": path.name,
        "input_sha256": digest,
        "interface": CAPTURE_INTERFACE,
        "capture_filter": CAPTURE_FILTER,
        "frames": frames,
        "total_frames": len(data_rows),
        "truncated": truncated,
    }
    if not frames:
        payload["note"] = "未在抓包文件中解码到 MQTT 报文，请检查捕获窗口、过滤器或是否加密。"
    return payload
