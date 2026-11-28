import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from mcp.server import MCPServer

from src.diagnosis.pcap import (
    CAPTURE_FILTER,
    CAPTURE_INTERFACE,
    MAX_FRAMES_LIMIT,
    read_pcap_frames,
)
from src.infrastructure.mqtt_lab import ExperimentCancelled, run_case
from src.models.request import LogRole
from src.tools.experiment_contracts import RUNS_ROOT, Scenario
from src.tools.local import LocalTools


@dataclass
class Job:
    scenario: Scenario
    capture: bool = False
    status: str = "running"
    message: str = ""
    manifest: dict | None = None
    capture_file: dict | None = None
    cancel: Event = field(default_factory=Event)
    done: Event = field(default_factory=Event)

    def result(self, run_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "run_id": run_id,
            "scenario": self.scenario,
            "capture": self.capture,
            "status": self.status,
            "message": self.message,
            "cancel_requested": self.cancel.is_set(),
            "manifest": self.manifest,
            "capture_file": self.capture_file,
        }


def create_experiment_server(root: Path = RUNS_ROOT) -> MCPServer:
    server = MCPServer("本地通信实验")
    root = root.resolve()
    jobs: dict[str, Job] = {}
    lock = Lock()

    def valid_id(run_id: str) -> bool:
        return re.fullmatch(r"[0-9a-f]{32}", run_id) is not None

    def missing_job(run_id: str) -> dict[str, Any]:
        if (root / run_id).exists():
            return {
                "ok": True,
                "run_id": run_id,
                "status": "unknown",
                "message": "本次目录已存在，当前服务无法确认上次运行的结果。",
            }
        return {"ok": True, "run_id": run_id, "status": "not_started"}

    def worker(run_id: str, job: Job, case_dir: Path) -> None:
        status, message, manifest, capture_file = "completed", "", None, None
        try:
            run_case(
                case_dir,
                run_id,
                job.scenario,
                capture=job.capture,
                cancel=job.cancel,
            )
            manifest = LocalTools(case_dir).case_manifest()
            if any(not item or not item["bytes"] for item in manifest["files"].values()):
                raise ValueError("本次实验缺少完整的三端日志。")
            if job.capture:
                pcap_path = case_dir / "mqtt.pcapng"
                if not pcap_path.is_file() or pcap_path.stat().st_size == 0:
                    raise ValueError("本次实验开启了抓包，但未生成有效的 mqtt.pcapng。")
                raw = pcap_path.read_bytes()
                capture_file = {
                    "file": pcap_path.name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "interface": CAPTURE_INTERFACE,
                    "capture_filter": CAPTURE_FILTER,
                }
        except ExperimentCancelled as error:
            notes = getattr(error, "__notes__", [])
            status = "failed" if notes else "cancelled"
            message = " / ".join([str(error), *notes])
        except Exception as error:
            status = "failed"
            message = " / ".join([str(error), *getattr(error, "__notes__", [])])

        with lock:
            if status == "completed" and job.cancel.is_set():
                status, message = "cancelled", "已完成收尾。"
            job.status = status
            job.message = message
            job.manifest = manifest if status == "completed" else None
            job.capture_file = capture_file if status == "completed" else None
            job.done.set()

    @server.tool()
    def run_experiment(
        run_id: str, scenario: Scenario, capture: bool = False
    ) -> dict[str, Any]:
        """发起一次受控 MQTT 实验。

        参数说明：
        - run_id：32 位小写十六进制字符（每次新实验须使用全新 ID）。
        - scenario：normal（正常收发）或 topic-mismatch（主题不匹配）。
        - capture：需要网络报文帧证据时设为 true，将在实验 lo 上抓取 TCP 1883。
        返回 running 后，请用同一个 run_id 调用 get_experiment_status（可传 wait_seconds）。
        """
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        with lock:
            existing = jobs.get(run_id)
            if existing is not None:
                if existing.scenario != scenario or existing.capture != capture:
                    return {
                        "ok": False,
                        "run_id": run_id,
                        "message": "同一 ID 的场景或抓包配置不一致。",
                    }
                return existing.result(run_id)
            case_dir = root / run_id
            if case_dir.exists():
                return missing_job(run_id)
            if any(job.status == "running" for job in jobs.values()):
                return {"ok": False, "run_id": run_id, "message": "当前已有实验在运行。"}
            try:
                case_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                return missing_job(run_id)
            except OSError as error:
                return {"ok": False, "run_id": run_id, "message": str(error)}
            job = Job(scenario=scenario, capture=capture)
            jobs[run_id] = job
            try:
                Thread(target=worker, args=(run_id, job, case_dir)).start()
            except RuntimeError as error:
                job.status, job.message = "failed", str(error)
                job.done.set()
            return job.result(run_id)

    @server.tool()
    def get_experiment_status(run_id: str, wait_seconds: int = 10) -> dict[str, Any]:
        """查询同一次实验的状态、日志清单和抓包文件元数据。

        当实验处于 running 时，默认最多等待 10 秒（完成会立刻唤醒返回，避免模型高频空轮询）；
        可传 0 至 12 的整数调整等待时长，传 0 表示不等待立即返回当前状态。
        """
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        if type(wait_seconds) is not int or not 0 <= wait_seconds <= 12:
            return {
                "ok": False,
                "run_id": run_id,
                "message": "wait_seconds 须为 0 至 12 的整数。",
            }
        with lock:
            job = jobs.get(run_id)
            if job is None:
                return missing_job(run_id)
            should_wait = wait_seconds > 0 and job.status == "running"
        if should_wait:
            job.done.wait(timeout=wait_seconds)
        with lock:
            return job.result(run_id)

    @server.tool()
    def cancel_experiment(run_id: str) -> dict[str, Any]:
        """请求取消一次实验，再通过状态工具确认最终结果。"""
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        with lock:
            job = jobs.get(run_id)
            if job is None:
                return missing_job(run_id)
            if job.status == "running":
                job.cancel.set()
            return job.result(run_id)

    @server.tool()
    def read_experiment_log(run_id: str, role: LogRole) -> dict[str, Any]:
        """读取已完成实验的指定角色日志原文（broker/publisher/subscriber），返回带 evidence_id 的行。"""
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        with lock:
            job = jobs.get(run_id)
            if job is None:
                return missing_job(run_id)
            if job.status != "completed":
                return {
                    "ok": False,
                    "run_id": run_id,
                    "status": job.status,
                    "message": f"实验尚未完成（当前状态：{job.status}），不能读取实验日志。",
                }
        try:
            data = LocalTools(root / run_id).read_log(role)
        except (OSError, ValueError) as error:
            return {
                "ok": False,
                "run_id": run_id,
                "message": f"实验日志读取失败：{error}",
            }
        return {"run_id": run_id, **data}

    @server.tool()
    def read_experiment_frames(run_id: str, max_frames: int = 40) -> dict[str, Any]:
        """读取已完成实验的 MQTT 抓包帧字段，返回带帧级 evidence_id（F-...）的列表。"""
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES_LIMIT:
            return {
                "ok": False,
                "run_id": run_id,
                "message": f"max_frames 须为 1 至 {MAX_FRAMES_LIMIT} 的整数。",
            }
        with lock:
            job = jobs.get(run_id)
            if job is None:
                return missing_job(run_id)
            if job.status != "completed":
                return {
                    "ok": False,
                    "run_id": run_id,
                    "status": job.status,
                    "message": f"实验尚未完成（当前状态：{job.status}），不能读取抓包帧。",
                }
            if not job.capture:
                return {
                    "ok": False,
                    "run_id": run_id,
                    "error": "capture_not_enabled",
                    "message": "本次实验未开启抓包，没有可用的报文文件。",
                }
        try:
            data = read_pcap_frames(
                root / run_id / "mqtt.pcapng", max_frames=max_frames
            )
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            return {
                "ok": False,
                "run_id": run_id,
                "message": f"抓包解析失败：{error}",
            }
        return {"run_id": run_id, **data}

    return server


def main() -> None:
    create_experiment_server().run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8001,
        streamable_http_path="/mcp",
    )


if __name__ == "__main__":
    main()
