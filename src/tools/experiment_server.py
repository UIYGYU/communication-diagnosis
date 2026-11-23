import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from mcp.server import MCPServer

from src.infrastructure.mqtt_lab import ExperimentCancelled, run_case
from src.tools.experiment_contracts import RUNS_ROOT, Scenario
from src.tools.local import LocalTools

@dataclass
class Job:
    scenario:Scenario
    status: str = "running"
    message: str = ""
    manifest: dict | None = None
    cancel: Event = field(default_factory=Event)

    def result(self, run_id: str) -> dict[str, Any]:
        return{
            "ok":True,
            "run_id":run_id,
            "scenario": self.scenario,
            "status": self.status,
            "message": self.message,
            "cancel_requested":self.cancel.is_set(),
            "manifest":self.manifest,
        }

def create_experiment_server(root: Path = RUNS_ROOT) -> MCPServer:
    server = MCPServer("本地通信实验")
    root = root.resolve()
    jobs: dict[str, Job] = {}
    lock = Lock()

    def valid_id(run_id:str)->bool:
        return re.fullmatch(r"[0-9a-f]{32}", run_id) is not None

    def missing_job(run_id:str) -> dict[str, Any]:
        if(root/run_id).exists():
            return {
                "ok": True, "run_id": run_id, "status": "unknown",
                "message": "本次目录已存在，当前服务无法确认上次运行的结果。",
            }
        return {"ok": True, "run_id":run_id, "status":"not_started"}

    def worker(run_id:str, job:Job, case_dir:Path) -> None:
        status, message, manifest = "completed", "", None
        try:
            run_case(case_dir, run_id, job.scenario, cancel=job.cancel)
            manifest = LocalTools(case_dir).case_manifest()
            if any(not item or not item["bytes"] for item in manifest["files"].values()):
                raise ValueError("本次实验缺少完整的三端日志。")
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

    @server.tool()
    def run_experiment(run_id: str, scenario: Scenario) -> dict[str, Any]:
        """发起一个已登记的实验，返回当前状态。"""
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        with lock:
            existing = jobs.get(run_id)
            if existing is not None:
                if existing.scenario != scenario:
                    return {"ok": False, "run_id": run_id, "message": "同一 ID 的场景不一致。"}
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
            job = Job(scenario=scenario)
            jobs[run_id] = job
            try:
                Thread(target=worker, args=(run_id, job, case_dir)).start()
            except RuntimeError as error:
                job.status, job.message = "failed", str(error)
            return job.result(run_id)

    @server.tool()
    def get_experiment_status(run_id: str) -> dict[str, Any]:
        """查询同一次实验的状态及完成后的日志清单。"""
        if not valid_id(run_id):
            return {"ok": False, "run_id": run_id, "message": "实验 ID 无效。"}
        with lock:
            job = jobs.get(run_id)
            return job.result(run_id) if job is not None else missing_job(run_id)

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