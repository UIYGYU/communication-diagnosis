"""运行一次独立的 MQTT 实验并保存案例日志。"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from threading import Event


IMAGE = (
    "eclipse-mosquitto@sha256:"
    "860aeb5d63ce509508f6dcfb2bc8a1c1dd2bc60555b562b45ce90e52d8dabb0f"
)
DOCKER = ["docker", "--context", "desktop-linux"]
TOPIC = "lab/status"
OTHER_TOPIC = "lab/alarm"
MESSAGE = "communication-diagnosis-v2"
LABEL = "communication-diagnosis.run"
BROKER_CONFIG = """listener 1883 127.0.0.1
allow_anonymous true
persistence false
log_dest stdout
log_type all
log_timestamp true
"""
# Windows 将容器内 ASCII 日志统一作为 UTF-8 读取。
TEXT = {"encoding": "utf-8", "errors": "replace"}

def docker(*args: str, timeout: float = 10) -> subprocess.CompletedProcess[str]:
    """同步执行既定 Docker 参数；业务代码决定怎样处理非零退出码。"""
    return subprocess.run(
        [*DOCKER, *args], capture_output=True, timeout=timeout, **TEXT
    )


def require_zero(result: subprocess.CompletedProcess[str], stage: str) -> None:
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{stage}失败（退出码 {result.returncode}）：{message}")


def broker_log(container: str) -> str:
    result = docker("logs", "--timestamps", container, timeout=5)
    require_zero(result, "读取 broker 日志")
    return result.stdout


def wait_for_log(container: str, pattern: str, *, timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        result = docker("logs", "--timestamps", container, timeout=remaining)
        require_zero(result, "检查 broker 就绪")
        if re.search(pattern, result.stdout, re.MULTILINE):
            return
        time.sleep(min(0.1, remaining))
    raise TimeoutError(f"等待 broker 日志超时：{pattern}")

def write_client_log(
    path: Path,
    command: list[str],
    exit_code: int,
    stdout: str,
    stderr: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "command": command,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def collect_subscriber(
    process: subprocess.Popen[str], output: Path
) -> subprocess.CompletedProcess[str]:
    """等待订阅进程结束，收集这次运行产生的两路日志。"""
    try:
        stdout, stderr = process.communicate(timeout=16)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        write_client_log(output, process.args, process.returncode, stdout, stderr)
        raise TimeoutError("等待订阅端退出超时") from None
    write_client_log(output, process.args, process.returncode, stdout, stderr)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def remove_own_container(container_id: str | None, run_id: str) -> None:
    if container_id is None:
        # docker run 超时时，容器可能已创建，但 CLI 还未返回 ID。
        found = docker("ps", "--all", "--quiet", "--filter", f"label={LABEL}={run_id}")
        require_zero(found, "核对本次容器")
        ids = found.stdout.split()
        if not ids:
            return
        if len(ids) != 1:
            raise RuntimeError("本次运行标记对应多个容器，请核对后清理")
        container_id = ids[0]
    require_zero(docker("rm", "--force", "--volumes", container_id), "清理本次容器")

class ExperimentCancelled(Exception):
    """运行器收到取消请求，正在走原有的清理流程。"""


def check_cancelled(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ExperimentCancelled("已请求取消实验")


def run_case(
    case_dir: Path,
    run_id: str,
    scenario: str,
    *,
    cancel: Event | None = None,
):
    if scenario not in {"normal", "topic-mismatch"}:
        raise ValueError(f"未知实验场景：{scenario}")
    container = f"communication-case-{run_id}"
    config = case_dir / "mosquitto.conf"
    config.write_text(BROKER_CONFIG, encoding="utf-8")
    subscriber_log = case_dir / "subscriber.json"
    sub_topic = OTHER_TOPIC if scenario == "topic-mismatch" else TOPIC
    container_id = None
    sub_process = None
    try:
        check_cancelled(cancel)
        created = docker(
            "run",
            "--detach",
            "--pull",
            "never",
            "--network",
            "none",
            "--name",
            container,
            "--label",
            f"{LABEL}={run_id}",
            "--mount",
            f"type=bind,src={config},dst=/tmp/mosquitto.conf,readonly",
            IMAGE,
            "mosquitto",
            "-c",
            "/tmp/mosquitto.conf",
            timeout=20,
        )
        require_zero(created, "启动本次 broker 容器")
        container_id = created.stdout.strip()
        wait_for_log(container_id, r"mosquitto version [^\r\n]+ running$")
        common = ["-h", "127.0.0.1", "-V", "mqttv5", "-d", "-q", "1"]
        sub_command = [
            *DOCKER,
            "exec",
            container_id,
            "mosquitto_sub",
            *common,
            "-i",
            "lab-sub",
            "-t",
            sub_topic,
            "-C",
            "1",
            "-W",
            "12",
            "-v",
        ]
        check_cancelled(cancel)
        sub_process = subprocess.Popen(
            sub_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **TEXT
        )
        wait_for_log(container_id, r"Sending SUBACK to lab-sub$")
        check_cancelled(cancel)
        pub = docker(
            "exec",
            container_id,
            "mosquitto_pub",
            *common,
            "-i",
            "lab-pub",
            "-t",
            TOPIC,
            "-m",
            MESSAGE,
            timeout=5,
        )
        check_cancelled(cancel)
        write_client_log(
            case_dir / "publisher.json",
            pub.args,
            pub.returncode,
            pub.stdout,
            pub.stderr,
        )
        require_zero(pub, "发布消息")
        sub = collect_subscriber(sub_process, subscriber_log)
        check_cancelled(cancel)
        received = f"{TOPIC} {MESSAGE}" in sub.stdout.splitlines()
        if scenario == "normal":
            require_zero(sub, "订阅消息")
            if not received:
                raise RuntimeError("订阅日志中未找到本次发送的主题与内容")
        elif sub.returncode != 27 or received:
            raise RuntimeError("主题不匹配场景未观察到预期的订阅超时")
    finally:
        original_error = sys.exception()
        try:
            try:
                if sub_process is not None and not subscriber_log.exists():
                    if sub_process.poll() is None:
                        sub_process.kill()
                    stdout, stderr = sub_process.communicate()
                    write_client_log(
                        subscriber_log,
                        sub_process.args,
                        sub_process.returncode,
                        stdout,
                        stderr,
                    )
                if container_id is not None:
                    (case_dir / "broker.log").write_text(
                        broker_log(container_id), encoding="utf-8"
                    )
            finally:
                remove_own_container(container_id, run_id)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(f"收尾时也遇到问题：{cleanup_error}")

def main() -> int:
    parser = argparse.ArgumentParser(description="生成一个本地 MQTT 日志案例")
    parser.add_argument(
        "scenario", nargs="?", choices=("normal", "topic-mismatch"), default="normal"
    )
    args = parser.parse_args()
    run_id = uuid4().hex
    case_dir = (Path("outputs") / "mqtt-lab" / run_id).resolve()
    case_dir.mkdir(parents=True)
    print(f"本次实验目录：{case_dir}")
    record = {
        "run_id": run_id,
        "scenario": args.scenario,
        "image": IMAGE,
        "mqtt_version": "5.0",
        "started_at": datetime.now(UTC).isoformat(),
        "publish_topic": TOPIC,
        "subscribe_topic": OTHER_TOPIC if args.scenario == "topic-mismatch" else TOPIC,
        "status": "incomplete",
    }
    exit_code = 1
    try:
        run_case(case_dir, run_id, args.scenario)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        record["notes"] = getattr(error, "__notes__", [])
        print(f"实验未完成：{error}", file=sys.stderr)
        for note in record["notes"]:
            print(note, file=sys.stderr)
    except KeyboardInterrupt as error:
        record["status"] = "cancelled"
        record["error"] = "用户中断"
        record["notes"] = getattr(error, "__notes__", [])
        print("实验已中断，详情见 run.json。", file=sys.stderr)
        exit_code = 130
    else:
        record["status"] = "completed"
        exit_code = 0
        print(f"场景 {args.scenario} 已观察到预期结果，三个日志文件已保存。")
    finally:
        record["ended_at"] = datetime.now(UTC).isoformat()
        (case_dir / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
