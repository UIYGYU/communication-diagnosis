"""应用入口：上传输入 → 独立案例快照 → 同一 Agent → MCP 工具 → 报告。"""

import asyncio
import hashlib
import json
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, Self
from uuid import uuid4

from openai import APIError, OpenAI

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from examples.stage2.models import ClientLog
from examples.stage4.agent_demo import run_agent_async
from examples.stage3.llm_client import api_error_message, create_client
from examples.stage3.tools import FILES, MAX_INPUT_BYTES
from examples.stage4.client import open_toolbox

APP_VERSION = "stage4-integrated-1"
SUPPORTED_SCOPE = [
    "仅结构化解析指定格式的 QoS 1 PUBLISH；其他日志仍作为原文证据。",
    "协议检索使用第三阶段教学摘要，尚未替换为正式规范原文库。",
    "本结果用于整合练习，不表示全部首版需求已验收通过。",
]


class UploadedLog(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["broker", "publisher", "subscriber"]
    file_name: str = Field(min_length=1, max_length=255)
    # 文件内容只在本任务中传递，不出现在 repr 或普通模型导出中。
    content: bytes = Field(repr=False, exclude=True)


class DiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=4000)
    files: list[UploadedLog] = Field(min_length=1, max_length=3)
    max_rounds: int = Field(default=6, ge=1, le=12)
    max_tool_calls: int = Field(default=8, ge=1, le=16)

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能全为空白。")
        return value

    @model_validator(mode="after")
    def check_files(self) -> Self:
        roles = [item.role for item in self.files]
        if len(roles) != len(set(roles)):
            raise ValueError("每种日志角色最多提供一个文件。")
        if sum(len(item.content) for item in self.files) > MAX_INPUT_BYTES:
            raise ValueError("日志合计超过 10 MiB。")
        return self


def input_manifest(request: DiagnosisRequest) -> list[dict]:
    """保留上传名称供用户识别，真正落盘始终使用固定角色文件名。"""
    return [
        {
            "role": item.role,
            "uploaded_name": item.file_name,
            "case_file": FILES[item.role],
            "bytes": len(item.content),
            "sha256": hashlib.sha256(item.content).hexdigest(),
        }
        for item in sorted(request.files, key=lambda value: value.role)
    ]


def request_identity(request: DiagnosisRequest) -> str:
    identity = {
        "app_version": APP_VERSION,
        "question": request.question,
        "files": input_manifest(request),
        "max_rounds": request.max_rounds,
        "max_tool_calls": request.max_tool_calls,
        "protocol_version": "MQTT-5.0",
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def readable_input_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, item['loc'])) or '输入'}: {item['msg']}"
        for item in error.errors(include_input=False)
    )


def validate_log_contents(request: DiagnosisRequest) -> None:
    """先检查原始输入；无效文件不启动模型，也不执行日志中的 command。"""
    for item in request.files:
        try:
            text = item.content.decode("utf-8-sig")
        except UnicodeError as error:
            raise ValueError(f"{item.role}不是有效的 UTF-8 日志。") from error
        if item.role == "broker":
            continue
        try:
            ClientLog.model_validate(json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{item.role} JSON 无法解析：第 {error.lineno} 行，第 {error.colno} 列。"
            ) from error
        except ValidationError as error:
            raise ValueError(
                f"{item.role} 不符合第三阶段客户端日志格式："
                "需要 command 字符串列表、整数 exit_code，以及字符串 stdout、stderr。"
            ) from error


async def analyze_async(
    request: DiagnosisRequest,
    *,
    client_factory: Callable[[], tuple[OpenAI, str]] | None = None,
    toolbox_factory: Callable[[Path], AbstractAsyncContextManager[Any]] | None = None,
) -> dict:
    """工厂参数仅便于离线测试；正常界面使用默认真实模型和 MCP 连接。"""
    task_id = uuid4().hex
    metadata = {
        "task_id": task_id,
        "app_version": APP_VERSION,
        "input_id": request_identity(request),
        "question": request.question,
        "input_manifest": input_manifest(request),
        "supported_scope": list(SUPPORTED_SCOPE),
    }
    result = {
        "mode": "learning_agent",
        "status": "failed",
        "reason": "任务尚未完成。",
        "report": None,
        "trace": [],
        "evidence": {},
        "clauses": {},
        "limits": {
            "max_rounds": request.max_rounds,
            "max_tool_calls": request.max_tool_calls,
        },
    }

    def fail(reason: str) -> dict:
        result.update(status="failed", reason=reason, report=None)
        result["trace"].append({"type": "application_error", "message": reason})
        return {**result, **metadata}

    try:
        validate_log_contents(request)
    except ValueError as error:
        return fail(str(error))

    # 配置验证失败时不启动工具子进程。这里的配置错误来自已审阅的 create_client。
    try:
        client, model = (client_factory or create_client)()
    except ValueError as error:
        return fail(str(error))
    except APIError as error:
        return fail(api_error_message(error))
    try:
        with client:
            with TemporaryDirectory(prefix="communication-case-") as temporary:
                case_dir = Path(temporary)
                for item in request.files:
                    # 不用上传名称构造路径；原始字节及其 SHA-256 保持一致。
                    (case_dir / FILES[item.role]).write_bytes(item.content)
                async with (toolbox_factory or open_toolbox)(case_dir) as toolbox:
                    result = await run_agent_async(
                        client,
                        model,
                        request.question,
                        toolbox,
                        max_rounds=request.max_rounds,
                        max_tool_calls=request.max_tool_calls,
                    )
                    metadata["transport"] = toolbox.transport
                    metadata["model"] = model
        # 退出顺序：MCP 会话和子进程 → 临时目录 → 模型客户端。
        return {**result, **metadata}
    except APIError as error:
        return fail(api_error_message(error))
    except Exception as error:
        # 包括 MCP 上下文可能抛出的 ExceptionGroup；不把原始异常或配置打印到界面。
        return fail(
            f"工具服务或应用执行失败（{type(error).__name__}）。"
            "请检查第四阶段文件、MCP 服务启动及依赖；本次未生成完成报告。"
        )


def analyze(
    request: DiagnosisRequest,
    *,
    client_factory: Callable[[], tuple[OpenAI, str]] | None = None,
    toolbox_factory: Callable[[Path], AbstractAsyncContextManager[Any]] | None = None,
) -> dict:
    """供 Streamlit 按钮调用；不在模块导入、页面展示或下载时自动运行。"""
    return asyncio.run(
        analyze_async(
            request, client_factory=client_factory, toolbox_factory=toolbox_factory
        )
    )
