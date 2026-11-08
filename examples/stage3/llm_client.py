"""第一步：从本机环境读取配置，向兼容 Responses 的代理发出真实请求。"""

import os
from urllib.parse import urlsplit

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI


def create_client() -> tuple[OpenAI, str]:
    names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    config = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise ValueError("缺少本机配置：" + ", ".join(missing))
    if any(value.startswith("填写") for value in config.values()):
        raise ValueError("请在本机 .env 中替换三个“填写……”占位值。")

    address = urlsplit(config["LLM_BASE_URL"])
    if address.scheme not in {"http", "https"} or not address.netloc:
        raise ValueError("LLM_BASE_URL 应为代理提供的 HTTP(S) API 根地址。")
    if address.username or address.password or address.query or address.fragment:
        raise ValueError("API 根地址不应包含账号、密码、查询参数或片段。")
    if address.path.rstrip("/").endswith(
        ("/responses", "/chat/completions", "/models")
    ):
        raise ValueError(
            "请填写 API 根地址，不要附加 responses、chat/completions 或 models。"
        )
    client = OpenAI(
        base_url=config["LLM_BASE_URL"],
        api_key=config["LLM_API_KEY"],
        timeout=30,
        max_retries=0,
    )
    return client, config["LLM_MODEL"]


def api_error_message(error: APIError) -> str:
    """不打印完整 SDK 异常，避免把代理地址或响应正文带入报告。"""
    if isinstance(error, APITimeoutError):
        return "模型请求超时：检查本机代理及上游连接，本次未自动重试。"
    if isinstance(error, APIConnectionError):
        return "模型连接失败：检查代理运行状态、API 根地址和网络。"
    if isinstance(error, APIStatusError):
        hints = {
            400: "请求格式或模型能力不兼容，请核对 Responses 与 function calling 支持。",
            401: "代理访问密钥或上游授权无效，请在本机检查。",
            403: "当前访问被拒绝，请检查本机代理授权与模型权限。",
            404: "接口路径或模型不存在，请核对 API 根地址和代理中的模型标识。",
            429: "请求受到限流或额度限制，请查看本机代理状态后再试。",
        }
        hint = hints.get(
            error.status_code, "代理或上游返回错误，请在本机检查服务状态。"
        )
        return f"模型 API 返回 HTTP {error.status_code}：{hint}"
    return "模型 API 返回无法处理的错误，请在本机核对接口兼容性。"


def main() -> int:
    try:
        client, model = create_client()
        with client:
            response = client.responses.create(
                model=model,
                instructions="请用中文简短回答。",
                input=[{"role": "user", "content": "请回复：模型接口已连通。"}],
                store=False,
                max_output_tokens=4096,
            )
        if response.status != "completed" or not response.output_text.strip():
            print("模型未返回完整文本；检查代理 Responses 支持及输出上限。")
            return 1
        print(response.output_text.strip())
        print("这一步只验证模型文本请求，工具调用在下一步验证。")
        return 0
    except ValueError as error:
        print(f"配置错误：{error}")
    except APIError as error:
        print(api_error_message(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
