import os
from openai import APIError, OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from openai.types.responses import Response

INSTRUCTIONS = """
请用中文简短回答。先读日志、根据实际结果回答。"""


def create_client() -> tuple[OpenAI, str]:
    """读取本机配置，返回客户端和模型标识。"""
    names = ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]
    config = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise ValueError("缺少本机配置：" + ",".join(missing))
    client = OpenAI(
        api_key=config["LLM_API_KEY"],
        base_url=config["LLM_BASE_URL"],
        timeout=30,
        max_retries=0,
    )
    return client, config["LLM_MODEL"]


def request_model(
    client: OpenAI,
    model: str,
    history: list[dict],
    *,
    tools: list[dict] | None = None,
    instructions: str = INSTRUCTIONS,
    max_output_tokens: int = 4096,
) -> Response:
    """发送输入，保留完整响应给调用方。"""
    params = {
        "model": model,
        "input": history,
        "instructions": instructions,
        "store": False,
        "max_output_tokens": max_output_tokens,
    }
    if tools is not None:
        params.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
    return client.responses.create(**params)


def main() -> int:
    """组织一次终端请求，成功返回 0，失败返回 1。"""
    try:
        question = input("请输入你的问题:").strip()
        if not question:
            raise ValueError("缺少问题")
        client, model = create_client()
        msg = [{"role": "user", "content": question}]
        with client:
            response = request_model(client, model, msg)
        if response.status != "completed" or not response.output_text.strip():
            print("模型未返回完整文本；检查代理 Responses 支持及输出上限。")
            return 1
        print(response.output_text.strip())
        return 0
    except APIError as error:
        if isinstance(error, APITimeoutError):
            err = "模型请求超时：检查本机代理及上游连接，本次未自动重试。"
        elif isinstance(error, APIConnectionError):
            err = "模型连接失败：检查代理运行状态、API 根地址和网络。"
        elif isinstance(error, APIStatusError):
            err = f"模型返回HTTP状态码{error.status_code}"
        else:
            err = "发生其他模型 API 错误。"
        print(f"{type(error).__name__}: {err}")
    except ValueError as error:
        print(f"输入或配置错误：{error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
