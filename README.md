# 通信诊断助手

结合通信软件联调经验，开发由模型选择工具、读取执行结果、依据日志与协议资料生成报告的 Agent 应用。MQTT 是首个可复现的验证对象。

## 代码结构

| 目录 | 用途 |
| --- | --- |
| `src` | 二版及后续版本共用的应用包，按职责逐步编写 |
| `examples/stage1` 至 `examples/stage4` | 首版环境、解析、模型工具循环、MCP 和界面实践 |
| `experiments/langchain_lab` | LangChain 高层工具循环对照实验 |
| `experiments/langgraph_lab` | LangGraph 节点、路由、状态及同案例追问实验 |

模型适配沿用现有 Responses SDK，工具通过 MCP 接入，协议检索采用 SQLite FTS5，界面采用 Streamlit。二版确定使用 LangGraph，后续逐步增加上下文摘要、跨会话记忆、主动实验与抓包。

## 本地运行约定

从代码项目根目录运行命令。`src` 自身是 Python 包，通过 `src.*` 或包内相对导入组织应用，无需先增加项目名包层级或打包安装。

首版完整入口位于 `examples/stage4`；二版从 `src/infrastructure/llm_client.py` 的模型适配开始亲手编写。模型地址、标识和密钥由本机环境提供。

## 本地文档

开发计划、教程和验证资料放在代码仓库外的工作区 `文档` 目录，单独管理，不上传到本仓库。求职简历也在仓库外单独保存。
