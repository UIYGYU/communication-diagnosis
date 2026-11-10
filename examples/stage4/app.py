"""整合入口：从代码根目录用 uv run --env-file .env -m streamlit run 启动。"""

import json

import streamlit as st
from pydantic import ValidationError

from examples.stage4.service import (
    APP_VERSION,
    SUPPORTED_SCOPE,
    DiagnosisRequest,
    UploadedLog,
    analyze,
    readable_input_error,
    request_identity,
)

st.set_page_config(page_title="通信诊断助手", layout="wide")
st.title("通信诊断助手")
st.caption("用户问题 → Agent → MCP 工具 → 模型报告")
st.warning("当前整合版只结构化识别 QoS 1 PUBLISH，并使用协议教学摘要。")

question = st.text_area(
    "想检查什么？",
    value="请查看发布者和订阅者日志，结合协议摘要说明能确认的收发情况与证据缺口。",
    max_chars=4000,
    key="question",
)
uploads = {
    "broker": st.file_uploader("服务端日志", type=["log", "txt"], key="broker"),
    "publisher": st.file_uploader("发布端记录", type=["json"], key="publisher"),
    "subscriber": st.file_uploader("订阅端记录", type=["json"], key="subscriber"),
}
with st.expander("执行上限"):
    max_rounds = st.number_input("模型请求轮数", min_value=1, max_value=12, value=6)
    max_tool_calls = st.number_input("工具调用次数", min_value=1, max_value=16, value=8)

selected = [(role, item) for role, item in uploads.items() if item is not None]
request = None
input_token = None
input_error = None
if selected and question.strip():
    try:
        request = DiagnosisRequest(
            question=question,
            files=[
                UploadedLog(role=role, file_name=item.name, content=item.getvalue())
                for role, item in selected
            ],
            max_rounds=max_rounds,
            max_tool_calls=max_tool_calls,
        )
        input_token = request_identity(request)
    except ValidationError as error:
        input_error = readable_input_error(error)

# 改问题、文件或预算即清除旧结果；没有输入或输入无效时同样清除。
if st.session_state.get("active_input") != input_token:
    st.session_state["active_input"] = input_token
    st.session_state.pop("result", None)
    st.session_state.pop("result_input", None)

if input_error:
    st.error(input_error)
if not selected:
    st.info("请至少选择一份日志；三种角色总计不超过 10 MiB。")
elif not question.strip():
    st.info("请输入要检查的问题。")

# 全文件只有这里调用 analyze；普通重绘、展开证据和下载都不重复请求模型。
if st.button("开始诊断", disabled=request is None, type="primary"):
    st.session_state.pop("result", None)
    st.session_state.pop("result_input", None)
    with st.spinner("Agent 正在调用模型和 MCP 工具……"):
        result = analyze(request)
    st.session_state["result"] = result
    st.session_state["result_input"] = input_token

if (
    input_token is not None
    and st.session_state.get("result_input") == input_token
    and "result" in st.session_state
):
    result = st.session_state["result"]
    status = result["status"]
    status_names = {
        "completed": "本次流程完成",
        "incomplete": "未完成",
        "failed": "失败",
    }
    st.write("任务状态：", status_names.get(status, status))
    if status == "completed":
        st.success(result["reason"])
    elif status == "incomplete":
        st.warning(result["reason"])
    else:
        st.error(result["reason"])

    if result.get("report") is not None:
        report = result["report"]
        st.subheader("诊断结果")
        st.write(report["summary"])
        for limitation in report["limitations"]:
            st.write("·", limitation)
        st.write("日志引用：", report["evidence_ids"])
        st.write("协议引用：", report["clause_ids"])

    with st.expander("查看工具调用轨迹", expanded=False):
        st.json(result["trace"])
    with st.expander("查看本次日志证据和协议摘要", expanded=False):
        st.json({"evidence": result["evidence"], "clauses": result["clauses"]})
    with st.expander("查看输入身份与支持范围", expanded=False):
        st.json(
            {
                "task_id": result["task_id"],
                "input_id": result["input_id"],
                "input_manifest": result["input_manifest"],
                "app_version": APP_VERSION,
                "supported_scope": SUPPORTED_SCOPE,
            }
        )
    download = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "下载本次报告与轨迹",
        data=download,
        file_name=f"通信诊断报告-{result['task_id']}.json",
        mime="application/json",
        on_click="ignore",
    )
else:
    st.write("任务状态：待诊断")

st.caption("修改本机 .env 后请停止并重新启动界面，使新的模型配置生效。")
