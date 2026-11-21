import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from src.application.service import analyze, create_session
from src.models.request import CaseConfig, TurnRequest
from src.application.cases import materialize_case
from src.models.request import CaseUpload, UploadedLog

st.set_page_config(page_title="通信诊断助手", layout="wide")
st.title("通信诊断助手")

if "diagnosis_session" not in st.session_state:
    st.session_state.diagnosis_session = None

with st.form("case_config"):
    source = st.radio("案例输入", ["现有目录", "上传日志"])
    broker_file = st.file_uploader("服务端日志", type=["log", "txt"], max_upload_size=1)
    publisher_file = st.file_uploader("发布端记录", type=["json"], max_upload_size=1)
    subscriber_file = st.file_uploader("订阅端记录", type=["json"], max_upload_size=1)
    case_dir_text = st.text_input("案例目录", "samples/case-01")
    transport = st.selectbox("工具连接", ["mcp-http", "mcp-stdio", "local"])
    mcp_url = st.text_input("HTTP MCP 地址", "http://127.0.0.1:8000/mcp")
    new_session = st.form_submit_button("新建会话")

if new_session:
    try:
        if source == "上传日志":
            files = []
            for role, uploaded_file in (
                ("broker", broker_file),
                ("publisher", publisher_file),
                ("subscriber", subscriber_file),
            ):
                if uploaded_file is None:
                    continue
                files.append(
                    UploadedLog(
                        role=role,
                        file_name=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                    )
                )
            caseupload = CaseUpload(files=files)
            case_dir = materialize_case(caseupload)
        else:
            case_dir = Path(case_dir_text)

        config = CaseConfig(case_dir=case_dir, transport=transport, mcp_url=mcp_url)
        session = create_session(config)

        st.session_state.diagnosis_session = session
        st.success("会话已创建")

    except (ValidationError, ValueError) as error:
        st.error(str(error))

session = st.session_state.diagnosis_session
if not session:
    st.info("请先选择案例并新建会话。")
    st.stop()

st.write("当前案例目录：", str(session.case_dir))
st.write("案例ID: ", session.case_id)
st.write("工具连接方式：", session.transport)
st.write("会话 ID：", session.thread_id)

with st.form("question"):
    question = st.text_area("想检查什么？")
    submitted = st.form_submit_button(
        "提交问题", disabled=session.busy or not session.usable
    )

if submitted:
    try:
        request = TurnRequest(question=question)
        with st.spinner("正在诊断……"):
            analyze(session, request)
    except (ValidationError, ValueError) as error:
        st.error(str(error))

for index, result in enumerate(session.turns, start=1):
    st.subheader(f"第 {index} 次诊断")
    st.write("问题: ", result["question"])
    st.write("状态: ", result["status"])
    st.write("原因: ", result["reason"])

    if result["status"] == "completed":
        report = result["report"]

        st.success("本次诊断流程完成")
        st.write("诊断结论：", report["summary"])

        st.write("局限与证据缺口：")
        for limitation in report["limitations"]:
            st.write("•", limitation)
        with st.expander("查看报告引用的日志证据"):
            for evidence_id in report["evidence_ids"]:
                st.write("证据 ID：", evidence_id)
                st.json(result["evidence"][evidence_id])

        with st.expander("查看报告引用的协议资料"):
            for clause_id in report["clause_ids"]:
                st.write("条款 ID：", clause_id)
                st.json(result["clauses"][clause_id])
    elif result["status"] == "incomplete":
        st.warning("本次诊断未完成，请查看上方原因。")
    else:
        st.error("本次诊断失败，请查看上方原因。")

    with st.expander("查看工具调用轨迹"):
        st.json(result["call_trace"])

    with st.expander("查看输入身份"):
        st.json(
            {
                "run_id": result["run_id"],
                "thread_id": result["thread_id"],
                "case_id": result["case_id"],
            }
        )
    if result.get("storage_warning"):
        st.warning(result["storage_warning"])

    with st.expander("查看本问题上下文与预算"):
        st.json(
            {
                "summary": result.get("summary"),
                "summary_calls": result.get("summary_calls", 0),
                "visible_evidence_ids": result.get("visible_evidence_ids", []),
                "visible_clause_ids": result.get("visible_clause_ids", []),
                "history_archive": result.get("history_archive"),
                "context_trace": result.get("context_trace", []),
            }
        )

    st.download_button(
        "下载本次报告与轨迹",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"diagnosis-{result['run_id']}.json",
        mime="application/json",
        key=f"download-{result['run_id']}",
        on_click="ignore",
    )


if not session.usable:
    st.warning("当前会话无法继续使用，请新建会话。")
