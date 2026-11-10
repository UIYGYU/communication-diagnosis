"""界面自动检查：上传对象和应用入口使用替身，不调用模型服务。

AppTest 未提供文件上传操作时，用合成上传对象检查页面业务状态；
浏览器中的真实选择文件、下载与布局仍需按教程人工检查。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
from streamlit.testing.v1 import AppTest

from examples.stage4 import service


APP_PATH = Path(__file__).with_name("app.py")


def setup_page(monkeypatch):
    uploads = {}
    calls = []

    def fake_upload(label, *, type, key):
        return uploads.get(key)

    def fake_analyze(request):
        calls.append(request)
        return {
            "task_id": f"test-{len(calls)}",
            "input_id": service.request_identity(request),
            "input_manifest": service.input_manifest(request),
            "status": "completed",
            "reason": "离线界面测试结果",
            "report": {
                "summary": "仅供界面测试",
                "limitations": ["未请求真实模型"],
                "evidence_ids": [],
                "clause_ids": [],
            },
            "trace": [],
            "evidence": {},
            "clauses": {},
        }

    monkeypatch.setattr(st, "file_uploader", fake_upload)
    monkeypatch.setattr(service, "analyze", fake_analyze)
    page = AppTest.from_file(str(APP_PATH), default_timeout=10)
    return page, uploads, calls


def upload(content=b"first synthetic log"):
    return SimpleNamespace(name="broker.log", getvalue=lambda: content)


def test_start_requires_input_and_does_not_call_model_on_rerun(monkeypatch):
    page, uploads, calls = setup_page(monkeypatch)
    page.run()
    assert not page.exception
    assert page.button[0].disabled
    assert calls == []
    uploads["broker"] = upload()
    page.run()
    assert not page.button[0].disabled
    assert calls == []
    page.button[0].click().run()
    assert not page.exception
    assert len(calls) == 1
    assert page.session_state["result"]["task_id"] == "test-1"
    download = page.get("download_button")
    assert len(download) == 1
    assert download[0].proto.ignore_rerun is True
    before = json.dumps(page.session_state["result"], sort_keys=True)
    page.run()
    assert len(calls) == 1
    assert json.dumps(page.session_state["result"], sort_keys=True) == before


def test_changing_question_or_upload_invalidates_previous_result(monkeypatch):
    page, uploads, calls = setup_page(monkeypatch)
    uploads["broker"] = upload()
    page.run()
    page.button[0].click().run()
    assert len(calls) == 1
    page.text_area(key="question").set_value("换一个问题").run()
    assert "result" not in page.session_state
    assert len(page.get("download_button")) == 0
    assert len(calls) == 1
    page.button[0].click().run()
    assert len(calls) == 2
    uploads["broker"] = upload(b"different synthetic log")
    page.run()
    assert "result" not in page.session_state
    assert len(page.get("download_button")) == 0
    assert len(calls) == 2
    uploads.clear()
    page.run()
    assert page.button[0].disabled
    assert not page.exception


def test_failed_run_does_not_keep_a_previous_success_report(monkeypatch):
    page, uploads, calls = setup_page(monkeypatch)
    uploads["broker"] = upload()
    page.run()
    page.button[0].click().run()
    previous = page.session_state["result"]

    def fail(request):
        return {
            **previous,
            "status": "failed",
            "reason": "模拟连接失败",
            "report": None,
        }

    monkeypatch.setattr(service, "analyze", fail)
    page.button[0].click().run()
    assert not page.exception
    assert page.session_state["result"]["status"] == "failed"
    assert page.session_state["result"]["report"] is None
    assert any("模拟连接失败" in error.value for error in page.error)
    assert len(calls) == 1
