import hashlib
import json
from pathlib import Path

from src.infrastructure.storage import write_case_snapshot
from src.models.request import CaseUpload
from src.tools.local import LocalTools

ROLE_FILES = {
    "broker": "broker.log",
    "publisher": "publisher.json",
    "subscriber": "subscriber.json"
}

def make_case_id(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()

def describe_upload(upload:CaseUpload) ->dict:
    files = {role:None for role in ROLE_FILES}
    for item in upload.files:
        files[item.role] = {
            "file":ROLE_FILES[item.role],
            "bytes":len(item.content),
            "sha256":hashlib.sha256(item.content).hexdigest()
        }
    return {"format": "communication-case-v1", "files": files}

def materialize_case(
        upload:CaseUpload,
        output_root:Path = Path("outputs/cases")
) -> Path:
    expected = describe_upload(upload)
    case_id = make_case_id(expected)

    files = {ROLE_FILES[item.role]: item.content for item in upload.files}
    root = write_case_snapshot(case_id, files, output_root)

    if expected != LocalTools(root).case_manifest():
        raise ValueError("保存后的案例内容与上传内容不一致。")
    return root