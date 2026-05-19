import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_COMPARE = REPO_ROOT / "tools" / "api_manifest_compare.py"
TOOL_MANIFEST = REPO_ROOT / "tools" / "api_manifest.py"
ORACLE_ROOT = Path(os.environ.get("PYART_ORACLE_ROOT", r"F:\nriet-rust-art\_oracle"))


def _load_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


api_manifest = _load_tool(TOOL_MANIFEST, "api_manifest_for_oracle_diff")
api_manifest_compare = _load_tool(TOOL_COMPARE, "api_manifest_compare_for_oracle_diff")


def _oracle_pyart_parent() -> Path | None:
    candidates = [
        ORACLE_ROOT / "pyart-main" / "pyart-main",
        ORACLE_ROOT / "pyart-main",
    ]
    for candidate in candidates:
        if (candidate / "pyart" / "__init__.py").is_file():
            return candidate
    return None


@pytest.mark.skipif(
    _oracle_pyart_parent() is None,
    reason="oracle pyart tree not available; set PYART_ORACLE_ROOT or unzip pyart-main.zip",
)
def test_current_pyart_api_manifest_matches_oracle(tmp_path, monkeypatch):
    oracle_parent = _oracle_pyart_parent()
    assert oracle_parent is not None

    current_manifest = api_manifest.build_manifest("pyart")
    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")

    for name in list(sys.modules):
        if name == "pyart" or name.startswith("pyart."):
            del sys.modules[name]

    monkeypatch.syspath_prepend(str(oracle_parent))
    oracle_manifest = api_manifest.build_manifest("pyart")
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text(json.dumps(oracle_manifest, indent=2), encoding="utf-8")

    report = api_manifest_compare.compare_manifests(current_manifest, oracle_manifest)
    if not report["ok"]:
        report_path = tmp_path / "api-diff.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pytest.fail(
            "API manifest drift vs oracle; see "
            f"{report_path} for missing_modules/extra_modules/signature_drift"
        )
