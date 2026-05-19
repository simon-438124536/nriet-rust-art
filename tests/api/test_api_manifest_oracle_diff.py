import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_COMPARE = REPO_ROOT / "tools" / "api_manifest_compare.py"
TOOL_MANIFEST = REPO_ROOT / "tools" / "api_manifest.py"
TOOL_STATIC = REPO_ROOT / "tools" / "api_manifest_static.py"
ORACLE_ROOT = Path(os.environ.get("PYART_ORACLE_ROOT", r"F:\nriet-rust-art\_oracle"))


def _load_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


api_manifest = _load_tool(TOOL_MANIFEST, "api_manifest_for_oracle_diff")
api_manifest_static = _load_tool(TOOL_STATIC, "api_manifest_static_for_oracle_diff")
api_manifest_compare = _load_tool(TOOL_COMPARE, "api_manifest_compare_for_oracle_diff")


def _oracle_pyart_parent() -> Path | None:
    candidates = [
        ORACLE_ROOT / "pyart-main" / "pyart-main",
        ORACLE_ROOT / "pyart-main",
    ]
    for candidate in candidates:
        if (candidate / "pyart" / "__init__.py").is_file():
            return candidate
        if candidate.name == "pyart" and (candidate / "__init__.py").is_file():
            return candidate.parent
    return None


@pytest.mark.skipif(
    _oracle_pyart_parent() is None,
    reason="oracle pyart tree not available; set PYART_ORACLE_ROOT or unzip pyart-main.zip",
)
def test_current_pyart_static_api_manifest_matches_oracle(tmp_path):
    oracle_parent = _oracle_pyart_parent()
    assert oracle_parent is not None

    current_static = api_manifest_static.build_static_manifest(
        REPO_ROOT / "python", "pyart"
    )
    oracle_static = api_manifest_static.build_static_manifest(oracle_parent, "pyart")
    oracle_pyart_dir = oracle_parent / "pyart"
    if not oracle_pyart_dir.is_dir():
        oracle_pyart_dir = oracle_parent

    report = api_manifest_compare.compare_manifests(
        current_static,
        oracle_static,
        oracle_module_basis=True,
        oracle_pyart_dir=oracle_pyart_dir,
    )
    if not report["ok"]:
        report_path = tmp_path / "api-diff.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pytest.fail(
            "Static API manifest drift vs oracle; see "
            f"{report_path} for missing_modules/extra_modules/signature_drift"
        )


@pytest.mark.skipif(
    _oracle_pyart_parent() is None,
    reason="oracle pyart tree not available; set PYART_ORACLE_ROOT or unzip pyart-main.zip",
)
def test_current_pyart_imported_root_public_api_matches_oracle():
    oracle_parent = _oracle_pyart_parent()
    assert oracle_parent is not None

    current_static = api_manifest_static.build_static_manifest(
        REPO_ROOT / "python", "pyart"
    )
    oracle_static = api_manifest_static.build_static_manifest(oracle_parent, "pyart")
    current_root = current_static["modules"].get("pyart", {})
    oracle_root = oracle_static["modules"].get("pyart", {})

    def _root_surface(module_info):
        return sorted(
            set(module_info.get("public_names", []))
            | set(module_info.get("relative_package_imports", []))
        )

    assert _root_surface(current_root) == _root_surface(oracle_root)
