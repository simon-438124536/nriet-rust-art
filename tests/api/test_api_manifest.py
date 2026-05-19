import importlib.util
import json
import os
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "api_manifest.py"
SPEC = importlib.util.spec_from_file_location("api_manifest_under_test", TOOL_PATH)
api_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(api_manifest)


def _drop_modules(prefix):
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def test_build_manifest_collects_public_api_and_import_errors(tmp_path, monkeypatch):
    package_dir = tmp_path / "samplepkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "\n".join(
            [
                "__all__ = ['PublicClass', 'public_func']",
                "",
                "def public_func(a, b='x'):",
                "    return b",
                "",
                "class PublicClass:",
                "    def __init__(self, value=1):",
                "        self.value = value",
                "",
                "def visible_but_not_exported():",
                "    return None",
            ]
        ),
        encoding="utf-8",
    )
    (package_dir / "sub.py").write_text(
        "\n".join(
            [
                "def sub_func(left, right=2):",
                "    return left + right",
                "",
                "class SubClass:",
                "    pass",
                "",
                "def _private():",
                "    return None",
            ]
        ),
        encoding="utf-8",
    )
    (package_dir / "bad.py").write_text(
        "import definitely_missing_manifest_dependency\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    _drop_modules("samplepkg")

    manifest = api_manifest.build_manifest("samplepkg")

    assert manifest["package"] == "samplepkg"
    assert manifest["imported_modules"] == ["samplepkg", "samplepkg.sub"]
    assert [error["module"] for error in manifest["import_errors"]] == [
        "samplepkg.bad"
    ]
    assert manifest["import_errors"][0]["error_type"] == "ModuleNotFoundError"

    root_module = manifest["modules"]["samplepkg"]
    assert root_module["__all__"] == ["PublicClass", "public_func"]
    assert root_module["public_names"] == ["PublicClass", "public_func"]
    assert "visible_but_not_exported" in root_module["dir"]
    assert "visible_but_not_exported" not in root_module["public_names"]
    assert root_module["public_signatures"]["public_func"] == {
        "kind": "callable",
        "signature": "(a, b='x')",
        "error": None,
    }
    assert root_module["public_signatures"]["PublicClass"] == {
        "kind": "class",
        "signature": "(value=1)",
        "error": None,
    }

    sub_module = manifest["modules"]["samplepkg.sub"]
    assert sub_module["__all__"] is None
    assert "sub_func" in sub_module["public_names"]
    assert "SubClass" in sub_module["public_names"]
    assert "_private" not in sub_module["public_names"]
    assert sub_module["public_signatures"]["sub_func"]["signature"] == (
        "(left, right=2)"
    )


def test_pyart_quiet_is_set_by_default_without_overwriting(tmp_path, monkeypatch):
    package_dir = tmp_path / "pyart"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "import os\nQUIET_AT_IMPORT = os.environ.get('PYART_QUIET')\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("PYART_QUIET", raising=False)
    _drop_modules("pyart")

    manifest = api_manifest.build_manifest("pyart", recursive=False)

    assert os.environ["PYART_QUIET"] == "1"
    assert "QUIET_AT_IMPORT" in manifest["modules"]["pyart"]["public_names"]
    _drop_modules("pyart")

    monkeypatch.setenv("PYART_QUIET", "already-set")
    api_manifest._ensure_pyart_quiet_default("pyart")
    assert os.environ["PYART_QUIET"] == "already-set"


def test_main_writes_manifest_json(tmp_path):
    package_dir = tmp_path / "clipkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "def entrypoint(flag=False):\n    return flag\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out" / "manifest.json"

    exit_code = api_manifest.main(
        [
            "--package",
            "clipkg",
            "--path",
            str(tmp_path),
            "--output",
            str(output_path),
            "--no-recursive",
        ]
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["import_errors"] == []
    assert manifest["modules"]["clipkg"]["public_signatures"]["entrypoint"][
        "signature"
    ] == "(flag=False)"
