import ast
import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CURRENT_PYART = ROOT / "python" / "pyart"
DEFAULT_ORACLE_PYART = Path(r"F:\nriet-rust-art\_oracle\pyart-main\pyart")


def _oracle_pyart_root():
    return Path(os.environ.get("PYART_ORACLE_ROOT", DEFAULT_ORACLE_PYART))


def _python_class_methods(path, class_name):
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _python_top_level_defs(path):
    module = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _cython_top_level_defs(path):
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(?:def|class|cdef class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if match:
            names.add(match.group(1))
    return names


def _cython_class_methods(path, class_name):
    text = path.read_text(encoding="utf-8").splitlines()
    in_class = False
    names = set()
    for line in text:
        if re.match(rf"^cdef class\s+{re.escape(class_name)}\b", line):
            in_class = True
            continue
        if in_class and line and not line.startswith((" ", "\t")):
            break
        if in_class:
            match = re.match(
                r"^\s+(?:def\s+|cdef\s+\w+\s+)([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                line,
            )
            if match:
                names.add(match.group(1))
    return names


@pytest.mark.skipif(
    not _oracle_pyart_root().exists(),
    reason="frozen Py-ART oracle source is not available",
)
def test_sigmet_static_api_matches_cython_oracle_surface():
    oracle = _oracle_pyart_root() / "io" / "_sigmetfile.pyx"
    current = CURRENT_PYART / "io" / "_sigmetfile.py"

    expected_top_level = _cython_top_level_defs(oracle)
    actual_top_level = _python_top_level_defs(current)
    assert expected_top_level <= actual_top_level

    expected_methods = _cython_class_methods(oracle, "SigmetFile")
    actual_methods = _python_class_methods(current, "SigmetFile")
    assert expected_methods <= actual_methods
