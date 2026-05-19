"""Test import helpers for source-tree execution."""

import importlib.util
import importlib.metadata
import os
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"

if os.environ.get("PYART_TEST_INSTALLED") != "1" and str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))


def pytest_sessionstart(session):
    """Fail fast if tests are accidentally pointed at the wrong pyart."""
    spec = importlib.util.find_spec("pyart")
    if spec is None or spec.origin is None:
        return

    origin = Path(spec.origin).resolve()
    if os.environ.get("PYART_TEST_INSTALLED") == "1":
        if origin.is_relative_to(PYTHON_SOURCE.resolve()):
            raise RuntimeError(
                f"PYART_TEST_INSTALLED=1 but pyart resolves to source tree: {origin}"
            )
        _assert_installed_rust_matches_repo(origin)
    elif not origin.is_relative_to(PYTHON_SOURCE.resolve()):
        raise RuntimeError(
            f"pyart resolves outside source tree during source tests: {origin}"
        )


def _assert_installed_rust_matches_repo(pyart_origin):
    import pyart
    import pyart._rust as rust

    pyart_package_dir = pyart_origin.parent
    rust_path = Path(rust.__file__).resolve()
    if rust_path.parent != pyart_package_dir:
        raise RuntimeError(
            "PYART_TEST_INSTALLED=1 but pyart._rust does not live beside the "
            f"installed pyart package: pyart={pyart_package_dir}, rust={rust_path}"
        )
    if rust_path.is_relative_to(PYTHON_SOURCE.resolve()):
        raise RuntimeError(
            f"PYART_TEST_INSTALLED=1 but pyart._rust resolves to source tree: {rust_path}"
        )

    expected_project_version = _toml_version(ROOT / "pyproject.toml", "project")
    installed_version = importlib.metadata.version("arm_pyart")
    if installed_version != expected_project_version:
        raise RuntimeError(
            "Installed arm_pyart version does not match this repo: "
            f"installed={installed_version}, repo={expected_project_version}"
        )
    if pyart.__version__ != expected_project_version:
        raise RuntimeError(
            "Installed pyart.__version__ does not match this repo: "
            f"pyart={pyart.__version__}, repo={expected_project_version}"
        )

    expected_rust_version = _toml_version(ROOT / "Cargo.toml", "package")
    if rust.version() != expected_rust_version:
        raise RuntimeError(
            "Installed pyart._rust version does not match this repo: "
            f"rust={rust.version()}, repo={expected_rust_version}"
        )


def _toml_version(path, table):
    with path.open("rb") as fileobj:
        return tomllib.load(fileobj)[table]["version"]
