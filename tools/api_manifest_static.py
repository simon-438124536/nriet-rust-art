"""Build an API manifest from Python source without importing the package."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any


def _api_manifest_module():
    path = Path(__file__).with_name("api_manifest.py")
    spec = importlib.util.spec_from_file_location("_api_manifest_dep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_manifest = _api_manifest_module()
SCHEMA_VERSION = _manifest.SCHEMA_VERSION
_jsonable_list = _manifest._jsonable_list

ALLOWED_CURRENT_EXTRA_MODULE_PREFIXES = (
    "pyart._rust_bridge",
    "pyart._rust",
)


def _package_dir(package_root: Path, package_name: str) -> Path:
    candidate = package_root / package_name
    if candidate.is_dir() and (candidate / "__init__.py").is_file():
        return candidate
    if package_root.name == package_name and (package_root / "__init__.py").is_file():
        return package_root
    raise FileNotFoundError(
        f"could not find package {package_name!r} under {package_root}"
    )


def _module_name(package_name: str, relative: Path) -> str:
    if relative == Path("__init__.py"):
        return package_name
    parts = list(relative.with_suffix("").parts)
    return package_name + "." + ".".join(parts)


def _public_names_from_ast(module: ast.Module) -> list[str]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return sorted(
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        )
    return sorted(
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    )


def _ast_callable_signature(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.ClassDef):
        kind = "class"
        signature = "()"
    else:
        kind = "callable"
        try:
            signature = (
                f"({ast.unparse(node.args)})" if hasattr(ast, "unparse") else "(...)"
            )
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "kind": kind,
                "signature": None,
                "error": {"error_type": type(exc).__name__, "message": str(exc)},
            }
    return {"kind": kind, "signature": signature, "error": None}


def _relative_package_import_names(module: ast.Module) -> list[str]:
    names: list[str] = []
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.level and node.level >= 1:
            if node.module is not None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.append(alias.asname or alias.name)
    return sorted(names)


def _module_all_from_ast(module: ast.Module) -> list[str] | None:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return sorted(
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        )
    return None


def _module_manifest_from_ast(module: ast.Module) -> dict[str, Any]:
    public_names = _public_names_from_ast(module)
    signatures: dict[str, dict[str, Any]] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in public_names:
                signatures[node.name] = _ast_callable_signature(node)
    return {
        "dir": public_names,
        "__all__": _jsonable_list(_module_all_from_ast(module)),
        "public_names": public_names,
        "relative_package_imports": _relative_package_import_names(module),
        "public_signatures": dict(sorted(signatures.items())),
        "signature_errors": {},
    }


def build_static_manifest(
    package_root: Path, package_name: str = "pyart"
) -> dict[str, Any]:
    """Return a manifest shaped like :func:`api_manifest.build_manifest`."""

    package_dir = _package_dir(package_root, package_name)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package": package_name,
        "imported_modules": [],
        "modules": {},
        "import_errors": [],
        "source": "static",
    }

    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)
        module_name = _module_name(package_name, relative)
        try:
            module_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            manifest["import_errors"].append(
                {
                    "module": module_name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "",
                }
            )
            continue
        manifest["imported_modules"].append(module_name)
        manifest["modules"][module_name] = _module_manifest_from_ast(module_ast)

    manifest["imported_modules"].sort()
    manifest["import_errors"].sort(key=lambda item: item["module"])
    manifest["modules"] = dict(sorted(manifest["modules"].items()))
    return manifest
