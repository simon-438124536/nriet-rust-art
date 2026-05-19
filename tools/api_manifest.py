"""Build a JSON API manifest for an importable Python package."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _ensure_pyart_quiet_default(package_name: str) -> None:
    root_name = package_name.split(".", 1)[0]
    if root_name == "pyart":
        os.environ.setdefault("PYART_QUIET", "1")


def _jsonable_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    try:
        return sorted(str(item) for item in value)
    except TypeError:
        return [repr(value)]


def _public_names(module: ModuleType, dir_names: Iterable[str]) -> list[str]:
    all_names = _jsonable_list(getattr(module, "__all__", None))
    if all_names is not None:
        return all_names
    return sorted(name for name in dir_names if not name.startswith("_"))


def _signature_record(obj: Any) -> dict[str, Any]:
    if inspect.isclass(obj):
        kind = "class"
    else:
        kind = "callable"

    try:
        signature = str(inspect.signature(obj))
        error = None
    except (TypeError, ValueError) as exc:
        signature = None
        error = {"error_type": type(exc).__name__, "message": str(exc)}

    return {"kind": kind, "signature": signature, "error": error}


def _module_manifest(module: ModuleType) -> dict[str, Any]:
    dir_names = sorted(dir(module))
    public_names = _public_names(module, dir_names)
    signatures: dict[str, dict[str, Any]] = {}
    signature_errors: dict[str, dict[str, str]] = {}

    for name in public_names:
        try:
            obj = getattr(module, name)
        except Exception as exc:  # pragma: no cover - defensive for dynamic APIs
            signature_errors[name] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            continue

        if inspect.isclass(obj) or callable(obj):
            signatures[name] = _signature_record(obj)

    return {
        "dir": dir_names,
        "__all__": _jsonable_list(getattr(module, "__all__", None)),
        "public_names": public_names,
        "public_signatures": dict(sorted(signatures.items())),
        "signature_errors": dict(sorted(signature_errors.items())),
    }


def _import_error(module_name: str, exc: BaseException) -> dict[str, str]:
    return {
        "module": module_name,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _iter_child_module_names(module: ModuleType) -> list[str]:
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return []
    prefix = module.__name__ + "."
    return sorted(info.name for info in pkgutil.iter_modules(module_path, prefix))


def build_manifest(package_name: str = "pyart", *, recursive: bool = True) -> dict[str, Any]:
    """Import *package_name* and return a JSON-serializable API manifest."""

    _ensure_pyart_quiet_default(package_name)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package": package_name,
        "imported_modules": [],
        "modules": {},
        "import_errors": [],
    }

    seen: set[str] = set()
    queue = [package_name]

    while queue:
        module_name = queue.pop(0)
        if module_name in seen:
            continue
        seen.add(module_name)

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            manifest["import_errors"].append(_import_error(module_name, exc))
            continue

        manifest["imported_modules"].append(module_name)
        manifest["modules"][module_name] = _module_manifest(module)

        if recursive:
            queue.extend(
                child for child in _iter_child_module_names(module) if child not in seen
            )

    manifest["imported_modules"].sort()
    manifest["import_errors"].sort(key=lambda item: item["module"])
    manifest["modules"] = dict(sorted(manifest["modules"].items()))
    return manifest


def _write_manifest(manifest: dict[str, Any], output: Path | None, indent: int) -> None:
    text = json.dumps(manifest, indent=indent, sort_keys=True)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a JSON API manifest for an importable Python package."
    )
    parser.add_argument(
        "package",
        nargs="?",
        default=None,
        help="Package/module to inspect. Defaults to pyart.",
    )
    parser.add_argument(
        "--package",
        dest="package_option",
        help="Package/module to inspect. Overrides the positional argument.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Prepend an import path before loading the package. May be repeated.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect the requested module/package, not child modules.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Defaults to 2.",
    )
    parser.add_argument(
        "--fail-on-import-error",
        action="store_true",
        help="Return exit code 1 when any import errors are captured.",
    )
    args = parser.parse_args(argv)

    for import_path in reversed(args.path):
        sys.path.insert(0, import_path)

    package_name = args.package_option or args.package or "pyart"
    manifest = build_manifest(package_name, recursive=not args.no_recursive)
    _write_manifest(manifest, args.output, args.indent)

    if args.fail_on_import_error and manifest["import_errors"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
