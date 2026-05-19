"""Compare two API manifests produced by tools.api_manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_manifests(
    current: dict[str, Any], oracle: dict[str, Any]
) -> dict[str, Any]:
    current_modules = set(current.get("imported_modules", []))
    oracle_modules = set(oracle.get("imported_modules", []))

    current_errors = {
        item["module"]: item for item in current.get("import_errors", [])
    }
    oracle_errors = {
        item["module"]: item for item in oracle.get("import_errors", [])
    }

    report: dict[str, Any] = {
        "package_current": current.get("package"),
        "package_oracle": oracle.get("package"),
        "missing_modules": sorted(oracle_modules - current_modules),
        "extra_modules": sorted(current_modules - oracle_modules),
        "import_error_drift": [],
        "public_name_drift": [],
        "signature_drift": [],
    }

    for module in sorted(oracle_errors.keys() - current_errors.keys()):
        report["import_error_drift"].append(
            {"module": module, "kind": "missing_error", "oracle": oracle_errors[module]}
        )
    for module in sorted(current_errors.keys() - oracle_errors.keys()):
        report["import_error_drift"].append(
            {"module": module, "kind": "extra_error", "current": current_errors[module]}
        )
    for module in sorted(current_errors.keys() & oracle_errors.keys()):
        if current_errors[module] != oracle_errors[module]:
            report["import_error_drift"].append(
                {
                    "module": module,
                    "kind": "changed_error",
                    "current": current_errors[module],
                    "oracle": oracle_errors[module],
                }
            )

    shared_modules = sorted(current_modules & oracle_modules)
    for module in shared_modules:
        current_mod = current.get("modules", {}).get(module, {})
        oracle_mod = oracle.get("modules", {}).get(module, {})
        current_names = current_mod.get("public_names", [])
        oracle_names = oracle_mod.get("public_names", [])
        if current_names != oracle_names:
            report["public_name_drift"].append(
                {
                    "module": module,
                    "missing_public_names": sorted(set(oracle_names) - set(current_names)),
                    "extra_public_names": sorted(set(current_names) - set(oracle_names)),
                }
            )

        for name in sorted(set(current_names) & set(oracle_names)):
            current_sig = current_mod.get("public_signatures", {}).get(name)
            oracle_sig = oracle_mod.get("public_signatures", {}).get(name)
            if current_sig != oracle_sig:
                report["signature_drift"].append(
                    {
                        "module": module,
                        "name": name,
                        "current": current_sig,
                        "oracle": oracle_sig,
                    }
                )

    report["ok"] = not any(
        report[key]
        for key in (
            "missing_modules",
            "extra_modules",
            "import_error_drift",
            "public_name_drift",
            "signature_drift",
        )
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    report = compare_manifests(_load(args.current), _load(args.oracle))
    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
