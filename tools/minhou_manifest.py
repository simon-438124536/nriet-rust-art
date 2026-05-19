"""Build an RSTM manifest for the Minhou comparison data root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

try:  # Works both as ``python tools/minhou_manifest.py`` and as a module.
    from .rstm_reference import (
        DEFAULT_CHUNK_BYTES,
        DEFAULT_HEADER_BYTES,
        build_reference_record,
    )
except ImportError:  # pragma: no cover - exercised by direct script usage.
    from rstm_reference import (  # type: ignore
        DEFAULT_CHUNK_BYTES,
        DEFAULT_HEADER_BYTES,
        build_reference_record,
    )


ENV_DATA_ROOT = "RSTM_DATA_ROOT"
SCHEMA_VERSION = "minhou-rstm-manifest-v1"


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve CLI/env/default Minhou data root without touching repo data."""

    if data_root is not None:
        return Path(data_root)

    env_root = os.environ.get(ENV_DATA_ROOT)
    if env_root:
        return Path(env_root)

    raise FileNotFoundError(
        "No RSTM data root was provided. Set RSTM_DATA_ROOT, pass "
        "--data-root, or run from a wrapper that provides an explicit "
        "operational data path."
    )


def _normalised_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_manifest_files(
    data_root: str | Path,
    *,
    patterns: Iterable[str] = ("*",),
    max_files: int | None = None,
) -> list[Path]:
    """Return a deterministic list of files under the data root."""

    root = Path(data_root)
    seen: set[Path] = set()
    files: list[Path] = []

    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)

    files.sort(key=lambda item: _normalised_relative(item, root).lower())
    if max_files is not None:
        if max_files < 0:
            raise ValueError("max_files must be non-negative")
        files = files[:max_files]
    return files


def build_manifest(
    data_root: str | Path | None = None,
    *,
    patterns: Iterable[str] = ("*",),
    max_files: int | None = None,
    header_bytes: int = DEFAULT_HEADER_BYTES,
    include_compressed_sha256: bool = False,
    include_decompressed_sha256: bool = False,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, object]:
    root = resolve_data_root(data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"RSTM data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RSTM data root is not a directory: {root}")

    pattern_list = list(patterns)
    files = iter_manifest_files(
        root, patterns=pattern_list, max_files=max_files
    )
    entries: list[dict[str, object]] = []

    for path in files:
        record = build_reference_record(
            path,
            header_bytes=header_bytes,
            include_compressed_sha256=include_compressed_sha256,
            include_decompressed_sha256=include_decompressed_sha256,
            chunk_bytes=chunk_bytes,
        )
        record["relative_path"] = _normalised_relative(path, root)
        record.pop("path", None)
        entries.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(root),
        "file_count": len(entries),
        "patterns": pattern_list,
        "header_bytes": header_bytes,
        "includes": {
            "compressed_sha256": include_compressed_sha256,
            "decompressed_sha256": include_decompressed_sha256,
        },
        "entries": entries,
    }


def write_manifest(manifest: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manifest for Minhou RSTM comparison data."
    )
    parser.add_argument(
        "--data-root",
        help=(
            "RSTM data root. Defaults to RSTM_DATA_ROOT, then the external "
            "Minhou comparison data folder when present."
        ),
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="recursive glob pattern to include; may be passed more than once",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="limit files after deterministic sorting; useful for smoke runs",
    )
    parser.add_argument(
        "--header-bytes",
        type=int,
        default=DEFAULT_HEADER_BYTES,
        help="number of logical payload bytes to preview",
    )
    parser.add_argument(
        "--compressed-sha256",
        action="store_true",
        help="include SHA256 of the on-disk bytes",
    )
    parser.add_argument(
        "--decompressed-sha256",
        action="store_true",
        help="include SHA256 of the logical decompressed payload",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="streaming hash chunk size",
    )
    parser.add_argument(
        "--out",
        help="write JSON manifest to this path instead of stdout",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_manifest(
        args.data_root,
        patterns=args.pattern or ("*",),
        max_files=args.max_files,
        header_bytes=args.header_bytes,
        include_compressed_sha256=args.compressed_sha256,
        include_decompressed_sha256=args.decompressed_sha256,
        chunk_bytes=args.chunk_bytes,
    )

    if args.out:
        write_manifest(manifest, args.out)
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
