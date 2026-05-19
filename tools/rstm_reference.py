"""Frozen Python reference helpers for RSTM file inventory.

The Rust rewrite can use this module as a small, stable oracle for byte-level
RSTM file handling before parser behavior is ported.  It deliberately avoids
Py-ART imports and external dependencies so it can run against operational data
roots and tiny synthetic fixtures in the same way.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Sequence


GZIP_MAGIC = b"\x1f\x8b"
DEFAULT_HEADER_BYTES = 256
DEFAULT_CHUNK_BYTES = 1024 * 1024
SCHEMA_VERSION = "rstm-reference-v1"


@dataclass(frozen=True)
class HeaderPreview:
    """Small logical-stream preview captured without reading the full file."""

    length_bytes: int
    hex: str
    ascii: str

    @classmethod
    def from_bytes(cls, data: bytes) -> "HeaderPreview":
        return cls(
            length_bytes=len(data),
            hex=data.hex(),
            ascii=_ascii_preview(data),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "length_bytes": self.length_bytes,
            "hex": self.hex,
            "ascii": self.ascii,
        }


def _ascii_preview(data: bytes) -> str:
    return "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in data)


def _read_prefix(path: Path, length: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(length)


def is_gzip_file(path: str | Path) -> bool:
    """Return True when the file starts with the gzip magic bytes."""

    return _read_prefix(Path(path), len(GZIP_MAGIC)) == GZIP_MAGIC


def _iter_stream(stream: BinaryIO, chunk_bytes: int) -> Iterator[bytes]:
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            break
        yield chunk


def iter_logical_bytes(
    path: str | Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    is_gzip: bool | None = None,
) -> Iterator[bytes]:
    """Yield file payload bytes, decompressing gzip streams when detected."""

    file_path = Path(path)
    gzip_detected = is_gzip_file(file_path) if is_gzip is None else is_gzip

    with file_path.open("rb") as raw_stream:
        if gzip_detected:
            with gzip.GzipFile(fileobj=raw_stream, mode="rb") as gzip_stream:
                yield from _iter_stream(gzip_stream, chunk_bytes)
        else:
            yield from _iter_stream(raw_stream, chunk_bytes)


def read_header_preview(
    path: str | Path,
    *,
    header_bytes: int = DEFAULT_HEADER_BYTES,
    is_gzip: bool | None = None,
) -> HeaderPreview:
    """Read only the first bytes of the logical RSTM stream."""

    if header_bytes < 0:
        raise ValueError("header_bytes must be non-negative")

    file_path = Path(path)
    gzip_detected = is_gzip_file(file_path) if is_gzip is None else is_gzip

    with file_path.open("rb") as raw_stream:
        if gzip_detected:
            with gzip.GzipFile(fileobj=raw_stream, mode="rb") as gzip_stream:
                preview = gzip_stream.read(header_bytes)
        else:
            preview = raw_stream.read(header_bytes)

    return HeaderPreview.from_bytes(preview)


def sha256_file(
    path: str | Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> str:
    """Hash the bytes exactly as stored on disk."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in _iter_stream(stream, chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_logical_payload(
    path: str | Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    is_gzip: bool | None = None,
) -> tuple[str, int]:
    """Hash logical payload bytes, decompressing gzip streams if needed."""

    digest = hashlib.sha256()
    total = 0
    for chunk in iter_logical_bytes(
        path, chunk_bytes=chunk_bytes, is_gzip=is_gzip
    ):
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def build_reference_record(
    path: str | Path,
    *,
    header_bytes: int = DEFAULT_HEADER_BYTES,
    include_compressed_sha256: bool = False,
    include_decompressed_sha256: bool = False,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, object]:
    """Build a deterministic byte-level reference record for one RSTM file."""

    file_path = Path(path)
    gzip_detected = is_gzip_file(file_path)
    stat = file_path.stat()
    preview = read_header_preview(
        file_path, header_bytes=header_bytes, is_gzip=gzip_detected
    )

    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "compression": "gzip" if gzip_detected else "none",
        "gzip_detected_by_magic": gzip_detected,
        "raw_magic_hex": _read_prefix(file_path, 4).hex(),
        "header_preview": preview.to_dict(),
    }

    if include_compressed_sha256:
        record["compressed_sha256"] = sha256_file(
            file_path, chunk_bytes=chunk_bytes
        )

    if include_decompressed_sha256:
        payload_sha256, payload_size = sha256_logical_payload(
            file_path, chunk_bytes=chunk_bytes, is_gzip=gzip_detected
        )
        record["decompressed_sha256"] = payload_sha256
        record["decompressed_size_bytes"] = payload_size

    return record


def build_reference_records(
    paths: Iterable[str | Path],
    *,
    header_bytes: int = DEFAULT_HEADER_BYTES,
    include_compressed_sha256: bool = False,
    include_decompressed_sha256: bool = False,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> list[dict[str, object]]:
    return [
        build_reference_record(
            path,
            header_bytes=header_bytes,
            include_compressed_sha256=include_compressed_sha256,
            include_decompressed_sha256=include_decompressed_sha256,
            chunk_bytes=chunk_bytes,
        )
        for path in paths
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit frozen Python RSTM byte-level reference records."
    )
    parser.add_argument("paths", nargs="+", help="RSTM files to inspect")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = build_reference_records(
        args.paths,
        header_bytes=args.header_bytes,
        include_compressed_sha256=args.compressed_sha256,
        include_decompressed_sha256=args.decompressed_sha256,
        chunk_bytes=args.chunk_bytes,
    )
    print(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
