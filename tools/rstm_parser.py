"""Frozen Python RSTM CAP_FMT logical-file parser (header + ray records).

This module extends the byte-level inventory helpers in ``rstm_reference`` with
operational layout parsing discovered from the MinHou comparison corpus.  The
parser is the oracle for the Rust ``pyart._rust`` helpers.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

from tools.rstm_reference import (
    GZIP_MAGIC,
    SCHEMA_VERSION,
    build_reference_record,
    is_gzip_file,
    iter_logical_bytes,
)

RSTM_MAGIC = b"RSTM"
FILE_HEADER_SIZE = 256
METADATA_BLOCK_SIZE = 256


@dataclass(frozen=True)
class RstmFileHeader:
    """Parsed 256-byte RSTM/CAP_FMT file header."""

    magic: bytes
    version_major: int
    version_minor: int
    header_words: int
    reserved_a: int
    reserved_b: int
    site_id: str
    site_name: str
    latitude: float
    longitude: float
    altitude_m: float
    nrays: int
    ngates: int
    scan_mode: str
    product_desc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "magic": self.magic.decode("ascii", errors="replace"),
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "header_words": self.header_words,
            "reserved_a": self.reserved_a,
            "reserved_b": self.reserved_b,
            "site_id": self.site_id,
            "site_name": self.site_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "nrays": self.nrays,
            "ngates": self.ngates,
            "scan_mode": self.scan_mode,
            "product_desc": self.product_desc,
            "file_header_size": FILE_HEADER_SIZE,
        }


@dataclass(frozen=True)
class RstmRayRecord:
    """One fixed-size ray slot in the logical payload."""

    index: int
    offset: int
    size: int
    ngates: int
    payload: bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "offset": self.offset,
            "size": self.size,
            "ngates": self.ngates,
            "payload_hex_prefix": self.payload[:32].hex(),
            "payload_ascii_prefix": _ascii_preview(self.payload[:32]),
        }


@dataclass(frozen=True)
class RstmParsedFile:
    """Logical RSTM file: frozen header plus ray records."""

    header: RstmFileHeader
    logical_size_bytes: int
    ray_data_offset: int
    ray_stride: int
    rays: tuple[RstmRayRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "header": self.header.to_dict(),
            "logical_size_bytes": self.logical_size_bytes,
            "ray_data_offset": self.ray_data_offset,
            "ray_stride": self.ray_stride,
            "ray_count": len(self.rays),
            "rays": [ray.to_dict() for ray in self.rays],
        }


def _ascii_preview(data: bytes) -> str:
    return "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in data)


def _c_string(data: bytes) -> str:
    return data.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def read_logical_bytes(path: str | Path) -> bytes:
    """Return the full logical payload bytes for one on-disk file."""

    chunks = list(iter_logical_bytes(path))
    return b"".join(chunks)


def parse_file_header(data: bytes) -> RstmFileHeader:
    """Parse the frozen 256-byte RSTM file header."""

    if len(data) < FILE_HEADER_SIZE:
        raise ValueError("RSTM logical payload shorter than file header")

    magic = data[:4]
    if magic != RSTM_MAGIC:
        raise ValueError(f"expected RSTM magic, found {magic!r}")

    version_major, version_minor, header_words = struct.unpack_from("<3H", data, 4)
    reserved_a, reserved_b = struct.unpack_from("<2I", data, 8)
    site_id = _c_string(data[32:40])
    site_name = _c_string(data[40:56])
    _pad0, _pad1, altitude_m = struct.unpack_from("<3f", data, 0x40)
    latitude, longitude = struct.unpack_from("<2f", data, 0x48)
    nrays, ngates = struct.unpack_from("<2I", data, 0x50)
    scan_mode = _c_string(data[0xA0:0xB0])
    product_desc = data[0xC0:0xE0].split(b"\x00", 1)[0].decode(
        "utf-8", errors="replace"
    )

    if nrays <= 0 or ngates <= 0:
        raise ValueError("nrays and ngates must be positive")

    return RstmFileHeader(
        magic=magic,
        version_major=version_major,
        version_minor=version_minor,
        header_words=header_words,
        reserved_a=reserved_a,
        reserved_b=reserved_b,
        site_id=site_id,
        site_name=site_name,
        latitude=float(latitude),
        longitude=float(longitude),
        altitude_m=float(altitude_m),
        nrays=int(nrays),
        ngates=int(ngates),
        scan_mode=scan_mode,
        product_desc=product_desc,
    )


def iter_ray_records(
    data: bytes,
    header: RstmFileHeader,
    *,
    ray_data_offset: int = FILE_HEADER_SIZE + METADATA_BLOCK_SIZE,
) -> Iterator[RstmRayRecord]:
    """Yield fixed-size ray records from the logical payload."""

    if len(data) < ray_data_offset:
        raise ValueError("logical payload shorter than ray data offset")

    payload_bytes = len(data) - ray_data_offset
    if payload_bytes <= 0:
        return

    if header.nrays <= 0:
        raise ValueError("header.nrays must be positive")

    ray_stride, remainder = divmod(payload_bytes, header.nrays)
    if ray_stride <= 0:
        raise ValueError("computed ray stride must be positive")

    for index in range(header.nrays):
        offset = ray_data_offset + index * ray_stride
        end = offset + ray_stride
        if index == header.nrays - 1:
            end = len(data)
        payload = data[offset:end]
        yield RstmRayRecord(
            index=index,
            offset=offset,
            size=end - offset,
            ngates=header.ngates,
            payload=payload,
        )


def parse_logical_payload(data: bytes) -> RstmParsedFile:
    """Parse a full logical RSTM payload."""

    header = parse_file_header(data)
    ray_data_offset = FILE_HEADER_SIZE + METADATA_BLOCK_SIZE
    rays = tuple(iter_ray_records(data, header, ray_data_offset=ray_data_offset))
    payload_bytes = len(data) - ray_data_offset
    ray_stride, _ = divmod(payload_bytes, header.nrays)
    return RstmParsedFile(
        header=header,
        logical_size_bytes=len(data),
        ray_data_offset=ray_data_offset,
        ray_stride=ray_stride,
        rays=rays,
    )


def parse_file(path: str | Path) -> RstmParsedFile:
    """Parse one on-disk RSTM file."""

    return parse_logical_payload(read_logical_bytes(path))


def build_inventory_record(
    path: str | Path,
    *,
    header_bytes: int = 256,
    include_ray_summary: bool = True,
    max_ray_summaries: int = 3,
) -> dict[str, object]:
    """Combine reference inventory fields with parsed header/ray summaries."""

    record = build_reference_record(path, header_bytes=header_bytes)
    parsed = parse_file(path)
    record["parsed_header"] = parsed.header.to_dict()
    if include_ray_summary:
        record["ray_stride"] = parsed.ray_stride
        record["ray_data_offset"] = parsed.ray_data_offset
        record["ray_summaries"] = [
            ray.to_dict() for ray in parsed.rays[: max(0, max_ray_summaries)]
        ]
    return record
