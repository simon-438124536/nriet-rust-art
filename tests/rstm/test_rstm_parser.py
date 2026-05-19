"""Parity between frozen Python RSTM parser and pyart._rust helpers."""

from __future__ import annotations

import gzip
import importlib.util
import struct
from pathlib import Path

import pytest

from tools.rstm_parser import parse_file, parse_file_header, parse_logical_payload


def _rust_parse_file_header():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.skip("pyart._rust is not installed")
    import pyart._rust as rust

    fn = getattr(rust, "_rstm_parse_file_header", None)
    if fn is None:
        pytest.skip("pyart._rust has not registered _rstm_parse_file_header yet")
    return fn


def _rust_parse_logical_payload():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.skip("pyart._rust is not installed")
    import pyart._rust as rust

    fn = getattr(rust, "_rstm_parse_logical_payload", None)
    if fn is None:
        pytest.skip("pyart._rust has not registered _rstm_parse_logical_payload yet")
    return fn


def _rust_parse_file():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.skip("pyart._rust is not installed")
    import pyart._rust as rust

    fn = getattr(rust, "_rstm_parse_file", None)
    if fn is None:
        pytest.skip("pyart._rust has not registered _rstm_parse_file yet")
    return fn


def _header_key_fields(header_dict: dict) -> dict[str, object]:
    return {
        "magic": header_dict["magic"],
        "version_major": header_dict["version_major"],
        "version_minor": header_dict["version_minor"],
        "header_words": header_dict["header_words"],
        "site_id": header_dict["site_id"],
        "site_name": header_dict["site_name"],
        "latitude": header_dict["latitude"],
        "longitude": header_dict["longitude"],
        "altitude_m": header_dict["altitude_m"],
        "nrays": header_dict["nrays"],
        "ngates": header_dict["ngates"],
        "scan_mode": header_dict["scan_mode"],
        "product_desc": header_dict["product_desc"],
    }


def _synthetic_logical_payload(nrays: int = 4, ngates: int = 8) -> bytes:
    header = bytearray(512)
    header[:4] = b"RSTM"
    header[4:10] = struct.pack("<3H", 1, 1, 1)
    header[32:40] = b"ZF001\x00\x00\x00"
    header[40:56] = b"SITE-01\x00".ljust(16, b"\x00")
    header[0x40:0x4C] = struct.pack("<3f", 0.0, 0.0, 120.0)
    header[0x48:0x50] = struct.pack("<2f", 26.5, 119.0)
    header[0x50:0x58] = struct.pack("<2I", nrays, ngates)
    header[0xA0:0xB0] = b"VCP21D\x00".ljust(16, b"\x00")
    header[0xC0:0xE0] = "测试产品".encode("utf-8").ljust(32, b"\x00")[:32]
    ray_bytes = bytearray(nrays * 16)
    for index in range(nrays):
        ray_bytes[index * 16] = index
    return bytes(header) + bytes(ray_bytes)


def test_python_parser_synthetic_roundtrip():
    data = _synthetic_logical_payload()
    parsed = parse_logical_payload(data)
    assert parsed.header.site_id == "ZF001"
    assert parsed.header.nrays == 4
    assert parsed.header.ngates == 8
    assert parsed.ray_stride == 16
    assert len(parsed.rays) == 4
    assert parsed.rays[0].payload[0] == 0
    assert parsed.rays[-1].index == 3
    assert parsed.rays[-1].payload[0] == 3


def test_rust_parse_file_header_matches_python():
    rust_header = _rust_parse_file_header()
    data = _synthetic_logical_payload()
    py_header = parse_file_header(data).to_dict()
    rust_out = rust_header(data)
    assert _header_key_fields(rust_out) == _header_key_fields(py_header)


def test_rust_parse_logical_payload_matches_python():
    rust_parse = _rust_parse_logical_payload()
    data = _synthetic_logical_payload(nrays=3, ngates=5)
    py_parsed = parse_logical_payload(data)
    rust_out = rust_parse(data)

    assert rust_out["logical_size_bytes"] == py_parsed.logical_size_bytes
    assert rust_out["ray_data_offset"] == py_parsed.ray_data_offset
    assert rust_out["ray_stride"] == py_parsed.ray_stride
    assert rust_out["ray_count"] == len(py_parsed.rays)
    assert _header_key_fields(rust_out["header"]) == _header_key_fields(
        py_parsed.header.to_dict()
    )

    for rust_ray, py_ray in zip(rust_out["rays"], py_parsed.rays, strict=True):
        assert rust_ray["index"] == py_ray.index
        assert rust_ray["offset"] == py_ray.offset
        assert rust_ray["size"] == py_ray.size
        assert rust_ray["ngates"] == py_ray.ngates
        assert rust_ray["payload_hex_prefix"] == py_ray.payload[:32].hex()
        assert rust_ray["payload_ascii_prefix"] == "".join(
            chr(byte) if 32 <= byte <= 126 else "."
            for byte in py_ray.payload[:32]
        )


def test_rust_parse_file_matches_python_on_disk(tmp_path):
    rust_parse_file = _rust_parse_file()
    path = tmp_path / "sample.rstm.gz"
    logical = _synthetic_logical_payload()
    path.write_bytes(gzip.compress(logical, mtime=0))

    py_parsed = parse_file(path)
    rust_out = rust_parse_file(str(path))
    assert rust_out["ray_count"] == len(py_parsed.rays)
    assert rust_out["ray_stride"] == py_parsed.ray_stride
