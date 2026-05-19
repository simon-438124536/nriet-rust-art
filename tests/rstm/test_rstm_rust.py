import gzip
import importlib.util
import os
from pathlib import Path

import pytest

from tools.minhou_manifest import ENV_DATA_ROOT, iter_manifest_files
from tools.rstm_reference import build_reference_record


def _rust_rstm_header_preview():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.skip("pyart._rust is not installed")

    import pyart._rust as rust

    fn = getattr(rust, "_rstm_header_preview", None)
    if fn is None:
        pytest.skip("pyart._rust has not registered _rstm_header_preview yet")
    return fn


def _reference_key_fields(path: Path, header_bytes: int) -> dict[str, object]:
    record = build_reference_record(path, header_bytes=header_bytes)
    return {
        "gzip_detected_by_magic": record["gzip_detected_by_magic"],
        "raw_magic_hex": record["raw_magic_hex"],
        "header_preview": record["header_preview"],
    }


def test_rust_rstm_header_preview_matches_reference_for_synthetic_files(tmp_path):
    rust_header_preview = _rust_rstm_header_preview()

    plain_named_gz = tmp_path / "plain.rstm.gz"
    plain_named_gz.write_bytes(b"RSTM\x00plain\npayload")

    gzip_named_raw = tmp_path / "compressed_without_suffix.rstm"
    gzip_named_raw.write_bytes(gzip.compress(b"RSTM-HEADER-1234567890", mtime=0))

    short_file = tmp_path / "short.rstm"
    short_file.write_bytes(b"A")

    for path in (plain_named_gz, gzip_named_raw, short_file):
        for header_bytes in (0, 1, 8, 32, 256):
            assert rust_header_preview(str(path), header_bytes) == _reference_key_fields(
                path, header_bytes
            )


def test_rust_rstm_header_preview_accepts_pathlike_objects(tmp_path):
    rust_header_preview = _rust_rstm_header_preview()
    path = tmp_path / "pathlike.rstm"
    path.write_bytes(b"RSTM pathlike")

    assert rust_header_preview(path, 4) == _reference_key_fields(path, 4)


def test_rust_rstm_header_preview_rejects_negative_header_bytes(tmp_path):
    rust_header_preview = _rust_rstm_header_preview()
    path = tmp_path / "sample.rstm"
    path.write_bytes(b"RSTM")

    with pytest.raises(ValueError, match="header_bytes must be non-negative"):
        rust_header_preview(str(path), -1)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed-mode Rust parity is only checked with PYART_TEST_INSTALLED=1",
)
@pytest.mark.skipif(
    not os.environ.get(ENV_DATA_ROOT),
    reason=f"set {ENV_DATA_ROOT} to run real RSTM Rust/Python parity checks",
)
def test_installed_rust_rstm_header_preview_matches_reference_on_real_data():
    rust_header_preview = _rust_rstm_header_preview()
    data_root = Path(os.environ[ENV_DATA_ROOT])
    if not data_root.exists():
        pytest.skip(f"{ENV_DATA_ROOT} does not exist: {data_root}")

    files = iter_manifest_files(data_root, max_files=5)
    if not files:
        pytest.skip(f"{ENV_DATA_ROOT} has no files: {data_root}")

    for path in files:
        assert rust_header_preview(str(path), 64) == _reference_key_fields(path, 64)
