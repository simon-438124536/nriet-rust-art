import gzip
import importlib.util
import os

import pytest


def test_native_extension_available_after_install():
    if importlib.util.find_spec("pyart._rust") is None:
        if os.environ.get("PYART_ALLOW_MISSING_RUST") == "1":
            pytest.skip("source-only run explicitly allowed missing pyart._rust")
        pytest.fail(
            "pyart._rust is required for installed-package validation; set "
            "PYART_ALLOW_MISSING_RUST=1 only for source-only smoke tests"
        )

    import pyart._rust as rust

    assert rust.rust_backend_ready() is True
    assert rust.version() == "0.1.0"
    assert rust.sum_f64([1.0, 2.5, 3.5]) == 7.0
    assert rust.is_gzip_magic(b"\x1f\x8brest")


def test_native_gzip_helpers_are_bounded():
    if importlib.util.find_spec("pyart._rust") is None:
        if os.environ.get("PYART_ALLOW_MISSING_RUST") == "1":
            pytest.skip("source-only run explicitly allowed missing pyart._rust")
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    payload = b"RSTM" + (b"x" * 32)
    compressed = gzip.compress(payload, mtime=0)

    assert rust.gzip_decompressed_len(compressed, 64) == len(payload)
    assert rust.gzip_decompress(compressed, 64) == payload

    with pytest.raises(ValueError, match="max_uncompressed_bytes"):
        rust.gzip_decompressed_len(compressed, 4)
    with pytest.raises(ValueError, match="max_uncompressed_bytes"):
        rust.gzip_decompress(compressed, 4)
