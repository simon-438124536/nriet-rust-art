import warnings

import numpy as np
import pytest

from pyart.io import mdv_common


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _fallback_decode_rle8(compr_data, key, decompr_size, monkeypatch):
    monkeypatch.setattr(mdv_common, "_rust_kernel", lambda _name: None)
    return mdv_common._decode_rle8(compr_data, key, decompr_size)


@pytest.mark.parametrize(
    ("compr_data", "key", "decompr_size", "expected"),
    [
        (b"abc", 255, 3, b"abc"),
        (bytes([255, 3, 7]), 255, 3, bytes([7, 7, 7])),
        (bytes([1, 255, 3, 7, 2]), 255, 5, bytes([1, 7, 7, 7, 2])),
        (bytes([255, 0, 9, 4]), 255, 1, bytes([4])),
    ],
)
def test_decode_rle8_python_fallback_reference_cases(
    monkeypatch, compr_data, key, decompr_size, expected
):
    actual = _fallback_decode_rle8(compr_data, key, decompr_size, monkeypatch)

    assert type(actual) is bytes
    assert actual == expected


def test_decode_rle8_dispatches_exact_bytes_to_private_rust_kernel(monkeypatch):
    calls = []

    def kernel(compr_data, key, decompr_size):
        calls.append((compr_data, key, decompr_size))
        return b"rust"

    monkeypatch.setattr(
        mdv_common,
        "_rust_kernel",
        lambda name: kernel if name == "_mdv_decode_rle8" else None,
    )

    actual = mdv_common._decode_rle8(
        bytes([1, 255, 2, 8]), np.uint8(255), np.int64(3)
    )

    assert actual == b"rust"
    assert calls == [(bytes([1, 255, 2, 8]), 255, 3)]


def test_decode_rle8_dispatches_long_all_literal_bytes_to_private_rust_kernel(
    monkeypatch,
):
    payload = bytes((index % 255 for index in range(300)))
    calls = []

    def kernel(compr_data, key, decompr_size):
        calls.append((len(compr_data), key, decompr_size))
        return b"rust"

    monkeypatch.setattr(
        mdv_common,
        "_rust_kernel",
        lambda name: kernel if name == "_mdv_decode_rle8" else None,
    )

    actual = mdv_common._decode_rle8(payload, 255, 300)

    assert actual == b"rust"
    assert calls == [(300, 255, 300)]


@pytest.mark.parametrize(
    ("compr_data", "key", "decompr_size"),
    [
        (bytearray(b"abc"), 255, 3),
        (memoryview(b"abc"), 255, 3),
        (bytes([255, 1, 2]), 999, 3),
        (bytes([255, 1, 2]), -1, 3),
        (bytes([255, 1, 2]), 255.0, 1),
    ],
)
def test_decode_rle8_unsupported_inputs_keep_python_fallback(
    monkeypatch, compr_data, key, decompr_size
):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(mdv_common, "_rust_kernel", rust_kernel)
    actual = mdv_common._decode_rle8(compr_data, key, decompr_size)
    expected = _fallback_decode_rle8(compr_data, key, decompr_size, monkeypatch)

    assert actual == expected


@pytest.mark.parametrize(
    ("compr_data", "key", "decompr_size"),
    [
        (bytes([1, 255]), 255, 4),
        (bytes([1, 255, 2]), 255, 4),
        (b"abcd", 255, 2),
        (bytes([255, 4, 9]), 255, 2),
        (b"ab", 255, -1),
        (b"ab", 255, 2.0),
        (bytes([255, 1, 2]), True, 3),
    ],
)
def test_decode_rle8_exceptional_or_overflow_cases_keep_python_surface(
    monkeypatch, compr_data, key, decompr_size
):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"exceptional input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(mdv_common, "_rust_kernel", rust_kernel)
    try:
        actual = mdv_common._decode_rle8(compr_data, key, decompr_size)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_decode_rle8(compr_data, key, decompr_size, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_decode_rle8(compr_data, key, decompr_size, monkeypatch)
        assert actual == expected


def test_decode_rle8_underfilled_output_remains_python_owned(monkeypatch):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"underfilled input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(mdv_common, "_rust_kernel", rust_kernel)

    actual = mdv_common._decode_rle8(b"ab", 255, 5)

    assert type(actual) is bytes
    assert len(actual) == 5
    assert actual[:2] == b"ab"


def test_decode_rle8_run_outputs_over_uint8_pointer_range_remain_python_owned(
    monkeypatch,
):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"large run input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(mdv_common, "_rust_kernel", rust_kernel)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        actual = mdv_common._decode_rle8(bytes([255, 255, 7, 8]), 255, 256)
        expected = _fallback_decode_rle8(
            bytes([255, 255, 7, 8]), 255, 256, monkeypatch
        )

    assert actual == expected


@pytest.mark.parametrize(
    ("compr_data", "key", "decompr_size"),
    [
        (b"abc", 255, 3),
        (bytes((index % 255 for index in range(300))), 255, 300),
        (bytes([255, 3, 7]), 255, 3),
        (bytes([1, 255, 3, 7, 2]), 255, 5),
        (bytes([255, 0, 9, 4]), 255, 1),
    ],
)
def test_real_rust_mdv_decode_rle8_matches_python_fallback(
    monkeypatch, compr_data, key, decompr_size
):
    rust = _rust_or_skip()

    expected = _fallback_decode_rle8(compr_data, key, decompr_size, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_mdv_decode_rle8":
            calls.append(name)
            return rust._mdv_decode_rle8
        return None

    monkeypatch.setattr(mdv_common, "_rust_kernel", rust_kernel)
    actual = mdv_common._decode_rle8(compr_data, key, decompr_size)

    assert calls == ["_mdv_decode_rle8"]
    assert actual == expected


@pytest.mark.parametrize(
    ("compr_data", "key", "decompr_size", "match"),
    [
        (bytes([255]), 255, 1, "truncated"),
        (b"abcd", 255, 2, "exceeds"),
        (bytes([255, 4, 9]), 255, 2, "exceeds"),
        (bytes([255, 255, 7, 8]), 255, 256, "pointer range"),
        (b"ab", 255, 5, "does not match"),
        (b"x", 255, 512 * 1024 * 1024 + 1, "maximum"),
    ],
)
def test_real_rust_mdv_decode_rle8_direct_rejects_malformed_inputs(
    compr_data, key, decompr_size, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._mdv_decode_rle8(compr_data, key, decompr_size)


@pytest.mark.parametrize(
    ("key", "decompr_size", "match"),
    [
        (True, 1, "key"),
        (-1, 1, "key"),
        (256, 1, "key"),
        (255, True, "decompr_size"),
        (255, -1, "decompr_size"),
    ],
)
def test_real_rust_mdv_decode_rle8_direct_rejects_bad_args(
    key, decompr_size, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._mdv_decode_rle8(b"x", key, decompr_size)
