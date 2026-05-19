import warnings

import numpy as np
import pytest

from pyart.io import nexrad_level3


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _threshold_bytes(halfwords, extra=b""):
    return np.array(halfwords, dtype=">i2").tobytes() + extra


def _object_for_msg134(threshold_data, raw_data):
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    obj.prod_descr = {"threshold_data": threshold_data}
    obj.raw_data = raw_data
    return obj


def _fallback_msg134(threshold_data, raw_data, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return _object_for_msg134(threshold_data, raw_data)._get_data_msg_134()


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


@pytest.mark.parametrize(
    ("threshold_data", "raw_data"),
    [
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
            np.array([[0, 2, 5, 10, 12]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, -1, 0x4000, 0]),
            np.array([[0, 2, 5]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, 300, 0x4000, 0]),
            np.array([[2, 10, 255]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([-31744, 0, 10, 0x4000, 0]),
            np.array([[2, 10]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0], extra=b"\xff\xff"),
            np.array([[2, 10]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
            np.empty((0, 3), dtype=np.uint8),
        ),
    ],
)
def test_msg134_python_fallback_reference_cases(
    monkeypatch, threshold_data, raw_data
):
    actual = _fallback_msg134(threshold_data, raw_data, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == np.ma.masked_array(np.zeros((1,), dtype=np.float32)).fill_value


def test_msg134_dispatches_dense_u8_to_private_rust_kernel(monkeypatch):
    calls = []
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    mask = np.array([[True, False]], dtype=bool)

    def kernel(threshold_data, raw_data):
        calls.append((threshold_data, raw_data.dtype, raw_data.shape))
        return data.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level3_msg_134_u8" else None,
    )
    threshold_data = _threshold_bytes([0x3C00, 0, 10, 0x4000, 0])
    raw_data = np.array([[0, 10]], dtype=np.uint8)

    actual = _object_for_msg134(threshold_data, raw_data)._get_data_msg_134()

    assert calls == [(threshold_data, np.dtype(np.uint8), (1, 2))]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("threshold_data", "raw_data"),
    [
        (_threshold_bytes([0x3C00, 0]), np.array([[2]], dtype=np.uint8)),
        (_threshold_bytes([0, 0, 10, 0x4000, 0]), np.array([[2]], dtype=np.uint8)),
        (_threshold_bytes([0x3C00, 0, 10, 0, 0]), np.array([[10]], dtype=np.uint8)),
        (_threshold_bytes([0x3C00, 0, 10, 0x4000, 0]), np.array([[2]], dtype=np.int64)),
        (_threshold_bytes([0x3C00, 0, 10, 0x4000, 0]), np.array([[2.0]], dtype=float)),
        (_threshold_bytes([0x3C00, 0, 10, 0x4000, 0]), np.array([2], dtype=np.uint8)),
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
        ),
    ],
)
def test_msg134_unsupported_inputs_keep_python_fallback(
    monkeypatch, threshold_data, raw_data
):
    def rust_kernel(name):
        if name != "_nexrad_level3_msg_134_u8":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported msg134 input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    obj = _object_for_msg134(threshold_data, raw_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            actual = obj._get_data_msg_134()
        except Exception as actual_error:
            with pytest.raises(type(actual_error)) as expected_error:
                _fallback_msg134(threshold_data, raw_data, monkeypatch)
            assert actual_error.args == expected_error.value.args
        else:
            expected = _fallback_msg134(threshold_data, raw_data, monkeypatch)
            _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("threshold_data", "raw_data"),
    [
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
            np.array([[0, 2, 5, 10, 12]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, -1, 0x4000, 0]),
            np.array([[0, 2, 5]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, 300, 0x4000, 0]),
            np.array([[2, 10, 255]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([-31744, 0, 10, 0x4000, 0]),
            np.array([[2, 10]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0], extra=b"\xff\xff"),
            np.array([[2, 10]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
            np.empty((0, 3), dtype=np.uint8),
        ),
    ],
)
def test_real_rust_msg134_matches_python_fallback(monkeypatch, threshold_data, raw_data):
    rust = _rust_or_skip()

    expected = _fallback_msg134(threshold_data, raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_nexrad_level3_msg_134_u8":
            calls.append(name)
            return rust._nexrad_level3_msg_134_u8
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_msg134(threshold_data, raw_data)._get_data_msg_134()

    assert calls == ["_nexrad_level3_msg_134_u8"]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("threshold_data", "raw_data", "match"),
    [
        (_threshold_bytes([0x3C00, 0]), np.array([[2]], dtype=np.uint8), "threshold_data"),
        (
            _threshold_bytes([0, 0, 10, 0x4000, 0]),
            np.array([[2]], dtype=np.uint8),
            "non-zero",
        ),
        (
            _threshold_bytes([0x3C00, 0, 10, 0, 0]),
            np.array([[10]], dtype=np.uint8),
            "non-zero",
        ),
        (
            _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
            "C-contiguous",
        ),
    ],
)
def test_real_rust_msg134_direct_rejects_unsafe_inputs(
    threshold_data, raw_data, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._nexrad_level3_msg_134_u8(threshold_data, raw_data)


def test_real_rust_msg134_direct_returns_data_and_mask():
    rust = _rust_or_skip()

    data, mask = rust._nexrad_level3_msg_134_u8(
        _threshold_bytes([0x3C00, 0, 10, 0x4000, 0]),
        np.array([[0, 2, 10]], dtype=np.uint8),
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.array([[True, False, False]]))
