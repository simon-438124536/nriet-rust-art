import numpy as np
import pytest

from pyart.io import nexrad_level3


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _threshold_bytes(flags=None, values=None, extra=b""):
    if flags is None:
        flags = np.zeros(16, dtype=np.uint8)
    if values is None:
        values = np.arange(16, dtype=np.uint8)
    return bytes(np.column_stack([flags, values]).ravel()) + extra


def _object_for_levels(threshold_data, raw_data):
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    obj.prod_descr = {"threshold_data": threshold_data}
    obj.raw_data = raw_data
    return obj


def _fallback_levels(threshold_data, raw_data, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return _object_for_levels(threshold_data, raw_data)._get_data_8_or_16_levels()


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
        (_threshold_bytes(), np.array([[0, 1, 15]], dtype=np.uint8)),
        (
            _threshold_bytes(
                flags=np.array(
                    [0x20, 0x01, 0x80] + [0] * 13,
                    dtype=np.uint8,
                ),
                values=np.array([0, 10, 20] + list(range(3, 16)), dtype=np.uint8),
            ),
            np.array([[1, 2, 3]], dtype=np.uint8),
        ),
        (
            _threshold_bytes(
                flags=np.array([0x30] + [0] * 15, dtype=np.uint8),
                values=np.arange(16, dtype=np.uint8),
            ),
            np.array([[1, 2]], dtype=np.uint8),
        ),
        (_threshold_bytes(extra=b"\x00\xff"), np.array([[0, 3]], dtype=np.uint8)),
        (_threshold_bytes(), np.empty((0, 4), dtype=np.uint8)),
    ],
)
def test_data_8_or_16_python_fallback_reference_cases(
    monkeypatch, threshold_data, raw_data
):
    actual = _fallback_levels(threshold_data, raw_data, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float64
    assert actual.fill_value == -999.0


def test_data_8_or_16_dispatches_dense_u8_to_private_rust_kernel(monkeypatch):
    calls = []
    data = np.array([[0.0, -999.0]], dtype=np.float64)
    mask = np.array([[False, True]], dtype=bool)

    def kernel(threshold_data, raw_data):
        calls.append((threshold_data, raw_data.dtype, raw_data.shape))
        return data.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level3_data_8_or_16_u8" else None,
    )
    threshold_data = _threshold_bytes()
    raw_data = np.array([[0, 1]], dtype=np.uint8)

    actual = _object_for_levels(threshold_data, raw_data)._get_data_8_or_16_levels()

    assert calls == [(threshold_data, np.dtype(np.uint8), (1, 2))]
    assert actual.dtype == np.float64
    assert actual.fill_value == -999.0
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("threshold_data", "raw_data"),
    [
        (b"\x00\x01", np.array([[0, 1]], dtype=np.uint8)),
        (_threshold_bytes(), np.array([[0, 1]], dtype=np.int16)),
        (_threshold_bytes(), np.array([[0, 1]], dtype=np.uint16)),
        (_threshold_bytes(), np.array([[0, 1]], dtype=np.int64)),
        (_threshold_bytes(), np.array([[0, 1]], dtype=">i2")),
        (_threshold_bytes(), np.array([[False, True]], dtype=bool)),
        (_threshold_bytes(), np.array([0, 1], dtype=np.uint8)),
        (_threshold_bytes(), np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2]),
        (_threshold_bytes(), np.array([[16]], dtype=np.uint8)),
    ],
)
def test_data_8_or_16_unsupported_inputs_keep_python_fallback(
    monkeypatch, threshold_data, raw_data
):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"unsupported data-level input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    obj = _object_for_levels(threshold_data, raw_data)
    try:
        actual = obj._get_data_8_or_16_levels()
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_levels(threshold_data, raw_data, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_levels(threshold_data, raw_data, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("threshold_data", "raw_data"),
    [
        (_threshold_bytes(), np.array([[0, 1, 15]], dtype=np.uint8)),
        (
            _threshold_bytes(
                flags=np.array([0x20, 0x01, 0x80] + [0] * 13, dtype=np.uint8),
                values=np.array([0, 10, 20] + list(range(3, 16)), dtype=np.uint8),
            ),
            np.array([[1, 2, 3]], dtype=np.uint8),
        ),
        (
            _threshold_bytes(flags=np.array([0x30] + [0] * 15, dtype=np.uint8)),
            np.array([[1, 2]], dtype=np.uint8),
        ),
        (_threshold_bytes(extra=b"\x00\xff"), np.array([[0, 3]], dtype=np.uint8)),
        (_threshold_bytes(), np.empty((0, 4), dtype=np.uint8)),
    ],
)
def test_real_rust_data_8_or_16_matches_python_fallback(
    monkeypatch, threshold_data, raw_data
):
    rust = _rust_or_skip()

    expected = _fallback_levels(threshold_data, raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_nexrad_level3_data_8_or_16_u8":
            calls.append(name)
            return rust._nexrad_level3_data_8_or_16_u8
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_levels(threshold_data, raw_data)._get_data_8_or_16_levels()

    assert calls == ["_nexrad_level3_data_8_or_16_u8"]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("threshold_data", "raw_data", "match"),
    [
        (b"\x00\x01", np.array([[0]], dtype=np.uint8), "threshold_data"),
        (_threshold_bytes(), np.array([[16]], dtype=np.uint8), "invalid entry"),
        (
            _threshold_bytes(),
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
            "C-contiguous",
        ),
    ],
)
def test_real_rust_data_8_or_16_direct_rejects_unsafe_inputs(
    threshold_data, raw_data, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._nexrad_level3_data_8_or_16_u8(threshold_data, raw_data)


def test_real_rust_data_8_or_16_direct_returns_data_and_mask():
    rust = _rust_or_skip()
    flags = np.array([0, 0x80] + [0] * 14, dtype=np.uint8)
    values = np.array([3, 4] + list(range(2, 16)), dtype=np.uint8)

    data, mask = rust._nexrad_level3_data_8_or_16_u8(
        _threshold_bytes(flags=flags, values=values),
        np.array([[0, 1]], dtype=np.uint8),
    )

    assert data.dtype == np.float64
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(data, np.array([[3.0, -999.0]]))
    np.testing.assert_array_equal(mask, np.array([[False, True]]))
