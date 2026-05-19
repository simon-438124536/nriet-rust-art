import numpy as np
import pytest

from pyart.io import nexrad_level3


SUB2_CODES = [94, 99, 153, 154, 155, 182, 186]


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _threshold_bytes(halfwords, extra=b""):
    return np.array(halfwords, dtype=">i2").tobytes() + extra


def _object_for_code(code, threshold_data, raw_data):
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    obj.msg_header = {"code": code}
    obj.prod_descr = {"threshold_data": threshold_data}
    obj.raw_data = raw_data
    return obj


def _fallback_get_data(code, threshold_data, raw_data, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return _object_for_code(code, threshold_data, raw_data).get_data()


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


@pytest.mark.parametrize(
    ("code", "threshold_data", "raw_data"),
    [
        (
            32,
            _threshold_bytes([10, 5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
        ),
        (
            32,
            _threshold_bytes([-10, -5]),
            np.array([[0, 1, 2, 65535]], dtype=np.uint16),
        ),
        (
            32,
            _threshold_bytes([10, 0], extra=b"\xff\xff"),
            np.empty((0, 4), dtype=np.uint16),
        ),
        (
            94,
            _threshold_bytes([10, 5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
        ),
        (
            186,
            _threshold_bytes([10, 5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint16),
        ),
        (
            153,
            _threshold_bytes([-10, 0], extra=b"\xff\xff"),
            np.empty((0, 4), dtype=np.uint8),
        ),
    ],
)
def test_msg32_and_scaled_sub2_python_fallback_reference_cases(
    monkeypatch, code, threshold_data, raw_data
):
    actual = _fallback_get_data(code, threshold_data, raw_data, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == np.ma.masked_array(np.zeros((1,), dtype=np.float32)).fill_value


@pytest.mark.parametrize(
    ("dtype", "kernel_name"),
    [
        (np.uint8, "_nexrad_level3_msg_32_u8"),
        (np.uint16, "_nexrad_level3_msg_32_u16"),
    ],
)
def test_msg32_dispatches_dense_unsigned_to_private_rust_kernel(
    monkeypatch, dtype, kernel_name
):
    calls = []
    data = np.array([[1.0, 1.5]], dtype=np.float32)
    mask = np.array([[True, False]], dtype=bool)

    def kernel(threshold_data, raw_data):
        calls.append((threshold_data, raw_data.dtype, raw_data.shape))
        return data.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )
    threshold_data = _threshold_bytes([10, 5])
    raw_data = np.array([[0, 1]], dtype=dtype)

    actual = _object_for_code(32, threshold_data, raw_data).get_data()

    assert calls == [(threshold_data, np.dtype(dtype), (1, 2))]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize("code", SUB2_CODES)
@pytest.mark.parametrize(
    ("dtype", "kernel_name"),
    [
        (np.uint8, "_nexrad_level3_msg_scaled_sub2_u8"),
        (np.uint16, "_nexrad_level3_msg_scaled_sub2_u16"),
    ],
)
def test_scaled_sub2_dispatches_all_codes_to_private_rust_kernel(
    monkeypatch, code, dtype, kernel_name
):
    calls = []
    data = np.array([[128.0, 1.0]], dtype=np.float32)
    mask = np.array([[True, False]], dtype=bool)

    def kernel(threshold_data, raw_data):
        calls.append((threshold_data, raw_data.dtype, raw_data.shape))
        return data.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )
    threshold_data = _threshold_bytes([10, 5])
    raw_data = np.array([[0, 2]], dtype=dtype)

    actual = _object_for_code(code, threshold_data, raw_data).get_data()

    assert calls == [(threshold_data, np.dtype(dtype), (1, 2))]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("code", "threshold_data", "raw_data"),
    [
        (32, _threshold_bytes([10]), np.array([[1]], dtype=np.uint8)),
        (94, bytearray(_threshold_bytes([10, 5])), np.array([[1]], dtype=np.uint8)),
        (32, _threshold_bytes([10, 5]), np.array([[1]], dtype=np.int16)),
        (94, _threshold_bytes([10, 5]), np.array([[1]], dtype=np.int64)),
        (32, _threshold_bytes([10, 5]), np.array([[1.0]], dtype=float)),
        (94, _threshold_bytes([10, 5]), np.array([0, 1, 2], dtype=np.uint8)),
        (
            32,
            _threshold_bytes([10, 5]),
            np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
        ),
    ],
)
def test_msg32_and_scaled_sub2_unsupported_inputs_keep_python_fallback(
    monkeypatch, code, threshold_data, raw_data
):
    def rust_kernel(name):
        if not name.startswith("_nexrad_level3_msg_32_") and not name.startswith(
            "_nexrad_level3_msg_scaled_sub2_"
        ):
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    obj = _object_for_code(code, threshold_data, raw_data)
    try:
        actual = obj.get_data()
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_get_data(code, threshold_data, raw_data, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_get_data(code, threshold_data, raw_data, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("code", "threshold_data", "raw_data"),
    [
        (
            32,
            _threshold_bytes([10, 5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
        ),
        (
            32,
            _threshold_bytes([-10, -5]),
            np.array([[0, 1, 2, 65535]], dtype=np.uint16),
        ),
        (
            32,
            _threshold_bytes([10, 0], extra=b"\xff\xff"),
            np.empty((0, 4), dtype=np.uint16),
        ),
        (
            94,
            _threshold_bytes([10, 5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
        ),
        (
            186,
            _threshold_bytes([10, 5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint16),
        ),
        (
            153,
            _threshold_bytes([-10, 0], extra=b"\xff\xff"),
            np.empty((0, 4), dtype=np.uint8),
        ),
    ],
)
def test_real_rust_msg32_and_scaled_sub2_match_python_fallback(
    monkeypatch, code, threshold_data, raw_data
):
    rust = _rust_or_skip()

    expected = _fallback_get_data(code, threshold_data, raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name.startswith("_nexrad_level3_msg_32_") or name.startswith(
            "_nexrad_level3_msg_scaled_sub2_"
        ):
            calls.append(name)
            return getattr(rust, name)
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_code(code, threshold_data, raw_data).get_data()

    assert calls
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize("code", SUB2_CODES)
@pytest.mark.parametrize(
    ("threshold_data", "raw_data"),
    [
        (
            _threshold_bytes([-10, -5]),
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
        ),
        (
            _threshold_bytes([10, 0], extra=b"\xff\xff"),
            np.array([[0, 1, 2, 255]], dtype=np.uint16),
        ),
    ],
)
def test_real_rust_scaled_sub2_matches_python_fallback_for_all_codes(
    monkeypatch, code, threshold_data, raw_data
):
    rust = _rust_or_skip()

    expected = _fallback_get_data(code, threshold_data, raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name.startswith("_nexrad_level3_msg_scaled_sub2_"):
            calls.append(name)
            return getattr(rust, name)
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_code(code, threshold_data, raw_data).get_data()

    assert calls
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("func_name", "raw_data"),
    [
        ("_nexrad_level3_msg_32_u8", np.array([[1]], dtype=np.uint8)),
        ("_nexrad_level3_msg_32_u16", np.array([[1]], dtype=np.uint16)),
        ("_nexrad_level3_msg_scaled_sub2_u8", np.array([[1]], dtype=np.uint8)),
        ("_nexrad_level3_msg_scaled_sub2_u16", np.array([[1]], dtype=np.uint16)),
    ],
)
def test_real_rust_msg32_and_scaled_sub2_direct_reject_short_threshold(
    func_name, raw_data
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match="threshold_data"):
        getattr(rust, func_name)(_threshold_bytes([10]), raw_data)


@pytest.mark.parametrize(
    ("func_name", "raw_data"),
    [
        (
            "_nexrad_level3_msg_32_u8",
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
        ),
        (
            "_nexrad_level3_msg_32_u16",
            np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
        ),
        (
            "_nexrad_level3_msg_scaled_sub2_u8",
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
        ),
        (
            "_nexrad_level3_msg_scaled_sub2_u16",
            np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
        ),
    ],
)
def test_real_rust_msg32_and_scaled_sub2_direct_reject_noncontiguous(
    func_name, raw_data
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match="C-contiguous"):
        getattr(rust, func_name)(_threshold_bytes([10, 5]), raw_data)


@pytest.mark.parametrize(
    ("func_name", "threshold_data", "raw_data"),
    [
        ("_nexrad_level3_msg_32_u8", bytearray(_threshold_bytes([10, 5])), np.array([[1]], dtype=np.uint8)),
        ("_nexrad_level3_msg_32_u8", _threshold_bytes([10, 5]), np.array([[1]], dtype=np.uint16)),
        ("_nexrad_level3_msg_32_u16", _threshold_bytes([10, 5]), np.array([[1]], dtype=np.uint8)),
        ("_nexrad_level3_msg_scaled_sub2_u8", _threshold_bytes([10, 5]), np.array([1], dtype=np.uint8)),
        ("_nexrad_level3_msg_scaled_sub2_u16", _threshold_bytes([10, 5]), np.array([1], dtype=np.uint16)),
    ],
)
def test_real_rust_msg32_and_scaled_sub2_direct_reject_binding_type_drift(
    func_name, threshold_data, raw_data
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        getattr(rust, func_name)(threshold_data, raw_data)


def test_real_rust_msg32_and_scaled_sub2_direct_return_data_and_mask():
    rust = _rust_or_skip()

    msg32_data, msg32_mask = rust._nexrad_level3_msg_32_u16(
        _threshold_bytes([10, 5]),
        np.array([[0, 1, 2, 255]], dtype=np.uint16),
    )
    sub2_u8_data, sub2_u8_mask = rust._nexrad_level3_msg_scaled_sub2_u8(
        _threshold_bytes([10, 5]),
        np.array([[0, 1, 2, 255]], dtype=np.uint8),
    )
    sub2_u16_data, sub2_u16_mask = rust._nexrad_level3_msg_scaled_sub2_u16(
        _threshold_bytes([10, 5]),
        np.array([[0, 1, 2, 255]], dtype=np.uint16),
    )

    assert msg32_data.dtype == np.float32
    assert sub2_u8_data.dtype == np.float32
    assert sub2_u16_data.dtype == np.float32
    assert msg32_mask.dtype == np.bool_
    np.testing.assert_array_equal(
        msg32_data,
        np.array([[1.0, 1.5, 2.0, 128.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sub2_u8_data,
        np.array([[128.0, 128.5, 1.0, 127.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sub2_u16_data,
        np.array([[32768.0, 32768.5, 1.0, 127.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        msg32_mask,
        np.array([[True, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(sub2_u8_mask, msg32_mask)
    np.testing.assert_array_equal(sub2_u16_mask, msg32_mask)
