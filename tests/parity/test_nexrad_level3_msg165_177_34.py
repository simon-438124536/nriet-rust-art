import numpy as np
import pytest

from pyart.io import nexrad_level3


MASK_ZERO_CODES = [165, 177]


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _object_for_code(code, raw_data):
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    obj.msg_header = {"code": code}
    obj.prod_descr = {"threshold_data": b""}
    obj.raw_data = raw_data
    return obj


def _fallback_get_data(code, raw_data, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return _object_for_code(code, raw_data).get_data()


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


@pytest.mark.parametrize(
    ("code", "raw_data"),
    [
        (165, np.array([[0, 1, 255]], dtype=np.uint8)),
        (177, np.array([[0, 1, 65535]], dtype=np.uint16)),
        (165, np.empty((0, 4), dtype=np.uint8)),
        (34, np.array([[0, 1, 255]], dtype=np.uint8)),
        (34, np.array([[0, 1, 65535]], dtype=np.uint16)),
        (34, np.empty((0, 4), dtype=np.uint16)),
    ],
)
def test_msg165_177_34_python_fallback_reference_cases(monkeypatch, code, raw_data):
    actual = _fallback_get_data(code, raw_data, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    if code in MASK_ZERO_CODES:
        assert actual.fill_value == 0.0
    else:
        assert actual.fill_value == np.ma.masked_array(np.zeros((1,), dtype=np.float32)).fill_value


@pytest.mark.parametrize("code", MASK_ZERO_CODES)
@pytest.mark.parametrize(
    ("dtype", "kernel_name"),
    [
        (np.uint8, "_nexrad_level3_mask_zero_u8"),
        (np.uint16, "_nexrad_level3_mask_zero_u16"),
    ],
)
def test_msg165_177_dispatches_dense_unsigned_to_private_rust_kernel(
    monkeypatch, code, dtype, kernel_name
):
    calls = []
    data = np.array([[0.0, 1.0]], dtype=np.float32)
    mask = np.array([[True, False]], dtype=bool)

    def kernel(raw_data):
        calls.append((raw_data.dtype, raw_data.shape))
        return data.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )

    actual = _object_for_code(code, np.array([[0, 1]], dtype=dtype)).get_data()

    assert calls == [(np.dtype(dtype), (1, 2))]
    assert actual.dtype == np.float32
    assert actual.fill_value == 0.0
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("dtype", "kernel_name"),
    [
        (np.uint8, "_nexrad_level3_copy_u8"),
        (np.uint16, "_nexrad_level3_copy_u16"),
    ],
)
def test_msg34_dispatches_dense_unsigned_to_private_rust_kernel(
    monkeypatch, dtype, kernel_name
):
    calls = []
    data = np.array([[0.0, 1.0]], dtype=np.float32)

    def kernel(raw_data):
        calls.append((raw_data.dtype, raw_data.shape))
        return data.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )

    actual = _object_for_code(34, np.array([[0, 1]], dtype=dtype)).get_data()

    assert calls == [(np.dtype(dtype), (1, 2))]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.zeros((1, 2), dtype=bool))


@pytest.mark.parametrize("code", [165, 177, 34])
@pytest.mark.parametrize(
    "raw_data",
    [
        np.array([[0, 1, 255]], dtype=np.int16),
        np.array([[0, 1, 255]], dtype=np.int64),
        np.array([[0.0, 1.0, 255.0]], dtype=float),
        np.array([0, 1, 255], dtype=np.uint8),
        np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
    ],
)
def test_msg165_177_34_unsupported_inputs_keep_python_fallback(
    monkeypatch, code, raw_data
):
    def rust_kernel(name):
        if not name.startswith("_nexrad_level3_mask_zero_") and not name.startswith(
            "_nexrad_level3_copy_"
        ):
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_code(code, raw_data).get_data()
    expected = _fallback_get_data(code, raw_data, monkeypatch)

    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("code", "raw_data"),
    [
        (165, np.array([[0, 1, 255]], dtype=np.uint8)),
        (177, np.array([[0, 1, 65535]], dtype=np.uint16)),
        (165, np.empty((0, 4), dtype=np.uint8)),
        (34, np.array([[0, 1, 255]], dtype=np.uint8)),
        (34, np.array([[0, 1, 65535]], dtype=np.uint16)),
        (34, np.empty((0, 4), dtype=np.uint16)),
    ],
)
def test_real_rust_msg165_177_34_match_python_fallback(
    monkeypatch, code, raw_data
):
    rust = _rust_or_skip()

    expected = _fallback_get_data(code, raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name.startswith("_nexrad_level3_mask_zero_") or name.startswith(
            "_nexrad_level3_copy_"
        ):
            calls.append(name)
            return getattr(rust, name)
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_code(code, raw_data).get_data()

    assert calls
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize("code", [165, 177, 34])
@pytest.mark.parametrize(
    "raw_data",
    [
        np.array([[0, 1, 255]], dtype=np.int16),
        np.array([[0.0, 1.0, 255.0]], dtype=float),
        np.array([0, 1, 255], dtype=np.uint8),
        np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
    ],
)
def test_real_rust_msg165_177_34_unsupported_inputs_do_not_dispatch(
    monkeypatch, code, raw_data
):
    rust = _rust_or_skip()

    expected = _fallback_get_data(code, raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name.startswith("_nexrad_level3_mask_zero_") or name.startswith(
            "_nexrad_level3_copy_"
        ):
            calls.append(name)
            return getattr(rust, name)
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_code(code, raw_data).get_data()

    assert calls == []
    _assert_masked_equal(actual, expected)


def test_msg34_output_does_not_alias_raw():
    raw_data = np.array([[1, 2]], dtype=np.uint8)
    actual = _object_for_code(34, raw_data).get_data()

    raw_data[0, 0] = 99

    np.testing.assert_array_equal(actual.data, np.array([[1.0, 2.0]], dtype=np.float32))


@pytest.mark.parametrize(
    ("func_name", "raw_data"),
    [
        (
            "_nexrad_level3_mask_zero_u8",
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
        ),
        (
            "_nexrad_level3_mask_zero_u16",
            np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
        ),
        (
            "_nexrad_level3_copy_u8",
            np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
        ),
        (
            "_nexrad_level3_copy_u16",
            np.arange(6, dtype=np.uint16).reshape(2, 3)[:, ::2],
        ),
    ],
)
def test_real_rust_msg165_177_34_direct_reject_noncontiguous(
    func_name, raw_data
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match="C-contiguous"):
        getattr(rust, func_name)(raw_data)


@pytest.mark.parametrize(
    ("func_name", "raw_data"),
    [
        ("_nexrad_level3_mask_zero_u8", np.array([[1]], dtype=np.uint16)),
        ("_nexrad_level3_mask_zero_u16", np.array([[1]], dtype=np.uint8)),
        ("_nexrad_level3_copy_u8", np.array([1], dtype=np.uint8)),
        ("_nexrad_level3_copy_u16", np.array([1], dtype=np.uint16)),
    ],
)
def test_real_rust_msg165_177_34_direct_reject_binding_type_drift(
    func_name, raw_data
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        getattr(rust, func_name)(raw_data)


def test_real_rust_msg165_177_34_direct_return_data_and_mask():
    rust = _rust_or_skip()

    data, mask = rust._nexrad_level3_mask_zero_u16(
        np.array([[0, 1, 65535]], dtype=np.uint16)
    )
    copied = rust._nexrad_level3_copy_u16(
        np.array([[0, 1, 65535]], dtype=np.uint16)
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    assert copied.dtype == np.float32
    np.testing.assert_array_equal(
        data,
        np.array([[0.0, 1.0, 65535.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(copied, data)
    np.testing.assert_array_equal(
        mask,
        np.array([[True, False, False]], dtype=bool),
    )
