import numpy as np
import pytest

from pyart.io import nexrad_level3


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _object_for_msg135(raw_data):
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    obj.msg_header = {"code": 135}
    obj.prod_descr = {"threshold_data": b"\x00" * 32}
    obj.raw_data = raw_data
    return obj


def _fallback_msg135(raw_data, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return _object_for_msg135(raw_data).get_data()


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


@pytest.mark.parametrize(
    "raw_data",
    [
        np.array([[0, 1, 2, 127, 128, 255]], dtype=np.uint8),
        np.empty((0, 4), dtype=np.uint8),
    ],
)
def test_msg135_python_fallback_reference_cases(monkeypatch, raw_data):
    actual = _fallback_msg135(raw_data, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32


def test_msg135_dispatches_dense_u8_to_private_rust_kernel(monkeypatch):
    calls = []
    data = np.array([[254.0, 0.0]], dtype=np.float32)
    mask = np.array([[True, False]], dtype=bool)

    def kernel(raw_data):
        calls.append((raw_data.dtype, raw_data.shape))
        return data.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level3_msg_135_u8" else None,
    )

    actual = _object_for_msg135(np.array([[0, 2]], dtype=np.uint8)).get_data()

    assert calls == [(np.dtype(np.uint8), (1, 2))]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.data, data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    "raw_data",
    [
        np.array([[0, 1, 2, 127, 128, 255]], dtype=np.int16),
        np.array([[0, 1, 2, 127, 128, 255]], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.uint8),
        np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
    ],
)
def test_msg135_unsupported_inputs_keep_python_fallback(monkeypatch, raw_data):
    def rust_kernel(name):
        if name != "_nexrad_level3_msg_135_u8":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported msg135 input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_msg135(raw_data).get_data()
    expected = _fallback_msg135(raw_data, monkeypatch)

    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    "raw_data",
    [
        np.array([[0, 1, 2, 127, 128, 255]], dtype=np.uint8),
        np.empty((0, 4), dtype=np.uint8),
    ],
)
def test_real_rust_msg135_matches_python_fallback(monkeypatch, raw_data):
    rust = _rust_or_skip()

    expected = _fallback_msg135(raw_data, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_nexrad_level3_msg_135_u8":
            calls.append(name)
            return rust._nexrad_level3_msg_135_u8
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = _object_for_msg135(raw_data).get_data()

    assert calls == ["_nexrad_level3_msg_135_u8"]
    _assert_masked_equal(actual, expected)


def test_real_rust_msg135_direct_rejects_noncontiguous_input():
    rust = _rust_or_skip()
    raw_data = np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._nexrad_level3_msg_135_u8(raw_data)


def test_real_rust_msg135_direct_returns_data_and_mask():
    rust = _rust_or_skip()

    data, mask = rust._nexrad_level3_msg_135_u8(
        np.array([[0, 1, 2, 127, 128, 255]], dtype=np.uint8)
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(
        data,
        np.array([[254.0, 255.0, 0.0, 125.0, 254.0, 125.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mask,
        np.array([[True, True, False, False, False, False]], dtype=bool),
    )
