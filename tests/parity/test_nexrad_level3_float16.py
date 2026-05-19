import math
import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.io import nexrad_level3  # noqa: E402


FLOAT16_CASES = [
    (0, 0.0),
    (1, 0.001953125),
    (1023, 1.998046875),
    (1024, 3.0517578125e-05),
    (0x3C00, 0.5),
    (0x7BFF, 32752.0),
    (0x8000, -0.0),
    (0x8400, -3.0517578125e-05),
    (0xFFFF, -65504.0),
    (-1, -65504.0),
    (-2, -65472.0),
    (-32768, -0.0),
    (-65536, 0.0),
    (1 << 20, 0.0),
]


def _fallback_int16_to_float16(value, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return nexrad_level3._int16_to_float16(value)


def _assert_same_scalar(actual, expected):
    assert type(actual) is type(expected)
    assert actual == expected
    if isinstance(actual, (float, np.floating)) and actual == 0:
        assert math.copysign(1.0, float(actual)) == math.copysign(1.0, float(expected))


@pytest.mark.parametrize(("value", "expected"), FLOAT16_CASES)
def test_int16_to_float16_python_fallback_reference_cases(
    monkeypatch, value, expected
):
    actual = _fallback_int16_to_float16(value, monkeypatch)

    assert type(actual) is float
    assert actual == expected
    if actual == 0:
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def test_int16_to_float16_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def kernel(value):
        calls.append(value)
        return 123.0

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level3_int16_to_float16" else None,
    )

    actual = nexrad_level3._int16_to_float16(-1)

    assert actual == 123.0
    assert calls == [-1]


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.int16(-1),
        np.uint16(0xFFFF),
        np.int64(-1),
        1 << 100,
        1.5,
        "1",
    ],
)
def test_int16_to_float16_keeps_python_path_for_unsupported_inputs(
    monkeypatch, value
):
    def fail_if_called(name):
        if name != "_nexrad_level3_int16_to_float16":
            return None

        def kernel(_value):
            raise AssertionError("unsupported _int16_to_float16 input used Rust")

        return kernel

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", fail_if_called)
    try:
        actual = nexrad_level3._int16_to_float16(value)
    except Exception as actual_error:
        monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            nexrad_level3._int16_to_float16(value)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_int16_to_float16(value, monkeypatch)
        _assert_same_scalar(actual, expected)


def test_msg_134_converts_numpy_halfwords_before_float16_decode(monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    halfwords = np.array([0x3C00, 0, 10, 0x4000, 0], dtype=">i2")
    obj.prod_descr = {"threshold_data": halfwords.tobytes() + (b"\x00" * 22)}
    obj.raw_data = np.array([[0, 2, 5, 10, 12]], dtype=np.uint8)

    actual = obj._get_data_msg_134()

    expected_data = np.array(
        [[0.0, 4.0, 10.0, np.exp(10.0), np.exp(12.0)]], dtype=np.float32
    )
    expected = np.ma.masked_array(expected_data, mask=obj.raw_data < 2)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(("value", "_expected"), FLOAT16_CASES)
def test_real_rust_int16_to_float16_matches_python_fallback(
    monkeypatch, value, _expected
):
    import pyart._rust as rust

    expected = _fallback_int16_to_float16(value, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_nexrad_level3_int16_to_float16":
            calls.append(name)
            return rust._nexrad_level3_int16_to_float16
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = nexrad_level3._int16_to_float16(value)

    assert calls == ["_nexrad_level3_int16_to_float16"]
    _assert_same_scalar(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust kernel is verified in installed-wheel mode",
)
def test_real_rust_int16_to_float16_direct_kernel():
    import pyart._rust as rust

    for value, expected in FLOAT16_CASES:
        actual = rust._nexrad_level3_int16_to_float16(value)
        assert actual == expected
        if actual == 0:
            assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust kernel is verified in installed-wheel mode",
)
@pytest.mark.parametrize("value", [True, np.int16(0), np.uint16(0xFFFF), np.int64(-1)])
def test_real_rust_int16_to_float16_direct_rejects_non_python_int(value):
    import pyart._rust as rust

    with pytest.raises(TypeError, match="Python int"):
        rust._nexrad_level3_int16_to_float16(value)
