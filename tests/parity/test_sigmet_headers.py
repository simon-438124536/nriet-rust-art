import os

import numpy as np
import pytest

from pyart.io import _sigmetfile


def _fallback_bin2(value, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    return _sigmetfile.bin2_to_angle(value)


def _fallback_bin4(value, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    return _sigmetfile.bin4_to_angle(value)


def _fallback_parse_ray_headers(ray_headers, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    return _sigmetfile._parse_ray_headers(ray_headers)


def _fallback_data_types_from_mask(words, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    return _sigmetfile._data_types_from_mask(*words)


def _assert_header_tuple_equal(actual, expected):
    assert len(actual) == len(expected) == 7
    for actual_item, expected_item in zip(actual, expected):
        assert actual_item.dtype == expected_item.dtype
        assert actual_item.shape == expected_item.shape
        np.testing.assert_array_equal(actual_item, expected_item)


def _sample_ray_headers(shape=(3, 6)):
    values = np.array(
        [
            [0, 16384, 32768, 49152, 25, 7],
            [65535, 1, 2, 3, 0, 65535],
            [32768, 32769, 32770, 32771, 12, 42],
        ],
        dtype=np.uint16,
    ).view(np.int16)
    return np.resize(values, shape).astype(np.int16, copy=False)


def test_sigmet_angle_python_fallback_preserves_scalar_and_array_contract(monkeypatch):
    assert type(_fallback_bin2(32768, monkeypatch)) is float
    assert type(_fallback_bin4(2147483648, monkeypatch)) is float

    bin2_scalar = _fallback_bin2(np.array(32768, dtype=np.uint16), monkeypatch)
    assert type(bin2_scalar) is np.float64
    assert bin2_scalar.shape == ()

    bin4_scalar = _fallback_bin4(np.array(2147483648, dtype=np.uint32), monkeypatch)
    assert type(bin4_scalar) is np.float64
    assert bin4_scalar.shape == ()

    bin2_array = _fallback_bin2(
        np.array([[0, 16384], [32768, 65535]], dtype=np.uint16), monkeypatch
    )
    assert type(bin2_array) is np.ndarray
    assert bin2_array.dtype == np.float64
    assert bin2_array.shape == (2, 2)

    bin4_array = _fallback_bin4(
        np.array([[0, 1073741824], [2147483648, 4294967295]], dtype=np.uint32),
        monkeypatch,
    )
    assert type(bin4_array) is np.ndarray
    assert bin4_array.dtype == np.float64
    assert bin4_array.shape == (2, 2)


def test_sigmet_angle_dispatches_dense_native_arrays_to_private_rust_kernel(monkeypatch):
    calls = []

    def fake_bin2(values):
        calls.append(("bin2", values.dtype, values.shape))
        return np.full(values.shape, 2.0, dtype=np.float64)

    def fake_bin4(values):
        calls.append(("bin4", values.dtype, values.shape))
        return np.full(values.shape, 4.0, dtype=np.float64)

    def rust_kernel(name):
        return {
            "_sigmet_bin2_to_angle_u16": fake_bin2,
            "_sigmet_bin4_to_angle_u32": fake_bin4,
        }.get(name)

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)

    bin2 = _sigmetfile.bin2_to_angle(np.arange(6, dtype=np.uint16).reshape(2, 3))
    bin4 = _sigmetfile.bin4_to_angle(np.arange(6, dtype=np.uint32).reshape(2, 3))

    assert calls == [
        ("bin2", np.dtype(np.uint16), (2, 3)),
        ("bin4", np.dtype(np.uint32), (2, 3)),
    ]
    np.testing.assert_array_equal(bin2, np.full((2, 3), 2.0))
    np.testing.assert_array_equal(bin4, np.full((2, 3), 4.0))


def test_sigmet_angle_unsupported_arrays_keep_python_fallback(monkeypatch):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)

    bin2_big_endian = np.array([0, 16384, 32768], dtype=">u2")
    bin2_strided = np.arange(6, dtype=np.uint16)[::2]
    bin2_zero_dim = np.array(32768, dtype=np.uint16)
    np.testing.assert_array_equal(
        _sigmetfile.bin2_to_angle(bin2_big_endian),
        360.0 * bin2_big_endian / 65536,
    )
    np.testing.assert_array_equal(
        _sigmetfile.bin2_to_angle(bin2_strided),
        360.0 * bin2_strided / 65536,
    )
    assert type(_sigmetfile.bin2_to_angle(bin2_zero_dim)) is np.float64

    bin4_big_endian = np.array([0, 1073741824], dtype=">u4")
    bin4_strided = np.arange(6, dtype=np.uint32)[::2]
    bin4_zero_dim = np.array(2147483648, dtype=np.uint32)
    np.testing.assert_array_equal(
        _sigmetfile.bin4_to_angle(bin4_big_endian),
        360.0 * bin4_big_endian / 4294967296,
    )
    np.testing.assert_array_equal(
        _sigmetfile.bin4_to_angle(bin4_strided),
        360.0 * bin4_strided / 4294967296,
    )
    assert type(_sigmetfile.bin4_to_angle(bin4_zero_dim)) is np.float64


def test_parse_ray_headers_python_fallback_preserves_dtypes_and_shapes(monkeypatch):
    ray_headers = _sample_ray_headers((2, 3, 6))
    result = _fallback_parse_ray_headers(ray_headers, monkeypatch)

    expected_dtypes = [
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        np.int16,
        np.uint16,
        np.int16,
    ]
    for item, dtype in zip(result, expected_dtypes):
        assert item.dtype == np.dtype(dtype)
        assert item.shape == (2, 3)


def test_parse_ray_headers_dispatches_dense_native_headers_to_private_rust_kernel(monkeypatch):
    calls = []
    sentinel = tuple(
        np.full((2, 3), index, dtype=dtype)
        for index, dtype in enumerate(
            [np.float64, np.float64, np.float64, np.float64, np.int16, np.uint16, np.int16]
        )
    )

    def fake_parse(ray_headers):
        calls.append((ray_headers.dtype, ray_headers.shape))
        return sentinel

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: fake_parse if name == "_sigmet_parse_ray_headers_i16" else None,
    )

    result = _sigmetfile._parse_ray_headers(np.zeros((2, 3, 6), dtype=np.int16))

    assert calls == [(np.dtype(np.int16), (2, 3, 6))]
    _assert_header_tuple_equal(result, sentinel)


def test_parse_ray_headers_unsupported_inputs_keep_python_fallback(monkeypatch):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)

    noncontiguous = np.zeros((3, 7), dtype=np.int16)[:, :6]
    one_dimensional = np.zeros((6,), dtype=np.int16)
    too_wide = np.zeros((3, 7), dtype=np.int16)
    big_endian = np.zeros((3, 6), dtype=">i2")

    _assert_header_tuple_equal(
        _sigmetfile._parse_ray_headers(noncontiguous),
        _fallback_parse_ray_headers(noncontiguous, monkeypatch),
    )
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    assert type(_sigmetfile._parse_ray_headers(one_dimensional)[0]) is np.float64

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    with pytest.raises(IndexError):
        _sigmetfile._parse_ray_headers(np.zeros((3, 5), dtype=np.int16))

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    _assert_header_tuple_equal(
        _sigmetfile._parse_ray_headers(too_wide),
        _fallback_parse_ray_headers(too_wide, monkeypatch),
    )

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    _assert_header_tuple_equal(
        _sigmetfile._parse_ray_headers(big_endian),
        _fallback_parse_ray_headers(big_endian, monkeypatch),
    )


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ((0, 0, 0, 0), []),
        ((1, 0, 0, 0), [0]),
        ((0b101, 0b10, 0, 0x80000000), [0, 2, 33, 127]),
        ((0xFFFFFFFF, 0, 0, 0), list(range(32))),
    ],
)
def test_sigmet_data_types_from_mask_python_fallback_reference(
    monkeypatch, words, expected
):
    actual = _fallback_data_types_from_mask(words, monkeypatch)

    assert actual == expected


def test_sigmet_data_types_from_mask_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def kernel(*words):
        calls.append(words)
        return [99]

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_data_types_from_mask_u32" else None,
    )

    actual = _sigmetfile._data_types_from_mask(np.uint32(1), np.int64(2), 0, 0)

    assert actual == [99]
    assert calls == [(1, 2, 0, 0)]


@pytest.mark.parametrize(
    "words",
    [
        (-1, 0, 0, 0),
        (1 << 40, 0, 0, 0),
        (True, 0, 0, 0),
        ("1", 0, 0, 0),
    ],
)
def test_sigmet_data_types_from_mask_unsupported_inputs_keep_python_fallback(
    monkeypatch, words
):
    def rust_kernel(name):
        if name != "_sigmet_data_types_from_mask_u32":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported mask input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    try:
        actual = _sigmetfile._data_types_from_mask(*words)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_data_types_from_mask(words, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_data_types_from_mask(words, monkeypatch)
        assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_sigmet_angles_and_headers_match_python_fallback(monkeypatch):
    import pyart._rust as rust

    bin2 = np.array([[0, 16384], [32768, 65535]], dtype=np.uint16)
    bin4 = np.array([[0, 1073741824], [2147483648, 4294967295]], dtype=np.uint32)
    ray_headers = _sample_ray_headers((2, 3, 6))

    expected_bin2 = _fallback_bin2(bin2, monkeypatch)
    expected_bin4 = _fallback_bin4(bin4, monkeypatch)
    expected_headers = _fallback_parse_ray_headers(ray_headers, monkeypatch)

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda name: getattr(rust, name, None))

    actual_bin2 = _sigmetfile.bin2_to_angle(bin2)
    actual_bin4 = _sigmetfile.bin4_to_angle(bin4)
    actual_headers = _sigmetfile._parse_ray_headers(ray_headers)

    np.testing.assert_array_equal(actual_bin2, expected_bin2)
    np.testing.assert_array_equal(actual_bin4, expected_bin4)
    _assert_header_tuple_equal(actual_headers, expected_headers)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "words",
    [
        (0, 0, 0, 0),
        (0b101, 0b10, 0, 0x80000000),
        (0xFFFFFFFF, 0xFFFFFFFF, 0, 1),
    ],
)
def test_real_rust_sigmet_data_types_from_mask_matches_python_fallback(
    monkeypatch, words
):
    import pyart._rust as rust

    expected = _fallback_data_types_from_mask(words, monkeypatch)
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda name: getattr(rust, name, None))

    actual = _sigmetfile._data_types_from_mask(*words)

    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("function_name", "value", "match"),
    [
        ("_sigmet_bin2_to_angle_u16", np.arange(6, dtype=np.uint16)[::2], "C-contiguous"),
        ("_sigmet_bin4_to_angle_u32", np.arange(6, dtype=np.uint32)[::2], "C-contiguous"),
        ("_sigmet_parse_ray_headers_i16", np.zeros((3, 7), dtype=np.int16)[:, :6], "C-contiguous"),
        ("_sigmet_parse_ray_headers_i16", np.zeros((6,), dtype=np.int16), "at least 2 dimensions"),
        ("_sigmet_parse_ray_headers_i16", np.zeros((3, 5), dtype=np.int16), "last dimension"),
    ],
)
def test_real_rust_sigmet_rejects_unsafe_direct_inputs(function_name, value, match):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        getattr(rust, function_name)(value)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "words",
    [
        (-1, 0, 0, 0),
        (1 << 40, 0, 0, 0),
        (True, 0, 0, 0),
        (np.bool_(True), 0, 0, 0),
        ("1", 0, 0, 0),
        (1.0, 0, 0, 0),
    ],
)
def test_real_rust_sigmet_data_types_from_mask_rejects_bad_words(words):
    import pyart._rust as rust

    with pytest.raises(ValueError, match="mask word"):
        rust._sigmet_data_types_from_mask_u32(*words)
