import os

import numpy as np
import pytest

from pyart.aux_io import _odim_sweep, odim_h5, sinarame_h5


class _FakeWhat:
    def __init__(self, attrs):
        self.attrs = attrs


class _FakeGroup:
    def __init__(self, raw_data, attrs):
        self._raw_data = raw_data
        self._what = _FakeWhat(attrs)

    def __getitem__(self, key):
        if key == "what":
            return self._what
        if key == "data":
            return self._raw_data
        raise KeyError(key)


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_odim_decode_u8"):
        pytest.skip("pyart._rust has no ODIM decode kernels")
    return rust


def _fallback_sweep(function, group, monkeypatch):
    monkeypatch.setattr(_odim_sweep, "_rust_kernel", lambda _name: None)
    return function(group)


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize(
    "function",
    [odim_h5._get_odim_h5_sweep_data, sinarame_h5._get_SINARAME_h5_sweep_data],
)
@pytest.mark.parametrize(
    "group",
    [
        _FakeGroup(np.array([[0, 1, 2]], dtype=np.uint8), {}),
        _FakeGroup(np.array([[0, 1, 255]], dtype=np.uint8), {"nodata": 255}),
        _FakeGroup(np.array([[0, 1, 2]], dtype=np.uint8), {"undetect": 1}),
        _FakeGroup(
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
            {"nodata": 255, "undetect": 1, "gain": 0.5, "offset": -1.0},
        ),
        _FakeGroup(
            np.array([[0, 1, 65535]], dtype=np.uint16),
            {"nodata": 65535, "undetect": 1, "gain": -0.25, "offset": 2.0},
        ),
    ],
)
def test_odim_like_sweep_data_python_fallback_reference_cases(
    monkeypatch, function, group
):
    actual = _fallback_sweep(function, group, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.shape == group._raw_data.shape


def test_odim_like_sweep_data_dispatches_dense_u8_to_private_rust(monkeypatch):
    raw = np.array([[0, 1, 255]], dtype=np.uint8)
    group = _FakeGroup(
        raw,
        {"nodata": 255, "undetect": 1, "gain": 0.5, "offset": -1.0},
    )
    calls = []
    out = np.array([[-1.0, 1.0, 255.0]], dtype=np.float64)
    mask = np.array([[False, True, True]], dtype=bool)

    def kernel(raw_arg, has_nodata, nodata, has_undetect, undetect, gain, offset):
        calls.append(
            (
                raw_arg.dtype,
                raw_arg.shape,
                has_nodata,
                nodata,
                has_undetect,
                undetect,
                gain,
                offset,
            )
        )
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        _odim_sweep,
        "_rust_kernel",
        lambda name: kernel if name == "_odim_decode_u8" else None,
    )

    actual = odim_h5._get_odim_h5_sweep_data(group)

    assert calls == [(np.dtype(np.uint8), (1, 3), True, 255, True, 1, 0.5, -1.0)]
    assert actual.dtype == np.float64
    assert actual.fill_value == 255
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    "function",
    [odim_h5._get_odim_h5_sweep_data, sinarame_h5._get_SINARAME_h5_sweep_data],
)
@pytest.mark.parametrize(
    "group",
    [
        _FakeGroup(np.array([[0, 1]], dtype=np.float32), {}),
        _FakeGroup(np.array([[0, 1]], dtype=np.int16), {}),
        _FakeGroup(np.array([[0, 1]], dtype=object), {}),
        _FakeGroup(np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2], {}),
        _FakeGroup(np.array([[0, 1]], dtype=np.uint8), {"gain": np.float32(0.5), "offset": np.float32(1.0)}),
        _FakeGroup(np.array([[0, 1]], dtype=np.uint8), {"gain": np.nan}),
        _FakeGroup(np.array([[0, 1]], dtype=np.uint8), {"nodata": np.nan}),
        _FakeGroup(np.array([[0, 1]], dtype=np.uint8), {"undetect": np.array([1, 2])}),
    ],
)
def test_odim_like_sweep_data_unsupported_inputs_keep_python_path(
    monkeypatch, function, group
):
    def rust_kernel(name):
        if name not in {"_odim_decode_u8", "_odim_decode_u16"}:
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_odim_sweep, "_rust_kernel", rust_kernel)
    try:
        actual = function(group)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_sweep(function, group, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_sweep(function, group, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust ODIM parity",
)
@pytest.mark.parametrize(
    "function",
    [odim_h5._get_odim_h5_sweep_data, sinarame_h5._get_SINARAME_h5_sweep_data],
)
@pytest.mark.parametrize(
    "group",
    [
        _FakeGroup(np.array([[0, 1, 2]], dtype=np.uint8), {}),
        _FakeGroup(
            np.array([[0, 1, 2, 255]], dtype=np.uint8),
            {"nodata": 255, "undetect": 1, "gain": 0.5, "offset": -1.0},
        ),
        _FakeGroup(
            np.array([[0, 1, 65535]], dtype=np.uint16),
            {"nodata": 65535, "undetect": 1, "gain": -0.25, "offset": 2.0},
        ),
    ],
)
def test_real_rust_odim_like_sweep_data_matches_python_fallback(
    monkeypatch, function, group
):
    rust = _rust_or_skip()
    expected = _fallback_sweep(function, group, monkeypatch)
    monkeypatch.setattr(_odim_sweep, "_rust_kernel", lambda name: getattr(rust, name, None))

    actual = function(group)

    _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust ODIM checks",
)
def test_real_rust_odim_direct_helper_preserves_masked_payloads():
    rust = _rust_or_skip()

    data, mask = rust._odim_decode_u8(
        np.array([[0, 1, 2, 255]], dtype=np.uint8),
        True,
        255,
        True,
        1,
        0.5,
        -1.0,
    )

    np.testing.assert_array_equal(
        data, np.array([[-1.0, 1.0, 0.0, 255.0]], dtype=np.float64)
    )
    np.testing.assert_array_equal(mask, np.array([[False, True, False, True]]))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust ODIM checks",
)
@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda rust: rust._odim_decode_u8(
                np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2],
                False,
                0,
                False,
                0,
                1.0,
                0.0,
            ),
            "C-contiguous",
        ),
        (
            lambda rust: rust._odim_decode_u16(
                np.array([1], dtype=np.uint16),
                False,
                0,
                False,
                0,
                np.inf,
                0.0,
            ),
            "finite",
        ),
    ],
)
def test_real_rust_odim_direct_rejects_unsafe_inputs(call, match):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        call(rust)
