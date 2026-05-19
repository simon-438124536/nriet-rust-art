import os

import numpy as np
import pytest

from pyart.aux_io import gamicfile


class _FakeGroup:
    def __init__(self, raw_data, fmt, dyn_range_min=-32.0, dyn_range_max=64.0):
        self._raw_data = raw_data
        self.attrs = {
            "dyn_range_min": dyn_range_min,
            "dyn_range_max": dyn_range_max,
            "format": fmt,
        }

    def __getitem__(self, item):
        return self._raw_data[item]


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_gamic_decode_uv8"):
        pytest.skip("pyart._rust has no GAMIC decode kernels")
    return rust


def _fallback_sweep(group, monkeypatch):
    monkeypatch.setattr(gamicfile, "_rust_kernel", lambda _name: None)
    return gamicfile._get_gamic_sweep_data(group)


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    actual_mask = np.ma.getmaskarray(actual)
    expected_mask = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize(
    "group",
    [
        _FakeGroup(
            np.array([[0, 1, 255], [2, 3, 4]], dtype=np.uint8),
            b"UV8",
            -32.0,
            64.0,
        ),
        _FakeGroup(
            np.array([[0, 1, 65535], [2, 3, 4]], dtype=np.uint16),
            "UV16",
            -1.0,
            1.0,
        ),
        _FakeGroup(
            np.array([[1.0, np.nan], [-2.5, 4.0]], dtype=np.float32),
            "F",
        ),
    ],
)
def test_gamic_sweep_data_python_fallback_reference_cases(monkeypatch, group):
    actual = _fallback_sweep(group, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.shape == group._raw_data.shape
    assert actual.fill_value == 1e20


def test_gamic_sweep_data_dispatches_uv8_to_private_rust(monkeypatch):
    raw = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    group = _FakeGroup(raw, b"UV8", -10.0, 20.0)
    calls = []
    out = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, False]], dtype=bool)

    def kernel(raw_arg, dyn_min, dyn_max):
        calls.append((raw_arg.dtype, raw_arg.shape, dyn_min, dyn_max))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        gamicfile,
        "_rust_kernel",
        lambda name: kernel if name == "_gamic_decode_uv8" else None,
    )

    actual = gamicfile._get_gamic_sweep_data(group)

    assert calls == [(np.dtype(np.uint8), (2, 2), -10.0, 20.0)]
    assert actual.dtype == np.float32
    assert actual.fill_value == 1e20
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


def test_gamic_sweep_data_dispatches_float_to_private_rust(monkeypatch):
    raw = np.array([[1.0, np.nan]], dtype=np.float32)
    group = _FakeGroup(raw, "F")
    calls = []
    out = np.array([[1.0, np.nan]], dtype=np.float32)
    mask = np.array([[False, True]], dtype=bool)

    def kernel(raw_arg):
        calls.append((raw_arg.dtype, raw_arg.shape))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        gamicfile,
        "_rust_kernel",
        lambda name: kernel if name == "_gamic_decode_f32" else None,
    )

    actual = gamicfile._get_gamic_sweep_data(group)

    assert calls == [(np.dtype(np.float32), (1, 2))]
    _assert_masked_equal(actual, np.ma.masked_array(out, mask=mask, dtype="float32"))


@pytest.mark.parametrize(
    "group",
    [
        _FakeGroup(np.array([[1, 2]], dtype=np.uint16), "UV8"),
        _FakeGroup(np.array([[1, 2]], dtype=np.uint8), "UV16"),
        _FakeGroup(np.array([[1.0, 2.0]], dtype=np.float64), "F"),
        _FakeGroup(np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2], "UV8"),
        _FakeGroup(np.array([[1, 2]], dtype=object), "UV8"),
        _FakeGroup(np.array([[1, 2]], dtype=np.uint8), "UNKNOWN"),
        _FakeGroup(np.array([[1, 2]], dtype=np.uint8), "UV8", np.inf, 1.0),
    ],
)
def test_gamic_sweep_data_unsupported_inputs_keep_python_path(monkeypatch, group):
    def rust_kernel(name):
        if name not in {"_gamic_decode_uv8", "_gamic_decode_uv16", "_gamic_decode_f32"}:
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(gamicfile, "_rust_kernel", rust_kernel)
    try:
        actual = gamicfile._get_gamic_sweep_data(group)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_sweep(group, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_sweep(group, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust GAMIC parity",
)
@pytest.mark.parametrize(
    "group",
    [
        _FakeGroup(
            np.array([[0, 1, 255], [2, 3, 4]], dtype=np.uint8),
            b"UV8",
            -32.0,
            64.0,
        ),
        _FakeGroup(
            np.array([[0, 1, 65535], [2, 3, 4]], dtype=np.uint16),
            "UV16",
            -1.0,
            1.0,
        ),
        _FakeGroup(
            np.array([[1.0, np.nan], [-2.5, 4.0]], dtype=np.float32),
            "F",
        ),
    ],
)
def test_real_rust_gamic_sweep_data_matches_python_fallback(monkeypatch, group):
    rust = _rust_or_skip()
    expected = _fallback_sweep(group, monkeypatch)
    monkeypatch.setattr(
        gamicfile,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )

    actual = gamicfile._get_gamic_sweep_data(group)

    _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust GAMIC checks",
)
def test_real_rust_gamic_direct_helpers_match_formulas():
    rust = _rust_or_skip()

    data_u8, mask_u8 = rust._gamic_decode_uv8(
        np.array([[0, 1, 255]], dtype=np.uint8), -32.0, 64.0
    )
    np.testing.assert_array_equal(mask_u8, np.array([[True, False, False]]))
    np.testing.assert_array_equal(
        data_u8,
        np.array([[-32.0, -31.62353, 64.0]], dtype=np.float32),
    )

    data_f, mask_f = rust._gamic_decode_f32(
        np.array([[1.0, np.nan]], dtype=np.float32)
    )
    np.testing.assert_array_equal(mask_f, np.array([[False, True]]))
    np.testing.assert_array_equal(data_f, np.array([[1.0, np.nan]], dtype=np.float32))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust GAMIC checks",
)
@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda rust: rust._gamic_decode_uv8(
                np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2], 0.0, 1.0
            ),
            "C-contiguous",
        ),
        (
            lambda rust: rust._gamic_decode_uv16(
                np.array([1], dtype=np.uint16), np.inf, 1.0
            ),
            "finite",
        ),
    ],
)
def test_real_rust_gamic_direct_rejects_unsafe_inputs(call, match):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        call(rust)
