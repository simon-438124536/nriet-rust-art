import os
import warnings

import numpy as np
import pytest
from scipy.ndimage import convolve1d

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import phase_proc  # noqa: E402


def _fallback_smooth_and_trim_scan(x, window_len, window, monkeypatch):
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    return phase_proc.smooth_and_trim_scan(x, window_len=window_len, window=window)


def _scipy_oracle_smooth_and_trim_scan(x, window_len, window):
    if x.ndim != 2:
        raise ValueError("smooth only accepts 2 dimension arrays.")
    if x.shape[1] < window_len:
        raise ValueError("Input dimension 1 needs to be bigger than window size.")
    if window_len < 3:
        return x
    valid_windows = ["flat", "hanning", "hamming", "bartlett", "blackman", "sg_smooth"]
    if window not in valid_windows:
        raise ValueError("Window is on of " + " ".join(valid_windows))
    if window == "flat":
        w = np.ones(int(window_len), dtype=np.float64)
    elif window == "sg_smooth":
        w = np.array([0.1, 0.25, 0.3, 0.25, 0.1], dtype=np.float64)
    else:
        w = eval("np." + window + "(window_len)")
    return convolve1d(x, w / w.sum(), axis=1)


@pytest.mark.parametrize(
    ("shape", "window", "window_len"),
    [
        ((1, 12), "flat", 3),
        ((1, 12), "hanning", 5),
        ((4, 16), "flat", 3),
        ((4, 16), "hanning", 5),
        ((4, 16), "hamming", 6),
        ((4, 16), "bartlett", 7),
        ((4, 16), "blackman", 8),
        ((4, 16), "sg_smooth", 5),
    ],
)
def test_smooth_and_trim_scan_matches_scipy_convolve1d_oracle(
    monkeypatch, shape, window, window_len
):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape).astype(np.float64)

    expected = _scipy_oracle_smooth_and_trim_scan(x, window_len, window)
    actual = _fallback_smooth_and_trim_scan(x, window_len, window, monkeypatch)

    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_smooth_and_trim_scan_reflect_boundary_impulse(monkeypatch):
    x = np.zeros((2, 8), dtype=np.float64)
    x[:, 0] = 1.0

    expected = convolve1d(x, np.ones(3, dtype=np.float64) / 3.0, axis=1)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    actual = phase_proc.smooth_and_trim_scan(x, window_len=3, window="flat")

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_smooth_and_trim_scan_non_contiguous_view_falls_back(monkeypatch):
    base = np.linspace(1.0, 24.0, 24, dtype=np.float64).reshape(4, 6)
    x = base[:, ::2]

    expected = _fallback_smooth_and_trim_scan(x, 3, "flat", monkeypatch)

    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("non-contiguous scan input should use fallback")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)
    actual = phase_proc.smooth_and_trim_scan(x, window_len=3, window="flat")

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("x", "window_len", "window"),
    [
        (np.ones((3, 8), dtype=np.float32), 3, "flat"),
        (np.ones((3, 8), dtype=np.int32), 3, "flat"),
        (np.ma.array(np.ones((3, 8), dtype=np.float64)), 3, "flat"),
        (np.array([[1.0, np.nan, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64), 3, "flat"),
        (np.array([[1.0, np.inf, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64), 3, "flat"),
        (np.ones((3, 8), dtype=np.float64), 3, "sg_smooth"),
        (np.ones((3, 8), dtype=np.float64), 2, "flat"),
    ],
)
def test_smooth_and_trim_scan_keeps_python_path_for_unsupported_inputs(
    monkeypatch, x, window_len, window
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported smooth_and_trim_scan input should use fallback")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        actual = phase_proc.smooth_and_trim_scan(x, window_len=window_len, window=window)
    expected = _fallback_smooth_and_trim_scan(x, window_len, window, monkeypatch)

    if window_len < 3:
        assert actual is x
        assert expected is x
    else:
        np.testing.assert_array_equal(actual, expected)


def test_smooth_and_trim_scan_short_width_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("short width should raise before Rust dispatch")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)
    x = np.ones((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="Input dimension 1"):
        phase_proc.smooth_and_trim_scan(x, window_len=3, window="flat")


@pytest.mark.parametrize(
    ("x", "window_len", "window", "error_type"),
    [
        (np.ones(8, dtype=np.float64), 3, "flat", ValueError),
        (np.ones((3, 2), dtype=np.float64), 3, "flat", ValueError),
        (np.ones((3, 8), dtype=np.float64), 3, "boxcar", ValueError),
    ],
)
def test_smooth_and_trim_scan_preserves_python_exception_edges(
    monkeypatch, x, window_len, window, error_type
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("exception edge should use Python fallback")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)

    with pytest.raises(error_type):
        phase_proc.smooth_and_trim_scan(x, window_len=window_len, window=window)


def test_smooth_and_trim_scan_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(x, weights, window_len):
        calls.append((x.dtype, x.shape, weights.dtype, weights.shape, window_len))
        return np.full(x.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: (
            rust_kernel if name == "_phase_proc_smooth_and_trim_scan_f64" else None
        ),
    )
    x = np.ones((2, 6), dtype=np.float64)

    actual = phase_proc.smooth_and_trim_scan(x, window_len=3, window="flat")

    np.testing.assert_array_equal(actual, np.full((2, 6), 7.0))
    assert calls == [(np.float64, (2, 6), np.float64, (3,), 3)]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("shape", "window", "window_len"),
    [
        ((1, 16), "flat", 3),
        ((4, 24), "hanning", 5),
        ((4, 24), "sg_smooth", 5),
    ],
)
def test_real_rust_smooth_and_trim_scan_matches_python_fallback(
    monkeypatch, shape, window, window_len
):
    import pyart._rust as rust

    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape).astype(np.float64)

    expected = _fallback_smooth_and_trim_scan(x, window_len, window, monkeypatch)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = phase_proc.smooth_and_trim_scan(x, window_len=window_len, window=window)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust helper checks are verified in installed-wheel mode",
)
def test_real_rust_smooth_and_trim_scan_helper_exists():
    import pyart._rust as rust

    assert hasattr(rust, "_phase_proc_smooth_and_trim_scan_f64")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("x", "weights", "window_len", "match"),
    [
        (
            np.ones((2, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64) / 3.0,
            3,
            "x width",
        ),
        (
            np.array([[1.0, np.nan, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64),
            np.ones(3, dtype=np.float64) / 3.0,
            3,
            "x must be finite",
        ),
        (
            np.ones((2, 6), dtype=np.float64),
            np.array([0.5, np.inf, 0.5], dtype=np.float64),
            3,
            "weights must be finite",
        ),
        (
            np.ones((2, 6), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            3,
            "weights sum",
        ),
    ],
)
def test_real_rust_smooth_and_trim_scan_rejects_unsafe_direct_inputs(
    x, weights, window_len, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._phase_proc_smooth_and_trim_scan_f64(x, weights, window_len)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_smooth_and_trim_scan_large_2d_is_exact_and_warning_free(
    monkeypatch,
):
    import pyart._rust as rust

    rng = np.random.default_rng(11)
    x = rng.standard_normal((32, 512)).astype(np.float64)

    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    expected = {
        (window, window_len): phase_proc.smooth_and_trim_scan(
            x, window_len=window_len, window=window
        )
        for window, window_len in [
            ("flat", 11),
            ("hanning", 12),
            ("hamming", 13),
            ("sg_smooth", 5),
        ]
    }

    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        actual = {
            key: phase_proc.smooth_and_trim_scan(x, window_len=key[1], window=key[0])
            for key in expected
        }

    assert warning_records == []
    for key, expected_values in expected.items():
        np.testing.assert_allclose(
            actual[key], expected_values, rtol=0.0, atol=1.0e-14
        )
