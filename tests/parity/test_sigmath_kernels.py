import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal

os.environ.setdefault("PYART_QUIET", "1")

from pyart.util import sigmath  # noqa: E402


def _radar(field):
    return SimpleNamespace(fields={"velocity": {"data": field}})


def _reference_angular_texture_2d(image, N, interval):
    if isinstance(N, int):
        N = (N, N)

    interval_max = interval
    interval_min = -interval
    half_width = (interval_max - interval_min) / 2.0
    center = interval_min + half_width

    im = (np.asarray(image) - center) / (half_width) * np.pi
    x = np.cos(im)
    y = np.sin(im)

    kernel = np.ones(N)
    xs = signal.convolve2d(x, kernel, mode="same", boundary="symm")
    ys = signal.convolve2d(y, kernel, mode="same", boundary="symm")
    ns = np.prod(N)

    xmean = xs / ns
    ymean = ys / ns
    norm = np.sqrt(xmean**2 + ymean**2)
    return np.sqrt(-2 * np.log(norm)) * (half_width) / np.pi


def _fallback_texture_along_ray(field, wind_size, monkeypatch):
    monkeypatch.setattr(sigmath, "_rust_kernel", lambda _name: None)
    return sigmath.texture_along_ray(_radar(field), "velocity", wind_size=wind_size)


def _assert_texture_close(actual, expected):
    assert np.ma.isMaskedArray(actual)
    assert np.ma.isMaskedArray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_allclose(actual.data, expected.data, rtol=0.0, atol=1.0e-14)


def test_angular_texture_2d_python_fallback_matches_scipy_symm_boundary(monkeypatch):
    monkeypatch.setattr(sigmath, "_rust_kernel", lambda _name: None)
    image = np.array([[1.0, 2.5, 4.0], [5.5, 7.0, 8.5]], dtype=np.float64)

    actual = sigmath.angular_texture_2d(image, (3, 5), 9.0)
    expected = _reference_angular_texture_2d(image, (3, 5), 9.0)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


def test_texture_along_ray_python_fallback_matches_rolling_std(monkeypatch):
    field = np.arange(16.0, dtype=np.float64).reshape(2, 8)

    actual = _fallback_texture_along_ray(field, 3, monkeypatch)

    expected = np.ma.array(np.full((2, 8), np.sqrt(2.0 / 3.0)))
    _assert_texture_close(actual, expected)


def test_texture_along_ray_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []
    field = np.arange(16.0, dtype=np.float64).reshape(2, 8)

    def rust_kernel(field_arg, wind_size):
        calls.append((field_arg.dtype, field_arg.shape, wind_size))
        return np.full(field_arg.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        sigmath,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_texture_along_ray_dense_f64" else None,
    )

    actual = sigmath.texture_along_ray(_radar(field), "velocity", wind_size=np.int64(5))

    assert calls == [(np.float64, (2, 8), 5)]
    expected = np.ma.array(np.full((2, 8), 7.0, dtype=np.float64))
    _assert_texture_close(actual, expected)


@pytest.mark.parametrize(
    ("field", "wind_size"),
    [
        (np.arange(16.0, dtype=np.float32).reshape(2, 8), 3),
        (np.arange(16.0, dtype=np.float64).reshape(2, 8)[:, ::2], 3),
        (
            np.ma.array(
                np.arange(16.0, dtype=np.float64).reshape(2, 8),
                mask=np.zeros((2, 8), dtype=bool),
            ),
            3,
        ),
        (np.arange(16.0, dtype=np.float64).reshape(2, 8), 2),
        (np.arange(16.0, dtype=np.float64).reshape(2, 8), 9),
        (np.array([[1.0, np.nan, 3.0]], dtype=np.float64), 3),
        (np.arange(16.0, dtype=np.float64).reshape(2, 8), True),
    ],
)
def test_texture_along_ray_keeps_python_path_for_unsupported_inputs(
    monkeypatch, field, wind_size
):
    def fail_if_called(name):
        if name != "_texture_along_ray_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported texture_along_ray input used Rust")

        return kernel

    monkeypatch.setattr(sigmath, "_rust_kernel", fail_if_called)
    try:
        actual = sigmath.texture_along_ray(_radar(field), "velocity", wind_size=wind_size)
    except Exception as actual_error:
        monkeypatch.setattr(sigmath, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            sigmath.texture_along_ray(_radar(field), "velocity", wind_size=wind_size)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_texture_along_ray(field, wind_size, monkeypatch)
        _assert_texture_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("wind_size", [3, 7])
def test_real_rust_texture_along_ray_matches_python_fallback(monkeypatch, wind_size):
    import pyart._rust as rust

    field = np.array(
        [
            [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0],
            [2.0, 5.0, 4.0, 8.0, 7.0, 9.0, 11.0, 12.0],
        ],
        dtype=np.float64,
    )
    expected = _fallback_texture_along_ray(field, wind_size, monkeypatch)
    calls = []

    def counted_kernel(field_arg, wind_arg):
        calls.append((field_arg.shape, wind_arg))
        return rust._texture_along_ray_dense_f64(field_arg, wind_arg)

    monkeypatch.setattr(
        sigmath,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_texture_along_ray_dense_f64" else None,
    )
    actual = sigmath.texture_along_ray(_radar(field), "velocity", wind_size=wind_size)

    assert calls == [((2, 8), wind_size)]
    _assert_texture_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_texture_along_ray_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._texture_along_ray_dense_f64(
            np.arange(16.0, dtype=np.float64).reshape(2, 8)[:, ::2], 3
        )
    with pytest.raises(ValueError, match="at least 3"):
        rust._texture_along_ray_dense_f64(np.ones((2, 8), dtype=np.float64), 1)
    with pytest.raises(ValueError, match="odd"):
        rust._texture_along_ray_dense_f64(np.ones((2, 8), dtype=np.float64), 4)
    with pytest.raises(ValueError, match="gate count"):
        rust._texture_along_ray_dense_f64(np.ones((2, 8), dtype=np.float64), 9)
    with pytest.raises(ValueError, match="finite"):
        rust._texture_along_ray_dense_f64(np.array([[np.nan, 1.0, 2.0]]), 3)


def test_angular_texture_2d_dispatches_to_private_rust_kernel_for_compatible_input(monkeypatch):
    calls = []
    image = np.arange(12.0, dtype=np.float64).reshape(3, 4)

    def rust_kernel(image_arg, window_rows, window_cols, interval):
        calls.append((image_arg.dtype, image_arg.shape, window_rows, window_cols, interval))
        return np.full(image_arg.shape, 42.0, dtype=np.float64)

    monkeypatch.setattr(
        sigmath,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_angular_texture_2d" else None,
    )

    result = sigmath.angular_texture_2d(image, 3, 8.0)

    assert calls == [(np.float64, (3, 4), 3, 3, 8.0)]
    np.testing.assert_array_equal(result, np.full((3, 4), 42.0))


@pytest.mark.parametrize(
    ("image", "window"),
    [
        (np.arange(12.0, dtype=np.float32).reshape(3, 4), 3),
        (np.arange(12.0, dtype=np.float64).reshape(3, 4), 4),
        (np.arange(12.0, dtype=np.float64).reshape(3, 4), (3, 4)),
        (np.arange(12.0, dtype=np.float64).reshape(3, 4).tolist(), 3),
        (np.ma.array(np.arange(12.0, dtype=np.float64).reshape(3, 4)), 3),
    ],
)
def test_angular_texture_2d_keeps_scipy_path_for_unsupported_inputs(
    monkeypatch, image, window
):
    def rust_kernel(*_args):
        raise AssertionError("unsupported input should use the SciPy fallback")

    monkeypatch.setattr(sigmath, "_rust_kernel", lambda _name: rust_kernel)

    actual = sigmath.angular_texture_2d(image, window, 8.0)
    expected = _reference_angular_texture_2d(image, window, 8.0)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_angular_texture_2d_matches_scipy_fallback(monkeypatch):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    rust_kernel = getattr(rust, "_angular_texture_2d", None)
    if rust_kernel is None:
        pytest.fail("pyart._rust has not registered _angular_texture_2d")

    image = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.5, 4.5, 6.5, 8.5],
            [9.0, 7.0, 5.0, 3.0],
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(sigmath, "_rust_kernel", lambda _name: None)
    expected = sigmath.angular_texture_2d(image, (3, 5), 9.5)
    monkeypatch.setattr(
        sigmath,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_angular_texture_2d" else None,
    )
    actual = sigmath.angular_texture_2d(image, (3, 5), 9.5)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
