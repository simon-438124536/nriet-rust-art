import os

import numpy as np
import pytest

from pyart.io import output_to_geotiff
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_geotiff_rgb_values_f64"):
        pytest.skip("pyart._rust has no GeoTIFF RGB kernel")
    return rust


def _fallback_rgb(data, monkeypatch, **kwargs):
    monkeypatch.setattr(output_to_geotiff, "_rust_kernel", lambda _name: None)
    return output_to_geotiff._get_rgb_values(data, **kwargs)


def _assert_rgb_equal(actual, expected):
    assert len(actual) == len(expected) == 4
    for actual_arr, expected_arr in zip(actual, expected):
        assert_exact_equal(actual_arr, expected_arr)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(vmin=0.0, vmax=10.0, color_levels=None, cmap="viridis", transpbg=True, op=0.5),
        dict(vmin=0.0, vmax=10.0, color_levels=7, cmap="plasma", transpbg=False, op=0.5),
    ],
)
def test_geotiff_rgb_values_python_fallback_reference_cases(monkeypatch, kwargs):
    data = np.array([[-5.0, 0.0, 5.0], [10.0, 20.0, np.inf]], dtype=np.float64)
    actual = _fallback_rgb(data, monkeypatch, **kwargs)

    assert actual[0].shape == data.shape
    assert actual[3].dtype == np.int64
    assert all(arr.dtype == np.int64 for arr in actual[:3])


@pytest.mark.parametrize("transpbg", [True, False])
def test_geotiff_rgb_values_nan_path_reference(monkeypatch, transpbg):
    data = np.array([[0.0, np.nan, 1.0]], dtype=np.float64)
    actual = _fallback_rgb(
        data,
        monkeypatch,
        vmin=0.0,
        vmax=1.0,
        color_levels=None,
        cmap="viridis",
        transpbg=transpbg,
        op=0.5,
    )

    assert actual[0].dtype == np.float64
    assert actual[1].dtype == np.float64
    assert actual[2].dtype == np.float64
    assert actual[3].dtype == np.int64
    assert np.isnan(actual[0][0, 1])
    assert actual[3][0, 1] == (0 if transpbg else 128)


def test_geotiff_rgb_values_dispatches_dense_float64_to_private_rust(monkeypatch):
    data = np.array([[0.0, 1.0]], dtype=np.float64)
    calls = []
    r = np.array([[1.0, 2.0]], dtype=np.float64)
    g = np.array([[3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0]], dtype=np.float64)
    a = np.array([[7, 8]], dtype=np.int64)

    def kernel(data_arg, lut_arg, vmin, vmax, color_levels, transpbg, op):
        calls.append((data_arg.dtype, data_arg.shape, lut_arg.shape, vmin, vmax, color_levels, transpbg, op))
        return r.copy(), g.copy(), b.copy(), a.copy(), False

    monkeypatch.setattr(
        output_to_geotiff,
        "_rust_kernel",
        lambda name: kernel if name == "_geotiff_rgb_values_f64" else None,
    )

    actual = output_to_geotiff._get_rgb_values(data, 0.0, 1.0, None, "viridis", True, 0.25)

    assert calls == [(np.dtype(np.float64), (1, 2), (256, 4), 0.0, 1.0, 255.0, True, 0.25)]
    assert all(arr.dtype == np.int64 for arr in actual[:3])
    _assert_rgb_equal(actual, (r.astype(np.int64), g.astype(np.int64), b.astype(np.int64), a))


def test_geotiff_rgb_values_rust_runtime_error_keeps_python_path(monkeypatch):
    data = np.array([[0.0, 1.0]], dtype=np.float64)

    def rust_kernel(name):
        if name != "_geotiff_rgb_values_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(output_to_geotiff, "_rust_kernel", rust_kernel)
    actual = output_to_geotiff._get_rgb_values(data, 0.0, 1.0, None, "viridis", True, 0.5)
    expected = _fallback_rgb(
        data,
        monkeypatch,
        vmin=0.0,
        vmax=1.0,
        color_levels=None,
        cmap="viridis",
        transpbg=True,
        op=0.5,
    )
    _assert_rgb_equal(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=object),
        np.ma.array([[0.0, 1.0]], mask=[[False, True]], dtype=np.float64),
        np.arange(6, dtype=np.float64).reshape(2, 3)[:, ::2],
        np.zeros((1, 1, 1), dtype=np.float64),
    ],
)
def test_geotiff_rgb_values_unsupported_inputs_keep_python_path(monkeypatch, data):
    def rust_kernel(name):
        if name == "_geotiff_rgb_values_f64":
            raise AssertionError("unsupported GeoTIFF input used Rust kernel")
        return None

    monkeypatch.setattr(output_to_geotiff, "_rust_kernel", rust_kernel)
    try:
        actual = output_to_geotiff._get_rgb_values(data, 0.0, 1.0, None, "viridis", True, 0.5)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_rgb(
                data,
                monkeypatch,
                vmin=0.0,
                vmax=1.0,
                color_levels=None,
                cmap="viridis",
                transpbg=True,
                op=0.5,
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_rgb(
            data,
            monkeypatch,
            vmin=0.0,
            vmax=1.0,
            color_levels=None,
            cmap="viridis",
            transpbg=True,
            op=0.5,
        )
        _assert_rgb_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(vmin=0.0, vmax=1.0, color_levels=None, cmap="viridis", transpbg=True, op=0.5),
        dict(vmin=0.0, vmax=10.0, color_levels=7, cmap="plasma", transpbg=False, op=0.5),
    ],
)
@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust GeoTIFF RGB parity",
)
def test_geotiff_rgb_values_real_rust_matches_python_fallback(monkeypatch, kwargs):
    data = np.array([[-5.0, 0.0, 0.5 / 255.0], [0.5, 1.0, np.nan]], dtype=np.float64)
    expected = _fallback_rgb(data, monkeypatch, **kwargs)
    monkeypatch.undo()

    actual = output_to_geotiff._get_rgb_values(data, **kwargs)

    _assert_rgb_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust GeoTIFF checks",
)
def test_geotiff_rgb_values_direct_rust_helper():
    rust = _rust_or_skip()
    cmap = output_to_geotiff.plt.get_cmap("viridis")
    lut = np.asarray(cmap(np.arange(256)), dtype=np.float64)
    data = np.array([[0.5 / 255.0, np.nan]], dtype=np.float64)

    r, g, b, a, has_nan = rust._geotiff_rgb_values_f64(data, lut, 0.0, 1.0, 255.0, True, 0.5)

    assert has_nan is True
    assert r[0, 0] == int(np.round(lut[0, 0] * 255))
    assert np.isnan(r[0, 1])
    assert a.tolist() == [[128, 0]]
    with pytest.raises(ValueError):
        rust._geotiff_rgb_values_f64(data, lut[:10], 0.0, 1.0, 255.0, True, 0.5)
