import os

import numpy as np
import pytest

from pyart.util import columnsect


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_columnsect_get_sweep_rays_f64"):
        pytest.skip("pyart._rust has no columnsect sweep-ray kernel")
    return rust


def _fallback_get_sweep_rays(sweep_azi, azimuth, azimuth_spread, monkeypatch):
    monkeypatch.setattr(columnsect, "_rust_kernel", lambda _name: None)
    return columnsect.get_sweep_rays(sweep_azi, azimuth, azimuth_spread)


@pytest.mark.parametrize(
    ("sweep_azi", "azimuth", "azimuth_spread"),
    [
        (np.array([9.5, 10.0, 10.49, 10.5, 11.0], dtype=np.float64), 10.0, 2),
        (np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64), 10.0, 1),
        (np.array([10.0, 9.75, 9.5, 10.25], dtype=np.float64), 10.0, 2),
        (np.array([9.0, 9.5, 10.0, 10.5], dtype=np.float64), 10.0, -1),
    ],
)
def test_get_sweep_rays_python_fallback_reference_cases(
    monkeypatch, sweep_azi, azimuth, azimuth_spread
):
    centerline, spread = _fallback_get_sweep_rays(
        sweep_azi, azimuth, azimuth_spread, monkeypatch
    )

    resolution = np.round((sweep_azi[1] - sweep_azi[0]), 3)
    expected_centerline = np.nonzero(np.abs(sweep_azi - azimuth) < 0.5)[0].tolist()
    expected_spread = np.nonzero(
        np.abs(sweep_azi - azimuth) < (resolution * azimuth_spread)
    )[0].tolist()
    assert centerline == expected_centerline
    assert spread == expected_spread


def test_get_sweep_rays_dispatches_dense_float64_to_private_rust(monkeypatch):
    sweep_azi = np.array([9.5, 10.0, 10.25, 10.75], dtype=np.float64)
    calls = []

    def kernel(sweep_arg, azimuth, spread_threshold):
        calls.append((sweep_arg.dtype, sweep_arg.shape, azimuth, spread_threshold))
        return np.array([1, 2], dtype=np.int64), np.array([0, 1, 2], dtype=np.int64)

    monkeypatch.setattr(
        columnsect,
        "_rust_kernel",
        lambda name: kernel if name == "_columnsect_get_sweep_rays_f64" else None,
    )

    centerline, spread = columnsect.get_sweep_rays(sweep_azi, 10.0, 4)

    assert calls == [(np.dtype(np.float64), (4,), 10.0, 2.0)]
    assert centerline == [1, 2]
    assert spread == [0, 1, 2]


def test_get_sweep_rays_rust_runtime_error_keeps_python_path(monkeypatch):
    sweep_azi = np.array([9.5, 10.0, 10.25, 10.75], dtype=np.float64)

    def rust_kernel(name):
        if name != "_columnsect_get_sweep_rays_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(columnsect, "_rust_kernel", rust_kernel)
    actual = columnsect.get_sweep_rays(sweep_azi, 10.0, 2)
    expected = _fallback_get_sweep_rays(sweep_azi, 10.0, 2, monkeypatch)

    assert actual == expected


def test_get_sweep_rays_oversized_input_keeps_python_path(monkeypatch):
    sweep_azi = np.array([9.5, 10.0, 10.25, 10.75], dtype=np.float64)

    def rust_kernel(name):
        if name == "_columnsect_get_sweep_rays_f64":
            raise AssertionError("oversized columnsect input used Rust kernel")
        return None

    monkeypatch.setattr(columnsect, "_rust_kernel", rust_kernel)
    monkeypatch.setattr(columnsect, "COLUMNSECT_RUST_MAX_RAYS", 2)

    actual = columnsect.get_sweep_rays(sweep_azi, 10.0, 2)
    expected = _fallback_get_sweep_rays(sweep_azi, 10.0, 2, monkeypatch)

    assert actual == expected


@pytest.mark.parametrize(
    "case",
    [
        lambda: ([9.5, 10.0, 10.5], 10.0, 2),
        lambda: (np.array([9.5, 10.0, 10.5], dtype=np.float32), 10.0, 2),
        lambda: (np.arange(8, dtype=np.float64)[::2], 4.0, 2),
        lambda: (np.array([[9.5, 10.0], [10.5, 11.0]], dtype=np.float64), 10.0, 2),
        lambda: (np.array([10.0], dtype=np.float64), 10.0, 2),
        lambda: (np.array([9.5, 10.0, 10.5], dtype=object), 10.0, 2),
        lambda: (
            np.ma.array([9.5, 10.0, 10.5], mask=[False, True, False]),
            10.0,
            2,
        ),
        lambda: (np.array([9.5, np.nan, 10.5], dtype=np.float64), 10.0, 2),
        lambda: (np.array([9.5, np.inf, 10.5], dtype=np.float64), 10.0, 2),
        lambda: (np.array([9.5, 10.0, 10.5], dtype=np.float64), np.array(10.0), 2),
    ],
)
def test_get_sweep_rays_unsupported_inputs_keep_python_path(monkeypatch, case):
    sweep_azi, azimuth, azimuth_spread = case()

    def rust_kernel(name):
        if name == "_columnsect_get_sweep_rays_f64":
            raise AssertionError("unsupported columnsect input used Rust kernel")
        return None

    monkeypatch.setattr(columnsect, "_rust_kernel", rust_kernel)
    try:
        actual = columnsect.get_sweep_rays(sweep_azi, azimuth, azimuth_spread)
    except Exception as actual_error:
        expected_sweep, expected_azimuth, expected_spread = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_get_sweep_rays(
                expected_sweep, expected_azimuth, expected_spread, monkeypatch
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected_sweep, expected_azimuth, expected_spread = case()
        expected = _fallback_get_sweep_rays(
            expected_sweep, expected_azimuth, expected_spread, monkeypatch
        )
        assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust columnsect parity",
)
def test_get_sweep_rays_real_rust_matches_python_fallback(monkeypatch):
    sweep_azi = np.array([9.0, 9.5, 10.0, 10.49, 10.5, 11.0], dtype=np.float64)
    expected = _fallback_get_sweep_rays(sweep_azi, 10.0, 2, monkeypatch)
    monkeypatch.undo()

    actual = columnsect.get_sweep_rays(sweep_azi, 10.0, 2)

    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust columnsect checks",
)
def test_get_sweep_rays_direct_rust_helper():
    rust = _rust_or_skip()
    sweep_azi = np.array([9.5, 10.0, 10.49, 10.5, 11.0], dtype=np.float64)

    centerline, spread = rust._columnsect_get_sweep_rays_f64(sweep_azi, 10.0, 1.0)

    assert centerline.dtype == np.int64
    assert spread.dtype == np.int64
    assert centerline.tolist() == [1, 2]
    assert spread.tolist() == [0, 1, 2, 3]

    with pytest.raises(ValueError):
        rust._columnsect_get_sweep_rays_f64(
            np.array([10.0], dtype=np.float64), 10.0, 1.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_get_sweep_rays_f64(
            np.array([10.0, np.nan], dtype=np.float64), 10.0, 1.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_get_sweep_rays_f64(
            np.zeros(1024 * 1024 + 1, dtype=np.float64), 0.0, 1.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_get_sweep_rays_f64(sweep_azi[::2], 10.0, 1.0)
    with pytest.raises(ValueError):
        rust._columnsect_get_sweep_rays_f64(sweep_azi, True, 1.0)
    with pytest.raises(Exception):
        rust._columnsect_get_sweep_rays_f64(
            np.array([10.0, 10.5], dtype=np.float32), 10.0, 1.0
        )
