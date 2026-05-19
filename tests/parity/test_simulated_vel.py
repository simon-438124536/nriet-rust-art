import os

import numpy as np
import pytest

from pyart.core import HorizontalWindProfile
from pyart.testing import make_empty_ppi_radar
from pyart.util import simulated_vel
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_simulated_radial_velocity_dense_f64"):
        pytest.skip("pyart._rust has no simulated velocity kernel")
    return rust


def _fallback_velocity(gate_u, gate_v, azimuths, elevations, monkeypatch):
    monkeypatch.setattr(simulated_vel, "_rust_kernel", lambda _name: None)
    return simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)


def _sample_velocity_inputs():
    gate_u = np.ma.masked_invalid(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    gate_v = np.ma.masked_invalid(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
    azimuths = np.deg2rad(np.array([0.0, 90.0], dtype=np.float32)).reshape(-1, 1)
    elevations = np.deg2rad(np.array([0.0, 30.0], dtype=np.float32)).reshape(-1, 1)
    return gate_u, gate_v, azimuths, elevations


def _assert_masked_velocity_equal(actual, expected, require_dense_mask=True):
    assert np.ma.isMaskedArray(actual)
    if require_dense_mask:
        assert isinstance(actual.mask, np.ndarray)
    assert actual.fill_value == np.float64(1.0e20)
    assert_exact_equal(actual, expected)


def test_simulated_radial_velocity_python_fallback_reference(monkeypatch):
    gate_u, gate_v, azimuths, elevations = _sample_velocity_inputs()

    actual = _fallback_velocity(gate_u, gate_v, azimuths, elevations, monkeypatch)

    assert actual.dtype == np.float64
    assert actual.shape == (2, 2)
    assert isinstance(actual.mask, np.ndarray)
    assert actual.mask.tolist() == [[False, False], [False, False]]
    assert actual.fill_value == np.float64(1.0e20)


def test_simulated_radial_velocity_dispatches_dense_float64_to_private_rust(monkeypatch):
    gate_u, gate_v, azimuths, elevations = _sample_velocity_inputs()
    values = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float64)
    calls = []

    def kernel(gate_u_arg, gate_v_arg, sin_az_arg, cos_az_arg, cos_el_arg):
        calls.append(
            (
                gate_u_arg.dtype,
                gate_u_arg.shape,
                gate_v_arg.dtype,
                gate_v_arg.shape,
                sin_az_arg.dtype,
                sin_az_arg.shape,
                cos_az_arg.shape,
                cos_el_arg.shape,
            )
        )
        return values.copy()

    monkeypatch.setattr(
        simulated_vel,
        "_rust_kernel",
        lambda name: kernel if name == "_simulated_radial_velocity_dense_f64" else None,
    )

    actual = simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)

    assert calls == [
        (
            np.dtype(np.float64),
            (2, 2),
            np.dtype(np.float64),
            (2, 2),
            np.dtype(np.float64),
            (2,),
            (2,),
            (2,),
        )
    ]
    assert actual.dtype == np.float64
    assert actual.mask.tolist() == [[False, False], [False, False]]
    assert_exact_equal(actual.data, values)
    assert actual.fill_value == np.float64(1.0e20)


def test_simulated_radial_velocity_rust_runtime_error_keeps_python_path(monkeypatch):
    gate_u, gate_v, azimuths, elevations = _sample_velocity_inputs()

    def rust_kernel(name):
        if name != "_simulated_radial_velocity_dense_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(simulated_vel, "_rust_kernel", rust_kernel)
    actual = simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)
    expected = _fallback_velocity(gate_u, gate_v, azimuths, elevations, monkeypatch)

    _assert_masked_velocity_equal(actual, expected)


def test_simulated_radial_velocity_oversized_output_keeps_python_path(monkeypatch):
    gate_u, gate_v, azimuths, elevations = _sample_velocity_inputs()

    def rust_kernel(name):
        if name == "_simulated_radial_velocity_dense_f64":
            raise AssertionError("oversized simulated velocity input used Rust kernel")
        return None

    monkeypatch.setattr(simulated_vel, "_rust_kernel", rust_kernel)
    monkeypatch.setattr(simulated_vel, "SIMULATED_VEL_RUST_MAX_OUTPUT_VALUES", 3)

    actual = simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)
    monkeypatch.setattr(simulated_vel, "_rust_kernel", lambda _name: None)
    expected = simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)

    _assert_masked_velocity_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        lambda: (
            np.ma.masked_invalid(np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)),
            np.ma.masked_invalid(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)),
            np.deg2rad(np.array([0.0, 90.0])).reshape(-1, 1),
            np.deg2rad(np.array([0.0, 30.0])).reshape(-1, 1),
        ),
        lambda: (
            np.ma.masked_invalid(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            np.ma.masked_invalid(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)),
            np.deg2rad(np.array([0.0, 90.0])).reshape(-1, 1),
            np.deg2rad(np.array([0.0, 30.0])).reshape(-1, 1),
        ),
        lambda: (
            np.ma.array(np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::3], copy=False),
            np.ma.array(np.arange(12, 24, dtype=np.float64).reshape(2, 6)[:, ::3], copy=False),
            np.deg2rad(np.array([0.0, 90.0])).reshape(-1, 1),
            np.deg2rad(np.array([0.0, 30.0])).reshape(-1, 1),
        ),
        lambda: (
            np.ma.masked_invalid(np.array([1.0, 2.0], dtype=np.float64)),
            np.ma.masked_invalid(np.array([5.0, 6.0], dtype=np.float64)),
            np.deg2rad(np.array([0.0, 90.0])).reshape(-1, 1),
            np.deg2rad(np.array([0.0, 30.0])).reshape(-1, 1),
        ),
    ],
)
def test_simulated_radial_velocity_unsupported_inputs_keep_python_path(monkeypatch, case):
    gate_u, gate_v, azimuths, elevations = case()

    def rust_kernel(name):
        if name == "_simulated_radial_velocity_dense_f64":
            raise AssertionError("unsupported simulated velocity input used Rust kernel")
        return None

    monkeypatch.setattr(simulated_vel, "_rust_kernel", rust_kernel)
    try:
        actual = simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)
    except Exception as actual_error:
        expected_gate_u, expected_gate_v, expected_azimuths, expected_elevations = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_velocity(
                expected_gate_u,
                expected_gate_v,
                expected_azimuths,
                expected_elevations,
                monkeypatch,
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected_gate_u, expected_gate_v, expected_azimuths, expected_elevations = case()
        expected = _fallback_velocity(
            expected_gate_u,
            expected_gate_v,
            expected_azimuths,
            expected_elevations,
            monkeypatch,
        )
        _assert_masked_velocity_equal(actual, expected, require_dense_mask=False)


def test_simulated_vel_from_profile_dispatch_surface(monkeypatch):
    radar = make_empty_ppi_radar(2, 2, 1)
    radar.azimuth["data"] = np.array([0.0, 90.0], dtype=np.float32)
    radar.elevation["data"] = np.array([0.0, 0.0], dtype=np.float32)
    radar.gate_altitude["data"] = np.array([[200.0, 300.0], [400.0, 500.0]], dtype=np.float64)
    profile = HorizontalWindProfile.from_u_and_v(
        np.array([0.0, 1000.0], dtype=np.float64),
        np.array([1.0, 3.0], dtype=np.float64),
        np.array([5.0, 7.0], dtype=np.float64),
    )
    calls = []

    def kernel(gate_u_arg, gate_v_arg, sin_az_arg, cos_az_arg, cos_el_arg):
        calls.append((gate_u_arg.shape, gate_v_arg.shape, sin_az_arg.shape, cos_el_arg.shape))
        return np.full(gate_u_arg.shape, 42.0, dtype=np.float64)

    monkeypatch.setattr(
        simulated_vel,
        "_rust_kernel",
        lambda name: kernel if name == "_simulated_radial_velocity_dense_f64" else None,
    )

    actual = simulated_vel.simulated_vel_from_profile(radar, profile)

    assert calls == [((2, 2), (2, 2), (2,), (2,))]
    assert actual["data"].shape == (2, 2)
    assert actual["data"].mask.tolist() == [[False, False], [False, False]]
    assert actual["data"].data.tolist() == [[42.0, 42.0], [42.0, 42.0]]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust simulated velocity parity",
)
def test_simulated_radial_velocity_real_rust_matches_python_fallback(monkeypatch):
    gate_u, gate_v, azimuths, elevations = _sample_velocity_inputs()
    expected = _fallback_velocity(gate_u, gate_v, azimuths, elevations, monkeypatch)
    monkeypatch.undo()

    actual = simulated_vel._simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)

    _assert_masked_velocity_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust simulated velocity checks",
)
def test_simulated_radial_velocity_direct_rust_helper():
    rust = _rust_or_skip()
    gate_u = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    gate_v = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    azimuths = np.deg2rad(np.array([0.0, 90.0]))
    elevations = np.deg2rad(np.array([0.0, 30.0]))

    actual = rust._simulated_radial_velocity_dense_f64(
        gate_u,
        gate_v,
        np.sin(azimuths),
        np.cos(azimuths),
        np.cos(elevations),
    )
    expected = gate_u * np.sin(azimuths).reshape(-1, 1) * np.cos(
        elevations
    ).reshape(-1, 1) + gate_v * np.cos(azimuths).reshape(-1, 1) * np.cos(
        elevations
    ).reshape(-1, 1)

    assert_exact_equal(actual, expected)
    with pytest.raises(ValueError):
        rust._simulated_radial_velocity_dense_f64(
            gate_u,
            gate_v[:, :1],
            np.sin(azimuths),
            np.cos(azimuths),
            np.cos(elevations),
        )
