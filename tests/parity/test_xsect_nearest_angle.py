import os

import numpy as np
import pytest

from pyart.testing import make_empty_ppi_radar, make_empty_rhi_radar
from pyart.util import xsect
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_xsect_nearest_angle_f64"):
        pytest.skip("pyart._rust has no xsect nearest-angle kernel")
    return rust


def _fallback_nearest(values, target, monkeypatch):
    monkeypatch.setattr(xsect, "_rust_kernel", lambda _name: None)
    return xsect._xsect_nearest_angle(values, target)


def _make_ppi_xsect_radar():
    radar = make_empty_ppi_radar(3, 4, 2)
    radar.azimuth["data"] = np.array([0.0, 90.0, 180.0, 270.0] * 2, dtype=np.float64)
    radar.elevation["data"] = np.array([0.5] * 4 + [1.5] * 4, dtype=np.float64)
    radar.fields["reflectivity"] = {
        "data": np.arange(radar.nrays * radar.ngates, dtype=np.float64).reshape(
            radar.nrays, radar.ngates
        ),
        "units": "dBZ",
    }
    return radar


def _make_rhi_xsect_radar():
    radar = make_empty_rhi_radar(3, 4, 2)
    radar.elevation["data"] = np.array([0.0, 1.0, 2.0, 3.0] * 2, dtype=np.float64)
    radar.azimuth["data"] = np.array([10.0] * 4 + [20.0] * 4, dtype=np.float64)
    radar.fields["reflectivity"] = {
        "data": np.arange(radar.nrays * radar.ngates, dtype=np.float64).reshape(
            radar.nrays, radar.ngates
        ),
        "units": "dBZ",
    }
    return radar


def _fallback_cross_section(func, radar, targets, monkeypatch, **kwargs):
    monkeypatch.setattr(xsect, "_rust_kernel", lambda _name: None)
    return func(radar, targets, **kwargs)


def _assert_xsect_radar_equal(actual, expected):
    assert actual.scan_type == expected.scan_type
    assert actual.nsweeps == expected.nsweeps
    for name in (
        "time",
        "azimuth",
        "elevation",
        "fixed_angle",
        "sweep_number",
        "sweep_mode",
        "sweep_start_ray_index",
        "sweep_end_ray_index",
    ):
        assert_exact_equal(getattr(actual, name)["data"], getattr(expected, name)["data"])
    assert actual.fields.keys() == expected.fields.keys()
    for field in actual.fields:
        assert_exact_equal(actual.fields[field]["data"], expected.fields[field]["data"])


@pytest.mark.parametrize(
    ("values", "target"),
    [
        (np.array([8.0, 10.0, 12.0], dtype=np.float64), 10.5),
        (np.array([9.5, 10.5], dtype=np.float64), 10.0),
        (np.array([9.0, np.nan, 10.0], dtype=np.float64), 10.0),
        (np.array([9.0, 10.0, 11.0], dtype=np.float64), np.nan),
    ],
)
def test_xsect_nearest_angle_python_fallback_reference(monkeypatch, values, target):
    index, distance = _fallback_nearest(values, target, monkeypatch)

    expected_distances = np.abs(values - target)
    expected_index = np.argmin(expected_distances)
    expected_distance = np.min(expected_distances)
    assert index == expected_index
    if np.isnan(expected_distance):
        assert np.isnan(distance)
    else:
        assert distance == expected_distance


def test_xsect_nearest_angle_dispatches_dense_float64_to_private_rust(monkeypatch):
    values = np.array([9.0, 10.0, 11.0], dtype=np.float64)
    calls = []

    def kernel(values_arg, target):
        calls.append((values_arg.dtype, values_arg.shape, target))
        return 2, 1.0

    monkeypatch.setattr(
        xsect,
        "_rust_kernel",
        lambda name: kernel if name == "_xsect_nearest_angle_f64" else None,
    )

    actual = xsect._xsect_nearest_angle(values, 10.0)

    assert calls == [(np.dtype(np.float64), (3,), 10.0)]
    assert actual == (2, 1.0)


def test_xsect_nearest_angle_runtime_error_keeps_python_path(monkeypatch):
    values = np.array([9.0, 10.0, 11.0], dtype=np.float64)

    def rust_kernel(name):
        if name != "_xsect_nearest_angle_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(xsect, "_rust_kernel", rust_kernel)
    actual = xsect._xsect_nearest_angle(values, 10.0)
    expected = _fallback_nearest(values, 10.0, monkeypatch)

    assert actual == expected


def test_xsect_nearest_angle_oversized_input_keeps_python_path(monkeypatch):
    values = np.array([9.0, 10.0, 11.0], dtype=np.float64)

    def rust_kernel(name):
        if name == "_xsect_nearest_angle_f64":
            raise AssertionError("oversized xsect input used Rust kernel")
        return None

    monkeypatch.setattr(xsect, "_rust_kernel", rust_kernel)
    monkeypatch.setattr(xsect, "XSECT_RUST_MAX_RAYS", 2)

    actual = xsect._xsect_nearest_angle(values, 10.0)
    expected = _fallback_nearest(values, 10.0, monkeypatch)

    assert actual == expected


@pytest.mark.parametrize(
    "case",
    [
        lambda: ([9.0, 10.0, 11.0], 10.0),
        lambda: (np.array([9.0, 10.0, 11.0], dtype=np.float32), 10.0),
        lambda: (np.arange(8, dtype=np.float64)[::2], 4.0),
        lambda: (np.array([[9.0, 10.0]], dtype=np.float64), 10.0),
        lambda: (np.array([], dtype=np.float64), 10.0),
        lambda: (np.array([9.0, 10.0, 11.0], dtype=object), 10.0),
        lambda: (np.ma.array([9.0, 10.0, 11.0], mask=[False, True, False]), 10.0),
        lambda: (np.array([9.0, 10.0, 11.0], dtype=np.float64), object()),
        lambda: (np.array([9.0, 10.0, 11.0], dtype=np.float64), True),
    ],
)
def test_xsect_nearest_angle_unsupported_inputs_keep_python_path(monkeypatch, case):
    values, target = case()

    def rust_kernel(name):
        if name == "_xsect_nearest_angle_f64":
            raise AssertionError("unsupported xsect input used Rust kernel")
        return None

    monkeypatch.setattr(xsect, "_rust_kernel", rust_kernel)
    try:
        actual = xsect._xsect_nearest_angle(values, target)
    except Exception as actual_error:
        expected_values, expected_target = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_nearest(expected_values, expected_target, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected_values, expected_target = case()
        expected = _fallback_nearest(expected_values, expected_target, monkeypatch)
        assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust xsect parity",
)
def test_xsect_nearest_angle_real_rust_matches_python_fallback(monkeypatch):
    values = np.array([9.5, 10.5, np.nan, 10.0], dtype=np.float64)
    expected = _fallback_nearest(values, 10.0, monkeypatch)
    monkeypatch.undo()

    actual = xsect._xsect_nearest_angle(values, 10.0)

    assert actual[0] == expected[0]
    if np.isnan(expected[1]):
        assert np.isnan(actual[1])
    else:
        assert actual[1] == expected[1]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust xsect checks",
)
def test_xsect_nearest_angle_direct_rust_helper():
    rust = _rust_or_skip()

    assert rust._xsect_nearest_angle_f64(
        np.array([9.5, 10.5], dtype=np.float64), 10.0
    ) == (0, 0.5)

    idx, dist = rust._xsect_nearest_angle_f64(
        np.array([9.0, np.nan, 10.0], dtype=np.float64), 10.0
    )
    assert idx == 1
    assert np.isnan(dist)

    with pytest.raises(ValueError):
        rust._xsect_nearest_angle_f64(np.array([], dtype=np.float64), 10.0)
    with pytest.raises(ValueError):
        rust._xsect_nearest_angle_f64(
            np.zeros(1024 * 1024 + 1, dtype=np.float64), 0.0
        )
    with pytest.raises(ValueError):
        rust._xsect_nearest_angle_f64(
            np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64)[::2], 10.0
        )
    with pytest.raises(ValueError):
        rust._xsect_nearest_angle_f64(np.array([9.0, 10.0], dtype=np.float64), True)


def test_cross_section_ppi_duplicate_targets_match_python_fallback(monkeypatch):
    expected = _fallback_cross_section(
        xsect.cross_section_ppi,
        _make_ppi_xsect_radar(),
        [180.0, 0.0, 180.0],
        monkeypatch,
    )
    monkeypatch.undo()

    actual = xsect.cross_section_ppi(
        _make_ppi_xsect_radar(), [180.0, 0.0, 180.0]
    )

    _assert_xsect_radar_equal(actual, expected)


def test_cross_section_rhi_preserves_target_order_like_python_fallback(monkeypatch):
    expected = _fallback_cross_section(
        xsect.cross_section_rhi,
        _make_rhi_xsect_radar(),
        [2.0, 0.0],
        monkeypatch,
    )
    monkeypatch.undo()

    actual = xsect.cross_section_rhi(_make_rhi_xsect_radar(), [2.0, 0.0])

    _assert_xsect_radar_equal(actual, expected)


@pytest.mark.parametrize(
    ("func", "radar_factory", "targets", "kwargs", "message"),
    [
        (
            xsect.cross_section_ppi,
            _make_ppi_xsect_radar,
            [10.0],
            {"az_tol": 1.0},
            "No azimuth found within tolerance",
        ),
        (
            xsect.cross_section_rhi,
            _make_rhi_xsect_radar,
            [10.0],
            {"el_tol": 1.0},
            "No elevation found within tolerance",
        ),
    ],
)
def test_cross_section_public_tolerance_error_matches_python_fallback(
    monkeypatch, func, radar_factory, targets, kwargs, message
):
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match=message) as expected_error:
            _fallback_cross_section(func, radar_factory(), targets, monkeypatch, **kwargs)
    monkeypatch.undo()

    with pytest.warns(UserWarning):
        with pytest.raises(type(expected_error.value)) as actual_error:
            func(radar_factory(), targets, **kwargs)

    assert actual_error.value.args == expected_error.value.args
