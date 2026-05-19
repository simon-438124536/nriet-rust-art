import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.core import transforms  # noqa: E402


def _reference_antenna_to_cartesian(ranges, azimuths, elevations):
    theta_e = np.deg2rad(elevations)
    theta_a = np.deg2rad(azimuths)
    earth_radius = 6371.0 * 1000.0 * 4.0 / 3.0
    r = ranges * 1000.0

    z = (r**2 + earth_radius**2 + 2.0 * r * earth_radius * np.sin(theta_e)) ** 0.5
    z -= earth_radius
    s = earth_radius * np.arcsin(r * np.cos(theta_e) / (earth_radius + z))
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z


def _reference_cartesian_to_antenna(x, y, z):
    ranges = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    elevations = np.rad2deg(np.arctan(z / np.sqrt(x**2.0 + y**2.0)))
    azimuths = np.rad2deg(np.arctan2(x, y))
    azimuths[azimuths < 0.0] += 360.0
    return ranges, azimuths, elevations


def _reference_geographic_to_cartesian_aeqd(lon, lat, lon_0, lat_0, radius):
    lon = np.atleast_1d(np.asarray(lon))
    lat = np.atleast_1d(np.asarray(lat))

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)
    lon_diff_rad = lon_rad - lon_0_rad

    arg_arccos = np.sin(lat_0_rad) * np.sin(lat_rad) + np.cos(lat_0_rad) * np.cos(
        lat_rad
    ) * np.cos(lon_diff_rad)
    arg_arccos[arg_arccos > 1] = 1
    arg_arccos[arg_arccos < -1] = -1
    c = np.arccos(arg_arccos)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = c / np.sin(c)
    k[c == 0] = 1

    x = radius * k * np.cos(lat_rad) * np.sin(lon_diff_rad)
    y = (
        radius
        * k
        * (
            np.cos(lat_0_rad) * np.sin(lat_rad)
            - np.sin(lat_0_rad) * np.cos(lat_rad) * np.cos(lon_diff_rad)
        )
    )
    return x, y


def _reference_cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, radius):
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)
    rho = np.sqrt(x * x + y * y)
    c = rho / radius

    with np.errstate(divide="ignore", invalid="ignore"):
        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat_0_rad)
            + y * np.sin(c) * np.cos(lat_0_rad) / rho
        )
    lat_deg = np.rad2deg(lat_rad)
    lat_deg[rho == 0] = lat_0

    x1 = x * np.sin(c)
    x2 = rho * np.cos(lat_0_rad) * np.cos(c) - y * np.sin(lat_0_rad) * np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)
    lon_deg = np.rad2deg(lon_rad)
    lon_deg[lon_deg > 180] -= 360.0
    lon_deg[lon_deg < -180] += 360.0
    return lon_deg, lat_deg


def test_antenna_to_cartesian_matches_oracle_formula_for_broadcast_arrays():
    ranges = np.array([[1.0], [2.5]], dtype=np.float64)
    azimuths = np.array([[0.0, 90.0, 225.0]], dtype=np.float64)
    elevations = np.array([[0.5, 1.0, 2.0]], dtype=np.float64)

    actual = transforms.antenna_to_cartesian(ranges, azimuths, elevations)
    expected = _reference_antenna_to_cartesian(ranges, azimuths, elevations)

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-9)


def test_cartesian_to_antenna_matches_oracle_formula_for_arrays():
    x = np.array([[-10.0, 0.0, 10.0], [5.0, -5.0, 1.0]], dtype=np.float64)
    y = np.array([[10.0, -10.0, 0.0], [-5.0, 5.0, -1.0]], dtype=np.float64)
    z = np.array([[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]], dtype=np.float64)

    actual = transforms.cartesian_to_antenna(x, y, z)
    expected = _reference_cartesian_to_antenna(x, y, z)

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-12)


def test_antenna_to_cartesian_preserves_scalar_return_shape():
    actual = transforms.antenna_to_cartesian(1.0, 2.0, 3.0)
    expected = _reference_antenna_to_cartesian(1.0, 2.0, 3.0)

    assert all(isinstance(value, np.floating) for value in actual)
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-9)


def test_cartesian_to_antenna_preserves_oracle_scalar_error():
    with pytest.raises(TypeError):
        transforms.cartesian_to_antenna(1.0, 2.0, 3.0)


def test_float32_arrays_keep_numpy_oracle_dtype_when_rust_is_available(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("float32 inputs should use the NumPy oracle path")

        return kernel

    monkeypatch.setattr(transforms, "_rust_kernel", fail_if_called)

    ranges = np.array([1.0, 2.0], dtype=np.float32)
    azimuths = np.array([0.0, 90.0], dtype=np.float32)
    elevations = np.array([0.5, 1.0], dtype=np.float32)
    antenna_result = transforms.antenna_to_cartesian(ranges, azimuths, elevations)

    assert [part.dtype for part in antenna_result] == [np.float32, np.float32, np.float32]

    x = np.array([1.0, -1.0], dtype=np.float32)
    y = np.array([1.0, 1.0], dtype=np.float32)
    z = np.array([0.5, 0.75], dtype=np.float32)
    cartesian_result = transforms.cartesian_to_antenna(x, y, z)

    assert [part.dtype for part in cartesian_result] == [np.float32, np.float32, np.float32]


def test_masked_arrays_keep_numpy_oracle_path_when_rust_is_available(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("masked arrays should use the NumPy oracle path")

        return kernel

    monkeypatch.setattr(transforms, "_rust_kernel", fail_if_called)

    ranges = np.ma.array([1.0, 2.0], mask=[False, True], dtype=np.float64)
    azimuths = np.array([0.0, 90.0], dtype=np.float64)
    elevations = np.array([0.5, 1.0], dtype=np.float64)

    with np.errstate(invalid="ignore"):
        result = transforms.antenna_to_cartesian(ranges, azimuths, elevations)

    assert np.ma.isMaskedArray(result[0])
    np.testing.assert_array_equal(result[0].mask, np.array([False, True]))


def test_float64_arrays_dispatch_to_private_rust_transform_kernels(monkeypatch):
    calls = []

    def fake_rust_kernel(name):
        def antenna_kernel(ranges, azimuths, elevations):
            calls.append((name, ranges.dtype, azimuths.dtype, elevations.dtype))
            shape = np.broadcast_shapes(ranges.shape, azimuths.shape, elevations.shape)
            return (
                np.full(shape, 1.0),
                np.full(shape, 2.0),
                np.full(shape, 3.0),
            )

        def cartesian_kernel(x, y, z):
            calls.append((name, x.dtype, y.dtype, z.dtype))
            shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
            return (
                np.full(shape, 4.0),
                np.full(shape, 5.0),
                np.full(shape, 6.0),
            )

        return {
            "_antenna_to_cartesian": antenna_kernel,
            "_cartesian_to_antenna": cartesian_kernel,
        }.get(name)

    monkeypatch.setattr(transforms, "_rust_kernel", fake_rust_kernel)

    ranges = np.array([[1.0], [2.0]], dtype=np.float64)
    azimuths = np.array([[0.0, 90.0]], dtype=np.float64)
    elevations = np.array(0.5, dtype=np.float64)
    antenna_result = transforms.antenna_to_cartesian(ranges, azimuths, elevations)

    for actual, expected in zip(antenna_result, (1.0, 2.0, 3.0)):
        np.testing.assert_array_equal(actual, np.full((2, 2), expected))

    x = np.array([[1.0], [-1.0]], dtype=np.float64)
    y = np.array([[1.0, -1.0]], dtype=np.float64)
    z = np.array(0.5, dtype=np.float64)
    cartesian_result = transforms.cartesian_to_antenna(x, y, z)

    for actual, expected in zip(cartesian_result, (4.0, 5.0, 6.0)):
        np.testing.assert_array_equal(actual, np.full((2, 2), expected))

    assert calls == [
        ("_antenna_to_cartesian", np.float64, np.float64, np.float64),
        ("_cartesian_to_antenna", np.float64, np.float64, np.float64),
    ]


def test_geographic_to_cartesian_aeqd_matches_oracle_formula(monkeypatch):
    monkeypatch.setattr(transforms, "_rust_kernel", lambda _name: None)
    lon = np.array([[-97.0, -96.5], [-98.25, -97.75]], dtype=np.float64)
    lat = np.array([[36.0], [35.25]], dtype=np.float64)

    actual = transforms.geographic_to_cartesian_aeqd(lon, lat, -97.0, 36.0)
    expected = _reference_geographic_to_cartesian_aeqd(lon, lat, -97.0, 36.0, 6370997.0)

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-9)


def test_cartesian_to_geographic_aeqd_matches_oracle_formula(monkeypatch):
    monkeypatch.setattr(transforms, "_rust_kernel", lambda _name: None)
    x = np.array([[0.0, 1000.0], [-2000.0, 3500.0]], dtype=np.float64)
    y = np.array([[0.0], [1500.0]], dtype=np.float64)

    actual = transforms.cartesian_to_geographic_aeqd(x, y, -97.0, 36.0)
    expected = _reference_cartesian_to_geographic_aeqd(x, y, -97.0, 36.0, 6370997.0)

    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-12)


def test_aeqd_dispatches_to_private_rust_kernels_for_float64_arrays(monkeypatch):
    calls = []

    def fake_rust_kernel(name):
        def geographic_kernel(lon, lat, lon_0, lat_0, radius):
            calls.append((name, lon.dtype, lat.dtype, lon_0, lat_0, radius))
            shape = np.broadcast_shapes(lon.shape, lat.shape)
            return np.full(shape, 7.0), np.full(shape, 8.0)

        def cartesian_kernel(x, y, lon_0, lat_0, radius):
            calls.append((name, x.dtype, y.dtype, lon_0, lat_0, radius))
            shape = np.broadcast_shapes(x.shape, y.shape)
            return np.full(shape, 9.0), np.full(shape, 10.0)

        return {
            "_geographic_to_cartesian_aeqd": geographic_kernel,
            "_cartesian_to_geographic_aeqd": cartesian_kernel,
        }.get(name)

    monkeypatch.setattr(transforms, "_rust_kernel", fake_rust_kernel)

    lon = np.array([[1.0], [2.0]], dtype=np.float64)
    lat = np.array([[3.0, 4.0]], dtype=np.float64)
    actual = transforms.geographic_to_cartesian_aeqd(lon, lat, 5.0, 6.0, R=11.0)
    np.testing.assert_array_equal(actual[0], np.full((2, 2), 7.0))
    np.testing.assert_array_equal(actual[1], np.full((2, 2), 8.0))

    x = np.array([[1.0], [2.0]], dtype=np.float64)
    y = np.array([[3.0, 4.0]], dtype=np.float64)
    actual = transforms.cartesian_to_geographic_aeqd(x, y, 5.0, 6.0, R=11.0)
    np.testing.assert_array_equal(actual[0], np.full((2, 2), 9.0))
    np.testing.assert_array_equal(actual[1], np.full((2, 2), 10.0))

    assert calls == [
        ("_geographic_to_cartesian_aeqd", np.float64, np.float64, 5.0, 6.0, 11.0),
        ("_cartesian_to_geographic_aeqd", np.float64, np.float64, 5.0, 6.0, 11.0),
    ]


def test_aeqd_float32_arrays_keep_numpy_oracle_path(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("float32 inputs should use the NumPy oracle path")

        return kernel

    monkeypatch.setattr(transforms, "_rust_kernel", fail_if_called)

    lon = np.array([-97.0, -96.5], dtype=np.float32)
    lat = np.array([36.0, 35.25], dtype=np.float32)
    geographic_result = transforms.geographic_to_cartesian_aeqd(lon, lat, -97.0, 36.0)
    expected_geographic = _reference_geographic_to_cartesian_aeqd(
        lon, lat, -97.0, 36.0, 6370997.0
    )
    for actual_part, expected_part in zip(geographic_result, expected_geographic):
        np.testing.assert_allclose(actual_part, expected_part)

    x = np.array([0.0, 1000.0], dtype=np.float32)
    y = np.array([0.0, 1500.0], dtype=np.float32)
    cartesian_result = transforms.cartesian_to_geographic_aeqd(x, y, -97.0, 36.0)
    expected_cartesian = _reference_cartesian_to_geographic_aeqd(
        x, y, -97.0, 36.0, 6370997.0
    )
    for actual_part, expected_part in zip(cartesian_result, expected_cartesian):
        np.testing.assert_allclose(actual_part, expected_part)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_aeqd_kernels_match_numpy_fallback(monkeypatch):
    lon = np.array([[-97.0, -96.5], [-98.25, -97.75]], dtype=np.float64)
    lat = np.array([[36.0], [35.25]], dtype=np.float64)
    x = np.array([[0.0, 1000.0], [-2000.0, 3500.0]], dtype=np.float64)
    y = np.array([[0.0], [1500.0]], dtype=np.float64)

    monkeypatch.setattr(transforms, "_rust_kernel", lambda _name: None)
    expected_geo = transforms.geographic_to_cartesian_aeqd(lon, lat, -97.0, 36.0)
    expected_cart = transforms.cartesian_to_geographic_aeqd(x, y, -97.0, 36.0)

    import pyart._rust as rust

    monkeypatch.setattr(
        transforms,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual_geo = transforms.geographic_to_cartesian_aeqd(lon, lat, -97.0, 36.0)
    actual_cart = transforms.cartesian_to_geographic_aeqd(x, y, -97.0, 36.0)

    for actual_part, expected_part in zip(actual_geo, expected_geo):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-9)
    for actual_part, expected_part in zip(actual_cart, expected_cart):
        np.testing.assert_allclose(actual_part, expected_part, rtol=1.0e-12, atol=1.0e-12)
