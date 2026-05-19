import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import qvp  # noqa: E402


class DummyQvpRadar:
    def __init__(self, azimuths, ranges, gate_latitude=None, gate_longitude=None):
        self.azimuth = {"data": azimuths}
        self.range = {"data": ranges}
        if gate_latitude is not None:
            self.gate_latitude = {"data": gate_latitude}
        if gate_longitude is not None:
            self.gate_longitude = {"data": gate_longitude}


def _fallback_find_rng_index(values, target, tolerance, monkeypatch):
    monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
    return qvp.find_rng_index(values, target, rng_tol=tolerance)


def _fallback_find_ang_index(values, target, tolerance, monkeypatch):
    monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
    return qvp.find_ang_index(values, target, ang_tol=tolerance)


def _fallback_find_neighbour_gates(radar, azi, rng, delta_azi, delta_rng, monkeypatch):
    monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
    return qvp.find_neighbour_gates(
        radar, azi, rng, delta_azi=delta_azi, delta_rng=delta_rng
    )


def _fallback_find_nearest_gate(radar, lat, lon, latlon_tol, monkeypatch):
    monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
    return qvp.find_nearest_gate(radar, lat, lon, latlon_tol=latlon_tol)


@pytest.mark.parametrize("func_name", ["find_rng_index", "find_ang_index"])
def test_qvp_find_index_python_fallback_preserves_oracle_return_type(
    monkeypatch, func_name
):
    values = np.array([0.0, 10.0, 20.0, 20.0], dtype=np.float64)
    func = getattr(qvp, func_name)
    monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)

    actual = func(values, 20.0, 0.0)

    assert type(actual) is np.int64
    assert actual == np.int64(2)


@pytest.mark.parametrize(
    ("values", "target", "tolerance", "expected"),
    [
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), 10.4, 0.4, np.int64(1)),
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), 10.4, 0.399, None),
        (np.array([0.0, 10.0, 20.0, 20.0], dtype=np.float64), 20.0, 0.0, np.int64(2)),
    ],
)
def test_qvp_find_index_dispatches_to_private_rust_kernel(
    monkeypatch, values, target, tolerance, expected
):
    calls = []

    def rust_kernel(values, target, tolerance):
        calls.append((values.dtype, values.shape, target, tolerance))
        return expected

    monkeypatch.setattr(
        qvp,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qvp_find_index_dense" else None,
    )

    actual = qvp.find_rng_index(values, target, rng_tol=tolerance)

    assert calls == [(np.float64, values.shape, float(target), float(tolerance))]
    assert actual is expected or actual == expected
    if expected is not None:
        assert type(actual) is np.int64


@pytest.mark.parametrize(
    ("values", "target", "tolerance"),
    [
        (np.array([], dtype=np.float64), 1.0, 0.0),
        (np.array([np.nan, 1.0], dtype=np.float64), 1.0, 0.0),
        (np.array([np.inf, 1.0], dtype=np.float64), 1.0, 0.0),
        (np.array([0.0, 1.0], dtype=np.float32), 1.0, 0.0),
        (np.array([0.0, 1.0, 2.0], dtype=np.float64)[::2], 1.0, 1.0),
        ([0.0, 1.0], 1.0, 0.0),
    ],
)
def test_qvp_find_index_keeps_python_path_for_unsupported_inputs(
    monkeypatch, values, target, tolerance
):
    try:
        expected = _fallback_find_rng_index(values, target, tolerance, monkeypatch)
    except Exception as expected_error:
        expected_error_type = type(expected_error)
    else:
        expected_error_type = None

    def fail_if_called(name):
        if name != "_qvp_find_index_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported qvp find-index input used Rust")

        return kernel

    monkeypatch.setattr(qvp, "_rust_kernel", fail_if_called)
    if expected_error_type is not None:
        with pytest.raises(expected_error_type):
            qvp.find_rng_index(values, target, rng_tol=tolerance)
    else:
        actual = qvp.find_rng_index(values, target, rng_tol=tolerance)
        assert actual == expected
        if expected is not None:
            assert type(actual) is type(expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("func_name", ["find_rng_index", "find_ang_index"])
@pytest.mark.parametrize(
    ("target", "tolerance"),
    [(10.4, 0.4), (10.4, 0.399), (20.0, 0.0)],
)
def test_real_rust_qvp_find_index_matches_python_fallback(
    monkeypatch, func_name, target, tolerance
):
    values = np.array([0.0, 10.0, 20.0, 20.0], dtype=np.float64)
    fallback = (
        _fallback_find_rng_index
        if func_name == "find_rng_index"
        else _fallback_find_ang_index
    )
    expected = fallback(values, target, tolerance, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        qvp,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    func = getattr(qvp, func_name)
    keyword = "rng_tol" if func_name == "find_rng_index" else "ang_tol"
    actual = func(values, target, **{keyword: tolerance})

    assert actual == expected
    if expected is not None:
        assert type(actual) is type(expected) is np.int64


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("values", "target", "tolerance", "match"),
    [
        (np.array([], dtype=np.float64), 1.0, 0.0, "empty sequence"),
        (np.array([np.nan, 1.0], dtype=np.float64), 1.0, 0.0, "finite"),
        (
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)[::2],
            1.0,
            0.0,
            "C-contiguous",
        ),
    ],
)
def test_real_rust_qvp_find_index_rejects_unsafe_direct_inputs(
    values, target, tolerance, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._qvp_find_index_dense(values, target, tolerance)


@pytest.mark.parametrize(
    ("azimuths", "ranges", "azi", "rng", "delta_azi", "delta_rng"),
    [
        (
            np.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64),
            np.array([0.0, 500.0, 1000.0, 1500.0], dtype=np.float64),
            10.0,
            750.0,
            6.0,
            600.0,
        ),
        (
            np.array([350.0, 355.0, 0.0, 5.0, 10.0, 20.0], dtype=np.float64),
            np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0], dtype=np.float64),
            0.0,
            1000.0,
            10.0,
            500.0,
        ),
        (
            np.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64),
            np.array([0.0, 500.0, 1000.0, 1500.0], dtype=np.float64),
            10.0,
            750.0,
            -6.0,
            -100.0,
        ),
    ],
)
def test_qvp_find_neighbour_gates_matches_python_fallback(
    monkeypatch, azimuths, ranges, azi, rng, delta_azi, delta_rng
):
    radar = DummyQvpRadar(azimuths, ranges)
    expected_rays, expected_ranges = _fallback_find_neighbour_gates(
        radar, azi, rng, delta_azi, delta_rng, monkeypatch
    )

    actual_rays, actual_ranges = qvp.find_neighbour_gates(
        radar, azi, rng, delta_azi=delta_azi, delta_rng=delta_rng
    )

    np.testing.assert_array_equal(actual_rays, expected_rays)
    np.testing.assert_array_equal(actual_ranges, expected_ranges)
    assert actual_rays.dtype == expected_rays.dtype
    assert actual_ranges.dtype == expected_ranges.dtype


def test_qvp_find_neighbour_gates_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(azimuths, ranges, azi, rng, delta_azi, delta_rng):
        calls.append(
            (
                azimuths.dtype,
                azimuths.shape,
                ranges.dtype,
                ranges.shape,
                azi,
                rng,
                delta_azi,
                delta_rng,
            )
        )
        return np.array([2, 3], dtype=np.int64), np.array([1], dtype=np.int64)

    monkeypatch.setattr(
        qvp,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qvp_find_neighbour_gates_dense" else None,
    )
    radar = DummyQvpRadar(
        np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64),
        np.array([0.0, 500.0, 1000.0], dtype=np.float64),
    )

    rays, ranges = qvp.find_neighbour_gates(
        radar, 10, 750, delta_azi=np.float64(6.0), delta_rng=600
    )

    assert calls == [
        (np.float64, (4,), np.float64, (3,), 10.0, 750.0, 6.0, 600.0)
    ]
    np.testing.assert_array_equal(rays, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(ranges, np.array([1], dtype=np.int64))


@pytest.mark.parametrize(
    ("azimuths", "ranges", "azi", "rng", "delta_azi", "delta_rng"),
    [
        (
            np.array([0.0, 10.0], dtype=np.float64),
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            None,
            100.0,
        ),
        (
            np.array([0.0, 10.0], dtype=np.float64),
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            10.0,
            None,
        ),
        (
            np.array([0.0, 10.0], dtype=np.float32),
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            10.0,
            100.0,
        ),
        (
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            10.0,
            100.0,
        ),
        (
            np.array([0.0, 10.0, 20.0], dtype=np.float64)[::2],
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            10.0,
            100.0,
        ),
        (
            [0.0, 10.0],
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            10.0,
            100.0,
        ),
        (
            np.ma.array([0.0, 10.0], mask=[False, True], dtype=np.float64),
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            10.0,
            100.0,
        ),
        (
            np.array([0.0, 10.0], dtype=np.float64),
            np.array([0.0, 1000.0], dtype=np.float64),
            0.0,
            0.0,
            "10",
            100.0,
        ),
    ],
)
def test_qvp_find_neighbour_gates_keeps_python_path_for_unsupported_inputs(
    monkeypatch, azimuths, ranges, azi, rng, delta_azi, delta_rng
):
    radar = DummyQvpRadar(azimuths, ranges)

    def fail_if_called(name):
        if name != "_qvp_find_neighbour_gates_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported qvp neighbour input used Rust")

        return kernel

    monkeypatch.setattr(qvp, "_rust_kernel", fail_if_called)
    try:
        actual = qvp.find_neighbour_gates(
            radar, azi, rng, delta_azi=delta_azi, delta_rng=delta_rng
        )
    except Exception as actual_error:
        monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            qvp.find_neighbour_gates(
                radar, azi, rng, delta_azi=delta_azi, delta_rng=delta_rng
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_find_neighbour_gates(
            radar, azi, rng, delta_azi, delta_rng, monkeypatch
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert type(actual[0]) is type(expected[0])
        assert type(actual[1]) is type(expected[1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("azi", "rng", "delta_azi", "delta_rng"),
    [(10.0, 750.0, 6.0, 600.0), (0.0, 1000.0, 10.0, 500.0)],
)
def test_real_rust_qvp_find_neighbour_gates_matches_python_fallback(
    monkeypatch, azi, rng, delta_azi, delta_rng
):
    import pyart._rust as rust

    kernel = getattr(rust, "_qvp_find_neighbour_gates_dense", None)
    assert kernel is not None

    radar = DummyQvpRadar(
        np.array([350.0, 355.0, 0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64),
        np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0], dtype=np.float64),
    )
    expected = _fallback_find_neighbour_gates(
        radar, azi, rng, delta_azi, delta_rng, monkeypatch
    )
    calls = []

    def rust_kernel(name):
        calls.append(name)
        return kernel if name == "_qvp_find_neighbour_gates_dense" else None

    monkeypatch.setattr(qvp, "_rust_kernel", rust_kernel)
    actual = qvp.find_neighbour_gates(
        radar, azi, rng, delta_azi=delta_azi, delta_rng=delta_rng
    )

    assert calls == ["_qvp_find_neighbour_gates_dense"]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[0].dtype == expected[0].dtype
    assert actual[1].dtype == expected[1].dtype


def test_qvp_find_nearest_gate_python_fallback_preserves_oracle_return_type(
    monkeypatch,
):
    radar = DummyQvpRadar(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1000.0, 2000.0], dtype=np.float64),
        np.array([[10.0, 10.0001], [9.9999, 10.0002]], dtype=np.float64),
        np.array([[20.0, 20.0001], [20.0001, 20.0002]], dtype=np.float64),
    )

    actual = _fallback_find_nearest_gate(radar, 10.0, 20.0, 0.0005, monkeypatch)

    assert type(actual[0]) is np.int64
    assert type(actual[1]) is np.int64
    assert type(actual[2]) is np.float64
    assert type(actual[3]) is np.float64
    assert actual == (np.int64(0), np.int64(0), np.float64(0.0), np.float64(1000.0))


def test_qvp_find_nearest_gate_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(gate_latitude, gate_longitude, lat, lon, latlon_tol):
        calls.append(
            (
                gate_latitude.dtype,
                gate_latitude.shape,
                gate_longitude.dtype,
                gate_longitude.shape,
                lat,
                lon,
                latlon_tol,
            )
        )
        return np.int64(1), np.int64(0)

    monkeypatch.setattr(
        qvp,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qvp_find_nearest_gate_dense" else None,
    )
    radar = DummyQvpRadar(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1000.0, 2000.0], dtype=np.float64),
        np.array([[10.0, 10.0001], [9.9999, 10.0002]], dtype=np.float64),
        np.array([[20.0, 20.0001], [20.0001, 20.0002]], dtype=np.float64),
    )

    actual = qvp.find_nearest_gate(radar, 10, 20, latlon_tol=np.float64(0.0005))

    assert calls == [
        (np.float64, (2, 2), np.float64, (2, 2), 10.0, 20.0, 0.0005)
    ]
    assert actual == (np.int64(1), np.int64(0), np.float64(1.0), np.float64(1000.0))


def test_qvp_find_nearest_gate_rust_none_preserves_warning(monkeypatch):
    def rust_kernel(*_args):
        return None

    monkeypatch.setattr(
        qvp,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qvp_find_nearest_gate_dense" else None,
    )
    radar = DummyQvpRadar(
        np.array([0.0], dtype=np.float64),
        np.array([1000.0], dtype=np.float64),
        np.array([[10.0]], dtype=np.float64),
        np.array([[20.0]], dtype=np.float64),
    )

    with pytest.warns(UserWarning, match="No data found at point lat 11"):
        actual = qvp.find_nearest_gate(radar, 11.0, 21.0, latlon_tol=0.0005)

    assert actual == (None, None, None, None)


@pytest.mark.parametrize(
    ("gate_latitude", "gate_longitude", "lat", "lon", "latlon_tol"),
    [
        (
            np.array([[10.0, np.nan]], dtype=np.float64),
            np.array([[20.0, 20.1]], dtype=np.float64),
            10.0,
            20.0,
            0.0005,
        ),
        (
            np.array([[10.0, 10.1]], dtype=np.float32),
            np.array([[20.0, 20.1]], dtype=np.float64),
            10.0,
            20.0,
            0.0005,
        ),
        (
            np.array([[10.0, 10.1, 10.2, 10.3]], dtype=np.float64)[:, ::2],
            np.array([[20.0, 20.1]], dtype=np.float64),
            10.0,
            20.0,
            0.0005,
        ),
        (
            np.array([[10.0, 10.1]], dtype=np.float64),
            np.array([[20.0]], dtype=np.float64),
            10.0,
            20.0,
            0.0005,
        ),
        (
            np.ma.array([[10.0, 10.1]], mask=[[False, True]], dtype=np.float64),
            np.array([[20.0, 20.1]], dtype=np.float64),
            10.0,
            20.0,
            0.0005,
        ),
        (
            np.array([[10.0, 10.1]], dtype=np.float64),
            np.array([[20.0, 20.1]], dtype=np.float64),
            np.nan,
            20.0,
            0.0005,
        ),
    ],
)
def test_qvp_find_nearest_gate_keeps_python_path_for_unsupported_inputs(
    monkeypatch, gate_latitude, gate_longitude, lat, lon, latlon_tol
):
    radar = DummyQvpRadar(
        np.array([0.0], dtype=np.float64),
        np.array([1000.0, 2000.0], dtype=np.float64),
        gate_latitude,
        gate_longitude,
    )

    def fail_if_called(name):
        if name != "_qvp_find_nearest_gate_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported qvp nearest input used Rust")

        return kernel

    monkeypatch.setattr(qvp, "_rust_kernel", fail_if_called)
    with warnings.catch_warnings(record=True):
        try:
            actual = qvp.find_nearest_gate(
                radar, lat, lon, latlon_tol=latlon_tol
            )
        except Exception as actual_error:
            monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)) as expected_error:
                qvp.find_nearest_gate(radar, lat, lon, latlon_tol=latlon_tol)
            assert actual_error.args == expected_error.value.args
        else:
            expected = _fallback_find_nearest_gate(
                radar, lat, lon, latlon_tol, monkeypatch
            )
            assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_qvp_find_nearest_gate_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    kernel = getattr(rust, "_qvp_find_nearest_gate_dense")
    radar = DummyQvpRadar(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1000.0, 2000.0], dtype=np.float64),
        np.array([[10.0, 10.0001], [9.9999, 10.0002]], dtype=np.float64),
        np.array([[20.0, 20.0001], [20.0001, 20.0002]], dtype=np.float64),
    )
    expected = _fallback_find_nearest_gate(radar, 10.0, 20.0, 0.0005, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_qvp_find_nearest_gate_dense":
            calls.append(name)
            return kernel
        return None

    monkeypatch.setattr(qvp, "_rust_kernel", rust_kernel)
    actual = qvp.find_nearest_gate(radar, 10.0, 20.0, latlon_tol=0.0005)

    assert calls == ["_qvp_find_nearest_gate_dense"]
    assert actual == expected
    assert type(actual[0]) is type(expected[0])
    assert type(actual[1]) is type(expected[1])
    assert type(actual[2]) is type(expected[2])
    assert type(actual[3]) is type(expected[3])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("gate_latitude", "gate_longitude", "match"),
    [
        (
            np.array([[10.0, 10.1, 10.2, 10.3]], dtype=np.float64)[:, ::2],
            np.array([[20.0, 20.1]], dtype=np.float64),
            "C-contiguous",
        ),
        (
            np.array([[10.0, np.nan]], dtype=np.float64),
            np.array([[20.0, 20.1]], dtype=np.float64),
            "finite",
        ),
        (
            np.array([[10.0, 10.1]], dtype=np.float64),
            np.array([[20.0]], dtype=np.float64),
            "same shape",
        ),
    ],
)
def test_real_rust_qvp_find_nearest_gate_rejects_unsafe_direct_inputs(
    gate_latitude, gate_longitude, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._qvp_find_nearest_gate_dense(
            gate_latitude, gate_longitude, 10.0, 20.0, 0.0005
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("azimuths", "ranges", "match"),
    [
        (
            np.array([0.0, 10.0, 20.0], dtype=np.float64)[::2],
            np.array([0.0, 1000.0], dtype=np.float64),
            "C-contiguous",
        ),
        (
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.0, 1000.0], dtype=np.float64),
            "finite",
        ),
    ],
)
def test_real_rust_qvp_find_neighbour_gates_rejects_unsafe_direct_inputs(
    azimuths, ranges, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._qvp_find_neighbour_gates_dense(
            azimuths, ranges, 0.0, 0.0, 10.0, 100.0
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust empty-array checks are verified in installed-wheel mode",
)
def test_real_rust_qvp_find_neighbour_gates_allows_empty_arrays():
    import pyart._rust as rust

    rays, ranges = rust._qvp_find_neighbour_gates_dense(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        0.0,
        0.0,
        10.0,
        100.0,
    )

    assert rays.dtype == np.int64
    assert ranges.dtype == np.int64
    assert rays.shape == (0,)
    assert ranges.shape == (0,)
