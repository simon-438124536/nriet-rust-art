import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import srv  # noqa: E402


class _FakeRadar:
    def __init__(self, data, ranges, azimuths, sweep_numbers):
        self.fields = {"velocity": {"data": data}}
        self.sweep_number = {"data": np.array(sweep_numbers)}
        self._ranges = ranges
        self._azimuths = azimuths

    def get_start_end(self, sweep):
        return self._ranges[int(sweep)]

    def get_azimuth(self, sweep=0):
        return self._azimuths[int(sweep)]


def _fallback_srv(radar, monkeypatch, **kwargs):
    monkeypatch.setattr(srv, "_rust_kernel", lambda _name: None)
    return srv.storm_relative_velocity(radar, **kwargs)


def test_storm_relative_velocity_python_fallback_preserves_single_sweep_end_exclusion(
    monkeypatch,
):
    data = np.arange(8, dtype=np.float64).reshape(4, 2)
    radar = _FakeRadar(data, {0: (1, 3)}, {0: np.array([0.0, 90.0])}, [0])

    actual = _fallback_srv(radar, monkeypatch, direction=0.0, speed=10.0)

    expected = data.copy()
    expected[1] = data[1] - 10.0
    expected[2] = data[2] - 0.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


def test_storm_relative_velocity_preserves_uv_branch_unboundlocalerror(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("u/v branch should fail before Rust dispatch")

        return kernel

    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    radar = _FakeRadar(data, {0: (0, 1)}, {0: np.array([0.0, 90.0])}, [0])
    monkeypatch.setattr(srv, "_rust_kernel", fail_if_called)

    with pytest.raises(UnboundLocalError):
        srv.storm_relative_velocity(radar, u=1.0, v=1.0)


@pytest.mark.parametrize("kwargs", [{"direction": "ne", "speed": 10.0}, {"u": 1.0}])
def test_storm_relative_velocity_preserves_argument_errors(monkeypatch, kwargs):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("argument errors should occur before Rust dispatch")

        return kernel

    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    radar = _FakeRadar(data, {0: (0, 1)}, {0: np.array([0.0, 90.0])}, [0])
    monkeypatch.setattr(srv, "_rust_kernel", fail_if_called)

    with pytest.raises(ValueError):
        srv.storm_relative_velocity(radar, **kwargs)


def test_storm_relative_velocity_does_not_mutate_original_velocity(monkeypatch):
    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    before = data.copy()
    radar = _FakeRadar(data, {0: (0, 1)}, {0: np.array([0.0, 90.0])}, [0])

    actual = _fallback_srv(radar, monkeypatch, direction=0.0, speed=10.0)

    assert actual is not radar.fields["velocity"]["data"]
    np.testing.assert_array_equal(radar.fields["velocity"]["data"], before)


def test_storm_relative_velocity_dispatches_to_private_rust_for_sweeps(monkeypatch):
    calls = []

    def rust_kernel(sr_data, velocity_data, angle_array, speed, alpha, start, stop):
        calls.append((angle_array.tolist(), speed, alpha, start, stop))
        sr_data[start:stop] = 99.0

    monkeypatch.setattr(
        srv,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_storm_relative_velocity_inplace" else None,
    )
    data = np.arange(8, dtype=np.float64).reshape(4, 2)
    radar = _FakeRadar(
        data,
        {0: (0, 1), 1: (2, 3)},
        {0: np.array([0.0, 90.0]), 1: np.array([180.0, 270.0])},
        [0, 1],
    )

    actual = srv.storm_relative_velocity(radar, direction=0.0, speed=10.0)

    np.testing.assert_array_equal(actual, np.full((4, 2), 99.0))
    assert calls == [
        ([0.0, 90.0], 10.0, 0.0, 0, 2),
        ([180.0, 270.0], 10.0, 0.0, 2, 4),
    ]


@pytest.mark.parametrize(
    "data",
    [
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.arange(8, dtype=np.int32).reshape(4, 2),
        np.ma.array(np.arange(8, dtype=np.float64).reshape(4, 2), mask=False),
    ],
)
def test_storm_relative_velocity_keeps_python_path_for_unsupported_data(
    monkeypatch, data
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported srv data should use Python fallback")

        return kernel

    radar = _FakeRadar(
        data,
        {0: (0, 1), 1: (2, 3)},
        {0: np.array([0.0, 90.0]), 1: np.array([180.0, 270.0])},
        [0, 1],
    )
    monkeypatch.setattr(srv, "_rust_kernel", fail_if_called)

    actual = srv.storm_relative_velocity(radar, direction=0.0, speed=10.0)

    monkeypatch.setattr(srv, "_rust_kernel", lambda _name: None)
    expected = srv.storm_relative_velocity(radar, direction=0.0, speed=10.0)
    np.testing.assert_array_equal(actual, expected)


def test_storm_relative_velocity_keeps_python_path_for_nonfinite_angles(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("nonfinite angle data should use Python fallback")

        return kernel

    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    radar = _FakeRadar(data, {0: (0, 1)}, {0: np.array([np.inf])}, [0])
    monkeypatch.setattr(srv, "_rust_kernel", fail_if_called)

    with pytest.warns(RuntimeWarning):
        actual = srv.storm_relative_velocity(radar, direction=0.0, speed=10.0)

    assert np.isnan(actual[0]).all()


def test_storm_relative_velocity_keeps_python_path_for_string_speed(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("string speed should use Python fallback")

        return kernel

    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    radar = _FakeRadar(data, {0: (0, 1)}, {0: np.array([0.0, 90.0])}, [0])
    monkeypatch.setattr(srv, "_rust_kernel", fail_if_called)

    with pytest.raises(TypeError):
        srv.storm_relative_velocity(radar, direction=0.0, speed="10")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_storm_relative_velocity_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    data = np.arange(12, dtype=np.float64).reshape(4, 3)
    radar = _FakeRadar(
        data,
        {0: (0, 1), 1: (2, 3)},
        {0: np.array([0.0, 90.0]), 1: np.array([180.0, 270.0])},
        [0, 1],
    )
    expected = _fallback_srv(radar, monkeypatch, direction=0.0, speed=10.0)
    monkeypatch.setattr(srv, "_rust_kernel", lambda name: getattr(rust, name, None))

    actual = srv.storm_relative_velocity(radar, direction=0.0, speed=10.0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_storm_relative_velocity_rejects_bad_ray_range():
    import pyart._rust as rust

    data = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="ray range"):
        rust._storm_relative_velocity_inplace(
            data.copy(),
            data,
            np.array([0.0], dtype=np.float64),
            10.0,
            0.0,
            0,
            3,
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust overlap checks are verified in installed-wheel mode",
)
def test_real_rust_storm_relative_velocity_allows_identical_input_output():
    import pyart._rust as rust

    data = np.arange(4, dtype=np.float64).reshape(2, 2)
    rust._storm_relative_velocity_inplace(
        data,
        data,
        np.array([0.0, 90.0], dtype=np.float64),
        10.0,
        0.0,
        0,
        2,
    )

    expected = np.array([[-10.0, -9.0], [2.0, 3.0]], dtype=np.float64)
    np.testing.assert_allclose(data, expected, rtol=0.0, atol=1.0e-14)
