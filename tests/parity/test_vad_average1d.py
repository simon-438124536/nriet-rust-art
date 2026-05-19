import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import vad  # noqa: E402


def _fallback_average(x, y, x_new, window, weight, monkeypatch, fill_value=99999.0):
    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
    return vad._Average1D(x, y, window, weight, fill_value=fill_value)(x_new)


def _fallback_interval_mean(data, current_z, wanted_z, monkeypatch):
    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
    return vad._interval_mean(data, current_z, wanted_z)


def _fallback_vad_calculation_b(
    velocities, azimuths, elevation, valid_ray_min, monkeypatch
):
    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
    return vad._vad_calculation_b(velocities, azimuths, elevation, valid_ray_min)


def test_average1d_equal_python_fallback_preserves_left_closed_right_open_window(
    monkeypatch,
):
    x = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([30.0, 10.0, 20.0], dtype=np.float64)
    x_new = np.array([2.0, 2.0, 10.0], dtype=np.float64)

    actual = _fallback_average(x, y, x_new, 1.0, "equal", monkeypatch)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.array([15.0, 15.0, 99999.0]))


@pytest.mark.parametrize("window", [0.0, -1.0])
def test_average1d_equal_python_fallback_preserves_empty_window_rules(
    monkeypatch, window
):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    actual = _fallback_average(
        x, y, np.array([2.0], dtype=np.float64), window, "equal", monkeypatch
    )

    np.testing.assert_array_equal(actual, np.array([99999.0], dtype=np.float64))


def test_average1d_idw_python_fallback_preserves_zero_distance_warning(
    monkeypatch,
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("zero-distance idw must stay on Python fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)
    interp = vad._Average1D(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([10.0, 20.0], dtype=np.float64),
        1.0,
        "idw",
    )

    with pytest.warns(RuntimeWarning):
        actual = interp(np.array([1.0], dtype=np.float64))

    assert actual.shape == (1,)
    assert np.isnan(actual[0])


@pytest.mark.parametrize(
    ("x", "y", "x_new", "window", "weight"),
    [
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([1.5], dtype=np.float32),
            1.0,
            "equal",
        ),
        (
            np.array([1.0, np.nan], dtype=np.float64),
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([1.5], dtype=np.float64),
            1.0,
            "equal",
        ),
        (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([10.0, 20.0], dtype=np.float64),
            [1.5],
            1.0,
            "equal",
        ),
        (
            np.ma.array([1.0, 2.0], dtype=np.float64),
            np.ma.array([10.0, 20.0], dtype=np.float64),
            np.array([1.5], dtype=np.float64),
            1.0,
            "equal",
        ),
        (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([1.5], dtype=np.float64),
            1.0,
            lambda _dist: None,
        ),
    ],
)
def test_average1d_keeps_python_path_for_unsupported_inputs(
    monkeypatch, x, y, x_new, window, weight
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported average1d input should use fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)
    interp = vad._Average1D(x, y, window, weight)

    actual = interp(x_new)

    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
    expected = vad._Average1D(x, y, window, weight)(x_new)
    np.testing.assert_array_equal(actual, expected)


def test_average1d_keeps_python_path_for_nonfinite_fill_value(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("non-finite fill value should use fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)
    interp = vad._Average1D(
        np.array([1.0], dtype=np.float64),
        np.array([10.0], dtype=np.float64),
        0.0,
        "equal",
        fill_value=np.inf,
    )

    np.testing.assert_array_equal(
        interp(np.array([1.0], dtype=np.float64)),
        np.array([np.inf], dtype=np.float64),
    )


def test_average1d_preserves_python_length_mismatch_constructor_behavior(
    monkeypatch,
):
    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)

    with pytest.raises(IndexError):
        vad._Average1D(
            np.array([2.0, 1.0], dtype=np.float64),
            np.array([20.0], dtype=np.float64),
            1.0,
            "equal",
        )

    interp = vad._Average1D(
        np.array([2.0, 1.0], dtype=np.float64),
        np.array([20.0, 10.0, 999.0], dtype=np.float64),
        1.0,
        "equal",
    )
    np.testing.assert_array_equal(
        interp(np.array([1.5], dtype=np.float64)),
        np.array([15.0], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "x_new",
    [
        np.array(1.5, dtype=np.float64),
        np.float64(1.5),
    ],
)
def test_average1d_preserves_python_scalar_xnew_exception(monkeypatch, x_new):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("scalar x_new should use Python fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)
    interp = vad._Average1D(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([10.0, 20.0], dtype=np.float64),
        1.0,
        "equal",
    )

    with pytest.raises(TypeError):
        interp(x_new)


def test_average1d_equal_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(x_sorted, y_sorted, x_new, window, fill_value):
        calls.append(
            (
                x_sorted.dtype,
                y_sorted.dtype,
                x_new.dtype,
                x_sorted.tolist(),
                y_sorted.tolist(),
                window,
                fill_value,
            )
        )
        return np.array([42.0, 43.0], dtype=np.float64)

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_average1d_equal" else None,
    )
    interp = vad._Average1D(
        np.array([2.0, 1.0], dtype=np.float64),
        np.array([20.0, 10.0], dtype=np.float64),
        1.0,
        "equal",
    )

    actual = interp(np.array([1.25, 1.75], dtype=np.float64))

    np.testing.assert_array_equal(actual, np.array([42.0, 43.0]))
    assert calls == [
        (
            np.float64,
            np.float64,
            np.float64,
            [1.0, 2.0],
            [10.0, 20.0],
            1.0,
            99999.0,
        )
    ]


def test_average1d_idw_dispatches_to_private_rust_kernel_without_exact_hits(
    monkeypatch,
):
    calls = []

    def rust_kernel(x_sorted, y_sorted, x_new, window, fill_value):
        calls.append((x_sorted.tolist(), y_sorted.tolist(), x_new.tolist()))
        return np.array([12.0], dtype=np.float64)

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_average1d_idw" else None,
    )
    interp = vad._Average1D(
        np.array([1.0, 4.0], dtype=np.float64),
        np.array([10.0, 20.0], dtype=np.float64),
        3.0,
        "idw",
    )

    actual = interp(np.array([2.0], dtype=np.float64))

    np.testing.assert_array_equal(actual, np.array([12.0]))
    assert calls == [([1.0, 4.0], [10.0, 20.0], [2.0])]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_average1d_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    x = np.array([3.0, 1.0, 4.0], dtype=np.float64)
    y = np.array([30.0, 10.0, 50.0], dtype=np.float64)
    x_new = np.array([2.0, 2.5, 20.0], dtype=np.float64)

    expected_equal = _fallback_average(x, y, x_new, 2.0, "equal", monkeypatch)
    expected_idw = _fallback_average(x, y, x_new, 2.0, "idw", monkeypatch)
    monkeypatch.setattr(vad, "_rust_kernel", lambda name: getattr(rust, name, None))

    np.testing.assert_array_equal(
        vad._Average1D(x, y, 2.0, "equal")(x_new), expected_equal
    )
    np.testing.assert_array_equal(vad._Average1D(x, y, 2.0, "idw")(x_new), expected_idw)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_average1d_rejects_mismatched_lengths_direct_call():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="same length"):
        rust._average1d_equal(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
            np.array([1.5], dtype=np.float64),
            1.0,
            99999.0,
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust warning behavior is verified in installed-wheel mode",
)
def test_real_rust_average1d_idw_zero_distance_warns_and_returns_nan():
    import pyart._rust as rust

    with pytest.warns(RuntimeWarning) as warning_records:
        actual = rust._average1d_idw(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1.0,
            99999.0,
        )

    messages = [str(record.message) for record in warning_records]
    assert any("divide by zero" in message for message in messages)
    assert any("invalid value" in message for message in messages)
    assert actual.shape == (1,)
    assert np.isnan(actual[0])


def test_interval_mean_python_fallback_preserves_nearest_index_slice_semantics(
    monkeypatch,
):
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    current_z = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    wanted_z = np.array([1.5, 2.5], dtype=np.float64)

    actual = _fallback_interval_mean(data, current_z, wanted_z, monkeypatch)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.array([20.0, 30.0], dtype=np.float64))


def test_interval_mean_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(data, current_z, wanted_z):
        calls.append((data.dtype, current_z.dtype, wanted_z.dtype, wanted_z.shape))
        return np.array([77.0, 88.0], dtype=np.float64)

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_vad_interval_mean" else None,
    )

    actual = vad._interval_mean(
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1.5, 2.5], dtype=np.float64),
    )

    np.testing.assert_array_equal(actual, np.array([77.0, 88.0], dtype=np.float64))
    assert calls == [(np.float64, np.float64, np.float64, (2,))]


def test_vad_calculation_b_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(velocities, sin_az, cos_az, elevation_scale):
        calls.append(
            (
                velocities.dtype,
                velocities.shape,
                sin_az.shape,
                cos_az.shape,
                elevation_scale,
            )
        )
        return (
            np.array([11.0, 12.0], dtype=np.float64),
            np.array([21.0, 22.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_vad_calculation_b_dense" else None,
    )

    velocities = np.ma.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64
    )
    azimuths = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)

    u_wind, v_wind = vad._vad_calculation_b(velocities, azimuths, 0.0, 4)

    np.testing.assert_array_equal(u_wind, np.array([11.0, 12.0]))
    np.testing.assert_array_equal(v_wind, np.array([21.0, 22.0]))
    assert calls == [(np.float64, (4, 2), (4,), (4,), 1.0)]


def test_vad_browning_preserves_pyart_window_precedence(monkeypatch):
    recorded_windows = []

    class DummyRadar:
        fields = {
            "velocity": {
                "data": np.ma.array(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=np.float64,
                )
            }
        }
        azimuth = {"data": np.array([0.0, 180.0], dtype=np.float64)}
        fixed_angle = {"data": np.array([0.0], dtype=np.float64)}
        gate_z = {"data": np.array([[100.0, 200.0]], dtype=np.float64)}

    class RecordingAverage1D:
        def __init__(self, _x, _y, window, _weight, fill_value=99999.0):
            recorded_windows.append((window, fill_value))

        def __call__(self, x_new, window=None):
            return np.zeros_like(x_new, dtype=np.float64)

    monkeypatch.setattr(
        vad,
        "_vad_calculation_b",
        lambda *_args: (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(vad, "_Average1D", RecordingAverage1D)
    monkeypatch.setattr(
        vad.HorizontalWindProfile,
        "from_u_and_v",
        staticmethod(lambda z, u, v: (z, u, v)),
    )

    z_want = np.array([10.0, 30.0], dtype=np.float64)
    vad.vad_browning(DummyRadar(), "velocity", z_want=z_want, window=4)

    assert recorded_windows == [(27.5, 99999.0), (27.5, 99999.0)]


@pytest.mark.parametrize(
    ("velocities", "azimuths", "elevation", "valid_ray_min"),
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            2,
        ),
        (
            np.ma.array(
                [[1.0, 2.0], [3.0, 4.0]],
                mask=[[False, True], [False, False]],
                dtype=np.float64,
            ),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            2,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            2,
        ),
        (
            np.ma.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            2,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float32),
            0.0,
            2,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            0.0,
            2,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            np.inf,
            2,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            3,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            1.5,
        ),
        (
            np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            -1,
        ),
        (
            np.ma.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
            ),
            np.array([45.0, 45.0, 45.0], dtype=np.float64),
            0.0,
            3,
        ),
        (
            np.ma.array(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64).T
            ),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
            2,
        ),
    ],
)
def test_vad_calculation_b_keeps_python_path_for_unsupported_inputs(
    monkeypatch, velocities, azimuths, elevation, valid_ray_min
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported vad calculation should use fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            actual = vad._vad_calculation_b(
                velocities, azimuths, elevation, valid_ray_min
            )
        except Exception as actual_error:
            monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                vad._vad_calculation_b(velocities, azimuths, elevation, valid_ray_min)
        else:
            expected = _fallback_vad_calculation_b(
                velocities, azimuths, elevation, valid_ray_min, monkeypatch
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    ("data", "current_z", "wanted_z"),
    [
        (
            np.array([10.0, 20.0, 30.0], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([0.5, 1.5], dtype=np.float32),
        ),
        (
            np.array([10, 20, 30], dtype=np.int32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.array([0.5, 1.5], dtype=np.float64),
        ),
        (
            np.array([10.0, 20.0, 30.0], dtype=np.float64),
            np.array([0.0, np.nan, 2.0], dtype=np.float64),
            np.array([0.5, 1.5], dtype=np.float64),
        ),
        (
            np.array([10.0, 20.0, 30.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.array([1.5, 2.5], dtype=np.float64),
        ),
        (
            np.ma.array([10.0, 20.0, 30.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.array([0.5, 1.5], dtype=np.float64),
        ),
    ],
)
def test_interval_mean_keeps_python_path_for_unsupported_inputs(
    monkeypatch, data, current_z, wanted_z
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported interval-mean input should use fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            actual = vad._interval_mean(data, current_z, wanted_z)
        except Exception as actual_error:
            monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                vad._interval_mean(data, current_z, wanted_z)
        else:
            expected = _fallback_interval_mean(data, current_z, wanted_z, monkeypatch)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("data", "current_z", "wanted_z", "error_type"),
    [
        (
            np.array([10.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([], dtype=np.float64),
            IndexError,
        ),
        (
            np.array([10.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            IndexError,
        ),
        (
            np.array([10.0], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            IndexError,
        ),
    ],
)
def test_interval_mean_preserves_python_exception_edges(
    monkeypatch, data, current_z, wanted_z, error_type
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("exception edges should use Python fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)

    with pytest.raises(error_type):
        vad._interval_mean(data, current_z, wanted_z)


def test_interval_mean_empty_slice_stays_on_python_warning_path(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("empty slices should use Python fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)
    data = np.array([10.0, 20.0], dtype=np.float64)
    current_z = np.array([0.0, 10.0], dtype=np.float64)
    wanted_z = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.warns(RuntimeWarning):
        actual = vad._interval_mean(data, current_z, wanted_z)

    np.testing.assert_array_equal(actual, np.array([np.nan, np.nan]))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_interval_mean_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    current_z = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    wanted_z = np.array([1.5, 2.5], dtype=np.float64)

    expected = _fallback_interval_mean(data, current_z, wanted_z, monkeypatch)
    monkeypatch.setattr(vad, "_rust_kernel", lambda name: getattr(rust, name, None))

    np.testing.assert_array_equal(vad._interval_mean(data, current_z, wanted_z), expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_interval_mean_rejects_mismatched_lengths_direct_call():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="same length"):
        rust._vad_interval_mean(
            np.array([10.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust warning behavior is verified in installed-wheel mode",
)
def test_real_rust_interval_mean_empty_slice_warns_and_returns_nan():
    import pyart._rust as rust

    with pytest.warns(RuntimeWarning):
        actual = rust._vad_interval_mean(
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([0.0, 10.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )

    np.testing.assert_array_equal(actual, np.array([np.nan, np.nan]))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_vad_calculation_b_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    velocities = np.ma.array(
        [
            [2.0, 5.0, 11.0],
            [7.0, 13.0, 17.0],
            [19.0, 23.0, 29.0],
            [31.0, 37.0, 41.0],
        ],
        dtype=np.float64,
    )
    azimuths = np.array([0.0, 82.5, 181.0, 274.0], dtype=np.float64)

    expected = _fallback_vad_calculation_b(velocities, azimuths, 3.5, 4, monkeypatch)
    monkeypatch.setattr(vad, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = vad._vad_calculation_b(velocities, azimuths, 3.5, 4)

    assert actual[0].dtype == expected[0].dtype == np.float64
    assert actual[1].dtype == expected[1].dtype == np.float64
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_vad_calculation_b_rejects_shape_mismatch_direct_call():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="match the number of velocity rays"):
        rust._vad_calculation_b_dense(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1.0,
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_vad_calculation_b_valid_ray_min_safe_range_matches_python(
    monkeypatch,
):
    import pyart._rust as rust

    velocities = np.ma.array(
        [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]],
        dtype=np.float64,
    )
    azimuths = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    sin_az = np.sin(np.deg2rad(azimuths))
    cos_az = np.cos(np.deg2rad(azimuths))

    for valid_ray_min in (0, 4):
        expected = _fallback_vad_calculation_b(
            velocities, azimuths, 0.0, valid_ray_min, monkeypatch
        )
        monkeypatch.setattr(vad, "_rust_kernel", lambda name: getattr(rust, name, None))
        actual = vad._vad_calculation_b(velocities, azimuths, 0.0, valid_ray_min)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_vad_calculation_b_rejects_degenerate_geometry_direct_call():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="non-singular"):
        rust._vad_calculation_b_dense(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            1.0,
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("velocities", "sin_az", "cos_az", "elevation_scale", "match"),
    [
        (
            np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "velocities must be finite",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "sin_az and cos_az must be finite",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            np.inf,
            "elevation_scale must be finite",
        ),
    ],
)
def test_real_rust_vad_calculation_b_rejects_nonfinite_direct_call(
    velocities, sin_az, cos_az, elevation_scale, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._vad_calculation_b_dense(velocities, sin_az, cos_az, elevation_scale)
