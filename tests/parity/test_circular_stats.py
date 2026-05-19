import importlib.util
import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.util import circular_stats  # noqa: E402


def _reference_mean_of_two_angles(angles1, angles2):
    x = (np.cos(angles1) + np.cos(angles2)) / 2.0
    y = (np.sin(angles1) + np.sin(angles2)) / 2.0
    return np.arctan2(y, x)


def _reference_angular_mean(angles):
    x = np.cos(angles)
    y = np.sin(angles)
    return np.arctan2(np.mean(y), np.mean(x))


def _reference_angular_std(angles):
    x = np.cos(angles)
    y = np.sin(angles)
    norm = np.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2)
    return np.sqrt(-2 * np.log(norm))


def _reference_interval_mean(dist, interval_min, interval_max):
    half_width = (interval_max - interval_min) / 2.0
    center = interval_min + half_width
    angles = (dist - center) / half_width * np.pi
    return (_reference_angular_mean(angles) * half_width / np.pi) + center


def _reference_interval_std(dist, interval_min, interval_max):
    half_width = (interval_max - interval_min) / 2.0
    center = interval_min + half_width
    angles = (dist - center) / half_width * np.pi
    return _reference_angular_std(angles) * half_width / np.pi


def _reference_compute_directional_stats(field, avg_type="mean", nvalid_min=1, axis=0):
    if avg_type == "mean":
        values = np.ma.mean(field, axis=axis)
    else:
        values = np.ma.median(field, axis=axis)

    valid = np.logical_not(np.ma.getmaskarray(field))
    nvalid = np.sum(valid, axis=axis, dtype=int)
    values[nvalid < nvalid_min] = np.ma.masked
    return values, nvalid


def _assert_directional_stats_equal(actual, expected):
    actual_values, actual_nvalid = actual
    expected_values, expected_nvalid = expected

    assert isinstance(actual_values, np.ma.MaskedArray)
    assert isinstance(expected_values, np.ma.MaskedArray)
    np.testing.assert_array_equal(actual_values.mask, expected_values.mask)
    np.testing.assert_allclose(
        actual_values.data,
        expected_values.data,
        rtol=1.0e-14,
        atol=1.0e-14,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual_nvalid, expected_nvalid)
    assert actual_nvalid.dtype == expected_nvalid.dtype


def test_circular_stats_python_fallback_matches_oracle_formulas(monkeypatch):
    monkeypatch.setattr(circular_stats, "_rust_kernel", lambda _name: None)
    angles1 = np.deg2rad(np.array([[350.0], [20.0]], dtype=np.float64))
    angles2 = np.deg2rad(np.array([[10.0, 90.0, 180.0]], dtype=np.float64))
    dist = np.array([[1.0, 2.5], [3.5, 5.0]], dtype=np.float64)

    np.testing.assert_allclose(
        circular_stats.mean_of_two_angles(angles1, angles2),
        _reference_mean_of_two_angles(angles1, angles2),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        circular_stats.angular_mean(angles1),
        _reference_angular_mean(angles1),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        circular_stats.angular_std(angles2),
        _reference_angular_std(angles2),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        circular_stats.interval_mean(dist, 0.0, 6.0),
        _reference_interval_mean(dist, 0.0, 6.0),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        circular_stats.interval_std(dist, 0.0, 6.0),
        _reference_interval_std(dist, 0.0, 6.0),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_compute_directional_stats_python_fallback_matches_oracle(monkeypatch):
    monkeypatch.setattr(circular_stats, "_rust_kernel", lambda _name: None)
    field = np.ma.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        mask=[[False, True, False], [False, False, False]],
        dtype=np.float64,
    )

    _assert_directional_stats_equal(
        circular_stats.compute_directional_stats(field, nvalid_min=2, axis=0),
        _reference_compute_directional_stats(field, nvalid_min=2, axis=0),
    )
    _assert_directional_stats_equal(
        circular_stats.compute_directional_stats(field, avg_type="median", axis=1),
        _reference_compute_directional_stats(field, avg_type="median", axis=1),
    )


def test_circular_stats_dispatch_to_private_rust_kernels_for_float64_arrays(
    monkeypatch,
):
    calls = []

    def fake_rust_kernel(name):
        def mean_kernel(angles1, angles2):
            calls.append((name, angles1.dtype, angles1.shape, angles2.shape))
            return np.full(np.broadcast_shapes(angles1.shape, angles2.shape), 1.25)

        def scalar_kernel(values, *bounds):
            calls.append((name, values.dtype, values.shape, bounds))
            return 2.5

        return {
            "_mean_of_two_angles": mean_kernel,
            "_angular_mean": scalar_kernel,
            "_angular_std": scalar_kernel,
            "_interval_mean": scalar_kernel,
            "_interval_std": scalar_kernel,
        }.get(name)

    monkeypatch.setattr(circular_stats, "_rust_kernel", fake_rust_kernel)
    angles1 = np.array([[0.0], [1.0]], dtype=np.float64)
    angles2 = np.array([[2.0, 3.0]], dtype=np.float64)
    dist = np.array([0.5, 1.5, 2.5], dtype=np.float64)

    np.testing.assert_array_equal(
        circular_stats.mean_of_two_angles(angles1, angles2),
        np.full((2, 2), 1.25),
    )
    assert circular_stats.angular_mean(angles1) == 2.5
    assert circular_stats.angular_std(angles2) == 2.5
    assert circular_stats.interval_mean(dist, 0.0, 3.0) == 2.5
    assert circular_stats.interval_std(dist, 0.0, 3.0) == 2.5

    assert calls == [
        ("_mean_of_two_angles", np.float64, (2, 1), (1, 2)),
        ("_angular_mean", np.float64, (2, 1), ()),
        ("_angular_std", np.float64, (1, 2), ()),
        ("_interval_mean", np.float64, (3,), (0.0, 3.0)),
        ("_interval_std", np.float64, (3,), (0.0, 3.0)),
    ]


def test_compute_directional_stats_dispatches_dense_mean_to_private_rust_kernel(
    monkeypatch,
):
    calls = []

    def fake_rust_kernel(name):
        if name != "_compute_directional_stats_mean_dense_f64":
            return None

        def kernel(field, axis):
            calls.append((name, field.dtype, field.shape, axis))
            return np.array([10.0, 20.0], dtype=np.float64), np.array([3, 1], dtype=int)

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fake_rust_kernel)
    field = np.arange(6.0, dtype=np.float64).reshape(2, 3)

    values, nvalid = circular_stats.compute_directional_stats(
        field, avg_type="mean", nvalid_min=2, axis=1
    )

    assert calls == [
        ("_compute_directional_stats_mean_dense_f64", np.float64, (2, 3), 1)
    ]
    np.testing.assert_array_equal(values.data, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(values.mask, np.array([False, True]))
    np.testing.assert_array_equal(nvalid, np.array([3, 1], dtype=int))


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        (
            np.ma.array(
                [[1.0, 2.0], [3.0, 4.0]],
                mask=[[False, True], [False, False]],
                dtype=np.float64,
            ),
            {"avg_type": "mean", "axis": 0},
        ),
        (np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]])), {"axis": 0}),
        (np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64), {"axis": 0}),
        (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), {"avg_type": "median"}),
        (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), {"axis": -1}),
    ],
)
def test_compute_directional_stats_keeps_python_path_for_unsupported_inputs(
    monkeypatch, field, kwargs
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported inputs should use the Python fallback")

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        actual = circular_stats.compute_directional_stats(field, **kwargs)
        expected = _reference_compute_directional_stats(field, **kwargs)

    _assert_directional_stats_equal(actual, expected)


@pytest.mark.parametrize("axis", [True, False, np.bool_(True), np.bool_(False)])
def test_compute_directional_stats_bool_axis_preserves_python_exception(
    monkeypatch, axis
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("bool axis should use the Python fallback")

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fail_if_called)
    field = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(TypeError) as actual_error:
        circular_stats.compute_directional_stats(field, axis=axis)
    with pytest.raises(TypeError) as expected_error:
        _reference_compute_directional_stats(field, axis=axis)

    assert actual_error.value.args == expected_error.value.args


@pytest.mark.parametrize(
    ("angles1", "angles2"),
    [
        (
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([2.0, 3.0], dtype=np.float32),
        ),
        (
            np.ma.array([0.0, 1.0], mask=[False, True], dtype=np.float64),
            np.array([2.0, 3.0], dtype=np.float64),
        ),
        (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        ),
        (
            np.array(0.0, dtype=np.float64),
            np.array(1.0, dtype=np.float64),
        ),
    ],
)
def test_mean_of_two_angles_keeps_numpy_path_for_unsupported_inputs(
    monkeypatch, angles1, angles2
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported inputs should use the NumPy fallback")

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fail_if_called)

    actual = circular_stats.mean_of_two_angles(angles1, angles2)
    expected = _reference_mean_of_two_angles(angles1, angles2)

    if np.ma.isMaskedArray(actual):
        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_allclose(actual.data, expected.data, rtol=1.0e-14, atol=1.0e-14)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


def test_mean_of_two_angles_preserves_numpy_broadcast_error(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("broadcast errors should come from NumPy fallback")

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fail_if_called)

    with pytest.raises(ValueError, match="operands could not be broadcast"):
        circular_stats.mean_of_two_angles(
            np.ones((2,), dtype=np.float64),
            np.ones((3,), dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("function_name", "args"),
    [
        ("angular_mean", (np.array([], dtype=np.float64),)),
        ("angular_std", (np.array([], dtype=np.float64),)),
        ("angular_mean", (np.array(1.0, dtype=np.float64),)),
        ("angular_std", (np.array(1.0, dtype=np.float64),)),
        ("angular_mean", (np.array([1.0, 2.0], dtype=np.float32),)),
        ("angular_std", (np.array([1.0, 2.0], dtype=np.float32),)),
    ],
)
def test_angular_stats_keep_numpy_path_for_unsupported_inputs(
    monkeypatch, function_name, args
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported inputs should use the NumPy fallback")

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fail_if_called)
    function = getattr(circular_stats, function_name)
    reference = (
        _reference_angular_mean if function_name == "angular_mean" else _reference_angular_std
    )

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = function(*args)
        expected = reference(*args)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.parametrize("function_name", ["interval_mean", "interval_std"])
def test_interval_stats_keep_numpy_path_for_zero_width_interval(
    monkeypatch, function_name
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("zero-width intervals should use the NumPy fallback")

        return kernel

    monkeypatch.setattr(circular_stats, "_rust_kernel", fail_if_called)
    function = getattr(circular_stats, function_name)

    with np.errstate(all="ignore"):
        actual = function(np.array([1.0, 2.0], dtype=np.float64), 1.0, 1.0)

    assert isinstance(actual, np.floating)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_circular_stats_match_numpy_fallback(monkeypatch):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    angles1 = np.deg2rad(np.array([[350.0], [15.0]], dtype=np.float64))
    angles2 = np.deg2rad(np.array([[10.0, 95.0, 180.0]], dtype=np.float64))
    dist = np.array([[0.5, 1.5], [3.0, 4.5]], dtype=np.float64)
    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    monkeypatch.setattr(circular_stats, "_rust_kernel", lambda _name: None)
    expected_mean = circular_stats.mean_of_two_angles(angles1, angles2)
    expected_angular_mean = circular_stats.angular_mean(angles1)
    expected_angular_std = circular_stats.angular_std(angles2)
    expected_interval_mean = circular_stats.interval_mean(dist, 0.0, 6.0)
    expected_interval_std = circular_stats.interval_std(dist, 0.0, 6.0)
    expected_directional_axis0 = circular_stats.compute_directional_stats(field, axis=0)
    expected_directional_axis1 = circular_stats.compute_directional_stats(
        field, nvalid_min=4, axis=1
    )

    import pyart._rust as rust

    monkeypatch.setattr(
        circular_stats,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual_mean = circular_stats.mean_of_two_angles(angles1, angles2)
    actual_angular_mean = circular_stats.angular_mean(angles1)
    actual_angular_std = circular_stats.angular_std(angles2)
    actual_interval_mean = circular_stats.interval_mean(dist, 0.0, 6.0)
    actual_interval_std = circular_stats.interval_std(dist, 0.0, 6.0)
    actual_directional_axis0 = circular_stats.compute_directional_stats(field, axis=0)
    actual_directional_axis1 = circular_stats.compute_directional_stats(
        field, nvalid_min=4, axis=1
    )

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        actual_angular_mean, expected_angular_mean, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        actual_angular_std, expected_angular_std, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        actual_interval_mean, expected_interval_mean, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        actual_interval_std, expected_interval_std, rtol=1.0e-14, atol=1.0e-14
    )
    _assert_directional_stats_equal(actual_directional_axis0, expected_directional_axis0)
    _assert_directional_stats_equal(actual_directional_axis1, expected_directional_axis1)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust is verified in installed-wheel mode",
)
def test_real_rust_compute_directional_stats_input_validation():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    values, nvalid = rust._compute_directional_stats_mean_dense_f64(field, 1)
    np.testing.assert_allclose(values, np.array([2.0, 5.0], dtype=np.float64))
    np.testing.assert_array_equal(nvalid, np.array([3, 3], dtype=int))

    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        rust._compute_directional_stats_mean_dense_f64(field, 2)
    with pytest.raises(ValueError, match="non-boolean integer"):
        rust._compute_directional_stats_mean_dense_f64(field, True)
    with pytest.raises(ValueError, match="non-boolean integer"):
        rust._compute_directional_stats_mean_dense_f64(field, np.bool_(True))
    with pytest.raises(ValueError, match="non-boolean integer"):
        rust._compute_directional_stats_mean_dense_f64(field, "1")
    with pytest.raises(ValueError, match="field must be a 2D"):
        rust._compute_directional_stats_mean_dense_f64(
            np.array([1.0, 2.0], dtype=np.float64), 0
        )
    with pytest.raises(ValueError, match="float64"):
        rust._compute_directional_stats_mean_dense_f64(
            np.array([[1, 2], [3, 4]], dtype=np.int64), 0
        )
    with pytest.raises(ValueError, match="float64"):
        rust._compute_directional_stats_mean_dense_f64(
            np.array([[object(), object()]], dtype=object), 0
        )
    with pytest.raises(ValueError, match="field must be C-contiguous"):
        rust._compute_directional_stats_mean_dense_f64(np.asfortranarray(field), 0)
    with pytest.raises(ValueError, match="field must contain only finite values"):
        rust._compute_directional_stats_mean_dense_f64(
            np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float64), 0
        )
    with pytest.raises(ValueError, match="field must be mask-free"):
        rust._compute_directional_stats_mean_dense_f64(
            np.ma.array(field, mask=[[False, True, False], [False, False, False]]),
            0,
        )
