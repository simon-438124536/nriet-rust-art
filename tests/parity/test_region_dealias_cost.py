import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import region_dealias  # noqa: E402


def _fallback_cost(nyq_vector, vels, sref, v_nyq_vel, nfeatures, monkeypatch):
    monkeypatch.setattr(region_dealias, "_rust_kernel", lambda _name: None)
    return region_dealias._cost_function(
        nyq_vector, vels, sref, v_nyq_vel, nfeatures
    )


def _fallback_gradient(nyq_vector, vels, sref, v_nyq_vel, nfeatures, monkeypatch):
    monkeypatch.setattr(region_dealias, "_rust_kernel", lambda _name: None)
    return region_dealias._gradient(nyq_vector, vels, sref, v_nyq_vel, nfeatures)


def _call_sweep_interval_splits(nyquist, interval_splits, velocities, nsweep):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = region_dealias._find_sweep_interval_splits(
            nyquist, interval_splits, velocities, nsweep
        )
    warning_info = [(warning.category, str(warning.message)) for warning in caught]
    return result, warning_info


def _fallback_sweep_interval_splits(
    nyquist, interval_splits, velocities, nsweep, monkeypatch
):
    monkeypatch.setattr(region_dealias, "_rust_kernel", lambda _name: None)
    return _call_sweep_interval_splits(nyquist, interval_splits, velocities, nsweep)


def _edge_aggregate_inputs():
    index1 = np.array([2, 1, 2, 1, 1], dtype=np.int32)
    index2 = np.array([5, 4, 5, 4, 3], dtype=np.int32)
    vel1 = np.array([2.0, 1.0, 20.0, 10.0, 100.0], dtype=np.float64)
    vel2 = np.array([3.0, 4.0, 30.0, 40.0, 400.0], dtype=np.float64)
    return index1, index2, vel1, vel2


def _run_edge_sum_with_edges(
    index1, index2, vel1, vel2, monkeypatch, rust_kernel=None
):
    def fake_fast_edge_finder(*_args):
        return (index1, index2), (vel1, vel2)

    monkeypatch.setattr(region_dealias, "_fast_edge_finder", fake_fast_edge_finder)
    monkeypatch.setattr(
        region_dealias,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_region_edge_sum_and_count" else None,
    )
    return region_dealias._edge_sum_and_count(
        np.ones((2, 3), dtype=np.int32),
        0,
        np.ones((2, 3), dtype=np.float32),
        False,
        0,
        0,
    )


def _fallback_edge_sum_and_count(index1, index2, vel1, vel2, monkeypatch):
    return _run_edge_sum_with_edges(index1, index2, vel1, vel2, monkeypatch)


def test_region_cost_and_gradient_python_fallback_preserve_oracle_formula(
    monkeypatch,
):
    nyq_vector = np.array([0.5, 1.5, 2.5, -0.5], dtype=np.float64)
    vels = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sref = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)

    cost = _fallback_cost(nyq_vector, vels, sref, 2.0, 4, monkeypatch)
    gradient = _fallback_gradient(nyq_vector, vels, sref, 2.0, 4, monkeypatch)

    assert isinstance(cost, np.float64)
    assert cost == np.float64(71.0)
    assert gradient.dtype == np.float64
    assert gradient.shape == nyq_vector.shape
    np.testing.assert_array_equal(
        gradient,
        np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float64),
    )


def test_region_cost_and_gradient_dispatch_to_private_rust_kernels(monkeypatch):
    calls = []

    def cost_kernel(nyq_vector, vels, sref, v_nyq_vel, nfeatures):
        calls.append(("cost", nyq_vector.dtype, vels.shape, v_nyq_vel, nfeatures))
        return np.float64(12.5)

    def gradient_kernel(nyq_vector, vels, sref, v_nyq_vel, nfeatures):
        calls.append(("gradient", nyq_vector.dtype, sref.shape, v_nyq_vel, nfeatures))
        return np.array([1.0, 2.0, 0.0], dtype=np.float64)

    def fake_kernel(name):
        return {
            "_region_cost_function": cost_kernel,
            "_region_gradient": gradient_kernel,
        }.get(name)

    monkeypatch.setattr(region_dealias, "_rust_kernel", fake_kernel)
    nyq_vector = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    vels = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    sref = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    assert region_dealias._cost_function(nyq_vector, vels, sref, 2.0, 2) == np.float64(
        12.5
    )
    np.testing.assert_array_equal(
        region_dealias._gradient(nyq_vector, vels, sref, 2.0, 2),
        np.array([1.0, 2.0, 0.0], dtype=np.float64),
    )

    assert calls == [
        ("cost", np.float64, (3,), 2.0, 2),
        ("gradient", np.float64, (3,), 2.0, 2),
    ]


@pytest.mark.parametrize(
    ("nyq_vector", "vels", "sref", "v_nyq_vel", "nfeatures"),
    [
        (
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            2.0,
            2,
        ),
        (
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            2,
        ),
        (
            [0.0, 1.0],
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            2,
        ),
        (
            np.ma.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            2,
        ),
        (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.nan,
            2,
        ),
        (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            0,
        ),
        (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            -1,
        ),
        (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            3,
        ),
        (
            np.array([[0.0, 1.0]], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            2,
        ),
    ],
)
def test_region_cost_and_gradient_keep_python_path_for_unsupported_inputs(
    monkeypatch, nyq_vector, vels, sref, v_nyq_vel, nfeatures
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported region-cost input should use fallback")

        return kernel

    monkeypatch.setattr(region_dealias, "_rust_kernel", fail_if_called)

    try:
        actual_cost = region_dealias._cost_function(
            nyq_vector, vels, sref, v_nyq_vel, nfeatures
        )
        actual_gradient = region_dealias._gradient(
            nyq_vector, vels, sref, v_nyq_vel, nfeatures
        )
    except Exception as actual_error:
        monkeypatch.setattr(region_dealias, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)):
            region_dealias._cost_function(nyq_vector, vels, sref, v_nyq_vel, nfeatures)
        return

    expected_cost = _fallback_cost(nyq_vector, vels, sref, v_nyq_vel, nfeatures, monkeypatch)
    expected_gradient = _fallback_gradient(
        nyq_vector, vels, sref, v_nyq_vel, nfeatures, monkeypatch
    )
    np.testing.assert_array_equal(actual_cost, expected_cost)
    np.testing.assert_array_equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize("nfeatures", [1.5, "2"])
def test_region_cost_preserves_non_integer_nfeatures_exception(
    monkeypatch, nfeatures
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("non-integer nfeatures should use Python fallback")

        return kernel

    monkeypatch.setattr(region_dealias, "_rust_kernel", fail_if_called)

    with pytest.raises(TypeError):
        region_dealias._cost_function(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            2.0,
            nfeatures,
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_region_cost_and_gradient_match_python_fallback(monkeypatch):
    import pyart._rust as rust

    nyq_vector = np.array([0.5, 1.5, 2.5, -0.5], dtype=np.float64)
    vels = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sref = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    expected_cost = _fallback_cost(nyq_vector, vels, sref, 2.0, 4, monkeypatch)
    expected_gradient = _fallback_gradient(nyq_vector, vels, sref, 2.0, 4, monkeypatch)
    monkeypatch.setattr(
        region_dealias,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )

    assert region_dealias._cost_function(nyq_vector, vels, sref, 2.0, 4) == expected_cost
    np.testing.assert_array_equal(
        region_dealias._gradient(nyq_vector, vels, sref, 2.0, 4),
        expected_gradient,
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_region_cost_rejects_out_of_range_nfeatures_direct_call():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="nfeatures"):
        rust._region_cost_function(
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            2.0,
            2,
        )


@pytest.mark.parametrize(
    ("velocities", "expected_warning"),
    [
        (np.array([-8.0, 0.0, 9.0], dtype=np.float64), False),
        (np.array([-18.0, -7.5, 3.0, 16.0], dtype=np.float64), True),
        (np.array([], dtype=np.float64), False),
    ],
)
def test_region_sweep_interval_splits_python_fallback_contract(
    monkeypatch, velocities, expected_warning
):
    result, warning_info = _fallback_sweep_interval_splits(
        10.0, 4, velocities, 3, monkeypatch
    )

    assert result.dtype == np.float64
    if expected_warning:
        assert warning_info == [
            (
                UserWarning,
                "Velocities outside of the Nyquist interval found in sweep 3.",
            )
        ]
    else:
        assert warning_info == []


def test_region_sweep_interval_splits_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(nyquist, interval_splits, velocities):
        calls.append((nyquist, interval_splits, velocities.dtype, velocities.shape))
        return -20.0, 20.0, 9, True

    monkeypatch.setattr(
        region_dealias,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_region_sweep_interval_splits" else None,
    )

    result, warning_info = _call_sweep_interval_splits(
        10, 4, np.array([-18.0, 16.0], dtype=np.float64), 7
    )

    assert calls == [(10.0, 4, np.float64, (2,))]
    np.testing.assert_array_equal(result, np.linspace(-20.0, 20.0, 9, endpoint=True))
    assert warning_info == [
        (
            UserWarning,
            "Velocities outside of the Nyquist interval found in sweep 7.",
        )
    ]


@pytest.mark.parametrize(
    ("nyquist", "interval_splits", "velocities"),
    [
        (10.0, 0, np.array([0.0, 1.0], dtype=np.float64)),
        (10.0, 2.5, np.array([0.0, 1.0], dtype=np.float64)),
        (-10.0, 4, np.array([0.0, 1.0], dtype=np.float64)),
        ("10", 4, np.array([0.0, 1.0], dtype=np.float64)),
        (10.0, 4, np.array([0.0, 1.0], dtype=np.float32)),
        (10.0, 4, np.array([0.0, np.nan], dtype=np.float64)),
        (10.0, 4, np.array([0.0, 1.0, 2.0], dtype=np.float64)[::2]),
        (10.0, 4, np.ma.array([0.0, 1.0], mask=[False, True], dtype=np.float64)),
        (10.0, 4, [0.0, 1.0]),
        (1.0e-308, 4, np.array([np.finfo(np.float64).max], dtype=np.float64)),
    ],
)
def test_region_sweep_interval_splits_keeps_python_path_for_unsupported_inputs(
    monkeypatch, nyquist, interval_splits, velocities
):
    def fail_if_called(name):
        if name != "_region_sweep_interval_splits":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported sweep interval input used Rust")

        return kernel

    monkeypatch.setattr(region_dealias, "_rust_kernel", fail_if_called)
    try:
        actual = _call_sweep_interval_splits(nyquist, interval_splits, velocities, 1)
    except Exception as actual_error:
        monkeypatch.setattr(region_dealias, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _call_sweep_interval_splits(nyquist, interval_splits, velocities, 1)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_sweep_interval_splits(
            nyquist, interval_splits, velocities, 1, monkeypatch
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "velocities",
    [
        np.array([-8.0, 0.0, 9.0], dtype=np.float64),
        np.array([-18.0, -7.5, 3.0, 16.0], dtype=np.float64),
        np.array([], dtype=np.float64),
    ],
)
def test_real_rust_region_sweep_interval_splits_matches_python_fallback(
    monkeypatch, velocities
):
    import pyart._rust as rust

    kernel = getattr(rust, "_region_sweep_interval_splits", None)
    assert kernel is not None

    expected = _fallback_sweep_interval_splits(10.0, 4, velocities, 5, monkeypatch)
    calls = []

    def rust_kernel(name):
        calls.append(name)
        return kernel if name == "_region_sweep_interval_splits" else None

    monkeypatch.setattr(region_dealias, "_rust_kernel", rust_kernel)
    actual = _call_sweep_interval_splits(10.0, 4, velocities, 5)

    assert calls == ["_region_sweep_interval_splits"]
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("nyquist", "interval_splits", "velocities", "match"),
    [
        (10.0, 0, np.array([0.0], dtype=np.float64), "interval_splits"),
        (-10.0, 4, np.array([0.0], dtype=np.float64), "nyquist"),
        (
            10.0,
            4,
            np.array([0.0, 1.0, 2.0], dtype=np.float64)[::2],
            "C-contiguous",
        ),
        (10.0, 4, np.array([np.nan], dtype=np.float64), "finite"),
        (
            1.0e-308,
            4,
            np.array([np.finfo(np.float64).max], dtype=np.float64),
            "extension",
        ),
    ],
)
def test_real_rust_region_sweep_interval_splits_rejects_unsafe_direct_inputs(
    nyquist, interval_splits, velocities, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._region_sweep_interval_splits(nyquist, interval_splits, velocities)


def test_region_edge_sum_and_count_python_fallback_matches_oracle_reduceat(monkeypatch):
    index1, index2, vel1, vel2 = _edge_aggregate_inputs()

    indices, count, velocities = _fallback_edge_sum_and_count(
        index1, index2, vel1, vel2, monkeypatch
    )

    np.testing.assert_array_equal(indices[0], np.array([1, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(indices[1], np.array([3, 4, 5], dtype=np.int32))
    np.testing.assert_array_equal(count, np.array([1, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal(velocities[0], np.array([100.0, 11.0, 22.0]))
    np.testing.assert_array_equal(velocities[1], np.array([400.0, 44.0, 33.0]))


def test_region_edge_sum_and_count_empty_edges_preserves_list_contract(monkeypatch):
    result = _fallback_edge_sum_and_count(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        monkeypatch,
    )

    assert result == (([], []), [], ([], []))


def test_region_edge_sum_and_count_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(index1, index2, vel1, vel2):
        calls.append((index1.dtype, index2.dtype, vel1.dtype, vel2.dtype, index1.shape))
        return (
            (np.array([9], dtype=np.int32), np.array([8], dtype=np.int32)),
            np.array([7], dtype=np.int32),
            (np.array([6.0], dtype=np.float64), np.array([5.0], dtype=np.float64)),
        )

    actual = _run_edge_sum_with_edges(
        *_edge_aggregate_inputs(), monkeypatch=monkeypatch, rust_kernel=rust_kernel
    )

    assert calls == [(np.int32, np.int32, np.float64, np.float64, (5,))]
    np.testing.assert_array_equal(actual[0][0], np.array([9], dtype=np.int32))
    np.testing.assert_array_equal(actual[0][1], np.array([8], dtype=np.int32))
    np.testing.assert_array_equal(actual[1], np.array([7], dtype=np.int32))
    np.testing.assert_array_equal(actual[2][0], np.array([6.0], dtype=np.float64))
    np.testing.assert_array_equal(actual[2][1], np.array([5.0], dtype=np.float64))


@pytest.mark.parametrize(
    ("index1", "index2", "vel1", "vel2"),
    [
        (
            np.array([1, 2], dtype=np.int64),
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([1, 2], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([2.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.int32)[::2],
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
        ),
        (
            np.ma.array([1, 2], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([1, 2], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, np.nan], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
        ),
    ],
)
def test_region_edge_sum_and_count_keeps_python_path_for_unsupported_inputs(
    monkeypatch, index1, index2, vel1, vel2
):
    def fail_if_called(*_args):
        raise AssertionError("unsupported edge aggregation input used Rust")

    actual = _run_edge_sum_with_edges(
        index1, index2, vel1, vel2, monkeypatch, rust_kernel=fail_if_called
    )
    expected = _fallback_edge_sum_and_count(index1, index2, vel1, vel2, monkeypatch)

    np.testing.assert_array_equal(actual[0][0], expected[0][0])
    np.testing.assert_array_equal(actual[0][1], expected[0][1])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2][0], expected[2][0])
    np.testing.assert_array_equal(actual[2][1], expected[2][1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_region_edge_sum_and_count_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    kernel = getattr(rust, "_region_edge_sum_and_count", None)
    assert kernel is not None

    edge_inputs = _edge_aggregate_inputs()
    expected = _fallback_edge_sum_and_count(*edge_inputs, monkeypatch=monkeypatch)
    calls = []

    def rust_kernel(index1, index2, vel1, vel2):
        calls.append((index1.shape, vel1.shape))
        return kernel(index1, index2, vel1, vel2)

    actual = _run_edge_sum_with_edges(*edge_inputs, monkeypatch=monkeypatch, rust_kernel=rust_kernel)

    assert calls == [((5,), (5,))]
    np.testing.assert_array_equal(actual[0][0], expected[0][0])
    np.testing.assert_array_equal(actual[0][1], expected[0][1])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2][0], expected[2][0])
    np.testing.assert_array_equal(actual[2][1], expected[2][1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("index1", "index2", "vel1", "vel2", "match"),
    [
        (
            np.array([1, 2], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
            "same length",
        ),
        (
            np.array([1, 2, 3], dtype=np.int32)[::2],
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
            "C-contiguous",
        ),
        (
            np.array([1, 2], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
            np.array([1.0, np.nan], dtype=np.float64),
            np.array([2.0, 1.0], dtype=np.float64),
            "finite",
        ),
    ],
)
def test_real_rust_region_edge_sum_and_count_rejects_unsafe_direct_inputs(
    index1, index2, vel1, vel2, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._region_edge_sum_and_count(index1, index2, vel1, vel2)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust empty-array checks are verified in installed-wheel mode",
)
def test_real_rust_region_edge_sum_and_count_allows_empty_arrays():
    import pyart._rust as rust

    indices, count, velocities = rust._region_edge_sum_and_count(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )

    assert indices[0].dtype == np.int32
    assert indices[1].dtype == np.int32
    assert count.dtype == np.int32
    assert velocities[0].dtype == np.float64
    assert velocities[1].dtype == np.float64
    assert indices[0].shape == indices[1].shape == count.shape == (0,)
