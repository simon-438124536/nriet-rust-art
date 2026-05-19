import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import bias_and_noise  # noqa: E402


def _fallback_first_mask(data, noise_threshold, monkeypatch):
    monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
    return bias_and_noise._first_mask(data, noise_threshold)


def _fallback_cloud_threshold(data, n_avg, nffts, monkeypatch):
    monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
    return bias_and_noise.cloud_threshold(data, n_avg=n_avg, nffts=nffts)


def _fallback_cloud_mask_4x4(mask1, data, counts_threshold, monkeypatch):
    monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
    return bias_and_noise._cloud_mask_4x4_count(mask1, data, counts_threshold)


def _assert_float_close(actual, expected):
    assert isinstance(actual, np.floating)
    assert isinstance(expected, np.floating)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12, equal_nan=True)
    if np.isfinite(expected):
        assert np.signbit(actual) == np.signbit(expected)


def test_first_mask_python_fallback_preserves_threshold_and_nan_rules(monkeypatch):
    data = np.array([0.0, 2.0, np.nan, np.inf, -np.inf], dtype=np.float64)

    actual = _fallback_first_mask(data, 1.0, monkeypatch)

    assert actual.dtype == np.int16
    np.testing.assert_array_equal(
        actual,
        np.array([0, 1, 0, 1, 0], dtype=np.int16),
    )


def test_first_mask_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(data, noise_threshold):
        calls.append((data.dtype, data.shape, data.flags.c_contiguous, noise_threshold))
        return np.array([9, 8, 7], dtype=np.int16)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_first_mask_f64" else None,
    )
    data = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    actual = bias_and_noise._first_mask(data, np.float64(1.0))

    np.testing.assert_array_equal(actual, np.array([9, 8, 7], dtype=np.int16))
    assert calls == [(np.float64, (3,), True, 1.0)]


def test_first_mask_dispatches_for_zero_dim_float64_array(monkeypatch):
    calls = []

    def rust_kernel(data, noise_threshold):
        calls.append((data.shape, noise_threshold))
        return np.array(5, dtype=np.int16)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_first_mask_f64" else None,
    )

    actual = bias_and_noise._first_mask(
        np.array(2.0, dtype=np.float64),
        np.array(1.0, dtype=np.float64),
    )

    assert actual.shape == ()
    assert actual.dtype == np.int16
    assert actual[()] == 5
    assert calls == [((), 1.0)]


@pytest.mark.parametrize(
    ("data", "noise_threshold"),
    [
        (np.array([0.0, 2.0], dtype=np.float32), 1.0),
        (np.array([0, 2], dtype=np.int32), 1.0),
        (np.array([0.0, 2.0], dtype=np.float64)[::-1], 1.0),
        (np.ma.array([0.0, 2.0], mask=[False, True], dtype=np.float64), 1.0),
        ([0.0, 2.0], 1.0),
        ([2.0, 3.0], [1.0, 1.0]),
        (np.array([0.0, 2.0], dtype=np.complex128), 1.0),
        (np.array([0.0, 2.0], dtype=object), 1.0),
        (np.array([0.0, 2.0], dtype=np.float64), 1.0 + 0j),
        (np.array([0.0, 2.0], dtype=np.float64), "1.0"),
        (np.array([0.0, 2.0], dtype=np.float64), "nan"),
        (np.array([0.0, 2.0], dtype=np.float64), np.array("1.0")),
        (np.array([0.0, 2.0], dtype=np.float64), np.array([1.0])),
        (np.array([0.0, 2.0], dtype=np.float64), np.array([1.0, 1.0])),
    ],
)
def test_first_mask_keeps_python_path_for_unsupported_inputs(
    monkeypatch, data, noise_threshold
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported first-mask input should use fallback")

        return kernel

    monkeypatch.setattr(bias_and_noise, "_rust_kernel", fail_if_called)

    try:
        actual = bias_and_noise._first_mask(data, noise_threshold)
    except Exception as actual_error:
        monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)):
            bias_and_noise._first_mask(data, noise_threshold)
    else:
        expected = _fallback_first_mask(data, noise_threshold, monkeypatch)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_first_mask_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    data = np.array([[0.0, 1.5], [np.nan, np.inf]], dtype=np.float64)
    expected = _fallback_first_mask(data, 1.0, monkeypatch)
    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )

    actual = bias_and_noise._first_mask(data, 1.0)

    assert actual.dtype == np.int16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust scalar/empty checks are verified in installed-wheel mode",
)
def test_real_rust_first_mask_direct_handles_scalar_and_empty_arrays():
    import pyart._rust as rust

    scalar = rust._first_mask_f64(np.array(2.0, dtype=np.float64), 1.0)
    empty = rust._first_mask_f64(np.array([], dtype=np.float64), 1.0)

    assert scalar.shape == ()
    assert scalar.dtype == np.int16
    assert scalar[()] == 1
    assert empty.shape == (0,)
    assert empty.dtype == np.int16


@pytest.mark.parametrize(
    ("data", "n_avg", "nffts"),
    [
        (np.array([-120.0, -100.0, -99.0, -80.0, -50.0], dtype=np.float64), 1.0, None),
        (np.array([-80.0, -70.0, -60.0, -50.0], dtype=np.float64), 4.0, 3),
        (np.array([-120.0, -110.0, -100.0], dtype=np.float64), 1.0, None),
        (np.array([], dtype=np.float64), 1.0, None),
        (np.array([-80.0, np.nan, -70.0], dtype=np.float64), 1.0, None),
        (np.array([-80.0, -np.inf, np.inf], dtype=np.float64), 1.0, None),
    ],
)
def test_cloud_threshold_python_fallback_reference(monkeypatch, data, n_avg, nffts):
    actual = _fallback_cloud_threshold(data, n_avg, nffts, monkeypatch)

    assert isinstance(actual, np.floating)


def test_cloud_threshold_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(data, n_avg, nffts):
        calls.append((data.dtype, data.shape, n_avg, nffts))
        return 7.0

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_cloud_threshold_f64" else None,
    )
    data = np.array([-80.0, -70.0, -60.0], dtype=np.float64)

    actual = bias_and_noise.cloud_threshold(data, n_avg=np.float64(2.0), nffts=None)

    assert isinstance(actual, np.float64)
    assert actual == np.float64(7.0)
    assert calls == [(np.float64, (3,), 2.0, 3)]


@pytest.mark.parametrize(
    ("data", "n_avg", "nffts"),
    [
        (np.array([-80.0, -70.0], dtype=np.float32), 1.0, None),
        (np.array([-80.0, -70.0, -60.0], dtype=np.float64)[::-1], 1.0, None),
        (np.ma.array([-80.0, -70.0], mask=[False, True], dtype=np.float64), 1.0, None),
        (np.array([[-80.0, -70.0]], dtype=np.float64), 1.0, None),
        ([-80.0, -70.0], 1.0, 2),
        (np.array([-80.0, np.nan], dtype=np.float64), 1.0, None),
        (np.array([-80.0, np.inf], dtype=np.float64), 1.0, None),
        (np.array([-80.0, -70.0], dtype=np.float64), -1.0, None),
        (np.array([-80.0, -70.0], dtype=np.float64), np.nan, None),
        (np.array([-80.0, -70.0], dtype=np.float64), 1.0, -1),
        (np.array([-80.0, -70.0], dtype=np.float64), 1.0, 3),
        (np.array([1001.0, -70.0], dtype=np.float64), 1.0, None),
    ],
)
def test_cloud_threshold_keeps_python_path_for_unsupported_inputs(
    monkeypatch, data, n_avg, nffts
):
    def fail_if_called(name):
        if name != "_cloud_threshold_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported cloud_threshold input used Rust")

        return kernel

    monkeypatch.setattr(bias_and_noise, "_rust_kernel", fail_if_called)
    try:
        actual = bias_and_noise.cloud_threshold(data, n_avg=n_avg, nffts=nffts)
    except Exception as actual_error:
        monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            bias_and_noise.cloud_threshold(data, n_avg=n_avg, nffts=nffts)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_cloud_threshold(data, n_avg, nffts, monkeypatch)
        _assert_float_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("data", "n_avg", "nffts"),
    [
        (np.array([-90.0, -80.0, -70.0, -60.0], dtype=np.float64), 1.0, None),
        (np.array([-80.0, -70.0, -60.0, -50.0], dtype=np.float64), 4.0, 3),
        (np.array([], dtype=np.float64), 1.0, None),
        (np.array([-120.0, -110.0, -100.0], dtype=np.float64), 1.0, None),
    ],
)
def test_real_rust_cloud_threshold_matches_python_fallback(
    monkeypatch, data, n_avg, nffts
):
    import pyart._rust as rust

    expected = _fallback_cloud_threshold(data.copy(), n_avg, nffts, monkeypatch)
    calls = []

    def counted_kernel(data_arg, n_avg_arg, nffts_arg):
        calls.append((data_arg.shape, n_avg_arg, nffts_arg))
        return rust._cloud_threshold_f64(data_arg, n_avg_arg, nffts_arg)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_cloud_threshold_f64" else None,
    )
    actual = bias_and_noise.cloud_threshold(data.copy(), n_avg=n_avg, nffts=nffts)

    assert calls == [(data.shape, float(n_avg), data.size if nffts is None else nffts)]
    _assert_float_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_cloud_threshold_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="one-dimensional"):
        rust._cloud_threshold_f64(np.ones((1, 2), dtype=np.float64), 1.0, 2)
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._cloud_threshold_f64(np.ones(4, dtype=np.float64)[::-1], 1.0, 4)
    with pytest.raises(ValueError, match="finite"):
        rust._cloud_threshold_f64(np.array([np.nan], dtype=np.float64), 1.0, 1)
    with pytest.raises(ValueError, match="data length"):
        rust._cloud_threshold_f64(np.ones(2, dtype=np.float64), 1.0, 3)
    with pytest.raises(ValueError, match="dense cloud-threshold range"):
        rust._cloud_threshold_f64(np.array([1001.0], dtype=np.float64), 1.0, 1)


def test_cloud_mask_4x4_python_fallback_reference(monkeypatch):
    mask1 = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int16,
    )
    data = np.zeros(mask1.shape, dtype=np.float64)

    actual = _fallback_cloud_mask_4x4(mask1, data, 6, monkeypatch)

    assert actual.dtype == np.int16
    np.testing.assert_array_equal(
        actual,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0],
            ],
            dtype=np.int16,
        ),
    )


def test_cloud_mask_4x4_dispatches_to_private_rust_kernel(monkeypatch):
    mask1 = np.ones((2, 3), dtype=np.int16)
    data = np.zeros(mask1.shape, dtype=np.float64)
    calls = []

    def rust_kernel(mask_arg, counts_threshold_arg):
        calls.append((mask_arg.dtype, mask_arg.shape, counts_threshold_arg))
        return np.full(mask_arg.shape, 7, dtype=np.int16)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_cloud_mask_4x4_count_i16" else None,
    )

    actual = bias_and_noise._cloud_mask_4x4_count(mask1, data, np.int64(12))

    np.testing.assert_array_equal(actual, np.full(mask1.shape, 7, dtype=np.int16))
    assert calls == [(np.int16, (2, 3), 12)]


@pytest.mark.parametrize(
    ("mask1", "data", "counts_threshold"),
    [
        (np.ones((2, 2), dtype=np.int32), np.zeros((2, 2)), 1),
        (np.ones((2, 2), dtype=np.int16).T, np.zeros((2, 2)), 1),
        (np.ma.array(np.ones((2, 2), dtype=np.int16)), np.zeros((2, 2)), 1),
        (np.array([[0, 2]], dtype=np.int16), np.zeros((1, 2)), 1),
        (np.ones((2, 2), dtype=np.int16), np.zeros((2, 2)).T, 1),
        (np.ones((2, 2), dtype=np.int16), np.ma.array(np.zeros((2, 2))), 1),
        (np.ones((2, 2), dtype=np.int16), np.zeros((1, 2)), 1),
        (np.ones((2, 2), dtype=np.int16), np.zeros((2, 2)), 1.5),
        (np.ones((2, 2), dtype=np.int16), np.zeros((2, 2)), "1"),
        (np.ones((2, 2), dtype=np.int16), np.zeros((2, 2)), True),
        (np.ones((2, 2), dtype=np.int16), np.zeros((2, 2)), -1),
        (np.ones((2, 2), dtype=np.int16), np.zeros((2, 2)), 17),
    ],
)
def test_cloud_mask_4x4_keeps_python_path_for_unsupported_inputs(
    monkeypatch, mask1, data, counts_threshold
):
    def fail_if_called(name):
        if name != "_cloud_mask_4x4_count_i16":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported cloud-mask input used Rust")

        return kernel

    monkeypatch.setattr(bias_and_noise, "_rust_kernel", fail_if_called)
    try:
        actual = bias_and_noise._cloud_mask_4x4_count(mask1, data, counts_threshold)
    except Exception as actual_error:
        monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            bias_and_noise._cloud_mask_4x4_count(mask1, data, counts_threshold)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_cloud_mask_4x4(
            mask1, data, counts_threshold, monkeypatch
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("counts_threshold", [0, 1, 6, 16])
def test_real_rust_cloud_mask_4x4_matches_python_fallback(
    monkeypatch, counts_threshold
):
    import pyart._rust as rust

    mask1 = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int16,
    )
    data = np.zeros(mask1.shape, dtype=np.float64)
    expected = _fallback_cloud_mask_4x4(
        mask1.copy(), data.copy(), counts_threshold, monkeypatch
    )

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = bias_and_noise._cloud_mask_4x4_count(
        mask1.copy(), data.copy(), counts_threshold
    )

    assert actual.dtype == np.int16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_cloud_mask_4x4_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._cloud_mask_4x4_count_i16(np.ones((2, 2), dtype=np.int16).T, 1)
    with pytest.raises(ValueError, match="0 or 1"):
        rust._cloud_mask_4x4_count_i16(np.array([[2]], dtype=np.int16), 1)
    with pytest.raises(ValueError, match="dense cloud-mask range"):
        rust._cloud_mask_4x4_count_i16(np.ones((1, 1), dtype=np.int16), -1)
    with pytest.raises(ValueError, match="dense cloud-mask range"):
        rust._cloud_mask_4x4_count_i16(np.ones((1, 1), dtype=np.int16), 17)
