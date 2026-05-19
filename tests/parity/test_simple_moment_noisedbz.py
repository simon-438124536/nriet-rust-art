import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import simple_moment_calculations as simple_moment  # noqa: E402


def _compute_noisedbz(nrays, noisedbz_val, ranges, ref_dist):
    return simple_moment.compute_noisedBZ(
        nrays, noisedbz_val, ranges, ref_dist, noise_field="noise"
    )["data"]


def _fallback_noisedbz(nrays, noisedbz_val, ranges, ref_dist, monkeypatch):
    monkeypatch.setattr(simple_moment, "_rust_kernel", lambda _name: None)
    return _compute_noisedbz(nrays, noisedbz_val, ranges, ref_dist)


def _assert_exact_masked_array(actual, expected):
    assert np.ma.isMaskedArray(actual)
    assert np.ma.isMaskedArray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.signbit(actual.data), np.signbit(expected.data))


def _assert_exact_ndarray(actual, expected):
    assert type(actual) is np.ndarray
    assert type(expected) is np.ndarray
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("nrays", [0, 1, 3, np.int64(2)])
def test_compute_noisedbz_python_fallback_positive_ranges(monkeypatch, nrays):
    ranges = np.array([1000.0, 2000.0, 4000.0], dtype=np.float64)
    actual = _fallback_noisedbz(nrays, 1.0, ranges, 1.0, monkeypatch)
    noisedbz_vec = 1.0 + 20.0 * np.ma.log10(1e-3 * ranges / 1.0)
    expected = np.tile(noisedbz_vec, (nrays, 1))

    _assert_exact_masked_array(actual, expected)


def test_compute_noisedbz_scalar_range_preserves_oracle_ndarray_fallback(monkeypatch):
    expected = _fallback_noisedbz(2, 1.0, 1000.0, 1.0, monkeypatch)

    def fail_if_called(name):
        if name != "_simple_moment_tile_rows_f64":
            return None

        def kernel(*_args):
            raise AssertionError("scalar noisedBZ range used Rust")

        return kernel

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    actual = _compute_noisedbz(2, 1.0, 1000.0, 1.0)

    assert not np.ma.isMaskedArray(actual)
    _assert_exact_ndarray(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed-wheel dispatch boundaries are verified in installed mode",
)
def test_real_rust_compute_noisedbz_scalar_range_stays_python_owned(monkeypatch):
    import pyart._rust as rust

    expected = _fallback_noisedbz(2, 1.0, 1000.0, 1.0, monkeypatch)

    def fail_tile(name):
        if name in (
            "_simple_moment_tile_rows_f64",
            "_simple_moment_tile_rows_masked_f64",
        ):

            def kernel(*_args):
                raise AssertionError("scalar noisedBZ range used Rust")

            return kernel
        return getattr(rust, name, None)

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_tile)
    actual = _compute_noisedbz(2, 1.0, 1000.0, 1.0)

    assert not np.ma.isMaskedArray(actual)
    _assert_exact_ndarray(actual, expected)


def test_compute_noisedbz_2d_range_preserves_oracle_tile_shape(monkeypatch):
    ranges = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float64)
    expected = _fallback_noisedbz(2, 1.0, ranges, 1.0, monkeypatch)

    def fail_if_called(name):
        if name != "_simple_moment_tile_rows_f64":
            return None

        def kernel(*_args):
            raise AssertionError("2D noisedBZ range used Rust")

        return kernel

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    actual = _compute_noisedbz(2, 1.0, ranges, 1.0)

    assert actual.shape == (4, 2)
    _assert_exact_masked_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed-wheel dispatch boundaries are verified in installed mode",
)
def test_real_rust_compute_noisedbz_2d_range_stays_python_owned(monkeypatch):
    import pyart._rust as rust

    ranges = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float64)
    expected = _fallback_noisedbz(2, 1.0, ranges, 1.0, monkeypatch)

    def fail_tile(name):
        if name in (
            "_simple_moment_tile_rows_f64",
            "_simple_moment_tile_rows_masked_f64",
        ):

            def kernel(*_args):
                raise AssertionError("2D noisedBZ range used Rust")

            return kernel
        return getattr(rust, name, None)

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_tile)
    actual = _compute_noisedbz(2, 1.0, ranges, 1.0)

    assert actual.shape == (4, 2)
    _assert_exact_masked_array(actual, expected)


def test_compute_noisedbz_dispatches_to_private_rust_tile_kernel(monkeypatch):
    calls = []

    def rust_kernel(values, nrays):
        calls.append((values.copy(), values.dtype, values.shape, nrays))
        return np.full((nrays, values.shape[0]), 7.0, dtype=np.float64)

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_simple_moment_tile_rows_f64" else None,
    )
    ranges = np.array([1000.0, 2000.0], dtype=np.float64)
    actual = _compute_noisedbz(np.int64(2), 1.0, ranges, 1.0)

    expected_vec = np.ma.getdata(1.0 + 20.0 * np.ma.log10(1e-3 * ranges / 1.0))
    assert calls[0][1:] == (np.float64, (2,), 2)
    np.testing.assert_array_equal(calls[0][0], expected_vec)
    assert np.ma.isMaskedArray(actual)
    assert actual.fill_value == np.ma.array(expected_vec).fill_value
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.zeros((2, 2), dtype=np.bool_))
    np.testing.assert_array_equal(actual.data, np.full((2, 2), 7.0, dtype=np.float64))


def test_compute_noisedbz_masked_vector_dispatches_to_private_rust_tile_kernel(
    monkeypatch,
):
    calls = []

    def rust_kernel(values, mask, nrays):
        calls.append((values.copy(), mask.copy(), values.dtype, mask.dtype, nrays))
        return (
            np.full((nrays, values.shape[0]), 7.0, dtype=np.float64),
            np.tile(mask, (nrays, 1)),
        )

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_simple_moment_tile_rows_masked_f64"
        else None,
    )
    ranges = np.ma.array(
        [1000.0, 2000.0, 4000.0],
        mask=[False, True, False],
        dtype=np.float64,
    )
    actual = _compute_noisedbz(2, 1.0, ranges, 1.0)
    expected_vec = 1.0 + 20.0 * np.ma.log10(1e-3 * ranges / 1.0)

    assert calls[0][2:] == (np.float64, np.bool_, 2)
    np.testing.assert_array_equal(calls[0][0], np.ma.getdata(expected_vec))
    np.testing.assert_array_equal(calls[0][1], np.ma.getmaskarray(expected_vec))
    assert actual.fill_value == expected_vec.fill_value
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.tile(np.ma.getmaskarray(expected_vec), (2, 1)),
    )
    np.testing.assert_array_equal(
        actual.data,
        np.full((2, 3), 7.0, dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("nrays", "ranges", "ref_dist", "exc_type", "match"),
    [
        (-1, np.array([1000.0], dtype=np.float64), 1.0, ValueError, "negative"),
        (1.5, np.array([1000.0], dtype=np.float64), 1.0, TypeError, "integer"),
    ],
)
def test_compute_noisedbz_keeps_python_path_for_unsupported_inputs(
    monkeypatch, nrays, ranges, ref_dist, exc_type, match
):
    def fail_if_called(name):
        if name != "_simple_moment_tile_rows_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported noisedBZ input used Rust")

        return kernel

    if exc_type is None:
        expected = _fallback_noisedbz(nrays, 1.0, ranges, ref_dist, monkeypatch)
        monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
        actual = _compute_noisedbz(nrays, 1.0, ranges, ref_dist)
        _assert_exact_masked_array(actual, expected)
    else:
        monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
        with pytest.raises(exc_type, match=match):
            _compute_noisedbz(nrays, 1.0, ranges, ref_dist)


@pytest.mark.parametrize(
    "ranges",
    [
        np.array([0.0, 1000.0], dtype=np.float64),
        np.array([-1.0, 1000.0], dtype=np.float64),
        np.ma.array([1000.0, 2000.0], mask=[False, True], dtype=np.float64),
    ],
)
def test_compute_noisedbz_masked_vectors_match_python_tile(monkeypatch, ranges):
    expected = _fallback_noisedbz(2, 1.0, ranges, 1.0, monkeypatch)
    calls = []

    def rust_kernel(values, mask, nrays):
        calls.append((values.copy(), mask.copy(), nrays))
        return np.tile(values, (nrays, 1)), np.tile(mask, (nrays, 1))

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_simple_moment_tile_rows_masked_f64"
        else None,
    )
    actual = _compute_noisedbz(2, 1.0, ranges, 1.0)

    assert len(calls) == 1
    _assert_exact_masked_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("nrays", "ranges"),
    [
        (0, np.array([1000.0, 2000.0], dtype=np.float64)),
        (2, np.array([1000.0, 2000.0], dtype=np.float64)),
        (3, np.array([], dtype=np.float64)),
    ],
)
def test_real_rust_compute_noisedbz_matches_python_tile(monkeypatch, nrays, ranges):
    expected = _fallback_noisedbz(nrays, 1.0, ranges, 1.0, monkeypatch)

    import pyart._rust as rust
    kernel = getattr(rust, "_simple_moment_tile_rows_f64", None)
    assert kernel is not None
    calls = []

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: (
            (lambda *args: (calls.append(tuple(arg.shape if hasattr(arg, "shape") else arg for arg in args)) or kernel(*args)))
            if name == "_simple_moment_tile_rows_f64"
            else getattr(rust, name, None)
        ),
    )
    actual = _compute_noisedbz(nrays, 1.0, ranges, 1.0)

    assert calls == [((ranges.shape[0],), nrays)]
    _assert_exact_masked_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "ranges",
    [
        np.array([0.0, 1000.0], dtype=np.float64),
        np.array([-1.0, 1000.0], dtype=np.float64),
        np.ma.array([1000.0, 2000.0], mask=[False, True], dtype=np.float64),
    ],
)
def test_real_rust_compute_noisedbz_masked_matches_python_tile(
    monkeypatch, ranges
):
    expected = _fallback_noisedbz(2, 1.0, ranges, 1.0, monkeypatch)

    import pyart._rust as rust
    kernel = getattr(rust, "_simple_moment_tile_rows_masked_f64", None)
    assert kernel is not None
    calls = []

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: (
            (lambda *args: (calls.append(tuple(arg.shape if hasattr(arg, "shape") else arg for arg in args)) or kernel(*args)))
            if name == "_simple_moment_tile_rows_masked_f64"
            else getattr(rust, name, None)
        ),
    )
    actual = _compute_noisedbz(2, 1.0, ranges, 1.0)

    assert calls == [((ranges.shape[0],), (ranges.shape[0],), 2)]
    _assert_exact_masked_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_tile_rows_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    values = np.arange(6.0, dtype=np.float64)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._simple_moment_tile_rows_f64(values, 2)
    with pytest.raises((OverflowError, TypeError, ValueError)):
        rust._simple_moment_tile_rows_f64(np.ones(2, dtype=np.float64), -1)
    with pytest.raises(ValueError, match="same length"):
        rust._simple_moment_tile_rows_masked_f64(
            np.ones(2, dtype=np.float64),
            np.array([False], dtype=np.bool_),
            2,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._simple_moment_tile_rows_masked_f64(
            values,
            np.array([False, True, False], dtype=np.bool_),
            2,
        )
    with pytest.raises(ValueError, match="mask-free"):
        rust._simple_moment_tile_rows_masked_f64(
            np.ma.array([1.0, 2.0], mask=[False, True]),
            np.array([False, True], dtype=np.bool_),
            2,
        )
    with pytest.raises(ValueError, match="float64"):
        rust._simple_moment_tile_rows_masked_f64(
            np.ones(2, dtype=np.float32),
            np.array([False, True], dtype=np.bool_),
            2,
        )
    with pytest.raises(ValueError, match="bool"):
        rust._simple_moment_tile_rows_masked_f64(
            np.ones(2, dtype=np.float64),
            np.array([0, 1], dtype=np.int8),
            2,
        )
