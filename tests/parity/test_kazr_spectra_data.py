import os

import numpy as np
import pytest

from pyart.aux_io import kazr_spectra


class _Values:
    def __init__(self, values):
        self.values = values


class _Selection:
    def __init__(self, values):
        self.values = values


class _LocatorMask:
    def __init__(self, values):
        self.values = values

    def sel(self, time):
        return _Selection(self.values)


class _FakeXrObj:
    def __init__(self, locs, spectra):
        self.locator_mask = _LocatorMask(locs)
        self.spectra = _Values(spectra)
        self.speclength = _Values(np.arange(spectra.shape[1]))


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_kazr_get_spectra"):
        pytest.skip("pyart._rust has no KAZR spectra kernel")
    return rust


def _fallback_get_spectra(xrobj, monkeypatch):
    monkeypatch.setattr(kazr_spectra, "_rust_kernel", lambda _name: None)
    return kazr_spectra._get_spectra(xrobj, 0)


def _assert_array_equal_with_nan(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual[~np.isnan(expected)], expected[~np.isnan(expected)])


def test_kazr_get_spectra_python_fallback_reference_cases(monkeypatch):
    locs = np.array([0.0, np.nan, 1.9, -9999.0, -0.8], dtype=np.float64)
    spectra = np.arange(15, dtype=np.float64).reshape(5, 3)
    xrobj = _FakeXrObj(locs, spectra)

    actual = _fallback_get_spectra(xrobj, monkeypatch)

    assert actual.dtype == np.float64
    assert actual.shape == (5, 3)
    np.testing.assert_array_equal(locs, np.array([0.0, -9999.0, 1.9, -9999.0, -0.8]))
    np.testing.assert_array_equal(np.isnan(actual), [[False] * 3, [True] * 3, [False] * 3, [True] * 3, [False] * 3])


def test_kazr_get_spectra_dispatches_dense_float64_to_private_rust(monkeypatch):
    locs = np.array([0.0, -9999.0, 2.0], dtype=np.float64)
    spectra = np.arange(12, dtype=np.float64).reshape(4, 3)
    xrobj = _FakeXrObj(locs, spectra)
    calls = []
    out = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [7.0, 8.0, 9.0]], dtype=np.float64)

    def kernel(spectra_arg, indices, missing, npulses):
        calls.append((spectra_arg.dtype, spectra_arg.shape, indices.copy(), missing.copy(), npulses))
        return out.copy()

    monkeypatch.setattr(
        kazr_spectra,
        "_rust_kernel",
        lambda name: kernel if name == "_kazr_get_spectra" else None,
    )

    actual = kazr_spectra._get_spectra(xrobj, 0)

    assert calls[0][0:2] == (np.dtype(np.float64), (4, 3))
    np.testing.assert_array_equal(calls[0][2], [0, 0, 2])
    np.testing.assert_array_equal(calls[0][3], [False, True, False])
    assert calls[0][4] == 3
    _assert_array_equal_with_nan(actual, out)


def test_kazr_get_spectra_object_locator_keeps_oracle_error(monkeypatch):
    locs = np.array([0.0, 1.0], dtype=object)
    spectra = np.arange(6, dtype=np.float64).reshape(2, 3)
    xrobj = _FakeXrObj(locs, spectra)

    def rust_kernel(name):
        if name == "_kazr_get_spectra":
            raise AssertionError("object locator used Rust kernel")
        return None

    monkeypatch.setattr(kazr_spectra, "_rust_kernel", rust_kernel)

    with pytest.raises(TypeError) as excinfo:
        kazr_spectra._get_spectra(xrobj, 0)
    assert "ufunc 'isnan' not supported" in excinfo.value.args[0]


def test_kazr_get_spectra_oversized_output_keeps_python_path(monkeypatch):
    locs = np.array([], dtype=np.float64)
    spectra = np.zeros((1, 1), dtype=np.float64)

    def rust_kernel(name):
        if name == "_kazr_get_spectra":
            raise AssertionError("oversized KAZR output used Rust kernel")
        return None

    monkeypatch.setattr(kazr_spectra, "_rust_kernel", rust_kernel)
    assert (
        kazr_spectra._get_spectra_rust(
            locs,
            spectra,
            kazr_spectra.KAZR_RUST_MAX_OUTPUT_VALUES + 1,
        )
        is None
    )


@pytest.mark.parametrize(
    ("locs", "spectra"),
    [
        (np.array([0.0, 9.0], dtype=np.float64), np.arange(6, dtype=np.float64).reshape(2, 3)),
        (np.array([np.inf], dtype=np.float64), np.arange(6, dtype=np.float64).reshape(2, 3)),
        (np.array([0.0, 1.0, -9999.0], dtype=np.float64)[::2], np.arange(6, dtype=np.float64).reshape(2, 3)),
        (np.array([0.0], dtype=np.float64), np.arange(6, dtype=np.float32).reshape(2, 3)),
    ],
)
def test_kazr_get_spectra_unsupported_inputs_keep_python_path(monkeypatch, locs, spectra):
    xrobj = _FakeXrObj(locs, spectra)

    def rust_kernel(name):
        if name == "_kazr_get_spectra":
            raise AssertionError("unsupported KAZR input used Rust kernel")
        return None

    monkeypatch.setattr(kazr_spectra, "_rust_kernel", rust_kernel)
    try:
        actual = kazr_spectra._get_spectra(xrobj, 0)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_get_spectra(_FakeXrObj(locs.copy(), spectra), monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_get_spectra(_FakeXrObj(locs.copy(), spectra), monkeypatch)
        _assert_array_equal_with_nan(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust KAZR parity",
)
def test_kazr_get_spectra_real_rust_matches_python_fallback(monkeypatch):
    locs = np.array([0.0, np.nan, 1.9, -9999.0, -0.8], dtype=np.float64)
    spectra = np.arange(15, dtype=np.float64).reshape(5, 3)
    expected = _fallback_get_spectra(_FakeXrObj(locs.copy(), spectra), monkeypatch)
    monkeypatch.undo()

    actual_locs = locs.copy()
    actual = kazr_spectra._get_spectra(_FakeXrObj(actual_locs, spectra), 0)

    _assert_array_equal_with_nan(actual, expected)
    np.testing.assert_array_equal(actual_locs, np.array([0.0, -9999.0, 1.9, -9999.0, -0.8]))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust KAZR checks",
)
def test_kazr_get_spectra_direct_rust_helper():
    rust = _rust_or_skip()
    spectra = np.arange(12, dtype=np.float64).reshape(4, 3)
    data = rust._kazr_get_spectra(
        spectra,
        np.array([0, 2, 0], dtype=np.int64),
        np.array([False, False, True], dtype=bool),
        3,
    )
    expected = np.array([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [np.nan, np.nan, np.nan]])
    _assert_array_equal_with_nan(data, expected)

    with pytest.raises(ValueError):
        rust._kazr_get_spectra(spectra, np.array([10], dtype=np.int64), np.array([False]), 3)
