import numpy as np
import pytest

from pyart.retrieve import cfad


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_cfad_normalize_dense_f64"):
        pytest.skip("pyart._rust has no CFAD normalize kernel in this test mode")
    return rust


def _fallback_normalize(freq, min_frac_thres, monkeypatch):
    monkeypatch.setattr(cfad, "_rust_kernel", lambda _name: None)
    return cfad._normalize_cfad(freq, min_frac_thres)


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


@pytest.mark.parametrize(
    ("freq", "min_frac_thres"),
    [
        (np.array([[1.0, 2.0], [3.0, 1.0], [10.0, 0.0]]), 0.5),
        (np.array([[1.0, 0.0], [2.0, 2.0]]), -1.0),
        (np.empty((2, 0), dtype=np.float64), 0.1),
    ],
)
def test_cfad_normalize_python_fallback_reference_cases(
    monkeypatch, freq, min_frac_thres
):
    actual = _fallback_normalize(freq, min_frac_thres, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float64
    assert actual.shape == freq.shape
    assert actual.fill_value == 1e20


def test_cfad_normalize_zero_rows_keep_python_warning_and_nan(monkeypatch):
    freq = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
        actual = _fallback_normalize(freq, 0.1, monkeypatch)

    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[True, True], [False, False]], dtype=bool),
    )
    assert np.isnan(actual.data[0, 0])
    assert np.isnan(actual.data[0, 1])


def test_cfad_normalize_zero_rows_unmasked_when_threshold_is_zero(monkeypatch):
    freq = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
        actual = _fallback_normalize(freq, 0.0, monkeypatch)

    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[False, False], [False, False]], dtype=bool),
    )
    assert np.isnan(actual.data[0, 0])
    assert np.isnan(actual.data[0, 1])


def test_cfad_normalize_dispatches_dense_freq_to_private_rust_kernel(monkeypatch):
    calls = []
    freq = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float64)
    out = np.array([[1.0 / 3.0, 2.0 / 3.0], [0.75, 0.25]], dtype=np.float64)
    mask = np.array([[True, True], [False, False]], dtype=bool)

    def kernel(freq_arg, min_frac_arg):
        calls.append((freq_arg.dtype, freq_arg.shape, min_frac_arg))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        cfad,
        "_rust_kernel",
        lambda name: kernel if name == "_cfad_normalize_dense_f64" else None,
    )

    actual = cfad._normalize_cfad(freq, 0.5)

    assert calls == [(np.dtype(np.float64), (2, 2), 0.5)]
    assert actual.fill_value == 1e20
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("freq", "min_frac_thres"),
    [
        (np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float64), 0.1),
        (np.array([[1.0, np.nan]], dtype=np.float64), 0.1),
        (np.empty((2, 0), dtype=np.float64), 0.1),
        (np.array([[1.0, 2.0]], dtype=np.float32), 0.1),
        (np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)[:, ::2], 0.1),
        (np.ma.array([[1.0, 2.0]], dtype=np.float64), 0.1),
        (np.array([1.0, 2.0], dtype=np.float64), 0.1),
        (np.array([[1.0, 2.0]], dtype=np.float64), True),
        (np.array([[1.0, 2.0]], dtype=np.float64), float("nan")),
    ],
)
def test_cfad_normalize_unsupported_inputs_keep_python_fallback(
    monkeypatch, freq, min_frac_thres
):
    def rust_kernel(name):
        if name != "_cfad_normalize_dense_f64":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(cfad, "_rust_kernel", rust_kernel)
    try:
        actual = cfad._normalize_cfad(freq, min_frac_thres)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_normalize(freq, min_frac_thres, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_normalize(freq, min_frac_thres, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("freq", "min_frac_thres"),
    [
        (np.array([[1.0, 2.0], [3.0, 1.0], [10.0, 0.0]]), 0.5),
        (np.array([[1.0, 0.0], [2.0, 2.0]]), -1.0),
    ],
)
def test_real_rust_cfad_normalize_matches_python_fallback(
    monkeypatch, freq, min_frac_thres
):
    rust = _rust_or_skip()

    expected = _fallback_normalize(freq, min_frac_thres, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_cfad_normalize_dense_f64":
            calls.append(name)
            return rust._cfad_normalize_dense_f64
        return None

    monkeypatch.setattr(cfad, "_rust_kernel", rust_kernel)
    actual = cfad._normalize_cfad(freq, min_frac_thres)

    assert calls == ["_cfad_normalize_dense_f64"]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("freq", "min_frac_thres", "match"),
    [
        (
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)[:, ::2],
            0.1,
            "C-contiguous",
        ),
        (np.empty((0, 2), dtype=np.float64), 0.1, "zero-size"),
        (np.array([[0.0, 0.0]], dtype=np.float64), 0.1, "row sums"),
        (np.array([[1.0, np.nan]], dtype=np.float64), 0.1, "finite"),
        (np.array([[1.0, 2.0]], dtype=np.float64), float("nan"), "finite"),
    ],
)
def test_real_rust_cfad_normalize_direct_rejects_unsafe_inputs(
    freq, min_frac_thres, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._cfad_normalize_dense_f64(freq, min_frac_thres)


@pytest.mark.parametrize(
    "freq",
    [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float64),
    ],
)
def test_real_rust_cfad_normalize_direct_rejects_binding_type_drift(freq):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        rust._cfad_normalize_dense_f64(freq, 0.1)
