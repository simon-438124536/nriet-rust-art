import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import qpe  # noqa: E402


class _Radar(SimpleNamespace):
    def check_field_exists(self, name):
        if name not in self.fields:
            raise KeyError("Field not available: " + name)


def _radar(kdp):
    return _Radar(
        fields={"kdp": {"data": kdp}},
        instrument_parameters={},
    )


def _compute_rain_rate_kdp(kdp, alpha=29.70, beta=0.85):
    return qpe.est_rain_rate_kdp(
        _radar(kdp),
        alpha=alpha,
        beta=beta,
        kdp_field="kdp",
        rr_field="rr",
    )["data"]


def _fallback_rain_rate_kdp(kdp, monkeypatch, alpha=29.70, beta=0.85):
    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
    return _compute_rain_rate_kdp(kdp, alpha=alpha, beta=beta)


def _assert_rain_rate_close(actual, expected):
    assert np.ma.isMaskedArray(actual)
    assert np.ma.isMaskedArray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    mask = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)
    np.testing.assert_array_equal(actual.data[mask], expected.data[mask])
    np.testing.assert_allclose(
        actual.data[~mask],
        expected.data[~mask],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(np.signbit(actual.data), np.signbit(expected.data))


def test_rain_rate_kdp_python_fallback_reference_preserves_input_clamp(monkeypatch):
    kdp = np.array([[-1.0, -0.0, 0.0, 0.5, 1.0, 3.0]], dtype=np.float64)

    actual = _fallback_rain_rate_kdp(kdp, monkeypatch)

    assert actual.dtype == np.float64
    assert np.ma.isMaskedArray(actual)
    np.testing.assert_array_equal(kdp, np.array([[0.0, -0.0, 0.0, 0.5, 1.0, 3.0]]))


def test_rain_rate_kdp_dispatches_to_private_rust_kernel_after_clamp(monkeypatch):
    kdp = np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)
    calls = []

    def rust_kernel(kdp_arg, alpha, beta):
        calls.append((kdp_arg.dtype, kdp_arg.copy(), alpha, beta))
        return np.full(kdp_arg.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qpe_rain_rate_kdp_dense_f64" else None,
    )

    actual = _compute_rain_rate_kdp(kdp, alpha=np.float64(0.5), beta=np.float64(1.2))

    assert len(calls) == 1
    assert calls[0][0] == np.dtype("float64")
    np.testing.assert_array_equal(calls[0][1], np.array([[0.0, 0.0, 1.0]]))
    assert calls[0][2:] == (0.5, 1.2)
    expected = np.ma.array(np.full(kdp.shape, 7.0, dtype=np.float64))
    _assert_rain_rate_close(actual, expected)


@pytest.mark.parametrize(
    ("kdp", "alpha", "beta"),
    [
        (np.array([[0.0, 1.0]], dtype=np.float32), 29.70, 0.85),
        (
            np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)[:, ::2],
            29.70,
            0.85,
        ),
        (
            np.ma.array([[0.0, 1.0]], mask=[[False, True]], dtype=np.float64),
            29.70,
            0.85,
        ),
        (np.array([[np.nan, 1.0]], dtype=np.float64), 29.70, 0.85),
        (np.array([[np.inf, 1.0]], dtype=np.float64), 29.70, 0.85),
        (np.array([[1.0e13, 1.0]], dtype=np.float64), 29.70, 0.85),
        (np.array([[0.0]], dtype=np.float64), 29.70, -1.0),
        (np.array([[1.0]], dtype=np.float64), np.nan, 0.85),
        (np.array([[1.0]], dtype=np.float64), 29.70, np.nan),
        (np.array([[1.0]], dtype=np.float64), "1.0", 0.85),
        (np.array([[1.0]], dtype=np.float64), 29.70, "1.0"),
        (np.array([[1.0]], dtype=np.float64), True, 0.85),
        (np.array([[1.0]], dtype=np.float64), 29.70, False),
        (np.array([[1.0]], dtype=np.float64), 1.0e60, 0.85),
        (np.array([[1.0e12]], dtype=np.float64), 29.70, 100.0),
    ],
)
def test_rain_rate_kdp_keeps_python_path_for_unsupported_inputs(
    monkeypatch, kdp, alpha, beta
):
    def fail_if_called(name):
        if name != "_qpe_rain_rate_kdp_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported rain-rate-KDP input used Rust")

        return kernel

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)
    try:
        actual = _compute_rain_rate_kdp(kdp, alpha=alpha, beta=beta)
    except Exception as actual_error:
        monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _compute_rain_rate_kdp(kdp, alpha=alpha, beta=beta)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_rain_rate_kdp(kdp, monkeypatch, alpha=alpha, beta=beta)
        _assert_rain_rate_close(actual, expected)


def test_rain_rate_kdp_missing_field_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field rain-rate-KDP path reached Rust")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)
    with pytest.raises(KeyError, match="Field not available: missing"):
        qpe.est_rain_rate_kdp(
            _radar(np.ones((1, 1))),
            alpha=29.70,
            beta=0.85,
            kdp_field="missing",
            rr_field="rr",
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_rain_rate_kdp_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    kdp = np.array([[-1.0, -0.0, 0.0, 0.5, 1.0, 3.0]], dtype=np.float64)
    expected_input = kdp.copy()
    expected = _fallback_rain_rate_kdp(expected_input, monkeypatch)
    calls = []

    def counted_kernel(kdp_arg, alpha, beta):
        calls.append((kdp_arg.shape, alpha, beta, kdp_arg.copy()))
        return rust._qpe_rain_rate_kdp_dense_f64(kdp_arg, alpha, beta)

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_qpe_rain_rate_kdp_dense_f64" else None,
    )
    actual = _compute_rain_rate_kdp(kdp)

    assert calls[0][:3] == ((1, 6), 29.70, 0.85)
    np.testing.assert_array_equal(calls[0][3], expected_input)
    np.testing.assert_array_equal(kdp, expected_input)
    _assert_rain_rate_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_rain_rate_kdp_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._qpe_rain_rate_kdp_dense_f64(
            np.arange(8.0, dtype=np.float64).reshape(2, 4)[:, ::2],
            29.70,
            0.85,
        )
    with pytest.raises(ValueError, match="finite"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([np.nan], dtype=np.float64), 29.70, 0.85)
    with pytest.raises(ValueError, match="alpha and beta"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([1.0], dtype=np.float64), np.nan, 0.85)
    with pytest.raises(ValueError, match="non-boolean"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([1.0], dtype=np.float64), True, 0.85)
    with pytest.raises(ValueError, match="non-boolean"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([1.0], dtype=np.float64), 29.70, False)
    with pytest.raises(ValueError, match="dense KDP rain-rate kernel range"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([-1.0], dtype=np.float64), 29.70, 0.85)
    with pytest.raises(ValueError, match="dense KDP rain-rate kernel range"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([1.0e13], dtype=np.float64), 29.70, 0.85)
    with pytest.raises(ValueError, match="dense KDP rain-rate kernel range"):
        rust._qpe_rain_rate_kdp_dense_f64(np.array([0.0], dtype=np.float64), 29.70, -1.0)
