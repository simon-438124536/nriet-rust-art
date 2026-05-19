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


def _radar(refl):
    return _Radar(fields={"refl": {"data": refl}})


def _compute_rain_rate_z(refl, alpha=0.0376, beta=0.6112):
    return qpe.est_rain_rate_z(
        _radar(refl),
        alpha=alpha,
        beta=beta,
        refl_field="refl",
        rr_field="rr",
    )["data"]


def _fallback_rain_rate_z(refl, monkeypatch, alpha=0.0376, beta=0.6112):
    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
    return _compute_rain_rate_z(refl, alpha=alpha, beta=beta)


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


def test_rain_rate_z_python_fallback_reference(monkeypatch):
    refl = np.array([[-10.0, -0.0, 0.0, 10.0, 30.0, 60.0, 100.0]], dtype=np.float64)

    actual = _fallback_rain_rate_z(refl, monkeypatch)

    assert actual.dtype == np.float64
    assert np.ma.isMaskedArray(actual)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.zeros(refl.shape, dtype=bool))


def test_rain_rate_z_dispatches_to_private_rust_kernel(monkeypatch):
    refl = np.array([[0.0, 10.0]], dtype=np.float64)
    calls = []

    def rust_kernel(refl_arg, alpha, beta):
        calls.append((refl_arg.dtype, refl_arg.shape, alpha, beta))
        return np.full(refl_arg.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qpe_rain_rate_z_dense_f64" else None,
    )

    actual = _compute_rain_rate_z(refl, alpha=np.float64(0.5), beta=np.float64(1.2))

    assert calls == [(np.float64, (1, 2), 0.5, 1.2)]
    expected = np.ma.array(np.full(refl.shape, 7.0, dtype=np.float64))
    _assert_rain_rate_close(actual, expected)


@pytest.mark.parametrize(
    ("refl", "alpha", "beta"),
    [
        (np.array([[0.0, 10.0]], dtype=np.float32), 0.0376, 0.6112),
        (
            np.array([[0.0, 10.0, 20.0, 30.0]], dtype=np.float64)[:, ::2],
            0.0376,
            0.6112,
        ),
        (
            np.ma.array([[0.0, 10.0]], mask=[[False, True]], dtype=np.float64),
            0.0376,
            0.6112,
        ),
        (np.array([[np.nan, 10.0]], dtype=np.float64), 0.0376, 0.6112),
        (np.array([[np.inf, 10.0]], dtype=np.float64), 0.0376, 0.6112),
        (np.array([[3100.0, 10.0]], dtype=np.float64), 0.0376, 0.6112),
        (np.array([[10.0]], dtype=np.float64), np.nan, 0.6112),
        (np.array([[10.0]], dtype=np.float64), 0.0376, np.nan),
        (np.array([[10.0]], dtype=np.float64), "1.0", 0.6112),
        (np.array([[10.0]], dtype=np.float64), 0.0376, "1.0"),
        (np.array([[10.0]], dtype=np.float64), True, 0.6112),
        (np.array([[10.0]], dtype=np.float64), 0.0376, False),
        (np.array([[10.0]], dtype=np.float64), 1.0e60, 0.6112),
        (np.array([[10.0]], dtype=np.float64), 0.0376, 300.0),
    ],
)
def test_rain_rate_z_keeps_python_path_for_unsupported_inputs(
    monkeypatch, refl, alpha, beta
):
    def fail_if_called(name):
        if name != "_qpe_rain_rate_z_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported rain-rate-Z input used Rust")

        return kernel

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)
    try:
        actual = _compute_rain_rate_z(refl, alpha=alpha, beta=beta)
    except Exception as actual_error:
        monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _compute_rain_rate_z(refl, alpha=alpha, beta=beta)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_rain_rate_z(refl, monkeypatch, alpha=alpha, beta=beta)
        _assert_rain_rate_close(actual, expected)


def test_rain_rate_z_missing_field_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field rain-rate-Z path reached Rust")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)
    with pytest.raises(KeyError, match="Field not available: missing"):
        qpe.est_rain_rate_z(
            _radar(np.ones((1, 1))), refl_field="missing", rr_field="rr"
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_rain_rate_z_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    refl = np.array([[-10.0, -0.0, 0.0, 10.0, 30.0, 60.0, 100.0]], dtype=np.float64)
    expected = _fallback_rain_rate_z(refl.copy(), monkeypatch)
    calls = []

    def counted_kernel(refl_arg, alpha, beta):
        calls.append((refl_arg.shape, alpha, beta))
        return rust._qpe_rain_rate_z_dense_f64(refl_arg, alpha, beta)

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_qpe_rain_rate_z_dense_f64" else None,
    )
    actual = _compute_rain_rate_z(refl.copy())

    assert calls == [((1, 7), 0.0376, 0.6112)]
    _assert_rain_rate_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_rain_rate_z_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._qpe_rain_rate_z_dense_f64(
            np.arange(8.0, dtype=np.float64).reshape(2, 4)[:, ::2],
            0.0376,
            0.6112,
        )
    with pytest.raises(ValueError, match="finite"):
        rust._qpe_rain_rate_z_dense_f64(np.array([np.nan], dtype=np.float64), 0.0376, 0.6112)
    with pytest.raises(ValueError, match="alpha and beta"):
        rust._qpe_rain_rate_z_dense_f64(np.array([10.0], dtype=np.float64), np.nan, 0.6112)
    with pytest.raises(ValueError, match="non-boolean"):
        rust._qpe_rain_rate_z_dense_f64(np.array([10.0], dtype=np.float64), True, 0.6112)
    with pytest.raises(ValueError, match="non-boolean"):
        rust._qpe_rain_rate_z_dense_f64(np.array([10.0], dtype=np.float64), 0.0376, False)
    with pytest.raises(ValueError, match="dense Z rain-rate kernel range"):
        rust._qpe_rain_rate_z_dense_f64(np.array([1001.0], dtype=np.float64), 0.0376, 0.6112)
    with pytest.raises(ValueError, match="dense Z rain-rate kernel range"):
        rust._qpe_rain_rate_z_dense_f64(np.array([10.0], dtype=np.float64), 0.0376, 300.0)
