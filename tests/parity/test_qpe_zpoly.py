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


def _compute_zpoly(refl):
    return qpe.est_rain_rate_zpoly(_radar(refl), refl_field="refl", rr_field="rr")[
        "data"
    ]


def _fallback_zpoly(refl, monkeypatch):
    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
    return _compute_zpoly(refl)


def _assert_zpoly_close(actual, expected):
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


def test_zpoly_python_fallback_reference(monkeypatch):
    refl = np.array([[-10.0, 0.0, 10.0, 30.0, 60.0, 100.0]], dtype=np.float64)

    actual = _fallback_zpoly(refl, monkeypatch)

    assert actual.dtype == np.float64
    assert np.ma.isMaskedArray(actual)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.zeros(refl.shape, dtype=bool))


def test_zpoly_dispatches_to_private_rust_kernel(monkeypatch):
    refl = np.array([[0.0, 10.0]], dtype=np.float64)
    calls = []

    def rust_kernel(refl_arg):
        calls.append((refl_arg.dtype, refl_arg.shape))
        return np.full(refl_arg.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qpe_zpoly_dense_f64" else None,
    )

    actual = _compute_zpoly(refl)

    assert calls == [(np.float64, (1, 2))]
    expected = np.ma.array(np.full(refl.shape, 7.0, dtype=np.float64))
    _assert_zpoly_close(actual, expected)


@pytest.mark.parametrize(
    "refl",
    [
        np.array([[0.0, 10.0]], dtype=np.float32),
        np.array([[0.0, 10.0, 20.0, 30.0]], dtype=np.float64)[:, ::2],
        np.ma.array([[0.0, 10.0]], mask=[[False, True]], dtype=np.float64),
        np.array([[np.nan, 10.0]], dtype=np.float64),
        np.array([[np.inf, 10.0]], dtype=np.float64),
        np.array([[1001.0, 10.0]], dtype=np.float64),
    ],
)
def test_zpoly_keeps_python_path_for_unsupported_inputs(monkeypatch, refl):
    expected = _fallback_zpoly(refl, monkeypatch)

    def fail_if_called(name):
        if name != "_qpe_zpoly_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported zpoly input used Rust")

        return kernel

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)
    actual = _compute_zpoly(refl)

    _assert_zpoly_close(actual, expected)


def test_zpoly_missing_field_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field zpoly path reached Rust")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)
    with pytest.raises(KeyError, match="Field not available: missing"):
        qpe.est_rain_rate_zpoly(
            _radar(np.ones((1, 1))), refl_field="missing", rr_field="rr"
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_zpoly_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    refl = np.array([[-10.0, -0.0, 0.0, 10.0, 30.0, 60.0, 100.0]], dtype=np.float64)
    expected = _fallback_zpoly(refl.copy(), monkeypatch)
    calls = []

    def counted_kernel(refl_arg):
        calls.append((refl_arg.shape,))
        return rust._qpe_zpoly_dense_f64(refl_arg)

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_qpe_zpoly_dense_f64" else None,
    )
    actual = _compute_zpoly(refl.copy())

    assert calls == [((1, 7),)]
    _assert_zpoly_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_zpoly_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._qpe_zpoly_dense_f64(
            np.arange(8.0, dtype=np.float64).reshape(2, 4)[:, ::2]
        )
    with pytest.raises(ValueError, match="finite"):
        rust._qpe_zpoly_dense_f64(np.array([np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="dense Z-poly kernel range"):
        rust._qpe_zpoly_dense_f64(np.array([1001.0], dtype=np.float64))
