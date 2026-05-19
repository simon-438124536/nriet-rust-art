import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import simple_moment_calculations as simple_moment  # noqa: E402


def _radar(rhohv, zdr):
    return SimpleNamespace(fields={"rhohv": {"data": rhohv}, "zdr": {"data": zdr}})


def _compute_cdr(rhohv, zdr):
    return simple_moment.compute_cdr(
        _radar(rhohv, zdr),
        rhohv_field="rhohv",
        zdr_field="zdr",
        cdr_field="cdr",
    )["data"]


def _fallback_cdr(rhohv, zdr, monkeypatch):
    monkeypatch.setattr(simple_moment, "_rust_kernel", lambda _name: None)
    return _compute_cdr(rhohv, zdr)


def _assert_cdr_close(actual, expected):
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
        atol=1.0e-13,
    )
    np.testing.assert_array_equal(np.signbit(actual.data), np.signbit(expected.data))


def test_compute_cdr_python_fallback_masks_invalid_domain(monkeypatch):
    rhohv = np.array([[0.0, 0.5, 0.9, 1.0, -1.0]], dtype=np.float64)
    zdr = np.array([[0.0, 1.0, -1.0, 0.0, 0.0]], dtype=np.float64)

    actual = _fallback_cdr(rhohv, zdr, monkeypatch)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), [[False, False, False, True, True]]
    )
    np.testing.assert_array_equal(actual.data[0, 3:], [10.0, 10.0])


def test_compute_cdr_dispatches_to_private_rust_kernel(monkeypatch):
    rhohv = np.array([[0.2, 0.3]], dtype=np.float64)
    zdr = np.array([[1.0, -1.0]], dtype=np.float64)
    calls = []

    def rust_kernel(rhohv_arg, zdr_arg):
        calls.append((rhohv_arg.dtype, rhohv_arg.shape, zdr_arg.dtype, zdr_arg.shape))
        return (
            np.full(rhohv_arg.shape, 7.0, dtype=np.float64),
            np.array([[False, True]], dtype=bool),
        )

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_simple_moment_cdr_dense_f64" else None,
    )

    actual = _compute_cdr(rhohv, zdr)

    assert calls == [(np.float64, (1, 2), np.float64, (1, 2))]
    expected = np.ma.array([[7.0, 7.0]], mask=[[False, True]])
    _assert_cdr_close(actual, expected)


@pytest.mark.parametrize(
    ("rhohv", "zdr"),
    [
        (
            np.ma.array([[0.2, 0.3]], mask=[[False, True]], dtype=np.float64),
            np.array([[1.0, -1.0]], dtype=np.float64),
        ),
        (
            np.array([[0.2, 0.3]], dtype=np.float64),
            np.ma.array([[1.0, -1.0]], mask=[[False, True]], dtype=np.float64),
        ),
        (
            np.array([[0.2, 0.3]], dtype=np.float32),
            np.array([[1.0, -1.0]], dtype=np.float64),
        ),
        (
            np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float64)[:, ::2],
            np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float64)[:, ::2],
        ),
        (
            np.array([[np.nan, 0.3]], dtype=np.float64),
            np.array([[1.0, -1.0]], dtype=np.float64),
        ),
        (
            np.array([[0.2, 0.3]], dtype=np.float64),
            np.array([[np.inf, -1.0]], dtype=np.float64),
        ),
    ],
)
def test_compute_cdr_keeps_python_path_for_unsupported_inputs(
    monkeypatch, rhohv, zdr
):
    expected = _fallback_cdr(rhohv.copy(), zdr.copy(), monkeypatch)

    def fail_if_called(name):
        if name != "_simple_moment_cdr_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported compute_cdr input used Rust")

        return kernel

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    actual = _compute_cdr(rhohv, zdr)

    _assert_cdr_close(actual, expected)


def test_compute_cdr_missing_field_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field compute_cdr path reached Rust")

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    with pytest.raises(KeyError, match="Field not available: missing"):
        simple_moment.compute_cdr(
            _radar(np.ones((1, 1)), np.ones((1, 1))),
            rhohv_field="missing",
            zdr_field="zdr",
            cdr_field="cdr",
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_compute_cdr_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    rhohv = np.array([[0.0, 0.5, 0.9, 1.0, -1.0]], dtype=np.float64)
    zdr = np.array([[0.0, 1.0, -1.0, 0.0, 0.0]], dtype=np.float64)
    expected = _fallback_cdr(rhohv.copy(), zdr.copy(), monkeypatch)
    calls = []

    def counted_kernel(rhohv_arg, zdr_arg):
        calls.append((rhohv_arg.shape, zdr_arg.shape))
        return rust._simple_moment_cdr_dense_f64(rhohv_arg, zdr_arg)

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_simple_moment_cdr_dense_f64" else None,
    )
    actual = _compute_cdr(rhohv.copy(), zdr.copy())

    assert calls == [((1, 5), (1, 5))]
    _assert_cdr_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_compute_cdr_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="same shape"):
        rust._simple_moment_cdr_dense_f64(
            np.ones((2, 3), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._simple_moment_cdr_dense_f64(
            np.ones((2, 6), dtype=np.float64)[:, ::2],
            np.ones((2, 6), dtype=np.float64)[:, ::2],
        )
    with pytest.raises(ValueError, match="finite"):
        rust._simple_moment_cdr_dense_f64(
            np.array([np.nan], dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dense CDR kernel range"):
        rust._simple_moment_cdr_dense_f64(
            np.ones(1, dtype=np.float64),
            np.array([4000.0], dtype=np.float64),
        )
