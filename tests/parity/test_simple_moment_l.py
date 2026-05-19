import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import simple_moment_calculations as simple_moment  # noqa: E402


def _radar(rhohv):
    return SimpleNamespace(fields={"rhohv": {"data": rhohv}})


def _compute_l(rhohv):
    return simple_moment.compute_l(
        _radar(rhohv),
        rhohv_field="rhohv",
        l_field="l",
    )["data"]


def _fallback_l(rhohv, monkeypatch):
    monkeypatch.setattr(simple_moment, "_rust_kernel", lambda _name: None)
    return _compute_l(rhohv)


def _assert_l_close(actual, expected):
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
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(np.signbit(actual.data), np.signbit(expected.data))


def _copy_preserving_noncontiguous_case(rhohv):
    if (
        type(rhohv) is np.ndarray
        and rhohv.dtype == np.float64
        and rhohv.shape == (1, 2)
        and not rhohv.flags.c_contiguous
    ):
        return np.array([[rhohv[0, 0], 0.3, rhohv[0, 1], 0.5]], dtype=np.float64)[
            :, ::2
        ]
    return rhohv.copy()


def test_compute_l_python_fallback_clamps_input_in_place(monkeypatch):
    rhohv = np.array([[0.0, 0.5, 0.999, 1.0, 1.2]], dtype=np.float64)

    actual = _fallback_l(rhohv, monkeypatch)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        rhohv,
        np.array([[0.0, 0.5, 0.999, 0.9999, 0.9999]], dtype=np.float64),
    )


def test_compute_l_dispatches_to_private_rust_kernel(monkeypatch):
    rhohv = np.array([[0.2, 1.2]], dtype=np.float64)
    calls = []

    def rust_kernel(rhohv_arg):
        calls.append((rhohv_arg.dtype, rhohv_arg.shape, rhohv_arg.flags.writeable))
        rhohv_arg[rhohv_arg >= 1.0] = 0.9999
        return (
            np.full(rhohv_arg.shape, 7.0, dtype=np.float64),
            np.array([[False, True]], dtype=bool),
        )

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_simple_moment_l_dense_f64" else None,
    )

    actual = _compute_l(rhohv)

    assert calls == [(np.dtype("float64"), (1, 2), True)]
    np.testing.assert_array_equal(rhohv, np.array([[0.2, 0.9999]], dtype=np.float64))
    expected = np.ma.array([[7.0, 7.0]], mask=[[False, True]])
    _assert_l_close(actual, expected)


@pytest.mark.parametrize(
    "rhohv",
    [
        np.array([[0.2, 0.3]], dtype=np.float32),
        np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float64)[:, ::2],
        np.ma.array([[0.2, 0.3]], mask=[[False, True]], dtype=np.float64),
        np.array([[np.nan, 0.3]], dtype=np.float64),
        np.array([[np.inf, 0.3]], dtype=np.float64),
        np.array([[-1.0e301, 0.3]], dtype=np.float64),
    ],
)
def test_compute_l_keeps_python_path_for_unsupported_inputs(monkeypatch, rhohv):
    expected_input = _copy_preserving_noncontiguous_case(rhohv)
    expected = _fallback_l(expected_input, monkeypatch)

    def fail_if_called(name):
        if name != "_simple_moment_l_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported compute_l input used Rust")

        return kernel

    actual_input = _copy_preserving_noncontiguous_case(rhohv)
    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    with np.errstate(all="ignore"):
        actual = _compute_l(actual_input)

    _assert_l_close(actual, expected)
    np.testing.assert_array_equal(actual_input, expected_input)


def test_compute_l_missing_field_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field compute_l path reached Rust")

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    with pytest.raises(KeyError, match="Field not available: missing"):
        simple_moment.compute_l(
            _radar(np.ones((1, 1))),
            rhohv_field="missing",
            l_field="l",
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_compute_l_matches_python_fallback_and_clamp(monkeypatch):
    import pyart._rust as rust

    rhohv = np.array([[0.0, 0.5, 0.999, 1.0, 1.2]], dtype=np.float64)
    expected_input = rhohv.copy()
    expected = _fallback_l(expected_input, monkeypatch)
    calls = []

    def counted_kernel(rhohv_arg):
        calls.append((rhohv_arg.shape, rhohv_arg.copy()))
        return rust._simple_moment_l_dense_f64(rhohv_arg)

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_simple_moment_l_dense_f64" else None,
    )
    actual = _compute_l(rhohv)

    assert len(calls) == 1
    assert calls[0][0] == (1, 5)
    np.testing.assert_array_equal(
        calls[0][1], np.array([[0.0, 0.5, 0.999, 1.0, 1.2]])
    )
    np.testing.assert_array_equal(rhohv, expected_input)
    _assert_l_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_compute_l_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._simple_moment_l_dense_f64(
            np.ones((2, 6), dtype=np.float64)[:, ::2]
        )
    with pytest.raises(ValueError, match="finite"):
        rust._simple_moment_l_dense_f64(np.array([np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="dense L kernel range"):
        rust._simple_moment_l_dense_f64(np.array([-1.0e301], dtype=np.float64))
