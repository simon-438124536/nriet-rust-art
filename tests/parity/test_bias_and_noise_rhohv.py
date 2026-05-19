import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import bias_and_noise  # noqa: E402


class _Radar(SimpleNamespace):
    pass


def _radar(urhohv, snr, zdr, nh, nv):
    return _Radar(
        fields={
            "urhohv": {"data": urhohv},
            "snr": {"data": snr},
            "zdr": {"data": zdr},
            "nh": {"data": nh},
            "nv": {"data": nv},
        }
    )


def _compute_correct_noise_rhohv(urhohv, snr, zdr, nh, nv):
    return bias_and_noise.correct_noise_rhohv(
        _radar(urhohv, snr, zdr, nh, nv),
        urhohv_field="urhohv",
        snr_field="snr",
        zdr_field="zdr",
        nh_field="nh",
        nv_field="nv",
        rhohv_field="rhohv",
    )["data"]


def _fallback_correct_noise_rhohv(urhohv, snr, zdr, nh, nv, monkeypatch):
    monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
    return _compute_correct_noise_rhohv(urhohv, snr, zdr, nh, nv)


def _assert_rhohv_close(actual, expected):
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


def _dense_inputs():
    urhohv = np.array([[0.1, 0.5, 0.95], [-0.2, 1.1, 0.7]], dtype=np.float64)
    snr = np.array([[10.0, 20.0, 35.0], [15.0, 25.0, 30.0]], dtype=np.float64)
    zdr = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float64)
    nh = np.array([[-45.0, -43.0, -40.0], [-42.0, -41.0, -39.0]], dtype=np.float64)
    nv = np.array([[-46.0, -44.0, -41.0], [-43.0, -40.0, -38.0]], dtype=np.float64)
    return urhohv, snr, zdr, nh, nv


def test_correct_noise_rhohv_python_fallback_reference(monkeypatch):
    inputs = _dense_inputs()

    actual = _fallback_correct_noise_rhohv(*inputs, monkeypatch)

    assert actual.dtype == np.float64
    assert np.ma.isMaskedArray(actual)
    assert np.max(actual) <= 1.0


def test_correct_noise_rhohv_dispatches_to_private_rust_kernel(monkeypatch):
    inputs = _dense_inputs()
    calls = []

    def rust_kernel(*args):
        calls.append(tuple((arg.dtype, arg.shape) for arg in args))
        return np.full(args[0].shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_correct_noise_rhohv_dense_f64" else None,
    )

    actual = _compute_correct_noise_rhohv(*inputs)

    assert calls == [tuple((np.dtype("float64"), inputs[0].shape) for _ in inputs)]
    expected = np.ma.array(np.full(inputs[0].shape, 7.0, dtype=np.float64))
    _assert_rhohv_close(actual, expected)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda arrays: (arrays[0].astype(np.float32), *arrays[1:]),
        lambda arrays: (arrays[0][:, ::2], *(array[:, ::2] for array in arrays[1:])),
        lambda arrays: (np.ma.array(arrays[0], mask=np.zeros(arrays[0].shape)), *arrays[1:]),
        lambda arrays: (np.array([[np.nan]], dtype=np.float64), *(array[:1, :1] for array in arrays[1:])),
        lambda arrays: (np.array([[np.inf]], dtype=np.float64), *(array[:1, :1] for array in arrays[1:])),
        lambda arrays: (arrays[0], arrays[1].astype(np.float32), *arrays[2:]),
        lambda arrays: (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4][:1, :1]),
        lambda arrays: (np.array([[1.0e7]], dtype=np.float64), *(array[:1, :1] for array in arrays[1:])),
        lambda arrays: (arrays[0], np.array([[301.0]], dtype=np.float64), *(array[:1, :1] for array in arrays[2:])),
        lambda arrays: (
            np.array([[1.0]], dtype=np.float64),
            np.array([[-1000.0]], dtype=np.float64),
            np.array([[1000.0]], dtype=np.float64),
            np.array([[-1000.0]], dtype=np.float64),
            np.array([[1000.0]], dtype=np.float64),
        ),
    ],
)
def test_correct_noise_rhohv_keeps_python_path_for_unsupported_inputs(
    monkeypatch, mutate
):
    inputs = mutate(_dense_inputs())
    with np.errstate(all="ignore"):
        expected = _fallback_correct_noise_rhohv(*inputs, monkeypatch)

    def fail_if_called(name):
        if name != "_correct_noise_rhohv_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported correct_noise_rhohv input used Rust")

        return kernel

    monkeypatch.setattr(bias_and_noise, "_rust_kernel", fail_if_called)
    with np.errstate(all="ignore"):
        try:
            actual = _compute_correct_noise_rhohv(*inputs)
        except Exception as actual_error:
            monkeypatch.setattr(bias_and_noise, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)) as expected_error:
                _compute_correct_noise_rhohv(*inputs)
            assert actual_error.args == expected_error.value.args
        else:
            _assert_rhohv_close(actual, expected)


def test_correct_noise_rhohv_missing_field_raises_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field correct_noise_rhohv path reached Rust")

    urhohv, snr, zdr, nh, nv = _dense_inputs()
    radar = _radar(urhohv, snr, zdr, nh, nv)
    del radar.fields["snr"]
    monkeypatch.setattr(bias_and_noise, "_rust_kernel", fail_if_called)

    with pytest.raises(KeyError, match="Field not available: snr"):
        bias_and_noise.correct_noise_rhohv(
            radar,
            urhohv_field="urhohv",
            snr_field="snr",
            zdr_field="zdr",
            nh_field="nh",
            nv_field="nv",
            rhohv_field="rhohv",
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust wrapper dispatch is verified in installed-wheel mode",
)
def test_real_rust_correct_noise_rhohv_overflow_risk_stays_python_owned(
    monkeypatch,
):
    inputs = (
        np.array([[1.0]], dtype=np.float64),
        np.array([[-1000.0]], dtype=np.float64),
        np.array([[1000.0]], dtype=np.float64),
        np.array([[-1000.0]], dtype=np.float64),
        np.array([[1000.0]], dtype=np.float64),
    )

    def fail_if_called(name):
        if name != "_correct_noise_rhohv_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("overflow-risk correct_noise_rhohv input used Rust")

        return kernel

    monkeypatch.setattr(bias_and_noise, "_rust_kernel", fail_if_called)
    with np.errstate(all="ignore"):
        actual = _compute_correct_noise_rhohv(*inputs)

    expected = _fallback_correct_noise_rhohv(*inputs, monkeypatch)
    _assert_rhohv_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_correct_noise_rhohv_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    inputs = tuple(array.copy() for array in _dense_inputs())
    expected = _fallback_correct_noise_rhohv(
        *(array.copy() for array in inputs), monkeypatch
    )
    calls = []

    def counted_kernel(*args):
        calls.append(tuple(arg.shape for arg in args))
        return rust._correct_noise_rhohv_dense_f64(*args)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_correct_noise_rhohv_dense_f64" else None,
    )
    actual = _compute_correct_noise_rhohv(*(array.copy() for array in inputs))

    assert calls == [tuple(array.shape for array in inputs)]
    _assert_rhohv_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_correct_noise_rhohv_guard_boundaries_match_python(monkeypatch):
    import pyart._rust as rust

    inputs = (
        np.array([[1.0e6, -1.0e6], [1.0, 0.2]], dtype=np.float64),
        np.array([[-300.0, 300.0], [0.0, 100.0]], dtype=np.float64),
        np.array([[300.0, -300.0], [300.0, 0.0]], dtype=np.float64),
        np.array([[150.0, -150.0], [300.0, 50.0]], dtype=np.float64),
        np.array([[-150.0, 150.0], [0.0, -250.0]], dtype=np.float64),
    )
    expected = _fallback_correct_noise_rhohv(
        *(array.copy() for array in inputs), monkeypatch
    )
    calls = []

    def counted_kernel(*args):
        calls.append(tuple(arg.shape for arg in args))
        return rust._correct_noise_rhohv_dense_f64(*args)

    monkeypatch.setattr(
        bias_and_noise,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_correct_noise_rhohv_dense_f64" else None,
    )
    actual = _compute_correct_noise_rhohv(*(array.copy() for array in inputs))

    assert calls == [tuple(array.shape for array in inputs)]
    _assert_rhohv_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_correct_noise_rhohv_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    inputs = _dense_inputs()
    with pytest.raises(ValueError, match="same shape"):
        rust._correct_noise_rhohv_dense_f64(
            inputs[0], inputs[1][:1, :1], inputs[2], inputs[3], inputs[4]
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._correct_noise_rhohv_dense_f64(*(array[:, ::2] for array in inputs))
    with pytest.raises(ValueError, match="finite"):
        rust._correct_noise_rhohv_dense_f64(
            np.array([[np.nan]], dtype=np.float64),
            *(array[:1, :1] for array in inputs[1:]),
        )
    with pytest.raises(ValueError, match="dense rhohv-noise kernel range"):
        rust._correct_noise_rhohv_dense_f64(
            np.array([[1.0e7]], dtype=np.float64),
            *(array[:1, :1] for array in inputs[1:]),
        )
    with pytest.raises(ValueError, match="dense rhohv-noise kernel range"):
        rust._correct_noise_rhohv_dense_f64(
            np.array([[1.0]], dtype=np.float64),
            np.array([[301.0]], dtype=np.float64),
            *(array[:1, :1] for array in inputs[2:]),
        )
    with pytest.raises(ValueError, match="nh and nv difference"):
        rust._correct_noise_rhohv_dense_f64(
            np.array([[1.0]], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
            np.array([[-200.0]], dtype=np.float64),
            np.array([[200.0]], dtype=np.float64),
        )
