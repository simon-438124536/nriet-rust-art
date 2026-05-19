import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import simple_moment_calculations as simple_moment  # noqa: E402


def _radar(refl, noised_bz):
    return SimpleNamespace(
        fields={
            "refl": {"data": refl},
            "noise": {"data": noised_bz},
        }
    )


def _compute_snr(refl, noised_bz):
    radar = _radar(refl, noised_bz)
    return simple_moment.compute_snr(
        radar, refl_field="refl", noise_field="noise", snr_field="snr"
    )["data"]


def _fallback_snr(refl, noised_bz, monkeypatch):
    monkeypatch.setattr(simple_moment, "_rust_kernel", lambda _name: None)
    return _compute_snr(refl, noised_bz)


def _assert_exact_array(actual, expected):
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_compute_snr_python_fallback_preserves_mask_union(monkeypatch):
    refl = np.ma.array(
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
        mask=[[False, True, False], [False, False, False]],
        dtype=np.float64,
    )
    noised_bz = np.ma.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        mask=[[False, False, False], [False, True, False]],
        dtype=np.float64,
    )

    actual = _fallback_snr(refl, noised_bz, monkeypatch)
    expected = refl - noised_bz

    assert np.ma.isMaskedArray(actual)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    _assert_exact_array(actual.filled(np.nan), expected.filled(np.nan))


def test_compute_snr_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(refl, noised_bz):
        calls.append((refl.dtype, refl.shape, noised_bz.dtype, noised_bz.shape))
        return np.full(refl.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_simple_moment_snr_dense" else None,
    )
    refl = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    noised_bz = np.ones((2, 3), dtype=np.float64)

    actual = _compute_snr(refl, noised_bz)

    assert calls == [(np.float64, (2, 3), np.float64, (2, 3))]
    np.testing.assert_array_equal(actual, np.full((2, 3), 7.0, dtype=np.float64))


@pytest.mark.parametrize(
    ("refl", "noised_bz"),
    [
        (
            np.arange(6.0, dtype=np.float32).reshape(2, 3),
            np.ones((2, 3), dtype=np.float32),
        ),
        (
            np.arange(6.0, dtype=np.float64).reshape(2, 3),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ),
        (
            np.arange(12.0, dtype=np.float64).reshape(2, 6)[:, ::2],
            np.ones((2, 3), dtype=np.float64),
        ),
    ],
)
def test_compute_snr_keeps_python_path_for_unsupported_inputs(
    monkeypatch, refl, noised_bz
):
    expected = _fallback_snr(refl, noised_bz, monkeypatch)

    def fail_if_called(name):
        if name != "_simple_moment_snr_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported SNR input used Rust")

        return kernel

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)
    actual = _compute_snr(refl, noised_bz)

    assert actual.dtype == expected.dtype
    _assert_exact_array(actual, expected)


def test_compute_snr_missing_fields_raise_before_rust(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("missing-field SNR path reached Rust")

    monkeypatch.setattr(simple_moment, "_rust_kernel", fail_if_called)

    with pytest.raises(KeyError, match="Field not available: missing"):
        simple_moment.compute_snr(_radar(np.ones((1, 1)), np.ones((1, 1))), "missing", "noise")
    with pytest.raises(KeyError, match="Field not available: missing"):
        simple_moment.compute_snr(_radar(np.ones((1, 1)), np.ones((1, 1))), "refl", "missing")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_compute_snr_matches_python_fallback(monkeypatch):
    refl = np.array([[0.0, -0.0, np.nan], [5.5, -2.0, 9.0]], dtype=np.float64)
    noised_bz = np.array([[-0.0, 0.0, 1.0], [2.5, -2.0, np.nan]], dtype=np.float64)
    expected = _fallback_snr(refl, noised_bz, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        simple_moment,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = _compute_snr(refl, noised_bz)

    assert actual.dtype == expected.dtype == np.float64
    _assert_exact_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_snr_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="same shape"):
        rust._simple_moment_snr_dense(
            np.ones((2, 3), dtype=np.float64), np.ones((3,), dtype=np.float64)
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._simple_moment_snr_dense(
            np.ones((2, 6), dtype=np.float64)[:, ::2],
            np.ones((2, 3), dtype=np.float64),
        )
