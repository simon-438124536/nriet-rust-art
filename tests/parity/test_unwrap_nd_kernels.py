import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import _unwrap_2d, _unwrap_3d  # noqa: E402


def _wrapped(values):
    return ((np.asarray(values, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi


def test_unwrap_2d_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_unwrap(image, mask, out, wrap_around):
        calls.append((image.dtype, mask.dtype, out.dtype, image.shape, list(wrap_around)))
        out[:] = image + 10.0

    monkeypatch.setattr(_unwrap_2d, "_rust_kernel", lambda: rust_unwrap)
    image = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    mask = np.zeros_like(image, dtype=np.uint8)
    out = np.zeros_like(image)

    result = _unwrap_2d.unwrap_2d(image, mask, out, [True, False])

    assert result is None
    assert calls == [(np.float64, np.uint8, np.float64, (2, 3), [True, False])]
    np.testing.assert_array_equal(out, image + 10.0)


def test_unwrap_3d_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_unwrap(image, mask, out, wrap_around):
        calls.append((image.dtype, mask.dtype, out.dtype, image.shape, list(wrap_around)))
        out[:] = image - 10.0

    monkeypatch.setattr(_unwrap_3d, "_rust_kernel", lambda: rust_unwrap)
    image = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    mask = np.zeros_like(image, dtype=np.uint8)
    out = np.zeros_like(image)

    result = _unwrap_3d.unwrap_3d(image, mask, out, [False, True, False])

    assert result is None
    assert calls == [(np.float64, np.uint8, np.float64, (2, 3, 4), [False, True, False])]
    np.testing.assert_array_equal(out, image - 10.0)


def test_unwrap_2d_without_backend_preserves_bootstrap_error(monkeypatch):
    monkeypatch.setattr(_unwrap_2d, "_rust_kernel", lambda: None)
    image = np.zeros((2, 2), dtype=np.float64)
    mask = np.zeros((2, 2), dtype=np.uint8)
    out = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(NotImplementedError, match="requires the C/Rust phase"):
        _unwrap_2d.unwrap_2d(image, mask, out, [False, False])


def test_unwrap_3d_without_backend_preserves_bootstrap_error(monkeypatch):
    monkeypatch.setattr(_unwrap_3d, "_rust_kernel", lambda: None)
    image = np.zeros((2, 2, 2), dtype=np.float64)
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    out = np.zeros((2, 2, 2), dtype=np.float64)

    with pytest.raises(NotImplementedError, match="requires the C/Rust phase"):
        _unwrap_3d.unwrap_3d(image, mask, out, [False, False, False])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_unwrap_2d_recovers_monotonic_phase_and_masks_minimum():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5, 4.5, 5.5],
        ],
        dtype=np.float64,
    )
    image = _wrapped(expected)
    mask = np.zeros(expected.shape, dtype=np.uint8)
    mask[0, 0] = 1
    out = np.full(expected.shape, -999.0, dtype=np.float64)

    _unwrap_2d.unwrap_2d(image, mask, out, [False, False])

    expected_with_mask = expected.copy()
    expected_with_mask[0, 0] = expected_with_mask[mask == 0].min()
    np.testing.assert_allclose(out, expected_with_mask, rtol=0.0, atol=1.0e-14)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_unwrap_3d_recovers_monotonic_phase_and_masks_minimum():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    z, y, x = np.indices((3, 4, 5), dtype=np.float64)
    expected = 0.25 + 0.5 * z + 0.75 * y + 0.9 * x
    image = _wrapped(expected)
    mask = np.zeros(expected.shape, dtype=np.uint8)
    mask[0, 0, 0] = 1
    out = np.full(expected.shape, -999.0, dtype=np.float64)

    _unwrap_3d.unwrap_3d(image, mask, out, [False, False, False])

    # The LJMU 3D oracle preserves relative phase but may choose a global 2pi offset.
    expected_with_mask = expected - 2 * np.pi
    expected_with_mask[0, 0, 0] = expected_with_mask[mask == 0].min()
    np.testing.assert_allclose(out, expected_with_mask, rtol=0.0, atol=1.0e-14)
