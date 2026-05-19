import os

import numpy as np
import pytest

from pyart.retrieve import cappi
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_cappi_height_index_f64"):
        pytest.skip("pyart._rust has no CAPPI height-index kernel")
    return rust


def _fallback_height_index(z_3d, height, monkeypatch):
    monkeypatch.setattr(cappi, "_rust_kernel", lambda _name: None)
    return cappi._cappi_height_index(z_3d, height)


@pytest.mark.parametrize(
    "height",
    [1000.0, np.nan],
)
def test_cappi_height_index_python_fallback_reference_cases(monkeypatch, height):
    z_3d = np.array(
        [
            [[0.0, 1000.0], [2000.0, np.nan]],
            [[1500.0, 500.0], [1000.0, 3000.0]],
            [[1000.0, 1000.0], [2500.0, 1000.0]],
        ],
        dtype=np.float64,
    )

    height_idx, selected_gate_z = _fallback_height_index(z_3d, height, monkeypatch)

    expected_idx = np.argmin(np.abs(z_3d - height), axis=0)
    expected_z = np.take_along_axis(z_3d, expected_idx[np.newaxis, :, :], axis=0).squeeze(axis=0)
    assert height_idx.dtype == np.int64
    assert selected_gate_z.dtype == np.float64
    assert_exact_equal(height_idx, expected_idx)
    assert_exact_equal(selected_gate_z, expected_z)


def test_cappi_height_index_dispatches_dense_float64_to_private_rust(monkeypatch):
    z_3d = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    idx = np.array([[1, 2], [0, 1]], dtype=np.int64)
    selected = np.array([[4.0, 9.0], [2.0, 7.0]], dtype=np.float64)
    calls = []

    def kernel(z_arg, height):
        calls.append((z_arg.dtype, z_arg.shape, height))
        return idx.copy(), selected.copy()

    monkeypatch.setattr(
        cappi,
        "_rust_kernel",
        lambda name: kernel if name == "_cappi_height_index_f64" else None,
    )

    actual_idx, actual_selected = cappi._cappi_height_index(z_3d, 5.0)

    assert calls == [(np.dtype(np.float64), (3, 2, 2), 5.0)]
    assert_exact_equal(actual_idx, idx)
    assert_exact_equal(actual_selected, selected)


def test_cappi_height_index_rust_runtime_error_keeps_python_path(monkeypatch):
    z_3d = np.arange(12, dtype=np.float64).reshape(3, 2, 2)

    def rust_kernel(name):
        if name != "_cappi_height_index_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(cappi, "_rust_kernel", rust_kernel)
    actual = cappi._cappi_height_index(z_3d, 5.0)
    expected = _fallback_height_index(z_3d, 5.0, monkeypatch)

    assert_exact_equal(actual, expected)


def test_cappi_height_index_oversized_volume_keeps_python_path(monkeypatch):
    z_3d = np.arange(12, dtype=np.float64).reshape(3, 2, 2)

    def rust_kernel(name):
        if name == "_cappi_height_index_f64":
            raise AssertionError("oversized CAPPI input used Rust kernel")
        return None

    monkeypatch.setattr(cappi, "_rust_kernel", rust_kernel)
    monkeypatch.setattr(cappi, "CAPPI_RUST_MAX_VOLUME_GATES", 3)

    actual = cappi._cappi_height_index(z_3d, 5.0)
    expected = _fallback_height_index(z_3d, 5.0, monkeypatch)

    assert_exact_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        lambda: (np.arange(12, dtype=np.float32).reshape(3, 2, 2), 5.0),
        lambda: (np.ma.array(np.arange(12, dtype=np.float64).reshape(3, 2, 2)), 5.0),
        lambda: (np.arange(24, dtype=np.float64).reshape(3, 2, 4)[:, :, ::2], 5.0),
        lambda: (np.arange(4, dtype=np.float64).reshape(2, 2), 5.0),
        lambda: (np.empty((0, 2, 2), dtype=np.float64), 5.0),
        lambda: (np.arange(12, dtype=np.float64).reshape(3, 2, 2), object()),
    ],
)
def test_cappi_height_index_unsupported_inputs_keep_python_path(monkeypatch, case):
    z_3d, height = case()

    def rust_kernel(name):
        if name == "_cappi_height_index_f64":
            raise AssertionError("unsupported CAPPI input used Rust kernel")
        return None

    monkeypatch.setattr(cappi, "_rust_kernel", rust_kernel)
    try:
        actual = cappi._cappi_height_index(z_3d, height)
    except Exception as actual_error:
        expected_z, expected_height = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_height_index(expected_z, expected_height, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected_z, expected_height = case()
        expected = _fallback_height_index(expected_z, expected_height, monkeypatch)
        assert_exact_equal(actual, expected)


def test_cappi_height_index_height_mask_reference(monkeypatch):
    z_3d = np.array([[[0.0, 1000.0]], [[2000.0, np.nan]]], dtype=np.float64)
    _, selected_gate_z = _fallback_height_index(z_3d, 1000.0, monkeypatch)

    height_mask = np.abs(selected_gate_z - 1000.0) > 0.0

    assert height_mask.tolist() == [[True, False]]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust CAPPI parity",
)
def test_cappi_height_index_real_rust_matches_python_fallback(monkeypatch):
    z_3d = np.array(
        [
            [[1000.0, 900.0, 10.0], [np.nan, 700.0, 400.0]],
            [[1000.0, np.nan, np.nan], [500.0, 800.0, np.nan]],
            [[2000.0, np.nan, 1000.0], [600.0, 900.0, 700.0]],
        ],
        dtype=np.float64,
    )
    expected = _fallback_height_index(z_3d, 1000.0, monkeypatch)
    monkeypatch.undo()

    actual = cappi._cappi_height_index(z_3d, 1000.0)

    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust CAPPI parity",
)
def test_cappi_height_index_real_rust_matches_python_for_nan_height(monkeypatch):
    z_3d = np.array(
        [
            [[1000.0, np.nan], [200.0, 300.0]],
            [[3000.0, np.nan], [400.0, 500.0]],
            [[2000.0, 500.0], [600.0, np.nan]],
        ],
        dtype=np.float64,
    )
    expected = _fallback_height_index(z_3d, np.nan, monkeypatch)
    monkeypatch.undo()

    actual = cappi._cappi_height_index(z_3d, np.nan)

    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust CAPPI checks",
)
def test_cappi_height_index_direct_rust_helper():
    rust = _rust_or_skip()
    z_3d = np.array([[[1000.0, 900.0]], [[1000.0, np.nan]], [[2000.0, np.nan]]])

    idx, selected = rust._cappi_height_index_f64(z_3d, 1000.0)

    assert idx.dtype == np.int64
    assert idx.tolist() == [[0, 1]]
    assert selected[0, 0] == 1000.0
    assert np.isnan(selected[0, 1])

    idx, selected = rust._cappi_height_index_f64(z_3d, np.nan)
    assert idx.tolist() == [[0, 0]]
    assert selected[0, 0] == 1000.0
    assert selected[0, 1] == 900.0

    with pytest.raises(ValueError):
        rust._cappi_height_index_f64(np.empty((0, 1, 1), dtype=np.float64), 1000.0)
