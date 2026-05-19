import copy
import os

import numpy as np
import pytest

from pyart.retrieve import advection
from pyart.testing import make_empty_grid


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_grid_displacement_peak_2d_f64"):
        pytest.skip("pyart._rust has no advection peak kernel")
    return rust


def _fallback_peak(data, monkeypatch):
    monkeypatch.setattr(advection, "_rust_kernel", lambda _name: None)
    return advection._grid_displacement_peak(data)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[4.0, 4.0], [1.0, 4.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 2.0], [9.0, 3.0, 4.0]], dtype=np.float64),
        np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
    ],
)
def test_grid_displacement_peak_python_fallback_reference_cases(monkeypatch, data):
    yshift, xshift = _fallback_peak(data, monkeypatch)

    row, col = data.shape
    expected_y, expected_x = np.unravel_index(np.argmax(data), data.shape)
    expected_y -= int(row / 2)
    expected_x -= int(col / 2)
    assert isinstance(yshift, np.integer)
    assert isinstance(xshift, np.integer)
    assert (yshift, xshift) == (expected_y, expected_x)


def test_grid_displacement_peak_dispatches_dense_float64_to_private_rust(monkeypatch):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    calls = []

    def kernel(data_arg):
        calls.append((data_arg.dtype, data_arg.shape))
        return 5, -3

    monkeypatch.setattr(
        advection,
        "_rust_kernel",
        lambda name: kernel if name == "_grid_displacement_peak_2d_f64" else None,
    )

    yshift, xshift = advection._grid_displacement_peak(data)

    assert calls == [(np.dtype(np.float64), (2, 2))]
    assert (type(yshift), type(xshift)) == (np.int64, np.int64)
    assert (yshift, xshift) == (np.int64(5), np.int64(-3))


def test_grid_displacement_peak_rust_runtime_error_keeps_python_path(monkeypatch):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    def rust_kernel(name):
        if name != "_grid_displacement_peak_2d_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(advection, "_rust_kernel", rust_kernel)
    actual = advection._grid_displacement_peak(data)
    expected = _fallback_peak(data, monkeypatch)

    assert actual == expected


@pytest.mark.parametrize(
    "case",
    [
        lambda: np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float64),
        lambda: np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float64),
        lambda: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        lambda: np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::2],
        lambda: np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]]),
        lambda: np.array([1.0, 2.0], dtype=np.float64),
        lambda: np.empty((0, 2), dtype=np.float64),
    ],
)
def test_grid_displacement_peak_unsupported_inputs_keep_python_path(monkeypatch, case):
    data = case()

    def rust_kernel(name):
        if name == "_grid_displacement_peak_2d_f64":
            raise AssertionError("unsupported advection peak input used Rust kernel")
        return None

    monkeypatch.setattr(advection, "_rust_kernel", rust_kernel)
    try:
        actual = advection._grid_displacement_peak(data)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_peak(case(), monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_peak(case(), monkeypatch)
        assert actual == expected


def test_grid_displacement_pc_uses_private_peak_helper(monkeypatch):
    grid1 = make_empty_grid((1, 4, 4), ((0.0, 0.0), (0.0, 30.0), (0.0, 60.0)))
    grid2 = copy.deepcopy(grid1)
    data = np.ma.array(np.arange(16, dtype=np.float64).reshape(1, 4, 4))
    grid1.fields["reflectivity"] = {"data": data}
    grid2.fields["reflectivity"] = {"data": data.copy()}

    monkeypatch.setattr(advection, "_grid_displacement_peak", lambda _data: (np.int64(1), np.int64(-2)))

    assert advection.grid_displacement_pc(grid1, grid2, "reflectivity", 0) == (
        np.int64(1),
        np.int64(-2),
    )
    assert advection.grid_displacement_pc(grid1, grid2, "reflectivity", 0, "distance") == (
        np.float64(10.0),
        np.float64(-40.0),
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust advection peak parity",
)
def test_grid_displacement_peak_real_rust_matches_python_fallback(monkeypatch):
    data = np.array([[1.0, 2.0, 2.0], [4.0, 4.0, 3.0]], dtype=np.float64)
    expected = _fallback_peak(data, monkeypatch)
    monkeypatch.undo()

    actual = advection._grid_displacement_peak(data)

    assert actual == expected
    assert (type(actual[0]), type(actual[1])) == (np.int64, np.int64)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust advection peak checks",
)
def test_grid_displacement_peak_direct_rust_helper():
    rust = _rust_or_skip()
    data = np.array([[1.0, 4.0], [4.0, 2.0]], dtype=np.float64)

    assert rust._grid_displacement_peak_2d_f64(data) == (-1, 0)
    with pytest.raises(ValueError):
        rust._grid_displacement_peak_2d_f64(np.array([[np.nan]], dtype=np.float64))
