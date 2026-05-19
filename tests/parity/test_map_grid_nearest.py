"""Parity for map_to_grid NEAREST row selection."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pyart.map.grid_mapper import _select_nearest_grid_value


def _rust_select_nearest():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.skip("pyart._rust is not installed")
    import pyart._rust as rust

    fn = getattr(rust, "_map_grid_select_nearest_row_f64", None)
    if fn is None:
        pytest.skip("pyart._rust has not registered _map_grid_select_nearest_row_f64")
    return fn


def test_select_nearest_grid_value_matches_argmin():
    rng = np.random.default_rng(1)
    dist2 = rng.uniform(0.0, 100.0, size=8)
    nn_field_data = rng.uniform(-10.0, 10.0, size=(8, 3))
    expected = nn_field_data[np.argmin(dist2)]
    actual = _select_nearest_grid_value(dist2, nn_field_data)
    np.testing.assert_array_equal(actual, expected)


def test_rust_nearest_row_matches_numpy_argmin_on_ties():
    rust_kernel = _rust_select_nearest()
    dist2 = np.array([5.0, 2.0, 2.0, 9.0], dtype=np.float64)
    nn_field_data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [9.0, 8.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = np.empty(2, dtype=np.float64)
    rust_kernel(dist2, nn_field_data, out)
    np.testing.assert_array_equal(out, nn_field_data[np.argmin(dist2)])
