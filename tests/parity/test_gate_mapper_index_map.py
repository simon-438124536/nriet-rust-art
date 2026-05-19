"""Parity for GateMapper KDTree post-processing (index_map build)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pyart.map.gate_mapper import (
    _build_index_map_rust,
    _can_use_rust_index_map_build,
)


def _rust_build_index_map():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.skip("pyart._rust is not installed")
    import pyart._rust as rust

    fn = getattr(rust, "_gate_mapper_build_index_map_f64", None)
    if fn is None:
        pytest.skip("pyart._rust has not registered _gate_mapper_build_index_map_f64")
    return fn


def _python_index_map(
    dists,
    inds,
    src_radar_time,
    dest_radar_time,
    distance_tolerance,
    time_tolerance,
):
    index_map = np.full((*dists.shape, 2), np.nan, dtype=np.float64)
    tree_len = len(dest_radar_time)
    inds = inds.copy()
    inds[inds == tree_len] = inds[inds == tree_len] - 1
    times = np.abs(src_radar_time - dest_radar_time[inds])
    inds = np.where(
        np.logical_and(
            times < time_tolerance, np.abs(dists) < distance_tolerance
        ),
        inds[:],
        -32767,
    ).astype(np.int64)
    dest_ngates = dest_radar_time.shape[0]
    index_map[:, :, 0] = (inds / dest_ngates).astype(int)
    index_map[:, :, 1] = inds - dest_ngates * (inds / dest_ngates).astype(int)
    return index_map


class _MapperStub:
    def __init__(self, src_time, dest_time, index_map):
        self.src_radar_time = src_time
        self.dest_radar_time = dest_time
        self._index_map = index_map
        self._time_tolerance = 2.5


def test_can_use_rust_index_map_build_rejects_non_int64_inds():
    dists = np.zeros((2, 3), dtype=np.float64)
    inds = np.zeros((2, 3), dtype=np.int32)
    src_time = np.zeros((2, 3), dtype=np.float64)
    dest_time = np.ones(10, dtype=np.float64)
    index_map = np.zeros((2, 3, 2), dtype=np.float64)
    assert not _can_use_rust_index_map_build(dists, inds, src_time, dest_time, index_map)


def test_rust_index_map_matches_python_oracle():
    rust_kernel = _rust_build_index_map()
    rng = np.random.default_rng(0)
    shape = (4, 5)
    dists = rng.uniform(0.0, 50.0, size=shape)
    inds = rng.integers(0, 12, size=shape, dtype=np.int64)
    src_time = rng.uniform(0.0, 100.0, size=shape)
    dest_time = rng.uniform(0.0, 100.0, size=12)
    distance_tolerance = 40.0
    time_tolerance = 3.0

    expected = _python_index_map(
        dists,
        inds,
        src_time,
        dest_time,
        distance_tolerance,
        time_tolerance,
    )

    index_map = np.full((*shape, 2), np.nan, dtype=np.float64)
    mapper = _MapperStub(src_time, dest_time, index_map)
    assert _build_index_map_rust(mapper, dists, inds, distance_tolerance)
    np.testing.assert_array_equal(mapper._index_map, expected)

    index_map_direct = np.full((*shape, 2), np.nan, dtype=np.float64)
    rust_kernel(
        dists,
        inds,
        src_time,
        dest_time,
        index_map_direct,
        distance_tolerance,
        time_tolerance,
        len(dest_time),
    )
    np.testing.assert_array_equal(index_map_direct, expected)


def test_rust_index_map_tree_len_sentinel_matches_python():
    rust_kernel = _rust_build_index_map()
    dists = np.array([[10.0, 5.0]], dtype=np.float64)
    inds = np.array([[5, 5]], dtype=np.int64)
    src_time = np.array([[1.0, 2.0]], dtype=np.float64)
    dest_time = np.arange(5, dtype=np.float64)
    distance_tolerance = 100.0
    time_tolerance = 100.0

    expected = _python_index_map(
        dists,
        inds,
        src_time,
        dest_time,
        distance_tolerance,
        time_tolerance,
    )
    index_map = np.full((1, 2, 2), np.nan, dtype=np.float64)
    rust_kernel(
        dists,
        inds,
        src_time,
        dest_time,
        index_map,
        distance_tolerance,
        time_tolerance,
        len(dest_time),
    )
    np.testing.assert_array_equal(index_map, expected)
