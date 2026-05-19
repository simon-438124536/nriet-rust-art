import os

import numpy as np
import pytest

import pyart.io.uf_write as uf_write
from pyart.testing import make_empty_ppi_radar


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_uf_ray_num_to_sweep_num_i32"):
        pytest.skip("pyart._rust has no UF ray-map kernel")
    return rust


def _fallback_ray_map(radar, monkeypatch):
    monkeypatch.setattr(uf_write, "_rust_kernel", lambda _name: None)
    return uf_write.UFRayCreator._calc_ray_num_to_sweep_num(radar)


def _radar_with_sweeps(nrays, starts, ends):
    radar = make_empty_ppi_radar(1, 1, nrays)
    radar.sweep_start_ray_index["data"] = np.asarray(starts, dtype=np.int32)
    radar.sweep_end_ray_index["data"] = np.asarray(ends, dtype=np.int32)
    return radar


def test_uf_ray_map_python_fallback_reference(monkeypatch):
    radar = make_empty_ppi_radar(1, 3, 3)

    actual = _fallback_ray_map(radar, monkeypatch)

    assert actual.dtype == np.int32
    np.testing.assert_array_equal(
        actual, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    )


def test_uf_ray_map_python_fallback_keeps_gaps_and_overwrite_order(monkeypatch):
    radar = _radar_with_sweeps(7, [2, 0, 4], [3, 1, 5])

    actual = _fallback_ray_map(radar, monkeypatch)

    np.testing.assert_array_equal(
        actual, np.array([1, 1, 0, 0, 2, 2, 0], dtype=np.int32)
    )


def test_uf_ray_map_dispatches_valid_radar_to_private_rust(monkeypatch):
    radar = make_empty_ppi_radar(1, 2, 2)
    calls = []

    def kernel(nrays, starts, ends):
        calls.append((nrays, starts.dtype, starts.copy(), ends.copy()))
        return np.array([9, 8, 7, 6], dtype=np.int32)

    monkeypatch.setattr(
        uf_write,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_ray_num_to_sweep_num_i32" else None,
    )

    actual = uf_write.UFRayCreator._calc_ray_num_to_sweep_num(radar)

    assert len(calls) == 1
    assert calls[0][0] == 4
    assert calls[0][1] == np.dtype(np.int32)
    np.testing.assert_array_equal(calls[0][2], np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(calls[0][3], np.array([1, 3], dtype=np.int32))
    np.testing.assert_array_equal(actual, np.array([9, 8, 7, 6], dtype=np.int32))


class _CustomRadarLike:
    nrays = 5

    def iter_slice(self):
        return iter([slice(1, 3), slice(3, 5)])


def test_uf_ray_map_custom_radar_like_object_keeps_python_path(monkeypatch):
    def fail_if_called(name):
        if name == "_uf_ray_num_to_sweep_num_i32":
            raise AssertionError("custom radar-like object used Rust")
        return None

    monkeypatch.setattr(uf_write, "_rust_kernel", fail_if_called)

    actual = uf_write.UFRayCreator._calc_ray_num_to_sweep_num(_CustomRadarLike())

    np.testing.assert_array_equal(actual, np.array([0, 0, 0, 1, 1], dtype=np.int32))


@pytest.mark.parametrize(
    ("starts", "ends"),
    [
        ([-1], [2]),
        ([3], [99]),
        ([3], [2]),
        ([0, 2], [1]),
    ],
)
def test_uf_ray_map_invalid_dense_ranges_keep_python_path(monkeypatch, starts, ends):
    radar = _radar_with_sweeps(5, starts, ends)

    def kernel(_nrays, _starts, _ends):
        return np.full((5,), 7, dtype=np.int32)

    monkeypatch.setattr(
        uf_write,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_ray_num_to_sweep_num_i32" else None,
    )

    actual = uf_write.UFRayCreator._calc_ray_num_to_sweep_num(radar)
    expected = _fallback_ray_map(_radar_with_sweeps(5, starts, ends), monkeypatch)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kernel_result",
    [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    ],
)
def test_uf_ray_map_bad_rust_output_keeps_python_path(monkeypatch, kernel_result):
    radar = _radar_with_sweeps(3, [0], [2])

    def kernel(_nrays, _starts, _ends):
        return kernel_result

    monkeypatch.setattr(
        uf_write,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_ray_num_to_sweep_num_i32" else None,
    )

    actual = uf_write.UFRayCreator._calc_ray_num_to_sweep_num(radar)
    expected = _fallback_ray_map(_radar_with_sweeps(3, [0], [2]), monkeypatch)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust UF ray-map parity",
)
def test_uf_ray_map_real_rust_matches_python_fallback(monkeypatch):
    radar = make_empty_ppi_radar(1, 3, 3)
    expected = _fallback_ray_map(radar, monkeypatch)
    monkeypatch.undo()

    actual = uf_write.UFRayCreator._calc_ray_num_to_sweep_num(radar)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust UF ray-map checks",
)
def test_uf_ray_map_direct_rust_helper():
    rust = _rust_or_skip()

    actual = rust._uf_ray_num_to_sweep_num_i32(
        7,
        np.array([2, 0, 4], dtype=np.int32),
        np.array([3, 1, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        actual, np.array([1, 1, 0, 0, 2, 2, 0], dtype=np.int32)
    )

    with pytest.raises(ValueError, match="outside nrays"):
        rust._uf_ray_num_to_sweep_num_i32(
            3, np.array([0], dtype=np.int32), np.array([3], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="non-negative"):
        rust._uf_ray_num_to_sweep_num_i32(
            3, np.array([-1], dtype=np.int32), np.array([1], dtype=np.int32)
        )
    with pytest.raises(ValueError, match=">= start"):
        rust._uf_ray_num_to_sweep_num_i32(
            3, np.array([2], dtype=np.int32), np.array([1], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._uf_ray_num_to_sweep_num_i32(
            3, np.arange(4, dtype=np.int32)[::2], np.array([1, 2], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="size limit"):
        rust._uf_ray_num_to_sweep_num_i32(
            uf_write.UF_RAY_MAP_RUST_MAX_RAYS + 1,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
