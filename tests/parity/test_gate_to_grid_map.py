import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.map import _gate_to_grid_map as gate_to_grid_map  # noqa: E402


def _mapper(dtype=np.float32):
    grid_shape = (2, 2, 3)
    grid_sum = np.zeros(grid_shape + (2,), dtype=dtype)
    grid_wsum = np.zeros(grid_shape + (2,), dtype=dtype)
    return gate_to_grid_map.GateToGridMapper(
        grid_shape,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        grid_sum,
        grid_wsum,
    )


def _offset_mapper():
    grid_shape = (2, 2, 3)
    grid_sum = np.zeros(grid_shape + (2,), dtype=np.float32)
    grid_wsum = np.zeros(grid_shape + (2,), dtype=np.float32)
    return gate_to_grid_map.GateToGridMapper(
        grid_shape,
        (0.5, -1.25, 2.0),
        (1.25, 0.5, -0.75),
        grid_sum,
        grid_wsum,
    )


def _fallback_roi(mapper, roi_func, monkeypatch):
    roi = np.empty((mapper.nz, mapper.ny, mapper.nx), dtype=np.float32)
    monkeypatch.setattr(gate_to_grid_map, "_rust_kernel", lambda _name: None)
    mapper.find_roi_for_grid(roi, roi_func)
    return roi


def test_find_bounds_dispatch_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_find_min(a, roi, step):
        calls.append(("min", a, roi, step))
        return 4

    def rust_find_max(a, roi, step, na):
        calls.append(("max", a, roi, step, na))
        return 7

    kernels = {
        "_gate_to_grid_find_min": rust_find_min,
        "_gate_to_grid_find_max": rust_find_max,
    }
    monkeypatch.setattr(gate_to_grid_map, "_rust_kernel", kernels.get)

    assert gate_to_grid_map.find_min(3.0, 1.0, 0.5) == 4
    assert gate_to_grid_map.find_max(3.0, 1.0, 0.5, 10) == 7
    assert calls == [
        ("min", 3.0, 1.0, 0.5),
        ("max", 3.0, 1.0, 0.5, 10),
    ]


def test_find_bounds_python_fallback_matches_oracle_clamping(monkeypatch):
    monkeypatch.setattr(gate_to_grid_map, "_rust_kernel", lambda _name: None)

    assert gate_to_grid_map.find_min(2.4, 1.0, 1.0) == 2
    assert gate_to_grid_map.find_min(-2.4, 1.0, 1.0) == 0
    assert gate_to_grid_map.find_min(2.4, 1.0, 0.0) == 0
    assert gate_to_grid_map.find_max(2.4, 1.0, 1.0, 3) == 2
    assert gate_to_grid_map.find_max(-2.4, 1.0, 1.0, 3) == -2
    assert gate_to_grid_map.find_max(2.4, 1.0, 0.0, 3) == 0


def test_find_roi_for_grid_dispatches_constant_to_private_rust(monkeypatch):
    mapper = _mapper()
    roi = np.empty((mapper.nz, mapper.ny, mapper.nx), dtype=np.float32)
    calls = []

    def rust_constant(roi_arg, constant_roi):
        calls.append((roi_arg.shape, roi_arg.dtype, constant_roi))
        roi_arg.fill(np.float32(constant_roi))

    monkeypatch.setattr(
        gate_to_grid_map,
        "_rust_kernel",
        lambda name: rust_constant
        if name == "_gate_to_grid_roi_constant_f32"
        else None,
    )

    mapper.find_roi_for_grid(roi, gate_to_grid_map.ConstantRoI(7.25))

    assert calls == [((2, 2, 3), np.dtype("float32"), np.float32(7.25))]
    np.testing.assert_array_equal(roi, np.full((2, 2, 3), 7.25, dtype=np.float32))


def test_find_roi_for_grid_dispatches_dist_to_private_rust(monkeypatch):
    mapper = _offset_mapper()
    roi = np.empty((mapper.nz, mapper.ny, mapper.nx), dtype=np.float32)
    calls = []

    def rust_dist(
        roi_arg,
        offsets,
        z_start,
        y_start,
        x_start,
        z_step,
        y_step,
        x_step,
        z_factor,
        xy_factor,
        min_radius,
    ):
        calls.append(
            (
                roi_arg.shape,
                offsets.copy(),
                z_start,
                y_start,
                x_start,
                z_step,
                y_step,
                x_step,
                z_factor,
                xy_factor,
                min_radius,
            )
        )
        roi_arg.fill(np.float32(11.0))

    monkeypatch.setattr(
        gate_to_grid_map,
        "_rust_kernel",
        lambda name: rust_dist if name == "_gate_to_grid_roi_dist_f32" else None,
    )

    roi_func = gate_to_grid_map.DistRoI(
        0.5, 0.25, 2.0, [(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)]
    )
    mapper.find_roi_for_grid(roi, roi_func)

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][1],
        np.array([(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)], dtype=np.float64),
    )
    assert calls[0][2:] == (
        0.5,
        -1.25,
        2.0,
        1.25,
        0.5,
        -0.75,
        0.5,
        0.25,
        2.0,
    )
    np.testing.assert_array_equal(roi, np.full((2, 2, 3), 11.0, dtype=np.float32))


def test_find_roi_for_grid_dispatches_dist_beam_to_private_rust(monkeypatch):
    mapper = _offset_mapper()
    roi = np.empty((mapper.nz, mapper.ny, mapper.nx), dtype=np.float32)
    calls = []

    def rust_dist_beam(
        roi_arg,
        offsets,
        h_factor,
        z_start,
        y_start,
        x_start,
        z_step,
        y_step,
        x_step,
        beam_factor,
        min_radius,
    ):
        calls.append(
            (
                roi_arg.shape,
                offsets.copy(),
                h_factor.copy(),
                z_start,
                y_start,
                x_start,
                z_step,
                y_step,
                x_step,
                beam_factor,
                min_radius,
            )
        )
        roi_arg.fill(np.float32(13.0))

    monkeypatch.setattr(
        gate_to_grid_map,
        "_rust_kernel",
        lambda name: rust_dist_beam
        if name == "_gate_to_grid_roi_dist_beam_f32"
        else None,
    )

    roi_func = gate_to_grid_map.DistBeamRoI(
        np.array([1.25, 0.5, 1.75], dtype=np.float32),
        1.1,
        0.9,
        2.5,
        [(1.0, 2.0, 3.0)],
    )
    mapper.find_roi_for_grid(roi, roi_func)

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][1], np.array([(1.0, 2.0, 3.0)], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        calls[0][2], np.array([1.25, 0.5, 1.75], dtype=np.float32)
    )
    assert calls[0][3:] == (
        0.5,
        -1.25,
        2.0,
        1.25,
        0.5,
        -0.75,
        roi_func.beam_factor,
        2.5,
    )
    np.testing.assert_array_equal(roi, np.full((2, 2, 3), 13.0, dtype=np.float32))


def test_find_roi_for_grid_keeps_python_path_for_custom_roi(monkeypatch):
    class CustomRoI(gate_to_grid_map.RoIFunction):
        def get_roi(self, z, y, x):
            return z + y + x

    def fail_if_called(name):
        if name.startswith("_gate_to_grid_roi_"):
            raise AssertionError("custom RoI must stay on the Python path")
        return None

    mapper = _mapper()
    roi = np.empty((mapper.nz, mapper.ny, mapper.nx), dtype=np.float32)
    monkeypatch.setattr(gate_to_grid_map, "_rust_kernel", fail_if_called)

    mapper.find_roi_for_grid(roi, CustomRoI())

    expected = np.empty_like(roi)
    for ix in range(mapper.nx):
        for iy in range(mapper.ny):
            for iz in range(mapper.nz):
                expected[iz, iy, ix] = iz + iy + ix
    np.testing.assert_array_equal(roi, expected)


def test_find_roi_for_grid_keeps_python_path_for_non_float32_output(monkeypatch):
    mapper = _mapper()
    roi = np.empty((mapper.nz, mapper.ny, mapper.nx), dtype=np.float64)

    def fail_if_called(name):
        if name.startswith("_gate_to_grid_roi_"):
            raise AssertionError("non-float32 RoI output must stay on Python path")
        return None

    monkeypatch.setattr(gate_to_grid_map, "_rust_kernel", fail_if_called)

    mapper.find_roi_for_grid(roi, gate_to_grid_map.ConstantRoI(3.5))

    assert roi.dtype == np.float64
    np.testing.assert_array_equal(roi, np.full(roi.shape, 3.5, dtype=np.float64))


def test_find_roi_for_grid_python_dist_beam_preserves_float32_h_factor_rounding(
    monkeypatch,
):
    mapper = _offset_mapper()
    h_factor = np.array([1.25, -0.5, 1.75], dtype=np.float32)
    roi_func = gate_to_grid_map.DistBeamRoI(
        h_factor,
        1.1,
        0.9,
        0.0,
        [(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)],
    )

    actual = _fallback_roi(mapper, roi_func, monkeypatch)
    expected = np.empty_like(actual)
    for ix in range(mapper.nx):
        x = mapper.x_start + mapper.x_step * ix
        for iy in range(mapper.ny):
            y = mapper.y_start + mapper.y_step * iy
            for iz in range(mapper.nz):
                z = mapper.z_start + mapper.z_step * iz
                min_roi = 999999999.0
                for z_offset, y_offset, x_offset in roi_func.offsets:
                    distance2 = (
                        (h_factor[0] * (z - z_offset)) ** 2
                        + (h_factor[1] * (y - y_offset)) ** 2
                        + (h_factor[2] * (x - x_offset)) ** 2
                    )
                    roi = gate_to_grid_map.sqrt(distance2) * roi_func.beam_factor
                    if roi < roi_func.min_radius:
                        roi = roi_func.min_radius
                    if roi < min_roi:
                        min_roi = roi
                expected[iz, iy, ix] = min_roi

    np.testing.assert_array_equal(actual, expected)


def test_map_gate_dispatches_to_private_rust_kernel_for_oracle_arrays(monkeypatch):
    mapper = _mapper()
    values = np.array([5.0, 9.0], dtype=np.float32)
    masks = np.array([0, 1], dtype=np.uint8)
    dist_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    calls = []

    def rust_map_gate(
        x,
        y,
        z,
        roi,
        x_start,
        y_start,
        z_start,
        x_step,
        y_step,
        z_step,
        nx,
        ny,
        nz,
        grid_sum,
        grid_wsum,
        min_dist2,
        values_arg,
        masks_arg,
        weighting_function,
        dist_factor_arg,
    ):
        calls.append(
            (
                x,
                y,
                z,
                roi,
                x_start,
                y_start,
                z_start,
                x_step,
                y_step,
                z_step,
                nx,
                ny,
                nz,
                grid_sum.dtype,
                grid_wsum.dtype,
                min_dist2.dtype,
                values_arg.dtype,
                masks_arg.dtype,
                weighting_function,
                dist_factor_arg.dtype,
            )
        )
        grid_sum[0, 0, 0, 0] = 123.0
        grid_wsum[0, 0, 0, 0] = 1.0
        min_dist2[0, 0, 0, 0] = 0.25
        return 1

    monkeypatch.setattr(
        gate_to_grid_map,
        "_rust_kernel",
        lambda name: rust_map_gate if name == "_gate_to_grid_map_gate" else None,
    )

    result = mapper.map_gate(
        0.25,
        0.25,
        0.0,
        1.25,
        values,
        masks,
        gate_to_grid_map.CRESSMAN,
        dist_factor,
    )

    assert result == 1
    assert calls == [
        (
            0.25,
            0.25,
            0.0,
            1.25,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            3,
            2,
            2,
            np.dtype("float32"),
            np.dtype("float32"),
            np.dtype("float64"),
            np.dtype("float32"),
            np.dtype("uint8"),
            gate_to_grid_map.CRESSMAN,
            np.dtype("float32"),
        )
    ]
    assert mapper.grid_sum[0, 0, 0, 0] == np.float32(123.0)
    assert mapper.grid_wsum[0, 0, 0, 0] == np.float32(1.0)
    assert mapper.min_dist2[0, 0, 0, 0] == 0.25


def test_map_gate_keeps_python_path_for_unsupported_mask_dtype(monkeypatch):
    mapper = _mapper()
    values = np.array([5.0, 9.0], dtype=np.float32)
    masks = np.array([False, True], dtype=np.bool_)
    dist_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def rust_map_gate(*_args):
        raise AssertionError("bool masks must use the Python fallback")

    monkeypatch.setattr(
        gate_to_grid_map,
        "_rust_kernel",
        lambda name: rust_map_gate if name == "_gate_to_grid_map_gate" else None,
    )

    result = mapper.map_gate(
        0.0,
        0.0,
        0.0,
        1.1,
        values,
        masks,
        gate_to_grid_map.CRESSMAN,
        dist_factor,
    )

    assert result == 1
    assert mapper.grid_sum[0, 0, 0, 0] > 0
    assert mapper.grid_sum[0, 0, 0, 1] == 0
    assert mapper.grid_wsum[0, 0, 0, 0] > 0
    assert mapper.grid_wsum[0, 0, 0, 1] == 0


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "weighting_function",
    [
        gate_to_grid_map.BARNES,
        gate_to_grid_map.CRESSMAN,
        gate_to_grid_map.NEAREST,
        gate_to_grid_map.BARNES2,
    ],
)
def test_real_rust_map_gate_matches_python_fallback(monkeypatch, weighting_function):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    rust_kernel = getattr(rust, "_gate_to_grid_map_gate", None)
    if rust_kernel is None:
        pytest.fail("pyart._rust has not registered _gate_to_grid_map_gate")

    values = np.array([5.0, 9.0], dtype=np.float32)
    masks = np.array([0, 1], dtype=np.uint8)
    dist_factor = np.array([1.0, 0.75, 1.25], dtype=np.float32)

    expected = _mapper()
    monkeypatch.setattr(gate_to_grid_map, "_rust_kernel", lambda _name: None)
    expected.map_gate(
        0.4,
        0.35,
        0.2,
        1.4,
        values,
        masks,
        weighting_function,
        dist_factor,
    )

    actual = _mapper()
    monkeypatch.setattr(
        gate_to_grid_map,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_gate_to_grid_map_gate" else None,
    )
    actual.map_gate(
        0.4,
        0.35,
        0.2,
        1.4,
        values,
        masks,
        weighting_function,
        dist_factor,
    )

    np.testing.assert_array_equal(actual.grid_sum, expected.grid_sum)
    np.testing.assert_array_equal(actual.grid_wsum, expected.grid_wsum)
    np.testing.assert_array_equal(actual.min_dist2, expected.min_dist2)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "roi_func",
    [
        gate_to_grid_map.ConstantRoI(7.25),
        gate_to_grid_map.DistRoI(
            0.5, 0.25, 2.0, [(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)]
        ),
        gate_to_grid_map.DistBeamRoI(
            np.array([1.25, 0.5, 1.75], dtype=np.float32),
            1.1,
            0.9,
            2.5,
            [(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)],
        ),
    ],
)
def test_real_rust_find_roi_for_grid_matches_python_fallback(monkeypatch, roi_func):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    mapper = _offset_mapper()
    expected = _fallback_roi(mapper, roi_func, monkeypatch)
    monkeypatch.undo()

    actual = np.empty_like(expected)
    mapper.find_roi_for_grid(actual, roi_func)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_find_roi_direct_helpers_validate_inputs():
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    roi = np.empty((2, 2, 3), dtype=np.float32)
    rust._gate_to_grid_roi_constant_f32(roi, np.float32(4.0))
    np.testing.assert_array_equal(roi, np.full(roi.shape, 4.0, dtype=np.float32))

    offsets = np.array([(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)], dtype=np.float64)
    rust._gate_to_grid_roi_dist_f32(
        roi,
        offsets,
        0.5,
        -1.25,
        2.0,
        1.25,
        0.5,
        -0.75,
        0.5,
        0.25,
        2.0,
    )
    assert roi.dtype == np.float32
    assert roi.shape == (2, 2, 3)

    h_factor = np.array([1.25, 0.5, 1.75], dtype=np.float32)
    rust._gate_to_grid_roi_dist_beam_f32(
        roi,
        offsets,
        h_factor,
        0.5,
        -1.25,
        2.0,
        1.25,
        0.5,
        -0.75,
        0.1,
        2.5,
    )
    assert roi.dtype == np.float32
    assert roi.shape == (2, 2, 3)

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._gate_to_grid_roi_constant_f32(roi[:, :, ::2], np.float32(1.0))
    with pytest.raises(ValueError, match="shape"):
        rust._gate_to_grid_roi_dist_f32(
            roi,
            np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(3, 1),
            0.5,
            -1.25,
            2.0,
            1.25,
            0.5,
            -0.75,
            0.5,
            0.25,
            2.0,
        )
    with pytest.raises(ValueError, match="length 3"):
        rust._gate_to_grid_roi_dist_beam_f32(
            roi,
            offsets,
            np.array([1.0, 2.0], dtype=np.float32),
            0.5,
            -1.25,
            2.0,
            1.25,
            0.5,
            -0.75,
            0.1,
            2.5,
        )
