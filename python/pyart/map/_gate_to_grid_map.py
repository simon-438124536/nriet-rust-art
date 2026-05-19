"""Pure Python bootstrap shim for gate-to-grid mapping helpers."""

from math import ceil, exp, floor, sqrt, tan

import numpy as np

from .._rust_bridge import get_rust_module


BARNES = 0
CRESSMAN = 1
NEAREST = 2
BARNES2 = 3
PI = 3.141592653589793
GATE_TO_GRID_ROI_RUST_MAX_POINTS = 128 * 1024 * 1024


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


class RoIFunction:
    """Base radius-of-influence function."""

    def get_roi(self, z, y, x):
        """Return the radius of influence for coordinates in meters."""
        return 0


class ConstantRoI(RoIFunction):
    """Constant radius-of-influence function."""

    def __init__(self, constant_roi):
        self.constant_roi = float(constant_roi)

    def get_roi(self, z, y, x):
        """Return the constant radius of influence."""
        return self.constant_roi


class DistRoI(RoIFunction):
    """Radius of influence which expands with distance from the radar."""

    def __init__(self, z_factor, xy_factor, min_radius, offsets):
        self.z_factor = float(z_factor)
        self.xy_factor = float(xy_factor)
        self.min_radius = float(min_radius)
        self.offsets = list(offsets)

    def get_roi(self, z, y, x):
        """Return the radius of influence for coordinates in meters."""
        min_roi = 999999999.0
        for z_offset, y_offset, x_offset in self.offsets:
            roi = self.z_factor * (z - z_offset) + self.xy_factor * sqrt(
                (x - x_offset) ** 2 + (y - y_offset) ** 2
            )
            if roi < self.min_radius:
                roi = self.min_radius
            if roi < min_roi:
                min_roi = roi
        return min_roi


class DistBeamRoI(RoIFunction):
    """Radius of influence which expands with distance from multiple radars."""

    def __init__(self, h_factor, nb, bsp, min_radius, offsets):
        self.h_factor = np.asarray(h_factor, dtype=np.float32)
        self.min_radius = float(min_radius)
        self.beam_factor = tan(float(nb) * float(bsp) * PI / 180.0)
        self.offsets = list(offsets)

    def get_roi(self, z, y, x):
        """Return the radius of influence for coordinates in meters."""
        min_roi = 999999999.0
        for z_offset, y_offset, x_offset in self.offsets:
            roi = (
                sqrt(
                    (self.h_factor[0] * (z - z_offset)) ** 2
                    + (self.h_factor[1] * (y - y_offset)) ** 2
                    + (self.h_factor[2] * (x - x_offset)) ** 2
                )
                * self.beam_factor
            )
            if roi < self.min_radius:
                roi = self.min_radius
            if roi < min_roi:
                min_roi = roi
        return min_roi


class GateToGridMapper:
    """
    Map radar gates to a regular grid using distance-weighted accumulation.
    """

    def __init__(self, grid_shape, grid_starts, grid_steps, grid_sum, grid_wsum):
        nz, ny, nx = grid_shape
        z_start, y_start, x_start = grid_starts
        z_step, y_step, x_step = grid_steps

        self.x_step = float(x_step)
        self.y_step = float(y_step)
        self.z_step = float(z_step)
        self.x_start = float(x_start)
        self.y_start = float(y_start)
        self.z_start = float(z_start)
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.grid_sum = grid_sum
        self.grid_wsum = grid_wsum
        self.nfields = int(grid_sum.shape[3])
        self.min_dist2 = 1e30 * np.ones((nz, ny, nx, self.nfields))

    def find_roi_for_grid(self, roi_array, roi_func):
        """Fill ``roi_array`` with radius-of-influence values."""
        if _find_roi_for_grid_rust(self, roi_array, roi_func):
            return
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    x = self.x_start + self.x_step * ix
                    y = self.y_start + self.y_step * iy
                    z = self.z_start + self.z_step * iz
                    roi_array[iz, iy, ix] = roi_func.get_roi(z, y, x)

    def map_gates_to_grid(
        self,
        ngates,
        nrays,
        gate_z,
        gate_y,
        gate_x,
        field_data,
        field_mask,
        excluded_gates,
        roi_func,
        weighting_function,
        dist_factor,
    ):
        """Map radar gates onto the regular grid."""
        dist_factor = np.asarray(dist_factor, dtype=np.float32)
        for nray in range(int(nrays)):
            for ngate in range(int(ngates)):
                if excluded_gates[nray, ngate]:
                    continue
                x = float(gate_x[nray, ngate])
                y = float(gate_y[nray, ngate])
                z = float(gate_z[nray, ngate])
                roi = roi_func.get_roi(z, y, x)
                self.map_gate(
                    x,
                    y,
                    z,
                    roi,
                    field_data[nray, ngate],
                    field_mask[nray, ngate],
                    int(weighting_function),
                    dist_factor,
                )

    def map_gate(self, x, y, z, roi, values, masks, weighting_function, dist_factor):
        """Map a single gate to the grid."""
        values_array = np.asarray(values)
        masks_array = np.asarray(masks)
        dist_factor_array = np.asarray(dist_factor)
        kernel = _rust_kernel("_gate_to_grid_map_gate")
        if kernel is not None and _can_use_map_gate_rust(
            self,
            x,
            y,
            z,
            roi,
            values_array,
            masks_array,
            weighting_function,
            dist_factor_array,
        ):
            return kernel(
                x,
                y,
                z,
                roi,
                self.x_start,
                self.y_start,
                self.z_start,
                self.x_step,
                self.y_step,
                self.z_step,
                self.nx,
                self.ny,
                self.nz,
                self.grid_sum,
                self.grid_wsum,
                self.min_dist2,
                values_array,
                masks_array,
                int(weighting_function),
                dist_factor_array,
            )

        use_oracle_float32_math = _uses_oracle_float32_math(
            self, values_array, masks_array, dist_factor_array
        )
        if use_oracle_float32_math:
            x = np.float32(x)
            y = np.float32(y)
            z = np.float32(z)
            roi = np.float32(roi)
            x_start = np.float32(self.x_start)
            y_start = np.float32(self.y_start)
            z_start = np.float32(self.z_start)
            x_step = np.float32(self.x_step)
            y_step = np.float32(self.y_step)
            z_step = np.float32(self.z_step)
            values = values_array
            masks = masks_array
            dist_factor = dist_factor_array
        else:
            x_start = self.x_start
            y_start = self.y_start
            z_start = self.z_start
            x_step = self.x_step
            y_step = self.y_step
            z_step = self.z_step

        x -= x_start
        y -= y_start
        z -= z_start

        x_min = find_min(x, roi, x_step)
        if x_min > self.nx - 1:
            return 0
        x_max = find_max(x, roi, x_step, self.nx)
        if x_max < 0:
            return 0

        y_min = find_min(y, roi, y_step)
        if y_min > self.ny - 1:
            return 0
        y_max = find_max(y, roi, y_step, self.ny)
        if y_max < 0:
            return 0

        z_min = find_min(z, roi, z_step)
        if z_min > self.nz - 1:
            return 0
        z_max = find_max(z, roi, z_step, self.nz)
        if z_max < 0:
            return 0

        roi2 = roi * roi
        if use_oracle_float32_math:
            roi2 = np.float32(roi2)

        if weighting_function == NEAREST:
            for xi in range(x_min, x_max + 1):
                for yi in range(y_min, y_max + 1):
                    for zi in range(z_min, z_max + 1):
                        xg = x_step * xi
                        yg = y_step * yi
                        zg = z_step * zi
                        dist2 = (
                            dist_factor[2] * (xg - x) ** 2
                            + dist_factor[1] * (yg - y) ** 2
                            + dist_factor[0] * (zg - z) ** 2
                        )
                        if use_oracle_float32_math:
                            dist2 = np.float32(dist2)
                        if dist2 >= roi2:
                            continue
                        for i in range(self.nfields):
                            if dist2 < self.min_dist2[zi, yi, xi, i]:
                                self.min_dist2[zi, yi, xi, i] = dist2
                                if masks[i]:
                                    self.grid_wsum[zi, yi, xi, i] = 0
                                    self.grid_sum[zi, yi, xi, i] = 0
                                else:
                                    self.grid_wsum[zi, yi, xi, i] = 1
                                    self.grid_sum[zi, yi, xi, i] = values[i]
            return 1

        for xi in range(x_min, x_max + 1):
            for yi in range(y_min, y_max + 1):
                for zi in range(z_min, z_max + 1):
                    xg = x_step * xi
                    yg = y_step * yi
                    zg = z_step * zi
                    dist2 = (
                        dist_factor[2] * (xg - x) * (xg - x)
                        + dist_factor[1] * (yg - y) * (yg - y)
                        + dist_factor[0] * (zg - z) * (zg - z)
                    )
                    if use_oracle_float32_math:
                        dist2 = np.float32(dist2)
                    if dist2 > roi2:
                        continue

                    if weighting_function == BARNES:
                        weight = exp(-dist2 / (2 * roi2)) + 1e-5
                    elif weighting_function == BARNES2:
                        weight = exp(-dist2 / (roi2 / 4)) + 1e-5
                    else:
                        weight = (roi2 - dist2) / (roi2 + dist2)
                    if use_oracle_float32_math:
                        weight = np.float32(weight)

                    for i in range(self.nfields):
                        if masks[i]:
                            continue
                        self.grid_sum[zi, yi, xi, i] += weight * values[i]
                        self.grid_wsum[zi, yi, xi, i] += weight
        return 1


def _can_use_map_gate_rust(
    mapper, x, y, z, roi, values, masks, weighting_function, dist_factor
):
    try:
        weighting_function = int(weighting_function)
    except (TypeError, ValueError, OverflowError):
        return False
    if weighting_function not in (BARNES, CRESSMAN, NEAREST, BARNES2):
        return False

    scalars = (
        x,
        y,
        z,
        roi,
        mapper.x_start,
        mapper.y_start,
        mapper.z_start,
        mapper.x_step,
        mapper.y_step,
        mapper.z_step,
    )
    try:
        if not all(np.isfinite(np.float32(value)) for value in scalars):
            return False
    except (TypeError, ValueError, OverflowError):
        return False

    grid_shape = (mapper.nz, mapper.ny, mapper.nx, mapper.nfields)
    for array, dtype, shape in (
        (mapper.grid_sum, np.float32, grid_shape),
        (mapper.grid_wsum, np.float32, grid_shape),
        (mapper.min_dist2, np.float64, grid_shape),
        (values, np.float32, (mapper.nfields,)),
        (masks, np.uint8, (mapper.nfields,)),
        (dist_factor, np.float32, (3,)),
    ):
        if not (
            isinstance(array, np.ndarray)
            and array.dtype == dtype
            and array.shape == shape
            and array.flags.c_contiguous
        ):
            return False

    if not (
        mapper.grid_sum.flags.writeable
        and mapper.grid_wsum.flags.writeable
        and mapper.min_dist2.flags.writeable
    ):
        return False

    return not (
        np.may_share_memory(mapper.grid_sum, mapper.grid_wsum)
        or np.may_share_memory(mapper.grid_sum, mapper.min_dist2)
        or np.may_share_memory(mapper.grid_wsum, mapper.min_dist2)
    )


def _find_roi_for_grid_rust(mapper, roi_array, roi_func):
    roi_array = np.asarray(roi_array)
    if not _can_use_roi_output(mapper, roi_array):
        return False

    if type(roi_func) is ConstantRoI:
        kernel = _rust_kernel("_gate_to_grid_roi_constant_f32")
        if kernel is None or not _finite_float32_scalar(roi_func.constant_roi):
            return False
        try:
            kernel(roi_array, np.float32(roi_func.constant_roi))
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    if type(roi_func) is DistRoI:
        offsets = _roi_offsets_array(roi_func.offsets)
        if offsets is None or not _finite_float64_scalars(
            mapper.z_start,
            mapper.y_start,
            mapper.x_start,
            mapper.z_step,
            mapper.y_step,
            mapper.x_step,
            roi_func.z_factor,
            roi_func.xy_factor,
            roi_func.min_radius,
        ):
            return False
        kernel = _rust_kernel("_gate_to_grid_roi_dist_f32")
        if kernel is None:
            return False
        try:
            kernel(
                roi_array,
                offsets,
                mapper.z_start,
                mapper.y_start,
                mapper.x_start,
                mapper.z_step,
                mapper.y_step,
                mapper.x_step,
                roi_func.z_factor,
                roi_func.xy_factor,
                roi_func.min_radius,
            )
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    if type(roi_func) is DistBeamRoI:
        offsets = _roi_offsets_array(roi_func.offsets)
        h_factor = np.asarray(roi_func.h_factor)
        if (
            offsets is None
            or not (
                type(h_factor) is np.ndarray
                and h_factor.dtype == np.float32
                and h_factor.shape == (3,)
                and h_factor.flags.c_contiguous
                and np.all(np.isfinite(h_factor))
            )
            or not _finite_float64_scalars(
                mapper.z_start,
                mapper.y_start,
                mapper.x_start,
                mapper.z_step,
                mapper.y_step,
                mapper.x_step,
                roi_func.beam_factor,
                roi_func.min_radius,
            )
        ):
            return False
        kernel = _rust_kernel("_gate_to_grid_roi_dist_beam_f32")
        if kernel is None:
            return False
        try:
            kernel(
                roi_array,
                offsets,
                h_factor,
                mapper.z_start,
                mapper.y_start,
                mapper.x_start,
                mapper.z_step,
                mapper.y_step,
                mapper.x_step,
                roi_func.beam_factor,
                roi_func.min_radius,
            )
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    return False


def _can_use_roi_output(mapper, roi_array):
    return (
        type(roi_array) is np.ndarray
        and roi_array.dtype == np.float32
        and roi_array.shape == (mapper.nz, mapper.ny, mapper.nx)
        and roi_array.flags.c_contiguous
        and roi_array.flags.writeable
        and roi_array.size <= GATE_TO_GRID_ROI_RUST_MAX_POINTS
    )


def _roi_offsets_array(offsets):
    try:
        offsets_array = np.asarray(offsets, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if offsets_array.size == 0:
        offsets_array = np.empty((0, 3), dtype=np.float64)
    if not (
        type(offsets_array) is np.ndarray
        and offsets_array.dtype == np.float64
        and offsets_array.ndim == 2
        and offsets_array.shape[1] == 3
        and offsets_array.flags.c_contiguous
        and np.all(np.isfinite(offsets_array))
    ):
        return None
    return offsets_array


def _finite_float32_scalar(value):
    try:
        return np.isfinite(np.float32(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_float64_scalars(*values):
    try:
        return all(np.isfinite(float(value)) for value in values)
    except (TypeError, ValueError, OverflowError):
        return False


def _uses_oracle_float32_math(mapper, values, masks, dist_factor):
    grid_shape = (mapper.nz, mapper.ny, mapper.nx, mapper.nfields)
    return (
        isinstance(mapper.grid_sum, np.ndarray)
        and mapper.grid_sum.dtype == np.float32
        and mapper.grid_sum.shape == grid_shape
        and isinstance(mapper.grid_wsum, np.ndarray)
        and mapper.grid_wsum.dtype == np.float32
        and mapper.grid_wsum.shape == grid_shape
        and isinstance(mapper.min_dist2, np.ndarray)
        and mapper.min_dist2.dtype == np.float64
        and mapper.min_dist2.shape == grid_shape
        and isinstance(values, np.ndarray)
        and values.dtype == np.float32
        and values.shape == (mapper.nfields,)
        and isinstance(masks, np.ndarray)
        and masks.dtype == np.uint8
        and masks.shape == (mapper.nfields,)
        and isinstance(dist_factor, np.ndarray)
        and dist_factor.dtype == np.float32
        and dist_factor.shape == (3,)
    )


def find_min(a, roi, step):
    """Find the minimum gate index for a dimension."""
    kernel = _rust_kernel("_gate_to_grid_find_min")
    if kernel is not None and _can_use_find_bound_rust(a, roi, step):
        return int(kernel(a, roi, step))
    return _find_min_python(a, roi, step)


def _find_min_python(a, roi, step):
    a = np.float32(a)
    roi = np.float32(roi)
    step = np.float32(step)
    if step == 0:
        return 0
    a_min = int(ceil((a - roi) / step))
    if a_min < 0:
        a_min = 0
    return a_min


def find_max(a, roi, step, na):
    """Find the maximum gate index for a dimension."""
    kernel = _rust_kernel("_gate_to_grid_find_max")
    if kernel is not None and _can_use_find_bound_rust(a, roi, step):
        return int(kernel(a, roi, step, int(na)))
    return _find_max_python(a, roi, step, na)


def _find_max_python(a, roi, step, na):
    a = np.float32(a)
    roi = np.float32(roi)
    step = np.float32(step)
    if step == 0:
        return 0
    a_max = int(floor((a + roi) / step))
    if a_max > na - 1:
        a_max = na - 1
    return a_max


def _can_use_find_bound_rust(a, roi, step):
    try:
        return all(np.isfinite(np.float32(value)) for value in (a, roi, step))
    except (TypeError, ValueError, OverflowError):
        return False
