"""Pure Python bootstrap for NEXRAD interpolation kernels."""

import numpy as np

from .._rust_bridge import get_rust_module


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _can_use_rust(data, scratch_ray, start, end, moment_ngates, interp_ngates):
    return (
        type(data) is np.ndarray
        and type(scratch_ray) is np.ndarray
        and data.ndim == 2
        and scratch_ray.ndim == 1
        and data.dtype == np.float32
        and scratch_ray.dtype == np.float32
        and data.flags.c_contiguous
        and scratch_ray.flags.c_contiguous
        and data.flags.writeable
        and scratch_ray.flags.writeable
        and not np.may_share_memory(data, scratch_ray)
        and start >= 0
        and moment_ngates >= 0
        and moment_ngates <= data.shape[1]
        and interp_ngates >= 0
        and interp_ngates <= data.shape[1]
        and interp_ngates <= scratch_ray.shape[0]
        and (end < start or end < data.shape[0])
    )


def _require_int(name, value):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _fast_interpolate_scan_4(
    data, scratch_ray, fill_value, start, end, moment_ngates, linear_interp
):
    """Interpolate a scan from 1000 m to 250 m spacing in place."""
    data_array = np.asarray(data)
    scratch_array = np.asarray(scratch_ray)
    start = _require_int("start", start)
    end = _require_int("end", end)
    moment_ngates = _require_int("moment_ngates", moment_ngates)
    linear_interp = _require_int("linear_interp", linear_interp)
    interp_ngates = 4 * moment_ngates

    kernel = _rust_kernel("_fast_interpolate_scan_4")
    if kernel is not None and _can_use_rust(
        data, scratch_ray, start, end, moment_ngates, interp_ngates
    ):
        kernel(
            data_array,
            scratch_array,
            float(fill_value),
            start,
            end,
            moment_ngates,
            int(linear_interp),
        )
        return None

    for ray_num in range(start, end + 1):
        for i in range(moment_ngates):
            gate_val = data[ray_num, i]
            scratch_ray[i * 4 + 0] = gate_val
            scratch_ray[i * 4 + 1] = gate_val
            scratch_ray[i * 4 + 2] = gate_val
            scratch_ray[i * 4 + 3] = gate_val

        if linear_interp:
            for i in range(2, interp_ngates - 4, 4):
                gate_val = scratch_ray[i]
                next_val = scratch_ray[i + 4]
                if gate_val == fill_value or next_val == fill_value:
                    continue
                delta = (next_val - gate_val) / 4.0
                scratch_ray[i + 0] = gate_val + delta * 0.5
                scratch_ray[i + 1] = gate_val + delta * 1.5
                scratch_ray[i + 2] = gate_val + delta * 2.5
                scratch_ray[i + 3] = gate_val + delta * 3.5

        for i in range(interp_ngates):
            data[ray_num, i] = scratch_ray[i]
    return None


def _fast_interpolate_scan_2(
    data, scratch_ray, fill_value, start, end, moment_ngates, linear_interp
):
    """Interpolate a scan from 300 m to 150 m spacing in place."""
    data_array = np.asarray(data)
    scratch_array = np.asarray(scratch_ray)
    start = _require_int("start", start)
    end = _require_int("end", end)
    moment_ngates = _require_int("moment_ngates", moment_ngates)
    linear_interp = _require_int("linear_interp", linear_interp)
    interp_ngates = 2 * moment_ngates - 1

    kernel = _rust_kernel("_fast_interpolate_scan_2")
    if kernel is not None and _can_use_rust(
        data, scratch_ray, start, end, moment_ngates, interp_ngates
    ):
        kernel(
            data_array,
            scratch_array,
            float(fill_value),
            start,
            end,
            moment_ngates,
            int(linear_interp),
        )
        return None

    for ray_num in range(start, end + 1):
        for i in range(moment_ngates):
            gate_val = data[ray_num, i]
            scratch_ray[i * 2 + 0] = gate_val
            if i != moment_ngates - 1:
                scratch_ray[i * 2 + 1] = gate_val

        if linear_interp:
            for i in range(1, interp_ngates - 2, 2):
                gate_val = scratch_ray[i]
                next_val = scratch_ray[i + 2]
                if gate_val == fill_value or next_val == fill_value:
                    continue
                delta = (next_val - gate_val) / 2.0
                scratch_ray[i + 0] = gate_val + delta * 0.5
                scratch_ray[i + 1] = gate_val + delta * 1.5

        for i in range(interp_ngates):
            data[ray_num, i] = scratch_ray[i]
    return None
