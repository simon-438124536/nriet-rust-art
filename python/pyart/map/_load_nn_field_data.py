"""Pure Python bootstrap shim for nearest-neighbor field loading."""

import numpy as np

from .._rust_bridge import get_rust_module


def _rust_kernel():
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, "_load_nn_field_data", None)


def _can_use_rust(data, nfields, npoints, r_nums, e_nums, sdata):
    if not (
        isinstance(data, np.ndarray)
        and data.ndim == 2
        and data.dtype == np.object_
        and isinstance(r_nums, np.ndarray)
        and r_nums.ndim == 1
        and r_nums.dtype == np.intc
        and isinstance(e_nums, np.ndarray)
        and e_nums.ndim == 1
        and e_nums.dtype == np.intc
        and isinstance(sdata, np.ndarray)
        and sdata.ndim == 2
        and sdata.dtype == np.float64
        and sdata.flags.writeable
    ):
        return False

    if nfields < 0 or npoints < 0:
        return False
    if nfields > data.shape[0] or nfields > sdata.shape[1]:
        return False
    if npoints > r_nums.shape[0] or npoints > e_nums.shape[0] or npoints > sdata.shape[0]:
        return False
    if npoints == 0:
        return True

    r_slice = r_nums[:npoints]
    e_slice = e_nums[:npoints]
    return (
        np.all(r_slice >= 0)
        and np.all(e_slice >= 0)
        and np.all(r_slice < data.shape[1])
    )


def _load_nn_field_data(data, nfields, npoints, r_nums, e_nums, sdata):
    """
    Load nearest-neighbor field data into ``sdata``.

    This mirrors the small Cython helper: ``data`` is an object array where
    ``data[j, r_num]`` yields the ray/gate field array indexed by ``e_num``.
    """
    data_array = np.asarray(data)
    r_nums_array = np.asarray(r_nums)
    e_nums_array = np.asarray(e_nums)
    sdata_array = np.asarray(sdata)

    kernel = _rust_kernel()
    if kernel is not None and _can_use_rust(
        data_array,
        int(nfields),
        int(npoints),
        r_nums_array,
        e_nums_array,
        sdata_array,
    ):
        kernel(data_array, nfields, npoints, r_nums_array, e_nums_array, sdata_array)
        return None

    for i in range(npoints):
        r_num = int(r_nums[i])
        e_num = int(e_nums[i])
        for j in range(nfields):
            sdata[i, j] = data[j, r_num][e_num]
    return None
