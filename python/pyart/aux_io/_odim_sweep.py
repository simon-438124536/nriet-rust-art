"""Private ODIM/SINARAME HDF5 sweep decode helpers."""

import numpy as np

from .._rust_bridge import get_rust_module


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _get_odim_like_h5_sweep_data_rust(raw_data, attrs):
    args = _can_use_rust_odim_like_sweep_data(raw_data, attrs)
    if args is None:
        return None
    kernel_name, raw_data, has_nodata, nodata, fill_value, has_undetect, undetect, gain, offset = args
    kernel = _rust_kernel(kernel_name)
    if kernel is None:
        return None
    data, mask = kernel(raw_data, has_nodata, nodata, has_undetect, undetect, gain, offset)
    result = np.ma.masked_array(data, mask=mask)
    if has_nodata:
        result.set_fill_value(fill_value)
    return result


def _can_use_rust_odim_like_sweep_data(raw_data, attrs):
    if not (
        type(raw_data) is np.ndarray
        and raw_data.ndim >= 1
        and raw_data.flags.c_contiguous
        and raw_data.dtype in (np.dtype(np.uint8), np.dtype(np.uint16))
    ):
        return None
    gain = _finite_float_attr(attrs.get("gain", 1.0))
    offset = _finite_float_attr(attrs.get("offset", 0.0))
    if gain is None or offset is None:
        return None
    if np.result_type(raw_data.dtype, attrs.get("gain", 1.0), attrs.get("offset", 0.0)) != np.dtype(
        np.float64
    ):
        return None

    nodata_attr = attrs.get("nodata") if "nodata" in attrs else None
    undetect_attr = attrs.get("undetect") if "undetect" in attrs else None
    has_nodata = "nodata" in attrs
    has_undetect = "undetect" in attrs
    nodata = 0
    undetect = 0
    fill_value = None
    if has_nodata:
        nodata = _integer_sentinel(nodata_attr, raw_data.dtype)
        if nodata is None:
            return None
        fill_value = nodata_attr
    if has_undetect:
        undetect = _integer_sentinel(undetect_attr, raw_data.dtype)
        if undetect is None:
            return None

    kernel_name = "_odim_decode_u8" if raw_data.dtype == np.uint8 else "_odim_decode_u16"
    return (
        kernel_name,
        raw_data,
        has_nodata,
        nodata,
        fill_value,
        has_undetect,
        undetect,
        gain,
        offset,
    )


def _finite_float_attr(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    array = np.asarray(value)
    if array.shape != ():
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _integer_sentinel(value, dtype):
    if isinstance(value, (bool, np.bool_)):
        return None
    array = np.asarray(value)
    if array.shape != ():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    integer = int(numeric)
    info = np.iinfo(dtype)
    if integer < info.min or integer > info.max:
        return None
    return integer
