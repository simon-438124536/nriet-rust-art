"""
Utilities for reading EDGE NetCDF files.

"""

import datetime

import netCDF4
import numpy as np

from .._rust_bridge import get_rust_module
from ..config import FileMetadata, get_fillvalue
from ..core.radar import Radar
from ..io.common import _test_arguments, make_time_unit_str


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def read_edge_netcdf(filename, **kwargs):
    """
    Read a EDGE NetCDF file.

    Parameters
    ----------
    filename : str
        Name of EDGE NetCDF file to read data from.

    Returns
    -------
    radar : Radar
        Radar object.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    # create metadata retrieval object
    filemetadata = FileMetadata("edge_netcdf")

    # Open netCDF4 file
    dset = netCDF4.Dataset(filename)
    nrays = len(dset.dimensions["Azimuth"])
    nbins = len(dset.dimensions["Gate"])

    # latitude, longitude and altitude
    latitude = filemetadata("latitude")
    longitude = filemetadata("longitude")
    altitude = filemetadata("altitude")
    latitude["data"] = np.array([dset.Latitude], "float64")
    longitude["data"] = np.array([dset.Longitude], "float64")
    altitude["data"] = np.array([dset.Height], "float64")

    # metadata
    metadata = filemetadata("metadata")
    metadata_mapping = {
        "vcp-value": "vcp",
        "radarName-value": "radar_name",
        "ConversionPlugin": "conversion_software",
    }
    for netcdf_attr, metadata_key in metadata_mapping.items():
        if netcdf_attr in dset.ncattrs():
            metadata[metadata_key] = dset.getncattr(netcdf_attr)

    # sweep_start_ray_index, sweep_end_ray_index
    sweep_start_ray_index = filemetadata("sweep_start_ray_index")
    sweep_end_ray_index = filemetadata("sweep_end_ray_index")
    sweep_start_ray_index["data"] = np.array([0], dtype="int32")
    sweep_end_ray_index["data"] = np.array([nrays - 1], dtype="int32")

    # sweep number
    sweep_number = filemetadata("sweep_number")
    sweep_number["data"] = np.array([0], dtype="int32")

    # sweep_type
    scan_type = "ppi"

    # sweep_mode, fixed_angle
    sweep_mode = filemetadata("sweep_mode")
    fixed_angle = filemetadata("fixed_angle")
    sweep_mode["data"] = np.array(1 * ["azimuth_surveillance"])
    fixed_angle["data"] = np.array([dset.Elevation], dtype="float32")

    # time
    time = filemetadata("time")
    start_time = datetime.datetime.utcfromtimestamp(dset.Time)
    time["units"] = make_time_unit_str(start_time)
    time["data"] = np.zeros((nrays,), dtype="float64")

    # range
    _range = filemetadata("range")
    step = float(dset.getncattr("MaximumRange-value")) / nbins * 1000.0
    _range["data"] = np.arange(nbins, dtype="float32") * step + step / 2
    _range["meters_to_center_of_first_gate"] = step / 2.0
    _range["meters_between_gates"] = step

    # elevation
    elevation = filemetadata("elevation")
    elevation_angle = dset.Elevation
    elevation["data"] = np.ones((nrays,), dtype="float32") * elevation_angle

    # azimuth
    azimuth = filemetadata("azimuth")
    azimuth["data"] = dset.variables["Azimuth"][:]

    # fields
    field_name = dset.TypeName

    field_data = np.ma.array(dset.variables[field_name][:])
    missing = dset.MissingData if "MissingData" in dset.ncattrs() else None
    range_folded = dset.RangeFolded if "RangeFolded" in dset.ncattrs() else None
    field_data = _mask_edge_field_data(field_data, missing, range_folded)

    fields = {field_name: filemetadata(field_name)}
    fields[field_name]["data"] = field_data
    fields[field_name]["units"] = dset.variables[field_name].Units
    fields[field_name]["_FillValue"] = get_fillvalue()

    # instrument_parameters
    instrument_parameters = {}

    if "PRF-value" in dset.ncattrs():
        dic = filemetadata("prt")
        prt = 1.0 / float(dset.getncattr("PRF-value"))
        dic["data"] = np.ones((nrays,), dtype="float32") * prt
        instrument_parameters["prt"] = dic

    if "PulseWidth-value" in dset.ncattrs():
        dic = filemetadata("pulse_width")
        pulse_width = dset.getncattr("PulseWidth-value") * 1.0e-6
        dic["data"] = np.ones((nrays,), dtype="float32") * pulse_width
        instrument_parameters["pulse_width"] = dic

    if "NyquistVelocity-value" in dset.ncattrs():
        dic = filemetadata("nyquist_velocity")
        nyquist_velocity = float(dset.getncattr("NyquistVelocity-value"))
        dic["data"] = np.ones((nrays,), dtype="float32") * nyquist_velocity
        instrument_parameters["nyquist_velocity"] = dic

    if "Beamwidth" in dset.variables:
        dic = filemetadata("radar_beam_width_h")
        dic["data"] = dset.variables["Beamwidth"][:]
        instrument_parameters["radar_beam_width_h"] = dic

    dset.close()

    return Radar(
        time,
        _range,
        fields,
        metadata,
        scan_type,
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth,
        elevation,
        instrument_parameters=instrument_parameters,
    )


def _mask_edge_field_data(field_data, missing, range_folded):
    rust_result = _mask_edge_field_data_rust(field_data, missing, range_folded)
    if rust_result is not None:
        return rust_result

    if missing is not None:
        field_data[field_data == missing] = np.ma.masked
    if range_folded is not None:
        field_data[field_data == range_folded] = np.ma.masked
    return field_data


def _mask_edge_field_data_rust(field_data, missing, range_folded):
    args = _can_use_rust_edge_mask(field_data, missing, range_folded)
    if args is None:
        return None
    data, existing_mask, kernel_name, missing_args, folded_args = args
    kernel = _rust_kernel(kernel_name)
    if kernel is None:
        return None

    try:
        mask = kernel(
            data,
            existing_mask,
            missing_args[0],
            missing_args[1],
            folded_args[0],
            folded_args[1],
        )
    except Exception:
        return None

    result = np.ma.array(field_data, copy=False)
    result.mask = np.asarray(mask, dtype=bool)
    return result


def _can_use_rust_edge_mask(field_data, missing, range_folded):
    if missing is None and range_folded is None:
        return None
    if not np.ma.isMaskedArray(field_data):
        return None

    data = np.asarray(field_data.data)
    if type(data) is not np.ndarray or not data.flags.c_contiguous:
        return None
    if data.dtype.byteorder not in ("=", "|"):
        return None

    dtype_map = {
        np.dtype("uint8"): "_edge_mask_u8",
        np.dtype("uint16"): "_edge_mask_u16",
        np.dtype("int16"): "_edge_mask_i16",
        np.dtype("int32"): "_edge_mask_i32",
        np.dtype("float32"): "_edge_mask_f32",
        np.dtype("float64"): "_edge_mask_f64",
    }
    kernel_name = dtype_map.get(data.dtype)
    if kernel_name is None:
        return None

    existing_mask = np.ma.getmaskarray(field_data)
    if existing_mask.shape != data.shape or not existing_mask.flags.c_contiguous:
        return None

    missing_args = _coerce_edge_marker(missing, data.dtype)
    folded_args = _coerce_edge_marker(range_folded, data.dtype)
    if missing_args is None or folded_args is None:
        return None

    return data, existing_mask, kernel_name, missing_args, folded_args


def _coerce_edge_marker(marker, dtype):
    if marker is None:
        return False, dtype.type(0).item()
    if isinstance(marker, (bool, np.bool_)):
        return None

    try:
        marker_array = np.asarray(marker)
    except (TypeError, ValueError):
        return None
    if marker_array.ndim != 0:
        return None
    if marker_array.dtype.kind in ("O", "S", "U"):
        return None

    if np.issubdtype(dtype, np.integer):
        try:
            marker_float = float(marker_array)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(marker_float) or not marker_float.is_integer():
            return None
        info = np.iinfo(dtype)
        if marker_float < info.min or marker_float > info.max:
            return None
        marker_int = int(marker_float)
        if float(marker_int) != marker_float:
            return None
        return True, dtype.type(marker_int).item()

    if np.issubdtype(dtype, np.floating):
        try:
            return True, dtype.type(marker_array.item()).item()
        except (TypeError, ValueError, OverflowError):
            return None

    return None
