"""
Function for creating simulated velocity fields.

"""

import numpy as np
from scipy import interpolate

from .._rust_bridge import get_rust_module
from ..config import get_fillvalue, get_metadata

SIMULATED_VEL_RUST_MAX_OUTPUT_VALUES = 512 * 1024 * 1024


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def simulated_vel_from_profile(
    radar, profile, interp_kind="linear", sim_vel_field=None
):
    """
    Create simulated radial velocities from a profile of horizontal winds.

    Parameters
    ----------
    radar : Radar
        Radar instance which provides the scanning parameters for the
        simulated radial velocities.
    profile : HorizontalWindProfile
        Profile of horizontal winds.
    interp_kind : str, optional
        Specifies the kind of interpolation used to determine the winds at a
        given height. Must be one of 'linear', 'nearest', 'zero', 'slinear',
        'quadratic', or 'cubic'. The the documentation for the SciPy
        scipy.interpolate.interp1d function for descriptions.
    sim_vel_field : str, optional
        Name to use for the simulated velocity field metadata. None will use
        the default field name from the Py-ART configuration file.

    Returns
    -------
    sim_vel : dict
        Dictionary containing a radar field of simulated radial velocities.

    """
    # parse parameters
    if sim_vel_field is None:
        sim_vel_field = "simulated_velocity"

    # radar parameters
    azimuths = np.deg2rad(radar.azimuth["data"]).reshape(-1, 1)
    elevations = np.deg2rad(radar.elevation["data"]).reshape(-1, 1)
    gate_altitudes = radar.gate_altitude["data"]

    if isinstance(gate_altitudes, np.ma.MaskedArray):
        gate_altitudes = gate_altitudes.filled(np.nan)

    # prepare wind profile for interpolation
    if isinstance(profile.height, np.ma.MaskedArray):
        height = profile.height.filled(np.nan)
    else:
        height = profile.height

    height_is_not_nan = ~np.isnan(height)
    winds = np.empty((2, len(height)), dtype=np.float64)
    if isinstance(profile.u_wind, np.ma.MaskedArray):
        winds[0] = profile.u_wind.filled(np.nan)
    else:
        winds[0] = profile.u_wind

    if isinstance(profile.v_wind, np.ma.MaskedArray):
        winds[1] = profile.v_wind.filled(np.nan)
    else:
        winds[1] = profile.v_wind

    wind_is_not_nan = np.logical_and(~np.isnan(winds[0]), ~np.isnan(winds[1]))
    no_nans = np.logical_and(height_is_not_nan, wind_is_not_nan)
    height = height[no_nans]

    winds_reshape = np.empty((2, len(winds[0][no_nans])), dtype=np.float64)
    winds_reshape[0] = winds[0][no_nans]
    winds_reshape[1] = winds[1][no_nans]
    wind_interp = interpolate.interp1d(
        height,
        winds_reshape,
        kind=interp_kind,
        bounds_error=False,
        fill_value=get_fillvalue(),
    )

    # interpolated wind speeds at all gates altitudes
    gate_winds = wind_interp(gate_altitudes)
    gate_u = np.ma.masked_invalid(gate_winds[0])
    gate_v = np.ma.masked_invalid(gate_winds[1])

    # calculate the radial velocity for all gates
    radial_vel = _simulated_radial_velocity(gate_u, gate_v, azimuths, elevations)

    sim_vel = get_metadata(sim_vel_field)
    sim_vel["data"] = radial_vel
    return sim_vel


def _simulated_radial_velocity(gate_u, gate_v, azimuths, elevations):
    rust_result = _simulated_radial_velocity_rust(gate_u, gate_v, azimuths, elevations)
    if rust_result is not None:
        return rust_result

    return gate_u * np.sin(azimuths) * np.cos(elevations) + gate_v * np.cos(
        azimuths
    ) * np.cos(elevations)


def _simulated_radial_velocity_rust(gate_u, gate_v, azimuths, elevations):
    args = _can_use_rust_simulated_radial_velocity(
        gate_u, gate_v, azimuths, elevations
    )
    if args is None:
        return None
    gate_u, gate_v, sin_azimuths, cos_azimuths, cos_elevations = args
    kernel = _rust_kernel("_simulated_radial_velocity_dense_f64")
    if kernel is None:
        return None

    try:
        values = kernel(gate_u, gate_v, sin_azimuths, cos_azimuths, cos_elevations)
    except Exception:
        return None

    return np.ma.array(values, mask=np.zeros(values.shape, dtype=bool))


def _can_use_rust_simulated_radial_velocity(gate_u, gate_v, azimuths, elevations):
    if np.ma.is_masked(gate_u) or np.ma.is_masked(gate_v):
        return None

    gate_u_data = np.asarray(gate_u.data if np.ma.isMaskedArray(gate_u) else gate_u)
    gate_v_data = np.asarray(gate_v.data if np.ma.isMaskedArray(gate_v) else gate_v)
    if (
        type(gate_u_data) is not np.ndarray
        or type(gate_v_data) is not np.ndarray
        or gate_u_data.dtype != np.float64
        or gate_v_data.dtype != np.float64
    ):
        return None
    if gate_u_data.ndim != 2 or gate_v_data.shape != gate_u_data.shape:
        return None
    if gate_u_data.size > SIMULATED_VEL_RUST_MAX_OUTPUT_VALUES:
        return None
    if not gate_u_data.flags.c_contiguous or not gate_v_data.flags.c_contiguous:
        return None

    azimuths = np.asarray(azimuths)
    elevations = np.asarray(elevations)
    if azimuths.shape != (gate_u_data.shape[0], 1) or elevations.shape != (
        gate_u_data.shape[0],
        1,
    ):
        return None
    if not (
        np.issubdtype(azimuths.dtype, np.number)
        and np.issubdtype(elevations.dtype, np.number)
    ):
        return None

    try:
        if not (
            np.all(np.isfinite(gate_u_data))
            and np.all(np.isfinite(gate_v_data))
            and np.all(np.isfinite(azimuths))
            and np.all(np.isfinite(elevations))
        ):
            return None
        max_component = max(
            float(np.max(np.abs(gate_u_data), initial=0.0)),
            float(np.max(np.abs(gate_v_data), initial=0.0)),
        )
    except (TypeError, ValueError, FloatingPointError):
        return None
    if max_component > np.finfo(np.float64).max / 4.0:
        return None

    sin_azimuths = np.asarray(np.sin(azimuths).reshape(-1), dtype=np.float64)
    cos_azimuths = np.asarray(np.cos(azimuths).reshape(-1), dtype=np.float64)
    cos_elevations = np.asarray(np.cos(elevations).reshape(-1), dtype=np.float64)
    if not (
        sin_azimuths.flags.c_contiguous
        and cos_azimuths.flags.c_contiguous
        and cos_elevations.flags.c_contiguous
    ):
        return None

    return gate_u_data, gate_v_data, sin_azimuths, cos_azimuths, cos_elevations
