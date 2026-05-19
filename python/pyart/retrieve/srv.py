"""
Calculation of storm-relative velocity from a radar object. Code written by
Edward C. Wolff. Modifications for single-sweep files suggested by Leanne
Blind.

"""

import math

import numpy as np

from .._rust_bridge import get_rust_module
from ..config import get_field_name


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _can_use_rust_storm_relative(
    sr_data, velocity_data, angle_array, speed, alpha, start, stop
):
    if not (
        type(sr_data) is np.ndarray
        and type(velocity_data) is np.ndarray
        and type(angle_array) is np.ndarray
        and sr_data.ndim == 2
        and velocity_data.ndim == 2
        and angle_array.ndim == 1
        and sr_data.dtype == np.float64
        and velocity_data.dtype == np.float64
        and angle_array.dtype == np.float64
        and sr_data.flags.c_contiguous
        and velocity_data.flags.c_contiguous
        and angle_array.flags.c_contiguous
        and sr_data.shape == velocity_data.shape
        and not np.ma.isMaskedArray(sr_data)
        and not np.ma.isMaskedArray(velocity_data)
        and not np.ma.isMaskedArray(angle_array)
    ):
        return False
    if not isinstance(speed, (bool, int, float, np.bool_, np.integer, np.floating)):
        return False
    if not isinstance(alpha, (bool, int, float, np.bool_, np.integer, np.floating)):
        return False
    speed = float(speed)
    alpha = float(alpha)
    start = int(start)
    stop = int(stop)
    if not (np.isfinite(speed) and np.isfinite(alpha)):
        return False
    if start < 0 or stop < start or stop > sr_data.shape[0]:
        return False
    if angle_array.shape[0] < stop - start or not np.isfinite(angle_array).all():
        return False
    return True


def _apply_storm_relative_velocity_rust(
    sr_data, velocity_data, angle_array, speed, alpha, start, stop
):
    kernel = _rust_kernel("_storm_relative_velocity_inplace")
    if kernel is None or not _can_use_rust_storm_relative(
        sr_data, velocity_data, angle_array, speed, alpha, start, stop
    ):
        return False
    kernel(sr_data, velocity_data, angle_array, float(speed), float(alpha), start, stop)
    return True


def storm_relative_velocity(
    radar, direction=None, speed=None, field=None, u=None, v=None
):
    """
    This function calculates storm-relative Doppler velocities.

    Parameters
    ----------
    radar: Radar
        Radar object used.
    direction: float or string
        Direction of the storm motion vector (where north equals 0 degrees).
        Accepts a float or a string with the abbreviation of a cardinal or
        ordinal/intercardinal direction (for example: N, SE, etc.). If both
        speed/direction and u/v are specified, speed/direction will be used.
    speed: string
        Speed of the storm motion vector.
        Units should be identical to those in the provided radar
        object. If both speed/direction and u/v are specified, speed/direction
        will be used.
    field: string, optional
        Velocity field to use for storm-relative calculation. A value of None
        will use the default field name as defined in the Py-ART configuration
        file.
    u: float, optional
        U-component of the storm motion
    v: float, optional
        V-component of the storm motion

    Returns
    -------
    sr_data : dict
        Field dictionary containing storm-relative Doppler velocities in the
        same units as original velocities and the specified storm speed.
        Array is stored under the 'data' key.

    """
    # Parse the field parameter
    if field is None:
        field = get_field_name("velocity")

    # Obtain velocity data and copy the array
    velocity_data = radar.fields[field]["data"]
    sr_data = velocity_data.copy()

    # Specify cardinal directions that can be interpreted
    direction_dict = {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315,
    }

    # Set the direction of the storm motion vector
    # When speed and direction are specified
    if direction is not None and speed is not None:
        if isinstance(direction, int) or isinstance(direction, float):
            alpha = direction
        elif isinstance(direction, str):
            if direction in direction_dict.keys():
                alpha = direction_dict[direction]
            else:
                raise ValueError("Direction string must be cardinal/ordinal direction")
        else:
            raise ValueError("Direction must be an integer, float, or string")
    # When u and v are specified
    elif u is not None:
        if v is not None:
            speed = np.sqrt((u**2) + (v**2))
            direction = 90 - np.rad2deg(math.atan2(v / speed, u / speed))
            if direction < 0:
                direction = direction + 360
        else:
            raise ValueError("Must specify both u and v components")
    else:
        raise ValueError("Must specify either speed and direction or u and v")

    # Calculates the storm relative velocities
    # If the radar file contains only one sweep (e.g. some research radars)
    if len(radar.sweep_number["data"]) == 1:
        sweep = 0
        start, end = radar.get_start_end(sweep)
        angle_array = radar.get_azimuth(sweep=sweep)
        ray_array = np.arange(start, end, 1)
        if not _apply_storm_relative_velocity_rust(
            sr_data, velocity_data, angle_array, speed, alpha, start, end
        ):
            for count, ray in enumerate(ray_array):
                correction = speed * np.cos(np.deg2rad(alpha - angle_array[count]))
                sr_data[ray] = velocity_data[ray] - correction
    # If the radar file contains several sweeps, one volume scan (e.g. NEXRAD)
    else:
        for sweep in radar.sweep_number["data"]:
            start, end = radar.get_start_end(sweep)
            angle_array = radar.get_azimuth(sweep=sweep)
            ray_array = np.arange(start, end + 1, 1)
            if not _apply_storm_relative_velocity_rust(
                sr_data, velocity_data, angle_array, speed, alpha, start, end + 1
            ):
                for count, ray in enumerate(ray_array):
                    correction = speed * np.cos(np.deg2rad(alpha - angle_array[count]))
                    sr_data[ray] = velocity_data[ray] - correction

    return sr_data
