"""
Functions for computing statistics on circular (directional) distributions.

"""

import numpy as np
from operator import index

from .._rust_bridge import get_rust_module

# For details on these computation see:
# https://en.wikipedia.org/wiki/Directional_statistics
# https://en.wikipedia.org/wiki/Mean_of_circular_quantities


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _can_use_rust_array64(*values):
    return all(
        type(value) is np.ndarray
        and value.dtype == np.float64
        and value.ndim > 0
        and value.size > 0
        and np.isfinite(value).all()
        and not np.ma.isMaskedArray(value)
        for value in values
    )


def _can_broadcast(*values):
    try:
        np.broadcast_shapes(*(value.shape for value in values))
    except ValueError:
        return False
    return True


def _native_mean_axis(field, avg_type, axis):
    if isinstance(axis, (bool, np.bool_)):
        return None
    if (
        avg_type != "mean"
        or type(field) is not np.ndarray
        or field.dtype != np.float64
        or field.ndim != 2
        or field.size == 0
        or not field.flags.c_contiguous
        or np.ma.isMaskedArray(field)
        or not np.isfinite(field).all()
    ):
        return None

    try:
        axis_int = index(axis)
    except TypeError:
        return None

    if axis_int not in (0, 1):
        return None
    return axis_int


def compute_directional_stats(field, avg_type="mean", nvalid_min=1, axis=0):
    """
    Computes the mean or the median along one of the axis (ray or range)

    Parameters
    ----------
    field : ndarray
        the radar field data
    avg_type :str
        the type of average: 'mean' or 'median'
    nvalid_min : int
        the minimum number of points to consider the stats valid. Default 1
    axis : int
        the axis along which to compute (0=ray, 1=range)

    Returns
    -------
    values : ndarray 1D
        The resultant statistics
    nvalid : ndarray 1D
        The number of valid points used in the computation
    """
    axis_int = _native_mean_axis(field, avg_type, axis)
    kernel = (
        None
        if axis_int is None
        else _rust_kernel("_compute_directional_stats_mean_dense_f64")
    )
    if kernel is not None:
        values, nvalid = kernel(field, axis_int)
        return np.ma.array(values, mask=nvalid < nvalid_min), nvalid

    if avg_type == "mean":
        values = np.ma.mean(field, axis=axis)
    else:
        values = np.ma.median(field, axis=axis)

    # Set to non-valid if there is not a minimum number of valid gates
    valid = np.logical_not(np.ma.getmaskarray(field))
    nvalid = np.sum(valid, axis=axis, dtype=int)
    values[nvalid < nvalid_min] = np.ma.masked

    return values, nvalid


def mean_of_two_angles(angles1, angles2):
    """
    Compute the element by element mean of two sets of angles.

    Parameters
    ----------
    angles1 : array
        First set of angles in radians.
    angles2 : array
        Second set of angles in radians.

    Returns
    -------
    mean : array
        Elements by element angular mean of the two sets of angles in radians.

    """
    angles1 = np.asanyarray(angles1)
    angles2 = np.asanyarray(angles2)
    kernel = _rust_kernel("_mean_of_two_angles")
    if (
        kernel is not None
        and _can_use_rust_array64(angles1, angles2)
        and _can_broadcast(angles1, angles2)
    ):
        return kernel(angles1, angles2)

    x = (np.cos(angles1) + np.cos(angles2)) / 2.0
    y = (np.sin(angles1) + np.sin(angles2)) / 2.0
    return np.arctan2(y, x)


def mean_of_two_angles_deg(angle1, angle2):
    """
    Compute the element by element mean of two sets of angles in degrees.

    Parameters
    ----------
    angle1 : array
        First set of angles in degrees.
    angle2 : array
        Second set of angles in degrees.

    Returns
    -------
    mean : array
        Elements by element angular mean of the two sets of angles in degrees.

    """
    return np.rad2deg(mean_of_two_angles(np.deg2rad(angle1), np.deg2rad(angle2)))


def angular_mean(angles):
    """
    Compute the mean of a distribution of angles in radians.

    Parameters
    ----------
    angles : array like
        Distribution of angles in radians.

    Returns
    -------
    mean : float
        The mean angle of the distribution in radians.

    """
    angles = np.asanyarray(angles)
    kernel = _rust_kernel("_angular_mean")
    if kernel is not None and _can_use_rust_array64(angles):
        return kernel(angles)

    x = np.cos(angles)
    y = np.sin(angles)
    return np.arctan2(np.mean(y), np.mean(x))


def angular_std(angles):
    """
    Compute the standard deviation of a distribution of angles in radians.

    Parameters
    ----------
    angles : array like
        Distribution of angles in radians.

    Returns
    -------
    std : float
        Standard deviation of the distribution.

    """
    angles = np.asanyarray(angles)
    kernel = _rust_kernel("_angular_std")
    if kernel is not None and _can_use_rust_array64(angles):
        return kernel(angles)

    x = np.cos(angles)
    y = np.sin(angles)
    norm = np.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2)
    return np.sqrt(-2 * np.log(norm))


def angular_mean_deg(angles):
    """
    Compute the mean of a distribution of angles in degrees.

    Parameters
    ----------
    angles : array like
        Distribution of angles in degrees.

    Returns
    -------
    mean : float
        The mean angle of the distribution in degrees.

    """
    return np.rad2deg(angular_mean(np.deg2rad(angles)))


def angular_std_deg(angles):
    """
    Compute the standard deviation of a distribution of angles in degrees.

    Parameters
    ----------
    angles : array like
        Distribution of angles in degrees.

    Returns
    -------
    std : float
        Standard deviation of the distribution.

    """
    return np.rad2deg(angular_std(np.deg2rad(angles)))


def interval_mean(dist, interval_min, interval_max):
    """
    Compute the mean of a distribution within an interval.

    Return the average of the array elements which are interpreted as being
    taken from a circular interval with endpoints given by interval_min and
    interval_max.

    Parameters
    ----------
    dist : array like
        Distribution of values within an interval.
    interval_min, interval_max : float
        The endpoints of the interval.

    Returns
    -------
    mean : float
        The mean value of the distribution.

    """
    # transform distribution from original interval to [-pi, pi]
    dist_array = np.asarray(dist)
    kernel = _rust_kernel("_interval_mean")
    if (
        kernel is not None
        and _can_use_rust_array64(dist_array)
        and interval_max != interval_min
    ):
        return kernel(dist_array, float(interval_min), float(interval_max))

    half_width = (interval_max - interval_min) / 2.0
    center = interval_min + half_width
    a = (dist_array - center) / (half_width) * np.pi

    # compute the angular mean and convert back to original interval
    a_mean = angular_mean(a)
    return (a_mean * (half_width) / np.pi) + center


def interval_std(dist, interval_min, interval_max):
    """
    Compute the standard deviation of a distribution within an interval.

    Return the standard deviation of the array elements which are interpreted
    as being taken from a circular interval with endpoints given by
    interval_min and interval_max.

    Parameters
    ----------
    dist : array_like
        Distribution of values within an interval.
    interval_min, interval_max : float
        The endpoints of the interval.

    Returns
    -------
    std : float
        The standard deviation of the distribution.

    """
    # transform distribution from original interval to [-pi, pi]
    dist_array = np.asarray(dist)
    kernel = _rust_kernel("_interval_std")
    if (
        kernel is not None
        and _can_use_rust_array64(dist_array)
        and interval_max != interval_min
    ):
        return kernel(dist_array, float(interval_min), float(interval_max))

    half_width = (interval_max - interval_min) / 2.0
    center = interval_min + half_width
    a = (dist_array - center) / (half_width) * np.pi

    # compute the angular standard dev. and convert back to original interval
    a_std = angular_std(a)
    return a_std * half_width / np.pi
