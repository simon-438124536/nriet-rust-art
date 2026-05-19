"""
Function for mathematical, signal processing and numerical routines.

"""

import operator

import numpy as np
from scipy import signal

from .._rust_bridge import get_rust_module


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _rust_window_shape(N):
    if isinstance(N, int) and not isinstance(N, bool):
        return (N, N)

    try:
        window = tuple(N)
    except TypeError:
        return None

    if len(window) != 2:
        return None
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in window):
        return None
    return window


def _can_use_rust(image, N):
    window = _rust_window_shape(N)
    return (
        window is not None
        and window[0] > 0
        and window[1] > 0
        and window[0] % 2 == 1
        and window[1] % 2 == 1
        and not np.ma.isMaskedArray(image)
        and type(image) is np.ndarray
        and image.ndim == 2
        and image.dtype == np.float64
        and image.shape[0] > 0
        and image.shape[1] > 0
    ), window


def _texture_along_ray_rust(fld, wind_size):
    if isinstance(wind_size, bool):
        return None
    try:
        wind_index = operator.index(wind_size)
    except TypeError:
        return None

    if not (
        type(fld) is np.ndarray
        and fld.ndim == 2
        and fld.dtype == np.float64
        and fld.flags.c_contiguous
        and wind_index >= 3
        and wind_index % 2 == 1
        and fld.shape[1] >= wind_index
        and np.isfinite(fld).all()
    ):
        return None

    kernel = _rust_kernel("_texture_along_ray_dense_f64")
    if kernel is None:
        return None
    return np.ma.array(kernel(fld, wind_index))


def angular_texture_2d(image, N, interval):
    """
    Compute the angular texture of an image. Uses convolutions
    in order to speed up texture calculation by a factor of ~50
    compared to using ndimage.generic_filter.

    Parameters
    ----------
    image : 2D array of floats
        The array containing the velocities in which to calculate
        texture from.
    N : int or 2-element tuple
        If int, this is the window size for calculating texture. The
        texture will be calculated from an N by N window centered
        around the gate. If tuple N defines the m x n dimensions of
        the window centered around the gate.
    interval : float
        The absolute value of the maximum velocity. In conversion to
        radial coordinates, pi will be defined to be interval
        and -pi will be -interval. It is recommended that interval be
        set to the Nyquist velocity.

    Returns
    -------
    std_dev : float array
        Texture of the radial velocity field.

    """
    # Set N as a tuple if input is int
    if isinstance(N, int):
        N = (N, N)

    kernel = _rust_kernel("_angular_texture_2d")
    can_use_rust, rust_window = _can_use_rust(image, N)
    if kernel is not None and can_use_rust:
        return kernel(image, rust_window[0], rust_window[1], float(interval))

    # transform distribution from original interval to [-pi, pi]
    interval_max = interval
    interval_min = -interval
    half_width = (interval_max - interval_min) / 2.0
    center = interval_min + half_width

    # Calculate parameters needed for angular std. dev
    im = (np.asarray(image) - center) / (half_width) * np.pi
    x = np.cos(im)
    y = np.sin(im)

    # Calculate convolution
    kernel = np.ones(N)
    xs = signal.convolve2d(x, kernel, mode="same", boundary="symm")
    ys = signal.convolve2d(y, kernel, mode="same", boundary="symm")
    ns = np.prod(N)

    # Calculate norm over specified window
    xmean = xs / ns
    ymean = ys / ns
    norm = np.sqrt(xmean**2 + ymean**2)
    std_dev = np.sqrt(-2 * np.log(norm)) * (half_width) / np.pi
    return std_dev


def rolling_window(a, window):
    """Create a rolling window object for application of functions
    eg: result=np.ma.std(array, 11), 1)."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def texture(radar, var):
    """Determine a texture field using an 11pt stdev
    texarray=texture(pyradarobj, field)."""
    fld = radar.fields[var]["data"]
    print(fld.shape)
    tex = np.ma.zeros(fld.shape)
    for timestep in range(tex.shape[0]):
        ray = np.ma.std(rolling_window(fld[timestep, :], 11), 1)
        tex[timestep, 5:-5] = ray
        tex[timestep, 0:4] = np.ones(4) * ray[0]
        tex[timestep, -5:] = np.ones(5) * ray[-1]
    return tex


def texture_along_ray(radar, var, wind_size=7):
    """
    Compute field texture along ray using a user specified
    window size.

    Parameters
    ----------
    radar : radar object
        The radar object where the field is.
    var : str
        Name of the field which texture has to be computed.
    wind_size : int, optional
        Optional. Size of the rolling window used.

    Returns
    -------
    tex : radar field
        The texture of the specified field.

    """
    half_wind = int((wind_size - 1) / 2)
    fld = radar.fields[var]["data"]
    rust_tex = _texture_along_ray_rust(fld, wind_size)
    if rust_tex is not None:
        return rust_tex

    tex = np.ma.zeros(fld.shape)
    for timestep in range(tex.shape[0]):
        ray = np.ma.std(rolling_window(fld[timestep, :], wind_size), 1)
        tex[timestep, half_wind:-half_wind] = ray
        tex[timestep, 0:half_wind] = np.ones(half_wind) * ray[0]
        tex[timestep, -half_wind:] = np.ones(half_wind) * ray[-1]
    return tex
