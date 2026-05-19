"""Module containing calculations for spectra moments."""

import numpy as np

from ..config import get_metadata
from .._rust_bridge import get_rust_module
from ..util.hildebrand_sekhon import estimate_noise_hs74


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _can_use_rust_spectra_limits(the_spectra):
    return (
        type(the_spectra) is np.ndarray
        and the_spectra.ndim == 2
        and the_spectra.dtype == np.float64
        and not np.ma.isMaskedArray(the_spectra)
    )


def _can_use_rust_peak_limits(the_spectra):
    return (
        _can_use_rust_spectra_limits(the_spectra)
        and the_spectra.flags.c_contiguous
    )


def _finite_real_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _can_use_rust_reflectivity(spectra, bins, wavelength):
    if not (
        type(spectra) is np.ndarray
        and type(bins) is np.ndarray
        and spectra.ndim in (2, 3)
        and bins.ndim == 1
        and spectra.dtype == np.float64
        and bins.dtype == np.float64
        and spectra.flags.c_contiguous
        and bins.flags.c_contiguous
        and spectra.shape[-1] == bins.size
        and bins.size >= 2
        and not np.ma.isMaskedArray(spectra)
        and not np.ma.isMaskedArray(bins)
        and np.isfinite(bins).all()
    ):
        return None
    wavelength = _finite_real_scalar(wavelength)
    if wavelength is None or abs(wavelength) < 1.0e-3 or abs(wavelength) > 1.0e6:
        return None
    diffs = np.diff(bins)
    if not (np.all(diffs >= 1.0e-6) and np.all(diffs <= 1.0e9)):
        return None
    finite = np.isfinite(spectra)
    if not np.logical_or(finite, np.isnan(spectra)).all():
        return None
    finite_values = spectra[finite]
    if finite_values.size == 0 or not (
        np.all(finite_values >= -300.0) and np.all(finite_values <= 1000.0)
    ):
        return None
    rows = finite.reshape(-1, finite.shape[-1])
    if not np.all(np.any(rows[:, :-1] & rows[:, 1:], axis=1)):
        return None
    return wavelength


def _can_use_rust_mean_velocity(spectra, bins, wavelength, ref):
    wavelength = _can_use_rust_reflectivity(spectra, bins, wavelength)
    if wavelength is None:
        return None
    if not (
        type(ref) is np.ndarray
        and ref.dtype == np.float64
        and ref.shape == spectra.shape[:-1]
        and ref.flags.c_contiguous
        and not np.ma.isMaskedArray(ref)
        and np.isfinite(ref).all()
        and (np.abs(ref) <= 1000.0).all()
    ):
        return None
    return wavelength


def _can_use_rust_spectral_width(spectra, bins, wavelength, ref, mean_vel):
    wavelength = _can_use_rust_mean_velocity(spectra, bins, wavelength, ref)
    if wavelength is None:
        return None
    if not (
        type(mean_vel) is np.ndarray
        and mean_vel.dtype == np.float64
        and mean_vel.shape == spectra.shape[:-1]
        and mean_vel.flags.c_contiguous
        and not np.ma.isMaskedArray(mean_vel)
        and np.isfinite(mean_vel).all()
        and (np.abs(mean_vel) <= 1.0e9).all()
    ):
        return None
    return wavelength


def _can_use_rust_spectral_shape_moment(
    spectra, bins, wavelength, ref, mean_vel, spec_width
):
    wavelength = _can_use_rust_spectral_width(
        spectra, bins, wavelength, ref, mean_vel
    )
    if wavelength is None:
        return None
    if not (
        type(spec_width) is np.ndarray
        and spec_width.dtype == np.float64
        and spec_width.shape == spectra.shape[:-1]
        and spec_width.flags.c_contiguous
        and not np.ma.isMaskedArray(spec_width)
        and np.isfinite(spec_width).all()
        and (spec_width > 0.0).all()
        and (np.abs(spec_width) <= 1.0e9).all()
    ):
        return None
    return wavelength


def spectra_moments(radar):
    """Retrieves the radar moments using a spectra radar object.

    Parameter
    ---------
    radar : RadarSpectra
        Radar spectra object to use for the calculations.

    Returns
    -------
    fields : dict
        Field dictionaries containing moment data.

    """
    field_list = {}
    times = len(radar.time.values)
    rng = len(radar.range.values)
    ref = np.zeros((times, rng))
    vel = np.zeros((times, rng))
    spec_width = np.zeros((times, rng))
    skew = np.zeros((times, rng))
    kurt = np.zeros((times, rng))
    spectra = radar.ds.spectra.values
    velocity_bins = radar.velocity_bins.values
    d_spectra = np.zeros((times, rng, len(velocity_bins) * 3))
    print(spectra.shape)
    wavelength = radar.ds.attrs["wavelength"]
    for i in range(times):
        if i % 100 == 0:
            print(f"Dealiasing {i}/{times}")
        spectra_idx = spectra[i, :, :]
        # Get the raw spectra, bins, and wavelength
        # Subtract the noise floor from the spectra. Also, return the integration limits
        noise_floor, left_limit, right_limit, the_spectra = _get_noise_floor_and_limits(
            spectra_idx, avg_window=5
        )
        d_spec, dealiased_bins = dealias_spectra(
            the_spectra, velocity_bins, wavelength, left_limit, right_limit
        )
        noise_floor = np.tile(noise_floor, (d_spec.shape[1], 1)).T
        d_spectra[i, :, :] = np.where(d_spec < noise_floor, np.nan, d_spec)
        del the_spectra, d_spec

    ref_dict = get_metadata("reflectivity")
    ref = _get_reflectivity(d_spectra, dealiased_bins, wavelength)
    ref_dict["data"] = ref
    field_list["reflectivity"] = ref_dict

    vel_dict = get_metadata("velocity")
    vel = _get_mean_velocity(d_spectra, dealiased_bins, wavelength, ref)
    vel_dict["data"] = vel
    field_list["velocity"] = vel_dict

    spec_dict = get_metadata("spectrum_width")
    spec_width = _get_spectral_width(d_spectra, dealiased_bins, wavelength, ref, vel)
    spec_dict["data"] = spec_width
    field_list["spectrum_width"] = spec_dict

    skew_dict = {"long_name": "skewness", "standard_name": "skewness"}
    skew = _get_skewness(d_spectra, dealiased_bins, wavelength, ref, vel, spec_width)
    skew_dict["data"] = skew
    skew_dict["coordinates"] = "elevation azimuth range"
    field_list["skewness"] = skew_dict

    kurt_dict = {"long_name": "kurtosis", "standard_name": "kurtosis"}
    kurt = _get_kurtosis(d_spectra, dealiased_bins, wavelength, ref, vel, spec_width)
    kurt_dict["data"] = kurt
    kurt_dict["coordinates"] = "elevation azimuth range"
    field_list["kurtosis"] = kurt_dict

    return field_list


def dealias_spectra(the_spectra, vel_bins, wavelength, left_limit, right_limit):
    """Dealias a spectra.

    Parameters
    ----------
    the_spectra : array
        Spectra field data to dealias.
    vel_bins : array
        Velocity bin data.
    wavelength : float
        Spectra radar wavelength.

    Returns
    -------
    new_spectra : array
        Dealiased spectra array.
    new_bins : array
        New velocity bins from dealiased spectra.

    """
    # Calculate mean of gate before to decide whether to dealias left
    # or right side of spectrum - continuity check!
    ref = _get_reflectivity(the_spectra, vel_bins, wavelength)
    mean_vel = _get_mean_velocity(the_spectra, vel_bins, wavelength, ref)
    mean_vel = np.array(mean_vel)
    new_bins = np.concatenate(
        [vel_bins - 2 * vel_bins[-1], vel_bins, vel_bins + 2 * vel_bins[-1]]
    )
    # Expand interval to -3Vn, 3Vn
    n_pts = len(vel_bins)
    new_spectra = np.nan * np.ones((the_spectra.shape[0], n_pts * 3))
    dealiased_already = np.zeros(the_spectra.shape[0])
    # Test for aliasing - look for tail on both right interval and left interval
    for i in range(the_spectra.shape[0]):
        # First test to dealias: peaks on both sides
        # Second test: Discontinuity in velocities
        if i > 1:
            second_vel = mean_vel[i - 1]
        else:
            second_vel = mean_vel[i]
        if np.isfinite(the_spectra[i, 0]) and np.isfinite(the_spectra[i, -1]):
            noise_region = np.where(np.isnan(the_spectra[i]))[0]
            if second_vel < 0:
                right_tail_len = int(the_spectra.shape[1] - noise_region[-1])
                new_spectra[i, n_pts - right_tail_len : n_pts] = the_spectra[
                    i, n_pts - right_tail_len : n_pts
                ]
                new_spectra[i, n_pts : 2 * n_pts - right_tail_len] = the_spectra[
                    i, 0 : n_pts - right_tail_len
                ]
            else:
                left_tail_len = int(noise_region[0])
                new_spectra[i, 2 * n_pts : 2 * n_pts + left_tail_len] = the_spectra[
                    i, 0:left_tail_len
                ]
                new_spectra[i, n_pts + left_tail_len : 2 * n_pts] = the_spectra[
                    i, left_tail_len:
                ]
            dealiased_already[i] = 1

    # Do a second check for continuity
    mean_vel = _get_mean_velocity(new_spectra, new_bins, wavelength, ref)

    for i in range(new_spectra.shape[0]):
        # First test to dealias: peaks on both sides
        # Second test: Discontinuity in velocities
        if i > 1:
            second_vel = mean_vel[i - 1]
        else:
            second_vel = mean_vel[i]
        # Discontinuity = more than 1 Nyquist switch in mean velocity
        if abs(mean_vel[i] - second_vel) > vel_bins[-1] and dealiased_already[i] == 0:
            noise_region = np.where(np.isnan(the_spectra[i]))[0]
            if mean_vel[i] > 0:
                right_tail_len = int(the_spectra.shape[1] - noise_region[-1])
                new_spectra[i, n_pts - right_tail_len : n_pts] = the_spectra[
                    i, n_pts - right_tail_len : n_pts
                ]
                new_spectra[i, n_pts : 2 * n_pts - right_tail_len] = the_spectra[
                    i, 0 : n_pts - right_tail_len
                ]
            else:
                left_tail_len = int(noise_region[0])
                new_spectra[i, 2 * n_pts : 2 * n_pts + left_tail_len] = the_spectra[
                    i, 0:left_tail_len
                ]
                new_spectra[i, n_pts + left_tail_len : 2 * n_pts] = the_spectra[
                    i, left_tail_len:
                ]
            mean_vel[i] = _get_mean_velocity(
                new_spectra[i], new_bins, wavelength, ref[i]
            )
            dealiased_already[i] = 1
        elif dealiased_already[i] == 0:
            new_spectra[i, n_pts : 2 * n_pts] = the_spectra[i]
    return new_spectra, new_bins


def _get_limits_dealiased_spectra(the_spectra):
    """Calculates limits for a dealiased spectra."""
    kernel = _rust_kernel("_spectra_limits_dealiased")
    if kernel is not None and _can_use_rust_spectra_limits(the_spectra):
        return kernel(the_spectra)

    left = np.zeros(the_spectra.shape[0])
    right = np.zeros(the_spectra.shape[0])
    new_spec = the_spectra.copy()
    for i in range(the_spectra.shape[0]):
        try:
            peak = np.nanargmax(the_spectra[i])
            j = peak
            while j > 0 and np.isfinite(the_spectra[i, j]):
                j = j - 1
            left[i] = j
            j = peak
            while j < the_spectra.shape[1] - 1 and np.isfinite(the_spectra[i, j]):
                j = j + 1
            right[i] = j
            new_spec[i, 0 : int(left[i]) - 1] = np.nan
            new_spec[i, int(right[i]) + 1 : -1] = np.nan
        except ValueError:
            left[i] = np.nan
            right[i] = np.nan
    return left, right, new_spec


def _get_reflectivity(spectra, bins, wavelength):
    """Calculates reflectivity from a RadarSpectra object."""
    kernel = _rust_kernel("_spectra_reflectivity_dense")
    wavelength_scalar = _can_use_rust_reflectivity(spectra, bins, wavelength)
    if kernel is not None and wavelength_scalar is not None:
        return kernel(spectra, bins, wavelength_scalar)

    spectra_linear = 10 ** (spectra / 10)
    radar_constant = 1e18 * wavelength**4 / (0.93 * np.pi**5)
    if len(spectra_linear.shape) == 2:
        spec_med = radar_constant * (spectra_linear[:, :-1] + spectra_linear[:, 1:]) / 2
        diffs = np.tile(np.diff(bins), (spectra.shape[0], 1))
    elif len(spectra_linear.shape) == 3:
        spec_med = (
            radar_constant * (spectra_linear[:, :, :-1] + spectra_linear[:, :, 1:]) / 2
        )
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
    else:
        spec_med = radar_constant * (spectra_linear[:-1] + spectra_linear[1:]) / 2
        diffs = np.diff(bins)
    ref = np.nansum(spec_med * diffs, axis=-1)
    return 10 * np.log10(ref)


def _get_mean_velocity(spectra, bins, wavelength, ref):
    """Calculates mean velocity from a RadarSpectra object."""
    kernel = _rust_kernel("_spectra_mean_velocity_dense")
    wavelength_scalar = _can_use_rust_mean_velocity(spectra, bins, wavelength, ref)
    if kernel is not None and wavelength_scalar is not None:
        return kernel(spectra, bins, wavelength_scalar, ref)

    spectra_linear = 10 ** (spectra / 10)
    radar_constant = 1e18 * wavelength**4 / (0.93 * np.pi**5)
    ref = 10 ** (ref / 10)
    if len(spectra_linear.shape) == 2:
        spec_med = radar_constant * (spectra_linear[:, :-1] + spectra_linear[:, 1:]) / 2
        diffs = np.tile(np.diff(bins), (spectra.shape[0], 1))
    elif len(spectra_linear.shape) == 3:
        spec_med = (
            radar_constant * (spectra_linear[:, :, :-1] + spectra_linear[:, :, 1:]) / 2
        )
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
    else:
        spec_med = radar_constant * (spectra_linear[:-1] + spectra_linear[1:]) / 2
        diffs = np.diff(bins)
    bins_med = (bins[:-1] + bins[1:]) / 2.0
    if len(spectra_linear.shape) == 2:
        bins_med = np.tile(bins_med, (spectra.shape[0], 1))
    elif len(spectra_linear.shape) == 3:
        bins_med = np.tile(bins_med, (spectra.shape[0], spectra.shape[1], 1))
    mean_vel = np.nansum(spec_med * bins_med * diffs, axis=-1) / ref
    return mean_vel


def _get_spectral_width(spectra, bins, wavelength, ref, mean_vel):
    """Calculates reflectivity from a RadarSpectra object."""
    kernel = _rust_kernel("_spectra_spectral_width_dense")
    wavelength_scalar = _can_use_rust_spectral_width(
        spectra, bins, wavelength, ref, mean_vel
    )
    if kernel is not None and wavelength_scalar is not None:
        return kernel(spectra, bins, wavelength_scalar, ref, mean_vel)

    spectra_linear = 10 ** (spectra / 10)
    radar_constant = 1e18 * wavelength**4 / (0.93 * np.pi**5)
    ref = 10 ** (ref / 10)
    if len(spectra_linear.shape) == 2:
        spec_med = radar_constant * (spectra_linear[:, :-1] + spectra_linear[:, 1:]) / 2
        diffs = np.tile(np.diff(bins), (spectra.shape[0], 1))
        mean_vel = np.tile(mean_vel, (len(bins) - 1, 1)).T
    elif len(spectra_linear.shape) == 3:
        spec_med = (
            radar_constant * (spectra_linear[:, :, :-1] + spectra_linear[:, :, 1:]) / 2
        )
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
        mean_vel = np.tile(mean_vel.T, (len(bins) - 1, 1, 1)).T
    else:
        spec_med = radar_constant * (spectra_linear[:-1] + spectra_linear[1:]) / 2
        diffs = np.diff(bins)
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
    bins_med = (bins[:-1] + bins[1:]) / 2.0
    if len(spectra_linear.shape) == 2:
        bins_med = np.tile(bins_med, (spectra.shape[0], 1))
    elif len(spectra_linear.shape) == 3:
        bins_med = np.tile(bins_med, (spectra.shape[0], spectra.shape[1], 1))
    spec_wid = np.nansum(spec_med * (bins_med - mean_vel) ** 2 * diffs, axis=-1) / ref
    return np.sqrt(spec_wid)


def _get_skewness(spectra, bins, wavelength, ref, mean_vel, spec_width):
    """Calculates skewness from a RadarSpectra object."""
    kernel = _rust_kernel("_spectra_skewness_dense")
    wavelength_scalar = _can_use_rust_spectral_shape_moment(
        spectra, bins, wavelength, ref, mean_vel, spec_width
    )
    if kernel is not None and wavelength_scalar is not None:
        return kernel(spectra, bins, wavelength_scalar, ref, mean_vel, spec_width)

    spectra_linear = 10 ** (spectra / 10)
    radar_constant = 1e18 * wavelength**4 / (0.93 * np.pi**5)
    ref = 10 ** (ref / 10)
    if len(spectra_linear.shape) == 2:
        spec_med = radar_constant * (spectra_linear[:, :-1] + spectra_linear[:, 1:]) / 2
        diffs = np.tile(np.diff(bins), (spectra.shape[0], 1))
        mean_vel = np.tile(mean_vel, (len(bins) - 1, 1)).T
    elif len(spectra_linear.shape) == 3:
        spec_med = (
            radar_constant * (spectra_linear[:, :, :-1] + spectra_linear[:, :, 1:]) / 2
        )
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
        mean_vel = np.tile(mean_vel.T, (len(bins) - 1, 1, 1)).T
    else:
        spec_med = radar_constant * (spectra_linear[:-1] + spectra_linear[1:]) / 2
        diffs = np.diff(bins)
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
    bins_med = (bins[:-1] + bins[1:]) / 2.0
    if len(spectra_linear.shape) == 2:
        bins_med = np.tile(bins_med, (spectra.shape[0], 1))
    elif len(spectra_linear.shape) == 3:
        bins_med = np.tile(bins_med, (spectra.shape[0], spectra.shape[1], 1))
    skew = np.nansum(spec_med * (bins_med - mean_vel) ** 3 * diffs, axis=-1) / ref
    return skew / spec_width**3


def _get_kurtosis(spectra, bins, wavelength, ref, mean_vel, spec_width):
    """Calculates a Kurtosis field from a RadarSpectra object."""
    kernel = _rust_kernel("_spectra_kurtosis_dense")
    wavelength_scalar = _can_use_rust_spectral_shape_moment(
        spectra, bins, wavelength, ref, mean_vel, spec_width
    )
    if kernel is not None and wavelength_scalar is not None:
        return kernel(spectra, bins, wavelength_scalar, ref, mean_vel, spec_width)

    spectra_linear = 10 ** (spectra / 10)
    radar_constant = 1e18 * wavelength**4 / (0.93 * np.pi**5)
    ref = 10 ** (ref / 10)
    if len(spectra_linear.shape) == 2:
        spec_med = radar_constant * (spectra_linear[:, :-1] + spectra_linear[:, 1:]) / 2
        diffs = np.tile(np.diff(bins), (spectra.shape[0], 1))
        mean_vel = np.tile(mean_vel, (len(bins) - 1, 1)).T
    elif len(spectra_linear.shape) == 3:
        spec_med = (
            radar_constant * (spectra_linear[:, :, :-1] + spectra_linear[:, :, 1:]) / 2
        )
        diffs = np.tile(np.diff(bins), (spectra.shape[0], spectra.shape[1], 1))
        mean_vel = np.tile(mean_vel.T, (len(bins) - 1, 1, 1)).T
    else:
        spec_med = radar_constant * (spectra_linear[:-1] + spectra_linear[1:]) / 2
        diffs = np.diff(bins)
    bins_med = (bins[:-1] + bins[1:]) / 2.0
    if len(spectra_linear.shape) == 2:
        bins_med = np.tile(bins_med, (spectra.shape[0], 1))
    elif len(spectra_linear.shape) == 3:
        bins_med = np.tile(bins_med, (spectra.shape[0], spectra.shape[1], 1))
    kurt = np.nansum(spec_med * (bins_med - mean_vel) ** 4 * diffs, axis=-1) / ref
    return kurt / spec_width**4


def _get_noise_floor_and_limits(the_spectra, avg_window=1):
    """Calculates noise floor and limits from a RadarSpectra object."""
    lin_spectra = 10 ** (the_spectra / 10.0)
    if avg_window > 1:
        for i in range(lin_spectra.shape[0]):
            lin_spectra[i] = np.convolve(
                lin_spectra[i], np.ones(avg_window) / avg_window, mode="same"
            )
    noise_floor_thresh = np.zeros((lin_spectra.shape[0],))

    for i in range(lin_spectra.shape[0]):
        noise_floor = estimate_noise_hs74(lin_spectra[i], navg=avg_window)
        noise_floor_thresh[i] = 10 * np.log10(noise_floor[0])
        the_spectra[i] = 10 * np.log10(lin_spectra[i] - noise_floor[0])
    the_spectra[lin_spectra < 0] = np.nan
    left, right = _get_spectra_peak_limits(the_spectra)
    spectra = np.ma.masked_where(np.isnan(the_spectra), the_spectra)
    return noise_floor_thresh, left, right, spectra


def _get_spectra_peak_limits(the_spectra):
    kernel = _rust_kernel("_spectra_peak_limits")
    if kernel is not None and _can_use_rust_peak_limits(the_spectra):
        return kernel(the_spectra)

    left = np.nan * np.ones(the_spectra.shape[0])
    right = np.nan * np.ones(the_spectra.shape[0])
    for i in range(the_spectra.shape[0]):
        try:
            peak_loc = np.nanargmax(the_spectra[i])
            j = peak_loc
            while np.isfinite(the_spectra[i, j]) and j > 0:
                j = j - 1
            left[i] = j
            j = peak_loc
            while np.isfinite(the_spectra[i, j]) and j < the_spectra.shape[1] - 1:
                j = j + 1
            right[i] = j
        except ValueError:
            left[i] = np.nan
            right[i] = np.nan
    return left, right
