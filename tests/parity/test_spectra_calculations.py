import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import spectra_calculations  # noqa: E402


def _reference_peak_limits(spectra):
    left = np.nan * np.ones(spectra.shape[0])
    right = np.nan * np.ones(spectra.shape[0])
    for i in range(spectra.shape[0]):
        try:
            peak_loc = np.nanargmax(spectra[i])
            j = peak_loc
            while np.isfinite(spectra[i, j]) and j > 0:
                j = j - 1
            left[i] = j
            j = peak_loc
            while np.isfinite(spectra[i, j]) and j < spectra.shape[1] - 1:
                j = j + 1
            right[i] = j
        except ValueError:
            left[i] = np.nan
            right[i] = np.nan
    return left, right


def _fallback_limits(spectra, monkeypatch):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_limits_dealiased_spectra(spectra)


def _fallback_peak_limits(spectra, monkeypatch):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_spectra_peak_limits(spectra)


def _fallback_reflectivity(spectra, bins, wavelength, monkeypatch):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_reflectivity(spectra, bins, wavelength)


def _fallback_mean_velocity(spectra, bins, wavelength, ref, monkeypatch):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_mean_velocity(spectra, bins, wavelength, ref)


def _fallback_spectral_width(spectra, bins, wavelength, ref, mean_vel, monkeypatch):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_spectral_width(
        spectra, bins, wavelength, ref, mean_vel
    )


def _fallback_skewness(
    spectra, bins, wavelength, ref, mean_vel, spec_width, monkeypatch
):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_skewness(
        spectra, bins, wavelength, ref, mean_vel, spec_width
    )


def _fallback_kurtosis(
    spectra, bins, wavelength, ref, mean_vel, spec_width, monkeypatch
):
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    return spectra_calculations._get_kurtosis(
        spectra, bins, wavelength, ref, mean_vel, spec_width
    )


def _assert_limits_equal(actual, expected):
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_array_equal(actual_part, expected_part)


def _assert_spectra_float_close(actual, expected, atol=1.0e-12):
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~(np.isnan(actual) | np.isnan(expected))
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=atol)


def test_spectra_limits_python_fallback_preserves_oracle_nan_slices(monkeypatch):
    spectra = np.array(
        [
            [5.0, 4.0, np.nan, 1.0],
            [np.nan, 1.0, 3.0, 2.0],
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, np.inf, 0.0, -np.inf],
        ],
        dtype=np.float64,
    )

    actual = _fallback_limits(spectra, monkeypatch)
    expected_left = np.array([0.0, 0.0, np.nan, 1.0])
    expected_right = np.array([2.0, 3.0, np.nan, 1.0])
    expected_spectra = np.array(
        [
            [np.nan, np.nan, np.nan, 1.0],
            [np.nan, np.nan, np.nan, 2.0],
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, np.inf, np.nan, -np.inf],
        ],
        dtype=np.float64,
    )

    _assert_limits_equal(actual, (expected_left, expected_right, expected_spectra))
    np.testing.assert_array_equal(spectra[0], np.array([5.0, 4.0, np.nan, 1.0]))


def test_spectra_peak_limits_python_fallback_matches_oracle_loop(monkeypatch):
    spectra = np.array(
        [
            [1.0, 3.0, 3.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, np.inf, 1.0, -np.inf],
        ],
        dtype=np.float64,
    )

    actual = _fallback_peak_limits(spectra, monkeypatch)
    expected = _reference_peak_limits(spectra)

    _assert_limits_equal(actual, expected)


def test_spectra_reflectivity_python_fallback_preserves_shapes(monkeypatch):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra_3d = np.stack([spectra_2d, spectra_2d + 1.0])

    out_2d = _fallback_reflectivity(spectra_2d, bins, 0.1, monkeypatch)
    out_3d = _fallback_reflectivity(spectra_3d, bins, 0.1, monkeypatch)

    assert out_2d.dtype == np.float64
    assert out_2d.shape == (2,)
    assert out_3d.dtype == np.float64
    assert out_3d.shape == (2, 2)


def test_spectra_reflectivity_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(spectra, bins, wavelength):
        calls.append((spectra.dtype, spectra.shape, bins.shape, wavelength))
        return np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_reflectivity_dense" else None,
    )
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    actual = spectra_calculations._get_reflectivity(spectra, bins, np.float64(0.1))

    np.testing.assert_array_equal(actual, np.array([1.0, 2.0], dtype=np.float64))
    assert calls == [(np.float64, (2, 3), (3,), 0.1)]


def test_spectra_mean_velocity_python_fallback_preserves_shapes(monkeypatch):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra_3d = np.stack([spectra_2d, spectra_2d + 1.0])
    ref_2d = _fallback_reflectivity(spectra_2d, bins, 0.1, monkeypatch)
    ref_3d = _fallback_reflectivity(spectra_3d, bins, 0.1, monkeypatch)

    out_2d = _fallback_mean_velocity(spectra_2d, bins, 0.1, ref_2d, monkeypatch)
    out_3d = _fallback_mean_velocity(spectra_3d, bins, 0.1, ref_3d, monkeypatch)

    assert out_2d.dtype == np.float64
    assert out_2d.shape == (2,)
    assert out_3d.dtype == np.float64
    assert out_3d.shape == (2, 2)


def test_spectra_mean_velocity_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(spectra, bins, wavelength, ref):
        calls.append((spectra.dtype, spectra.shape, bins.shape, wavelength, ref.shape))
        return np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_mean_velocity_dense" else None,
    )
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array([10.0, 11.0], dtype=np.float64)

    actual = spectra_calculations._get_mean_velocity(spectra, bins, np.float64(0.1), ref)

    np.testing.assert_array_equal(actual, np.array([1.0, 2.0], dtype=np.float64))
    assert calls == [(np.float64, (2, 3), (3,), 0.1, (2,))]


def test_spectral_width_python_fallback_preserves_shapes(monkeypatch):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra_3d = np.stack([spectra_2d, spectra_2d + 1.0])
    ref_2d = _fallback_reflectivity(spectra_2d, bins, 0.1, monkeypatch)
    ref_3d = _fallback_reflectivity(spectra_3d, bins, 0.1, monkeypatch)
    mean_vel_2d = _fallback_mean_velocity(spectra_2d, bins, 0.1, ref_2d, monkeypatch)
    mean_vel_3d = _fallback_mean_velocity(spectra_3d, bins, 0.1, ref_3d, monkeypatch)

    out_2d = _fallback_spectral_width(
        spectra_2d, bins, 0.1, ref_2d, mean_vel_2d, monkeypatch
    )
    out_3d = _fallback_spectral_width(
        spectra_3d, bins, 0.1, ref_3d, mean_vel_3d, monkeypatch
    )

    assert out_2d.dtype == np.float64
    assert out_2d.shape == (2,)
    assert out_3d.dtype == np.float64
    assert out_3d.shape == (2, 2)


def test_spectral_width_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(spectra, bins, wavelength, ref, mean_vel):
        calls.append(
            (spectra.dtype, spectra.shape, bins.shape, wavelength, ref.shape, mean_vel.shape)
        )
        return np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_spectral_width_dense" else None,
    )
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array([10.0, 11.0], dtype=np.float64)
    mean_vel = np.array([0.4, 0.6], dtype=np.float64)

    actual = spectra_calculations._get_spectral_width(
        spectra, bins, np.float64(0.1), ref, mean_vel
    )

    np.testing.assert_array_equal(actual, np.array([1.0, 2.0], dtype=np.float64))
    assert calls == [(np.float64, (2, 3), (3,), 0.1, (2,), (2,))]


def test_skewness_python_fallback_preserves_shapes(monkeypatch):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra_3d = np.stack([spectra_2d, spectra_2d + 1.0])
    ref_2d = _fallback_reflectivity(spectra_2d, bins, 0.1, monkeypatch)
    ref_3d = _fallback_reflectivity(spectra_3d, bins, 0.1, monkeypatch)
    mean_vel_2d = _fallback_mean_velocity(spectra_2d, bins, 0.1, ref_2d, monkeypatch)
    mean_vel_3d = _fallback_mean_velocity(spectra_3d, bins, 0.1, ref_3d, monkeypatch)
    width_2d = _fallback_spectral_width(
        spectra_2d, bins, 0.1, ref_2d, mean_vel_2d, monkeypatch
    )
    width_3d = _fallback_spectral_width(
        spectra_3d, bins, 0.1, ref_3d, mean_vel_3d, monkeypatch
    )

    out_2d = _fallback_skewness(
        spectra_2d, bins, 0.1, ref_2d, mean_vel_2d, width_2d, monkeypatch
    )
    out_3d = _fallback_skewness(
        spectra_3d, bins, 0.1, ref_3d, mean_vel_3d, width_3d, monkeypatch
    )

    assert out_2d.dtype == np.float64
    assert out_2d.shape == (2,)
    assert out_3d.dtype == np.float64
    assert out_3d.shape == (2, 2)


def test_skewness_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(spectra, bins, wavelength, ref, mean_vel, spec_width):
        calls.append(
            (
                spectra.dtype,
                spectra.shape,
                bins.shape,
                wavelength,
                ref.shape,
                mean_vel.shape,
                spec_width.shape,
            )
        )
        return np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_skewness_dense" else None,
    )
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array([10.0, 11.0], dtype=np.float64)
    mean_vel = np.array([0.4, 0.6], dtype=np.float64)
    spec_width = np.array([1.0, 1.1], dtype=np.float64)

    actual = spectra_calculations._get_skewness(
        spectra, bins, np.float64(0.1), ref, mean_vel, spec_width
    )

    np.testing.assert_array_equal(actual, np.array([1.0, 2.0], dtype=np.float64))
    assert calls == [(np.float64, (2, 3), (3,), 0.1, (2,), (2,), (2,))]


def test_kurtosis_python_fallback_preserves_shapes(monkeypatch):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra_3d = np.stack([spectra_2d, spectra_2d + 1.0])
    ref_2d = _fallback_reflectivity(spectra_2d, bins, 0.1, monkeypatch)
    ref_3d = _fallback_reflectivity(spectra_3d, bins, 0.1, monkeypatch)
    mean_vel_2d = _fallback_mean_velocity(spectra_2d, bins, 0.1, ref_2d, monkeypatch)
    mean_vel_3d = _fallback_mean_velocity(spectra_3d, bins, 0.1, ref_3d, monkeypatch)
    width_2d = _fallback_spectral_width(
        spectra_2d, bins, 0.1, ref_2d, mean_vel_2d, monkeypatch
    )
    width_3d = _fallback_spectral_width(
        spectra_3d, bins, 0.1, ref_3d, mean_vel_3d, monkeypatch
    )

    out_2d = _fallback_kurtosis(
        spectra_2d, bins, 0.1, ref_2d, mean_vel_2d, width_2d, monkeypatch
    )
    out_3d = _fallback_kurtosis(
        spectra_3d, bins, 0.1, ref_3d, mean_vel_3d, width_3d, monkeypatch
    )

    assert out_2d.dtype == np.float64
    assert out_2d.shape == (2,)
    assert out_3d.dtype == np.float64
    assert out_3d.shape == (2, 2)


def test_kurtosis_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(spectra, bins, wavelength, ref, mean_vel, spec_width):
        calls.append(
            (
                spectra.dtype,
                spectra.shape,
                bins.shape,
                wavelength,
                ref.shape,
                mean_vel.shape,
                spec_width.shape,
            )
        )
        return np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_kurtosis_dense" else None,
    )
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array([10.0, 11.0], dtype=np.float64)
    mean_vel = np.array([0.4, 0.6], dtype=np.float64)
    spec_width = np.array([1.0, 1.1], dtype=np.float64)

    actual = spectra_calculations._get_kurtosis(
        spectra, bins, np.float64(0.1), ref, mean_vel, spec_width
    )

    np.testing.assert_array_equal(actual, np.array([1.0, 2.0], dtype=np.float64))
    assert calls == [(np.float64, (2, 3), (3,), 0.1, (2,), (2,), (2,))]


@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64)[:, ::-1],
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.ma.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.nan,
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([np.nan], dtype=np.float64),
        ),
    ],
)
def test_spectra_mean_velocity_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra, bins, wavelength, ref
):
    def fail_if_called(name):
        if name != "_spectra_mean_velocity_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported mean-velocity input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        try:
            actual = spectra_calculations._get_mean_velocity(
                spectra, bins, wavelength, ref
            )
        except Exception as actual_error:
            monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                spectra_calculations._get_mean_velocity(spectra, bins, wavelength, ref)
        else:
            expected = _fallback_mean_velocity(spectra, bins, wavelength, ref, monkeypatch)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "mean_vel"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            np.array(0.5, dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64)[:, ::-1],
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.ma.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.nan,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([np.nan], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([[0.5]], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
        ),
    ],
)
def test_spectral_width_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra, bins, wavelength, ref, mean_vel
):
    def fail_if_called(name):
        if name != "_spectra_spectral_width_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported spectral-width input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        try:
            actual = spectra_calculations._get_spectral_width(
                spectra, bins, wavelength, ref, mean_vel
            )
        except Exception as actual_error:
            monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                spectra_calculations._get_spectral_width(
                    spectra, bins, wavelength, ref, mean_vel
                )
        else:
            expected = _fallback_spectral_width(
                spectra, bins, wavelength, ref, mean_vel, monkeypatch
            )
            np.testing.assert_array_equal(actual, expected)


def test_spectral_width_1d_preserves_python_fallback_exception(monkeypatch):
    spectra = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array(1.0, dtype=np.float64)
    mean_vel = np.array(0.5, dtype=np.float64)

    def fail_if_called(name):
        if name != "_spectra_spectral_width_dense":
            return None

        def kernel(*_args):
            raise AssertionError("1D spectral-width input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)
    with np.errstate(all="ignore"):
        with pytest.raises(IndexError) as actual_error:
            spectra_calculations._get_spectral_width(
                spectra, bins, 0.1, ref, mean_vel
            )

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    with np.errstate(all="ignore"):
        with pytest.raises(IndexError) as expected_error:
            spectra_calculations._get_spectral_width(
                spectra, bins, 0.1, ref, mean_vel
            )
    assert actual_error.value.args == expected_error.value.args


@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "mean_vel", "spec_width"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            np.array(0.5, dtype=np.float64),
            np.array(1.0, dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64)[:, ::-1],
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.ma.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.nan,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([[0.5]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
        ),
    ],
)
def test_skewness_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra, bins, wavelength, ref, mean_vel, spec_width
):
    def fail_if_called(name):
        if name != "_spectra_skewness_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported skewness input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        try:
            actual = spectra_calculations._get_skewness(
                spectra, bins, wavelength, ref, mean_vel, spec_width
            )
        except Exception as actual_error:
            monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                spectra_calculations._get_skewness(
                    spectra, bins, wavelength, ref, mean_vel, spec_width
                )
        else:
            expected = _fallback_skewness(
                spectra, bins, wavelength, ref, mean_vel, spec_width, monkeypatch
            )
            np.testing.assert_array_equal(actual, expected)


def test_skewness_1d_preserves_python_fallback_exception(monkeypatch):
    spectra = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array(1.0, dtype=np.float64)
    mean_vel = np.array(0.5, dtype=np.float64)
    spec_width = np.array(1.0, dtype=np.float64)

    def fail_if_called(name):
        if name != "_spectra_skewness_dense":
            return None

        def kernel(*_args):
            raise AssertionError("1D skewness input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)
    with np.errstate(all="ignore"):
        with pytest.raises(IndexError) as actual_error:
            spectra_calculations._get_skewness(
                spectra, bins, 0.1, ref, mean_vel, spec_width
            )

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    with np.errstate(all="ignore"):
        with pytest.raises(IndexError) as expected_error:
            spectra_calculations._get_skewness(
                spectra, bins, 0.1, ref, mean_vel, spec_width
            )
    assert actual_error.value.args == expected_error.value.args


@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "mean_vel", "spec_width"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            np.array(0.5, dtype=np.float64),
            np.array(1.0, dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64)[:, ::-1],
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.ma.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.nan,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([[0.5]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
        ),
    ],
)
def test_kurtosis_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra, bins, wavelength, ref, mean_vel, spec_width
):
    def fail_if_called(name):
        if name != "_spectra_kurtosis_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported kurtosis input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        try:
            actual = spectra_calculations._get_kurtosis(
                spectra, bins, wavelength, ref, mean_vel, spec_width
            )
        except Exception as actual_error:
            monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                spectra_calculations._get_kurtosis(
                    spectra, bins, wavelength, ref, mean_vel, spec_width
                )
        else:
            expected = _fallback_kurtosis(
                spectra, bins, wavelength, ref, mean_vel, spec_width, monkeypatch
            )
            np.testing.assert_array_equal(actual, expected)


def test_kurtosis_1d_preserves_python_fallback_result(monkeypatch):
    spectra = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ref = np.array(1.0, dtype=np.float64)
    mean_vel = np.array(0.5, dtype=np.float64)
    spec_width = np.array(1.0, dtype=np.float64)

    def fail_if_called(name):
        if name != "_spectra_kurtosis_dense":
            return None

        def kernel(*_args):
            raise AssertionError("1D kurtosis input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)
    with np.errstate(all="ignore"):
        actual = spectra_calculations._get_kurtosis(
            spectra, bins, 0.1, ref, mean_vel, spec_width
        )

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    with np.errstate(all="ignore"):
        expected = spectra_calculations._get_kurtosis(
            spectra, bins, 0.1, ref, mean_vel, spec_width
        )
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64)[:, ::-1],
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.ma.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[np.nan, np.nan, np.nan]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.nan,
        ),
        (
            np.array([[-400.0, -399.0, -398.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0e-12, 2.0e-12], dtype=np.float64),
            0.1,
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            1.0e-6,
        ),
    ],
)
def test_spectra_reflectivity_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra, bins, wavelength
):
    def fail_if_called(name):
        if name != "_spectra_reflectivity_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported reflectivity input should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        try:
            actual = spectra_calculations._get_reflectivity(spectra, bins, wavelength)
        except Exception as actual_error:
            monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                spectra_calculations._get_reflectivity(spectra, bins, wavelength)
        else:
            expected = _fallback_reflectivity(spectra, bins, wavelength, monkeypatch)
            np.testing.assert_array_equal(actual, expected)


def test_spectra_peak_limits_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(spectra):
        calls.append((spectra.dtype, spectra.shape, spectra.flags.c_contiguous))
        return (
            np.array([3.0], dtype=np.float64),
            np.array([4.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_peak_limits" else None,
    )
    spectra = np.array([[1.0, 2.0, np.nan]], dtype=np.float64)

    actual = spectra_calculations._get_spectra_peak_limits(spectra)

    assert calls == [(np.float64, (1, 3), True)]
    _assert_limits_equal(
        actual,
        (
            np.array([3.0], dtype=np.float64),
            np.array([4.0], dtype=np.float64),
        ),
    )


@pytest.mark.parametrize(
    "spectra",
    [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)[:, ::-1],
        np.array([1.0, 2.0], dtype=np.float64),
        np.ma.array([[1.0, 2.0]], mask=[[False, True]], dtype=np.float64),
    ],
)
def test_spectra_peak_limits_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported spectra inputs should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    if np.ndim(spectra) != 2:
        with pytest.raises(Exception):
            spectra_calculations._get_spectra_peak_limits(spectra)
        return

    expected = _reference_peak_limits(spectra)
    actual = spectra_calculations._get_spectra_peak_limits(spectra)

    _assert_limits_equal(actual, expected)


def test_spectra_limits_dispatches_to_private_rust_kernel_for_float64_2d_arrays(
    monkeypatch,
):
    calls = []

    def rust_kernel(spectra):
        calls.append((spectra.dtype, spectra.shape))
        return (
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.full_like(spectra, 42.0),
        )

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_spectra_limits_dealiased" else None,
    )
    spectra = np.array([[1.0, 2.0, np.nan]], dtype=np.float64)

    actual = spectra_calculations._get_limits_dealiased_spectra(spectra)

    assert calls == [(np.float64, (1, 3))]
    _assert_limits_equal(
        actual,
        (
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.full_like(spectra, 42.0),
        ),
    )


@pytest.mark.parametrize(
    "spectra",
    [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float64),
        np.ma.array([[1.0, 2.0]], mask=[[False, True]], dtype=np.float64),
    ],
)
def test_spectra_limits_keeps_python_path_for_unsupported_inputs(
    monkeypatch, spectra
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported spectra inputs should use fallback")

        return kernel

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)

    if np.ndim(spectra) != 2:
        with pytest.raises(Exception):
            spectra_calculations._get_limits_dealiased_spectra(spectra)
        return

    expected = _fallback_limits(spectra, monkeypatch)
    monkeypatch.setattr(spectra_calculations, "_rust_kernel", fail_if_called)
    actual = spectra_calculations._get_limits_dealiased_spectra(spectra)

    _assert_limits_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_spectra_limits_match_python_fallback(monkeypatch):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    spectra = np.array(
        [
            [5.0, 4.0, np.nan, 1.0],
            [np.nan, 1.0, 3.0, 2.0],
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, np.inf, 0.0, -np.inf],
        ],
        dtype=np.float64,
    )

    expected = _fallback_limits(spectra, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = spectra_calculations._get_limits_dealiased_spectra(spectra)

    _assert_limits_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_spectra_limits_handles_empty_bin_axis(monkeypatch):
    spectra = np.empty((3, 0), dtype=np.float64)
    expected = _fallback_limits(spectra, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = spectra_calculations._get_limits_dealiased_spectra(spectra)

    _assert_limits_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_spectra_peak_limits_match_python_fallback(monkeypatch):
    spectra = np.array(
        [
            [1.0, 3.0, 3.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, np.inf, 1.0, -np.inf],
        ],
        dtype=np.float64,
    )
    expected = _fallback_peak_limits(spectra, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = spectra_calculations._get_spectra_peak_limits(spectra)

    _assert_limits_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ndim", [2, 3])
def test_real_rust_spectra_reflectivity_matches_python_fallback(monkeypatch, ndim):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra = spectra_2d if ndim == 2 else np.stack([spectra_2d, spectra_2d + 1.0])
    expected = _fallback_reflectivity(spectra, bins, 0.1, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = spectra_calculations._get_reflectivity(spectra, bins, 0.1)

    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ndim", [2, 3])
def test_real_rust_spectra_mean_velocity_matches_python_fallback(monkeypatch, ndim):
    bins = np.array([-2.0, -0.5, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra = spectra_2d if ndim == 2 else np.stack([spectra_2d, spectra_2d + 1.0])
    ref = _fallback_reflectivity(spectra, bins, 0.1, monkeypatch)
    expected = _fallback_mean_velocity(spectra, bins, 0.1, ref, monkeypatch)

    import pyart._rust as rust

    calls = []

    def counted_kernel(spectra_arg, bins_arg, wavelength_arg, ref_arg):
        calls.append((spectra_arg.shape, bins_arg.shape, wavelength_arg, ref_arg.shape))
        return rust._spectra_mean_velocity_dense(
            spectra_arg, bins_arg, wavelength_arg, ref_arg
        )

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_spectra_mean_velocity_dense" else None,
    )
    actual = spectra_calculations._get_mean_velocity(spectra, bins, 0.1, ref)

    assert calls == [(spectra.shape, bins.shape, 0.1, ref.shape)]
    _assert_spectra_float_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ndim", [2, 3])
def test_real_rust_spectral_width_matches_python_fallback(monkeypatch, ndim):
    bins = np.array([-2.0, -0.5, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra = spectra_2d if ndim == 2 else np.stack([spectra_2d, spectra_2d + 1.0])
    ref = _fallback_reflectivity(spectra, bins, 0.1, monkeypatch)
    mean_vel = _fallback_mean_velocity(spectra, bins, 0.1, ref, monkeypatch)
    expected = _fallback_spectral_width(
        spectra, bins, 0.1, ref, mean_vel, monkeypatch
    )

    import pyart._rust as rust

    calls = []

    def counted_kernel(spectra_arg, bins_arg, wavelength_arg, ref_arg, mean_vel_arg):
        calls.append(
            (
                spectra_arg.shape,
                bins_arg.shape,
                wavelength_arg,
                ref_arg.shape,
                mean_vel_arg.shape,
            )
        )
        return rust._spectra_spectral_width_dense(
            spectra_arg, bins_arg, wavelength_arg, ref_arg, mean_vel_arg
        )

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: counted_kernel
        if name == "_spectra_spectral_width_dense"
        else None,
    )
    actual = spectra_calculations._get_spectral_width(
        spectra, bins, 0.1, ref, mean_vel
    )

    assert calls == [(spectra.shape, bins.shape, 0.1, ref.shape, mean_vel.shape)]
    _assert_spectra_float_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ndim", [2, 3])
def test_real_rust_skewness_matches_python_fallback(monkeypatch, ndim):
    bins = np.array([-2.0, -0.5, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra = spectra_2d if ndim == 2 else np.stack([spectra_2d, spectra_2d + 1.0])
    ref = _fallback_reflectivity(spectra, bins, 0.1, monkeypatch)
    mean_vel = _fallback_mean_velocity(spectra, bins, 0.1, ref, monkeypatch)
    spec_width = _fallback_spectral_width(
        spectra, bins, 0.1, ref, mean_vel, monkeypatch
    )
    expected = _fallback_skewness(
        spectra, bins, 0.1, ref, mean_vel, spec_width, monkeypatch
    )

    import pyart._rust as rust

    calls = []

    def counted_kernel(
        spectra_arg, bins_arg, wavelength_arg, ref_arg, mean_vel_arg, spec_width_arg
    ):
        calls.append(
            (
                spectra_arg.shape,
                bins_arg.shape,
                wavelength_arg,
                ref_arg.shape,
                mean_vel_arg.shape,
                spec_width_arg.shape,
            )
        )
        return rust._spectra_skewness_dense(
            spectra_arg,
            bins_arg,
            wavelength_arg,
            ref_arg,
            mean_vel_arg,
            spec_width_arg,
        )

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_spectra_skewness_dense" else None,
    )
    actual = spectra_calculations._get_skewness(
        spectra, bins, 0.1, ref, mean_vel, spec_width
    )

    assert calls == [
        (spectra.shape, bins.shape, 0.1, ref.shape, mean_vel.shape, spec_width.shape)
    ]
    _assert_spectra_float_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ndim", [2, 3])
def test_real_rust_kurtosis_matches_python_fallback(monkeypatch, ndim):
    bins = np.array([-2.0, -0.5, 1.0, 4.0], dtype=np.float64)
    spectra_2d = np.array(
        [[1.0, 2.0, np.nan, 4.0], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float64,
    )
    spectra = spectra_2d if ndim == 2 else np.stack([spectra_2d, spectra_2d + 1.0])
    ref = _fallback_reflectivity(spectra, bins, 0.1, monkeypatch)
    mean_vel = _fallback_mean_velocity(spectra, bins, 0.1, ref, monkeypatch)
    spec_width = _fallback_spectral_width(
        spectra, bins, 0.1, ref, mean_vel, monkeypatch
    )
    expected = _fallback_kurtosis(
        spectra, bins, 0.1, ref, mean_vel, spec_width, monkeypatch
    )

    import pyart._rust as rust

    calls = []

    def counted_kernel(
        spectra_arg, bins_arg, wavelength_arg, ref_arg, mean_vel_arg, spec_width_arg
    ):
        calls.append(
            (
                spectra_arg.shape,
                bins_arg.shape,
                wavelength_arg,
                ref_arg.shape,
                mean_vel_arg.shape,
                spec_width_arg.shape,
            )
        )
        return rust._spectra_kurtosis_dense(
            spectra_arg,
            bins_arg,
            wavelength_arg,
            ref_arg,
            mean_vel_arg,
            spec_width_arg,
        )

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: counted_kernel if name == "_spectra_kurtosis_dense" else None,
    )
    actual = spectra_calculations._get_kurtosis(
        spectra, bins, 0.1, ref, mean_vel, spec_width
    )

    assert calls == [
        (spectra.shape, bins.shape, 0.1, ref.shape, mean_vel.shape, spec_width.shape)
    ]
    _assert_spectra_float_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_spectra_reflectivity_1d_stays_on_python_fallback(monkeypatch):
    bins = np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float64)
    spectra = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    expected = _fallback_reflectivity(spectra, bins, 0.1, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = spectra_calculations._get_reflectivity(spectra, bins, 0.1)

    assert isinstance(actual, np.floating)
    assert type(actual) is type(expected)
    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "match"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            "2D or 3D",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            0.1,
            "bins length",
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            "finite or NaN",
        ),
        (
            np.array([[np.nan, np.nan, np.nan]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            "adjacent finite pair",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            "strictly increasing",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.0,
            "wavelength",
        ),
    ],
)
def test_real_rust_spectra_reflectivity_rejects_unsafe_direct_inputs(
    spectra, bins, wavelength, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._spectra_reflectivity_dense(spectra, bins, wavelength)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "match"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            "2D or 3D",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            "bins length",
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            "finite or NaN",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            "strictly increasing",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.0,
            np.array([1.0], dtype=np.float64),
            "wavelength",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            "ref shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([np.nan], dtype=np.float64),
            "ref must be finite",
        ),
    ],
)
def test_real_rust_spectra_mean_velocity_rejects_unsafe_direct_inputs(
    spectra, bins, wavelength, ref, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._spectra_mean_velocity_dense(spectra, bins, wavelength, ref)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "mean_vel", "match"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            np.array(0.5, dtype=np.float64),
            "2D or 3D",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            "bins length",
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            "finite or NaN",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            "strictly increasing",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.0,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            "wavelength",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            "ref shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([[0.5]], dtype=np.float64),
            "mean_velocity shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
            "mean_velocity must be finite",
        ),
    ],
)
def test_real_rust_spectral_width_rejects_unsafe_direct_inputs(
    spectra, bins, wavelength, ref, mean_vel, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._spectra_spectral_width_dense(spectra, bins, wavelength, ref, mean_vel)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "mean_vel", "spec_width", "match"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            np.array(0.5, dtype=np.float64),
            np.array(1.0, dtype=np.float64),
            "2D or 3D",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "bins length",
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "finite or NaN",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "strictly increasing",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.0,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "wavelength",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "ref shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([[0.5]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "mean_velocity shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            "spectral_width shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            "finite and positive",
        ),
    ],
)
def test_real_rust_skewness_rejects_unsafe_direct_inputs(
    spectra, bins, wavelength, ref, mean_vel, spec_width, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._spectra_skewness_dense(
            spectra, bins, wavelength, ref, mean_vel, spec_width
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("spectra", "bins", "wavelength", "ref", "mean_vel", "spec_width", "match"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array(1.0, dtype=np.float64),
            np.array(0.5, dtype=np.float64),
            np.array(1.0, dtype=np.float64),
            "2D or 3D",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "bins length",
        ),
        (
            np.array([[1.0, np.inf, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "finite or NaN",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 2.0, 1.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "strictly increasing",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.0,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "wavelength",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "ref shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([[0.5]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "mean_velocity shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            "spectral_width shape",
        ),
        (
            np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            "finite and positive",
        ),
    ],
)
def test_real_rust_kurtosis_rejects_unsafe_direct_inputs(
    spectra, bins, wavelength, ref, mean_vel, spec_width, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._spectra_kurtosis_dense(
            spectra, bins, wavelength, ref, mean_vel, spec_width
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_noise_floor_limits_preserve_masked_output_contract(monkeypatch):
    spectra = np.array(
        [
            [1.0, 2.0, 8.0, 2.0, 1.0],
            [2.0, 3.0, 9.0, 3.0, 2.0],
        ],
        dtype=np.float64,
    )
    expected_input = spectra.copy()
    actual_input = spectra.copy()

    monkeypatch.setattr(spectra_calculations, "_rust_kernel", lambda _name: None)
    with np.errstate(invalid="ignore"):
        expected = spectra_calculations._get_noise_floor_and_limits(expected_input)

    import pyart._rust as rust

    monkeypatch.setattr(
        spectra_calculations,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    with np.errstate(invalid="ignore"):
        actual = spectra_calculations._get_noise_floor_and_limits(actual_input)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3].data, expected[3].data)
    np.testing.assert_array_equal(actual[3].mask, expected[3].mask)
    assert actual[3].fill_value == expected[3].fill_value
    assert actual[3].dtype == expected[3].dtype
    np.testing.assert_array_equal(actual_input, expected_input)
