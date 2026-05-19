use ndarray::{Array1, ArrayD, IxDyn};
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const SPECTRA_MIN_DB: f64 = -300.0;
const SPECTRA_MAX_DB: f64 = 1000.0;
const SPECTRA_MIN_WAVELENGTH_ABS: f64 = 1.0e-3;
const SPECTRA_MAX_WAVELENGTH_ABS: f64 = 1.0e6;
const SPECTRA_MIN_BIN_DIFF: f64 = 1.0e-6;
const SPECTRA_MAX_BIN_DIFF: f64 = 1.0e9;

#[pyfunction(name = "_spectra_limits_dealiased")]
pub fn spectra_limits_dealiased<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let spectra = spectra.as_array();
    let (nrays, nbins) = spectra.dim();
    let mut left = Array1::<f64>::zeros(nrays);
    let mut right = Array1::<f64>::zeros(nrays);
    let mut new_spec = spectra.to_owned();

    for ray in 0..nrays {
        let Some(peak) = nanargmax_first_row(&spectra, ray, nbins) else {
            left[ray] = f64::NAN;
            right[ray] = f64::NAN;
            continue;
        };

        let mut j = peak;
        while j > 0 && spectra[[ray, j]].is_finite() {
            j -= 1;
        }
        left[ray] = j as f64;

        j = peak;
        while j < nbins - 1 && spectra[[ray, j]].is_finite() {
            j += 1;
        }
        right[ray] = j as f64;

        let left_index = left[ray] as usize;
        let left_stop = if left_index == 0 {
            nbins.saturating_sub(1)
        } else {
            left_index.saturating_sub(1)
        };
        for bin in 0..left_stop {
            new_spec[[ray, bin]] = f64::NAN;
        }

        let right_start = right[ray] as usize + 1;
        let right_stop = nbins.saturating_sub(1);
        for bin in right_start..right_stop {
            new_spec[[ray, bin]] = f64::NAN;
        }
    }

    Ok((
        PyArray1::from_owned_array(py, left),
        PyArray1::from_owned_array(py, right),
        PyArray2::from_owned_array(py, new_spec),
    ))
}

#[pyfunction(name = "_spectra_peak_limits")]
pub fn spectra_peak_limits<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let spectra = spectra.as_array();
    let (nrays, nbins) = spectra.dim();
    let mut left = Array1::<f64>::from_elem(nrays, f64::NAN);
    let mut right = Array1::<f64>::from_elem(nrays, f64::NAN);

    for ray in 0..nrays {
        let Some(peak) = nanargmax_first_row(&spectra, ray, nbins) else {
            continue;
        };

        let mut j = peak;
        while spectra[[ray, j]].is_finite() && j > 0 {
            j -= 1;
        }
        left[ray] = j as f64;

        j = peak;
        while spectra[[ray, j]].is_finite() && j < nbins - 1 {
            j += 1;
        }
        right[ray] = j as f64;
    }

    Ok((
        PyArray1::from_owned_array(py, left),
        PyArray1::from_owned_array(py, right),
    ))
}

#[pyfunction(name = "_spectra_reflectivity_dense")]
pub fn spectra_reflectivity_dense<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArrayDyn<'py, f64>,
    bins: PyReadonlyArray1<'py, f64>,
    wavelength: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let spectra = spectra.as_array();
    let bins = bins.as_array();
    validate_reflectivity_inputs(spectra.clone(), bins, wavelength)?;
    let out = reflectivity_kernel(spectra, bins, wavelength)?;
    Ok(PyArrayDyn::from_owned_array(py, out))
}

#[pyfunction(name = "_spectra_mean_velocity_dense")]
pub fn spectra_mean_velocity_dense<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArrayDyn<'py, f64>,
    bins: PyReadonlyArray1<'py, f64>,
    wavelength: f64,
    ref_db: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let spectra = spectra.as_array();
    let bins = bins.as_array();
    let ref_db = ref_db.as_array();
    validate_reflectivity_inputs(spectra.clone(), bins, wavelength)?;
    validate_mean_velocity_ref_inputs(spectra.clone(), ref_db.clone())?;
    let out = mean_velocity_kernel(spectra, bins, wavelength, ref_db)?;
    Ok(PyArrayDyn::from_owned_array(py, out))
}

#[pyfunction(name = "_spectra_spectral_width_dense")]
pub fn spectra_spectral_width_dense<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArrayDyn<'py, f64>,
    bins: PyReadonlyArray1<'py, f64>,
    wavelength: f64,
    ref_db: PyReadonlyArrayDyn<'py, f64>,
    mean_velocity: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let spectra = spectra.as_array();
    let bins = bins.as_array();
    let ref_db = ref_db.as_array();
    let mean_velocity = mean_velocity.as_array();
    validate_reflectivity_inputs(spectra.clone(), bins, wavelength)?;
    validate_mean_velocity_ref_inputs(spectra.clone(), ref_db.clone())?;
    validate_spectral_width_mean_velocity_inputs(spectra.clone(), mean_velocity.clone())?;
    let out = spectral_width_kernel(spectra, bins, wavelength, ref_db, mean_velocity)?;
    Ok(PyArrayDyn::from_owned_array(py, out))
}

#[pyfunction(name = "_spectra_skewness_dense")]
pub fn spectra_skewness_dense<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArrayDyn<'py, f64>,
    bins: PyReadonlyArray1<'py, f64>,
    wavelength: f64,
    ref_db: PyReadonlyArrayDyn<'py, f64>,
    mean_velocity: PyReadonlyArrayDyn<'py, f64>,
    spectral_width: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let spectra = spectra.as_array();
    let bins = bins.as_array();
    let ref_db = ref_db.as_array();
    let mean_velocity = mean_velocity.as_array();
    let spectral_width = spectral_width.as_array();
    validate_reflectivity_inputs(spectra.clone(), bins, wavelength)?;
    validate_mean_velocity_ref_inputs(spectra.clone(), ref_db.clone())?;
    validate_spectral_width_mean_velocity_inputs(spectra.clone(), mean_velocity.clone())?;
    validate_shape_moment_width_inputs(spectra.clone(), spectral_width.clone())?;
    let out = shape_moment_kernel(
        spectra,
        bins,
        wavelength,
        ref_db,
        mean_velocity,
        spectral_width,
        3,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, out))
}

#[pyfunction(name = "_spectra_kurtosis_dense")]
pub fn spectra_kurtosis_dense<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArrayDyn<'py, f64>,
    bins: PyReadonlyArray1<'py, f64>,
    wavelength: f64,
    ref_db: PyReadonlyArrayDyn<'py, f64>,
    mean_velocity: PyReadonlyArrayDyn<'py, f64>,
    spectral_width: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let spectra = spectra.as_array();
    let bins = bins.as_array();
    let ref_db = ref_db.as_array();
    let mean_velocity = mean_velocity.as_array();
    let spectral_width = spectral_width.as_array();
    validate_reflectivity_inputs(spectra.clone(), bins, wavelength)?;
    validate_mean_velocity_ref_inputs(spectra.clone(), ref_db.clone())?;
    validate_spectral_width_mean_velocity_inputs(spectra.clone(), mean_velocity.clone())?;
    validate_shape_moment_width_inputs(spectra.clone(), spectral_width.clone())?;
    let out = shape_moment_kernel(
        spectra,
        bins,
        wavelength,
        ref_db,
        mean_velocity,
        spectral_width,
        4,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, out))
}

fn validate_mean_velocity_ref_inputs(
    spectra: ndarray::ArrayViewD<'_, f64>,
    ref_db: ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    let expected_shape = &spectra.shape()[..spectra.ndim() - 1];
    if ref_db.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "ref shape must match spectra leading dimensions",
        ));
    }
    if !ref_db.is_standard_layout() {
        return Err(PyValueError::new_err("ref must be C-contiguous"));
    }
    for &value in ref_db.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("ref must be finite"));
        }
        if !(-1000.0..=1000.0).contains(&value) {
            return Err(PyValueError::new_err(
                "ref values are outside the dense mean-velocity kernel range",
            ));
        }
    }
    Ok(())
}

fn validate_spectral_width_mean_velocity_inputs(
    spectra: ndarray::ArrayViewD<'_, f64>,
    mean_velocity: ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    let expected_shape = &spectra.shape()[..spectra.ndim() - 1];
    if mean_velocity.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "mean_velocity shape must match spectra leading dimensions",
        ));
    }
    if !mean_velocity.is_standard_layout() {
        return Err(PyValueError::new_err("mean_velocity must be C-contiguous"));
    }
    for &value in mean_velocity.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("mean_velocity must be finite"));
        }
        if value.abs() > 1.0e9 {
            return Err(PyValueError::new_err(
                "mean_velocity values are outside the dense spectral-width kernel range",
            ));
        }
    }
    Ok(())
}

fn validate_shape_moment_width_inputs(
    spectra: ndarray::ArrayViewD<'_, f64>,
    spectral_width: ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    let expected_shape = &spectra.shape()[..spectra.ndim() - 1];
    if spectral_width.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "spectral_width shape must match spectra leading dimensions",
        ));
    }
    if !spectral_width.is_standard_layout() {
        return Err(PyValueError::new_err("spectral_width must be C-contiguous"));
    }
    for &value in spectral_width.iter() {
        if !value.is_finite() || value <= 0.0 {
            return Err(PyValueError::new_err(
                "spectral_width must be finite and positive",
            ));
        }
        if value.abs() > 1.0e9 {
            return Err(PyValueError::new_err(
                "spectral_width values are outside the dense shape-moment kernel range",
            ));
        }
    }
    Ok(())
}

fn validate_reflectivity_inputs(
    spectra: ndarray::ArrayViewD<'_, f64>,
    bins: ndarray::ArrayView1<'_, f64>,
    wavelength: f64,
) -> PyResult<()> {
    let ndim = spectra.ndim();
    if ndim != 2 && ndim != 3 {
        return Err(PyValueError::new_err("spectra must be 2D or 3D"));
    }
    if !spectra.is_standard_layout() || !bins.is_standard_layout() {
        return Err(PyValueError::new_err(
            "spectra and bins must be C-contiguous",
        ));
    }
    let nbins = spectra.shape()[ndim - 1];
    if bins.len() != nbins {
        return Err(PyValueError::new_err(
            "bins length must match spectra last dimension",
        ));
    }
    if nbins < 2 {
        return Err(PyValueError::new_err("at least two bins are required"));
    }
    if !wavelength.is_finite()
        || wavelength.abs() < SPECTRA_MIN_WAVELENGTH_ABS
        || wavelength.abs() > SPECTRA_MAX_WAVELENGTH_ABS
    {
        return Err(PyValueError::new_err(
            "wavelength must be finite and within the dense kernel range",
        ));
    }
    for idx in 0..bins.len() {
        let value = bins[idx];
        if !value.is_finite() {
            return Err(PyValueError::new_err("bins must be finite"));
        }
        if idx > 0 {
            let diff = value - bins[idx - 1];
            if !(SPECTRA_MIN_BIN_DIFF..=SPECTRA_MAX_BIN_DIFF).contains(&diff) {
                return Err(PyValueError::new_err(
                    "bins must be strictly increasing with finite dense-kernel spacing",
                ));
            }
        }
    }

    let mut finite_pair_by_row = vec![false; spectra.len() / nbins];
    let spectra_slice = spectra
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("spectra must be contiguous"))?;
    for (idx, &value) in spectra_slice.iter().enumerate() {
        if value.is_infinite() {
            return Err(PyValueError::new_err("spectra must be finite or NaN"));
        }
        if value.is_finite() && !(SPECTRA_MIN_DB..=SPECTRA_MAX_DB).contains(&value) {
            return Err(PyValueError::new_err(
                "spectra values are outside the dense kernel range",
            ));
        }
        if idx % nbins > 0 {
            let prev = spectra_slice[idx - 1];
            if prev.is_finite() && value.is_finite() {
                finite_pair_by_row[idx / nbins] = true;
            }
        }
    }
    if finite_pair_by_row.iter().any(|&has_pair| !has_pair) {
        return Err(PyValueError::new_err(
            "each spectra row must include at least one adjacent finite pair",
        ));
    }

    Ok(())
}

fn mean_velocity_kernel(
    spectra: ndarray::ArrayViewD<'_, f64>,
    bins: ndarray::ArrayView1<'_, f64>,
    wavelength: f64,
    ref_db: ndarray::ArrayViewD<'_, f64>,
) -> PyResult<ArrayD<f64>> {
    let ndim = spectra.ndim();
    let nbins = bins.len();
    let leading_shape = &spectra.shape()[..ndim - 1];
    let leading_len = leading_shape.iter().product::<usize>();
    let mut out = ArrayD::<f64>::zeros(IxDyn(leading_shape));
    let spectra_slice = spectra
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("spectra must be contiguous"))?;
    let ref_slice = ref_db
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ref must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("output allocation must be contiguous"))?;
    let radar_constant = 1.0e18 * wavelength.powi(4) / (0.93 * std::f64::consts::PI.powi(5));

    for row in 0..leading_len {
        let offset = row * nbins;
        let mut total = 0.0;
        for bin in 0..nbins - 1 {
            let left = spectra_slice[offset + bin];
            let right = spectra_slice[offset + bin + 1];
            let spec_med =
                radar_constant * (10.0_f64.powf(left / 10.0) + 10.0_f64.powf(right / 10.0)) / 2.0;
            let bins_med = (bins[bin] + bins[bin + 1]) / 2.0;
            let term = spec_med * bins_med * (bins[bin + 1] - bins[bin]);
            if !term.is_nan() {
                total += term;
            }
        }
        out_slice[row] = total / 10.0_f64.powf(ref_slice[row] / 10.0);
    }

    Ok(out)
}

fn reflectivity_kernel(
    spectra: ndarray::ArrayViewD<'_, f64>,
    bins: ndarray::ArrayView1<'_, f64>,
    wavelength: f64,
) -> PyResult<ArrayD<f64>> {
    let ndim = spectra.ndim();
    let nbins = bins.len();
    let leading_shape = &spectra.shape()[..ndim - 1];
    let leading_len = leading_shape.iter().product::<usize>();
    let mut out = ArrayD::<f64>::zeros(IxDyn(leading_shape));
    let spectra_slice = spectra
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("spectra must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("output allocation must be contiguous"))?;
    let radar_constant = 1.0e18 * wavelength.powi(4) / (0.93 * std::f64::consts::PI.powi(5));

    for row in 0..leading_len {
        let offset = row * nbins;
        let mut total = 0.0;
        for bin in 0..nbins - 1 {
            let left = spectra_slice[offset + bin];
            let right = spectra_slice[offset + bin + 1];
            let spec_med =
                radar_constant * (10.0_f64.powf(left / 10.0) + 10.0_f64.powf(right / 10.0)) / 2.0;
            let term = spec_med * (bins[bin + 1] - bins[bin]);
            if !term.is_nan() {
                total += term;
            }
        }
        if total == 0.0 {
            out_slice[row] = f64::NEG_INFINITY;
            continue;
        }
        if !total.is_finite() || total < 0.0 {
            return Err(PyValueError::new_err(
                "reflectivity sum must be non-negative and finite",
            ));
        }
        out_slice[row] = 10.0 * total.log10();
    }

    Ok(out)
}

fn spectral_width_kernel(
    spectra: ndarray::ArrayViewD<'_, f64>,
    bins: ndarray::ArrayView1<'_, f64>,
    wavelength: f64,
    ref_db: ndarray::ArrayViewD<'_, f64>,
    mean_velocity: ndarray::ArrayViewD<'_, f64>,
) -> PyResult<ArrayD<f64>> {
    let ndim = spectra.ndim();
    let nbins = bins.len();
    let leading_shape = &spectra.shape()[..ndim - 1];
    let leading_len = leading_shape.iter().product::<usize>();
    let mut out = ArrayD::<f64>::zeros(IxDyn(leading_shape));
    let spectra_slice = spectra
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("spectra must be contiguous"))?;
    let ref_slice = ref_db
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ref must be contiguous"))?;
    let mean_velocity_slice = mean_velocity
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mean_velocity must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("output allocation must be contiguous"))?;
    let radar_constant = 1.0e18 * wavelength.powi(4) / (0.93 * std::f64::consts::PI.powi(5));

    for row in 0..leading_len {
        let offset = row * nbins;
        let mut total = 0.0;
        for bin in 0..nbins - 1 {
            let left = spectra_slice[offset + bin];
            let right = spectra_slice[offset + bin + 1];
            let spec_med =
                radar_constant * (10.0_f64.powf(left / 10.0) + 10.0_f64.powf(right / 10.0)) / 2.0;
            let bins_med = (bins[bin] + bins[bin + 1]) / 2.0;
            let velocity_delta = bins_med - mean_velocity_slice[row];
            let term = spec_med * velocity_delta.powi(2) * (bins[bin + 1] - bins[bin]);
            if !term.is_nan() {
                total += term;
            }
        }
        let ref_linear = 10.0_f64.powf(ref_slice[row] / 10.0);
        out_slice[row] = (total / ref_linear).sqrt();
    }

    Ok(out)
}

fn shape_moment_kernel(
    spectra: ndarray::ArrayViewD<'_, f64>,
    bins: ndarray::ArrayView1<'_, f64>,
    wavelength: f64,
    ref_db: ndarray::ArrayViewD<'_, f64>,
    mean_velocity: ndarray::ArrayViewD<'_, f64>,
    spectral_width: ndarray::ArrayViewD<'_, f64>,
    order: i32,
) -> PyResult<ArrayD<f64>> {
    let ndim = spectra.ndim();
    let nbins = bins.len();
    let leading_shape = &spectra.shape()[..ndim - 1];
    let leading_len = leading_shape.iter().product::<usize>();
    let mut out = ArrayD::<f64>::zeros(IxDyn(leading_shape));
    let spectra_slice = spectra
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("spectra must be contiguous"))?;
    let ref_slice = ref_db
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ref must be contiguous"))?;
    let mean_velocity_slice = mean_velocity
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mean_velocity must be contiguous"))?;
    let spectral_width_slice = spectral_width
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("spectral_width must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("output allocation must be contiguous"))?;
    let radar_constant = 1.0e18 * wavelength.powi(4) / (0.93 * std::f64::consts::PI.powi(5));

    for row in 0..leading_len {
        let offset = row * nbins;
        let mut total = 0.0;
        for bin in 0..nbins - 1 {
            let left = spectra_slice[offset + bin];
            let right = spectra_slice[offset + bin + 1];
            let spec_med =
                radar_constant * (10.0_f64.powf(left / 10.0) + 10.0_f64.powf(right / 10.0)) / 2.0;
            let bins_med = (bins[bin] + bins[bin + 1]) / 2.0;
            let velocity_delta = bins_med - mean_velocity_slice[row];
            let term = spec_med * velocity_delta.powi(order) * (bins[bin + 1] - bins[bin]);
            if !term.is_nan() {
                total += term;
            }
        }
        let ref_linear = 10.0_f64.powf(ref_slice[row] / 10.0);
        let central_moment = total / ref_linear;
        out_slice[row] = central_moment / spectral_width_slice[row].powi(order);
    }

    Ok(out)
}

fn nanargmax_first_row(
    spectra: &ndarray::ArrayView2<'_, f64>,
    ray: usize,
    nbins: usize,
) -> Option<usize> {
    let mut best_index = None;
    let mut best_value = f64::NEG_INFINITY;

    for bin in 0..nbins {
        let value = spectra[[ray, bin]];
        if value.is_nan() {
            continue;
        }
        if best_index.is_none() || value > best_value {
            best_index = Some(bin);
            best_value = value;
        }
    }

    best_index
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(spectra_limits_dealiased, module)?)?;
    module.add_function(wrap_pyfunction!(spectra_peak_limits, module)?)?;
    module.add_function(wrap_pyfunction!(spectra_reflectivity_dense, module)?)?;
    module.add_function(wrap_pyfunction!(spectra_mean_velocity_dense, module)?)?;
    module.add_function(wrap_pyfunction!(spectra_spectral_width_dense, module)?)?;
    module.add_function(wrap_pyfunction!(spectra_skewness_dense, module)?)?;
    module.add_function(wrap_pyfunction!(spectra_kurtosis_dense, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nanargmax_ignores_nan_and_keeps_first_maximum() {
        let spectra = array![[f64::NAN, 2.0, 2.0, 1.0]];
        assert_eq!(nanargmax_first_row(&spectra.view(), 0, 4), Some(1));
    }

    #[test]
    fn nanargmax_returns_none_for_all_nan_rows() {
        let spectra = array![[f64::NAN, f64::NAN]];
        assert_eq!(nanargmax_first_row(&spectra.view(), 0, 2), None);
    }
}
