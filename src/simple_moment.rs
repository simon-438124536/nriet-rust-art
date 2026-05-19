use ndarray::{Array2, ArrayD, IxDyn, Zip};
use numpy::{PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

const CDR_MAX_ABS_ZDR_DB: f64 = 3000.0;
const L_MIN_RHOHV: f64 = -1.0e300;

#[pyfunction(name = "_simple_moment_snr_dense")]
fn simple_moment_snr_dense<'py>(
    py: Python<'py>,
    refl: PyReadonlyArrayDyn<'py, f64>,
    noised_bz: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let refl = refl.as_array();
    let noised_bz = noised_bz.as_array();

    if refl.shape() != noised_bz.shape() {
        return Err(PyValueError::new_err(
            "refl and noisedBZ must have the same shape",
        ));
    }
    if !refl.is_standard_layout() || !noised_bz.is_standard_layout() {
        return Err(PyValueError::new_err(
            "refl and noisedBZ must be C-contiguous",
        ));
    }

    let mut output = ArrayD::<f64>::zeros(IxDyn(refl.shape()));
    Zip::from(&mut output)
        .and(refl)
        .and(noised_bz)
        .for_each(|slot, &refl_value, &noise_value| {
            *slot = refl_value - noise_value;
        });

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_simple_moment_tile_rows_f64")]
fn simple_moment_tile_rows_f64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    rows: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = values.as_array();
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }

    let cols = values.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("tile output is too large"))?;
    let mut output = Vec::with_capacity(total);
    for _ in 0..rows {
        output.extend(values.iter().copied());
    }

    let output = Array2::from_shape_vec((rows, cols), output)
        .map_err(|_| PyValueError::new_err("tile output shape is invalid"))?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction(name = "_simple_moment_tile_rows_masked_f64")]
fn simple_moment_tile_rows_masked_f64<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    rows: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    reject_simple_moment_masked_array(py, "values", values)?;
    reject_simple_moment_masked_array(py, "mask", mask)?;
    let values = values
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("values must be a 1D float64 array"))?;
    let mask = mask
        .extract::<PyReadonlyArray1<'py, bool>>()
        .map_err(|_| PyValueError::new_err("mask must be a 1D bool array"))?;
    let values = values.as_array();
    let mask = mask.as_array();
    if values.len() != mask.len() {
        return Err(PyValueError::new_err(
            "values and mask must have the same length",
        ));
    }
    if !values.is_standard_layout() || !mask.is_standard_layout() {
        return Err(PyValueError::new_err(
            "values and mask must be C-contiguous",
        ));
    }
    let cols = values.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("tile output is too large"))?;
    let mut output_data = Vec::with_capacity(total);
    let mut output_mask = Vec::with_capacity(total);
    for _ in 0..rows {
        output_data.extend(values.iter().copied());
        output_mask.extend(mask.iter().copied());
    }

    let output_data = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|_| PyValueError::new_err("tile data output shape is invalid"))?;
    let output_mask = Array2::from_shape_vec((rows, cols), output_mask)
        .map_err(|_| PyValueError::new_err("tile mask output shape is invalid"))?;
    Ok((
        PyArray2::from_owned_array(py, output_data),
        PyArray2::from_owned_array(py, output_mask),
    ))
}

#[pyfunction(name = "_simple_moment_cdr_dense_f64")]
fn simple_moment_cdr_dense_f64<'py>(
    py: Python<'py>,
    rhohv: PyReadonlyArrayDyn<'py, f64>,
    zdr_db: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<bool>>)> {
    let rhohv = rhohv.as_array();
    let zdr_db = zdr_db.as_array();
    validate_cdr_inputs(rhohv.clone(), zdr_db.clone())?;

    let mut output = ArrayD::<f64>::zeros(IxDyn(rhohv.shape()));
    let mut mask = ArrayD::<bool>::from_elem(IxDyn(rhohv.shape()), false);
    Zip::from(&mut output)
        .and(&mut mask)
        .and(rhohv)
        .and(zdr_db)
        .for_each(|slot, mask_slot, &rhohv_value, &zdr_db_value| {
            let zdr = 10.0_f64.powf(0.1 * zdr_db_value);
            let inv_zdr = 1.0 / zdr;
            let sqrt_inv_zdr = inv_zdr.sqrt();
            let numerator = 1.0 + inv_zdr - 2.0 * rhohv_value * sqrt_inv_zdr;
            let denominator = 1.0 + inv_zdr + 2.0 * rhohv_value * sqrt_inv_zdr;
            let ratio = numerator / denominator;

            if denominator == 0.0 || !ratio.is_finite() || ratio <= 0.0 {
                *slot = 10.0;
                *mask_slot = true;
            } else {
                let cdr = 10.0 * ratio.log10();
                if cdr.is_finite() {
                    *slot = cdr;
                } else {
                    *slot = 10.0;
                    *mask_slot = true;
                }
            }
        });

    Ok((
        PyArrayDyn::from_owned_array(py, output),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_simple_moment_l_dense_f64")]
fn simple_moment_l_dense_f64<'py>(
    py: Python<'py>,
    mut rhohv: PyReadwriteArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<bool>>)> {
    {
        let rhohv_view = rhohv.as_array();
        validate_l_inputs(rhohv_view.clone())?;
    }

    let mut rhohv_mut = rhohv.as_array_mut();
    let mut output = ArrayD::<f64>::zeros(IxDyn(rhohv_mut.shape()));
    let mask = ArrayD::<bool>::from_elem(IxDyn(rhohv_mut.shape()), false);
    Zip::from(&mut output)
        .and(&mut rhohv_mut)
        .for_each(|slot, rhohv_value| {
            if *rhohv_value >= 1.0 {
                *rhohv_value = 0.9999;
            }
            *slot = -(1.0 - *rhohv_value).log10();
        });

    Ok((
        PyArrayDyn::from_owned_array(py, output),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

fn validate_l_inputs(rhohv: ndarray::ArrayViewD<'_, f64>) -> PyResult<()> {
    if !rhohv.is_standard_layout() {
        return Err(PyValueError::new_err("rhohv must be C-contiguous"));
    }
    for &rhohv_value in rhohv.iter() {
        if !rhohv_value.is_finite() {
            return Err(PyValueError::new_err("rhohv must be finite"));
        }
        if rhohv_value < L_MIN_RHOHV {
            return Err(PyValueError::new_err(
                "rhohv values are outside the dense L kernel range",
            ));
        }
    }
    Ok(())
}

fn validate_cdr_inputs(
    rhohv: ndarray::ArrayViewD<'_, f64>,
    zdr_db: ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    if rhohv.shape() != zdr_db.shape() {
        return Err(PyValueError::new_err(
            "rhohv and zdrdB must have the same shape",
        ));
    }
    if !rhohv.is_standard_layout() || !zdr_db.is_standard_layout() {
        return Err(PyValueError::new_err(
            "rhohv and zdrdB must be C-contiguous",
        ));
    }

    for (&rhohv_value, &zdr_db_value) in rhohv.iter().zip(zdr_db.iter()) {
        if !rhohv_value.is_finite() || !zdr_db_value.is_finite() {
            return Err(PyValueError::new_err("rhohv and zdrdB must be finite"));
        }
        if zdr_db_value.abs() > CDR_MAX_ABS_ZDR_DB {
            return Err(PyValueError::new_err(
                "zdrdB values are outside the dense CDR kernel range",
            ));
        }
    }

    Ok(())
}

#[pyfunction(name = "_simple_moment_clamp_ge_f64")]
fn simple_moment_clamp_ge_f64(
    mut values: PyReadwriteArrayDyn<'_, f64>,
    threshold: f64,
    replacement: f64,
) -> PyResult<()> {
    if !threshold.is_finite() || !replacement.is_finite() {
        return Err(PyValueError::new_err(
            "threshold and replacement must be finite",
        ));
    }
    let mut values = values.as_array_mut();
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }

    for value in values.iter_mut() {
        if *value >= threshold {
            *value = replacement;
        }
    }
    Ok(())
}

fn reject_simple_moment_masked_array(
    py: Python<'_>,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let numpy_ma = py.import("numpy.ma")?;
    if numpy_ma
        .getattr("isMaskedArray")?
        .call1((value,))?
        .extract::<bool>()?
    {
        return Err(PyValueError::new_err(format!("{name} must be mask-free")));
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(simple_moment_snr_dense, module)?)?;
    module.add_function(wrap_pyfunction!(simple_moment_tile_rows_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        simple_moment_tile_rows_masked_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(simple_moment_cdr_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(simple_moment_l_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(simple_moment_clamp_ge_f64, module)?)?;
    Ok(())
}
