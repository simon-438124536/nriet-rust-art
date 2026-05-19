use ndarray::{ArrayD, IxDyn, Zip};
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool};

const QPE_ZPOLY_MAX_ABS_REFL: f64 = 1000.0;
const QPE_RAIN_RATE_Z_MAX_ABS_REFL: f64 = 1000.0;
const QPE_RAIN_RATE_Z_MAX_ABS_ALPHA: f64 = 1.0e50;
const QPE_RAIN_RATE_Z_MAX_ABS_EXPONENT: f64 = 250.0;
const QPE_RAIN_RATE_KDP_MAX_VALUE: f64 = 1.0e12;

#[pyfunction(name = "_qpe_coeff_rkdp")]
fn qpe_coeff_rkdp(freq_band: &str) -> PyResult<(f64, f64)> {
    match freq_band {
        "S" => Ok((50.70, 0.8500)),
        "C" => Ok((29.70, 0.8500)),
        "X" => Ok((15.81, 0.7992)),
        _ => Err(PyValueError::new_err("freq_band must be one of S, C, or X")),
    }
}

#[pyfunction(name = "_qpe_coeff_ra")]
fn qpe_coeff_ra(freq_band: &str) -> PyResult<(f64, f64)> {
    match freq_band {
        "S" => Ok((3100.0, 1.03)),
        "C" => Ok((250.0, 0.91)),
        "X" => Ok((45.5, 0.83)),
        _ => Err(PyValueError::new_err("freq_band must be one of S, C, or X")),
    }
}

#[pyfunction(name = "_qpe_zpoly_dense_f64")]
fn qpe_zpoly_dense_f64<'py>(
    py: Python<'py>,
    refl: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let refl = refl.as_array();
    validate_zpoly_inputs(&refl)?;

    let mut output = ArrayD::<f64>::zeros(IxDyn(refl.shape()));
    Zip::from(&mut output).and(refl).for_each(|slot, &value| {
        let value2 = value * value;
        let value3 = value * value2;
        let value4 = value * value3;
        let exponent = -2.3 + 0.17 * value - 5.1e-3 * value2 + 9.8e-5 * value3 - 6e-7 * value4;
        *slot = 10.0_f64.powf(exponent);
    });

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_qpe_rain_rate_z_dense_f64")]
fn qpe_rain_rate_z_dense_f64<'py>(
    py: Python<'py>,
    refl: PyReadonlyArrayDyn<'py, f64>,
    alpha: &Bound<'_, PyAny>,
    beta: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let alpha = extract_qpe_float_scalar(alpha)?;
    let beta = extract_qpe_float_scalar(beta)?;
    let refl = refl.as_array();
    validate_rain_rate_z_inputs(&refl, alpha, beta)?;

    let mut output = ArrayD::<f64>::zeros(IxDyn(refl.shape()));
    Zip::from(&mut output).and(refl).for_each(|slot, &value| {
        let linear_reflectivity = 10.0_f64.powf(0.1 * value);
        *slot = alpha * linear_reflectivity.powf(beta);
    });

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_qpe_rain_rate_kdp_dense_f64")]
fn qpe_rain_rate_kdp_dense_f64<'py>(
    py: Python<'py>,
    kdp: PyReadonlyArrayDyn<'py, f64>,
    alpha: &Bound<'_, PyAny>,
    beta: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let alpha = extract_qpe_float_scalar(alpha)?;
    let beta = extract_qpe_float_scalar(beta)?;
    let kdp = kdp.as_array();
    validate_rain_rate_kdp_inputs(&kdp, alpha, beta)?;

    let mut output = ArrayD::<f64>::zeros(IxDyn(kdp.shape()));
    Zip::from(&mut output).and(kdp).for_each(|slot, &value| {
        *slot = alpha * value.powf(beta);
    });

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_qpe_rain_rate_a_dense_f64")]
fn qpe_rain_rate_a_dense_f64<'py>(
    py: Python<'py>,
    att: PyReadonlyArrayDyn<'py, f64>,
    alpha: &Bound<'_, PyAny>,
    beta: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let alpha = extract_qpe_float_scalar(alpha)?;
    let beta = extract_qpe_float_scalar(beta)?;
    let att = att.as_array();
    validate_rain_rate_a_inputs(&att, alpha, beta)?;

    let mut output = ArrayD::<f64>::zeros(IxDyn(att.shape()));
    Zip::from(&mut output).and(att).for_each(|slot, &value| {
        *slot = alpha * value.powf(beta);
    });

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_qpe_threshold_blend_dense_f64")]
fn qpe_threshold_blend_dense_f64(
    rain_main: Bound<'_, PyArrayDyn<f64>>,
    rain_secondary: PyReadonlyArrayDyn<'_, f64>,
    thresh: &Bound<'_, PyAny>,
    thresh_max: bool,
) -> PyResult<()> {
    let thresh = extract_qpe_threshold_scalar(thresh)?;
    let mut rain_main = rain_main
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("rain_main must be writable and unborrowed"))?;
    let rain_main_view = rain_main.as_array();
    let rain_secondary = rain_secondary.as_array();
    validate_threshold_blend_inputs(&rain_main_view, &rain_secondary, thresh)?;
    drop(rain_main_view);

    let mut rain_main = rain_main.as_array_mut();
    Zip::from(&mut rain_main)
        .and(rain_secondary)
        .for_each(|main_value, &secondary_value| {
            let use_secondary = if thresh_max {
                *main_value > thresh
            } else {
                *main_value < thresh
            };
            if use_secondary {
                *main_value = secondary_value;
            }
        });
    Ok(())
}

fn extract_qpe_float_scalar(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    let type_name = value.get_type().name()?.to_str()?.to_owned();
    if value.is_instance_of::<PyBool>() || type_name == "bool" || type_name == "bool_" {
        return Err(PyValueError::new_err(
            "alpha and beta must be non-boolean scalars",
        ));
    }
    value
        .extract::<f64>()
        .map_err(|_| PyValueError::new_err("alpha and beta must be numeric scalars"))
}

fn extract_qpe_threshold_scalar(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    let type_name = value.get_type().name()?.to_str()?.to_owned();
    if value.is_instance_of::<PyBool>() || type_name == "bool" || type_name == "bool_" {
        return Err(PyValueError::new_err(
            "threshold must be a non-boolean scalar",
        ));
    }
    value
        .extract::<f64>()
        .map_err(|_| PyValueError::new_err("threshold must be a numeric scalar"))
}

fn validate_threshold_blend_inputs(
    rain_main: &ndarray::ArrayViewD<'_, f64>,
    rain_secondary: &ndarray::ArrayViewD<'_, f64>,
    thresh: f64,
) -> PyResult<()> {
    if rain_main.shape() != rain_secondary.shape() {
        return Err(PyValueError::new_err(
            "rain_main and rain_secondary must have the same shape",
        ));
    }
    if !rain_main.is_standard_layout() || !rain_secondary.is_standard_layout() {
        return Err(PyValueError::new_err(
            "rain_main and rain_secondary must be C-contiguous",
        ));
    }
    if !thresh.is_finite() {
        return Err(PyValueError::new_err("threshold must be finite"));
    }
    for (&main_value, &secondary_value) in rain_main.iter().zip(rain_secondary.iter()) {
        if !main_value.is_finite() || !secondary_value.is_finite() {
            return Err(PyValueError::new_err(
                "rain_main and rain_secondary must be finite",
            ));
        }
    }
    Ok(())
}

fn validate_rain_rate_kdp_inputs(
    kdp: &ndarray::ArrayViewD<'_, f64>,
    alpha: f64,
    beta: f64,
) -> PyResult<()> {
    if !kdp.is_standard_layout() {
        return Err(PyValueError::new_err("kdp must be C-contiguous"));
    }
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(PyValueError::new_err("alpha and beta must be finite"));
    }
    if alpha.abs() > QPE_RAIN_RATE_Z_MAX_ABS_ALPHA {
        return Err(PyValueError::new_err(
            "alpha is outside the dense KDP rain-rate kernel range",
        ));
    }
    let mut has_positive = false;
    for &value in kdp.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("kdp must be finite"));
        }
        if !(0.0..=QPE_RAIN_RATE_KDP_MAX_VALUE).contains(&value) {
            return Err(PyValueError::new_err(
                "kdp values are outside the dense KDP rain-rate kernel range",
            ));
        }
        if value > 0.0 {
            has_positive = true;
            let exponent = value.ln() * beta;
            if !exponent.is_finite() || exponent.abs() > QPE_RAIN_RATE_Z_MAX_ABS_EXPONENT {
                return Err(PyValueError::new_err(
                    "kdp and beta produce values outside the dense KDP rain-rate kernel range",
                ));
            }
        }
    }
    if !has_positive && beta < 0.0 {
        return Err(PyValueError::new_err(
            "kdp and beta produce values outside the dense KDP rain-rate kernel range",
        ));
    }
    Ok(())
}

fn validate_rain_rate_a_inputs(
    att: &ndarray::ArrayViewD<'_, f64>,
    alpha: f64,
    beta: f64,
) -> PyResult<()> {
    if !att.is_standard_layout() {
        return Err(PyValueError::new_err("att must be C-contiguous"));
    }
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(PyValueError::new_err("alpha and beta must be finite"));
    }
    if alpha.abs() > QPE_RAIN_RATE_Z_MAX_ABS_ALPHA {
        return Err(PyValueError::new_err(
            "alpha is outside the dense attenuation rain-rate kernel range",
        ));
    }
    let mut has_positive = false;
    for &value in att.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("att must be finite"));
        }
        if !(0.0..=QPE_RAIN_RATE_KDP_MAX_VALUE).contains(&value) {
            return Err(PyValueError::new_err(
                "att values are outside the dense attenuation rain-rate kernel range",
            ));
        }
        if value > 0.0 {
            has_positive = true;
            let exponent = value.ln() * beta;
            if !exponent.is_finite() || exponent.abs() > QPE_RAIN_RATE_Z_MAX_ABS_EXPONENT {
                return Err(PyValueError::new_err(
                    "att and beta produce values outside the dense attenuation rain-rate kernel range",
                ));
            }
        }
    }
    if !has_positive && beta < 0.0 {
        return Err(PyValueError::new_err(
            "att and beta produce values outside the dense attenuation rain-rate kernel range",
        ));
    }
    Ok(())
}

fn validate_rain_rate_z_inputs(
    refl: &ndarray::ArrayViewD<'_, f64>,
    alpha: f64,
    beta: f64,
) -> PyResult<()> {
    if !refl.is_standard_layout() {
        return Err(PyValueError::new_err("refl must be C-contiguous"));
    }
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(PyValueError::new_err("alpha and beta must be finite"));
    }
    if alpha.abs() > QPE_RAIN_RATE_Z_MAX_ABS_ALPHA {
        return Err(PyValueError::new_err(
            "alpha is outside the dense Z rain-rate kernel range",
        ));
    }
    for &value in refl.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("refl must be finite"));
        }
        if value.abs() > QPE_RAIN_RATE_Z_MAX_ABS_REFL {
            return Err(PyValueError::new_err(
                "refl values are outside the dense Z rain-rate kernel range",
            ));
        }
        let exponent = 0.1 * value * beta;
        if !exponent.is_finite() || exponent.abs() > QPE_RAIN_RATE_Z_MAX_ABS_EXPONENT {
            return Err(PyValueError::new_err(
                "refl and beta produce values outside the dense Z rain-rate kernel range",
            ));
        }
    }
    Ok(())
}

fn validate_zpoly_inputs(refl: &ndarray::ArrayViewD<'_, f64>) -> PyResult<()> {
    if !refl.is_standard_layout() {
        return Err(PyValueError::new_err("refl must be C-contiguous"));
    }
    for &value in refl.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("refl must be finite"));
        }
        if value.abs() > QPE_ZPOLY_MAX_ABS_REFL {
            return Err(PyValueError::new_err(
                "refl values are outside the dense Z-poly kernel range",
            ));
        }
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(qpe_coeff_rkdp, module)?)?;
    module.add_function(wrap_pyfunction!(qpe_coeff_ra, module)?)?;
    module.add_function(wrap_pyfunction!(qpe_zpoly_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(qpe_rain_rate_z_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(qpe_rain_rate_kdp_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(qpe_rain_rate_a_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(qpe_threshold_blend_dense_f64, module)?)?;
    Ok(())
}
