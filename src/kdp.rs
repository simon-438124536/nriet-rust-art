use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const INVALID_FINITE_ORDER: &str = "Invalid finite_order";

#[pyfunction]
pub fn lowpass_maesaka_term(
    k: PyReadonlyArray2<'_, f64>,
    dr: f64,
    finite_order: &str,
    mut d2kdr2: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    if finite_order != "low" {
        return Err(PyValueError::new_err(INVALID_FINITE_ORDER));
    }

    let k_view = k.as_array();
    let (nr, ng) = k_view.dim();
    if d2kdr2.as_array().dim() != (nr, ng) {
        return Err(PyValueError::new_err(
            "output array shape must match input array shape",
        ));
    }
    validate_term_shape(nr, ng)?;

    let dr2 = dr.powi(2);
    let mut out = d2kdr2.as_array_mut();
    for r in 0..nr {
        for g in 0..ng {
            out[[r, g]] = if g > 0 && g < ng - 1 {
                (k_view[[r, g + 1]] - 2.0 * k_view[[r, g]] + k_view[[r, g - 1]]) / dr2
            } else if g == 0 {
                (k_view[[r, g]] - 2.0 * k_view[[r, g + 1]] + k_view[[r, g + 2]]) / dr2
            } else {
                (k_view[[r, g]] - 2.0 * k_view[[r, g - 1]] + k_view[[r, g - 2]]) / dr2
            };
        }
    }

    Ok(())
}

#[pyfunction]
pub fn lowpass_maesaka_jac(
    d2kdr2: PyReadonlyArray2<'_, f64>,
    dr: f64,
    clpf: f64,
    finite_order: &str,
    mut djlpfdk: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    if finite_order != "low" {
        return Err(PyValueError::new_err(INVALID_FINITE_ORDER));
    }

    let d2_view = d2kdr2.as_array();
    let (nr, ng) = d2_view.dim();
    if djlpfdk.as_array().dim() != (nr, ng) {
        return Err(PyValueError::new_err(
            "output array shape must match input array shape",
        ));
    }
    validate_jac_shape(nr, ng)?;

    let scale = clpf / dr.powi(2);
    let mut out = djlpfdk.as_array_mut();
    for r in 0..nr {
        for g in 0..ng {
            out[[r, g]] = if g > 2 && g < ng - 3 {
                scale * (d2_view[[r, g - 1]] - 2.0 * d2_view[[r, g]] + d2_view[[r, g + 1]])
            } else if g == 2 {
                scale
                    * (d2_view[[r, g - 2]] + d2_view[[r, g - 1]] - 2.0 * d2_view[[r, g]]
                        + d2_view[[r, g + 1]])
            } else if g == 1 {
                scale * (d2_view[[r, g + 1]] - 2.0 * d2_view[[r, g]] - 2.0 * d2_view[[r, g - 1]])
            } else if g == 0 {
                scale * (d2_view[[r, g]] + d2_view[[r, g + 1]])
            } else if g == ng - 3 {
                scale
                    * (d2_view[[r, g + 2]] + d2_view[[r, g + 1]] - 2.0 * d2_view[[r, g]]
                        + d2_view[[r, g - 1]])
            } else if g == ng - 2 {
                scale * (d2_view[[r, g - 1]] - 2.0 * d2_view[[r, g]] - 2.0 * d2_view[[r, g + 1]])
            } else {
                scale * (d2_view[[r, g]] + d2_view[[r, g - 1]])
            };
        }
    }

    Ok(())
}

#[pyfunction]
pub fn forward_reverse_phidp<'py>(
    py: Python<'py>,
    k: PyReadonlyArray2<'py, f64>,
    phi_near: PyReadonlyArray1<'py, f64>,
    phi_far: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let k_view = k.as_array();
    let phi_near_view = phi_near.as_array();
    let phi_far_view = phi_far.as_array();
    let (nr, ng) = k_view.dim();
    validate_forward_reverse_inputs(&k_view, &phi_near_view, &phi_far_view)?;

    let mut phidp_f = Array2::<f64>::zeros((nr, ng));
    let mut phidp_r = Array2::<f64>::zeros((nr, ng));

    for r in 0..nr {
        if ng == 0 {
            continue;
        }

        let near = phi_near_view[r];
        phidp_f[[r, 0]] = 0.0 + near;
        let mut forward_sum = 0.0;
        for g in 1..ng {
            let prev = k_view[[r, g - 1]];
            forward_sum += prev * prev;
            phidp_f[[r, g]] = forward_sum + near;
        }

        let far = phi_far_view[r];
        phidp_r[[r, ng - 1]] = far - 0.0;
        let mut reverse_sum = 0.0;
        for g in (1..ng).rev() {
            let value = k_view[[r, g]];
            reverse_sum += value * value;
            phidp_r[[r, g - 1]] = far - reverse_sum;
        }
    }

    Ok((
        PyArray2::from_owned_array(py, phidp_f),
        PyArray2::from_owned_array(py, phidp_r),
    ))
}

#[pyfunction(name = "_kdp_range_resolution_uniform")]
pub fn kdp_range_resolution_uniform(
    ranges: PyReadonlyArray1<'_, f64>,
    atol: f64,
) -> PyResult<Option<f64>> {
    let ranges = ranges.as_array();
    validate_range_resolution_inputs(&ranges, atol)?;

    let dr0 = ranges[1] - ranges[0];
    for index in 2..ranges.len() {
        let diff = (ranges[index] - ranges[index - 1]) - dr0;
        if diff.abs() > atol {
            return Ok(None);
        }
    }
    Ok(Some(dr0))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lowpass_maesaka_term, module)?)?;
    module.add_function(wrap_pyfunction!(lowpass_maesaka_jac, module)?)?;
    module.add_function(wrap_pyfunction!(forward_reverse_phidp, module)?)?;
    module.add_function(wrap_pyfunction!(kdp_range_resolution_uniform, module)?)?;
    Ok(())
}

fn validate_term_shape(nr: usize, ng: usize) -> PyResult<()> {
    if nr != 0 && ng != 0 && ng < 3 {
        return Err(PyValueError::new_err(
            "lowpass_maesaka_term requires at least 3 range gates",
        ));
    }
    Ok(())
}

fn validate_jac_shape(nr: usize, ng: usize) -> PyResult<()> {
    if nr != 0 && (ng == 1 || ng == 2 || ng == 3) {
        return Err(PyValueError::new_err(
            "lowpass_maesaka_jac received an unsupported range gate count",
        ));
    }
    Ok(())
}

fn validate_forward_reverse_inputs(
    k: &ndarray::ArrayView2<'_, f64>,
    phi_near: &ndarray::ArrayView1<'_, f64>,
    phi_far: &ndarray::ArrayView1<'_, f64>,
) -> PyResult<()> {
    let (nr, _) = k.dim();
    if !k.is_standard_layout() || !phi_near.is_standard_layout() || !phi_far.is_standard_layout() {
        return Err(PyValueError::new_err(
            "k, phi_near, and phi_far must be C-contiguous",
        ));
    }
    if phi_near.len() != nr || phi_far.len() != nr {
        return Err(PyValueError::new_err(
            "boundary condition arrays must match the number of rays",
        ));
    }
    Ok(())
}

fn validate_range_resolution_inputs(
    ranges: &ndarray::ArrayView1<'_, f64>,
    atol: f64,
) -> PyResult<()> {
    if ranges.len() < 2 {
        return Err(PyValueError::new_err(
            "range data must contain at least two gates",
        ));
    }
    if !ranges.is_standard_layout() {
        return Err(PyValueError::new_err("ranges must be C-contiguous"));
    }
    if !atol.is_finite() || atol < 0.0 {
        return Err(PyValueError::new_err(
            "atol must be finite and non-negative",
        ));
    }
    if ranges.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("ranges must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jac_shape_rejects_all_unsafe_small_gate_counts() {
        assert!(validate_jac_shape(1, 1).is_err());
        assert!(validate_jac_shape(1, 2).is_err());
        assert!(validate_jac_shape(1, 3).is_err());
        assert!(validate_jac_shape(1, 4).is_ok());
        assert!(validate_jac_shape(0, 2).is_ok());
    }

    #[test]
    fn forward_reverse_validation_rejects_boundary_length_mismatch() {
        let k = ndarray::Array2::<f64>::zeros((2, 3));
        let near = ndarray::Array1::<f64>::zeros(1);
        let far = ndarray::Array1::<f64>::zeros(2);
        assert!(validate_forward_reverse_inputs(&k.view(), &near.view(), &far.view()).is_err());
    }
}
