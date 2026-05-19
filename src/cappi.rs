use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const CAPPI_MAX_VOLUME_GATES: usize = 512 * 1024 * 1024;

#[pyfunction(name = "_cappi_height_index_f64")]
pub fn cappi_height_index_f64<'py>(
    py: Python<'py>,
    z_3d: PyReadonlyArray3<'py, f64>,
    height: f64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
    let z_3d = z_3d.as_array();
    let (height_idx, selected_gate_z) = cappi_height_index_values(&z_3d, height)?;
    Ok((
        PyArray2::from_owned_array(py, height_idx),
        PyArray2::from_owned_array(py, selected_gate_z),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(cappi_height_index_f64, module)?)?;
    Ok(())
}

fn cappi_height_index_values(
    z_3d: &ndarray::ArrayView3<'_, f64>,
    height: f64,
) -> PyResult<(Array2<i64>, Array2<f64>)> {
    if !z_3d.is_standard_layout() {
        return Err(PyValueError::new_err("z_3d must be C-contiguous"));
    }
    let (nsweeps, nrays, ngates) = z_3d.dim();
    if nsweeps == 0 {
        return Err(PyValueError::new_err(
            "attempt to get argmin of an empty sequence",
        ));
    }
    let volume_len = nsweeps
        .checked_mul(nrays)
        .and_then(|value| value.checked_mul(ngates))
        .ok_or_else(|| PyValueError::new_err("CAPPI volume size overflow"))?;
    if volume_len > CAPPI_MAX_VOLUME_GATES {
        return Err(PyValueError::new_err(
            "CAPPI volume exceeds native safety cap",
        ));
    }

    let mut height_idx = Array2::<i64>::zeros((nrays, ngates));
    let mut selected_gate_z = Array2::<f64>::zeros((nrays, ngates));

    for ray in 0..nrays {
        for gate in 0..ngates {
            let mut best_idx = 0_usize;
            let mut best_dist = (z_3d[(0, ray, gate)] - height).abs();
            for sweep in 1..nsweeps {
                let dist = (z_3d[(sweep, ray, gate)] - height).abs();
                if !best_dist.is_nan() && (dist.is_nan() || dist < best_dist) {
                    best_idx = sweep;
                    best_dist = dist;
                }
            }
            height_idx[(ray, gate)] = best_idx as i64;
            selected_gate_z[(ray, gate)] = z_3d[(best_idx, ray, gate)];
        }
    }

    Ok((height_idx, selected_gate_z))
}

#[cfg(test)]
mod tests {
    use super::cappi_height_index_values;
    use ndarray::array;

    #[test]
    fn height_index_uses_first_tie_and_first_nan() {
        let z = array![
            [[1000.0, 900.0, 10.0]],
            [[3000.0, f64::NAN, f64::NAN]],
            [[1000.0, f64::NAN, 1000.0]]
        ];
        let (idx, selected) = cappi_height_index_values(&z.view(), 1000.0).unwrap();
        assert_eq!(idx[(0, 0)], 0);
        assert_eq!(idx[(0, 1)], 1);
        assert_eq!(idx[(0, 2)], 1);
        assert_eq!(selected[(0, 0)], 1000.0);
        assert!(selected[(0, 1)].is_nan());
        assert!(selected[(0, 2)].is_nan());
    }

    #[test]
    fn height_nan_selects_first_sweep_like_numpy_argmin() {
        let z = array![
            [[1000.0, f64::NAN]],
            [[3000.0, f64::NAN]],
            [[2000.0, 500.0]]
        ];
        let (idx, selected) = cappi_height_index_values(&z.view(), f64::NAN).unwrap();
        assert_eq!(idx[(0, 0)], 0);
        assert_eq!(idx[(0, 1)], 0);
        assert_eq!(selected[(0, 0)], 1000.0);
        assert!(selected[(0, 1)].is_nan());
    }
}
