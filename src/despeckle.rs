use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "_despeckle_check_for_360")]
pub fn despeckle_check_for_360(az: PyReadonlyArray1<'_, f64>, delta: f64) -> PyResult<bool> {
    let az = az.as_array();
    if az.is_empty() {
        return Err(PyValueError::new_err("az must include at least one value"));
    }
    Ok(check_for_360_kernel(az, delta))
}

fn check_for_360_kernel(az: ndarray::ArrayView1<'_, f64>, delta: f64) -> bool {
    let edge_gap = (az[0] - az[az.len() - 1]).abs();
    if !(edge_gap < delta || edge_gap > 360.0 - delta) {
        return false;
    }

    let min_az = az.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let max_az = az
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    if max_az - min_az <= 360.0 - delta {
        return false;
    }

    let low_threshold = (360.0 - delta).to_radians().sin();
    let high_threshold = delta.to_radians().sin();
    let has_low = az
        .iter()
        .any(|value| value.to_radians().sin() < low_threshold);
    let has_high = az
        .iter()
        .any(|value| value.to_radians().sin() > high_threshold);
    has_low && has_high
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(despeckle_check_for_360, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn check_for_360_matches_core_examples() {
        assert!(check_for_360_kernel(
            array![0.0, 90.0, 180.0, 270.0, 359.0].view(),
            5.0
        ));
        assert!(!check_for_360_kernel(array![10.0, 20.0].view(), 5.0));
        assert!(!check_for_360_kernel(array![0.0].view(), 5.0));
    }
}
