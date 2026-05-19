use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "_storm_relative_velocity_inplace")]
pub fn storm_relative_velocity_inplace(
    sr_data: Bound<'_, PyArray2<f64>>,
    velocity_data: Bound<'_, PyArray2<f64>>,
    angle_array: Bound<'_, PyArray1<f64>>,
    speed: f64,
    alpha: f64,
    start: isize,
    stop: isize,
) -> PyResult<()> {
    let velocity = {
        let velocity = velocity_data
            .try_readonly()
            .map_err(|_| PyValueError::new_err("velocity_data is already mutably borrowed"))?;
        velocity.as_array().to_owned()
    };
    let angles = {
        let angles = angle_array
            .try_readonly()
            .map_err(|_| PyValueError::new_err("angle_array is already mutably borrowed"))?;
        angles.as_array().to_owned()
    };
    let mut sr_data = sr_data
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("sr_data must be writable and unborrowed"))?;
    let mut output = sr_data.as_array_mut();

    let (rows, cols) = output.dim();
    if velocity.dim() != (rows, cols) {
        return Err(PyValueError::new_err(
            "sr_data and velocity_data must have the same shape",
        ));
    }
    if start < 0 || stop < start {
        return Err(PyValueError::new_err("invalid ray range"));
    }
    let start = start as usize;
    let stop = stop as usize;
    if stop > rows {
        return Err(PyValueError::new_err(
            "ray range exceeds velocity_data rows",
        ));
    }
    if angles.len() < stop - start {
        return Err(PyValueError::new_err(
            "angle_array length must cover ray range",
        ));
    }

    for (offset, ray) in (start..stop).enumerate() {
        let correction = speed * (alpha - angles[offset]).to_radians().cos();
        for gate in 0..cols {
            output[[ray, gate]] = velocity[[ray, gate]] - correction;
        }
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(storm_relative_velocity_inplace, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn rust_degrees_cosine_matches_expected_quadrants() {
        let speed = 10.0_f64;
        assert_eq!(speed * (0.0_f64 - 0.0_f64).to_radians().cos(), 10.0);
        assert!((speed * (90.0_f64 - 0.0_f64).to_radians().cos()).abs() < 1.0e-12);
        assert_eq!(speed * (180.0_f64 - 0.0_f64).to_radians().cos(), -10.0);
    }
}
