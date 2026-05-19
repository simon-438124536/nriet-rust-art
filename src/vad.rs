use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyfunction(name = "_average1d_equal")]
pub fn average1d_equal<'py>(
    py: Python<'py>,
    x_sorted: PyReadonlyArray1<'py, f64>,
    y_sorted: PyReadonlyArray1<'py, f64>,
    x_new: PyReadonlyArray1<'py, f64>,
    window: f64,
    fill_value: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_sorted = x_sorted.as_array();
    let y_sorted = y_sorted.as_array();
    let x_new = x_new.as_array();
    validate_average1d_inputs(x_sorted.len(), y_sorted.len())?;
    let out = average1d_equal_kernel(x_sorted, y_sorted, x_new, window, fill_value);
    Ok(PyArray1::from_owned_array(py, out))
}

#[pyfunction(name = "_average1d_idw")]
pub fn average1d_idw<'py>(
    py: Python<'py>,
    x_sorted: PyReadonlyArray1<'py, f64>,
    y_sorted: PyReadonlyArray1<'py, f64>,
    x_new: PyReadonlyArray1<'py, f64>,
    window: f64,
    fill_value: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_sorted = x_sorted.as_array();
    let y_sorted = y_sorted.as_array();
    let x_new = x_new.as_array();
    validate_average1d_inputs(x_sorted.len(), y_sorted.len())?;
    let (out, saw_zero_distance) =
        average1d_idw_kernel(x_sorted, y_sorted, x_new, window, fill_value);
    if saw_zero_distance {
        let runtime_warning = py.get_type::<PyRuntimeWarning>();
        PyErr::warn(
            py,
            &runtime_warning,
            c"divide by zero encountered in divide",
            1,
        )?;
        PyErr::warn(
            py,
            &runtime_warning,
            c"invalid value encountered in scalar divide",
            1,
        )?;
    }
    Ok(PyArray1::from_owned_array(py, out))
}

#[pyfunction(name = "_vad_interval_mean")]
pub fn interval_mean<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    current_z: PyReadonlyArray1<'py, f64>,
    wanted_z: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_array();
    let current_z = current_z.as_array();
    let wanted_z = wanted_z.as_array();
    validate_interval_mean_inputs(data.len(), current_z.len(), wanted_z.len())?;
    let (out, saw_empty_slice) = interval_mean_kernel(data, current_z, wanted_z);
    if saw_empty_slice {
        let runtime_warning = py.get_type::<PyRuntimeWarning>();
        PyErr::warn(py, &runtime_warning, c"Mean of empty slice.", 1)?;
        PyErr::warn(
            py,
            &runtime_warning,
            c"invalid value encountered in scalar divide",
            1,
        )?;
    }
    Ok(PyArray1::from_owned_array(py, out))
}

#[pyfunction(name = "_vad_inverse_dist_squared")]
pub fn inverse_dist_squared<'py>(
    py: Python<'py>,
    dist: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dist = dist.as_array();
    if !dist.is_standard_layout() {
        return Err(PyValueError::new_err("dist must be C-contiguous"));
    }
    if dist
        .iter()
        .any(|value| !value.is_finite() || value.abs() <= 1.0e-150 || value.abs() >= 1.0e150)
    {
        return Err(PyValueError::new_err(
            "dist must be finite, non-zero, and within the supported range",
        ));
    }

    Ok(PyArray1::from_owned_array(
        py,
        inverse_dist_squared_kernel(dist),
    ))
}

#[pyfunction(name = "_vad_calculation_b_dense")]
pub fn vad_calculation_b_dense<'py>(
    py: Python<'py>,
    velocities: PyReadonlyArray2<'py, f64>,
    sin_az: PyReadonlyArray1<'py, f64>,
    cos_az: PyReadonlyArray1<'py, f64>,
    elevation_scale: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let velocities = velocities.as_array();
    let sin_az = sin_az.as_array();
    let cos_az = cos_az.as_array();
    validate_vad_calculation_b_inputs(velocities, sin_az, cos_az, elevation_scale)?;
    let (u_mean, v_mean) = vad_calculation_b_kernel(velocities, sin_az, cos_az, elevation_scale);
    Ok((
        PyArray1::from_owned_array(py, u_mean),
        PyArray1::from_owned_array(py, v_mean),
    ))
}

#[pyfunction(name = "_vad_calculation_m_dense")]
pub fn vad_calculation_m_dense<'py>(
    py: Python<'py>,
    velocity_field: &Bound<'py, PyAny>,
    sin_az: &Bound<'py, PyAny>,
    cos_az: &Bound<'py, PyAny>,
    elevation_scale: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    reject_vad_masked_array(py, "velocity_field", velocity_field)?;
    reject_vad_masked_array(py, "sin_az", sin_az)?;
    reject_vad_masked_array(py, "cos_az", cos_az)?;

    let velocity_field = velocity_field
        .extract::<PyReadonlyArray2<'py, f64>>()
        .map_err(|_| PyValueError::new_err("velocity_field must be a 2D float64 array"))?;
    let sin_az = sin_az
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("sin_az must be a 1D float64 array"))?;
    let cos_az = cos_az
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("cos_az must be a 1D float64 array"))?;
    let velocity_field = velocity_field.as_array();
    let sin_az = sin_az.as_array();
    let cos_az = cos_az.as_array();
    validate_vad_calculation_m_inputs(velocity_field, sin_az, cos_az, elevation_scale)?;
    let (speed, angle) = vad_calculation_m_kernel(velocity_field, sin_az, cos_az, elevation_scale);
    Ok((
        PyArray1::from_owned_array(py, speed),
        PyArray1::from_owned_array(py, angle),
    ))
}

fn validate_average1d_inputs(x_len: usize, y_len: usize) -> PyResult<()> {
    if x_len != y_len {
        return Err(PyValueError::new_err(
            "x_sorted and y_sorted must have the same length",
        ));
    }
    Ok(())
}

fn validate_vad_calculation_b_inputs(
    velocities: ndarray::ArrayView2<'_, f64>,
    sin_az: ndarray::ArrayView1<'_, f64>,
    cos_az: ndarray::ArrayView1<'_, f64>,
    elevation_scale: f64,
) -> PyResult<()> {
    let (nrays, _nbins) = velocities.dim();
    if nrays == 0 {
        return Err(PyValueError::new_err(
            "velocities must include at least one ray",
        ));
    }
    if sin_az.len() != nrays || cos_az.len() != nrays {
        return Err(PyValueError::new_err(
            "sin_az and cos_az must match the number of velocity rays",
        ));
    }
    if !elevation_scale.is_finite() {
        return Err(PyValueError::new_err("elevation_scale must be finite"));
    }
    if velocities.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("velocities must be finite"));
    }
    if sin_az.iter().any(|value| !value.is_finite())
        || cos_az.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err("sin_az and cos_az must be finite"));
    }
    let determinant = vad_design_determinant(sin_az, cos_az);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(PyValueError::new_err(
            "azimuth design matrix must be finite and non-singular",
        ));
    }
    Ok(())
}

fn reject_vad_masked_array(py: Python<'_>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
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

fn validate_vad_calculation_m_inputs(
    velocity_field: ndarray::ArrayView2<'_, f64>,
    sin_az: ndarray::ArrayView1<'_, f64>,
    cos_az: ndarray::ArrayView1<'_, f64>,
    elevation_scale: f64,
) -> PyResult<()> {
    let (nrays, _nbins) = velocity_field.dim();
    if !velocity_field.is_standard_layout() {
        return Err(PyValueError::new_err("velocity_field must be C-contiguous"));
    }
    if !sin_az.is_standard_layout() || !cos_az.is_standard_layout() {
        return Err(PyValueError::new_err(
            "sin_az and cos_az must be C-contiguous",
        ));
    }
    if nrays == 0 || nrays % 2 != 0 {
        return Err(PyValueError::new_err(
            "velocity_field must include a positive even number of rays",
        ));
    }
    if sin_az.len() != nrays || cos_az.len() != nrays {
        return Err(PyValueError::new_err(
            "sin_az and cos_az must match the number of velocity rays",
        ));
    }
    if !elevation_scale.is_finite() {
        return Err(PyValueError::new_err("elevation_scale must be finite"));
    }
    if velocity_field.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("velocity_field must be finite"));
    }
    if sin_az.iter().any(|value| !value.is_finite())
        || cos_az.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err("sin_az and cos_az must be finite"));
    }

    let (sum_sin_squared, sum_sin_cos, sum_cos_squared) = vad_design_terms(sin_az, cos_az);
    if !sum_sin_squared.is_finite() || sum_sin_squared == 0.0 {
        return Err(PyValueError::new_err(
            "azimuth design matrix must have non-zero sin-squared weight",
        ));
    }
    let denominator = sum_cos_squared - sum_sin_cos * sum_sin_cos / sum_sin_squared;
    if !denominator.is_finite() || denominator == 0.0 {
        return Err(PyValueError::new_err(
            "azimuth design matrix must be finite and non-singular",
        ));
    }
    Ok(())
}

fn validate_interval_mean_inputs(
    data_len: usize,
    current_z_len: usize,
    wanted_z_len: usize,
) -> PyResult<()> {
    if data_len != current_z_len {
        return Err(PyValueError::new_err(
            "data and current_z must have the same length",
        ));
    }
    if current_z_len == 0 {
        return Err(PyValueError::new_err(
            "current_z must include at least one value",
        ));
    }
    if wanted_z_len < 2 {
        return Err(PyValueError::new_err(
            "wanted_z must include at least two values",
        ));
    }
    Ok(())
}

fn average1d_equal_kernel(
    x_sorted: ndarray::ArrayView1<'_, f64>,
    y_sorted: ndarray::ArrayView1<'_, f64>,
    x_new: ndarray::ArrayView1<'_, f64>,
    window: f64,
    fill_value: f64,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(x_new.len());
    for (i, &center) in x_new.iter().enumerate() {
        let (start, stop) = search_window(x_sorted, center, window);
        if start >= stop {
            out[i] = fill_value;
            continue;
        }
        let mut total = 0.0;
        for idx in start..stop {
            total += y_sorted[idx];
        }
        out[i] = total / (stop - start) as f64;
    }
    out
}

fn inverse_dist_squared_kernel(dist: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(dist.len());
    for (slot, &value) in out.iter_mut().zip(dist.iter()) {
        *slot = 1.0 / (value * value);
    }
    out
}

fn average1d_idw_kernel(
    x_sorted: ndarray::ArrayView1<'_, f64>,
    y_sorted: ndarray::ArrayView1<'_, f64>,
    x_new: ndarray::ArrayView1<'_, f64>,
    window: f64,
    fill_value: f64,
) -> (Array1<f64>, bool) {
    let mut out = Array1::<f64>::zeros(x_new.len());
    let mut saw_zero_distance = false;
    for (i, &center) in x_new.iter().enumerate() {
        let (start, stop) = search_window(x_sorted, center, window);
        if start >= stop {
            out[i] = fill_value;
            continue;
        }

        let mut weighted_total = 0.0;
        let mut weight_total = 0.0;
        for idx in start..stop {
            let dist = x_sorted[idx] - center;
            if dist == 0.0 {
                saw_zero_distance = true;
            }
            let mut weight = 1.0 / (dist * dist);
            if weight.is_nan() {
                weight = 99999.0;
            }
            weighted_total += y_sorted[idx] * weight;
            weight_total += weight;
        }
        out[i] = weighted_total / weight_total;
    }
    (out, saw_zero_distance)
}

fn search_window(
    x_sorted: ndarray::ArrayView1<'_, f64>,
    center: f64,
    window: f64,
) -> (usize, usize) {
    let bottom = center - window;
    let top = center + window;
    (
        searchsorted_left(x_sorted, bottom),
        searchsorted_left(x_sorted, top),
    )
}

fn searchsorted_left(x_sorted: ndarray::ArrayView1<'_, f64>, value: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = x_sorted.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x_sorted[mid] < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn interval_mean_kernel(
    data: ndarray::ArrayView1<'_, f64>,
    current_z: ndarray::ArrayView1<'_, f64>,
    wanted_z: ndarray::ArrayView1<'_, f64>,
) -> (Array1<f64>, bool) {
    let delta = wanted_z[1] - wanted_z[0];
    let mut out = Array1::<f64>::zeros(wanted_z.len());
    let mut saw_empty_slice = false;
    for (i, &center) in wanted_z.iter().enumerate() {
        let lower_target = center - delta / 2.0;
        let upper_target = center + delta / 2.0;
        let lower = nearest_squared_index(current_z, lower_target);
        let upper = nearest_squared_index(current_z, upper_target);
        if lower >= upper {
            out[i] = f64::NAN;
            saw_empty_slice = true;
            continue;
        }
        let mut total = 0.0;
        for idx in lower..upper {
            total += data[idx];
        }
        out[i] = total / (upper - lower) as f64;
    }
    (out, saw_empty_slice)
}

fn nearest_squared_index(values: ndarray::ArrayView1<'_, f64>, target: f64) -> usize {
    let mut best_index = 0usize;
    let mut best_distance = (values[0] - target).powi(2);
    for (idx, &value) in values.iter().enumerate().skip(1) {
        let distance = (value - target).powi(2);
        if distance < best_distance {
            best_distance = distance;
            best_index = idx;
        }
    }
    best_index
}

fn vad_calculation_b_kernel(
    velocities: ndarray::ArrayView2<'_, f64>,
    sin_az: ndarray::ArrayView1<'_, f64>,
    cos_az: ndarray::ArrayView1<'_, f64>,
    elevation_scale: f64,
) -> (Array1<f64>, Array1<f64>) {
    let (nrays, nbins) = velocities.dim();
    let mut u_mean = Array1::<f64>::zeros(nbins);
    let mut v_mean = Array1::<f64>::zeros(nbins);

    for gate in 0..nbins {
        let mut total = 0.0;
        for ray in 0..nrays {
            total += velocities[[ray, gate]];
        }
        let mean_velocity = total / nrays as f64;

        let mut sum_cos_vel_dev = 0.0;
        let mut sum_sin_vel_dev = 0.0;
        let mut sum_sin_cos_az = 0.0;
        let mut sum_sin_squared_az = 0.0;
        let mut sum_cos_squared_az = 0.0;

        for ray in 0..nrays {
            let sin = sin_az[ray];
            let cos = cos_az[ray];
            let velocity_deviation = velocities[[ray, gate]] - mean_velocity;
            sum_cos_vel_dev += cos * velocity_deviation;
            sum_sin_vel_dev += sin * velocity_deviation;
            sum_sin_cos_az += sin * cos;
            sum_sin_squared_az += sin * sin;
            sum_cos_squared_az += cos * cos;
        }

        let a = sum_sin_squared_az;
        let b = sum_sin_cos_az;
        let c = sum_sin_cos_az;
        let d = sum_cos_squared_az;
        let b_1 = sum_sin_vel_dev;
        let b_2 = sum_cos_vel_dev;
        let determinant = a * d - b * c;
        let x_1 = (d * b_1 - b * b_2) / determinant;
        let x_2 = (a * b_2 - c * b_1) / determinant;
        u_mean[gate] = x_1 * elevation_scale;
        v_mean[gate] = x_2 * elevation_scale;
    }

    (u_mean, v_mean)
}

fn vad_calculation_m_kernel(
    velocity_field: ndarray::ArrayView2<'_, f64>,
    sin_az: ndarray::ArrayView1<'_, f64>,
    cos_az: ndarray::ArrayView1<'_, f64>,
    elevation_scale: f64,
) -> (Array1<f64>, Array1<f64>) {
    let (nrays, nbins) = velocity_field.dim();
    let half_rays = nrays / 2;
    let mut speed = Array1::<f64>::zeros(nbins);
    let mut angle = Array1::<f64>::zeros(nbins);

    let (sum_sin_squared, sum_sin_cos, sum_cos_squared) = vad_design_terms(sin_az, cos_az);
    let denominator = sum_cos_squared - sum_sin_cos * sum_sin_cos / sum_sin_squared;

    for gate in 0..nbins {
        let mut pair_sum = 0.0;
        for ray in 0..half_rays {
            pair_sum += velocity_field[[ray, gate]] + velocity_field[[ray + half_rays, gate]];
        }
        let u_m = (pair_sum / nrays as f64).floor();

        let mut sum_cminusu_mcos = 0.0;
        let mut sum_cminusu_msin = 0.0;
        for ray in 0..nrays {
            let centered = velocity_field[[ray, gate]] - u_m;
            sum_cminusu_mcos += cos_az[ray] * centered;
            sum_cminusu_msin += sin_az[ray] * centered;
        }

        let b_value =
            (sum_cminusu_mcos - (sum_sin_cos * sum_cminusu_msin / sum_sin_squared)) / denominator;
        let a_value = (sum_cminusu_msin - b_value * sum_sin_cos) / sum_sin_squared;
        speed[gate] = (a_value * a_value + b_value * b_value).sqrt() * elevation_scale;
        angle[gate] = a_value.atan2(b_value);
    }

    (speed, angle)
}

fn vad_design_terms(
    sin_az: ndarray::ArrayView1<'_, f64>,
    cos_az: ndarray::ArrayView1<'_, f64>,
) -> (f64, f64, f64) {
    let mut sum_sin_cos_az = 0.0;
    let mut sum_sin_squared_az = 0.0;
    let mut sum_cos_squared_az = 0.0;
    for (&sin, &cos) in sin_az.iter().zip(cos_az.iter()) {
        sum_sin_cos_az += sin * cos;
        sum_sin_squared_az += sin * sin;
        sum_cos_squared_az += cos * cos;
    }
    (sum_sin_squared_az, sum_sin_cos_az, sum_cos_squared_az)
}

fn vad_design_determinant(
    sin_az: ndarray::ArrayView1<'_, f64>,
    cos_az: ndarray::ArrayView1<'_, f64>,
) -> f64 {
    let (sum_sin_squared_az, sum_sin_cos_az, sum_cos_squared_az) = vad_design_terms(sin_az, cos_az);
    sum_sin_squared_az * sum_cos_squared_az - sum_sin_cos_az * sum_sin_cos_az
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(average1d_equal, module)?)?;
    module.add_function(wrap_pyfunction!(average1d_idw, module)?)?;
    module.add_function(wrap_pyfunction!(interval_mean, module)?)?;
    module.add_function(wrap_pyfunction!(inverse_dist_squared, module)?)?;
    module.add_function(wrap_pyfunction!(vad_calculation_b_dense, module)?)?;
    module.add_function(wrap_pyfunction!(vad_calculation_m_dense, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn search_window_matches_left_closed_right_open_numpy_window() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        assert_eq!(search_window(x.view(), 1.5, 1.0), (1, 3));
        assert_eq!(search_window(x.view(), 0.0, 0.0), (0, 0));
        assert_eq!(search_window(x.view(), 2.0, -1.0), (3, 1));
    }

    #[test]
    fn inverse_dist_squared_matches_reference() {
        let dist = array![-2.0, -0.5, 0.5, 2.0];
        let expected = array![0.25, 4.0, 4.0, 0.25];

        assert_eq!(inverse_dist_squared_kernel(dist.view()), expected);
    }

    #[test]
    fn average_equal_uses_fill_for_empty_windows() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![10.0, 20.0, 30.0];
        let x_new = array![0.5, 10.0];
        let out = average1d_equal_kernel(x.view(), y.view(), x_new.view(), 0.6, 99999.0);
        assert_eq!(out, array![15.0, 99999.0]);
    }

    #[test]
    fn interval_mean_uses_nearest_bounds_and_python_slice_exclusion() {
        let data = array![10.0, 20.0, 30.0, 40.0];
        let current_z = array![0.0, 1.0, 2.0, 3.0];
        let wanted_z = array![1.5, 2.5];
        let (out, saw_empty) = interval_mean_kernel(data.view(), current_z.view(), wanted_z.view());
        assert!(!saw_empty);
        assert_eq!(out, array![20.0, 30.0]);
    }

    #[test]
    fn interval_mean_marks_empty_slices_without_panicking() {
        let data = array![10.0, 20.0];
        let current_z = array![0.0, 10.0];
        let wanted_z = array![0.0, 1.0];
        let (out, saw_empty) = interval_mean_kernel(data.view(), current_z.view(), wanted_z.view());
        assert!(saw_empty);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
    }

    #[test]
    fn vad_calculation_b_matches_known_dense_solution() {
        let velocities = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let sin_az = array![0.0, 1.0, 0.0, -1.0];
        let cos_az = array![1.0, 0.0, -1.0, 0.0];

        let (u, v) = vad_calculation_b_kernel(velocities.view(), sin_az.view(), cos_az.view(), 1.0);

        assert_eq!(u, array![-2.0, -2.0]);
        assert_eq!(v, array![-2.0, -2.0]);
    }

    #[test]
    fn vad_calculation_m_uses_floor_mean_and_returns_speed_angle() {
        let velocities = ndarray::arr2(&[[-1.1, 2.2], [3.3, -4.4], [5.5, 6.6], [-7.7, 8.8]]);
        let sin_az = array![0.0, 1.0, 0.0, -1.0];
        let cos_az = array![1.0, 0.0, -1.0, 0.0];

        let (speed, angle) =
            vad_calculation_m_kernel(velocities.view(), sin_az.view(), cos_az.view(), 1.0);

        let expected_speed = array![6.4140470843298305, 6.957010852370435];
        let expected_angle = array![2.1112158270654806, -1.892546881191539];
        for idx in 0..speed.len() {
            assert!((speed[idx] - expected_speed[idx]).abs() < 1.0e-12);
            assert!((angle[idx] - expected_angle[idx]).abs() < 1.0e-12);
        }
    }
}
