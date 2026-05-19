use ndarray::{Array1, Array2, ArrayD, IxDyn, Zip};
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

const SIMULATED_VEL_MAX_OUTPUT_VALUES: usize = 512 * 1024 * 1024;
const IMAGE_MUTE_MAX_OUTPUT_VALUES: usize = 512 * 1024 * 1024;
const COLUMNSECT_MAX_RAYS: usize = 1024 * 1024;

#[pyfunction]
pub fn estimate_noise_hs74(
    spectrum: PyReadonlyArray1<'_, f64>,
    navg: f64,
    nnoise_min: usize,
) -> PyResult<(f64, f64, f64, usize)> {
    if navg == 0.0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }

    let mut sorted_spectrum = spectrum.as_slice()?.to_vec();
    if sorted_spectrum.is_empty() {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
    }
    sorted_spectrum.sort_by(|left, right| left.total_cmp(right));

    let mut nnoise = sorted_spectrum.len();
    let rtest = 1.0 + 1.0 / navg;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    for (i, pwr) in sorted_spectrum.iter().copied().enumerate() {
        let npts = i + 1;
        sum1 += pwr;
        sum2 += pwr * pwr;

        if npts < nnoise_min {
            continue;
        }

        if (npts as f64) * sum2 < sum1 * sum1 * rtest {
            nnoise = npts;
        } else {
            sum1 -= pwr;
            sum2 -= pwr * pwr;
            break;
        }
    }

    let mean = sum1 / nnoise as f64;
    let var = sum2 / nnoise as f64 - mean * mean;
    let threshold = sorted_spectrum[nnoise - 1];
    Ok((mean, threshold, var, nnoise))
}

#[pyfunction(name = "_mean_of_two_angles")]
pub fn mean_of_two_angles<'py>(
    py: Python<'py>,
    angles1: PyReadonlyArrayDyn<'py, f64>,
    angles2: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let angles1 = angles1.as_array();
    let angles2 = angles2.as_array();
    let shape = broadcast_shape(&[angles1.shape(), angles2.shape()])?;

    let angles1 = angles1
        .broadcast(IxDyn(&shape))
        .ok_or_else(broadcast_error)?;
    let angles2 = angles2
        .broadcast(IxDyn(&shape))
        .ok_or_else(broadcast_error)?;
    let mut output = ArrayD::<f64>::zeros(IxDyn(&shape));

    Zip::from(&mut output)
        .and(angles1)
        .and(angles2)
        .for_each(|slot, &angle1, &angle2| {
            let x = (angle1.cos() + angle2.cos()) / 2.0;
            let y = (angle1.sin() + angle2.sin()) / 2.0;
            *slot = y.atan2(x);
        });

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_angular_mean")]
pub fn angular_mean(angles: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
    let angles = angles.as_array();
    if angles.is_empty() {
        return Ok(f64::NAN);
    }

    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    for &angle in angles.iter() {
        x_sum += angle.cos();
        y_sum += angle.sin();
    }
    let count = angles.len() as f64;
    Ok((y_sum / count).atan2(x_sum / count))
}

#[pyfunction(name = "_angular_std")]
pub fn angular_std(angles: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
    let angles = angles.as_array();
    if angles.is_empty() {
        return Ok(f64::NAN);
    }

    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    for &angle in angles.iter() {
        x_sum += angle.cos();
        y_sum += angle.sin();
    }
    let count = angles.len() as f64;
    let x_mean = x_sum / count;
    let y_mean = y_sum / count;
    let norm = (x_mean * x_mean + y_mean * y_mean).sqrt();
    Ok((-2.0 * norm.ln()).sqrt())
}

#[pyfunction(name = "_interval_mean")]
pub fn interval_mean(
    dist: PyReadonlyArrayDyn<'_, f64>,
    interval_min: f64,
    interval_max: f64,
) -> PyResult<f64> {
    let half_width = (interval_max - interval_min) / 2.0;
    let center = interval_min + half_width;
    let dist = dist.as_array();
    if dist.is_empty() {
        return Ok(f64::NAN);
    }

    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    for &value in dist.iter() {
        let angle = (value - center) / half_width * std::f64::consts::PI;
        x_sum += angle.cos();
        y_sum += angle.sin();
    }
    let count = dist.len() as f64;
    let angle_mean = (y_sum / count).atan2(x_sum / count);
    Ok((angle_mean * half_width / std::f64::consts::PI) + center)
}

#[pyfunction(name = "_interval_std")]
pub fn interval_std(
    dist: PyReadonlyArrayDyn<'_, f64>,
    interval_min: f64,
    interval_max: f64,
) -> PyResult<f64> {
    let half_width = (interval_max - interval_min) / 2.0;
    let center = interval_min + half_width;
    let dist = dist.as_array();
    if dist.is_empty() {
        return Ok(f64::NAN);
    }

    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    for &value in dist.iter() {
        let angle = (value - center) / half_width * std::f64::consts::PI;
        x_sum += angle.cos();
        y_sum += angle.sin();
    }
    let count = dist.len() as f64;
    let x_mean = x_sum / count;
    let y_mean = y_sum / count;
    let norm = (x_mean * x_mean + y_mean * y_mean).sqrt();
    Ok((-2.0 * norm.ln()).sqrt() * half_width / std::f64::consts::PI)
}

#[pyfunction(name = "_compute_directional_stats_mean_dense_f64")]
pub fn compute_directional_stats_mean_dense_f64<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    axis: &Bound<'_, PyAny>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
    let numpy_ma = py.import("numpy.ma")?;
    if numpy_ma
        .getattr("isMaskedArray")?
        .call1((field,))?
        .extract::<bool>()?
    {
        return Err(PyValueError::new_err("field must be mask-free"));
    }

    let axis = extract_directional_axis(axis)?;
    let field = field
        .extract::<PyReadonlyArrayDyn<'py, f64>>()
        .map_err(|_| PyValueError::new_err("field must be a 2D float64 array"))?;
    let array = field.as_array();
    if array.ndim() != 2 {
        return Err(PyValueError::new_err("field must be a 2D float64 array"));
    }
    if axis != 0 && axis != 1 {
        return Err(PyValueError::new_err("axis must be 0 or 1"));
    }
    if array.is_empty() {
        return Err(PyValueError::new_err("field must be non-empty"));
    }
    if !array.is_standard_layout() {
        return Err(PyValueError::new_err("field must be C-contiguous"));
    }

    let data = field.as_slice()?;
    if data.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "field must contain only finite values",
        ));
    }

    let shape = array.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    let (values, nvalid) = if axis == 0 {
        let mut values = vec![0.0; ncols];
        for row in data.chunks_exact(ncols) {
            for (col, value) in row.iter().copied().enumerate() {
                values[col] += value;
            }
        }
        for value in &mut values {
            *value /= nrows as f64;
        }
        (values, vec![nrows as i64; ncols])
    } else {
        let mut values = Vec::with_capacity(nrows);
        for row in data.chunks_exact(ncols) {
            values.push(row.iter().copied().sum::<f64>() / ncols as f64);
        }
        (values, vec![ncols as i64; nrows])
    };

    Ok((
        PyArrayDyn::from_owned_array(py, Array1::from_vec(values).into_dyn()),
        PyArrayDyn::from_owned_array(py, Array1::from_vec(nvalid).into_dyn()),
    ))
}

#[pyfunction(name = "_simulated_radial_velocity_dense_f64")]
pub fn simulated_radial_velocity_dense_f64<'py>(
    py: Python<'py>,
    gate_u: PyReadonlyArray2<'py, f64>,
    gate_v: PyReadonlyArray2<'py, f64>,
    sin_azimuths: PyReadonlyArray1<'py, f64>,
    cos_azimuths: PyReadonlyArray1<'py, f64>,
    cos_elevations: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let gate_u = gate_u.as_array();
    let gate_v = gate_v.as_array();
    let sin_azimuths = sin_azimuths.as_array();
    let cos_azimuths = cos_azimuths.as_array();
    let cos_elevations = cos_elevations.as_array();

    let values = simulated_radial_velocity_values_f64(
        &gate_u,
        &gate_v,
        &sin_azimuths,
        &cos_azimuths,
        &cos_elevations,
    )?;
    Ok(PyArray2::from_owned_array(py, values))
}

#[pyfunction(name = "_image_mute_mask_dense_f64")]
pub fn image_mute_mask_dense_f64<'py>(
    py: Python<'py>,
    data_to_mute: PyReadonlyArrayDyn<'py, f64>,
    data_mute_by: PyReadonlyArrayDyn<'py, f64>,
    mute_threshold: f64,
    has_field_threshold: bool,
    field_threshold: f64,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let data_to_mute = data_to_mute.as_array();
    let data_mute_by = data_mute_by.as_array();
    let mask = image_mute_mask_values_f64(
        &data_to_mute,
        &data_mute_by,
        mute_threshold,
        has_field_threshold,
        field_threshold,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_columnsect_get_sweep_rays_f64")]
pub fn columnsect_get_sweep_rays_f64<'py>(
    py: Python<'py>,
    sweep_azi: PyReadonlyArray1<'py, f64>,
    azimuth: &Bound<'_, PyAny>,
    spread_threshold: &Bound<'_, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let azimuth = extract_non_bool_f64(azimuth, "azimuth")?;
    let spread_threshold = extract_non_bool_f64(spread_threshold, "spread_threshold")?;
    if !azimuth.is_finite() || !spread_threshold.is_finite() {
        return Err(PyValueError::new_err(
            "azimuth and spread_threshold must be finite",
        ));
    }

    let sweep_azi = sweep_azi.as_array();
    if sweep_azi.len() < 2 {
        return Err(PyValueError::new_err(
            "sweep_azi must contain at least two rays",
        ));
    }
    if sweep_azi.len() > COLUMNSECT_MAX_RAYS {
        return Err(PyValueError::new_err("sweep_azi exceeds native safety cap"));
    }
    if !sweep_azi.is_standard_layout() {
        return Err(PyValueError::new_err("sweep_azi must be C-contiguous"));
    }
    if sweep_azi.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "sweep_azi must contain only finite values",
        ));
    }

    let mut centerline = Vec::<i64>::new();
    let mut spread = Vec::<i64>::new();
    for (idx, &value) in sweep_azi.iter().enumerate() {
        let distance = (value - azimuth).abs();
        if distance < 0.5 {
            centerline.push(idx as i64);
        }
        if distance < spread_threshold {
            spread.push(idx as i64);
        }
    }

    Ok((
        PyArray1::from_owned_array(py, Array1::from_vec(centerline)),
        PyArray1::from_owned_array(py, Array1::from_vec(spread)),
    ))
}

#[pyfunction(name = "_columnsect_nearest_ray_index_f64")]
pub fn columnsect_nearest_ray_index_f64(
    sweep_azi: PyReadonlyArray1<'_, f64>,
    azimuth: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let azimuth = extract_non_bool_f64(azimuth, "azimuth")?;
    if !azimuth.is_finite() {
        return Err(PyValueError::new_err("azimuth must be finite"));
    }

    let sweep_azi = sweep_azi.as_array();
    if sweep_azi.is_empty() {
        return Err(PyValueError::new_err(
            "attempt to get argmin of an empty sequence",
        ));
    }
    if sweep_azi.len() > COLUMNSECT_MAX_RAYS {
        return Err(PyValueError::new_err("sweep_azi exceeds native safety cap"));
    }
    if !sweep_azi.is_standard_layout() {
        return Err(PyValueError::new_err("sweep_azi must be C-contiguous"));
    }

    Ok(nearest_ray_index_values(&sweep_azi, azimuth) as i64)
}

#[pyfunction(name = "_columnsect_get_column_rays_rhi_f64")]
pub fn columnsect_get_column_rays_rhi_f64<'py>(
    py: Python<'py>,
    azimuths: PyReadonlyArray1<'py, f64>,
    sweep_starts: PyReadonlyArray1<'py, i64>,
    sweep_ends: PyReadonlyArray1<'py, i64>,
    azimuth: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let azimuth = extract_non_bool_f64(azimuth, "azimuth")?;
    if !azimuth.is_finite() {
        return Err(PyValueError::new_err("azimuth must be finite"));
    }

    let azimuths = azimuths.as_array();
    let sweep_starts = sweep_starts.as_array();
    let sweep_ends = sweep_ends.as_array();
    if !azimuths.is_standard_layout()
        || !sweep_starts.is_standard_layout()
        || !sweep_ends.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "RHI column-ray inputs must be C-contiguous",
        ));
    }
    if azimuths.len() > COLUMNSECT_MAX_RAYS {
        return Err(PyValueError::new_err("azimuths exceeds native safety cap"));
    }
    if sweep_starts.len() != sweep_ends.len() {
        return Err(PyValueError::new_err(
            "sweep start/end arrays must have the same length",
        ));
    }

    let mut rays = Vec::<i64>::new();
    for (&start, &stop) in sweep_starts.iter().zip(sweep_ends.iter()) {
        if start < 0 || stop < 0 {
            return Err(PyValueError::new_err(
                "sweep start/end indexes must be nonnegative",
            ));
        }
        let start = start as usize;
        let stop = stop as usize;
        if start > azimuths.len() || stop > azimuths.len() {
            return Err(PyValueError::new_err(
                "sweep start/end indexes exceed azimuth length",
            ));
        }
        for idx in start..stop {
            let distance = (azimuths[idx] - azimuth).abs();
            if distance < 1.0 {
                rays.push(idx as i64);
            }
        }
    }

    Ok(PyArray1::from_owned_array(py, Array1::from_vec(rays)))
}

#[pyfunction(name = "_xsect_nearest_angle_f64")]
pub fn xsect_nearest_angle_f64(
    values: PyReadonlyArray1<'_, f64>,
    target: &Bound<'_, PyAny>,
) -> PyResult<(i64, f64)> {
    let target = extract_non_bool_f64(target, "target")?;
    let values = values.as_array();
    if values.is_empty() {
        return Err(PyValueError::new_err(
            "attempt to get argmin of an empty sequence",
        ));
    }
    if values.len() > COLUMNSECT_MAX_RAYS {
        return Err(PyValueError::new_err("values exceeds native safety cap"));
    }
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }

    let index = nearest_ray_index_values(&values, target);
    let distance = (values[index] - target).abs();
    Ok((index as i64, distance))
}

fn extract_directional_axis(axis: &Bound<'_, PyAny>) -> PyResult<isize> {
    let type_name = axis.get_type().name()?.to_str()?.to_owned();
    if axis.is_instance_of::<pyo3::types::PyBool>() || type_name == "bool" || type_name == "bool_" {
        return Err(PyValueError::new_err("axis must be a non-boolean integer"));
    }
    axis.extract::<isize>()
        .map_err(|_| PyValueError::new_err("axis must be a non-boolean integer"))
}

fn extract_non_bool_f64(value: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let type_name = value.get_type().name()?.to_str()?.to_owned();
    if value.is_instance_of::<pyo3::types::PyBool>() || type_name == "bool" || type_name == "bool_"
    {
        return Err(PyValueError::new_err(format!("{name} must be numeric")));
    }
    value
        .extract::<f64>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be numeric")))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(estimate_noise_hs74, module)?)?;
    module.add_function(wrap_pyfunction!(mean_of_two_angles, module)?)?;
    module.add_function(wrap_pyfunction!(angular_mean, module)?)?;
    module.add_function(wrap_pyfunction!(angular_std, module)?)?;
    module.add_function(wrap_pyfunction!(interval_mean, module)?)?;
    module.add_function(wrap_pyfunction!(interval_std, module)?)?;
    module.add_function(wrap_pyfunction!(
        compute_directional_stats_mean_dense_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        simulated_radial_velocity_dense_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(image_mute_mask_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(columnsect_get_sweep_rays_f64, module)?)?;
    module.add_function(wrap_pyfunction!(columnsect_nearest_ray_index_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        columnsect_get_column_rays_rhi_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(xsect_nearest_angle_f64, module)?)?;
    Ok(())
}

fn simulated_radial_velocity_values_f64(
    gate_u: &ndarray::ArrayView2<'_, f64>,
    gate_v: &ndarray::ArrayView2<'_, f64>,
    sin_azimuths: &ndarray::ArrayView1<'_, f64>,
    cos_azimuths: &ndarray::ArrayView1<'_, f64>,
    cos_elevations: &ndarray::ArrayView1<'_, f64>,
) -> PyResult<Array2<f64>> {
    if !gate_u.is_standard_layout()
        || !gate_v.is_standard_layout()
        || !sin_azimuths.is_standard_layout()
        || !cos_azimuths.is_standard_layout()
        || !cos_elevations.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "simulated velocity inputs must be C-contiguous",
        ));
    }
    if gate_v.dim() != gate_u.dim() {
        return Err(PyValueError::new_err("gate_u and gate_v shapes must match"));
    }
    let (nrays, ngates) = gate_u.dim();
    if sin_azimuths.len() != nrays || cos_azimuths.len() != nrays || cos_elevations.len() != nrays {
        return Err(PyValueError::new_err(
            "azimuth and elevation vectors must match the ray dimension",
        ));
    }
    let output_len = nrays
        .checked_mul(ngates)
        .ok_or_else(|| PyValueError::new_err("simulated velocity output size overflow"))?;
    if output_len > SIMULATED_VEL_MAX_OUTPUT_VALUES {
        return Err(PyValueError::new_err(
            "simulated velocity output exceeds native safety cap",
        ));
    }
    if gate_u.iter().any(|value| !value.is_finite())
        || gate_v.iter().any(|value| !value.is_finite())
        || sin_azimuths.iter().any(|value| !value.is_finite())
        || cos_azimuths.iter().any(|value| !value.is_finite())
        || cos_elevations.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "simulated velocity inputs must be finite",
        ));
    }

    let mut out = Array2::<f64>::zeros((nrays, ngates));
    for ray in 0..nrays {
        let sin_az = sin_azimuths[ray];
        let cos_az = cos_azimuths[ray];
        let cos_el = cos_elevations[ray];
        for gate in 0..ngates {
            out[(ray, gate)] =
                (gate_u[(ray, gate)] * sin_az) * cos_el + (gate_v[(ray, gate)] * cos_az) * cos_el;
        }
    }
    Ok(out)
}

fn image_mute_mask_values_f64(
    data_to_mute: &ndarray::ArrayViewD<'_, f64>,
    data_mute_by: &ndarray::ArrayViewD<'_, f64>,
    mute_threshold: f64,
    has_field_threshold: bool,
    field_threshold: f64,
) -> PyResult<ArrayD<bool>> {
    if !data_to_mute.is_standard_layout() || !data_mute_by.is_standard_layout() {
        return Err(PyValueError::new_err(
            "image mute inputs must be C-contiguous",
        ));
    }
    if data_to_mute.shape() != data_mute_by.shape() {
        return Err(PyValueError::new_err(
            "image mute inputs must have identical shape",
        ));
    }
    if data_to_mute.len() > IMAGE_MUTE_MAX_OUTPUT_VALUES {
        return Err(PyValueError::new_err(
            "image mute output exceeds native safety cap",
        ));
    }

    let mut out = ArrayD::<bool>::from_elem(IxDyn(data_to_mute.shape()), false);
    for ((out_value, &field_value), &mute_value) in out
        .iter_mut()
        .zip(data_to_mute.iter())
        .zip(data_mute_by.iter())
    {
        let mute_filter = mute_value <= mute_threshold;
        *out_value = if has_field_threshold {
            mute_filter && field_value >= field_threshold
        } else {
            mute_filter
        };
    }
    Ok(out)
}

fn nearest_ray_index_values(sweep_azi: &ndarray::ArrayView1<'_, f64>, azimuth: f64) -> usize {
    let mut best_idx = 0_usize;
    let mut best_dist = (sweep_azi[0] - azimuth).abs();
    for (idx, &value) in sweep_azi.iter().enumerate().skip(1) {
        let dist = (value - azimuth).abs();
        if !best_dist.is_nan() && (dist.is_nan() || dist < best_dist) {
            best_idx = idx;
            best_dist = dist;
        }
    }
    best_idx
}

fn broadcast_shape(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let ndim = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut out = vec![1; ndim];

    for shape in shapes {
        for (axis_from_end, &dim) in shape.iter().rev().enumerate() {
            let axis = ndim - 1 - axis_from_end;
            let current = out[axis];
            if current == 1 {
                out[axis] = dim;
            } else if dim != 1 && dim != current {
                return Err(broadcast_error());
            }
        }
    }

    Ok(out)
}

fn broadcast_error() -> PyErr {
    PyValueError::new_err("operands could not be broadcast together")
}

#[cfg(test)]
mod tests {
    #[test]
    fn circular_scalar_formula_matches_reference() {
        let angle1 = 350.0_f64.to_radians();
        let angle2 = 10.0_f64.to_radians();
        let x = (angle1.cos() + angle2.cos()) / 2.0;
        let y = (angle1.sin() + angle2.sin()) / 2.0;

        assert!(y.atan2(x).abs() < 1.0e-15);
    }

    #[test]
    fn columnsect_index_scan_uses_strict_thresholds() {
        let values = ndarray::array![9.5, 10.0, 10.49, 10.5, 11.0];
        let mut centerline = Vec::<i64>::new();
        let mut spread = Vec::<i64>::new();
        for (idx, &value) in values.iter().enumerate() {
            let distance = f64::abs(value - 10.0);
            if distance < 0.5 {
                centerline.push(idx as i64);
            }
            if distance < 1.0 {
                spread.push(idx as i64);
            }
        }
        assert_eq!(centerline, vec![1, 2]);
        assert_eq!(spread, vec![0, 1, 2, 3]);
    }

    #[test]
    fn columnsect_nearest_uses_first_tie_and_first_nan() {
        let values = ndarray::array![10.5, 9.5, f64::NAN, 10.0];
        assert_eq!(super::nearest_ray_index_values(&values.view(), 10.0), 2);

        let values = ndarray::array![9.5, 10.5];
        assert_eq!(super::nearest_ray_index_values(&values.view(), 10.0), 0);
    }
}
