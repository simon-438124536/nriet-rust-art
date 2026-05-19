use ndarray::{s, Array1, Array2, ArrayD, IxDyn, Zip};
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const HIGH_RELIABILITY: f64 = 9_999_999.0;
const NOMASK: u8 = 0;
const MASK: u8 = 1;
const CLOUD_THRESHOLD_MAX_ABS_DB: f64 = 1000.0;
const CLOUD_THRESHOLD_MAX_N_AVG: f64 = 1.0e12;
const CLOUD_MASK_BOX_WIDTH: usize = 4;
const RHOHV_NOISE_MAX_ABS_URHOHV: f64 = 1.0e6;
const RHOHV_NOISE_MAX_ABS_DB: f64 = 300.0;

#[pyfunction]
pub fn unwrap_1d(
    image: PyReadonlyArray1<'_, f64>,
    mut unwrapped_image: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let image_view = image.as_array();
    let mut out = unwrapped_image.as_array_mut();
    let n = image_view.len();

    if out.len() != n {
        return Err(PyValueError::new_err(
            "image and unwrapped_image must have the same length",
        ));
    }
    if n == 0 {
        return Err(PyIndexError::new_err(
            "index 0 is out of bounds for axis 0 with size 0",
        ));
    }

    let mut periods: i64 = 0;
    out[0] = image_view[0];
    for i in 1..n {
        let difference = image_view[i] - image_view[i - 1];
        if difference > std::f64::consts::PI {
            periods -= 1;
        } else if difference < -std::f64::consts::PI {
            periods += 1;
        }
        out[i] = image_view[i] + 2.0 * std::f64::consts::PI * periods as f64;
    }

    Ok(())
}

#[pyfunction(name = "_first_mask_f64")]
pub fn first_mask_f64<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    noise_threshold: f64,
) -> PyResult<Bound<'py, PyArrayDyn<i16>>> {
    let out = data.as_array().mapv(|value| {
        if value > noise_threshold {
            1_i16
        } else {
            0_i16
        }
    });
    Ok(PyArrayDyn::from_owned_array(py, out))
}

#[pyfunction(name = "_cloud_threshold_f64")]
pub fn cloud_threshold_f64(
    data: PyReadonlyArrayDyn<'_, f64>,
    n_avg: f64,
    nffts: usize,
) -> PyResult<f64> {
    let data = data.as_array();
    validate_cloud_threshold_inputs(&data, n_avg, nffts)?;
    Ok(cloud_threshold_kernel(&data, n_avg, nffts))
}

#[pyfunction(name = "_cloud_mask_4x4_count_i16")]
pub fn cloud_mask_4x4_count_i16<'py>(
    py: Python<'py>,
    mask1: PyReadonlyArray2<'py, i16>,
    counts_threshold: i64,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    let mask1 = mask1.as_array();
    validate_cloud_mask_4x4_inputs(&mask1, counts_threshold)?;
    let output = cloud_mask_4x4_count_kernel(&mask1, counts_threshold);
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction(name = "_correct_noise_rhohv_dense_f64")]
pub fn correct_noise_rhohv_dense_f64<'py>(
    py: Python<'py>,
    urhohv: PyReadonlyArrayDyn<'py, f64>,
    snrd_b_h: PyReadonlyArrayDyn<'py, f64>,
    zdr_db: PyReadonlyArrayDyn<'py, f64>,
    nh: PyReadonlyArrayDyn<'py, f64>,
    nv: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let urhohv = urhohv.as_array();
    let snrd_b_h = snrd_b_h.as_array();
    let zdr_db = zdr_db.as_array();
    let nh = nh.as_array();
    let nv = nv.as_array();
    validate_correct_noise_rhohv_inputs(&urhohv, &snrd_b_h, &zdr_db, &nh, &nv)?;

    let mut output = ArrayD::<f64>::zeros(IxDyn(urhohv.shape()));
    Zip::from(&mut output)
        .and(urhohv)
        .and(snrd_b_h)
        .and(zdr_db)
        .and(nh)
        .and(nv)
        .for_each(
            |slot, &urhohv_value, &snr_db_value, &zdr_db_value, &nh_value, &nv_value| {
                let snr_h = 10.0_f64.powf(0.1 * snr_db_value);
                let zdr = 10.0_f64.powf(0.1 * zdr_db_value);
                let alpha = 10.0_f64.powf(0.1 * (nh_value - nv_value));
                let mut rhohv_data =
                    urhohv_value * ((1.0 + 1.0 / snr_h) * (1.0 + zdr / (alpha * snr_h))).sqrt();
                if rhohv_data > 1.0 {
                    rhohv_data = 1.0;
                }
                *slot = rhohv_data;
            },
        );

    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_region_cost_function")]
pub fn region_cost_function<'py>(
    py: Python<'py>,
    nyq_vector: PyReadonlyArray1<'py, f64>,
    vels_slice_means: PyReadonlyArray1<'py, f64>,
    svels_slice_means: PyReadonlyArray1<'py, f64>,
    v_nyq_vel: f64,
    nfeatures: isize,
) -> PyResult<Py<PyAny>> {
    let nyq_vector = nyq_vector.as_array();
    let vels_slice_means = vels_slice_means.as_array();
    let svels_slice_means = svels_slice_means.as_array();
    validate_region_inputs(
        nyq_vector.len(),
        vels_slice_means.len(),
        svels_slice_means.len(),
        nfeatures,
    )?;
    let nfeatures = nfeatures.max(0) as usize;
    let cost = region_cost_function_kernel(
        nyq_vector,
        vels_slice_means,
        svels_slice_means,
        v_nyq_vel,
        nfeatures,
    );
    let np = py.import("numpy")?;
    Ok(np.getattr("float64")?.call1((cost,))?.unbind())
}

#[pyfunction(name = "_region_gradient")]
pub fn region_gradient<'py>(
    py: Python<'py>,
    nyq_vector: PyReadonlyArray1<'py, f64>,
    vels_slice_means: PyReadonlyArray1<'py, f64>,
    svels_slice_means: PyReadonlyArray1<'py, f64>,
    v_nyq_vel: f64,
    nfeatures: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let nyq_vector = nyq_vector.as_array();
    let vels_slice_means = vels_slice_means.as_array();
    let svels_slice_means = svels_slice_means.as_array();
    validate_region_inputs(
        nyq_vector.len(),
        vels_slice_means.len(),
        svels_slice_means.len(),
        nfeatures,
    )?;
    let nfeatures = nfeatures.max(0) as usize;
    let gradient = region_gradient_kernel(
        nyq_vector,
        vels_slice_means,
        svels_slice_means,
        v_nyq_vel,
        nfeatures,
    );
    Ok(PyArray1::from_owned_array(py, gradient))
}

#[pyfunction(name = "_region_sweep_interval_splits")]
pub fn region_sweep_interval_splits(
    nyquist: f64,
    interval_splits: isize,
    velocities: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, usize, bool)> {
    let velocities = velocities.as_array();
    validate_sweep_interval_inputs(nyquist, interval_splits, &velocities)?;
    sweep_interval_splits_kernel(nyquist, interval_splits as usize, &velocities)
}

type EdgeSumReturn<'py> = (
    (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>),
    Bound<'py, PyArray1<i32>>,
    (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>),
);

#[pyfunction(name = "_region_edge_sum_and_count")]
pub fn region_edge_sum_and_count<'py>(
    py: Python<'py>,
    index1: PyReadonlyArray1<'py, i32>,
    index2: PyReadonlyArray1<'py, i32>,
    vel1: PyReadonlyArray1<'py, f64>,
    vel2: PyReadonlyArray1<'py, f64>,
) -> PyResult<EdgeSumReturn<'py>> {
    let index1 = index1.as_array();
    let index2 = index2.as_array();
    let vel1 = vel1.as_array();
    let vel2 = vel2.as_array();
    validate_edge_sum_inputs(index1, index2, vel1, vel2)?;
    let (out_index1, out_index2, count, out_vel1, out_vel2) =
        edge_sum_and_count_kernel(index1, index2, vel1, vel2);
    Ok((
        (
            PyArray1::from_vec(py, out_index1),
            PyArray1::from_vec(py, out_index2),
        ),
        PyArray1::from_vec(py, count),
        (
            PyArray1::from_vec(py, out_vel1),
            PyArray1::from_vec(py, out_vel2),
        ),
    ))
}

#[pyfunction(name = "_phase_proc_smooth_and_trim_f64")]
pub fn phase_proc_smooth_and_trim_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    window_len: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array();
    let weights = weights.as_array();
    validate_smooth_and_trim_inputs(x, weights, window_len)?;
    let out = smooth_and_trim_kernel(x, weights, window_len as usize);
    Ok(PyArray1::from_owned_array(py, out))
}

#[pyfunction(name = "_phase_proc_smooth_and_trim_scan_f64")]
pub fn phase_proc_smooth_and_trim_scan_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    window_len: isize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = x.as_array();
    let weights = weights.as_array();
    validate_smooth_and_trim_scan_inputs(x, weights, window_len)?;
    let out = smooth_and_trim_scan_kernel(x, weights, window_len as usize);
    Ok(PyArray2::from_owned_array(py, out))
}

#[pyfunction(name = "_phase_proc_unwrap_masked_degrees_f64")]
pub fn phase_proc_unwrap_masked_degrees_f64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = values.as_array();
    let mask = mask.as_array();
    validate_unwrap_masked_inputs(values, mask)?;
    Ok(PyArray1::from_vec(
        py,
        unwrap_masked_degrees_kernel(values, mask),
    ))
}

#[pyfunction(name = "_phase_proc_det_sys_phase_dense")]
pub fn phase_proc_det_sys_phase_dense<'py>(
    py: Python<'py>,
    ncp: PyReadonlyArray2<'py, f64>,
    rhv: PyReadonlyArray2<'py, f64>,
    phidp: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    last_ray_idx: isize,
    ncp_lev: f64,
    rhv_lev: f64,
) -> PyResult<Py<PyAny>> {
    let ncp = ncp.as_array();
    let rhv = rhv.as_array();
    let phidp = phidp.as_array();
    let weights = weights.as_array();
    validate_det_sys_phase_inputs(ncp, rhv, phidp, weights, last_ray_idx)?;
    if !ncp_lev.is_finite() || !rhv_lev.is_finite() {
        return Err(PyValueError::new_err("thresholds must be finite"));
    }
    let phase =
        det_sys_phase_dense_kernel(ncp, rhv, phidp, weights, last_ray_idx, ncp_lev, rhv_lev);
    phase_to_py(py, phase)
}

#[pyfunction(name = "_phase_proc_det_sys_phase_gf_dense")]
pub fn phase_proc_det_sys_phase_gf_dense<'py>(
    py: Python<'py>,
    phidp: PyReadonlyArray2<'py, f64>,
    radar_meteo: PyReadonlyArray2<'py, bool>,
    weights: PyReadonlyArray1<'py, f64>,
    last_ray_idx: isize,
) -> PyResult<Py<PyAny>> {
    let phidp = phidp.as_array();
    let radar_meteo = radar_meteo.as_array();
    let weights = weights.as_array();
    validate_det_sys_phase_gf_inputs(phidp, radar_meteo, weights, last_ray_idx)?;
    let phase = det_sys_phase_gf_dense_kernel(phidp, radar_meteo, weights, last_ray_idx);
    phase_to_py(py, phase)
}

#[pyfunction(name = "_phase_proc_fzl_index_dense")]
pub fn phase_proc_fzl_index_dense<'py>(
    py: Python<'py>,
    ranges: PyReadonlyArray1<'py, f64>,
    fzl: f64,
    elevation: f64,
    radar_height: f64,
) -> PyResult<Py<PyAny>> {
    let ranges = ranges.as_array();
    validate_fzl_index_inputs(ranges, fzl, elevation, radar_height)?;
    fzl_index_to_py(py, fzl_index_kernel(ranges, fzl, elevation, radar_height))
}

#[pyfunction(name = "_attenuation_prepare_phidp_dense")]
pub fn attenuation_prepare_phidp_dense<'py>(
    py: Python<'py>,
    phidp: PyReadonlyArray2<'py, f64>,
    phidp_mask: PyReadonlyArray2<'py, bool>,
    mask_fzl: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let phidp = phidp.as_array();
    let phidp_mask = phidp_mask.as_array();
    let mask_fzl = mask_fzl.as_array();
    validate_prepare_phidp_inputs(phidp, phidp_mask, mask_fzl)?;
    let out = prepare_phidp_kernel(phidp, phidp_mask, mask_fzl);
    Ok(PyArray2::from_owned_array(py, out))
}

#[pyfunction(name = "_attenuation_end_gate_from_excluded_mask")]
pub fn attenuation_end_gate_from_excluded_mask<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let gate_excluded = gate_excluded.as_array();
    validate_end_gate_mask_inputs(gate_excluded)?;
    Ok(PyArray1::from_vec(
        py,
        end_gate_from_excluded_mask_kernel(gate_excluded)?,
    ))
}

#[pyfunction(name = "_attenuation_param_attzphi")]
pub fn attenuation_param_attzphi(freq_band: &str) -> PyResult<(f64, f64, f64, f64)> {
    match freq_band {
        "S" => Ok((0.02, 0.64884, 0.15917, 1.0804)),
        "C" => Ok((0.08, 0.64884, 0.3, 1.0804)),
        "X" => Ok((0.31916, 0.64884, 0.15917, 1.0804)),
        _ => Err(PyValueError::new_err("freq_band must be one of S, C, or X")),
    }
}

#[pyfunction(name = "_attenuation_param_attphilinear")]
pub fn attenuation_param_attphilinear(freq_band: &str) -> PyResult<(f64, f64)> {
    match freq_band {
        "S" => Ok((0.04, 0.004)),
        "C" => Ok((0.08, 0.03)),
        "X" => Ok((0.28, 0.04)),
        _ => Err(PyValueError::new_err("freq_band must be one of S, C, or X")),
    }
}

#[pyfunction]
pub fn unwrap_2d(
    image: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, u8>,
    mut unwrapped_image: PyReadwriteArray2<'_, f64>,
    wrap_around: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let wrap_x = wrap_around.get_item(1)?.is_truthy()?;
    let wrap_y = wrap_around.get_item(0)?.is_truthy()?;

    let image_shape = image.as_array().dim();
    let mask_shape = mask.as_array().dim();
    let output_shape = unwrapped_image.as_array().dim();
    if mask_shape != image_shape || output_shape != image_shape {
        return Err(PyValueError::new_err(
            "image, mask, and unwrapped_image must have the same shape",
        ));
    }

    let (height, width) = image_shape;
    if height == 0 || width == 0 {
        return Err(PyIndexError::new_err(
            "index 0 is out of bounds for an empty unwrap_2d input",
        ));
    }

    let result = unwrap_2d_kernel(
        image.as_slice()?,
        mask.as_slice()?,
        width,
        height,
        wrap_x,
        wrap_y,
    )?;
    unwrapped_image.as_slice_mut()?.copy_from_slice(&result);
    Ok(())
}

#[pyfunction]
pub fn unwrap_3d(
    image: PyReadonlyArray3<'_, f64>,
    mask: PyReadonlyArray3<'_, u8>,
    mut unwrapped_image: PyReadwriteArray3<'_, f64>,
    wrap_around: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let wrap_x = wrap_around.get_item(2)?.is_truthy()?;
    let wrap_y = wrap_around.get_item(1)?.is_truthy()?;
    let wrap_z = wrap_around.get_item(0)?.is_truthy()?;

    let image_shape = image.as_array().dim();
    let mask_shape = mask.as_array().dim();
    let output_shape = unwrapped_image.as_array().dim();
    if mask_shape != image_shape || output_shape != image_shape {
        return Err(PyValueError::new_err(
            "image, mask, and unwrapped_image must have the same shape",
        ));
    }

    let (depth, height, width) = image_shape;
    if depth == 0 || height == 0 || width == 0 {
        return Err(PyIndexError::new_err(
            "index 0 is out of bounds for an empty unwrap_3d input",
        ));
    }

    let result = unwrap_3d_kernel(
        image.as_slice()?,
        mask.as_slice()?,
        width,
        height,
        depth,
        wrap_x,
        wrap_y,
        wrap_z,
    )?;
    unwrapped_image.as_slice_mut()?.copy_from_slice(&result);
    Ok(())
}

#[pyfunction]
pub fn _fast_edge_finder<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'_, i32>,
    data: PyReadonlyArray2<'_, f32>,
    rays_wrap_around: i32,
    max_gap_x: i32,
    max_gap_y: i32,
    total_nodes: i32,
) -> PyResult<(
    (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>),
    (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>),
)> {
    let labels_view = labels.as_array();
    let data_view = data.as_array();
    if labels_view.dim() != data_view.dim() {
        return Err(PyValueError::new_err(
            "labels and data must have the same shape",
        ));
    }

    let capacity = total_nodes.max(0) as usize * 4;
    let mut l_index = Vec::with_capacity(capacity);
    let mut n_index = Vec::with_capacity(capacity);
    let mut l_velo = Vec::with_capacity(capacity);
    let mut n_velo = Vec::with_capacity(capacity);

    let (nx, ny) = labels_view.dim();
    let right = nx as isize - 1;
    let bottom = ny as isize - 1;
    let wrap = rays_wrap_around != 0;

    for x_index in 0..nx {
        for y_index in 0..ny {
            let label = labels_view[[x_index, y_index]];
            if label == 0 {
                continue;
            }

            let vel = data_view[[x_index, y_index]] as f64;

            let mut x_check = x_index as isize - 1;
            if x_check == -1 && wrap {
                x_check = right;
            }
            if x_check != -1 {
                let (neighbor, nvel) = scan_x(
                    &labels_view,
                    &data_view,
                    x_check,
                    y_index,
                    -1,
                    max_gap_x,
                    wrap,
                    right,
                );
                add_edge(
                    label,
                    neighbor,
                    vel,
                    nvel,
                    &mut l_index,
                    &mut n_index,
                    &mut l_velo,
                    &mut n_velo,
                );
            }

            x_check = x_index as isize + 1;
            if x_check == right + 1 && wrap {
                x_check = 0;
            }
            if x_check != right + 1 {
                let (neighbor, nvel) = scan_x(
                    &labels_view,
                    &data_view,
                    x_check,
                    y_index,
                    1,
                    max_gap_x,
                    wrap,
                    right,
                );
                add_edge(
                    label,
                    neighbor,
                    vel,
                    nvel,
                    &mut l_index,
                    &mut n_index,
                    &mut l_velo,
                    &mut n_velo,
                );
            }

            let mut y_check = y_index as isize - 1;
            if y_check != -1 {
                let (neighbor, nvel) = scan_y(
                    &labels_view,
                    &data_view,
                    x_index,
                    y_check,
                    -1,
                    max_gap_y,
                    bottom,
                );
                add_edge(
                    label,
                    neighbor,
                    vel,
                    nvel,
                    &mut l_index,
                    &mut n_index,
                    &mut l_velo,
                    &mut n_velo,
                );
            }

            y_check = y_index as isize + 1;
            if y_check != bottom + 1 {
                let (neighbor, nvel) = scan_y(
                    &labels_view,
                    &data_view,
                    x_index,
                    y_check,
                    1,
                    max_gap_y,
                    bottom,
                );
                add_edge(
                    label,
                    neighbor,
                    vel,
                    nvel,
                    &mut l_index,
                    &mut n_index,
                    &mut l_velo,
                    &mut n_velo,
                );
            }
        }
    }

    Ok((
        (
            PyArray1::from_vec(py, l_index),
            PyArray1::from_vec(py, n_index),
        ),
        (
            PyArray1::from_vec(py, l_velo),
            PyArray1::from_vec(py, n_velo),
        ),
    ))
}

#[pyfunction(name = "_common_dealias_limits_dense")]
pub fn common_dealias_limits_dense(
    data: PyReadonlyArrayDyn<'_, f64>,
    nyquist_vel: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<(f64, f64)> {
    let data = data.as_array();
    let nyquist_vel = nyquist_vel.as_array();
    validate_common_dealias_limits_inputs(&data, &nyquist_vel)?;

    let max_abs_vel = data
        .iter()
        .map(|value| value.abs())
        .fold(f64::NEG_INFINITY, f64::max);
    let max_nyq_vel = nyquist_vel
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_nyq_int = 2.0 * max_nyq_vel;
    let added_intervals = ((max_abs_vel - max_nyq_vel) / max_nyq_int).ceil();
    let max_valid_velocity = max_nyq_vel + added_intervals * max_nyq_int;
    Ok((-max_valid_velocity, max_valid_velocity))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(first_mask_f64, module)?)?;
    module.add_function(wrap_pyfunction!(cloud_threshold_f64, module)?)?;
    module.add_function(wrap_pyfunction!(cloud_mask_4x4_count_i16, module)?)?;
    module.add_function(wrap_pyfunction!(correct_noise_rhohv_dense_f64, module)?)?;
    module.add_function(wrap_pyfunction!(region_cost_function, module)?)?;
    module.add_function(wrap_pyfunction!(region_gradient, module)?)?;
    module.add_function(wrap_pyfunction!(region_sweep_interval_splits, module)?)?;
    module.add_function(wrap_pyfunction!(region_edge_sum_and_count, module)?)?;
    module.add_function(wrap_pyfunction!(common_dealias_limits_dense, module)?)?;
    module.add_function(wrap_pyfunction!(phase_proc_smooth_and_trim_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        phase_proc_smooth_and_trim_scan_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        phase_proc_unwrap_masked_degrees_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(phase_proc_det_sys_phase_dense, module)?)?;
    module.add_function(wrap_pyfunction!(phase_proc_det_sys_phase_gf_dense, module)?)?;
    module.add_function(wrap_pyfunction!(phase_proc_fzl_index_dense, module)?)?;
    module.add_function(wrap_pyfunction!(attenuation_prepare_phidp_dense, module)?)?;
    module.add_function(wrap_pyfunction!(
        attenuation_end_gate_from_excluded_mask,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(attenuation_param_attzphi, module)?)?;
    module.add_function(wrap_pyfunction!(attenuation_param_attphilinear, module)?)?;
    module.add_function(wrap_pyfunction!(unwrap_1d, module)?)?;
    module.add_function(wrap_pyfunction!(unwrap_2d, module)?)?;
    module.add_function(wrap_pyfunction!(unwrap_3d, module)?)?;
    module.add_function(wrap_pyfunction!(_fast_edge_finder, module)?)?;
    Ok(())
}

fn validate_unwrap_masked_inputs(
    values: ndarray::ArrayView1<'_, f64>,
    mask: ndarray::ArrayView1<'_, bool>,
) -> PyResult<()> {
    if values.len() != mask.len() {
        return Err(PyValueError::new_err(
            "values and mask must have the same length",
        ));
    }
    if values.len() < 2 {
        return Err(PyValueError::new_err(
            "unwrap_masked input must have at least two elements",
        ));
    }
    if !values.is_standard_layout() || !mask.is_standard_layout() {
        return Err(PyValueError::new_err(
            "values and mask must be C-contiguous",
        ));
    }
    let mut valid_count = 0_usize;
    for (&value, &is_masked) in values.iter().zip(mask.iter()) {
        if is_masked {
            continue;
        }
        if !value.is_finite() {
            return Err(PyValueError::new_err("unmasked values must be finite"));
        }
        valid_count += 1;
    }
    if valid_count < 2 {
        return Err(PyValueError::new_err(
            "unwrap_masked input must have at least two valid values",
        ));
    }
    Ok(())
}

fn unwrap_masked_degrees_kernel(
    values: ndarray::ArrayView1<'_, f64>,
    mask: ndarray::ArrayView1<'_, bool>,
) -> Vec<f64> {
    let valid: Vec<f64> = values
        .iter()
        .zip(mask.iter())
        .filter_map(|(&value, &is_masked)| if is_masked { None } else { Some(value) })
        .collect();
    let mut output = Vec::with_capacity(valid.len());
    output.push(valid[0]);
    let mut periods = 0_i64;
    for i in 1..valid.len() {
        let diff = valid[i] - valid[i - 1];
        if diff > 180.0 {
            periods -= 1;
        } else if diff < -180.0 {
            periods += 1;
        }
        output.push(valid[i] + 360.0 * periods as f64);
    }
    output
}

fn validate_correct_noise_rhohv_inputs(
    urhohv: &ndarray::ArrayViewD<'_, f64>,
    snrd_b_h: &ndarray::ArrayViewD<'_, f64>,
    zdr_db: &ndarray::ArrayViewD<'_, f64>,
    nh: &ndarray::ArrayViewD<'_, f64>,
    nv: &ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    for array in [snrd_b_h, zdr_db, nh, nv] {
        if array.shape() != urhohv.shape() {
            return Err(PyValueError::new_err(
                "all rhohv noise inputs must have the same shape",
            ));
        }
    }
    for array in [urhohv, snrd_b_h, zdr_db, nh, nv] {
        if !array.is_standard_layout() {
            return Err(PyValueError::new_err(
                "all rhohv noise inputs must be C-contiguous",
            ));
        }
    }
    for &value in urhohv.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("urhohv must be finite"));
        }
        if value.abs() > RHOHV_NOISE_MAX_ABS_URHOHV {
            return Err(PyValueError::new_err(
                "urhohv values are outside the dense rhohv-noise kernel range",
            ));
        }
    }
    for (name, array) in [
        ("snrdB_h", snrd_b_h),
        ("zdrdB", zdr_db),
        ("nh", nh),
        ("nv", nv),
    ] {
        for &value in array.iter() {
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!("{name} must be finite")));
            }
            if value.abs() > RHOHV_NOISE_MAX_ABS_DB {
                return Err(PyValueError::new_err(format!(
                    "{name} values are outside the dense rhohv-noise kernel range"
                )));
            }
        }
    }
    for (&nh_value, &nv_value) in nh.iter().zip(nv.iter()) {
        if (nh_value - nv_value).abs() > RHOHV_NOISE_MAX_ABS_DB {
            return Err(PyValueError::new_err(
                "nh and nv difference is outside the dense rhohv-noise kernel range",
            ));
        }
    }
    Ok(())
}

fn validate_cloud_mask_4x4_inputs(
    mask1: &ndarray::ArrayView2<'_, i16>,
    counts_threshold: i64,
) -> PyResult<()> {
    if !mask1.is_standard_layout() {
        return Err(PyValueError::new_err("mask1 must be C-contiguous"));
    }
    if !(0..=16).contains(&counts_threshold) {
        return Err(PyValueError::new_err(
            "counts_threshold must be within the dense cloud-mask range",
        ));
    }
    for &value in mask1.iter() {
        if value != 0 && value != 1 {
            return Err(PyValueError::new_err("mask1 values must be 0 or 1"));
        }
    }
    Ok(())
}

fn cloud_mask_4x4_count_kernel(
    mask1: &ndarray::ArrayView2<'_, i16>,
    counts_threshold: i64,
) -> Array2<i16> {
    let (rows, cols) = mask1.dim();
    let mut output = Array2::<i16>::zeros((rows, cols));
    for ((row, col), slot) in output.indexed_iter_mut() {
        let row_start = row.saturating_sub(CLOUD_MASK_BOX_WIDTH / 2);
        let row_end = row.saturating_add(CLOUD_MASK_BOX_WIDTH / 2).min(rows);
        let col_start = col.saturating_sub(CLOUD_MASK_BOX_WIDTH / 2);
        let col_end = col.saturating_add(CLOUD_MASK_BOX_WIDTH / 2).min(cols);
        let count = mask1
            .slice(s![row_start..row_end, col_start..col_end])
            .iter()
            .map(|&value| i64::from(value))
            .sum::<i64>();
        if count >= counts_threshold {
            *slot = 1;
        }
    }
    output
}

fn validate_cloud_threshold_inputs(
    data: &ndarray::ArrayViewD<'_, f64>,
    n_avg: f64,
    nffts: usize,
) -> PyResult<()> {
    if data.ndim() != 1 {
        return Err(PyValueError::new_err("data must be one-dimensional"));
    }
    if !data.is_standard_layout() {
        return Err(PyValueError::new_err("data must be C-contiguous"));
    }
    if !n_avg.is_finite() || !(0.0..=CLOUD_THRESHOLD_MAX_N_AVG).contains(&n_avg) {
        return Err(PyValueError::new_err(
            "n_avg must be finite and within the dense cloud-threshold range",
        ));
    }
    if nffts > data.len() {
        return Err(PyValueError::new_err("nffts must not exceed data length"));
    }
    for &value in data.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("data must be finite"));
        }
        if value.abs() > CLOUD_THRESHOLD_MAX_ABS_DB {
            return Err(PyValueError::new_err(
                "data values are outside the dense cloud-threshold range",
            ));
        }
    }
    Ok(())
}

fn cloud_threshold_kernel(data: &ndarray::ArrayViewD<'_, f64>, n_avg: f64, nffts: usize) -> f64 {
    let mut data_linear = data
        .iter()
        .map(|value| 10.0_f64.powf(*value / 10.0))
        .collect::<Vec<_>>();
    data_linear.sort_by(|left, right| left.partial_cmp(right).unwrap());

    let nthld = 10.0_f64.powf(-10.0);
    let mut dsum = 0.0;
    let mut sum_sq = 0.0;
    let mut n = 0.0;
    let mut sum_ns = 0.0;
    let mut num_ns = None;
    let sqrt_n_avg = n_avg.sqrt();

    for value in data_linear.iter().take(nffts).copied() {
        if value > nthld {
            dsum += value;
            sum_sq += value.powf(2.0);
            n += 1.0;
            let a3 = dsum * dsum;
            let a1 = sqrt_n_avg * (n * sum_sq - a3);
            if n > nffts as f64 / 4.0 {
                if a1 <= a3 {
                    sum_ns = dsum;
                    num_ns = Some(n);
                }
            } else {
                sum_ns = dsum;
                num_ns = Some(n);
            }
        }
    }

    let n_mean = num_ns.map_or(f64::NAN, |num| sum_ns / num);
    if n_mean == 0.0 {
        f64::NAN
    } else {
        10.0 * n_mean.log10()
    }
}

fn validate_prepare_phidp_inputs(
    phidp: ndarray::ArrayView2<'_, f64>,
    phidp_mask: ndarray::ArrayView2<'_, bool>,
    mask_fzl: ndarray::ArrayView2<'_, bool>,
) -> PyResult<()> {
    if phidp.dim() != phidp_mask.dim() || phidp.dim() != mask_fzl.dim() {
        return Err(PyValueError::new_err(
            "phidp, phidp_mask, and mask_fzl must have the same shape",
        ));
    }
    if !phidp.is_standard_layout()
        || !phidp_mask.is_standard_layout()
        || !mask_fzl.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "phidp, phidp_mask, and mask_fzl must be C-contiguous",
        ));
    }
    Ok(())
}

fn validate_end_gate_mask_inputs(gate_excluded: ndarray::ArrayView2<'_, bool>) -> PyResult<()> {
    if !gate_excluded.is_standard_layout() {
        return Err(PyValueError::new_err("gate_excluded must be C-contiguous"));
    }
    if gate_excluded.dim().1 > i32::MAX as usize {
        return Err(PyValueError::new_err("ngates must fit in int32"));
    }
    Ok(())
}

fn validate_common_dealias_limits_inputs(
    data: &ndarray::ArrayViewD<'_, f64>,
    nyquist_vel: &ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    if data.is_empty() {
        return Err(PyValueError::new_err("data must be non-empty"));
    }
    if nyquist_vel.is_empty() {
        return Err(PyValueError::new_err("nyquist_vel must be non-empty"));
    }
    if !data.is_standard_layout() || !nyquist_vel.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data and nyquist_vel must be C-contiguous",
        ));
    }
    if data.iter().any(|value| !value.is_finite())
        || nyquist_vel.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err("data and nyquist_vel must be finite"));
    }
    Ok(())
}

fn prepare_phidp_kernel(
    phidp: ndarray::ArrayView2<'_, f64>,
    phidp_mask: ndarray::ArrayView2<'_, bool>,
    mask_fzl: ndarray::ArrayView2<'_, bool>,
) -> Array2<f64> {
    let (nrays, ngates) = phidp.dim();
    let mut out = Array2::<f64>::zeros((nrays, ngates));

    for ray in 0..nrays {
        if ngates == 0 {
            continue;
        }
        let mut current =
            prepare_phidp_value(phidp[[ray, 0]], phidp_mask[[ray, 0]], mask_fzl[[ray, 0]]);
        out[[ray, 0]] = current;
        for gate in 1..ngates {
            let value = prepare_phidp_value(
                phidp[[ray, gate]],
                phidp_mask[[ray, gate]],
                mask_fzl[[ray, gate]],
            );
            current = numpy_maximum(current, value);
            out[[ray, gate]] = current;
        }
    }

    out
}

fn end_gate_from_excluded_mask_kernel(
    gate_excluded: ndarray::ArrayView2<'_, bool>,
) -> PyResult<Vec<i32>> {
    let (nrays, ngates) = gate_excluded.dim();
    if ngates > i32::MAX as usize {
        return Err(PyValueError::new_err("ngates must fit in int32"));
    }

    let no_excluded_gate = ngates as i32 - 1;
    let mut out = vec![no_excluded_gate; nrays];
    for ray in 0..nrays {
        for gate in 0..ngates {
            if gate_excluded[[ray, gate]] {
                out[ray] = if gate == 0 { 0 } else { (gate - 1) as i32 };
                break;
            }
        }
    }
    Ok(out)
}

fn prepare_phidp_value(value: f64, phidp_mask: bool, mask_fzl: bool) -> f64 {
    if phidp_mask || mask_fzl || value < 0.0 {
        0.0
    } else {
        value
    }
}

fn numpy_maximum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left > right {
        left
    } else {
        right
    }
}

fn validate_fzl_index_inputs(
    ranges: ndarray::ArrayView1<'_, f64>,
    fzl: f64,
    elevation: f64,
    radar_height: f64,
) -> PyResult<()> {
    if !fzl.is_finite() || !elevation.is_finite() || !radar_height.is_finite() {
        return Err(PyValueError::new_err(
            "fzl, elevation, and radar_height must be finite",
        ));
    }
    if ranges.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("ranges must be finite"));
    }
    Ok(())
}

fn validate_det_sys_phase_inputs(
    ncp: ndarray::ArrayView2<'_, f64>,
    rhv: ndarray::ArrayView2<'_, f64>,
    phidp: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    last_ray_idx: isize,
) -> PyResult<()> {
    if ncp.dim() != rhv.dim() || ncp.dim() != phidp.dim() {
        return Err(PyValueError::new_err(
            "ncp, rhv, and phidp must have the same shape",
        ));
    }
    if ncp.iter().any(|value| !value.is_finite()) || rhv.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("ncp and rhv must be finite"));
    }
    validate_det_sys_phase_common(phidp, weights, last_ray_idx)
}

fn validate_det_sys_phase_gf_inputs(
    phidp: ndarray::ArrayView2<'_, f64>,
    radar_meteo: ndarray::ArrayView2<'_, bool>,
    weights: ndarray::ArrayView1<'_, f64>,
    last_ray_idx: isize,
) -> PyResult<()> {
    if phidp.dim() != radar_meteo.dim() {
        return Err(PyValueError::new_err(
            "phidp and radar_meteo must have the same shape",
        ));
    }
    validate_det_sys_phase_common(phidp, weights, last_ray_idx)
}

fn validate_det_sys_phase_common(
    phidp: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    last_ray_idx: isize,
) -> PyResult<()> {
    let (nrays, _) = phidp.dim();
    if last_ray_idx >= nrays as isize {
        return Err(PyValueError::new_err(
            "last_ray_idx must be within phidp rows",
        ));
    }
    if weights.len() != 9 {
        return Err(PyValueError::new_err("weights length must be 9"));
    }
    if phidp.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("phidp must be finite"));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("weights must be finite"));
    }
    Ok(())
}

fn validate_smooth_and_trim_inputs(
    x: ndarray::ArrayView1<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    window_len: isize,
) -> PyResult<()> {
    if window_len < 3 {
        return Err(PyValueError::new_err("window_len must be at least 3"));
    }
    let window_len = window_len as usize;
    if weights.len() != window_len {
        return Err(PyValueError::new_err(
            "weights length must match window_len",
        ));
    }
    if x.len() < window_len {
        return Err(PyValueError::new_err(
            "x length must be at least window_len",
        ));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("x must be finite"));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("weights must be finite"));
    }
    Ok(())
}

fn validate_smooth_and_trim_scan_inputs(
    x: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    window_len: isize,
) -> PyResult<()> {
    if window_len < 3 {
        return Err(PyValueError::new_err("window_len must be at least 3"));
    }
    let window_len = window_len as usize;
    if weights.len() != window_len {
        return Err(PyValueError::new_err(
            "weights length must match window_len",
        ));
    }
    let (_, width) = x.dim();
    if width < window_len {
        return Err(PyValueError::new_err("x width must be at least window_len"));
    }
    if !x.is_standard_layout() || !weights.is_standard_layout() {
        return Err(PyValueError::new_err("x and weights must be C-contiguous"));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("x must be finite"));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("weights must be finite"));
    }
    let weight_sum: f64 = weights.iter().sum();
    if !weight_sum.is_finite() || weight_sum == 0.0 {
        return Err(PyValueError::new_err(
            "weights sum must be finite and nonzero",
        ));
    }
    Ok(())
}

fn phase_to_py(py: Python<'_>, phase: Option<f64>) -> PyResult<Py<PyAny>> {
    match phase {
        Some(value) => {
            let np = py.import("numpy")?;
            Ok(np.getattr("float64")?.call1((value,))?.unbind())
        }
        None => Ok(py.None()),
    }
}

fn fzl_index_to_py(py: Python<'_>, result: FzlIndexResult) -> PyResult<Py<PyAny>> {
    match result {
        FzlIndexResult::MinimumWindow => Ok(6_i32.into_pyobject(py)?.unbind().into()),
        FzlIndexResult::GateIndex(index) => {
            let np = py.import("numpy")?;
            Ok(np.getattr("int64")?.call1((index as i64,))?.unbind())
        }
        FzlIndexResult::NoGateBelow => Err(PyValueError::new_err(
            "zero-size array to reduction operation maximum which has no identity",
        )),
    }
}

fn validate_region_inputs(
    nyq_len: usize,
    vels_len: usize,
    svels_len: usize,
    nfeatures: isize,
) -> PyResult<()> {
    if nfeatures <= 0 {
        return Ok(());
    }
    let nfeatures = nfeatures as usize;
    if nfeatures > nyq_len || nfeatures > vels_len || nfeatures > svels_len {
        return Err(PyValueError::new_err(
            "nfeatures must not exceed input array lengths",
        ));
    }
    Ok(())
}

fn validate_sweep_interval_inputs(
    nyquist: f64,
    interval_splits: isize,
    velocities: &ndarray::ArrayView1<'_, f64>,
) -> PyResult<()> {
    if !nyquist.is_finite() || nyquist <= 0.0 {
        return Err(PyValueError::new_err("nyquist must be finite and positive"));
    }
    if interval_splits <= 0 {
        return Err(PyValueError::new_err("interval_splits must be positive"));
    }
    if !velocities.is_standard_layout() {
        return Err(PyValueError::new_err("velocities must be C-contiguous"));
    }
    if velocities.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("velocities must be finite"));
    }
    Ok(())
}

fn sweep_interval_splits_kernel(
    nyquist: f64,
    interval_splits: usize,
    velocities: &ndarray::ArrayView1<'_, f64>,
) -> PyResult<(f64, f64, usize, bool)> {
    let mut add_start = 0_usize;
    let mut add_end = 0_usize;
    let interval = (2.0 * nyquist) / interval_splits as f64;
    if !interval.is_finite() || interval <= 0.0 {
        return Err(PyValueError::new_err(
            "interval must be finite and positive",
        ));
    }
    let mut outside_nyquist = false;

    if !velocities.is_empty() {
        let mut min_vel = velocities[0];
        let mut max_vel = velocities[0];
        for value in velocities.iter().skip(1) {
            if *value < min_vel {
                min_vel = *value;
            }
            if *value > max_vel {
                max_vel = *value;
            }
        }

        if max_vel > nyquist || min_vel < -nyquist {
            outside_nyquist = true;
            add_start = ceil_nonnegative_to_usize((max_vel - nyquist) / interval)?;
            add_end = ceil_nonnegative_to_usize(-(min_vel + nyquist) / interval)?;
        }
    }

    let start = -nyquist - add_start as f64 * interval;
    let end = nyquist + add_end as f64 * interval;
    if !start.is_finite() || !end.is_finite() {
        return Err(PyValueError::new_err("interval bounds must be finite"));
    }
    let num = interval_splits
        .checked_add(1)
        .and_then(|value| value.checked_add(add_start))
        .and_then(|value| value.checked_add(add_end))
        .ok_or_else(|| PyValueError::new_err("interval split count exceeds usize range"))?;
    Ok((start, end, num, outside_nyquist))
}

fn ceil_nonnegative_to_usize(value: f64) -> PyResult<usize> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("interval extension must be finite"));
    }
    let ceiled = value.ceil().max(0.0);
    if ceiled > usize::MAX as f64 {
        return Err(PyValueError::new_err(
            "interval extension exceeds usize range",
        ));
    }
    Ok(ceiled as usize)
}

fn validate_edge_sum_inputs(
    index1: ndarray::ArrayView1<'_, i32>,
    index2: ndarray::ArrayView1<'_, i32>,
    vel1: ndarray::ArrayView1<'_, f64>,
    vel2: ndarray::ArrayView1<'_, f64>,
) -> PyResult<()> {
    let len = index1.len();
    if index2.len() != len || vel1.len() != len || vel2.len() != len {
        return Err(PyValueError::new_err(
            "edge index and velocity arrays must have the same length",
        ));
    }
    if !index1.is_standard_layout()
        || !index2.is_standard_layout()
        || !vel1.is_standard_layout()
        || !vel2.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "edge index and velocity arrays must be C-contiguous",
        ));
    }
    if vel1.iter().any(|value| !value.is_finite()) || vel2.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("edge velocities must be finite"));
    }
    Ok(())
}

fn edge_sum_and_count_kernel(
    index1: ndarray::ArrayView1<'_, i32>,
    index2: ndarray::ArrayView1<'_, i32>,
    vel1: ndarray::ArrayView1<'_, f64>,
    vel2: ndarray::ArrayView1<'_, f64>,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..index1.len()).collect();
    order.sort_by(|left, right| {
        index2[*left]
            .cmp(&index2[*right])
            .then_with(|| index1[*left].cmp(&index1[*right]))
    });

    let mut out_index1 = Vec::new();
    let mut out_index2 = Vec::new();
    let mut count = Vec::new();
    let mut out_vel1 = Vec::new();
    let mut out_vel2 = Vec::new();

    let mut current_pair: Option<(i32, i32)> = None;
    for idx in order {
        let pair = (index1[idx], index2[idx]);
        if current_pair == Some(pair) {
            let last = count.len() - 1;
            count[last] += 1;
            out_vel1[last] += vel1[idx];
            out_vel2[last] += vel2[idx];
        } else {
            current_pair = Some(pair);
            out_index1.push(pair.0);
            out_index2.push(pair.1);
            count.push(1);
            out_vel1.push(vel1[idx]);
            out_vel2.push(vel2[idx]);
        }
    }

    (out_index1, out_index2, count, out_vel1, out_vel2)
}

enum FzlIndexResult {
    MinimumWindow,
    GateIndex(usize),
    NoGateBelow,
}

fn fzl_index_kernel(
    ranges: ndarray::ArrayView1<'_, f64>,
    fzl: f64,
    elevation: f64,
    radar_height: f64,
) -> FzlIndexResult {
    let earth_radius = 6371.0 * 1000.0;
    let effective_radius = 4.0 * earth_radius / 3.0;
    let sin_elevation = (elevation * std::f64::consts::PI / 180.0).sin();
    let mut all_above = true;
    let mut last_below = None;

    for (idx, &range) in ranges.iter().enumerate() {
        let z = radar_height
            + (range * range
                + effective_radius * effective_radius
                + 2.0 * range * effective_radius * sin_elevation)
                .sqrt()
            - effective_radius;
        if z <= fzl {
            all_above = false;
        }
        if z < fzl {
            last_below = Some(idx);
        }
    }

    if all_above {
        FzlIndexResult::MinimumWindow
    } else if let Some(idx) = last_below {
        FzlIndexResult::GateIndex(idx)
    } else {
        FzlIndexResult::NoGateBelow
    }
}

fn det_sys_phase_dense_kernel(
    ncp: ndarray::ArrayView2<'_, f64>,
    rhv: ndarray::ArrayView2<'_, f64>,
    phidp: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    last_ray_idx: isize,
    ncp_lev: f64,
    rhv_lev: f64,
) -> Option<f64> {
    if last_ray_idx < 0 {
        return None;
    }
    let (_, ngates) = phidp.dim();
    let mut phases = Vec::new();
    for radial in 0..=last_ray_idx as usize {
        let mut selected = Vec::with_capacity(ngates);
        for gate in 0..ngates {
            if ncp[[radial, gate]] > ncp_lev && rhv[[radial, gate]] > rhv_lev {
                selected.push(phidp[[radial, gate]]);
            }
        }
        push_phase_if_enough_points(selected, weights, &mut phases);
    }
    median_or_none(phases)
}

fn det_sys_phase_gf_dense_kernel(
    phidp: ndarray::ArrayView2<'_, f64>,
    radar_meteo: ndarray::ArrayView2<'_, bool>,
    weights: ndarray::ArrayView1<'_, f64>,
    last_ray_idx: isize,
) -> Option<f64> {
    if last_ray_idx < 0 {
        return None;
    }
    let (_, ngates) = phidp.dim();
    let mut phases = Vec::new();
    for radial in 0..=last_ray_idx as usize {
        let mut selected = Vec::with_capacity(ngates);
        for gate in 0..ngates {
            if radar_meteo[[radial, gate]] {
                selected.push(phidp[[radial, gate]]);
            }
        }
        push_phase_if_enough_points(selected, weights, &mut phases);
    }
    median_or_none(phases)
}

fn push_phase_if_enough_points(
    selected: Vec<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    phases: &mut Vec<f64>,
) {
    if selected.len() <= 25 {
        return;
    }
    let selected = Array1::from_vec(selected);
    let smoothed = smooth_and_trim_kernel(selected.view(), weights, 9);
    let mut min_phase = smoothed[0];
    for idx in 1..25 {
        if smoothed[idx] < min_phase {
            min_phase = smoothed[idx];
        }
    }
    phases.push(min_phase);
}

fn median_or_none(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite phase values"));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

fn mirror_index(idx: isize, size: usize) -> usize {
    if size <= 1 {
        return 0;
    }
    let size = size as isize;
    let mut idx = idx;
    while idx < 0 || idx >= size {
        if idx < 0 {
            idx = -idx - 1;
        }
        if idx >= size {
            idx = 2 * size - idx - 1;
        }
    }
    idx as usize
}

fn convolve1d_reflect_row(row: &[f64], weights: &[f64]) -> Vec<f64> {
    let wlen = weights.len();
    let mut reversed = Vec::with_capacity(wlen);
    for value in weights.iter().rev() {
        reversed.push(*value);
    }
    let origin = if wlen & 1 == 1 { 0 } else { -1 };
    let offset = (wlen / 2) as isize + origin;
    let n = row.len();
    let mut out = vec![0.0; n];
    for (out_idx, slot) in out.iter_mut().enumerate().take(n) {
        let mut total = 0.0;
        for (weight_idx, weight) in reversed.iter().enumerate().take(wlen) {
            let sample_idx = mirror_index(out_idx as isize - offset + weight_idx as isize, n);
            total += weight * row[sample_idx];
        }
        *slot = total;
    }
    out
}

fn smooth_and_trim_scan_kernel(
    x: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    _window_len: usize,
) -> Array2<f64> {
    let (nrows, ncols) = x.dim();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    let weight_vec: Vec<f64> = weights.iter().copied().collect();
    for row_idx in 0..nrows {
        let row: Vec<f64> = (0..ncols).map(|col| x[[row_idx, col]]).collect();
        let smoothed = convolve1d_reflect_row(&row, &weight_vec);
        out.row_mut(row_idx).assign(&Array1::from_vec(smoothed));
    }
    out
}

fn smooth_and_trim_kernel(
    x: ndarray::ArrayView1<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    window_len: usize,
) -> Array1<f64> {
    let half = window_len / 2;
    let mut reflected = Vec::with_capacity(x.len() + 2 * (window_len - 1));
    for idx in (1..window_len).rev() {
        reflected.push(x[idx]);
    }
    reflected.extend(x.iter().copied());
    for idx in (x.len() - window_len + 1..x.len()).rev() {
        reflected.push(x[idx]);
    }

    let mut out = Array1::<f64>::zeros(x.len());
    for out_idx in 0..x.len() {
        let valid_idx = out_idx + half;
        let mut total = 0.0;
        for weight_idx in 0..window_len {
            total += weights[weight_idx] * reflected[valid_idx + weight_idx];
        }
        out[out_idx] = total;
    }
    out
}

fn region_cost_function_kernel(
    nyq_vector: ndarray::ArrayView1<'_, f64>,
    vels_slice_means: ndarray::ArrayView1<'_, f64>,
    svels_slice_means: ndarray::ArrayView1<'_, f64>,
    v_nyq_vel: f64,
    nfeatures: usize,
) -> f64 {
    let mut cost = 0.0;
    for reg in 0..nfeatures {
        let add_value = (vels_slice_means[reg] + round_half_to_even(nyq_vector[reg]) * v_nyq_vel
            - svels_slice_means[reg])
            .powi(2);
        if add_value.is_finite() {
            cost += add_value;
        }
    }
    cost
}

fn region_gradient_kernel(
    nyq_vector: ndarray::ArrayView1<'_, f64>,
    vels_slice_means: ndarray::ArrayView1<'_, f64>,
    svels_slice_means: ndarray::ArrayView1<'_, f64>,
    v_nyq_vel: f64,
    nfeatures: usize,
) -> Array1<f64> {
    let mut gradient = Array1::<f64>::zeros(nyq_vector.len());
    for reg in 0..nfeatures {
        let add_value = vels_slice_means[reg] + round_half_to_even(nyq_vector[reg]) * v_nyq_vel
            - svels_slice_means[reg];
        if add_value.is_finite() {
            gradient[reg] = 2.0 * add_value * v_nyq_vel;
        }

        let min_index = nearest_other_region_diff_index(vels_slice_means, reg, nfeatures);
        if (min_index as f64) < v_nyq_vel {
            gradient[reg] = 0.0;
        }
    }
    gradient
}

fn nearest_other_region_diff_index(
    vels_slice_means: ndarray::ArrayView1<'_, f64>,
    reg: usize,
    nfeatures: usize,
) -> usize {
    if nfeatures <= 1 {
        return 0;
    }

    let base = vels_slice_means[reg];
    let mut best_position = 0usize;
    let mut best_diff = f64::INFINITY;
    let mut position = 0usize;
    for idx in 0..nfeatures {
        if idx == reg {
            continue;
        }
        let diff = (base - vels_slice_means[idx]).powi(2);
        if diff < best_diff {
            best_diff = diff;
            best_position = position;
        }
        position += 1;
    }
    best_position
}

fn round_half_to_even(value: f64) -> f64 {
    value.round_ties_even()
}

#[derive(Clone, Debug)]
struct UnwrapNode {
    increment: i32,
    number_in_group: usize,
    value: f64,
    reliability: f64,
    input_mask: u8,
    extended_mask: u8,
    head: usize,
    last: usize,
    next: Option<usize>,
}

#[derive(Clone, Debug)]
struct UnwrapEdge {
    reliability: f64,
    pointer_1: usize,
    pointer_2: usize,
    increment: i32,
    order: usize,
}

fn unwrap_2d_kernel(
    image: &[f64],
    mask: &[u8],
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
) -> PyResult<Vec<f64>> {
    let image_size = checked_mul(width, height, "image is too large")?;
    if image.len() != image_size || mask.len() != image_size {
        return Err(PyValueError::new_err(
            "image and mask buffers must match image dimensions",
        ));
    }

    let extended_mask = extend_mask_2d(mask, width, height, wrap_x, wrap_y);
    let mut nodes = initialise_nodes(image, mask, &extended_mask);
    calculate_reliability_2d(image, &mut nodes, width, height, wrap_x, wrap_y);
    let mut edges = build_edges_2d(&nodes, width, height, wrap_x, wrap_y);
    sort_edges(&mut edges);
    gather_nodes(&mut nodes, &edges);
    unwrap_nodes(&mut nodes);
    mask_nodes(&mut nodes);
    Ok(nodes.into_iter().map(|node| node.value).collect())
}

fn unwrap_3d_kernel(
    volume: &[f64],
    mask: &[u8],
    width: usize,
    height: usize,
    depth: usize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) -> PyResult<Vec<f64>> {
    let frame_size = checked_mul(width, height, "volume frame is too large")?;
    let volume_size = checked_mul(frame_size, depth, "volume is too large")?;
    if volume.len() != volume_size || mask.len() != volume_size {
        return Err(PyValueError::new_err(
            "image and mask buffers must match volume dimensions",
        ));
    }

    let extended_mask = extend_mask_3d(mask, width, height, depth, wrap_x, wrap_y, wrap_z);
    let mut nodes = initialise_nodes(volume, mask, &extended_mask);
    calculate_reliability_3d(
        volume, &mut nodes, width, height, depth, wrap_x, wrap_y, wrap_z,
    );
    let mut edges = build_edges_3d(&nodes, width, height, depth, wrap_x, wrap_y, wrap_z)?;
    sort_edges(&mut edges);
    gather_nodes(&mut nodes, &edges);
    unwrap_nodes(&mut nodes);
    mask_nodes(&mut nodes);
    Ok(nodes.into_iter().map(|node| node.value).collect())
}

fn checked_mul(a: usize, b: usize, message: &'static str) -> PyResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| PyValueError::new_err(message))
}

fn initialise_nodes(values: &[f64], mask: &[u8], extended_mask: &[u8]) -> Vec<UnwrapNode> {
    let mut rand_state = 1_u32;
    values
        .iter()
        .zip(mask.iter())
        .zip(extended_mask.iter())
        .enumerate()
        .map(|(index, ((&value, &input_mask), &extended_mask))| {
            rand_state = rand_state.wrapping_mul(214013).wrapping_add(2531011);
            let rand_value = ((rand_state >> 16) & 0x7fff) as f64;
            UnwrapNode {
                increment: 0,
                number_in_group: 1,
                value,
                reliability: HIGH_RELIABILITY + rand_value,
                input_mask,
                extended_mask,
                head: index,
                last: index,
                next: None,
            }
        })
        .collect()
}

fn extend_mask_2d(mask: &[u8], width: usize, height: usize, wrap_x: bool, wrap_y: bool) -> Vec<u8> {
    let mut extended = vec![MASK; mask.len()];
    if width < 3 || height < 3 {
        return extended;
    }

    for row in 0..height {
        for col in 0..width {
            let on_x_border = col == 0 || col == width - 1;
            let on_y_border = row == 0 || row == height - 1;
            let border_count = usize::from(on_x_border) + usize::from(on_y_border);
            let candidate = border_count == 0
                || (border_count == 1 && ((on_x_border && wrap_x) || (on_y_border && wrap_y)));
            if candidate && all_neighbors_unmasked_2d(mask, width, height, row, col, wrap_x, wrap_y)
            {
                extended[idx2(row, col, width)] = NOMASK;
            }
        }
    }

    extended
}

fn all_neighbors_unmasked_2d(
    mask: &[u8],
    width: usize,
    height: usize,
    row: usize,
    col: usize,
    wrap_x: bool,
    wrap_y: bool,
) -> bool {
    for dy in -1_isize..=1 {
        for dx in -1_isize..=1 {
            let Some(neighbor_row) = checked_wrap_index(row as isize + dy, height, wrap_y) else {
                return false;
            };
            let Some(neighbor_col) = checked_wrap_index(col as isize + dx, width, wrap_x) else {
                return false;
            };
            if mask[idx2(neighbor_row, neighbor_col, width)] != NOMASK {
                return false;
            }
        }
    }
    true
}

fn extend_mask_3d(
    mask: &[u8],
    width: usize,
    height: usize,
    depth: usize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) -> Vec<u8> {
    let mut extended = vec![MASK; mask.len()];
    if width < 3 || height < 3 || depth < 3 {
        return extended;
    }

    for z in 0..depth {
        for row in 0..height {
            for col in 0..width {
                let on_x_border = col == 0 || col == width - 1;
                let on_y_border = row == 0 || row == height - 1;
                let on_z_border = z == 0 || z == depth - 1;
                let border_count =
                    usize::from(on_x_border) + usize::from(on_y_border) + usize::from(on_z_border);
                let candidate = border_count == 0
                    || (border_count == 1
                        && ((on_x_border && wrap_x)
                            || (on_y_border && wrap_y)
                            || (on_z_border && wrap_z)));
                if candidate
                    && all_neighbors_unmasked_3d(
                        mask, width, height, depth, z, row, col, wrap_x, wrap_y, wrap_z,
                    )
                {
                    extended[idx3(z, row, col, width, height)] = NOMASK;
                }
            }
        }
    }

    extended
}

fn all_neighbors_unmasked_3d(
    mask: &[u8],
    width: usize,
    height: usize,
    depth: usize,
    z: usize,
    row: usize,
    col: usize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) -> bool {
    for dz in -1_isize..=1 {
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                let Some(neighbor_z) = checked_wrap_index(z as isize + dz, depth, wrap_z) else {
                    return false;
                };
                let Some(neighbor_row) = checked_wrap_index(row as isize + dy, height, wrap_y)
                else {
                    return false;
                };
                let Some(neighbor_col) = checked_wrap_index(col as isize + dx, width, wrap_x)
                else {
                    return false;
                };
                if mask[idx3(neighbor_z, neighbor_row, neighbor_col, width, height)] != NOMASK {
                    return false;
                }
            }
        }
    }
    true
}

fn calculate_reliability_2d(
    image: &[f64],
    nodes: &mut [UnwrapNode],
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
) {
    if width >= 3 && height >= 3 {
        for row in 1..height - 1 {
            for col in 1..width - 1 {
                let index = idx2(row, col, width);
                if nodes[index].extended_mask == NOMASK {
                    nodes[index].reliability =
                        reliability_2d_at(image, width, height, row, col, wrap_x, wrap_y);
                }
            }
        }
    }

    if wrap_x && width >= 2 && height >= 3 {
        for row in 1..height - 1 {
            let left = idx2(row, 0, width);
            if nodes[left].extended_mask == NOMASK {
                nodes[left].reliability =
                    reliability_2d_at(image, width, height, row, 0, wrap_x, wrap_y);
            }
            let right = idx2(row, width - 1, width);
            if nodes[right].extended_mask == NOMASK {
                nodes[right].reliability =
                    reliability_2d_at(image, width, height, row, width - 1, wrap_x, wrap_y);
            }
        }
    }

    if wrap_y && height >= 2 && width >= 3 {
        for col in 1..width - 1 {
            let top = idx2(0, col, width);
            if nodes[top].extended_mask == NOMASK {
                nodes[top].reliability =
                    reliability_2d_at(image, width, height, 0, col, wrap_x, wrap_y);
            }
            let bottom = idx2(height - 1, col, width);
            if nodes[bottom].extended_mask == NOMASK {
                nodes[bottom].reliability =
                    reliability_2d_at(image, width, height, height - 1, col, wrap_x, wrap_y);
            }
        }
    }
}

fn calculate_reliability_3d(
    volume: &[f64],
    nodes: &mut [UnwrapNode],
    width: usize,
    height: usize,
    depth: usize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) {
    if width >= 3 && height >= 3 && depth >= 3 {
        for z in 1..depth - 1 {
            for row in 1..height - 1 {
                for col in 1..width - 1 {
                    let index = idx3(z, row, col, width, height);
                    if nodes[index].extended_mask == NOMASK {
                        nodes[index].reliability = reliability_3d_at(
                            volume, width, height, depth, z, row, col, wrap_x, wrap_y, wrap_z,
                        );
                    }
                }
            }
        }
    }

    if wrap_x && width >= 2 && height >= 3 && depth >= 3 {
        for z in 1..depth - 1 {
            for row in 1..height - 1 {
                let front = idx3(z, row, 0, width, height);
                if nodes[front].extended_mask == NOMASK {
                    nodes[front].reliability = reliability_3d_at(
                        volume, width, height, depth, z, row, 0, wrap_x, wrap_y, wrap_z,
                    );
                }
                let rear = idx3(z, row, width - 1, width, height);
                if nodes[rear].extended_mask == NOMASK {
                    nodes[rear].reliability = reliability_3d_at(
                        volume,
                        width,
                        height,
                        depth,
                        z,
                        row,
                        width - 1,
                        wrap_x,
                        wrap_y,
                        wrap_z,
                    );
                }
            }
        }
    }

    if wrap_y && height >= 2 && width >= 3 && depth >= 3 {
        for z in 1..depth - 1 {
            for col in 1..width - 1 {
                let left = idx3(z, 0, col, width, height);
                if nodes[left].extended_mask == NOMASK {
                    nodes[left].reliability = reliability_3d_at(
                        volume, width, height, depth, z, 0, col, wrap_x, wrap_y, wrap_z,
                    );
                }
                let right = idx3(z, height - 1, col, width, height);
                if nodes[right].extended_mask == NOMASK {
                    nodes[right].reliability = reliability_3d_at(
                        volume,
                        width,
                        height,
                        depth,
                        z,
                        height - 1,
                        col,
                        wrap_x,
                        wrap_y,
                        wrap_z,
                    );
                }
            }
        }
    }

    if wrap_z && depth >= 2 && width >= 3 && height >= 3 {
        for row in 1..height - 1 {
            for col in 1..width - 1 {
                let bottom = idx3(0, row, col, width, height);
                if nodes[bottom].extended_mask == NOMASK {
                    nodes[bottom].reliability = reliability_3d_at(
                        volume, width, height, depth, 0, row, col, wrap_x, wrap_y, wrap_z,
                    );
                }
                let top = idx3(depth - 1, row, col, width, height);
                if nodes[top].extended_mask == NOMASK {
                    nodes[top].reliability = reliability_3d_at(
                        volume,
                        width,
                        height,
                        depth,
                        depth - 1,
                        row,
                        col,
                        wrap_x,
                        wrap_y,
                        wrap_z,
                    );
                }
            }
        }
    }
}

fn reliability_2d_at(
    image: &[f64],
    width: usize,
    height: usize,
    row: usize,
    col: usize,
    wrap_x: bool,
    wrap_y: bool,
) -> f64 {
    const DIRECTIONS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];
    let center = image[idx2(row, col, width)];
    DIRECTIONS
        .iter()
        .map(|&(dx, dy)| {
            let left = get2_wrapped(
                image,
                width,
                height,
                row as isize - dy,
                col as isize - dx,
                wrap_x,
                wrap_y,
            );
            let right = get2_wrapped(
                image,
                width,
                height,
                row as isize + dy,
                col as isize + dx,
                wrap_x,
                wrap_y,
            );
            let delta = wrap_phase(left - center) - wrap_phase(center - right);
            delta * delta
        })
        .sum()
}

fn reliability_3d_at(
    volume: &[f64],
    width: usize,
    height: usize,
    depth: usize,
    z: usize,
    row: usize,
    col: usize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) -> f64 {
    const DIRECTIONS: [(isize, isize, isize); 13] = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (-1, 1, 0),
        (1, 1, 1),
        (0, 1, 1),
        (-1, 1, 1),
        (1, 0, 1),
        (-1, 0, 1),
        (1, -1, 1),
        (0, -1, 1),
        (-1, -1, 1),
    ];
    let center = volume[idx3(z, row, col, width, height)];
    DIRECTIONS
        .iter()
        .map(|&(dx, dy, dz)| {
            let left = get3_wrapped(
                volume,
                width,
                height,
                depth,
                z as isize - dz,
                row as isize - dy,
                col as isize - dx,
                wrap_x,
                wrap_y,
                wrap_z,
            );
            let right = get3_wrapped(
                volume,
                width,
                height,
                depth,
                z as isize + dz,
                row as isize + dy,
                col as isize + dx,
                wrap_x,
                wrap_y,
                wrap_z,
            );
            let delta = wrap_phase(left - center) - wrap_phase(center - right);
            delta * delta
        })
        .sum()
}

fn build_edges_2d(
    nodes: &[UnwrapNode],
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
) -> Vec<UnwrapEdge> {
    let mut edges = Vec::with_capacity(2 * width * height);
    let mut order = 0_usize;

    for row in 0..height {
        for col in 0..width.saturating_sub(1) {
            push_edge(
                nodes,
                &mut edges,
                idx2(row, col, width),
                idx2(row, col + 1, width),
                &mut order,
            );
        }
    }
    if wrap_x {
        for row in 0..height {
            push_edge(
                nodes,
                &mut edges,
                idx2(row, width - 1, width),
                idx2(row, 0, width),
                &mut order,
            );
        }
    }

    for row in 0..height.saturating_sub(1) {
        for col in 0..width {
            push_edge(
                nodes,
                &mut edges,
                idx2(row, col, width),
                idx2(row + 1, col, width),
                &mut order,
            );
        }
    }
    if wrap_y {
        for col in 0..width {
            push_edge(
                nodes,
                &mut edges,
                idx2(height - 1, col, width),
                idx2(0, col, width),
                &mut order,
            );
        }
    }

    edges
}

fn build_edges_3d(
    nodes: &[UnwrapNode],
    width: usize,
    height: usize,
    depth: usize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) -> PyResult<Vec<UnwrapEdge>> {
    let frame_size = checked_mul(width, height, "volume frame is too large")?;
    let volume_size = checked_mul(frame_size, depth, "volume is too large")?;
    let mut edges = Vec::with_capacity(3 * volume_size);
    let mut order = 0_usize;

    for z in 0..depth {
        for row in 0..height {
            for col in 0..width.saturating_sub(1) {
                push_edge(
                    nodes,
                    &mut edges,
                    idx3(z, row, col, width, height),
                    idx3(z, row, col + 1, width, height),
                    &mut order,
                );
            }
        }
    }
    if wrap_x {
        for z in 0..depth {
            for row in 0..height {
                push_edge(
                    nodes,
                    &mut edges,
                    idx3(z, row, width - 1, width, height),
                    idx3(z, row, 0, width, height),
                    &mut order,
                );
            }
        }
    }

    for z in 0..depth {
        for row in 0..height.saturating_sub(1) {
            for col in 0..width {
                push_edge(
                    nodes,
                    &mut edges,
                    idx3(z, row, col, width, height),
                    idx3(z, row + 1, col, width, height),
                    &mut order,
                );
            }
        }
    }
    if wrap_y {
        for z in 0..depth {
            for col in 0..width {
                push_edge(
                    nodes,
                    &mut edges,
                    idx3(z, height - 1, col, width, height),
                    idx3(z, 0, col, width, height),
                    &mut order,
                );
            }
        }
    }

    for z in 0..depth.saturating_sub(1) {
        for row in 0..height {
            for col in 0..width {
                push_edge(
                    nodes,
                    &mut edges,
                    idx3(z, row, col, width, height),
                    idx3(z + 1, row, col, width, height),
                    &mut order,
                );
            }
        }
    }
    if wrap_z {
        for row in 0..height {
            for col in 0..width {
                push_edge(
                    nodes,
                    &mut edges,
                    idx3(depth - 1, row, col, width, height),
                    idx3(0, row, col, width, height),
                    &mut order,
                );
            }
        }
    }

    Ok(edges)
}

fn push_edge(
    nodes: &[UnwrapNode],
    edges: &mut Vec<UnwrapEdge>,
    pointer_1: usize,
    pointer_2: usize,
    order: &mut usize,
) {
    if nodes[pointer_1].input_mask == NOMASK && nodes[pointer_2].input_mask == NOMASK {
        edges.push(UnwrapEdge {
            reliability: nodes[pointer_1].reliability + nodes[pointer_2].reliability,
            pointer_1,
            pointer_2,
            increment: find_wrap(nodes[pointer_1].value, nodes[pointer_2].value),
            order: *order,
        });
        *order += 1;
    }
}

fn sort_edges(edges: &mut [UnwrapEdge]) {
    edges.sort_by(|left, right| {
        left.reliability
            .total_cmp(&right.reliability)
            .then_with(|| left.order.cmp(&right.order))
    });
}

fn gather_nodes(nodes: &mut [UnwrapNode], edges: &[UnwrapEdge]) {
    for edge in edges {
        let pixel_1 = edge.pointer_1;
        let pixel_2 = edge.pointer_2;
        let head_1 = nodes[pixel_1].head;
        let head_2 = nodes[pixel_2].head;
        if head_1 == head_2 {
            continue;
        }

        if nodes[pixel_2].next.is_none() && nodes[pixel_2].head == pixel_2 {
            let last_1 = nodes[head_1].last;
            nodes[last_1].next = Some(pixel_2);
            nodes[head_1].last = pixel_2;
            nodes[head_1].number_in_group += 1;
            nodes[pixel_2].head = head_1;
            nodes[pixel_2].increment = nodes[pixel_1].increment - edge.increment;
        } else if nodes[pixel_1].next.is_none() && nodes[pixel_1].head == pixel_1 {
            let last_2 = nodes[head_2].last;
            nodes[last_2].next = Some(pixel_1);
            nodes[head_2].last = pixel_1;
            nodes[head_2].number_in_group += 1;
            nodes[pixel_1].head = head_2;
            nodes[pixel_1].increment = nodes[pixel_2].increment + edge.increment;
        } else if nodes[head_1].number_in_group > nodes[head_2].number_in_group {
            let last_1 = nodes[head_1].last;
            let last_2 = nodes[head_2].last;
            nodes[last_1].next = Some(head_2);
            nodes[head_1].last = last_2;
            nodes[head_1].number_in_group += nodes[head_2].number_in_group;
            let increment = nodes[pixel_1].increment - edge.increment - nodes[pixel_2].increment;
            update_group(nodes, head_2, head_1, increment);
        } else {
            let last_1 = nodes[head_1].last;
            let last_2 = nodes[head_2].last;
            nodes[last_2].next = Some(head_1);
            nodes[head_2].last = last_1;
            nodes[head_2].number_in_group += nodes[head_1].number_in_group;
            let increment = nodes[pixel_2].increment + edge.increment - nodes[pixel_1].increment;
            update_group(nodes, head_1, head_2, increment);
        }
    }
}

fn update_group(nodes: &mut [UnwrapNode], start: usize, new_head: usize, increment: i32) {
    let mut current = Some(start);
    while let Some(index) = current {
        nodes[index].head = new_head;
        nodes[index].increment += increment;
        current = nodes[index].next;
    }
}

fn unwrap_nodes(nodes: &mut [UnwrapNode]) {
    for node in nodes.iter_mut() {
        node.value += TWO_PI * f64::from(node.increment);
    }
}

fn mask_nodes(nodes: &mut [UnwrapNode]) {
    let mut min_value = 99_999_999.0;
    for node in nodes.iter() {
        if node.input_mask == NOMASK && node.value < min_value {
            min_value = node.value;
        }
    }
    for node in nodes.iter_mut() {
        if node.input_mask == MASK {
            node.value = min_value;
        }
    }
}

fn idx2(row: usize, col: usize, width: usize) -> usize {
    row * width + col
}

fn idx3(z: usize, row: usize, col: usize, width: usize, height: usize) -> usize {
    z * width * height + row * width + col
}

fn get2_wrapped(
    image: &[f64],
    width: usize,
    height: usize,
    row: isize,
    col: isize,
    wrap_x: bool,
    wrap_y: bool,
) -> f64 {
    let wrapped_row = wrap_index(row, height, wrap_y);
    let wrapped_col = wrap_index(col, width, wrap_x);
    image[idx2(wrapped_row, wrapped_col, width)]
}

fn get3_wrapped(
    volume: &[f64],
    width: usize,
    height: usize,
    depth: usize,
    z: isize,
    row: isize,
    col: isize,
    wrap_x: bool,
    wrap_y: bool,
    wrap_z: bool,
) -> f64 {
    let wrapped_z = wrap_index(z, depth, wrap_z);
    let wrapped_row = wrap_index(row, height, wrap_y);
    let wrapped_col = wrap_index(col, width, wrap_x);
    volume[idx3(wrapped_z, wrapped_row, wrapped_col, width, height)]
}

fn wrap_index(index: isize, len: usize, wrap_around: bool) -> usize {
    if index < 0 {
        debug_assert!(wrap_around);
        len - 1
    } else if index >= len as isize {
        debug_assert!(wrap_around);
        0
    } else {
        index as usize
    }
}

fn checked_wrap_index(index: isize, len: usize, wrap_around: bool) -> Option<usize> {
    if index < 0 {
        wrap_around.then_some(len - 1)
    } else if index >= len as isize {
        wrap_around.then_some(0)
    } else {
        Some(index as usize)
    }
}

fn wrap_phase(value: f64) -> f64 {
    if value > std::f64::consts::PI {
        value - TWO_PI
    } else if value < -std::f64::consts::PI {
        value + TWO_PI
    } else {
        value
    }
}

fn find_wrap(left_value: f64, right_value: f64) -> i32 {
    let difference = left_value - right_value;
    if difference > std::f64::consts::PI {
        -1
    } else if difference < -std::f64::consts::PI {
        1
    } else {
        0
    }
}

fn scan_x(
    labels: &ndarray::ArrayView2<'_, i32>,
    data: &ndarray::ArrayView2<'_, f32>,
    mut x_check: isize,
    y_index: usize,
    step: isize,
    max_gap_x: i32,
    rays_wrap_around: bool,
    right: isize,
) -> (i32, f64) {
    let mut neighbor = labels[[x_check as usize, y_index]];
    let mut nvel = data[[x_check as usize, y_index]] as f64;
    if neighbor == 0 {
        for _ in 0..max_gap_x {
            x_check += step;
            if x_check == -1 {
                if rays_wrap_around {
                    x_check = right;
                } else {
                    break;
                }
            } else if x_check == right + 1 {
                if rays_wrap_around {
                    x_check = 0;
                } else {
                    break;
                }
            }
            neighbor = labels[[x_check as usize, y_index]];
            nvel = data[[x_check as usize, y_index]] as f64;
            if neighbor != 0 {
                break;
            }
        }
    }
    (neighbor, nvel)
}

fn scan_y(
    labels: &ndarray::ArrayView2<'_, i32>,
    data: &ndarray::ArrayView2<'_, f32>,
    x_index: usize,
    mut y_check: isize,
    step: isize,
    max_gap_y: i32,
    bottom: isize,
) -> (i32, f64) {
    let mut neighbor = labels[[x_index, y_check as usize]];
    let mut nvel = data[[x_index, y_check as usize]] as f64;
    if neighbor == 0 {
        for _ in 0..max_gap_y {
            y_check += step;
            if y_check == -1 || y_check == bottom + 1 {
                break;
            }
            neighbor = labels[[x_index, y_check as usize]];
            nvel = data[[x_index, y_check as usize]] as f64;
            if neighbor != 0 {
                break;
            }
        }
    }
    (neighbor, nvel)
}

fn add_edge(
    label: i32,
    neighbor: i32,
    vel: f64,
    nvel: f64,
    l_index: &mut Vec<i32>,
    n_index: &mut Vec<i32>,
    l_velo: &mut Vec<f64>,
    n_velo: &mut Vec<f64>,
) {
    if neighbor == label || neighbor == 0 {
        return;
    }
    l_index.push(label);
    n_index.push(neighbor);
    l_velo.push(vel);
    n_velo.push(nvel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn extended_mask_2d_keeps_only_oracle_reliability_candidates() {
        let mask = vec![NOMASK; 5 * 5];
        let extended = extend_mask_2d(&mask, 5, 5, false, false);

        assert_eq!(extended.iter().filter(|&&value| value == NOMASK).count(), 9);
        assert_eq!(extended[idx2(2, 2, 5)], NOMASK);
        assert_eq!(extended[idx2(0, 2, 5)], MASK);
        assert_eq!(extended[idx2(2, 0, 5)], MASK);

        let extended_wrap_x = extend_mask_2d(&mask, 5, 5, true, false);
        assert_eq!(
            extended_wrap_x
                .iter()
                .filter(|&&value| value == NOMASK)
                .count(),
            15
        );
        assert_eq!(extended_wrap_x[idx2(2, 0, 5)], NOMASK);
        assert_eq!(extended_wrap_x[idx2(0, 0, 5)], MASK);
    }

    #[test]
    fn extended_mask_2d_expands_input_mask_before_reliability() {
        let mut mask = vec![NOMASK; 5 * 5];
        mask[idx2(2, 2, 5)] = MASK;

        let extended = extend_mask_2d(&mask, 5, 5, false, false);

        assert!(extended.iter().all(|&value| value == MASK));
    }

    #[test]
    fn extended_mask_3d_keeps_faces_only_when_axis_wraps() {
        let mask = vec![NOMASK; 5 * 5 * 5];
        let extended = extend_mask_3d(&mask, 5, 5, 5, false, false, false);

        assert_eq!(
            extended.iter().filter(|&&value| value == NOMASK).count(),
            27
        );
        assert_eq!(extended[idx3(2, 2, 2, 5, 5)], NOMASK);
        assert_eq!(extended[idx3(0, 2, 2, 5, 5)], MASK);

        let extended_wrap_z = extend_mask_3d(&mask, 5, 5, 5, false, false, true);
        assert_eq!(
            extended_wrap_z
                .iter()
                .filter(|&&value| value == NOMASK)
                .count(),
            45
        );
        assert_eq!(extended_wrap_z[idx3(0, 2, 2, 5, 5)], NOMASK);
        assert_eq!(extended_wrap_z[idx3(0, 0, 2, 5, 5)], MASK);
    }

    #[test]
    fn smooth_and_trim_kernel_reflects_and_returns_input_length() {
        let x = array![1.0, 2.0, 4.0, 8.0];
        let weights = array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let out = smooth_and_trim_kernel(x.view(), weights.view(), 3);

        assert_eq!(out.len(), x.len());
        let expected = array![
            5.0 * weights[0],
            7.0 * weights[0],
            14.0 * weights[0],
            20.0 * weights[0]
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn unwrap_masked_degrees_preserves_gap_order_and_thresholds() {
        let values = array![10.0, 999.0, 350.0, 20.0, -170.0];
        let mask = array![false, true, false, false, false];

        let out = unwrap_masked_degrees_kernel(values.view(), mask.view());

        assert_eq!(out, vec![10.0, -10.0, 20.0, 190.0]);

        let thresholds = array![0.0, 180.0, 0.0, -180.0];
        let no_mask = array![false, false, false, false];
        let out = unwrap_masked_degrees_kernel(thresholds.view(), no_mask.view());
        assert_eq!(out, vec![0.0, 180.0, 0.0, -180.0]);
    }

    #[test]
    fn end_gate_from_excluded_mask_matches_oracle_edges() {
        let mask = array![
            [false, false, false, false],
            [true, false, false, false],
            [false, true, false, false],
            [false, false, false, true],
            [true, true, true, true],
        ];

        let out = end_gate_from_excluded_mask_kernel(mask.view()).unwrap();

        assert_eq!(out, vec![3, 0, 0, 2, 0]);

        let zero_gates = ndarray::Array2::<bool>::from_shape_vec((2, 0), vec![]).unwrap();
        let out = end_gate_from_excluded_mask_kernel(zero_gates.view()).unwrap();
        assert_eq!(out, vec![-1, -1]);
    }

    #[test]
    fn sweep_interval_splits_expands_outside_nyquist() {
        let velocities = array![-18.0, -7.5, 3.0, 16.0];
        let (start, end, num, outside) =
            sweep_interval_splits_kernel(10.0, 4, &velocities.view()).unwrap();

        assert_eq!(start, -20.0);
        assert_eq!(end, 20.0);
        assert_eq!(num, 9);
        assert!(outside);
    }

    #[test]
    fn sweep_interval_splits_empty_keeps_default_limits() {
        let velocities = ndarray::Array1::<f64>::zeros(0);
        let (start, end, num, outside) =
            sweep_interval_splits_kernel(10.0, 4, &velocities.view()).unwrap();

        assert_eq!(start, -10.0);
        assert_eq!(end, 10.0);
        assert_eq!(num, 5);
        assert!(!outside);
    }

    #[test]
    fn sweep_interval_splits_rejects_unbounded_extension() {
        let velocities = array![f64::MAX];

        assert!(sweep_interval_splits_kernel(1.0e-308, 4, &velocities.view()).is_err());
    }

    #[test]
    fn edge_sum_and_count_sorts_and_reduces_duplicate_edges() {
        let index1 = array![2, 1, 2, 1, 1];
        let index2 = array![5, 4, 5, 4, 3];
        let vel1 = array![2.0, 1.0, 20.0, 10.0, 100.0];
        let vel2 = array![3.0, 4.0, 30.0, 40.0, 400.0];

        let (out_index1, out_index2, count, out_vel1, out_vel2) =
            edge_sum_and_count_kernel(index1.view(), index2.view(), vel1.view(), vel2.view());

        assert_eq!(out_index1, vec![1, 1, 2]);
        assert_eq!(out_index2, vec![3, 4, 5]);
        assert_eq!(count, vec![1, 2, 2]);
        assert_eq!(out_vel1, vec![100.0, 11.0, 22.0]);
        assert_eq!(out_vel2, vec![400.0, 44.0, 33.0]);
    }
}
