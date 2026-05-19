use ndarray::{ArrayView1, ArrayViewMut3, ArrayViewMut4};
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2, PyReadwriteArray3,
    PyReadwriteArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

const BARNES: i32 = 0;
const NEAREST: i32 = 2;
const BARNES2: i32 = 3;
const GATE_TO_GRID_ROI_MAX_POINTS: usize = 128 * 1024 * 1024;

#[pyfunction]
pub fn _load_nn_field_data<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, Py<PyAny>>,
    nfields: i32,
    npoints: i32,
    r_nums: PyReadonlyArray1<'py, i32>,
    e_nums: PyReadonlyArray1<'py, i32>,
    mut sdata: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let data_view = data.as_array();
    let r_view = r_nums.as_array();
    let e_view = e_nums.as_array();
    let mut out = sdata.as_array_mut();
    let out_shape = out.dim();
    if nfields < 0 || npoints < 0 {
        return Err(PyValueError::new_err(
            "nfields and npoints must be non-negative",
        ));
    }
    let nfields = nfields as usize;
    let npoints = npoints as usize;
    if nfields > data_view.dim().0 || nfields > out_shape.1 {
        return Err(PyValueError::new_err("nfields exceeds available fields"));
    }
    if npoints > r_view.len() || npoints > e_view.len() || npoints > out_shape.0 {
        return Err(PyValueError::new_err("npoints exceeds available points"));
    }

    for i in 0..npoints {
        let r_num = unsigned_int_index(
            *r_view
                .get(i)
                .ok_or_else(|| index_error("r_nums", i, r_view.len()))?,
        );
        let e_num = unsigned_int_index(
            *e_view
                .get(i)
                .ok_or_else(|| index_error("e_nums", i, e_view.len()))?,
        );

        for j in 0..nfields {
            let field = data_view
                .get((j, r_num))
                .ok_or_else(|| index_error_2d("data", j, r_num, data_view.dim()))?
                .bind(py);
            let value = field.get_item(e_num)?.extract::<f64>()?;
            let slot = out
                .get_mut((i, j))
                .ok_or_else(|| index_error_2d("sdata", i, j, out_shape))?;
            *slot = value;
        }
    }

    Ok(())
}

#[pyfunction(name = "_gate_to_grid_find_min")]
pub fn py_gate_to_grid_find_min(a: f32, roi: f32, step: f32) -> i32 {
    gate_to_grid_find_min(a, roi, step)
}

#[pyfunction(name = "_gate_to_grid_find_max")]
pub fn py_gate_to_grid_find_max(a: f32, roi: f32, step: f32, na: i32) -> i32 {
    gate_to_grid_find_max(a, roi, step, na)
}

#[pyfunction(name = "_gate_to_grid_roi_constant_f32")]
pub fn py_gate_to_grid_roi_constant_f32(
    mut roi_array: PyReadwriteArray3<'_, f32>,
    constant_roi: f32,
) -> PyResult<()> {
    validate_roi_output(&roi_array)?;
    if !constant_roi.is_finite() {
        return Err(PyValueError::new_err("constant_roi must be finite"));
    }

    let roi_array = roi_array.as_array_mut();
    gate_to_grid_roi_constant_f32(roi_array, constant_roi);
    Ok(())
}

#[pyfunction(name = "_gate_to_grid_roi_dist_f32")]
#[allow(clippy::too_many_arguments)]
pub fn py_gate_to_grid_roi_dist_f32(
    mut roi_array: PyReadwriteArray3<'_, f32>,
    offsets: PyReadonlyArray2<'_, f64>,
    z_start: f64,
    y_start: f64,
    x_start: f64,
    z_step: f64,
    y_step: f64,
    x_step: f64,
    z_factor: f64,
    xy_factor: f64,
    min_radius: f64,
) -> PyResult<()> {
    validate_roi_output(&roi_array)?;
    validate_offsets(&offsets)?;
    validate_f64_scalars(&[
        z_start, y_start, x_start, z_step, y_step, x_step, z_factor, xy_factor, min_radius,
    ])?;

    let roi_array = roi_array.as_array_mut();
    let offsets = offsets.as_array();
    gate_to_grid_roi_dist_f32(
        roi_array, offsets, z_start, y_start, x_start, z_step, y_step, x_step, z_factor, xy_factor,
        min_radius,
    );
    Ok(())
}

#[pyfunction(name = "_gate_to_grid_roi_dist_beam_f32")]
#[allow(clippy::too_many_arguments)]
pub fn py_gate_to_grid_roi_dist_beam_f32(
    mut roi_array: PyReadwriteArray3<'_, f32>,
    offsets: PyReadonlyArray2<'_, f64>,
    h_factor: PyReadonlyArray1<'_, f32>,
    z_start: f64,
    y_start: f64,
    x_start: f64,
    z_step: f64,
    y_step: f64,
    x_step: f64,
    beam_factor: f64,
    min_radius: f64,
) -> PyResult<()> {
    validate_roi_output(&roi_array)?;
    validate_offsets(&offsets)?;
    if !h_factor.is_c_contiguous() || h_factor.len() != 3 {
        return Err(PyValueError::new_err(
            "h_factor must be a C-contiguous array of length 3",
        ));
    }
    validate_f64_scalars(&[
        z_start,
        y_start,
        x_start,
        z_step,
        y_step,
        x_step,
        beam_factor,
        min_radius,
    ])?;

    let h_factor_view = h_factor.as_array();
    if h_factor_view.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("h_factor values must be finite"));
    }

    let roi_array = roi_array.as_array_mut();
    let offsets = offsets.as_array();
    gate_to_grid_roi_dist_beam_f32(
        roi_array,
        offsets,
        h_factor_view,
        z_start,
        y_start,
        x_start,
        z_step,
        y_step,
        x_step,
        beam_factor,
        min_radius,
    );
    Ok(())
}

#[pyfunction(name = "_gate_to_grid_map_gate")]
#[allow(clippy::too_many_arguments)]
pub fn py_gate_to_grid_map_gate<'py>(
    x: f32,
    y: f32,
    z: f32,
    roi: f32,
    x_start: f32,
    y_start: f32,
    z_start: f32,
    x_step: f32,
    y_step: f32,
    z_step: f32,
    nx: i32,
    ny: i32,
    nz: i32,
    mut grid_sum: PyReadwriteArray4<'py, f32>,
    mut grid_wsum: PyReadwriteArray4<'py, f32>,
    mut min_dist2: PyReadwriteArray4<'py, f64>,
    values: PyReadonlyArray1<'py, f32>,
    masks: PyReadonlyArray1<'py, u8>,
    weighting_function: i32,
    dist_factor: PyReadonlyArray1<'py, f32>,
) -> PyResult<i32> {
    if nx < 0 || ny < 0 || nz < 0 {
        return Err(PyValueError::new_err(
            "grid dimensions must be non-negative",
        ));
    }

    let mut grid_sum = grid_sum.as_array_mut();
    let mut grid_wsum = grid_wsum.as_array_mut();
    let mut min_dist2 = min_dist2.as_array_mut();
    let values = values.as_array();
    let masks = masks.as_array();
    let dist_factor = dist_factor.as_array();

    let shape = grid_sum.dim();
    if grid_wsum.dim() != shape || min_dist2.dim() != shape {
        return Err(PyValueError::new_err(
            "grid_sum, grid_wsum, and min_dist2 must have the same shape",
        ));
    }
    if shape.0 != nz as usize || shape.1 != ny as usize || shape.2 != nx as usize {
        return Err(PyValueError::new_err(
            "mapper dimensions must match grid array shape",
        ));
    }
    if values.len() != shape.3 || masks.len() != shape.3 {
        return Err(PyValueError::new_err(
            "values and masks length must match grid field count",
        ));
    }
    if dist_factor.len() != 3 {
        return Err(PyValueError::new_err("dist_factor must have length 3"));
    }

    Ok(gate_to_grid_map_gate(
        x,
        y,
        z,
        roi,
        x_start,
        y_start,
        z_start,
        x_step,
        y_step,
        z_step,
        nx,
        ny,
        nz,
        grid_sum.view_mut(),
        grid_wsum.view_mut(),
        min_dist2.view_mut(),
        values,
        masks,
        weighting_function,
        dist_factor,
    ))
}

#[pyfunction(name = "_gate_mapper_apply_field_f64")]
pub fn py_gate_mapper_apply_field_f64(
    index_map: PyReadonlyArray3<'_, f64>,
    src_data: PyReadonlyArray2<'_, f64>,
    src_mask: PyReadonlyArray2<'_, bool>,
    mut out_data: PyReadwriteArray2<'_, f64>,
    mut out_mask: PyReadwriteArray2<'_, bool>,
) -> PyResult<()> {
    if !index_map.is_c_contiguous()
        || !src_data.is_c_contiguous()
        || !src_mask.is_c_contiguous()
        || !out_data.is_c_contiguous()
        || !out_mask.is_c_contiguous()
    {
        return Err(PyValueError::new_err(
            "index_map, source arrays, and output arrays must be C-contiguous",
        ));
    }

    let index_map = index_map.as_array();
    let src_data = src_data.as_array();
    let src_mask = src_mask.as_array();
    let mut out_data = out_data.as_array_mut();
    let mut out_mask = out_mask.as_array_mut();
    validate_gate_mapper_inputs(
        index_map,
        src_data,
        src_mask,
        out_data.view(),
        out_mask.view(),
    )?;
    gate_mapper_apply_field_f64(
        index_map,
        src_data,
        src_mask,
        out_data.view_mut(),
        out_mask.view_mut(),
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_load_nn_field_data, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_to_grid_find_min, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_to_grid_find_max, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_to_grid_roi_constant_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_to_grid_roi_dist_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_to_grid_roi_dist_beam_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_to_grid_map_gate, module)?)?;
    module.add_function(wrap_pyfunction!(py_gate_mapper_apply_field_f64, module)?)?;
    Ok(())
}

fn gate_to_grid_find_min(a: f32, roi: f32, step: f32) -> i32 {
    if step == 0.0 {
        return 0;
    }
    let mut a_min = ((a - roi) / step).ceil() as i32;
    if a_min < 0 {
        a_min = 0;
    }
    a_min
}

fn gate_to_grid_find_max(a: f32, roi: f32, step: f32, na: i32) -> i32 {
    if step == 0.0 {
        return 0;
    }
    let mut a_max = ((a + roi) / step).floor() as i32;
    if a_max > na - 1 {
        a_max = na - 1;
    }
    a_max
}

fn validate_roi_output(roi_array: &PyReadwriteArray3<'_, f32>) -> PyResult<()> {
    if !roi_array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "roi_array must be C-contiguous float32",
        ));
    }
    let size = roi_array.len();
    if size > GATE_TO_GRID_ROI_MAX_POINTS {
        return Err(PyValueError::new_err("roi_array exceeds native size limit"));
    }
    Ok(())
}

fn validate_offsets(offsets: &PyReadonlyArray2<'_, f64>) -> PyResult<()> {
    if !offsets.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "offsets must be C-contiguous float64",
        ));
    }
    if offsets.shape()[1] != 3 {
        return Err(PyValueError::new_err("offsets must have shape (n, 3)"));
    }
    let offsets_view = offsets.as_array();
    if offsets_view.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("offset values must be finite"));
    }
    Ok(())
}

fn validate_f64_scalars(values: &[f64]) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("ROI scalar values must be finite"));
    }
    Ok(())
}

fn gate_to_grid_roi_constant_f32(mut roi_array: ArrayViewMut3<'_, f32>, constant_roi: f32) {
    roi_array.fill(constant_roi);
}

#[allow(clippy::too_many_arguments)]
fn gate_to_grid_roi_dist_f32(
    mut roi_array: ArrayViewMut3<'_, f32>,
    offsets: ndarray::ArrayView2<'_, f64>,
    z_start: f64,
    y_start: f64,
    x_start: f64,
    z_step: f64,
    y_step: f64,
    x_step: f64,
    z_factor: f64,
    xy_factor: f64,
    min_radius: f64,
) {
    let (nz, ny, nx) = roi_array.dim();
    for ix in 0..nx {
        let x = x_start + x_step * ix as f64;
        for iy in 0..ny {
            let y = y_start + y_step * iy as f64;
            for iz in 0..nz {
                let z = z_start + z_step * iz as f64;
                let mut min_roi = 999999999.0_f64;
                for offset in offsets.outer_iter() {
                    let z_offset = offset[0];
                    let y_offset = offset[1];
                    let x_offset = offset[2];
                    let mut roi = z_factor * (z - z_offset)
                        + xy_factor * ((x - x_offset).powi(2) + (y - y_offset).powi(2)).sqrt();
                    if roi < min_radius {
                        roi = min_radius;
                    }
                    if roi < min_roi {
                        min_roi = roi;
                    }
                }
                roi_array[[iz, iy, ix]] = min_roi as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gate_to_grid_roi_dist_beam_f32(
    mut roi_array: ArrayViewMut3<'_, f32>,
    offsets: ndarray::ArrayView2<'_, f64>,
    h_factor: ArrayView1<'_, f32>,
    z_start: f64,
    y_start: f64,
    x_start: f64,
    z_step: f64,
    y_step: f64,
    x_step: f64,
    beam_factor: f64,
    min_radius: f64,
) {
    let (nz, ny, nx) = roi_array.dim();
    let hz = h_factor[0];
    let hy = h_factor[1];
    let hx = h_factor[2];
    for ix in 0..nx {
        let x = x_start + x_step * ix as f64;
        for iy in 0..ny {
            let y = y_start + y_step * iy as f64;
            for iz in 0..nz {
                let z = z_start + z_step * iz as f64;
                let mut min_roi = 999999999.0_f64;
                for offset in offsets.outer_iter() {
                    let z_offset = offset[0];
                    let y_offset = offset[1];
                    let x_offset = offset[2];
                    let dz = hz * (z - z_offset) as f32;
                    let dy = hy * (y - y_offset) as f32;
                    let dx = hx * (x - x_offset) as f32;
                    let distance2 = dz.powi(2) + dy.powi(2) + dx.powi(2);
                    let mut roi = (distance2 as f64).sqrt() * beam_factor;
                    if roi < min_radius {
                        roi = min_radius;
                    }
                    if roi < min_roi {
                        min_roi = roi;
                    }
                }
                roi_array[[iz, iy, ix]] = min_roi as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gate_to_grid_map_gate(
    mut x: f32,
    mut y: f32,
    mut z: f32,
    roi: f32,
    x_start: f32,
    y_start: f32,
    z_start: f32,
    x_step: f32,
    y_step: f32,
    z_step: f32,
    nx: i32,
    ny: i32,
    nz: i32,
    mut grid_sum: ArrayViewMut4<'_, f32>,
    mut grid_wsum: ArrayViewMut4<'_, f32>,
    mut min_dist2: ArrayViewMut4<'_, f64>,
    values: ArrayView1<'_, f32>,
    masks: ArrayView1<'_, u8>,
    weighting_function: i32,
    dist_factor: ArrayView1<'_, f32>,
) -> i32 {
    x -= x_start;
    y -= y_start;
    z -= z_start;

    let x_min = gate_to_grid_find_min(x, roi, x_step);
    if x_min > nx - 1 {
        return 0;
    }
    let x_max = gate_to_grid_find_max(x, roi, x_step, nx);
    if x_max < 0 {
        return 0;
    }

    let y_min = gate_to_grid_find_min(y, roi, y_step);
    if y_min > ny - 1 {
        return 0;
    }
    let y_max = gate_to_grid_find_max(y, roi, y_step, ny);
    if y_max < 0 {
        return 0;
    }

    let z_min = gate_to_grid_find_min(z, roi, z_step);
    if z_min > nz - 1 {
        return 0;
    }
    let z_max = gate_to_grid_find_max(z, roi, z_step, nz);
    if z_max < 0 {
        return 0;
    }

    let roi2 = roi * roi;
    let nfields = values.len();

    if weighting_function == NEAREST {
        for xi in x_min..=x_max {
            let xiu = xi as usize;
            let xg = x_step * xi as f32;
            for yi in y_min..=y_max {
                let yiu = yi as usize;
                let yg = y_step * yi as f32;
                for zi in z_min..=z_max {
                    let ziu = zi as usize;
                    let zg = z_step * zi as f32;
                    let dist2 = dist_factor[2] * (xg - x).powi(2)
                        + dist_factor[1] * (yg - y).powi(2)
                        + dist_factor[0] * (zg - z).powi(2);
                    if dist2 >= roi2 {
                        continue;
                    }
                    for i in 0..nfields {
                        if (dist2 as f64) < min_dist2[[ziu, yiu, xiu, i]] {
                            min_dist2[[ziu, yiu, xiu, i]] = dist2 as f64;
                            if masks[i] != 0 {
                                grid_wsum[[ziu, yiu, xiu, i]] = 0.0;
                                grid_sum[[ziu, yiu, xiu, i]] = 0.0;
                            } else {
                                grid_wsum[[ziu, yiu, xiu, i]] = 1.0;
                                grid_sum[[ziu, yiu, xiu, i]] = values[i];
                            }
                        }
                    }
                }
            }
        }
        return 1;
    }

    for xi in x_min..=x_max {
        let xiu = xi as usize;
        let xg = x_step * xi as f32;
        for yi in y_min..=y_max {
            let yiu = yi as usize;
            let yg = y_step * yi as f32;
            for zi in z_min..=z_max {
                let ziu = zi as usize;
                let zg = z_step * zi as f32;
                let dist2 = dist_factor[2] * (xg - x) * (xg - x)
                    + dist_factor[1] * (yg - y) * (yg - y)
                    + dist_factor[0] * (zg - z) * (zg - z);
                if dist2 > roi2 {
                    continue;
                }

                let weight = if weighting_function == BARNES {
                    let ratio = dist2 / (2.0 * roi2);
                    ((-(ratio as f64)).exp() + 1e-5_f64) as f32
                } else if weighting_function == BARNES2 {
                    let ratio = dist2 / (roi2 / 4.0);
                    ((-(ratio as f64)).exp() + 1e-5_f64) as f32
                } else {
                    (roi2 - dist2) / (roi2 + dist2)
                };

                for i in 0..nfields {
                    if masks[i] != 0 {
                        continue;
                    }
                    grid_sum[[ziu, yiu, xiu, i]] += weight * values[i];
                    grid_wsum[[ziu, yiu, xiu, i]] += weight;
                }
            }
        }
    }
    1
}

fn validate_gate_mapper_inputs(
    index_map: ndarray::ArrayView3<'_, f64>,
    src_data: ndarray::ArrayView2<'_, f64>,
    src_mask: ndarray::ArrayView2<'_, bool>,
    out_data: ndarray::ArrayView2<'_, f64>,
    out_mask: ndarray::ArrayView2<'_, bool>,
) -> PyResult<()> {
    let src_shape = src_data.dim();
    if index_map.dim() != (src_shape.0, src_shape.1, 2) {
        return Err(PyValueError::new_err(
            "index_map must have shape (src_nrays, src_ngates, 2)",
        ));
    }
    if src_mask.dim() != src_shape {
        return Err(PyValueError::new_err(
            "src_data and src_mask must have the same shape",
        ));
    }
    if out_data.dim() != out_mask.dim() {
        return Err(PyValueError::new_err(
            "out_data and out_mask must have the same shape",
        ));
    }
    if !index_map.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err("index_map values must be finite"));
    }
    validate_gate_mapper_indices(index_map, out_data.dim())
}

fn validate_gate_mapper_indices(
    index_map: ndarray::ArrayView3<'_, f64>,
    out_shape: (usize, usize),
) -> PyResult<()> {
    for ray in 0..index_map.dim().0 {
        for gate in 0..index_map.dim().1 {
            let dest_ray = checked_gate_mapper_index(index_map[[ray, gate, 0]])?;
            let dest_gate = checked_gate_mapper_index(index_map[[ray, gate, 1]])?;
            if dest_ray <= 0 {
                continue;
            }
            if dest_ray as usize >= out_shape.0
                || dest_gate < 0
                || dest_gate as usize >= out_shape.1
            {
                return Err(PyValueError::new_err(
                    "positive destination indexes must be in bounds",
                ));
            }
        }
    }
    Ok(())
}

fn checked_gate_mapper_index(value: f64) -> PyResult<isize> {
    if !value.is_finite() || value < isize::MIN as f64 || value > isize::MAX as f64 {
        return Err(PyValueError::new_err(
            "index_map values must fit in platform index range",
        ));
    }
    Ok(value as isize)
}

fn gate_mapper_apply_field_f64(
    index_map: ndarray::ArrayView3<'_, f64>,
    src_data: ndarray::ArrayView2<'_, f64>,
    src_mask: ndarray::ArrayView2<'_, bool>,
    mut out_data: ndarray::ArrayViewMut2<'_, f64>,
    mut out_mask: ndarray::ArrayViewMut2<'_, bool>,
) -> PyResult<()> {
    let (src_nrays, src_ngates) = src_data.dim();
    for ray in 0..src_nrays {
        for gate in 0..src_ngates {
            let dest_ray = checked_gate_mapper_index(index_map[[ray, gate, 0]])?;
            if dest_ray <= 0 {
                continue;
            }
            let dest_gate = checked_gate_mapper_index(index_map[[ray, gate, 1]])?;
            let target = (dest_ray as usize, dest_gate as usize);
            if src_mask[[ray, gate]] {
                out_mask[target] = true;
            } else {
                out_data[target] = src_data[[ray, gate]];
                out_mask[target] = false;
            }
        }
    }
    Ok(())
}

fn unsigned_int_index(value: i32) -> usize {
    value as u32 as usize
}

fn index_error(name: &str, index: usize, len: usize) -> PyErr {
    PyIndexError::new_err(format!(
        "{name} index {index} is out of bounds for length {len}"
    ))
}

fn index_error_2d(name: &str, row: usize, col: usize, shape: (usize, usize)) -> PyErr {
    PyIndexError::new_err(format!(
        "{name} index ({row}, {col}) is out of bounds for shape ({}, {})",
        shape.0, shape.1
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array4};

    #[test]
    fn gate_to_grid_bounds_match_oracle_clamping() {
        assert_eq!(gate_to_grid_find_min(2.4, 1.0, 1.0), 2);
        assert_eq!(gate_to_grid_find_min(-2.4, 1.0, 1.0), 0);
        assert_eq!(gate_to_grid_find_min(2.4, 1.0, 0.0), 0);
        assert_eq!(gate_to_grid_find_max(2.4, 1.0, 1.0, 3), 2);
        assert_eq!(gate_to_grid_find_max(-2.4, 1.0, 1.0, 3), -2);
        assert_eq!(gate_to_grid_find_max(2.4, 1.0, 0.0, 3), 0);
    }

    #[test]
    fn gate_to_grid_map_gate_cressman_accumulates_unmasked_fields() {
        let mut grid_sum = Array4::<f32>::zeros((1, 2, 2, 2));
        let mut grid_wsum = Array4::<f32>::zeros((1, 2, 2, 2));
        let mut min_dist2 = Array4::<f64>::from_elem((1, 2, 2, 2), 1e30);
        let values = array![10.0_f32, 20.0_f32];
        let masks = array![0_u8, 1_u8];
        let dist_factor = array![1.0_f32, 1.0_f32, 1.0_f32];

        let mapped = gate_to_grid_map_gate(
            0.0,
            0.0,
            0.0,
            1.5,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            2,
            2,
            1,
            grid_sum.view_mut(),
            grid_wsum.view_mut(),
            min_dist2.view_mut(),
            values.view(),
            masks.view(),
            1,
            dist_factor.view(),
        );

        assert_eq!(mapped, 1);
        assert!(grid_sum[[0, 0, 0, 0]] > 0.0);
        assert_eq!(grid_sum[[0, 0, 0, 1]], 0.0);
        assert!(grid_wsum[[0, 0, 0, 0]] > 0.0);
        assert_eq!(grid_wsum[[0, 0, 0, 1]], 0.0);
    }

    #[test]
    fn gate_to_grid_map_gate_nearest_replaces_only_closer_values() {
        let mut grid_sum = Array4::<f32>::zeros((1, 1, 1, 1));
        let mut grid_wsum = Array4::<f32>::zeros((1, 1, 1, 1));
        let mut min_dist2 = Array4::<f64>::from_elem((1, 1, 1, 1), 0.01);
        let values = array![7.0_f32];
        let masks = array![0_u8];
        let dist_factor = array![1.0_f32, 1.0_f32, 1.0_f32];

        gate_to_grid_map_gate(
            0.2,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1,
            1,
            1,
            grid_sum.view_mut(),
            grid_wsum.view_mut(),
            min_dist2.view_mut(),
            values.view(),
            masks.view(),
            NEAREST,
            dist_factor.view(),
        );
        assert_eq!(grid_sum[[0, 0, 0, 0]], 0.0);

        gate_to_grid_map_gate(
            0.05,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1,
            1,
            1,
            grid_sum.view_mut(),
            grid_wsum.view_mut(),
            min_dist2.view_mut(),
            values.view(),
            masks.view(),
            NEAREST,
            dist_factor.view(),
        );
        assert_eq!(grid_sum[[0, 0, 0, 0]], 7.0);
        assert_eq!(grid_wsum[[0, 0, 0, 0]], 1.0);
    }

    #[test]
    fn gate_mapper_apply_field_preserves_order_and_masked_payloads() {
        let index_map = ndarray::Array3::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0],
        )
        .unwrap();
        let src_data = array![[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];
        let src_mask = array![[false, true, false], [false, true, false],];
        let mut out_data =
            ndarray::Array2::from_shape_vec((3, 3), (100..109).map(|v| v as f64).collect())
                .unwrap();
        let mut out_mask = ndarray::Array2::<bool>::from_elem((3, 3), true);

        gate_mapper_apply_field_f64(
            index_map.view(),
            src_data.view(),
            src_mask.view(),
            out_data.view_mut(),
            out_mask.view_mut(),
        )
        .unwrap();

        assert_eq!(out_data[[1, 1]], 12.0);
        assert!(!out_mask[[1, 1]]);
        assert_eq!(out_data[[2, 0]], 13.0);
        assert!(out_mask[[2, 0]]);
        assert_eq!(out_data[[0, 2]], 102.0);
        assert!(out_mask[[0, 2]]);
    }
}
