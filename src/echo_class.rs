use ndarray::{Array2, Array3, ArrayD};
use numpy::{
    PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn, PyReadwriteArray2, PyReadwriteArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "_assign_to_class_scan_no_entropy")]
pub fn py_assign_to_class_scan_no_entropy<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    masks: PyReadonlyArray3<'py, bool>,
    mass_centers: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let data = data.as_array();
    let masks = masks.as_array();
    let mass_centers = mass_centers.as_array();
    let weights = weights.as_array();

    let (nvariables, nrays, nbins) = data.dim();
    if masks.dim() != (nvariables, nrays, nbins) {
        return Err(PyValueError::new_err("mask shape must match data shape"));
    }

    let (nclasses, center_variables) = mass_centers.dim();
    if nclasses == 0 {
        return Err(PyValueError::new_err(
            "mass_centers must include at least one class",
        ));
    }
    if nvariables == 0 {
        return Err(PyValueError::new_err(
            "data must include at least one variable",
        ));
    }
    if center_variables != nvariables {
        return Err(PyValueError::new_err(
            "mass_centers variable count must match data variable count",
        ));
    }
    if weights.len() != nvariables {
        return Err(PyValueError::new_err(
            "weights length must match data variable count",
        ));
    }

    let output = assign_to_class_scan_no_entropy_kernel(
        data.view(),
        masks.view(),
        mass_centers.view(),
        weights.view(),
    );
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction(name = "_atwt2d")]
pub fn py_atwt2d<'py>(
    py: Python<'py>,
    mut data: PyReadwriteArray2<'py, f64>,
    max_scale: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (ny, nx) = data.as_array().dim();
    validate_atwt_shape(ny, nx, max_scale)?;
    let mut data = data.as_array_mut();
    let wt = atwt2d_kernel(data.view_mut(), max_scale, ny, nx);
    Ok(PyArray3::from_owned_array(py, wt))
}

#[pyfunction(name = "_create_radial_mask_circular")]
pub fn py_create_radial_mask_circular(
    mut mask_array: PyReadwriteArray2<'_, f64>,
    min_rad_km: f64,
    max_rad_km: f64,
    x_pixsize: f64,
    y_pixsize: f64,
    center_x: f64,
    center_y: f64,
) -> PyResult<()> {
    let (rows, cols) = mask_array.as_array().dim();
    if rows != cols {
        return Err(PyValueError::new_err(
            "circular radial mask Rust path requires a square array",
        ));
    }

    let mut mask = mask_array.as_array_mut();
    create_radial_mask_circular_kernel(
        mask.view_mut(),
        min_rad_km,
        max_rad_km,
        x_pixsize,
        y_pixsize,
        center_x,
        center_y,
    );
    Ok(())
}

#[pyfunction(name = "_echo_class_get_freq_band")]
pub fn py_get_freq_band(freq: f64) -> Option<&'static str> {
    get_freq_band_kernel(freq)
}

#[pyfunction(name = "_echo_class_standardize_linear")]
pub fn py_standardize_linear<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    mx: f64,
    mn: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if !mx.is_finite() || !mn.is_finite() || mx == mn {
        return Err(PyValueError::new_err(
            "mx and mn must be finite and distinct",
        ));
    }

    let data = data.as_array();
    if !data.is_standard_layout() {
        return Err(PyValueError::new_err("data must be C-contiguous"));
    }
    if data
        .iter()
        .any(|value| !value.is_finite() || value.abs() >= 1.0e100)
    {
        return Err(PyValueError::new_err(
            "data must be finite and within the supported range",
        ));
    }

    let output = standardize_linear_kernel(data.view(), mx, mn);
    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_assign_feature_radius_km")]
pub fn py_assign_feature_radius_km<'py>(
    py: Python<'py>,
    field_bkg: PyReadonlyArrayDyn<'py, f64>,
    val_for_max_rad: f64,
    max_rad: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let field_bkg = field_bkg.as_array();
    if !field_bkg.is_standard_layout() {
        return Err(PyValueError::new_err("field_bkg must be C-contiguous"));
    }
    if !val_for_max_rad.is_finite()
        || !max_rad.is_finite()
        || field_bkg.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "field_bkg, val_for_max_rad, and max_rad must be finite",
        ));
    }

    let output = assign_feature_radius_km_kernel(field_bkg.view(), val_for_max_rad, max_rad);
    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_echo_class_wt_label_classes_f64")]
#[allow(clippy::too_many_arguments)]
pub fn py_echo_class_wt_label_classes_f64<'py>(
    py: Python<'py>,
    wt_sum: PyReadonlyArrayDyn<'py, f64>,
    dbz_data: PyReadonlyArrayDyn<'py, f64>,
    core_wt_threshold: f64,
    conv_wt_threshold: f64,
    min_reflectivity: f64,
    conv_min_refl: f64,
    conv_core_threshold: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let wt_sum = wt_sum.as_array();
    let dbz_data = dbz_data.as_array();
    validate_wt_label_inputs(&wt_sum, &dbz_data)?;
    let output = echo_class_wt_label_classes_kernel(
        wt_sum.view(),
        dbz_data.view(),
        core_wt_threshold,
        conv_wt_threshold,
        min_reflectivity,
        conv_min_refl,
        conv_core_threshold,
    );
    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_classify_feature_array_f64")]
#[allow(clippy::too_many_arguments)]
pub fn py_classify_feature_array_f64(
    field: PyReadonlyArrayDyn<'_, f64>,
    field_mask: PyReadonlyArrayDyn<'_, bool>,
    mut feature_array: PyReadwriteArrayDyn<'_, f64>,
    core_array: PyReadonlyArrayDyn<'_, f64>,
    core_mask: PyReadonlyArrayDyn<'_, bool>,
    nosfcecho: f64,
    feat_val: f64,
    bkgd_val: f64,
    weakecho: f64,
    core: f64,
    mindbzuse: f64,
    weakechothres: f64,
) -> PyResult<()> {
    let field = field.as_array();
    let field_mask = field_mask.as_array();
    let core_array = core_array.as_array();
    let core_mask = core_mask.as_array();
    let mut feature_array = feature_array.as_array_mut();
    validate_classify_feature_inputs(&field, &field_mask, &feature_array, &core_array, &core_mask)?;
    classify_feature_array_kernel(
        field,
        field_mask,
        feature_array.view_mut(),
        core_array,
        core_mask,
        nosfcecho,
        feat_val,
        bkgd_val,
        weakecho,
        core,
        mindbzuse,
        weakechothres,
    );
    Ok(())
}

#[pyfunction(name = "_core_scalar_scheme_f64")]
pub fn py_core_scalar_scheme<'py>(
    py: Python<'py>,
    field: PyReadonlyArrayDyn<'py, f64>,
    field_mask: PyReadonlyArrayDyn<'py, bool>,
    field_bkg: PyReadonlyArrayDyn<'py, f64>,
    field_bkg_mask: PyReadonlyArrayDyn<'py, bool>,
    max_diff: f64,
    always_core_thres: f64,
    core: f64,
    use_addition: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let field = field.as_array();
    let field_mask = field_mask.as_array();
    let field_bkg = field_bkg.as_array();
    let field_bkg_mask = field_bkg_mask.as_array();
    validate_core_scalar_inputs(&field, &field_mask, &field_bkg, &field_bkg_mask)?;
    if !max_diff.is_finite() || !always_core_thres.is_finite() || !core.is_finite() {
        return Err(PyValueError::new_err(
            "max_diff, always_core_thres, and CORE must be finite",
        ));
    }

    let output = core_scalar_scheme_kernel(
        field.view(),
        field_mask.view(),
        field_bkg.view(),
        field_bkg_mask.view(),
        max_diff,
        always_core_thres,
        core,
        use_addition,
    );
    Ok(PyArrayDyn::from_owned_array(py, output))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        py_assign_to_class_scan_no_entropy,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_atwt2d, module)?)?;
    module.add_function(wrap_pyfunction!(py_create_radial_mask_circular, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_freq_band, module)?)?;
    module.add_function(wrap_pyfunction!(py_standardize_linear, module)?)?;
    module.add_function(wrap_pyfunction!(py_assign_feature_radius_km, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_echo_class_wt_label_classes_f64,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_classify_feature_array_f64, module)?)?;
    module.add_function(wrap_pyfunction!(py_core_scalar_scheme, module)?)?;
    Ok(())
}

fn get_freq_band_kernel(freq: f64) -> Option<&'static str> {
    if freq >= 2.0e9 && freq < 4.0e9 {
        return Some("S");
    }
    if freq >= 4.0e9 && freq < 8.0e9 {
        return Some("C");
    }
    if freq >= 8.0e9 && freq <= 12.0e9 {
        return Some("X");
    }
    None
}

fn standardize_linear_kernel(data: ndarray::ArrayViewD<'_, f64>, mx: f64, mn: f64) -> ArrayD<f64> {
    let mut output = ArrayD::<f64>::zeros(data.raw_dim());
    for (slot, value) in output.iter_mut().zip(data.iter()) {
        let mut standardized = 2.0 * (*value - mn) / (mx - mn) - 1.0;
        if *value < mn {
            standardized = -1.0;
        } else if *value > mx {
            standardized = 1.0;
        }
        *slot = standardized;
    }
    output
}

fn assign_feature_radius_km_kernel(
    field_bkg: ndarray::ArrayViewD<'_, f64>,
    val_for_max_rad: f64,
    max_rad: f64,
) -> ArrayD<f64> {
    let mut output = ArrayD::<f64>::ones(field_bkg.raw_dim());
    for (slot, &value) in output.iter_mut().zip(field_bkg.iter()) {
        if value >= val_for_max_rad - 15.0 {
            *slot = max_rad - 3.0;
        }
        if value >= val_for_max_rad - 10.0 {
            *slot = max_rad - 2.0;
        }
        if value >= val_for_max_rad - 5.0 {
            *slot = max_rad - 1.0;
        }
        if value >= val_for_max_rad {
            *slot = max_rad;
        }
    }
    output
}

fn validate_wt_label_inputs(
    wt_sum: &ndarray::ArrayViewD<'_, f64>,
    dbz_data: &ndarray::ArrayViewD<'_, f64>,
) -> PyResult<()> {
    if wt_sum.shape() != dbz_data.shape() {
        return Err(PyValueError::new_err(
            "wt_sum and dbz_data must have the same shape",
        ));
    }
    if !wt_sum.is_standard_layout() || !dbz_data.is_standard_layout() {
        return Err(PyValueError::new_err(
            "wt_sum and dbz_data must be C-contiguous",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn echo_class_wt_label_classes_kernel(
    wt_sum: ndarray::ArrayViewD<'_, f64>,
    dbz_data: ndarray::ArrayViewD<'_, f64>,
    core_wt_threshold: f64,
    conv_wt_threshold: f64,
    min_reflectivity: f64,
    conv_min_refl: f64,
    conv_core_threshold: f64,
) -> ArrayD<f64> {
    let mut output = ArrayD::<f64>::zeros(wt_sum.raw_dim());
    ndarray::Zip::from(&mut output)
        .and(wt_sum)
        .and(dbz_data)
        .for_each(|slot, &wt_value, &dbz_value| {
            // The frozen oracle's second np.where uses else=0, so it resets
            // first-pass convective cores unless this second condition matches.
            let _first_pass_class =
                if wt_value >= conv_wt_threshold && dbz_value >= conv_core_threshold {
                    -3.0
                } else {
                    0.0
                };
            let mut wt_class = if wt_value >= core_wt_threshold && dbz_value >= conv_min_refl {
                -3.0
            } else {
                0.0
            };
            if wt_value < core_wt_threshold
                && wt_value >= conv_wt_threshold
                && dbz_value >= conv_min_refl
            {
                wt_class = -2.0;
            }
            if wt_class == 0.0 && dbz_value >= min_reflectivity {
                wt_class = -1.0;
            }

            let positive_class = -wt_class;
            *slot = if positive_class == 0.0 {
                f64::NAN
            } else {
                positive_class
            };
        });
    output
}

fn validate_classify_feature_inputs(
    field: &ndarray::ArrayViewD<'_, f64>,
    field_mask: &ndarray::ArrayViewD<'_, bool>,
    feature_array: &ndarray::ArrayViewMutD<'_, f64>,
    core_array: &ndarray::ArrayViewD<'_, f64>,
    core_mask: &ndarray::ArrayViewD<'_, bool>,
) -> PyResult<()> {
    if field.ndim() == 0 {
        return Err(PyValueError::new_err(
            "field must have at least one dimension",
        ));
    }
    if field.shape() != field_mask.shape()
        || field.shape() != feature_array.shape()
        || field.shape() != core_array.shape()
        || field.shape() != core_mask.shape()
    {
        return Err(PyValueError::new_err(
            "field, masks, feature_array, and core_array must have the same shape",
        ));
    }
    if !field.is_standard_layout()
        || !field_mask.is_standard_layout()
        || !feature_array.is_standard_layout()
        || !core_array.is_standard_layout()
        || !core_mask.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "field, masks, feature_array, and core_array must be C-contiguous",
        ));
    }
    Ok(())
}

fn validate_core_scalar_inputs(
    field: &ndarray::ArrayViewD<'_, f64>,
    field_mask: &ndarray::ArrayViewD<'_, bool>,
    field_bkg: &ndarray::ArrayViewD<'_, f64>,
    field_bkg_mask: &ndarray::ArrayViewD<'_, bool>,
) -> PyResult<()> {
    if field.ndim() == 0 {
        return Err(PyValueError::new_err(
            "field must have at least one dimension",
        ));
    }
    if field.shape() != field_mask.shape()
        || field.shape() != field_bkg.shape()
        || field.shape() != field_bkg_mask.shape()
    {
        return Err(PyValueError::new_err(
            "field, field_mask, field_bkg, and field_bkg_mask must have the same shape",
        ));
    }
    if !field.is_standard_layout()
        || !field_mask.is_standard_layout()
        || !field_bkg.is_standard_layout()
        || !field_bkg_mask.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "field, field_mask, field_bkg, and field_bkg_mask must be C-contiguous",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn classify_feature_array_kernel(
    field: ndarray::ArrayViewD<'_, f64>,
    field_mask: ndarray::ArrayViewD<'_, bool>,
    mut feature_array: ndarray::ArrayViewMutD<'_, f64>,
    core_array: ndarray::ArrayViewD<'_, f64>,
    core_mask: ndarray::ArrayViewD<'_, bool>,
    nosfcecho: f64,
    feat_val: f64,
    bkgd_val: f64,
    weakecho: f64,
    core: f64,
    mindbzuse: f64,
    weakechothres: f64,
) {
    ndarray::Zip::from(&mut feature_array)
        .and(field)
        .and(field_mask)
        .and(core_array)
        .and(core_mask)
        .for_each(
            |slot, &value, &is_field_masked, &core_value, &is_core_masked| {
                let mut classification = bkgd_val;
                if is_field_masked {
                    classification = nosfcecho;
                }
                if !is_core_masked && core_value == core {
                    classification = feat_val;
                }
                if value < weakechothres {
                    classification = weakecho;
                }
                if value < mindbzuse {
                    classification = nosfcecho;
                }
                *slot = classification;
            },
        );
}

#[allow(clippy::too_many_arguments)]
fn core_scalar_scheme_kernel(
    field: ndarray::ArrayViewD<'_, f64>,
    field_mask: ndarray::ArrayViewD<'_, bool>,
    field_bkg: ndarray::ArrayViewD<'_, f64>,
    field_bkg_mask: ndarray::ArrayViewD<'_, bool>,
    max_diff: f64,
    always_core_thres: f64,
    core: f64,
    use_addition: bool,
) -> ArrayD<f64> {
    let mut output = ArrayD::<f64>::zeros(field.raw_dim());
    ndarray::Zip::from(&mut output)
        .and(field)
        .and(field_mask)
        .and(field_bkg)
        .and(field_bkg_mask)
        .for_each(
            |slot, &field_value, &is_field_masked, &bkg_value, &is_bkg_masked| {
                if is_field_masked || is_bkg_masked {
                    *slot = 0.0;
                    return;
                }
                let mut zdiff = if use_addition {
                    (max_diff + bkg_value) - bkg_value
                } else {
                    (max_diff * bkg_value) - bkg_value
                };
                if zdiff < 0.0 || bkg_value < 0.0 {
                    zdiff = 0.0;
                }
                if field_value >= always_core_thres || (field_value - bkg_value) >= zdiff {
                    *slot = core;
                } else {
                    *slot = 0.0;
                }
            },
        );
    output
}

fn create_radial_mask_circular_kernel(
    mut mask_array: ndarray::ArrayViewMut2<'_, f64>,
    min_rad_km: f64,
    max_rad_km: f64,
    x_pixsize: f64,
    y_pixsize: f64,
    center_x: f64,
    center_y: f64,
) {
    let (xsize, ysize) = mask_array.dim();
    for j in 0..ysize {
        for i in 0..xsize {
            let x_range_sq = ((center_x - i as f64) * x_pixsize).powi(2);
            let y_range_sq = ((center_y - j as f64) * y_pixsize).powi(2);
            let circ_range = (x_range_sq + y_range_sq).sqrt();
            mask_array[[j, i]] = if circ_range <= max_rad_km && circ_range >= min_rad_km {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn atwt2d_kernel(
    mut data: ndarray::ArrayViewMut2<'_, f64>,
    max_scale: usize,
    ny: usize,
    nx: usize,
) -> Array3<f64> {
    let mut wt = Array3::<f64>::zeros((max_scale, ny, nx));
    let mut temp1 = Array2::<f64>::zeros((ny, nx));
    let mut temp2 = Array2::<f64>::zeros((ny, nx));
    let sf = (0.0625_f64, 0.25_f64, 0.375_f64);

    for scale in 1..=max_scale {
        let x1 = 2_usize.pow((scale - 1) as u32);
        let x2 = 2 * x1;

        for i in 0..nx {
            let prev2 = mirror_index(i, x2);
            let prev1 = mirror_index(i, x1);
            let next1 = forward_mirror_index(i, x1, nx);
            let next2 = forward_mirror_index(i, x2, nx);

            for j in 0..ny {
                temp1[[j, i]] = sf.0 * (data[[j, prev2]] + data[[j, next2]])
                    + sf.1 * (data[[j, prev1]] + data[[j, next1]])
                    + sf.2 * data[[j, i]];
            }
        }

        for i in 0..ny {
            let prev2 = mirror_index(i, x2);
            let prev1 = mirror_index(i, x1);
            let next1 = forward_mirror_index(i, x1, ny);
            let next2 = forward_mirror_index(i, x2, ny);

            for j in 0..nx {
                temp2[[i, j]] = sf.0 * (temp1[[prev2, j]] + temp1[[next2, j]])
                    + sf.1 * (temp1[[prev1, j]] + temp1[[next1, j]])
                    + sf.2 * temp1[[i, j]];
            }
        }

        for j in 0..ny {
            for i in 0..nx {
                wt[[scale - 1, j, i]] = data[[j, i]] - temp2[[j, i]];
                data[[j, i]] = temp2[[j, i]];
            }
        }
    }

    wt
}

fn validate_atwt_shape(ny: usize, nx: usize, max_scale: usize) -> PyResult<()> {
    if max_scale == 0 {
        return Ok(());
    }
    let min_dim = ny.min(nx);
    if min_dim == 0 {
        return Err(PyValueError::new_err(
            "_atwt2d requires non-empty dimensions when max_scale is positive",
        ));
    }
    let max_possible_scales = usize::BITS as usize - 1 - min_dim.leading_zeros() as usize;
    let max_supported = max_possible_scales.saturating_sub(1);
    if max_scale > max_supported {
        return Err(PyValueError::new_err(
            "max_scale exceeds supported ATWT scale for input shape",
        ));
    }
    Ok(())
}

fn mirror_index(index: usize, offset: usize) -> usize {
    index.abs_diff(offset)
}

fn forward_mirror_index(index: usize, offset: usize, len: usize) -> usize {
    let candidate = index + offset;
    if candidate > len - 1 {
        2 * (len - 1) - candidate
    } else {
        candidate
    }
}

fn assign_to_class_scan_no_entropy_kernel(
    data: ndarray::ArrayView3<'_, f64>,
    masks: ndarray::ArrayView3<'_, bool>,
    mass_centers: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
) -> Array2<u8> {
    let (nvariables, nrays, nbins) = data.dim();
    let nclasses = mass_centers.dim().0;
    let mut output = Array2::<u8>::zeros((nrays, nbins));

    for ray in 0..nrays {
        for bin in 0..nbins {
            if masks[[0, ray, bin]] {
                output[[ray, bin]] = 0;
                continue;
            }

            let mut best_class = 0usize;
            let mut best_distance =
                class_distance_squared(data, masks, mass_centers, weights, 0, ray, bin, nvariables);

            for class_idx in 1..nclasses {
                let distance = class_distance_squared(
                    data,
                    masks,
                    mass_centers,
                    weights,
                    class_idx,
                    ray,
                    bin,
                    nvariables,
                );
                if distance < best_distance {
                    best_distance = distance;
                    best_class = class_idx;
                }
            }

            output[[ray, bin]] = ((best_class + 1) % 256) as u8;
        }
    }

    output
}

fn class_distance_squared(
    data: ndarray::ArrayView3<'_, f64>,
    masks: ndarray::ArrayView3<'_, bool>,
    mass_centers: ndarray::ArrayView2<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    class_idx: usize,
    ray: usize,
    bin: usize,
    nvariables: usize,
) -> f64 {
    let mut total = 0.0;
    for variable in 0..nvariables {
        if masks[[variable, ray, bin]] {
            continue;
        }
        let diff = mass_centers[[class_idx, variable]] - data[[variable, ray, bin]];
        total += diff * diff * weights[variable];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn get_freq_band_matches_thresholds() {
        assert_eq!(get_freq_band_kernel(2.0e9), Some("S"));
        assert_eq!(get_freq_band_kernel(4.0e9), Some("C"));
        assert_eq!(get_freq_band_kernel(8.0e9), Some("X"));
        assert_eq!(get_freq_band_kernel(12.0e9), Some("X"));
        assert_eq!(get_freq_band_kernel(2.0e9 - 1.0), None);
        assert_eq!(get_freq_band_kernel(12.0e9 + 1.0), None);
        assert_eq!(get_freq_band_kernel(f64::NAN), None);
    }

    #[test]
    fn standardize_linear_clips_to_unit_interval() {
        let data = array![-20.0, -10.0, 0.0, 60.0, 70.0].into_dyn();
        let output = standardize_linear_kernel(data.view(), 60.0, -10.0);
        let expected = array![-1.0, -1.0, -0.7142857142857143, 1.0, 1.0].into_dyn();

        assert_eq!(output, expected);
    }

    #[test]
    fn assign_feature_radius_matches_step_thresholds() {
        let field_bkg = array![0.0, 5.0, 10.0, 15.0, 20.0].into_dyn();
        let output = assign_feature_radius_km_kernel(field_bkg.view(), 20.0, 5.0);
        let expected = array![1.0, 2.0, 3.0, 4.0, 5.0].into_dyn();

        assert_eq!(output, expected);
    }

    #[test]
    fn echo_class_wt_label_classes_matches_ordered_threshold_rules() {
        let wt_sum = array![[5.0, 3.0, 3.0, 1.0, 1.0, f64::NAN, f64::INFINITY]].into_dyn();
        let dbz_data = array![[30.0, 50.0, 30.0, 11.0, 5.0, 50.0, 30.0]].into_dyn();

        let output = echo_class_wt_label_classes_kernel(
            wt_sum.view(),
            dbz_data.view(),
            5.0,
            2.0,
            10.0,
            30.0,
            40.0,
        );

        assert_eq!(output[[0, 0]], 3.0);
        assert_eq!(output[[0, 1]], 2.0);
        assert_eq!(output[[0, 2]], 2.0);
        assert_eq!(output[[0, 3]], 1.0);
        assert!(output[[0, 4]].is_nan());
        assert_eq!(output[[0, 5]], 1.0);
        assert_eq!(output[[0, 6]], 3.0);
    }

    #[test]
    fn classify_feature_array_matches_ordered_python_assignments() {
        let field = array![[0.0, 10.0, 20.0], [30.0, 40.0, 4.0]].into_dyn();
        let field_mask = array![[false, false, false], [false, true, false]].into_dyn();
        let core_array = array![[0.0, 9.0, 0.0], [9.0, 9.0, 0.0]].into_dyn();
        let core_mask = array![[false, false, false], [false, false, false]].into_dyn();
        let mut feature = ArrayD::<f64>::zeros(field.raw_dim());
        classify_feature_array_kernel(
            field.view(),
            field_mask.view(),
            feature.view_mut(),
            core_array.view(),
            core_mask.view(),
            0.0,
            2.0,
            1.0,
            3.0,
            9.0,
            5.0,
            15.0,
        );
        let expected = array![[0.0, 3.0, 1.0], [2.0, 2.0, 0.0]].into_dyn();

        assert_eq!(feature, expected);
    }

    #[test]
    fn core_scalar_scheme_masks_background_before_always_core_threshold() {
        let field = array![30.0, 10.0, 30.0].into_dyn();
        let field_mask = array![false, false, true].into_dyn();
        let field_bkg = array![10.0, 10.0, 10.0].into_dyn();
        let field_bkg_mask = array![true, false, false].into_dyn();
        let output = core_scalar_scheme_kernel(
            field.view(),
            field_mask.view(),
            field_bkg.view(),
            field_bkg_mask.view(),
            2.0,
            25.0,
            9.0,
            false,
        );
        let expected = array![0.0, 0.0, 0.0].into_dyn();

        assert_eq!(output, expected);
    }

    #[test]
    fn classifies_with_masked_non_first_variables_ignored() {
        let data = array![[[0.1, 0.9], [0.2, 0.8]], [[0.1, 0.1], [0.9, 0.9]]];
        let masks = array![
            [[false, false], [false, false]],
            [[false, true], [true, false]]
        ];
        let mass_centers = array![[0.0, 0.0], [1.0, 1.0]];
        let weights = array![1.0, 1.0];

        let output = assign_to_class_scan_no_entropy_kernel(
            data.view(),
            masks.view(),
            mass_centers.view(),
            weights.view(),
        );

        assert_eq!(output, array![[1_u8, 2_u8], [1_u8, 2_u8]]);
    }

    #[test]
    fn first_variable_mask_sets_zero_class_value() {
        let data = array![[[0.2, 0.8]], [[0.2, 0.8]]];
        let masks = array![[[false, true]], [[false, false]]];
        let mass_centers = array![[0.0, 0.0], [1.0, 1.0]];
        let weights = array![1.0, 1.0];

        let output = assign_to_class_scan_no_entropy_kernel(
            data.view(),
            masks.view(),
            mass_centers.view(),
            weights.view(),
        );

        assert_eq!(output, array![[1_u8, 0_u8]]);
    }

    #[test]
    fn atwt_shape_validation_rejects_direct_unsafe_scales() {
        assert!(validate_atwt_shape(8, 8, 2).is_ok());
        assert!(validate_atwt_shape(4, 4, 1).is_ok());
        assert!(validate_atwt_shape(2, 4, 0).is_ok());
        assert!(validate_atwt_shape(2, 4, 1).is_err());
        assert!(validate_atwt_shape(0, 4, 1).is_err());
    }
}
