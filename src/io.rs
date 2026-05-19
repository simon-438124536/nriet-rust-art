use ndarray::{Array2, ArrayD, ArrayView1, ArrayViewMut1, ArrayViewMut2, IxDyn};
use numpy::{
    npyffi::NPY_ARRAY_WRITEABLE, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayDyn,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArray2,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::collections::BTreeMap;
use std::ptr;

const MDV_RLE8_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
const NEXRAD_AF1F_MAX_OUTPUT_BINS: usize = 512 * 1024 * 1024;
const NEXRAD_LEVEL2_SCAN_MSGS_MAX_RECORDS: usize = 1024 * 1024;
const KAZR_MAX_OUTPUT_VALUES: usize = 512 * 1024 * 1024;
const RAINBOW_WRL_MAX_OUTPUT_GATES: usize = 512 * 1024 * 1024;
const SIGMET_RECORD_WORDS: usize = 3_072;
const SIGMET_TIME_ORDER_MAX_RAYS: usize = 1024 * 1024;
const UF_RAY_MAP_MAX_RAYS: usize = 1024 * 1024;
const UF_SWEEP_LIMITS_MAX_RAYS: usize = 1024 * 1024;

#[pyfunction(name = "_fast_interpolate_scan_4")]
pub fn py_fast_interpolate_scan_4(
    mut data: PyReadwriteArray2<'_, f32>,
    mut scratch_ray: PyReadwriteArray1<'_, f32>,
    fill_value: f32,
    start: isize,
    end: isize,
    moment_ngates: usize,
    linear_interp: i32,
) -> PyResult<()> {
    let mut data = data.as_array_mut();
    let mut scratch_ray = scratch_ray.as_array_mut();
    interpolate_scan_4(
        data.view_mut(),
        scratch_ray.view_mut(),
        fill_value,
        start,
        end,
        moment_ngates,
        linear_interp != 0,
    )
}

#[pyfunction(name = "_fast_interpolate_scan_2")]
pub fn py_fast_interpolate_scan_2(
    mut data: PyReadwriteArray2<'_, f32>,
    mut scratch_ray: PyReadwriteArray1<'_, f32>,
    fill_value: f32,
    start: isize,
    end: isize,
    moment_ngates: usize,
    linear_interp: i32,
) -> PyResult<()> {
    let mut data = data.as_array_mut();
    let mut scratch_ray = scratch_ray.as_array_mut();
    interpolate_scan_2(
        data.view_mut(),
        scratch_ray.view_mut(),
        fill_value,
        start,
        end,
        moment_ngates,
        linear_interp != 0,
    )
}

#[pyfunction(name = "_mask_gates_not_collected")]
pub fn py_mask_gates_not_collected(
    mut mask: PyReadwriteArray2<'_, u8>,
    nbins: PyReadonlyArray1<'_, i64>,
) -> PyResult<()> {
    let mut mask = mask.as_array_mut();
    let nbins = nbins.as_array();
    mask_gates_not_collected(mask.view_mut(), &nbins)
}

#[pyfunction(name = "_uf_sweep_limits_i32")]
pub fn py_uf_sweep_limits_i32<'py>(
    py: Python<'py>,
    ray_sweep_numbers: PyReadonlyArray1<'py, i32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
    let ray_sweep_numbers = ray_sweep_numbers.as_array();
    if ray_sweep_numbers.len() > UF_SWEEP_LIMITS_MAX_RAYS {
        return Err(PyValueError::new_err(
            "ray_sweep_numbers exceeds native size limit",
        ));
    }
    if !ray_sweep_numbers.is_standard_layout() {
        return Err(PyValueError::new_err(
            "ray_sweep_numbers must be C-contiguous",
        ));
    }
    let ray_sweep_numbers = ray_sweep_numbers
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ray_sweep_numbers must be C-contiguous"))?;
    let (first, last) = uf_sweep_limits_i32(ray_sweep_numbers);
    Ok((PyArray1::from_vec(py, first), PyArray1::from_vec(py, last)))
}

#[pyfunction(name = "_uf_ray_num_to_sweep_num_i32")]
pub fn py_uf_ray_num_to_sweep_num_i32<'py>(
    py: Python<'py>,
    nrays: usize,
    starts: PyReadonlyArray1<'py, i32>,
    ends: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if nrays > UF_RAY_MAP_MAX_RAYS {
        return Err(PyValueError::new_err("nrays exceeds native size limit"));
    }
    let starts = starts.as_array();
    let ends = ends.as_array();
    if starts.len() > UF_RAY_MAP_MAX_RAYS {
        return Err(PyValueError::new_err(
            "sweep count exceeds native size limit",
        ));
    }
    if starts.len() != ends.len() {
        return Err(PyValueError::new_err("starts and ends lengths must match"));
    }
    if !starts.is_standard_layout() || !ends.is_standard_layout() {
        return Err(PyValueError::new_err(
            "starts and ends must be C-contiguous",
        ));
    }
    let starts = starts
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("starts must be C-contiguous"))?;
    let ends = ends
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ends must be C-contiguous"))?;
    Ok(PyArray1::from_vec(
        py,
        uf_ray_num_to_sweep_num_i32(nrays, starts, ends)?,
    ))
}

#[pyfunction(name = "_cfradial_unpack_variable_gate_dense")]
pub fn py_cfradial_unpack_variable_gate_dense(
    fdata: &Bound<'_, PyUntypedArray>,
    out_data: &Bound<'_, PyUntypedArray>,
    mut out_mask: PyReadwriteArray2<'_, bool>,
    ray_n_gates: PyReadonlyArray1<'_, i64>,
    ray_start_index: PyReadonlyArray1<'_, i64>,
) -> PyResult<()> {
    if fdata.ndim() != 1 {
        return Err(PyValueError::new_err("fdata must be one-dimensional"));
    }
    if out_data.ndim() != 2 {
        return Err(PyValueError::new_err("out_data must be two-dimensional"));
    }
    if !fdata.is_c_contiguous() || !out_data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "fdata and out_data must be C-contiguous",
        ));
    }
    if !untyped_array_is_writeable(out_data) {
        return Err(PyValueError::new_err("out_data must be writable"));
    }

    let f_dtype = fdata.dtype();
    let out_dtype = out_data.dtype();
    if f_dtype.num() != out_dtype.num()
        || f_dtype.itemsize() != out_dtype.itemsize()
        || f_dtype.byteorder() != out_dtype.byteorder()
    {
        return Err(PyValueError::new_err(
            "fdata and out_data must have identical dtype",
        ));
    }
    if !matches!(f_dtype.kind(), b'b' | b'i' | b'u' | b'f' | b'c') {
        return Err(PyValueError::new_err(
            "fdata dtype must be numeric for native CF/Radial unpacking",
        ));
    }
    let itemsize = f_dtype.itemsize();

    let source_len = fdata.shape()[0];
    let out_shape = out_data.shape();
    let (nrows, ncols) = (out_shape[0], out_shape[1]);
    let gates = ray_n_gates.as_array();
    let starts = ray_start_index.as_array();
    let mut mask = out_mask.as_array_mut();
    if !gates.is_standard_layout() || !starts.is_standard_layout() || !mask.is_standard_layout() {
        return Err(PyValueError::new_err(
            "ray metadata and out_mask must be C-contiguous",
        ));
    }
    if mask.dim() != (nrows, ncols) {
        return Err(PyValueError::new_err(
            "out_mask shape must match out_data shape",
        ));
    }

    let plan = cfradial_unpack_plan(source_len, nrows, ncols, &gates, &starts)?;
    let src_ptr = unsafe { (*fdata.as_array_ptr()).data as *const u8 };
    let dst_ptr = unsafe { (*out_data.as_array_ptr()).data as *mut u8 };
    cfradial_unpack_copy_bytes(src_ptr, dst_ptr, itemsize, ncols, &plan, &mut mask)
}

#[pyfunction(name = "_gamic_decode_uv8")]
pub fn py_gamic_decode_uv8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArrayDyn<'py, u8>,
    dyn_range_min: f64,
    dyn_range_max: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = gamic_decode_unsigned(&raw_data, dyn_range_min, dyn_range_max, 255.0)?;
    Ok((
        PyArrayDyn::from_owned_array(py, data),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_gamic_decode_uv16")]
pub fn py_gamic_decode_uv16<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArrayDyn<'py, u16>,
    dyn_range_min: f64,
    dyn_range_max: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = gamic_decode_unsigned(&raw_data, dyn_range_min, dyn_range_max, 65_535.0)?;
    Ok((
        PyArrayDyn::from_owned_array(py, data),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_gamic_decode_f32")]
pub fn py_gamic_decode_f32<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = gamic_decode_f32(&raw_data)?;
    Ok((
        PyArrayDyn::from_owned_array(py, data),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_odim_decode_u8")]
pub fn py_odim_decode_u8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArrayDyn<'py, u8>,
    has_nodata: bool,
    nodata: u8,
    has_undetect: bool,
    undetect: u8,
    gain: f64,
    offset: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = odim_decode_unsigned(
        &raw_data,
        has_nodata,
        nodata,
        has_undetect,
        undetect,
        gain,
        offset,
    )?;
    Ok((
        PyArrayDyn::from_owned_array(py, data),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_odim_decode_u16")]
pub fn py_odim_decode_u16<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArrayDyn<'py, u16>,
    has_nodata: bool,
    nodata: u16,
    has_undetect: bool,
    undetect: u16,
    gain: f64,
    offset: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = odim_decode_unsigned(
        &raw_data,
        has_nodata,
        nodata,
        has_undetect,
        undetect,
        gain,
        offset,
    )?;
    Ok((
        PyArrayDyn::from_owned_array(py, data),
        PyArrayDyn::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_cdm_moment_u8")]
pub fn py_nexrad_cdm_moment_u8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u8>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = nexrad_cdm_moment_unsigned(&raw_data, scale, add_offset)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_cdm_moment_u16")]
pub fn py_nexrad_cdm_moment_u16<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u16>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = nexrad_cdm_moment_unsigned(&raw_data, scale, add_offset)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_cdm_moment_i8")]
pub fn py_nexrad_cdm_moment_i8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, i8>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = nexrad_cdm_moment_numeric(&raw_data, scale, add_offset)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_cdm_moment_i16")]
pub fn py_nexrad_cdm_moment_i16<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, i16>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = nexrad_cdm_moment_numeric(&raw_data, scale, add_offset)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_cdm_moment_f32")]
pub fn py_nexrad_cdm_moment_f32<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, f32>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = nexrad_cdm_moment_numeric(&raw_data, scale, add_offset)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_cdm_moment_f64")]
pub fn py_nexrad_cdm_moment_f64<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, f64>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    let (data, mask) = nexrad_cdm_moment_numeric(&raw_data, scale, add_offset)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_chl_extract_integer_fields")]
pub fn py_chl_extract_integer_fields<'py>(
    py: Python<'py>,
    raw_data: &[u8],
    ngates: usize,
    field_nums: Vec<i64>,
    formats: Vec<i32>,
    dat_factors: Vec<f64>,
    dat_biases: Vec<f64>,
    fld_factors: Vec<f64>,
) -> PyResult<Bound<'py, PyList>> {
    let fields = chl_extract_integer_fields(
        raw_data,
        ngates,
        &field_nums,
        &formats,
        &dat_factors,
        &dat_biases,
        &fld_factors,
    )?;
    let out = PyList::empty(py);
    for field in fields {
        out.append((
            field.field_num,
            PyArray2::from_owned_array(py, field.data),
            PyArray2::from_owned_array(py, field.mask),
            field.format_code,
        ))?;
    }
    Ok(out)
}

#[pyfunction(name = "_rainbow_wrl_get_data_u8")]
pub fn py_rainbow_wrl_get_data_u8<'py>(
    py: Python<'py>,
    databin: PyReadonlyArrayDyn<'py, u8>,
    nrays: usize,
    nbins: usize,
    maxbin: usize,
    datamin: f64,
    scale: f64,
    fill_value: f64,
    wrap_phidp: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let databin = databin.as_array();
    let (data, mask) = rainbow_wrl_get_data(
        &databin, nrays, nbins, maxbin, datamin, scale, fill_value, wrap_phidp,
    )?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_rainbow_wrl_get_data_u16")]
pub fn py_rainbow_wrl_get_data_u16<'py>(
    py: Python<'py>,
    databin: PyReadonlyArrayDyn<'py, u16>,
    nrays: usize,
    nbins: usize,
    maxbin: usize,
    datamin: f64,
    scale: f64,
    fill_value: f64,
    wrap_phidp: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let databin = databin.as_array();
    let (data, mask) = rainbow_wrl_get_data(
        &databin, nrays, nbins, maxbin, datamin, scale, fill_value, wrap_phidp,
    )?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_kazr_get_spectra")]
pub fn py_kazr_get_spectra<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArray2<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    missing: PyReadonlyArray1<'py, bool>,
    npulses: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let spectra = spectra.as_array();
    let indices = indices.as_array();
    let missing = missing.as_array();
    let out = kazr_get_spectra(&spectra, &indices, &missing, npulses)?;
    Ok(PyArray2::from_owned_array(py, out))
}

#[pyfunction(name = "_geotiff_rgb_values_f64")]
pub fn py_geotiff_rgb_values_f64<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    rgba_lut: PyReadonlyArray2<'py, f64>,
    vmin: f64,
    vmax: f64,
    color_levels: f64,
    transpbg: bool,
    op: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
    bool,
)> {
    let data = data.as_array();
    let rgba_lut = rgba_lut.as_array();
    let (r, g, b, a, has_nan) =
        geotiff_rgb_values_f64(&data, &rgba_lut, vmin, vmax, color_levels, transpbg, op)?;
    Ok((
        PyArray2::from_owned_array(py, r),
        PyArray2::from_owned_array(py, g),
        PyArray2::from_owned_array(py, b),
        PyArray2::from_owned_array(py, a),
        has_nan,
    ))
}

#[pyfunction(name = "_edge_mask_u8")]
pub fn py_edge_mask_u8<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, u8>,
    existing_mask: PyReadonlyArrayDyn<'py, bool>,
    has_missing: bool,
    missing: u8,
    has_folded: bool,
    folded: u8,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let mask = edge_mask(
        &data.as_array(),
        &existing_mask.as_array(),
        has_missing,
        missing,
        has_folded,
        folded,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_edge_mask_u16")]
pub fn py_edge_mask_u16<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, u16>,
    existing_mask: PyReadonlyArrayDyn<'py, bool>,
    has_missing: bool,
    missing: u16,
    has_folded: bool,
    folded: u16,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let mask = edge_mask(
        &data.as_array(),
        &existing_mask.as_array(),
        has_missing,
        missing,
        has_folded,
        folded,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_edge_mask_i16")]
pub fn py_edge_mask_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, i16>,
    existing_mask: PyReadonlyArrayDyn<'py, bool>,
    has_missing: bool,
    missing: i16,
    has_folded: bool,
    folded: i16,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let mask = edge_mask(
        &data.as_array(),
        &existing_mask.as_array(),
        has_missing,
        missing,
        has_folded,
        folded,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_edge_mask_i32")]
pub fn py_edge_mask_i32<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, i32>,
    existing_mask: PyReadonlyArrayDyn<'py, bool>,
    has_missing: bool,
    missing: i32,
    has_folded: bool,
    folded: i32,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let mask = edge_mask(
        &data.as_array(),
        &existing_mask.as_array(),
        has_missing,
        missing,
        has_folded,
        folded,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_edge_mask_f32")]
pub fn py_edge_mask_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f32>,
    existing_mask: PyReadonlyArrayDyn<'py, bool>,
    has_missing: bool,
    missing: f32,
    has_folded: bool,
    folded: f32,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let mask = edge_mask(
        &data.as_array(),
        &existing_mask.as_array(),
        has_missing,
        missing,
        has_folded,
        folded,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_edge_mask_f64")]
pub fn py_edge_mask_f64<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    existing_mask: PyReadonlyArrayDyn<'py, bool>,
    has_missing: bool,
    missing: f64,
    has_folded: bool,
    folded: f64,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let mask = edge_mask(
        &data.as_array(),
        &existing_mask.as_array(),
        has_missing,
        missing,
        has_folded,
        folded,
    )?;
    Ok(PyArrayDyn::from_owned_array(py, mask))
}

#[pyfunction(name = "_nexrad_level3_int16_to_float16")]
pub fn py_nexrad_level3_int16_to_float16(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if value.get_type().name()?.to_str()? != "int" {
        return Err(PyTypeError::new_err("value must be a Python int"));
    }
    Ok(nexrad_level3_int16_to_float16(value.extract()?))
}

#[pyfunction(name = "_nexrad_level2_scan_msgs_i64")]
pub fn py_nexrad_level2_scan_msgs_i64<'py>(
    py: Python<'py>,
    elevation_numbers: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyList>> {
    let elevation_numbers = elevation_numbers.as_array();
    if elevation_numbers.len() > NEXRAD_LEVEL2_SCAN_MSGS_MAX_RECORDS {
        return Err(PyValueError::new_err(
            "elevation_numbers exceeds native size limit",
        ));
    }
    if !elevation_numbers.is_standard_layout() {
        return Err(PyValueError::new_err(
            "elevation_numbers must be C-contiguous",
        ));
    }
    let elevation_numbers = elevation_numbers
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("elevation_numbers must be C-contiguous"))?;
    let scan_msgs = nexrad_level2_scan_msgs_i64(elevation_numbers)?;
    let out = PyList::empty(py);
    for scan_msg in scan_msgs {
        out.append(PyArray1::from_vec(py, scan_msg))?;
    }
    Ok(out)
}

#[pyfunction(name = "_nexrad_level2_msg_nums_i64")]
pub fn py_nexrad_level2_msg_nums_i64<'py>(
    py: Python<'py>,
    scan_msgs: Vec<PyReadonlyArray1<'py, i64>>,
    scans: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let scans = scans.as_array();
    if scans.is_empty() {
        return Err(PyValueError::new_err("scans must be non-empty"));
    }
    if scans.len() > NEXRAD_LEVEL2_SCAN_MSGS_MAX_RECORDS {
        return Err(PyValueError::new_err("scans exceeds native size limit"));
    }
    if !scans.is_standard_layout() {
        return Err(PyValueError::new_err("scans must be C-contiguous"));
    }
    let scans = scans
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("scans must be C-contiguous"))?;

    let mut scan_msg_values = Vec::with_capacity(scan_msgs.len());
    for scan_msg in &scan_msgs {
        let scan_msg = scan_msg.as_array();
        if !scan_msg.is_standard_layout() {
            return Err(PyValueError::new_err("scan_msgs must be C-contiguous"));
        }
        scan_msg_values.push(
            scan_msg
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("scan_msgs must be C-contiguous"))?
                .to_vec(),
        );
    }

    Ok(PyArray1::from_vec(
        py,
        nexrad_level2_msg_nums_i64(&scan_msg_values, scans)?,
    ))
}

#[pyfunction(name = "_sigmet_bin2_to_angle_u16")]
pub fn py_sigmet_bin2_to_angle_u16<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, u16>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let values = values.as_array();
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }

    let output = values.mapv(sigmet_bin2_to_angle);
    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_sigmet_bin4_to_angle_u32")]
pub fn py_sigmet_bin4_to_angle_u32<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, u32>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let values = values.as_array();
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }

    let output = values.mapv(sigmet_bin4_to_angle);
    Ok(PyArrayDyn::from_owned_array(py, output))
}

#[pyfunction(name = "_sigmet_parse_ray_headers_i16")]
pub fn py_sigmet_parse_ray_headers_i16<'py>(
    py: Python<'py>,
    ray_headers: PyReadonlyArrayDyn<'py, i16>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<u16>>,
    Bound<'py, PyArrayDyn<i16>>,
)> {
    let ray_headers = ray_headers.as_array();
    if !ray_headers.is_standard_layout() {
        return Err(PyValueError::new_err("ray_headers must be C-contiguous"));
    }
    let ndim = ray_headers.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err(
            "ray_headers must have at least 2 dimensions",
        ));
    }
    let shape = ray_headers.shape();
    if shape[ndim - 1] != 6 {
        return Err(PyValueError::new_err(
            "ray_headers last dimension must have length 6",
        ));
    }

    let out_shape = IxDyn(&shape[..ndim - 1]);
    let nrays = ray_headers.len() / 6;
    let mut az0 = Vec::with_capacity(nrays);
    let mut el0 = Vec::with_capacity(nrays);
    let mut az1 = Vec::with_capacity(nrays);
    let mut el1 = Vec::with_capacity(nrays);
    let mut nbins = Vec::with_capacity(nrays);
    let mut time = Vec::with_capacity(nrays);
    let mut prf_flag = Vec::with_capacity(nrays);

    for header in ray_headers
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ray_headers must be C-contiguous"))?
        .chunks_exact(6)
    {
        az0.push(sigmet_bin2_to_angle(header[0] as u16));
        el0.push(sigmet_bin2_to_angle(header[1] as u16));
        az1.push(sigmet_bin2_to_angle(header[2] as u16));
        el1.push(sigmet_bin2_to_angle(header[3] as u16));
        nbins.push(header[4]);
        time.push(header[5] as u16);
        prf_flag.push(header[0].rem_euclid(2));
    }

    Ok((
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape.clone(), az0)?),
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape.clone(), el0)?),
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape.clone(), az1)?),
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape.clone(), el1)?),
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape.clone(), nbins)?),
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape.clone(), time)?),
        PyArrayDyn::from_owned_array(py, array_from_shape_vec(out_shape, prf_flag)?),
    ))
}

#[pyfunction(name = "_sigmet_time_ordered_by_reversal_i32")]
pub fn py_sigmet_time_ordered_by_reversal_i32(
    ref_time: PyReadonlyArray1<'_, i32>,
    rays_per_sweep: PyReadonlyArray1<'_, i64>,
) -> PyResult<bool> {
    let ref_time = ref_time.as_array();
    let rays_per_sweep = rays_per_sweep.as_array();
    let ref_time = sigmet_time_order_ref_time_slice(&ref_time)?;
    let rays_per_sweep = sigmet_time_order_rays_slice(&rays_per_sweep)?;
    sigmet_time_ordered_by_reversal_i32(ref_time, rays_per_sweep)
}

#[pyfunction(name = "_sigmet_time_ordered_by_roll_i32")]
pub fn py_sigmet_time_ordered_by_roll_i32(
    ref_time: PyReadonlyArray1<'_, i32>,
    rays_per_sweep: PyReadonlyArray1<'_, i64>,
) -> PyResult<bool> {
    let ref_time = ref_time.as_array();
    let rays_per_sweep = rays_per_sweep.as_array();
    let ref_time = sigmet_time_order_ref_time_slice(&ref_time)?;
    let rays_per_sweep = sigmet_time_order_rays_slice(&rays_per_sweep)?;
    sigmet_time_ordered_by_roll_i32(ref_time, rays_per_sweep)
}

#[pyfunction(name = "_sigmet_time_ordered_by_reverse_roll_i32")]
pub fn py_sigmet_time_ordered_by_reverse_roll_i32(
    ref_time: PyReadonlyArray1<'_, i32>,
    rays_per_sweep: PyReadonlyArray1<'_, i64>,
) -> PyResult<bool> {
    let ref_time = ref_time.as_array();
    let rays_per_sweep = rays_per_sweep.as_array();
    let ref_time = sigmet_time_order_ref_time_slice(&ref_time)?;
    let rays_per_sweep = sigmet_time_order_rays_slice(&rays_per_sweep)?;
    sigmet_time_ordered_by_reverse_roll_i32(ref_time, rays_per_sweep)
}

#[pyfunction(name = "_sigmet_time_order_roll_index_i32")]
pub fn py_sigmet_time_order_roll_index_i32<'py>(
    py: Python<'py>,
    ref_time: PyReadonlyArray1<'py, i32>,
    rays_per_sweep: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ref_time = ref_time.as_array();
    let rays_per_sweep = rays_per_sweep.as_array();
    let ref_time = sigmet_time_order_ref_time_slice(&ref_time)?;
    let rays_per_sweep = sigmet_time_order_rays_slice(&rays_per_sweep)?;
    Ok(PyArray1::from_vec(
        py,
        sigmet_time_order_roll_index_i32(ref_time, rays_per_sweep)?,
    ))
}

#[pyfunction(name = "_sigmet_time_order_reverse_index_i32")]
pub fn py_sigmet_time_order_reverse_index_i32<'py>(
    py: Python<'py>,
    ref_time: PyReadonlyArray1<'py, i32>,
    rays_per_sweep: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ref_time = ref_time.as_array();
    let rays_per_sweep = rays_per_sweep.as_array();
    let ref_time = sigmet_time_order_ref_time_slice(&ref_time)?;
    let rays_per_sweep = sigmet_time_order_rays_slice(&rays_per_sweep)?;
    Ok(PyArray1::from_vec(
        py,
        sigmet_time_order_reverse_index_i32(ref_time, rays_per_sweep)?,
    ))
}

#[pyfunction(name = "_sigmet_time_order_full_index_i32")]
pub fn py_sigmet_time_order_full_index_i32<'py>(
    py: Python<'py>,
    ref_time: PyReadonlyArray1<'py, i32>,
    rays_per_sweep: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ref_time = ref_time.as_array();
    let rays_per_sweep = rays_per_sweep.as_array();
    let ref_time = sigmet_time_order_ref_time_slice(&ref_time)?;
    let rays_per_sweep = sigmet_time_order_rays_slice(&rays_per_sweep)?;
    Ok(PyArray1::from_vec(
        py,
        sigmet_time_order_full_index_i32(ref_time, rays_per_sweep)?,
    ))
}

#[pyfunction(name = "_sigmet_data_types_from_mask_u32")]
pub fn py_sigmet_data_types_from_mask_u32(
    word0: &Bound<'_, PyAny>,
    word1: &Bound<'_, PyAny>,
    word2: &Bound<'_, PyAny>,
    word3: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    let words = [
        extract_sigmet_mask_word(word0)?,
        extract_sigmet_mask_word(word1)?,
        extract_sigmet_mask_word(word2)?,
        extract_sigmet_mask_word(word3)?,
    ];
    Ok(sigmet_data_types_from_mask(words))
}

#[pyfunction(name = "_sigmet_decode_ray_current_record_i16")]
pub fn py_sigmet_decode_ray_current_record_i16(
    rbuf: PyReadonlyArray1<'_, i16>,
    rbuf_pos: isize,
    nbins: usize,
    mut out: PyReadwriteArray1<'_, i16>,
) -> PyResult<Option<(i32, isize)>> {
    let rbuf = rbuf.as_array();
    let mut out = out.as_array_mut();
    sigmet_decode_ray_current_record_i16(&rbuf, rbuf_pos, nbins, &mut out)
}

#[pyfunction(name = "_sigmet_convert_like_dbt2_dense_i16")]
pub fn py_sigmet_convert_like_dbt2_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_like_dbt2_dense_i16(&data, &nbins)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_like_dbt_dense_i16")]
pub fn py_sigmet_convert_like_dbt_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_like_dbt_dense_i16(&data, &nbins)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_like_sqi_dense_i16")]
pub fn py_sigmet_convert_like_sqi_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::LikeSqi)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_vel_dense_i16")]
pub fn py_sigmet_convert_vel_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::Vel)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_velc_dense_i16")]
pub fn py_sigmet_convert_velc_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::VelC)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_width_dense_i16")]
pub fn py_sigmet_convert_width_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::Width)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_zdr_dense_i16")]
pub fn py_sigmet_convert_zdr_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::Zdr)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_kdp_dense_i16")]
pub fn py_sigmet_convert_kdp_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::Kdp)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_phidp_dense_i16")]
pub fn py_sigmet_convert_phidp_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::PhiDp)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_hclass_dense_i16")]
pub fn py_sigmet_convert_hclass_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u8_dense_i16(&data, &nbins, SigmetU8Mode::HClass)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_like_sqi2_dense_i16")]
pub fn py_sigmet_convert_like_sqi2_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u16_dense_i16(&data, &nbins, SigmetU16Mode::LikeSqi2)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_width2_dense_i16")]
pub fn py_sigmet_convert_width2_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u16_dense_i16(&data, &nbins, SigmetU16Mode::Width2)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_phidp2_dense_i16")]
pub fn py_sigmet_convert_phidp2_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u16_dense_i16(&data, &nbins, SigmetU16Mode::PhiDp2)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_sigmet_convert_hclass2_dense_i16")]
pub fn py_sigmet_convert_hclass2_dense_i16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i16>,
    nbins: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let data = data.as_array();
    let nbins = nbins.as_array();
    let (out, mask) = sigmet_convert_u16_dense_i16(&data, &nbins, SigmetU16Mode::HClass2)?;
    Ok((
        PyArray2::from_owned_array(py, out),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_mdv_decode_rle8")]
pub fn py_mdv_decode_rle8<'py>(
    py: Python<'py>,
    compr_data: &[u8],
    key: &Bound<'_, PyAny>,
    decompr_size: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let key = extract_mdv_key(key)?;
    let decompr_size = extract_mdv_decompr_size(decompr_size)?;
    let output = mdv_decode_rle8_exact(compr_data, key, decompr_size)?;
    Ok(PyBytes::new(py, &output))
}

#[pyfunction(name = "_nexrad_af1f_decode_rle_u8")]
pub fn py_nexrad_af1f_decode_rle_u8<'py>(
    py: Python<'py>,
    rle_data: &[u8],
    nbins: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let nbins = extract_nexrad_af1f_nbins(nbins)?;
    let output = nexrad_af1f_decode_rle_exact(rle_data, nbins)?;
    Ok(PyArray1::from_vec(py, output))
}

#[pyfunction(name = "_nexrad_level3_data_8_or_16_u8")]
pub fn py_nexrad_level3_data_8_or_16_u8<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_data_8_or_16(threshold_data, &raw_data)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_msg_134_u8")]
pub fn py_nexrad_level3_msg_134_u8<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_msg_134(threshold_data, &raw_data)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_msg_135_u8")]
pub fn py_nexrad_level3_msg_135_u8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_msg_135(&raw_data);
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_msg_138_u8")]
pub fn py_nexrad_level3_msg_138_u8<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let data = nexrad_level3_msg_138(threshold_data, &raw_data)?;
    Ok(PyArray2::from_owned_array(py, data))
}

#[pyfunction(name = "_nexrad_level3_msg_32_u8")]
pub fn py_nexrad_level3_msg_32_u8<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_msg_scaled_u8(threshold_data, &raw_data, false)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_msg_32_u16")]
pub fn py_nexrad_level3_msg_32_u16<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u16>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_msg_scaled_u16(threshold_data, &raw_data, false)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_msg_scaled_sub2_u8")]
pub fn py_nexrad_level3_msg_scaled_sub2_u8<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_msg_scaled_u8(threshold_data, &raw_data, true)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_msg_scaled_sub2_u16")]
pub fn py_nexrad_level3_msg_scaled_sub2_u16<'py>(
    py: Python<'py>,
    threshold_data: &[u8],
    raw_data: PyReadonlyArray2<'py, u16>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_msg_scaled_u16(threshold_data, &raw_data, true)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_mask_zero_u8")]
pub fn py_nexrad_level3_mask_zero_u8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_mask_zero_u8(&raw_data);
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_mask_zero_u16")]
pub fn py_nexrad_level3_mask_zero_u16<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u16>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>)> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let (data, mask) = nexrad_level3_mask_zero_u16(&raw_data);
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

#[pyfunction(name = "_nexrad_level3_copy_u8")]
pub fn py_nexrad_level3_copy_u8<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    Ok(PyArray2::from_owned_array(
        py,
        nexrad_level3_copy_u8(&raw_data),
    ))
}

#[pyfunction(name = "_nexrad_level3_copy_u16")]
pub fn py_nexrad_level3_copy_u16<'py>(
    py: Python<'py>,
    raw_data: PyReadonlyArray2<'py, u16>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let raw_data = raw_data.as_array();
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    Ok(PyArray2::from_owned_array(
        py,
        nexrad_level3_copy_u16(&raw_data),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_fast_interpolate_scan_4, module)?)?;
    module.add_function(wrap_pyfunction!(py_fast_interpolate_scan_2, module)?)?;
    module.add_function(wrap_pyfunction!(py_mask_gates_not_collected, module)?)?;
    module.add_function(wrap_pyfunction!(py_uf_sweep_limits_i32, module)?)?;
    module.add_function(wrap_pyfunction!(py_uf_ray_num_to_sweep_num_i32, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_cfradial_unpack_variable_gate_dense,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_gamic_decode_uv8, module)?)?;
    module.add_function(wrap_pyfunction!(py_gamic_decode_uv16, module)?)?;
    module.add_function(wrap_pyfunction!(py_gamic_decode_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_odim_decode_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_odim_decode_u16, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_cdm_moment_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_cdm_moment_u16, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_cdm_moment_i8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_cdm_moment_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_cdm_moment_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_cdm_moment_f64, module)?)?;
    module.add_function(wrap_pyfunction!(py_chl_extract_integer_fields, module)?)?;
    module.add_function(wrap_pyfunction!(py_rainbow_wrl_get_data_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_rainbow_wrl_get_data_u16, module)?)?;
    module.add_function(wrap_pyfunction!(py_kazr_get_spectra, module)?)?;
    module.add_function(wrap_pyfunction!(py_geotiff_rgb_values_f64, module)?)?;
    module.add_function(wrap_pyfunction!(py_edge_mask_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_edge_mask_u16, module)?)?;
    module.add_function(wrap_pyfunction!(py_edge_mask_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_edge_mask_i32, module)?)?;
    module.add_function(wrap_pyfunction!(py_edge_mask_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_edge_mask_f64, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_int16_to_float16, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level2_scan_msgs_i64, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level2_msg_nums_i64, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_bin2_to_angle_u16, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_bin4_to_angle_u32, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_parse_ray_headers_i16, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_time_ordered_by_reversal_i32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_time_ordered_by_roll_i32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_time_ordered_by_reverse_roll_i32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_time_order_roll_index_i32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_time_order_reverse_index_i32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_time_order_full_index_i32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_data_types_from_mask_u32,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_decode_ray_current_record_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_like_dbt2_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_like_dbt_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_like_sqi_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_convert_vel_dense_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_convert_velc_dense_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_convert_width_dense_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_convert_zdr_dense_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_convert_kdp_dense_i16, module)?)?;
    module.add_function(wrap_pyfunction!(py_sigmet_convert_phidp_dense_i16, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_hclass_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_like_sqi2_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_width2_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_phidp2_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sigmet_convert_hclass2_dense_i16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_mdv_decode_rle8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_af1f_decode_rle_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_data_8_or_16_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_msg_134_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_msg_135_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_msg_138_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_msg_32_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_msg_32_u16, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_nexrad_level3_msg_scaled_sub2_u8,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_nexrad_level3_msg_scaled_sub2_u16,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_mask_zero_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_mask_zero_u16, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_copy_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_nexrad_level3_copy_u16, module)?)?;
    Ok(())
}

fn nexrad_level3_int16_to_float16(value: i64) -> f64 {
    let raw = (value as u64) & 0xffff;
    let sign = ((raw & 0x8000) >> 15) as u32;
    let exponent = ((raw & 0x7c00) >> 10) as i32;
    let fraction = (raw & 0x03ff) as f64;
    let sign_multiplier = if sign == 0 { 1.0 } else { -1.0 };

    if exponent == 0 {
        sign_multiplier * 2.0 * (fraction / 2.0_f64.powi(10))
    } else {
        sign_multiplier * 2.0_f64.powi(exponent - 16) * (1.0 + fraction / 2.0_f64.powi(10))
    }
}

fn untyped_array_is_writeable(array: &Bound<'_, PyUntypedArray>) -> bool {
    unsafe { (*array.as_array_ptr()).flags & NPY_ARRAY_WRITEABLE != 0 }
}

fn cfradial_unpack_plan(
    source_len: usize,
    nrows: usize,
    ncols: usize,
    ray_n_gates: &ndarray::ArrayView1<'_, i64>,
    ray_start_index: &ndarray::ArrayView1<'_, i64>,
) -> PyResult<Vec<(usize, usize)>> {
    if ray_n_gates.len() != nrows || ray_start_index.len() != nrows {
        return Err(PyValueError::new_err(
            "ray metadata length must match output rows",
        ));
    }

    let mut plan = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let gates = usize::try_from(ray_n_gates[row])
            .map_err(|_| PyValueError::new_err("ray_n_gates values must be nonnegative"))?;
        let start = usize::try_from(ray_start_index[row])
            .map_err(|_| PyValueError::new_err("ray_start_index values must be nonnegative"))?;
        if gates > ncols {
            return Err(PyValueError::new_err(
                "ray_n_gates values must fit within output columns",
            ));
        }
        let end = start
            .checked_add(gates)
            .ok_or_else(|| PyValueError::new_err("ray_start_index + ray_n_gates overflows"))?;
        if start > source_len || end > source_len {
            return Err(PyValueError::new_err(
                "ray source slice exceeds fdata length",
            ));
        }
        plan.push((start, gates));
    }
    Ok(plan)
}

fn cfradial_unpack_copy_bytes(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    itemsize: usize,
    ncols: usize,
    plan: &[(usize, usize)],
    out_mask: &mut ArrayViewMut2<'_, bool>,
) -> PyResult<()> {
    for (row, &(start, gates)) in plan.iter().enumerate() {
        let bytes = gates
            .checked_mul(itemsize)
            .ok_or_else(|| PyValueError::new_err("CF/Radial byte copy size overflows"))?;
        if bytes != 0 {
            let src_offset = start
                .checked_mul(itemsize)
                .ok_or_else(|| PyValueError::new_err("CF/Radial source offset overflows"))?;
            let dst_elem = row
                .checked_mul(ncols)
                .ok_or_else(|| PyValueError::new_err("CF/Radial output offset overflows"))?;
            let dst_offset = dst_elem
                .checked_mul(itemsize)
                .ok_or_else(|| PyValueError::new_err("CF/Radial output offset overflows"))?;
            unsafe {
                ptr::copy_nonoverlapping(src_ptr.add(src_offset), dst_ptr.add(dst_offset), bytes);
            }
        }
        for col in 0..gates {
            out_mask[[row, col]] = false;
        }
    }
    Ok(())
}

fn gamic_decode_unsigned<T>(
    raw_data: &ndarray::ArrayViewD<'_, T>,
    dyn_range_min: f64,
    dyn_range_max: f64,
    divisor: f64,
) -> PyResult<(ArrayD<f32>, ArrayD<bool>)>
where
    T: Copy + Into<f64> + PartialEq + From<u8>,
{
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    if !dyn_range_min.is_finite() || !dyn_range_max.is_finite() {
        return Err(PyValueError::new_err("dynamic range values must be finite"));
    }
    let scale = (dyn_range_max - dyn_range_min) / divisor;
    let offset = dyn_range_min;
    let zero = T::from(0_u8);
    let shape = IxDyn(raw_data.shape());
    let mut data = ArrayD::<f32>::zeros(shape.clone());
    let mut mask = ArrayD::<bool>::from_elem(shape, false);
    for ((out, mask_out), value) in data.iter_mut().zip(mask.iter_mut()).zip(raw_data.iter()) {
        *out = (((*value).into() * scale) + offset) as f32;
        *mask_out = *value == zero;
    }
    Ok((data, mask))
}

fn gamic_decode_f32(
    raw_data: &ndarray::ArrayViewD<'_, f32>,
) -> PyResult<(ArrayD<f32>, ArrayD<bool>)> {
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    let data = raw_data.to_owned();
    let mask = raw_data.mapv(f32::is_nan);
    Ok((data, mask))
}

fn odim_decode_unsigned<T>(
    raw_data: &ndarray::ArrayViewD<'_, T>,
    has_nodata: bool,
    nodata: T,
    has_undetect: bool,
    undetect: T,
    gain: f64,
    offset: f64,
) -> PyResult<(ArrayD<f64>, ArrayD<bool>)>
where
    T: Copy + Into<f64> + PartialEq,
{
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    if !gain.is_finite() || !offset.is_finite() {
        return Err(PyValueError::new_err("gain and offset must be finite"));
    }
    let shape = IxDyn(raw_data.shape());
    let mut data = ArrayD::<f64>::zeros(shape.clone());
    let mut mask = ArrayD::<bool>::from_elem(shape, false);
    for ((out, mask_out), value) in data.iter_mut().zip(mask.iter_mut()).zip(raw_data.iter()) {
        let masked = (has_nodata && *value == nodata) || (has_undetect && *value == undetect);
        *mask_out = masked;
        *out = if masked {
            (*value).into()
        } else {
            (*value).into() * gain + offset
        };
    }
    Ok((data, mask))
}

fn nexrad_cdm_moment_unsigned<T>(
    raw_data: &ndarray::ArrayView2<'_, T>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Array2<f64>, Array2<bool>)>
where
    T: Copy + Into<f64> + PartialOrd + From<u8>,
{
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    if !scale.is_finite() || !add_offset.is_finite() {
        return Err(PyValueError::new_err("scale and add_offset must be finite"));
    }
    let shape = raw_data.dim();
    let mut data = Array2::<f64>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    let threshold = T::from(1_u8);
    for ((out, mask_out), value) in data.iter_mut().zip(mask.iter_mut()).zip(raw_data.iter()) {
        let masked = *value <= threshold;
        *mask_out = masked;
        *out = if masked {
            (*value).into()
        } else {
            (*value).into() * scale + add_offset
        };
    }
    Ok((data, mask))
}

fn nexrad_cdm_moment_numeric<T>(
    raw_data: &ndarray::ArrayView2<'_, T>,
    scale: f64,
    add_offset: f64,
) -> PyResult<(Array2<f64>, Array2<bool>)>
where
    T: Copy + Into<f64> + PartialOrd,
{
    if !raw_data.is_standard_layout() {
        return Err(PyValueError::new_err("raw_data must be C-contiguous"));
    }
    if !scale.is_finite() || !add_offset.is_finite() {
        return Err(PyValueError::new_err("scale and add_offset must be finite"));
    }
    let shape = raw_data.dim();
    let mut data = Array2::<f64>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((out, mask_out), value) in data.iter_mut().zip(mask.iter_mut()).zip(raw_data.iter()) {
        let raw = (*value).into();
        let masked = raw <= 1.0;
        *mask_out = masked;
        *out = if masked {
            raw
        } else {
            raw * scale + add_offset
        };
    }
    Ok((data, mask))
}

struct ChlIntegerField {
    field_num: i64,
    format_code: i32,
    data: Array2<f64>,
    mask: Array2<bool>,
}

fn chl_extract_integer_fields(
    raw_data: &[u8],
    ngates: usize,
    field_nums: &[i64],
    formats: &[i32],
    dat_factors: &[f64],
    dat_biases: &[f64],
    fld_factors: &[f64],
) -> PyResult<Vec<ChlIntegerField>> {
    if ngates == 0 {
        return Err(PyValueError::new_err("ngates must be positive"));
    }
    let field_count = field_nums.len();
    if formats.len() != field_count
        || dat_factors.len() != field_count
        || dat_biases.len() != field_count
        || fld_factors.len() != field_count
    {
        return Err(PyValueError::new_err(
            "field metadata vectors must have identical length",
        ));
    }
    if field_count == 0 {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(field_count);
    let mut record_size = 0_usize;
    for &format_code in formats {
        offsets.push(record_size);
        record_size = record_size
            .checked_add(chl_format_size(format_code)?)
            .ok_or_else(|| PyValueError::new_err("CHL record size overflows usize"))?;
    }
    if record_size == 0 || raw_data.len() % record_size != 0 {
        return Err(PyValueError::new_err(
            "raw_data length must be a multiple of the CHL record size",
        ));
    }
    let record_count = raw_data.len() / record_size;
    if record_count % ngates != 0 {
        return Err(PyValueError::new_err(
            "raw_data record count must be divisible by ngates",
        ));
    }
    let nrays = record_count / ngates;

    let mut fields = Vec::new();
    for field_index in 0..field_count {
        let format_code = formats[field_index];
        if format_code == 2 {
            continue;
        }
        if !dat_factors[field_index].is_finite()
            || !dat_biases[field_index].is_finite()
            || !fld_factors[field_index].is_finite()
            || fld_factors[field_index] == 0.0
        {
            return Err(PyValueError::new_err(
                "CHL integer scale factors must be finite and fld_factor nonzero",
            ));
        }

        let mut data = Array2::<f64>::zeros((nrays, ngates));
        let mut mask = Array2::<bool>::from_elem((nrays, ngates), false);
        for ray in 0..nrays {
            for gate in 0..ngates {
                let record_index = ray
                    .checked_mul(ngates)
                    .and_then(|base| base.checked_add(gate))
                    .ok_or_else(|| PyValueError::new_err("CHL record index overflows usize"))?;
                let byte_index = record_index
                    .checked_mul(record_size)
                    .and_then(|base| base.checked_add(offsets[field_index]))
                    .ok_or_else(|| PyValueError::new_err("CHL byte index overflows usize"))?;
                let raw = chl_read_integer_value(raw_data, byte_index, format_code)?;
                let masked = raw == 0.0;
                mask[(ray, gate)] = masked;
                data[(ray, gate)] = if masked {
                    0.0
                } else {
                    (raw * dat_factors[field_index] + dat_biases[field_index])
                        / fld_factors[field_index]
                };
            }
        }
        fields.push(ChlIntegerField {
            field_num: field_nums[field_index],
            format_code,
            data,
            mask,
        });
    }
    Ok(fields)
}

fn chl_format_size(format_code: i32) -> PyResult<usize> {
    match format_code {
        0 => Ok(1),
        1 => Ok(8),
        2 => Ok(4),
        3 => Ok(2),
        _ => Err(PyValueError::new_err("unsupported CHL format code")),
    }
}

fn chl_read_integer_value(raw_data: &[u8], byte_index: usize, format_code: i32) -> PyResult<f64> {
    match format_code {
        0 => Ok(f64::from(*raw_data.get(byte_index).ok_or_else(|| {
            PyValueError::new_err("CHL uint8 field read exceeds raw_data length")
        })?)),
        1 => {
            let bytes: [u8; 8] = raw_data
                .get(byte_index..byte_index + 8)
                .ok_or_else(|| {
                    PyValueError::new_err("CHL uint64 field read exceeds raw_data length")
                })?
                .try_into()
                .map_err(|_| PyValueError::new_err("CHL uint64 field read failed"))?;
            Ok(u64::from_ne_bytes(bytes) as f64)
        }
        3 => {
            let bytes: [u8; 2] = raw_data
                .get(byte_index..byte_index + 2)
                .ok_or_else(|| {
                    PyValueError::new_err("CHL uint16 field read exceeds raw_data length")
                })?
                .try_into()
                .map_err(|_| PyValueError::new_err("CHL uint16 field read failed"))?;
            Ok(f64::from(u16::from_ne_bytes(bytes)))
        }
        _ => Err(PyValueError::new_err(
            "CHL integer extraction supports uint8, uint16, and uint64",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn rainbow_wrl_get_data<T>(
    databin: &ndarray::ArrayViewD<'_, T>,
    nrays: usize,
    nbins: usize,
    maxbin: usize,
    datamin: f64,
    scale: f64,
    fill_value: f64,
    wrap_phidp: bool,
) -> PyResult<(Array2<f64>, Array2<bool>)>
where
    T: Copy + Into<f64> + PartialEq + From<u8>,
{
    if !databin.is_standard_layout() {
        return Err(PyValueError::new_err("databin must be C-contiguous"));
    }
    if nbins > maxbin {
        return Err(PyValueError::new_err("nbins must be <= maxbin"));
    }
    let output_len = nrays
        .checked_mul(maxbin)
        .ok_or_else(|| PyValueError::new_err("RAINBOW output dimensions overflow usize"))?;
    if output_len > RAINBOW_WRL_MAX_OUTPUT_GATES {
        return Err(PyValueError::new_err(format!(
            "RAINBOW output exceeds native limit ({RAINBOW_WRL_MAX_OUTPUT_GATES} gates)"
        )));
    }
    let expected_len = nrays
        .checked_mul(nbins)
        .ok_or_else(|| PyValueError::new_err("RAINBOW dimensions overflow usize"))?;
    if databin.len() != expected_len {
        return Err(PyValueError::new_err(
            "databin length must equal nrays * nbins",
        ));
    }
    if !datamin.is_finite() || !scale.is_finite() || !fill_value.is_finite() {
        return Err(PyValueError::new_err(
            "datamin, scale, and fill_value must be finite",
        ));
    }

    let mut data = Array2::<f64>::from_elem((nrays, maxbin), fill_value);
    let mut mask = Array2::<bool>::from_elem((nrays, maxbin), true);
    let zero = T::from(0_u8);
    let fill32 = fill_value as f32;

    for (idx, value) in databin.iter().enumerate() {
        let ray = idx / nbins;
        let gate = idx % nbins;
        let masked = *value == zero;
        let mut out32 = if masked {
            f64::from(fill32)
        } else {
            f64::from((datamin + (*value).into() * scale) as f32)
        };
        if wrap_phidp && out32 > 180.0 {
            out32 -= 360.0;
        }
        data[(ray, gate)] = out32;
        mask[(ray, gate)] = masked;
    }
    Ok((data, mask))
}

fn kazr_get_spectra(
    spectra: &ndarray::ArrayView2<'_, f64>,
    indices: &ndarray::ArrayView1<'_, i64>,
    missing: &ndarray::ArrayView1<'_, bool>,
    npulses: usize,
) -> PyResult<Array2<f64>> {
    if !spectra.is_standard_layout()
        || !indices.is_standard_layout()
        || !missing.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "spectra, indices, and missing must be C-contiguous",
        ));
    }
    if spectra.shape()[1] != npulses {
        return Err(PyValueError::new_err(
            "spectra second dimension must match npulses",
        ));
    }
    if indices.len() != missing.len() {
        return Err(PyValueError::new_err(
            "indices and missing must have identical length",
        ));
    }
    let output_len = indices
        .len()
        .checked_mul(npulses)
        .ok_or_else(|| PyValueError::new_err("KAZR output dimensions overflow usize"))?;
    if output_len > KAZR_MAX_OUTPUT_VALUES {
        return Err(PyValueError::new_err(format!(
            "KAZR spectra output exceeds native limit ({KAZR_MAX_OUTPUT_VALUES} values)"
        )));
    }

    let mut out = Array2::<f64>::zeros((indices.len(), npulses));
    for row in 0..indices.len() {
        if missing[row] {
            for gate in 0..npulses {
                out[(row, gate)] = f64::NAN;
            }
            continue;
        }
        let src = usize::try_from(indices[row])
            .map_err(|_| PyValueError::new_err("KAZR spectra index must be nonnegative"))?;
        if src >= spectra.shape()[0] {
            return Err(PyValueError::new_err("KAZR spectra index out of bounds"));
        }
        for gate in 0..npulses {
            out[(row, gate)] = spectra[(src, gate)];
        }
    }
    Ok(out)
}

fn geotiff_rgb_values_f64(
    data: &ndarray::ArrayView2<'_, f64>,
    rgba_lut: &ndarray::ArrayView2<'_, f64>,
    vmin: f64,
    vmax: f64,
    color_levels: f64,
    transpbg: bool,
    op: f64,
) -> PyResult<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<i64>, bool)> {
    if !data.is_standard_layout() || !rgba_lut.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data and rgba_lut must be C-contiguous",
        ));
    }
    if rgba_lut.dim() != (256, 4) {
        return Err(PyValueError::new_err("rgba_lut must have shape (256, 4)"));
    }
    if !vmin.is_finite() || !vmax.is_finite() || !color_levels.is_finite() || !op.is_finite() {
        return Err(PyValueError::new_err(
            "vmin, vmax, color_levels, and op must be finite",
        ));
    }
    if vmax == vmin {
        return Err(PyValueError::new_err("vmax must differ from vmin"));
    }

    let alpha = round_ties_even_i64(op * 255.0)?;
    let shape = data.dim();
    let mut rarr = Array2::<f64>::zeros(shape);
    let mut garr = Array2::<f64>::zeros(shape);
    let mut barr = Array2::<f64>::zeros(shape);
    let mut aarr = Array2::<i64>::zeros(shape);
    let mut has_nan = false;
    let denom = vmax - vmin;

    for (((r_out, g_out), b_out), (a_out, value)) in rarr
        .iter_mut()
        .zip(garr.iter_mut())
        .zip(barr.iter_mut())
        .zip(aarr.iter_mut().zip(data.iter()))
    {
        let val = ((*value - vmin) / denom) * color_levels;
        if val.is_nan() {
            has_nan = true;
            *r_out = f64::NAN;
            *g_out = f64::NAN;
            *b_out = f64::NAN;
            *a_out = if transpbg { 0 } else { alpha };
            continue;
        }

        let clamped = if val < 0.0 {
            0.0
        } else if val > 255.0 {
            255.0
        } else {
            val
        };
        let ind = usize::try_from(round_ties_even_i64(clamped)?)
            .map_err(|_| PyValueError::new_err("GeoTIFF color index out of range"))?;
        if ind > 255 {
            return Err(PyValueError::new_err("GeoTIFF color index out of range"));
        }

        *r_out = round_ties_even_i64(geotiff_lut_channel(rgba_lut, ind, 0)? * 255.0)? as f64;
        *g_out = round_ties_even_i64(geotiff_lut_channel(rgba_lut, ind, 1)? * 255.0)? as f64;
        *b_out = round_ties_even_i64(geotiff_lut_channel(rgba_lut, ind, 2)? * 255.0)? as f64;
        *a_out = alpha;
    }
    Ok((rarr, garr, barr, aarr, has_nan))
}

fn geotiff_lut_channel(
    rgba_lut: &ndarray::ArrayView2<'_, f64>,
    index: usize,
    channel: usize,
) -> PyResult<f64> {
    let value = rgba_lut[(index, channel)];
    if !value.is_finite() {
        return Err(PyValueError::new_err(
            "GeoTIFF RGBA lookup table must be finite",
        ));
    }
    Ok(value)
}

fn edge_mask<T>(
    data: &ndarray::ArrayViewD<'_, T>,
    existing_mask: &ndarray::ArrayViewD<'_, bool>,
    has_missing: bool,
    missing: T,
    has_folded: bool,
    folded: T,
) -> PyResult<ArrayD<bool>>
where
    T: Copy + PartialEq,
{
    if !data.is_standard_layout() || !existing_mask.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data and existing_mask must be C-contiguous",
        ));
    }
    if data.shape() != existing_mask.shape() {
        return Err(PyValueError::new_err(
            "data and existing_mask shapes must match",
        ));
    }

    let mut out = existing_mask.to_owned();
    for (out_value, data_value) in out.iter_mut().zip(data.iter()) {
        if *out_value {
            continue;
        }
        if (has_missing && *data_value == missing) || (has_folded && *data_value == folded) {
            *out_value = true;
        }
    }
    Ok(out)
}

fn round_ties_even_i64(value: f64) -> PyResult<i64> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(PyValueError::new_err("value cannot be rounded to int64"));
    }
    let floor = value.floor();
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        let base = floor as i64;
        if base & 1 == 0 {
            floor
        } else {
            floor + 1.0
        }
    };
    Ok(rounded as i64)
}

fn nexrad_level2_scan_msgs_i64(elevation_numbers: &[i64]) -> PyResult<Vec<Vec<i64>>> {
    let Some(&max_elevation) = elevation_numbers.iter().max() else {
        return Err(PyValueError::new_err("elevation_numbers must be non-empty"));
    };
    if max_elevation <= 0 {
        return Ok(Vec::new());
    }
    let nscans = usize::try_from(max_elevation)
        .map_err(|_| PyValueError::new_err("elevation_number is too large"))?;
    if nscans > NEXRAD_LEVEL2_SCAN_MSGS_MAX_RECORDS {
        return Err(PyValueError::new_err(
            "elevation_number exceeds native size limit",
        ));
    }

    let mut scan_msgs = vec![Vec::new(); nscans];
    for (record_index, &elevation_number) in elevation_numbers.iter().enumerate() {
        if elevation_number <= 0 || elevation_number > max_elevation {
            continue;
        }
        scan_msgs[(elevation_number - 1) as usize].push(record_index as i64);
    }
    Ok(scan_msgs)
}

fn nexrad_level2_msg_nums_i64(scan_msgs: &[Vec<i64>], scans: &[i64]) -> PyResult<Vec<i64>> {
    let mut total = 0_usize;
    for &scan in scans {
        if scan < 0 {
            return Err(PyValueError::new_err("scan index must be non-negative"));
        }
        let scan_index =
            usize::try_from(scan).map_err(|_| PyValueError::new_err("scan index is too large"))?;
        if scan_index >= scan_msgs.len() {
            return Err(PyValueError::new_err("scan index is out of range"));
        }
        total = total
            .checked_add(scan_msgs[scan_index].len())
            .ok_or_else(|| PyValueError::new_err("message index output is too large"))?;
        if total > NEXRAD_LEVEL2_SCAN_MSGS_MAX_RECORDS {
            return Err(PyValueError::new_err(
                "message index output exceeds native size limit",
            ));
        }
    }

    let mut out = Vec::with_capacity(total);
    for &scan in scans {
        out.extend_from_slice(&scan_msgs[scan as usize]);
    }
    Ok(out)
}

fn sigmet_bin2_to_angle(value: u16) -> f64 {
    360.0 * f64::from(value) / 65_536.0
}

fn sigmet_bin4_to_angle(value: u32) -> f64 {
    360.0 * f64::from(value) / 4_294_967_296.0
}

fn sigmet_time_order_ref_time_slice<'a>(ref_time: &'a ArrayView1<'_, i32>) -> PyResult<&'a [i32]> {
    if ref_time.len() > SIGMET_TIME_ORDER_MAX_RAYS {
        return Err(PyValueError::new_err("ref_time exceeds native size limit"));
    }
    if !ref_time.is_standard_layout() {
        return Err(PyValueError::new_err("ref_time must be C-contiguous"));
    }
    ref_time
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ref_time must be C-contiguous"))
}

fn sigmet_time_order_rays_slice<'a>(
    rays_per_sweep: &'a ArrayView1<'_, i64>,
) -> PyResult<&'a [i64]> {
    if rays_per_sweep.len() > SIGMET_TIME_ORDER_MAX_RAYS {
        return Err(PyValueError::new_err(
            "rays_per_sweep exceeds native size limit",
        ));
    }
    if !rays_per_sweep.is_standard_layout() {
        return Err(PyValueError::new_err("rays_per_sweep must be C-contiguous"));
    }
    let rays_per_sweep = rays_per_sweep
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rays_per_sweep must be C-contiguous"))?;
    if rays_per_sweep.iter().any(|&nrays| nrays < 0) {
        return Err(PyValueError::new_err("rays_per_sweep must be non-negative"));
    }
    Ok(rays_per_sweep)
}

fn sigmet_time_order_sweep_end(
    start: usize,
    nrays: i64,
    ref_len: usize,
) -> PyResult<Option<usize>> {
    if nrays == 0 || nrays == 1 {
        return Ok(None);
    }
    let nrays = usize::try_from(nrays)
        .map_err(|_| PyValueError::new_err("rays_per_sweep must be non-negative"))?;
    let end = start
        .checked_add(nrays)
        .ok_or_else(|| PyValueError::new_err("rays_per_sweep exceeds ref_time length"))?;
    if end > ref_len {
        return Err(PyValueError::new_err(
            "rays_per_sweep exceeds ref_time length",
        ));
    }
    Ok(Some(end))
}

fn sigmet_count_negative_diffs(slice: &[i32]) -> usize {
    slice
        .windows(2)
        .filter(|pair| pair[1].wrapping_sub(pair[0]) < 0)
        .count()
}

fn sigmet_count_negative_diffs_reversed(slice: &[i32]) -> usize {
    (1..slice.len())
        .rev()
        .filter(|&index| slice[index - 1].wrapping_sub(slice[index]) < 0)
        .count()
}

fn sigmet_min_diff_argmin(slice: &[i32]) -> (i32, usize) {
    let mut min_diff = slice[1].wrapping_sub(slice[0]);
    let mut argmin = 0_usize;
    for index in 1..slice.len() - 1 {
        let diff = slice[index + 1].wrapping_sub(slice[index]);
        if diff < min_diff {
            min_diff = diff;
            argmin = index;
        }
    }
    (min_diff, argmin)
}

fn sigmet_time_ordered_by_reversal_i32(ref_time: &[i32], rays_per_sweep: &[i64]) -> PyResult<bool> {
    let mut start = 0_usize;
    for &nrays in rays_per_sweep {
        let Some(end) = sigmet_time_order_sweep_end(start, nrays, ref_time.len())? else {
            continue;
        };
        let sweep = &ref_time[start..end];
        start = end;

        let mut all_increasing = true;
        let mut all_decreasing = true;
        for pair in sweep.windows(2) {
            let diff = pair[1].wrapping_sub(pair[0]);
            if diff < 0 {
                all_increasing = false;
            }
            if diff > 0 {
                all_decreasing = false;
            }
        }
        if !(all_increasing || all_decreasing) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sigmet_time_ordered_by_roll_i32(ref_time: &[i32], rays_per_sweep: &[i64]) -> PyResult<bool> {
    let mut start = 0_usize;
    for &nrays in rays_per_sweep {
        let Some(end) = sigmet_time_order_sweep_end(start, nrays, ref_time.len())? else {
            continue;
        };
        let sweep = &ref_time[start..end];
        start = end;

        let mut count = sigmet_count_negative_diffs(sweep);
        if sweep[0].wrapping_sub(sweep[sweep.len() - 1]) < 0 {
            count += 1;
        }
        if count != 0 && count != 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sigmet_time_ordered_by_reverse_roll_i32(
    ref_time: &[i32],
    rays_per_sweep: &[i64],
) -> PyResult<bool> {
    let mut start = 0_usize;
    for &nrays in rays_per_sweep {
        let Some(end) = sigmet_time_order_sweep_end(start, nrays, ref_time.len())? else {
            continue;
        };
        let sweep = &ref_time[start..end];
        start = end;

        let mut first = sweep[0];
        let mut last = sweep[sweep.len() - 1];
        let mut count = sigmet_count_negative_diffs(sweep);
        if count > 0 {
            count = sigmet_count_negative_diffs_reversed(sweep);
            first = last;
            last = sweep[0];
        }
        if first.wrapping_sub(last) < 0 {
            count += 1;
        }
        if count != 0 && count != 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sigmet_identity_order(len: usize) -> Vec<i64> {
    (0..len).map(|index| index as i64).collect()
}

fn sigmet_time_order_roll_index_i32(
    ref_time: &[i32],
    rays_per_sweep: &[i64],
) -> PyResult<Vec<i64>> {
    let mut order = sigmet_identity_order(ref_time.len());
    let mut start = 0_usize;
    for &nrays in rays_per_sweep {
        let Some(end) = sigmet_time_order_sweep_end(start, nrays, ref_time.len())? else {
            continue;
        };
        let sweep = &ref_time[start..end];
        let (min_diff, argmin) = sigmet_min_diff_argmin(sweep);
        if min_diff < 0 {
            let shift_left = argmin + 1;
            let len = sweep.len();
            for offset in 0..len {
                order[start + offset] = (start + ((offset + shift_left) % len)) as i64;
            }
        }
        start = end;
    }
    Ok(order)
}

fn sigmet_time_order_reverse_index_i32(
    ref_time: &[i32],
    rays_per_sweep: &[i64],
) -> PyResult<Vec<i64>> {
    let mut order = sigmet_identity_order(ref_time.len());
    let mut start = 0_usize;
    for &nrays in rays_per_sweep {
        let Some(end) = sigmet_time_order_sweep_end(start, nrays, ref_time.len())? else {
            continue;
        };
        let sweep = &ref_time[start..end];
        let (min_diff, _) = sigmet_min_diff_argmin(sweep);
        if min_diff < 0 {
            let len = sweep.len();
            for offset in 0..len {
                order[start + offset] = (end - 1 - offset) as i64;
            }
        }
        start = end;
    }
    Ok(order)
}

fn sigmet_time_order_full_index_i32(
    ref_time: &[i32],
    rays_per_sweep: &[i64],
) -> PyResult<Vec<i64>> {
    let mut order = sigmet_identity_order(ref_time.len());
    let mut start = 0_usize;
    for &nrays in rays_per_sweep {
        let Some(end) = sigmet_time_order_sweep_end(start, nrays, ref_time.len())? else {
            continue;
        };
        let sweep = &ref_time[start..end];
        let (min_diff, _) = sigmet_min_diff_argmin(sweep);
        if min_diff < 0 {
            let mut sort_idx: Vec<usize> = (0..sweep.len()).collect();
            sort_idx.sort_by_key(|&index| sweep[index]);
            for (offset, source_offset) in sort_idx.into_iter().enumerate() {
                order[start + offset] = (start + source_offset) as i64;
            }
        }
        start = end;
    }
    Ok(order)
}

fn extract_sigmet_mask_word(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    let type_name = value.get_type().name()?;
    if type_name.to_str()? == "bool" || type_name.to_str()? == "bool_" {
        return Err(PyValueError::new_err(
            "mask word must be an integer in the range 0..=0xffffffff",
        ));
    }
    value.extract::<u32>().map_err(|_| {
        PyValueError::new_err("mask word must be an integer in the range 0..=0xffffffff")
    })
}

fn sigmet_data_types_from_mask(words: [u32; 4]) -> Vec<usize> {
    let mut data_types = Vec::new();
    for (word_index, word) in words.iter().copied().enumerate() {
        for bit in 0..32 {
            if (word >> bit) & 1 == 1 {
                data_types.push(word_index * 32 + bit);
            }
        }
    }
    data_types
}

fn sigmet_decode_ray_current_record_i16(
    rbuf: &ndarray::ArrayView1<'_, i16>,
    rbuf_pos: isize,
    nbins: usize,
    out: &mut ArrayViewMut1<'_, i16>,
) -> PyResult<Option<(i32, isize)>> {
    if !rbuf.is_standard_layout() || !out.is_standard_layout() {
        return Err(PyValueError::new_err("rbuf and out must be C-contiguous"));
    }
    if rbuf.len() != SIGMET_RECORD_WORDS {
        return Err(PyValueError::new_err("rbuf length must be 3072 words"));
    }
    let expected_out_len = nbins
        .checked_add(6)
        .ok_or_else(|| PyValueError::new_err("nbins is too large"))?;
    if out.len() != expected_out_len {
        return Err(PyValueError::new_err(
            "out length must equal nbins plus 6 ray header words",
        ));
    }
    if !(-1..SIGMET_RECORD_WORDS as isize).contains(&rbuf_pos) {
        return Err(PyValueError::new_err(
            "rbuf_pos must be in the range -1..3071",
        ));
    }

    if !sigmet_ray_current_record_can_decode(rbuf, rbuf_pos, expected_out_len) {
        return Ok(None);
    }

    let (status, next_pos) =
        sigmet_ray_current_record_decode_unchecked(rbuf, rbuf_pos, expected_out_len, out);
    Ok(Some((status, next_pos)))
}

fn sigmet_ray_next_pos(pos: isize) -> Option<usize> {
    let next = pos.checked_add(1)?;
    if next < 0 || next >= SIGMET_RECORD_WORDS as isize {
        None
    } else {
        Some(next as usize)
    }
}

fn sigmet_ray_run_words(compression_code: i32) -> usize {
    (compression_code + 32_768) as usize
}

fn sigmet_ray_current_record_can_decode(
    rbuf: &ndarray::ArrayView1<'_, i16>,
    rbuf_pos: isize,
    out_len: usize,
) -> bool {
    let Some(first_pos) = sigmet_ray_next_pos(rbuf_pos) else {
        return false;
    };
    let mut pos = first_pos as isize;
    let mut compression_code = i32::from(rbuf[first_pos]);
    let mut out_pos = 0_usize;

    while compression_code != 1 {
        let Some(payload_pos) = sigmet_ray_next_pos(pos) else {
            return false;
        };
        pos = payload_pos as isize;

        if compression_code < 0 {
            let words = sigmet_ray_run_words(compression_code);
            let Some(next_out_pos) = out_pos.checked_add(words) else {
                return false;
            };
            if next_out_pos > out_len {
                return false;
            }
            let Some(next_pos) = payload_pos.checked_add(words) else {
                return false;
            };
            if next_pos >= SIGMET_RECORD_WORDS {
                return false;
            }
            out_pos = next_out_pos;
            pos = next_pos as isize;
        } else {
            let count = compression_code as usize;
            let Some(next_out_pos) = out_pos.checked_add(count) else {
                return false;
            };
            if next_out_pos > out_len {
                return true;
            }
            out_pos = next_out_pos;
        }

        compression_code = i32::from(rbuf[pos as usize]);
    }

    true
}

fn sigmet_ray_current_record_decode_unchecked(
    rbuf: &ndarray::ArrayView1<'_, i16>,
    rbuf_pos: isize,
    out_len: usize,
    out: &mut ArrayViewMut1<'_, i16>,
) -> (i32, isize) {
    let first_pos = sigmet_ray_next_pos(rbuf_pos).expect("current-record scan verified first pos");
    let mut pos = first_pos as isize;
    let mut compression_code = i32::from(rbuf[first_pos]);
    let mut out_pos = 0_usize;

    if compression_code == 1 {
        out[4] = -1;
        return (0, pos);
    }

    while compression_code != 1 {
        let payload_pos =
            sigmet_ray_next_pos(pos).expect("current-record scan verified payload pos");
        pos = payload_pos as isize;

        if compression_code < 0 {
            let words = sigmet_ray_run_words(compression_code);
            let next_out_pos = out_pos + words;
            let next_pos = payload_pos + words;
            if words != 0 {
                let src = rbuf.slice(ndarray::s![payload_pos..next_pos]);
                let mut dst = out.slice_mut(ndarray::s![out_pos..next_out_pos]);
                dst.assign(&src);
            }
            out_pos = next_out_pos;
            pos = next_pos as isize;
        } else {
            let count = compression_code as usize;
            let next_out_pos = out_pos + count;
            if next_out_pos > out_len {
                return (-1, pos);
            }
            for value in &mut out.slice_mut(ndarray::s![out_pos..next_out_pos]) {
                *value = 0;
            }
            out_pos = next_out_pos;
        }

        compression_code = i32::from(rbuf[pos as usize]);
    }

    (0, pos)
}

fn sigmet_convert_like_dbt2_dense_i16(
    data: &ndarray::ArrayView2<'_, i16>,
    nbins: &ndarray::ArrayView1<'_, i64>,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if !data.is_standard_layout() || !nbins.is_standard_layout() {
        return Err(PyValueError::new_err("data and nbins must be C-contiguous"));
    }

    let (nrays, full_nbins) = data.dim();
    if nbins.len() != nrays {
        return Err(PyValueError::new_err(
            "nbins length must match the number of rays",
        ));
    }
    if nbins.iter().any(|&value| value < 0) {
        return Err(PyValueError::new_err("nbins values must be non-negative"));
    }

    let mut out = Array2::<f32>::zeros((nrays, full_nbins));
    let mut mask = Array2::<bool>::from_elem((nrays, full_nbins), false);
    for ((ray, gate), &raw) in data.indexed_iter() {
        let raw_u16 = raw as u16;
        out[[ray, gate]] = ((f64::from(raw_u16) - 32_768.0) / 100.0) as f32;
        mask[[ray, gate]] = raw_u16 == 0;
    }

    for ray in 0..nrays {
        let nbin = usize::try_from(nbins[ray])
            .map_err(|_| PyValueError::new_err("nbins values are too large"))?;
        if nbin >= full_nbins {
            continue;
        }
        for gate in nbin..full_nbins {
            mask[[ray, gate]] = true;
        }
    }

    Ok((out, mask))
}

fn sigmet_convert_like_dbt_dense_i16(
    data: &ndarray::ArrayView2<'_, i16>,
    nbins: &ndarray::ArrayView1<'_, i64>,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if !data.is_standard_layout() || !nbins.is_standard_layout() {
        return Err(PyValueError::new_err("data and nbins must be C-contiguous"));
    }

    let (nrays, full_nbins) = data.dim();
    if nbins.len() != nrays {
        return Err(PyValueError::new_err(
            "nbins length must match the number of rays",
        ));
    }
    if nbins.iter().any(|&value| value < 0) {
        return Err(PyValueError::new_err("nbins values must be non-negative"));
    }

    let mut out = Array2::<f32>::zeros((nrays, full_nbins));
    let mut mask = Array2::<bool>::from_elem((nrays, full_nbins), false);
    for ray in 0..nrays {
        for gate in 0..full_nbins {
            let value = data[[ray, gate / 2]].to_ne_bytes()[gate % 2];
            out[[ray, gate]] = ((f64::from(value) - 64.0) / 2.0) as f32;
            mask[[ray, gate]] = value == 0;
        }
    }

    for ray in 0..nrays {
        let nbin = usize::try_from(nbins[ray])
            .map_err(|_| PyValueError::new_err("nbins values are too large"))?;
        if nbin >= full_nbins {
            continue;
        }
        for gate in nbin..full_nbins {
            mask[[ray, gate]] = true;
        }
    }

    Ok((out, mask))
}

#[derive(Clone, Copy)]
enum SigmetU8Mode {
    LikeSqi,
    Vel,
    VelC,
    Width,
    Zdr,
    Kdp,
    PhiDp,
    HClass,
}

fn sigmet_convert_u8_dense_i16(
    data: &ndarray::ArrayView2<'_, i16>,
    nbins: &ndarray::ArrayView1<'_, i64>,
    mode: SigmetU8Mode,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if !data.is_standard_layout() || !nbins.is_standard_layout() {
        return Err(PyValueError::new_err("data and nbins must be C-contiguous"));
    }

    let (nrays, full_nbins) = data.dim();
    if nbins.len() != nrays {
        return Err(PyValueError::new_err(
            "nbins length must match the number of rays",
        ));
    }
    if nrays == 0 {
        return Err(PyValueError::new_err(
            "cannot reshape array of size 0 into shape (0,newaxis)",
        ));
    }
    if nbins.iter().any(|&value| value < 0) {
        return Err(PyValueError::new_err("nbins values must be non-negative"));
    }

    let mut out = Array2::<f32>::zeros((nrays, full_nbins));
    let mut mask = Array2::<bool>::from_elem((nrays, full_nbins), false);
    for ray in 0..nrays {
        for gate in 0..full_nbins {
            let value = data[[ray, gate / 2]].to_ne_bytes()[gate % 2];
            out[[ray, gate]] = match mode {
                SigmetU8Mode::LikeSqi => ((f64::from(value) - 1.0) / 253.0).sqrt() as f32,
                SigmetU8Mode::Vel => ((f64::from(value) - 128.0) / 127.0) as f32,
                SigmetU8Mode::VelC => (((f64::from(value) - 128.0) / 127.0) * 75.0) as f32,
                SigmetU8Mode::Width => (f64::from(value) / 256.0) as f32,
                SigmetU8Mode::Zdr => ((f64::from(value) - 128.0) / 16.0) as f32,
                SigmetU8Mode::Kdp => sigmet_kdp_u8_value(value),
                SigmetU8Mode::PhiDp => (180.0 * ((f64::from(value) - 1.0) / 254.0)) as f32,
                SigmetU8Mode::HClass => f32::from(value),
            };
            mask[[ray, gate]] = match mode {
                SigmetU8Mode::Vel | SigmetU8Mode::Width | SigmetU8Mode::Zdr => value == 0,
                SigmetU8Mode::LikeSqi
                | SigmetU8Mode::VelC
                | SigmetU8Mode::Kdp
                | SigmetU8Mode::PhiDp
                | SigmetU8Mode::HClass => value == 0 || value == 255,
            };
        }
    }

    for ray in 0..nrays {
        let nbin = usize::try_from(nbins[ray])
            .map_err(|_| PyValueError::new_err("nbins values are too large"))?;
        if nbin >= full_nbins {
            continue;
        }
        for gate in nbin..full_nbins {
            mask[[ray, gate]] = true;
        }
    }

    Ok((out, mask))
}

fn sigmet_kdp_u8_value(value: u8) -> f32 {
    if value > 128 {
        (0.25 * 600.0_f64.powf((f64::from(value) - 129.0) / 126.0)) as f32
    } else if value < 128 {
        (-0.25 * 600.0_f64.powf((127.0 - f64::from(value)) / 126.0)) as f32
    } else {
        0.0
    }
}

#[derive(Clone, Copy)]
enum SigmetU16Mode {
    LikeSqi2,
    Width2,
    PhiDp2,
    HClass2,
}

fn sigmet_convert_u16_dense_i16(
    data: &ndarray::ArrayView2<'_, i16>,
    nbins: &ndarray::ArrayView1<'_, i64>,
    mode: SigmetU16Mode,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if !data.is_standard_layout() || !nbins.is_standard_layout() {
        return Err(PyValueError::new_err("data and nbins must be C-contiguous"));
    }

    let (nrays, full_nbins) = data.dim();
    if nbins.len() != nrays {
        return Err(PyValueError::new_err(
            "nbins length must match the number of rays",
        ));
    }
    if nbins.iter().any(|&value| value < 0) {
        return Err(PyValueError::new_err("nbins values must be non-negative"));
    }

    let mut out = Array2::<f32>::zeros((nrays, full_nbins));
    let mut mask = Array2::<bool>::from_elem((nrays, full_nbins), false);
    for ((ray, gate), &raw) in data.indexed_iter() {
        let raw_u16 = raw as u16;
        out[[ray, gate]] = match mode {
            SigmetU16Mode::LikeSqi2 => ((f64::from(raw_u16) - 1.0) / 65_533.0) as f32,
            SigmetU16Mode::Width2 => (f64::from(raw_u16) / 100.0) as f32,
            SigmetU16Mode::PhiDp2 => (360.0 * (f64::from(raw_u16) - 1.0) / 65_534.0) as f32,
            SigmetU16Mode::HClass2 => f32::from(raw_u16),
        };
        mask[[ray, gate]] = !matches!(mode, SigmetU16Mode::HClass2) && raw_u16 == 0;
    }

    for ray in 0..nrays {
        let nbin = usize::try_from(nbins[ray])
            .map_err(|_| PyValueError::new_err("nbins values are too large"))?;
        if nbin >= full_nbins {
            continue;
        }
        for gate in nbin..full_nbins {
            mask[[ray, gate]] = true;
        }
    }

    Ok((out, mask))
}

fn extract_mdv_key(value: &Bound<'_, PyAny>) -> PyResult<u8> {
    let type_name = value.get_type().name()?;
    if type_name.to_str()? == "bool" || type_name.to_str()? == "bool_" {
        return Err(PyValueError::new_err(
            "key must be an integer in the range 0..=255",
        ));
    }
    value
        .extract::<u8>()
        .map_err(|_| PyValueError::new_err("key must be an integer in the range 0..=255"))
}

fn extract_mdv_decompr_size(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let type_name = value.get_type().name()?;
    if type_name.to_str()? == "bool" || type_name.to_str()? == "bool_" {
        return Err(PyValueError::new_err(
            "decompr_size must be a non-negative integer",
        ));
    }
    value
        .extract::<usize>()
        .map_err(|_| PyValueError::new_err("decompr_size must be a non-negative integer"))
}

fn mdv_decode_rle8_exact(compr_data: &[u8], key: u8, decompr_size: usize) -> PyResult<Vec<u8>> {
    let decoded_len = mdv_rle8_checked_decoded_len(compr_data, key, decompr_size)?;
    let mut output = Vec::with_capacity(decoded_len);
    let mut data_ptr = 0_usize;
    while data_ptr != compr_data.len() {
        let value = compr_data[data_ptr];
        if value != key {
            output.push(value);
            data_ptr += 1;
        } else {
            let count = compr_data[data_ptr + 1] as usize;
            output.resize(output.len() + count, compr_data[data_ptr + 2]);
            data_ptr += 3;
        }
    }
    Ok(output)
}

fn mdv_rle8_checked_decoded_len(
    compr_data: &[u8],
    key: u8,
    decompr_size: usize,
) -> PyResult<usize> {
    if decompr_size > MDV_RLE8_MAX_OUTPUT_BYTES {
        return Err(PyValueError::new_err(
            "decompr_size exceeds maximum native RLE8 output size",
        ));
    }

    let mut decoded_len = 0_usize;
    let mut data_ptr = 0_usize;
    let mut saw_run = false;
    while data_ptr != compr_data.len() {
        let value = compr_data[data_ptr];
        if value != key {
            decoded_len = decoded_len
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("decoded RLE8 data is too large"))?;
            data_ptr += 1;
        } else {
            saw_run = true;
            if data_ptr + 2 >= compr_data.len() {
                return Err(PyValueError::new_err("encoded RLE8 run is truncated"));
            }
            let count = compr_data[data_ptr + 1] as usize;
            decoded_len = decoded_len
                .checked_add(count)
                .ok_or_else(|| PyValueError::new_err("decoded RLE8 data is too large"))?;
            data_ptr += 3;
        }
        if decoded_len > decompr_size {
            return Err(PyValueError::new_err(
                "decoded RLE8 data exceeds decompr_size",
            ));
        }
        if saw_run && decoded_len > 255 {
            return Err(PyValueError::new_err(
                "decoded RLE8 run output exceeds safe Python pointer range",
            ));
        }
    }
    if decoded_len != decompr_size {
        return Err(PyValueError::new_err(
            "decoded RLE8 data length does not match decompr_size",
        ));
    }
    Ok(decoded_len)
}

fn extract_nexrad_af1f_nbins(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let type_name = value.get_type().name()?;
    if type_name.to_str()? == "bool" || type_name.to_str()? == "bool_" {
        return Err(PyValueError::new_err(
            "nbins must be a non-negative integer",
        ));
    }
    value
        .extract::<usize>()
        .map_err(|_| PyValueError::new_err("nbins must be a non-negative integer"))
}

fn nexrad_af1f_decode_rle_exact(rle_data: &[u8], nbins: usize) -> PyResult<Vec<u8>> {
    let decoded_len = nexrad_af1f_checked_decoded_len(rle_data, nbins)?;
    let mut output = Vec::with_capacity(decoded_len);
    for &byte in rle_data {
        let run = (byte >> 4) as usize;
        let color = byte & 0x0f;
        output.resize(output.len() + run, color);
    }
    Ok(output)
}

fn nexrad_af1f_checked_decoded_len(rle_data: &[u8], nbins: usize) -> PyResult<usize> {
    if nbins > NEXRAD_AF1F_MAX_OUTPUT_BINS {
        return Err(PyValueError::new_err(
            "nbins exceeds maximum native AF1F output size",
        ));
    }

    let mut decoded_len = 0_usize;
    for &byte in rle_data {
        decoded_len = decoded_len
            .checked_add((byte >> 4) as usize)
            .ok_or_else(|| PyValueError::new_err("decoded AF1F data is too large"))?;
        if decoded_len > nbins {
            return Err(PyValueError::new_err("decoded AF1F data exceeds nbins"));
        }
    }
    if decoded_len != nbins {
        return Err(PyValueError::new_err(
            "decoded AF1F data length does not match nbins",
        ));
    }
    Ok(decoded_len)
}

fn nexrad_level3_data_8_or_16(
    threshold_data: &[u8],
    raw_data: &ndarray::ArrayView2<'_, u8>,
) -> PyResult<(Array2<f64>, Array2<bool>)> {
    if threshold_data.len() < 32 {
        return Err(PyValueError::new_err(
            "threshold_data must contain at least 32 bytes",
        ));
    }

    let mut data_levels = [0.0_f64; 16];
    let flags0 = threshold_data[0];
    let mut scale = 1.0;
    if flags0 & 2_u8.pow(5) != 0 {
        scale = 1.0 / 20.0;
    }
    if flags0 & 2_u8.pow(4) != 0 {
        scale = 1.0 / 10.0;
    }

    for index in 0..16 {
        let flag = threshold_data[index * 2];
        let value = f64::from(threshold_data[index * 2 + 1]);
        let sign = if flag & 0x01 == 0 { 1.0 } else { -1.0 };
        data_levels[index] = value * sign * scale;
        if flag & 0x80 == 0x80 {
            data_levels[index] = -999.0;
        }
    }

    let shape = raw_data.dim();
    let mut data = Array2::<f64>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        if raw >= 16 {
            return Err(PyValueError::new_err("invalid entry in choice array"));
        }
        let value = data_levels[raw as usize];
        data[[ray, gate]] = value;
        mask[[ray, gate]] = value == -999.0;
    }
    Ok((data, mask))
}

fn nexrad_level3_msg_134(
    threshold_data: &[u8],
    raw_data: &ndarray::ArrayView2<'_, u8>,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if threshold_data.len() < 10 {
        return Err(PyValueError::new_err(
            "threshold_data must contain at least 10 bytes",
        ));
    }

    let hw31 = read_be_i16(threshold_data, 0)? as i64;
    let hw32 = read_be_i16(threshold_data, 2)? as i64;
    let hw33 = read_be_i16(threshold_data, 4)?;
    let hw34 = read_be_i16(threshold_data, 6)? as i64;
    let hw35 = read_be_i16(threshold_data, 8)? as i64;

    let linear_scale = nexrad_level3_int16_to_float16(hw31);
    let linear_offset = nexrad_level3_int16_to_float16(hw32);
    let log_start = hw33;
    let log_scale = nexrad_level3_int16_to_float16(hw34);
    let log_offset = nexrad_level3_int16_to_float16(hw35);
    if !linear_scale.is_finite() || !log_scale.is_finite() {
        return Err(PyValueError::new_err("msg 134 scale values must be finite"));
    }
    if linear_scale == 0.0 || log_scale == 0.0 {
        return Err(PyValueError::new_err(
            "msg 134 scale values must be non-zero",
        ));
    }

    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        let value = f64::from(raw);
        let scaled = if i16::from(raw) < log_start {
            (value - linear_offset) / linear_scale
        } else {
            ((value - log_offset) / log_scale).exp()
        };
        data[[ray, gate]] = scaled as f32;
        mask[[ray, gate]] = raw < 2;
    }
    Ok((data, mask))
}

fn read_be_i16(data: &[u8], offset: usize) -> PyResult<i16> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| PyValueError::new_err("threshold_data is too short"))?;
    Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
}

fn nexrad_level3_msg_135(raw_data: &ndarray::ArrayView2<'_, u8>) -> (Array2<f32>, Array2<bool>) {
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        let mut value = raw.wrapping_sub(2);
        if raw >= 128 {
            value = value.wrapping_sub(128);
        }
        data[[ray, gate]] = f32::from(value);
        mask[[ray, gate]] = raw <= 1;
    }
    (data, mask)
}

fn nexrad_level3_msg_138(
    threshold_data: &[u8],
    raw_data: &ndarray::ArrayView2<'_, u8>,
) -> PyResult<Array2<f32>> {
    if threshold_data.len() < 4 {
        return Err(PyValueError::new_err(
            "threshold_data must contain at least 4 bytes",
        ));
    }
    let hw31 = read_be_i16(threshold_data, 0)? as f64;
    let hw32 = read_be_i16(threshold_data, 2)? as f64;
    let offset = hw31 / 100.0;
    let scale = hw32 / 100.0;
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        data[[ray, gate]] = (f64::from(raw) * scale + offset) as f32;
    }
    Ok(data)
}

fn nexrad_level3_msg_scaled_u8(
    threshold_data: &[u8],
    raw_data: &ndarray::ArrayView2<'_, u8>,
    subtract_two: bool,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if threshold_data.len() < 4 {
        return Err(PyValueError::new_err(
            "threshold_data must contain at least 4 bytes",
        ));
    }
    let hw31 = read_be_i16(threshold_data, 0)? as f64;
    let hw32 = read_be_i16(threshold_data, 2)? as f64;
    let offset = hw31 / 10.0;
    let scale = hw32 / 10.0;
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        let value = if subtract_two {
            raw.wrapping_sub(2)
        } else {
            raw
        };
        data[[ray, gate]] = (f64::from(value) * scale + offset) as f32;
        mask[[ray, gate]] = raw < 2;
    }
    Ok((data, mask))
}

fn nexrad_level3_msg_scaled_u16(
    threshold_data: &[u8],
    raw_data: &ndarray::ArrayView2<'_, u16>,
    subtract_two: bool,
) -> PyResult<(Array2<f32>, Array2<bool>)> {
    if threshold_data.len() < 4 {
        return Err(PyValueError::new_err(
            "threshold_data must contain at least 4 bytes",
        ));
    }
    let hw31 = read_be_i16(threshold_data, 0)? as f64;
    let hw32 = read_be_i16(threshold_data, 2)? as f64;
    let offset = hw31 / 10.0;
    let scale = hw32 / 10.0;
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        let value = if subtract_two {
            raw.wrapping_sub(2)
        } else {
            raw
        };
        data[[ray, gate]] = (f64::from(value) * scale + offset) as f32;
        mask[[ray, gate]] = raw < 2;
    }
    Ok((data, mask))
}

fn nexrad_level3_mask_zero_u8(
    raw_data: &ndarray::ArrayView2<'_, u8>,
) -> (Array2<f32>, Array2<bool>) {
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        data[[ray, gate]] = f32::from(raw);
        mask[[ray, gate]] = raw == 0;
    }
    (data, mask)
}

fn nexrad_level3_mask_zero_u16(
    raw_data: &ndarray::ArrayView2<'_, u16>,
) -> (Array2<f32>, Array2<bool>) {
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    let mut mask = Array2::<bool>::from_elem(shape, false);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        data[[ray, gate]] = f32::from(raw);
        mask[[ray, gate]] = raw == 0;
    }
    (data, mask)
}

fn nexrad_level3_copy_u8(raw_data: &ndarray::ArrayView2<'_, u8>) -> Array2<f32> {
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        data[[ray, gate]] = f32::from(raw);
    }
    data
}

fn nexrad_level3_copy_u16(raw_data: &ndarray::ArrayView2<'_, u16>) -> Array2<f32> {
    let shape = raw_data.dim();
    let mut data = Array2::<f32>::zeros(shape);
    for ((ray, gate), &raw) in raw_data.indexed_iter() {
        data[[ray, gate]] = f32::from(raw);
    }
    data
}

fn array_from_shape_vec<T>(shape: IxDyn, values: Vec<T>) -> PyResult<ArrayD<T>> {
    ArrayD::from_shape_vec(shape, values)
        .map_err(|err| PyValueError::new_err(format!("invalid output shape: {err}")))
}

fn uf_sweep_limits_i32(ray_sweep_numbers: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut limits: BTreeMap<i32, (i32, i32)> = BTreeMap::new();
    for (ray_index, &sweep_number) in ray_sweep_numbers.iter().enumerate() {
        let ray_index = ray_index as i32;
        limits
            .entry(sweep_number)
            .and_modify(|(_, last)| *last = ray_index)
            .or_insert((ray_index, ray_index));
    }
    let mut first = Vec::with_capacity(limits.len());
    let mut last = Vec::with_capacity(limits.len());
    for (_sweep_number, (first_ray, last_ray)) in limits {
        first.push(first_ray);
        last.push(last_ray);
    }
    (first, last)
}

fn uf_ray_num_to_sweep_num_i32(nrays: usize, starts: &[i32], ends: &[i32]) -> PyResult<Vec<i32>> {
    if starts.len() != ends.len() {
        return Err(PyValueError::new_err("starts and ends lengths must match"));
    }
    let mut ray_map = vec![0_i32; nrays];
    for (isweep, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
        if start < 0 || end < 0 {
            return Err(PyValueError::new_err("sweep bounds must be non-negative"));
        }
        if end < start {
            return Err(PyValueError::new_err("sweep end must be >= start"));
        }
        let start = usize::try_from(start)
            .map_err(|_| PyValueError::new_err("sweep start is too large"))?;
        let end =
            usize::try_from(end).map_err(|_| PyValueError::new_err("sweep end is too large"))?;
        if end >= nrays {
            return Err(PyValueError::new_err("sweep end is outside nrays"));
        }
        let isweep = i32::try_from(isweep)
            .map_err(|_| PyValueError::new_err("sweep index exceeds int32"))?;
        for value in &mut ray_map[start..=end] {
            *value = isweep;
        }
    }
    Ok(ray_map)
}

fn mask_gates_not_collected(
    mut mask: ArrayViewMut2<'_, u8>,
    nbins: &ndarray::ArrayView1<'_, i64>,
) -> PyResult<()> {
    if !mask.is_standard_layout() || !nbins.is_standard_layout() {
        return Err(PyValueError::new_err("mask and nbins must be C-contiguous"));
    }

    let (nrays, full_nbins) = mask.dim();
    if nbins.len() != nrays {
        return Err(PyValueError::new_err(
            "nbins length must match the number of rays",
        ));
    }
    if nbins.iter().any(|&value| value < 0) {
        return Err(PyValueError::new_err("nbins values must be non-negative"));
    }

    for ray in 0..nrays {
        let nbin = usize::try_from(nbins[ray])
            .map_err(|_| PyValueError::new_err("nbins values are too large"))?;
        if nbin >= full_nbins {
            continue;
        }
        for gate in nbin..full_nbins {
            mask[[ray, gate]] = 1;
        }
    }

    Ok(())
}

fn interpolate_scan_4(
    mut data: ArrayViewMut2<'_, f32>,
    mut scratch_ray: ArrayViewMut1<'_, f32>,
    fill_value: f32,
    start: isize,
    end: isize,
    moment_ngates: usize,
    linear_interp: bool,
) -> PyResult<()> {
    let interp_ngates = moment_ngates
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("moment_ngates is too large"))?;
    let Some((start, end)) = validate_scan_args(
        &data,
        &scratch_ray,
        start,
        end,
        moment_ngates,
        interp_ngates,
    )?
    else {
        return Ok(());
    };

    for ray_num in start..=end {
        for i in 0..moment_ngates {
            let gate_val = data[[ray_num, i]];
            scratch_ray[i * 4] = gate_val;
            scratch_ray[i * 4 + 1] = gate_val;
            scratch_ray[i * 4 + 2] = gate_val;
            scratch_ray[i * 4 + 3] = gate_val;
        }

        if linear_interp && interp_ngates > 4 {
            for i in (2..(interp_ngates - 4)).step_by(4) {
                let gate_val = scratch_ray[i];
                let next_val = scratch_ray[i + 4];
                if gate_val == fill_value || next_val == fill_value {
                    continue;
                }
                let delta = (next_val - gate_val) / 4.0;
                scratch_ray[i] = gate_val + delta * 0.5;
                scratch_ray[i + 1] = gate_val + delta * 1.5;
                scratch_ray[i + 2] = gate_val + delta * 2.5;
                scratch_ray[i + 3] = gate_val + delta * 3.5;
            }
        }

        for i in 0..interp_ngates {
            data[[ray_num, i]] = scratch_ray[i];
        }
    }

    Ok(())
}

fn interpolate_scan_2(
    mut data: ArrayViewMut2<'_, f32>,
    mut scratch_ray: ArrayViewMut1<'_, f32>,
    fill_value: f32,
    start: isize,
    end: isize,
    moment_ngates: usize,
    linear_interp: bool,
) -> PyResult<()> {
    let interp_ngates = if moment_ngates == 0 {
        0
    } else {
        moment_ngates
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| PyValueError::new_err("moment_ngates is too large"))?
    };
    let Some((start, end)) = validate_scan_args(
        &data,
        &scratch_ray,
        start,
        end,
        moment_ngates,
        interp_ngates,
    )?
    else {
        return Ok(());
    };

    for ray_num in start..=end {
        for i in 0..moment_ngates {
            let gate_val = data[[ray_num, i]];
            scratch_ray[i * 2] = gate_val;
            if i != moment_ngates - 1 {
                scratch_ray[i * 2 + 1] = gate_val;
            }
        }

        if linear_interp && interp_ngates > 2 {
            for i in (1..(interp_ngates - 2)).step_by(2) {
                let gate_val = scratch_ray[i];
                let next_val = scratch_ray[i + 2];
                if gate_val == fill_value || next_val == fill_value {
                    continue;
                }
                let delta = (next_val - gate_val) / 2.0;
                scratch_ray[i] = gate_val + delta * 0.5;
                scratch_ray[i + 1] = gate_val + delta * 1.5;
            }
        }

        for i in 0..interp_ngates {
            data[[ray_num, i]] = scratch_ray[i];
        }
    }

    Ok(())
}

fn validate_scan_args(
    data: &ArrayViewMut2<'_, f32>,
    scratch_ray: &ArrayViewMut1<'_, f32>,
    start: isize,
    end: isize,
    moment_ngates: usize,
    interp_ngates: usize,
) -> PyResult<Option<(usize, usize)>> {
    if start < 0 {
        return Err(PyValueError::new_err("start must be non-negative"));
    }
    if end < start {
        return Ok(None);
    }

    let start = start as usize;
    let end = end as usize;
    let (nrays, ngates) = data.dim();
    if end >= nrays {
        return Err(PyValueError::new_err("end ray is outside data"));
    }
    if moment_ngates > ngates {
        return Err(PyValueError::new_err("moment_ngates exceeds data gates"));
    }
    if interp_ngates > ngates {
        return Err(PyValueError::new_err(
            "interpolated gate count exceeds data gates",
        ));
    }
    if interp_ngates > scratch_ray.len() {
        return Err(PyValueError::new_err(
            "interpolated gate count exceeds scratch_ray",
        ));
    }

    Ok(Some((start, end)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn nexrad_level3_int16_to_float16_matches_reference_cases() {
        let cases = [
            (0, 0.0),
            (1, 0.001953125),
            (1023, 1.998046875),
            (1024, 3.0517578125e-05),
            (0x3c00, 0.5),
            (0x7bff, 32752.0),
            (0x8000, -0.0),
            (0x8400, -3.0517578125e-05),
            (0xffff, -65504.0),
            (-1, -65504.0),
            (-2, -65472.0),
            (-32768, -0.0),
            (-65536, 0.0),
        ];

        for (value, expected) in cases {
            let actual = nexrad_level3_int16_to_float16(value);
            assert_eq!(actual, expected);
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        }
    }

    #[test]
    fn nexrad_level2_scan_msgs_match_where_reference() {
        let scan_msgs = nexrad_level2_scan_msgs_i64(&[2_i64, 1, 2, 3]).unwrap();

        assert_eq!(scan_msgs, vec![vec![1_i64], vec![0_i64, 2], vec![3_i64]]);
    }

    #[test]
    fn nexrad_level2_scan_msgs_preserve_empty_missing_scans() {
        let scan_msgs = nexrad_level2_scan_msgs_i64(&[3_i64, 3]).unwrap();

        assert_eq!(scan_msgs, vec![Vec::<i64>::new(), Vec::new(), vec![0, 1]]);
    }

    #[test]
    fn nexrad_level2_scan_msgs_reject_empty_direct_input() {
        assert!(nexrad_level2_scan_msgs_i64(&[]).is_err());
    }

    #[test]
    fn nexrad_level2_msg_nums_preserve_order_duplicates_and_empty_scans() {
        let scan_msgs = vec![vec![0_i64, 2], Vec::new(), vec![1_i64]];

        assert_eq!(
            nexrad_level2_msg_nums_i64(&scan_msgs, &[2_i64, 0, 2]).unwrap(),
            vec![1_i64, 0, 2, 1]
        );
        assert_eq!(
            nexrad_level2_msg_nums_i64(&scan_msgs, &[1_i64]).unwrap(),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn nexrad_level2_msg_nums_reject_direct_invalid_scan_index() {
        let scan_msgs = vec![vec![0_i64]];

        assert!(nexrad_level2_msg_nums_i64(&scan_msgs, &[-1_i64]).is_err());
        assert!(nexrad_level2_msg_nums_i64(&scan_msgs, &[1_i64]).is_err());
    }

    #[test]
    fn interpolate_scan_4_keeps_fill_value_segments_nearest() {
        let fill = -9999.0_f32;
        let mut data: Array2<f32> = Array2::zeros((1, 16));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = fill;
        data[[0, 2]] = 5.0;
        data[[0, 3]] = 9.0;
        let mut scratch = Array1::<f32>::zeros(16);

        interpolate_scan_4(data.view_mut(), scratch.view_mut(), fill, 0, 0, 4, true).unwrap();

        assert_eq!(
            data.row(0).to_vec(),
            vec![
                1.0, 1.0, 1.0, 1.0, fill, fill, fill, fill, 5.0, 5.0, 5.5, 6.5, 7.5, 8.5, 9.0, 9.0,
            ]
        );
    }

    #[test]
    fn interpolate_scan_2_linear_matches_oracle_offsets() {
        let mut data: Array2<f32> = array![[10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0]];
        let mut scratch = Array1::<f32>::zeros(7);

        interpolate_scan_2(data.view_mut(), scratch.view_mut(), -9999.0, 0, 0, 4, true).unwrap();

        assert_eq!(
            data.row(0).to_vec(),
            vec![10.0, 12.5, 17.5, 22.5, 27.5, 30.0, 40.0]
        );
    }

    #[test]
    fn mask_gates_not_collected_marks_only_tail_bins() {
        let mut mask: Array2<u8> = Array2::zeros((4, 5));
        let nbins = ndarray::array![0_i64, 2, 5, 6];

        mask_gates_not_collected(mask.view_mut(), &nbins.view()).unwrap();

        assert_eq!(
            mask,
            ndarray::array![
                [1_u8, 1, 1, 1, 1],
                [0_u8, 0, 1, 1, 1],
                [0_u8, 0, 0, 0, 0],
                [0_u8, 0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn uf_sweep_limits_match_sorted_unique_first_last_reference() {
        let ray_sweep_numbers = [3_i32, 1, 3, 2, 1, 2, 2];

        let (first, last) = uf_sweep_limits_i32(&ray_sweep_numbers);

        assert_eq!(first, vec![1_i32, 3, 0]);
        assert_eq!(last, vec![4_i32, 6, 2]);
    }

    #[test]
    fn uf_sweep_limits_handles_empty_input() {
        let (first, last) = uf_sweep_limits_i32(&[]);

        assert!(first.is_empty());
        assert!(last.is_empty());
    }

    #[test]
    fn uf_ray_num_to_sweep_num_matches_reference_fill() {
        let starts = [2_i32, 0, 4];
        let ends = [3_i32, 1, 5];

        let ray_map = uf_ray_num_to_sweep_num_i32(7, &starts, &ends).unwrap();

        assert_eq!(ray_map, vec![1_i32, 1, 0, 0, 2, 2, 0]);
    }

    #[test]
    fn uf_ray_num_to_sweep_num_later_sweeps_overwrite_earlier_sweeps() {
        let starts = [0_i32, 2];
        let ends = [3_i32, 4];

        let ray_map = uf_ray_num_to_sweep_num_i32(5, &starts, &ends).unwrap();

        assert_eq!(ray_map, vec![0_i32, 0, 1, 1, 1]);
    }

    #[test]
    fn uf_ray_num_to_sweep_num_rejects_invalid_direct_bounds() {
        assert!(uf_ray_num_to_sweep_num_i32(3, &[0_i32], &[3_i32]).is_err());
        assert!(uf_ray_num_to_sweep_num_i32(3, &[-1_i32], &[1_i32]).is_err());
        assert!(uf_ray_num_to_sweep_num_i32(3, &[2_i32], &[1_i32]).is_err());
        assert!(uf_ray_num_to_sweep_num_i32(3, &[0_i32], &[1_i32, 2]).is_err());
    }

    #[test]
    fn mdv_decode_rle8_matches_reference_cases() {
        let cases = [
            (b"abc".as_slice(), 255_u8, 3_usize, b"abc".to_vec()),
            (&[255, 3, 7], 255_u8, 3_usize, vec![7, 7, 7]),
            (&[1, 255, 3, 7, 2], 255_u8, 5_usize, vec![1, 7, 7, 7, 2]),
            (&[255, 0, 9, 4], 255_u8, 1_usize, vec![4]),
        ];

        for (compr_data, key, decompr_size, expected) in cases {
            assert_eq!(
                mdv_decode_rle8_exact(compr_data, key, decompr_size).unwrap(),
                expected
            );
        }

        let long_literal: Vec<u8> = (0..300).map(|index| (index % 255) as u8).collect();
        assert_eq!(
            mdv_decode_rle8_exact(&long_literal, 255, long_literal.len()).unwrap(),
            long_literal
        );
    }

    #[test]
    fn mdv_decode_rle8_rejects_malformed_inputs() {
        assert!(mdv_decode_rle8_exact(&[255], 255, 1).is_err());
        assert!(mdv_decode_rle8_exact(b"abcd", 255, 2).is_err());
        assert!(mdv_decode_rle8_exact(&[255, 4, 9], 255, 2).is_err());
        assert!(mdv_decode_rle8_exact(&[255, 255, 7, 8], 255, 256).is_err());
        assert!(mdv_decode_rle8_exact(b"ab", 255, 5).is_err());
    }

    #[test]
    fn nexrad_af1f_decode_rle_matches_reference_cases() {
        let cases = [
            (&[0x31][..], 3_usize, vec![1, 1, 1]),
            (&[0x21, 0x12, 0x03, 0x24], 5_usize, vec![1, 1, 2, 4, 4]),
            (&[0x00, 0x15], 1_usize, vec![5]),
            (&[][..], 0_usize, vec![]),
        ];

        for (rle_data, nbins, expected) in cases {
            assert_eq!(
                nexrad_af1f_decode_rle_exact(rle_data, nbins).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn nexrad_af1f_decode_rle_rejects_mismatched_lengths() {
        assert!(nexrad_af1f_decode_rle_exact(&[0x11], 2).is_err());
        assert!(nexrad_af1f_decode_rle_exact(&[0x31], 2).is_err());
    }

    #[test]
    fn nexrad_msg_135_matches_uint8_wraparound_reference() {
        let raw = ndarray::array![[0_u8, 1, 2, 127, 128, 255]];

        let (data, mask) = nexrad_level3_msg_135(&raw.view());

        assert_eq!(
            data,
            ndarray::array![[254.0_f32, 255.0, 0.0, 125.0, 254.0, 125.0]]
        );
        assert_eq!(
            mask,
            ndarray::array![[true, true, false, false, false, false]]
        );
    }

    #[test]
    fn nexrad_msg_138_matches_linear_scaling_reference() {
        let raw = ndarray::array![[0_u8, 1, 255]];
        let threshold_data = [0_u8, 100, 0, 5];

        let data = nexrad_level3_msg_138(&threshold_data, &raw.view()).unwrap();

        assert_eq!(data, ndarray::array![[1.0_f32, 1.05, 13.75]]);
    }

    #[test]
    fn nexrad_msg_32_matches_masked_linear_scaling_reference() {
        let raw = ndarray::array![[0_u16, 1, 2, 255]];
        let threshold_data = [0_u8, 10, 0, 5];

        let (data, mask) =
            nexrad_level3_msg_scaled_u16(&threshold_data, &raw.view(), false).unwrap();

        assert_eq!(data, ndarray::array![[1.0_f32, 1.5, 2.0, 128.5]]);
        assert_eq!(mask, ndarray::array![[true, true, false, false]]);
    }

    #[test]
    fn nexrad_scaled_sub2_matches_unsigned_wraparound_reference() {
        let raw_u8 = ndarray::array![[0_u8, 1, 2, 255]];
        let raw_u16 = ndarray::array![[0_u16, 1, 2, 255]];
        let threshold_data = [0_u8, 10, 0, 5];

        let (data_u8, mask_u8) =
            nexrad_level3_msg_scaled_u8(&threshold_data, &raw_u8.view(), true).unwrap();
        let (data_u16, mask_u16) =
            nexrad_level3_msg_scaled_u16(&threshold_data, &raw_u16.view(), true).unwrap();

        assert_eq!(data_u8, ndarray::array![[128.0_f32, 128.5, 1.0, 127.5]]);
        assert_eq!(
            data_u16,
            ndarray::array![[32768.0_f32, 32768.5, 1.0, 127.5]]
        );
        assert_eq!(mask_u8, ndarray::array![[true, true, false, false]]);
        assert_eq!(mask_u16, ndarray::array![[true, true, false, false]]);
    }

    #[test]
    fn nexrad_mask_zero_and_copy_match_reference() {
        let raw = ndarray::array![[0_u16, 1, 255, 65535]];

        let (data, mask) = nexrad_level3_mask_zero_u16(&raw.view());
        let copied = nexrad_level3_copy_u16(&raw.view());

        assert_eq!(data, ndarray::array![[0.0_f32, 1.0, 255.0, 65535.0]]);
        assert_eq!(copied, data);
        assert_eq!(mask, ndarray::array![[true, false, false, false]]);
    }

    #[test]
    fn sigmet_data_types_from_mask_matches_low_to_high_bit_order() {
        assert_eq!(
            sigmet_data_types_from_mask([0b101, 0b10, 0, 0x8000_0000]),
            vec![0, 2, 33, 127]
        );
        assert_eq!(
            sigmet_data_types_from_mask([0, 0, 0, 0]),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn sigmet_time_order_helpers_keep_zero_one_sweep_start_behavior() {
        let rays = [1_i64, 3];

        assert!(sigmet_time_ordered_by_reversal_i32(&[3_i32, 2, 1, 10], &rays).unwrap());
        assert!(sigmet_time_ordered_by_roll_i32(&[2_i32, 3, 1, 10], &rays).unwrap());
        assert!(sigmet_time_ordered_by_reverse_roll_i32(&[3_i32, 2, 1, 10], &rays).unwrap());
    }

    #[test]
    fn sigmet_time_order_helpers_match_reference_false_cases() {
        assert!(!sigmet_time_ordered_by_reversal_i32(&[0_i32, 2, 1, 3], &[4_i64]).unwrap());
        assert!(!sigmet_time_ordered_by_roll_i32(&[0_i32, 3, 1, 2], &[4_i64]).unwrap());
        assert!(!sigmet_time_ordered_by_reverse_roll_i32(&[0_i32, 3, 1, 2], &[4_i64]).unwrap());
    }

    #[test]
    fn sigmet_time_order_helpers_match_int32_wrapping_subtraction() {
        let ref_time = [i32::MIN, i32::MAX];
        let rays = [2_i64];

        assert!(sigmet_time_ordered_by_roll_i32(&ref_time, &rays).unwrap());
        assert!(sigmet_time_ordered_by_reverse_roll_i32(&ref_time, &rays).unwrap());
    }

    #[test]
    fn sigmet_time_order_helpers_reject_direct_out_of_bounds_and_negative_rays() {
        assert!(sigmet_time_ordered_by_reversal_i32(&[1_i32, 2], &[3_i64]).is_err());
        assert!(sigmet_time_ordered_by_roll_i32(&[1_i32, 2], &[3_i64]).is_err());
        assert!(sigmet_time_ordered_by_reverse_roll_i32(&[1_i32, 2], &[3_i64]).is_err());
        assert!(sigmet_time_ordered_by_roll_i32(&[1_i32, 2], &[-1_i64]).is_err());
    }

    #[test]
    fn sigmet_time_order_index_helpers_match_reference_orders() {
        assert_eq!(
            sigmet_time_order_roll_index_i32(&[2_i32, 3, 1, 10], &[1_i64, 3]).unwrap(),
            vec![2_i64, 0, 1, 3]
        );
        assert_eq!(
            sigmet_time_order_reverse_index_i32(&[3_i32, 2, 1, 10], &[1_i64, 3]).unwrap(),
            vec![2_i64, 1, 0, 3]
        );
        assert_eq!(
            sigmet_time_order_full_index_i32(&[2_i32, 1, 1, 5], &[4_i64]).unwrap(),
            vec![1_i64, 2, 0, 3]
        );
    }

    #[test]
    fn sigmet_time_order_index_helpers_reject_direct_out_of_bounds() {
        assert!(sigmet_time_order_roll_index_i32(&[1_i32, 2], &[3_i64]).is_err());
        assert!(sigmet_time_order_reverse_index_i32(&[1_i32, 2], &[3_i64]).is_err());
        assert!(sigmet_time_order_full_index_i32(&[1_i32, 2], &[3_i64]).is_err());
    }

    #[test]
    fn sigmet_like_dbt2_conversion_matches_reference() {
        let data = ndarray::array![[0_u16, 32_768, 32_769, 65_535], [100_u16, 200, 300, 400]]
            .mapv(|value| value as i16);
        let nbins = ndarray::array![4_i64, 2];

        let (out, mask) = sigmet_convert_like_dbt2_dense_i16(&data.view(), &nbins.view()).unwrap();

        assert_eq!(
            out,
            ndarray::array![
                [-327.68_f32, 0.0, 0.01, 327.67],
                [-326.68, -325.68, -324.68, -323.68]
            ]
        );
        assert_eq!(
            mask,
            ndarray::array![[true, false, false, false], [false, false, true, true]]
        );
    }

    #[test]
    fn sigmet_like_dbt_conversion_matches_native_byte_view_reference() {
        let data = ndarray::array![
            [0x0000_u16, 0x0101, 0x4001, 0x8040],
            [0x00ff_u16, 0xff00, 0x4142, 0x4243]
        ]
        .mapv(|value| value as i16);
        let nbins = ndarray::array![4_i64, 2];

        let (out, mask) = sigmet_convert_like_dbt_dense_i16(&data.view(), &nbins.view()).unwrap();

        assert_eq!(
            out,
            ndarray::array![[-32.0_f32, -32.0, -31.5, -31.5], [95.5, -32.0, -32.0, 95.5]]
        );
        assert_eq!(
            mask,
            ndarray::array![[true, true, false, false], [false, true, true, true]]
        );
    }

    #[test]
    fn sigmet_ray_current_record_decode_matches_reference_runs() {
        let mut rbuf = Array1::<i16>::ones(SIGMET_RECORD_WORDS);
        rbuf[10] = 2;
        rbuf[11] = -32765;
        rbuf[12] = 10;
        rbuf[13] = 11;
        rbuf[14] = 12;
        rbuf[15] = 1;
        let mut out = Array1::<i16>::ones(14);

        let result =
            sigmet_decode_ray_current_record_i16(&rbuf.view(), 9, 8, &mut out.view_mut()).unwrap();

        assert_eq!(result, Some((0, 15)));
        assert_eq!(
            out.to_vec(),
            vec![0, 0, 10, 11, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        );
    }

    #[test]
    fn sigmet_ray_current_record_decode_handles_missing_corrupt_and_split() {
        let mut missing = Array1::<i16>::ones(SIGMET_RECORD_WORDS);
        missing[10] = 1;
        let mut missing_out = Array1::<i16>::ones(14);
        let missing_result = sigmet_decode_ray_current_record_i16(
            &missing.view(),
            9,
            8,
            &mut missing_out.view_mut(),
        )
        .unwrap();
        assert_eq!(missing_result, Some((0, 10)));
        assert_eq!(missing_out[4], -1);

        let mut corrupt = Array1::<i16>::ones(SIGMET_RECORD_WORDS);
        corrupt[10] = 20;
        let mut corrupt_out = Array1::<i16>::ones(10);
        let corrupt_result = sigmet_decode_ray_current_record_i16(
            &corrupt.view(),
            9,
            4,
            &mut corrupt_out.view_mut(),
        )
        .unwrap();
        assert_eq!(corrupt_result, Some((-1, 11)));
        assert_eq!(corrupt_out.to_vec(), vec![1; 10]);

        let mut split = Array1::<i16>::ones(SIGMET_RECORD_WORDS);
        split[3070] = -32766;
        split[3071] = 99;
        let mut split_out = Array1::<i16>::ones(12);
        let split_result =
            sigmet_decode_ray_current_record_i16(&split.view(), 3069, 6, &mut split_out.view_mut())
                .unwrap();
        assert_eq!(split_result, None);
        assert_eq!(split_out.to_vec(), vec![1; 12]);
    }

    #[test]
    fn sigmet_u8_modes_match_native_byte_view_reference() {
        let data = ndarray::array![
            [
                i16::from_ne_bytes([0, 1]),
                i16::from_ne_bytes([127, 128]),
                i16::from_ne_bytes([255, 0]),
                0,
                0
            ],
            [
                i16::from_ne_bytes([255, 0]),
                i16::from_ne_bytes([1, 2]),
                i16::from_ne_bytes([3, 0]),
                0,
                0
            ]
        ];
        let raw = ndarray::array![[0_u8, 1, 127, 128, 255], [255_u8, 0, 1, 2, 3]];
        let nbins = ndarray::array![5_i64, 2];

        let (sqi, sqi_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::LikeSqi)
                .unwrap();
        let (vel, vel_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::Vel).unwrap();
        let (velc, velc_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::VelC).unwrap();
        let (width, width_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::Width).unwrap();
        let (zdr, zdr_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::Zdr).unwrap();
        let (kdp, kdp_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::Kdp).unwrap();
        let (phidp, phidp_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::PhiDp).unwrap();
        let (hclass, hclass_mask) =
            sigmet_convert_u8_dense_i16(&data.view(), &nbins.view(), SigmetU8Mode::HClass).unwrap();

        for (&actual, &expected) in sqi.iter().zip(
            raw.mapv(|value| ((f64::from(value) - 1.0) / 253.0).sqrt() as f32)
                .iter(),
        ) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, expected);
            }
        }
        assert_eq!(
            vel,
            raw.mapv(|value| ((f64::from(value) - 128.0) / 127.0) as f32)
        );
        assert_eq!(
            velc,
            raw.mapv(|value| (((f64::from(value) - 128.0) / 127.0) * 75.0) as f32)
        );
        assert_eq!(width, raw.mapv(|value| (f64::from(value) / 256.0) as f32));
        assert_eq!(
            zdr,
            raw.mapv(|value| ((f64::from(value) - 128.0) / 16.0) as f32)
        );
        assert_eq!(kdp, raw.mapv(sigmet_kdp_u8_value));
        assert_eq!(
            phidp,
            raw.mapv(|value| (180.0 * ((f64::from(value) - 1.0) / 254.0)) as f32)
        );
        assert_eq!(hclass, raw.mapv(f32::from));
        assert_eq!(
            vel_mask,
            ndarray::array![
                [true, false, false, false, false],
                [false, true, true, true, true]
            ]
        );
        assert_eq!(width_mask, vel_mask);
        assert_eq!(zdr_mask, vel_mask);
        assert_eq!(
            velc_mask,
            ndarray::array![
                [true, false, false, false, true],
                [true, true, true, true, true]
            ]
        );
        assert_eq!(sqi_mask, velc_mask);
        assert_eq!(phidp_mask, velc_mask);
        assert_eq!(kdp_mask, velc_mask);
        assert_eq!(hclass_mask, velc_mask);
    }

    #[test]
    fn sigmet_u16_modes_match_reference_formulas() {
        let data = ndarray::array![[0_u16, 1, 32_768, 65_535], [2_u16, 3, 4, 5]]
            .mapv(|value| value as i16);
        let nbins = ndarray::array![4_i64, 2];

        let (sqi, sqi_mask) =
            sigmet_convert_u16_dense_i16(&data.view(), &nbins.view(), SigmetU16Mode::LikeSqi2)
                .unwrap();
        let (width, width_mask) =
            sigmet_convert_u16_dense_i16(&data.view(), &nbins.view(), SigmetU16Mode::Width2)
                .unwrap();
        let (phidp, phidp_mask) =
            sigmet_convert_u16_dense_i16(&data.view(), &nbins.view(), SigmetU16Mode::PhiDp2)
                .unwrap();
        let (hclass, hclass_mask) =
            sigmet_convert_u16_dense_i16(&data.view(), &nbins.view(), SigmetU16Mode::HClass2)
                .unwrap();

        assert_eq!(
            sqi,
            data.mapv(|raw| ((f64::from(raw as u16) - 1.0) / 65_533.0) as f32)
        );
        assert_eq!(
            width,
            data.mapv(|raw| (f64::from(raw as u16) / 100.0) as f32)
        );
        assert_eq!(
            phidp,
            data.mapv(|raw| (360.0 * (f64::from(raw as u16) - 1.0) / 65_534.0) as f32)
        );
        assert_eq!(hclass, data.mapv(|raw| f32::from(raw as u16)));
        assert_eq!(
            sqi_mask,
            ndarray::array![[true, false, false, false], [false, false, true, true]]
        );
        assert_eq!(width_mask, sqi_mask);
        assert_eq!(phidp_mask, sqi_mask);
        assert_eq!(
            hclass_mask,
            ndarray::array![[false, false, false, false], [false, false, true, true]]
        );
    }
}
