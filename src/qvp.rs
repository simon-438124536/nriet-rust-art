use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyfunction(name = "_qvp_find_index_dense")]
pub fn qvp_find_index_dense<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    target: f64,
    tolerance: f64,
) -> PyResult<Option<Py<PyAny>>> {
    let values = values.as_array();
    validate_find_index_inputs(&values, target, tolerance)?;
    let index = find_index_kernel(&values, target, tolerance);
    match index {
        Some(index) => {
            let np = py.import("numpy")?;
            Ok(Some(np.getattr("int64")?.call1((index,))?.unbind()))
        }
        None => Ok(None),
    }
}

#[pyfunction(name = "_qvp_find_neighbour_gates_dense")]
pub fn qvp_find_neighbour_gates_dense<'py>(
    py: Python<'py>,
    azimuths: PyReadonlyArray1<'py, f64>,
    ranges: PyReadonlyArray1<'py, f64>,
    azi: f64,
    rng: f64,
    delta_azi: f64,
    delta_rng: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let azimuths = azimuths.as_array();
    let ranges = ranges.as_array();
    validate_find_neighbour_inputs(&azimuths, &ranges, azi, rng, delta_azi, delta_rng)?;
    let (ray_indices, range_indices) =
        find_neighbour_gates_kernel(&azimuths, &ranges, azi, rng, delta_azi, delta_rng)?;
    Ok((
        PyArray1::from_owned_array(py, Array1::from_vec(ray_indices)),
        PyArray1::from_owned_array(py, Array1::from_vec(range_indices)),
    ))
}

#[pyfunction(name = "_qvp_find_nearest_gate_dense")]
pub fn qvp_find_nearest_gate_dense<'py>(
    py: Python<'py>,
    gate_latitude: PyReadonlyArray2<'py, f64>,
    gate_longitude: PyReadonlyArray2<'py, f64>,
    lat: f64,
    lon: f64,
    latlon_tol: f64,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let gate_latitude = gate_latitude.as_array();
    let gate_longitude = gate_longitude.as_array();
    validate_find_nearest_gate_inputs(&gate_latitude, &gate_longitude, lat, lon, latlon_tol)?;
    match find_nearest_gate_kernel(&gate_latitude, &gate_longitude, lat, lon, latlon_tol)? {
        Some((ray, gate)) => {
            let np = py.import("numpy")?;
            Ok(Some((
                np.getattr("int64")?.call1((ray,))?.unbind(),
                np.getattr("int64")?.call1((gate,))?.unbind(),
            )))
        }
        None => Ok(None),
    }
}

#[pyfunction(name = "_qvp_project_to_vertical_none_dense_f64")]
pub fn qvp_project_to_vertical_none_dense_f64<'py>(
    py: Python<'py>,
    data_in: &Bound<'py, PyAny>,
    data_height: &Bound<'py, PyAny>,
    grid_height: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>)> {
    reject_masked_array(py, "data_in", data_in)?;
    reject_masked_array(py, "data_height", data_height)?;
    reject_masked_array(py, "grid_height", grid_height)?;

    let data_in = data_in
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("data_in must be a 1D float64 array"))?;
    let data_height = data_height
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("data_height must be a 1D float64 array"))?;
    let grid_height = grid_height
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("grid_height must be a 1D float64 array"))?;

    let data_in = data_in.as_array();
    let data_height = data_height.as_array();
    let grid_height = grid_height.as_array();
    validate_project_to_vertical_none_inputs(&data_in, &data_height, &grid_height)?;
    let (data, mask) = project_to_vertical_none_kernel(&data_in, &data_height, &grid_height);
    Ok((
        PyArray1::from_owned_array(py, Array1::from_vec(data)),
        PyArray1::from_owned_array(py, Array1::from_vec(mask)),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(qvp_find_index_dense, module)?)?;
    module.add_function(wrap_pyfunction!(qvp_find_neighbour_gates_dense, module)?)?;
    module.add_function(wrap_pyfunction!(qvp_find_nearest_gate_dense, module)?)?;
    module.add_function(wrap_pyfunction!(
        qvp_project_to_vertical_none_dense_f64,
        module
    )?)?;
    Ok(())
}

fn reject_masked_array(py: Python<'_>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
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

fn validate_find_index_inputs(
    values: &ndarray::ArrayView1<'_, f64>,
    target: f64,
    tolerance: f64,
) -> PyResult<()> {
    if values.is_empty() {
        return Err(PyValueError::new_err(
            "attempt to get argmin of an empty sequence",
        ));
    }
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }
    if !target.is_finite() || !tolerance.is_finite() {
        return Err(PyValueError::new_err("target and tolerance must be finite"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(())
}

fn validate_find_neighbour_inputs(
    azimuths: &ndarray::ArrayView1<'_, f64>,
    ranges: &ndarray::ArrayView1<'_, f64>,
    azi: f64,
    rng: f64,
    delta_azi: f64,
    delta_rng: f64,
) -> PyResult<()> {
    if !azimuths.is_standard_layout() || !ranges.is_standard_layout() {
        return Err(PyValueError::new_err(
            "azimuths and ranges must be C-contiguous",
        ));
    }
    if !(azi.is_finite() && rng.is_finite() && delta_azi.is_finite() && delta_rng.is_finite()) {
        return Err(PyValueError::new_err(
            "azi, rng, delta_azi, and delta_rng must be finite",
        ));
    }
    if azimuths.iter().any(|value| !value.is_finite())
        || ranges.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err("azimuths and ranges must be finite"));
    }
    Ok(())
}

fn validate_find_nearest_gate_inputs(
    gate_latitude: &ndarray::ArrayView2<'_, f64>,
    gate_longitude: &ndarray::ArrayView2<'_, f64>,
    lat: f64,
    lon: f64,
    latlon_tol: f64,
) -> PyResult<()> {
    if gate_latitude.dim() != gate_longitude.dim() {
        return Err(PyValueError::new_err(
            "gate_latitude and gate_longitude must have the same shape",
        ));
    }
    if !gate_latitude.is_standard_layout() || !gate_longitude.is_standard_layout() {
        return Err(PyValueError::new_err(
            "gate_latitude and gate_longitude must be C-contiguous",
        ));
    }
    if !(lat.is_finite() && lon.is_finite() && latlon_tol.is_finite()) {
        return Err(PyValueError::new_err(
            "lat, lon, and latlon_tol must be finite",
        ));
    }
    if gate_latitude.iter().any(|value| !value.is_finite())
        || gate_longitude.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "gate_latitude and gate_longitude must be finite",
        ));
    }
    Ok(())
}

fn validate_project_to_vertical_none_inputs(
    data_in: &ndarray::ArrayView1<'_, f64>,
    data_height: &ndarray::ArrayView1<'_, f64>,
    grid_height: &ndarray::ArrayView1<'_, f64>,
) -> PyResult<()> {
    if data_in.is_empty() {
        return Err(PyValueError::new_err("data_in must be non-empty"));
    }
    if data_in.len() != data_height.len() {
        return Err(PyValueError::new_err(
            "data_in and data_height must have the same length",
        ));
    }
    if grid_height.len() < 2 {
        return Err(PyValueError::new_err(
            "grid_height must contain at least two points",
        ));
    }
    if !data_in.is_standard_layout()
        || !data_height.is_standard_layout()
        || !grid_height.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "data_in, data_height, and grid_height must be C-contiguous",
        ));
    }
    if data_in.iter().any(|value| !value.is_finite())
        || data_height.iter().any(|value| !value.is_finite())
        || grid_height.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "data_in, data_height, and grid_height must be finite",
        ));
    }
    Ok(())
}

fn find_index_kernel(
    values: &ndarray::ArrayView1<'_, f64>,
    target: f64,
    tolerance: f64,
) -> Option<usize> {
    let mut best_index = 0_usize;
    let mut best_distance = (values[0] - target).abs();
    for (index, value) in values.iter().enumerate().skip(1) {
        let distance = (*value - target).abs();
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }

    if best_distance > tolerance {
        None
    } else {
        Some(best_index)
    }
}

fn project_to_vertical_none_kernel(
    data_in: &ndarray::ArrayView1<'_, f64>,
    data_height: &ndarray::ArrayView1<'_, f64>,
    grid_height: &ndarray::ArrayView1<'_, f64>,
) -> (Vec<f64>, Vec<bool>) {
    let tolerance = (grid_height[1] - grid_height[0]) / 2.0;
    let mut data = vec![0.0; grid_height.len()];
    let mut mask = vec![true; grid_height.len()];

    for (grid_index, &height) in grid_height.iter().enumerate() {
        if let Some(data_index) = find_index_kernel(data_height, height, tolerance) {
            data[grid_index] = data_in[data_index];
            mask[grid_index] = false;
        }
    }

    (data, mask)
}

fn find_neighbour_gates_kernel(
    azimuths: &ndarray::ArrayView1<'_, f64>,
    ranges: &ndarray::ArrayView1<'_, f64>,
    azi: f64,
    rng: f64,
    delta_azi: f64,
    delta_rng: f64,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let mut azi_max = azi + delta_azi;
    let mut azi_min = azi - delta_azi;
    if azi_max > 360.0 {
        azi_max -= 360.0;
    }
    if azi_min < 0.0 {
        azi_min += 360.0;
    }

    let mut ray_indices = Vec::new();
    for (index, value) in azimuths.iter().enumerate() {
        let inside = if azi_max > azi_min {
            *value < azi_max && *value > azi_min
        } else {
            *value > azi_min || *value < azi_max
        };
        if inside {
            ray_indices.push(index_to_i64(index)?);
        }
    }

    let lower_rng = rng - delta_rng;
    let upper_rng = rng + delta_rng;
    let mut range_indices = Vec::new();
    for (index, value) in ranges.iter().enumerate() {
        if *value < upper_rng && *value > lower_rng {
            range_indices.push(index_to_i64(index)?);
        }
    }

    Ok((ray_indices, range_indices))
}

fn find_nearest_gate_kernel(
    gate_latitude: &ndarray::ArrayView2<'_, f64>,
    gate_longitude: &ndarray::ArrayView2<'_, f64>,
    lat: f64,
    lon: f64,
    latlon_tol: f64,
) -> PyResult<Option<(i64, i64)>> {
    let lat_upper = lat + latlon_tol;
    let lat_lower = lat - latlon_tol;
    let lon_upper = lon + latlon_tol;
    let lon_lower = lon - latlon_tol;
    let mut best: Option<(usize, usize, f64)> = None;

    for ((ray, gate), gate_lat) in gate_latitude.indexed_iter() {
        if !(*gate_lat < lat_upper && *gate_lat > lat_lower) {
            continue;
        }
        let gate_lon = gate_longitude[[ray, gate]];
        if !(gate_lon < lon_upper && gate_lon > lon_lower) {
            continue;
        }
        let distance = (*gate_lat - lat).abs();
        match best {
            Some((_, _, best_distance)) if distance >= best_distance => {}
            _ => best = Some((ray, gate, distance)),
        }
    }

    match best {
        Some((ray, gate, _)) => Ok(Some((index_to_i64(ray)?, index_to_i64(gate)?))),
        None => Ok(None),
    }
}

fn index_to_i64(index: usize) -> PyResult<i64> {
    i64::try_from(index).map_err(|_| PyValueError::new_err("index exceeds int64 range"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn find_index_keeps_first_tie_and_tolerance_boundary() {
        let values = array![1.0, 2.0, 2.0, 3.0];
        assert_eq!(find_index_kernel(&values.view(), 2.0, 0.0), Some(1));
        assert_eq!(find_index_kernel(&values.view(), 2.6, 0.4), Some(3));
        assert_eq!(find_index_kernel(&values.view(), 2.6, 0.399), None);
    }

    #[test]
    fn find_index_validation_rejects_empty_values() {
        let values = ndarray::Array1::<f64>::zeros(0);
        assert!(validate_find_index_inputs(&values.view(), 1.0, 0.0).is_err());
    }

    #[test]
    fn project_to_vertical_none_matches_half_grid_tolerance() {
        let data = array![10.0, 20.0, 30.0];
        let data_height = array![0.0, 100.0, 200.0];
        let grid_height = array![0.0, 50.0, 100.0, 150.0, 200.0, 250.0];

        let (projected, mask) =
            project_to_vertical_none_kernel(&data.view(), &data_height.view(), &grid_height.view());

        assert_eq!(projected, vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0]);
        assert_eq!(mask, vec![false, true, false, true, false, true]);
    }

    #[test]
    fn project_to_vertical_none_keeps_first_tie() {
        let data = array![10.0, 20.0];
        let data_height = array![0.0, 100.0];
        let grid_height = array![50.0, 150.0];

        let (projected, mask) =
            project_to_vertical_none_kernel(&data.view(), &data_height.view(), &grid_height.view());

        assert_eq!(projected, vec![10.0, 20.0]);
        assert_eq!(mask, vec![false, false]);
    }

    #[test]
    fn neighbour_gates_handles_strict_bounds_and_wraparound() {
        let azimuths = array![350.0, 355.0, 0.0, 5.0, 10.0, 20.0];
        let ranges = array![0.0, 500.0, 1000.0, 1500.0, 2000.0];

        let (rays, gates) =
            find_neighbour_gates_kernel(&azimuths.view(), &ranges.view(), 0.0, 1000.0, 10.0, 500.0)
                .unwrap();

        assert_eq!(rays, vec![1, 2, 3]);
        assert_eq!(gates, vec![2]);
    }

    #[test]
    fn neighbour_gates_handles_non_wrapped_and_negative_delta_like_python() {
        let azimuths = array![0.0, 5.0, 10.0, 15.0, 20.0];
        let ranges = array![0.0, 500.0, 1000.0, 1500.0];

        let (rays, gates) =
            find_neighbour_gates_kernel(&azimuths.view(), &ranges.view(), 10.0, 750.0, 6.0, -100.0)
                .unwrap();

        assert_eq!(rays, vec![1, 2, 3]);
        assert_eq!(gates, Vec::<i64>::new());
    }
}
