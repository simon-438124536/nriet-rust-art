use ndarray::{Array1, ArrayD, IxDyn, Zip};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const EFFECTIVE_EARTH_RADIUS_M: f64 = 6371.0 * 1000.0 * 4.0 / 3.0;

fn antenna_to_cartesian_kernel(
    range_km: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
) -> (f64, f64, f64) {
    let theta_e = elevation_deg.to_radians();
    let theta_a = azimuth_deg.to_radians();
    let r = range_km * 1000.0;

    let z = (r.powi(2)
        + EFFECTIVE_EARTH_RADIUS_M.powi(2)
        + 2.0 * r * EFFECTIVE_EARTH_RADIUS_M * theta_e.sin())
    .sqrt()
        - EFFECTIVE_EARTH_RADIUS_M;
    let s = EFFECTIVE_EARTH_RADIUS_M * (r * theta_e.cos() / (EFFECTIVE_EARTH_RADIUS_M + z)).asin();
    let x = s * theta_a.sin();
    let y = s * theta_a.cos();
    (x, y, z)
}

fn cartesian_to_antenna_kernel(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let range = (x.powf(2.0) + y.powf(2.0) + z.powf(2.0)).sqrt();
    let elevation = (z / (x.powf(2.0) + y.powf(2.0)).sqrt()).atan().to_degrees();
    let mut azimuth = x.atan2(y).to_degrees();
    if azimuth < 0.0 {
        azimuth += 360.0;
    }
    (range, azimuth, elevation)
}

fn geographic_to_cartesian_aeqd_kernel(
    lon_deg: f64,
    lat_deg: f64,
    lon_0_deg: f64,
    lat_0_deg: f64,
    radius: f64,
) -> (f64, f64) {
    let lon_rad = lon_deg.to_radians();
    let lat_rad = lat_deg.to_radians();
    let lon_0_rad = lon_0_deg.to_radians();
    let lat_0_rad = lat_0_deg.to_radians();
    let lon_diff_rad = lon_rad - lon_0_rad;

    let mut arg_arccos =
        lat_0_rad.sin() * lat_rad.sin() + lat_0_rad.cos() * lat_rad.cos() * lon_diff_rad.cos();
    arg_arccos = arg_arccos.clamp(-1.0, 1.0);

    let c = arg_arccos.acos();
    let k = if c == 0.0 { 1.0 } else { c / c.sin() };
    let x = radius * k * lat_rad.cos() * lon_diff_rad.sin();
    let y = radius
        * k
        * (lat_0_rad.cos() * lat_rad.sin() - lat_0_rad.sin() * lat_rad.cos() * lon_diff_rad.cos());
    (x, y)
}

fn cartesian_to_geographic_aeqd_kernel(
    x: f64,
    y: f64,
    lon_0_deg: f64,
    lat_0_deg: f64,
    radius: f64,
) -> (f64, f64) {
    let lat_0_rad = lat_0_deg.to_radians();
    let lon_0_rad = lon_0_deg.to_radians();
    let rho = (x * x + y * y).sqrt();
    let c = rho / radius;

    let lat_deg = if rho == 0.0 {
        lat_0_deg
    } else {
        let lat_rad = (c.cos() * lat_0_rad.sin() + y * c.sin() * lat_0_rad.cos() / rho).asin();
        lat_rad.to_degrees()
    };

    let x1 = x * c.sin();
    let x2 = rho * lat_0_rad.cos() * c.cos() - y * lat_0_rad.sin() * c.sin();
    let mut lon_deg = (lon_0_rad + x1.atan2(x2)).to_degrees();
    if lon_deg > 180.0 {
        lon_deg -= 360.0;
    }
    if lon_deg < -180.0 {
        lon_deg += 360.0;
    }

    (lon_deg, lat_deg)
}

#[pyfunction(name = "_antenna_to_cartesian")]
fn py_antenna_to_cartesian<'py>(
    py: Python<'py>,
    ranges: PyReadonlyArrayDyn<'py, f64>,
    azimuths: PyReadonlyArrayDyn<'py, f64>,
    elevations: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
)> {
    let ranges = ranges.as_array();
    let azimuths = azimuths.as_array();
    let elevations = elevations.as_array();
    let shape = broadcast_shape(&[ranges.shape(), azimuths.shape(), elevations.shape()])?;

    let ranges = ranges
        .broadcast(IxDyn(&shape))
        .ok_or_else(broadcast_error)?;
    let azimuths = azimuths
        .broadcast(IxDyn(&shape))
        .ok_or_else(broadcast_error)?;
    let elevations = elevations
        .broadcast(IxDyn(&shape))
        .ok_or_else(broadcast_error)?;

    let mut x_out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let mut y_out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let mut z_out = ArrayD::<f64>::zeros(IxDyn(&shape));

    Zip::from(&mut x_out)
        .and(&mut y_out)
        .and(&mut z_out)
        .and(ranges)
        .and(azimuths)
        .and(elevations)
        .for_each(|x_slot, y_slot, z_slot, &range, &azimuth, &elevation| {
            let (x, y, z) = antenna_to_cartesian_kernel(range, azimuth, elevation);
            *x_slot = x;
            *y_slot = y;
            *z_slot = z;
        });

    Ok((
        PyArrayDyn::from_owned_array(py, x_out),
        PyArrayDyn::from_owned_array(py, y_out),
        PyArrayDyn::from_owned_array(py, z_out),
    ))
}

#[pyfunction(name = "_cartesian_to_antenna")]
fn py_cartesian_to_antenna<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
    z: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
)> {
    let x = x.as_array();
    let y = y.as_array();
    let z = z.as_array();
    let shape = broadcast_shape(&[x.shape(), y.shape(), z.shape()])?;

    let x = x.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;
    let y = y.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;
    let z = z.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;

    let mut ranges_out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let mut azimuths_out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let mut elevations_out = ArrayD::<f64>::zeros(IxDyn(&shape));

    Zip::from(&mut ranges_out)
        .and(&mut azimuths_out)
        .and(&mut elevations_out)
        .and(x)
        .and(y)
        .and(z)
        .for_each(|range_slot, azimuth_slot, elevation_slot, &x, &y, &z| {
            let (range, azimuth, elevation) = cartesian_to_antenna_kernel(x, y, z);
            *range_slot = range;
            *azimuth_slot = azimuth;
            *elevation_slot = elevation;
        });

    Ok((
        PyArrayDyn::from_owned_array(py, ranges_out),
        PyArrayDyn::from_owned_array(py, azimuths_out),
        PyArrayDyn::from_owned_array(py, elevations_out),
    ))
}

#[pyfunction(name = "_geographic_to_cartesian_aeqd")]
fn py_geographic_to_cartesian_aeqd<'py>(
    py: Python<'py>,
    lon: PyReadonlyArrayDyn<'py, f64>,
    lat: PyReadonlyArrayDyn<'py, f64>,
    lon_0: f64,
    lat_0: f64,
    radius: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let lon = lon.as_array();
    let lat = lat.as_array();
    let shape = broadcast_shape(&[lon.shape(), lat.shape()])?;

    let lon = lon.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;
    let lat = lat.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;

    let mut x_out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let mut y_out = ArrayD::<f64>::zeros(IxDyn(&shape));

    Zip::from(&mut x_out)
        .and(&mut y_out)
        .and(lon)
        .and(lat)
        .for_each(|x_slot, y_slot, &lon, &lat| {
            let (x, y) = geographic_to_cartesian_aeqd_kernel(lon, lat, lon_0, lat_0, radius);
            *x_slot = x;
            *y_slot = y;
        });

    Ok((
        PyArrayDyn::from_owned_array(py, x_out),
        PyArrayDyn::from_owned_array(py, y_out),
    ))
}

#[pyfunction(name = "_cartesian_to_geographic_aeqd")]
fn py_cartesian_to_geographic_aeqd<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
    lon_0: f64,
    lat_0: f64,
    radius: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let x = x.as_array();
    let y = y.as_array();
    let shape = broadcast_shape(&[x.shape(), y.shape()])?;

    let x = x.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;
    let y = y.broadcast(IxDyn(&shape)).ok_or_else(broadcast_error)?;

    let mut lon_out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let mut lat_out = ArrayD::<f64>::zeros(IxDyn(&shape));

    Zip::from(&mut lon_out)
        .and(&mut lat_out)
        .and(x)
        .and(y)
        .for_each(|lon_slot, lat_slot, &x, &y| {
            let (lon, lat) = cartesian_to_geographic_aeqd_kernel(x, y, lon_0, lat_0, radius);
            *lon_slot = lon;
            *lat_slot = lat;
        });

    Ok((
        PyArrayDyn::from_owned_array(py, lon_out),
        PyArrayDyn::from_owned_array(py, lat_out),
    ))
}

#[pyfunction(name = "_interpolate_axes_edges_f32")]
fn py_interpolate_axes_edges_f32<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let values = values.as_array();
    validate_edge_values(&values)?;
    Ok(PyArray1::from_owned_array(
        py,
        interpolate_edges_f32(&values, false)?,
    ))
}

#[pyfunction(name = "_interpolate_range_edges_f32")]
fn py_interpolate_range_edges_f32<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let values = values.as_array();
    validate_edge_values(&values)?;
    Ok(PyArray1::from_owned_array(
        py,
        interpolate_edges_f32(&values, true)?,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_antenna_to_cartesian, module)?)?;
    module.add_function(wrap_pyfunction!(py_cartesian_to_antenna, module)?)?;
    module.add_function(wrap_pyfunction!(py_geographic_to_cartesian_aeqd, module)?)?;
    module.add_function(wrap_pyfunction!(py_cartesian_to_geographic_aeqd, module)?)?;
    module.add_function(wrap_pyfunction!(py_interpolate_axes_edges_f32, module)?)?;
    module.add_function(wrap_pyfunction!(py_interpolate_range_edges_f32, module)?)?;
    Ok(())
}

fn validate_edge_values(values: &ndarray::ArrayView1<'_, f32>) -> PyResult<()> {
    if !values.is_standard_layout() {
        return Err(PyValueError::new_err("values must be C-contiguous"));
    }
    if values.len() < 2 {
        return Err(PyValueError::new_err(
            "values must contain at least two centers",
        ));
    }
    Ok(())
}

fn interpolate_edges_f32(
    values: &ndarray::ArrayView1<'_, f32>,
    clamp_negative: bool,
) -> PyResult<Array1<f32>> {
    let n = values.len();
    let out_len = n
        .checked_add(1)
        .ok_or_else(|| PyValueError::new_err("edge output length is too large"))?;
    let mut edges = Array1::<f32>::zeros(out_len);

    for index in 1..n {
        edges[index] = (values[index - 1] + values[index]) / 2.0;
    }
    edges[0] = values[0] - (values[1] - values[0]) / 2.0;
    edges[n] = values[n - 1] - (values[n - 2] - values[n - 1]) / 2.0;
    if clamp_negative {
        for edge in edges.iter_mut() {
            if *edge < 0.0 {
                *edge = 0.0;
            }
        }
    }
    Ok(edges)
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
    use super::*;

    #[test]
    fn antenna_kernel_matches_reference_value() {
        let (x, y, z) = antenna_to_cartesian_kernel(1.0, 2.0, 3.0);

        assert!((x - 34.85145327354068).abs() < 1.0e-9);
        assert!((y - 998.0150432188092).abs() < 1.0e-9);
        assert!((z - 52.39465512149036).abs() < 1.0e-9);
    }

    #[test]
    fn cartesian_kernel_adjusts_negative_azimuth() {
        let (range, azimuth, elevation) = cartesian_to_antenna_kernel(-1.0, 2.0, 3.0);

        assert!((range - 14.0_f64.sqrt()).abs() < 1.0e-12);
        assert!((azimuth - 333.434948822922).abs() < 1.0e-12);
        assert!((elevation - 53.30077479951012).abs() < 1.0e-12);
    }

    #[test]
    fn geographic_aeqd_kernel_matches_reference_value() {
        let (x, y) = geographic_to_cartesian_aeqd_kernel(-96.5, 35.25, -97.0, 36.0, 6370997.0);

        assert!((x - 45404.27864049221).abs() < 1.0e-9);
        assert!((y + 83280.406112787).abs() < 1.0e-9);
    }

    #[test]
    fn cartesian_aeqd_kernel_handles_projection_center() {
        let (lon, lat) = cartesian_to_geographic_aeqd_kernel(0.0, 0.0, -97.0, 36.0, 6370997.0);

        assert!((lon + 97.0).abs() < 1.0e-12);
        assert!((lat - 36.0).abs() < 1.0e-12);
    }

    #[test]
    fn interpolate_edges_f32_matches_axis_and_range_reference() {
        let values = ndarray::array![-1.0_f32, 1.0, 5.0];

        assert_eq!(
            interpolate_edges_f32(&values.view(), false).unwrap(),
            ndarray::array![-2.0_f32, 0.0, 3.0, 7.0]
        );
        assert_eq!(
            interpolate_edges_f32(&values.view(), true).unwrap(),
            ndarray::array![0.0_f32, 0.0, 3.0, 7.0]
        );
    }
}
