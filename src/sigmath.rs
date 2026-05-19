use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "_angular_texture_2d")]
pub fn py_angular_texture_2d<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    window_rows: usize,
    window_cols: usize,
    interval: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let image = image.as_array();
    let output = angular_texture_2d_kernel(image, window_rows, window_cols, interval)?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction(name = "_texture_along_ray_dense_f64")]
pub fn py_texture_along_ray_dense_f64<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    wind_size: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data = data.as_array();
    let output = texture_along_ray_kernel(data, wind_size)?;
    Ok(PyArray2::from_owned_array(py, output))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_angular_texture_2d, module)?)?;
    module.add_function(wrap_pyfunction!(py_texture_along_ray_dense_f64, module)?)?;
    Ok(())
}

fn angular_texture_2d_kernel(
    image: ArrayView2<'_, f64>,
    window_rows: usize,
    window_cols: usize,
    interval: f64,
) -> PyResult<Array2<f64>> {
    validate_window(window_rows, window_cols)?;

    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("image must be non-empty"));
    }

    let ns = window_rows
        .checked_mul(window_cols)
        .ok_or_else(|| PyValueError::new_err("window is too large"))? as f64;
    let row_radius = (window_rows / 2) as isize;
    let col_radius = (window_cols / 2) as isize;
    let mut output = Array2::<f64>::zeros((rows, cols));

    for row in 0..rows {
        for col in 0..cols {
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;

            for window_row in 0..window_rows {
                let source_row =
                    symmetric_index(row as isize + window_row as isize - row_radius, rows);
                for window_col in 0..window_cols {
                    let source_col =
                        symmetric_index(col as isize + window_col as isize - col_radius, cols);
                    let radians = image[[source_row, source_col]] / interval * std::f64::consts::PI;
                    x_sum += radians.cos();
                    y_sum += radians.sin();
                }
            }

            let x_mean = x_sum / ns;
            let y_mean = y_sum / ns;
            let norm = (x_mean.powi(2) + y_mean.powi(2)).sqrt();
            output[[row, col]] = (-2.0 * norm.ln()).sqrt() * interval / std::f64::consts::PI;
        }
    }

    Ok(output)
}

fn texture_along_ray_kernel(data: ArrayView2<'_, f64>, wind_size: usize) -> PyResult<Array2<f64>> {
    validate_texture_along_ray_inputs(data, wind_size)?;

    let (rows, cols) = data.dim();
    let half_wind = (wind_size - 1) / 2;
    let mut output = Array2::<f64>::zeros((rows, cols));

    for row in 0..rows {
        let mut ray_values = Vec::with_capacity(cols - wind_size + 1);
        for start_col in 0..=cols - wind_size {
            let mut total = 0.0;
            for offset in 0..wind_size {
                total += data[[row, start_col + offset]];
            }
            let mean = total / wind_size as f64;
            let mut square_total = 0.0;
            for offset in 0..wind_size {
                let diff = data[[row, start_col + offset]] - mean;
                square_total += diff * diff;
            }
            ray_values.push((square_total / wind_size as f64).sqrt());
        }

        for (idx, value) in ray_values.iter().copied().enumerate() {
            output[[row, half_wind + idx]] = value;
        }
        let first = ray_values[0];
        let last = ray_values[ray_values.len() - 1];
        for col in 0..half_wind {
            output[[row, col]] = first;
        }
        for col in cols - half_wind..cols {
            output[[row, col]] = last;
        }
    }

    Ok(output)
}

fn validate_texture_along_ray_inputs(data: ArrayView2<'_, f64>, wind_size: usize) -> PyResult<()> {
    if !data.is_standard_layout() {
        return Err(PyValueError::new_err("data must be C-contiguous"));
    }
    if wind_size < 3 {
        return Err(PyValueError::new_err("wind_size must be at least 3"));
    }
    if wind_size % 2 == 0 {
        return Err(PyValueError::new_err("wind_size must be odd"));
    }
    if data.dim().1 < wind_size {
        return Err(PyValueError::new_err(
            "wind_size must not exceed gate count",
        ));
    }
    for &value in data.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("data must be finite"));
        }
    }
    Ok(())
}

fn validate_window(window_rows: usize, window_cols: usize) -> PyResult<()> {
    if window_rows == 0 || window_cols == 0 {
        return Err(PyValueError::new_err("window dimensions must be positive"));
    }
    if window_rows % 2 == 0 || window_cols % 2 == 0 {
        return Err(PyValueError::new_err("window dimensions must be odd"));
    }
    Ok(())
}

fn symmetric_index(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }

    let period = (len * 2) as isize;
    let mut wrapped = index % period;
    if wrapped < 0 {
        wrapped += period;
    }

    if wrapped >= len as isize {
        (period - wrapped - 1) as usize
    } else {
        wrapped as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn symmetric_index_includes_edge_value() {
        let reflected: Vec<usize> = (-3..7).map(|index| symmetric_index(index, 3)).collect();

        assert_eq!(reflected, vec![2, 1, 0, 0, 1, 2, 2, 1, 0, 0]);
    }

    #[test]
    fn angular_texture_matches_reference_center_value() {
        let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let output = angular_texture_2d_kernel(image.view(), 3, 3, 8.0).unwrap();

        assert!((output[[0, 0]] - 1.5175009661625614).abs() < 1.0e-14);
        assert!((output[[1, 2]] - 1.5175009661625611).abs() < 1.0e-14);
    }
}
