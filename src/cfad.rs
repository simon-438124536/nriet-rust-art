use ndarray::{Array1, Array2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "_cfad_normalize_dense_f64")]
fn cfad_normalize_dense_f64<'py>(
    py: Python<'py>,
    freq: PyReadonlyArray2<'py, f64>,
    min_frac_thres: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    let freq = freq.as_array();
    let (data, mask) = cfad_normalize(&freq, min_frac_thres)?;
    Ok((
        PyArray2::from_owned_array(py, data),
        PyArray2::from_owned_array(py, mask),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(cfad_normalize_dense_f64, module)?)?;
    Ok(())
}

fn cfad_normalize(
    freq: &ndarray::ArrayView2<'_, f64>,
    min_frac_thres: f64,
) -> PyResult<(Array2<f64>, Array2<bool>)> {
    if !freq.is_standard_layout() {
        return Err(PyValueError::new_err("freq must be C-contiguous"));
    }
    let (nheight, nfield) = freq.dim();
    if nheight == 0 {
        return Err(PyValueError::new_err(
            "zero-size array to reduction operation maximum which has no identity",
        ));
    }
    if !min_frac_thres.is_finite() {
        return Err(PyValueError::new_err("min_frac_thres must be finite"));
    }

    let mut freq_sum = Array1::<f64>::zeros(nheight);
    for row in 0..nheight {
        let mut sum = 0.0;
        for col in 0..nfield {
            let value = freq[[row, col]];
            if !value.is_finite() {
                return Err(PyValueError::new_err("freq values must be finite"));
            }
            sum += value;
        }
        if sum <= 0.0 {
            return Err(PyValueError::new_err(
                "freq row sums must be positive for native CFAD normalization",
            ));
        }
        freq_sum[row] = sum;
    }

    let max_sum = freq_sum.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let point_thres = min_frac_thres * max_sum;
    let mut data = Array2::<f64>::zeros((nheight, nfield));
    let mut mask = Array2::<bool>::from_elem((nheight, nfield), false);
    for row in 0..nheight {
        let row_sum = freq_sum[row];
        let row_mask = row_sum < point_thres;
        for col in 0..nfield {
            data[[row, col]] = freq[[row, col]] / row_sum;
            mask[[row, col]] = row_mask;
        }
    }

    Ok((data, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfad_normalize_matches_row_fraction_and_threshold_mask() {
        let freq = ndarray::array![[1.0, 2.0], [3.0, 1.0], [10.0, 0.0]];

        let (data, mask) = cfad_normalize(&freq.view(), 0.5).unwrap();

        assert_eq!(
            data,
            ndarray::array![[1.0 / 3.0, 2.0 / 3.0], [3.0 / 4.0, 1.0 / 4.0], [1.0, 0.0]]
        );
        assert_eq!(
            mask,
            ndarray::array![[true, true], [true, true], [false, false]]
        );
    }
}
