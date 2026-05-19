use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "_grid_displacement_peak_2d_f64")]
pub fn grid_displacement_peak_2d_f64(
    imageccorshift: PyReadonlyArray2<'_, f64>,
) -> PyResult<(isize, isize)> {
    let imageccorshift = imageccorshift.as_array();
    grid_displacement_peak(&imageccorshift)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(grid_displacement_peak_2d_f64, module)?)?;
    Ok(())
}

fn grid_displacement_peak(
    imageccorshift: &ndarray::ArrayView2<'_, f64>,
) -> PyResult<(isize, isize)> {
    if !imageccorshift.is_standard_layout() {
        return Err(PyValueError::new_err("imageccorshift must be C-contiguous"));
    }
    let (nrows, ncols) = imageccorshift.dim();
    if nrows == 0 || ncols == 0 {
        return Err(PyValueError::new_err(
            "attempt to get argmax of an empty sequence",
        ));
    }

    let data = imageccorshift
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("imageccorshift must be C-contiguous"))?;
    let mut max_idx = 0_usize;
    let mut max_value = data[0];
    if !max_value.is_finite() {
        return Err(PyValueError::new_err("imageccorshift must be finite"));
    }
    for (idx, &value) in data.iter().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(PyValueError::new_err("imageccorshift must be finite"));
        }
        if value > max_value {
            max_value = value;
            max_idx = idx;
        }
    }

    let row = max_idx / ncols;
    let col = max_idx % ncols;
    let yshift = row as isize - (nrows / 2) as isize;
    let xshift = col as isize - (ncols / 2) as isize;
    Ok((yshift, xshift))
}

#[cfg(test)]
mod tests {
    use super::grid_displacement_peak;
    use ndarray::array;

    #[test]
    fn peak_uses_first_row_major_maximum() {
        let data = array![[1.0, 4.0], [4.0, 2.0]];
        assert_eq!(grid_displacement_peak(&data.view()).unwrap(), (-1, 0));
    }
}
