use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn _gatefilter_merge<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
    marked: PyReadonlyArray2<'py, bool>,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let gate_excluded = gate_excluded.as_array();
    let marked = marked.as_array();
    let output = gatefilter_merge_kernel(gate_excluded, marked, op)?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction]
pub fn _gatefilter_compare_merge<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
    data: PyReadonlyArray2<'py, f64>,
    data_mask: PyReadonlyArray2<'py, bool>,
    value: f64,
    comparator: &str,
    invert_marked: bool,
    exclude_masked: bool,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let gate_excluded = gate_excluded.as_array();
    let data = data.as_array();
    let data_mask = data_mask.as_array();
    let output = gatefilter_compare_merge_kernel(
        gate_excluded,
        data,
        data_mask,
        value,
        comparator,
        invert_marked,
        exclude_masked,
        op,
    )?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction]
pub fn _gatefilter_interval_merge<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
    data: PyReadonlyArray2<'py, f64>,
    data_mask: PyReadonlyArray2<'py, bool>,
    v1: f64,
    v2: f64,
    mode: &str,
    inclusive: bool,
    invert_marked: bool,
    exclude_masked: bool,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let gate_excluded = gate_excluded.as_array();
    let data = data.as_array();
    let data_mask = data_mask.as_array();
    let output = gatefilter_interval_merge_kernel(
        gate_excluded,
        data,
        data_mask,
        v1,
        v2,
        mode,
        inclusive,
        invert_marked,
        exclude_masked,
        op,
    )?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction]
pub fn _gatefilter_finite_merge<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
    data: PyReadonlyArray2<'py, f64>,
    data_mask: PyReadonlyArray2<'py, bool>,
    exclude_masked: bool,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let gate_excluded = gate_excluded.as_array();
    let data = data.as_array();
    let data_mask = data_mask.as_array();
    let output =
        gatefilter_finite_merge_kernel(gate_excluded, data, data_mask, exclude_masked, op)?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction]
pub fn _gatefilter_last_gates_merge<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
    n_gates: isize,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let gate_excluded = gate_excluded.as_array();
    let output = gatefilter_last_gates_merge_kernel(gate_excluded, n_gates, op)?;
    Ok(PyArray2::from_owned_array(py, output))
}

#[pyfunction]
pub fn _gatefilter_transition_merge<'py>(
    py: Python<'py>,
    gate_excluded: PyReadonlyArray2<'py, bool>,
    transitions: PyReadonlyArray1<'py, f64>,
    trans_value: f64,
    invert_marked: bool,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let gate_excluded = gate_excluded.as_array();
    let transitions = transitions.as_array();
    let output = gatefilter_transition_merge_kernel(
        gate_excluded,
        transitions,
        trans_value,
        invert_marked,
        op,
    )?;
    Ok(PyArray2::from_owned_array(py, output))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_gatefilter_merge, module)?)?;
    module.add_function(wrap_pyfunction!(_gatefilter_compare_merge, module)?)?;
    module.add_function(wrap_pyfunction!(_gatefilter_interval_merge, module)?)?;
    module.add_function(wrap_pyfunction!(_gatefilter_finite_merge, module)?)?;
    module.add_function(wrap_pyfunction!(_gatefilter_last_gates_merge, module)?)?;
    module.add_function(wrap_pyfunction!(_gatefilter_transition_merge, module)?)?;
    Ok(())
}

fn gatefilter_compare_merge_kernel(
    gate_excluded: ArrayView2<'_, bool>,
    data: ArrayView2<'_, f64>,
    data_mask: ArrayView2<'_, bool>,
    value: f64,
    comparator: &str,
    invert_marked: bool,
    exclude_masked: bool,
    op: &str,
) -> PyResult<Array2<bool>> {
    if gate_excluded.dim() != data.dim() || gate_excluded.dim() != data_mask.dim() {
        return Err(PyValueError::new_err(
            "gate_excluded, data, and data_mask must have the same shape",
        ));
    }
    if !gate_excluded.is_standard_layout()
        || !data.is_standard_layout()
        || !data_mask.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "gate_excluded, data, and data_mask must be C-contiguous",
        ));
    }

    let (rows, cols) = gate_excluded.dim();
    let mut marked = Array2::<bool>::default((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let mark = if data_mask[[row, col]] {
                exclude_masked
            } else {
                let comparison = compare_value(data[[row, col]], value, comparator)?;
                if invert_marked {
                    !comparison
                } else {
                    comparison
                }
            };
            marked[[row, col]] = mark;
        }
    }

    gatefilter_merge_kernel(gate_excluded, marked.view(), op)
}

fn compare_value(data: f64, value: f64, comparator: &str) -> PyResult<bool> {
    match comparator {
        "lt" => Ok(data < value),
        "le" => Ok(data <= value),
        "gt" => Ok(data > value),
        "ge" => Ok(data >= value),
        "eq" => Ok(data == value),
        "ne" => Ok(data != value),
        _ => Err(PyValueError::new_err("invalid comparator")),
    }
}

fn gatefilter_interval_merge_kernel(
    gate_excluded: ArrayView2<'_, bool>,
    data: ArrayView2<'_, f64>,
    data_mask: ArrayView2<'_, bool>,
    v1: f64,
    v2: f64,
    mode: &str,
    inclusive: bool,
    invert_marked: bool,
    exclude_masked: bool,
    op: &str,
) -> PyResult<Array2<bool>> {
    if gate_excluded.dim() != data.dim() || gate_excluded.dim() != data_mask.dim() {
        return Err(PyValueError::new_err(
            "gate_excluded, data, and data_mask must have the same shape",
        ));
    }
    if !gate_excluded.is_standard_layout()
        || !data.is_standard_layout()
        || !data_mask.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "gate_excluded, data, and data_mask must be C-contiguous",
        ));
    }

    let (rows, cols) = gate_excluded.dim();
    let mut marked = Array2::<bool>::default((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let mark = if data_mask[[row, col]] {
                exclude_masked
            } else {
                let comparison = compare_interval(data[[row, col]], v1, v2, mode, inclusive)?;
                if invert_marked {
                    !comparison
                } else {
                    comparison
                }
            };
            marked[[row, col]] = mark;
        }
    }

    gatefilter_merge_kernel(gate_excluded, marked.view(), op)
}

fn compare_interval(data: f64, v1: f64, v2: f64, mode: &str, inclusive: bool) -> PyResult<bool> {
    match (mode, inclusive) {
        ("inside", true) => Ok(data >= v1 && data <= v2),
        ("inside", false) => Ok(data > v1 && data < v2),
        ("outside", true) => Ok(data <= v1 || data >= v2),
        ("outside", false) => Ok(data < v1 || data > v2),
        _ => Err(PyValueError::new_err("invalid interval mode")),
    }
}

fn gatefilter_finite_merge_kernel(
    gate_excluded: ArrayView2<'_, bool>,
    data: ArrayView2<'_, f64>,
    data_mask: ArrayView2<'_, bool>,
    exclude_masked: bool,
    op: &str,
) -> PyResult<Array2<bool>> {
    if gate_excluded.dim() != data.dim() || gate_excluded.dim() != data_mask.dim() {
        return Err(PyValueError::new_err(
            "gate_excluded, data, and data_mask must have the same shape",
        ));
    }
    if !gate_excluded.is_standard_layout()
        || !data.is_standard_layout()
        || !data_mask.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "gate_excluded, data, and data_mask must be C-contiguous",
        ));
    }

    let (rows, cols) = gate_excluded.dim();
    let mut marked = Array2::<bool>::default((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            marked[[row, col]] = if data_mask[[row, col]] {
                exclude_masked
            } else {
                !data[[row, col]].is_finite()
            };
        }
    }

    gatefilter_merge_kernel(gate_excluded, marked.view(), op)
}

fn gatefilter_last_gates_merge_kernel(
    gate_excluded: ArrayView2<'_, bool>,
    n_gates: isize,
    op: &str,
) -> PyResult<Array2<bool>> {
    if !gate_excluded.is_standard_layout() {
        return Err(PyValueError::new_err("gate_excluded must be C-contiguous"));
    }

    let (rows, cols) = gate_excluded.dim();
    let start = normalize_python_slice_start(cols, -(n_gates as i128));
    let mut marked = Array2::<bool>::default((rows, cols));
    for row in 0..rows {
        for col in start..cols {
            marked[[row, col]] = true;
        }
    }

    gatefilter_merge_kernel(gate_excluded, marked.view(), op)
}

fn gatefilter_transition_merge_kernel(
    gate_excluded: ArrayView2<'_, bool>,
    transitions: ndarray::ArrayView1<'_, f64>,
    trans_value: f64,
    invert_marked: bool,
    op: &str,
) -> PyResult<Array2<bool>> {
    if !gate_excluded.is_standard_layout() || !transitions.is_standard_layout() {
        return Err(PyValueError::new_err(
            "gate_excluded and transitions must be C-contiguous",
        ));
    }
    let (rows, cols) = gate_excluded.dim();
    if transitions.len() != rows {
        return Err(PyValueError::new_err(
            "transitions length must match gate_excluded rows",
        ));
    }

    let mut marked = Array2::<bool>::default((rows, cols));
    for row in 0..rows {
        let mut mark = transitions[row] == trans_value;
        if invert_marked {
            mark = !mark;
        }
        if mark {
            for col in 0..cols {
                marked[[row, col]] = true;
            }
        }
    }

    gatefilter_merge_kernel(gate_excluded, marked.view(), op)
}

fn normalize_python_slice_start(len: usize, mut start: i128) -> usize {
    let len = len as i128;
    if start < 0 {
        start += len;
    }
    start.clamp(0, len) as usize
}

fn gatefilter_merge_kernel(
    gate_excluded: ArrayView2<'_, bool>,
    marked: ArrayView2<'_, bool>,
    op: &str,
) -> PyResult<Array2<bool>> {
    if gate_excluded.dim() != marked.dim() {
        return Err(PyValueError::new_err(
            "gate_excluded and marked must have the same shape",
        ));
    }

    let (rows, cols) = gate_excluded.dim();
    let mut output = Array2::<bool>::default((rows, cols));

    match op {
        "or" => {
            for row in 0..rows {
                for col in 0..cols {
                    output[[row, col]] = gate_excluded[[row, col]] || marked[[row, col]];
                }
            }
        }
        "and" => {
            for row in 0..rows {
                for col in 0..cols {
                    output[[row, col]] = gate_excluded[[row, col]] && marked[[row, col]];
                }
            }
        }
        "new" => {
            output.assign(&marked);
        }
        _ => {
            return Err(PyValueError::new_err((
                "invalid 'op' parameter: ",
                op.to_string(),
            )))
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn merge_or_matches_numpy_logic() {
        let gate = array![[false, true, false], [true, false, true]];
        let marked = array![[false, false, true], [true, true, false]];

        let output = gatefilter_merge_kernel(gate.view(), marked.view(), "or").unwrap();

        assert_eq!(output, array![[false, true, true], [true, true, true]]);
    }

    #[test]
    fn merge_and_matches_numpy_logic() {
        let gate = array![[false, true, false], [true, false, true]];
        let marked = array![[false, false, true], [true, true, false]];

        let output = gatefilter_merge_kernel(gate.view(), marked.view(), "and").unwrap();

        assert_eq!(output, array![[false, false, false], [true, false, false]]);
    }

    #[test]
    fn merge_new_replaces_with_marked() {
        let gate = array![[false, true], [true, false]];
        let marked = array![[true, false], [false, true]];

        let output = gatefilter_merge_kernel(gate.view(), marked.view(), "new").unwrap();

        assert_eq!(output, marked);
    }

    #[test]
    fn compare_merge_masks_then_inverts_for_include_methods() {
        let gate = array![[false, true, false], [true, false, true]];
        let data = ndarray::arr2(&[[0.0, 2.0, f64::NAN], [4.0, 5.0, 6.0]]);
        let data_mask = array![[false, true, false], [false, false, false]];

        let output = gatefilter_compare_merge_kernel(
            gate.view(),
            data.view(),
            data_mask.view(),
            3.0,
            "lt",
            true,
            true,
            "or",
        )
        .unwrap();

        assert_eq!(output, array![[false, true, true], [true, true, true]]);
    }

    #[test]
    fn compare_merge_include_mask_fill_is_not_inverted() {
        let gate = array![[true, true, true]];
        let data = ndarray::arr2(&[[0.0, 2.0, 4.0]]);
        let data_mask = array![[false, true, false]];

        let output = gatefilter_compare_merge_kernel(
            gate.view(),
            data.view(),
            data_mask.view(),
            3.0,
            "lt",
            true,
            true,
            "and",
        )
        .unwrap();

        assert_eq!(output, array![[false, true, true]]);
    }

    #[test]
    fn interval_merge_handles_inside_inclusive_and_mask_fill() {
        let gate = array![[false, true, false], [true, false, true]];
        let data = ndarray::arr2(&[[0.0, 2.0, f64::NAN], [4.0, 5.0, 6.0]]);
        let data_mask = array![[false, true, false], [false, false, false]];

        let output = gatefilter_interval_merge_kernel(
            gate.view(),
            data.view(),
            data_mask.view(),
            2.0,
            5.0,
            "inside",
            true,
            false,
            true,
            "or",
        )
        .unwrap();

        assert_eq!(output, array![[false, true, false], [true, true, true]]);
    }

    #[test]
    fn interval_merge_include_mask_fill_is_not_inverted() {
        let gate = array![[true, true, true]];
        let data = ndarray::arr2(&[[0.0, 2.0, 4.0]]);
        let data_mask = array![[false, true, false]];

        let output = gatefilter_interval_merge_kernel(
            gate.view(),
            data.view(),
            data_mask.view(),
            1.0,
            3.0,
            "inside",
            true,
            true,
            false,
            "and",
        )
        .unwrap();

        assert_eq!(output, array![[true, false, true]]);
    }

    #[test]
    fn finite_merge_marks_nan_and_infinity() {
        let gate = array![[false, true, false, true]];
        let data = ndarray::arr2(&[[1.0, f64::NAN, f64::INFINITY, 4.0]]);
        let data_mask = array![[false, false, false, true]];

        let output =
            gatefilter_finite_merge_kernel(gate.view(), data.view(), data_mask.view(), true, "or")
                .unwrap();

        assert_eq!(output, array![[false, true, true, true]]);
    }

    #[test]
    fn last_gates_merge_preserves_python_slice_edges() {
        let gate = array![[false, true, false], [true, false, true]];

        let all = gatefilter_last_gates_merge_kernel(gate.view(), 0, "new").unwrap();
        let from_second = gatefilter_last_gates_merge_kernel(gate.view(), -1, "new").unwrap();
        let none = gatefilter_last_gates_merge_kernel(gate.view(), -3, "new").unwrap();
        let over = gatefilter_last_gates_merge_kernel(gate.view(), 10, "new").unwrap();

        assert_eq!(all, array![[true, true, true], [true, true, true]]);
        assert_eq!(
            from_second,
            array![[false, true, true], [false, true, true]]
        );
        assert_eq!(none, array![[false, false, false], [false, false, false]]);
        assert_eq!(over, array![[true, true, true], [true, true, true]]);
    }

    #[test]
    fn transition_merge_marks_rows_and_inverts_for_include_methods() {
        let gate = array![[false, true], [true, false], [false, false]];
        let transitions = ndarray::arr1(&[0.0, 1.0, f64::NAN]);

        let exclude =
            gatefilter_transition_merge_kernel(gate.view(), transitions.view(), 1.0, false, "new")
                .unwrap();
        let include =
            gatefilter_transition_merge_kernel(gate.view(), transitions.view(), 0.0, true, "new")
                .unwrap();

        assert_eq!(
            exclude,
            array![[false, false], [true, true], [false, false]]
        );
        assert_eq!(include, array![[false, false], [true, true], [true, true]]);
    }
}
