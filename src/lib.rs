use pyo3::prelude::*;

/// Returns true when the compiled Rust extension is loaded by Python.
#[pyfunction]
fn rust_backend_ready() -> bool {
    true
}

/// Version reported by the Rust crate.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A tiny numeric kernel used as an initial packaging and smoke-test target.
#[pyfunction]
fn sum_f64(values: Vec<f64>) -> f64 {
    values.iter().copied().sum()
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(rust_backend_ready, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(sum_f64, module)?)?;
    Ok(())
}
