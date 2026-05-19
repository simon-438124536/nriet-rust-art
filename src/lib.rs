use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod advection;
mod cappi;
mod cfad;
mod correct;
mod despeckle;
mod echo_class;
mod filters;
mod io;
mod kdp;
mod map;
mod qpe;
mod qvp;
mod rstm;
mod sigmath;
mod simple_moment;
mod spectra;
mod srv;
mod transforms;
mod util;
mod vad;

const DEFAULT_GZIP_MAX_UNCOMPRESSED_BYTES: usize = 512 * 1024 * 1024;
const GZIP_CHUNK_BYTES: usize = 64 * 1024;

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

/// Return true when the byte buffer begins with the gzip magic bytes.
#[pyfunction]
fn is_gzip_magic(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b
}

/// Return the uncompressed byte length from a gzip member.
#[pyfunction]
#[pyo3(signature = (data, max_uncompressed_bytes = DEFAULT_GZIP_MAX_UNCOMPRESSED_BYTES))]
fn gzip_decompressed_len(data: &[u8], max_uncompressed_bytes: usize) -> PyResult<usize> {
    let (len, _) = read_gzip_bounded(data, max_uncompressed_bytes, false)?;
    Ok(len)
}

/// Return a decompressed gzip payload.
#[pyfunction]
#[pyo3(signature = (data, max_uncompressed_bytes = DEFAULT_GZIP_MAX_UNCOMPRESSED_BYTES))]
fn gzip_decompress<'py>(
    py: Python<'py>,
    data: &[u8],
    max_uncompressed_bytes: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let (_, out) = read_gzip_bounded(data, max_uncompressed_bytes, true)?;
    Ok(PyBytes::new(py, &out))
}

fn read_gzip_bounded(
    data: &[u8],
    max_uncompressed_bytes: usize,
    keep_output: bool,
) -> PyResult<(usize, Vec<u8>)> {
    use flate2::read::GzDecoder;
    use std::io::{Cursor, Read};

    let mut decoder = GzDecoder::new(Cursor::new(data));
    let mut chunk = [0_u8; GZIP_CHUNK_BYTES];
    let mut out = if keep_output { Vec::new() } else { Vec::new() };
    let mut total = 0_usize;

    loop {
        let read = decoder.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read)
            .ok_or_else(|| PyValueError::new_err("gzip payload exceeds size limit"))?;
        if total > max_uncompressed_bytes {
            return Err(PyValueError::new_err(format!(
                "gzip payload exceeds max_uncompressed_bytes ({max_uncompressed_bytes})"
            )));
        }
        if keep_output {
            out.extend_from_slice(&chunk[..read]);
        }
    }

    Ok((total, out))
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(rust_backend_ready, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(sum_f64, module)?)?;
    module.add_function(wrap_pyfunction!(is_gzip_magic, module)?)?;
    module.add_function(wrap_pyfunction!(gzip_decompressed_len, module)?)?;
    module.add_function(wrap_pyfunction!(gzip_decompress, module)?)?;
    advection::register(module)?;
    cappi::register(module)?;
    kdp::register(module)?;
    transforms::register(module)?;
    correct::register(module)?;
    cfad::register(module)?;
    despeckle::register(module)?;
    echo_class::register(module)?;
    filters::register(module)?;
    io::register(module)?;
    map::register(module)?;
    qpe::register(module)?;
    qvp::register(module)?;
    rstm::register(module)?;
    simple_moment::register(module)?;
    sigmath::register(module)?;
    spectra::register(module)?;
    srv::register(module)?;
    util::register(module)?;
    vad::register(module)?;
    Ok(())
}
