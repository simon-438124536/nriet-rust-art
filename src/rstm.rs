use flate2::read::GzDecoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::fs::File;
use std::io::{Read, Take};
use std::path::Path;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const DEFAULT_HEADER_BYTES: usize = 256;
const RAW_MAGIC_BYTES: usize = 4;

#[pyfunction(name = "_rstm_header_preview")]
#[pyo3(signature = (path, header_bytes = DEFAULT_HEADER_BYTES as isize))]
fn py_rstm_header_preview<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
    header_bytes: isize,
) -> PyResult<Bound<'py, PyDict>> {
    if header_bytes < 0 {
        return Err(PyValueError::new_err("header_bytes must be non-negative"));
    }

    let os = py.import("os")?;
    let fspath = os.call_method1("fspath", (path,))?;
    let path_string: String = fspath.extract()?;
    let record = build_header_preview_record(Path::new(&path_string), header_bytes as usize)?;

    let header_preview = PyDict::new(py);
    header_preview.set_item("length_bytes", record.header_preview.length_bytes)?;
    header_preview.set_item("hex", record.header_preview.hex)?;
    header_preview.set_item("ascii", record.header_preview.ascii)?;

    let out = PyDict::new(py);
    out.set_item("gzip_detected_by_magic", record.gzip_detected_by_magic)?;
    out.set_item("raw_magic_hex", record.raw_magic_hex)?;
    out.set_item("header_preview", header_preview)?;
    Ok(out)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_rstm_header_preview, module)?)?;
    Ok(())
}

struct HeaderPreview {
    length_bytes: usize,
    hex: String,
    ascii: String,
}

struct HeaderPreviewRecord {
    gzip_detected_by_magic: bool,
    raw_magic_hex: String,
    header_preview: HeaderPreview,
}

fn build_header_preview_record(path: &Path, header_bytes: usize) -> PyResult<HeaderPreviewRecord> {
    let raw_magic = read_prefix(path, RAW_MAGIC_BYTES)?;
    let gzip_detected = is_gzip_magic(&raw_magic);
    let preview = read_logical_header(path, header_bytes, gzip_detected)?;

    Ok(HeaderPreviewRecord {
        gzip_detected_by_magic: gzip_detected,
        raw_magic_hex: to_hex(&raw_magic),
        header_preview: HeaderPreview {
            length_bytes: preview.len(),
            hex: to_hex(&preview),
            ascii: ascii_preview(&preview),
        },
    })
}

fn is_gzip_magic(data: &[u8]) -> bool {
    data.len() >= GZIP_MAGIC.len() && data[0] == GZIP_MAGIC[0] && data[1] == GZIP_MAGIC[1]
}

fn read_prefix(path: &Path, length: usize) -> PyResult<Vec<u8>> {
    let mut file = File::open(path)?;
    read_take(&mut file, length)
}

fn read_logical_header(path: &Path, header_bytes: usize, is_gzip: bool) -> PyResult<Vec<u8>> {
    let file = File::open(path)?;
    if is_gzip {
        let mut decoder = GzDecoder::new(file);
        read_take(&mut decoder, header_bytes)
    } else {
        let mut raw = file;
        read_take(&mut raw, header_bytes)
    }
}

fn read_take<R: Read>(reader: &mut R, max_bytes: usize) -> PyResult<Vec<u8>> {
    let mut out = Vec::new();
    let mut limited: Take<&mut R> = reader.take(max_bytes as u64);
    limited.read_to_end(&mut out)?;
    Ok(out)
}

fn to_hex(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len() * 2);
    for byte in data {
        use std::fmt::Write;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn ascii_preview(data: &[u8]) -> String {
    data.iter()
        .map(|byte| {
            if (32..=126).contains(byte) {
                char::from(*byte)
            } else {
                '.'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("nriet-rstm-{name}-{nanos}"))
    }

    fn gzip_bytes(payload: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn header_preview_reads_plain_file_prefix() {
        let path = temp_path("plain");
        fs::write(&path, b"RSTM\x00plain-header").unwrap();

        let record = build_header_preview_record(&path, 8).unwrap();

        assert!(!record.gzip_detected_by_magic);
        assert_eq!(record.raw_magic_hex, "5253544d");
        assert_eq!(record.header_preview.length_bytes, 8);
        assert_eq!(record.header_preview.hex, "5253544d00706c61");
        assert_eq!(record.header_preview.ascii, "RSTM.pla");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn header_preview_decompresses_only_logical_gzip_header() {
        let path = temp_path("gzip");
        fs::write(&path, gzip_bytes(b"RSTM-HEADER-1234567890")).unwrap();

        let record = build_header_preview_record(&path, 11).unwrap();

        assert!(record.gzip_detected_by_magic);
        assert_eq!(record.raw_magic_hex, "1f8b0800");
        assert_eq!(record.header_preview.length_bytes, 11);
        assert_eq!(record.header_preview.hex, "5253544d2d484541444552");
        assert_eq!(record.header_preview.ascii, "RSTM-HEADER");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn zero_byte_preview_keeps_detection_fields() {
        let path = temp_path("zero");
        fs::write(&path, b"AB").unwrap();

        let record = build_header_preview_record(&path, 0).unwrap();

        assert!(!record.gzip_detected_by_magic);
        assert_eq!(record.raw_magic_hex, "4142");
        assert_eq!(record.header_preview.length_bytes, 0);
        assert_eq!(record.header_preview.hex, "");
        assert_eq!(record.header_preview.ascii, "");

        let _ = fs::remove_file(path);
    }
}
