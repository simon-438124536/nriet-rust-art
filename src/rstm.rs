use flate2::read::GzDecoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::fs::{self, File};
use std::io::{Read, Take};
use std::path::Path;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const DEFAULT_HEADER_BYTES: usize = 256;
const RAW_MAGIC_BYTES: usize = 4;
const RSTM_MAGIC: &[u8] = b"RSTM";
const FILE_HEADER_SIZE: usize = 256;
const METADATA_BLOCK_SIZE: usize = 256;
const MAX_RAY_PAYLOAD_PREFIX_BYTES: usize = 64;

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

#[pyfunction(name = "_rstm_build_reference_record")]
#[pyo3(signature = (path, header_bytes = DEFAULT_HEADER_BYTES as isize))]
fn py_rstm_build_reference_record<'py>(
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
    let record = build_reference_record(Path::new(&path_string), header_bytes as usize)?;

    let header_preview = PyDict::new(py);
    header_preview.set_item("length_bytes", record.header_preview.length_bytes)?;
    header_preview.set_item("hex", record.header_preview.hex)?;
    header_preview.set_item("ascii", record.header_preview.ascii)?;

    let out = PyDict::new(py);
    out.set_item("schema_version", "rstm-reference-v1")?;
    out.set_item("path", record.path)?;
    out.set_item("size_bytes", record.size_bytes)?;
    out.set_item(
        "compression",
        if record.gzip_detected_by_magic {
            "gzip"
        } else {
            "none"
        },
    )?;
    out.set_item("gzip_detected_by_magic", record.gzip_detected_by_magic)?;
    out.set_item("raw_magic_hex", record.raw_magic_hex)?;
    out.set_item("header_preview", header_preview)?;
    Ok(out)
}

#[pyfunction(name = "_rstm_parse_file_header")]
fn py_rstm_parse_file_header<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    header_dict(py, &parse_file_header_bytes(data)?)
}

#[pyfunction(name = "_rstm_parse_file")]
fn py_rstm_parse_file<'py>(
    py: Python<'py>,
    path: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let os = py.import("os")?;
    let fspath = os.call_method1("fspath", (path,))?;
    let path_string: String = fspath.extract()?;
    let data = read_logical_payload(Path::new(&path_string))?;
    parsed_file_dict(py, &data)
}

#[pyfunction(name = "_rstm_parse_logical_payload")]
fn py_rstm_parse_logical_payload<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    parsed_file_dict(py, data)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_rstm_header_preview, module)?)?;
    module.add_function(wrap_pyfunction!(py_rstm_build_reference_record, module)?)?;
    module.add_function(wrap_pyfunction!(py_rstm_parse_file_header, module)?)?;
    module.add_function(wrap_pyfunction!(py_rstm_parse_file, module)?)?;
    module.add_function(wrap_pyfunction!(py_rstm_parse_logical_payload, module)?)?;
    Ok(())
}

struct ParsedHeader {
    version_major: u16,
    version_minor: u16,
    header_words: u16,
    reserved_a: u32,
    reserved_b: u32,
    site_id: String,
    site_name: String,
    latitude: f32,
    longitude: f32,
    altitude_m: f32,
    nrays: u32,
    ngates: u32,
    scan_mode: String,
    product_desc: String,
}

struct ParsedRayRecord {
    index: usize,
    offset: usize,
    size: usize,
    ngates: u32,
    payload_prefix: Vec<u8>,
}

struct ParsedFile {
    header: ParsedHeader,
    logical_size_bytes: usize,
    ray_data_offset: usize,
    ray_stride: usize,
    rays: Vec<ParsedRayRecord>,
}

fn parse_file_header_bytes(data: &[u8]) -> PyResult<ParsedHeader> {
    if data.len() < FILE_HEADER_SIZE {
        return Err(PyValueError::new_err(
            "RSTM logical payload shorter than file header",
        ));
    }
    if &data[..4] != RSTM_MAGIC {
        return Err(PyValueError::new_err("expected RSTM magic"));
    }

    let version_major = u16::from_le_bytes(data[4..6].try_into().unwrap());
    let version_minor = u16::from_le_bytes(data[6..8].try_into().unwrap());
    let header_words = u16::from_le_bytes(data[8..10].try_into().unwrap());
    let reserved_a = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let reserved_b = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let site_id = c_string(&data[32..40]);
    let site_name = c_string(&data[40..56]);
    let _pad0 = f32::from_le_bytes(data[0x40..0x44].try_into().unwrap());
    let _pad1 = f32::from_le_bytes(data[0x44..0x48].try_into().unwrap());
    let altitude_m = f32::from_le_bytes(data[0x48..0x4C].try_into().unwrap());
    let latitude = f32::from_le_bytes(data[0x48..0x4C].try_into().unwrap());
    let longitude = f32::from_le_bytes(data[0x4C..0x50].try_into().unwrap());
    let nrays = u32::from_le_bytes(data[0x50..0x54].try_into().unwrap());
    let ngates = u32::from_le_bytes(data[0x54..0x58].try_into().unwrap());
    let scan_mode = c_string(&data[0xA0..0xB0]);
    let product_desc = String::from_utf8_lossy(
        data[0xC0..0xE0]
            .split(|byte| *byte == 0)
            .next()
            .unwrap_or(&[]),
    )
    .into_owned();

    if nrays == 0 || ngates == 0 {
        return Err(PyValueError::new_err("nrays and ngates must be positive"));
    }

    Ok(ParsedHeader {
        version_major,
        version_minor,
        header_words,
        reserved_a,
        reserved_b,
        site_id,
        site_name,
        latitude,
        longitude,
        altitude_m,
        nrays,
        ngates,
        scan_mode,
        product_desc,
    })
}

fn parse_logical_payload(data: &[u8]) -> PyResult<ParsedFile> {
    let header = parse_file_header_bytes(data)?;
    let ray_data_offset = FILE_HEADER_SIZE + METADATA_BLOCK_SIZE;
    if data.len() < ray_data_offset {
        return Err(PyValueError::new_err(
            "logical payload shorter than ray data offset",
        ));
    }
    let payload_bytes = data.len() - ray_data_offset;
    if payload_bytes == 0 {
        return Err(PyValueError::new_err("empty ray payload"));
    }
    let ray_stride = payload_bytes / header.nrays as usize;
    if ray_stride == 0 {
        return Err(PyValueError::new_err(
            "computed ray stride must be positive",
        ));
    }

    let mut rays = Vec::with_capacity(header.nrays as usize);
    for index in 0..header.nrays as usize {
        let offset = ray_data_offset + index * ray_stride;
        let end = if index + 1 == header.nrays as usize {
            data.len()
        } else {
            offset + ray_stride
        };
        let slice = &data[offset..end];
        let prefix_len = slice.len().min(MAX_RAY_PAYLOAD_PREFIX_BYTES);
        rays.push(ParsedRayRecord {
            index,
            offset,
            size: end - offset,
            ngates: header.ngates,
            payload_prefix: slice[..prefix_len].to_vec(),
        });
    }

    Ok(ParsedFile {
        header,
        logical_size_bytes: data.len(),
        ray_data_offset,
        ray_stride,
        rays,
    })
}

fn read_logical_payload(path: &Path) -> PyResult<Vec<u8>> {
    let raw_magic = read_prefix(path, RAW_MAGIC_BYTES)?;
    let gzip_detected = is_gzip_magic(&raw_magic);
    let mut file = File::open(path)?;
    if gzip_detected {
        let mut decoder = GzDecoder::new(file);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out)?;
        Ok(out)
    } else {
        let mut out = Vec::new();
        file.read_to_end(&mut out)?;
        Ok(out)
    }
}

fn c_string(data: &[u8]) -> String {
    data.split(|byte| *byte == 0)
        .next()
        .unwrap_or(data)
        .iter()
        .map(|byte| char::from(*byte))
        .collect::<String>()
        .trim()
        .to_string()
}

fn header_dict<'py>(py: Python<'py>, header: &ParsedHeader) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("schema_version", "rstm-reference-v1")?;
    out.set_item("magic", "RSTM")?;
    out.set_item("version_major", header.version_major)?;
    out.set_item("version_minor", header.version_minor)?;
    out.set_item("header_words", header.header_words)?;
    out.set_item("reserved_a", header.reserved_a)?;
    out.set_item("reserved_b", header.reserved_b)?;
    out.set_item("site_id", &header.site_id)?;
    out.set_item("site_name", &header.site_name)?;
    out.set_item("latitude", header.latitude)?;
    out.set_item("longitude", header.longitude)?;
    out.set_item("altitude_m", header.altitude_m)?;
    out.set_item("nrays", header.nrays)?;
    out.set_item("ngates", header.ngates)?;
    out.set_item("scan_mode", &header.scan_mode)?;
    out.set_item("product_desc", &header.product_desc)?;
    out.set_item("file_header_size", FILE_HEADER_SIZE)?;
    Ok(out)
}

fn parsed_file_dict<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let parsed = parse_logical_payload(data)?;
    let out = PyDict::new(py);
    out.set_item("header", header_dict(py, &parsed.header)?)?;
    out.set_item("logical_size_bytes", parsed.logical_size_bytes)?;
    out.set_item("ray_data_offset", parsed.ray_data_offset)?;
    out.set_item("ray_stride", parsed.ray_stride)?;
    let rays = PyList::empty(py);
    for ray in &parsed.rays {
        let item = PyDict::new(py);
        item.set_item("index", ray.index)?;
        item.set_item("offset", ray.offset)?;
        item.set_item("size", ray.size)?;
        item.set_item("ngates", ray.ngates)?;
        item.set_item("payload_hex_prefix", to_hex(&ray.payload_prefix))?;
        item.set_item("payload_ascii_prefix", ascii_preview(&ray.payload_prefix))?;
        rays.append(item)?;
    }
    out.set_item("rays", rays)?;
    out.set_item("ray_count", parsed.rays.len())?;
    Ok(out)
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

struct ReferenceRecord {
    path: String,
    size_bytes: u64,
    gzip_detected_by_magic: bool,
    raw_magic_hex: String,
    header_preview: HeaderPreview,
}

fn build_reference_record(path: &Path, header_bytes: usize) -> PyResult<ReferenceRecord> {
    let metadata = fs::metadata(path)?;
    let preview_record = build_header_preview_record(path, header_bytes)?;
    Ok(ReferenceRecord {
        path: path_to_lossy_string(path),
        size_bytes: metadata.len(),
        gzip_detected_by_magic: preview_record.gzip_detected_by_magic,
        raw_magic_hex: preview_record.raw_magic_hex,
        header_preview: preview_record.header_preview,
    })
}

fn path_to_lossy_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
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
