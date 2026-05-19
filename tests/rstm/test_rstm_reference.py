import gzip
import hashlib

from tools.rstm_reference import (
    build_reference_record,
    is_gzip_file,
    read_header_preview,
    sha256_file,
    sha256_logical_payload,
)


def test_gzip_detection_uses_magic_bytes_not_extension(tmp_path):
    plain_named_gz = tmp_path / "plain.rstm.gz"
    plain_named_gz.write_bytes(b"RSTM plain payload")

    gzip_named_raw = tmp_path / "compressed.rstm"
    payload = b"RSTM compressed payload"
    gzip_named_raw.write_bytes(gzip.compress(payload, mtime=0))

    assert is_gzip_file(plain_named_gz) is False
    assert is_gzip_file(gzip_named_raw) is True


def test_header_preview_reads_logical_payload_for_gzip(tmp_path):
    path = tmp_path / "misnamed.bin"
    path.write_bytes(gzip.compress(b"RSTM-HEADER-1234567890", mtime=0))

    preview = read_header_preview(path, header_bytes=11)

    assert preview.length_bytes == 11
    assert preview.ascii == "RSTM-HEADER"
    assert preview.hex == b"RSTM-HEADER".hex()


def test_reference_hashes_are_opt_in_and_have_distinct_meaning(tmp_path):
    payload = b"RSTM logical payload"
    path = tmp_path / "gzip_without_extension"
    path.write_bytes(gzip.compress(payload, mtime=0))

    default_record = build_reference_record(path)
    assert "compressed_sha256" not in default_record
    assert "decompressed_sha256" not in default_record

    hashed_record = build_reference_record(
        path,
        include_compressed_sha256=True,
        include_decompressed_sha256=True,
    )

    assert hashed_record["compression"] == "gzip"
    assert hashed_record["compressed_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert hashed_record["decompressed_sha256"] == hashlib.sha256(
        payload
    ).hexdigest()
    assert hashed_record["decompressed_size_bytes"] == len(payload)


def test_plain_logical_hash_matches_file_hash(tmp_path):
    payload = b"RSTM plain bytes"
    path = tmp_path / "plain_with_gz_suffix.gz"
    path.write_bytes(payload)

    logical_sha, logical_size = sha256_logical_payload(path)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert logical_sha == hashlib.sha256(payload).hexdigest()
    assert logical_size == len(payload)
