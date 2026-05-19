import gzip
import hashlib

from tools.minhou_manifest import build_manifest, iter_manifest_files


def test_manifest_sorts_files_and_records_relative_paths(tmp_path):
    data_root = tmp_path / "data"
    nested = data_root / "nested"
    nested.mkdir(parents=True)
    (nested / "b.rstm").write_bytes(b"RSTM-B")
    (data_root / "a.rstm.gz").write_bytes(b"RSTM-A")

    manifest = build_manifest(data_root, header_bytes=4)

    assert manifest["schema_version"] == "minhou-rstm-manifest-v1"
    assert manifest["file_count"] == 2
    assert [entry["relative_path"] for entry in manifest["entries"]] == [
        "a.rstm.gz",
        "nested/b.rstm",
    ]
    assert manifest["entries"][0]["gzip_detected_by_magic"] is False
    assert manifest["entries"][0]["header_preview"]["ascii"] == "RSTM"


def test_manifest_hash_flags_are_explicit(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    payload = b"RSTM compressed"
    compressed_path = data_root / "sample_without_gz_suffix.rstm"
    compressed_path.write_bytes(gzip.compress(payload, mtime=0))

    default_manifest = build_manifest(data_root)
    default_entry = default_manifest["entries"][0]
    assert "compressed_sha256" not in default_entry
    assert "decompressed_sha256" not in default_entry

    hashed_manifest = build_manifest(
        data_root,
        include_compressed_sha256=True,
        include_decompressed_sha256=True,
    )
    hashed_entry = hashed_manifest["entries"][0]

    assert hashed_entry["compression"] == "gzip"
    assert hashed_entry["compressed_sha256"] == hashlib.sha256(
        compressed_path.read_bytes()
    ).hexdigest()
    assert hashed_entry["decompressed_sha256"] == hashlib.sha256(
        payload
    ).hexdigest()


def test_manifest_patterns_and_max_files_are_applied_after_sort(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "b.rstm").write_bytes(b"B")
    (data_root / "a.rstm").write_bytes(b"A")
    (data_root / "ignored.txt").write_bytes(b"ignored")

    files = iter_manifest_files(data_root, patterns=("*.rstm",), max_files=1)

    assert [path.name for path in files] == ["a.rstm"]
