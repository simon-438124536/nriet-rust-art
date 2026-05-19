import os
from pathlib import Path

import pytest

from tools.minhou_manifest import ENV_DATA_ROOT, build_manifest


@pytest.mark.skipif(
    not os.environ.get(ENV_DATA_ROOT),
    reason="set RSTM_DATA_ROOT to run operational Minhou manifest checks",
)
def test_operational_minhou_root_smoke():
    data_root = Path(os.environ[ENV_DATA_ROOT])
    if not data_root.exists():
        pytest.skip(f"{ENV_DATA_ROOT} does not exist: {data_root}")

    manifest = build_manifest(data_root, max_files=3, header_bytes=32)

    assert manifest["file_count"] > 0
    for entry in manifest["entries"]:
        assert entry["relative_path"]
        assert "header_preview" in entry
        assert "decompressed_sha256" not in entry


@pytest.mark.skipif(
    not os.environ.get(ENV_DATA_ROOT),
    reason="set RSTM_DATA_ROOT to run operational Minhou acceptance checks",
)
def test_operational_minhou_root_matches_frozen_acceptance_counts():
    data_root = Path(os.environ[ENV_DATA_ROOT])
    if not data_root.exists():
        pytest.skip(f"{ENV_DATA_ROOT} does not exist: {data_root}")

    manifest = build_manifest(data_root, header_bytes=4)
    entries = manifest["entries"]

    assert manifest["file_count"] == 107
    assert sum(entry["size_bytes"] for entry in entries) == 6_021_086_889
    assert sum(
        1 for entry in entries if entry["relative_path"].lower().endswith(".bz2")
    ) == 30
    assert sum(
        1 for entry in entries if entry["relative_path"].lower().endswith(".z")
    ) == 77
    assert all(entry["gzip_detected_by_magic"] for entry in entries)
    assert all(entry["header_preview"]["ascii"].startswith("RSTM") for entry in entries)
