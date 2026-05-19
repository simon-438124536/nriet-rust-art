import os
import struct

import numpy as np
import pytest

import pyart
from pyart.io import _sigmetfile

os.environ.setdefault("PYART_QUIET", "1")


def _fallback_mask_gates_not_collected(mask, nbins, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    return _sigmetfile._mask_gates_not_collected(mask, nbins)


def test_sigmet_angle_helpers_accept_scalars_and_arrays():
    assert _sigmetfile.bin2_to_angle(32768) == 180.0
    assert _sigmetfile.bin4_to_angle(2147483648) == 180.0

    np.testing.assert_allclose(
        _sigmetfile.bin2_to_angle(np.array([0, 16384, 32768], dtype=np.uint16)),
        np.array([0.0, 90.0, 180.0]),
    )
    np.testing.assert_allclose(
        _sigmetfile.bin4_to_angle(np.array([0, 1073741824], dtype=np.uint32)),
        np.array([0.0, 90.0]),
    )


def test_unpack_structure_returns_named_fields():
    payload = struct.pack("hhi hh".replace(" ", ""), 24, 1, 12, 0, 3)
    result = _sigmetfile._unpack_structure(payload, _sigmetfile.STRUCTURE_HEADER)

    assert result == {
        "structure_identifier": 24,
        "format_version": 1,
        "bytes_in_structure": 12,
        "reserved": 0,
        "flag": 3,
    }


@pytest.mark.parametrize(
    "path, scan_mode, fixed_angle",
    [
        (pyart.testing.SIGMET_PPI_FILE, 4, 0.4998779296875),
        (pyart.testing.SIGMET_RHI_FILE, 2, 37.298583984375),
    ],
)
def test_sigmet_file_constructor_parses_fixture_headers(path, scan_mode, fixed_angle):
    sigmet = _sigmetfile.SigmetFile(path)
    try:
        assert sigmet.data_types == [9]
        assert sigmet.data_type_names == ["DBZ2"]
        assert sigmet.ndata_types == 1
        assert sigmet.product_hdr["product_end"]["number_bins"] == 25
        assert sigmet.ingest_header["ingest_configuration"]["number_rays_sweep"] == 20
        scan_info = sigmet.ingest_header["task_configuration"]["task_scan_info"]
        assert scan_info["number_sweeps"] == 1
        assert scan_info["antenna_scan_mode"] == scan_mode

        data, metadata = sigmet.read_data()
        assert data["DBZ2"].shape == (1, 20, 25)
        assert metadata["DBZ2"]["nbins"].shape == (1, 20)
        angle = _sigmetfile.bin2_to_angle(
            sigmet.ingest_data_headers["DBZ2"][0]["fixed_angle"]
        )
        assert angle == fixed_angle
    finally:
        sigmet.close()


def test_sigmet_file_constructor_accepts_file_like_object():
    with open(pyart.testing.SIGMET_PPI_FILE, "rb") as fh:
        sigmet = _sigmetfile.SigmetFile(fh)
        assert sigmet.data_type_names == ["DBZ2"]
        assert sigmet.product_hdr["product_end"]["number_bins"] == 25
        sigmet.close()


def test_sigmet_read_data_preserves_nonuniform_bin_masking():
    sigmet = _sigmetfile.SigmetFile(pyart.testing.SIGMET_PPI_FILE)
    try:
        data, _ = sigmet.read_data()
    finally:
        sigmet.close()

    reflectivity = data["DBZ2"]
    assert reflectivity.dtype == np.float32
    assert reflectivity.shape == (1, 20, 25)
    assert reflectivity[0, 0, 0] == 0.0
    assert reflectivity[0, 19, 14] == 0.0
    assert reflectivity[0, 19, 15] is np.ma.masked
    assert int(np.ma.getmaskarray(reflectivity).sum()) == 10


def test_convert_sigmet_data_one_byte_dbt_matches_oracle_issue_299():
    data = np.ones((2, 2), dtype=np.int16) * 257
    nbins = np.ones((2,), dtype=np.int16) * 2

    result = _sigmetfile.convert_sigmet_data(1, data, nbins)

    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    assert result.fill_value == -9999.0
    np.testing.assert_array_equal(result, np.full((2, 2), -31.5, dtype=np.float32))


def test_mask_gates_not_collected_python_fallback_marks_tail_bins(monkeypatch):
    mask = np.zeros((4, 5), dtype=np.uint8)
    nbins = np.array([0, 2, 5, 6], dtype=np.int16)

    result = _fallback_mask_gates_not_collected(mask, nbins, monkeypatch)

    assert result is None
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_mask_gates_not_collected_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(mask, nbins):
        calls.append((mask.dtype, mask.shape, nbins.dtype, nbins.copy()))
        mask[:, :] = 7

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_mask_gates_not_collected" else None,
    )
    mask = np.zeros((2, 4), dtype=np.uint8)
    nbins = np.array([1, 3], dtype=np.int16)

    result = _sigmetfile._mask_gates_not_collected(mask, nbins)

    assert result is None
    assert calls[0][0:3] == (np.uint8, (2, 4), np.int64)
    np.testing.assert_array_equal(calls[0][3], np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(mask, np.full((2, 4), 7, dtype=np.uint8))


def test_mask_gates_not_collected_keeps_negative_nbins_on_python_path(monkeypatch):
    nbins = np.array([-1, 2], dtype=np.int16)
    expected = np.zeros((2, 4), dtype=np.uint8)
    _fallback_mask_gates_not_collected(expected, nbins, monkeypatch)

    def fail_if_called(name):
        if name != "_mask_gates_not_collected":
            return None

        def kernel(*_args):
            raise AssertionError("negative nbins input used Rust")

        return kernel

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", fail_if_called)
    actual = np.zeros((2, 4), dtype=np.uint8)
    result = _sigmetfile._mask_gates_not_collected(actual, nbins)

    assert result is None
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_mask_gates_not_collected_matches_python_fallback(monkeypatch):
    nbins = np.array([0, 2, 5, 6], dtype=np.int16)
    expected = np.zeros((4, 5), dtype=np.uint8)
    _fallback_mask_gates_not_collected(expected, nbins, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = np.zeros((4, 5), dtype=np.uint8)
    result = _sigmetfile._mask_gates_not_collected(actual, nbins)

    assert result is None
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("mask", "nbins", "match"),
    [
        (
            np.zeros((2, 4), dtype=np.uint8),
            np.array([1], dtype=np.int64),
            "nbins length",
        ),
        (
            np.zeros((2, 4), dtype=np.uint8),
            np.array([1, -1], dtype=np.int64),
            "non-negative",
        ),
        (
            np.zeros((2, 4), dtype=np.uint8)[:, ::-1],
            np.array([1, 2], dtype=np.int64),
            "C-contiguous",
        ),
    ],
)
def test_real_rust_mask_gates_not_collected_rejects_unsafe_direct_inputs(
    mask, nbins, match
):
    import pyart._rust as rust

    before = mask.copy()
    with pytest.raises(ValueError, match=match):
        rust._mask_gates_not_collected(mask, nbins)

    np.testing.assert_array_equal(mask, before)


def test_public_read_sigmet_uses_compat_reader_for_tiny_ppi_fixture():
    with pytest.warns(UserWarning, match="SIGMET module is deprecated"):
        radar = pyart.io.read_sigmet(pyart.testing.SIGMET_PPI_FILE)

    assert radar.metadata["original_container"] == "sigmet"
    assert radar.scan_type == "ppi"
    assert radar.nrays == 20
    assert radar.ngates == 25
    assert radar.fields["reflectivity"]["data"].shape == (20, 25)
    assert radar.fields["reflectivity"]["data"][19, 15] is np.ma.masked


@pytest.mark.parametrize(
    "time_ordered", ["roll", "reverse", "reverse_and_roll", "full", "sequential"]
)
def test_public_read_sigmet_time_ordering_paths_are_python3_compatible(time_ordered):
    with pytest.warns(UserWarning):
        radar = pyart.io.read_sigmet(
            pyart.testing.SIGMET_PPI_FILE, time_ordered=time_ordered
        )

    assert radar.nrays == 20
    assert radar.ngates == 25
