import struct

import numpy as np
import pytest

from pyart.io import nexrad_level3


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _fallback_decode_af1f_rle(rle_data, nbins, monkeypatch):
    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    return nexrad_level3._decode_af1f_rle(rle_data, nbins)


def _af1f_buffer(nbins, radials):
    buf = bytearray(30)
    buf[16:30] = struct.pack(
        ">hhhhhhh",
        nexrad_level3.AF1F,
        0,
        nbins,
        0,
        0,
        1000,
        len(radials),
    )
    for index, rle_data in enumerate(radials):
        assert len(rle_data) % 2 == 0
        buf.extend(struct.pack(">hhh", len(rle_data) // 2, index * 10, 10))
        buf.extend(rle_data)
    return bytes(buf)


@pytest.mark.parametrize(
    ("rle_data", "nbins", "expected"),
    [
        (bytes([0x31]), 3, np.array([1, 1, 1], dtype=np.uint8)),
        (bytes([0x21, 0x12, 0x03, 0x24]), 5, np.array([1, 1, 2, 4, 4], dtype=np.uint8)),
        (bytes([0x00, 0x15]), 1, np.array([5], dtype=np.uint8)),
        (b"", 0, np.array([], dtype=np.uint8)),
    ],
)
def test_af1f_rle_python_fallback_reference_cases(
    monkeypatch, rle_data, nbins, expected
):
    actual = _fallback_decode_af1f_rle(rle_data, nbins, monkeypatch)

    assert type(actual) is np.ndarray
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_af1f_rle_dispatches_exact_bytes_to_private_rust_kernel(monkeypatch):
    calls = []

    def kernel(rle_data, nbins):
        calls.append((rle_data, nbins))
        return np.array([9, 9, 9], dtype=np.uint8)

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_af1f_decode_rle_u8" else None,
    )

    actual = nexrad_level3._decode_af1f_rle(bytes([0x31]), np.int64(3))

    assert calls == [(bytes([0x31]), 3)]
    np.testing.assert_array_equal(actual, np.array([9, 9, 9], dtype=np.uint8))


@pytest.mark.parametrize(
    ("rle_data", "nbins"),
    [
        (bytearray([0x31]), 3),
        (memoryview(bytes([0x31])), 3),
        (bytes([0x31]), True),
        (bytes([0x31]), -1),
        (bytes([0x31]), 3.0),
        (bytes([0x11]), 3),
        (bytes([0x31]), 2),
    ],
)
def test_af1f_rle_unsupported_inputs_keep_python_fallback(
    monkeypatch, rle_data, nbins
):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"unsupported AF1F input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = nexrad_level3._decode_af1f_rle(rle_data, nbins)
    expected = _fallback_decode_af1f_rle(rle_data, nbins, monkeypatch)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_af1f_rle_oversized_output_is_rejected_before_python_allocation(monkeypatch):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"oversized AF1F input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)

    with pytest.raises(ValueError, match="maximum native AF1F"):
        nexrad_level3._decode_af1f_rle(bytes([0x31]), 512 * 1024 * 1024 + 1)


def test_af1f_symbology_block_dispatches_each_exact_radial(monkeypatch):
    calls = []
    payload = _af1f_buffer(3, [bytes([0x21, 0x12]), bytes([0x03, 0x33])])

    def kernel(rle_data, nbins):
        calls.append((rle_data, nbins))
        return np.array([len(calls), len(calls), len(calls)], dtype=np.uint8)

    monkeypatch.setattr(
        nexrad_level3,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_af1f_decode_rle_u8" else None,
    )
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)

    obj._read_symbology_block(payload, 16, nexrad_level3.AF1F)

    assert calls == [(bytes([0x21, 0x12]), 3), (bytes([0x03, 0x33]), 3)]
    np.testing.assert_array_equal(
        obj.raw_data,
        np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8),
    )
    assert obj.radial_headers == [
        {"nbytes": 1, "angle_start": 0, "angle_delta": 10},
        {"nbytes": 1, "angle_start": 10, "angle_delta": 10},
    ]


@pytest.mark.parametrize(
    ("nbins", "rle_data"),
    [
        (3, bytes([0x21, 0x00])),
        (2, bytes([0x31, 0x00])),
    ],
)
def test_af1f_symbology_length_mismatch_keeps_python_exception_surface(
    monkeypatch, nbins, rle_data
):
    def rust_kernel(name):
        def fail(*_args):
            raise AssertionError(f"mismatched AF1F input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    payload = _af1f_buffer(nbins, [rle_data])
    obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)

    with pytest.raises(ValueError) as actual_error:
        obj._read_symbology_block(payload, 16, nexrad_level3.AF1F)

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", lambda _name: None)
    fallback_obj = nexrad_level3.NEXRADLevel3File.__new__(nexrad_level3.NEXRADLevel3File)
    with pytest.raises(type(actual_error.value)) as expected_error:
        fallback_obj._read_symbology_block(payload, 16, nexrad_level3.AF1F)
    assert actual_error.value.args == expected_error.value.args


@pytest.mark.parametrize(
    ("rle_data", "nbins"),
    [
        (bytes([0x31]), 3),
        (bytes([0x21, 0x12, 0x03, 0x24]), 5),
        (bytes([0x00, 0x15]), 1),
        (b"", 0),
    ],
)
def test_real_rust_af1f_rle_matches_python_fallback(monkeypatch, rle_data, nbins):
    rust = _rust_or_skip()

    expected = _fallback_decode_af1f_rle(rle_data, nbins, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_nexrad_af1f_decode_rle_u8":
            calls.append(name)
            return rust._nexrad_af1f_decode_rle_u8
        return None

    monkeypatch.setattr(nexrad_level3, "_rust_kernel", rust_kernel)
    actual = nexrad_level3._decode_af1f_rle(rle_data, nbins)

    assert calls == ["_nexrad_af1f_decode_rle_u8"]
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("rle_data", "nbins", "match"),
    [
        (bytes([0x11]), 2, "does not match"),
        (bytes([0x31]), 2, "exceeds"),
        (bytes([0x31]), 512 * 1024 * 1024 + 1, "maximum"),
    ],
)
def test_real_rust_af1f_rle_direct_rejects_malformed_inputs(
    rle_data, nbins, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._nexrad_af1f_decode_rle_u8(rle_data, nbins)


@pytest.mark.parametrize("nbins", [True, -1])
def test_real_rust_af1f_rle_direct_rejects_bad_nbins(nbins):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match="nbins"):
        rust._nexrad_af1f_decode_rle_u8(b"x", nbins)
