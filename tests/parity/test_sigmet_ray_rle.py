import numpy as np
import pytest

from pyart.io import _sigmetfile


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_sigmet_decode_ray_current_record_i16"):
        pytest.skip("pyart._rust has no SIGMET ray RLE kernel in this test mode")
    return rust


def _sigmet_obj(rbuf, rbuf_pos):
    obj = _sigmetfile.SigmetFile.__new__(_sigmetfile.SigmetFile)
    obj._rbuf = rbuf
    obj._rbuf_pos = rbuf_pos
    obj.debug = False
    return obj


def _python_get_ray(rbuf, rbuf_pos, nbins, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    obj = _sigmet_obj(rbuf.copy(), rbuf_pos)
    out = np.ones(nbins + 6, dtype=np.int16)
    status = obj._get_ray(nbins, out)
    return status, obj._rbuf_pos, out


def _run_direct_rust(rbuf, rbuf_pos, nbins):
    rust = _rust_or_skip()
    out = np.ones(nbins + 6, dtype=np.int16)
    result = rust._sigmet_decode_ray_current_record_i16(rbuf, rbuf_pos, nbins, out)
    return result, out


def _record_with_codes(start_pos, entries):
    rbuf = np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16)
    pos = start_pos + 1
    for entry in entries:
        if isinstance(entry, (list, tuple, np.ndarray)):
            values = np.asarray(entry, dtype=np.int16)
            rbuf[pos: pos + len(values)] = values
            pos += len(values)
        else:
            rbuf[pos] = entry
            pos += 1
    return rbuf


def test_sigmet_ray_rle_dispatches_current_record_to_private_rust_kernel(
    monkeypatch,
):
    calls = []
    rbuf = np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16)
    out = np.ones(12, dtype=np.int16)

    def kernel(rbuf_arg, rbuf_pos_arg, nbins_arg, out_arg):
        calls.append((rbuf_arg.dtype, rbuf_arg.shape, rbuf_pos_arg, nbins_arg))
        out_arg[:3] = np.array([7, 8, 9], dtype=np.int16)
        return 0, 42

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel
        if name == "_sigmet_decode_ray_current_record_i16"
        else None,
    )
    obj = _sigmet_obj(rbuf, 9)

    status = obj._get_ray(6, out)

    assert status == 0
    assert obj._rbuf_pos == 42
    assert calls == [(np.dtype(np.int16), (3072,), 9, 6)]
    np.testing.assert_array_equal(out[:3], np.array([7, 8, 9], dtype=np.int16))


def test_sigmet_ray_rle_current_record_matches_python_control_flow(monkeypatch):
    rbuf = _record_with_codes(9, [2, -32765, [10, 11, 12], 1])
    expected_status, expected_pos, expected_out = _python_get_ray(
        rbuf, 9, 8, monkeypatch
    )

    result, out = _run_direct_rust(rbuf, 9, 8)

    assert result == (expected_status, expected_pos)
    np.testing.assert_array_equal(out, expected_out)


def test_sigmet_ray_rle_missing_ray_matches_python(monkeypatch):
    rbuf = _record_with_codes(9, [1])
    expected_status, expected_pos, expected_out = _python_get_ray(
        rbuf, 9, 8, monkeypatch
    )

    result, out = _run_direct_rust(rbuf, 9, 8)

    assert result == (expected_status, expected_pos)
    np.testing.assert_array_equal(out, expected_out)


def test_sigmet_ray_rle_corrupt_zero_run_matches_python(monkeypatch):
    rbuf = _record_with_codes(9, [20])
    expected_status, expected_pos, expected_out = _python_get_ray(
        rbuf, 9, 4, monkeypatch
    )

    result, out = _run_direct_rust(rbuf, 9, 4)

    assert result == (expected_status, expected_pos)
    np.testing.assert_array_equal(out, expected_out)


def test_sigmet_ray_rle_returns_none_for_cross_record_split():
    rust = _rust_or_skip()
    rbuf = np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16)
    rbuf[3070] = -32766
    rbuf[3071] = 99
    out = np.ones(12, dtype=np.int16)

    result = rust._sigmet_decode_ray_current_record_i16(rbuf, 3069, 6, out)

    assert result is None
    np.testing.assert_array_equal(out, np.ones(12, dtype=np.int16))


@pytest.mark.parametrize(
    ("rbuf", "out", "match"),
    [
        (
            np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16)[::2],
            np.ones(8, dtype=np.int16),
            "C-contiguous",
        ),
        (
            np.ones((_sigmetfile.RECORD_SIZE // 2) - 1, dtype=np.int16),
            np.ones(8, dtype=np.int16),
            "rbuf length",
        ),
        (
            np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16),
            np.ones(7, dtype=np.int16),
            "out length",
        ),
        (
            np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16),
            np.ones(8, dtype=np.int16)[::2],
            "C-contiguous",
        ),
    ],
)
def test_real_rust_sigmet_ray_rle_direct_rejects_unsafe_inputs(
    rbuf, out, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._sigmet_decode_ray_current_record_i16(rbuf, 0, 2, out)


@pytest.mark.parametrize(
    ("rbuf", "out"),
    [
        (
            np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=">i2"),
            np.ones(8, dtype=np.int16),
        ),
        (
            np.ones(_sigmetfile.RECORD_SIZE // 2, dtype=np.int16),
            np.ones(8, dtype=">i2"),
        ),
    ],
)
def test_real_rust_sigmet_ray_rle_direct_rejects_binding_type_drift(
    rbuf, out
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        rust._sigmet_decode_ray_current_record_i16(rbuf, 0, 2, out)
