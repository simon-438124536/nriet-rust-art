import os

import numpy as np
import pytest

from pyart.aux_io import rainbow_wrl
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_rainbow_wrl_get_data_u8"):
        pytest.skip("pyart._rust has no RAINBOW data kernels")
    return rust


def _rawdata(databin, datatype="dBZ", datamin=0.0, datamax=360.0, datadepth=8.0):
    return {
        "data": databin,
        "@min": datamin,
        "@max": datamax,
        "@depth": datadepth,
        "@type": datatype,
    }


def _fallback_get_data(rawdata, nrays, nbins, maxbin, monkeypatch):
    monkeypatch.setattr(rainbow_wrl, "_rust_kernel", lambda _name: None)
    return rainbow_wrl._get_data(rawdata, nrays, nbins, maxbin)


def _assert_masked_equal(actual, expected):
    assert_exact_equal(actual, expected)


@pytest.mark.parametrize(
    ("rawdata", "nrays", "nbins", "maxbin"),
    [
        (
            _rawdata(
                np.array([0, 1, 2, 255], dtype=np.uint8),
                "dBZ",
                -32.0,
                64.0,
                8.0,
            ),
            2,
            2,
            3,
        ),
        (
            _rawdata(
                np.array([0, 128, 200, 255], dtype=np.uint8),
                "PhiDP",
                0.0,
                360.0,
                8.0,
            ),
            2,
            2,
            3,
        ),
        (
            _rawdata(
                np.array([[0, 1, 300], [65535, 2, 3]], dtype=np.uint16),
                "uPhiDPu",
                -10.0,
                720.0,
                10.0,
            ),
            2,
            3,
            5,
        ),
    ],
)
def test_rainbow_wrl_get_data_python_fallback_reference_cases(
    monkeypatch, rawdata, nrays, nbins, maxbin
):
    actual = _fallback_get_data(rawdata, nrays, nbins, maxbin, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.shape == (nrays, maxbin)
    assert actual.dtype == np.float64
    assert actual.fill_value == np.float64(-9999.0)


def test_rainbow_wrl_get_data_dispatches_dense_u8_to_private_rust(monkeypatch):
    raw = np.array([0, 1, 2, 3], dtype=np.uint8)
    rawdata = _rawdata(raw, "uPhiDP", -32.0, 64.0, 8.0)
    calls = []
    out = np.array([[-9999.0, 1.0, -9999.0], [2.0, 3.0, -9999.0]], dtype=np.float64)
    mask = np.array([[True, False, True], [False, False, True]], dtype=bool)

    def kernel(databin, nrays, nbins, maxbin, datamin, scale, fill_value, wrap_phidp):
        calls.append(
            (
                databin.dtype,
                databin.shape,
                nrays,
                nbins,
                maxbin,
                datamin,
                scale,
                fill_value,
                wrap_phidp,
            )
        )
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        rainbow_wrl,
        "_rust_kernel",
        lambda name: kernel if name == "_rainbow_wrl_get_data_u8" else None,
    )

    actual = rainbow_wrl._get_data(rawdata, 2, 2, 3)

    assert calls == [
        (np.dtype(np.uint8), (4,), 2, 2, 3, -32.0, 0.375, -9999.0, True)
    ]
    _assert_masked_equal(actual, np.ma.array(out, mask=mask, fill_value=-9999.0))


def test_rainbow_wrl_get_data_custom_fill_value_preserves_float64_payload(monkeypatch):
    rawdata = _rawdata(np.array([0, 1, 2, 0], dtype=np.uint8), "dBZ", -32.0, 64.0, 8.0)
    config_globals = rainbow_wrl.get_fillvalue.__globals__
    old_fill = config_globals["_FILL_VALUE"]
    custom_fill = -9999.125123
    try:
        config_globals["_FILL_VALUE"] = custom_fill
        expected = _fallback_get_data(rawdata, 2, 2, 3, monkeypatch)
        monkeypatch.undo()

        actual = rainbow_wrl._get_data(rawdata, 2, 2, 3)
    finally:
        config_globals["_FILL_VALUE"] = old_fill

    _assert_masked_equal(actual, expected)
    assert actual.data[0, 0] == np.float64(np.float32(custom_fill))
    assert actual.data[0, 2] == custom_fill


def test_rainbow_wrl_get_data_oversized_output_keeps_python_path(monkeypatch):
    rawdata = _rawdata(np.array([], dtype=np.uint8), "dBZ")

    def rust_kernel(name):
        if name in {"_rainbow_wrl_get_data_u8", "_rainbow_wrl_get_data_u16"}:
            raise AssertionError("oversized RAINBOW output used Rust kernel")
        return None

    monkeypatch.setattr(rainbow_wrl, "_rust_kernel", rust_kernel)
    assert (
        rainbow_wrl._get_data_rust(
            rawdata["data"],
            0.0,
            360.0,
            8.0,
            "dBZ",
            rainbow_wrl.RAINBOW_RUST_MAX_OUTPUT_GATES + 1,
            0,
            1,
        )
        is None
    )


@pytest.mark.parametrize(
    ("rawdata", "nrays", "nbins", "maxbin"),
    [
        (_rawdata(np.arange(8, dtype=np.uint8)[::2], "PhiDP"), 2, 2, 3),
        (_rawdata(np.array([0, 1, 2, 3], dtype=object), "dBZ"), 2, 2, 3),
        (_rawdata(np.array([0, "bad", 2, 3], dtype=object), "dBZ"), 2, 2, 3),
        (_rawdata(np.array(1, dtype=np.uint8), "dBZ"), 1, 1, 1),
        (_rawdata(1, "dBZ"), 1, 1, 1),
        (_rawdata(np.array([0, 1, 2], dtype=np.uint8), "dBZ"), 2, 2, 3),
        (_rawdata(np.array([0, 1, 2, 3], dtype=np.uint8), "dBZ"), 2, 2, 1),
        (_rawdata(np.array([0, 1, 2, 3], dtype=np.int16), "dBZ"), 2, 2, 3),
    ],
)
def test_rainbow_wrl_get_data_unsupported_inputs_keep_python_path(
    monkeypatch, rawdata, nrays, nbins, maxbin
):
    def rust_kernel(name):
        if name in {"_rainbow_wrl_get_data_u8", "_rainbow_wrl_get_data_u16"}:
            raise AssertionError(f"unsupported input used Rust kernel {name}")
        return None

    monkeypatch.setattr(rainbow_wrl, "_rust_kernel", rust_kernel)
    try:
        actual = rainbow_wrl._get_data(rawdata, nrays, nbins, maxbin)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_get_data(rawdata, nrays, nbins, maxbin, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_get_data(rawdata, nrays, nbins, maxbin, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust RAINBOW parity",
)
@pytest.mark.parametrize(
    ("rawdata", "nrays", "nbins", "maxbin"),
    [
        (
            _rawdata(
                np.array([0, 1, 2, 255], dtype=np.uint8),
                "dBZ",
                -32.0,
                64.0,
                8.0,
            ),
            2,
            2,
            3,
        ),
        (
            _rawdata(
                np.array([0, 128, 200, 255], dtype=np.uint8),
                "PhiDP",
                0.0,
                360.0,
                8.0,
            ),
            2,
            2,
            3,
        ),
        (
            _rawdata(
                np.array([[0, 1, 300], [65535, 2, 3]], dtype=np.uint16),
                "uPhiDP",
                -10.0,
                720.0,
                10.0,
            ),
            2,
            3,
            5,
        ),
        (_rawdata(np.array([], dtype=np.uint8), "dBZ"), 0, 3, 4),
    ],
)
def test_rainbow_wrl_get_data_real_rust_matches_python_fallback(
    monkeypatch, rawdata, nrays, nbins, maxbin
):
    expected = _fallback_get_data(rawdata, nrays, nbins, maxbin, monkeypatch)
    monkeypatch.undo()

    actual = rainbow_wrl._get_data(rawdata, nrays, nbins, maxbin)

    _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust RAINBOW checks",
)
def test_rainbow_wrl_get_data_direct_rust_helper(monkeypatch):
    rust = _rust_or_skip()
    rawdata = _rawdata(np.array([0, 128, 200, 255], dtype=np.uint8), "PhiDP")
    expected = _fallback_get_data(rawdata, 2, 2, 3, monkeypatch)

    data, mask = rust._rainbow_wrl_get_data_u8(
        rawdata["data"],
        2,
        2,
        3,
        0.0,
        (360.0 - 0.0) / 2**8.0,
        -9999.0,
        True,
    )
    actual = np.ma.array(data, mask=mask, fill_value=-9999.0)
    _assert_masked_equal(actual, expected)

    with pytest.raises(ValueError):
        rust._rainbow_wrl_get_data_u8(rawdata["data"], 3, 2, 3, 0.0, 1.0, -9999.0, False)
