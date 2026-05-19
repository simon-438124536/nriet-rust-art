import numpy as np
import pytest

from pyart.io import _sigmetfile


DBZ2 = 9
DBT = 1
VEL = 3
WIDTH = 4
ZDR = 5
WIDTH2 = 11
KDP = 14
PHIDP = 16
VELC = 17
SQI = 18
RHOHV = 19
RHOHV2 = 20
SQI2 = 23
PHIDP2 = 24
RHOH = 46
RHOH2 = 47
RHOV = 48
RHOV2 = 49
HCLASS = 55
HCLASS2 = 56
PMI8 = 75
PMI16 = 76

LIKE_SQI_TYPES = [SQI, RHOHV, RHOH, RHOV, PMI8]
LIKE_SQI2_TYPES = [RHOHV2, SQI2, RHOH2, RHOV2, PMI16]
U8_LIKE_SQI_CASE = (SQI, "_sigmet_convert_like_sqi_dense_i16")
U8_SIMPLE_CASES = [
    (VEL, "_sigmet_convert_vel_dense_i16"),
    (VELC, "_sigmet_convert_velc_dense_i16"),
    (WIDTH, "_sigmet_convert_width_dense_i16"),
    (ZDR, "_sigmet_convert_zdr_dense_i16"),
    (KDP, "_sigmet_convert_kdp_dense_i16"),
    (PHIDP, "_sigmet_convert_phidp_dense_i16"),
    (HCLASS, "_sigmet_convert_hclass_dense_i16"),
]
U16_SIMPLE_CASES = [
    (SQI2, "_sigmet_convert_like_sqi2_dense_i16"),
    (WIDTH2, "_sigmet_convert_width2_dense_i16"),
    (PHIDP2, "_sigmet_convert_phidp2_dense_i16"),
    (HCLASS2, "_sigmet_convert_hclass2_dense_i16"),
]


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _dbt2_data(values):
    return np.array(values, dtype=np.uint16).view(np.int16)


def _sigmet_u8_data(values):
    byte_values = np.asarray(values, dtype=np.uint8)
    if byte_values.ndim != 2:
        raise AssertionError("byte values must be 2-D")
    raw = np.zeros(
        (byte_values.shape[0], byte_values.shape[1] * 2),
        dtype=np.uint8,
    )
    raw[:, : byte_values.shape[1]] = byte_values
    return raw.view(np.int16)


def _fallback_convert(data_type, data, nbins, monkeypatch):
    monkeypatch.setattr(_sigmetfile, "_rust_kernel", lambda _name: None)
    return _sigmetfile.convert_sigmet_data(data_type, data, nbins)


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))


def _u16_simple_expected(data_type, data, nbins):
    raw = data.view(np.uint16)
    if data_type in LIKE_SQI2_TYPES:
        out = (raw - 1.0) / 65533.0
    elif data_type == WIDTH2:
        out = raw / 100.0
    elif data_type == PHIDP2:
        out = 360.0 * (raw - 1.0) / 65534.0
    elif data_type == HCLASS2:
        out = raw
    else:
        raise AssertionError(f"unexpected data_type {data_type}")

    mask = np.zeros(raw.shape, dtype=bool)
    if data_type != HCLASS2:
        mask[raw == 0] = True
    for ray, nbin in enumerate(nbins):
        if nbin < raw.shape[1]:
            mask[ray, int(nbin):] = True
    return out.astype(np.float32), mask


def _sigmet_u8_view(data):
    nrays, nbin = data.shape
    return data.view("(2,) uint8").reshape(nrays, -1)[:, :nbin]


def _u8_simple_expected(data_type, data, nbins):
    raw = _sigmet_u8_view(data)
    if data_type == VEL:
        out = (raw - 128.0) / 127.0
    elif data_type in LIKE_SQI_TYPES:
        with np.errstate(invalid="ignore"):
            out = np.sqrt((raw - 1.0) / 253.0)
    elif data_type == VELC:
        out = (raw - 128.0) / 127.0 * 75.0
    elif data_type == WIDTH:
        out = raw / 256.0
    elif data_type == ZDR:
        out = (raw - 128.0) / 16.0
    elif data_type == KDP:
        out = np.empty_like(raw, dtype=np.float32)
        out[raw > 128] = 0.25 * np.power(600.0, (raw[raw > 128] - 129.0) / 126.0)
        out[raw < 128] = -0.25 * np.power(600.0, (127.0 - raw[raw < 128]) / 126.0)
        out[raw == 128] = 0.0
    elif data_type == PHIDP:
        out = 180.0 * ((raw - 1.0) / 254.0)
    elif data_type == HCLASS:
        out = raw
    else:
        raise AssertionError(f"unexpected data_type {data_type}")

    mask = np.zeros(raw.shape, dtype=bool)
    if data_type in LIKE_SQI_TYPES or data_type in (VELC, KDP, PHIDP, HCLASS):
        mask[(raw == 0) | (raw == 255)] = True
    else:
        mask[raw == 0] = True
    for ray, nbin in enumerate(nbins):
        if nbin < raw.shape[1]:
            mask[ray, int(nbin):] = True
    return out.astype(np.float32), mask


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 32768, 32769, 65535], [100, 200, 300, 400]]),
            np.array([4, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 32768, 65535], [0, 1, 2]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _dbt2_data([[32768, 32769, 65535]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _dbt2_data(np.empty((0, 4), dtype=np.uint16)),
            np.array([], dtype=np.int16),
        ),
    ],
)
def test_sigmet_like_dbt2_python_fallback_reference_cases(
    monkeypatch, data, nbins
):
    actual = _fallback_convert(DBZ2, data, nbins, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape


def test_sigmet_like_dbt2_dispatches_dense_i16_to_private_rust_kernel(monkeypatch):
    calls = []
    data = _dbt2_data([[0, 32768], [65535, 1]])
    nbins = np.array([2, 1], dtype=np.int16)
    out = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]], dtype=bool)

    def kernel(data_arg, nbins_arg):
        calls.append((data_arg.dtype, data_arg.shape, nbins_arg.dtype, nbins_arg.copy()))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_convert_like_dbt2_dense_i16" else None,
    )

    actual = _sigmetfile.convert_sigmet_data(DBZ2, data, nbins)

    assert len(calls) == 1
    assert calls[0][0:3] == (np.dtype(np.int16), (2, 2), np.dtype(np.int64))
    np.testing.assert_array_equal(calls[0][3], np.array([2, 1], dtype=np.int64))
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("data_type", "data", "nbins"),
    [
        (
            DBZ2,
            _dbt2_data([[0, 32768, 32769, 65535], [100, 200, 300, 400]])[:, ::2],
            np.array([2, 2], dtype=np.int16),
        ),
        (
            DBZ2,
            np.array([[0, -32768], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            DBZ2,
            _dbt2_data([[0, 32768], [1, 2]]),
            np.array([-1, 2], dtype=np.int16),
        ),
        (
            DBZ2,
            _dbt2_data([0, 32768, 65535]),
            np.array([3], dtype=np.int16),
        ),
        (
            DBZ2,
            np.ma.array(_dbt2_data([[0, 32768], [1, 2]])),
            np.array([2, 2], dtype=np.int16),
        ),
    ],
)
def test_sigmet_like_dbt2_unsupported_inputs_keep_python_fallback(
    monkeypatch, data_type, data, nbins
):
    def rust_kernel(name):
        if name != "_sigmet_convert_like_dbt2_dense_i16":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    try:
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_convert(data_type, data, nbins, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_convert(data_type, data, nbins, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 32768, 32769, 65535], [100, 200, 300, 400]]),
            np.array([4, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 32768, 65535], [0, 1, 2]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _dbt2_data([[32768, 32769, 65535]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _dbt2_data(np.empty((0, 4), dtype=np.uint16)),
            np.array([], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_like_dbt2_matches_python_fallback(
    monkeypatch, data, nbins
):
    rust = _rust_or_skip()

    expected = _fallback_convert(DBZ2, data, nbins, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_dbt2_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_dbt2_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    actual = _sigmetfile.convert_sigmet_data(DBZ2, data, nbins)

    assert calls == ["_sigmet_convert_like_dbt2_dense_i16"]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 32768, 32769, 65535], [100, 200, 300, 400]])[:, ::2],
            np.array([2, 2], dtype=np.int16),
        ),
        (
            np.array([[0, -32768], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 32768], [1, 2]]),
            np.array([-1, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([0, 32768, 65535]),
            np.array([3], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_like_dbt2_unsupported_inputs_do_not_dispatch(
    monkeypatch, data, nbins
):
    rust = _rust_or_skip()

    try:
        expected = _fallback_convert(DBZ2, data, nbins, monkeypatch)
    except Exception as expected_error:
        expected = expected_error
    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_dbt2_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_dbt2_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)) as actual_error:
            _sigmetfile.convert_sigmet_data(DBZ2, data, nbins)
        assert actual_error.value.args == expected.args
    else:
        actual = _sigmetfile.convert_sigmet_data(DBZ2, data, nbins)
        _assert_masked_equal(actual, expected)

    assert calls == []


@pytest.mark.parametrize(
    ("data", "nbins", "match"),
    [
        (
            _dbt2_data([[0, 32768], [1, 2]])[:, ::2],
            np.array([1, 1], dtype=np.int64),
            "C-contiguous",
        ),
        (
            _dbt2_data([[0, 32768], [1, 2]]),
            np.array([2], dtype=np.int64),
            "nbins length",
        ),
        (
            _dbt2_data([[0, 32768], [1, 2]]),
            np.array([2, -1], dtype=np.int64),
            "non-negative",
        ),
    ],
)
def test_real_rust_sigmet_like_dbt2_direct_rejects_unsafe_inputs(
    data, nbins, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._sigmet_convert_like_dbt2_dense_i16(data, nbins)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (np.array([[0, -32768]], dtype=">i2"), np.array([2], dtype=np.int64)),
        (_dbt2_data([0, 32768]), np.array([2], dtype=np.int64)),
        (_dbt2_data([[0, 32768]]), np.array([2], dtype=np.int16)),
        (_dbt2_data([[0, 32768]]), np.array([[2]], dtype=np.int64)),
    ],
)
def test_real_rust_sigmet_like_dbt2_direct_rejects_binding_type_drift(
    data, nbins
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        rust._sigmet_convert_like_dbt2_dense_i16(data, nbins)


def test_real_rust_sigmet_like_dbt2_direct_return_data_and_mask():
    rust = _rust_or_skip()

    data, mask = rust._sigmet_convert_like_dbt2_dense_i16(
        _dbt2_data([[0, 32768, 32769, 65535], [100, 200, 300, 400]]),
        np.array([4, 2], dtype=np.int64),
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(
        data,
        np.array(
            [
                [-327.68, 0.0, 0.01, 327.67],
                [-326.68, -325.68, -324.68, -323.68],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, False, False, False],
                [False, False, True, True],
            ],
            dtype=bool,
        ),
    )


@pytest.mark.parametrize(("data_type", "_kernel_name"), U8_SIMPLE_CASES)
def test_sigmet_u8_simple_formula_and_mask_oracle(
    monkeypatch, data_type, _kernel_name
):
    data = _sigmet_u8_data(
        [
            [0, 1, 127, 128, 255],
            [255, 0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
        ]
    )
    nbins = np.array([5, 2, 0, 10], dtype=np.int64)

    actual = _fallback_convert(data_type, data, nbins, monkeypatch)
    expected_data, expected_mask = _u8_simple_expected(data_type, data, nbins)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape
    assert isinstance(actual.mask, np.ndarray)
    np.testing.assert_array_equal(actual.data, expected_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), expected_mask)


def test_sigmet_like_sqi_keeps_python_warning_before_tail_mask(monkeypatch):
    data = _sigmet_u8_data([[1, 0, 255, 0]])
    nbins = np.array([2], dtype=np.int16)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        actual = _fallback_convert(SQI, data, nbins, monkeypatch)

    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape
    assert np.isnan(actual.data[0, 1])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[False, True, True, True]], dtype=bool),
    )


@pytest.mark.parametrize("data_type", LIKE_SQI_TYPES)
def test_sigmet_like_sqi_dispatches_only_when_raw_zero_absent(
    monkeypatch, data_type
):
    calls = []
    data = _sigmet_u8_data([[1, 2, 255], [3, 4, 5]])
    nbins = np.array([3, 2], dtype=np.int16)
    out = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    mask = np.array([[False, False, True], [False, False, True]], dtype=bool)

    def kernel(data_arg, nbins_arg):
        calls.append((data_arg.dtype, data_arg.shape, nbins_arg.dtype, nbins_arg.copy()))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_convert_like_sqi_dense_i16" else None,
    )

    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert len(calls) == 1
    assert calls[0][0:3] == (np.dtype(np.int16), (2, 3), np.dtype(np.int64))
    np.testing.assert_array_equal(calls[0][3], np.array([3, 2], dtype=np.int64))
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize("data_type", LIKE_SQI_TYPES)
def test_sigmet_like_sqi_raw_zero_keeps_python_warning_and_fallback(
    monkeypatch, data_type
):
    data = _sigmet_u8_data([[1, 2, 255, 0]])
    nbins = np.array([2], dtype=np.int16)

    def rust_kernel(name):
        if name != "_sigmet_convert_like_sqi_dense_i16":
            return None

        def fail(*_args):
            raise AssertionError("raw-zero like_sqi input used Rust")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert np.isnan(actual.data[0, 3])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[False, False, True, True]], dtype=bool),
    )


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _sigmet_u8_data([[0, 1, 127, 128, 255], [255, 0, 1, 2, 3]]),
            np.array([5, 2], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[0, 1, 255], [0, 1, 2]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _sigmet_u8_data([[1, 2, 255]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _sigmet_u8_data(np.empty((2, 0), dtype=np.uint8)),
            np.array([0, 2], dtype=np.int16),
        ),
    ],
)
@pytest.mark.parametrize(("data_type", "_kernel_name"), U8_SIMPLE_CASES)
def test_sigmet_u8_simple_python_fallback_reference_cases(
    monkeypatch, data_type, _kernel_name, data, nbins
):
    actual = _fallback_convert(data_type, data, nbins, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape
    assert isinstance(actual.mask, np.ndarray)


@pytest.mark.parametrize(("data_type", "kernel_name"), U8_SIMPLE_CASES)
def test_sigmet_u8_simple_dispatches_dense_i16_to_private_rust_kernel(
    monkeypatch, data_type, kernel_name
):
    calls = []
    data = _sigmet_u8_data([[0, 1], [255, 2]])
    nbins = np.array([2, 1], dtype=np.int16)
    out = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]], dtype=bool)

    def kernel(data_arg, nbins_arg):
        calls.append((data_arg.dtype, data_arg.shape, nbins_arg.dtype, nbins_arg.copy()))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )

    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert len(calls) == 1
    assert calls[0][0:3] == (np.dtype(np.int16), (2, 2), np.dtype(np.int64))
    np.testing.assert_array_equal(calls[0][3], np.array([2, 1], dtype=np.int64))
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _sigmet_u8_data([[1, 2, 127, 128, 255], [255, 1, 2, 3, 4]]),
            np.array([5, 2], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[1, 2, 255], [3, 4, 5]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _sigmet_u8_data([[1, 2, 255]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _sigmet_u8_data(np.empty((2, 0), dtype=np.uint8)),
            np.array([0, 2], dtype=np.int16),
        ),
    ],
)
@pytest.mark.parametrize("data_type", LIKE_SQI_TYPES)
def test_real_rust_sigmet_like_sqi_matches_python_fallback_without_raw_zero(
    monkeypatch, data_type, data, nbins
):
    rust = _rust_or_skip()

    expected = _fallback_convert(data_type, data, nbins, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_sqi_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_sqi_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert calls == ["_sigmet_convert_like_sqi_dense_i16"]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize("data_type", LIKE_SQI_TYPES)
def test_real_rust_sigmet_like_sqi_raw_zero_does_not_dispatch(
    monkeypatch, data_type
):
    rust = _rust_or_skip()
    data = _sigmet_u8_data([[1, 2, 255, 0]])
    nbins = np.array([2], dtype=np.int16)
    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_sqi_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_sqi_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert calls == []
    assert np.isnan(actual.data[0, 3])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[False, False, True, True]], dtype=bool),
    )


@pytest.mark.parametrize(
    ("data", "nbins", "match"),
    [
        (
            _sigmet_u8_data([[1, 2], [3, 4]])[:, ::2],
            np.array([1, 1], dtype=np.int64),
            "C-contiguous",
        ),
        (
            _sigmet_u8_data([[1, 2], [3, 4]]),
            np.array([2], dtype=np.int64),
            "nbins length",
        ),
        (
            _sigmet_u8_data(np.empty((0, 4), dtype=np.uint8)),
            np.array([], dtype=np.int64),
            "cannot reshape",
        ),
        (
            _sigmet_u8_data([[1, 2], [3, 4]]),
            np.array([2, -1], dtype=np.int64),
            "non-negative",
        ),
    ],
)
def test_real_rust_sigmet_like_sqi_direct_rejects_unsafe_inputs(data, nbins, match):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._sigmet_convert_like_sqi_dense_i16(data, nbins)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (np.array([[1, 2]], dtype=">i2"), np.array([2], dtype=np.int64)),
        (_sigmet_u8_data([[1, 2]]).reshape(2), np.array([2], dtype=np.int64)),
        (_sigmet_u8_data([[1, 2]]), np.array([2], dtype=np.int16)),
        (_sigmet_u8_data([[1, 2]]), np.array([[2]], dtype=np.int64)),
    ],
)
def test_real_rust_sigmet_like_sqi_direct_rejects_binding_type_drift(
    data, nbins
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        rust._sigmet_convert_like_sqi_dense_i16(data, nbins)


def test_real_rust_sigmet_like_sqi_direct_return_data_and_mask():
    rust = _rust_or_skip()
    source = _sigmet_u8_data([[0, 1, 2, 255], [3, 4, 5, 6]])
    nbins = np.array([4, 2], dtype=np.int64)

    data, mask = rust._sigmet_convert_like_sqi_dense_i16(source, nbins)
    expected_data, expected_mask = _u8_simple_expected(SQI, source, nbins)

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(mask, expected_mask)


def test_real_rust_sigmet_kdp_direct_matches_all_byte_codes():
    rust = _rust_or_skip()
    source = _sigmet_u8_data([np.arange(256, dtype=np.uint8)])
    nbins = np.array([256], dtype=np.int64)

    data, mask = rust._sigmet_convert_kdp_dense_i16(source, nbins)
    expected_data, expected_mask = _u8_simple_expected(KDP, source, nbins)

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize(("data_type", "kernel_name"), U8_SIMPLE_CASES)
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _sigmet_u8_data([[0, 1, 127, 128, 255], [255, 0, 1, 2, 3]])[:, ::2],
            np.array([3, 3], dtype=np.int16),
        ),
        (
            np.array([[0, -1], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[0, 1, 2], [3, 4, 5]]),
            np.array([-1, 3], dtype=np.int16),
        ),
        (
            _sigmet_u8_data(np.empty((0, 4), dtype=np.uint8)),
            np.array([], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[0, 1, 255]]).reshape(3),
            np.array([3], dtype=np.int16),
        ),
        (
            np.ma.array(_sigmet_u8_data([[0, 1], [2, 3]])),
            np.array([2, 2], dtype=np.int16),
        ),
    ],
)
def test_sigmet_u8_simple_unsupported_inputs_keep_python_fallback(
    monkeypatch, data_type, kernel_name, data, nbins
):
    def rust_kernel(name):
        if name != kernel_name:
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    try:
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_convert(data_type, data, nbins, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_convert(data_type, data, nbins, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(("data_type", "kernel_name"), U8_SIMPLE_CASES)
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _sigmet_u8_data([[0, 1, 127, 128, 255], [255, 0, 1, 2, 3]]),
            np.array([5, 2], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[0, 1, 255], [0, 1, 2]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _sigmet_u8_data([[1, 2, 255]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _sigmet_u8_data(np.empty((2, 0), dtype=np.uint8)),
            np.array([0, 2], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_u8_simple_matches_python_fallback(
    monkeypatch, data_type, kernel_name, data, nbins
):
    rust = _rust_or_skip()

    expected = _fallback_convert(data_type, data, nbins, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == kernel_name:
            calls.append(name)
            return getattr(rust, kernel_name)
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert calls == [kernel_name]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(("data_type", "kernel_name"), U8_SIMPLE_CASES)
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _sigmet_u8_data([[0, 1, 127, 128, 255], [255, 0, 1, 2, 3]])[:, ::2],
            np.array([3, 3], dtype=np.int16),
        ),
        (
            np.array([[0, -1], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[0, 1, 2], [3, 4, 5]]),
            np.array([-1, 3], dtype=np.int16),
        ),
        (
            _sigmet_u8_data(np.empty((0, 4), dtype=np.uint8)),
            np.array([], dtype=np.int16),
        ),
        (
            _sigmet_u8_data([[0, 1, 255]]).reshape(3),
            np.array([3], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_u8_simple_unsupported_inputs_do_not_dispatch(
    monkeypatch, data_type, kernel_name, data, nbins
):
    rust = _rust_or_skip()

    try:
        expected = _fallback_convert(data_type, data, nbins, monkeypatch)
    except Exception as expected_error:
        expected = expected_error
    calls = []

    def rust_kernel(name):
        if name == kernel_name:
            calls.append(name)
            return getattr(rust, kernel_name)
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)) as actual_error:
            _sigmetfile.convert_sigmet_data(data_type, data, nbins)
        assert actual_error.value.args == expected.args
    else:
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)
        _assert_masked_equal(actual, expected)

    assert calls == []


@pytest.mark.parametrize("kernel_name", [case[1] for case in U8_SIMPLE_CASES])
@pytest.mark.parametrize(
    ("data", "nbins", "match"),
    [
        (
            _sigmet_u8_data([[0, 1], [2, 3]])[:, ::2],
            np.array([1, 1], dtype=np.int64),
            "C-contiguous",
        ),
        (
            _sigmet_u8_data([[0, 1], [2, 3]]),
            np.array([2], dtype=np.int64),
            "nbins length",
        ),
        (
            _sigmet_u8_data(np.empty((0, 4), dtype=np.uint8)),
            np.array([], dtype=np.int64),
            "cannot reshape",
        ),
        (
            _sigmet_u8_data([[0, 1], [2, 3]]),
            np.array([2, -1], dtype=np.int64),
            "non-negative",
        ),
    ],
)
def test_real_rust_sigmet_u8_simple_direct_rejects_unsafe_inputs(
    kernel_name, data, nbins, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        getattr(rust, kernel_name)(data, nbins)


@pytest.mark.parametrize("kernel_name", [case[1] for case in U8_SIMPLE_CASES])
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (np.array([[0, -1]], dtype=">i2"), np.array([2], dtype=np.int64)),
        (_sigmet_u8_data([[0, 1]]).reshape(2), np.array([2], dtype=np.int64)),
        (_sigmet_u8_data([[0, 1]]), np.array([2], dtype=np.int16)),
        (_sigmet_u8_data([[0, 1]]), np.array([[2]], dtype=np.int64)),
    ],
)
def test_real_rust_sigmet_u8_simple_direct_rejects_binding_type_drift(
    kernel_name, data, nbins
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        getattr(rust, kernel_name)(data, nbins)


@pytest.mark.parametrize(("data_type", "kernel_name"), U8_SIMPLE_CASES)
def test_real_rust_sigmet_u8_simple_direct_return_data_and_mask(
    data_type, kernel_name
):
    rust = _rust_or_skip()
    source = _sigmet_u8_data(
        [
            [0, 1, 127, 128, 255],
            [255, 0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
        ]
    )
    nbins = np.array([5, 2, 0, 10], dtype=np.int64)

    data, mask = getattr(rust, kernel_name)(source, nbins)
    expected_data, expected_mask = _u8_simple_expected(data_type, source, nbins)

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("kernel_name", [case[1] for case in U8_SIMPLE_CASES])
def test_real_rust_sigmet_u8_simple_direct_accepts_int64_max_empty_tail(
    kernel_name,
):
    rust = _rust_or_skip()

    data, mask = getattr(rust, kernel_name)(
        _sigmet_u8_data(np.empty((1, 0), dtype=np.uint8)),
        np.array([np.iinfo(np.int64).max], dtype=np.int64),
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    assert data.shape == (1, 0)
    assert mask.shape == (1, 0)


@pytest.mark.parametrize(
    "data_type", [RHOHV2, SQI2, RHOH2, RHOV2, PMI16, WIDTH2, PHIDP2, HCLASS2]
)
def test_sigmet_u16_simple_formula_and_mask_oracle(monkeypatch, data_type):
    data = _dbt2_data(
        [
            [0, 1, 2, 65534, 65535],
            [65535, 0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
        ]
    )
    nbins = np.array([5, 2, 0, 10], dtype=np.int64)

    actual = _fallback_convert(data_type, data, nbins, monkeypatch)
    expected_data, expected_mask = _u16_simple_expected(data_type, data, nbins)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape
    assert isinstance(actual.mask, np.ndarray)
    np.testing.assert_array_equal(actual.data, expected_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), expected_mask)


def test_sigmet_hclass2_all_valid_keeps_ndarray_mask(monkeypatch):
    data = _dbt2_data([[0, 1, 65535]])
    nbins = np.array([3], dtype=np.int64)

    actual = _fallback_convert(HCLASS2, data, nbins, monkeypatch)

    assert isinstance(actual.mask, np.ndarray)
    assert actual.mask.shape == data.shape
    assert not actual.mask.any()
    np.testing.assert_array_equal(actual.data, np.array([[0, 1, 65535]], dtype=np.float32))


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 1, 2, 65534, 65535], [65535, 0, 1, 2, 3]]),
            np.array([5, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 1, 65535], [0, 1, 2]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _dbt2_data([[1, 2, 65535]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _dbt2_data(np.empty((0, 4), dtype=np.uint16)),
            np.array([], dtype=np.int16),
        ),
        (
            np.empty((2, 0), dtype=np.int16),
            np.array([0, 2], dtype=np.int16),
        ),
        (
            np.empty((0, 0), dtype=np.int16),
            np.array([], dtype=np.int16),
        ),
    ],
)
@pytest.mark.parametrize("data_type", [SQI2, WIDTH2, PHIDP2, HCLASS2])
def test_sigmet_u16_simple_python_fallback_reference_cases(
    monkeypatch, data_type, data, nbins
):
    actual = _fallback_convert(data_type, data, nbins, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape
    assert isinstance(actual.mask, np.ndarray)


@pytest.mark.parametrize(("data_type", "kernel_name"), U16_SIMPLE_CASES)
def test_sigmet_u16_simple_dispatches_dense_i16_to_private_rust_kernel(
    monkeypatch, data_type, kernel_name
):
    calls = []
    data = _dbt2_data([[0, 1], [65535, 2]])
    nbins = np.array([2, 1], dtype=np.int16)
    out = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]], dtype=bool)

    def kernel(data_arg, nbins_arg):
        calls.append((data_arg.dtype, data_arg.shape, nbins_arg.dtype, nbins_arg.copy()))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )

    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert len(calls) == 1
    assert calls[0][0:3] == (np.dtype(np.int16), (2, 2), np.dtype(np.int64))
    np.testing.assert_array_equal(calls[0][3], np.array([2, 1], dtype=np.int64))
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize("data_type", LIKE_SQI2_TYPES)
def test_sigmet_like_sqi2_family_dispatches_same_kernel(monkeypatch, data_type):
    calls = []
    data = _dbt2_data([[1, 2]])
    nbins = np.array([2], dtype=np.int16)

    def kernel(data_arg, nbins_arg):
        calls.append((data_arg.dtype, data_arg.shape, nbins_arg.dtype))
        return (
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[False, False]], dtype=bool),
        )

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_convert_like_sqi2_dense_i16" else None,
    )

    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert calls == [(np.dtype(np.int16), (1, 2), np.dtype(np.int64))]
    np.testing.assert_array_equal(actual.data, np.array([[0.0, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.array([[False, False]]))


@pytest.mark.parametrize(("data_type", "kernel_name"), U16_SIMPLE_CASES)
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 1, 2, 65534, 65535], [65535, 0, 1, 2, 3]])[:, ::2],
            np.array([3, 3], dtype=np.int16),
        ),
        (
            np.array([[0, -1], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 1, 2], [3, 4, 5]]),
            np.array([-1, 3], dtype=np.int16),
        ),
        (
            _dbt2_data([0, 1, 65535]),
            np.array([3], dtype=np.int16),
        ),
        (
            np.ma.array(_dbt2_data([[0, 1], [2, 3]])),
            np.array([2, 2], dtype=np.int16),
        ),
    ],
)
def test_sigmet_u16_simple_unsupported_inputs_keep_python_fallback(
    monkeypatch, data_type, kernel_name, data, nbins
):
    def rust_kernel(name):
        if name != kernel_name:
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    try:
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_convert(data_type, data, nbins, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_convert(data_type, data, nbins, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(("data_type", "kernel_name"), U16_SIMPLE_CASES)
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 1, 2, 65534, 65535], [65535, 0, 1, 2, 3]]),
            np.array([5, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 1, 65535], [0, 1, 2]]),
            np.array([0, 3], dtype=np.uint16),
        ),
        (
            _dbt2_data([[1, 2, 65535]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            _dbt2_data(np.empty((0, 4), dtype=np.uint16)),
            np.array([], dtype=np.int16),
        ),
        (
            np.empty((2, 0), dtype=np.int16),
            np.array([0, 2], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_u16_simple_matches_python_fallback(
    monkeypatch, data_type, kernel_name, data, nbins
):
    rust = _rust_or_skip()

    expected = _fallback_convert(data_type, data, nbins, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == kernel_name:
            calls.append(name)
            return getattr(rust, kernel_name)
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)

    assert calls == [kernel_name]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(("data_type", "kernel_name"), U16_SIMPLE_CASES)
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data([[0, 1, 2, 65534, 65535], [65535, 0, 1, 2, 3]])[:, ::2],
            np.array([3, 3], dtype=np.int16),
        ),
        (
            np.array([[0, -1], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0, 1, 2], [3, 4, 5]]),
            np.array([-1, 3], dtype=np.int16),
        ),
        (
            _dbt2_data([0, 1, 65535]),
            np.array([3], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_u16_simple_unsupported_inputs_do_not_dispatch(
    monkeypatch, data_type, kernel_name, data, nbins
):
    rust = _rust_or_skip()

    try:
        expected = _fallback_convert(data_type, data, nbins, monkeypatch)
    except Exception as expected_error:
        expected = expected_error
    calls = []

    def rust_kernel(name):
        if name == kernel_name:
            calls.append(name)
            return getattr(rust, kernel_name)
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)) as actual_error:
            _sigmetfile.convert_sigmet_data(data_type, data, nbins)
        assert actual_error.value.args == expected.args
    else:
        actual = _sigmetfile.convert_sigmet_data(data_type, data, nbins)
        _assert_masked_equal(actual, expected)

    assert calls == []


@pytest.mark.parametrize("kernel_name", [case[1] for case in U16_SIMPLE_CASES])
@pytest.mark.parametrize(
    ("data", "nbins", "match"),
    [
        (
            _dbt2_data([[0, 1], [2, 3]])[:, ::2],
            np.array([1, 1], dtype=np.int64),
            "C-contiguous",
        ),
        (
            _dbt2_data([[0, 1], [2, 3]]),
            np.array([2], dtype=np.int64),
            "nbins length",
        ),
        (
            _dbt2_data([[0, 1], [2, 3]]),
            np.array([2, -1], dtype=np.int64),
            "non-negative",
        ),
    ],
)
def test_real_rust_sigmet_u16_simple_direct_rejects_unsafe_inputs(
    kernel_name, data, nbins, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        getattr(rust, kernel_name)(data, nbins)


@pytest.mark.parametrize("kernel_name", [case[1] for case in U16_SIMPLE_CASES])
@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (np.array([[0, -1]], dtype=">i2"), np.array([2], dtype=np.int64)),
        (_dbt2_data([0, 1]), np.array([2], dtype=np.int64)),
        (_dbt2_data([[0, 1]]), np.array([2], dtype=np.int16)),
        (_dbt2_data([[0, 1]]), np.array([[2]], dtype=np.int64)),
    ],
)
def test_real_rust_sigmet_u16_simple_direct_rejects_binding_type_drift(
    kernel_name, data, nbins
):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        getattr(rust, kernel_name)(data, nbins)


@pytest.mark.parametrize(("data_type", "kernel_name"), U16_SIMPLE_CASES)
def test_real_rust_sigmet_u16_simple_direct_return_data_and_mask(
    data_type, kernel_name
):
    rust = _rust_or_skip()
    source = _dbt2_data(
        [
            [0, 1, 2, 65534, 65535],
            [65535, 0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
        ]
    )
    nbins = np.array([5, 2, 0, 10], dtype=np.int64)

    data, mask = getattr(rust, kernel_name)(source, nbins)
    expected_data, expected_mask = _u16_simple_expected(data_type, source, nbins)

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("kernel_name", [case[1] for case in U16_SIMPLE_CASES])
def test_real_rust_sigmet_u16_simple_direct_accepts_int64_max_empty_tail(
    kernel_name,
):
    rust = _rust_or_skip()

    data, mask = getattr(rust, kernel_name)(
        np.empty((1, 0), dtype=np.int16),
        np.array([np.iinfo(np.int64).max], dtype=np.int64),
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    assert data.shape == (1, 0)
    assert mask.shape == (1, 0)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data(
                [[0x0000, 0x0101, 0x4001, 0x8040], [0x00FF, 0xFF00, 0x4142, 0x4243]]
            ),
            np.array([4, 2], dtype=np.int16),
        ),
        (
            np.ones((2, 2), dtype=np.int16) * 257,
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0x0101, 0x0101]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            np.empty((1, 0), dtype=np.int16),
            np.array([0], dtype=np.int16),
        ),
    ],
)
def test_sigmet_like_dbt_python_fallback_reference_cases(monkeypatch, data, nbins):
    actual = _fallback_convert(DBT, data, nbins, monkeypatch)

    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    assert actual.shape == data.shape


def test_sigmet_like_dbt_dispatches_dense_i16_to_private_rust_kernel(monkeypatch):
    calls = []
    data = np.ones((2, 2), dtype=np.int16) * 257
    nbins = np.array([2, 1], dtype=np.int16)
    out = np.array([[-31.5, -31.5], [-31.5, -31.5]], dtype=np.float32)
    mask = np.array([[False, False], [False, True]], dtype=bool)

    def kernel(data_arg, nbins_arg):
        calls.append((data_arg.dtype, data_arg.shape, nbins_arg.dtype, nbins_arg.copy()))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        _sigmetfile,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_convert_like_dbt_dense_i16" else None,
    )

    actual = _sigmetfile.convert_sigmet_data(DBT, data, nbins)

    assert len(calls) == 1
    assert calls[0][0:3] == (np.dtype(np.int16), (2, 2), np.dtype(np.int64))
    np.testing.assert_array_equal(calls[0][3], np.array([2, 1], dtype=np.int64))
    assert actual.dtype == np.float32
    assert actual.fill_value == -9999.0
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data(
                [[0x0000, 0x0101, 0x4001, 0x8040], [0x00FF, 0xFF00, 0x4142, 0x4243]]
            )[:, ::2],
            np.array([2, 2], dtype=np.int16),
        ),
        (
            np.array([[0, 257], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            np.ones((2, 2), dtype=np.int16) * 257,
            np.array([-1, 2], dtype=np.int16),
        ),
        (
            np.ones((0, 2), dtype=np.int16) * 257,
            np.array([], dtype=np.int16),
        ),
        (
            np.ones((2,), dtype=np.int16) * 257,
            np.array([2], dtype=np.int16),
        ),
        (
            np.ma.array(np.ones((2, 2), dtype=np.int16) * 257),
            np.array([2, 2], dtype=np.int16),
        ),
    ],
)
def test_sigmet_like_dbt_unsupported_inputs_keep_python_fallback(
    monkeypatch, data, nbins
):
    def rust_kernel(name):
        if name != "_sigmet_convert_like_dbt_dense_i16":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    try:
        actual = _sigmetfile.convert_sigmet_data(DBT, data, nbins)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_convert(DBT, data, nbins, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_convert(DBT, data, nbins, monkeypatch)
        _assert_masked_equal(actual, expected)


def test_sigmet_like_dbt_zero_ray_keeps_python_reshape_error(monkeypatch):
    data = np.ones((0, 2), dtype=np.int16) * 257
    nbins = np.array([], dtype=np.int16)

    def rust_kernel(name):
        if name != "_sigmet_convert_like_dbt_dense_i16":
            return None

        def fail(*_args):
            raise AssertionError("zero-ray like_dbt input used Rust")

        return fail

    with pytest.raises(ValueError) as expected_error:
        _fallback_convert(DBT, data, nbins, monkeypatch)

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    with pytest.raises(ValueError) as actual_error:
        _sigmetfile.convert_sigmet_data(DBT, data, nbins)

    assert actual_error.value.args == expected_error.value.args


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data(
                [[0x0000, 0x0101, 0x4001, 0x8040], [0x00FF, 0xFF00, 0x4142, 0x4243]]
            ),
            np.array([4, 2], dtype=np.int16),
        ),
        (
            np.ones((2, 2), dtype=np.int16) * 257,
            np.array([2, 2], dtype=np.int16),
        ),
        (
            _dbt2_data([[0x0101, 0x0101]]),
            np.array([2**40], dtype=np.int64),
        ),
        (
            np.empty((1, 0), dtype=np.int16),
            np.array([0], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_like_dbt_matches_python_fallback(
    monkeypatch, data, nbins
):
    rust = _rust_or_skip()

    expected = _fallback_convert(DBT, data, nbins, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_dbt_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_dbt_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    actual = _sigmetfile.convert_sigmet_data(DBT, data, nbins)

    assert calls == ["_sigmet_convert_like_dbt_dense_i16"]
    _assert_masked_equal(actual, expected)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (
            _dbt2_data(
                [[0x0000, 0x0101, 0x4001, 0x8040], [0x00FF, 0xFF00, 0x4142, 0x4243]]
            )[:, ::2],
            np.array([2, 2], dtype=np.int16),
        ),
        (
            np.array([[0, 257], [1, 2]], dtype=">i2"),
            np.array([2, 2], dtype=np.int16),
        ),
        (
            np.ones((2, 2), dtype=np.int16) * 257,
            np.array([-1, 2], dtype=np.int16),
        ),
        (
            np.ones((0, 2), dtype=np.int16) * 257,
            np.array([], dtype=np.int16),
        ),
        (
            np.ones((2,), dtype=np.int16) * 257,
            np.array([2], dtype=np.int16),
        ),
    ],
)
def test_real_rust_sigmet_like_dbt_unsupported_inputs_do_not_dispatch(
    monkeypatch, data, nbins
):
    rust = _rust_or_skip()

    try:
        expected = _fallback_convert(DBT, data, nbins, monkeypatch)
    except Exception as expected_error:
        expected = expected_error
    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_dbt_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_dbt_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)) as actual_error:
            _sigmetfile.convert_sigmet_data(DBT, data, nbins)
        assert actual_error.value.args == expected.args
    else:
        actual = _sigmetfile.convert_sigmet_data(DBT, data, nbins)
        _assert_masked_equal(actual, expected)

    assert calls == []


def test_real_rust_sigmet_like_dbt_zero_ray_does_not_dispatch(monkeypatch):
    rust = _rust_or_skip()
    data = np.ones((0, 2), dtype=np.int16) * 257
    nbins = np.array([], dtype=np.int16)

    with pytest.raises(ValueError) as expected_error:
        _fallback_convert(DBT, data, nbins, monkeypatch)

    calls = []

    def rust_kernel(name):
        if name == "_sigmet_convert_like_dbt_dense_i16":
            calls.append(name)
            return rust._sigmet_convert_like_dbt_dense_i16
        return None

    monkeypatch.setattr(_sigmetfile, "_rust_kernel", rust_kernel)
    with pytest.raises(ValueError) as actual_error:
        _sigmetfile.convert_sigmet_data(DBT, data, nbins)

    assert calls == []
    assert actual_error.value.args == expected_error.value.args


@pytest.mark.parametrize(
    ("data", "nbins", "match"),
    [
        (
            np.ones((2, 2), dtype=np.int16)[:, ::2],
            np.array([1, 1], dtype=np.int64),
            "C-contiguous",
        ),
        (
            np.ones((2, 2), dtype=np.int16) * 257,
            np.array([2], dtype=np.int64),
            "nbins length",
        ),
        (
            np.ones((2, 2), dtype=np.int16) * 257,
            np.array([2, -1], dtype=np.int64),
            "non-negative",
        ),
    ],
)
def test_real_rust_sigmet_like_dbt_direct_rejects_unsafe_inputs(data, nbins, match):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        rust._sigmet_convert_like_dbt_dense_i16(data, nbins)


@pytest.mark.parametrize(
    ("data", "nbins"),
    [
        (np.array([[0, 257]], dtype=">i2"), np.array([2], dtype=np.int64)),
        (np.ones((2,), dtype=np.int16) * 257, np.array([2], dtype=np.int64)),
        (np.ones((1, 2), dtype=np.int16) * 257, np.array([2], dtype=np.int16)),
        (np.ones((1, 2), dtype=np.int16) * 257, np.array([[2]], dtype=np.int64)),
    ],
)
def test_real_rust_sigmet_like_dbt_direct_rejects_binding_type_drift(data, nbins):
    rust = _rust_or_skip()

    with pytest.raises(TypeError):
        rust._sigmet_convert_like_dbt_dense_i16(data, nbins)


def test_real_rust_sigmet_like_dbt_direct_return_data_and_mask():
    rust = _rust_or_skip()

    data, mask = rust._sigmet_convert_like_dbt_dense_i16(
        _dbt2_data(
            [[0x0000, 0x0101, 0x4001, 0x8040], [0x00FF, 0xFF00, 0x4142, 0x4243]]
        ),
        np.array([4, 2], dtype=np.int64),
    )

    assert data.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(
        data,
        np.array(
            [
                [-32.0, -32.0, -31.5, -31.5],
                [95.5, -32.0, -32.0, 95.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, True, False, False],
                [False, True, True, True],
            ],
            dtype=bool,
        ),
    )
