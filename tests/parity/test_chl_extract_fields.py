import os

import numpy as np
import pytest

from pyart.io import chl
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_chl_extract_integer_fields"):
        pytest.skip("pyart._rust has no CHL extraction kernel")
    return rust


def _make_chl_file(columns, formats, field_nums=None, scale_info=None):
    if field_nums is None:
        field_nums = list(range(len(formats)))
    dtype = np.dtype(",".join(chl.DATA_FORMAT[format_code] for format_code in formats))
    shape = columns[0].shape
    if dtype.names is None:
        records = np.asarray(columns[0], dtype=chl.DATA_FORMAT[formats[0]]).ravel()
    else:
        records = np.zeros(shape[0] * shape[1], dtype=dtype)
        for idx, column in enumerate(columns):
            records[dtype.names[idx]] = np.asarray(column, dtype=chl.DATA_FORMAT[formats[idx]]).ravel()

    obj = chl.ChlFile.__new__(chl.ChlFile)
    obj.ngates = shape[1]
    obj._dstring = records.tobytes()
    obj._dtype = ",".join(chl.DATA_FORMAT[format_code] for format_code in formats)
    obj._field_nums = list(field_nums)
    obj.fields = {}
    obj.field_info = {}
    for idx, (field_num, format_code) in enumerate(zip(field_nums, formats)):
        scale = (2, 1, 3)
        if scale_info is not None and field_num in scale_info:
            scale = scale_info[field_num]
        obj.field_info[field_num] = {
            "format": format_code,
            "dat_factor": scale[0],
            "dat_bias": scale[1],
            "fld_factor": scale[2],
        }
    return obj


def _mixed_chl_file(scale_info=None):
    return _make_chl_file(
        [
            np.array([[0, 1, 2], [255, 3, 0]], dtype=np.uint8),
            np.array([[0, 10, 20], [30, 0, 65535]], dtype=np.uint16),
            np.array([[0.0, 1.0e-8, 1.0e-5], [np.nan, -1.0e-8, 2.5]], dtype=np.float32),
            np.array([[0, 2**53 + 1, 42], [7, 0, 11]], dtype=np.uint64),
        ],
        [0, 3, 2, 1],
        field_nums=[0, 5, 7, 11],
        scale_info=scale_info,
    )


def _fallback_fields(obj, monkeypatch):
    monkeypatch.setattr(chl, "_rust_kernel", lambda _name: None)
    obj._extract_fields()
    return obj.fields


def test_chl_extract_fields_python_fallback_reference_cases(monkeypatch):
    fields = _fallback_fields(_mixed_chl_file(), monkeypatch)

    assert fields[0].dtype == np.float64
    assert fields[5].dtype == np.float64
    assert fields[7].dtype == np.float32
    assert fields[11].dtype == np.float64
    assert fields[0].fill_value == np.uint8(0)
    assert fields[5].fill_value == np.uint16(0)
    assert fields[7].fill_value == np.float32(0)
    assert fields[11].fill_value == np.uint64(0)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(fields[0]),
        np.array([[True, False, False], [False, False, True]]),
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(fields[7]),
        np.array([[True, True, False], [False, True, False]]),
    )


def test_chl_extract_fields_dispatches_integer_fields_to_private_rust(monkeypatch):
    obj = _mixed_chl_file()
    calls = []

    def kernel(raw_data, ngates, field_nums, formats, dat_factors, dat_biases, fld_factors):
        calls.append((len(raw_data), ngates, field_nums, formats, dat_factors, dat_biases, fld_factors))
        return [
            (
                0,
                np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 0.0]], dtype=np.float64),
                np.array([[True, False, False], [False, False, True]], dtype=bool),
                0,
            )
        ]

    monkeypatch.setattr(
        chl,
        "_rust_kernel",
        lambda name: kernel if name == "_chl_extract_integer_fields" else None,
    )

    obj._extract_fields()

    assert calls == [
        (
            len(obj._dstring),
            3,
            [0, 5, 7, 11],
            [0, 3, 2, 1],
            [2.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 0.0, 1.0],
            [3.0, 3.0, 1.0, 3.0],
        )
    ]
    assert obj.fields[0].fill_value == np.uint8(0)
    np.testing.assert_array_equal(obj.fields[0].data[0], [0.0, 10.0, 20.0])
    assert obj.fields[7].dtype == np.float32


def test_chl_extract_fields_unsafe_scale_keeps_python_path(monkeypatch):
    obj = _make_chl_file(
        [
            np.array([[0, 1, 2]], dtype=np.uint8),
            np.array([[3, 4, 5]], dtype=np.uint8),
        ],
        [0, 0],
        scale_info={0: (2, 1, 0)},
    )

    def rust_kernel(name):
        if name == "_chl_extract_integer_fields":
            raise AssertionError("unsafe CHL scale used Rust kernel")
        return None

    monkeypatch.setattr(chl, "_rust_kernel", rust_kernel)

    obj._extract_fields()

    assert obj.fields[0].dtype == np.float64
    assert 0 in obj.fields


@pytest.mark.parametrize(
    ("format_code", "values"),
    [
        (0, np.array([[0, 1, 2]], dtype=np.uint8)),
        (2, np.array([[0.0, 1.0, 2.0]], dtype=np.float32)),
    ],
)
def test_chl_extract_fields_single_field_keeps_python_oracle_error(
    monkeypatch, format_code, values
):
    obj = _make_chl_file([values], [format_code])

    def rust_kernel(name):
        if name == "_chl_extract_integer_fields":
            raise AssertionError("single-field CHL record used Rust kernel")
        return None

    monkeypatch.setattr(chl, "_rust_kernel", rust_kernel)

    with pytest.raises(TypeError) as excinfo:
        obj._extract_fields()
    assert excinfo.value.args == ("'NoneType' object is not subscriptable",)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust CHL parity",
)
def test_chl_extract_fields_real_rust_matches_python_fallback(monkeypatch):
    expected = _fallback_fields(_mixed_chl_file(), monkeypatch)
    monkeypatch.undo()

    actual_obj = _mixed_chl_file()
    actual_obj._extract_fields()

    assert set(actual_obj.fields) == set(expected)
    for field_num in sorted(expected):
        assert_exact_equal(actual_obj.fields[field_num], expected[field_num])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust CHL checks",
)
def test_chl_extract_integer_fields_direct_rust_helper(monkeypatch):
    rust = _rust_or_skip()
    expected = _fallback_fields(_mixed_chl_file(), monkeypatch)
    obj = _mixed_chl_file()

    result = rust._chl_extract_integer_fields(
        obj._dstring,
        obj.ngates,
        obj._field_nums,
        [obj.field_info[field_num]["format"] for field_num in obj._field_nums],
        [2.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 0.0, 1.0],
        [3.0, 3.0, 1.0, 3.0],
    )

    result_by_field = {field_num: np.ma.masked_array(data, mask=mask) for field_num, data, mask, _fmt in result}
    for field_num in (0, 5, 11):
        np.testing.assert_array_equal(result_by_field[field_num].data, expected[field_num].data)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(result_by_field[field_num]),
            np.ma.getmaskarray(expected[field_num]),
        )

    with pytest.raises(ValueError):
        rust._chl_extract_integer_fields(b"\x00\x01", 3, [0], [0], [1.0], [0.0], [1.0])
