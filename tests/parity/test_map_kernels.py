import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.map import _load_nn_field_data  # noqa: E402


def _field_data_object_array():
    data = np.empty((2, 3), dtype=object)
    data[0, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    data[0, 1] = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    data[0, 2] = [20.0, 21.0, 22.0]
    data[1, 0] = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    data[1, 1] = np.array([110.0, 111.0, 112.0], dtype=np.float32)
    data[1, 2] = [120.0, 121.0, 122.0]
    return data


def _expected_loaded_values(data, r_nums, e_nums, nfields, npoints, dtype=np.float64):
    expected = np.empty((npoints, nfields), dtype=dtype)
    for i in range(npoints):
        r_num = int(r_nums[i])
        e_num = int(e_nums[i])
        for j in range(nfields):
            expected[i, j] = data[j, r_num][e_num]
    return expected


def test_load_nn_field_data_python_fallback_matches_object_array_oracle(monkeypatch):
    monkeypatch.setattr(_load_nn_field_data, "_rust_kernel", lambda: None)
    data = _field_data_object_array()
    r_nums = np.array([0, 1, 2, 0], dtype=np.intc)
    e_nums = np.array([2, 1, 0, 1], dtype=np.intc)
    sdata = np.full((4, 2), -999.0, dtype=np.float64)

    result = _load_nn_field_data._load_nn_field_data(
        data, 2, 4, r_nums, e_nums, sdata
    )

    assert result is None
    np.testing.assert_array_equal(
        sdata, _expected_loaded_values(data, r_nums, e_nums, 2, 4)
    )


def test_load_nn_field_data_uses_private_rust_kernel_for_oracle_arrays(monkeypatch):
    calls = []
    data = _field_data_object_array()
    r_nums = np.array([2, 1, 0], dtype=np.intc)
    e_nums = np.array([0, 2, 1], dtype=np.intc)
    sdata = np.full((3, 2), -999.0, dtype=np.float64)
    expected = _expected_loaded_values(data, r_nums, e_nums, 2, 3)

    def rust_load(data_arg, nfields, npoints, r_nums_arg, e_nums_arg, sdata_arg):
        calls.append(
            (
                data_arg.dtype,
                nfields,
                npoints,
                r_nums_arg.dtype,
                e_nums_arg.dtype,
                sdata_arg.dtype,
            )
        )
        sdata_arg[:] = expected
        return "ignored"

    monkeypatch.setattr(_load_nn_field_data, "_rust_kernel", lambda: rust_load)

    result = _load_nn_field_data._load_nn_field_data(
        data, 2, 3, r_nums, e_nums, sdata
    )

    assert result is None
    assert calls == [(np.dtype("O"), 2, 3, np.dtype(np.intc), np.dtype(np.intc), np.float64)]
    np.testing.assert_array_equal(sdata, expected)


def test_load_nn_field_data_preserves_non_float64_sdata_dtype_on_fallback(monkeypatch):
    def rust_load(*_args):
        raise AssertionError("non-float64 sdata should use the Python fallback")

    monkeypatch.setattr(_load_nn_field_data, "_rust_kernel", lambda: rust_load)
    data = _field_data_object_array()
    r_nums = np.array([0, 1, 2], dtype=np.intc)
    e_nums = np.array([1, 2, 0], dtype=np.intc)
    sdata = np.full((3, 2), -999.0, dtype=np.float32)

    _load_nn_field_data._load_nn_field_data(data, 2, 3, r_nums, e_nums, sdata)

    assert sdata.dtype == np.float32
    np.testing.assert_array_equal(
        sdata, _expected_loaded_values(data, r_nums, e_nums, 2, 3, dtype=np.float32)
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_load_nn_field_data_matches_python_fallback(monkeypatch):
    data = _field_data_object_array()
    r_nums = np.array([0, 1, 2, 0], dtype=np.intc)
    e_nums = np.array([2, 1, 0, 1], dtype=np.intc)
    rust_sdata = np.full((4, 2), -999.0, dtype=np.float64)
    python_sdata = np.full((4, 2), -999.0, dtype=np.float64)

    _load_nn_field_data._load_nn_field_data(data, 2, 4, r_nums, e_nums, rust_sdata)
    monkeypatch.setattr(_load_nn_field_data, "_rust_kernel", lambda: None)
    _load_nn_field_data._load_nn_field_data(
        data, 2, 4, r_nums, e_nums, python_sdata
    )

    np.testing.assert_array_equal(rust_sdata, python_sdata)
