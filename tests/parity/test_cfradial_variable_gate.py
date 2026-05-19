import os

import numpy as np
import pytest

from pyart.io import cfradial


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_cfradial_unpack_variable_gate_dense"):
        pytest.skip("pyart._rust has no CF/Radial variable-gate kernel")
    return rust


def _fallback_unpack(fdata, shape, ray_n_gates, ray_start_index, monkeypatch):
    monkeypatch.setattr(cfradial, "_rust_kernel", lambda _name: None)
    dic = {"data": fdata, "units": "example"}
    result = cfradial._unpack_variable_gate_field_dic(
        dic, shape, ray_n_gates, ray_start_index
    )
    assert result is None
    assert dic["units"] == "example"
    return dic["data"]


def _assert_masked_surface_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    actual_mask = np.ma.getmaskarray(actual)
    expected_mask = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_array_equal(actual.data[~actual_mask], expected.data[~expected_mask])


def test_cfradial_variable_gate_python_fallback_dense_reference(monkeypatch):
    fdata = np.array([10, 11, 12, 13, 14], dtype=np.int16)

    actual = _fallback_unpack(
        fdata,
        (2, 3),
        np.array([2, 3], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        monkeypatch,
    )

    assert actual.dtype == np.int16
    assert actual.fill_value == np.ma.masked_all((1,), dtype=np.int16).fill_value
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[False, False, True], [False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(actual.data[0, :2], np.array([10, 11], dtype=np.int16))
    np.testing.assert_array_equal(actual.data[1, :3], np.array([12, 13, 14], dtype=np.int16))


def test_cfradial_variable_gate_python_fallback_preserves_masked_source(monkeypatch):
    fdata = np.ma.array(
        [10.0, 20.0, 30.0, 40.0],
        mask=[False, True, False, False],
        dtype=np.float32,
    )

    actual = _fallback_unpack(
        fdata,
        (2, 3),
        np.array([2, 2], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        monkeypatch,
    )

    assert actual.dtype == np.float32
    assert actual.fill_value == np.ma.masked_all((1,), dtype=np.float32).fill_value
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([[False, True, True], [False, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(actual.data[[0, 1], [0, 0]], np.array([10.0, 30.0], dtype=np.float32))


def test_cfradial_variable_gate_dispatches_dense_inputs_to_private_rust(monkeypatch):
    calls = []
    fdata = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)

    def kernel(fdata_arg, out_data, out_mask, gates_arg, starts_arg):
        calls.append(
            (
                fdata_arg.dtype,
                fdata_arg.shape,
                out_data.dtype,
                out_data.shape,
                out_mask.dtype,
                out_mask.shape,
                gates_arg.dtype,
                starts_arg.dtype,
                gates_arg.copy(),
                starts_arg.copy(),
                out_mask.copy(),
            )
        )
        out_data[0, :2] = fdata_arg[0:2]
        out_data[1, :2] = fdata_arg[2:4]
        out_mask[0, :2] = False
        out_mask[1, :2] = False

    monkeypatch.setattr(
        cfradial,
        "_rust_kernel",
        lambda name: kernel if name == "_cfradial_unpack_variable_gate_dense" else None,
    )
    dic = {"data": fdata, "long_name": "test field"}

    result = cfradial._unpack_variable_gate_field_dic(
        dic,
        (2, 3),
        np.array([2, 2], dtype=np.int32),
        np.array([0, 2], dtype=np.uint32),
    )

    assert result is None
    assert dic["long_name"] == "test field"
    assert len(calls) == 1
    call = calls[0]
    assert call[0:8] == (
        np.dtype(np.float32),
        (4,),
        np.dtype(np.float32),
        (2, 3),
        np.dtype(bool),
        (2, 3),
        np.dtype(np.int64),
        np.dtype(np.int64),
    )
    np.testing.assert_array_equal(call[8], np.array([2, 2], dtype=np.int64))
    np.testing.assert_array_equal(call[9], np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(call[10], np.ones((2, 3), dtype=bool))
    np.testing.assert_array_equal(
        np.ma.getmaskarray(dic["data"]),
        np.array([[False, False, True], [False, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(dic["data"].data[:, :2], fdata.reshape(2, 2))


@pytest.mark.parametrize(
    ("fdata", "shape", "ray_n_gates", "ray_start_index"),
    [
        (
            np.ma.array([1.0, 2.0], mask=[False, True], dtype=np.float64),
            (1, 2),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        (
            np.arange(6, dtype=np.float32)[::2],
            (1, 2),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        (
            np.array([{"a": 1}], dtype=object),
            (1, 1),
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        (
            np.array([[1.0, 2.0]], dtype=np.float64),
            (1, 2),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        (
            np.arange(4, dtype=np.int16),
            (2, 2),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        (
            np.arange(4, dtype=np.int16),
            (1, 2),
            np.array([-1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        (
            np.arange(4, dtype=np.int16),
            (1, 2),
            np.array([1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
        ),
        (
            np.arange(4, dtype=np.int16),
            (1, 2),
            np.array([3], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ),
    ],
)
def test_cfradial_variable_gate_unsupported_inputs_keep_python_path(
    monkeypatch, fdata, shape, ray_n_gates, ray_start_index
):
    def rust_kernel(name):
        if name != "_cfradial_unpack_variable_gate_dense":
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    original = fdata
    monkeypatch.setattr(cfradial, "_rust_kernel", rust_kernel)
    dic = {"data": fdata}
    try:
        result = cfradial._unpack_variable_gate_field_dic(
            dic, shape, ray_n_gates, ray_start_index
        )
    except Exception as actual_error:
        assert dic["data"] is original
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_unpack(
                fdata, shape, ray_n_gates, ray_start_index, monkeypatch
            )
        assert actual_error.args == expected_error.value.args
    else:
        assert result is None
        expected = _fallback_unpack(
            fdata, shape, ray_n_gates, ray_start_index, monkeypatch
        )
        _assert_masked_surface_equal(dic["data"], expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust CF/Radial parity",
)
@pytest.mark.parametrize(
    ("fdata", "shape", "ray_n_gates", "ray_start_index"),
    [
        (
            np.array([10, 11, 12, 13, 14], dtype=np.int16),
            (2, 3),
            np.array([2, 3], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        ),
        (
            np.array([1.5, 2.5], dtype=np.float32),
            (2, 2),
            np.array([0, 2], dtype=np.int64),
            np.array([2, 0], dtype=np.int64),
        ),
    ],
)
def test_real_rust_cfradial_variable_gate_matches_python_fallback(
    monkeypatch, fdata, shape, ray_n_gates, ray_start_index
):
    rust = _rust_or_skip()
    expected = _fallback_unpack(
        fdata.copy(), shape, ray_n_gates, ray_start_index, monkeypatch
    )
    monkeypatch.setattr(
        cfradial,
        "_rust_kernel",
        lambda name: rust._cfradial_unpack_variable_gate_dense
        if name == "_cfradial_unpack_variable_gate_dense"
        else None,
    )
    dic = {"data": fdata.copy()}

    result = cfradial._unpack_variable_gate_field_dic(
        dic, shape, ray_n_gates, ray_start_index
    )

    assert result is None
    _assert_masked_surface_equal(dic["data"], expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust CF/Radial checks",
)
def test_real_rust_cfradial_variable_gate_direct_mutates_output_buffers():
    rust = _rust_or_skip()
    fdata = np.array([10, 11, 12, 13, 14], dtype=np.int16)
    out_data = np.full((2, 3), -1, dtype=np.int16)
    out_mask = np.ones((2, 3), dtype=bool)

    rust._cfradial_unpack_variable_gate_dense(
        fdata,
        out_data,
        out_mask,
        np.array([2, 3], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        out_data, np.array([[10, 11, -1], [12, 13, 14]], dtype=np.int16)
    )
    np.testing.assert_array_equal(
        out_mask, np.array([[False, False, True], [False, False, False]])
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust CF/Radial checks",
)
@pytest.mark.parametrize(
    ("fdata", "out_data", "ray_n_gates", "ray_start_index", "match"),
    [
        (
            np.arange(6, dtype=np.float32)[::2],
            np.full((1, 2), -1, dtype=np.float32),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
            "C-contiguous",
        ),
        (
            np.arange(4, dtype=np.int16),
            np.full((1, 2), -1, dtype=np.float32),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
            "identical dtype",
        ),
        (
            np.array([{"a": 1}], dtype=object),
            np.empty((1, 1), dtype=object),
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            "numeric",
        ),
        (
            np.arange(4, dtype=np.int16),
            np.full((1, 2), -1, dtype=np.int16),
            np.array([-1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            "nonnegative",
        ),
        (
            np.arange(4, dtype=np.int16),
            np.full((1, 4), -1, dtype=np.int16),
            np.array([3], dtype=np.int64),
            np.array([2], dtype=np.int64),
            "exceeds fdata length",
        ),
    ],
)
def test_real_rust_cfradial_variable_gate_direct_rejects_unsafe_inputs(
    fdata, out_data, ray_n_gates, ray_start_index, match
):
    rust = _rust_or_skip()
    out_mask = np.ones(out_data.shape, dtype=bool)
    original_data = out_data.copy()
    original_mask = out_mask.copy()

    with pytest.raises(ValueError, match=match):
        rust._cfradial_unpack_variable_gate_dense(
            fdata, out_data, out_mask, ray_n_gates, ray_start_index
        )

    np.testing.assert_array_equal(out_data, original_data)
    np.testing.assert_array_equal(out_mask, original_mask)
