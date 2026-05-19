import os

import numpy as np
import pytest

from pyart.aux_io import edge_netcdf
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_edge_mask_i16"):
        pytest.skip("pyart._rust has no EDGE mask kernel")
    return rust


def _masked_copy(field_data):
    return np.ma.array(field_data, copy=True)


def _fallback_mask(field_data, missing, folded, monkeypatch, copy=True):
    monkeypatch.setattr(edge_netcdf, "_rust_kernel", lambda _name: None)
    return edge_netcdf._mask_edge_field_data(np.ma.array(field_data, copy=copy), missing, folded)


def test_edge_mask_python_fallback_reference_case(monkeypatch):
    field_data = np.ma.array(
        np.array([[1, -999, -888], [5, 6, -999]], dtype=np.int16),
        mask=[[False, False, False], [True, False, False]],
    )
    before_payload = field_data.data.copy()

    actual = _fallback_mask(field_data, -999, -888, monkeypatch)

    assert actual.dtype == field_data.dtype
    assert_exact_equal(actual.data, before_payload)
    assert np.ma.getmaskarray(actual).tolist() == [
        [False, True, True],
        [True, False, True],
    ]


def test_edge_mask_dispatches_dense_i16_to_private_rust(monkeypatch):
    field_data = np.ma.array(
        np.array([[1, -999], [-888, 5]], dtype=np.int16),
        mask=[[False, False], [True, False]],
    )
    calls = []
    rust_mask = np.array([[False, True], [True, False]], dtype=bool)

    def kernel(data_arg, existing_mask_arg, has_missing, missing, has_folded, folded):
        calls.append(
            (
                data_arg.dtype,
                data_arg.shape,
                existing_mask_arg.dtype,
                existing_mask_arg.tolist(),
                has_missing,
                missing,
                has_folded,
                folded,
            )
        )
        return rust_mask.copy()

    monkeypatch.setattr(
        edge_netcdf,
        "_rust_kernel",
        lambda name: kernel if name == "_edge_mask_i16" else None,
    )

    actual = edge_netcdf._mask_edge_field_data(_masked_copy(field_data), -999, -888)

    assert calls == [
        (
            np.dtype(np.int16),
            (2, 2),
            np.dtype(bool),
            [[False, False], [True, False]],
            True,
            -999,
            True,
            -888,
        )
    ]
    assert_exact_equal(actual.data, field_data.data)
    assert np.array_equal(np.ma.getmaskarray(actual), rust_mask)
    assert actual.fill_value == field_data.fill_value


def test_edge_mask_rust_runtime_error_keeps_python_path(monkeypatch):
    field_data = np.ma.array(np.array([[1, -999]], dtype=np.int16))

    def rust_kernel(name):
        if name != "_edge_mask_i16":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(edge_netcdf, "_rust_kernel", rust_kernel)
    actual = edge_netcdf._mask_edge_field_data(_masked_copy(field_data), -999, None)
    expected = _fallback_mask(field_data, -999, None, monkeypatch)

    assert_exact_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        lambda: (np.ma.array(np.arange(12, dtype=np.int16).reshape(3, 4)[:, ::2]), 2, None),
        lambda: (np.ma.array(np.array([1, 2], dtype=">i2")), 1, None),
        lambda: (np.ma.array(np.array([1, 2], dtype=np.int16)), np.array([1, 2]), None),
        lambda: (np.ma.array(np.array([1, 2], dtype=np.int16)), "1", None),
        lambda: (np.ma.array(np.array(["a", "b"], dtype=object)), "a", None),
        lambda: (np.ma.array(np.array([1 + 0j, 2 + 0j], dtype=np.complex64)), 1 + 0j, None),
    ],
)
def test_edge_mask_unsupported_inputs_keep_python_path(monkeypatch, case):
    field_data, missing, folded = case()

    def rust_kernel(name):
        if name.startswith("_edge_mask_"):
            raise AssertionError("unsupported EDGE input used Rust kernel")
        return None

    monkeypatch.setattr(edge_netcdf, "_rust_kernel", rust_kernel)
    try:
        actual = edge_netcdf._mask_edge_field_data(np.ma.array(field_data, copy=False), missing, folded)
    except Exception as actual_error:
        expected_field_data, _, _ = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_mask(expected_field_data, missing, folded, monkeypatch, copy=False)
        assert actual_error.args == expected_error.value.args
    else:
        expected_field_data, _, _ = case()
        expected = _fallback_mask(expected_field_data, missing, folded, monkeypatch, copy=False)
        assert_exact_equal(actual, expected)


def test_edge_mask_no_match_attr_materializes_dense_mask(monkeypatch):
    field_data = np.ma.array(np.array([1, 2], dtype=np.int16))

    actual = _fallback_mask(field_data, -1, None, monkeypatch)

    assert isinstance(actual.mask, np.ndarray)
    assert actual.mask.dtype == bool
    assert actual.mask.tolist() == [False, False]


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust EDGE mask parity",
)
@pytest.mark.parametrize(
    "field_data,missing,folded",
    [
        (
            np.ma.array(
                np.array([[1, -999], [-888, 5]], dtype=np.int16),
                mask=[[False, False], [True, False]],
            ),
            -999,
            -888,
        ),
        (np.ma.array(np.array([[0, 255, 254]], dtype=np.uint8)), 255, 254),
        (np.ma.array(np.array([[np.nan, np.inf, -np.inf, 0.0]], dtype=np.float32)), np.nan, np.inf),
        (np.ma.array(np.array([[1, 2]], dtype=np.int16)), -1, None),
    ],
)
def test_edge_mask_real_rust_matches_python_fallback(monkeypatch, field_data, missing, folded):
    expected = _fallback_mask(field_data, missing, folded, monkeypatch)
    monkeypatch.undo()

    actual = edge_netcdf._mask_edge_field_data(_masked_copy(field_data), missing, folded)

    assert_exact_equal(actual, expected)
    if missing == -1:
        assert isinstance(actual.mask, np.ndarray)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust EDGE checks",
)
def test_edge_mask_direct_rust_helper():
    rust = _rust_or_skip()
    data = np.array([[1, -999], [-888, 5]], dtype=np.int16)
    existing = np.array([[False, False], [True, False]], dtype=bool)

    mask = rust._edge_mask_i16(data, existing, True, -999, True, -888)

    assert mask.dtype == bool
    assert mask.tolist() == [[False, True], [True, False]]
    with pytest.raises(ValueError):
        rust._edge_mask_i16(data, np.zeros((4,), dtype=bool), True, -999, False, 0)
