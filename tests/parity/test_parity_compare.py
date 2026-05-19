import numpy as np
import pytest

from tools.parity_compare import ParityMismatch, assert_exact_equal


def test_exact_array_includes_dtype_and_nan_positions():
    expected = np.array([1.0, np.nan], dtype=np.float64)
    actual = np.array([1.0, np.nan], dtype=np.float64)

    assert_exact_equal(actual, expected)

    with pytest.raises(ParityMismatch):
        assert_exact_equal(actual.astype(np.float32), expected)


def test_masked_array_includes_mask_and_fill_value():
    expected = np.ma.array([1, 2], mask=[False, True], fill_value=-9999)
    actual = np.ma.array([1, 2], mask=[False, True], fill_value=-9999)

    assert_exact_equal(actual, expected)

    drift = np.ma.array([1, 2], mask=[True, False], fill_value=-9999)
    with pytest.raises(ParityMismatch):
        assert_exact_equal(drift, expected)


def test_nested_metadata_compares_exactly():
    expected = {"field": {"data": np.array([1, 2], dtype=np.int16)}}
    actual = {"field": {"data": np.array([1, 2], dtype=np.int16)}}

    assert_exact_equal(actual, expected)

    actual["field"]["data"][1] = 3
    with pytest.raises(ParityMismatch):
        assert_exact_equal(actual, expected)
