"""Exact parity comparison helpers for Py-ART oracle tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParityMismatch(AssertionError):
    """Raised when two values differ under the exact parity contract."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def assert_exact_equal(actual: Any, expected: Any, path: str = "value") -> None:
    """Assert exact equality for arrays, masked arrays, scalars, and metadata."""
    if np.ma.isMaskedArray(actual) or np.ma.isMaskedArray(expected):
        _assert_masked_array(actual, expected, path)
        return

    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        _assert_array(actual, expected, path)
        return

    if isinstance(actual, dict) or isinstance(expected, dict):
        _assert_dict(actual, expected, path)
        return

    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        _assert_sequence(actual, expected, path)
        return

    if isinstance(actual, float) or isinstance(expected, float):
        _assert_float(actual, expected, path)
        return

    if actual != expected:
        raise ParityMismatch(path, f"{actual!r} != {expected!r}")


def _assert_array(actual: Any, expected: Any, path: str) -> None:
    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
        raise ParityMismatch(path, f"type drift: {type(actual)!r} != {type(expected)!r}")
    if actual.shape != expected.shape:
        raise ParityMismatch(path, f"shape drift: {actual.shape!r} != {expected.shape!r}")
    if actual.dtype != expected.dtype:
        raise ParityMismatch(path, f"dtype drift: {actual.dtype!r} != {expected.dtype!r}")

    actual_nan = np.isnan(actual) if np.issubdtype(actual.dtype, np.floating) else np.zeros(actual.shape, dtype=bool)
    expected_nan = np.isnan(expected) if np.issubdtype(expected.dtype, np.floating) else np.zeros(expected.shape, dtype=bool)
    if not np.array_equal(actual_nan, expected_nan):
        raise ParityMismatch(path, "NaN position drift")

    comparable_actual = actual[~actual_nan] if actual_nan.any() else actual
    comparable_expected = expected[~expected_nan] if expected_nan.any() else expected
    if not np.array_equal(comparable_actual, comparable_expected):
        raise ParityMismatch(path, "array value drift")


def _assert_masked_array(actual: Any, expected: Any, path: str) -> None:
    if not np.ma.isMaskedArray(actual) or not np.ma.isMaskedArray(expected):
        raise ParityMismatch(path, f"masked-array type drift: {type(actual)!r} != {type(expected)!r}")
    if actual.fill_value != expected.fill_value:
        raise ParityMismatch(path, f"fill value drift: {actual.fill_value!r} != {expected.fill_value!r}")
    if np.ma.is_masked(actual) != np.ma.is_masked(expected):
        raise ParityMismatch(path, "masked state drift")
    if not np.array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)):
        raise ParityMismatch(path, "mask drift")
    _assert_array(np.asarray(actual.data), np.asarray(expected.data), f"{path}.data")


def _assert_dict(actual: Any, expected: Any, path: str) -> None:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ParityMismatch(path, f"dict type drift: {type(actual)!r} != {type(expected)!r}")
    if set(actual) != set(expected):
        raise ParityMismatch(path, f"key drift: {sorted(actual)!r} != {sorted(expected)!r}")
    for key in sorted(actual, key=str):
        assert_exact_equal(actual[key], expected[key], f"{path}[{key!r}]")


def _assert_sequence(actual: Any, expected: Any, path: str) -> None:
    if type(actual) is not type(expected):
        raise ParityMismatch(path, f"sequence type drift: {type(actual)!r} != {type(expected)!r}")
    if len(actual) != len(expected):
        raise ParityMismatch(path, f"length drift: {len(actual)} != {len(expected)}")
    for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        assert_exact_equal(actual_item, expected_item, f"{path}[{idx}]")


def _assert_float(actual: Any, expected: Any, path: str) -> None:
    if math.isnan(float(actual)) or math.isnan(float(expected)):
        if not (math.isnan(float(actual)) and math.isnan(float(expected))):
            raise ParityMismatch(path, f"NaN drift: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise ParityMismatch(path, f"float drift: {actual!r} != {expected!r}")
