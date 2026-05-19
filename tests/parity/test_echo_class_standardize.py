import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import echo_class  # noqa: E402
from tools.parity_compare import assert_exact_equal  # noqa: E402


def _fallback_standardize(data, field_name, monkeypatch, **kwargs):
    monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: None)
    return echo_class._standardize(data, field_name, **kwargs)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("Zh", {}),
        ("ZDR", {}),
        ("custom", {"mx": 2.0, "mn": -2.0}),
    ],
)
def test_standardize_linear_python_fallback_reference_cases(
    monkeypatch, field_name, kwargs
):
    data = np.array([[-20.0, -10.0, 0.0], [2.0, 60.0, 70.0]], dtype=np.float64)

    actual = _fallback_standardize(data.copy(), field_name, monkeypatch, **kwargs)

    assert type(actual) is np.ndarray
    assert actual.dtype == np.float64
    assert actual.shape == data.shape


def test_standardize_linear_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(data, mx, mn):
        calls.append((data.dtype, data.shape, mx, mn))
        return np.full(data.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_echo_class_standardize_linear" else None,
    )
    data = np.array([-20.0, -10.0, 0.0, 60.0, 70.0], dtype=np.float64)

    actual = echo_class._standardize(data, "Zh")

    assert calls == [(np.float64, (5,), 60.0, -10.0)]
    np.testing.assert_array_equal(actual, np.full((5,), 7.0, dtype=np.float64))
    np.testing.assert_array_equal(
        data, np.array([-20.0, -10.0, 0.0, 60.0, 70.0], dtype=np.float64)
    )


@pytest.mark.parametrize(
    ("field_name", "data", "kwargs"),
    [
        ("Zh", np.array([-10.0, 0.0], dtype=np.float32), {}),
        ("Zh", np.array([-10.0, np.nan], dtype=np.float64), {}),
        ("Zh", np.array([-10.0, np.inf], dtype=np.float64), {}),
        ("Zh", np.array([-10.0, 1.0e100], dtype=np.float64), {}),
        ("Zh", np.ma.array([-10.0, 0.0], mask=[False, True]), {}),
        ("KDP", np.array([-10.0, 0.0], dtype=np.float64), {}),
        ("RhoHV", np.array([0.9, 1.1], dtype=np.float64), {}),
        ("relH", np.array([-10.0, 0.0], dtype=np.float64), {}),
        ("bad", np.array([-10.0, 0.0], dtype=np.float64), {}),
        ("Zh", [-10.0, 0.0], {}),
        ("Zh", np.array([-10.0, 0.0], dtype=np.float64), {"mx": 1.0, "mn": 1.0}),
    ],
)
def test_standardize_linear_keeps_python_path_for_unsupported_inputs(
    monkeypatch, field_name, data, kwargs
):
    def fail_if_called(name):
        if name != "_echo_class_standardize_linear":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported _standardize input used Rust")

        return kernel

    monkeypatch.setattr(echo_class, "_rust_kernel", fail_if_called)
    data_actual = data.copy() if hasattr(data, "copy") else list(data)
    try:
        actual = echo_class._standardize(data_actual, field_name, **kwargs)
    except Exception as actual_error:
        data_expected = data.copy() if hasattr(data, "copy") else list(data)
        monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            echo_class._standardize(data_expected, field_name, **kwargs)
        assert actual_error.args == expected_error.value.args
    else:
        data_expected = data.copy() if hasattr(data, "copy") else list(data)
        expected = _fallback_standardize(data_expected, field_name, monkeypatch, **kwargs)
        assert_exact_equal(actual, expected)
        assert_exact_equal(data_actual, data_expected)


def test_standardize_linear_keeps_python_path_for_noncontiguous_input(monkeypatch):
    def fail_if_called(name):
        if name != "_echo_class_standardize_linear":
            return None

        def kernel(*_args):
            raise AssertionError("noncontiguous _standardize input used Rust")

        return kernel

    data = np.array([-10.0, 0.0, 60.0], dtype=np.float64)[::2]
    monkeypatch.setattr(echo_class, "_rust_kernel", fail_if_called)
    actual = echo_class._standardize(data, "Zh")

    expected = _fallback_standardize(data, "Zh", monkeypatch)
    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("Zh", {}),
        ("ZDR", {}),
        ("custom", {"mx": 2.0, "mn": -2.0}),
    ],
)
def test_real_rust_standardize_linear_matches_python_fallback(
    monkeypatch, field_name, kwargs
):
    import pyart._rust as rust

    data = np.array([[-20.0, -10.0, 0.0], [2.0, 60.0, 70.0]], dtype=np.float64)
    expected = _fallback_standardize(data.copy(), field_name, monkeypatch, **kwargs)
    calls = []

    def rust_kernel(name):
        if name == "_echo_class_standardize_linear":
            calls.append(name)
            return rust._echo_class_standardize_linear
        return None

    monkeypatch.setattr(echo_class, "_rust_kernel", rust_kernel)
    actual_input = data.copy()
    actual = echo_class._standardize(actual_input, field_name, **kwargs)

    assert calls == ["_echo_class_standardize_linear"]
    assert_exact_equal(actual, expected)
    np.testing.assert_array_equal(actual_input, data)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("data", "mx", "mn", "match"),
        [
            (np.array([0.0, np.nan], dtype=np.float64), 1.0, 0.0, "finite"),
            (np.array([0.0, 1.0e100], dtype=np.float64), 1.0, 0.0, "supported range"),
            (np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)[::2], 1.0, 0.0, "C-contiguous"),
            (np.array([0.0, 1.0], dtype=np.float64), 1.0, 1.0, "finite and distinct"),
        ],
    )
def test_real_rust_standardize_linear_rejects_direct_unsafe_inputs(
    data, mx, mn, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._echo_class_standardize_linear(data, mx, mn)
