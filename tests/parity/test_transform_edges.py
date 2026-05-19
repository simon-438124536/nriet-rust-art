import numpy as np
import pytest

from pyart.core import transforms


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    return rust


def _fallback_range_edges(values, monkeypatch):
    monkeypatch.setattr(transforms, "_rust_kernel", lambda _name: None)
    return transforms._interpolate_range_edges(values)


def _fallback_axes_edges(values, monkeypatch):
    monkeypatch.setattr(transforms, "_rust_kernel", lambda _name: None)
    return transforms._interpolate_axes_edges(values)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, 1.0, 5.0], dtype=np.float32),
        np.array([-1.0, 1.0, 5.0], dtype=np.float32),
        np.array([np.nan, 2.0, 4.0], dtype=np.float32),
    ],
)
def test_interpolate_edges_python_fallback_reference_cases(monkeypatch, values):
    range_edges = _fallback_range_edges(values, monkeypatch)
    axes_edges = _fallback_axes_edges(values, monkeypatch)

    assert range_edges.dtype == np.float32
    assert axes_edges.dtype == np.float32
    assert range_edges.shape == (values.shape[0] + 1,)
    assert axes_edges.shape == (values.shape[0] + 1,)


@pytest.mark.parametrize(
    ("function_name", "function"),
    [
        ("_interpolate_range_edges_f32", transforms._interpolate_range_edges),
        ("_interpolate_axes_edges_f32", transforms._interpolate_axes_edges),
    ],
)
def test_interpolate_edges_dispatches_float32_to_private_rust_kernel(
    monkeypatch, function_name, function
):
    calls = []

    def kernel(values):
        calls.append((values.dtype, values.shape))
        return np.array([9.0, 8.0, 7.0], dtype=np.float32)

    monkeypatch.setattr(
        transforms,
        "_rust_kernel",
        lambda name: kernel if name == function_name else None,
    )

    actual = function(np.array([1.0, 3.0], dtype=np.float32))

    assert calls == [(np.dtype(np.float32), (2,))]
    np.testing.assert_array_equal(actual, np.array([9.0, 8.0, 7.0], dtype=np.float32))


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, 1.0, 5.0], dtype=np.float64),
        np.array([0, 1, 5], dtype=np.int32),
        np.array([1.0], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.arange(6, dtype=np.float32)[::2],
    ],
)
@pytest.mark.parametrize(
    ("function_name", "function", "fallback"),
    [
        ("_interpolate_range_edges_f32", transforms._interpolate_range_edges, _fallback_range_edges),
        ("_interpolate_axes_edges_f32", transforms._interpolate_axes_edges, _fallback_axes_edges),
    ],
)
def test_interpolate_edges_unsupported_inputs_keep_python_fallback(
    monkeypatch, values, function_name, function, fallback
):
    def rust_kernel(name):
        if name != function_name:
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported edge input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(transforms, "_rust_kernel", rust_kernel)
    try:
        actual = function(values)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            fallback(values, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = fallback(values, monkeypatch)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("function_name", "function", "fallback"),
    [
        ("_interpolate_range_edges_f32", transforms._interpolate_range_edges, _fallback_range_edges),
        ("_interpolate_axes_edges_f32", transforms._interpolate_axes_edges, _fallback_axes_edges),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, 1.0, 5.0], dtype=np.float32),
        np.array([-1.0, 1.0, 5.0], dtype=np.float32),
        np.array([np.nan, 2.0, 4.0], dtype=np.float32),
    ],
)
def test_real_rust_interpolate_edges_match_python_fallback(
    monkeypatch, function_name, function, fallback, values
):
    rust = _rust_or_skip()

    expected = fallback(values, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == function_name:
            calls.append(name)
            return getattr(rust, name)
        return None

    monkeypatch.setattr(transforms, "_rust_kernel", rust_kernel)
    actual = function(values)

    assert calls == [function_name]
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("function_name", "values", "match"),
    [
        ("_interpolate_range_edges_f32", np.array([1.0], dtype=np.float32), "at least two"),
        ("_interpolate_axes_edges_f32", np.array([1.0], dtype=np.float32), "at least two"),
        (
            "_interpolate_range_edges_f32",
            np.arange(6, dtype=np.float32)[::2],
            "C-contiguous",
        ),
        (
            "_interpolate_axes_edges_f32",
            np.arange(6, dtype=np.float32)[::2],
            "C-contiguous",
        ),
    ],
)
def test_real_rust_interpolate_edges_direct_rejects_unsafe_inputs(
    function_name, values, match
):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        getattr(rust, function_name)(values)
