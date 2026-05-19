import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import vad  # noqa: E402


def _fallback_inverse_dist_squared(dist, monkeypatch):
    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vad._inverse_dist_squared(dist)
    return result, [(item.category, str(item.message)) for item in caught]


def _call_inverse_dist_squared(dist):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vad._inverse_dist_squared(dist)
    return result, [(item.category, str(item.message)) for item in caught]


def test_inverse_dist_squared_python_fallback_reference(monkeypatch):
    dist = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)

    actual, caught = _fallback_inverse_dist_squared(dist, monkeypatch)

    np.testing.assert_array_equal(actual, np.array([0.25, 4.0, 4.0, 0.25]))
    assert actual.dtype == np.float64
    assert caught == []


def test_inverse_dist_squared_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(dist):
        calls.append((dist.dtype, dist.shape))
        return np.full(dist.shape, 3.0, dtype=np.float64)

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_vad_inverse_dist_squared" else None,
    )
    dist = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)

    actual, caught = _call_inverse_dist_squared(dist)

    assert calls == [(np.float64, (4,))]
    np.testing.assert_array_equal(actual, np.full((4,), 3.0, dtype=np.float64))
    assert caught == []


@pytest.mark.parametrize(
    "dist",
    [
        np.array([-2.0, 0.0, 2.0], dtype=np.float64),
        np.array([-2.0, np.nan, 2.0], dtype=np.float64),
        np.array([-2.0, np.inf, 2.0], dtype=np.float64),
        np.array([-2.0, 1.0e150, 2.0], dtype=np.float64),
        np.array([-2.0, 1.0e-150, 2.0], dtype=np.float64),
        np.array([-2.0, 0.5, 2.0], dtype=np.float32),
        np.array([[-2.0, 0.5], [2.0, 4.0]], dtype=np.float64),
        np.array([-2.0, 0.5, 2.0, 4.0], dtype=np.float64)[::2],
        np.ma.array([-2.0, 0.5, 2.0], mask=[False, True, False]),
        [-2.0, 0.5, 2.0],
    ],
)
def test_inverse_dist_squared_keeps_python_path_for_unsupported_inputs(
    monkeypatch, dist
):
    def fail_if_called(name):
        if name != "_vad_inverse_dist_squared":
            return None

        def kernel(_dist):
            raise AssertionError("unsupported _inverse_dist_squared input used Rust")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)
    try:
        actual, actual_warnings = _call_inverse_dist_squared(dist)
    except Exception as actual_error:
        monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            vad._inverse_dist_squared(dist)
        assert actual_error.args == expected_error.value.args
    else:
        expected, expected_warnings = _fallback_inverse_dist_squared(dist, monkeypatch)
        np.testing.assert_array_equal(actual, expected)
        assert actual_warnings == expected_warnings


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_inverse_dist_squared_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    dist = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    expected, expected_warnings = _fallback_inverse_dist_squared(dist, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_vad_inverse_dist_squared":
            calls.append(name)
            return rust._vad_inverse_dist_squared
        return None

    monkeypatch.setattr(vad, "_rust_kernel", rust_kernel)
    actual, actual_warnings = _call_inverse_dist_squared(dist)

    assert calls == ["_vad_inverse_dist_squared"]
    np.testing.assert_array_equal(actual, expected)
    assert actual_warnings == expected_warnings


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("dist", "match"),
    [
        (np.array([-2.0, 0.0, 2.0], dtype=np.float64), "finite, non-zero"),
        (np.array([-2.0, np.nan, 2.0], dtype=np.float64), "finite, non-zero"),
        (np.array([-2.0, 1.0e150, 2.0], dtype=np.float64), "supported range"),
        (np.array([-2.0, 0.5, 2.0, 4.0], dtype=np.float64)[::2], "C-contiguous"),
    ],
)
def test_real_rust_inverse_dist_squared_rejects_direct_unsafe_inputs(dist, match):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._vad_inverse_dist_squared(dist)
