import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import _common_dealias  # noqa: E402


def _fallback_set_limits(data, nyquist_vel, monkeypatch):
    monkeypatch.setattr(_common_dealias, "_rust_kernel", lambda _name: None)
    out = {"existing": 1}
    result = _common_dealias._set_limits(data, nyquist_vel, out)
    return result, out


def _set_limits(data, nyquist_vel):
    out = {"existing": 1}
    result = _common_dealias._set_limits(data, nyquist_vel, out)
    return result, out


@pytest.mark.parametrize(
    ("data", "nyquist_vel", "expected"),
    [
        (
            np.array([-5.0, 2.0, 7.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
            {"existing": 1, "valid_min": -10.0, "valid_max": 10.0},
        ),
        (
            np.array([-5.0, 2.0, 20.0], dtype=np.float64),
            np.array([8.0], dtype=np.float64),
            {"existing": 1, "valid_min": -24.0, "valid_max": 24.0},
        ),
        (
            np.array([[0.0, -0.0]], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            {"existing": 1, "valid_min": np.nan, "valid_max": np.nan},
        ),
    ],
)
def test_set_limits_python_fallback_dense_cases(monkeypatch, data, nyquist_vel, expected):
    result, actual = _fallback_set_limits(data, nyquist_vel, monkeypatch)

    assert result is None
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        if isinstance(expected_value, float) and np.isnan(expected_value):
            assert np.isnan(actual[key])
        else:
            assert actual[key] == expected_value


def test_set_limits_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(data, nyquist_vel):
        calls.append((data.dtype, data.shape, nyquist_vel.dtype, nyquist_vel.shape))
        return -3.0, 3.0

    monkeypatch.setattr(
        _common_dealias,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_common_dealias_limits_dense" else None,
    )

    result, actual = _set_limits(
        np.array([-5.0, 2.0, 7.0], dtype=np.float64),
        np.array([10.0], dtype=np.float64),
    )

    assert result is None
    assert calls == [(np.float64, (3,), np.float64, (1,))]
    assert actual == {"existing": 1, "valid_min": -3.0, "valid_max": 3.0}


@pytest.mark.parametrize(
    ("data", "nyquist_vel"),
    [
        (np.ma.array([1.0, 2.0], mask=[True, True]), np.array([8.0])),
        (np.ma.array([1.0, 20.0], mask=[False, True]), np.array([8.0])),
        (np.array([1.0, np.nan]), np.array([8.0])),
        (np.array([1.0, np.inf]), np.array([8.0])),
        (np.array([1.0, 2.0]), np.ma.array([8.0, 9.0], mask=[True, False])),
        (np.array([1.0, 2.0]), np.array([np.nan])),
        (np.array([], dtype=np.float64), np.array([8.0], dtype=np.float64)),
        (np.array([1, 2], dtype=np.int32), np.array([8.0], dtype=np.float64)),
        ([1.0, 2.0], np.array([8.0], dtype=np.float64)),
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)[::2],
            np.array([8.0], dtype=np.float64),
        ),
    ],
)
def test_set_limits_keeps_python_path_for_unsupported_inputs(
    monkeypatch, data, nyquist_vel
):
    def fail_if_called(name):
        if name != "_common_dealias_limits_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported _set_limits input used Rust")

        return kernel

    monkeypatch.setattr(_common_dealias, "_rust_kernel", fail_if_called)
    try:
        actual_result, actual = _set_limits(data, nyquist_vel)
    except Exception as actual_error:
        monkeypatch.setattr(_common_dealias, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _set_limits(data, nyquist_vel)
        assert actual_error.args == expected_error.value.args
    else:
        expected_result, expected = _fallback_set_limits(data, nyquist_vel, monkeypatch)
        assert actual_result is expected_result is None
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            if isinstance(expected_value, float) and np.isnan(expected_value):
                assert np.isnan(actual[key])
            else:
                assert actual[key] == expected_value


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_set_limits_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    kernel = getattr(rust, "_common_dealias_limits_dense")
    data = np.array([-5.0, 2.0, 20.0], dtype=np.float64)
    nyquist_vel = np.array([8.0], dtype=np.float64)
    expected_result, expected = _fallback_set_limits(data, nyquist_vel, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_common_dealias_limits_dense":
            calls.append(name)
            return kernel
        return None

    monkeypatch.setattr(_common_dealias, "_rust_kernel", rust_kernel)
    actual_result, actual = _set_limits(data, nyquist_vel)

    assert calls == ["_common_dealias_limits_dense"]
    assert actual_result is expected_result is None
    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("data", "nyquist_vel", "match"),
    [
        (np.array([], dtype=np.float64), np.array([8.0], dtype=np.float64), "non-empty"),
        (np.array([1.0], dtype=np.float64), np.array([], dtype=np.float64), "non-empty"),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64)[::2],
            np.array([8.0], dtype=np.float64),
            "C-contiguous",
        ),
        (np.array([1.0, np.nan], dtype=np.float64), np.array([8.0], dtype=np.float64), "finite"),
    ],
)
def test_real_rust_set_limits_rejects_unsafe_direct_inputs(data, nyquist_vel, match):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._common_dealias_limits_dense(data, nyquist_vel)
