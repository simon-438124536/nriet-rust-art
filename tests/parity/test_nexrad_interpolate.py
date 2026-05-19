import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.io import nexrad_interpolate  # noqa: E402


def _scan4_expected(values, *, fill_value=-9999.0, linear_interp=True):
    values = np.asarray(values, dtype=np.float32)
    out = np.repeat(values, 4).astype(np.float32)
    if linear_interp:
        for i in range(2, out.shape[0] - 4, 4):
            gate_val = out[i]
            next_val = out[i + 4]
            if gate_val == fill_value or next_val == fill_value:
                continue
            delta = np.float32((next_val - gate_val) / np.float32(4.0))
            out[i] = gate_val + delta * np.float32(0.5)
            out[i + 1] = gate_val + delta * np.float32(1.5)
            out[i + 2] = gate_val + delta * np.float32(2.5)
            out[i + 3] = gate_val + delta * np.float32(3.5)
    return out


def _scan2_expected(values, *, fill_value=-9999.0, linear_interp=True):
    values = np.asarray(values, dtype=np.float32)
    out = np.empty((2 * values.shape[0] - 1,), dtype=np.float32)
    for i, gate_val in enumerate(values):
        out[i * 2] = gate_val
        if i != values.shape[0] - 1:
            out[i * 2 + 1] = gate_val
    if linear_interp:
        for i in range(1, out.shape[0] - 2, 2):
            gate_val = out[i]
            next_val = out[i + 2]
            if gate_val == fill_value or next_val == fill_value:
                continue
            delta = np.float32((next_val - gate_val) / np.float32(2.0))
            out[i] = gate_val + delta * np.float32(0.5)
            out[i + 1] = gate_val + delta * np.float32(1.5)
    return out


def test_scan4_python_fallback_linear_and_nearest(monkeypatch):
    monkeypatch.setattr(nexrad_interpolate, "_rust_kernel", lambda _name: None)
    data = np.zeros((2, 16), dtype=np.float32)
    data[0, :4] = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    data[1, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scratch = np.zeros(16, dtype=np.float32)

    result = nexrad_interpolate._fast_interpolate_scan_4(
        data, scratch, -9999.0, 0, 0, 4, True
    )

    assert result is None
    np.testing.assert_array_equal(data[0], _scan4_expected([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_equal(data[1, :4], np.array([1.0, 2.0, 3.0, 4.0]))

    nexrad_interpolate._fast_interpolate_scan_4(data, scratch, -9999.0, 1, 1, 4, False)

    np.testing.assert_array_equal(
        data[1], _scan4_expected([1.0, 2.0, 3.0, 4.0], linear_interp=False)
    )


def test_scan2_python_fallback_linear_nearest_and_fill_guard(monkeypatch):
    monkeypatch.setattr(nexrad_interpolate, "_rust_kernel", lambda _name: None)
    fill_value = np.float32(-9999.0)
    data = np.zeros((3, 7), dtype=np.float32)
    data[0, :4] = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    data[1, :4] = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    data[2, :4] = np.array([1.0, fill_value, 5.0, 9.0], dtype=np.float32)
    scratch = np.zeros(7, dtype=np.float32)

    nexrad_interpolate._fast_interpolate_scan_2(
        data, scratch, fill_value, 0, 0, 4, True
    )
    nexrad_interpolate._fast_interpolate_scan_2(
        data, scratch, fill_value, 1, 1, 4, False
    )
    nexrad_interpolate._fast_interpolate_scan_2(
        data, scratch, fill_value, 2, 2, 4, True
    )

    np.testing.assert_array_equal(
        data[0], _scan2_expected([10.0, 20.0, 30.0, 40.0], fill_value=fill_value)
    )
    np.testing.assert_array_equal(
        data[1],
        _scan2_expected([3.0, 5.0, 7.0, 9.0], fill_value=fill_value, linear_interp=False),
    )
    np.testing.assert_array_equal(
        data[2], _scan2_expected([1.0, fill_value, 5.0, 9.0], fill_value=fill_value)
    )


def test_scan4_dispatches_to_rust_kernel_for_oracle_compatible_arrays(monkeypatch):
    calls = []

    def rust_scan4(data, scratch_ray, fill_value, start, end, moment_ngates, linear_interp):
        calls.append(
            (
                data.dtype,
                scratch_ray.dtype,
                fill_value,
                start,
                end,
                moment_ngates,
                linear_interp,
            )
        )
        data[start, : 4 * moment_ngates] = 42.0

    monkeypatch.setattr(
        nexrad_interpolate,
        "_rust_kernel",
        lambda name: rust_scan4 if name == "_fast_interpolate_scan_4" else None,
    )
    data = np.zeros((1, 16), dtype=np.float32)
    data[0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scratch = np.zeros(16, dtype=np.float32)

    result = nexrad_interpolate._fast_interpolate_scan_4(
        data, scratch, -9999.0, 0, 0, 4, True
    )

    assert result is None
    assert calls == [(np.dtype("float32"), np.dtype("float32"), -9999.0, 0, 0, 4, 1)]
    np.testing.assert_array_equal(data[0], np.full(16, 42.0, dtype=np.float32))


def test_scan2_keeps_python_path_for_fill_guard_when_not_rust_compatible(monkeypatch):
    def rust_scan2(*_args):
        raise AssertionError("Rust kernel must not run for non-float32 data")

    monkeypatch.setattr(
        nexrad_interpolate,
        "_rust_kernel",
        lambda name: rust_scan2 if name == "_fast_interpolate_scan_2" else None,
    )
    fill_value = -9999.0
    data = np.zeros((1, 7), dtype=np.float64)
    data[0, :4] = np.array([1.0, fill_value, 5.0, 9.0], dtype=np.float64)
    scratch = np.zeros(7, dtype=np.float64)

    nexrad_interpolate._fast_interpolate_scan_2(
        data, scratch, fill_value, 0, 0, 4, True
    )

    np.testing.assert_array_equal(
        data[0], _scan2_expected([1.0, fill_value, 5.0, 9.0], fill_value=fill_value)
    )


def test_scan_parameters_reject_non_integer_values():
    data = np.zeros((1, 16), dtype=np.float32)
    scratch = np.zeros(16, dtype=np.float32)

    with pytest.raises(TypeError):
        nexrad_interpolate._fast_interpolate_scan_4(
            data, scratch, -9999.0, 0.5, 0, 4, True
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_scan4_matches_python_fallback(monkeypatch):
    data_rust = np.zeros((1, 16), dtype=np.float32)
    data_rust[0, :4] = np.array([1.0, -9999.0, 5.0, 9.0], dtype=np.float32)
    data_py = data_rust.copy()
    scratch_rust = np.zeros(16, dtype=np.float32)
    scratch_py = np.zeros(16, dtype=np.float32)

    nexrad_interpolate._fast_interpolate_scan_4(
        data_rust, scratch_rust, -9999.0, 0, 0, 4, True
    )
    monkeypatch.setattr(nexrad_interpolate, "_rust_kernel", lambda _name: None)
    nexrad_interpolate._fast_interpolate_scan_4(
        data_py, scratch_py, -9999.0, 0, 0, 4, True
    )

    np.testing.assert_array_equal(data_rust, data_py)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_scan2_matches_python_fallback(monkeypatch):
    data_rust = np.zeros((1, 7), dtype=np.float32)
    data_rust[0, :4] = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    data_py = data_rust.copy()
    scratch_rust = np.zeros(7, dtype=np.float32)
    scratch_py = np.zeros(7, dtype=np.float32)

    nexrad_interpolate._fast_interpolate_scan_2(
        data_rust, scratch_rust, -9999.0, 0, 0, 4, True
    )
    monkeypatch.setattr(nexrad_interpolate, "_rust_kernel", lambda _name: None)
    nexrad_interpolate._fast_interpolate_scan_2(
        data_py, scratch_py, -9999.0, 0, 0, 4, True
    )

    np.testing.assert_array_equal(data_rust, data_py)
