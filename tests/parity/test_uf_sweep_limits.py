import os

import numpy as np
import pytest

import pyart
import pyart.io.uffile as uffile


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_uf_sweep_limits_i32"):
        pytest.skip("pyart._rust has no UF sweep-limits kernel")
    return rust


def _uf_obj(ray_sweep_numbers, nsweeps=None):
    obj = uffile.UFFile.__new__(uffile.UFFile)
    obj.ray_sweep_numbers = ray_sweep_numbers
    if nsweeps is None:
        nsweeps = len(np.unique(ray_sweep_numbers))
    obj.nsweeps = nsweeps
    return obj


def _fallback_limits(ray_sweep_numbers, monkeypatch, nsweeps=None):
    monkeypatch.setattr(uffile, "_rust_kernel", lambda _name: None)
    return _uf_obj(ray_sweep_numbers, nsweeps)._get_sweep_limits()


def test_uf_sweep_limits_python_fallback_reference(monkeypatch):
    ray_sweep_numbers = np.array([3, 1, 3, 2, 1, 2, 2], dtype=np.int32)

    first, last = _fallback_limits(ray_sweep_numbers, monkeypatch)

    assert first.dtype == np.int32
    assert last.dtype == np.int32
    np.testing.assert_array_equal(first, np.array([1, 3, 0], dtype=np.int32))
    np.testing.assert_array_equal(last, np.array([4, 6, 2], dtype=np.int32))


def test_uf_sweep_limits_python_fallback_empty(monkeypatch):
    first, last = _fallback_limits(np.array([], dtype=np.int32), monkeypatch)

    assert first.dtype == np.int32
    assert last.dtype == np.int32
    assert first.shape == (0,)
    assert last.shape == (0,)


def test_uf_sweep_limits_dispatches_dense_int32_to_private_rust(monkeypatch):
    calls = []

    def kernel(ray_sweep_numbers):
        calls.append((ray_sweep_numbers.dtype, ray_sweep_numbers.shape))
        return (
            np.array([10, 20], dtype=np.int32),
            np.array([11, 21], dtype=np.int32),
        )

    monkeypatch.setattr(
        uffile,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_sweep_limits_i32" else None,
    )

    first, last = _uf_obj(np.array([2, 1, 2], dtype=np.int32), nsweeps=2)._get_sweep_limits()

    assert calls == [(np.dtype(np.int32), (3,))]
    np.testing.assert_array_equal(first, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(last, np.array([11, 21], dtype=np.int32))


@pytest.mark.parametrize(
    "case",
    [
        lambda: np.array([3, 1, 3, 2, 1, 2], dtype=np.int32)[::2],
        lambda: np.array([[1, 2], [1, 3]], dtype=np.int32),
        lambda: np.array([1.0, np.nan], dtype=np.float64),
        lambda: np.array(["b", "a", "b"], dtype=object),
        lambda: np.array(1, dtype=np.int32),
    ],
)
def test_uf_sweep_limits_unsupported_inputs_keep_python_path(monkeypatch, case):
    def fail_if_called(name):
        if name == "_uf_sweep_limits_i32":
            raise AssertionError("unsupported UF sweep input used Rust kernel")
        return None

    ray_sweep_numbers = case()
    monkeypatch.setattr(uffile, "_rust_kernel", fail_if_called)

    try:
        actual = _uf_obj(ray_sweep_numbers)._get_sweep_limits()
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_limits(case(), monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_limits(case(), monkeypatch)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "kernel_result",
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([0, 2], dtype=np.int32),
        ),
        (
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
        ),
        (
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 2]], dtype=np.int32),
        ),
    ],
)
def test_uf_sweep_limits_bad_rust_output_keeps_python_path(
    monkeypatch, kernel_result
):
    def kernel(_ray_sweep_numbers):
        return kernel_result

    monkeypatch.setattr(
        uffile,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_sweep_limits_i32" else None,
    )
    ray_sweep_numbers = np.array([1, 2, 1], dtype=np.int32)

    actual = _uf_obj(ray_sweep_numbers)._get_sweep_limits()
    expected = _fallback_limits(ray_sweep_numbers, monkeypatch)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_uf_sweep_limits_inconsistent_nsweeps_keeps_python_exception(monkeypatch):
    def kernel(_ray_sweep_numbers):
        return (
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        )

    monkeypatch.setattr(
        uffile,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_sweep_limits_i32" else None,
    )
    ray_sweep_numbers = np.array([1, 2, 1], dtype=np.int32)

    with pytest.raises(IndexError) as actual_error:
        _uf_obj(ray_sweep_numbers, nsweeps=1)._get_sweep_limits()
    with pytest.raises(IndexError) as expected_error:
        _fallback_limits(ray_sweep_numbers, monkeypatch, nsweeps=1)
    assert actual_error.value.args == expected_error.value.args


def test_uf_sweep_limits_oversized_nsweeps_keeps_python_partial_fill(monkeypatch):
    def kernel(_ray_sweep_numbers):
        return (
            np.array([0, 1], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
        )

    monkeypatch.setattr(
        uffile,
        "_rust_kernel",
        lambda name: kernel if name == "_uf_sweep_limits_i32" else None,
    )
    ray_sweep_numbers = np.array([1, 2, 1], dtype=np.int32)

    first, last = _uf_obj(ray_sweep_numbers, nsweeps=3)._get_sweep_limits()

    assert first.dtype == np.int32
    assert last.dtype == np.int32
    assert first.shape == (3,)
    assert last.shape == (3,)
    np.testing.assert_array_equal(first[:2], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(last[:2], np.array([2, 1], dtype=np.int32))


def test_public_read_uf_sample_preserves_sweep_limit_surface():
    radar = pyart.io.read_uf(pyart.testing.UF_FILE)

    assert radar.metadata["original_container"] == "UF"
    assert radar.sweep_start_ray_index["data"].dtype == np.int32
    assert radar.sweep_end_ray_index["data"].dtype == np.int32
    assert radar.sweep_start_ray_index["data"].shape == (radar.nsweeps,)
    assert radar.sweep_end_ray_index["data"].shape == (radar.nsweeps,)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust UF sweep-limit parity",
)
def test_uf_sweep_limits_real_rust_matches_python_fallback(monkeypatch):
    ray_sweep_numbers = np.array([3, 1, 3, 2, 1, 2, 2], dtype=np.int32)
    expected = _fallback_limits(ray_sweep_numbers, monkeypatch)
    monkeypatch.undo()

    actual = _uf_obj(ray_sweep_numbers)._get_sweep_limits()

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust UF sweep-limit checks",
)
def test_uf_sweep_limits_direct_rust_helper():
    rust = _rust_or_skip()

    first, last = rust._uf_sweep_limits_i32(
        np.array([3, 1, 3, 2, 1, 2, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(first, np.array([1, 3, 0], dtype=np.int32))
    np.testing.assert_array_equal(last, np.array([4, 6, 2], dtype=np.int32))

    first, last = rust._uf_sweep_limits_i32(np.array([], dtype=np.int32))
    assert first.dtype == np.int32
    assert last.dtype == np.int32
    assert first.shape == (0,)
    assert last.shape == (0,)

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._uf_sweep_limits_i32(np.arange(6, dtype=np.int32)[::2])
    with pytest.raises(ValueError, match="size limit"):
        rust._uf_sweep_limits_i32(
            np.zeros(uffile.UF_SWEEP_LIMITS_RUST_MAX_RAYS + 1, dtype=np.int32)
        )
