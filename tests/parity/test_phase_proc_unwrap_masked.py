import os

import numpy as np
import pytest
from numpy import ma

from pyart.correct import phase_proc


def _fallback_unwrap(lon, monkeypatch, centered=False, copy=True):
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    return phase_proc.unwrap_masked(lon, centered=centered, copy=copy)


def _assert_unwrap_equal(actual, expected):
    if ma.isMaskedArray(expected):
        assert ma.isMaskedArray(actual)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(ma.getmaskarray(actual), ma.getmaskarray(expected))
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.fill_value == expected.fill_value
    else:
        assert type(actual) is np.ndarray
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_unwrap_masked_python_fallback_reference_plain(monkeypatch):
    lon = np.array([10.0, 350.0, 20.0, -170.0], dtype=np.float64)

    actual = _fallback_unwrap(lon, monkeypatch)

    np.testing.assert_array_equal(
        actual, np.array([10.0, -10.0, 20.0, 190.0], dtype=np.float64)
    )


def test_unwrap_masked_dispatches_to_private_rust_kernel_for_masked(monkeypatch):
    lon = ma.array(
        [10.0, 999.0, 350.0, 20.0],
        mask=[False, True, False, False],
        fill_value=-1234.5,
    )
    calls = []

    def rust_kernel(data, mask):
        calls.append((data.copy(), mask.copy()))
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_phase_proc_unwrap_masked_degrees_f64"
        else None,
    )

    actual = phase_proc.unwrap_masked(lon, centered=False)

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][0], np.array([10.0, 999.0, 350.0, 20.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        calls[0][1], np.array([False, True, False, False], dtype=bool)
    )
    expected = ma.array(
        [1.0, 999.0, 2.0, 3.0],
        mask=[False, True, False, False],
        fill_value=-1234.5,
    )
    _assert_unwrap_equal(actual, expected)
    assert actual.fill_value == -1234.5


def test_unwrap_masked_dispatches_centering_to_numpy_round(monkeypatch):
    lon = np.array([0.0, 10.0, 20.0], dtype=np.float64)

    def rust_kernel(_data, _mask):
        return np.array([180.0, 180.0, 180.0], dtype=np.float64)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_phase_proc_unwrap_masked_degrees_f64"
        else None,
    )

    actual = phase_proc.unwrap_masked(lon, centered=True)

    np.testing.assert_array_equal(
        actual, np.array([180.0, 180.0, 180.0], dtype=np.float64)
    )


def test_unwrap_masked_bad_rust_output_keeps_python_path(monkeypatch):
    lon = np.array([10.0, 350.0, 20.0], dtype=np.float64)
    expected = _fallback_unwrap(lon.copy(), monkeypatch)
    monkeypatch.undo()

    def rust_kernel(_data, _mask):
        return np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_phase_proc_unwrap_masked_degrees_f64"
        else None,
    )

    actual = phase_proc.unwrap_masked(lon.copy())

    _assert_unwrap_equal(actual, expected)


@pytest.mark.parametrize(
    "lon_factory",
    [
        lambda: np.array([10.0], dtype=np.float64),
        lambda: ma.array([10.0, 20.0], mask=[True, False], fill_value=999.0),
        lambda: np.array([[10.0, 20.0]], dtype=np.float64),
        lambda: np.array([10.0, np.nan], dtype=np.float64),
    ],
)
def test_unwrap_masked_unsupported_inputs_keep_python_path(monkeypatch, lon_factory):
    def fail_if_called(name):
        if name == "_phase_proc_unwrap_masked_degrees_f64":
            raise AssertionError("unsupported unwrap input used Rust")
        return None

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)
    lon = lon_factory()

    try:
        actual = phase_proc.unwrap_masked(lon)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_unwrap(lon_factory(), monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_unwrap(lon_factory(), monkeypatch)
        _assert_unwrap_equal(actual, expected)


def test_unwrap_masked_fill_value_and_invalid_mask_match_python(monkeypatch):
    lon = ma.array(
        [10.0, 999.0, 350.0, np.inf, 20.0, np.nan, -170.0],
        mask=[False, True, False, False, False, False, False],
        fill_value=-777.0,
    )
    expected = _fallback_unwrap(lon.copy(), monkeypatch)
    monkeypatch.undo()

    actual = phase_proc.unwrap_masked(lon.copy())

    _assert_unwrap_equal(actual, expected)


@pytest.mark.parametrize(
    "lon_factory",
    [
        lambda: np.array([10.0, 350.0, 20.0], dtype=np.float64),
        lambda: ma.array(
            [10.0, 999.0, 350.0, 20.0],
            mask=[False, True, False, False],
            fill_value=-321.0,
        ),
    ],
)
def test_unwrap_masked_copy_flag_preserves_oracle_no_alias_behavior(
    monkeypatch, lon_factory
):
    expected = _fallback_unwrap(lon_factory(), monkeypatch, copy=False)
    monkeypatch.undo()

    source = lon_factory()
    original_data = np.array(ma.getdata(source), copy=True)
    original_mask = np.array(ma.getmaskarray(source), copy=True)
    actual = phase_proc.unwrap_masked(source, copy=False)

    _assert_unwrap_equal(actual, expected)
    assert actual is not source
    np.testing.assert_array_equal(ma.getdata(source), original_data)
    np.testing.assert_array_equal(ma.getmaskarray(source), original_mask)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust unwrap_masked parity",
)
@pytest.mark.parametrize(
    "lon, centered",
    [
        (np.array([10.0, 350.0, 20.0, -170.0], dtype=np.float64), False),
        (np.array([0.0, 180.0, 0.0, -180.0], dtype=np.float64), False),
        (
            ma.array(
                [10.0, 999.0, 350.0, 20.0, -170.0],
                mask=[False, True, False, False, False],
                fill_value=-999.0,
            ),
            False,
        ),
        (np.array([350.0, 10.0, 30.0, 50.0], dtype=np.float64), True),
    ],
)
def test_unwrap_masked_real_rust_matches_python_fallback(monkeypatch, lon, centered):
    expected = _fallback_unwrap(lon.copy(), monkeypatch, centered=centered)
    monkeypatch.undo()

    actual = phase_proc.unwrap_masked(lon.copy(), centered=centered)

    _assert_unwrap_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust unwrap_masked checks",
)
def test_unwrap_masked_direct_rust_helper():
    import pyart._rust as rust

    values = np.array([10.0, 999.0, 350.0, 20.0, -170.0], dtype=np.float64)
    mask = np.array([False, True, False, False, False], dtype=bool)

    actual = rust._phase_proc_unwrap_masked_degrees_f64(values, mask)
    np.testing.assert_array_equal(
        actual, np.array([10.0, -10.0, 20.0, 190.0], dtype=np.float64)
    )

    actual = rust._phase_proc_unwrap_masked_degrees_f64(
        np.array([0.0, 180.0, 0.0, -180.0], dtype=np.float64),
        np.array([False, False, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        actual, np.array([0.0, 180.0, 0.0, -180.0], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="same length"):
        rust._phase_proc_unwrap_masked_degrees_f64(
            values, np.array([False, True], dtype=bool)
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._phase_proc_unwrap_masked_degrees_f64(values[::2], mask[::2])
    with pytest.raises(ValueError, match="at least two valid"):
        rust._phase_proc_unwrap_masked_degrees_f64(
            values, np.array([True, True, True, True, False], dtype=bool)
        )
    with pytest.raises(ValueError, match="finite"):
        rust._phase_proc_unwrap_masked_degrees_f64(
            np.array([10.0, np.nan], dtype=np.float64),
            np.array([False, False], dtype=bool),
        )
