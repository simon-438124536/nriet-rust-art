import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import _echo_class_wt  # noqa: E402


def _fallback_atwt2d(data, max_scale, monkeypatch):
    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", lambda _name: None)
    return _echo_class_wt.atwt2d(data, max_scale=max_scale)


def _fallback_label_classes(wt_sum, dbz_data, monkeypatch, **kwargs):
    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", lambda _name: None)
    return _echo_class_wt.label_classes(wt_sum, dbz_data, **_label_kwargs(kwargs))


def _label_kwargs(overrides=None):
    kwargs = {
        "core_wt_threshold": 5.0,
        "conv_wt_threshold": 2.0,
        "min_reflectivity": 10.0,
        "conv_min_refl": 30.0,
        "conv_core_threshold": 40.0,
    }
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _assert_atwt_equal(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def _assert_array_equal_with_warning(actual, expected):
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype == np.int32


def test_label_classes_python_fallback_matches_oracle_thresholds_and_nan_warning(
    monkeypatch,
):
    wt_sum = np.array(
        [[5.0, 3.0, 3.0, 1.0, 1.0, np.nan, np.inf]],
        dtype=np.float64,
    )
    dbz_data = np.array(
        [[30.0, 50.0, 30.0, 11.0, 5.0, 50.0, 30.0]],
        dtype=np.float64,
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        actual = _fallback_label_classes(wt_sum, dbz_data, monkeypatch)

    sentinel = np.iinfo(np.int32).min
    expected = np.array([[3, 2, 2, 1, sentinel, 1, 3]], dtype=np.int32)
    _assert_array_equal_with_warning(actual, expected)
    assert any("invalid value encountered in cast" in str(item.message) for item in records)


def test_label_classes_dispatches_to_private_rust_precursor(monkeypatch):
    calls = []

    def rust_kernel(wt_sum, dbz_data, *thresholds):
        calls.append(
            (
                wt_sum.dtype,
                wt_sum.shape,
                dbz_data.dtype,
                dbz_data.shape,
                thresholds,
            )
        )
        return np.array([[3.0, 2.0]], dtype=np.float64)

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_echo_class_wt_label_classes_f64"
        else None,
    )

    actual = _echo_class_wt.label_classes(
        np.array([[5.0, 3.0]], dtype=np.float64),
        np.array([[30.0, 30.0]], dtype=np.float64),
        **_label_kwargs(),
    )

    assert calls == [
        (
            np.dtype("float64"),
            (1, 2),
            np.dtype("float64"),
            (1, 2),
            (5.0, 2.0, 10.0, 30.0, 40.0),
        )
    ]
    np.testing.assert_array_equal(actual, np.array([[3, 2]], dtype=np.int32))


def test_label_classes_rust_precursor_keeps_numpy_cast_warning_surface(monkeypatch):
    def rust_kernel(*_args):
        return np.array([[np.nan]], dtype=np.float64)

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_echo_class_wt_label_classes_f64"
        else None,
    )
    old_seterr = np.seterr(invalid="raise")
    try:
        with pytest.raises(FloatingPointError):
            _echo_class_wt.label_classes(
                np.array([[1.0]], dtype=np.float64),
                np.array([[1.0]], dtype=np.float64),
                **_label_kwargs(),
            )
    finally:
        np.seterr(**old_seterr)


@pytest.mark.parametrize(
    ("wt_sum", "dbz_data", "kwargs"),
    [
        (
            np.array([[5.0, 3.0]], dtype=np.float32),
            np.array([[30.0, 30.0]], dtype=np.float32),
            {},
        ),
        (
            np.array([[5.0, 3.0], [1.0, 1.0]], dtype=np.float64)[:, ::-1],
            np.array([[30.0, 30.0], [11.0, 5.0]], dtype=np.float64)[:, ::-1],
            {},
        ),
        (
            np.array([5.0, 3.0, 1.0], dtype=np.float64),
            np.array([[30.0], [30.0]], dtype=np.float64),
            {},
        ),
    ],
)
def test_label_classes_keeps_python_path_for_unsupported_inputs(
    monkeypatch, wt_sum, dbz_data, kwargs
):
    expected = _fallback_label_classes(wt_sum, dbz_data, monkeypatch, **kwargs)

    def fail_if_called(name):
        if name != "_echo_class_wt_label_classes_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported label_classes input used Rust")

        return kernel

    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", fail_if_called)
    actual = _echo_class_wt.label_classes(
        wt_sum, dbz_data, **_label_kwargs(kwargs)
    )

    _assert_array_equal_with_warning(actual, expected)


def test_label_classes_object_threshold_keeps_oracle_exception(monkeypatch):
    def fail_if_called(name):
        if name != "_echo_class_wt_label_classes_f64":
            return None

        def kernel(*_args):
            raise AssertionError("object threshold used Rust")

        return kernel

    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", fail_if_called)
    with pytest.raises(TypeError):
        _echo_class_wt.label_classes(
            np.array([[5.0, 3.0]], dtype=np.float64),
            np.array([[30.0, 30.0]], dtype=np.float64),
            **_label_kwargs({"core_wt_threshold": object()}),
        )


def test_label_classes_masked_inputs_use_underlying_data_like_oracle(monkeypatch):
    wt_sum = np.ma.array([[5.0, 1.0]], mask=[[True, False]], dtype=np.float64)
    dbz_data = np.ma.array([[30.0, 5.0]], mask=[[True, False]], dtype=np.float64)
    expected = _fallback_label_classes(wt_sum, dbz_data, monkeypatch)

    calls = []

    def rust_kernel(wt_sum_data, dbz_data_data, *_thresholds):
        calls.append((wt_sum_data.copy(), dbz_data_data.copy()))
        return np.array([[3.0, np.nan]], dtype=np.float64)

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_echo_class_wt_label_classes_f64"
        else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = _echo_class_wt.label_classes(wt_sum, dbz_data, **_label_kwargs())

    np.testing.assert_array_equal(calls[0][0], np.array([[5.0, 1.0]]))
    np.testing.assert_array_equal(calls[0][1], np.array([[30.0, 5.0]]))
    _assert_array_equal_with_warning(actual, expected)


def test_atwt2d_python_fallback_matches_stable_shape_and_mutation(monkeypatch):
    data = np.arange(64.0, dtype=np.float64).reshape(8, 8)
    expected_input = data.copy()

    wt, background = _fallback_atwt2d(expected_input, 2, monkeypatch)

    assert wt.shape == (2, 8, 8)
    assert background is expected_input
    np.testing.assert_array_equal(background, expected_input)
    assert not np.array_equal(data, expected_input)


def test_atwt2d_dispatches_to_private_rust_kernel_for_float64_arrays(monkeypatch):
    calls = []

    def rust_kernel(data, max_scale):
        calls.append((data.dtype, data.shape, data.flags.c_contiguous, max_scale))
        data[:] = 7.0
        return np.full((max_scale,) + data.shape, 3.0, dtype=np.float64)

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_atwt2d" else None,
    )
    data = np.arange(16.0, dtype=np.float64).reshape(4, 4)

    wt, background = _echo_class_wt.atwt2d(data, max_scale=99)

    assert calls == [(np.float64, (4, 4), True, 1)]
    np.testing.assert_array_equal(wt, np.full((1, 4, 4), 3.0))
    assert background is data
    np.testing.assert_array_equal(data, np.full((4, 4), 7.0))


@pytest.mark.parametrize(
    "data",
    [
        np.arange(16.0, dtype=np.float32).reshape(4, 4),
        np.arange(16.0, dtype=np.float64).reshape(4, 4)[:, ::-1],
        np.ma.array(np.arange(16.0, dtype=np.float64).reshape(4, 4)),
        np.arange(4.0, dtype=np.float64),
        [[1.0, 2.0], [3.0, 4.0]],
    ],
)
def test_atwt2d_keeps_python_path_for_unsupported_inputs(monkeypatch, data):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported ATWT inputs should use fallback")

        return kernel

    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", fail_if_called)

    if not isinstance(data, np.ndarray) or np.ndim(data) != 2:
        with pytest.raises(Exception):
            _echo_class_wt.atwt2d(data, max_scale=1)
        return

    expected_input = data.copy()
    expected = _fallback_atwt2d(expected_input, 1, monkeypatch)
    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", fail_if_called)

    actual_input = data
    actual = _echo_class_wt.atwt2d(actual_input, max_scale=1)

    _assert_atwt_equal(actual, expected)
    np.testing.assert_array_equal(actual_input, expected_input)


@pytest.mark.parametrize(
    ("shape", "expected_error"),
    [
        ((1, 4), IndexError),
        ((4, 1), IndexError),
        ((1, 1), ValueError),
    ],
)
def test_atwt2d_singleton_axes_preserve_python_exception_path(
    monkeypatch, shape, expected_error
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("singleton-axis inputs should use Python fallback")

        return kernel

    monkeypatch.setattr(_echo_class_wt, "_rust_kernel", fail_if_called)
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)

    with pytest.raises(expected_error):
        _echo_class_wt.atwt2d(data, max_scale=1)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("shape", "max_scale"),
    [
        ((8, 8), 2),
        ((6, 8), 2),
        ((4, 4), 99),
        ((2, 4), 99),
    ],
)
def test_real_rust_atwt2d_matches_python_fallback(monkeypatch, shape, max_scale):
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 3.0
    expected_input = base.copy()
    actual_input = base.copy()

    expected = _fallback_atwt2d(expected_input, max_scale, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = _echo_class_wt.atwt2d(actual_input, max_scale=max_scale)

    _assert_atwt_equal(actual, expected)
    np.testing.assert_array_equal(actual_input, expected_input)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_atwt2d_rejects_direct_unsafe_scale_without_mutation():
    import pyart._rust as rust

    data = np.arange(8.0, dtype=np.float64).reshape(2, 4)
    before = data.copy()

    with pytest.raises(ValueError) as exc_info:
        rust._atwt2d(data, 1)

    assert exc_info.value.args == (
        "max_scale exceeds supported ATWT scale for input shape",
    )
    np.testing.assert_array_equal(data, before)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_label_classes_matches_python_fallback(monkeypatch):
    wt_sum = np.array(
        [[5.0, 3.0, 3.0, 1.0, 1.0, np.nan, np.inf]],
        dtype=np.float64,
    )
    dbz_data = np.array(
        [[30.0, 50.0, 30.0, 11.0, 5.0, 50.0, 30.0]],
        dtype=np.float64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = _fallback_label_classes(wt_sum, dbz_data, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = _echo_class_wt.label_classes(wt_sum, dbz_data, **_label_kwargs())

    _assert_array_equal_with_warning(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed warning parity is verified in installed-wheel mode",
)
def test_real_rust_label_classes_preserves_seterr_invalid_raise(monkeypatch):
    import pyart._rust as rust

    monkeypatch.setattr(
        _echo_class_wt,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    old_seterr = np.seterr(invalid="raise")
    try:
        with pytest.raises(FloatingPointError):
            _echo_class_wt.label_classes(
                np.array([[1.0]], dtype=np.float64),
                np.array([[1.0]], dtype=np.float64),
                **_label_kwargs(),
            )
    finally:
        np.seterr(**old_seterr)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_label_classes_direct_helper_returns_precursor_values():
    import pyart._rust as rust

    wt_sum = np.array([[5.0, 3.0, 3.0, 1.0, 1.0]], dtype=np.float64)
    dbz_data = np.array([[30.0, 50.0, 30.0, 11.0, 5.0]], dtype=np.float64)
    actual = rust._echo_class_wt_label_classes_f64(
        wt_sum,
        dbz_data,
        5.0,
        2.0,
        10.0,
        30.0,
        40.0,
    )

    expected = np.array([[3.0, 2.0, 2.0, 1.0, np.nan]], dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_label_classes_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    wt_sum = np.array([[5.0, 3.0], [1.0, 1.0]], dtype=np.float64)
    dbz_data = np.array([[30.0, 30.0], [11.0, 5.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._echo_class_wt_label_classes_f64(
            wt_sum[:, ::-1],
            dbz_data[:, ::-1],
            5.0,
            2.0,
            10.0,
            30.0,
            40.0,
        )

    with pytest.raises(ValueError, match="same shape"):
        rust._echo_class_wt_label_classes_f64(
            wt_sum,
            dbz_data[:, :1],
            5.0,
            2.0,
            10.0,
            30.0,
            40.0,
        )
