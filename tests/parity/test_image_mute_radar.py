import copy
import os

import numpy as np
import pytest

from pyart.testing import make_empty_ppi_radar
from pyart.util import radar_utils
from tools.parity_compare import assert_exact_equal


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_image_mute_mask_dense_f64"):
        pytest.skip("pyart._rust has no image mute mask kernel")
    return rust


def _fallback_mask(data_to_mute, data_mute_by, mute_threshold, field_threshold, monkeypatch):
    monkeypatch.setattr(radar_utils, "_rust_kernel", lambda _name: None)
    return radar_utils._image_mute_mask(
        data_to_mute, data_mute_by, mute_threshold, field_threshold
    )


def _make_radar(data_to_mute, data_mute_by):
    data_to_mute = np.ma.array(data_to_mute, copy=True)
    data_mute_by = np.ma.array(data_mute_by, copy=True)
    radar = make_empty_ppi_radar(data_to_mute.shape[1], data_to_mute.shape[0], 1)
    radar.fields["reflectivity"] = {
        "data": data_to_mute,
        "units": "dBZ",
        "long_name": "Reflectivity",
    }
    radar.fields["rhohv"] = {
        "data": data_mute_by,
        "units": "1",
        "long_name": "RhoHV",
    }
    return radar


@pytest.mark.parametrize(
    "field_threshold",
    [None, 2.0],
)
def test_image_mute_mask_python_fallback_reference_cases(monkeypatch, field_threshold):
    data_to_mute = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.8], [np.nan, -np.inf]], dtype=np.float64)

    actual = _fallback_mask(data_to_mute, data_mute_by, 0.5, field_threshold, monkeypatch)

    assert actual.shape == data_to_mute.shape
    assert actual.dtype == bool
    if field_threshold is None:
        assert actual.tolist() == [[True, False], [False, True]]
    else:
        assert actual.tolist() == [[False, False], [False, True]]


def test_image_mute_mask_dispatches_dense_float64_to_private_rust(monkeypatch):
    data_to_mute = np.array([[1.0, 2.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.9]], dtype=np.float64)
    mask = np.array([[True, False]], dtype=bool)
    calls = []

    def kernel(data_arg, mute_arg, mute_threshold, has_field_threshold, field_threshold):
        calls.append(
            (
                data_arg.dtype,
                data_arg.shape,
                mute_arg.dtype,
                mute_arg.shape,
                mute_threshold,
                has_field_threshold,
                field_threshold,
            )
        )
        return mask.copy()

    monkeypatch.setattr(
        radar_utils,
        "_rust_kernel",
        lambda name: kernel if name == "_image_mute_mask_dense_f64" else None,
    )

    actual = radar_utils._image_mute_mask(data_to_mute, data_mute_by, 0.5, 1.5)

    assert calls == [
        (np.dtype(np.float64), (1, 2), np.dtype(np.float64), (1, 2), 0.5, True, 1.5)
    ]
    assert_exact_equal(actual, mask)


def test_image_mute_mask_rust_runtime_error_keeps_python_path(monkeypatch):
    data_to_mute = np.array([[1.0, 2.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.9]], dtype=np.float64)

    def rust_kernel(name):
        if name != "_image_mute_mask_dense_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(radar_utils, "_rust_kernel", rust_kernel)
    actual = radar_utils._image_mute_mask(data_to_mute, data_mute_by, 0.5, None)
    expected = _fallback_mask(data_to_mute, data_mute_by, 0.5, None, monkeypatch)

    assert_exact_equal(actual, expected)


def test_image_mute_mask_oversized_output_keeps_python_path(monkeypatch):
    data_to_mute = np.array([[1.0, 2.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.9]], dtype=np.float64)

    def rust_kernel(name):
        if name == "_image_mute_mask_dense_f64":
            raise AssertionError("oversized image mute input used Rust kernel")
        return None

    monkeypatch.setattr(radar_utils, "_rust_kernel", rust_kernel)
    monkeypatch.setattr(radar_utils, "IMAGE_MUTE_RUST_MAX_OUTPUT_VALUES", 1)

    actual = radar_utils._image_mute_mask(data_to_mute, data_mute_by, 0.5, None)
    expected = _fallback_mask(data_to_mute, data_mute_by, 0.5, None, monkeypatch)

    assert_exact_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        lambda: (
            np.ma.array([[1.0, 2.0]], mask=[[False, True]]),
            np.array([[0.1, 0.9]], dtype=np.float64),
            0.5,
            None,
        ),
        lambda: (
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[0.1, 0.9]], dtype=np.float32),
            0.5,
            None,
        ),
        lambda: (
            np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::3],
            np.arange(12, 24, dtype=np.float64).reshape(2, 6)[:, ::3],
            20.0,
            None,
        ),
        lambda: (
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[0.1], [0.9]], dtype=np.float64),
            0.5,
            1.0,
        ),
        lambda: (
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[0.1, 0.9]], dtype=np.float64),
            "0.5",
            None,
        ),
    ],
)
def test_image_mute_mask_unsupported_inputs_keep_python_path(monkeypatch, case):
    data_to_mute, data_mute_by, mute_threshold, field_threshold = case()

    def rust_kernel(name):
        if name == "_image_mute_mask_dense_f64":
            raise AssertionError("unsupported image mute input used Rust kernel")
        return None

    monkeypatch.setattr(radar_utils, "_rust_kernel", rust_kernel)
    try:
        actual = radar_utils._image_mute_mask(
            data_to_mute, data_mute_by, mute_threshold, field_threshold
        )
    except Exception as actual_error:
        expected_data, expected_mute, expected_threshold, expected_field_threshold = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_mask(
                expected_data,
                expected_mute,
                expected_threshold,
                expected_field_threshold,
                monkeypatch,
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected_data, expected_mute, expected_threshold, expected_field_threshold = case()
        expected = _fallback_mask(
            expected_data,
            expected_mute,
            expected_threshold,
            expected_field_threshold,
            monkeypatch,
        )
        assert_exact_equal(actual, expected)


def test_image_mute_radar_public_fields_match_python_fallback(monkeypatch):
    data_to_mute = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.8], [0.2, 0.3]], dtype=np.float64)
    radar = _make_radar(data_to_mute, data_mute_by)
    expected_radar = copy.deepcopy(radar)

    expected = radar_utils.image_mute_radar(
        expected_radar, "reflectivity", "rhohv", 0.5, field_threshold=2.5
    )

    mask = np.array([[False, False], [True, True]], dtype=bool)
    monkeypatch.setattr(radar_utils, "_image_mute_mask", lambda *_args: mask.copy())
    actual = radar_utils.image_mute_radar(
        radar, "reflectivity", "rhohv", 0.5, field_threshold=2.5
    )

    assert actual.fields["nonmuted_reflectivity"]["long_name"] == "Non-muted reflectivity"
    assert actual.fields["muted_reflectivity"]["long_name"] == "Muted reflectivity"
    assert_exact_equal(
        actual.fields["nonmuted_reflectivity"]["data"],
        expected.fields["nonmuted_reflectivity"]["data"],
    )
    assert_exact_equal(
        actual.fields["muted_reflectivity"]["data"],
        expected.fields["muted_reflectivity"]["data"],
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust image mute parity",
)
@pytest.mark.parametrize("field_threshold", [None, 2.0])
def test_image_mute_mask_real_rust_matches_python_fallback(monkeypatch, field_threshold):
    data_to_mute = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.8], [np.nan, -np.inf]], dtype=np.float64)
    expected = _fallback_mask(data_to_mute, data_mute_by, 0.5, field_threshold, monkeypatch)
    monkeypatch.undo()

    actual = radar_utils._image_mute_mask(
        data_to_mute, data_mute_by, 0.5, field_threshold
    )

    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust image mute checks",
)
def test_image_mute_mask_direct_rust_helper():
    rust = _rust_or_skip()
    data_to_mute = np.array([[1.0, 3.0]], dtype=np.float64)
    data_mute_by = np.array([[0.1, 0.9]], dtype=np.float64)

    assert rust._image_mute_mask_dense_f64(data_to_mute, data_mute_by, 0.5, True, 2.0).tolist() == [
        [False, False]
    ]
    with pytest.raises(ValueError):
        rust._image_mute_mask_dense_f64(
            data_to_mute, data_mute_by.reshape(2, 1), 0.5, False, 0.0
        )
