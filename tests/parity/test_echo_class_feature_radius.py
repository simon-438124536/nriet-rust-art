import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import _echo_class  # noqa: E402
from tools.parity_compare import assert_exact_equal  # noqa: E402


def _fallback_assign_feature_radius(field_bkg, val_for_max_rad, monkeypatch, max_rad=5):
    monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
    return _echo_class.assign_feature_radius_km(
        field_bkg, val_for_max_rad, max_rad=max_rad
    )


@pytest.mark.parametrize("max_rad", [5, 6.5, np.float64(5.0)])
def test_assign_feature_radius_python_fallback_reference_cases(monkeypatch, max_rad):
    field_bkg = np.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)

    actual = _fallback_assign_feature_radius(
        field_bkg, 20.0, monkeypatch, max_rad=max_rad
    )

    expected = np.array(
        [1.0, max_rad - 3.0, max_rad - 2.0, max_rad - 1.0, max_rad],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


def test_assign_feature_radius_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(field_bkg, val_for_max_rad, max_rad):
        calls.append((field_bkg.dtype, field_bkg.shape, val_for_max_rad, max_rad))
        return np.full(field_bkg.shape, 8.0, dtype=np.float64)

    monkeypatch.setattr(
        _echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_assign_feature_radius_km" else None,
    )
    field_bkg = np.array([[0.0, 10.0], [15.0, 20.0]], dtype=np.float64)

    actual = _echo_class.assign_feature_radius_km(field_bkg, 20.0, max_rad=5.0)

    assert calls == [(np.float64, (2, 2), 20.0, 5.0)]
    np.testing.assert_array_equal(actual, np.full((2, 2), 8.0, dtype=np.float64))
    np.testing.assert_array_equal(
        field_bkg, np.array([[0.0, 10.0], [15.0, 20.0]], dtype=np.float64)
    )


@pytest.mark.parametrize(
    ("field_bkg", "val_for_max_rad", "max_rad"),
    [
        (np.array([0.0, 10.0, 20.0], dtype=np.float32), 20.0, 5.0),
        (np.ma.array([0.0, 10.0, 20.0], mask=[False, True, False]), 20.0, 5.0),
        (np.array([0.0, np.nan, 20.0], dtype=np.float64), 20.0, 5.0),
        (np.array([0.0, np.inf, 20.0], dtype=np.float64), 20.0, 5.0),
        (np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)[::2], 20.0, 5.0),
        ([0.0, 10.0, 20.0], 20.0, 5.0),
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), np.nan, 5.0),
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), 20.0, np.inf),
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), "bad", 5.0),
    ],
)
def test_assign_feature_radius_keeps_python_path_for_unsupported_inputs(
    monkeypatch, field_bkg, val_for_max_rad, max_rad
):
    def fail_if_called(name):
        if name != "_assign_feature_radius_km":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported assign_feature_radius input used Rust")

        return kernel

    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)
    try:
        actual = _echo_class.assign_feature_radius_km(
            field_bkg, val_for_max_rad, max_rad=max_rad
        )
    except Exception as actual_error:
        monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _echo_class.assign_feature_radius_km(
                field_bkg, val_for_max_rad, max_rad=max_rad
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_assign_feature_radius(
            field_bkg, val_for_max_rad, monkeypatch, max_rad=max_rad
        )
        assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("max_rad", [5, 6.5, np.float64(5.0)])
def test_real_rust_assign_feature_radius_matches_python_fallback(
    monkeypatch, max_rad
):
    import pyart._rust as rust

    field_bkg = np.array([[0.0, 10.0], [15.0, 20.0]], dtype=np.float64)
    expected = _fallback_assign_feature_radius(
        field_bkg, 20.0, monkeypatch, max_rad=max_rad
    )
    calls = []

    def rust_kernel(name):
        if name == "_assign_feature_radius_km":
            calls.append(name)
            return rust._assign_feature_radius_km
        return None

    monkeypatch.setattr(_echo_class, "_rust_kernel", rust_kernel)
    actual_input = field_bkg.copy()
    actual = _echo_class.assign_feature_radius_km(
        actual_input, 20.0, max_rad=max_rad
    )

    assert calls == ["_assign_feature_radius_km"]
    assert_exact_equal(actual, expected)
    np.testing.assert_array_equal(actual_input, field_bkg)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("field_bkg", "val_for_max_rad", "max_rad", "match"),
    [
        (np.array([0.0, np.nan, 20.0], dtype=np.float64), 20.0, 5.0, "finite"),
        (np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)[::2], 20.0, 5.0, "C-contiguous"),
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), np.nan, 5.0, "finite"),
        (np.array([0.0, 10.0, 20.0], dtype=np.float64), 20.0, np.inf, "finite"),
    ],
)
def test_real_rust_assign_feature_radius_rejects_direct_unsafe_inputs(
    field_bkg, val_for_max_rad, max_rad, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._assign_feature_radius_km(field_bkg, val_for_max_rad, max_rad)
