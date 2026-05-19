import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import _echo_class  # noqa: E402


def _fallback_radial_mask(mask, *args, **kwargs):
    original_kernel = _echo_class._rust_kernel
    _echo_class._rust_kernel = lambda _name: None
    try:
        return _echo_class.create_radial_mask(mask, *args, **kwargs)
    finally:
        _echo_class._rust_kernel = original_kernel


def test_create_radius_mask_python_fallback_matches_oracle_annulus(monkeypatch):
    monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)

    actual = _echo_class.create_radius_mask(5, 2.0, 1.0, 0.5, 2)
    expected = np.zeros((5, 5))
    _echo_class.create_radial_mask(expected, 0, 2.0, 1.0, 0.5, 2, 2, True)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_create_radial_mask_dispatches_to_private_rust_kernel_for_square_float64(
    monkeypatch,
):
    calls = []

    def rust_kernel(mask, min_rad, max_rad, x_pix, y_pix, center_x, center_y):
        calls.append(
            (
                mask.dtype,
                mask.shape,
                min_rad,
                max_rad,
                x_pix,
                y_pix,
                center_x,
                center_y,
            )
        )
        mask[:] = 4.0

    monkeypatch.setattr(
        _echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_create_radial_mask_circular" else None,
    )
    mask = np.zeros((3, 3), dtype=np.float64)

    result = _echo_class.create_radial_mask(mask, 0.5, 1.5, 1.0, 2.0, 1, 1, True)

    assert result is mask
    assert calls == [
        (np.float64, (3, 3), 0.5, 1.5, 1.0, 2.0, 1.0, 1.0)
    ]
    np.testing.assert_array_equal(mask, np.full((3, 3), 4.0))


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.int32),
        np.ma.array(np.zeros((3, 3), dtype=np.float64)),
    ],
)
def test_create_radial_mask_keeps_python_path_for_unsupported_dtypes_and_masks(
    monkeypatch, mask
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported radial-mask inputs should use fallback")

        return kernel

    expected = _fallback_radial_mask(mask.copy(), 0, 1.5, 1.0, 1.0, 1, 1, True)
    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)

    actual = _echo_class.create_radial_mask(mask.copy(), 0, 1.5, 1.0, 1.0, 1, 1, True)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_create_radial_mask_keeps_python_path_for_square_non_circular(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("non-circular masks should use fallback")

        return kernel

    expected = _fallback_radial_mask(
        np.zeros((3, 3), dtype=np.float64), 0, 1.5, 1.0, 1.0, 1, 1, False
    )
    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)
    actual = _echo_class.create_radial_mask(
        np.zeros((3, 3), dtype=np.float64), 0, 1.5, 1.0, 1.0, 1, 1, False
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "circular", "message"),
    [
        ((2, 3), True, "axis 0 with size 2"),
        ((2, 3), False, "axis 0 with size 2"),
        ((3, 2), True, "axis 1 with size 2"),
        ((3, 2), False, "axis 1 with size 2"),
    ],
)
def test_create_radial_mask_preserves_rectangular_fallback_exception(
    monkeypatch, shape, circular, message
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("rectangular masks should use fallback")

        return kernel

    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)

    with pytest.raises(IndexError):
        _echo_class.create_radial_mask(
            np.zeros(shape, dtype=np.float64), 0, 1.5, 1.0, 1.0, 1, 1, circular
        )

    monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
    with pytest.raises(IndexError, match=message):
        _echo_class.create_radial_mask(
            np.zeros(shape, dtype=np.float64), 0, 1.5, 1.0, 1.0, 1, 1, circular
        )


@pytest.mark.parametrize(
    "args",
    [
        ("bad", 1.5, 1.0, 1.0, 1, 1, True),
        (0, "bad", 1.0, 1.0, 1, 1, True),
        (0, 1.5, "bad", 1.0, 1, 1, True),
        (0, 1.5, np.nan, 1.0, 1, 1, True),
    ],
)
def test_create_radial_mask_keeps_python_path_for_unsupported_scalars(
    monkeypatch, args
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported scalar inputs should use fallback")

        return kernel

    base = np.zeros((3, 3), dtype=np.float64)
    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)

    try:
        actual = _echo_class.create_radial_mask(base.copy(), *args)
    except Exception as actual_error:
        monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)):
            _echo_class.create_radial_mask(base.copy(), *args)
    else:
        monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
        expected = _echo_class.create_radial_mask(base.copy(), *args)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_create_radial_mask_matches_python_fallback(monkeypatch):
    mask = np.zeros((5, 5), dtype=np.float64)
    expected = _fallback_radial_mask(mask.copy(), 0.5, 2.0, 1.0, 0.5, 2, 2, True)

    import pyart._rust as rust

    monkeypatch.setattr(
        _echo_class,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual_input = mask.copy()
    actual = _echo_class.create_radial_mask(
        actual_input, 0.5, 2.0, 1.0, 0.5, 2, 2, True
    )

    assert actual is actual_input
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_radial_mask_rejects_rectangular_direct_call_without_mutation():
    import pyart._rust as rust

    mask = np.zeros((2, 3), dtype=np.float64)
    before = mask.copy()

    with pytest.raises(ValueError) as exc_info:
        rust._create_radial_mask_circular(mask, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    assert exc_info.value.args == (
        "circular radial mask Rust path requires a square array",
    )
    np.testing.assert_array_equal(mask, before)
