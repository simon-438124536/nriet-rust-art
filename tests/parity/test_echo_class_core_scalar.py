import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import _echo_class  # noqa: E402
from tools.parity_compare import assert_exact_equal  # noqa: E402


def _sample_inputs():
    field = np.ma.array(
        [[10.0, 30.0, 40.0], [np.nan, 20.0, 4.0]],
        mask=[[False, False, False], [False, True, False]],
    )
    field_bkg = np.ma.array(
        [[5.0, 10.0, 30.0], [-1.0, 20.0, 4.0]],
        mask=[[False, False, False], [False, False, True]],
    )
    return field, field_bkg


def _fallback_core_scalar(field, field_bkg, monkeypatch, use_addition=False):
    monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
    return _echo_class.core_scalar_scheme(
        field,
        field_bkg,
        2.0,
        25.0,
        9.0,
        use_addition=use_addition,
    )


@pytest.mark.parametrize("use_addition", [False, True])
def test_core_scalar_python_fallback_reference(monkeypatch, use_addition):
    field, field_bkg = _sample_inputs()

    actual = _fallback_core_scalar(field, field_bkg, monkeypatch, use_addition)

    expected = np.ma.array(
        [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]],
        mask=[[False, False, False], [False, True, False]],
    )
    assert_exact_equal(actual, expected)


def test_core_scalar_dispatches_to_private_rust_kernel(monkeypatch):
    field, field_bkg = _sample_inputs()
    field.set_fill_value(-1234.5)
    calls = []

    def rust_kernel(
        field_data,
        field_mask,
        field_bkg_data,
        field_bkg_mask,
        max_diff,
        always_core_thres,
        core,
        use_addition,
    ):
        calls.append(
            (
                field_data.dtype,
                field_data.shape,
                field_mask.dtype,
                field_bkg_data.dtype,
                field_bkg_mask.dtype,
                max_diff,
                always_core_thres,
                core,
                use_addition,
            )
        )
        return np.full(field_data.shape, 5.0, dtype=np.float64)

    monkeypatch.setattr(
        _echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_core_scalar_scheme_f64" else None,
    )

    actual = _echo_class.core_scalar_scheme(
        field, field_bkg, 2.0, 25.0, 9.0, use_addition=True
    )

    assert calls == [
        (np.float64, (2, 3), bool, np.float64, bool, 2.0, 25.0, 9.0, True)
    ]
    expected = np.ma.array(np.full(field.shape, 5.0), mask=np.ma.getmaskarray(field))
    expected.set_fill_value(field.fill_value)
    assert_exact_equal(actual, expected)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda field, field_bkg: (field.data, field_bkg),
        lambda field, field_bkg: (
            np.ma.array(field.data.astype(np.float32), mask=field.mask),
            field_bkg,
        ),
        lambda field, field_bkg: (
            np.ma.array(field.data[:, ::2], mask=np.ma.getmaskarray(field)[:, ::2]),
            field_bkg[:, ::2],
        ),
        lambda field, field_bkg: (field, field_bkg.astype(np.float32)),
        lambda field, field_bkg: (field, field_bkg[:, :2]),
        lambda field, field_bkg: (
            np.ma.array(np.array(10.0), mask=False),
            np.ma.array(np.array(5.0), mask=False),
        ),
    ],
)
def test_core_scalar_keeps_python_path_for_unsupported_inputs(monkeypatch, mutate):
    field, field_bkg = mutate(*_sample_inputs())

    def fail_if_called(name):
        if name != "_core_scalar_scheme_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported core_scalar input used Rust")

        return kernel

    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)
    try:
        actual = _echo_class.core_scalar_scheme(field, field_bkg, 2.0, 25.0, 9.0)
    except Exception as actual_error:
        field_expected, field_bkg_expected = mutate(*_sample_inputs())
        monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _echo_class.core_scalar_scheme(field_expected, field_bkg_expected, 2.0, 25.0, 9.0)
        assert actual_error.args == expected_error.value.args
    else:
        field_expected, field_bkg_expected = mutate(*_sample_inputs())
        expected = _fallback_core_scalar(field_expected, field_bkg_expected, monkeypatch)
        assert_exact_equal(actual, expected)


def test_core_scalar_keeps_python_path_for_nonfinite_scalars(monkeypatch):
    field, field_bkg = _sample_inputs()

    def fail_if_called(name):
        if name == "_core_scalar_scheme_f64":
            raise AssertionError("non-finite scalar input used Rust")
        return None

    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)
    actual = _echo_class.core_scalar_scheme(field, field_bkg, np.nan, 25.0, 9.0)

    monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
    expected = _echo_class.core_scalar_scheme(field, field_bkg, np.nan, 25.0, 9.0)
    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("use_addition", [False, True])
def test_real_rust_core_scalar_matches_python_fallback(monkeypatch, use_addition):
    import pyart._rust as rust

    field, field_bkg = _sample_inputs()
    field.set_fill_value(-1234.5)
    expected = _fallback_core_scalar(field.copy(), field_bkg.copy(), monkeypatch, use_addition)
    calls = []

    def rust_kernel(name):
        if name == "_core_scalar_scheme_f64":
            calls.append(name)
            return rust._core_scalar_scheme_f64
        return None

    monkeypatch.setattr(_echo_class, "_rust_kernel", rust_kernel)
    actual = _echo_class.core_scalar_scheme(
        field, field_bkg, 2.0, 25.0, 9.0, use_addition=use_addition
    )

    assert calls == ["_core_scalar_scheme_f64"]
    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_core_scalar_rejects_mismatched_shapes():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="same shape"):
        rust._core_scalar_scheme_f64(
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=bool),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=bool),
            2.0,
            25.0,
            9.0,
            False,
        )
