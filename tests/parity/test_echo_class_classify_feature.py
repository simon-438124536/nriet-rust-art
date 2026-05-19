import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import _echo_class  # noqa: E402
from tools.parity_compare import assert_exact_equal  # noqa: E402


ARGS = (0, 2, 1, 3, 9, 5.0, 15.0)


def _sample_inputs():
    field = np.ma.array(
        [[0.0, 10.0, 20.0], [30.0, 40.0, 4.0]],
        mask=[[False, False, False], [False, True, False]],
    )
    feature = np.ma.array(
        np.full(field.shape, -7.0, dtype=np.float64),
        mask=[[False, True, False], [False, False, False]],
    )
    core = np.ma.array(
        [[0.0, 9.0, 0.0], [9.0, 9.0, 0.0]],
        mask=[[False, False, False], [False, False, False]],
    )
    return field, feature, core


def _fallback_classify(field, feature, core, monkeypatch):
    monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
    return _echo_class.classify_feature_array(field, feature, core, *ARGS)


def test_classify_feature_array_python_fallback_reference(monkeypatch):
    field, feature, core = _sample_inputs()

    actual = _fallback_classify(field, feature, core, monkeypatch)

    expected = np.ma.array(
        [[0.0, 3.0, 1.0], [2.0, 2.0, 0.0]],
        mask=np.zeros(field.shape, dtype=bool),
    )
    assert actual is feature
    assert_exact_equal(actual, expected)


def test_classify_feature_array_dispatches_to_private_rust_kernel(monkeypatch):
    field, feature, core = _sample_inputs()
    calls = []

    def rust_kernel(
        field_data,
        field_mask,
        feature_data,
        core_data,
        core_mask,
        *scalar_values,
    ):
        calls.append(
            (
                field_data.dtype,
                field_data.shape,
                field_mask.dtype,
                feature_data.dtype,
                core_data.dtype,
                core_mask.dtype,
                scalar_values,
            )
        )
        feature_data[:] = 4.0

    monkeypatch.setattr(
        _echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_classify_feature_array_f64" else None,
    )

    actual = _echo_class.classify_feature_array(field, feature, core, *ARGS)

    assert actual is feature
    assert calls == [
        (
            np.float64,
            (2, 3),
            bool,
            np.float64,
            np.float64,
            bool,
            tuple(float(value) for value in ARGS),
        )
    ]
    np.testing.assert_array_equal(feature.data, np.full(field.shape, 4.0))
    np.testing.assert_array_equal(np.ma.getmaskarray(feature), np.zeros(field.shape, dtype=bool))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda field, feature, core: (field.data, feature, core),
        lambda field, feature, core: (
            np.ma.array(field.data.astype(np.float32), mask=field.mask),
            feature,
            core,
        ),
        lambda field, feature, core: (
            np.ma.array(field.data[:, ::2], mask=np.ma.getmaskarray(field)[:, ::2]),
            feature[:, ::2],
            core[:, ::2],
        ),
        lambda field, feature, core: (field, feature.astype(np.int32), core),
        lambda field, feature, core: (
            field,
            _readonly_mask_feature(feature),
            core,
        ),
        lambda field, feature, core: (field, feature[:, :2], core),
        lambda field, feature, core: (field, feature, core.astype(np.int32)),
        lambda field, feature, core: (field, feature, core[:, :2]),
        lambda field, feature, core: (
            np.ma.array(np.array(4.0), mask=False),
            np.array(0.0, dtype=np.float64),
            np.array(0.0, dtype=np.float64),
        ),
    ],
)
def test_classify_feature_array_keeps_python_path_for_unsupported_inputs(
    monkeypatch, mutate
):
    field, feature, core = mutate(*_sample_inputs())

    def fail_if_called(name):
        if name != "_classify_feature_array_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported classify_feature_array input used Rust")

        return kernel

    monkeypatch.setattr(_echo_class, "_rust_kernel", fail_if_called)
    try:
        actual = _echo_class.classify_feature_array(field, feature, core, *ARGS)
    except Exception as actual_error:
        field_expected, feature_expected, core_expected = mutate(*_sample_inputs())
        monkeypatch.setattr(_echo_class, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            _echo_class.classify_feature_array(
                field_expected, feature_expected, core_expected, *ARGS
            )
        assert actual_error.args == expected_error.value.args
    else:
        field_expected, feature_expected, core_expected = mutate(*_sample_inputs())
        expected = _fallback_classify(
            field_expected, feature_expected, core_expected, monkeypatch
        )
        assert_exact_equal(actual, expected)


def _readonly_mask_feature(feature):
    mask = np.ma.getmaskarray(feature).copy()
    mask.flags.writeable = False
    return np.ma.array(feature.data.copy(), mask=mask, copy=False)


def test_classify_feature_array_preserves_nan_comparison_rules(monkeypatch):
    field = np.ma.array([np.nan, 4.0, 10.0, 20.0], mask=[False, False, False, False])
    feature = np.zeros((4,), dtype=np.float64)
    core = np.array([0.0, 9.0, 0.0, 0.0], dtype=np.float64)
    expected = _fallback_classify(field.copy(), feature.copy(), core.copy(), monkeypatch)

    actual = _echo_class.classify_feature_array(field, feature, core, *ARGS)

    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_classify_feature_array_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    field, feature, core = _sample_inputs()
    expected = _fallback_classify(field.copy(), feature.copy(), core.copy(), monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_classify_feature_array_f64":
            calls.append(name)
            return rust._classify_feature_array_f64
        return None

    monkeypatch.setattr(_echo_class, "_rust_kernel", rust_kernel)
    field, feature, core = _sample_inputs()
    actual = _echo_class.classify_feature_array(field, feature, core, *ARGS)

    assert calls == ["_classify_feature_array_f64"]
    assert actual is feature
    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_classify_feature_array_rejects_mismatched_shapes():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="same shape"):
        rust._classify_feature_array_f64(
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=bool),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=bool),
            *tuple(float(value) for value in ARGS),
        )
