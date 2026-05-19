import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import echo_class  # noqa: E402
from tools.parity_compare import assert_exact_equal  # noqa: E402


def _sample_scan_inputs(dtype=np.float64, include_mask=True):
    fields = {
        "a": np.array([[0.1, 0.9, 0.2], [0.8, 0.4, 0.7]], dtype=dtype),
        "b": np.array([[0.1, 0.1, 0.9], [0.8, 0.6, 0.3]], dtype=dtype),
    }
    if include_mask:
        fields["a"] = np.ma.array(
            fields["a"], mask=np.array([[False, True, False], [False, False, False]])
        )
        fields["b"] = np.ma.array(
            fields["b"], mask=np.array([[False, False, True], [False, True, False]])
        )

    mass_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]], dtype=dtype)
    weights = np.array([1.0, 0.75], dtype=dtype)
    return fields, mass_centers, ("a", "b"), weights


def _python_oracle(monkeypatch, fields, mass_centers, var_names, weights, t_vals=None):
    monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: None)
    return echo_class._assign_to_class_scan(
        fields,
        mass_centers,
        var_names=var_names,
        weights=weights,
        t_vals=t_vals,
    )


def test_assign_to_class_scan_no_entropy_dispatches_to_private_kernel(monkeypatch):
    fields, mass_centers, var_names, weights = _sample_scan_inputs()
    expected_hydroclass, expected_entropy, expected_t_dist = _python_oracle(
        monkeypatch, fields, mass_centers, var_names, weights
    )
    assert expected_entropy is None
    assert expected_t_dist is None
    calls = []

    def rust_kernel(data_stack, mask_stack, mass_centers_arg, weights_arg):
        calls.append(
            (
                data_stack.dtype,
                data_stack.shape,
                mask_stack.dtype,
                mask_stack.shape,
                mass_centers_arg.dtype,
                weights_arg.dtype,
            )
        )
        np.testing.assert_array_equal(mask_stack[0], np.ma.getmaskarray(fields["a"]))
        np.testing.assert_array_equal(mask_stack[1], np.ma.getmaskarray(fields["b"]))
        return np.asarray(expected_hydroclass.data, dtype=np.uint8)

    monkeypatch.setattr(
        echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_assign_to_class_scan_no_entropy" else None,
    )

    actual_hydroclass, actual_entropy, actual_t_dist = echo_class._assign_to_class_scan(
        fields, mass_centers, var_names=var_names, weights=weights, t_vals=None
    )

    assert calls == [
        (np.float64, (2, 2, 3), bool, (2, 2, 3), np.float64, np.float64)
    ]
    assert actual_entropy is None
    assert actual_t_dist is None
    assert_exact_equal(actual_hydroclass, expected_hydroclass)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fields, mass_centers, weights: (
            {**fields, "a": np.asarray(fields["a"], dtype=np.float32)},
            mass_centers,
            weights,
        ),
        lambda fields, mass_centers, weights: (
            fields,
            mass_centers.astype(np.float32),
            weights,
        ),
        lambda fields, mass_centers, weights: (
            {**fields, "a": np.ma.array([[np.nan, 0.2, 0.4], [0.1, 0.3, 0.5]])},
            mass_centers,
            weights,
        ),
        lambda fields, mass_centers, weights: (
            fields,
            mass_centers,
            np.array([1.0, -0.5], dtype=np.float64),
        ),
    ],
)
def test_assign_to_class_scan_no_entropy_keeps_python_path_for_risky_inputs(
    monkeypatch, mutate
):
    fields, mass_centers, var_names, weights = _sample_scan_inputs(include_mask=False)
    fields, mass_centers, weights = mutate(fields, mass_centers, weights)
    expected_hydroclass, expected_entropy, expected_t_dist = _python_oracle(
        monkeypatch, fields, mass_centers, var_names, weights
    )

    def rust_kernel(*_args):
        raise AssertionError("risky input should use the Python fallback")

    monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: rust_kernel)
    actual_hydroclass, actual_entropy, actual_t_dist = echo_class._assign_to_class_scan(
        fields, mass_centers, var_names=var_names, weights=weights, t_vals=None
    )

    assert actual_entropy is expected_entropy
    assert actual_t_dist is expected_t_dist
    assert_exact_equal(actual_hydroclass, expected_hydroclass)


def test_assign_to_class_scan_entropy_branch_keeps_python_path(monkeypatch):
    fields, mass_centers, var_names, weights = _sample_scan_inputs(include_mask=False)
    t_vals = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    expected = _python_oracle(monkeypatch, fields, mass_centers, var_names, weights, t_vals)

    def rust_kernel(*_args):
        raise AssertionError("entropy branch should use the Python fallback")

    monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: rust_kernel)
    actual = echo_class._assign_to_class_scan(
        fields, mass_centers, var_names=var_names, weights=weights, t_vals=t_vals
    )

    assert_exact_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_assign_to_class_scan_no_entropy_matches_python_fallback(monkeypatch):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    rust_kernel = getattr(rust, "_assign_to_class_scan_no_entropy", None)
    if rust_kernel is None:
        pytest.fail("pyart._rust has not registered _assign_to_class_scan_no_entropy")

    fields, mass_centers, var_names, weights = _sample_scan_inputs()
    expected_hydroclass, _, _ = _python_oracle(
        monkeypatch, fields, mass_centers, var_names, weights
    )
    monkeypatch.setattr(
        echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_assign_to_class_scan_no_entropy" else None,
    )

    actual_hydroclass, actual_entropy, actual_t_dist = echo_class._assign_to_class_scan(
        fields, mass_centers, var_names=var_names, weights=weights, t_vals=None
    )

    assert actual_entropy is None
    assert actual_t_dist is None
    assert_exact_equal(actual_hydroclass, expected_hydroclass)
