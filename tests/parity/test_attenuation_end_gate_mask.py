import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import attenuation  # noqa: E402


def _fallback_end_gate(mask, nrays, ngates, monkeypatch):
    monkeypatch.setattr(attenuation, "_rust_kernel", lambda _name: None)
    return attenuation._end_gate_arr_from_excluded_mask(mask, nrays, ngates)


def _python_end_gate_oracle(mask, nrays, ngates):
    out = np.zeros(nrays, dtype="int32")
    for ray in range(nrays):
        ind_rng = np.where(mask[ray, :] == 1)[0]
        if len(ind_rng) > 0:
            if ind_rng[0] > 0:
                out[ray] = ind_rng[0] - 1
            else:
                out[ray] = 0
        else:
            out[ray] = ngates - 1
    return out


def test_end_gate_python_fallback_matches_oracle_edges(monkeypatch):
    mask = np.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, False, True],
            [True, True, True, True],
        ],
        dtype=np.bool_,
    )

    actual = _fallback_end_gate(mask, 5, 4, monkeypatch)

    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, np.array([3, 0, 0, 2, 0], dtype=np.int32))

    zero_gates = np.zeros((2, 0), dtype=np.bool_)
    actual = _fallback_end_gate(zero_gates, 2, 0, monkeypatch)
    np.testing.assert_array_equal(actual, np.array([-1, -1], dtype=np.int32))


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int8),
        np.array(
            [[False, False, True, False], [False, False, False, True]],
            dtype=np.bool_,
        )[:, ::-1],
    ],
)
def test_end_gate_unsupported_inputs_keep_python_path(monkeypatch, mask):
    expected = _python_end_gate_oracle(mask, mask.shape[0], mask.shape[1])

    def fail_if_called(name):
        if name != "_attenuation_end_gate_from_excluded_mask":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported end_gate input used Rust")

        return kernel

    monkeypatch.setattr(attenuation, "_rust_kernel", fail_if_called)
    actual = attenuation._end_gate_arr_from_excluded_mask(
        mask, mask.shape[0], mask.shape[1]
    )

    np.testing.assert_array_equal(actual, expected)


def test_end_gate_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(mask):
        calls.append((mask.dtype, mask.shape, mask.flags.c_contiguous, mask.copy()))
        return np.array([9, 8], dtype=np.int32)

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_attenuation_end_gate_from_excluded_mask"
        else None,
    )
    mask = np.array([[False, True, False], [False, False, False]], dtype=np.bool_)

    actual = attenuation._end_gate_arr_from_excluded_mask(mask, 2, 3)

    assert calls[0][0:3] == (np.bool_, (2, 3), True)
    np.testing.assert_array_equal(calls[0][3], mask)
    np.testing.assert_array_equal(actual, np.array([9, 8], dtype=np.int32))


@pytest.mark.parametrize(
    "rust_output",
    [
        np.array([1, 2], dtype=np.int64),
        np.array([[1, 2]], dtype=np.int32),
    ],
)
def test_end_gate_bad_rust_output_falls_back(monkeypatch, rust_output):
    mask = np.array([[False, True, False], [False, False, False]], dtype=np.bool_)
    expected = _fallback_end_gate(mask, 2, 3, monkeypatch)

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: (lambda _mask: rust_output)
        if name == "_attenuation_end_gate_from_excluded_mask"
        else None,
    )

    actual = attenuation._end_gate_arr_from_excluded_mask(mask, 2, 3)

    np.testing.assert_array_equal(actual, expected)


def test_get_mask_fzl_temperature_branch_uses_shared_end_gate_helper(monkeypatch):
    mask = np.array([[False, False, True], [False, False, False]], dtype=np.bool_)
    radar = SimpleNamespace(
        nrays=2,
        ngates=3,
        fields={"temperature": {"data": np.zeros((2, 3), dtype=np.float64)}},
    )

    monkeypatch.setattr(
        attenuation,
        "temp_based_gate_filter",
        lambda *_args, **_kwargs: SimpleNamespace(gate_excluded=mask),
    )
    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: (lambda _mask: np.array([7, 6], dtype=np.int32))
        if name == "_attenuation_end_gate_from_excluded_mask"
        else None,
    )

    mask_fzl, end_gate_arr = attenuation.get_mask_fzl(
        radar, temp_ref="temperature", temp_field="temperature"
    )

    np.testing.assert_array_equal(mask_fzl, mask == 1)
    np.testing.assert_array_equal(end_gate_arr, np.array([7, 6], dtype=np.int32))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_end_gate_helper_matches_python_fallback(monkeypatch):
    mask = np.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, False, True],
            [True, True, True, True],
        ],
        dtype=np.bool_,
    )
    expected = _fallback_end_gate(mask, 5, 4, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = attenuation._end_gate_arr_from_excluded_mask(mask, 5, 4)

    assert actual.dtype == expected.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed wrapper fallback is verified in installed-wheel mode",
)
def test_end_gate_installed_wrapper_falls_back_after_rust_value_error(monkeypatch):
    mask = np.array([[False, True, False], [False, False, False]], dtype=np.bool_)
    expected = _fallback_end_gate(mask, 2, 3, monkeypatch)

    def rust_kernel(_mask):
        raise ValueError("synthetic installed Rust rejection")

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_attenuation_end_gate_from_excluded_mask"
        else None,
    )

    actual = attenuation._end_gate_arr_from_excluded_mask(mask, 2, 3)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_end_gate_helper_rejects_non_contiguous_input():
    import pyart._rust as rust

    mask = np.array(
        [[False, False, True], [False, True, False]],
        dtype=np.bool_,
    )[:, ::-1]

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._attenuation_end_gate_from_excluded_mask(mask)
