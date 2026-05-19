import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import attenuation  # noqa: E402


def _fallback_prepare_phidp(phidp, mask_fzl, monkeypatch):
    monkeypatch.setattr(attenuation, "_rust_kernel", lambda _name: None)
    return attenuation._prepare_phidp(phidp, mask_fzl)


def _assert_exact_array(actual, expected):
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_prepare_phidp_python_fallback_matches_oracle_edges(monkeypatch):
    phidp = np.array(
        [
            [-0.0, 0.0, -1.0, 2.0, np.nan, 5.0],
            [1.0, 0.5, 2.0, -0.0, 3.0, 1.0],
        ],
        dtype=np.float64,
    )
    mask_fzl = np.array(
        [
            [False, False, False, True, False, False],
            [False, True, False, False, False, False],
        ],
        dtype=np.bool_,
    )

    actual = _fallback_prepare_phidp(phidp, mask_fzl, monkeypatch)

    mask_phidp = np.ma.getmaskarray(phidp)
    mask_phidp = np.logical_or(mask_phidp, mask_fzl)
    mask_phidp = np.logical_or(mask_phidp, phidp < 0.0)
    corr_phidp = np.ma.masked_where(mask_phidp, phidp)
    expected = np.maximum.accumulate(corr_phidp.filled(fill_value=0.0), axis=1)

    assert actual.dtype == expected.dtype == np.float64
    _assert_exact_array(actual, expected)


def test_prepare_phidp_python_fallback_preserves_masked_array_contract(monkeypatch):
    phidp = np.ma.array(
        [[1.0, -2.0, 3.0, 2.0], [-0.0, 4.0, 1.0, np.nan]],
        mask=[[False, True, False, False], [False, False, True, False]],
        dtype=np.float64,
    )
    mask_fzl = np.array(
        [[False, False, True, False], [False, False, False, False]],
        dtype=np.bool_,
    )

    actual = _fallback_prepare_phidp(phidp, mask_fzl, monkeypatch)

    mask_phidp = np.ma.getmaskarray(phidp)
    mask_phidp = np.logical_or(mask_phidp, mask_fzl)
    mask_phidp = np.logical_or(mask_phidp, phidp < 0.0)
    corr_phidp = np.ma.masked_where(mask_phidp, phidp)
    expected = np.maximum.accumulate(corr_phidp.filled(fill_value=0.0), axis=1)

    assert type(actual) is np.ndarray
    assert actual.dtype == expected.dtype == np.float64
    _assert_exact_array(actual, expected)


def test_prepare_phidp_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(phidp_data, phidp_mask, mask_fzl):
        calls.append(
            (
                phidp_data.dtype,
                phidp_data.shape,
                phidp_mask.dtype,
                phidp_mask.copy(),
                mask_fzl.dtype,
                mask_fzl.copy(),
            )
        )
        return np.full(phidp_data.shape, 7.0, dtype=np.float64)

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: rust_kernel
        if name == "_attenuation_prepare_phidp_dense"
        else None,
    )
    phidp = np.ma.array(
        [[1.0, 2.0], [3.0, 4.0]],
        mask=[[False, True], [False, False]],
        dtype=np.float64,
    )
    mask_fzl = np.array([[False, False], [True, False]], dtype=np.bool_)

    actual = attenuation._prepare_phidp(phidp, mask_fzl)

    assert calls[0][0:3] == (np.float64, (2, 2), np.bool_)
    np.testing.assert_array_equal(calls[0][3], np.ma.getmaskarray(phidp))
    assert calls[0][4] == np.bool_
    np.testing.assert_array_equal(calls[0][5], mask_fzl)
    np.testing.assert_array_equal(actual, np.full((2, 2), 7.0, dtype=np.float64))


@pytest.mark.parametrize(
    ("phidp", "mask_fzl"),
    [
        (
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[False, False]], dtype=np.bool_),
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)[:, ::-1],
            np.array([[False, False], [False, False]], dtype=np.bool_),
        ),
        (
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[0, 1]], dtype=np.int8),
        ),
    ],
)
def test_prepare_phidp_keeps_python_path_for_unsupported_inputs(
    monkeypatch, phidp, mask_fzl
):
    expected = _fallback_prepare_phidp(phidp, mask_fzl, monkeypatch)

    def fail_if_called(name):
        if name != "_attenuation_prepare_phidp_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported prepare_phidp input used Rust")

        return kernel

    monkeypatch.setattr(attenuation, "_rust_kernel", fail_if_called)
    actual = attenuation._prepare_phidp(phidp, mask_fzl)

    _assert_exact_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("masked", [False, True])
def test_real_rust_prepare_phidp_matches_python_fallback(monkeypatch, masked):
    data = np.array(
        [[-0.0, 0.0, -1.0, 2.0, np.nan, 5.0], [1.0, 0.5, 2.0, -0.0, 3.0, 1.0]],
        dtype=np.float64,
    )
    phidp = (
        np.ma.array(
            data,
            mask=[
                [False, False, False, True, False, False],
                [False, True, False, False, False, False],
            ],
        )
        if masked
        else data
    )
    mask_fzl = np.array(
        [
            [False, False, False, False, False, False],
            [False, False, True, False, False, False],
        ],
        dtype=np.bool_,
    )
    expected = _fallback_prepare_phidp(phidp, mask_fzl, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = attenuation._prepare_phidp(phidp, mask_fzl)

    assert actual.dtype == expected.dtype == np.float64
    _assert_exact_array(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("phidp", "phidp_mask", "mask_fzl", "match"),
    [
        (
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 2), dtype=np.bool_),
            np.zeros((2, 3), dtype=np.bool_),
            "same shape",
        ),
        (
            np.ones((2, 3), dtype=np.float64)[:, ::-1],
            np.zeros((2, 3), dtype=np.bool_),
            np.zeros((2, 3), dtype=np.bool_),
            "C-contiguous",
        ),
    ],
)
def test_real_rust_prepare_phidp_rejects_unsafe_direct_inputs(
    phidp, phidp_mask, mask_fzl, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._attenuation_prepare_phidp_dense(phidp, phidp_mask, mask_fzl)
