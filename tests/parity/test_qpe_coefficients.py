import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import qpe  # noqa: E402


@pytest.mark.parametrize(
    ("freq", "rkdp", "ra"),
    [
        (2e9, (50.70, 0.8500), (3100.0, 1.03)),
        (np.nextafter(4e9, 0.0), (50.70, 0.8500), (3100.0, 1.03)),
        (4e9, (29.70, 0.8500), (250.0, 0.91)),
        (np.nextafter(8e9, 0.0), (29.70, 0.8500), (250.0, 0.91)),
        (8e9, (15.81, 0.7992), (45.5, 0.83)),
        (12e9, (15.81, 0.7992), (45.5, 0.83)),
    ],
)
def test_qpe_coeff_python_fallback_matches_band_boundaries(monkeypatch, freq, rkdp, ra):
    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert qpe._get_coeff_rkdp(freq) == rkdp
        assert qpe._get_coeff_ra(freq) == ra


@pytest.mark.parametrize(
    ("func_name", "kernel_name", "freq", "expected_band", "expected"),
    [
        ("_get_coeff_rkdp", "_qpe_coeff_rkdp", 1e9, "S", (50.70, 0.8500)),
        ("_get_coeff_rkdp", "_qpe_coeff_rkdp", 13e9, "X", (15.81, 0.7992)),
        ("_get_coeff_ra", "_qpe_coeff_ra", 1e9, "S", (3100.0, 1.03)),
        ("_get_coeff_ra", "_qpe_coeff_ra", 13e9, "X", (45.5, 0.83)),
    ],
)
def test_qpe_coeff_out_of_range_warning_then_private_rust_dispatch(
    monkeypatch, func_name, kernel_name, freq, expected_band, expected
):
    calls = []

    def fake_kernel(band):
        calls.append((kernel_name, band))
        return expected

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda name: fake_kernel if name == kernel_name else None,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = getattr(qpe, func_name)(freq)

    assert actual == expected
    assert calls == [(kernel_name, expected_band)]
    messages = [str(warning.message) for warning in caught]
    assert "Unknown frequency band" in messages
    assert any("Radar frequency out of range" in message for message in messages)


@pytest.mark.parametrize("_get_coeff", [qpe._get_coeff_rkdp, qpe._get_coeff_ra])
def test_qpe_coeff_nan_preserves_oracle_unboundlocalerror(monkeypatch, _get_coeff):
    def fail_if_called(_name):
        raise AssertionError("NaN frequency reached Rust QPE coefficient kernel")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)

    with pytest.warns(UserWarning, match="Unknown frequency band"):
        with pytest.raises(UnboundLocalError):
            _get_coeff(np.nan)


def test_qpe_coeff_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(name):
        if name == "_qpe_coeff_rkdp":
            return lambda band: calls.append((name, band)) or (1.0, 2.0)
        if name == "_qpe_coeff_ra":
            return lambda band: calls.append((name, band)) or (3.0, 4.0)
        return None

    monkeypatch.setattr(qpe, "_rust_kernel", rust_kernel)

    assert qpe._get_coeff_rkdp(2e9) == (1.0, 2.0)
    assert qpe._get_coeff_ra(8e9) == (3.0, 4.0)
    assert calls == [("_qpe_coeff_rkdp", "S"), ("_qpe_coeff_ra", "X")]


def test_qpe_coeff_monkeypatched_tables_keep_python_path(monkeypatch):
    monkeypatch.setattr(qpe, "_coeff_rkdp_table", lambda: {"S": (1.0, 2.0)})
    monkeypatch.setattr(qpe, "_coeff_ra_table", lambda: {"X": (3.0, 4.0)})

    def fail_if_called(_name):
        raise AssertionError("monkeypatched QPE coefficient table used Rust")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)

    assert qpe._get_coeff_rkdp(2e9) == (1.0, 2.0)
    assert qpe._get_coeff_ra(8e9) == (3.0, 4.0)


def test_qpe_coeff_array_valued_monkeypatched_table_keeps_python_path(monkeypatch):
    expected = np.array([1.0, 2.0], dtype=np.float64)
    monkeypatch.setattr(qpe, "_coeff_rkdp_table", lambda: {"S": expected})

    def fail_if_called(_name):
        raise AssertionError("array-valued QPE coefficient table used Rust")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)

    actual = qpe._get_coeff_rkdp(2e9)
    assert actual is expected


@pytest.mark.parametrize(
    ("func_name", "table_name", "freq"),
    [
        ("_get_coeff_rkdp", "_coeff_rkdp_table", 4e9),
        ("_get_coeff_ra", "_coeff_ra_table", 8e9),
    ],
)
def test_qpe_coeff_incomplete_monkeypatched_table_preserves_oracle_error(
    monkeypatch, func_name, table_name, freq
):
    monkeypatch.setattr(qpe, table_name, lambda: {"S": (1.0, 2.0)})

    def fail_if_called(_name):
        raise AssertionError("incomplete QPE coefficient table used Rust")

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)

    with pytest.raises(UnboundLocalError):
        getattr(qpe, func_name)(freq)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_qpe_coeff_helpers_match_python_tables(monkeypatch):
    import pyart._rust as rust

    assert rust._qpe_coeff_rkdp("S") == qpe._coeff_rkdp_table()["S"]
    assert rust._qpe_coeff_rkdp("C") == qpe._coeff_rkdp_table()["C"]
    assert rust._qpe_coeff_rkdp("X") == qpe._coeff_rkdp_table()["X"]
    assert rust._qpe_coeff_ra("S") == qpe._coeff_ra_table()["S"]
    assert rust._qpe_coeff_ra("C") == qpe._coeff_ra_table()["C"]
    assert rust._qpe_coeff_ra("X") == qpe._coeff_ra_table()["X"]

    with pytest.raises(ValueError, match="freq_band must be one of S, C, or X"):
        rust._qpe_coeff_rkdp("K")
    with pytest.raises(ValueError, match="freq_band must be one of S, C, or X"):
        rust._qpe_coeff_ra("K")

    monkeypatch.setattr(qpe, "_rust_kernel", lambda name: getattr(rust, name, None))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert qpe._get_coeff_rkdp(4e9) == qpe._coeff_rkdp_table()["C"]
        assert qpe._get_coeff_ra(12e9) == qpe._coeff_ra_table()["X"]
