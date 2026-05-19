import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import attenuation  # noqa: E402


@pytest.mark.parametrize(
    ("freq", "attzphi", "attphilinear"),
    [
        (2e9, (0.02, 0.64884, 0.15917, 1.0804), (0.04, 0.004)),
        (np.nextafter(4e9, 0.0), (0.02, 0.64884, 0.15917, 1.0804), (0.04, 0.004)),
        (4e9, (0.08, 0.64884, 0.3, 1.0804), (0.08, 0.03)),
        (np.nextafter(8e9, 0.0), (0.08, 0.64884, 0.3, 1.0804), (0.08, 0.03)),
        (8e9, (0.31916, 0.64884, 0.15917, 1.0804), (0.28, 0.04)),
        (12e9, (0.31916, 0.64884, 0.15917, 1.0804), (0.28, 0.04)),
    ],
)
def test_attenuation_param_python_fallback_matches_band_boundaries(
    monkeypatch, freq, attzphi, attphilinear
):
    monkeypatch.setattr(attenuation, "_rust_kernel", lambda _name: None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert attenuation._get_param_attzphi(freq) == attzphi
        assert attenuation._get_param_attphilinear(freq) == attphilinear


def test_attenuation_param_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(name):
        if name == "_attenuation_param_attzphi":
            return lambda band: calls.append((name, band)) or (1.0, 2.0, 3.0, 4.0)
        if name == "_attenuation_param_attphilinear":
            return lambda band: calls.append((name, band)) or (5.0, 6.0)
        return None

    monkeypatch.setattr(attenuation, "_rust_kernel", rust_kernel)

    assert attenuation._get_param_attzphi(2e9) == (1.0, 2.0, 3.0, 4.0)
    assert attenuation._get_param_attphilinear(8e9) == (5.0, 6.0)
    assert calls == [
        ("_attenuation_param_attzphi", "S"),
        ("_attenuation_param_attphilinear", "X"),
    ]


@pytest.mark.parametrize("freq", [1e9, 13e9, np.nan])
def test_attzphi_out_of_range_preserves_oracle_typeerror(monkeypatch, freq):
    def fail_if_called(_name):
        raise AssertionError("attzphi out-of-range path reached Rust")

    monkeypatch.setattr(attenuation, "_rust_kernel", fail_if_called)

    with pytest.warns(UserWarning, match="Unknown frequency band"):
        with pytest.raises(TypeError, match="NoneType"):
            attenuation._get_param_attzphi(freq)


@pytest.mark.parametrize(
    ("freq", "expected_band", "expected"),
    [
        (1e9, "S", (0.04, 0.004)),
        (13e9, "X", (0.28, 0.04)),
    ],
)
def test_attphilinear_out_of_range_warning_then_private_rust_dispatch(
    monkeypatch, freq, expected_band, expected
):
    calls = []

    def rust_kernel(band):
        calls.append(band)
        return expected

    monkeypatch.setattr(
        attenuation,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_attenuation_param_attphilinear" else None,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = attenuation._get_param_attphilinear(freq)

    assert actual == expected
    assert calls == [expected_band]
    messages = [str(warning.message) for warning in caught]
    assert "Unknown frequency band" in messages
    assert any("Radar frequency out of range" in message for message in messages)


def test_attphilinear_nan_preserves_oracle_unboundlocalerror(monkeypatch):
    def fail_if_called(_name):
        raise AssertionError("attphilinear NaN path reached Rust")

    monkeypatch.setattr(attenuation, "_rust_kernel", fail_if_called)

    with pytest.warns(UserWarning, match="Unknown frequency band"):
        with pytest.raises(UnboundLocalError):
            attenuation._get_param_attphilinear(np.nan)


def test_attenuation_param_monkeypatched_tables_keep_python_path(monkeypatch):
    monkeypatch.setattr(
        attenuation,
        "_param_attzphi_table",
        lambda: {"S": (1.0, 2.0, 3.0, 4.0)},
    )
    monkeypatch.setattr(
        attenuation,
        "_param_attphilinear_table",
        lambda: {"X": (5.0, 6.0)},
    )

    def fail_if_called(_name):
        raise AssertionError("monkeypatched attenuation parameter table used Rust")

    monkeypatch.setattr(attenuation, "_rust_kernel", fail_if_called)

    assert attenuation._get_param_attzphi(2e9) == (1.0, 2.0, 3.0, 4.0)
    assert attenuation._get_param_attphilinear(8e9) == (5.0, 6.0)


@pytest.mark.parametrize(
    ("func_name", "table_name", "freq", "warns"),
    [
        ("_get_param_attzphi", "_param_attzphi_table", 4e9, True),
        ("_get_param_attphilinear", "_param_attphilinear_table", 8e9, False),
    ],
)
def test_attenuation_param_incomplete_monkeypatched_table_preserves_oracle_error(
    monkeypatch, func_name, table_name, freq, warns
):
    monkeypatch.setattr(attenuation, table_name, lambda: {"S": (1.0, 2.0)})

    def fail_if_called(_name):
        raise AssertionError("incomplete attenuation parameter table used Rust")

    monkeypatch.setattr(attenuation, "_rust_kernel", fail_if_called)

    if warns:
        with pytest.warns(UserWarning, match="Radar frequency out of range"):
            with pytest.raises(UnboundLocalError):
                getattr(attenuation, func_name)(freq)
    else:
        with pytest.raises(UnboundLocalError):
            getattr(attenuation, func_name)(freq)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_attenuation_param_helpers_match_python_tables(monkeypatch):
    import pyart._rust as rust

    attzphi_kernel = getattr(rust, "_attenuation_param_attzphi")
    attphilinear_kernel = getattr(rust, "_attenuation_param_attphilinear")

    assert attzphi_kernel("S") == attenuation._param_attzphi_table()["S"]
    assert attzphi_kernel("C") == attenuation._param_attzphi_table()["C"]
    assert attzphi_kernel("X") == attenuation._param_attzphi_table()["X"]
    assert attphilinear_kernel("S") == attenuation._param_attphilinear_table()["S"]
    assert attphilinear_kernel("C") == attenuation._param_attphilinear_table()["C"]
    assert attphilinear_kernel("X") == attenuation._param_attphilinear_table()["X"]

    with pytest.raises(ValueError, match="freq_band must be one of S, C, or X"):
        attzphi_kernel("K")
    with pytest.raises(ValueError, match="freq_band must be one of S, C, or X"):
        attphilinear_kernel("K")

    calls = []

    def rust_kernel(name):
        if name == "_attenuation_param_attzphi":
            return lambda band: calls.append((name, band)) or attzphi_kernel(band)
        if name == "_attenuation_param_attphilinear":
            return lambda band: calls.append((name, band)) or attphilinear_kernel(band)
        return None

    monkeypatch.setattr(attenuation, "_rust_kernel", rust_kernel)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert attenuation._get_param_attzphi(4e9) == attenuation._param_attzphi_table()["C"]
        assert attenuation._get_param_attphilinear(12e9) == attenuation._param_attphilinear_table()["X"]

    assert calls == [
        ("_attenuation_param_attzphi", "C"),
        ("_attenuation_param_attphilinear", "X"),
    ]

    calls.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert attenuation._get_param_attphilinear(13e9) == attenuation._param_attphilinear_table()["X"]
    assert calls == [("_attenuation_param_attphilinear", "X")]
    messages = [str(warning.message) for warning in caught]
    assert "Unknown frequency band" in messages
    assert any("Radar frequency out of range" in message for message in messages)

    with pytest.warns(UserWarning, match="Unknown frequency band"):
        with pytest.raises(TypeError, match="NoneType"):
            attenuation._get_param_attzphi(13e9)
    with pytest.warns(UserWarning, match="Unknown frequency band"):
        with pytest.raises(UnboundLocalError):
            attenuation._get_param_attphilinear(np.nan)
