import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import echo_class  # noqa: E402


FREQ_BAND_CASES = [
    (2.0e9 - 1.0, None),
    (2.0e9, "S"),
    (4.0e9 - 1.0, "S"),
    (4.0e9, "C"),
    (8.0e9 - 1.0, "C"),
    (8.0e9, "X"),
    (12.0e9, "X"),
    (12.0e9 + 1.0, None),
    (float("nan"), None),
    (float("inf"), None),
    (-1.0, None),
]


def _fallback_get_freq_band(freq, monkeypatch):
    monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = echo_class.get_freq_band(freq)
    return result, [(item.category, str(item.message)) for item in caught]


def _call_get_freq_band(freq):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = echo_class.get_freq_band(freq)
    return result, [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize(("freq", "expected"), FREQ_BAND_CASES)
def test_get_freq_band_python_fallback_reference_cases(monkeypatch, freq, expected):
    result, caught = _fallback_get_freq_band(freq, monkeypatch)

    assert result == expected
    if expected is None:
        assert caught == [(UserWarning, "Unknown frequency band")]
    else:
        assert caught == []


def test_get_freq_band_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(freq):
        calls.append(freq)
        return "Q"

    monkeypatch.setattr(
        echo_class,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_echo_class_get_freq_band" else None,
    )

    result, caught = _call_get_freq_band(3.0e9)

    assert result == "Q"
    assert calls == [3.0e9]
    assert caught == []


def test_get_freq_band_dispatch_warns_when_rust_reports_unknown(monkeypatch):
    monkeypatch.setattr(
        echo_class,
        "_rust_kernel",
        lambda name: (lambda _freq: None)
        if name == "_echo_class_get_freq_band"
        else None,
    )

    result, caught = _call_get_freq_band(1.0)

    assert result is None
    assert caught == [(UserWarning, "Unknown frequency band")]


@pytest.mark.parametrize("freq", [np.float32(3.0e9), True, False, "3", object()])
def test_get_freq_band_keeps_python_path_for_unsupported_inputs(monkeypatch, freq):
    def fail_if_called(name):
        if name != "_echo_class_get_freq_band":
            return None

        def kernel(_freq):
            raise AssertionError("unsupported get_freq_band input used Rust")

        return kernel

    monkeypatch.setattr(echo_class, "_rust_kernel", fail_if_called)
    try:
        actual_result, actual_warnings = _call_get_freq_band(freq)
    except Exception as actual_error:
        expected_result = None
        monkeypatch.setattr(echo_class, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            expected_result, _ = _call_get_freq_band(freq)
        assert expected_result is None
        assert actual_error.args == expected_error.value.args
    else:
        expected_result, expected_warnings = _fallback_get_freq_band(freq, monkeypatch)
        assert actual_result == expected_result
        assert actual_warnings == expected_warnings


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(("freq", "_expected"), FREQ_BAND_CASES)
def test_real_rust_get_freq_band_matches_python_fallback(monkeypatch, freq, _expected):
    import pyart._rust as rust

    expected_result, expected_warnings = _fallback_get_freq_band(freq, monkeypatch)
    calls = []

    def rust_kernel(name):
        if name == "_echo_class_get_freq_band":
            calls.append(name)
            return rust._echo_class_get_freq_band
        return None

    monkeypatch.setattr(echo_class, "_rust_kernel", rust_kernel)
    actual_result, actual_warnings = _call_get_freq_band(freq)

    assert calls == ["_echo_class_get_freq_band"]
    assert actual_result == expected_result
    assert actual_warnings == expected_warnings


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust kernel is verified in installed-wheel mode",
)
def test_real_rust_get_freq_band_direct_kernel():
    import pyart._rust as rust

    assert rust._echo_class_get_freq_band(2.0e9) == "S"
    assert rust._echo_class_get_freq_band(4.0e9) == "C"
    assert rust._echo_class_get_freq_band(8.0e9) == "X"
    assert rust._echo_class_get_freq_band(12.0e9 + 1.0) is None
    assert rust._echo_class_get_freq_band(float("nan")) is None
