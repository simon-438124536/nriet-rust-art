import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.util import hildebrand_sekhon  # noqa: E402


def _reference_estimate_noise_hs74(spectrum, navg=1, nnoise_min=1):
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)

    rtest = 1 + 1 / navg
    sum1 = 0.0
    sum2 = 0.0
    for i, pwr in enumerate(sorted_spectrum):
        npts = i + 1
        sum1 += pwr
        sum2 += pwr * pwr

        if npts < nnoise_min:
            continue

        if npts * sum2 < sum1 * sum1 * rtest:
            nnoise = npts
        else:
            sum1 -= pwr
            sum2 -= pwr * pwr
            break

    mean = sum1 / nnoise
    var = sum2 / nnoise - mean * mean
    threshold = sorted_spectrum[nnoise - 1]
    return mean, threshold, var, nnoise


def test_estimate_noise_hs74_python_fallback_matches_reference(monkeypatch):
    monkeypatch.setattr(hildebrand_sekhon, "_rust_kernel", lambda: None)
    spectrum = np.array([4.0, 1.0, 2.0, 1.5, 3.0, 10.0], dtype=np.float64)

    actual = hildebrand_sekhon.estimate_noise_hs74(
        spectrum, navg=2, nnoise_min=2
    )
    expected = _reference_estimate_noise_hs74(spectrum, navg=2, nnoise_min=2)

    assert actual == expected


def test_estimate_noise_hs74_dispatches_for_float64_arrays(monkeypatch):
    calls = []

    def rust_kernel(spectrum, navg, nnoise_min):
        calls.append((spectrum.dtype, spectrum.shape, navg, nnoise_min))
        return 1.0, 2.0, 3.0, 4

    monkeypatch.setattr(hildebrand_sekhon, "_rust_kernel", lambda: rust_kernel)
    spectrum = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert hildebrand_sekhon.estimate_noise_hs74(spectrum, 1, 1) == (
        1.0,
        2.0,
        3.0,
        4,
    )
    assert calls == [(np.dtype("float64"), (3,), 1.0, 1)]


def test_estimate_noise_hs74_keeps_python_path_for_non_float64(monkeypatch):
    def rust_kernel(*_args):
        raise AssertionError("non-float64 spectra must use Python fallback")

    monkeypatch.setattr(hildebrand_sekhon, "_rust_kernel", lambda: rust_kernel)
    spectrum = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    actual = hildebrand_sekhon.estimate_noise_hs74(spectrum, 1, 1)
    expected = _reference_estimate_noise_hs74(spectrum, 1, 1)

    assert actual == expected


@pytest.mark.parametrize(
    ("spectrum", "navg"),
    [
        (np.array([], dtype=np.float64), 1),
        (np.array([1.0, 2.0], dtype=np.float64), 0),
    ],
)
def test_estimate_noise_hs74_exception_parity_for_degenerate_inputs(
    monkeypatch, spectrum, navg
):
    monkeypatch.setattr(hildebrand_sekhon, "_rust_kernel", lambda: None)
    with pytest.raises(Exception) as expected_error:
        hildebrand_sekhon.estimate_noise_hs74(spectrum, navg=navg, nnoise_min=1)

    class_name = type(expected_error.value)
    monkeypatch.undo()

    with pytest.raises(class_name):
        hildebrand_sekhon.estimate_noise_hs74(spectrum, navg=navg, nnoise_min=1)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("spectrum", "navg"),
    [
        (np.array([], dtype=np.float64), 1),
        (np.array([1.0, 2.0], dtype=np.float64), 0),
    ],
)
def test_real_rust_estimate_noise_hs74_degenerate_exceptions_match_fallback(
    monkeypatch, spectrum, navg
):
    monkeypatch.setattr(hildebrand_sekhon, "_rust_kernel", lambda: None)
    with pytest.raises(Exception) as expected_error:
        hildebrand_sekhon.estimate_noise_hs74(spectrum, navg=navg, nnoise_min=1)
    expected_type = type(expected_error.value)

    monkeypatch.undo()
    with pytest.raises(expected_type):
        hildebrand_sekhon.estimate_noise_hs74(spectrum, navg=navg, nnoise_min=1)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_estimate_noise_hs74_matches_python_fallback(monkeypatch):
    spectrum = np.array([8.0, 1.0, 1.2, 1.5, 2.0, 6.0, 0.9], dtype=np.float64)

    actual = hildebrand_sekhon.estimate_noise_hs74(
        spectrum, navg=3, nnoise_min=2
    )
    monkeypatch.setattr(hildebrand_sekhon, "_rust_kernel", lambda: None)
    expected = hildebrand_sekhon.estimate_noise_hs74(
        spectrum, navg=3, nnoise_min=2
    )

    assert actual == expected
