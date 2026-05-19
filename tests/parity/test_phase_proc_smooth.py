import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import phase_proc  # noqa: E402


def _fallback_smooth_and_trim(x, window_len, window, monkeypatch):
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    return phase_proc.smooth_and_trim(x, window_len=window_len, window=window)


def _fallback_det_sys_phase(ncp, rhv, phidp, last_ray_idx, ncp_lev, rhv_lev, monkeypatch):
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    return phase_proc._det_sys_phase(ncp, rhv, phidp, last_ray_idx, ncp_lev, rhv_lev)


def _fallback_det_sys_phase_gf(phidp, last_ray_idx, radar_meteo, monkeypatch):
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    return phase_proc._det_sys_phase_gf(phidp, last_ray_idx, radar_meteo)


def _system_phase_inputs():
    ncp = np.full((3, 32), 0.8, dtype=np.float64)
    rhv = np.full((3, 32), 0.9, dtype=np.float64)
    phidp = np.vstack(
        [
            np.linspace(10.0, 41.0, 32, dtype=np.float64),
            np.linspace(2.0, 33.0, 32, dtype=np.float64),
            np.linspace(50.0, 81.0, 32, dtype=np.float64),
        ]
    )
    return ncp, rhv, phidp


@pytest.mark.parametrize(
    ("window", "window_len"),
    [
        ("flat", 3),
        ("hanning", 5),
        ("hamming", 6),
        ("bartlett", 7),
        ("blackman", 8),
        ("sg_smooth", 5),
    ],
)
def test_smooth_and_trim_python_fallback_preserves_window_outputs(
    monkeypatch, window, window_len
):
    x = np.linspace(1.0, 8.0, 8, dtype=np.float64)

    actual = _fallback_smooth_and_trim(x, window_len, window, monkeypatch)

    assert actual.dtype == np.float64
    assert actual.shape == x.shape


def test_smooth_and_trim_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(x, weights, window_len):
        calls.append((x.dtype, x.shape, weights.dtype, weights.shape, window_len))
        return np.array([9.0, 8.0, 7.0], dtype=np.float64)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: (
            rust_kernel if name == "_phase_proc_smooth_and_trim_f64" else None
        ),
    )
    x = np.array([1.0, 2.0, 4.0], dtype=np.float64)

    actual = phase_proc.smooth_and_trim(x, window_len=3, window="flat")

    np.testing.assert_array_equal(actual, np.array([9.0, 8.0, 7.0]))
    assert calls == [(np.float64, (3,), np.float64, (3,), 3)]


def test_det_sys_phase_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(ncp, rhv, phidp, weights, last_ray_idx, ncp_lev, rhv_lev):
        calls.append(
            (
                ncp.dtype,
                rhv.shape,
                phidp.shape,
                weights.shape,
                last_ray_idx,
                ncp_lev,
                rhv_lev,
            )
        )
        return np.float64(12.5)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: (
            rust_kernel if name == "_phase_proc_det_sys_phase_dense" else None
        ),
    )
    ncp, rhv, phidp = _system_phase_inputs()

    actual = phase_proc._det_sys_phase(ncp, rhv, phidp, 2, np.float64(0.4), 0.6)

    assert actual == np.float64(12.5)
    assert calls == [(np.float64, (3, 32), (3, 32), (9,), 2, 0.4, 0.6)]


def test_det_sys_phase_gf_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(phidp, radar_meteo, weights, last_ray_idx):
        calls.append((phidp.dtype, radar_meteo.dtype, weights.shape, last_ray_idx))
        return None

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: (
            rust_kernel if name == "_phase_proc_det_sys_phase_gf_dense" else None
        ),
    )
    _, _, phidp = _system_phase_inputs()
    radar_meteo = np.ones(phidp.shape, dtype=np.bool_)

    actual = phase_proc._det_sys_phase_gf(phidp, 2, radar_meteo)

    assert actual is None
    assert calls == [(np.float64, np.bool_, (9,), 2)]


@pytest.mark.parametrize(
    ("x", "window_len", "window"),
    [
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), 3, "flat"),
        (np.array([1, 2, 3], dtype=np.int32), 3, "flat"),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)[::2], 3, "flat"),
        (np.ma.array([1.0, 2.0, 3.0], dtype=np.float64), 3, "flat"),
        (np.array([1.0, np.nan, 3.0], dtype=np.float64), 3, "flat"),
        (np.array([1.0, np.inf, 3.0], dtype=np.float64), 3, "flat"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), 2, "flat"),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64), 3, "sg_smooth"),
    ],
)
def test_smooth_and_trim_keeps_python_path_for_unsupported_inputs(
    monkeypatch, x, window_len, window
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported smooth_and_trim input should use fallback")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        actual = phase_proc.smooth_and_trim(x, window_len=window_len, window=window)
    expected = _fallback_smooth_and_trim(x, window_len, window, monkeypatch)

    if window_len < 3:
        assert actual is x
        assert expected is x
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda ncp, rhv, phidp: (ncp.astype(np.float32), rhv, phidp, 2, 0.4, 0.6),
        lambda ncp, rhv, phidp: (np.ma.array(ncp), rhv, phidp, 2, 0.4, 0.6),
        lambda ncp, rhv, phidp: (ncp.T, rhv.T, phidp.T, 2, 0.4, 0.6),
        lambda ncp, rhv, phidp: (ncp, rhv[:, :-1], phidp, 2, 0.4, 0.6),
        lambda ncp, rhv, phidp: (ncp, rhv, phidp.copy(), -1, 0.4, 0.6),
        lambda ncp, rhv, phidp: (ncp, rhv, phidp.copy(), 3, 0.4, 0.6),
        lambda ncp, rhv, phidp: (ncp, rhv, phidp.copy(), 2, np.array([0.4]), 0.6),
        lambda ncp, rhv, phidp: (ncp, rhv, phidp.copy(), 2, np.nan, 0.6),
    ],
)
def test_det_sys_phase_keeps_python_path_for_unsupported_inputs(monkeypatch, mutator):
    def fail_kernel(*_args):
        raise AssertionError("unsupported det_sys_phase input should use fallback")

    def fail_if_called(name):
        if name != "_phase_proc_det_sys_phase_dense":
            return None

        def kernel(*_args):
            return fail_kernel(*_args)

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)
    ncp, rhv, phidp = _system_phase_inputs()
    args = mutator(ncp, rhv, phidp)

    with np.errstate(all="ignore"):
        try:
            actual = phase_proc._det_sys_phase(*args)
        except Exception as actual_error:
            monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                phase_proc._det_sys_phase(*args)
        else:
            expected = _fallback_det_sys_phase(*args, monkeypatch)
            assert actual == expected or (
                isinstance(actual, np.floating)
                and isinstance(expected, np.floating)
                and np.isnan(actual)
                and np.isnan(expected)
            )


@pytest.mark.parametrize(
    "radar_meteo",
    [
        np.ones((3, 32), dtype=np.int32),
        np.full((3, 32), np.nan, dtype=np.float64),
        np.ma.array(np.ones((3, 32), dtype=np.bool_)),
    ],
)
def test_det_sys_phase_gf_keeps_python_path_for_unsupported_masks(
    monkeypatch, radar_meteo
):
    def fail_kernel(*_args):
        raise AssertionError("unsupported det_sys_phase_gf input should use fallback")

    def fail_if_called(name):
        if name != "_phase_proc_det_sys_phase_gf_dense":
            return None

        def kernel(*_args):
            return fail_kernel(*_args)

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)
    _, _, phidp = _system_phase_inputs()

    actual = phase_proc._det_sys_phase_gf(phidp, 2, radar_meteo)
    expected = _fallback_det_sys_phase_gf(phidp, 2, radar_meteo, monkeypatch)
    assert actual == expected or (
        isinstance(actual, np.floating)
        and isinstance(expected, np.floating)
        and np.isnan(actual)
        and np.isnan(expected)
    )


def test_det_sys_phase_selection_count_and_threshold_boundaries(monkeypatch):
    ncp, rhv, phidp = _system_phase_inputs()
    ncp[0, 25:] = 0.4
    rhv[1, 26:] = 0.6
    ncp[2, :] = 0.4

    actual = _fallback_det_sys_phase(ncp, rhv, phidp, 2, 0.4, 0.6, monkeypatch)

    assert isinstance(actual, np.float64)
    expected_only_second_ray = _fallback_det_sys_phase(
        ncp[1:2], rhv[1:2], phidp[1:2], 0, 0.4, 0.6, monkeypatch
    )
    assert actual == expected_only_second_ray


@pytest.mark.parametrize(
    ("x", "window_len", "window", "error_type"),
    [
        (np.ones((2, 3), dtype=np.float64), 3, "flat", ValueError),
        (np.array([1.0, 2.0], dtype=np.float64), 3, "flat", ValueError),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), 3, "boxcar", ValueError),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), 3.0, "flat", TypeError),
    ],
)
def test_smooth_and_trim_preserves_python_exception_edges(
    monkeypatch, x, window_len, window, error_type
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("exception edge should use Python fallback")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)

    with pytest.raises(error_type):
        phase_proc.smooth_and_trim(x, window_len=window_len, window=window)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("window", "window_len"),
    [
        ("flat", 3),
        ("hanning", 5),
        ("hamming", 6),
        ("bartlett", 7),
        ("blackman", 8),
        ("sg_smooth", 5),
    ],
)
def test_real_rust_smooth_and_trim_matches_python_fallback(
    monkeypatch, window, window_len
):
    import pyart._rust as rust

    x = np.array([0.5, 2.0, 3.5, 7.0, 11.0, 13.0, 17.0, 19.0], dtype=np.float64)

    expected = _fallback_smooth_and_trim(x, window_len, window, monkeypatch)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = phase_proc.smooth_and_trim(x, window_len=window_len, window=window)

    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_smooth_and_trim_large_vectors_are_exact_and_warning_free(
    monkeypatch,
):
    import pyart._rust as rust

    base = np.linspace(-1000.0, 1000.0, 257, dtype=np.float64)
    x = (
        np.sin(base / 17.0) * 1.0e8
        + np.cos(base / 11.0) * 1.0e-6
        + (np.arange(base.size, dtype=np.float64) % 7.0)
    )

    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    expected = {
        (window, window_len): phase_proc.smooth_and_trim(
            x, window_len=window_len, window=window
        )
        for window, window_len in [
            ("flat", 11),
            ("hanning", 12),
            ("hamming", 13),
            ("bartlett", 14),
            ("blackman", 15),
            ("sg_smooth", 5),
        ]
    }

    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        actual = {
            key: phase_proc.smooth_and_trim(x, window_len=key[1], window=key[0])
            for key in expected
        }

    assert warning_records == []
    for key, expected_values in expected.items():
        np.testing.assert_array_equal(actual[key], expected_values)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("x", "weights", "window_len", "match"),
    [
        (
            np.array([1.0, 2.0], dtype=np.float64),
            np.ones(3, dtype=np.float64) / 3.0,
            3,
            "x length",
        ),
        (
            np.array([1.0, np.nan, 3.0], dtype=np.float64),
            np.ones(3, dtype=np.float64) / 3.0,
            3,
            "x must be finite",
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([0.5, np.inf, 0.5], dtype=np.float64),
            3,
            "weights must be finite",
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.ones(2, dtype=np.float64) / 2.0,
            3,
            "weights length",
        ),
    ],
)
def test_real_rust_smooth_and_trim_rejects_unsafe_direct_inputs(
    x, weights, window_len, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._phase_proc_smooth_and_trim_f64(x, weights, window_len)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    ncp, rhv, phidp = _system_phase_inputs()

    expected = _fallback_det_sys_phase(ncp, rhv, phidp, 2, 0.4, 0.6, monkeypatch)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = phase_proc._det_sys_phase(ncp, rhv, phidp, 2, 0.4, 0.6)

    assert isinstance(actual, np.float64)
    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_returns_none_when_no_good_radials(monkeypatch):
    import pyart._rust as rust

    ncp, rhv, phidp = _system_phase_inputs()
    ncp[:] = 0.1

    expected = _fallback_det_sys_phase(ncp, rhv, phidp, 2, 0.4, 0.6, monkeypatch)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = phase_proc._det_sys_phase(ncp, rhv, phidp, 2, 0.4, 0.6)

    assert actual is expected is None


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_gf_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    _, _, phidp = _system_phase_inputs()
    radar_meteo = np.ones(phidp.shape, dtype=np.bool_)
    radar_meteo[0, 26:] = False
    radar_meteo[1, :] = False

    expected = _fallback_det_sys_phase_gf(phidp, 2, radar_meteo, monkeypatch)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = phase_proc._det_sys_phase_gf(phidp, 2, radar_meteo)

    assert isinstance(actual, np.float64)
    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("ncp", "rhv", "phidp", "weights", "last_ray_idx", "ncp_lev", "rhv_lev", "match"),
    [
        (
            np.ones((2, 4), dtype=np.float64),
            np.ones((2, 3), dtype=np.float64),
            np.ones((2, 4), dtype=np.float64),
            np.hanning(9) / np.hanning(9).sum(),
            1,
            0.4,
            0.6,
            "same shape",
        ),
        (
            np.ones((2, 4), dtype=np.float64),
            np.ones((2, 4), dtype=np.float64),
            np.ones((2, 4), dtype=np.float64),
            np.hanning(9) / np.hanning(9).sum(),
            2,
            0.4,
            0.6,
            "last_ray_idx",
        ),
        (
            np.array([[np.nan]], dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            np.hanning(9) / np.hanning(9).sum(),
            0,
            0.4,
            0.6,
            "ncp and rhv",
        ),
    ],
)
def test_real_rust_det_sys_phase_rejects_unsafe_direct_inputs(
    ncp, rhv, phidp, weights, last_ray_idx, ncp_lev, rhv_lev, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._phase_proc_det_sys_phase_dense(
            ncp, rhv, phidp, weights, last_ray_idx, ncp_lev, rhv_lev
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_direct_rejects_nonfinite_thresholds():
    import pyart._rust as rust

    ncp, rhv, phidp = _system_phase_inputs()

    with pytest.raises(ValueError, match="thresholds must be finite"):
        rust._phase_proc_det_sys_phase_dense(
            ncp,
            rhv,
            phidp,
            np.hanning(9) / np.hanning(9).sum(),
            2,
            np.nan,
            0.6,
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_helper_falls_back_for_nonfinite_threshold(
    monkeypatch,
):
    import pyart._rust as rust

    ncp, rhv, phidp = _system_phase_inputs()
    expected = _fallback_det_sys_phase(ncp, rhv, phidp, 2, np.nan, 0.6, monkeypatch)
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = phase_proc._det_sys_phase(ncp, rhv, phidp, 2, np.nan, 0.6)

    assert actual is expected is None


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_gf_direct_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    _, _, phidp = _system_phase_inputs()
    radar_meteo = np.zeros(phidp.shape, dtype=np.bool_)
    radar_meteo[0, :26] = True
    radar_meteo[2, :] = True

    expected = _fallback_det_sys_phase_gf(phidp, 2, radar_meteo, monkeypatch)
    direct = rust._phase_proc_det_sys_phase_gf_dense(
        phidp, radar_meteo, np.hanning(9) / np.hanning(9).sum(), 2
    )

    assert isinstance(direct, np.float64)
    assert direct == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_det_sys_phase_gf_direct_returns_none_when_no_good_radials():
    import pyart._rust as rust

    _, _, phidp = _system_phase_inputs()
    radar_meteo = np.zeros(phidp.shape, dtype=np.bool_)
    radar_meteo[:, :25] = True

    direct = rust._phase_proc_det_sys_phase_gf_dense(
        phidp, radar_meteo, np.hanning(9) / np.hanning(9).sum(), 2
    )

    assert direct is None


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("phidp", "radar_meteo", "weights", "last_ray_idx", "match"),
    [
        (
            np.ones((2, 4), dtype=np.float64),
            np.ones((2, 3), dtype=np.bool_),
            np.hanning(9) / np.hanning(9).sum(),
            1,
            "same shape",
        ),
        (
            np.ones((2, 4), dtype=np.float64),
            np.ones((2, 4), dtype=np.bool_),
            np.hanning(9) / np.hanning(9).sum(),
            2,
            "last_ray_idx",
        ),
        (
            np.array([[np.nan]], dtype=np.float64),
            np.ones((1, 1), dtype=np.bool_),
            np.hanning(9) / np.hanning(9).sum(),
            0,
            "phidp must be finite",
        ),
        (
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.bool_),
            np.ones(8, dtype=np.float64),
            0,
            "weights length",
        ),
    ],
)
def test_real_rust_det_sys_phase_gf_rejects_unsafe_direct_inputs(
    phidp, radar_meteo, weights, last_ray_idx, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._phase_proc_det_sys_phase_gf_dense(
            phidp, radar_meteo, weights, last_ray_idx
        )
