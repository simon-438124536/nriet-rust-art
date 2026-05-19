import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import phase_proc  # noqa: E402


def _fallback_fzl_index(fzl, ranges, elevation, radar_height, monkeypatch):
    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    return phase_proc.fzl_index(fzl, ranges, elevation, radar_height)


def test_fzl_index_python_fallback_return_types(monkeypatch):
    ranges = np.array([0.0, 100.0, 1000.0], dtype=np.float64)

    below = _fallback_fzl_index(1000.0, ranges, 0.5, 100.0, monkeypatch)
    minimum = _fallback_fzl_index(0.0, ranges, 0.5, 100.0, monkeypatch)

    assert isinstance(below, np.integer)
    assert below == np.int64(2)
    assert type(minimum) is int
    assert minimum == 6


def test_fzl_index_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(ranges, fzl, elevation, radar_height):
        calls.append((ranges.dtype, ranges.shape, fzl, elevation, radar_height))
        return np.int64(7)

    monkeypatch.setattr(
        phase_proc,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_phase_proc_fzl_index_dense" else None,
    )

    actual = phase_proc.fzl_index(
        np.float64(1000.0),
        np.array([0.0, 100.0, 1000.0], dtype=np.float64),
        np.float64(0.5),
        np.array([100.0], dtype=np.float64),
    )

    assert actual == np.int64(7)
    assert calls == [(np.float64, (3,), 1000.0, 0.5, 100.0)]


@pytest.mark.parametrize(
    ("fzl", "ranges", "elevation", "radar_height"),
    [
        (1000.0, np.array([0.0, 100.0], dtype=np.float32), 0.5, 100.0),
        (1000.0, np.array([0.0, 100.0], dtype=np.float64)[::-1], 0.5, 100.0),
        (1000.0, np.ma.array([0.0, 100.0], dtype=np.float64), 0.5, 100.0),
        (1000.0, np.array([0.0, np.nan], dtype=np.float64), 0.5, 100.0),
        (np.nan, np.array([0.0, 100.0], dtype=np.float64), 0.5, 100.0),
        (1000.0, np.array([0.0, 100.0], dtype=np.float64), np.inf, 100.0),
        (1000.0, np.array([0.0, 100.0], dtype=np.float64), 0.5, np.nan),
        (
            1000.0,
            np.array([0.0, 100.0], dtype=np.float64),
            np.array([0.5, 0.6], dtype=np.float64),
            100.0,
        ),
        (
            1000.0,
            np.array([0.0, 100.0], dtype=np.float64),
            0.5,
            np.array([100.0, 101.0], dtype=np.float64),
        ),
        (True, np.array([0.0, 100.0], dtype=np.float64), 0.5, 100.0),
    ],
)
def test_fzl_index_keeps_python_path_for_unsupported_inputs(
    monkeypatch, fzl, ranges, elevation, radar_height
):
    def fail_if_called(name):
        if name != "_phase_proc_fzl_index_dense":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported fzl_index input should use fallback")

        return kernel

    monkeypatch.setattr(phase_proc, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"):
        try:
            actual = phase_proc.fzl_index(fzl, ranges, elevation, radar_height)
        except Exception as actual_error:
            monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                phase_proc.fzl_index(fzl, ranges, elevation, radar_height)
        else:
            expected = _fallback_fzl_index(
                fzl, ranges, elevation, radar_height, monkeypatch
            )
            assert actual == expected
            assert type(actual) is type(expected)


@pytest.mark.parametrize(
    ("fzl", "ranges", "elevation", "radar_height"),
    [
        (1000.0, np.array([], dtype=np.float64), 0.5, 100.0),
        (0.0, np.array([0.0, 100.0], dtype=np.float64), 0.5, 100.0),
    ],
)
def test_fzl_index_minimum_window_edges(monkeypatch, fzl, ranges, elevation, radar_height):
    actual = _fallback_fzl_index(fzl, ranges, elevation, radar_height, monkeypatch)

    assert type(actual) is int
    assert actual == 6


def test_fzl_index_equality_without_below_gate_preserves_value_error(monkeypatch):
    ranges = np.array([0.0], dtype=np.float64)
    radar_height = 100.0

    with pytest.raises(ValueError, match="zero-size array"):
        _fallback_fzl_index(radar_height, ranges, 0.0, radar_height, monkeypatch)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("fzl", "ranges", "elevation", "radar_height"),
    [
        (1000.0, np.array([0.0, 100.0, 1000.0], dtype=np.float64), 0.5, 100.0),
        (500.0, np.linspace(0.0, 20_000.0, 25, dtype=np.float64), 1.2, 50.0),
        (1000.0, np.array([], dtype=np.float64), 0.5, 100.0),
        (0.0, np.array([0.0, 100.0], dtype=np.float64), 0.5, 100.0),
        (100.0, np.array([0.0], dtype=np.float64), 0.0, 100.0),
    ],
)
def test_real_rust_fzl_index_matches_python_fallback(
    monkeypatch, fzl, ranges, elevation, radar_height
):
    import pyart._rust as rust

    try:
        expected = _fallback_fzl_index(fzl, ranges, elevation, radar_height, monkeypatch)
    except Exception as expected_error:
        monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
        with pytest.raises(type(expected_error), match="zero-size array"):
            phase_proc.fzl_index(fzl, ranges, elevation, radar_height)
    else:
        monkeypatch.setattr(phase_proc, "_rust_kernel", lambda name: getattr(rust, name, None))
        actual = phase_proc.fzl_index(fzl, ranges, elevation, radar_height)
        assert actual == expected
        assert type(actual) is type(expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("ranges", "fzl", "elevation", "radar_height", "match"),
    [
        (
            np.array([0.0, np.nan], dtype=np.float64),
            1000.0,
            0.5,
            100.0,
            "ranges must be finite",
        ),
        (
            np.array([0.0, 100.0], dtype=np.float64),
            np.inf,
            0.5,
            100.0,
            "must be finite",
        ),
        (
            np.array([0.0, 100.0], dtype=np.float64),
            1000.0,
            np.nan,
            100.0,
            "must be finite",
        ),
        (
            np.array([0.0, 100.0], dtype=np.float64),
            1000.0,
            0.5,
            np.inf,
            "must be finite",
        ),
    ],
)
def test_real_rust_fzl_index_rejects_unsafe_direct_inputs(
    ranges, fzl, elevation, radar_height, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._phase_proc_fzl_index_dense(ranges, fzl, elevation, radar_height)
