import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import despeckle  # noqa: E402


def _fallback_check_for_360(az, delta, monkeypatch):
    monkeypatch.setattr(despeckle, "_rust_kernel", lambda _name: None)
    return despeckle._check_for_360(az, delta)


@pytest.mark.parametrize(
    ("az", "expected"),
    [
        (np.array([0.0, 90.0, 180.0, 270.0, 359.0], dtype=np.float64), True),
        (np.array([10.0, 20.0, 30.0], dtype=np.float64), False),
        (np.array([0.0], dtype=np.float64), False),
    ],
)
def test_check_for_360_python_fallback_matches_oracle_examples(
    monkeypatch, az, expected
):
    actual = _fallback_check_for_360(az, 5.0, monkeypatch)

    assert type(actual) is bool
    assert actual is expected


def test_check_for_360_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(az, delta):
        calls.append((az.dtype, az.shape, delta))
        return True

    monkeypatch.setattr(
        despeckle,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_despeckle_check_for_360" else None,
    )
    az = np.array([0.0, 90.0, 180.0, 270.0, 359.0], dtype=np.float64)

    assert despeckle._check_for_360(az, np.float64(5.0)) is True
    assert calls == [(np.float64, (5,), 5.0)]


@pytest.mark.parametrize(
    ("az", "delta"),
    [
        (np.array([0.0, 359.0], dtype=np.float32), 5.0),
        (np.array([0, 359], dtype=np.int32), 5.0),
        (np.array([0.0, np.nan, 359.0], dtype=np.float64), 5.0),
        (np.array([0.0, 359.0], dtype=np.float64)[::-1], 5.0),
        (np.ma.array([0.0, 359.0], dtype=np.float64), 5.0),
        ([0.0, 359.0], 5.0),
        (np.array(0.0, dtype=np.float64), 5.0),
        (np.array([[0.0, 359.0]], dtype=np.float64), 5.0),
        (np.array([0.0, 359.0], dtype=np.complex128), 5.0),
        (np.array([0.0, 359.0], dtype=object), 5.0),
        (np.array([0.0, 359.0], dtype=np.float64), "5.0"),
        (np.array([0.0, 359.0], dtype=np.float64), None),
        (np.array([0.0, 359.0], dtype=np.float64), [5.0]),
        (np.array([0.0, 359.0], dtype=np.float64), np.array(5.0)),
        (np.array([0.0, 359.0], dtype=np.float64), np.array([5.0])),
        (np.array([0.0, 359.0], dtype=np.float64), 5.0 + 0j),
        (np.array([0.0, 359.0], dtype=np.float64), np.nan),
    ],
)
def test_check_for_360_keeps_python_path_for_unsupported_inputs(
    monkeypatch, az, delta
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported check-for-360 input should use fallback")

        return kernel

    monkeypatch.setattr(despeckle, "_rust_kernel", fail_if_called)

    try:
        actual = despeckle._check_for_360(az, delta)
    except Exception as actual_error:
        monkeypatch.setattr(despeckle, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)):
            despeckle._check_for_360(az, delta)
    else:
        expected = _fallback_check_for_360(az, delta, monkeypatch)
        assert actual == expected


def test_check_for_360_preserves_empty_array_index_error(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("empty input should use Python fallback")

        return kernel

    monkeypatch.setattr(despeckle, "_rust_kernel", fail_if_called)

    with pytest.raises(IndexError):
        despeckle._check_for_360(np.array([], dtype=np.float64), 5.0)


def test_check_for_360_keeps_python_path_for_inf_warning(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("inf inputs should use Python fallback")

        return kernel

    monkeypatch.setattr(despeckle, "_rust_kernel", fail_if_called)

    with pytest.warns(RuntimeWarning):
        actual = despeckle._check_for_360(
            np.array([0.0, 90.0, np.inf], dtype=np.float64), 5.0
        )

    assert actual is False


def test_check_for_360_keeps_python_path_for_infinite_delta_warning(monkeypatch):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("infinite delta should use Python fallback")

        return kernel

    monkeypatch.setattr(despeckle, "_rust_kernel", fail_if_called)

    with pytest.warns(RuntimeWarning):
        actual = despeckle._check_for_360(
            np.array([0.0, 90.0, 180.0, 270.0, 359.0], dtype=np.float64),
            np.inf,
        )

    assert actual is False


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_check_for_360_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    az = np.array([0.0, 90.0, 180.0, 270.0, 359.0], dtype=np.float64)
    expected = _fallback_check_for_360(az, 5.0, monkeypatch)
    monkeypatch.setattr(despeckle, "_rust_kernel", lambda name: getattr(rust, name, None))

    assert despeckle._check_for_360(az, 5.0) is expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
def test_real_rust_check_for_360_rejects_empty_direct_call():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="az"):
        rust._despeckle_check_for_360(np.array([], dtype=np.float64), 5.0)
