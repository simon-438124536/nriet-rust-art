import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import vad  # noqa: E402


def _fallback_vad_calculation_m(velocity_field, azimuth, elevation, monkeypatch):
    monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
    return vad._vad_calculation_m(velocity_field, azimuth, elevation)


def test_vad_calculation_m_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(velocity_field, sin_az, cos_az, elevation_scale):
        calls.append(
            (
                velocity_field.dtype,
                velocity_field.shape,
                sin_az.dtype,
                sin_az.shape,
                cos_az.dtype,
                cos_az.shape,
                elevation_scale,
            )
        )
        return (
            np.array([11.0, 12.0], dtype=np.float64),
            np.array([21.0, 22.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_vad_calculation_m_dense" else None,
    )

    velocity_field = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64
    )
    azimuth = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)

    speed, angle = vad._vad_calculation_m(velocity_field, azimuth, 0.0)

    np.testing.assert_array_equal(speed, np.array([11.0, 12.0], dtype=np.float64))
    np.testing.assert_array_equal(angle, np.array([21.0, 22.0], dtype=np.float64))
    assert calls == [
        (np.float64, (4, 2), np.float64, (4,), np.float64, (4,), 1.0)
    ]


def test_vad_calculation_m_source_dispatch_matches_python_fallback(monkeypatch):
    velocity_field = np.array(
        [
            [-1.1, 2.2, 9.5],
            [3.3, -4.4, -8.5],
            [5.5, 6.6, 7.5],
            [-7.7, 8.8, -6.5],
        ],
        dtype=np.float64,
    )
    azimuth = np.array([0.0, 82.5, 181.0, 274.0], dtype=np.float64)
    expected = _fallback_vad_calculation_m(velocity_field, azimuth, 3.5, monkeypatch)

    def rust_kernel(velocity_field, sin_az, cos_az, elevation_scale):
        import pyart._rust as rust

        return rust._vad_calculation_m_dense(
            velocity_field, sin_az, cos_az, elevation_scale
        )

    try:
        import pyart._rust  # noqa: F401
    except ImportError:
        pytest.skip("pyart._rust is not available in source-only mode")

    monkeypatch.setattr(
        vad,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_vad_calculation_m_dense" else None,
    )
    actual = vad._vad_calculation_m(velocity_field, azimuth, 3.5)

    assert actual[0].dtype == expected[0].dtype == np.float64
    assert actual[1].dtype == expected[1].dtype == np.float64
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("velocity_field", "azimuth", "elevation"),
    [
        (
            np.ma.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                dtype=np.float64,
            ),
            np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64),
            0.0,
        ),
        (
            np.array(
                [[1.0, np.nan], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                dtype=np.float64,
            ),
            np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64),
            0.0,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
            np.array([0.0, 90.0, 180.0], dtype=np.float64),
            0.0,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float32),
            0.0,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            0.0,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 90.0], dtype=np.float64),
            np.inf,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([45.0, 45.0], dtype=np.float64),
            0.0,
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64).T,
            np.array([0.0, 90.0], dtype=np.float64),
            0.0,
        ),
    ],
)
def test_vad_calculation_m_keeps_python_path_for_unsupported_inputs(
    monkeypatch, velocity_field, azimuth, elevation
):
    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("unsupported Michelson VAD input should use fallback")

        return kernel

    monkeypatch.setattr(vad, "_rust_kernel", fail_if_called)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            actual = vad._vad_calculation_m(velocity_field, azimuth, elevation)
        except Exception as actual_error:
            monkeypatch.setattr(vad, "_rust_kernel", lambda _name: None)
            with pytest.raises(type(actual_error)):
                vad._vad_calculation_m(velocity_field, azimuth, elevation)
        else:
            expected = _fallback_vad_calculation_m(
                velocity_field, azimuth, elevation, monkeypatch
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_vad_calculation_m_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    velocity_field = np.array(
        [
            [-1.1, 2.2, 9.5],
            [3.3, -4.4, -8.5],
            [5.5, 6.6, 7.5],
            [-7.7, 8.8, -6.5],
        ],
        dtype=np.float64,
    )
    azimuth = np.array([0.0, 82.5, 181.0, 274.0], dtype=np.float64)

    expected = _fallback_vad_calculation_m(velocity_field, azimuth, 3.5, monkeypatch)
    monkeypatch.setattr(vad, "_rust_kernel", lambda name: getattr(rust, name, None))
    actual = vad._vad_calculation_m(velocity_field, azimuth, 3.5)

    assert actual[0].dtype == expected[0].dtype == np.float64
    assert actual[1].dtype == expected[1].dtype == np.float64
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-12)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception checks are verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("velocity_field", "sin_az", "cos_az", "elevation_scale", "match"),
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            1.0,
            "positive even number of rays",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1.0,
            "match the number of velocity rays",
        ),
        (
            np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "velocity_field must be finite",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "sin_az and cos_az must be finite",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            np.inf,
            "elevation_scale must be finite",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            1.0,
            "non-singular",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64).T,
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "C-contiguous",
        ),
        (
            np.ma.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                mask=[[False, True], [False, False], [False, False], [False, False]],
                dtype=np.float64,
            ),
            np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64),
            1.0,
            "mask-free",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "2D float64",
        ),
        (
            np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "2D float64",
        ),
        (
            np.array([[object(), object()], [object(), object()]], dtype=object),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "2D float64",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.ma.array([0.0, 1.0], mask=[False, True], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "mask-free",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            "1D float64",
        ),
    ],
)
def test_real_rust_vad_calculation_m_rejects_unsafe_direct_inputs(
    velocity_field, sin_az, cos_az, elevation_scale, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._vad_calculation_m_dense(velocity_field, sin_az, cos_az, elevation_scale)
