import os

import numpy as np
import pytest

from pyart.testing import make_empty_ppi_radar, make_empty_rhi_radar
from pyart.util import columnsect


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    for name in (
        "_columnsect_nearest_ray_index_f64",
        "_columnsect_get_column_rays_rhi_f64",
    ):
        if not hasattr(rust, name):
            pytest.skip(f"pyart._rust has no {name} kernel")
    return rust


def _fallback_get_column_rays(radar, azimuth, monkeypatch):
    monkeypatch.setattr(columnsect, "_rust_kernel", lambda _name: None)
    return columnsect.get_column_rays(radar, azimuth)


def _ppi_radar():
    radar = make_empty_ppi_radar(1, 4, 2)
    radar.azimuth["data"] = np.array(
        [8.0, 10.0, 12.0, 14.0, 20.0, np.nan, 21.0, 22.0],
        dtype=np.float64,
    )
    return radar


def _rhi_radar():
    radar = make_empty_rhi_radar(1, 5, 2)
    radar.azimuth["data"] = np.array(
        [9.1, 10.0, 10.99, 11.01, 10.0, 9.2, 10.5, 11.0, np.nan, 10.0],
        dtype=np.float64,
    )
    return radar


def test_get_column_rays_ppi_python_fallback_reference(monkeypatch):
    radar = _ppi_radar()

    actual = _fallback_get_column_rays(radar, 10.0, monkeypatch)

    assert actual == [1, 5]


def test_get_column_rays_rhi_python_fallback_reference(monkeypatch):
    radar = _rhi_radar()

    actual = _fallback_get_column_rays(radar, 10.0, monkeypatch)

    assert actual == [0, 1, 2, 5, 6]


def test_get_column_rays_keeps_public_azimuth_validation(monkeypatch):
    radar = _ppi_radar()
    monkeypatch.setattr(
        columnsect,
        "_rust_kernel",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("invalid public azimuth reached Rust")
        ),
    )

    with pytest.raises(ValueError, match="azimuth not valid"):
        columnsect.get_column_rays(radar, 0.0)
    with pytest.raises(ValueError, match="azimuth not valid"):
        columnsect.get_column_rays(radar, 360.0)
    with pytest.raises(TypeError, match="radar azimuth type not valid"):
        columnsect.get_column_rays(radar, np.int64(10))


def test_get_column_rays_ppi_dispatches_float64_sweeps_to_private_rust(monkeypatch):
    radar = _ppi_radar()
    calls = []

    def kernel(sweep_azi, azimuth):
        calls.append((sweep_azi.dtype, sweep_azi.shape, azimuth))
        return 1

    monkeypatch.setattr(
        columnsect,
        "_rust_kernel",
        lambda name: kernel if name == "_columnsect_nearest_ray_index_f64" else None,
    )

    actual = columnsect.get_column_rays(radar, 10.0)

    assert calls == [
        (np.dtype(np.float64), (4,), 10.0),
        (np.dtype(np.float64), (4,), 10.0),
    ]
    assert actual == [1, 5]


def test_get_column_rays_rhi_dispatches_float64_volume_to_private_rust(monkeypatch):
    radar = _rhi_radar()
    expected = np.array([0, 1, 2, 5, 6], dtype=np.int64)
    calls = []

    def kernel(azimuths, starts, ends, azimuth):
        calls.append((azimuths.dtype, azimuths.shape, starts.dtype, ends.dtype, azimuth))
        return expected.copy()

    monkeypatch.setattr(
        columnsect,
        "_rust_kernel",
        lambda name: kernel if name == "_columnsect_get_column_rays_rhi_f64" else None,
    )

    actual = columnsect.get_column_rays(radar, 10.0)

    assert calls == [(np.dtype(np.float64), (10,), np.dtype(np.int64), np.dtype(np.int64), 10.0)]
    assert actual == expected.tolist()


def test_get_column_rays_rust_runtime_error_keeps_python_path(monkeypatch):
    radar = _ppi_radar()

    def rust_kernel(name):
        if name != "_columnsect_nearest_ray_index_f64":
            return None

        def fail(*_args):
            raise ValueError("native failure")

        return fail

    monkeypatch.setattr(columnsect, "_rust_kernel", rust_kernel)
    actual = columnsect.get_column_rays(radar, 10.0)
    expected = _fallback_get_column_rays(radar, 10.0, monkeypatch)

    assert actual == expected


@pytest.mark.parametrize(
    "case",
    [
        lambda: (_ppi_radar(), True),
        lambda: (_rhi_radar(), True),
    ],
)
def test_get_column_rays_azimuths_that_rust_rejects_keep_python_path(monkeypatch, case):
    radar, azimuth = case()

    def rust_kernel(name):
        if name.startswith("_columnsect_"):
            raise AssertionError("unsupported azimuth used Rust kernel")
        return None

    monkeypatch.setattr(columnsect, "_rust_kernel", rust_kernel)
    try:
        actual = columnsect.get_column_rays(radar, azimuth)
    except Exception as actual_error:
        expected_radar, expected_azimuth = case()
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_get_column_rays(expected_radar, expected_azimuth, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected_radar, expected_azimuth = case()
        expected = _fallback_get_column_rays(
            expected_radar, expected_azimuth, monkeypatch
        )
        assert actual == expected


def test_get_column_rays_unsupported_arrays_keep_python_path(monkeypatch):
    ppi = _ppi_radar()
    ppi.azimuth["data"] = ppi.azimuth["data"].astype(np.float32)
    rhi = _rhi_radar()
    rhi.azimuth["data"] = rhi.azimuth["data"].astype(np.float32)

    def rust_kernel(name):
        if name.startswith("_columnsect_"):
            raise AssertionError("unsupported column-ray input used Rust kernel")
        return None

    monkeypatch.setattr(columnsect, "_rust_kernel", rust_kernel)

    assert columnsect.get_column_rays(ppi, 10.0) == _fallback_get_column_rays(
        _ppi_radar(), 10.0, monkeypatch
    )
    assert columnsect.get_column_rays(rhi, 10.0) == _fallback_get_column_rays(
        _rhi_radar(), 10.0, monkeypatch
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust column-ray parity",
)
@pytest.mark.parametrize("radar_factory", [_ppi_radar, _rhi_radar])
def test_get_column_rays_real_rust_matches_python_fallback(monkeypatch, radar_factory):
    radar = radar_factory()
    expected = _fallback_get_column_rays(radar, 10.0, monkeypatch)
    monkeypatch.undo()

    actual = columnsect.get_column_rays(radar, 10.0)

    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust column-ray checks",
)
def test_get_column_rays_direct_rust_helpers():
    rust = _rust_or_skip()

    assert rust._columnsect_nearest_ray_index_f64(
        np.array([10.5, 9.5], dtype=np.float64), 10.0
    ) == 0
    assert rust._columnsect_nearest_ray_index_f64(
        np.array([10.5, np.nan, 10.0], dtype=np.float64), 10.0
    ) == 1

    azimuths = np.array(
        [9.1, 10.0, 10.99, 11.01, 10.0, 9.2, 10.5, 11.0, np.nan, 10.0],
        dtype=np.float64,
    )
    starts = np.array([0, 5], dtype=np.int64)
    ends = np.array([4, 9], dtype=np.int64)
    rays = rust._columnsect_get_column_rays_rhi_f64(azimuths, starts, ends, 10.0)
    assert rays.dtype == np.int64
    assert rays.tolist() == [0, 1, 2, 5, 6]

    with pytest.raises(ValueError):
        rust._columnsect_nearest_ray_index_f64(np.array([], dtype=np.float64), 10.0)
    with pytest.raises(ValueError):
        rust._columnsect_nearest_ray_index_f64(
            np.zeros(1024 * 1024 + 1, dtype=np.float64), 10.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_nearest_ray_index_f64(
            np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)[::2], 10.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_get_column_rays_rhi_f64(
            azimuths, np.array([-1], dtype=np.int64), np.array([1], dtype=np.int64), 10.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_get_column_rays_rhi_f64(
            azimuths, np.array([0], dtype=np.int64), np.array([11], dtype=np.int64), 10.0
        )
    with pytest.raises(ValueError):
        rust._columnsect_get_column_rays_rhi_f64(
            azimuths,
            np.array([0, 5], dtype=np.int64),
            np.array([4], dtype=np.int64),
            10.0,
        )
