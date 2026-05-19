import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import phase_proc  # noqa: E402


def _make_radar(ranges, sweep=0, nray=2):
    return SimpleNamespace(
        range={"data": np.asarray(ranges, dtype=np.float64)},
        fixed_angle={"data": np.asarray([0.5], dtype=np.float64)},
        altitude={"data": np.asarray([100.0], dtype=np.float64)},
        sweep_start_ray_index={"data": np.asarray([0], dtype=np.int32)},
        sweep_end_ray_index={"data": np.asarray([nray - 1], dtype=np.int32)},
        nrays=nray,
        ngates=len(ranges),
    )


def test_det_process_range_matches_fzl_index_and_sweep_bounds(monkeypatch):
    ranges = np.linspace(0.0, 1000.0, 8, dtype=np.float64)
    radar = _make_radar(ranges, nray=3)

    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    gate_end, ray_start, ray_end = phase_proc.det_process_range(
        radar, sweep=0, fzl=500.0, doc=2
    )

    expected_gate_end = min(phase_proc.fzl_index(500.0, ranges, 0.5, 100.0), len(ranges) - 2)
    assert gate_end == expected_gate_end
    assert ray_start == 0
    assert ray_end == 3


def test_det_process_range_doc_none_uses_full_range(monkeypatch):
    ranges = np.linspace(0.0, 1000.0, 6, dtype=np.float64)
    radar = _make_radar(ranges, nray=2)

    monkeypatch.setattr(phase_proc, "_rust_kernel", lambda _name: None)
    gate_end, _, _ = phase_proc.det_process_range(radar, sweep=0, fzl=500.0, doc=None)

    assert gate_end == min(phase_proc.fzl_index(500.0, ranges, 0.5, 100.0), len(ranges))
