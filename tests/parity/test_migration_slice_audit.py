"""Lightweight audit that core migration areas expose Rust dispatch hooks."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYART = REPO_ROOT / "python" / "pyart"

EXPECTED_RUST_HOOKS = {
    "correct/phase_proc.py": "_phase_proc_smooth_and_trim_scan_f64",
    "correct/attenuation.py": "_attenuation_end_gate_from_excluded_mask",
    "map/gate_mapper.py": "_gate_mapper_apply_field_f64",
    "retrieve/_kdp_proc.py": "lowpass_maesaka_term",
    "filters/gatefilter.py": "_rust_kernel",
    "io/_sigmetfile.py": "_rust_kernel",
}


@pytest.mark.parametrize(("relative_path", "needle"), sorted(EXPECTED_RUST_HOOKS.items()))
def test_expected_modules_reference_private_rust_helpers(relative_path, needle):
    source = (PYART / relative_path).read_text(encoding="utf-8")
    assert needle in source
