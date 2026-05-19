"""Microbenchmark Python fallback vs Rust fast path for selected native slices."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import phase_proc  # noqa: E402


def _bench(fn, repeats: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(timings),
        "p95_s": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
    }


def bench_smooth_and_trim_scan(repeats: int, warmup: int) -> dict[str, object]:
    x = np.random.default_rng(0).standard_normal((64, 2048)).astype(np.float64)

    def python_path():
        phase_proc._rust_kernel = lambda _name: None  # type: ignore[method-assign]
        return phase_proc.smooth_and_trim_scan(x, window_len=11, window="hanning")

    def rust_path():
        import pyart._rust as rust

        phase_proc._rust_kernel = lambda name: getattr(rust, name, None)  # type: ignore[method-assign]
        return phase_proc.smooth_and_trim_scan(x, window_len=11, window="hanning")

    return {
        "slice": "smooth_and_trim_scan",
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "python": _bench(python_path, repeats, warmup),
        "rust": _bench(rust_path, repeats, warmup),
    }


SLICES = {
    "smooth_and_trim_scan": bench_smooth_and_trim_scan,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/benchmarks"),
    )
    args = parser.parse_args()

    result = SLICES[args.slice](args.repeats, args.warmup)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.slice}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
