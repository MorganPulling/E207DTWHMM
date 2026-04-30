"""Run the fixed train70 benchmark.

Usage:
    python -m scripts.run_benchmark hmm_train70
    python -m scripts.run_benchmark hmm_train70 --mode least_recordings
    python -m scripts.run_benchmark offline_dtw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.hmm_benchmark import BENCHMARK_MODES, METHODS, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--mode", choices=sorted(BENCHMARK_MODES), default="paper_test")
    args = parser.parse_args(argv)

    metrics_frame = run_benchmark(args.method, mode=args.mode)
    print(
        f"Completed {args.method} {args.mode} benchmark with "
        f"{len(metrics_frame)} pair(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
