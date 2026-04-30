"""Run the fixed train70 benchmark.

Usage:
    python -m scripts.run_benchmark hmm_train70
    python -m scripts.run_benchmark offline_dtw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.hmm_benchmark import METHODS, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    args = parser.parse_args(argv)

    metrics_frame = run_benchmark(args.method)
    print(
        f"Completed {args.method} paper_test benchmark with "
        f"{len(metrics_frame)} pair(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
