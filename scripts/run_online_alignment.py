"""Run causal HMM filtering on a query recording."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm_alignment.features import extract_chroma
from hmm_alignment.hmm import ReferenceHMM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/reference_hmm.npz")
    parser.add_argument("--query", required=True)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--output", default="artifacts/alignment_path.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ReferenceHMM.load(args.model)
    query_features = extract_chroma(args.query, sr=args.sr, hop_length=args.hop_length)
    result = model.filter(query_features, return_posteriors=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, map_path=result.map_path, mean_path=result.mean_path)
    print(f"Saved alignment: {output}")
    print(f"Frames aligned: {len(result.map_path)}")
    print(f"Final MAP reference frame: {result.map_path[-1]}")


if __name__ == "__main__":
    main()
