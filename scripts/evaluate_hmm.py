"""Evaluate HMM filtering against offline-DTW pseudo-labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm_alignment.data import select_reference_and_queries
from hmm_alignment.dtw import dtw_pseudo_labels
from hmm_alignment.evaluation import frame_error_metrics
from hmm_alignment.features import extract_chroma
from hmm_alignment.hmm import ReferenceHMM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/reference_hmm.npz")
    parser.add_argument("--piece", default="Chopin_Op030No2")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=1)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--plot", default="artifacts/evaluation_path.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path, query_paths = select_reference_and_queries(
        args.piece,
        reference_index=args.reference_index,
        data_root=args.data_root,
    )
    if args.query_index < 0 or args.query_index >= len(query_paths):
        raise SystemExit(f"--query-index must be in [0, {len(query_paths) - 1}]")
    query_path = query_paths[args.query_index]

    model = ReferenceHMM.load(args.model)
    reference_features = extract_chroma(
        reference_path,
        sr=args.sr,
        hop_length=args.hop_length,
    )
    query_features = extract_chroma(query_path, sr=args.sr, hop_length=args.hop_length)
    target = dtw_pseudo_labels(query_features, reference_features)
    result = model.filter(query_features)
    metrics = frame_error_metrics(result.map_path, target)

    print(f"Reference: {reference_path}")
    print(f"Query: {query_path}")
    print(metrics)

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(target, label="Offline DTW pseudo-label", linewidth=1.5)
    plt.plot(result.map_path, label="HMM filtered MAP", linewidth=1.0)
    plt.plot(result.mean_path, label="HMM posterior mean", linewidth=1.0)
    plt.xlabel("Query frame")
    plt.ylabel("Reference frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
