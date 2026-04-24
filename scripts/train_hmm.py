"""Train a reference-specific HMM from same-piece recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm_alignment.data import select_reference_and_queries
from hmm_alignment.features import extract_chroma
from hmm_alignment.training import train_reference_hmm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", default="Chopin_Op030No2")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-train-queries", type=int, default=5)
    parser.add_argument("--max-jump", type=int, default=4)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--output", default="artifacts/reference_hmm.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path, query_paths = select_reference_and_queries(
        args.piece,
        reference_index=args.reference_index,
        data_root=args.data_root,
        max_queries=args.max_train_queries,
    )
    if not query_paths:
        raise SystemExit("Need at least one query recording to train.")

    print(f"Reference: {reference_path}")
    print(f"Training queries: {len(query_paths)}")
    reference_features = extract_chroma(
        reference_path,
        sr=args.sr,
        hop_length=args.hop_length,
    )
    training_features = [
        extract_chroma(path, sr=args.sr, hop_length=args.hop_length)
        for path in query_paths
    ]
    model = train_reference_hmm(
        reference_features,
        training_features,
        max_jump=args.max_jump,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved model: {output}")
    print(f"Jump probabilities: {model.jump_probs}")
    print(f"Covariance: {model.covariance}")


if __name__ == "__main__":
    main()
