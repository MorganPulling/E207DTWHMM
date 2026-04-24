"""Train a reference-specific HMM from same-piece recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm_alignment.data import list_recordings, select_reference_and_queries
from hmm_alignment.features import extract_chroma
from hmm_alignment.training import train_reference_hmm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", default="Chopin_Op030No2")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument(
        "--train-recording-count",
        type=int,
        default=None,
        help=(
            "Restrict reference and query recordings to the first N sorted "
            "recordings for the piece. Useful for fixed train/test splits."
        ),
    )
    parser.add_argument(
        "--train-query-count",
        type=int,
        default=None,
        help=(
            "Use the first N sorted recordings as training queries while still "
            "allowing any recording in the piece to be the reference."
        ),
    )
    parser.add_argument("--max-jump", type=int, default=4)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--output", default="artifacts/reference_hmm.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_query_count is not None and args.train_recording_count is not None:
        raise SystemExit("Use only one of --train-query-count or --train-recording-count.")
    if args.train_query_count is not None:
        recordings = list_recordings(args.piece, data_root=args.data_root)
        if not recordings:
            raise SystemExit(f"No WAV recordings found for piece {args.piece!r}.")
        if args.reference_index < 0 or args.reference_index >= len(recordings):
            raise SystemExit(
                f"--reference-index must be in [0, {len(recordings) - 1}]."
            )
        train_recordings = recordings[: args.train_query_count]
        if not train_recordings:
            raise SystemExit("--train-query-count must select at least one query recording.")
        reference_path = recordings[args.reference_index]
        query_paths = [path for path in train_recordings if path != reference_path]
        if args.max_train_queries is not None:
            query_paths = query_paths[: args.max_train_queries]
    elif args.train_recording_count is None:
        reference_path, query_paths = select_reference_and_queries(
            args.piece,
            reference_index=args.reference_index,
            data_root=args.data_root,
            max_queries=args.max_train_queries,
        )
    else:
        recordings = list_recordings(args.piece, data_root=args.data_root)
        train_recordings = recordings[: args.train_recording_count]
        if len(train_recordings) < 2:
            raise SystemExit("--train-recording-count must select at least two recordings.")
        if args.reference_index < 0 or args.reference_index >= len(train_recordings):
            raise SystemExit(
                "--reference-index must be inside the selected training recording subset."
            )
        reference_path = train_recordings[args.reference_index]
        query_paths = [
            path for idx, path in enumerate(train_recordings) if idx != args.reference_index
        ]
        if args.max_train_queries is not None:
            query_paths = query_paths[: args.max_train_queries]
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
