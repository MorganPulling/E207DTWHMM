"""Fixed train70 held-out benchmark for E207DTWHMM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from hmm_alignment.features import extract_chroma, frames_to_times
from hmm_alignment.hmm import ReferenceHMM
from scripts import data_io, metrics, offline_dtw
from scripts.config import DATA_DIR, DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE, METRICS_DIR
from scripts.models import AlignmentResult, FeatureSequence, Recording, RecordingPair


METHODS = ("hmm_train70", "offline_dtw")
MODEL_DIR = Path("artifacts/train70_models")
TRAIN_PERCENT = 70
MAX_WARP_FACTOR = 2.0


def run_benchmark(method: str) -> pd.DataFrame:
    """Run the fixed held-out benchmark and write paper-style CSV outputs."""

    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}")

    recordings = data_io.discover_recordings(DATA_DIR)
    pairs = _heldout_pairs(recordings)
    if not pairs:
        raise ValueError("No train70 held-out benchmark pairs were found.")

    rows: list[dict[str, object]] = []
    error_frames: list[pd.DataFrame] = []
    feature_cache: dict[Path, FeatureSequence] = {}

    for pair in tqdm(pairs, desc=f"{method}_paper_test"):
        result = _run_pair(pair, method, feature_cache)
        reference_beats = data_io.load_beat_timestamps(pair.reference.beats_path)
        query_beats = data_io.load_beat_timestamps(pair.query.beats_path)
        row = metrics.compute_alignment_metrics(result, reference_beats, query_beats)
        row.update(result.metadata)
        row.update({"piece": pair.piece, "pair_id": pair.pair_id})
        error_frame = metrics.compute_alignment_error_trace(result, reference_beats, query_beats)
        error_frame["piece"] = pair.piece
        error_frame["pair_id"] = pair.pair_id
        rows.append(row)
        error_frames.append(error_frame)

    metrics_frame = pd.DataFrame(rows)
    errors_frame = pd.concat(error_frames, ignore_index=True)
    _write_outputs(method, metrics_frame, errors_frame)
    return metrics_frame


def _heldout_pairs(recordings: list[Recording]) -> list[RecordingPair]:
    grouped: dict[str, list[Recording]] = {}
    for recording in recordings:
        if recording.beats_path is not None:
            grouped.setdefault(recording.piece, []).append(recording)

    pairs: list[RecordingPair] = []
    for piece, piece_recordings in grouped.items():
        ordered = sorted(piece_recordings, key=lambda item: item.audio_path)
        train_count = len(ordered) * TRAIN_PERCENT // 100
        index_by_id = {recording.recording_id: index for index, recording in enumerate(ordered)}
        heldout_queries = ordered[train_count:]
        for reference in ordered:
            reference_index = index_by_id[reference.recording_id]
            if not _model_path(piece, reference_index).exists():
                continue
            for query in heldout_queries:
                if query.recording_id == reference.recording_id:
                    continue
                pair = RecordingPair(piece=piece, reference=reference, query=query)
                if _warp_factor(pair) <= MAX_WARP_FACTOR:
                    pairs.append(pair)

    return sorted(pairs, key=lambda pair: (pair.piece, pair.pair_id))


def _run_pair(
    pair: RecordingPair,
    method: str,
    feature_cache: dict[Path, FeatureSequence],
) -> AlignmentResult:
    reference_features = _features(pair.reference, feature_cache)
    query_features = _features(pair.query, feature_cache)

    if method == "offline_dtw":
        return offline_dtw.run_offline_dtw(reference_features, query_features)

    reference_index = _reference_index(pair.reference)
    model_path = _model_path(pair.piece, reference_index)
    model = ReferenceHMM.load(model_path)
    result = model.filter(query_features.values)
    reference_indices = np.clip(result.map_path, 0, len(reference_features.frame_times) - 1)
    query_indices = np.arange(len(query_features.frame_times), dtype=np.int64)
    return AlignmentResult(
        method_name="hmm_train70",
        reference_id=pair.reference.recording_id,
        query_id=pair.query.recording_id,
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times,
        path=np.column_stack([reference_indices, query_indices]),
        metadata={
            "model_path": str(model_path),
            "train_percent": TRAIN_PERCENT,
            "reference_index": reference_index,
            "decoder": "filter_map",
            "max_jump": model.max_jump,
        },
    )


def _features(recording: Recording, feature_cache: dict[Path, FeatureSequence]) -> FeatureSequence:
    cached = feature_cache.get(recording.audio_path)
    if cached is not None:
        return cached

    values = extract_chroma(
        recording.audio_path,
        sr=DEFAULT_SAMPLE_RATE,
        hop_length=DEFAULT_HOP_LENGTH,
    )
    frame_times = frames_to_times(
        np.arange(values.shape[0]),
        sr=DEFAULT_SAMPLE_RATE,
        hop_length=DEFAULT_HOP_LENGTH,
    )
    feature_sequence = FeatureSequence(
        values=values,
        frame_times=np.asarray(frame_times, dtype=np.float64),
        sample_rate=DEFAULT_SAMPLE_RATE,
        hop_length=DEFAULT_HOP_LENGTH,
        feature_name="chroma_stft",
        metadata={"recording_id": recording.recording_id},
    )
    feature_cache[recording.audio_path] = feature_sequence
    return feature_sequence


def _write_outputs(method: str, metrics_frame: pd.DataFrame, errors_frame: pd.DataFrame) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = METRICS_DIR / f"{method}_paper_test"
    metrics_frame.to_csv(f"{prefix}_pairs.csv", index=False)
    metrics.summarize_metrics(metrics_frame).to_csv(f"{prefix}_summary.csv", index=False)
    metrics.summarize_metrics_by_piece(metrics_frame).to_csv(
        f"{prefix}_piece_summary.csv",
        index=False,
    )
    metrics.summarize_error_by_track_phase(errors_frame).to_csv(
        f"{prefix}_phase_summary.csv",
        index=False,
    )
    errors_frame.to_csv(f"{prefix}_beat_errors.csv", index=False)
    metrics.compute_tolerance_curve(errors_frame).to_csv(
        f"{prefix}_tolerance_curve.csv",
        index=False,
    )


def _reference_index(reference: Recording) -> int:
    recordings = sorted(
        path for path in reference.audio_path.parent.glob("*.wav") if path.is_file()
    )
    return recordings.index(reference.audio_path)


def _model_path(piece: str, reference_index: int) -> Path:
    return MODEL_DIR / f"{piece}_reference{reference_index}_train{TRAIN_PERCENT}_hmm.npz"


def _warp_factor(pair: RecordingPair) -> float:
    reference_duration = _duration(pair.reference.audio_path)
    query_duration = _duration(pair.query.audio_path)
    return max(reference_duration, query_duration) / min(reference_duration, query_duration)


def _duration(audio_path: Path) -> float:
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)
