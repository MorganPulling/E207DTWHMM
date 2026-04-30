"""Fixed train70 reference-to-held-out benchmark for E207DTWHMM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
BENCHMARK_MODES = ("paper_test", "least_recordings")
MODEL_DIR = Path("artifacts/train70_models")
TRAIN_PERCENT = 70
MAX_WARP_FACTOR = 2.0
MODEL_FILENAME_RE = re.compile(
    r"^(?P<piece>.+)_reference(?P<reference_index>\d+)_train(?P<train_percent>\d+)_hmm\.npz$"
)


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    piece: str
    reference_index: int
    path: Path


def run_benchmark(method: str, mode: str = "paper_test") -> pd.DataFrame:
    """Run the fixed train-reference-to-held-out benchmark and write outputs."""

    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}")
    mode = mode.strip().lower()
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"mode must be one of: {', '.join(BENCHMARK_MODES)}")

    recordings = data_io.discover_recordings(DATA_DIR)
    pairs = _heldout_pairs(recordings, mode=mode)
    if not pairs:
        raise ValueError(f"No train70 held-out benchmark pairs were found for mode {mode!r}.")

    rows: list[dict[str, object]] = []
    error_frames: list[pd.DataFrame] = []
    feature_cache: dict[Path, FeatureSequence] = {}

    for pair in tqdm(pairs, desc=f"{method}_{mode}"):
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
    _write_outputs(method, mode, metrics_frame, errors_frame)
    return metrics_frame


def _heldout_pairs(recordings: list[Recording], mode: str = "paper_test") -> list[RecordingPair]:
    grouped: dict[str, list[Recording]] = {}
    for recording in recordings:
        if recording.beats_path is not None:
            grouped.setdefault(recording.piece, []).append(recording)

    models = _model_specs()
    if mode == "least_recordings":
        piece = _least_recordings_piece(grouped, models)
        models = [model for model in models if model.piece == piece]

    pairs: list[RecordingPair] = []
    for model in models:
        piece_recordings = grouped.get(model.piece)
        if not piece_recordings:
            continue
        ordered = sorted(piece_recordings, key=lambda item: item.audio_path)
        if model.reference_index >= len(ordered):
            raise ValueError(
                f"Model {model.path} references recording index {model.reference_index}, "
                f"but piece {model.piece!r} only has {len(ordered)} annotated recordings."
            )
        train_count = len(ordered) * TRAIN_PERCENT // 100
        heldout_queries = ordered[train_count:]
        reference = _with_model_metadata(ordered[model.reference_index], model)
        for query in heldout_queries:
            if query.recording_id == reference.recording_id:
                continue
            pair = RecordingPair(piece=model.piece, reference=reference, query=query)
            if _warp_factor(pair) <= MAX_WARP_FACTOR:
                pairs.append(pair)

    return sorted(pairs, key=lambda pair: (pair.piece, pair.pair_id))


def _least_recordings_piece(
    grouped: dict[str, list[Recording]],
    models: list[_ModelSpec],
) -> str | None:
    valid_model_pieces = {
        model.piece
        for model in models
        if model.piece in grouped and model.reference_index < len(grouped[model.piece])
    }
    candidates = [
        (len(recordings), piece)
        for piece, recordings in grouped.items()
        if piece in valid_model_pieces
    ]
    if not candidates:
        return None
    return min(candidates)[1]


def _run_pair(
    pair: RecordingPair,
    method: str,
    feature_cache: dict[Path, FeatureSequence],
) -> AlignmentResult:
    reference_features = _features(pair.reference, feature_cache)
    query_features = _features(pair.query, feature_cache)

    if method == "offline_dtw":
        return offline_dtw.run_offline_dtw(reference_features, query_features)

    reference_index = _pair_reference_index(pair)
    model_path = _pair_model_path(pair, reference_index)
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


def _write_outputs(
    method: str,
    mode: str,
    metrics_frame: pd.DataFrame,
    errors_frame: pd.DataFrame,
) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = METRICS_DIR / f"{method}_{mode}"
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


def _pair_reference_index(pair: RecordingPair) -> int:
    reference_index = pair.reference.metadata.get("reference_index")
    if reference_index is not None:
        return int(reference_index)
    return _reference_index(pair.reference)


def _pair_model_path(pair: RecordingPair, reference_index: int) -> Path:
    model_path = pair.reference.metadata.get("model_path")
    if model_path is not None:
        return Path(model_path)
    return _model_path(pair.piece, reference_index)


def _model_specs() -> list[_ModelSpec]:
    if not MODEL_DIR.exists():
        return []

    specs: list[_ModelSpec] = []
    for path in sorted(MODEL_DIR.glob("*.npz")):
        match = MODEL_FILENAME_RE.match(path.name)
        if match is None:
            continue
        train_percent = int(match.group("train_percent"))
        if train_percent != TRAIN_PERCENT:
            continue
        specs.append(
            _ModelSpec(
                piece=match.group("piece"),
                reference_index=int(match.group("reference_index")),
                path=path,
            )
        )
    return specs


def _with_model_metadata(reference: Recording, model: _ModelSpec) -> Recording:
    return Recording(
        piece=reference.piece,
        recording_id=reference.recording_id,
        audio_path=reference.audio_path,
        beats_path=reference.beats_path,
        metadata={
            **reference.metadata,
            "model_path": str(model.path),
            "reference_index": model.reference_index,
        },
    )


def _model_path(piece: str, reference_index: int) -> Path:
    return MODEL_DIR / f"{piece}_reference{reference_index}_train{TRAIN_PERCENT}_hmm.npz"


def _warp_factor(pair: RecordingPair) -> float:
    reference_duration = _duration(pair.reference.audio_path)
    query_duration = _duration(pair.query.audio_path)
    return max(reference_duration, query_duration) / min(reference_duration, query_duration)


def _duration(audio_path: Path) -> float:
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)
