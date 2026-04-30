"""Fixed train70 held-out benchmark for HMM and offline DTW."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from .features import extract_chroma
from .hmm import ReferenceHMM
from .offline_dtw import run_offline_dtw


DATA_ROOT = Path("data")
MODEL_DIR = Path("artifacts/train70_models")
OUTPUT_DIR = Path("outputs/metrics")
METHODS = {"hmm_train70", "offline_dtw"}
SAMPLE_RATE = 22050
HOP_LENGTH = 512
TRAIN_PERCENT = 70
MAX_WARP_FACTOR = 2.0
TOLERANCES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def run_benchmark(method: str) -> pd.DataFrame:
    """Run the fixed paper-test-style benchmark and write CSV outputs."""

    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(METHODS))}")

    recordings_by_piece = _discover_recordings()
    pairs = _heldout_pairs(recordings_by_piece)
    if not pairs:
        raise ValueError("No held-out benchmark pairs were found.")

    feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, object]] = []
    error_rows: list[pd.DataFrame] = []

    for pair in tqdm(pairs, desc=f"{method}_paper_test"):
        result = _align_pair(pair, method, feature_cache)
        reference_beats = _load_beats(pair["reference"]["beats_path"])
        query_beats = _load_beats(pair["query"]["beats_path"])
        metric_row, error_frame = _score_alignment(result, reference_beats, query_beats)
        metric_row.update(
            {
                "piece": pair["piece"],
                "pair_id": pair["pair_id"],
                **result["metadata"],
            }
        )
        error_frame["piece"] = pair["piece"]
        error_frame["pair_id"] = pair["pair_id"]
        metric_rows.append(metric_row)
        error_rows.append(error_frame)

    metrics_frame = pd.DataFrame(metric_rows)
    errors_frame = pd.concat(error_rows, ignore_index=True)
    _write_outputs(method, metrics_frame, errors_frame)
    return metrics_frame


def _discover_recordings() -> dict[str, list[dict[str, object]]]:
    recordings_by_piece: dict[str, list[dict[str, object]]] = {}
    audio_root = DATA_ROOT / "wav_22050_mono"
    beat_root = DATA_ROOT / "annotations_beat"
    for piece_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        recordings = []
        for index, audio_path in enumerate(sorted(piece_dir.glob("*.wav"))):
            beats_path = beat_root / piece_dir.name / f"{audio_path.stem}.beat"
            if beats_path.exists():
                recordings.append(
                    {
                        "piece": piece_dir.name,
                        "recording_id": audio_path.stem,
                        "audio_path": audio_path,
                        "beats_path": beats_path,
                        "index": index,
                    }
                )
        recordings_by_piece[piece_dir.name] = recordings
    return recordings_by_piece


def _heldout_pairs(recordings_by_piece: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for piece, recordings in recordings_by_piece.items():
        train_count = len(recordings) * TRAIN_PERCENT // 100
        heldout_queries = recordings[train_count:]
        for reference in recordings:
            if not _model_path(piece, int(reference["index"])).exists():
                continue
            for query in heldout_queries:
                if query["recording_id"] == reference["recording_id"]:
                    continue
                pair = {
                    "piece": piece,
                    "reference": reference,
                    "query": query,
                    "pair_id": f"{reference['recording_id']}__{query['recording_id']}",
                }
                if _warp_factor(reference, query) <= MAX_WARP_FACTOR:
                    pairs.append(pair)
    return sorted(pairs, key=lambda item: (str(item["piece"]), str(item["pair_id"])))


def _align_pair(
    pair: dict[str, object],
    method: str,
    feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    reference = pair["reference"]
    query = pair["query"]
    assert isinstance(reference, dict)
    assert isinstance(query, dict)
    reference_values, reference_times = _features(reference["audio_path"], feature_cache)
    query_values, query_times = _features(query["audio_path"], feature_cache)

    if method == "offline_dtw":
        return run_offline_dtw(
            reference_values,
            query_values,
            reference_times,
            query_times,
            reference_id=str(reference["recording_id"]),
            query_id=str(query["recording_id"]),
        )

    model_path = _model_path(str(pair["piece"]), int(reference["index"]))
    model = ReferenceHMM.load(model_path)
    hmm_result = model.filter(query_values)
    reference_indices = np.clip(hmm_result.map_path, 0, len(reference_times) - 1)
    query_indices = np.arange(len(query_times), dtype=np.int64)
    return {
        "method_name": "hmm_train70",
        "reference_id": reference["recording_id"],
        "query_id": query["recording_id"],
        "reference_times": reference_times[reference_indices],
        "query_times": query_times,
        "path": np.column_stack([reference_indices, query_indices]),
        "metadata": {
            "model_path": str(model_path),
            "train_percent": TRAIN_PERCENT,
            "reference_index": int(reference["index"]),
            "decoder": "filter_map",
            "max_jump": model.max_jump,
        },
    }


def _features(
    audio_path: object,
    feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(audio_path)
    cached = feature_cache.get(path)
    if cached is not None:
        return cached
    values = extract_chroma(path, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    times = librosa.frames_to_time(
        np.arange(values.shape[0]),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    result = (values, np.asarray(times, dtype=np.float64))
    feature_cache[path] = result
    return result


def _score_alignment(
    result: dict[str, object],
    reference_beats: np.ndarray,
    query_beats: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    reference_path, query_path = _collapse_duplicates(
        np.asarray(result["reference_times"], dtype=np.float64),
        np.asarray(result["query_times"], dtype=np.float64),
    )
    num_beats = min(len(reference_beats), len(query_beats))
    query_eval = query_beats[:num_beats]
    reference_eval = reference_beats[:num_beats]
    estimated_reference = np.interp(
        query_eval,
        query_path,
        reference_path,
        left=reference_path[0],
        right=reference_path[-1],
    )
    errors = estimated_reference - reference_eval
    abs_errors = np.abs(errors)
    metric_row: dict[str, object] = {
        "method_name": result["method_name"],
        "reference_id": result["reference_id"],
        "query_id": result["query_id"],
        "num_beats_used": int(num_beats),
        "mean_error_s": float(np.mean(errors)),
        "mean_abs_error_s": float(np.mean(abs_errors)),
        "median_abs_error_s": float(np.median(abs_errors)),
        "rmse_s": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_s": float(np.max(abs_errors)),
        "p95_abs_error_s": float(np.percentile(abs_errors, 95)),
    }
    for tolerance in TOLERANCES:
        metric_row[f"within_{int(round(tolerance * 1000))}ms"] = float(
            np.mean(abs_errors <= tolerance)
        )
    error_frame = pd.DataFrame(
        {
            "method_name": result["method_name"],
            "reference_id": result["reference_id"],
            "query_id": result["query_id"],
            "beat_index": np.arange(num_beats, dtype=np.int64),
            "reference_beat_time_s": reference_eval,
            "query_beat_time_s": query_eval,
            "estimated_reference_time_s": estimated_reference,
            "error_s": errors,
            "abs_error_s": abs_errors,
        }
    )
    return metric_row, error_frame


def _write_outputs(method: str, metrics_frame: pd.DataFrame, errors_frame: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUTPUT_DIR / f"{method}_paper_test"
    metrics_frame.to_csv(f"{prefix}_pairs.csv", index=False)
    _summary(metrics_frame).to_csv(f"{prefix}_summary.csv", index=False)
    _piece_summary(metrics_frame).to_csv(f"{prefix}_piece_summary.csv", index=False)
    _phase_summary(errors_frame).to_csv(f"{prefix}_phase_summary.csv", index=False)
    errors_frame.to_csv(f"{prefix}_beat_errors.csv", index=False)
    _tolerance_curve(errors_frame).to_csv(f"{prefix}_tolerance_curve.csv", index=False)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("method_name", dropna=False)
        .agg(
            num_pairs=("method_name", "size"),
            mean_mae_s=("mean_abs_error_s", "mean"),
            median_mae_s=("mean_abs_error_s", "median"),
            mean_rmse_s=("rmse_s", "mean"),
            mean_p95_abs_error_s=("p95_abs_error_s", "mean"),
            mean_within_100ms=("within_100ms", "mean"),
            mean_within_200ms=("within_200ms", "mean"),
        )
        .reset_index()
    )


def _piece_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method_name", "piece"], dropna=False)
        .agg(
            num_pairs=("pair_id", "size"),
            mean_mae_s=("mean_abs_error_s", "mean"),
            median_mae_s=("mean_abs_error_s", "median"),
            mean_rmse_s=("rmse_s", "mean"),
            mean_within_100ms=("within_100ms", "mean"),
            mean_within_200ms=("within_200ms", "mean"),
        )
        .reset_index()
    )


def _phase_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    pair_lengths = working.groupby("pair_id", dropna=False)["beat_index"].transform("max") + 1
    phase_position = (working["beat_index"].to_numpy(dtype=np.float64) + 0.5) / pair_lengths
    working["track_phase"] = np.where(phase_position <= 0.5, "early", "late")
    working["within_100ms"] = working["abs_error_s"] <= 0.1
    working["within_200ms"] = working["abs_error_s"] <= 0.2
    return (
        working.groupby(["method_name", "track_phase"], dropna=False)
        .agg(
            num_predictions=("beat_index", "size"),
            mean_abs_error_s=("abs_error_s", "mean"),
            rmse_s=("error_s", lambda values: float(np.sqrt(np.mean(values**2)))),
            within_100ms=("within_100ms", "mean"),
            within_200ms=("within_200ms", "mean"),
        )
        .reset_index()
    )


def _tolerance_curve(frame: pd.DataFrame) -> pd.DataFrame:
    tolerances = np.arange(0.0, 1.01, 0.01)
    rows = []
    for method_name, method_frame in frame.groupby("method_name", dropna=False):
        abs_errors = method_frame["abs_error_s"].to_numpy(dtype=np.float64)
        for tolerance in tolerances:
            correct_rate = float(np.mean(abs_errors <= tolerance))
            rows.append(
                {
                    "method_name": method_name,
                    "tolerance_s": float(tolerance),
                    "correct_rate": correct_rate,
                    "error_rate": 1.0 - correct_rate,
                    "num_predictions": int(abs_errors.size),
                }
            )
    return pd.DataFrame(rows)


def _collapse_duplicates(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_x, inverse = np.unique(x_values, return_inverse=True)
    y_sums = np.zeros_like(unique_x, dtype=np.float64)
    counts = np.zeros_like(unique_x, dtype=np.int64)
    np.add.at(y_sums, inverse, y_values)
    np.add.at(counts, inverse, 1)
    return unique_x, y_sums / counts


def _load_beats(path: object) -> np.ndarray:
    data = np.genfromtxt(Path(path), dtype=float, comments="%", usecols=0)
    return np.asarray(data, dtype=np.float64)


def _model_path(piece: str, reference_index: int) -> Path:
    return MODEL_DIR / f"{piece}_reference{reference_index}_train{TRAIN_PERCENT}_hmm.npz"


def _warp_factor(reference: dict[str, object], query: dict[str, object]) -> float:
    reference_duration = _duration(reference["audio_path"])
    query_duration = _duration(query["audio_path"])
    return max(reference_duration, query_duration) / min(reference_duration, query_duration)


def _duration(audio_path: object) -> float:
    info = sf.info(Path(audio_path))
    return float(info.frames) / float(info.samplerate)
