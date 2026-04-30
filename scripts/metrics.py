"""Evaluation metrics for alignment quality."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scripts.models import AlignmentResult


# Tolerances in seconds
DEFAULT_TOLERANCES = (0.01, 0.020, 0.050, 0.100, 0.200, 0.500)
DEFAULT_TOLERANCE_STEP_S = 0.01
DEFAULT_MAX_TOLERANCE_S = 1.0


def estimate_query_times(
    alignment_result: AlignmentResult,
    reference_times: np.ndarray,
) -> np.ndarray:
    """Interpolate query timestamps for arbitrary reference timestamps."""

    ref_path, query_path = _collapse_duplicates(
        alignment_result.reference_times,
        alignment_result.query_times,
    )
    return np.interp(
        np.asarray(reference_times, dtype=np.float64),
        ref_path,
        query_path,
        left=query_path[0],
        right=query_path[-1],
    )


def estimate_reference_times(
    alignment_result: AlignmentResult,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate reference timestamps for arbitrary query timestamps."""

    query_path, reference_path = _collapse_duplicates(
        alignment_result.query_times,
        alignment_result.reference_times,
    )
    return np.interp(
        np.asarray(query_times, dtype=np.float64),
        query_path,
        reference_path,
        left=reference_path[0],
        right=reference_path[-1],
    )


def compute_alignment_metrics(
    alignment_result: AlignmentResult,
    reference_beats: np.ndarray,
    query_beats: np.ndarray,
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
) -> dict[str, float | int | str]:
    """Compare an estimated query-to-reference alignment against beat correspondence."""

    error_frame = compute_alignment_error_trace(
        alignment_result,
        reference_beats=reference_beats,
        query_beats=query_beats,
    )
    errors = error_frame["error_s"].to_numpy(dtype=np.float64)
    abs_errors = error_frame["abs_error_s"].to_numpy(dtype=np.float64)

    metrics: dict[str, float | int | str] = {
        "method_name": alignment_result.method_name,
        "reference_id": alignment_result.reference_id,
        "query_id": alignment_result.query_id,
        "num_beats_used": int(error_frame.shape[0]),
        "mean_error_s": float(np.mean(errors)),
        "mean_abs_error_s": float(np.mean(abs_errors)),
        "median_abs_error_s": float(np.median(abs_errors)),
        "rmse_s": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_s": float(np.max(abs_errors)),
        "p95_abs_error_s": float(np.percentile(abs_errors, 95)),
    }

    for tolerance in tolerances:
        milliseconds = int(round(tolerance * 1000))
        metrics[f"within_{milliseconds}ms"] = float(np.mean(abs_errors <= tolerance))

    return metrics


def compute_alignment_error_trace(
    alignment_result: AlignmentResult,
    reference_beats: np.ndarray,
    query_beats: np.ndarray,
) -> pd.DataFrame:
    """Return per-beat alignment errors for one reference/query pair."""

    reference_beats = np.asarray(reference_beats, dtype=np.float64)
    query_beats = np.asarray(query_beats, dtype=np.float64)
    num_beats = min(reference_beats.size, query_beats.size)
    if num_beats == 0:
        raise ValueError("Both recordings must contain at least one beat timestamp.")

    query_eval = query_beats[:num_beats]
    reference_eval = reference_beats[:num_beats]
    estimated_reference = estimate_reference_times(alignment_result, query_eval)
    errors = estimated_reference - reference_eval
    abs_errors = np.abs(errors)

    return pd.DataFrame(
        {
            "method_name": alignment_result.method_name,
            "reference_id": alignment_result.reference_id,
            "query_id": alignment_result.query_id,
            "beat_index": np.arange(num_beats, dtype=np.int64),
            "reference_beat_time_s": reference_eval,
            "query_beat_time_s": query_eval,
            "estimated_reference_time_s": estimated_reference,
            "error_s": errors,
            "abs_error_s": abs_errors,
        }
    )


def summarize_metrics(metric_rows: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    """Aggregate per-pair metrics into a compact summary table."""

    frame = metric_rows if isinstance(metric_rows, pd.DataFrame) else pd.DataFrame(metric_rows)
    if frame.empty:
        return pd.DataFrame()

    aggregation_spec: dict[str, tuple[str, str]] = {
        "num_pairs": ("method_name", "size"),
        "mean_mae_s": ("mean_abs_error_s", "mean"),
        "median_mae_s": ("mean_abs_error_s", "median"),
        "mean_rmse_s": ("rmse_s", "mean"),
        "mean_p95_abs_error_s": ("p95_abs_error_s", "mean"),
    }
    within_columns = sorted(column for column in frame.columns if column.startswith("within_"))
    for column in within_columns:
        aggregation_spec[f"mean_{column}"] = (column, "mean")

    return frame.groupby("method_name", dropna=False).agg(**aggregation_spec).reset_index()


def summarize_metrics_by_piece(
    metric_rows: pd.DataFrame | list[dict[str, object]],
) -> pd.DataFrame:
    """Aggregate per-pair metrics by method and piece."""

    frame = metric_rows if isinstance(metric_rows, pd.DataFrame) else pd.DataFrame(metric_rows)
    if frame.empty:
        return pd.DataFrame()

    return (
        frame.groupby(["method_name", "piece"], dropna=False)
        .agg(
            num_pairs=("pair_id", "size"),
            mean_mae_s=("mean_abs_error_s", "mean"),
            median_mae_s=("mean_abs_error_s", "median"),
            mean_rmse_s=("rmse_s", "mean"),
            mean_p95_abs_error_s=("p95_abs_error_s", "mean"),
            mean_within_100ms=("within_100ms", "mean"),
            mean_within_200ms=("within_200ms", "mean"),
        )
        .reset_index()
    )


def summarize_error_by_track_phase(
    error_rows: pd.DataFrame | list[dict[str, object]],
) -> pd.DataFrame:
    """Aggregate beat errors for the first and second halves of each pair."""

    frame = error_rows if isinstance(error_rows, pd.DataFrame) else pd.DataFrame(error_rows)
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    pair_lengths = working.groupby("pair_id", dropna=False)["beat_index"].transform("max") + 1
    normalized_position = (working["beat_index"].to_numpy(dtype=np.float64) + 0.5) / pair_lengths.to_numpy(
        dtype=np.float64
    )
    working["track_phase"] = np.where(normalized_position <= 0.5, "early", "late")
    working["sq_error_s"] = working["error_s"].to_numpy(dtype=np.float64) ** 2
    working["within_100ms"] = working["abs_error_s"].to_numpy(dtype=np.float64) <= 0.1
    working["within_200ms"] = working["abs_error_s"].to_numpy(dtype=np.float64) <= 0.2

    summary = (
        working.groupby(["method_name", "track_phase"], dropna=False)
        .agg(
            num_predictions=("beat_index", "size"),
            mean_abs_error_s=("abs_error_s", "mean"),
            rmse_s=("sq_error_s", lambda values: float(np.sqrt(np.mean(values)))),
            within_100ms=("within_100ms", "mean"),
            within_200ms=("within_200ms", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["method_name", "track_phase"]).reset_index(drop=True)


def build_tolerance_grid(
    max_tolerance_s: float = DEFAULT_MAX_TOLERANCE_S,
    step_s: float = DEFAULT_TOLERANCE_STEP_S,
) -> np.ndarray:
    """Build an inclusive tolerance grid for error-rate sweeps."""

    if max_tolerance_s <= 0:
        raise ValueError("max_tolerance_s must be positive.")
    if step_s <= 0:
        raise ValueError("step_s must be positive.")

    grid = np.arange(0.0, max_tolerance_s + (step_s * 0.5), step_s, dtype=np.float64)
    if grid.size == 0 or grid[-1] < max_tolerance_s:
        grid = np.append(grid, max_tolerance_s)
    return np.unique(np.clip(grid, 0.0, max_tolerance_s))


def compute_tolerance_curve(
    error_rows: pd.DataFrame | list[dict[str, object]],
    tolerances: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Aggregate error-rate-versus-tolerance curves from per-beat error rows."""

    frame = error_rows if isinstance(error_rows, pd.DataFrame) else pd.DataFrame(error_rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["method_name", "tolerance_s", "correct_rate", "error_rate", "num_predictions"]
        )

    tolerance_values = (
        np.asarray(tuple(tolerances), dtype=np.float64)
        if tolerances is not None
        else build_tolerance_grid()
    )
    tolerance_values = np.sort(np.unique(tolerance_values))

    rows: list[dict[str, float | int | str]] = []
    for method_name, method_frame in frame.groupby("method_name", dropna=False):
        abs_errors = method_frame["abs_error_s"].to_numpy(dtype=np.float64)
        if abs_errors.size == 0:
            continue
        for tolerance in tolerance_values:
            correct_rate = float(np.mean(abs_errors <= tolerance))
            rows.append(
                {
                    "method_name": method_name,
                    "tolerance_s": float(tolerance),
                    "correct_rate": correct_rate,
                    "error_rate": float(1.0 - correct_rate),
                    "num_predictions": int(abs_errors.size),
                }
            )
    return pd.DataFrame(rows)


def _collapse_duplicates(
    reference_times: np.ndarray,
    query_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_times = np.asarray(reference_times, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    unique_reference, inverse = np.unique(reference_times, return_inverse=True)

    query_sums = np.zeros_like(unique_reference, dtype=np.float64)
    counts = np.zeros_like(unique_reference, dtype=np.int64)
    np.add.at(query_sums, inverse, query_times)
    np.add.at(counts, inverse, 1)

    averaged_query = query_sums / counts
    return unique_reference, averaged_query
