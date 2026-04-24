"""Evaluation helpers for HMM alignment outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameErrorMetrics:
    mean_abs_error: float
    median_abs_error: float
    within_1_frame: float
    within_5_frames: float
    within_10_frames: float


def frame_error_metrics(predicted: np.ndarray, target: np.ndarray) -> FrameErrorMetrics:
    """Compare predicted and target reference-frame paths."""

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have the same shape")
    errors = np.abs(predicted - target)
    return FrameErrorMetrics(
        mean_abs_error=float(np.mean(errors)),
        median_abs_error=float(np.median(errors)),
        within_1_frame=float(np.mean(errors <= 1)),
        within_5_frames=float(np.mean(errors <= 5)),
        within_10_frames=float(np.mean(errors <= 10)),
    )
