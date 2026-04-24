"""Supervised HMM parameter estimation from DTW pseudo-labels."""

from __future__ import annotations

import numpy as np

from .dtw import dtw_pseudo_labels
from .hmm import ReferenceHMM


def estimate_jump_probs(
    label_sets: list[np.ndarray],
    max_jump: int = 4,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Estimate bounded jump probabilities from monotone label sequences."""

    if max_jump < 0:
        raise ValueError("max_jump must be nonnegative")
    counts = np.full(max_jump + 1, smoothing, dtype=np.float64)
    for labels in label_sets:
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) < 2:
            continue
        jumps = np.diff(labels)
        jumps = jumps[(jumps >= 0) & (jumps <= max_jump)]
        counts += np.bincount(jumps, minlength=max_jump + 1)[: max_jump + 1]
    return counts / counts.sum()


def estimate_shared_diagonal_covariance(
    reference_features: np.ndarray,
    training_feature_sets: list[np.ndarray],
    label_sets: list[np.ndarray],
    variance_floor: float = 1e-6,
) -> np.ndarray:
    """Estimate shared diagonal covariance from aligned residuals."""

    reference = np.asarray(reference_features, dtype=np.float64)
    residuals = []
    for query, labels in zip(training_feature_sets, label_sets):
        query = np.asarray(query, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if len(query) != len(labels):
            raise ValueError("each query feature matrix must match its label length")
        residuals.append(query - reference[labels])
    if not residuals:
        raise ValueError("at least one training feature set is required")
    stacked = np.vstack(residuals)
    return np.maximum(np.mean(stacked * stacked, axis=0), variance_floor)


def train_reference_hmm(
    reference_features: np.ndarray,
    training_feature_sets: list[np.ndarray],
    max_jump: int = 4,
    smoothing: float = 1e-3,
    variance_floor: float = 1e-6,
) -> ReferenceHMM:
    """Train the recommended first-pass reference-specific HMM."""

    if not training_feature_sets:
        raise ValueError("training_feature_sets must contain at least one query")
    labels = [
        dtw_pseudo_labels(query_features, reference_features)
        for query_features in training_feature_sets
    ]
    jump_probs = estimate_jump_probs(labels, max_jump=max_jump, smoothing=smoothing)
    covariance = estimate_shared_diagonal_covariance(
        reference_features,
        training_feature_sets,
        labels,
        variance_floor=variance_floor,
    )
    return ReferenceHMM(
        reference_features=reference_features,
        jump_probs=jump_probs,
        covariance=covariance,
        covariance_type="diagonal",
        hard_start=True,
        variance_floor=variance_floor,
    )
