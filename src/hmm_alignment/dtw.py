"""Offline DTW utilities for generating HMM pseudo-labels."""

from __future__ import annotations

import librosa
import numpy as np

from .features import l2_normalize_frames


def local_cosine_cost(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return a framewise cosine-distance cost matrix shaped (T, R)."""

    query = l2_normalize_frames(query_features, eps=eps)
    reference = l2_normalize_frames(reference_features, eps=eps)
    return 1.0 - np.clip(query @ reference.T, -1.0, 1.0)


def dtw_pseudo_labels(
    query_features: np.ndarray,
    reference_features: np.ndarray,
) -> np.ndarray:
    """Align query to reference and return one reference-frame label per query frame.

    Labels are zero-based reference frame indices. If DTW emits multiple reference
    matches for a query frame, the median reference frame is used. Any missing
    query frames are filled by interpolation.
    """

    query = np.asarray(query_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("query_features and reference_features must be 2D")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference feature dimensions must match")
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("query and reference must both contain at least one frame")

    cost = local_cosine_cost(query, reference)
    _, path = librosa.sequence.dtw(C=cost)

    by_query: list[list[int]] = [[] for _ in range(len(query))]
    for query_idx, ref_idx in path:
        if 0 <= query_idx < len(query) and 0 <= ref_idx < len(reference):
            by_query[int(query_idx)].append(int(ref_idx))

    labels = np.full(len(query), np.nan, dtype=np.float64)
    for idx, refs in enumerate(by_query):
        if refs:
            labels[idx] = np.median(refs)

    known = np.flatnonzero(~np.isnan(labels))
    if known.size == 0:
        raise RuntimeError("DTW path did not produce any valid query/reference pairs")
    if known.size < len(labels):
        missing = np.flatnonzero(np.isnan(labels))
        labels[missing] = np.interp(missing, known, labels[known])

    labels = np.rint(labels).astype(np.int64)
    labels = np.clip(labels, 0, len(reference) - 1)
    return np.maximum.accumulate(labels)
