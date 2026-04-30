"""Shared data models for benchmark loading, alignment, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Recording:
    """Single benchmark recording plus optional beat annotations."""

    piece: str
    recording_id: str
    audio_path: Path
    beats_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecordingPair:
    """Directed benchmark case between two performances of the same piece."""

    piece: str
    reference: Recording
    query: Recording

    @property
    def pair_id(self) -> str:
        return f"{self.reference.recording_id}__{self.query.recording_id}"


@dataclass(slots=True)
class FeatureSequence:
    """Frame-level representation used by alignment methods."""

    values: np.ndarray
    frame_times: np.ndarray
    sample_rate: int
    hop_length: int
    feature_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("FeatureSequence.values must be 2D (frames x dims).")
        if self.frame_times.ndim != 1:
            raise ValueError("FeatureSequence.frame_times must be 1D.")
        if self.values.shape[0] != self.frame_times.shape[0]:
            raise ValueError("FeatureSequence values and frame_times must align.")


@dataclass(slots=True)
class AlignmentResult:
    """Monotonic alignment path between reference and query recordings."""

    method_name: str
    reference_id: str
    query_id: str
    reference_times: np.ndarray
    query_times: np.ndarray
    path: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_times.ndim != 1 or self.query_times.ndim != 1:
            raise ValueError("AlignmentResult times must be 1D arrays.")
        if self.reference_times.shape != self.query_times.shape:
            raise ValueError("AlignmentResult time arrays must have the same shape.")
        if self.path.ndim != 2 or self.path.shape[1] != 2:
            raise ValueError("AlignmentResult.path must have shape (N, 2).")
        if self.path.shape[0] != self.reference_times.shape[0]:
            raise ValueError("AlignmentResult.path must align with the time arrays.")
