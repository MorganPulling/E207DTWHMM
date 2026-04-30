"""Feature extraction utilities for alignment experiments."""

from __future__ import annotations

import librosa
import numpy as np

from scripts.config import DEFAULT_FRAME_LENGTH, DEFAULT_HOP_LENGTH
from scripts.models import FeatureSequence


def compute_features(
    audio: np.ndarray,
    sr: int,
    feature_name: str = "chroma_stft",
    hop_length: int = DEFAULT_HOP_LENGTH,
    frame_length: int = DEFAULT_FRAME_LENGTH,
) -> FeatureSequence:
    """Compute a frame-level feature representation for alignment."""

    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Audio input must be a mono waveform.")
    if signal.size == 0:
        raise ValueError("Audio input must contain at least one sample.")

    normalized = librosa.util.normalize(signal)

    n_fft = min(frame_length, signal.size)

    if feature_name == "chroma_stft":
        feature_matrix = librosa.feature.chroma_stft(
            y=normalized,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    elif feature_name == "mfcc":
        feature_matrix = librosa.feature.mfcc(
            y=normalized,
            sr=sr,
            n_mfcc=20,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    else:
        raise ValueError(
            f"Unsupported feature_name={feature_name!r}. Use 'chroma_stft' or 'mfcc'."
        )

    frame_times = librosa.frames_to_time(
        np.arange(feature_matrix.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return FeatureSequence(
        values=np.ascontiguousarray(feature_matrix.T, dtype=np.float32),
        frame_times=np.asarray(frame_times, dtype=np.float64),
        sample_rate=int(sr),
        hop_length=hop_length,
        feature_name=feature_name,
        metadata={"frame_length": frame_length, "n_fft": n_fft},
    )
