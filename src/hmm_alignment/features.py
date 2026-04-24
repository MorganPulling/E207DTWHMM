"""Audio feature extraction helpers for reference-specific alignment."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def l2_normalize_frames(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a frame-major feature matrix to unit L2 norm per frame."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape (num_frames, num_features)")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def extract_chroma(
    audio_path: str | Path,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 4096,
) -> np.ndarray:
    """Load an audio file and return L2-normalized chroma frames shaped (T, 12)."""

    y, actual_sr = librosa.load(Path(audio_path), sr=sr, mono=True)
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=actual_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        norm=None,
    )
    return l2_normalize_frames(chroma.T)


def frames_to_times(frames: np.ndarray, sr: int = 22050, hop_length: int = 512) -> np.ndarray:
    """Convert frame indices to seconds."""

    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)


def times_to_frames(times: np.ndarray, sr: int = 22050, hop_length: int = 512) -> np.ndarray:
    """Convert times in seconds to integer frame indices."""

    return librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
