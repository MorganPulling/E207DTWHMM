"""Reference-specific HMM alignment package."""

from .dtw import dtw_pseudo_labels, local_cosine_cost
from .features import extract_chroma, frames_to_times, l2_normalize_frames, times_to_frames
from .hmm import AlignmentResult, ReferenceHMM
from .training import train_reference_hmm

__all__ = [
    "AlignmentResult",
    "ReferenceHMM",
    "dtw_pseudo_labels",
    "extract_chroma",
    "frames_to_times",
    "l2_normalize_frames",
    "local_cosine_cost",
    "times_to_frames",
    "train_reference_hmm",
]
