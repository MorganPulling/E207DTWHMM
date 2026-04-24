"""Dataset discovery helpers for the Mazurka-style local data layout."""

from __future__ import annotations

from pathlib import Path


def list_pieces(data_root: str | Path = "data") -> list[str]:
    wav_root = Path(data_root) / "wav_22050_mono"
    return sorted(path.name for path in wav_root.iterdir() if path.is_dir())


def list_recordings(piece: str, data_root: str | Path = "data") -> list[Path]:
    piece_root = Path(data_root) / "wav_22050_mono" / piece
    return sorted(piece_root.glob("*.wav"))


def select_reference_and_queries(
    piece: str,
    reference_index: int = 0,
    data_root: str | Path = "data",
    max_queries: int | None = None,
) -> tuple[Path, list[Path]]:
    recordings = list_recordings(piece, data_root=data_root)
    if not recordings:
        raise ValueError(f"no WAV recordings found for piece {piece!r}")
    if reference_index < 0 or reference_index >= len(recordings):
        raise IndexError(f"reference_index must be in [0, {len(recordings) - 1}]")
    reference = recordings[reference_index]
    queries = [path for idx, path in enumerate(recordings) if idx != reference_index]
    if max_queries is not None:
        queries = queries[:max_queries]
    return reference, queries
