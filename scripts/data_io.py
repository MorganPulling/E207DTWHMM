"""Data loading helpers for benchmark assets and annotations."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from scripts.config import RAW_DATA_DIR
from scripts.models import Recording, RecordingPair


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
BEAT_SUFFIXES = {".txt", ".csv", ".tsv", ".beat"}
MANIFEST_FILENAMES = ("mazurka_manifest.csv", "manifest.csv")
MANIFEST_COLUMNS = {"piece", "recording_id", "audio_path", "beats_path"}


def discover_recordings(dataset_root: Path | str | None = None) -> list[Recording]:
    """Discover recordings from a manifest file or an inferred folder layout."""

    root = Path(dataset_root) if dataset_root is not None else RAW_DATA_DIR
    manifest_path = _find_manifest(root)
    if manifest_path is not None:
        return _discover_from_manifest(manifest_path)
    if _has_split_mazurka_layout(root):
        return _discover_from_split_layout(root)
    return _discover_from_directory(root)


def build_recording_pairs(
    recordings: Iterable[Recording],
    require_annotations: bool = True,
) -> list[RecordingPair]:
    """Build all directed within-piece recording pairs for evaluation."""

    grouped: dict[str, list[Recording]] = {}
    for recording in recordings:
        if require_annotations and recording.beats_path is None:
            continue
        grouped.setdefault(recording.piece, []).append(recording)

    pairs: list[RecordingPair] = []
    for piece, piece_recordings in grouped.items():
        ordered = sorted(piece_recordings, key=lambda item: item.recording_id)
        for reference, query in itertools.permutations(ordered, 2):
            pairs.append(RecordingPair(piece=piece, reference=reference, query=query))
    return pairs


def load_audio(
    audio_path: Path | str,
    sample_rate: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio file through librosa."""

    waveform, sr = librosa.load(
        Path(audio_path),
        sr=sample_rate,
        mono=mono,
    )
    return np.asarray(waveform, dtype=np.float64), int(sr)


def load_beat_timestamps(beats_path: Path | str) -> np.ndarray:
    """Load beat timestamps from a text, CSV, or TSV file.

    The loader accepts one timestamp per line or a table and extracts the first
    numeric column it finds.
    """

    path = Path(beats_path)
    if not path.exists():
        raise FileNotFoundError(f"Beat annotation file does not exist: {path}")

    delimiter, comments = _beat_file_parse_options(path)
    try:
        data = np.genfromtxt(path, delimiter=delimiter, dtype=float, comments=comments)
    except ValueError as exc:
        raise ValueError(f"Could not parse beat timestamps from {path}") from exc

    if data.size == 0:
        return np.array([], dtype=np.float64)

    if data.ndim == 0:
        data = np.array([float(data)], dtype=np.float64)
    elif data.ndim == 1:
        if np.isnan(data[0]):
            data = _load_numeric_column_with_header(path, delimiter, comments)
    else:
        if np.isnan(data).all():
            data = _load_numeric_column_with_header(path, delimiter, comments)
        else:
            data = data[:, 0]

    clean = np.asarray(data, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    return np.sort(clean)


def load_pair_audio(
    pair: RecordingPair,
    sample_rate: int | None = None,
) -> tuple[tuple[np.ndarray, int], tuple[np.ndarray, int]]:
    """Load both recordings in a pair using the same sample-rate policy."""

    reference_audio = load_audio(pair.reference.audio_path, sample_rate=sample_rate)
    query_audio = load_audio(pair.query.audio_path, sample_rate=sample_rate)
    return reference_audio, query_audio


def _find_manifest(root: Path) -> Path | None:
    for filename in MANIFEST_FILENAMES:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def _discover_from_manifest(manifest_path: Path) -> list[Recording]:
    recordings: list[Recording] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = MANIFEST_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {missing}"
            )
        for row in reader:
            audio_path = _resolve_path(manifest_path.parent, row["audio_path"])
            beats_path = _resolve_path(manifest_path.parent, row["beats_path"])
            recordings.append(
                Recording(
                    piece=row["piece"],
                    recording_id=row["recording_id"],
                    audio_path=audio_path,
                    beats_path=beats_path,
                )
            )
    return recordings


def _has_split_mazurka_layout(root: Path) -> bool:
    return (root / "wav_22050_mono").is_dir() and (root / "annotations_beat").is_dir()


def _discover_from_split_layout(root: Path) -> list[Recording]:
    audio_root = root / "wav_22050_mono"
    beats_root = root / "annotations_beat"
    beat_index = _index_beat_files(beats_root)

    recordings: list[Recording] = []
    audio_files = sorted(
        path for path in audio_root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    for audio_path in audio_files:
        piece = audio_path.parent.name
        recording_id = audio_path.stem
        beats_path = _match_split_layout_beats(
            piece=piece,
            recording_id=recording_id,
            beats_root=beats_root,
            beat_index=beat_index,
        )
        recordings.append(
            Recording(
                piece=piece,
                recording_id=recording_id,
                audio_path=audio_path,
                beats_path=beats_path,
            )
        )
    return recordings


def _discover_from_directory(root: Path) -> list[Recording]:
    recordings: list[Recording] = []
    audio_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    beat_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in BEAT_SUFFIXES
        and "beat" in path.stem.lower()
    )

    for audio_path in audio_files:
        piece = audio_path.parent.parent.name if audio_path.parent != root else audio_path.stem
        beats_path = _match_beats_file(audio_path, beat_files)
        recordings.append(
            Recording(
                piece=piece,
                recording_id=audio_path.stem,
                audio_path=audio_path,
                beats_path=beats_path,
            )
        )
    return recordings


def _match_beats_file(audio_path: Path, beat_files: list[Path]) -> Path | None:
    sibling_candidates = [path for path in beat_files if path.parent == audio_path.parent]
    for candidate in sibling_candidates:
        if audio_path.stem in candidate.stem or candidate.stem in audio_path.stem:
            return candidate
    if sibling_candidates:
        return sibling_candidates[0]

    piece_name = audio_path.parent.parent.name
    piece_candidates = [path for path in beat_files if piece_name and piece_name in path.parts]
    for candidate in piece_candidates:
        if audio_path.stem in candidate.stem:
            return candidate
    return piece_candidates[0] if piece_candidates else None


def _index_beat_files(beats_root: Path) -> dict[str, list[Path]]:
    indexed_paths: dict[str, list[Path]] = defaultdict(list)
    for beat_path in sorted(beats_root.rglob("*")):
        if beat_path.is_file() and beat_path.suffix.lower() in BEAT_SUFFIXES:
            indexed_paths[beat_path.stem].append(beat_path)
    return indexed_paths


def _match_split_layout_beats(
    piece: str,
    recording_id: str,
    beats_root: Path,
    beat_index: dict[str, list[Path]],
) -> Path | None:
    piece_dir = beats_root / piece
    for suffix in sorted(BEAT_SUFFIXES):
        candidate = piece_dir / f"{recording_id}{suffix}"
        if candidate.exists():
            return candidate

    stem_matches = beat_index.get(recording_id, [])
    if not stem_matches:
        return None

    piece_matches = [path for path in stem_matches if piece in path.parts]
    if piece_matches:
        return piece_matches[0]
    return stem_matches[0]


def _load_numeric_column_with_header(
    path: Path,
    delimiter: str | None,
    comments: str | None,
) -> np.ndarray:
    raw_rows = np.genfromtxt(path, delimiter=delimiter, dtype=str, comments=comments)
    if raw_rows.ndim == 1:
        raw_rows = raw_rows.reshape(-1, 1)

    for column_index in range(raw_rows.shape[1]):
        try:
            numeric = raw_rows[1:, column_index].astype(float)
        except ValueError:
            continue
        numeric = numeric[np.isfinite(numeric)]
        if numeric.size > 0:
            return numeric
    raise ValueError(f"Could not find a numeric timestamp column in {path}")


def _beat_file_parse_options(path: Path) -> tuple[str | None, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t", None
    if suffix == ".beat":
        return None, "%"
    return ",", None


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
