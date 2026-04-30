from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmm_alignment.dtw import dtw_pseudo_labels, local_cosine_cost
from hmm_alignment.evaluation import frame_error_metrics
from hmm_alignment.features import l2_normalize_frames
from hmm_alignment.hmm import ReferenceHMM
from hmm_alignment.training import estimate_jump_probs, train_reference_hmm
from scripts import hmm_benchmark as benchmark
from scripts.models import FeatureSequence, Recording, RecordingPair
from scripts.offline_dtw import run_offline_dtw


def test_l2_normalize_frames_produces_unit_nonzero_rows() -> None:
    features = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = l2_normalize_frames(features)
    assert np.allclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_local_cosine_cost_shape_and_bounds() -> None:
    query = np.eye(3)
    reference = np.eye(3)[:2]
    cost = local_cosine_cost(query, reference)
    assert cost.shape == (3, 2)
    assert np.all(cost >= 0.0)
    assert np.all(cost <= 2.0)


def test_dtw_pseudo_labels_are_bounded_and_query_length() -> None:
    reference = np.eye(4)
    query = np.vstack([reference[0], reference[1], reference[1], reference[2]])
    labels = dtw_pseudo_labels(query, reference)
    assert labels.shape == (len(query),)
    assert labels.min() >= 0
    assert labels.max() < len(reference)
    assert np.all(np.diff(labels) >= 0)


def test_estimate_jump_probs_sums_to_one() -> None:
    probs = estimate_jump_probs(
        [np.array([0, 0, 1, 3]), np.array([0, 1, 1, 2])],
        max_jump=3,
        smoothing=1e-3,
    )
    assert probs.shape == (4,)
    assert np.all(probs > 0)
    assert np.isclose(probs.sum(), 1.0)


def test_filter_posteriors_sum_to_one_and_are_monotone() -> None:
    reference = np.eye(5)
    query = reference.copy()
    model = ReferenceHMM(
        reference_features=reference,
        jump_probs=np.array([0.2, 0.8]),
        covariance=np.full(reference.shape[1], 0.01),
    )
    result = model.filter(query, return_posteriors=True)
    assert result.posteriors is not None
    assert np.allclose(result.posteriors.sum(axis=1), 1.0)
    assert np.all(np.diff(result.map_path) >= 0)
    assert np.all(np.diff(result.map_path) <= model.max_jump)


def test_viterbi_path_is_monotone() -> None:
    reference = np.eye(5)
    query = reference.copy()
    model = ReferenceHMM(
        reference_features=reference,
        jump_probs=np.array([0.1, 0.9]),
        covariance=np.full(reference.shape[1], 0.01),
    )
    path = model.viterbi(query)
    assert path.shape == (len(query),)
    assert np.all(np.diff(path) >= 0)
    assert np.all(np.diff(path) <= model.max_jump)


def test_final_state_absorbs_filter_mass() -> None:
    reference = np.eye(3)
    query = np.repeat(reference[-1][None, :], 5, axis=0)
    model = ReferenceHMM(
        reference_features=reference,
        jump_probs=np.array([0.0, 0.0, 1.0]),
        covariance=np.full(reference.shape[1], 0.01),
    )
    result = model.filter(query, return_posteriors=True)
    assert result.posteriors is not None
    assert np.argmax(result.posteriors[-1]) == len(reference) - 1


def test_model_save_load_preserves_predictions() -> None:
    reference = np.eye(4)
    model = ReferenceHMM(
        reference_features=reference,
        jump_probs=np.array([0.25, 0.75]),
        covariance=np.full(reference.shape[1], 0.05),
    )
    path = Path("artifacts/test_model_roundtrip.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    loaded = ReferenceHMM.load(path)
    assert np.array_equal(model.filter(reference).map_path, loaded.filter(reference).map_path)


def test_train_reference_hmm_returns_usable_model() -> None:
    reference = np.eye(4)
    query = np.vstack([reference[0], reference[1], reference[1], reference[2], reference[3]])
    model = train_reference_hmm(reference, [query], max_jump=2)
    result = model.filter(query)
    assert len(result.map_path) == len(query)


def test_frame_error_metrics() -> None:
    metrics = frame_error_metrics(np.array([0, 1, 3]), np.array([0, 2, 2]))
    assert metrics.mean_abs_error == 2 / 3
    assert metrics.within_1_frame == 1.0


def test_offline_dtw_returns_monotone_result() -> None:
    reference = np.eye(4)
    query = np.vstack([reference[0], reference[1], reference[1], reference[2], reference[3]])
    reference_features = FeatureSequence(
        values=reference,
        frame_times=np.arange(len(reference), dtype=float),
        sample_rate=22050,
        hop_length=512,
        feature_name="chroma_stft",
    )
    query_features = FeatureSequence(
        values=query,
        frame_times=np.arange(len(query), dtype=float),
        sample_rate=22050,
        hop_length=512,
        feature_name="chroma_stft",
    )
    result = run_offline_dtw(reference_features, query_features)
    path = result.path
    assert result.method_name == "offline_dtw"
    assert path.shape[1] == 2
    assert np.all(np.diff(path[:, 0]) >= 0)
    assert np.all(np.diff(path[:, 1]) >= 0)


def test_heldout_pairs_align_heldout_queries_to_model_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "MODEL_DIR", tmp_path)
    piece = "Piece"
    recordings = []
    for index in range(10):
        recordings.append(
            Recording(
                piece=piece,
                recording_id=f"r{index}",
                audio_path=tmp_path / f"r{index}.wav",
                beats_path=tmp_path / f"r{index}.beat",
            )
        )
    model_path = tmp_path / f"{piece}_reference2_train70_hmm.npz"
    model_path.touch()
    monkeypatch.setattr(benchmark, "_warp_factor", lambda pair: 1.0)

    pairs = benchmark._heldout_pairs(recordings)

    assert pairs
    assert {pair.reference.recording_id for pair in pairs} == {"r2"}
    assert {int(pair.query.recording_id[1:]) for pair in pairs} == {7, 8, 9}
    assert all(pair.reference.recording_id != pair.query.recording_id for pair in pairs)
    assert {pair.reference.metadata["model_path"] for pair in pairs} == {str(model_path)}


def test_least_recordings_mode_selects_smallest_eligible_piece(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "MODEL_DIR", tmp_path)
    recordings = []
    for piece, count in {
        "PieceB": 3,
        "PieceA": 3,
        "PieceC": 4,
        "NoModel": 2,
        "InvalidModel": 1,
    }.items():
        for index in range(count):
            recordings.append(
                Recording(
                    piece=piece,
                    recording_id=f"{piece}_r{index}",
                    audio_path=tmp_path / piece / f"r{index}.wav",
                    beats_path=tmp_path / piece / f"r{index}.beat",
                )
            )
    for piece in ("PieceA", "PieceB", "PieceC"):
        (tmp_path / f"{piece}_reference0_train70_hmm.npz").touch()
    (tmp_path / "InvalidModel_reference10_train70_hmm.npz").touch()
    monkeypatch.setattr(benchmark, "_warp_factor", lambda pair: 1.0)

    pairs = benchmark._heldout_pairs(recordings, mode="least_recordings")

    assert pairs
    assert {pair.piece for pair in pairs} == {"PieceA"}
    assert {pair.reference.recording_id for pair in pairs} == {"PieceA_r0"}
    assert {pair.query.recording_id for pair in pairs} == {"PieceA_r2"}


def test_hmm_train70_alignment_uses_existing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "MODEL_DIR", tmp_path)
    piece = "Piece"
    reference = np.eye(4)
    query = np.vstack([reference[0], reference[1], reference[2], reference[3]])
    model = train_reference_hmm(reference, [query], max_jump=2)
    model_path = tmp_path / f"{piece}_reference0_train70_hmm.npz"
    model.save(model_path)

    pair = RecordingPair(
        piece=piece,
        reference=Recording(piece, "ref", tmp_path / "ref.wav", tmp_path / "ref.beat"),
        query=Recording(piece, "query", tmp_path / "query.wav", tmp_path / "query.beat"),
    )

    def fake_features(recording: Recording, feature_cache: dict[Path, FeatureSequence]):
        values = reference if recording.recording_id == "ref" else query
        return FeatureSequence(
            values=values,
            frame_times=np.arange(len(values), dtype=float),
            sample_rate=22050,
            hop_length=512,
            feature_name="chroma_stft",
            metadata={"recording_id": recording.recording_id},
        )

    monkeypatch.setattr(benchmark, "_features", fake_features)
    monkeypatch.setattr(benchmark, "_reference_index", lambda reference_recording: 0)
    result = benchmark._run_pair(pair, "hmm_train70", {})

    assert result.method_name == "hmm_train70"
    assert result.metadata["model_path"] == str(model_path)
    assert result.path.shape == (len(query), 2)


def test_run_benchmark_cli_accepts_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_benchmark as cli

    calls = []

    def fake_run_benchmark(method: str, mode: str = "paper_test"):
        calls.append((method, mode))
        return [object()]

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    assert cli.main(["hmm_train70"]) == 0
    assert cli.main(["hmm_train70", "--mode", "least_recordings"]) == 0
    assert calls == [("hmm_train70", "paper_test"), ("hmm_train70", "least_recordings")]
    with pytest.raises(SystemExit):
        cli.main(["hmm_train70", "--no-save"])
