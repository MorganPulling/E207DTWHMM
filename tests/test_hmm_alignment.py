from __future__ import annotations

from pathlib import Path

import numpy as np

from hmm_alignment.dtw import dtw_pseudo_labels, local_cosine_cost
from hmm_alignment.evaluation import frame_error_metrics
from hmm_alignment.features import l2_normalize_frames
from hmm_alignment.hmm import ReferenceHMM
from hmm_alignment.training import estimate_jump_probs, train_reference_hmm


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
