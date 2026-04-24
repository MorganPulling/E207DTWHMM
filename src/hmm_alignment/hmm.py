"""Reference-specific HMM model and inference routines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    """Output from causal HMM filtering."""

    map_path: np.ndarray
    mean_path: np.ndarray
    posteriors: np.ndarray | None = None


@dataclass
class ReferenceHMM:
    """A bounded-forward HMM whose states are frames of one reference recording."""

    reference_features: np.ndarray
    jump_probs: np.ndarray
    covariance: np.ndarray
    covariance_type: str = "diagonal"
    hard_start: bool = True
    variance_floor: float = 1e-6

    def __post_init__(self) -> None:
        self.reference_features = np.asarray(self.reference_features, dtype=np.float64)
        self.jump_probs = np.asarray(self.jump_probs, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)

        if self.reference_features.ndim != 2:
            raise ValueError("reference_features must have shape (R, d)")
        if self.jump_probs.ndim != 1 or len(self.jump_probs) == 0:
            raise ValueError("jump_probs must be a non-empty 1D array")
        if np.any(self.jump_probs < 0):
            raise ValueError("jump_probs must be nonnegative")
        jump_sum = self.jump_probs.sum()
        if not np.isfinite(jump_sum) or jump_sum <= 0:
            raise ValueError("jump_probs must have positive finite mass")
        self.jump_probs = self.jump_probs / jump_sum

        if self.covariance_type not in {"diagonal", "isotropic"}:
            raise ValueError("covariance_type must be 'diagonal' or 'isotropic'")
        if self.covariance_type == "diagonal":
            if self.covariance.shape != (self.reference_features.shape[1],):
                raise ValueError("diagonal covariance must have shape (d,)")
        elif self.covariance.shape not in {(), (1,)}:
            raise ValueError("isotropic covariance must be scalar")
        self.covariance = np.maximum(self.covariance, self.variance_floor)

    @property
    def num_states(self) -> int:
        return self.reference_features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.reference_features.shape[1]

    @property
    def max_jump(self) -> int:
        return len(self.jump_probs) - 1

    def log_emissions(self, query_features: np.ndarray) -> np.ndarray:
        """Compute Gaussian log emission likelihoods shaped (T, R)."""

        query = np.asarray(query_features, dtype=np.float64)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != self.feature_dim:
            raise ValueError(f"query_features must have shape (T, {self.feature_dim})")

        diff = query[:, None, :] - self.reference_features[None, :, :]
        if self.covariance_type == "diagonal":
            var = self.covariance
            mahal = np.sum((diff * diff) / var[None, None, :], axis=2)
            log_det = np.sum(np.log(var))
        else:
            var_scalar = float(np.asarray(self.covariance).reshape(-1)[0])
            mahal = np.sum(diff * diff, axis=2) / var_scalar
            log_det = self.feature_dim * np.log(var_scalar)
        return -0.5 * (self.feature_dim * np.log(2.0 * np.pi) + log_det + mahal)

    def _predict(self, alpha: np.ndarray) -> np.ndarray:
        """Predict the next posterior through bounded forward transitions."""

        pred = np.zeros_like(alpha)
        final = self.num_states - 1
        for state, mass in enumerate(alpha):
            if mass == 0:
                continue
            if state == final:
                pred[final] += mass
                continue
            for jump, prob in enumerate(self.jump_probs):
                target = min(state + jump, final)
                pred[target] += mass * prob
        return pred

    def filter(
        self,
        query_features: np.ndarray,
        return_posteriors: bool = False,
    ) -> AlignmentResult:
        """Run causal normalized filtering over query features."""

        log_emit = self.log_emissions(query_features)
        emissions = _safe_exp_normalized_rows(log_emit)
        posteriors = np.zeros_like(emissions)

        alpha = np.zeros(self.num_states, dtype=np.float64)
        if self.hard_start:
            alpha[0] = 1.0
        else:
            alpha[:] = 1.0 / self.num_states

        for t in range(emissions.shape[0]):
            if t > 0:
                alpha = self._predict(alpha)
            alpha = alpha * emissions[t]
            total = alpha.sum()
            if not np.isfinite(total) or total <= 0:
                alpha = emissions[t] / emissions[t].sum()
            else:
                alpha = alpha / total
            posteriors[t] = alpha

        states = np.arange(self.num_states, dtype=np.float64)
        map_path = np.argmax(posteriors, axis=1).astype(np.int64)
        mean_path = posteriors @ states
        return AlignmentResult(
            map_path=map_path,
            mean_path=mean_path,
            posteriors=posteriors if return_posteriors else None,
        )

    def viterbi(self, query_features: np.ndarray) -> np.ndarray:
        """Decode the best monotone state path in log space."""

        log_emit = self.log_emissions(query_features)
        num_frames, num_states = log_emit.shape
        log_jump = np.log(np.maximum(self.jump_probs, np.finfo(float).tiny))
        delta = np.full((num_frames, num_states), -np.inf, dtype=np.float64)
        back = np.zeros((num_frames, num_states), dtype=np.int64)

        if self.hard_start:
            delta[0, 0] = log_emit[0, 0]
        else:
            delta[0] = -np.log(num_states) + log_emit[0]

        for t in range(1, num_frames):
            for prev in range(num_states):
                if not np.isfinite(delta[t - 1, prev]):
                    continue
                if prev == num_states - 1:
                    score = delta[t - 1, prev] + log_emit[t, prev]
                    if score > delta[t, prev]:
                        delta[t, prev] = score
                        back[t, prev] = prev
                    continue
                for jump, log_prob in enumerate(log_jump):
                    target = min(prev + jump, num_states - 1)
                    score = delta[t - 1, prev] + log_prob + log_emit[t, target]
                    if score > delta[t, target]:
                        delta[t, target] = score
                        back[t, target] = prev

        path = np.zeros(num_frames, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(num_frames - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def save(self, path: str | Path) -> None:
        """Save this model as a compressed NumPy artifact."""

        np.savez_compressed(
            Path(path),
            reference_features=self.reference_features,
            jump_probs=self.jump_probs,
            covariance=self.covariance,
            covariance_type=np.array(self.covariance_type),
            hard_start=np.array(self.hard_start),
            variance_floor=np.array(self.variance_floor),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceHMM":
        """Load a model saved by :meth:`save`."""

        data = np.load(Path(path), allow_pickle=False)
        return cls(
            reference_features=data["reference_features"],
            jump_probs=data["jump_probs"],
            covariance=data["covariance"],
            covariance_type=str(data["covariance_type"].item()),
            hard_start=bool(data["hard_start"].item()),
            variance_floor=float(data["variance_floor"].item()),
        )


def _safe_exp_normalized_rows(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values, axis=1, keepdims=True)
    values = np.exp(shifted)
    totals = values.sum(axis=1, keepdims=True)
    return values / np.maximum(totals, np.finfo(float).tiny)
