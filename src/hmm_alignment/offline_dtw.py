"""Paper-style offline DTW baseline."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


MOVE_DIAGONAL = np.int8(0)
MOVE_QUERY_DOUBLE = np.int8(1)
MOVE_REFERENCE_DOUBLE = np.int8(2)
MOVE_START = np.int8(-1)

DIAGONAL_WEIGHT = 2.0
QUERY_DOUBLE_WEIGHT = 3.0
REFERENCE_DOUBLE_WEIGHT = 3.0


def run_offline_dtw(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    reference_times: np.ndarray,
    query_times: np.ndarray,
    reference_id: str = "reference",
    query_id: str = "query",
    metric: str = "cosine",
) -> dict[str, object]:
    """Run offline DTW and return the common benchmark result dictionary."""

    reference_values = _prepare_values(reference_values)
    query_values = _prepare_values(query_values)
    if reference_values.shape[0] == 0 or query_values.shape[0] == 0:
        raise ValueError("offline DTW requires non-empty inputs")

    metric = metric.strip().lower()
    if metric == "cosine":
        local_cost = _cosine_cost(reference_values, query_values)
    else:
        local_cost = np.ascontiguousarray(
            cdist(reference_values, query_values, metric=metric),
            dtype=np.float64,
        )
    total_cost, backpointers = _accumulate_cost(local_cost)
    path = _backtrack_path(backpointers)

    return {
        "method_name": "offline_dtw",
        "reference_id": reference_id,
        "query_id": query_id,
        "reference_times": np.asarray(reference_times, dtype=np.float64)[path[:, 0]],
        "query_times": np.asarray(query_times, dtype=np.float64)[path[:, 1]],
        "path": path,
        "metadata": {
            "distance_metric": metric,
            "total_cost": total_cost,
            "normalized_cost": total_cost / max(len(path), 1),
            "step_pattern": "paper_1-1_1-2_2-1",
        },
    }


def _prepare_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("DTW inputs must be 2D")
    return np.ascontiguousarray(values)


def _cosine_cost(reference_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    reference_norms = np.linalg.norm(reference_values, axis=1)
    query_norms = np.linalg.norm(query_values, axis=1)
    reference = reference_values.copy()
    query = query_values.copy()
    valid_reference = reference_norms > 0.0
    valid_query = query_norms > 0.0
    reference[valid_reference] /= reference_norms[valid_reference, None]
    query[valid_query] /= query_norms[valid_query, None]
    local_cost = np.ascontiguousarray(1.0 - reference @ query.T, dtype=np.float64)
    local_cost[~valid_reference, :] = np.nan
    local_cost[:, ~valid_query] = np.nan
    return local_cost


def _accumulate_cost(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
    local_cost = np.ascontiguousarray(local_cost, dtype=np.float64)
    if local_cost.ndim != 2 or 0 in local_cost.shape:
        raise ValueError("DTW local_cost must be a non-empty 2D matrix")
    return _accumulate_cost_impl(local_cost)


def _backtrack_path(backpointers: np.ndarray) -> np.ndarray:
    backpointers = np.ascontiguousarray(backpointers, dtype=np.int8)
    return _backtrack_path_impl(backpointers)


if njit is not None:

    @njit(cache=True)
    def _accumulate_cost_impl(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
        num_reference, num_query = local_cost.shape
        accumulated = np.full((num_reference, num_query), np.inf, dtype=np.float64)
        backpointers = np.full((num_reference, num_query), MOVE_START, dtype=np.int8)
        accumulated[0, 0] = DIAGONAL_WEIGHT * local_cost[0, 0]

        for reference_index in range(num_reference):
            for query_index in range(num_query):
                if reference_index == 0 and query_index == 0:
                    continue
                local_value = local_cost[reference_index, query_index]
                best_cost = np.inf
                best_move = MOVE_START

                if reference_index >= 1 and query_index >= 1:
                    candidate = accumulated[reference_index - 1, query_index - 1]
                    if np.isfinite(candidate):
                        candidate += DIAGONAL_WEIGHT * local_value
                        if candidate < best_cost:
                            best_cost = candidate
                            best_move = MOVE_DIAGONAL
                if reference_index >= 1 and query_index >= 2:
                    candidate = accumulated[reference_index - 1, query_index - 2]
                    if np.isfinite(candidate):
                        candidate += QUERY_DOUBLE_WEIGHT * local_value
                        if candidate < best_cost:
                            best_cost = candidate
                            best_move = MOVE_QUERY_DOUBLE
                if reference_index >= 2 and query_index >= 1:
                    candidate = accumulated[reference_index - 2, query_index - 1]
                    if np.isfinite(candidate):
                        candidate += REFERENCE_DOUBLE_WEIGHT * local_value
                        if candidate < best_cost:
                            best_cost = candidate
                            best_move = MOVE_REFERENCE_DOUBLE

                accumulated[reference_index, query_index] = best_cost
                backpointers[reference_index, query_index] = best_move

        return float(accumulated[-1, -1]), backpointers

    @njit(cache=True)
    def _backtrack_path_impl(backpointers: np.ndarray) -> np.ndarray:
        i = backpointers.shape[0] - 1
        j = backpointers.shape[1] - 1
        if backpointers[i, j] == MOVE_START and (i > 0 or j > 0):
            raise ValueError("No valid DTW alignment path exists.")

        path_length = 1
        while i > 0 or j > 0:
            move = backpointers[i, j]
            if move == MOVE_DIAGONAL:
                i -= 1
                j -= 1
            elif move == MOVE_QUERY_DOUBLE:
                i -= 1
                j -= 2
            elif move == MOVE_REFERENCE_DOUBLE:
                i -= 2
                j -= 1
            else:
                raise ValueError("Invalid DTW backpointer.")
            path_length += 1

        path = np.empty((path_length, 2), dtype=np.int64)
        i = backpointers.shape[0] - 1
        j = backpointers.shape[1] - 1
        path_index = path_length - 1
        path[path_index, 0] = i
        path[path_index, 1] = j
        while path_index > 0:
            move = backpointers[i, j]
            if move == MOVE_DIAGONAL:
                i -= 1
                j -= 1
            elif move == MOVE_QUERY_DOUBLE:
                i -= 1
                j -= 2
            elif move == MOVE_REFERENCE_DOUBLE:
                i -= 2
                j -= 1
            else:
                raise ValueError("Invalid DTW backpointer.")
            path_index -= 1
            path[path_index, 0] = i
            path[path_index, 1] = j
        return path

else:

    def _accumulate_cost_impl(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
        num_reference, num_query = local_cost.shape
        accumulated = np.full((num_reference, num_query), np.inf, dtype=np.float64)
        backpointers = np.full((num_reference, num_query), MOVE_START, dtype=np.int8)
        accumulated[0, 0] = DIAGONAL_WEIGHT * local_cost[0, 0]

        for reference_index in range(num_reference):
            for query_index in range(num_query):
                if reference_index == 0 and query_index == 0:
                    continue
                local_value = local_cost[reference_index, query_index]
                candidates = []
                if reference_index >= 1 and query_index >= 1:
                    candidates.append(
                        (
                            accumulated[reference_index - 1, query_index - 1]
                            + DIAGONAL_WEIGHT * local_value,
                            MOVE_DIAGONAL,
                        )
                    )
                if reference_index >= 1 and query_index >= 2:
                    candidates.append(
                        (
                            accumulated[reference_index - 1, query_index - 2]
                            + QUERY_DOUBLE_WEIGHT * local_value,
                            MOVE_QUERY_DOUBLE,
                        )
                    )
                if reference_index >= 2 and query_index >= 1:
                    candidates.append(
                        (
                            accumulated[reference_index - 2, query_index - 1]
                            + REFERENCE_DOUBLE_WEIGHT * local_value,
                            MOVE_REFERENCE_DOUBLE,
                        )
                    )
                finite = [(cost, move) for cost, move in candidates if np.isfinite(cost)]
                if finite:
                    best_cost, best_move = min(finite, key=lambda item: item[0])
                    accumulated[reference_index, query_index] = best_cost
                    backpointers[reference_index, query_index] = best_move

        return float(accumulated[-1, -1]), backpointers

    def _backtrack_path_impl(backpointers: np.ndarray) -> np.ndarray:
        i, j = np.array(backpointers.shape) - 1
        if backpointers[i, j] == MOVE_START and (i > 0 or j > 0):
            raise ValueError("No valid DTW alignment path exists.")
        path = [(int(i), int(j))]
        while i > 0 or j > 0:
            move = int(backpointers[i, j])
            if move == int(MOVE_DIAGONAL):
                i -= 1
                j -= 1
            elif move == int(MOVE_QUERY_DOUBLE):
                i -= 1
                j -= 2
            elif move == int(MOVE_REFERENCE_DOUBLE):
                i -= 2
                j -= 1
            else:
                raise ValueError("Invalid DTW backpointer.")
            path.append((int(i), int(j)))
        path.reverse()
        return np.asarray(path, dtype=np.int64)
