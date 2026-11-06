"""One full-Train E79 RECENCY gamma update using the frozen CLS pilot recipe."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit

from src.models import cls_pos_recency_v1 as pilot


CANDIDATE_ID = "CLS_POS_RECENCY_E79_V1"
FIT_ID = "CLS_POS_RECENCY_E79_V1_E79_RECENCY"
ORIGIN = 79
HALF_LIFE_CYCLES = 8.0
WEIGHT_FLOOR = 0.25
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
BLEND_WEIGHT_POS = 0.50


def recency_base_weights_e79(cycles: Sequence[int]) -> np.ndarray:
    raw = np.asarray(cycles).reshape(-1)
    pilot.require(
        len(raw) > 0
        and raw.dtype.kind in "biuf"
        and np.isfinite(raw).all()
        and np.equal(raw, np.floor(raw)).all(),
        "E79 cycle codes must be finite integers",
    )
    cycle = raw.astype(np.int64, copy=False)
    pilot.require(
        np.all((cycle >= 47) & (cycle <= ORIGIN)),
        "E79 fit cycles must remain within official Train cycles 47-79",
    )
    age = ORIGIN - cycle
    return np.maximum(
        WEIGHT_FLOOR,
        np.power(2.0, -age.astype(np.float64) / HALF_LIFE_CYCLES),
    )


def normalized_class_weights(cycles: Sequence[int], target: Sequence[int]) -> np.ndarray:
    return pilot.normalize_class_mass(recency_base_weights_e79(cycles), target)


def fit_gamma(
    x_position: sparse.spmatrix,
    z_bag: Sequence[float],
    target: Sequence[int],
    weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Use the pilot's frozen C=1 summed-logistic L-BFGS-B optimizer."""
    return pilot.fit_weighted_gamma(x_position, z_bag, target, weights)


def positional_probability(
    x_position: sparse.spmatrix, z_bag: Sequence[float], gamma: Sequence[float]
) -> np.ndarray:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    offset = np.asarray(z_bag, dtype=np.float64).reshape(-1)
    coefficient = np.asarray(gamma, dtype=np.float64).reshape(-1)
    pilot.require(
        x.shape == (len(offset), len(coefficient))
        and np.isfinite(x.data).all()
        and np.isfinite(offset).all()
        and np.isfinite(coefficient).all(),
        "E79 positional prediction inputs are misaligned or non-finite",
    )
    return expit(offset + np.asarray(x @ coefficient, dtype=np.float64).reshape(-1))


def route_probabilities(
    present: Sequence[bool], p_a02: Sequence[float], p_pos: Sequence[float]
) -> np.ndarray:
    available = np.asarray(present, dtype=bool).reshape(-1)
    p_a = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    p_p = np.asarray(p_pos, dtype=np.float64).reshape(-1)
    pilot.require(
        len(available) == len(p_a) == len(p_p)
        and np.isfinite(p_a).all()
        and np.isfinite(p_p).all()
        and np.all((p_a >= 0.0) & (p_a <= 1.0))
        and np.all((p_p >= 0.0) & (p_p <= 1.0)),
        "E79 probability route arrays are invalid",
    )
    out = p_a.copy()
    out[available] = (
        BLEND_WEIGHT_POS * p_p[available]
        + (1.0 - BLEND_WEIGHT_POS) * p_a[available]
    )
    pilot.require(np.array_equal(out[~available], p_a[~available]),
                  "Missing-stream probabilities changed from original A02")
    return out


def apply_frozen_thresholds(
    probability: Sequence[float], present: Sequence[bool]
) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    available = np.asarray(present, dtype=bool).reshape(-1)
    pilot.require(
        len(p) == len(available)
        and np.isfinite(p).all()
        and np.all((p >= 0.0) & (p <= 1.0)),
        "E79 threshold inputs are invalid",
    )
    labels = np.zeros(len(p), dtype=np.int8)
    labels[available] = (p[available] >= PRESENT_THRESHOLD).astype(np.int8)
    labels[~available] = (p[~available] >= MISSING_THRESHOLD).astype(np.int8)
    return labels
