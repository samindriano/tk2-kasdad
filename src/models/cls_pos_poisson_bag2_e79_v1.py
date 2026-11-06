"""Pure E79 wrapper for the frozen two-replica positional-gamma benchmark.

This module performs no I/O or fit on import.  Numerical fitting and policy
probability construction reuse the already-reviewed BAG2 pilot helpers; E79
identity, gamma lineage, and unchanged thresholds are pinned here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


CANDIDATE_ID = "CLS_POS_POISSON_BAG2_E79_V1"
FIT_IDS = (f"{CANDIDATE_ID}_E79_P17", f"{CANDIDATE_ID}_E79_P137")
SEEDS = (17, 137)
TRAIN_ROWS = 50_608
EXPECTED_CYCLES = tuple(range(47, 80))
EXPECTED_POSITION_SHAPE = (50_608, 17_259)
EXPECTED_POSITION_NNZ = 1_638_979
EXPECTED_POSITION_SHA256 = "32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942"
EXPECTED_BASE_SHAPE = (50_608, 20_427)
EXPECTED_BASE_NNZ = 3_855_766
EXPECTED_BASE_SHA256 = "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b"
EXPECTED_ORDERED_TRAIN_IDS_SHA256 = "9a96fa28e72b08a6022dd005aed877dadcc5e3dd5053200bcdd12711312c26b2"
ORIGINAL_GAMMA_FILE_SHA256 = "d741d016f990d8742208edc919f5c58305b7579980d059b202609df4e5b3e650"
ORIGINAL_GAMMA_ARRAY_SHA256 = "31f8bde49ca3fa85f866e7fb685d44825bd77e5cc2d11ae848494aca687ac0b3"
OFFICIAL_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
RIDGE_C = 1.0
OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}


def require(condition: bool, message: str) -> None:
    """A runtime gate that remains active under ``python -O``."""
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def fit_id_for_seed(seed: int) -> str:
    require(seed in SEEDS, f"Unfrozen E79 Poisson seed: {seed}")
    return FIT_IDS[SEEDS.index(seed)]


def load_pilot_helpers() -> Any:
    """Load the pure pilot helpers by file path without importing audit packages."""
    path = Path(__file__).resolve().with_name("cls_pos_poisson_bag2_v1.py")
    name = "_cls_pos_poisson_bag2_v1_pure_helpers"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Unable to load frozen BAG2 pilot helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(module.RIDGE_C == RIDGE_C and module.OPTIMIZER_OPTIONS == OPTIMIZER_OPTIONS,
            "Reused BAG2 optimizer settings differ from the frozen E79 recipe")
    require(module.PRESENT_THRESHOLD == PRESENT_THRESHOLD
            and module.MISSING_THRESHOLD == MISSING_THRESHOLD,
            "Reused BAG2 threshold policy differs from frozen E79 thresholds")
    return module


def validate_matrix_bundle(built: Mapping[str, Any]) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Validate the old gamma builder's parity receipt and return only fit inputs.

    The legacy builder's returned ``matrices`` mapping contains only ``base``
    and ``position``.  Its local ``checks`` dictionary is validated inside
    ``g._build_train_features`` and is intentionally not indexed here.
    """
    matrices = built.get("matrices")
    require(isinstance(matrices, Mapping), "Legacy feature builder omitted matrix parity metadata")
    require(set(matrices) == {"base", "position"},
            "Unexpected legacy matrix metadata keys; refusing implicit fallback")
    base_meta = matrices["base"]
    position_meta = matrices["position"]
    require(tuple(base_meta.get("shape", ())) == EXPECTED_BASE_SHAPE
            and base_meta.get("nnz") == EXPECTED_BASE_NNZ
            and base_meta.get("logical_sha256") == EXPECTED_BASE_SHA256,
            "Exact full-Train C2_B base matrix parity failed")
    require(tuple(position_meta.get("shape", ())) == EXPECTED_POSITION_SHAPE
            and position_meta.get("nnz") == EXPECTED_POSITION_NNZ
            and position_meta.get("logical_sha256") == EXPECTED_POSITION_SHA256,
            "Exact full-Train positional matrix parity failed")
    base = sparse.csr_matrix(built.get("base"), dtype=np.float64)
    position = sparse.csr_matrix(built.get("position"), dtype=np.float64)
    offset = np.asarray(built.get("offset"), dtype=np.float64).reshape(-1)
    require(base.shape == EXPECTED_BASE_SHAPE and position.shape == EXPECTED_POSITION_SHAPE,
            "Rebuilt full-Train matrix shapes differ from frozen E79")
    require(len(offset) == TRAIN_ROWS and np.isfinite(offset).all(),
            "Rebuilt C2_B offset is not a finite full-Train vector")
    require(np.isfinite(base.data).all() and np.isfinite(position.data).all(),
            "Rebuilt full-Train matrices contain non-finite values")
    require(built.get("ids_sha256") == EXPECTED_ORDERED_TRAIN_IDS_SHA256,
            "Ordered official Train IDs differ from the frozen E79 population")
    return base, position, offset


def extract_original_gamma(payload: Any, *, expected_length: int = 17_259) -> np.ndarray:
    require(isinstance(payload, Mapping) and "gamma" in payload,
            "The pinned CGeom full-Train artifact must expose payload['gamma']")
    gamma = np.asarray(payload["gamma"], dtype=np.float64).reshape(-1)
    require(gamma.shape == (expected_length,) and np.isfinite(gamma).all(),
            "Pinned full-champion gamma has an invalid schema")
    require(payload.get("fit_rows") == TRAIN_ROWS,
            "Pinned full-champion gamma is not attested to all official Train rows")
    return gamma


def build_probabilities(
    position: sparse.spmatrix,
    offset: Sequence[float],
    gamma_original: Sequence[float],
    gamma17: Sequence[float],
    gamma137: Sequence[float],
    p_a02: Sequence[float],
    present: Sequence[bool],
) -> dict[str, np.ndarray]:
    pilot = load_pilot_helpers()
    return pilot.build_policy_probabilities(position, offset, gamma_original,
                                            gamma17, gamma137, p_a02, present)


def apply_thresholds(probability: Sequence[float], present: Sequence[bool]) -> np.ndarray:
    return load_pilot_helpers().apply_fixed_thresholds(probability, present)
