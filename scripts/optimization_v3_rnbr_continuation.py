"""Bounded six-fit R-NBR continuation at origins E59/E63/E71.

This runner is a fresh, versioned wrapper. It does not edit or relax the frozen
four-fit pilot. Each authorized "fit" is one past-only residual-memory build.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

# Keep numerical libraries inside the authorized two-thread resource boundary.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "2"

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_official as pilot  # noqa: E402
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS, correction_delta, predict_from_saved_neighbors,
    shuffle_residuals_within_cycle_decile, weighted_median_residual,
)

_THREAD_LIMITER = threadpool_limits(limits=2)

ORIGINS = (59, 63, 71)
ARMS = ("observed", "shuffled")
FIT_IDS = tuple(f"RNBR_CONT_E{o}_{arm.upper()}" for o in ORIGINS for arm in ARMS)
EXPECTED_ROWS = {59: 6493, 63: 7078, 71: 7462}
QUERY_CYCLES = {59: (60, 63), 63: (64, 67), 71: (72, 75)}
BASE_LEDGER_SHA = "29ec18718bf0dadae033098c1dec95136bd56d0ec32d9aa5ad13f92cd6a663a9"
DECISION_SOURCE_COMMIT = "5bbead1cd6274e28cf492919b9e4847974e8fc55"
AMENDMENT_ID = "RNBR_CONT_CAP76_2026_11_03"
RESERVATION_KEY = "REGRESSION_RNBR_CONTINUATION:THREE_ORIGIN_SIX_FIT"
CARD_ID = "REGRESSION_RNBR_CONTINUATION"
STAGE = "THREE_ORIGIN_SIX_FIT_CONTINUATION"
FREEZE = ROOT / "experiments/optimization_v3/rnbr_continuation_v1/freeze.json"
PROTOCOL = ROOT / "experiments/optimization_v3/rnbr_continuation_v1/protocol.json"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_CONTINUATION_OFFICIAL_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
TRAIN = ROOT / "data/raw/regression/Train.csv"
PILOT_FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
A02_FREEZE = ROOT / "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json"
A02_PREFLIGHT = ROOT / "reports/optimization_v3/a02_continuation_v1/preflight_v2.json"
PILOT_REPORT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json"
R06_REPORT = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json"
R06_VERIFY = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json"
R22_RESULT = ROOT / "reports/optimization_v3/final_continuation/R22/result.json"
R22_VERIFY = ROOT / "reports/optimization_v3/final_continuation/R22/verification.json"
R22_FREEZE = ROOT / "experiments/optimization_v3/final_continuation/R22/freeze.json"
GEOMETRY_FIELDS = ("score_rows", "score_record_ids", "score_cycles", "neighbor_count",
                   "neighbor_source_rows", "neighbor_ids", "neighbor_similarities")


class ContinuationError(RuntimeError):
    """Fail-closed identity, provenance, accounting, or output error."""


def _canonical(value: Any) -> bytes:
    return pilot.canonical_bytes(value)


def _sha(path: Path) -> str:
    return pilot.sha256_file(path)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _geometry_paths(origin: int, arm: str) -> dict[str, Path]:
    d = ROOT / "experiments/runs/optimization_v3_a02_continuation_v1" / f"E{origin}" / arm
    return {"npz": d / "predictions.npz", "predictions": d / "predictions.json",
            "receipt": d / "fit_receipt.json", "state": d / "fit_state.json",
            "model": d / "model.joblib"}


def _r22_paths(origin: int) -> tuple[Path, Path]:
    root = (ROOT / "experiments/runs/optimization_v3_batch1/R22" if origin in (55, 67)
            else ROOT / "experiments/runs/optimization_v3_final_continuation/R22")
    return root / f"R22_fit{origin}.npz", root / f"R22_fit{origin}.json"


def _metadata_integrity(receipt: Mapping[str, Any], field: str) -> bool:
    payload = {k: v for k, v in receipt.items() if k != field}
    return receipt.get(field) == pilot.sha256_bytes(_canonical(payload))


def _verify_geometry_artifact(origin: int, arm: str, expected_a02_sig: str) -> dict[str, Any]:
    paths = _geometry_paths(origin, arm)
    for p in paths.values():
        if not p.is_file():
            raise ContinuationError(f"Missing saved A02 artifact: {_relative(p)}")
    pred = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    rec = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    fit_id = f"A02_CONT_E{origin}_{arm.upper()}"
    if (pred.get("fit_id") != fit_id or pred.get("residual_arm") != arm
            or pred.get("origin_cycle") != origin or pred.get("freeze_signature") != expected_a02_sig
            or pred.get("query_labels_passed_to_prediction") is not False
            or pred.get("test_data_used") is not False
            or rec.get("fit_id") != fit_id or rec.get("residual_arm") != arm
            or rec.get("origin_cycle") != origin or rec.get("freeze_signature") != expected_a02_sig
            or rec.get("query_labels_passed_to_fit") is not False
            or rec.get("test_data_used") is not False
            or not _metadata_integrity(rec, "receipt_sha256")
            or state.get("state_sha256") != pilot.sha256_bytes(_canonical({k: v for k, v in state.items() if k != "state_sha256"}))
            or rec.get("state_sha256") != pilot.sha256_bytes(_canonical(state))
            or pred.get("fit_receipt_sha256") != rec.get("receipt_sha256")
            or pred.get("npz_sha256") != _sha(paths["npz"])):
        raise ContinuationError(f"A02 saved geometry provenance failed at E{origin}/{arm}")
    return {"fit_id": fit_id, "npz_sha256": _sha(paths["npz"]),
            "predictions_sha256": _sha(paths["predictions"]),
            "receipt_sha256": _sha(paths["receipt"]), "state_sha256": _sha(paths["state"]),
            "model_sha256": _sha(paths["model"]), "freeze_signature": expected_a02_sig}


def make_freeze() -> dict[str, Any]:
    """Create source/input manifest from hashes and target-free metadata only."""
    if _sha(TRAIN) != "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082":
        raise ContinuationError("Official Regression Train SHA differs from frozen lineage")
    pf = json.loads(PILOT_FREEZE.read_text(encoding="utf-8"))
    a02 = json.loads(A02_FREEZE.read_text(encoding="utf-8"))
    a02_pre = json.loads(A02_PREFLIGHT.read_text(encoding="utf-8"))
    if (a02_pre.get("status") != "A02_CONTINUATION_PREFLIGHT_PASS"
            or a02_pre.get("target_column_loaded") is not False
            or a02_pre.get("test_or_sample_loaded") is not False
            or a02.get("freeze_signature") != a02_pre.get("freeze_signature")):
        raise ContinuationError("A02 target-free preflight/freeze identity is not PASS")
    a02_sig = str(a02_pre["freeze_signature"])

    r06_inventory = pf["r06_lineage"]["cache_inventory"]
    r06 = {}
    for cutoff_text, item in sorted(r06_inventory.items(), key=lambda kv: int(kv[0])):
        cutoff = int(cutoff_text)
        if cutoff not in {55, 59, 63, 67, 71, 75}:
            continue
        npz, side = ROOT / item["npz"], ROOT / item["json"]
        if _sha(npz) != item["npz_sha256"] or _sha(side) != item["sidecar_sha256"]:
            raise ContinuationError(f"Frozen R06 cache changed at cutoff {cutoff}")
        r06[str(cutoff)] = {"npz": item["npz"], "sidecar": item["json"],
                            "npz_sha256": item["npz_sha256"],
                            "sidecar_sha256": item["sidecar_sha256"],
                            "freeze_signature": item["freeze_signature"]}
    if set(map(int, r06)) != {55, 59, 63, 67, 71, 75}:
        raise ContinuationError("R06 cache inventory is incomplete for declared origins")

    geometry: dict[str, Any] = {}
    for origin in ORIGINS:
        geometry[str(origin)] = {}
        for arm in ARMS:
            geometry[str(origin)][arm] = _verify_geometry_artifact(origin, arm, a02_sig)

    r22 = {}
    r22_result = json.loads(R22_RESULT.read_text(encoding="utf-8"))
    r22_verify = json.loads(R22_VERIFY.read_text(encoding="utf-8"))
    r22_freeze = json.loads(R22_FREEZE.read_text(encoding="utf-8"))
    if (r22_result.get("status") != "RESULT_VERIFIED" or r22_verify.get("status") != "PASS"
            or r22_result.get("freeze_signature") != r22_freeze.get("signature")
            or r22_verify.get("freeze_signature") != r22_result.get("freeze_signature")):
        raise ContinuationError("R22_ORIGINAL comparator is not backed by its verified frozen report")
    for origin in (55, 59, 63, 67, 71, 75):
        npz, side = _r22_paths(origin)
        if not npz.is_file() or not side.is_file():
            raise ContinuationError(f"Missing verified R22 comparison cache at origin {origin}")
        # The final continuation report pins E59/E63/E71/E75; the earlier pilot freeze
        # pins E55/E67. The sidecar's own identity and integrity digests are also checked.
        if origin in (59, 63, 71, 75):
            receipt = r22_result.get("fit_receipts", {}).get(str(origin), {})
            if receipt.get("npz_sha256") != _sha(npz) or receipt.get("sidecar_sha256") != _sha(side):
                raise ContinuationError(f"R22 final-continuation receipt differs at origin {origin}")
        else:
            pilot_spec = pf.get("evaluation_references", {}).get("R22_W4" if origin == 67 else "R22_W1")
            if not pilot_spec or pilot_spec.get("npz_sha256") != _sha(npz) or pilot_spec.get("sidecar_sha256") != _sha(side):
                # E55 is verified by the final R22 freeze below when the pilot report has no W1 entry.
                if origin != 55:
                    raise ContinuationError(f"R22 pilot freeze differs at origin {origin}")
                frozen_r22 = r22_freeze["identity"].get("input_sha256", {})
                if (frozen_r22.get("R22_pilot55_npz") != _sha(npz)
                        or frozen_r22.get("R22_pilot55_sidecar") != _sha(side)):
                    raise ContinuationError("R22 E55 comparator differs from frozen R22 pilot input")
        sidecar = pilot._verify_sidecar(side, npz, expected_npz_hash=_sha(npz), card="R22", cutoff=origin)
        r22[str(origin)] = {"npz": _relative(npz), "sidecar": _relative(side),
                            "npz_sha256": _sha(npz), "sidecar_sha256": _sha(side),
                            "identity_sha256": sidecar.get("identity_sha256"),
                            "prediction_array": "R22_ORIGINAL" if origin in (59, 63, 71, 75) else "prediction"}

    code_paths = [
        "scripts/optimization_v3_rnbr_continuation.py",
        "scripts/optimization_v3_rnbr_continuation_verify.py",
        "src/models/regression_neighbor_rnbr.py",
        "scripts/optimization_v3_rnbr_official.py",
        "scripts/optimization_v3_rnbr_official_verify.py",
        "tests/test_optimization_v3_rnbr_continuation.py",
        "scripts/optimization_v3_rnbr_continuation_synthetic.py",
        "experiments/optimization_v3/rnbr_continuation_v1/protocol.json",
    ]
    code = {rel: _sha(ROOT / rel) for rel in code_paths}
    evidence_paths = [
        "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json",
        "reports/optimization_v3/rnbr_pilot_v1/PROTOCOL.md",
        "reports/optimization_v3/rnbr_pilot_v1/PREPARATION_PREFLIGHT.json",
        "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/RESULTS.md",
        "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json",
        "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json",
        "experiments/optimization_v3/a02_continuation_v1/protocol.json",
        "reports/optimization_v3/a02_continuation_v1/preflight_v2.json",
        "reports/optimization_v3/a02_continuation_v1/verification.json",
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json",
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json",
        "reports/optimization_v3/final_continuation/R22/result.json",
        "reports/optimization_v3/final_continuation/R22/verification.json",
        "experiments/optimization_v3/final_continuation/R22/freeze.json",
        "reports/optimization_v3/rnbr_continuation_v1/pre_execution_qa.json",
    ]
    inputs = {rel: _sha(ROOT / rel) for rel in evidence_paths}
    inputs["data/raw/regression/Train.csv"] = _sha(TRAIN)

    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "RNBR_THREE_ORIGIN_SIX_FIT_CONTINUATION_V1",
        "authorized_base_commit": DECISION_SOURCE_COMMIT,
        "official_train": {"path": "data/raw/regression/Train.csv", "rows": 50608,
                           "sha256": _sha(TRAIN), "columns_opened_at_preflight": ["record_id", "cycle_code", "signal_stream"],
                           "target_fit_scope": "present-stream Regression Train donor rows only, cycles 56..origin"},
        "pilot_freeze_signature": pf["freeze_signature"],
        "a02_freeze_signature": a02_sig,
        "r06_cache_inventory": r06,
        "a02_geometry_inventory": geometry,
        "r22_comparison_inventory": r22,
        "historical_r06": {"results": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json",
                           "verification": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json"},
        "origins": list(ORIGINS), "arms": list(ARMS), "fit_ids": list(FIT_IDS), "fit_count": 6,
        "fit_semantics": "one residual-memory build per origin and arm; no estimator refit",
        "recipe": {"donor_cycles": [56, "origin inclusive"], "donor_present_stream_only": True,
                   "donor_r06_oof_cutoff": "frozen per-cycle cutoff strictly before donor cycle",
                   "query_baseline": "R06 prediction cache at exact origin for all query rows",
                   "neighbors": 32, "similarity_weight": "cosine squared", "min_neighbors": 5,
                   "weighted_median_tie": "midpoint at exact half mass; residual then source row stable order",
                   "shrinkage": 0.25, "delta_clip": [-5.0, 5.0], "prediction_clip": [0.0, 100.0],
                   "missing_or_under_supported": "unchanged exact R06 origin prediction",
                   "shuffle": "within exact cycle x fixed R06 prediction decile; default_rng(42) per origin; source row stable order"},
        "query_windows": {"59": {"cycles": [60, 63], "rows": EXPECTED_ROWS[59]},
                          "63": {"cycles": [64, 67], "rows": EXPECTED_ROWS[63]},
                          "71": {"cycles": [72, 75], "rows": EXPECTED_ROWS[71]}},
        "evaluation_views": {"W1": {"cycles": [56, 59], "rows": 4114, "recipe": "exact R06 fallback"},
                             "W2": {"origin": 59, "cycles": [60, 63], "rows": 6493},
                             "W3": {"origin": 63, "cycles": [64, 67], "rows": 7078},
                             "W4": {"origin": 67, "cycles": [68, 71], "rows": 4950, "recipe": "verified inherited pilot"},
                             "W5": {"origin": 71, "cycles": [72, 75], "rows": 7462},
                             "W6": {"origin": 75, "cycles": [76, 79], "rows": 8491, "recipe": "verified inherited pilot"},
                             "new_origin_aggregate": {"windows": ["W2", "W3", "W5"], "rows": 21033},
                             "full_primary": {"windows": ["W1", "W2", "W3", "W4", "W5", "W6"], "rows": 38588},
                             "late_W5_W6": {"windows": ["W5", "W6"], "rows": 15953},
                             "inherited_H12_L": {"origin": 67, "cycles": [68, 79], "rows": 20903,
                                                 "source": "unchanged existing E67 pilot outputs"}},
        "prospective_gates": {"full_primary_gain_vs_R06_min": 0.03,
                              "full_primary_gain_vs_shuffle_min": 0.01,
                              "new_origin_gain_vs_R06_min": 0.0,
                              "new_origin_gain_vs_shuffle_min": 0.0,
                              "learned_primary_windows_nonnegative_min": 3,
                              "late_gain_vs_R06_strictly_positive": True,
                              "H12_L_harm_vs_R06_max": 0.05,
                              "per_learned_window_harm_vs_R06_max_exclusive": 0.25,
                              "supported_group_min_n": 100,
                              "supported_tail_or_missing_harm_max": 0.25,
                              "missing_incremental_change_exactly_zero": True},
        "code_sha256": code, "evidence_sha256": inputs,
        "source_boundary": {"test": "not opened", "sample_submission": "not opened", "kaggle": "not accessed",
                            "classification_labels": "not opened", "query_targets": "not opened before six durable verified fit bundles",
                            "only_official_regression_train_donor_targets": True},
        "resources": {"concurrent_fit_jobs": 1, "threads": 2, "job_rss_cap_bytes": 6442450944,
                      "free_memory_reserve_bytes": 2147483648},
    }
    body["freeze_signature"] = pilot.sha256_bytes(_canonical(body))
    return body


def write_freeze() -> dict[str, Any]:
    freeze = make_freeze()
    rendered = json.dumps(freeze, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if FREEZE.exists():
        current = json.loads(FREEZE.read_text(encoding="utf-8"))
        if current == freeze:
            return freeze
        # Before the authorization ledger is amended, permit correcting this
        # task-owned generated freeze. Once any continuation event/output exists,
        # the manifest is immutable and no rewrite is allowed.
        if _sha(LEDGER) != BASE_LEDGER_SHA or (RUN_ROOT / "bundles").exists():
            raise ContinuationError("Existing continuation freeze is immutable after reservation or fit output")
        staging = FREEZE.with_name(FREEZE.name + ".staging")
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered); stream.flush(); os.fsync(stream.fileno())
        os.replace(staging, FREEZE)
    else:
        with FREEZE.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered); stream.flush(); os.fsync(stream.fileno())
    return freeze


def load_freeze(*, require_clean: bool = False) -> dict[str, Any]:
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    body = {k: v for k, v in data.items() if k != "freeze_signature"}
    if data.get("freeze_signature") != pilot.sha256_bytes(_canonical(body)) or data != make_freeze():
        raise ContinuationError("Continuation freeze signature or current source/input hashes differ")
    if require_clean and pilot._run_git("status", "--porcelain", "--untracked-files=all"):
        raise ContinuationError("Official fit requires committed preparation/freeze and a clean working tree")
    return data


def targetfree_train() -> dict[str, np.ndarray]:
    return pilot.load_targetfree_train()


def load_r06(freeze: Mapping[str, Any], train: Mapping[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    return pilot.load_r06_predictions(train, json.loads(PILOT_FREEZE.read_text(encoding="utf-8")))


def load_geometry(origin: int, freeze: Mapping[str, Any], train: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    a02_sig = str(freeze["a02_freeze_signature"])
    for arm in ARMS:
        paths = _geometry_paths(origin, arm)
        _verify_geometry_artifact(origin, arm, a02_sig)
        with np.load(paths["npz"], allow_pickle=False) as z:
            missing = set(GEOMETRY_FIELDS) - set(z.files)
            if missing:
                raise ContinuationError(f"A02 geometry missing fields at E{origin}/{arm}: {sorted(missing)}")
            # Only target-free geometry and identity arrays are touched; NPZ labels remain unopened.
            loaded[arm] = {k: np.asarray(z[k]).copy() for k in GEOMETRY_FIELDS}
    for field in GEOMETRY_FIELDS:
        if not np.array_equal(loaded["observed"][field], loaded["shuffled"][field]):
            raise ContinuationError(f"A02 arms do not share exact geometry E{origin}/{field}")
    g = loaded["observed"]
    rows = np.asarray(g["score_rows"], dtype=np.int64)
    ids = np.asarray(g["score_record_ids"]).astype(str)
    cycles = np.asarray(g["score_cycles"], dtype=np.int64)
    low, high = QUERY_CYCLES[origin]
    expected = np.flatnonzero((train["cycles"] >= low) & (train["cycles"] <= high)).astype(np.int64)
    if (len(rows) != EXPECTED_ROWS[origin] or not np.array_equal(rows, expected)
            or not np.array_equal(ids, train["ids"][rows])
            or not np.array_equal(cycles, train["cycles"][rows])):
        raise ContinuationError(f"A02 query rows/IDs/cycles differ from E{origin} declared horizon")
    present = train["present"][rows]
    counts = np.asarray(g["neighbor_count"], dtype=np.int64)
    if np.any((counts == 0) != ~present):
        raise ContinuationError(f"A02 missing-stream geometry differs from Train at E{origin}")
    return g


def _canonical_neighbor_id(value: Any) -> str:
    return f"str_:{str(value)}"


def validate_geometry(origin: int, train: Mapping[str, np.ndarray], geometry: Mapping[str, np.ndarray],
                      donor_rows: np.ndarray) -> dict[str, int]:
    cycles, rows = train["cycles"], np.asarray(geometry["score_rows"], dtype=np.int64)
    donor_ids = train["ids"][donor_rows]
    donor_cycles = cycles[donor_rows]
    donor_lookup = {int(r): i for i, r in enumerate(donor_rows)}
    qids = np.asarray(geometry["score_record_ids"]).astype(str)
    qcycles = np.asarray(geometry["score_cycles"], dtype=np.int64)
    nr = np.asarray(geometry["neighbor_source_rows"], dtype=np.int64)
    ni = np.asarray(geometry["neighbor_ids"]).astype(str)
    sim = np.asarray(geometry["neighbor_similarities"], dtype=np.float64)
    counts = np.asarray(geometry["neighbor_count"], dtype=np.int64)
    if len(np.unique(donor_rows)) != len(donor_rows) or len(np.unique(donor_ids)) != len(donor_ids):
        raise ContinuationError("Donor row positions and IDs must be unique")
    if len(np.unique(rows)) != len(rows) or len(np.unique(qids)) != len(qids):
        raise ContinuationError("Query row positions and IDs must be unique")
    if np.intersect1d(donor_rows, rows).size or set(donor_ids).intersection(qids):
        raise ContinuationError("Donor and query row/ID identities overlap")
    if (nr.ndim != 2 or nr.shape[0] != len(rows) or nr.shape[1] != 32 or ni.shape != nr.shape
            or sim.shape != nr.shape or counts.shape != (len(rows),) or np.any(counts < 0) or np.any(counts > 32)):
        raise ContinuationError("Saved geometry shapes/counts differ from frozen top32")
    if not np.isfinite(sim).all() or np.any((sim < 0.0) | (sim > 1.0 + 1e-12)):
        raise ContinuationError("Saved geometry cosine values invalid")
    if np.any((donor_cycles < 56) | (donor_cycles > origin)) or not np.all(train["present"][donor_rows]):
        raise ContinuationError("Donor bank is outside present-stream cycles 56..origin")
    if np.any(qcycles <= origin) or np.any(qcycles > origin + 4):
        raise ContinuationError("Query horizon is not the declared next four cycles")
    cutoffs = np.asarray([EXPECTED_SOURCE_CUTOFFS[int(c)] for c in donor_cycles], dtype=np.int64)
    if np.any(cutoffs >= donor_cycles):
        raise ContinuationError("A donor R06 OOF cutoff is not strictly before its cycle")
    checked = 0
    missing = ~train["present"][rows]
    for i, k0 in enumerate(counts):
        k = int(k0)
        if (np.any(nr[i, k:] != -1) or np.any(ni[i, k:] != "") or np.any(sim[i, k:] != 0.0)
                or (missing[i] and k != 0)):
            raise ContinuationError("Geometry padding/missing-stream structure is invalid")
        if not k:
            continue
        active = nr[i, :k]
        if (np.any(active < 0) or len(np.unique(active)) != k or np.any(sim[i, :k] <= 0.0)
                or sorted(range(k), key=lambda j: (-sim[i, j], active[j])) != list(range(k))):
            raise ContinuationError("Neighbor uniqueness, positivity, or tie order failed")
        for j, source_row in enumerate(active):
            at = donor_lookup.get(int(source_row))
            if at is None or ni[i, j] != _canonical_neighbor_id(donor_ids[at]):
                raise ContinuationError("Neighbor row/typed ID does not map to the frozen present donor")
            if donor_cycles[at] >= qcycles[i]:
                raise ContinuationError("Neighbor source is not strictly past its query cycle")
            checked += 1
    return {"donor_rows": len(donor_rows), "query_rows": len(rows), "neighbor_rows_checked": checked,
            "missing_queries": int(missing.sum()), "positive_present_queries": int(np.sum((~missing) & (counts > 0)))}


def _r06_aligned(rows: np.ndarray, cycles: np.ndarray, r06: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    # OOF prediction source for donors: fixed cutoff per cycle, strictly earlier than that cycle.
    out = np.empty(len(rows), dtype=np.float64)
    cutoffs = np.asarray([EXPECTED_SOURCE_CUTOFFS[int(c)] for c in cycles], dtype=np.int64)
    for cutoff in np.unique(cutoffs):
        ix = np.flatnonzero(cutoffs == cutoff)
        source = r06[int(cutoff)]
        lookup = {int(r): float(v) for r, v in zip(source["rows"], source["prediction"])}
        try:
            out[ix] = [lookup[int(rows[i])] for i in ix]
        except KeyError as exc:
            raise ContinuationError(f"R06 donor OOF cutoff {cutoff} misses row {exc.args[0]}") from exc
    if not np.isfinite(out).all():
        raise ContinuationError("Nonfinite donor OOF predictions")
    return out


def _origin_baseline(origin: int, query_rows: np.ndarray, r06: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    # Query baseline semantics are distinct from donor OOF alignment: use the full origin cache.
    source = r06[origin]
    lookup = {int(r): float(v) for r, v in zip(source["rows"], source["prediction"])}
    try:
        values = np.asarray([lookup[int(r)] for r in query_rows], dtype=np.float64)
    except KeyError as exc:
        raise ContinuationError(f"R06 origin-{origin} cache misses query row {exc.args[0]}") from exc
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ContinuationError("R06 origin query baseline invalid")
    return values


def build_payload(origin: int, arm: str, train: Mapping[str, np.ndarray],
                  r06: Mapping[int, Mapping[str, np.ndarray]], geometry: Mapping[str, np.ndarray],
                  donor_truth: Mapping[str, np.ndarray], freeze: Mapping[str, Any], source_commit: str
                  ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if origin not in ORIGINS or arm not in ARMS:
        raise ContinuationError("Only authorized E59/E63/E71 observed/shuffled IDs are permitted")
    donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin) & train["present"]).astype(np.int64)
    donor_cycles = train["cycles"][donor_rows]
    donor_ids = train["ids"][donor_rows]
    query_rows = np.asarray(geometry["score_rows"], dtype=np.int64)
    query_cycles = np.asarray(geometry["score_cycles"], dtype=np.int64)
    query_ids = np.asarray(geometry["score_record_ids"]).astype(str)
    missing = ~train["present"][query_rows]
    checks = validate_geometry(origin, train, geometry, donor_rows)
    y = np.asarray(donor_truth["target"], dtype=np.float64)
    if (not np.array_equal(donor_truth["rows"], donor_rows)
            or not np.array_equal(donor_truth["ids"].astype(str), donor_ids)
            or not np.array_equal(donor_truth["cycles"].astype(np.int64), donor_cycles)
            or y.shape != donor_rows.shape or not np.isfinite(y).all() or np.any((y < 0) | (y > 100))):
        raise ContinuationError("Donor targets do not exactly align with present-stream bank")
    donor_r06 = _r06_aligned(donor_rows, donor_cycles, r06)
    original_residuals = y - donor_r06
    used = original_residuals.copy() if arm == "observed" else shuffle_residuals_within_cycle_decile(
        original_residuals, donor_cycles, donor_r06, donor_rows, seed=42)
    baseline = _origin_baseline(origin, query_rows, r06)
    prediction = predict_from_saved_neighbors(
        donor_rows=donor_rows, donor_residuals=used, query_r06_predictions=baseline,
        query_missing=missing, neighbor_rows=geometry["neighbor_source_rows"],
        neighbor_cosines=geometry["neighbor_similarities"], neighbor_counts=geometry["neighbor_count"])
    fit_id = f"RNBR_CONT_E{origin}_{arm.upper()}"
    arrays = {
        "score_rows": query_rows, "score_record_ids": query_ids, "score_cycles": query_cycles,
        "prediction": prediction["prediction"], "baseline_prediction": baseline,
        "delta": prediction["delta"], "query_missing": missing,
        "neighbor_count_used": prediction["neighbor_count_used"],
        "neighbor_count": np.asarray(geometry["neighbor_count"], dtype=np.int16),
        "neighbor_source_rows": np.asarray(geometry["neighbor_source_rows"], dtype=np.int64),
        "neighbor_ids": np.asarray(geometry["neighbor_ids"]).astype(str),
        "neighbor_similarities": np.asarray(geometry["neighbor_similarities"], dtype=np.float64),
        "donor_rows": donor_rows, "donor_record_ids": donor_ids.astype(str),
        "donor_cycles": donor_cycles, "donor_source_cutoffs": np.asarray(
            [EXPECTED_SOURCE_CUTOFFS[int(c)] for c in donor_cycles], dtype=np.int64),
        "donor_r06_predictions": donor_r06, "donor_residuals": original_residuals,
        "donor_residuals_used": used,
    }
    if any(np.asarray(v).dtype.kind == "O" for v in arrays.values()):
        raise ContinuationError("Object arrays are prohibited in continuation bundles")
    identity = {"schema_version": 1, "fit_id": fit_id, "origin": origin, "arm": arm,
                "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
                "dataset_tag": "OFFICIAL_REGRESSION_TRAIN_DONOR_ONLY",
                "formula": "clip(R06_origin + clip(0.25 * weighted_median(R06_OOF_residual, cosine^2), -5, +5), 0, 100)",
                "donor_rows": len(donor_rows), "query_rows": len(query_rows),
                "query_cycles": [int(query_cycles.min()), int(query_cycles.max())],
                "query_labels_read": False, "test_or_sample_submission_opened": False,
                "classification_labels_used": False, "new_estimator_fit_count": 0,
                "memory_construction_fit_count": 1}
    audit = {"target_rows_read": len(donor_rows), "target_values_sha256": pilot.sha256_array(y, "<f8"),
             "donor_residual_sha256": pilot.sha256_array(original_residuals, "<f8"),
             "assigned_residual_sha256": pilot.sha256_array(used, "<f8"),
             "origin_baseline_sha256": pilot.sha256_array(baseline, "<f8"),
             "geometry_check": checks, "fallback_rows": int(np.sum(missing | (np.asarray(geometry["neighbor_count"]) < 5))),
             "missing_rows_unchanged": int(missing.sum()),
             "resources": {"threads": 2, "rss_cap_bytes": 6442450944,
                           "rss_bytes_after_build": int(psutil.Process().memory_info().rss)}}
    return arrays, {"identity": identity, "audit": audit}


def _fit_dir(fit_id: str) -> Path:
    if fit_id not in FIT_IDS:
        raise ContinuationError(f"Unexpected fit ID: {fit_id}")
    return RUN_ROOT / "bundles" / fit_id


def persist_bundle(fit_id: str, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> dict[str, Any]:
    dst = _fit_dir(fit_id)
    staging = dst.with_name(dst.name + ".staging")
    if dst.exists() or staging.exists():
        raise ContinuationError(f"Refusing to overwrite or rebuild an existing output: {dst}")
    staging.mkdir(parents=True, exist_ok=False)
    npz = staging / "arrays.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema_version": 1, "identity": dict(meta["identity"]),
               "identity_sha256": pilot.sha256_bytes(_canonical(meta["identity"])),
               "npz_sha256": _sha(npz), "metadata": dict(meta["audit"]),
               "array_shapes": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
               "array_dtypes": {k: str(np.asarray(v).dtype) for k, v in arrays.items()}}
    receipt["integrity_sha256"] = pilot.sha256_bytes(_canonical(receipt))
    with (staging / "receipt.json").open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    with (staging / "fit_state.json").open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"fit_id": fit_id, "freeze_signature": meta["identity"]["freeze_signature"],
                            "state": "DURABLE_PENDING_INDEPENDENT_VERIFICATION"}, indent=2) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.rename(staging, dst)
    return receipt


def load_bundle(fit_id: str, freeze: Mapping[str, Any], *, source_commit: str | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folder = _fit_dir(fit_id)
    npz, rec_path = folder / "arrays.npz", folder / "receipt.json"
    if not folder.is_dir() or not npz.is_file() or not rec_path.is_file():
        raise ContinuationError(f"Missing or incomplete continuation bundle: {fit_id}")
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    identity = receipt.get("identity", {})
    body = {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    if (receipt.get("integrity_sha256") != pilot.sha256_bytes(_canonical(body))
            or receipt.get("identity_sha256") != pilot.sha256_bytes(_canonical(identity))
            or receipt.get("npz_sha256") != _sha(npz)
            or identity.get("fit_id") != fit_id
            or identity.get("freeze_signature") != freeze["freeze_signature"]
            or (source_commit and identity.get("source_commit") != source_commit)):
        raise ContinuationError(f"Continuation receipt/hash/identity mismatch: {fit_id}")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    if (any(v.dtype.kind == "O" for v in arrays.values())
            or receipt.get("array_shapes") != {k: list(v.shape) for k, v in arrays.items()}
            or receipt.get("array_dtypes") != {k: str(v.dtype) for k, v in arrays.items()}
            or not np.isfinite(arrays["prediction"]).all()
            or np.any((arrays["prediction"] < 0) | (arrays["prediction"] > 100))):
        raise ContinuationError(f"Continuation arrays fail safe dtype/shape/value checks: {fit_id}")
    return arrays, receipt


def _read_events() -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError) as exc:
        raise ContinuationError("Shared budget ledger cannot be read as intact JSONL") from exc


def _snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    auth = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    amendments = [e for e in events if e.get("event") in {"BUDGET_CAP_AMENDMENT", "BUDGET_CATEGORY_AMENDMENT"}]
    if len(auth) != 1:
        raise ContinuationError("Expected exactly one shared allowance authorization")
    last = amendments[-1] if amendments else {}
    caps = dict(last.get("effective_caps", auth[0]["category_caps"]))
    cap = int(last.get("global_cap", auth[0]["cap"]))
    reserves = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    released = {e.get("reservation_key"): int(e.get("unused_fits", 0)) for e in events
                if e.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active_cat = {c: sum(int(e.get("fits", 0)) - released.get(e.get("reservation_key"), 0)
                         for e in reserves if e.get("category") == c) for c in caps}
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    pending = [s for s in starts if not any(e.get("fit_id") == s.get("fit_id")
                                              and e.get("event") in {"FIT_SAVED", "FIT_FAILED"} for e in events)]
    return {"events": len(events), "global_cap": cap, "caps": caps,
            "active_by_category": active_cat, "active_reserved": sum(active_cat.values()),
            "started": len(starts), "saved": len(saved), "failed": len(failed), "pending": len(pending)}


def _append_event(event: Mapping[str, Any]) -> None:
    value = dict(event)
    value.setdefault("at_utc", pd.Timestamp.utcnow().isoformat())
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(LEDGER, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def amend_and_reserve(freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    events = _read_events(); before_hash = _sha(LEDGER); snap = _snapshot(events)
    expected_before_caps = {"pilot": 38, "continuation": 31, "control_replay": 0, "technical_retry": 1}
    if before_hash == BASE_LEDGER_SHA:
        if (snap["global_cap"] != 70 or snap["caps"] != expected_before_caps
                or snap["active_reserved"] != 70 or snap["started"] != 70 or snap["saved"] != 67
                or snap["failed"] != 3 or snap["pending"] != 0):
            raise ContinuationError(f"Live pre-amendment accounting differs from Sol checkpoint: {snap}")
        _append_event({"event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
                       "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
                       "authorized_by": "Sol decision; Samuel-approved six-fit continuation",
                       "prior_global_cap": 70, "global_cap": 76,
                       "prior_effective_caps": expected_before_caps,
                       "effective_caps": {"pilot": 38, "continuation": 37, "control_replay": 0, "technical_retry": 1},
                       "active_reservations_before": snap["active_by_category"],
                       "active_total_before": snap["active_reserved"],
                       "historical_started_saved_failed_before": {"started": 70, "saved": 67, "failed": 3},
                       "pre_amendment_event_rows": snap["events"], "pre_amendment_ledger_sha256": before_hash,
                       "source_commit": source_commit, "freeze_signature": freeze["freeze_signature"],
                       "reason": "Exactly six authorized continuation fits at origins 59, 63, and 71; no refund/retry."})
        events = _read_events(); snap = _snapshot(events)
    else:
        caps_events = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
        if len(caps_events) != 1:
            raise ContinuationError(f"Ledger differs from checkpoint without unique matching amendment: {before_hash}")
        amendment = caps_events[0]
        if (amendment.get("global_cap") != 76 or amendment.get("effective_caps") !=
                {"pilot": 38, "continuation": 37, "control_replay": 0, "technical_retry": 1}
                or amendment.get("freeze_signature") != freeze["freeze_signature"]):
            raise ContinuationError("Existing continuation cap amendment differs from frozen recipe")
    if snap["global_cap"] != 76 or snap["caps"] != {"pilot": 38, "continuation": 37, "control_replay": 0, "technical_retry": 1}:
        raise ContinuationError(f"Effective six-fit cap not applied exactly: {snap}")
    reserved = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(reserved) > 1:
        raise ContinuationError("Duplicate six-fit reservation key")
    if not reserved:
        if (snap["active_reserved"] != 70 or snap["started"] != 70 or snap["pending"] != 0
                or snap["active_by_category"].get("continuation") != 31):
            raise ContinuationError(f"Pre-reservation ledger state differs: {snap}")
        _append_event({"event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
                       "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                       "category": "continuation", "fits": 6, "fit_ids": list(FIT_IDS),
                       "amendment_id": AMENDMENT_ID, "global_cap": 76, "effective_category_cap": 37,
                       "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit})
        events = _read_events()
    else:
        if (reserved[0].get("fits") != 6 or reserved[0].get("fit_ids") != list(FIT_IDS)
                or reserved[0].get("amendment_id") != AMENDMENT_ID
                or reserved[0].get("freeze_signature") != freeze["freeze_signature"]):
            raise ContinuationError("Existing six-fit reservation differs")
    snap = _snapshot(events)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("reservation_key") == RESERVATION_KEY]
    if (snap["global_cap"] != 76 or snap["active_reserved"] != 76 or snap["started"] != 70
            or snap["pending"] != 0 or len(starts) != len(set(e.get("fit_id") for e in starts))):
        raise ContinuationError(f"Post-reservation budget state differs: {snap}")
    return {"snapshot": snap, "ledger_sha256": _sha(LEDGER)}


def _start(fit_id: str, origin: int, arm: str, freeze: Mapping[str, Any], source_commit: str) -> None:
    events = _read_events()
    if _snapshot(events)["pending"]:
        raise ContinuationError("A different shared-ledger fit is pending; no concurrent work allowed")
    if any(e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id for e in events):
        raise ContinuationError(f"Fit ID already started; no retry/replay is authorized: {fit_id}")
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    snap = _snapshot(events)
    if (len(reservation) != 1 or fit_id not in reservation[0].get("fit_ids", [])
            or snap["started"] >= snap["global_cap"] or snap["active_by_category"].get("continuation", 0) > 37):
        raise ContinuationError("Authorized fit slot/cap unavailable")
    count = sum(e.get("event") == "FIT_STARTED" and e.get("reservation_key") == RESERVATION_KEY for e in events)
    if count >= 6:
        raise ContinuationError("Six authorized starts already consumed")
    _append_event({"event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
                   "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                   "fit_id": fit_id, "cutoff": origin, "component": f"{arm}_residual_memory",
                   "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
                   "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit})


def _finish(fit_id: str, status: str, receipt_sha: str | None = None, error: str | None = None) -> None:
    events = _read_events()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    if len(starts) != 1:
        raise ContinuationError(f"Missing or duplicate start event for {fit_id}")
    if any(e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"} for e in events):
        raise ContinuationError(f"Fit already has a terminal event: {fit_id}")
    if status not in {"FIT_SAVED", "FIT_FAILED"}:
        raise ContinuationError("Invalid ledger terminal event")
    _append_event({"event": status, "batch_id": "optimization_v3_research_next_v1",
                   "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                   "fit_id": fit_id, "cutoff": starts[0]["cutoff"],
                   "receipt_sha256": receipt_sha, "error": error})


def resource_preflight() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    if vm.available < 2147483648:
        raise ContinuationError(f"Free-memory reserve below 2 GiB: {vm.available}")
    competing = []
    own = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info.get("pid") == own:
            continue
        cmd = " ".join(proc.info.get("cmdline") or []).lower()
        name = str(proc.info.get("name") or "").lower()
        active_markers = ("optimization_v3_rnbr_continuation.py --run-all",
                          "optimization_v3_rnbr_official.py --run",
                          "optimization_v3_rnbr_serialization_recovery.py")
        if name.startswith("python") and any(marker in cmd for marker in active_markers):
            competing.append({"pid": proc.info["pid"], "cmdline": cmd[:200]})
    if competing:
        raise ContinuationError(f"Another project fit process is active: {competing}")
    pools = threadpool_info()
    if any(int(pool.get("num_threads", 0)) > 2 for pool in pools):
        raise ContinuationError(f"A numerical thread pool exceeds the authorized two-thread limit: {pools}")
    return {"available_bytes": vm.available, "job_rss_cap_bytes": 6442450944,
            "reserved_free_bytes": 2147483648, "threads": 2,
            "threadpools": pools, "active_competitors": competing}


def _verify_fit_independent(fit_id: str, freeze: Mapping[str, Any], train: Mapping[str, np.ndarray],
                            r06: Mapping[int, Mapping[str, np.ndarray]], source_commit: str) -> dict[str, Any]:
    # Independent verifier is deliberately a separate module with an independent formula implementation.
    from scripts.optimization_v3_rnbr_continuation_verify import verify_fit_bundle
    arrays, receipt = load_bundle(fit_id, freeze)
    return verify_fit_bundle(fit_id, arrays, receipt, freeze, train, r06, source_commit)


def run_fit(origin: int, arm: str, freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    fit_id = f"RNBR_CONT_E{origin}_{arm.upper()}"
    if fit_id not in FIT_IDS:
        raise ContinuationError(f"Fit ID outside explicit authorization: {fit_id}")
    resource_preflight()
    train = targetfree_train()
    r06 = load_r06(freeze, train)
    geometry = load_geometry(origin, freeze, train)
    donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin) & train["present"]).astype(np.int64)
    _start(fit_id, origin, arm, freeze, source_commit)
    try:
        donor = pilot.read_regression_targets(donor_rows, purpose=f"R-NBR continuation donor fit {fit_id}")
        donor["rows"] = donor_rows
        arrays, meta = build_payload(origin, arm, train, r06, geometry, donor, freeze, source_commit)
        peak = psutil.Process().memory_info().rss
        if peak > 6442450944:
            raise ContinuationError(f"Process RSS exceeds 6 GiB cap after payload construction: {peak}")
        meta["audit"]["resources"]["rss_bytes_at_persist"] = int(peak)
        receipt = persist_bundle(fit_id, arrays, meta)
        verified = _verify_fit_independent(fit_id, freeze, train, r06, source_commit)
        state_path = _fit_dir(fit_id) / "fit_state.json"
        staging = state_path.with_name(state_path.name + ".staging")
        with staging.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"fit_id": fit_id, "freeze_signature": freeze["freeze_signature"],
                                "state": "DURABLE_AND_INDEPENDENTLY_VERIFIED",
                                "verification": verified}, indent=2, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(staging, state_path)
        _finish(fit_id, "FIT_SAVED", receipt_sha=_sha(_fit_dir(fit_id) / "receipt.json"))
        return {"fit_id": fit_id, "receipt_sha256": _sha(_fit_dir(fit_id) / "receipt.json"), "verified": verified}
    except Exception as exc:
        _finish(fit_id, "FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reserve", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()
    try:
        if args.write_freeze:
            print(json.dumps(write_freeze(), indent=2, ensure_ascii=False))
            return 0
        freeze = load_freeze(require_clean=args.run_all)
        if args.preflight:
            tf = targetfree_train(); r06 = load_r06(freeze, tf)
            check = {str(o): validate_geometry(o, tf, load_geometry(o, freeze, tf),
                                                np.flatnonzero((tf["cycles"] >= 56) & (tf["cycles"] <= o) & tf["present"]))
                     for o in ORIGINS}
            payload = {"status": "PASS", "freeze_signature": freeze["freeze_signature"],
                       "source_commit_at_preflight": pilot._run_git("rev-parse", "HEAD"),
                       "official_train_sha256": _sha(TRAIN), "official_train_rows": len(tf["rows"]),
                       "target_column_loaded": False, "test_or_sample_opened": False,
                       "classification_labels_opened": False, "geometry_support": check,
                       "query_row_positions_sha256": {str(o): pilot.sha256_array(
                           load_geometry(o, freeze, tf)["score_rows"], "<i8") for o in ORIGINS},
                       "r06_cutoffs": sorted(r06), "resource_preflight": resource_preflight()}
            out_path = ROOT / "reports/optimization_v3/rnbr_continuation_v1/targetfree_preflight.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 0
        source_commit = pilot._run_git("rev-parse", "HEAD")
        if args.reserve:
            print(json.dumps(amend_and_reserve(freeze, source_commit), indent=2))
            return 0
        if args.run_all:
            reservation = amend_and_reserve(freeze, source_commit)
            completed = []
            for origin in ORIGINS:
                for arm in ARMS:
                    completed.append(run_fit(origin, arm, freeze, source_commit))
            print(json.dumps({"status": "SIX_FITS_COMPLETE", "reservation": reservation,
                              "fits": completed, "ledger_sha256": _sha(LEDGER)}, indent=2))
            return 0
        parser.print_help()
        return 2
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
