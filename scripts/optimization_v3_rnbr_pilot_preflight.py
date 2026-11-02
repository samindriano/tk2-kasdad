"""Read-only, target-free provenance preflight for the R-NBR two-origin pilot.

This checks the already saved R06 lineage and A02 row/ID/cycle/cosine geometry.
It does not load Regression targets, Classification labels, Test/SampleSubmission,
candidate predictions, construct an official residual memory, or mutate a ledger.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_preflight as readiness  # noqa: E402


EXPECTED_LEDGER_SHA256 = "6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147"
OLD_INVENTORY = ROOT / "reports/optimization_v3/rnbr_readiness_v1/INVENTORY.json"
OUTPUT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/PREPARATION_PREFLIGHT.json"


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Refusing to overwrite different pilot preflight: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()


def _saved_query_support(train: Any, origin: int, start: int, stop: int,
                        present: np.ndarray) -> dict[str, Any]:
    cycles = train["cycle_code"].to_numpy(dtype=np.int64)
    run_root = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
    output = {}
    counts_by_arm = {}
    expected_rows = np.flatnonzero((cycles >= start) & (cycles <= stop))
    for arm in ("observed", "shuffled"):
        npz_path = run_root / f"E{origin}" / arm / "predictions.npz"
        with np.load(npz_path, allow_pickle=False) as archive:
            rows = np.asarray(archive["score_rows"], dtype=np.int64)
            neighbor_count = np.asarray(archive["neighbor_count"], dtype=np.int64)
        selected = (cycles[rows] >= start) & (cycles[rows] <= stop)
        query_rows = rows[selected]
        counts = neighbor_count[selected]
        if not np.array_equal(query_rows, expected_rows):
            raise RuntimeError(f"Saved A02 support rows differ for E{origin}/{arm}")
        query_present = present[query_rows]
        if np.any(counts[~query_present] != 0) or np.any((counts[query_present] < 0) | (counts[query_present] > 32)):
            raise RuntimeError(f"Saved A02 neighbor counts violate present/missing routing at E{origin}/{arm}")
        counts_by_arm[arm] = counts
        output[arm] = {
            "query_rows": int(len(query_rows)),
            "present_query_rows": int(query_present.sum()),
            "missing_query_rows_unchanged_R06": int((~query_present).sum()),
            "present_query_rows_with_at_least_5_neighbors": int(np.sum(query_present & (counts >= 5))),
            "present_query_rows_with_1_to_4_neighbors_unchanged_R06": int(np.sum(query_present & (counts < 5))),
            "present_query_neighbor_count_min": int(counts[query_present].min()) if query_present.any() else None,
            "present_query_neighbor_count_median": float(np.median(counts[query_present])) if query_present.any() else None,
        }
    if not np.array_equal(counts_by_arm["observed"], counts_by_arm["shuffled"]):
        raise RuntimeError(f"A02 observed/shuffled query support differs at E{origin}")
    if output["observed"] != output["shuffled"]:
        raise RuntimeError(f"A02 observed/shuffled support summaries differ at E{origin}")
    return {"origin": origin, "query_cycles": [start, stop], "arms": output,
            "observed_shuffled_support_identical": True,
            "support_arrays_loaded": ["score_rows", "neighbor_count"],
            "target_and_prediction_arrays_loaded": False}


def run_preflight() -> dict[str, Any]:
    ledger_before = readiness.sha256_file(readiness.LEDGER)
    if ledger_before != EXPECTED_LEDGER_SHA256:
        raise RuntimeError(f"Research Next ledger changed: {ledger_before}")
    inventory_before = readiness.sha256_file(OLD_INVENTORY)

    source = readiness.verify_source_lineage()
    train, alignment = readiness.compare_train_features()
    caches = readiness.verify_caches(train)
    geometry = readiness.verify_saved_a02_geometry(train, caches)
    budget = readiness.budget_snapshot()
    ledger_events = [json.loads(line) for line in readiness.LEDGER.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    cap_records = [event for event in ledger_events if event.get("event") in
                   {"BUDGET_CATEGORY_AMENDMENT", "BUDGET_CAP_AMENDMENT"}]
    if not cap_records:
        raise RuntimeError("No effective category-cap record found in the current ledger")
    category_caps = cap_records[-1].get("effective_caps")

    cycles = train["cycle_code"].to_numpy(dtype=np.int64)
    present = train["signal_stream"].fillna("").astype(str).str.strip().ne("").to_numpy()
    support = {}
    query_support = {}
    primary_support = {}
    for origin, start, stop in ((67, 68, 79), (75, 76, 79)):
        donor_mask = (cycles >= 56) & (cycles <= origin) & present
        query_mask = (cycles >= start) & (cycles <= stop)
        query_present = query_mask & present
        query_missing = query_mask & ~present
        eligible = next(item for item in geometry if int(item["origin"]) == origin)
        query_support[str(origin)] = _saved_query_support(train, origin, start, stop, present)
        expected_query_rows = int(query_mask.sum())
        if expected_query_rows != int(eligible["arms"]["observed"]["query_rows"]):
            raise RuntimeError(f"Saved A02 query row count differs at origin {origin}")
        support[str(origin)] = {
            "donor_cycles": [56, origin],
            "eligible_present_donor_rows": int(donor_mask.sum()),
            "query_cycles": [start, stop],
            "query_rows": expected_query_rows,
            "present_query_rows": int(query_present.sum()),
            "missing_query_rows_unchanged_R06": int(query_missing.sum()),
            "source_prediction_npz_sha256": {
                arm: eligible["arms"][arm]["predictions_npz_sha256"]
                for arm in ("observed", "shuffled")
            },
            "geometry_cache_lineage": {
                arm: {
                    key: eligible["arms"][arm][key]
                    for key in ("fit_id", "freeze_signature", "freeze_manifest",
                                "predictions_npz_sha256", "model_sha256", "state_sha256",
                                "bank_rows", "bank_cycle_range", "bank_matrix_shape",
                                "query_rows", "first_four_rows", "neighbors_checked",
                                "unique_donor_rows_used", "geometry_hashes",
                                "target_derived_prediction_arrays_loaded")
                }
                for arm in ("observed", "shuffled")
            },
            "observed_shuffled_geometry_identical": eligible["observed_and_shuffled_geometry_identical"],
            "target_free_geometry_verified": eligible["target_free_geometry_reuse"],
            "geometry_fit_ids": {arm: eligible["arms"][arm]["fit_id"]
                                  for arm in ("observed", "shuffled")},
        }

    for origin, start, stop, window in ((67, 68, 71, "W4"), (75, 76, 79, "W6")):
        primary_support[str(origin)] = _saved_query_support(train, origin, start, stop, present)
        expected_window_rows = int(np.sum((cycles >= start) & (cycles <= stop)))
        if primary_support[str(origin)]["arms"]["observed"]["query_rows"] != expected_window_rows:
            raise RuntimeError(f"Saved A02 primary {window} subset differs at origin {origin}")

    primary_rows = sum(item["arms"]["observed"]["query_rows"] for item in primary_support.values())
    h12_l_rows = support["67"]["query_rows"]
    r06_h12 = next(item for item in caches if int(item["cutoff"]) == 67)
    if primary_rows != 13441 or h12_l_rows != 20903:
        raise RuntimeError(
            f"Frozen supported views differ: primary={primary_rows}, H12_L={h12_l_rows}"
        )
    if (r06_h12["all_score_rows"] != h12_l_rows
            or r06_h12["all_scored_cycles"] != [68, 79]):
        raise RuntimeError("E67 R06 saved predictions do not span the full H12_L horizon")

    if alignment.get("rows_each") != 50608 or alignment.get("status") != "PASS":
        raise RuntimeError("Official Train feature alignment did not match frozen source expectations")
    if budget.get("effective_cap") != 66 or budget.get("remaining_capacity") != 0:
        raise RuntimeError("Current fit-budget snapshot differs from the reviewed cap-66 checkpoint")
    if category_caps != {"continuation": 31, "control_replay": 0, "pilot": 34, "technical_retry": 1}:
        raise RuntimeError(f"Effective category caps differ from checkpoint: {category_caps}")
    if budget.get("pending_started_fits") != 0:
        raise RuntimeError("A pending Research Next fit appeared; refuse B0 progression")
    if any(arm["target_derived_prediction_arrays_loaded"]
           for item in geometry for arm in item["arms"].values()):
        raise RuntimeError("Target-derived A02 arrays were unexpectedly marked as loaded")
    if any(item["target_values_loaded"] for item in caches):
        raise RuntimeError("R06 cache audit unexpectedly accessed target values")

    ledger_after = readiness.sha256_file(readiness.LEDGER)
    inventory_after = readiness.sha256_file(OLD_INVENTORY)
    if ledger_after != ledger_before or inventory_after != inventory_before:
        raise RuntimeError("Read-only preflight changed the ledger or prior readiness inventory")

    return {
        "audit": "R-NBR pilot B0 target-free preparation preflight",
        "status": "PASS",
        "source_repo_head_before_B0_changes": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": {
            "official_regression_train_target_loaded": False,
            "official_classification_target_loaded": False,
            "test_or_sample_submission_accessed": False,
            "official_residual_memory_constructed": False,
            "official_candidate_predictions_created": False,
            "fit_started": False,
            "ledger_mutated": False,
        },
        "official_train_target_free_alignment": {
            **alignment,
            "classification_train_path": "data/raw/classification/Train.csv",
            "regression_train_path": "data/raw/regression/Train.csv",
            "interpretation": "Shared target-free field equality is provenance evidence only; it does not expand the frozen R06 feature allowlist.",
        },
        "R06_source_recipe_lineage": source,
        "R06_saved_cache_inventory": [{
            "cutoff": item["cutoff"], "cache_npz": item["cache_npz"],
            "cache_json": item["cache_json"], "npz_sha256": item["npz_sha256"],
            "sidecar_sha256": item["sidecar_sha256"], "freeze_signature": item["freeze_signature"],
            "all_score_rows": item["all_score_rows"],
            "all_scored_cycles": item["all_scored_cycles"],
            "first_four_cycle_window": item["first_four_cycle_window"],
            "first_four_rows": item["first_four_rows"],
            "target_values_loaded": item["target_values_loaded"],
        } for item in caches],
        "A02_saved_geometry": support,
        "saved_neighbor_support": query_support,
        "primary_window_neighbor_support": primary_support,
        "frozen_view_support": {
            "primary_W4_plus_W6_rows": primary_rows,
            "W4_rows": primary_support["67"]["arms"]["observed"]["query_rows"],
            "W6_rows": primary_support["75"]["arms"]["observed"]["query_rows"],
            "H12_L_rows": h12_l_rows,
            "H12_L_definition": "all E67 queries in cycles 68-79; E67 candidate/control and E67 R06 baseline; no E75 inputs",
            "H12_L_source_origin": 67,
            "H12_L_cycle_range": [68, 79],
            "H12_L_R06_cutoff": 67,
        },
        "view_provenance": {
            "primary_W4": {"source_origin": 67, "query_cycles": [68, 71],
                            "geometry_fit_ids": support["67"]["geometry_fit_ids"],
                            "R06_cache_cutoff": 67},
            "primary_W6": {"source_origin": 75, "query_cycles": [76, 79],
                            "geometry_fit_ids": support["75"]["geometry_fit_ids"],
                            "R06_cache_cutoff": 75},
            "H12_L": {"source_origin": 67, "query_cycles": [68, 79],
                      "geometry_fit_ids": support["67"]["geometry_fit_ids"],
                      "R06_cache_cutoff": 67,
                      "R06_cache_score_rows": r06_h12["all_score_rows"],
                      "E75_inputs_included": False},
        },
        "supersedes_prior_B0": {
            "prior_commit": "dbaa3bb0cf3e621daa84537bc8e62f4134153030",
            "prior_H12_error": "The prior version mixed E67 W4 and E75 W6 with a synthetic R06-only W5 and mislabeled it H12_L despite n=20,903.",
            "correction": "This successor uses only full E67 cycles 68-79 for H12_L; the prior commit remains in Git history.",
        },
        "current_ledger_snapshot": {**budget, "effective_category_caps": category_caps},
        "pilot_fit_plan": {
            "requested_task_estimator_fits": 4,
            "origins": [67, 75], "arms": ["observed", "shuffled"],
            "authorization_status": "NOT_AUTHORIZED_NOT_RESERVED",
        },
        "immutable_guards": {
            "ledger_sha256_before": ledger_before,
            "ledger_sha256_after": ledger_after,
            "ledger_unchanged": True,
            "prior_rnbr_readiness_inventory_sha256_before": inventory_before,
            "prior_rnbr_readiness_inventory_sha256_after": inventory_after,
            "prior_readiness_inventory_unchanged": True,
        },
    }


def main() -> int:
    result = run_preflight()
    _write_once(OUTPUT, result)
    print(json.dumps({"status": result["status"],
                      "pilot_origins": sorted(map(int, result["A02_saved_geometry"])),
                      "official_fit_starts": 0,
                      "ledger_sha256": result["immutable_guards"]["ledger_sha256_after"],
                      "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
