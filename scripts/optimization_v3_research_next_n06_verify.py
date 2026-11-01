"""Independent post-fit row, prediction, metric, and budget audit for N06."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from scripts import optimization_v3_research_next_n00 as n00
from scripts import optimization_v3_research_next_n06 as runner
from src.models.classification_models import _text_values
from src.models.research_next_budget import ledger_summary, read_events
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle
from src.models.regression_research_text import _array_digest
from src.models.optimization_v3_batch1_r06 import FEATURE_NAMES
from src.utils.paths import task_files


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if not np.isfinite(error).all() or error.ndim != 1 or not len(error):
        raise ValueError("N06 verifier received empty/nonfinite errors")
    return {"n": int(error.size), "mae": float(np.abs(error).sum(dtype=np.float64) / error.size),
            "bias": float(error.sum(dtype=np.float64) / error.size)}


def _rank_component(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    """Independent tie grouping and zero-based sorted-order rank reconstruction."""
    output = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        selected = np.flatnonzero(cycles == cycle)
        order = np.argsort(raw[selected], kind="mergesort")
        ordered = raw[selected][order]
        positions = np.empty(len(order), dtype=np.float64)
        left = 0
        while left < len(ordered):
            right = left + 1
            while right < len(ordered) and ordered[right] == ordered[left]:
                right += 1
            # one-based average rank = ((left+1)+right)/2
            positions[left:right] = 100.0 * (left + 1 + right) / (2.0 * len(ordered))
            left = right
        restored = np.empty(len(order), dtype=np.float64)
        restored[order] = positions
        output[selected] = np.minimum(100.0, np.maximum(0.0, 0.5 * raw[selected] + 0.5 * restored))
    return output


def _close(left: float, right: float) -> bool:
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= 1e-12


def verify() -> dict[str, Any]:
    frozen = json.loads(runner.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N06 verifier found a bad freeze signature")
    if not runner._is_ancestor(frozen["execution_context"]["source_commit"]):
        raise RuntimeError("N06 verifier source commit is not an ancestor")
    train_path = Path(task_files("regression")["train"])
    if file_sha256(train_path) != frozen["identity"]["official_train_sha256"]:
        raise RuntimeError("N06 verifier official Train hash differs")
    frame = pd.read_csv(train_path)
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    controls = {}
    cache_checks: dict[str, bool] = {}
    for cutoff in runner.ORIGINS:
        r22_bundle = n00._load_artifact("R22", cutoff, frame)
        r06_bundle = n00._load_artifact("R06", cutoff, frame)
        r22 = n00.reconstruct_regression(r22_bundle)
        r06 = n00.reconstruct_regression(r06_bundle)
        frozen_cache = frozen["identity"]["cache_lineage"][str(cutoff)]
        cache_checks[f"{cutoff}:r22_hashes"] = (
            frozen_cache.get("R22_npz_sha256") == r22_bundle["npz_sha256"]
            and frozen_cache.get("R22_sidecar_sha256") == r22_bundle["sidecar_sha256"]
            and frozen_cache.get("R22_identity_sha256") == r22_bundle["sidecar"]["identity_sha256"]
        )
        cache_checks[f"{cutoff}:r06_hashes"] = (
            frozen_cache.get("R06_npz_sha256") == r06_bundle["npz_sha256"]
            and frozen_cache.get("R06_sidecar_sha256") == r06_bundle["sidecar_sha256"]
            and frozen_cache.get("R06_identity_sha256") == r06_bundle["sidecar"]["identity_sha256"]
        )
        controls[cutoff] = {"r22": r22["prediction"], "r06": r06["prediction"],
                            "legacy": r06["legacy"], "rows": r22["rows"],
                            "cycles": r22["cycles"]}

    result_path = runner.REPORT_ROOT / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("status") != "COMPLETE_INFORMATION" or result.get("continuation_eligible") is not False
            or result.get("automatic_promotion") is not False or result.get("fit_count") != 4):
        raise RuntimeError("N06 result does not preserve the information-only scope")
    checks: dict[str, bool] = dict(cache_checks)
    source_commit = frozen["identity"].get("source_commit")
    checks["task_source_commit_matches_freeze_execution"] = (
        bool(source_commit)
        and frozen["execution_context"].get("source_commit") is not None
        and runner._is_ancestor(source_commit)
        and runner._is_ancestor(frozen["execution_context"]["source_commit"])
    )
    for relative, expected_hash in frozen["identity"]["source_sha256"].items():
        path = runner.ROOT / Path(relative)
        working_hash = file_sha256(path) if path.is_file() else None
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=runner.ROOT, capture_output=True, check=True
        ).stdout
        checks[f"source:{relative}"] = (
            working_hash == expected_hash
            and hashlib.sha256(committed).hexdigest() == expected_hash
            and path.read_bytes() == committed
        )
    bundles = {}
    for seed in runner.SEEDS:
        bundles[seed] = {}
        for cutoff in runner.ORIGINS:
            path = runner._bundle_path(cutoff, seed)
            identity = runner._bundle_identity(frozen["freeze_signature"], cutoff, seed)
            bundle = read_bundle(path, identity, task="regression")
            arrays = bundle.arrays
            metadata = bundle.receipt["metadata"].get("diagnostics", {})
            source = controls[cutoff]
            expected_rows = source["rows"].astype(np.int64)
            fit_rows = np.flatnonzero(cycles <= cutoff)
            checks[f"{seed}:{cutoff}:rows"] = np.array_equal(arrays["score_rows"], expected_rows)
            checks[f"{seed}:{cutoff}:ids"] = np.array_equal(
                arrays["score_record_ids"].astype(str), frame.record_id.to_numpy(dtype=str)[expected_rows]
            )
            checks[f"{seed}:{cutoff}:cycle_and_controls"] = (
                np.array_equal(arrays["cycle_code"], cycles[expected_rows])
                and np.array_equal(arrays["baseline_prediction"], source["r22"])
                and np.array_equal(arrays["R06_original_prediction"], source["r06"])
                and np.array_equal(arrays["legacy_prediction"], source["legacy"])
            )
            checks[f"{seed}:{cutoff}:fit_prefix"] = (
                metadata.get("fit_rows") == len(fit_rows)
                and metadata.get("fit_row_indices_sha256") == _array_digest(fit_rows, "<i8")
                and metadata.get("fit_target_sha256") == _array_digest(target[fit_rows], "<f8")
                and metadata.get("score_target_passed_to_fit") is False
                and metadata.get("outer_score_labels_read_for_fit") is False
                and metadata.get("test_data_used") is False
            )
            model_state = metadata.get("model_state", {})
            checks[f"{seed}:{cutoff}:original_r06_state"] = (
                model_state.get("fit_rows") == len(fit_rows)
                and model_state.get("target_sha256") == _array_digest(target[fit_rows], "<f8")
                and model_state.get("input_feature_names") == list(FEATURE_NAMES)
                and model_state.get("validation_labels_passed_to_fit") is False
                and model_state.get("test_data_used") is False
            )
            mapped = _rank_component(arrays["raw_candidate"].astype(np.float64), arrays["cycle_code"].astype(np.int64))
            expected = np.clip(0.75 * source["legacy"] + 0.25 * mapped, 0, 100)
            checks[f"{seed}:{cutoff}:rank_and_blend"] = (
                np.allclose(mapped, arrays["rank50_component"], atol=1e-12, rtol=0)
                and np.allclose(expected, arrays["prediction"], atol=1e-12, rtol=0)
            )
            bundles[seed][cutoff] = arrays

    views: dict[str, list[tuple[int, int, int]]] = {
        "primary_W1_W4": [(55, 56, 59), (67, 68, 71)],
        "H12_pooled": [(55, 56, 67), (67, 68, 79)],
        "H12_cutoff55": [(55, 56, 67)], "H12_cutoff67": [(67, 68, 79)],
        "W1": [(55, 56, 59)], "W2": [(55, 60, 63)], "W3": [(55, 64, 67)],
        "W4": [(67, 68, 71)], "W5": [(67, 72, 75)], "W6": [(67, 76, 79)],
    }
    missing = _text_values(frame.signal_stream)[1].astype(bool)
    for seed in runner.SEEDS:
        for view, parts in views.items():
            row_seq, candidate_seq = [], []
            refs: dict[str, list[np.ndarray]] = {key: [] for key in ("r22", "r06", "legacy")}
            cycles_seq = []
            for cutoff, lo, hi in parts:
                item = bundles[seed][cutoff]
                mask = (item["cycle_code"] >= lo) & (item["cycle_code"] <= hi)
                row_seq.append(item["score_rows"][mask].astype(np.int64))
                cycles_seq.append(item["cycle_code"][mask].astype(np.int64))
                candidate_seq.append(item["prediction"][mask].astype(np.float64))
                refs["r22"].append(item["baseline_prediction"][mask].astype(np.float64))
                refs["r06"].append(item["R06_original_prediction"][mask].astype(np.float64))
                refs["legacy"].append(item["legacy_prediction"][mask].astype(np.float64))
            rows = np.concatenate(row_seq)
            y = target[rows]
            candidate = np.concatenate(candidate_seq)
            references = {key: np.concatenate(value) for key, value in refs.items()}
            cycles_view = np.concatenate(cycles_seq)
            if len(rows) != len(np.unique(rows)):
                raise RuntimeError(f"N06 verifier detected overlapping rows in {view}")
            observed = result["seeds"][str(seed)]["views"][view]
            checks[f"{seed}:{view}:n"] = int(observed["rows"]) == len(rows)
            checks[f"{seed}:{view}:candidate_metric"] = (
                int(observed["metrics"]["candidate"]["n"]) == len(rows)
                and _close(observed["metrics"]["candidate"]["mae"], _metric(y, candidate)["mae"])
                and _close(observed["metrics"]["candidate"]["bias"], _metric(y, candidate)["bias"])
            )
            for key, prediction in references.items():
                ref_metric = _metric(y, prediction)
                gain = ref_metric["mae"] - _metric(y, candidate)["mae"]
                checks[f"{seed}:{view}:{key}_metric_and_gain"] = (
                    _close(observed["metrics"][key]["mae"], ref_metric["mae"])
                    and _close(observed["metrics"][f"gain_vs_{key}"], gain)
                )
            group_masks = {
                "signal_stream_missing": missing[rows],
                "signal_stream_present": ~missing[rows],
                "observed_threat_rank_le_5_evaluation_only": y <= 5,
                "observed_threat_rank_ge_95_evaluation_only": y >= 95,
            }
            for name, mask in group_masks.items():
                group = observed["groups"][name]
                checks[f"{seed}:{view}:{name}:support"] = (
                    group["n"] == int(mask.sum())
                    and group["supported_minimum_100"] is bool(mask.sum() >= 100)
                )
                if mask.any():
                    checks[f"{seed}:{view}:{name}:metrics"] = _close(
                        group["candidate"]["mae"], _metric(y[mask], candidate[mask])["mae"]
                    )
                    for key, prediction in references.items():
                        reference_metric = _metric(y[mask], prediction[mask])
                        checks[f"{seed}:{view}:{name}:reference_{key}_mae"] = _close(
                            group[key]["mae"], reference_metric["mae"]
                        )
                        checks[f"{seed}:{view}:{name}:gain_{key}"] = _close(
                            group[f"gain_vs_{key}"],
                            reference_metric["mae"] - _metric(y[mask], candidate[mask])["mae"],
                        )
                else:
                    checks[f"{seed}:{view}:{name}:metrics"] = group["candidate"] is None
                    for key in references:
                        checks[f"{seed}:{view}:{name}:gain_{key}"] = (
                            group[key] is None and group[f"gain_vs_{key}"] is None
                        )
            per_cycle = observed["per_cycle"]
            for cycle in np.unique(cycles_view):
                mask = cycles_view == cycle
                checks[f"{seed}:{view}:cycle{cycle}:mae"] = _close(
                    per_cycle[str(int(cycle))]["candidate"]["mae"],
                    _metric(y[mask], candidate[mask])["mae"],
                )
                for key, prediction in references.items():
                    reference_metric = _metric(y[mask], prediction[mask])
                    checks[f"{seed}:{view}:cycle{cycle}:reference_{key}_mae"] = _close(
                        per_cycle[str(int(cycle))][key]["mae"], reference_metric["mae"]
                    )
                    checks[f"{seed}:{view}:cycle{cycle}:gain_{key}"] = _close(
                        per_cycle[str(int(cycle))][f"gain_vs_{key}"],
                        reference_metric["mae"] - _metric(y[mask], candidate[mask])["mae"],
                    )

    events = read_events(runner.LEDGER_PATH)
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    releases = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"
                and e.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"
              and e.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"
             and e.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"
              and e.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    expected_fit_ids = {runner._fit_id(c, s) for c in runner.ORIGINS for s in runner.SEEDS}
    checks["one_exact_four_fit_reservation"] = len(reservations) == 1 and reservations[0].get("fits") == 4
    checks["exact_four_started_and_saved"] = (
        len(starts) == 4 and len(saved) == 4 and not failed
        and {e.get("fit_id") for e in starts} == expected_fit_ids
        and {e.get("fit_id") for e in saved} == expected_fit_ids
    )
    checks["reservation_released_zero_unused_once"] = len(releases) == 1 and releases[0].get("unused_fits") == 0
    budget = ledger_summary(runner.LEDGER_PATH)
    checks["global_and_pilot_caps_respected"] = (
        budget["authorized_cap"] == 60 and budget["active_reserved"] <= 60
        and budget["category_reservations"].get("pilot", 0) <= 26
    )
    checks["safety_and_no_promotion"] = (
        result["safety"].get("test_data_used") is False
        and result["safety"].get("sample_submission_used") is False
        and result["safety"].get("kaggle_calls") == 0
        and result.get("continuation_eligible") is False
        and result.get("automatic_promotion") is False
    )
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failures else "FAIL", "checks": checks,
        "failed_checks": failures, "verified_fit_count": len(saved),
        "verified_fit_ids": sorted(expected_fit_ids),
        "bundle_receipts": result.get("fit_receipts"), "budget": budget,
        "train_sha256": file_sha256(train_path), "results_sha256": file_sha256(result_path),
    }


if __name__ == "__main__":
    report = verify()
    runner._write_once(runner.REPORT_ROOT / "verification.json", report)
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
