"""Independent verification of N04_DIAGNOSTIC bundles, weights, and MAE tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import optimization_v3_research_next_n00 as n00
from scripts import optimization_v3_research_next_n04 as n04
from src.models.research_next_budget import ledger_summary, read_events
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle
from src.models.regression_research_text import HYBRID_FEATURES
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
CARD_ID = "N04_DIAGNOSTIC"
STAGE = "DIAGNOSTIC"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04_DIAGNOSTIC/amendment_01/freeze.json"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RESULT_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N04_DIAGNOSTIC/amendment_01/results.json"
VERIFY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N04_DIAGNOSTIC/amendment_01/verification.json"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N04_DIAGNOSTIC/amendment_01"
ORIGINS = (55, 67)
PARITIES = (0, 1)


def _sha256_array(values: Any, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _near(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0, atol=tolerance, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _load_train() -> tuple[Path, pd.DataFrame, np.ndarray]:
    path = Path(task_files("regression")["train"])
    frame = pd.read_csv(path, usecols=["record_id", "threat_rank", *HYBRID_FEATURES])
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (len(frame) != 50_608 or not frame.record_id.is_unique
            or not np.array_equal(np.unique(cycles), np.arange(47, 80))
            or not np.isfinite(target).all() or np.any((target < 0) | (target > 100))):
        raise RuntimeError("N04_DIAGNOSTIC independent verifier: official Train schema differs")
    return path, frame, target


def _sources(frame: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    output = {}
    cycle_all = frame.cycle_code.to_numpy(dtype=np.int64)
    ids_all = frame.record_id.to_numpy(dtype=str)
    for cutoff in ORIGINS:
        r22_bundle = n00._load_artifact("R22", cutoff, frame)
        r06_bundle = n00._load_artifact("R06", cutoff, frame)
        r22 = n00.reconstruct_regression(r22_bundle)
        r06 = n00.reconstruct_regression(r06_bundle)
        rows = r22["rows"].astype(np.int64)
        expected = np.flatnonzero((cycle_all > cutoff) & (cycle_all <= cutoff + 12))
        if (not np.array_equal(rows, expected)
                or not np.array_equal(r06["rows"], rows)
                or not np.array_equal(r22_bundle["arrays"]["score_record_ids"].astype(str), ids_all[rows])
                or not np.array_equal(r06_bundle["arrays"]["score_record_ids"].astype(str), ids_all[rows])):
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: R22/R06 cache rows mismatch at {cutoff}")
        output[cutoff] = {
            "rows": rows, "ids": ids_all[rows], "cycles": cycle_all[rows],
            "R22_ORIGINAL": r22["prediction"].astype(np.float64),
            "legacy": r22["legacy"].astype(np.float64),
            "R06_ORIGINAL": r06["prediction"].astype(np.float64),
            "R22_npz_sha256": r22_bundle["npz_sha256"],
            "R22_sidecar_sha256": r22_bundle["sidecar_sha256"],
            "R06_npz_sha256": r06_bundle["npz_sha256"],
            "R06_sidecar_sha256": r06_bundle["sidecar_sha256"],
        }
    return output


def _domain_bundle_path_and_identity(freeze_identity: Mapping[str, Any], cutoff: int,
                                     parity: int) -> tuple[Path, dict[str, Any]]:
    key = f"{cutoff}:{'odd' if parity else 'even'}"
    audit = freeze_identity["recipe"]["reused_domain_bundle_lineage"][key]
    relative = audit["bundle_path"]
    bundle_path = ROOT / Path(relative)
    if cutoff == 55 and parity == 0:
        original = json.loads(n04.PREVIOUS_FREEZE_PATH.read_text(encoding="utf-8"))
        signature = original["freeze_signature"]
    else:
        current = json.loads(n04.FREEZE_PATH.read_text(encoding="utf-8"))
        signature = current["freeze_signature"]
    identity = n04._bundle_identity(signature, cutoff, "domain", parity)
    if file_sha256(bundle_path / "arrays.npz") != audit["bundle_npz_sha256"]:
        raise RuntimeError(f"N04_DIAGNOSTIC verifier: frozen domain artifact changed: {key}")
    return bundle_path, identity


def _domain_reconstruct(frame: pd.DataFrame, target: np.ndarray,
                        freeze_identity: Mapping[str, Any]) -> tuple[
                            dict[int, dict[str, Any]], dict[str, Any]
                        ]:
    cycles_all = frame.cycle_code.to_numpy(dtype=np.int64)
    folds_by_cutoff: dict[int, list[dict[str, Any]]] = {cutoff: [] for cutoff in ORIGINS}
    audit = {}
    ledger = read_events(LEDGER_PATH)
    n04_evidence = json.loads((n04.REPORT_ROOT / "verification.json").read_text(encoding="utf-8"))
    for cutoff in ORIGINS:
        for parity in PARITIES:
            name = "odd" if parity else "even"
            bundle_path, identity = _domain_bundle_path_and_identity(freeze_identity, cutoff, parity)
            bundle = read_bundle(bundle_path, identity, task="classification")
            rows = bundle.arrays["score_rows"].astype(np.int64)
            ids = bundle.arrays["score_record_ids"].astype(str)
            cycles = cycles_all[rows]
            labels = bundle.arrays["domain_label"].astype(np.int8)
            probability = bundle.arrays["prediction"].astype(np.float64)
            fit_rows = np.flatnonzero(cycles_all <= cutoff)
            recent_cycles = np.unique(cycles_all[fit_rows])[-4:]
            expected_rows = fit_rows[cycles_all[fit_rows] % 2 == parity]
            expected_labels = np.isin(cycles_all[expected_rows], recent_cycles).astype(np.int8)
            if (not np.array_equal(rows, expected_rows)
                    or not np.array_equal(ids, frame.record_id.to_numpy(dtype=str)[rows])
                    or not np.array_equal(labels, expected_labels)
                    or np.any(cycles % 2 != parity)
                    or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: domain rows/labels invalid at {cutoff}/{name}")
            auc = n04._independent_auc(labels, probability)
            saved_auc = bundle.receipt["metadata"]["diagnostics"]["independent_heldout_auc"]
            reported_fold = next(item for item in n04_evidence["cutoffs"][str(cutoff)]["folds"]
                                 if item["parity"] == name)
            if (not np.isclose(auc, saved_auc, rtol=0, atol=1e-14)
                    or not np.isclose(auc, reported_fold["auc_pairwise"], rtol=0, atol=1e-14)
                    or bundle.receipt["receipt_sha256"] != reported_fold["bundle_receipt_sha256"]):
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: domain AUC evidence differs at {cutoff}/{name}")
            folds_by_cutoff[cutoff].append({"rows": rows, "ids": ids, "labels": labels,
                                            "probability": probability, "auc": auc})
            fit_id = n04._fit_id("domain", cutoff, parity)
            events = [event for event in ledger if event.get("fit_id") == fit_id]
            if (sum(event.get("event") == "FIT_STARTED" for event in events) != 1
                    or sum(event.get("event") == "FIT_SAVED" for event in events) != 1):
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: saved historical domain fit missing {fit_id}")
            audit[f"{cutoff}:{name}"] = {"auc": auc, "rows": len(rows),
                                          "receipt_sha256": bundle.receipt["receipt_sha256"],
                                          "npz_sha256": file_sha256(bundle_path / "arrays.npz")}

    oof = {}
    for cutoff in ORIGINS:
        parts = folds_by_cutoff[cutoff]
        row = np.concatenate([part["rows"] for part in parts])
        order = np.argsort(row, kind="mergesort")
        row = row[order]
        expected = np.flatnonzero(cycles_all <= cutoff)
        if len(np.unique(row)) != len(row) or not np.array_equal(row, expected):
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: OOF domain rows incomplete at {cutoff}")
        labels = np.concatenate([part["labels"] for part in parts])[order]
        probability = np.concatenate([part["probability"] for part in parts])[order]
        cycles = cycles_all[row]
        if not np.array_equal(labels, np.isin(cycles, np.unique(cycles)[-4:]).astype(np.int8)):
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: pseudo-recent label assignment differs at {cutoff}")
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        odds = clipped / (1 - clipped)
        source = labels == 0
        pre_normalized = np.ones(len(row), dtype=np.float64)
        pre_normalized[source] = np.clip(odds[source], .25, 4.0)
        weights = pre_normalized / pre_normalized.mean()
        weight_sha = _sha256_array(weights)
        report_diag = json.loads((n04.REPORT_ROOT / "results.json").read_text(encoding="utf-8"))["domain_oof"][str(cutoff)]["weight_diagnostics"]
        if (weight_sha != report_diag["weight_sha256"]
                or not np.isclose(weights.mean(), 1, atol=1e-12, rtol=0)):
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: reconstructed density weights differ at {cutoff}")
        aucs = [part["auc"] for part in parts]
        oof[cutoff] = {"rows": row, "ids": frame.record_id.to_numpy(dtype=str)[row],
                       "cycles": cycles, "labels": labels, "probability": probability,
                       "weights": weights, "fold_aucs": aucs,
                       "weight_sha256": weight_sha,
                       "ess_fraction": float(weights.sum() ** 2 / np.square(weights).sum() / len(weights)),
                       "source_clip_fraction": float(np.mean(
                           (odds[source] <= .25) | (odds[source] >= 4.0))),
                       "target_sha256": _sha256_array(target[row])}
    return oof, audit


def _rank50(raw: Any, cycles: Any) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    groups = np.asarray(cycles, dtype=np.int64)
    result = np.empty_like(values)
    for cycle in np.unique(groups):
        mask = groups == cycle
        result[mask] = np.clip(.5 * values[mask] + .5 *
                               (100.0 * rankdata(values[mask], method="average") / int(mask.sum())), 0, 100)
    return result


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction)))


def _view_metrics(target: np.ndarray, candidate: np.ndarray, shuffle: np.ndarray,
                  baseline: np.ndarray, legacy: np.ndarray, r06: np.ndarray,
                  missing: np.ndarray, rows: np.ndarray, *, name: str) -> dict[str, Any]:
    result = {
        "name": name, "n": int(len(target)), "rows_sha256": _sha256_array(rows, "<i8"),
        "metrics": {"candidate": _mae(target, candidate), "R22_ORIGINAL": _mae(target, baseline),
                    "within_cycle_shuffle": _mae(target, shuffle),
                    "protected_legacy": _mae(target, legacy),
                    "R06_ORIGINAL_secondary": _mae(target, r06)},
    }
    result["gain_vs_R22"] = result["metrics"]["R22_ORIGINAL"] - result["metrics"]["candidate"]
    result["gain_vs_shuffle"] = result["metrics"]["within_cycle_shuffle"] - result["metrics"]["candidate"]
    groups = {"stream_missing": missing,
              "observed_target_le5_evaluation_only": target <= 5,
              "observed_target_ge95_evaluation_only": target >= 95}
    result["groups"] = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        result["groups"][name] = {"n": n, "supported": n >= 100}
        if not n:
            continue
        row = result["groups"][name]
        row.update({
            "candidate_mae": _mae(target[mask], candidate[mask]),
            "R22_ORIGINAL_mae": _mae(target[mask], baseline[mask]),
            "within_cycle_shuffle_mae": _mae(target[mask], shuffle[mask]),
            "legacy_mae": _mae(target[mask], legacy[mask]),
            "R06_ORIGINAL_secondary_mae": _mae(target[mask], r06[mask]),
        })
        row["harm_vs_R22"] = row["candidate_mae"] - row["R22_ORIGINAL_mae"]
        row["gain_vs_shuffle"] = row["within_cycle_shuffle_mae"] - row["candidate_mae"]
        row["harm_vs_legacy"] = row["candidate_mae"] - row["legacy_mae"]
    return result


def _verify() -> dict[str, Any]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    identity = frozen["identity"]
    signature = frozen["freeze_signature"]
    if signature != canonical_sha256(identity):
        raise RuntimeError("N04_DIAGNOSTIC verifier: freeze signature mismatch")
    commit = frozen.get("execution_context", {}).get("source_commit")
    if (not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                     cwd=ROOT, check=False).returncode != 0):
        raise RuntimeError("N04_DIAGNOSTIC verifier: source commit is not an ancestor")
    checked_hashes = {}
    for relative, expected in identity["input_and_source_sha256"].items():
        path = ROOT / Path(relative)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: frozen hash changed: {relative}")
        checked_hashes[relative] = actual

    train_path, frame, target = _load_train()
    sources = _sources(frame)
    oof, domain_audit = _domain_reconstruct(frame, target, identity)
    original_results = json.loads((n04.REPORT_ROOT / "results.json").read_text(encoding="utf-8"))
    if original_results["status"] != "SKIPPED_SCIENTIFIC_PREFLIGHT":
        raise RuntimeError("N04_DIAGNOSTIC verifier: original N04 verdict changed")
    ledger = read_events(LEDGER_PATH)
    reservation = [event for event in ledger if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    if len(reservation) != 1 or int(reservation[0]["fits"]) != 4 or reservation[0]["category"] != "pilot":
        raise RuntimeError("N04_DIAGNOSTIC verifier: exact new amendment reservation missing")

    bundles: dict[int, dict[str, Any]] = {cutoff: {} for cutoff in ORIGINS}
    fit_audits = []
    for cutoff in ORIGINS:
        source = sources[cutoff]
        prefix_rows = np.flatnonzero(frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff)
        prefix_target = target[prefix_rows]
        cycles_fit = frame.cycle_code.to_numpy(dtype=np.int64)[prefix_rows]
        for component in ("weighted", "within_cycle_shuffle"):
            fit_id = f"{CARD_ID}:{STAGE}:{cutoff}:{component}"
            component_weights = oof[cutoff]["weights"].copy()
            if component == "within_cycle_shuffle":
                rng = np.random.default_rng(42 + cutoff)
                shuffled = component_weights.copy()
                for cycle in np.unique(oof[cutoff]["cycles"]):
                    mask = oof[cutoff]["cycles"] == cycle
                    shuffled[mask] = rng.permutation(component_weights[mask])
                component_weights = shuffled
            bundle_identity = {"batch_id": BATCH, "card_id": CARD_ID,
                               "freeze_signature": signature, "stage": STAGE,
                               "cutoff": cutoff, "component": component}
            path = BUNDLE_ROOT / f"fit{cutoff}" / component
            bundle = read_bundle(path, bundle_identity, task="regression")
            arrays = bundle.arrays
            diag = bundle.receipt["metadata"].get("diagnostics", {})
            expected_mapped = _rank50(arrays["raw_component"], source["cycles"])
            expected_prediction = np.clip(.75 * source["legacy"] + .25 * expected_mapped, 0, 100)
            checks = {
                "score_rows": np.array_equal(arrays["score_rows"], source["rows"]),
                "record_ids": np.array_equal(arrays["score_record_ids"].astype(str), source["ids"]),
                "cycle_codes": np.array_equal(arrays["cycle_code"], source["cycles"]),
                "R22_baseline": np.array_equal(arrays["baseline_prediction"], source["R22_ORIGINAL"]),
                "legacy_component": np.array_equal(arrays["legacy_champion"], source["legacy"]),
                "R06_secondary": np.array_equal(arrays["r06_original"], source["R06_ORIGINAL"]),
                "rank50_average_ties": np.allclose(arrays["rank50_component"], expected_mapped, rtol=0, atol=1e-12),
                "blend_075_025_clip": np.allclose(arrays["prediction"], expected_prediction, rtol=0, atol=1e-12),
                "fit_rows": diag.get("fit_rows") == len(prefix_rows),
                "fit_index_hash": diag.get("fit_row_indices_sha256") == _sha256_array(prefix_rows, "<i8"),
                "fit_target_hash": diag.get("fit_target_sha256") == _sha256_array(prefix_target),
                "sample_weight_hash": diag.get("fit_sample_weight_sha256") == _sha256_array(component_weights),
                "model_received_weights": diag.get("model_state", {}).get("sample_weight_passed_to_fit") is True,
                "model_weight_hash": diag.get("model_state", {}).get("sample_weight_sha256") == _sha256_array(component_weights),
                "model_fit_design_hash_matches_transform_audit": (
                    diag.get("model_state", {}).get("fit_design_sha256")
                    == diag.get("fit_design_sha256")
                    == diag.get("transform_state", {}).get("fit_design_sha256")
                ),
                "model_target_hash_matches_fit_target": (
                    diag.get("model_state", {}).get("target_sha256") == diag.get("fit_target_sha256")
                ),
                "model_parameters_match_frozen_N04": (
                    diag.get("model_state", {}).get("parameters") == n04.XGB_PARAMETERS
                ),
                "model_cutoff_matches_fit": diag.get("model_state", {}).get("cutoff") == cutoff,
                "score_labels_not_fit": diag.get("score_target_passed_to_fit") is False,
                "test_not_used": diag.get("test_data_used") is False,
            }
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: {fit_id} failed {failed}")
            starts = [event for event in ledger if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
            saved = [event for event in ledger if event.get("event") == "FIT_SAVED" and event.get("fit_id") == fit_id]
            failed_events = [event for event in ledger if event.get("event") == "FIT_FAILED" and event.get("fit_id") == fit_id]
            if len(starts) != 1 or len(saved) != 1 or failed_events:
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: ledger state invalid for {fit_id}")
            bundles[cutoff][component] = arrays
            fit_audits.append({"fit_id": fit_id, "fit_rows": int(diag["fit_rows"]),
                               "score_rows": len(arrays["score_rows"]),
                               "receipt_sha256": bundle.receipt["receipt_sha256"],
                               "npz_sha256": bundle.receipt["npz_sha256"], "checks": checks})
    diag_starts = [event for event in ledger if event.get("event") == "FIT_STARTED"
                   and event.get("card_id") == CARD_ID and event.get("stage") == STAGE]
    if len(diag_starts) != 4:
        raise RuntimeError("N04_DIAGNOSTIC verifier: task estimator fit count is not exactly four")

    report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if (report.get("freeze_signature") != signature
            or report.get("status") != "DIAGNOSTIC_ONLY"
            or report.get("original_N04_status_preserved") != "SKIPPED_SCIENTIFIC_PREFLIGHT"
            or report.get("selection_eligibility", {}).get("eligible_for_continuation") is not False
            or report.get("amendment_not_outcome_blind") is not True):
        raise RuntimeError("N04_DIAGNOSTIC verifier: report status/authorization mismatch")
    windows = {"W1": (55, 56, 59), "W2": (55, 60, 63), "W3": (55, 64, 67),
               "W4": (67, 68, 71), "W5": (67, 72, 75), "W6": (67, 76, 79)}
    verified_views = {}
    view_definitions = {name: {cutoff: (first, last)} for name, (cutoff, first, last) in windows.items()}
    view_definitions["primary_W1_W4_pooled"] = {55: (56, 59), 67: (68, 71)}
    view_definitions["H12_W1_W6_pooled"] = {55: (56, 67), 67: (68, 79)}
    for cutoff in ORIGINS:
        h12_name = f"H12_origin_{cutoff}"
        view_definitions[h12_name] = {
            cutoff: (cutoff + 1, cutoff + 12 if cutoff == 55 else 79)
        }
    saved_views = {name.split("_")[0]: view
                   for name, view in report["windows_W1_W6"].items()}
    saved_views.update({
        "primary_W1_W4_pooled": report["primary_W1_W4_pooled"],
        "H12_W1_W6_pooled": report["H12_W1_W6_pooled"],
    })
    saved_views.update({f"H12_origin_{cutoff}": report["cutoffs"][str(cutoff)]["H12_view"]
                        for cutoff in ORIGINS})
    for view_name, ranges in view_definitions.items():
        candidates, shuffles, baseline, legacy, r06, masks = [], [], [], [], [], []
        target_parts, row_parts = [], []
        for cutoff, (first_cycle, last_cycle) in ranges.items():
            source = sources[cutoff]
            mask = (source["cycles"] >= first_cycle) & (source["cycles"] <= last_cycle)
            rows = source["rows"][mask]
            if not len(rows):
                continue
            target_parts.append(target[rows])
            candidates.append(bundles[cutoff]["weighted"]["prediction"][mask])
            shuffles.append(bundles[cutoff]["within_cycle_shuffle"]["prediction"][mask])
            baseline.append(source["R22_ORIGINAL"][mask])
            legacy.append(source["legacy"][mask])
            r06.append(source["R06_ORIGINAL"][mask])
            masks.append(n04._text_values(frame.iloc[rows]["signal_stream"])[1].astype(bool))
            row_parts.append(rows)
        y = np.concatenate(target_parts)
        candidate = np.concatenate(candidates)
        shuffle = np.concatenate(shuffles)
        r22 = np.concatenate(baseline)
        protected = np.concatenate(legacy)
        r06_values = np.concatenate(r06)
        missing = np.concatenate(masks)
        rows = np.concatenate(row_parts)
        if len(rows) != len(np.unique(rows)):
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: duplicate rows in {view_name}")
        order = np.argsort(rows, kind="mergesort")
        rows, y, candidate, shuffle, r22, protected, r06_values, missing = (
            values[order] for values in
            (rows, y, candidate, shuffle, r22, protected, r06_values, missing)
        )
        recomputed = _view_metrics(y, candidate, shuffle, r22, protected, r06_values,
                                   missing, rows, name=view_name)
        stored = saved_views[view_name]
        if (recomputed["n"] != stored["n"]
                or recomputed["rows_sha256"] != stored["rows_sha256"]
                or not _near(recomputed["metrics"]["candidate"], stored["metrics"]["candidate"])
                or not _near(recomputed["metrics"]["R22_ORIGINAL"], stored["metrics"]["R22_ORIGINAL"])
                or not _near(recomputed["metrics"]["within_cycle_shuffle"], stored["metrics"]["within_cycle_shuffle"])
                or not _near(recomputed["metrics"]["protected_legacy"], stored["metrics"]["protected_legacy"])
                or not _near(recomputed["metrics"]["R06_ORIGINAL_secondary"], stored["metrics"]["R06_ORIGINAL_secondary"])
                or not _near(recomputed["gain_vs_R22"], stored["gain_vs_R22"])
                or not _near(recomputed["gain_vs_shuffle"], stored["gain_vs_shuffle"])):
            raise RuntimeError(f"N04_DIAGNOSTIC verifier: MAE or matched-control metrics differ in {view_name}")
        for group_name, group in recomputed["groups"].items():
            saved_group = stored["groups"][group_name]
            if group["n"] != saved_group["n"] or group["supported"] != saved_group["supported"]:
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: subgroup support differs in {view_name}/{group_name}")
            if group["n"] and (not _near(group["candidate_mae"], saved_group["candidate_mae"])
                                or not _near(group["R22_ORIGINAL_mae"], saved_group["R22_ORIGINAL_mae"])
                                or not _near(group["within_cycle_shuffle_mae"], saved_group["within_cycle_shuffle_mae"])
                                or not _near(group["legacy_mae"], saved_group["legacy_mae"])
                                or not _near(group["R06_ORIGINAL_secondary_mae"], saved_group["R06_ORIGINAL_secondary_mae"])
                                or not _near(group["harm_vs_R22"], saved_group["harm_vs_R22"])
                                or not _near(group["gain_vs_shuffle"], saved_group["gain_vs_shuffle"])
                                or not _near(group["harm_vs_legacy"], saved_group["harm_vs_legacy"])):
                raise RuntimeError(f"N04_DIAGNOSTIC verifier: subgroup MAE differs in {view_name}/{group_name}")
        verified_views[view_name] = {"n": recomputed["n"],
                                     "gain_vs_R22": recomputed["gain_vs_R22"],
                                     "gain_vs_shuffle": recomputed["gain_vs_shuffle"],
                                     "candidate_mae": recomputed["metrics"]["candidate"],
                                     "R22_mae": recomputed["metrics"]["R22_ORIGINAL"],
                                     "shuffle_mae": recomputed["metrics"]["within_cycle_shuffle"],
                                     "subgroup_count": len(recomputed["groups"])}

    verification = {
        "status": "PASS", "verification_method": "independent domain OOF/weight reconstruction, task bundle/policy/MAE recomputation",
        "freeze_signature": signature, "verified_input_source_hash_count": len(checked_hashes),
        "original_N04_status": original_results["status"],
        "original_N04_freeze_signature": identity["original_N04_freeze_signature"],
        "reused_domain_fit_count": len(domain_audit), "reused_domains": domain_audit,
        "verified_task_fit_count": len(fit_audits), "task_fits": fit_audits,
        "verified_views": verified_views,
        "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0,
        "automatic_continuation_eligibility": False,
        "budget": ledger_summary(LEDGER_PATH),
    }
    if VERIFY_PATH.exists():
        existing = json.loads(VERIFY_PATH.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(verification):
            raise RuntimeError("N04_DIAGNOSTIC verifier refuses to replace a different receipt")
    else:
        VERIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERIFY_PATH.write_text(json.dumps(verification, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    return verification


def main() -> int:
    result = _verify()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
