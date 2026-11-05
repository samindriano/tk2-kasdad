"""Independent verifier for REG follow-up V2 receipts and diagnostic metrics.

This module deliberately does not import or call the result builder. The
pre-fit command reads only identity artifacts; diagnostic verification, if
authorized later, checks persisted target/prediction rows against saved receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2"
EXPERIMENT_DIR = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2"
RUN_DIR = ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2"
PREFIT_FREEZE = EXPERIMENT_DIR / "PREFIT_FREEZE.json"
IDENTITY_PREFLIGHT = REPORT_DIR / "IDENTITY_PREFLIGHT.json"
SYNTHETIC_QA = REPORT_DIR / "SYNTHETIC_QA.json"
DIAGNOSTICS = REPORT_DIR / "DIAGNOSTICS.json"
ROOT_SOURCE_PASS = EXPERIMENT_DIR / "ROOT_SOURCE_PASS.json"
ROOT_PRODUCTION_PASS = EXPERIMENT_DIR / "ROOT_PRODUCTION_PASS.json"
LEDGER = RUN_DIR / "lp_ledger.jsonl"
FINAL_STATE = RUN_DIR / "fits" / "FINAL_RECENCY_H8.json"
FINAL_APPLICATION_FREEZE = REPORT_DIR / "FINAL_APPLICATION_FREEZE.json"
PRODUCTION_RESULT = REPORT_DIR / "PRODUCTION_RESULT.json"
SUBMISSION_DIR = ROOT / "submissions/reg_final_wave_followup_v2"
OUTPUTS = {
    "A": SUBMISSION_DIR / "recency_affine_h8_e79.csv",
    "B": SUBMISSION_DIR / "global_affine_present_only_e79.csv",
}
MANIFESTS = {"A": REPORT_DIR / "MANIFEST_A.json", "B": REPORT_DIR / "MANIFEST_B.json"}
EXPERIMENT = "REG_FINAL_WAVE_FOLLOWUP_V2"
FIT_IDS = ("D1_GLOBAL", "D1_RECENCY_H8", "D2_GLOBAL", "D2_RECENCY_H8", "FINAL_RECENCY_H8")
DIAGNOSTIC_FIT_IDS = FIT_IDS[:4]

SOURCE_RELATIVE = {
    "runner": "scripts/reg_final_wave_followup_v2.py",
    "verifier": "scripts/verify_reg_final_wave_followup_v2.py",
    "tests": "tests/test_reg_final_wave_followup_v2.py",
    "protocol": "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/PROTOCOL.md",
    "synthetic_qa": "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/SYNTHETIC_QA.json",
    "v2_identity_helpers": "scripts/reg_final_wave_v2.py",
    "r17_lp_helper": "src/models/optimization_v3_batch1_r17.py",
    "selected_row_reader": "scripts/reg_position_failure_mechanism_v1.py",
    "direct_prediction_source": "scripts/reg_position_direct_l1_v1.py",
    "r16_h8_rationale_source": "src/models/optimization_v3_batch1_r16.py",
}


class VerificationError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    h = hashlib.sha256(_canonical({"dtype": str(array.dtype), "shape": list(array.shape)}))
    if array.dtype.kind in "USO":
        h.update(_canonical(array.astype(str).tolist()))
    else:
        h.update(np.ascontiguousarray(array).tobytes(order="C"))
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def _verify_signed(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != hashlib.sha256(_canonical(body)).hexdigest():
        raise VerificationError("Canonical receipt digest mismatch")
    return body


def verify_prefit() -> dict[str, Any]:
    freeze = _read_json(PREFIT_FREEZE)
    body = _verify_signed(freeze)
    if body.get("experiment_id") != EXPERIMENT or body.get("status") != "PREFIT_FROZEN_IDENTITY_ONLY":
        raise VerificationError("Prefit freeze has wrong experiment/status")
    source_hashes = body.get("source_hashes", {})
    if set(source_hashes) != set(SOURCE_RELATIVE):
        raise VerificationError("Source hash set differs from the V2 contract")
    for name, rel in SOURCE_RELATIVE.items():
        path = ROOT / rel
        if not path.is_file() or _sha_file(path) != source_hashes[name]:
            raise VerificationError(f"Frozen source hash mismatch: {name}")
    input_hashes = body.get("input_hashes", {})
    for name, item in input_hashes.items():
        path = Path(item["path"])
        if not path.is_file() or _sha_file(path) != item["sha256"]:
            raise VerificationError(f"Frozen input hash mismatch: {name}")
    if input_hashes.get("official_train", {}).get("sha256") != (
        "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
    ):
        raise VerificationError("Frozen official Train file identity differs")
    qa = _read_json(SYNTHETIC_QA)
    qa_body = _verify_signed(qa)
    if (qa_body.get("status") != "PASS_SYNTHETIC_ONLY"
            or qa_body.get("official_labels_accessed") is not False
            or qa_body.get("official_lp_starts") != 0
            or qa_body.get("test_or_sample_accessed") is not False):
        raise VerificationError("Synthetic QA does not preserve the zero-official-access boundary")
    preflight = _read_json(IDENTITY_PREFLIGHT)
    preflight_body = _verify_signed(preflight)
    if (preflight_body.get("status") != "PASS_IDENTITY_ONLY_PREFIT"
            or preflight_body.get("freeze_sha256") != _sha_file(PREFIT_FREEZE)
            or preflight_body.get("freeze_signature") != freeze["receipt_sha256"]
            or preflight_body.get("official_target_rows_parsed") != 0
            or preflight_body.get("official_lp_starts") != 0
            or preflight_body.get("official_scoring_performed") is not False
            or preflight_body.get("test_csv_or_sample_opened") is not False
            or preflight_body.get("csv_created") is not False):
        raise VerificationError("Identity preflight is not bound or violates source-only limits")
    if preflight_body.get("source_commit") != body.get("source_commit"):
        raise VerificationError("Identity preflight source commit differs from freeze")
    if body.get("official_labels_accessed") is not False or body.get("official_lp_starts") != 0:
        raise VerificationError("Prefit freeze reports official access or LP starts")
    try:
        current_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=True, text=True, capture_output=True).stdout.strip()
        current_branch = subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                                        check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VerificationError(f"Unable to verify current git source identity: {exc}") from exc
    if current_head != body.get("source_commit") or current_branch != body.get("branch"):
        raise VerificationError("Current branch/HEAD differs from the frozen source identity")
    support = body.get("support_identity", {})
    required_support_fields = {"rows_sha256", "ids_sha256", "origins_sha256",
                               "original_bundle_indices_sha256", "cycles_sha256",
                               "direct_prediction_sha256", "missing_mask_sha256", "row_counts"}
    if not required_support_fields.issubset(support):
        raise VerificationError("Frozen OOF support identity lacks physical-origin bindings")
    expected_counts = {"unique_oof": 20903, "E67_68_75": 12412, "E75_76_79": 8491,
                       "E79": 29421, "E79_missing": 1062}
    if support.get("row_counts") != expected_counts:
        raise VerificationError("Frozen OOF/E79 support counts differ from the recipe")
    expected_source_paths = {name: str((ROOT / rel).resolve()) for name, rel in SOURCE_RELATIVE.items()}
    if (preflight_body.get("source_hashes") != source_hashes
            or preflight_body.get("source_paths") != expected_source_paths):
        raise VerificationError("Identity preflight source paths/hashes differ from freeze")
    return {"status": "PASS_IDENTITY_ONLY_PREFIT", "source_commit": body["source_commit"],
            "freeze_sha256": _sha_file(PREFIT_FREEZE), "official_labels_accessed": False,
            "official_lp_starts": 0, "test_or_sample_accessed": False,
            "candidate_csvs_created": False}


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"n": int(len(err)), "mae": float(np.abs(err).sum() / len(err)),
            "mean_error": float(err.sum() / len(err)),
            "p90_absolute_error": float(np.quantile(np.abs(err), .90))}


def _groups(y: np.ndarray, pred: np.ndarray, cycles: np.ndarray,
            missing: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    masks = {"present": ~missing, "missing": missing,
             "target_le_5": y <= 5.0, "target_ge_95": y >= 95.0}
    for name, mask in masks.items():
        n = int(np.count_nonzero(mask))
        if n < 200:
            output[name] = {"n": n, "status": "UNKNOWN_UNSUPPORTED", "mae": None}
        else:
            output[name] = {"status": "SUPPORTED", **_metric(y[mask], pred[mask])}
    output["by_cycle"] = {str(int(c)): _metric(y[cycles == c], pred[cycles == c])
                          for c in np.unique(cycles)}
    return output


def _assert_numeric_equal(actual: Any, expected: Any, label: str, tol: float = 1e-12) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise VerificationError(f"Metric fields differ: {label}")
        for key, value in expected.items():
            _assert_numeric_equal(actual[key], value, f"{label}.{key}", tol)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not np.isclose(float(actual), float(expected), atol=tol, rtol=tol):
            raise VerificationError(f"Metric value differs: {label}")
        return
    if actual != expected:
        raise VerificationError(f"Metric field differs: {label}")


def _recompute_stage(record: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / record["diagnostic_rows_path"]
    if _sha_file(path) != record.get("diagnostic_rows_sha256"):
        raise VerificationError(f"Diagnostic row bundle hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in z.files}
    n = len(arrays["ids"])
    if (len(np.unique(arrays["ids"])) != n or any(len(v) != n for v in arrays.values())
            or not np.isfinite(arrays["targets"]).all()):
        raise VerificationError("Saved diagnostic rows have duplicate IDs or misaligned arrays")
    required = ("raw_direct", "causal_global_all_rows", "causal_recency_h8_all_rows",
                "causal_global_present_only", "prior_global_postselected_descriptive_only")
    for name in required:
        p = arrays[f"prediction__{name}"]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 100)):
            raise VerificationError(f"Invalid saved prediction range for {name}")
    if not np.array_equal(arrays["prediction__causal_global_present_only"][arrays["missing"]],
                          arrays["prediction__raw_direct"][arrays["missing"]]):
        raise VerificationError("Causal global present-only policy changed missing rows")
    score_lo, score_hi = record["score_cycles"]
    if not np.array_equal(np.unique(arrays["cycles"]), np.arange(score_lo, score_hi + 1)):
        raise VerificationError("Diagnostic cycles differ from the frozen score window")
    metrics = {}
    for name in required:
        p = arrays[f"prediction__{name}"]
        metrics[name] = {"overall": _metric(arrays["targets"], p),
                         "groups": _groups(arrays["targets"], p, arrays["cycles"], arrays["missing"])}
    _assert_numeric_equal(record.get("metrics"), metrics, f"{record['stage_id']}.metrics")
    _assert_numeric_equal(record.get("mae_differences"), _mae_differences(metrics),
                          f"{record['stage_id']}.mae_differences")
    direct = arrays["prediction__raw_direct"]
    prediction_audit = {}
    for name in required:
        pred = arrays[f"prediction__{name}"]
        order = np.argsort(direct, kind="stable")
        change = np.abs(pred - direct)
        prediction_audit[name] = {
            "rows": int(n), "minimum": float(pred.min()), "maximum": float(pred.max()),
            "range_within_0_100": bool(np.all((pred >= 0) & (pred <= 100))),
            "monotone_non_decreasing_in_direct_score": bool(np.all(np.diff(pred[order]) >= -1e-12)),
            "changed_rows": int(np.count_nonzero(change > 0)),
            "mean_absolute_change_vs_direct": float(change.mean()),
            "maximum_absolute_change_vs_direct": float(change.max()),
        }
    _assert_numeric_equal(record.get("prediction_audit"), prediction_audit,
                          f"{record['stage_id']}.prediction_audit")
    harm_flags = _harm_flags(metrics)
    _assert_numeric_equal(record.get("supported_missing_tail_harm_flags"), harm_flags,
                          f"{record['stage_id']}.supported_missing_tail_harm_flags")
    return arrays


def _harm_flags(metrics: dict[str, Any]) -> dict[str, Any]:
    baseline_groups = metrics["causal_global_all_rows"]["groups"]
    result = {}
    for policy, record in metrics.items():
        if policy == "causal_global_all_rows":
            continue
        groups = {}
        for name in ("missing", "target_le_5", "target_ge_95"):
            base = baseline_groups[name]
            candidate = record["groups"][name]
            supported = base.get("status") == "SUPPORTED" and candidate.get("status") == "SUPPORTED"
            delta = float(candidate["mae"] - base["mae"]) if supported else None
            groups[name] = {"status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED",
                            "mae_delta_vs_causal_global": delta,
                            "risk_flag_gt_0_25": bool(supported and delta > 0.25)}
        result[policy] = groups
    return result


def _mae_differences(metrics: dict[str, Any]) -> dict[str, Any]:
    direct = float(metrics["raw_direct"]["overall"]["mae"])
    global_mae = float(metrics["causal_global_all_rows"]["overall"]["mae"])
    return {
        "vs_raw_direct": {name: float(value["overall"]["mae"] - direct)
                          for name, value in metrics.items()},
        "vs_causal_global_all_rows": {name: float(value["overall"]["mae"] - global_mae)
                                      for name, value in metrics.items()},
    }


def _verify_source_pass(freeze: dict[str, Any]) -> tuple[dict[str, Any], str]:
    approval = _read_json(ROOT_SOURCE_PASS)
    _verify_signed(approval)
    if (approval.get("verdict") != "PASS" or approval.get("experiment_id") != EXPERIMENT
            or approval.get("source_commit") != freeze.get("source_commit")
            or approval.get("prefit_freeze_sha256") != _sha_file(PREFIT_FREEZE)
            or approval.get("source_hashes") != freeze.get("source_hashes")
            or approval.get("input_hashes") != freeze.get("input_hashes")
            or approval.get("authorized_fit_ids") != list(DIAGNOSTIC_FIT_IDS)):
        raise VerificationError("Root SOURCE_PASS does not bind the exact diagnostic source/input scope")
    return approval, _sha_file(ROOT_SOURCE_PASS)


def _rebuild_oof_and_targets(freeze: dict[str, Any]) -> dict[str, Any]:
    # Use only the pinned identity loader and the selected-row reader, never the V2 builder.
    import scripts.reg_final_wave_v2 as identity_helpers
    from scripts.reg_position_failure_mechanism_v1 import read_scored_query

    identity_helpers.verify_files(check_train=True)
    base_manifest, _base_qa, direct_freeze = identity_helpers.verify_base()
    test_ids, test_direct, _direct_check = identity_helpers.base_csv_values(base_manifest)
    with np.load(identity_helpers.PINS["base_archive"][0], allow_pickle=False) as archive:
        archive_ids = np.asarray(archive["record_id"])
        archive_direct = np.asarray(archive["prediction"], dtype=np.float64)
        test_missing = np.asarray(archive["missing"], dtype=bool)
    if (archive_ids.dtype.kind != "U" or not np.array_equal(test_ids, archive_ids)
            or not np.array_equal(test_direct, archive_direct)
            or len(test_ids) != 29421 or int(test_missing.sum()) != 1062):
        raise VerificationError("Saved E79 identity/prediction/missing mask differs from its signed archive")
    e67, e75 = identity_helpers.load_bundle(67), identity_helpers.load_bundle(75)
    identity_helpers.verify_prefixes(e67, e75, direct_freeze)
    plan = identity_helpers.stitch_oof_rows(e67, e75)
    rows = np.asarray([entry[0] for entry in plan], dtype=np.int64)
    origins = np.asarray([entry[1] for entry in plan], dtype=np.int8)
    indices = np.asarray([entry[2] for entry in plan], dtype=np.int64)
    ids = np.asarray([entry[3] for entry in plan], dtype=str)
    cycles = np.asarray([entry[4] for entry in plan], dtype=np.int16)
    direct_values, missing = [], []
    for _row, origin, index, _rid, _cycle in plan:
        bundle = e67 if origin == 67 else e75
        direct_values.append(float(bundle["prediction"][index]))
        missing.append(bool(bundle["missing_mask"][index]))
    direct = np.asarray(direct_values, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    support = freeze.get("support_identity", {})
    rebuilt = {
        "rows_sha256": _array_sha(rows), "ids_sha256": _array_sha(ids),
        "origins_sha256": _array_sha(origins),
        "original_bundle_indices_sha256": _array_sha(indices),
        "cycles_sha256": _array_sha(cycles),
        "direct_prediction_sha256": _array_sha(direct, "<f8"),
        "missing_mask_sha256": _array_sha(missing),
        "row_counts": {"unique_oof": int(len(rows)),
                        "E67_68_75": int(np.sum((cycles >= 68) & (cycles <= 75))),
                        "E75_76_79": int(np.sum((cycles >= 76) & (cycles <= 79))),
                        "E79": int(len(test_ids)), "E79_missing": int(test_missing.sum())},
    }
    if len(rows) != 20903 or len(np.unique(rows)) != len(rows) or len(np.unique(ids)) != len(ids):
        raise VerificationError("Rebuilt OOF support is not the frozen unique 20,903 rows")
    for field, value in rebuilt.items():
        if support.get(field) != value:
            raise VerificationError(f"Frozen OOF identity mismatch: {field}")
    labels, _text_load, parsed_rows, audit = read_scored_query(
        identity_helpers.SOURCE, rows, ids, cycles, expected_total_rows=50608)
    targets = np.asarray(labels, dtype=np.float64)
    if (not np.array_equal(parsed_rows, rows) or len(targets) != len(rows)
            or audit.get("sha256") != identity_helpers.TRAIN_SHA
            or audit.get("fit_target_rows_parsed") != 0
            or not np.isfinite(targets).all()):
        raise VerificationError("Selected-row Train label identity does not match the frozen OOF rows")
    return {"rows": rows, "origins": origins, "indices": indices, "ids": ids,
            "cycles": cycles, "direct": direct, "missing": missing,
            "targets": targets, "target_audit": audit}


def _recompute_h8(cycles: np.ndarray, reference_cycle: int) -> tuple[np.ndarray, dict[str, Any]]:
    cycles = np.asarray(cycles, dtype=np.int64)
    raw = np.exp2(-(reference_cycle - cycles).astype(np.float64) / 8.0)
    weights = raw / raw.mean()
    return weights, {
        "formula": "raw=2**(-(reference_cycle-cycle)/8); normalized=raw/mean(raw)",
        "reference_cycle": int(reference_cycle), "half_life_cycles": 8,
        "normalization": "arithmetic mean over exact fit rows equals one",
        "fit_rows": int(len(weights)), "sum": float(weights.sum()),
        "mean": float(weights.mean()), "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / np.dot(weights, weights)),
        "cycle_counts": {str(int(cycle)): int(np.sum(cycles == cycle))
                         for cycle in np.unique(cycles)},
        "weights_sha256": _array_sha(weights, "<f8"),
    }


def _verify_solver_objective(solver: dict[str, Any], targets: np.ndarray,
                            predictions: np.ndarray,
                            weights: np.ndarray | None) -> None:
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if (y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p)
            or not np.isfinite(y).all() or not np.isfinite(p).all()):
        raise VerificationError("Fit objective verifier received invalid aligned rows")
    intercept, slope = float(solver.get("intercept", float("nan"))), float(solver.get("slope", float("nan")))
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope < 0:
        raise VerificationError("Fit solver state has invalid affine coefficients")
    residual = np.abs(y - intercept - slope * p)
    if weights is None:
        recomputed = float(residual.sum())
        recorded = solver.get("objective_l1_sum")
        if not isinstance(recorded, (int, float)) or not np.isclose(
                recomputed, float(recorded), atol=1e-7, rtol=1e-8):
            raise VerificationError("Unweighted LP objective does not match its bound rows/coefficients")
    else:
        w = np.asarray(weights, dtype=np.float64)
        if (w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0)
                or solver.get("objective_matches_recomputed") is not True):
            raise VerificationError("Weighted LP objective or weight vector is invalid")
        recomputed = float(np.dot(w, residual))
        for name in ("weighted_l1_sum", "solver_objective"):
            recorded = solver.get(name)
            if not isinstance(recorded, (int, float)) or not np.isclose(
                    recomputed, float(recorded), atol=max(1e-5, float(w.sum()) * 1e-9), rtol=1e-8):
                raise VerificationError(f"Weighted LP objective field {name} does not match bound rows")


def _verify_lp_ledger_and_state(fit_id: str, state_sha: str,
                                expected_payload: dict[str, Any],
                                fit_targets: np.ndarray, fit_predictions: np.ndarray,
                                fit_weights: np.ndarray | None) -> dict[str, Any]:
    if not LEDGER.is_file():
        raise VerificationError("LP ledger is absent")
    events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]
    selected = [event for event in events if event.get("fit_id") == fit_id]
    reserve = [event for event in selected if event.get("event") == "RESERVE"]
    started = [event for event in selected if event.get("event") == "LP_STARTED"]
    complete = [event for event in selected if event.get("event") == "LP_COMPLETE"]
    failed = [event for event in selected if event.get("event") == "LP_FAILED"]
    if len(reserve) != 1 or len(started) != 1 or len(complete) != 1 or failed:
        raise VerificationError(f"LP ledger does not contain one complete lifecycle for {fit_id}")
    reservation_id = reserve[0].get("reservation_id")
    for event in (reserve[0], started[0], complete[0]):
        if (event.get("reservation_id") != reservation_id
                or event.get("fit_input") != expected_payload
                or event.get("source_commit") != expected_payload["source_commit"]
                or event.get("root_source_pass_sha256") != expected_payload["root_source_pass_sha256"]):
            raise VerificationError(f"LP ledger binding mismatch for {fit_id}")
    path = RUN_DIR / "fits" / f"{fit_id}.json"
    state = _read_json(path)
    body = _verify_signed(state)
    if (_sha_file(path) != state_sha or complete[0].get("state_sha256") != state_sha
            or body.get("status") != "LP_COMPLETE"):
        raise VerificationError(f"LP state/terminal ledger digest mismatch for {fit_id}")
    for key, value in expected_payload.items():
        if body.get(key) != value:
            raise VerificationError(f"LP state identity mismatch for {fit_id}: {key}")
    if not body.get("solver_result", {}).get("success"):
        raise VerificationError(f"LP solver state is not successful for {fit_id}")
    _verify_solver_objective(body["solver_result"], fit_targets, fit_predictions, fit_weights)
    return state


def _verify_stage_bindings(record: dict[str, Any], oof: dict[str, Any],
                          freeze: dict[str, Any], approval_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = {"D1": (71, 72, 75, 71), "D2": (75, 76, 79, 75)}
    stage_id = record.get("stage_id")
    if stage_id not in settings:
        raise VerificationError(f"Unexpected diagnostic stage: {stage_id}")
    cutoff, score_lo, score_hi, reference = settings[stage_id]
    cycles = oof["cycles"]
    history = (cycles >= 68) & (cycles <= cutoff)
    score = (cycles >= score_lo) & (cycles <= score_hi)
    if (record.get("fit_cycles") != [68, cutoff] or record.get("score_cycles") != [score_lo, score_hi]
            or not history.any() or not score.any() or np.any(history & score)):
        raise VerificationError(f"Temporal fit/evaluation masks differ for {stage_id}")
    y_fit, p_fit, c_fit = oof["targets"][history], oof["direct"][history], cycles[history]
    row_identity = {name: oof[key][history] for name, key in (
        ("rows", "rows"), ("ids", "ids"), ("origins", "origins"), ("indices", "indices"))}
    fit_hashes = {name: _array_sha(value) for name, value in row_identity.items()}
    if record.get("fit_row_identity") != fit_hashes:
        raise VerificationError(f"Per-fit row identity hashes differ for {stage_id}")
    score_identity = {
        "physical_rows_sha256": _array_sha(oof["rows"][score]),
        "record_ids_sha256": _array_sha(oof["ids"][score]),
        "prediction_origins_sha256": _array_sha(oof["origins"][score]),
        "original_bundle_indices_sha256": _array_sha(oof["indices"][score]),
        "direct_predictions_sha256": _array_sha(oof["direct"][score], "<f8"),
    }
    if (record.get("score_row_identity") != score_identity
            or record.get("fit_rows") != int(history.sum())
            or record.get("score_rows") != int(score.sum())
            or record.get("fit_ids_sha256") != _array_sha(oof["ids"][history])
            or record.get("score_ids_sha256") != _array_sha(oof["ids"][score])
            or record.get("score_targets_sha256") != _array_sha(oof["targets"][score], "<f8")):
        raise VerificationError(f"Diagnostic support/target receipt differs for {stage_id}")

    weights, weight_digest = _recompute_h8(c_fit, reference)
    _assert_numeric_equal(record.get("weights"), weight_digest,
                          f"{stage_id}.weights")

    source_pass_sha = approval_sha
    freeze_sha = _sha_file(PREFIT_FREEZE)
    states = {}
    for fit_id, weight_values, state_sha_field in (
        (f"{stage_id}_GLOBAL", None, "global_fit_state_sha256"),
        (f"{stage_id}_RECENCY_H8", weights, "recency_fit_state_sha256"),
    ):
        expected = {
            "experiment_id": EXPERIMENT, "fit_id": fit_id, "fit_rows": int(len(y_fit)),
            "cycles_sha256": _array_sha(c_fit),
            "physical_rows_sha256": _array_sha(row_identity["rows"]),
            "record_ids_sha256": _array_sha(row_identity["ids"]),
            "prediction_origins_sha256": _array_sha(row_identity["origins"]),
            "original_bundle_indices_sha256": _array_sha(row_identity["indices"]),
            "targets_sha256": _array_sha(y_fit, "<f8"),
            "predictions_sha256": _array_sha(p_fit, "<f8"),
            "weights_sha256": _array_sha(weight_values, "<f8") if weight_values is not None else None,
            "source_commit": freeze["source_commit"], "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"], "prefit_freeze_sha256": freeze_sha,
            "root_source_pass_sha256": source_pass_sha, "fit_ids_allowlist": list(FIT_IDS),
        }
        states[fit_id] = _verify_lp_ledger_and_state(
            fit_id, record.get(state_sha_field), expected, y_fit, p_fit, weight_values)
    gstate = states[f"{stage_id}_GLOBAL"]["solver_result"]
    hstate = states[f"{stage_id}_RECENCY_H8"]["solver_result"]
    if (record.get("global_coefficients") != {"intercept": gstate["intercept"], "slope": gstate["slope"]}
            or record.get("recency_coefficients") != {"intercept": hstate["intercept"], "slope": hstate["slope"]}):
        raise VerificationError(f"Reported affine coefficients differ from signed fit states for {stage_id}")
    return {"global": states[f"{stage_id}_GLOBAL"], "h8": states[f"{stage_id}_RECENCY_H8"]}, {
        "history": history, "score": score, "weights": weights,
        "global_coefficients": gstate, "h8_coefficients": hstate,
    }


def verify_diagnostics(*, allow_production_start: bool = False) -> dict[str, Any]:
    prefit = verify_prefit()
    freeze_body = _verify_signed(_read_json(PREFIT_FREEZE))
    report = _read_json(DIAGNOSTICS)
    body = _verify_signed(report)
    _approval, approval_sha = _verify_source_pass(freeze_body)
    if (body.get("experiment_id") != EXPERIMENT
            or body.get("status") != "FOUR_CAUSAL_DIAGNOSTICS_COMPLETE"
            or body.get("source_commit") != prefit["source_commit"]
            or body.get("prefit_freeze_sha256") != prefit["freeze_sha256"]
            or body.get("root_source_pass_sha256") != approval_sha
            or body.get("source_hashes") != freeze_body.get("source_hashes")
            or body.get("input_hashes") != freeze_body.get("input_hashes")
            or body.get("test_or_sample_accessed") is not False
            or body.get("csv_created") is not False
            or body.get("kaggle_upload") is not False
            or body.get("final_production_lp_started") is not False
            or body.get("diagnostic_lp_ids") != list(DIAGNOSTIC_FIT_IDS)):
        raise VerificationError("Diagnostic report identity/scope mismatch")
    records = body.get("windows", [])
    if [r.get("stage_id") for r in records] != ["D1", "D2"]:
        raise VerificationError("Diagnostic report must contain ordered D1 and D2 only")
    oof = _rebuild_oof_and_targets(freeze_body)
    if body.get("train_target_audit") != oof["target_audit"]:
        raise VerificationError("Diagnostic Train target audit differs from independently selected rows")
    stage_arrays = []
    stage_bindings = []
    for record in records:
        states, binding = _verify_stage_bindings(record, oof, freeze_body, approval_sha)
        arrays = _recompute_stage(record)
        score = binding["score"]
        for field, actual in (("rows", oof["rows"][score]), ("ids", oof["ids"][score]),
                              ("origins", oof["origins"][score]), ("indices", oof["indices"][score]),
                              ("cycles", oof["cycles"][score]), ("targets", oof["targets"][score]),
                              ("missing", oof["missing"][score])):
            if not np.array_equal(arrays[field], actual):
                raise VerificationError(f"Persisted {record['stage_id']} diagnostic {field} differs from rebuilt OOF")
        direct = oof["direct"][score]
        expected_global = np.clip(binding["global_coefficients"]["intercept"]
                                  + binding["global_coefficients"]["slope"] * direct, 0.0, 100.0)
        expected_h8 = np.clip(binding["h8_coefficients"]["intercept"]
                              + binding["h8_coefficients"]["slope"] * direct, 0.0, 100.0)
        raw_global = (binding["global_coefficients"]["intercept"]
                      + binding["global_coefficients"]["slope"] * direct)
        raw_h8 = (binding["h8_coefficients"]["intercept"]
                  + binding["h8_coefficients"]["slope"] * direct)
        raw_prior = 2.169196428141562 + 0.9744451489746011 * direct
        clip_counts = {"global": int(np.count_nonzero((raw_global < 0) | (raw_global > 100))),
                       "recency_h8": int(np.count_nonzero((raw_h8 < 0) | (raw_h8 > 100))),
                       "prior_global_descriptive": int(np.count_nonzero((raw_prior < 0) | (raw_prior > 100)))}
        if record.get("clip_count") != clip_counts:
            raise VerificationError(f"Prediction clip counts differ for {record['stage_id']}")
        expected_present = expected_global.copy()
        expected_present[oof["missing"][score]] = direct[oof["missing"][score]]
        expected_prior = np.clip(2.169196428141562 + 0.9744451489746011 * direct, 0.0, 100.0)
        for name, expected in (("raw_direct", direct), ("causal_global_all_rows", expected_global),
                               ("causal_recency_h8_all_rows", expected_h8),
                               ("causal_global_present_only", expected_present),
                               ("prior_global_postselected_descriptive_only", expected_prior)):
            if not np.array_equal(arrays[f"prediction__{name}"], expected):
                raise VerificationError(f"{record['stage_id']} prediction equation mismatch: {name}")
        stage_arrays.append(arrays)
        stage_bindings.append((states, binding))
    ids = np.concatenate([a["ids"] for a in stage_arrays])
    if len(np.unique(ids)) != len(ids) or len(ids) != 15953:
        raise VerificationError("Pooled W5+W6 support is not 15,953 unique IDs")
    y = np.concatenate([a["targets"] for a in stage_arrays])
    cyc = np.concatenate([a["cycles"] for a in stage_arrays])
    missing = np.concatenate([a["missing"] for a in stage_arrays])
    names = ("raw_direct", "causal_global_all_rows", "causal_recency_h8_all_rows",
             "causal_global_present_only", "prior_global_postselected_descriptive_only")
    metrics = {}
    for name in names:
        pred = np.concatenate([a[f"prediction__{name}"] for a in stage_arrays])
        metrics[name] = {"overall": _metric(y, pred), "groups": _groups(y, pred, cyc, missing)}
    prediction_audits = {}
    direct_all = np.concatenate([a["prediction__raw_direct"] for a in stage_arrays])
    for name in names:
        pred = np.concatenate([a[f"prediction__{name}"] for a in stage_arrays])
        order = np.argsort(direct_all, kind="stable")
        changes = np.abs(pred - direct_all)
        prediction_audits[name] = {
            "rows": int(len(pred)), "minimum": float(pred.min()), "maximum": float(pred.max()),
            "range_within_0_100": bool(np.all((pred >= 0) & (pred <= 100))),
            "monotone_non_decreasing_in_direct_score": bool(np.all(np.diff(pred[order]) >= -1e-12)),
            "changed_rows": int(np.count_nonzero(changes > 0)),
            "mean_absolute_change_vs_direct": float(changes.mean()),
            "maximum_absolute_change_vs_direct": float(changes.max()),
        }
    harm_flags = _harm_flags(metrics)
    pooled = body.get("pooled_unique_W5_W6", {})
    _assert_numeric_equal(pooled.get("metrics"), metrics, "pooled_unique_W5_W6.metrics")
    _assert_numeric_equal(pooled.get("prediction_audit"), prediction_audits,
                          "pooled_unique_W5_W6.prediction_audit")
    _assert_numeric_equal(pooled.get("supported_missing_tail_harm_flags"), harm_flags,
                          "pooled_unique_W5_W6.supported_missing_tail_harm_flags")
    _assert_numeric_equal(pooled.get("mae_differences"), _mae_differences(metrics),
                          "pooled_unique_W5_W6.mae_differences")
    window_deltas = {}
    for record in records:
        m = record["metrics"]
        window_deltas[record["stage_id"]] = {
            "recency_minus_global_mae": (m["causal_recency_h8_all_rows"]["overall"]["mae"]
                                         - m["causal_global_all_rows"]["overall"]["mae"]),
            "global_present_only_minus_global_mae": (
                m["causal_global_present_only"]["overall"]["mae"]
                - m["causal_global_all_rows"]["overall"]["mae"]),
        }
    _assert_numeric_equal(pooled.get("window_delta_mae"), window_deltas,
                          "pooled_unique_W5_W6.window_delta_mae")
    pooled_global = metrics["causal_global_all_rows"]["overall"]["mae"]
    pooled_h8 = metrics["causal_recency_h8_all_rows"]["overall"]["mae"]
    pooled_present = metrics["causal_global_present_only"]["overall"]["mae"]
    h8_pass = (pooled_global - pooled_h8 >= 0.02
               and all(delta["recency_minus_global_mae"] <= 0.03
                       for delta in window_deltas.values()))
    present_pass = (pooled_present <= pooled_global
                    and all(delta["global_present_only_minus_global_mae"] <= 0.03
                            for delta in window_deltas.values()))
    expected_screens = {
        "recency_h8_support": {"required_pooled_gain": 0.02, "max_window_harm": 0.03,
                               "status": "PASS_SCREEN" if h8_pass else "FAIL_SCREEN"},
        "global_present_only_support": {"required_pooled_no_harm": True,
                                        "max_window_harm": 0.03,
                                        "status": "PASS_SCREEN" if present_pass else "FAIL_SCREEN"},
    }
    _assert_numeric_equal(pooled.get("predeclared_screen"), expected_screens,
                          "pooled_unique_W5_W6.predeclared_screen")
    if pooled.get("rows") != len(y) or pooled.get("unique_record_ids") != len(np.unique(ids)):
        raise VerificationError("Pooled diagnostic denominator mismatch")
    if (body.get("unique_evaluation_rows") != len(y)
            or body.get("official_lp_starts") != len(DIAGNOSTIC_FIT_IDS)):
        raise VerificationError("Diagnostic row count or official LP start count differs from the plan")
    ledger_events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]
    started_ids = [event.get("fit_id") for event in ledger_events
                   if event.get("event") == "LP_STARTED"]
    expected_started_ids = list(DIAGNOSTIC_FIT_IDS)
    if allow_production_start:
        expected_started_ids.append("FINAL_RECENCY_H8")
    if started_ids != expected_started_ids:
        raise VerificationError("LP ledger has unexpected starts, order, or an unstarted final production fit")
    return {"status": "PASS_INDEPENDENT_DIAGNOSTIC_RECOMPUTATION",
            "stages": [r["stage_id"] for r in records], "pooled_unique_rows": len(y),
            "metric_recomputation": "PASS", "test_or_sample_accessed": False}


def _read_candidate_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise VerificationError(f"Candidate CSV is empty: {path}") from exc
        if header != ["record_id", "threat_rank"]:
            raise VerificationError(f"Candidate CSV schema differs: {path}")
        ids, predictions = [], []
        for row_no, row in enumerate(reader, 2):
            if len(row) != 2:
                raise VerificationError(f"Candidate CSV row {row_no} has wrong width: {path}")
            ids.append(row[0])
            try:
                predictions.append(float(row[1]))
            except ValueError as exc:
                raise VerificationError(f"Candidate CSV row {row_no} has a nonnumeric prediction") from exc
    return np.asarray(ids, dtype=str), np.asarray(predictions, dtype=np.float64)


def verify_production() -> dict[str, Any]:
    prefit = verify_prefit()
    freeze = _verify_signed(_read_json(PREFIT_FREEZE))
    source_pass, source_pass_sha = _verify_source_pass(freeze)
    _diagnostic_check = verify_diagnostics(allow_production_start=True)
    diagnostic = _verify_signed(_read_json(DIAGNOSTICS))
    production_pass = _read_json(ROOT_PRODUCTION_PASS)
    _verify_signed(production_pass)
    production_pass_sha = _sha_file(ROOT_PRODUCTION_PASS)
    if (production_pass.get("verdict") != "PASS"
            or production_pass.get("experiment_id") != EXPERIMENT
            or production_pass.get("source_commit") != freeze.get("source_commit")
            or production_pass.get("prefit_freeze_sha256") != prefit["freeze_sha256"]
            or production_pass.get("root_source_pass_sha256") != source_pass_sha
            or production_pass.get("diagnostics_sha256") != _sha_file(DIAGNOSTICS)
            or production_pass.get("source_hashes") != freeze.get("source_hashes")
            or production_pass.get("input_hashes") != freeze.get("input_hashes")
            or production_pass.get("authorized_fit_ids") != ["FINAL_RECENCY_H8"]
            or production_pass.get("authorized_candidates") != ["A", "B"]
            or production_pass.get("candidate_csvs_max") != 2):
        raise VerificationError("Root production approval does not bind the exact final fit/candidate scope")
    result = _verify_signed(_read_json(PRODUCTION_RESULT))
    if (result.get("status") != "CANDIDATES_READY_LOCAL_EXPLORATORY"
            or result.get("root_source_pass_sha256") != source_pass_sha
            or result.get("root_production_pass_sha256") != production_pass_sha
            or result.get("diagnostics_sha256") != _sha_file(DIAGNOSTICS)
            or result.get("official_lp_starts") != len(FIT_IDS)
            or result.get("official_lp_start_cap") != len(FIT_IDS)
            or result.get("test_labels_read") is not False
            or result.get("sample_submission_read") is not False
            or result.get("kaggle_upload") is not False
            or result.get("official_labels_accessed") is not True):
        raise VerificationError("Production result receipt identity/scope mismatch")

    final_state = _read_json(FINAL_STATE)
    final_body = _verify_signed(final_state)
    final_state_sha = _sha_file(FINAL_STATE)
    app_body = _verify_signed(_read_json(FINAL_APPLICATION_FREEZE))
    if (app_body.get("status") != "FINAL_RECENCY_APPLICATION_FROZEN"
            or app_body.get("prefit_freeze_sha256") != prefit["freeze_sha256"]
            or app_body.get("source_pass_sha256") != source_pass_sha
            or app_body.get("production_pass_sha256") != production_pass_sha
            or app_body.get("diagnostics_sha256") != _sha_file(DIAGNOSTICS)
            or app_body.get("final_fit_state_sha256") != final_state_sha
            or app_body.get("source_hashes") != freeze.get("source_hashes")
            or app_body.get("input_hashes") != freeze.get("input_hashes")
            or app_body.get("test_labels_read") is not False
            or app_body.get("sample_submission_read") is not False):
        raise VerificationError("Final application freeze is not bound to approved local production")

    oof = _rebuild_oof_and_targets(freeze)
    history = (oof["cycles"] >= 68) & (oof["cycles"] <= 79)
    weights, weight_info = _recompute_h8(oof["cycles"][history], 79)
    row_identity = {name: oof[key][history] for name, key in (
        ("rows", "rows"), ("ids", "ids"), ("origins", "origins"), ("indices", "indices"))}
    final_expected = {
        "experiment_id": EXPERIMENT, "fit_id": "FINAL_RECENCY_H8", "fit_rows": int(history.sum()),
        "cycles_sha256": _array_sha(oof["cycles"][history]),
        "physical_rows_sha256": _array_sha(row_identity["rows"]),
        "record_ids_sha256": _array_sha(row_identity["ids"]),
        "prediction_origins_sha256": _array_sha(row_identity["origins"]),
        "original_bundle_indices_sha256": _array_sha(row_identity["indices"]),
        "targets_sha256": _array_sha(oof["targets"][history], "<f8"),
        "predictions_sha256": _array_sha(oof["direct"][history], "<f8"),
        "weights_sha256": _array_sha(weights, "<f8"),
        "source_commit": freeze["source_commit"], "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"], "prefit_freeze_sha256": prefit["freeze_sha256"],
        "root_source_pass_sha256": source_pass_sha, "fit_ids_allowlist": list(FIT_IDS),
    }
    final_state = _verify_lp_ledger_and_state("FINAL_RECENCY_H8", final_state_sha, final_expected,
                                              oof["targets"][history], oof["direct"][history], weights)
    if (final_state.get("status") != "LP_COMPLETE"
            or final_state.get("solver_result", {}).get("slope", -1) < 0
            or app_body.get("final_fit_state_sha256") != final_state_sha):
        raise VerificationError("Final H8 state is incomplete or violates the nonnegative-slope contract")
    ledger_events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]
    final_started = [event.get("fit_id") for event in ledger_events
                     if event.get("event") == "LP_STARTED"]
    if final_started != list(FIT_IDS):
        raise VerificationError("Final production LP ledger order/cap differs from the five-slot plan")

    import scripts.reg_final_wave_v2 as identity_helpers
    identity_helpers.verify_files(check_train=True)
    base_manifest, _base_qa, direct_freeze = identity_helpers.verify_base()
    ids, direct_values, _direct_check = identity_helpers.base_csv_values(base_manifest)
    with np.load(identity_helpers.PINS["base_archive"][0], allow_pickle=False) as archive:
        archived_ids = np.asarray(archive["record_id"])
        direct_archive = np.asarray(archive["prediction"], dtype=np.float64)
        missing = np.asarray(archive["missing"], dtype=bool)
    if (not np.array_equal(ids, archived_ids) or not np.array_equal(direct_values, direct_archive)
            or ids.shape != (29421,) or len(np.unique(ids)) != len(ids) or int(missing.sum()) != 1062):
        raise VerificationError("Saved E79 identity bundle failed independent round-trip identity")
    if (app_body.get("test_identity_sha256") != _array_sha(ids)
            or app_body.get("test_direct_prediction_sha256") != _array_sha(direct_values, "<f8")
            or app_body.get("test_missing_mask_sha256") != _array_sha(missing)):
        raise VerificationError("Final application freeze E79 identities differ from saved source")

    global_ids, global_values = _read_candidate_csv(identity_helpers.OUT_CSV)
    if (not np.array_equal(global_ids, ids) or not np.isfinite(global_values).all()
            or not np.array_equal(global_values, direct_values * 0.9744451489746011 + 2.169196428141562)):
        raise VerificationError("Saved prior global-affine CSV differs from E79 identity/coefficients")
    expected_a = np.clip(float(final_state["solver_result"]["intercept"])
                         + float(final_state["solver_result"]["slope"]) * direct_values, 0.0, 100.0)
    raw_a = (float(final_state["solver_result"]["intercept"])
             + float(final_state["solver_result"]["slope"]) * direct_values)
    expected_b = global_values.copy()
    expected_b[missing] = direct_values[missing]
    expected_by_candidate = {"A": expected_a, "B": expected_b}
    expected_fit_sha = {"A": final_state_sha,
                        "B": freeze["input_hashes"]["global_fit_receipt"]["sha256"]}
    expected_recipes = {
        "A": {"identity": "RECENCY_AFFINE_H8",
              "intercept": final_state["solver_result"]["intercept"],
              "slope": final_state["solver_result"]["slope"],
              "weights": weight_info,
              "apply_to": "all saved E79 direct predictions, including former fallback rows",
              "clip_at_prediction_only": True,
              "clip_count": int(np.count_nonzero((raw_a < 0) | (raw_a > 100)))},
        "B": {"identity": "GLOBAL_AFFINE_PRESENT_ONLY",
              "intercept": 2.169196428141562, "slope": 0.9744451489746011,
              "apply_to": "saved signed global affine CSV present rows; raw direct restored on exact missing mask",
              "new_lp_calls": 0},
    }
    for candidate in ("A", "B"):
        csv_info = result.get(f"candidate_{candidate}", {})
        manifest = _verify_signed(_read_json(MANIFESTS[candidate]))
        output_path = OUTPUTS[candidate]
        candidate_ids, candidate_values = _read_candidate_csv(output_path)
        csv_sha = _sha_file(output_path)
        prediction_sha = _array_sha(expected_by_candidate[candidate], "<f8")
        if (csv_info.get("status") != "READY_LOCAL_EXPLORATORY"
                or csv_info.get("csv_path") != output_path.relative_to(ROOT).as_posix()
                or csv_info.get("csv_sha256") != csv_sha
                or csv_info.get("rows") != 29421
                or csv_info.get("manifest_sha256") != _sha_file(MANIFESTS[candidate])
                or manifest.get("candidate") != ("RECENCY_AFFINE_H8" if candidate == "A" else "GLOBAL_AFFINE_PRESENT_ONLY")
                or manifest.get("status") != "READY_LOCAL_EXPLORATORY"
                or manifest.get("csv_path") != output_path.relative_to(ROOT).as_posix()
                or manifest.get("csv_sha256") != csv_sha
                or manifest.get("rows") != 29421
                or manifest.get("ordered_ids_sha256") != _array_sha(ids)
                or manifest.get("prediction_sha256") != prediction_sha
                or manifest.get("fit_state_sha256") != expected_fit_sha[candidate]
                or manifest.get("input_hashes") != freeze["input_hashes"]
                or manifest.get("support_rows") != 20903
                or manifest.get("train_target_sha256") != oof["target_audit"].get("query_target_values_sha256")
                or manifest.get("recipe") != expected_recipes[candidate]
                or manifest.get("test_or_sample_csv_opened") is not False
                or manifest.get("kaggle_upload") is not False
                or not np.array_equal(candidate_ids, ids)
                or not np.array_equal(candidate_values, expected_by_candidate[candidate])
                or not np.isfinite(candidate_values).all()
                or np.any((candidate_values < 0) | (candidate_values > 100))):
            raise VerificationError(f"Candidate {candidate} CSV/manifest differs from its approved recipe")
    return {"status": "PASS_INDEPENDENT_PRODUCTION_AND_CSV_RECOMPUTATION",
            "candidates": ["A", "B"], "candidate_rows_each": 29421,
            "test_or_sample_csv_opened": False, "kaggle_upload": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-prefit", action="store_true")
    group.add_argument("--verify-diagnostics", action="store_true")
    group.add_argument("--verify-production", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify_prefit:
            result = verify_prefit()
        elif args.verify_diagnostics:
            result = verify_diagnostics()
        else:
            result = verify_production()
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
                          "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
