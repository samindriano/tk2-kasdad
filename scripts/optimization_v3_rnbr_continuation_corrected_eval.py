"""Post-fit corrected per-view evaluation for the frozen RNBR continuation.

The committed frozen verifier is preserved. Its shared prediction maps were
overwritten with E67 values before W5/W6 metrics were computed. This adapter
reuses the same frozen inputs, recipes, windows, and gates while keeping each
view's prediction maps isolated, and archives the original verifier output.
No fitting is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_continuation as runner  # noqa: E402
from scripts import optimization_v3_rnbr_continuation_verify as verifier  # noqa: E402
from scripts import optimization_v3_rnbr_official as pilot  # noqa: E402
from scripts.optimization_v3_rnbr_continuation_recovery_bridge import (  # noqa: E402
    load_verified_e67_observed,
    verified_fit_source_commit,
)

REPORT_DIR = ROOT / "reports/optimization_v3/rnbr_continuation_v1"
FIT_IDS = [f"RNBR_CONT_E{o}_{a.upper()}" for o in runner.ORIGINS for a in runner.ARMS]
WINDOWS = {
    "W1": (55, 56, 59), "W2": (59, 60, 63), "W3": (63, 64, 67),
    "W4": (67, 68, 71), "W5": (71, 72, 75), "W6": (75, 76, 79),
}


class CorrectedEvaluationError(RuntimeError):
    """Fail-closed error in the post-fit corrected evaluation."""


def _map(rows: np.ndarray, prediction: np.ndarray, expected: np.ndarray, label: str) -> dict[int, float]:
    rows = np.asarray(rows, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.int64)
    if (len(rows) != len(prediction) or len(np.unique(rows)) != len(rows)
            or not np.array_equal(np.sort(rows), np.sort(expected))
            or not np.isfinite(prediction).all()):
        raise CorrectedEvaluationError(f"{label} prediction rows do not exactly cover the declared window")
    return {int(r): float(p) for r, p in zip(rows, prediction)}


def _subset_bundle(arrays: dict[str, np.ndarray], rows: np.ndarray, label: str) -> dict[int, float]:
    row_values = np.asarray(arrays["score_rows"], dtype=np.int64)
    mask = np.isin(row_values, rows)
    return _map(row_values[mask], np.asarray(arrays["prediction"])[mask], rows, label)


def assemble_prediction_maps(
    view_rows: dict[str, np.ndarray],
    continuation_arrays: dict[str, dict[str, np.ndarray]],
    pilot_arrays: dict[tuple[int, str], dict[str, np.ndarray]],
    r06_by_origin: dict[int, dict[int, float]],
    h12_rows: np.ndarray,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]], dict[int, float], dict[int, float]]:
    """Build isolated primary windows and a separate inherited E67 H12 view."""
    observed: dict[str, dict[int, float]] = {}
    shuffled: dict[str, dict[int, float]] = {}
    for view, (origin, _, _) in WINDOWS.items():
        rows = np.asarray(view_rows[view], dtype=np.int64)
        if view == "W1":
            obs_map = _map(rows, np.asarray([r06_by_origin[origin][int(r)] for r in rows]), rows, "W1 R06 fallback")
            shuffle_map = dict(obs_map)
        elif view in ("W2", "W3", "W5"):
            fit_id = f"RNBR_CONT_E{origin}_OBSERVED"
            shuffle_id = f"RNBR_CONT_E{origin}_SHUFFLED"
            obs_map = _subset_bundle(continuation_arrays[fit_id], rows, view)
            shuffle_map = _subset_bundle(continuation_arrays[shuffle_id], rows, f"{view} shuffled")
        else:
            obs_map = _subset_bundle(pilot_arrays[(origin, "observed")], rows, view)
            shuffle_map = _subset_bundle(pilot_arrays[(origin, "shuffled")], rows, f"{view} shuffled")
        observed[view], shuffled[view] = obs_map, shuffle_map
    # H12 is intentionally built separately; overlapping rows retain E67 predictions
    # here without replacing primary W5/E71 or W6/E75 forecasts.
    h12_observed = _subset_bundle(pilot_arrays[(67, "observed")], h12_rows, "H12 observed E67")
    h12_shuffled = _subset_bundle(pilot_arrays[(67, "shuffled")], h12_rows, "H12 shuffled E67")
    return observed, shuffled, h12_observed, h12_shuffled


def _load_frozen_r22(freeze: dict[str, Any], train: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    """Read only the frozen R22 prediction arrays required by the original comparison."""
    values = {}
    for origin_text, spec in freeze["r22_comparison_inventory"].items():
        origin = int(origin_text)
        path = ROOT / spec["npz"]
        if pilot.sha256_file(path) != spec["npz_sha256"]:
            raise CorrectedEvaluationError(f"Frozen R22 cache hash mismatch at E{origin}")
        with np.load(path, allow_pickle=False) as z:
            rows = np.asarray(z["score_rows"], dtype=np.int64).copy()
            prediction = np.asarray(z[spec["prediction_array"]], dtype=np.float64).copy()
            ids = np.asarray(z["score_record_ids"]).astype(str) if "score_record_ids" in z.files else None
            cycles = np.asarray(z["score_cycles"], dtype=np.int64) if "score_cycles" in z.files else None
        if (len(rows) != len(prediction) or len(np.unique(rows)) != len(rows)
                or np.any((rows < 0) | (rows >= len(train["ids"])))
                or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))):
            raise CorrectedEvaluationError(f"Frozen R22 predictions fail row/value validation at E{origin}")
        if ids is not None and not np.array_equal(ids, train["ids"][rows]):
            raise CorrectedEvaluationError(f"Frozen R22 IDs do not align with official Train at E{origin}")
        if cycles is not None and not np.array_equal(cycles, train["cycles"][rows]):
            raise CorrectedEvaluationError(f"Frozen R22 cycles do not align with official Train at E{origin}")
        values[origin] = {"rows": rows, "prediction": prediction}
    return values


def _archive_initial_outputs() -> dict[str, str]:
    archived = {}
    for name in ("results.json", "RESULTS.md"):
        source = REPORT_DIR / name
        dest = REPORT_DIR / f"frozen_verifier_output_pre_correction_{name.replace('.', '_', 1)}"
        if not source.is_file():
            raise CorrectedEvaluationError(f"Expected initial frozen evaluator output is missing: {name}")
        raw = source.read_bytes()
        if dest.exists():
            archived_raw = dest.read_bytes()
            if archived_raw != raw:
                already_corrected = (b"POSTFIT_CORRECTED_ISOLATED_VIEW_MAPS_V1" in raw
                                     if name == "results.json"
                                     else b"corrected post-fit results" in raw)
                if not already_corrected:
                    raise CorrectedEvaluationError(f"Refusing to replace a different preserved evaluator output: {dest.name}")
            archived[name] = hashlib.sha256(archived_raw).hexdigest()
        else:
            dest.write_bytes(raw)
            archived[name] = hashlib.sha256(raw).hexdigest()
    return archived


def evaluate() -> dict[str, Any]:
    freeze = runner.load_freeze()
    source_commit = verified_fit_source_commit(freeze)
    train = runner.targetfree_train()
    r06_cache = runner.load_r06(freeze, train)

    # Verify all six durable fit bundles again before any query targets are opened.
    verified = {}
    verified_arrays: dict[str, dict[str, np.ndarray]] = {}
    for fit_id in FIT_IDS:
        arrays, receipt = runner.load_bundle(fit_id, freeze, source_commit=source_commit)
        state_path = runner.RUN_ROOT / "bundles" / fit_id / "fit_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("fit_id") != fit_id
                or state.get("freeze_signature") != freeze["freeze_signature"]
                or state.get("state") != "DURABLE_AND_INDEPENDENTLY_VERIFIED"
                or state.get("verification", {}).get("status") != "PASS"):
            raise CorrectedEvaluationError(f"Fit persistence state failed before target access: {fit_id}")
        independent = verifier.verify_fit_bundle(fit_id, arrays, receipt, freeze, train, r06_cache, source_commit)
        if independent.get("status") != "PASS":
            raise CorrectedEvaluationError(f"Independent bundle check failed: {fit_id}")
        verified_arrays[fit_id] = arrays
        receipt_path = runner.RUN_ROOT / "bundles" / fit_id / "receipt.json"
        npz_path = runner.RUN_ROOT / "bundles" / fit_id / "arrays.npz"
        verified[fit_id] = {"status": "PASS", "fit_id": fit_id,
                            "donor_rows": independent["donor_rows"],
                            "query_rows": independent["query_rows"],
                            "independent_neighbor_values_recomputed": independent["independent_neighbor_values_recomputed"],
                            "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                            "receipt_integrity_sha256": receipt["integrity_sha256"],
                            "bundle_npz_sha256": receipt["npz_sha256"],
                            "query_targets_read_during_fit_verification": False}

    prior_path = pilot.REPORT_ROOT / "INDEPENDENT_VERIFICATION.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    pilot_freeze = json.loads(runner.PILOT_FREEZE.read_text(encoding="utf-8"))
    if (prior.get("status") != "PASS"
            or prior.get("freeze_signature") != pilot_freeze.get("freeze_signature")):
        raise CorrectedEvaluationError("Inherited pilot verification is not PASS")
    # Validate/recover the E67-observed safe serialization, then adapt only in memory.
    e67_safe, e67_receipt, bridge_audit = load_verified_e67_observed()
    old_loader = pilot.load_bundle

    def recovered_pilot_loader(fit_id: str, requested_freeze: Any, *, source_commit: str | None = None):
        if fit_id == "RNBR_PILOT_E67_OBSERVED":
            if requested_freeze.get("freeze_signature") != e67_receipt["identity"].get("freeze_signature"):
                raise CorrectedEvaluationError("Recovered E67 pilot requested under a different freeze")
            if source_commit is not None and source_commit != e67_receipt["identity"].get("source_commit"):
                raise CorrectedEvaluationError("Recovered E67 pilot requested under a different fit commit")
            return e67_safe, e67_receipt
        return old_loader(fit_id, requested_freeze, source_commit=source_commit)

    pilot.load_bundle = recovered_pilot_loader
    try:
        e67_obs, e67_shuffle = verifier._pilot_arrays(67, pilot_freeze)
        e75_obs, e75_shuffle = verifier._pilot_arrays(75, pilot_freeze)
    finally:
        pilot.load_bundle = old_loader

    ledger = verifier._verify_final_ledger()
    if ledger.get("status") != "PASS":
        raise CorrectedEvaluationError("Final shared fit ledger does not reconcile")

    # Query Train targets are allowed only after all six new fits have passed again.
    eval_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= 79)).astype(np.int64)
    truth = verifier._target_rows(eval_rows, train, "post-fit corrected per-view RNBR continuation evaluation")
    y_by = {int(r): float(y) for r, y in zip(eval_rows, truth["target"])}
    missing_by = {int(r): bool(not train["present"][r]) for r in eval_rows}
    r22_cache = _load_frozen_r22(freeze, train)

    view_rows: dict[str, np.ndarray] = {}
    r06_by: dict[int, float] = {}
    r22_by: dict[int, float] = {}
    origin_by_view: dict[str, int] = {}
    for view, (origin, lo, hi) in WINDOWS.items():
        rows = np.flatnonzero((train["cycles"] >= lo) & (train["cycles"] <= hi)).astype(np.int64)
        view_rows[view] = rows
        origin_by_view[view] = origin
        r06_by.update({int(r): float(p) for r, p in zip(rows, verifier._cache_r06(origin, rows, r06_cache))})
        r22 = r22_cache[origin]
        r22_by.update({int(r): float(p) for r, p in zip(rows, verifier._cache_map(r22, rows))})

    # All six views are exact and disjoint; composites reuse those row sets without remapping.
    view_rows["new_origin_aggregate"] = np.concatenate([view_rows[v] for v in ("W2", "W3", "W5")])
    view_rows["full_primary"] = np.concatenate([view_rows[v] for v in WINDOWS])
    view_rows["late_W5_W6"] = np.concatenate([view_rows["W5"], view_rows["W6"]])
    h12_rows = np.flatnonzero((train["cycles"] >= 68) & (train["cycles"] <= 79)).astype(np.int64)
    view_rows["inherited_H12_L_E67"] = h12_rows
    pilot_arrays = {(67, "observed"): e67_obs, (67, "shuffled"): e67_shuffle,
                    (75, "observed"): e75_obs, (75, "shuffled"): e75_shuffle}
    r06_by_origin = {
        origin: {int(r): float(p) for r, p in zip(cache["rows"], cache["prediction"])}
        for origin, cache in r06_cache.items()
    }
    per_view_obs, per_view_shuffle, h12_obs, h12_shuffle = assemble_prediction_maps(
        view_rows, verified_arrays, pilot_arrays, r06_by_origin, h12_rows)
    obs_by = {r: p for view in WINDOWS for r, p in per_view_obs[view].items()}
    shuffle_by = {r: p for view in WINDOWS for r, p in per_view_shuffle[view].items()}
    if (len(obs_by) != sum(len(per_view_obs[v]) for v in WINDOWS)
            or len(shuffle_by) != sum(len(per_view_shuffle[v]) for v in WINDOWS)):
        raise CorrectedEvaluationError("Primary origin windows overlap or overwrite one another")

    metrics: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for name, rows in view_rows.items():
        if name == "inherited_H12_L_E67":
            h12_r06 = {int(r): float(p) for r, p in zip(rows, verifier._cache_r06(67, rows, r06_cache))}
            h12_r22 = {int(r): float(p) for r, p in zip(rows, verifier._cache_map(r22_cache[67], rows))}
            metrics[name] = verifier._view_metrics(name, rows, y_by, h12_obs, h12_shuffle, h12_r06, h12_r22, missing_by)
            groups[name] = verifier._group_metrics(name, rows, y_by, h12_obs, h12_r06, missing_by)
        else:
            metrics[name] = verifier._view_metrics(name, rows, y_by, obs_by, shuffle_by, r06_by, r22_by, missing_by)
            groups[name] = verifier._group_metrics(name, rows, y_by, obs_by, r06_by, missing_by)

    learned = ["W2", "W3", "W4", "W5", "W6"]
    primary, newagg, late, h12 = (metrics[k] for k in ("full_primary", "new_origin_aggregate", "late_W5_W6", "inherited_H12_L_E67"))
    gates: dict[str, Any] = {
        "full_primary_gain_vs_R06_ge_0_03": primary["gain_vs_R06"] >= .03,
        "full_primary_gain_vs_shuffle_ge_0_01": primary["gain_vs_matched_shuffle"] >= .01,
        "new_origin_gain_vs_R06_nonnegative": newagg["gain_vs_R06"] >= 0,
        "new_origin_gain_vs_shuffle_nonnegative": newagg["gain_vs_matched_shuffle"] >= 0,
        "learned_primary_windows_nonnegative_at_least_3_of_5": sum(metrics[w]["gain_vs_R06"] >= 0 for w in learned) >= 3,
        "late_W5_W6_gain_vs_R06_positive": late["gain_vs_R06"] > 0,
        "inherited_H12_L_harm_vs_R06_le_0_05": -h12["gain_vs_R06"] <= .05,
        "each_learned_window_harm_vs_R06_lt_0_25": all(-metrics[w]["gain_vs_R06"] < .25 for w in learned),
    }
    subgroup_gates = {}
    for view, value in groups.items():
        subgroup_gates[view] = {}
        for group, item in value.items():
            subgroup_gates[view][group] = (None if not item["supported"] else bool(
                item["gate"] and (group != "missing_stream" or item["missing_incremental_exact_zero"])))
    gates["supported_missing_and_target_tail_group_guards"] = subgroup_gates
    gates["all_supported_subgroup_guards_pass"] = all(
        passed is not False for per_view in subgroup_gates.values() for passed in per_view.values())

    hist = json.loads(runner.R06_VERIFY.read_text(encoding="utf-8"))
    hist_result = json.loads(runner.R06_REPORT.read_text(encoding="utf-8"))
    archived_initial = _archive_initial_outputs()
    result = {
        "schema_version": 1,
        "status": "EVALUATED_DIAGNOSTIC_ONLY",
        "evaluation_adapter": "POSTFIT_CORRECTED_ISOLATED_VIEW_MAPS_V1",
        "frozen_verifier_output_status": "SUPERSEDED_SHARED_MAP_OVERWRITE_BUG",
        "frozen_verifier_bug": "E67 H12 predictions were merged into shared W1-W6 maps before W5/W6 and aggregate metrics; corrected maps remain isolated per declared origin.",
        "frozen_verifier_original_output_sha256": archived_initial,
        "freeze_signature": freeze["freeze_signature"],
        "fit_source_commit": source_commit,
        "postfit_adapter_commit": pilot._run_git("rev-parse", "HEAD"),
        "new_fit_verification": verified,
        "ledger_reconciliation": ledger,
        "view_origins": origin_by_view,
        "view_denominators": {k: int(len(v)) for k, v in view_rows.items()},
        "views": metrics,
        "subgroups_vs_R06": groups,
        "prospective_gate_checks": gates,
        "gate_pass": all(v is True for k, v in gates.items() if k != "supported_missing_and_target_tail_group_guards") and gates["all_supported_subgroup_guards_pass"],
        "historical_legacy_risk": {
            "archived_R06_verification_status": hist.get("status"),
            "historical_R06_legacy_guard_summary": hist_result.get("qualification"),
            "historical_R06_W1_missing_stream": hist_result.get("groups", {}).get("views", {}).get("W1", {}).get("signal_stream_missing"),
            "archived_R06_result_path": str(runner.R06_REPORT.relative_to(ROOT)).replace("\\", "/"),
            "interpretation": "Historical R06-vs-legacy failures remain separate; incremental pass does not clear them.",
        },
        "source_boundaries": {
            "target_file": "official Regression Train.csv only",
            "target_rows_evaluated": int(len(eval_rows)),
            "query_targets_read_after_all_six_bundles_reverified": True,
            "test_or_sample_submission_opened": False,
            "kaggle_accessed": False,
            "full_train_refit": False,
            "additional_fits": 0,
            "champion_changed": False,
        },
        "inherited_E67_recovery": bridge_audit,
    }
    _write_reports(result)
    audit = {"status": "PASS", "frozen_verifier_issue": result["frozen_verifier_bug"],
             "superseded_output_sha256": archived_initial,
             "corrected_evaluation_path": "reports/optimization_v3/rnbr_continuation_v1/results.json",
             "corrected_gate_pass": result["gate_pass"],
             "fit_source_commit": source_commit,
             "postfit_adapter_commit": result["postfit_adapter_commit"],
             "no_new_fit": True, "test_sample_kaggle_accessed": False}
    (REPORT_DIR / "frozen_evaluator_bug_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bridge_verification = json.loads((REPORT_DIR / "recovery_bridge_verification.json").read_text(encoding="utf-8"))
    bridge_verification.update({"corrected_evaluation_status": result["status"],
                                "corrected_gate_pass": result["gate_pass"],
                                "frozen_verifier_bug_detected": True,
                                "corrected_report_path": "reports/optimization_v3/rnbr_continuation_v1/results.json"})
    (REPORT_DIR / "recovery_bridge_verification.json").write_text(json.dumps(bridge_verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _write_reports(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# R-NBR three-origin continuation — corrected post-fit results", "",
        "**Status:** `EVALUATED_DIAGNOSTIC_ONLY`; this tranche does not qualify a global model.", "",
        f"**Freeze:** `{result['freeze_signature']}`  ",
        f"**Fit source commit:** `{result['fit_source_commit']}`  ",
        f"**Post-fit adapter commit:** `{result['postfit_adapter_commit']}`", "",
        "> The initial frozen-verifier output is preserved in `frozen_verifier_output_pre_correction_*` and superseded. It overlaid E67 predictions onto shared W5/W6 maps before computing the primary views. The corrected results below use isolated per-view maps and the frozen origins/gates.", "",
        "## Main views", "",
        "| View | n | R06 MAE | R-NBR observed | Shuffled | R22_ORIGINAL | Gain vs R06 | Gain vs shuffle | Gain vs R22 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in result["views"].items():
        lines.append(f"| {name} | {view['rows']} | {view['mae']['R06']:.6f} | {view['mae']['R_NBR_observed']:.6f} | {view['mae']['R_NBR_shuffled']:.6f} | {view['mae']['R22_ORIGINAL']:.6f} | {view['gain_vs_R06']:+.6f} | {view['gain_vs_matched_shuffle']:+.6f} | {view['gain_vs_R22_ORIGINAL']:+.6f} |")
    lines.extend(["", "## Gate summary", ""])
    for key, value in result["prospective_gate_checks"].items():
        if key == "supported_missing_and_target_tail_group_guards":
            continue
        lines.append(f"- `{key}`: **{value}**")
    lines.extend(["", "## Supported subgroup harm vs R06", "", "Harm is candidate MAE minus origin-matched R06 MAE; positive values are worse. Minimum support is the frozen n≥100 rule.", "", "| View | Missing stream n / harm | Target ≤5 n / harm | Target ≥95 n / harm |", "|---|---:|---:|---:|"])
    for view, groups in result["subgroups_vs_R06"].items():
        cells = []
        for key in ("missing_stream", "target_le_5", "target_ge_95"):
            item = groups[key]
            cells.append(f"{item['n']} / {item['harm_vs_R06']:+.4f}" if item["supported"] else f"{item['n']} / unsupported")
        lines.append(f"| {view} | " + " | ".join(cells) + " |")
    legacy = result["historical_legacy_risk"]["historical_R06_W1_missing_stream"]
    lines.extend(["", f"Archived R06-vs-legacy W1 missing-stream guard remains failed: harm **+{legacy['harm']:.6f} MAE** on n={legacy['n']}; continuation does not clear this historical risk.", "",
                  "## Interpretation", "",
                  "Results are diagnostic and origin comparisons are descriptive. The fit identities and recipes are unchanged; corrected scoring changes only map assembly so each declared window uses its frozen origin. No full-Train refit, Test access, Kaggle access, or additional fit occurred.", "",
                  "## Reproduction", "",
                  "The six official bundles remain under the ignored continuation run directory. The original frozen-verifier files and outputs are preserved; this report uses `scripts/optimization_v3_rnbr_continuation_corrected_eval.py`."])
    (REPORT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    try:
        result = evaluate()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "gate_pass": result["gate_pass"],
                      "views": result["views"], "prospective_gate_checks": result["prospective_gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
