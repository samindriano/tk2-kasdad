"""Evaluate saved V3 recovery bundles with the original rank50 clipping order.

The v1 verifier omitted the component-level rank50 clip when independently
rebuilding regression predictions. This evaluator keeps the frozen recipe:
clip rank50(raw) to [0,100], blend 0.75 champion + 0.25 mapped component, then
clip the final prediction. It performs no estimator fits and never opens Test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import optimization_v3_expanded_benchmark as base
from scripts import optimization_v3_expanded_recovery as recovery
ROOT = base.ROOT
REPORT_DIR = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation"
EVALUATED = ("C17", "C11", "R21", "R06")


def _verify_regression_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    views: Mapping[str, Mapping[str, np.ndarray]],
    records: Mapping[int, Mapping[str, Any]],
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts import optimization_v3_final_continuation_r22 as reg_report

    y = np.asarray(target, dtype=np.float64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    missing_all = reg_report.stream_missing_mask(frame["signal_stream"].tolist())
    verified: dict[str, Any] = {
        "views": {}, "rank_mapping": {}, "group_failures": [],
        "rank50_component_clip_before_blend": True,
    }
    for cutoff, record in records.items():
        raw = np.asarray(record["raw_candidate"], dtype=np.float64)
        rows = np.asarray(record["score_rows"], dtype=np.int64)
        score_cycles = cycles[rows]
        mapped = np.empty(raw.shape, dtype=np.float64)
        for cycle in np.unique(score_cycles):
            mask = score_cycles == cycle
            mapped[mask] = 0.5 * raw[mask] + 0.5 * (
                100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
            )
        mapped = np.clip(mapped, 0.0, 100.0)
        rebuilt = np.clip(
            0.75 * np.asarray(record["baseline_prediction"], dtype=np.float64)
            + 0.25 * mapped,
            0.0,
            100.0,
        )
        if not np.allclose(rebuilt, np.asarray(record["prediction"], dtype=np.float64),
                           atol=1e-10, rtol=0):
            raise base.ExpandedCampaignError(
                f"Independent clipped rank50 recipe reconstruction failed for cutoff {cutoff}"
            )
        verified["rank_mapping"][str(cutoff)] = {
            "status": "PASS", "full_cycle_rows": int(len(rows)),
            "average_ties": True, "mapped_once_before_blend": True,
            "component_clip_before_blend": [0.0, 100.0],
        }

    metric_rows: dict[str, Any] = {}
    group_failures: list[tuple[str, str]] = []
    for view_name in reg_report.REQUIRED_VIEWS:
        view = views[view_name]
        rows = np.asarray(view["rows"], dtype=np.int64)
        baseline = np.asarray(view["baseline"], dtype=np.float64)
        candidate = np.asarray(view["candidate"], dtype=np.float64)
        if (len(np.unique(rows)) != len(rows)
                or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()
                or np.any((candidate < 0) | (candidate > 100))):
            raise base.ExpandedCampaignError(f"Independent regression view integrity failed: {view_name}")
        actual = y[rows]
        candidate_mae = float(np.mean(np.abs(actual - candidate)))
        champion_mae = float(np.mean(np.abs(actual - baseline)))
        saved = declared["views"][view_name]
        gain = champion_mae - candidate_mae
        if (int(saved["n"]) != len(rows)
                or not np.isclose(candidate_mae, float(saved["candidate_mae"]), atol=1e-12, rtol=0)
                or not np.isclose(champion_mae, float(saved["champion_mae"]), atol=1e-12, rtol=0)
                or not np.isclose(gain, float(saved["gain_vs_champion"]), atol=1e-12, rtol=0)):
            raise base.ExpandedCampaignError(f"Independent regression metric mismatch: {view_name}")
        metric_rows[view_name] = {
            "n": int(len(rows)), "candidate_mae": candidate_mae,
            "champion_mae": champion_mae, "gain_vs_champion": gain,
        }
        masks = {
            "signal_stream_missing": missing_all[rows],
            "threat_rank_le_5": actual <= 5.0,
            "threat_rank_ge_95": actual >= 95.0,
        }
        for group_name, mask in masks.items():
            n = int(mask.sum())
            support = n >= reg_report.MIN_GROUP_SUPPORT
            if n:
                group_candidate = float(np.mean(np.abs(actual[mask] - candidate[mask])))
                group_baseline = float(np.mean(np.abs(actual[mask] - baseline[mask])))
                harm = group_candidate - group_baseline
            else:
                group_candidate = group_baseline = harm = None
            saved_group = declared["groups"]["views"][view_name][group_name]
            if (n != int(saved_group["n"]) or support != bool(saved_group["supported"])
                    or (harm is not None and not np.isclose(harm, float(saved_group["harm"]),
                                                             atol=1e-12, rtol=0))):
                raise base.ExpandedCampaignError(
                    f"Independent subgroup guard mismatch: {view_name}/{group_name}"
                )
            if support and harm > reg_report.MAX_GROUP_HARM:
                group_failures.append((view_name, group_name))
    expected_failures = sorted((item["view"], item["group"])
                               for item in declared["qualification"]["group_failures"])
    if sorted(group_failures) != expected_failures:
        raise base.ExpandedCampaignError("Independent regression subgroup failure set differs")

    late_rows = np.concatenate([views["W5"]["rows"], views["W6"]["rows"]]).astype(np.int64)
    late_candidate = np.concatenate([views["W5"]["candidate"], views["W6"]["candidate"]])
    late_baseline = np.concatenate([views["W5"]["baseline"], views["W6"]["baseline"]])
    late_actual = y[late_rows]
    late_candidate_mae = float(np.mean(np.abs(late_actual - late_candidate)))
    late_champion_mae = float(np.mean(np.abs(late_actual - late_baseline)))
    late_gain = late_champion_mae - late_candidate_mae
    if (len(np.unique(late_rows)) != base.EXPECTED_LATE_ROWS
            or not np.isclose(late_candidate_mae, float(declared["late_W5_W6"]["candidate_mae"]),
                              atol=1e-12, rtol=0)
            or not np.isclose(late_champion_mae, float(declared["late_W5_W6"]["champion_mae"]),
                              atol=1e-12, rtol=0)
            or not np.isclose(late_gain, float(declared["late_W5_W6"]["gain_vs_champion"]),
                              atol=1e-12, rtol=0)):
        raise base.ExpandedCampaignError("Independent W5+W6 regression metric mismatch")
    gates = {
        "primary_gain_at_least_0_03": metric_rows["primary"]["gain_vs_champion"] >= 0.03,
        "H12_harm_at_most_0_05": metric_rows["H12"]["gain_vs_champion"] >= -0.05,
        "at_least_four_of_six_windows_nonnegative": sum(
            metric_rows[f"W{i}"]["gain_vs_champion"] >= 0.0 for i in range(1, 7)
        ) >= 4,
        "W5_W6_gain_positive": late_gain > 0.0,
        "all_supported_group_harm_at_most_0_25": not group_failures,
    }
    if gates != declared["qualification"]["gates"]:
        raise base.ExpandedCampaignError("Independent regression gate arithmetic differs")
    verified["views"] = metric_rows
    verified["groups"] = {"support_minimum": reg_report.MIN_GROUP_SUPPORT,
                           "maximum_allowed_harm": reg_report.MAX_GROUP_HARM}
    verified["late_W5_W6"] = {
        "n": int(len(late_rows)), "candidate_mae": late_candidate_mae,
        "champion_mae": late_champion_mae, "gain_vs_champion": late_gain,
    }
    verified["qualification"] = {
        "gates": gates, "qualified": all(gates.values()),
        "supported_group_failures": [list(item) for item in group_failures],
    }
    verified["status"] = "PASS"
    verified["test_data_used"] = False
    return verified


def evaluate() -> dict[str, Any]:
    recovery_identity, cls_context, reg_context, _ = recovery._verify_recovery_freeze()
    run_report = json.loads(recovery.RUN_REPORT_PATH.read_text(encoding="utf-8"))
    if run_report.get("status") != "RECOVERY_FITS_COMPLETE":
        raise base.ExpandedCampaignError("Recovery bundles are incomplete; evaluation remains closed")
    recovery_signature = recovery.json_sha256(recovery_identity)
    _parent_freeze, _parent_preflight, parent_signature = recovery._parent_state()
    artifacts = base._active_card_artifacts()
    candidate_provenance = {
        "C17": ("optimization_v3_expanded_benchmark_v1",
                ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1", parent_signature),
        "C11": ("optimization_v3_expanded_benchmark_v1",
                ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1", parent_signature),
        "R21": (recovery.CAMPAIGN_ID, recovery.RECOVERY_RUN_DIR, recovery_signature),
        "R06": (recovery.CAMPAIGN_ID, recovery.RECOVERY_RUN_DIR, recovery_signature),
    }
    results: dict[str, Any] = {}
    verifications: dict[str, Any] = {}
    original_id, original_run = base.CAMPAIGN_ID, base.RUN_DIR
    try:
        for card, (campaign_id, run_dir, signature) in candidate_provenance.items():
            base.CAMPAIGN_ID, base.RUN_DIR = campaign_id, run_dir
            classification = card in {"C17", "C11"}
            context = cls_context if classification else reg_context
            records = base._records_for_candidate(card, cls_context, reg_context, artifacts, signature)
            receipts = []
            for cutoff, record in records.items():
                metadata = record["metadata"]
                state = metadata.get("state", {})
                receipts.append({
                    "cutoff": int(cutoff),
                    "fit_rows": int(metadata.get("fit_rows", state.get("fit_rows", 0))),
                    "score_rows": int(len(record["score_rows"])),
                    "predictor_fits": int(metadata.get("fit_predictor_count", 0)),
                    "bundle_sha256": record.get("bundle_sha256"),
                    "sidecar_sha256": record.get("sidecar_sha256"),
                    "reused_pilot": cutoff in base.PILOT_CUTOFFS,
                    "test_data_used": False,
                })
            if classification:
                result, verified = base._classification_candidate_report(
                    card, context["frame"], np.asarray(context["target"], dtype=np.int64),
                    records, signature,
                )
                result["late_W5_W6"] = result["qualification"]["late_W5_W6"]
            else:
                from scripts import optimization_v3_final_continuation_r22 as reg_report
                cycles = np.asarray(context["cycles"], dtype=np.int64)
                views = base._regression_views(context["frame"], cycles, records)
                metrics = reg_report._policy_metrics(
                    context["frame"], np.asarray(context["target"], dtype=np.float64),
                    views, "candidate",
                )
                qualification = reg_report._qualification(metrics)
                result = {
                    "card_id": card, "candidate": card, "task": "regression",
                    "status": "CANDIDATE_METRICS_READY_FOR_INDEPENDENT_QA",
                    "freeze_signature": signature, "views": metrics["views"],
                    "late_W5_W6": metrics["late_W5_W6"],
                    "diagnostics": metrics["diagnostics"], "groups": metrics["groups"],
                    "qualification": {**qualification,
                                      "status": "QUALIFIED" if qualification["qualified"]
                                      else "EXPERIMENTAL_GATE_EXCEPTION"},
                    "metric_formula": "mean(abs(threat_rank - prediction)); gain = champion MAE - candidate MAE",
                    "outcome_reuse_warning": "Train temporal-development labels were previously viewed; not an untouched holdout.",
                    "test_data_used": False, "sample_submission_read": False,
                }
                verified = _verify_regression_metrics(
                    context["frame"], np.asarray(context["target"], dtype=np.float64),
                    views, records, result,
                )
            result["fit_origin_receipts"] = receipts
            result["continuation_fit_accounting"] = {
                "new_fit_cutoffs": list(base.NEW_CUTOFFS),
                "reused_pilot_cutoffs": list(base.PILOT_CUTOFFS),
                "new_predictor_fits": 4, "test_data_used": False,
            }
            verified["candidate"] = card
            results[card], verifications[card] = result, verified
            candidate_dir = REPORT_DIR / card
            candidate_dir.mkdir(parents=True, exist_ok=True)
            base._write_or_match(candidate_dir / "results.json", result)
            base._write_or_match(candidate_dir / "verification.json", verified)
    finally:
        base.CAMPAIGN_ID, base.RUN_DIR = original_id, original_run

    r11_blocker = {
        "candidate": "R11", "status": "INTEGRITY_BLOCKED_NOT_EVALUATED",
        "reason": "Four continuation bundle writes failed on object-dtype diagnostic controls. The fitted models were not persisted; recovery requires 36 predictor fits, above the remaining retry headroom of 11.",
        "scores_opened": False, "recipe_or_gate_changed": False,
    }
    base._write_or_match(REPORT_DIR / "R11_BLOCKER.json", r11_blocker)
    campaign = {
        "schema_version": 1,
        "campaign_id": "optimization_v3_expanded_benchmark_v1_final_evaluation",
        "parent_freeze_signature": parent_signature,
        "recovery_freeze_signature": recovery_signature,
        "status": "FOUR_CANDIDATES_EVALUATED_R11_INTEGRITY_BLOCKED",
        "evaluated_candidates": list(candidate_provenance),
        "blocked_candidates": {"R11": r11_blocker},
        "rank50_verifier_correction": {
            "source": "R21 and R06 frozen recipes clip mapped rank50(raw) to [0,100] before the 0.75/0.25 blend.",
            "change": "Only independent audit reconstruction was corrected to apply the existing component clip before blending.",
            "estimators_or_gates_changed": False,
        },
        "candidates": {
            card: {
                "qualification": results[card]["qualification"]["status"],
                "primary": results[card]["views"]["primary"],
                "H12": results[card]["views"]["H12"],
                "late_W5_W6": results[card].get("late_W5_W6", {}),
                "qualification_gates": results[card]["qualification"]["gates"],
                "verification_status": verifications[card].get("status", "PASS"),
                "result_sha256": base.json_sha256(results[card]),
                "verification_sha256": base.json_sha256(verifications[card]),
            }
            for card in candidate_provenance
        },
        "test_data_used": False,
        "public_score_used": False,
    }
    base._write_or_match(REPORT_DIR / "CAMPAIGN_REPORT.json", campaign)
    lines = [
        "# V3 expanded benchmark recovery evaluation", "",
        f"Parent freeze: `{parent_signature}`", f"Recovery freeze: `{recovery_signature}`", "",
        "| Candidate | Gate status | Primary | H12 | W5+W6 | Verification |",
        "|---|---|---:|---:|---:|---|",
    ]
    for card in candidate_provenance:
        row = campaign["candidates"][card]
        primary, h12, late = row["primary"], row["H12"], row["late_W5_W6"]
        if card in {"C17", "C11"}:
            primary_gain = primary["gain_positive_is_better"]
            h12_gain = h12["gain_positive_is_better"]
            late_gain = late["gain_positive_is_better"]
        else:
            primary_gain = primary["gain_vs_champion"]
            h12_gain = h12["gain_vs_champion"]
            late_gain = late["gain_vs_champion"]
        lines.append(f"| {card} | {row['qualification']} | {primary_gain:.6f} | {h12_gain:.6f} | {late_gain:.6f} | {row['verification_status']} |")
    lines.extend(["", "R11 is blocked before metric evaluation; its performance gate is unknown. Performance failures on the other candidates remain eligible for the authorized experimental submission exception. No Test rows, public scores, or new estimator fits were used in this evaluation.", ""])
    summary_path = REPORT_DIR / "CAMPAIGN_SUMMARY.md"
    summary = "\n".join(lines)
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise base.ExpandedCampaignError("Existing final evaluation summary differs; refusing overwrite")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8", newline="\n")
    return campaign


if __name__ == "__main__":
    print(json.dumps(base._json_clean(evaluate()), sort_keys=True, indent=2, allow_nan=False))
