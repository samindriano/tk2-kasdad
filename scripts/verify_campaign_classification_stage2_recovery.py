"""Independent check of the report-only full-six result reassembly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.audit import sha256 as file_sha256, write_json
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest
from scripts.research_campaign_classification_stage2 import (
    CANDIDATES,
    NEW_FIT_ENDS,
    PRIMARY_WINDOWS,
    RUN_DIR as STAGE2_RUN_DIR,
    STAGE1_RUN_DIR,
    STAGE1_VERIFICATION_PATH,
    _expected_rows,
    _load_round1_control,
    blend_gate,
    complementarity_counts,
    load_train,
    verify_freeze as verify_stage2_freeze,
)
from scripts.assemble_campaign_classification_stage2 import (
    REPORT_DIR,
    RESULT_PATH,
    _read_json,
    verify_recovery_freeze,
)


def independent_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    probability = np.asarray(p, dtype=float)
    if actual.ndim != 1 or probability.shape != actual.shape or not np.isfinite(probability).all():
        raise AssertionError("Metric arrays must be finite, aligned vectors")
    if ((probability < 0) | (probability > 1)).any():
        raise AssertionError("Positive-class predictions must lie in [0,1]")
    pred = probability >= 0.50
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, pred, labels=[0, 1], zero_division=0
    )
    return {
        "n": int(len(actual)),
        "macro_f1": float(f1_score(actual, pred, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, pred, labels=[0, 1]).tolist(),
        "per_class": [
            {"label": label, "support": int(support[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate([0, 1])
        ],
        "positive_rate": float(actual.mean()) if len(actual) else None,
        "positive_prediction_rate": float(pred.mean()) if len(pred) else None,
        "average_precision": float(average_precision_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
        "brier_score": float(np.mean((probability - actual) ** 2)) if len(actual) else None,
        "false_positive": int(((actual == 0) & pred).sum()),
        "false_negative": int(((actual == 1) & ~pred).sum()),
    }


def _close(a: float, b: float, label: str) -> None:
    if not np.isclose(float(a), float(b), atol=1e-12, rtol=0):
        raise AssertionError(f"{label}: {a:.16g} != {b:.16g}")


def _assert_metrics(record: dict[str, Any], y: np.ndarray, p: np.ndarray, label: str) -> None:
    expected = independent_metrics(y, p)
    for name in ("n", "confusion_matrix", "false_positive", "false_negative"):
        if record[name] != expected[name]:
            raise AssertionError(f"{label}.{name} mismatch")
    for name in ("macro_f1", "positive_rate", "positive_prediction_rate", "average_precision", "brier_score"):
        if expected[name] is None:
            if record[name] is not None:
                raise AssertionError(f"{label}.{name} should be null")
        else:
            _close(record[name], expected[name], f"{label}.{name}")
    if len(record["per_class"]) != 2:
        raise AssertionError(f"{label}.per_class missing")
    for i, item in enumerate(record["per_class"]):
        reference = expected["per_class"][i]
        if item["label"] != reference["label"] or item["support"] != reference["support"]:
            raise AssertionError(f"{label}.class{i} support mismatch")
        for name in ("precision", "recall", "f1"):
            _close(item[name], reference[name], f"{label}.class{i}.{name}")


def _load_window(frame: pd.DataFrame, y: np.ndarray, name: str, recipe: str,
                 fit_end: int, start: int, end: int) -> dict[str, np.ndarray]:
    if fit_end in (55, 67):
        path = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
        with np.load(path, allow_pickle=False) as archive:
            all_rows = archive["outer_rows"].astype(np.int64, copy=True)
            all_candidate = archive["candidate_probability"].astype(float, copy=True)
            all_routed = archive["routed_probability"].astype(float, copy=True)
            all_c4 = archive["c4_probability"].astype(float, copy=True)
            all_missing = archive["missing_stream_mask"].astype(bool, copy=True)
        all_cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[all_rows]
        mask = (all_cycle >= start) & (all_cycle <= end)
        rows = all_rows[mask]
        candidate, routed, c4, missing = all_candidate[mask], all_routed[mask], all_c4[mask], all_missing[mask]
    else:
        cache = STAGE2_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
        meta_path = STAGE2_RUN_DIR / f"fit{fit_end}__{recipe}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_sha256") != file_sha256(cache):
            raise AssertionError(f"{name}/{recipe}: new-fit archive checksum mismatch")
        fit_cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
        fit_rows = np.flatnonzero((fit_cycle >= 47) & (fit_cycle <= fit_end))
        features = meta["features"]
        fingerprint = _fit_fingerprint(frame.iloc[fit_rows][features].copy(), tuple(features), y[fit_rows])
        state = meta["fit_state"]
        if (state["fit_data_sha256"] != fingerprint or state["fit_rows"] != len(fit_rows)
                or state["input_feature_names"] != features
                or meta["outer_labels_used_for_fit_or_selection"] is not False):
            raise AssertionError(f"{name}/{recipe}: training-fold provenance mismatch")
        with np.load(cache, allow_pickle=False) as archive:
            rows = archive["outer_rows"].astype(np.int64, copy=True)
            candidate = archive["candidate_probability"].astype(float, copy=True)
            routed = archive["routed_probability"].astype(float, copy=True)
            c4 = archive["c4_probability"].astype(float, copy=True)
            missing = archive["missing_stream_mask"].astype(bool, copy=True)
            margin = archive["decision_margin"].astype(float, copy=True) if "decision_margin" in archive else None
        if not np.array_equal(rows, _expected_rows(frame, fit_end, end)):
            raise AssertionError(f"{name}/{recipe}: validation row membership mismatch")
        c1_rows, c1_p, _ = _load_round1_control(frame, fit_end, end, "C1_B")
        c2_rows, c2_p, _ = _load_round1_control(frame, fit_end, end, "C2_B")
        live_missing = stream_missing_mask(frame.iloc[rows])
        if (not np.array_equal(rows, c1_rows) or not np.array_equal(rows, c2_rows)
                or not np.array_equal(missing, live_missing)):
            raise AssertionError(f"{name}/{recipe}: control or stream availability row mismatch")
        if not np.array_equal(c4, np.where(missing, c1_p, c2_p)):
            raise AssertionError(f"{name}/{recipe}: C4 baseline does not match audited controls")
        if not np.array_equal(routed, np.where(missing, c1_p, candidate)):
            raise AssertionError(f"{name}/{recipe}: candidate route incorrect")
        for arr, field in ((candidate, "candidate_probability"), (routed, "routed_probability"),
                           (c4, "C4_HGB_probability")):
            if array_digest(arr) != meta[f"{field}_sha256"]:
                raise AssertionError(f"{name}/{recipe}: {field} digest mismatch")
        if recipe == "DC_SVC" and (margin is None or not np.array_equal(margin >= 0, candidate >= 0.50)):
            raise AssertionError(f"{name}/{recipe}: fixed margin-zero rule mismatch")
        if recipe == "DC_SVC" and array_digest(margin) != meta["margin_sha256"]:
            raise AssertionError(f"{name}/{recipe}: SVC margin hash mismatch")
    expected_rows = _expected_rows(frame, start - 1, end)
    if not np.array_equal(rows, expected_rows):
        raise AssertionError(f"{name}/{recipe}: primary window rows differ")
    if not np.array_equal(missing, stream_missing_mask(frame.iloc[rows])):
        raise AssertionError(f"{name}/{recipe}: current Train stream mask mismatch")
    return {"rows": rows, "candidate_probability": candidate,
            "routed_probability": routed, "c4_probability": c4,
            "missing_stream_mask": missing}


def verify(data_dir: Path | None = None) -> dict[str, Any]:
    recovery = verify_recovery_freeze(data_dir)
    frozen_stage2 = verify_stage2_freeze(data_dir)
    report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if (report.get("status") != "FULL_SIX_COMPLETE_REVIEW_REQUIRED"
            or report.get("freeze_signature") != frozen_stage2["signature"]
            or report.get("report_recovery_freeze_signature") != recovery["signature"]):
        raise AssertionError("Recovered report and freeze identities differ")
    if report.get("test_read") is not False or report.get("kaggle_submission") is not False:
        raise AssertionError("Recovered report violates Test/Kaggle boundary")
    if report.get("new_fit_count") != 12:
        raise AssertionError("Recovered report does not contain exactly 12 new fits")
    frame, y, train = load_train(data_dir)
    if file_sha256(train) != recovery["identity"]["classification_train_sha256"]:
        raise AssertionError("Official Train hash mismatch")
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)
    window_data = {name: {} for name in PRIMARY_WINDOWS}
    verified_pairs = []
    for name, (fit_end, start, end) in PRIMARY_WINDOWS.items():
        for recipe in CANDIDATES:
            record = _load_window(frame, y, name, recipe, fit_end, start, end)
            rows = record["rows"]
            _assert_metrics(report["window_metrics"][name][recipe]["candidate"], y[rows],
                            record["routed_probability"], f"{name}/{recipe}/candidate")
            _assert_metrics(report["window_metrics"][name][recipe]["C4_HGB"], y[rows],
                            record["c4_probability"], f"{name}/{recipe}/C4_HGB")
            window_data[name][recipe] = record
            verified_pairs.append(f"{name}/{recipe}")
    full_vectors = {}
    expected_all_rows = np.flatnonzero(pd.Series(cycle).between(56, 79).to_numpy())
    for recipe in CANDIDATES:
        pieces = [window_data[name][recipe] for name in PRIMARY_WINDOWS]
        rows = np.concatenate([piece["rows"] for piece in pieces])
        candidate = np.concatenate([piece["candidate_probability"] for piece in pieces])
        routed = np.concatenate([piece["routed_probability"] for piece in pieces])
        c4 = np.concatenate([piece["c4_probability"] for piece in pieces])
        missing = np.concatenate([piece["missing_stream_mask"] for piece in pieces])
        if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), expected_all_rows):
            raise AssertionError(f"{recipe}: full-six validation coverage incomplete/duplicated")
        full_vectors[recipe] = {"rows": rows, "candidate_probability": candidate,
                                "routed_probability": routed, "c4_probability": c4,
                                "missing_stream_mask": missing}
        record = report["full_primary"][recipe]
        _assert_metrics(record["candidate"], y[rows], routed, f"full/{recipe}/candidate")
        _assert_metrics(record["C4_HGB"], y[rows], c4, f"full/{recipe}/C4_HGB")
        for subgroup, mask in (("stream_present", ~missing), ("stream_missing", missing)):
            _assert_metrics(report["subgroups"][recipe][subgroup]["candidate"], y[rows[mask]], routed[mask],
                            f"full/{recipe}/{subgroup}/candidate")
            _assert_metrics(report["subgroups"][recipe][subgroup]["C4_HGB"], y[rows[mask]], c4[mask],
                            f"full/{recipe}/{subgroup}/C4_HGB")
        present = ~missing
        paired = complementarity_counts(y[rows[present]], candidate[present], c4[present])
        for key, value in paired.items():
            if report["complementarity"][recipe][key] != value:
                raise AssertionError(f"full/{recipe}: paired correction count mismatch {key}")
        if blend_gate(paired) != report["candidate_gate"][recipe]["blend_gate_passed"]:
            raise AssertionError(f"full/{recipe}: fixed blend gate outcome mismatch")
        for label, fit_end, start, end in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
            path = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
            with np.load(path, allow_pickle=False) as archive:
                hrows = archive["outer_rows"].astype(np.int64, copy=True)
                hrouted = archive["routed_probability"].astype(float, copy=True)
                hc4 = archive["c4_probability"].astype(float, copy=True)
            selected = (cycle[hrows] >= start) & (cycle[hrows] <= end)
            current = report["h12"][recipe][label]
            _assert_metrics(current["candidate"], y[hrows[selected]], hrouted[selected], f"{recipe}/{label}/candidate")
            _assert_metrics(current["C4_HGB"], y[hrows[selected]], hc4[selected], f"{recipe}/{label}/C4_HGB")
    best = max(CANDIDATES, key=lambda recipe: report["full_primary"][recipe]["candidate"]["macro_f1"])
    blend = report["fixed_50_50_blend"]
    if blend.get("best_candidate") != best:
        raise AssertionError("Fixed blend did not use the best full-six pooled Macro F1 candidate")
    should_blend = report["candidate_gate"][best]["blend_gate_passed"]
    if blend.get("evaluated") is not should_blend:
        raise AssertionError("Blend eligibility differs from fixed gate")
    if should_blend:
        record = full_vectors[best]
        rows = record["rows"]
        present = ~record["missing_stream_mask"]
        probability = np.where(present,
                              0.50 * record["candidate_probability"] + 0.50 * record["c4_probability"],
                              record["c4_probability"])
        _assert_metrics(blend["metrics"], y[rows], probability, "fixed_50_50_blend")
    chart_hashes = report.get("charts", {})
    for name in ("full_six_window_macro_f1.png", "paired_present_errors.png"):
        path = REPORT_DIR / name
        if chart_hashes.get(name) != file_sha256(path):
            raise AssertionError(f"Recovered chart hash mismatch: {name}")
    verification = {
        "status": "PASS",
        "stage2_freeze_signature": frozen_stage2["signature"],
        "recovery_freeze_signature": recovery["signature"],
        "new_model_fits": 0,
        "reused_and_new_window_candidate_pairs_verified": len(verified_pairs),
        "stage2_new_fit_artifacts_verified": 12,
        "metrics_independently_recomputed": ["Macro F1", "per-class precision/recall/F1", "confusion matrix", "average precision", "Brier", "per-cycle Macro F1"],
        "official_test_read": False,
        "kaggle_submission": False,
        "train_sha256": file_sha256(train),
    }
    write_json(REPORT_DIR / "full_six_recovery_verification.json", verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    verification = verify(args.data_dir)
    print(verification["status"], verification["stage2_new_fit_artifacts_verified"], "new fit artifacts")


if __name__ == "__main__":
    main()

