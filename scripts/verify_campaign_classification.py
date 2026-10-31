"""Independently verify the bounded dual-campaign classification outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.audit import sha256 as file_sha256, write_json
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest, digest_json
from scripts.research_campaign_classification import (
    CANDIDATES,
    EARLY_ENDS,
    REPORT_DIR,
    RUN_DIR,
    _expected_rows,
    load_c1b,
    load_c2b,
    load_train,
    verify_amendment,
)


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), atol=1e-12, rtol=0):
        raise AssertionError(f"{label}: {actual:.16g} != {expected:.16g}")


def independent_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if actual.ndim != 1 or p.shape != actual.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Invalid labels or positive-class scores")
    prediction = (p >= 0.50).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, prediction, labels=[0, 1], zero_division=0
    )
    return {
        "n": int(len(actual)),
        "macro_f1": float(f1_score(actual, prediction, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, prediction, labels=[0, 1]).tolist(),
        "per_class": [
            {"label": label, "support": int(support[index]), "precision": float(precision[index]),
             "recall": float(recall[index]), "f1": float(f1[index])}
            for index, label in enumerate([0, 1])
        ],
        "positive_rate": float(actual.mean()) if len(actual) else None,
        "positive_prediction_rate": float(prediction.mean()) if len(prediction) else None,
        "false_positive": int(((actual == 0) & (prediction == 1)).sum()),
        "false_negative": int(((actual == 1) & (prediction == 0)).sum()),
    }


def _assert_metric_record(record: dict[str, Any], y: np.ndarray, probability: np.ndarray, label: str) -> None:
    expected = independent_metrics(y, probability)
    for field in ("n", "confusion_matrix", "false_positive", "false_negative"):
        if record[field] != expected[field]:
            raise AssertionError(f"{label}.{field} differs: {record[field]} != {expected[field]}")
    for field in ("macro_f1", "positive_rate", "positive_prediction_rate"):
        _close(record[field], expected[field], f"{label}.{field}")
    if len(record["per_class"]) != 2:
        raise AssertionError(f"{label}.per_class missing")
    for index, current in enumerate(record["per_class"]):
        reference = expected["per_class"][index]
        if current["label"] != reference["label"] or current["support"] != reference["support"]:
            raise AssertionError(f"{label}.class{index} identity/support differs")
        for field in ("precision", "recall", "f1"):
            _close(current[field], reference[field], f"{label}.class{index}.{field}")


def verify(data_dir: Path | None = None) -> dict[str, Any]:
    frozen = verify_amendment(data_dir)
    result_path = REPORT_DIR / "stage1_results.json"
    if not result_path.is_file():
        raise FileNotFoundError("Stage 1 result report is missing")
    import json
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if report.get("status") != "STAGE1_COMPLETE_REVIEW_REQUIRED" or report.get("freeze_signature") != frozen["signature"]:
        raise AssertionError("Stage 1 report status/freeze identity differs")
    if report.get("test_read") is not False or report.get("kaggle_submission") is not False:
        raise AssertionError("Campaign report violates its Test/Kaggle boundary")
    frame, y, train_path = load_train(data_dir)
    if file_sha256(train_path) != frozen["identity"]["classification_train_sha256"]:
        raise AssertionError("Official Train does not match its frozen hash")
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    checks: list[str] = []
    expected_score_slices = {
        55: {"primary_W1_or_W4": (56, 59), "H12_E_or_H12_L": (56, 67)},
        67: {"primary_W1_or_W4": (68, 71), "H12_E_or_H12_L": (68, 79)},
    }
    for fit_end in EARLY_ENDS:
        fit_key = str(fit_end)
        fit_report = report["fits"][fit_key]
        horizon = 67 if fit_end == 55 else 79
        rows = _expected_rows(frame, fit_end, horizon)
        c1_rows, c1_probability, _ = load_c1b(frame, fit_end)
        c2_rows, c2_probability, _ = load_c2b(frame, fit_end)
        if not np.array_equal(rows, c1_rows) or not np.array_equal(rows, c2_rows):
            raise AssertionError(f"fit{fit_end}: control output row IDs differ")
        missing = stream_missing_mask(frame.iloc[rows])
        c4_probability = np.where(missing, c1_probability, c2_probability)
        if array_digest(c4_probability) != fit_report["C4_HGB_probability_sha256"]:
            raise AssertionError(f"fit{fit_end}: C4_HGB baseline vector hash differs")
        fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
        features = fit_report["fit_features"]
        fit_features = frame.iloc[fit_rows][features].copy()
        y_fit = y[fit_rows]
        expected_fit_fingerprint = _fit_fingerprint(fit_features, tuple(features), y_fit)
        if int(fit_report["fit_rows"]) != len(fit_rows):
            raise AssertionError(f"fit{fit_end}: fit row count mismatch")
        for recipe in CANDIDATES:
            tag = f"fit{fit_end}__{recipe}"
            meta_path = RUN_DIR / f"{tag}.json"
            cache_path = RUN_DIR / f"{tag}.npz"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("cache_sha256") != file_sha256(cache_path):
                raise AssertionError(f"{tag}: cache hash mismatch")
            state = meta["fit_state"]
            if state["fit_data_sha256"] != expected_fit_fingerprint or state["fit_data_sha256"] != meta["fit_data_sha256"]:
                raise AssertionError(f"{tag}: learned state fit data fingerprint mismatch")
            if state["fit_rows"] != len(fit_rows) or state["input_feature_names"] != features:
                raise AssertionError(f"{tag}: model feature/fit-row audit mismatch")
            if any(name in state["input_feature_names"] for name in ("record_id", "artifact_signal", "threat_rank")):
                raise AssertionError(f"{tag}: ID/target or excluded threat_rank appeared in model inputs")
            if meta.get("outer_labels_used_for_fit_or_selection") is not False:
                raise AssertionError(f"{tag}: outer labels were not explicitly barred from fit/selection")
            if state.get("campaign", {}).get("missing_stream_indicator_retained") is not True:
                raise AssertionError(f"{tag}: missing-stream indicator was not retained")
            with np.load(cache_path, allow_pickle=False) as archive:
                cached_rows = archive["outer_rows"].astype(np.int64, copy=True)
                candidate = archive["candidate_probability"].astype(float, copy=True)
                routed = archive["routed_probability"].astype(float, copy=True)
                stored_c4 = archive["c4_probability"].astype(float, copy=True)
                stored_missing = archive["missing_stream_mask"].astype(bool, copy=True)
                margin = archive["decision_margin"].astype(float, copy=True) if "decision_margin" in archive else None
            if not np.array_equal(cached_rows, rows) or not np.array_equal(stored_missing, missing):
                raise AssertionError(f"{tag}: cached outer rows or availability mask mismatch")
            if not np.array_equal(stored_c4, c4_probability):
                raise AssertionError(f"{tag}: cached C4 baseline differs from audited Round 1 route")
            if not np.isfinite(candidate).all() or ((candidate < 0) | (candidate > 1)).any():
                raise AssertionError(f"{tag}: invalid candidate probabilities")
            if not np.array_equal(routed[missing], c1_probability[missing]):
                raise AssertionError(f"{tag}: missing-stream route changed C1_B predictions")
            expected_route = np.where(missing, c1_probability, candidate)
            if not np.array_equal(routed, expected_route):
                raise AssertionError(f"{tag}: route does not equal candidate-present/C1_B-missing")
            if array_digest(candidate) != meta["candidate_probability_sha256"] or array_digest(routed) != meta["routed_probability_sha256"]:
                raise AssertionError(f"{tag}: cached probability digest mismatch")
            if recipe == "DC_SVC":
                if margin is None or not np.isfinite(margin).all():
                    raise AssertionError(f"{tag}: LinearSVC decision margins missing/invalid")
                if not np.array_equal((margin >= 0).astype(np.int8), (candidate >= 0.5).astype(np.int8)):
                    raise AssertionError(f"{tag}: fixed margin-zero decisions differ from the uncalibrated probability view")
                if "uncalibrated" not in meta["prediction_diagnostics"].get("decision_rule", ""):
                    raise AssertionError(f"{tag}: margin probability view is not labeled uncalibrated")
            elif recipe == "DC_NBSVM":
                if state.get("nb_svm", {}).get("weighted_word_block") is not True:
                    raise AssertionError(f"{tag}: NB-SVM fit-only word weighting absent from audit")
            elif recipe == "DC_WORD_CHAR":
                if not state.get("char_text", {}).get("parameters"):
                    raise AssertionError(f"{tag}: character vectorizer parameters absent from audit")
            for name, (start, end) in expected_score_slices[fit_end].items():
                selected = (cycle[rows] >= start) & (cycle[rows] <= end)
                selected_rows = rows[selected]
                _assert_metric_record(
                    fit_report["candidates"][recipe]["scores"][name]["candidate"],
                    y[selected_rows], routed[selected], f"{tag}.{name}.candidate",
                )
                _assert_metric_record(
                    fit_report["candidates"][recipe]["scores"][name]["C4_HGB"],
                    y[selected_rows], c4_probability[selected], f"{tag}.{name}.C4_HGB",
                )
            checks.append(tag)
        for name, (start, end) in expected_score_slices[fit_end].items():
            selected = (cycle[rows] >= start) & (cycle[rows] <= end)
            _assert_metric_record(fit_report["control_metrics"][name], y[rows[selected]],
                                  c4_probability[selected], f"fit{fit_end}.{name}.C4_HGB")
    chart = REPORT_DIR / "stage1_h12_macro_f1.png"
    if report.get("chart_sha256") != file_sha256(chart):
        raise AssertionError("Stage 1 chart checksum mismatch")
    verification = {
        "status": "PASS",
        "freeze_signature": frozen["signature"],
        "independent_verifier": "sklearn confusion_matrix, fixed-label precision_recall_fscore_support, and f1_score",
        "verified_candidate_fold_artifacts": checks,
        "verified_scoring_slices_per_candidate": 4,
        "train_sha256": file_sha256(train_path),
        "official_test_read": False,
        "kaggle_submission": False,
    }
    write_json(REPORT_DIR / "stage1_verification.json", verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    result = verify(args.data_dir)
    print(result["status"], len(result["verified_candidate_fold_artifacts"]), "fold artifacts")


if __name__ == "__main__":
    main()
