"""Zero-fit score decomposition for POSITIONAL_TEXT_PILOT_V1.

Reads only the frozen pilot matrices, estimators, prediction bundles, official
Classification Train, and the existing append-only budget ledger. This module
does not fit estimators or preprocessors and does not write to the run cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from src.models.optimization_v3_positional_text_pilot_v1 import (
    canonical_sha256, sha256_array, sha256_csr,
)

PILOT = "POSITIONAL_TEXT_PILOT_V1"
RUNTIME = ROOT / "experiments/runs/optimization_v3_research_next_v1" / PILOT
REPORT = ROOT / "reports/optimization_v3/positional_text_pilot_v1"
OUT = ROOT / "reports/optimization_v3/positional_text_diagnostic_v1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_FREEZE = "f310d0340ea4402677ca9e86aa29c43b1536ded902c0f6a6f52cb53bd333c292"
ORIGINS = (67, 75)
VIEWS = {"E67": {"W4": (68, 71), "W5": (72, 75), "W6": (76, 79), "H12_E67": (68, 79)},
         "E75": {"W6": (76, 79)}}
ARMS = ("bag", "position", "shuffled_position")
EPS = 2e-12


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_from_partition(base: Any, position: Any, base_coef: Any,
                            position_coef: Any, intercept: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (base contribution, position contribution, exact partition sum)."""
    base_score = np.asarray(base @ np.asarray(base_coef, dtype=np.float64).reshape(-1)).ravel()
    position_score = np.asarray(position @ np.asarray(position_coef, dtype=np.float64).reshape(-1)).ravel()
    total = float(np.asarray(intercept, dtype=np.float64).reshape(-1)[0]) + base_score + position_score
    return base_score, position_score, total


def route_with_fallback(present: Any, raw_probability: Any,
                        fallback_probability: Any) -> np.ndarray:
    present = np.asarray(present, dtype=bool)
    raw = np.asarray(raw_probability, dtype=np.float64)
    fallback = np.asarray(fallback_probability, dtype=np.float64)
    if raw.shape != present.shape or fallback.shape != present.shape:
        raise ValueError("route inputs must share one row shape")
    routed = np.where(present, raw, fallback)
    if not np.array_equal(routed[~present], fallback[~present]):
        raise AssertionError("missing-stream fallback changed")
    return routed


def fixed_k_mask(scores: Any, k: int, source_rows: Any) -> np.ndarray:
    """Select exactly k scores, descending by score then ascending source row."""
    values = np.asarray(scores, dtype=np.float64)
    rows = np.asarray(source_rows, dtype=np.int64)
    if values.ndim != 1 or rows.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("fixed-K ranking needs finite aligned one-dimensional arrays")
    if k < 0 or k > len(values):
        raise ValueError("k is outside [0, n]")
    chosen = np.zeros(len(values), dtype=bool)
    if k:
        order = np.lexsort((rows, -values))
        chosen[order[:k]] = True
    return chosen


def validate_binary_class_order(classes: Any) -> np.ndarray:
    result = np.asarray(classes)
    if not np.array_equal(result, np.asarray([0, 1])):
        raise ValueError("binary score decomposition requires class order [0, 1]")
    return result


def run_synthetic_checks() -> dict[str, Any]:
    """Fit-free contract checks confined to an explicit system-temp directory."""
    import tempfile

    ledger_before = file_sha(LEDGER)
    with tempfile.TemporaryDirectory(prefix="positional-text-diagnostic-v1-") as temp_name:
        temp_root = Path(temp_name).resolve()
        if not temp_root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise AssertionError("synthetic diagnostic temporary root escaped the system temp directory")
        temp_ledger = temp_root / "budget.jsonl"
        temp_ledger.write_text('{"fixture":"temporary-only"}\n', encoding="utf-8")
        temp_ledger_sha = file_sha(temp_ledger)

        x_base = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
        x_pos = sparse.csr_matrix(np.asarray([[2.0, 0.0], [0.0, -1.0], [0.0, 0.0]]))
        b0 = np.asarray([0.2, -0.3]); p0 = np.asarray([0.5, 0.7]); intercept = np.asarray([-0.4])
        bx, px, z = decision_from_partition(x_base, x_pos, b0, p0, intercept)
        direct = float(intercept[0]) + np.asarray(x_base @ b0).ravel() + np.asarray(x_pos @ p0).ravel()
        if not np.array_equal(z, direct) or not np.allclose(expit(z), expit(direct), atol=0, rtol=0):
            raise AssertionError("synthetic logit partition/sigmoid identity failed")
        empty_position = sparse.csr_matrix((3, 0), dtype=np.float64)
        _, empty_contribution, empty_z = decision_from_partition(x_base, empty_position, b0,
                                                                   np.asarray([], dtype=np.float64), intercept)
        if np.any(empty_contribution) or not np.array_equal(empty_z, float(intercept[0]) + bx):
            raise AssertionError("empty positional block did not reduce to intercept plus base score")
        if not np.array_equal(validate_binary_class_order([0, 1]), np.asarray([0, 1])):
            raise AssertionError("valid class order was not accepted")
        try:
            validate_binary_class_order([1, 0])
        except ValueError:
            pass
        else:
            raise AssertionError("reversed class order was not rejected")
        tied = fixed_k_mask([0.5, 0.5, 0.5], 2, [9, 3, 7])
        if not np.array_equal(tied, [False, True, True]):
            raise AssertionError("fixed-K ties did not resolve by source-row order")
        allzero_k0 = fixed_k_mask([0.0, 0.0, 0.0], 0, [0, 1, 2])
        allone_kn = fixed_k_mask([1.0, 1.0, 1.0], 3, [0, 1, 2])
        if allzero_k0.any() or not allone_kn.all():
            raise AssertionError("fixed-K all-zero/all-one edge case failed")
        routed = route_with_fallback([True, False, False], [0.3, 0.4, 0.5], [0.8, 0.7, 0.6])
        if not np.array_equal(routed, [0.3, 0.7, 0.6]):
            raise AssertionError("exact missing-route fallback check failed")
        if file_sha(temp_ledger) != temp_ledger_sha:
            raise AssertionError("synthetic diagnostics altered their temporary ledger unexpectedly")
    ledger_after = file_sha(LEDGER)
    if ledger_before != ledger_after:
        raise AssertionError("synthetic tests changed the live ledger")
    return {"status": "PASS", "cases": ["logit_partition_sigmoid", "empty_position_block",
             "class_order_valid_and_reversed", "fixed_k_ties", "fixed_k_allzero_allone",
             "exact_missing_fallback", "temporary_only_paths", "live_ledger_unchanged"],
            "temporary_root_prefix": "positional-text-diagnostic-v1-",
            "temp_ledger_relative_path": "<TemporaryDirectory>/budget.jsonl",
            "live_ledger_sha256_before": ledger_before, "live_ledger_sha256_after": ledger_after}


def binary_metrics(y_true: Any, probability: Any) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("metrics require aligned binary labels and probabilities")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("metrics require finite probabilities in [0, 1]")
    pred = p >= 0.5
    truth = y == 1
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    f1_pos = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1_neg = (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"n": int(len(y)), "positive": int(truth.sum()), "negative": int((~truth).sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "macro_f1": float((f1_pos + f1_neg) / 2),
            "positive_f1": float(f1_pos), "precision": float(precision),
            "positive_recall": float(recall), "predicted_positive": int(pred.sum())}


def flip_summary(y_true: Any, baseline_probability: Any,
                 changed_probability: Any) -> dict[str, int]:
    y = np.asarray(y_true, dtype=np.int8) == 1
    before = np.asarray(baseline_probability, dtype=np.float64) >= 0.5
    after = np.asarray(changed_probability, dtype=np.float64) >= 0.5
    changed = before != after
    before_correct = before == y
    after_correct = after == y
    useful = changed & ~before_correct & after_correct
    harmful = changed & before_correct & ~after_correct
    return {"changed_predictions": int(changed.sum()),
            "helpful_flips": int(useful.sum()), "harmful_flips": int(harmful.sum()),
            "net_correct_change": int(useful.sum() - harmful.sum()),
            "true_positive_change": int(np.count_nonzero(after & y) - np.count_nonzero(before & y))}


def _check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    actual = file_sha(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} actual={actual}")


def _feature_source() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    freeze_path = RUNTIME / "FREEZE.json"
    freeze = json_read(freeze_path)
    if freeze.get("freeze_signature") != EXPECTED_FREEZE:
        raise ValueError("pilot freeze signature differs from Sol-reviewed scope")
    if freeze.get("input_sha256") != EXPECTED_TRAIN_SHA or freeze.get("test_or_sample_submission_opened") is not False:
        raise ValueError("frozen input or Test/Sample boundary is inconsistent")
    accounting = json_read(REPORT / "FIT_ACCOUNTING.json")
    result_path = REPORT / "RESULTS.json"
    _check_hash(result_path, accounting["results_sha256"], "committed pilot results")
    if accounting["freeze_signature"] != EXPECTED_FREEZE or accounting["train_sha256"] != EXPECTED_TRAIN_SHA:
        raise ValueError("committed accounting does not match the pilot freeze")
    for relpath, expected in freeze["source_hashes"].items():
        _check_hash(ROOT / relpath, expected, f"frozen source {relpath}")
    ledger_before = file_sha(LEDGER)
    if ledger_before != accounting["live_ledger_sha256"]:
        raise ValueError("live ledger hash differs from saved completed pilot accounting")
    expected_counts = accounting["ledger_accounting"]
    if (expected_counts["global_cap"] != 105 or expected_counts["fit_starts"] != 105
            or expected_counts["fit_saved"] != 102 or expected_counts["fit_failed"] != 3
            or expected_counts["fit_pending"] != 0):
        raise ValueError("saved fit ledger accounting differs from the authorized completed pilot")
    units = {item["fit_id"]: item for item in accounting["fit_units"]}
    if len(units) != 18:
        raise ValueError(f"expected 18 fit accounting receipts; found {len(units)}")
    return accounting, {"units": units, "ledger_before": ledger_before, "freeze": freeze,
                        "pilot_result": json_read(result_path)}


def _load_origin(origin: int, accounting: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    name = f"E{origin}"
    manifest_path = RUNTIME / f"MATRIX_MANIFEST_{name}.json"
    pred_path = RUNTIME / name / f"predictions_{name}.npz"
    verification_path = RUNTIME / f"PREDICTION_VERIFICATION_{name}.json"
    artifact_index = state["units"]
    _check_hash(manifest_path, accounting["matrix_manifests"][name]["sha256"], f"{name} matrix manifest")
    _check_hash(pred_path, accounting["prediction_bundles"][name]["sha256"], f"{name} prediction bundle")
    _check_hash(verification_path, accounting["prediction_bundles"][name]["verification_sha256"], f"{name} prediction verifier")
    manifest = json_read(manifest_path)
    if manifest.get("origin") != origin or manifest.get("freeze_signature") != EXPECTED_FREEZE:
        raise ValueError(f"{name} manifest origin/freeze mismatch")
    verification = json_read(verification_path)
    if verification.get("status") != "PASS_PREDICTION_INTEGRITY":
        raise ValueError(f"{name} saved independent prediction check is not PASS")

    matrices: dict[str, sparse.csr_matrix] = {}
    for matrix_name in ("score_base", "score_position", "score_shuffled_position"):
        entry = manifest["matrices"][matrix_name]
        path = ROOT / entry["path"]
        _check_hash(path, entry["sha256"], f"{name}/{matrix_name} matrix")
        matrix = sparse.load_npz(path).tocsr()
        if list(matrix.shape) != entry["shape"] or sha256_csr(matrix) != entry["logical_sha256"]:
            raise ValueError(f"{name}/{matrix_name} logical matrix mismatch")
        matrices[matrix_name] = matrix
    base = matrices["score_base"]
    base_width = base.shape[1]
    for augmented_name in ("score_position", "score_shuffled_position"):
        augmented = matrices[augmented_name]
        if augmented.shape[0] != base.shape[0] or augmented.shape[1] <= base_width:
            raise ValueError(f"{name}/{augmented_name} feature dimensions cannot be partitioned")
        if sha256_csr(augmented[:, :base_width]) != sha256_csr(base):
            raise ValueError(f"{name}/{augmented_name} base prefix differs from saved unordered matrix")

    units: dict[str, dict[str, Any]] = {}
    for component in ("BAG_LR", "POSITION_LR", "SHUFFLED_POSITION_LR", "MISSING_HGB", "C2B_TFIDF"):
        fit_id = f"{PILOT}_{name}_{component}"
        receipt = artifact_index[fit_id]
        unit_path = RUNTIME / "units" / f"{fit_id}.joblib"
        _check_hash(unit_path, receipt["artifact_sha256"], f"{fit_id} fitted state")
        units[component] = joblib.load(unit_path)
    vectorizer = units["C2B_TFIDF"]
    vocabulary_idf_hash = canonical_sha256({
        "vocabulary": sorted((str(k), int(v)) for k, v in vectorizer.vocabulary_.items()),
        "idf_sha256": sha256_array(vectorizer.idf_),
    })
    if vocabulary_idf_hash != manifest["shared_preprocessing_sha256"]["tfidf_vocabulary_idf"]:
        raise ValueError(f"{name} saved TF-IDF vocabulary/IDF differs from matrix manifest")
    with np.load(pred_path, allow_pickle=False) as loaded:
        predictions = {key: loaded[key].copy() for key in loaded.files}
    if (int(predictions["origin"][0]) != origin
            or predictions["matrix_manifest_sha256"][0] != file_sha(manifest_path)):
        raise ValueError(f"{name} prediction bundle is not linked to its saved matrix manifest")
    rows = predictions["score_rows"].astype(np.int64, copy=False)
    ids = predictions["score_record_ids"].astype(str, copy=False)
    cycles = predictions["score_cycles"].astype(np.int64, copy=False)
    present = predictions["present_mask"].astype(bool, copy=False)
    if not (len(rows) == len(ids) == len(cycles) == len(present) == base.shape[0]):
        raise ValueError(f"{name} prediction and matrix row counts differ")
    # The manifest hashes pandas' object-dtype strings; NPZ round-trip stores fixed-width Unicode.
    if sha256_array(ids.astype(object)) != manifest["score_row_metadata"]["record_ids_sha256"]:
        raise ValueError(f"{name} prediction IDs differ from matrix manifest")

    logits: dict[str, Any] = {}
    for arm, matrix_name, probability_name in (
        ("bag", None, "prob_bag"),
        ("position", "score_position", "prob_position"),
        ("shuffled_position", "score_shuffled_position", "prob_shuffled_position"),
    ):
        art = units["BAG_LR" if arm == "bag" else "POSITION_LR" if arm == "position" else "SHUFFLED_POSITION_LR"]
        estimator = art["estimator"]
        classes = validate_binary_class_order(estimator.classes_)
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
        expected_width = base_width if arm == "bag" else matrices[matrix_name].shape[1]
        if coef.shape != (1, expected_width) or intercept.shape != (1,):
            raise ValueError(f"{name} {arm} coefficient shape does not match frozen matrix")
        receipt = artifact_index[f"{PILOT}_{name}_{'BAG_LR' if arm == 'bag' else 'POSITION_LR' if arm == 'position' else 'SHUFFLED_POSITION_LR'}"]
        if receipt["fit_metadata"].get("score_matrix_sha256") != (
                manifest["matrices"]["score_base"]["logical_sha256"] if arm == "bag"
                else manifest["matrices"][matrix_name]["logical_sha256"]):
            raise ValueError(f"{name} {arm} model receipt score-matrix hash mismatch")
        raw_saved = np.asarray(predictions[probability_name], dtype=np.float64)
        artifact_probability = np.asarray(art["probability"], dtype=np.float64)
        if raw_saved.shape != (len(rows),) or not np.array_equal(raw_saved, artifact_probability):
            raise ValueError(f"{name} {arm} saved raw probabilities disagree with fit receipt artifact")
        if arm == "bag":
            base_score = np.asarray(base @ coef[0]).ravel()
            position_score = np.zeros(len(rows), dtype=np.float64)
            logit = float(intercept[0]) + base_score
            drop_logit = logit.copy()
        else:
            augmented = matrices[matrix_name]
            base_score, position_score, logit = decision_from_partition(
                base, augmented[:, base_width:], coef[0, :base_width], coef[0, base_width:], intercept)
            drop_logit = float(intercept[0]) + base_score
            if not np.allclose(logit, float(intercept[0]) + np.asarray(augmented @ coef[0]).ravel(),
                               atol=EPS, rtol=EPS):
                raise AssertionError(f"{name} {arm} partition does not reconstruct full augmented logit")
        reconstructed = expit(logit)
        max_abs_error = float(np.max(np.abs(reconstructed - raw_saved)))
        if not np.allclose(reconstructed, raw_saved, atol=EPS, rtol=EPS):
            raise ValueError(f"{name} {arm} sigmoid(logit) differs from saved raw LR probability by {max_abs_error}")
        logits[arm] = {"base_score": base_score, "position_score": position_score,
                       "logit": logit, "drop_logit": drop_logit,
                       "drop_probability": expit(drop_logit), "probability": raw_saved,
                       "max_abs_sigmoid_error": max_abs_error,
                       "intercept": float(intercept[0]), "coef": coef[0]}

    fallback = np.asarray(predictions["prob_missing_hgb"], dtype=np.float64)
    route_probs = {
        "bag": route_with_fallback(present, logits["bag"]["probability"], fallback),
        "position_full": route_with_fallback(present, logits["position"]["probability"], fallback),
        "position_drop": route_with_fallback(present, logits["position"]["drop_probability"], fallback),
        "shuffle_full": route_with_fallback(present, logits["shuffled_position"]["probability"], fallback),
        "shuffle_drop": route_with_fallback(present, logits["shuffled_position"]["drop_probability"], fallback),
    }
    for arm, saved_name in (("bag", "prob_route_bag"), ("position_full", "prob_route_position"),
                            ("shuffle_full", "prob_route_shuffled_position")):
        if not np.array_equal(route_probs[arm], predictions[saved_name]):
            raise ValueError(f"{name} {arm} availability route does not exactly match saved route")
    if not np.array_equal(predictions["prob_missing_hgb"], units["MISSING_HGB"]["probability"]):
        raise ValueError(f"{name} HGB fallback probability differs from saved exact fallback")
    if sparse.csr_matrix(matrices["score_position"][:, base_width:][~present]).nnz != 0:
        raise ValueError(f"{name} missing-stream rows unexpectedly have positional features")
    return {"origin": origin, "manifest": manifest, "predictions": predictions,
            "matrices": matrices, "units": units, "logits": logits,
            "route_probs": route_probs, "record_ids": ids, "source_rows": rows,
            "cycles": cycles, "present": present, "fallback": fallback}


def _format(v: float | int | None, digits: int = 6) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)


def _metric_table(data: dict[str, Any], y: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    origin = data["origin"]
    cycles = data["cycles"]
    present = data["present"]
    metrics: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for view, (lo, hi) in VIEWS[f"E{origin}"].items():
        mask = (cycles >= lo) & (cycles <= hi)
        ym = y[mask]
        for model_name, prob_name in (("bag", "bag"), ("position_full", "position_full"),
                                      ("position_drop", "position_drop"), ("shuffle_full", "shuffle_full"),
                                      ("shuffle_drop", "shuffle_drop")):
            metrics.append({"origin": origin, "view": view, "model": model_name,
                            **binary_metrics(ym, data["route_probs"][prob_name][mask])})
        for full_name, drop_name, arm in (("position_full", "position_drop", "position"),
                                          ("shuffle_full", "shuffle_drop", "shuffled_position")):
            changed = mask & present
            flips.append({"origin": origin, "view": view, "arm": arm,
                          **flip_summary(y[changed], data["route_probs"][drop_name][changed],
                                         data["route_probs"][full_name][changed])})
    return metrics, flips


def _logit_summaries(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    origin = data["origin"]
    cycles = data["cycles"]
    present = data["present"]
    bag = data["logits"]["bag"]
    base = data["matrices"]["score_base"]
    bag_coef = bag["coef"]
    for view, (lo, hi) in VIEWS[f"E{origin}"].items():
        mask = (cycles >= lo) & (cycles <= hi) & present
        for arm in ("position", "shuffled_position"):
            item = data["logits"][arm]
            augmented_base_coef = item["coef"][:base.shape[1]]
            base_shift = (item["intercept"] - bag["intercept"]
                          + np.asarray(base @ (augmented_base_coef - bag_coef)).ravel())
            position = item["position_score"]
            total = base_shift + position
            for component, values in (("position_contribution", position),
                                      ("augmented_base_and_intercept_minus_bag", base_shift),
                                      ("total_augmented_logit_minus_bag", total)):
                selected = values[mask]
                result.append({"origin": origin, "view": view, "arm": arm,
                               "component": component, "n": int(mask.sum()),
                               "mean": float(np.mean(selected)),
                               "sd": float(np.std(selected, ddof=0)),
                               "mean_abs": float(np.mean(np.abs(selected))),
                               "median_abs": float(np.median(np.abs(selected))),
                               "p95_abs": float(np.quantile(np.abs(selected), .95)),
                               "positive_fraction": float(np.mean(selected > 0)),
                               "negative_fraction": float(np.mean(selected < 0)),
                               "zero_fraction": float(np.mean(selected == 0))})
    return result


def _text_documents(values: pd.Series, analyzer: Any) -> tuple[list[list[str]], np.ndarray]:
    docs: list[list[str]] = []
    missing = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values.to_numpy(dtype=object)):
        if pd.isna(value):
            document = ""
        else:
            document = str(value).strip()
        missing[i] = not bool(document)
        # Same fitted C2_B analyzer; retain unigrams only, matching position builder.
        docs.append([term for term in analyzer(document) if " " not in term])
    return docs, missing


def _length_oov(data: dict[str, Any], score_frame: pd.DataFrame, y: np.ndarray) -> list[dict[str, Any]]:
    origin = data["origin"]
    vectorizer = data["units"]["C2B_TFIDF"]
    analyzer = vectorizer.build_analyzer()
    docs, missing = _text_documents(score_frame["signal_stream"], analyzer)
    if not np.array_equal(missing, ~data["present"]):
        raise ValueError(f"E{origin} analyzer missingness differs from exact saved route mask")
    unigram_vocab = {term for term in vectorizer.vocabulary_ if " " not in term}
    lengths = np.asarray([len(tokens) for tokens in docs], dtype=np.int64)
    oov_counts = np.asarray([sum(token not in unigram_vocab for token in tokens) for tokens in docs], dtype=np.int64)
    oov_rate = np.divide(oov_counts, lengths, out=np.zeros(len(lengths), dtype=np.float64), where=lengths > 0)
    bin_names = np.full(len(lengths), "", dtype=object)
    bin_names[(lengths >= 0) & (lengths <= 20)] = "00-20"
    bin_names[(lengths >= 21) & (lengths <= 40)] = "21-40"
    bin_names[(lengths >= 41) & (lengths <= 80)] = "41-80"
    bin_names[lengths > 80] = "81+"
    present = data["present"]
    cycles = data["cycles"]
    result = []
    for bin_name in ("00-20", "21-40", "41-80", "81+"):
        mask = present & (bin_names == bin_name) & np.isin(cycles, [76, 77, 78, 79])
        if not mask.any():
            result.append({"origin": origin, "length_bin": bin_name, "n": 0})
            continue
        base = data["matrices"]["score_base"]
        pos_score = data["logits"]["position"]["position_score"]
        base_shift = (data["logits"]["position"]["intercept"] - data["logits"]["bag"]["intercept"]
                      + np.asarray(base @ (data["logits"]["position"]["coef"][:base.shape[1]]
                                           - data["logits"]["bag"]["coef"])).ravel())
        p_drop = data["route_probs"]["position_drop"]
        p_full = data["route_probs"]["position_full"]
        result.append({"origin": origin, "length_bin": bin_name, "n": int(mask.sum()),
                       "mean_length": float(lengths[mask].mean()),
                       "median_length": float(np.median(lengths[mask])),
                       "oov_token_count": int(oov_counts[mask].sum()),
                       "analyzed_token_count": int(lengths[mask].sum()),
                       "oov_occurrence_rate": float(oov_counts[mask].sum() / lengths[mask].sum()) if lengths[mask].sum() else None,
                       "mean_document_oov_rate": float(oov_rate[mask].mean()),
                       "unigram_vocabulary_size": int(len(unigram_vocab)),
                       "mean_abs_position_logit": float(np.abs(pos_score[mask]).mean()),
                       "mean_abs_base_shift": float(np.abs(base_shift[mask]).mean()),
                       "bag": binary_metrics(y[mask], data["route_probs"]["bag"][mask]),
                       "drop_position": binary_metrics(y[mask], p_drop[mask]),
                       "full_position": binary_metrics(y[mask], p_full[mask])})
    return result


def _matched_k(data: dict[str, Any], y: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for cycle in sorted(np.unique(data["cycles"])):
        subset = (data["cycles"] == cycle) & data["present"]
        if not subset.any():
            result.append({"origin": data["origin"], "cycle": int(cycle), "n": 0, "k": 0})
            continue
        source_rows = data["source_rows"][subset]
        p_bag = data["route_probs"]["bag"][subset]
        p_pos = data["route_probs"]["position_full"][subset]
        truth = y[subset]
        k = int(np.count_nonzero(p_bag >= 0.5))
        bag_select = fixed_k_mask(p_bag, k, source_rows)
        pos_select = fixed_k_mask(p_pos, k, source_rows)
        if (int(bag_select.sum()) != k or int(pos_select.sum()) != k
                or not np.array_equal(bag_select, p_bag >= 0.5)):
            raise AssertionError("fixed-K selector changed bag threshold set or positive count")
        bag_prob = bag_select.astype(np.float64)
        pos_prob = pos_select.astype(np.float64)
        # Metrics consume scores at the unchanged threshold; binary probabilities here encode top-K labels.
        m_bag = binary_metrics(truth, bag_prob)
        m_pos = binary_metrics(truth, pos_prob)
        result.append({"origin": data["origin"], "cycle": int(cycle), "n": int(subset.sum()), "k": k,
                       "bag": m_bag, "position_matched_k": m_pos,
                       "delta_macro_f1": m_pos["macro_f1"] - m_bag["macro_f1"],
                       "delta_tp": m_pos["tp"] - m_bag["tp"],
                       "delta_fp": m_pos["fp"] - m_bag["fp"],
                       "delta_fn": m_pos["fn"] - m_bag["fn"]})
    return result


def _shared_w6(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ai = {str(rid): i for i, rid in enumerate(a["record_ids"])
          if int(a["cycles"][i]) >= 76 and int(a["cycles"][i]) <= 79}
    bi = {str(rid): i for i, rid in enumerate(b["record_ids"])
          if int(b["cycles"][i]) >= 76 and int(b["cycles"][i]) <= 79}
    common = sorted(set(ai) & set(bi))
    if len(common) != 8491 or set(ai) != set(bi):
        raise ValueError(f"shared W6 IDs are not an exact full overlap: E67={len(ai)} E75={len(bi)} common={len(common)}")
    ia = np.asarray([ai[rid] for rid in common], dtype=np.int64)
    ib = np.asarray([bi[rid] for rid in common], dtype=np.int64)
    if not np.array_equal(a["present"][ia], b["present"][ib]):
        raise ValueError("shared W6 stream availability differs across origins")
    def summary(x: np.ndarray) -> dict[str, float]:
        return {"mean": float(np.mean(x)), "mean_abs": float(np.mean(np.abs(x))),
                "median_abs": float(np.median(np.abs(x))), "p95_abs": float(np.quantile(np.abs(x), .95))}
    present = a["present"][ia]
    components: dict[str, Any] = {}
    for component in ("position", "base_shift_vs_bag", "total_logit_shift_vs_bag"):
        values_by_origin = []
        for data, indices in ((a, ia), (b, ib)):
            pos = data["logits"]["position"]["position_score"]
            base = data["matrices"]["score_base"]
            augmented = data["logits"]["position"]
            bag = data["logits"]["bag"]
            shift = (augmented["intercept"] - bag["intercept"]
                     + np.asarray(base @ (augmented["coef"][:base.shape[1]] - bag["coef"])).ravel())
            values = (pos if component == "position" else shift if component == "base_shift_vs_bag"
                      else pos + shift)[indices]
            values_by_origin.append(values[present])
        x, y = values_by_origin
        delta = y - x
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
        components[component] = {"e67": summary(x), "e75": summary(y),
                                 "paired_e75_minus_e67": summary(delta),
                                 "paired_pearson_correlation": corr,
                                 "paired_sign_agreement": float(np.mean(np.sign(x) == np.sign(y)))}
    return {"rows_all": len(common), "rows_present": int(present.sum()), "rows_missing": int((~present).sum()),
            "components": components, "origins_share_same_w6_labels": True,
            "warning": "same score rows; origin comparison is paired/history-sensitive and not independent replication"}


def make_plots(result: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = result["logit_contribution_summaries"]
    labels = [f"E{x['origin']} {x['view']}" for x in summaries
              if x["component"] == "position_contribution" and x["arm"] == "position"]
    unique = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    for ax, arm in zip(axes, ("position", "shuffled_position")):
        series = []
        for label in unique:
            origin_v, view = label.split(" ")
            origin = int(origin_v[1:])
            by_comp = {x["component"]: x for x in summaries
                       if x["origin"] == origin and x["view"] == view and x["arm"] == arm}
            series.append([by_comp[k]["mean_abs"] for k in (
                "position_contribution", "augmented_base_and_intercept_minus_bag", "total_augmented_logit_minus_bag")])
        x = np.arange(len(unique)); width = 0.24
        for j, name in enumerate(("|position|", "|base + intercept shift vs bag|", "|total logit change vs bag|")):
            ax.bar(x + (j - 1) * width, [v[j] for v in series], width, label=name)
        ax.set_title("Real position" if arm == "position" else "Shuffled-position control")
        ax.set_ylabel("Mean absolute logit contribution")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, ncols=3, fontsize=8)
    axes[-1].set_xticks(np.arange(len(unique)), unique, rotation=20, ha="right")
    fig.suptitle("Augmented LR score decomposition on present-stream rows")
    fig.savefig(OUT / "logit_component_summary.png", dpi=160)
    plt.close(fig)

    bins = ["00-20", "21-40", "41-80", "81+"]
    fig, (ax_oov, ax_contrib) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for origin in ORIGINS:
        rows = [x for x in result["w6_length_oov"] if x.get("origin") == origin]
        rows_by_bin = {x["length_bin"]: x for x in rows}
        y_oov = [100 * rows_by_bin[b].get("oov_occurrence_rate", 0.0) for b in bins]
        y_pos = [rows_by_bin[b].get("mean_abs_position_logit", 0.0) for b in bins]
        counts = [rows_by_bin[b].get("n", 0) for b in bins]
        ax_oov.plot(bins, y_oov, marker="o", label=f"E{origin}")
        ax_contrib.plot(bins, y_pos, marker="o", label=f"E{origin}")
        for x, n in zip(bins, counts):
            if origin == 67:
                ax_oov.annotate(f"n={n}", (x, y_oov[bins.index(x)]), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=7)
    ax_oov.set_title("Unigram OOV occurrence rate")
    ax_oov.set_ylabel("OOV occurrences / analyzed tokens (%)")
    ax_contrib.set_title("Real-position contribution")
    ax_contrib.set_ylabel("Mean |position logit|")
    for ax in (ax_oov, ax_contrib):
        ax.set_xlabel("Fixed token-length bin")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Shared W6 present-stream rows by fixed length bin; descriptive only")
    fig.savefig(OUT / "w6_length_oov_summary.png", dpi=160)
    plt.close(fig)


def _report_markdown(result: dict[str, Any]) -> str:
    lines = ["# Positional text score decomposition — ZERO FITS", "",
             "## Decision scope", "",
             "This diagnostic decomposes saved pilot Logistic Regression scores. It is adaptive/descriptive and uses overlapping validation histories; it is not causal attribution, a fair refit ablation, independent replication, or a Test policy. The committed three-third recipe remains retired under its original gates.", "",
             "## Integrity and source boundaries", "",
             f"- Diagnostic source commit: `{result['provenance']['diagnostic_source_commit']}`; script SHA-256 `{result['provenance']['diagnostic_script_sha256']}`.",
             f"- Pilot source commit `{result['provenance']['pilot_source_commit']}`; freeze signature `{result['provenance']['freeze_signature']}`; Train SHA-256 `{result['provenance']['train_sha256']}`.",
             f"- Live ledger SHA-256 before and after: `{result['integrity']['ledger_sha256_before']}` / `{result['integrity']['ledger_sha256_after']}`; exact same = `{result['integrity']['ledger_unchanged']}`. Fit cap stayed 105; no fit units were consumed.",
             f"- Matrix/prediction/model hashes and numerical checks are in [RESULTS.json](RESULTS.json). Official Classification Train only; no Test/Sample/Kaggle access.", "",
             "## Exact score reconstruction", "",
             "For each saved binary LR arm, the logit was partitioned as `intercept + X_base @ beta_base + X_position @ beta_position`. The dropped-block diagnostic sets only `X_position @ beta_position` to zero and retains the augmented model's own base coefficients and intercept. It is not a refitted model. The original bag model was separately reconstructed from its own coefficients.", "",
             "| Origin | Arm | Max abs sigmoid error vs saved raw LR probability | Class order |", "|---:|---|---:|---|"]
    for item in result["decomposition_checks"]:
        lines.append(f"| E{item['origin']} | {item['arm']} | {item['max_abs_sigmoid_error']:.3g} | `{item['class_order']}` |")
    lines += ["", "The routed probability for every missing/blank stream row exactly matches the saved shared C1_B HGB fallback across bag, full, drop-block and shuffled routes.", "",
              "## Fixed-view metrics", "",
              "Macro F1 is primary; scores use the original 0.50 threshold and original availability route.", "",
              "| Origin | View | n | Model | Macro F1 | Δ vs bag | Positive recall | TP | FP | FN |", "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    lookup = {(x["origin"], x["view"], x["model"]): x for x in result["metrics"]}
    for origin in ORIGINS:
        for view in VIEWS[f"E{origin}"]:
            bag = lookup[(origin, view, "bag")]
            for model in ("bag", "position_full", "position_drop", "shuffle_full", "shuffle_drop"):
                row = lookup[(origin, view, model)]
                lines.append(f"| E{origin} | {view} | {row['n']} | {model} | {row['macro_f1']:.6f} | {row['macro_f1']-bag['macro_f1']:+.6f} | {row['positive_recall']:.6f} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines += ["", "## Contribution magnitudes and helpful/harmful flips", "",
              "The base-shift term is `(augmented intercept − bag intercept) + X_base @ (augmented base coefficients − bag coefficients)`. All summaries below use present-stream rows. Helpful/harmful flips compare full augmented scores against that same augmented model with its positional contribution zeroed.", "",
              "| Origin | View | Arm | Component | n | Mean | Mean abs | Median abs | P95 abs |", "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for x in result["logit_contribution_summaries"]:
        lines.append(f"| E{x['origin']} | {x['view']} | {x['arm']} | {x['component']} | {x['n']} | {x['mean']:+.5f} | {x['mean_abs']:.5f} | {x['median_abs']:.5f} | {x['p95_abs']:.5f} |")
    lines += ["", "| Origin | View | Arm | Changed labels | Helpful | Harmful | Net correct | Δ TP |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for x in result["flips"]:
        lines.append(f"| E{x['origin']} | {x['view']} | {x['arm']} | {x['changed_predictions']} | {x['helpful_flips']} | {x['harmful_flips']} | {x['net_correct_change']:+d} | {x['true_positive_change']:+d} |")
    lines += ["", "## Shared W6 history comparison and fixed length/OOV summary", "",
              "Both origins score the same 8,491 W6 rows; the 7,965 present-stream rows are paired by official `record_id`. E75−E67 contribution differences compare two fit histories on the same cohort, not independent samples.", "",
              f"- Present rows: {result['shared_w6']['rows_present']}; missing/blank rows: {result['shared_w6']['rows_missing']}. Paired position-contribution Pearson r: {result['shared_w6']['components']['position']['paired_pearson_correlation']:.4f}; sign agreement: {result['shared_w6']['components']['position']['paired_sign_agreement']:.3f}.",
              "", "| W6 component | E67 mean abs | E75 mean abs | Paired Δ mean abs | Paired r | Sign agreement |", "|---|---:|---:|---:|---:|---:|"]
    for component, values in result["shared_w6"]["components"].items():
        lines.append(f"| {component} | {values['e67']['mean_abs']:.5f} | {values['e75']['mean_abs']:.5f} | {values['paired_e75_minus_e67']['mean_abs']:.5f} | {values['paired_pearson_correlation']:.4f} | {values['paired_sign_agreement']:.3f} |")
    for origin in ORIGINS:
        rows = [x for x in result["w6_length_oov"] if x.get("origin") == origin and x.get("n")]
        oov = sum(x["oov_token_count"] for x in rows)
        tokens = sum(x["analyzed_token_count"] for x in rows)
        vocabulary_size = rows[0]["unigram_vocabulary_size"]
        lines.append(f"- E{origin} saved prefix unigram vocabulary size: {vocabulary_size}; W6 present-stream OOV occurrences: {oov}/{tokens} ({100*oov/tokens:.2f}%).")
    lines += [
              "- Fixed bins are 0–20, 21–40, 41–80 and 81+ analyzer unigrams. Rows are present-stream W6 only; OOV is the occurrence share of analyzer unigrams absent from that origin's saved fit-prefix unigram vocabulary. These are post hoc descriptions, not routing rules.", "",
              "| Origin | Length bin | n | Mean length | OOV occurrences / tokens | OOV rate | Mean abs position logit | Bag F1 | Drop F1 | Full F1 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in result["w6_length_oov"]:
        if not x.get("n"):
            continue
        lines.append(f"| E{x['origin']} | {x['length_bin']} | {x['n']} | {x['mean_length']:.1f} | {x['oov_token_count']} / {x['analyzed_token_count']} | {100*x['oov_occurrence_rate']:.2f}% | {x['mean_abs_position_logit']:.5f} | {x['bag']['macro_f1']:.5f} | {x['drop_position']['macro_f1']:.5f} | {x['full_position']['macro_f1']:.5f} |")
    lines += ["", "## Fixed-K matched-positive-count ranking", "",
              "Once per origin/cycle, K is the number of present rows the saved bag predicts positive at 0.50. The same K is selected by score rank for bag and position, with ties broken by original Train source row. The selector did not read labels; labels are used only for the resulting F1/confusion counts.", "",
              "| Origin | Cycle | n present | K | Bag F1 | Position F1 at K | Δ F1 | Δ TP | Δ FP | Δ FN |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in result["matched_k"]:
        if "bag" not in x:
            continue
        lines.append(f"| E{x['origin']} | {x['cycle']} | {x['n']} | {x['k']} | {x['bag']['macro_f1']:.6f} | {x['position_matched_k']['macro_f1']:.6f} | {x['delta_macro_f1']:+.6f} | {x['delta_tp']:+d} | {x['delta_fp']:+d} | {x['delta_fn']:+d} |")
    metrics = {(x["origin"], x["view"], x["model"]): x for x in result["metrics"]}
    e67_drop = metrics[(67, "H12_E67", "position_drop")]["macro_f1"]
    e67_full = metrics[(67, "H12_E67", "position_full")]["macro_f1"]
    e67_bag = metrics[(67, "H12_E67", "bag")]["macro_f1"]
    e75_drop = metrics[(75, "W6", "position_drop")]["macro_f1"]
    e75_full = metrics[(75, "W6", "position_full")]["macro_f1"]
    e75_bag = metrics[(75, "W6", "bag")]["macro_f1"]
    lines += ["", "## Finding and scope of the recommendation", "",
              f"The score-level positional term changes decisions inside each augmented LR: E67 H12 Macro F1 is {e67_drop:.6f} with the term zeroed and {e67_full:.6f} with it restored; E75 W6 is {e75_drop:.6f} versus {e75_full:.6f}. Yet the augmented models are only {e67_full-e67_bag:+.6f} versus bag at E67 H12 and {e75_full-e75_bag:+.6f} at E75 W6. The no-position counterfactual also falls below the separately fitted bag model at both origins. This points to co-adaptation of positional and base coefficients: the positional score moves useful rows, but the augmented fit's base score differs from the bag fit. Because zeroing a block after fitting is not a refit, this describes these saved scores; it does not estimate generalizable incremental value.",
              "", "That evidence supports only one narrow review question: whether a positional correction can improve the saved bag score while bag coefficients remain frozen. The matched-positive-count ranking result varies by cycle and history, so it does not justify a deployment threshold or a stable-gain claim. Fixed length/OOV patterns are post hoc descriptions."]
    lines += ["", "## Figures", "", "![Logit component summaries](logit_component_summary.png)", "", "![W6 OOV and contribution by fixed length bin](w6_length_oov_summary.png)", "",
              "## QA and reproduction", "",
              f"- Synthetic self-tests: `{result['qa']['synthetic_tests_status']}`; cases: {', '.join(result['qa']['synthetic_cases'])}. Temporary ledger path: `{result['qa']['temp_ledger_relative_path']}` under `{result['qa']['temporary_root_prefix']}*` (outside the repository). Live ledger assertion `{result['integrity']['ledger_unchanged']}`.",
              f"- Command: `python scripts/optimization_v3_positional_text_diagnostic_v1.py --run`; tests: `python -m unittest tests.test_optimization_v3_positional_text_diagnostic_v1 -v`.",
              "- No model/preprocessor fitting or transformation, parameter search, Test/Sample/Kaggle access, install, notebook edit, or push was performed.", "",
              "## Judgment", "", result["judgment"], ""]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    accounting, state = _feature_source()
    ledger_before = state["ledger_before"]
    try:
        synthetic = run_synthetic_checks()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        train_hash = file_sha(TRAIN)
        if train_hash != EXPECTED_TRAIN_SHA:
            raise ValueError(f"official Classification Train hash mismatch: {train_hash}")
        loaded = [_load_origin(origin, accounting, state) for origin in ORIGINS]
        # Labels/features are read only after both origins and all prediction/matrix/model hashes pass.
        frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "signal_stream", "artifact_signal"])
        if len(frame) != 50608 or frame["record_id"].astype(str).duplicated().any():
            raise ValueError("official Train shape/record_id uniqueness check failed")
        judgment = (
            "ONE_SPECIFIC_FOLLOWUP_FOR_SOL_REVIEW: if Sol authorizes one more pilot, "
            "test a frozen-bag-offset positional residual recipe at E67 and E75, with "
            "exactly four predictor fits total (real position and the fixed seed-42 "
            "shuffled-position control at each origin) and zero preprocessing fits. "
            "Freeze each saved bag LR intercept and coefficient vector. On the existing "
            "fit-prefix matrices, optimize only positional gamma with the fixed objective "
            "sum_i logaddexp(0, -(2*y_i-1)*(z_bag_i + Xpos_i @ gamma)) + "
            "||gamma||^2/(2*C), C=1, zero initialization, float64 L-BFGS-B, maxiter=2000, "
            "ftol=1e-12, no retry and fail on non-convergence; no learned intercept. "
            "Use existing temporal and Q4 gates unchanged, threshold 0.50, and the exact "
            "saved HGB fallback for missing streams. This is a proposed conditional "
            "incremental-information check, not causal evidence or deployment selection. "
            "It requires a separate explicit fit-cap authorization; no fit was performed here. "
            "If any frozen gate fails, stop further positional fits; a pass advances only to review."
        )
        outcome: dict[str, Any] = {"schema_version": 1, "diagnostic_id": "POSITIONAL_TEXT_DIAGNOSTIC_V1",
                                  "status": "PASS_ZERO_FIT_DECOMPOSITION", "judgment": judgment,
                                  "provenance": {}, "integrity": {}, "decomposition_checks": [],
                                  "metrics": [], "flips": [], "logit_contribution_summaries": [],
                                  "matched_k": [], "w6_length_oov": [], "shared_w6": {}, "qa": {}}
        result_meta = state["pilot_result"]
        outcome["provenance"] = {
            "diagnostic_source_commit": head,
            "diagnostic_script_sha256": file_sha(Path(__file__).resolve()),
            "test_script_sha256": file_sha(ROOT / "tests/test_optimization_v3_positional_text_diagnostic_v1.py"),
            "pilot_source_commit": state["freeze"]["source_commit"],
            "freeze_signature": EXPECTED_FREEZE, "train_sha256": train_hash,
            "recipe_sha256": state["freeze"]["recipe_sha256"],
            "pilot_results_sha256": accounting["results_sha256"],
            "matrix_manifests": accounting["matrix_manifests"],
            "score_matrix_files": {f"E{data['origin']}": {
                key: {"file_sha256": data["manifest"]["matrices"][key]["sha256"],
                      "logical_sha256": data["manifest"]["matrices"][key]["logical_sha256"],
                      "shape": data["manifest"]["matrices"][key]["shape"]}
                for key in ("score_base", "score_position", "score_shuffled_position")}
                for data in loaded},
            "prediction_bundles": accounting["prediction_bundles"],
            "fitted_state_sha256": {item["fit_id"]: item["artifact_sha256"] for item in accounting["fit_units"]
                                    if item["fit_id"].endswith(("BAG_LR", "POSITION_LR", "SHUFFLED_POSITION_LR", "MISSING_HGB", "C2B_TFIDF"))},
            "model_artifact_sha256": {item["fit_id"]:item["artifact_sha256"] for item in accounting["fit_units"]
                                       if item["fit_id"].endswith(("BAG_LR", "POSITION_LR", "SHUFFLED_POSITION_LR", "MISSING_HGB", "C2B_TFIDF"))}}
        outcome["integrity"] = {"ledger_sha256_before": ledger_before,
                                "official_train_rows": len(frame),
                                "caps_before": result_meta["ledger_snapshot"],
                                "no_fit_calls": True, "no_preprocessor_transform_calls": True,
                                "test_or_sample_opened": False}
        for data in loaded:
            origin = data["origin"]
            score_rows = data["source_rows"]
            selected = frame.iloc[score_rows]
            if not np.array_equal(selected["record_id"].astype(str).to_numpy(), data["record_ids"]):
                raise ValueError(f"E{origin} official Train ID order mismatch")
            cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
            if not np.array_equal(cycles, data["cycles"]):
                raise ValueError(f"E{origin} official Train cycle alignment mismatch")
            y = pd.to_numeric(selected["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
            if not np.isin(y, (0, 1)).all():
                raise ValueError(f"E{origin} score labels are not binary")
            metrics, flips = _metric_table(data, y)
            outcome["metrics"].extend(metrics)
            outcome["flips"].extend(flips)
            outcome["logit_contribution_summaries"].extend(_logit_summaries(data))
            outcome["matched_k"].extend(_matched_k(data, y))
            outcome["w6_length_oov"].extend(_length_oov(data, selected, y))
            for arm in ARMS:
                item = data["logits"][arm]
                outcome["decomposition_checks"].append({
                    "origin": origin, "arm": arm, "class_order": [0, 1],
                    "max_abs_sigmoid_error": item["max_abs_sigmoid_error"],
                    "raw_probabilities_match_saved_artifact": True,
                    "base_width": int(data["matrices"]["score_base"].shape[1]),
                    "position_width": int(data["matrices"]["score_position"].shape[1]
                                           - data["matrices"]["score_base"].shape[1]) if arm == "position"
                                      else int(data["matrices"]["score_shuffled_position"].shape[1]
                                               - data["matrices"]["score_base"].shape[1]) if arm == "shuffled_position" else 0,
                    "missing_fallback_exact": True,
                    "missing_rows_zero_position_block": bool(arm == "bag" or
                        sparse.csr_matrix(data["matrices"]["score_position" if arm == "position" else "score_shuffled_position"][:, data["matrices"]["score_base"].shape[1]:][~data["present"]]).nnz == 0)})
        outcome["shared_w6"] = _shared_w6(loaded[0], loaded[1])
        outcome["integrity"]["ledger_sha256_after"] = file_sha(LEDGER)
        outcome["integrity"]["ledger_unchanged"] = outcome["integrity"]["ledger_sha256_before"] == outcome["integrity"]["ledger_sha256_after"]
        if not outcome["integrity"]["ledger_unchanged"]:
            raise RuntimeError("live append-only ledger hash changed during zero-fit diagnostics")
        outcome["qa"] = {"synthetic_tests_status": synthetic["status"],
                         "synthetic_cases": synthetic["cases"],
                         "temporary_root_prefix": synthetic["temporary_root_prefix"],
                         "temp_ledger_relative_path": synthetic["temp_ledger_relative_path"],
                         "temporary_paths_inside_system_temp": True,
                         "synthetic_live_ledger_sha256_before": synthetic["live_ledger_sha256_before"],
                         "synthetic_live_ledger_sha256_after": synthetic["live_ledger_sha256_after"]}
        return outcome
    finally:
        after = file_sha(LEDGER)
        if after != ledger_before:
            raise RuntimeError(f"live budget ledger changed during diagnostic: before={ledger_before} after={after}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True,
                        help="run the frozen zero-fit diagnostic; no fit or refit is available")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_plots(result)
    (OUT / "RESULTS.md").write_text(_report_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "origins": ORIGINS,
                      "ledger_unchanged": result["integrity"]["ledger_unchanged"],
                      "results": str(OUT / "RESULTS.json"), "report": str(OUT / "RESULTS.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
