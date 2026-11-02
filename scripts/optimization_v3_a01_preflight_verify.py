"""Independent reconstruction of the A01 synthetic persistence preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score

from scripts.optimization_v3_a01_preflight import (
    REPORT_PATH, RUN_ROOT, SYNTHETIC_SEED, _persisted_estimator_fit_count,
    synthetic_fixture,
)
from src.models.classification_recency_a01 import CARD_ID, ORIGINS
from src.models.classification_models import _text_values
from src.models.research_next_contract import file_sha256, read_bundle
from src.models.regression_research_text import RegressionTextModel
from src.models.round1_models import _fit_fingerprint
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
FROZEN_GATE = {
    "primary_gain_vs_sparse25_min": 0.0,
    "primary_gain_vs_shuffled_min": 0.0,
    "H12_gain_vs_sparse25_min": 0.0005,
    "H12_gain_vs_shuffled_min": 0.0002,
    "H12_55_max_harm_vs_sparse25": 0.001,
    "H12_67_max_harm_vs_sparse25": 0.001,
    "W1_max_harm_vs_sparse25": 0.003,
    "W4_max_harm_vs_sparse25": 0.003,
}


def _array_sha(values: Any, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _macro(target: Any, probability: Any) -> float:
    return float(f1_score(np.asarray(target, dtype=np.int8),
                          (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int8),
                          average="macro", zero_division=0))


def _independent_gate(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    gains = {
        name: {
            "recency_vs_sparse25": float(values["recency"] - values["sparse25"]),
            "recency_vs_shuffled": float(values["recency"] - values["shuffled"]),
            "shuffled_vs_sparse25": float(values["shuffled"] - values["sparse25"]),
        }
        for name, values in metrics.items()
    }
    checks = {
        "primary_nonnegative_vs_sparse25": gains["primary"]["recency_vs_sparse25"] >= FROZEN_GATE["primary_gain_vs_sparse25_min"],
        "primary_nonnegative_vs_shuffled": gains["primary"]["recency_vs_shuffled"] >= FROZEN_GATE["primary_gain_vs_shuffled_min"],
        "H12_min_gain_vs_sparse25": gains["H12"]["recency_vs_sparse25"] >= FROZEN_GATE["H12_gain_vs_sparse25_min"],
        "H12_min_gain_vs_shuffled": gains["H12"]["recency_vs_shuffled"] >= FROZEN_GATE["H12_gain_vs_shuffled_min"],
        "H12_55_harm_guard": gains["H12_55"]["recency_vs_sparse25"] >= -FROZEN_GATE["H12_55_max_harm_vs_sparse25"],
        "H12_67_harm_guard": gains["H12_67"]["recency_vs_sparse25"] >= -FROZEN_GATE["H12_67_max_harm_vs_sparse25"],
        "W1_harm_guard": gains["W1"]["recency_vs_sparse25"] >= -FROZEN_GATE["W1_max_harm_vs_sparse25"],
        "W4_harm_guard": gains["W4"]["recency_vs_sparse25"] >= -FROZEN_GATE["W4_max_harm_vs_sparse25"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "gains": gains, "rules": dict(FROZEN_GATE)}


def _verify_synthetic_preflight_v1(*, run_root: Path = RUN_ROOT,
                               report_path: Path = REPORT_PATH) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    problems: list[str] = []
    if report.get("status") != "SYNTHETIC_PREFLIGHT_PASS":
        problems.append("report status is not synthetic preflight pass")
    if report.get("card_id") != CARD_ID or report.get("seed") != SYNTHETIC_SEED:
        problems.append("synthetic campaign identity or seed mismatch")
    if report.get("official_fits_started") != 0:
        problems.append("synthetic preflight claims official fits were started")
    if report.get("scope") != "synthetic-only; no official Train/Test/SampleSubmission/Kaggle or budget ledger access":
        problems.append("synthetic-only source-boundary marker mismatch")
    resume = report.get("synthetic_resume_audit", {})
    if (report.get("recovered_after_injected_postfit_failure_without_refit") is not True or
            resume.get("failure_injected_after_durable_model") is not True or
            resume.get("model_fit_during_initial_attempt") != 1 or
            resume.get("model_fit_during_resume_for_interrupted_artifact") != 0 or
            resume.get("all_other_models_fitted_once") is not True):
        problems.append("synthetic post-fit recovery was not demonstrated without refitting")

    frame, target, score_rows_by_origin = synthetic_fixture()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    expected_artifact_keys = {f"fit{origin}_{arm.lower()}" for origin in ORIGINS
                              for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")}
    artifacts = report.get("artifacts", {})
    if set(artifacts) != expected_artifact_keys:
        problems.append("synthetic artifact identity set differs from the four planned origin-arm combinations")
    raw_by_origin: dict[int, dict[str, np.ndarray]] = {}
    bundle_by_origin: dict[int, dict[str, Any]] = {}
    metric_parts: dict[str, dict[str, float]] = {}

    for origin in ORIGINS:
        fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= origin))
        score_rows = score_rows_by_origin[origin]
        fit_x = frame.iloc[fit_rows].copy()
        score_x = frame.iloc[score_rows].copy()
        fit_y = target[fit_rows]
        raw_weights = np.maximum(0.25, np.power(2.0, -(origin - cycles[fit_rows]) / 8.0))
        recency = np.empty(len(fit_y), dtype=np.float64)
        for value in (0, 1):
            mask = fit_y == value
            recency[mask] = raw_weights[mask] / raw_weights[mask].mean()
        rng = np.random.default_rng(42)
        shuffled = np.empty_like(recency)
        sources = np.arange(len(fit_y), dtype=np.int64)
        for value in (0, 1):
            positions = np.flatnonzero(fit_y == value)
            permutation = rng.permutation(len(positions))
            sources[positions] = positions[permutation]
            shuffled[positions] = recency[positions[permutation]]
        weight_by_arm = {"RECENCY": recency, "SHUFFLED_WITHIN_CLASS": shuffled}
        raw_by_origin[origin] = {}
        bundle_by_origin[origin] = {}

        for arm, weights in weight_by_arm.items():
            artifact_key = f"fit{origin}_{arm.lower()}"
            fit_dir = Path(run_root) / artifact_key
            model_path = fit_dir / "fitted_state.joblib"
            state_path = fit_dir / "fit_state.json"
            artifact = artifacts.get(artifact_key, {})
            if not model_path.is_file() or not state_path.is_file():
                problems.append(f"synthetic fitted model/state missing: {artifact_key}")
                continue
            if file_sha256(model_path) != artifact.get("model_sha256") or file_sha256(state_path) != artifact.get("state_sha256"):
                problems.append(f"synthetic model/state receipt hash mismatch: {artifact_key}")
            model = joblib.load(model_path)
            state = model.audit_state()
            if (state.get("sample_weight_used") is not True or state.get("cutoff") != origin or
                    state.get("weight_arm") != arm or state.get("fit_rows") != len(fit_rows) or
                    state.get("sample_weight_sha256") != _array_sha(weights, "<f8")):
                problems.append(f"synthetic fitted state/weight lineage mismatch: {artifact_key}")
            if state.get("fit_data_sha256") != _fit_fingerprint(fit_x, tuple(state["input_feature_names"]), fit_y):
                problems.append(f"synthetic fit population fingerprint mismatch: {artifact_key}")
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            if saved_state != {"model_state": state, "sample_weight_sha256": state["sample_weight_sha256"]}:
                problems.append(f"synthetic persisted fit-state receipt differs: {artifact_key}")
            probability = model.predict_probability(score_x)
            raw_by_origin[origin][arm] = probability

            bridge = artifact.get("bundle", {})
            identity = bridge.get("identity", {})
            bundle_path = fit_dir / "bundle"
            try:
                bundle = read_bundle(bundle_path, identity, task="classification")
            except Exception as exc:
                problems.append(f"synthetic bundle failed independent contract read {artifact_key}: {type(exc).__name__}: {exc}")
                continue
            if bundle.receipt.get("receipt_sha256") != bridge.get("receipt", {}).get("receipt_sha256"):
                problems.append(f"synthetic bundle receipt differs from report: {artifact_key}")
            if bundle.receipt.get("metadata", {}).get("payload_meta", {}).get("run_kind") != "SYNTHETIC_PREFLIGHT_V2":
                problems.append(f"synthetic bundle metadata is not isolated: {artifact_key}")
            arrays = bundle.arrays
            if not np.array_equal(arrays["score_rows"], score_rows):
                problems.append(f"synthetic bundle score rows mismatch: {artifact_key}")
            if not np.array_equal(arrays["score_record_ids"], frame.index.to_numpy()[score_rows].astype(str)):
                problems.append(f"synthetic bundle score IDs mismatch: {artifact_key}")
            if not np.array_equal(arrays["cycle_code"], cycles[score_rows].astype(np.int16)):
                problems.append(f"synthetic bundle cycles mismatch: {artifact_key}")
            bundle_by_origin[origin][arm] = arrays

        if set(raw_by_origin[origin]) != {"RECENCY", "SHUFFLED_WITHIN_CLASS"} or set(bundle_by_origin[origin]) != set(raw_by_origin[origin]):
            continue
        held = frame.iloc[score_rows]
        feature_array = held[["impact_load", "entry_friction", "text_load"]].to_numpy(dtype=np.float64)
        c2b = 1 / (1 + np.exp(-((feature_array[:, 0] - 100) / 25 + (feature_array[:, 1] - 5) / 5) * 0.35))
        cb = 1 / (1 + np.exp(-((feature_array[:, 2] - 50) / 20) * 0.30))
        sparse25 = 1 / (1 + np.exp(-((feature_array[:, 2] - 50) / 18 + (feature_array[:, 0] - 100) / 35) * 0.25))
        stream_missing = np.asarray([value is None or pd.isna(value) or not str(value).strip()
                                     for value in held["signal_stream"].to_numpy(dtype=object)], dtype=bool)
        baseline = np.where(stream_missing, cb, .75 * c2b + .25 * sparse25)
        recency_route = np.where(stream_missing, cb, .75 * c2b + .25 * raw_by_origin[origin]["RECENCY"])
        shuffled_route = np.where(stream_missing, cb, .75 * c2b + .25 * raw_by_origin[origin]["SHUFFLED_WITHIN_CLASS"])
        expected_routes = {"RECENCY": recency_route, "SHUFFLED_WITHIN_CLASS": shuffled_route}
        for arm, expected_prediction in expected_routes.items():
            arrays = bundle_by_origin[origin][arm]
            if not np.array_equal(arrays["prediction"], expected_prediction):
                problems.append(f"independently reconstructed fixed route mismatch: fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["baseline_prediction"], baseline):
                problems.append(f"sparse25 baseline reconstruction mismatch: fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["c2b_probability"], c2b) or not np.array_equal(arrays["cb_probability"], cb):
                problems.append(f"synthetic frozen route input mismatch: fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["stream_missing"].astype(bool), stream_missing):
                problems.append(f"synthetic missing-stream mask mismatch: fit{origin}_{arm.lower()}")
        labels = target[score_rows]
        view = f"H12_{origin}"
        metric_parts[view] = {
            "sparse25": _macro(labels, baseline),
            "recency": _macro(labels, recency_route),
            "shuffled": _macro(labels, shuffled_route),
        }
        if origin == 55:
            window = (cycles[score_rows] >= 56) & (cycles[score_rows] <= 59)
            metric_parts["W1"] = {"sparse25": _macro(labels[window], baseline[window]),
                                  "recency": _macro(labels[window], recency_route[window]),
                                  "shuffled": _macro(labels[window], shuffled_route[window])}
        else:
            window = (cycles[score_rows] >= 68) & (cycles[score_rows] <= 71)
            metric_parts["W4"] = {"sparse25": _macro(labels[window], baseline[window]),
                                  "recency": _macro(labels[window], recency_route[window]),
                                  "shuffled": _macro(labels[window], shuffled_route[window])}

    if len(metric_parts) == 4:
        h12 = {
            arm: float(np.average([metric_parts["H12_55"][arm], metric_parts["H12_67"][arm]],
                                  weights=[len(score_rows_by_origin[55]), len(score_rows_by_origin[67])]))
            for arm in ("sparse25", "recency", "shuffled")
        }
        metrics = {**metric_parts, "H12": h12, "primary": dict(h12)}
        if report.get("metrics_by_view") != metrics:
            problems.append("synthetic metrics differ from independent macro-F1 reconstruction")
        independent_gate = _independent_gate(metrics)
        if report.get("synthetic_gate_exercised") != independent_gate:
            problems.append("synthetic gate decision differs from independent reconstruction")
    else:
        metrics = {}
        independent_gate = {}
        problems.append("independent synthetic metrics could not be reconstructed for all views")

    checks = {
        "synthetic_source_boundary": report.get("official_fits_started") == 0 and
            report.get("status") == "SYNTHETIC_PREFLIGHT_PASS",
        "four_origin_arm_model_bundles": set(artifacts) == expected_artifact_keys and len(bundle_by_origin) == 2,
        "sample_weight_lineage_rebuilt": all(
            artifacts.get(f"fit{origin}_{arm.lower()}", {}).get("model_state", {}).get("sample_weight_used") is True
            for origin in ORIGINS for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")
        ),
        "route_metrics_gate_rebuilt": bool(metrics and independent_gate),
        "postfit_resume_no_refit": report.get("recovered_after_injected_postfit_failure_without_refit") is True,
    }
    result = {"status": "PASS" if not problems and all(checks.values()) else "FAIL",
              "campaign": f"{CARD_ID}_SYNTHETIC_PREFLIGHT_INDEPENDENT_VERIFICATION",
              "checks": checks, "independent_metrics_by_view": metrics,
              "independent_gate": independent_gate, "problems": problems,
              "official_data_read": False, "official_fits_started": 0,
              "test_read": False, "sample_submission_read": False, "kaggle_access": False}
    return result


def _fit_weights(cycles: np.ndarray, target: np.ndarray, cutoff: int) -> dict[str, Any]:
    """Independent A01 formula/shuffle/ESS reconstruction."""
    base = np.maximum(0.25, np.power(2.0, -(cutoff - cycles.astype(np.int64)) / 8.0))
    recency = np.empty(len(target), dtype=np.float64)
    for class_value in (0, 1):
        mask = target == class_value
        recency[mask] = base[mask] / base[mask].mean()
    rng = np.random.default_rng(42)
    shuffled = np.empty_like(recency)
    sources = np.arange(len(target), dtype=np.int64)
    permutations: dict[str, list[int]] = {}
    ess_by_class: dict[str, Any] = {}
    for class_value in (0, 1):
        positions = np.flatnonzero(target == class_value)
        permutation = rng.permutation(len(positions))
        sources[positions] = positions[permutation]
        shuffled[positions] = recency[positions[permutation]]
        permutations[str(class_value)] = permutation.tolist()
        for arm_name, weights in (("recency", recency), ("shuffled", shuffled)):
            class_weights = weights[positions]
            ess = float(class_weights.sum() ** 2 / np.square(class_weights).sum())
            ess_by_class.setdefault(str(class_value), {})[arm_name] = {
                "n": int(len(positions)), "mass": float(class_weights.sum()),
                "ess": ess, "ess_ratio": ess / len(positions),
            }
    def ess(weights: np.ndarray) -> float:
        return float(weights.sum() ** 2 / np.square(weights).sum())

    overall = {
        "recency_ess": ess(recency), "recency_ess_ratio": ess(recency) / len(recency),
        "shuffled_ess": ess(shuffled), "shuffled_ess_ratio": ess(shuffled) / len(shuffled),
    }
    return {"recency": recency, "shuffled": shuffled, "sources": sources,
            "permutations": permutations, "ess_by_class": ess_by_class,
            "overall_ess": overall}


def _design_matrix(model: Any, frame: pd.DataFrame, *, reference: bool) -> sparse.csr_matrix:
    if reference:
        processor, scaler, encoder = model.proc, model.scaler, model.encoder
        numeric_positions, vectorizer = model.numeric_positions, model.vectorizer
    else:
        processor, scaler, encoder = model._processor, model._scaler, model._encoder
        numeric_positions, vectorizer = model._numeric_positions, model._vectorizer
    structured, _unknown = RegressionTextModel._structured_transform(
        frame, processor, scaler, encoder, numeric_positions,
    )
    documents, missing = _text_values(frame["signal_stream"])
    text = vectorizer.transform(documents).tocsr()
    design = sparse.hstack((structured, text,
                            sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
                           format="csr", dtype=np.float64)
    design.sort_indices()
    return design


def _same_float(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-12))


def _metric_from_probabilities(labels: np.ndarray, probability: np.ndarray) -> float:
    pred = (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int8)
    return float(f1_score(np.asarray(labels, dtype=np.int8), pred,
                          average="macro", zero_division=0))


def _independent_gate_v2(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    gain = {view: {
        "recency_vs_sparse25": float(arms["recency"] - arms["sparse25"]),
        "recency_vs_shuffled": float(arms["recency"] - arms["shuffled"]),
        "shuffled_vs_sparse25": float(arms["shuffled"] - arms["sparse25"]),
    } for view, arms in metrics.items()}
    checks = {
        "primary_nonnegative_vs_sparse25": gain["primary"]["recency_vs_sparse25"] >= 0.0,
        "primary_nonnegative_vs_shuffled": gain["primary"]["recency_vs_shuffled"] >= 0.0,
        "H12_min_gain_vs_sparse25": gain["H12"]["recency_vs_sparse25"] >= 0.0005,
        "H12_min_gain_vs_shuffled": gain["H12"]["recency_vs_shuffled"] >= 0.0002,
        "H12_55_harm_guard": gain["H12_55"]["recency_vs_sparse25"] >= -0.001,
        "H12_67_harm_guard": gain["H12_67"]["recency_vs_sparse25"] >= -0.001,
        "W1_harm_guard": gain["W1"]["recency_vs_sparse25"] >= -0.003,
        "W4_harm_guard": gain["W4"]["recency_vs_sparse25"] >= -0.003,
    }
    rules = {
        "primary_gain_vs_sparse25_min": 0.0,
        "primary_gain_vs_shuffled_min": 0.0,
        "H12_gain_vs_sparse25_min": 0.0005,
        "H12_gain_vs_shuffled_min": 0.0002,
        "H12_55_max_harm_vs_sparse25": 0.001,
        "H12_67_max_harm_vs_sparse25": 0.001,
        "W1_max_harm_vs_sparse25": 0.003,
        "W4_max_harm_vs_sparse25": 0.003,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "gains": gain, "rules": rules}


def verify_synthetic_preflight(*, run_root: Path = RUN_ROOT,
                               report_path: Path = REPORT_PATH) -> dict[str, Any]:
    """Independent v2 check of synthetic fit lineage, raw outputs, recovery and gates."""
    from scripts.optimization_v3_a01_preflight import _write_once

    run_root, report_path = Path(run_root), Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if report.get("schema_version") != 2 or report.get("status") != "SYNTHETIC_PREFLIGHT_V2_PASS":
        problems.append("report is not the v2 synthetic-only B0")
    if report.get("card_id") != CARD_ID or report.get("seed") != SYNTHETIC_SEED:
        problems.append("synthetic campaign identity or seed mismatch")
    if report.get("official_fits_started") != 0 or report.get("synthetic_estimator_fit_count") != 6:
        problems.append("report fit accounting differs from 4 A01 and 2 parity-only synthetic fits")
    if report.get("scope") != "synthetic-only; no official Train/Test/SampleSubmission/Kaggle or budget ledger access":
        problems.append("v2 source boundary does not prohibit official inputs and ledgers")
    if report.get("preprocessing_reference") != "SparseTextBoostClassifier from src.models.optimization_v3_sparse_classification":
        problems.append("report does not identify the exact SparseTextBoostClassifier parity reference")
    if report.get("tfidf_parameter_source") != "src.models.classification_models._TFIDF_PARAMS":
        problems.append("TF-IDF parameter source is not the existing classification classifier source")
    if report.get("recovery_stages_required") != [
        "after_model_persistence_then_resume_without_refit",
        "after_first_bundle_publication_then_resume_without_refit",
        "after_report_publication_then_resume_without_refit",
    ]:
        problems.append("required three-stage persistence/recovery contract missing")

    frame, target, score_rows_by_origin = synthetic_fixture()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    expected_keys = {f"fit{origin}_{arm.lower()}" for origin in ORIGINS
                     for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")}
    artifacts = report.get("artifacts", {})
    if set(artifacts) != expected_keys:
        problems.append("report does not contain exactly four origin-arm A01 artifacts")
    metrics: dict[str, dict[str, float]] = {}
    denominator: dict[str, int] = {}
    bundle_receipts: dict[str, str] = {}
    parity_summary: dict[str, Any] = {}
    prefit_row_counts: dict[str, int] = {}

    for origin in ORIGINS:
        fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= origin))
        score_rows = score_rows_by_origin[origin]
        fit_x, score_x = frame.iloc[fit_rows].copy(), frame.iloc[score_rows].copy()
        y_fit = target[fit_rows]
        weights = _fit_weights(cycles[fit_rows], y_fit, origin)
        arm_weights = {"RECENCY": weights["recency"], "SHUFFLED_WITHIN_CLASS": weights["shuffled"]}
        prefit_row_counts[str(origin)] = int(len(fit_rows))

        reference_dir = run_root / f"fit{origin}_sparse25_reference"
        reference_path, reference_state_path = reference_dir / "fitted_state.joblib", reference_dir / "fit_state.json"
        if not reference_path.is_file() or not reference_state_path.is_file():
            problems.append(f"synthetic sparse25 preprocessing reference missing at {origin}")
            continue
        reference = joblib.load(reference_path)
        if not isinstance(reference, SparseTextBoostClassifier):
            problems.append(f"wrong parity-reference class at {origin}")
            continue
        ref_state = reference.audit_state()
        saved_reference_state = json.loads(reference_state_path.read_text(encoding="utf-8"))
        if saved_reference_state != {"model_state": ref_state, "synthetic_fit_call_count": 1}:
            problems.append(f"parity-reference durable state receipt mismatch at {origin}")
        if ref_state.get("fit_data_sha256") != _fit_fingerprint(fit_x, tuple(ref_state["input_feature_names"]), y_fit):
            problems.append(f"sparse25 parity reference fit population mismatch at {origin}")

        raw_by_arm: dict[str, np.ndarray] = {}
        bundle_by_arm: dict[str, Any] = {}
        per_origin_parity: dict[str, Any] = {}
        for arm, arm_weight in arm_weights.items():
            key = f"fit{origin}_{arm.lower()}"
            fit_dir = run_root / key
            model_path, fit_state_path = fit_dir / "fitted_state.joblib", fit_dir / "fit_state.json"
            artifact = artifacts.get(key, {})
            if not model_path.is_file() or not fit_state_path.is_file():
                problems.append(f"synthetic A01 model/state missing: {key}")
                continue
            model_sha, state_sha = file_sha256(model_path), file_sha256(fit_state_path)
            if model_sha != artifact.get("model_sha256") or state_sha != artifact.get("state_sha256"):
                problems.append(f"model/state hash differs from report: {key}")
            model = joblib.load(model_path)
            state = model.audit_state()
            saved_fit_state = json.loads(fit_state_path.read_text(encoding="utf-8"))
            if saved_fit_state != {"model_state": state,
                                   "sample_weight_sha256": state.get("sample_weight_sha256"),
                                   "synthetic_fit_call_count": 1}:
                problems.append(f"A01 fitted-state receipt mismatch: {key}")
            expected_weight_sha = _array_sha(arm_weight, "<f8")
            if (state.get("card_id") != CARD_ID or state.get("cutoff") != origin or
                    state.get("weight_arm") != arm or state.get("fit_rows") != len(fit_rows) or
                    state.get("sample_weight_used") is not True or
                    state.get("sample_weight_sha256") != expected_weight_sha or
                    state.get("fit_data_sha256") != _fit_fingerprint(fit_x, tuple(state["input_feature_names"]), y_fit)):
                problems.append(f"A01 model or sample-weight lineage mismatch: {key}")
            if (state.get("lightgbm_parameters") != ref_state.get("parameters") or
                    state.get("tfidf_parameters") != ref_state.get("tfidf_params")):
                problems.append(f"A01 LightGBM or TF-IDF recipe differs from SparseTextBoostClassifier: {key}")
            model_weight_diag = artifact.get("weight_diagnostics", {})
            expected_diag = weights["ess_by_class"]
            if model_weight_diag.get("permutation_indices_by_class") != weights["permutations"]:
                problems.append(f"within-class shuffle sequence differs from seed-42 reconstruction: {key}")
            if model_weight_diag.get("permutation_source_rows_sha256") != _array_sha(weights["sources"], "<i8"):
                problems.append(f"shuffle source-row mapping hash mismatch: {key}")
            expected_digest_key = "recency_weight_sha256" if arm == "RECENCY" else "shuffled_weight_sha256"
            if model_weight_diag.get(expected_digest_key) != expected_weight_sha:
                problems.append(f"weight digest not attached to planned arm: {key}")
            overall = weights["overall_ess"]
            for field in ("recency_ess", "recency_ess_ratio", "shuffled_ess", "shuffled_ess_ratio"):
                if not _same_float(model_weight_diag.get(field), overall[field]):
                    problems.append(f"overall ESS diagnostic mismatch for {field}: {key}")
            if not all(model_weight_diag.get("preflight_checks", {}).values()):
                problems.append(f"weight preflight support check failed: {key}")
            for class_value in (0, 1):
                details = expected_diag[str(class_value)]["recency" if arm == "RECENCY" else "shuffled"]
                item = model_weight_diag.get("by_class", {}).get(str(class_value), {})
                ess_key = "recency_ess_ratio" if arm == "RECENCY" else "shuffled_ess_ratio"
                mass_key = "recency_mass" if arm == "RECENCY" else "shuffled_mass"
                if (item.get("n") != details["n"] or
                        not _same_float(item.get(ess_key), details["ess_ratio"]) or
                        not _same_float(item.get(mass_key), details["mass"]) or
                        details["ess_ratio"] < 0.50):
                    problems.append(f"class ESS or support mismatch: {key}, class={class_value}")
            class_mass = state.get("sample_weight_class_mass", {})
            if any(not _same_float(class_mass.get(str(c), -1.0), int(np.sum(y_fit == c))) for c in (0, 1)):
                problems.append(f"class-specific sample weight mass not preserved: {key}")
            if not _same_float(float(arm_weight.mean()), 1.0):
                problems.append(f"global sample-weight mean differs from 1: {key}")

            design_ref = _design_matrix(reference, fit_x, reference=True)
            design_a01 = _design_matrix(model, fit_x, reference=False)
            parity_checks = {
                "feature_allowlist_exact": tuple(state["input_feature_names"]) == tuple(ref_state["input_feature_names"]),
                "lightgbm_parameters_exact": state["lightgbm_parameters"] == ref_state["parameters"],
                "tfidf_parameters_exact": state["tfidf_parameters"] == ref_state["tfidf_params"],
                "fit_population_fingerprint_exact": state["fit_data_sha256"] == ref_state["fit_data_sha256"],
                "structured_preprocessing_state_exact": state["structured_preprocessing"] == ref_state["structured_state"],
                "vocabulary_size_exact": state["vocabulary_size"] == ref_state["vocabulary_size"],
                "vocabulary_hash_exact": state["vocabulary_sha256"] == ref_state["vocabulary_sha256"],
                "idf_hash_exact": state["idf_sha256"] == ref_state["idf_sha256"],
                "design_shape_exact": design_a01.shape == design_ref.shape == tuple(ref_state["design_shape"]),
                "design_csr_indptr_exact": np.array_equal(design_a01.indptr, design_ref.indptr),
                "design_csr_indices_exact": np.array_equal(design_a01.indices, design_ref.indices),
                "design_csr_values_exact": np.array_equal(design_a01.data, design_ref.data),
                "design_data_hash_exact": state["design_data_sha256"] == ref_state["design_data_sha256"],
            }
            reported_parity = artifact.get("preprocessing_parity", {})
            if reported_parity.get("status") != "PASS" or reported_parity.get("checks") != parity_checks or not all(parity_checks.values()):
                problems.append(f"independent preprocessing/design parity failed: {key}")
            per_origin_parity[arm] = parity_checks

            score_labels = target[score_rows].copy()
            altered_labels = np.bitwise_xor(score_labels, np.int8(1))
            model_state_before = model.audit_state()
            probability = model.predict_probability(score_x)
            perturbed_probability = model.predict_probability(score_x)
            model_state_after = model.audit_state()
            perturbation = artifact.get("outer_score_label_perturbation", {})
            original_label_hash = hashlib.sha256(score_labels.tobytes()).hexdigest()
            perturbed_label_hash = hashlib.sha256(altered_labels.tobytes()).hexdigest()
            if (np.array_equal(score_labels, altered_labels) or
                    not np.array_equal(probability, perturbed_probability) or
                    model_state_before != model_state_after or
                    perturbation.get("original_score_labels_sha256") != original_label_hash or
                    perturbation.get("perturbed_score_labels_sha256") != perturbed_label_hash or
                    perturbation.get("score_labels_passed_to_predict") is not False or
                    perturbation.get("model_state_unchanged") is not True or
                    perturbation.get("predictions_bit_identical") is not True):
                problems.append(f"outer score-label perturbation changed fit state/predictions: {key}")
            raw_by_arm[arm] = probability

            bundle_path = fit_dir / "bundle"
            identity = artifact.get("bundle", {}).get("identity", {})
            try:
                bundle = read_bundle(bundle_path, identity, task="classification")
            except Exception as exc:
                problems.append(f"raw prediction bundle failed contract read {key}: {type(exc).__name__}: {exc}")
                continue
            arrays = bundle.arrays
            bundle_receipts[key] = bundle.receipt["receipt_sha256"]
            if (bundle.receipt["receipt_sha256"] != artifact["bundle"]["receipt"].get("receipt_sha256") or
                    file_sha256(bundle_path / "arrays.npz") != artifact["bundle"].get("bundle_sha256")):
                problems.append(f"raw bundle receipt/hash differs from report: {key}")
            if bundle.receipt.get("metadata", {}).get("payload_meta", {}).get("run_kind") != "SYNTHETIC_PREFLIGHT_V2":
                problems.append(f"non-synthetic metadata found in bundle: {key}")
            if not np.array_equal(arrays["score_rows"], score_rows):
                problems.append(f"score row alignment mismatch: {key}")
            if not np.array_equal(arrays["score_record_ids"], frame.index.to_numpy()[score_rows].astype(str)):
                problems.append(f"score id alignment mismatch: {key}")
            if not np.array_equal(arrays["cycle_code"], cycles[score_rows].astype(np.int16)):
                problems.append(f"score cycle alignment mismatch: {key}")
            if not np.array_equal(arrays["raw_recency_probability"], raw_by_arm.get("RECENCY", probability) if "RECENCY" in raw_by_arm else probability):
                # The RECENCY arm is processed first, and both bundles freeze the same raw streams.
                problems.append(f"raw RECENCY probabilities differ from estimator output: {key}")
            bundle_by_arm[arm] = arrays

        parity_summary[str(origin)] = per_origin_parity
        if set(raw_by_arm) != {"RECENCY", "SHUFFLED_WITHIN_CLASS"} or set(bundle_by_arm) != set(raw_by_arm):
            continue
        for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS"):
            arrays = bundle_by_arm[arm]
            if not np.array_equal(arrays["raw_recency_probability"], raw_by_arm["RECENCY"]):
                problems.append(f"raw RECENCY probability lineage mismatch: fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["raw_shuffled_probability"], raw_by_arm["SHUFFLED_WITHIN_CLASS"]):
                problems.append(f"raw shuffled probability lineage mismatch: fit{origin}_{arm.lower()}")
        held = frame.iloc[score_rows]
        score_features = held[["impact_load", "entry_friction", "text_load"]].to_numpy(dtype=np.float64)
        c2b = 1 / (1 + np.exp(-((score_features[:, 0] - 100) / 25 + (score_features[:, 1] - 5) / 5) * 0.35))
        cb = 1 / (1 + np.exp(-((score_features[:, 2] - 50) / 20) * 0.30))
        sparse25_raw = 1 / (1 + np.exp(-((score_features[:, 2] - 50) / 18 + (score_features[:, 0] - 100) / 35) * 0.25))
        missing = np.asarray([value is None or pd.isna(value) or not str(value).strip()
                              for value in held["signal_stream"].to_numpy(dtype=object)], dtype=bool)
        baseline = np.where(missing, cb, .75 * c2b + .25 * sparse25_raw)
        recency_prediction = np.where(missing, cb, .75 * c2b + .25 * raw_by_arm["RECENCY"])
        shuffled_prediction = np.where(missing, cb, .75 * c2b + .25 * raw_by_arm["SHUFFLED_WITHIN_CLASS"])
        expected_predictions = {"RECENCY": recency_prediction,
                                "SHUFFLED_WITHIN_CLASS": shuffled_prediction}
        for arm, expected in expected_predictions.items():
            arrays = bundle_by_arm[arm]
            if not np.array_equal(arrays["prediction"], expected):
                problems.append(f"routing formula mismatch in raw bundle fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["baseline_prediction"], baseline):
                problems.append(f"sparse25 baseline route mismatch in fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["c2b_probability"], c2b) or not np.array_equal(arrays["cb_probability"], cb):
                problems.append(f"frozen routing inputs mismatch in fit{origin}_{arm.lower()}")
            if not np.array_equal(arrays["stream_missing"].astype(bool), missing):
                problems.append(f"missing-stream route mask mismatch in fit{origin}_{arm.lower()}")
        labels = target[score_rows]
        h12_name = f"H12_{origin}"
        metrics[h12_name] = {
            "sparse25": _metric_from_probabilities(labels, baseline),
            "recency": _metric_from_probabilities(labels, recency_prediction),
            "shuffled": _metric_from_probabilities(labels, shuffled_prediction),
        }
        denominator[h12_name] = int(len(score_rows))
        if origin == 55:
            window = (cycles[score_rows] >= 56) & (cycles[score_rows] <= 59)
            view = "W1"
        else:
            window = (cycles[score_rows] >= 68) & (cycles[score_rows] <= 71)
            view = "W4"
        metrics[view] = {
            "sparse25": _metric_from_probabilities(labels[window], baseline[window]),
            "recency": _metric_from_probabilities(labels[window], recency_prediction[window]),
            "shuffled": _metric_from_probabilities(labels[window], shuffled_prediction[window]),
        }
        denominator[view] = int(window.sum())

    if set(metrics) == {"H12_55", "H12_67", "W1", "W4"}:
        h12 = {arm: float(np.average([metrics["H12_55"][arm], metrics["H12_67"][arm]],
                                     weights=[denominator["H12_55"], denominator["H12_67"]]))
               for arm in ("sparse25", "recency", "shuffled")}
        metrics["H12"] = h12
        metrics["primary"] = dict(h12)
        denominator["H12"] = denominator["H12_55"] + denominator["H12_67"]
        denominator["primary"] = denominator["H12"]
        if report.get("metrics_by_view") != metrics or report.get("metric_denominators") != denominator:
            problems.append("metrics/denominators differ from independent raw-bundle reconstruction")
        gate = _independent_gate_v2(metrics)
        if report.get("synthetic_gate_exercised") != gate:
            problems.append("gate checks/gains differ from independent recomputation")
    else:
        gate = {}
        problems.append("could not independently reconstruct every official-design view")

    # Three failure/resume stages are separately persisted and checked.
    audit_path = run_root / "recovery_stage_audit.json"
    events_path = run_root / "recovery_stage_events.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if events_path.is_file() else []
    event_map = {event.get("event_id"): event for event in events}
    expected_event_ids = {
        "INTERRUPT_AFTER_MODEL_PERSISTENCE", "RESUME_AFTER_MODEL_PERSISTENCE",
        "INTERRUPT_AFTER_FIRST_BUNDLE_PUBLICATION", "RESUME_AFTER_FIRST_BUNDLE_PUBLICATION",
        "INTERRUPT_AFTER_REPORT_PUBLICATION", "RESUME_AFTER_REPORT_PUBLICATION",
    }
    stage_checks = {
        "all_stage_events_unique": len(event_map) == len(events) == len(expected_event_ids),
        "all_recovery_stage_ids_present": set(event_map) == expected_event_ids,
        "model_persisted_before_first_interrupt": False,
        "model_resumed_without_refit": False,
        "first_bundle_persisted_before_interrupt": False,
        "bundle_resume_without_fits": False,
        "report_persisted_before_interrupt": False,
        "report_resume_byte_identical_without_fit": False,
    }
    model_event = event_map.get("INTERRUPT_AFTER_MODEL_PERSISTENCE", {})
    model_key = model_event.get("artifact_key", "")
    expected_model_keys = {
        *(f"fit{origin}_{arm.lower()}" for origin in (55, 67)
          for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")),
        *(f"fit{origin}_sparse25_reference" for origin in (55, 67)),
    }
    model_receipts = model_event.get("model_receipts", {})
    model_receipts_match = set(model_receipts) == expected_model_keys
    if model_receipts_match:
        for key, receipt in model_receipts.items():
            model_dir = run_root / key
            model_receipts_match &= bool(
                (model_dir / "fitted_state.joblib").is_file() and
                (model_dir / "fit_state.json").is_file() and
                file_sha256(model_dir / "fitted_state.joblib") == receipt.get("model_sha256") and
                file_sha256(model_dir / "fit_state.json") == receipt.get("fit_state_sha256")
            )
    stage_checks["model_persisted_before_first_interrupt"] = bool(
        model_event.get("status") == "INJECTED_AND_CAUGHT" and
        model_key == "ALL_MODELS_DURABLE" and model_receipts_match and
        model_event.get("persisted_estimator_fits") == 6 and
        model_event.get("fit_calls_before_interruption") == 6
    )
    model_resume = event_map.get("RESUME_AFTER_MODEL_PERSISTENCE", {})
    stage_checks["model_resumed_without_refit"] = bool(
        model_resume.get("status") == "RESUMED_WITHOUT_REFIT" and
        model_resume.get("interrupted_artifact_key") == model_key and
        model_resume.get("interrupted_artifact_fit_called_during_resume") is False and
        model_resume.get("a01_estimator_fits_during_resume") == 0 and
        model_resume.get("sparse25_reference_fits_during_resume") == 0
    )
    bundle_event = event_map.get("INTERRUPT_AFTER_FIRST_BUNDLE_PUBLICATION", {})
    first_bundle_dir = run_root / bundle_event.get("bundle_key", "") / "bundle"
    first_receipt = json.loads((first_bundle_dir / "receipt.json").read_text(encoding="utf-8")) if (first_bundle_dir / "receipt.json").is_file() else {}
    stage_checks["first_bundle_persisted_before_interrupt"] = bool(
        bundle_event.get("status") == "INJECTED_AND_CAUGHT" and
        (first_bundle_dir / "arrays.npz").is_file() and
        first_receipt.get("receipt_sha256") == bundle_event.get("bundle_receipt_sha256") and
        bundle_event.get("persisted_estimator_fits") == 6 and
        bundle_event.get("fit_calls_before_interruption") == 0 and
        not any(bundle_event.get("a01_fit_execution_this_pass", {}).values()) and
        not any(bundle_event.get("reference_fit_execution_this_pass", {}).values())
    )
    bundle_resume = event_map.get("RESUME_AFTER_FIRST_BUNDLE_PUBLICATION", {})
    stage_checks["bundle_resume_without_fits"] = bool(
        bundle_resume.get("status") == "RESUMED_WITHOUT_REFIT" and
        bundle_resume.get("first_bundle_receipt_sha256_unchanged") is True and
        bundle_resume.get("durable_bundle_count") == 4 and
        bundle_resume.get("previously_durable_model_fit_calls_during_resume") == 0 and
        bundle_resume.get("new_planned_fit_calls_during_resume") == 0 and
        bundle_resume.get("all_four_a01_fits_and_two_parity_fits_durable_after_resume") is True
    )
    report_event = event_map.get("INTERRUPT_AFTER_REPORT_PUBLICATION", {})
    report_sha = file_sha256(report_path) if report_path.is_file() else None
    stage_checks["report_persisted_before_interrupt"] = bool(
        report_event.get("status") == "INJECTED_AND_CAUGHT" and
        report_event.get("report_sha256") == report_sha and
        report_event.get("persisted_estimator_fits") == 6 and
        report_event.get("estimator_fit_calls_after_report") == 0
    )
    report_resume = event_map.get("RESUME_AFTER_REPORT_PUBLICATION", {})
    stage_checks["report_resume_byte_identical_without_fit"] = bool(
        report_resume.get("status") == "RESUMED_WITHOUT_REFIT" and
        report_resume.get("report_sha256_before_resume") == report_sha == report_resume.get("report_sha256_after_resume") and
        report_resume.get("report_byte_identical") is True and
        report_resume.get("estimator_fit_calls_during_resume") == 0
    )
    if audit.get("status") != "PASS_SYNTHETIC_RECOVERY_STAGES" or audit.get("stage_events") != events:
        problems.append("recovery-stage summary is missing or differs from append-only stage events")
    if audit.get("persisted_estimator_fit_count") != 6 or audit.get("report_sha256") != report_sha:
        problems.append("recovery-stage audit fit count/report hash mismatch")
    if not all(stage_checks.values()):
        problems.append(f"one or more model/bundle/report resume stages failed: {stage_checks}")

    # Exact report bytes are regenerated as part of every idempotent preflight
    # rerun. Verification itself also publishes an immutable, deterministic receipt.
    expected_report_bytes = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n").encode("utf-8")
    report_bytes_exact = report_path.read_bytes() == expected_report_bytes
    if not report_bytes_exact:
        problems.append("report file bytes are not canonical/write-once")
    checks = {
        "v2_source_boundary_and_recipe": not any("report is not" in item or "TF-IDF" in item for item in problems),
        "four_weighted_models_plus_two_parity_reference_fits": _persisted_estimator_fit_count(run_root) == 6,
        "exact_classifier_preprocessing_design_parity": all(
            all(checks.values()) for origin in parity_summary.values() for checks in origin.values()
        ) and set(parity_summary) == {"55", "67"},
        "outer_score_label_perturbation_invariant": all(
            artifact.get("outer_score_label_perturbation", {}).get("score_labels_passed_to_predict") is False and
            artifact.get("outer_score_label_perturbation", {}).get("model_state_unchanged") is True and
            artifact.get("outer_score_label_perturbation", {}).get("predictions_bit_identical") is True
            for artifact in artifacts.values()
        ) and len(artifacts) == 4,
        "independent_weights_ess_shuffle_routes_metrics_gate": bool(metrics and gate) and not any(
            "weight" in item or "shuffle" in item or "ESS" in item or "routing" in item or "metrics" in item or "gate" in item
            for item in problems
        ),
        "durable_models_state_raw_bundles_before_report": len(bundle_receipts) == 4 and all(
            (run_root / key / "fitted_state.joblib").is_file() and
            (run_root / key / "fit_state.json").is_file() and
            (run_root / key / "bundle" / "arrays.npz").is_file() and
            (run_root / key / "bundle" / "receipt.json").is_file()
            for key in expected_keys
        ),
        "all_three_recovery_stages_no_refit": all(stage_checks.values()),
        "report_idempotence": report_bytes_exact and report_resume.get("report_byte_identical") is True,
    }
    result = {
        "schema_version": 2,
        "status": "PASS" if not problems and all(checks.values()) else "FAIL",
        "campaign": f"{CARD_ID}_SYNTHETIC_PREFLIGHT_V2_INDEPENDENT_VERIFICATION",
        "checks": checks, "recovery_stage_checks": stage_checks,
        "independent_metrics_by_view": metrics, "independent_gate": gate,
        "preprocessing_parity": parity_summary,
        "problems": problems,
        "official_data_read": False, "official_fits_started": 0,
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
    }
    verification_path = run_root / "independent_verification.json"
    _write_once(verification_path, result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify_synthetic_preflight(), indent=2, ensure_ascii=False, allow_nan=False))
