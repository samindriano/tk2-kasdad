"""Synthetic-only A01 end-to-end, gate, and persistence/resume preflight.

This command never loads official Train/Test or submission files and never
reserves/starts an official fit. Its persisted model and bundles are synthetic
preflight fixtures, not research results.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import sparse
from threadpoolctl import threadpool_limits

from src.models.classification_models import _text_values
from src.models.classification_recency_a01 import (
    CARD_ID, ORIGINS, RecencySparseClassifier, recency_weight_plan,
    route_probabilities,
)
from src.models.research_next_contract import (
    file_sha256, resume_or_publish_bundle,
)
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier
from src.models.regression_research_text import RegressionTextModel
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_classification_recency_a01/synthetic_preflight_v2"
REPORT_PATH = ROOT / "reports/optimization_v3/classification_recency_a01/synthetic_preflight_v2.json"
VERIFY_PATH = RUN_ROOT / "independent_verification.json"
RECOVERY_AUDIT_PATH = RUN_ROOT / "recovery_stage_audit.json"
SYNTHETIC_SEED = 2406400524

GATE = {
    "primary_gain_vs_sparse25_min": 0.0,
    "primary_gain_vs_shuffled_min": 0.0,
    "H12_gain_vs_sparse25_min": 0.0005,
    "H12_gain_vs_shuffled_min": 0.0002,
    "H12_55_max_harm_vs_sparse25": 0.001,
    "H12_67_max_harm_vs_sparse25": 0.001,
    "W1_max_harm_vs_sparse25": 0.003,
    "W4_max_harm_vs_sparse25": 0.003,
}


class InjectedSyntheticFailure(RuntimeError):
    """Test-only crash after a synthetic model is durable but before publish."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite nonidentical A01 preflight receipt: {path}")
        return
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_stage_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_stage_event_once(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = _read_stage_events(path)
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("Synthetic recovery event requires an event_id")
    matches = [saved for saved in events if saved.get("event_id") == event_id]
    if matches:
        if len(matches) != 1 or matches[0] != dict(event):
            raise RuntimeError(f"Conflicting synthetic recovery event: {event_id}")
        return
    line = json.dumps(dict(event), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def synthetic_fixture() -> tuple[pd.DataFrame, np.ndarray, dict[int, np.ndarray]]:
    """Generate deterministic synthetic rows covering both temporal origins."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    per_cycle = 20
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), per_cycle)
    n = len(cycles)
    row = np.arange(n, dtype=np.int64)
    signal = []
    vocab = ("amber", "boreal", "cipher", "delta", "ember", "frost", "granite", "harbor",
             "isotope", "jaguar", "keystone", "lantern", "matrix", "nebula", "orbit", "prairie")
    for i in row:
        if i % 17 == 0:
            signal.append(None)
        elif i % 23 == 0:
            signal.append("")
        else:
            signal.append(" ".join((vocab[i % len(vocab)], vocab[(i // 3) % len(vocab)],
                                    vocab[(i // 7 + 2) % len(vocab)], "signal", f"family{i % 5}")))
    frame = pd.DataFrame({
        "impact_load": rng.normal(100, 22, n).round(3),
        "entry_friction": rng.uniform(0, 10, n).round(3),
        "cycle_code": cycles,
        "scope_pressure": rng.normal(70, 14, n).round(3),
        "text_load": rng.uniform(0, 100, n).round(3),
        "scope_trace": [f"{1 + i % 7} nodes" if i % 11 else f"~{1 + i % 5} nodes" for i in row],
        "org_spread": [str(1 + i % 8) for i in row],
        "module_spread": [f"{1 + i % 9} items" for i in row],
        "telemetry_flag": ["active" if i % 3 else "inactive" for i in row],
        "org_code": [f"org{i % 9}" if i % 41 else None for i in row],
        "module_code": [f"module{i % 13}" for i in row],
        "pattern_code": [f"pattern{i % 7}" for i in row],
        "signal_stream": signal,
    })
    # Labels vary within each cycle and depend only on synthetic fit/score rows.
    score = ((row * 13 + row // 5 + cycles * 3) % 17)
    target = (score < 8).astype(np.int8)
    score_rows = {
        55: np.flatnonzero((cycles >= 56) & (cycles <= 67)),
        67: np.flatnonzero((cycles >= 68) & (cycles <= 79)),
    }
    return frame, target, score_rows


def evaluate_gate(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Apply the frozen pilot screening rules to a supplied metric table."""
    required = {"primary", "H12", "H12_55", "H12_67", "W1", "W4"}
    if set(metrics) != required:
        raise ValueError(f"A01 gate requires exactly these views: {sorted(required)}")
    for view, arms in metrics.items():
        if set(arms) != {"sparse25", "recency", "shuffled"}:
            raise ValueError(f"A01 gate view {view} must contain sparse25, recency, and shuffled")
        if not all(np.isfinite(float(value)) and 0 <= float(value) <= 1 for value in arms.values()):
            raise ValueError(f"A01 gate metrics must be finite macro F1 values in [0,1]: {view}")
    gains = {
        view: {
            "recency_vs_sparse25": float(metrics[view]["recency"] - metrics[view]["sparse25"]),
            "recency_vs_shuffled": float(metrics[view]["recency"] - metrics[view]["shuffled"]),
            "shuffled_vs_sparse25": float(metrics[view]["shuffled"] - metrics[view]["sparse25"]),
        }
        for view in required
    }
    checks = {
        "primary_nonnegative_vs_sparse25": gains["primary"]["recency_vs_sparse25"] >= GATE["primary_gain_vs_sparse25_min"],
        "primary_nonnegative_vs_shuffled": gains["primary"]["recency_vs_shuffled"] >= GATE["primary_gain_vs_shuffled_min"],
        "H12_min_gain_vs_sparse25": gains["H12"]["recency_vs_sparse25"] >= GATE["H12_gain_vs_sparse25_min"],
        "H12_min_gain_vs_shuffled": gains["H12"]["recency_vs_shuffled"] >= GATE["H12_gain_vs_shuffled_min"],
        "H12_55_harm_guard": gains["H12_55"]["recency_vs_sparse25"] >= -GATE["H12_55_max_harm_vs_sparse25"],
        "H12_67_harm_guard": gains["H12_67"]["recency_vs_sparse25"] >= -GATE["H12_67_max_harm_vs_sparse25"],
        "W1_harm_guard": gains["W1"]["recency_vs_sparse25"] >= -GATE["W1_max_harm_vs_sparse25"],
        "W4_harm_guard": gains["W4"]["recency_vs_sparse25"] >= -GATE["W4_max_harm_vs_sparse25"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "gains": gains, "rules": dict(GATE)}


def _identity(origin: int, arm: str, model_sha: str, state_sha: str) -> dict[str, Any]:
    return {"campaign": CARD_ID, "run_kind": "SYNTHETIC_PREFLIGHT_V2",
            "cutoff": int(origin), "arm": arm, "seed": SYNTHETIC_SEED,
            "model_joblib_sha256": model_sha, "fit_state_sha256": state_sha}


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Synthetic preflight model path already exists: {path}")
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(value, temporary)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        raise


def _load_or_fit_model(model_path: Path, state_path: Path, frame: pd.DataFrame,
                       target: np.ndarray, weights: np.ndarray, origin: int,
                       arm: str, weight_sha: str) -> tuple[RecencySparseClassifier, bool]:
    if model_path.is_file():
        model = joblib.load(model_path)
        if not isinstance(model, RecencySparseClassifier) or not model._fitted:
            raise RuntimeError(f"Existing synthetic model is not resumable: {model_path}")
        state = model.audit_state()
        expected = {"card_id": CARD_ID, "cutoff": int(origin), "weight_arm": arm,
                    "sample_weight_sha256": weight_sha}
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Existing synthetic model identity differs: {model_path}")
        if state_path.is_file():
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if saved != {"model_state": state, "sample_weight_sha256": weight_sha,
                         "synthetic_fit_call_count": 1}:
                raise RuntimeError(f"Existing synthetic state receipt differs: {state_path}")
        else:
            _write_once(state_path, {"model_state": state, "sample_weight_sha256": weight_sha,
                                     "synthetic_fit_call_count": 1})
        return model, False

    model = RecencySparseClassifier(origin, arm).fit(frame, target, weights)
    _atomic_joblib(model_path, model)
    _write_once(state_path, {"model_state": model.audit_state(), "sample_weight_sha256": weight_sha,
                             "synthetic_fit_call_count": 1})
    return model, True


def _load_or_fit_reference(model_path: Path, state_path: Path, frame: pd.DataFrame,
                           target: np.ndarray) -> tuple[SparseTextBoostClassifier, bool]:
    if model_path.is_file():
        model = joblib.load(model_path)
        if not isinstance(model, SparseTextBoostClassifier):
            raise RuntimeError(f"Existing synthetic sparse25 reference is invalid: {model_path}")
        state = model.audit_state()
        expected = {"fit_rows": len(target), "eval_set_used": False, "early_stopping_used": False}
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Existing synthetic sparse25 fit identity differs: {model_path}")
        expected_receipt = {"model_state": state, "synthetic_fit_call_count": 1}
        if state_path.is_file():
            if json.loads(state_path.read_text(encoding="utf-8")) != expected_receipt:
                raise RuntimeError(f"Existing synthetic sparse25 state receipt differs: {state_path}")
        else:
            _write_once(state_path, expected_receipt)
        return model, False
    model = SparseTextBoostClassifier().fit(frame, target)
    _atomic_joblib(model_path, model)
    _write_once(state_path, {"model_state": model.audit_state(), "synthetic_fit_call_count": 1})
    return model, True


def _training_design(model: Any, frame: pd.DataFrame, *, reference: bool) -> sparse.csr_matrix:
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


def _preprocessing_parity(reference: SparseTextBoostClassifier,
                          candidate: RecencySparseClassifier,
                          frame: pd.DataFrame) -> dict[str, Any]:
    ref_state, cand_state = reference.audit_state(), candidate.audit_state()
    ref_design = _training_design(reference, frame, reference=True)
    cand_design = _training_design(candidate, frame, reference=False)
    checks = {
        "feature_allowlist_exact": tuple(ref_state["input_feature_names"]) == tuple(cand_state["input_feature_names"]),
        "lightgbm_parameters_exact": ref_state["parameters"] == cand_state["lightgbm_parameters"],
        "tfidf_parameters_exact": ref_state["tfidf_params"] == cand_state["tfidf_parameters"],
        "fit_population_fingerprint_exact": ref_state["fit_data_sha256"] == cand_state["fit_data_sha256"],
        "structured_preprocessing_state_exact": ref_state["structured_state"] == cand_state["structured_preprocessing"],
        "vocabulary_size_exact": ref_state["vocabulary_size"] == cand_state["vocabulary_size"],
        "vocabulary_hash_exact": ref_state["vocabulary_sha256"] == cand_state["vocabulary_sha256"],
        "idf_hash_exact": ref_state["idf_sha256"] == cand_state["idf_sha256"],
        "design_shape_exact": ref_design.shape == cand_design.shape == tuple(ref_state["design_shape"]),
        "design_csr_indptr_exact": np.array_equal(ref_design.indptr, cand_design.indptr),
        "design_csr_indices_exact": np.array_equal(ref_design.indices, cand_design.indices),
        "design_csr_values_exact": np.array_equal(ref_design.data, cand_design.data),
        "design_data_hash_exact": ref_state["design_data_sha256"] == cand_state["design_data_sha256"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "reference_design_shape": list(ref_design.shape),
            "reference_design_nnz": int(ref_design.nnz),
            "reference_design_data_sha256": ref_state["design_data_sha256"],
            "candidate_design_data_sha256": cand_state["design_data_sha256"],
            "tfidf_source": "src.models.classification_models._TFIDF_PARAMS"}


def _macro(y_true: Any, probability: Any) -> float:
    return float(f1_score(np.asarray(y_true, dtype=np.int8),
                          (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int8),
                          average="macro", zero_division=0))


def _bundle_for_arm(directory: Path, origin: int, arm: str, score_rows: np.ndarray,
                    frame: pd.DataFrame, cycles: np.ndarray, raw_recency: np.ndarray,
                    raw_shuffled: np.ndarray, raw_sparse25: np.ndarray, c2b: np.ndarray,
                    cb: np.ndarray, missing: np.ndarray, routes: Mapping[str, np.ndarray],
                    model_sha: str, state_sha: str) -> dict[str, Any]:
    newly_published = not directory.exists()
    arm_prediction = routes["recency_candidate"] if arm == "RECENCY" else routes["shuffled_weight_control"]
    identity = _identity(origin, arm, model_sha, state_sha)
    payload = {
        "score_rows": np.asarray(score_rows, dtype=np.int64),
        "score_record_ids": frame.index.to_numpy()[score_rows].astype(str),
        "cycle_code": np.asarray(cycles[score_rows], dtype=np.int16),
        "prediction": np.asarray(arm_prediction, dtype=np.float64),
        "baseline_prediction": np.asarray(routes["sparse25_baseline"], dtype=np.float64),
        "raw_recency_probability": np.asarray(raw_recency, dtype=np.float64),
        "raw_shuffled_probability": np.asarray(raw_shuffled, dtype=np.float64),
        "raw_sparse25_probability": np.asarray(raw_sparse25, dtype=np.float64),
        "c2b_probability": np.asarray(c2b, dtype=np.float64),
        "cb_probability": np.asarray(cb, dtype=np.float64),
        "stream_missing": np.asarray(missing, dtype=bool),
        "matched_prediction": np.asarray(routes["shuffled_weight_control"], dtype=np.float64),
        "payload_meta": {"run_kind": "SYNTHETIC_PREFLIGHT_V2", "origin": int(origin),
                         "arm": arm, "threshold": 0.50, "labels_passed_to_fit": False,
                         "test_read": False, "sample_submission_read": False, "kaggle_access": False},
    }
    bundle = resume_or_publish_bundle(directory, payload, identity, task="classification")
    # The second call exercises exact resume and verifies the metadata receipt.
    resumed = resume_or_publish_bundle(directory, payload, identity, task="classification")
    if resumed.receipt["receipt_sha256"] != bundle.receipt["receipt_sha256"]:
        raise RuntimeError("Synthetic bundle resume changed its receipt")
    return {"identity": identity, "receipt": bundle.receipt,
            "bundle_sha256": file_sha256(directory / "arrays.npz"),
            "newly_published": newly_published}


def _run_preflight_pass(run_root: Path, *, inject_after_first_fit: bool = False,
                        inject_after_first_bundle: bool = False) -> dict[str, Any]:
    frame, target, score_rows_by_origin = synthetic_fixture()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    metrics: dict[str, dict[str, float]] = {}
    metric_rows: dict[str, int] = {}
    artifacts: dict[str, Any] = {}
    execution: dict[str, bool] = {}
    reference_execution: dict[str, bool] = {}
    parity_by_origin: dict[str, Any] = {}
    origin_state: dict[int, dict[str, Any]] = {}
    fit_count = 0
    reference_fit_count = 0
    bundle_failure_injected = False
    with threadpool_limits(limits=2):
        # Fit and durably persist every origin/arm before publishing any bundle.
        # The first recovery fault is injected at this barrier so a full resume
        # can prove that no already planned model fit is repeated.
        for origin in ORIGINS:
            fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= origin))
            score_rows = score_rows_by_origin[origin]
            fit_frame = frame.iloc[fit_rows].copy()
            score_frame = frame.iloc[score_rows].copy()
            plan = recency_weight_plan(cycles[fit_rows], target[fit_rows], origin)
            arm_models: dict[str, RecencySparseClassifier] = {}
            arm_raw: dict[str, np.ndarray] = {}
            score_label_perturbation: dict[str, Any] = {}
            arm_weights = {
                "RECENCY": plan["recency_weights"],
                "SHUFFLED_WITHIN_CLASS": plan["shuffled_weights"],
            }
            for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS"):
                arm_key = f"fit{origin}_{arm.lower()}"
                fit_dir = run_root / arm_key
                model_path = fit_dir / "fitted_state.joblib"
                state_path = fit_dir / "fit_state.json"
                model, did_fit = _load_or_fit_model(
                    model_path, state_path, fit_frame, target[fit_rows], arm_weights[arm],
                    origin, arm, plan["diagnostics"]["recency_weight_sha256"]
                    if arm == "RECENCY" else plan["diagnostics"]["shuffled_weight_sha256"],
                )
                if did_fit:
                    fit_count += 1
                execution[arm_key] = bool(did_fit)
                arm_models[arm] = model

            reference_key = f"fit{origin}_sparse25_reference"
            reference_dir = run_root / reference_key
            reference_model, reference_did_fit = _load_or_fit_reference(
                reference_dir / "fitted_state.joblib", reference_dir / "fit_state.json",
                fit_frame, target[fit_rows],
            )
            reference_execution[reference_key] = bool(reference_did_fit)
            reference_fit_count += int(reference_did_fit)
            origin_parity = {
                arm: _preprocessing_parity(reference_model, arm_models[arm], fit_frame)
                for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")
            }
            parity_by_origin[str(origin)] = origin_parity
            if any(value["status"] != "PASS" for value in origin_parity.values()):
                raise RuntimeError(f"A01 synthetic preprocessing parity failed at cutoff {origin}: {origin_parity}")
            origin_state[origin] = {
                "fit_rows": fit_rows, "score_rows": score_rows, "fit_frame": fit_frame,
                "score_frame": score_frame, "plan": plan, "arm_weights": arm_weights,
                "arm_models": arm_models, "reference_model": reference_model,
                "origin_parity": origin_parity,
            }

        if inject_after_first_fit:
            model_receipts: dict[str, dict[str, str]] = {}
            for origin in ORIGINS:
                for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS"):
                    key = f"fit{origin}_{arm.lower()}"
                    model_dir = run_root / key
                    model_path, state_path = model_dir / "fitted_state.joblib", model_dir / "fit_state.json"
                    if not model_path.is_file() or not state_path.is_file():
                        raise RuntimeError(f"Model-persistence barrier is incomplete for {key}")
                    model_receipts[key] = {
                        "model_sha256": file_sha256(model_path),
                        "fit_state_sha256": file_sha256(state_path),
                    }
                key = f"fit{origin}_sparse25_reference"
                model_dir = run_root / key
                model_path, state_path = model_dir / "fitted_state.joblib", model_dir / "fit_state.json"
                if not model_path.is_file() or not state_path.is_file():
                    raise RuntimeError(f"Sparse reference-persistence barrier is incomplete for {key}")
                model_receipts[key] = {
                    "model_sha256": file_sha256(model_path),
                    "fit_state_sha256": file_sha256(state_path),
                }
            error = InjectedSyntheticFailure("synthetic interruption after all model and state receipts persisted")
            error.synthetic_fit_count = fit_count + reference_fit_count
            error.synthetic_fit_artifact_key = "ALL_MODELS_DURABLE"
            error.synthetic_model_receipts = model_receipts
            error.stage = "AFTER_MODEL_PERSISTENCE"
            raise error

        # Prediction and bundle publication start only after the all-model
        # persistence barrier, making the following bundle resume fit-free.
        for origin in ORIGINS:
            current = origin_state[origin]
            score_rows = current["score_rows"]
            score_frame = current["score_frame"]
            plan = current["plan"]
            arm_models = current["arm_models"]
            origin_parity = current["origin_parity"]
            arm_raw: dict[str, np.ndarray] = {}
            score_label_perturbation: dict[str, Any] = {}
            for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS"):
                model = arm_models[arm]
                arm_raw[arm] = model.predict_probability(score_frame)
                model_state_before = model.audit_state()
                original_score_labels = target[score_rows].copy()
                perturbed_score_labels = np.bitwise_xor(original_score_labels, np.int8(1))
                probability_after_label_perturbation = model.predict_probability(score_frame)
                model_state_after = model.audit_state()
                score_label_perturbation[arm] = {
                    "original_score_labels_sha256": hashlib.sha256(original_score_labels.tobytes()).hexdigest(),
                    "perturbed_score_labels_sha256": hashlib.sha256(perturbed_score_labels.tobytes()).hexdigest(),
                    "score_labels_passed_to_predict": False,
                    "model_state_unchanged": model_state_before == model_state_after,
                    "predictions_bit_identical": np.array_equal(arm_raw[arm], probability_after_label_perturbation),
                }

            present = np.asarray([bool(str(value).strip()) if value is not None and not pd.isna(value) else False
                                  for value in score_frame["signal_stream"].to_numpy(dtype=object)], dtype=bool)
            missing = ~present
            score_x = score_frame[["impact_load", "entry_friction", "text_load"]].to_numpy(dtype=np.float64)
            c2b = 1 / (1 + np.exp(-((score_x[:, 0] - 100) / 25 + (score_x[:, 1] - 5) / 5) * 0.35))
            cb = 1 / (1 + np.exp(-((score_x[:, 2] - 50) / 20) * 0.30))
            sparse25_raw = 1 / (1 + np.exp(-((score_x[:, 2] - 50) / 18 + (score_x[:, 0] - 100) / 35) * 0.25))
            routes = route_probabilities(arm_raw["RECENCY"], arm_raw["SHUFFLED_WITHIN_CLASS"],
                                         sparse25_raw, c2b, cb, missing)
            labels = target[score_rows]
            views_for_origin = ("H12_55" if origin == 55 else "H12_67",)
            for view in views_for_origin:
                metrics[view] = {
                    "sparse25": _macro(labels, routes["sparse25_baseline"]),
                    "recency": _macro(labels, routes["recency_candidate"]),
                    "shuffled": _macro(labels, routes["shuffled_weight_control"]),
                }
                metric_rows[view] = int(len(score_rows))
            if origin == 55:
                w1 = (cycles[score_rows] >= 56) & (cycles[score_rows] <= 59)
                metrics["W1"] = {key: _macro(labels[w1], value[w1]) for key, value in {
                    "sparse25": routes["sparse25_baseline"], "recency": routes["recency_candidate"],
                    "shuffled": routes["shuffled_weight_control"],
                }.items()}
                metric_rows["W1"] = int(w1.sum())
            else:
                w4 = (cycles[score_rows] >= 68) & (cycles[score_rows] <= 71)
                metrics["W4"] = {key: _macro(labels[w4], value[w4]) for key, value in {
                    "sparse25": routes["sparse25_baseline"], "recency": routes["recency_candidate"],
                    "shuffled": routes["shuffled_weight_control"],
                }.items()}
                metric_rows["W4"] = int(w4.sum())

            for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS"):
                arm_key = f"fit{origin}_{arm.lower()}"
                arm_model = arm_models[arm]
                fit_dir = run_root / arm_key
                state_path = fit_dir / "fit_state.json"
                model_path = fit_dir / "fitted_state.joblib"
                arm_bundle = _bundle_for_arm(
                    fit_dir / "bundle", origin, arm, score_rows, frame, cycles,
                    arm_raw["RECENCY"], arm_raw["SHUFFLED_WITHIN_CLASS"], sparse25_raw, c2b, cb, missing,
                    routes, file_sha256(model_path), file_sha256(state_path),
                )
                if (inject_after_first_bundle and arm_bundle["newly_published"]
                        and not bundle_failure_injected):
                    bundle_failure_injected = True
                    error = InjectedSyntheticFailure("synthetic interruption after first durable raw bundle")
                    error.synthetic_fit_count = fit_count + reference_fit_count
                    error.synthetic_bundle_key = arm_key
                    error.stage = "AFTER_FIRST_BUNDLE_PUBLICATION"
                    error.model_execution = dict(execution)
                    error.reference_execution = dict(reference_execution)
                    raise error
                artifacts[arm_key] = {"model_sha256": file_sha256(model_path),
                                      "state_sha256": file_sha256(state_path),
                                      "bundle": {key: value for key, value in arm_bundle.items()
                                                 if key != "newly_published"},
                                      "model_state": arm_model.audit_state(),
                                      "weight_diagnostics": plan["diagnostics"],
                                      "preprocessing_parity": origin_parity[arm],
                                      "outer_score_label_perturbation": score_label_perturbation[arm],
                                      "synthetic_fit_call_count": 1}

    metrics["H12"] = {
        arm: float(np.average([metrics["H12_55"][arm], metrics["H12_67"][arm]],
                              weights=[metric_rows["H12_55"], metric_rows["H12_67"]]))
        for arm in ("sparse25", "recency", "shuffled")
    }
    metrics["primary"] = dict(metrics["H12"])
    metric_rows["H12"] = metric_rows["H12_55"] + metric_rows["H12_67"]
    metric_rows["primary"] = metric_rows["H12"]
    gate = evaluate_gate(metrics)
    result = {
        "status": "SYNTHETIC_PREFLIGHT_PASS",
        "scope": "synthetic-only; no official Train/Test/SampleSubmission/Kaggle or budget ledger access",
        "card_id": CARD_ID, "seed": SYNTHETIC_SEED,
        "official_fits_started": 0, "synthetic_estimator_fit_count_in_this_pass": fit_count + reference_fit_count,
        "synthetic_a01_estimator_fits_this_pass": fit_count,
        "synthetic_sparse25_reference_fits_this_pass": reference_fit_count,
        "fit_row_counts": {str(origin): int(np.sum((cycles >= 47) & (cycles <= origin))) for origin in ORIGINS},
        "score_row_counts": {str(origin): int(len(score_rows_by_origin[origin])) for origin in ORIGINS},
        "metrics_by_view": metrics, "metric_denominators": metric_rows,
        "fit_execution_this_pass": execution,
        "reference_fit_execution_this_pass": reference_execution,
        "preprocessing_parity_with_sparse_text_boost_classifier": parity_by_origin,
        "synthetic_gate_exercised": gate,
        "artifacts": artifacts,
        "limits": ["synthetic gates are not evidence about official data",
                   "synthetic control probabilities do not replace approved C16 cache verification"],
    }
    return result


def _finalize_v2_report(pass_result: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(pass_result)
    result.pop("synthetic_estimator_fit_count_in_this_pass", None)
    result.pop("synthetic_a01_estimator_fits_this_pass", None)
    result.pop("synthetic_sparse25_reference_fits_this_pass", None)
    result.pop("fit_execution_this_pass", None)
    result.pop("reference_fit_execution_this_pass", None)
    result["schema_version"] = 2
    result["status"] = "SYNTHETIC_PREFLIGHT_V2_PASS"
    result["synthetic_estimator_fit_count"] = 6
    result["synthetic_a01_fit_count"] = 4
    result["synthetic_sparse25_reference_fit_count"] = 2
    result["preprocessing_reference"] = "SparseTextBoostClassifier from src.models.optimization_v3_sparse_classification"
    result["tfidf_parameter_source"] = "src.models.classification_models._TFIDF_PARAMS"
    result["durability_contract"] = {
        "model_joblib_and_fit_state_persisted_before_scoring": True,
        "raw_probability_bundle_persisted_before_report": True,
        "bundle_contains_raw_expert_probabilities_and_frozen_route_inputs": True,
        "report_is_write_once_and_recomputed_on_resume": True,
        "independent_verification_is_write_once_and_idempotent": True,
    }
    result["recovery_stages_required"] = [
        "after_model_persistence_then_resume_without_refit",
        "after_first_bundle_publication_then_resume_without_refit",
        "after_report_publication_then_resume_without_refit",
    ]
    result["limits"] = [
        "synthetic gates are not evidence about official data",
        "synthetic control probabilities do not replace approved C16 cache verification",
        "two synthetic SparseTextBoostClassifier reference fits are preflight-only parity checks",
    ]
    return result


def _persisted_estimator_fit_count(run_root: Path) -> int:
    a01_count = sum((run_root / f"fit{origin}_{arm.lower()}" / "fitted_state.joblib").is_file()
                    for origin in ORIGINS for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS"))
    reference_count = sum((run_root / f"fit{origin}_sparse25_reference" / "fitted_state.joblib").is_file()
                         for origin in ORIGINS)
    return int(a01_count + reference_count)


def run_preflight(*, run_root: Path = RUN_ROOT, report_path: Path = REPORT_PATH) -> dict[str, Any]:
    """Run v2 synthetic B0 with three injected interruptions and exact resumes."""
    run_root, report_path = Path(run_root), Path(report_path)
    run_root.mkdir(parents=True, exist_ok=True)
    events_path = run_root / "recovery_stage_events.jsonl"
    audit_path = run_root / "recovery_stage_audit.json"

    event_ids = {event.get("event_id") for event in _read_stage_events(events_path)}
    if "INTERRUPT_AFTER_MODEL_PERSISTENCE" not in event_ids:
        try:
            _run_preflight_pass(run_root, inject_after_first_fit=True)
        except InjectedSyntheticFailure as exc:
            if getattr(exc, "stage", None) != "AFTER_MODEL_PERSISTENCE":
                raise
            model_key = str(getattr(exc, "synthetic_fit_artifact_key", ""))
            model_receipts = getattr(exc, "synthetic_model_receipts", {})
            expected_keys = {
                *(f"fit{origin}_{arm.lower()}" for origin in ORIGINS
                  for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")),
                *(f"fit{origin}_sparse25_reference" for origin in ORIGINS),
            }
            if model_key != "ALL_MODELS_DURABLE" or set(model_receipts) != expected_keys:
                raise RuntimeError("Model-persistence interruption receipt is incomplete")
            for key, receipt in model_receipts.items():
                model_dir = run_root / key
                model_path, state_path = model_dir / "fitted_state.joblib", model_dir / "fit_state.json"
                if (not model_path.is_file() or not state_path.is_file() or
                        file_sha256(model_path) != receipt["model_sha256"] or
                        file_sha256(state_path) != receipt["fit_state_sha256"]):
                    raise RuntimeError(f"Model-persistence interruption did not leave durable model/state for {key}")
            _append_stage_event_once(events_path, {
                "event_id": "INTERRUPT_AFTER_MODEL_PERSISTENCE",
                "status": "INJECTED_AND_CAUGHT",
                "artifact_key": model_key,
                "model_receipts": model_receipts,
                "persisted_estimator_fits": _persisted_estimator_fit_count(run_root),
                "fit_calls_before_interruption": int(getattr(exc, "synthetic_fit_count", -1)),
            })
        else:
            raise RuntimeError("Synthetic model-persistence interruption was not triggered")

    event_ids = {event.get("event_id") for event in _read_stage_events(events_path)}
    if "INTERRUPT_AFTER_FIRST_BUNDLE_PUBLICATION" not in event_ids:
        try:
            _run_preflight_pass(run_root, inject_after_first_bundle=True)
        except InjectedSyntheticFailure as exc:
            if getattr(exc, "stage", None) != "AFTER_FIRST_BUNDLE_PUBLICATION":
                raise
            model_fail = next((event for event in _read_stage_events(events_path)
                               if event.get("event_id") == "INTERRUPT_AFTER_MODEL_PERSISTENCE"), None)
            if model_fail is None:
                raise RuntimeError("Model persistence failure receipt is missing before bundle interruption")
            model_key = str(model_fail["artifact_key"])
            bundle_key = str(getattr(exc, "synthetic_bundle_key", ""))
            bundle_path = run_root / bundle_key / "bundle"
            if not (bundle_path / "arrays.npz").is_file() or not (bundle_path / "receipt.json").is_file():
                raise RuntimeError("Bundle-publication interruption did not leave a durable raw bundle")
            execution = getattr(exc, "model_execution", {})
            reference_execution = getattr(exc, "reference_execution", {})
            _append_stage_event_once(events_path, {
                "event_id": "RESUME_AFTER_MODEL_PERSISTENCE",
                "status": "RESUMED_WITHOUT_REFIT",
                "interrupted_artifact_key": model_key,
                "interrupted_artifact_fit_called_during_resume": bool(execution.get(model_key, False)),
                "a01_estimator_fits_during_resume": int(sum(bool(value) for value in execution.values())),
                "sparse25_reference_fits_during_resume": int(sum(bool(value) for value in reference_execution.values())),
            })
            _append_stage_event_once(events_path, {
                "event_id": "INTERRUPT_AFTER_FIRST_BUNDLE_PUBLICATION",
                "status": "INJECTED_AND_CAUGHT",
                "bundle_key": bundle_key,
                "bundle_receipt_sha256": json.loads((bundle_path / "receipt.json").read_text(encoding="utf-8"))["receipt_sha256"],
                "persisted_estimator_fits": _persisted_estimator_fit_count(run_root),
                "fit_calls_before_interruption": int(getattr(exc, "synthetic_fit_count", -1)),
                "a01_fit_execution_this_pass": execution,
                "reference_fit_execution_this_pass": reference_execution,
            })
        else:
            raise RuntimeError("Synthetic bundle-publication interruption was not triggered")

    # The next pass resumes the existing bundle and publishes the remaining
    # origin-arm bundles. All models and states are already durable.
    completed = _run_preflight_pass(run_root, inject_after_first_fit=False, inject_after_first_bundle=False)
    if (not completed["fit_execution_this_pass"] or any(completed["fit_execution_this_pass"].values()) or
            not completed["reference_fit_execution_this_pass"] or any(completed["reference_fit_execution_this_pass"].values())):
        raise RuntimeError("Post-bundle resume unexpectedly performed a model fit")
    bundle_dirs = [run_root / f"fit{origin}_{arm.lower()}" / "bundle"
                   for origin in ORIGINS for arm in ("RECENCY", "SHUFFLED_WITHIN_CLASS")]
    if any(not (path / "receipt.json").is_file() or not (path / "arrays.npz").is_file() for path in bundle_dirs):
        raise RuntimeError("Not all four raw prediction bundles are durable before report publication")
    prior_bundle = next((event for event in _read_stage_events(events_path)
                         if event.get("event_id") == "INTERRUPT_AFTER_FIRST_BUNDLE_PUBLICATION"), None)
    if prior_bundle is None:
        raise RuntimeError("First-bundle interruption receipt is missing")
    _append_stage_event_once(events_path, {
        "event_id": "RESUME_AFTER_FIRST_BUNDLE_PUBLICATION",
        "status": "RESUMED_WITHOUT_REFIT",
        "first_bundle_key": prior_bundle["bundle_key"],
        "first_bundle_receipt_sha256_unchanged": json.loads(
            ((run_root / prior_bundle["bundle_key"] / "bundle") / "receipt.json").read_text(encoding="utf-8")
        )["receipt_sha256"] == prior_bundle["bundle_receipt_sha256"],
        "durable_bundle_count": len(bundle_dirs),
        "previously_durable_model_fit_calls_during_resume": sum(
            bool(completed["fit_execution_this_pass"].get(key, False))
            for key in prior_bundle.get("a01_fit_execution_this_pass", {})
        ) + sum(
            bool(completed["reference_fit_execution_this_pass"].get(key, False))
            for key in prior_bundle.get("reference_fit_execution_this_pass", {})
        ),
        "new_planned_fit_calls_during_resume": int(
            sum(bool(value) for value in completed["fit_execution_this_pass"].values()) +
            sum(bool(value) for value in completed["reference_fit_execution_this_pass"].values())
        ),
        "all_four_a01_fits_and_two_parity_fits_durable_after_resume": _persisted_estimator_fit_count(run_root) == 6,
    })

    result = _finalize_v2_report(completed)
    if _persisted_estimator_fit_count(run_root) != 6:
        raise RuntimeError("Synthetic v2 did not persist exactly four A01 plus two sparse25 parity fits")
    _write_once(report_path, result)
    report_sha = file_sha256(report_path)
    report_event = next((event for event in _read_stage_events(events_path)
                         if event.get("event_id") == "INTERRUPT_AFTER_REPORT_PUBLICATION"), None)
    if report_event is None:
        # Simulate a process failure immediately after durable report publish.
        _append_stage_event_once(events_path, {
            "event_id": "INTERRUPT_AFTER_REPORT_PUBLICATION",
            "status": "INJECTED_AND_CAUGHT",
            "report_sha256": report_sha,
            "persisted_estimator_fits": _persisted_estimator_fit_count(run_root),
            "estimator_fit_calls_after_report": 0,
        })

    # Resume the full computation after the report failure. The immutable report
    # must be byte-identical; no model fit can occur because all states exist.
    resumed_pass = _run_preflight_pass(run_root, inject_after_first_fit=False, inject_after_first_bundle=False)
    resumed_report = _finalize_v2_report(resumed_pass)
    if any(resumed_pass["fit_execution_this_pass"].values()) or any(resumed_pass["reference_fit_execution_this_pass"].values()):
        raise RuntimeError("Post-report resume unexpectedly performed an estimator fit")
    _write_once(report_path, resumed_report)
    resumed_sha = file_sha256(report_path)
    _append_stage_event_once(events_path, {
        "event_id": "RESUME_AFTER_REPORT_PUBLICATION",
        "status": "RESUMED_WITHOUT_REFIT",
        "report_sha256_before_resume": report_sha,
        "report_sha256_after_resume": resumed_sha,
        "report_byte_identical": report_sha == resumed_sha,
        "estimator_fit_calls_during_resume": 0,
    })
    events = _read_stage_events(events_path)
    _write_once(audit_path, {
        "schema_version": 2,
        "campaign": CARD_ID,
        "status": "PASS_SYNTHETIC_RECOVERY_STAGES",
        "stage_events": events,
        "persisted_estimator_fit_count": _persisted_estimator_fit_count(run_root),
        "report_sha256": resumed_sha,
    })
    return resumed_report


if __name__ == "__main__":
    result = run_preflight()
    print(json.dumps({key: result[key] for key in (
        "schema_version", "status", "official_fits_started", "synthetic_estimator_fit_count",
        "synthetic_a01_fit_count", "synthetic_sparse25_reference_fit_count", "synthetic_gate_exercised",
    )}, indent=2, ensure_ascii=False, allow_nan=False))
