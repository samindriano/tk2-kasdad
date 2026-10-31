"""Independent verifier for the fixed CatBoost native-text pilot artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.research_campaign_classification_stage2 import (
    ROUND1_AUDITS,
    ROUND1_CACHE,
    _load_round1_control,
    load_train,
)
from scripts.optimization_v2_catboost_text import verify_freeze as verify_v1_freeze
from src.models.classification_models import _fit_fingerprint
from src.models.classification_models import _text_values
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v2_catboost_text import (
    EMPTY_STREAM_SENTINEL,
    INPUT_FEATURE_NAMES,
    build_catboost_native_text_classifier,
)
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import EXPANDED_FEATURES


CONFIG = ROOT / "experiments/optimization_v2/classification_text"
V1_PROTOCOL_PATH = CONFIG / "protocol.json"
V1_FREEZE_PATH = CONFIG / "freeze.json"
V1_FAILURE_RECEIPT = ROOT / "reports/optimization_v2/classification_text/prefit_failure_v1.json"
V1_FAILURE_CACHE_COPY = ROOT / "experiments/runs/optimization_v2_classification_text/preflight_failure_v1.json"
PROTOCOL_PATH = CONFIG / "protocol_v2.json"
FREEZE_PATH = CONFIG / "freeze_v2.json"
RESULT_PATH = ROOT / "reports/optimization_v2/classification_text/pilot_results_v2.json"
CACHE_DIR = ROOT / "experiments/runs/optimization_v2_classification_text/v2"
OLD_CB_DIR = ROOT / "experiments/runs/dual_cls_catboost_v1"
OLD_CB_FREEZE = ROOT / "experiments/dual_campaign/classification/catboost_freeze.json"
ENDS = (55, 67)
HORIZONS = {55: 67, 67: 79}
WINDOWS = {"W1": (55, 56, 59), "W4": (67, 68, 71)}
SOURCE_FILES = (
    "src/models/optimization_v2_catboost_text.py",
    "scripts/optimization_v2_catboost_text.py",
    "scripts/optimization_v2_verify_catboost_text.py",
    "scripts/optimization_v2_catboost_text_v2.py",
    "scripts/optimization_v2_verify_catboost_text_v2.py",
    "tests/test_optimization_v2_catboost_text.py",
    "src/models/campaign_classification_catboost.py",
    "src/models/classification_models.py",
    "src/models/classification_metrics.py",
    "src/models/classification_round2_models.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/data/audit.py",
    "src/utils/paths.py",
    "scripts/research_campaign_classification.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/research_campaign_classification_catboost.py",
)
EXPECTED_FIXED_PARAMETERS = {
    "loss_function": "Logloss",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "use_best_model": False,
}
EXPECTED_TEXT_PROCESSING = {
    "tokenizers": [{
        "tokenizer_id": "Space", "delimiter": " ",
        "separator_type": "ByDelimiter", "lowercasing": "false",
    }],
    "dictionaries": [{
        "dictionary_id": "Unigram", "gram_order": "1",
        "occurence_lower_bound": "3", "max_dictionary_size": "20000",
    }],
    "feature_processing": {"default": [{
        "tokenizers_names": ["Space"], "dictionaries_names": ["Unigram"],
        "feature_calcers": ["BoW", "NaiveBayes"],
    }]},
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return f"official_classification_train/{path.name}"


def _input_files() -> dict[str, Any]:
    _, _, train = load_train(None)
    inputs: dict[str, Path] = {
        "official_classification_train": train,
        "round1_catboost_freeze": OLD_CB_FREEZE,
        "v1_pre_fit_failure_receipt": V1_FAILURE_RECEIPT,
        "v1_pre_fit_failure_cache_copy": V1_FAILURE_CACHE_COPY,
    }
    for end in ENDS:
        inputs[f"fit{end}/round1_c2b_cache"] = ROUND1_CACHE / f"fit{end}__C2_B.npz"
        inputs[f"fit{end}/round1_fit_audit"] = ROUND1_AUDITS / f"fit{end}.json"
        inputs[f"fit{end}/round1_catboost_cache"] = OLD_CB_DIR / f"fit{end}.npz"
        inputs[f"fit{end}/round1_catboost_metadata"] = OLD_CB_DIR / f"fit{end}.json"
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen Train/control input missing: {missing}")
    return {key: {"path": _relative(path), "sha256": _sha(path)}
            for key, path in inputs.items()}


def _recomputed_identity() -> dict[str, Any]:
    return {
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "v1_lineage_sha256": {
            "protocol": _sha(V1_PROTOCOL_PATH),
            "freeze": _sha(V1_FREEZE_PATH),
            "failure_receipt": _sha(V1_FAILURE_RECEIPT),
            "failure_cache_copy": _sha(V1_FAILURE_CACHE_COPY),
        },
        "source_code_sha256": {name: code_digest(ROOT / name) for name in SOURCE_FILES},
        "input_files": _input_files(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scikit-learn", "catboost", "threadpoolctl"]},
    }


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    probability = np.asarray(p, dtype=float)
    if actual.ndim != 1 or probability.shape != actual.shape:
        raise AssertionError("Metric inputs are not aligned vectors")
    if not len(actual):
        raise AssertionError("Cannot compute metrics on an empty view")
    if not np.isin(actual, (0, 1)).all() or not np.isfinite(probability).all():
        raise AssertionError("Metric inputs contain invalid labels or probabilities")
    if ((probability < 0) | (probability > 1)).any():
        raise AssertionError("Probabilities fall outside [0,1]")
    pred = probability >= 0.50
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (actual, pred.astype(np.int8)), 1)
    classes = []
    for label in (0, 1):
        tp = int(cm[label, label])
        fp = int(cm[:, label].sum() - tp)
        fn = int(cm[label, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        classes.append({"label": label, "support": int(cm[label].sum()),
                        "precision": precision, "recall": recall, "f1": f1})
    support = cm.sum(axis=1)
    both = bool((support > 0).all())
    return {
        "n": int(len(actual)),
        "threshold": 0.5,
        "confusion_matrix": cm.tolist(),
        "per_class": classes,
        "macro_f1": float(np.mean([item["f1"] for item in classes])),
        "positive_rate": float(actual.mean()),
        "positive_prediction_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(actual, probability)) if both else None,
        "brier_score": float(np.mean((probability - actual) ** 2)),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
    }


def _score_rows(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                probability: np.ndarray) -> dict[str, Any]:
    base = _metric(y[rows], probability)
    cycles = pd.to_numeric(frame.iloc[rows]["cycle_code"], errors="raise").to_numpy(dtype=int)
    by_cycle = []
    for cycle in sorted(np.unique(cycles)):
        use = cycles == cycle
        by_cycle.append({"cycle": int(cycle), **_metric(y[rows[use]], probability[use])})
    base["per_cycle"] = by_cycle
    base["equal_cycle_macro_f1"] = float(np.mean([item["macro_f1"] for item in by_cycle]))
    base["cycles"] = [int(value) for value in sorted(np.unique(cycles))]
    return base


def _score_or_none(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                   probability: np.ndarray) -> dict[str, Any] | None:
    if len(rows) == 0:
        return None
    return _score_rows(frame, y, rows, probability)


def _evaluate(frame: pd.DataFrame, y: np.ndarray, record: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = record["rows"]
    missing = record["missing"]
    current = record["current_champion"]
    candidate = record["candidate"]

    def segments(probability: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = {
            "all": _score_or_none(frame, y, rows, probability),
            "stream_present": _score_or_none(frame, y, rows[~missing], probability[~missing]),
            "stream_missing_or_blank": _score_or_none(frame, y, rows[missing], probability[missing]),
            "entry_friction": {},
        }
        values = frame.iloc[rows]["entry_friction"].to_numpy(dtype=object)
        labels = ["__MISSING__" if pd.isna(value) else str(value).strip() for value in values]
        for value in sorted(set(labels)):
            use = np.asarray(labels, dtype=object) == value
            output["entry_friction"][value] = _score_or_none(
                frame, y, rows[use], probability[use],
            )
        return output

    old = segments(current)
    new = segments(candidate)
    return {
        "n": int(len(rows)),
        "cycle_codes": [int(value) for value in sorted(pd.unique(frame.iloc[rows]["cycle_code"]))],
        "current_champion": old,
        "native_text_candidate": new,
        "delta": {
            "all_macro_f1": float(new["all"]["macro_f1"] - old["all"]["macro_f1"]),
            "stream_present_macro_f1": float(new["stream_present"]["macro_f1"]
                                             - old["stream_present"]["macro_f1"]),
            "stream_missing_or_blank_macro_f1": float(new["stream_missing_or_blank"]["macro_f1"]
                                                       - old["stream_missing_or_blank"]["macro_f1"]),
        },
    }


def _compare(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"Structure mismatch at {path}")
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"List length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        if actual is None or not np.isclose(float(expected), float(actual), rtol=0, atol=1e-12):
            raise AssertionError(f"Numeric mismatch at {path}: {expected!r} != {actual!r}")
    elif expected != actual:
        raise AssertionError(f"Value mismatch at {path}: {expected!r} != {actual!r}")


def _load_control(frame: pd.DataFrame, y: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    horizon = HORIZONS[end]
    rows, c2, c2_identity = _load_round1_control(frame, end, horizon, "C2_B")
    cache_path = OLD_CB_DIR / f"fit{end}.npz"
    meta_path = cache_path.with_suffix(".json")
    meta = _read(meta_path)
    frozen = _read(OLD_CB_FREEZE)
    with np.load(cache_path, allow_pickle=False) as z:
        old_rows = z["rows"].astype(np.int64, copy=True)
        cb = z["candidate"].astype(np.float64, copy=True)
    if not np.array_equal(rows, old_rows):
        raise AssertionError(f"CB_MISSING source rows differ for cutoff {end}")
    if meta.get("freeze_signature") != frozen.get("signature"):
        raise AssertionError(f"CB_MISSING source freeze mismatch for cutoff {end}")
    if meta.get("npz_sha256") != _sha(cache_path) or meta.get("prediction_sha256") != array_digest(cb):
        raise AssertionError(f"CB_MISSING source digest mismatch for cutoff {end}")
    miss = stream_missing_mask(frame.iloc[rows])
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise")
    fit = cycles.between(47, end).to_numpy()
    state = meta["state"]
    cb_features = tuple(state["input_feature_names"])
    if cb_features != tuple(EXPANDED_FEATURES):
        raise AssertionError(f"CB_MISSING source feature list differs for cutoff {end}")
    if int(state["fit_rows"]) != int(fit.sum()):
        raise AssertionError(f"CB_MISSING source must be fitted on all rows through cutoff {end}")
    expected_fp = _fit_fingerprint(frame.loc[fit, list(cb_features)], cb_features, y[fit])
    if expected_fp != state["fit_data_sha256"]:
        raise AssertionError(f"CB_MISSING source fit data fingerprint differs for cutoff {end}")
    if not np.isfinite(cb).all() or ((cb < 0) | (cb > 1)).any():
        raise AssertionError(f"CB_MISSING source scores invalid for cutoff {end}")
    c2_audit = _read(ROUND1_AUDITS / f"fit{end}.json").get("C2_B", {})
    c2_state = c2_audit.get("fitted_state", {})
    c2_features = tuple([*EXPANDED_FEATURES, "signal_stream"])
    if tuple(c2_state.get("input_feature_names", ())) != c2_features:
        raise AssertionError(f"C2_B source feature list mismatch for cutoff {end}")
    if (int(c2_state.get("fit_rows", -1)) != int(fit.sum())
            or int(c2_audit.get("fit_rows", -1)) != int(fit.sum())):
        raise AssertionError(f"C2_B source must be fitted on all rows through cutoff {end}")
    if c2_audit.get("fitted_state_sha256") != digest_json(c2_state):
        raise AssertionError(f"C2_B fitted-state source hash mismatch for cutoff {end}")
    c2_fit_fp = _fit_fingerprint(frame.loc[fit, list(c2_features)], c2_features, y[fit])
    if c2_state.get("fit_data_sha256") != c2_fit_fp:
        raise AssertionError(f"C2_B source full-fit fingerprint mismatch for cutoff {end}")
    if c2_audit.get("outer_validation_labels_passed_to_fit_or_selection") is not False:
        raise AssertionError(f"C2_B source did not certify outer-label isolation for cutoff {end}")
    return rows, c2, cb, {"missing": miss, "c2_identity": c2_identity,
                          "cb_cache_sha256": _sha(cache_path),
                          "cb_metadata_sha256": _sha(meta_path),
                          "c2_fit_fingerprint": c2_fit_fp,
                          "c2_fitted_state_sha256": c2_audit.get("fitted_state_sha256"),
                          "c2_audit_sha256": _sha(ROUND1_AUDITS / f"fit{end}.json"),
                          "cb_fit_fingerprint": expected_fp}


def verify() -> dict[str, Any]:
    protocol = _read(PROTOCOL_PATH)
    freeze = _read(FREEZE_PATH)
    v1_freeze = verify_v1_freeze()
    if v1_freeze.get("signature") != protocol.get("supersedes_freeze_signature"):
        raise AssertionError("V2 verification is detached from preserved v1 freeze")
    if _sha(V1_FAILURE_RECEIPT) != _sha(V1_FAILURE_CACHE_COPY):
        raise AssertionError("Tracked and ignored copies of the v1 failure receipt do not match")
    current_identity = _recomputed_identity()
    if freeze.get("identity") != current_identity or freeze.get("signature") != digest_json(current_identity):
        raise AssertionError("Pilot freeze identity does not match source, Train, control caches, or runtime")
    if protocol.get("source_head_observed_at_assignment") != freeze.get("source_head_observed_at_assignment"):
        raise AssertionError("Pilot protocol and freeze source HEAD differ")
    if not CACHE_DIR.is_dir():
        raise FileNotFoundError(f"Pilot cache directory missing: {CACHE_DIR}")

    frame, y, _ = load_train(None)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    real_stream_mask = ~stream_missing_mask(frame)
    if frame.loc[real_stream_mask, "signal_stream"].astype(str).map(
        lambda value: EMPTY_STREAM_SENTINEL in value.split()
    ).any():
        raise AssertionError("Official Train contains the reserved empty-stream sentinel token")
    records: dict[int, dict[str, np.ndarray]] = {}
    verified_fits = {}

    for end in ENDS:
        path = CACHE_DIR / f"fit{end}.npz"
        meta_path = path.with_suffix(".json")
        meta = _read(meta_path)
        if meta.get("status") != "FIT_COMPLETE" or meta.get("freeze_signature") != freeze["signature"]:
            raise AssertionError(f"Candidate metadata freeze/status mismatch for cutoff {end}")
        rows, c2, cb_missing, controls = _load_control(frame, y, end)
        with np.load(path, allow_pickle=False) as z:
            stored_rows = z["rows"].astype(np.int64, copy=True)
            present_rows = z["present_rows"].astype(np.int64, copy=True)
            present_probability = z["present_probability"].astype(np.float64, copy=True)
            candidate_route = z["candidate_route"].astype(np.float64, copy=True)
        np.testing.assert_array_equal(stored_rows, rows)
        missing = controls["missing"]
        expected_present = rows[~missing]
        np.testing.assert_array_equal(present_rows, expected_present)
        if (meta["npz_sha256"] != _sha(path)
                or meta["prediction_sha256"]["rows"] != array_digest(rows.astype(np.float64))
                or meta["prediction_sha256"]["present_rows"] != array_digest(present_rows.astype(np.float64))
                or meta["prediction_sha256"]["present_probability"] != array_digest(present_probability)
                or meta["prediction_sha256"]["candidate_route"] != array_digest(candidate_route)):
            raise AssertionError(f"Candidate artifact digest mismatch for cutoff {end}")

        fit = (cycles >= 47) & (cycles <= end)
        fit_rows = int(fit.sum())
        expected_missing_fit = int(stream_missing_mask(frame.loc[fit]).sum())
        state = meta["state"]
        state_for_hash = dict(state)
        state_hash = state_for_hash.pop("model_state_sha256", None)
        raw_text, raw_missing = _text_values(frame.loc[fit, "signal_stream"])
        normalized_text = [EMPTY_STREAM_SENTINEL if is_missing else value
                           for value, is_missing in zip(raw_text, raw_missing)]
        expected_text_hash = _json_sha(normalized_text)
        if (state.get("input_feature_names") != INPUT_FEATURE_NAMES
                or state.get("fit_rows") != fit_rows
                or state.get("fit_stream_missing_rows") != expected_missing_fit
                or state.get("empty_stream_sentinel") != EMPTY_STREAM_SENTINEL
                or state.get("text_feature_names") != ["signal_stream"]
                or state.get("text_feature_indices") != [len(state["output_feature_names"]) - 1]
                or state.get("fixed_parameters") != EXPECTED_FIXED_PARAMETERS
                or state.get("text_processing") != EXPECTED_TEXT_PROCESSING
                or state.get("tree_count") != 500
                or state.get("fit_text_sha256") != expected_text_hash
                or state_hash != _json_sha(state_for_hash)
                or state.get("decision_threshold") != 0.50
                or state.get("eval_set_used") is not False
                or state.get("early_stopping_used") is not False
                or state.get("test_labels_or_test_text_used") is not False):
            raise AssertionError(f"Candidate fit boundary/audit state mismatch for cutoff {end}")
        if set(("record_id", "artifact_signal", "threat_rank")).intersection(state["input_feature_names"]):
            raise AssertionError(f"Candidate feature list leaks a forbidden field for cutoff {end}")
        fit_frame = frame.loc[fit, list(INPUT_FEATURE_NAMES)]
        expected_fp = _fit_fingerprint(fit_frame, tuple(INPUT_FEATURE_NAMES), y[fit])
        if state.get("fit_data_sha256") != expected_fp or meta.get("fit_fingerprint_recomputed") != expected_fp:
            raise AssertionError(f"Candidate fit fingerprint mismatch for cutoff {end}")
        category_counts = state["fit_only_structured_preprocessor"]["category_counts"]
        if any(int(item["training_rows"]) != fit_rows for item in category_counts.values()):
            raise AssertionError(f"Structured category preprocessing was not fit on all fold rows for cutoff {end}")
        if meta.get("controls", {}).get("cb_missing", {}).get("cache_sha256") != controls["cb_cache_sha256"]:
            raise AssertionError(f"Pilot metadata is detached from original CB_MISSING cache for cutoff {end}")
        if meta.get("controls", {}).get("cb_missing", {}).get("metadata_sha256") != controls["cb_metadata_sha256"]:
            raise AssertionError(f"Pilot metadata is detached from original CB_MISSING metadata for cutoff {end}")
        c2_fit_record = meta.get("controls", {}).get("c2b_fit", {})
        if (c2_fit_record.get("fit_rows") != fit_rows
                or c2_fit_record.get("fit_data_sha256") != controls["c2_fit_fingerprint"]
                or c2_fit_record.get("fitted_state_sha256") != controls["c2_fitted_state_sha256"]
                or c2_fit_record.get("audit_sha256") != controls["c2_audit_sha256"]):
            raise AssertionError(f"Pilot metadata is detached from full-fit C2_B identity for cutoff {end}")
        if meta.get("controls", {}).get("c2b_outer_control") != controls["c2_identity"]:
            raise AssertionError(f"Pilot metadata C2_B cache/audit identity mismatch for cutoff {end}")

        if len(present_probability) != len(expected_present) or not np.isfinite(present_probability).all():
            raise AssertionError(f"Native-text present probabilities invalid for cutoff {end}")
        if ((present_probability < 0) | (present_probability > 1)).any():
            raise AssertionError(f"Native-text probabilities outside [0,1] for cutoff {end}")
        current = np.where(missing, cb_missing, c2)
        reconstructed = current.copy()
        reconstructed[~missing] = present_probability
        np.testing.assert_array_equal(candidate_route, reconstructed)
        np.testing.assert_array_equal(candidate_route[missing], cb_missing[missing])
        np.testing.assert_array_equal(current[~missing], c2[~missing])
        records[end] = {"rows": rows, "missing": missing,
                        "current_champion": current, "candidate": reconstructed}
        verified_fits[str(end)] = {
            "status": "PASS",
            "fit_rows": fit_rows,
            "present_prediction_rows": int(len(expected_present)),
            "fit_stream_missing_rows": expected_missing_fit,
            "missing_outer_rows": int(missing.sum()),
            "candidate_fit_fingerprint": expected_fp,
            "missing_branch_exact": True,
            "champion_present_branch_exact": True,
            "candidate_cache_sha256": _sha(path),
            "original_cb_missing_cache_sha256": controls["cb_cache_sha256"],
            "original_c2b_cache_sha256": controls["c2_identity"]["cache_sha256"],
        }

    windows: dict[str, dict[str, np.ndarray]] = {}
    for name, (end, low, high) in WINDOWS.items():
        rec = records[end]
        use = (cycles[rec["rows"]] >= low) & (cycles[rec["rows"]] <= high)
        windows[name] = {key: value[use] for key, value in rec.items()}

    pooled_primary = {key: np.concatenate([windows["W1"][key], windows["W4"][key]])
                      for key in windows["W1"]}
    pooled_h12 = {key: np.concatenate([records[55][key], records[67][key]])
                  for key in records[55]}
    expected_views = {
        "W1": _evaluate(frame, y, windows["W1"]),
        "W4": _evaluate(frame, y, windows["W4"]),
        "H12_E": _evaluate(frame, y, records[55]),
        "H12_L": _evaluate(frame, y, records[67]),
        "primary_W1_W4": _evaluate(frame, y, pooled_primary),
        "H12_pooled": _evaluate(frame, y, pooled_h12),
    }
    expected_gate = {
        "primary_W1_W4_improved": expected_views["primary_W1_W4"]["delta"]["all_macro_f1"] > 0,
        "H12_pooled_improved": expected_views["H12_pooled"]["delta"]["all_macro_f1"] > 0,
        "all_individual_pilot_views_within_0.01": all(
            expected_views[name]["delta"]["all_macro_f1"] >= -0.01
            for name in ("W1", "W4", "H12_E", "H12_L")
        ),
        "full_six_authorized": False,
    }
    expected_gate["pilot_gate_pass"] = bool(
        (expected_gate["primary_W1_W4_improved"] or expected_gate["H12_pooled_improved"])
        and expected_gate["all_individual_pilot_views_within_0.01"]
    )

    report = _read(RESULT_PATH)
    if report.get("status") != "VERIFIED_PILOT_REVIEW_REQUIRED":
        raise AssertionError("Pilot report missing review-required status")
    if report.get("freeze_signature") != freeze["signature"]:
        raise AssertionError("Pilot report freeze signature mismatch")
    if report.get("pilot_fits") != [55, 67] or report.get("test_data_read") is not False:
        raise AssertionError("Pilot report claims wrong scope or Test boundary")
    _compare(expected_views, report["views"], "views")
    _compare(expected_gate, report["acceptance"], "acceptance")
    if report["acceptance"].get("full_six_authorized") is not False:
        raise AssertionError("Pilot report improperly authorizes full-six continuation")

    return {
        "status": "PASS",
        "freeze_signature": freeze["signature"],
        "official_train_sha256": current_identity["input_files"]["official_classification_train"]["sha256"],
        "verified_fits": verified_fits,
        "pilot_gate": expected_gate,
        "test_data_read": False,
        "kaggle_api_or_upload": False,
        "independent_metrics": "Recomputed from cached row predictions and official Train labels using independent count-based metrics and sklearn average precision.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true", help="Write a new independent verification receipt.")
    args = parser.parse_args()
    result = verify()
    if args.write_report:
        path = ROOT / "reports/optimization_v2/classification_text/verification.json"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
