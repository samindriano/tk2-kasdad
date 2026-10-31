"""Independently verify the frozen four-fit Classification unigram pilot."""
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

from scripts.research_campaign_classification import ROUND1_AUDITS, ROUND1_CACHE, load_train
from scripts.optimization_v2_unigram import (
    CONFIG_DIR,
    FREEZE_PATH,
    OLD_CB_DIR,
    OLD_CB_FREEZE_PATH,
    PILOT_ENDS,
    PROTOCOL_PATH,
    REPORT_DIR,
    ROOT as RUNNER_ROOT,
    RUN_DIR,
    SOURCE_FILES,
    VARIANTS,
)
from src.data.audit import sha256 as file_sha256
from src.models.classification_metrics import binary_metrics
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import EXPANDED_FEATURES


RESULT_PATH = REPORT_DIR / "pilot_results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
HORIZONS = {55: 67, 67: 79}
VIEWS = {"W1": (55, 56, 59), "H12_E": (55, 56, 67),
         "W4": (67, 68, 71), "H12_L": (67, 68, 79)}
EXPECTED_FEATURES = [*EXPANDED_FEATURES, "signal_stream"]
ORIGINAL_TFIDF = {
    "ngram_range": [1, 2], "min_df": 3, "max_df": 0.995,
    "max_features": 20000, "sublinear_tf": True,
    "lowercase": True, "strip_accents": "unicode", "dtype": "float64",
}
EXPECTED_VARIANT_PARAMS = {
    "unigram_counts": {"ngram_range": [1, 1], "min_df": 3, "max_df": 0.995,
                        "max_features": 20000, "sublinear_tf": True,
                        "lowercase": True, "strip_accents": "unicode", "dtype": "float64",
                        "binary": False},
    "unigram_binary": {"ngram_range": [1, 1], "min_df": 3, "max_df": 0.995,
                       "max_features": 20000, "sublinear_tf": False,
                       "lowercase": True, "strip_accents": "unicode", "dtype": "float64",
                       "binary": True},
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return file_sha256(path)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"official_classification_train/{path.name}"


def _identity() -> dict[str, Any]:
    _, _, train_path = load_train(None)
    inputs: dict[str, Path] = {
        "official_classification_train": train_path,
        "round1_catboost_freeze": OLD_CB_FREEZE_PATH,
    }
    for end in PILOT_ENDS:
        inputs[f"fit{end}/round1_c2b_cache"] = ROUND1_CACHE / f"fit{end}__C2_B.npz"
        inputs[f"fit{end}/round1_fit_audit"] = ROUND1_AUDITS / f"fit{end}.json"
        inputs[f"fit{end}/round1_catboost_cache"] = OLD_CB_DIR / f"fit{end}.npz"
        inputs[f"fit{end}/round1_catboost_metadata"] = OLD_CB_DIR / f"fit{end}.json"
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official Train or archived control input missing: {missing}")
    return {
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "source_code_sha256": {name: code_digest(ROOT / name) for name in SOURCE_FILES},
        "input_files": {key: {"path": _relative(path), "sha256": _sha(path)}
                        for key, path in inputs.items()},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
    }


def _expected_rows(frame: pd.DataFrame, end: int) -> np.ndarray:
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    return np.flatnonzero((cycle > end) & (cycle <= HORIZONS[end]))


def _load_controls(frame: pd.DataFrame, y: np.ndarray, end: int) -> dict[str, Any]:
    rows_expected = _expected_rows(frame, end)
    c2_cache = ROUND1_CACHE / f"fit{end}__C2_B.npz"
    c2_audit_path = ROUND1_AUDITS / f"fit{end}.json"
    with np.load(c2_cache, allow_pickle=False) as archive:
        c2_rows = archive["outer_rows"].astype(np.int64, copy=True)
        c2_probability = archive["probability"].astype(np.float64, copy=True)
    c2_audit = _read(c2_audit_path)["C2_B"]
    c2_state = c2_audit.get("fitted_state", {})
    fit = pd.to_numeric(frame["cycle_code"], errors="raise").between(47, end).to_numpy()
    if not np.array_equal(c2_rows, rows_expected) or c2_probability.shape != c2_rows.shape:
        raise AssertionError(f"C2_B cached outer rows mismatch at cutoff {end}")
    if (c2_audit.get("model") != "C2_B"
            or tuple(c2_state.get("input_feature_names", ())) != tuple(EXPECTED_FEATURES)
            or int(c2_state.get("fit_rows", -1)) != int(fit.sum())
            or int(c2_audit.get("fit_rows", -1)) != int(fit.sum())):
        raise AssertionError(f"C2_B full-fit identity/allowlist mismatch at cutoff {end}")
    c2_fp = _fit_fingerprint(frame.loc[fit, EXPECTED_FEATURES], tuple(EXPECTED_FEATURES), y[fit])
    if (c2_state.get("fit_data_sha256") != c2_fp
            or c2_audit.get("fitted_state_sha256") != digest_json(c2_state)
            or c2_audit.get("outer_validation_labels_passed_to_fit_or_selection") is not False
            or c2_audit.get("probability_sha256") != array_digest(c2_probability)):
        raise AssertionError(f"C2_B fit/score provenance mismatch at cutoff {end}")

    cb_cache = OLD_CB_DIR / f"fit{end}.npz"
    cb_meta_path = cb_cache.with_suffix(".json")
    cb_meta = _read(cb_meta_path)
    cb_freeze = _read(OLD_CB_FREEZE_PATH)
    with np.load(cb_cache, allow_pickle=False) as archive:
        cb_rows = archive["rows"].astype(np.int64, copy=True)
        cb_probability = archive["candidate"].astype(np.float64, copy=True)
    cb_state = cb_meta.get("state", {})
    cb_features = tuple(EXPANDED_FEATURES)
    cb_fp = _fit_fingerprint(frame.loc[fit, list(cb_features)], cb_features, y[fit])
    if (not np.array_equal(cb_rows, rows_expected)
            or cb_probability.shape != rows_expected.shape
            or cb_meta.get("freeze_signature") != cb_freeze.get("signature")
            or cb_meta.get("npz_sha256") != _sha(cb_cache)
            or cb_meta.get("prediction_sha256") != array_digest(cb_probability)
            or cb_state.get("id") != "CB_MISSING"
            or tuple(cb_state.get("input_feature_names", ())) != cb_features
            or int(cb_state.get("fit_rows", -1)) != int(fit.sum())
            or cb_state.get("fit_data_sha256") != cb_fp):
        raise AssertionError(f"Archived CB_MISSING full-fit identity/provenance mismatch at cutoff {end}")
    for name, probability in (("C2_B", c2_probability), ("CB_MISSING", cb_probability)):
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise AssertionError(f"Archived {name} probabilities invalid at cutoff {end}")
    missing = stream_missing_mask(frame.iloc[rows_expected])
    current = np.where(missing, cb_probability, c2_probability)
    return {
        "rows": rows_expected,
        "missing": missing,
        "current": current,
        "c2": c2_probability,
        "cb": cb_probability,
        "fit_rows": int(fit.sum()),
        "c2_fit_fingerprint": c2_fp,
        "cb_fit_fingerprint": cb_fp,
        "c2_cache_sha256": _sha(c2_cache),
        "c2_audit_sha256": _sha(c2_audit_path),
        "cb_cache_sha256": _sha(cb_cache),
        "cb_metadata_sha256": _sha(cb_meta_path),
    }


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    base = binary_metrics(actual, p, threshold=0.50)
    return {
        "n": base["n"],
        "macro_f1": base["macro_f1"],
        "confusion_matrix": base["confusion_matrix"],
        "per_class": base["per_class"],
        "positive_rate": base["positive_rate"],
        "positive_prediction_rate": base["positive_prediction_rate"],
        "average_precision": float(average_precision_score(actual, p)) if len(np.unique(actual)) == 2 else None,
        "brier_score": base["brier_score"],
        "false_positive": base["false_positive"],
        "false_negative": base["false_negative"],
        "threshold": 0.50,
    }


def _score_rows(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                probability: np.ndarray) -> dict[str, Any]:
    result = _metric(y[rows], probability)
    cycles = pd.to_numeric(frame.iloc[rows]["cycle_code"], errors="raise").to_numpy(dtype=int)
    per_cycle = []
    for cycle in sorted(np.unique(cycles)):
        use = cycles == cycle
        per_cycle.append({"cycle": int(cycle), **_metric(y[rows][use], probability[use])})
    result["per_cycle"] = per_cycle
    result["equal_cycle_macro_f1"] = float(np.mean([item["macro_f1"] for item in per_cycle]))
    result["cycles"] = [int(value) for value in sorted(np.unique(cycles))]
    return result


def _score_view(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                current: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    missing = stream_missing_mask(frame.iloc[rows])
    segments = {}
    for name, mask in (("present", ~missing), ("missing_or_blank", missing)):
        segment_rows = rows[mask]
        segments[name] = {
            "n": int(len(segment_rows)),
            "current_champion": _metric(y[segment_rows], current[mask]) if len(segment_rows) else None,
            "candidate": _metric(y[segment_rows], candidate[mask]) if len(segment_rows) else None,
        }
        if len(segment_rows):
            segments[name]["delta_macro_f1"] = float(
                segments[name]["candidate"]["macro_f1"] - segments[name]["current_champion"]["macro_f1"]
            )
    current_metrics = _score_rows(frame, y, rows, current)
    candidate_metrics = _score_rows(frame, y, rows, candidate)
    return {"n": int(len(rows)), "current_champion": current_metrics,
            "candidate": candidate_metrics,
            "delta_macro_f1": float(candidate_metrics["macro_f1"] - current_metrics["macro_f1"]),
            "stream_segments": segments}


def _validate_candidate_state(state: dict[str, Any], variant: str, end: int,
                              frame: pd.DataFrame, y: np.ndarray) -> None:
    fit = pd.to_numeric(frame["cycle_code"], errors="raise").between(47, end).to_numpy()
    if (state.get("id") != f"OV2_{variant.upper()}"
            or state.get("kind") != "lr_hybrid"
            or state.get("input_feature_names") != EXPECTED_FEATURES
            or int(state.get("fit_rows", -1)) != int(fit.sum())):
        raise AssertionError(f"Candidate identity/fit-population/feature allowlist mismatch: {variant}/fit{end}")
    expected_fp = _fit_fingerprint(frame.loc[fit, EXPECTED_FEATURES], tuple(EXPECTED_FEATURES), y[fit])
    if state.get("fit_data_sha256") != expected_fp:
        raise AssertionError(f"Candidate fit fingerprint mismatch: {variant}/fit{end}")
    state_body = dict(state)
    stored_state_hash = state_body.pop("model_state_sha256", None)
    if stored_state_hash != digest_json(state_body):
        raise AssertionError(f"Candidate state hash mismatch: {variant}/fit{end}")
    representation = state.get("text_representation", {})
    actual_text = representation.get("actual_text_state", {})
    if (representation.get("variant") != variant
            or representation.get("base_c2b_tfidf_parameters") != ORIGINAL_TFIDF
            or actual_text.get("variant") != variant
            or actual_text.get("parameters") != EXPECTED_VARIANT_PARAMS[variant]
            or actual_text.get("vocabulary_size", 0) <= 0
            or not actual_text.get("vocabulary_sha256")
            or not actual_text.get("idf_sha256")):
        raise AssertionError(f"Candidate vectorizer audit parameters/state mismatch: {variant}/fit{end}")


def _compare(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"Result structure mismatch at {path}")
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"Result list mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        if actual is None or not np.isclose(float(expected), float(actual), rtol=0, atol=1e-12):
            raise AssertionError(f"Result numeric mismatch at {path}: {expected!r} != {actual!r}")
    elif expected != actual:
        raise AssertionError(f"Result value mismatch at {path}: {expected!r} != {actual!r}")


def verify() -> dict[str, Any]:
    frozen = _read(FREEZE_PATH)
    current_identity = _identity()
    if (frozen.get("identity") != current_identity
            or frozen.get("signature") != digest_json(current_identity)):
        raise AssertionError("Frozen source, protocol, Train, controls, or runtime identity mismatch")
    protocol = _read(PROTOCOL_PATH)
    result = _read(RESULT_PATH)
    if result.get("freeze_signature") != frozen["signature"]:
        raise AssertionError("Pilot result is detached from the frozen protocol")
    if result.get("candidate_fit_count") != 4 or result.get("candidate_fit_cutoffs") != [55, 67]:
        raise AssertionError("Pilot did not report exactly four fits at cutoffs 55 and 67")
    if result.get("test_read") is not False or result.get("threshold") != 0.50:
        raise AssertionError("Pilot test/threshold policy mismatch")
    frame, y, train_path = load_train(None)
    if _sha(train_path) != frozen["identity"]["input_files"]["official_classification_train"]["sha256"]:
        raise AssertionError("Official Train changed after freeze")
    controls = {end: _load_controls(frame, y, end) for end in PILOT_ENDS}

    fits: dict[str, dict[int, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for end in PILOT_ENDS:
            cache_path = RUN_DIR / f"fit{end}__{variant}.npz"
            metadata_path = cache_path.with_suffix(".json")
            metadata = _read(metadata_path)
            control = controls[end]
            with np.load(cache_path, allow_pickle=False) as archive:
                rows = archive["outer_rows"].astype(np.int64, copy=True)
                present_rows = archive["present_rows"].astype(np.int64, copy=True)
                present_probability = archive["present_probability"].astype(np.float64, copy=True)
                routed = archive["routed_probability"].astype(np.float64, copy=True)
                current = archive["current_probability"].astype(np.float64, copy=True)
                missing = archive["missing_stream_mask"].astype(bool, copy=True)
            expected_present_rows = control["rows"][~control["missing"]]
            if (not np.array_equal(rows, control["rows"])
                    or not np.array_equal(present_rows, expected_present_rows)
                    or not np.array_equal(missing, control["missing"])
                    or not np.array_equal(current, control["current"])):
                raise AssertionError(f"Candidate row routing/current champion mismatch: {variant}/fit{end}")
            if (metadata.get("freeze_signature") != frozen["signature"]
                    or metadata.get("variant") != variant
                    or metadata.get("fit_end") != end
                    or metadata.get("threshold") != 0.50
                    or metadata.get("outer_validation_labels_passed_to_fit_or_selection") is not False
                    or metadata.get("cache_sha256") != _sha(cache_path)
                    or metadata.get("present_probability_sha256") != array_digest(present_probability)
                    or metadata.get("routed_probability_sha256") != array_digest(routed)):
                raise AssertionError(f"Candidate artifact metadata/digest mismatch: {variant}/fit{end}")
            if metadata.get("controls", {}).get("c2b_fit", {}).get("fit_data_sha256") != control["c2_fit_fingerprint"]:
                raise AssertionError(f"Candidate detached from C2_B full-fit identity: {variant}/fit{end}")
            if metadata.get("controls", {}).get("cb_missing", {}).get("fit_data_sha256") != control["cb_fit_fingerprint"]:
                raise AssertionError(f"Candidate detached from CB_MISSING full-fit identity: {variant}/fit{end}")
            if (metadata["controls"]["cb_missing"]["cache_sha256"] != control["cb_cache_sha256"]
                    or metadata["controls"]["cb_missing"]["metadata_sha256"] != control["cb_metadata_sha256"]):
                raise AssertionError(f"Candidate detached from exact CB_MISSING artifacts: {variant}/fit{end}")
            if (present_probability.shape != expected_present_rows.shape
                    or not np.isfinite(present_probability).all()
                    or ((present_probability < 0) | (present_probability > 1)).any()):
                raise AssertionError(f"Invalid candidate present-stream scores: {variant}/fit{end}")
            reconstructed = control["cb"].copy()
            reconstructed[~control["missing"]] = present_probability
            if (not np.array_equal(routed, reconstructed)
                    or not np.array_equal(routed[control["missing"]], control["cb"][control["missing"]])
                    or not np.array_equal(current[~control["missing"]], control["c2"][~control["missing"]])):
                raise AssertionError(f"Availability-route invariant failed: {variant}/fit{end}")
            state = metadata.get("fit_state", {})
            _validate_candidate_state(state, variant, end, frame, y)
            fit_rows = int(pd.to_numeric(frame["cycle_code"], errors="raise").between(47, end).sum())
            if metadata.get("fit_rows") != fit_rows or metadata.get("input_feature_names") != EXPECTED_FEATURES:
                raise AssertionError(f"Candidate fit boundary mismatch: {variant}/fit{end}")
            fits[variant][end] = {
                "outer_rows": rows,
                "present_rows": present_rows,
                "present_probability": present_probability,
                "routed_probability": routed,
                "current_probability": current,
                "missing": missing,
                "metadata": metadata,
            }

    calculated: dict[str, Any] = {
        "status": result["status"],
        "protocol_id": protocol["id"],
        "freeze_signature": frozen["signature"],
        "candidate_fit_count": 4,
        "candidate_fit_cutoffs": [55, 67],
        "test_read": False,
        "threshold": 0.50,
        "variants": {},
        "pooled": {},
        "continuation_gate": {},
        "adaptive_evidence_note": result["adaptive_evidence_note"],
    }

    def view_arrays(variant: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        end, low, high = VIEWS[name]
        record = fits[variant][end]
        cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)[record["outer_rows"]]
        use = (cycles >= low) & (cycles <= high)
        return (record["outer_rows"][use], record["current_probability"][use],
                record["routed_probability"][use])

    def join(left: tuple[np.ndarray, np.ndarray, np.ndarray],
             right: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.concatenate([left[0], right[0]])
        if len(np.unique(rows)) != len(rows):
            raise AssertionError("Pooled pilot rows overlap")
        return rows, np.concatenate([left[1], right[1]]), np.concatenate([left[2], right[2]])

    for variant in VARIANTS:
        views: dict[str, Any] = {}
        fit_records: dict[str, Any] = {}
        for name in ("W1", "W4", "H12_E", "H12_L"):
            rows, current, candidate = view_arrays(variant, name)
            views[name] = _score_view(frame, y, rows, current, candidate)
        for end in PILOT_ENDS:
            metadata = fits[variant][end]["metadata"]
            fit_records[str(end)] = {
                "fit_rows": metadata["fit_rows"],
                "present_prediction_rows": metadata["present_prediction_rows"],
                "missing_validation_rows": metadata["missing_validation_rows"],
                "runtime_seconds": metadata["runtime_seconds"],
                "fit_state": metadata["fit_state"],
                "cache_sha256": metadata["cache_sha256"],
                "metadata_sha256": _sha(RUN_DIR / f"fit{end}__{variant}.json"),
            }
        primary = join(view_arrays(variant, "W1"), view_arrays(variant, "W4"))
        h12 = join(view_arrays(variant, "H12_E"), view_arrays(variant, "H12_L"))
        pooled = {"W1+W4": _score_view(frame, y, *primary),
                  "H12_E+H12_L": _score_view(frame, y, *h12)}
        calculated["variants"][variant] = {"fit_records": fit_records, "views": views}
        calculated["pooled"][variant] = pooled
        deltas = {name: views[name]["delta_macro_f1"] for name in ("W1", "W4", "H12_E", "H12_L")}
        primary_gain = pooled["W1+W4"]["delta_macro_f1"]
        h12_gain = pooled["H12_E+H12_L"]["delta_macro_f1"]
        pooled_gate = primary_gain > 0 or h12_gain > 0
        guardrail = min(deltas.values()) >= -0.005 - 1e-12
        passed = pooled_gate and guardrail
        calculated["continuation_gate"][variant] = {
            "pooled_primary_delta_macro_f1": primary_gain,
            "pooled_h12_delta_macro_f1": h12_gain,
            "individual_view_delta_macro_f1": deltas,
            "pooled_improvement_gate": bool(pooled_gate),
            "individual_guardrail_gate": bool(guardrail),
            "pilot_gate_pass": bool(passed),
            "next_action": "Main review only; no additional fits are authorized by this pilot." if passed
                           else "Retire this family; no additional fits are authorized by this pilot.",
        }
    _compare(result, calculated, "pilot_results")

    verification = {
        "status": "PASS",
        "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "freeze_signature": frozen["signature"],
        "independent_metric_recalculation": "PASS",
        "fit_count": 4,
        "cutoffs": [55, 67],
        "candidate_variants": list(VARIANTS),
        "all_rows_through_cutoff_fit_verified": True,
        "training_fingerprint_and_feature_allowlist_verified": True,
        "fit_only_text_state_audit_present": True,
        "official_test_or_sample_opened": False,
        "threshold": 0.50,
        "route_invariants": "PASS: candidate computed only on present rows; archived CB_MISSING retained exactly on missing/blank rows; incumbent C2_B retained exactly on present rows.",
        "control_input_hashes": frozen["identity"]["input_files"],
        "candidate_cache_hashes": {
            f"fit{end}/{variant}": _sha(RUN_DIR / f"fit{end}__{variant}.npz")
            for variant in VARIANTS for end in PILOT_ENDS
        },
        "pilot_gate": calculated["continuation_gate"],
    }
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    args = parser.parse_args()
    if not args.verify:
        parser.error("Use --verify")
    verified = verify()
    if VERIFICATION_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite independent verification: {VERIFICATION_PATH}")
    VERIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VERIFICATION_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(verified, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(verified["status"], "verified", verified["fit_count"], "fits")


if __name__ == "__main__":
    main()
