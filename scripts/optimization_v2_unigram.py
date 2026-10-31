"""Run the four fixed Classification unigram pilot fits on official Train only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.research_campaign_classification import (
    ROUND1_AUDITS,
    ROUND1_CACHE,
    load_train,
)
from scripts.research_campaign_classification_stage2 import (
    _expected_rows,
    _load_round1_control,
    compact_metrics,
    score_rows,
)
from src.data.audit import sha256 as file_sha256
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v2_unigram import build_unigram_classifier
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import EXPANDED_FEATURES
from src.utils.paths import PROJECT_ROOT


CONFIG_DIR = ROOT / "experiments/optimization_v2/classification_unigram"
PROTOCOL_PATH = CONFIG_DIR / "protocol.json"
FREEZE_PATH = CONFIG_DIR / "freeze.json"
REPORT_DIR = ROOT / "reports/optimization_v2/classification_unigram"
RUN_DIR = ROOT / "experiments/runs/optimization_v2_classification_unigram"
OLD_CB_DIR = ROOT / "experiments/runs/dual_cls_catboost_v1"
OLD_CB_FREEZE_PATH = ROOT / "experiments/dual_campaign/classification/catboost_freeze.json"
PILOT_ENDS = (55, 67)
HORIZONS = {55: 67, 67: 79}
WINDOWS = {55: {"W1": (56, 59), "H12_E": (56, 67)},
           67: {"W4": (68, 71), "H12_L": (68, 79)}}
VARIANTS = ("unigram_counts", "unigram_binary")
SOURCE_FILES = (
    "experiments/optimization_v2/classification_unigram/protocol.json",
    "src/models/optimization_v2_unigram.py",
    "scripts/optimization_v2_unigram.py",
    "scripts/optimization_v2_verify_unigram.py",
    "tests/test_optimization_v2_unigram.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/models/campaign_classification_catboost.py",
    "src/utils/paths.py",
    "src/data/audit.py",
    "scripts/research_campaign_classification.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/research_campaign_classification_catboost.py",
    "experiments/classification_round1/protocol.json",
    "experiments/classification_round1/freeze.json",
    "experiments/dual_campaign/classification/catboost_protocol.json",
    "experiments/dual_campaign/classification/catboost_freeze.json",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return file_sha256(path)


def _save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"official_classification_train/{path.name}"


def _input_paths(train_path: Path) -> dict[str, Path]:
    inputs = {"official_classification_train": train_path,
              "round1_catboost_freeze": OLD_CB_FREEZE_PATH}
    for end in PILOT_ENDS:
        inputs[f"fit{end}/round1_c2b_cache"] = ROUND1_CACHE / f"fit{end}__C2_B.npz"
        inputs[f"fit{end}/round1_fit_audit"] = ROUND1_AUDITS / f"fit{end}.json"
        inputs[f"fit{end}/round1_catboost_cache"] = OLD_CB_DIR / f"fit{end}.npz"
        inputs[f"fit{end}/round1_catboost_metadata"] = OLD_CB_DIR / f"fit{end}.json"
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official Train or required archived control input missing: {missing}")
    return inputs


def identity() -> dict[str, Any]:
    _, _, train_path = load_train(None)
    paths = _input_paths(train_path)
    return {
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "source_code_sha256": {name: code_digest(ROOT / name) for name in SOURCE_FILES},
        "input_files": {key: {"path": _relative(path), "sha256": _sha(path)}
                        for key, path in paths.items()},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
    }


def freeze_protocol() -> dict[str, Any]:
    current = identity()
    if FREEZE_PATH.exists():
        frozen = _read(FREEZE_PATH)
        if frozen.get("identity") != current or frozen.get("signature") != digest_json(current):
            raise RuntimeError("Existing unigram freeze differs; preserve it and issue a versioned protocol amendment")
        return frozen
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError("Refusing to freeze after pilot cache artifacts exist")
    frozen = {
        "status": "FROZEN_BEFORE_REAL_FITS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_observed_at_assignment": _read(PROTOCOL_PATH)["source_head_observed_at_assignment"],
        "identity": current,
        "signature": digest_json(current),
        "hash_convention": "Training source is UTF-8 LF-normalized; protocol and input/control artifacts are byte-hashed; Train is exact official CSV bytes.",
        "source_boundary": "Official Classification Train only; Test and SampleSubmission were not opened.",
        "fit_boundary": "Four fixed model fits only: variants unigram_counts/unigram_binary at cutoffs 55 and 67.",
    }
    _save_new(FREEZE_PATH, frozen)
    return frozen


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("Freeze code, protocol, Train, and archived controls before any real fit")
    frozen = _read(FREEZE_PATH)
    current = identity()
    if frozen.get("identity") != current or frozen.get("signature") != digest_json(current):
        raise RuntimeError("Unigram pilot freeze does not match current code, protocol, Train, controls, or runtime")
    return frozen


def _validate_c2b_fit(frame: pd.DataFrame, y: np.ndarray, fit_end: int) -> dict[str, Any]:
    audit_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    audit_root = _read(audit_path)
    audit = audit_root.get("C2_B")
    if not isinstance(audit, dict) or audit.get("model") != "C2_B":
        raise ValueError(f"C2_B fit audit identity mismatch at cutoff {fit_end}")
    state = audit.get("fitted_state", {})
    features = tuple([*EXPANDED_FEATURES, "signal_stream"])
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise")
    fit = cycles.between(47, fit_end).to_numpy()
    if tuple(state.get("input_feature_names", ())) != features:
        raise ValueError(f"C2_B input feature allowlist mismatch at cutoff {fit_end}")
    if int(state.get("fit_rows", -1)) != int(fit.sum()) or int(audit.get("fit_rows", -1)) != int(fit.sum()):
        raise ValueError(f"C2_B archived fit population mismatch at cutoff {fit_end}")
    expected_fp = _fit_fingerprint(frame.loc[fit, list(features)], features, y[fit])
    if state.get("fit_data_sha256") != expected_fp:
        raise ValueError(f"C2_B archived full-fit fingerprint mismatch at cutoff {fit_end}")
    if audit.get("fitted_state_sha256") != digest_json(state):
        raise ValueError(f"C2_B archived state digest mismatch at cutoff {fit_end}")
    if audit.get("outer_validation_labels_passed_to_fit_or_selection") is not False:
        raise ValueError(f"C2_B archived audit does not certify outer-label isolation at cutoff {fit_end}")
    return {"fit_rows": int(fit.sum()), "fit_data_sha256": expected_fp,
            "fitted_state_sha256": audit["fitted_state_sha256"],
            "audit_sha256": _sha(audit_path)}


def _load_cb_missing(frame: pd.DataFrame, y: np.ndarray, fit_end: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = OLD_CB_DIR / f"fit{fit_end}.npz"
    metadata_path = cache_path.with_suffix(".json")
    metadata = _read(metadata_path)
    old_freeze = _read(OLD_CB_FREEZE_PATH)
    horizon = HORIZONS[fit_end]
    rows, _, c2_identity = _load_round1_control(frame, fit_end, horizon, "C2_B")
    with np.load(cache_path, allow_pickle=False) as archive:
        old_rows = archive["rows"].astype(np.int64, copy=True)
        probability = archive["candidate"].astype(np.float64, copy=True)
    if not np.array_equal(rows, old_rows):
        raise ValueError(f"Archived CB_MISSING rows differ from C2_B at cutoff {fit_end}")
    if metadata.get("freeze_signature") != old_freeze.get("signature"):
        raise ValueError(f"Archived CB_MISSING cache is detached from freeze at cutoff {fit_end}")
    if metadata.get("npz_sha256") != _sha(cache_path) or metadata.get("prediction_sha256") != array_digest(probability):
        raise ValueError(f"Archived CB_MISSING score/cache digest mismatch at cutoff {fit_end}")
    if probability.shape != rows.shape or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Archived CB_MISSING probabilities invalid at cutoff {fit_end}")
    fit = pd.to_numeric(frame["cycle_code"], errors="raise").between(47, fit_end).to_numpy()
    state = metadata.get("state", {})
    cb_features = tuple(EXPANDED_FEATURES)
    if state.get("id") != "CB_MISSING" or tuple(state.get("input_feature_names", ())) != cb_features:
        raise ValueError(f"Archived CB_MISSING feature allowlist mismatch at cutoff {fit_end}")
    if int(state.get("fit_rows", -1)) != int(fit.sum()):
        raise ValueError(f"Archived CB_MISSING must be fit on every row through cutoff {fit_end}")
    expected_fp = _fit_fingerprint(frame.loc[fit, list(cb_features)], cb_features, y[fit])
    if state.get("fit_data_sha256") != expected_fp:
        raise ValueError(f"Archived CB_MISSING fit fingerprint mismatch at cutoff {fit_end}")
    return rows, probability, {
        "cache_sha256": _sha(cache_path),
        "metadata_sha256": _sha(metadata_path),
        "prediction_sha256": array_digest(probability),
        "freeze_signature": metadata.get("freeze_signature"),
        "fit_rows": int(fit.sum()),
        "fit_data_sha256": expected_fp,
        "c2b_outer_control": c2_identity,
    }


def preflight_controls(frame: pd.DataFrame, y: np.ndarray) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for fit_end in PILOT_ENDS:
        horizon = HORIZONS[fit_end]
        rows, c2_probability, c2_identity = _load_round1_control(frame, fit_end, horizon, "C2_B")
        expected = _expected_rows(frame, fit_end, horizon)
        if not np.array_equal(rows, expected):
            raise ValueError(f"C2_B validation row identity mismatch at cutoff {fit_end}")
        c2_fit = _validate_c2b_fit(frame, y, fit_end)
        cb_rows, cb_probability, cb_identity = _load_cb_missing(frame, y, fit_end)
        if not np.array_equal(rows, cb_rows):
            raise ValueError(f"C2_B and CB_MISSING controls do not align at cutoff {fit_end}")
        missing = stream_missing_mask(frame.iloc[rows])
        records[fit_end] = {
            "rows": rows,
            "c2_probability": c2_probability,
            "cb_missing_probability": cb_probability,
            "missing": missing,
            "current_probability": np.where(missing, cb_probability, c2_probability),
            "identity": {"c2b_fit": c2_fit, "cb_missing": cb_identity},
        }
    return records


def _fit_variant(frame: pd.DataFrame, y: np.ndarray, fit_end: int,
                 variant: str, controls: dict[str, Any], freeze_signature: str) -> dict[str, Any]:
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
    rows = controls["rows"]
    missing = controls["missing"]
    present_rows = rows[~missing]
    model = build_unigram_classifier(variant)
    feature_names = list(model.input_feature_names)
    forbidden = {"record_id", "artifact_signal", "threat_rank"}
    if forbidden.intersection(feature_names):
        raise RuntimeError(f"Forbidden feature in {variant} input allowlist")
    X_fit = frame.iloc[fit_rows][feature_names].copy()
    y_fit = y[fit_rows]
    X_present = frame.iloc[present_rows][feature_names].copy()
    started = perf_counter()
    with threadpool_limits(limits=2):
        model.fit(X_fit, y_fit)
        state = model.audit_state()
        present_probability = np.asarray(model.predict_proba(X_present), dtype=np.float64)
        diagnostics = model.prediction_diagnostics()
    if digest_json(state) != digest_json(model.audit_state()):
        raise RuntimeError(f"{variant} fitted state changed during present-row prediction")
    if present_probability.shape != present_rows.shape or not np.isfinite(present_probability).all():
        raise RuntimeError(f"{variant} returned invalid present-stream probabilities")
    if ((present_probability < 0) | (present_probability > 1)).any():
        raise RuntimeError(f"{variant} probability fell outside [0,1]")
    routed = controls["cb_missing_probability"].copy()
    routed[~missing] = present_probability
    if not np.array_equal(routed[missing], controls["cb_missing_probability"][missing]):
        raise RuntimeError("Candidate changed the archived CB_MISSING branch")
    cache_path = RUN_DIR / f"fit{fit_end}__{variant}.npz"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite a prior fit artifact: {cache_path}")
    np.savez_compressed(
        cache_path,
        outer_rows=rows.astype(np.int64),
        present_rows=present_rows.astype(np.int64),
        present_probability=present_probability,
        routed_probability=routed,
        current_probability=controls["current_probability"],
        missing_stream_mask=missing.astype(np.uint8),
    )
    meta = {
        "status": "FIT_COMPLETE",
        "freeze_signature": freeze_signature,
        "variant": variant,
        "fit_end": int(fit_end),
        "fit_cycles": [47, int(fit_end)],
        "fit_rows": int(len(fit_rows)),
        "validation_cycles": [fit_end + 1, HORIZONS[fit_end]],
        "validation_rows": int(len(rows)),
        "present_prediction_rows": int(len(present_rows)),
        "missing_validation_rows": int(missing.sum()),
        "input_feature_names": feature_names,
        "fit_data_sha256": state["fit_data_sha256"],
        "fit_state": state,
        "prediction_diagnostics": diagnostics,
        "threshold": 0.50,
        "outer_validation_labels_passed_to_fit_or_selection": False,
        "route": "Candidate probabilities are computed only for present-stream rows; archived CB_MISSING probabilities are copied exactly to missing/blank rows.",
        "controls": controls["identity"],
        "runtime_seconds": perf_counter() - started,
        "present_probability_sha256": array_digest(present_probability),
        "routed_probability_sha256": array_digest(routed),
    }
    meta["cache_sha256"] = _sha(cache_path)
    _save_new(metadata_path, meta)
    return {"outer_rows": rows, "present_rows": present_rows,
            "present_probability": present_probability, "routed_probability": routed,
            "current_probability": controls["current_probability"], "missing": missing,
            "metadata": meta}


def _score_view(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                current: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    current_metrics = score_rows(frame, y, rows, current)
    candidate_metrics = score_rows(frame, y, rows, candidate)
    missing = stream_missing_mask(frame.iloc[rows])
    segment_metrics = {}
    for name, mask in (("present", ~missing), ("missing_or_blank", missing)):
        segment_rows = rows[mask]
        segment_metrics[name] = {
            "n": int(len(segment_rows)),
            "current_champion": compact_metrics(y[segment_rows], current[mask]) if len(segment_rows) else None,
            "candidate": compact_metrics(y[segment_rows], candidate[mask]) if len(segment_rows) else None,
        }
        if len(segment_rows):
            segment_metrics[name]["delta_macro_f1"] = float(
                segment_metrics[name]["candidate"]["macro_f1"]
                - segment_metrics[name]["current_champion"]["macro_f1"]
            )
    return {
        "n": int(len(rows)),
        "current_champion": current_metrics,
        "candidate": candidate_metrics,
        "delta_macro_f1": float(candidate_metrics["macro_f1"] - current_metrics["macro_f1"]),
        "stream_segments": segment_metrics,
    }


def _view_rows(record: dict[str, Any], frame: pd.DataFrame,
               low: int, high: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    use = (cycles[record["outer_rows"]] >= low) & (cycles[record["outer_rows"]] <= high)
    return (record["outer_rows"][use], record["current_probability"][use],
            record["routed_probability"][use])


def _concatenate(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.concatenate([part[0] for part in parts])
    current = np.concatenate([part[1] for part in parts])
    candidate = np.concatenate([part[2] for part in parts])
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Pooled evaluation rows overlap")
    return rows, current, candidate


def assemble_results(frame: pd.DataFrame, y: np.ndarray,
                     fits: dict[str, dict[int, dict[str, Any]]], freeze_signature: str) -> dict[str, Any]:
    view_ranges = {"W1": (55, 56, 59), "H12_E": (55, 56, 67),
                   "W4": (67, 68, 71), "H12_L": (67, 68, 79)}
    results: dict[str, Any] = {
        "status": "PILOT_COMPLETE_REVIEW_REQUIRED",
        "protocol_id": _read(PROTOCOL_PATH)["id"],
        "freeze_signature": freeze_signature,
        "candidate_fit_count": sum(len(records) for records in fits.values()),
        "candidate_fit_cutoffs": [55, 67],
        "test_read": False,
        "threshold": 0.50,
        "variants": {},
        "pooled": {},
        "continuation_gate": {},
        "adaptive_evidence_note": "Cycles 56-79 have prior-use history; this is adaptive development evidence, not an untouched holdout.",
    }
    for variant in VARIANTS:
        variant_result: dict[str, Any] = {"fit_records": {}, "views": {}}
        for end, names in ((55, ("W1", "H12_E")), (67, ("W4", "H12_L"))):
            record = fits[variant][end]
            window_data: dict[str, Any] = {}
            for name in names:
                _, low, high = view_ranges[name]
                rows, current, candidate = _view_rows(record, frame, low, high)
                window_data[name] = _score_view(frame, y, rows, current, candidate)
                variant_result["views"][name] = window_data[name]
            variant_result["fit_records"][str(end)] = {
                "fit_rows": record["metadata"]["fit_rows"],
                "present_prediction_rows": record["metadata"]["present_prediction_rows"],
                "missing_validation_rows": record["metadata"]["missing_validation_rows"],
                "runtime_seconds": record["metadata"]["runtime_seconds"],
                "fit_state": record["metadata"]["fit_state"],
                "cache_sha256": record["metadata"]["cache_sha256"],
                "metadata_sha256": _sha(RUN_DIR / f"fit{end}__{variant}.json"),
            }
        primary = _concatenate([
            _view_rows(fits[variant][55], frame, 56, 59),
            _view_rows(fits[variant][67], frame, 68, 71),
        ])
        h12 = _concatenate([
            _view_rows(fits[variant][55], frame, 56, 67),
            _view_rows(fits[variant][67], frame, 68, 79),
        ])
        pooled = {"W1+W4": _score_view(frame, y, *primary),
                  "H12_E+H12_L": _score_view(frame, y, *h12)}
        results["variants"][variant] = variant_result
        results["pooled"][variant] = pooled
        individual_deltas = [variant_result["views"][name]["delta_macro_f1"]
                             for name in ("W1", "W4", "H12_E", "H12_L")]
        primary_gain = pooled["W1+W4"]["delta_macro_f1"]
        h12_gain = pooled["H12_E+H12_L"]["delta_macro_f1"]
        improves_pool = primary_gain > 0 or h12_gain > 0
        guardrail = min(individual_deltas) >= -0.005 - 1e-12
        results["continuation_gate"][variant] = {
            "pooled_primary_delta_macro_f1": primary_gain,
            "pooled_h12_delta_macro_f1": h12_gain,
            "individual_view_delta_macro_f1": {
                name: variant_result["views"][name]["delta_macro_f1"]
                for name in ("W1", "W4", "H12_E", "H12_L")
            },
            "pooled_improvement_gate": bool(improves_pool),
            "individual_guardrail_gate": bool(guardrail),
            "pilot_gate_pass": bool(improves_pool and guardrail),
            "next_action": "Main review only; no additional fits are authorized by this pilot." if improves_pool and guardrail
                           else "Retire this family; no additional fits are authorized by this pilot.",
        }
    return results


def run_pilot() -> dict[str, Any]:
    frozen = verify_freeze()
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"Pilot output exists; refusing retry or overwrite: {RUN_DIR}")
    if (REPORT_DIR / "pilot_results.json").exists() or (REPORT_DIR / "preflight_failure.json").exists():
        raise FileExistsError("A prior pilot result or prefit failure is preserved; issue an amendment before retry")
    frame, y, train_path = load_train(None)
    if _sha(train_path) != frozen["identity"]["input_files"]["official_classification_train"]["sha256"]:
        raise RuntimeError("Official Train hash differs from the prefit freeze")
    try:
        controls = preflight_controls(frame, y)
    except Exception as exc:
        _save_new(REPORT_DIR / "preflight_failure.json", {
            "status": "PREFIT_CONTROL_PREFLIGHT_FAILED_PRESERVED",
            "freeze_signature": frozen["signature"],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "all_cutoffs_preflighted_before_candidate_fits": list(PILOT_ENDS),
            "candidate_fits_started": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise
    _save_new(RUN_DIR / "run_started.json", {
        "status": "FOUR_FIT_PILOT_STARTED",
        "freeze_signature": frozen["signature"],
        "fit_plan": [{"variant": variant, "fit_end": end}
                     for variant in VARIANTS for end in PILOT_ENDS],
        "controls_preflighted": True,
        "candidate_fits_started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    fit_records: dict[str, dict[int, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for end in PILOT_ENDS:
            try:
                record = _fit_variant(frame, y, end, variant, controls[end], frozen["signature"])
                fit_records[variant][end] = record
                print(f"FIT_COMPLETE variant={variant} fit_end={end} seconds={record['metadata']['runtime_seconds']:.1f}", flush=True)
            except Exception as exc:
                _save_new(REPORT_DIR / f"fit{end}__{variant}_failure.json", {
                    "status": "FIT_FAILURE_PRESERVED",
                    "freeze_signature": frozen["signature"],
                    "variant": variant,
                    "fit_end": end,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "automatic_retry_forbidden": True,
                })
                raise
    result = assemble_results(frame, y, fit_records, frozen["signature"])
    _save_new(REPORT_DIR / "pilot_results.json", result)
    print("PILOT_RESULTS", json.dumps(result["continuation_gate"], sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-only", action="store_true")
    group.add_argument("--run-pilot", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        frozen = freeze_protocol()
        print(frozen["status"], frozen["signature"])
    else:
        result = run_pilot()
        print(result["status"], result["candidate_fit_count"], "fits", REPORT_DIR / "pilot_results.json")


if __name__ == "__main__":
    main()
