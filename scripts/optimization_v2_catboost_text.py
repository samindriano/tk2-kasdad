"""Frozen two-cutoff pilot for CatBoost native text on Classification Train.

Only cutoffs 55 and 67 are fit. The script never opens official Test data and
never submits to Kaggle. After this pilot is independently verified, it stops
for Main review; no later folds are implemented in this runner.
"""
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

from scripts.research_campaign_classification_stage2 import (
    PRIMARY_WINDOWS,
    ROUND1_AUDITS,
    ROUND1_CACHE,
    _load_round1_control,
    load_train,
    score_rows,
)
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v2_catboost_text import (
    EMPTY_STREAM_SENTINEL,
    INPUT_FEATURE_NAMES,
    build_catboost_native_text_classifier,
)
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import EXPANDED_FEATURES


CONFIG = ROOT / "experiments/optimization_v2/classification_text"
PROTOCOL_PATH = CONFIG / "protocol.json"
FREEZE_PATH = CONFIG / "freeze.json"
REPORT_DIR = ROOT / "reports/optimization_v2/classification_text"
CACHE_DIR = ROOT / "experiments/runs/optimization_v2_classification_text"
OLD_CB_DIR = ROOT / "experiments/runs/dual_cls_catboost_v1"
OLD_CB_FREEZE = ROOT / "experiments/dual_campaign/classification/catboost_freeze.json"
PILOT_ENDS = (55, 67)
HORIZONS = {55: 67, 67: 79}
WINDOWS = {55: ("W1", 56, 59), 67: ("W4", 68, 71)}
SOURCE_FILES = (
    "src/models/optimization_v2_catboost_text.py",
    "scripts/optimization_v2_catboost_text.py",
    "scripts/optimization_v2_verify_catboost_text.py",
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
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        # Avoid disclosing a personal raw-data directory in committed manifests.
        return f"official_classification_train/{path.name}"


def _input_files() -> dict[str, str]:
    _, _, train_path = load_train(None)
    paths: dict[str, Path] = {"official_classification_train": train_path}
    paths["round1_catboost_freeze"] = OLD_CB_FREEZE
    for end in PILOT_ENDS:
        paths[f"fit{end}/round1_c2b_cache"] = ROUND1_CACHE / f"fit{end}__C2_B.npz"
        paths[f"fit{end}/round1_fit_audit"] = ROUND1_AUDITS / f"fit{end}.json"
        paths[f"fit{end}/round1_catboost_cache"] = OLD_CB_DIR / f"fit{end}.npz"
        paths[f"fit{end}/round1_catboost_metadata"] = OLD_CB_DIR / f"fit{end}.json"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Train/control input files are missing: {missing}")
    return {key: {"path": _relative(path), "sha256": _sha(path)}
            for key, path in paths.items()}


def identity() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"Frozen protocol source is missing: {PROTOCOL_PATH}")
    return {
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "source_code_sha256": {name: code_digest(ROOT / name) for name in SOURCE_FILES},
        "input_files": _input_files(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "catboost", "threadpoolctl"]
        },
    }


def freeze_protocol() -> dict[str, Any]:
    current = identity()
    if FREEZE_PATH.exists():
        frozen = _read(FREEZE_PATH)
        if frozen.get("identity") != current or frozen.get("signature") != digest_json(current):
            raise RuntimeError("Existing text-pilot freeze differs; preserve it and issue a versioned amendment")
        return frozen
    if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
        raise FileExistsError(f"Refusing to freeze after pilot run artifacts exist: {CACHE_DIR}")
    frozen = {
        "status": "FROZEN_BEFORE_REAL_FITS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_observed_at_assignment": _read(PROTOCOL_PATH)["source_head_observed_at_assignment"],
        "identity": current,
        "signature": digest_json(current),
        "source_boundary": "Official Classification Train and the exact archived C2_B / CB_MISSING controls only; no Test read.",
        "fit_boundary": "Only fit cutoffs 55 and 67; both candidate models fit every row through the cutoff, including blank streams.",
    }
    _save_new(FREEZE_PATH, frozen)
    return frozen


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("Freeze code, protocol, Train, and control identities before any real fit")
    frozen = _read(FREEZE_PATH)
    current = identity()
    if frozen.get("identity") != current or frozen.get("signature") != digest_json(current):
        raise RuntimeError("Text-pilot freeze does not match current source, protocol, Train, controls, or runtime")
    return frozen


def _expected_rows(frame: pd.DataFrame, fit_end: int, horizon: int) -> np.ndarray:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    return np.flatnonzero((cycles > fit_end) & (cycles <= horizon))


def _load_cb_missing(frame: pd.DataFrame, y: np.ndarray, fit_end: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = OLD_CB_DIR / f"fit{fit_end}.npz"
    meta_path = cache_path.with_suffix(".json")
    meta = _read(meta_path)
    old_freeze = _read(OLD_CB_FREEZE)
    horizon = HORIZONS[fit_end]
    rows, c2_probability, c2_identity = _load_round1_control(frame, fit_end, horizon, "C2_B")
    with np.load(cache_path, allow_pickle=False) as archive:
        old_rows = archive["rows"].astype(np.int64, copy=True)
        cb_probability = archive["candidate"].astype(np.float64, copy=True)
    if not np.array_equal(rows, old_rows):
        raise ValueError(f"Archived CB_MISSING row identity differs at fit{fit_end}")
    if meta.get("freeze_signature") != old_freeze.get("signature"):
        raise ValueError(f"Archived CB_MISSING cache is detached from its frozen protocol at fit{fit_end}")
    if meta.get("npz_sha256") != _sha(cache_path) or meta.get("prediction_sha256") != array_digest(cb_probability):
        raise ValueError(f"Archived CB_MISSING cache hash mismatch at fit{fit_end}")
    if not np.isfinite(cb_probability).all() or ((cb_probability < 0) | (cb_probability > 1)).any():
        raise ValueError(f"Archived CB_MISSING probabilities invalid at fit{fit_end}")

    fit = pd.to_numeric(frame["cycle_code"], errors="raise").between(47, fit_end).to_numpy()
    missing_fit = stream_missing_mask(frame.loc[fit])
    state = meta.get("state", {})
    if state.get("id") != "CB_MISSING" or int(state.get("fit_rows", -1)) != int(missing_fit.sum()):
        raise ValueError(f"Archived CB_MISSING fit mask mismatch at fit{fit_end}")
    cb_fields = tuple(state.get("input_feature_names", ()))
    if cb_fields != tuple(EXPANDED_FEATURES):
        raise ValueError(f"Archived CB_MISSING feature allowlist changed at fit{fit_end}")
    fit_frame = frame.loc[fit].loc[~missing_fit, list(cb_fields)]
    fit_y = y[fit][~missing_fit]
    if _fit_fingerprint(fit_frame, cb_fields, fit_y) != state.get("fit_data_sha256"):
        raise ValueError(f"Archived CB_MISSING training fingerprint mismatch at fit{fit_end}")

    return rows, cb_probability, {
        "rows_sha256": array_digest(rows.astype(np.float64)),
        "c2b": c2_identity,
        "cb_missing": {
            "cache_sha256": _sha(cache_path),
            "metadata_sha256": _sha(meta_path),
            "probability_sha256": array_digest(cb_probability),
            "freeze_signature": meta["freeze_signature"],
            "fit_rows": int(missing_fit.sum()),
        },
        "c2b_probability_sha256": array_digest(c2_probability),
    }


def _score_or_none(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                   probability: np.ndarray) -> dict[str, Any] | None:
    if len(rows) == 0:
        return None
    return score_rows(frame, y, rows, probability)


def _evaluate(frame: pd.DataFrame, y: np.ndarray, record: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = record["rows"]
    missing = record["missing"]
    current = record["current_champion"]
    candidate = record["candidate"]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Evaluation view contains duplicate Train row indexes")

    def segments(probability: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = {
            "all": _score_or_none(frame, y, rows, probability),
            "stream_present": _score_or_none(frame, y, rows[~missing], probability[~missing]),
            "stream_missing_or_blank": _score_or_none(frame, y, rows[missing], probability[missing]),
            "entry_friction": {},
        }
        values = frame.iloc[rows]["entry_friction"].to_numpy(dtype=object)
        labels = ["__MISSING__" if pd.isna(v) else str(v).strip() for v in values]
        for value in sorted(set(labels)):
            mask = np.asarray(labels, dtype=object) == value
            output["entry_friction"][value] = _score_or_none(
                frame, y, rows[mask], probability[mask],
            )
        return output

    current_metrics = segments(current)
    candidate_metrics = segments(candidate)
    current_f1 = current_metrics["all"]["macro_f1"]
    candidate_f1 = candidate_metrics["all"]["macro_f1"]
    return {
        "n": int(len(rows)),
        "cycle_codes": [int(v) for v in sorted(pd.unique(frame.iloc[rows]["cycle_code"]))],
        "current_champion": current_metrics,
        "native_text_candidate": candidate_metrics,
        "delta": {
            "all_macro_f1": float(candidate_f1 - current_f1),
            "stream_present_macro_f1": float(
                candidate_metrics["stream_present"]["macro_f1"]
                - current_metrics["stream_present"]["macro_f1"]
            ),
            "stream_missing_or_blank_macro_f1": float(
                candidate_metrics["stream_missing_or_blank"]["macro_f1"]
                - current_metrics["stream_missing_or_blank"]["macro_f1"]
            ),
        },
    }


def _slice(record: dict[str, np.ndarray], cycles: np.ndarray,
           low: int, high: int) -> dict[str, np.ndarray]:
    use = (cycles[record["rows"]] >= low) & (cycles[record["rows"]] <= high)
    return {key: value[use] for key, value in record.items()}


def _concat(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {key: np.concatenate([part[key] for part in records]) for key in records[0]}
    if len(np.unique(result["rows"])) != len(result["rows"]):
        raise ValueError("Pooled pilot evaluation contains overlapping Train row indexes")
    return result


def run_pilot() -> dict[str, Any]:
    frozen = verify_freeze()
    if CACHE_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite pilot artifacts or retry a prior run: {CACHE_DIR}")
    frame, y, train_path = load_train(None)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    real_stream_mask = ~stream_missing_mask(frame)
    sentinel_collisions = frame.loc[real_stream_mask, "signal_stream"].astype(str).map(
        lambda value: EMPTY_STREAM_SENTINEL in value.split()
    )
    if sentinel_collisions.any():
        raise RuntimeError("Official Train contains the reserved empty-stream sentinel token")
    CACHE_DIR.mkdir(parents=True, exist_ok=False)
    records: dict[int, dict[str, np.ndarray]] = {}
    model_summaries: dict[str, Any] = {}

    for fit_end in PILOT_ENDS:
        horizon = HORIZONS[fit_end]
        rows, cb_missing_probability, control_identity = _load_cb_missing(frame, y, fit_end)
        c2_rows, c2_probability, _ = _load_round1_control(frame, fit_end, horizon, "C2_B")
        np.testing.assert_array_equal(rows, c2_rows)
        outer = frame.iloc[rows]
        missing = stream_missing_mask(outer)
        present_rows = rows[~missing]
        fit = (cycles >= 47) & (cycles <= fit_end)
        fit_frame = frame.loc[fit, list(INPUT_FEATURE_NAMES)].copy()
        fit_y = y[fit]
        model = build_catboost_native_text_classifier()
        if set(model.input_feature_names) - set([*EXPANDED_FEATURES, "signal_stream"]):
            raise RuntimeError("Candidate feature list exceeds the frozen expanded/text allowlist")
        started = perf_counter()
        try:
            with threadpool_limits(limits=2):
                model.fit(fit_frame, fit_y)
                state = model.audit_state()
                present_probability = model.predict_proba(
                    frame.iloc[present_rows][model.input_feature_names].copy()
                )
        except Exception as exc:
            failure_path = CACHE_DIR / f"fit{fit_end}_failure.json"
            _save_new(failure_path, {
                "status": "FAILED_PRESERVED_NO_RETRY",
                "freeze_signature": frozen["signature"],
                "fit_end": fit_end,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Do not retry under this freeze. Preserve the failure and prepare a versioned protocol amendment before another attempt.",
            })
            raise
        elapsed = perf_counter() - started

        expected_fit_fp = _fit_fingerprint(
            fit_frame, tuple(INPUT_FEATURE_NAMES), fit_y,
        )
        expected_missing_fit_rows = int(stream_missing_mask(frame.loc[fit]).sum())
        if (state["fit_rows"] != int(fit.sum())
                or state["fit_data_sha256"] != expected_fit_fp
                or state["fit_stream_missing_rows"] != expected_missing_fit_rows):
            raise RuntimeError(f"Native-text fit boundary mismatch at cutoff {fit_end}")
        if set(("record_id", "artifact_signal", "threat_rank")).intersection(state["input_feature_names"]):
            raise RuntimeError("Candidate input list contains a forbidden/leaking field")
        if len(present_probability) != len(present_rows):
            raise RuntimeError("Candidate present-row predictions are misaligned")

        current = np.where(missing, cb_missing_probability, c2_probability)
        candidate_route = current.copy()
        candidate_route[~missing] = present_probability
        if not np.array_equal(candidate_route[missing], cb_missing_probability[missing]):
            raise RuntimeError("Candidate changed a missing-stream CB_MISSING branch probability")
        if not np.array_equal(current[~missing], c2_probability[~missing]):
            raise RuntimeError("Current champion present branch differs from exact C2_B cache")
        if not np.isfinite(present_probability).all() or ((present_probability < 0) | (present_probability > 1)).any():
            raise RuntimeError("Candidate present-row probabilities are invalid")

        cache_path = CACHE_DIR / f"fit{fit_end}.npz"
        np.savez_compressed(
            cache_path,
            rows=rows.astype(np.int64),
            present_rows=present_rows.astype(np.int64),
            present_probability=present_probability.astype(np.float64),
            candidate_route=candidate_route.astype(np.float64),
        )
        meta_path = cache_path.with_suffix(".json")
        metadata = {
            "status": "FIT_COMPLETE",
            "freeze_signature": frozen["signature"],
            "fit_end": fit_end,
            "horizon": horizon,
            "train_path_label": _relative(train_path),
            "fit_rows": int(fit.sum()),
            "present_prediction_rows": int(len(present_rows)),
            "missing_or_blank_outer_rows": int(missing.sum()),
            "state": state,
            "controls": control_identity,
            "fit_fingerprint_recomputed": expected_fit_fp,
            "missing_fit_rows_recomputed": expected_missing_fit_rows,
            "prediction_sha256": {
                "rows": array_digest(rows.astype(np.float64)),
                "present_rows": array_digest(present_rows.astype(np.float64)),
                "present_probability": array_digest(present_probability),
                "candidate_route": array_digest(candidate_route),
            },
            "npz_sha256": _sha(cache_path),
            "fit_predict_seconds": float(elapsed),
            "test_data_read": False,
            "kaggle_api_or_upload": False,
        }
        _save_new(meta_path, metadata)
        records[fit_end] = {
            "rows": rows,
            "missing": missing,
            "candidate": candidate_route,
            "current_champion": current,
        }
        model_summaries[str(fit_end)] = {
            "cache_sha256": _sha(cache_path),
            "metadata_sha256": _sha(meta_path),
            "state": state,
        }
        print(f"FITTED native-text cutoff={fit_end} fit_rows={int(fit.sum())} seconds={elapsed:.1f}", flush=True)

    w1_name, w1_low, w1_high = WINDOWS[55]
    w4_name, w4_low, w4_high = WINDOWS[67]
    w1 = _slice(records[55], cycles, w1_low, w1_high)
    w4 = _slice(records[67], cycles, w4_low, w4_high)
    h12_e = records[55]
    h12_l = records[67]
    views = {
        w1_name: _evaluate(frame, y, w1),
        w4_name: _evaluate(frame, y, w4),
        "H12_E": _evaluate(frame, y, h12_e),
        "H12_L": _evaluate(frame, y, h12_l),
        "primary_W1_W4": _evaluate(frame, y, _concat([w1, w4])),
        "H12_pooled": _evaluate(frame, y, _concat([h12_e, h12_l])),
    }
    no_view_worse = all(views[name]["delta"]["all_macro_f1"] >= -0.01
                        for name in (w1_name, w4_name, "H12_E", "H12_L"))
    primary_improved = views["primary_W1_W4"]["delta"]["all_macro_f1"] > 0.0
    h12_improved = views["H12_pooled"]["delta"]["all_macro_f1"] > 0.0
    result = {
        "status": "VERIFIED_PILOT_REVIEW_REQUIRED",
        "freeze_signature": frozen["signature"],
        "pilot_fits": [55, 67],
        "official_train_sha256": frozen["identity"]["input_files"]["official_classification_train"]["sha256"],
        "test_data_read": False,
        "kaggle_api_or_upload": False,
        "model_fits": model_summaries,
        "views": views,
        "acceptance": {
            "primary_W1_W4_improved": bool(primary_improved),
            "H12_pooled_improved": bool(h12_improved),
            "all_individual_pilot_views_within_0.01": bool(no_view_worse),
            "pilot_gate_pass": bool((primary_improved or h12_improved) and no_view_worse),
            "full_six_authorized": False,
        },
        "interpretation": "Temporal development comparison on official Train folds. H12 overlaps these development cycles and is not an untouched holdout.",
        "next_action": "Stop for Main review. Do not fit cutoffs 59, 63, 71, or 75 until explicitly authorized.",
    }
    _save_new(REPORT_DIR / "pilot_results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="Freeze exact source, Train, cache, and control identities before new fits.")
    parser.add_argument("--pilot", action="store_true", help="Run only the pre-registered cutoff 55 and 67 pilot.")
    args = parser.parse_args()
    if args.freeze == args.pilot:
        parser.error("Choose exactly one of --freeze or --pilot")
    if args.freeze:
        record = freeze_protocol()
        print(record["signature"])
    else:
        result = run_pilot()
        print("PILOT_GATE", result["acceptance"], flush=True)


if __name__ == "__main__":
    main()
