"""R14: EF1-only raw sparse-L1 specialist, frozen temporal pilot runner.

The default mode is read-only. The pilot can run only after an explicit freeze
has been saved and the official Train/cache/resource/budget preflight passes.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import optimization_v3_batch1_r03 as r03
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    evaluation_views_for_card,
    exclusive_training_slot,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    stream_missing_mask,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r14 import (
    CARD_ID,
    FEATURE_NAMES,
    MIN_EF1_FIT_ROWS,
    MIN_EF1_SCORE_ROWS,
    ef1_mask,
    fit_predict as r14_fit_predict,
)
from src.models.optimization_v3_batch1_r03 import estimate_sparse_memory
from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import _TFIDF_PARAMS
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R14.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R14"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R14"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R14"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
ATTEMPTS_PATH = REPORT_DIR / "attempts.jsonl"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
TRAINING_LOCK_PATH = RUN_DIR.parent / ".training.lock"

TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
MAX_SUCCESSFUL_FITS = 180
PILOT_FITS = 2
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
MIN_FREE_RAM_BYTES = 4 * 1024**3
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064, "H12": 38_588}
EXPECTED_CONTROL_METRICS = r03.EXPECTED_CONTROL_METRICS
EXPECTED_GATE = {
    "pilot_primary_MAE_gain_min": 0.03,
    "pilot_H12_MAE_harm_max": 0.05,
    "individual_pilot_view_MAE_harm_max": 0.5,
    "supported_group_n_min": 100,
    "supported_tail_or_missing_MAE_harm_max": 0.25,
}
PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl")


class R14RunnerError(BatchIntegrityError):
    """Raised when an R14 preflight, freeze, pilot, or verification is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R14RunnerError(f"Expected a JSON object at {path}")
    return value


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    expected = {
        "id": "R14", "task": "regression", "name": "EF1 sparse specialist",
        "family": "specialist_EF", "priority": "P2", "policy": "R_EF25_RAW",
        "pilot_predictor_fits": 2, "evaluation": "STANDARD",
        "condition": "No additional dependency", "queue_order": 32,
    }
    for name, value in expected.items():
        if card.get(name) != value:
            raise R14RunnerError(f"Canonical R14 card changed at {name}: {card.get(name)!r}")
    if (card.get("change") != (
        "Train original sparse L1 model only on parsed entry_friction==1; "
        "predict raw target scale and clip0..100. Only EF1 rows use25% raw "
        "expert+75%incumbent; other rows exact incumbent. No subgroup re-ranking."
    ) or card.get("preflight_checks") != [
        ">=500 EF1 fit rows and>=100 subgroup validation rows/view; telemetry_flag is not EF"
    ]):
        raise R14RunnerError("Canonical R14 change or support predicate changed")
    gate = queue["evaluation"]["regression_gate"]
    for key, value in EXPECTED_GATE.items():
        if float(gate.get(key, float("nan"))) != float(value):
            raise R14RunnerError(f"Canonical regression gate changed at {key}")
    if card.get("status") != "PLANNED_NOT_RUN" or card.get("implementation") != "NOT_IMPLEMENTED":
        raise R14RunnerError("R14 queue status changed; do not execute from a stale card copy")


def _read_official_train() -> tuple[Path, pd.DataFrame, np.ndarray]:
    paths = task_files("regression")
    train_path = paths.get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Regression Train is unavailable")
    train_path = Path(train_path).resolve()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise R14RunnerError("Official Regression Train hash differs from the canonical card")
    columns = ["record_id", *FEATURE_NAMES]
    frame = pd.read_csv(train_path, usecols=columns)
    if len(frame) != 50_608:
        raise R14RunnerError(f"Expected 50,608 Train rows, found {len(frame)}")
    if set(frame.columns) != {"record_id", *FEATURE_NAMES} or frame.columns.has_duplicates:
        raise R14RunnerError("Train schema differs from the R14 feature allowlist")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise R14RunnerError("Official Train record_id contains missing/blank values")
    if frame["record_id"].astype(str).duplicated().any():
        raise R14RunnerError("Official Train record_id is not unique")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise R14RunnerError("Official Train cycles differ from the frozen 47–79 range")
    if set(FEATURE_NAMES) & {"record_id", "threat_rank", "artifact_signal", "pulse_a", "pulse_b",
                            "audit_lane", "checksum_hint"}:
        raise R14RunnerError("R14 feature allowlist contains a protected/identity field")
    return train_path, frame, cycles


def _support_snapshot(frame: pd.DataFrame, cycles: np.ndarray,
                      views: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, Any]:
    ef = ef1_mask(frame["entry_friction"])
    parsed_ef = pd.to_numeric(frame["entry_friction"], errors="coerce")
    output: dict[str, Any] = {
        "definition": "EF1 is parsed entry_friction == 1; telemetry_flag is not used",
        "entry_friction_parse_missing_rows": int(parsed_ef.isna().sum()),
        "fit": {}, "views": {}, "score_cycle_counts": {},
        "minimum_fit_rows": MIN_EF1_FIT_ROWS,
        "minimum_validation_rows_per_view": MIN_EF1_SCORE_ROWS,
    }
    for cutoff in (55, 67):
        fit_mask = cycles <= cutoff
        output["fit"][str(cutoff)] = {
            "outer_fit_rows": int(fit_mask.sum()),
            "EF1_fit_rows": int((fit_mask & ef).sum()),
            "minimum_pass": bool(int((fit_mask & ef).sum()) >= MIN_EF1_FIT_ROWS),
        }
        score_mask = (cycles > cutoff) & (cycles <= cutoff + 12)
        output["score_cycle_counts"][str(cutoff)] = {
            str(int(cycle)): {"rows": int((score_mask & (cycles == cycle)).sum()),
                              "EF1_rows": int((score_mask & (cycles == cycle) & ef).sum())}
            for cycle in np.unique(cycles[score_mask])
        }
    for name, segments in views.items():
        rows: list[np.ndarray] = []
        for cutoff, low, high in segments:
            mask = (cycles >= int(low)) & (cycles <= int(high))
            if np.any(mask & (cycles <= int(cutoff))):
                raise R14RunnerError(f"R14 view {name} overlaps its fit period")
            rows.append(np.flatnonzero(mask))
        combined = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        if len(combined) != len(np.unique(combined)):
            raise R14RunnerError(f"R14 view {name} has overlapping score rows")
        n_ef1 = int(ef[combined].sum())
        output["views"][name] = {
            "rows": int(len(combined)), "EF1_rows": n_ef1,
            "minimum_pass": bool(n_ef1 >= MIN_EF1_SCORE_ROWS),
        }
    for name, segments in {
        "H12_E": [(55, 56, 67)], "H12_L": [(67, 68, 79)],
    }.items():
        rows = np.flatnonzero(np.isin(cycles, np.concatenate([
            np.arange(int(low), int(high) + 1) for _cutoff, low, high in segments
        ])))
        n_ef1 = int(ef[rows].sum())
        output["views"][name] = {"rows": int(len(rows)), "EF1_rows": n_ef1,
                                 "minimum_pass": bool(n_ef1 >= MIN_EF1_SCORE_ROWS)}
    return output


def _budget_snapshot() -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        return {"status": "BLOCKED_LEDGER", "path": str(LEDGER_PATH),
                "reason": "Active runtime budget ledger is missing"}
    reservations: dict[str, tuple[int, str | None]] = {}
    try:
        for line_number, raw in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            event = json.loads(raw)
            if event.get("event") == "BUDGET_RESERVED":
                key = str(event["budget_key"])
                count = int(event["predictor_fits"])
                if count < 0:
                    raise ValueError(f"negative fit reservation on line {line_number}")
                card = str(event["card_id"]) if event.get("card_id") is not None else None
                item = (count, card)
                if key in reservations:
                    if reservations[key] != item:
                        raise ValueError(f"conflicting reservation identity for {key}")
                    raise ValueError(f"duplicate budget reservation for {key}")
                reservations[key] = item
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED_LEDGER", "path": str(LEDGER_PATH), "error": str(exc)}
    total = sum(count for count, _card in reservations.values())
    own = reservations.get("R14:PILOT")
    if own is not None and own != (PILOT_FITS, CARD_ID):
        return {"status": "BLOCKED_BUDGET", "path": str(LEDGER_PATH),
                "reserved_total": total,
                "existing_R14_reservation": {"predictor_fits": own[0], "card_id": own[1]}}
    next_total = total if own is not None else total + PILOT_FITS
    return {"status": "PASS" if next_total <= MAX_SUCCESSFUL_FITS else "BLOCKED_BUDGET",
            "path": str(LEDGER_PATH), "reserved_total": total,
            "card_reservation_present": own is not None,
            "existing_R14_reservation": own[0] if own else None,
            "existing_R14_card_id": own[1] if own else None,
            "next_total": next_total, "cap": MAX_SUCCESSFUL_FITS,
            "card_predictor_fits": PILOT_FITS}


def _resource_snapshot(frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        fit_x = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
        subset = fit_x.loc[ef1_mask(fit_x["entry_friction"])].copy()
        estimate = estimate_sparse_memory(subset)
        free = estimate.get("available_physical_bytes")
        estimate["minimum_free_ram_bytes"] = MIN_FREE_RAM_BYTES
        estimate["job_cap_bytes"] = JOB_CAP_BYTES
        estimate["free_ram_reserve_bytes"] = FREE_RAM_RESERVE_BYTES
        estimate["free_ram_minimum_pass"] = bool(free is not None and int(free) >= MIN_FREE_RAM_BYTES)
        estimate["status"] = "PASS" if (
            estimate.get("status") == "PASS" and estimate["free_ram_minimum_pass"]
            and int(estimate.get("estimated_peak_bytes", JOB_CAP_BYTES + 1)) <= JOB_CAP_BYTES
        ) else ("UNKNOWN_RESOURCE" if free is None else "BLOCKED_RESOURCE")
        estimate["fit_population"] = "parsed entry_friction == 1"
        result[str(cutoff)] = estimate
    return result


def _control_and_lineage(frame: pd.DataFrame, cycles: np.ndarray,
                         views: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    lineage = r03._validate_manifest_and_historical_freeze()
    controls = r03._load_control_arrays(frame, cycles, views)
    for cutoff, expected in ((55, tuple(range(56, 68))), (67, tuple(range(68, 80)))):
        found = tuple(sorted(np.unique(controls[cutoff]["score_cycle"]).tolist()))
        if found != expected:
            raise R14RunnerError(f"Frozen control score cycles differ at cutoff {cutoff}")
        if len(controls[cutoff]["score_rows"]) != int(np.sum(cycles <= cutoff + 12) - np.sum(cycles <= cutoff)):
            raise R14RunnerError(f"Frozen control row count differs at cutoff {cutoff}")
    return lineage, controls


def _recipe(card: Mapping[str, Any], views: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    tfidf = dict(_TFIDF_PARAMS)
    tfidf["ngram_range"] = list(tfidf["ngram_range"])
    tfidf["dtype"] = "float64"
    return _safe_json({
        "card_id": CARD_ID,
        "queue_recipe": dict(card),
        "feature_allowlist": list(FEATURE_NAMES),
        "feature_roles": {"record_id": "identity_only", "threat_rank": "fit_target_only"},
        "fit_population": "outer cycle fit rows AND parsed entry_friction == 1",
        "minimum_support": {"EF1_fit_rows_per_cutoff": MIN_EF1_FIT_ROWS,
                            "EF1_validation_rows_per_view": MIN_EF1_SCORE_ROWS},
        "model": {
            "class": "src.models.optimization_v2_regression.SparseHybridRegressor",
            "recipe": "lgbm_sparse_l1",
            "estimator": "lightgbm.LGBMRegressor",
            "parameters": dict(SPARSE_LGB_PARAMETERS),
            "objective": "regression_l1",
            "target_transform": "identity threat_rank in [0,100]",
            "early_stopping": False,
            "tfidf": tfidf,
            "preprocessing": "structured parser/imputer/scaler/OHE and TF-IDF fit only on the EF1 outer-fit rows",
            "design": "CSR sparse matrix; no densification; missing-stream indicator retained",
        },
        "prediction_policy": {
            "card_wording_interpretation": "Clip the raw target-scale expert prediction to [0,100] before blending; clip the final blend again as a range guard.",
            "EF1": "clip(0.75*incumbent + 0.25*clip(raw_EF1_expert,0,100),0,100)",
            "other_rows": "incumbent copied exactly",
            "subgroup_rank_or_calibration": "none",
            "incumbent": "accepted v2_reg_sparse_lgb25, reconstructed row-exactly from frozen caches",
            "fit_predictor_fits": PILOT_FITS,
        },
        "evaluation": {"profile": "STANDARD", "views": views,
                       "fit_rows": EXPECTED_FIT_ROWS,
                       "view_rows": EXPECTED_VIEW_ROWS,
                       "cutoffs": [55, 67], "horizon_cycles": 12},
        "support_snapshot": support,
        "control_metrics": EXPECTED_CONTROL_METRICS,
        "gate": EXPECTED_GATE,
        "resource_policy": {"threads": THREADS, "concurrent_training_jobs": 1,
                            "minimum_free_RAM_bytes": MIN_FREE_RAM_BYTES,
                            "free_RAM_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                            "job_memory_cap_bytes": JOB_CAP_BYTES,
                            "resource_estimate_is_observed": False},
        "test_or_sample_used": False,
        "kaggle_used": False,
    })


def _maps(train_path: Path) -> tuple[dict[str, Path], list[Path]]:
    inputs = {
        "official_regression_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_R14_card": CARD_PATH,
    }
    for name, path in r03.INPUTS.items():
        if name == "official_regression_train":
            continue
        inputs[f"incumbent_{name}"] = Path(path)
    sources = list(r03.SOURCES) + [
        Path(__file__), ROOT / "src/models/optimization_v3_batch1_r14.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v2_regression.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_research_tabular.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/utils/paths.py",
    ]
    unique_sources = list(dict.fromkeys(Path(path) for path in sources))
    return inputs, unique_sources


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    train_path, frame, cycles = _read_official_train()
    views = evaluation_views_for_card(queue, CARD_ID)
    expected_views = {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    if views != expected_views:
        raise R14RunnerError(f"Canonical R14 STANDARD views changed: {views}")
    for name, expected_n in EXPECTED_VIEW_ROWS.items():
        rows = sum(int(np.sum((cycles >= low) & (cycles <= high)))
                   for _cutoff, low, high in views[name])
        if rows != expected_n:
            raise R14RunnerError(f"R14 {name} denominator changed: {rows} != {expected_n}")
    support = _support_snapshot(frame, cycles, views)
    support_pass = all(row["minimum_pass"] for row in support["fit"].values()) and all(
        row["minimum_pass"] for row in support["views"].values()
    )
    lineage, controls = _control_and_lineage(frame, cycles, views)
    # Control metric computation is deliberately deferred until after the freeze;
    # preflight checks the frozen lineage and row-aligned components only.
    resources = _resource_snapshot(frame, cycles)
    budget = _budget_snapshot()
    try:
        with exclusive_training_slot(TRAINING_LOCK_PATH):
            lock_status = "PASS"
    except BatchIntegrityError as exc:
        lock_status = "BLOCKED_TRAINING_SLOT"
        lock_error = str(exc)
    else:
        lock_error = None

    input_files, source_files = _maps(train_path)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=source_files, evaluation_views=views, packages=PACKAGE_NAMES,
    )
    recipe = _recipe(card, views, support)
    identity.update({
        "recipe": recipe,
        "recipe_sha256": json_sha256(recipe),
        "control_lineage": lineage,
        "support_snapshot": support,
        "resource_policy": recipe["resource_policy"],
        "expected_outer_fit_rows": EXPECTED_FIT_ROWS,
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "control_row_id_sha256": {
            str(cutoff): r03._hash_ordered_ids(values["score_record_ids"])
            for cutoff, values in controls.items()
        },
    })
    signature = json_sha256(identity)
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise R14RunnerError("Existing R14 recipe differs from the canonical preflight recipe")
    if FREEZE_PATH.exists():
        frozen = _read_json(FREEZE_PATH)
        if frozen.get("identity") != _safe_json(identity) or frozen.get("signature") != signature:
            raise R14RunnerError("Existing R14 freeze differs from current source/data/recipe identity")

    resource_pass = all(value.get("status") == "PASS" for value in resources.values())
    status = "PREFLIGHT_PASS"
    if not support_pass:
        status = "SKIPPED_SUPPORT"
    elif not resource_pass:
        status = "BLOCKED_RESOURCE"
    elif budget.get("status") != "PASS":
        status = str(budget.get("status"))
    elif lock_status != "PASS":
        status = lock_status
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "official Regression Train/support/cache/resource/budget preflight; no candidate fit, prediction, label metric, Test, SampleSubmission, or Kaggle access",
        "train_path": str(train_path),
        "official_train_sha256": file_sha256(train_path),
        "input_sha256": {name: file_sha256(path) for name, path in sorted(input_files.items())},
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files},
        "freeze_identity_sha256": signature,
        "recipe": recipe,
        "control_lineage": lineage,
        "support": support,
        "memory_estimate": resources,
        "budget_preflight": budget,
        "training_slot_preflight": {"status": lock_status, "path": str(TRAINING_LOCK_PATH),
                                    "error": lock_error, "max_concurrent_jobs": 1},
        "candidate_prediction_or_metric_computed": False,
        "test_sample_kaggle_access": False,
        "freeze_exists": FREEZE_PATH.is_file(),
        "preflight_report_is_read_only": True,
        "python_version": platform.python_version(),
    }
    return {"public": public, "identity": identity, "recipe": recipe,
            "frame": frame, "cycles": cycles, "views": views,
            "controls": controls, "train_path": train_path,
            "artifact_dir": RUN_DIR,
            "support_pass": support_pass, "resources_pass": resource_pass,
            "input_files": input_files, "source_files": source_files}


def _write_or_require_same(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != _safe_json(dict(value)):
            raise R14RunnerError(f"Existing frozen artifact differs: {path}")
    else:
        write_json_exclusive(path, _safe_json(dict(value)))


def _control_loader(context: Mapping[str, Any]):
    controls = context["controls"]

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        saved = controls.get(int(cutoff))
        if saved is None or not np.array_equal(saved["score_rows"], score_rows):
            raise R14RunnerError(f"R14 incumbent cache rows differ at cutoff {cutoff}")
        return {
            "baseline_prediction": saved["baseline_prediction"],
            "ridge25_prediction": saved["ridge25_prediction"],
            "old_sparse_raw": saved["old_sparse_raw"],
            "old_sparse_rank50": saved["old_sparse_rank50"],
            "control_rows": saved["score_rows"],
        }
    return load


def _resource_guarded_adapter(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    fit_x = args[0]
    fit_ef = fit_x.loc[ef1_mask(fit_x["entry_friction"])].copy()
    resource = estimate_sparse_memory(fit_ef)
    free = resource.get("available_physical_bytes")
    if (resource.get("status") != "PASS" or free is None
            or int(free) < MIN_FREE_RAM_BYTES
            or int(resource.get("estimated_peak_bytes", JOB_CAP_BYTES + 1)) > JOB_CAP_BYTES):
        raise R14RunnerError("R14 per-fit resource guard failed before model fit")
    result = dict(r14_fit_predict(*args, **kwargs))
    result.setdefault("fit_metadata", {})["resource_preflight"] = {
        **resource,
        "status": "PASS",
        "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES,
        "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "concurrent_training_jobs": 1,
        "estimate_is_observed": False,
    }
    return result


def _independent_rank50(raw: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    result = np.empty(len(raw), dtype=np.float64)
    for value in np.unique(cycle):
        mask = cycle == value
        result[mask] = 0.5 * raw[mask] + 50.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return np.clip(result, 0.0, 100.0)


def _fingerprint_value(value: Any) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "bool:" + ("1" if value else "0")
    if isinstance(value, numbers.Integral):
        return f"int:{int(value)}"
    if isinstance(value, numbers.Real):
        return "float:" + float(value).hex()
    return "str:" + str(value)


def _fit_fingerprint_independent(X: pd.DataFrame, y: np.ndarray) -> str:
    digest = hashlib.sha256()

    def add(text: str) -> None:
        encoded = text.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add(json.dumps(FEATURE_NAMES, ensure_ascii=False, separators=(",", ":")))
    add(str(len(X)))
    for column in FEATURE_NAMES:
        for value in X[column].to_numpy(dtype=object):
            add(_fingerprint_value(value))
    for value in y:
        add(_fingerprint_value(float(value)))
    return digest.hexdigest()


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    if y.ndim != 1 or prediction.shape != y.shape or not len(y):
        raise R14RunnerError("Verifier MAE inputs are empty or misaligned")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise R14RunnerError("Verifier MAE inputs are nonfinite")
    return float(np.abs(y - prediction).mean(dtype=np.float64))


def _view_rows(frame: pd.DataFrame, cycles: np.ndarray,
               per_cutoff: Mapping[int, Mapping[str, Any]],
               segments: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    for cutoff, low, high in segments:
        result = per_cutoff.get(int(cutoff))
        if result is None:
            raise R14RunnerError(f"Missing pilot bundle for view origin {cutoff}")
        score_rows = result["score_rows"]
        mask = (cycles[score_rows] >= int(low)) & (cycles[score_rows] <= int(high))
        rows_parts.append(score_rows[mask])
        pred_parts.append(result["prediction"][mask])
        base_parts.append(result["baseline_prediction"][mask])
    rows = np.concatenate(rows_parts)
    if len(rows) != len(np.unique(rows)):
        raise R14RunnerError("R14 evaluation view contains duplicate Train rows")
    return rows, np.concatenate(pred_parts), np.concatenate(base_parts)


def _verify_pilot(context: Mapping[str, Any], report: Mapping[str, Any],
                  target: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    if (report.get("card_id") != CARD_ID
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"
            or report.get("fit_predictor_count") != PILOT_FITS
            or report.get("cutoffs") != [55, 67]
            or report.get("test_data_used") is not False):
        raise R14RunnerError("R14 generic pilot report identity, fit count, or boundary is invalid")
    signature = json_sha256(context["identity"])
    if report.get("freeze_signature") != signature:
        raise R14RunnerError("R14 pilot report does not match the frozen identity")
    frame = context["frame"]
    cycles = context["cycles"]
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise R14RunnerError("Official regression target is invalid or misaligned")
    per_cutoff: dict[int, dict[str, np.ndarray]] = {}

    for cutoff in (55, 67):
        controls = context["controls"][cutoff]
        score_rows = np.asarray(controls["score_rows"], dtype=np.int64)
        bundle_path = Path(context.get("artifact_dir", RUN_DIR)) / f"R14_fit{cutoff}.npz"
        bundle_identity = {"freeze_signature": signature, "card_id": CARD_ID,
                           "cutoff": cutoff, "score_rows": score_rows.tolist()}
        arrays, sidecar = read_bundle(bundle_path, bundle_identity)
        if (not np.array_equal(arrays.get("score_rows"), score_rows)
                or not np.array_equal(arrays.get("score_record_ids"), ids[score_rows])
                or len(np.unique(score_rows)) != len(score_rows)):
            raise R14RunnerError(f"R14 saved score row positions/IDs differ at cutoff {cutoff}")
        metadata = sidecar["metadata"]
        fit_rows = np.flatnonzero(cycles <= cutoff)
        fit_mask = ef1_mask(frame.iloc[fit_rows]["entry_friction"])
        ef_fit_rows = fit_rows[fit_mask]
        score_mask = ef1_mask(frame.iloc[score_rows]["entry_friction"])
        raw = np.asarray(arrays.get("raw_candidate"), dtype=np.float64)
        prediction = np.asarray(arrays.get("prediction"), dtype=np.float64)
        baseline = np.asarray(arrays.get("baseline_prediction"), dtype=np.float64)
        saved_mask = np.asarray(arrays.get("audit__expert_applied_mask"), dtype=np.uint8).astype(bool)
        if any(values.shape != score_rows.shape for values in (raw, prediction, baseline, saved_mask)):
            raise R14RunnerError(f"R14 bundle arrays are misaligned at cutoff {cutoff}")
        if not np.array_equal(saved_mask, score_mask) or np.any(raw[~score_mask] != 0.0):
            raise R14RunnerError(f"R14 expert application mask/sentinel differs at cutoff {cutoff}")
        if not np.isfinite(raw).all() or not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
            raise R14RunnerError(f"R14 saved predictions are nonfinite at cutoff {cutoff}")
        if np.any((prediction < 0.0) | (prediction > 100.0)) or np.any((baseline < 0.0) | (baseline > 100.0)):
            raise R14RunnerError(f"R14 saved predictions fall outside [0,100] at cutoff {cutoff}")

        state = metadata.get("state", {})
        fit_meta = metadata.get("fit_metadata", {})
        if (int(metadata.get("fit_predictor_count", -1)) != 1
                or int(metadata.get("fit_rows", -1)) != len(fit_rows)
                or int(metadata.get("score_rows", -1)) != len(score_rows)
                or int(fit_meta.get("ef1_fit_rows", -1)) != len(ef_fit_rows)
                or int(state.get("fit_rows", -1)) != len(ef_fit_rows)
                or fit_meta.get("expert_fit_population") != "parsed entry_friction == 1"
                or fit_meta.get("score_labels_passed") is not False):
            raise R14RunnerError(f"R14 fit receipt does not prove EF1-only fit at cutoff {cutoff}")
        expected_fingerprint = _fit_fingerprint_independent(
            frame.iloc[ef_fit_rows].loc[:, list(FEATURE_NAMES)], target[ef_fit_rows]
        )
        target_sha = hashlib.sha256(np.asarray(target[ef_fit_rows], dtype="<f8").tobytes(order="C")).hexdigest()
        if state.get("fit_data_sha256") != expected_fingerprint or state.get("target_sha256") != target_sha:
            raise R14RunnerError(f"R14 saved model was not fit on the exact EF1 rows at cutoff {cutoff}")

        ridge = np.asarray(arrays.get("control__ridge25_prediction"), dtype=np.float64)
        sparse_raw = np.asarray(arrays.get("control__old_sparse_raw"), dtype=np.float64)
        saved_rank = np.asarray(arrays.get("control__old_sparse_rank50"), dtype=np.float64)
        control_rows = np.asarray(arrays.get("control__control_rows"), dtype=np.int64)
        score_cycles = cycles[score_rows]
        if (not np.array_equal(control_rows, score_rows)
                or ridge.shape != score_rows.shape or sparse_raw.shape != score_rows.shape
                or saved_rank.shape != score_rows.shape):
            raise R14RunnerError(f"R14 incumbent cache alignment differs at cutoff {cutoff}")
        if not np.array_equal(saved_rank, _independent_rank50(sparse_raw, score_cycles)):
            raise R14RunnerError(f"R14 old sparse rank50 reconstruction differs at cutoff {cutoff}")
        expected_baseline = np.clip(0.75 * ridge + 0.25 * saved_rank, 0.0, 100.0)
        if not np.allclose(baseline, expected_baseline, rtol=0.0, atol=1e-10):
            raise R14RunnerError(f"R14 incumbent route changed at cutoff {cutoff}")
        expected_prediction = baseline.copy()
        expected_prediction[score_mask] = (
            0.75 * baseline[score_mask] + 0.25 * np.clip(raw[score_mask], 0.0, 100.0)
        )
        expected_prediction = np.clip(expected_prediction, 0.0, 100.0)
        if not np.allclose(prediction, expected_prediction, rtol=0.0, atol=1e-10):
            raise R14RunnerError(f"R14 fixed EF1 blend does not reconstruct at cutoff {cutoff}")
        if not np.array_equal(prediction[~score_mask], baseline[~score_mask]):
            raise R14RunnerError(f"R14 changed non-EF1 incumbent rows at cutoff {cutoff}")
        if len(np.unique(score_cycles)) != 12 or not np.array_equal(
            np.unique(score_cycles), np.arange(cutoff + 1, cutoff + 13)
        ):
            raise R14RunnerError(f"R14 score horizon is incomplete at cutoff {cutoff}")
        per_cutoff[cutoff] = {"score_rows": score_rows, "prediction": prediction,
                              "baseline_prediction": baseline}

    view_segments = dict(context["views"])
    view_segments["H12_E"] = [(55, 56, 67)]
    view_segments["H12_L"] = [(67, 68, 79)]
    view_metrics: dict[str, Any] = {}
    view_data: dict[str, dict[str, np.ndarray]] = {}
    for name, segments in view_segments.items():
        rows, pred, base = _view_rows(frame, cycles, per_cutoff, segments)
        observed = {"rows": int(len(rows)), "candidate_mae": _mae(target[rows], pred),
                    "baseline_mae": _mae(target[rows], base)}
        observed["gain_positive_is_better"] = observed["baseline_mae"] - observed["candidate_mae"]
        if name in report.get("views", {}):
            reported = report["views"][name]
            if (int(reported.get("rows", -1)) != len(rows)
                    or abs(float(reported["candidate"]["mae"]) - observed["candidate_mae"]) > 1e-10
                    or abs(float(reported["baseline"]["mae"]) - observed["baseline_mae"]) > 1e-10):
                raise R14RunnerError(f"R14 generic aggregation differs from independent {name} metrics")
        view_metrics[name] = observed
        view_data[name] = {"rows": rows, "prediction": pred, "baseline": base}
    if (view_metrics["primary"]["rows"] != EXPECTED_VIEW_ROWS["primary"]
            or view_metrics["H12"]["rows"] != EXPECTED_VIEW_ROWS["H12"]
            or abs(view_metrics["primary"]["baseline_mae"] - EXPECTED_CONTROL_METRICS["primary_mae"]) > 1e-9
            or abs(view_metrics["H12"]["baseline_mae"] - EXPECTED_CONTROL_METRICS["H12_mae"]) > 1e-9):
        raise R14RunnerError("R14 row-exact incumbent metrics differ from the accepted V2 control")

    group_metrics: dict[str, Any] = {}
    ef = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy()
    missing_stream = stream_missing_mask(frame["signal_stream"])
    for view_name in ("primary", "H12_E", "H12_L"):
        data = view_data[view_name]
        rows = data["rows"]
        masks = {
            "EF0": ef[rows] == 0,
            "EF1": ef[rows] == 1,
            "tail_low_le5": target[rows] <= 5,
            "tail_high_ge95": target[rows] >= 95,
            "stream_missing": missing_stream[rows],
        }
        group_metrics[view_name] = {}
        for group_name, mask in masks.items():
            n = int(mask.sum())
            supported = n >= int(EXPECTED_GATE["supported_group_n_min"])
            row = {"n": n, "supported": supported}
            if supported:
                row["candidate_mae"] = _mae(target[rows][mask], data["prediction"][mask])
                row["baseline_mae"] = _mae(target[rows][mask], data["baseline"][mask])
                row["harm_candidate_minus_baseline"] = row["candidate_mae"] - row["baseline_mae"]
            else:
                row.update({"candidate_mae": None, "baseline_mae": None,
                            "harm_candidate_minus_baseline": None})
            group_metrics[view_name][group_name] = row

    individual_harms = {
        name: view_metrics[name]["candidate_mae"] - view_metrics[name]["baseline_mae"]
        for name in ("W1", "W4")
    }
    guarded_harms = [
        values[group]["harm_candidate_minus_baseline"]
        for view in ("primary", "H12_E", "H12_L")
        for group, values in group_metrics[view].items()
        if group in {"tail_low_le5", "tail_high_ge95", "stream_missing"}
        and values["supported"]
    ]
    max_guarded_harm = max(guarded_harms) if guarded_harms else None
    primary_gain = view_metrics["primary"]["gain_positive_is_better"]
    h12_harm = view_metrics["H12"]["candidate_mae"] - view_metrics["H12"]["baseline_mae"]
    gate_checks = {
        "primary_gain_at_least_0_03": primary_gain >= EXPECTED_GATE["pilot_primary_MAE_gain_min"],
        "H12_harm_at_most_0_05": h12_harm <= EXPECTED_GATE["pilot_H12_MAE_harm_max"],
        "W1_W4_harm_at_most_0_5": all(
            value <= EXPECTED_GATE["individual_pilot_view_MAE_harm_max"]
            for value in individual_harms.values()
        ),
        "supported_tail_or_missing_harm_at_most_0_25": (
            max_guarded_harm is not None
            and max_guarded_harm <= EXPECTED_GATE["supported_tail_or_missing_MAE_harm_max"]
        ),
    }
    decision = "ADVANCE" if all(gate_checks.values()) else "RETIRE"
    verification = {
        "card_id": CARD_ID, "status": "INDEPENDENT_QA_PASS", "decision": decision,
        "freeze_signature": signature, "fit_predictor_count": PILOT_FITS,
        "cutoffs": [55, 67], "views": view_metrics,
        "individual_pilot_view_harms": individual_harms,
        "supported_subgroup_metrics": group_metrics,
        "maximum_supported_tail_or_missing_harm": max_guarded_harm,
        "gate_checks": gate_checks, "gate_definition": EXPECTED_GATE,
        "control_reconstruction": "0.75*cached Ridge25 + 0.25*independently recomputed rank50(cached sparse raw), clipped [0,100]",
        "route_reconstruction": "EF1 only: 0.75*incumbent + 0.25*clipped raw expert; all others array-equal incumbent",
        "expert_clipping_order": "Raw expert clipped to [0,100] before the 25% blend; final blend clipped again.",
        "fit_population_independently_verified": True,
        "target_rows_used_for_fit": "outer temporal fit rows with parsed entry_friction == 1 only",
        "test_data_used": False, "sample_submission_used": False, "kaggle_used": False,
    }
    gate = {
        "card_id": CARD_ID, "status": decision,
        "freeze_signature": signature, "thresholds": EXPECTED_GATE,
        "observed": {"primary_gain": primary_gain, "H12_harm": h12_harm,
                     "W1_harm": individual_harms["W1"], "W4_harm": individual_harms["W4"],
                     "max_supported_tail_or_missing_harm": max_guarded_harm},
        "checks": gate_checks,
        "decision_basis": "Frozen row-exact temporal pilot and independent verification only; no Test/Kaggle evidence.",
    }
    return verification, gate


def _fit_accounting() -> dict[str, Any]:
    if not ATTEMPTS_PATH.is_file():
        raise R14RunnerError("R14 attempt log is missing")
    events = [json.loads(line) for line in ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if any(event.get("card_id") != CARD_ID for event in events):
        raise R14RunnerError("R14 attempt log contains events for another card")
    by_cutoff: dict[str, Any] = {}
    total_saved = 0
    total_failed = 0
    for cutoff in (55, 67):
        subset = [event for event in events if int(event.get("cutoff", -1)) == cutoff]
        started = sum(event.get("event") == "FIT_STARTED" for event in subset)
        saved = sum(event.get("event") == "FIT_SAVED" for event in subset)
        failed = sum(event.get("event") == "FIT_FAILED" for event in subset)
        resumed = sum(event.get("event") == "RESUMED_VERIFIED" for event in subset)
        if started < 1 or saved > 1 or saved + resumed < 1 or failed > started:
            raise R14RunnerError(f"R14 attempts do not reconcile at cutoff {cutoff}")
        # A FIT_SAVED bundle may be read and logged as RESUMED_VERIFIED more
        # than once after a crash. It remains one successful predictor fit.
        total_saved += 1
        total_failed += failed
        path = RUN_DIR / f"R14_fit{cutoff}.npz"
        arrays, sidecar = read_bundle(path, {
            "freeze_signature": json_sha256(_read_json(FREEZE_PATH)["identity"]),
            "card_id": CARD_ID, "cutoff": cutoff,
            "score_rows": context_score_rows_from_sidecar(path),
        })
        metadata = sidecar["metadata"]
        resource = metadata.get("fit_metadata", {}).get("resource_preflight", {})
        if int(metadata.get("fit_predictor_count", -1)) != 1 or resource.get("status") != "PASS":
            raise R14RunnerError(f"R14 saved fit count/resource receipt is invalid at cutoff {cutoff}")
        by_cutoff[str(cutoff)] = {
            "fit_attempt_count": int(started), "fit_saved_event_count": int(saved),
            "resume_verified_event_count": int(resumed), "successful_bundle_count": 1,
            "failed_attempt_count": int(failed),
            "fit_predictor_count": int(metadata.get("fit_predictor_count", -1)),
            "outer_fit_rows": int(metadata.get("fit_rows", -1)),
            "ef1_fit_rows": int(metadata.get("fit_metadata", {}).get("ef1_fit_rows", -1)),
            "score_rows": int(metadata.get("score_rows", -1)),
            "fit_seconds": float(metadata.get("fit_seconds", 0.0)),
            "resource_preflight_status": metadata.get("fit_metadata", {}).get("resource_preflight", {}).get("status"),
        }
    if total_saved != PILOT_FITS:
        raise R14RunnerError("R14 successful fit accounting differs from declared pilot")
    budget = _budget_snapshot()
    if (budget.get("status") != "PASS" or budget.get("card_reservation_present") is not True
            or budget.get("existing_R14_card_id") != CARD_ID):
        raise R14RunnerError("R14 completed fits do not reconcile to the matching global budget reservation")
    return {"budget_reserved_predictor_fits": PILOT_FITS,
            "verified_successful_predictor_fits": total_saved,
            "failed_attempts": total_failed, "global_budget": budget, "cutoffs": by_cutoff,
            "attempt_log": "reports/optimization_v3/batch1/R14/attempts.jsonl"}


def context_score_rows_from_sidecar(path: Path) -> list[int]:
    """Read only bundle identity metadata to avoid trusting arrays for identity."""
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise R14RunnerError(f"R14 fit sidecar is missing: {sidecar_path}")
    sidecar = _read_json(sidecar_path)
    identity = sidecar.get("identity", {})
    rows = identity.get("score_rows")
    if not isinstance(rows, list) or not rows:
        raise R14RunnerError("R14 fit sidecar has no score row identity")
    return [int(value) for value in rows]


def _publish(context: Mapping[str, Any], report: Mapping[str, Any],
             verification: Mapping[str, Any], gate: Mapping[str, Any],
             accounting: Mapping[str, Any]) -> None:
    results = {
        "card_id": CARD_ID,
        "status": "PILOT_COMPLETE_" + str(gate["status"]),
        "preflight": _read_json(PREFLIGHT_PATH),
        "engine_report": report,
        "fit_accounting": accounting,
        "verification": verification,
        "gate": gate,
        "resource_note": "Memory values are conservative estimates, not observed peak RSS.",
        "boundary": {"official_train_only": True, "test_sample_kaggle_used": False},
    }
    _write_or_require_same(RESULTS_PATH, results)
    _write_or_require_same(VERIFICATION_PATH, verification)
    _write_or_require_same(GATE_PATH, gate)
    summary = {
        "card_id": CARD_ID, "decision": gate["status"],
        "fit_predictor_count": accounting["verified_successful_predictor_fits"],
        "cutoffs": [55, 67],
        "primary_mae_gain": verification["views"]["primary"]["gain_positive_is_better"],
        "H12_mae_harm": (verification["views"]["H12"]["candidate_mae"]
                         - verification["views"]["H12"]["baseline_mae"]),
        "W1_harm": verification["individual_pilot_view_harms"]["W1"],
        "W4_harm": verification["individual_pilot_view_harms"]["W4"],
        "test_sample_kaggle_used": False,
        "details": ["results.json", "verification.json", "gate.json"],
    }
    _write_or_require_same(SUMMARY_PATH, summary)


def _reconcile_budget(verification: Mapping[str, Any]) -> None:
    paths = {
        "attempts": ATTEMPTS_PATH, "engine_report": ENGINE_REPORT_PATH,
        "fit55_npz": RUN_DIR / "R14_fit55.npz", "fit55_sidecar": RUN_DIR / "R14_fit55.json",
        "fit67_npz": RUN_DIR / "R14_fit67.npz", "fit67_sidecar": RUN_DIR / "R14_fit67.json",
        "recipe": RECIPE_PATH, "freeze": FREEZE_PATH, "preflight": PREFLIGHT_PATH,
        "results": RESULTS_PATH, "verification": VERIFICATION_PATH,
        "gate": GATE_PATH, "summary": SUMMARY_PATH,
    }
    evidence = {name: {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
                for name, path in paths.items()}
    event = {
        "event": "CARD_FIT_BUDGET_RECONCILED",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "card_id": CARD_ID, "fit_cutoffs": [55, 67],
        "predictor_fits_verified": int(verification["fit_predictor_count"]),
        "verification_status": verification["status"], "evidence": evidence,
    }
    if LEDGER_PATH.is_file():
        for raw in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            prior = json.loads(raw)
            if prior.get("event") == "CARD_FIT_BUDGET_RECONCILED" and prior.get("card_id") == CARD_ID:
                expected_prior = dict(event)
                expected_prior["at_utc"] = prior.get("at_utc")
                if prior != expected_prior:
                    raise R14RunnerError("Existing R14 budget reconciliation differs from current evidence")
                return
    append_event(LEDGER_PATH, event)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False, verify_only: bool = False) -> dict[str, Any]:
    modes = (preflight_only, freeze_only, run_frozen_pilot, verify_only)
    if sum(modes) != 1:
        raise R14RunnerError("Choose exactly one R14 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise R14RunnerError(f"R14 is blocked by {context['public']['status']}")
    if freeze_only:
        _write_or_require_same(RECIPE_PATH, context["recipe"])
        ensure_frozen(FREEZE_PATH, context["identity"])
        preflight_report = dict(context["public"])
        preflight_report["status"] = "PREFLIGHT_PASS"
        _write_or_require_same(PREFLIGHT_PATH, preflight_report)
        logged = ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines() if ATTEMPTS_PATH.exists() else []
        if not any(json.loads(line).get("event") == "PREFLIGHT_PASS" for line in logged if line.strip()):
            append_event(ATTEMPTS_PATH, {
                "at_utc": datetime.now(timezone.utc).isoformat(), "card_id": CARD_ID,
                "event": "PREFLIGHT_PASS", "freeze_signature": json_sha256(context["identity"]),
                "support_status": "PASS", "resource_status": "PASS", "budget_status": "PASS",
            })
        return {"card_id": CARD_ID, "status": "FROZEN_NO_FIT",
                "freeze_signature": json_sha256(context["identity"]),
                "recipe_path": str(RECIPE_PATH), "freeze_path": str(FREEZE_PATH),
                "preflight_path": str(PREFLIGHT_PATH)}
    if not RECIPE_PATH.is_file() or not FREEZE_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise R14RunnerError("R14 requires committed recipe, freeze, and passing preflight before fitting")
    frozen = _read_json(FREEZE_PATH)
    if frozen.get("identity") != _safe_json(context["identity"]):
        raise R14RunnerError("R14 current inputs/source/recipe differ from the committed freeze")
    if _read_json(PREFLIGHT_PATH).get("freeze_identity_sha256") != json_sha256(context["identity"]):
        raise R14RunnerError("R14 saved preflight does not match the current freeze")
    if verify_only:
        if not ENGINE_REPORT_PATH.is_file():
            raise R14RunnerError("R14 pilot engine report is missing")
        report = _read_json(ENGINE_REPORT_PATH)
        if file_sha256(context["train_path"]) != TRAIN_SHA256:
            raise R14RunnerError("Official Train changed before R14 saved-output verification")
        target = pd.to_numeric(pd.read_csv(context["train_path"], usecols=["threat_rank"])["threat_rank"],
                               errors="raise").to_numpy(dtype=np.float64)
        if file_sha256(context["train_path"]) != TRAIN_SHA256:
            raise R14RunnerError("Official Train changed while R14 verification labels were loaded")
        verification, gate = _verify_pilot(context, report, target)
        accounting = _fit_accounting()
        _publish(context, report, verification, gate, accounting)
        _reconcile_budget(verification)
        return {"verification": verification, "gate": gate, "fit_accounting": accounting}

    if not file_sha256(context["train_path"]) == TRAIN_SHA256:
        raise R14RunnerError("Official Train changed immediately before R14 pilot")
    target_frame = pd.read_csv(context["train_path"], usecols=["threat_rank"])
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise R14RunnerError("Official Train changed while target values were loaded")
    target = pd.to_numeric(target_frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if target.shape != (len(context["frame"]),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise R14RunnerError("Official Train target is invalid")
    engine_report = run_pilot(
        card=card, artifact_dir=RUN_DIR, budget_ledger_path=LEDGER_PATH,
        max_successful_fits=MAX_SUCCESSFUL_FITS,
        frame=context["frame"], target=target, feature_names=FEATURE_NAMES,
        adapter=_resource_guarded_adapter, control_loader=_control_loader(context),
        freeze_identity=context["identity"], freeze_path=FREEZE_PATH,
        event_path=ATTEMPTS_PATH, evaluation_views=context["views"],
        threads=THREADS, report_path=ENGINE_REPORT_PATH,
    )
    verification, gate = _verify_pilot(context, engine_report, target)
    accounting = _fit_accounting()
    _publish(context, engine_report, verification, gate, accounting)
    _reconcile_budget(verification)
    return {"report": engine_report, "verification": verification,
            "gate": gate, "fit_accounting": accounting}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run-frozen-pilot", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot,
                      verify_only=args.verify_only)
    print(json.dumps(_safe_json(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
