"""Freeze and evaluate C20 using existing, row-aligned prediction caches only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    metric,
)
from src.models.optimization_v3_batch1_c20 import (
    compose_c20_predictions,
    fit_only_q80,
    missing_stream_mask,
)
from scripts.optimization_v3_batch1_verify import _classification_group_metrics
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
CARD_ID = "C20"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
CUT_OFFS = (55, 59, 63, 67, 71, 75)
LONG_SCORE_CUTOFFS = frozenset({55, 67})
EXPECTED_ROWS = {55: 17685, 59: 6493, 63: 7078, 67: 20903, 71: 7462, 75: 8491}
EXPECTED_BASELINE = {"primary": 0.829944, "H12": 0.802966}
TRAIN_COLUMNS = (
    "record_id", "cycle_code", "text_load", "signal_stream", "entry_friction",
    "org_code", "module_code", "pattern_code", "artifact_signal",
)
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C20.md"
BASELINE_PATH = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
Q80_PROTOCOL_PATH = ROOT / "experiments/autonomous_eda/classification_volume/protocol.json"
Q80_FREEZE_PATH = ROOT / "experiments/autonomous_eda/classification_volume/freeze.json"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C20"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C20"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RESULTS_PATH = REPORT_DIR / "results.json"
SCRIPT_PATH = ROOT / "scripts/optimization_v3_batch1_c20.py"
VERIFY_PATH = ROOT / "scripts/optimization_v3_batch1_c20_verify.py"
HELPER_PATH = ROOT / "src/models/optimization_v3_batch1_c20.py"
GROUP_VERIFIER_PATH = ROOT / "scripts/optimization_v3_batch1_verify.py"
VIEWS_HELPER_PATH = ROOT / "src/models/optimization_v3_batch1.py"
PRODUCER_PATHS = (
    ROOT / "scripts/research_autonomous_volume_route.py",
    ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
    ROOT / "src/models/optimization_v3_sparse_classification.py",
    ROOT / "src/models/classification_round2_models.py",
    ROOT / "src/models/campaign_classification_catboost.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected a JSON object in {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise BatchIntegrityError(f"Refusing to overwrite different artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _read_train(*, include_target: bool) -> tuple[Path, pd.DataFrame]:
    train_path = task_files("classification").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Classification Train is unavailable")
    train_path = Path(train_path)
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train hash differs from C20 card")
    columns = list(TRAIN_COLUMNS if include_target else TRAIN_COLUMNS[:-1])
    frame = pd.read_csv(train_path, usecols=columns)
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Expected 50,608 official Train rows, got {len(frame)}")
    ids = frame["record_id"].astype(str)
    if frame["record_id"].isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise BatchIntegrityError("Official Train record_id is missing, blank, or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Train cycle range differs from 47–79")
    if include_target:
        y = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y, [0, 1]).all():
            raise BatchIntegrityError("Classification target is not binary")
    return train_path, frame


def _cache_paths(cutoff: int) -> dict[str, Path]:
    sparse_root = (ROOT / "experiments/runs/optimization_v3_classification_sparse"
                   if cutoff in LONG_SCORE_CUTOFFS
                   else ROOT / "experiments/runs/optimization_v3_classification_sparse_full")
    return {
        "sparse25": sparse_root / f"fit{cutoff}.npz",
        "c2b": ROOT / f"experiments/runs/classification_round2_v1/fit{cutoff}__C2_B.npz",
        "cb": ROOT / f"experiments/runs/dual_cls_catboost_v1/fit{cutoff}.npz",
    }


def _expected_cache_rows(cycles: np.ndarray, cutoff: int) -> np.ndarray:
    horizon = 12 if cutoff in LONG_SCORE_CUTOFFS else 4
    return np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + horizon)).astype(np.int64)


def _load_cache(path: Path, role: str, expected_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen {role} prediction cache is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if role == "c2b":
            row_key, probability_key = "outer_rows", "probability"
        elif role == "sparse25":
            row_key, probability_key = "rows", "blend25"
        elif role == "cb":
            row_key, probability_key = "rows", "candidate"
        else:
            raise BatchIntegrityError(f"Unknown C20 cache role: {role}")
        if row_key not in saved.files or probability_key not in saved.files:
            raise BatchIntegrityError(f"Cache {path} lacks {row_key}/{probability_key}")
        rows = np.asarray(saved[row_key], dtype=np.int64)
        probability = np.asarray(saved[probability_key], dtype=np.float64)
    if not np.array_equal(rows, expected_rows):
        raise BatchIntegrityError(f"{role} cache rows do not exactly match frozen Train cycles for {path.name}")
    if probability.shape != rows.shape or not np.isfinite(probability).all():
        raise BatchIntegrityError(f"{role} cache has invalid probability shape or nonfinite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise BatchIntegrityError(f"{role} probability cache is outside [0,1]")
    return rows, probability


def _cache_input_paths() -> list[Path]:
    paths: list[Path] = []
    for cutoff in CUT_OFFS:
        for cache in _cache_paths(cutoff).values():
            paths.extend((cache, cache.with_suffix(".json")))
    return paths


def _current_head() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _prepare(queue: dict[str, Any]) -> dict[str, Any]:
    cards = {str(card["id"]): card for card in queue.get("cards", [])}
    card = cards.get(CARD_ID)
    if card is None or card.get("task") != "classification" or card.get("evaluation") != "CACHE_SIX":
        raise BatchIntegrityError("Canonical C20 card is missing or has an unexpected evaluation profile")
    if int(card.get("pilot_predictor_fits", -1)) != 0:
        raise BatchIntegrityError("C20 must remain cache-only with zero new model fits")
    views = evaluation_views_for_card(queue, CARD_ID)
    if tuple(int(value) for value in queue["evaluation"]["CACHE_SIX"]["primary_cutoffs"]) != CUT_OFFS:
        raise BatchIntegrityError("C20 CACHE_SIX cutoff list changed")
    train_path, frame = _read_train(include_target=False)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    text_load = pd.to_numeric(frame["text_load"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(text_load).all():
        raise BatchIntegrityError("C20 q80 requires finite text_load values across official Train")

    q80_freeze = _read_json(Q80_FREEZE_PATH)
    protocol = _read_json(Q80_PROTOCOL_PATH)
    if file_sha256(Q80_PROTOCOL_PATH) != q80_freeze.get("protocol_sha256"):
        raise BatchIntegrityError("Inherited q80 protocol no longer matches its historical freeze")
    q80_definition = protocol.get("candidate", {}).get("threshold_definition", "")
    required_phrases = ("all rows", "cycle_code <= fit_end", "ordinary linear", "text_load > q80")
    if any(phrase not in q80_definition for phrase in required_phrases):
        raise BatchIntegrityError("Inherited q80 definition is ambiguous or has changed")

    baseline = _read_json(BASELINE_PATH)
    if baseline.get("candidate_id") != "v3_cls_sparse25" or baseline.get("status") != "PREPARED_AND_QA_PASS":
        raise BatchIntegrityError("C20 frozen sparse25 control manifest is missing or invalid")
    expected_recipe = {
        "kind": "availability_present_blend",
        "present": [{"model": "C2_B", "weight": 0.75},
                    {"model": "SPARSE_BINARY", "weight": 0.25}],
        "missing": "CB_MISSING", "threshold": 0.5,
    }
    if baseline.get("recipe") != expected_recipe:
        raise BatchIntegrityError("C20 control recipe differs from the frozen sparse25 manifest")

    thresholds: dict[str, float] = {}
    route_support: dict[str, dict[str, int]] = {}
    cache_metadata: dict[str, Any] = {}
    cache_data: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    missing_all = missing_stream_mask(frame["signal_stream"])
    for cutoff in CUT_OFFS:
        expected_rows = _expected_cache_rows(cycles, cutoff)
        if len(expected_rows) != EXPECTED_ROWS[cutoff]:
            raise BatchIntegrityError(f"C20 cutoff {cutoff} score population changed")
        q80 = fit_only_q80(text_load, cycles, cutoff)
        thresholds[str(cutoff)] = q80
        current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        paths = _cache_paths(cutoff)
        for role, path in paths.items():
            rows, probability = _load_cache(path, role, expected_rows)
            current[role] = (rows, probability)
        cache_data[cutoff] = current
        _candidate, counts = compose_c20_predictions(
            stream_missing=missing_all[expected_rows], text_load=text_load[expected_rows], q80=q80,
            sparse25_probability=current["sparse25"][1],
            c2b_probability=current["c2b"][1], cb_probability=current["cb"][1],
        )
        route_support[str(cutoff)] = counts
        cache_metadata[str(cutoff)] = {
            "score_rows": len(expected_rows),
            "score_cycle_min": int(cycles[expected_rows].min()),
            "score_cycle_max": int(cycles[expected_rows].max()),
            "input_npz_sha256": {role: file_sha256(path) for role, path in paths.items()},
        }

    input_paths = [train_path, QUEUE_PATH, CARD_PATH, BASELINE_PATH,
                   Q80_PROTOCOL_PATH, Q80_FREEZE_PATH, *_cache_input_paths()]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"C20 frozen input is missing: {missing_inputs}")
    input_hashes = {str(path.relative_to(ROOT)): file_sha256(path)
                    for path in sorted(set(input_paths))}
    source_paths = [*PRODUCER_PATHS, SCRIPT_PATH, VERIFY_PATH, HELPER_PATH,
                    GROUP_VERIFIER_PATH, VIEWS_HELPER_PATH]
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"C20 source is missing: {missing_sources}")
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path)
                     for path in sorted(set(source_paths))}

    recipe = {
        "card_id": CARD_ID,
        "task": "classification",
        "fit_count": 0,
        "control_manifest": "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
        "route": {
            "missing": "CB_MISSING takes precedence",
            "present_and_text_load_strictly_above_q80": "sparse25 blend25 cache",
            "remaining_present": "C2_B probability cache",
            "threshold": "inherited fit-only q80 over all rows with cycle_code <= fit_end",
            "quantile_method": "NumPy ordinary linear interpolation (method='linear')",
            "threshold_source": "experiments/autonomous_eda/classification_volume/protocol.json",
            "threshold_population": "all fit rows, including missing-stream rows",
            "equality_rule": "text_load == q80 remains on C2_B",
            "probability_threshold": 0.5,
        },
        "cutoffs": list(CUT_OFFS),
        "q80_by_fit_end": thresholds,
        "route_counts_by_cache_cutoff": route_support,
        "cache_rows": {key: value["score_rows"] for key, value in cache_metadata.items()},
        "evaluation_views": views,
        "new_model_fits": 0,
        "test_or_sample_access": False,
        "kaggle_access": False,
        "numpy_version": np.__version__,
    }
    identity = {
        "recipe": recipe,
        "train_sha256": file_sha256(train_path),
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "source_commit": _current_head(),
        "historical_q80_freeze_signature": q80_freeze.get("freeze_signature"),
        "historical_q80_protocol_sha256": q80_freeze.get("protocol_sha256"),
        "expected_baseline_metrics_rounded": EXPECTED_BASELINE,
    }
    return {
        "card": card, "views": views, "train_path": train_path,
        "frame": frame, "cycles": cycles, "text_load": text_load,
        "missing": missing_all, "cache_data": cache_data,
        "thresholds": thresholds, "route_support": route_support,
        "cache_metadata": cache_metadata, "recipe": recipe,
        "identity": identity, "signature": json_sha256(identity),
    }


def _freeze(queue: dict[str, Any]) -> dict[str, Any]:
    context = _prepare(queue)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    _write_once(RECIPE_PATH, context["recipe"])
    _write_once(FREEZE_PATH, {
        "status": "FROZEN_BEFORE_C20_SCORING",
        "identity": context["identity"],
        "signature": context["signature"],
        "thresholds_feature_only": context["thresholds"],
        "target_read": False,
        "candidate_metric_computed": False,
        "test_read": False,
    })
    return {
        "card_id": CARD_ID, "status": "FROZEN_READY",
        "freeze_signature": context["signature"],
        "q80_by_fit_end": context["thresholds"],
        "route_counts_by_cache_cutoff": context["route_support"],
        "fit_count": 0, "target_read": False, "candidate_metric_computed": False,
        "test_read": False, "kaggle_access": False,
    }


def _frozen_matches(context: dict[str, Any]) -> None:
    frozen = _read_json(FREEZE_PATH)
    if (frozen.get("status") != "FROZEN_BEFORE_C20_SCORING"
            or frozen.get("signature") != context["signature"]
            or json_sha256(frozen.get("identity", {})) != context["signature"]):
        raise BatchIntegrityError("C20 freeze is missing or differs from current inputs/recipe")
    normalized_recipe = json.loads(json.dumps(context["recipe"], sort_keys=True, allow_nan=False))
    if _read_json(RECIPE_PATH) != normalized_recipe:
        raise BatchIntegrityError("C20 recipe differs from its frozen identity")


def _view_arrays(context: dict[str, Any], segments: list[tuple[int, int, int]]) -> tuple[np.ndarray, ...]:
    cycles = context["cycles"]
    candidate_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    cutoff_parts: list[np.ndarray] = []
    for cutoff, low, high in segments:
        rows = np.flatnonzero((cycles >= low) & (cycles <= high)).astype(np.int64)
        cached_rows, sparse25 = context["cache_data"][cutoff]["sparse25"]
        c2b_rows, c2b = context["cache_data"][cutoff]["c2b"]
        cb_rows, cb = context["cache_data"][cutoff]["cb"]
        if not (np.array_equal(rows, cached_rows[np.isin(cached_rows, rows)])
                and np.array_equal(rows, c2b_rows[np.isin(c2b_rows, rows)])
                and np.array_equal(rows, cb_rows[np.isin(cb_rows, rows)])):
            raise BatchIntegrityError(f"C20 view rows are not aligned for cutoff {cutoff}")
        pos = np.searchsorted(cached_rows, rows)
        sparse_p = sparse25[pos]
        c2b_p = c2b[np.searchsorted(c2b_rows, rows)]
        cb_p = cb[np.searchsorted(cb_rows, rows)]
        missing = context["missing"][rows]
        cand, _ = compose_c20_predictions(
            stream_missing=missing, text_load=context["text_load"][rows],
            q80=context["thresholds"][str(cutoff)], sparse25_probability=sparse_p,
            c2b_probability=c2b_p, cb_probability=cb_p,
        )
        base = np.where(missing, cb_p, sparse_p)
        candidate_parts.append(cand)
        baseline_parts.append(base)
        row_parts.append(rows)
        cutoff_parts.append(np.full(len(rows), cutoff, dtype=np.int64))
    rows = np.concatenate(row_parts)
    if len(rows) != len(np.unique(rows)):
        raise BatchIntegrityError("C20 evaluation view contains overlapping row IDs")
    order = np.argsort(rows, kind="stable")
    return (rows[order], np.concatenate(candidate_parts)[order],
            np.concatenate(baseline_parts)[order], np.concatenate(cutoff_parts)[order])


def _extended_metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    result = metric("classification", y, probability)
    predicted = np.asarray(probability, dtype=np.float64) >= 0.5
    labels = np.asarray(y, dtype=np.int64)
    positive_n = int(np.sum(labels == 1))
    result["positive_recall"] = (float(np.sum(predicted & (labels == 1)) / positive_n)
                                 if positive_n else None)
    result["confusion_matrix"] = [
        [int(np.sum((labels == 0) & (~predicted))), int(np.sum((labels == 0) & predicted))],
        [int(np.sum((labels == 1) & (~predicted))), int(np.sum((labels == 1) & predicted))],
    ]
    return result


def _score(context: dict[str, Any]) -> dict[str, Any]:
    _frozen_matches(context)
    train_path, frame = _read_train(include_target=True)
    if file_sha256(train_path) != context["identity"]["train_sha256"]:
        raise BatchIntegrityError("C20 Train changed after freeze")
    y_all = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    pending: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for name, segments in context["views"].items():
        rows, candidate_p, baseline_p, row_cutoffs = _view_arrays(context, segments)
        pending[name] = (rows, candidate_p, baseline_p, row_cutoffs)
        baseline_metrics[name] = _extended_metric(y_all[rows], baseline_p)
    for name, expected in EXPECTED_BASELINE.items():
        actual = baseline_metrics[name]["macro_f1"]
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-6):
            raise BatchIntegrityError(
                f"C20 reconstructed baseline {name}={actual:.9f} differs from frozen card {expected:.6f}"
            )

    views: dict[str, Any] = {}
    group_diagnostics: dict[str, Any] = {}
    for name, (rows, candidate_p, baseline_p, row_cutoffs) in pending.items():
        candidate_metric = _extended_metric(y_all[rows], candidate_p)
        baseline_metric = baseline_metrics[name]
        views[name] = {
            "rows": int(len(rows)),
            "baseline": baseline_metric,
            "candidate": candidate_metric,
            "gain_positive_is_better": float(candidate_metric["macro_f1"] - baseline_metric["macro_f1"]),
            "route_counts": context["route_support"],
        }
        if name in {"primary", "H12", "H12_E", "H12_L"}:
            group_diagnostics[name] = _classification_group_metrics(
                frame, y_all[rows], candidate_p, baseline_p, rows, row_cutoffs,
            )
    return {
        "card_id": CARD_ID, "status": "CACHE_RECOMPOSITION_COMPLETE",
        "evaluation_profile": "CACHE_SIX", "cutoffs": list(CUT_OFFS),
        "q80_by_fit_end": context["thresholds"],
        "route_counts_by_cache_cutoff": context["route_support"],
        "views": views, "supported_group_diagnostics": group_diagnostics,
        "fit_accounting": {"predictor_fits": 0, "failed_fits": 0, "cache_only": True},
        "freeze_signature": context["signature"],
        "train_sha256": context["identity"]["train_sha256"],
        "input_sha256": context["identity"]["input_sha256"],
        "source_sha256": context["identity"]["source_sha256"],
        "source_commit": context["identity"]["source_commit"],
        "target_rows_read": 50_608, "test_data_used": False, "sample_submission_used": False,
        "kaggle_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-cache", action="store_true")
    args = parser.parse_args()
    queue = _read_json(QUEUE_PATH)
    if args.freeze_only:
        result = _freeze(queue)
    else:
        context = _prepare(queue)
        if args.preflight_only:
            result = {
                "card_id": CARD_ID, "status": "PREFLIGHT_PASS",
                "freeze_signature": context["signature"],
                "q80_by_fit_end": context["thresholds"],
                "route_counts_by_cache_cutoff": context["route_support"],
                "cache_metadata": context["cache_metadata"], "new_model_fits": 0,
                "target_read": False, "candidate_metric_computed": False,
                "test_read": False, "kaggle_access": False,
            }
        else:
            report = _score(context)
            _write_once(RESULTS_PATH, report)
            result = {"card_id": CARD_ID, "status": "RESULTS_WRITTEN",
                      "freeze_signature": context["signature"],
                      "results_path": str(RESULTS_PATH.relative_to(ROOT)),
                      "fit_count": 0, "test_data_used": False, "kaggle_access": False}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

