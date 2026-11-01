"""Independent replay and gate check for the C20 cache-only route."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.optimization_v3_batch1_verify import _classification_group_metrics, _macro_f1_independent
from src.models.optimization_v3_batch1 import BatchIntegrityError, file_sha256, json_sha256
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
CUT_OFFS = (55, 59, 63, 67, 71, 75)
LONG_CUTOFFS = frozenset({55, 67})
EXPECTED_ROWS = {55: 17685, 59: 6493, 63: 7078, 67: 20903, 71: 7462, 75: 8491}
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C20"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C20"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
TRAIN_COLUMNS = (
    "record_id", "cycle_code", "text_load", "signal_stream", "entry_friction",
    "org_code", "module_code", "pattern_code", "artifact_signal",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected a JSON object: {path}")
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


def _read_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    path = task_files("classification").get("train")
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Official Classification Train is unavailable")
    path = Path(path)
    if file_sha256(path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train hash differs from frozen C20 source")
    frame = pd.read_csv(path, usecols=list(TRAIN_COLUMNS))
    if len(frame) != 50_608 or frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Train row count or record_id uniqueness differs")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80)) or not np.isin(target, [0, 1]).all():
        raise BatchIntegrityError("Official Train cycle or target validation failed")
    return path, frame, cycles, target


def _missing_mask(values: pd.Series) -> np.ndarray:
    values = pd.Series(values, dtype="object")
    missing = values.isna().to_numpy(dtype=bool)
    present_positions = np.flatnonzero(~missing)
    if len(present_positions):
        missing[present_positions] = values.iloc[present_positions].map(
            lambda value: not str(value).strip()
        ).to_numpy(dtype=bool)
    return missing


def _load_cache(path: Path, role: str, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if role == "c2b":
            rows = np.asarray(saved["outer_rows"], dtype=np.int64)
            probability = np.asarray(saved["probability"], dtype=np.float64)
        elif role == "sparse25":
            rows = np.asarray(saved["rows"], dtype=np.int64)
            probability = np.asarray(saved["blend25"], dtype=np.float64)
        elif role == "cb":
            rows = np.asarray(saved["rows"], dtype=np.int64)
            probability = np.asarray(saved["candidate"], dtype=np.float64)
        else:
            raise BatchIntegrityError(f"Unknown frozen C20 cache role: {role}")
    if not np.array_equal(rows, expected) or probability.shape != rows.shape:
        raise BatchIntegrityError(f"Independent {role} cache alignment failed: {path.name}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise BatchIntegrityError(f"Independent {role} probability validation failed: {path.name}")
    return rows, probability


def _cache_paths(cutoff: int) -> dict[str, Path]:
    sparse_root = (ROOT / "experiments/runs/optimization_v3_classification_sparse"
                   if cutoff in LONG_CUTOFFS
                   else ROOT / "experiments/runs/optimization_v3_classification_sparse_full")
    return {
        "sparse25": sparse_root / f"fit{cutoff}.npz",
        "c2b": ROOT / f"experiments/runs/classification_round2_v1/fit{cutoff}__C2_B.npz",
        "cb": ROOT / f"experiments/runs/dual_cls_catboost_v1/fit{cutoff}.npz",
    }


def _manual_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(probability, dtype=np.float64) >= 0.5
    yb = np.asarray(y, dtype=np.int64)
    if yb.shape != predicted.shape or not len(yb):
        raise BatchIntegrityError("Independent metric arrays are empty or misaligned")
    macro = _macro_f1_independent(yb, predicted.astype(np.int64))
    positive_count = int(np.sum(yb == 1))
    recall = (float(np.sum(predicted & (yb == 1)) / positive_count)
              if positive_count else None)
    tn = int(np.sum((yb == 0) & (~predicted)))
    fp = int(np.sum((yb == 0) & predicted))
    fn = int(np.sum((yb == 1) & (~predicted)))
    tp = int(np.sum((yb == 1) & predicted))
    return {"n": int(len(yb)), "macro_f1": macro, "positive_recall": recall,
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def _same_metric(actual: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    for key in ("macro_f1", "positive_recall"):
        left, right = actual.get(key), expected.get(key)
        if left is None or right is None:
            if left != right:
                raise BatchIntegrityError(f"C20 verification differs for {name}.{key}")
        elif not np.isclose(float(left), float(right), rtol=0.0, atol=1e-12):
            raise BatchIntegrityError(f"C20 verification differs for {name}.{key}")
    if int(actual.get("n", -1)) != int(expected.get("n", -2)):
        raise BatchIntegrityError(f"C20 verification row count differs for {name}")


def verify() -> tuple[dict[str, Any], dict[str, Any], str]:
    frozen = _read_json(FREEZE_PATH)
    identity = frozen.get("identity", {})
    signature = str(frozen.get("signature", ""))
    if (frozen.get("status") != "FROZEN_BEFORE_C20_SCORING"
            or not signature or json_sha256(identity) != signature):
        raise BatchIntegrityError("C20 frozen identity or signature is invalid")
    recipe = _read_json(RECIPE_PATH)
    if recipe != identity.get("recipe"):
        raise BatchIntegrityError("C20 recipe differs from frozen identity")
    for relative, digest in identity.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise BatchIntegrityError(f"C20 frozen input hash mismatch: {relative}")
    for relative, digest in identity.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise BatchIntegrityError(f"C20 frozen source hash mismatch: {relative}")

    result = _read_json(RESULTS_PATH)
    if result.get("freeze_signature") != signature:
        raise BatchIntegrityError("C20 result references a different freeze")
    train_path, frame, cycles, y_all = _read_train()
    text_load = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=np.float64)
    missing = _missing_mask(frame["signal_stream"])
    caches: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    q80s: dict[str, float] = {}
    route_counts: dict[str, dict[str, int]] = {}
    score_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cutoff in CUT_OFFS:
        fit = cycles <= cutoff
        if not fit.any() or not np.isfinite(text_load[fit]).all():
            raise BatchIntegrityError(f"C20 fit rows invalid for cutoff {cutoff}")
        q80 = float(np.quantile(text_load[fit], 0.80, method="linear"))
        q80s[str(cutoff)] = q80
        horizon = 12 if cutoff in LONG_CUTOFFS else 4
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + horizon)).astype(np.int64)
        if len(expected_rows) != EXPECTED_ROWS[cutoff]:
            raise BatchIntegrityError(f"C20 expected score row count changed at cutoff {cutoff}")
        current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role, path in _cache_paths(cutoff).items():
            current[role] = _load_cache(path, role, expected_rows)
        caches[cutoff] = current
        sparse = current["sparse25"][1]
        c2b = current["c2b"][1]
        cb = current["cb"][1]
        absent = missing[expected_rows]
        high = (~absent) & (text_load[expected_rows] > q80)
        low = (~absent) & ~high
        pred = np.empty(len(expected_rows), dtype=np.float64)
        pred[absent] = cb[absent]
        pred[high] = sparse[high]
        pred[low] = c2b[low]
        baseline = np.where(absent, cb, sparse)
        score_map[cutoff] = (expected_rows, pred, baseline)
        route_counts[str(cutoff)] = {
            "missing_cb": int(absent.sum()),
            "present_high_sparse25": int(high.sum()),
            "present_other_c2b": int(low.sum()),
        }
    if q80s != identity["recipe"].get("q80_by_fit_end"):
        raise BatchIntegrityError("Independent C20 q80 thresholds differ from freeze")
    if route_counts != identity["recipe"].get("route_counts_by_cache_cutoff"):
        raise BatchIntegrityError("Independent C20 route counts differ from freeze")

    views = identity["recipe"].get("evaluation_views", {})
    verified_views: dict[str, Any] = {}
    group_diagnostics: dict[str, Any] = {}
    for name, segments in views.items():
        all_rows: list[np.ndarray] = []
        all_candidate: list[np.ndarray] = []
        all_baseline: list[np.ndarray] = []
        all_cutoff: list[np.ndarray] = []
        for cutoff, low_cycle, high_cycle in segments:
            cached_rows, candidate_all, baseline_all = score_map[int(cutoff)]
            selected = (cycles[cached_rows] >= int(low_cycle)) & (cycles[cached_rows] <= int(high_cycle))
            rows = cached_rows[selected]
            all_rows.append(rows)
            all_candidate.append(candidate_all[selected])
            all_baseline.append(baseline_all[selected])
            all_cutoff.append(np.full(len(rows), int(cutoff), dtype=np.int64))
        rows = np.concatenate(all_rows)
        candidate_p = np.concatenate(all_candidate)
        baseline_p = np.concatenate(all_baseline)
        row_cutoffs = np.concatenate(all_cutoff)
        if len(rows) != len(np.unique(rows)):
            raise BatchIntegrityError(f"C20 {name} view has duplicate rows")
        order = np.argsort(rows, kind="stable")
        rows, candidate_p, baseline_p, row_cutoffs = (
            rows[order], candidate_p[order], baseline_p[order], row_cutoffs[order]
        )
        cand_metric = _manual_metrics(y_all[rows], candidate_p)
        base_metric = _manual_metrics(y_all[rows], baseline_p)
        gain = float(cand_metric["macro_f1"] - base_metric["macro_f1"])
        verified_views[name] = {
            "rows": len(rows), "baseline": base_metric, "candidate": cand_metric,
            "gain_positive_is_better": gain,
        }
        result_view = result.get("views", {}).get(name, {})
        if int(result_view.get("rows", -1)) != len(rows):
            raise BatchIntegrityError(f"C20 result row count differs in {name}")
        if not np.isclose(gain, float(result_view.get("gain_positive_is_better", np.nan)),
                          rtol=0.0, atol=1e-12):
            raise BatchIntegrityError(f"C20 result metric differs in {name}")
        _same_metric(cand_metric, result_view.get("candidate", {}), f"{name}.candidate")
        _same_metric(base_metric, result_view.get("baseline", {}), f"{name}.baseline")
        if name == "primary":
            group_diagnostics[name] = _classification_group_metrics(
                frame, y_all[rows], candidate_p, baseline_p, rows, row_cutoffs,
            )

    primary_gain = verified_views["primary"]["gain_positive_is_better"]
    h12_gain = verified_views["H12"]["gain_positive_is_better"]
    windows = [f"W{i}" for i in range(1, 7)]
    window_gains = {name: verified_views[name]["gain_positive_is_better"] for name in windows}
    supported_failures = group_diagnostics["primary"].get("supported_guard_failures", [])
    checks = {
        "full_primary_gain_at_least_0_0005": primary_gain >= 0.0005,
        "full_H12_harm_at_most_0_001": h12_gain >= -0.001,
        "all_individual_window_harm_at_most_0_005": all(value >= -0.005 for value in window_gains.values()),
        "at_least_four_of_six_windows_nonnegative": sum(value >= 0.0 for value in window_gains.values()) >= 4,
        "supported_subgroup_guards": len(supported_failures) == 0,
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    verification = {
        "card_id": "C20", "status": "INDEPENDENT_QA_PASS", "decision": decision,
        "freeze_signature": signature,
        "independent_metric_implementation": "scripts/optimization_v3_batch1_c20_verify.py:_manual_metrics",
        "train_sha256": file_sha256(train_path), "target_read_from_official_train_only": True,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
        "new_model_fits": 0, "views": verified_views, "window_gains": window_gains,
        "supported_group_diagnostics": group_diagnostics,
        "supported_subgroup_failures": supported_failures,
        "gate_checks": checks,
    }
    gate = {
        "card_id": "C20", "status": decision,
        "decision_basis": "Frozen cache-only route; full-six row-matched controls and independent replay.",
        "thresholds": {
            "full_primary_gain_min": 0.0005, "full_H12_harm_max": 0.001,
            "individual_window_harm_max": 0.005, "full_nonnegative_windows_min": 4,
            "supported_group_n_min": 200, "positive_and_negative_per_group_min": 30,
            "supported_group_F1_harm_max": 0.01, "positive_class_recall_harm_max": 0.02,
        },
        "checks": checks, "supported_subgroup_failures": supported_failures,
    }
    summary = (
        "# C20 cache-only q80 recomposition\n\n"
        f"- Decision: **{decision}** (independent QA: PASS).\n"
        f"- Full-six Macro F1: {verified_views['primary']['baseline']['macro_f1']:.6f} baseline → "
        f"{verified_views['primary']['candidate']['macro_f1']:.6f} candidate; gain {primary_gain:+.6f} "
        f"on {verified_views['primary']['rows']:,} rows.\n"
        f"- H12 Macro F1: {verified_views['H12']['baseline']['macro_f1']:.6f} baseline → "
        f"{verified_views['H12']['candidate']['macro_f1']:.6f} candidate; gain {h12_gain:+.6f} "
        f"on {verified_views['H12']['rows']:,} rows.\n"
        f"- Nonnegative primary windows: {sum(value >= 0 for value in window_gains.values())}/6.\n"
        "- Zero new model fits; official Train labels only. Test/Sample/Kaggle were not accessed.\n"
    )
    return verification, gate, summary


def main() -> int:
    verification, gate, summary = verify()
    _write_once(VERIFICATION_PATH, verification)
    _write_once(GATE_PATH, gate)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing C20 summary differs; refusing overwrite")
    else:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary)
    print(json.dumps({"card_id": "C20", "status": verification["status"],
                      "decision": verification["decision"],
                      "gate_checks": verification["gate_checks"],
                      "test_data_used": False, "kaggle_access": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

