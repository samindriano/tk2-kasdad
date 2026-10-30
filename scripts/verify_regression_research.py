"""Independent audit of saved prediction-only regression research stages.

Stage 1 recomputes per-cycle rank mappings from frozen Round 2 control vectors,
then independently checks saved vectors and score summaries. It never fits a
model and does not pass target values to the mapping functions.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import PROJECT_ROOT, task_files


CONFIG = PROJECT_ROOT / "experiments" / "regression_research"
REPORT = PROJECT_ROOT / "reports" / "regression_research"
RUNS = PROJECT_ROOT / "experiments" / "runs"
STAGE1_RESULT = REPORT / "stage1_ranking.json"
STAGE1_FREEZE = CONFIG / "stage1_ranking_freeze.json"
TOLERANCE = 1e-10
QUANTILE_LABELS = (0, .01, .05, .25, .5, .75, .95, .99, 1)
QUANTILES = np.asarray(QUANTILE_LABELS, dtype=float)
TEXT_EXTENSIONS = {".py", ".md", ".json", ".txt", ".ipynb", ".toml", ".yaml", ".yml", ".csv", ""}
CONTROL_MODELS = ("R1_C", "R1_D", "B_D_RECENT8", "A_NO_CODES")
POLICY_ALPHA: dict[str, float | None] = {
    "raw": None,
    "clip": 0.0,
    "rank25": 0.25,
    "rank50": 0.5,
    "rank75": 0.75,
    "rank100": 1.0,
}
EXPECTED_SUBGROUPS = (
    "EF0", "EF1", "region_low", "region_middle", "region_high",
    "tail_low_le5", "tail_high_ge95", "stream_missing", "stream_present",
)


class VerificationError(RuntimeError):
    """Raised when an artifact does not reconcile to its frozen source."""


def _sha_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(values: Any, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _compact_json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text_digest(path: Path) -> str:
    value = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protected_digest(path: Path) -> str:
    if path.suffix.lower() in TEXT_EXTENSIONS or path.name.startswith("."):
        return _text_digest(path)
    return _sha_bytes(path)


def _close(actual: Any, expected: Any, label: str, *, atol: float = 2e-11, rtol: float = 2e-12) -> None:
    if expected is None or actual is None:
        if actual is not expected:
            raise VerificationError(f"{label}: expected {expected!r}, got {actual!r}")
        return
    a, e = float(actual), float(expected)
    if not math.isfinite(a) or not math.isfinite(e) or not math.isclose(a, e, rel_tol=rtol, abs_tol=atol):
        raise VerificationError(f"{label}: expected {e:.17g}, got {a:.17g}")


def _check_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise VerificationError(f"{label}: expected {expected!r}, got {actual!r}")


def average_percentile(prediction: Any, cycles: Any) -> np.ndarray:
    """Map finite scores to within-cycle 1-based average ranks in [0,100].

    The function signature contains predictions and cycle codes only; target
    values cannot influence ranks or their tie treatment.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    cycle = np.asarray(cycles)
    if pred.ndim != 1 or cycle.ndim != 1 or cycle.shape != pred.shape:
        raise ValueError("prediction and cycles must be aligned one-dimensional arrays")
    if not np.isfinite(pred).all() or pd.isna(cycle).any():
        raise ValueError("rank mapping requires finite predictions and nonmissing cycles")
    mapped = np.empty(len(pred), dtype=np.float64)
    for code in pd.unique(cycle):
        mask = cycle == code
        # pandas' average-tie rank is an independent implementation from the
        # ranking-stage scipy.rankdata call.
        rank = pd.Series(pred[mask]).rank(method="average", ascending=True).to_numpy(dtype=np.float64)
        mapped[mask] = 100.0 * rank / int(mask.sum())
    return mapped


def map_policy(prediction: Any, cycles: Any, policy: str) -> np.ndarray:
    """Apply a preregistered Stage 1 mapping without consulting labels."""
    pred = np.asarray(prediction, dtype=np.float64)
    cycle = np.asarray(cycles)
    if pred.ndim != 1 or cycle.ndim != 1 or cycle.shape != pred.shape:
        raise ValueError("prediction and cycles must be aligned one-dimensional arrays")
    if not np.isfinite(pred).all():
        raise ValueError("prediction must contain only finite values")
    if policy not in POLICY_ALPHA:
        raise ValueError(f"unknown Stage 1 policy: {policy}")
    alpha = POLICY_ALPHA[policy]
    if alpha is None:
        return pred.copy()
    if alpha == 0.0:
        return np.clip(pred, 0.0, 100.0)
    rank = average_percentile(pred, cycle)
    return np.clip((1.0 - alpha) * pred + alpha * rank, 0.0, 100.0)


def _metric_values(y: Any, prediction: Any) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if actual.ndim != 1 or estimate.ndim != 1 or actual.shape != estimate.shape:
        raise ValueError("actual and prediction must be aligned one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError("actual and prediction values must be finite")
    n = len(actual)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "signed_error": None,
                "over_rate": None, "under_rate": None, "equal_rate": None,
                "absolute_error_p95": None, "prediction_quantiles": None,
                "target_quantiles": None, "below_zero": 0, "above_100": 0}
    error = estimate - actual
    abs_error = np.abs(error)
    p_quantiles, y_quantiles = np.quantile(estimate, QUANTILES), np.quantile(actual, QUANTILES)
    abs_q = np.quantile(abs_error, QUANTILES)[6]
    return {
        "n": int(n),
        "mae": math.fsum(float(value) for value in abs_error) / n,
        "rmse": math.sqrt(math.fsum(float(value) * float(value) for value in error) / n),
        "signed_error": math.fsum(float(value) for value in error) / n,
        "over_rate": int(np.count_nonzero(error > TOLERANCE)) / n,
        "under_rate": int(np.count_nonzero(error < -TOLERANCE)) / n,
        "equal_rate": int(np.count_nonzero(abs_error <= TOLERANCE)) / n,
        "absolute_error_p95": float(abs_q),
        "prediction_quantiles": {str(q): float(value) for q, value in zip(QUANTILE_LABELS, p_quantiles)},
        "target_quantiles": {str(q): float(value) for q, value in zip(QUANTILE_LABELS, y_quantiles)},
        "below_zero": int(np.count_nonzero(estimate < 0)),
        "above_100": int(np.count_nonzero(estimate > 100)),
    }


def _rank_correlation(left: Any, right: Any) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("correlation vectors must be aligned one-dimensional arrays")
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    am = math.fsum(float(value) for value in ar) / len(ar)
    bm = math.fsum(float(value) for value in br) / len(br)
    ac = ar - am
    bc = br - bm
    denom = math.sqrt(math.fsum(float(v) ** 2 for v in ac) * math.fsum(float(v) ** 2 for v in bc))
    return math.fsum(float(x) * float(y) for x, y in zip(ac, bc)) / denom


def _stream_missing(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(frame), dtype=bool)
    for i, value in enumerate(frame["signal_stream"].to_numpy(dtype=object)):
        if isinstance(value, str):
            result[i] = not value.strip()
        else:
            try:
                result[i] = bool(pd.isna(value))
            except (TypeError, ValueError):
                result[i] = False
    return result


def _group_masks(frame: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "EF0": frame["entry_friction"].eq(0).to_numpy(),
        "EF1": frame["entry_friction"].eq(1).to_numpy(),
        "region_low": y < (100.0 / 3.0),
        "region_middle": (y >= (100.0 / 3.0)) & (y < (200.0 / 3.0)),
        "region_high": y >= (200.0 / 3.0),
        "tail_low_le5": y <= 5.0,
        "tail_high_ge95": y >= 95.0,
        "stream_missing": _stream_missing(frame),
        "stream_present": ~_stream_missing(frame),
    }


def independent_summary(frame: pd.DataFrame, prediction: Any) -> dict[str, Any]:
    required = {"threat_rank", "cycle_code", "entry_friction", "signal_stream"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing {sorted(missing)}")
    y = frame["threat_rank"].to_numpy(dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if len(y) != len(pred):
        raise ValueError("frame and prediction lengths differ")
    cycles = frame["cycle_code"].to_numpy()
    per_cycle = []
    for cycle in sorted(pd.unique(cycles)):
        mask = cycles == cycle
        per_cycle.append({"cycle": int(cycle), **_metric_values(y[mask], pred[mask])})
    cycle_maes = [item["mae"] for item in per_cycle if item["mae"] is not None]
    groups = _group_masks(frame, y)
    return {
        "raw": _metric_values(y, pred),
        "clipped": _metric_values(y, np.clip(pred, 0, 100)),
        "equal_cycle_mae": math.fsum(float(value) for value in cycle_maes) / len(cycle_maes) if cycle_maes else None,
        "per_cycle": per_cycle,
        "subgroups": {name: _metric_values(y[mask], pred[mask]) for name, mask in groups.items()},
        "impact_groups": {},
        "pooled_spearman": _rank_correlation(y, pred),
        "equal_cycle_spearman": _equal_cycle_spearman(frame, y, pred),
    }


def _equal_cycle_spearman(frame: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> float | None:
    correlations = []
    cycles = frame["cycle_code"].to_numpy()
    for cycle in sorted(pd.unique(cycles)):
        mask = cycles == cycle
        value = _rank_correlation(y[mask], pred[mask])
        if value is not None and math.isfinite(value):
            correlations.append(value)
    return math.fsum(correlations) / len(correlations) if correlations else None


def _verify_metric_node(saved: dict[str, Any], y: np.ndarray, pred: np.ndarray, label: str) -> int:
    expected = _metric_values(y, pred)
    for key in ("n", "below_zero", "above_100"):
        _check_equal(saved.get(key), expected[key], f"{label}.{key}")
    for key in ("mae", "rmse", "signed_error", "over_rate", "under_rate", "equal_rate", "absolute_error_p95"):
        _close(saved.get(key), expected[key], f"{label}.{key}")
    for distribution in ("prediction_quantiles", "target_quantiles"):
        got, want = saved.get(distribution), expected[distribution]
        if got is None or want is None:
            _check_equal(got, want, f"{label}.{distribution}")
        else:
            _check_equal(set(got), set(want), f"{label}.{distribution}.keys")
            for key in want:
                _close(got[key], want[key], f"{label}.{distribution}[{key}]")
    return 1


def verify_summary_node(frame: pd.DataFrame, prediction: Any, saved: dict[str, Any], label: str,
                        *, impact_mode: str = "empty") -> dict[str, int]:
    y = frame["threat_rank"].to_numpy(dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    checks = {"metric_nodes": 0, "cycle_nodes": 0, "subgroup_nodes": 0}
    checks["metric_nodes"] += _verify_metric_node(saved["raw"], y, pred, f"{label}.raw")
    checks["metric_nodes"] += _verify_metric_node(saved["clipped"], y, np.clip(pred, 0, 100), f"{label}.clipped")

    cycles = frame["cycle_code"].to_numpy()
    expected_cycles = sorted(pd.unique(cycles))
    _check_equal(len(saved["per_cycle"]), len(expected_cycles), f"{label}.per_cycle.length")
    cycle_maes = []
    for item, cycle in zip(saved["per_cycle"], expected_cycles):
        _check_equal(int(item["cycle"]), int(cycle), f"{label}.per_cycle.cycle")
        mask = cycles == cycle
        checks["cycle_nodes"] += _verify_metric_node(item, y[mask], pred[mask], f"{label}.cycle[{cycle}]")
        if len(y[mask]):
            cycle_maes.append(_metric_values(y[mask], pred[mask])["mae"])
    equal_cycle = math.fsum(float(v) for v in cycle_maes) / len(cycle_maes) if cycle_maes else None
    _close(saved.get("equal_cycle_mae"), equal_cycle, f"{label}.equal_cycle_mae")

    masks = _group_masks(frame, y)
    _check_equal(set(saved["subgroups"]), set(EXPECTED_SUBGROUPS), f"{label}.subgroups.keys")
    for name, mask in masks.items():
        checks["subgroup_nodes"] += _verify_metric_node(
            saved["subgroups"][name], y[mask], pred[mask], f"{label}.subgroups.{name}"
        )
    impact_checks = 0
    if impact_mode == "empty":
        _check_equal(saved.get("impact_groups"), {}, f"{label}.impact_groups")
    elif impact_mode == "counts":
        if "impact_load" not in frame:
            raise ValueError("impact-count verification requires impact_load")
        impact_values = frame["impact_load"].to_numpy()
        unique_values = sorted(pd.unique(impact_values))
        expected_keys = {repr(float(value)) for value in unique_values}
        _check_equal(set(saved.get("impact_groups", {})), expected_keys, f"{label}.impact_groups.keys")
        for value in unique_values:
            group_key = repr(float(value))
            record = saved["impact_groups"][group_key]
            _close(record.get("impact_load"), float(value), f"{label}.impact_groups.{group_key}.impact_load")
            mask = impact_values == value
            _check_equal(record.get("metrics", {}).get("n"), int(mask.sum()),
                         f"{label}.impact_groups.{group_key}.n")
            regions = {
                "region_low": y < (100.0 / 3.0),
                "region_middle": (y >= (100.0 / 3.0)) & (y < (200.0 / 3.0)),
                "region_high": y >= (200.0 / 3.0),
            }
            _check_equal(set(record.get("by_target_region", {})), set(regions),
                         f"{label}.impact_groups.{group_key}.regions")
            for name, region_mask in regions.items():
                expected_n = int(np.count_nonzero(mask & region_mask))
                _check_equal(record["by_target_region"][name].get("n"), expected_n,
                             f"{label}.impact_groups.{group_key}.{name}.n")
                impact_checks += 1
            impact_checks += 1
    else:
        raise ValueError(f"unknown impact verification mode: {impact_mode}")

    pooled = _rank_correlation(y, pred)
    _close(saved.get("pooled_spearman"), pooled, f"{label}.pooled_spearman", atol=2e-10, rtol=2e-10)
    cycle_rhos = []
    for cycle in expected_cycles:
        mask = cycles == cycle
        rho = _rank_correlation(y[mask], pred[mask])
        if rho is not None and math.isfinite(rho):
            cycle_rhos.append(rho)
    equal_rho = math.fsum(cycle_rhos) / len(cycle_rhos) if cycle_rhos else None
    _close(saved.get("equal_cycle_spearman"), equal_rho,
           f"{label}.equal_cycle_spearman", atol=2e-10, rtol=2e-10)
    checks["impact_group_count_nodes"] = impact_checks
    return checks


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_freeze(freeze: dict[str, Any], root: Path, files: dict[str, Path]) -> dict[str, Any]:
    identity = freeze["identity"]
    _check_equal(freeze.get("signature"), _json_digest(identity), "stage1 freeze signature")
    current_code = {relative: _text_digest(root / relative) for relative in identity["code_sha256"]}
    _check_equal(current_code, identity["code_sha256"], "frozen Stage1 source code hashes")
    input_hashes = {name: _sha_bytes(path) for name, path in files.items()}
    _check_equal(input_hashes, identity["regression_input_sha256"], "frozen regression input hashes")
    package_versions = {
        package: importlib.metadata.version(package)
        for package in identity["packages"]
    }
    _check_equal(package_versions, identity["packages"], "frozen package versions")
    protected = freeze.get("protected_files", {})
    protected_checked = 0
    for relative, expected in protected.items():
        path = root / relative
        if not path.is_file():
            raise VerificationError(f"frozen protected file is missing: {relative}")
        _check_equal(_protected_digest(path), expected, f"protected file {relative}")
        protected_checked += 1
    return {"freeze_signature": freeze["signature"], "protected_files_checked": protected_checked,
            "source_code_hashes": current_code, "input_hashes": input_hashes}


def _source_run_path(run_id: str, root: Path) -> Path:
    if not re.fullmatch(r"regression_round2_[A-Za-z0-9_-]+", run_id):
        raise VerificationError(f"unexpected Round2 run id: {run_id!r}")
    runs_root = (root / "experiments" / "runs").resolve()
    run = (runs_root / run_id).resolve()
    if run == runs_root or not run.is_relative_to(runs_root):
        raise VerificationError("Round2 source run escaped experiments/runs")
    return run


def _load_source_vector(run_dir: Path, end: int, model: str
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    archive = run_dir / f"fit{end}__{model}.npz"
    metadata_path = run_dir / f"fit{end}__{model}.json"
    if not archive.is_file() or not metadata_path.is_file():
        raise VerificationError(f"missing frozen Round2 vector pair: {archive.name}")
    metadata = _load_json(metadata_path)
    with np.load(archive, allow_pickle=False) as content:
        required = {"fit_rows", "validation_rows", "prediction"}
        if set(content.files) != required:
            raise VerificationError(f"unexpected source NPZ members in {archive.name}: {content.files}")
        fit_rows, rows, pred = (content[name].copy() for name in ("fit_rows", "validation_rows", "prediction"))
    if fit_rows.ndim != 1 or fit_rows.dtype.kind not in "iu" or rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise VerificationError(f"source row positions are not integer arrays: {archive.name}")
    if pred.ndim != 1 or pred.shape != rows.shape or not np.isfinite(pred).all():
        raise VerificationError(f"source prediction vector is malformed: {archive.name}")
    pred_hash = _array_digest(pred, "<f8")
    fit_hash = _array_digest(fit_rows, "<i8")
    rows_hash = _array_digest(rows, "<i8")
    _check_equal(metadata.get("prediction_sha256"), pred_hash, f"{archive.name}.prediction_sha256")
    _check_equal(metadata.get("fit_positions_sha256"), fit_hash, f"{archive.name}.fit_positions_sha256")
    _check_equal(metadata.get("validation_positions_sha256"), rows_hash,
                 f"{archive.name}.validation_positions_sha256")
    _check_equal(metadata.get("fit_rows"), len(fit_rows), f"{archive.name}.fit_rows")
    _check_equal(metadata.get("validation_rows"), len(rows), f"{archive.name}.validation_rows")
    if metadata.get("validation_labels_passed_to_fit") is not False:
        raise VerificationError(f"source metadata does not attest label-blind fitting: {archive.name}")
    if metadata.get("early_stopping_validation_used") is not False:
        raise VerificationError(f"source metadata used validation for early stopping: {archive.name}")
    return fit_rows.astype(np.int64, copy=False), rows.astype(np.int64, copy=False), pred, {
        "npz_sha256": _sha_bytes(archive), "metadata_sha256": _sha_bytes(metadata_path),
        "prediction_sha256": pred_hash, "fit_positions_sha256": fit_hash,
        "validation_positions_sha256": rows_hash,
    }


def _expected_fit_rows(frame: pd.DataFrame, window: dict[str, Any], history: str) -> np.ndarray:
    fit_start, fit_end = map(int, window["fit"])
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    lower = fit_start
    if history == "recent8":
        lower = max(fit_start, fit_end - 8 + 1)
    elif history != "expanding":
        raise VerificationError(f"Stage1 control has unexpected history policy {history!r}")
    return np.flatnonzero((cycles >= lower) & (cycles <= fit_end)).astype(np.int64)


def _expected_source_validation_rows(frame: pd.DataFrame, protocol: dict[str, Any], fit_end: int) -> np.ndarray:
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    expected = np.zeros(len(frame), dtype=bool)
    windows = protocol["primary_windows"] + protocol["secondary_windows"]
    matched = [window for window in windows if int(window["fit"][1]) == int(fit_end)]
    if not matched:
        raise VerificationError(f"no frozen validation window matches fit end {fit_end}")
    for window in matched:
        lower, upper = map(int, window["validation"])
        expected |= (cycles >= lower) & (cycles <= upper)
    return np.flatnonzero(expected).astype(np.int64)


def verify_stage1(root: Path = PROJECT_ROOT, *, write: bool = True) -> dict[str, Any]:
    """Verify Stage 1 mappings, saved NPZ files, metrics, aggregate, and stress test."""
    root = Path(root).resolve()
    result_path = root / "reports" / "regression_research" / "stage1_ranking.json"
    freeze_path = root / "experiments" / "regression_research" / "stage1_ranking_freeze.json"
    r2_protocol_path = root / "experiments" / "regression_round2" / "protocol.json"
    r2_result_path = root / "reports" / "regression_round2" / "results.json"
    data_files = task_files("regression")
    if any(not data_files.get(key) for key in ("train", "test", "sample")):
        raise VerificationError("Frozen regression Train, Test, and Sample files are required")
    freeze = _load_json(freeze_path)
    freeze_check = _verify_freeze(freeze, root, {key: Path(data_files[key]) for key in ("train", "test", "sample")})
    frozen_protocol = freeze["identity"]["protocol"]
    if freeze.get("stage") != "stage1_ranking":
        raise VerificationError("freeze receipt is not for Stage 1 ranking")
    if frozen_protocol.get("controls") != list(CONTROL_MODELS):
        raise VerificationError("frozen Stage 1 controls differ from verifier contract")
    if frozen_protocol.get("policies") != list(POLICY_ALPHA):
        raise VerificationError("frozen Stage 1 policies differ from verifier contract")
    if frozen_protocol.get("weights") != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise VerificationError("frozen Stage 1 weights differ from verifier contract")

    result = _load_json(result_path)
    _check_equal(result.get("stage"), "stage1_ranking", "Stage 1 result stage")
    _check_equal(result.get("freeze_signature"), freeze["signature"], "Stage 1 result freeze signature")
    _check_equal(result.get("fresh_fits"), 0, "Stage 1 new fits")
    r2_protocol = _load_json(r2_protocol_path)
    r2_freeze = _load_json(root / "experiments" / "regression_round2" / "freeze.json")
    _check_equal(freeze["identity"]["round2_signature"], r2_freeze.get("signature"), "Round2 freeze signature")
    source_result = _load_json(r2_result_path)
    _check_equal(source_result.get("stage"), "all", "Round2 source result stage")
    if source_result.get("partial") is not False:
        raise VerificationError("Round2 source result is marked partial")
    _check_equal(source_result.get("freeze_signature"), r2_freeze.get("signature"), "Round2 results freeze signature")
    run_dir = _source_run_path(str(source_result["run_id"]), root)
    source_result_hash = _sha_bytes(r2_result_path)
    r2_identity_hash = _sha_bytes(root / "experiments" / "regression_round2" / "freeze.json")
    source_job_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for job in source_result.get("jobs", []):
        key = (str(job["family"]), str(job["window"]), str(job["model"]))
        if key in source_job_index:
            raise VerificationError(f"duplicate Round2 source result row: {key}")
        source_job_index[key] = job

    frame = pd.read_csv(data_files["train"])
    if "threat_rank" not in frame or "cycle_code" not in frame:
        raise VerificationError("training data lacks threat_rank or cycle_code")
    if frame["threat_rank"].isna().any() or not np.isfinite(frame["threat_rank"].to_numpy(dtype=np.float64)).all():
        raise VerificationError("training target is missing or nonfinite")
    results_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    cache_npz_hashes: dict[str, str] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    vectors: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    summary_checks = {"metric_nodes": 0, "cycle_nodes": 0, "subgroup_nodes": 0,
                      "impact_group_count_nodes": 0}
    source_artifact_count = 0
    mapping_jobs_checked = 0

    model_protocol = {str(recipe["id"]): recipe for recipe in r2_protocol["recipes"]}
    for family, window_key in (("primary", "primary_windows"), ("secondary", "secondary_windows")):
        for window in r2_protocol[window_key]:
            window_id = str(window["id"])
            valid_lo, valid_hi = map(int, window["validation"])
            fit_end = int(window["fit"][1])
            for model in CONTROL_MODELS:
                source_key = (family, window_id, model)
                if source_key not in source_job_index:
                    raise VerificationError(f"missing Round2 source job {source_key}")
                source_job = source_job_index[source_key]
                _check_equal(int(source_job["fit_end"]), fit_end, f"{source_key}.fit_end")
                recipe = model_protocol.get(model)
                if not recipe:
                    raise VerificationError(f"model {model} absent from frozen Round2 protocol")
                _check_equal(str(source_job.get("history")), str(recipe["history"]), f"{source_key}.history")
                fit_rows, source_rows, source_pred, source_hash = _load_source_vector(run_dir, fit_end, model)
                expected_fit = _expected_fit_rows(frame, window, str(recipe["history"]))
                if not np.array_equal(fit_rows, expected_fit):
                    raise VerificationError(f"Round2 fit positions differ from frozen history policy: {source_key}")
                expected_future = _expected_source_validation_rows(frame, r2_protocol, fit_end)
                if not np.array_equal(source_rows, expected_future):
                    raise VerificationError(f"Round2 prediction positions differ from strict future rows: {source_key}")
                if np.intersect1d(fit_rows, source_rows, assume_unique=True).size:
                    raise VerificationError(f"Round2 fit and validation rows overlap: {source_key}")
                if not np.all(np.diff(fit_rows) > 0) or not np.all(np.diff(source_rows) > 0):
                    raise VerificationError(f"Round2 row positions are not strictly sorted: {source_key}")
                source_hashes[f"{fit_end}__{model}"] = source_hash
                source_artifact_count += 1

                source_cycle = frame.iloc[source_rows]["cycle_code"].to_numpy(dtype=np.int64)
                keep = (source_cycle >= valid_lo) & (source_cycle <= valid_hi)
                rows = source_rows[keep].copy()
                raw = source_pred[keep].copy()
                expected_rows = np.flatnonzero(
                    frame["cycle_code"].between(valid_lo, valid_hi).to_numpy()
                ).astype(np.int64)
                if not np.array_equal(rows, expected_rows):
                    raise VerificationError(f"Stage1 selection does not retain exact window positions: {source_key}")
                _check_equal(source_job.get("prediction_sha256"), _array_digest(raw, "<f8"),
                             f"Round2 window prediction hash {source_key}")
                _check_equal(source_job.get("validation_positions_sha256"), _array_digest(rows, "<i8"),
                             f"Round2 window positions hash {source_key}")
                selected_frame = frame.iloc[rows].reset_index(drop=True)
                cycles = selected_frame["cycle_code"].to_numpy()
                for policy in POLICY_ALPHA:
                    expected_prediction = map_policy(raw, cycles, policy)
                    cache_name = f"ranking__{window_id}__{model}__{policy}"
                    cache_path = root / "experiments" / "runs" / "regression_research_v1" / f"{cache_name}.npz"
                    if not cache_path.is_file():
                        raise VerificationError(f"missing saved Stage1 prediction NPZ: {cache_name}")
                    with np.load(cache_path, allow_pickle=False) as cached:
                        if set(cached.files) != {"validation_rows", "prediction"}:
                            raise VerificationError(f"unexpected Stage1 NPZ members: {cache_name}")
                        saved_rows, saved_pred = cached["validation_rows"].copy(), cached["prediction"].copy()
                    if not np.array_equal(saved_rows, rows):
                        raise VerificationError(f"saved Stage1 validation rows differ: {cache_name}")
                    if not np.array_equal(saved_pred, expected_prediction):
                        raise VerificationError(f"saved Stage1 mapping differs from independent map: {cache_name}")
                    pred_hash = _array_digest(expected_prediction, "<f8")
                    matching_result = [job for job in result["jobs"] if (
                        job.get("family"), job.get("window"), job.get("model"), job.get("policy")
                    ) == (family, window_id, model, policy)]
                    _check_equal(len(matching_result), 1, f"Stage1 report job count {cache_name}")
                    saved_job = matching_result[0]
                    _check_equal(saved_job.get("prediction_sha256"), pred_hash, f"{cache_name}.prediction_sha256")
                    checks = verify_summary_node(selected_frame, expected_prediction, saved_job, cache_name)
                    for key, count in checks.items():
                        summary_checks[key] += count
                    cache_npz_hashes[f"{cache_name}.npz"] = _sha_bytes(cache_path)
                    results_by_key[(family, window_id, model, policy)] = saved_job
                    vectors[(family, window_id, model, policy)] = (rows, expected_prediction)
                    mapping_jobs_checked += 1

    expected_count = sum(len(r2_protocol[key]) for key in ("primary_windows", "secondary_windows")) * len(CONTROL_MODELS) * len(POLICY_ALPHA)
    _check_equal(len(result.get("jobs", [])), expected_count, "Stage1 job count")
    _check_equal(mapping_jobs_checked, expected_count, "Stage1 mappings checked")

    aggregate_cells = 0
    for family, window_key in (("primary", "primary_windows"), ("secondary", "secondary_windows")):
        family_out = result.get("aggregate", {}).get(family, {})
        _check_equal(set(family_out), {f"{model}__{policy}" for model in CONTROL_MODELS for policy in POLICY_ALPHA},
                     f"aggregate.{family}.keys")
        for model in CONTROL_MODELS:
            for policy in POLICY_ALPHA:
                parts = [vectors[(family, str(window["id"]), model, policy)] for window in r2_protocol[window_key]]
                rows = np.concatenate([part[0] for part in parts])
                pred = np.concatenate([part[1] for part in parts])
                if len(np.unique(rows)) != len(rows):
                    raise VerificationError(f"aggregate rows overlap: {family}/{model}/{policy}")
                order = np.argsort(rows, kind="stable")
                rows, pred = rows[order], pred[order]
                aggregate_frame = frame.iloc[rows].reset_index(drop=True)
                cell_name = f"{model}__{policy}"
                summary = family_out[cell_name]
                checks = verify_summary_node(aggregate_frame, pred, summary, f"aggregate.{family}.{cell_name}")
                for key, count in checks.items():
                    summary_checks[key] += count
                aggregate_cells += 1
    _check_equal(aggregate_cells, 2 * len(CONTROL_MODELS) * len(POLICY_ALPHA), "aggregate cell count")

    stress_expected = []
    batch = frozen_protocol["batch_stress"]
    for window in r2_protocol["primary_windows"]:
        window_id = str(window["id"])
        for model in CONTROL_MODELS:
            rows, raw = vectors[("primary", window_id, model, "raw")]
            part = frame.iloc[rows].reset_index(drop=True)
            cycles = part["cycle_code"].to_numpy(dtype=np.int64)
            y = part["threat_rank"].to_numpy(dtype=np.float64)
            for fraction in batch["retained_fractions"]:
                indices: list[int] = []
                for cycle in np.unique(cycles):
                    positions = np.flatnonzero(cycles == cycle)
                    rng = np.random.default_rng(np.random.SeedSequence([
                        int(batch["seed"]), int(cycle), int(float(fraction) * 100)
                    ]))
                    size = max(1, int(len(positions) * float(fraction)))
                    indices.extend(rng.choice(positions, size, replace=False).tolist())
                subset_indices = np.sort(np.asarray(indices, dtype=np.int64))
                subset_y = y[subset_indices]
                subset_cycle = cycles[subset_indices]
                for alpha in (0.5, 1.0):
                    full_mapped = map_policy(raw, cycles, "rank50" if alpha == .5 else "rank100")[subset_indices]
                    subset_policy = "rank50" if alpha == .5 else "rank100"
                    subset_mapped = map_policy(raw[subset_indices], subset_cycle, subset_policy)
                    full_mae = math.fsum(float(v) for v in np.abs(full_mapped - subset_y)) / len(subset_y)
                    subset_mae = math.fsum(float(v) for v in np.abs(subset_mapped - subset_y)) / len(subset_y)
                    mapping_change = math.fsum(float(v) for v in np.abs(full_mapped - subset_mapped)) / len(subset_y)
                    stress_expected.append({
                        "model": model, "window": window_id, "fraction": float(fraction),
                        "rank_weight": alpha, "n": int(len(subset_y)),
                        "full_mae": full_mae, "subset_mae": subset_mae,
                        "mean_abs_mapping_change": mapping_change,
                    })
    saved_stress = result.get("batch_stress", [])
    _check_equal(len(saved_stress), len(stress_expected), "batch stress result count")
    for i, (saved, expected) in enumerate(zip(saved_stress, stress_expected)):
        for key in ("model", "window", "n"):
            _check_equal(saved.get(key), expected[key], f"batch_stress[{i}].{key}")
        for key in ("fraction", "rank_weight", "full_mae", "subset_mae", "mean_abs_mapping_change"):
            _close(saved.get(key), expected[key], f"batch_stress[{i}].{key}")

    receipt = {
        "stage": "stage1_ranking_verification",
        "status": "PASS",
        "freeze_signature": freeze["signature"],
        "artifacts": {
            "verifier_code_sha256": _text_digest(Path(__file__)),
            "stage1_result_sha256": _sha_bytes(result_path),
            "round2_result_sha256": source_result_hash,
            "round2_freeze_sha256": r2_identity_hash,
            "training_csv_sha256": _sha_bytes(Path(data_files["train"])),
            "source_vector_files": source_hashes,
            "ranking_npz_files": cache_npz_hashes,
        },
        "checks": {
            "freeze": freeze_check,
            "source_round2_vectors_checked": source_artifact_count,
            "source_controls": list(CONTROL_MODELS),
            "mappings_checked": mapping_jobs_checked,
            "aggregate_cells_checked": aggregate_cells,
            "batch_stress_rows_checked": len(stress_expected),
            "metric_nodes_checked": summary_checks["metric_nodes"],
            "cycle_metric_nodes_checked": summary_checks["cycle_nodes"],
            "subgroup_metric_nodes_checked": summary_checks["subgroup_nodes"],
            "impact_group_count_nodes_checked": summary_checks["impact_group_count_nodes"],
            "ranking_npz_files_checked": len(cache_npz_hashes),
            "mapping_target_access": False,
            "metric_means_use_math_fsum": True,
            "all_reported_quantile_fields_checked": True,
            "official_test_labels_or_features_read": False,
            "model_fit_or_prediction_run": False,
        },
        "notes": [
            "Rank mappings are recomputed from saved Round 2 predictions using only cycle codes and predictions.",
            "Metrics are recomputed from training labels after mapping; MAE, bias, RMSE, and rates use math.fsum or exact counts.",
            "No model was refit and no official test file was read for this verification.",
        ],
    }
    if write:
        output = root / "reports" / "regression_research" / "verification_stage1.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return receipt


STRUCTURED_FEATURES = (
    "impact_load", "entry_friction", "cycle_code", "scope_pressure", "text_load",
    "scope_trace", "org_spread", "module_spread", "telemetry_flag", "org_code",
    "module_code", "pattern_code",
)
TEXT_FEATURES = {
    "text_ridge": ("signal_stream",),
    "hybrid_ridge": (*STRUCTURED_FEATURES, "signal_stream"),
    "svd_hgb": (*STRUCTURED_FEATURES, "signal_stream"),
    "lgb_inner": STRUCTURED_FEATURES,
}
LGB_CANDIDATE_PARAMS = {
    "base": {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 1.0},
    "regularized": {"num_leaves": 15, "min_child_samples": 80, "reg_lambda": 3.0},
    "deeper": {"num_leaves": 63, "min_child_samples": 30, "reg_lambda": 2.0},
}
LGB_COMMON_PARAMS = {
    "boosting_type": "gbdt", "objective": "regression_l1", "learning_rate": 0.05,
    "n_estimators": 400, "max_depth": -1, "min_child_weight": 0.001,
    "reg_alpha": 0.0, "colsample_bytree": 1.0, "subsample": 1.0,
    "subsample_freq": 0, "random_state": 42, "n_jobs": 2,
    "verbosity": -1, "deterministic": True, "force_col_wise": True,
}
INNER_POLICY_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
OUTER_POLICY_NAMES = ("raw", "rank50", "rank100", "inner")
REFINEMENT_RECIPES = ("svd128", "svd_noscope32", "structured_ridge")
REFINEMENT_POLICIES = ("raw", "inner")
REFINEMENT_STRUCTURED = STRUCTURED_FEATURES
REFINEMENT_HYBRID = (*STRUCTURED_FEATURES, "signal_stream")
REFINEMENT_NO_SCOPE = (
    "impact_load", "entry_friction", "cycle_code", "text_load", "telemetry_flag",
    "org_code", "module_code", "pattern_code", "signal_stream",
)


def _fingerprint_value(value: Any) -> str:
    try:
        if bool(pd.isna(value)):
            return "null"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "bool:" + ("1" if value else "0")
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "float:" + (number.hex() if math.isfinite(number) else repr(number))
    return "str:" + str(value)


def _fit_fingerprint(frame: pd.DataFrame, fit_rows: np.ndarray, features: tuple[str, ...]) -> str:
    digest = hashlib.sha256()

    def add_text(text: str) -> None:
        encoded = text.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add_text(json.dumps(features, ensure_ascii=False, separators=(",", ":")))
    add_text(str(len(fit_rows)))
    selected = frame.iloc[fit_rows]
    for feature in features:
        for value in selected[feature].to_numpy(dtype=object):
            add_text(_fingerprint_value(value))
    for value in selected["threat_rank"].to_numpy(dtype=np.float64):
        add_text(_fingerprint_value(float(value)))
    return digest.hexdigest()


def _rank_weight_prediction(prediction: Any, cycles: Any, weight: float) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    cycle = np.asarray(cycles)
    if pred.ndim != 1 or cycle.ndim != 1 or pred.shape != cycle.shape or not np.isfinite(pred).all():
        raise ValueError("mapping requires finite aligned prediction and cycle arrays")
    if weight == 0:
        return np.clip(pred, 0.0, 100.0)
    rank = average_percentile(pred, cycle)
    return np.clip((1.0 - float(weight)) * pred + float(weight) * rank, 0.0, 100.0)


def _positions(frame: pd.DataFrame, lower: int, upper: int) -> np.ndarray:
    return np.flatnonzero(frame["cycle_code"].between(int(lower), int(upper)).to_numpy()).astype(np.int64)


def _stage23_outer_rows(frame: pd.DataFrame, protocol: dict[str, Any], end: int) -> tuple[int, np.ndarray]:
    windows = protocol["primary_windows"] + protocol["secondary_windows"]
    matching = [window for window in windows if int(window["fit"][1]) == int(end)]
    if not matching:
        raise VerificationError(f"no frozen Stage23 window matches cutoff {end}")
    last = max(int(window["validation"][1]) for window in matching)
    return last, _positions(frame, int(end) + 1, last)


def _stage23_model_path(cache: Path, kind: str, param: str, fit_rows: np.ndarray,
                        validation_rows: np.ndarray, frame: pd.DataFrame) -> tuple[str, Path, Path]:
    fit_end = int(frame.iloc[fit_rows]["cycle_code"].max())
    validation_end = int(frame.iloc[validation_rows]["cycle_code"].max())
    label = f"model__{kind}__{param}__fit{fit_end}__to{validation_end}"
    return label, cache / f"{label}.npz", cache / f"{label}.json"


def _verify_text_state(state: dict[str, Any], kind: str, fit_n: int,
                       requested_svd_components: int = 32) -> None:
    tfidf = state.get("tfidf", {})
    tfidf_params = tfidf.get("parameters", {})
    _check_equal(tfidf_params, {
        "ngram_range": [1, 2], "min_df": 2, "max_features": 20000,
        "sublinear_tf": True, "dtype": "float64",
    }, f"{kind}.TFIDF parameters")
    if not tfidf.get("vocabulary_sha256") or not tfidf.get("idf_sha256"):
        if tfidf.get("vocabulary_size") != 0:
            raise VerificationError(f"{kind} has an incomplete TF-IDF audit")
    if "vocabulary" in tfidf or "terms" in tfidf:
        raise VerificationError(f"{kind} audit contains raw vocabulary material")
    expected_features = list(TEXT_FEATURES[kind])
    _check_equal(state.get("input_feature_names"), expected_features, f"{kind}.input_feature_names")
    _check_equal(state.get("fit_rows"), fit_n, f"{kind}.state.fit_rows")
    _check_equal(state.get("parameters", {}).get("missing_stream_indicator"), True,
                 f"{kind}.missing_stream_indicator")
    if kind in {"text_ridge", "hybrid_ridge"}:
        _check_equal(state["parameters"].get("ridge"),
                     {"alpha": 10.0, "solver": "lsqr", "tol": 1e-4}, f"{kind}.Ridge parameters")
    if kind == "svd_hgb":
        _check_equal(state["parameters"].get("hgb"), {
            "loss": "absolute_error", "learning_rate": .05, "max_iter": 200,
            "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.0,
            "max_bins": 255, "early_stopping": False, "random_state": 42,
        }, "svd_hgb HGB parameters")
        svd = state["parameters"].get("svd", {})
        _check_equal({k: svd.get(k) for k in ("algorithm", "n_iter", "random_state", "n_components_requested")},
                     {"algorithm": "randomized", "n_iter": 7, "random_state": 42,
                      "n_components_requested": requested_svd_components}, "svd_hgb SVD parameters")
        structure = state.get("transform", {}).get("structured", {})
        names = structure.get("transformed_feature_names", [])
        native_mask = state.get("transform", {}).get("structured_categorical_mask", [])
        expected_mask = [str(name).endswith("__ordinal") for name in names]
        _check_equal(native_mask, expected_mask, "svd_hgb native structured categorical mask")
        design_mask = state.get("transform", {}).get("design_categorical_mask", [])
        component_count = int(svd.get("n_components_actual", -1))
        _check_equal(design_mask, native_mask + [False] * (component_count + 1),
                     "svd_hgb extended design categorical mask")
        shape = state.get("training_design_shape", [])
        if len(shape) != 2 or shape[0] != fit_n or shape[1] != len(design_mask):
            raise VerificationError("svd_hgb transformed design shape does not match fit rows and category mask")
    if kind == "hybrid_ridge":
        if not state.get("transform", {}).get("missing_indicator"):
            raise VerificationError("hybrid_ridge audit misses stream availability flag")
        transform = state.get("transform", {})
        if transform.get("structured_width", 0) <= 0 or transform.get("text_width", -1) < 0:
            raise VerificationError("hybrid_ridge structured/text dimensions are invalid")


def _verify_lgb_state(state: dict[str, Any], param: str, fit_n: int) -> None:
    expected = {**LGB_COMMON_PARAMS, **LGB_CANDIDATE_PARAMS[param]}
    _check_equal(state.get("actual_hyperparameters"), expected, f"lgb_inner.{param}.hyperparameters")
    _check_equal(state.get("objective"), "regression_l1", f"lgb_inner.{param}.objective")
    _check_equal(state.get("params_id"), param, f"lgb_inner.{param}.params_id")
    _check_equal(state.get("fit_rows"), fit_n, f"lgb_inner.{param}.fit_rows")
    _check_equal(state.get("early_stopping_used"), False, f"lgb_inner.{param}.early_stopping_used")
    _check_equal(state.get("validation_data_passed_to_fit"), False,
                 f"lgb_inner.{param}.validation_data_passed_to_fit")
    preprocessing = state.get("preprocessing", {})
    transformed = preprocessing.get("transformed_feature_names", [])
    expected_categorical = [i for i, name in enumerate(transformed) if str(name).endswith("__ordinal")]
    _check_equal(state.get("categorical_feature_indices"), expected_categorical,
                 f"lgb_inner.{param}.categorical positions")
    for feature, count in preprocessing.get("category_counts", {}).items():
        _check_equal(count.get("training_rows"), fit_n, f"lgb_inner.{param}.{feature}.training_rows")


def _load_stage23_model(cache: Path, frame: pd.DataFrame, kind: str, param: str,
                        expected_fit: np.ndarray, expected_validation: np.ndarray,
                        freeze_signature: str) -> tuple[np.ndarray, dict[str, Any], str]:
    label, archive, meta_path = _stage23_model_path(
        cache, kind, param, expected_fit, expected_validation, frame
    )
    if not archive.is_file() or not meta_path.is_file():
        raise VerificationError(f"missing Stage23 model cache pair: {label}")
    metadata = _load_json(meta_path)
    with np.load(archive, allow_pickle=False) as content:
        if set(content.files) != {"fit_rows", "validation_rows", "prediction"}:
            raise VerificationError(f"unexpected Stage23 model NPZ members: {label}")
        fit_rows, validation_rows, prediction = (
            content[name].copy() for name in ("fit_rows", "validation_rows", "prediction")
        )
    if not np.array_equal(fit_rows, expected_fit) or not np.array_equal(validation_rows, expected_validation):
        raise VerificationError(f"Stage23 cached row positions differ from frozen temporal split: {label}")
    if prediction.ndim != 1 or prediction.shape != validation_rows.shape or not np.isfinite(prediction).all():
        raise VerificationError(f"Stage23 prediction vector is invalid: {label}")
    fit_hash = _array_digest(fit_rows, "<i8")
    validation_hash = _array_digest(validation_rows, "<i8")
    prediction_hash = _array_digest(prediction, "<f8")
    _check_equal(metadata.get("fit_positions_sha256"), fit_hash, f"{label}.fit_positions_sha256")
    _check_equal(metadata.get("validation_positions_sha256"), validation_hash,
                 f"{label}.validation_positions_sha256")
    _check_equal(metadata.get("prediction_sha256"), prediction_hash, f"{label}.prediction_sha256")
    _check_equal(metadata.get("fit_rows"), len(fit_rows), f"{label}.fit_rows")
    _check_equal(metadata.get("validation_rows"), len(validation_rows), f"{label}.validation_rows")
    features = TEXT_FEATURES.get(kind)
    if features is None:
        raise VerificationError(f"unexpected Stage23 recipe: {kind}")
    _check_equal(metadata.get("features"), list(features), f"{label}.features")
    state = metadata.get("state", {})
    _check_equal(state.get("input_feature_names"), list(features), f"{label}.state.input_feature_names")
    _check_equal(state.get("fit_data_sha256"), _fit_fingerprint(frame, fit_rows, features),
                 f"{label}.fit_data_sha256")
    _check_equal(metadata.get("state_sha256"), _json_digest(state), f"{label}.state_sha256")
    _check_equal(metadata.get("key"), _json_digest({
        "freeze": freeze_signature, "kind": kind, "param": param,
        "fit": fit_hash, "validation": validation_hash,
    }), f"{label}.cache key")
    _check_equal(metadata.get("validation_labels_passed_to_fit"), False,
                 f"{label}.validation_labels_passed_to_fit")
    _check_equal(metadata.get("test_features_used_for_fit"), False, f"{label}.test_features_used_for_fit")
    _check_equal(state.get("fit_rows"), len(fit_rows), f"{label}.state.fit_rows")
    if kind in {"text_ridge", "hybrid_ridge", "svd_hgb"}:
        _check_equal(state.get("target_sha256"), _array_digest(
            frame.iloc[fit_rows]["threat_rank"].to_numpy(dtype=np.float64), "<f8"
        ), f"{label}.target_sha256")
        state_copy = dict(state)
        internal_digest = state_copy.pop("state_sha256", None)
        _check_equal(internal_digest, _compact_json_digest(state_copy), f"{label}.nested_state_sha256")
        _verify_text_state(state, kind, len(fit_rows))
    else:
        _verify_lgb_state(state, param, len(fit_rows))
    return prediction, metadata, _sha_bytes(archive)


def _selected_inner_choice(records: list[dict[str, Any]]) -> dict[str, Any]:
    complexity_order = {"regularized": 0, "base": 1, "deeper": 2}
    best = min(float(record["mae"]) for record in records)
    eligible = [record for record in records if float(record["mae"]) <= best + 1e-10]
    selected = sorted(eligible, key=lambda record: (
        float(record["rank_weight"]), complexity_order.get(str(record["param"]), 0)
    ))[0]
    return {"param": str(selected["param"]), "rank_weight": float(selected["rank_weight"]),
            "mae": float(selected["mae"])}


def verify_stage23(root: Path = PROJECT_ROOT, *, write: bool = True) -> dict[str, Any]:
    """Verify saved Stage23 candidate arrays, inner selections, and outer metrics."""
    root = Path(root).resolve()
    config = root / "experiments" / "regression_research"
    report = root / "reports" / "regression_research"
    cache = root / "experiments" / "runs" / "regression_research_v1"
    freeze = _load_json(config / "stage23_models_freeze.json")
    data_files = task_files("regression")
    if any(not data_files.get(key) for key in ("train", "test", "sample")):
        raise VerificationError("frozen regression Train, Test, and Sample files are required")
    freeze_check = _verify_freeze(freeze, root, {key: Path(data_files[key]) for key in ("train", "test", "sample")})
    _check_equal(freeze.get("stage"), "stage23_models", "Stage23 freeze stage")
    protocol = freeze["identity"]["protocol"]
    recipes = tuple(protocol["recipes"])
    _check_equal(recipes, ("text_ridge", "hybrid_ridge", "svd_hgb", "lgb_inner"), "Stage23 recipes")
    _check_equal(tuple(protocol["outer_policies"]), OUTER_POLICY_NAMES, "Stage23 outer policies")
    _check_equal(tuple(protocol["rank_weights"]), INNER_POLICY_WEIGHTS, "Stage23 inner rank weights")
    result_path = report / "stage23_models.json"
    result = _load_json(result_path)
    _check_equal(result.get("stage"), "stage23_models", "Stage23 result stage")
    _check_equal(result.get("freeze_signature"), freeze.get("signature"), "Stage23 result freeze signature")
    _check_equal(result.get("selection_boundary"),
                 "Joint parameter/mapping selection uses strictly past inner labels only. Outer windows reused development; no Test prediction here.",
                 "Stage23 selection boundary")
    r2_freeze = _load_json(root / "experiments" / "regression_round2" / "freeze.json")
    _check_equal(freeze["identity"]["round2_signature"], r2_freeze.get("signature"), "Stage23 Round2 freeze signature")
    frame = pd.read_csv(data_files["train"])
    if frame["threat_rank"].isna().any() or not np.isfinite(frame["threat_rank"].to_numpy(dtype=np.float64)).all():
        raise VerificationError("training target is missing or nonfinite")
    r2_protocol = _load_json(root / "experiments" / "regression_round2" / "protocol.json")
    outer_windows = r2_protocol["primary_windows"] + r2_protocol["secondary_windows"]
    end_values = sorted({int(window["fit"][1]) for window in outer_windows})
    expected_inner_keys = {(end, recipe) for end in end_values for recipe in recipes}
    saved_inner_index = {(int(item["fit_end"]), str(item["recipe"])): item for item in result.get("inner", [])}
    _check_equal(len(saved_inner_index), len(result.get("inner", [])), "Stage23 inner duplicate keys")
    _check_equal(set(saved_inner_index), expected_inner_keys, "Stage23 inner selection cells")
    _check_equal(set(result.get("audits", {})), {str(end) for end in end_values}, "Stage23 audit cutoff keys")
    model_artifact_hashes: dict[str, str] = {}
    evaluation_hashes: dict[str, str] = {}
    model_artifacts_checked = 0
    selection_checks = 0
    selection_records_checked = 0
    row_positions_checked = 0
    model_meta_checks = 0
    model_vectors: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    selected_by_end_recipe: dict[tuple[int, str], dict[str, Any]] = {}

    for end in end_values:
        inner_fit_rows = _positions(frame, 47, end - 4)
        inner_validation_rows = _positions(frame, end - 3, end)
        outer_fit_rows = _positions(frame, 47, end)
        last, outer_validation_rows = _stage23_outer_rows(frame, r2_protocol, end)
        if not len(inner_fit_rows) or not len(inner_validation_rows) or not len(outer_validation_rows):
            raise VerificationError(f"empty temporal split at cutoff {end}")
        if not (frame.iloc[inner_fit_rows].cycle_code.max() < frame.iloc[inner_validation_rows].cycle_code.min()
                <= end < frame.iloc[outer_validation_rows].cycle_code.min()):
            raise VerificationError(f"Stage23 inner/outer chronology is not strictly past at cutoff {end}")
        for recipe in recipes:
            item = saved_inner_index[(end, recipe)]
            _check_equal(item.get("inner_fit_max"), end - 4, f"inner.{end}.{recipe}.fit_max")
            _check_equal(item.get("inner_validation"), [end - 3, end], f"inner.{end}.{recipe}.validation")
            candidate_params = ("base", "regularized", "deeper") if recipe == "lgb_inner" else ("base",)
            audit = result["audits"][str(end)].get(recipe)
            if not audit:
                raise VerificationError(f"missing Stage23 audit for {end}/{recipe}")
            choices: list[dict[str, Any]] = []
            for param in candidate_params:
                if param not in audit.get("inner", {}):
                    raise VerificationError(f"missing Stage23 inner fit audit for {end}/{recipe}/{param}")
                inner_raw, metadata, artifact_hash = _load_stage23_model(
                    cache, frame, recipe, param, inner_fit_rows, inner_validation_rows, freeze["signature"]
                )
                label, archive, meta_path = _stage23_model_path(
                    cache, recipe, param, inner_fit_rows, inner_validation_rows, frame
                )
                meta_audit = audit["inner"][param]
                _check_equal(meta_audit, metadata, f"audits.{end}.{recipe}.inner.{param}")
                model_artifact_hashes[f"{label}.npz"] = artifact_hash
                model_artifact_hashes[f"{label}.json"] = _sha_bytes(meta_path)
                model_artifacts_checked += 1
                model_meta_checks += 1
                row_positions_checked += 2
                cycle_codes = frame.iloc[inner_validation_rows]["cycle_code"].to_numpy()
                y_inner = frame.iloc[inner_validation_rows]["threat_rank"].to_numpy(dtype=np.float64)
                for weight in INNER_POLICY_WEIGHTS:
                    mapped = _rank_weight_prediction(inner_raw, cycle_codes, weight)
                    mae = math.fsum(float(value) for value in np.abs(mapped - y_inner)) / len(y_inner)
                    choices.append({"param": param, "rank_weight": float(weight), "mae": mae})
            saved_choices = item.get("all_choices", [])
            _check_equal(len(saved_choices), len(choices), f"inner.{end}.{recipe}.choice_count")
            for index, (saved_choice, expected_choice) in enumerate(zip(saved_choices, choices)):
                _check_equal(saved_choice.get("param"), expected_choice["param"],
                             f"inner.{end}.{recipe}.choice[{index}].param")
                _close(saved_choice.get("rank_weight"), expected_choice["rank_weight"],
                       f"inner.{end}.{recipe}.choice[{index}].weight")
                _close(saved_choice.get("mae"), expected_choice["mae"],
                       f"inner.{end}.{recipe}.choice[{index}].mae")
                selection_records_checked += 1
            selected = _selected_inner_choice(choices)
            for key in ("param",):
                _check_equal(item.get("selected", {}).get(key), selected[key], f"inner.{end}.{recipe}.selected.{key}")
            for key in ("rank_weight", "mae"):
                _close(item.get("selected", {}).get(key), selected[key], f"inner.{end}.{recipe}.selected.{key}")
            _check_equal(audit.get("selection"), item["selected"], f"audits.{end}.{recipe}.selection")
            selected_by_end_recipe[(end, recipe)] = selected
            selection_checks += 1

            outer_param = selected["param"]
            if outer_param not in audit.get("inner", {}) and recipe != "lgb_inner":
                raise VerificationError(f"outer selected parameter is not a preregistered candidate: {recipe}/{end}")
            outer_raw, outer_meta, artifact_hash = _load_stage23_model(
                cache, frame, recipe, outer_param, outer_fit_rows, outer_validation_rows, freeze["signature"]
            )
            _check_equal(audit.get("outer"), outer_meta, f"audits.{end}.{recipe}.outer")
            model_label, model_archive, model_meta_path = _stage23_model_path(
                cache, recipe, outer_param, outer_fit_rows, outer_validation_rows, frame
            )
            model_artifact_hashes[f"{model_label}.npz"] = artifact_hash
            model_artifact_hashes[f"{model_label}.json"] = _sha_bytes(model_meta_path)
            model_artifacts_checked += 1
            model_meta_checks += 1
            row_positions_checked += 2
            model_vectors[(end, recipe, "raw")] = (outer_validation_rows, outer_raw)

            all_cycles = frame.iloc[outer_validation_rows]["cycle_code"].to_numpy()
            for policy in OUTER_POLICY_NAMES:
                if policy == "raw":
                    expected_full = outer_raw.copy()
                elif policy == "rank50":
                    expected_full = _rank_weight_prediction(outer_raw, all_cycles, .5)
                elif policy == "rank100":
                    expected_full = _rank_weight_prediction(outer_raw, all_cycles, 1.0)
                else:
                    expected_full = _rank_weight_prediction(outer_raw, all_cycles, selected["rank_weight"])
                eval_path = cache / f"eval__fit{end}__{recipe}__{policy}.npz"
                if not eval_path.is_file():
                    raise VerificationError(f"missing Stage23 evaluation vector: {eval_path.name}")
                with np.load(eval_path, allow_pickle=False) as saved:
                    if set(saved.files) != {"validation_rows", "prediction"}:
                        raise VerificationError(f"unexpected Stage23 eval NPZ members: {eval_path.name}")
                    saved_rows = saved["validation_rows"].copy()
                    saved_prediction = saved["prediction"].copy()
                if not np.array_equal(saved_rows, outer_validation_rows):
                    raise VerificationError(f"Stage23 evaluation rows differ from outer split: {eval_path.name}")
                if not np.array_equal(saved_prediction, expected_full):
                    raise VerificationError(f"Stage23 saved mapping differs from independent mapping: {eval_path.name}")
                evaluation_hashes[eval_path.name] = _sha_bytes(eval_path)
                model_vectors[(end, recipe, policy)] = (saved_rows, expected_full)

    expected_job_keys = {
        (recipe, policy, family, str(window["id"]))
        for family, windows in (("primary", r2_protocol["primary_windows"]),
                                ("secondary", r2_protocol["secondary_windows"]))
        for window in windows
        for recipe in recipes
        for policy in OUTER_POLICY_NAMES
    }
    saved_jobs = result.get("jobs", [])
    job_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for job in saved_jobs:
        key = (str(job["recipe"]), str(job["policy"]), str(job["family"]), str(job["window"]))
        if key in job_index:
            raise VerificationError(f"duplicate Stage23 score job: {key}")
        job_index[key] = job
    _check_equal(set(job_index), expected_job_keys, "Stage23 job keys")
    eval_job_checks = 0
    group_checks = {"metric_nodes": 0, "cycle_nodes": 0, "subgroup_nodes": 0, "impact_group_count_nodes": 0}
    for family, windows in (("primary", r2_protocol["primary_windows"]),
                            ("secondary", r2_protocol["secondary_windows"])):
        for window in windows:
            window_id = str(window["id"])
            lower, upper = map(int, window["validation"])
            end = int(window["fit"][1])
            for recipe in recipes:
                for policy in OUTER_POLICY_NAMES:
                    rows, full_prediction = model_vectors[(end, recipe, policy)]
                    cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
                    keep = (cycles >= lower) & (cycles <= upper)
                    selected_rows = rows[keep]
                    prediction = full_prediction[keep]
                    expected_rows = _positions(frame, lower, upper)
                    if not np.array_equal(selected_rows, expected_rows):
                        raise VerificationError(f"Stage23 output job row positions differ: {family}/{window_id}/{recipe}/{policy}")
                    job = job_index[(recipe, policy, family, window_id)]
                    _check_equal(job.get("prediction_sha256"), _array_digest(prediction, "<f8"),
                                 f"Stage23 job prediction hash {family}/{window_id}/{recipe}/{policy}")
                    summary_frame = frame.iloc[selected_rows].reset_index(drop=True)
                    checks = verify_summary_node(
                        summary_frame, prediction, job, f"Stage23.{family}.{window_id}.{recipe}.{policy}"
                    )
                    for key, count in checks.items():
                        group_checks[key] += count
                    eval_job_checks += 1

    aggregate_cells = 0
    for family, windows in (("primary", r2_protocol["primary_windows"]),
                            ("secondary", r2_protocol["secondary_windows"])):
        family_aggregate = result.get("aggregate", {}).get(family, {})
        expected_aggregate_keys = {f"{recipe}__{policy}" for recipe in recipes for policy in OUTER_POLICY_NAMES}
        _check_equal(set(family_aggregate), expected_aggregate_keys, f"Stage23 aggregate.{family}.keys")
        for recipe in recipes:
            for policy in OUTER_POLICY_NAMES:
                row_parts, pred_parts = [], []
                for window in windows:
                    end = int(window["fit"][1])
                    full_rows, full_pred = model_vectors[(end, recipe, policy)]
                    low, high = map(int, window["validation"])
                    keep = frame.iloc[full_rows]["cycle_code"].between(low, high).to_numpy()
                    row_parts.append(full_rows[keep]); pred_parts.append(full_pred[keep])
                rows, prediction = np.concatenate(row_parts), np.concatenate(pred_parts)
                if len(np.unique(rows)) != len(rows):
                    raise VerificationError(f"Stage23 aggregate rows overlap: {family}/{recipe}/{policy}")
                order = np.argsort(rows, kind="stable")
                rows, prediction = rows[order], prediction[order]
                aggregate_frame = frame.iloc[rows].reset_index(drop=True)
                summary = family_aggregate[f"{recipe}__{policy}"]
                checks = verify_summary_node(
                    aggregate_frame, prediction, summary,
                    f"Stage23.aggregate.{family}.{recipe}.{policy}", impact_mode="counts"
                )
                for key, count in checks.items():
                    group_checks[key] += count
                aggregate_cells += 1
    _check_equal(aggregate_cells, 2 * len(recipes) * len(OUTER_POLICY_NAMES), "Stage23 aggregate cell count")
    _check_equal(eval_job_checks, len(expected_job_keys), "Stage23 per-window job count")

    receipt = {
        "stage": "stage23_models_verification",
        "status": "PASS",
        "freeze_signature": freeze["signature"],
        "artifacts": {
            "verifier_code_sha256": _text_digest(Path(__file__)),
            "stage23_result_sha256": _sha_bytes(result_path),
            "training_csv_sha256": _sha_bytes(Path(data_files["train"])),
            "model_cache_files": model_artifact_hashes,
            "evaluation_cache_files": evaluation_hashes,
        },
        "checks": {
            "freeze": freeze_check,
            "model_artifacts_checked": model_artifacts_checked,
            "model_metadata_checks": model_meta_checks,
            "fit_and_validation_position_arrays_checked": row_positions_checked,
            "inner_selection_cells_checked": selection_checks,
            "inner_choice_scores_checked": selection_records_checked,
            "inner_selected_weights_checked_against_frozen_tie_rule": selection_checks,
            "evaluation_mapping_arrays_checked": len(evaluation_hashes),
            "outer_score_jobs_checked": eval_job_checks,
            "aggregate_cells_checked": aggregate_cells,
            "metric_nodes_checked": group_checks["metric_nodes"],
            "cycle_metric_nodes_checked": group_checks["cycle_nodes"],
            "subgroup_metric_nodes_checked": group_checks["subgroup_nodes"],
            "impact_group_support_nodes_checked": group_checks["impact_group_count_nodes"],
            "fit_fingerprints_recomputed_from_allowed_train_columns_and_targets": True,
            "validation_labels_passed_to_fit": False,
            "test_features_used_for_fit_or_selection": False,
            "official_test_labels_or_features_read": False,
            "model_refits_or_predictions_run": False,
        },
        "notes": [
            "Inner candidate scores are recomputed from saved inner prediction vectors and past labels; joint parameter/rank-weight selection is rechecked against the frozen tie rule.",
            "Outer maps are recomputed from saved full-window model vectors, then compared to evaluation NPZ vectors and per-window/aggregate metrics.",
            "Impact-group counts and target-region support are checked; full impact-group metric distributions are not independently recomputed.",
            "No model was fit or prediction generated during verification; official test rows were not parsed.",
        ],
    }
    if write:
        output = report / "verification_stage23.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def _verify_sealed_refinement_state(state: dict[str, Any], label: str) -> None:
    state_copy = dict(state)
    nested = state_copy.pop("state_sha256", None)
    _check_equal(nested, _compact_json_digest(state_copy), f"{label}.sealed_state_sha256")


def _verify_refinement_nested_text(state: dict[str, Any], label: str,
                                   requested_svd_components: int = 32) -> None:
    nested = dict(state)
    digest = nested.pop("state_sha256", None)
    _check_equal(digest, _compact_json_digest(nested), f"{label}.nested_text_state_sha256")
    _verify_text_state(state, "svd_hgb" if state.get("kind") == "svd_hgb" else "hybrid_ridge",
                       int(state.get("fit_rows", -1)), requested_svd_components)


def _verify_refinement_state(state: dict[str, Any], recipe: str, frame: pd.DataFrame,
                             fit_rows: np.ndarray) -> None:
    fit_n = len(fit_rows)
    _check_equal(state.get("recipe"), recipe, f"{recipe}.recipe")
    _check_equal(state.get("fit_rows"), fit_n, f"{recipe}.state.fit_rows")
    _check_equal(state.get("test_data_used_for_fit"), False, f"{recipe}.test_data_used_for_fit")
    _check_equal(state.get("validation_data_passed_to_fit"), False,
                 f"{recipe}.validation_data_passed_to_fit")
    if recipe == "structured_ridge":
        _check_equal(state.get("input_feature_names"), list(REFINEMENT_STRUCTURED),
                     "structured_ridge.input_feature_names")
        _check_equal(state.get("structured_ridge_alpha"), 10.0, "structured_ridge.alpha")
        _check_equal(state.get("constructed_signal_stream"), {
            "value": "", "fit_rows": fit_n, "actual_signal_stream_consumed": False,
            "constant_missing_indicator": 1,
        }, "structured_ridge.constructed_signal_stream")
        delegate = state.get("delegated_model", {})
        _check_equal(delegate.get("input_feature_names"), list(REFINEMENT_HYBRID),
                     "structured_ridge.delegated.input_feature_names")
        _check_equal(delegate.get("fit_rows"), fit_n, "structured_ridge.delegated.fit_rows")
        augmented = frame.iloc[fit_rows].loc[:, list(REFINEMENT_STRUCTURED)].copy()
        augmented["signal_stream"] = ""
        augmented = augmented.loc[:, list(REFINEMENT_HYBRID)]
        fingerprint = _fit_fingerprint_from_selected(
            augmented, frame.iloc[fit_rows]["threat_rank"].to_numpy(dtype=np.float64),
            REFINEMENT_HYBRID,
        )
        _check_equal(state.get("fit_data_sha256"), fingerprint, "structured_ridge.fit_data_sha256")
        _check_equal(delegate.get("fit_data_sha256"), fingerprint, "structured_ridge.delegated.fit_data_sha256")
        target_hash = _array_digest(frame.iloc[fit_rows]["threat_rank"].to_numpy(dtype=np.float64), "<f8")
        _check_equal(state.get("target_sha256"), target_hash, "structured_ridge.target_sha256")
        _check_equal(delegate.get("target_sha256"), target_hash, "structured_ridge.delegated.target_sha256")
        _check_equal(delegate.get("parameters", {}).get("ridge"),
                     {"alpha": 10.0, "solver": "lsqr", "tol": 1e-4}, "structured_ridge.Ridge parameters")
        _check_equal(delegate.get("tfidf", {}).get("vocabulary_size"), 0,
                     "structured_ridge.empty text vocabulary")
        _check_equal(delegate.get("tfidf", {}).get("fallback"), "all_training_text_empty",
                     "structured_ridge.empty text fallback")
        _check_equal(delegate.get("fit_missing_stream_rows"), fit_n,
                     "structured_ridge.constant missing indicator rows")
        _verify_refinement_nested_text(delegate, "structured_ridge.delegated")
        return

    expected_features = REFINEMENT_HYBRID if recipe == "svd128" else REFINEMENT_NO_SCOPE
    _check_equal(state.get("input_feature_names"), list(expected_features), f"{recipe}.input_feature_names")
    fingerprint = _fit_fingerprint(frame, fit_rows, tuple(expected_features))
    target_hash = _array_digest(frame.iloc[fit_rows]["threat_rank"].to_numpy(dtype=np.float64), "<f8")
    if recipe == "svd128":
        _check_equal(state.get("svd_components_requested"), 128, "svd128.requested components")
        delegate = state.get("delegated_model", {})
        _check_equal(delegate.get("fit_data_sha256"), fingerprint, "svd128.fit_data_sha256")
        _check_equal(delegate.get("target_sha256"), target_hash, "svd128.target_sha256")
        _check_equal(delegate.get("input_feature_names"), list(REFINEMENT_HYBRID),
                     "svd128.delegated.input_feature_names")
        svd = delegate.get("parameters", {}).get("svd", {})
        _check_equal(svd.get("n_components_requested"), 128, "svd128.delegated requested components")
        _check_equal(state.get("svd_components_actual"), svd.get("n_components_actual"),
                     "svd128 actual component count")
        if not 1 <= int(state.get("svd_components_actual", 0)) <= 128:
            raise VerificationError("svd128 component count is outside the frozen cap")
        _verify_refinement_nested_text(delegate, "svd128.delegated", requested_svd_components=128)
        transform = delegate.get("transform", {})
        structured_audit = transform.get("structured", {})
        names = structured_audit.get("transformed_feature_names", [])
        category_names = set(structured_audit.get("categorical_feature_names", []))
        cat_positions = [i for i, name in enumerate(names) if name in category_names]
        expected_positions = [i for i, name in enumerate(names) if str(name).endswith("__ordinal")]
        _check_equal(cat_positions, expected_positions, "svd128 native categorical positions")
        native_mask = transform.get("structured_categorical_mask", [])
        _check_equal(native_mask, [i in expected_positions for i in range(len(names))],
                     "svd128 native categorical mask")
        design_mask = transform.get("design_categorical_mask", [])
        _check_equal(design_mask, native_mask + [False] * (int(state["svd_components_actual"]) + 1),
                     "svd128 design categorical mask")
        if len(design_mask) != delegate.get("training_design_shape", [0, -1])[1]:
            raise VerificationError("svd128 design mask does not match fitted design width")
        return

    _check_equal(state.get("fit_data_sha256"), fingerprint, "svd_noscope32.fit_data_sha256")
    _check_equal(state.get("target_sha256"), target_hash, "svd_noscope32.target_sha256")
    removed = ["scope_pressure", "scope_trace", "org_spread", "module_spread"]
    _check_equal(state.get("removed_scope_spread_sources_and_derived_flags"), removed,
                 "svd_noscope32 removed source allowlist")
    preprocessing = state.get("structured_preprocessing", {})
    names = preprocessing.get("transformed_feature_names", [])
    native_positions = [i for i, name in enumerate(names) if str(name).endswith("__ordinal")]
    struct_mask = [i in native_positions for i in range(len(names))]
    svd = state.get("svd", {})
    _check_equal({k: svd.get(k) for k in ("algorithm", "n_iter", "random_state", "n_components_requested")},
                 {"algorithm": "randomized", "n_iter": 7, "random_state": 42,
                  "n_components_requested": 32}, "svd_noscope32 SVD parameters")
    actual_components = int(svd.get("n_components_actual", -1))
    if not 0 <= actual_components <= 32:
        raise VerificationError("svd_noscope32 component count is outside the frozen cap")
    expected_design_mask = struct_mask + [False] * (actual_components + 1)
    _check_equal(state.get("native_categorical_mask"), expected_design_mask,
                 "svd_noscope32 design categorical mask")
    _check_equal(state.get("hgb_parameters", {}).get("categorical_features"), expected_design_mask,
                 "svd_noscope32 estimator categorical mask")
    if state.get("training_design_shape") != [fit_n, len(expected_design_mask)]:
        raise VerificationError("svd_noscope32 design shape differs from its mask")
    # Reuse the same fixed HGB and TF-IDF policy as the frozen Stage2/3 text model.
    _check_equal({k: state["hgb_parameters"].get(k) for k in (
        "loss", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
        "l2_regularization", "max_bins", "early_stopping", "random_state",
    )}, {
        "loss": "absolute_error", "learning_rate": .05, "max_iter": 200,
        "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.0,
        "max_bins": 255, "early_stopping": False, "random_state": 42,
    }, "svd_noscope32 HGB parameters")
    _check_equal(state.get("tfidf", {}).get("parameters"), {
        "ngram_range": [1, 2], "min_df": 2, "max_features": 20000,
        "sublinear_tf": True, "dtype": "float64",
    }, "svd_noscope32 TF-IDF parameters")


def _fit_fingerprint_from_selected(X: pd.DataFrame, y: np.ndarray,
                                   features: tuple[str, ...]) -> str:
    """Fingerprint an already selected (possibly constructed) feature matrix."""
    digest = hashlib.sha256()

    def add(text: str) -> None:
        encoded = text.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add(json.dumps(features, ensure_ascii=False, separators=(",", ":")))
    add(str(len(X)))
    for feature in features:
        for value in X[feature].to_numpy(dtype=object):
            add(_fingerprint_value(value))
    for value in np.asarray(y, dtype=np.float64):
        add(_fingerprint_value(float(value)))
    return digest.hexdigest()


def _load_refinement_model(cache: Path, frame: pd.DataFrame, recipe: str,
                           fit_rows: np.ndarray, validation_rows: np.ndarray,
                           freeze_signature: str) -> tuple[np.ndarray, dict[str, Any], str, str]:
    fit_end = int(frame.iloc[fit_rows].cycle_code.max())
    validation_end = int(frame.iloc[validation_rows].cycle_code.max())
    label = f"refine__{recipe}__fit{fit_end}__to{validation_end}"
    archive, metadata_path = cache / f"{label}.npz", cache / f"{label}.json"
    if not archive.is_file() or not metadata_path.is_file():
        raise VerificationError(f"missing Stage4 model cache pair: {label}")
    metadata = _load_json(metadata_path)
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"fit_rows", "validation_rows", "prediction"}:
            raise VerificationError(f"unexpected Stage4 model NPZ members: {label}")
        saved_fit, saved_validation, prediction = (
            data[name].copy() for name in ("fit_rows", "validation_rows", "prediction")
        )
    if not np.array_equal(saved_fit, fit_rows) or not np.array_equal(saved_validation, validation_rows):
        raise VerificationError(f"Stage4 fit/validation positions differ from strict temporal split: {label}")
    if prediction.ndim != 1 or prediction.shape != validation_rows.shape or not np.isfinite(prediction).all():
        raise VerificationError(f"Stage4 prediction vector is invalid: {label}")
    fit_hash, validation_hash = _array_digest(saved_fit, "<i8"), _array_digest(saved_validation, "<i8")
    prediction_hash = _array_digest(prediction, "<f8")
    _check_equal(metadata.get("fit_positions_sha256"), fit_hash, f"{label}.fit positions hash")
    _check_equal(metadata.get("validation_positions_sha256"), validation_hash, f"{label}.validation positions hash")
    _check_equal(metadata.get("prediction_sha256"), prediction_hash, f"{label}.prediction hash")
    _check_equal(metadata.get("fit_rows"), len(fit_rows), f"{label}.fit_rows")
    _check_equal(metadata.get("kind"), recipe, f"{label}.kind")
    _check_equal(metadata.get("key"), _json_digest({
        "freeze": freeze_signature, "kind": recipe, "fit": fit_hash, "validation": validation_hash,
    }), f"{label}.cache key")
    _check_equal(metadata.get("validation_labels_passed_to_fit"), False,
                 f"{label}.validation_labels_passed_to_fit")
    state = metadata.get("state", {})
    _check_equal(metadata.get("state_sha256"), _json_digest(state), f"{label}.state_sha256")
    _verify_sealed_refinement_state(state, label)
    _verify_refinement_state(state, recipe, frame, fit_rows)
    allowed = {
        "svd128": list(REFINEMENT_HYBRID),
        "svd_noscope32": list(REFINEMENT_NO_SCOPE),
        "structured_ridge": list(REFINEMENT_STRUCTURED),
    }[recipe]
    _check_equal(metadata.get("features"), allowed, f"{label}.feature allowlist")
    return prediction, metadata, _sha_bytes(archive), _sha_bytes(metadata_path)


def verify_stage4(root: Path = PROJECT_ROOT, *, write: bool = True) -> dict[str, Any]:
    """Verify the three fixed Stage4 hypotheses from saved vectors only."""
    root = Path(root).resolve()
    config = root / "experiments" / "regression_research"
    report = root / "reports" / "regression_research"
    cache = root / "experiments" / "runs" / "regression_research_v1"
    freeze = _load_json(config / "stage4_refinement_freeze.json")
    data_files = task_files("regression")
    if any(not data_files.get(key) for key in ("train", "test", "sample")):
        raise VerificationError("frozen regression Train, Test, and Sample files are required")
    freeze_check = _verify_freeze(freeze, root, {key: Path(data_files[key]) for key in ("train", "test", "sample")})
    _check_equal(freeze.get("stage"), "stage4_refinement", "Stage4 freeze stage")
    protocol = freeze["identity"]["protocol"]
    _check_equal(tuple(protocol["recipes"]), REFINEMENT_RECIPES, "Stage4 recipe contract")
    _check_equal(tuple(protocol["outer_policies"]), REFINEMENT_POLICIES, "Stage4 outer policy contract")
    _check_equal(tuple(protocol["rank_weights"]), INNER_POLICY_WEIGHTS, "Stage4 rank-weight contract")
    result_path = report / "stage4_refinement.json"
    result = _load_json(result_path)
    _check_equal(result.get("stage"), "stage4_refinement", "Stage4 result stage")
    _check_equal(result.get("freeze_signature"), freeze.get("signature"), "Stage4 result freeze signature")
    r2_protocol = _load_json(root / "experiments" / "regression_round2" / "protocol.json")
    windows = r2_protocol["primary_windows"] + r2_protocol["secondary_windows"]
    ends = sorted({int(window["fit"][1]) for window in windows})
    frame = pd.read_csv(data_files["train"])
    _check_equal(len(result.get("inner", [])), len(ends) * len(REFINEMENT_RECIPES), "Stage4 inner cells")
    inner_index = {(int(item["fit_end"]), str(item["recipe"])): item for item in result["inner"]}
    _check_equal(set(inner_index), {(end, recipe) for end in ends for recipe in REFINEMENT_RECIPES},
                 "Stage4 inner cell keys")
    _check_equal(set(result.get("audits", {})), {str(end) for end in ends}, "Stage4 audit cutoffs")
    inner_selections: dict[tuple[int, str], dict[str, Any]] = {}
    model_artifacts: dict[str, str] = {}
    evaluation_artifacts: dict[str, str] = {}
    choices_checked = 0
    model_refs_checked = 0
    evaluation_arrays_checked = 0
    row_arrays_checked = 0
    model_vectors: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for end in ends:
        inner_fit = _positions(frame, 47, end - 4)
        inner_validation = _positions(frame, end - 3, end)
        outer_fit = _positions(frame, 47, end)
        last, outer_validation = _stage23_outer_rows(frame, r2_protocol, end)
        if not (frame.iloc[inner_fit].cycle_code.max() < frame.iloc[inner_validation].cycle_code.min()
                <= end < frame.iloc[outer_validation].cycle_code.min()):
            raise VerificationError(f"Stage4 split is not strictly past at cutoff {end}")
        for recipe in REFINEMENT_RECIPES:
            item = inner_index[(end, recipe)]
            audit = result["audits"][str(end)][recipe]
            inner_raw, inner_meta, inner_npz_hash, inner_json_hash = _load_refinement_model(
                cache, frame, recipe, inner_fit, inner_validation, freeze["signature"]
            )
            outer_raw, outer_meta, outer_npz_hash, outer_json_hash = _load_refinement_model(
                cache, frame, recipe, outer_fit, outer_validation, freeze["signature"]
            )
            _check_equal(audit.get("inner"), inner_meta, f"Stage4.{end}.{recipe}.inner metadata")
            _check_equal(audit.get("outer"), outer_meta, f"Stage4.{end}.{recipe}.outer metadata")
            for meta, npz_hash, json_hash in ((inner_meta, inner_npz_hash, inner_json_hash),
                                               (outer_meta, outer_npz_hash, outer_json_hash)):
                model_artifacts[f"{meta['kind']}__fit{meta['fit_rows']}__{npz_hash[:12]}.npz"] = npz_hash
                model_artifacts[f"{meta['kind']}__{meta['fit_positions_sha256'][:12]}__{json_hash[:12]}.json"] = json_hash
                model_refs_checked += 1
                row_arrays_checked += 2
            y_inner = frame.iloc[inner_validation]["threat_rank"].to_numpy(dtype=np.float64)
            cycle_inner = frame.iloc[inner_validation]["cycle_code"].to_numpy()
            expected_choices = []
            for weight in INNER_POLICY_WEIGHTS:
                mapped = _rank_weight_prediction(inner_raw, cycle_inner, weight)
                mae = math.fsum(float(value) for value in np.abs(mapped - y_inner)) / len(y_inner)
                expected_choices.append({"rank_weight": float(weight), "mae": mae})
            saved_choices = item.get("choices", [])
            _check_equal(len(saved_choices), len(expected_choices), f"Stage4.{end}.{recipe}.choice count")
            for index, (saved, expected) in enumerate(zip(saved_choices, expected_choices)):
                _close(saved.get("rank_weight"), expected["rank_weight"],
                       f"Stage4.{end}.{recipe}.choice[{index}].weight")
                _close(saved.get("mae"), expected["mae"], f"Stage4.{end}.{recipe}.choice[{index}].mae")
                choices_checked += 1
            best = min(choice["mae"] for choice in expected_choices)
            chosen = min((choice for choice in expected_choices if choice["mae"] <= best + 1e-10),
                         key=lambda choice: choice["rank_weight"])
            for key in ("rank_weight", "mae"):
                _close(item.get("selected", {}).get(key), chosen[key], f"Stage4.{end}.{recipe}.selected.{key}")
                _close(audit.get("selection", {}).get(key), chosen[key], f"Stage4 audit.{end}.{recipe}.selection.{key}")
            inner_selections[(end, recipe)] = {"rank_weight": chosen["rank_weight"], "mae": chosen["mae"]}
            model_vectors[(end, recipe, "raw")] = (outer_validation, outer_raw)
            outer_cycles = frame.iloc[outer_validation]["cycle_code"].to_numpy()
            for policy_name in REFINEMENT_POLICIES:
                expected_full = (outer_raw.copy() if policy_name == "raw" else
                                 _rank_weight_prediction(outer_raw, outer_cycles, chosen["rank_weight"]))
                eval_path = cache / f"eval_refine__fit{end}__{recipe}__{policy_name}.npz"
                if not eval_path.is_file():
                    raise VerificationError(f"missing Stage4 evaluation array: {eval_path.name}")
                with np.load(eval_path, allow_pickle=False) as saved:
                    if set(saved.files) != {"validation_rows", "prediction"}:
                        raise VerificationError(f"unexpected Stage4 evaluation NPZ members: {eval_path.name}")
                    saved_rows, saved_prediction = saved["validation_rows"].copy(), saved["prediction"].copy()
                if not np.array_equal(saved_rows, outer_validation) or not np.array_equal(saved_prediction, expected_full):
                    raise VerificationError(f"Stage4 mapped vector differs from independent reconstruction: {eval_path.name}")
                evaluation_artifacts[eval_path.name] = _sha_bytes(eval_path)
                model_vectors[(end, recipe, policy_name)] = (saved_rows, expected_full)
                evaluation_arrays_checked += 1

    expected_jobs = {
        (recipe, policy, family, str(window["id"]))
        for family, family_windows in (("primary", r2_protocol["primary_windows"]),
                                      ("secondary", r2_protocol["secondary_windows"]))
        for window in family_windows for recipe in REFINEMENT_RECIPES for policy in REFINEMENT_POLICIES
    }
    job_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for job in result.get("jobs", []):
        key = (str(job["recipe"]), str(job["policy"]), str(job["family"]), str(job["window"]))
        if key in job_index:
            raise VerificationError(f"duplicate Stage4 job: {key}")
        job_index[key] = job
    _check_equal(set(job_index), expected_jobs, "Stage4 per-window jobs")
    summary_counts = {"metric_nodes": 0, "cycle_nodes": 0, "subgroup_nodes": 0,
                      "impact_group_count_nodes": 0}
    jobs_checked = 0
    for family, family_windows in (("primary", r2_protocol["primary_windows"]),
                                   ("secondary", r2_protocol["secondary_windows"])):
        for window in family_windows:
            lower, upper = map(int, window["validation"])
            end, window_id = int(window["fit"][1]), str(window["id"])
            for recipe in REFINEMENT_RECIPES:
                for policy in REFINEMENT_POLICIES:
                    rows, full_prediction = model_vectors[(end, recipe, policy)]
                    keep = frame.iloc[rows]["cycle_code"].between(lower, upper).to_numpy()
                    selected_rows = rows[keep]
                    prediction = full_prediction[keep]
                    expected_rows = _positions(frame, lower, upper)
                    if not np.array_equal(selected_rows, expected_rows):
                        raise VerificationError(f"Stage4 per-window row positions differ: {family}/{window_id}/{recipe}/{policy}")
                    job = job_index[(recipe, policy, family, window_id)]
                    _check_equal(job.get("prediction_sha256"), _array_digest(prediction, "<f8"),
                                 f"Stage4 job prediction hash {family}/{window_id}/{recipe}/{policy}")
                    checks = verify_summary_node(frame.iloc[selected_rows].reset_index(drop=True), prediction,
                                                 job, f"Stage4.{family}.{window_id}.{recipe}.{policy}")
                    for key, count in checks.items():
                        summary_counts[key] += count
                    jobs_checked += 1
    aggregate_cells = 0
    for family, family_windows in (("primary", r2_protocol["primary_windows"]),
                                   ("secondary", r2_protocol["secondary_windows"])):
        family_aggregate = result.get("aggregate", {}).get(family, {})
        expected_aggregate = {f"{recipe}__{policy}" for recipe in REFINEMENT_RECIPES for policy in REFINEMENT_POLICIES}
        _check_equal(set(family_aggregate), expected_aggregate, f"Stage4 aggregate.{family}.keys")
        for recipe in REFINEMENT_RECIPES:
            for policy in REFINEMENT_POLICIES:
                row_parts, pred_parts = [], []
                for window in family_windows:
                    end = int(window["fit"][1])
                    rows, prediction = model_vectors[(end, recipe, policy)]
                    lo, hi = map(int, window["validation"])
                    keep = frame.iloc[rows]["cycle_code"].between(lo, hi).to_numpy()
                    row_parts.append(rows[keep]); pred_parts.append(prediction[keep])
                rows = np.concatenate(row_parts)
                prediction = np.concatenate(pred_parts)
                if len(np.unique(rows)) != len(rows):
                    raise VerificationError(f"Stage4 aggregate validation rows overlap: {family}/{recipe}/{policy}")
                order = np.argsort(rows, kind="stable")
                aggregate_frame = frame.iloc[rows[order]].reset_index(drop=True)
                summary = family_aggregate[f"{recipe}__{policy}"]
                checks = verify_summary_node(aggregate_frame, prediction[order], summary,
                                             f"Stage4.aggregate.{family}.{recipe}.{policy}", impact_mode="counts")
                for key, count in checks.items():
                    summary_counts[key] += count
                aggregate_cells += 1

    receipt = {
        "stage": "stage4_refinement_verification",
        "status": "PASS",
        "freeze_signature": freeze["signature"],
        "artifacts": {
            "verifier_code_sha256": _text_digest(Path(__file__)),
            "stage4_result_sha256": _sha_bytes(result_path),
            "training_csv_sha256": _sha_bytes(Path(data_files["train"])),
            "refinement_model_files": model_artifacts,
            "evaluation_cache_files": evaluation_artifacts,
        },
        "checks": {
            "freeze": freeze_check,
            "fit_cache_references_checked": model_refs_checked,
            "unique_fit_cache_files_checked": len(model_artifacts) // 2,
            "fit_and_validation_position_arrays_checked": row_arrays_checked,
            "inner_selection_cells_checked": len(inner_selections),
            "inner_choice_scores_checked": choices_checked,
            "evaluation_mapping_arrays_checked": evaluation_arrays_checked,
            "outer_score_jobs_checked": jobs_checked,
            "aggregate_cells_checked": aggregate_cells,
            "metric_nodes_checked": summary_counts["metric_nodes"],
            "cycle_metric_nodes_checked": summary_counts["cycle_nodes"],
            "subgroup_metric_nodes_checked": summary_counts["subgroup_nodes"],
            "impact_group_support_nodes_checked": summary_counts["impact_group_count_nodes"],
            "fit_fingerprints_recomputed_from_allowed_train_columns_and_targets": True,
            "validation_labels_passed_to_fit": False,
            "test_features_used_for_fit_or_selection": False,
            "official_test_labels_or_features_read": False,
            "model_refits_or_predictions_run": False,
        },
        "notes": [
            "Inner rank-weight choices are recomputed from saved inner vectors and past labels; selection is checked against the lower-weight tie rule.",
            "Outer raw and inner-mapped arrays, fit fingerprints, state hashes, row positions, and output metrics are independently reconciled.",
            "Impact-group counts and target-region support are checked; full impact-group metric distributions are not independently recomputed.",
            "No model was refit and official test rows were not parsed.",
        ],
    }
    if write:
        output = report / "verification_stage4.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def _load_lgb_text_model(cache: Path, frame: pd.DataFrame, param: str,
                         fit_rows: np.ndarray, validation_rows: np.ndarray,
                         freeze_signature: str) -> tuple[np.ndarray, dict[str, Any], str, str]:
    fit_end = int(frame.iloc[fit_rows].cycle_code.max())
    validation_end = int(frame.iloc[validation_rows].cycle_code.max())
    label = f"lgbtext__{param}__fit{fit_end}__to{validation_end}"
    archive, metadata_path = cache / f"{label}.npz", cache / f"{label}.json"
    if not archive.is_file() or not metadata_path.is_file():
        raise VerificationError(f"missing Stage5 model cache pair: {label}")
    metadata = _load_json(metadata_path)
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"fit_rows", "validation_rows", "prediction"}:
            raise VerificationError(f"unexpected Stage5 model NPZ members: {label}")
        saved_fit, saved_validation, prediction = (
            data[name].copy() for name in ("fit_rows", "validation_rows", "prediction")
        )
    if not np.array_equal(saved_fit, fit_rows) or not np.array_equal(saved_validation, validation_rows):
        raise VerificationError(f"Stage5 fit/validation positions differ from frozen split: {label}")
    if prediction.ndim != 1 or prediction.shape != validation_rows.shape or not np.isfinite(prediction).all():
        raise VerificationError(f"Stage5 prediction vector is invalid: {label}")
    fit_hash, validation_hash = _array_digest(saved_fit, "<i8"), _array_digest(saved_validation, "<i8")
    prediction_hash = _array_digest(prediction, "<f8")
    _check_equal(metadata.get("fit_positions_sha256"), fit_hash, f"{label}.fit positions hash")
    _check_equal(metadata.get("validation_positions_sha256"), validation_hash, f"{label}.validation positions hash")
    _check_equal(metadata.get("prediction_sha256"), prediction_hash, f"{label}.prediction hash")
    _check_equal(metadata.get("fit_rows"), len(fit_rows), f"{label}.fit_rows")
    _check_equal(metadata.get("param"), param, f"{label}.param")
    _check_equal(metadata.get("key"), _json_digest({
        "freeze": freeze_signature, "param": param, "fit": fit_hash, "validation": validation_hash,
    }), f"{label}.cache key")
    _check_equal(metadata.get("validation_labels_passed_to_fit"), False,
                 f"{label}.validation_labels_passed_to_fit")
    metadata_features = list(REFINEMENT_HYBRID)
    _check_equal(metadata.get("features"), metadata_features, f"{label}.feature allowlist")
    state = metadata.get("state", {})
    _check_equal(metadata.get("state_sha256"), _json_digest(state), f"{label}.state_sha256")
    state_copy = dict(state)
    nested_hash = state_copy.pop("state_sha256", None)
    _check_equal(nested_hash, _compact_json_digest(state_copy), f"{label}.nested state hash")
    _check_equal(state.get("input_feature_names"), metadata_features, f"{label}.state feature allowlist")
    _check_equal(state.get("params_id"), param, f"{label}.state.params_id")
    _check_equal(state.get("objective"), "regression_l1", f"{label}.objective")
    _check_equal(state.get("fit_rows"), len(fit_rows), f"{label}.state.fit_rows")
    _check_equal(state.get("fit_data_sha256"), _fit_fingerprint(frame, fit_rows, tuple(metadata_features)),
                 f"{label}.fit_data_sha256")
    _check_equal(state.get("target_sha256"), _array_digest(
        frame.iloc[fit_rows]["threat_rank"].to_numpy(dtype=np.float64), "<f8"
    ), f"{label}.target_sha256")
    _check_equal(state.get("validation_data_passed_to_fit"), False,
                 f"{label}.validation_data_passed_to_fit")
    _check_equal(state.get("early_stopping_used"), False, f"{label}.early_stopping_used")
    _check_equal(state.get("test_data_used_for_fit"), False, f"{label}.test_data_used_for_fit")
    expected_params = {
        "base": {**LGB_COMMON_PARAMS, **LGB_CANDIDATE_PARAMS["base"]},
        "regularized": {**LGB_COMMON_PARAMS, **LGB_CANDIDATE_PARAMS["regularized"]},
    }[param]
    _check_equal(state.get("actual_hyperparameters"), expected_params, f"{label}.actual_hyperparameters")
    _check_equal(state.get("categorical_negative_values_are_missing"), True,
                 f"{label}.categorical negative-value policy")
    preprocessing = state.get("structured_preprocessing", {})
    names = preprocessing.get("transformed_feature_names", [])
    structured_cats = set(preprocessing.get("categorical_feature_names", []))
    expected_indices = [i for i, name in enumerate(names) if name in structured_cats]
    _check_equal(state.get("categorical_feature_indices"), expected_indices,
                 f"{label}.categorical feature indices")
    _check_equal(state.get("categorical_feature_argument"), expected_indices,
                 f"{label}.categorical feature argument")
    _check_equal(state.get("structured_categorical_mask"),
                 [i in expected_indices for i in range(len(names))], f"{label}.structured categorical mask")
    svd = state.get("svd", {})
    _check_equal({k: svd.get(k) for k in ("algorithm", "n_iter", "random_state", "n_components_requested")},
                 {"algorithm": "randomized", "n_iter": 7, "random_state": 42,
                  "n_components_requested": 32}, f"{label}.SVD parameters")
    actual_components = int(svd.get("n_components_actual", -1))
    if not 0 <= actual_components <= 32:
        raise VerificationError(f"{label} SVD components exceed the frozen cap")
    tfidf = state.get("tfidf", {})
    _check_equal(tfidf.get("parameters"), {
        "ngram_range": [1, 2], "min_df": 2, "max_features": 20000,
        "sublinear_tf": True, "dtype": "float64",
    }, f"{label}.TFIDF parameters")
    if "vocabulary" in tfidf or "terms" in tfidf:
        raise VerificationError(f"{label} audit contains raw vocabulary text")
    design_names = state.get("transformed_feature_names", [])
    expected_design_names = (
        list(names)
        + [f"signal_stream_svd_{i + 1}" for i in range(actual_components)]
        + ["signal_stream_missing"]
    )
    _check_equal(design_names, expected_design_names, f"{label}.transformed feature names")
    shape = state.get("training_design_shape", [])
    if len(shape) != 2 or shape != [len(fit_rows), len(design_names)]:
        raise VerificationError(f"{label} design shape differs from fit rows/feature schema")
    for feature, group in preprocessing.get("category_counts", {}).items():
        _check_equal(group.get("training_rows"), len(fit_rows), f"{label}.{feature}.training_rows")
    return prediction, metadata, _sha_bytes(archive), _sha_bytes(metadata_path)


def verify_stage5(root: Path = PROJECT_ROOT, *, write: bool = True) -> dict[str, Any]:
    """Verify the two fixed LightGBM-by-SVD32 configurations from saved vectors."""
    root = Path(root).resolve()
    config = root / "experiments" / "regression_research"
    report = root / "reports" / "regression_research"
    cache = root / "experiments" / "runs" / "regression_research_v1"
    freeze = _load_json(config / "stage5_lgb_text_freeze.json")
    data_files = task_files("regression")
    if any(not data_files.get(key) for key in ("train", "test", "sample")):
        raise VerificationError("frozen regression Train, Test, and Sample files are required")
    freeze_check = _verify_freeze(freeze, root, {key: Path(data_files[key]) for key in ("train", "test", "sample")})
    _check_equal(freeze.get("stage"), "stage5_lgb_text", "Stage5 freeze stage")
    protocol = freeze["identity"]["protocol"]
    params = tuple(protocol["params"])
    _check_equal(params, ("base", "regularized"), "Stage5 LightGBM parameter set")
    _check_equal(tuple(protocol["rank_weights"]), INNER_POLICY_WEIGHTS, "Stage5 rank-weight grid")
    _check_equal(tuple(protocol["outer_policies"]), REFINEMENT_POLICIES, "Stage5 outer policies")
    result_path = report / "stage5_lgb_text.json"
    result = _load_json(result_path)
    _check_equal(result.get("stage"), "stage5_lgb_text", "Stage5 result stage")
    _check_equal(result.get("freeze_signature"), freeze.get("signature"), "Stage5 result freeze signature")
    r2_protocol = _load_json(root / "experiments" / "regression_round2" / "protocol.json")
    windows = r2_protocol["primary_windows"] + r2_protocol["secondary_windows"]
    ends = sorted({int(window["fit"][1]) for window in windows})
    frame = pd.read_csv(data_files["train"])
    if frame["threat_rank"].isna().any() or not np.isfinite(frame["threat_rank"].to_numpy(dtype=np.float64)).all():
        raise VerificationError("training target is missing or nonfinite")
    inner_index = {int(item["fit_end"]): item for item in result.get("inner", [])}
    _check_equal(set(inner_index), set(ends), "Stage5 inner cutoff keys")
    _check_equal(set(result.get("audits", {})), {str(end) for end in ends}, "Stage5 audit cutoff keys")
    model_hashes: dict[str, str] = {}
    eval_hashes: dict[str, str] = {}
    model_references = 0
    model_rows_checked = 0
    choice_scores_checked = 0
    selection_cells_checked = 0
    model_vectors: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    selected_weights: dict[int, dict[str, Any]] = {}
    for end in ends:
        inner_fit = _positions(frame, 47, end - 4)
        inner_validation = _positions(frame, end - 3, end)
        outer_fit = _positions(frame, 47, end)
        _, outer_validation = _stage23_outer_rows(frame, r2_protocol, end)
        if not (frame.iloc[inner_fit].cycle_code.max() < frame.iloc[inner_validation].cycle_code.min()
                <= end < frame.iloc[outer_validation].cycle_code.min()):
            raise VerificationError(f"Stage5 inner/outer split is not strictly past at {end}")
        item = inner_index[end]
        audit = result["audits"][str(end)]
        expected_choices = []
        inner_predictions: dict[str, np.ndarray] = {}
        for param in params:
            prediction, metadata, archive_hash, meta_hash = _load_lgb_text_model(
                cache, frame, param, inner_fit, inner_validation, freeze["signature"]
            )
            _check_equal(audit.get("inner", {}).get(param), metadata,
                         f"Stage5.{end}.{param}.inner metadata")
            name = f"lgbtext__{param}__fit{int(frame.iloc[inner_fit].cycle_code.max())}__to{int(frame.iloc[inner_validation].cycle_code.max())}"
            model_hashes[f"{name}.npz"] = archive_hash
            model_hashes[f"{name}.json"] = meta_hash
            model_references += 1
            model_rows_checked += 2
            inner_predictions[param] = prediction
            cycles = frame.iloc[inner_validation]["cycle_code"].to_numpy()
            y_inner = frame.iloc[inner_validation]["threat_rank"].to_numpy(dtype=np.float64)
            for weight in INNER_POLICY_WEIGHTS:
                mapped = _rank_weight_prediction(prediction, cycles, weight)
                mae = math.fsum(float(value) for value in np.abs(mapped - y_inner)) / len(y_inner)
                expected_choices.append({"param": param, "rank_weight": float(weight), "mae": mae})
        saved_choices = item.get("choices", [])
        _check_equal(len(saved_choices), len(expected_choices), f"Stage5.{end}.inner choice count")
        for index, (saved, expected) in enumerate(zip(saved_choices, expected_choices)):
            _check_equal(saved.get("param"), expected["param"], f"Stage5.{end}.choice[{index}].param")
            _close(saved.get("rank_weight"), expected["rank_weight"], f"Stage5.{end}.choice[{index}].weight")
            _close(saved.get("mae"), expected["mae"], f"Stage5.{end}.choice[{index}].mae")
            choice_scores_checked += 1
        selected = _selected_inner_choice(expected_choices)
        _check_equal(item.get("selected", {}).get("param"), selected["param"], f"Stage5.{end}.selected.param")
        _close(item.get("selected", {}).get("rank_weight"), selected["rank_weight"], f"Stage5.{end}.selected.weight")
        _close(item.get("selected", {}).get("mae"), selected["mae"], f"Stage5.{end}.selected.mae")
        _check_equal(audit.get("selection"), item["selected"], f"Stage5.{end}.audit selection")
        selected_weights[end] = selected
        outer, outer_meta, archive_hash, meta_hash = _load_lgb_text_model(
            cache, frame, selected["param"], outer_fit, outer_validation, freeze["signature"]
        )
        _check_equal(audit.get("outer"), outer_meta, f"Stage5.{end}.outer metadata")
        outer_name = f"lgbtext__{selected['param']}__fit{int(frame.iloc[outer_fit].cycle_code.max())}__to{int(frame.iloc[outer_validation].cycle_code.max())}"
        model_hashes[f"{outer_name}.npz"] = archive_hash
        model_hashes[f"{outer_name}.json"] = meta_hash
        model_references += 1
        model_rows_checked += 2
        all_cycles = frame.iloc[outer_validation]["cycle_code"].to_numpy()
        for policy in REFINEMENT_POLICIES:
            expected = outer.copy() if policy == "raw" else _rank_weight_prediction(
                outer, all_cycles, selected["rank_weight"]
            )
            eval_path = cache / f"eval_lgbtext__fit{end}__{policy}.npz"
            if not eval_path.is_file():
                raise VerificationError(f"missing Stage5 eval vector: {eval_path.name}")
            with np.load(eval_path, allow_pickle=False) as saved:
                if set(saved.files) != {"validation_rows", "prediction"}:
                    raise VerificationError(f"unexpected Stage5 eval NPZ members: {eval_path.name}")
                saved_rows, saved_prediction = saved["validation_rows"].copy(), saved["prediction"].copy()
            if not np.array_equal(saved_rows, outer_validation) or not np.array_equal(saved_prediction, expected):
                raise VerificationError(f"Stage5 mapping array mismatch: {eval_path.name}")
            eval_hashes[eval_path.name] = _sha_bytes(eval_path)
            model_vectors[(end, policy)] = (saved_rows, expected)
    jobs_expected = {
        (policy, family, str(window["id"]))
        for family, family_windows in (("primary", r2_protocol["primary_windows"]),
                                      ("secondary", r2_protocol["secondary_windows"]))
        for window in family_windows for policy in REFINEMENT_POLICIES
    }
    jobs = {}
    for job in result.get("jobs", []):
        key = (str(job["policy"]), str(job["family"]), str(job["window"]))
        if key in jobs:
            raise VerificationError(f"duplicate Stage5 score job {key}")
        if job.get("recipe") != "lgb_text":
            raise VerificationError("unexpected Stage5 scored recipe")
        jobs[key] = job
    _check_equal(set(jobs), jobs_expected, "Stage5 score jobs")
    summary_counts = {"metric_nodes": 0, "cycle_nodes": 0, "subgroup_nodes": 0,
                      "impact_group_count_nodes": 0}
    jobs_checked = 0
    for family, family_windows in (("primary", r2_protocol["primary_windows"]),
                                   ("secondary", r2_protocol["secondary_windows"])):
        for window in family_windows:
            end, window_id = int(window["fit"][1]), str(window["id"])
            low, high = map(int, window["validation"])
            for policy in REFINEMENT_POLICIES:
                rows, prediction = model_vectors[(end, policy)]
                keep = frame.iloc[rows]["cycle_code"].between(low, high).to_numpy()
                rows, prediction = rows[keep], prediction[keep]
                if not np.array_equal(rows, _positions(frame, low, high)):
                    raise VerificationError(f"Stage5 per-window rows differ: {family}/{window_id}/{policy}")
                job = jobs[(policy, family, window_id)]
                _check_equal(job.get("prediction_sha256"), _array_digest(prediction, "<f8"),
                             f"Stage5 score prediction hash {family}/{window_id}/{policy}")
                checks = verify_summary_node(frame.iloc[rows].reset_index(drop=True), prediction, job,
                                             f"Stage5.{family}.{window_id}.{policy}")
                for key, count in checks.items():
                    summary_counts[key] += count
                jobs_checked += 1
    aggregate_cells = 0
    for family, family_windows in (("primary", r2_protocol["primary_windows"]),
                                   ("secondary", r2_protocol["secondary_windows"])):
        family_aggregate = result.get("aggregate", {}).get(family, {})
        _check_equal(set(family_aggregate), {f"lgb_text__{policy}" for policy in REFINEMENT_POLICIES},
                     f"Stage5 aggregate.{family}.keys")
        for policy in REFINEMENT_POLICIES:
            row_parts, prediction_parts = [], []
            for window in family_windows:
                end = int(window["fit"][1])
                rows, prediction = model_vectors[(end, policy)]
                low, high = map(int, window["validation"])
                keep = frame.iloc[rows]["cycle_code"].between(low, high).to_numpy()
                row_parts.append(rows[keep]); prediction_parts.append(prediction[keep])
            rows = np.concatenate(row_parts); prediction = np.concatenate(prediction_parts)
            if len(np.unique(rows)) != len(rows):
                raise VerificationError(f"Stage5 aggregate rows overlap: {family}/{policy}")
            order = np.argsort(rows, kind="stable")
            summary = family_aggregate[f"lgb_text__{policy}"]
            checks = verify_summary_node(
                frame.iloc[rows[order]].reset_index(drop=True), prediction[order], summary,
                f"Stage5.aggregate.{family}.{policy}", impact_mode="counts"
            )
            for key, count in checks.items():
                summary_counts[key] += count
            aggregate_cells += 1
    receipt = {
        "stage": "stage5_lgb_text_verification",
        "status": "PASS",
        "freeze_signature": freeze["signature"],
        "artifacts": {
            "verifier_code_sha256": _text_digest(Path(__file__)),
            "stage5_result_sha256": _sha_bytes(result_path),
            "training_csv_sha256": _sha_bytes(Path(data_files["train"])),
            "model_cache_files": model_hashes,
            "evaluation_cache_files": eval_hashes,
        },
        "checks": {
            "freeze": freeze_check,
            "fit_cache_references_checked": model_references,
            "unique_fit_cache_files_checked": len(model_hashes) // 2,
            "fit_and_validation_position_arrays_checked": model_rows_checked,
            "inner_selection_cells_checked": len(selected_weights),
            "inner_choice_scores_checked": choice_scores_checked,
            "evaluation_mapping_arrays_checked": len(eval_hashes),
            "outer_score_jobs_checked": jobs_checked,
            "aggregate_cells_checked": aggregate_cells,
            "metric_nodes_checked": summary_counts["metric_nodes"],
            "cycle_metric_nodes_checked": summary_counts["cycle_nodes"],
            "subgroup_metric_nodes_checked": summary_counts["subgroup_nodes"],
            "impact_group_support_nodes_checked": summary_counts["impact_group_count_nodes"],
            "fit_fingerprints_recomputed_from_allowed_train_columns_and_targets": True,
            "validation_labels_passed_to_fit": False,
            "test_features_used_for_fit_or_selection": False,
            "official_test_labels_or_features_read": False,
            "model_refits_or_predictions_run": False,
        },
        "notes": [
            "Both fixed LightGBM parameter configurations are audited over the same SVD32 representation; no new parameter grid is introduced.",
            "Inner MAEs and tie-rule selection are recomputed from saved vectors, then the selected model's outer mapping and metrics are reconciled.",
            "Impact-group support counts and target-region counts are checked; full impact-group metric distributions are not independently recomputed.",
            "No model was refit and official test rows were not parsed.",
        ],
    }
    if write:
        output = report / "verification_stage5.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Independently verify saved regression research results.")
    parser.add_argument("--stage", choices=("all", "stage1", "stage23", "stage4", "stage5"), default="all")
    args = parser.parse_args()
    receipts = {}
    if args.stage in {"all", "stage1"}:
        receipts["stage1"] = verify_stage1()
    if args.stage in {"all", "stage23"}:
        receipts["stage23"] = verify_stage23()
    if args.stage in {"all", "stage4"}:
        receipts["stage4"] = verify_stage4()
    if args.stage in {"all", "stage5"}:
        receipts["stage5"] = verify_stage5()
    print(json.dumps({name: {"status": receipt["status"], "checks": receipt["checks"]}
                      for name, receipt in receipts.items()}, indent=2))


if __name__ == "__main__":
    main()
