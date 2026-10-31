"""Single E=67, past-only regression residual-head pilot; never reads Test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm
import scipy
import sklearn
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.optimization_v3_residual import OOFResidualHead  # noqa: E402
from src.models.round1_models import EXPANDED_FEATURES  # noqa: E402
from src.utils.paths import data_root  # noqa: E402

EXP_DIR = ROOT / "experiments/optimization_v3/regression_residual"
REPORT_DIR = ROOT / "reports/optimization_v3/regression_residual"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_reg_residual"
PROTOCOL_PATH = EXP_DIR / "protocol.json"
FREEZE_PATH = EXP_DIR / "freeze.json"
REPORT_PATH = REPORT_DIR / "pilot.json"
PRED_PATH = RUN_DIR / "predictions.npz"
RECEIPT_PATH = RUN_DIR / "run_receipt.json"
TRAIN_PATH = data_root() / "regression" / "Train.csv"
BASE_CACHE_PATH = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
PILOT_CACHE_PATH = ROOT / "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz"
FULL_CACHE_PATH = ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
DIAGNOSTIC_CACHE_PATH = ROOT / "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz"
FULL_REPORT_PATH = ROOT / "reports/optimization_v2/regression_repr/fullsix.json"
FULL_FREEZE_PATH = ROOT / "experiments/optimization_v2/regression_repr/fullsix_freeze.json"
PILOT_FREEZE_PATH = ROOT / "experiments/optimization_v2/regression_repr/freeze.json"
DUAL_REPORT_PATH = ROOT / "reports/dual_campaign/regression/stage1_paired_blend.json"
DUAL_FREEZE_PATH = ROOT / "experiments/dual_campaign/regression/freeze.json"
DUAL_VERIFICATION_PATH = ROOT / "reports/dual_campaign/regression/verification.json"
PILOT_VERIFICATION_PATH = ROOT / "reports/optimization_v2/regression_repr/verification.json"
FULL_VERIFICATION_PATH = ROOT / "reports/optimization_v2/regression_repr/fullsix_verification_context.json"
SCORE_NAMES = ("champion", "lgb_svd", "ridge", "lgb_sparse")
POLICY_CUTOFFS = (55, 59, 63, 67)
FIXED_INPUTS = (
    "scripts/optimization_v3_reg_residual.py",
    "scripts/optimization_v3_verify_reg_residual.py",
    "tests/test_optimization_v3_reg_residual.py",
    "src/models/optimization_v3_residual.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/campaign_regression.py",
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.kind in "OUS":
        payload = json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _current_components(frame: pd.DataFrame, base: Any, pilot: Any, full: Any,
                        cutoff: int, low_cycle: int, high_cycle: int) -> dict[str, np.ndarray]:
    """Rebuild the frozen mapped OOF components, with exact cycle-window row checks."""
    sparse_cache = pilot if cutoff in (55, 67) else full
    all_rows = np.asarray(sparse_cache[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
    raw_sparse = np.asarray(sparse_cache[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
    cycles_all = frame.iloc[all_rows].cycle_code.to_numpy(dtype=int)
    if cutoff == 55 and (cycles_all.min() != 56 or cycles_all.max() != 67):
        raise ValueError("cutoff55 sparse cache must cover cycles56-67")
    if cutoff == 67 and (cycles_all.min() != 68 or cycles_all.max() != 79):
        raise ValueError("cutoff67 sparse cache must cover cycles68-79")
    selected = (cycles_all >= low_cycle) & (cycles_all <= high_cycle)
    rows = all_rows[selected]
    expected_rows = np.flatnonzero(frame.cycle_code.between(low_cycle, high_cycle).to_numpy())
    if not np.array_equal(rows, expected_rows):
        raise ValueError(f"cutoff{cutoff} positions do not equal exact cycle range {low_cycle}-{high_cycle}")
    if not np.array_equal(all_rows, np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)):
        raise ValueError(f"cutoff{cutoff} row identities differ between champion components")
    use = selected
    raw_sparse = raw_sparse[use]
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    sparse_rank50 = np.empty(len(rows), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        sparse_rank50[mask] = 0.5 * raw_sparse[mask] + 0.5 * 100.0 * rankdata(
            raw_sparse[mask], method="average") / int(mask.sum())
    sparse_rank50 = np.clip(sparse_rank50, 0.0, 100.0)

    old_rows = np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)
    old_cycles = frame.iloc[old_rows].cycle_code.to_numpy(dtype=int)
    old_selected = (old_cycles >= low_cycle) & (old_cycles <= high_cycle)
    lgb_svd = np.asarray(base[f"lgb_{cutoff}"], dtype=np.float64)[old_selected]
    ridge = np.asarray(base[f"ridge_{cutoff}"], dtype=np.float64)[old_selected]
    ridge25 = np.asarray(base[f"blend_ridge25_{cutoff}"], dtype=np.float64)[old_selected]
    if not np.allclose(ridge25, np.clip(0.75 * lgb_svd + 0.25 * ridge, 0, 100), rtol=0, atol=1e-10):
        raise ValueError(f"cutoff{cutoff} Ridge25 component blend identity does not verify")
    champion = np.clip(0.75 * ridge25 + 0.25 * sparse_rank50, 0.0, 100.0)
    return {"rows": rows, "champion": champion, "lgb_svd": lgb_svd,
            "ridge": ridge, "lgb_sparse": sparse_rank50,
            "raw_sparse_lgb": raw_sparse}


def _stack_blocks(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    merged = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    order = np.argsort(merged["rows"], kind="stable")
    return {key: value[order] for key, value in merged.items()}


def _assemble(frame: pd.DataFrame) -> dict[str, Any]:
    for path in (TRAIN_PATH, BASE_CACHE_PATH, PILOT_CACHE_PATH, FULL_CACHE_PATH,
                 DIAGNOSTIC_CACHE_PATH, FULL_REPORT_PATH, FULL_FREEZE_PATH, PILOT_FREEZE_PATH,
                 DUAL_REPORT_PATH, DUAL_FREEZE_PATH, DUAL_VERIFICATION_PATH,
                 PILOT_VERIFICATION_PATH, FULL_VERIFICATION_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha_file(TRAIN_PATH) != "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082":
        raise ValueError("Official Regression Train SHA differs from frozen input")
    base = np.load(BASE_CACHE_PATH, allow_pickle=False)
    pilot = np.load(PILOT_CACHE_PATH, allow_pickle=False)
    full = np.load(FULL_CACHE_PATH, allow_pickle=False)
    diagnostic = np.load(DIAGNOSTIC_CACHE_PATH, allow_pickle=False)
    dual_report = json.loads(DUAL_REPORT_PATH.read_text(encoding="utf-8"))
    dual_verification = json.loads(DUAL_VERIFICATION_PATH.read_text(encoding="utf-8"))
    pilot_verification = json.loads(PILOT_VERIFICATION_PATH.read_text(encoding="utf-8"))
    full_verification = json.loads(FULL_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if dual_verification.get("status") != "PASS" or dual_report["derived_prediction_cache"]["sha256"] != sha_file(BASE_CACHE_PATH):
        raise ValueError("Frozen Ridge25 mapped-component cache receipt/hash mismatch")
    if pilot_verification.get("status") != "PASS" or pilot_verification.get("prediction_cache_sha256") != sha_file(PILOT_CACHE_PATH):
        raise ValueError("Frozen V2 pilot prediction-cache receipt/hash mismatch")
    if full_verification.get("status") != "PASS" or full_verification.get("prediction_cache_sha256") != sha_file(FULL_CACHE_PATH):
        raise ValueError("Frozen V2 full-six prediction-cache receipt/hash mismatch")

    train_blocks = [
        _current_components(frame, base, pilot, full, 55, 56, 59),
        _current_components(frame, base, pilot, full, 59, 60, 63),
        _current_components(frame, base, pilot, full, 63, 64, 67),
    ]
    train = _stack_blocks(train_blocks)
    h12_early = _current_components(frame, base, pilot, full, 55, 56, 67)
    w1 = _current_components(frame, base, pilot, full, 55, 56, 59)
    h12_late = _current_components(frame, base, pilot, full, 67, 68, 79)
    w4 = _current_components(frame, base, pilot, full, 67, 68, 71)
    if len(train["rows"]) != 17685 or len(np.unique(train["rows"])) != 17685:
        raise ValueError("Residual head training support must be 17,685 unique OOF rows")
    if not np.array_equal(train["rows"], h12_early["rows"]):
        raise ValueError("Residual training rows must equal H12_E cycles56-67")
    if len(h12_late["rows"]) != 20903 or len(w4["rows"]) != 4950:
        raise ValueError("Expected validation support differs from frozen cycle windows")
    if not np.array_equal(w1["rows"], h12_early["rows"][frame.iloc[h12_early["rows"]].cycle_code.to_numpy() <= 59]):
        raise ValueError("W1 must be an exact source-order subset of H12_E")
    if not np.array_equal(w4["rows"], h12_late["rows"][frame.iloc[h12_late["rows"]].cycle_code.to_numpy() <= 71]):
        raise ValueError("W4 must be an exact source-order subset of H12_L")

    # Independent source comparison against the prior cache-only diagnostic receipt.
    primary_rows = np.asarray(diagnostic["primary_rows"], dtype=np.int64)
    primary_components = np.column_stack([
        diagnostic["primary_prediction"], diagnostic["primary_lgb_svd32"],
        diagnostic["primary_hybrid_ridge"], diagnostic["primary_sparse_lgb_rank50"]])
    if len(np.unique(primary_rows)) != len(primary_rows):
        raise ValueError("Derived OOF diagnostic cache contains duplicate row IDs")
    primary_map = {int(row): i for i, row in enumerate(primary_rows)}
    train_ix = np.asarray([primary_map[int(row)] for row in train["rows"]], dtype=int)
    train_components = np.column_stack([train[name] for name in SCORE_NAMES])
    if not np.allclose(primary_components[train_ix], train_components, rtol=0, atol=1e-12):
        raise ValueError("Training OOF components differ from row-exact primary diagnostic arrays")
    h12_rows = np.asarray(diagnostic["h12_rows"], dtype=np.int64)
    h12_components = np.column_stack([
        diagnostic["h12_prediction"], diagnostic["h12_lgb_svd32"],
        diagnostic["h12_hybrid_ridge"], diagnostic["h12_sparse_lgb_rank50"]])
    if len(np.unique(h12_rows)) != len(h12_rows):
        raise ValueError("Derived H12 diagnostic cache contains duplicate row IDs")
    h12_map = {int(row): i for i, row in enumerate(h12_rows)}
    late_ix = np.asarray([h12_map[int(row)] for row in h12_late["rows"]], dtype=int)
    late_components = np.column_stack([h12_late[name] for name in SCORE_NAMES])
    if not np.allclose(h12_components[late_ix], late_components, rtol=0, atol=1e-12):
        raise ValueError("H12_L evaluation components differ from row-exact H12 diagnostic arrays")
    return {"train": train, "w1": w1, "h12_early": h12_early, "w4": w4,
            "h12_late": h12_late, "frame": frame, "source_cache_hashes": {
                "train": sha_file(TRAIN_PATH), "dual": sha_file(BASE_CACHE_PATH),
                "pilot": sha_file(PILOT_CACHE_PATH), "fullsix": sha_file(FULL_CACHE_PATH),
                "diagnostic_oof": sha_file(DIAGNOSTIC_CACHE_PATH)},
            "fold_manifests": _fold_manifests(train_blocks + [h12_late], frame)}


def _fold_manifests(blocks: list[dict[str, np.ndarray]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for block in blocks:
        rows = block["rows"]
        score = np.column_stack([block[name] for name in SCORE_NAMES])
        cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
        cutoff = int(cycles.min()) - 1
        output.append({"outer_fit_cutoff": cutoff,
                       "cycle_range": [int(cycles.min()), int(cycles.max())],
                       "rows": int(len(rows)), "row_positions_sha256": sha_array(rows),
                       "score_matrix_sha256": sha_array(score),
                       "target_sha256": sha_array(frame.iloc[rows].threat_rank.to_numpy(float)),
                       "score_names": list(SCORE_NAMES)})
    return output


def _summary(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = np.asarray(pred, float) - np.asarray(y, float)
    abs_error = np.abs(error)
    return {"n": int(len(y)), "mae": float(abs_error.mean()),
            "bias_prediction_minus_target": float(error.mean()),
            "median_signed_residual": float(np.median(error)),
            "over_rate": float(np.mean(error > 0)), "under_rate": float(np.mean(error < 0)),
            "p90_absolute_error": float(np.quantile(abs_error, .90)),
            "p95_absolute_error": float(np.quantile(abs_error, .95))}


def _group_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for name, mask in {
        "EF0": frame.entry_friction.to_numpy() == 0,
        "EF1": frame.entry_friction.to_numpy() == 1,
        "stream_missing": frame.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
        "stream_present": ~frame.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
        "tail_le5": frame.threat_rank.to_numpy(float) <= 5,
        "tail_ge95": frame.threat_rank.to_numpy(float) >= 95,
        "middle": (frame.threat_rank.to_numpy(float) > 5) & (frame.threat_rank.to_numpy(float) < 95),
    }.items():
        if np.any(mask):
            result[name] = int(mask.sum())
    return result


def _comparison(frame: pd.DataFrame, base_prediction: np.ndarray,
                candidate_prediction: np.ndarray) -> dict[str, Any]:
    target = frame.threat_rank.to_numpy(dtype=float)
    base = _summary(target, base_prediction)
    candidate = _summary(target, candidate_prediction)
    groups = {}
    rules = {
        "EF0": frame.entry_friction.to_numpy() == 0,
        "EF1": frame.entry_friction.to_numpy() == 1,
        "stream_missing": frame.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
        "stream_present": ~frame.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
        "tail_le5": target <= 5,
        "tail_ge95": target >= 95,
        "middle": (target > 5) & (target < 95),
    }
    for name, mask in rules.items():
        if mask.any():
            groups[name] = {"n": int(mask.sum()), "baseline": _summary(target[mask], base_prediction[mask]),
                            "candidate": _summary(target[mask], candidate_prediction[mask])}
    return {"baseline": base, "candidate": candidate,
            "delta_mae_candidate_minus_baseline": candidate["mae"] - base["mae"],
            "groups": groups}


def _input_hashes() -> dict[str, str]:
    paths = [ROOT / rel for rel in FIXED_INPUTS]
    paths += [TRAIN_PATH, BASE_CACHE_PATH, PILOT_CACHE_PATH, FULL_CACHE_PATH, DIAGNOSTIC_CACHE_PATH,
              FULL_REPORT_PATH, FULL_FREEZE_PATH, PILOT_FREEZE_PATH, DUAL_REPORT_PATH,
              DUAL_FREEZE_PATH, DUAL_VERIFICATION_PATH, PILOT_VERIFICATION_PATH,
              FULL_VERIFICATION_PATH]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha_file(path) for path in paths}


def _protocol() -> dict[str, Any]:
    return {
        "experiment_id": "optimization_v3_regression_residual_head_E67",
        "purpose": "One frozen temporal pilot of a shallow L1 residual correction head on the V2 three-component champion.",
        "model": {"class": "src.models.optimization_v3_residual.OOFResidualHead",
                  "task": "regression", "score_names": list(SCORE_NAMES),
                  "input_features": list(EXPANDED_FEATURES),
                  "utility_fixed_parameters": {"objective": "regression_l1", "n_estimators": 200,
                      "num_leaves": 7, "min_child_samples": 80, "reg_lambda": 10.0,
                      "learning_rate": 0.03, "random_state": 42, "n_jobs": 2,
                      "shrinkage": 0.5},
                  "fit_count": 1, "fit_cutoff": 67},
        "incumbent": {"recipe": "0.5625 LGB SVD32 + 0.1875 Hybrid Ridge + 0.25 sparse LGB rank50",
                      "source": "original row-exact mapped component caches; no end-51 synthetic champion",
                      "sparse_policy": "within-cycle average-rank blend 0.5 raw + 0.5 percentile, clipped [0,100]",
                      "base_policy": "cached past-only component mappings and fixed Ridge25 blend, verified row-by-row"},
        "temporal_protocol": {
            "correction_train": {"cycles": [56, 67], "blocks": [
                {"fit_cutoff": 55, "cycles": [56, 59]},
                {"fit_cutoff": 59, "cycles": [60, 63]},
                {"fit_cutoff": 63, "cycles": [64, 67]}],
                "rows": 17685, "all_component_scores_out_of_fit": True},
            "primary_fallback_W1": {"fit_cutoff": 55, "cycles": [56, 59], "corrected": False},
            "long_horizon_fallback_H12_E": {"fit_cutoff": 55, "cycles": [56, 67], "corrected": False},
            "primary_eval_W4": {"fit_cutoff": 67, "cycles": [68, 71], "corrected": True},
            "long_horizon_eval_H12_L": {"fit_cutoff": 67, "cycles": [68, 79], "corrected": True},
            "pilot_primary_pooled": "W1 unchanged plus W4 corrected; disjoint rows, non-consecutive windows, report as two-window policy summary only",
            "pilot_H12_pooled": "H12_E unchanged plus H12_L corrected; overlapping temporal views and not independent replication",
            "W2_W3": "training blocks for the one E67 head; report incumbent/baseline only, do not evaluate corrected outputs on training labels",
        },
        "score_labels": "threat_rank minus incumbent prediction is the head training residual; prediction is incumbent + 0.5 * learned correction, clipped [0,100].",
        "selection_gate": {"improvement": "at least one of pilot_primary_pooled or pilot_H12_pooled has candidate MAE lower than baseline",
                           "harm_guard": "W4 and H12_L candidate MAE harm must each be <= 0.5",
                           "subgroups": "report bias, EF, stream availability, and target tails descriptively; no tail routing"},
        "boundaries": {"official_train_only": True, "test_read": False, "kaggle": False,
                       "candidate_search": False, "rank_weight_search": False,
                       "rank_policy_52_55_reuse": False,
                       "early_oof_boundary": "end-51 component caches are raw predecessor predictions; their rank policy used 52-55 labels and no sparse-LGB component exists at cutoff51, so they are excluded from head fit."},
    }


def freeze_only() -> dict[str, Any]:
    protocol = _protocol()
    if PROTOCOL_PATH.exists():
        prior_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if prior_protocol != protocol:
            raise FileExistsError("Protocol path already exists with different bytes; preserve and investigate")
    else:
        _write_json(PROTOCOL_PATH, protocol)
    assembled = _assemble(pd.read_csv(TRAIN_PATH))
    inputs = _input_hashes()
    if FREEZE_PATH.exists():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if (existing.get("protocol_sha256") != sha_file(PROTOCOL_PATH)
                or existing.get("source_and_input_sha256") != inputs):
            raise FileExistsError("Freeze path already exists with different inputs; preserve and investigate")
        return existing
    freeze = {
        "experiment_id": protocol["experiment_id"],
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "protocol_sha256": sha_file(PROTOCOL_PATH), "source_and_input_sha256": inputs,
        "official_train_rows": 50608, "official_train_sha256": inputs["data/raw/regression/Train.csv"],
        "actual_oof_fold_manifests": assembled["fold_manifests"],
        "training_score_matrix_sha256": sha_array(np.column_stack([assembled["train"][x] for x in SCORE_NAMES])),
        "training_row_positions_sha256": sha_array(assembled["train"]["rows"]),
        "training_target_sha256": sha_array(assembled["frame"].iloc[assembled["train"]["rows"]].threat_rank.to_numpy(float)),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "fit_completed": False, "test_read": False,
    }
    _write_json(FREEZE_PATH, freeze)
    return freeze


def _load_verified_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if sha_file(PROTOCOL_PATH) != freeze["protocol_sha256"]:
        raise ValueError("Frozen protocol bytes changed")
    if freeze["source_and_input_sha256"] != _input_hashes():
        raise ValueError("Frozen source, raw data, or prediction cache identity changed")
    return protocol, freeze


def run_once() -> dict[str, Any]:
    protocol, freeze = _load_verified_freeze()
    if REPORT_PATH.exists() or PRED_PATH.exists() or RECEIPT_PATH.exists():
        raise FileExistsError("A run output already exists; do not overwrite this one-fit pilot")
    frame = pd.read_csv(TRAIN_PATH)
    assembled = _assemble(frame)
    fit_rows = assembled["train"]["rows"]
    fit_features = frame.iloc[fit_rows].loc[:, list(EXPANDED_FEATURES)]
    fit_scores = np.column_stack([assembled["train"][x] for x in SCORE_NAMES])
    fit_y = frame.iloc[fit_rows].threat_rank.to_numpy(dtype=float)
    head = OOFResidualHead("regression", list(SCORE_NAMES)).fit(fit_features, fit_scores, fit_y)
    audit = head.audit_state()
    validation_features = frame.iloc[assembled["h12_late"]["rows"]].loc[:, list(EXPANDED_FEATURES)]
    validation_scores = np.column_stack([assembled["h12_late"][x] for x in SCORE_NAMES])
    corrected_h12_late = head.predict(validation_features, validation_scores)
    h12_late_base = assembled["h12_late"]["champion"]
    h12_cycle = frame.iloc[assembled["h12_late"]["rows"]].cycle_code.to_numpy(dtype=int)
    w4_mask = h12_cycle <= 71
    corrected_w4 = corrected_h12_late[w4_mask]
    w1 = assembled["w1"]
    early = assembled["h12_early"]
    h12 = assembled["h12_late"]
    metrics = {
        "W1_unchanged_fallback": _comparison(frame.iloc[w1["rows"]], w1["champion"], w1["champion"]),
        "H12_E_unchanged_fallback": _comparison(frame.iloc[early["rows"]], early["champion"], early["champion"]),
        "W4_corrected": _comparison(frame.iloc[h12["rows"][w4_mask]], h12_late_base[w4_mask], corrected_w4),
        "H12_L_corrected": _comparison(frame.iloc[h12["rows"]], h12_late_base, corrected_h12_late),
    }
    primary_rows = np.concatenate([w1["rows"], h12["rows"][w4_mask]])
    primary_base = np.concatenate([w1["champion"], h12_late_base[w4_mask]])
    primary_candidate = np.concatenate([w1["champion"], corrected_w4])
    primary_frame = frame.iloc[primary_rows]
    metrics["pilot_primary_pooled_W1_plus_W4"] = _comparison(primary_frame, primary_base, primary_candidate)
    h12_rows = np.concatenate([early["rows"], h12["rows"]])
    h12_base = np.concatenate([early["champion"], h12_late_base])
    h12_candidate = np.concatenate([early["champion"], corrected_h12_late])
    metrics["pilot_H12_pooled_E_unchanged_plus_L_corrected"] = _comparison(
        frame.iloc[h12_rows], h12_base, h12_candidate)
    pooled_primary_gain = metrics["pilot_primary_pooled_W1_plus_W4"]["delta_mae_candidate_minus_baseline"] < 0
    pooled_h12_gain = metrics["pilot_H12_pooled_E_unchanged_plus_L_corrected"]["delta_mae_candidate_minus_baseline"] < 0
    w4_harm = metrics["W4_corrected"]["delta_mae_candidate_minus_baseline"]
    h12l_harm = metrics["H12_L_corrected"]["delta_mae_candidate_minus_baseline"]
    gate_pass = (pooled_primary_gain or pooled_h12_gain) and w4_harm <= 0.5 and h12l_harm <= 0.5
    predictions = {
        "fit_rows": fit_rows, "fit_scores": fit_scores, "fit_target": fit_y,
        "eval_rows_h12_l": h12["rows"], "eval_base_h12_l": h12_late_base,
        "eval_corrected_h12_l": corrected_h12_late,
        "eval_scores_h12_l": validation_scores,
        "eval_w4_rows": h12["rows"][w4_mask], "eval_w4_base": h12_late_base[w4_mask],
        "eval_w4_corrected": corrected_w4,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(PRED_PATH, **predictions)
    receipt = {
        "experiment_id": protocol["experiment_id"], "freeze_sha256": sha_file(FREEZE_PATH),
        "predictions_sha256": sha_file(PRED_PATH), "fit_count": 1,
        "fit_row_positions_sha256": sha_array(fit_rows), "fit_score_matrix_sha256": sha_array(fit_scores),
        "fit_target_sha256": sha_array(fit_y), "validation_h12l_score_matrix_sha256": sha_array(validation_scores),
        "head_state": audit, "head_output_sha256": sha_array(corrected_h12_late),
        "all_predictions_finite_and_bounded": bool(np.isfinite(corrected_h12_late).all() and
                                                      np.all((corrected_h12_late >= 0) & (corrected_h12_late <= 100))),
        "test_read": False, "kaggle": False,
    }
    _write_json(RECEIPT_PATH, receipt)
    report = {
        "experiment_id": protocol["experiment_id"], "status": "FIT_COMPLETE_AWAITING_VERIFICATION",
        "source_commit_at_freeze": freeze["source_commit"],
        "freeze_sha256": sha_file(FREEZE_PATH), "official_train_sha256": sha_file(TRAIN_PATH),
        "fit_cutoff": 67, "fit_rows": len(fit_rows),
        "training_cycles_and_blocks": protocol["temporal_protocol"]["correction_train"],
        "fit_state": audit, "metrics": metrics,
        "acceptance_gate": {"pooled_primary_improved": bool(pooled_primary_gain),
                             "pooled_H12_improved": bool(pooled_h12_gain),
                             "W4_MAE_harm_le_0_5": bool(w4_harm <= 0.5),
                             "H12_L_MAE_harm_le_0_5": bool(h12l_harm <= 0.5),
                             "gate_pass": bool(gate_pass)},
        "outputs": {"predictions": str(PRED_PATH.relative_to(ROOT)).replace("\\", "/"),
                    "run_receipt": str(RECEIPT_PATH.relative_to(ROOT)).replace("\\", "/")},
        "interpretation_limits": [
            "Only W4 and H12_L receive the fitted correction; W1 and H12_E are explicit unchanged fallbacks.",
            "The pooled primary metric covers W1 plus W4 only; it is a non-consecutive two-window policy summary, not the six-window primary protocol.",
            "The pooled H12 metric is a past-unchanged plus future-corrected policy summary; H12 overlaps primary and is not independent replication.",
            "W2/W3 residual rows train the head and are not scored as corrected outputs.",
            "Temporal windows and subgroup labels have been used in prior adaptive development; this pilot is not an untouched holdout.",
            "Target tails are post-score descriptive groups and cannot define a deployable routing rule.",
        ],
    }
    _write_json(REPORT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        result = freeze_only()
        print(json.dumps({"status": "FROZEN_NO_FIT", "freeze_sha256": sha_file(FREEZE_PATH),
                          "training_rows": result.get("fit_rows", 17685)}, indent=2))
    elif args.audit_only:
        frame = pd.read_csv(TRAIN_PATH)
        data = _assemble(frame)
        print(json.dumps({"status": "AUDIT_PASS_NO_FIT", "fit_rows": len(data["train"]["rows"]),
                          "w1_rows": len(data["w1"]["rows"]), "w4_rows": len(data["w4"]["rows"]),
                          "h12_e_rows": len(data["h12_early"]["rows"]),
                          "h12_l_rows": len(data["h12_late"]["rows"]),
                          "source_hashes": data["source_cache_hashes"]}, indent=2))
    else:
        result = run_once()
        print(json.dumps({"status": result["status"], "acceptance_gate": result["acceptance_gate"],
                          "metrics": {k: v["candidate"] for k, v in result["metrics"].items()}}, indent=2))


if __name__ == "__main__":
    main()
