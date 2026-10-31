"""Independent lineage, refit, and metric verification for the E=67 pilot."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.optimization_v3_residual import OOFResidualHead  # noqa: E402
from src.models.round1_models import EXPANDED_FEATURES  # noqa: E402
from src.utils.paths import data_root  # noqa: E402
from scripts.optimization_v3_reg_residual import (  # noqa: E402
    BASE_CACHE_PATH, DIAGNOSTIC_CACHE_PATH, FULL_CACHE_PATH,
    PILOT_CACHE_PATH, SCORE_NAMES, TRAIN_PATH,
    FREEZE_PATH, PROTOCOL_PATH, REPORT_PATH, PRED_PATH, RECEIPT_PATH, REPORT_DIR,
    FIXED_INPUTS, sha_array, sha_file,
)

VERIFY_PATH = REPORT_DIR / "verification.json"


def _component_block(train: pd.DataFrame, base: Any, pilot: Any, full: Any,
                     cutoff: int, first_cycle: int, last_cycle: int) -> dict[str, np.ndarray]:
    cache = pilot if cutoff in (55, 67) else full
    prediction_rows = np.asarray(cache[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
    raw_sparse = np.asarray(cache[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
    cycle_all = train.iloc[prediction_rows].cycle_code.to_numpy(dtype=int)
    mask = (cycle_all >= first_cycle) & (cycle_all <= last_cycle)
    rows = prediction_rows[mask]
    expected = np.flatnonzero(train.cycle_code.between(first_cycle, last_cycle).to_numpy())
    if not np.array_equal(rows, expected):
        raise AssertionError(f"source rows differ from cycles {first_cycle}-{last_cycle}")
    if not np.array_equal(prediction_rows, np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)):
        raise AssertionError("component cache row vectors differ")
    raw_sparse = raw_sparse[mask]
    cycles = train.iloc[rows].cycle_code.to_numpy(dtype=int)
    sparse_rank = np.empty(len(rows), dtype=float)
    for code in np.unique(cycles):
        selected = cycles == code
        ranks = rankdata(raw_sparse[selected], method="average")
        sparse_rank[selected] = .5 * raw_sparse[selected] + .5 * 100 * ranks / int(selected.sum())
    old_rows = np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)
    old_cycles = train.iloc[old_rows].cycle_code.to_numpy(dtype=int)
    old_mask = (old_cycles >= first_cycle) & (old_cycles <= last_cycle)
    lgb = np.asarray(base[f"lgb_{cutoff}"], dtype=float)[old_mask]
    ridge = np.asarray(base[f"ridge_{cutoff}"], dtype=float)[old_mask]
    ridge25 = np.asarray(base[f"blend_ridge25_{cutoff}"], dtype=float)[old_mask]
    if not np.allclose(ridge25, np.clip(.75 * lgb + .25 * ridge, 0, 100), atol=1e-10, rtol=0):
        raise AssertionError("cached Ridge25 no longer decomposes into its components")
    sparse_rank = np.clip(sparse_rank, 0, 100)
    champion = np.clip(.75 * ridge25 + .25 * sparse_rank, 0, 100)
    return {"rows": rows, "champion": champion, "lgb_svd": lgb,
            "ridge": ridge, "lgb_sparse": sparse_rank}


def _stack(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    data = {name: np.concatenate([block[name] for block in blocks]) for name in blocks[0]}
    order = np.argsort(data["rows"], kind="stable")
    return {name: values[order] for name, values in data.items()}


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    residual = p - y
    absolute = np.abs(residual)
    return {"n": int(len(y)), "mae": float(absolute.mean()),
            "bias_prediction_minus_target": float(residual.mean()),
            "median_signed_residual": float(np.median(residual)),
            "over_rate": float(np.mean(residual > 0)), "under_rate": float(np.mean(residual < 0)),
            "p90_absolute_error": float(np.quantile(absolute, .9)),
            "p95_absolute_error": float(np.quantile(absolute, .95))}


def _compare(frame: pd.DataFrame, base: np.ndarray, adjusted: np.ndarray) -> dict[str, Any]:
    y = frame.threat_rank.to_numpy(float)
    masks = {"EF0": frame.entry_friction.to_numpy() == 0,
             "EF1": frame.entry_friction.to_numpy() == 1,
             "stream_missing": frame.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
             "stream_present": ~frame.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
             "tail_le5": y <= 5, "tail_ge95": y >= 95,
             "middle": (y > 5) & (y < 95)}
    groups = {}
    for label, mask in masks.items():
        if mask.any():
            groups[label] = {"n": int(mask.sum()), "baseline": _metric(y[mask], base[mask]),
                             "candidate": _metric(y[mask], adjusted[mask])}
    mb, mc = _metric(y, base), _metric(y, adjusted)
    return {"baseline": mb, "candidate": mc, "delta_mae_candidate_minus_baseline": mc["mae"] - mb["mae"],
            "groups": groups}


def _verify_source_hashes(freeze: dict[str, Any]) -> None:
    for rel, expected in freeze["source_and_input_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or sha_file(path) != expected:
            raise AssertionError(f"frozen input changed or missing: {rel}")
    if sha_file(PROTOCOL_PATH) != freeze["protocol_sha256"]:
        raise AssertionError("protocol hash mismatch")
    if sha_file(TRAIN_PATH) != "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082":
        raise AssertionError("official Train hash mismatch")


def verify() -> dict[str, Any]:
    if VERIFY_PATH.exists():
        raise FileExistsError("Independent verification receipt already exists")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    saved = np.load(PRED_PATH, allow_pickle=False)
    _verify_source_hashes(freeze)
    train = pd.read_csv(TRAIN_PATH)
    dual = np.load(BASE_CACHE_PATH, allow_pickle=False)
    pilot = np.load(PILOT_CACHE_PATH, allow_pickle=False)
    full = np.load(FULL_CACHE_PATH, allow_pickle=False)

    fit = _stack([_component_block(train, dual, pilot, full, 55, 56, 59),
                  _component_block(train, dual, pilot, full, 59, 60, 63),
                  _component_block(train, dual, pilot, full, 63, 64, 67)])
    early = _component_block(train, dual, pilot, full, 55, 56, 67)
    w1 = _component_block(train, dual, pilot, full, 55, 56, 59)
    late = _component_block(train, dual, pilot, full, 67, 68, 79)
    w4 = _component_block(train, dual, pilot, full, 67, 68, 71)
    fit_scores = np.column_stack([fit[name] for name in SCORE_NAMES])
    fit_y = train.iloc[fit["rows"]].threat_rank.to_numpy(float)
    if not np.array_equal(fit["rows"], saved["fit_rows"]):
        raise AssertionError("saved head training rows differ from original OOF fold caches")
    np.testing.assert_allclose(fit_scores, saved["fit_scores"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit_y, saved["fit_target"], rtol=0, atol=0)
    if not np.array_equal(late["rows"], saved["eval_rows_h12_l"]):
        raise AssertionError("saved evaluation row IDs differ from cutoff67 original caches")
    validation_scores = np.column_stack([late[name] for name in SCORE_NAMES])
    np.testing.assert_allclose(validation_scores, saved["eval_scores_h12_l"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(late["champion"], saved["eval_base_h12_l"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(w1["champion"], early["champion"][
        train.iloc[early["rows"]].cycle_code.to_numpy() <= 59], rtol=0, atol=1e-12)
    if not np.array_equal(w4["rows"], saved["eval_w4_rows"]):
        raise AssertionError("saved W4 IDs differ from exact cutoff67 cycle68-71 subset")
    np.testing.assert_allclose(late["champion"][w4mask], saved["eval_w4_base"], rtol=0, atol=1e-12)
    if not np.array_equal(fit["rows"], np.sort(fit["rows"])):
        raise AssertionError("head training rows are not in canonical source-position order")
    if np.intersect1d(fit["rows"], late["rows"]).size:
        raise AssertionError("head training and validation rows overlap")
    train_cycles = train.iloc[fit["rows"]].cycle_code.to_numpy(dtype=int)
    if not (train_cycles.min() == 56 and train_cycles.max() == 67):
        raise AssertionError("head training cycle range differs from 56-67")
    if int(train_cycles.max()) >= int(train.iloc[late["rows"]].cycle_code.min()):
        raise AssertionError("head training labels must strictly predate correction evaluation")

    # Deterministic independent head refit; no candidate search or base-model fit.
    head = OOFResidualHead("regression", list(SCORE_NAMES))
    head.fit(train.iloc[fit["rows"]].loc[:, list(EXPANDED_FEATURES)], fit_scores, fit_y)
    if head.audit_state() != receipt["head_state"]:
        raise AssertionError("independently refit residual-head state differs from receipt")
    corrected = head.predict(train.iloc[late["rows"]].loc[:, list(EXPANDED_FEATURES)], validation_scores)
    np.testing.assert_allclose(corrected, saved["eval_corrected_h12_l"], rtol=0, atol=1e-12)
    if sha_array(corrected) != receipt["head_output_sha256"]:
        raise AssertionError("independently refit output hash differs from run receipt")
    if not np.isfinite(corrected).all() or np.any((corrected < 0) | (corrected > 100)):
        raise AssertionError("corrected predictions must be finite and in [0,100]")

    metrics = {}
    metrics["W1_unchanged_fallback"] = _compare(train.iloc[w1["rows"]], w1["champion"], w1["champion"])
    metrics["H12_E_unchanged_fallback"] = _compare(train.iloc[early["rows"]], early["champion"], early["champion"])
    h12_cycle = train.iloc[late["rows"]].cycle_code.to_numpy()
    w4mask = h12_cycle <= 71
    metrics["W4_corrected"] = _compare(train.iloc[late["rows"][w4mask]], late["champion"][w4mask], corrected[w4mask])
    metrics["H12_L_corrected"] = _compare(train.iloc[late["rows"]], late["champion"], corrected)
    pooled_primary_rows = np.concatenate([w1["rows"], late["rows"][w4mask]])
    pooled_primary_base = np.concatenate([w1["champion"], late["champion"][w4mask]])
    pooled_primary_adjusted = np.concatenate([w1["champion"], corrected[w4mask]])
    metrics["pilot_primary_pooled_W1_plus_W4"] = _compare(
        train.iloc[pooled_primary_rows], pooled_primary_base, pooled_primary_adjusted)
    pooled_h12_rows = np.concatenate([early["rows"], late["rows"]])
    pooled_h12_base = np.concatenate([early["champion"], late["champion"]])
    pooled_h12_adjusted = np.concatenate([early["champion"], corrected])
    metrics["pilot_H12_pooled_E_unchanged_plus_L_corrected"] = _compare(
        train.iloc[pooled_h12_rows], pooled_h12_base, pooled_h12_adjusted)
    if metrics != report["metrics"]:
        raise AssertionError("independent metric/subgroup reconstruction differs from report")

    p_gain = metrics["pilot_primary_pooled_W1_plus_W4"]["delta_mae_candidate_minus_baseline"] < 0
    h_gain = metrics["pilot_H12_pooled_E_unchanged_plus_L_corrected"]["delta_mae_candidate_minus_baseline"] < 0
    w4_harm = metrics["W4_corrected"]["delta_mae_candidate_minus_baseline"] <= .5
    h12_harm = metrics["H12_L_corrected"]["delta_mae_candidate_minus_baseline"] <= .5
    gate = {"pooled_primary_improved": bool(p_gain), "pooled_H12_improved": bool(h_gain),
            "W4_MAE_harm_le_0_5": bool(w4_harm), "H12_L_MAE_harm_le_0_5": bool(h12_harm),
            "gate_pass": bool((p_gain or h_gain) and w4_harm and h12_harm)}
    if gate != report["acceptance_gate"]:
        raise AssertionError("acceptance gate differs from independent reconstruction")
    checks = {
        "freeze_source_hashes_match": True,
        "official_train_rows_match": int(len(train)) == 50608,
        "fit_rows_exactly_actual_oof_56_67": len(fit["rows"]) == 17685,
        "each_fit_score_comes_from_earlier_outer_cutoff": int(train_cycles.max()) < int(train.iloc[late["rows"]].cycle_code.min()),
        "w1_and_h12_e_remain_bit_identical_fallbacks": True,
        "w4_h12_l_rows_and_scores_exact": True,
        "independent_head_refit_state_matches": True,
        "independent_prediction_replay_matches": True,
        "independent_metric_reconstruction_matches": True,
        "acceptance_gate_reconstructed": True,
        "test_read": False, "kaggle": False,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    verification = {"status": "PASS", "experiment_id": protocol["experiment_id"],
                    "freeze_sha256": sha_file(FREEZE_PATH), "report_sha256": sha_file(REPORT_PATH),
                    "run_receipt_sha256": sha_file(RECEIPT_PATH), "predictions_sha256": sha_file(PRED_PATH),
                    "head_refits_for_verification": 1, "checks": checks,
                    "acceptance_gate": gate,
                    "metrics": {key: value["candidate"] for key, value in metrics.items()}}
    VERIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_PATH.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
