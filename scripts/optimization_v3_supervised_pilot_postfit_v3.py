"""Versioned post-fit repair for subgroup residual alignment, with zero-refit scope."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import optimization_v3_supervised_pilot as pilot
from scripts import optimization_v3_verify_supervised_pilot as verifier
from scripts import optimization_v3_supervised_pilot_postfit_v2 as v2
from src.models.round1_evaluation import code_digest, digest_json

CONFIG = pilot.CONFIG
OUT = pilot.OUT
CACHE = pilot.CACHE
AMENDMENT = CONFIG / "postfit_amendment_v3.json"
ADAPTER = Path(__file__).relative_to(ROOT).as_posix()
TEST = "tests/test_optimization_v3_supervised_pilot_postfit_v3.py"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _empty_metric() -> dict:
    return {"n": 0, "mae": None, "rmse": None,
            "bias_prediction_minus_target": None, "median_signed_residual": None,
            "p90_absolute_error": None, "p95_absolute_error": None,
            "over_rate": None, "under_rate": None, "tie_rate": None}


def _production_error(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("Regression subgroup targets and predictions must use the same row mask")
    if not len(y):
        return _empty_metric()
    return pilot.regression_error(y, p)


def compare_reg_view_fixed(frame: pd.DataFrame, target: np.ndarray, rows: np.ndarray,
                          predictions: dict[str, np.ndarray]) -> dict:
    """Frozen runner logic with the subgroup prediction mask applied consistently."""
    rows = np.asarray(rows, dtype=np.int64)
    y = np.asarray(target[rows], dtype=float)
    cycles = pd.to_numeric(frame.iloc[rows].cycle_code, errors="raise").to_numpy(dtype=int)
    entry = pd.to_numeric(frame.iloc[rows].entry_friction, errors="coerce").to_numpy(dtype=float)
    stream = frame.iloc[rows].signal_stream
    missing = stream.isna().to_numpy() | stream.fillna("").astype(str).str.strip().eq("").to_numpy()
    group_masks = {"EF0": entry == 0, "EF1": entry == 1,
                   "tail_le5": y <= 5, "tail_ge95": y >= 95,
                   "stream_missing": missing, "stream_present": ~missing}
    result = {"n": int(len(rows)), "metrics": {}, "per_cycle": {}, "subgroups": {}}
    for name, prediction in predictions.items():
        p = np.asarray(prediction, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f"Regression prediction alignment/finite check failed for {name}")
        overall = pilot.regression_error(y, p)
        per_cycle = {str(int(c)): pilot.regression_error(y[cycles == c], p[cycles == c])
                     for c in np.unique(cycles)}
        overall["equal_cycle_mae"] = float(np.mean([item["mae"] for item in per_cycle.values()]))
        result["metrics"][name] = overall
        result["per_cycle"][name] = per_cycle
    champion = result["metrics"]["champion"]
    result["delta_vs_champion"] = {}
    for name in ["raw", "rank50", "blend25"]:
        metric = result["metrics"][name]
        result["delta_vs_champion"][name] = {
            "mae_gain_champion_minus_candidate": champion["mae"] - metric["mae"],
            "rmse_gain_champion_minus_candidate": champion["rmse"] - metric["rmse"],
            "bias_shift_candidate_minus_champion": metric["bias_prediction_minus_target"] - champion["bias_prediction_minus_target"],
            "equal_cycle_mae_gain": champion["equal_cycle_mae"] - metric["equal_cycle_mae"],
        }
    for group, mask in group_masks.items():
        group_metrics = {name: _production_error(y[mask], np.asarray(p)[mask])
                         for name, p in predictions.items()}
        result["subgroups"][group] = {
            "n": int(mask.sum()),
            "metrics": group_metrics,
            "mae_gain_vs_champion": {
                name: (group_metrics["champion"]["mae"] - group_metrics[name]["mae"]
                       if mask.any() else None)
                for name in predictions if name != "champion"
            },
        }
    return result


def _independent_error(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("Independent regression subgroup arrays are misaligned")
    if not len(y):
        return _empty_metric()
    return verifier.regression_metric_independent(y, p)


def regression_view_independent(frame: pd.DataFrame, target: np.ndarray,
                                rows: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    """Verifier-side recomputation using independently implemented residual metrics."""
    rows = np.asarray(rows, dtype=np.int64)
    y = np.asarray(target[rows], dtype=float)
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    entry = pd.to_numeric(frame.iloc[rows].entry_friction, errors="coerce").to_numpy(dtype=float)
    stream = frame.iloc[rows].signal_stream
    missing = stream.isna().to_numpy() | stream.fillna("").astype(str).str.strip().eq("").to_numpy()
    masks = {"EF0": entry == 0, "EF1": entry == 1, "tail_le5": y <= 5,
             "tail_ge95": y >= 95, "stream_missing": missing, "stream_present": ~missing}
    metrics, per_cycle = {}, {}
    for name, prediction in predictions.items():
        p = np.asarray(prediction, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f"Independent prediction alignment failed for {name}")
        metrics[name] = _independent_error(y, p)
        by_cycle = {str(int(c)): _independent_error(y[cycles == c], p[cycles == c])
                    for c in np.unique(cycles)}
        metrics[name]["equal_cycle_mae"] = float(np.mean([item["mae"] for item in by_cycle.values()]))
        per_cycle[name] = by_cycle
    champion = metrics["champion"]
    delta = {name: {
        "mae_gain_champion_minus_candidate": champion["mae"] - metrics[name]["mae"],
        "rmse_gain_champion_minus_candidate": champion["rmse"] - metrics[name]["rmse"],
        "bias_shift_candidate_minus_champion": metrics[name]["bias_prediction_minus_target"] - champion["bias_prediction_minus_target"],
        "equal_cycle_mae_gain": champion["equal_cycle_mae"] - metrics[name]["equal_cycle_mae"],
    } for name in ["raw", "rank50", "blend25"]}
    subgroups = {}
    for group, mask in masks.items():
        group_metrics = {name: _independent_error(y[mask], np.asarray(p)[mask])
                         for name, p in predictions.items()}
        subgroups[group] = {
            "n": int(mask.sum()),
            "metrics": group_metrics,
            "mae_gain_vs_champion": {
                name: (group_metrics["champion"]["mae"] - group_metrics[name]["mae"]
                       if mask.any() else None)
                for name in predictions if name != "champion"
            },
        }
    return {"n": int(len(rows)), "metrics": metrics, "per_cycle": per_cycle,
            "delta_vs_champion": delta, "subgroups": subgroups}


def make_amendment_identity() -> dict:
    base = v2.make_amendment_identity()
    out_paths = [OUT / "pilot.json", OUT / "FULLPILOT.md", OUT / "verification.json",
                 OUT / "postfit_adapter_run_v3.json"]
    present = [str(path) for path in out_paths if path.exists()]
    if present:
        raise FileExistsError(f"Refusing V3 metrics while derived outputs already exist: {present}")
    v2_receipt = CONFIG / "postfit_amendment_v2.json"
    if not v2_receipt.is_file():
        raise FileNotFoundError("Preserved V2 post-fit amendment receipt is required")
    source_hashes = {
        "scripts/optimization_v3_supervised_pilot_postfit_v2.py": code_digest(ROOT / "scripts/optimization_v3_supervised_pilot_postfit_v2.py"),
        ADAPTER: code_digest(Path(__file__)),
        TEST: code_digest(ROOT / TEST),
    }
    return {
        "v2_amendment_sha256_exact_bytes": sha(v2_receipt),
        "v2_amendment_signature": read_json(v2_receipt)["signature"],
        "v2_source_and_cache_identity": base,
        "v3_source_code_sha256_lf_utf8": source_hashes,
        "output_cache_sha256_exact_bytes": base["four_fit_output_sha256_exact_bytes"],
        "new_official_fits": 0,
        "test_read": False,
        "kaggle_access": False,
        "repair": "Align every subgroup target and prediction with the same row mask; emit null subgroup metrics and gains for zero-support groups. Independent verifier recomputes these values on the same rows.",
    }


def freeze_amendment() -> dict:
    identity = make_amendment_identity()
    receipt = {"status": "FROZEN_POSTFIT_V3_BEFORE_METRICS",
               "at_utc": datetime.now(timezone.utc).isoformat(),
               "identity": identity, "signature": digest_json(identity)}
    if AMENDMENT.exists():
        old = json.loads(AMENDMENT.read_text(encoding="utf-8"))
        if old.get("identity") != identity or old.get("signature") != receipt["signature"]:
            raise AssertionError("Existing V3 amendment differs; preserve it and use a new version")
        return old
    write_new(AMENDMENT, receipt)
    return receipt


def check_amendment() -> dict:
    receipt = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    current = make_amendment_identity()
    if receipt.get("identity") != current or receipt.get("signature") != digest_json(current):
        raise AssertionError("V3 post-fit source/input/output identity changed")
    return receipt


def aggregate_and_verify() -> dict:
    amendment = check_amendment()
    # Patch only in-memory analysis functions; frozen source files remain byte-identical.
    pilot.compare_reg_view = compare_reg_view_fixed
    original_write = pilot.write_new

    def tagged_pilot_write(path: Path, payload: dict) -> None:
        if Path(path) == OUT / "pilot.json":
            payload["postfit_adapter"] = {
                "version": "postfit_v3",
                "amendment_signature": amendment["signature"],
                "official_refits": 0,
                "frozen_runner_source_edited": False,
            }
        original_write(path, payload)

    pilot.write_new = tagged_pilot_write
    original_v2_write = v2.write_new

    def suppress_failed_v2_summary(path: Path, payload: dict) -> None:
        # Do not create a misleading V2 summary when the V3 fixed adapter is active.
        if Path(path).name == "postfit_adapter_run_v2.json":
            return
        original_v2_write(path, payload)

    v2.write_new = suppress_failed_v2_summary
    result = v2.aggregate()

    # Independent verifier uses its own residual metric implementation and its own subset masks.
    verifier._reg_view = regression_view_independent
    verified = verifier.verify()
    if verified.get("status") != "PASS" or verified.get("newfits") != 0:
        raise AssertionError("Independent supervised pilot verification did not PASS")
    final = {
        "status": "PASS_POSTFIT_V3_AND_INDEPENDENT_VERIFICATION",
        "v3_amendment_signature": amendment["signature"],
        "v2_amendment_signature_preserved": read_json(CONFIG / "postfit_amendment_v2.json")["signature"],
        "pilot_json_sha256_exact_bytes": sha(OUT / "pilot.json"),
        "fullpilot_markdown_sha256_exact_bytes": sha(OUT / "FULLPILOT.md"),
        "independent_verification_sha256_exact_bytes": sha(OUT / "verification.json"),
        "independent_verification_status": verified["status"],
        "official_refits": 0,
        "test_read": False,
        "kaggle_access": False,
    }
    write_new(OUT / "postfit_adapter_run_v3.json", final)
    return {"status": final["status"], "result": result,
            "verification": verified, "receipt": final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-amendment", action="store_true")
    group.add_argument("--aggregate-and-verify", action="store_true")
    args = parser.parse_args()
    if args.freeze_amendment:
        receipt = freeze_amendment()
        print(receipt["signature"])
    else:
        result = aggregate_and_verify()
        report = result["result"]
        print(json.dumps({"status": result["status"],
                          "classification_gate": report["classification"]["advance_gate_vs_latest_sparse25"],
                          "regression_gate": report["regression"]["advance_gate_vs_current_champion"],
                          "verification_status": result["verification"]["status"],
                          "new_refits": 0}, indent=2, allow_nan=False), flush=True)
