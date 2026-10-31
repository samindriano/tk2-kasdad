"""Run the predeclared SVD32-vs-SVD128 paired LightGBM ablation.

Only official regression Train is read. The Stage 5 SVD32 inner-selected
parameter/rank-weight policy is held fixed; six SVD128 outer fits are run.
No model selection, full-train refit, Test inference, or submission occurs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.autonomous_regression import build_autonomous_regression
from src.models.regression_research_common import mapped_scores, score_summary
from src.models.regression_round2_evaluation import load_protocol, positions_digest
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT / "experiments/autonomous_eda/regression"
REPORT = PROJECT_ROOT / "reports/autonomous_eda/regression"
RUN = PROJECT_ROOT / "experiments/runs/autonomous_reg_v1"
STAGE5_RUN = PROJECT_ROOT / "experiments/runs/regression_research_v1"
STAGE5_REPORT = PROJECT_ROOT / "reports/regression_research/stage5_lgb_text.json"
STAGE5_FREEZE = PROJECT_ROOT / "experiments/regression_research/stage5_lgb_text_freeze.json"
ROUND1_FREEZE = PROJECT_ROOT / "experiments/round1/freeze.json"

PROTOCOL_PATH = ROOT / "protocol.json"
FREEZE_PATH = ROOT / "freeze.json"
PRIOR_AMENDMENT_PATH = ROOT / "postfit_reporting_amendment_v3.json"
AMENDMENT_PATH = ROOT / "postfit_reporting_amendment_v4.json"
INNER_PROTOCOL_PATH = ROOT / "inner_refinement_protocol.json"
INNER_FREEZE_PATH = ROOT / "inner_refinement_freeze.json"
INNER_REPORT_JSON = REPORT / "inner_refinement.json"
INNER_REPORT_MD = REPORT / "inner_refinement.md"
CODE_PATHS = [
    "src/models/autonomous_regression.py",
    "scripts/research_autonomous_regression.py",
    "scripts/verify_autonomous_regression.py",
]
REFERENCE_PATHS = [
    "experiments/regression_research/stage5_lgb_text_freeze.json",
    "reports/regression_research/stage5_lgb_text.json",
    "src/models/regression_research_lgb_text.py",
    "scripts/research_regression_lgb_text.py",
    "experiments/regression_research/stage4_refinement_freeze.json",
    "src/models/regression_research_refinement.py",
]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _window_specs() -> tuple[list[dict], list[dict]]:
    protocol = load_protocol()
    return protocol["primary_windows"], protocol["secondary_windows"]


def _stage5_choices(stage5: dict) -> dict[int, dict]:
    selected = {int(item["fit_end"]): dict(item["selected"]) for item in stage5["inner"]}
    required = {55, 59, 63, 67, 71, 75}
    if set(selected) != required:
        raise RuntimeError(f"Stage 5 fit-end choices differ: {sorted(selected)}")
    return selected


def select_inner_policy(records: list[dict]) -> dict:
    """Use Stage 5's exact MAE/tie rule for a grid within one inner fold."""
    if not records:
        raise ValueError("Inner selection requires at least one parameter/rank-weight result")
    if any(item["param"] not in {"base", "regularized"} for item in records):
        raise ValueError("Inner selection received a parameter outside the frozen two-value grid")
    if any(not np.isfinite(float(item["mae"])) or not 0 <= float(item["rank_weight"]) <= 1 for item in records):
        raise ValueError("Inner selection metrics must be finite and rank weights in [0, 1]")
    best = min(float(item["mae"]) for item in records)
    close = [item for item in records if float(item["mae"]) <= best + 1e-10]
    order = {"regularized": 0, "base": 1}
    return dict(sorted(close, key=lambda item: (float(item["rank_weight"]), order[item["param"]]))[0])


def _load_control(
    frame: pd.DataFrame,
    end: int,
    selected: dict,
    primary: list[dict],
    secondary: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    max_cycle = max(
        int(window["validation"][1])
        for window in [*primary, *secondary]
        if int(window["fit"][1]) == end
    )
    stem = f"lgbtext__{selected['param']}__fit{end}__to{max_cycle}"
    raw_path, meta_path = STAGE5_RUN / f"{stem}.npz", STAGE5_RUN / f"{stem}.json"
    mapped_path = STAGE5_RUN / f"eval_lgbtext__fit{end}__inner.npz"
    if not raw_path.is_file() or not meta_path.is_file() or not mapped_path.is_file():
        raise FileNotFoundError(f"Missing frozen Stage 5 control vectors for fit_end={end}")
    meta = _load_json(meta_path)
    if meta.get("param") != selected["param"] or meta.get("state_sha256") != digest_json(meta.get("state")):
        raise RuntimeError(f"Stage 5 control audit mismatch for fit_end={end}")
    if meta.get("prediction_sha256") is None:
        raise RuntimeError("Stage 5 control prediction hash is missing")
    with np.load(raw_path, allow_pickle=False) as z:
        fit_rows, rows, raw = z["fit_rows"].copy(), z["validation_rows"].copy(), z["prediction"].copy()
    if array_digest(raw) != meta["prediction_sha256"]:
        raise RuntimeError(f"Stage 5 raw control prediction hash mismatch for fit_end={end}")
    expected_fit = np.flatnonzero(frame.cycle_code.between(47, end).to_numpy())
    if not np.array_equal(fit_rows, expected_fit):
        raise RuntimeError(f"Stage 5 control fit row positions mismatch for fit_end={end}")
    expected_rows = np.flatnonzero(frame.cycle_code.between(end + 1, max_cycle).to_numpy())
    if not np.array_equal(rows, expected_rows):
        raise RuntimeError(f"Stage 5 control validation row positions mismatch for fit_end={end}")
    with np.load(mapped_path, allow_pickle=False) as z:
        mapped_rows, mapped = z["validation_rows"].copy(), z["prediction"].copy()
    if not np.array_equal(mapped_rows, rows):
        raise RuntimeError(f"Stage 5 mapped row positions mismatch for fit_end={end}")
    expected_mapped = mapped_scores(raw, frame.iloc[rows].cycle_code.to_numpy(), float(selected["rank_weight"]))
    if not np.array_equal(mapped, expected_mapped):
        raise RuntimeError(f"Stage 5 mapped control does not match frozen rank weight at fit_end={end}")
    return rows, mapped, {
        "fit_end": end,
        "param": selected["param"],
        "rank_weight": float(selected["rank_weight"]),
        "raw_prediction_sha256": array_digest(raw),
        "mapped_prediction_sha256": array_digest(mapped),
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_positions_sha256": positions_digest(rows),
        "state_sha256": meta["state_sha256"],
    }


def _current_identity(frame: pd.DataFrame, train_path: Path, control_vectors: dict,
                      control_audits: dict) -> dict:
    stage5 = _load_json(STAGE5_REPORT)
    stage5_freeze = _load_json(STAGE5_FREEZE)
    round1_freeze = _load_json(ROUND1_FREEZE)
    train_sha = _sha_file(train_path)
    expected_train_sha = round1_freeze["identity"]["input_sha256"]["train"]
    if train_sha != expected_train_sha:
        raise RuntimeError("Official regression Train hash differs from the frozen Round 1 input")
    if stage5.get("freeze_signature") != stage5_freeze.get("signature"):
        raise RuntimeError("Stage 5 report is not tied to its frozen identity")
    expected_primary = 11.931600241417073
    expected_h12 = 14.704040710404005
    control_result = _measure(frame, control_vectors)
    if abs(control_result["aggregate"]["primary"]["metrics"]["raw_mae"] - expected_primary) > 1e-10:
        raise RuntimeError("Reconstructed Stage 5 control primary MAE differs from frozen result")
    if abs(control_result["aggregate"]["h12"]["metrics"]["raw_mae"] - expected_h12) > 1e-10:
        raise RuntimeError("Reconstructed Stage 5 control H12 MAE differs from frozen result")
    return {
        "train_sha256": train_sha,
        "train_rows": int(len(frame)),
        "train_columns": list(frame.columns),
        "stage5_signature": stage5_freeze["signature"],
        "stage5_source_code_sha256": stage5_freeze["identity"]["code_sha256"],
        "stage5_report_sha256": _sha_file(STAGE5_REPORT),
        "round1_train_sha256": expected_train_sha,
        "autonomous_code_sha256": {path: code_digest(PROJECT_ROOT / path) for path in CODE_PATHS},
        "reference_sha256": {path: code_digest(PROJECT_ROOT / path) for path in REFERENCE_PATHS},
        "protocol_sha256": digest_json(_load_json(PROTOCOL_PATH)),
        "controls": control_audits,
        "control_primary_mae": control_result["aggregate"]["primary"]["metrics"]["raw_mae"],
        "control_h12_mae": control_result["aggregate"]["h12"]["metrics"]["raw_mae"],
        "runtime_packages": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]
        },
    }


def freeze(frame: pd.DataFrame, train_path: Path, control_vectors: dict,
           control_audits: dict, *, report_only: bool = False) -> dict:
    protocol = _load_json(PROTOCOL_PATH)
    identity = _current_identity(frame, train_path, control_vectors, control_audits)
    signature = digest_json(identity)
    if FREEZE_PATH.exists():
        receipt = _load_json(FREEZE_PATH)
        if receipt.get("signature") != digest_json(receipt.get("identity")):
            raise RuntimeError("Autonomous regression pre-fit receipt signature is invalid")
        if receipt.get("identity") != identity:
            if not report_only:
                raise RuntimeError("Frozen code differs; use --report-only for the approved post-fit reporting correction")
            identity = _authorize_postfit_reporting_amendment(receipt, identity, control_audits)
        return receipt
    receipt = {
        "status": "FROZEN_BEFORE_SVD128_OUTER_FITS",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": protocol,
        "identity": identity,
        "signature": signature,
    }
    _write_json(FREEZE_PATH, receipt, exclusive=True)
    return receipt


def _identity_without_reporting_code(identity: dict) -> dict:
    value = json.loads(json.dumps(identity))
    for path in ["scripts/research_autonomous_regression.py", "scripts/verify_autonomous_regression.py"]:
        value["autonomous_code_sha256"].pop(path, None)
    return value


def _candidate_cache_manifest(controls: dict, freeze_signature: str) -> dict:
    entries = {}
    for end_text, control in sorted(controls.items(), key=lambda item: int(item[0])):
        end = int(end_text)
        candidates = sorted(RUN.glob(f"svd128__{control['param']}__fit{end}__to*.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one completed SVD128 fit cache for fit_end={end}")
        meta_path = candidates[0]
        npz_path = meta_path.with_suffix(".npz")
        if not npz_path.is_file():
            raise RuntimeError(f"SVD128 fit cache is incomplete for fit_end={end}")
        meta = _load_json(meta_path)
        with np.load(npz_path, allow_pickle=False) as z:
            fit_rows, rows = z["fit_rows"].copy(), z["validation_rows"].copy()
            raw, mapped, explained = z["raw_prediction"].copy(), z["mapped_prediction"].copy(), z["explained_variance_ratio"].copy()
        expected_key = digest_json({
            "freeze_signature": freeze_signature,
            "params_id": control["param"],
            "fit_positions_sha256": positions_digest(fit_rows),
            "validation_positions_sha256": positions_digest(rows),
        })
        if meta.get("key") != expected_key or meta.get("state_sha256") != digest_json(meta.get("state")):
            raise RuntimeError(f"SVD128 cache key or fit state is invalid at fit_end={end}")
        if meta.get("validation_labels_passed_to_fit") or meta.get("test_data_read_or_used"):
            raise RuntimeError(f"SVD128 cache boundary flag invalid at fit_end={end}")
        if any([
            array_digest(raw) != meta.get("raw_prediction_sha256"),
            array_digest(mapped) != meta.get("mapped_prediction_sha256"),
            array_digest(explained) != meta.get("explained_variance_ratio_sha256"),
            positions_digest(fit_rows) != meta.get("fit_positions_sha256"),
            positions_digest(rows) != meta.get("validation_positions_sha256"),
        ]):
            raise RuntimeError(f"SVD128 output integrity failed at fit_end={end}")
        entries[str(end)] = {
            "fit_end": end,
            "param": control["param"],
            "rank_weight_frozen_from_stage5": control["rank_weight"],
            "fit_positions_sha256": positions_digest(fit_rows),
            "validation_positions_sha256": positions_digest(rows),
            "state_sha256": meta["state_sha256"],
            "raw_prediction_sha256": meta["raw_prediction_sha256"],
            "mapped_prediction_sha256": meta["mapped_prediction_sha256"],
            "explained_variance_ratio_sha256": meta["explained_variance_ratio_sha256"],
            "npz_file_sha256": _sha_file(npz_path),
            "metadata_file_sha256": _sha_file(meta_path),
            "prediction_cache_key": expected_key,
        }
    if len(entries) != 6:
        raise RuntimeError("Post-fit amendment requires all six original SVD128 outer fit caches")
    return entries


def _authorize_postfit_reporting_amendment(receipt: dict, current_identity: dict,
                                           control_audits: dict) -> dict:
    original_identity = receipt["identity"]
    if _identity_without_reporting_code(original_identity) != _identity_without_reporting_code(current_identity):
        raise RuntimeError("Post-fit amendment refused: data, protocol, model code, controls, or references changed")
    old_hashes = original_identity["autonomous_code_sha256"]
    new_hashes = current_identity["autonomous_code_sha256"]
    changed = [
        path for path in ["scripts/research_autonomous_regression.py", "scripts/verify_autonomous_regression.py"]
        if old_hashes.get(path) != new_hashes.get(path)
    ]
    if not changed:
        raise RuntimeError("Identity changed without a reporting-code amendment")
    if not _load_json(PROTOCOL_PATH).get("fit_budget", {}).get("paired_svd128_outer_fits") == 6:
        raise RuntimeError("Frozen six-fit protocol check failed")
    cache_manifest = _candidate_cache_manifest(control_audits, receipt["signature"])
    amendment = {
        "status": "APPROVED_POSTFIT_REPORTING_ONLY_AMENDMENT",
        "approval_context": "Parent agent authorized report-only correction after the six frozen candidate fits completed.",
        "original_freeze_signature": receipt["signature"],
        "original_protocol_sha256": digest_json(_load_json(PROTOCOL_PATH)),
        "stable_identity_sha256": digest_json(_identity_without_reporting_code(original_identity)),
        "pre_fit_reporting_code_sha256": {path: old_hashes[path] for path in changed},
        "post_fit_reporting_code_sha256": {path: new_hashes[path] for path in changed},
        "supersedes_postfit_reporting_amendment_sha256": (
            _sha_file(PRIOR_AMENDMENT_PATH) if PRIOR_AMENDMENT_PATH.is_file() else None
        ),
        "unchanged_model_code_sha256": old_hashes["src/models/autonomous_regression.py"],
        "scope": "Correct aggregation to select canonical W1-W6 primary validation rows once per cycle and compute target heterogeneity over those 38,588 unique rows. H12_E/H12_L remain separately reported and deliberately overlap primary windows. Render target p10-p90 and median on the natural 0-100 axis. The parent-approved, separately frozen inner-refinement stage is now available in the runner; it does not alter or reuse the six Stage 1 model fits except through hash-verified raw prediction reuse. No Stage 1 fit, parameter, feature, mapping, fold, data, protocol, metric, or gate changed.",
        "fit_budget": {"original_svd128_outer_fits": 6, "new_fits": 0, "full_train_fits": 0, "test_inference": False},
        "candidate_cache_manifest": cache_manifest,
    }
    amendment["signature"] = digest_json(amendment)
    if AMENDMENT_PATH.exists():
        old = _load_json(AMENDMENT_PATH)
        if old != amendment:
            raise RuntimeError("Existing post-fit reporting amendment differs from the verified amendment")
    else:
        _write_json(AMENDMENT_PATH, amendment, exclusive=True)
    return current_identity


def _freeze_inner_stage(frame: pd.DataFrame, train_path: Path, base_receipt: dict) -> dict:
    protocol = _load_json(INNER_PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_INNER_FIT":
        raise RuntimeError("Inner refinement protocol has not been frozen before fitting")
    stage5_freeze = _load_json(STAGE5_FREEZE)
    identity = {
        "protocol_sha256": digest_json(protocol),
        "first_stage_freeze_signature": base_receipt["signature"],
        "first_stage_report_sha256": _sha_file(REPORT / "results.json"),
        "first_stage_postfit_amendment_sha256": _sha_file(AMENDMENT_PATH),
        "stage5_signature": stage5_freeze["signature"],
        "train_sha256": _sha_file(train_path),
        "train_rows": int(len(frame)),
        "code_sha256": {path: code_digest(PROJECT_ROOT / path) for path in CODE_PATHS},
        "reference_sha256": {
            path: code_digest(PROJECT_ROOT / path)
            for path in [
                "experiments/autonomous_eda/regression/protocol.json",
                "experiments/autonomous_eda/regression/freeze.json",
                "experiments/autonomous_eda/regression/postfit_reporting_amendment_v1.json",
                "experiments/autonomous_eda/regression/postfit_reporting_amendment_v2.json",
                "experiments/autonomous_eda/regression/postfit_reporting_amendment_v3.json",
                "experiments/autonomous_eda/regression/postfit_reporting_amendment_v4.json",
                "reports/autonomous_eda/regression/results.json",
                "reports/regression_research/stage5_lgb_text.json",
                "experiments/regression_research/stage5_lgb_text_freeze.json",
            ]
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]
        },
    }
    signature = digest_json(identity)
    if INNER_FREEZE_PATH.is_file():
        old = _load_json(INNER_FREEZE_PATH)
        if old.get("identity") != identity or old.get("signature") != signature:
            raise RuntimeError("Inner refinement freeze identity changed")
        return old
    receipt = {
        "status": "FROZEN_BEFORE_FIRST_INNER_FIT",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": protocol,
        "identity": identity,
        "signature": signature,
    }
    _write_json(INNER_FREEZE_PATH, receipt, exclusive=True)
    return receipt


def _cache_one(frame: pd.DataFrame, end: int, selected: dict, rows: np.ndarray, fit_rows: np.ndarray,
               freeze_receipt: dict) -> tuple[np.ndarray, dict]:
    stem = f"svd128__{selected['param']}__fit{end}__to{int(frame.iloc[rows].cycle_code.max())}"
    npz_path, json_path = RUN / f"{stem}.npz", RUN / f"{stem}.json"
    key = digest_json({
        "freeze_signature": freeze_receipt["signature"],
        "params_id": selected["param"],
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_positions_sha256": positions_digest(rows),
    })
    if npz_path.exists() != json_path.exists():
        raise RuntimeError(f"Incomplete SVD128 cache pair: {stem}")
    if npz_path.exists():
        meta = _load_json(json_path)
        with np.load(npz_path, allow_pickle=False) as z:
            cached_fit, cached_rows = z["fit_rows"].copy(), z["validation_rows"].copy()
            raw, mapped = z["raw_prediction"].copy(), z["mapped_prediction"].copy()
            explained = z["explained_variance_ratio"].copy()
        if (meta.get("key") != key or meta.get("state_sha256") != digest_json(meta.get("state"))
                or not np.array_equal(cached_fit, fit_rows) or not np.array_equal(cached_rows, rows)
                or array_digest(raw) != meta.get("raw_prediction_sha256")
                or array_digest(mapped) != meta.get("mapped_prediction_sha256")
                or array_digest(explained) != meta.get("explained_variance_ratio_sha256")):
            raise RuntimeError(f"SVD128 cache audit mismatch: {stem}")
        return mapped, meta

    model = build_autonomous_regression(selected["param"])
    features = list(model.feature_names)
    if "threat_rank" in features or "record_id" in features:
        raise RuntimeError("Target/identifier is in the declared feature allowlist")
    fit_frame = frame.iloc[fit_rows]
    start = perf_counter()
    with threadpool_limits(limits=2):
        model.fit(fit_frame.loc[:, features].copy(), fit_frame["threat_rank"].to_numpy(copy=True))
        state = model.audit_state()
        raw = model.predict(frame.iloc[rows].loc[:, features].copy())
    if state["svd"]["n_components_requested"] != 128:
        raise RuntimeError("SVD128 request is absent from model audit state")
    if state["validation_data_passed_to_fit"] or state["test_data_used_for_fit"]:
        raise RuntimeError("Model fit audit flags validation or Test use")
    if raw.shape != (len(rows),) or not np.isfinite(raw).all():
        raise RuntimeError("SVD128 model returned invalid raw predictions")
    mapped = mapped_scores(raw, frame.iloc[rows].cycle_code.to_numpy(), float(selected["rank_weight"]))
    if not np.isfinite(mapped).all() or np.any((mapped < 0) | (mapped > 100)):
        raise RuntimeError("SVD128 mapped predictions are invalid or out of range")
    explained = np.asarray(model._svd.explained_variance_ratio_, dtype=np.float64)
    if explained.shape != (state["svd"]["n_components_actual"],) or not np.isfinite(explained).all():
        raise RuntimeError("SVD spectrum is misaligned or nonfinite")
    audit = {
        "key": key,
        "fit_end": int(end),
        "params_id": selected["param"],
        "rank_weight_frozen_from_stage5": float(selected["rank_weight"]),
        "fit_rows": int(len(fit_rows)),
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_rows": int(len(rows)),
        "validation_positions_sha256": positions_digest(rows),
        "state": state,
        "state_sha256": digest_json(state),
        "raw_prediction_sha256": array_digest(raw),
        "mapped_prediction_sha256": array_digest(mapped),
        "explained_variance_ratio_sha256": array_digest(explained),
        "target_blind_svd_spectrum": {
            "component_count": int(len(explained)),
            "variance_ratio_1_32": float(explained[:32].sum()),
            "variance_ratio_33_64": float(explained[32:64].sum()),
            "variance_ratio_65_128": float(explained[64:128].sum()),
            "variance_ratio_1_128": float(explained.sum()),
        },
        "seconds": float(perf_counter() - start),
        "validation_labels_passed_to_fit": False,
        "test_data_read_or_used": False,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    temp_npz = npz_path.with_suffix(".npz.tmp")
    with temp_npz.open("wb") as handle:
        np.savez_compressed(
            handle, fit_rows=fit_rows, validation_rows=rows,
            raw_prediction=raw, mapped_prediction=mapped,
            explained_variance_ratio=explained,
        )
    os.replace(temp_npz, npz_path)
    temp_json = json_path.with_suffix(".json.tmp")
    _write_json(temp_json, audit)
    os.replace(temp_json, json_path)
    print(f"fit_end={end} param={selected['param']} SVD128 rows={len(fit_rows)} seconds={audit['seconds']:.1f}", flush=True)
    return mapped, audit


def _fit_inner_prediction(frame: pd.DataFrame, end: int, param: str,
                          fit_rows: np.ndarray, validation_rows: np.ndarray,
                          receipt: dict) -> tuple[np.ndarray, dict]:
    fit_end, validation_start = end - 4, end - 3
    stem = f"inner128__{param}__fit{fit_end}__val{end}"
    npz_path, json_path = RUN / f"{stem}.npz", RUN / f"{stem}.json"
    key = digest_json({
        "inner_freeze_signature": receipt["signature"],
        "params_id": param,
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_positions_sha256": positions_digest(validation_rows),
    })
    if npz_path.exists() != json_path.exists():
        raise RuntimeError(f"Incomplete inner SVD128 cache pair: {stem}")
    if npz_path.exists():
        meta = _load_json(json_path)
        with np.load(npz_path, allow_pickle=False) as z:
            cached_fit, cached_val = z["fit_rows"].copy(), z["validation_rows"].copy()
            prediction = z["raw_prediction"].copy()
            explained = z["explained_variance_ratio"].copy()
        if (meta.get("key") != key or meta.get("state_sha256") != digest_json(meta.get("state"))
                or not np.array_equal(cached_fit, fit_rows) or not np.array_equal(cached_val, validation_rows)
                or array_digest(prediction) != meta.get("raw_prediction_sha256")
                or array_digest(explained) != meta.get("explained_variance_ratio_sha256")):
            raise RuntimeError(f"Inner SVD128 cache audit mismatch: {stem}")
        meta["cache_reused"] = True
        return prediction, meta

    if frame.iloc[fit_rows].cycle_code.max() >= frame.iloc[validation_rows].cycle_code.min():
        raise RuntimeError(f"Inner fit/validation time order failed for cutoff {end}")
    model = build_autonomous_regression(param)
    features = list(model.feature_names)
    fit_frame = frame.iloc[fit_rows]
    start = perf_counter()
    with threadpool_limits(limits=2):
        model.fit(fit_frame.loc[:, features].copy(), fit_frame.threat_rank.to_numpy(copy=True))
        state = model.audit_state()
        prediction = model.predict(frame.iloc[validation_rows].loc[:, features].copy())
    if (state["svd"]["n_components_requested"] != 128 or state["validation_data_passed_to_fit"]
            or state["test_data_used_for_fit"]):
        raise RuntimeError(f"Inner SVD128 audit boundary failed for cutoff {end}")
    if prediction.shape != (len(validation_rows),) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Inner SVD128 predictions invalid for cutoff {end}")
    explained = np.asarray(model._svd.explained_variance_ratio_, dtype=np.float64)
    meta = {
        "key": key,
        "outer_fit_end": int(end),
        "inner_fit_cycle_range": [47, fit_end],
        "inner_validation_cycle_range": [validation_start, end],
        "params_id": param,
        "fit_rows": int(len(fit_rows)),
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_rows": int(len(validation_rows)),
        "validation_positions_sha256": positions_digest(validation_rows),
        "state": state,
        "state_sha256": digest_json(state),
        "raw_prediction_sha256": array_digest(prediction),
        "explained_variance_ratio_sha256": array_digest(explained),
        "seconds": float(perf_counter() - start),
        "validation_labels_passed_to_fit": False,
        "test_data_read_or_used": False,
        "cache_reused": False,
        "actual_fit_performed": True,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    temp_npz = npz_path.with_suffix(".npz.tmp")
    with temp_npz.open("wb") as handle:
        np.savez_compressed(
            handle, fit_rows=fit_rows, validation_rows=validation_rows,
            raw_prediction=prediction, explained_variance_ratio=explained,
        )
    os.replace(temp_npz, npz_path)
    temp_json = json_path.with_suffix(".json.tmp")
    _write_json(temp_json, meta)
    os.replace(temp_json, json_path)
    print(f"inner end={end} param={param} fit={len(fit_rows)} val={len(validation_rows)} s={meta['seconds']:.1f}", flush=True)
    return prediction, meta


def _fit_selected_outer_raw(frame: pd.DataFrame, end: int, param: str,
                            stage1_param: str, rows: np.ndarray, fit_rows: np.ndarray,
                            receipt: dict) -> tuple[np.ndarray, dict]:
    max_cycle = int(frame.iloc[rows].cycle_code.max())
    if param == stage1_param:
        stem = f"svd128__{param}__fit{end}__to{max_cycle}"
        npz_path, meta_path = RUN / f"{stem}.npz", RUN / f"{stem}.json"
        if not npz_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Matching Stage 1 SVD128 paired cache is missing: {stem}")
        meta = _load_json(meta_path)
        with np.load(npz_path, allow_pickle=False) as z:
            old_fit, old_rows = z["fit_rows"].copy(), z["validation_rows"].copy()
            raw, explained = z["raw_prediction"].copy(), z["explained_variance_ratio"].copy()
        if (not np.array_equal(old_fit, fit_rows) or not np.array_equal(old_rows, rows)
                or meta.get("state_sha256") != digest_json(meta.get("state"))
                or array_digest(raw) != meta.get("raw_prediction_sha256")
                or array_digest(explained) != meta.get("explained_variance_ratio_sha256")):
            raise RuntimeError(f"Matching Stage 1 SVD128 raw cache failed audit: {stem}")
        return raw, {
            "fit_end": int(end), "params_id": param,
            "source": "reused_first_stage_paired_outer_fit",
            "actual_fit_performed": False,
            "state_sha256": meta["state_sha256"],
            "raw_prediction_sha256": array_digest(raw),
            "fit_positions_sha256": positions_digest(fit_rows),
            "validation_positions_sha256": positions_digest(rows),
        }

    stem = f"outer128selected__{param}__fit{end}__to{max_cycle}"
    npz_path, meta_path = RUN / f"{stem}.npz", RUN / f"{stem}.json"
    key = digest_json({
        "inner_freeze_signature": receipt["signature"],
        "params_id": param,
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_positions_sha256": positions_digest(rows),
    })
    if npz_path.exists() != meta_path.exists():
        raise RuntimeError(f"Incomplete selected outer cache pair: {stem}")
    if npz_path.exists():
        meta = _load_json(meta_path)
        with np.load(npz_path, allow_pickle=False) as z:
            old_fit, old_rows = z["fit_rows"].copy(), z["validation_rows"].copy()
            raw = z["raw_prediction"].copy()
        if (meta.get("key") != key or meta.get("state_sha256") != digest_json(meta.get("state"))
                or not np.array_equal(old_fit, fit_rows) or not np.array_equal(old_rows, rows)
                or array_digest(raw) != meta.get("raw_prediction_sha256")):
            raise RuntimeError(f"Selected outer cache audit mismatch: {stem}")
        meta["cache_reused"] = True
        return raw, meta

    model = build_autonomous_regression(param)
    features = list(model.feature_names)
    fit_frame = frame.iloc[fit_rows]
    start = perf_counter()
    with threadpool_limits(limits=2):
        model.fit(fit_frame.loc[:, features].copy(), fit_frame.threat_rank.to_numpy(copy=True))
        state = model.audit_state()
        raw = model.predict(frame.iloc[rows].loc[:, features].copy())
    if state["validation_data_passed_to_fit"] or state["test_data_used_for_fit"]:
        raise RuntimeError(f"Selected outer SVD128 audit boundary failed: {stem}")
    if raw.shape != (len(rows),) or not np.isfinite(raw).all():
        raise RuntimeError(f"Selected outer SVD128 predictions invalid: {stem}")
    meta = {
        "key": key, "fit_end": int(end), "params_id": param,
        "fit_rows": int(len(fit_rows)), "fit_positions_sha256": positions_digest(fit_rows),
        "validation_rows": int(len(rows)), "validation_positions_sha256": positions_digest(rows),
        "state": state, "state_sha256": digest_json(state),
        "raw_prediction_sha256": array_digest(raw),
        "seconds": float(perf_counter() - start),
        "validation_labels_passed_to_fit": False, "test_data_read_or_used": False,
        "cache_reused": False, "actual_fit_performed": True,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    temp_npz = npz_path.with_suffix(".npz.tmp")
    with temp_npz.open("wb") as handle:
        np.savez_compressed(handle, fit_rows=fit_rows, validation_rows=rows, raw_prediction=raw)
    os.replace(temp_npz, npz_path)
    temp_json = meta_path.with_suffix(".json.tmp")
    _write_json(temp_json, meta)
    os.replace(temp_json, meta_path)
    print(f"selected outer end={end} param={param} refit rows={len(fit_rows)} seconds={meta['seconds']:.1f}", flush=True)
    return raw, meta


def _collect_predictions(frame: pd.DataFrame, vectors: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    primary, _ = _window_specs()
    # Outer horizons intentionally overlap because long H12 predictions reuse
    # the earliest and middle fit cutoffs. The primary windows select each
    # development row once and define canonical target-heterogeneity coverage.
    rows, pred = _window_vectors(frame, vectors, primary)
    if len(rows) != 38588 or len(rows) != len(np.unique(rows)):
        raise RuntimeError(f"Canonical W1-W6 coverage is not 38,588 unique rows: {len(rows)}")
    return rows, pred


def _window_vectors(frame: pd.DataFrame, vectors: dict[int, tuple[np.ndarray, np.ndarray]], windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows_all, pred_all = [], []
    for window in windows:
        end = int(window["fit"][1])
        rows, pred = vectors[end]
        mask = frame.iloc[rows].cycle_code.between(*window["validation"]).to_numpy()
        rows_all.append(rows[mask])
        pred_all.append(pred[mask])
    rows, pred = np.concatenate(rows_all), np.concatenate(pred_all)
    order = np.argsort(rows)
    return rows[order], pred[order]


def _simple_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    return {
        "n": int(len(actual)),
        "raw_mae": float(np.mean(np.abs(error))),
        "signed_error": float(np.mean(error)),
        "over_rate": float(np.mean(error > 1e-10)),
        "under_rate": float(np.mean(error < -1e-10)),
        "absolute_error_p95": float(np.quantile(np.abs(error), .95)),
    }


def _measure(frame: pd.DataFrame, vectors: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict:
    primary, secondary = _window_specs()
    windows = {}
    for window in [*primary, *secondary]:
        name = window["id"]
        rows, pred = _window_vectors(frame, vectors, [window])
        summary = score_summary(frame.iloc[rows], pred, impact=False)
        windows[name] = {"rows": rows, "prediction": pred, "summary": summary}
    aggregates = {}
    for group, spec in [("primary", primary), ("h12", secondary)]:
        rows, pred = _window_vectors(frame, vectors, spec)
        summary = score_summary(frame.iloc[rows], pred, impact=False)
        aggregates[group] = {
            "rows": rows,
            "prediction": pred,
            "metrics": {
                "raw_mae": summary["raw"]["mae"],
                "equal_cycle_mae": summary["equal_cycle_mae"],
                "signed_error": summary["raw"]["signed_error"],
                "absolute_error_p95": summary["raw"]["absolute_error_p95"],
            },
            "summary": summary,
        }
    return {"windows": windows, "aggregate": aggregates}


def _paired_metrics(frame: pd.DataFrame, rows: np.ndarray, candidate: np.ndarray, control: np.ndarray) -> dict:
    actual = frame.iloc[rows].threat_rank.to_numpy(dtype=float)
    cycle = frame.iloc[rows].cycle_code.to_numpy()
    error_new = np.abs(candidate - actual)
    error_old = np.abs(control - actual)
    delta = error_new - error_old
    by_cycle = []
    for value in sorted(pd.unique(cycle)):
        mask = cycle == value
        by_cycle.append({
            "cycle": int(value),
            "n": int(mask.sum()),
            "control_mae": float(error_old[mask].mean()),
            "candidate_mae": float(error_new[mask].mean()),
            "delta_mae": float(delta[mask].mean()),
        })
    return {
        "n": int(len(rows)),
        "control_mae": float(error_old.mean()),
        "candidate_mae": float(error_new.mean()),
        "delta_mae": float(delta.mean()),
        "candidate_win_rate": float(np.mean(delta < -1e-10)),
        "control_win_rate": float(np.mean(delta > 1e-10)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-10)),
        "by_cycle": by_cycle,
    }


def _analysis(frame: pd.DataFrame, candidate_vectors: dict, control_vectors: dict, audits: dict) -> dict:
    candidate = _measure(frame, candidate_vectors)
    control = _measure(frame, control_vectors)
    result: dict = {
        "windows": {},
        "aggregate": {},
        "spectra_by_fit_end": {},
        "target_heterogeneity_by_validation_cycle": {},
    }
    for key, candidate_window in candidate["windows"].items():
        control_window = control["windows"][key]
        pair = _paired_metrics(
            frame,
            candidate_window["rows"],
            candidate_window["prediction"],
            control_window["prediction"],
        )
        result["windows"][key] = {
            "rows": pair["n"],
            "control_mae": pair["control_mae"],
            "candidate_mae": pair["candidate_mae"],
            "delta_mae": pair["delta_mae"],
            "control_equal_cycle_mae": control_window["summary"]["equal_cycle_mae"],
            "candidate_equal_cycle_mae": candidate_window["summary"]["equal_cycle_mae"],
            "control_signed_error": float(np.mean(control_window["prediction"] - frame.iloc[control_window["rows"]].threat_rank.to_numpy(dtype=float))),
            "candidate_signed_error": float(np.mean(candidate_window["prediction"] - frame.iloc[candidate_window["rows"]].threat_rank.to_numpy(dtype=float))),
            "paired_win_rate": pair["candidate_win_rate"],
            "control_subgroups": control_window["summary"]["subgroups"],
            "candidate_subgroups": candidate_window["summary"]["subgroups"],
            "per_cycle": pair["by_cycle"],
        }
    for key in ["primary", "h12"]:
        left, right = candidate["aggregate"][key], control["aggregate"][key]
        pair = _paired_metrics(frame, left["rows"], left["prediction"], right["prediction"])
        result["aggregate"][key] = {
            "n": pair["n"],
            "control": right["metrics"],
            "candidate": left["metrics"],
            "delta_mae": pair["delta_mae"],
            "candidate_win_rate": pair["candidate_win_rate"],
            "control_win_rate": pair["control_win_rate"],
            "tie_rate": pair["tie_rate"],
            "subgroups": {
                name: {
                    "control": right["summary"]["subgroups"][name],
                    "candidate": left["summary"]["subgroups"][name],
                }
                for name in ["EF0", "EF1", "tail_low_le5", "tail_high_ge95", "stream_missing", "stream_present",
                             "region_low", "region_middle", "region_high"]
            },
            "per_cycle": pair["by_cycle"],
        }
    for end, audit in sorted(audits.items(), key=lambda item: int(item[0])):
        result["spectra_by_fit_end"][end] = audit["target_blind_svd_spectrum"]
    rows, _ = _collect_predictions(frame, candidate_vectors)
    val = frame.iloc[rows]
    for cycle, group in val.groupby("cycle_code", sort=True):
        y = group.threat_rank.to_numpy(dtype=float)
        result["target_heterogeneity_by_validation_cycle"][str(int(cycle))] = {
            "n": int(len(y)),
            "mean": float(np.mean(y)),
            "sd": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "p10": float(np.quantile(y, .10)),
            "median": float(np.median(y)),
            "p90": float(np.quantile(y, .90)),
        }
    return result


def _plot_results(result: dict, path: Path) -> list[str]:
    figures = path / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = []

    # Figure 1 keeps target-blind text spectrum and target distribution distinct.
    spectrum = result["spectra_by_fit_end"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ends = [int(k) for k in sorted(spectrum, key=int)]
    labels = [f"{end}" for end in ends]
    blocks = ["variance_ratio_1_32", "variance_ratio_33_64", "variance_ratio_65_128"]
    values = np.array([[spectrum[str(end)][block] for block in blocks] for end in ends])
    bottom = np.zeros(len(ends))
    colors = ["#2563EB", "#0F766E", "#F59E0B"]
    for idx, (name, color) in enumerate(zip(["1–32", "33–64", "65–128"], colors)):
        axes[0].bar(labels, values[:, idx] * 100, bottom=bottom * 100, label=name, color=color)
        bottom += values[:, idx]
    axes[0].set_title("TF-IDF variance explained by SVD block")
    axes[0].set_xlabel("Fit-end cycle")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].legend(title="Components", frameon=False)
    axes[0].grid(axis="y", alpha=.22)

    heter = result["target_heterogeneity_by_validation_cycle"]
    cycles = [int(k) for k in sorted(heter, key=int)]
    p10 = [heter[str(c)]["p10"] for c in cycles]
    median = [heter[str(c)]["median"] for c in cycles]
    p90 = [heter[str(c)]["p90"] for c in cycles]
    axes[1].fill_between(cycles, p10, p90, color="#C4B5FD", alpha=.55, label="p10–p90")
    axes[1].plot(cycles, median, color="#6D28D9", marker="o", linewidth=1.8, label="median")
    axes[1].set_title("Observed target distribution on validation cycles")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("threat_rank (0–100)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.25)
    fig.suptitle("Representation spectrum and target heterogeneity", fontsize=13, fontweight="bold")
    target = figures / "spectrum_and_target_heterogeneity.png"
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)
    written.append(target.relative_to(path).as_posix())

    cycle_rows = result["aggregate"]["primary"]["per_cycle"] + result["aggregate"]["h12"]["per_cycle"]
    # Same cycles occur in primary/H12 coverage; deduplicate with the primary/H12 value equality check.
    by_cycle = {int(item["cycle"]): item for item in cycle_rows}
    cyc = sorted(by_cycle)
    deltas = [by_cycle[c]["delta_mae"] for c in cyc]
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    colors = ["#047857" if value < 0 else "#B91C1C" for value in deltas]
    ax.bar(cyc, deltas, color=colors, width=.72)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_title("Paired per-cycle MAE difference: SVD128 minus SVD32")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Δ MAE (negative favors SVD128)")
    ax.grid(axis="y", alpha=.22)
    fig.text(.01, -.01, "Cycle-level view of paired validation rows; SVD32 rank weights and model parameters are frozen.", fontsize=9)
    target = figures / "paired_cycle_mae_delta.png"
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)
    written.append(target.relative_to(path).as_posix())
    return written


def _render_markdown(result: dict, report_path: Path) -> None:
    ag = result["aggregate"]
    lines = [
        "# SVD128 LightGBM paired ablation",
        "",
        "## Research question",
        "",
        "Could TF-IDF information represented by SVD components 33–128 add predictive signal for `threat_rank` beyond the frozen SVD32 LightGBM recipe?",
        "",
        "This is a representation-resolution ablation. SVD uses no target values; explained-variance summaries are target-blind descriptions of the fit-fold text matrices. Paired errors are retrospective model evidence on the existing temporal development windows, not a causal or semantic claim.",
        "",
        "## Frozen comparison",
        "",
        "Only the requested SVD width changes. The selected Stage 5 parameter and per-cycle rank weight remain fixed for each fit end. Six outer fits cover the same six primary windows and H12 early/later periods; no SVD128 parameter or mapping selection was performed.",
        "",
        "| Evaluation | Rows | SVD32 MAE | SVD128 MAE | Δ MAE | Equal-cycle SVD32 | Equal-cycle SVD128 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in [("primary", "Primary cycles 56–79"), ("h12", "H12 pooled (56–67, 68–79)")]:
        v = ag[key]
        lines.append(
            f"| {title} | {v['n']:,} | {v['control']['raw_mae']:.6f} | {v['candidate']['raw_mae']:.6f} | {v['delta_mae']:+.6f} | {v['control']['equal_cycle_mae']:.6f} | {v['candidate']['equal_cycle_mae']:.6f} |"
        )
    lines += ["", "### Temporal windows", "", "| Window | Rows | SVD32 MAE | SVD128 MAE | Δ MAE | Paired candidate win rate |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in result["windows"].items():
        lines.append(f"| {key} | {value['rows']:,} | {value['control_mae']:.6f} | {value['candidate_mae']:.6f} | {value['delta_mae']:+.6f} | {value['paired_win_rate']:.3f} |")
    lines += ["", "## Spectrum and target heterogeneity", "", "The first figure separates fit-fold, target-blind explained variance by SVD component block from observed `threat_rank` p10–p90 span and median on a common 0–100 scale. It does not establish that any component has a particular meaning. The cycle-level paired error plot shows where added representation helps or hurts.", ""]
    for image in result["figures"]:
        lines.append(f"![{Path(image).stem}]({image})")
        lines.append("")
    lines += ["## Interpretation and limitations", "", f"The paired SVD128 ablation changes primary MAE by {ag['primary']['delta_mae']:+.6f} and H12 MAE by {ag['h12']['delta_mae']:+.6f} relative to the exact frozen Stage 5 controls. A negative value favors SVD128. Interpret this as evidence about this one LightGBM recipe and these reused temporal windows; the windows are development data, so results are adaptive rather than an untouched holdout.", "", "TF-IDF/SVD is not semantically interpreted. Greater explained variance is not automatically predictive value. Missing and present streams, entry-friction groups, target regions, and target tails are summarized as model guardrails; this analysis does not ask an explanatory question about those member-owned EDA features. No full-train/Test fit or Kaggle action was performed.", "", "## Reproducibility", "", "From the repository root with official raw data available at `data/raw/regression/Train.csv` (or `TP2_DATA_DIR`):", "", "```powershell", "python scripts/research_autonomous_regression.py", "python scripts/verify_autonomous_regression.py", "```", "", "The run manifest freezes code, Train input identity, Stage 5 control-vector hashes, selected parameters, rank weights, fold definitions, fit audits, and prediction hashes. Model fit receives only its training rows and their targets."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _inner_outer_comparison(frame: pd.DataFrame, inner_vectors: dict,
                            fixed_vectors: dict) -> dict:
    inner, fixed = _measure(frame, inner_vectors), _measure(frame, fixed_vectors)
    report = {"windows": {}, "aggregate": {}}
    for key, item in inner["windows"].items():
        old = fixed["windows"][key]
        pair = _paired_metrics(frame, item["rows"], item["prediction"], old["prediction"])
        report["windows"][key] = {
            "n": pair["n"],
            "fixed_policy_mae": pair["control_mae"],
            "inner_selected_mae": pair["candidate_mae"],
            "delta_mae": pair["delta_mae"],
            "candidate_win_rate": pair["candidate_win_rate"],
            "per_cycle": pair["by_cycle"],
        }
    for key in ["primary", "h12"]:
        item, old = inner["aggregate"][key], fixed["aggregate"][key]
        pair = _paired_metrics(frame, item["rows"], item["prediction"], old["prediction"])
        report["aggregate"][key] = {
            "n": pair["n"],
            "fixed_policy": fixed["aggregate"][key]["metrics"],
            "inner_selected": inner["aggregate"][key]["metrics"],
            "delta_mae": pair["delta_mae"],
            "candidate_win_rate": pair["candidate_win_rate"],
            "per_cycle": pair["by_cycle"],
        }
    return report


def _plot_inner_comparison(result: dict) -> str:
    keys = ["W1", "W2", "W3", "W4", "W5", "W6"]
    stage1 = _load_json(REPORT / "results.json")
    windows = result["paired_vs_stage5"]["windows"]
    fixed = stage1["windows"]
    inner = result["selected_vs_stage5"]["windows"]
    x = np.arange(len(keys))
    width = .26
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    ax.bar(x - width, [fixed[k]["candidate_mae"] for k in keys], width, label="SVD128 fixed", color="#0F766E")
    ax.bar(x, [inner[k]["candidate_mae"] for k in keys], width, label="SVD128 inner-selected", color="#7C3AED")
    ax.bar(x + width, [windows[k]["control_mae"] for k in keys], width, label="SVD32 Stage 5", color="#64748B")
    ax.set_xticks(x, keys)
    ax.set_ylabel("MAE")
    ax.set_title("Primary temporal windows: fixed and inner-selected SVD128")
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=.22)
    target = REPORT / "figures/inner_refinement_window_mae.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return target.relative_to(REPORT).as_posix()


def _render_inner_markdown(result: dict) -> None:
    lines = [
        "# SVD128 LightGBM inner-selected refinement",
        "",
        "## Frozen refinement",
        "",
        "After the paired SVD128 ablation showed a near-neutral, mixed primary effect but lower MAE on both H12 halves, the parent approved this bounded re-selection of the existing two LightGBM parameter sets and five rank weights. Only past inner-validation labels selected each fold policy. No new features, model family, parameter values, ranking weights, folds, Test prediction, or leaderboard use were introduced.",
        "",
        "The twelve inner fits cover six past-only folds, two parameter sets per fold. The outer stage reuses the paired SVD128 fit when its parameter matches the inner choice and fits only an absent selected parameter when needed.",
        "",
        "## Selections",
        "",
        "| Outer fit end | Stage 5 SVD32 choice | SVD128 inner choice | Inner MAE | Changed parameter? | Changed weight? |",
        "|---:|---|---|---:|---|---|",
    ]
    for item in result["selections"]:
        lines.append(
            f"| {item['outer_fit_end']} | {item['stage5_svd32']['param']} / {item['stage5_svd32']['rank_weight']:.2f} | {item['selected']['param']} / {item['selected']['rank_weight']:.2f} | {item['selected']['mae']:.6f} | {item['parameter_changed_vs_stage5']} | {item['weight_changed_vs_stage5']} |"
        )
    lines += ["", "## Temporal results", "", "| Evaluation | Rows | SVD32 Stage 5 | SVD128 fixed policy | SVD128 inner-selected | Inner vs fixed Δ | Inner vs SVD32 Δ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, title in [("primary", "Primary W1–W6"), ("h12", "H12 pooled")]:
        fixed = result["paired_vs_stage5"]["aggregate"][key]
        selected = result["selected_vs_stage5"]["aggregate"][key]
        within = result["inner_vs_fixed"]["aggregate"][key]
        lines.append(
            f"| {title} | {selected['n']:,} | {fixed['control']['raw_mae']:.6f} | {fixed['candidate']['raw_mae']:.6f} | {selected['candidate']['raw_mae']:.6f} | {within['delta_mae']:+.6f} | {selected['delta_mae']:+.6f} |"
        )
    lines += ["", "### Per-window MAE", "", "| Window | SVD32 | SVD128 fixed | SVD128 inner-selected | Δ inner vs fixed | Δ inner vs SVD32 |", "|---|---:|---:|---:|---:|---:|"]
    for key in ["W1", "W2", "W3", "W4", "W5", "W6", "H12_E", "H12_L"]:
        fixed = result["paired_vs_stage5"]["windows"][key]
        selected = result["selected_vs_stage5"]["windows"][key]
        within = result["inner_vs_fixed"]["windows"][key]
        lines.append(f"| {key} | {fixed['control_mae']:.6f} | {fixed['candidate_mae']:.6f} | {selected['candidate_mae']:.6f} | {within['delta_mae']:+.6f} | {selected['delta_mae']:+.6f} |")
    lines += ["", "## Guardrails", "", "Primary and H12 subgroup error tables, tails, missing/present streams, signed error, and cycle deltas are stored in `inner_refinement.json`. These diagnostics do not redefine the member EDA questions.", "", "![Primary temporal comparison](figures/inner_refinement_window_mae.png)", "", "## Fit and boundary audit", "", f"Exactly {result['fit_budget']['inner_svd128_fits']} inner SVD128 fits were performed. {result['fit_budget']['selected_outer_svd128_fits']} additional selected-parameter outer fits were performed; {result['fit_budget']['reused_first_stage_outer_fits']} outer fits were reused from the fixed paired stage. No full-train fit, official Test access, Kaggle submission, or leaderboard feedback occurred.", "", "The standard errors remain internal development evidence. Any later full-train parameter selection requires a separate decision and a fit-cycle 47–75 / validation-cycle 76–79 boundary; this report does not select a final inference recipe."]
    INNER_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_inner_refinement() -> None:
    train_path = task_files("regression")["train"]
    if train_path is None:
        raise FileNotFoundError("Official regression Train.csv was not found")
    frame = pd.read_csv(train_path)
    stage5 = _load_json(STAGE5_REPORT)
    choices_stage5 = _stage5_choices(stage5)
    primary, secondary = _window_specs()
    control_vectors, control_audits = {}, {}
    for end, choice in sorted(choices_stage5.items()):
        rows, pred, audit = _load_control(frame, end, choice, primary, secondary)
        control_vectors[end] = (rows, pred)
        control_audits[str(end)] = audit
    base_receipt = freeze(frame, train_path, control_vectors, control_audits, report_only=True)
    # Keep the first-stage report tied to the final report-only correction before
    # freezing this new selection stage. This reuses six existing fit caches.
    _verify_inner_base_artifacts(frame, base_receipt)
    stage_receipt = _freeze_inner_stage(frame, train_path, base_receipt)
    protocol = _load_json(INNER_PROTOCOL_PATH)
    if digest_json(protocol) != stage_receipt["identity"]["protocol_sha256"]:
        raise RuntimeError("Inner refinement protocol changed after stage freeze")
    inner_vectors_by_end: dict[int, dict[str, tuple[np.ndarray, dict]]] = {}
    selections, audits_inner = [], {}
    for fold in protocol["temporal_inner_folds"]:
        end = int(fold["outer_fit_end"])
        lo_fit, hi_fit = fold["inner_fit"]
        lo_val, hi_val = fold["inner_validation"]
        fit_rows = np.flatnonzero(frame.cycle_code.between(lo_fit, hi_fit).to_numpy())
        val_rows = np.flatnonzero(frame.cycle_code.between(lo_val, hi_val).to_numpy())
        if frame.iloc[fit_rows].cycle_code.max() >= frame.iloc[val_rows].cycle_code.min():
            raise RuntimeError(f"Inner fold chronology failed at fit_end={end}")
        per_param = {}
        records = []
        y_val = frame.iloc[val_rows].threat_rank.to_numpy(dtype=float)
        cycles_val = frame.iloc[val_rows].cycle_code.to_numpy()
        for param in protocol["candidate_grid"]["params"]:
            raw, meta = _fit_inner_prediction(frame, end, param, fit_rows, val_rows, stage_receipt)
            per_param[param] = (raw, meta)
            for weight in protocol["candidate_grid"]["rank_weights"]:
                pred = mapped_scores(raw, cycles_val, float(weight))
                records.append({
                    "param": param,
                    "rank_weight": float(weight),
                    "mae": float(np.mean(np.abs(pred - y_val))),
                })
        selected = select_inner_policy(records)
        stage5_choice = choices_stage5[end]
        inner_vectors_by_end[end] = per_param
        audits_inner[str(end)] = {param: meta for param, (_, meta) in per_param.items()}
        selections.append({
            "outer_fit_end": end,
            "inner_fit_cycles": [int(lo_fit), int(hi_fit)],
            "inner_validation_cycles": [int(lo_val), int(hi_val)],
            "selected": selected,
            "stage5_svd32": {
                "param": stage5_choice["param"],
                "rank_weight": float(stage5_choice["rank_weight"]),
            },
            "parameter_changed_vs_stage5": selected["param"] != stage5_choice["param"],
            "weight_changed_vs_stage5": not np.isclose(selected["rank_weight"], float(stage5_choice["rank_weight"])),
            "choices": records,
        })

    selected_outer_vectors, selected_outer_audits = {}, {}
    outer_fit_count, outer_reuse_count = 0, 0
    for entry in selections:
        end = entry["outer_fit_end"]
        choice = entry["selected"]
        max_cycle = max(
            int(window["validation"][1])
            for window in [*primary, *secondary]
            if int(window["fit"][1]) == end
        )
        rows = np.flatnonzero(frame.cycle_code.between(end + 1, max_cycle).to_numpy())
        fit_rows = np.flatnonzero(frame.cycle_code.between(47, end).to_numpy())
        raw, audit = _fit_selected_outer_raw(
            frame, end, choice["param"], entry["stage5_svd32"]["param"],
            rows, fit_rows, stage_receipt,
        )
        if audit.get("source") == "reused_first_stage_paired_outer_fit":
            outer_reuse_count += 1
        elif audit.get("actual_fit_performed"):
            outer_fit_count += 1
        mapped = mapped_scores(raw, frame.iloc[rows].cycle_code.to_numpy(), float(choice["rank_weight"]))
        if not np.isfinite(mapped).all() or np.any((mapped < 0) | (mapped > 100)):
            raise RuntimeError(f"Selected outer prediction invalid at fit_end={end}")
        selected_outer_vectors[end] = (rows, mapped)
        selected_outer_audits[str(end)] = audit
        entry["outer"] = {
            "source": audit.get("source", "new_selected_parameter_fit"),
            "fit_rows": int(len(fit_rows)),
            "validation_rows": int(len(rows)),
            "raw_prediction_sha256": array_digest(raw),
            "mapped_prediction_sha256": array_digest(mapped),
            "state_sha256": audit["state_sha256"],
        }

    # Fixed SVD128 uses the originally frozen Stage 5 parameter and rank weight.
    fixed_vectors = {}
    for end, choice in sorted(choices_stage5.items()):
        stem = f"svd128__{choice['param']}__fit{end}__to{int(frame.iloc[control_vectors[end][0]].cycle_code.max())}"
        with np.load(RUN / f"{stem}.npz", allow_pickle=False) as z:
            rows = z["validation_rows"].copy()
            mapped = z["mapped_prediction"].copy()
        fixed_vectors[end] = (rows, mapped)

    stage1_result = _load_json(REPORT / "results.json")
    selected_vs_stage5 = _analysis(frame, selected_outer_vectors, control_vectors, stage1_result["fit_audits"])
    fixed_vs_stage5 = _analysis(frame, fixed_vectors, control_vectors, {
        key: {"target_blind_svd_spectrum": value["target_blind_svd_spectrum"]}
        for key, value in _load_json(REPORT / "results.json")["fit_audits"].items()
    })
    inner_vs_fixed = _inner_outer_comparison(frame, selected_outer_vectors, fixed_vectors)
    result = {
        "status": "COMPLETED_FROZEN_SVD128_INNER_REFINEMENT",
        "protocol": protocol,
        "freeze_signature": stage_receipt["signature"],
        "first_stage_freeze_signature": base_receipt["signature"],
        "first_stage_amendment_sha256": _sha_file(AMENDMENT_PATH),
        "first_stage_report_sha256": _sha_file(REPORT / "results.json"),
        "selections": selections,
        "fit_audits_inner": audits_inner,
        "fit_audits_selected_outer": selected_outer_audits,
        "selected_vs_stage5": selected_vs_stage5,
        "paired_vs_stage5": fixed_vs_stage5,
        "inner_vs_fixed": inner_vs_fixed,
        "fit_budget": {
            "inner_svd128_fits": sum(
                int(meta.get("actual_fit_performed", False))
                for by_param in audits_inner.values() for meta in by_param.values()
            ),
            "selected_outer_svd128_fits": outer_fit_count,
            "reused_first_stage_outer_fits": outer_reuse_count,
            "full_train_fits": 0,
            "test_inference": False,
            "kaggle_submissions": 0,
        },
        "selection_labels_used": "Inner fold only, as frozen; outer labels are used only after choice for evaluation.",
        "adaptive_stage_gate": "Parent-approved single SVD128 inner refinement; no further model family or grid expansion.",
    }
    result["figure"] = _plot_inner_comparison(result)
    _write_json(INNER_REPORT_JSON, result)
    _render_inner_markdown(result)
    print("SVD128 inner-refined primary MAE:", selected_vs_stage5["aggregate"]["primary"]["candidate"]["raw_mae"], flush=True)
    print("SVD128 inner-refined H12 MAE:", selected_vs_stage5["aggregate"]["h12"]["candidate"]["raw_mae"], flush=True)
    print("Inner/new-outer/reused fit counts:", result["fit_budget"]["inner_svd128_fits"], outer_fit_count, outer_reuse_count, flush=True)


def _verify_inner_base_artifacts(frame: pd.DataFrame, base_receipt: dict) -> None:
    if not AMENDMENT_PATH.is_file():
        raise RuntimeError("Final first-stage post-fit amendment is not present")
    stage1 = _load_json(REPORT / "results.json")
    if stage1.get("freeze_signature") != base_receipt["signature"]:
        raise RuntimeError("First-stage paired report does not match its pre-fit freeze")
    if stage1.get("postfit_reporting_amendment_sha256") != _sha_file(AMENDMENT_PATH):
        raise RuntimeError("First-stage paired report does not match its latest report-only amendment")
    if stage1["aggregate"]["primary"]["n"] != 38588 or stage1["aggregate"]["h12"]["n"] != 38588:
        raise RuntimeError("First-stage primary/H12 denominator is not the frozen 38,588 rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only", action="store_true",
        help="Use all six completed frozen SVD128 caches and refuse any missing fit; no model fit is run.",
    )
    parser.add_argument(
        "--run-inner-refinement", action="store_true",
        help="Run only the separately frozen 12-fit SVD128 inner-selection refinement and selected outer fits.",
    )
    args = parser.parse_args()
    if args.report_only and args.run_inner_refinement:
        parser.error("Choose either --report-only or --run-inner-refinement")
    if args.run_inner_refinement:
        run_inner_refinement()
        return
    train_path = task_files("regression")["train"]
    if train_path is None:
        raise FileNotFoundError("Official regression Train.csv was not found")
    if not STAGE5_REPORT.is_file() or not STAGE5_FREEZE.is_file():
        raise FileNotFoundError("Frozen Stage 5 LightGBM control report/freeze is missing")
    frame = pd.read_csv(train_path)
    stage5 = _load_json(STAGE5_REPORT)
    selected = _stage5_choices(stage5)
    primary, secondary = _window_specs()
    control_vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    control_audits: dict[str, dict] = {}
    for end, choice in sorted(selected.items()):
        rows, mapped, control_meta = _load_control(frame, end, choice, primary, secondary)
        control_vectors[end] = (rows, mapped)
        control_audits[str(end)] = control_meta
    receipt = freeze(frame, train_path, control_vectors, control_audits, report_only=args.report_only)
    RUN.mkdir(parents=True, exist_ok=True)

    candidate_vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    audits = {}
    total_start = perf_counter()
    if args.report_only:
        missing = []
        for end, choice in sorted(selected.items()):
            expected_max = max(
                int(window["validation"][1])
                for window in [*primary, *secondary]
                if int(window["fit"][1]) == end
            )
            base = RUN / f"svd128__{choice['param']}__fit{end}__to{expected_max}"
            if not base.with_suffix(".npz").is_file() or not base.with_suffix(".json").is_file():
                missing.append(str(base))
        if missing:
            raise FileNotFoundError("--report-only requires all six completed SVD128 cache pairs: " + ", ".join(missing))
    for end, choice in sorted(selected.items()):
        fit_rows = np.flatnonzero(frame.cycle_code.between(47, end).to_numpy())
        expected_max = max(
            int(window["validation"][1])
            for window in [*primary, *secondary]
            if int(window["fit"][1]) == end
        )
        rows = np.flatnonzero(frame.cycle_code.between(end + 1, expected_max).to_numpy())
        if not len(fit_rows) or not len(rows) or frame.iloc[fit_rows].cycle_code.max() >= frame.iloc[rows].cycle_code.min():
            raise RuntimeError(f"Invalid chronological row boundary at fit_end={end}")
        prediction, audit = _cache_one(frame, end, choice, rows, fit_rows, receipt)
        candidate_vectors[end] = (rows, prediction)
        audits[str(end)] = audit
    result = _analysis(frame, candidate_vectors, control_vectors, audits)
    result["status"] = "COMPLETED_SVD128_PAIRED_OUTER_ABLATION"
    result["freeze_signature"] = receipt["signature"]
    result["postfit_reporting_amendment_sha256"] = (
        _sha_file(AMENDMENT_PATH) if AMENDMENT_PATH.is_file() else None
    )
    result["stage5_reference_signature"] = receipt["identity"]["stage5_signature"]
    result["selection"] = selected
    result["fit_audits"] = audits
    result["control_audits"] = control_audits
    result["elapsed_seconds"] = float(perf_counter() - total_start)
    result["fit_budget"] = {"outer_svd128_fits": len(audits), "inner_svd128_fits": 0, "full_train_fits": 0, "test_inference": False}
    result["adaptive_stage_gate"] = {
        "status": "REVIEW_PARENT_BEFORE_ANY_INNER_REFINEMENT",
        "predeclared_criteria": _load_json(PROTOCOL_PATH)["adaptive_refinement_gate"],
        "decision": "PENDING_PARENT_REVIEW",
    }
    result["figures"] = _plot_results(result, REPORT)
    REPORT.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT / "results.json", result)
    _render_markdown(result, REPORT / "discovery.md")
    print("primary MAE SVD32 -> SVD128:", result["aggregate"]["primary"]["control"]["raw_mae"], "->", result["aggregate"]["primary"]["candidate"]["raw_mae"], flush=True)
    print("H12 MAE SVD32 -> SVD128:", result["aggregate"]["h12"]["control"]["raw_mae"], "->", result["aggregate"]["h12"]["candidate"]["raw_mae"], flush=True)
    print("Reports:", REPORT, flush=True)


if __name__ == "__main__":
    main()
