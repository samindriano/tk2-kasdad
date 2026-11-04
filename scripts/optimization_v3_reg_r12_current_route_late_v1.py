"""Frozen two-origin late-history pilot for the new R12 current-R06 route.

The stages are intentionally separate. Preflight and synthetic-check never
open Train targets. Each --run-origin performs one ranker fit only. Evaluation
is refused until both saved bundles pass the independent prelabel verifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_batch1_r12 import RANKER_BASE_PARAMETERS
from src.models.optimization_v3_reg_r12_current_route_late_v1 import (
    NAMESPACE,
    ORIGINS,
    PREPROCESSING_MAX_FITS,
    RANKER_MAX_FITS,
    SCORE_CYCLES,
    build_routes,
    fit_predict_current_route,
)
from src.models.regression_research_text import HYBRID_FEATURES, _TFIDF_PARAMS, _text_documents


MAIN_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
TRAIN_PATH = MAIN_ROOT / "data/raw/regression/Train.csv"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
R12_RECIPE_PATH = MAIN_ROOT / "reports/optimization_v3/batch1/R12/recipe.json"
R12_FREEZE_PATH = MAIN_ROOT / "experiments/optimization_v3/batch1/R12/freeze.json"
R12_MODEL_PATH = ROOT / "src/models/optimization_v3_batch1_r12.py"
R12_RUNNER_PATH = ROOT / "scripts/optimization_v3_batch1_r12.py"
R06_CAMPAIGN_FREEZE = MAIN_ROOT / "experiments/optimization_v3/expanded_benchmark_v1_recovery_v2/freeze.json"
R06_RECIPE_PATH = MAIN_ROOT / "experiments/optimization_v3/batch1/R06/recipe_revision_02.json"
R06_FREEZE_PATH = MAIN_ROOT / "experiments/optimization_v3/batch1/R06/freeze_revision_02.json"
R06_ACTIVE_RUN = MAIN_ROOT / "experiments/runs/optimization_v3_batch1/R06"
R06_RECOVERY_RUN = MAIN_ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06"
EXPERIMENT_DIR = ROOT / "experiments/REG_R12_CURRENT_ROUTE_LATE_V1"
PROTOCOL_PATH = EXPERIMENT_DIR / "PROTOCOL.json"
PREFLIGHT_PATH = EXPERIMENT_DIR / "PREFLIGHT.json"
RUN_DIR = ROOT / "experiments/runs/REG_R12_CURRENT_ROUTE_LATE_V1"
REPORT_DIR = ROOT / "reports/optimization_v3/REG_R12_CURRENT_ROUTE_LATE_V1"
ATTEMPT_PATH = RUN_DIR / "attempts.jsonl"
PRELABEL_PATH = EXPERIMENT_DIR / "PRELABEL_PASS.json"
RESULTS_PATH = REPORT_DIR / "RESULTS.json"
SUMMARY_PATH = REPORT_DIR / "SUMMARY.md"
VERIFIER_PATH = ROOT / "scripts/verify_optimization_v3_reg_r12_current_route_late_v1_prelabel.py"
TEST_PATH = ROOT / "tests/test_optimization_v3_reg_r12_current_route_late_v1.py"

R06_BUNDLE_HASHES = {
    71: {
        "json": "f506ed8f0e97517d6af4d591d89e5cf4cea96640a47bb4d73235573f4c0b0aa8",
        "npz": "47c97efd70dba87b8fd4187e31e85122d325528688553f7c9e8e6a81dcf58894",
    },
    75: {
        "json": "61f3ab25d1cbc954cb74f47bdf823cc72113b1a2e19394c61802da1bdba7ca45",
        "npz": "61ef463a5bf5ee06f3c68162b8a7f874664b2736161979342731de98b5e09abb",
    },
}
R06_CAMPAIGN_SIGNATURE = "6cdecf6e0729b625981f23c7b7be4152a12806011a0a1852dd978eaec30d21ae"
R06_CAMPAIGN_FREEZE_SHA256 = "fcc6f16fc524729fb57d292647fd007e71ddc729ad9ff95a694d156a3b34ac7a"
R06_RECIPE_SHA256 = "6814978b3228fb741429dbd6b8af6d2c6caa8044eff0e079bbeb50066643b788"
R06_FREEZE_SHA256 = "a93302bb1132bc07a31d71597ce80f2c5e7544e49346ca8f155bfc6830ba6f11"
R06_RECIPE_FINGERPRINT = {
    "id": "V3_BATCH1_R06_CHAR_SPARSE_L1",
    "recipe": "char_wb TF-IDF ngrams 3-5 + structured/OHE + missing indicator; LightGBM regression_l1",
    "parameters": {
        "boosting_type": "gbdt", "colsample_bytree": 1.0, "deterministic": True,
        "force_col_wise": True, "learning_rate": 0.05, "max_depth": -1,
        "min_child_samples": 40, "min_child_weight": 0.001, "n_estimators": 400,
        "n_jobs": 2, "num_leaves": 31, "objective": "regression_l1",
        "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "subsample": 1.0, "subsample_freq": 0, "verbosity": -1,
    },
    "input_feature_names": [
        "impact_load", "entry_friction", "cycle_code", "scope_pressure", "text_load",
        "scope_trace", "org_spread", "module_spread", "telemetry_flag", "org_code",
        "module_code", "pattern_code", "signal_stream",
    ],
    "char_tfidf_parameters": {
        "analyzer": "char_wb", "dtype": "float64", "lowercase": True,
        "max_features": 30000, "min_df": 3, "ngram_range": [3, 5], "norm": "l2",
        "smooth_idf": True, "strip_accents": None, "sublinear_tf": True,
        "token_pattern": None, "use_idf": True,
    },
    "missing_stream_indicator": True,
    "svd_used": False,
    "word_tfidf_used": False,
    "native_categorical_features_used": False,
}

MIN_FREE_GIB = 4.0
RESERVE_GIB = 2.0
MAX_JOB_GIB = 6.0
THREADS = 2
SUPPORTED_GROUP_N_MINIMUM = 100
GROUP_HARM_MAX = 0.25


class GateError(RuntimeError):
    """Fail-closed pilot precondition or artifact integrity violation."""


def _group_supported(n: int) -> bool:
    return int(n) >= SUPPORTED_GROUP_N_MINIMUM


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"Cannot read required JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"Expected an object in {path}")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, _canonical_bytes(value) + b"\n")


def _append_event(event: Mapping[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(dict(event)) + b"\n"
    fd = os.open(ATTEMPT_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _git_dirty_for(paths: list[Path]) -> list[str]:
    rels = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    result = subprocess.run(["git", "status", "--porcelain", "--", *rels], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _source_paths() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "src/models/optimization_v3_reg_r12_current_route_late_v1.py",
        VERIFIER_PATH,
        TEST_PATH,
        R12_RECIPE_PATH,
        R12_FREEZE_PATH,
        R12_RUNNER_PATH,
        R12_MODEL_PATH,
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/optimization_v2_regression.py",
        ROOT / "scripts/optimization_v3_batch1_r06.py",
        ROOT / "src/models/optimization_v3_batch1_r06.py",
        R06_RECIPE_PATH,
        R06_FREEZE_PATH,
    ]


def _row_hash(values: Any, *, dtype: str) -> str:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise GateError("row hash input must be one-dimensional")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _id_hash(ids: Any) -> str:
    values = np.asarray(ids, dtype=str)
    return _sha_bytes("\n".join(values.tolist()).encode("utf-8"))


def _load_feature_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    if not TRAIN_PATH.is_file():
        raise GateError(f"Official Train source is missing: {TRAIN_PATH}")
    source_hash = _sha_file(TRAIN_PATH)
    if source_hash != TRAIN_SHA256:
        raise GateError("Official regression Train SHA-256 differs from the pinned source")
    with TRAIN_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header or len(header) != len(set(header)):
        raise GateError("Train header is empty or has duplicate fields")
    required = {"record_id", *HYBRID_FEATURES}
    absent = sorted(required - set(header))
    if absent or "threat_rank" not in header:
        raise GateError(f"Train schema differs from the frozen feature/target contract; absent={absent}")
    if tuple(HYBRID_FEATURES) != tuple(R06_RECIPE_FINGERPRINT["input_feature_names"]):
        raise GateError("R12 feature allowlist differs from the pinned current R06 feature order")
    # `threat_rank` is deliberately excluded from usecols. Targets are read by
    # _read_targets_for_positions only for an authorized fit prefix or after
    # both outputs pass independent prelabel verification.
    frame = pd.read_csv(TRAIN_PATH, usecols=["record_id", *HYBRID_FEATURES], dtype={"record_id": "string"})
    if frame.columns.has_duplicates or len(frame) != 50_608:
        raise GateError("Official Train row count or selected-column schema differs from the freeze")
    ids = frame["record_id"].astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids) or np.any(ids == ""):
        raise GateError("Official Train record IDs are blank or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise GateError("Official Train cycle support differs from cycles 47–79")
    features = frame.loc[:, list(HYBRID_FEATURES)].copy()
    features.index = np.arange(len(features), dtype=np.int64)
    return features, ids, cycles, source_hash


def _read_targets_for_positions(
    positions: np.ndarray,
    *,
    expected_ids: np.ndarray,
    expected_cycles: np.ndarray,
    maximum_cycle: int,
    minimum_cycle: int | None = None,
    train_path: Path = TRAIN_PATH,
) -> np.ndarray:
    """Index/convert threat_rank only at selected physical rows.

    The CSV parser reads each row, but the target field is never indexed or
    converted unless the physical row is in the explicit whitelist.
    """
    rows = np.asarray(positions, dtype=np.int64)
    wanted = set(int(x) for x in rows)
    if len(wanted) != len(rows) or not wanted:
        raise GateError("target whitelist must be nonempty and contain unique rows")
    if expected_ids.shape != rows.shape or expected_cycles.shape != rows.shape:
        raise GateError("target whitelist IDs/cycles are not aligned")
    expected_idx_by_row = {int(row): index for index, row in enumerate(rows)}
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_pos, cycle_pos, target_pos = header.index("record_id"), header.index("cycle_code"), header.index("threat_rank")
        found: dict[int, float] = {}
        for row_position, values in enumerate(reader):
            if row_position not in wanted:
                continue
            row_id = values[id_pos]
            cycle = int(float(values[cycle_pos]))
            expected_idx = expected_idx_by_row[row_position]
            if row_id != expected_ids[expected_idx]:
                raise GateError("target whitelist record ID does not match its frozen Train row")
            if cycle != int(expected_cycles[expected_idx]):
                raise GateError("target whitelist cycle does not match its frozen Train row")
            if cycle > maximum_cycle or (minimum_cycle is not None and cycle < minimum_cycle):
                raise GateError("attempted to read a target outside its authorized cycle boundary")
            # The only target access/conversion occurs after whitelist and
            # cycle checks. No unselected target cell is inspected by code.
            found[row_position] = float(values[target_pos])
    if set(found) != wanted:
        raise GateError("one or more whitelisted target rows were not found")
    result = np.asarray([found[int(row)] for row in rows], dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 100)):
        raise GateError("whitelisted targets are outside [0,100] or nonfinite")
    return result


def _recipe_fingerprint(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": state.get("id"),
        "recipe": state.get("recipe"),
        "parameters": state.get("parameters"),
        "input_feature_names": state.get("input_feature_names"),
        "char_tfidf_parameters": state.get("char_tfidf", {}).get("parameters"),
        "missing_stream_indicator": state.get("missing_stream_indicator"),
        "svd_used": state.get("svd_used"),
        "word_tfidf_used": state.get("word_tfidf_used"),
        "native_categorical_features_used": state.get("native_categorical_features_used"),
    }


def _read_r06_bundle(cutoff: int) -> tuple[dict[str, Any], dict[str, np.ndarray], Path, Path]:
    if cutoff not in R06_BUNDLE_HASHES:
        raise GateError(f"No pinned current R06 bundle for origin {cutoff}")
    root = R06_RECOVERY_RUN
    json_path = root / f"R06_fit{cutoff}.json"
    npz_path = root / f"R06_fit{cutoff}.npz"
    expected = R06_BUNDLE_HASHES[cutoff]
    if _sha_file(json_path) != expected["json"] or _sha_file(npz_path) != expected["npz"]:
        raise GateError(f"R06 origin {cutoff} bundle hash changed")
    sidecar = _json(json_path)
    if sidecar.get("npz_sha256") != expected["npz"]:
        raise GateError(f"R06 origin {cutoff} sidecar NPZ hash mismatch")
    identity, metadata = sidecar.get("identity", {}), sidecar.get("metadata", {})
    if (identity.get("card_id") != "R06" or int(identity.get("cutoff", -1)) != cutoff
            or identity.get("campaign_freeze_signature") != R06_CAMPAIGN_SIGNATURE
            or metadata.get("freeze_signature") != R06_CAMPAIGN_SIGNATURE
            or metadata.get("test_data_used") is not False
            or metadata.get("validation_labels_passed_to_fit") is not False
            or int(metadata.get("fit_predictor_count", -1)) != 1):
        raise GateError(f"R06 origin {cutoff} sidecar provenance or fit boundary is invalid")
    if _recipe_fingerprint(metadata.get("state", {})) != R06_RECIPE_FINGERPRINT:
        raise GateError(f"R06 origin {cutoff} model recipe differs from current R06")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    for key in ("score_rows", "score_record_ids", "raw_candidate", "prediction", "baseline_prediction",
                "audit__candidate_rank50", "score_cycles"):
        if key not in arrays:
            raise GateError(f"R06 origin {cutoff} bundle lacks {key}")
    if arrays["score_record_ids"].dtype.kind != "U":
        raise GateError(f"R06 origin {cutoff} IDs are not safe Unicode arrays")
    return sidecar, arrays, json_path, npz_path


def _validate_r06_campaign() -> dict[str, Any]:
    if _sha_file(R06_CAMPAIGN_FREEZE) != R06_CAMPAIGN_FREEZE_SHA256:
        raise GateError("R06 recovery campaign freeze hash changed")
    campaign = _json(R06_CAMPAIGN_FREEZE)
    identity = campaign.get("identity", {})
    if campaign.get("signature") != R06_CAMPAIGN_SIGNATURE:
        raise GateError("R06 recovery campaign signature differs from its bundles")
    if identity.get("input_sha256", {}).get("regression_controls:official_regression_train") != TRAIN_SHA256:
        raise GateError("R06 recovery campaign is not pinned to the official Train source")
    if identity.get("input_sha256", {}).get("R06:freeze") != R06_FREEZE_SHA256:
        raise GateError("R06 recovery campaign does not bind the current R06 freeze")
    if identity.get("input_sha256", {}).get("R06:recipe") != R06_RECIPE_SHA256:
        raise GateError("R06 recovery campaign does not bind the current R06 recipe")
    for key, path in {
        "R06:pilot55:npz": R06_ACTIVE_RUN / "R06_fit55.npz",
        "R06:pilot55:sidecar": R06_ACTIVE_RUN / "R06_fit55.json",
        "R06:pilot67:npz": R06_ACTIVE_RUN / "R06_fit67.npz",
        "R06:pilot67:sidecar": R06_ACTIVE_RUN / "R06_fit67.json",
    }.items():
        if identity.get("input_sha256", {}).get(key) != _sha_file(path):
            raise GateError(f"R06 recovery campaign input binding differs for {key}")
    return {
        "campaign_id": identity.get("campaign_id"),
        "campaign_freeze_sha256": R06_CAMPAIGN_FREEZE_SHA256,
        "campaign_signature": R06_CAMPAIGN_SIGNATURE,
        "parent_campaign_id": identity.get("parent_campaign_id"),
        "source_head_before_freeze": identity.get("source_head_before_freeze"),
        "official_regression_train_sha256": identity["input_sha256"]["regression_controls:official_regression_train"],
        "current_r06_freeze_sha256": R06_FREEZE_SHA256,
        "current_r06_recipe_sha256": R06_RECIPE_SHA256,
    }


def _validate_r06_outputs(features: pd.DataFrame, ids: np.ndarray, cycles: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    campaign_receipt = _validate_r06_campaign()
    outputs: dict[int, dict[str, np.ndarray]] = {}
    fingerprints: list[dict[str, Any]] = []
    for cutoff in (55, 67, 71, 75):
        root = R06_ACTIVE_RUN if cutoff in (55, 67) else R06_RECOVERY_RUN
        sidecar_path = root / f"R06_fit{cutoff}.json"
        npz_path = root / f"R06_fit{cutoff}.npz"
        sidecar = _json(sidecar_path)
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        fingerprint = _recipe_fingerprint(state)
        if fingerprint != R06_RECIPE_FINGERPRINT:
            raise GateError(f"R06 fit{cutoff} recipe differs from the frozen current R06 recipe")
        fingerprints.append(fingerprint)
    if any(item != fingerprints[0] for item in fingerprints[1:]):
        raise GateError("R06 recipe changed between saved origins")

    for cutoff in (71, 75):
        sidecar, arrays, json_path, npz_path = _read_r06_bundle(cutoff)
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        bundle_ids = arrays["score_record_ids"].astype(str)
        if rows.shape != bundle_ids.shape or np.any((rows < 0) | (rows >= len(ids))):
            raise GateError(f"R06 origin {cutoff} row positions are invalid")
        if not np.array_equal(bundle_ids, ids[rows]):
            raise GateError(f"R06 origin {cutoff} IDs do not match official Train positions")
        score_cycles = cycles[rows]
        expected = np.asarray((72, 73, 74, 75) if cutoff == 71 else (76, 77, 78, 79), dtype=np.int64)
        if (not np.array_equal(np.unique(score_cycles), expected)
                or not np.array_equal(np.asarray(arrays["score_cycles"], dtype=np.int64), score_cycles)):
            raise GateError(f"R06 origin {cutoff} does not contain the exact primary query cycles")
        cycle_counts = {int(cycle): int(np.sum(score_cycles == cycle)) for cycle in expected}
        raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
        base = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        pred = np.asarray(arrays["prediction"], dtype=np.float64)
        if not (raw.shape == base.shape == pred.shape == rows.shape):
            raise GateError(f"R06 origin {cutoff} score arrays are not row-aligned")
        reconstructed_rank50 = np.empty(len(raw), dtype=np.float64)
        for cycle in expected:
            mask = score_cycles == cycle
            pct = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
            reconstructed_rank50[mask] = np.clip(0.5 * raw[mask] + 0.5 * pct, 0.0, 100.0)
        rank50_delta = float(np.max(np.abs(reconstructed_rank50 - arrays["audit__candidate_rank50"])))
        prediction_delta = float(np.max(np.abs(np.clip(0.75 * base + 0.25 * reconstructed_rank50, 0, 100) - pred)))
        if rank50_delta != 0.0 or prediction_delta != 0.0:
            raise GateError(f"R06 origin {cutoff} saved prediction reconstruction failed")
        fit_rows = int(sidecar.get("metadata", {}).get("fit_rows", -1))
        if fit_rows != int(np.sum(cycles <= cutoff)):
            raise GateError(f"R06 origin {cutoff} fit-row count does not match its temporal prefix")
        outputs[cutoff] = {
            "rows": rows,
            "ids": bundle_ids,
            "cycles": score_cycles,
            "prediction": pred,
            "legacy_baseline": base,
            "cycle_counts": np.asarray([cycle_counts[int(c)] for c in score_cycles], dtype=np.int64),
        }
        outputs[cutoff]["bundle_json_sha256"] = np.asarray([_sha_file(json_path)])
        outputs[cutoff]["bundle_npz_sha256"] = np.asarray([_sha_file(npz_path)])
        outputs[cutoff]["bundle_identity_sha256"] = np.asarray([sidecar.get("identity_sha256", "")])
        outputs[cutoff]["rank50_max_abs_delta"] = np.asarray([rank50_delta])
        outputs[cutoff]["prediction_max_abs_delta"] = np.asarray([prediction_delta])
    del features
    outputs["campaign_receipt"] = campaign_receipt  # type: ignore[assignment]
    return outputs


def _group_feature_preflight(features: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for origin, query_cycles in ((71, (72, 73, 74, 75)), (75, (76, 77, 78, 79))):
        fit_mask = cycles <= origin
        score_mask = np.isin(cycles, query_cycles)
        fit_text = pd.to_numeric(features.loc[fit_mask, "text_load"], errors="coerce").to_numpy(dtype=np.float64)
        fit_finite = fit_text[np.isfinite(fit_text)]
        if len(fit_finite) == 0:
            raise GateError(f"origin {origin} has no finite fit text_load for frozen quartile groups")
        edges = np.quantile(fit_finite, [0.25, 0.50, 0.75], method="linear")
        score_text = pd.to_numeric(features.loc[score_mask, "text_load"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(score_text)
        group_index = np.full(len(score_text), -1, dtype=np.int8)
        group_index[finite] = np.searchsorted(edges, score_text[finite], side="left").astype(np.int8)
        score_stream = features.loc[score_mask, "signal_stream"]
        missing = score_stream.isna().to_numpy(dtype=bool) | score_stream.fillna("").astype(str).str.strip().eq("").to_numpy(dtype=bool)
        entry = pd.to_numeric(features.loc[score_mask, "entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        counts = {f"text_load_Q{i + 1}": int(np.sum(group_index == i)) for i in range(4)}
        counts.update({
            "availability_present": int(np.sum(~missing)),
            "availability_missing": int(np.sum(missing)),
            "entry_friction_0": int(np.sum(entry == 0.0)),
            "entry_friction_1": int(np.sum(entry == 1.0)),
        })
        output[str(origin)] = {
            "fit_rows": int(fit_mask.sum()),
            "query_rows": int(score_mask.sum()),
            "text_load_quantile_edges_fit_only": [float(x) for x in edges],
            "text_load_quantile_method": "linear quantiles 0.25/0.50/0.75 on finite fit-prefix text_load; Q bins are left-inclusive at tied cutpoints via searchsorted(side='left')",
            "feature_group_support": counts,
            "groups_supported_before_fit": {
                name: _group_supported(n)
                for name, n in counts.items()
            },
            "target_tail_support": "UNKNOWN until postfit evaluation labels are opened",
        }
        unsupported = [name for name, n in counts.items() if not _group_supported(n)]
        if unsupported:
            raise GateError(f"origin {origin} has unsupported feature gate groups before fit: {unsupported}")
    return output


def _resource_preflight(features: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    fit_rows = np.flatnonzero(cycles <= 75)
    documents, _ = _text_documents(features.iloc[fit_rows]["signal_stream"])
    analyzer = TfidfVectorizer(**_TFIDF_PARAMS).build_analyzer()
    nnz_upper = int(sum(len(analyzer(document)) for document in documents))
    sparse_upper_bytes = int(nnz_upper * 16 + (len(fit_rows) + 1) * 8)
    sparse_upper_gib = float(sparse_upper_bytes / (1024 ** 3))
    estimated_peak_gib = float(2.25 + 3.0 * sparse_upper_gib)
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    allowed_gib = float(min(MAX_JOB_GIB, available_gib - RESERVE_GIB))
    if available_gib < MIN_FREE_GIB or estimated_peak_gib > allowed_gib:
        raise GateError(
            f"resource preflight blocked: free={available_gib:.3f}GiB, "
            f"estimated_peak={estimated_peak_gib:.3f}GiB, allowed={allowed_gib:.3f}GiB"
        )
    return {
        "fit_rows_largest_origin_75": int(len(fit_rows)),
        "word_ngram_nnz_upper_bound": nnz_upper,
        "sparse_design_upper_gib": sparse_upper_gib,
        "estimated_peak_job_gib": estimated_peak_gib,
        "estimate_method": "2.25GiB fixed allowance + 3x fit-only sparse word ngram upper bound; score rows are below the largest fit population",
        "available_ram_gib_observed": available_gib,
        "projected_reserve_after_estimated_peak_gib": float(available_gib - estimated_peak_gib),
        "allowed_peak_gib": allowed_gib,
        "minimum_free_ram_gib": MIN_FREE_GIB,
        "reserve_gib": RESERVE_GIB,
        "max_job_gib": MAX_JOB_GIB,
        "concurrent_fit_jobs": 1,
        "threads_per_job": THREADS,
        "predictor_fits_performed": 0,
    }


def _validate_origin_support(cycles: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    result = {}
    for origin, score_cycles in SCORE_CYCLES.items():
        fit_rows = np.flatnonzero(cycles <= origin)
        score_rows = np.flatnonzero(np.isin(cycles, score_cycles))
        fit_cycle_set = np.unique(cycles[fit_rows])
        if not np.array_equal(fit_cycle_set, np.arange(47, origin + 1, dtype=np.int64)):
            raise GateError(f"origin {origin} fit prefix has missing or unexpected cycles")
        expected_n = int(sum(np.sum(cycles == cycle) for cycle in score_cycles))
        if len(score_rows) != expected_n:
            raise GateError(f"origin {origin} query cycle rows are incomplete")
        result[str(origin)] = {
            "fit_rows": int(len(fit_rows)),
            "fit_cycle_min": int(fit_cycle_set.min()),
            "fit_cycle_max": int(fit_cycle_set.max()),
            "fit_cycle_count": int(len(fit_cycle_set)),
            "score_cycles": list(score_cycles),
            "score_rows": int(len(score_rows)),
            "score_cycle_counts": {str(int(c)): int(np.sum(cycles == c)) for c in score_cycles},
            "fit_row_positions_sha256": _row_hash(fit_rows, dtype="<i8"),
            "fit_record_ids_sha256": _id_hash(ids[fit_rows]),
            "score_row_positions_sha256": _row_hash(score_rows, dtype="<i8"),
            "score_record_ids_sha256": _id_hash(ids[score_rows]),
        }
    return result


def _preflight() -> dict[str, Any]:
    features, ids, cycles, source_hash = _load_feature_data()
    r06 = _validate_r06_outputs(features, ids, cycles)
    support = _validate_origin_support(cycles, ids)
    group_support = _group_feature_preflight(features, cycles)
    resources = _resource_preflight(features, cycles)
    w5_rows = np.flatnonzero(np.isin(cycles, (72, 73, 74, 75)))
    w6_rows = np.flatnonzero(np.isin(cycles, (76, 77, 78, 79)))
    if np.intersect1d(w5_rows, w6_rows).size or len(np.union1d(w5_rows, w6_rows)) != len(w5_rows) + len(w6_rows):
        raise GateError("W5 and W6 primary windows overlap or contain duplicate rows")
    r06_rowsets = {cutoff: set(map(int, r06[cutoff]["rows"])) for cutoff in (71, 75)}
    if set(map(int, w5_rows)) != r06_rowsets[71] or set(map(int, w6_rows)) != r06_rowsets[75]:
        raise GateError("same-origin current R06 outputs do not exactly equal the frozen W5/W6 row sets")
    campaign = r06["campaign_receipt"]
    return {
        "namespace": NAMESPACE,
        "status": "PREFLIGHT_PASS_NO_TARGET_LABELS_NO_FITS",
        "created_at_local": pd.Timestamp.now().isoformat(),
        "source": {
            "train_path": str(TRAIN_PATH),
            "train_sha256": source_hash,
            "train_rows": int(len(features)),
            "unique_record_ids": int(len(np.unique(ids))),
            "cycles": [int(cycles.min()), int(cycles.max())],
            "target_values_read": False,
            "test_sample_submission_or_kaggle_opened": False,
        },
        "r06_campaign": campaign,
        "r06_bundles": {
            str(cutoff): {
                "json_sha256": str(r06[cutoff]["bundle_json_sha256"][0]),
                "npz_sha256": str(r06[cutoff]["bundle_npz_sha256"][0]),
                "identity_sha256": str(r06[cutoff]["bundle_identity_sha256"][0]),
                "rows": int(len(r06[cutoff]["rows"])),
                "ids_match_train": True,
                "rank50_reconstruction_max_abs_delta": float(r06[cutoff]["rank50_max_abs_delta"][0]),
                "prediction_reconstruction_max_abs_delta": float(r06[cutoff]["prediction_max_abs_delta"][0]),
            }
            for cutoff in (71, 75)
        },
        "origin_support": support,
        "feature_group_preflight": group_support,
        "primary_windows": {
            "W5": {"cycles": [72, 73, 74, 75], "origin": 71, "rows": int(len(w5_rows)),
                   "row_positions_sha256": _row_hash(w5_rows, dtype="<i8"),
                   "record_ids_sha256": _id_hash(ids[w5_rows]), "current_r06_origin": 71},
            "W6": {"cycles": [76, 77, 78, 79], "origin": 75, "rows": int(len(w6_rows)),
                   "row_positions_sha256": _row_hash(w6_rows, dtype="<i8"),
                   "record_ids_sha256": _id_hash(ids[w6_rows]), "current_r06_origin": 75},
            "pooled_unique_rows": int(len(w5_rows) + len(w6_rows)),
            "E71_H8_diagnostic_cycles": list(range(72, 80)),
            "E71_H8_diagnostic_only_overlaps_W5_W6": True,
        },
        "resources": resources,
        "fit_accounting": {
            "predictor_fit_cap": 2,
            "planned_predictor_fits": ["origin71", "origin75"],
            "preprocessing_fit_cap": 8,
            "planned_learned_preprocessing_per_origin": ["word_tfidf", "structured_processor", "numeric_scaler", "categorical_ohe"],
            "concurrent_fit_jobs": 1,
            "threads_per_job": THREADS,
            "repeats_retries_seed_grid_or_extra_control_fits": False,
        },
        "namespace_empty_before_fit": not RUN_DIR.exists(),
        "main_checkout_write": False,
    }


def _protocol_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = _source_paths()
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise GateError(f"Cannot pin missing source files: {missing}")
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path): _sha_file(path)
                     for path in source_paths}
    r12_recipe = _json(R12_RECIPE_PATH)
    r12_freeze = _json(R12_FREEZE_PATH)
    if _sha_file(R12_RECIPE_PATH) != "e76861b798a212f7b1b2bf70764db49c67a957adfa345d7173af203df939069d":
        raise GateError("Original R12 recipe changed")
    if r12_freeze.get("signature") != "dbc69182a35d02032763d463c630c49ff52b2d803ac487202aefd465b25d78d3":
        raise GateError("Original R12 freeze signature changed")
    estimator = r12_recipe["estimator"]
    if r12_recipe.get("features") != list(HYBRID_FEATURES):
        raise GateError("R12 feature allowlist differs from the implementation's exact ordered allowlist")
    if estimator.get("parameters_base") != RANKER_BASE_PARAMETERS:
        raise GateError("R12 ranker base parameters differ from the frozen implementation")
    if estimator.get("parameters_per_fit", {}).get("label_gain") != list(range(10)):
        raise GateError("R12 label gains differ from the frozen 0..9 grade contract")
    if estimator.get("parameters_per_fit", {}).get("lambdarank_truncation_level") != "maximum group size among fit cycles":
        raise GateError("R12 truncation rule differs from the maximum fit-cycle group size")
    if (estimator["class"] != "lightgbm.LGBMRanker"
            or estimator["target_grade"] != "clip(floor(threat_rank / 10), 0, 9)"
            or estimator["group_order"] != "stable cycle sort; labels and CSR rows sorted identically"
            or estimator["score_inverse_mapping"] != "stable-sort score rows by cycle then inverse-map to original row positions"):
        raise GateError("R12 ranker grade, query sort, or inverse-map rule differs from its frozen recipe")
    text_recipe = r12_recipe.get("preprocessing", {}).get("text", {})
    expected_text_recipe = {
        "dtype": "float64",
        "max_features": int(_TFIDF_PARAMS["max_features"]),
        "min_df": int(_TFIDF_PARAMS["min_df"]),
        "ngram_range": list(_TFIDF_PARAMS["ngram_range"]),
        "sublinear_tf": bool(_TFIDF_PARAMS["sublinear_tf"]),
    }
    if text_recipe != expected_text_recipe:
        raise GateError("R12 word TF-IDF parameters differ from the pinned recipe")
    if (r12_recipe.get("preprocessing", {}).get("structured") !=
            "RegressionTextModel._structured_fit/_structured_transform; fit-prefix numeric imputation/scaling and rare-category OHE"
            or r12_recipe["preprocessing"].get("missing_stream_indicator") is not True
            or r12_recipe["preprocessing"].get("sparse_design") is not True
            or r12_recipe["preprocessing"].get("svd") is not False
            or r12_recipe["preprocessing"].get("full_word_design_densified") is not False):
        raise GateError("R12 structured/text sparsity preprocessing differs from the frozen recipe")
    return {
        "namespace": NAMESPACE,
        "protocol_version": 1,
        "purpose": "Two-origin, late-history, outcome-reused diagnostic pilot for a new R12 current-R06 route; not a retroactive R12 pass.",
        "source_commit_before_protocol": _git_head(),
        "source_sha256": source_hashes,
        "input_sha256": {
            "official_regression_train": preflight["source"]["train_sha256"],
            "R12_original_recipe": _sha_file(R12_RECIPE_PATH),
            "R12_original_freeze": _sha_file(R12_FREEZE_PATH),
            "R06_current_recipe": _sha_file(R06_RECIPE_PATH),
            "R06_current_freeze": _sha_file(R06_FREEZE_PATH),
            "R06_recovery_campaign_freeze": R06_CAMPAIGN_FREEZE_SHA256,
            "R06_fit71_json": preflight["r06_bundles"]["71"]["json_sha256"],
            "R06_fit71_npz": preflight["r06_bundles"]["71"]["npz_sha256"],
            "R06_fit75_json": preflight["r06_bundles"]["75"]["json_sha256"],
            "R06_fit75_npz": preflight["r06_bundles"]["75"]["npz_sha256"],
        },
        "fit_origins": [71, 75],
        "fit_prefix": "all official Train rows with cycle_code <= origin; labels only for each origin fit prefix",
        "primary_nonoverlap_windows": {
            "W5": {"origin": 71, "cycles": [72, 73, 74, 75], "comparator": "saved current R06_fit71 exact row IDs"},
            "W6": {"origin": 75, "cycles": [76, 77, 78, 79], "comparator": "saved current R06_fit75 exact row IDs"},
            "pooled": "W5 union W6 by unique Train row; do not double-count W6 inside the E71 H8 diagnostic",
        },
        "diagnostic_only": {
            "E71_H8": "R12 origin71 scores cycles72-79. W5 uses exact R06 origin71; overlapping W6 is contextually compared to exact R06 origin75. This mixed-origin H8 view is diagnostic only and not part of any primary gate or pooled denominator.",
        },
        "ranker": {
            "class": estimator["class"],
            "parameters_base": estimator["parameters_base"],
            "label_gain": estimator["parameters_per_fit"]["label_gain"],
            "truncation": estimator["parameters_per_fit"]["lambdarank_truncation_level"],
            "grade": estimator["target_grade"],
            "query": estimator["query_definition"],
            "stable_group_order": estimator["group_order"],
            "score_inverse_map": estimator["score_inverse_mapping"],
        },
        "features": r12_recipe["features"],
        "preprocessing": {
            "structured": r12_recipe["preprocessing"]["structured"],
            "text": r12_recipe["preprocessing"]["text"],
            "fit_scope": "fit prefix only at each origin",
            "missing_stream_indicator": True,
            "sparse_design": True,
            "svd": False,
            "dense_full_word_matrix": False,
            "learned_preprocessing_units_per_origin": ["word_tfidf", "structured_processor", "numeric_scaler", "categorical_ohe"],
        },
        "primary_policy": {
            "rank_map": "100*average_rank(raw_R12_score_within_complete_cycle)/complete_cycle_n",
            "tie_method": "average",
            "present": "clip(0.75*saved_current_R06_prediction + 0.25*R12_cycle_percentile, 0, 100)",
            "missing": "bit-identical saved current R06 prediction",
            "ranking_before_mask": True,
            "new_policy_name": NAMESPACE,
        },
        "fixed_zero_fit_control": {
            "present": "clip(0.75*saved_current_R06_prediction + 0.25*average_rank_percentile(saved_current_R06_prediction within complete cycle), 0, 100)",
            "missing": "bit-identical saved current R06 prediction",
            "tie_method": "average",
        },
        "secondary_only": {
            "original_legacy_R12_route": "Use only the exact saved legacy baseline carried by the row-aligned R06 bundle; never select it. This is reported only if prelabel verification proves that baseline exact and aligned.",
        },
        "group_gates": {
            "feature_groups": ["availability_present", "availability_missing", "text_load_Q1", "text_load_Q2", "text_load_Q3", "text_load_Q4", "entry_friction_0", "entry_friction_1"],
            "text_load_edges": "Per-origin quartiles at fit-prefix quantiles 0.25/0.50/0.75, computed from finite fit-prefix feature values only; thresholds pinned in PREFLIGHT.json before fits.",
            "outcome_groups": ["target_le_5", "target_ge_95"],
            "support_rule": "n >= 100; unsupported groups are UNKNOWN and cannot produce an overall PASS.",
            "harm_rule": "candidate MAE minus exact same-origin R06 MAE <= 0.25 in each primary window for every supported group.",
            "no_group_interaction_grid": True,
        },
        "primary_gates": {
            "W5_gain_vs_same_origin_R06_min": 0.0,
            "W6_gain_vs_same_origin_R06_min": 0.0,
            "pooled_unique_gain_vs_same_origin_R06_min": 0.03,
            "pooled_unique_gain_vs_mapping_only_control_min": 0.01,
            "supported_group_harm_max_each_window": 0.25,
            "unsupported_group_status": "UNKNOWN; no automatic pass",
            "failure_action": "close rescue; no automatic E79, CSV, or upload",
        },
        "resource_policy": {
            "minimum_free_ram_gib_before_fit": MIN_FREE_GIB,
            "projected_reserve_gib": RESERVE_GIB,
            "max_job_rss_gib": MAX_JOB_GIB,
            "concurrent_fit_jobs": 1,
            "threads_per_job": THREADS,
            "resource_preflight": preflight["resources"],
        },
        "learned_unit_budget": {
            "predictor_fits_max": 2,
            "exact_origins": [71, 75],
            "preprocessing_fits_max": 8,
            "preprocessing_per_origin": ["word_tfidf", "structured_processor", "numeric_scaler", "categorical_ohe"],
            "no_retries": True,
            "no_seed_grid": True,
            "no_policy_grid": True,
        },
        "prelabel_boundary": {
            "score_query_labels_opened_before_both_origin_bundles_saved": False,
            "origin75_fit_prefix_reuses_W5_labels_after_origin71_bundle": True,
            "independent_prelabel_verifier_required": True,
            "validation_labels_passed_to_fit": False,
            "evaluation_labels_opened_by_prelabel_verifier": False,
            "test_sample_submission_kaggle_accessed": False,
        },
        "adaptivity_disclosure": "Cycles 72-75 are reused as fit labels at origin75 after serving as E71 query rows; E71/E75 Train views were already explored. The new predictions are not a fresh holdout.",
        "fit_support": preflight["origin_support"],
        "feature_group_preflight": preflight["feature_group_preflight"],
        "resource_fit_preflight": preflight["resources"],
        "packages": {
            name: __import__(module).__version__ if hasattr(__import__(module), "__version__") else platform.python_version()
            for name, module in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                                 ("scikit-learn", "sklearn"), ("lightgbm", "lightgbm"),
                                 ("threadpoolctl", "threadpoolctl"), ("psutil", "psutil"))
        },
        "no_main_write": True,
        "no_shared_cls_ledger_access": True,
    }


def _verify_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    identity = protocol.get("identity")
    if not isinstance(identity, dict) or protocol.get("signature") != _sha_bytes(_canonical_bytes(identity)):
        raise GateError("Protocol signature does not match its identity")
    expected_sources = identity.get("source_sha256", {})
    for path_text, expected in expected_sources.items():
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or _sha_file(path) != expected:
            raise GateError(f"Frozen source hash changed: {path_text}")
    repo_paths = [Path(path_text) if Path(path_text).is_absolute() else ROOT / path_text
                  for path_text in expected_sources]
    repo_paths = [path for path in repo_paths if path.is_relative_to(ROOT)]
    repo_paths.extend([PROTOCOL_PATH, PREFLIGHT_PATH])
    dirty = _git_dirty_for(repo_paths)
    if dirty:
        raise GateError(f"Frozen source files are not cleanly committed: {dirty}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--",
         *[str(path.relative_to(ROOT)).replace("\\", "/") for path in repo_paths]],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        raise GateError("Protocol and source files are not all tracked in the committed worktree")
    if _git_head() == identity.get("source_commit_before_protocol"):
        raise GateError("Frozen source/protocol commit is not present before official fit")
    return {"protocol_signature": protocol["signature"], "source_file_count": len(expected_sources),
            "prepared_parent_commit": identity.get("source_commit_before_protocol"), "current_commit": _git_head()}


def freeze_protocol() -> dict[str, Any]:
    if RUN_DIR.exists() or REPORT_DIR.exists():
        raise GateError("New namespace already contains run/report outputs; refusing to overwrite")
    if PROTOCOL_PATH.exists() or PREFLIGHT_PATH.exists():
        raise GateError("Protocol or preflight already exists; refusing to rewrite")
    preflight = _preflight()
    if preflight["resources"]["available_ram_gib_observed"] < MIN_FREE_GIB:
        raise GateError("Pre-fit free RAM is below the frozen minimum")
    preflight["source_files"] = {
        str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path): _sha_file(path)
        for path in _source_paths()
    }
    preflight["source_commit_before_protocol"] = _git_head()
    identity = _protocol_identity(preflight)
    signature = _sha_bytes(_canonical_bytes(identity))
    protocol = {"identity": identity, "signature": signature, "signature_method": "SHA-256 of canonical UTF-8 identity JSON"}
    protocol_digest = _sha_bytes(_canonical_bytes(protocol))
    preflight_receipt = {
        **preflight,
        "protocol_signature": signature,
        "protocol_file_sha256": protocol_digest,
        "status": "FROZEN_PREFIT_NO_TARGET_LABELS_NO_FITS",
    }
    preflight_receipt["receipt_signature"] = _sha_bytes(_canonical_bytes(preflight_receipt))
    _write_json_exclusive(PROTOCOL_PATH, protocol)
    _write_json_exclusive(PREFLIGHT_PATH, preflight_receipt)
    return {"status": preflight_receipt["status"], "protocol_signature": signature,
            "protocol_file_sha256": protocol_digest, "preflight": preflight_receipt}


def _load_frozen_protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _json(PROTOCOL_PATH)
    preflight = _json(PREFLIGHT_PATH)
    verified = _verify_protocol(protocol)
    if preflight.get("protocol_signature") != protocol.get("signature"):
        raise GateError("Preflight receipt does not bind the frozen protocol")
    receipt_body = {key: value for key, value in preflight.items() if key != "receipt_signature"}
    if preflight.get("receipt_signature") != _sha_bytes(_canonical_bytes(receipt_body)):
        raise GateError("Preflight receipt signature does not verify")
    if preflight.get("protocol_file_sha256") != _sha_bytes(_canonical_bytes(protocol)):
        raise GateError("Preflight receipt protocol-file hash does not verify")
    if preflight.get("source_commit_before_protocol") != protocol["identity"].get("source_commit_before_protocol"):
        raise GateError("Preflight receipt source commit differs from protocol")
    if preflight.get("source_files") != protocol["identity"].get("source_sha256"):
        raise GateError("Preflight source hashes differ from protocol source pins")
    for name, expected in preflight.get("source_files", {}).items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or _sha_file(path) != expected:
            raise GateError(f"Preflight source hash changed: {name}")
    return protocol, preflight, verified


def _validate_saved_r06_for_scores(
    rows: np.ndarray,
    ids: np.ndarray,
    cycles: np.ndarray,
    r06: Mapping[int, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline_by_row: dict[int, tuple[str, float, float, int]] = {}
    for origin in (71, 75):
        record = r06[origin]
        for i, row in enumerate(record["rows"]):
            key = int(row)
            if key in baseline_by_row:
                raise GateError("Current R06 bundles overlap on row positions")
            baseline_by_row[key] = (
                str(record["ids"][i]),
                float(record["prediction"][i]),
                float(record["legacy_baseline"][i]),
                int(origin),
            )
    r06_pred = np.empty(len(rows), dtype=np.float64)
    legacy = np.empty(len(rows), dtype=np.float64)
    baseline_origin = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        item = baseline_by_row.get(int(row))
        if item is None or item[0] != str(ids[i]):
            raise GateError("Saved current R06 has no exact ID-aligned baseline for a score row")
        r06_pred[i], legacy[i], baseline_origin[i] = item[1], item[2], item[3]
    score_cycles = cycles[rows]
    return r06_pred, legacy, baseline_origin, score_cycles


def _unit_logger(origin: int):
    unit_counts = {"predictor": 0, "preprocessing": 0}
    started: dict[str, dict[str, Any]] = {}

    def unit_start(name: str, details: Mapping[str, Any]) -> str:
        unit_type = "predictor" if name == "lgbm_ranker_predictor" else "preprocessing"
        unit_counts[unit_type] += 1
        all_events = []
        if ATTEMPT_PATH.exists():
            all_events = [json.loads(line) for line in ATTEMPT_PATH.read_text(encoding="utf-8").splitlines() if line]
        global_predictors = sum(event.get("event") == "LEARNED_UNIT_STARTED" and event.get("unit_type") == "predictor" for event in all_events)
        global_preprocessing = sum(event.get("event") == "LEARNED_UNIT_STARTED" and event.get("unit_type") == "preprocessing" for event in all_events)
        if global_predictors + (unit_type == "predictor") > RANKER_MAX_FITS or global_preprocessing + (unit_type == "preprocessing") > PREPROCESSING_MAX_FITS:
            raise GateError("Learned-unit cap would be exceeded; stopping before the next learned fit")
        unit_id = f"origin{origin}:{unit_counts[unit_type]}:{name}"
        record = {
            "event": "LEARNED_UNIT_STARTED",
            "unit_id": unit_id,
            "unit_type": unit_type,
            "origin": origin,
            "unit_name": name,
            "timestamp_local": pd.Timestamp.now().isoformat(),
            "details": dict(details),
        }
        _append_event(record)
        started[unit_id] = record
        return unit_id

    def unit_finish(unit_id: str, details: Mapping[str, Any]) -> None:
        record = started.get(unit_id)
        if record is None:
            raise GateError("Learned-unit completion has no durable start record")
        _append_event({
            "event": "LEARNED_UNIT_COMPLETED" if details.get("status") != "FAILED" else "LEARNED_UNIT_FAILED",
            "unit_id": unit_id,
            "origin": origin,
            "timestamp_local": pd.Timestamp.now().isoformat(),
            "details": dict(details),
        })
    return unit_start, unit_finish, unit_counts


def _write_run_bundle(
    origin: int,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    model_artifact: bytes,
    protocol: Mapping[str, Any],
    commit_info: Mapping[str, Any],
) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.npz"
    model_path = npz_path.with_suffix(".txt")
    json_path = npz_path.with_suffix(".json")
    if any(path.exists() for path in (model_path, npz_path, json_path)):
        raise GateError(f"Origin {origin} output already exists; automatic retry is prohibited")
    if not isinstance(model_artifact, bytes) or not model_artifact:
        raise GateError("Fitted LightGBM model text must be persisted as nonempty bytes")
    model_sha = _sha_bytes(model_artifact)
    state = metadata.get("state", {})
    if state and state.get("fitted_model_sha256") != model_sha:
        raise GateError("Fitted model artifact hash differs from the model state receipt")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    npz_bytes = buffer.getvalue()
    npz_sha = _sha_bytes(npz_bytes)
    _write_exclusive(model_path, model_artifact)
    _write_exclusive(npz_path, npz_bytes)
    identity = {
        "namespace": NAMESPACE,
        "protocol_signature": protocol["signature"],
        "origin": origin,
        "score_cycles": list(SCORE_CYCLES[origin]),
        "train_sha256": TRAIN_SHA256,
        "r06_bundle_sha256": protocol["identity"]["input_sha256"],
        "row_positions_sha256": _row_hash(arrays["score_rows"], dtype="<i8"),
        "score_record_ids_sha256": _id_hash(arrays["score_record_ids"]),
        "model_artifact_file": model_path.name,
        "model_format": "lightgbm_native_text",
        "model_sha256": model_sha,
        "npz_sha256": npz_sha,
        "committed_source": dict(commit_info),
    }
    identity_sha = _sha_bytes(_canonical_bytes(identity))
    sidecar_body = {
        "identity": identity,
        "identity_sha256": identity_sha,
        "metadata": dict(metadata),
        "model_sha256": model_sha,
        "npz_sha256": npz_sha,
    }
    sidecar_body["integrity_sha256"] = _sha_bytes(_canonical_bytes(sidecar_body))
    _write_json_exclusive(json_path, sidecar_body)
    return {"model_path": str(model_path), "model_sha256": model_sha,
            "npz_path": str(npz_path), "json_path": str(json_path), "npz_sha256": npz_sha,
            "json_sha256": _sha_file(json_path), "rows": int(len(arrays["score_rows"]))}


def run_origin(origin: int) -> dict[str, Any]:
    origin = int(origin)
    if origin not in ORIGINS:
        raise GateError("Only origins 71 and 75 are authorized")
    protocol, preflight, commit_info = _load_frozen_protocol()
    prior_events: list[dict[str, Any]] = []
    if RUN_DIR.exists() and ATTEMPT_PATH.exists():
        events = [json.loads(line) for line in ATTEMPT_PATH.read_text(encoding="utf-8").splitlines() if line]
        prior_events = events
        if any(int(event.get("origin", -1)) == origin and event.get("event") in
               {"ORIGIN_PREFIT_PASS", "ORIGIN_ATTEMPT_STARTED", "ORIGIN_FAILED_NO_RETRY", "ORIGIN_BUNDLE_SAVED"}
               for event in events):
            raise GateError(f"Origin {origin} already has an attempt marker; retries are prohibited")
        if any(event.get("event") == "ORIGIN_FAILED_NO_RETRY" for event in events):
            raise GateError("A prior origin failed; the frozen pilot stops without continuing or retrying")
    if origin == 71 and prior_events:
        raise GateError("Origin 71 must be the first and only initial origin attempt")
    if origin == 75 and not any(event.get("event") == "ORIGIN_BUNDLE_SAVED" and int(event.get("origin", -1)) == 71 for event in prior_events):
        raise GateError("Origin 75 is authorized only after the origin 71 bundle is saved")
    expected_outputs = [RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.txt",
                        RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.npz",
                        RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.json"]
    if any(path.exists() for path in expected_outputs):
        raise GateError(f"Origin {origin} artifact already exists; automatic retry is prohibited")
    features, ids, cycles, source_hash = _load_feature_data()
    r06 = _validate_r06_outputs(features, ids, cycles)
    if _group_feature_preflight(features, cycles) != preflight["feature_group_preflight"]:
        raise GateError("Feature-gate group edges or support changed after protocol freeze")
    resource = _resource_preflight(features, cycles)
    fit_rows = np.flatnonzero(cycles <= origin)
    score_rows = np.flatnonzero(np.isin(cycles, SCORE_CYCLES[origin]))
    if len(fit_rows) != int(preflight["origin_support"][str(origin)]["fit_rows"]):
        raise GateError("Fit-prefix row count changed after protocol freeze")
    expected_query = np.asarray(SCORE_CYCLES[origin], dtype=np.int64)
    expected_counts = np.asarray([int(np.sum(cycles[score_rows] == c)) for c in cycles[score_rows]], dtype=np.int64)
    fit_x = features.iloc[fit_rows].copy()
    fit_x.index = fit_rows
    score_x = features.iloc[score_rows].copy()
    score_x.index = score_rows
    fit_y = _read_targets_for_positions(
        fit_rows,
        expected_ids=ids[fit_rows],
        expected_cycles=cycles[fit_rows],
        maximum_cycle=origin,
    )
    if int(np.sum(cycles[fit_rows] <= origin)) != len(fit_y):
        raise GateError("Fit label whitelist does not match the origin prefix")
    r06_pred, legacy, baseline_origin, score_cycles = _validate_saved_r06_for_scores(score_rows, ids[score_rows], cycles, r06)
    if not np.array_equal(np.unique(score_cycles), expected_query):
        raise GateError("Score rows differ from the frozen full-cycle query set")
    expected_support = preflight["origin_support"][str(origin)]
    if (_row_hash(fit_rows, dtype="<i8") != expected_support["fit_row_positions_sha256"]
            or _id_hash(ids[fit_rows]) != expected_support["fit_record_ids_sha256"]
            or _row_hash(score_rows, dtype="<i8") != expected_support["score_row_positions_sha256"]
            or _id_hash(ids[score_rows]) != expected_support["score_record_ids_sha256"]):
        raise GateError("Fit/query row IDs or positions differ from the signed preflight support")
    resource = _resource_preflight(features, cycles)
    if resource["available_ram_gib_observed"] < MIN_FREE_GIB or resource["projected_reserve_after_estimated_peak_gib"] < RESERVE_GIB:
        raise GateError("Immediate pre-fit resource check failed; no learned unit will start")
    start_unit, finish_unit, unit_counts = _unit_logger(origin)
    _append_event({
        "event": "ORIGIN_PREFIT_PASS",
        "origin": origin,
        "timestamp_local": pd.Timestamp.now().isoformat(),
        "protocol_signature": protocol["signature"],
        "available_ram_gib": resource["available_ram_gib_observed"],
        "estimated_peak_gib": resource["estimated_peak_job_gib"],
        "fit_rows": int(len(fit_rows)),
        "score_rows": int(len(score_rows)),
        "fit_label_rows": int(len(fit_y)),
        "score_query_label_values_read": False,
    })
    _append_event({"event": "ORIGIN_ATTEMPT_STARTED", "origin": origin,
                   "timestamp_local": pd.Timestamp.now().isoformat(),
                   "protocol_signature": protocol["signature"],
                   "resource_recheck_available_ram_gib": resource["available_ram_gib_observed"],
                   "resource_recheck_projected_reserve_gib": resource["projected_reserve_after_estimated_peak_gib"]})
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=THREADS):
            result = fit_predict_current_route(
                fit_x,
                fit_y,
                score_x,
                current_r06=r06_pred,
                legacy_baseline=legacy,
                expected_cycle_counts=expected_counts,
                cutoff=origin,
                unit_start=start_unit,
                unit_finish=finish_unit,
            )
    except Exception as exc:
        _append_event({
            "event": "ORIGIN_FAILED_NO_RETRY",
            "origin": origin,
            "timestamp_local": pd.Timestamp.now().isoformat(),
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
            "predictor_starts": unit_counts["predictor"],
            "preprocessing_starts": unit_counts["preprocessing"],
        })
        raise
    arrays = {
        "score_rows": score_rows.astype(np.int64),
        "score_record_ids": ids[score_rows].astype(str),
        "audit__score_cycle": score_cycles.astype(np.int64),
        "audit__expected_full_cycle_count": expected_counts.astype(np.int64),
        "audit__present_mask": result["present_mask"].astype(bool),
        "audit__current_r06_origin": baseline_origin.astype(np.int64),
        "current_r06_prediction": r06_pred.astype(np.float64),
        "legacy_baseline": legacy.astype(np.float64),
        "raw_candidate": result["raw_candidate"].astype(np.float64),
        "audit__rank_percentile": result["rank_percentile"].astype(np.float64),
        "audit__mapping_only_rank_percentile": result["mapping_only_rank_percentile"].astype(np.float64),
        "prediction": result["candidate"].astype(np.float64),
        "mapping_only_control": result["mapping_only_control"].astype(np.float64),
        "original_legacy_route_secondary": result["original_legacy_route"].astype(np.float64),
    }
    row_hash = _row_hash(score_rows, dtype="<i8")
    if not np.array_equal(arrays["score_record_ids"], ids[arrays["score_rows"]]):
        raise GateError("Candidate score IDs drifted before persistence")
    metadata = {
        "state": result["state"],
        "fit_rows": int(len(fit_rows)),
        "score_rows": int(len(score_rows)),
        "fit_label_scope": f"official Train rows with cycle_code <= {origin}",
        "fit_row_positions_sha256": _row_hash(fit_rows, dtype="<i8"),
        "fit_target_values_printed": False,
        "score_query_label_values_read": False,
        "validation_labels_passed_to_fit": False,
        "score_row_positions_sha256": row_hash,
        "score_record_ids_sha256": _id_hash(ids[score_rows]),
        "same_origin_r06_rows_in_primary": True,
        "current_r06_origin_by_cycle": {str(int(c)): int(np.unique(baseline_origin[score_cycles == c])[0]) for c in expected_query},
        "E71_H8_mixed_origin_diagnostic_only": origin == 71,
        "predictor_fits": result["predictor_fit_count"],
        "preprocessing_fit_count": result["preprocessing_fit_count"],
        "preprocessing_fit_calls": result["state"]["preprocessing_fit_calls"],
        "resource_preflight": resource,
        "official_test_accessed": False,
        "fit_labels_source_sha256": source_hash,
    }
    bundle = _write_run_bundle(origin, arrays, metadata, result["fitted_model"], protocol, commit_info)
    _append_event({
        "event": "ORIGIN_BUNDLE_SAVED",
        "origin": origin,
        "timestamp_local": pd.Timestamp.now().isoformat(),
        "bundle": bundle,
        "predictor_starts": unit_counts["predictor"],
        "preprocessing_starts": unit_counts["preprocessing"],
        "protocol_signature": protocol["signature"],
    })
    return {"status": "ORIGIN_SAVED_NO_QUERY_LABELS", "origin": origin, "bundle": bundle,
            "predictor_starts": unit_counts["predictor"], "preprocessing_starts": unit_counts["preprocessing"]}


def synthetic_check() -> dict[str, Any]:
    """Exercise the real exclusive NPZ/sidecar persistence and recovery path."""
    with tempfile.TemporaryDirectory(prefix="reg_r12_current_route_synthetic_") as tmp:
        root = Path(tmp)
        payload = {
            "score_rows": np.asarray([2, 4, 6, 8], dtype=np.int64),
            "score_record_ids": np.asarray(["SYN-2", "SYN-4", "SYN-6", "SYN-8"], dtype="<U5"),
            "audit__score_cycle": np.asarray([72, 72, 73, 73], dtype=np.int64),
            "prediction": np.asarray([1.5, 2.5, 3.5, 4.5], dtype=np.float64),
        }
        protocol = {"signature": "SYNTHETIC", "identity": {"input_sha256": {}}}
        with patch(f"{__name__}.RUN_DIR", root):
            bundle = _write_run_bundle(
                71,
                payload,
                {"synthetic": True, "score_query_label_values_read": False,
                 "validation_labels_passed_to_fit": False},
                b"synthetic-lightgbm-text-model\n",
                protocol,
                {"current_commit": "SYNTHETIC"},
            )
        npz_path = Path(bundle["npz_path"])
        model_path = Path(bundle["model_path"])
        json_path = Path(bundle["json_path"])
        recovered_sidecar = _json(json_path)
        sidecar_body = {key: value for key, value in recovered_sidecar.items() if key != "integrity_sha256"}
        if recovered_sidecar.get("identity_sha256") != _sha_bytes(_canonical_bytes(recovered_sidecar["identity"])):
            raise GateError("Synthetic persistence identity hash did not recover")
        if recovered_sidecar.get("integrity_sha256") != _sha_bytes(_canonical_bytes(sidecar_body)):
            raise GateError("Synthetic persistence sidecar integrity did not recover")
        if _sha_file(npz_path) != recovered_sidecar["npz_sha256"]:
            raise GateError("Synthetic persistence NPZ hash did not recover")
        if (_sha_file(model_path) != recovered_sidecar["model_sha256"]
                or model_path.read_bytes() != b"synthetic-lightgbm-text-model\n"):
            raise GateError("Synthetic persistence fitted-model artifact did not recover")
        with np.load(npz_path, allow_pickle=False) as recovered:
            for key in payload:
                if not np.array_equal(recovered[key], payload[key]):
                    raise GateError(f"Synthetic persistence/recovery failed for {key}")
        try:
            _write_exclusive(npz_path, npz_path.read_bytes())
        except FileExistsError:
            pass
        else:
            raise GateError("Synthetic persistence did not refuse overwrite")
    return {"status": "SYNTHETIC_PERSISTENCE_RECOVERY_PASS_NO_FITS_NO_TARGETS", "arrays": 4,
            "model_artifact_persisted": True, "overwrite_refused": True,
            "model_fit_called": False, "target_labels_read": False}


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64))))


def _metrics_for_mask(target: np.ndarray, candidate: np.ndarray, comparator: np.ndarray,
                      mapping: np.ndarray, legacy: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(np.sum(mask))
    if n == 0:
        return {"n": 0, "status": "UNKNOWN_EMPTY"}
    candidate_mae = _mae(target[mask], candidate[mask])
    comparator_mae = _mae(target[mask], comparator[mask])
    mapping_mae = _mae(target[mask], mapping[mask])
    legacy_mae = _mae(target[mask], legacy[mask])
    return {
        "n": n,
        "candidate_mae": candidate_mae,
        "current_r06_mae": comparator_mae,
        "mapping_only_control_mae": mapping_mae,
        "original_r12_legacy_route_mae_secondary": legacy_mae,
        "gain_vs_current_r06": comparator_mae - candidate_mae,
        "gain_vs_mapping_only_control": mapping_mae - candidate_mae,
        "gain_vs_original_legacy_route_secondary": legacy_mae - candidate_mae,
    }


def _target_groups(y: np.ndarray, features: pd.DataFrame, origin: int,
                   fit_edges: list[float], present: np.ndarray) -> dict[str, np.ndarray]:
    groups = {
        "availability_present": present,
        "availability_missing": ~present,
        "entry_friction_0": pd.to_numeric(features["entry_friction"], errors="coerce").to_numpy(float) == 0.0,
        "entry_friction_1": pd.to_numeric(features["entry_friction"], errors="coerce").to_numpy(float) == 1.0,
        "target_le_5": y <= 5.0,
        "target_ge_95": y >= 95.0,
    }
    text = pd.to_numeric(features["text_load"], errors="coerce").to_numpy(float)
    finite = np.isfinite(text)
    q = np.full(len(text), -1, dtype=np.int8)
    q[finite] = np.searchsorted(np.asarray(fit_edges, dtype=float), text[finite], side="left").astype(np.int8)
    for i in range(4):
        groups[f"text_load_Q{i+1}"] = q == i
    return groups


def evaluate() -> dict[str, Any]:
    protocol, preflight, commit_info = _load_frozen_protocol()
    if not PRELABEL_PATH.is_file():
        raise GateError("Independent prelabel PASS is absent; evaluation labels remain closed")
    prelabel = _json(PRELABEL_PATH)
    if prelabel.get("protocol_signature") != protocol.get("signature") or prelabel.get("status") != "PRELABEL_PASS":
        raise GateError("Independent prelabel receipt does not authorize evaluation")
    if prelabel.get("signature") != _sha_bytes(_canonical_bytes({k: v for k, v in prelabel.items() if k != "signature"})):
        raise GateError("Independent prelabel receipt signature does not verify")
    if RESULTS_PATH.exists() or SUMMARY_PATH.exists():
        raise GateError("Evaluation outputs already exist; refusing to overwrite")
    features, ids, cycles, _ = _load_feature_data()
    bundles = {}
    for origin in (71, 75):
        npz_path = RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.npz"
        sidecar_path = npz_path.with_suffix(".json")
        if _sha_file(npz_path) != prelabel["bundles"][str(origin)]["npz_sha256"] or _sha_file(sidecar_path) != prelabel["bundles"][str(origin)]["json_sha256"]:
            raise GateError("Run bundle changed after independent prelabel verification")
        with np.load(npz_path, allow_pickle=False) as data:
            bundles[origin] = {key: np.asarray(data[key]) for key in data.files}
    primary_parts: dict[str, dict[str, np.ndarray]] = {}
    for name, origin, low, high in (("W5", 71, 72, 75), ("W6", 75, 76, 79)):
        block = bundles[origin]
        mask = (block["audit__score_cycle"] >= low) & (block["audit__score_cycle"] <= high)
        primary_parts[name] = {key: value[mask] for key, value in block.items()}
    pooled_rows = np.concatenate([primary_parts["W5"]["score_rows"], primary_parts["W6"]["score_rows"]]).astype(np.int64)
    pooled_ids = np.concatenate([primary_parts["W5"]["score_record_ids"], primary_parts["W6"]["score_record_ids"]]).astype(str)
    if len(np.unique(pooled_rows)) != len(pooled_rows) or len(np.unique(pooled_ids)) != len(pooled_ids):
        raise GateError("W5/W6 primary windows are not unique by row and ID")
    y = _read_targets_for_positions(
        pooled_rows,
        expected_ids=pooled_ids,
        expected_cycles=cycles[pooled_rows],
        maximum_cycle=79,
        minimum_cycle=72,
    )
    y_by_row = {int(row): float(target) for row, target in zip(pooled_rows, y)}

    r06 = _validate_r06_outputs(features, ids, cycles)
    windows: dict[str, Any] = {}
    group_results: dict[str, Any] = {}
    for name in ("W5", "W6"):
        block = primary_parts[name]
        row = block["score_rows"].astype(np.int64)
        local_y = np.asarray([y_by_row[int(x)] for x in row], dtype=np.float64)
        local_features = features.iloc[row].copy()
        present = block["audit__present_mask"].astype(bool)
        comparator = block["current_r06_prediction"].astype(np.float64)
        candidate = block["prediction"].astype(np.float64)
        mapping = block["mapping_only_control"].astype(np.float64)
        legacy = block["original_legacy_route_secondary"].astype(np.float64)
        whole = np.ones(len(row), dtype=bool)
        windows[name] = _metrics_for_mask(local_y, candidate, comparator, mapping, legacy, whole)
        origin = 71 if name == "W5" else 75
        fit_edges = preflight["feature_group_preflight"][str(origin)]["text_load_quantile_edges_fit_only"]
        groups = _target_groups(local_y, local_features, origin, fit_edges, present)
        group_results[name] = {}
        unsupported = []
        failures = []
        for group_name, mask in groups.items():
            n = int(mask.sum())
            if not _group_supported(n):
                group_results[name][group_name] = {"n": n, "status": "UNKNOWN_UNSUPPORTED_NO_PASS"}
                unsupported.append(group_name)
                continue
            harm = _mae(local_y[mask], candidate[mask]) - _mae(local_y[mask], comparator[mask])
            record = {"n": n, "status": "SUPPORTED", "candidate_minus_current_r06_mae": harm,
                      "harm_max": GROUP_HARM_MAX, "pass": bool(harm <= GROUP_HARM_MAX)}
            group_results[name][group_name] = record
            if harm > GROUP_HARM_MAX:
                failures.append(group_name)
        group_results[name]["gate_status"] = "FAIL" if failures else ("UNKNOWN" if unsupported else "PASS")
        group_results[name]["unsupported_groups"] = unsupported
        group_results[name]["failed_groups"] = failures

    w5 = primary_parts["W5"]
    w6 = primary_parts["W6"]
    pooled_candidate = np.concatenate([w5["prediction"], w6["prediction"]]).astype(np.float64)
    pooled_r06 = np.concatenate([w5["current_r06_prediction"], w6["current_r06_prediction"]]).astype(np.float64)
    pooled_mapping = np.concatenate([w5["mapping_only_control"], w6["mapping_only_control"]]).astype(np.float64)
    pooled_legacy = np.concatenate([w5["original_legacy_route_secondary"], w6["original_legacy_route_secondary"]]).astype(np.float64)
    pooled = _metrics_for_mask(y, pooled_candidate, pooled_r06, pooled_mapping, pooled_legacy, np.ones(len(y), bool))
    gates = {
        "W5_gain_vs_same_origin_R06_ge_0": bool(windows["W5"]["gain_vs_current_r06"] >= 0.0),
        "W6_gain_vs_same_origin_R06_ge_0": bool(windows["W6"]["gain_vs_current_r06"] >= 0.0),
        "pooled_unique_gain_vs_R06_ge_0_03": bool(pooled["gain_vs_current_r06"] >= 0.03),
        "pooled_unique_gain_vs_mapping_only_ge_0_01": bool(pooled["gain_vs_mapping_only_control"] >= 0.01),
        "supported_group_harms_each_window_le_0_25": all(group_results[w]["gate_status"] == "PASS" for w in ("W5", "W6")),
    }
    if any(group_results[w]["gate_status"] == "FAIL" for w in ("W5", "W6")):
        overall = "FAIL_CLOSE_RESCUE"
    elif any(group_results[w]["gate_status"] == "UNKNOWN" for w in ("W5", "W6")):
        overall = "UNKNOWN_NO_PASS"
    else:
        overall = "PASS" if all(gates.values()) else "FAIL_CLOSE_RESCUE"

    h8_block = bundles[71]
    h8_rows = h8_block["score_rows"].astype(np.int64)
    h8_y = np.asarray([y_by_row[int(x)] for x in h8_rows], dtype=np.float64)
    h8 = _metrics_for_mask(
        h8_y,
        h8_block["prediction"].astype(np.float64),
        h8_block["current_r06_prediction"].astype(np.float64),
        h8_block["mapping_only_control"].astype(np.float64),
        h8_block["original_legacy_route_secondary"].astype(np.float64),
        np.ones(len(h8_rows), dtype=bool),
    )
    cycle_metrics = {}
    for name, origin, lo, hi in (("W5", 71, 72, 75), ("W6", 75, 76, 79)):
        block = primary_parts[name]
        local_y = np.asarray([y_by_row[int(x)] for x in block["score_rows"]], dtype=np.float64)
        for cycle in range(lo, hi + 1):
            mask = block["audit__score_cycle"] == cycle
            cycle_metrics[str(cycle)] = _metrics_for_mask(
                local_y, block["prediction"], block["current_r06_prediction"],
                block["mapping_only_control"], block["original_legacy_route_secondary"], mask,
            )
    result = {
        "namespace": NAMESPACE,
        "protocol_signature": protocol["signature"],
        "status": overall,
        "metric": "MAE; positive gain = comparator MAE - candidate MAE; positive harm = candidate MAE - comparator MAE",
        "rows": {"W5": windows["W5"]["n"], "W6": windows["W6"]["n"], "pooled_unique": int(len(y))},
        "windows": windows,
        "pooled_unique": pooled,
        "E71_H8_diagnostic_only_overlapping_W5_W6": h8,
        "per_cycle_primary": cycle_metrics,
        "supported_groups": group_results,
        "frozen_gates": gates,
        "adaptivity": "W5 labels are reused as fit labels by origin75 after serving as E71 query labels. These predictions are not fresh holdout evidence.",
        "selection": "No policy was changed after outcomes were opened; mapping-only control is fixed and zero-fit.",
        "original_legacy_route": "Secondary comparison only; exact saved legacy baseline was ID-aligned in the R06 bundles.",
        "no_test_or_kaggle": True,
    }
    _write_json_exclusive(RESULTS_PATH, result)
    lines = [
        f"# {NAMESPACE}",
        "",
        "## Decision",
        "",
        f"**{overall}** — gates and zero-fit matched comparisons are recorded in `RESULTS.json`.",
        "",
        "| View | n | New route MAE | Current R06 MAE | Mapping-only MAE | Gain vs R06 | Gain vs mapping-only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("W5", "W6"):
        item = windows[name]
        lines.append(f"| {name} | {item['n']} | {item['candidate_mae']:.6f} | {item['current_r06_mae']:.6f} | {item['mapping_only_control_mae']:.6f} | {item['gain_vs_current_r06']:+.6f} | {item['gain_vs_mapping_only_control']:+.6f} |")
    lines.append(f"| pooled unique | {pooled['n']} | {pooled['candidate_mae']:.6f} | {pooled['current_r06_mae']:.6f} | {pooled['mapping_only_control_mae']:.6f} | {pooled['gain_vs_current_r06']:+.6f} | {pooled['gain_vs_mapping_only_control']:+.6f} |")
    lines.extend([
        "",
        "The E71 H8 figure in `RESULTS.json` is diagnostic only, overlaps W5/W6, and is not included in the pooled denominator. Missing rows are exactly current R06 by construction.",
        "",
        "The original R12 legacy-baseline route is secondary only. W5/W6 are reused development outcomes; W5 labels enter the origin75 training prefix, and these predictions are not fresh holdout evidence.",
        "",
        "No Test, SampleSubmission, Kaggle, upload, dependency installation, policy grid, retry, or R06 refit was performed.",
        "",
        f"Protocol signature: `{protocol['signature']}`",
        f"Prelabel receipt signature: `{prelabel['signature']}`",
    ])
    _write_exclusive(SUMMARY_PATH, ("\n".join(lines) + "\n").encode("utf-8"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--synthetic-check", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-origin", type=int, choices=ORIGINS)
    modes.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        payload = _preflight()
    elif args.synthetic_check:
        payload = synthetic_check()
    elif args.freeze_only:
        payload = freeze_protocol()
    elif args.run_origin is not None:
        payload = run_origin(args.run_origin)
    else:
        payload = evaluate()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
