"""Independent target-free integrity gate for the late R12 current-R06 pilot.

This verifier intentionally does not import the experiment runner or access the
Train target column. It checks protocol and receipt signatures, source pins,
saved R06 provenance, complete-cycle rows, output formulas, and learned-unit
accounting before the evaluation command may open query labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
TRAIN_PATH = MAIN_ROOT / "data/raw/regression/Train.csv"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
NAMESPACE = "REG_R12_CURRENT_ROUTE_LATE_V1"
FEATURES = (
    "impact_load", "entry_friction", "cycle_code", "scope_pressure", "text_load",
    "scope_trace", "org_spread", "module_spread", "telemetry_flag", "org_code",
    "module_code", "pattern_code", "signal_stream",
)
EXPERIMENT_DIR = ROOT / "experiments/REG_R12_CURRENT_ROUTE_LATE_V1"
PROTOCOL_PATH = EXPERIMENT_DIR / "PROTOCOL.json"
PREFLIGHT_PATH = EXPERIMENT_DIR / "PREFLIGHT.json"
PRELABEL_PATH = EXPERIMENT_DIR / "PRELABEL_PASS.json"
RUN_DIR = ROOT / "experiments/runs/REG_R12_CURRENT_ROUTE_LATE_V1"
ATTEMPT_PATH = RUN_DIR / "attempts.jsonl"
R06_CAMPAIGN_FREEZE = MAIN_ROOT / "experiments/optimization_v3/expanded_benchmark_v1_recovery_v2/freeze.json"
R06_RECIPE_PATH = MAIN_ROOT / "experiments/optimization_v3/batch1/R06/recipe_revision_02.json"
R06_FREEZE_PATH = MAIN_ROOT / "experiments/optimization_v3/batch1/R06/freeze_revision_02.json"
R06_ACTIVE_RUN = MAIN_ROOT / "experiments/runs/optimization_v3_batch1/R06"
R06_RECOVERY_RUN = MAIN_ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06"
R06_CAMPAIGN_SHA256 = "fcc6f16fc524729fb57d292647fd007e71ddc729ad9ff95a694d156a3b34ac7a"
R06_CAMPAIGN_SIGNATURE = "6cdecf6e0729b625981f23c7b7be4152a12806011a0a1852dd978eaec30d21ae"
R06_RECIPE_SHA256 = "6814978b3228fb741429dbd6b8af6d2c6caa8044eff0e079bbeb50066643b788"
R06_FREEZE_SHA256 = "a93302bb1132bc07a31d71597ce80f2c5e7544e49346ca8f155bfc6830ba6f11"
R12_RECIPE_SHA256 = "e76861b798a212f7b1b2bf70764db49c67a957adfa345d7173af203df939069d"
R12_FREEZE_SIGNATURE = "dbc69182a35d02032763d463c630c49ff52b2d803ac487202aefd465b25d78d3"
R06_BUNDLES = {
    71: {
        "json": "f506ed8f0e97517d6af4d591d89e5cf4cea96640a47bb4d73235573f4c0b0aa8",
        "npz": "47c97efd70dba87b8fd4187e31e85122d325528688553f7c9e8e6a81dcf58894",
        "cycles": (72, 73, 74, 75),
    },
    75: {
        "json": "61f3ab25d1cbc954cb74f47bdf823cc72113b1a2e19394c61802da1bdba7ca45",
        "npz": "61ef463a5bf5ee06f3c68162b8a7f874664b2736161979342731de98b5e09abb",
        "cycles": (76, 77, 78, 79),
    },
}
R06_PARAMETERS = {
    "boosting_type": "gbdt", "colsample_bytree": 1.0, "deterministic": True,
    "force_col_wise": True, "learning_rate": 0.05, "max_depth": -1,
    "min_child_samples": 40, "min_child_weight": 0.001, "n_estimators": 400,
    "n_jobs": 2, "num_leaves": 31, "objective": "regression_l1",
    "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 1.0,
    "subsample": 1.0, "subsample_freq": 0, "verbosity": -1,
}
R06_CHAR_PARAMETERS = {
    "analyzer": "char_wb", "dtype": "float64", "lowercase": True,
    "max_features": 30000, "min_df": 3, "ngram_range": [3, 5], "norm": "l2",
    "smooth_idf": True, "strip_accents": None, "sublinear_tf": True,
    "token_pattern": None, "use_idf": True,
}


class VerificationError(RuntimeError):
    """A target-free gate failed; query labels must remain closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Cannot read required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def _row_hash(values: Any) -> str:
    array = np.asarray(values, dtype="<i8")
    if array.ndim != 1:
        raise VerificationError("Row-position hash input must be a vector")
    return _sha_bytes(array.tobytes(order="C"))


def _id_hash(values: Any) -> str:
    ids = np.asarray(values, dtype=str)
    if ids.ndim != 1:
        raise VerificationError("Record-ID hash input must be a vector")
    return _sha_bytes("\n".join(ids.tolist()).encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _same_float_bits(name: str, actual: Any, expected: Any) -> None:
    left = np.asarray(actual, dtype="<f8")
    right = np.asarray(expected, dtype="<f8")
    if left.shape != right.shape or not np.array_equal(left.view("<u8"), right.view("<u8")):
        raise VerificationError(f"{name} differs from exact reconstructed float64 values")


def _cycle_percentiles(values: np.ndarray, cycles: np.ndarray,
                       expected_cycles: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty(len(values), dtype=np.float64)
    counts = np.empty(len(values), dtype=np.int64)
    if set(np.unique(cycles).tolist()) != set(expected_cycles):
        raise VerificationError("Saved score rows do not contain exactly the frozen cycles")
    for cycle in expected_cycles:
        mask = cycles == cycle
        size = int(mask.sum())
        if size == 0:
            raise VerificationError(f"Complete cycle {cycle} has no score rows")
        result[mask] = 100.0 * rankdata(values[mask], method="average") / size
        counts[mask] = size
    return result, counts


def _verify_git_state(source_hashes: Mapping[str, str]) -> dict[str, str]:
    repo_paths = []
    for name in source_hashes:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_relative_to(ROOT):
            repo_paths.append(path)
    repo_paths.extend([PROTOCOL_PATH, PREFLIGHT_PATH])
    rels = [path.relative_to(ROOT).as_posix() for path in repo_paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rels], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.splitlines()
    _require(not any(line.strip() for line in dirty), f"Pinned source/protocol worktree paths are dirty: {dirty}")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", *rels], cwd=ROOT,
                             check=False, capture_output=True, text=True)
    _require(tracked.returncode == 0, "Pinned source/protocol paths are not all tracked")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    return {"head": head}


def _load_source_and_protocol() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, dict[str, str]]:
    protocol = _read_json(PROTOCOL_PATH)
    preflight = _read_json(PREFLIGHT_PATH)
    identity = protocol.get("identity")
    _require(isinstance(identity, dict) and identity.get("namespace") == NAMESPACE,
             "Protocol namespace or identity is invalid")
    _require(protocol.get("signature") == _sha_bytes(_canonical(identity)), "Protocol signature is invalid")
    receipt_body = {key: value for key, value in preflight.items() if key != "receipt_signature"}
    _require(preflight.get("receipt_signature") == _sha_bytes(_canonical(receipt_body)),
             "Preflight receipt signature is invalid")
    _require(preflight.get("protocol_signature") == protocol["signature"],
             "Preflight receipt does not bind the protocol")
    _require(preflight.get("protocol_file_sha256") == _sha_bytes(_canonical(protocol)),
             "Preflight protocol file hash is invalid")
    source_hashes = identity.get("source_sha256", {})
    _require(preflight.get("source_files") == source_hashes,
             "Preflight source hashes differ from signed protocol source pins")
    for name, expected in source_hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        _require(path.is_file() and _sha_file(path) == expected, f"Pinned source changed: {name}")
    git = _verify_git_state(source_hashes)
    _require(git["head"] != identity.get("source_commit_before_protocol"),
             "Frozen protocol/source has not been committed after its prepared source revision")
    inputs = identity.get("input_sha256", {})
    _require(inputs.get("official_regression_train") == TRAIN_SHA256, "Train hash pin differs")
    _require(_sha_file(TRAIN_PATH) == TRAIN_SHA256, "Official Train bytes differ from frozen source")
    _require(_sha_file(R06_CAMPAIGN_FREEZE) == R06_CAMPAIGN_SHA256, "Current R06 campaign freeze changed")
    _require(_sha_file(R06_RECIPE_PATH) == R06_RECIPE_SHA256, "Current R06 recipe changed")
    _require(_sha_file(R06_FREEZE_PATH) == R06_FREEZE_SHA256, "Current R06 freeze changed")
    r12_recipe = MAIN_ROOT / "reports/optimization_v3/batch1/R12/recipe.json"
    r12_freeze = MAIN_ROOT / "experiments/optimization_v3/batch1/R12/freeze.json"
    _require(_sha_file(r12_recipe) == R12_RECIPE_SHA256, "Original R12 recipe changed")
    _require(_read_json(r12_freeze).get("signature") == R12_FREEZE_SIGNATURE,
             "Original R12 freeze signature changed")

    with TRAIN_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    _require(header is not None and len(header) == len(set(header)), "Train header is missing or duplicated")
    _require(set(FEATURES).issubset(header) and "record_id" in header and "threat_rank" in header,
             "Train schema lacks a required field")
    # Explicit usecols prevents the verifier from loading or parsing the target.
    frame = pd.read_csv(TRAIN_PATH, usecols=["record_id", *FEATURES], dtype={"record_id": "string"})
    _require(len(frame) == 50_608 and tuple(FEATURES) == tuple(identity.get("features", [])),
             "Train row count or frozen feature order differs")
    ids = frame["record_id"].astype(str).to_numpy()
    _require(len(np.unique(ids)) == len(ids) and not np.any(ids == ""), "Train IDs are blank or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    _require(np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)),
             "Train cycle support differs from 47..79")
    return protocol, preflight, frame, ids, cycles, git


def _verify_r06_campaign(protocol: Mapping[str, Any]) -> None:
    campaign = _read_json(R06_CAMPAIGN_FREEZE)
    identity = campaign.get("identity", {})
    input_hashes = identity.get("input_sha256", {})
    protocol_inputs = protocol["identity"]["input_sha256"]
    _require(campaign.get("signature") == R06_CAMPAIGN_SIGNATURE, "R06 campaign signature changed")
    _require(input_hashes.get("regression_controls:official_regression_train") == TRAIN_SHA256,
             "R06 campaign is not bound to the pinned Train")
    _require(input_hashes.get("R06:freeze") == R06_FREEZE_SHA256
             and input_hashes.get("R06:recipe") == R06_RECIPE_SHA256,
             "R06 campaign is not bound to the current R06 freeze and recipe")
    _require(protocol_inputs.get("R06_recovery_campaign_freeze") == R06_CAMPAIGN_SHA256,
             "Protocol does not bind the current R06 recovery campaign")
    for key, path in {
        "R06:pilot55:npz": R06_ACTIVE_RUN / "R06_fit55.npz",
        "R06:pilot55:sidecar": R06_ACTIVE_RUN / "R06_fit55.json",
        "R06:pilot67:npz": R06_ACTIVE_RUN / "R06_fit67.npz",
        "R06:pilot67:sidecar": R06_ACTIVE_RUN / "R06_fit67.json",
    }.items():
        _require(input_hashes.get(key) == _sha_file(path), f"R06 campaign source binding changed: {key}")


def _verify_feature_group_preflight(frame: pd.DataFrame, cycles: np.ndarray,
                                    preflight: Mapping[str, Any]) -> None:
    for origin, query_cycles in ((71, (72, 73, 74, 75)), (75, (76, 77, 78, 79))):
        fit_mask = cycles <= origin
        score_mask = np.isin(cycles, query_cycles)
        fit_text = pd.to_numeric(frame.loc[fit_mask, "text_load"], errors="coerce").to_numpy(dtype=np.float64)
        finite_fit_text = fit_text[np.isfinite(fit_text)]
        _require(len(finite_fit_text) > 0, f"Origin {origin} has no finite fit text_load")
        edges = np.quantile(finite_fit_text, [0.25, 0.50, 0.75], method="linear")
        score_text = pd.to_numeric(frame.loc[score_mask, "text_load"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(score_text)
        quartiles = np.full(len(score_text), -1, dtype=np.int8)
        quartiles[finite] = np.searchsorted(edges, score_text[finite], side="left").astype(np.int8)
        stream = frame.loc[score_mask, "signal_stream"]
        missing = stream.isna().to_numpy(dtype=bool) | stream.fillna("").astype(str).str.strip().eq("").to_numpy(dtype=bool)
        friction = pd.to_numeric(frame.loc[score_mask, "entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        counts = {f"text_load_Q{i + 1}": int(np.sum(quartiles == i)) for i in range(4)}
        counts.update({
            "availability_present": int(np.sum(~missing)),
            "availability_missing": int(np.sum(missing)),
            "entry_friction_0": int(np.sum(friction == 0.0)),
            "entry_friction_1": int(np.sum(friction == 1.0)),
        })
        recorded = preflight["feature_group_preflight"][str(origin)]
        _require(recorded.get("text_load_quantile_edges_fit_only") == [float(value) for value in edges]
                 and recorded.get("feature_group_support") == counts
                 and recorded.get("groups_supported_before_fit") == {name: (n >= 100) for name, n in counts.items()},
                 f"Origin {origin} feature-group thresholds or support changed from signed preflight")
        _require(all(n >= 100 for n in counts.values()),
                 f"Origin {origin} has a feature group below the frozen n >= 100 support gate")


def _read_current_r06(origin: int, ids: np.ndarray, cycles: np.ndarray,
                      protocol: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected = R06_BUNDLES[origin]
    json_path = R06_RECOVERY_RUN / f"R06_fit{origin}.json"
    npz_path = R06_RECOVERY_RUN / f"R06_fit{origin}.npz"
    _require(_sha_file(json_path) == expected["json"] and _sha_file(npz_path) == expected["npz"],
             f"Current R06 origin {origin} hash differs")
    inputs = protocol["identity"]["input_sha256"]
    _require(inputs[f"R06_fit{origin}_json"] == expected["json"]
             and inputs[f"R06_fit{origin}_npz"] == expected["npz"],
             f"Protocol R06 origin {origin} hash pins differ")
    sidecar = _read_json(json_path)
    metadata = sidecar.get("metadata", {})
    state = metadata.get("state", {})
    _require(sidecar.get("npz_sha256") == expected["npz"], f"Current R06 origin {origin} sidecar NPZ hash differs")
    _require(sidecar.get("identity", {}).get("card_id") == "R06"
             and int(sidecar["identity"].get("cutoff", -1)) == origin
             and sidecar["identity"].get("campaign_freeze_signature") == R06_CAMPAIGN_SIGNATURE,
             f"Current R06 origin {origin} identity is invalid")
    _require(metadata.get("freeze_signature") == R06_CAMPAIGN_SIGNATURE
             and metadata.get("test_data_used") is False
             and metadata.get("validation_labels_passed_to_fit") is False
             and int(metadata.get("fit_predictor_count", -1)) == 1,
             f"Current R06 origin {origin} fit boundary is invalid")
    expected_fingerprint = {
        "id": "V3_BATCH1_R06_CHAR_SPARSE_L1",
        "recipe": "char_wb TF-IDF ngrams 3-5 + structured/OHE + missing indicator; LightGBM regression_l1",
        "parameters": R06_PARAMETERS,
        "input_feature_names": list(FEATURES),
        "char_tfidf_parameters": R06_CHAR_PARAMETERS,
        "missing_stream_indicator": True,
        "svd_used": False,
        "word_tfidf_used": False,
        "native_categorical_features_used": False,
    }
    fingerprint = {
        "id": state.get("id"), "recipe": state.get("recipe"), "parameters": state.get("parameters"),
        "input_feature_names": state.get("input_feature_names"),
        "char_tfidf_parameters": state.get("char_tfidf", {}).get("parameters"),
        "missing_stream_indicator": state.get("missing_stream_indicator"),
        "svd_used": state.get("svd_used"), "word_tfidf_used": state.get("word_tfidf_used"),
        "native_categorical_features_used": state.get("native_categorical_features_used"),
    }
    _require(fingerprint == expected_fingerprint, f"Current R06 origin {origin} recipe fingerprint differs")
    _require(int(metadata.get("fit_rows", -1)) == int(np.sum(cycles <= origin)),
             f"Current R06 origin {origin} fit-row boundary differs")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name in ("score_rows", "score_record_ids", "raw_candidate", "prediction", "baseline_prediction",
                 "audit__candidate_rank50", "score_cycles"):
        _require(name in arrays, f"Current R06 origin {origin} lacks {name}")
    rows = arrays["score_rows"].astype(np.int64)
    score_ids = arrays["score_record_ids"]
    _require(score_ids.dtype.kind == "U" and np.array_equal(score_ids.astype(str), ids[rows]),
             f"Current R06 origin {origin} IDs do not match Train row positions")
    score_cycles = cycles[rows]
    _require(np.array_equal(np.unique(score_cycles), np.asarray(expected["cycles"], dtype=np.int64))
             and np.array_equal(arrays["score_cycles"].astype(np.int64), score_cycles),
             f"Current R06 origin {origin} does not contain exact complete query cycles")
    raw = arrays["raw_candidate"].astype(np.float64)
    baseline = arrays["baseline_prediction"].astype(np.float64)
    rank50 = np.empty(len(rows), dtype=np.float64)
    for cycle in expected["cycles"]:
        mask = score_cycles == cycle
        pct = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        rank50[mask] = np.clip(0.5 * raw[mask] + 0.5 * pct, 0.0, 100.0)
    _same_float_bits(f"R06 origin {origin} rank50", arrays["audit__candidate_rank50"], rank50)
    _same_float_bits(f"R06 origin {origin} prediction", arrays["prediction"],
                     np.clip(0.75 * baseline + 0.25 * rank50, 0.0, 100.0))
    _require(np.isfinite(raw).all() and np.isfinite(baseline).all()
             and np.isfinite(arrays["prediction"]).all(), f"Current R06 origin {origin} has nonfinite score values")
    return {
        "rows": rows, "ids": score_ids.astype(str), "cycles": score_cycles,
        "prediction": arrays["prediction"].astype(np.float64),
        "legacy_baseline": baseline,
        "json_sha256": expected["json"], "npz_sha256": expected["npz"],
    }


def _expected_preprocessing_calls(frame: pd.DataFrame, fit_rows: np.ndarray) -> list[str]:
    stream = frame.iloc[fit_rows]["signal_stream"].to_numpy(dtype=object)
    has_text = any(not pd.isna(value) and bool(str(value).strip()) for value in stream)
    return ((["word_tfidf"] if has_text else [])
            + ["structured_processor", "numeric_scaler", "categorical_ohe"])


def _verify_attempts(frame: pd.DataFrame, cycles: np.ndarray) -> list[dict[str, Any]]:
    _require(ATTEMPT_PATH.is_file(), "Attempt ledger is missing")
    events = [json.loads(line) for line in ATTEMPT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(not any(event.get("event") == "ORIGIN_FAILED_NO_RETRY" for event in events),
             "Attempt ledger contains a failed origin; pilot is closed")
    _require(not any(event.get("event") == "LEARNED_UNIT_FAILED" for event in events),
             "Attempt ledger contains a failed learned unit")
    for event_name in ("ORIGIN_PREFIT_PASS", "ORIGIN_ATTEMPT_STARTED", "ORIGIN_BUNDLE_SAVED"):
        found = [event for event in events if event.get("event") == event_name]
        _require(len(found) == 2 and {int(item.get("origin", -1)) for item in found} == {71, 75},
                 f"Expected exactly one {event_name} for origins 71 and 75")
    for event in events:
        _require(int(event.get("origin", -1)) in {71, 75}, "Attempt ledger contains an unauthorized origin")
        if event.get("event") == "ORIGIN_ATTEMPT_STARTED":
            _require(float(event.get("resource_recheck_available_ram_gib", 0)) >= 4.0
                     and float(event.get("resource_recheck_projected_reserve_gib", 0)) >= 2.0,
                     "Immediate pre-fit resource receipt is below the frozen threshold")
    starts = [event for event in events if event.get("event") == "LEARNED_UNIT_STARTED"]
    finishes = [event for event in events if event.get("event") in {"LEARNED_UNIT_COMPLETED", "LEARNED_UNIT_FAILED"}]
    preprocessing_starts = sum(event.get("unit_type") == "preprocessing" for event in starts)
    _require(sum(event.get("unit_type") == "predictor" for event in starts) == 2
             and preprocessing_starts <= 8,
             "Learned-unit counts exceed or differ from the frozen two-origin budget")
    allowed = {"word_tfidf", "structured_processor", "numeric_scaler", "categorical_ohe", "lgbm_ranker_predictor"}
    _require(all(event.get("unit_name") in allowed for event in starts), "Attempt ledger contains an unauthorized learned unit")
    _require(all(sum(done.get("unit_id") == start.get("unit_id") for done in finishes) == 1 for start in starts),
             "Every learned start must have exactly one completion record")
    _require(len(finishes) == len(starts), "Attempt ledger has an unmatched or duplicate learned-unit completion")
    for origin in (71, 75):
        local = [event for event in starts if int(event["origin"]) == origin]
        _require(sum(event.get("unit_name") == "lgbm_ranker_predictor" for event in local) == 1,
                 f"Origin {origin} must have exactly one ranker fit")
        fit_rows = np.flatnonzero(cycles <= origin)
        expected_names = _expected_preprocessing_calls(frame, fit_rows)
        actual_names = [event.get("unit_name") for event in local if event.get("unit_type") == "preprocessing"]
        _require(actual_names == expected_names,
                 f"Origin {origin} learned preprocessing calls differ from fit-prefix data and recipe")
        _require(all(done.get("details", {}).get("status") != "FAILED" for done in finishes
                     if any(done.get("unit_id") == start.get("unit_id") for start in local)),
                 f"Origin {origin} has a failed learned unit")
    return events


def _verify_bundle_file_integrity(origin: int) -> dict[str, Any]:
    npz_path = RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.npz"
    model_path = RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.txt"
    sidecar_path = npz_path.with_suffix(".json")
    _require(npz_path.is_file() and model_path.is_file() and sidecar_path.is_file(),
             f"Origin {origin} is missing its model, NPZ, or JSON artifact")
    sidecar = _read_json(sidecar_path)
    sidecar_body = {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
    _require(sidecar.get("integrity_sha256") == _sha_bytes(_canonical(sidecar_body)),
             f"Origin {origin} sidecar integrity failed")
    identity = sidecar.get("identity", {})
    model_hash = _sha_file(model_path)
    _require(sidecar.get("identity_sha256") == _sha_bytes(_canonical(identity)),
             f"Origin {origin} identity signature failed")
    _require(sidecar.get("npz_sha256") == _sha_file(npz_path), f"Origin {origin} NPZ hash differs from sidecar")
    _require(identity.get("model_artifact_file") == model_path.name
             and identity.get("model_format") == "lightgbm_native_text"
             and identity.get("model_sha256") == model_hash
             and sidecar.get("model_sha256") == model_hash,
             f"Origin {origin} fitted model artifact is absent or does not match its hash receipt")
    return sidecar


def _verify_new_bundle(origin: int, protocol: Mapping[str, Any], preflight: Mapping[str, Any],
                       frame: pd.DataFrame, ids: np.ndarray, cycles: np.ndarray,
                       current_r06: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    npz_path = RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.npz"
    model_path = RUN_DIR / f"R12_CURRENT_ROUTE_fit{origin}.txt"
    sidecar_path = npz_path.with_suffix(".json")
    sidecar = _verify_bundle_file_integrity(origin)
    identity = sidecar.get("identity", {})
    _require(identity.get("namespace") == NAMESPACE and identity.get("origin") == origin
             and identity.get("protocol_signature") == protocol["signature"]
             and identity.get("train_sha256") == TRAIN_SHA256,
             f"Origin {origin} bundle identity does not match protocol")
    _require(identity.get("score_cycles") == list(range(72, 80) if origin == 71 else range(76, 80)),
             f"Origin {origin} bundle cycles differ from frozen scope")
    input_hashes = protocol["identity"]["input_sha256"]
    _require(identity.get("r06_bundle_sha256") == input_hashes,
             f"Origin {origin} bundle is not bound to the exact current R06 bundle hashes")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "score_rows", "score_record_ids", "audit__score_cycle", "audit__expected_full_cycle_count",
        "audit__present_mask", "audit__current_r06_origin", "current_r06_prediction", "legacy_baseline",
        "raw_candidate", "audit__rank_percentile", "audit__mapping_only_rank_percentile", "prediction",
        "mapping_only_control", "original_legacy_route_secondary",
    }
    _require(set(arrays) == required, f"Origin {origin} bundle array names differ from the frozen contract")
    rows = arrays["score_rows"].astype(np.int64)
    score_ids = arrays["score_record_ids"]
    _require(score_ids.dtype.kind == "U" and rows.ndim == 1 and np.array_equal(score_ids.astype(str), ids[rows]),
             f"Origin {origin} row positions and Unicode IDs do not match official Train")
    score_cycles = cycles[rows]
    expected_cycles = tuple(range(72, 80) if origin == 71 else range(76, 80))
    _require(np.array_equal(np.unique(score_cycles), np.asarray(expected_cycles, dtype=np.int64)),
             f"Origin {origin} score rows do not cover complete expected cycles")
    support = preflight["origin_support"][str(origin)]
    _require(_row_hash(rows) == support["score_row_positions_sha256"]
             and _id_hash(score_ids) == support["score_record_ids_sha256"],
             f"Origin {origin} score row positions/IDs differ from signed preflight")
    score_counts = np.asarray([int(np.sum(score_cycles == cycle)) for cycle in score_cycles], dtype=np.int64)
    _require(np.array_equal(arrays["audit__score_cycle"].astype(np.int64), score_cycles)
             and np.array_equal(arrays["audit__expected_full_cycle_count"].astype(np.int64), score_counts),
             f"Origin {origin} score cycle arrays or full-cycle counts differ")

    fit_rows = np.flatnonzero(cycles <= origin)
    _require(len(fit_rows) == support["fit_rows"]
             and _row_hash(fit_rows) == support["fit_row_positions_sha256"]
             and _id_hash(ids[fit_rows]) == support["fit_record_ids_sha256"],
             f"Origin {origin} fit prefix positions/IDs differ from frozen support")
    metadata = sidecar.get("metadata", {})
    state = metadata.get("state", {})
    preprocessing_calls = _expected_preprocessing_calls(frame, fit_rows)
    _require(metadata.get("fit_rows") == len(fit_rows) and metadata.get("score_rows") == len(rows)
             and metadata.get("fit_row_positions_sha256") == _row_hash(fit_rows)
             and metadata.get("fit_label_scope") == f"official Train rows with cycle_code <= {origin}"
             and metadata.get("fit_target_values_printed") is False
             and metadata.get("score_query_label_values_read") is False
             and metadata.get("validation_labels_passed_to_fit") is False
             and metadata.get("official_test_accessed") is False,
             f"Origin {origin} sidecar fit/query label boundary is invalid")
    fit_cycles = cycles[fit_rows]
    fit_counts = np.asarray([int(np.sum(fit_cycles == cycle)) for cycle in range(47, origin + 1)], dtype=np.int64)
    expected_parameters = {
        "boosting_type": "gbdt", "colsample_bytree": 1.0, "deterministic": True,
        "force_col_wise": True, "learning_rate": 0.05, "max_depth": -1,
        "min_child_samples": 40, "min_child_weight": 0.001, "n_estimators": 200,
        "n_jobs": 2, "num_leaves": 15, "objective": "lambdarank", "random_state": 42,
        "reg_alpha": 0.0, "reg_lambda": 3.0, "subsample": 1.0, "subsample_freq": 0,
        "verbosity": -1, "label_gain": list(range(10)),
        "lambdarank_truncation_level": int(fit_counts.max()),
    }
    _require(state.get("namespace") == NAMESPACE and state.get("cutoff") == origin
             and state.get("parameters_explicit") == expected_parameters
             and state.get("grade_definition") == "clip(floor(threat_rank/10),0,9)"
             and state.get("query_cycles") == list(range(47, origin + 1))
             and state.get("query_group_sizes") == fit_counts.tolist()
             and state.get("maximum_query_size") == int(fit_counts.max())
             and state.get("fitted_trees") == 200
             and state.get("predictor_fit_count") == 1
             and state.get("preprocessing_fit_calls") == preprocessing_calls
             and state.get("validation_labels_passed_to_fit") is False
             and state.get("test_data_used") is False,
             f"Origin {origin} R12 fit state differs from its exact two-origin recipe")
    _require(state.get("fitted_model_sha256") == _sha_file(model_path),
             f"Origin {origin} fitted model hash differs from the durably saved LightGBM text")
    state_body = {key: value for key, value in state.items() if key != "state_sha256"}
    _require(state.get("state_sha256") == _sha_bytes(_canonical(state_body)),
             f"Origin {origin} model state hash is invalid")
    _require(metadata.get("predictor_fits") == 1 and metadata.get("preprocessing_fit_count") == len(preprocessing_calls)
             and metadata.get("preprocessing_fit_calls") == state.get("preprocessing_fit_calls")
             and metadata.get("score_record_ids_sha256") == _id_hash(score_ids),
             f"Origin {origin} learned-unit sidecar counts or score ID hash differ")

    # Verify row-aligned saved R06 prediction and legacy baseline, including
    # the context-only mixed-origin portion of the origin71 H8 output.
    by_row: dict[int, tuple[str, float, float, int]] = {}
    for r06_origin in (71, 75):
        baseline = current_r06[r06_origin]
        for index, row in enumerate(baseline["rows"]):
            _require(int(row) not in by_row, "Saved R06 origins overlap on a score row")
            by_row[int(row)] = (str(baseline["ids"][index]), float(baseline["prediction"][index]),
                                float(baseline["legacy_baseline"][index]), r06_origin)
    expected_r06 = np.asarray([by_row[int(row)][1] for row in rows], dtype=np.float64)
    expected_legacy = np.asarray([by_row[int(row)][2] for row in rows], dtype=np.float64)
    expected_r06_origin = np.asarray([by_row[int(row)][3] for row in rows], dtype=np.int64)
    _require(all(by_row[int(row)][0] == str(record_id) for row, record_id in zip(rows, score_ids)),
             f"Origin {origin} current R06 row identity is not exact")
    _same_float_bits(f"Origin {origin} current R06", arrays["current_r06_prediction"], expected_r06)
    _same_float_bits(f"Origin {origin} legacy baseline", arrays["legacy_baseline"], expected_legacy)
    _require(np.array_equal(arrays["audit__current_r06_origin"].astype(np.int64), expected_r06_origin),
             f"Origin {origin} current R06 origin-by-cycle map differs")
    _require(np.array_equal(arrays["audit__present_mask"].astype(bool),
                            frame.iloc[rows]["signal_stream"].notna().to_numpy()
                            & frame.iloc[rows]["signal_stream"].fillna("").astype(str).str.strip().ne("").to_numpy()),
             f"Origin {origin} availability mask differs from signal_stream")

    raw = arrays["raw_candidate"].astype(np.float64)
    candidate_percentile, exact_counts = _cycle_percentiles(raw, score_cycles, expected_cycles)
    r06_percentile, _ = _cycle_percentiles(expected_r06, score_cycles, expected_cycles)
    legacy_percentile, _ = _cycle_percentiles(expected_legacy, score_cycles, expected_cycles)
    _require(np.array_equal(exact_counts, score_counts), f"Origin {origin} cycle percentile denominator changed")
    _same_float_bits(f"Origin {origin} R12 percentiles", arrays["audit__rank_percentile"], candidate_percentile)
    _same_float_bits(f"Origin {origin} R06 mapping percentiles", arrays["audit__mapping_only_rank_percentile"], r06_percentile)
    present = arrays["audit__present_mask"].astype(bool)
    candidate_present = np.clip(0.75 * expected_r06 + 0.25 * candidate_percentile, 0.0, 100.0)
    mapping_present = np.clip(0.75 * expected_r06 + 0.25 * r06_percentile, 0.0, 100.0)
    legacy_present = np.clip(0.75 * expected_legacy + 0.25 * candidate_percentile, 0.0, 100.0)
    expected_candidate = np.where(present, candidate_present, expected_r06)
    expected_mapping = np.where(present, mapping_present, expected_r06)
    expected_original = np.where(present, legacy_present, expected_legacy)
    _same_float_bits(f"Origin {origin} primary route", arrays["prediction"], expected_candidate)
    _same_float_bits(f"Origin {origin} mapping-only control", arrays["mapping_only_control"], expected_mapping)
    _same_float_bits(f"Origin {origin} secondary legacy route", arrays["original_legacy_route_secondary"], expected_original)
    _same_float_bits(f"Origin {origin} missing-row R06 fallback", arrays["prediction"][~present], expected_r06[~present])
    for name in ("prediction", "mapping_only_control", "original_legacy_route_secondary"):
        _require(np.isfinite(arrays[name]).all() and np.all((arrays[name] >= 0.0) & (arrays[name] <= 100.0)),
                 f"Origin {origin} {name} has invalid values")

    return {"origin": origin, "rows": int(len(rows)), "row_positions_sha256": _row_hash(rows),
            "record_ids_sha256": _id_hash(score_ids), "npz_sha256": _sha_file(npz_path),
            "json_sha256": _sha_file(sidecar_path), "model_sha256": _sha_file(model_path),
            "predictor_fits": 1, "preprocessing_fits": len(preprocessing_calls),
            "score_query_label_values_read": False,
            "formula_reconstruction": "exact float64 bit equality"}


def verify_prelabel() -> dict[str, Any]:
    _require(not PRELABEL_PATH.exists(), "Prelabel PASS already exists; refusing to overwrite")
    protocol, preflight, frame, ids, cycles, git = _load_source_and_protocol()
    _verify_feature_group_preflight(frame, cycles, preflight)
    _verify_r06_campaign(protocol)
    current_r06 = {origin: _read_current_r06(origin, ids, cycles, protocol) for origin in (71, 75)}
    events = _verify_attempts(frame, cycles)
    expected_run_names = {
        "attempts.jsonl",
        "R12_CURRENT_ROUTE_fit71.json", "R12_CURRENT_ROUTE_fit71.npz", "R12_CURRENT_ROUTE_fit71.txt",
        "R12_CURRENT_ROUTE_fit75.json", "R12_CURRENT_ROUTE_fit75.npz", "R12_CURRENT_ROUTE_fit75.txt",
    }
    _require({path.name for path in RUN_DIR.iterdir()} == expected_run_names,
             "Run namespace contains missing or unauthorized files")
    bundle_receipts = {
        str(origin): _verify_new_bundle(origin, protocol, preflight, frame, ids, cycles, current_r06)
        for origin in (71, 75)
    }
    w5 = np.flatnonzero(np.isin(cycles, (72, 73, 74, 75)))
    w6 = np.flatnonzero(np.isin(cycles, (76, 77, 78, 79)))
    _require(not np.intersect1d(w5, w6).size and len(np.union1d(w5, w6)) == len(w5) + len(w6),
             "Primary W5/W6 windows overlap or duplicate rows")
    windows = preflight["primary_windows"]
    _require(_row_hash(w5) == windows["W5"]["row_positions_sha256"]
             and _id_hash(ids[w5]) == windows["W5"]["record_ids_sha256"]
             and _row_hash(w6) == windows["W6"]["row_positions_sha256"]
             and _id_hash(ids[w6]) == windows["W6"]["record_ids_sha256"],
             "Primary window row or ID hashes differ from the signed preflight")
    _require(set(current_r06[71]["rows"].tolist()) == set(w5.tolist())
             and set(current_r06[75]["rows"].tolist()) == set(w6.tolist()),
             "Same-origin current R06 bundles do not exactly cover primary W5/W6 rows")
    _require(len(events) > 0, "No attempt evidence was verified")
    body = {
        "namespace": NAMESPACE,
        "status": "PRELABEL_PASS",
        "created_at_local": pd.Timestamp.now().isoformat(),
        "protocol_signature": protocol["signature"],
        "protocol_file_sha256": preflight["protocol_file_sha256"],
        "preflight_receipt_signature": preflight["receipt_signature"],
        "source_commit": git["head"],
        "train_sha256": TRAIN_SHA256,
        "evaluation_label_values_opened": False,
        "score_query_label_values_read": False,
        "origin75_fit_prefix_reused_W5_labels_after_origin71_bundle": True,
        "test_sample_submission_or_kaggle_accessed": False,
        "r06_provenance_verified": True,
        "complete_cycle_and_id_checks": True,
        "candidate_and_control_formulas_bit_exact": True,
        "learned_unit_starts": {
            "ranker": 2,
            "preprocessing": sum(event.get("unit_type") == "preprocessing"
                                 for event in events if event.get("event") == "LEARNED_UNIT_STARTED"),
        },
        "bundles": bundle_receipts,
        "attempt_event_count": len(events),
        "file_namespace": sorted(expected_run_names),
    }
    signed = {**body, "signature": _sha_bytes(_canonical(body))}
    encoded = _canonical(signed) + b"\n"
    fd = os.open(PRELABEL_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return signed


def main() -> int:
    result = verify_prelabel()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
