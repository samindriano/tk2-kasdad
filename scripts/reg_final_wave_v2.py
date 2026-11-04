"""One-shot, outcome-blind REG final-wave affine calibration runner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.reg_position_failure_mechanism_v1 import read_scored_query
from src.models.optimization_v3_batch1_r17 import apply_affine, solve_nonnegative_l1

KASDAD = ROOT.parent
SOURCE = KASDAD / "TP 2"
FRONTIER = KASDAD / "tk2-regression-frontier"
TRAIN = SOURCE / "data/raw/regression/Train.csv"
BASE_CSV = ROOT / "submissions/reg_current_r06_e79_exploratory_v1/direct_position_real_e79.csv"
BASE_MANIFEST = ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/INFERENCE_MANIFEST_B.json"
BASE_QA = ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/CSV_QA_B.json"
BASE_EVENTS = ROOT / "reports/optimization_v3/reg_current_r06_e79_sol_uploads_v1/B/UPLOAD_EVENTS.jsonl"
DIRECT_RUN = FRONTIER / "experiments/runs/reg_position_direct_l1_v1"
DIRECT_PRED = DIRECT_RUN / "predictions"
DIRECT_FREEZE = DIRECT_RUN / "FREEZE_V4.json"
DIRECT_PRELABEL = FRONTIER / "reports/optimization_v3/reg_position_direct_l1_v1/PRELABEL_VERIFICATION.json"
R06_ARCHIVE = SOURCE / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz"
R06_STATE = SOURCE / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/fit_state.json"
R06_MANIFEST = SOURCE / "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json"
R06_CSV = SOURCE / "submissions/optimization_v3_expanded_v1/r06_original.csv"
R16_DIR = SOURCE / "experiments/runs/optimization_v3_batch1/R16"

R17_SOURCE = ROOT / "src/models/optimization_v3_batch1_r17.py"
READER_SOURCE = ROOT / "scripts/reg_position_failure_mechanism_v1.py"
DIRECT_SOURCE = FRONTIER / "scripts/reg_position_direct_l1_v1.py"
R16_MODEL_SOURCE = SOURCE / "src/models/optimization_v3_batch1_r16.py"
R16_RUNNER_SOURCE = SOURCE / "scripts/optimization_v3_batch1_r16.py"
R06_ATTEMPT = SOURCE / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/attempt.json"
DOC = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/FEASIBILITY_AND_PROTOCOL_V2.md"
SYNTH_QA = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/SYNTHETIC_QA_V2.json"
FREEZE = ROOT / "experiments/REG_FINAL_WAVE_V1/PREFIT_FREEZE_V2.json"
REVIEW = ROOT / "experiments/REG_FINAL_WAVE_V1/INDEPENDENT_REVIEW_V2.json"
RUN = ROOT / "experiments/runs/REG_FINAL_WAVE_V2"
OUT_CSV = ROOT / "submissions/reg_final_wave_v2/current_direct_affine_l1_e79.csv"
OUT_STATE = RUN / "CALIBRATOR_STATE.json"
OUT_RECEIPT = RUN / "FIT_RECEIPT.json"
OUT_MANIFEST = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/INFERENCE_MANIFEST_V2.json"
OUT_QA = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/CSV_QA_V2.json"
LEDGER = RUN / "fit_ledger.jsonl"
LOCK = RUN / "SINGLE_JOB.lock"

TRAIN_SHA = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
PINS = {
    "base_csv": (BASE_CSV, "49ae7f668aea99e433c2bec72ed5ff51126fb12d888c3185ec048730d767ea6f"),
    "base_manifest": (BASE_MANIFEST, "d2bd7dea17feab1740ca4630bb5cf7ace62caf8746051b3322cc38d168674484"),
    "base_qa": (BASE_QA, "bc19efcbbf57ae0e621fe1b338f00e1726380ed9a736947f6bf760b8a9fa67c8"),
    "base_upload_events": (BASE_EVENTS, "156738202b84d944847fc4eaec80a73a71276a08f212ee7ff2c0392e7a8c958c"),
    "base_archive": (ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/inference_B.npz", "e29b1eabb1692729749b25d88b7cf86d1bdf44c34ce74ab964b124074f5817f2"),
    "base_trained_freeze": (ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/TRAINED_STATE_FREEZE_B.json", "3364e75a8c791cebbd9fd4e2693149fa1429e7e5ef19f68bb1cadb6828b51218"),
    "base_prelabel": (ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PRELABEL_PASS_B.json", "a4771eacfe808e5645977008827ebc9552c288dbed5910606649e3d4670716c3"),
    "direct_E67": (DIRECT_PRED / "REG_POSITION_DIRECT_L1_V1_E67_REAL.npz", "116d4c51f43f96a9597b8d1c6a63af10adf022f21ac7c797f92666c7feea425f"),
    "direct_E67_receipt": (DIRECT_PRED / "REG_POSITION_DIRECT_L1_V1_E67_REAL.json", "d057b0783b311aec6afae52e1ac63dc984bd72f7cbb76210d7893d85bfdf089d"),
    "direct_E75": (DIRECT_PRED / "REG_POSITION_DIRECT_L1_V1_E75_REAL.npz", "da8e99eacfa9cb6ba3cfdac776a5e410b390e6ac90618ba22edea144c577ba39"),
    "direct_E75_receipt": (DIRECT_PRED / "REG_POSITION_DIRECT_L1_V1_E75_REAL.json", "b140894cd293afacff9121e39e49e4af4d584bd7083e7f9ff23d4372e302e489"),
    "direct_freeze": (DIRECT_FREEZE, "8294ec5197f115bf11fae1a90dacda91fcc1ad789134e228357384d958994097"),
    "direct_prelabel": (DIRECT_PRELABEL, "7aaf6e4585d29fb92678542c61ea4b7d94256508733f9a5c25161b6c3f90d914"),
    "direct_source": (DIRECT_SOURCE, "6f6dbf8b32dc5fc2ba8952aa77c9f157a2d9df7ea48e4683ab3c21508dacdc92"),
    "query_reader": (READER_SOURCE, "664ad522782e7ea84679aede04c8d9376cc94c05e3138ee90fc054fc974a2bb6"),
    "r17_source": (R17_SOURCE, "1d5d74cdf8215b8df654f5492cfa9ba396873e2bcbf58075550d9da1f331619d"),
    "r06_archive": (R06_ARCHIVE, "ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba"),
    "r06_state": (R06_STATE, "6227dae63e73ee4cf282729bac0221dc558cc08111975adf54f7348cee3f9f0a"),
    "r06_manifest": (R06_MANIFEST, "3f77e4ac148ada5fb79e5e2c16e7975e77726f4bfcc42a29f9234f801c697d67"),
    "r06_csv": (R06_CSV, "4a72b13e0d6ac93e18adee23010bb15ddf15fcb2a9543dc903154d5af60ee64f"),
    "r06_attempt": (R06_ATTEMPT, "7c4290bf18a2d0136d224f730e7100806369fc0b55e04958d813c864201ba72d"),
    "r16_fit55_npz": (R16_DIR / "R16_fit55.npz", "7a3329d3d8f78deb2d00e7baf72709989ac15628d36d353939fe7e8b349e88cf"),
    "r16_fit55_json": (R16_DIR / "R16_fit55.json", "68319eb49a3487440acf35368a1b8b8146c9f429860917be17ce02d37400c9d3"),
    "r16_fit67_npz": (R16_DIR / "R16_fit67.npz", "b165c410b217e05046483eafda7db773752cf96120e166dee476ed42c8e5ce54"),
    "r16_fit67_json": (R16_DIR / "R16_fit67.json", "7895e53c6ff2c8e04c678038653fd367bbf08ffe2fe784434e04bcd2cbfa7f2e"),
    "r16_pilot": (R16_DIR / "pilot_engine_report.json", "4f6c9ed51c993aa64bcf16c110c09029b856eed1d19ea07c1c1b1df2d3915517"),
    "r16_model_source": (R16_MODEL_SOURCE, "65e08b26da55de088fdec017b454f28594fd5920e78d04a2182787cc7c2a8ba4"),
    "r16_runner_source": (R16_RUNNER_SOURCE, "16ec66d016de06407901449cbd033860b095623c844cea46b126ca4fc53cde56"),
}

class GateError(RuntimeError):
    """Fail-closed input, protocol, or single-fit gate."""

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")

def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha_pilot_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(canonical_bytes({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(canonical_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()

def sha_ordered_ids(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(np.asarray(value, dtype=str).tolist())).hexdigest()

def sha_f64_bytes(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()

def write_once(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()

def append_event(path: Path, event: dict[str, Any]) -> None:
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value

def verify_signed(value: dict[str, Any], field: str, label: str) -> None:
    body = {k: v for k, v in value.items() if k != field and not (field == "signature" and k == "signature_method")}
    if value.get(field) != canonical_sha(body):
        raise GateError(f"{label} integrity signature failed")

def verify_files(*, check_train: bool) -> dict[str, str]:
    actual = {}
    for name, (path, expected) in PINS.items():
        if not path.is_file() or sha_file(path) != expected:
            raise GateError(f"Pinned input changed or missing: {name} ({path})")
        actual[name] = expected
    if check_train and sha_file(TRAIN) != TRAIN_SHA:
        raise GateError("Regression Train raw hash differs from frozen source")
    return actual

def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

def verify_base() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, qa = read_json(BASE_MANIFEST), read_json(BASE_QA)
    pilot_freeze, pilot_prelabel = read_json(DIRECT_FREEZE), read_json(DIRECT_PRELABEL)
    verify_signed(manifest, "signature", "base manifest")
    verify_signed(qa, "signature", "base CSV QA")
    if (manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or manifest.get("csv_sha256") != PINS["base_csv"][1]
            or manifest.get("prediction_rows") != 29421
            or manifest.get("test_labels_loaded") is not False
            or manifest.get("missing_rows") != 1062
            or manifest.get("missing_fallback_bit_exact") is not True
            or manifest.get("upload_performed") is not False):
        raise GateError("Base inference manifest identity or boundary changed")
    if (qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or qa.get("csv_sha256") != PINS["base_csv"][1]
            or qa.get("missing_rows") != 1062
            or qa.get("missing_fallback_bit_exact") is not True
            or qa.get("test_labels_read") is not False):
        raise GateError("Base CSV QA is not the exact pinned label-blind PASS")
    if (pilot_freeze.get("freeze_signature") != "80614c292413e00fd4234bb1d4da4f3b40f2aea019bb6d510dab035e356b4003"
            or pilot_prelabel.get("status") != "PASS"
            or pilot_prelabel.get("query_targets_read") is not False):
        raise GateError("Direct pilot freeze or label-blind prelabel receipt changed")
    return manifest, qa, pilot_freeze

def skip_A_evidence() -> dict[str, Any]:
    inventory = sorted(p.name for p in R16_DIR.iterdir() if p.is_file())
    expected = ["R16_fit55.json", "R16_fit55.npz", "R16_fit67.json", "R16_fit67.npz", "pilot_engine_report.json"]
    if inventory != expected:
        raise GateError(f"R16 artifact inventory changed: {inventory}")
    with np.load(R06_ARCHIVE, allow_pickle=False) as archive:
        members = list(archive.files)
    expected_members = ["champion", "raw_candidate", "rank_mapped_candidate", "cycle_code", "prediction"]
    if members != expected_members:
        raise GateError(f"R06 full-Train test archive inventory changed: {members}")
    r06_manifest, r06_state, attempt = read_json(R06_MANIFEST), read_json(R06_STATE), read_json(R06_ATTEMPT)
    if (r06_manifest.get("candidate") != "R06_ORIGINAL"
            or r06_manifest.get("prediction_artifact", {}).get("sha256") != PINS["r06_archive"][1]
            or r06_manifest.get("fit_state") != r06_state.get("fit_state")
            or r06_state.get("candidate") != "R06_ORIGINAL"
            or r06_state.get("fit_state", {}).get("id") != "V3_BATCH1_R06_CHAR_SPARSE_L1"
            or attempt.get("candidate") != "R06_ORIGINAL"):
        raise GateError("Pinned R06 manifest/state/attempt provenance no longer agrees")
    legacy = r06_manifest.get("prediction_audit", {}).get("champion_reuse", {})
    legacy_components = {str(c.get("model")): c.get("prediction_sha256")
                         for c in legacy.get("components", [])}
    required_legacy_ids = ("lgb", "ridge", "lgbm_sparse_l1")
    if legacy.get("champion_reference") != "57835122" or set(legacy_components) != set(required_legacy_ids):
        raise GateError("R06 manifest no longer identifies the exact legacy champion components")
    return {
        "status": "SKIP_A_SCOPE_LIMITED_NOT_FOUND",
        "reason": "required full-Train Ridge and sparseRank50 vectors are not present in the pinned R16/R06 artifacts",
        "search_scope": [
            "TP 2/experiments/runs/optimization_v3_batch1/R16/*",
            "TP 2/reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json",
            "TP 2/experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/{attempt.json,fit_state.json,predictions.npz}",
            "pinned R16 model and runner source files",
        ],
        "broader_workspace_search_performed": False,
        "broader_workspace_availability": "UNKNOWN",
        "r16_inventory": inventory,
        "r06_test_archive_members": members,
        "r06_legacy_champion_reference": legacy.get("champion_reference"),
        "r06_legacy_component_prediction_digests_only": legacy_components,
        "r06_current_fit_state_id": r06_state["fit_state"]["id"],
        "required_missing_components": [
            "full-Train Ridge prediction vector (R06 manifest records a digest, but the pinned archive lacks the vector)",
            "full-Train sparseRank50 prediction vector (not present in the pinned R16/R06 output set)",
        ],
        "composite_decomposition_prohibited": True,
        "candidate_fit_units_used": 0,
        "preprocessing_fit_units_used": 0,
    }

def duplicate_check(manifest: dict[str, Any]) -> dict[str, Any]:
    events = []
    if BASE_EVENTS.is_file():
        events = [json.loads(line) for line in BASE_EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if OUT_CSV.exists() or OUT_MANIFEST.exists():
        raise GateError("Affine output already exists; refusing overwrite")
    if any(e.get("fileName") == OUT_CSV.name or "CURRENT_DIRECT_AFFINE_L1_E79" in str(e.get("description", "")).upper() for e in events):
        raise GateError("Exact affine candidate identity already has a local upload event")
    if not any(e.get("fileName") == "direct_position_real_e79.csv" and e.get("ref") == 57988146 for e in events):
        raise GateError("Existing base upload event no longer matches the duplicate-check record")
    if manifest.get("csv_sha256") != PINS["base_csv"][1]:
        raise GateError("Existing base upload is not the frozen input CSV")
    return {"affine_identity_previously_uploaded": False,
            "base_identity_previously_uploaded": True, "base_ref": 57988146,
            "base_csv_sha256": PINS["base_csv"][1], "scores_used_for_selection": False}

def base_csv_values(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(BASE_CSV, dtype={"record_id": str}, float_precision="round_trip")
    if list(frame.columns) != ["record_id", "threat_rank"] or len(frame) != 29421:
        raise GateError("Base CSV schema or count changed")
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    pred = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if len(set(ids.tolist())) != len(ids) or not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 100)):
        raise GateError("Base CSV IDs or prediction values failed checks")
    ids_hash = sha_ordered_ids(ids)
    if ids_hash != manifest.get("ordered_official_test_ids_sha256"):
        raise GateError("Base CSV ordered IDs differ from the signed official-ID digest")
    with np.load(PINS["base_archive"][0], allow_pickle=False) as archive:
        archived_ids = np.asarray(archive["record_id"], dtype=str)
        archived_prediction = np.asarray(archive["prediction"], dtype=np.float64)
        archived_baseline = np.asarray(archive["baseline"], dtype=np.float64)
        archived_missing = np.asarray(archive["missing"], dtype=bool)
    if (not np.array_equal(ids, archived_ids) or not np.array_equal(pred, archived_prediction)
            or sha_f64_bytes(pred) != manifest.get("prediction_sha256")
            or int(archived_missing.sum()) != 1062
            or not np.array_equal(pred[archived_missing], archived_baseline[archived_missing])):
        raise GateError("Base CSV float values are not bit-exact with the signed saved prediction archive")
    return ids, pred, {"rows": len(ids), "ordered_ids_sha256": ids_hash,
                       "prediction_sha256": sha_pilot_array(pred, dtype="<f8"),
                       "archive_prediction_sha256": manifest.get("prediction_sha256"),
                       "csv_matches_archive_bit_exact": True,
                       "missing_rows": int(archived_missing.sum()),
                       "missing_rows_exact_r06_fallback": True,
                       "float_parser": "pandas float_precision=round_trip"}

def source_hashes() -> dict[str, str]:
    return {"script": sha_file(Path(__file__).resolve()), "protocol_doc": sha_file(DOC),
            "synthetic_qa": sha_file(SYNTH_QA), "r17_source": sha_file(R17_SOURCE),
            "query_reader": sha_file(READER_SOURCE)}

def synthetic_qa() -> None:
    if SYNTH_QA.is_file():
        report = read_json(SYNTH_QA)
        expected_plan = [
            [10, 67, 1, "id10", 68], [20, 67, 3, "id20", 69],
            [30, 75, 2, "id30", 76], [40, 75, 1, "id40", 77],
            [50, 75, 3, "id50", 78], [60, 75, 0, "id60", 79],
        ]
        if (report.get("status") != "PASS_SYNTHETIC_ONLY"
                or report.get("solver_source_sha256") != PINS["r17_source"][1]
                or report.get("candidate_data_read") is not False
                or report.get("Train_read") is not False
                or report.get("Test_or_Sample_read") is not False
                or report.get("candidate_fits") != 0
                or report.get("synthetic_solver_calls") != 1
                or report.get("preprocessing_fits") != 0
                or report.get("synthetic_rows") != 19
                or report.get("solver", {}).get("success") is not True
                or report.get("solver", {}).get("slope", -1) < 0
                or report.get("origin_stitch_plan") != expected_plan
                or report.get("origin_stitch_order_pass") is not True
                or report.get("duplicate_identity_mutation_rejected") is not True
                or report.get("max_abs_prediction_error", float("inf")) > 1e-6):
            raise GateError("Existing synthetic QA receipt is not a valid PASS for this protocol")
        print(json.dumps({"status": report["status"], "receipt_sha256": sha_file(SYNTH_QA),
                          "solver_reexecuted": False}, sort_keys=True))
        return
    if sha_file(R17_SOURCE) != PINS["r17_source"][1]:
        raise GateError("R17 source changed before synthetic QA")
    p = np.linspace(1.0, 90.0, 19, dtype=np.float64)
    y = 3.0 + 0.7 * p
    result = solve_nonnegative_l1(y, p, threads=2)
    if not result.get("success") or float(result.get("slope", -1)) < 0:
        raise GateError("Synthetic R17 LP did not return a valid nonnegative solution")
    err = float(np.max(np.abs(apply_affine(p, result["intercept"], result["slope"]) - y)))
    if err > 1e-6:
        raise GateError(f"Synthetic affine reconstruction error {err}")
    synthetic67 = {
        "query_rows": np.asarray([50, 10, 40, 20, 30, 60], dtype=np.int64),
        "query_record_ids": np.asarray(["id50", "id10", "id40", "id20", "id30", "id60"]),
        "query_cycles": np.asarray([78, 68, 77, 69, 76, 79], dtype=np.int64),
    }
    synthetic75 = {
        "query_rows": np.asarray([60, 40, 30, 50], dtype=np.int64),
        "query_record_ids": np.asarray(["id60", "id40", "id30", "id50"]),
        "query_cycles": np.asarray([79, 77, 76, 78], dtype=np.int64),
    }
    plan = stitch_oof_rows(synthetic67, synthetic75)
    expected_plan = [
        (10, 67, 1, "id10", 68), (20, 67, 3, "id20", 69),
        (30, 75, 2, "id30", 76), (40, 75, 1, "id40", 77),
        (50, 75, 3, "id50", 78), (60, 75, 0, "id60", 79),
    ]
    if plan != expected_plan:
        raise GateError(f"Synthetic origin stitch selected wrong source indices: {plan}")
    corrupted75 = {k: v.copy() for k, v in synthetic75.items()}
    corrupted75["query_record_ids"][2] = "wrong-id"
    duplicate_identity_rejected = False
    try:
        stitch_oof_rows(synthetic67, corrupted75)
    except GateError:
        duplicate_identity_rejected = True
    if not duplicate_identity_rejected:
        raise GateError("Synthetic E75 duplicate-ID mutation was not rejected")
    report = {"schema_version": 1, "status": "PASS_SYNTHETIC_ONLY",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "solver_source_sha256": PINS["r17_source"][1], "solver": result,
              "synthetic_rows": len(p), "expected_intercept": 3.0, "expected_slope": 0.7,
              "max_abs_prediction_error": err, "candidate_data_read": False,
              "Train_read": False, "Test_or_Sample_read": False,
              "candidate_fits": 0, "synthetic_solver_calls": 1, "preprocessing_fits": 0,
              "origin_stitch_synthetic_rows": len(plan),
              "origin_stitch_plan": [list(row) for row in plan],
              "origin_stitch_order_pass": True,
              "duplicate_identity_mutation_rejected": duplicate_identity_rejected}
    digest = write_once(SYNTH_QA, report)
    print(json.dumps({"status": report["status"], "sha256": digest, "max_abs_error": err}, sort_keys=True))

def prepare_freeze() -> None:
    if FREEZE.exists():
        raise GateError("Prefit freeze already exists; refusing replacement")
    if sha_file(R17_SOURCE) != PINS["r17_source"][1] or sha_file(READER_SOURCE) != PINS["query_reader"][1]:
        raise GateError("Solver or selected-row reader source hash changed")
    actual = verify_files(check_train=False)
    manifest, _qa, _pilot = verify_base()
    _ids, _pred, base_check = base_csv_values(manifest)
    candidate_a = skip_A_evidence()
    duplicate = duplicate_check(manifest)
    qa = read_json(SYNTH_QA)
    if qa.get("status") != "PASS_SYNTHETIC_ONLY" or qa.get("candidate_data_read") is not False:
        raise GateError("Synthetic QA missing or candidate data was read")
    body = {
        "schema_version": 1, "namespace": "KASDAD_REG_FINAL_WAVE_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(ROOT), "base_head": "635b6371c503f81ded3a8ac9482f0066a5b5a65c",
        "branch": "codex/reg-position-failure-mechanism-v1",
        "scientific_status": "EXPLORATORY_ONLY_IN_SAMPLE_CALIBRATION_UNVALIDATED",
        "candidate_A": candidate_a,
        "candidate_B": {
            "identity": "CURRENT_DIRECT_AFFINE_L1_E79", "baseline": "CURRENT_DIRECT_POSITION",
            "recipe": "R17 scipy.optimize.linprog(method='highs'): minimize sum(abs(y-a-b*p)), b>=0; apply clip(a+b*p,0,100) to every row of the existing B CSV",
            "oof_support": "E67 only cycles 68-75 plus E75 only cycles 76-79; unique physical rows and record IDs exactly 20,903",
            "target_read": "Train threat_rank only on the 20,903 already-scored OOF rows; no fit-prefix labels parsed",
            "test_inference": "transform existing B CSV only; no Test.csv, SampleSubmission.csv, new model.predict, refit, or external data",
            "missing_policy": "transform all rows including 1,062 former exact-R06 fallback rows; known tail risk",
            "validation_claim": None, "base_csv_sha256": PINS["base_csv"][1],
            "base_ordered_ids_sha256": base_check["ordered_ids_sha256"],
            "base_uploaded_previously_ref": 57988146, "new_affine_identity_uploaded": False},
        "budgets": {"calibrator_fits_max": 1, "preprocessing_fits": 0, "lp_calls_max": 1,
                    "synthetic_lp_calls_in_v1": 1,
                    "synthetic_lp_calls_in_v2_persisted_qa": 1,
                    "synthetic_lp_calls_in_v2_write_once_retry": 1,
                    "synthetic_lp_calls_total_before_candidate": 3,
                    "min_rows": 5000, "required_rows": 20903,
                    "threads": 2, "concurrent_jobs": 1, "min_available_ram_bytes": 4294967296,
                    "reserved_ram_bytes": 2147483648, "estimated_peak_bytes": 536870912,
                    "no_retry_after_lp_started": True},
        "input_hashes": {k: {"path": str(PINS[k][0]), "sha256": actual[k]} for k in PINS},
        "train": {"path": str(TRAIN), "sha256": TRAIN_SHA,
                  "access_rule": "hash and parse selected rows only in run mode after independent review"},
        "duplicate_check": duplicate, "source_hashes": source_hashes(),
        "review_required": True, "review_path": str(REVIEW),
        "output_paths": {"csv": str(OUT_CSV), "state": str(OUT_STATE), "receipt": str(OUT_RECEIPT),
                         "manifest": str(OUT_MANIFEST), "csv_qa": str(OUT_QA), "ledger": str(LEDGER)},
        }
    signed = {**body, "signature": canonical_sha(body),
              "signature_method": "SHA-256 canonical JSON integrity seal"}
    digest = write_once(FREEZE, signed)
    print(json.dumps({"status": "PREFIT_FREEZE_CREATED", "sha256": digest,
                      "signature": signed["signature"], "candidate_A": candidate_a["status"]}, sort_keys=True))

def load_bundle(origin: int) -> dict[str, np.ndarray]:
    fit_id = f"REG_POSITION_DIRECT_L1_V1_E{origin}_REAL"
    path = DIRECT_PRED / f"{fit_id}.npz"
    receipt = read_json(path.with_suffix(".json"))
    verify_signed(receipt, "receipt_sha256", f"{fit_id} receipt")
    if (receipt.get("fit_id") != fit_id or receipt.get("query_targets_read") is not False
            or receipt.get("bundle_sha256") != sha_file(path)):
        raise GateError(f"{fit_id} receipt does not bind its label-blind bundle")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"origin", "arm", "fit_rows", "query_rows", "query_record_ids", "query_cycles",
                "baseline", "raw_auxiliary", "mapped_auxiliary", "prediction", "missing_mask"}
    if not required.issubset(arrays) or int(arrays["origin"][0]) != origin or str(arrays["arm"][0]) != "REAL":
        raise GateError(f"{fit_id} bundle identity/fields differ")
    n = len(arrays["query_rows"])
    for name in ("query_record_ids", "query_cycles", "baseline", "raw_auxiliary",
                 "mapped_auxiliary", "prediction", "missing_mask"):
        if arrays[name].ndim != 1 or len(arrays[name]) != n:
            raise GateError(f"{fit_id} field {name} is misaligned")
    checks = {"query_cycles_sha256": sha_pilot_array(arrays["query_cycles"]),
              "query_ids_sha256": sha_pilot_array(arrays["query_record_ids"]),
              "raw_auxiliary_sha256": sha_pilot_array(arrays["raw_auxiliary"], dtype="<f8"),
              "mapped_auxiliary_sha256": sha_pilot_array(arrays["mapped_auxiliary"], dtype="<f8"),
              "prediction_sha256": sha_pilot_array(arrays["prediction"], dtype="<f8")}
    if any(receipt.get(k) != v for k, v in checks.items()):
        raise GateError(f"{fit_id} array digest differs from its receipt")
    prelabel = read_json(DIRECT_PRELABEL)
    frozen_prediction = prelabel.get("predictions_verified", {}).get(str(origin), {}).get("REAL", {})
    if (frozen_prediction.get("bundle_sha256") != sha_file(path)
            or frozen_prediction.get("receipt_sha256") != sha_file(path.with_suffix(".json"))
            or frozen_prediction.get("missing_route_exact") is not True
            or frozen_prediction.get("n_query") != n):
        raise GateError(f"{fit_id} does not match the independent frozen prelabel receipt")
    current = np.clip(0.75 * arrays["baseline"] + 0.25 * arrays["mapped_auxiliary"], 0.0, 100.0)
    missing = arrays["missing_mask"].astype(bool)
    current[missing] = arrays["baseline"][missing]
    if not np.allclose(current, arrays["prediction"], rtol=0.0, atol=1e-10):
        raise GateError(f"{fit_id} current direct policy does not reconstruct")
    return arrays

def verify_prefixes(e67: dict[str, np.ndarray], e75: dict[str, np.ndarray],
                    freeze: dict[str, Any]) -> None:
    for origin, arrays, nfit, fit_bounds, query_bounds in (
        (67, e67, 29705, [47, 67], [68, 79]),
        (75, e75, 42117, [47, 75], [76, 79])):
        state = freeze.get("origins", {}).get(str(origin), {})
        if (len(arrays["fit_rows"]) != nfit or state.get("fit_rows") != nfit
                or state.get("fit_cycle_bounds") != fit_bounds
                or state.get("query_cycle_bounds") != query_bounds):
            raise GateError(f"E{origin} temporal training prefix differs from its freeze")
        expected = {"fit_row_indices_sha256": sha_pilot_array(arrays["fit_rows"]),
                    "query_row_indices_sha256": sha_pilot_array(arrays["query_rows"]),
                    "query_record_ids_sha256": sha_pilot_array(arrays["query_record_ids"]),
                    "query_cycles_sha256": sha_pilot_array(arrays["query_cycles"])}
        if any(state.get(k) != v for k, v in expected.items()):
            raise GateError(f"E{origin} row/cycle arrays do not match the frozen pilot")
        baseline = state.get("baseline", {})
        if (baseline.get("prediction_sha256") != sha_pilot_array(arrays["baseline"], dtype="<f8")
                or baseline.get("record_ids_sha256") != sha_pilot_array(arrays["query_record_ids"])
                or baseline.get("cycles_sha256") != sha_pilot_array(arrays["query_cycles"])):
            raise GateError(f"E{origin} same-origin R06 output does not match its freeze")

def stitch_oof_rows(e67: dict[str, Any], e75: dict[str, Any]) -> list[tuple[int, int, int, str, int]]:
    """Return sorted (row, origin, original_bundle_index, record_id, cycle) selections."""
    rows67 = np.asarray(e67["query_rows"], dtype=np.int64)
    ids67 = np.asarray(e67["query_record_ids"], dtype=str)
    cyc67 = np.asarray(e67["query_cycles"], dtype=np.int64)
    rows75 = np.asarray(e75["query_rows"], dtype=np.int64)
    ids75 = np.asarray(e75["query_record_ids"], dtype=str)
    cyc75 = np.asarray(e75["query_cycles"], dtype=np.int64)
    if (len(set(rows67.tolist())) != len(rows67) or len(set(rows75.tolist())) != len(rows75)
            or set(np.unique(cyc67).tolist()) - set(range(68, 80))
            or set(np.unique(cyc75).tolist()) - set(range(76, 80))):
        raise GateError("Synthetic/real origin arrays contain duplicate rows or out-of-scope cycles")
    use67 = (cyc67 >= 68) & (cyc67 <= 75)
    use75 = (cyc75 >= 76) & (cyc75 <= 79)
    late67 = {int(rows67[i]): int(i) for i in np.flatnonzero((cyc67 >= 76) & (cyc67 <= 79))}
    selected75 = {int(rows75[i]): int(i) for i in np.flatnonzero(use75)}
    if set(late67) != set(selected75):
        raise GateError("E75 rows are not the exact E67 late-cycle duplicate subset")
    for row, i75 in selected75.items():
        i67 = late67[row]
        if str(ids75[i75]) != str(ids67[i67]) or int(cyc75[i75]) != int(cyc67[i67]):
            raise GateError("E75 duplicate row ID/cycle differs from E67")
    plan = [(int(rows67[i]), 67, int(i), str(ids67[i]), int(cyc67[i]))
            for i in np.flatnonzero(use67)]
    plan.extend((int(rows75[i]), 75, int(i), str(ids75[i]), int(cyc75[i]))
                for i in np.flatnonzero(use75))
    if (len({row for row, *_ in plan}) != len(plan)
            or len({record_id for _, _, _, record_id, _ in plan}) != len(plan)):
        raise GateError("Stitched physical rows or record IDs are not unique")
    return sorted(plan, key=lambda item: item[0])

def assemble_support() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest, _qa, pilot_freeze = verify_base()
    verify_files(check_train=True)
    ids_test, base_prediction, base_check = base_csv_values(manifest)
    duplicate = duplicate_check(manifest)
    skip_A_evidence()
    e67, e75 = load_bundle(67), load_bundle(75)
    verify_prefixes(e67, e75, pilot_freeze)
    plan = stitch_oof_rows(e67, e75)
    rows = np.asarray([row for row, _origin, _index, _id, _cycle in plan], dtype=np.int64)
    ids = np.asarray([record_id for _row, _origin, _index, record_id, _cycle in plan], dtype=str)
    cycles = np.asarray([cycle for _row, _origin, _index, _id, cycle in plan], dtype=np.int64)
    current67 = np.clip(0.75 * e67["baseline"] + 0.25 * e67["mapped_auxiliary"], 0.0, 100.0)
    current75 = np.clip(0.75 * e75["baseline"] + 0.25 * e75["mapped_auxiliary"], 0.0, 100.0)
    current67[e67["missing_mask"].astype(bool)] = e67["baseline"][e67["missing_mask"].astype(bool)]
    current75[e75["missing_mask"].astype(bool)] = e75["baseline"][e75["missing_mask"].astype(bool)]
    prediction = np.asarray([
        (current67 if origin == 67 else current75)[index]
        for _row, origin, index, _id, _cycle in plan
    ], dtype=np.float64)
    if len(rows) != 20903 or len(set(ids.tolist())) != 20903 or set(np.unique(cycles)) != set(range(68, 80)):
        raise GateError("Stitched support must be exactly 20,903 unique rows across cycles 68-79")
    labels, _text_load, parsed_rows, audit = read_scored_query(SOURCE, rows, ids, cycles, 50608)
    if not np.array_equal(parsed_rows, rows) or len(labels) != 20903:
        raise GateError("Selected-row reader returned a different support")
    if not np.isfinite(labels).all() or np.any((labels < 0.0) | (labels > 100.0)):
        raise GateError("Selected target values failed finite/range checks")
    support = {
        "rows": len(rows), "unique_ids": len(set(ids.tolist())),
        "cycle_counts": {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)},
        "rows_sha256": sha_pilot_array(rows), "ids_sha256": sha_pilot_array(ids),
        "cycles_sha256": sha_pilot_array(cycles),
        "current_direct_oof_sha256": sha_pilot_array(prediction, dtype="<f8"),
        "targets_sha256": audit["query_target_values_sha256"],
        "target_read_audit": audit,
        "E67_used_cycles": [68, 75], "E75_used_cycles": [76, 79],
        "E67_W6_used": False, "strict_temporal_prefixes": True,
        "base_csv": base_check, "duplicate_check": duplicate,
        "B_test_ids_sha256": sha_ordered_ids(ids_test),
        "test_features_or_labels_read_in_this_run": False,
    }
    return rows, ids, cycles, labels, prediction, support

def verify_review_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = read_json(FREEZE)
    verify_signed(freeze, "signature", "prefit freeze")
    if freeze.get("namespace") != "KASDAD_REG_FINAL_WAVE_V2":
        raise GateError("Prefit freeze namespace mismatch")
    if (freeze.get("candidate_A", {}).get("status") != "SKIP_A_SCOPE_LIMITED_NOT_FOUND"
            or freeze.get("candidate_B", {}).get("identity") != "CURRENT_DIRECT_AFFINE_L1_E79"
            or freeze.get("budgets", {}).get("calibrator_fits_max") != 1):
        raise GateError("Prefit candidate identity or fit budget changed")
    for name, (path, expected) in PINS.items():
        frozen = freeze.get("input_hashes", {}).get(name)
        if frozen != {"path": str(path), "sha256": expected}:
            raise GateError(f"Prefit freeze does not bind input {name}")
    if source_hashes() != freeze.get("source_hashes"):
        raise GateError("Committed source, protocol, or synthetic QA differs from the freeze")
    verify_files(check_train=False)
    if not REVIEW.is_file():
        raise GateError("Independent reviewer PASS receipt is missing")
    review = read_json(REVIEW)
    verify_signed(review, "signature", "review receipt")
    if (review.get("verdict") != "PASS" or review.get("freeze_signature") != freeze["signature"]
            or review.get("reviewed_artifacts") != freeze["source_hashes"]):
        raise GateError("Reviewer PASS does not cover this exact source/freeze/QA set")
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    str(review.get("reviewed_commit", "")), head], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if git("branch", "--show-current") != freeze.get("branch") or git("status", "--porcelain"):
        raise GateError("Expected branch or clean committed worktree before fit")
    qa = read_json(SYNTH_QA)
    if qa.get("status") != "PASS_SYNTHETIC_ONLY" or qa.get("candidate_data_read") is not False:
        raise GateError("Synthetic entrypoint QA is not a candidate-blind PASS")
    return freeze, review

def resource_gate() -> dict[str, Any]:
    available, reserve, estimate = int(psutil.virtual_memory().available), 2 * 1024**3, 512 * 1024**2
    if available < 4 * 1024**3 or estimate > available - reserve:
        raise GateError(f"RAM preflight failed: available={available}, reserve={reserve}, estimate={estimate}")
    return {"status": "PASS", "available_bytes_before_fit": available,
            "minimum_available_bytes": 4 * 1024**3, "reserve_bytes": reserve,
            "estimated_peak_bytes": estimate, "estimate_observed": False,
            "threads": 2, "concurrent_jobs": 1}

def one_job_gate() -> dict[str, Any]:
    markers = ("optimization_v3", "c10_trigram", "reg_position", "reg_final_wave",
               "lightgbm", "lgbm", "train.py", "fit.py")
    active = []
    inspected = 0
    own_ancestry = {parent.pid for parent in psutil.Process(os.getpid()).parents()}
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = proc.info
            pid = int(info.get("pid") or -1)
            name = str(info.get("name") or "").casefold()
            if pid == os.getpid() or pid in own_ancestry:
                continue
            command = " ".join(info.get("cmdline") or []).casefold()
            if not command or not ("python" in name or "pwsh" in name or "powershell" in name):
                continue
            inspected += 1
            if any(marker in command for marker in markers):
                active.append({"pid": pid, "process_name": name,
                               "command_sha256": hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if active:
        raise GateError(f"One-job gate found another active project experiment process: {active}")
    return {"status": "PASS", "other_active_project_jobs": 0,
            "inspected_python_shell_processes": inspected, "markers": list(markers)}

def write_csv_once(path: Path, ids: np.ndarray, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "threat_rank"])
        for record_id, value in zip(ids, values, strict=True):
            writer.writerow([str(record_id), format(float(value), ".17g")])
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)

def run_once() -> None:
    freeze, review = verify_review_gate()
    resource = resource_gate()
    rows, ids, cycles, labels, prediction, support = assemble_support()
    if len(labels) < 5000 or len(labels) != 20903:
        raise GateError(f"Fixed OOF row gate failed: {len(labels)}")
    manifest, _qa, _pilot = verify_base()
    test_ids, base_prediction, base_check = base_csv_values(manifest)

    # Refresh both limits immediately before the single solver call.
    resource = resource_gate()
    one_job = one_job_gate()

    RUN.mkdir(parents=True, exist_ok=False)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(b"single_job=true\n")
        stream.flush()
        os.fsync(stream.fileno())
    append_event(LEDGER, {"event": "LP_STARTED", "candidate": "CURRENT_DIRECT_AFFINE_L1_E79",
                          "time_utc": datetime.now(timezone.utc).isoformat(), "fit_rows": len(labels),
                          "threads": 2, "concurrent_jobs": 1, "reviewed_commit": review["reviewed_commit"],
                          "freeze_signature": freeze["signature"],
                          "prediction_sha256": support["current_direct_oof_sha256"],
                          "target_sha256": support["targets_sha256"]})
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        solver = solve_nonnegative_l1(labels, prediction, threads=2)
    elapsed = float(time.perf_counter() - started)
    if not solver.get("success") or solver.get("fallback"):
        write_once(RUN / "FIT_FAILURE.json", {"status": "FIT_FAILED_NO_RETRY", "solver": solver,
                                                "wall_seconds": elapsed, "candidate_fits": 1})
        append_event(LEDGER, {"event": "LP_FAILED", "wall_seconds": elapsed,
                              "time_utc": datetime.now(timezone.utc).isoformat()})
        raise GateError("Single permitted R17 LP failed; no retry and no CSV emitted")

    intercept, slope = float(solver["intercept"]), float(solver["slope"])
    calibrated = apply_affine(base_prediction, intercept, slope)
    if len(calibrated) != 29421 or not np.isfinite(calibrated).all() or np.any((calibrated < 0) | (calibrated > 100)):
        raise GateError("Transformed prediction vector failed size, finite, or range gate")
    state = {"status": "FIT_SAVED", "candidate": "CURRENT_DIRECT_AFFINE_L1_E79",
             "baseline": "CURRENT_DIRECT_POSITION", "intercept": intercept, "slope": slope,
             "solver": solver, "solver_wall_seconds": elapsed, "fit_rows": len(labels),
             "objective": "in-sample L1 fit loss only; not validation",
             "support": support, "base_csv": base_check, "resource_gate": resource,
        "one_job_gate": one_job,
             "freeze_signature": freeze["signature"], "reviewed_commit": review["reviewed_commit"],
             "created_utc": datetime.now(timezone.utc).isoformat()}
    state_hash = write_once(OUT_STATE, state)
    csv_hash = write_csv_once(OUT_CSV, test_ids, calibrated)
    written = pd.read_csv(OUT_CSV, dtype={"record_id": str}, float_precision="round_trip")
    out_ids = written["record_id"].astype(str).to_numpy(dtype=str)
    out_values = pd.to_numeric(written["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    ids_hash = sha_ordered_ids(out_ids)
    if (len(written) != 29421 or not np.array_equal(out_ids, test_ids)
            or ids_hash != manifest.get("ordered_official_test_ids_sha256")
            or not np.array_equal(out_values, calibrated)
            or not np.isfinite(out_values).all() or np.any((out_values < 0) | (out_values > 100))):
        raise GateError("Written CSV failed local schema/order/range QA")

    body = {"schema_version": 1, "namespace": "KASDAD_REG_FINAL_WAVE_V2",
            "candidate": "CURRENT_DIRECT_AFFINE_L1_E79", "baseline": "CURRENT_DIRECT_POSITION",
            "status": "EXPLORATORY_READY_PENDING_ROOT_UPLOAD",
            "scientific_status": "IN_SAMPLE_CALIBRATION_UNVALIDATED",
            "csv_path": str(OUT_CSV), "csv_sha256": csv_hash, "rows": len(written),
            "ordered_ids_sha256": ids_hash, "prediction_sha256": sha_pilot_array(out_values, dtype="<f8"),
            "range": [float(out_values.min()), float(out_values.max())],
            "base_csv_sha256": PINS["base_csv"][1], "base_upload_ref": 57988146,
            "state_sha256": state_hash, "intercept": intercept, "slope": slope,
            "fit_rows": len(labels), "fit_objective_l1_sum": solver.get("objective_l1_sum"),
            "fit_objective_mae": float(solver["objective_l1_sum"]) / len(labels),
            "validation_claim": None, "test_features_read": False, "test_labels_read": False,
            "model_predict_calls": 0, "candidate_refits": 0, "preprocessing_fits": 0,
            "scipy_lp_calls": 1, "former_r06_fallback_rows_transformed": 1062,
            "fallback_restored_after_affine": False,
            "tail_risk": "Affine transform also changes former R06 fallback rows and may harm tails.",
            "upload_performed": False, "source_hashes": freeze["source_hashes"]}
    signed_manifest = {**body, "signature": canonical_sha(body),
                       "signature_method": "SHA-256 canonical JSON integrity seal"}
    manifest_hash = write_once(OUT_MANIFEST, signed_manifest)
    receipt_body = {"status": "PASS_LOCAL_OUTPUT_QA", "candidate": body["candidate"],
                    "csv_sha256": csv_hash, "manifest_sha256": manifest_hash,
                    "rows": len(written), "columns": list(written.columns),
                    "ordered_ids_sha256": ids_hash, "finite_in_range": True,
                    "same_id_order_as_frozen_base": True, "test_labels_read": False,
                    "upload_performed": False}
    receipt = {**receipt_body, "signature": canonical_sha(receipt_body),
               "signature_method": "SHA-256 canonical JSON integrity seal"}
    receipt_hash = write_once(OUT_QA, receipt)
    write_once(OUT_RECEIPT, {"status": "FIT_AND_OUTPUT_SAVED", "state_sha256": state_hash,
                             "csv_sha256": csv_hash, "manifest_sha256": manifest_hash,
                             "csv_qa_sha256": receipt_hash, "lp_calls": 1,
                             "candidate_fits": 1, "preprocessing_fits": 0,
                             "test_features_read": False, "test_labels_read": False,
                             "upload_performed": False})
    append_event(LEDGER, {"event": "LP_COMPLETE_CSV_READY", "time_utc": datetime.now(timezone.utc).isoformat(),
                          "csv_sha256": csv_hash, "state_sha256": state_hash,
                          "manifest_sha256": manifest_hash, "upload_performed": False})
    print(json.dumps({"status": body["status"], "csv": str(OUT_CSV), "csv_sha256": csv_hash,
                      "fit_rows": len(labels), "intercept": intercept, "slope": slope,
                      "in_sample_objective_mae": body["fit_objective_mae"]}, sort_keys=True))

def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--prepare-freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.synthetic_qa:
            synthetic_qa()
        elif args.prepare_freeze:
            prepare_freeze()
        else:
            run_once()
        return 0
    except Exception as exc:
        print(f"FINAL_WAVE_NO_GO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
