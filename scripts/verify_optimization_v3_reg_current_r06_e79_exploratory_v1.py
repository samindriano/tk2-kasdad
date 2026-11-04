"""Independent prelabel and output CSV verifier for the REG E79 batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
TRAIN_PATH = SOURCE_ROOT / "data/raw/regression/Train.csv"
TEST_PATH = SOURCE_ROOT / "data/raw/regression/Test.csv"
SAMPLE_PATH = SOURCE_ROOT / "data/raw/regression/SampleSubmission.csv"
ANCHOR_CSV = SOURCE_ROOT / "submissions/optimization_v3_expanded_v1/r06_original.csv"
ANCHOR_MANIFEST = SOURCE_ROOT / "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json"
ANCHOR_PREDICTIONS = SOURCE_ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz"
ANCHOR_FIT_STATE = SOURCE_ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/fit_state.json"
PUBLIC_REGISTRY = SOURCE_ROOT / "reports/submissions/reconciliation_20261102/REGISTRY.json"
PROTOCOL_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PROTOCOL.json"
INITIAL_FREEZE_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PREFIT_TRAIN_FREEZE.json"
SUCCESSOR_FREEZE_V2_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PREFIT_TRAIN_FREEZE_V2.json"
FREEZE_PATH = Path(os.environ.get(
    "REG_CURRENT_R06_E79_PREFIT_FREEZE_PATH", str(INITIAL_FREEZE_PATH)
)).resolve()
REPORT_DIR = ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
RUN_ROOT = ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
CSV_ROOT = ROOT / "submissions/reg_current_r06_e79_exploratory_v1"

SOURCE_FILES = (
    "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PROTOCOL.json",
    "scripts/optimization_v3_reg_current_r06_e79_exploratory_v1.py",
    "scripts/verify_optimization_v3_reg_current_r06_e79_exploratory_v1.py",
    "src/models/optimization_v3_reg_r12_current_route_late_v1.py",
    "src/models/optimization_v3_batch1_r12.py",
    "src/models/optimization_v2_regression.py",
    "src/models/classification_models.py",
    "src/models/positional_text_features.py",
    "src/models/regression_research_text.py",
    "src/models/regression_research_tabular.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/reg_position_l1_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "scripts/reg_position_direct_l1_v1.py",
    "reports/optimization_v3/REG_R12_CURRENT_ROUTE_LATE_V1/RESULTS.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_pilot_v1/RECIPE.json",
)
INPUT_FILES = {
    "official_regression_train": TRAIN_PATH,
    "official_regression_test": TEST_PATH,
    "official_regression_sample_submission": SAMPLE_PATH,
    "current_r06_e79_final_csv": ANCHOR_CSV,
    "current_r06_e79_manifest": ANCHOR_MANIFEST,
    "current_r06_e79_prediction_archive": ANCHOR_PREDICTIONS,
    "current_r06_e79_fit_state": ANCHOR_FIT_STATE,
    "r06_public_receipt_registry": PUBLIC_REGISTRY,
}


class VerifyError(RuntimeError):
    """Independent verification failure."""


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


def sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "OUS":
        payload = canonical_bytes(array.astype(str).tolist())
    else:
        payload = np.ascontiguousarray(array).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerifyError(f"Invalid JSON: {path}") from exc
    if not isinstance(result, dict):
        raise VerifyError(f"Expected a JSON object: {path}")
    return result


def sign_once(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    signed = {**body, "signature": canonical_sha(body),
              "signature_method": "SHA-256 canonical JSON"}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(signed, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return signed


def verify_signed(path: Path) -> dict[str, Any]:
    signed = read_json(path)
    body = {key: value for key, value in signed.items()
            if key not in {"signature", "signature_method"}}
    if signed.get("signature") != canonical_sha(body):
        raise VerifyError(f"Signature mismatch: {path}")
    return signed


def current_source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise VerifyError(f"Missing frozen source file: {relative}")
        result[relative] = sha_file(path)
    return result


def current_input_hashes() -> dict[str, str]:
    result = {}
    for key, path in INPUT_FILES.items():
        if not path.is_file():
            raise VerifyError(f"Missing immutable input: {path}")
        result[key] = sha_file(path)
    return result


def trained_paths(candidate: str) -> tuple[Path, Path, Path]:
    return (
        REPORT_DIR / f"TRAINED_STATE_FREEZE_{candidate}.json",
        REPORT_DIR / f"PRELABEL_PASS_{candidate}.json",
        REPORT_DIR / f"INFERENCE_MANIFEST_{candidate}.json",
    )


def predecessor_freeze_v2(active_freeze: dict[str, Any]) -> dict[str, Any]:
    meta = active_freeze.get("previous_successor_freeze", {})
    prior_path = ROOT / meta.get("path", "")
    prior = verify_signed(prior_path)
    if (sha_file(prior_path) != meta.get("sha256")
            or prior.get("signature") != meta.get("signature")
            or prior.get("status") != "PREFIT_TRAIN_FREEZE_PASS"
            or prior.get("version") != 2
            or prior.get("input_sha256") != active_freeze.get("input_sha256")
            or prior.get("resource_attestation", {}).get("signature")
               != active_freeze.get("resource_attestation", {}).get("signature")):
        raise VerifyError("Active V3 freeze does not preserve the signed V2 predecessor")
    changed_sources = {
        path for path, digest in prior.get("source_sha256", {}).items()
        if active_freeze.get("source_sha256", {}).get(path) != digest
    }
    if changed_sources != {
        "scripts/optimization_v3_reg_current_r06_e79_exploratory_v1.py",
        "scripts/verify_optimization_v3_reg_current_r06_e79_exploratory_v1.py",
    }:
        raise VerifyError("V3 and V2 source lineages differ outside the receipt-gate repair")
    return prior


def evidence_freeze(candidate: str, active_freeze: dict[str, Any]) -> dict[str, Any]:
    if candidate.upper() == "A" and active_freeze.get("version") == 3:
        return predecessor_freeze_v2(active_freeze)
    return active_freeze


def candidate_csv(candidate: str) -> Path:
    name = "r12_current_route_e79.csv" if candidate == "A" else "direct_position_real_e79.csv"
    return CSV_ROOT / name


def ledger_events() -> list[dict[str, Any]]:
    path = RUN_ROOT / "fit_ledger.jsonl"
    if not path.is_file():
        raise VerifyError("Fit ledger is missing")
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise VerifyError(f"Invalid fit ledger line {number}") from exc
    return result


def prelabel(candidate: str) -> dict[str, Any]:
    candidate = candidate.upper()
    trained_path, prelabel_path, _ = trained_paths(candidate)
    if prelabel_path.exists():
        raise VerifyError("Prelabel pass receipt exists; overwrite/retry forbidden")
    freeze = verify_signed(FREEZE_PATH)
    if freeze.get("status") != "PREFIT_TRAIN_FREEZE_PASS":
        raise VerifyError("Prefit freeze is not a passing signed Train-only freeze")
    if current_source_hashes() != freeze["source_sha256"]:
        raise VerifyError("Frozen source hashes changed")
    if current_input_hashes() != freeze["input_sha256"]:
        raise VerifyError("Frozen immutable input hashes changed")
    trained = verify_signed(trained_path)
    if trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY":
        raise VerifyError("Candidate trained-state freeze status is not valid")
    if (trained.get("candidate") != candidate
            or trained.get("prefit_freeze_signature") != freeze["signature"]
            or trained.get("source_commit") != freeze["source_commit"]
            or trained.get("source_sha256") != freeze["source_sha256"]
            or trained.get("input_sha256") != freeze["input_sha256"]):
        raise VerifyError("Trained state is not bound to the signed prefit freeze")
    if (trained.get("test_features_loaded") is not False
            or trained.get("test_labels_loaded") is not False
            or trained.get("kaggle_accessed") is not False
            or trained.get("retry_performed") is not False):
        raise VerifyError("Trained-state boundary receipt is not clean")
    if candidate == "B":
        qa_a = verify_signed(REPORT_DIR / "CSV_QA_A.json")
        manifest_a = verify_signed(REPORT_DIR / "INFERENCE_MANIFEST_A.json")
        prelabel_a = verify_signed(REPORT_DIR / "PRELABEL_PASS_A.json")
        prior_freeze = predecessor_freeze_v2(freeze)
        if (qa_a.get("status") != "PASS_INDEPENDENT_CSV_QA"
                or qa_a.get("candidate") != "A"
                or qa_a.get("active_successor_freeze_signature") != freeze.get("signature")
                or qa_a.get("inference_manifest_signature") != manifest_a.get("signature")
                or qa_a.get("prefit_freeze_signature") != prior_freeze.get("signature")
                or qa_a.get("input_sha256") != prior_freeze.get("input_sha256")
                or qa_a.get("source_sha256") != prior_freeze.get("source_sha256")
                or prelabel_a.get("prefit_freeze_signature") != prior_freeze.get("signature")
                or prelabel_a.get("source_sha256") != prior_freeze.get("source_sha256")
                or manifest_a.get("prefit_freeze_signature") != prior_freeze.get("signature")
                or manifest_a.get("input_sha256") != prior_freeze.get("input_sha256")
                or manifest_a.get("prelabel_signature") != prelabel_a.get("signature")
                or (manifest_a.get("source_sha256") is not None
                    and manifest_a.get("source_sha256") != prior_freeze.get("source_sha256"))
                or qa_a.get("csv_sha256") != sha_file(candidate_csv("A"))):
            raise VerifyError("Candidate B prelabel gate lacks a valid candidate A CSV QA handoff")
    state_path = ROOT / trained["candidate_state_path"]
    if not state_path.is_file() or sha_file(state_path) != trained["candidate_state_sha256"]:
        raise VerifyError("Trained-state bundle hash mismatch")

    events = ledger_events()
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    expected_a = ["A_TFIDF", "A_STRUCTURED", "A_SCALER", "A_OHE", "A_R12_RANKER"]
    expected_b = ["B_SHARED_WORD_TFIDF", "B_STRUCTURED", "B_SCALER", "B_OHE",
                  "B_DIRECT_POSITION_REAL_REGRESSOR"]
    expected_all = expected_a if candidate == "A" else expected_a + expected_b
    if ([event.get("fit_id") for event in starts] != expected_all
            or [event.get("fit_id") for event in saved] != expected_all
            or failed
            or len(starts) > 10
            or any(event.get("attempt") != 1 for event in starts)):
        raise VerifyError("Fit ledger does not prove exact sequential starts/saves without retries")
    candidate_ids = expected_a if candidate == "A" else expected_b
    receipts = trained.get("fit_receipts", [])
    if [receipt.get("fit_id") for receipt in receipts] != candidate_ids:
        raise VerifyError("Trained receipt learned-unit list is not the exact candidate plan")
    for receipt in receipts:
        artifact = ROOT / receipt["artifact_path"]
        if not artifact.is_file() or sha_file(artifact) != receipt["artifact_sha256"]:
            raise VerifyError(f"Learned-unit artifact hash mismatch: {receipt.get('fit_id')}")
    expected_units = 5 if candidate == "A" else 10
    if (trained.get("global_learned_units_total") != expected_units
            or trained.get("global_predictor_fits_total") != (1 if candidate == "A" else 2)
            or trained.get("global_preprocessing_fits_total") != (4 if candidate == "A" else 8)):
        raise VerifyError("Global learned-unit cap reconciliation failed")

    body = {
        "schema_version": 1,
        "namespace": "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1",
        "candidate": candidate,
        "candidate_id": trained["candidate_id"],
        "status": "PRELABEL_PASS",
        "prefit_freeze_signature": freeze["signature"],
        "trained_state_signature": trained["signature"],
        "trained_state_sha256": sha_file(trained_path),
        "input_sha256": freeze["input_sha256"],
        "source_sha256": freeze["source_sha256"],
        "fit_ids_verified": candidate_ids,
        "global_fit_starts": len(starts),
        "global_fit_saves": len(saved),
        "global_predictor_fits": 1 if candidate == "A" else 2,
        "global_preprocessing_fits": 4 if candidate == "A" else 8,
        "global_learned_units": expected_units,
        "failed_units": 0,
        "retries": 0,
        "test_file_bytes_hashed_only": True,
        "test_features_loaded": False,
        "test_labels_loaded": False,
        "evaluation_label_values_opened": False,
        "kaggle_accessed": False,
    }
    return sign_once(prelabel_path, body)


def verify_anchor_values() -> np.ndarray:
    protocol = read_json(PROTOCOL_PATH)
    anchor = protocol["anchor"]
    manifest = read_json(ANCHOR_MANIFEST)
    if (sha_file(ANCHOR_MANIFEST) != anchor["manifest_sha256"]
            or sha_file(ANCHOR_CSV) != anchor["csv_sha256"]
            or sha_file(ANCHOR_PREDICTIONS) != anchor["prediction_archive_sha256"]
            or sha_file(ANCHOR_FIT_STATE) != anchor["fit_state_sha256"]):
        raise VerifyError("Current R06 E79 anchor artifact hashes do not match the protocol")
    with np.load(ANCHOR_PREDICTIONS, allow_pickle=False) as archive:
        if "prediction" not in archive.files:
            raise VerifyError("Current R06 archive has no final prediction member")
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
    if sha_array(prediction, dtype="<f8") != anchor["final_prediction_digest"]:
        raise VerifyError("Current R06 final prediction digest is wrong")
    base = pd.read_csv(ANCHOR_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    values = pd.to_numeric(base["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (manifest.get("candidate") != "R06_ORIGINAL"
            or len(base) != 29421
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 100))
            or not np.array_equal(values, prediction)):
        raise VerifyError("Current R06 final mapped CSV does not match its prediction archive")
    return values


def csv_qa(candidate: str) -> dict[str, Any]:
    candidate = candidate.upper()
    trained_path, prelabel_path, manifest_path = trained_paths(candidate)
    qa_path = REPORT_DIR / f"CSV_QA_{candidate}.json"
    if qa_path.exists():
        raise VerifyError("CSV QA receipt exists; overwrite/retry forbidden")
    freeze = verify_signed(FREEZE_PATH)
    evidence = evidence_freeze(candidate, freeze)
    trained = verify_signed(trained_path)
    prelabel_receipt = verify_signed(prelabel_path)
    manifest = verify_signed(manifest_path)
    if (prelabel_receipt.get("status") != "PRELABEL_PASS"
            or manifest.get("status") != "CSV_READY_PENDING_INDEPENDENT_QA"
            or manifest.get("candidate") != candidate
            or trained.get("candidate") != candidate
            or trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY"
            or trained.get("prefit_freeze_signature") != evidence.get("signature")
            or trained.get("source_sha256") != evidence.get("source_sha256")
            or trained.get("input_sha256") != evidence.get("input_sha256")
            or manifest.get("trained_state_signature") != trained.get("signature")
            or manifest.get("prelabel_signature") != prelabel_receipt.get("signature")
            or prelabel_receipt.get("trained_state_signature") != trained.get("signature")
            or prelabel_receipt.get("prefit_freeze_signature") != evidence.get("signature")
            or prelabel_receipt.get("source_sha256") != evidence.get("source_sha256")
            or prelabel_receipt.get("input_sha256") != evidence.get("input_sha256")
            or manifest.get("prefit_freeze_signature") != evidence.get("signature")
            or manifest.get("input_sha256") != evidence.get("input_sha256")
            or (manifest.get("source_sha256") is not None
                and manifest.get("source_sha256") != evidence.get("source_sha256"))
            or manifest.get("test_features_loaded_after_trained_state_freeze") is not True
            or manifest.get("test_labels_loaded") is not False
            or manifest.get("kaggle_accessed") is not False
            or manifest.get("upload_performed") is not False):
        raise VerifyError("Candidate inference is not ready for independent CSV QA")
    if current_source_hashes() != freeze["source_sha256"] or current_input_hashes() != freeze["input_sha256"]:
        raise VerifyError("Frozen source/input hashes changed before CSV QA")
    csv_path = candidate_csv(candidate)
    raw_path = ROOT / manifest["raw_prediction_archive"]
    if (sha_file(csv_path) != manifest["csv_sha256"]
            or sha_file(raw_path) != manifest["raw_prediction_archive_sha256"]):
        raise VerifyError("Candidate CSV or raw prediction archive hash mismatch")

    test = pd.read_csv(TEST_PATH, usecols=["record_id", "cycle_code", "signal_stream"],
                       dtype={"record_id": "string"})
    sample = pd.read_csv(SAMPLE_PATH, usecols=["record_id"], dtype={"record_id": "string"})
    candidate_frame = pd.read_csv(csv_path, dtype={"record_id": "string"},
                                  float_precision="round_trip")
    if len(test) != 29421 or len(sample) != 29421 or len(candidate_frame) != 29421:
        raise VerifyError("CSV QA row count differs from official Test/Sample")
    if list(candidate_frame.columns) != ["record_id", "threat_rank"]:
        raise VerifyError("Candidate CSV schema is not the official two-column schema")
    ids = test["record_id"].astype(str).to_numpy(dtype=str)
    sample_ids = sample["record_id"].astype(str).to_numpy(dtype=str)
    csv_ids = candidate_frame["record_id"].astype(str).to_numpy(dtype=str)
    if (len(np.unique(ids)) != len(ids) or not np.array_equal(ids, sample_ids)
            or not np.array_equal(ids, csv_ids)):
        raise VerifyError("Candidate/Test/Sample official IDs or order differ")
    cycles = pd.to_numeric(test["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)):
        raise VerifyError("Official Test cycles are incomplete")
    expected_counts = manifest.get("test_cycle_counts")
    observed_counts = {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)}
    if expected_counts != observed_counts:
        raise VerifyError("Candidate inference cycle counts differ from independently read Test features")
    with np.load(raw_path, allow_pickle=False) as archive:
        required = {"record_id", "raw", "cycle_code", "missing", "baseline", "prediction"}
        if not required.issubset(set(archive.files)):
            raise VerifyError("Candidate raw archive is missing required members")
        raw_ids = archive["record_id"].astype(str)
        raw = np.asarray(archive["raw"], dtype=np.float64)
        archive_cycles = np.asarray(archive["cycle_code"], dtype=np.int64)
        missing = np.asarray(archive["missing"], dtype=bool)
        archived_base = np.asarray(archive["baseline"], dtype=np.float64)
        archived_prediction = np.asarray(archive["prediction"], dtype=np.float64)
    if (not np.array_equal(raw_ids, ids)
            or not np.array_equal(archive_cycles, cycles)
            or raw.shape != cycles.shape
            or missing.shape != cycles.shape
            or archived_base.shape != cycles.shape
            or archived_prediction.shape != cycles.shape
            or not np.isfinite(raw).all()):
        raise VerifyError("Candidate raw vector row identity, shape, or finiteness failed")
    baseline_values = verify_anchor_values()
    baseline_frame = pd.read_csv(ANCHOR_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    baseline_ids = baseline_frame["record_id"].astype(str).to_numpy(dtype=str)
    if not np.array_equal(baseline_ids, ids) or not np.array_equal(archived_base, baseline_values):
        raise VerifyError("Candidate did not use exact current R06 final mapped E79 predictions")
    from src.models.classification_models import _text_values
    _, independent_missing = _text_values(test["signal_stream"])
    independent_missing = np.asarray(independent_missing, dtype=bool)
    if not np.array_equal(independent_missing, missing):
        raise VerifyError("Candidate missing mask differs from the official Test text availability")
    percentile = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    if candidate == "A":
        expected = np.clip(0.75 * baseline_values + 0.25 * percentile, 0.0, 100.0)
    else:
        auxiliary_rank50 = 0.5 * raw + 0.5 * percentile
        expected = np.clip(0.75 * baseline_values + 0.25 * auxiliary_rank50, 0.0, 100.0)
    expected[missing] = baseline_values[missing]
    parsed_values = pd.to_numeric(candidate_frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.isfinite(parsed_values).all()
            or np.any((parsed_values < 0) | (parsed_values > 100))
            or not np.array_equal(parsed_values, expected)
            or not np.array_equal(archived_prediction, expected)
            or not np.array_equal(parsed_values[missing], baseline_values[missing])
            or manifest.get("raw_score_sha256") != sha_array(raw, dtype="<f8")
            or manifest.get("prediction_sha256") != sha_array(expected, dtype="<f8")
            or manifest.get("baseline_prediction_member_digest") != read_json(PROTOCOL_PATH)["anchor"]["final_prediction_digest"]
            or manifest.get("missing_rows") != int(missing.sum())
            or manifest.get("missing_fallback_bit_exact") is not True):
        raise VerifyError("Independent formula/range/missing-fallback reconstruction failed")
    body = {
        "schema_version": 1,
        "namespace": "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1",
        "candidate": candidate,
        "candidate_id": manifest["candidate_id"],
        "status": "PASS_INDEPENDENT_CSV_QA",
        "scientific_status": "EXPLORATORY_ONLY_PRIOR_LOCAL_GATES_FAILED",
        "prefit_freeze_signature": evidence["signature"],
        "input_sha256": evidence["input_sha256"],
        "source_sha256": evidence["source_sha256"],
        "active_successor_freeze_signature": freeze["signature"],
        "active_source_sha256": freeze["source_sha256"],
        "inference_manifest_signature": manifest["signature"],
        "inference_manifest_sha256": sha_file(manifest_path),
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "csv_sha256": sha_file(csv_path),
        "raw_prediction_archive_sha256": sha_file(raw_path),
        "official_test_sha256": sha_file(TEST_PATH),
        "sample_submission_sha256": sha_file(SAMPLE_PATH),
        "rows": int(len(candidate_frame)),
        "ordered_test_ids_sha256": sha_array(ids),
        "ids_unique_and_ordered_with_test_sample_and_r06": True,
        "columns": list(candidate_frame.columns),
        "finite_in_range": True,
        "missing_rows": int(missing.sum()),
        "missing_fallback_bit_exact": True,
        "full_cycle_rank_before_missing_mask": True,
        "test_labels_read": False,
        "kaggle_accessed": False,
        "upload_performed": False,
    }
    return sign_once(qa_path, body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prelabel", "csv"), required=True)
    parser.add_argument("--candidate", choices=("A", "B"), required=True)
    args = parser.parse_args()
    result = prelabel(args.candidate) if args.phase == "prelabel" else csv_qa(args.candidate)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
