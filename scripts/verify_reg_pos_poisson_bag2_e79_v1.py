"""Independent read-only QA for the authorized E79 POS Poisson artifact set."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "REG_POS_POISSON_BAG2_E79_V1"
RUN = ROOT / "experiments/runs/REG_POS_POISSON_BAG2_E79_V1"
PARENT = ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/inference_B.npz"
PARENT_MANIFEST = ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/INFERENCE_MANIFEST_B.json"
SOURCE_REPORT = ROOT / "reports/optimization_v3/REG_POS_POISSON_BAG2_E79_V1/SOURCE_PREPARATION_R4.json"
RUNNER = ROOT / "scripts/reg_pos_poisson_bag2_e79_v1.py"
FINAL_PASS = ROOT / "experiments/reg_pos_poisson_bag2_e79_v1/ROOT_FINAL_SOURCE_PASS_R2.json"
FINAL_CSV = ROOT / "submissions/reg_pos_poisson_bag2_e79_v1/pos_poisson_bag2_final_e79.csv"
CHAMPION_CSV = ROOT / "submissions/reg_monotone_hinge_p50_v1/hinge_p50_final_exploratory_e79.csv"
SAMPLE_CSV = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\data\raw\regression\SampleSubmission.csv")
TEST_CSV = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\data\raw\regression\Test.csv")
R06_BASELINE_CSV = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\submissions\optimization_v3_expanded_v1\r06_original.csv")
QA_REPORT = ROOT / "reports/optimization_v3/REG_POS_POISSON_BAG2_E79_V1/FINAL_ARTIFACT_QA_R1.json"
FIT_IDS = ("REG_POS_POISSON_BAG2_E79_V1_P17", "REG_POS_POISSON_BAG2_E79_V1_P137")
HINGE = {"intercept": 1.7288079036674198, "slope_low": 0.9933285599956824,
         "slope_high": 0.957778072288735, "knot": 50.0}


class QAError(RuntimeError):
    pass


def canon_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canon_sha(value: Any) -> str:
    return hashlib.sha256(canon_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    if not path.is_file():
        raise QAError(f"Missing required artifact: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(canon_bytes({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(canon_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def parent_bytes_sha(value: Any) -> str:
    array = np.asarray(value, dtype="<f8")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QAError(f"Expected a JSON object: {path}")
    return value


def check_signature(path: Path, key: str) -> dict[str, Any]:
    value = read_json(path)
    body = {k: v for k, v in value.items() if k != key}
    if value.get(key) != canon_sha(body):
        raise QAError(f"Canonical {key} mismatch: {path}")
    return value


def rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    output = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        output[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return output


def apply_hinge(direct: np.ndarray, missing: np.ndarray) -> np.ndarray:
    output = np.asarray(direct, dtype=np.float64).copy()
    present = ~np.asarray(missing, dtype=bool)
    x = output[present]
    y = (HINGE["intercept"] + HINGE["slope_low"] * np.minimum(x, HINGE["knot"])
         + HINGE["slope_high"] * np.maximum(x - HINGE["knot"], 0.0))
    output[present] = np.clip(y, 0.0, 100.0)
    return output


def exact(actual: Any, expected: Any, label: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise QAError(f"Exact readback/parity failed: {label}")


def verify() -> dict[str, Any]:
    source = read_json(SOURCE_REPORT)
    source_body = {k: v for k, v in source.items() if k != "source_preparation_sha256"}
    source_hash = file_sha(SOURCE_REPORT)
    runner_hash = file_sha(RUNNER)
    if (source.get("source_preparation_sha256") != canon_sha(source_body)
            or source.get("source_script_sha256") != runner_hash):
        raise QAError("R4 source preparation self-hash or runner hash mismatch")
    approval = check_signature(FINAL_PASS, "approval_sha256")
    if (approval.get("status") != "PASS" or approval.get("execute_final_pipeline") is not True
            or approval.get("source_report_sha256") != source_hash
            or approval.get("source_preparation_sha256") != source["source_preparation_sha256"]
            or approval.get("source_script_sha256") != runner_hash
            or approval.get("scope", {}).get("fit_ids") != list(FIT_IDS)):
        raise QAError("Final source pass does not bind the exact R4 source")

    parent_manifest = read_json(PARENT_MANIFEST)
    parent_hash = file_sha(PARENT)
    pos_freeze = read_json(ROOT / "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1.json")
    final_manifest_path = RUN / "FINAL_MANIFEST_R1.json"
    final_manifest = read_json(final_manifest_path)
    manifest_body = {k: v for k, v in final_manifest.items() if k != "manifest_sha256"}
    if (final_manifest.get("manifest_sha256") != canon_sha(manifest_body)
            or final_manifest.get("namespace") != NAMESPACE
            or final_manifest.get("source_preparation_sha256") != source["source_preparation_sha256"]
            or final_manifest.get("final_source_pass_sha256") != file_sha(FINAL_PASS)
            or final_manifest.get("status") != "CSV_READY_PENDING_INDEPENDENT_QA"):
        raise QAError("Final execution manifest self-hash or source binding differs")
    if (final_manifest.get("fit_ids") != list(FIT_IDS)
            or final_manifest.get("predictor_fits_started") != 2
            or final_manifest.get("preprocessing_fits") != 0
            or final_manifest.get("test_labels_loaded") is not False
            or final_manifest.get("sample_labels_loaded") is not False
            or final_manifest.get("upload_performed") is not False
            or final_manifest.get("champion_parity_passed_before_candidate_csv") is not True):
        raise QAError("Fit accounting, label boundary, or pre-CSV champion gate differs")

    ledger_path = RUN / "fit_ledger.jsonl"
    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saves = [e for e in events if e.get("event") == "FIT_SAVED"]
    failures = [e for e in events if e.get("event") == "FIT_FAILED"]
    if ([e.get("fit_id") for e in starts] != list(FIT_IDS)
            or [e.get("fit_id") for e in saves] != list(FIT_IDS) or failures):
        raise QAError("Append-only final fit ledger differs from the authorized two-fit order")
    fit_evidence = {}
    train_ids_hash = None
    design_hash = None
    for fit_id in FIT_IDS:
        receipt_path = RUN / "units" / f"{fit_id}.json"
        receipt = check_signature(receipt_path, "receipt_sha256")
        artifact_path = RUN / "units" / f"{fit_id}.joblib"
        artifact_hash = file_sha(artifact_path)
        ledger_save = next(e for e in saves if e["fit_id"] == fit_id)
        if (receipt.get("namespace") != NAMESPACE or receipt.get("fit_id") != fit_id
                or receipt.get("synthetic") is not False
                or receipt.get("preprocessing_fit_calls") != 0
                or receipt.get("underlying_fit_calls") != ["LGBMRegressor.fit x1 with sample_weight"]
                or receipt.get("parameters") != pos_freeze.get("resolved_parameters")
                or receipt.get("seed") != (17 if fit_id.endswith("P17") else 137)
                or receipt.get("artifact_sha256") != artifact_hash
                or ledger_save.get("artifact_sha256") != artifact_hash
                or ledger_save.get("receipt_sha256") != file_sha(receipt_path)):
            raise QAError(f"Fit artifact/receipt/ledger mismatch: {fit_id}")
        if train_ids_hash is None:
            train_ids_hash = receipt["train_ids_sha256"]
            design_hash = receipt["fit_design_sha256"]
        if (receipt.get("train_rows") != 50608 or receipt.get("train_ids_sha256") != train_ids_hash
                or receipt.get("fit_design_sha256") != design_hash
                or receipt.get("fit_target_sha256") != starts[0].get("target_sha256")):
            raise QAError(f"Fit row/design/target identity differs: {fit_id}")
        if not np.isclose(receipt.get("sample_weight_sum", float("nan")), 50608, rtol=0, atol=1e-8):
            raise QAError(f"Poisson sample weights are not normalized for {fit_id}")
        fit_evidence[fit_id] = {
            "artifact_sha256": artifact_hash,
            "receipt_sha256": file_sha(receipt_path),
            "receipt_signature": receipt["receipt_sha256"],
            "seed": receipt["seed"],
            "train_rows": receipt["train_rows"],
            "fit_design_sha256": receipt["fit_design_sha256"],
            "fit_target_sha256": receipt["fit_target_sha256"],
            "poisson_counts_sha256": receipt["poisson_counts_sha256"],
            "sample_weight_sha256": receipt["sample_weight_sha256"],
            "sample_weight_sum": receipt["sample_weight_sum"],
            "preprocessing_fit_calls": receipt["preprocessing_fit_calls"],
        }

    prediction_path = RUN / "predictions" / "E79.npz"
    prediction_receipt_path = RUN / "predictions" / "E79.json"
    prediction_sha = file_sha(prediction_path)
    pred_receipt = check_signature(prediction_receipt_path, "receipt_sha256")
    if (pred_receipt.get("prediction_sha256") != prediction_sha
            or pred_receipt.get("fit_ids") != list(FIT_IDS)
            or pred_receipt.get("fit_units_started") != 2
            or pred_receipt.get("fit_units_saved") != 2
            or pred_receipt.get("test_labels_loaded") is not False):
        raise QAError("Prediction archive receipt differs")
    with np.load(prediction_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    expected_members = {"record_id", "cycle_code", "missing", "baseline", "original_raw",
                        "replica17_raw", "replica137_raw", "mean_raw", "mapped_t50",
                        "direct_prediction", "final_prediction"}
    if set(arrays) != expected_members or set(pred_receipt.get("members", {})) != expected_members:
        raise QAError("Prediction archive member inventory differs")
    for name, array in arrays.items():
        if pred_receipt["members"][name] != array_sha(array):
            raise QAError(f"Prediction archive member digest mismatch: {name}")
    if len(arrays["record_id"]) != 29421 or len(np.unique(arrays["record_id"].astype(str))) != 29421:
        raise QAError("E79 output row count or record ID deduplication failed")

    with np.load(PARENT, allow_pickle=False) as saved:
        parent = {key: saved[key] for key in saved.files}
    if parent_hash != final_manifest.get("parent_archive_sha256") or parent_hash != parent_manifest.get("raw_prediction_archive_sha256"):
        raise QAError("Parent R06 B archive hash differs from the final manifest")
    ids = arrays["record_id"].astype(str)
    cycles = arrays["cycle_code"].astype(np.int64)
    missing = arrays["missing"].astype(bool)
    baseline = arrays["baseline"].astype(np.float64)
    original_raw = arrays["original_raw"].astype(np.float64)
    rep17 = arrays["replica17_raw"].astype(np.float64)
    rep137 = arrays["replica137_raw"].astype(np.float64)
    if (not np.array_equal(ids, parent["record_id"].astype(str))
            or not np.array_equal(cycles, parent["cycle_code"])
            or not np.array_equal(missing, parent["missing"].astype(bool))
            or not np.array_equal(baseline, parent["baseline"].astype(np.float64))
            or not np.array_equal(original_raw, parent["raw"].astype(np.float64))):
        raise QAError("Frozen raw array row/mask/baseline/original POS replay differs")
    r06_baseline = pd.read_csv(R06_BASELINE_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    r06_ids = r06_baseline["record_id"].astype(str).to_numpy(dtype=str)
    r06_values = pd.to_numeric(r06_baseline["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    exact(r06_ids, ids, "frozen R06 four-learner baseline ID order")
    exact(r06_values, baseline, "frozen R06 four-learner baseline values")
    if file_sha(R06_BASELINE_CSV) != parent_manifest.get("baseline_csv_sha256"):
        raise QAError("Frozen R06 baseline CSV hash differs from parent manifest")
    if parent_bytes_sha(original_raw) != parent_manifest.get("raw_score_sha256"):
        raise QAError("Parent producer bytes-only raw digest does not match")

    original_t50 = 0.5 * original_raw + 0.5 * rank50(original_raw, cycles)
    original_direct = np.clip(0.75 * baseline + 0.25 * original_t50, 0.0, 100.0)
    original_direct[missing] = baseline[missing]
    exact(original_direct, parent["prediction"].astype(np.float64), "original B DIRECT replay")
    old_champion = apply_hinge(original_direct, missing)

    mean_raw = (original_raw + rep17 + rep137) / 3.0
    mapped_t50 = 0.5 * mean_raw + 0.5 * rank50(mean_raw, cycles)
    candidate_direct = np.clip(0.75 * baseline + 0.25 * mapped_t50, 0.0, 100.0)
    candidate_direct[missing] = baseline[missing]
    candidate_final = apply_hinge(candidate_direct, missing)
    exact(mean_raw, arrays["mean_raw"], "3-member mean raw"); exact(mapped_t50, arrays["mapped_t50"], "T50 map")
    exact(candidate_direct, arrays["direct_prediction"], "candidate direct blend")
    exact(candidate_final, arrays["final_prediction"], "final hinge application")
    exact(candidate_final[missing].view("<u8"), baseline[missing].view("<u8"), "missing fallback bits")
    if not np.isfinite(candidate_final).all() or np.any((candidate_final < 0) | (candidate_final > 100)):
        raise QAError("Final predictions are non-finite or outside [0,100]")

    test = pd.read_csv(TEST_CSV, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    sample = pd.read_csv(SAMPLE_CSV, usecols=["record_id"], dtype={"record_id": "string"})
    test_ids = test["record_id"].astype(str).to_numpy(dtype=str)
    sample_ids = sample["record_id"].astype(str).to_numpy(dtype=str)
    test_cycles = pd.to_numeric(test["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    exact(ids, test_ids, "Test feature-only ID order")
    exact(ids, sample_ids, "SampleSubmission ID order")
    exact(cycles, test_cycles, "Test feature-only cycle order")
    cycle_counts = {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)}
    if cycle_counts != parent_manifest.get("test_cycle_counts"):
        raise QAError("Cycle history counts differ from frozen R06 parent")

    candidate_ids = None
    candidate_values = None
    with_candidate = pd.read_csv(FINAL_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    candidate_ids = with_candidate["record_id"].astype(str).to_numpy(dtype=str)
    candidate_values = pd.to_numeric(with_candidate["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    champion = pd.read_csv(CHAMPION_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    champion_ids = champion["record_id"].astype(str).to_numpy(dtype=str)
    champion_values = pd.to_numeric(champion["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    exact(candidate_ids, ids, "candidate CSV row ID order")
    exact(candidate_values, candidate_final, "candidate CSV values")
    exact(champion_ids, ids, "frozen champion CSV row ID order")
    exact(old_champion, champion_values, "original champion final hinge parity")
    if (file_sha(FINAL_CSV) != final_manifest.get("csv_sha256")
            or final_manifest.get("csv_path") != FINAL_CSV.relative_to(ROOT).as_posix()
            or len(candidate_values) != 29421):
        raise QAError("Candidate CSV hash or row count differs from final manifest")
    changed = candidate_values != champion_values
    delta = candidate_values - champion_values

    body = {
        "schema_version": 1,
        "namespace": NAMESPACE,
        "status": "PASS_FINAL_ARTIFACT_QA",
        "source_report_sha256": source_hash,
        "source_preparation_sha256": source["source_preparation_sha256"],
        "source_script_sha256": runner_hash,
        "final_source_pass_sha256": file_sha(FINAL_PASS),
        "final_source_pass_approval_sha256": approval["approval_sha256"],
        "final_manifest_path": final_manifest_path.relative_to(ROOT).as_posix(),
        "final_manifest_sha256": file_sha(final_manifest_path),
        "parent_manifest_sha256": file_sha(PARENT_MANIFEST),
        "parent_archive_sha256": parent_hash,
        "parent_raw_bytes_sha256": parent_bytes_sha(original_raw),
        "prediction_archive_sha256": prediction_sha,
        "prediction_receipt_sha256": file_sha(prediction_receipt_path),
        "fit_ledger_sha256": file_sha(ledger_path),
        "fit_ids": list(FIT_IDS),
        "fit_units_started": len(starts),
        "fit_units_saved": len(saves),
        "fit_units_failed": len(failures),
        "preprocessing_fits": 0,
        "fit_artifacts": fit_evidence,
        "rows": len(ids),
        "unique_record_ids": len(np.unique(ids)),
        "record_ids_sha256": array_sha(ids),
        "cycle_counts": cycle_counts,
        "missing_rows": int(missing.sum()),
        "missing_fallback_bit_exact": True,
        "original_raw_replay_bit_exact": True,
        "original_direct_replay_bit_exact": True,
        "original_final_hinge_champion_parity_bit_exact": True,
        "candidate_prediction_readback_bit_exact": True,
        "candidate_csv_round_trip_bit_exact": True,
        "candidate_csv_path": FINAL_CSV.relative_to(ROOT).as_posix(),
        "candidate_csv_sha256": file_sha(FINAL_CSV),
        "candidate_csv_values_sha256": array_sha(candidate_values, dtype="<f8"),
        "frozen_champion_csv_sha256": file_sha(CHAMPION_CSV),
        "changed_rows_vs_frozen_champion": int(changed.sum()),
        "mean_absolute_prediction_change_vs_champion": float(np.mean(np.abs(delta))),
        "maximum_absolute_prediction_change_vs_champion": float(np.max(np.abs(delta))),
        "test_feature_identity_columns_read": True,
        "test_labels_loaded": False,
        "kaggle_accessed": False,
        "upload_performed": False,
    }
    body["qa_sha256"] = canon_sha(body)
    if QA_REPORT.exists():
        if QA_REPORT.read_bytes() != json.dumps(body, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False, indent=2).encode("utf-8") + b"\n":
            raise QAError("QA report exists with different bytes; no overwrite")
    else:
        QA_REPORT.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             indent=2).encode("utf-8") + b"\n"
        with QA_REPORT.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return {"status": body["status"], "qa_report": QA_REPORT.relative_to(ROOT).as_posix(),
            "qa_sha256": body["qa_sha256"], "qa_report_sha256": file_sha(QA_REPORT),
            "rows": len(ids), "changed_rows": int(changed.sum()),
            "missing_rows": int(missing.sum()), "fit_units_started": len(starts),
            "fit_units_saved": len(saves), "test_labels_loaded": False,
            "upload_performed": False}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
