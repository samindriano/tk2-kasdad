"""Independent replay verifier for the frozen REG E79 B-only artifact."""
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_ROWS = 29421
N_MISSING = 1062
FREEZE = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/B_ONLY_PREFIT_FREEZE.json"
CSV_OUT = ROOT / "submissions/reg_final_wave_followup_v2_b_only/global_present_only_e79.csv"
MANIFEST = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/INFERENCE_MANIFEST_B_ONLY.json"
QA_OUT = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/INDEPENDENT_QA.json"
GLOBAL_CSV = ROOT / "submissions/reg_final_wave_v2/current_direct_affine_l1_e79.csv"
DIRECT_CSV = ROOT / "submissions/reg_current_r06_e79_exploratory_v1/direct_position_real_e79.csv"
DIRECT_ARCHIVE = ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/inference_B.npz"
GLOBAL_MANIFEST = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/INFERENCE_MANIFEST_V2.json"
DIRECT_MANIFEST = ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/INFERENCE_MANIFEST_B.json"
APPLICATION_SOURCE = ROOT / "scripts/reg_final_wave_followup_v2_b_only.py"
VERIFIER_SOURCE = ROOT / "scripts/verify_reg_final_wave_followup_v2_b_only.py"
SOURCE_FILES = {
    "application_source": APPLICATION_SOURCE,
    "independent_verifier": VERIFIER_SOURCE,
    "tests": ROOT / "tests/test_reg_final_wave_followup_v2_b_only.py",
    "protocol": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/PROTOCOL.md",
    "previous_synthetic_qa": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/SYNTHETIC_QA.json",
    "synthetic_qa": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/SYNTHETIC_QA_R2.json",
}


class VerificationError(RuntimeError):
    """The independent output replay failed."""


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


def sha_ids(ids: list[str]) -> str:
    return hashlib.sha256(canonical_bytes(ids)).hexdigest()


def sha_predictions(values: np.ndarray) -> str:
    vector = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256(canonical_bytes({"shape": vector.shape, "dtype": str(vector.dtype)}))
    digest.update(vector.tobytes(order="C"))
    return digest.hexdigest()


def sha_raw_f64(values: np.ndarray) -> str:
    vector = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(vector.tobytes(order="C")).hexdigest()


def sha_missing(mask: np.ndarray) -> str:
    vector = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    digest = hashlib.sha256(canonical_bytes({"shape": vector.shape, "dtype": "bool"}))
    digest.update(vector.astype(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def same_bits(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.ascontiguousarray(np.asarray(left, dtype="<f8"))
    b = np.ascontiguousarray(np.asarray(right, dtype="<f8"))
    return a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def verify_row_aligned_output(output_ids: list[str], archive_ids: list[str],
                              output_values: np.ndarray, expected: np.ndarray,
                              missing: np.ndarray, global_values: np.ndarray,
                              direct_values: np.ndarray) -> None:
    if output_ids != archive_ids:
        raise VerificationError("Output ID order differs from the saved archive")
    if (not same_bits(output_values, expected)
            or not same_bits(output_values[~missing], global_values[~missing])
            or not same_bits(output_values[missing], direct_values[missing])):
        raise VerificationError("Output values differ from the ID-aligned frozen formula")


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    import pandas as pd
    frame = pd.read_csv(path, dtype={"record_id": str}, float_precision="round_trip")
    if list(frame.columns) != ["record_id", "threat_rank"]:
        raise VerificationError(f"Unexpected columns for {path}: {list(frame.columns)}")
    ids = frame["record_id"].astype(str).tolist()
    values = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if len(set(ids)) != len(ids):
        raise VerificationError(f"Duplicate IDs in {path}")
    return ids, values


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise VerificationError(f"Refusing to overwrite QA receipt: {path}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def verify() -> None:
    freeze = read_json(FREEZE)
    freeze_body = {k: v for k, v in freeze.items()
                   if k != "signature" and k != "signature_method"}
    if (freeze.get("status") != "SOURCE_AND_INPUTS_FROZEN_BEFORE_DATA_LOAD"
            or freeze.get("signature") != canonical_sha(freeze_body)):
        raise VerificationError("B-only source/input freeze signature failed")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                            text=True, stdout=subprocess.PIPE).stdout.strip()
    try:
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                        freeze["source_commit"], commit], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (KeyError, subprocess.CalledProcessError) as exc:
        raise VerificationError("Current HEAD is not descended from the frozen source commit") from exc
    if set(freeze.get("source_hashes", {})) != set(SOURCE_FILES):
        raise VerificationError("Freeze source inventory is incomplete or changed")
    for name, path in SOURCE_FILES.items():
        if sha_file(path) != freeze["source_hashes"].get(name):
            raise VerificationError(f"Frozen code hash changed: {name}")
    for name, entry in freeze.get("input_hashes", {}).items():
        path = Path(entry["path"])
        if not path.is_file() or sha_file(path) != entry["sha256"]:
            raise VerificationError(f"Frozen input changed: {name}")

    manifest = read_json(MANIFEST)
    manifest_body = {k: v for k, v in manifest.items()
                     if k != "signature" and k != "signature_method"}
    if manifest.get("signature") != canonical_sha(manifest_body):
        raise VerificationError("B-only output manifest signature failed")
    global_manifest = read_json(GLOBAL_MANIFEST)
    direct_manifest = read_json(DIRECT_MANIFEST)
    for label, value in (("global", global_manifest), ("direct", direct_manifest)):
        body = {k: v for k, v in value.items()
                if k != "signature" and k != "signature_method"}
        if value.get("signature") != canonical_sha(body):
            raise VerificationError(f"Frozen {label} input manifest signature failed")
    if (manifest.get("source_commit") != freeze.get("source_commit")
            or manifest.get("freeze_sha256") != sha_file(FREEZE)
            or manifest.get("csv_sha256") != sha_file(CSV_OUT)
            or manifest.get("status") != "B_ONLY_ARTIFACT_READY_PENDING_INDEPENDENT_QA"):
        raise VerificationError("B-only output manifest does not bind the frozen artifact")

    global_ids, global_values = load_csv(GLOBAL_CSV)
    direct_ids, direct_values = load_csv(DIRECT_CSV)
    output_ids, output_values = load_csv(CSV_OUT)
    with np.load(DIRECT_ARCHIVE, allow_pickle=False) as archive:
        archive_ids = np.asarray(archive["record_id"], dtype=str).tolist()
        archive_direct = np.asarray(archive["prediction"], dtype=np.float64)
        archive_baseline = np.asarray(archive["baseline"], dtype=np.float64)
        missing = np.asarray(archive["missing"], dtype=np.bool_)
    if (len(output_ids) != N_ROWS or len(set(output_ids)) != N_ROWS
            or len(global_ids) != N_ROWS or len(set(global_ids)) != N_ROWS
            or len(direct_ids) != N_ROWS or len(set(direct_ids)) != N_ROWS
            or len(archive_ids) != N_ROWS or len(set(archive_ids)) != N_ROWS
            or int(missing.sum()) != N_MISSING
            or output_ids != archive_ids or global_ids != archive_ids or direct_ids != archive_ids
            or not np.isfinite(global_values).all() or not np.isfinite(direct_values).all()
            or not np.isfinite(output_values).all()
            or np.any((output_values < 0) | (output_values > 100))
            or not same_bits(direct_values, archive_direct)
            or not same_bits(archive_direct[missing], archive_baseline[missing])
            or sha_predictions(global_values) != global_manifest.get("prediction_sha256")
            or sha_raw_f64(direct_values) != direct_manifest.get("prediction_sha256")):
        raise VerificationError("Saved CSV/archive row identity, range, or missing route failed")

    # Independently rebuild each output value by its ID, not by assuming file offsets.
    global_map = {record_id: float(value) for record_id, value in zip(global_ids, global_values, strict=True)}
    direct_map = {record_id: float(value) for record_id, value in zip(direct_ids, direct_values, strict=True)}
    expected = np.asarray([
        direct_map[record_id] if missing[index] else global_map[record_id]
        for index, record_id in enumerate(archive_ids)
    ], dtype=np.float64)
    verify_row_aligned_output(output_ids, archive_ids, output_values, expected,
                              missing, global_values, direct_values)
    if same_bits(output_values, global_values) or same_bits(output_values, direct_values):
        raise VerificationError("B-only output duplicates an existing full candidate vector")

    changed_global = int(np.count_nonzero(
        np.asarray(output_values, dtype="<f8").view("<u8") !=
        np.asarray(global_values, dtype="<f8").view("<u8")))
    changed_direct = int(np.count_nonzero(
        np.asarray(output_values, dtype="<f8").view("<u8") !=
        np.asarray(direct_values, dtype="<f8").view("<u8")))
    expected_global_id_hash = freeze["expected_output"]["ordered_ids_sha256"]
    if (sha_ids(output_ids) != expected_global_id_hash
            or manifest.get("input_hashes") !=
            {name: entry["sha256"] for name, entry in freeze["input_hashes"].items()}
            or manifest.get("ordered_ids_sha256") != expected_global_id_hash
            or manifest.get("prediction_sha256") != sha_predictions(output_values)
            or manifest.get("missing_rows") != N_MISSING
            or manifest.get("missing_mask_sha256") != sha_missing(missing)
            or manifest.get("current_best_comparison", {}).get("changed_rows_vs_global") != changed_global
            or manifest.get("current_best_comparison", {}).get("changed_rows_vs_direct") != changed_direct
            or not np.array_equal(np.asarray(manifest.get("range"), dtype=np.float64),
                                  np.asarray([output_values.min(), output_values.max()]))):
        raise VerificationError("Output manifest counts, range, or ordered identity differ from replay")

    qa_body = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_B_ONLY_REPLAY_QA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(CSV_OUT),
        "csv_sha256": sha_file(CSV_OUT),
        "manifest_sha256": sha_file(MANIFEST),
        "freeze_sha256": sha_file(FREEZE),
        "source_commit": commit,
        "rows": len(output_ids),
        "columns": ["record_id", "threat_rank"],
        "ordered_ids_sha256": sha_ids(output_ids),
        "prediction_sha256": sha_predictions(output_values),
        "range": [float(output_values.min()), float(output_values.max())],
        "missing_rows": int(missing.sum()),
        "missing_mask_sha256": sha_missing(missing),
        "present_global_bit_exact": True,
        "missing_direct_bit_exact": True,
        "row_alignment_rebuilt_by_id": True,
        "full_vector_duplicate_detected": False,
        "changed_rows_vs_global": changed_global,
        "changed_rows_vs_direct": changed_direct,
        "test_metrics_computed": False,
        "train_targets_read": False,
        "test_or_sample_read": False,
        "upload_performed": False,
    }
    digest = write_once(QA_OUT, {**qa_body, "signature": canonical_sha(qa_body),
                                 "signature_method": "SHA-256 canonical JSON integrity seal"})
    print(json.dumps({"status": qa_body["status"], "qa": str(QA_OUT),
                      "qa_sha256": digest, "csv_sha256": qa_body["csv_sha256"],
                      "rows": len(output_ids), "missing_rows": int(missing.sum()),
                      "changed_rows_vs_global": changed_global,
                      "changed_rows_vs_direct": changed_direct,
                      "range": qa_body["range"], "test_metrics_computed": False,
                      "upload_performed": False}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    try:
        verify()
    except (VerificationError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_B_ONLY_REPLAY_QA", "error": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
