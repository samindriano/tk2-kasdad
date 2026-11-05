"""Independent row, formula, and source-pin verification for REG R1 CSVs."""
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "reg_simplification_benchmark_v1"
EXPERIMENT_DIR = ROOT / "experiments" / NAMESPACE
REPORT_DIR = ROOT / "reports" / "optimization_v3" / NAMESPACE
MANIFEST_PATH = EXPERIMENT_DIR / "INPUT_MANIFEST_R1.json"
BUILD_RECEIPT = REPORT_DIR / "BUILD_RECEIPT_R1.json"
EXPECTED_ROWS = 29_421
EXPECTED_MISSING = 1_062
KNOT = 50.0
HINGE = {
    "intercept": 1.7288079036674198,
    "slope_low": 0.9933285599956824,
    "slope_high": 0.957778072288735,
}
OUTPUTS = {
    "NO_POS": ROOT / "submissions" / NAMESPACE / "reg_no_pos_hinge_e79.csv",
    "NO_CHAR": ROOT / "submissions" / NAMESPACE / "reg_no_char_hinge_e79.csv",
}


class VerificationError(RuntimeError):
    """Fail-closed input, source, CSV, or recipe verification condition."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_array_sha(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    header = json.dumps({"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True,
                        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(header)
    if array.dtype.kind in "OUS":
        values = json.dumps(array.astype(str).tolist(), sort_keys=True, ensure_ascii=False,
                            allow_nan=False, separators=(",", ":")).encode("utf-8")
        digest.update(values)
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def _verify_signature(document: dict[str, Any]) -> None:
    body = {key: value for key, value in document.items()
            if key not in {"signature", "signature_method"}}
    if document.get("signature") != _canonical_sha(body):
        raise VerificationError("Signed source receipt has a canonical SHA mismatch")


def _manifest_inputs() -> dict[str, Path]:
    manifest = _read_json(MANIFEST_PATH)
    if manifest.get("namespace") != NAMESPACE or manifest.get("zero_new_fits") is not True:
        raise VerificationError("Input manifest namespace or zero-fit policy differs")
    records = manifest.get("inputs")
    if not isinstance(records, dict) or not records:
        raise VerificationError("Input manifest has no pinned inputs")
    paths: dict[str, Path] = {}
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise VerificationError(f"Malformed input pin: {name}")
        path = Path(record["path"]).resolve()
        if not path.is_file() or _sha(path) != record.get("sha256"):
            raise VerificationError(f"Input pin failed: {name}")
        paths[name] = path
    required = {"current_r06_archive", "r06_archive", "sample_submission", "r06_manifest",
                "current_r06_manifest", "final_hinge_report"}
    if not required.issubset(paths):
        raise VerificationError("Required saved input pins are absent")
    return paths


def _verify_source_commit() -> tuple[str, dict[str, str]]:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise VerificationError("Cannot resolve implementation source commit")
    commit = result.stdout.strip()
    required = [
        Path(__file__).resolve(),
        ROOT / "scripts/reg_simplification_benchmark_v1.py",
        ROOT / "tests/test_reg_simplification_benchmark_v1.py",
        REPORT_DIR / "PROTOCOL_R1.md",
        REPORT_DIR / "EXECUTION_POLICY_R1.json",
        MANIFEST_PATH,
    ]
    hashes: dict[str, str] = {}
    for path in required:
        relative = path.resolve().relative_to(ROOT).as_posix()
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                                   capture_output=True, check=False)
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != _sha(path):
            raise VerificationError(f"Source bytes are not committed unchanged: {relative}")
        hashes[relative] = _sha(path)
    return commit, hashes


def _load_saved_vectors(paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(paths["r06_archive"], allow_pickle=False) as r06_archive:
            if set(r06_archive.files) != {"champion", "raw_candidate", "rank_mapped_candidate",
                                          "cycle_code", "prediction"}:
                raise VerificationError("Pinned R06 NPZ member inventory differs")
            legacy_three = np.asarray(r06_archive["champion"], dtype=np.float64)
            character_rank50 = np.asarray(r06_archive["rank_mapped_candidate"], dtype=np.float64)
            r06_prediction = np.asarray(r06_archive["prediction"], dtype=np.float64)
            r06_cycles = np.asarray(r06_archive["cycle_code"])
        with np.load(paths["current_r06_archive"], allow_pickle=False) as current_archive:
            if set(current_archive.files) != {"record_id", "raw", "cycle_code", "missing",
                                              "baseline", "prediction"}:
                raise VerificationError("Pinned E79 NPZ member inventory differs")
            source_ids = np.asarray(current_archive["record_id"])
            ids = source_ids.astype(str)
            raw_pos = np.asarray(current_archive["raw"], dtype=np.float64)
            cycles = np.asarray(current_archive["cycle_code"])
            missing = np.asarray(current_archive["missing"], dtype=bool)
            baseline = np.asarray(current_archive["baseline"], dtype=np.float64)
            current_prediction = np.asarray(current_archive["prediction"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise VerificationError("Cannot load allowlisted saved predictions") from exc

    vectors = (ids, raw_pos, cycles, missing, legacy_three, character_rank50, r06_prediction,
               r06_cycles, baseline, current_prediction)
    if any(value.ndim != 1 or len(value) != EXPECTED_ROWS for value in vectors):
        raise VerificationError("A saved vector has an unexpected shape")
    if len(set(ids.tolist())) != EXPECTED_ROWS or int(missing.sum()) != EXPECTED_MISSING:
        raise VerificationError("Saved IDs or missing-row count differs")
    if not np.array_equal(cycles, r06_cycles):
        raise VerificationError("R06 and E79 cycle order differs")
    if not np.isfinite(raw_pos).all() or not np.isfinite(legacy_three).all():
        raise VerificationError("Saved component vector is nonfinite")
    if not np.array_equal(r06_prediction.view("<u8"), baseline.view("<u8")):
        raise VerificationError("E79 baseline is not bit-exact to saved final R06")
    r06_rebuilt = np.clip(0.75 * legacy_three + 0.25 * character_rank50, 0.0, 100.0)
    if not np.allclose(r06_rebuilt, r06_prediction, rtol=0.0, atol=1e-10):
        raise VerificationError("Saved R06 full-champion blend does not reconstruct")

    sample_ids: list[str] = []
    with paths["sample_submission"].open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["record_id", "threat_rank"]:
            raise VerificationError("Official SampleSubmission header differs")
        for row in reader:
            if len(row) != 2:
                raise VerificationError("Official SampleSubmission has malformed row")
            sample_ids.append(row[0])
    if len(sample_ids) != EXPECTED_ROWS or sample_ids != ids.tolist():
        raise VerificationError("Saved E79 IDs differ from official sample order")

    hinge_doc = _read_json(paths["final_hinge_report"])
    _verify_signature(hinge_doc)
    fit = hinge_doc.get("final_slot", {})
    if (fit.get("fit_id") != "HINGE_FINAL"
            or fit.get("coefficients") != HINGE
            or hinge_doc.get("test_labels_read") is not False
            or hinge_doc.get("kaggle_upload") is not False):
        raise VerificationError("Fixed final HINGE source receipt differs")
    current_manifest = _read_json(paths["current_r06_manifest"])
    _verify_signature(current_manifest)
    if (current_manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or current_manifest.get("test_labels_loaded") is not False
            or current_manifest.get("upload_performed") is not False
            or current_manifest.get("baseline_member_used") != "manifest-bound final mapped R06 prediction"
            or current_manifest.get("baseline_raw_candidate_member_used") is not False):
        raise VerificationError("E79 inference manifest identity/boundary differs")
    r06_manifest = _read_json(paths["r06_manifest"])
    if (r06_manifest.get("candidate") != "R06_ORIGINAL"
            or r06_manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or r06_manifest.get("recipe", {}).get("blend")
            != "clip(0.75*three-component champion + 0.25*R06 rank50, 0, 100)"
            or source_array_sha(r06_prediction, dtype="<f8")
            != r06_manifest.get("prediction_audit", {}).get("prediction_sha256")
            or source_array_sha(baseline, dtype="<f8")
            != current_manifest.get("baseline_prediction_member_digest")
            or source_array_sha(current_prediction, dtype="<f8")
            != current_manifest.get("prediction_sha256")
            or source_array_sha(source_ids)
            != current_manifest.get("ordered_official_test_ids_sha256")):
        raise VerificationError("Saved prediction/ID digests differ from source manifests")

    no_pos = _hinge_present(r06_prediction, missing)
    position_rank50 = np.empty_like(raw_pos)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentiles = 100.0 * rankdata(raw_pos[mask], method="average") / int(mask.sum())
        position_rank50[mask] = 0.5 * raw_pos[mask] + 0.5 * percentiles
    no_char_raw = legacy_three.copy()
    present = ~missing
    no_char_raw[present] = np.clip(
        (9.0 / 13.0) * legacy_three[present] + (4.0 / 13.0) * position_rank50[present],
        0.0, 100.0)
    no_char = _hinge_present(no_char_raw, missing)
    if (not np.array_equal(no_pos[missing].view("<u8"), r06_prediction[missing].view("<u8"))
            or not np.array_equal(no_char[missing].view("<u8"), legacy_three[missing].view("<u8"))):
        raise VerificationError("Candidate missing-row route is not bit-exact")
    if not np.isfinite(no_pos).all() or not np.isfinite(no_char).all():
        raise VerificationError("Recomputed candidate vector is nonfinite")
    return {"ids": ids, "no_pos": no_pos, "no_char": no_char}, {
        "r06_prediction_max_abs_reconstruction_error": float(np.max(np.abs(r06_rebuilt-r06_prediction))),
        "missing_rows": int(missing.sum()),
        "present_rows": int(present.sum()),
        "cycle_count": int(len(np.unique(cycles))),
        "official_sample_order_exact": True,
        "saved_r06_baseline_bit_exact": True,
        "test_labels_read": False,
        "new_fits": 0,
    }


def _hinge_present(raw: np.ndarray, missing: np.ndarray) -> np.ndarray:
    output = raw.copy()
    present = ~missing
    x = raw[present]
    output[present] = np.clip(
        HINGE["intercept"] + HINGE["slope_low"] * np.minimum(x, KNOT)
        + HINGE["slope_high"] * np.maximum(x - KNOT, 0.0),
        0.0, 100.0)
    return output


def _read_output(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    ids: list[str] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["record_id", "threat_rank"]:
            raise VerificationError(f"Output CSV header differs: {path}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise VerificationError(f"Malformed output CSV row {row_number}: {path}")
            try:
                value = float(row[1])
            except ValueError as exc:
                raise VerificationError(f"Non-numeric prediction at row {row_number}") from exc
            if not np.isfinite(value) or value < 0.0 or value > 100.0:
                raise VerificationError(f"Prediction out of range at row {row_number}")
            if format(value, ".17g") != row[1]:
                raise VerificationError(f"Prediction is not in canonical round-trip form at row {row_number}")
            ids.append(row[0])
            values.append(value)
    if len(ids) != EXPECTED_ROWS or ids != expected_ids.tolist():
        raise VerificationError(f"Output rows/order differ from official sample: {path}")
    if len(set(ids)) != EXPECTED_ROWS:
        raise VerificationError(f"Duplicate output record_id: {path}")
    return np.asarray(values, dtype=np.float64)


def verify(candidate: str) -> dict[str, Any]:
    if candidate not in OUTPUTS:
        raise VerificationError(f"Unknown candidate: {candidate}")
    paths = _manifest_inputs()
    source_commit, source_hashes = _verify_source_commit()
    build = _read_json(BUILD_RECEIPT)
    if (build.get("status") != "PASS_REG_SIMPLIFICATION_E79_BUILD"
            or build.get("test_labels_read") is not False
            or build.get("new_predictor_fits") != 0
            or build.get("new_preprocessing_fits") != 0
            or build.get("new_calibrator_fits") != 0):
        raise VerificationError("Runner build receipt is missing or outside the frozen scope")
    vectors, recomputation = _load_saved_vectors(paths)
    key = "no_pos" if candidate == "NO_POS" else "no_char"
    path = OUTPUTS[candidate]
    actual = _read_output(path, vectors["ids"])
    expected = vectors[key]
    if not np.array_equal(actual.view("<u8"), expected.view("<u8")):
        raise VerificationError(f"CSV values differ from independent recipe recomputation: {candidate}")
    file_sha = _sha(path)
    recorded = build.get("candidates", {}).get(candidate, {})
    if file_sha != recorded.get("sha256") or len(actual) != recorded.get("rows"):
        raise VerificationError("Build receipt does not match independently verified CSV")
    receipt = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_REG_SIMPLIFICATION_CSV_QA",
        "candidate": candidate,
        "csv_path": str(path.resolve()),
        "csv_sha256": file_sha,
        "rows": int(len(actual)),
        "columns": ["record_id", "threat_rank"],
        "exact_prediction_vector_match": True,
        "max_abs_prediction_difference": 0.0,
        "independent_recipe_recomputation": True,
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "input_hashes": {name: _sha(value) for name, value in paths.items()},
        "recomputation": recomputation,
        "test_labels_read": False,
        "new_fits": 0,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = REPORT_DIR / f"INDEPENDENT_VERIFICATION_{candidate}_R1.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with receipt_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise VerificationError(f"Refusing to overwrite independent receipt: {receipt_path}") from exc
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=tuple(OUTPUTS))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.candidate), indent=2))
        return 0
    except VerificationError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
