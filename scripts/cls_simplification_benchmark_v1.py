"""Build one frozen NO_POS Classification E79 CSV from verified cached A02 pA.

No estimator or preprocessing fit is performed here. The E79 probability archive,
Test IDs/availability, and SampleSubmission schema are opened only after the
committed pre-access freeze and the per-lane resource/idempotency lock pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/cls_simplification_benchmark_v1"
FREEZE_PATH = REPORT / "PRE_ACCESS_FREEZE.json"
E79_NPZ = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
E79_MANIFEST = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_MANIFEST.json"
E79_QA = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_QA.json"
E79_APPLICATION_FREEZE = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_APPLICATION_FREEZE_V2.json"
TEST_CSV = ROOT / "data/raw/classification/Test.csv"
SAMPLE_CSV = ROOT / "data/raw/classification/SampleSubmission.csv"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_simplification_benchmark_v1/classification_no_pos_e79.csv"
RUN_MANIFEST = REPORT / "RUN_MANIFEST.json"
LANE_LOCK = ROOT / "experiments/runs/cls_simplification_benchmark_v1/.run.lock"

PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
EXPECTED_ROWS = 29_421
EXPECTED_E79_SHA256 = "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"
EXPECTED_TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
EXPECTED_SAMPLE_SHA256 = "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7"
COMPETITION = "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027"
SUBMISSION_MESSAGE = "Samuel CLS Simplified A02 Only Frozen Benchmark"

SOURCE_PATHS = (
    "scripts/cls_simplification_benchmark_v1.py",
    "tests/test_cls_simplification_benchmark_v1.py",
    "reports/optimization_v3/cls_simplification_benchmark_v1/PROTOCOL.md",
)
FREEZE_RELATIVE = "reports/optimization_v3/cls_simplification_benchmark_v1/PRE_ACCESS_FREEZE.json"
COMPARISON_CSVS = {
    "current_public_champion": "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv",
    "historical_a02": "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_labels(labels: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(labels, dtype=np.int8)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _require_unique_ids(ids: Sequence[str], label: str) -> None:
    if not ids or any(not value for value in ids):
        raise ValueError(f"{label} contains an empty ID or no rows")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate record_id values")


def route_no_pos(probability: Any, missing_mask: Any) -> np.ndarray:
    """A02 pA on present rows at .50; the same frozen pA fallback at .5323097."""
    p_a = np.asarray(probability, dtype=np.float64)
    missing = np.asarray(missing_mask, dtype=bool)
    if p_a.ndim != 1 or missing.ndim != 1 or p_a.shape != missing.shape or p_a.size == 0:
        raise ValueError("A02 pA and missing mask must be nonempty aligned 1-D vectors")
    if not np.isfinite(p_a).all() or np.any((p_a < 0.0) | (p_a > 1.0)):
        raise ValueError("A02 pA must contain only finite probabilities in [0, 1]")
    labels = np.where(missing, p_a >= MISSING_THRESHOLD, p_a >= PRESENT_THRESHOLD)
    return labels.astype(np.int8)


def _read_ids_missing(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    # Only these two feature columns are materialized; artifact_signal is not loaded.
    frame = pd.read_csv(
        path, usecols=["record_id", "signal_stream"], dtype="string",
        keep_default_na=False, encoding="utf-8-sig",
    )
    ids = frame["record_id"].astype(str).tolist()
    missing = frame["signal_stream"].str.strip().eq("").to_numpy(dtype=bool)
    _require_unique_ids(ids, "Official Test")
    return ids, missing, ["record_id", "signal_stream"]


def _read_sample_ids_and_header(path: Path) -> tuple[list[str], list[str]]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if header != ["record_id", "artifact_signal"]:
        raise ValueError(f"Unexpected SampleSubmission columns: {header}")
    frame = pd.read_csv(path, usecols=["record_id"], dtype={"record_id": "string"},
                        keep_default_na=False, encoding="utf-8-sig")
    ids = frame["record_id"].astype(str).tolist()
    _require_unique_ids(ids, "SampleSubmission")
    return ids, header


def _read_submission(path: Path, expected_ids: Sequence[str]) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["record_id", "artifact_signal"]:
            raise ValueError(f"Unexpected submission schema in {path}: {header}")
        ids: list[str] = []
        labels: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2 or row[1] not in {"0", "1"}:
                raise ValueError(f"Non-integer binary label or malformed row in {path}:{row_number}")
            ids.append(row[0])
            labels.append(int(row[1]))
    _require_unique_ids(ids, str(path))
    if ids != list(expected_ids):
        raise ValueError(f"Submission record_id order differs from official Test: {path}")
    return np.asarray(labels, dtype=np.int8)


def write_csv_once(path: Path, ids: Sequence[str], labels: Any) -> str:
    values = np.asarray(labels)
    if values.shape != (len(ids),) or not np.isin(values, [0, 1]).all():
        raise ValueError("Submission labels must be row-aligned integer 0/1 values")
    payload = "record_id,artifact_signal\n" + "".join(
        f"{record_id},{int(label)}\n" for record_id, label in zip(ids, values)
    )
    encoded = payload.encode("utf-8")
    expected_sha = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != expected_sha:
            raise FileExistsError(f"Refusing to overwrite different candidate CSV: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return expected_sha


def _read_label_vector_if_official(path: Path, expected_ids: Sequence[str]) -> np.ndarray | None:
    if not path.is_file() or path.resolve() == OUTPUT_CSV.resolve():
        return None
    try:
        return _read_submission(path, expected_ids)
    except (ValueError, UnicodeError, OSError):
        return None


def scan_local_duplicate_vectors(ids: Sequence[str], labels: np.ndarray) -> list[str]:
    found: list[str] = []
    submission_root = ROOT / "submissions"
    if not submission_root.exists():
        return found
    candidate_hash = sha256_labels(labels)
    for path in sorted(submission_root.rglob("*.csv")):
        other = _read_label_vector_if_official(path, ids)
        if other is not None and sha256_labels(other) == candidate_hash:
            found.append(path.relative_to(ROOT).as_posix())
    return found


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _verify_frozen_source(freeze: dict[str, Any]) -> dict[str, str]:
    source_hashes = freeze["source_hashes"]
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        expected = source_hashes.get(relative)
        if expected is None or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Committed source hash differs from pre-access freeze: {relative}")
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True
        ).strip()
        if status:
            raise RuntimeError(f"Source/freeze file is not committed cleanly: {relative}")
    freeze_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", FREEZE_RELATIVE], cwd=ROOT, text=True
    ).strip()
    if freeze_status:
        raise RuntimeError("Pre-access freeze is not committed cleanly")
    return {relative: source_hashes[relative] for relative in SOURCE_PATHS}


def _verify_frozen_inputs(freeze: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    all_inputs = dict(freeze["audit"]["input_hashes"])
    all_inputs.update(freeze["input_hashes"])
    for relative, expected in all_inputs.items():
        if relative.startswith("../"):
            path = (ROOT / relative).resolve()
        else:
            path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input SHA-256 mismatch: {relative}")
        hashes[relative] = actual
    return hashes


def _verify_upstream_freezes() -> dict[str, Any]:
    application = json.loads(E79_APPLICATION_FREEZE.read_text(encoding="utf-8"))
    manifest = json.loads(E79_MANIFEST.read_text(encoding="utf-8"))
    qa = json.loads(E79_QA.read_text(encoding="utf-8"))
    if application.get("status") != "FROZEN_BEFORE_E79_PROBABILITY_BUNDLE_ACCESS":
        raise RuntimeError("Upstream E79 application freeze status is not valid")
    if application.get("test_labels_read") is not False:
        raise RuntimeError("Upstream E79 application freeze reports Test-label access")
    if application.get("final_selector_threshold") != MISSING_THRESHOLD:
        raise RuntimeError("Upstream A02 missing threshold differs from the frozen recipe")
    upstream_inputs = application.get("application_input_hashes", {})
    required_upstream_inputs = {
        "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz": EXPECTED_E79_SHA256,
        "data/raw/classification/Test.csv": EXPECTED_TEST_SHA256,
    }
    if any(upstream_inputs.get(key) != value for key, value in required_upstream_inputs.items()):
        raise RuntimeError("Upstream application freeze does not bind the expected E79/Test hashes")
    if manifest.get("status") != "READY_FOR_ROOT_REVIEW_ONE_UPLOAD_MAX":
        raise RuntimeError("Upstream E79 manifest status is not ready")
    if manifest.get("test_labels_read") is not False or manifest.get("threshold_operator") != ">=":
        raise RuntimeError("Upstream E79 manifest has an incompatible label/threshold contract")
    if manifest.get("selected_missing_threshold") != MISSING_THRESHOLD:
        raise RuntimeError("Upstream E79 manifest missing threshold differs")
    if qa.get("status") != "PASS_LOCAL_E79_QA":
        raise RuntimeError("Upstream E79 application QA did not pass")
    return {"application_freeze": application, "manifest": manifest, "qa": qa}


class _LaneFileLock:
    """Small per-candidate idempotency lock; distinct from the shared heavy-job lock."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except (ImportError, OSError) as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("NO_POS lane is already active or lock could not be acquired") from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.stream is not None:
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


def _available_physical_bytes() -> int:
    if os.name != "nt":
        try:
            import psutil  # type: ignore[import-not-found]
            return int(psutil.virtual_memory().available)
        except ImportError as exc:
            raise RuntimeError("Cannot read available system RAM on this platform") from exc
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def _resource_gate() -> dict[str, int | bool]:
    available = _available_physical_bytes()
    estimate = 256 * 1024 * 1024
    start_minimum = int(2.5 * 1024**3)
    reserve = 1 * 1024**3
    if available < start_minimum or available - estimate < reserve:
        raise RuntimeError(
            f"RAM gate failed before cached-vector load: available={available}, "
            f"minimum_start={start_minimum}, estimated_increment={estimate}, reserve={reserve}"
        )
    return {"available_before_bytes": available, "estimated_incremental_bytes": estimate,
            "minimum_start_bytes": start_minimum, "minimum_reserve_bytes": reserve,
            "pass": True}


def _compare_required_vectors(ids: Sequence[str], labels: np.ndarray) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for label, relative in COMPARISON_CSVS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required historical comparison CSV is missing: {relative}")
        prior = _read_submission(path, ids)
        comparison[label] = {
            "path": relative,
            "csv_sha256": sha256_file(path),
            "label_vector_sha256": sha256_labels(prior),
            "exact_vector_match": bool(np.array_equal(prior, labels)),
            "changed_rows": int(np.count_nonzero(prior != labels)),
        }
    return comparison


def _write_json_once(path: Path, value: dict[str, Any]) -> str:
    body = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    encoded = body.encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != expected:
            raise FileExistsError(f"Refusing to overwrite different report: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return expected


def build_candidate() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("benchmark_id") != "CLS_SIMPLIFICATION_NO_POS_V1":
        raise RuntimeError("Unexpected pre-access freeze benchmark_id")
    if RUN_MANIFEST.exists():
        raise RuntimeError("A RUN_MANIFEST already exists; refusing a second candidate build")
    with _LaneFileLock(LANE_LOCK):
        resource = _resource_gate()
        source_hashes = _verify_frozen_source(freeze)
        input_hashes = _verify_frozen_inputs(freeze)
        upstream = _verify_upstream_freezes()

        with np.load(E79_NPZ, allow_pickle=False) as bundle:
            required = {"record_ids", "missing_mask", "a02_probability"}
            if not required.issubset(bundle.files):
                raise RuntimeError("Frozen E79 bundle lacks the NO_POS inputs")
            ids = np.asarray(bundle["record_ids"], dtype="U").tolist()
            bundle_missing = np.asarray(bundle["missing_mask"], dtype=bool)
            p_a = np.asarray(bundle["a02_probability"], dtype=np.float64)
        _require_unique_ids(ids, "E79 probability bundle")
        if len(ids) != EXPECTED_ROWS or bundle_missing.shape != (EXPECTED_ROWS,) or p_a.shape != (EXPECTED_ROWS,):
            raise RuntimeError("Frozen E79 pA/mask/ID vectors do not have exactly 29,421 rows")
        test_ids, test_missing, test_fields = _read_ids_missing(TEST_CSV)
        sample_ids, sample_header = _read_sample_ids_and_header(SAMPLE_CSV)
        if ids != test_ids or ids != sample_ids:
            raise RuntimeError("E79, official Test, and SampleSubmission IDs/order differ")
        if not np.array_equal(bundle_missing, test_missing):
            raise RuntimeError("E79 missing mask differs from Test signal_stream availability")

        labels = route_no_pos(p_a, bundle_missing)
        comparisons = _compare_required_vectors(ids, labels)
        duplicate_vectors = scan_local_duplicate_vectors(ids, labels)
        current_csv_sha = write_csv_once(OUTPUT_CSV, ids, labels)
        reread = _read_submission(OUTPUT_CSV, ids)
        if not np.array_equal(labels, reread):
            raise RuntimeError("CSV reread labels differ from the frozen NO_POS route")
        if len(reread) != EXPECTED_ROWS or not np.isin(reread, [0, 1]).all():
            raise RuntimeError("CSV row-count or binary-label QA failed")

        available_after = _available_physical_bytes()
        resource_pass_after = available_after >= int(1 * 1024**3)
        manifest = {
            "schema_version": 1,
            "benchmark_id": "CLS_SIMPLIFICATION_NO_POS_V1",
            "status": "CSV_QA_PASS_READY_FOR_UPLOAD_PREFLIGHT" if resource_pass_after
                else "CSV_QA_PASS_BUT_RESOURCE_RESERVE_FAILED",
            "source_commit": _git_head(),
            "pre_access_freeze_sha256": sha256_file(FREEZE_PATH),
            "source_hashes": source_hashes,
            "input_hashes": input_hashes,
            "upstream_e79": {
                "manifest_status": upstream["manifest"]["status"],
                "application_freeze_status": upstream["application_freeze"]["status"],
                "missing_threshold": MISSING_THRESHOLD,
                "threshold_operator": ">=",
                "test_labels_read": False,
            },
            "recipe": {
                "candidate": "NO_POS",
                "present_probability": "frozen E79 a02_probability (A02 pA)",
                "missing_probability": "same A02 pA, which exactly preserves the frozen CatBoost/availability fallback",
                "present_threshold": PRESENT_THRESHOLD,
                "missing_threshold": MISSING_THRESHOLD,
                "availability": "signal_stream stripped empty => missing",
                "new_fits": 0,
                "new_preprocessing_fits": 0,
                "new_model_inference": 0,
                "test_fields_read": test_fields,
                "test_target_read": False,
            },
            "csv": {
                "path": OUTPUT_CSV.relative_to(ROOT).as_posix(),
                "sha256": current_csv_sha,
                "label_vector_sha256": sha256_labels(reread),
                "ordered_id_sha256": sha256_ids(ids),
                "rows": len(reread),
                "columns": sample_header,
                "unique_ids": len(set(ids)),
                "label_counts": {"0": int(np.sum(reread == 0)), "1": int(np.sum(reread == 1))},
                "required_comparisons": comparisons,
                "duplicate_local_vectors": duplicate_vectors,
            },
            "resource_policy": {
                "policy": "user-authorized two-lane parallel light cached-vector/CSV work",
                "estimated_incremental_bytes": resource["estimated_incremental_bytes"],
                "available_before_bytes": resource["available_before_bytes"],
                "available_after_bytes": available_after,
                "max_parallel_light_jobs": 2,
                "minimum_parallel_start_bytes": resource["minimum_start_bytes"],
                "minimum_reserve_bytes": resource["minimum_reserve_bytes"],
                "reserve_pass_after": resource_pass_after,
                "heavy_shared_lock_required": False,
                "lane_lock": str(LANE_LOCK.relative_to(ROOT)),
            },
            "competition": COMPETITION,
            "submission_message": SUBMISSION_MESSAGE,
            "upload_authorization": "at most one Classification upload, only after live history/rules/quota/idempotency checks",
        }
        _write_json_once(RUN_MANIFEST, manifest)
        return manifest


def self_test() -> dict[str, Any]:
    import tempfile

    p = np.asarray([0.49, 0.50, MISSING_THRESHOLD, MISSING_THRESHOLD - 1e-8, 0.60])
    missing = np.asarray([False, False, True, True, False])
    labels = route_no_pos(p, missing)
    expected = np.asarray([0, 1, 1, 0, 1], dtype=np.int8)
    if not np.array_equal(labels, expected):
        raise AssertionError("threshold and availability route self-test failed")
    with tempfile.TemporaryDirectory(prefix="cls_no_pos_qa_") as directory:
        path = Path(directory) / "synthetic_submission.csv"
        ids = ["T-1", "T-2", "T-3", "T-4", "T-5"]
        digest = write_csv_once(path, ids, labels)
        replay = _read_submission(path, ids)
        if not np.array_equal(replay, labels) or sha256_file(path) != digest:
            raise AssertionError("synthetic CSV write/reread self-test failed")
        try:
            write_csv_once(path, ids, 1 - labels)
        except FileExistsError:
            pass
        else:
            raise AssertionError("synthetic idempotency self-test failed")
    return {"status": "PASS", "checks": ["threshold ties use >=", "availability routing",
            "binary CSV schema and round-trip", "exclusive-write idempotency"], "new_fits": 0,
            "official_data_access": False, "kaggle_access": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="run synthetic CSV checks only")
    group.add_argument("--build", action="store_true", help="build and independently reread frozen E79 CSV")
    args = parser.parse_args(argv)
    result = self_test() if args.self_test else build_candidate()
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
