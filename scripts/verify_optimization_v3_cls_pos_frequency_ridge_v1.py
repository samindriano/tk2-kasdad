"""Independent, target-free verification for POS_FREQUENCY_RIDGE_V1 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pos_frequency_ridge_v1 import canonical_sha256, sha256_array


class VerificationError(AssertionError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _row_metadata(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    selected = frame.iloc[rows]
    ids = selected["record_id"].astype(str).to_numpy()
    cycles = selected["cycle_code"].to_numpy(dtype=np.int64)
    return {
        "rows": int(len(rows)),
        "cycle_min": int(cycles.min()),
        "cycle_max": int(cycles.max()),
        "row_indices_sha256": _array_sha256(rows),
        "record_ids_sha256": _array_sha256(ids),
        "cycles_sha256": _array_sha256(cycles),
    }


def verify_source_preflight(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("campaign_id") != "CLS_POS_FREQUENCY_RIDGE_V1":
        raise VerificationError("Source preflight campaign ID mismatch")
    if (report.get("schema_version") != 2
            or report.get("source_preflight_revision") != "R2_SHARED_LOCK_DURABLE_RESUME"
            or not report.get("shared_job_lock_path")):
        raise VerificationError("Source preflight does not identify the active R2 lock/resume contract")
    if (report.get("target_values_opened") is not False
            or report.get("test_or_sample_submission_accessed") is not False
            or report.get("kaggle_accessed") is not False
            or report.get("official_fit_started") is not False):
        raise VerificationError("Source preflight claims forbidden access or a fit")
    if report.get("train_sha256") != "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c":
        raise VerificationError("Source preflight Train hash differs from frozen official input")
    expected_rows = {"E67": (29705, 20903), "E75": (42117, 8491)}
    for key, (fit_rows, score_rows) in expected_rows.items():
        origin = report.get("origin_identity_and_cache_checks", {}).get(key, {})
        if origin.get("fit_rows") != fit_rows or origin.get("score_rows") != score_rows:
            raise VerificationError(f"{key} identity population differs from the frozen protocol")
    hashes = report.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes or any(len(str(v)) != 64 for v in hashes.values()):
        raise VerificationError("Source file hashes are absent or malformed")
    for relative, expected_sha in hashes.items():
        source_path = ROOT / relative
        if not source_path.is_file() or sha256_file(source_path) != expected_sha:
            raise VerificationError(f"Executable source hash mismatch: {relative}")
    train_path = ROOT / "data/raw/classification/Train.csv"
    if sha256_file(train_path) != report["train_sha256"]:
        raise VerificationError("Official Train byte hash differs from source-preflight receipt")
    identity = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    identity["cycle_code"] = pd.to_numeric(identity["cycle_code"], errors="raise").astype(np.int64)
    for origin, (fit_rows_expected, score_rows_expected) in expected_rows.items():
        cycle = identity["cycle_code"].to_numpy(dtype=np.int64)
        fit_rows = np.flatnonzero(cycle <= int(origin[1:]))
        score_rows = np.flatnonzero(cycle > int(origin[1:]))
        actual = report["origin_identity_and_cache_checks"][origin]
        if (_row_metadata(identity, fit_rows) != actual["fit_row_metadata"]
                or _row_metadata(identity, score_rows) != actual["score_row_metadata"]):
            raise VerificationError(f"{origin} Train ID/cycle row mapping does not reproduce")
        manifest_path = ROOT / f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_{origin}.json"
        if sha256_file(manifest_path) != actual["matrix_manifest_sha256"]:
            raise VerificationError(f"{origin} matrix manifest hash differs from preflight")
        manifest = _read_json(manifest_path)
        for name, expected_matrix_sha in actual["matrix_file_hashes"].items():
            matrix_path = ROOT / manifest["matrices"][name]["path"]
            if sha256_file(matrix_path) != expected_matrix_sha:
                raise VerificationError(f"{origin} matrix file hash mismatch: {name}")
        prediction_path = ROOT / f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/{origin}/predictions_{origin}.npz"
        if sha256_file(prediction_path) != actual["prediction_bundle_sha256"]:
            raise VerificationError(f"{origin} prediction bundle hash differs from preflight")
    return {
        "status": "PASS_SOURCE_PREFLIGHT_INDEPENDENT",
        "campaign_id": report["campaign_id"],
        "source_commit": report["source_commit"],
        "source_hashes_count": len(hashes),
        "target_values_opened": False,
        "test_accessed": False,
        "kaggle_accessed": False,
    }


def verify_gamma_artifact(artifact_path: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    if receipt.get("fit_kind") != "PREDICTOR" or receipt.get("campaign_id") != "CLS_POS_FREQUENCY_RIDGE_V1":
        raise VerificationError("Gamma receipt campaign/fit kind mismatch")
    artifact_sha = sha256_file(artifact_path)
    if receipt.get("artifact_sha256") != artifact_sha:
        raise VerificationError("Gamma artifact file hash differs from receipt")
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma"}:
            raise VerificationError("Gamma artifact contains unexpected arrays")
        gamma = np.asarray(archive["gamma"], dtype=np.float64)
    if gamma.ndim != 1 or not np.isfinite(gamma).all():
        raise VerificationError("Gamma must be a finite vector")
    if sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise VerificationError("Gamma content hash differs from receipt")
    return {
        "status": "PASS_GAMMA_ARTIFACT_INDEPENDENT",
        "fit_id": receipt.get("fit_id"),
        "gamma_width": int(len(gamma)),
        "gamma_sha256": sha256_array(gamma),
        "artifact_sha256": artifact_sha,
        "test_accessed": False,
        "score_labels_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-preflight", type=Path)
    group.add_argument("--gamma-artifact", nargs=2, metavar=("NPZ", "RECEIPT"), type=Path)
    args = parser.parse_args(argv)
    if args.source_preflight:
        result = verify_source_preflight(args.source_preflight)
    else:
        result = verify_gamma_artifact(args.gamma_artifact[0], args.gamma_artifact[1])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
