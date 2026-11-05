"""Independent, target-free verification for POS_FREQUENCY_RIDGE_V1 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

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


def verify_source_preflight(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("campaign_id") != "CLS_POS_FREQUENCY_RIDGE_V1":
        raise VerificationError("Source preflight campaign ID mismatch")
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
