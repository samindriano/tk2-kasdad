"""Run one authorized zero-fit inference continuation from the frozen E79 states."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reg_position_real_e79_benchmark_v1 as bench
from reg_position_real_e79_hash_compat_v2 import (
    EXPECTED_FIT_ARTIFACTS,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_FREEZE_SIGNATURE,
    EXPECTED_LEGACY_PREDICTION_SHA256,
    EXPECTED_ORIGINAL_WORKTREE_SOURCE_HASHES,
    EXPECTED_PREFLIGHT_SHA256,
    EXPECTED_R06_FILE_HASHES,
    EXPECTED_STRUCTURED_PREDICTION_SHA256,
    ORIGINAL_COMMIT,
    SOURCE_ROOT,
    WORKTREE_ROOT,
    validate_legacy_prediction_manifest_digest,
)

COMPAT_REL = "scripts/reg_position_real_e79_hash_compat_v2.py"
INFER_REL = "scripts/reg_position_real_e79_inference_v2.py"
VERIFY_REL = "scripts/verify_reg_position_real_e79_benchmark_v2.py"
TEST_REL = "tests/test_reg_position_real_e79_hash_compat_v2.py"
AUTHORIZATION_REL = (
    "reports/optimization_v3/reg_position_real_e79_benchmark_v1/"
    "INFERENCE_CONTINUATION_AUTHORIZATION_V2.md"
)
RUN_REL = Path("experiments/runs/reg_position_real_e79_benchmark_v1")
REPORT_REL = Path("reports/optimization_v3/reg_position_real_e79_benchmark_v1")
CSV_REL = Path(
    "submissions/optimization_v3_reg_position_real_e79_exploratory_benchmark_v1.csv"
)
R06_MANIFEST_REL = Path(
    "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json"
)
R06_NPZ_REL = Path(
    "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz"
)
R06_CSV_REL = Path("submissions/optimization_v3_expanded_v1/r06_original.csv")
TEST_SHA256 = bench.TEST_SHA256
SAMPLE_SHA256 = bench.SAMPLE_SHA256


class ContinuationError(RuntimeError):
    """Raised when the approved frozen continuation boundary fails."""


def _resolve_root(root: Path, expected: str, name: str) -> Path:
    resolved = root.resolve()
    if os.path.normcase(str(resolved)) != os.path.normcase(str(Path(expected).resolve())):
        raise ContinuationError(f"{name} differs from the root-approved exact path")
    return resolved


def _git_head(worktree: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True, capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContinuationError("Could not bind inference continuation to a local commit")
    return result.stdout.strip()


def _read_bound_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected a JSON object at {path}")
    return value


def _check_fit_state(output: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in (output / "fit_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    starts = [event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event.get("fit_id") for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    expected_ids = ["WORD_TFIDF_FULL_TRAIN", "REAL_POSITION_RESIDUAL_L1_E79"]
    if starts != expected_ids or saved != expected_ids or failed:
        raise ContinuationError("Frozen fit ledger is not exactly two successful starts")
    if freeze.get("accounting", {}).get("fit_starts") != 2:
        raise ContinuationError("Training freeze does not bind exactly two starts")
    if freeze.get("accounting", {}).get("fit_saved") != 2:
        raise ContinuationError("Training freeze does not bind exactly two saved units")
    bound_units: dict[str, Any] = {}
    sections = {
        "WORD_TFIDF_FULL_TRAIN": "vectorizer",
        "REAL_POSITION_RESIDUAL_L1_E79": "head",
    }
    for unit, section in sections.items():
        expected = EXPECTED_FIT_ARTIFACTS[unit]
        artifact = output / "states" / f"{unit}.joblib"
        receipt_path = output / "receipts" / f"{unit}.json"
        receipt = _read_bound_json(receipt_path)
        hashes = {
            "artifact_sha256": bench._sha_file(artifact),
            "receipt_sha256": receipt.get("receipt_sha256"),
            "receipt_file_sha256": bench._sha_file(receipt_path),
        }
        checks = {key: hashes[key] == expected[key] for key in expected}
        checks["freeze_artifact_matches"] = (
            freeze.get(section, {}).get("artifact_sha256") == expected["artifact_sha256"]
        )
        checks["freeze_receipt_matches"] = (
            freeze.get(section, {}).get("receipt_sha256") == expected["receipt_sha256"]
        )
        if not all(checks.values()):
            raise ContinuationError(f"Saved model/receipt binding failed for {unit}: {checks}")
        bound_units[unit] = {**hashes, "checks": checks}
    return {
        "fit_ids": starts,
        "fit_starts_before_continuation": len(starts),
        "fit_saved_before_continuation": len(saved),
        "fit_starts_added_by_continuation": 0,
        "retries": 0,
        "units": bound_units,
    }


def _validate_authorized_scope(
    source: Path, worktree: Path, output: Path,
) -> dict[str, Any]:
    if output.resolve() != (worktree / RUN_REL).resolve():
        raise ContinuationError("Output root escaped the isolated benchmark namespace")
    preflight_path = output / "preflight.json"
    freeze_path = output / "trained_freeze.json"
    preflight_sha = bench._sha_file(preflight_path)
    freeze_sha = bench._sha_file(freeze_path)
    if preflight_sha != EXPECTED_PREFLIGHT_SHA256:
        raise ContinuationError("Original immutable zero-fit preflight hash changed")
    if freeze_sha != EXPECTED_FREEZE_SHA256:
        raise ContinuationError("Original trained freeze hash changed")
    preflight = _read_bound_json(preflight_path)
    freeze = _read_bound_json(freeze_path)
    if preflight.get("worktree_source_hashes") != EXPECTED_ORIGINAL_WORKTREE_SOURCE_HASHES:
        raise ContinuationError("Original preflight source hash inventory changed")
    if freeze.get("freeze_signature") != EXPECTED_FREEZE_SIGNATURE:
        raise ContinuationError("Original trained freeze signature changed")
    if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST_FEATURE_ACCESS":
        raise ContinuationError("Training freeze is not the authorized pre-Test freeze")
    if freeze.get("test_features_accessed") is not False:
        raise ContinuationError("Original training freeze records Test feature access")
    original_source_checks = {
        relative: bench._sha_file(worktree / relative) == expected
        for relative, expected in EXPECTED_ORIGINAL_WORKTREE_SOURCE_HASHES.items()
    }
    if not all(original_source_checks.values()):
        raise ContinuationError(
            "Preflight-frozen original worktree source changed: "
            + ", ".join(key for key, value in original_source_checks.items() if not value)
        )
    r06_checks = {
        relative: bench._sha_file(source / relative) == expected
        for relative, expected in EXPECTED_R06_FILE_HASHES.items()
    }
    if not all(r06_checks.values()):
        raise ContinuationError(
            "Root-approved R06 source artifact changed: "
            + ", ".join(key for key, value in r06_checks.items() if not value)
        )
    fit_state = _check_fit_state(output, freeze)
    csv_path = worktree / CSV_REL
    report_manifest_path = worktree / REPORT_REL / "INFERENCE_MANIFEST.json"
    if (output / "inference").exists() or csv_path.exists() or report_manifest_path.exists():
        raise ContinuationError("Prior inference output exists; refusing overwrite or retry")
    if not (source / R06_MANIFEST_REL).is_file():
        raise ContinuationError("R06 manifest is missing")
    return {
        "source_root": str(source),
        "worktree_root": str(worktree),
        "output_root": str(output),
        "original_worktree_commit": ORIGINAL_COMMIT,
        "amendment_git_head": _git_head(worktree),
        "preflight_path": preflight_path.relative_to(worktree).as_posix(),
        "preflight_sha256": preflight_sha,
        "original_frozen_source_hashes": original_source_checks,
        "trained_freeze_path": freeze_path.relative_to(worktree).as_posix(),
        "trained_freeze_sha256": freeze_sha,
        "trained_freeze_signature": freeze["freeze_signature"],
        "fit_accounting": fit_state,
        "r06_source_file_hashes": r06_checks,
        "r06_expected_file_hashes": EXPECTED_R06_FILE_HASHES,
        "official_test_sha256": TEST_SHA256,
        "official_sample_sha256": SAMPLE_SHA256,
        "prior_v1_attempt": {
            "status": "FAIL_CLOSED",
            "error": "R06_ORIGINAL final Test prediction values/cycles failed",
            "reason": "legacy raw-byte manifest digest was compared to a structured array digest",
            "test_features_opened": False,
            "csv_created": False,
            "fit_starts_added": 0,
        },
    }


def _load_r06_test_baseline_v2(source_root: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    source = source_root.resolve()
    manifest_path = bench._contained(source, R06_MANIFEST_REL)
    prediction_path = bench._contained(source, R06_NPZ_REL)
    csv_path = bench._contained(source, R06_CSV_REL)
    manifest = bench._read_json(manifest_path)
    metadata_checks = {
        "candidate_is_r06_original": manifest.get("candidate") == "R06_ORIGINAL",
        "manifest_status_is_prepared_validated": manifest.get("status") == "PREPARED_VALIDATED_NOT_SUBMITTED",
        "manifest_test_rows": manifest.get("test_rows") == bench.EXPECTED_TEST_ROWS,
        "manifest_sample_rows": manifest.get("sample_submission_rows") == bench.EXPECTED_TEST_ROWS,
        "manifest_test_input_hash": manifest.get("official_input_hashes", {}).get("test") == TEST_SHA256,
        "manifest_sample_input_hash": manifest.get("official_input_hashes", {}).get("sample") == SAMPLE_SHA256,
        "manifest_prediction_path": manifest.get("prediction_artifact", {}).get("path_gitignored") == R06_NPZ_REL.as_posix(),
        "manifest_prediction_file_hash": manifest.get("prediction_artifact", {}).get("sha256") == bench._sha_file(prediction_path),
        "manifest_csv_path": manifest.get("csv", {}).get("path_gitignored") == R06_CSV_REL.as_posix(),
        "manifest_csv_rows": manifest.get("csv", {}).get("rows") == bench.EXPECTED_TEST_ROWS,
        "manifest_csv_columns": manifest.get("csv", {}).get("columns") == ["record_id", "threat_rank"],
        "manifest_csv_file_hash": manifest.get("csv", {}).get("sha256") == bench._sha_file(csv_path),
    }
    if not all(metadata_checks.values()):
        failed = [key for key, passed in metadata_checks.items() if not passed]
        raise bench.BenchmarkError("R06 manifest metadata checks failed: " + ", ".join(failed))
    with bench.np.load(prediction_path, allow_pickle=False) as archive:
        if not {"prediction", "cycle_code"}.issubset(archive.files):
            raise bench.BenchmarkError("R06 prediction archive lacks prediction or cycle_code")
        raw_prediction = bench.np.asarray(archive["prediction"]).copy()
        raw_cycles = bench.np.asarray(archive["cycle_code"]).copy()
    baseline = bench.np.asarray(raw_prediction, dtype=bench.np.float64)
    cycles = bench.np.asarray(raw_cycles, dtype=bench.np.int64)
    csv = bench.pd.read_csv(csv_path, dtype={"record_id": "string"})
    csv_values = (
        bench.pd.to_numeric(csv["threat_rank"], errors="coerce").to_numpy(dtype=bench.np.float64)
        if "threat_rank" in csv.columns else bench.np.asarray([], dtype=bench.np.float64)
    )
    ids = csv.record_id.astype(str).to_numpy(dtype=str) if "record_id" in csv.columns else bench.np.asarray([], dtype=str)
    max_delta = (
        float(bench.np.max(bench.np.abs(csv_values - baseline)))
        if len(csv_values) == len(baseline) and len(baseline) else None
    )
    raw_hash_contract = validate_legacy_prediction_manifest_digest(
        manifest.get("prediction_audit", {}).get("prediction_sha256"), baseline,
    )
    checks = {
        "archive_prediction_is_float64": raw_prediction.dtype == bench.np.dtype("float64"),
        "archive_prediction_is_one_dimensional": baseline.ndim == 1,
        "archive_prediction_row_count": len(baseline) == bench.EXPECTED_TEST_ROWS,
        "archive_prediction_finite": bool(bench.np.isfinite(baseline).all()),
        "archive_prediction_range": bool(not bench.np.any((baseline < 0.0) | (baseline > 100.0))),
        "csv_schema": csv.columns.tolist() == ["record_id", "threat_rank"],
        "csv_row_count": len(csv) == bench.EXPECTED_TEST_ROWS,
        "csv_prediction_finite": bool(bench.np.isfinite(csv_values).all()),
        "csv_prediction_range": bool(not bench.np.any((csv_values < 0.0) | (csv_values > 100.0))),
        "csv_prediction_matches_archive_within_1e_12": max_delta is not None and max_delta <= 1e-12,
        "csv_ids_present_and_unique": (
            len(ids) == bench.EXPECTED_TEST_ROWS
            and "record_id" in csv.columns and not csv.record_id.isna().any()
            and len(set(ids.tolist())) == bench.EXPECTED_TEST_ROWS
        ),
        "cycles_are_int64": raw_cycles.dtype == bench.np.dtype("int64"),
        "cycle_row_count": len(cycles) == bench.EXPECTED_TEST_ROWS,
        "cycles_cover_80_through_91": bench.np.array_equal(
            bench.np.unique(cycles), bench.np.arange(80, 92, dtype=bench.np.int64)
        ),
        "expected_legacy_raw_digest": raw_hash_contract["legacy_raw_le_f8_sha256"] == EXPECTED_LEGACY_PREDICTION_SHA256,
        "expected_structured_digest": raw_hash_contract["structured_array_sha256"] == EXPECTED_STRUCTURED_PREDICTION_SHA256,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise bench.BenchmarkError("R06 baseline checks failed: " + ", ".join(failed))
    audit = {
        "manifest_path": R06_MANIFEST_REL.as_posix(),
        "manifest_sha256": bench._sha_file(manifest_path),
        "prediction_path": R06_NPZ_REL.as_posix(),
        "prediction_file_sha256": bench._sha_file(prediction_path),
        "csv_path": R06_CSV_REL.as_posix(),
        "csv_sha256": bench._sha_file(csv_path),
        "max_csv_archive_abs_delta": max_delta,
        "cycle_values_sha256": bench._sha_array(cycles, "<i8"),
        "gate_checks": checks,
        "metadata_checks": metadata_checks,
        "hash_contract": raw_hash_contract,
    }
    return baseline, cycles, ids, audit


def _adapter_file_hashes(worktree: Path) -> dict[str, str]:
    paths = (COMPAT_REL, INFER_REL, VERIFY_REL, TEST_REL, AUTHORIZATION_REL)
    hashes = {relative: bench._sha_file(worktree / relative) for relative in paths}
    if any(len(value) != 64 for value in hashes.values()):
        raise ContinuationError("A versioned continuation file could not be hash-bound")
    return hashes


def _save_manifest(manifest: dict[str, Any], worktree: Path, output: Path) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = bench._canonical_sha(manifest)
    paths = (
        output / "inference" / "manifest.json",
        worktree / REPORT_REL / "INFERENCE_MANIFEST.json",
    )
    for path in paths:
        bench._atomic_json(path, manifest)
    if any(bench._read_json(path) != manifest for path in paths):
        raise ContinuationError("Versioned inference manifest copies differ after persistence")


def run_inference_v2(
    source_root: Path, worktree_root: Path, output_root: Path, csv_path: Path | None = None,
) -> dict[str, Any]:
    source = _resolve_root(source_root, SOURCE_ROOT, "Source root")
    worktree = _resolve_root(worktree_root, WORKTREE_ROOT, "Worktree root")
    output = output_root.resolve()
    binding = _validate_authorized_scope(source, worktree, output)
    binding["adapter_file_hashes"] = _adapter_file_hashes(worktree)
    binding["adapter_id"] = "REG_POSITION_REAL_E79_ZERO_FIT_INFERENCE_V2"
    binding["authorization"] = (
        "Sol-authorized continuation from the same two durably saved model states; "
        "no model fit, retry, recipe change, or upload."
    )
    binding["authorization_binding_sha256"] = bench._canonical_sha(binding)
    original_loader = bench._load_r06_test_baseline
    original_atomic_json = bench._atomic_json
    deferred_manifests: dict[str, Any] = {}
    manifest_targets = {
        (output / "inference" / "manifest.json").resolve(),
        (worktree / REPORT_REL / "INFERENCE_MANIFEST.json").resolve(),
    }

    def defer_unversioned_manifest(path: Path, value: Any) -> None:
        resolved = Path(path).resolve()
        if resolved in manifest_targets:
            key = str(resolved)
            if key in deferred_manifests:
                raise ContinuationError(f"Duplicate manifest write attempted: {resolved}")
            deferred_manifests[key] = copy.deepcopy(value)
            return
        original_atomic_json(path, value)

    bench._load_r06_test_baseline = _load_r06_test_baseline_v2
    bench._atomic_json = defer_unversioned_manifest
    try:
        manifest = bench.run_inference(source, worktree, output, csv_path)
    finally:
        bench._load_r06_test_baseline = original_loader
        bench._atomic_json = original_atomic_json
    expected_targets = {str(path) for path in manifest_targets}
    if set(deferred_manifests) != expected_targets:
        raise ContinuationError("Expected exactly two deferred version 1 manifest writes")
    if any(value != manifest for value in deferred_manifests.values()):
        raise ContinuationError("The two version 1 manifest payloads differ from inference output")
    original_manifest_hash = manifest.get("manifest_sha256")
    if bench._canonical_sha({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }) != original_manifest_hash:
        raise ContinuationError("Original inference manifest canonical signature failed")
    continuation = {
        **binding,
        "test_features_opened_after_baseline_gate": True,
        "test_labels_read": False,
        "fit_starts_after_continuation": 2,
        "fit_starts_added_by_continuation": 0,
        "upload_performed": False,
    }
    continuation["continuation_signature"] = bench._canonical_sha(continuation)
    manifest["inference_continuation"] = continuation
    manifest["baseline_hash_compatibility"] = manifest["baseline"]["hash_contract"]
    _save_manifest(manifest, worktree, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    args = parser.parse_args()
    try:
        manifest = run_inference_v2(
            args.source_root, args.worktree_root, args.output_root, args.csv_path,
        )
        print(json.dumps({
            "status": manifest["status"],
            "scientific_status": manifest["scientific_status"],
            "submission": manifest["submission"],
            "fit_starts": manifest["inference_continuation"]["fit_accounting"]["fit_starts_before_continuation"],
            "fit_starts_added_by_continuation": manifest["inference_continuation"]["fit_starts_added_by_continuation"],
            "baseline_hash_contract": manifest["baseline_hash_compatibility"],
            "manifest_sha256": manifest["manifest_sha256"],
        }, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED", "error_type": type(exc).__name__, "error": str(exc),
        }, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
