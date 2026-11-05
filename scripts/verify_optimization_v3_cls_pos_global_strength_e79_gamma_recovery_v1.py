"""Independent readback and tamper verifier for the E79 gamma recovery audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3"
FREEZE_REL = f"{REPORT_DIR_REL}/GAMMA_ARTIFACT_RECOVERY_FREEZE.json"
RUNNER_REL = "scripts/audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py"
VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py"
TEST_REL = "tests/test_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py"
PROTOCOL_REL = f"{REPORT_DIR_REL}/GAMMA_ARTIFACT_RECOVERY_PROTOCOL.md"
OUTPUT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/recovery_v1"
ATTEMPT_REL = f"{OUTPUT_REL}/ATTEMPT.started.json"
RECEIPT_REL = f"{OUTPUT_REL}/RECOVERY_RECEIPT.json"
ORIGINAL_GAMMA_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/gamma.npz"
ORIGINAL_ATTEMPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/ATTEMPT.started.json"
ORIGINAL_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/fit_receipt.json"
ORIGINAL_FREEZE_REL = f"{REPORT_DIR_REL}/GAMMA_SOURCE_FREEZE.json"
ORIGINAL_PASS_REL = f"{REPORT_DIR_REL}/GAMMA_ROOT_SOURCE_PASS.json"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_GAMMA_FILE_SHA256 = "4bc240bb6b53eb623f5f099246b70572f987b8c3b0f8dfc123f49029d3c079ab"
EXPECTED_FAILURE_RECEIPT_SHA256 = "05058193fb2cfb32e70d4d2d939fa3abf75b3e1aa77a947eda0a9ccddfb70566"
FROZEN_SOURCE_FILES = (RUNNER_REL, VERIFIER_REL, TEST_REL, PROTOCOL_REL)


class VerificationError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_file_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".md", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True).stdout.strip()


def _git_blob_sha(root: Path, commit: str, relative: str) -> str:
    raw = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if Path(relative).suffix.lower() in {".md", ".json"}:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def verify_source(root: Path = ROOT) -> dict[str, Any]:
    freeze_path = root / FREEZE_REL
    freeze = _json(freeze_path)
    body = dict(freeze)
    digest = body.pop("freeze_sha256", None)
    checks: dict[str, bool] = {
        "freeze_digest": digest == _canonical_sha256(body),
        "freeze_identity": freeze.get("status") == "FROZEN_ZERO_OPTIMIZER_ARTIFACT_RECOVERY_AUDIT",
        "source_file_set": set(freeze.get("source_file_sha256", {})) == set(FROZEN_SOURCE_FILES),
    }
    head = _git(root, "rev-parse", "HEAD")
    prepared = str(freeze.get("prepared_from_head", ""))
    checks["prepared_head_is_ancestor"] = bool(prepared) and subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", prepared, head], check=False).returncode == 0
    checks["freeze_committed_exactly"] = _git_blob_sha(root, head, FREEZE_REL) == _stable_file_sha256(freeze_path)
    for rel, expected in freeze.get("source_file_sha256", {}).items():
        checks[f"working_source:{rel}"] = _stable_file_sha256(root / rel) == expected
        checks[f"committed_source:{rel}"] = _git_blob_sha(root, head, rel) == expected
    checks["original_gamma_artifact_unchanged"] = _file_sha256(root / ORIGINAL_GAMMA_REL) == EXPECTED_GAMMA_FILE_SHA256
    checks["original_failure_receipt_unchanged"] = _file_sha256(root / ORIGINAL_RECEIPT_REL) == EXPECTED_FAILURE_RECEIPT_SHA256
    checks["original_attempt_unchanged"] = _file_sha256(root / ORIGINAL_ATTEMPT_REL) == freeze.get("lineage", {}).get("original_attempt_marker_sha256")
    checks["original_freeze_unchanged"] = _stable_file_sha256(root / ORIGINAL_FREEZE_REL) == freeze.get("lineage", {}).get("original_source_freeze_file_sha256")
    checks["original_pass_unchanged"] = _stable_file_sha256(root / ORIGINAL_PASS_REL) == freeze.get("lineage", {}).get("original_source_pass_file_sha256")
    if not all(checks.values()):
        raise VerificationError(f"Recovery source verification failed: {checks}")
    report = {"status": "PASS_INDEPENDENT_RECOVERY_SOURCE_VERIFICATION",
              "checks": checks, "recovery_freeze_sha256": digest,
              "recovery_freeze_file_sha256": _stable_file_sha256(freeze_path),
              "head": head, "target_or_test_values_loaded": False}
    report["verification_sha256"] = _canonical_sha256(report)
    return report


def verify_official(root: Path = ROOT) -> dict[str, Any]:
    source = verify_source(root)
    attempt_path, receipt_path = root / ATTEMPT_REL, root / RECEIPT_REL
    if not attempt_path.is_file() or not receipt_path.is_file():
        raise VerificationError("Recovery audit has no attempt marker and terminal receipt")
    attempt, receipt = _json(attempt_path), _json(receipt_path)
    receipt_body = dict(receipt)
    claimed = receipt_body.pop("receipt_sha256", None)
    checks = {
        "receipt_self_hash": claimed == _canonical_sha256(receipt_body),
        "attempt_status": attempt.get("status") == "ZERO_FIT_RECOVERY_AUDIT_STARTED",
        "attempt_fit_count_zero": attempt.get("new_optimizer_or_estimator_fit_count") == 0,
        "receipt_source_binding": receipt.get("recovery_freeze_sha256") == source["recovery_freeze_sha256"],
        "original_status_still_fail": receipt.get("original_fit_status") == "FAIL_E79_GAMMA_FIT_NO_RETRY",
        "unknown_optimizer_metadata_preserved": receipt.get("original_optimizer_success_metadata") == "UNKNOWN_NOT_PERSISTED",
        "no_test_csv_network": receipt.get("test_or_sample_read") is False and receipt.get("csv_created") is False
            and receipt.get("network_contacted") is False and receipt.get("kaggle_contacted") is False,
        "new_fit_counts_zero": receipt.get("new_estimator_fit_calls") == 0
            and receipt.get("new_preprocessing_fit_calls") == 0 and receipt.get("prediction_calls") == 0,
        "all_runtime_fit_guards_zero": all(value == 0 for value in receipt.get("fit_call_guards", {}).values()),
        "attempt_binding": receipt.get("attempt_marker_sha256") == _file_sha256(attempt_path),
        "original_gamma_binding": receipt.get("input_sha256", {}).get("original_gamma_artifact_file_sha256") == EXPECTED_GAMMA_FILE_SHA256,
        "original_failed_receipt_binding": receipt.get("input_sha256", {}).get("original_failure_receipt_file_sha256") == EXPECTED_FAILURE_RECEIPT_SHA256,
        "target_train_binding": receipt.get("input_sha256", {}).get("official_train_file_sha256") == TRAIN_SHA256,
    }
    for name, expected in receipt.get("optimizer_calls", {}).items():
        checks[f"zero_optimizer_call:{name}"] = expected == 0
    cert = receipt.get("certificate", {})
    if receipt.get("status") == "RECOVERED_ARTIFACT_NUMERICALLY_VERIFIED":
        checks["certificate_status"] = cert.get("status") == "PASS_NUMERICAL_CERTIFICATE"
        checks["certificate_checks_all_pass"] = all(cert.get("checks", {}).values())
        checks["certificate_rule"] = cert.get("absolute_acceptance_rule") == "gap_bound <= max(1e-4, 1e-8*abs(loss_final))"
        checks["certificate_bound_pass"] = cert.get("absolute_excess_objective_bound", float("inf")) <= cert.get("absolute_bound_threshold", -1.0)
        checks["gamma_vector_hash"] = cert.get("gamma_vector_sha256") == receipt.get("input_sha256", {}).get("original_gamma_vector_sha256")
    elif receipt.get("status") == "FAIL_RECOVERY_NUMERICAL_CERTIFICATE_NO_RETRY":
        checks["terminal_certificate_failure"] = cert.get("status") == "FAIL_NUMERICAL_CERTIFICATE"
    else:
        checks["recognized_terminal_status"] = receipt.get("status") == "FAIL_RECOVERY_AUDIT_NO_RETRY"
    if not all(checks.values()):
        raise VerificationError(f"Official recovery readback/tamper checks failed: {checks}")
    result = {"status": "PASS_INDEPENDENT_RECOVERY_RECEIPT_READBACK",
              "terminal_status": receipt.get("status"), "checks": checks,
              "recovery_receipt_file_sha256": _file_sha256(receipt_path),
              "recovery_receipt_sha256": claimed,
              "original_gamma_file_sha256": _file_sha256(root / ORIGINAL_GAMMA_REL),
              "original_failure_receipt_file_sha256": _file_sha256(root / ORIGINAL_RECEIPT_REL),
              "target_or_test_values_reloaded": False}
    result["verification_sha256"] = _canonical_sha256(result)
    return result


def verify_synthetic(output_dir: Path) -> dict[str, Any]:
    attempt_path, gamma_path = output_dir / "ATTEMPT.started.json", output_dir / "gamma.npz"
    receipt_path = output_dir / "RECOVERY_RECEIPT.json"
    if not all(path.is_file() for path in (attempt_path, gamma_path, receipt_path)):
        raise VerificationError("Synthetic recovery bundle is incomplete")
    attempt, receipt = _json(attempt_path), _json(receipt_path)
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    with np.load(gamma_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma_float64"}:
            raise VerificationError("Synthetic gamma bundle schema mismatch")
        gamma = np.asarray(archive["gamma_float64"]).copy()
    gamma_digest = _array_sha256(gamma)
    checks = {
        "attempt_identity": attempt.get("status") == "SYNTHETIC_ZERO_FIT_AUDIT_STARTED",
        "receipt_digest": claimed == _canonical_sha256(body),
        "gamma_file_hash": receipt.get("gamma_file_sha256") == _file_sha256(gamma_path),
        "gamma_vector_hash": receipt.get("gamma_vector_sha256") == gamma_digest,
        "certificate_pass": receipt.get("certificate", {}).get("status") == "PASS_NUMERICAL_CERTIFICATE"
            and all(receipt.get("certificate", {}).get("checks", {}).values()),
        "optimizer_calls_zero": all(value == 0 for value in receipt.get("optimizer_calls", {}).values()),
        "no_test_sample_csv_network": receipt.get("test_sample_csv_network_access") is False,
    }
    if not all(checks.values()):
        raise VerificationError(f"Synthetic recovery verification failed: {checks}")
    result = {"status": "PASS_INDEPENDENT_SYNTHETIC_RECOVERY_READBACK",
              "checks": checks, "gamma_file_sha256": _file_sha256(gamma_path),
              "receipt_file_sha256": _file_sha256(receipt_path),
              "target_or_test_values_loaded": False}
    result["verification_sha256"] = _canonical_sha256(result)
    return result


def _array_sha256(array: np.ndarray) -> str:
    header = _canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-source", action="store_true")
    group.add_argument("--verify-official", action="store_true")
    group.add_argument("--verify-synthetic", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_source:
            result = verify_source()
        elif args.verify_official:
            result = verify_official()
        else:
            if args.output_dir is None:
                raise VerificationError("--verify-synthetic requires --output-dir")
            result = verify_synthetic(args.output_dir.resolve())
    except Exception as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_RECOVERY_VERIFICATION",
                          "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
