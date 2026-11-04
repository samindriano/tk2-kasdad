"""Versioned, reader-only recovery for the CLS_POS_A02_META_V1 R2 receipt.

The frozen runner remains unchanged. In scikit-learn 1.8, its exact frozen
``penalty='l2'`` recipe records a deprecation FutureWarning. R2's reader
mistakenly treated every warning class as a convergence warning. This adapter
accepts only that source-verified FutureWarning, while keeping the original
numeric, shape, iteration, state-hash, receipt, ledger, and fit-signature gates.
It never fits a model itself; ``--resume-approved-run`` delegates to the frozen
runner, which reuses the durable D1_PAIR state and performs only the remaining
predeclared fits if all frozen checks pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from scripts import optimization_v3_cls_pos_a02_meta_v1 as frozen
from src.models.research_next_contract import file_sha256

ROOT = frozen.ROOT
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests/test_optimization_v3_cls_pos_a02_meta_v1_warning_recovery.py"
RECOVERY_PREFLIGHT = frozen.REPORT_DIR / "RUNTIME_RECOVERY_PREFLIGHT_R1.json"
EXPECTED_FROZEN_SCRIPT_SHA256 = "ffbc29346a53900f2b12da0d713f7ad23e60d469c50ac7d16e1b7de921bcc740"
EXPECTED_FREEZE_RECEIPT_SHA256 = "b1bf3398a37400ed0854c9b3805fd14e33ff2dc98a9025597c8e7bfc01bb0441"
EXPECTED_APPROVAL_SHA256 = "6bee52699e4b6622d37608ca3b0ac591399ab69badf7111895e5234e84604344"
SUPPORTED_SKLEARN_VERSION = "1.8.0"
PENALTY_DEPRECATION_MARKER = "'penalty' was deprecated in version 1.8"


class RecoveryError(RuntimeError):
    """Fail-closed error for recovery provenance or model-state validation."""


def _logistic_source() -> tuple[Path, str, str]:
    path = Path(inspect.getfile(LogisticRegression)).resolve()
    payload = path.read_bytes()
    source = payload.decode("utf-8")
    return path, hashlib.sha256(payload).hexdigest(), source


def _supported_futurewarning(receipt: Mapping[str, Any], *,
                             sklearn_version: str,
                             logistic_source: str) -> bool:
    """Identify only the known 1.8 explicit-penalty deprecation warning."""
    if sklearn_version != SUPPORTED_SKLEARN_VERSION:
        return False
    if receipt.get("estimator") != frozen.MODEL_PARAMS:
        return False
    marker_at = logistic_source.find(PENALTY_DEPRECATION_MARKER)
    if marker_at < 0:
        return False
    source_window = logistic_source[max(0, marker_at - 180):marker_at + 500]
    return "warnings.warn(" in source_window and "FutureWarning" in source_window


def read_model_state(path: Path, receipt: Mapping[str, Any], *,
                     sklearn_version: str | None = None,
                     logistic_source: str | None = None) -> dict[str, np.ndarray]:
    """Read a persisted state with a narrow, source-verified warning allowance."""
    path = Path(path)
    if file_sha256(path) != receipt.get("model_state_file_sha256"):
        raise RecoveryError("Model state file hash differs from fit receipt")
    with np.load(path, allow_pickle=False) as saved:
        required = ("coef", "intercept", "classes", "n_iter")
        if not set(required).issubset(saved.files):
            raise RecoveryError("Saved model state lacks required arrays")
        state = {key: np.asarray(saved[key]).copy() for key in required}
    if not all(np.isfinite(np.asarray(state[k], dtype=np.float64)).all()
               for k in ("coef", "intercept")):
        raise RecoveryError("Saved model state contains nonfinite parameters")
    if (state["coef"].shape != (1, int(receipt["feature_width"]))
            or state["intercept"].shape != (1,)
            or state["classes"].shape != (2,)):
        raise RecoveryError("Saved model coefficient shape is invalid")
    if not np.array_equal(state["classes"], np.asarray([0, 1], dtype=state["classes"].dtype)):
        raise RecoveryError("Saved model class order is not [0,1]")
    n_iter = np.asarray(state["n_iter"])
    if n_iter.shape != (1,) or int(n_iter[0]) < 0 or int(n_iter[0]) >= frozen.MAX_ITER:
        raise RecoveryError("Saved model iteration count is invalid or did not converge")
    if receipt.get("n_iter") != [int(n_iter[0])]:
        raise RecoveryError("Saved model iteration count does not match its receipt")

    warning_categories = receipt.get("convergence_warnings")
    if not isinstance(warning_categories, list) or any(not isinstance(x, str) for x in warning_categories):
        raise RecoveryError("Fit receipt warning-category metadata is invalid")
    if "ConvergenceWarning" in warning_categories:
        raise RecoveryError("ConvergenceWarning is never recoverable")
    if warning_categories == []:
        return state
    if warning_categories != ["FutureWarning"]:
        raise RecoveryError(f"Unexpected warning categories are not recoverable: {warning_categories}")
    if sklearn_version is None or logistic_source is None:
        current_path, current_sha, current_source = _logistic_source()
        del current_path, current_sha
        sklearn_version = sklearn.__version__
        logistic_source = current_source
    if not _supported_futurewarning(receipt, sklearn_version=sklearn_version,
                                    logistic_source=logistic_source):
        raise RecoveryError("FutureWarning does not match the verified sklearn 1.8 penalty deprecation")
    return state


def _verify_d1_pair_checkpoint() -> dict[str, Any]:
    freeze = frozen._validate_prefit_freeze()
    if freeze.get("receipt_sha256") != EXPECTED_FREEZE_RECEIPT_SHA256:
        raise RecoveryError("R2 prefit freeze receipt differs from the reviewed checkpoint")
    if frozen.file_sha256(frozen.SCRIPT) != EXPECTED_FROZEN_SCRIPT_SHA256:
        raise RecoveryError("Frozen R2 training script changed; reader recovery cannot proceed")
    approval, approval_sha = frozen._require_root_approval()
    if approval_sha != EXPECTED_APPROVAL_SHA256 or approval.get("status") != "PASS":
        raise RecoveryError("Root approval differs from the reviewed R2 binding")

    ledger_events = frozen._ledger()
    grouped = frozen._validate_fit_ledger(ledger_events)
    if set(grouped) != {"D1_PAIR"}:
        raise RecoveryError(f"Unexpected official fit IDs at recovery checkpoint: {sorted(grouped)}")
    d1 = grouped["D1_PAIR"]
    if set(d1) != {"RESERVE", "FIT_START", "FIT_COMPLETE"}:
        raise RecoveryError("D1_PAIR ledger is not a single complete fit with no failure")
    receipt_path = frozen._fit_paths("D1_PAIR")[1]
    state_path = frozen._fit_paths("D1_PAIR")[0]
    receipt = frozen._receipt(receipt_path)
    complete = d1["FIT_COMPLETE"]
    if complete.get("receipt_sha256") != receipt["receipt_sha256"]:
        raise RecoveryError("D1_PAIR completion event does not bind the durable receipt")
    current_path, current_source_sha, source = _logistic_source()
    state = read_model_state(state_path, receipt,
                             sklearn_version=sklearn.__version__,
                             logistic_source=source)
    if receipt.get("convergence_warnings") != ["FutureWarning"]:
        raise RecoveryError("D1_PAIR is not the single reviewed warning-metadata case")
    if receipt.get("n_iter") != [6] or state["n_iter"].tolist() != [6]:
        raise RecoveryError("D1_PAIR convergence state differs from the reviewed six-iteration fit")
    if frozen.E79_FREEZE.exists():
        raise RecoveryError("E79 application freeze unexpectedly exists before final states")
    if frozen.PAIR_CSV.exists():
        raise RecoveryError("A PAIR candidate CSV unexpectedly exists before final-state inference")
    if frozen.PREFIT_GATE_CHECK.exists() is False:
        raise RecoveryError("R2 fail-closed gate evidence is missing")
    frozen.resource_gate()

    return {
        "schema_version": 1,
        "experiment_id": frozen.EXPERIMENT,
        "status": "RUNTIME_RECOVERY_PREFLIGHT_PASS",
        "recovery_id": "CLS_POS_A02_META_V1_WARNING_RECOVERY_R1",
        "authorization": "Sol root approved a separate, reader-only repair; no recipe change or refit.",
        "source_commit": frozen._git_head(),
        "frozen_prefit_commit": freeze["prefit_commit"],
        "frozen_prefit_script_sha256": EXPECTED_FROZEN_SCRIPT_SHA256,
        "prefit_freeze_receipt_sha256": freeze["receipt_sha256"],
        "root_approval_sha256": approval_sha,
        "recovery_adapter_sha256": file_sha256(SCRIPT),
        "recovery_test_sha256": file_sha256(TESTS),
        "sklearn_warning_provenance": {
            "version": sklearn.__version__,
            "source_path": str(current_path),
            "source_file_sha256": current_source_sha,
            "source_marker": PENALTY_DEPRECATION_MARKER,
            "source_evidence": "LogisticRegression.fit warns FutureWarning for explicit penalty in sklearn 1.8; frozen config is penalty='l2'.",
        },
        "repair_rule": {
            "accept": "no warnings, or exactly one FutureWarning when sklearn is 1.8.0, frozen estimator parameters match, and installed source verifies the explicit-penalty deprecation",
            "reject": "ConvergenceWarning, unknown/duplicate warning categories, nonfinite parameters, invalid shapes/classes, n_iter mismatch, n_iter >= max_iter, or state/receipt hash mismatch",
            "frozen_recipe_changed": False,
            "original_fit_refit": False,
        },
        "d1_pair": {
            "fit_receipt_sha256": receipt["receipt_sha256"],
            "model_state_file_sha256": receipt["model_state_file_sha256"],
            "fit_signature_sha256": receipt["fit_signature_sha256"],
            "fit_complete_receipt_sha256": complete["receipt_sha256"],
            "ledger_sha256": file_sha256(frozen.LEDGER),
            "n_iter": receipt["n_iter"],
            "warning_categories": receipt["convergence_warnings"],
            "fit_rows": receipt["fit_rows"],
            "selection_cycle_range": receipt["selection_cycle_range"],
            "fit_refit": False,
            "evaluation_labels_read": False,
        },
        "official_fit_accounting_before_resume": {
            "fit_starts": 1,
            "fit_completions": 1,
            "fit_failures": 0,
            "fit_ids": ["D1_PAIR"],
            "remaining_predeclared_fits": 5,
            "official_fit_cap": frozen.MAX_OFFICIAL_FITS,
        },
        "test_probabilities_loaded": False,
        "sample_submission_loaded": False,
        "kaggle_upload": False,
    }


def preflight() -> dict[str, Any]:
    body = _verify_d1_pair_checkpoint()
    receipt_sha = frozen._write_json_once(RECOVERY_PREFLIGHT, body)
    return {**body, "receipt_sha256": receipt_sha}


def _validate_preflight_receipt() -> dict[str, Any]:
    report = frozen._receipt(RECOVERY_PREFLIGHT)
    if report.get("status") != "RUNTIME_RECOVERY_PREFLIGHT_PASS":
        raise RecoveryError("Recovery preflight report is not PASS")
    if report.get("recovery_adapter_sha256") != file_sha256(SCRIPT):
        raise RecoveryError("Recovery adapter changed after preflight")
    if report.get("recovery_test_sha256") != file_sha256(TESTS):
        raise RecoveryError("Recovery tests changed after preflight")
    if not frozen._git_ancestor(report.get("source_commit", ""), frozen._git_head()):
        raise RecoveryError("Recovery source commit is not an ancestor of current HEAD")
    checkpoint = _verify_d1_pair_checkpoint()
    for key in ("prefit_freeze_receipt_sha256", "root_approval_sha256", "recovery_adapter_sha256",
                "recovery_test_sha256", "sklearn_warning_provenance"):
        if report.get(key) != checkpoint.get(key):
            raise RecoveryError(f"Recovery preflight binding changed: {key}")
    d1 = report["d1_pair"]
    if (d1.get("fit_receipt_sha256") != checkpoint["d1_pair"]["fit_receipt_sha256"]
            or d1.get("model_state_file_sha256") != checkpoint["d1_pair"]["model_state_file_sha256"]
            or d1.get("ledger_sha256") != checkpoint["d1_pair"]["ledger_sha256"]):
        raise RecoveryError("D1_PAIR recovery checkpoint changed after preflight")
    return report


@contextmanager
def _patched_reader() -> Iterator[None]:
    old_reader = frozen._read_model_state
    frozen._read_model_state = read_model_state
    try:
        yield
    finally:
        frozen._read_model_state = old_reader


def resume_approved_run() -> dict[str, Any]:
    _validate_preflight_receipt()
    with _patched_reader():
        return frozen.run_official()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--resume-approved-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = preflight() if args.preflight else resume_approved_run()
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "RECOVERY_BLOCKED_OR_FAILED",
                          "type": type(exc).__name__, "message": str(exc)}, indent=2),
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
