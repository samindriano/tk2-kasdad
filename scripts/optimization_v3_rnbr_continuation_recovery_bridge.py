"""Post-fit bridge to the previously verified safe-Unicode E67 pilot recovery.

This helper does not modify the frozen pilot runner or continuation recipe. It
allows only the already verified E67-observed recovered artifact to be read in
place of its preserved failed object-dtype serialization; the original remains
untouched. Use it after the new six continuation bundles are complete.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_continuation as runner  # noqa: E402
from scripts import optimization_v3_rnbr_continuation_verify as verifier  # noqa: E402
from scripts import optimization_v3_rnbr_official as pilot  # noqa: E402

RECOVERY_REPORT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/SERIALIZATION_RECOVERY_V2.json"
ADAPTER_FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze_serialization_adapter_02.json"


class BridgeError(RuntimeError):
    """A mismatch in immutable inherited pilot-recovery evidence."""


def assert_frozen_source_bytes(committed: bytes, current: bytes, relative: str) -> None:
    """Compare exact bytes so Windows console decoding cannot hide code drift."""
    if committed != current:
        raise BridgeError(f"Frozen fit source differs from the recorded source commit: {relative}")


def load_verified_e67_observed() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load only safe recovered arrays after validating recovery report and hashes."""
    report = json.loads(RECOVERY_REPORT.read_text(encoding="utf-8"))
    adapter = json.loads(ADAPTER_FREEZE.read_text(encoding="utf-8"))
    detail = report.get("serialization_recovery_detail", {})
    recovery_event = detail.get("recovery_event", {})
    verification = report.get("independent_verification", {})
    frozen = json.loads(runner.PILOT_FREEZE.read_text(encoding="utf-8"))
    if (verification.get("status") != "PASS"
            or verification.get("freeze_signature") != frozen.get("freeze_signature")
            or detail.get("status") != "SAME_FIT_OUTPUT_RECOVERED_NO_REFIT"
            or detail.get("fit_id") != "RNBR_PILOT_E67_OBSERVED"
            or recovery_event.get("refit_performed") is not False
            or recovery_event.get("original_terminal_event_preserved") is not True
            or adapter.get("adapter_signature") != recovery_event.get("serialization_adapter_signature")):
        raise BridgeError("Historical E67 recovery is not verified as the same fit/no-refit")
    bundle = ROOT / recovery_event["recovery_bundle_path"]
    receipt_path = ROOT / recovery_event["recovery_receipt_path"]
    if pilot.sha256_file(bundle) != detail.get("recovery_npz_sha256"):
        raise BridgeError("Recovered E67 NPZ differs from verified recovery report")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    if (receipt.get("identity", {}).get("fit_id") != "RNBR_PILOT_E67_OBSERVED"
            or receipt.get("identity", {}).get("freeze_signature") != frozen.get("freeze_signature")
            or receipt.get("npz_sha256") != detail.get("recovery_npz_sha256")
            or receipt.get("integrity_sha256") != detail.get("recovery_receipt_sha256")
            or receipt.get("identity_sha256") != pilot.sha256_bytes(pilot.canonical_bytes(receipt["identity"]))
            or receipt.get("integrity_sha256") != pilot.sha256_bytes(pilot.canonical_bytes(body))):
        raise BridgeError("Recovered E67 receipt identity/canonical integrity mismatch")
    with np.load(bundle, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in z.files}
    if (any(value.dtype.kind == "O" for value in arrays.values())
            or receipt.get("array_shapes") != {key: list(value.shape) for key, value in arrays.items()}
            or receipt.get("array_dtypes") != {key: str(value.dtype) for key, value in arrays.items()}):
        raise BridgeError("Recovered E67 arrays fail pickle-free dtype/shape checks")
    for key, expected in receipt.get("array_sha256", {}).items():
        if key not in arrays or pilot.sha256_array(arrays[key]) != expected:
            raise BridgeError(f"Recovered E67 per-array hash differs: {key}")
    expected_fit = next((x for x in verification.get("fit_bundle_verification", [])
                         if x.get("fit_id") == "RNBR_PILOT_E67_OBSERVED"), None)
    if (not expected_fit or expected_fit.get("status") != "PASS"
            or expected_fit.get("bundle_npz_sha256") != detail.get("recovery_npz_sha256")
            or expected_fit.get("receipt_sha256") != receipt.get("integrity_sha256")):
        raise BridgeError("Independent pilot verification does not pin recovered E67 artifact")
    return arrays, receipt, {"status": "PASS", "fit_id": "RNBR_PILOT_E67_OBSERVED",
                             "npz_sha256": receipt["npz_sha256"],
                             "receipt_integrity_sha256": receipt["integrity_sha256"],
                             "source_freeze_signature": frozen["freeze_signature"],
                             "adapter_signature": adapter["adapter_signature"],
                             "original_failed_artifact_preserved": True,
                             "refit_performed": False,
                             "query_labels_read_by_bridge": False,
                             "test_or_sample_submission_opened": False}


def precheck_all_inherited_pilot_bundles() -> dict[str, Any]:
    arrays, receipt, bridge = load_verified_e67_observed()
    pilot_freeze = json.loads(runner.PILOT_FREEZE.read_text(encoding="utf-8"))
    safe_ids = {"RNBR_PILOT_E67_OBSERVED": (arrays, receipt)}
    for fit_id in ("RNBR_PILOT_E67_SHUFFLED", "RNBR_PILOT_E75_OBSERVED", "RNBR_PILOT_E75_SHUFFLED"):
        safe_ids[fit_id] = pilot.load_bundle(fit_id, pilot_freeze)
    counts = {}
    for fit_id, (bundle_arrays, _) in safe_ids.items():
        if any(value.dtype.kind == "O" for value in bundle_arrays.values()):
            raise BridgeError(f"Inherited pilot bundle contains an unsafe object array: {fit_id}")
        counts[fit_id] = {"rows": int(len(bundle_arrays["score_rows"])),
                          "prediction_finite": bool(np.isfinite(bundle_arrays["prediction"]).all())}
    return {"status": "PASS", "recovered_E67_observed": bridge,
            "all_four_pilot_bundles_pickle_free_readable": counts,
            "query_labels_read": False, "Test_or_SampleSubmission_opened": False}


def verified_fit_source_commit(freeze: dict[str, Any]) -> str:
    """Recover the immutable fit source commit from all six receipts and pin its code."""
    fit_ids = [f"RNBR_CONT_E{origin}_{arm.upper()}"
               for origin in runner.ORIGINS for arm in runner.ARMS]
    source_commits: set[str] = set()
    for fit_id in fit_ids:
        _, receipt = runner.load_bundle(fit_id, freeze)
        source = receipt.get("identity", {}).get("source_commit")
        if not isinstance(source, str) or len(source) != 40:
            raise BridgeError(f"Missing valid fit source commit in receipt: {fit_id}")
        source_commits.add(source)
    if len(source_commits) != 1:
        raise BridgeError(f"Continuation receipts disagree on fit source commit: {sorted(source_commits)}")
    source_commit = next(iter(source_commits))
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise BridgeError("Fit source commit is not an ancestor of the current review checkpoint") from exc
    pinned_paths = (
        "scripts/optimization_v3_rnbr_continuation.py",
        "scripts/optimization_v3_rnbr_continuation_verify.py",
        "src/models/regression_neighbor_rnbr.py",
        "experiments/optimization_v3/rnbr_continuation_v1/freeze.json",
        "experiments/optimization_v3/rnbr_continuation_v1/protocol.json",
    )
    for relative in pinned_paths:
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        assert_frozen_source_bytes(committed, current, relative)
    return source_commit


def evaluate() -> dict[str, Any]:
    freeze = runner.load_freeze()
    safe_arrays, safe_receipt, bridge = load_verified_e67_observed()
    fit_source_commit = verified_fit_source_commit(freeze)
    old_loader = pilot.load_bundle

    def load_with_recovered_e67(fit_id: str, pilot_freeze: Any, *, source_commit: str | None = None):
        if fit_id == "RNBR_PILOT_E67_OBSERVED":
            if pilot_freeze.get("freeze_signature") != safe_receipt["identity"].get("freeze_signature"):
                raise BridgeError("Requested pilot freeze differs from recovered E67 artifact")
            if source_commit is not None and source_commit != safe_receipt["identity"].get("source_commit"):
                raise BridgeError("Unexpected source commit requested for recovered E67 artifact")
            return safe_arrays, safe_receipt
        return old_loader(fit_id, pilot_freeze, source_commit=source_commit)

    pilot.load_bundle = load_with_recovered_e67
    try:
        bridge_commit = pilot._run_git("rev-parse", "HEAD")
        result = verifier.write_reports(freeze, fit_source_commit)
    finally:
        pilot.load_bundle = old_loader
    out = ROOT / "reports/optimization_v3/rnbr_continuation_v1/recovery_bridge_verification.json"
    payload = {"status": "PASS", "fit_source_commit": fit_source_commit,
               "postfit_bridge_commit": bridge_commit,
               "bridge": bridge, "continuation_freeze_signature": freeze["freeze_signature"],
               "evaluation_status": result["status"], "gate_pass": result["gate_pass"],
               "query_targets_accessed_after_new_fits_verified": True}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--check-recovered", "--evaluate"}:
        print("Usage: python scripts/optimization_v3_rnbr_continuation_recovery_bridge.py --check-recovered|--evaluate", file=sys.stderr)
        return 2
    try:
        if sys.argv[1] == "--check-recovered":
            receipt = precheck_all_inherited_pilot_bundles()
            out = ROOT / "reports/optimization_v3/rnbr_continuation_v1/recovery_bridge_precheck.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(receipt, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
