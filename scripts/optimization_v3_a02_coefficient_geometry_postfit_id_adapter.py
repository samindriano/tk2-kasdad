"""Authorized zero-fit adapter for the CGEOM_V1 expected record-ID dtype.

The frozen evaluator remains unchanged. This adapter converts only source-derived
expected query IDs from pandas object dtype to lossless NumPy Unicode before the
frozen verifier's strict stored-array checks. It also validates original frozen
Git blobs separately from the later amendment evaluator commit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.models import research_next_contract  # noqa: E402
import optimization_v3_positional_text_pilot_v1 as positional  # noqa: E402
import optimization_v3_a02_coefficient_geometry_v1 as runner  # noqa: E402
import verify_optimization_v3_a02_coefficient_geometry_v1 as frozen_verifier  # noqa: E402


AMENDMENT_ID = "CGEOM_V1_POSTFIT_ID_VERIFIER_V1"
AUTHORIZATION_PATH = ROOT / "reports/optimization_v3/a02_coefficient_geometry_v1/POSTFIT_ID_VERIFIER_AUTHORIZATION.md"
ADAPTER_PATH = ROOT / "scripts/optimization_v3_a02_coefficient_geometry_postfit_id_adapter.py"
TEST_PATH = ROOT / "tests/test_optimization_v3_a02_coefficient_geometry_postfit_id_adapter.py"
AMENDMENT_RUN_ROOT = runner.RUN_ROOT / "postfit_id_verifier_amendment"
FAILURE_PATH = AMENDMENT_RUN_ROOT / "POSTFIT_ID_FAILURE.json"
QA_PATH = AMENDMENT_RUN_ROOT / "SYNTHETIC_QA.json"
AMENDMENT_SOURCE_PATH = AMENDMENT_RUN_ROOT / "AMENDMENT_SOURCE.json"
AMENDMENT_RECEIPT_PATH = AMENDMENT_RUN_ROOT / "POSTFIT_ID_AMENDMENT.json"
SCORE_RECEIPT_PATH = AMENDMENT_RUN_ROOT / "POSTFIT_SCORE_RECEIPT.json"
EVALUATION_RECEIPT_PATH = AMENDMENT_RUN_ROOT / "POSTFIT_EVALUATION_RECEIPT.json"
REPORT_RECEIPT_PATH = AMENDMENT_RUN_ROOT / "POSTFIT_REPORT_RECEIPT.json"
COMPLETION_RECEIPT_PATH = AMENDMENT_RUN_ROOT / "POSTFIT_COMPLETION_RECEIPT.json"
EXPECTED_PRECOMPLETION_LEDGER_SHA256 = "bcedd3b589c3c496176fa24e0f5d4da316907541389cecbd8361ce133a5179e7"
ORIGINAL_SOURCE_COMMIT = "ddd64736155a64e70c5453c921aedcbd910397f8"
ATOL = 1e-12


class AmendmentError(RuntimeError):
    """A frozen source, output, or amendment boundary failed closed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AmendmentError(f"Required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AmendmentError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    runner.write_create_once(path, runner.json_bytes(dict(value)))


def explicit_unicode_ids(values: Any) -> np.ndarray:
    """Convert source-derived, one-dimensional string IDs losslessly to dtype U."""
    source = np.asarray(values)
    if source.ndim != 1 or not len(source):
        raise AmendmentError("Expected record IDs must be a nonempty one-dimensional vector")
    items = source.tolist()
    if any(not isinstance(item, str) for item in items):
        raise AmendmentError("Expected record IDs must already be non-null strings")
    result = np.asarray(items, dtype=str)
    if result.dtype.kind != "U" or result.tolist() != items:
        raise AmendmentError("Expected record-ID Unicode conversion was lossy")
    return result


def assert_array_match(field: str, expected: Any, stored: Any) -> None:
    """Apply the frozen shape/dtype/value policy, with one ID-field adapter."""
    expected_array = explicit_unicode_ids(expected) if field == "query_record_ids" else np.asarray(expected)
    stored_array = np.asarray(stored)
    if stored_array.dtype.kind == "O":
        raise AmendmentError(f"Stored object array is prohibited: {field}")
    if expected_array.dtype.kind == "O":
        raise AmendmentError(f"Expected object array was not normalized: {field}")
    if stored_array.shape != expected_array.shape:
        raise AmendmentError(f"Stored shape mismatch for {field}: {stored_array.shape} != {expected_array.shape}")
    if stored_array.dtype != expected_array.dtype:
        raise AmendmentError(f"Stored dtype mismatch for {field}: {stored_array.dtype} != {expected_array.dtype}")
    if expected_array.dtype.kind in "f":
        if not np.allclose(stored_array, expected_array, atol=ATOL, rtol=0, equal_nan=False):
            raise AmendmentError(f"Stored numeric values differ for {field} beyond {ATOL}")
    elif not np.array_equal(stored_array, expected_array):
        raise AmendmentError(f"Stored values or order differ for {field}")


def adapt_source_ids(source: Mapping[str, Any]) -> dict[str, Any]:
    """Copy source mapping and convert only score_ids; every other field is untouched."""
    adapted = dict(source)
    adapted["score_ids"] = explicit_unicode_ids(source["score_ids"])
    return adapted


def _amendment_source_identity() -> dict[str, Any]:
    current_commit = _git("rev-parse", "HEAD")
    adapter_hash = sha256_file(ADAPTER_PATH)
    test_hash = sha256_file(TEST_PATH)
    authorization_hash = sha256_file(AUTHORIZATION_PATH)
    return {
        "amendment_id": AMENDMENT_ID,
        "amendment_source_commit": current_commit,
        "adapter_path": ADAPTER_PATH.relative_to(ROOT).as_posix(),
        "adapter_sha256": adapter_hash,
        "test_path": TEST_PATH.relative_to(ROOT).as_posix(),
        "test_sha256": test_hash,
        "authorization_path": AUTHORIZATION_PATH.relative_to(ROOT).as_posix(),
        "authorization_sha256": authorization_hash,
        "original_frozen_source_commit": ORIGINAL_SOURCE_COMMIT,
    }


def verify_original_lineage(*, require_adapter_qa: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify clean amendment checkout and all original frozen bytes and Git blobs."""
    runner.require_clean_tree()
    frozen_raw = _json(runner.FREEZE_PATH)
    frozen = dict(frozen_raw)
    signature = frozen.pop("freeze_signature", None)
    if signature != runner.canonical_sha256(frozen):
        raise AmendmentError("Original CGEOM freeze signature does not verify")
    frozen["freeze_signature"] = signature
    current_commit = _git("rev-parse", "HEAD")
    if frozen.get("source_commit") != ORIGINAL_SOURCE_COMMIT:
        raise AmendmentError("Original fitted source commit differs from the authorized amendment base")
    if subprocess.run(["git", "merge-base", "--is-ancestor", ORIGINAL_SOURCE_COMMIT, current_commit], cwd=ROOT).returncode != 0:
        raise AmendmentError("Frozen training source is not an ancestor of the amendment evaluator")
    if runner.source_hashes() != frozen.get("source_hashes"):
        raise AmendmentError("A frozen source file changed in the working tree")
    blob_ids: dict[str, str] = {}
    for relative, expected_hash in frozen["source_hashes"].items():
        path = ROOT / relative
        if sha256_file(path) != expected_hash:
            raise AmendmentError(f"Working source hash differs from frozen identity: {relative}")
        original_blob = subprocess.check_output(["git", "show", f"{ORIGINAL_SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        if sha256_bytes(original_blob) != expected_hash:
            raise AmendmentError(f"Original Git blob bytes differ from recorded frozen hash: {relative}")
        blob_ids[relative] = _git("rev-parse", f"{ORIGINAL_SOURCE_COMMIT}:{relative}")
    if sha256_file(runner.TRAIN) != runner.TRAIN_SHA256 or frozen.get("input_sha256") != runner.TRAIN_SHA256:
        raise AmendmentError("Official Classification Train hash differs from the freeze")
    if sha256_file(runner.RECIPE_PATH) != frozen.get("recipe_sha256"):
        raise AmendmentError("Frozen recipe hash changed")
    preflight = _json(runner.PREFLIGHT_PATH)
    synthetic = _json(runner.SYNTHETIC_QA_PATH)
    if (preflight.get("source_commit") != ORIGINAL_SOURCE_COMMIT
            or preflight.get("source_hashes") != frozen["source_hashes"]
            or preflight.get("synthetic_qa_sha256") != sha256_file(runner.SYNTHETIC_QA_PATH)
            or synthetic.get("source_commit") != ORIGINAL_SOURCE_COMMIT
            or synthetic.get("source_hashes") != frozen["source_hashes"]
            or preflight.get("status") != "PASS_TARGET_FREE_PREFIT_RESOURCE_AND_LINEAGE_PREFLIGHT"
            or synthetic.get("status") != "PASS_SYNTHETIC_END_TO_END_AND_PERSISTENCE_RESUME"):
        raise AmendmentError("Original target-free preflight or synthetic QA no longer matches the freeze")
    amendment_source = _amendment_source_identity()
    amendment_source["original_source_hashes"] = dict(frozen["source_hashes"])
    amendment_source["original_git_blob_ids"] = blob_ids
    amendment_source["original_verifier_path"] = "scripts/verify_optimization_v3_a02_coefficient_geometry_v1.py"
    amendment_source["original_verifier_sha256"] = frozen["source_hashes"][amendment_source["original_verifier_path"]]
    amendment_source["amendment_source_identity_sha256"] = runner.canonical_sha256(amendment_source)
    if require_adapter_qa:
        qa = _json(QA_PATH)
        if (qa.get("status") != "PASS_POSTFIT_ID_ADAPTER_SYNTHETIC_AND_PRODUCTION_READBACK"
                or qa.get("amendment_source_identity_sha256") != amendment_source["amendment_source_identity_sha256"]):
            raise AmendmentError("Postfit adapter QA does not bind the current committed adapter")
    return frozen, amendment_source


@contextmanager
def frozen_evaluation_context(frozen: Mapping[str, Any]) -> Iterator[None]:
    """Supply only the independently revalidated original freeze to frozen routines."""
    old = runner.load_freeze
    runner.load_freeze = lambda: dict(frozen)  # type: ignore[assignment]
    try:
        yield
    finally:
        runner.load_freeze = old  # type: ignore[assignment]


@contextmanager
def unicode_expected_ids_context() -> Iterator[None]:
    """Patch only source-derived expected score_ids during frozen reconstruction."""
    old = runner.load_origin_sources

    def adapted_loader(frame: Any, origin: int) -> dict[str, Any]:
        return adapt_source_ids(old(frame, origin))

    runner.load_origin_sources = adapted_loader  # type: ignore[assignment]
    try:
        yield
    finally:
        runner.load_origin_sources = old  # type: ignore[assignment]


def production_schema_readback(frame: Any) -> list[dict[str, Any]]:
    """Read immutable NPZ payloads with allow_pickle=False and check source IDs."""
    if "artifact_signal" in frame.columns or "threat_rank" in frame.columns:
        raise AmendmentError("Production readback must use feature-only Train")
    readbacks = []
    for fit_id in runner.FIT_IDS:
        origin = int(fit_id.split("_E", 1)[1][:2])
        arm = fit_id.rsplit("_", 1)[1].lower()
        source = runner.load_origin_sources(frame, origin)
        artifact_path, receipt_path = runner._unit_paths(fit_id)
        arrays = frozen_verifier._npz(artifact_path)
        expected_ids = explicit_unicode_ids(source["score_ids"])
        assert_array_match("query_record_ids", expected_ids, arrays["query_record_ids"])
        receipt = _json(receipt_path)
        if (receipt.get("fit_id") != fit_id or receipt.get("origin") != origin
                or receipt.get("arm") != arm or receipt.get("artifact_sha256") != sha256_file(artifact_path)):
            raise AmendmentError(f"Production receipt mismatch for {fit_id}")
        readbacks.append({
            "fit_id": fit_id, "shape": list(arrays["query_record_ids"].shape),
            "stored_id_dtype": str(arrays["query_record_ids"].dtype),
            "source_id_dtype_before_adapter": str(np.asarray(source["score_ids"]).dtype),
            "id_value_order_match": True, "object_members_rejected_by_frozen_reader": True,
            "artifact_sha256": sha256_file(artifact_path), "receipt_sha256": sha256_file(receipt_path),
        })
    return readbacks


def _failure_diagnostic(frozen: Mapping[str, Any], amendment_source: Mapping[str, Any],
                        readbacks: list[dict[str, Any]]) -> dict[str, Any]:
    first = readbacks[0]
    diagnostic = {
        "status": "PRESERVED_AND_REPRODUCED_EXPECTED_ID_DTYPE_DIAGNOSTIC",
        "original_failure": "Stored production output shape/dtype mismatch at E67/aligned/query_record_ids",
        "original_frozen_verifier_sha256": amendment_source["original_verifier_sha256"],
        "stored_dtype": first["stored_id_dtype"],
        "expected_dtype_before_amendment": first["source_id_dtype_before_adapter"],
        "shape": first["shape"], "source_id_values_and_order_match": True,
        "only_expected_query_id_representation_changed": True,
        "all_four_production_units_readback": readbacks,
        "labels_opened": False, "input_sha256": runner.TRAIN_SHA256,
        "freeze_signature": frozen["freeze_signature"],
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(FAILURE_PATH, diagnostic)
    return diagnostic


def run_synthetic_qa() -> dict[str, Any]:
    frozen, amendment_source = verify_original_lineage()
    if runner.PRELABEL_PATH.exists() or (runner.RUN_ROOT / "QUERY_LABEL_ACCESS.json").exists() or runner.RESULT_PATH.exists():
        raise AmendmentError("Synthetic QA is permitted only before prelabel or label scoring")
    ledger_before = sha256_file(runner.LEDGER)
    if ledger_before != EXPECTED_PRECOMPLETION_LEDGER_SHA256:
        raise AmendmentError("Shared ledger differs before adapter synthetic QA")
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_optimization_v3_a02_coefficient_geometry_postfit_id_adapter.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if process.returncode != 0:
        raise AmendmentError("Focused postfit synthetic QA failed:\n" + (process.stdout + process.stderr)[-6000:])
    frame = positional.load_features()
    readbacks = production_schema_readback(frame)
    diagnostic = _failure_diagnostic(frozen, amendment_source, readbacks)
    ledger_after = sha256_file(runner.LEDGER)
    if ledger_after != ledger_before or {"artifact_signal", "threat_rank"} & set(frame.columns):
        raise AmendmentError("Adapter synthetic/readback changed ledger or loaded targets")
    summary_line = next((line.strip() for line in process.stdout.splitlines() if " passed" in line), "tests passed")
    qa = {
        "status": "PASS_POSTFIT_ID_ADAPTER_SYNTHETIC_AND_PRODUCTION_READBACK",
        "amendment_id": AMENDMENT_ID, "amendment_source": amendment_source,
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "frozen_source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": runner.TRAIN_SHA256, "ledger_sha256_before": ledger_before,
        "ledger_sha256_after": ledger_after, "ledger_unchanged": True,
        "target_columns_loaded": [], "query_labels_opened": False,
        "pytest_exit_code": process.returncode, "pytest_summary": summary_line,
        "production_schema_readback": readbacks,
        "preserved_failure_diagnostic_sha256": sha256_file(FAILURE_PATH),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(QA_PATH, qa)
    return qa


def run_prelabel() -> dict[str, Any]:
    frozen, amendment_source = verify_original_lineage(require_adapter_qa=True)
    if runner.PRELABEL_PATH.exists():
        old = _json(runner.PRELABEL_PATH)
        if (old.get("status") == "PASS_INDEPENDENT_NUMERICAL_AND_BOUNDARY_VERIFICATION"
                and old.get("amendment_source_identity_sha256") == amendment_source["amendment_source_identity_sha256"]
                and sha256_file(runner.PRELABEL_PATH) == _json(AMENDMENT_RECEIPT_PATH).get("prelabel_sha256")):
            return old
        raise AmendmentError("Existing prelabel artifact is not an identical amended PASS; preserve and stop")
    if ((runner.RUN_ROOT / "QUERY_LABEL_ACCESS.json").exists() or runner.RESULT_PATH.exists()
            or sha256_file(runner.LEDGER) != EXPECTED_PRECOMPLETION_LEDGER_SHA256):
        raise AmendmentError("Labels may be scored only after a clean, unchanged prelabel state")
    frame = positional.load_features()
    if {"artifact_signal", "threat_rank"} & set(frame.columns):
        raise AmendmentError("Prelabel must not load target columns")
    ledger_before = sha256_file(runner.LEDGER)
    ledger_snapshot_before = runner._budget_snapshot(runner._ledger_events())
    identities = []
    with frozen_evaluation_context(frozen), unicode_expected_ids_context():
        for origin in runner.ORIGINS:
            for arm in runner.ARMS:
                identities.append(frozen_verifier._reconstruct_origin(frame, origin, arm))
    events = runner._ledger_events()
    snapshot = runner._assert_budget(events, phase="after_all_fits")
    own = runner._own_events(events)
    expected_ids = set(runner.FIT_IDS)
    if (snapshot["fit_starts"] != 118 or snapshot["fit_saved"] != 114
            or snapshot["fit_failed"] != 4 or snapshot["fit_pending"] != 0
            or {event["fit_id"] for event in own["FIT_STARTED"]} != expected_ids
            or {event["fit_id"] for event in own["FIT_SAVED"]} != expected_ids
            or own["FIT_FAILED"] or len(identities) != 4):
        raise AmendmentError(f"Independent fit/budget accounting failed: {snapshot}")
    ledger_after = sha256_file(runner.LEDGER)
    if ledger_after != ledger_before or snapshot != ledger_snapshot_before:
        raise AmendmentError("Shared ledger changed during zero-fit prelabel verification")
    artifacts = {item["fit_id"]: item["artifact_sha256"] for item in identities}
    receipts = {item["fit_id"]: item["receipt_sha256"] for item in identities}
    prelabel = {
        "status": "PASS_INDEPENDENT_NUMERICAL_AND_BOUNDARY_VERIFICATION",
        "pilot_id": runner.PILOT_ID, "source_commit": frozen["source_commit"],
        "source_hashes": frozen["source_hashes"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": runner.TRAIN_SHA256,
        "preflight_sha256": sha256_file(runner.PREFLIGHT_PATH),
        "synthetic_qa_sha256": sha256_file(runner.SYNTHETIC_QA_PATH),
        "ledger_sha256_before": ledger_before, "ledger_sha256": ledger_after,
        "ledger_snapshot": snapshot, "independent_unit_reconstructions": identities,
        "unit_artifact_sha256": artifacts, "unit_receipt_sha256": receipts,
        "labels_opened": False, "test_or_sample_submission_opened": False,
        "verification_amendment_id": AMENDMENT_ID,
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "amendment_source": amendment_source,
        "id_expected_dtype_adapter": "source-derived expected query_record_ids converted losslessly to NumPy Unicode only",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    prelabel_bytes = runner.json_bytes(prelabel)
    prelabel_sha = sha256_bytes(prelabel_bytes)
    amendment_receipt = {
        "status": "PASS_ZERO_FIT_POSTFIT_ID_VERIFIER_AMENDMENT",
        "amendment_id": AMENDMENT_ID,
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "amendment_source": amendment_source,
        "original_frozen_source_commit": frozen["source_commit"],
        "freeze_signature": frozen["freeze_signature"], "input_sha256": runner.TRAIN_SHA256,
        "preserved_failure_diagnostic_sha256": sha256_file(FAILURE_PATH),
        "original_verifier_sha256": amendment_source["original_verifier_sha256"],
        "prelabel_sha256": prelabel_sha,
        "unit_artifact_sha256": artifacts, "unit_receipt_sha256": receipts,
        "ledger_sha256_before": ledger_before, "ledger_sha256_after": ledger_after,
        "fit_count": 0, "new_fit_reservation_count": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(runner.PRELABEL_PATH, prelabel)
    _write_once(AMENDMENT_RECEIPT_PATH, amendment_receipt)
    return prelabel


def _verify_prelabel_amendment(frozen: Mapping[str, Any], amendment_source: Mapping[str, Any]) -> dict[str, Any]:
    prelabel = _json(runner.PRELABEL_PATH)
    receipt = _json(AMENDMENT_RECEIPT_PATH)
    if (prelabel.get("status") != "PASS_INDEPENDENT_NUMERICAL_AND_BOUNDARY_VERIFICATION"
            or prelabel.get("labels_opened") is not False
            or prelabel.get("source_commit") != frozen["source_commit"]
            or prelabel.get("freeze_signature") != frozen["freeze_signature"]
            or prelabel.get("amendment_source_identity_sha256") != amendment_source["amendment_source_identity_sha256"]
            or receipt.get("prelabel_sha256") != sha256_file(runner.PRELABEL_PATH)
            or receipt.get("amendment_source_identity_sha256") != amendment_source["amendment_source_identity_sha256"]):
        raise AmendmentError("Postfit prelabel PASS/receipt does not bind the verified adapter and original freeze")
    return prelabel


def score_original_once() -> dict[str, Any]:
    frozen, amendment_source = verify_original_lineage(require_adapter_qa=True)
    _verify_prelabel_amendment(frozen, amendment_source)
    query_access = runner.RUN_ROOT / "QUERY_LABEL_ACCESS.json"
    if query_access.exists() or runner.RESULT_PATH.exists() or SCORE_RECEIPT_PATH.exists():
        raise AmendmentError("Scoring was already started or is ambiguous; do not reopen query labels")
    before = sha256_file(runner.LEDGER)
    if before != EXPECTED_PRECOMPLETION_LEDGER_SHA256:
        raise AmendmentError("Shared ledger changed before original scoring")
    with frozen_evaluation_context(frozen):
        result = runner.score_predictions()
    if not query_access.is_file() or not runner.RESULT_PATH.is_file():
        raise AmendmentError("Original scorer did not produce its expected label-access/result receipts")
    receipt = {
        "status": "PASS_ORIGINAL_FROZEN_SCORING_AFTER_AMENDED_PRELABEL_PASS",
        "amendment_id": AMENDMENT_ID,
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "amendment_source": amendment_source,
        "prelabel_sha256": sha256_file(runner.PRELABEL_PATH),
        "query_label_access_sha256": sha256_file(query_access),
        "result_sha256": sha256_file(runner.RESULT_PATH),
        "freeze_signature": frozen["freeze_signature"],
        "source_commit": frozen["source_commit"], "input_sha256": runner.TRAIN_SHA256,
        "ledger_sha256_before": before, "ledger_sha256_after": sha256_file(runner.LEDGER),
        "query_labels_opened_after_independent_prelabel_pass": True,
        "scoring_policy": "original frozen runner.score_predictions, unchanged",
        "status_from_original_result": result["status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if receipt["ledger_sha256_after"] != before:
        raise AmendmentError("Original scoring unexpectedly changed the shared budget ledger")
    _write_once(SCORE_RECEIPT_PATH, receipt)
    return receipt


def verify_scored_once() -> dict[str, Any]:
    frozen, amendment_source = verify_original_lineage(require_adapter_qa=True)
    _verify_prelabel_amendment(frozen, amendment_source)
    score_receipt = _json(SCORE_RECEIPT_PATH)
    if (score_receipt.get("result_sha256") != sha256_file(runner.RESULT_PATH)
            or score_receipt.get("query_label_access_sha256") != sha256_file(runner.RUN_ROOT / "QUERY_LABEL_ACCESS.json")
            or score_receipt.get("ledger_sha256_after") != sha256_file(runner.LEDGER)):
        raise AmendmentError("Scored result or access receipt changed after the original scoring call")
    if runner.SCORED_VERIFICATION_PATH.exists():
        if EVALUATION_RECEIPT_PATH.exists():
            existing = _json(EVALUATION_RECEIPT_PATH)
            result = _json(runner.RESULT_PATH)
            if (existing.get("status") != "PASS_ORIGINAL_INDEPENDENT_METRICS_AND_GATES"
                    or existing.get("amendment_source_identity_sha256") != amendment_source["amendment_source_identity_sha256"]
                    or existing.get("prelabel_sha256") != sha256_file(runner.PRELABEL_PATH)
                    or existing.get("score_receipt_sha256") != sha256_file(SCORE_RECEIPT_PATH)
                    or existing.get("result_sha256") != sha256_file(runner.RESULT_PATH)
                    or existing.get("scored_verification_sha256") != sha256_file(runner.SCORED_VERIFICATION_PATH)
                    or existing.get("candidate_status") != result.get("status")
                    or existing.get("gate_counts") != result.get("gate_counts")
                    or existing.get("ledger_sha256") != sha256_file(runner.LEDGER)):
                raise AmendmentError("Existing independent-evaluation receipt does not bind current outputs")
            return _json(runner.SCORED_VERIFICATION_PATH)
        raise AmendmentError("Scored verification exists without amendment receipt; preserve and stop")
    with frozen_evaluation_context(frozen):
        verified = frozen_verifier.verify_scored_metrics()
    if verified.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES":
        raise AmendmentError("Independent original metric/gate verifier did not PASS")
    evaluation_receipt = {
        "status": "PASS_ORIGINAL_INDEPENDENT_METRICS_AND_GATES",
        "amendment_id": AMENDMENT_ID,
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "amendment_source": amendment_source,
        "prelabel_sha256": sha256_file(runner.PRELABEL_PATH),
        "score_receipt_sha256": sha256_file(SCORE_RECEIPT_PATH),
        "result_sha256": sha256_file(runner.RESULT_PATH),
        "scored_verification_sha256": sha256_file(runner.SCORED_VERIFICATION_PATH),
        "candidate_status": _json(runner.RESULT_PATH)["status"],
        "gate_counts": _json(runner.RESULT_PATH)["gate_counts"],
        "labels_opened_after_prelabel_pass": True,
        "ledger_sha256": sha256_file(runner.LEDGER),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(EVALUATION_RECEIPT_PATH, evaluation_receipt)
    return verified


def publish_and_complete() -> dict[str, Any]:
    frozen, amendment_source = verify_original_lineage(require_adapter_qa=True)
    _verify_prelabel_amendment(frozen, amendment_source)
    result = _json(runner.RESULT_PATH)
    scored = _json(runner.SCORED_VERIFICATION_PATH)
    score_receipt = _json(SCORE_RECEIPT_PATH)
    evaluation_receipt = _json(EVALUATION_RECEIPT_PATH)
    if (scored.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES"
            or evaluation_receipt.get("result_sha256") != sha256_file(runner.RESULT_PATH)
            or evaluation_receipt.get("scored_verification_sha256") != sha256_file(runner.SCORED_VERIFICATION_PATH)
            or score_receipt.get("ledger_sha256_after") != sha256_file(runner.LEDGER)):
        raise AmendmentError("Cannot publish before original independent metrics/gates PASS")
    if runner.COMPLETION_PATH.exists() or (runner.REPORT_ROOT / "RESULTS.md").exists():
        raise AmendmentError("Completion/report outputs already exist; preserve and stop")
    ledger_before = sha256_file(runner.LEDGER)
    if ledger_before != EXPECTED_PRECOMPLETION_LEDGER_SHA256:
        raise AmendmentError("Shared ledger changed before report publication")
    # Render the original final report first, before the single completion event.
    report_output = runner.publish_report()
    report_hashes = {}
    for name in ("RESULTS.md", "RESULTS.json", "PRELABEL_VERIFICATION.json", "SCORED_VERIFICATION.json", "COMPLETION_RECEIPT.json"):
        path = runner.REPORT_ROOT / name
        if not path.is_file():
            raise AmendmentError(f"Original report publication omitted {name}")
        report_hashes[name] = sha256_file(path)
    report_receipt = {
        "status": "PASS_ORIGINAL_REPORTS_PUBLISHED",
        "amendment_id": AMENDMENT_ID,
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "evaluation_receipt_sha256": sha256_file(EVALUATION_RECEIPT_PATH),
        "report_hashes": report_hashes,
        "report_paths": report_output["report_paths"],
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(REPORT_RECEIPT_PATH, report_receipt)
    ledger_before = sha256_file(runner.LEDGER)
    if ledger_before != EXPECTED_PRECOMPLETION_LEDGER_SHA256:
        raise AmendmentError("Shared ledger changed before the one authorized CARD_COMPLETED event")
    events = runner._ledger_events()
    prior = [event for event in events if event.get("event") == "CARD_COMPLETED" and event.get("card_id") == runner.CARD_ID]
    if prior:
        raise AmendmentError("Unexpected prior CARD_COMPLETED event exists; preserve and stop")
    completion_event = {
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": runner.CARD_ID, "stage": runner.STAGE, "amendment_id": AMENDMENT_ID,
        "freeze_signature": frozen["freeze_signature"], "status": result["status"],
        "fit_ids": list(runner.FIT_IDS), "result_sha256": sha256_file(runner.RESULT_PATH),
        "scored_verification_sha256": sha256_file(runner.SCORED_VERIFICATION_PATH),
        "prelabel_verification_sha256": sha256_file(runner.PRELABEL_PATH),
        "amendment_receipt_sha256": sha256_file(AMENDMENT_RECEIPT_PATH),
        "evaluation_receipt_sha256": sha256_file(EVALUATION_RECEIPT_PATH),
        "report_receipt_sha256": sha256_file(REPORT_RECEIPT_PATH),
    }
    runner.append_budget_event(completion_event, path=runner.LEDGER)
    final_events = runner._ledger_events()
    final_snapshot = runner._budget_snapshot(final_events)
    matching = [event for event in final_events if event.get("event") == "CARD_COMPLETED" and event.get("card_id") == runner.CARD_ID]
    if (len(matching) != 1 or final_snapshot["fit_starts"] != 118 or final_snapshot["fit_saved"] != 114
            or final_snapshot["fit_failed"] != 4 or final_snapshot["fit_pending"] != 0
            or final_snapshot["global_cap"] != 121):
        raise AmendmentError(f"Final completion accounting mismatch: {final_snapshot}")
    completion = {
        "status": result["status"], "completion_event": matching[0],
        "completion_event_sha256": runner.canonical_sha256(matching[0]),
        "ledger_sha256_before_completion": ledger_before,
        "ledger_sha256_after_completion": sha256_file(runner.LEDGER),
        "ledger_snapshot_after_completion": final_snapshot,
        "result_sha256": sha256_file(runner.RESULT_PATH),
        "scored_verification_sha256": sha256_file(runner.SCORED_VERIFICATION_PATH),
        "report_receipt_sha256": sha256_file(REPORT_RECEIPT_PATH),
        "prelabel_verification_sha256": sha256_file(runner.PRELABEL_PATH),
        "amendment_source_identity_sha256": amendment_source["amendment_source_identity_sha256"],
        "query_labels_opened_only_after_prelabel_pass": True,
    }
    _write_once(COMPLETION_RECEIPT_PATH, completion)
    addendum = _render_addendum(result, completion, amendment_source, report_hashes)
    addendum_path = runner.REPORT_ROOT / "POSTFIT_EVALUATOR_AMENDMENT.md"
    runner.write_create_once(addendum_path, addendum.encode("utf-8"))
    for source, name in (
        (FAILURE_PATH, "POSTFIT_ID_FAILURE.json"),
        (AMENDMENT_RECEIPT_PATH, "POSTFIT_ID_AMENDMENT.json"),
        (SCORE_RECEIPT_PATH, "POSTFIT_SCORE_RECEIPT.json"),
        (EVALUATION_RECEIPT_PATH, "POSTFIT_EVALUATION_RECEIPT.json"),
        (REPORT_RECEIPT_PATH, "POSTFIT_REPORT_RECEIPT.json"),
        (COMPLETION_RECEIPT_PATH, "POSTFIT_COMPLETION_RECEIPT.json"),
    ):
        runner.write_create_once(runner.REPORT_ROOT / name, source.read_bytes())
    return {"status": result["status"], "candidate_status": result["status"],
            "gate_counts": result["gate_counts"], "completion_event": matching[0],
            "ledger_sha256": completion["ledger_sha256_after_completion"],
            "report_paths": report_output["report_paths"] | {"POSTFIT_EVALUATOR_AMENDMENT.md": runner.relpath(addendum_path)}}


def _render_addendum(result: Mapping[str, Any], completion: Mapping[str, Any],
                     source: Mapping[str, Any], report_hashes: Mapping[str, str]) -> str:
    lines = [
        "# CGEOM_V1 — catatan amendment verifier pascafit", "",
        f"Status kandidat: `{result['status']}`. Gate tetap sesuai recipe asli; amendment tidak mengubah gate atau prediksi.", "",
        "## Perbaikan terbatas", "",
        "Verifier awal membandingkan `query_record_ids` hasil NPZ Unicode dengan array object yang berasal dari pandas. Semua ID sumber dan output memiliki shape, nilai, dan urutan yang sama. Adapter mengonversi hanya ID ekspektasi yang berasal dari source menjadi NumPy Unicode sebelum pemeriksaan shape/dtype/nilai frozen; larangan stored object arrays, pemeriksaan numerik, toleransi 1e-12, bank, donor, cycle, serta mass check tetap aktif.", "",
        f"Frozen training source: `{source['original_frozen_source_commit']}`; amendment evaluator: `{source['amendment_source_commit']}`.",
        f"Adapter SHA-256: `{source['adapter_sha256']}`; tests SHA-256: `{source['test_sha256']}`.",
        f"Original verifier SHA-256: `{source['original_verifier_sha256']}`; freeze: `{result['freeze_signature']}`.",
        f"Prelabel SHA-256: `{sha256_file(runner.PRELABEL_PATH)}`; independent score verifier: `{completion['scored_verification_sha256']}`.",
        f"Completion event SHA-256: `{completion['completion_event_sha256']}`; final ledger SHA-256: `{completion['ledger_sha256_after_completion']}`.", "",
        "## Batas dan status", "",
        "Tidak ada fit, refit, predictor reconstruction baru, perubahan artefak prediksi, Test, Kaggle, atau notebook. Query labels dibuka hanya setelah rekonstruksi numerik/boundary independen PASS. Hasil memakai reused development labels dan berstatus diagnostic-only; hasil tidak mengubah champion atau memberi izin submission.", "",
        "Laporan metrik dan gate lengkap ada di `RESULTS.md`; machine-readable details ada di `RESULTS.json`, `PRELABEL_VERIFICATION.json`, dan `SCORED_VERIFICATION.json`.", "",
        "### Hash laporan utama", "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in report_hashes.items())
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--prelabel", action="store_true")
    group.add_argument("--score-once", action="store_true")
    group.add_argument("--verify-scored", action="store_true")
    group.add_argument("--publish-and-complete", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic_qa:
        result = run_synthetic_qa()
    elif args.prelabel:
        result = run_prelabel()
    elif args.score_once:
        result = score_original_once()
    elif args.verify_scored:
        result = verify_scored_once()
    else:
        result = publish_and_complete()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
