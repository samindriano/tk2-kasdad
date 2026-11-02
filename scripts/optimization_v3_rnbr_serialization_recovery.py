"""Serialization-only recovery for the failed-readback R-NBR fit.

This adapter never refits or reconstructs residuals from labels. It safely
normalizes ID arrays, recovers the same persisted E67 observed state into a new
bundle, records append-only recovery lineage, and can execute the three still
reserved memory fits under the original recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_official as runner  # noqa: E402
from scripts import optimization_v3_rnbr_official_verify as independent  # noqa: E402

PARENT_FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
PREVIOUS_ADAPTER_FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze_serialization_adapter_01.json"
ADAPTER_FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze_serialization_adapter_02.json"
ADAPTER_SOURCE = "scripts/optimization_v3_rnbr_serialization_recovery.py"
ADAPTER_TESTS = "tests/test_optimization_v3_rnbr_serialization_recovery.py"
RECOVERY_REPORT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/SERIALIZATION_RECOVERY_V2.json"
ORIGINAL_COMMIT = "738ad02ab8e13d87c4473ad36e296c1a87e3bca0"
EXPECTED_LEDGER_AFTER_FAILURE = "69424b6047c9889360f18f402b7e0ea477d646e6077209a07ca8d10f18ee4a47"
FIT_ID = "RNBR_PILOT_E67_OBSERVED"
ORIGINAL_BUNDLE = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1/bundles/RNBR_PILOT_E67_OBSERVED"
ORIGINAL_NPZ = ORIGINAL_BUNDLE / "arrays.npz"
ORIGINAL_RECEIPT = ORIGINAL_BUNDLE / "receipt.json"
RECOVERY_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1/recovered_bundles"
RECOVERY_BUNDLE = RECOVERY_ROOT / FIT_ID
LEDGER_EVENT = "FIT_OUTPUT_RECOVERED"
OBJECT_ID_FIELDS = frozenset({"score_record_ids", "donor_record_ids", "neighbor_ids"})


class RecoveryError(RuntimeError):
    """A source, identity, serialization, or recovery invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RecoveryError(f"Refusing to overwrite existing different recovery artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def adapter_freeze_body() -> dict[str, Any]:
    parent = json.loads(PARENT_FREEZE.read_text(encoding="utf-8"))
    previous_adapter = json.loads(PREVIOUS_ADAPTER_FREEZE.read_text(encoding="utf-8"))
    if sha_bytes(canonical_bytes({k: v for k, v in parent.items() if k != "freeze_signature"})) != parent.get("freeze_signature"):
        raise RecoveryError("Original R-NBR freeze signature is invalid")
    return {
        "schema_version": 2,
        "adapter_id": "RNBR_SAFE_UNICODE_SERIALIZATION_RECOVERY_02",
        "parent_freeze_path": str(PARENT_FREEZE.relative_to(ROOT)).replace("\\", "/"),
        "parent_freeze_sha256": sha_file(PARENT_FREEZE),
        "parent_freeze_signature": parent["freeze_signature"],
        "supersedes_adapter_freeze": {
            "path": str(PREVIOUS_ADAPTER_FREEZE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha_file(PREVIOUS_ADAPTER_FREEZE),
            "signature": previous_adapter.get("adapter_signature"),
            "status": "SUPERSEDED_BEFORE_OFFICIAL_RECOVERY_OR_FIT",
            "reason": "Its target-free verifier applied donor-cycle R06 OOF alignment to query baseline predictions.",
        },
        "original_source_commit": ORIGINAL_COMMIT,
        "adapter_code_sha256": {
            ADAPTER_SOURCE: sha_file(ROOT / ADAPTER_SOURCE),
            ADAPTER_TESTS: sha_file(ROOT / ADAPTER_TESTS),
        },
        "original_failed_output": {
            "fit_id": FIT_ID,
            "npz_path": str(ORIGINAL_NPZ.relative_to(ROOT)).replace("\\", "/"),
            "npz_sha256": sha_file(ORIGINAL_NPZ),
            "receipt_path": str(ORIGINAL_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
            "receipt_sha256_file": sha_file(ORIGINAL_RECEIPT),
        },
        "serialization_only_amendment": {
            "object_array_field": "donor_record_ids",
            "replacement": "fixed-width NumPy Unicode derived from verified official Train IDs by saved donor row index",
            "model_or_residual_formula_changed": False,
            "feature_or_geometry_changed": False,
            "split_or_gate_changed": False,
            "fit_count_added": 0,
            "original_FIT_FAILED_event_rewritten": False,
        },
        "prediction_lineage_semantics": {
            "query_baseline": "Use the frozen R06 cache at the fit origin for every row in that query horizon.",
            "donor_residual_base": "Use the past-only per-cycle R06 OOF cutoff mapping for donor rows.",
            "source_implementation": "scripts/optimization_v3_rnbr_official.py:build_official_payload",
            "original_independent_verifier_overridden_in_memory": True,
            "frozen_source_files_modified": False,
        },
        "inherited_recipe_and_input_identity": {
            "recipe_freeze_signature": parent["freeze_signature"],
            "train_sha256": parent["official_train"]["sha256"],
            "r06_lineage": parent["r06_lineage"],
            "a02_saved_geometry": parent["a02_saved_geometry"],
            "protocol_sha256": parent["protocol"]["sha256"],
            "fit_ids": parent["fit_ids"],
        },
        "new_fit_ids_authorized": ["RNBR_PILOT_E67_SHUFFLED", "RNBR_PILOT_E75_OBSERVED",
                                   "RNBR_PILOT_E75_SHUFFLED"],
        "recovered_fit_id": FIT_ID,
        "source_boundaries": {"query_targets_read_during_recovery": False,
                               "test_or_sample_submission_opened": False,
                               "classification_labels_used": False,
                               "pickle_loading": False},
    }


def make_adapter_freeze() -> dict[str, Any]:
    body = adapter_freeze_body()
    body["adapter_signature"] = sha_bytes(canonical_bytes(body))
    return body


def write_adapter_freeze() -> dict[str, Any]:
    payload = make_adapter_freeze()
    _write_json_once(ADAPTER_FREEZE, payload)
    return payload


def load_adapter_freeze(*, require_clean: bool = False) -> dict[str, Any]:
    payload = json.loads(ADAPTER_FREEZE.read_text(encoding="utf-8"))
    signature = payload.get("adapter_signature")
    if signature != sha_bytes(canonical_bytes({k: v for k, v in payload.items() if k != "adapter_signature"})):
        raise RecoveryError("Serialization adapter freeze signature mismatch")
    if payload != make_adapter_freeze():
        raise RecoveryError("Serialization adapter inputs/code differ from the frozen successor")
    runner.load_freeze(require_clean=require_clean)
    if require_clean and subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip():
        raise RecoveryError("Recovery/fit start requires a clean committed adapter tree")
    return payload


def _git_blob_sha(commit: str, relative: str) -> str:
    content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return sha_bytes(content)


def verify_original_source_lineage(parent: Mapping[str, Any]) -> None:
    fit_receipt = json.loads(ORIGINAL_RECEIPT.read_text(encoding="utf-8"))
    identity = fit_receipt["identity"]
    if identity.get("source_commit") != ORIGINAL_COMMIT or identity.get("freeze_signature") != parent["freeze_signature"]:
        raise RecoveryError("Failed bundle does not point to the expected original fit source/freeze")
    for relative, expected_hash in parent["code_sha256"].items():
        if _git_blob_sha(ORIGINAL_COMMIT, relative) != expected_hash:
            raise RecoveryError(f"Original code lineage differs at {relative}")
    original_runner = subprocess.check_output(
        ["git", "show", f"{ORIGINAL_COMMIT}:scripts/optimization_v3_rnbr_official.py"], cwd=ROOT, text=True)
    if '"donor_record_ids": donor_ids' not in original_runner:
        raise RecoveryError("Original frozen writer source does not prove donor IDs derive from donor rows")


def inspect_npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise RecoveryError(f"Unsupported NPZ NPY header version {version}")
            headers[member[:-4]] = {"shape": list(shape), "dtype": str(dtype),
                                    "fortran_order": bool(fortran)}
    return headers


def load_nonobject_arrays(path: Path, headers: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    objects = [name for name, meta in headers.items() if meta["dtype"] == "object"]
    if set(objects) != {"donor_record_ids"}:
        raise RecoveryError(f"Unexpected object-array fields in failed state: {objects}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name, meta in headers.items():
            if meta["dtype"] == "object":
                continue
            array = np.asarray(archive[name]).copy()
            if array.dtype.kind == "O":
                raise RecoveryError(f"Safe-load unexpectedly produced object dtype for {name}")
            arrays[name] = array
    return arrays


def normalize_identifier_arrays(arrays: Mapping[str, Any], *, canonical_donor_ids: np.ndarray | None = None
                                 ) -> dict[str, np.ndarray]:
    """Make only known ID arrays Unicode; reject every other object array."""
    output: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "O":
            if key == "donor_record_ids" and canonical_donor_ids is not None:
                array = np.asarray(canonical_donor_ids, dtype=str)
            elif key in OBJECT_ID_FIELDS:
                flat = array.reshape(-1)
                if any(not isinstance(item, (str, np.str_)) for item in flat):
                    raise RecoveryError(f"Object ID field contains a non-string value: {key}")
                array = np.asarray(flat.tolist(), dtype=str).reshape(array.shape)
            else:
                raise RecoveryError(f"Object array is prohibited before persistence: {key}")
        if array.dtype.kind not in "biufUS":
            raise RecoveryError(f"Unsafe payload dtype {array.dtype} for {key}")
        output[key] = np.asarray(array).copy()
    if set(output) != set(arrays):
        raise RecoveryError("Identifier normalization changed the array schema")
    if any(value.dtype.kind == "O" for value in output.values()):
        raise RecoveryError("Object dtype remains after normalization")
    return output


def _fit_events() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    events = runner._read_ledger()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    failures = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == FIT_ID]
    recovered = [e for e in events if e.get("event") == LEDGER_EVENT and e.get("fit_id") == FIT_ID]
    if len(starts) != 1 or len(failures) != 1 or len(recovered) > 1:
        raise RecoveryError("Expected one immutable start/failure and at most one recovery event")
    if failures[0].get("error") != "ValueError: Object arrays cannot be loaded when allow_pickle=False":
        raise RecoveryError("Historical E67 observed failure event differs from the known readback failure")
    return events, starts[0], failures[0]


def _id_digest(values: np.ndarray) -> str:
    return runner.sha256_array(np.asarray(values, dtype=str))


def _expected_donor_r06_oof(rows: np.ndarray, cycles: np.ndarray,
                             cache: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    """Past-only, per-cycle R06 cutoff mapping used for donor residuals."""
    return runner.aligned_r06_predictions(rows, cycles, cache)


def _expected_query_origin_baseline(rows: np.ndarray, origin: int,
                                    cache: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    """R06 origin model prediction for every row in its frozen query horizon."""
    source = cache.get(int(origin))
    if source is None:
        raise RecoveryError(f"Missing frozen query-origin R06 cache: {origin}")
    source_rows = np.asarray(source["rows"], dtype=np.int64)
    predictions = np.asarray(source["prediction"], dtype=np.float64)
    if (source_rows.ndim != 1 or predictions.shape != source_rows.shape
            or len(np.unique(source_rows)) != len(source_rows)
            or not np.isfinite(predictions).all()):
        raise RecoveryError(f"Invalid R06 query-origin cache arrays: {origin}")
    lookup = {int(row): float(prediction) for row, prediction in zip(source_rows, predictions)}
    try:
        result = np.asarray([lookup[int(row)] for row in np.asarray(rows, dtype=np.int64)], dtype=np.float64)
    except KeyError as exc:
        raise RecoveryError(f"R06 origin {origin} does not contain query row {exc.args[0]}") from exc
    return result


def _verify_fit_with_origin_query_baseline(fit_id: str, parent: Mapping[str, Any],
                                           train: Mapping[str, np.ndarray],
                                           cache: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """Use the frozen verifier but correct only its query-baseline lookup in memory."""
    origin = int(fit_id.split("_")[2][1:])
    geometry = runner.load_geometry(origin, train, parent)
    query_rows = np.asarray(geometry["score_rows"], dtype=np.int64)
    query_cycles = np.asarray(train["cycles"][query_rows], dtype=np.int64)
    original_r06_lookup = independent._r06_for_rows

    def recipe_r06_lookup(rows: np.ndarray, cycles: np.ndarray,
                          source_cache: Mapping[int, Mapping[str, np.ndarray]],
                          prediction_key: str) -> np.ndarray:
        if (prediction_key == "prediction"
                and np.array_equal(np.asarray(rows, dtype=np.int64), query_rows)
                and np.array_equal(np.asarray(cycles, dtype=np.int64), query_cycles)):
            return _expected_query_origin_baseline(query_rows, origin, source_cache)
        return original_r06_lookup(rows, cycles, source_cache, prediction_key)

    independent._r06_for_rows = recipe_r06_lookup
    try:
        return independent.verify_fit(fit_id, dict(parent), dict(train), dict(cache))
    finally:
        independent._r06_for_rows = original_r06_lookup


def repair_source_arrays(*, original_npz: Path, headers: Mapping[str, Mapping[str, Any]],
                         donor_rows: np.ndarray, train_ids: np.ndarray) -> dict[str, np.ndarray]:
    arrays = load_nonobject_arrays(original_npz, headers)
    expected_ids = np.asarray(train_ids[np.asarray(donor_rows, dtype=np.int64)], dtype=str)
    if len(expected_ids) != len(donor_rows) or not len(expected_ids):
        raise RecoveryError("Source-derived donor ID map is incomplete")
    arrays["donor_record_ids"] = expected_ids
    normalized = normalize_identifier_arrays(arrays, canonical_donor_ids=expected_ids)
    if normalized["donor_record_ids"].dtype.kind != "U":
        raise RecoveryError("Reconstructed donor IDs are not fixed-width Unicode")
    return normalized


def _load_parent() -> dict[str, Any]:
    parent = runner.load_freeze(require_clean=False)
    if sha_file(ORIGINAL_NPZ) != "f8f33a71419459a74a70c3edb2bc38191daea11a4628c516a2b89597f2a6972b":
        raise RecoveryError("Original failed NPZ hash changed")
    receipt = json.loads(ORIGINAL_RECEIPT.read_text(encoding="utf-8"))
    integrity_body = {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    if (receipt.get("npz_sha256") != sha_file(ORIGINAL_NPZ)
            or receipt.get("identity_sha256") != sha_bytes(canonical_bytes(receipt["identity"]))
            or receipt.get("integrity_sha256") != sha_bytes(canonical_bytes(integrity_body))):
        raise RecoveryError("Original output receipt/hash integrity failed")
    verify_original_source_lineage(parent)
    return parent


def _validate_recovered_arrays(arrays: Mapping[str, np.ndarray], receipt: Mapping[str, Any],
                               parent: Mapping[str, Any]) -> dict[str, Any]:
    train = runner.load_targetfree_train()
    cache = runner.load_r06_predictions(train, parent)
    geometry = runner.load_geometry(67, train, parent)
    rows = np.asarray(arrays["score_rows"], dtype=np.int64)
    ids = np.asarray(arrays["score_record_ids"], dtype=str)
    cycles = np.asarray(arrays["score_cycles"], dtype=np.int64)
    expected_query_rows = np.flatnonzero((train["cycles"] >= 68) & (train["cycles"] <= 79)).astype(np.int64)
    expected_donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= 67)
                                         & train["present"]).astype(np.int64)
    if (not np.array_equal(rows, expected_query_rows)
            or not np.array_equal(ids, train["ids"][rows].astype(str))
            or not np.array_equal(cycles, train["cycles"][rows])
            or not np.array_equal(arrays["donor_rows"], expected_donor_rows)
            or not np.array_equal(arrays["donor_record_ids"].astype(str), train["ids"][expected_donor_rows].astype(str))
            or not np.array_equal(arrays["donor_cycles"], train["cycles"][expected_donor_rows])):
        raise RecoveryError("Recovered E67 donor/query rows or source-derived IDs do not match official Train")
    donor_ids = np.asarray(train["ids"][expected_donor_rows], dtype=str)
    if not np.array_equal(arrays["donor_record_ids"].astype(str), donor_ids):
        raise RecoveryError("Reconstructed donor ID values differ from canonical source ID join")
    expected_cutoffs = np.asarray([runner.EXPECTED_SOURCE_CUTOFFS[int(c)]
                                   for c in train["cycles"][expected_donor_rows]], dtype=np.int64)
    if not np.array_equal(arrays["donor_source_cutoffs"], expected_cutoffs):
        raise RecoveryError("Saved donor R06 source-cutoff assignment differs")
    expected_r06 = _expected_donor_r06_oof(expected_donor_rows,
                                           train["cycles"][expected_donor_rows], cache)
    if not np.array_equal(arrays["donor_r06_predictions"], expected_r06):
        raise RecoveryError("Saved donor OOF R06 values do not match frozen cutoff/row membership")
    expected_base = _expected_query_origin_baseline(rows, origin=67, cache=cache)
    if not np.array_equal(arrays["baseline_prediction"], expected_base):
        raise RecoveryError("Saved query R06 base predictions differ from the frozen complete E67 horizon")
    for name in ("score_rows", "score_record_ids", "score_cycles", "neighbor_count",
                 "neighbor_source_rows", "neighbor_ids", "neighbor_similarities"):
        if not np.array_equal(arrays[name], geometry[name]):
            raise RecoveryError(f"Recovered query geometry differs from target-free A02 cache: {name}")
    expected_missing = ~train["present"][rows]
    if (not np.array_equal(arrays["query_missing"], expected_missing)
            or np.any(arrays["neighbor_count"][expected_missing] != 0)):
        raise RecoveryError("Saved query availability/fallback flags differ")
    expected_windows = np.where(cycles <= 71, "W4", np.where(cycles <= 75, "W5", "W6")).astype("U2")
    if not np.array_equal(arrays["score_window"], expected_windows):
        raise RecoveryError("Saved query window assignment differs")
    meta = receipt["metadata"]
    residual_sha = runner.sha256_array(arrays["donor_residuals"], "<f8")
    used_sha = runner.sha256_array(arrays["donor_residuals_used"], "<f8")
    if (residual_sha != meta.get("donor_residual_sha256")
            or used_sha != meta.get("arm_residual_sha256")
            or not np.array_equal(arrays["donor_residuals"], arrays["donor_residuals_used"])):
        raise RecoveryError("Saved observed residual state or its value hashes do not match original receipt")
    # Verify the source-target digest directly for donor rows only. This check
    # does not subtract targets from predictions or rebuild the residual memory.
    donor_target = runner.read_regression_targets(expected_donor_rows, purpose="RECOVERY_DONOR_HASH_ONLY")
    if (not np.array_equal(donor_target["ids"].astype(str), donor_ids)
            or not np.array_equal(donor_target["cycles"], train["cycles"][expected_donor_rows])
            or runner.sha256_array(donor_target["target"], "<f8") != meta.get("donor_target_values_sha256")):
        raise RecoveryError("Donor-only source target digest does not match the original saved fit receipt")
    del donor_target
    # Independent formula check consumes only the saved residual vector and
    # target-free geometry. It never reads query labels or builds residuals.
    independent_delta, independent_prediction = independent.independent_prediction_recompute(arrays)
    if (not np.allclose(arrays["delta"], independent_delta, atol=1e-12, rtol=0)
            or not np.allclose(arrays["prediction"], independent_prediction, atol=1e-12, rtol=0)
            or not np.array_equal(arrays["prediction"][expected_missing], expected_base[expected_missing])
            or np.any(np.asarray([v.dtype.kind == "O" for v in arrays.values()]))):
        raise RecoveryError("Independent formula/missing fallback/safe dtype verification failed")
    return {
        "status": "PASS", "fit_id": FIT_ID,
        "donor_rows": int(len(expected_donor_rows)), "query_rows": int(len(expected_query_rows)),
        "donor_id_sha256": _id_digest(donor_ids), "query_id_sha256": _id_digest(ids),
        "donor_residual_sha256": residual_sha, "arm_residual_sha256": used_sha,
        "donor_target_values_sha256_verified_only": meta["donor_target_values_sha256"],
        "donor_r06_membership_sha256": runner.sha256_array(expected_r06, "<f8"),
        "query_geometry_rows_checked": int(np.sum(geometry["neighbor_count"])),
        "missing_queries_unchanged": int(np.sum(expected_missing)),
        "independent_delta_sha256": runner.sha256_array(independent_delta, "<f8"),
        "independent_prediction_sha256": runner.sha256_array(independent_prediction, "<f8"),
        "query_targets_read": False, "test_or_sample_submission_opened": False,
        "residual_memory_reconstructed_from_target": False,
    }


def _atomic_repair_bundle(arrays: Mapping[str, np.ndarray], original_receipt: Mapping[str, Any],
                          parent: Mapping[str, Any], adapter: Mapping[str, Any],
                          formula_audit: Mapping[str, Any], failure_event: Mapping[str, Any]) -> dict[str, Any]:
    safe = normalize_identifier_arrays(arrays, canonical_donor_ids=arrays["donor_record_ids"])
    if RECOVERY_BUNDLE.exists() or RECOVERY_BUNDLE.with_name(RECOVERY_BUNDLE.name + ".staging").exists():
        raise RecoveryError("Recovery bundle destination/staging exists; only exact verified resume is allowed")
    staging = RECOVERY_BUNDLE.with_name(RECOVERY_BUNDLE.name + ".staging")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    npz = staging / "arrays.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, **safe)
        stream.flush()
        os.fsync(stream.fileno())
    original_npz_hash = sha_file(ORIGINAL_NPZ)
    original_receipt_file_hash = sha_file(ORIGINAL_RECEIPT)
    repair_lineage = {
        "recovery_type": "SAME_FIT_OUTPUT_SERIALIZATION_REPAIR_NO_REFIT",
        "original_fit_id": FIT_ID,
        "original_bundle_npz_sha256": original_npz_hash,
        "original_bundle_receipt_file_sha256": original_receipt_file_hash,
        "original_bundle_receipt_integrity_sha256": original_receipt["integrity_sha256"],
        "original_failure_event_sha256": sha_bytes(canonical_bytes(failure_event)),
        "original_failure_event_preserved": True,
        "original_source_commit": original_receipt["identity"]["source_commit"],
        "repair_source_commit": source_commit(),
        "parent_recipe_freeze_signature": parent["freeze_signature"],
        "serialization_adapter_signature": adapter["adapter_signature"],
        "object_dtype_fields_replaced": ["donor_record_ids"],
        "id_source": "official Regression Train record_id by saved donor_rows; full input SHA verified",
        "no_memory_fit_or_target_subtraction": True,
        "formula_audit": dict(formula_audit),
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "identity": dict(original_receipt["identity"]),
        "identity_sha256": runner.sha256_bytes(canonical_bytes(original_receipt["identity"])),
        "npz_sha256": sha_file(npz),
        "metadata": {**original_receipt["metadata"],
                     "output_recovered_no_refit": True,
                     "serialization_adapter_signature": adapter["adapter_signature"],
                     "serialization_normalization": {"field": "donor_record_ids",
                                                     "original_dtype": "object",
                                                     "repaired_dtype": str(safe["donor_record_ids"].dtype),
                                                     "row_index_source_join_sha256": formula_audit["donor_id_sha256"]}},
        "array_shapes": {k: list(v.shape) for k, v in safe.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in safe.items()},
        "array_sha256": {k: runner.sha256_array(v) for k, v in safe.items()},
        "repair_lineage": repair_lineage,
    }
    receipt["integrity_sha256"] = runner.sha256_bytes(canonical_bytes(receipt))
    receipt_path = staging / "receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, RECOVERY_BUNDLE)
    return receipt


def _load_repaired_bundle(parent: Mapping[str, Any], adapter: Mapping[str, Any]
                          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz, receipt_path = RECOVERY_BUNDLE / "arrays.npz", RECOVERY_BUNDLE / "receipt.json"
    if not npz.is_file() or not receipt_path.is_file():
        raise RecoveryError("Recovered output bundle is absent or incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    identity = receipt.get("identity", {})
    lineage = receipt.get("repair_lineage", {})
    events, _, failed = _fit_events()
    if (receipt.get("integrity_sha256") != sha_bytes(canonical_bytes(body))
            or receipt.get("npz_sha256") != sha_file(npz)
            or receipt.get("identity_sha256") != sha_bytes(canonical_bytes(identity))
            or identity.get("fit_id") != FIT_ID
            or identity.get("freeze_signature") != parent["freeze_signature"]
            or identity.get("source_commit") != ORIGINAL_COMMIT
            or lineage.get("original_bundle_npz_sha256") != sha_file(ORIGINAL_NPZ)
            or lineage.get("original_bundle_receipt_file_sha256") != sha_file(ORIGINAL_RECEIPT)
            or lineage.get("original_failure_event_sha256") != sha_bytes(canonical_bytes(failed))
            or lineage.get("serialization_adapter_signature") != adapter["adapter_signature"]):
        raise RecoveryError("Recovered bundle receipt linkage/hash differs")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in z.files}
    if (any(value.dtype.kind == "O" for value in arrays.values())
            or receipt.get("array_shapes") != {k: list(v.shape) for k, v in arrays.items()}
            or receipt.get("array_dtypes") != {k: str(v.dtype) for k, v in arrays.items()}
            or receipt.get("array_sha256") != {k: runner.sha256_array(v) for k, v in arrays.items()}):
        raise RecoveryError("Recovered bundle arrays fail pickle-free readback/schema/hash validation")
    return arrays, receipt


def _append_recovery_event(parent: Mapping[str, Any], adapter: Mapping[str, Any],
                           receipt: Mapping[str, Any], recovery_npz_hash: str,
                           failed_event: Mapping[str, Any]) -> dict[str, Any]:
    events = runner._read_ledger()
    existing = [e for e in events if e.get("event") == LEDGER_EVENT and e.get("fit_id") == FIT_ID]
    expected_fields = {
        "event": LEDGER_EVENT, "batch_id": "optimization_v3_research_next_v1",
        "card_id": runner.CARD_ID, "stage": runner.STAGE, "fit_id": FIT_ID, "cutoff": 67,
        "original_terminal_status": "FIT_FAILED", "original_terminal_event_preserved": True,
        "refit_performed": False, "fit_slot_consumed_again": False,
        "bundle_receipt_sha256": receipt["integrity_sha256"],
        "fit_state_sha256": receipt["npz_sha256"],
        "recovered_bundle_npz_sha256": recovery_npz_hash,
        "original_bundle_npz_sha256": sha_file(ORIGINAL_NPZ),
        "original_failure_event_sha256": sha_bytes(canonical_bytes(failed_event)),
        "parent_recipe_freeze_signature": parent["freeze_signature"],
        "serialization_adapter_signature": adapter["adapter_signature"],
        "recovery_code_sha256": adapter["adapter_code_sha256"][ADAPTER_SOURCE],
    }
    if existing:
        if len(existing) != 1 or any(existing[0].get(k) != v for k, v in expected_fields.items()):
            raise RecoveryError("Existing fit recovery event is not an exact idempotent match")
        return existing[0]
    if sha_file(runner.LEDGER) != EXPECTED_LEDGER_AFTER_FAILURE:
        raise RecoveryError("Shared ledger changed after the known FIT_FAILED checkpoint")
    events_now, started, failed = _fit_events()
    if failed != failed_event or started.get("fit_id") != FIT_ID:
        raise RecoveryError("Original start/failure lineage changed before recovery event")
    event = {
        **expected_fields,
        "recovery_bundle_path": str((RECOVERY_BUNDLE / "arrays.npz").relative_to(ROOT)).replace("\\", "/"),
        "recovery_receipt_path": str((RECOVERY_BUNDLE / "receipt.json").relative_to(ROOT)).replace("\\", "/"),
        "recovery_source_commit": source_commit(),
        "original_fit_started_event_sha256": sha_bytes(canonical_bytes(started)),
        "original_terminal_event_preserved": True,
        "refit_performed": False,
        "fit_slot_consumed_again": False,
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    runner._append_event(event)
    return event


def _load_or_create_recovery_bundle(*, arrays: Mapping[str, np.ndarray],
                                    original_receipt: Mapping[str, Any],
                                    parent: Mapping[str, Any], adapter: Mapping[str, Any],
                                    formula_audit: Mapping[str, Any],
                                    failed_event: Mapping[str, Any],
                                    recovery_event_exists: bool
                                    ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    staging = RECOVERY_BUNDLE.with_name(RECOVERY_BUNDLE.name + ".staging")
    if recovery_event_exists or RECOVERY_BUNDLE.exists():
        if staging.exists():
            raise RecoveryError("Recovery staging directory exists; inspect it before any resume")
        recovered_arrays, recovered_receipt = _load_repaired_bundle(parent, adapter)
        if set(arrays) != set(recovered_arrays) or any(
                not np.array_equal(arrays[k], recovered_arrays[k]) for k in arrays):
            raise RecoveryError("Existing recovery bundle differs from reconstructed safe representation")
        return recovered_arrays, recovered_receipt, dict(formula_audit)

    receipt = _atomic_repair_bundle(arrays, original_receipt, parent, adapter, formula_audit, failed_event)
    recovered_arrays, receipt = _load_repaired_bundle(parent, adapter)
    # Re-run all invariants after durable readback, still without query truth.
    checked_audit = _validate_recovered_arrays(recovered_arrays, receipt, parent)
    return recovered_arrays, receipt, checked_audit


def recover_existing_output(*, adapter: Mapping[str, Any]) -> dict[str, Any]:
    parent = _load_parent()
    if runner.sha256_file(runner.REG_TRAIN) != runner.TRAIN_SHA:
        raise RecoveryError("Official Train input hash differs")
    if sha_file(runner.LEDGER) not in (EXPECTED_LEDGER_AFTER_FAILURE,):
        # An already-appended exact recovery event is handled idempotently below.
        events = runner._read_ledger()
        if not any(e.get("event") == LEDGER_EVENT and e.get("fit_id") == FIT_ID for e in events):
            raise RecoveryError("Live ledger is not at the exact failed-fit checkpoint")
    events, started, failed = _fit_events()
    recovery_events = [e for e in events if e.get("event") == LEDGER_EVENT and e.get("fit_id") == FIT_ID]
    receipt = json.loads(ORIGINAL_RECEIPT.read_text(encoding="utf-8"))
    original_headers = inspect_npz_headers(ORIGINAL_NPZ)
    # Re-read only identifier/cycle/availability columns; targets are not part
    # of the row-ID reconstruction or formula replay.
    train = runner.load_targetfree_train()
    cache = runner.load_r06_predictions(train, parent)
    geometry = runner.load_geometry(67, train, parent)
    with np.load(ORIGINAL_NPZ, allow_pickle=False) as z:
        donor_rows = np.asarray(z["donor_rows"], dtype=np.int64).copy()
    arrays = repair_source_arrays(original_npz=ORIGINAL_NPZ, headers=original_headers,
                                  donor_rows=donor_rows, train_ids=train["ids"])
    formula_audit = _validate_recovered_arrays(arrays, receipt, parent)
    arrays, receipt, formula_audit = _load_or_create_recovery_bundle(
        arrays=arrays, original_receipt=receipt, parent=parent, adapter=adapter,
        formula_audit=formula_audit, failed_event=failed,
        recovery_event_exists=bool(recovery_events),
    )
    event = _append_recovery_event(parent, adapter, receipt, receipt["npz_sha256"], failed)
    return {
        "status": "SAME_FIT_OUTPUT_RECOVERED_NO_REFIT",
        "fit_id": FIT_ID, "original_started_at": started.get("at_utc"),
        "original_failed_at": failed.get("at_utc"),
        "original_failure_event_preserved": True,
        "original_npz_sha256": sha_file(ORIGINAL_NPZ),
        "original_receipt_file_sha256": sha_file(ORIGINAL_RECEIPT),
        "recovery_npz_sha256": receipt["npz_sha256"],
        "recovery_receipt_sha256": receipt["integrity_sha256"],
        "recovery_event": event,
        "formula_audit": formula_audit,
        "query_targets_read": False,
        "memory_fit_count_added": 0,
    }


def _load_recovery_for_runner(parent: Mapping[str, Any], adapter: Mapping[str, Any]
                              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, receipt = _load_repaired_bundle(parent, adapter)
    # Formula/lineage is checked each time the adapter makes the existing slot
    # visible to the frozen runner; no residual memory is constructed here.
    _validate_recovered_arrays(arrays, receipt, parent)
    return arrays, receipt


def _load_fit_bundle(fit_id: str, parent: Mapping[str, Any], adapter: Mapping[str, Any],
                     source_commit_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if fit_id == FIT_ID:
        arrays, receipt = _load_recovery_for_runner(parent, adapter)
        return arrays, receipt
    return runner.load_bundle(fit_id, parent, source_commit=source_commit_id)


def _install_serialization_adapter(parent: Mapping[str, Any], adapter: Mapping[str, Any]) -> None:
    base_build = runner.build_official_payload
    base_persist = runner.persist_bundle
    base_bundle_dir = runner._bundle_dir
    base_load_bundle = runner.load_bundle
    base_finish_fit = runner.finish_fit
    base_append_event = runner._append_event
    base_write_once = runner._write_json_once

    def safe_builder(**kwargs: Any):
        arrays, metadata = base_build(**kwargs)
        arrays = normalize_identifier_arrays(arrays)
        metadata["audit"]["serialization_adapter_signature"] = adapter["adapter_signature"]
        metadata["audit"]["serialization_adapter_changed_model_or_formula"] = False
        return arrays, metadata

    def safe_persist(fit_id: str, arrays: Mapping[str, Any], metadata: Mapping[str, Any]):
        normalized = normalize_identifier_arrays(arrays)
        if any(value.dtype.kind == "O" for value in normalized.values()):
            raise RecoveryError("Refusing to persist object dtype in a memory fit bundle")
        copied = dict(metadata)
        copied_audit = dict(copied["audit"])
        copied_audit["serialization_adapter_signature"] = adapter["adapter_signature"]
        copied_audit["serialization_adapter_changed_model_or_formula"] = False
        copied["audit"] = copied_audit
        return base_persist(fit_id, normalized, copied)

    def patched_bundle_dir(fit_id: str) -> Path:
        return RECOVERY_BUNDLE if fit_id == FIT_ID else base_bundle_dir(fit_id)

    def patched_load_bundle(fit_id: str, freeze: Mapping[str, Any], *, source_commit: str | None = None):
        if fit_id == FIT_ID:
            return _load_recovery_for_runner(parent, adapter)
        return base_load_bundle(fit_id, freeze, source_commit=source_commit)

    def patched_finish_fit(fit_id: str, status: str, *, receipt_sha256: str | None = None,
                           error: str | None = None):
        if fit_id == FIT_ID:
            if status == "FIT_SAVED":
                return None  # Preserve the historical FIT_FAILED terminal event.
            raise RecoveryError("Recovered fit terminal lineage is immutable")
        return base_finish_fit(fit_id, status, receipt_sha256=receipt_sha256, error=error)

    def patched_append_event(event: Mapping[str, Any]):
        payload = dict(event)
        if payload.get("event") == "FIT_STARTED" and payload.get("fit_id") in runner.FIT_IDS:
            payload["serialization_adapter_signature"] = adapter["adapter_signature"]
            payload["parent_recipe_freeze_signature"] = parent["freeze_signature"]
            payload["source_change_scope"] = "safe_unicode_ID_serialization_only"
        return base_append_event(payload)

    result_path = runner.REPORT_ROOT / "PILOT_RESULT.json"

    def patched_write_once(path: Path, payload: Mapping[str, Any]):
        if path == result_path:
            final = dict(payload)
            events = runner._read_ledger()
            snap = runner.ledger_snapshot(events)
            recovered = [e for e in events if e.get("event") == LEDGER_EVENT
                         and e.get("fit_id") == FIT_ID]
            failed = [e for e in events if e.get("event") == "FIT_FAILED"]
            final["serialization_recovery"] = {
                "adapter_signature": adapter["adapter_signature"],
                "recovered_fit_ids": [FIT_ID] if recovered else [],
                "refit_count_for_recovered_fit": 0,
                "historical_fit_failed_events_retained": len(failed),
                "recovered_failed_fit_events": len(recovered),
                "unresolved_failed_fit_events": len(failed) - len(recovered),
                "ledger_final_snapshot": snap,
                "ledger_final_sha256": runner.sha256_file(runner.LEDGER),
            }
            payload = final
        return base_write_once(path, payload)

    runner.build_official_payload = safe_builder
    runner.persist_bundle = safe_persist
    runner._bundle_dir = patched_bundle_dir
    runner.load_bundle = patched_load_bundle
    runner.finish_fit = patched_finish_fit
    runner._append_event = patched_append_event
    runner._write_json_once = patched_write_once


def independent_verify_available(*, all_fits: bool, parent: Mapping[str, Any],
                                 adapter: Mapping[str, Any], source_commit_id: str) -> dict[str, Any]:
    train = runner.load_targetfree_train()
    cache = runner.load_r06_predictions(train, parent)
    events = runner._read_ledger()
    recovered_event = [e for e in events if e.get("event") == LEDGER_EVENT and e.get("fit_id") == FIT_ID]
    recovery_arrays = None
    recovery_receipt = None
    if recovered_event:
        recovery_arrays, recovery_receipt = _load_recovery_for_runner(parent, adapter)
    available = []
    fits: dict[str, dict[str, np.ndarray]] = {}
    for fit_id in runner.FIT_IDS:
        if fit_id == FIT_ID:
            if recovery_arrays is None:
                continue
            arrays = recovery_arrays
            identity = recovery_receipt["identity"]
            if identity.get("source_commit") != ORIGINAL_COMMIT:
                raise RecoveryError("Recovered fit lost original source commit lineage")
            audit = _validate_recovered_arrays(arrays, recovery_receipt, parent)
            receipt = recovery_receipt
            item = {"fit_id": fit_id, "origin": 67, "arm": "observed", "status": "PASS",
                    "terminal_history": "FIT_FAILED_PRESERVED_OUTPUT_RECOVERED_NO_REFIT",
                    "donor_rows": audit["donor_rows"], "query_rows": audit["query_rows"],
                    "bundle_npz_sha256": receipt["npz_sha256"],
                    "receipt_sha256": receipt["integrity_sha256"],
                    "donor_residual_sha256_recomputed": audit["donor_residual_sha256"],
                    "donor_target_read_scope": "donor hash only; residual memory not rebuilt"}
        else:
            directory = runner._bundle_dir(fit_id)
            terminal = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
            if not terminal and not directory.is_dir():
                continue
            arrays, receipt = runner.load_bundle(fit_id, parent, source_commit=source_commit_id)
            item = _verify_fit_with_origin_query_baseline(fit_id, parent, train, cache)
            if terminal and terminal[0].get("receipt_sha256") != receipt.get("integrity_sha256"):
                raise RecoveryError(f"Saved FIT_SAVED receipt mismatches {fit_id}")
        available.append(item)
        fits[fit_id] = arrays
    expected_saved_fit_ids = set(fits)
    if all_fits and expected_saved_fit_ids != set(runner.FIT_IDS):
        raise RecoveryError(f"Expected four verified IDs including one recovery; found {sorted(expected_saved_fit_ids)}")
    result: dict[str, Any] = {
        "status": "PASS", "freeze_signature": parent["freeze_signature"],
        "serialization_adapter_signature": adapter["adapter_signature"],
        "fit_bundle_verification": available,
        "query_truth_loaded_after_all_four_fit_bundles_verified": False,
        "classification_labels_loaded": False, "test_or_sample_submission_opened": False,
        "recovered_fit_lineage": {"fit_id": FIT_ID, "refit_performed": False,
                                   "original_failed_event_preserved": bool(recovered_event),
                                   "recovery_bundle_receipt_sha256": (recovery_receipt or {}).get("integrity_sha256")},
    }
    if all_fits:
        metrics, comparison_lineage = independent._compare_predictions(parent, train, fits)
        result["query_truth_loaded_after_all_four_fit_bundles_verified"] = True
        result["metrics_and_views"] = metrics
        result["comparison_lineage"] = comparison_lineage
        result["source_boundaries"] = {
            "official_train_only": True, "query_truth_rows": 20903,
            "query_truth_cycles": [68, 79], "query_truth_loaded_after_bundle_verification": True,
            "test_or_sample_submission_opened": False, "classification_labels_loaded": False,
            "kaggle_access": False, "full_train_refit": False,
        }
        result["bundle_receipts"] = [
            {"fit_id": fit_id, "npz_sha256": (recovery_receipt if fit_id == FIT_ID else
                                                 runner.load_bundle(fit_id, parent, source_commit=source_commit_id)[1])["npz_sha256"],
             "receipt_sha256": (recovery_receipt if fit_id == FIT_ID else
                                  runner.load_bundle(fit_id, parent, source_commit=source_commit_id)[1])["integrity_sha256"]}
            for fit_id in runner.FIT_IDS
        ]
        independent._write_once(independent.REPORT, result)
    return result


def run_authorized() -> dict[str, Any]:
    adapter = load_adapter_freeze(require_clean=True)
    parent = runner.load_freeze(require_clean=True)
    source_commit_id = source_commit()
    # Recover and independently verify the failed slot before any new fit.
    recovery_result = recover_existing_output(adapter=adapter)
    if recovery_result.get("status") != "SAME_FIT_OUTPUT_RECOVERED_NO_REFIT":
        raise RecoveryError("Same-fit recovery did not pass")
    _install_serialization_adapter(parent, adapter)
    original_verifier = runner._run_independent_verifier
    def adapter_verifier(*, all_fits: bool) -> dict[str, Any]:
        return independent_verify_available(all_fits=all_fits, parent=parent,
                                             adapter=adapter, source_commit_id=source_commit_id)
    runner._run_independent_verifier = adapter_verifier
    # The failed FIT_STARTED/FIT_FAILED history stays unchanged; the adapter
    # makes the repaired durable state resumable and lets the frozen runner
    # start only the other three IDs already in the four-slot reservation.
    result = runner.run_official()
    final = dict(result)
    final["serialization_recovery_detail"] = recovery_result
    final["source_commit"] = source_commit_id
    final["fit_count_summary"] = {
        "unique_authorized_fit_ids": 4, "memory_constructions_charged": 4,
        "recovered_same_fit_no_refit": 1, "new_memory_fits_started_after_recovery": 3,
        "extra_or_retry_fit_slots": 0,
    }
    _write_json_once(RECOVERY_REPORT, final)
    return final


def safe_adapter_preflight() -> dict[str, Any]:
    adapter = load_adapter_freeze(require_clean=False)
    parent = _load_parent()
    events, started, failed = _fit_events()
    headers = inspect_npz_headers(ORIGINAL_NPZ)
    objects = [name for name, meta in headers.items() if meta["dtype"] == "object"]
    if objects != ["donor_record_ids"]:
        raise RecoveryError(f"Unexpected object array inventory: {objects}")
    return {
        "status": "PASS_RECOVERY_READY", "parent_freeze_signature": parent["freeze_signature"],
        "adapter_signature": adapter["adapter_signature"], "current_head": source_commit(),
        "ledger_sha256": sha_file(runner.LEDGER),
        "original_fit_id": FIT_ID, "original_started_event_sha256": sha_bytes(canonical_bytes(started)),
        "original_failed_event_sha256": sha_bytes(canonical_bytes(failed)),
        "original_bundle_npz_sha256": sha_file(ORIGINAL_NPZ),
        "original_bundle_receipt_sha256": json.loads(ORIGINAL_RECEIPT.read_text(encoding="utf-8"))["integrity_sha256"],
        "all_npz_array_headers": headers, "object_dtype_fields": objects,
        "target_values_loaded": False, "query_targets_loaded": False,
        "test_or_sample_submission_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--recover-only", action="store_true")
    modes.add_argument("--run-authorized", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = write_adapter_freeze()
    elif args.preflight:
        result = safe_adapter_preflight()
    elif args.recover_only:
        adapter = load_adapter_freeze(require_clean=True)
        result = recover_existing_output(adapter=adapter)
    else:
        result = run_authorized()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
