"""Independent target-blind verifier for center-local application source R3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import zipfile

import numpy as np

try:
    import reg_center_local_application_v1 as runner
except ImportError:  # pragma: no cover - package execution fallback
    from scripts import reg_center_local_application_v1 as runner


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path, *, signed: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise runner.GateError(f"Independent verifier cannot read {path}") from exc
    if not isinstance(value, dict):
        raise runner.GateError(f"Independent verifier expected object at {path}")
    if signed:
        signature = value.get("signature")
        body = dict(value)
        body.pop("signature", None)
        if not isinstance(signature, str) or signature != _canonical(body):
            raise runner.GateError(f"Independent signature failure: {path}")
    return value


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise runner.GateError("Independent identity hash rejects object arrays")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _identity(path: Path, *, synthetic: bool) -> dict[str, Any]:
    expected_members = {f"{name}.npy" for name in runner.ARCHIVE_FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise runner.GateError("Independent verifier cannot inspect the OOF NPZ") from exc
    if members != expected_members:
        raise runner.GateError("Independent verifier found unknown/missing OOF NPZ members")
    try:
        with np.load(path, allow_pickle=False) as archive:
            # Deliberately access only identity/prediction arrays. `targets` is not indexed.
            arrays = {name: np.asarray(archive[name]).copy() for name in runner.IDENTITY_FIELDS}
    except (OSError, ValueError, KeyError) as exc:
        raise runner.GateError("Independent non-target OOF array read failed") from exc

    if any(value.ndim != 1 for value in arrays.values()):
        raise runner.GateError("Independent OOF arrays are not one-dimensional")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise runner.GateError("Independent OOF arrays are misaligned")
    n = lengths.pop()
    ids = arrays["record_ids"]
    physical = arrays["physical_rows"]
    origins = arrays["prediction_origins"].astype(np.int64, copy=False)
    indices = arrays["original_bundle_indices"].astype(np.int64, copy=False)
    cycles = arrays["cycles"].astype(np.int64, copy=False)
    direct = arrays["direct_predictions"]
    missing = arrays["missing_mask"]
    if (ids.dtype.hasobject or ids.dtype.kind not in "US"
            or physical.dtype.kind not in "iu"
            or arrays["prediction_origins"].dtype.kind not in "iu"
            or arrays["original_bundle_indices"].dtype.kind not in "iu"
            or arrays["cycles"].dtype.kind not in "iu"
            or direct.dtype.kind != "f" or direct.dtype.itemsize != 8
            or missing.dtype.kind != "b" or not np.isfinite(direct).all()
            or np.any((direct < 0.0) | (direct > 100.0))):
        raise runner.GateError("Independent OOF dtype/range validation failed")
    ids_text = ids.astype(str)
    pairs = np.column_stack((origins, indices))
    if (len(np.unique(ids_text)) != n or len(np.unique(physical)) != n
            or len(np.unique(pairs, axis=0)) != n or np.any(cycles <= origins)):
        raise runner.GateError("Independent OOF identity uniqueness or causal check failed")

    missing_b = missing.astype(bool, copy=False)
    origin_counts = {int(k): int(v) for k, v in zip(*np.unique(origins, return_counts=True))}
    present_counts = {int(k): int(np.sum((origins == k) & (~missing_b))) for k in np.unique(origins)}
    missing_counts = {int(k): int(np.sum((origins == k) & missing_b)) for k in np.unique(origins)}
    cycle_counts = {int(k): int(v) for k, v in zip(*np.unique(cycles, return_counts=True))}
    cycles_by_origin = {
        int(origin): tuple(int(value) for value in np.unique(cycles[origins == origin]))
        for origin in np.unique(origins)
    }
    if not synthetic:
        if (n != runner.EXPECTED_TOTAL
                or int((~missing_b).sum()) != runner.EXPECTED_PRESENT
                or int(missing_b.sum()) != runner.EXPECTED_MISSING
                or origin_counts != runner.EXPECTED_ORIGIN_COUNTS
                or present_counts != runner.EXPECTED_PRESENT_BY_ORIGIN
                or cycle_counts != runner.EXPECTED_CYCLE_COUNTS
                or cycles_by_origin != runner.EXPECTED_CYCLES_BY_ORIGIN):
            raise runner.GateError("Independent population/origin/cycle counts differ from frozen scope")

    basis = np.maximum(0.0, 1.0 - np.abs(direct.astype(np.float64) - 50.0) / 10.0)
    active = (~missing_b) & (basis > 0.0)
    by_origin: dict[str, Any] = {}
    for origin in sorted(origin_counts):
        subset = origins == origin
        by_origin[str(origin)] = {
            "all_rows": origin_counts[origin],
            "present_rows": present_counts[origin],
            "missing_rows": missing_counts[origin],
            "cycles": list(cycles_by_origin[origin]),
            "cycle_counts": {str(cycle): int(np.sum(subset & (cycles == cycle)))
                             for cycle in cycles_by_origin[origin]},
            "unique_record_ids": int(len(np.unique(ids_text[subset]))),
            "unique_origin_bundle_pairs": int(len(np.unique(pairs[subset], axis=0))),
            "present_b_positive_rows": int(np.sum(subset & active)),
        }
    return {
        "rows_total": n,
        "record_ids_unique": int(len(np.unique(ids_text))),
        "physical_rows_unique": int(len(np.unique(physical))),
        "origin_bundle_pairs_unique": int(len(np.unique(pairs, axis=0))),
        "present_rows": int((~missing_b).sum()),
        "missing_rows": int(missing_b.sum()),
        "present_b_positive_rows": int(active.sum()),
        "present_b_zero_rows": int(((~missing_b) & (basis == 0.0)).sum()),
        "causal_rule": "every OOF prediction origin is strictly earlier than its target cycle",
        "causal_boundary_pass": bool(np.all(cycles > origins)),
        "origins": by_origin,
        "identity_hashes": {name: _hash_array(arrays[name]) for name in runner.IDENTITY_FIELDS},
        "loaded_non_target_members": list(runner.IDENTITY_FIELDS),
        "targets_member_present_but_not_loaded": True,
    }


def _current_inventory(layout: runner.Layout) -> dict[str, Any]:
    sources = {}
    inputs = {}
    for destination, paths in ((sources, layout.source_paths), (inputs, layout.input_paths)):
        for name, raw_path in paths.items():
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise runner.GateError(f"Independent verifier missing pinned file: {name}")
            destination[name] = {"path": str(path), "sha256": _sha(path)}
    return {"sources": sources, "inputs": inputs}


def _verify_producer(layout: runner.Layout, inventory: Mapping[str, Any]) -> dict[str, Any]:
    input_pins = inventory["inputs"]
    if not layout.synthetic:
        for key, digest in runner.EXPECTED_REAL_INPUT_HASHES.items():
            if input_pins[key]["sha256"] != digest:
                raise runner.GateError(f"Independent producer pin mismatch: {key}")
    paths = layout.input_paths
    parent_freeze = _read(paths["parent_freeze"], signed=True)
    parent_pass = _read(paths["parent_pass"], signed=True)
    parent_preflight = _read(paths["parent_preflight"], signed=True)
    parent_report = _read(paths["parent_report"], signed=True)
    fit_state = _read(paths["parent_fit_state"])
    verification = _read(paths["parent_verification"])
    receipt = _read(paths["parent_receipt"])
    blocked = _read(paths["blocked_r1"])
    freeze_sha = input_pins["parent_freeze"]["sha256"]
    pass_sha = input_pins["parent_pass"]["sha256"]
    preflight_sha = input_pins["parent_preflight"]["sha256"]
    report_sha = input_pins["parent_report"]["sha256"]
    state_sha = input_pins["parent_fit_state"]["sha256"]
    slot = parent_report.get("final_slot", {})
    bundle_path = Path(input_pins["oof_bundle"]["path"]).resolve()
    if (parent_pass.get("verdict") != "PASS"
            or parent_pass.get("source_commit") != parent_report.get("source_commit")
            or parent_pass.get("prefit_freeze_sha256") != freeze_sha
            or parent_pass.get("identity_preflight_sha256") != preflight_sha
            or parent_preflight.get("prefit_freeze_sha256") != freeze_sha
            or parent_preflight.get("source_commit") != parent_report.get("source_commit")
            or parent_report.get("prefit_freeze_sha256") != freeze_sha
            or parent_report.get("identity_preflight_sha256") != preflight_sha
            or parent_report.get("new_root_source_pass_sha256") != pass_sha
            or parent_report.get("status") != "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION"
            or slot.get("fit_id") != "HINGE_FINAL"
            or Path(slot.get("fit_input_path", "")).resolve() != bundle_path
            or slot.get("fit_input_sha256") != input_pins["oof_bundle"]["sha256"]
            or slot.get("fit_state_sha256") != state_sha
            or slot.get("coefficients") != runner.HINGE
            or fit_state.get("status") != "LP_COMPLETE"
            or fit_state.get("fit_id") != "HINGE_FINAL"
            or fit_state.get("root_source_pass_sha256") != pass_sha
            or verification.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or verification.get("final_report_sha256") != report_sha
            or verification.get("source_pass_sha256") != pass_sha
            or receipt.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or receipt.get("final_report_sha256") != report_sha
            or receipt.get("verification_sha256") != input_pins["parent_verification"]["sha256"]
            or receipt.get("fit_state_sha256") != state_sha
            or blocked.get("status") != "BLOCKED_SOURCE_POPULATION_AMBIGUITY"):
        raise runner.GateError("Independent FULLREG producer ancestry check failed")
    producer_commit = str(parent_report.get("source_commit", ""))
    if not producer_commit or parent_freeze.get("source_commit") != producer_commit:
        raise runner.GateError("Independent producer commit binding failed")
    if not layout.synthetic:
        result = subprocess.run(["git", "-C", str(layout.root), "merge-base", "--is-ancestor",
                                 producer_commit, "HEAD"], capture_output=True, check=False)
        if result.returncode:
            raise runner.GateError("Independent verifier cannot establish producer ancestry")
    return {
        "producer_source_commit": producer_commit,
        "producer_source_commit_is_ancestor": not layout.synthetic,
        "parent_freeze_sha256": freeze_sha,
        "parent_source_pass_sha256": pass_sha,
        "parent_identity_preflight_sha256": preflight_sha,
        "parent_final_report_sha256": report_sha,
        "parent_fit_state_sha256": state_sha,
        "oof_bundle_sha256": input_pins["oof_bundle"]["sha256"],
        "blocked_r1_sha256": input_pins["blocked_r1"]["sha256"],
        "fixed_hinge_coefficients": dict(runner.HINGE),
    }


def verify_source(root: Path, *, synthetic: bool = False) -> dict[str, Any]:
    layout = runner.make_layout(root, synthetic=synthetic)
    freeze = _read(layout.freeze, signed=True)
    preflight = _read(layout.preflight, signed=True)
    inventory = _current_inventory(layout)
    boundary = freeze.get("access_boundary", {})
    if (freeze.get("status") != "SOURCE_FROZEN_PENDING_NON_TARGET_PREFLIGHT"
            or freeze.get("experiment_id") != runner.EXPERIMENT_ID
            or freeze.get("source_inventory") != inventory
            or freeze.get("source_inventory_sha256") != _canonical(inventory)):
        raise runner.GateError("Independent source/input byte pins differ from freeze")
    if any(boundary.get(key) is not False for key in (
            "targets_loaded", "e79_read", "test_or_sample_read", "csv_created",
            "kaggle_accessed", "fit_started")):
        raise runner.GateError("Independent source freeze has an open access boundary")
    producer = _verify_producer(layout, inventory)
    if freeze.get("producer_ancestry") != producer:
        raise runner.GateError("Independent producer ancestry differs from source freeze")
    identity = _identity(layout.oof_bundle, synthetic=synthetic)
    if (preflight.get("status") != "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS"
            or preflight.get("experiment_id") != runner.EXPERIMENT_ID
            or preflight.get("preparation_freeze_sha256") != _sha(layout.freeze)
            or preflight.get("source_inventory_sha256") != freeze.get("source_inventory_sha256")
            or preflight.get("producer_source_commit") != producer["producer_source_commit"]
            or preflight.get("identity") != identity
            or preflight.get("fixed_hinge_coefficients") != runner.HINGE
            or preflight.get("actions", {}).get("target_array_loaded") is not False
            or preflight.get("actions", {}).get("target_values_loaded") is not False
            or preflight.get("actions", {}).get("fit_started") is not False
            or preflight.get("actions", {}).get("e79_read") is not False
            or preflight.get("actions", {}).get("test_or_sample_read") is not False
            or preflight.get("actions", {}).get("csv_created") is not False
            or preflight.get("actions", {}).get("kaggle_accessed") is not False):
        raise runner.GateError("Independent target-blind preflight replay failed")
    receipt_body = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_SOURCE_READY_VERIFICATION",
        "experiment_id": runner.EXPERIMENT_ID,
        "preparation_freeze_sha256": _sha(layout.freeze),
        "identity_preflight_sha256": _sha(layout.preflight),
        "source_inventory_sha256": freeze["source_inventory_sha256"],
        "identity_summary_sha256": _canonical(identity),
        "rows_total": identity["rows_total"],
        "present_rows": identity["present_rows"],
        "missing_rows": identity["missing_rows"],
        "causal_boundary_pass": identity["causal_boundary_pass"],
        "loaded_npz_members": list(runner.IDENTITY_FIELDS),
        "target_array_loaded": False,
        "fit_started": False,
        "e79_read": False,
        "test_or_sample_read": False,
        "csv_created": False,
        "kaggle_accessed": False,
    }
    receipt_path = layout.source_verification
    if receipt_path.exists():
        existing = _read(receipt_path, signed=True)
        existing_body = dict(existing)
        existing_body.pop("signature", None)
        if existing_body != receipt_body:
            raise runner.GateError("Existing source verification receipt conflicts with replay")
        return existing
    signed = dict(receipt_body)
    signed["signature"] = _canonical(signed)
    runner.write_once(receipt_path, signed)
    return signed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--real", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--synthetic-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    result = verify_source(args.root.resolve(), synthetic=args.synthetic_fixture)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except runner.GateError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
