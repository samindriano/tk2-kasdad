"""Source-gated single scalar center-local calibration for the frozen FULLREG hinge.

This lane is deliberately split: --freeze, --preflight, and --verify-source are
target-blind. A one-fit, target-access pass is required before --fit. E79/Test,
submission CSV, and Kaggle access are never part of this experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REG_CENTER_LOCAL_APPLICATION_V1_R3"
NAMESPACE = "reg_center_local_application_v1"
FIT_ID = "LOCAL_CENTER_FINAL"
OOF_PATH = Path("experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz")
PARENT_REPORT_PATH = Path("reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_REPORT_R1.json")
PARENT_FREEZE_PATH = Path("experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_FREEZE_R1.json")
PARENT_PASS_PATH = Path("experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_ROOT_SOURCE_PASS_R1.json")
PARENT_PREFLIGHT_PATH = Path("reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_PREFLIGHT_R1.json")
PARENT_VERIFICATION_PATH = Path("reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_VERIFICATION_R1.json")
PARENT_RECEIPT_PATH = Path("reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_RECEIPT_R1.json")
PARENT_FIT_STATE_PATH = Path("experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fits/HINGE_FINAL.json")
BLOCKED_R1_PATH = Path("reports/optimization_v3/reg_center_local_application_v1/SOURCE_PREP_BLOCKED_R1.json")

EXPECTED_REAL_INPUT_HASHES = {
    "oof_bundle": "e75f70cccd1bb231499060b84155af876173214d984461e39e4c1364e4c5b2bf",
    "parent_report": "fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73",
    "parent_freeze": "e3522ab957381dc32c1e37ba41863781a736291982b7c8f1364dd4d8c7def049",
    "parent_pass": "ca35d7f580ee2fd2c5e7173309e89482b69c3eba7a2c29a7adbd0a1c791acd2f",
    "parent_preflight": "281022a94eaf8dd07831d73476811e29217020d7dcee12a02f00564053847521",
    "parent_verification": "af859d6efcf90b3044ba55c37f2ae3e5e27132ae520c1cd90aad28580fadf97c",
    "parent_receipt": "364a0cbe8258127e6ff963f9117e11a8d61c1ec05bb3bc4d2ca5962f117ad89f",
    "parent_fit_state": "a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a",
    "blocked_r1": "a3c9377ba5a802c4f0ea84ef3589fcaf20a8932016e2d37e6f0368d1fc74eb0f",
    "prior_r2_freeze": "87c87b6e29e9a1638b615921e3dd0178388caceef7bd202bccdfe639993199d6",
    "prior_r2_preflight": "68e08cde4687d66ddebb9c2b9bc3c6cff2f52ed64340e5e479815a70516902b0",
    "prior_r2_verification": "8b13e5d233ff22c6e110ee50212066d9d679a73f61b4917f94b83925495b875f",
}

EXPECTED_TOTAL = 20_903
EXPECTED_PRESENT = 19_528
EXPECTED_MISSING = 1_375
EXPECTED_ORIGIN_COUNTS = {67: 12_412, 75: 8_491}
EXPECTED_PRESENT_BY_ORIGIN = {67: 11_563, 75: 7_965}
EXPECTED_CYCLE_COUNTS = {
    68: 577, 69: 1_079, 70: 1_605, 71: 1_689,
    72: 1_811, 73: 1_880, 74: 1_814, 75: 1_957,
    76: 1_822, 77: 2_335, 78: 2_146, 79: 2_188,
}
EXPECTED_CYCLES_BY_ORIGIN = {67: tuple(range(68, 76)), 75: tuple(range(76, 80))}
IDENTITY_FIELDS = (
    "physical_rows", "record_ids", "prediction_origins",
    "original_bundle_indices", "cycles", "direct_predictions", "missing_mask",
)
ARCHIVE_FIELDS = set(IDENTITY_FIELDS) | {"targets"}
HINGE = {"intercept": 1.7288079036674198,
         "slope_low": 0.9933285599956824,
         "slope_high": 0.957778072288735}
CENTER = 50.0
HALF_WIDTH = 10.0
DELTA_BOUNDS = (-2.0, 0.0)
ROOT_PASS_KEYS = {
    "verdict", "experiment_id", "pass_kind", "producer_source_commit",
    "producer_final_report_sha256", "preparation_freeze_sha256",
    "identity_preflight_sha256", "source_inventory_sha256",
    "authorized_fit_ids", "calibrator_fit_cap", "authorized_calibrator_fits",
    "base_model_fits", "preprocessing_fits", "official_lp_starts",
    "fit_total_oof_rows", "fit_present_rows", "excluded_missing_rows",
    "target_array_access_authorized", "e79_access_authorized",
    "test_or_sample_access_authorized", "csv_creation_authorized",
    "kaggle_upload_authorized", "signature",
}

TARGET_MEMBER_READ_COUNT = 0


class GateError(RuntimeError):
    """Fail-closed source, identity, authorization, or durable-state failure."""


@dataclass(frozen=True)
class Layout:
    root: Path
    synthetic: bool = False

    @property
    def experiment_dir(self) -> Path:
        return self.root / "experiments" / NAMESPACE

    @property
    def report_dir(self) -> Path:
        return self.root / "reports" / "optimization_v3" / NAMESPACE

    @property
    def freeze(self) -> Path:
        return self.experiment_dir / "PREFIT_FREEZE_R3.json"

    @property
    def preflight(self) -> Path:
        return self.report_dir / "IDENTITY_PREFLIGHT_R3.json"

    @property
    def source_verification(self) -> Path:
        return self.report_dir / "SOURCE_READY_VERIFICATION_R3.json"

    @property
    def root_pass(self) -> Path:
        return self.experiment_dir / "ROOT_SOURCE_PASS_R3.json"

    @property
    def fit_dir(self) -> Path:
        return self.experiment_dir / "fits_R3"

    @property
    def fit_state(self) -> Path:
        return self.fit_dir / f"{FIT_ID}.json"

    @property
    def fit_receipt(self) -> Path:
        return self.fit_dir / f"{FIT_ID}_RECEIPT.json"

    @property
    def fit_ledger(self) -> Path:
        return self.experiment_dir / "calibrator_ledger_R3.jsonl"

    @property
    def oof_bundle(self) -> Path:
        return self.root / OOF_PATH

    @property
    def blocked_r1(self) -> Path:
        return self.root / BLOCKED_R1_PATH

    @property
    def input_paths(self) -> dict[str, Path]:
        return {
            "oof_bundle": self.oof_bundle,
            "parent_report": self.root / PARENT_REPORT_PATH,
            "parent_freeze": self.root / PARENT_FREEZE_PATH,
            "parent_pass": self.root / PARENT_PASS_PATH,
            "parent_preflight": self.root / PARENT_PREFLIGHT_PATH,
            "parent_verification": self.root / PARENT_VERIFICATION_PATH,
            "parent_receipt": self.root / PARENT_RECEIPT_PATH,
            "parent_fit_state": self.root / PARENT_FIT_STATE_PATH,
            "blocked_r1": self.blocked_r1,
            "prior_r2_freeze": self.root / "experiments/reg_center_local_application_v1/PREFIT_FREEZE_R2.json",
            "prior_r2_preflight": self.root / "reports/optimization_v3/reg_center_local_application_v1/IDENTITY_PREFLIGHT_R2.json",
            "prior_r2_verification": self.root / "reports/optimization_v3/reg_center_local_application_v1/SOURCE_READY_VERIFICATION_R2.json",
        }

    @property
    def source_paths(self) -> dict[str, Path]:
        return {
            "runner": self.root / "scripts/reg_center_local_application_v1.py",
            "verifier": self.root / "scripts/verify_reg_center_local_application_v1.py",
            "tests": self.root / "tests/test_reg_center_local_application_v1.py",
            "policy": self.root / "reports/optimization_v3/reg_center_local_application_v1/POLICY_R3.md",
            "root_pass_contract": self.root / "reports/optimization_v3/reg_center_local_application_v1/ROOT_SOURCE_PASS_CONTRACT_R3.md",
            "parent_final_runner": self.root / "scripts/reg_monotone_hinge_p50_final_application_v1.py",
            "parent_final_verifier": self.root / "scripts/verify_reg_monotone_hinge_p50_final_application_v1.py",
            "hinge_core": self.root / "scripts/reg_monotone_hinge_p50_v1.py",
        }


def make_layout(root: Path = ROOT, *, synthetic: bool = False) -> Layout:
    return Layout(Path(root).resolve(), synthetic)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _signed_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("signature", None)
    return body


def read_signed(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"Cannot read signed JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("signature") != canonical_sha(_signed_body(value)):
        raise GateError(f"Canonical signature mismatch: {path}")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise GateError(f"Refusing to overwrite immutable receipt: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _write_signed_once(path: Path, body: Mapping[str, Any]) -> str:
    complete = dict(body)
    complete["signature"] = canonical_sha(complete)
    return write_once(path, complete)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value


def _git_head(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _is_ancestor(root: Path, commit: str) -> bool:
    result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                             commit, "HEAD"], capture_output=True, check=False)
    return result.returncode == 0


def _identity_hash(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise GateError("Object identity arrays are not permitted")
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _inventory(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise GateError(f"Pinned source/input is missing: {name} ({path})")
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _verify_parent_chain(layout: Layout) -> dict[str, Any]:
    pins = _inventory(layout.input_paths)
    if not layout.synthetic:
        for name, expected in EXPECTED_REAL_INPUT_HASHES.items():
            observed = pins[name]["sha256"]
            if observed != expected:
                raise GateError(f"Pinned FULLREG producer input changed: {name}")

    freeze = read_signed(layout.input_paths["parent_freeze"])
    parent_pass = read_signed(layout.input_paths["parent_pass"])
    preflight = read_signed(layout.input_paths["parent_preflight"])
    report = read_signed(layout.input_paths["parent_report"])
    fit_state = _json(layout.input_paths["parent_fit_state"])
    verification = _json(layout.input_paths["parent_verification"])
    receipt = _json(layout.input_paths["parent_receipt"])
    blocked = _json(layout.input_paths["blocked_r1"])

    freeze_sha = pins["parent_freeze"]["sha256"]
    pass_sha = pins["parent_pass"]["sha256"]
    preflight_sha = pins["parent_preflight"]["sha256"]
    report_sha = pins["parent_report"]["sha256"]
    state_sha = pins["parent_fit_state"]["sha256"]
    bundle_sha = pins["oof_bundle"]["sha256"]
    slot = report.get("final_slot", {})
    if (freeze.get("experiment_id") != "REG_MONOTONE_HINGE_P50_FINAL_APPLICATION_R1"
            or parent_pass.get("verdict") != "PASS"
            or parent_pass.get("experiment_id") != freeze.get("experiment_id")
            or parent_pass.get("source_commit") != report.get("source_commit")
            or parent_pass.get("prefit_freeze_sha256") != freeze_sha
            or parent_pass.get("identity_preflight_sha256") != preflight_sha
            or preflight.get("prefit_freeze_sha256") != freeze_sha
            or preflight.get("source_commit") != report.get("source_commit")
            or preflight.get("status") != "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT"
            or report.get("status") != "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION"
            or report.get("prefit_freeze_sha256") != freeze_sha
            or report.get("identity_preflight_sha256") != preflight_sha
            or report.get("new_root_source_pass_sha256") != pass_sha
            or slot.get("fit_id") != "HINGE_FINAL"
            or Path(slot.get("fit_input_path", "")).resolve() != layout.oof_bundle.resolve()
            or slot.get("fit_input_sha256") != bundle_sha
            or slot.get("fit_state_sha256") != state_sha
            or slot.get("coefficients") != HINGE
            or fit_state.get("status") != "LP_COMPLETE"
            or fit_state.get("fit_id") != "HINGE_FINAL"
            or fit_state.get("root_source_pass_sha256") != pass_sha
            or verification.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or verification.get("final_report_sha256") != report_sha
            or verification.get("source_pass_sha256") != pass_sha
            or receipt.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or receipt.get("final_report_sha256") != report_sha
            or receipt.get("verification_sha256") != pins["parent_verification"]["sha256"]
            or receipt.get("fit_state_sha256") != state_sha
            or blocked.get("status") != "BLOCKED_SOURCE_POPULATION_AMBIGUITY"):
        raise GateError("Pinned FULLREG final-hinge producer ancestry failed closed")

    producer_commit = str(report.get("source_commit", ""))
    if not producer_commit or producer_commit != freeze.get("source_commit"):
        raise GateError("Producer source commit is missing or inconsistent")
    if not layout.synthetic and not _is_ancestor(layout.root, producer_commit):
        raise GateError("Frozen final-hinge producer commit is not an ancestor of this checkout")

    return {
        "producer_source_commit": producer_commit,
        "producer_source_commit_is_ancestor": not layout.synthetic,
        "parent_freeze_sha256": freeze_sha,
        "parent_source_pass_sha256": pass_sha,
        "parent_identity_preflight_sha256": preflight_sha,
        "parent_final_report_sha256": report_sha,
        "parent_fit_state_sha256": state_sha,
        "oof_bundle_sha256": bundle_sha,
        "blocked_r1_sha256": pins["blocked_r1"]["sha256"],
        "fixed_hinge_coefficients": dict(HINGE),
    }


def _source_paths(layout: Layout) -> dict[str, Path]:
    return layout.source_paths


def _all_inventory(layout: Layout) -> dict[str, Any]:
    return {
        "sources": _inventory(_source_paths(layout)),
        "inputs": _inventory(layout.input_paths),
    }


def create_freeze(layout: Layout) -> dict[str, Any]:
    producer = _verify_parent_chain(layout)
    inventory = _all_inventory(layout)
    source_inventory_sha = canonical_sha(inventory)
    body = {
        "schema_version": 2,
        "status": "SOURCE_FROZEN_PENDING_NON_TARGET_PREFLIGHT",
        "experiment_id": EXPERIMENT_ID,
        "prepared_on_head_informational_only": _git_head(layout.root),
        "source_inventory": inventory,
        "source_inventory_sha256": source_inventory_sha,
        "producer_ancestry": producer,
        "population_resolution": {
            "requested_universe_rows": EXPECTED_TOTAL,
            "eligible_present_rows": EXPECTED_PRESENT,
            "missing_rows_excluded_from_fit": EXPECTED_MISSING,
            "interpretation": "All unique OOF rows define the universe; only rows with missing_mask=false are fit-eligible.",
        },
        "fit_contract": {
            "fit_id": FIT_ID,
            "base_model": "frozen FULLREG HINGE_FINAL",
            "new_base_model_fits": 0,
            "preprocessing_fits": 0,
            "local_basis": "max(0, 1 - abs(raw_DIRECT - 50) / 10)",
            "delta_bounds": list(DELTA_BOUNDS),
            "optimizer": "exact lower weighted median of (y-g(raw_DIRECT))/b with weights b over present b>0 rows",
            "zero_basis_rows": "contribute no choice-dependent objective",
            "missing_rows": "excluded from fit and unchanged at application",
        },
        "access_boundary": {
            "targets_loaded": False,
            "e79_read": False,
            "test_or_sample_read": False,
            "csv_created": False,
            "kaggle_accessed": False,
            "fit_started": False,
        },
    }
    _write_signed_once(layout.freeze, body)
    return _json(layout.freeze)


def _verify_freeze(layout: Layout) -> dict[str, Any]:
    freeze = read_signed(layout.freeze)
    inventory = _all_inventory(layout)
    boundary = freeze.get("access_boundary", {})
    if (freeze.get("experiment_id") != EXPERIMENT_ID
            or freeze.get("status") != "SOURCE_FROZEN_PENDING_NON_TARGET_PREFLIGHT"
            or freeze.get("source_inventory") != inventory
            or freeze.get("source_inventory_sha256") != canonical_sha(inventory)
            or freeze.get("producer_ancestry") != _verify_parent_chain(layout)
            or any(boundary.get(key) is not False for key in (
                "targets_loaded", "e79_read", "test_or_sample_read", "csv_created",
                "kaggle_accessed", "fit_started"))):
        raise GateError("Source freeze, file pins, or parent ancestry changed")
    return freeze


def _load_non_target_identity(path: Path) -> dict[str, np.ndarray]:
    expected_members = {f"{name}.npy" for name in ARCHIVE_FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise GateError(f"Cannot inspect frozen OOF NPZ members: {path}") from exc
    if members != expected_members:
        raise GateError(f"Frozen OOF NPZ schema mismatch: missing={expected_members-members}, extra={members-expected_members}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            # The targets member is intentionally not indexed in this function.
            return {name: np.asarray(archive[name]).copy() for name in IDENTITY_FIELDS}
    except (OSError, ValueError, KeyError) as exc:
        raise GateError(f"Cannot load non-target OOF identity arrays: {path}") from exc


def local_basis(direct: Any) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise GateError("DIRECT predictions must be finite one-dimensional float64 values")
    return np.maximum(0.0, 1.0 - np.abs(p - CENTER) / HALF_WIDTH)


def hinge_prediction(direct: Any, coefficients: Mapping[str, float] = HINGE) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise GateError("DIRECT predictions must be finite one-dimensional float64 values")
    return (float(coefficients["intercept"])
            + float(coefficients["slope_low"]) * np.minimum(p, CENTER)
            + float(coefficients["slope_high"]) * np.maximum(p - CENTER, 0.0))


def _validate_identity(arrays: Mapping[str, np.ndarray], *, synthetic: bool = False) -> dict[str, Any]:
    values = {name: np.asarray(arrays[name]) for name in IDENTITY_FIELDS}
    n_values = {len(value) for value in values.values() if value.ndim == 1}
    if any(value.ndim != 1 for value in values.values()) or len(n_values) != 1:
        raise GateError("OOF non-target identity arrays are not aligned vectors")
    n = n_values.pop()
    ids = values["record_ids"]
    physical = values["physical_rows"]
    origins = values["prediction_origins"]
    bundle_idx = values["original_bundle_indices"]
    cycles = values["cycles"]
    direct = values["direct_predictions"]
    missing = values["missing_mask"]
    if (ids.dtype.hasobject or ids.dtype.kind not in "US"
            or physical.dtype.kind not in "iu" or origins.dtype.kind not in "iu"
            or bundle_idx.dtype.kind not in "iu" or cycles.dtype.kind not in "iu"
            or direct.dtype.kind != "f" or direct.dtype.itemsize != 8
            or missing.dtype.kind != "b" or not np.isfinite(direct).all()
            or np.any((direct < 0.0) | (direct > 100.0))):
        raise GateError("OOF non-target identity dtypes or values are invalid")
    ids_text = ids.astype(str)
    origin_i = origins.astype(np.int64, copy=False)
    cycle_i = cycles.astype(np.int64, copy=False)
    missing_b = missing.astype(bool, copy=False)
    if (len(np.unique(ids_text)) != n or len(np.unique(physical)) != n
            or len(np.unique(np.stack([origin_i, bundle_idx.astype(np.int64)], axis=1), axis=0)) != n
            or np.any(cycle_i <= origin_i)):
        raise GateError("OOF record/origin identity or causal boundary failed")
    origin_counts = {int(k): int(v) for k, v in zip(*np.unique(origin_i, return_counts=True))}
    present_by_origin = {int(k): int(np.sum((origin_i == k) & (~missing_b))) for k in np.unique(origin_i)}
    missing_by_origin = {int(k): int(np.sum((origin_i == k) & missing_b)) for k in np.unique(origin_i)}
    cycle_counts = {int(k): int(v) for k, v in zip(*np.unique(cycle_i, return_counts=True))}
    cycles_by_origin = {
        int(origin): tuple(int(x) for x in np.unique(cycle_i[origin_i == origin]))
        for origin in np.unique(origin_i)
    }
    if not synthetic:
        if (n != EXPECTED_TOTAL or int((~missing_b).sum()) != EXPECTED_PRESENT
                or int(missing_b.sum()) != EXPECTED_MISSING
                or origin_counts != EXPECTED_ORIGIN_COUNTS
                or present_by_origin != EXPECTED_PRESENT_BY_ORIGIN
                or cycle_counts != EXPECTED_CYCLE_COUNTS
                or cycles_by_origin != EXPECTED_CYCLES_BY_ORIGIN):
            raise GateError("Frozen OOF population, origin, or cycle counts differ from the resolved scope")
    basis = local_basis(direct)
    active = (~missing_b) & (basis > 0.0)
    origin_summary = {}
    for origin in sorted(origin_counts):
        mask_origin = origin_i == origin
        origin_summary[str(origin)] = {
            "all_rows": origin_counts[origin],
            "present_rows": present_by_origin[origin],
            "missing_rows": missing_by_origin[origin],
            "cycles": list(cycles_by_origin[origin]),
            "cycle_counts": {str(c): int(np.sum(mask_origin & (cycle_i == c)))
                             for c in cycles_by_origin[origin]},
            "unique_record_ids": int(len(np.unique(ids_text[mask_origin]))),
            "unique_origin_bundle_pairs": int(len(np.unique(np.stack([
                origin_i[mask_origin], bundle_idx[mask_origin].astype(np.int64)
            ], axis=1), axis=0))),
            "present_b_positive_rows": int(np.sum(mask_origin & active)),
        }
    return {
        "rows_total": n,
        "record_ids_unique": int(len(np.unique(ids_text))),
        "physical_rows_unique": int(len(np.unique(physical))),
        "origin_bundle_pairs_unique": int(len(np.unique(np.stack([origin_i, bundle_idx.astype(np.int64)], axis=1), axis=0))),
        "present_rows": int((~missing_b).sum()),
        "missing_rows": int(missing_b.sum()),
        "present_b_positive_rows": int(active.sum()),
        "present_b_zero_rows": int(((~missing_b) & (basis == 0.0)).sum()),
        "causal_rule": "every OOF prediction origin is strictly earlier than its target cycle",
        "causal_boundary_pass": bool(np.all(cycle_i > origin_i)),
        "origins": origin_summary,
        "identity_hashes": {name: _identity_hash(values[name]) for name in IDENTITY_FIELDS},
        "loaded_non_target_members": list(IDENTITY_FIELDS),
        "targets_member_present_but_not_loaded": True,
    }


def create_preflight(layout: Layout) -> dict[str, Any]:
    freeze = _verify_freeze(layout)
    arrays = _load_non_target_identity(layout.oof_bundle)
    identity = _validate_identity(arrays, synthetic=layout.synthetic)
    body = {
        "schema_version": 2,
        "status": "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS",
        "experiment_id": EXPERIMENT_ID,
        "preparation_freeze_sha256": sha256_file(layout.freeze),
        "source_inventory_sha256": freeze["source_inventory_sha256"],
        "producer_source_commit": freeze["producer_ancestry"]["producer_source_commit"],
        "identity": identity,
        "fixed_hinge_coefficients": dict(HINGE),
        "fit_population": {
            "universe_rows": identity["rows_total"],
            "eligible_present_rows": identity["present_rows"],
            "excluded_missing_rows": identity["missing_rows"],
            "choice_informative_rows_present_and_b_positive": identity["present_b_positive_rows"],
            "zero_basis_rows_choice_independent": identity["present_b_zero_rows"],
        },
        "actions": {
            "target_array_loaded": False,
            "target_values_loaded": False,
            "fit_started": False,
            "e79_read": False,
            "test_or_sample_read": False,
            "csv_created": False,
            "kaggle_accessed": False,
        },
    }
    _write_signed_once(layout.preflight, body)
    return _json(layout.preflight)


def _root_pass_expected(layout: Layout, freeze: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "pass_kind": "SYNTHETIC_TEST_ONLY" if layout.synthetic else "ONE_LOCAL_SCALAR_FIT",
        "producer_source_commit": freeze["producer_ancestry"]["producer_source_commit"],
        "producer_final_report_sha256": freeze["producer_ancestry"]["parent_final_report_sha256"],
        "preparation_freeze_sha256": sha256_file(layout.freeze),
        "identity_preflight_sha256": sha256_file(layout.preflight),
        "source_inventory_sha256": freeze["source_inventory_sha256"],
        "authorized_fit_ids": [FIT_ID],
        "calibrator_fit_cap": 1,
        "authorized_calibrator_fits": 1,
        "base_model_fits": 0,
        "preprocessing_fits": 0,
        "official_lp_starts": 0,
        "fit_total_oof_rows": int(preflight["identity"]["rows_total"]),
        "fit_present_rows": int(preflight["identity"]["present_rows"]),
        "excluded_missing_rows": int(preflight["identity"]["missing_rows"]),
        "target_array_access_authorized": True,
        "e79_access_authorized": False,
        "test_or_sample_access_authorized": False,
        "csv_creation_authorized": False,
        "kaggle_upload_authorized": False,
    }


def validate_root_pass(layout: Layout) -> tuple[dict[str, Any], str]:
    freeze = _verify_freeze(layout)
    preflight = read_signed(layout.preflight)
    body = read_signed(layout.root_pass)
    if set(body) != ROOT_PASS_KEYS:
        raise GateError("Root source pass schema contains missing or extra fields")
    expected = _root_pass_expected(layout, freeze, preflight)
    observed = _signed_body(body)
    if observed != expected:
        raise GateError("Root source pass does not authorize exactly the frozen one-fit scope")
    expected_present = int(preflight.get("identity", {}).get("present_rows", -1))
    if (preflight.get("status") != "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS"
            or preflight.get("actions", {}).get("target_values_loaded") is not False
            or preflight.get("fit_population", {}).get("eligible_present_rows") != expected_present
            or (not layout.synthetic and expected_present != EXPECTED_PRESENT)):
        raise GateError("Target-blind identity preflight is not the resolved present-only pass")
    return body, sha256_file(layout.root_pass)


def lower_weighted_median_interval(values: Any, weights: Any) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 1 or w.ndim != 1 or x.shape != w.shape or len(x) == 0
            or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0.0)):
        raise GateError("Weighted lower median requires aligned finite values and positive weights")
    ratios = [float(weight).as_integer_ratio() for weight in w]
    denominator = max(den for _, den in ratios)
    integer_weights = [num * (denominator // den) for num, den in ratios]
    order = np.argsort(x, kind="mergesort")
    grouped_values: list[float] = []
    grouped_weights: list[int] = []
    for index in order.tolist():
        value = float(x[index])
        weight = integer_weights[index]
        if grouped_values and grouped_values[-1] == value:
            grouped_weights[-1] += weight
        else:
            grouped_values.append(value)
            grouped_weights.append(weight)
    total = sum(grouped_weights)
    cumulative = 0
    lower = None
    upper = None
    for value, weight in zip(grouped_values, grouped_weights):
        cumulative += weight
        if lower is None and 2 * cumulative >= total:
            lower = value
        if 2 * cumulative > total:
            upper = value
            break
    if lower is None:
        raise GateError("Weighted lower median accumulation failed")
    if upper is None:
        upper = grouped_values[-1]
    return float(lower), float(upper), len(x)


def _bounded_interval(low: float, high: float) -> tuple[float, float]:
    return (float(np.clip(low, *DELTA_BOUNDS)), float(np.clip(high, *DELTA_BOUNDS)))


def apply_local(direct: Any, missing_mask: Any, delta: float) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    missing = np.asarray(missing_mask)
    if p.ndim != 1 or missing.ndim != 1 or p.shape != missing.shape or missing.dtype.kind != "b":
        raise GateError("Application requires aligned DIRECT and boolean missing arrays")
    result = p.copy()
    present = ~missing
    base = hinge_prediction(p[present])
    corrected = base + float(delta) * local_basis(p[present])
    result[present] = np.clip(corrected, 0.0, 100.0)
    if not np.array_equal(result[missing].view(np.uint64), p[missing].view(np.uint64)):
        raise GateError("Missing route did not preserve raw DIRECT float64 bits")
    return result


def _append_ledger(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _ledger_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise GateError("Fit ledger is malformed; refusing retry") from exc
        if not isinstance(value, dict):
            raise GateError("Fit ledger event is not an object")
        events.append(value)
    return events


def _read_target_member_after_pass(path: Path, expected_n: int) -> np.ndarray:
    global TARGET_MEMBER_READ_COUNT
    TARGET_MEMBER_READ_COUNT += 1
    try:
        with np.load(path, allow_pickle=False) as archive:
            y = np.asarray(archive["targets"]).copy()
    except (OSError, ValueError, KeyError) as exc:
        raise GateError("Authorized target member could not be loaded") from exc
    if (y.ndim != 1 or len(y) != expected_n or y.dtype.kind != "f"
            or y.dtype.itemsize != 8 or not np.isfinite(y).all()
            or np.any((y < 0.0) | (y > 100.0))):
        raise GateError("Authorized target member shape, dtype, or range failed")
    return y


def _validate_completed_fit(layout: Layout, pass_sha: str) -> dict[str, Any]:
    state = read_signed(layout.fit_state)
    receipt = read_signed(layout.fit_receipt)
    state_sha = sha256_file(layout.fit_state)
    events = _ledger_events(layout.fit_ledger)
    if (state.get("status") != "PASS_LOCAL_SCALAR_FIT_COMPLETE"
            or state.get("fit_id") != FIT_ID
            or state.get("root_source_pass_sha256") != pass_sha
            or state.get("preparation_freeze_sha256") != sha256_file(layout.freeze)
            or state.get("identity_preflight_sha256") != sha256_file(layout.preflight)
            or state.get("fit_population", {}).get("eligible_present_rows")
            != int(read_signed(layout.preflight)["identity"]["present_rows"])
            or receipt.get("status") != "PASS_DURABLE_LOCAL_FIT_RECEIPT"
            or receipt.get("fit_state_sha256") != state_sha
            or receipt.get("root_source_pass_sha256") != pass_sha
            or not events or events[-1].get("event") != "FIT_COMPLETE"
            or events[-1].get("fit_state_sha256") != state_sha):
        raise GateError("Completed fit state, receipt, or ledger does not bind")
    return {"status": receipt["status"], "fit_state_sha256": state_sha,
            "fit_receipt_sha256": sha256_file(layout.fit_receipt),
            "delta": state["delta"], "resumed": True}


def fit_local(layout: Layout) -> dict[str, Any]:
    freeze = _verify_freeze(layout)
    preflight = read_signed(layout.preflight)
    root_pass, pass_sha = validate_root_pass(layout)
    if layout.fit_receipt.exists():
        return _validate_completed_fit(layout, pass_sha)
    if layout.fit_state.exists():
        raise GateError("Fit state exists without its receipt; refusing a second fit")
    prior_events = _ledger_events(layout.fit_ledger)
    if prior_events:
        raise GateError("A fit reservation already exists without a complete durable result")

    arrays = _load_non_target_identity(layout.oof_bundle)
    identity = _validate_identity(arrays, synthetic=layout.synthetic)
    if identity != preflight.get("identity"):
        raise GateError("OOF identity changed between preflight and fit boundary")
    if (sha256_file(layout.oof_bundle)
            != freeze["producer_ancestry"]["oof_bundle_sha256"]):
        raise GateError("Frozen OOF bundle changed before target access")
    reservation = {
        "event": "FIT_RESERVED",
        "experiment_id": EXPERIMENT_ID,
        "fit_id": FIT_ID,
        "root_source_pass_sha256": pass_sha,
        "preparation_freeze_sha256": sha256_file(layout.freeze),
        "identity_preflight_sha256": sha256_file(layout.preflight),
        "target_array_access_authorized": root_pass["target_array_access_authorized"],
    }
    _append_ledger(layout.fit_ledger, reservation)

    y = _read_target_member_after_pass(layout.oof_bundle, identity["rows_total"])
    direct = arrays["direct_predictions"].astype(np.float64, copy=False)
    missing = arrays["missing_mask"].astype(bool, copy=False)
    basis = local_basis(direct)
    active = (~missing) & (basis > 0.0)
    if not np.any(active):
        raise GateError("No present positive-basis rows; scalar delta is not identified")
    base = hinge_prediction(direct)
    ratios = (y[active] - base[active]) / basis[active]
    raw_low, raw_high, active_n = lower_weighted_median_interval(ratios, basis[active])
    bounded_low, bounded_high = _bounded_interval(raw_low, raw_high)
    delta = bounded_low
    active_objective = float(np.abs((y[active] - base[active]) - delta * basis[active]).sum())
    state_body = {
        "schema_version": 2,
        "status": "PASS_LOCAL_SCALAR_FIT_COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "fit_id": FIT_ID,
        "root_source_pass_sha256": pass_sha,
        "preparation_freeze_sha256": sha256_file(layout.freeze),
        "identity_preflight_sha256": sha256_file(layout.preflight),
        "oof_bundle_sha256": sha256_file(layout.oof_bundle),
        "fixed_hinge_coefficients": dict(HINGE),
        "basis": "max(0, 1 - abs(raw_DIRECT - 50) / 10)",
        "delta_bounds": list(DELTA_BOUNDS),
        "optimizer": "exact lower weighted median of (y-g(raw_DIRECT))/b, weights=b, on present b>0 rows",
        "delta_unbounded_minimizer_interval": [raw_low, raw_high],
        "delta_bounded_minimizer_interval": [bounded_low, bounded_high],
        "delta": delta,
        "fit_population": {
            "universe_rows": identity["rows_total"],
            "eligible_present_rows": identity["present_rows"],
            "excluded_missing_rows": identity["missing_rows"],
            "choice_informative_rows_present_and_b_positive": active_n,
            "zero_basis_rows_choice_independent": identity["present_b_zero_rows"],
        },
        "choice_dependent_objective": {
            "population": "present b>0 only",
            "unweighted_absolute_residual_sum": active_objective,
            "b_zero_rows_contribute": "no choice-dependent objective",
        },
        "target_access": {
            "authorized_target_member": "targets",
            "target_array_loaded_after_pass": True,
            "target_values_in_fit_objective": active_n,
            "missing_target_values_used": False,
            "e79_read": False,
            "test_or_sample_read": False,
        },
        "application": {
            "present_route": "clip(frozen hinge(raw_DIRECT) + delta*b(raw_DIRECT), 0, 100)",
            "missing_route": "unchanged raw DIRECT float64 bits",
            "application_performed": False,
            "csv_created": False,
        },
    }
    state_sha = _write_signed_once(layout.fit_state, state_body)
    receipt_body = {
        "schema_version": 1,
        "status": "PASS_DURABLE_LOCAL_FIT_RECEIPT",
        "experiment_id": EXPERIMENT_ID,
        "fit_id": FIT_ID,
        "root_source_pass_sha256": pass_sha,
        "preparation_freeze_sha256": sha256_file(layout.freeze),
        "identity_preflight_sha256": sha256_file(layout.preflight),
        "fit_state_sha256": state_sha,
        "ledger_reservation_sha256": canonical_sha(reservation),
        "delta": delta,
        "target_array_loaded_after_pass": True,
        "e79_read": False,
        "test_or_sample_read": False,
        "csv_created": False,
        "kaggle_upload": False,
    }
    receipt_sha = _write_signed_once(layout.fit_receipt, receipt_body)
    _append_ledger(layout.fit_ledger, {
        "event": "FIT_COMPLETE", "experiment_id": EXPERIMENT_ID,
        "fit_id": FIT_ID, "root_source_pass_sha256": pass_sha,
        "fit_state_sha256": state_sha, "fit_receipt_sha256": receipt_sha,
    })
    return {"status": "PASS_LOCAL_SCALAR_FIT_COMPLETE", "delta": delta,
            "fit_state_sha256": state_sha, "fit_receipt_sha256": receipt_sha,
            "resumed": False}


def verify_fit(layout: Layout) -> dict[str, Any]:
    _, pass_sha = validate_root_pass(layout)
    return _validate_completed_fit(layout, pass_sha)


def _verify_source_subprocess(layout: Layout) -> dict[str, Any]:
    verifier = layout.root / "scripts/verify_reg_center_local_application_v1.py"
    result = subprocess.run([sys.executable, str(verifier), "--root", str(layout.root),
                             "--synthetic-fixture" if layout.synthetic else "--real"],
                            cwd=layout.root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GateError(f"Independent source verifier failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _synthetic_source_files(root: Path) -> None:
    layout = make_layout(root, synthetic=True)
    for path in layout.source_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / path.relative_to(root)
        if source.is_file():
            shutil.copyfile(source, path)
        else:
            path.write_text("synthetic pinned source\n", encoding="utf-8")


def _write_synthetic_parent_chain(layout: Layout) -> None:
    paths = layout.input_paths
    bundle = layout.oof_bundle
    bundle.parent.mkdir(parents=True, exist_ok=True)
    n = 400
    origins = np.concatenate([np.full(200, 67, dtype=np.int64), np.full(200, 75, dtype=np.int64)])
    cycles = np.concatenate([
        np.resize(np.arange(68, 76, dtype=np.int64), 200),
        np.resize(np.arange(76, 80, dtype=np.int64), 200),
    ])
    direct = np.linspace(35.0, 65.0, n, dtype=np.float64)
    missing = np.zeros(n, dtype=bool)
    missing[::20] = True
    basis = local_basis(direct)
    target = hinge_prediction(direct) - 0.6 * basis
    np.savez_compressed(bundle,
        physical_rows=np.arange(1000, 1000 + n, dtype=np.int64),
        record_ids=np.asarray([f"synthetic-oof-{i:04d}" for i in range(n)]),
        prediction_origins=origins,
        original_bundle_indices=np.concatenate([np.arange(200), np.arange(200)]).astype(np.int64),
        cycles=cycles,
        direct_predictions=direct,
        missing_mask=missing,
        targets=target.astype(np.float64),
    )
    paths["parent_freeze"].parent.mkdir(parents=True, exist_ok=True)
    paths["parent_pass"].parent.mkdir(parents=True, exist_ok=True)
    paths["parent_preflight"].parent.mkdir(parents=True, exist_ok=True)
    parent_freeze = {"experiment_id": "REG_MONOTONE_HINGE_P50_FINAL_APPLICATION_R1",
                     "source_commit": "synthetic-producer-commit", "status": "SYNTHETIC_PARENT_FREEZE"}
    _write_signed_once(paths["parent_freeze"], parent_freeze)
    parent_freeze_sha = sha256_file(paths["parent_freeze"])
    parent_preflight = {"experiment_id": parent_freeze["experiment_id"],
                        "status": "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT",
                        "source_commit": parent_freeze["source_commit"],
                        "prefit_freeze_sha256": parent_freeze_sha}
    _write_signed_once(paths["parent_preflight"], parent_preflight)
    parent_preflight_sha = sha256_file(paths["parent_preflight"])
    parent_pass = {"experiment_id": parent_freeze["experiment_id"], "verdict": "PASS",
                   "source_commit": parent_freeze["source_commit"],
                   "prefit_freeze_sha256": parent_freeze_sha,
                   "identity_preflight_sha256": parent_preflight_sha}
    _write_signed_once(paths["parent_pass"], parent_pass)
    parent_pass_sha = sha256_file(paths["parent_pass"])
    fit_state = {"status": "LP_COMPLETE", "fit_id": "HINGE_FINAL",
                 "root_source_pass_sha256": parent_pass_sha}
    write_once(paths["parent_fit_state"], fit_state)
    state_sha = sha256_file(paths["parent_fit_state"])
    report_body = {
        "experiment_id": parent_freeze["experiment_id"],
        "status": "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION",
        "source_commit": "synthetic-producer-commit",
        "prefit_freeze_sha256": parent_freeze_sha,
        "identity_preflight_sha256": parent_preflight_sha,
        "new_root_source_pass_sha256": parent_pass_sha,
        "final_slot": {"fit_id": "HINGE_FINAL",
                       "fit_input_path": str(bundle.resolve()),
                       "fit_input_sha256": sha256_file(bundle),
                       "fit_state_sha256": state_sha,
                       "coefficients": dict(HINGE)},
    }
    _write_signed_once(paths["parent_report"], report_body)
    report_sha = sha256_file(paths["parent_report"])
    write_once(paths["parent_verification"], {
        "status": "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY",
        "final_report_sha256": report_sha,
        "source_pass_sha256": parent_pass_sha,
    })
    verification_sha = sha256_file(paths["parent_verification"])
    write_once(paths["parent_receipt"], {
        "status": "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY",
        "final_report_sha256": report_sha,
        "verification_sha256": verification_sha,
        "fit_state_sha256": state_sha,
    })
    write_once(layout.blocked_r1, {
        "status": "BLOCKED_SOURCE_POPULATION_AMBIGUITY",
        "superseded_by": "R3 present-only population resolution; historical receipt retained",
    })
    # Preserve the earlier R2 receipt boundary as synthetic input evidence.
    write_once(paths["prior_r2_freeze"], {"synthetic_r2_artifact": "freeze"})
    write_once(paths["prior_r2_preflight"], {"synthetic_r2_artifact": "preflight"})
    write_once(paths["prior_r2_verification"], {"synthetic_r2_artifact": "verification"})
    marker = {"status": "SYNTHETIC_FIXTURE_ONLY", "experiment_id": EXPERIMENT_ID}
    _write_signed_once(layout.root / "SYNTHETIC_FIXTURE_R3.json", marker)


def _synthetic_root_pass(layout: Layout) -> None:
    freeze = _verify_freeze(layout)
    preflight = read_signed(layout.preflight)
    body = _root_pass_expected(layout, freeze, preflight)
    body["pass_kind"] = "SYNTHETIC_TEST_ONLY"
    _write_signed_once(layout.root_pass, body)


def synthetic_e2e() -> dict[str, Any]:
    global TARGET_MEMBER_READ_COUNT
    TARGET_MEMBER_READ_COUNT = 0
    with tempfile.TemporaryDirectory(prefix="reg-center-local-r2-") as temp:
        root = Path(temp).resolve()
        layout = make_layout(root, synthetic=True)
        _synthetic_source_files(root)
        _write_synthetic_parent_chain(layout)
        fixture = _json(root / "SYNTHETIC_FIXTURE_R3.json")
        if fixture.get("status") != "SYNTHETIC_FIXTURE_ONLY":
            raise GateError("Synthetic fixture marker missing")
        _cli(["--freeze", "--root", str(root), "--synthetic-fixture"])
        _cli(["--preflight", "--root", str(root), "--synthetic-fixture"])
        source_verification = _cli(["--verify-source", "--root", str(root), "--synthetic-fixture"])
        _synthetic_root_pass(layout)
        first = _cli(["--fit", "--root", str(root), "--synthetic-fixture"])
        reads_after_fit = TARGET_MEMBER_READ_COUNT
        resumed = _cli(["--fit", "--root", str(root), "--synthetic-fixture"])
        reads_after_resume = TARGET_MEMBER_READ_COUNT
        fit_verification = _cli(["--verify-fit", "--root", str(root), "--synthetic-fixture"])
        if reads_after_fit != 1 or reads_after_resume != reads_after_fit:
            raise GateError("Synthetic resume reloaded the target array")
        state = read_signed(layout.fit_state)
        if not math.isclose(float(state["delta"]), -0.6, rel_tol=0.0, abs_tol=1e-12):
            raise GateError("Synthetic closed-form scalar fit did not recover its known delta")
        original_receipt_text = layout.fit_receipt.read_text(encoding="utf-8")
        receipt_tamper = _json(layout.fit_receipt)
        receipt_tamper["delta"] = 123.0
        layout.fit_receipt.write_text(json.dumps(receipt_tamper), encoding="utf-8")
        receipt_rejected = False
        try:
            _cli(["--fit", "--root", str(root), "--synthetic-fixture"])
        except GateError:
            receipt_rejected = True
        if not receipt_rejected or TARGET_MEMBER_READ_COUNT != reads_after_fit:
            raise GateError("Synthetic receipt tamper was not rejected before target access")
        layout.fit_receipt.write_text(original_receipt_text, encoding="utf-8")
        original_state_text = layout.fit_state.read_text(encoding="utf-8")
        state_tamper = read_signed(layout.fit_state)
        state_tamper["delta"] = 123.0
        layout.fit_state.write_text(json.dumps(state_tamper), encoding="utf-8")
        state_rejected = False
        try:
            _cli(["--fit", "--root", str(root), "--synthetic-fixture"])
        except GateError:
            state_rejected = True
        if not state_rejected or TARGET_MEMBER_READ_COUNT != reads_after_fit:
            raise GateError("Synthetic state tamper was not rejected before target access")
        layout.fit_state.write_text(original_state_text, encoding="utf-8")
        return {
            "status": "PASS_SYNTHETIC_CLI_DURABLE_RESUME_TAMPER",
            "source_verification": source_verification["status"],
            "first_fit": first["status"],
            "resume": resumed["status"],
            "fit_verification": fit_verification["status"],
            "known_delta": state["delta"],
            "target_member_reads": TARGET_MEMBER_READ_COUNT,
            "resume_target_reload": False,
            "receipt_tamper_rejected_before_target_access": receipt_rejected,
            "state_tamper_rejected_before_target_access": state_rejected,
            "loaded_npz_members": source_verification["loaded_npz_members"],
            "real_data_accessed": False,
        }


def _layout_from_args(args: argparse.Namespace) -> Layout:
    root = Path(args.root).resolve() if args.root else ROOT
    synthetic = bool(args.synthetic_fixture)
    if synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if root == ROOT or (temp_root not in root.parents and root != temp_root):
            raise GateError("Synthetic fixture mode is restricted to a temporary directory")
        marker_path = root / "SYNTHETIC_FIXTURE_R3.json"
        if marker_path.is_file():
            marker = read_signed(marker_path)
            if marker.get("status") != "SYNTHETIC_FIXTURE_ONLY":
                raise GateError("Synthetic fixture marker is invalid")
    elif root != ROOT:
        raise GateError("Alternate roots require the isolated synthetic fixture flag")
    return make_layout(root, synthetic=synthetic)


def _cli(argv: Sequence[str]) -> dict[str, Any]:
    return main(argv, emit=False)


def main(argv: Sequence[str] | None = None, *, emit: bool = True) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--verify-source", action="store_true")
    action.add_argument("--fit", action="store_true")
    action.add_argument("--verify-fit", action="store_true")
    action.add_argument("--synthetic-qa", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.synthetic_qa:
            result = synthetic_e2e()
        else:
            layout = _layout_from_args(args)
            if args.freeze:
                result = {"status": create_freeze(layout)["status"],
                          "freeze_sha256": sha256_file(layout.freeze)}
            elif args.preflight:
                result = {"status": create_preflight(layout)["status"],
                          "preflight_sha256": sha256_file(layout.preflight)}
            elif args.verify_source:
                result = _verify_source_subprocess(layout)
            elif args.fit:
                result = fit_local(layout)
            else:
                result = verify_fit(layout)
    except GateError:
        raise
    if emit:
        print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        main()
    except GateError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
