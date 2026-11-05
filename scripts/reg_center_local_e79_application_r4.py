"""Source-gated deterministic E79 application of the approved FULLREG local delta.

Freeze/preflight/verification use pinned metadata only. The E79 prediction
archive is not opened and no CSV is created until a separate application pass.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
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
EXPERIMENT_ID = "REG_CENTER_LOCAL_E79_APPLICATION_V1_R4"
APP_NAMESPACE = "reg_center_local_application_v1"
FIT_EXPERIMENT_ID = "REG_CENTER_LOCAL_APPLICATION_V1_R3"
FIT_ID = "LOCAL_CENTER_FINAL"
EXPECTED_ROWS = 29_421
EXPECTED_MISSING = 1_062
EXPECTED_COLUMNS = ("record_id", "threat_rank")
REQUIRED_E79_MEMBERS = ("record_id", "prediction", "baseline", "missing")
FORBIDDEN_MEMBER_TOKENS = ("target", "label", "outcome", "threat_rank", "score")
HINGE = {
    "intercept": 1.7288079036674198,
    "slope_low": 0.9933285599956824,
    "slope_high": 0.957778072288735,
}
DELTA_BOUNDS = (-2.0, 0.0)
ROOT_PASS_KEYS = {
    "verdict", "experiment_id", "pass_kind", "application_freeze_sha256",
    "application_preflight_sha256", "source_inventory_sha256",
    "local_fit_state_sha256", "independent_fit_qa_sha256",
    "e79_archive_sha256", "output_csv_path", "authorized_actions",
    "base_model_fits", "calibrator_fits", "e79_prediction_access_authorized",
    "e79_target_values_authorized", "test_or_sample_access_authorized",
    "csv_creation_authorized", "kaggle_upload_authorized", "signature",
}


class ApplicationGateError(RuntimeError):
    """Fail-closed source, E79 identity, approval, or CSV integrity failure."""


@dataclass(frozen=True)
class Layout:
    root: Path
    synthetic: bool = False

    @property
    def experiment_dir(self) -> Path:
        return self.root / "experiments" / APP_NAMESPACE

    @property
    def report_dir(self) -> Path:
        return self.root / "reports" / "optimization_v3" / APP_NAMESPACE

    @property
    def freeze(self) -> Path:
        return self.experiment_dir / "E79_APPLICATION_FREEZE_R4.json"

    @property
    def preflight(self) -> Path:
        return self.report_dir / "E79_APPLICATION_PREFLIGHT_R4.json"

    @property
    def source_verification(self) -> Path:
        return self.report_dir / "E79_APPLICATION_SOURCE_VERIFICATION_R4.json"

    @property
    def root_pass(self) -> Path:
        return self.experiment_dir / "E79_APPLICATION_ROOT_PASS_R4.json"

    @property
    def output_csv(self) -> Path:
        return self.root / "submissions/reg_center_local_application_v1/fullreg_center_local_r4.csv"

    @property
    def application_report(self) -> Path:
        return self.report_dir / "E79_APPLICATION_REPORT_R4.json"

    @property
    def application_verification(self) -> Path:
        return self.report_dir / "E79_APPLICATION_VERIFICATION_R4.json"

    @property
    def application_receipt(self) -> Path:
        return self.report_dir / "E79_APPLICATION_RECEIPT_R4.json"

    @property
    def fit_paths(self) -> dict[str, Path]:
        return {
            "fit_freeze": self.root / "experiments/reg_center_local_application_v1/PREFIT_FREEZE_R3.json",
            "fit_preflight": self.root / "reports/optimization_v3/reg_center_local_application_v1/IDENTITY_PREFLIGHT_R3.json",
            "fit_root_pass": self.root / "experiments/reg_center_local_application_v1/ROOT_SOURCE_PASS_R3.json",
            "fit_state": self.root / "experiments/reg_center_local_application_v1/fits_R3/LOCAL_CENTER_FINAL.json",
            "fit_receipt": self.root / "experiments/reg_center_local_application_v1/fits_R3/LOCAL_CENTER_FINAL_RECEIPT.json",
            "fit_ledger": self.root / "experiments/reg_center_local_application_v1/calibrator_ledger_R3.jsonl",
            "fit_numerical_qa_original": self.root / "reports/optimization_v3/reg_center_local_application_v1/INDEPENDENT_FIT_NUMERICAL_QA_R3.json",
            "fit_numerical_qa": self.root / "reports/optimization_v3/reg_center_local_application_v1/INDEPENDENT_FIT_NUMERICAL_QA_R3_REPLAY1.json",
            "hinge_oof_bundle": self.root / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz",
        }

    @property
    def parent_paths(self) -> dict[str, Path]:
        return {
            "parent_freeze": self.root / "experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_FREEZE_R1.json",
            "parent_pass": self.root / "experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_ROOT_SOURCE_PASS_R1.json",
            "parent_preflight": self.root / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_PREFLIGHT_R1.json",
            "parent_report": self.root / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_REPORT_R1.json",
            "parent_verification": self.root / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_VERIFICATION_R1.json",
            "parent_receipt": self.root / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_RECEIPT_R1.json",
        }

    @property
    def source_paths(self) -> dict[str, Path]:
        return {
            "application_runner": self.root / "scripts/reg_center_local_e79_application_r4.py",
            "application_verifier": self.root / "scripts/verify_reg_center_local_e79_application_r4.py",
            "application_tests": self.root / "tests/test_reg_center_local_e79_application_r4.py",
            "application_policy": self.root / "reports/optimization_v3/reg_center_local_application_v1/E79_APPLICATION_POLICY_R4.md",
            "application_pass_contract": self.root / "reports/optimization_v3/reg_center_local_application_v1/E79_APPLICATION_PASS_CONTRACT_R4.md",
            "prior_r3_application_pass": self.root / "experiments/reg_center_local_application_v1/E79_APPLICATION_ROOT_PASS_R3.json",
            "prior_r3_failure_record": self.root / "reports/optimization_v3/reg_center_local_application_v1/E79_APPLICATION_FAILURE_R3.json",
            "fit_runner": self.root / "scripts/reg_center_local_application_v1.py",
            "fit_verifier": self.root / "scripts/verify_reg_center_local_application_v1.py",
            "fit_numerical_verifier_original": self.root / "scripts/verify_reg_center_local_application_fit_v1.py",
            "fit_numerical_verifier_replay": self.root / "scripts/verify_reg_center_local_application_fit_replay_v1.py",
            "final_hinge_runner": self.root / "scripts/reg_monotone_hinge_p50_final_application_v1.py",
            "csv_writer_source": self.root / "scripts/reg_monotone_hinge_p50_v4.py",
            "csv_verifier_source": self.root / "scripts/verify_reg_monotone_hinge_p50_v4.py",
        }

    @property
    def prior_application_paths(self) -> dict[str, Path]:
        return {
            "freeze_r1": self.experiment_dir / "E79_APPLICATION_FREEZE_R1.json",
            "preflight_r1": self.report_dir / "E79_APPLICATION_PREFLIGHT_R1.json",
            "freeze_r2": self.experiment_dir / "E79_APPLICATION_FREEZE_R2.json",
            "preflight_r2": self.report_dir / "E79_APPLICATION_PREFLIGHT_R2.json",
            "freeze_r3": self.experiment_dir / "E79_APPLICATION_FREEZE_R3.json",
            "preflight_r3": self.report_dir / "E79_APPLICATION_PREFLIGHT_R3.json",
            "root_pass_r3": self.experiment_dir / "E79_APPLICATION_ROOT_PASS_R3.json",
            "failure_record_r3": self.report_dir / "E79_APPLICATION_FAILURE_R3.json",
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


def ordered_ids_sha256(value: Any) -> str:
    """Match reg_final_wave_v2.sha_ordered_ids for the inherited E79 pin."""
    ids = np.asarray(value, dtype=str).tolist()
    payload = json.dumps(ids, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode(
                             "utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("signature", None)
    return body


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ApplicationGateError(f"Cannot read metadata JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ApplicationGateError(f"Expected JSON object: {path}")
    return result


def read_signed(path: Path) -> dict[str, Any]:
    result = read_json(path)
    signature = result.get("signature")
    if not isinstance(signature, str) or signature != canonical_sha(_unsigned(result)):
        raise ApplicationGateError(f"Canonical signature mismatch: {path}")
    return result


def write_once(path: Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ApplicationGateError(f"Refusing to overwrite immutable receipt: {target}") from exc
    return hashlib.sha256(payload).hexdigest()


def write_signed_once(path: Path, body: Mapping[str, Any]) -> str:
    signed = dict(body)
    signed["signature"] = canonical_sha(signed)
    return write_once(path, signed)


def _file_inventory(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    result = {}
    for name, raw in paths.items():
        path = Path(raw).resolve()
        if not path.is_file():
            raise ApplicationGateError(f"Pinned metadata/source file is missing: {name}")
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _expected_archive_pin(layout: Layout) -> dict[str, str]:
    parent = read_signed(layout.parent_paths["parent_freeze"])
    item = parent.get("input_hashes", {}).get("e79_archive")
    if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
        raise ApplicationGateError("Parent final freeze does not bind the E79 archive")
    return {"path": str(Path(item["path"]).resolve()), "sha256": str(item["sha256"])}


def _metadata_paths(layout: Layout) -> dict[str, Path]:
    parent_freeze = read_signed(layout.parent_paths["parent_freeze"])
    inputs = parent_freeze["input_hashes"]
    names = (
        "e79_manifest", "e79_qa", "e79_prelabel_receipt",
        "e79_trained_state_freeze", "direct_freeze", "direct_prelabel_receipt",
    )
    return {name: Path(inputs[name]["path"]).resolve() for name in names}


def _fit_and_parent_pins(layout: Layout) -> dict[str, Any]:
    fit_freeze = read_signed(layout.fit_paths["fit_freeze"])
    fit_preflight = read_signed(layout.fit_paths["fit_preflight"])
    fit_pass = read_signed(layout.fit_paths["fit_root_pass"])
    fit_state = read_signed(layout.fit_paths["fit_state"])
    fit_receipt = read_signed(layout.fit_paths["fit_receipt"])
    fit_qa = read_signed(layout.fit_paths["fit_numerical_qa"])
    parent_freeze = read_signed(layout.parent_paths["parent_freeze"])
    parent_pass = read_signed(layout.parent_paths["parent_pass"])
    parent_preflight = read_signed(layout.parent_paths["parent_preflight"])
    parent_report = read_signed(layout.parent_paths["parent_report"])
    parent_verification = read_json(layout.parent_paths["parent_verification"])
    parent_receipt = read_json(layout.parent_paths["parent_receipt"])

    pins = {
        **_file_inventory(layout.fit_paths),
        **_file_inventory(layout.parent_paths),
        **_file_inventory(_metadata_paths(layout)),
    }
    fit_pass_sha = pins["fit_root_pass"]["sha256"]
    fit_state_sha = pins["fit_state"]["sha256"]
    fit_receipt_sha = pins["fit_receipt"]["sha256"]
    fit_qa_sha = pins["fit_numerical_qa"]["sha256"]
    fit_freeze_sha = pins["fit_freeze"]["sha256"]
    fit_preflight_sha = pins["fit_preflight"]["sha256"]
    parent_freeze_sha = pins["parent_freeze"]["sha256"]
    parent_pass_sha = pins["parent_pass"]["sha256"]
    parent_preflight_sha = pins["parent_preflight"]["sha256"]
    parent_report_sha = pins["parent_report"]["sha256"]
    fit_delta = float(fit_state.get("delta", math.nan))

    if (fit_state.get("status") != "PASS_LOCAL_SCALAR_FIT_COMPLETE"
            or fit_state.get("experiment_id") != FIT_EXPERIMENT_ID
            or fit_state.get("fit_id") != FIT_ID
            or fit_state.get("root_source_pass_sha256") != fit_pass_sha
            or fit_state.get("preparation_freeze_sha256") != fit_freeze_sha
            or fit_state.get("identity_preflight_sha256") != fit_preflight_sha
            or fit_receipt.get("status") != "PASS_DURABLE_LOCAL_FIT_RECEIPT"
            or fit_receipt.get("fit_state_sha256") != fit_state_sha
            or fit_receipt.get("root_source_pass_sha256") != fit_pass_sha
            or fit_qa.get("status") != "PASS_INDEPENDENT_NUMERICAL_FIT_AUDIT"
            or fit_qa.get("fit_state_sha256") != fit_state_sha
            or fit_qa.get("root_source_pass_sha256") != fit_pass_sha
            or fit_qa.get("independent_numerical_replay", {}).get("delta_abs_difference") != 0.0
            or not DELTA_BOUNDS[0] <= fit_delta <= DELTA_BOUNDS[1]
            or parent_pass.get("verdict") != "PASS"
            or parent_pass.get("prefit_freeze_sha256") != parent_freeze_sha
            or parent_pass.get("identity_preflight_sha256") != parent_preflight_sha
            or parent_preflight.get("prefit_freeze_sha256") != parent_freeze_sha
            or parent_report.get("prefit_freeze_sha256") != parent_freeze_sha
            or parent_report.get("identity_preflight_sha256") != parent_preflight_sha
            or parent_report.get("new_root_source_pass_sha256") != parent_pass_sha
            or parent_report.get("status") != "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION"
            or parent_verification.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or parent_verification.get("final_report_sha256") != parent_report_sha
            or parent_receipt.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or parent_receipt.get("final_report_sha256") != parent_report_sha
            or parent_receipt.get("verification_sha256") != pins["parent_verification"]["sha256"]
            or parent_receipt.get("fit_state_sha256") != parent_report["final_slot"]["fit_state_sha256"]):
        raise ApplicationGateError("Fit or FULLREG final-hinge producer receipt chain failed")

    e79_pin = _expected_archive_pin(layout)
    e79 = parent_freeze.get("application_contract", {})
    e79_audit = parent_preflight.get("e79_raw_direct_audit", {})
    metadata = {}
    for key, path in _metadata_paths(layout).items():
        item = parent_freeze["input_hashes"][key]
        actual = pins[key]["sha256"]
        if actual != item["sha256"]:
            raise ApplicationGateError(f"Pinned E79 lineage metadata changed: {key}")
        try:
            metadata[key] = read_json(path)
        except ApplicationGateError:
            # Direct freeze metadata is hash-pinned by the parent; it is not needed
            # to derive the E79 contract and may use a non-JSON format in future.
            metadata[key] = {}
    manifest = metadata["e79_manifest"]
    qa = metadata["e79_qa"]
    prelabel = metadata["e79_prelabel_receipt"]
    trained = metadata["e79_trained_state_freeze"]
    if (str(e79_pin["sha256"]) != parent_freeze.get("application_contract", {}).get("raw_direct_archive_sha256")
            or int(e79.get("rows", -1)) != EXPECTED_ROWS
            or int(e79.get("missing_rows", -1)) != EXPECTED_MISSING
            or e79.get("raw_direct_archive_sha256") != e79_pin["sha256"]
            or e79.get("test_or_sample_read") is not False
            or manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or manifest.get("prediction_rows") != EXPECTED_ROWS
            or manifest.get("missing_rows") != EXPECTED_MISSING
            or manifest.get("test_labels_loaded") is not False
            or manifest.get("upload_performed") is not False
            or qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or qa.get("missing_rows") != EXPECTED_MISSING
            or qa.get("test_labels_read") is not False
            or prelabel.get("status") != "PRELABEL_PASS"
            or prelabel.get("test_labels_loaded") is not False
            or trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY"
            or trained.get("test_labels_loaded") is not False
            or e79_audit.get("archive_sha256") != e79_pin["sha256"]
            or e79_audit.get("rows") != EXPECTED_ROWS
            or e79_audit.get("unique_ids") != EXPECTED_ROWS
            or e79_audit.get("missing_rows") != EXPECTED_MISSING
            or e79_audit.get("missing_route_bit_exact_to_baseline") is not True):
        raise ApplicationGateError("Inherited E79 datahash, identity, or label boundary failed")

    return {
        "file_pins": pins,
        "fit": {
            "fit_state_sha256": fit_state_sha,
            "fit_receipt_sha256": fit_receipt_sha,
            "independent_fit_qa_sha256": fit_qa_sha,
            "fit_root_source_pass_sha256": fit_pass_sha,
            "fit_delta": fit_delta,
            "fit_preparation_freeze_sha256": fit_freeze_sha,
            "fit_identity_preflight_sha256": fit_preflight_sha,
        },
        "hinge_producer": {
            "final_report_sha256": parent_report_sha,
            "final_source_pass_sha256": parent_pass_sha,
            "final_application_preflight_sha256": parent_preflight_sha,
            "final_freeze_sha256": parent_freeze_sha,
        },
        "e79_deferred_input": {
            "path": e79_pin["path"],
            "expected_sha256_from_parent_freeze": e79_pin["sha256"],
            "live_archive_hash_rechecked": False,
            "archive_opened": False,
            "expected_rows": EXPECTED_ROWS,
            "expected_missing_rows": EXPECTED_MISSING,
            "ordered_ids_sha256_from_parent_preflight": e79_audit["ordered_ids_sha256"],
            "raw_direct_sha256_from_parent_preflight": e79_audit["raw_direct_sha256"],
            "required_members": list(REQUIRED_E79_MEMBERS),
            "expected_extra_non_target_members": e79_audit.get("unloaded_extra_member_names", []),
            "missing_route_source": "baseline member; prelabel audit established raw DIRECT bits equal baseline on missing rows",
        },
        "lineage_files_sha256": pins,
    }


def _source_paths(layout: Layout) -> dict[str, Path]:
    return layout.source_paths


def _inventory(layout: Layout) -> dict[str, Any]:
    files = {
        "sources": _file_inventory(_source_paths(layout)),
        "fit_and_producer_receipts": _fit_and_parent_pins(layout)["file_pins"],
    }
    # The E79 archive is inherited as an expected datahash. It is deliberately not
    # opened or rehashed until a separately signed application pass is accepted.
    deferred = _fit_and_parent_pins(layout)["e79_deferred_input"]
    return {"files": files, "deferred_e79_input": deferred}


def _git_head(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def create_prior_failure_record(layout: Layout) -> dict[str, Any]:
    """Record the root-reported R3 failure without touching E79 or outputs."""
    prior = layout.prior_application_paths
    old_freeze = read_signed(prior["freeze_r3"])
    old_preflight = read_signed(prior["preflight_r3"])
    old_pass = read_signed(prior["root_pass_r3"])
    pass_sha = sha256_file(prior["root_pass_r3"])
    expected_output = layout.root / "submissions/reg_center_local_application_v1/fullreg_center_local_r3.csv"
    expected_pass = {
        "verdict": "PASS",
        "experiment_id": "REG_CENTER_LOCAL_E79_APPLICATION_V1_R3",
        "pass_kind": "FINAL_E79_LOCAL_APPLICATION",
        "application_freeze_sha256": sha256_file(prior["freeze_r3"]),
        "application_preflight_sha256": sha256_file(prior["preflight_r3"]),
        "source_inventory_sha256": old_freeze["inventory_sha256"],
        "local_fit_state_sha256": old_freeze["lineage"]["fit"]["fit_state_sha256"],
        "independent_fit_qa_sha256": old_freeze["lineage"]["fit"]["independent_fit_qa_sha256"],
        "e79_archive_sha256": old_freeze["lineage"]["e79_deferred_input"]["expected_sha256_from_parent_freeze"],
        "output_csv_path": str(expected_output.resolve()),
        "authorized_actions": ["READ_E79_ALLOWLIST", "CREATE_LOCAL_CSV", "VERIFY_LOCAL_CSV"],
        "base_model_fits": 0,
        "calibrator_fits": 0,
        "e79_prediction_access_authorized": True,
        "e79_target_values_authorized": False,
        "test_or_sample_access_authorized": False,
        "csv_creation_authorized": True,
        "kaggle_upload_authorized": False,
    }
    if (set(old_pass) != ROOT_PASS_KEYS or _unsigned(old_pass) != expected_pass
            or old_preflight.get("status") != "PASS_E79_APPLICATION_SOURCE_PREFLIGHT_PENDING_ROOT_PASS"
            or old_pass.get("application_preflight_sha256") != sha256_file(prior["preflight_r3"])):
        raise ApplicationGateError("Preserved R3 application pass does not bind its frozen source")

    r3_output_paths = (
        expected_output,
        layout.report_dir / "E79_APPLICATION_REPORT_R1.json",
        layout.report_dir / "E79_APPLICATION_VERIFICATION_R1.json",
        layout.report_dir / "E79_APPLICATION_RECEIPT_R1.json",
    )
    if any(path.exists() for path in r3_output_paths):
        raise ApplicationGateError("R3 failed attempt has partial outputs; retain and reconcile before R4")

    body = {
        "schema_version": 1,
        "status": "ROOT_REPORTED_R3_APPLICATION_ATTEMPT_FAILED_BEFORE_NPZ_OPEN",
        "experiment_id": "REG_CENTER_LOCAL_E79_APPLICATION_V1_R3",
        "root_application_pass_sha256": pass_sha,
        "application_freeze_sha256": sha256_file(prior["freeze_r3"]),
        "application_preflight_sha256": sha256_file(prior["preflight_r3"]),
        "output_csv_path": str(expected_output.resolve()),
        "fit_state_sha256": old_freeze["lineage"]["fit"]["fit_state_sha256"],
        "failure": {
            "exception_type": "KeyError",
            "missing_mapping_key": "sha256",
            "runner_location": "scripts/reg_center_local_e79_application_v1.py:_load_e79_after_pass",
            "failure_phase": "after archive file existence check and byte hashing; before expected-hash comparison and NPZ open",
            "archive_bytes_hashed": True,
            "archive_hash_comparison_completed": False,
            "npz_container_opened": False,
            "prediction_arrays_loaded": False,
            "csv_created": False,
            "application_report_created": False,
            "independent_csv_verification_created": False,
            "application_receipt_created": False,
            "kaggle_upload": False,
            "evidence_source": "root-reported runtime result; not replayed on real E79",
        },
    }
    path = prior["failure_record_r3"]
    if path.exists():
        existing = read_signed(path)
        if _unsigned(existing) != body:
            raise ApplicationGateError("Existing R3 failure record conflicts with root-reported attempt")
        return existing
    write_signed_once(path, body)
    return read_signed(path)


def _supersedes(layout: Layout) -> dict[str, Any]:
    prior = layout.prior_application_paths
    if any(not path.is_file() for path in prior.values()):
        raise ApplicationGateError("Required R1-R3 application history is incomplete")
    record = read_signed(prior["failure_record_r3"])
    r3_pass_sha = sha256_file(prior["root_pass_r3"])
    if (record.get("status") != "ROOT_REPORTED_R3_APPLICATION_ATTEMPT_FAILED_BEFORE_NPZ_OPEN"
            or record.get("root_application_pass_sha256") != r3_pass_sha
            or record.get("failure", {}).get("npz_container_opened") is not False
            or record.get("failure", {}).get("csv_created") is not False):
        raise ApplicationGateError("R3 failed-attempt record does not bind the preserved root pass")
    names = {
        "freeze_r1": "freeze_r1_sha256",
        "preflight_r1": "preflight_r1_sha256",
        "freeze_r2": "freeze_r2_sha256",
        "preflight_r2": "preflight_r2_sha256",
        "freeze_r3": "freeze_r3_sha256",
        "preflight_r3": "preflight_r3_sha256",
        "root_pass_r3": "root_pass_r3_sha256",
        "failure_record_r3": "failure_record_r3_sha256",
    }
    return {
        **{output: sha256_file(prior[name]) for name, output in names.items()},
        "reason": "R3 consumed one application pass but failed at the expected E79 pin key before NPZ open; R4 fixes the key and adds end-to-end synthetic pass-to-receipt coverage.",
    }


def create_freeze(layout: Layout) -> dict[str, Any]:
    lineage = _fit_and_parent_pins(layout)
    inventory = _inventory(layout)
    prior_application = _supersedes(layout)
    body = {
        "schema_version": 1,
        "status": "E79_APPLICATION_SOURCE_FROZEN_PENDING_ROOT_PASS",
        "experiment_id": EXPERIMENT_ID,
        "supersedes": prior_application,
        "prepared_on_head_informational_only": _git_head(layout.root),
        "inventory": inventory,
        "inventory_sha256": canonical_sha(inventory),
        "lineage": lineage,
        "application_contract": {
            "candidate_id": "FULLREG_HINGE_PLUS_CENTER_LOCAL_R4_E79",
            "formula": "present: clip(g(raw_DIRECT) + delta*b(raw_DIRECT), 0, 100); missing: raw DIRECT float64 bits unchanged",
            "fixed_hinge_coefficients": dict(HINGE),
            "triangle_center": 50.0,
            "triangle_half_width": 10.0,
            "delta": lineage["fit"]["fit_delta"],
            "delta_bounds": list(DELTA_BOUNDS),
            "output_path": str(layout.output_csv.resolve()),
            "output_columns": list(EXPECTED_COLUMNS),
            "rows": EXPECTED_ROWS,
            "missing_rows": EXPECTED_MISSING,
            "kaggle_upload": False,
        },
        "access_boundary": {
            "e79_archive_opened": False,
            "e79_archive_bytes_rehashed": False,
            "test_or_sample_read": False,
            "csv_created": False,
            "kaggle_accessed": False,
        },
    }
    write_signed_once(layout.freeze, body)
    return read_json(layout.freeze)


def _verify_freeze(layout: Layout, *, allow_output: bool = False) -> dict[str, Any]:
    freeze = read_signed(layout.freeze)
    inventory = _inventory(layout)
    boundary = freeze.get("access_boundary", {})
    if (freeze.get("experiment_id") != EXPERIMENT_ID
            or freeze.get("status") != "E79_APPLICATION_SOURCE_FROZEN_PENDING_ROOT_PASS"
            or freeze.get("supersedes") != _supersedes(layout)
            or freeze.get("inventory") != inventory
            or freeze.get("inventory_sha256") != canonical_sha(inventory)
            or freeze.get("lineage") != _fit_and_parent_pins(layout)
            or any(boundary.get(key) is not False for key in (
                "e79_archive_opened", "e79_archive_bytes_rehashed", "test_or_sample_read",
                "csv_created", "kaggle_accessed"))):
        raise ApplicationGateError("Application freeze or metadata pins changed")
    if not allow_output and layout.output_csv.exists():
        raise ApplicationGateError("Application output exists before its root pass")
    return freeze


def create_preflight(layout: Layout) -> dict[str, Any]:
    freeze = _verify_freeze(layout)
    deferred = freeze["lineage"]["e79_deferred_input"]
    body = {
        "schema_version": 1,
        "status": "PASS_E79_APPLICATION_SOURCE_PREFLIGHT_PENDING_ROOT_PASS",
        "experiment_id": EXPERIMENT_ID,
        "application_freeze_sha256": sha256_file(layout.freeze),
        "inventory_sha256": freeze["inventory_sha256"],
        "source_datahash": {
            "e79_archive_path": deferred["path"],
            "e79_expected_sha256_inherited": deferred["expected_sha256_from_parent_freeze"],
            "live_archive_hash_rechecked": False,
            "e79_archive_opened": False,
        },
        "e79_contract_from_signed_parent_receipts": {
            "rows": EXPECTED_ROWS,
            "unique_ids": EXPECTED_ROWS,
            "missing_rows": EXPECTED_MISSING,
            "ordered_ids_sha256": deferred["ordered_ids_sha256_from_parent_preflight"],
            "raw_direct_sha256": deferred["raw_direct_sha256_from_parent_preflight"],
            "required_non_target_members": list(REQUIRED_E79_MEMBERS),
            "extra_non_target_member_names_not_loaded": deferred["expected_extra_non_target_members"],
            "output_columns": list(EXPECTED_COLUMNS),
        },
        "fit": {
            "fit_state_sha256": freeze["lineage"]["fit"]["fit_state_sha256"],
            "fit_receipt_sha256": freeze["lineage"]["fit"]["fit_receipt_sha256"],
            "independent_fit_qa_sha256": freeze["lineage"]["fit"]["independent_fit_qa_sha256"],
            "delta": freeze["lineage"]["fit"]["fit_delta"],
        },
        "actions": {
            "e79_archive_opened": False,
            "test_or_sample_read": False,
            "csv_created": False,
            "kaggle_upload": False,
            "base_model_fits": 0,
            "calibrator_fits": 0,
        },
    }
    write_signed_once(layout.preflight, body)
    return read_json(layout.preflight)


def _application_pass_expected(layout: Layout,
                               freeze: Mapping[str, Any],
                               preflight: Mapping[str, Any]) -> dict[str, Any]:
    lineage = freeze["lineage"]
    return {
        "verdict": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "pass_kind": "FINAL_E79_LOCAL_APPLICATION",
        "application_freeze_sha256": sha256_file(layout.freeze),
        "application_preflight_sha256": sha256_file(layout.preflight),
        "source_inventory_sha256": freeze["inventory_sha256"],
        "local_fit_state_sha256": lineage["fit"]["fit_state_sha256"],
        "independent_fit_qa_sha256": lineage["fit"]["independent_fit_qa_sha256"],
        "e79_archive_sha256": lineage["e79_deferred_input"]["expected_sha256_from_parent_freeze"],
        "output_csv_path": str(layout.output_csv.resolve()),
        "authorized_actions": ["READ_E79_ALLOWLIST", "CREATE_LOCAL_CSV", "VERIFY_LOCAL_CSV"],
        "base_model_fits": 0,
        "calibrator_fits": 0,
        "e79_prediction_access_authorized": True,
        "e79_target_values_authorized": False,
        "test_or_sample_access_authorized": False,
        "csv_creation_authorized": True,
        "kaggle_upload_authorized": False,
    }


def validate_application_pass(layout: Layout) -> tuple[dict[str, Any], str]:
    freeze = _verify_freeze(layout, allow_output=True)
    preflight = read_signed(layout.preflight)
    approval = read_signed(layout.root_pass)
    if set(approval) != ROOT_PASS_KEYS:
        raise ApplicationGateError("Application root pass has missing/extra keys")
    if _unsigned(approval) != _application_pass_expected(layout, freeze, preflight):
        raise ApplicationGateError("Application pass does not authorize exactly the frozen E79 route")
    if (preflight.get("status") != "PASS_E79_APPLICATION_SOURCE_PREFLIGHT_PENDING_ROOT_PASS"
            or preflight.get("source_datahash", {}).get("live_archive_hash_rechecked") is not False
            or preflight.get("actions", {}).get("e79_archive_opened") is not False
            or preflight.get("actions", {}).get("csv_created") is not False
            or preflight.get("actions", {}).get("test_or_sample_read") is not False):
        raise ApplicationGateError("Application source preflight is not the expected target-blind receipt")
    return approval, sha256_file(layout.root_pass)


def triangle_basis(raw_direct: Any) -> np.ndarray:
    direct = np.asarray(raw_direct, dtype=np.float64)
    if direct.ndim != 1 or not np.isfinite(direct).all():
        raise ApplicationGateError("Raw DIRECT predictions must be finite 1D float64")
    return np.maximum(0.0, 1.0 - np.abs(direct - 50.0) / 10.0)


def apply_values(raw_direct: Any, missing_mask: Any, delta: float) -> np.ndarray:
    direct = np.asarray(raw_direct, dtype="<f8")
    missing = np.asarray(missing_mask)
    if (direct.ndim != 1 or missing.ndim != 1 or direct.shape != missing.shape
            or missing.dtype.kind != "b" or not np.isfinite(direct).all()
            or np.any((direct < 0.0) | (direct > 100.0))
            or not math.isfinite(float(delta))
            or not DELTA_BOUNDS[0] <= float(delta) <= DELTA_BOUNDS[1]):
        raise ApplicationGateError("Raw DIRECT/missing/delta application contract failed")
    result = direct.copy()
    present = ~missing
    p = direct[present]
    base = (HINGE["intercept"] + HINGE["slope_low"] * np.minimum(p, 50.0)
            + HINGE["slope_high"] * np.maximum(p - 50.0, 0.0))
    result[present] = np.clip(base + float(delta) * triangle_basis(p), 0.0, 100.0)
    if not np.array_equal(result[missing].view("<u8"), direct[missing].view("<u8")):
        raise ApplicationGateError("Missing E79 predictions are not raw DIRECT bit-exact")
    return result


def render_csv(record_ids: Any, values: Any) -> bytes:
    ids = np.asarray(record_ids, dtype=str)
    preds = np.asarray(values, dtype="<f8")
    if (ids.ndim != 1 or preds.ndim != 1 or ids.shape != preds.shape or len(ids) == 0
            or len(np.unique(ids)) != len(ids) or np.any(ids == "")
            or not np.isfinite(preds).all() or np.any((preds < 0.0) | (preds > 100.0))):
        raise ApplicationGateError("CSV rows require aligned unique IDs and finite values in [0,100]")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EXPECTED_COLUMNS)
    for record_id, value in zip(ids.tolist(), preds.tolist(), strict=True):
        writer.writerow((record_id, format(float(value), ".17g")))
    return buffer.getvalue().encode("utf-8")


def verify_csv_bytes(payload: bytes, record_ids: Any, expected_values: Any,
                     missing_mask: Any, raw_direct: Any, *,
                     expected_rows: int | None = None,
                     expected_missing: int | None = None) -> dict[str, Any]:
    ids = np.asarray(record_ids, dtype=str)
    expected = np.asarray(expected_values, dtype="<f8")
    missing = np.asarray(missing_mask, dtype=bool)
    raw = np.asarray(raw_direct, dtype="<f8")
    if not (ids.shape == expected.shape == missing.shape == raw.shape):
        raise ApplicationGateError("CSV verifier inputs are misaligned")
    try:
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ApplicationGateError("CSV is not valid UTF-8/CSV") from exc
    if not rows or tuple(rows[0]) != EXPECTED_COLUMNS:
        raise ApplicationGateError("CSV header differs from record_id,threat_rank")
    if len(rows) != len(ids) + 1:
        raise ApplicationGateError("CSV row count differs from frozen ID vector")
    actual_ids = [row[0] for row in rows[1:] if len(row) == 2]
    if len(actual_ids) != len(ids) or actual_ids != ids.tolist() or len(set(actual_ids)) != len(actual_ids):
        raise ApplicationGateError("CSV IDs are duplicated, reordered, or changed")
    try:
        actual_values = np.asarray([float(row[1]) for row in rows[1:]], dtype="<f8")
    except (ValueError, IndexError) as exc:
        raise ApplicationGateError("CSV threat_rank field is malformed") from exc
    if (not np.isfinite(actual_values).all() or np.any((actual_values < 0.0) | (actual_values > 100.0))
            or not np.array_equal(actual_values.view("<u8"), expected.view("<u8"))
            or not np.array_equal(actual_values[missing].view("<u8"), raw[missing].view("<u8"))):
        raise ApplicationGateError("CSV values/range/missing fallback differ from frozen application")
    if expected_rows is not None and len(ids) != expected_rows:
        raise ApplicationGateError("CSV row count differs from E79 production contract")
    if expected_missing is not None and int(missing.sum()) != expected_missing:
        raise ApplicationGateError("CSV missing count differs from E79 production contract")
    return {
        "rows": len(ids),
        "columns": list(EXPECTED_COLUMNS),
        "ordered_ids_sha256": ordered_ids_sha256(ids),
        "missing_rows": int(missing.sum()),
        "missing_route_bit_exact": True,
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_csv_idempotent(path: Path, payload: bytes) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ApplicationGateError("Existing application CSV differs; refusing overwrite")
        return True
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ApplicationGateError("Application CSV appeared concurrently") from exc
    return False


def _load_e79_after_pass(layout: Layout, expected_pin: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(expected_pin["path"])
    if (not path.is_file()
            or sha256_file(path) != expected_pin["expected_sha256_from_parent_freeze"]):
        raise ApplicationGateError("Live E79 archive bytes differ from inherited frozen datahash")
    try:
        with zipfile.ZipFile(path) as zip_archive:
            names = {Path(name).stem for name in zip_archive.namelist() if name.endswith(".npy")}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ApplicationGateError("Cannot inspect approved E79 NPZ") from exc
    if not set(REQUIRED_E79_MEMBERS).issubset(names):
        raise ApplicationGateError("E79 archive lacks required prediction/identity arrays")
    forbidden = sorted(name for name in names - set(REQUIRED_E79_MEMBERS)
                       if any(token in name.casefold() for token in FORBIDDEN_MEMBER_TOKENS))
    if forbidden:
        raise ApplicationGateError(f"E79 archive contains target-like member names: {forbidden}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["record_id"], dtype=str).copy()
            raw = np.asarray(archive["prediction"], dtype="<f8").copy()
            baseline = np.asarray(archive["baseline"], dtype="<f8").copy()
            missing = np.asarray(archive["missing"], dtype=bool).copy()
    except (OSError, ValueError, KeyError) as exc:
        raise ApplicationGateError("Approved E79 prediction-only member read failed") from exc
    if (ids.ndim != 1 or raw.ndim != 1 or baseline.ndim != 1 or missing.ndim != 1
            or not (len(ids) == len(raw) == len(baseline) == len(missing) == EXPECTED_ROWS)
            or len(np.unique(ids)) != EXPECTED_ROWS
            or not np.isfinite(raw).all() or not np.isfinite(baseline).all()
            or np.any((raw < 0.0) | (raw > 100.0))
            or int(missing.sum()) != EXPECTED_MISSING
            or not np.array_equal(raw[missing].view("<u8"), baseline[missing].view("<u8"))):
        raise ApplicationGateError("Approved E79 identity/prediction/missing contract failed")
    return ids, raw, missing


def apply_once(layout: Layout) -> dict[str, Any]:
    approval, pass_sha = validate_application_pass(layout)
    freeze = read_signed(layout.freeze)
    lineage = freeze["lineage"]
    if layout.application_receipt.exists():
        return verify_application_outputs(layout, pass_sha)
    if any(path.exists() for path in (layout.output_csv, layout.application_report,
                                      layout.application_verification)):
        raise ApplicationGateError("Partial application outputs exist without a complete receipt")
    expected_pin = lineage["e79_deferred_input"]
    ids, raw, missing = _load_e79_after_pass(layout, expected_pin)
    if (ordered_ids_sha256(ids)
            != expected_pin["ordered_ids_sha256_from_parent_preflight"]):
        raise ApplicationGateError("Live ordered E79 IDs differ from signed parent preflight")
    fit_state = read_signed(layout.fit_paths["fit_state"])
    delta = float(fit_state["delta"])
    values = apply_values(raw, missing, delta)
    payload = render_csv(ids, values)
    csv_audit = verify_csv_bytes(payload, ids, values, missing, raw,
                                 expected_rows=EXPECTED_ROWS,
                                 expected_missing=EXPECTED_MISSING)
    resumed = write_csv_idempotent(layout.output_csv, payload)
    report_body = {
        "schema_version": 1,
        "status": "PASS_LOCAL_E79_APPLICATION_PENDING_INDEPENDENT_VERIFICATION",
        "experiment_id": EXPERIMENT_ID,
        "root_application_pass_sha256": pass_sha,
        "application_freeze_sha256": sha256_file(layout.freeze),
        "application_preflight_sha256": sha256_file(layout.preflight),
        "fit_state_sha256": lineage["fit"]["fit_state_sha256"],
        "fit_receipt_sha256": lineage["fit"]["fit_receipt_sha256"],
        "independent_fit_qa_sha256": lineage["fit"]["independent_fit_qa_sha256"],
        "e79_archive_sha256": expected_pin["expected_sha256_from_parent_freeze"],
        "e79_loaded_members": list(REQUIRED_E79_MEMBERS),
        "unloaded_non_target_members": expected_pin["expected_extra_non_target_members"],
        "rows": EXPECTED_ROWS,
        "missing_rows": EXPECTED_MISSING,
        "ordered_ids_sha256": csv_audit["ordered_ids_sha256"],
        "delta": delta,
        "fixed_hinge_coefficients": dict(HINGE),
        "csv_path": str(layout.output_csv.resolve()),
        "csv_sha256": csv_audit["csv_sha256"],
        "csv_columns": list(EXPECTED_COLUMNS),
        "missing_route_bit_exact": True,
        "resumed_identical_output": resumed,
        "base_model_fits": 0,
        "calibrator_fits": 0,
        "test_or_sample_read": False,
        "e79_target_values_loaded": False,
        "kaggle_upload": False,
    }
    write_signed_once(layout.application_report, report_body)
    verifier_path = layout.root / "scripts/verify_reg_center_local_e79_application_r4.py"
    completed = subprocess.run([sys.executable, str(verifier_path), "--verify-application",
                                "--root", str(layout.root)], cwd=layout.root,
                               capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ApplicationGateError(f"Independent E79 CSV verifier failed: {completed.stderr.strip()}")
    independent = json.loads(completed.stdout)
    write_once(layout.application_verification, independent)
    receipt = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_LOCAL_E79_APPLICATION",
        "experiment_id": EXPERIMENT_ID,
        "root_application_pass_sha256": pass_sha,
        "application_report_sha256": sha256_file(layout.application_report),
        "application_verification_sha256": sha256_file(layout.application_verification),
        "csv_sha256": sha256_file(layout.output_csv),
        "kaggle_upload": False,
    }
    write_once(layout.application_receipt, receipt)
    return {"status": receipt["status"], "csv_sha256": receipt["csv_sha256"],
            "verification_status": independent["status"], "resumed": resumed}


def verify_application_outputs(layout: Layout, pass_sha: str) -> dict[str, Any]:
    report = read_signed(layout.application_report)
    verification = read_json(layout.application_verification)
    receipt = read_json(layout.application_receipt)
    if (report.get("status") != "PASS_LOCAL_E79_APPLICATION_PENDING_INDEPENDENT_VERIFICATION"
            or report.get("root_application_pass_sha256") != pass_sha
            or report.get("csv_sha256") != sha256_file(layout.output_csv)
            or verification.get("status") != "PASS_INDEPENDENT_LOCAL_E79_APPLICATION"
            or verification.get("application_report_sha256") != sha256_file(layout.application_report)
            or verification.get("csv_sha256") != sha256_file(layout.output_csv)
            or receipt.get("status") != "PASS_INDEPENDENT_LOCAL_E79_APPLICATION"
            or receipt.get("application_report_sha256") != sha256_file(layout.application_report)
            or receipt.get("application_verification_sha256") != sha256_file(layout.application_verification)
            or receipt.get("csv_sha256") != sha256_file(layout.output_csv)):
        raise ApplicationGateError("Existing E79 application output/receipts do not bind")
    return {"status": receipt["status"], "csv_sha256": receipt["csv_sha256"],
            "verification_status": verification["status"], "resumed": True}


def _copy_synthetic_source_tree(layout: Layout) -> None:
    real_root = Path(__file__).resolve().parents[1]
    source_targets = set(layout.source_paths.values()) | set(layout.prior_application_paths.values())
    for target in source_targets:
        relative = Path(target).relative_to(layout.root)
        source = real_root / relative
        if not source.is_file():
            raise ApplicationGateError(f"Synthetic source fixture is missing: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _write_plain_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _build_synthetic_application_root(layout: Layout) -> tuple[Path, np.ndarray, np.ndarray]:
    """Build only temporary fake producer receipts and a synthetic E79-like NPZ."""
    _copy_synthetic_source_tree(layout)

    n = EXPECTED_ROWS
    ids = np.asarray([f"synthetic-e79-{index:05d}" for index in range(n)])
    raw = np.linspace(0.0, 100.0, n, dtype="<f8")
    baseline = raw.copy()
    missing = np.zeros(n, dtype=bool)
    missing[:EXPECTED_MISSING] = True
    archive_path = layout.root / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/inference_B.npz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as stream:
        np.savez(stream, record_id=ids, prediction=raw, baseline=baseline,
                 missing=missing, cycle_code=np.arange(n, dtype="<i4"), raw=raw)
    archive_sha = sha256_file(archive_path)
    raw_sha = hashlib.sha256(np.ascontiguousarray(raw, dtype="<f8").tobytes()).hexdigest()

    metadata_relpaths = {
        "e79_manifest": Path("reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/INFERENCE_MANIFEST_B.json"),
        "e79_qa": Path("reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/CSV_QA_B.json"),
        "e79_prelabel_receipt": Path("reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PRELABEL_PASS_B.json"),
        "e79_trained_state_freeze": Path("reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/TRAINED_STATE_FREEZE_B.json"),
        "direct_freeze": Path("experiments/runs/reg_position_direct_l1_v1/FREEZE_V4.json"),
        "direct_prelabel_receipt": Path("reports/optimization_v3/reg_position_direct_l1_v1/PRELABEL_VERIFICATION.json"),
    }
    metadata_bodies = {
        "e79_manifest": {"candidate_id": "DIRECT_POSITION_REAL_E79", "prediction_rows": n,
                         "missing_rows": EXPECTED_MISSING, "test_labels_loaded": False,
                         "upload_performed": False},
        "e79_qa": {"status": "PASS_INDEPENDENT_CSV_QA", "missing_rows": EXPECTED_MISSING,
                   "test_labels_read": False},
        "e79_prelabel_receipt": {"status": "PRELABEL_PASS", "test_labels_loaded": False},
        "e79_trained_state_freeze": {"status": "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY",
                                     "test_labels_loaded": False},
        "direct_freeze": {"status": "SYNTHETIC_DIRECT_FREEZE"},
        "direct_prelabel_receipt": {"status": "SYNTHETIC_DIRECT_PRELABEL"},
    }
    metadata_hashes: dict[str, dict[str, str]] = {}
    for key, relative in metadata_relpaths.items():
        path = layout.root / relative
        _write_plain_json_once(path, metadata_bodies[key])
        metadata_hashes[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}

    fit_freeze = layout.fit_paths["fit_freeze"]
    fit_preflight = layout.fit_paths["fit_preflight"]
    fit_pass = layout.fit_paths["fit_root_pass"]
    fit_state = layout.fit_paths["fit_state"]
    fit_receipt = layout.fit_paths["fit_receipt"]
    fit_qa = layout.fit_paths["fit_numerical_qa"]
    for path, body in (
        (fit_freeze, {"status": "SYNTHETIC_FIT_FREEZE"}),
        (fit_preflight, {"status": "SYNTHETIC_FIT_PREFLIGHT"}),
        (fit_pass, {"verdict": "PASS", "pass_kind": "ONE_LOCAL_SCALAR_FIT"}),
    ):
        write_signed_once(path, body)
    fit_state_body = {
        "status": "PASS_LOCAL_SCALAR_FIT_COMPLETE",
        "experiment_id": FIT_EXPERIMENT_ID,
        "fit_id": FIT_ID,
        "delta": -1.25,
        "root_source_pass_sha256": sha256_file(fit_pass),
        "preparation_freeze_sha256": sha256_file(fit_freeze),
        "identity_preflight_sha256": sha256_file(fit_preflight),
    }
    write_signed_once(fit_state, fit_state_body)
    write_signed_once(fit_receipt, {
        "status": "PASS_DURABLE_LOCAL_FIT_RECEIPT",
        "fit_state_sha256": sha256_file(fit_state),
        "root_source_pass_sha256": sha256_file(fit_pass),
    })
    write_signed_once(fit_qa, {
        "status": "PASS_INDEPENDENT_NUMERICAL_FIT_AUDIT",
        "fit_state_sha256": sha256_file(fit_state),
        "root_source_pass_sha256": sha256_file(fit_pass),
        "independent_numerical_replay": {"delta_abs_difference": 0.0},
    })
    _write_plain_json_once(layout.fit_paths["fit_numerical_qa_original"],
                           {"status": "PRESERVED_SYNTHETIC_ORIGINAL_QA"})
    ledger_payload = (
        json.dumps({"event": "FIT_RESERVED", "fit_id": FIT_ID}, sort_keys=True) + "\n"
        + json.dumps({"event": "FIT_COMPLETE", "fit_id": FIT_ID}, sort_keys=True) + "\n"
    ).encode("utf-8")
    layout.fit_paths["fit_ledger"].parent.mkdir(parents=True, exist_ok=True)
    with layout.fit_paths["fit_ledger"].open("xb") as stream:
        stream.write(ledger_payload)
    bundle_path = layout.fit_paths["hinge_oof_bundle"]
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(b"synthetic lineage placeholder; no arrays opened")

    parent_freeze = layout.parent_paths["parent_freeze"]
    parent_preflight = layout.parent_paths["parent_preflight"]
    parent_pass = layout.parent_paths["parent_pass"]
    parent_report = layout.parent_paths["parent_report"]
    parent_verification = layout.parent_paths["parent_verification"]
    parent_receipt = layout.parent_paths["parent_receipt"]
    input_hashes = {
        **metadata_hashes,
        "e79_archive": {"path": str(archive_path.resolve()), "sha256": archive_sha},
    }
    write_signed_once(parent_freeze, {
        "input_hashes": input_hashes,
        "application_contract": {
            "raw_direct_archive_sha256": archive_sha,
            "rows": n,
            "missing_rows": EXPECTED_MISSING,
            "test_or_sample_read": False,
        },
    })
    parent_freeze_sha = sha256_file(parent_freeze)
    parent_audit = {
        "archive_sha256": archive_sha,
        "rows": n,
        "unique_ids": n,
        "missing_rows": EXPECTED_MISSING,
        "missing_route_bit_exact_to_baseline": True,
        "ordered_ids_sha256": ordered_ids_sha256(ids),
        "raw_direct_sha256": raw_sha,
        "unloaded_extra_member_names": ["cycle_code", "raw"],
    }
    write_signed_once(parent_preflight, {
        "prefit_freeze_sha256": parent_freeze_sha,
        "e79_raw_direct_audit": parent_audit,
    })
    parent_preflight_sha = sha256_file(parent_preflight)
    write_signed_once(parent_pass, {
        "verdict": "PASS",
        "prefit_freeze_sha256": parent_freeze_sha,
        "identity_preflight_sha256": parent_preflight_sha,
    })
    parent_pass_sha = sha256_file(parent_pass)
    parent_fit_state_sha = "synthetic-parent-fit-state-sha256"
    write_signed_once(parent_report, {
        "status": "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION",
        "prefit_freeze_sha256": parent_freeze_sha,
        "identity_preflight_sha256": parent_preflight_sha,
        "new_root_source_pass_sha256": parent_pass_sha,
        "final_slot": {"fit_state_sha256": parent_fit_state_sha},
    })
    parent_report_sha = sha256_file(parent_report)
    _write_plain_json_once(parent_verification, {
        "status": "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY",
        "final_report_sha256": parent_report_sha,
    })
    _write_plain_json_once(parent_receipt, {
        "status": "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY",
        "final_report_sha256": parent_report_sha,
        "verification_sha256": sha256_file(parent_verification),
        "fit_state_sha256": parent_fit_state_sha,
    })
    return archive_path, ids, missing


def _synthetic_full_application() -> dict[str, Any]:
    real_prior_pass = ROOT / "experiments/reg_center_local_application_v1/E79_APPLICATION_ROOT_PASS_R3.json"
    real_prior_pass_sha = sha256_file(real_prior_pass)
    with tempfile.TemporaryDirectory(prefix="center-local-e79-full-application-") as temp:
        layout = make_layout(Path(temp), synthetic=True)
        archive_path, ids, missing = _build_synthetic_application_root(layout)
        freeze = create_freeze(layout)
        preflight = create_preflight(layout)
        source_verification = _verify_application_source_subprocess(layout)
        if source_verification.get("status") != "PASS_INDEPENDENT_E79_APPLICATION_SOURCE_VERIFICATION":
            raise ApplicationGateError("Synthetic full application source verification failed")
        write_signed_once(layout.root_pass, _application_pass_expected(layout, freeze, preflight))
        first = apply_once(layout)
        second = apply_once(layout)
        report = read_signed(layout.application_report)
        receipt = read_json(layout.application_receipt)
        if (first.get("status") != "PASS_INDEPENDENT_LOCAL_E79_APPLICATION"
                or first.get("resumed") is not False
                or second.get("status") != first.get("status")
                or second.get("resumed") is not True
                or second.get("csv_sha256") != first.get("csv_sha256")
                or receipt.get("csv_sha256") != first.get("csv_sha256")
                or report.get("rows") != EXPECTED_ROWS
                or report.get("missing_rows") != EXPECTED_MISSING
                or report.get("e79_loaded_members") != list(REQUIRED_E79_MEMBERS)
                or report.get("unloaded_non_target_members") != ["cycle_code", "raw"]
                or not layout.output_csv.is_file()
                or not layout.application_verification.is_file()
                or not layout.application_receipt.is_file()
                or Path(temp).resolve() not in Path(archive_path).resolve().parents
                or sha256_file(real_prior_pass) != real_prior_pass_sha):
            raise ApplicationGateError("Synthetic full application lifecycle/readback failed")
        return {
            "status": "PASS_SYNTHETIC_E79_FULL_APPLICATION_LIFECYCLE",
            "source_verification_status": source_verification["status"],
            "first_application_status": first["status"],
            "first_run_resumed": first["resumed"],
            "second_run_resumed": second["resumed"],
            "csv_sha256": first["csv_sha256"],
            "rows": len(ids),
            "missing_rows": int(missing.sum()),
            "archive_hash_checked": True,
            "allowlisted_members_loaded": list(REQUIRED_E79_MEMBERS),
            "harmless_extra_members_unloaded": ["cycle_code", "raw"],
            "independent_output_verification": first["verification_status"],
            "production_csv_created": False,
            "real_e79_read": False,
            "kaggle_upload": False,
            "all_synthetic_files_inside_temp": True,
        }


def _synthetic_qa() -> dict[str, Any]:
    n = 80
    ids = np.asarray([f"synthetic-e79-{i:04d}" for i in range(n)])
    raw = np.linspace(0.0, 100.0, n, dtype="<f8")
    missing = np.zeros(n, dtype=bool)
    missing[[2, 17, 41, 66]] = True
    delta = -1.25
    values = apply_values(raw, missing, delta)
    payload = render_csv(ids, values)
    audit = verify_csv_bytes(payload, ids, values, missing, raw,
                              expected_rows=n, expected_missing=int(missing.sum()))
    with tempfile.TemporaryDirectory(prefix="center-local-e79-synthetic-") as temp:
        path = Path(temp) / "candidate.csv"
        first_resumed = write_csv_idempotent(path, payload)
        second_resumed = write_csv_idempotent(path, payload)
        parsed = verify_csv_bytes(path.read_bytes(), ids, values, missing, raw,
                                  expected_rows=n, expected_missing=int(missing.sum()))
        tamper_checks = {}
        malformed = {
            "reordered": render_csv(ids[::-1], values[::-1]),
            "header": b"record_id,prediction\n" + b"\n".join(payload.splitlines()[1:]) + b"\n",
            "range": b"record_id,threat_rank\nsynthetic-e79-0000,101\n" + b"\n".join(payload.splitlines()[2:]) + b"\n",
            "missing_route": render_csv(ids, np.where(missing, np.nextafter(values, 100.0), values)),
            "duplicate_id": (b"record_id,threat_rank\n"
                             + b"synthetic-e79-0000," + format(float(values[0]), ".17g").encode("ascii") + b"\n"
                             + b"synthetic-e79-0000," + format(float(values[1]), ".17g").encode("ascii") + b"\n"
                             + b"\n".join(payload.splitlines()[3:]) + b"\n"),
        }
        for name, candidate in malformed.items():
            try:
                verify_csv_bytes(candidate, ids, values, missing, raw,
                                 expected_rows=n, expected_missing=int(missing.sum()))
            except ApplicationGateError:
                tamper_checks[name] = True
            else:
                tamper_checks[name] = False
        tamper_checks["output_file_conflict"] = False
        conflict = path.read_bytes() + b"tamper"
        try:
            write_csv_idempotent(path, conflict)
        except ApplicationGateError:
            tamper_checks["output_file_conflict"] = True
        if not all(tamper_checks.values()):
            raise ApplicationGateError("Synthetic CSV tamper suite did not reject every mutation")
        full_application = _synthetic_full_application()
        return {
            "status": "PASS_SYNTHETIC_E79_APPLICATION_CSV_QA",
            "rows": n,
            "missing_rows": int(missing.sum()),
            "columns": EXPECTED_COLUMNS,
            "first_write_resumed": first_resumed,
            "second_write_resumed": second_resumed,
            "idempotent_csv_sha256": parsed["csv_sha256"],
            "checks": tamper_checks,
            "real_e79_read": False,
            "csv_written_outside_temp": False,
            "production_csv_created": False,
            "kaggle_upload": False,
            "full_application": full_application,
        }


def _verify_application_source_subprocess(layout: Layout) -> dict[str, Any]:
    verifier = layout.root / "scripts/verify_reg_center_local_e79_application_r4.py"
    result = subprocess.run([sys.executable, str(verifier), "--verify-source",
                             "--root", str(layout.root)], cwd=layout.root,
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ApplicationGateError(f"Independent E79 source verifier failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _layout_from_args(root: Path | None, synthetic: bool) -> Layout:
    resolved = Path(root).resolve() if root else ROOT
    if synthetic:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if resolved == ROOT or (temp_root not in resolved.parents and resolved != temp_root):
            raise ApplicationGateError("Synthetic E79 QA is restricted to temporary directories")
    elif resolved != ROOT:
        raise ApplicationGateError("Alternate roots require isolated synthetic mode")
    return make_layout(resolved, synthetic=synthetic)


def main(argv: Sequence[str] | None = None, *, emit: bool = True) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--record-prior-failure", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--verify-source", action="store_true")
    action.add_argument("--apply", action="store_true")
    action.add_argument("--synthetic-qa", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.synthetic_qa:
        result = _synthetic_qa()
    else:
        layout = _layout_from_args(args.root, args.synthetic_fixture)
        if args.record_prior_failure:
            record = create_prior_failure_record(layout)
            result = {"status": record["status"],
                      "failure_record_sha256": sha256_file(layout.prior_application_paths["failure_record_r3"])}
        elif args.freeze:
            result = {"status": create_freeze(layout)["status"],
                      "freeze_sha256": sha256_file(layout.freeze)}
        elif args.preflight:
            result = {"status": create_preflight(layout)["status"],
                      "preflight_sha256": sha256_file(layout.preflight)}
        elif args.verify_source:
            result = _verify_application_source_subprocess(layout)
        else:
            result = apply_once(layout)
    if emit:
        print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        main()
    except ApplicationGateError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
