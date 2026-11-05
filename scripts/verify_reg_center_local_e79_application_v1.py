"""Independent source and approved-output verification for the E79 app lane."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

import numpy as np

import reg_center_local_e79_application_v1 as runner


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise runner.ApplicationGateError(f"Cannot read verification metadata: {path}") from exc
    if not isinstance(value, dict):
        raise runner.ApplicationGateError(f"Expected metadata object: {path}")
    return value


def _read_signed(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("signature") != _canonical({k: v for k, v in value.items() if k != "signature"}):
        raise runner.ApplicationGateError(f"Canonical signature mismatch: {path}")
    return value


def _file_inventory(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    result = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise runner.ApplicationGateError(f"Pinned source or metadata missing: {name}")
        result[name] = {"path": str(path), "sha256": _sha_file(path)}
    return result


def _metadata_paths(layout: runner.Layout, parent_freeze: Mapping[str, Any]) -> dict[str, Path]:
    inputs = parent_freeze["input_hashes"]
    names = ("e79_manifest", "e79_qa", "e79_prelabel_receipt",
             "e79_trained_state_freeze", "direct_freeze", "direct_prelabel_receipt")
    return {name: Path(inputs[name]["path"]).resolve() for name in names}


def _replay_lineage(layout: runner.Layout) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute producer pins and boundary facts without opening the E79 archive."""
    fit_paths = layout.fit_paths
    parent_paths = layout.parent_paths
    fit_freeze = _read_signed(fit_paths["fit_freeze"])
    fit_preflight = _read_signed(fit_paths["fit_preflight"])
    fit_pass = _read_signed(fit_paths["fit_root_pass"])
    fit_state = _read_signed(fit_paths["fit_state"])
    fit_receipt = _read_signed(fit_paths["fit_receipt"])
    fit_qa = _read_signed(fit_paths["fit_numerical_qa"])
    parent_freeze = _read_signed(parent_paths["parent_freeze"])
    parent_pass = _read_signed(parent_paths["parent_pass"])
    parent_preflight = _read_signed(parent_paths["parent_preflight"])
    parent_report = _read_signed(parent_paths["parent_report"])
    parent_verification = _read_json(parent_paths["parent_verification"])
    parent_receipt = _read_json(parent_paths["parent_receipt"])

    metadata_paths = _metadata_paths(layout, parent_freeze)
    pins = {
        **_file_inventory(fit_paths),
        **_file_inventory(parent_paths),
        **_file_inventory(metadata_paths),
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
    delta = float(fit_state.get("delta", math.nan))

    if (fit_state.get("status") != "PASS_LOCAL_SCALAR_FIT_COMPLETE"
            or fit_state.get("experiment_id") != runner.FIT_EXPERIMENT_ID
            or fit_state.get("fit_id") != runner.FIT_ID
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
            or not runner.DELTA_BOUNDS[0] <= delta <= runner.DELTA_BOUNDS[1]):
        raise runner.ApplicationGateError("Independent fit producer replay failed")

    e79_input = parent_freeze.get("input_hashes", {}).get("e79_archive", {})
    archive_path = str(Path(e79_input.get("path", "")).resolve())
    archive_sha = str(e79_input.get("sha256", ""))
    app_contract = parent_freeze.get("application_contract", {})
    parent_audit = parent_preflight.get("e79_raw_direct_audit", {})
    if not archive_sha or not archive_path:
        raise runner.ApplicationGateError("Parent freeze does not bind an E79 archive")

    metadata: dict[str, dict[str, Any]] = {}
    for name in ("e79_manifest", "e79_qa", "e79_prelabel_receipt", "e79_trained_state_freeze"):
        metadata[name] = _read_json(metadata_paths[name])
    manifest = metadata["e79_manifest"]
    qa = metadata["e79_qa"]
    prelabel = metadata["e79_prelabel_receipt"]
    trained = metadata["e79_trained_state_freeze"]
    if (archive_sha != app_contract.get("raw_direct_archive_sha256")
            or int(app_contract.get("rows", -1)) != runner.EXPECTED_ROWS
            or int(app_contract.get("missing_rows", -1)) != runner.EXPECTED_MISSING
            or app_contract.get("test_or_sample_read") is not False
            or manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or manifest.get("prediction_rows") != runner.EXPECTED_ROWS
            or manifest.get("missing_rows") != runner.EXPECTED_MISSING
            or manifest.get("test_labels_loaded") is not False
            or manifest.get("upload_performed") is not False
            or qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or qa.get("missing_rows") != runner.EXPECTED_MISSING
            or qa.get("test_labels_read") is not False
            or prelabel.get("status") != "PRELABEL_PASS"
            or prelabel.get("test_labels_loaded") is not False
            or trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY"
            or trained.get("test_labels_loaded") is not False
            or parent_audit.get("archive_sha256") != archive_sha
            or parent_audit.get("rows") != runner.EXPECTED_ROWS
            or parent_audit.get("unique_ids") != runner.EXPECTED_ROWS
            or parent_audit.get("missing_rows") != runner.EXPECTED_MISSING
            or parent_audit.get("missing_route_bit_exact_to_baseline") is not True):
        raise runner.ApplicationGateError("Inherited E79 metadata or label boundary replay failed")

    if (parent_pass.get("verdict") != "PASS"
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
        raise runner.ApplicationGateError("Independent FULLREG producer receipt replay failed")

    lineage = {
        "file_pins": pins,
        "fit": {
            "fit_state_sha256": fit_state_sha,
            "fit_receipt_sha256": fit_receipt_sha,
            "independent_fit_qa_sha256": fit_qa_sha,
            "fit_root_source_pass_sha256": fit_pass_sha,
            "fit_delta": delta,
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
            "path": archive_path,
            "expected_sha256_from_parent_freeze": archive_sha,
            "live_archive_hash_rechecked": False,
            "archive_opened": False,
            "expected_rows": runner.EXPECTED_ROWS,
            "expected_missing_rows": runner.EXPECTED_MISSING,
            "ordered_ids_sha256_from_parent_preflight": parent_audit["ordered_ids_sha256"],
            "raw_direct_sha256_from_parent_preflight": parent_audit["raw_direct_sha256"],
            "required_members": list(runner.REQUIRED_E79_MEMBERS),
            "expected_extra_non_target_members": parent_audit.get("unloaded_extra_member_names", []),
            "missing_route_source": "baseline member; prelabel audit established raw DIRECT bits equal baseline on missing rows",
        },
        "lineage_files_sha256": pins,
    }
    inventory = {
        "files": {
            "sources": _file_inventory(layout.source_paths),
            "fit_and_producer_receipts": pins,
        },
        "deferred_e79_input": lineage["e79_deferred_input"],
    }
    return lineage, inventory


def verify_source(root: Path) -> dict[str, Any]:
    layout = runner.make_layout(root)
    freeze = _read_signed(layout.freeze)
    preflight = _read_signed(layout.preflight)
    lineage, inventory = _replay_lineage(layout)
    prior_paths = {
        "freeze_r1_sha256": layout.experiment_dir / "E79_APPLICATION_FREEZE_R1.json",
        "preflight_r1_sha256": layout.report_dir / "E79_APPLICATION_PREFLIGHT_R1.json",
        "freeze_r2_sha256": layout.experiment_dir / "E79_APPLICATION_FREEZE_R2.json",
        "preflight_r2_sha256": layout.report_dir / "E79_APPLICATION_PREFLIGHT_R2.json",
    }
    supersedes = {
        **{name: _sha_file(path) for name, path in prior_paths.items()},
        "reason": "R1 and R2 independent source replays failed before E79 access; R3 validates the unsigned preflight body while separately checking its signature.",
    }
    if (freeze.get("experiment_id") != runner.EXPERIMENT_ID
            or freeze.get("status") != "E79_APPLICATION_SOURCE_FROZEN_PENDING_ROOT_PASS"
            or freeze.get("supersedes") != supersedes
            or freeze.get("inventory") != inventory
            or freeze.get("inventory_sha256") != _canonical(inventory)
            or freeze.get("lineage") != lineage):
        raise runner.ApplicationGateError("Independent source inventory or ancestry replay failed")
    boundary = freeze.get("access_boundary", {})
    if any(boundary.get(key) is not False for key in (
            "e79_archive_opened", "e79_archive_bytes_rehashed", "test_or_sample_read",
            "csv_created", "kaggle_accessed")):
        raise runner.ApplicationGateError("Source freeze reports an open E79/output boundary")
    deferred = lineage["e79_deferred_input"]
    expected_preflight = {
        "schema_version": 1,
        "status": "PASS_E79_APPLICATION_SOURCE_PREFLIGHT_PENDING_ROOT_PASS",
        "experiment_id": runner.EXPERIMENT_ID,
        "application_freeze_sha256": _sha_file(layout.freeze),
        "inventory_sha256": freeze["inventory_sha256"],
        "source_datahash": {
            "e79_archive_path": deferred["path"],
            "e79_expected_sha256_inherited": deferred["expected_sha256_from_parent_freeze"],
            "live_archive_hash_rechecked": False,
            "e79_archive_opened": False,
        },
        "e79_contract_from_signed_parent_receipts": {
            "rows": runner.EXPECTED_ROWS,
            "unique_ids": runner.EXPECTED_ROWS,
            "missing_rows": runner.EXPECTED_MISSING,
            "ordered_ids_sha256": deferred["ordered_ids_sha256_from_parent_preflight"],
            "raw_direct_sha256": deferred["raw_direct_sha256_from_parent_preflight"],
            "required_non_target_members": list(runner.REQUIRED_E79_MEMBERS),
            "extra_non_target_member_names_not_loaded": deferred["expected_extra_non_target_members"],
            "output_columns": list(runner.EXPECTED_COLUMNS),
        },
        "fit": {
            "fit_state_sha256": lineage["fit"]["fit_state_sha256"],
            "fit_receipt_sha256": lineage["fit"]["fit_receipt_sha256"],
            "independent_fit_qa_sha256": lineage["fit"]["independent_fit_qa_sha256"],
            "delta": lineage["fit"]["fit_delta"],
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
    if {key: value for key, value in preflight.items() if key != "signature"} != expected_preflight:
        raise runner.ApplicationGateError("Independent source preflight replay failed")
    body = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_E79_APPLICATION_SOURCE_VERIFICATION",
        "experiment_id": runner.EXPERIMENT_ID,
        "application_freeze_sha256": _sha_file(layout.freeze),
        "application_preflight_sha256": _sha_file(layout.preflight),
        "source_inventory_sha256": freeze["inventory_sha256"],
        "lineage_files_sha256": lineage["lineage_files_sha256"],
        "e79_archive_expected_sha256": deferred["expected_sha256_from_parent_freeze"],
        "live_e79_archive_hash_rechecked": False,
        "e79_archive_opened": False,
        "test_or_sample_read": False,
        "csv_created": False,
        "kaggle_upload": False,
    }
    path = layout.source_verification
    if path.exists():
        existing = _read_signed(path)
        if {k: v for k, v in existing.items() if k != "signature"} != body:
            raise runner.ApplicationGateError("Existing source verification receipt conflicts")
        return existing
    body["signature"] = _canonical(body)
    _write_once(path, body)
    return body


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
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
        raise runner.ApplicationGateError("Refusing to overwrite source verification receipt") from exc


def verify_application(root: Path) -> dict[str, Any]:
    """Independently replay an application only after the exact root pass exists."""
    layout = runner.make_layout(root)
    _approval, pass_sha = runner.validate_application_pass(layout)
    report = _read_signed(layout.application_report)
    expected_archive = report.get("e79_archive_sha256")
    archive_path = Path(_read_signed(layout.freeze)["lineage"]["e79_deferred_input"]["path"])
    if _sha_file(archive_path) != expected_archive:
        raise runner.ApplicationGateError("Approved E79 archive hash changed")
    with zipfile.ZipFile(archive_path) as archive:
        member_names = {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}
    forbidden = sorted(name for name in member_names - set(runner.REQUIRED_E79_MEMBERS)
                       if any(token in name.casefold() for token in runner.FORBIDDEN_MEMBER_TOKENS))
    if not set(runner.REQUIRED_E79_MEMBERS).issubset(member_names) or forbidden:
        raise runner.ApplicationGateError("Approved archive member allowlist failed independent replay")
    with np.load(archive_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["record_id"], dtype=str).copy()
        raw = np.asarray(archive["prediction"], dtype="<f8").copy()
        baseline = np.asarray(archive["baseline"], dtype="<f8").copy()
        missing = np.asarray(archive["missing"], dtype=bool).copy()
    if (len(ids) != runner.EXPECTED_ROWS or len(set(ids.tolist())) != runner.EXPECTED_ROWS
            or int(missing.sum()) != runner.EXPECTED_MISSING
            or not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 100.0))
            or not np.array_equal(raw[missing].view("<u8"), baseline[missing].view("<u8"))
            or _ordered_hash(ids) != _read_signed(layout.parent_paths["parent_preflight"])["e79_raw_direct_audit"]["ordered_ids_sha256"]):
        raise runner.ApplicationGateError("Approved E79 identity or missing route failed independent replay")
    delta = float(report["delta"])
    values = raw.copy()
    present = ~missing
    p = raw[present]
    coeff = runner.HINGE
    base = (coeff["intercept"] + coeff["slope_low"] * np.minimum(p, 50.0)
            + coeff["slope_high"] * np.maximum(p - 50.0, 0.0))
    center = np.maximum(0.0, 1.0 - np.abs(p - 50.0) / 10.0)
    values[present] = np.minimum(100.0, np.maximum(0.0, base + delta * center))
    csv_bytes = layout.output_csv.read_bytes()
    try:
        rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"), newline="")))
        csv_ids = [row[0] for row in rows[1:]]
        csv_values = np.asarray([float(row[1]) for row in rows[1:]], dtype="<f8")
    except (UnicodeDecodeError, csv.Error, ValueError, IndexError) as exc:
        raise runner.ApplicationGateError("Approved CSV is malformed") from exc
    if (not rows or tuple(rows[0]) != runner.EXPECTED_COLUMNS or len(rows) != runner.EXPECTED_ROWS + 1
            or csv_ids != ids.tolist() or len(set(csv_ids)) != runner.EXPECTED_ROWS
            or not np.isfinite(csv_values).all() or np.any((csv_values < 0.0) | (csv_values > 100.0))
            or not np.array_equal(csv_values.view("<u8"), values.view("<u8"))
            or not np.array_equal(csv_values[missing].view("<u8"), raw[missing].view("<u8"))):
        raise runner.ApplicationGateError("Approved CSV does not match independent application replay")
    if (report.get("status") != "PASS_LOCAL_E79_APPLICATION_PENDING_INDEPENDENT_VERIFICATION"
            or report.get("root_application_pass_sha256") != pass_sha
            or report.get("rows") != runner.EXPECTED_ROWS
            or report.get("missing_rows") != runner.EXPECTED_MISSING
            or report.get("csv_sha256") != hashlib.sha256(csv_bytes).hexdigest()
            or report.get("base_model_fits") != 0 or report.get("calibrator_fits") != 0
            or report.get("test_or_sample_read") is not False
            or report.get("e79_target_values_loaded") is not False
            or report.get("kaggle_upload") is not False):
        raise runner.ApplicationGateError("Application report does not bind replayed output")
    return {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_LOCAL_E79_APPLICATION",
        "experiment_id": runner.EXPERIMENT_ID,
        "application_report_sha256": _sha_file(layout.application_report),
        "application_pass_sha256": pass_sha,
        "e79_archive_sha256": expected_archive,
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "rows": runner.EXPECTED_ROWS,
        "missing_rows": runner.EXPECTED_MISSING,
        "ordered_ids_sha256": _ordered_hash(ids),
        "columns": list(runner.EXPECTED_COLUMNS),
        "missing_route_bit_exact": True,
        "base_model_fits": 0,
        "calibrator_fits": 0,
        "e79_target_values_loaded": False,
        "test_or_sample_read": False,
        "kaggle_upload": False,
    }


def _ordered_hash(value: Any) -> str:
    ids = np.asarray(value, dtype=str).tolist()
    payload = json.dumps(ids, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-source", action="store_true")
    modes.add_argument("--verify-application", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_source(args.root.resolve()) if args.verify_source else verify_application(args.root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except runner.ApplicationGateError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
