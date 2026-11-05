"""Separately gated one-slot exploratory E79 application for the fixed-p50 hinge.

Freeze and preflight are outcome-blind. The official Train target reader, final
LP, and CSV writer stay behind a new root pass that authorizes HINGE_FINAL only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reg_final_wave_v2 as v2  # noqa: E402
from scripts import reg_monotone_hinge_p50_v4 as r4  # noqa: E402
from scripts.reg_position_failure_mechanism_v1 import read_scored_query  # noqa: E402

EXPERIMENT_ID = "REG_MONOTONE_HINGE_P50_FINAL_APPLICATION_R1"
REPORT_DIR = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1"
EXPERIMENT_DIR = ROOT / "experiments/reg_monotone_hinge_p50_v1"
RUN_DIR = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1"
SUBMISSION_DIR = ROOT / "submissions/reg_monotone_hinge_p50_v1"

SCRIPT = Path(__file__).resolve()
VERIFIER = ROOT / "scripts/verify_reg_monotone_hinge_p50_final_application_v1.py"
TESTS = ROOT / "tests/test_reg_monotone_hinge_p50_final_application_v1.py"
PROTOCOL = REPORT_DIR / "FINAL_APPLICATION_PROTOCOL_R1.md"
SYNTHETIC_QA = REPORT_DIR / "FINAL_APPLICATION_SYNTHETIC_QA_R1.json"
DECISION_DOC = REPORT_DIR / "SOL_DIAGNOSTIC_DECISION_R4.md"
SOURCE_REVIEW_R4 = REPORT_DIR / "SOL_SOURCE_REVIEW_R4.md"
GITATTRIBUTES = ROOT / ".gitattributes"

FREEZE = EXPERIMENT_DIR / "FINAL_APPLICATION_FREEZE_R1.json"
PREFLIGHT = REPORT_DIR / "FINAL_APPLICATION_PREFLIGHT_R1.json"
ROOT_SOURCE_PASS = EXPERIMENT_DIR / "FINAL_APPLICATION_ROOT_SOURCE_PASS_R1.json"
FINAL_REPORT = REPORT_DIR / "FINAL_APPLICATION_REPORT_R1.json"
FINAL_VERIFICATION = REPORT_DIR / "FINAL_APPLICATION_VERIFICATION_R1.json"
FINAL_RECEIPT = REPORT_DIR / "FINAL_APPLICATION_RECEIPT_R1.json"
OUTPUT_CSV = SUBMISSION_DIR / "hinge_p50_final_exploratory_e79.csv"

# The third event belongs to the already existing R4 three-slot ledger.
LEDGER = r4.LEDGER
SHARED_JOB_LOCK = r4.SHARED_JOB_LOCK
FIT_ID = "HINGE_FINAL"
AUTHORIZED_FIT_IDS = (FIT_ID,)
TOTAL_LP_CAP = 3
PRIOR_DIAGNOSTIC_FIT_IDS = ("HINGE_D1", "HINGE_D2")
EXPECTED_OOF_ROWS = 20_903
EXPECTED_TEST_ROWS = 29_421
EXPECTED_MISSING_ROWS = 1_062
KNOT = 50.0
SLOPE_BOUNDS = (0.0, 2.0)
THREADS = 2
TIME_LIMIT_SECONDS = 180

R4_FREEZE_SHA256 = "97163e0ff187da7ac82f9e84319db89979c00eba0998016dd14ca25910226bfa"
R4_ROOT_SOURCE_PASS_SHA256 = "adbeaac6a53afd095eb53ffe372eb89de9ec1f3e47dd7794ecafb163ff394bac"
R4_PREFLIGHT_SHA256 = "edacdb903da79380e6232cfbae48350acd72e9511e9adc25deaf172fc4356810"
R4_DIAGNOSTICS_SHA256 = "ea6a77cbb43c836e08136a8c417e02384084deabc1c42fa94226f719e20f7b1e"
R4_VERIFICATION_SHA256 = "f89a2e2f21afacd79f88d3253e76a3207e25c7cd29fae5f5e8cbdc21586f9f2f"
R4_RESOURCE_SHA256 = "cf1f1363df0ed76b2ec2c1e56e9ff4899d11f3f92165286a408de74e9ce349e8"

R4_DIAGNOSTICS = r4.DIAGNOSTICS
R4_VERIFICATION = r4.VERIFICATION
R4_RESOURCE_RECEIPT = r4.RUN_DIR / "RESOURCE_SNAPSHOTS_R4.json"
R4_FINAL_STATE_PATHS = {
    fit_id: r4.RUN_DIR / "fits" / f"{fit_id}.json"
    for fit_id in PRIOR_DIAGNOSTIC_FIT_IDS
}

PREFLIGHT_REHASH_KEYS = (
    "r4_identity_freeze", "r4_root_source_pass", "r4_identity_preflight",
    "r4_diagnostics", "r4_diagnostics_verification", "r4_resource_receipt",
    "sol_diagnostic_decision_r4", "sol_source_review_r4", "r4_lp_ledger",
    "r4_fit_state_HINGE_D1", "r4_fit_state_HINGE_D2",
    "e79_archive", "e79_manifest", "e79_qa", "e79_trained_state_freeze",
    "e79_prelabel_receipt", "official_train_deferred",
    "direct_freeze", "direct_prelabel_receipt",
    "direct_E67", "direct_E67_receipt", "direct_E75", "direct_E75_receipt",
    "user_revised_resource_policy",
)

E79_NPZ_ALLOWLIST = ("record_id", "prediction", "baseline", "missing")
E79_FORBIDDEN_NAME_TOKENS = ("target", "label", "outcome", "threat_rank", "score")
OUTPUT_COLUMNS = ("record_id", "threat_rank")


class GateError(RuntimeError):
    """Fail-closed identity, source, resource, slot, or output boundary error."""


def sha_file(path: Path) -> str:
    return r4.file_sha256(Path(path))


def canonical_sha(value: Any) -> str:
    return r4.canonical_sha(value)


def _source_paths() -> dict[str, Path]:
    return {
        "final_application_runner": SCRIPT,
        "final_application_verifier": VERIFIER,
        "final_application_tests": TESTS,
        "final_application_protocol": PROTOCOL,
        "final_application_synthetic_qa": SYNTHETIC_QA,
        "diagnostic_decision_r4": DECISION_DOC,
        "diagnostic_source_review_r4": SOURCE_REVIEW_R4,
        "gitattributes": GITATTRIBUTES,
        "r4_runner": Path(r4.__file__).resolve(),
        "r4_verifier": r4.VERIFIER,
        "r4_tests": r4.TESTS,
        "r4_protocol": r4.PROTOCOL,
        "r4_synthetic_qa": r4.SYNTHETIC_QA,
        "r1_hinge_core": Path(r4.r1.__file__).resolve(),
        "r1_hinge_verifier": Path(r4.r1.__file__).with_name("verify_reg_monotone_hinge_p50_v1.py"),
        "direct_helpers": Path(v2.__file__).resolve(),
        "direct_model_source": v2.PINS["direct_source"][0],
        "selected_train_row_reader": Path(read_scored_query.__code__.co_filename).resolve(),
    }


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in _source_paths().items():
        if not path.is_file():
            raise GateError(f"Final-application source is missing: {name} ({path})")
        result[name] = sha_file(path)
    return result


def _source_ancestor(commit: str) -> bool:
    result = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                             commit, "HEAD"], capture_output=True)
    return result.returncode == 0


def _assert_committed_sources(hashes: Mapping[str, str]) -> None:
    for name, path in _source_paths().items():
        try:
            relative = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            # External helper sources are content-pinned in the freeze; they are
            # not part of this repository's commit tree.
            continue
        result = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                                capture_output=True)
        if result.returncode != 0 or hashlib.sha256(result.stdout).hexdigest() != hashes.get(name):
            raise GateError(f"Final-application source is not committed at HEAD: {relative}")


def _expected_r4_pins() -> dict[str, tuple[Path, str]]:
    return {
        "r4_identity_freeze": (r4.PREFIT_FREEZE, R4_FREEZE_SHA256),
        "r4_root_source_pass": (r4.ROOT_SOURCE_PASS, R4_ROOT_SOURCE_PASS_SHA256),
        "r4_identity_preflight": (r4.IDENTITY_PREFLIGHT, R4_PREFLIGHT_SHA256),
        "r4_diagnostics": (R4_DIAGNOSTICS, R4_DIAGNOSTICS_SHA256),
        "r4_diagnostics_verification": (R4_VERIFICATION, R4_VERIFICATION_SHA256),
        "r4_resource_receipt": (R4_RESOURCE_RECEIPT, R4_RESOURCE_SHA256),
        "sol_diagnostic_decision_r4": (DECISION_DOC, sha_file(DECISION_DOC)),
        "sol_source_review_r4": (SOURCE_REVIEW_R4, sha_file(SOURCE_REVIEW_R4)),
        "r4_lp_ledger": (LEDGER, sha_file(LEDGER)),
        "r4_fit_state_HINGE_D1": (R4_FINAL_STATE_PATHS["HINGE_D1"], sha_file(R4_FINAL_STATE_PATHS["HINGE_D1"])),
        "r4_fit_state_HINGE_D2": (R4_FINAL_STATE_PATHS["HINGE_D2"], sha_file(R4_FINAL_STATE_PATHS["HINGE_D2"])),
    }


def _r4_preflight_input_pin(freeze: Mapping[str, Any], name: str) -> tuple[Path, str]:
    item = freeze.get("input_hashes", {}).get(name)
    if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
        raise GateError(f"R4 identity freeze lacks required input pin: {name}")
    return Path(item["path"]), str(item["sha256"])


def _source_pin_from_r4(freeze: Mapping[str, Any], name: str) -> tuple[Path, str]:
    path, expected = _r4_preflight_input_pin(freeze, name)
    if not path.is_file() or sha_file(path) != expected:
        raise GateError(f"Inherited R4 input pin changed: {name}")
    return path, expected


def _verify_r4_predecessor_with_all_pins() -> tuple[dict[str, Any], dict[str, Any], str]:
    """Rehash every frozen R4 input; this hashes Train bytes but parses no labels."""
    freeze, approval, pass_sha = r4.verify_source_pass()
    if (pass_sha != R4_ROOT_SOURCE_PASS_SHA256
            or sha_file(r4.PREFIT_FREEZE) != R4_FREEZE_SHA256
            or sha_file(r4.IDENTITY_PREFLIGHT) != R4_PREFLIGHT_SHA256):
        raise GateError("R4 predecessor freeze/pass/preflight differs from reviewed source")
    return freeze, approval, pass_sha


def _lineage_and_decision() -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    r4_freeze, _r4_approval, r4_pass_sha = _verify_r4_predecessor_with_all_pins()
    if (r4_pass_sha != R4_ROOT_SOURCE_PASS_SHA256
            or sha_file(r4.PREFIT_FREEZE) != R4_FREEZE_SHA256
            or sha_file(r4.IDENTITY_PREFLIGHT) != R4_PREFLIGHT_SHA256):
        raise GateError("R4 predecessor freeze/pass/preflight differs from the reviewed source")
    if (sha_file(R4_DIAGNOSTICS) != R4_DIAGNOSTICS_SHA256
            or sha_file(R4_VERIFICATION) != R4_VERIFICATION_SHA256
            or sha_file(R4_RESOURCE_RECEIPT) != R4_RESOURCE_SHA256):
        raise GateError("R4 diagnostic result receipts differ from Sol's independent review")

    diagnostics = json.loads(R4_DIAGNOSTICS.read_text(encoding="utf-8"))
    verification = json.loads(R4_VERIFICATION.read_text(encoding="utf-8"))
    if (diagnostics.get("status") != "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
            or diagnostics.get("root_source_pass_sha256") != r4_pass_sha
            or diagnostics.get("diagnostic_fit_ids") != list(PRIOR_DIAGNOSTIC_FIT_IDS)
            or diagnostics.get("official_lp_starts") != 2
            or diagnostics.get("decision", {}).get("status") != "EXPLORATORY_ROOT_REVIEW"
            or diagnostics.get("decision", {}).get("qualified_gate") is not False
            or diagnostics.get("decision", {}).get("exploratory_root_review_gate") is not True
            or verification.get("status") != "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
            or verification.get("diagnostics_sha256") != R4_DIAGNOSTICS_SHA256
            or verification.get("root_source_pass_sha256") != r4_pass_sha
            or verification.get("pooled_unique_rows") != r4.EXPECTED_POOLED_ROWS
            or verification.get("official_lp_starts") != 2):
        raise GateError("R4 diagnostics do not match the reviewed exploratory-only decision")

    stages = {item.get("fit_id"): item for item in diagnostics.get("stages", [])}
    if set(stages) != set(PRIOR_DIAGNOSTIC_FIT_IDS):
        raise GateError("R4 diagnostics must contain exactly D1 and D2")
    window_gains: dict[str, float] = {}
    p90_worsening: dict[str, float] = {}
    upper_tail_harm: dict[str, float] = {}
    for fit_id in PRIOR_DIAGNOSTIC_FIT_IDS:
        record = stages[fit_id]
        metrics = record["score_metrics"]
        window_gains[fit_id] = float(record["gain_vs_causal_global_present_only"])
        p90_worsening[fit_id] = float(
            metrics["hinge_present_only"]["overall"]["p90_absolute_error"]
            - metrics["causal_global_present_only"]["overall"]["p90_absolute_error"])
        upper_tail_harm[fit_id] = float(
            record["supported_subgroup_harm"]["groups"]["target_ge_95"]["harm"])
    decision = diagnostics["decision"]
    if (any(value <= 0.0 for value in window_gains.values())
            or any(value <= 0.0 for value in p90_worsening.values())
            or any(value <= 0.0 for value in upper_tail_harm.values())
            or float(decision["pooled_gain_vs_causal_global_present_only"]) >= 0.02
            or float(decision["maximum_supported_subgroup_harm"]) >= 0.25):
        raise GateError("R4 exploratory entry/counterevidence values differ from Sol's decision")
    return r4_freeze, diagnostics, r4_pass_sha, {
        "scientific_qualification": "FAIL",
        "diagnostic_status": "EXPLORATORY_ROOT_REVIEW",
        "pooled_gain_vs_causal_global_present_only": float(
            decision["pooled_gain_vs_causal_global_present_only"]),
        "qualified_threshold": 0.02,
        "qualified_gate": False,
        "exploratory_review_gate": True,
        "window_gains": window_gains,
        "window_p90_absolute_error_worsening": p90_worsening,
        "upper_tail_target_ge_95_harm": upper_tail_harm,
        "maximum_supported_subgroup_harm": float(
            decision["maximum_supported_subgroup_harm"]),
        "upload_authorized": False,
        "user_batch_preference": "collect several research results before selecting a benchmark batch",
    }


def _validate_prior_ledger(r4_freeze: Mapping[str, Any], *,
                           allow_final: bool = False) -> dict[str, dict[str, list[dict[str, Any]]]]:
    ledger = r4.read_ledger(LEDGER)
    expected_ids = set(PRIOR_DIAGNOSTIC_FIT_IDS)
    if allow_final and FIT_ID in ledger:
        expected_ids.add(FIT_ID)
    if set(ledger) != expected_ids:
        raise GateError(f"Cumulative ledger fit IDs differ from the allowed prefix: {sorted(ledger)}")
    for fit_id in PRIOR_DIAGNOSTIC_FIT_IDS:
        stage = ledger.get(fit_id, {})
        if (len(stage.get("RESERVE", [])) != 1
                or len(stage.get("LP_STARTED", [])) != 1
                or len(stage.get("LP_COMPLETE", [])) != 1
                or stage.get("LP_FAILED")
                or stage["RESERVE"][0].get("attempt") != 1
                or stage["RESERVE"][0].get("retry_allowed") is not False
                or stage["LP_STARTED"][0].get("retry_allowed") is not False):
            raise GateError(f"Historical {fit_id} accounting is not one completed non-retry slot")
        state_path = R4_FINAL_STATE_PATHS[fit_id]
        state_sha = sha_file(state_path)
        if state_sha != stage["LP_COMPLETE"][0].get("state_sha256"):
            raise GateError(f"Historical {fit_id} fit-state SHA differs from its completed ledger event")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("fit_id") != fit_id or state.get("status") != "LP_COMPLETE"
                or state.get("fit_input", {}).get("root_source_pass_sha256")
                    != R4_ROOT_SOURCE_PASS_SHA256):
            raise GateError(f"Historical {fit_id} state is not the R4 completed diagnostic")
    starts = sum(len(ledger.get(fit_id, {}).get("LP_STARTED", [])) for fit_id in ledger)
    final_starts = len(ledger.get(FIT_ID, {}).get("LP_STARTED", []))
    if starts != 2 + final_starts:
        raise GateError(f"Cumulative LP start count differs from slot prefix: {starts}")
    if FIT_ID in ledger and not allow_final:
        raise GateError("HINGE_FINAL already has an event; its slot is consumed and cannot be retried")
    return ledger


def _ledger_prefix_sha_before_final(path: Path = LEDGER) -> str:
    """Hash the exact original R4 JSONL prefix and reject rows after HINGE_FINAL."""
    if not path.is_file():
        raise GateError("Cumulative R4 ledger is absent")
    prefix: list[bytes] = []
    final_seen = False
    for raw_line in path.read_bytes().splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            raise GateError("Cumulative ledger has a partial final line")
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GateError("Cumulative ledger contains malformed JSON") from exc
        if event.get("fit_id") == FIT_ID:
            final_seen = True
        elif final_seen:
            raise GateError("R4 ledger contains an event after HINGE_FINAL began")
        else:
            prefix.append(raw_line)
    return hashlib.sha256(b"".join(prefix)).hexdigest()


def _input_hashes(r4_freeze: Mapping[str, Any], r4_freeze_sha: str,
                  r4_pass_sha: str, source_map: Mapping[str, str]) -> tuple[
                      dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    inputs: dict[str, dict[str, str]] = {}
    preflight: dict[str, dict[str, str]] = {}
    for name, (path, expected) in _expected_r4_pins().items():
        if not path.is_file() or sha_file(path) != expected:
            raise GateError(f"R4 handoff pin changed: {name}")
        inputs[name] = {"path": str(path.resolve()), "sha256": expected}
        preflight[name] = inputs[name]
    inputs["r4_source_commit"] = {
        "path": str(r4.PREFIT_FREEZE.resolve()),
        "sha256": str(r4_freeze.get("source_commit")),
    }
    inputs["r4_freeze_sha256"] = {"path": str(r4.PREFIT_FREEZE.resolve()),
                                  "sha256": r4_freeze_sha}
    inputs["r4_source_pass_sha256"] = {"path": str(r4.ROOT_SOURCE_PASS.resolve()),
                                       "sha256": r4_pass_sha}
    for name, old_key in (
        ("e79_archive", "r3_input__pinned_base_archive"),
        ("e79_manifest", "r3_input__pinned_base_manifest"),
        ("e79_qa", "r3_input__pinned_base_qa"),
        ("e79_trained_state_freeze", "r3_input__pinned_base_trained_freeze"),
        ("e79_prelabel_receipt", "r3_input__pinned_base_prelabel"),
        ("direct_freeze", "r3_input__pinned_direct_freeze"),
        ("direct_prelabel_receipt", "r3_input__pinned_direct_prelabel"),
        ("direct_E67", "r3_input__pinned_direct_E67"),
        ("direct_E67_receipt", "r3_input__pinned_direct_E67_receipt"),
        ("direct_E75", "r3_input__pinned_direct_E75"),
        ("direct_E75_receipt", "r3_input__pinned_direct_E75_receipt"),
        ("user_revised_resource_policy", "user_revised_resource_policy"),
    ):
        path, expected = _r4_preflight_input_pin(r4_freeze, old_key)
        inputs[name] = {"path": str(path.resolve()), "sha256": expected}
        if name in PREFLIGHT_REHASH_KEYS:
            preflight[name] = inputs[name]
    train_path, train_sha = _r4_preflight_input_pin(r4_freeze, "r3_input__official_train")
    inputs["official_train_deferred"] = {"path": str(train_path.resolve()), "sha256": train_sha}
    preflight["official_train_deferred"] = inputs["official_train_deferred"]
    # `source_map` is frozen separately; it is not an input file map entry.
    if not source_map:
        raise GateError("Final-application source hashes are empty")
    if set(PREFLIGHT_REHASH_KEYS) != set(preflight):
        raise GateError("Final-application preflight allowlist is incomplete")
    return inputs, preflight


def freeze_final_application() -> tuple[Path, str]:
    if FREEZE.exists() or PREFLIGHT.exists() or ROOT_SOURCE_PASS.exists():
        raise GateError("Final-application freeze/preflight/pass already exists; no overwrite")
    r4_freeze, _r4_diagnostics, r4_pass_sha, decision_context = _lineage_and_decision()
    prior_ledger = _validate_prior_ledger(r4_freeze)
    ledger_sha = sha_file(LEDGER)
    if len(prior_ledger) != 2:
        raise GateError("Final-application source requires the existing completed D1/D2 ledger only")
    source_map = source_hashes()
    _assert_committed_sources(source_map)
    r4_freeze_sha = sha_file(r4.PREFIT_FREEZE)
    inputs, preflight_inputs = _input_hashes(r4_freeze, r4_freeze_sha,
                                              r4_pass_sha, source_map)
    if sha_file(LEDGER) != ledger_sha:
        raise GateError("R4 ledger changed while final-application freeze was assembled")
    base = json.loads(_source_pin_from_r4(r4_freeze, "r3_input__pinned_base_manifest")[0]
                      .read_text(encoding="utf-8"))
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "FINAL_APPLICATION_FROZEN_PENDING_IDENTITY_PREFLIGHT",
        "source_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         check=True, capture_output=True, text=True).stdout.strip(),
        "source_hashes": source_map,
        "input_hashes": inputs,
        "preflight_input_allowlist": sorted(preflight_inputs),
        "preflight_input_hashes": preflight_inputs,
        "deferred_target_input": inputs["official_train_deferred"],
        "r4_predecessor": {
            "freeze_path": str(r4.PREFIT_FREEZE.resolve()),
            "freeze_sha256": r4_freeze_sha,
            "source_pass_path": str(r4.ROOT_SOURCE_PASS.resolve()),
            "source_pass_sha256": r4_pass_sha,
            "identity_preflight_sha256": R4_PREFLIGHT_SHA256,
            "diagnostics_sha256": R4_DIAGNOSTICS_SHA256,
            "independent_verification_sha256": R4_VERIFICATION_SHA256,
            "decision_path": str(DECISION_DOC.resolve()),
            "decision_sha256": sha_file(DECISION_DOC),
        },
        "prior_slot_accounting": {
            "ledger_path": str(LEDGER.resolve()),
            "ledger_sha256_before_final_slot": ledger_sha,
            "completed_fit_ids": list(PRIOR_DIAGNOSTIC_FIT_IDS),
            "completed_starts": 2,
            "r4_diagnostic_fit_state_sha256": {
                fit_id: sha_file(R4_FINAL_STATE_PATHS[fit_id])
                for fit_id in PRIOR_DIAGNOSTIC_FIT_IDS},
            "next_fit_id": FIT_ID,
            "final_slot_reserved": False,
            "reset_or_retry": False,
        },
        "final_authorization_contract": {
            "required_new_root_pass_path": str(ROOT_SOURCE_PASS.resolve()),
            "required_authorized_fit_ids": [FIT_ID],
            "required_final_starts": 1,
            "total_cap_including_r4": TOTAL_LP_CAP,
            "r4_diagnostic_pass_authorizes_final": False,
            "target_reader_allowed_before_new_pass": False,
            "final_lp_allowed_before_new_pass": False,
            "official_csv_allowed_before_new_pass": False,
        },
        "fit_contract": {
            "fit_id": FIT_ID,
            "rows": EXPECTED_OOF_ROWS,
            "prediction_origins_and_cycles": {
                "E67_cycles_68_75": 12_412,
                "E75_cycles_76_79": 8_491,
            },
            "objective": "unweighted sum of absolute errors",
            "model": "a + sL*min(p,50) + sH*max(p-50,0)",
            "knot": KNOT,
            "slope_bounds": list(SLOPE_BOUNDS),
            "threads": THREADS,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "no_d1_d2_refit": True,
            "no_global_control_fit": True,
            "no_new_base_model_fit": True,
        },
        "application_contract": {
            "raw_direct_archive_path": str(Path(inputs["e79_archive"]["path"]).resolve()),
            "raw_direct_archive_sha256": inputs["e79_archive"]["sha256"],
            "ids_sha256_from_signed_manifest": base.get("ordered_official_test_ids_sha256"),
            "rows": EXPECTED_TEST_ROWS,
            "missing_rows": EXPECTED_MISSING_ROWS,
            "columns": list(OUTPUT_COLUMNS),
            "present_route": "clip(a + sL*min(raw_DIRECT,50) + sH*max(raw_DIRECT-50,0),0,100)",
            "missing_route": "preserve raw_DIRECT float64 bits exactly",
            "test_or_sample_read": False,
            "upload": False,
        },
        "decision_context": decision_context,
        "candidate_admission": {
            "status": "EXPLORATORY_ONLY_PENDING_ROOT_REVIEW_AND_BATCH_SELECTION",
            "scientific_qualification": "FAIL",
            "validation_claim": None,
            "upload_authorized": False,
        },
        "preflight_status": "NOT_RUN",
        "official_lp_starts": 0,
        "train_targets_read": False,
        "train_target_source_hash_verified": True,
        "test_or_sample_read": False,
        "csv_created": False,
        "kaggle_upload": False,
    }
    body["signature"] = canonical_sha(body)
    return FREEZE, r4.write_json_once(FREEZE, body)


def verify_final_freeze(*, preflight_only: bool = False) -> tuple[dict[str, Any], str]:
    if not FREEZE.is_file():
        raise GateError("Final-application freeze is absent")
    freeze_sha = sha_file(FREEZE)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    body = {key: value for key, value in freeze.items() if key != "signature"}
    if (freeze.get("signature") != canonical_sha(body)
            or freeze.get("schema_version") != 1
            or freeze.get("experiment_id") != EXPERIMENT_ID
            or freeze.get("status") != "FINAL_APPLICATION_FROZEN_PENDING_IDENTITY_PREFLIGHT"
            or freeze.get("official_lp_starts") != 0
            or freeze.get("train_targets_read") is not False
            or freeze.get("train_target_source_hash_verified") is not True
            or freeze.get("test_or_sample_read") is not False
            or freeze.get("csv_created") is not False
            or not _source_ancestor(str(freeze.get("source_commit", "")))):
        raise GateError("Final-application freeze signature, status, or source ancestry failed")
    if source_hashes() != freeze.get("source_hashes"):
        raise GateError("Final-application source bytes differ from freeze")
    expected_inputs = freeze.get("preflight_input_hashes", {})
    if set(expected_inputs) != set(PREFLIGHT_REHASH_KEYS):
        raise GateError("Final-application freeze preflight input allowlist differs")
    for name in PREFLIGHT_REHASH_KEYS:
        item = expected_inputs[name]
        path = Path(item["path"])
        if not path.is_file():
            raise GateError(f"Final-application preflight input is absent: {name}")
        if name == "r4_lp_ledger" and not preflight_only:
            if _ledger_prefix_sha_before_final(path) != item["sha256"]:
                raise GateError("R4 ledger no longer preserves the frozen D1/D2 prefix")
            r4_freeze = json.loads(r4.PREFIT_FREEZE.read_text(encoding="utf-8"))
            ledger = r4.read_ledger(path)
            _validate_prior_ledger(r4_freeze, allow_final=FIT_ID in ledger)
        else:
            if sha_file(path) != item["sha256"]:
                raise GateError(f"Final-application preflight input changed: {name}")
            if name == "r4_lp_ledger" and FIT_ID in r4.read_ledger(path):
                raise GateError("HINGE_FINAL ledger slot was modified before identity preflight")
    deferred = freeze.get("deferred_target_input", {})
    train_path, train_sha = _r4_preflight_input_pin(
        json.loads(r4.PREFIT_FREEZE.read_text(encoding="utf-8")), "r3_input__official_train")
    if (deferred != {"path": str(train_path.resolve()), "sha256": train_sha}
            or freeze.get("input_hashes", {}).get("official_train_deferred") != deferred):
        raise GateError("Deferred Train target source does not match the inherited R4 pin")
    if not preflight_only:
        if not PREFLIGHT.is_file():
            raise GateError("Final-application identity preflight is absent")
        preflight, preflight_sha = _read_preflight(freeze, freeze_sha)
        if preflight.get("status") != "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT" or not preflight_sha:
            raise GateError("Final-application identity preflight did not pass")
    return freeze, freeze_sha


def _api_contracts() -> dict[str, Any]:
    from scripts import verify_reg_monotone_hinge_p50_v4 as r4verifier
    expected = {
        "selected_train_row_reader": (read_scored_query,
            ["source_root", "selected_rows", "expected_ids", "expected_cycles", "expected_total_rows"]),
        "oof_identity_loader": (r4.load_oof_identity, ["freeze"]),
        "oof_identity_validator": (r4.verify_oof_identity, ["identity", "expected"]),
        "final_solver": (r4.solve_hinge_l1, ["p_values", "y_values", "purpose"]),
        "csv_writer": (r4.write_candidate_csv,
            ["path", "record_ids", "threat_rank", "missing_mask", "raw_direct"]),
        "independent_csv_verifier": (r4verifier.verify_candidate_csv,
            ["path", "expected_ids", "expected_values", "missing_mask", "raw_direct", "production"]),
        "base_identity_manifest": (v2.verify_base, []),
    }
    report: dict[str, Any] = {}
    for name, (function, names) in expected.items():
        signature = inspect.signature(function)
        actual = list(signature.parameters)
        if actual != names:
            raise GateError(f"Live helper API differs from frozen R1 contract: {name} {actual}")
        report[name] = {
            "module": function.__module__,
            "function": function.__name__,
            "parameters": actual,
            "source_path": str(Path(inspect.getsourcefile(function) or "").resolve()),
            "source_sha256": sha_file(Path(inspect.getsourcefile(function) or "")),
        }
    return report


def _load_e79_raw_direct(freeze: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray,
        np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read only the pinned E79 prediction/identity archive and signed metadata."""
    manifest_path = Path(freeze["input_hashes"]["e79_manifest"]["path"])
    qa_path = Path(freeze["input_hashes"]["e79_qa"]["path"])
    archive_path = Path(freeze["input_hashes"]["e79_archive"]["path"])
    for name, path in (("e79_manifest", manifest_path), ("e79_qa", qa_path),
                       ("e79_archive", archive_path)):
        item = freeze["input_hashes"][name]
        if not path.is_file() or sha_file(path) != item["sha256"]:
            raise GateError(f"Pinned E79 non-target input changed: {name}")
    manifest, qa, _direct_freeze = v2.verify_base()
    with np.load(archive_path, allow_pickle=False) as archive:
        members = set(archive.files)
        if not set(E79_NPZ_ALLOWLIST).issubset(members):
            raise GateError(f"Pinned E79 archive lacks required non-target members: {sorted(members)}")
        forbidden = sorted(name for name in members - set(E79_NPZ_ALLOWLIST)
                           if any(token in name.casefold() for token in E79_FORBIDDEN_NAME_TOKENS))
        if forbidden:
            raise GateError(f"Pinned E79 archive exposes target-like member names: {forbidden}")
        ids = np.asarray(archive["record_id"], dtype=str).copy()
        raw = np.asarray(archive["prediction"], dtype="<f8").copy()
        baseline = np.asarray(archive["baseline"], dtype="<f8").copy()
        missing = np.asarray(archive["missing"], dtype=bool).copy()
    if (ids.ndim != 1 or raw.ndim != 1 or baseline.ndim != 1 or missing.ndim != 1
            or not (len(ids) == len(raw) == len(baseline) == len(missing) == EXPECTED_TEST_ROWS)
            or len(np.unique(ids)) != EXPECTED_TEST_ROWS
            or not np.isfinite(raw).all() or not np.isfinite(baseline).all()
            or np.any((raw < 0.0) | (raw > 100.0))
            or int(missing.sum()) != EXPECTED_MISSING_ROWS
            or not np.array_equal(raw[missing].view("<u8"), baseline[missing].view("<u8"))):
        raise GateError("Pinned E79 raw DIRECT identity/prediction/missing route differs from contract")
    ids_sha = v2.sha_ordered_ids(ids)
    prediction_sha = v2.sha_f64_bytes(raw)
    if (manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or manifest.get("ordered_official_test_ids_sha256") != ids_sha
            or manifest.get("prediction_sha256") != prediction_sha
            or manifest.get("prediction_rows") != EXPECTED_TEST_ROWS
            or manifest.get("missing_rows") != EXPECTED_MISSING_ROWS
            or manifest.get("test_labels_loaded") is not False
            or manifest.get("upload_performed") is not False
            or qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or qa.get("test_labels_read") is not False
            or qa.get("missing_rows") != EXPECTED_MISSING_ROWS):
        raise GateError("Pinned E79 signed manifest/QA do not bind archive identity and boundary")
    audit = {
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": freeze["input_hashes"]["e79_archive"]["sha256"],
        "loaded_npz_members": list(E79_NPZ_ALLOWLIST),
        "unloaded_extra_member_names": sorted(members - set(E79_NPZ_ALLOWLIST)),
        "rows": len(ids),
        "unique_ids": int(len(np.unique(ids))),
        "ordered_ids_sha256": ids_sha,
        "raw_direct_sha256": r4.r1.tagged_array_sha256(raw, dtype="<f8"),
        "raw_direct_sha256_bytes": prediction_sha,
        "missing_rows": int(missing.sum()),
        "missing_route_bit_exact_to_baseline": True,
        "target_like_members_loaded": False,
        "csv_read": False,
        "test_or_sample_read": False,
    }
    return ids, raw, missing, manifest, qa, audit


def _validate_current_r4_and_ledger(freeze: Mapping[str, Any], *,
                                    allow_final: bool = False) -> tuple[
                                        dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    r4_freeze, _approval, pass_sha = _verify_r4_predecessor_with_all_pins()
    if pass_sha != R4_ROOT_SOURCE_PASS_SHA256:
        raise GateError("R4 diagnostic pass changed; it is lineage only, not final authorization")
    ledger = _validate_prior_ledger(r4_freeze, allow_final=allow_final)
    frozen = freeze["prior_slot_accounting"]
    if (frozen.get("completed_fit_ids") != list(PRIOR_DIAGNOSTIC_FIT_IDS)
            or frozen.get("completed_starts") != 2
            or frozen.get("next_fit_id") != FIT_ID
            or _ledger_prefix_sha_before_final(LEDGER)
                != frozen.get("ledger_sha256_before_final_slot")):
        raise GateError("Existing R4 ledger no longer matches frozen third-slot precondition")
    return r4_freeze, ledger


def _build_preflight_body(freeze: Mapping[str, Any], freeze_sha: str,
                          identity: Mapping[str, Any], identity_hashes: Mapping[str, Any],
                          e79_audit: Mapping[str, Any], manifest: Mapping[str, Any],
                          qa: Mapping[str, Any], api_contracts: Mapping[str, Any],
                          ledger: Mapping[str, Any], resources: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT",
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"],
        "prefit_freeze_sha256": freeze_sha,
        "r4_predecessor": freeze["r4_predecessor"],
        "prior_slot_accounting": {
            "ledger_sha256": freeze["prior_slot_accounting"]["ledger_sha256_before_final_slot"],
            "fit_ids": list(PRIOR_DIAGNOSTIC_FIT_IDS),
            "starts": 2,
            "complete_events": 2,
            "failed_events": 0,
            "hinge_final_reserved": FIT_ID in ledger,
            "next_fit_id": FIT_ID,
        },
        "oof_identity": {
            "row_identity_hashes": dict(identity_hashes),
            "row_counts": dict(identity["row_counts"]),
            "unique_rows": len(identity["rows"]),
            "selected_cycles": [68, 79],
            "targets_loaded": False,
        },
        "e79_raw_direct_audit": dict(e79_audit),
        "signed_base_manifest_contract": {
            "candidate_id": manifest.get("candidate_id"),
            "ordered_official_test_ids_sha256": manifest.get("ordered_official_test_ids_sha256"),
            "prediction_rows": manifest.get("prediction_rows"),
            "missing_rows": manifest.get("missing_rows"),
            "output_columns": list(OUTPUT_COLUMNS),
            "qa_status": qa.get("status"),
            "test_labels_read": qa.get("test_labels_read"),
        },
        "api_contracts": dict(api_contracts),
        "resource_snapshots": [dict(item) for item in resources],
        "target_reader_called": False,
        "train_targets_parsed": False,
        "train_target_source_hash_verified": True,
        "test_or_sample_reader_called": False,
        "test_or_sample_read": False,
        "lp_reserved": False,
        "official_lp_starts": 0,
        "ledger_created_or_modified": False,
        "csv_created": False,
        "kaggle_upload": False,
    }


def identity_preflight() -> dict[str, Any]:
    freeze, freeze_sha = verify_final_freeze(preflight_only=True)
    if ROOT_SOURCE_PASS.exists():
        raise GateError("New final-application source pass already exists; preflight is a pre-pass stage")
    if any(path.exists() for path in (RUN_DIR, OUTPUT_CSV, FINAL_REPORT,
                                      FINAL_VERIFICATION, FINAL_RECEIPT)):
        raise GateError("Final-application output already exists before preflight")
    resources: list[dict[str, Any]] = []
    with r4.SharedHeavyJobLock(SHARED_JOB_LOCK):
        resources.append(r4.resource_snapshot(boundary="final_application_preflight_locked_initial",
                                              initial=True))
        v2.one_job_gate()
        r4_freeze, _approval, r4_pass_sha = _verify_r4_predecessor_with_all_pins()
        if r4_pass_sha != R4_ROOT_SOURCE_PASS_SHA256:
            raise GateError("R4 predecessor source pass changed during final-application preflight")
        _lineage, ledger = _validate_current_r4_and_ledger(freeze)
        identity = r4.load_oof_identity(r4_freeze)
        identity_hashes = r4.verify_oof_identity(identity, r4_freeze["support_identity"])
        resources.append(r4.resource_snapshot(boundary="final_application_preflight_after_oof_identity"))
        ids, raw, missing, manifest, qa, e79_audit = _load_e79_raw_direct(freeze)
        if ids.shape != (EXPECTED_TEST_ROWS,) or raw.shape != ids.shape or missing.shape != ids.shape:
            raise GateError("E79 raw DIRECT non-target contract returned misaligned vectors")
        api_contracts = _api_contracts()
        resources.append(r4.resource_snapshot(boundary="final_application_preflight_after_e79_identity"))
        if source_hashes() != freeze.get("source_hashes"):
            raise GateError("Final-application sources changed during identity preflight")
        for name in PREFLIGHT_REHASH_KEYS:
            item = freeze["preflight_input_hashes"][name]
            if sha_file(Path(item["path"])) != item["sha256"]:
                raise GateError(f"Final-application preflight input changed during read: {name}")
        body = _build_preflight_body(freeze, freeze_sha, identity, identity_hashes,
            e79_audit, manifest, qa, api_contracts, ledger, resources)
        body["signature"] = canonical_sha(body)
        r4.write_json_once(PREFLIGHT, body)
        report_sha = sha_file(PREFLIGHT)
        return {"status": body["status"], "identity_preflight_sha256": report_sha,
            "oof_rows": body["oof_identity"]["unique_rows"],
            "e79_rows": body["e79_raw_direct_audit"]["rows"],
            "missing_rows": body["e79_raw_direct_audit"]["missing_rows"],
            "target_reader_called": False, "official_lp_starts": 0,
            "csv_created": False}


def _read_preflight(freeze: Mapping[str, Any], freeze_sha: str) -> tuple[dict[str, Any], str]:
    if not PREFLIGHT.is_file():
        raise GateError("Final-application identity preflight is absent")
    report_sha = sha_file(PREFLIGHT)
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    body = {key: value for key, value in report.items() if key != "signature"}
    if (report.get("signature") != canonical_sha(body)
            or report.get("status") != "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT"
            or report.get("experiment_id") != EXPERIMENT_ID
            or report.get("prefit_freeze_sha256") != freeze_sha
            or report.get("source_hashes") != freeze.get("source_hashes")
            or report.get("input_hashes") != freeze.get("input_hashes")
            or report.get("official_lp_starts") != 0
            or report.get("target_reader_called") is not False
            or report.get("train_targets_parsed") is not False
            or report.get("train_target_source_hash_verified") is not True
            or report.get("test_or_sample_reader_called") is not False
            or report.get("test_or_sample_read") is not False
            or report.get("lp_reserved") is not False
            or report.get("ledger_created_or_modified") is not False
            or report.get("csv_created") is not False
            or report.get("kaggle_upload") is not False):
        raise GateError("Final-application preflight signature/status/boundary failed")
    if (report.get("prior_slot_accounting", {}).get("ledger_sha256")
            != freeze.get("prior_slot_accounting", {}).get("ledger_sha256_before_final_slot")
            or report.get("prior_slot_accounting", {}).get("starts") != 2
            or report.get("prior_slot_accounting", {}).get("failed_events") != 0
            or report.get("prior_slot_accounting", {}).get("hinge_final_reserved") is not False
            or report.get("prior_slot_accounting", {}).get("next_fit_id") != FIT_ID):
        raise GateError("Final-application preflight slot/ledger boundary differs from freeze")
    identity = report.get("oof_identity", {})
    if (identity.get("unique_rows") != EXPECTED_OOF_ROWS
            or identity.get("row_counts") != {
                "unique_oof": EXPECTED_OOF_ROWS,
                "E67_68_75": 12_412,
                "E75_76_79": 8_491}
            or identity.get("targets_loaded") is not False):
        raise GateError("Final-application preflight OOF identity counts differ from fixed scope")
    e79 = report.get("e79_raw_direct_audit", {})
    if (e79.get("rows") != EXPECTED_TEST_ROWS
            or e79.get("unique_ids") != EXPECTED_TEST_ROWS
            or e79.get("missing_rows") != EXPECTED_MISSING_ROWS
            or e79.get("missing_route_bit_exact_to_baseline") is not True
            or e79.get("target_like_members_loaded") is not False
            or e79.get("csv_read") is not False
            or e79.get("test_or_sample_read") is not False):
        raise GateError("Final-application preflight E79 identity/prediction contract failed")
    if set(report.get("api_contracts", {})) != {
            "selected_train_row_reader", "oof_identity_loader", "oof_identity_validator",
            "final_solver", "csv_writer", "independent_csv_verifier", "base_identity_manifest"}:
        raise GateError("Final-application preflight omitted a production helper API contract")
    snapshots = report.get("resource_snapshots", [])
    if [item.get("boundary") for item in snapshots] != [
            "final_application_preflight_locked_initial",
            "final_application_preflight_after_oof_identity",
            "final_application_preflight_after_e79_identity"]:
        raise GateError("Final-application preflight resource boundary sequence differs")
    return report, report_sha


def validate_final_source_pass_body(approval: Mapping[str, Any], freeze: Mapping[str, Any],
                                    freeze_sha: str, preflight_sha: str) -> None:
    body = {key: value for key, value in approval.items() if key != "signature"}
    if approval.get("signature") != canonical_sha(body):
        raise GateError("New final-application root source pass signature failed")
    if (approval.get("verdict") != "PASS"
            or approval.get("pass_kind") != "FINAL_APPLICATION"
            or approval.get("experiment_id") != EXPERIMENT_ID
            or approval.get("source_commit") != freeze.get("source_commit")
            or approval.get("source_hashes") != freeze.get("source_hashes")
            or approval.get("input_hashes") != freeze.get("input_hashes")
            or approval.get("prefit_freeze_sha256") != freeze_sha
            or approval.get("identity_preflight_sha256") != preflight_sha
            or approval.get("r4_diagnostic_source_pass_sha256") != R4_ROOT_SOURCE_PASS_SHA256
            or approval.get("r4_diagnostics_sha256") != R4_DIAGNOSTICS_SHA256
            or approval.get("authorized_fit_ids") != [FIT_ID]
            or approval.get("authorized_final_lp_starts") != 1
            or approval.get("prior_official_lp_starts") != 2
            or approval.get("total_official_lp_start_cap") != TOTAL_LP_CAP
            or approval.get("authorized_csv_creation") is not True
            or approval.get("test_or_sample_access_authorized") is not False
            or approval.get("kaggle_upload_authorized") is not False):
        raise GateError("New source pass does not bind the isolated HINGE_FINAL application")


def verify_final_source_pass() -> tuple[dict[str, Any], dict[str, Any], str]:
    freeze, freeze_sha = verify_final_freeze()
    preflight, preflight_sha = _read_preflight(freeze, freeze_sha)
    if not ROOT_SOURCE_PASS.is_file():
        raise GateError("New final-application root source pass is absent")
    approval = json.loads(ROOT_SOURCE_PASS.read_text(encoding="utf-8"))
    validate_final_source_pass_body(approval, freeze, freeze_sha, preflight_sha)
    return freeze, approval, sha_file(ROOT_SOURCE_PASS)


def _current_final_pass_sha(expected_sha: str) -> str:
    _freeze, _approval, current = verify_final_source_pass()
    if current != expected_sha:
        raise GateError("New final-application source pass changed during execution")
    return current


def _identity_for_final_application(r4_freeze: Mapping[str, Any]) -> dict[str, Any]:
    identity = r4.load_oof_identity(r4_freeze)
    r4.verify_oof_identity(identity, r4_freeze["support_identity"])
    if len(identity["rows"]) != EXPECTED_OOF_ROWS:
        raise GateError("Final HINGE support must contain exactly 20,903 frozen OOF rows")
    return identity


def _load_train_targets_after_new_pass(identity: Mapping[str, Any],
                                       approval_sha: str,
                                       freeze: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    _current_final_pass_sha(approval_sha)
    r4_freeze, _approval, r4_pass_sha = r4.verify_source_pass()
    if r4_pass_sha != R4_ROOT_SOURCE_PASS_SHA256:
        raise GateError("R4 diagnostic source pass changed before final Train target access")
    r4.verify_oof_identity(identity, r4_freeze["support_identity"])
    target_pin = freeze["deferred_target_input"]
    train_path = Path(target_pin["path"])
    if not train_path.is_file() or sha_file(train_path) != target_pin["sha256"]:
        raise GateError("Frozen Train target source changed after new final source pass")
    labels, _text_load, parsed_rows, audit = read_scored_query(
        v2.SOURCE, np.asarray(identity["rows"], dtype=np.int64),
        np.asarray(identity["ids"], dtype=str), np.asarray(identity["cycles"], dtype=np.int64),
        expected_total_rows=50_608)
    y = np.asarray(labels, dtype="<f8")
    if (not np.array_equal(parsed_rows, identity["rows"])
            or y.shape != (EXPECTED_OOF_ROWS,)
            or not np.all(np.isfinite(y)) or np.any((y < 0.0) | (y > 100.0))
            or audit.get("fit_target_rows_parsed") != 0
            or audit.get("sha256") != target_pin["sha256"]):
        raise GateError("Selected Train target boundary differs from frozen 20,903-row OOF identity")
    return y, audit


def _fit_arrays(identity: Mapping[str, Any], targets: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {
        "physical_rows": np.array(identity["rows"], dtype=np.int64, copy=True),
        "record_ids": np.array(identity["ids"], dtype=str, copy=True),
        "prediction_origins": np.array(identity["origins"], dtype=np.int8, copy=True),
        "original_bundle_indices": np.array(identity["indices"], dtype=np.int64, copy=True),
        "cycles": np.array(identity["cycles"], dtype=np.int16, copy=True),
        "direct_predictions": np.array(identity["direct"], dtype="<f8", copy=True),
        "missing_mask": np.array(identity["missing"], dtype=bool, copy=True),
        "targets": np.array(targets, dtype="<f8", copy=True),
    }
    if any(array.ndim != 1 for array in arrays.values()) or len({len(x) for x in arrays.values()}) != 1:
        raise GateError("Final HINGE fit inputs must be aligned one-dimensional arrays")
    return arrays


def _fit_input_identity(arrays: Mapping[str, np.ndarray], bundle_sha: str,
                         freeze: Mapping[str, Any], freeze_sha: str,
                         approval_sha: str) -> dict[str, Any]:
    digest_keys = {
        "physical_rows": "physical_rows_sha256",
        "record_ids": "record_ids_sha256",
        "prediction_origins": "prediction_origins_sha256",
        "original_bundle_indices": "original_bundle_indices_sha256",
        "cycles": "cycles_sha256",
        "missing_mask": "missing_mask_sha256",
    }
    hashes = {out: r4.r1.tagged_array_sha256(arrays[key]) for key, out in digest_keys.items()}
    hashes["direct_prediction_sha256"] = r4.r1.tagged_array_sha256(
        arrays["direct_predictions"], dtype="<f8")
    hashes["target_sha256"] = r4.r1.tagged_array_sha256(arrays["targets"], dtype="<f8")
    hashes["direct_raw_f64_sha256"] = r4.r1.raw_direct_f64_sha256(arrays["direct_predictions"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "fit_id": FIT_ID,
        "fit_rows": len(arrays["targets"]),
        "row_hashes": hashes,
        "fit_input_bundle_sha256": bundle_sha,
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"],
        "prefit_freeze_sha256": freeze_sha,
        "root_source_pass_sha256": approval_sha,
    }


def _save_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    return r4.write_npz_once(path, arrays)


def _save_json_once(path: Path, body: Mapping[str, Any]) -> str:
    return r4.write_json_once(path, dict(body))


def _ledger_stage(ledger: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return ledger.get(FIT_ID, {kind: [] for kind in ("RESERVE", "LP_STARTED", "LP_COMPLETE", "LP_FAILED")})


def _validate_saved_final_state(fit_path: Path, slack_path: Path,
                                expected: Mapping[str, Any]) -> dict[str, Any]:
    if not fit_path.is_file():
        raise GateError("HINGE_FINAL start has no saved fit state; consumed slot cannot retry")
    state = json.loads(fit_path.read_text(encoding="utf-8"))
    if (state.get("fit_id") != FIT_ID or state.get("fit_input") != dict(expected)
            or state.get("root_source_pass_sha256") != expected.get("root_source_pass_sha256")):
        raise GateError("Saved HINGE_FINAL state differs from the immutable final fit input")
    if state.get("status") == "LP_COMPLETE":
        if not slack_path.is_file() or sha_file(slack_path) != state.get("solver_slacks_sha256"):
            raise GateError("Saved HINGE_FINAL lacks its immutable solver slack artifact")
    elif state.get("status") != "LP_FAILED_TERMINAL":
        raise GateError("Saved HINGE_FINAL state has an unknown terminal status")
    return state


def _run_final_fit_slot(arrays: Mapping[str, np.ndarray], expected: Mapping[str, Any],
                        approval_sha: str, freeze: Mapping[str, Any],
                        resource_check: Any) -> dict[str, Any]:
    fit_path = RUN_DIR / "fits" / f"{FIT_ID}.json"
    input_path = RUN_DIR / "fit_inputs" / f"{FIT_ID}.npz"
    slack_path = RUN_DIR / "solver_slacks" / f"{FIT_ID}.npz"
    bundle_sha = _save_npz_once(input_path, arrays)
    if bundle_sha != expected.get("fit_input_bundle_sha256"):
        raise GateError("HINGE_FINAL fit input bundle hash differs from its identity")
    _current_final_pass_sha(approval_sha)
    ledger = r4.read_ledger(LEDGER)
    _validate_prior_ledger(json.loads(r4.PREFIT_FREEZE.read_text(encoding="utf-8")), allow_final=True)
    stage = _ledger_stage(ledger)
    reserve = stage["RESERVE"][0] if stage["RESERVE"] else None
    started = stage["LP_STARTED"][0] if stage["LP_STARTED"] else None
    complete = stage["LP_COMPLETE"][0] if stage["LP_COMPLETE"] else None
    failed = stage["LP_FAILED"][0] if stage["LP_FAILED"] else None
    if failed:
        raise GateError("HINGE_FINAL already failed terminally; no retry")
    if complete or (started and fit_path.exists()):
        state = _validate_saved_final_state(fit_path, slack_path, expected)
        state_sha = sha_file(fit_path)
        if state.get("status") != "LP_COMPLETE":
            if not complete:
                r4.append_ledger(LEDGER, {
                    "event": "LP_FAILED", "fit_id": FIT_ID,
                    "reservation_id": reserve["reservation_id"], "fit_input": dict(expected),
                    "source_commit": freeze["source_commit"],
                    "root_source_pass_sha256": approval_sha, "state_sha256": state_sha,
                    "recovered_after_state_publish": True,
                    "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
            raise GateError("HINGE_FINAL saved as failed; consumed slot cannot retry")
        if complete:
            if complete[0].get("state_sha256") != state_sha:
                raise GateError("HINGE_FINAL terminal ledger hash differs from saved state")
        else:
            r4.append_ledger(LEDGER, {
                "event": "LP_COMPLETE", "fit_id": FIT_ID,
                "reservation_id": reserve["reservation_id"], "fit_input": dict(expected),
                "source_commit": freeze["source_commit"],
                "root_source_pass_sha256": approval_sha, "state_sha256": state_sha,
                "recovered_after_state_publish": True,
                "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
        return state
    if started:
        raise GateError("HINGE_FINAL has LP_STARTED without a fit state; ambiguous consumed slot")
    if complete or failed:
        raise GateError("HINGE_FINAL terminal ledger event lacks a recoverable state")
    if reserve:
        if (reserve.get("fit_input") != dict(expected)
                or reserve.get("source_commit") != freeze["source_commit"]
                or reserve.get("root_source_pass_sha256") != approval_sha
                or reserve.get("retry_allowed") is not False
                or reserve.get("attempt") != 1):
            raise GateError("Existing HINGE_FINAL reservation differs; no retry")
    else:
        if r4._next_fit_id(ledger) != FIT_ID:
            raise GateError("Only the pre-existing third HINGE_FINAL slot is available")
        if sha_file(LEDGER) != freeze["prior_slot_accounting"]["ledger_sha256_before_final_slot"]:
            raise GateError("R4 ledger changed after final-application freeze; no slot reservation")
        reservation_id = str(uuid.uuid4())
        reserve_resource = dict(resource_check(boundary="HINGE_FINAL_reserve", initial=True))
        r4.append_ledger(LEDGER, {
            "event": "RESERVE", "fit_id": FIT_ID, "reservation_id": reservation_id,
            "fit_input": dict(expected), "source_commit": freeze["source_commit"],
            "root_source_pass_sha256": approval_sha, "resource": reserve_resource,
            "attempt": 1, "retry_allowed": False,
            "at_utc": datetime.now(timezone.utc).isoformat()})
        ledger = r4.read_ledger(LEDGER)
        reserve = _ledger_stage(ledger)["RESERVE"][0]
    _current_final_pass_sha(approval_sha)
    start_resource = dict(resource_check(boundary="HINGE_FINAL_lp_start", initial=True))
    v2.one_job_gate()
    r4.append_ledger(LEDGER, {
        "event": "LP_STARTED", "fit_id": FIT_ID,
        "reservation_id": reserve["reservation_id"], "fit_input": dict(expected),
        "source_commit": freeze["source_commit"],
        "root_source_pass_sha256": approval_sha, "resource": start_resource,
        "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
    started_at = time.perf_counter()
    try:
        solver = r4.solve_hinge_l1(arrays["direct_predictions"], arrays["targets"], purpose="final")
        slacks = np.asarray(solver.pop("_slacks"), dtype="<f8")
        if (solver.get("success") is not True or slacks.shape != arrays["targets"].shape
                or not np.all(np.isfinite(slacks)) or np.any(slacks < -1e-8)):
            raise GateError("HINGE_FINAL solver or primal slack validation failed")
        prediction = (solver["intercept"] + solver["slope_low"] * np.minimum(
            arrays["direct_predictions"], KNOT) + solver["slope_high"] * np.maximum(
            arrays["direct_predictions"] - KNOT, 0.0))
        if np.max(np.abs(arrays["targets"] - prediction) - slacks) > 1e-7:
            raise GateError("HINGE_FINAL primal slack inequality failed")
        slack_sha = _save_npz_once(slack_path, {"slacks": slacks})
        status, failure = "LP_COMPLETE", None
    except Exception as exc:
        solver = {"success": False, "error_type": type(exc).__name__, "error": str(exc)}
        slack_sha = None
        status, failure = "LP_FAILED_TERMINAL", {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.perf_counter() - started_at
    after_resource = dict(resource_check(boundary="HINGE_FINAL_after_solver", initial=False)) \
        if status == "LP_COMPLETE" else None
    state = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "fit_id": FIT_ID,
        "status": status, "fit_input": dict(expected), "solver_result": solver,
        "solver_slacks_path": str(slack_path.resolve()) if slack_sha else None,
        "solver_slacks_sha256": slack_sha,
        "resources_after_solver": after_resource,
        "solver_wall_seconds": float(elapsed), "failure": failure,
        "root_source_pass_sha256": approval_sha,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_sha = _save_json_once(fit_path, state)
    event = "LP_COMPLETE" if status == "LP_COMPLETE" else "LP_FAILED"
    r4.append_ledger(LEDGER, {
        "event": event, "fit_id": FIT_ID,
        "reservation_id": reserve["reservation_id"], "fit_input": dict(expected),
        "source_commit": freeze["source_commit"],
        "root_source_pass_sha256": approval_sha, "state_sha256": state_sha,
        "recovered_after_state_publish": False,
        "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
    if status != "LP_COMPLETE":
        raise GateError("HINGE_FINAL failed terminally; third slot consumed and no retry is allowed")
    return state


def _existing_final_fit_arrays(expected: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    path = RUN_DIR / "fit_inputs" / f"{FIT_ID}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        names = ("physical_rows", "record_ids", "prediction_origins", "original_bundle_indices",
                 "cycles", "direct_predictions", "missing_mask", "targets")
        if not set(names).issubset(archive.files):
            raise GateError("Saved HINGE_FINAL fit input bundle is incomplete")
        arrays = {name: np.asarray(archive[name]).copy() for name in names}
    if r4.file_sha256(path) != expected.get("fit_input_bundle_sha256"):
        raise GateError("Saved HINGE_FINAL fit input bundle SHA differs; no retry")
    return arrays


def _validate_saved_fit_against_identity(identity: Mapping[str, Any],
                                         arrays: Mapping[str, np.ndarray],
                                         expected: Mapping[str, Any],
                                         freeze: Mapping[str, Any], freeze_sha: str,
                                         approval_sha: str) -> None:
    """Check the exact saved bundle digest and its non-target OOF row identity."""
    computed = _fit_input_identity(
        arrays, str(expected.get("fit_input_bundle_sha256", "")),
        freeze, freeze_sha, approval_sha)
    if computed != dict(expected):
        raise GateError("Saved HINGE_FINAL bundle identity differs from its reservation")
    comparisons = {
        "physical_rows": np.asarray(identity["rows"], dtype=np.int64),
        "record_ids": np.asarray(identity["ids"], dtype=str),
        "prediction_origins": np.asarray(identity["origins"], dtype=np.int8),
        "original_bundle_indices": np.asarray(identity["indices"], dtype=np.int64),
        "cycles": np.asarray(identity["cycles"], dtype=np.int16),
        "direct_predictions": np.asarray(identity["direct"], dtype="<f8"),
        "missing_mask": np.asarray(identity["missing"], dtype=bool),
    }
    if any(not np.array_equal(arrays[name], expected_values)
           for name, expected_values in comparisons.items()):
        raise GateError("Saved HINGE_FINAL non-target rows differ from frozen OOF identity")


def _apply_hinge(raw: np.ndarray, missing: np.ndarray,
                 params: Mapping[str, float]) -> np.ndarray:
    raw = np.asarray(raw, dtype="<f8")
    missing = np.asarray(missing, dtype=bool)
    if (raw.shape != missing.shape or raw.ndim != 1 or not np.isfinite(raw).all()
            or np.any((raw < 0.0) | (raw > 100.0))):
        raise GateError("Pinned E79 raw DIRECT vector is invalid")
    out = raw.copy()
    present = ~missing
    transformed = (float(params["intercept"])
        + float(params["slope_low"]) * np.minimum(raw[present], KNOT)
        + float(params["slope_high"]) * np.maximum(raw[present] - KNOT, 0.0))
    out[present] = np.clip(transformed, 0.0, 100.0)
    if not np.array_equal(out[missing].view("<u8"), raw[missing].view("<u8")):
        raise GateError("Final hinge application changed raw DIRECT missing bits")
    return out


def _write_application_manifest(path: Path, body: Mapping[str, Any]) -> str:
    manifest = dict(body)
    manifest["signature"] = canonical_sha(manifest)
    return _save_json_once(path, manifest)


def run_final_application() -> dict[str, Any]:
    freeze, approval, approval_sha = verify_final_source_pass()
    freeze_sha = sha_file(FREEZE)
    preflight, preflight_sha = _read_preflight(freeze, freeze_sha)
    if (OUTPUT_CSV.exists() and not (RUN_DIR / "fits" / f"{FIT_ID}.json").exists()):
        raise GateError("Final application CSV exists without a recoverable HINGE_FINAL state")
    resources: list[dict[str, Any]] = []
    with r4.SharedHeavyJobLock(SHARED_JOB_LOCK):
        resources.append(r4.resource_snapshot(boundary="final_application_locked_initial", initial=True))
        v2.one_job_gate()
        _r4_freeze, prior_ledger = _validate_current_r4_and_ledger(freeze,
            allow_final=FIT_ID in r4.read_ledger(LEDGER))
        identity = _identity_for_final_application(
            json.loads(r4.PREFIT_FREEZE.read_text(encoding="utf-8")))
        resources.append(r4.resource_snapshot(boundary="final_application_after_oof_identity"))
        ids, raw_direct, missing, manifest, qa, e79_audit = _load_e79_raw_direct(freeze)
        if (manifest.get("ordered_official_test_ids_sha256")
                != preflight.get("e79_raw_direct_audit", {}).get("ordered_ids_sha256")
                or e79_audit.get("archive_sha256")
                    != preflight.get("e79_raw_direct_audit", {}).get("archive_sha256")):
            raise GateError("Final E79 identity differs from preflight")
        resources.append(r4.resource_snapshot(boundary="final_application_after_e79_identity"))
        stage = _ledger_stage(prior_ledger)
        if stage["LP_STARTED"]:
            # A started or completed slot must recover from the saved target bundle;
            # never call the Train reader again and never invoke a second LP.
            expected_hint = stage["RESERVE"][0].get("fit_input", {})
            arrays = _existing_final_fit_arrays(expected_hint)
            if arrays is None:
                raise GateError("HINGE_FINAL start has no saved input bundle; ambiguous slot, no retry")
            expected = expected_hint
            _validate_saved_fit_against_identity(
                identity, arrays, expected, freeze, freeze_sha, approval_sha)
            target_audit = {"recovered_from_saved_fit_input": True,
                            "query_target_rows_parsed": EXPECTED_OOF_ROWS,
                            "fit_target_rows_parsed": 0,
                            "sha256": freeze["deferred_target_input"]["sha256"]}
            state = _run_final_fit_slot(arrays, expected, approval_sha, freeze,
                                         lambda **kw: r4.resource_snapshot(**kw))
        elif stage["RESERVE"]:
            # A pre-start reservation is recoverable only from its exact immutable
            # fit-input bundle. Do not re-read Train or replace the reservation.
            expected_hint = stage["RESERVE"][0].get("fit_input", {})
            arrays = _existing_final_fit_arrays(expected_hint)
            if arrays is None:
                raise GateError("Reserved HINGE_FINAL has no saved exact input bundle; no retry")
            expected = expected_hint
            if (expected.get("root_source_pass_sha256") != approval_sha
                    or expected.get("prefit_freeze_sha256") != freeze_sha):
                raise GateError("Reserved HINGE_FINAL input is not bound to this source pass/freeze")
            _validate_saved_fit_against_identity(
                identity, arrays, expected, freeze, freeze_sha, approval_sha)
            recovered_targets, target_audit = _load_train_targets_after_new_pass(
                identity, approval_sha, freeze)
            if not np.array_equal(
                    arrays["targets"].view("<u8"),
                    np.asarray(recovered_targets, dtype="<f8").view("<u8")):
                raise GateError("Reserved HINGE_FINAL target bundle differs from pinned Train query")
            target_audit = {**target_audit, "recovered_from_saved_fit_input": True}
            state = _run_final_fit_slot(arrays, expected, approval_sha, freeze,
                                         lambda **kw: r4.resource_snapshot(**kw))
        else:
            targets, target_audit = _load_train_targets_after_new_pass(
                identity, approval_sha, freeze)
            resources.append(r4.resource_snapshot(boundary="final_application_after_selected_train_targets"))
            arrays = _fit_arrays(identity, targets)
            input_path = RUN_DIR / "fit_inputs" / f"{FIT_ID}.npz"
            bundle_sha = _save_npz_once(input_path, arrays)
            expected = _fit_input_identity(arrays, bundle_sha, freeze, freeze_sha, approval_sha)
            _current_final_pass_sha(approval_sha)
            state = _run_final_fit_slot(arrays, expected, approval_sha, freeze,
                                         lambda **kw: r4.resource_snapshot(**kw))
        params = {key: float(state["solver_result"][key])
                  for key in ("intercept", "slope_low", "slope_high")}
        output_values = _apply_hinge(raw_direct, missing, params)
        _current_final_pass_sha(approval_sha)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        if not OUTPUT_CSV.exists():
            r4.write_candidate_csv(OUTPUT_CSV, ids, output_values,
                                   missing_mask=missing, raw_direct=raw_direct)
        from scripts.verify_reg_monotone_hinge_p50_final_application_v1 import verify_final_application
        report_body = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "candidate_id": "HINGE_P50_FINAL_EXPLORATORY_E79",
            "status": "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION",
            "scientific_qualification": "FAIL",
            "validation_claim": None,
            "source_commit": freeze["source_commit"],
            "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"],
            "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "new_root_source_pass_sha256": approval_sha,
            "r4_diagnostic_pass_sha256": R4_ROOT_SOURCE_PASS_SHA256,
            "r4_diagnostics_sha256": R4_DIAGNOSTICS_SHA256,
            "r4_diagnostics_verification_sha256": R4_VERIFICATION_SHA256,
            "prior_slot_accounting": freeze["prior_slot_accounting"],
            "final_slot": {"fit_id": FIT_ID, "fit_rows": len(arrays["targets"]),
                "fit_state_path": str((RUN_DIR / "fits" / f"{FIT_ID}.json").resolve()),
                "fit_state_sha256": sha_file(RUN_DIR / "fits" / f"{FIT_ID}.json"),
                "fit_input_path": str((RUN_DIR / "fit_inputs" / f"{FIT_ID}.npz").resolve()),
                "fit_input_sha256": expected["fit_input_bundle_sha256"],
                "solver_slacks_path": state["solver_slacks_path"],
                "solver_slacks_sha256": state["solver_slacks_sha256"],
                "coefficients": params,
                "objective_unweighted_sum_absolute": state["solver_result"]["objective_unweighted_sum_absolute"],
                "solver_wall_seconds": state["solver_wall_seconds"]},
            "train_target_audit": target_audit,
            "e79_raw_direct_audit": e79_audit,
            "csv_path": str(OUTPUT_CSV.resolve()),
            "csv_sha256": sha_file(OUTPUT_CSV),
            "csv_rows": len(ids), "csv_columns": list(OUTPUT_COLUMNS),
            "ordered_ids_sha256": v2.sha_ordered_ids(ids),
            "prediction_sha256": r4.r1.tagged_array_sha256(output_values, dtype="<f8"),
            "missing_rows": int(missing.sum()),
            "missing_fallback_bit_exact": bool(np.array_equal(
                output_values[missing].view("<u8"), raw_direct[missing].view("<u8"))),
            "diagnostic_decision_context": freeze["decision_context"],
            "benchmark_batch_selection_complete": False,
            "test_or_sample_read": False,
            "test_labels_read": False,
            "kaggle_upload": False,
            "r4_d1_d2_refit": False,
            "global_control_refit": False,
            "new_base_model_fits": 0,
        }
        _write_application_manifest(FINAL_REPORT, report_body)
        verification = verify_final_application(report_dir=REPORT_DIR, freeze=freeze,
            approval_sha=approval_sha, report_path=FINAL_REPORT)
        _save_json_once(FINAL_VERIFICATION, verification)
        receipt_body = {
            "schema_version": 1, "experiment_id": EXPERIMENT_ID,
            "status": "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY",
            "final_report_sha256": sha_file(FINAL_REPORT),
            "verification_sha256": sha_file(FINAL_VERIFICATION),
            "csv_sha256": sha_file(OUTPUT_CSV),
            "fit_state_sha256": report_body["final_slot"]["fit_state_sha256"],
            "official_lp_starts_total": 3,
            "candidate_admission": "EXPLORATORY_ONLY_PENDING_BATCH_SELECTION",
            "kaggle_upload": False,
        }
        _save_json_once(FINAL_RECEIPT, receipt_body)
        return {"status": report_body["status"],
                "verification_status": verification["status"],
                "report_sha256": sha_file(FINAL_REPORT),
                "verification_sha256": sha_file(FINAL_VERIFICATION),
                "csv_sha256": sha_file(OUTPUT_CSV),
                "official_lp_starts_total": 3,
                "kaggle_upload": False}


def verify_final_application() -> dict[str, Any]:
    freeze, _approval, approval_sha = verify_final_source_pass()
    from scripts.verify_reg_monotone_hinge_p50_final_application_v1 import verify_final_application as verifier
    result = verifier(report_dir=REPORT_DIR, freeze=freeze, approval_sha=approval_sha,
                      report_path=FINAL_REPORT)
    _save_json_once(FINAL_VERIFICATION, result)
    return {"status": result["status"], "verification_sha256": sha_file(FINAL_VERIFICATION)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-final-application", action="store_true")
    actions.add_argument("--identity-preflight", action="store_true")
    actions.add_argument("--synthetic-qa", action="store_true")
    actions.add_argument("--run-final-application", action="store_true")
    actions.add_argument("--verify-final-application", action="store_true")
    args = parser.parse_args()
    if args.freeze_final_application:
        path, digest = freeze_final_application()
        print(json.dumps({"status": "FINAL_APPLICATION_FROZEN_ONLY",
                          "path": str(path), "sha256": digest}, sort_keys=True))
        return 0
    if args.identity_preflight:
        print(json.dumps(identity_preflight(), sort_keys=True))
        return 0
    if args.synthetic_qa:
        from scripts.verify_reg_monotone_hinge_p50_final_application_v1 import write_synthetic_qa
        print(json.dumps(write_synthetic_qa(), sort_keys=True))
        return 0
    if args.run_final_application:
        print(json.dumps(run_final_application(), sort_keys=True))
        return 0
    print(json.dumps(verify_final_application(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
