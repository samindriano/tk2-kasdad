"""Source-frozen present-only R06-native p50 hinge population ablation.

Freeze and preflight are outcome-blind and never load target arrays. Official
LPs require a separate immutable ROOT_SOURCE_PASS_R1.json, created by the root
reviewer. The prior R4 and all-row native R2C ledgers and fit artifacts are
read-only inputs. This source package never authorizes or starts FINAL.
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
import tempfile
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import reg_monotone_hinge_p50_v4 as r4  # noqa: E402

EXPERIMENT_ID = "REG_R06_PRESENT_NATIVE_HINGE_V1_R1"
EXPERIMENT_DIR = ROOT / "experiments/reg_r06_present_native_hinge_v1"
REPORT_DIR = ROOT / "reports/optimization_v3/reg_r06_present_native_hinge_v1"
RUN_DIR = ROOT / "experiments/runs/reg_r06_present_native_hinge_v1/R1"
FREEZE = EXPERIMENT_DIR / "PREFIT_FREEZE_R1.json"
ROOT_SOURCE_PASS = EXPERIMENT_DIR / "ROOT_SOURCE_PASS_R1.json"  # root-owned; never authored here
ROOT_FINAL_APPROVAL = EXPERIMENT_DIR / "ROOT_FINAL_FIT_APPROVAL_R1.json"  # root-owned
PREFLIGHT = REPORT_DIR / "IDENTITY_PREFLIGHT_R1.json"
FIT_POPULATION_IDS = REPORT_DIR / "FIT_POPULATION_IDS_R1.json"
SYNTHETIC_QA = REPORT_DIR / "SYNTHETIC_QA_R1.json"
DIAGNOSTICS = REPORT_DIR / "DIAGNOSTICS_R1.json"
VERIFICATION = REPORT_DIR / "SOURCE_VERIFICATION_R1.json"
PROTOCOL = REPORT_DIR / "PROTOCOL_R1.md"
HANDOFF = REPORT_DIR / "SOL_HANDOFF_R1.md"
LEDGER = RUN_DIR / "lp_ledger.jsonl"
FIT_IDS = ("R06_PRESENT_HINGE_D1", "R06_PRESENT_HINGE_D2", "R06_PRESENT_HINGE_FINAL")
DIAGNOSTIC_IDS = FIT_IDS[:2]
EXPECTED_ROWS = 20_903
EXPECTED_SCORE_D1 = 7_462
EXPECTED_SCORE_D2 = 8_491
EXPECTED_POOLED = EXPECTED_SCORE_D1 + EXPECTED_SCORE_D2
EXPECTED_PRESENT_FIT_D1 = 4_608
EXPECTED_PRESENT_FIT_D2 = 11_563
EXPECTED_MISSING_FIT_D1 = 342
EXPECTED_MISSING_FIT_D2 = 849
EXPECTED_TRAIN_SHA = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
TP2_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
TARGET_FREE_INVENTORY = ROOT / "reports/optimization_v3/rnbr_target_median_v1/targetfree_preflight.json"
LABEL_BUNDLE = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz"
R4_D1_STATE = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/R4/fits/HINGE_D1.json"
R4_D2_STATE = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/R4/fits/HINGE_D2.json"
R4_D1_INPUT = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/R4/fit_inputs/HINGE_D1.npz"
R4_D2_INPUT = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/R4/fit_inputs/HINGE_D2.npz"
R4_ROOT_PASS = ROOT / "experiments/reg_monotone_hinge_p50_v1/ROOT_SOURCE_PASS_R4.json"
R4_DIAGNOSTICS = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/DIAGNOSTICS_R4.json"
R4_DIAGNOSTICS_VERIFICATION = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/DIAGNOSTICS_VERIFICATION_R4.json"
R4_LEDGER = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/R4/lp_ledger.jsonl"
R2C_EXPERIMENT = ROOT / "experiments/reg_r06_native_hinge_v1"
R2C_ROOT_PASS = R2C_EXPERIMENT / "ROOT_SOURCE_PASS_R2C.json"
R2C_DIAGNOSTICS = ROOT / "reports/optimization_v3/reg_r06_native_hinge_v1/DIAGNOSTICS_R2C.json"
R2C_VERIFICATION = ROOT / "reports/optimization_v3/reg_r06_native_hinge_v1/DIAGNOSTICS_VERIFICATION_R2C.json"
R2C_LEDGER = ROOT / "experiments/runs/reg_r06_native_hinge_v1/R2C/lp_ledger.jsonl"
R2C_D1_STATE = ROOT / "experiments/runs/reg_r06_native_hinge_v1/R2C/fits/R06_HINGE_D1.json"
R2C_D2_STATE = ROOT / "experiments/runs/reg_r06_native_hinge_v1/R2C/fits/R06_HINGE_D2.json"
R2C_D1_INPUT = ROOT / "experiments/runs/reg_r06_native_hinge_v1/R2C/fit_inputs/R06_HINGE_D1.npz"
R2C_D2_INPUT = ROOT / "experiments/runs/reg_r06_native_hinge_v1/R2C/fit_inputs/R06_HINGE_D2.npz"
RESOURCE_POLICY = ROOT / "reports/optimization_v3/RESOURCE_POLICY_USER_REVISED_20261105.json"
OFFICIAL_TRAIN = TP2_ROOT / "data/raw/regression/Train.csv"
SHARED_LOCK = r4.SHARED_JOB_LOCK
KNOT = 50.0
SLOPE_BOUNDS = [0.0, 2.0]
THREADS = 2
MIN_GROUP_N = 200
MIN_CYCLE_N = 50

SOURCE_FILES = {
    "runner": Path(__file__).resolve(),
    "verifier": ROOT / "scripts/verify_reg_r06_present_native_hinge_v1.py",
    "tests": ROOT / "tests/test_reg_r06_present_native_hinge_v1.py",
    "protocol": PROTOCOL,
    "handoff": HANDOFF,
    "hinge_solver_r4": ROOT / "scripts/reg_monotone_hinge_p50_v4.py",
    "hinge_core_r1": ROOT / "scripts/reg_monotone_hinge_p50_v1.py",
    "direct_identity_helpers_v3": ROOT / "scripts/reg_monotone_hinge_p50_v3.py",
    "final_wave_helpers_v2": ROOT / "scripts/reg_final_wave_v2.py",
}


class GateError(RuntimeError):
    """Fail-closed source, identity, authorization, or execution error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    arr = np.asarray(value)
    prefix = canonical_bytes({"dtype": arr.dtype.str, "shape": list(arr.shape)}) + b"\0"
    if arr.dtype.kind in "USO":
        payload = b"\0".join(str(item).encode("utf-8", errors="surrogatepass")
                              for item in arr.reshape(-1).tolist())
    else:
        payload = np.ascontiguousarray(arr).tobytes(order="C")
    return hashlib.sha256(prefix + payload).hexdigest()


def write_once(path: Path, body: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(body), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise GateError(f"Immutable output exists with different bytes: {path}")
        return hashlib.sha256(payload).hexdigest()
    temp = None
    try:
        with tempfile.NamedTemporaryFile("wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise GateError(f"Refusing to overwrite immutable output: {path}") from exc
    finally:
        if temp is not None:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(payload).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _target_free_cache_inventory() -> tuple[dict[str, Any], str]:
    if not TARGET_FREE_INVENTORY.is_file():
        raise GateError(f"Pinned R06 cache inventory is absent: {TARGET_FREE_INVENTORY}")
    digest = file_sha256(TARGET_FREE_INVENTORY)
    report = json.loads(TARGET_FREE_INVENTORY.read_text(encoding="utf-8"))
    if (report.get("status") != "PASS_TARGET_FREE_PREFLIGHT"
            or report.get("target_values_opened") is not False):
        raise GateError("R06 cache inventory is not an outcome-blind target-free receipt")
    inventory = report.get("input_hashes", {}).get("base_r06_cache_inventory", {})
    if not all(c in inventory for c in ("67", "75")):
        raise GateError("R06 fit67/fit75 OOF caches are not both pinned")
    return inventory, digest


def _source_input_paths() -> dict[str, Path]:
    inventory, _ = _target_free_cache_inventory()
    paths: dict[str, Path] = {
        "target_free_inventory": TARGET_FREE_INVENTORY,
        "official_train": OFFICIAL_TRAIN,
        "oof_label_and_identity_bundle": LABEL_BUNDLE,
        "r4_hinge_d1_state": R4_D1_STATE,
        "r4_hinge_d2_state": R4_D2_STATE,
        "r4_hinge_d1_input": R4_D1_INPUT,
        "r4_hinge_d2_input": R4_D2_INPUT,
        "r4_root_source_pass": R4_ROOT_PASS,
        "r4_diagnostics": R4_DIAGNOSTICS,
        "r4_diagnostics_verification": R4_DIAGNOSTICS_VERIFICATION,
        "r4_lp_ledger": R4_LEDGER,
        "r2c_root_source_pass": R2C_ROOT_PASS,
        "r2c_diagnostics": R2C_DIAGNOSTICS,
        "r2c_independent_verification": R2C_VERIFICATION,
        "r2c_lp_ledger": R2C_LEDGER,
        "r2c_d1_state": R2C_D1_STATE,
        "r2c_d2_state": R2C_D2_STATE,
        "r2c_d1_input": R2C_D1_INPUT,
        "r2c_d2_input": R2C_D2_INPUT,
        "resource_policy": RESOURCE_POLICY,
    }
    for origin in ("67", "75"):
        record = inventory[origin]
        paths[f"r06_fit{origin}_npz"] = TP2_ROOT / record["npz"]
        paths[f"r06_fit{origin}_sidecar"] = TP2_ROOT / record["sidecar"]
    return paths


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise GateError(f"Required source file is absent: {path}")
        hashes[name] = file_sha256(path)
    return hashes


def _prior_slot_accounting() -> dict[str, Any]:
    if not R4_LEDGER.is_file():
        raise GateError("Existing R4 fit ledger is absent; prior budget cannot be reconciled")
    events: dict[str, list[str]] = {}
    for line in R4_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        events.setdefault(str(item.get("fit_id")), []).append(str(item.get("event")))
    starts = sum(row.count("LP_STARTED") for row in events.values())
    if set(events) != {"HINGE_D1", "HINGE_D2", "HINGE_FINAL"} or starts != 3:
        raise GateError("R4 prior slot ledger differs from the observed three-start history")
    if not R2C_LEDGER.is_file():
        raise GateError("R2C native ledger is absent; prior native slots cannot be reconciled")
    native_events: dict[str, list[str]] = {}
    for line in R2C_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        native_events.setdefault(str(item.get("fit_id")), []).append(str(item.get("event")))
    native_starts = sum(row.count("LP_STARTED") for row in native_events.values())
    if (set(native_events) != {"R06_HINGE_D1", "R06_HINGE_D2"}
            or native_starts != 2
            or len([line for line in R2C_LEDGER.read_text(encoding="utf-8").splitlines()
                    if line.strip()]) != 6
            or any("LP_FAILED" in row for row in native_events.values())):
        raise GateError("Prior R2C native ledger differs from its two completed diagnostic starts")
    return {
        "old_r4_ledger_path": str(R4_LEDGER.resolve()),
        "old_r4_ledger_sha256": file_sha256(R4_LEDGER),
        "old_r4_fit_ids": {key: list(value) for key, value in sorted(events.items())},
        "old_r4_lp_starts_already_consumed": starts,
        "prior_native_ledger_path": str(R2C_LEDGER.resolve()),
        "prior_native_ledger_sha256": file_sha256(R2C_LEDGER),
        "prior_native_fit_ids": {key: list(value) for key, value in sorted(native_events.items())},
        "prior_native_lp_starts_already_consumed": native_starts,
        "new_pilot_ledger_path": str(LEDGER.resolve()),
        "new_pilot_lp_start_cap": 3,
        "new_pilot_diagnostic_slot_cap": 2,
        "new_pilot_conditional_final_slot_cap": 1,
        "new_pilot_initial_starts": 0,
        "old_native_final_slot_repurposed": False,
        "old_ledger_reset_or_rewritten": False,
    }


def create_prefit_freeze() -> dict[str, Any]:
    if ROOT_SOURCE_PASS.exists() or LEDGER.exists():
        raise GateError("New root pass or fit ledger already exists; refusing a new prefit freeze")
    paths = _source_input_paths()
    input_hashes = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise GateError(f"Required frozen input is absent: {path}")
        input_hashes[name] = {"path": str(path.resolve()), "sha256": file_sha256(path),
                              "bytes": path.stat().st_size}
    inventory, inventory_sha = _target_free_cache_inventory()
    for origin in ("67", "75"):
        expected = inventory[origin]
        for kind in ("npz", "sidecar"):
            name = f"r06_fit{origin}_{kind}"
            if input_hashes[name]["sha256"] != expected[f"{kind}_sha256"]:
                raise GateError(f"R06 fit{origin} {kind} differs from target-free inventory")
        sidecar = json.loads(Path(input_hashes[f"r06_fit{origin}_sidecar"]["path"]).read_text(
            encoding="utf-8"))
        if sidecar.get("npz_sha256") != input_hashes[f"r06_fit{origin}_npz"]["sha256"]:
            raise GateError(f"R06 fit{origin} sidecar does not bind its prediction archive")
    if input_hashes["official_train"]["sha256"] != EXPECTED_TRAIN_SHA:
        raise GateError("Official Train source hash differs from the frozen historical contract")
    r4_pass = json.loads(R4_ROOT_PASS.read_text(encoding="utf-8"))
    r4_diagnostics = json.loads(R4_DIAGNOSTICS.read_text(encoding="utf-8"))
    r4_verification = json.loads(R4_DIAGNOSTICS_VERIFICATION.read_text(encoding="utf-8"))
    r4_pass_sha = file_sha256(R4_ROOT_PASS)
    if (r4_pass.get("verdict") != "PASS"
            or r4_pass.get("authorized_fit_ids") != ["HINGE_D1", "HINGE_D2"]
            or r4_pass.get("source_commit") != "0298c57cad5f3d921730f1c023b01369a676715d"
            or r4_diagnostics.get("root_source_pass_sha256") != r4_pass_sha
            or r4_diagnostics.get("diagnostic_fit_ids") != ["HINGE_D1", "HINGE_D2"]
            or r4_verification.get("status") != "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
            or r4_verification.get("root_source_pass_sha256") != r4_pass_sha):
        raise GateError("Saved R4 same-cutoff comparators lack a passing root/replay lineage")
    r2c_pass = json.loads(R2C_ROOT_PASS.read_text(encoding="utf-8"))
    r2c_signature = r2c_pass.pop("signature", None)
    if r2c_signature != canonical_sha(r2c_pass):
        raise GateError("Prior all-row native source pass has an invalid signature")
    r2c_pass["signature"] = r2c_signature
    r2c_report = json.loads(R2C_DIAGNOSTICS.read_text(encoding="utf-8"))
    r2c_verification = json.loads(R2C_VERIFICATION.read_text(encoding="utf-8"))
    r2c_report_signature = r2c_report.pop("signature", None)
    r2c_verification_signature = r2c_verification.pop("signature", None)
    if (r2c_report_signature != canonical_sha(r2c_report)
            or r2c_verification_signature != canonical_sha(r2c_verification)):
        raise GateError("Prior R2C diagnostics or independent receipt signature is invalid")
    r2c_report["signature"] = r2c_report_signature
    r2c_verification["signature"] = r2c_verification_signature
    r2c_events = [json.loads(line) for line in R2C_LEDGER.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    r2c_pass_sha = file_sha256(R2C_ROOT_PASS)
    if (r2c_pass.get("experiment_id") != "REG_R06_NATIVE_HINGE_V1_R2C"
            or r2c_pass.get("verdict") != "PASS_SOURCE_AND_INPUTS"
            or r2c_pass.get("authorized_fit_ids") != ["R06_HINGE_D1", "R06_HINGE_D2"]
            or r2c_pass.get("final_fit_authorized") is not False
            or r2c_report.get("root_source_pass_sha256") != r2c_pass_sha
            or r2c_report.get("qualification_gate", {}).get("pass") is not False
            or r2c_report.get("exploratory_gate", {}).get("pass") is not False
            or r2c_report.get("root_review_eligible") is not False
            or r2c_verification.get("status") != "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
            or r2c_verification.get("root_source_pass_sha256") != r2c_pass_sha
            or sum(event.get("event") == "LP_STARTED" for event in r2c_events) != 2
            or {event.get("source_commit") for event in r2c_events} !=
               {"a82af2eaa909daac884b393be636cc6c736722ce"}
            or len(r2c_events) != 6
            or {event.get("fit_id") for event in r2c_events} !=
               {"R06_HINGE_D1", "R06_HINGE_D2"}
            or any(event.get("fit_id") == "R06_HINGE_FINAL" for event in r2c_events)):
        raise GateError("Prior R2C all-row diagnostics or two-slot history failed lineage checks")
    r4_ledger_events = [json.loads(line) for line in R4_LEDGER.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    historical_states = {}
    for fit_id, path, expected_rows, input_path in (
            ("HINGE_D1", R4_D1_STATE, 4_950, R4_D1_INPUT),
            ("HINGE_D2", R4_D2_STATE, 12_412, R4_D2_INPUT)):
        historical = json.loads(path.read_text(encoding="utf-8"))
        solver = historical.get("solver_result", {})
        fit_input = historical.get("fit_input", {})
        if (historical.get("status") != "LP_COMPLETE"
                or historical.get("fit_id") != fit_id
                or solver.get("success") is not True
                or solver.get("fit_rows") != expected_rows
                or solver.get("knot") != KNOT
                or solver.get("slope_bounds") != SLOPE_BOUNDS
                or fit_input.get("fit_input_bundle_sha256") != file_sha256(input_path)):
            raise GateError(f"Saved historical {fit_id} comparator failed identity checks")
        ledger_rows = [event for event in r4_ledger_events
                       if event.get("fit_id") == fit_id]
        completed = [event for event in ledger_rows if event.get("event") == "LP_COMPLETE"]
        reserved = [event for event in ledger_rows if event.get("event") == "RESERVE"]
        if (len(completed) != 1 or len(reserved) != 1
                or completed[0].get("state_sha256") != file_sha256(path)
                or reserved[0].get("fit_input", {}).get(
                    "fit_input_bundle_sha256") != file_sha256(input_path)
                or reserved[0].get("root_source_pass_sha256") != r4_pass_sha):
            raise GateError(f"Saved historical {fit_id} state differs from its R4 ledger")
        historical_states[fit_id] = {
            "state_sha256": file_sha256(path),
            "fit_input_sha256": file_sha256(input_path),
            "fit_rows": expected_rows,
            "coefficients": {key: float(solver[key]) for key in
                             ("intercept", "slope_low", "slope_high", "knot")},
        }
    all_row_states = {}
    for fit_id, path, input_path, expected_rows in (
            ("R06_HINGE_D1", R2C_D1_STATE, R2C_D1_INPUT, 4_950),
            ("R06_HINGE_D2", R2C_D2_STATE, R2C_D2_INPUT, 12_412)):
        state = json.loads(path.read_text(encoding="utf-8"))
        solver = state.get("solver_result", {})
        fit_input = state.get("fit_input", {})
        events = [event for event in r2c_events if event.get("fit_id") == fit_id]
        reserved = [event for event in events if event.get("event") == "RESERVE"]
        completed = [event for event in events if event.get("event") == "LP_COMPLETE"]
        if (state.get("status") != "LP_COMPLETE" or state.get("fit_id") != fit_id
                or solver.get("success") is not True or solver.get("fit_rows") != expected_rows
                or solver.get("knot") != KNOT or solver.get("slope_bounds") != SLOPE_BOUNDS
                or fit_input.get("fit_input_bundle_sha256") != file_sha256(input_path)
                or len(reserved) != 1 or len(completed) != 1
                or completed[0].get("state_sha256") != file_sha256(path)
                or reserved[0].get("fit_input_sha256") != file_sha256(input_path)
                or reserved[0].get("root_source_pass_sha256") != r2c_pass_sha):
            raise GateError(f"Saved all-row native {fit_id} control failed its R2C ledger lineage")
        all_row_states[fit_id] = {
            "state_sha256": file_sha256(path),
            "fit_input_sha256": file_sha256(input_path),
            "fit_rows": expected_rows,
            "coefficients": {key: float(solver[key]) for key in
                             ("intercept", "slope_low", "slope_high", "knot")},
        }
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "IDENTITY_FROZEN_PENDING_NON_TARGET_PREFLIGHT_AND_ROOT_REVIEW_R1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT.resolve()),
        "branch": _git_value("branch", "--show-current"),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_hashes": _source_hashes(),
        "source_commit_must_be_ancestor": True,
        "input_hashes": input_hashes,
        "target_free_inventory_sha256": inventory_sha,
        "cache_assignment": {
            "R06_fit67_cycles": [68, 69, 70, 71, 72, 73, 74, 75],
            "R06_fit75_cycles": [76, 77, 78, 79],
            "R06_fit71_used": False,
            "prediction_member": "prediction",
        },
        "identity_contract": {
            "cached_label_bundle": str(LABEL_BUNDLE.resolve()),
            "rows": EXPECTED_ROWS,
            "cycles": list(range(68, 80)),
            "D1_fit_cycles": [68, 69, 70, 71],
            "D1_score_cycles": [72, 73, 74, 75],
            "D2_fit_cycles": [68, 69, 70, 71, 72, 73, 74, 75],
            "D2_score_cycles": [76, 77, 78, 79],
            "fit_population": "present rows only; missing rows excluded before solver input construction",
            "D1_present_fit_rows": EXPECTED_PRESENT_FIT_D1,
            "D2_present_fit_rows": EXPECTED_PRESENT_FIT_D2,
            "D1_missing_fit_rows_excluded": EXPECTED_MISSING_FIT_D1,
            "D2_missing_fit_rows_excluded": EXPECTED_MISSING_FIT_D2,
            "conditional_final_present_fit_rows": 19_870,
            "final_fit_id": FIT_IDS[2],
        },
        "calibration": {
            "form": "intercept + slope_low*min(p,50) + slope_high*max(p-50,0)",
            "knot": KNOT,
            "slope_bounds": SLOPE_BOUNDS,
            "objective": "unweighted sum absolute error",
            "solver": "pinned R4 HiGHS solver; 2 threads; 180 second time limit",
            "fit_rows": "present only for D1, D2, and conditional FINAL",
            "application": "clip to [0,100] on present score rows; missing rows bit-exact raw R06",
            "no_knots_grids_weights_or_seed_search": True,
        },
        "comparators": {
            "primary": "same-history R06 with the saved same-cutoff R4 HINGE_D1 or HINGE_D2 coefficients",
            "mechanism_control": "saved all-row native R2C hinge D1/D2 coefficients applied to the same R06 score rows",
            "separate_full_model": "original DIRECT predictions with the matching saved R4 HINGE_D1 or HINGE_D2 coefficients; diagnostic context only",
            "raw": "same-history raw R06 predictions",
            "historical_cutoff_states": historical_states,
            "all_row_native_states": all_row_states,
            "full_oof_final_coefficients_used_for_temporal_comparison": False,
        },
        "mechanism_gate": {
            "candidate_must_beat_all_row_native_both_windows": True,
            "gain_strictly_greater_than": 0.0,
        },
        "qualification_gate": {
            "pooled_gain_vs_primary_min": 0.005,
            "each_window_gain_vs_primary_min": 0.0,
            "max_supported_availability_or_tail_harm_vs_primary_le": 0.25,
        },
        "exploratory_gate": {
            "pooled_gain_vs_primary_min": 0.0,
            "each_window_harm_vs_primary_le": 0.02,
            "supported_group_harm_vs_primary_le": 0.25,
            "group_minimum_rows": MIN_GROUP_N,
            "cycle_minimum_rows": MIN_CYCLE_N,
        },
        "final_fit_policy": {
            "root_review_eligible_if_qualification_or_exploratory_pass": True,
            "separate_root_final_approval_required": True,
            "source_pass_never_authorizes_final_fit": True,
            "final_fit_is_in_sample_calibration_not_holdout_validation": True,
            "new_pilot_slot_cap": 3,
            "new_pilot_diagnostic_slot_cap": 2,
            "new_pilot_conditional_final_slot_cap": 1,
            "old_r4_ledger_is_separate_and_immutable": True,
            "prior_native_diagnostic_slots_already_consumed": 2,
            "old_native_final_slot_repurposed": False,
        },
        "resources": {
            "resource_policy_path": str(RESOURCE_POLICY.resolve()),
            "resource_policy_sha256": file_sha256(RESOURCE_POLICY),
            "shared_lock_path": str(SHARED_LOCK.resolve()),
            "threads": THREADS,
            "official_lp_starts_cap": 3,
            "new_base_or_preprocessing_fits": 0,
        },
        "prior_slot_accounting": _prior_slot_accounting(),
        "root_source_pass_path": str(ROOT_SOURCE_PASS.resolve()),
        "root_final_approval_path": str(ROOT_FINAL_APPROVAL.resolve()),
        "root_source_pass_created_by_runner": False,
        "test_or_sample_accessed": False,
        "test_labels_read": False,
        "kaggle_accessed": False,
        "csv_created": False,
        "official_lp_starts": 0,
    }
    body["signature"] = canonical_sha(body)
    digest = write_once(FREEZE, body)
    return {"status": body["status"], "path": str(FREEZE.resolve()), "sha256": digest}


def verify_freeze() -> tuple[dict[str, Any], str]:
    if not FREEZE.is_file():
        raise GateError("Prefit freeze is missing; run freeze first")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    signature = freeze.pop("signature", None)
    if signature != canonical_sha(freeze):
        raise GateError("Prefit freeze signature is invalid")
    freeze["signature"] = signature
    if (freeze.get("experiment_id") != EXPERIMENT_ID
            or freeze.get("status") !=
               "IDENTITY_FROZEN_PENDING_NON_TARGET_PREFLIGHT_AND_ROOT_REVIEW_R1"):
        raise GateError("Prefit freeze identity or status is invalid")
    for name, item in freeze.get("input_hashes", {}).items():
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise GateError(f"Frozen input changed: {name}")
    if freeze.get("source_hashes") != _source_hashes():
        raise GateError("Source code changed after the prefit freeze")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
         str(freeze["source_commit"]), _git_value("rev-parse", "HEAD")],
        capture_output=True, text=True, check=False)
    if ancestor.returncode != 0:
        raise GateError("Frozen source commit is not an ancestor of the current checkout")
    return freeze, file_sha256(FREEZE)


def _load_npz_selected(path: Path, names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        members = list(archive.files)
        missing = sorted(set(names) - set(members))
        if missing:
            raise GateError(f"Pinned NPZ lacks required members {missing}: {path}")
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    return arrays, members


def _assert_vector(data: Mapping[str, np.ndarray], name: str, n: int) -> None:
    if data[name].ndim != 1 or len(data[name]) != n:
        raise GateError(f"{name} has invalid row shape")


def apply_hinge(raw: Any, missing: Any, missing_fallback: Any,
                coefficients: Mapping[str, Any]) -> np.ndarray:
    p = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(missing, dtype=bool)
    fallback = np.asarray(missing_fallback, dtype=np.float64)
    if p.ndim != 1 or mask.shape != p.shape or fallback.shape != p.shape:
        raise GateError("Hinge application arrays are not aligned vectors")
    low = np.minimum(p, KNOT)
    high = np.maximum(p - KNOT, 0.0)
    candidate = np.clip(float(coefficients["intercept"])
                        + float(coefficients["slope_low"]) * low
                        + float(coefficients["slope_high"]) * high, 0.0, 100.0)
    out = np.where(mask, fallback, candidate)
    if not np.array_equal(out[mask].view(np.uint64), fallback[mask].view(np.uint64)):
        raise GateError("Missing predictions were not preserved bit-exactly")
    return out


def _load_non_target_identity(freeze: Mapping[str, Any]) -> dict[str, Any]:
    paths = {name: Path(item["path"]) for name, item in freeze["input_hashes"].items()}
    label_names = ("physical_rows", "record_ids", "prediction_origins",
                   "original_bundle_indices", "cycles", "missing_mask",
                   "direct_predictions")
    labels, label_members = _load_npz_selected(paths["oof_label_and_identity_bundle"], label_names)
    n = len(labels["record_ids"])
    for name in label_names:
        _assert_vector(labels, name, n)
    if (n != EXPECTED_ROWS or len(np.unique(labels["physical_rows"])) != n
            or len(np.unique(labels["record_ids"])) != n
            or not np.array_equal(np.unique(labels["cycles"]), np.arange(68, 80))
            or not np.all(np.isfinite(labels["direct_predictions"]))
            or np.any((labels["direct_predictions"] < 0) | (labels["direct_predictions"] > 100))
            or labels["missing_mask"].dtype.kind != "b"):
        raise GateError("Cached OOF identity bundle is malformed")
    by_id = {str(record_id): i for i, record_id in enumerate(labels["record_ids"])}
    if len(by_id) != n:
        raise GateError("Cached OOF record IDs are not unique")

    r06 = np.empty(n, dtype=np.float64)
    r06_origin = np.full(n, -1, dtype=np.int16)
    member_names: dict[str, list[str]] = {}
    for origin, allowed_cycles, cycle_member in (
            (67, set(range(68, 76)), "audit__cycle_code"),
            (75, set(range(76, 80)), "score_cycles")):
        cache, members = _load_npz_selected(
            paths[f"r06_fit{origin}_npz"],
            ("score_rows", "score_record_ids", "prediction", cycle_member))
        ids = cache["score_record_ids"].astype(str)
        cycles = np.asarray(cache[cycle_member], dtype=np.int64)
        predictions = np.asarray(cache["prediction"], dtype=np.float64)
        rows = np.asarray(cache["score_rows"], dtype=np.int64)
        m = len(ids)
        for name in ("score_rows", "score_record_ids", "prediction", cycle_member):
            _assert_vector(cache, name, m)
        if (len(np.unique(ids)) != m or not np.all(np.isfinite(predictions))
                or np.any((predictions < 0) | (predictions > 100))):
            raise GateError(f"R06 fit{origin} predictions/IDs are malformed")
        selected = np.isin(cycles, list(allowed_cycles))
        actual = {(str(ids[j]), int(cycles[j]), int(rows[j]))
                  for j in np.flatnonzero(selected)}
        expected = {(str(labels["record_ids"][j]), int(labels["cycles"][j]),
                     int(labels["physical_rows"][j]))
                    for j in range(n) if int(labels["cycles"][j]) in allowed_cycles}
        if actual != expected or len(actual) != int(np.sum(selected)):
            raise GateError(f"R06 fit{origin} query IDs/cycles/positions differ from cached OOF rows")
        for j in np.flatnonzero(selected):
            label_i = by_id[str(ids[j])]
            if int(labels["cycles"][label_i]) != int(cycles[j]) or r06_origin[label_i] != -1:
                raise GateError(f"R06 fit{origin} cycle/source assignment is inconsistent")
            r06[label_i] = predictions[j]
            r06_origin[label_i] = origin
        member_names[str(origin)] = members
    if np.any(r06_origin < 0) or not np.array_equal(r06_origin, labels["prediction_origins"]):
        raise GateError("R06 E67/E75 assignments differ from the frozen OOF origin identity")

    missing = labels["missing_mask"].astype(bool)
    if not np.array_equal(labels["direct_predictions"][missing], r06[missing]):
        raise GateError("Cached DIRECT missing-row fallback is not bit-exact to R06")
    cycles_all = labels["cycles"].astype(int)
    fit_population_records: dict[str, Any] = {}
    score_record_hashes: dict[str, str] = {}
    for fit_id, fit_mask, score_mask in (
            (FIT_IDS[0], (cycles_all >= 68) & (cycles_all <= 71),
             (cycles_all >= 72) & (cycles_all <= 75)),
            (FIT_IDS[1], (cycles_all >= 68) & (cycles_all <= 75),
             (cycles_all >= 76) & (cycles_all <= 79))):
        fit_cycle_indices = np.flatnonzero(fit_mask)
        present_indices = fit_cycle_indices[~missing[fit_cycle_indices]]
        excluded_indices = fit_cycle_indices[missing[fit_cycle_indices]]
        score_indices = np.flatnonzero(score_mask)
        expected_fit = EXPECTED_PRESENT_FIT_D1 if fit_id == FIT_IDS[0] else EXPECTED_PRESENT_FIT_D2
        expected_excluded = EXPECTED_MISSING_FIT_D1 if fit_id == FIT_IDS[0] else EXPECTED_MISSING_FIT_D2
        expected_score = EXPECTED_SCORE_D1 if fit_id == FIT_IDS[0] else EXPECTED_SCORE_D2
        if (len(present_indices) != expected_fit or len(excluded_indices) != expected_excluded
                or len(score_indices) != expected_score
                or np.any(missing[present_indices]) or np.any(~missing[excluded_indices])):
            raise GateError(f"Present-only fit population or score-window counts differ for {fit_id}")
        fit_population_records[fit_id] = {
            "fit_rows_before_missing_exclusion": int(len(fit_cycle_indices)),
            "present_fit_rows": int(len(present_indices)),
            "missing_fit_rows_excluded": int(len(excluded_indices)),
            "present_rows": [{
                "record_id": str(labels["record_ids"][i]),
                "physical_row": int(labels["physical_rows"][i]),
                "cycle": int(labels["cycles"][i]),
                "prediction_origin": int(labels["prediction_origins"][i]),
            } for i in present_indices],
            "excluded_missing_rows": [{
                "record_id": str(labels["record_ids"][i]),
                "physical_row": int(labels["physical_rows"][i]),
                "cycle": int(labels["cycles"][i]),
                "prediction_origin": int(labels["prediction_origins"][i]),
            } for i in excluded_indices],
        }
        score_record_hashes[fit_id] = array_sha256(labels["record_ids"][score_indices])
    historical_states = {}
    for window_id, state_key, expected_fit_id, expected_rows in (
            ("D1", "r4_hinge_d1_state", "HINGE_D1", 4_950),
            ("D2", "r4_hinge_d2_state", "HINGE_D2", 12_412)):
        state = json.loads(paths[state_key].read_text(encoding="utf-8"))
        coefficients = state.get("solver_result", {})
        if (state.get("status") != "LP_COMPLETE"
                or state.get("fit_id") != expected_fit_id
                or state.get("fit_input", {}).get("fit_rows") != expected_rows
                or coefficients.get("success") is not True
                or coefficients.get("fit_rows") != expected_rows
                or coefficients.get("knot") != KNOT
                or coefficients.get("slope_bounds") != SLOPE_BOUNDS):
            raise GateError(f"Past-only {expected_fit_id} comparator identity failed")
        historical_states[window_id] = {
            "state_sha256": file_sha256(paths[state_key]),
            "coefficients": {key: float(coefficients[key]) for key in
                             ("intercept", "slope_low", "slope_high", "knot")},
            "fit_rows": expected_rows,
            "fit_id": expected_fit_id,
        }
    primary_by_window = {}
    full_by_window = {}
    all_row_native_by_window = {}
    for window_id, fit_id in (("D1", FIT_IDS[0]), ("D2", FIT_IDS[1])):
        score_indices = _stage_indices(labels["cycles"].astype(int), fit_id, missing)[1]
        coefficients = json.loads(paths[
            "r4_hinge_d1_state" if window_id == "D1" else "r4_hinge_d2_state"
        ].read_text(encoding="utf-8"))["solver_result"]
        all_row_state = json.loads(paths[
            "r2c_d1_state" if window_id == "D1" else "r2c_d2_state"
        ].read_text(encoding="utf-8"))["solver_result"]
        primary_full = np.full(n, np.nan, dtype=np.float64)
        direct_full = np.full(n, np.nan, dtype=np.float64)
        all_row_full = np.full(n, np.nan, dtype=np.float64)
        primary_full[score_indices] = apply_hinge(
            r06[score_indices], missing[score_indices], r06[score_indices], coefficients)
        direct_full[score_indices] = apply_hinge(
            labels["direct_predictions"][score_indices], missing[score_indices],
            r06[score_indices], coefficients)
        all_row_full[score_indices] = apply_hinge(
            r06[score_indices], missing[score_indices], r06[score_indices], all_row_state)
        primary_by_window[window_id] = primary_full
        full_by_window[window_id] = direct_full
        all_row_native_by_window[window_id] = all_row_full
    d1_fit_cycle_rows = int(np.sum((cycles_all >= 68) & (cycles_all <= 71)))
    d1_fit_missing = int(np.sum((cycles_all >= 68) & (cycles_all <= 71) & missing))
    d2_fit_cycle_rows = int(np.sum((cycles_all >= 68) & (cycles_all <= 75)))
    d2_fit_missing = int(np.sum((cycles_all >= 68) & (cycles_all <= 75) & missing))
    counts = {
        "fit_D1_cycle_rows": d1_fit_cycle_rows,
        "fit_D1_missing_excluded": d1_fit_missing,
        "fit_D1_present_rows": d1_fit_cycle_rows - d1_fit_missing,
        "score_D1": int(np.sum((labels["cycles"] >= 72) & (labels["cycles"] <= 75))),
        "fit_D2_cycle_rows": d2_fit_cycle_rows,
        "fit_D2_missing_excluded": d2_fit_missing,
        "fit_D2_present_rows": d2_fit_cycle_rows - d2_fit_missing,
        "score_D2": int(np.sum((labels["cycles"] >= 76) & (labels["cycles"] <= 79))),
        "conditional_final_present_rows": int(np.sum(~missing)),
    }
    expected_counts = {
        "fit_D1_cycle_rows": 4_950,
        "fit_D1_missing_excluded": EXPECTED_MISSING_FIT_D1,
        "fit_D1_present_rows": EXPECTED_PRESENT_FIT_D1,
        "score_D1": EXPECTED_SCORE_D1,
        "fit_D2_cycle_rows": 12_412,
        "fit_D2_missing_excluded": EXPECTED_MISSING_FIT_D2,
        "fit_D2_present_rows": EXPECTED_PRESENT_FIT_D2,
        "score_D2": EXPECTED_SCORE_D2,
        "conditional_final_present_rows": 19_870,
    }
    if counts != expected_counts:
        raise GateError(f"Frozen cycle split counts differ: {counts}")
    return {
        "labels": labels, "label_npz_members": label_members,
        "r06_prediction": r06, "r06_origin": r06_origin,
        "same_cutoff_hinge_on_r06": primary_by_window,
        "same_cutoff_hinge_on_direct": full_by_window,
        "all_row_native_hinge": all_row_native_by_window,
        "historical_calibrators": historical_states,
        "counts": counts, "cache_npz_members": member_names,
        "missing_count": int(np.sum(missing)),
        "fit_population_records": fit_population_records,
        "score_record_id_hashes": score_record_hashes,
        "identity_hashes": {
            "record_ids": array_sha256(labels["record_ids"]),
            "physical_rows": array_sha256(labels["physical_rows"]),
            "prediction_origins": array_sha256(labels["prediction_origins"]),
            "original_bundle_indices": array_sha256(labels["original_bundle_indices"]),
            "cycles": array_sha256(labels["cycles"]),
            "missing_mask": array_sha256(missing),
            "r06_oof_prediction": array_sha256(r06),
            "r06_prediction_origin": array_sha256(r06_origin),
            "direct_predictions": array_sha256(labels["direct_predictions"]),
            "same_cutoff_hinge_on_r06_D1": array_sha256(primary_by_window["D1"]),
            "same_cutoff_hinge_on_r06_D2": array_sha256(primary_by_window["D2"]),
            "same_cutoff_hinge_on_direct_D1": array_sha256(full_by_window["D1"]),
            "same_cutoff_hinge_on_direct_D2": array_sha256(full_by_window["D2"]),
            "all_row_native_hinge_D1": array_sha256(all_row_native_by_window["D1"]),
            "all_row_native_hinge_D2": array_sha256(all_row_native_by_window["D2"]),
            "D1_present_fit_record_ids": array_sha256(np.asarray([
                row["record_id"] for row in fit_population_records[FIT_IDS[0]]["present_rows"]
            ])),
            "D2_present_fit_record_ids": array_sha256(np.asarray([
                row["record_id"] for row in fit_population_records[FIT_IDS[1]]["present_rows"]
            ])),
            "score_D1_record_ids": score_record_hashes[FIT_IDS[0]],
            "score_D2_record_ids": score_record_hashes[FIT_IDS[1]],
        },
    }


def identity_preflight() -> dict[str, Any]:
    if ROOT_SOURCE_PASS.exists() or LEDGER.exists():
        raise GateError("Preflight must finish before root pass or official fit ledger exists")
    freeze, freeze_sha = verify_freeze()
    with r4.SharedHeavyJobLock(SHARED_LOCK):
        resources = [r4.resource_snapshot(boundary="initial_locked_before_non_target_identity",
                                          initial=True)]
        identity = _load_non_target_identity(freeze)
        resources.append(r4.resource_snapshot(boundary="after_non_target_identity"))
    id_body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "prefit_freeze_sha256": freeze_sha,
        "source_commit": freeze["source_commit"],
        "fit_populations": identity["fit_population_records"],
        "score_record_id_hashes": identity["score_record_id_hashes"],
        "fit_record_id_hashes": {
            fit_id: identity["identity_hashes"][
                "D1_present_fit_record_ids" if fit_id == FIT_IDS[0]
                else "D2_present_fit_record_ids"]
            for fit_id in DIAGNOSTIC_IDS
        },
        "score_window_record_id_hashes": {
            fit_id: identity["score_record_id_hashes"][fit_id] for fit_id in DIAGNOSTIC_IDS
        },
    }
    id_body["signature"] = canonical_sha(id_body)
    fit_ids_sha = write_once(FIT_POPULATION_IDS, id_body)
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS_R1",
        "prefit_freeze_path": str(FREEZE.resolve()),
        "prefit_freeze_sha256": freeze_sha,
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"],
        "cache_assignment": freeze["cache_assignment"],
        "row_counts": identity["counts"],
        "fit_population_ids_path": str(FIT_POPULATION_IDS.resolve()),
        "fit_population_ids_sha256": fit_ids_sha,
        "unique_record_ids": EXPECTED_ROWS,
        "unique_physical_rows": EXPECTED_ROWS,
        "cycles": list(range(68, 80)),
        "prediction_origins": {"67": 12_412, "75": 8_491},
        "missing_rows_in_cached_oof": identity["missing_count"],
        "array_hashes": identity["identity_hashes"],
        "npz_member_audit": {
            "label_bundle_members_seen": identity["label_npz_members"],
            "target_member_present": "targets" in identity["label_npz_members"],
            "target_member_loaded": False,
            "r06_cache_members_seen": identity["cache_npz_members"],
            "test_or_sample_members_loaded": False,
        },
        "historical_same_cutoff_comparators": identity["historical_calibrators"],
        "full_oof_final_coefficients_loaded": False,
        "all_row_native_mechanism_control_hashes": {
            key: value for key, value in identity["identity_hashes"].items()
            if key.startswith("all_row_native_hinge_")
        },
        "same_cutoff_comparator_hashes": {
            key: value for key, value in identity["identity_hashes"].items()
            if key.startswith("same_cutoff_")
        },
        "target_reader_called": False,
        "train_target_values_loaded": False,
        "score_target_values_loaded": False,
        "official_lp_starts": 0,
        "new_ledger_created": False,
        "csv_created": False,
        "test_labels_read": False,
        "test_or_sample_accessed": False,
        "kaggle_accessed": False,
        "root_source_pass_exists": False,
        "resource_snapshots": resources,
    }
    body["signature"] = canonical_sha(body)
    digest = write_once(PREFLIGHT, body)
    return {"status": body["status"], "path": str(PREFLIGHT.resolve()), "sha256": digest,
            "rows": EXPECTED_ROWS, "diagnostic_starts": 0}


def _load_root_authorization() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    freeze, freeze_sha = verify_freeze()
    if not PREFLIGHT.is_file():
        raise GateError("Non-target identity preflight is missing")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    preflight_sig = preflight.pop("signature", None)
    if preflight_sig != canonical_sha(preflight):
        raise GateError("Identity preflight signature is invalid")
    preflight["signature"] = preflight_sig
    audit = preflight.get("npz_member_audit", {})
    if (preflight.get("status") !=
            "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS_R1"
            or preflight.get("prefit_freeze_sha256") != freeze_sha
            or audit.get("target_member_loaded") is not False
            or preflight.get("train_target_values_loaded") is not False
            or preflight.get("official_lp_starts") != 0):
        raise GateError("Identity preflight is not a clean non-target pass")
    if not ROOT_SOURCE_PASS.is_file():
        raise GateError("ROOT_SOURCE_PASS_R1.json is absent; no official fit may start")
    approval = json.loads(ROOT_SOURCE_PASS.read_text(encoding="utf-8"))
    approval_sig = approval.pop("signature", None)
    if approval_sig != canonical_sha(approval):
        raise GateError("Root source-pass signature is invalid")
    approval["signature"] = approval_sig
    if (approval.get("experiment_id") != EXPERIMENT_ID
            or approval.get("verdict") != "PASS_SOURCE_AND_INPUTS"
            or approval.get("prefit_freeze_sha256") != freeze_sha
            or approval.get("identity_preflight_sha256") != file_sha256(PREFLIGHT)
            or approval.get("source_commit") != freeze["source_commit"]
            or approval.get("source_hashes") != freeze["source_hashes"]
            or approval.get("input_hashes") != freeze["input_hashes"]
            or approval.get("authorized_fit_ids") != list(DIAGNOSTIC_IDS)
            or approval.get("final_fit_authorized") is not False):
        raise GateError("Root source pass does not bind this exact source/input freeze")
    return freeze, freeze_sha, approval, file_sha256(ROOT_SOURCE_PASS)


def _load_final_approval(freeze: Mapping[str, Any], freeze_sha: str,
                         source_pass_sha: str, diagnostics_sha: str) -> tuple[dict[str, Any], str]:
    if not ROOT_FINAL_APPROVAL.is_file():
        raise GateError("Separate root final-fit approval is absent; FINAL has not started")
    approval = json.loads(ROOT_FINAL_APPROVAL.read_text(encoding="utf-8"))
    signature = approval.pop("signature", None)
    if signature != canonical_sha(approval):
        raise GateError("Root final-fit approval signature is invalid")
    approval["signature"] = signature
    if (approval.get("experiment_id") != EXPERIMENT_ID
            or approval.get("verdict") != "AUTHORIZE_FINAL_FIT"
            or approval.get("fit_id") != FIT_IDS[2]
            or approval.get("authorized_fit_ids") != [FIT_IDS[2]]
            or approval.get("final_fit_authorized") is not True
            or approval.get("prefit_freeze_sha256") != freeze_sha
            or approval.get("identity_preflight_sha256") != file_sha256(PREFLIGHT)
            or approval.get("root_source_pass_sha256") != source_pass_sha
            or approval.get("diagnostics_sha256") != diagnostics_sha
            or approval.get("source_commit") != freeze["source_commit"]
            or approval.get("source_hashes") != freeze["source_hashes"]
            or approval.get("input_hashes") != freeze["input_hashes"]):
        raise GateError("Root final-fit approval does not bind this exact present-only diagnostic package")
    return approval, file_sha256(ROOT_FINAL_APPROVAL)


def _validate_ledger(path: Path = LEDGER) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_fit: dict[str, dict[str, list[dict[str, Any]]]] = {}
    order: dict[str, list[str]] = {}
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                fit_id, kind = event["fit_id"], event["event"]
                if fit_id not in FIT_IDS or kind not in {"RESERVE", "LP_STARTED", "LP_COMPLETE", "LP_FAILED"}:
                    raise GateError("New ledger contains unknown fit or event")
                bucket = by_fit.setdefault(fit_id, {key: [] for key in
                    ("RESERVE", "LP_STARTED", "LP_COMPLETE", "LP_FAILED")})
                bucket[kind].append(event)
                order.setdefault(fit_id, []).append(kind)
        except (ValueError, KeyError, TypeError) as exc:
            raise GateError("New ledger is malformed; no retry") from exc
    reserved = []
    starts = 0
    for fit_id in FIT_IDS:
        bucket = by_fit.get(fit_id)
        if not bucket:
            continue
        if any(len(bucket[kind]) > 1 for kind in bucket):
            raise GateError(f"Duplicate ledger event for {fit_id}; no retry")
        sequence = order[fit_id]
        allowed = (["RESERVE"], ["RESERVE", "LP_STARTED"],
                   ["RESERVE", "LP_STARTED", "LP_COMPLETE"],
                   ["RESERVE", "LP_STARTED", "LP_FAILED"])
        if sequence not in allowed:
            raise GateError(f"Invalid ledger sequence for {fit_id}; no retry")
        reserved.append(fit_id)
        starts += len(bucket["LP_STARTED"])
        reserve = bucket["RESERVE"][0]
        for kind in ("LP_STARTED", "LP_COMPLETE", "LP_FAILED"):
            for event in bucket[kind]:
                if (event.get("fit_input_sha256") != reserve.get("fit_input_sha256")
                        or event.get("root_source_pass_sha256")
                           != reserve.get("root_source_pass_sha256")
                        or event.get("root_final_approval_sha256")
                           != reserve.get("root_final_approval_sha256")
                        or event.get("diagnostics_sha256") != reserve.get("diagnostics_sha256")):
                    raise GateError("Ledger event does not match its reservation")
    if reserved != list(FIT_IDS[:len(reserved)]) or starts > 3:
        raise GateError("New pilot budget is not a contiguous three-slot prefix")
    return by_fit


def _append_ledger(event: Mapping[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    with LEDGER.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_once(path: Path, arrays: Mapping[str, Any]) -> str:
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != hashlib.sha256(payload).hexdigest():
            raise GateError(f"Immutable NPZ output differs: {path}")
        return file_sha256(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise GateError(f"Refusing to overwrite immutable NPZ: {path}") from exc
    finally:
        if temp is not None:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(payload).hexdigest()


def _fit_input_arrays(identity: Mapping[str, Any], indices: np.ndarray,
                      targets: np.ndarray) -> dict[str, np.ndarray]:
    labels = identity["labels"]
    if np.any(labels["missing_mask"][indices]):
        raise GateError("Present-only calibrator input contains a missing row")
    return {
        "physical_rows": np.asarray(labels["physical_rows"][indices], dtype=np.int64),
        "record_ids": np.asarray(labels["record_ids"][indices]).astype(str),
        "prediction_origins": np.asarray(labels["prediction_origins"][indices], dtype=np.int8),
        "cycles": np.asarray(labels["cycles"][indices], dtype=np.int16),
        "r06_prediction_origins": np.asarray(identity["r06_origin"][indices], dtype=np.int16),
        "r06_predictions": np.asarray(identity["r06_prediction"][indices], dtype="<f8"),
        "missing_mask": np.asarray(labels["missing_mask"][indices], dtype=bool),
        "targets": np.asarray(targets[indices], dtype="<f8"),
    }


def _stage_indices(cycles: np.ndarray, fit_id: str,
                   missing_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    if missing_mask is None:
        raise GateError("Present-only fit routing requires the explicit missing mask")
    missing = np.asarray(missing_mask, dtype=bool)
    if cycles.ndim != 1 or missing.shape != cycles.shape:
        raise GateError("Cycle and missing vectors are not aligned")
    if fit_id == FIT_IDS[0]:
        fit_mask = (cycles >= 68) & (cycles <= 71)
        score_mask = (cycles >= 72) & (cycles <= 75)
        purpose = "diagnostic"
    elif fit_id == FIT_IDS[1]:
        fit_mask = (cycles >= 68) & (cycles <= 75)
        score_mask = (cycles >= 76) & (cycles <= 79)
        purpose = "diagnostic"
    elif fit_id == FIT_IDS[2]:
        fit_mask = (cycles >= 68) & (cycles <= 79)
        score_mask = np.zeros(len(cycles), dtype=bool)
        purpose = "final"
    else:
        raise GateError(f"Unknown fit ID: {fit_id}")
    fit_mask &= ~missing
    return np.flatnonzero(fit_mask), np.flatnonzero(score_mask), purpose


def _fit_input_identity(arrays: Mapping[str, np.ndarray], bundle_sha: str,
                        freeze_sha: str, approval_sha: str, fit_id: str,
                        final_approval_sha: str | None = None,
                        diagnostics_sha: str | None = None) -> dict[str, Any]:
    identity = {
        "experiment_id": EXPERIMENT_ID, "fit_id": fit_id,
        "fit_rows": len(arrays["targets"]),
        "fit_input_bundle_sha256": bundle_sha,
        "fit_input_hashes": {name: array_sha256(value) for name, value in arrays.items()},
        "prefit_freeze_sha256": freeze_sha,
        "root_source_pass_sha256": approval_sha,
    }
    if fit_id == FIT_IDS[2]:
        identity["root_final_approval_sha256"] = final_approval_sha
        identity["diagnostics_sha256"] = diagnostics_sha
    return identity


def _run_fit_slot(fit_id: str, arrays: Mapping[str, np.ndarray], purpose: str,
                  freeze_sha: str, approval_sha: str,
                  final_approval_sha: str | None = None,
                  diagnostics_sha: str | None = None) -> dict[str, Any]:
    ledger = _validate_ledger()
    prior = ledger.get(fit_id)
    if prior:
        if prior["LP_COMPLETE"]:
            state_path = RUN_DIR / "fits" / f"{fit_id}.json"
            if not state_path.is_file():
                raise GateError("Completed slot lacks saved state; no retry")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (state.get("status") != "LP_COMPLETE"
                    or state.get("root_source_pass_sha256") != approval_sha
                    or state.get("fit_input", {}).get("fit_input_bundle_sha256")
                       != prior["RESERVE"][0].get("fit_input_sha256")
                    or state.get("root_final_approval_sha256") != final_approval_sha
                    or state.get("diagnostics_sha256") != diagnostics_sha):
                raise GateError("Saved completed slot differs from its ledger; no retry")
            return state
        raise GateError(f"Fit slot {fit_id} is already reserved/started/failed; no retry")
    next_index = sum(1 for key in FIT_IDS if key in ledger)
    if fit_id != FIT_IDS[next_index]:
        raise GateError(f"Fit order violation: expected {FIT_IDS[next_index]}, got {fit_id}")
    input_path = RUN_DIR / "fit_inputs" / f"{fit_id}.npz"
    state_path = RUN_DIR / "fits" / f"{fit_id}.json"
    slack_path = RUN_DIR / "solver_slacks" / f"{fit_id}.npz"
    if state_path.exists() or slack_path.exists():
        raise GateError("Fit output exists without a ledger completion; no retry")
    with r4.SharedHeavyJobLock(SHARED_LOCK):
        before = r4.resource_snapshot(boundary=f"{fit_id}_initial_locked_before_start",
                                      initial=True)
        bundle_sha = _write_npz_once(input_path, arrays)
        fit_input = _fit_input_identity(arrays, bundle_sha, freeze_sha, approval_sha, fit_id,
                                        final_approval_sha, diagnostics_sha)
        reservation_id = canonical_sha({"fit_id": fit_id, "fit_input_sha256": bundle_sha,
                                        "root_source_pass_sha256": approval_sha,
                                        "root_final_approval_sha256": final_approval_sha,
                                        "diagnostics_sha256": diagnostics_sha})[:24]
        common = {"fit_id": fit_id, "reservation_id": reservation_id,
                  "fit_input_sha256": bundle_sha, "source_commit": _git_value("rev-parse", "HEAD"),
                  "root_source_pass_sha256": approval_sha,
                  "root_final_approval_sha256": final_approval_sha,
                  "diagnostics_sha256": diagnostics_sha}
        _append_ledger({"event": "RESERVE", **common, "resources": before})
        _validate_ledger()
        start_resources = r4.resource_snapshot(boundary=f"{fit_id}_fit_start")
        _append_ledger({"event": "LP_STARTED", **common, "retry_allowed": False,
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                        "resources": start_resources})
        started = time.perf_counter()
        try:
            solver = r4.solve_hinge_l1(arrays["r06_predictions"], arrays["targets"],
                                       purpose=purpose)
            slacks = np.asarray(solver.pop("_slacks"), dtype="<f8")
            post = r4.resource_snapshot(boundary=f"{fit_id}_after_solver")
            slack_sha = _write_npz_once(slack_path, {"slacks": slacks})
            state = {
                "schema_version": 1, "experiment_id": EXPERIMENT_ID,
                "fit_id": fit_id, "status": "LP_COMPLETE",
                "fit_input": fit_input, "solver_result": solver,
                "solver_slacks_path": str(slack_path.resolve()),
                "solver_slacks_sha256": slack_sha,
                "solver_wall_seconds": time.perf_counter() - started,
                "resources_before_start": before, "resources_after_solver": post,
                "root_source_pass_sha256": approval_sha,
                "root_final_approval_sha256": final_approval_sha,
                "diagnostics_sha256": diagnostics_sha,
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            }
            state_sha = write_once(state_path, state)
            _append_ledger({"event": "LP_COMPLETE", **common, "state_sha256": state_sha,
                            "solver_slacks_sha256": slack_sha,
                            "completed_utc": datetime.now(timezone.utc).isoformat()})
            _validate_ledger()
            return state
        except Exception as exc:
            failure = {
                "schema_version": 1, "experiment_id": EXPERIMENT_ID,
                "fit_id": fit_id, "status": "LP_FAILED", "fit_input": fit_input,
                "failure": f"{type(exc).__name__}: {exc}",
                "root_source_pass_sha256": approval_sha,
                "root_final_approval_sha256": final_approval_sha,
                "diagnostics_sha256": diagnostics_sha,
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            }
            state_sha = write_once(state_path, failure)
            _append_ledger({"event": "LP_FAILED", **common, "state_sha256": state_sha,
                            "failure": failure["failure"]})
            raise


def _load_cached_targets_after_pass(freeze: Mapping[str, Any], approval_sha: str,
                                    identity: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    _freeze, _freeze_sha, _approval, current_pass_sha = _load_root_authorization()
    if current_pass_sha != approval_sha:
        raise GateError("Root source pass changed before cached Train target access")
    path = Path(freeze["input_hashes"]["oof_label_and_identity_bundle"]["path"])
    data, _ = _load_npz_selected(path, ("record_ids", "cycles", "targets"))
    if (not np.array_equal(data["record_ids"], identity["labels"]["record_ids"])
            or not np.array_equal(data["cycles"], identity["labels"]["cycles"])):
        raise GateError("Cached Train targets are not aligned to frozen non-target identity")
    targets = np.asarray(data["targets"], dtype=np.float64)
    if (targets.shape != (EXPECTED_ROWS,) or not np.all(np.isfinite(targets))
            or np.any((targets < 0.0) | (targets > 100.0))):
        raise GateError("Cached target vector failed the selected OOF row/range gate")
    return targets, {
        "source_path": str(path.resolve()), "source_sha256": file_sha256(path),
        "selected_rows": EXPECTED_ROWS, "target_values_loaded_after_root_pass": True,
        "target_array_sha256": array_sha256(targets),
        "train_csv_reopened": False, "test_labels_read": False,
    }


def _metric_summary(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    absolute = np.abs(error)
    if not len(error):
        return {"n": 0, "status": "UNKNOWN"}
    return {
        "n": int(len(error)),
        "status": "MEASURED",
        "mae": float(np.mean(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
        "mean_error_prediction_minus_target": float(np.mean(error)),
    }


def _summary_block(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                   mask: np.ndarray, *, supported_minimum: int,
                   support_name: str) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    n = int(np.sum(selected))
    if n < supported_minimum:
        return {"n": n, "supported": False, "status": "UNKNOWN",
                "support_rule": f"n >= {supported_minimum} for {support_name}"}
    return {
        "n": n, "supported": True, "status": "MEASURED",
        "support_rule": f"n >= {supported_minimum} for {support_name}",
        "metrics": {name: _metric_summary(y[selected], pred[selected])
                    for name, pred in predictions.items()},
    }


def _diagnostic_summary(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                        missing: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    full = _summary_block(y, predictions, np.ones(len(y), dtype=bool),
                          supported_minimum=1, support_name="full rows")
    present = _summary_block(y, predictions, ~missing,
                             supported_minimum=1, support_name="present rows")
    groups = {
        "availability_present": _summary_block(
            y, predictions, ~missing, supported_minimum=MIN_GROUP_N,
            support_name="availability_present"),
        "availability_missing": _summary_block(
            y, predictions, missing, supported_minimum=MIN_GROUP_N,
            support_name="availability_missing"),
        "target_le_5": _summary_block(
            y, predictions, y <= 5.0, supported_minimum=MIN_GROUP_N,
            support_name="target_le_5"),
        "target_ge_95": _summary_block(
            y, predictions, y >= 95.0, supported_minimum=MIN_GROUP_N,
            support_name="target_ge_95"),
    }
    for name, group in groups.items():
        if group.get("supported"):
            metrics = group["metrics"]
            group["candidate_harm_vs_primary"] = (
                metrics["present_native_hinge"]["mae"]
                - metrics["transported_r4_hinge_on_r06"]["mae"])
            group["candidate_gain_vs_primary"] = -group["candidate_harm_vs_primary"]
    per_cycle = {}
    for cycle in sorted(set(map(int, cycles))):
        mask = cycles == cycle
        per_cycle[str(cycle)] = _summary_block(
            y, predictions, mask, supported_minimum=MIN_CYCLE_N,
            support_name=f"cycle_{cycle}")
    return {
        "n": int(len(y)),
        "present_rows": int(np.sum(~missing)),
        "missing_rows": int(np.sum(missing)),
        "full_rows": full,
        "present_only": present,
        "availability_and_tail_groups": groups,
        "per_cycle": per_cycle,
        "all_required_groups_supported": all(g.get("supported") for g in groups.values()),
        "all_scored_cycles_supported": all(g.get("supported") for g in per_cycle.values()),
    }


def _primary_metrics(summary: Mapping[str, Any]) -> tuple[float, float]:
    candidate = summary["metrics"]["present_native_hinge"]["mae"]
    primary = summary["metrics"]["transported_r4_hinge_on_r06"]["mae"]
    return primary - candidate, candidate - primary


def _validate_diagnostic_state_mapping(
        states: Mapping[str, Mapping[str, Any]],
        expected_fit_rows: Mapping[str, int] | None = None) -> None:
    expected = dict(expected_fit_rows or {
        FIT_IDS[0]: EXPECTED_PRESENT_FIT_D1,
        FIT_IDS[1]: EXPECTED_PRESENT_FIT_D2,
    })
    if set(states) != set(expected):
        raise GateError("Diagnostic state mapping must contain exactly D1 and D2")
    for fit_id, expected_rows in expected.items():
        state = states[fit_id]
        if (state.get("fit_id") != fit_id or state.get("status") != "LP_COMPLETE"
                or state.get("solver_result", {}).get("fit_rows") != expected_rows
                or state.get("fit_input", {}).get("fit_rows") != expected_rows
                or state.get("fit_input", {}).get("fit_id") != fit_id):
            raise GateError(f"{fit_id} cannot be replaced by a FINAL or other-cutoff state")


def _validate_saved_diagnostic_slots_before_targets(freeze_sha: str,
                                                    approval_sha: str) -> None:
    expected = {FIT_IDS[0]: EXPECTED_PRESENT_FIT_D1,
                FIT_IDS[1]: EXPECTED_PRESENT_FIT_D2}
    ledger = _validate_ledger()
    for fit_id, expected_rows in expected.items():
        path = RUN_DIR / "fits" / f"{fit_id}.json"
        prior = ledger.get(fit_id)
        if not path.exists() and not prior:
            continue
        if not path.exists() or not prior or not prior["RESERVE"]:
            raise GateError("Saved diagnostic output and ledger are not a matched pair")
        state = json.loads(path.read_text(encoding="utf-8"))
        reserve = prior["RESERVE"][0]
        if (state.get("fit_id") != fit_id or state.get("status") != "LP_COMPLETE"
                or state.get("fit_input", {}).get("fit_rows") != expected_rows
                or state.get("fit_input", {}).get("fit_id") != fit_id
                or state.get("fit_input", {}).get("prefit_freeze_sha256") != freeze_sha
                or state.get("root_source_pass_sha256") != approval_sha
                or state.get("fit_input", {}).get("root_source_pass_sha256") != approval_sha
                or reserve.get("fit_input_sha256") != state.get("fit_input", {}).get(
                    "fit_input_bundle_sha256")
                or not prior["LP_COMPLETE"]
                or prior["LP_COMPLETE"][0].get("state_sha256") != file_sha256(path)):
            raise GateError("Saved diagnostic slot is bound to the wrong calibration cutoff")
    if (RUN_DIR / "fits" / f"{FIT_IDS[2]}.json").exists():
        raise GateError("Conditional FINAL state cannot substitute for either diagnostic cutoff")


def _root_review_eligible(mechanism_pass: bool, qualification_pass: bool,
                          exploratory_pass: bool) -> bool:
    return bool(mechanism_pass and (qualification_pass or exploratory_pass))


def _score_window(identity: Mapping[str, Any], targets: np.ndarray,
                  candidate: np.ndarray, indices: np.ndarray,
                  window_id: str) -> dict[str, Any]:
    y = targets[indices]
    predictions = {
        "present_native_hinge": candidate[indices],
        "all_row_native_hinge": identity["all_row_native_hinge"][window_id][indices],
        "transported_r4_hinge_on_r06": identity["same_cutoff_hinge_on_r06"][window_id][indices],
        "matching_r4_hinge_on_direct": identity["same_cutoff_hinge_on_direct"][window_id][indices],
        "raw_r06": identity["r06_prediction"][indices],
    }
    missing = identity["labels"]["missing_mask"][indices].astype(bool)
    cycles = identity["labels"]["cycles"][indices].astype(int)
    summary = _diagnostic_summary(y, predictions, missing, cycles)
    gain, harm = _primary_metrics(summary["full_rows"])
    summary["gain_vs_transported_r4_primary"] = gain
    summary["harm_vs_transported_r4_primary"] = harm
    summary["gain_vs_all_row_native_mechanism"] = (
        summary["full_rows"]["metrics"]["all_row_native_hinge"]["mae"]
        - summary["full_rows"]["metrics"]["present_native_hinge"]["mae"])
    summary["raw_r06_gain_vs_candidate"] = (
        summary["full_rows"]["metrics"]["raw_r06"]["mae"]
        - summary["full_rows"]["metrics"]["present_native_hinge"]["mae"])
    return summary


def _evaluate_diagnostics(identity: Mapping[str, Any], targets: np.ndarray,
                          states: Mapping[str, Mapping[str, Any]], *,
                          synthetic: bool = False) -> dict[str, Any]:
    synthetic_fit_rows = ({
        FIT_IDS[0]: int(states[FIT_IDS[0]]["fit_input"]["fit_rows"]),
        FIT_IDS[1]: int(states[FIT_IDS[1]]["fit_input"]["fit_rows"]),
    } if synthetic else None)
    _validate_diagnostic_state_mapping(states, expected_fit_rows=synthetic_fit_rows)
    cycles = identity["labels"]["cycles"].astype(int)
    windows = {}
    candidate_parts = []
    indices_parts = []
    for fit_id, window_id in ((FIT_IDS[0], "D1"), (FIT_IDS[1], "D2")):
        _, score_indices, _ = _stage_indices(
            cycles, fit_id, identity["labels"]["missing_mask"].astype(bool))
        coefficients = states[fit_id]["solver_result"]
        prediction = apply_hinge(
            identity["r06_prediction"][score_indices],
            identity["labels"]["missing_mask"][score_indices],
            identity["r06_prediction"][score_indices], coefficients)
        full_candidate = np.full(len(cycles), np.nan, dtype=np.float64)
        full_candidate[score_indices] = prediction
        windows[window_id] = _score_window(identity, targets, full_candidate,
                                           score_indices, window_id)
        candidate_parts.append(prediction)
        indices_parts.append(score_indices)
    indices = np.concatenate(indices_parts)
    candidate = np.concatenate(candidate_parts)
    if (not synthetic and (len(indices) != EXPECTED_POOLED
                           or len(np.unique(indices)) != EXPECTED_POOLED)):
        raise GateError("D1 and D2 score windows are not disjoint unique rows")
    if len(np.unique(indices)) != len(indices):
        raise GateError("D1 and D2 score windows overlap")
    pooled_predictions = {
        "present_native_hinge": candidate,
        "all_row_native_hinge": np.concatenate([
            identity["all_row_native_hinge"]["D1"][indices_parts[0]],
            identity["all_row_native_hinge"]["D2"][indices_parts[1]]]),
        "transported_r4_hinge_on_r06": np.concatenate([
            identity["same_cutoff_hinge_on_r06"]["D1"][indices_parts[0]],
            identity["same_cutoff_hinge_on_r06"]["D2"][indices_parts[1]]]),
        "matching_r4_hinge_on_direct": np.concatenate([
            identity["same_cutoff_hinge_on_direct"]["D1"][indices_parts[0]],
            identity["same_cutoff_hinge_on_direct"]["D2"][indices_parts[1]]]),
        "raw_r06": identity["r06_prediction"][indices],
    }
    y = targets[indices]
    missing = identity["labels"]["missing_mask"][indices].astype(bool)
    pooled = _diagnostic_summary(y, pooled_predictions, missing,
                                 identity["labels"]["cycles"][indices].astype(int))
    pooled_gain, pooled_harm = _primary_metrics(pooled["full_rows"])

    all_summaries = [pooled, *windows.values()]
    supported_group_harms = [
        group["candidate_harm_vs_primary"]
        for summary in all_summaries
        for group in summary["availability_and_tail_groups"].values()
        if group.get("supported")
    ]
    gates_supported = all(summary["all_required_groups_supported"]
                          and summary["all_scored_cycles_supported"]
                          for summary in all_summaries)
    max_group_harm = max(supported_group_harms) if supported_group_harms else None
    mechanism_gains = {
        name: float(windows[name]["gain_vs_all_row_native_mechanism"])
        for name in ("D1", "D2")
    }
    mechanism_pass = bool(
        mechanism_gains["D1"] > 0.0 and mechanism_gains["D2"] > 0.0
    )
    qualification = bool(
        gates_supported
        and pooled_gain >= 0.005
        and windows["D1"]["gain_vs_transported_r4_primary"] >= 0.0
        and windows["D2"]["gain_vs_transported_r4_primary"] >= 0.0
        and max_group_harm is not None and max_group_harm <= 0.25
    )
    exploratory = bool(
        gates_supported
        and pooled_gain >= 0.0
        and windows["D1"]["harm_vs_transported_r4_primary"] <= 0.02
        and windows["D2"]["harm_vs_transported_r4_primary"] <= 0.02
        and max_group_harm is not None and max_group_harm <= 0.25
    )
    root_review_eligible = _root_review_eligible(mechanism_pass, qualification, exploratory)
    return {
        "schema_version": 2, "experiment_id": EXPERIMENT_ID,
        "status": "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW",
        "windows": windows,
        "pooled_unique": {
            **pooled, "rows": len(indices), "unique_rows": len(np.unique(indices)),
            "gain_vs_transported_r4_primary": pooled_gain,
            "harm_vs_transported_r4_primary": pooled_harm,
            "gain_vs_all_row_native_mechanism": float(
                pooled["full_rows"]["metrics"]["all_row_native_hinge"]["mae"]
                - pooled["full_rows"]["metrics"]["present_native_hinge"]["mae"]),
            "max_supported_group_harm_vs_primary": max_group_harm,
        },
        "mechanism_gate": {
            "comparator": "saved all-row native R2C hinge on the same score rows",
            "each_window_gain_strictly_greater_than": 0.0,
            "window_gains": mechanism_gains,
            "pass": mechanism_pass,
        },
        "qualification_gate": {
            "pooled_gain_vs_primary_min": 0.005,
            "each_window_gain_vs_transported_primary_min": 0.0,
            "max_supported_availability_or_tail_harm_vs_primary_le": 0.25,
            "pass": qualification,
        },
        "exploratory_gate": {
            "pooled_gain_vs_primary_min": 0.0,
            "each_window_harm_vs_transported_primary_le": 0.02,
            "supported_group_harm_vs_primary_le": 0.25,
            "pass": exploratory,
        },
        "raw_r06_comparison": {
            "reported_in_full_present_tail_and_cycle_summaries": True,
            "gate_role": "diagnostic_only",
        },
        "root_review_eligible": root_review_eligible,
        "final_fit_started": False,
        "final_fit_requires_separate_root_approval": True,
        "full_oof_final_coefficients_used_in_temporal_diagnostic_gates": False,
        "full_direct_model_comparison_is_separate_from_primary_gate": True,
        "final_fit_not_a_holdout_claim": True,
        "unknown_unsupported_groups_block_scientific_pass": True,
        "test_labels_read": False, "test_or_sample_accessed": False,
        "csv_created": False, "kaggle_accessed": False,
    }


def run_diagnostics() -> dict[str, Any]:
    freeze, freeze_sha, _approval, approval_sha = _load_root_authorization()
    _validate_saved_diagnostic_slots_before_targets(freeze_sha, approval_sha)
    identity = _load_non_target_identity(freeze)
    targets, target_audit = _load_cached_targets_after_pass(freeze, approval_sha, identity)
    states: dict[str, Mapping[str, Any]] = {}
    for fit_id in DIAGNOSTIC_IDS:
        fit_indices, _, purpose = _stage_indices(
            identity["labels"]["cycles"].astype(int), fit_id,
            identity["labels"]["missing_mask"].astype(bool))
        arrays = _fit_input_arrays(identity, fit_indices, targets)
        state = _run_fit_slot(fit_id, arrays, purpose, freeze_sha, approval_sha)
        if state.get("status") != "LP_COMPLETE":
            raise GateError(f"Diagnostic fit did not complete: {fit_id}")
        states[fit_id] = state
    report = _evaluate_diagnostics(identity, targets, states)
    report.update({
        "prefit_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": file_sha256(PREFLIGHT),
        "root_source_pass_sha256": approval_sha,
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "official_lp_starts": len(DIAGNOSTIC_IDS),
        "fit_states": {
            fit_id: {"path": str((RUN_DIR / "fits" / f"{fit_id}.json").resolve()),
                     "sha256": file_sha256(RUN_DIR / "fits" / f"{fit_id}.json")}
            for fit_id in DIAGNOSTIC_IDS
        },
        "train_target_audit": target_audit,
        "base_or_preprocessing_fits": 0,
        "final_fit_started": False,
    })
    report["signature"] = canonical_sha(report)
    digest = write_once(DIAGNOSTICS, report)
    return {"status": report["status"], "path": str(DIAGNOSTICS.resolve()),
            "sha256": digest,
            "root_review_eligible": report["root_review_eligible"],
            "qualification_pass": report["qualification_gate"]["pass"],
            "exploratory_pass": report["exploratory_gate"]["pass"],
            "final_fit_started": False}


def run_final_fit() -> dict[str, Any]:
    freeze, freeze_sha, _approval, approval_sha = _load_root_authorization()
    _validate_saved_diagnostic_slots_before_targets(freeze_sha, approval_sha)
    if not DIAGNOSTICS.is_file():
        raise GateError("Diagnostic report is missing; final slot remains closed")
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    signature = diagnostics.pop("signature", None)
    if signature != canonical_sha(diagnostics):
        raise GateError("Diagnostic report signature is invalid")
    diagnostics["signature"] = signature
    qualification = diagnostics.get("qualification_gate", {}).get("pass") is True
    exploratory = diagnostics.get("exploratory_gate", {}).get("pass") is True
    mechanism_pass = diagnostics.get("mechanism_gate", {}).get("pass") is True
    eligible = _root_review_eligible(mechanism_pass, qualification, exploratory)
    if (diagnostics.get("root_review_eligible") is not eligible
            or not eligible
            or diagnostics.get("final_fit_started") is not False
            or diagnostics.get("final_fit_requires_separate_root_approval") is not True
            or diagnostics.get("root_source_pass_sha256") != approval_sha
            or diagnostics.get("prefit_freeze_sha256") != freeze_sha):
        return {"status": "FINAL_SLOT_CLOSED_DIAGNOSTICS_NOT_ROOT_REVIEW_ELIGIBLE",
                "fit_started": False}
    diagnostics_sha = file_sha256(DIAGNOSTICS)
    _final_approval, final_approval_sha = _load_final_approval(
        freeze, freeze_sha, approval_sha, diagnostics_sha)
    identity = _load_non_target_identity(freeze)
    targets, target_audit = _load_cached_targets_after_pass(freeze, approval_sha, identity)
    fit_indices, _, purpose = _stage_indices(
        identity["labels"]["cycles"].astype(int), FIT_IDS[2],
        identity["labels"]["missing_mask"].astype(bool))
    arrays = _fit_input_arrays(identity, fit_indices, targets)
    state = _run_fit_slot(FIT_IDS[2], arrays, purpose, freeze_sha, approval_sha,
                          final_approval_sha, diagnostics_sha)
    receipt = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "status": state.get("status"), "fit_id": FIT_IDS[2],
        "state_path": str((RUN_DIR / "fits" / f"{FIT_IDS[2]}.json").resolve()),
        "state_sha256": file_sha256(RUN_DIR / "fits" / f"{FIT_IDS[2]}.json"),
        "root_source_pass_sha256": approval_sha,
        "root_final_approval_sha256": final_approval_sha,
        "diagnostics_sha256": diagnostics_sha,
        "final_fit_is_in_sample_calibration_not_holdout_validation": True,
        "train_target_audit": target_audit,
        "test_labels_read": False, "csv_created": False, "kaggle_accessed": False,
    }
    receipt["signature"] = canonical_sha(receipt)
    path = REPORT_DIR / "FINAL_FIT_RECEIPT_R1.json"
    return {"status": receipt["status"], "path": str(path.resolve()),
            "sha256": write_once(path, receipt)}


def _write_text_once(path: Path, events: list[Mapping[str, Any]]) -> None:
    payload = b"".join(json.dumps(dict(item), sort_keys=True, separators=(",", ":")).encode()
                      + b"\n" for item in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise GateError(f"Synthetic immutable output differs: {path}")
    if not path.exists():
        path.write_bytes(payload)


def _synthetic_metric_e2e() -> dict[str, Any]:
    per_cycle = 400
    cycles = np.repeat(np.arange(68, 80, dtype=np.int16), per_cycle)
    pattern = np.tile(np.asarray([0.0, 2.0, 4.0, 50.0, 70.0, 90.0, 96.0, 100.0]), 50)
    targets = np.tile(pattern, 12).astype(np.float64)
    row_index = np.arange(len(cycles))
    raw = np.clip(targets + 4.0 + 6.0 * np.sin(row_index * 0.071), 0.0, 100.0)
    missing = (row_index % 4) == 0
    direct = raw.copy()
    labels = {"cycles": cycles, "missing_mask": missing}
    primary_by_window: dict[str, np.ndarray] = {}
    direct_by_window: dict[str, np.ndarray] = {}
    allrow_by_window: dict[str, np.ndarray] = {}
    primary_coefficients = {
        "D1": {"intercept": 1.5, "slope_low": 0.98, "slope_high": 0.99, "knot": KNOT},
        "D2": {"intercept": 1.2, "slope_low": 0.99, "slope_high": 0.98, "knot": KNOT},
    }
    allrow_coefficients = {
        "D1": {"intercept": 2.4, "slope_low": 0.96, "slope_high": 0.96, "knot": KNOT},
        "D2": {"intercept": 2.1, "slope_low": 0.97, "slope_high": 0.95, "knot": KNOT},
    }
    for window_id, fit_id in (("D1", FIT_IDS[0]), ("D2", FIT_IDS[1])):
        score_indices = _stage_indices(cycles, fit_id, missing)[1]
        primary = np.full(len(cycles), np.nan, dtype=np.float64)
        direct_hinge = np.full(len(cycles), np.nan, dtype=np.float64)
        allrow = np.full(len(cycles), np.nan, dtype=np.float64)
        primary[score_indices] = apply_hinge(
            raw[score_indices], missing[score_indices], raw[score_indices],
            primary_coefficients[window_id])
        direct_hinge[score_indices] = apply_hinge(
            direct[score_indices], missing[score_indices], raw[score_indices],
            primary_coefficients[window_id])
        allrow[score_indices] = apply_hinge(
            raw[score_indices], missing[score_indices], raw[score_indices],
            allrow_coefficients[window_id])
        primary_by_window[window_id] = primary
        direct_by_window[window_id] = direct_hinge
        allrow_by_window[window_id] = allrow
    identity = {
        "labels": labels,
        "r06_prediction": raw,
        "same_cutoff_hinge_on_r06": primary_by_window,
        "same_cutoff_hinge_on_direct": direct_by_window,
        "all_row_native_hinge": allrow_by_window,
    }
    states: dict[str, Mapping[str, Any]] = {}
    fit_counts: dict[str, int] = {}
    fitted_coefficients: dict[str, dict[str, float]] = {}
    for fit_id in DIAGNOSTIC_IDS:
        fit_indices, _, _ = _stage_indices(cycles, fit_id, missing)
        fit_counts[fit_id] = len(fit_indices)
        solver = r4.solve_hinge_l1(raw[fit_indices], targets[fit_indices], purpose="diagnostic")
        solver.pop("_slacks", None)
        fitted_coefficients[fit_id] = {
            key: float(solver[key]) for key in
            ("intercept", "slope_low", "slope_high", "knot")
        }
        states[fit_id] = {
            "fit_id": fit_id, "status": "LP_COMPLETE",
            "fit_input": {"fit_id": fit_id, "fit_rows": len(fit_indices)},
            "solver_result": solver,
        }
    report = _evaluate_diagnostics(identity, targets, states, synthetic=True)
    return {
        "status": "PASS_SYNTHETIC_METRIC_E2E",
        "fixture_rows": len(cycles),
        "score_rows": report["pooled_unique"]["rows"],
        "fit_rows_present_only": fit_counts,
        "candidate_coefficients": fitted_coefficients,
        "missing_rows_excluded_from_fits": True,
        "missing_score_predictions_bit_exact_raw": all(
            np.array_equal(
                identity["all_row_native_hinge"][window_id][
                    _stage_indices(cycles, fit_id, missing)[1]
                ][missing[_stage_indices(cycles, fit_id, missing)[1]]].view(np.uint64),
                raw[_stage_indices(cycles, fit_id, missing)[1]][
                    missing[_stage_indices(cycles, fit_id, missing)[1]]
                ].view(np.uint64),
            )
            for window_id, fit_id in (("D1", FIT_IDS[0]), ("D2", FIT_IDS[1]))
        ),
        "fit_and_score_metrics": report,
    }


def _synthetic_checks() -> dict[str, Any]:
    rows = 240
    ids = np.asarray([f"S{i:04d}" for i in range(rows)])
    cycles = np.repeat(np.arange(68, 80, dtype=np.int16), rows // 12)
    rng = np.random.default_rng(20261105)
    raw = np.linspace(0.0, 100.0, rows, dtype=np.float64)
    missing = np.arange(rows) % 7 == 0
    coeffs = {"intercept": 1.25, "slope_low": 0.9, "slope_high": 1.1, "knot": 50.0}
    applied = apply_hinge(raw, missing, raw, coeffs)
    missing_exact = np.array_equal(applied[missing].view(np.uint64), raw[missing].view(np.uint64))
    order = rng.permutation(rows)
    expected = {(str(ids[i]), int(cycles[i])): float(raw[i]) for i in range(rows)}
    shuffled = {(str(ids[i]), int(cycles[i])): float(raw[i]) for i in order}
    id_cycle_join = expected == shuffled
    mask = (cycles >= 68) & (cycles <= 71)
    present_indices, score_indices, _ = _stage_indices(cycles, FIT_IDS[0], missing)
    present_only_routing = bool(
        len(present_indices) == int(np.sum(mask & ~missing))
        and len(score_indices) == int(np.sum((cycles >= 72) & (cycles <= 75)))
        and not np.any(missing[present_indices])
    )
    calls = {"target_reader": 0, "solver": 0}

    def guarded(pass_ok: bool) -> None:
        if not pass_ok:
            raise GateError("root pass required")
        calls["target_reader"] += 1
        calls["solver"] += 1

    try:
        guarded(False)
        denied = False
    except GateError:
        denied = True
    no_read_or_fit = calls == {"target_reader": 0, "solver": 0}
    substituted = {
        FIT_IDS[0]: {"fit_id": FIT_IDS[2], "status": "LP_COMPLETE",
                     "fit_input": {"fit_id": FIT_IDS[2], "fit_rows": 19_870},
                     "solver_result": {"fit_rows": 19_870}},
        FIT_IDS[1]: {"fit_id": FIT_IDS[1], "status": "LP_COMPLETE",
                     "fit_input": {"fit_id": FIT_IDS[1], "fit_rows": EXPECTED_PRESENT_FIT_D2},
                     "solver_result": {"fit_rows": EXPECTED_PRESENT_FIT_D2}},
    }
    try:
        _validate_diagnostic_state_mapping(substituted)
        final_substitution_rejected = False
    except GateError:
        final_substitution_rejected = True
    mechanism_gate_mandatory = (
        _root_review_eligible(False, True, True) is False
        and _root_review_eligible(True, False, True) is True
    )

    temp_root = Path(tempfile.mkdtemp(prefix="r06_present_native_hinge_synth_"))
    try:
        ledger_path = temp_root / "ledger.jsonl"
        event = {"fit_id": FIT_IDS[0], "fit_input_sha256": "a" * 64,
                 "root_source_pass_sha256": "b" * 64}
        _write_text_once(ledger_path, [
            {"event": "RESERVE", **event}, {"event": "LP_STARTED", **event},
            {"event": "LP_COMPLETE", **event},
        ])
        complete_ok = bool(_validate_ledger(ledger_path)[FIT_IDS[0]]["LP_COMPLETE"])
        retry_path = temp_root / "retry.jsonl"
        _write_text_once(retry_path, [
            {"event": "RESERVE", **event}, {"event": "LP_STARTED", **event},
            {"event": "LP_FAILED", **event}, {"event": "LP_STARTED", **event},
        ])
        try:
            _validate_ledger(retry_path)
            retry_rejected = False
        except GateError:
            retry_rejected = True
        lock_path = temp_root / "shared.lock"
        with r4.SharedHeavyJobLock(lock_path):
            lock_acquired = True
        resource_pass = r4.resource_decision(3 * 1024**3, initial=True)["pass"]
        resource_fail = not r4.resource_decision(512 * 1024**2, initial=True)["pass"]
    finally:
        import shutil
        shutil.rmtree(temp_root)
    checks = {
        "synthetic_only": True,
        "missing_route_bit_exact": bool(missing_exact),
        "shuffled_id_cycle_join": bool(id_cycle_join),
        "present_only_fit_routing_excludes_missing": present_only_routing,
        "no_target_reader_or_solver_without_root_pass": bool(denied and no_read_or_fit),
        "final_state_substitution_rejected_by_cutoff_gate": bool(final_substitution_rejected),
        "mechanism_gate_is_mandatory_for_root_review": mechanism_gate_mandatory,
        "final_fit_is_not_automatic_after_review_eligibility": True,
        "completed_one_shot_ledger_accepted": bool(complete_ok),
        "retry_ledger_rejected": bool(retry_rejected),
        "shared_lock_acquisition": bool(lock_acquired),
        "resource_gate_accepts_sufficient_and_rejects_insufficient": bool(resource_pass and resource_fail),
        "official_lp_starts": 0, "target_values_loaded": False,
        "test_labels_read": False, "csv_created": False, "kaggle_accessed": False,
    }
    if not all(value for key, value in checks.items() if key not in {
            "official_lp_starts", "target_values_loaded", "test_labels_read",
            "csv_created", "kaggle_accessed"}):
        raise GateError("Synthetic QA failed")
    return checks


def run_synthetic_qa() -> dict[str, Any]:
    checks = _synthetic_checks()
    metrics_e2e = _synthetic_metric_e2e()
    if SYNTHETIC_QA.exists():
        existing = json.loads(SYNTHETIC_QA.read_text(encoding="utf-8"))
        signature = existing.pop("signature", None)
        if (signature != canonical_sha(existing)
                or existing.get("experiment_id") != EXPERIMENT_ID
                or existing.get("status") != "PASS_SYNTHETIC_SOURCE_AND_RECEIPT_E2E"
                or existing.get("checks") != checks
                or existing.get("metrics_e2e") != metrics_e2e):
            raise GateError("Existing synthetic QA receipt does not match this source run")
        return {"status": existing["status"], "path": str(SYNTHETIC_QA.resolve()),
                "sha256": file_sha256(SYNTHETIC_QA)}
    body = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "status": "PASS_SYNTHETIC_SOURCE_AND_RECEIPT_E2E",
        "checks": checks,
        "metrics_e2e": metrics_e2e,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["signature"] = canonical_sha(body)
    return {"status": body["status"], "path": str(SYNTHETIC_QA.resolve()),
            "sha256": write_once(SYNTHETIC_QA, body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "preflight", "synthetic-qa",
                                              "verify-source-ready", "run-diagnostics",
                                              "run-final"))
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = create_prefit_freeze()
        elif args.command == "preflight":
            result = identity_preflight()
        elif args.command == "synthetic-qa":
            result = run_synthetic_qa()
        elif args.command == "verify-source-ready":
            from scripts.verify_reg_r06_present_native_hinge_v1 import verify_source_ready
            result = verify_source_ready(write_receipt=True)
        elif args.command == "run-diagnostics":
            result = run_diagnostics()
        else:
            result = run_final_fit()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except GateError as exc:
        print(json.dumps({"status": "NO_GO", "error": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
