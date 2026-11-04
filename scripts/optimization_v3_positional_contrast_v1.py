"""Four-fit POS_CONTRAST_V1 pilot using frozen POS_BAG_OFFSET inputs.

This is a small adapter over the already reviewed offset solver and accounting
helpers.  It never opens Test or SampleSubmission.  In particular, the A02
development NPZ contains a historical label field; this module loads only
named prediction/identity/group arrays and never indexes that label field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from src.models.positional_contrast_v1 import contrast_block, expand_contrast_coefficients
from scripts import optimization_v3_positional_bag_offset_v1 as offset
from scripts.verify_optimization_v3_positional_bag_offset_v1 import (
    verify_prelabel_outputs as verify_parent_prelabel_outputs,
)
from scripts.verify_optimization_v3_positional_contrast_v1 import (
    verify_prelabel_outputs,
    verify_scored_results,
)


REPORT_ROOT = ROOT / "reports/optimization_v3/positional_contrast_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
TRAIN = ROOT / "data/raw/classification/Train.csv"
RECIPE = REPORT_ROOT / "RECIPE.json"
PREFIT_FREEZE_REPORT = REPORT_ROOT / "PREFIT_FREEZE.json"
PREFIT_AMENDMENT_REPORT = REPORT_ROOT / "PREFIT_AMENDMENT.json"
PROTOCOL = ROOT / "reports/optimization_v3/posmix_followup_v1/SOL_NEXT_RESEARCH.md"
ERROR_AUDIT = ROOT / "reports/optimization_v3/posmix_followup_v1/ERROR_AUDIT.json"
BAG_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
A02_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
A02_REPORT = ROOT / "reports/optimization_v3/a02_neighbors_v1/verification.json"

CARD_ID = "POS_CONTRAST_V1"
AMENDMENT_ID = "POS_CONTRAST_V1_CAP137_2026_11_04"
PILOT_RESERVATION_KEY = "POS_CONTRAST_V1:PILOT_4"
E79_RESERVATION_KEY = "POS_CONTRAST_V1:CONDITIONAL_E79_AFTER_SOL_REVIEW"
STAGE = "POS_CONTRAST_V1"
FIT_IDS = (
    "POS_CONTRAST_V1_E67_REAL",
    "POS_CONTRAST_V1_E67_SHUFFLED",
    "POS_CONTRAST_V1_E75_REAL",
    "POS_CONTRAST_V1_E75_SHUFFLED",
)
E79_ID = "POS_CONTRAST_V1_E79_REAL"
EXPECTED_LEDGER_SHA = "8380d884fabb3cf19edfd606b2b23c7839b9267d16c6660e579444e8089f2d22"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_PARENT_FREEZE = "ba58b1e446a6be8546789c5a56596fbdbf075efc0ee19fcc1f41e02af7b419d7"
BASE_CAPS = {"continuation": 49, "control_replay": 2, "pilot": 80, "technical_retry": 1}
NEW_CAPS = {"continuation": 50, "control_replay": 2, "pilot": 84, "technical_retry": 1}
BASE_SNAPSHOT = {
    "event_rows": 332, "global_cap": 132, "effective_category_caps": BASE_CAPS,
    "active_reservations_by_category": {
        "continuation": 49, "control_replay": 2, "pilot": 77, "technical_retry": 1,
    }, "active_reservations": 129, "fit_starts": 129, "fit_saved": 123,
    "fit_failed": 6, "fit_pending": 0,
}
E79_REVIEW_CYCLES = (76, 79)
PREFIT_REBIND_ID = "POS_CONTRAST_V1_PREFIT_SOURCE_REBIND_1"
FAILED_DISPATCH_COMMIT = "ba79ec30fd6a8b17f66cfde8186e79282ca1a494"
FAILED_DISPATCH_ERROR = "TypeError: _check_ledger() got an unexpected keyword argument 'freeze_signature'"

_ORIGINAL_LOAD = offset._load_origin_payload
_ORIGINAL_VERIFY = offset.verify_saved_inputs


class ContrastPilotError(RuntimeError):
    pass


def _json_read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContrastPilotError(f"Expected JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(x) for x in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(_canonical(value) + b"\n")
    with temp.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_paths() -> dict[str, Path]:
    return {
        "runner": ROOT / "scripts/optimization_v3_positional_contrast_v1.py",
        "verifier": ROOT / "scripts/verify_optimization_v3_positional_contrast_v1.py",
        "contrast_model": ROOT / "src/models/positional_contrast_v1.py",
        "offset_model": ROOT / "src/models/positional_bag_offset_v1.py",
        "offset_runner": ROOT / "scripts/optimization_v3_positional_bag_offset_v1.py",
        "contrast_tests": ROOT / "tests/test_positional_contrast_v1.py",
        "pilot_tests": ROOT / "tests/test_optimization_v3_positional_contrast_v1.py",
        "recipe": RECIPE,
        "protocol": PROTOCOL,
    }


def _source_hashes() -> dict[str, str]:
    return {key: offset.sha256_file(path) for key, path in _source_paths().items()}


def _assert_source_unchanged(freeze: Mapping[str, Any]) -> None:
    if _source_hashes() != freeze["source_hashes"]:
        raise ContrastPilotError("A frozen source, recipe, protocol, or test file changed")
    source_commit = str(freeze["source_commit"])
    if _git("merge-base", "--is-ancestor", source_commit, "HEAD") is None:
        raise ContrastPilotError("Frozen source commit is not an ancestor of current HEAD")


def _ledger_snapshot() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = offset._ledger_events(LEDGER)
    return events, offset.ledger_snapshot(events)


def _load_a02(origin: int, *, include_groups: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load only explicitly named non-target arrays from the mixed historical NPZ."""
    root = A02_ROOT / f"E{origin}" / "observed"
    metadata = _json_read(root / "predictions.json")
    receipt = _json_read(root / "fit_receipt.json")
    path = root / "predictions.npz"
    if metadata.get("fit_id") != f"A02_E{origin}_OBSERVED" or metadata.get("query_labels_passed_to_prediction") is not False:
        raise ContrastPilotError(f"A02 E{origin} cache is not the original label-isolated observed route")
    if receipt.get("fit_id") != metadata["fit_id"] or receipt.get("query_labels_passed_to_fit") is not False or receipt.get("test_data_used") is not False:
        raise ContrastPilotError(f"A02 E{origin} fit receipt source boundary mismatch")
    safe_keys = ["score_rows", "score_record_ids", "score_cycles", "prediction", "missing_mask"]
    if include_groups:
        safe_keys.extend(["entry_friction", "text_load_group"])
    values = _load_named_npz(path, safe_keys)
    prediction_sha = hashlib.sha256(np.asarray(values["prediction"], dtype="<f8").tobytes(order="C")).hexdigest()
    rows_sha = hashlib.sha256(np.asarray(values["score_rows"], dtype="<i8").tobytes(order="C")).hexdigest()
    ids_payload = json.dumps(values["score_record_ids"].astype(str).tolist(), separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
    ids_sha = hashlib.sha256(ids_payload).hexdigest()
    if prediction_sha != metadata["diagnostics"]["candidate_probability_sha256"]:
        raise ContrastPilotError(f"A02 E{origin} selected probability array hash mismatch")
    if rows_sha != metadata["score_row_positions_sha256"]:
        raise ContrastPilotError(f"A02 E{origin} selected row-index hash mismatch")
    if ids_sha != metadata["score_record_ids_sha256"]:
        raise ContrastPilotError(f"A02 E{origin} selected ID hash mismatch")
    if not np.isfinite(values["prediction"]).all() or ((values["prediction"] < 0) | (values["prediction"] > 1)).any():
        raise ContrastPilotError(f"A02 E{origin} probabilities are invalid")
    return {"metadata": metadata, "receipt": receipt, "prediction_json_sha256": offset.sha256_file(root / "predictions.json"),
            "receipt_sha256": offset.sha256_file(root / "fit_receipt.json"),
            "safe_array_hashes": {key: offset.sha256_array(value) for key, value in values.items()}}, values


def _load_named_npz(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Read only named arrays; never touch unrelated label arrays in an NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _load_bag_prediction(origin: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ready = _json_read(BAG_ROOT / "PREDICTIONS_READY.json")
    if ready.get("card_id") != "POS_BAG_OFFSET_V1" or ready.get("query_labels_opened") is not False:
        raise ContrastPilotError("Frozen POS_BAG prediction readiness is invalid")
    path = BAG_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    entry = ready["predictions"][f"E{origin}"]
    if offset.sha256_file(path) != entry["sha256"]:
        raise ContrastPilotError(f"POS_BAG E{origin} output hash mismatch")
    keys = ["origin", "fit_rows", "score_rows", "score_record_ids", "score_cycles", "present_mask",
            "prob_bag", "prob_missing_hgb", "prob_offset_real"]
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]).copy() for key in keys}
    return {"ready_sha256": offset.sha256_file(BAG_ROOT / "PREDICTIONS_READY.json"),
            "output_sha256": offset.sha256_file(path), "output_path": path}, values


def _parent_freeze() -> dict[str, Any]:
    value = _json_read(BAG_ROOT / "FREEZE.json")
    if value.get("freeze_signature") != EXPECTED_PARENT_FREEZE:
        raise ContrastPilotError("Historical POS_BAG freeze signature changed")
    return value


def _load_raw_origin(origin: int) -> tuple[dict[str, Any], dict[str, np.ndarray], Any, dict[str, Any]]:
    return _ORIGINAL_LOAD(origin, _parent_freeze())


def _contrast_payload(origin: int, freeze: Mapping[str, Any]):
    """Adapter consumed by the proven offset fitter; only position suffix changes."""
    manifest, prediction, bag, payload = _load_raw_origin(origin)
    matrices = payload["matrices"]
    for arm_name in ("position", "shuffled_position"):
        for prefix in ("fit", "score"):
            augmented_name = f"{prefix}_{arm_name}"
            base_name = f"{prefix}_base"
            augmented = sparse.csr_matrix(matrices[augmented_name], dtype=np.float64)
            base_matrix = sparse.csr_matrix(matrices[base_name], dtype=np.float64)
            projected = contrast_block(augmented, base_matrix.shape[1])
            matrices[augmented_name] = sparse.hstack((base_matrix, projected), format="csr", dtype=np.float64)
    return manifest, prediction, bag, payload


def _verify_parent_inputs() -> dict[str, Any]:
    if offset.sha256_file(TRAIN) != EXPECTED_TRAIN_SHA:
        raise ContrastPilotError("Official Classification Train hash mismatch")
    legacy = _ORIGINAL_VERIFY(include_current_code=False)
    bag_freeze = _parent_freeze()
    old_prelabel = verify_parent_prelabel_outputs(ROOT, BAG_ROOT, bag_freeze)
    if old_prelabel.get("status") != "PASS" or old_prelabel.get("query_labels_read") is not False:
        raise ContrastPilotError("Original POS_BAG frozen pP prelabel verification is not PASS/closed")
    identity = offset._read_identity_frame()
    origins: dict[str, Any] = {}
    for origin in (67, 75):
        suffix = f"E{origin}"
        _, bag_pred = _load_bag_prediction(origin)
        a02_meta, a02 = _load_a02(origin)
        if (not np.array_equal(bag_pred["score_rows"], a02["score_rows"])
                or not np.array_equal(bag_pred["score_record_ids"].astype(str), a02["score_record_ids"].astype(str))
                or not np.array_equal(bag_pred["score_cycles"].astype(np.int64), a02["score_cycles"].astype(np.int64))):
            raise ContrastPilotError(f"E{origin} POS/A02 prediction rows are not identical")
        if len(bag_pred["score_rows"]) != (20903 if origin == 67 else 8491):
            raise ContrastPilotError(f"E{origin} prediction population size mismatch")
        sub = identity.iloc[bag_pred["score_rows"]]
        if (not np.array_equal(sub["record_id"].astype(str).to_numpy(), bag_pred["score_record_ids"].astype(str))
                or not np.array_equal(sub["cycle_code"].to_numpy(dtype=np.int64), bag_pred["score_cycles"].astype(np.int64))):
            raise ContrastPilotError(f"E{origin} Train identity join mismatch")
        present = np.asarray(bag_pred["present_mask"], dtype=bool)
        if not np.array_equal(np.asarray(a02["missing_mask"], dtype=bool), ~present):
            raise ContrastPilotError(f"E{origin} POS/A02 availability masks differ")
        p_p = np.asarray(bag_pred["prob_offset_real"], dtype=np.float64)
        hgb = np.asarray(bag_pred["prob_missing_hgb"], dtype=np.float64)
        p_a = np.asarray(a02["prediction"], dtype=np.float64)
        if not np.array_equal(p_p[~present], hgb[~present]):
            raise ContrastPilotError(f"E{origin} historical POS missing route is not exact")
        origins[suffix] = {
            "score_rows_sha256": offset.sha256_array(bag_pred["score_rows"]),
            "score_record_ids_sha256": offset.sha256_array(bag_pred["score_record_ids"].astype(str)),
            "score_cycles_sha256": offset.sha256_array(bag_pred["score_cycles"].astype(np.int64)),
            "present_mask_sha256": offset.sha256_array(present),
            "a02_safe_array_hashes": a02_meta["safe_array_hashes"],
            "a02_predictions_json_sha256": a02_meta["prediction_json_sha256"],
            "a02_fit_receipt_sha256": a02_meta["receipt_sha256"],
            "pos_bag_prediction_bundle_sha256": _load_bag_prediction(origin)[0]["output_sha256"],
            "old_pos_prelabel_score_rows": len(p_p),
            "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
        }
        # Compute and freeze contrast logical hashes without reading any query labels.
        manifest, _, _, payload = _load_raw_origin(origin)
        matrices = payload["matrices"]
        widths = {}
        for matrix_arm in ("position", "shuffled_position"):
            for prefix in ("fit", "score"):
                base_m = sparse.csr_matrix(matrices[f"{prefix}_base"], dtype=np.float64)
                aug_m = sparse.csr_matrix(matrices[f"{prefix}_{matrix_arm}"], dtype=np.float64)
                z = contrast_block(aug_m, base_m.shape[1])
                widths[f"{prefix}_{matrix_arm}"] = {
                    "shape": list(z.shape), "sha256": offset.sha256_csr(z),
                    "base_width": int(base_m.shape[1]),
                    "augmented_sha256": offset.sha256_csr(aug_m),
                }
        origins[suffix]["contrast_matrices"] = widths
        origins[suffix]["matrix_manifest_sha256"] = legacy["origins"][suffix]["matrix_manifest_sha256"]
    a02_verification = _json_read(A02_REPORT)
    if a02_verification.get("status") != "PASS":
        raise ContrastPilotError("Historical A02 independent verification is not PASS")
    return {
        "train_sha256": offset.sha256_file(TRAIN), "train_rows": len(identity),
        "legacy_input_verification": legacy, "pos_bag_prelabel_sha256": offset.sha256_file(BAG_ROOT / "PRELABEL_VERIFICATION.json"),
        "pos_bag_predictions_ready_sha256": offset.sha256_file(BAG_ROOT / "PREDICTIONS_READY.json"),
        "a02_verification_sha256": offset.sha256_file(A02_REPORT),
        "a02_verification_status": a02_verification.get("status"),
        "origins": origins,
        "test_or_sample_access": False,
    }


def preflight() -> dict[str, Any]:
    if not TRAIN.is_file() or not LEDGER.is_file():
        raise ContrastPilotError("Official Train or Research Next ledger is missing")
    if _git("status", "--porcelain"):
        raise ContrastPilotError("Commit the implementation and run preflight from a clean tree")
    if _git("branch", "--show-current") != "main":
        raise ContrastPilotError("Expected the authorized main branch")
    events, snapshot = _ledger_snapshot()
    if snapshot != BASE_SNAPSHOT or offset.sha256_file(LEDGER) != EXPECTED_LEDGER_SHA:
        raise ContrastPilotError(f"Live ledger differs from authorized pre-amendment checkpoint: {snapshot}")
    if any(e.get("fit_id") in (*FIT_IDS, E79_ID) for e in events):
        raise ContrastPilotError("A POS_CONTRAST_V1 fit ID already appears in the ledger")
    parent = _verify_parent_inputs()
    resource = offset._resource_snapshot(1024**3, initial=True)
    source_hashes = _source_hashes()
    result = {
        "schema_version": 1, "card_id": CARD_ID, "status": "PASS_READY_TO_FREEZE",
        "source_commit": _git("rev-parse", "HEAD"), "source_hashes": source_hashes,
        "input_sha256": offset.sha256_file(TRAIN), "ledger_sha256": offset.sha256_file(LEDGER),
        "ledger_snapshot": snapshot, "parent_verification": parent, "resource": resource,
        "fit_ids": list(FIT_IDS), "conditional_e79_id": E79_ID,
        "query_labels_opened": False, "test_or_sample_access": False,
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(RUN_ROOT / "PREFLIGHT.json", result)
    return result


def freeze() -> dict[str, Any]:
    pre = _json_read(RUN_ROOT / "PREFLIGHT.json")
    if pre.get("status") != "PASS_READY_TO_FREEZE" or pre.get("query_labels_opened") is not False:
        raise ContrastPilotError("A passing label-closed preflight is required")
    if pre["source_commit"] != _git("rev-parse", "HEAD") or pre["source_hashes"] != _source_hashes():
        raise ContrastPilotError("Source/recipe changed after preflight")
    if offset.sha256_file(LEDGER) != pre["ledger_sha256"]:
        raise ContrastPilotError("Live ledger changed after preflight")
    bag_freeze = _parent_freeze()
    body = {
        "schema_version": 1, "card_id": CARD_ID, "source_commit": pre["source_commit"],
        "source_hashes": pre["source_hashes"], "recipe_sha256": offset.sha256_file(RECIPE),
        "protocol_sha256": offset.sha256_file(PROTOCOL), "error_audit_sha256": offset.sha256_file(ERROR_AUDIT),
        "input_sha256": pre["input_sha256"], "input_rows": 50608,
        "pre_reservation_ledger_sha256": pre["ledger_sha256"],
        "pre_reservation_ledger_bytes": LEDGER.stat().st_size,
        "pre_reservation_ledger_snapshot": pre["ledger_snapshot"],
        "parent_freeze_signature": bag_freeze["freeze_signature"],
        "parent_bag_freeze": bag_freeze,
        "parent_verification": pre["parent_verification"],
        "fit_ids": list(FIT_IDS), "conditional_e79_id": E79_ID,
        "origins": pre["parent_verification"]["origins"],
        "optimizer": _json_read(RECIPE)["optimizer"], "query_labels_opened": False,
        "test_or_sample_access": False,
    }
    body["freeze_signature"] = _sha_obj(body)
    _atomic_json(RUN_ROOT / "FREEZE.json", body)
    _atomic_json(REPORT_ROOT / "PREFLIGHT.json", pre)
    _atomic_json(REPORT_ROOT / "FREEZE.json", body)
    return body


def _read_signed_freeze(path: Path) -> dict[str, Any]:
    data = _json_read(path)
    signature = data.get("freeze_signature")
    body = {key: value for key, value in data.items() if key != "freeze_signature"}
    if signature != _sha_obj(body):
        raise ContrastPilotError(f"Freeze signature mismatch: {path.name}")
    return data


def _read_base_freeze() -> dict[str, Any]:
    run_freeze = _read_signed_freeze(RUN_ROOT / "FREEZE.json")
    report_freeze = _read_signed_freeze(REPORT_ROOT / "FREEZE.json")
    if run_freeze != report_freeze:
        raise ContrastPilotError("Run-root and reported predecessor freezes differ")
    return run_freeze


def _validate_successor_freeze(*, require_rebind: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _read_base_freeze()
    if not PREFIT_FREEZE_REPORT.is_file():
        if _source_hashes() != base["source_hashes"]:
            raise ContrastPilotError("Source changed after the original freeze; successor freeze is required")
        return base, base
    active = _read_signed_freeze(RUN_ROOT / "PREFIT_FREEZE.json")
    if active != _read_signed_freeze(PREFIT_FREEZE_REPORT):
        raise ContrastPilotError("Reported and run-root prefit freezes differ")
    if (active.get("schema_version") != 2
            or active.get("prior_freeze_signature") != base["freeze_signature"]
            or active.get("prior_source_commit") != base["source_commit"]
            or active.get("source_hashes") != _source_hashes()):
        raise ContrastPilotError("Successor freeze does not match the committed source or original freeze")
    attempt = active.get("prefit_dispatch_attempt", {})
    if (active.get("prefit_amendment_id") != PREFIT_REBIND_ID
            or attempt.get("attempt_source_commit") != FAILED_DISPATCH_COMMIT
            or attempt.get("result") != "PRE_DISPATCH_REJECTED_ZERO_FITS"
            or attempt.get("error") != FAILED_DISPATCH_ERROR
            or attempt.get("estimator_fit_calls") != 0
            or attempt.get("fit_started_ids") != [] or attempt.get("terminal_fit_ids") != []
            or attempt.get("ledger_sha256_after_attempt") != active.get("reservation_checkpoint_ledger_sha256")):
        raise ContrastPilotError("Successor freeze lacks exact zero-fit recovery evidence")
    if _git("merge-base", "--is-ancestor", active["source_commit"], "HEAD") != "":
        raise ContrastPilotError("Successor source commit is not an ancestor of the current checkout")
    for key, value in base.items():
        if key in {"schema_version", "source_commit", "source_hashes", "freeze_signature"}:
            continue
        if active.get(key) != value:
            raise ContrastPilotError(f"Successor freeze changed original frozen field: {key}")
    for rel in ("recipe_sha256", "protocol_sha256", "error_audit_sha256", "input_sha256"):
        if active.get(rel) != base.get(rel):
            raise ContrastPilotError(f"Successor freeze changed protected identity: {rel}")
    if (offset.sha256_file(RECIPE) != base["recipe_sha256"]
            or offset.sha256_file(PROTOCOL) != base["protocol_sha256"]
            or offset.sha256_file(ERROR_AUDIT) != base["error_audit_sha256"]):
        raise ContrastPilotError("Frozen recipe, protocol, or source audit changed")
    if require_rebind:
        _verify_prefit_rebind(base, active)
    return base, active


def prepare_prefit_amendment() -> dict[str, Any]:
    """Freeze a narrow source fix without changing recipe, gates, inputs or fit slots."""
    if _git("status", "--porcelain"):
        raise ContrastPilotError("Commit the prefit adapter fix before successor freeze")
    if _git("branch", "--show-current") != "main":
        raise ContrastPilotError("Prefit successor freeze requires the authorized main branch")
    base = _read_base_freeze()
    if PREFIT_FREEZE_REPORT.exists():
        _, active = _validate_successor_freeze(require_rebind=False)
        return active
    events, snapshot = _ledger_snapshot()
    ledger_sha = offset.sha256_file(LEDGER)
    ledger_bytes = LEDGER.stat().st_size
    expected_reserved = {
        "event_rows": 335, "global_cap": 137, "effective_category_caps": NEW_CAPS,
        "active_reservations_by_category": {"continuation": 50, "control_replay": 2, "pilot": 81, "technical_retry": 1},
        "active_reservations": 134, "fit_starts": 129, "fit_saved": 123,
        "fit_failed": 6, "fit_pending": 0,
    }
    own = [event for event in events if event.get("fit_id") in (*FIT_IDS, E79_ID)]
    starts = [event for event in own if event.get("event") == "FIT_STARTED"]
    terminal = [event for event in own if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    artifacts = [path for fit_id in FIT_IDS for path in (
        RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json") if path.exists()]
    amendments = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT" and event.get("amendment_id") == AMENDMENT_ID]
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED" and event.get("reservation_key") in {PILOT_RESERVATION_KEY, E79_RESERVATION_KEY}]
    if any(event.get("event") == "PREFIT_SOURCE_REBIND" and event.get("card_id") == CARD_ID for event in events):
        raise ContrastPilotError("A prefit rebind event already exists without its matching successor record")
    if (snapshot != expected_reserved or ledger_sha != _json_read(RUN_ROOT / "RESERVATION.json").get("ledger_sha256")
            or starts or terminal or artifacts or len(amendments) != 1 or len(reservations) != 2
            or any(event.get("freeze_signature") != base["freeze_signature"] for event in (*amendments, *reservations))):
        raise ContrastPilotError("The exact reservations or zero-start prefit state changed")
    _git("merge-base", "--is-ancestor", FAILED_DISPATCH_COMMIT, "HEAD")
    attempt = {
        "attempt_source_commit": FAILED_DISPATCH_COMMIT,
        "command": "python -m scripts.optimization_v3_positional_contrast_v1 run-all",
        "result": "PRE_DISPATCH_REJECTED_ZERO_FITS",
        "error": FAILED_DISPATCH_ERROR,
        "estimator_fit_calls": 0, "fit_started_ids": [], "terminal_fit_ids": [],
        "ledger_sha256_after_attempt": ledger_sha,
        "recovery": "callback now accepts and validates the active successor freeze signature",
    }
    body = {key: value for key, value in base.items() if key != "freeze_signature"}
    body.update({
        "schema_version": 2, "source_commit": _git("rev-parse", "HEAD"),
        "source_hashes": _source_hashes(), "prior_freeze_signature": base["freeze_signature"],
        "prior_source_commit": base["source_commit"], "prefit_dispatch_attempt": attempt,
        "reservation_checkpoint_ledger_sha256": ledger_sha,
        "reservation_checkpoint_ledger_bytes": ledger_bytes,
        "reservation_checkpoint_snapshot": snapshot,
        "prefit_amendment_id": PREFIT_REBIND_ID,
    })
    body["freeze_signature"] = _sha_obj(body)
    _atomic_json(RUN_ROOT / "PREFIT_FREEZE.json", body)
    _atomic_json(PREFIT_FREEZE_REPORT, body)
    return body


def _verify_prefit_rebind(base: Mapping[str, Any], active: Mapping[str, Any]) -> dict[str, Any]:
    record = _json_read(PREFIT_AMENDMENT_REPORT)
    event = record.get("event")
    if not isinstance(event, dict) or record.get("event_sha256") != _sha_obj(event):
        raise ContrastPilotError("Prefit source rebind report is missing or corrupt")
    ledger_bytes = LEDGER.read_bytes()
    prefix_bytes = int(active["reservation_checkpoint_ledger_bytes"])
    prefix = ledger_bytes[:prefix_bytes]
    tail = ledger_bytes[prefix_bytes:]
    first_tail_line = tail.splitlines()[0] if tail else b""
    if (hashlib.sha256(prefix).hexdigest() != active["reservation_checkpoint_ledger_sha256"]
            or json.loads(first_tail_line.decode("utf-8")) != event):
        raise ContrastPilotError("Prefit rebind is not append-only after the frozen reservation checkpoint")
    events, _ = _ledger_snapshot()
    matches = [item for item in events if item.get("event") == "PREFIT_SOURCE_REBIND" and item.get("card_id") == CARD_ID]
    if len(matches) != 1 or matches[0] != event:
        raise ContrastPilotError("Append-only prefit source rebind event differs from report")
    expected = {
        "amendment_id": AMENDMENT_ID, "prefit_amendment_id": PREFIT_REBIND_ID,
        "prior_freeze_signature": base["freeze_signature"], "freeze_signature": active["freeze_signature"],
        "prior_source_commit": base["source_commit"], "source_commit": active["source_commit"],
        "fit_ids": list(FIT_IDS), "conditional_fit_id": E79_ID,
        "reservation_keys": [PILOT_RESERVATION_KEY, E79_RESERVATION_KEY],
        "added_fit_slots": 0, "estimator_fit_calls_before_rebind": 0,
        "fit_starts_before_rebind": [], "terminal_fit_ids_before_rebind": [],
        "ledger_sha256_before_rebind": active["reservation_checkpoint_ledger_sha256"],
        "ledger_snapshot_before_rebind": active["reservation_checkpoint_snapshot"],
        "caps_unchanged": True, "recipe_gates_populations_unchanged": True,
    }
    if any(event.get(key) != value for key, value in expected.items()):
        raise ContrastPilotError("Prefit source rebind changed authorization or lacks zero-start proof")
    if record.get("status") != "PASS_REBOUND_NO_NEW_SLOTS":
        raise ContrastPilotError("Prefit rebind status is not passing")
    return record


def rebind_prefit_amendment() -> dict[str, Any]:
    base, active = _validate_successor_freeze(require_rebind=False)
    events, snapshot = _ledger_snapshot()
    if offset.sha256_file(LEDGER) != active["reservation_checkpoint_ledger_sha256"] or snapshot != active["reservation_checkpoint_snapshot"]:
        raise ContrastPilotError("Live ledger moved after successor freeze and before source rebind")
    if (_git("status", "--porcelain")
            or _git("merge-base", "--is-ancestor", active["source_commit"], "HEAD") != ""):
        raise ContrastPilotError("Commit successor freeze and keep its source commit in ancestry before rebind")
    own = [event for event in events if event.get("fit_id") in (*FIT_IDS, E79_ID)]
    if any(event.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"} for event in own):
        raise ContrastPilotError("A fit event appeared before the source rebind")
    amendment = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT" and event.get("amendment_id") == AMENDMENT_ID]
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED" and event.get("reservation_key") in {PILOT_RESERVATION_KEY, E79_RESERVATION_KEY}]
    if (len(amendment) != 1 or len(reservations) != 2
            or amendment[0].get("freeze_signature") != base["freeze_signature"]
            or any(item.get("freeze_signature") != base["freeze_signature"] for item in reservations)):
        raise ContrastPilotError("Existing reservation lineage is not the expected predecessor freeze")
    existing = [event for event in events if event.get("event") == "PREFIT_SOURCE_REBIND" and event.get("card_id") == CARD_ID]
    if existing:
        return _verify_prefit_rebind(base, active)
    event = {
        "event": "PREFIT_SOURCE_REBIND", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "amendment_id": AMENDMENT_ID, "prefit_amendment_id": PREFIT_REBIND_ID,
        "prior_freeze_signature": base["freeze_signature"], "freeze_signature": active["freeze_signature"],
        "prior_source_commit": base["source_commit"], "source_commit": active["source_commit"],
        "fit_ids": list(FIT_IDS), "conditional_fit_id": E79_ID,
        "reservation_keys": [PILOT_RESERVATION_KEY, E79_RESERVATION_KEY],
        "ledger_sha256_before_rebind": offset.sha256_file(LEDGER),
        "ledger_snapshot_before_rebind": snapshot,
        "fit_starts_before_rebind": [], "terminal_fit_ids_before_rebind": [],
        "estimator_fit_calls_before_rebind": 0, "added_fit_slots": 0,
        "caps_unchanged": True, "recipe_gates_populations_unchanged": True,
        "reason": "Narrow pre-fit callback signature/validation repair; existing reservations remain the sole source of fit slots.",
    }
    if event["ledger_sha256_before_rebind"] != active["reservation_checkpoint_ledger_sha256"]:
        raise ContrastPilotError("Ledger prefix changed before append-only source rebind")
    persisted_event = offset._append_ledger(event, path=LEDGER, official=True)
    record = {"status": "PASS_REBOUND_NO_NEW_SLOTS", "event": persisted_event,
              "event_sha256": _sha_obj(persisted_event), "ledger_sha256_after_rebind": offset.sha256_file(LEDGER),
              "active_freeze_signature": active["freeze_signature"], "added_fit_slots": 0}
    _atomic_json(RUN_ROOT / "PREFIT_AMENDMENT.json", record)
    _atomic_json(PREFIT_AMENDMENT_REPORT, record)
    return record


def load_freeze() -> dict[str, Any]:
    base, active = _validate_successor_freeze(require_rebind=True)
    if offset.sha256_file(RECIPE) != base["recipe_sha256"]:
        raise ContrastPilotError("Frozen recipe changed")
    if offset.sha256_file(PROTOCOL) != base["protocol_sha256"] or offset.sha256_file(ERROR_AUDIT) != base["error_audit_sha256"]:
        raise ContrastPilotError("Frozen protocol or root error audit changed")
    if active["source_hashes"] != _source_hashes():
        raise ContrastPilotError("Active source hashes changed")
    return active


def _check_ledger(*, allow_current: str | None = None,
                  freeze_signature: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    freeze_data = load_freeze()
    base_freeze = _read_base_freeze()
    if freeze_signature is not None and freeze_signature != freeze_data["freeze_signature"]:
        raise ContrastPilotError("Budget callback received a different freeze signature")
    events, snapshot = _ledger_snapshot()
    prefix_bytes = int(freeze_data["pre_reservation_ledger_bytes"])
    with LEDGER.open("rb") as stream:
        prefix = stream.read(prefix_bytes)
    if hashlib.sha256(prefix).hexdigest() != freeze_data["pre_reservation_ledger_sha256"]:
        raise ContrastPilotError("Append-only pre-amendment ledger prefix changed")
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    pilot_res = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == PILOT_RESERVATION_KEY]
    e79_res = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == E79_RESERVATION_KEY]
    if len(amendments) != 1 or len(pilot_res) != 1 or len(e79_res) != 1:
        raise ContrastPilotError("Cap amendment and two exact reservations are incomplete or duplicated")
    amendment, pilot, e79 = amendments[0], pilot_res[0], e79_res[0]
    if (amendment.get("card_id") != CARD_ID or amendment.get("prior_global_cap") != 132 or amendment.get("global_cap") != 137
            or amendment.get("prior_effective_caps") != BASE_CAPS or amendment.get("effective_caps") != NEW_CAPS
            or amendment.get("freeze_signature") != base_freeze["freeze_signature"]
            or amendment.get("source_commit") != base_freeze["source_commit"]):
        raise ContrastPilotError("Live budget amendment differs from freeze")
    if (pilot.get("fit_ids") != list(FIT_IDS) or pilot.get("fits") != 4 or pilot.get("category") != "pilot"
            or pilot.get("effective_category_cap") != 84 or pilot.get("freeze_signature") != base_freeze["freeze_signature"]):
        raise ContrastPilotError("Pilot fit reservation differs from frozen four-ID order")
    if (e79.get("fit_ids") != [E79_ID] or e79.get("fits") != 1 or e79.get("category") != "continuation"
            or e79.get("effective_category_cap") != 50 or e79.get("stage") != "CONDITIONAL_E79_AFTER_SOL_REVIEW"
            or e79.get("freeze_signature") != base_freeze["freeze_signature"]):
        raise ContrastPilotError("Conditional E79 reservation differs from protocol")
    released = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == E79_RESERVATION_KEY]
    if len(released) > 1 or (released and (released[0].get("unused_fits") != 1 or released[0].get("fit_ids") != [E79_ID])):
        raise ContrastPilotError("Conditional E79 release event is invalid")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    own_ids = [e.get("fit_id") for e in starts]
    terminal_ids = [e.get("fit_id") for e in (*saved, *failed)]
    if len(own_ids) != len(set(own_ids)) or len(terminal_ids) != len(set(terminal_ids)) or not set(terminal_ids).issubset(set(own_ids)):
        raise ContrastPilotError("Pilot start/terminal fit accounting is inconsistent")
    pending = [fit_id for fit_id in own_ids if fit_id not in set(terminal_ids)]
    if len(pending) > 1 or (allow_current is None and pending) or (allow_current is not None and any(x != allow_current for x in pending)):
        raise ContrastPilotError("Unexpected pending pilot fit")
    if (snapshot["global_cap"] != 137 or snapshot["effective_category_caps"] != NEW_CAPS
            or snapshot["fit_starts"] != 129 + len(starts)
            or snapshot["fit_saved"] != 123 + len(saved)
            or snapshot["fit_failed"] != 6 + len(failed)
            or snapshot["fit_pending"] != len(pending)
            or snapshot["active_reservations_by_category"]["pilot"] != 81
            or snapshot["active_reservations_by_category"]["continuation"] != (49 if released else 50)
            or snapshot["active_reservations_by_category"]["control_replay"] != 2
            or snapshot["active_reservations_by_category"]["technical_retry"] != 1
            or snapshot["fit_starts"] > 133):
        raise ContrastPilotError(f"Live ledger accounting drifted: {snapshot}")
    return events, snapshot


def reserve() -> dict[str, Any]:
    freeze_data = load_freeze()
    events, snapshot = _ledger_snapshot()
    existing = any(e.get("amendment_id") == AMENDMENT_ID for e in events)
    if existing:
        _check_ledger()
        return _json_read(RUN_ROOT / "RESERVATION.json")
    if offset.sha256_file(LEDGER) != freeze_data["pre_reservation_ledger_sha256"] or snapshot != BASE_SNAPSHOT:
        raise ContrastPilotError("Live ledger moved after frozen pre-reservation snapshot")
    _configure_offset_module()
    offset._append_ledger({
        "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
        "prior_global_cap": 132, "global_cap": 137,
        "prior_effective_caps": BASE_CAPS, "effective_caps": NEW_CAPS,
        "fit_ids": [*FIT_IDS, E79_ID], "source_commit": freeze_data["source_commit"],
        "freeze_signature": freeze_data["freeze_signature"],
        "pre_amendment_ledger_sha256": freeze_data["pre_reservation_ledger_sha256"],
        "reason": "Authorized four POS_CONTRAST_V1 pilot fits plus one conditional E79 fit after independent review by Sol; no retry allowance.",
    }, path=LEDGER, official=True)
    for category, fits, ids, reservation_key, stage, category_cap in (
        ("pilot", 4, list(FIT_IDS), PILOT_RESERVATION_KEY, STAGE, 84),
        ("continuation", 1, [E79_ID], E79_RESERVATION_KEY, "CONDITIONAL_E79_AFTER_SOL_REVIEW", 50),
    ):
        offset._append_ledger({
            "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
            "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
            "reservation_key": reservation_key, "category": category, "fits": fits,
            "fit_ids": ids, "stage": stage, "effective_category_cap": category_cap,
            "global_cap": 137, "source_commit": freeze_data["source_commit"],
            "freeze_signature": freeze_data["freeze_signature"],
        }, path=LEDGER, official=True)
    events, snapshot = _check_ledger()
    result = {
        "status": "RESERVED", "amendment_id": AMENDMENT_ID,
        "pilot_fit_ids": list(FIT_IDS), "conditional_e79_fit_id": E79_ID,
        "conditional_e79_start_authorized": False,
        "ledger_sha256": offset.sha256_file(LEDGER), "ledger_snapshot": snapshot,
        "events": [e for e in events if e.get("amendment_id") == AMENDMENT_ID],
        "freeze_signature": freeze_data["freeze_signature"],
    }
    _atomic_json(RUN_ROOT / "RESERVATION.json", result)
    _atomic_json(REPORT_ROOT / "RESERVATION.json", result)
    return result


def _configure_offset_module() -> None:
    offset.CARD_ID = CARD_ID
    offset.AMENDMENT_ID = AMENDMENT_ID
    offset.FIT_IDS = FIT_IDS
    offset.RESERVATION_KEY = PILOT_RESERVATION_KEY
    offset.STAGE = STAGE
    offset.LEDGER = LEDGER
    offset.RUN_ROOT = RUN_ROOT
    offset.REPORT_ROOT = REPORT_ROOT
    offset._load_origin_payload = _contrast_payload
    offset._check_live_budget = _check_ledger
    offset.verify_saved_inputs = _verify_parent_inputs
    offset.load_freeze = load_freeze


def run_all_fits() -> list[dict[str, Any]]:
    _configure_offset_module()
    freeze_data = load_freeze()
    _check_ledger()
    reservation = _json_read(RUN_ROOT / "RESERVATION.json")
    if reservation.get("pilot_fit_ids") != list(FIT_IDS):
        raise ContrastPilotError("Persisted reservation does not match the four fit IDs")
    output: list[dict[str, Any]] = []
    with offset._fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        for fit_id in FIT_IDS:
            events, _ = _check_ledger()
            if any(e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id for e in events):
                output.append({"fit_id": fit_id, "status": "failed_consumed_no_retry"})
                continue
            try:
                output.append(offset._run_fit(fit_id, freeze_data))
            except Exception as exc:
                now = offset._ledger_events(LEDGER)
                failed = any(e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id for e in now)
                started = any(e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id for e in now)
                if not (failed and started):
                    raise
                output.append({"fit_id": fit_id, "status": "failed_consumed_no_retry",
                               "error": f"{type(exc).__name__}: {exc}"})
    _atomic_json(RUN_ROOT / "FIT_SUMMARY.json", {"fit_order": list(FIT_IDS), "results": output,
                                                   "ledger_sha256": offset.sha256_file(LEDGER),
                                                   "snapshot": _ledger_snapshot()[1]})
    events, _ = _check_ledger()
    failures = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    terminal = {e.get("fit_id") for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") in FIT_IDS}
    if failures and set(FIT_IDS).issubset(terminal):
        _finalize_fit_failure(failures)
    return output


def _finalize_fit_failure(failures: list[dict[str, Any]]) -> None:
    freeze_data = load_freeze()
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "FAIL_FIT_CONVERGENCE_OR_INTERRUPTION",
        "qualified_candidate": False, "champion_change": False,
        "full_train_refit_authorized": False, "upload_authorized": False,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "input_sha256": freeze_data["input_sha256"], "query_labels_opened": False,
        "test_or_sample_access": False, "fit_ids": list(FIT_IDS), "failed_fit_events": failures,
        "ledger_snapshot": _ledger_snapshot()[1],
        "interpretation": "A pilot gamma estimator failed or was interrupted. The fit ID was consumed without retry. Query labels remained closed and no predictive metrics were evaluated.",
    }
    _atomic_json(RUN_ROOT / "RESULTS.json", result)
    _finalize_conditional_e79(False, result)
    report = "\n".join([
        "# POS_CONTRAST_V1 — fit failure checkpoint", "",
        "**Status:** `FAIL_FIT_CONVERGENCE_OR_INTERRUPTION`. No query labels or predictive metrics were opened.", "",
        "| Fit ID | Status |", "|---|---|",
        *[f"| {fit_id} | {'FAILED' if any(x.get('fit_id') == fit_id for x in failures) else 'SAVED'} |" for fit_id in FIT_IDS],
        "", "Failed/interrupted IDs were consumed and not retried. The conditional E79 slot is released. No full-Train fit, Test inference, or upload is authorized.", "",
    ])
    (REPORT_ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    completion = {"event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
                  "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
                  "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
                  "status": result["status"], "result_sha256": offset.sha256_file(RUN_ROOT / "RESULTS.json"),
                  "query_labels_opened": False, "test_or_sample_access": False,
                  "conditional_e79_started": False, "ledger_snapshot_at_completion": _ledger_snapshot()[1]}
    offset._append_ledger(completion, path=LEDGER, official=True)
    accounting = {"completion_event": completion, "ledger_sha256_after_completion": offset.sha256_file(LEDGER),
                  "ledger_snapshot_after_completion": _ledger_snapshot()[1], "fit_ids": list(FIT_IDS),
                  "conditional_e79_status": "RELEASED_FIT_FAILURE"}
    _atomic_json(RUN_ROOT / "ACCOUNTING.json", accounting)
    _atomic_json(REPORT_ROOT / "ACCOUNTING.json", accounting)


def _fit_gamma(fit_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = RUN_ROOT / "units" / f"{fit_id}.npz"
    receipt_path = RUN_ROOT / "receipts" / f"{fit_id}.json"
    receipt = _json_read(receipt_path)
    if offset.sha256_file(artifact) != receipt.get("artifact_sha256"):
        raise ContrastPilotError(f"Fit artifact hash mismatch: {fit_id}")
    with np.load(artifact, allow_pickle=False) as archive:
        gamma = np.asarray(archive["gamma"], dtype=np.float64).copy()
    if offset.sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise ContrastPilotError(f"Fit coefficient hash mismatch: {fit_id}")
    return gamma, receipt


def _fixed_route(q: np.ndarray, p_a: np.ndarray, p_p: np.ndarray, present: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_p = np.asarray(p_p, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    if not (len(q) == len(p_a) == len(p_p) == len(present)):
        raise ContrastPilotError("Policy arrays do not align")
    out = p_p.copy()
    out[present] = 0.5 * q[present] + 0.5 * p_a[present]
    return out


def build_predictions() -> dict[str, Any]:
    freeze_data = load_freeze()
    events, _ = _check_ledger()
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    if len(saved) != 4 or failed:
        raise ContrastPilotError("Prediction build requires all four pilot fits durably saved")
    output: dict[str, Any] = {}
    for origin in (67, 75):
        manifest, legacy, bag, payload = _contrast_payload(origin, freeze_data)
        matrices = payload["matrices"]
        base_score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
        z0 = np.asarray(bag.intercept_[0] + base_score @ bag.coef_[0], dtype=np.float64).reshape(-1)
        _, old = _load_bag_prediction(origin)
        _, a02 = _load_a02(origin)
        present = np.asarray(old["present_mask"], dtype=bool)
        p_p = np.asarray(old["prob_offset_real"], dtype=np.float64)
        p_a = np.asarray(a02["prediction"], dtype=np.float64)
        q_by_arm: dict[str, np.ndarray] = {}
        for arm, suffix in (("REAL", "position"), ("SHUFFLED", "shuffled_position")):
            gamma, receipt = _fit_gamma(f"{CARD_ID}_E{origin}_{arm}")
            augmented = sparse.csr_matrix(matrices[f"score_{suffix}"], dtype=np.float64)
            z = augmented[:, base_score.shape[1]:].tocsr()
            if offset.sha256_csr(z) != receipt["matrix"]["score_position_sha256"]:
                raise ContrastPilotError(f"E{origin} {arm} score matrix differs from fit receipt")
            q = offset.offset_probability(z, z0, gamma)
            if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
                raise ContrastPilotError(f"E{origin} {arm} q probabilities are invalid")
            q_by_arm[arm] = q
        real = _fixed_route(q_by_arm["REAL"], p_a, p_p, present)
        shuffled = _fixed_route(q_by_arm["SHUFFLED"], p_a, p_p, present)
        mix50 = _fixed_route(p_p, p_a, p_p, present)
        dest = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        offset._atomic_npz(
            dest, origin=np.asarray([origin], dtype=np.int64),
            fit_rows=np.asarray(old["fit_rows"], dtype=np.int64),
            score_rows=np.asarray(old["score_rows"], dtype=np.int64),
            score_record_ids=np.asarray(old["score_record_ids"]).astype(str),
            score_cycles=np.asarray(old["score_cycles"], dtype=np.int64),
            present_mask=present, prob_q_real=q_by_arm["REAL"], prob_q_shuffled=q_by_arm["SHUFFLED"],
            prob_pP=p_p, prob_a02=p_a, prob_mix50=mix50,
            prob_real_blend=real, prob_shuffled_blend=shuffled,
            prob_bag=np.asarray(old["prob_bag"], dtype=np.float64),
            matrix_manifest_sha256=np.asarray([offset.sha256_file(PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json")]),
            a02_safe_probability_sha256=np.asarray([offset.sha256_array(p_a)]),
        )
        output[f"E{origin}"] = {
            "path": dest.relative_to(ROOT).as_posix(), "sha256": offset.sha256_file(dest),
            "rows": len(real), "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
            "score_rows_sha256": offset.sha256_array(old["score_rows"]),
            "score_record_ids_sha256": offset.sha256_array(np.asarray(old["score_record_ids"]).astype(str)),
            "score_cycles_sha256": offset.sha256_array(old["score_cycles"].astype(np.int64)),
        }
    ready = {
        "schema_version": 1, "card_id": CARD_ID, "freeze_signature": freeze_data["freeze_signature"],
        "source_commit": freeze_data["source_commit"], "input_sha256": freeze_data["input_sha256"],
        "predictions": output, "all_four_fit_receipts_durable": True,
        "query_labels_opened": False, "test_or_sample_access": False,
    }
    _atomic_json(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    return ready


def verify_prelabel() -> dict[str, Any]:
    freeze_data = load_freeze()
    result = verify_prelabel_outputs(ROOT, RUN_ROOT, freeze_data)
    if result.get("status") != "PASS" or result.get("query_labels_read") is not False:
        raise ContrastPilotError("Independent prelabel verification did not pass with labels closed")
    _atomic_json(RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
    _atomic_json(REPORT_ROOT / "PRELABEL_VERIFICATION.json", result)
    return result


def score() -> dict[str, Any]:
    freeze_data = load_freeze()
    prelabel_path = RUN_ROOT / "PRELABEL_VERIFICATION.json"
    prelabel = _json_read(prelabel_path)
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    if prelabel.get("status") != "PASS" or prelabel.get("query_labels_read") is not False:
        raise ContrastPilotError("Metrics stay closed until independent prelabel verification PASS")
    if prelabel.get("freeze_signature") != freeze_data["freeze_signature"] or prelabel.get("predictions_ready_sha256") != offset.sha256_file(ready_path):
        raise ContrastPilotError("Prelabel verification is not bound to current prediction outputs")
    events, snapshot = _check_ledger()
    if len([e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]) != 4:
        raise ContrastPilotError("All four durable fits are required before outcome read")
    # First and only query-label access in this campaign: official Train validation rows.
    labels_frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "artifact_signal"], dtype={"record_id": "string"})
    origin_results: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for origin in (67, 75):
        path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        with np.load(path, allow_pickle=False) as a:
            pred = {k: np.asarray(a[k]).copy() for k in a.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        sub = labels_frame.iloc[rows]
        ids = sub["record_id"].astype(str).to_numpy()
        if not np.array_equal(ids, pred["score_record_ids"].astype(str)):
            raise ContrastPilotError(f"E{origin} score IDs changed before metric read")
        y = pd.to_numeric(sub["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        cycles = pd.to_numeric(sub["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        _, a02 = _load_a02(origin, include_groups=True)
        if not np.array_equal(a02["score_rows"], rows):
            raise ContrastPilotError(f"E{origin} group metadata row alignment changed")
        views = ({"H12_E67": (cycles >= 68) & (cycles <= 79),
                  "W4": (cycles >= 68) & (cycles <= 71),
                  "W5": (cycles >= 72) & (cycles <= 75),
                  "W6": (cycles >= 76) & (cycles <= 79)} if origin == 67
                 else {"W6": (cycles >= 76) & (cycles <= 79)})
        policies = {
            "contrast_real": pred["prob_real_blend"],
            "contrast_shuffled": pred["prob_shuffled_blend"],
            "MIX50": pred["prob_mix50"], "pP_oldPOS": pred["prob_pP"],
            "bag": pred["prob_bag"],
        }
        per_view: dict[str, Any] = {}
        subgroup_rows: list[dict[str, Any]] = []
        cycle_rows: list[dict[str, Any]] = []
        for view, mask in views.items():
            view_metrics = {name: offset._metric(y[mask], np.asarray(prob, dtype=np.float64)[mask])
                            for name, prob in policies.items()}
            present_view = mask & np.asarray(pred["present_mask"], dtype=bool)
            view_metrics["contrast_q_real_present"] = offset._metric(y[present_view], pred["prob_q_real"][present_view])
            view_metrics["contrast_q_shuffled_present"] = offset._metric(y[present_view], pred["prob_q_shuffled"][present_view])
            view_metrics["A02_observed_present"] = offset._metric(y[present_view], pred["prob_a02"][present_view])
            view_delta = {
                "real_minus_mix50": view_metrics["contrast_real"]["macro_f1"] - view_metrics["MIX50"]["macro_f1"],
                "shuffled_minus_mix50": view_metrics["contrast_shuffled"]["macro_f1"] - view_metrics["MIX50"]["macro_f1"],
                "real_minus_shuffled": view_metrics["contrast_real"]["macro_f1"] - view_metrics["contrast_shuffled"]["macro_f1"],
            }
            per_view[view] = {"rows": int(mask.sum()), "metrics": view_metrics, "deltas": view_delta}
            if origin == 67 and view in {"W4", "W5", "W6"}:
                gate = {"gate": f"E67_{view}_harm_vs_MIX50_max_0.003", "delta": view_delta["real_minus_mix50"],
                        "limit": -0.003, "pass": bool(view_delta["real_minus_mix50"] >= -0.003)}
                all_gates.append(gate)
            if (origin == 67 and view == "H12_E67") or (origin == 75 and view == "W6"):
                all_gates.extend([
                    {"gate": f"E{origin}_{view}_real_gain_vs_MIX50_min_0.0005", "delta": view_delta["real_minus_mix50"],
                     "limit": 0.0005, "pass": bool(view_delta["real_minus_mix50"] >= 0.0005)},
                    {"gate": f"E{origin}_{view}_real_gain_vs_shuffled_min_0", "delta": view_delta["real_minus_shuffled"],
                     "limit": 0.0, "pass": bool(view_delta["real_minus_shuffled"] >= 0.0)},
                ])
            groups: list[tuple[str, np.ndarray]] = [
                ("availability_present", np.asarray(pred["present_mask"], dtype=bool)),
                ("availability_missing", ~np.asarray(pred["present_mask"], dtype=bool)),
                ("entry_friction_0", np.asarray(a02["entry_friction"] == 0)),
                ("entry_friction_1", np.asarray(a02["entry_friction"] == 1)),
            ]
            for quartile in range(4):
                groups.append((f"text_load_Q{quartile+1}", np.asarray(a02["text_load_group"] == quartile)))
            for group, group_mask in groups:
                use = mask & group_mask
                n = int(use.sum())
                positives = int(y[use].sum())
                negatives = n - positives
                support = n >= 200 and positives >= 30 and negatives >= 30
                if support:
                    mr = offset._metric(y[use], np.asarray(pred["prob_real_blend"], dtype=np.float64)[use])
                    mb = offset._metric(y[use], np.asarray(pred["prob_mix50"], dtype=np.float64)[use])
                    delta_f1 = mr["macro_f1"] - mb["macro_f1"]
                    delta_recall = mr["positive_recall"] - mb["positive_recall"]
                    guard = delta_f1 >= -0.01 and delta_recall >= -0.02
                    all_gates.append({"gate": f"E{origin}_{view}_{group}_subgroup_guard", "delta_macro_f1": delta_f1,
                                      "delta_positive_recall": delta_recall, "support": n, "positive": positives,
                                      "negative": negatives, "pass": bool(guard)})
                    subgroup_rows.append({"view": view, "group": group, "status": "SUPPORTED", "n": n,
                                          "positive": positives, "negative": negatives,
                                          "real_macro_f1": mr["macro_f1"], "mix50_macro_f1": mb["macro_f1"],
                                          "delta_macro_f1": delta_f1, "real_positive_recall": mr["positive_recall"],
                                          "mix50_positive_recall": mb["positive_recall"],
                                          "delta_positive_recall": delta_recall})
                else:
                    subgroup_rows.append({"view": view, "group": group, "status": "UNKNOWN_UNSUPPORTED",
                                          "n": n, "positive": positives, "negative": negatives})
        for cycle in sorted(np.unique(cycles)):
            cm = cycles == cycle
            real_f1 = offset._metric(y[cm], pred["prob_real_blend"][cm])["macro_f1"]
            mix_f1 = offset._metric(y[cm], pred["prob_mix50"][cm])["macro_f1"]
            cycle_rows.append({"cycle": int(cycle), "n": int(cm.sum()), "real_macro_f1": real_f1,
                               "mix50_macro_f1": mix_f1, "delta": real_f1 - mix_f1})
        origin_results[f"E{origin}"] = {"rows": len(rows), "views": per_view,
                                         "subgroups": subgroup_rows, "per_cycle": cycle_rows,
                                         "query_labels_opened": True}
    passed = bool(all(g.get("pass") is True for g in all_gates))
    score_body = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "PILOT_PASS_PENDING_SOL_REVIEW" if passed else "PILOT_GATE_FAILED",
        "qualified_candidate": passed, "champion_change": False,
        "full_train_refit_authorized": False, "upload_authorized": False,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "input_sha256": freeze_data["input_sha256"], "query_labels_opened": True,
        "test_or_sample_access": False, "score_source": "official Classification Train temporal query rows only",
        "origins": origin_results, "gate_checks": all_gates, "all_gates_pass": passed,
        "ledger_snapshot_before_completion": snapshot,
        "fit_ids": list(FIT_IDS), "fit_failures": [],
        "limitations": ["E67/E75 overlap in W6; adaptive follow-up, not fresh holdout evidence.",
                         "Common-mode removal is not pure bag removal because sublinear TF is applied within thirds."],
    }
    _atomic_json(RUN_ROOT / "RESULTS.json", score_body)
    verification = verify_scored_results(ROOT, RUN_ROOT, freeze_data, score_body)
    _atomic_json(RUN_ROOT / "QA.json", verification)
    score_body["independent_verification_sha256"] = offset.sha256_file(RUN_ROOT / "QA.json")
    _atomic_json(RUN_ROOT / "RESULTS.json", score_body)
    _write_final_report(score_body, verification)
    _finalize_conditional_e79(passed, score_body)
    events, final_snapshot = _ledger_snapshot()
    result_sha = offset.sha256_file(RUN_ROOT / "RESULTS.json")
    completion = {
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
        "status": score_body["status"], "result_sha256": result_sha,
        "query_labels_opened": True, "test_or_sample_access": False,
        "conditional_e79_started": False, "ledger_snapshot_at_completion": final_snapshot,
    }
    offset._append_ledger(completion, path=LEDGER, official=True)
    accounting = {"completion_event": completion, "completion_event_sha256": _sha_obj(completion),
                  "ledger_sha256_after_completion": offset.sha256_file(LEDGER),
                  "ledger_snapshot_after_completion": _ledger_snapshot()[1],
                  "fit_ids": list(FIT_IDS), "conditional_e79_id": E79_ID,
                  "conditional_e79_status": "RESERVED_PENDING_SOL_REVIEW" if passed else "RELEASED_GATE_FAILED"}
    _atomic_json(RUN_ROOT / "ACCOUNTING.json", accounting)
    _atomic_json(REPORT_ROOT / "RESULTS.json", score_body)
    _atomic_json(REPORT_ROOT / "QA.json", verification)
    _atomic_json(REPORT_ROOT / "ACCOUNTING.json", accounting)
    return score_body


def _finalize_conditional_e79(passed: bool, result: Mapping[str, Any]) -> None:
    if passed:
        return
    events, _ = _check_ledger()
    if any(e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == E79_RESERVATION_KEY for e in events):
        return
    offset._append_ledger({
        "event": "RESERVATION_UNUSED_RELEASED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
        "reservation_key": E79_RESERVATION_KEY, "category": "continuation",
        "unused_fits": 1, "fit_ids": [E79_ID],
        "reason": f"Pilot ended {result['status']}; E79 requires a passing pilot and separate Sol review.",
        "freeze_signature": result["freeze_signature"],
    }, path=LEDGER, official=True)


def _write_final_report(result: Mapping[str, Any], verification: Mapping[str, Any]) -> None:
    _write_visuals(result)
    lines = [
        "# POS_CONTRAST_V1 — pilot results", "",
        f"**Status:** `{result['status']}`. The pilot is diagnostic and does not alter the champion.", "",
        "## Frozen question and comparison", "",
        "Does removing equal-across-thirds coefficient freedom improve the fixed positional/A02 route? The real candidate uses `0.5 * q + 0.5 * pA` on present-stream rows; missing-stream rows follow the original pP fallback exactly. The shuffled arm is a matched-capacity control. Primary comparisons are against the frozen MIX50 policy.", "",
        "The follow-up was selected after earlier POS/A02/MIX50 results and error patterns were visible. E67/E75 are overlapping sensitivity views, not fresh or outcome-blind holdouts.", "",
        "## Primary results", "",
        "| View | n | Real Macro F1 | MIX50 | Δ real−MIX50 | Shuffled | Δ real−shuffled |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for origin, views in (("E67", ("H12_E67", "W4", "W5", "W6")), ("E75", ("W6",))):
        for view in views:
            item = result["origins"][origin]["views"][view]
            m = item["metrics"]
            d = item["deltas"]
            lines.append(f"| {origin} {view} | {item['rows']} | {m['contrast_real']['macro_f1']:.6f} | {m['MIX50']['macro_f1']:.6f} | {d['real_minus_mix50']:+.6f} | {m['contrast_shuffled']['macro_f1']:.6f} | {d['real_minus_shuffled']:+.6f} |")
    lines += ["", "## Gates and subgroup checks", "", "| Gate | Result | Threshold | Status |", "|---|---:|---:|---|"]
    for gate in result["gate_checks"]:
        value = gate.get("delta", gate.get("delta_macro_f1", "UNKNOWN"))
        limit = gate.get("limit", "-0.01 F1 / -0.02 recall")
        lines.append(f"| {gate['gate']} | {value} | {limit} | {'PASS' if gate.get('pass') else 'FAIL'} |")
    unknown = sum(1 for data in result["origins"].values() for row in data["subgroups"] if row["status"] == "UNKNOWN_UNSUPPORTED")
    lines += ["", f"Unsupported subgroup views are explicitly `UNKNOWN_UNSUPPORTED` ({unknown}); none are treated as passing.",
              "", "## Interpretation and limitations", "",
              "The centered projection removes equal coefficient freedom across three position blocks. It does not remove all bag information: sublinear TF is applied within each third, so the common direction is not exactly an unordered bag vector. This experiment tests a fixed capacity restriction and does not establish that common-mode contribution is noise.",
              "", "The result applies only to the specified Train temporal windows and frozen route. It does not authorize a full-Train fit, Test inference, Kaggle upload, or a follow-up experiment. E67 W6 overlaps E75 W6.",
              "", "## QA", "",
              f"Independent prelabel and metric verification: `{verification.get('status')}`. Synthetic end-to-end QA was completed before the official fit. Official estimator fits: 4 planned, no retries. Conditional E79 start: not performed.",
        "", "## Visual summary", "",
        "The first figure summarizes the real-route Macro F1 difference from MIX50 across the frozen evaluation views; the second shows the corresponding per-cycle difference. E67 W6 and E75 W6 overlap in cycles 76–79 and must not be read as independent replications.",
        "", "![Fixed-route Macro F1 differences across views](primary_gates.png)",
        "", "![Per-cycle real-route Macro F1 difference versus MIX50](per_cycle_delta.png)",
              "", f"Source commit: `{result['source_commit']}`; freeze signature: `{result['freeze_signature']}`.", ""]
    (REPORT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_visuals(result: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    view_keys = (("E67", "H12_E67"), ("E67", "W4"), ("E67", "W5"),
                 ("E67", "W6"), ("E75", "W6"))
    labels = [f"{origin} {view}" for origin, view in view_keys]
    deltas = [result["origins"][origin]["views"][view]["deltas"]["real_minus_mix50"]
              for origin, view in view_keys]
    thresholds = [0.0005 if (origin, view) in (("E67", "H12_E67"), ("E75", "W6")) else -0.003
                  for origin, view in view_keys]
    colors = ["#2f6b9a" if delta >= threshold else "#b64949"
              for delta, threshold in zip(deltas, thresholds)]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    bars = ax.bar(labels, deltas, color=colors, width=0.68)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.scatter(range(len(labels)), thresholds, marker="_", s=430, linewidths=2.2,
               color="#222222", label="view-specific minimum gate")
    ax.set_ylabel("Macro F1 difference: real route − MIX50")
    ax.set_title("Fixed-route contrast versus MIX50")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="best")
    for bar, delta in zip(bars, deltas):
        ax.annotate(f"{delta:+.4f}", (bar.get_x() + bar.get_width() / 2, delta),
                    xytext=(0, 4 if delta >= 0 else -13), textcoords="offset points",
                    ha="center", va="bottom" if delta >= 0 else "top", fontsize=8)
    fig.savefig(REPORT_ROOT / "primary_gates.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    for origin, color, marker in (("E67", "#2f6b9a", "o"), ("E75", "#cc7b29", "s")):
        rows = result["origins"][origin]["per_cycle"]
        cycles = [row["cycle"] for row in rows]
        cycle_deltas = [row["delta"] for row in rows]
        ax.plot(cycles, cycle_deltas, color=color, marker=marker, markersize=3.6,
                linewidth=1.4, label=f"{origin} real − MIX50")
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("cycle_code")
    ax.set_ylabel("Per-cycle Macro F1 difference")
    ax.set_title("Per-cycle sensitivity of the fixed route")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(REPORT_ROOT / "per_cycle_delta.png", dpi=170)
    plt.close(fig)


def run_synthetic_lifecycle(temp_root: Path) -> dict[str, Any]:
    """Temporary-only transform/fit/receipt/recovery/prediction/metric lifecycle."""
    temp_root = Path(temp_root).resolve()
    temp_base = Path(tempfile.gettempdir()).resolve()
    if not temp_root.is_relative_to(temp_base) or temp_root == ROOT or LEDGER.resolve() == (temp_root / "budget.jsonl").resolve():
        raise ContrastPilotError("Synthetic lifecycle requires an explicit temporary-only root")
    production_hash_before = offset.sha256_file(LEDGER)
    temp_root.mkdir(parents=True, exist_ok=True)
    ledger = temp_root / "budget.jsonl"
    artifact = temp_root / "units/SYNTHETIC_CONTRAST.npz"
    receipt_path = temp_root / "receipts/SYNTHETIC_CONTRAST.json"
    output_path = temp_root / "predictions/SYNTHETIC.npz"
    freeze = {"freeze_signature": "synthetic-freeze", "source_commit": "synthetic"}
    x_base = sparse.csr_matrix(np.arange(72, dtype=np.float64).reshape(24, 3) / 100)
    x_f = sparse.csr_matrix(np.eye(24, 6, dtype=np.float64))
    x_m = sparse.csr_matrix(np.roll(np.eye(24, 6, dtype=np.float64), 1, axis=0))
    x_l = sparse.csr_matrix(np.roll(np.eye(24, 6, dtype=np.float64), 2, axis=0))
    augmented = sparse.hstack((x_base, x_f, x_m, x_l), format="csr")
    z = contrast_block(augmented, x_base.shape[1])
    manual_z = sparse.hstack(((x_f - x_l) / np.sqrt(2.0),
                              (x_f - 2.0 * x_m + x_l) / np.sqrt(6.0)), format="csr")
    if offset.sha256_csr(z) != offset.sha256_csr(manual_z):
        raise ContrastPilotError("Synthetic independent contrast formula mismatch")
    offset_logit = np.linspace(-0.2, 0.2, z.shape[0])
    y = (np.arange(z.shape[0]) % 2).astype(np.int8)
    fit = np.arange(18, dtype=np.int64)
    score_rows = np.arange(18, 24, dtype=np.int64)
    theta, optimizer = offset.fit_offset_coefficients(z[fit], offset_logit[fit], y[fit], c=1.0)
    embedded = expand_contrast_coefficients(theta)
    old_position = sparse.hstack((x_f, x_m, x_l), format="csr")
    if (not np.allclose(np.asarray(z @ theta).ravel(), np.asarray(old_position @ embedded).ravel(), atol=1e-12, rtol=0)
            or not np.isclose(np.dot(theta, theta), np.dot(embedded, embedded), atol=1e-12, rtol=0)):
        raise ContrastPilotError("Synthetic independent contrast logit/L2 reconstruction failed")
    fit_logits = offset_logit[fit] + np.asarray(z[fit] @ theta).ravel()
    signed_fit = 2.0 * y[fit].astype(np.float64) - 1.0
    independent_loss = float(np.logaddexp(0.0, -signed_fit * fit_logits).sum() + 0.5 * np.dot(theta, theta))
    independent_zero = float(np.logaddexp(0.0, -signed_fit * offset_logit[fit]).sum())
    independent_gradient = np.asarray(z[fit].T @ (expit(fit_logits) - y[fit]) + theta).reshape(-1)
    if (abs(independent_loss - optimizer["loss_final"]) > 1e-6
            or abs(independent_zero - optimizer["loss_gamma0"]) > 1e-6
            or np.max(np.abs(independent_gradient), initial=0.0) > max(1.1e-6, optimizer["gradient_inf_norm"] * 1.05)):
        raise ContrastPilotError("Synthetic independent objective/gradient/penalty reconstruction failed")
    # Simulate a durable artifact after FIT_STARTED but before FIT_SAVED.
    offset._append_ledger({"event": "FIT_STARTED", "fit_id": "SYNTHETIC_CONTRAST", "freeze_signature": freeze["freeze_signature"]}, path=ledger)
    offset._atomic_npz(artifact, gamma=theta)
    receipt = {"fit_id": "SYNTHETIC_CONTRAST", "freeze_signature": freeze["freeze_signature"],
               "artifact_sha256": offset.sha256_file(artifact), "gamma_sha256": offset.sha256_array(theta),
               "optimizer": optimizer}
    _atomic_json(receipt_path, receipt)
    before_recovery = len(offset._ledger_events(ledger))
    resumed = offset._resume_started_fit("SYNTHETIC_CONTRAST", freeze, offset._ledger_events(ledger),
                                         ledger_path=ledger, artifact_path=artifact,
                                         receipt_path=receipt_path, official=False)
    after_recovery = offset._ledger_events(ledger)
    if resumed != "resumed_from_durable_receipt" or len(after_recovery) != before_recovery + 1:
        raise ContrastPilotError("Synthetic durable resume did not append a single FIT_SAVED without retry")
    with np.load(artifact, allow_pickle=False) as a:
        theta_saved = np.asarray(a["gamma"], dtype=np.float64)
    p_p = np.linspace(0.1, 0.9, len(score_rows))
    p_a = np.linspace(0.8, 0.2, len(score_rows))
    present = np.asarray([True, True, False, True, False, True])
    q = offset.offset_probability(z[score_rows], offset_logit[score_rows], theta_saved)
    if not np.allclose(q, expit(offset_logit[score_rows] + np.asarray(z[score_rows] @ theta_saved).ravel()), atol=1e-14, rtol=0):
        raise ContrastPilotError("Synthetic independent score-logit replay mismatch")
    route = _fixed_route(q, p_a, p_p, present)
    if not np.array_equal(route[~present], p_p[~present]):
        raise ContrastPilotError("Synthetic missing route was not exact")
    ids = np.asarray([f"S{i}" for i in score_rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offset._atomic_npz(output_path, ids=ids, score_rows=score_rows, probability=route,
                       present=present, truth=y[score_rows])
    # Adversarial row and source/bundle mismatches must be rejected.
    from scripts.verify_optimization_v3_positional_contrast_v1 import assert_ids_match, verify_file_hash
    try:
        assert_ids_match(ids[::-1], ids)
        id_rejected = False
    except AssertionError:
        id_rejected = True
    bad_receipt = dict(receipt, freeze_signature="tampered-freeze")
    try:
        from scripts.verify_optimization_v3_positional_contrast_v1 import verify_receipt_binding
        verify_receipt_binding(bad_receipt, fit_id="SYNTHETIC_CONTRAST", freeze_signature=freeze["freeze_signature"],
                               artifact_sha256=receipt["artifact_sha256"])
        source_rejected = False
    except AssertionError:
        source_rejected = True
    try:
        verify_file_hash(output_path, "0" * 64)
        bundle_rejected = False
    except AssertionError:
        bundle_rejected = True
    if not (id_rejected and source_rejected and bundle_rejected):
        raise ContrastPilotError("Synthetic tamper guards did not reject adversarial data")
    with np.load(output_path, allow_pickle=False) as a:
        actual = np.asarray(a["probability"], dtype=np.float64)
        truth = np.asarray(a["truth"], dtype=np.int8)
        assert np.array_equal(np.asarray(a["ids"]).astype(str), ids.astype(str))
    from scripts.verify_optimization_v3_positional_contrast_v1 import independent_metric
    metric = independent_metric(truth, actual)
    if metric["n"] != len(score_rows) or not np.isfinite(metric["macro_f1"]):
        raise ContrastPilotError("Synthetic independent metric reconstruction failed")
    production_hash_after = offset.sha256_file(LEDGER)
    if production_hash_before != production_hash_after:
        raise ContrastPilotError("Synthetic-only execution mutated the production ledger")
    return {
        "status": "PASS", "temporary_root": str(temp_root),
        "production_ledger_sha256_before": production_hash_before,
        "production_ledger_sha256_after": production_hash_after,
        "durable_resume": resumed, "optimizer_fit_count": 1,
        "fit_artifact_sha256": offset.sha256_file(artifact), "prediction_bundle_sha256": offset.sha256_file(output_path),
        "missing_route_exact": True, "tampered_ids_rejected": id_rejected,
        "source_mismatch_rejected": source_rejected, "bundle_mismatch_guard_exercised": bundle_rejected,
        "contrast_formula_independently_reconstructed": True,
        "contrast_logits_and_l2_independently_reconstructed": True,
        "objective_gradient_and_penalty_independently_reconstructed": True,
        "independent_metric": metric, "query_labels": "synthetic only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "freeze", "reserve", "run-all", "predictions",
                                              "verify-prelabel", "score", "prepare-prefit-amendment",
                                              "rebind-prefit-amendment", "synthetic"])
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight": result = preflight()
        elif args.command == "freeze": result = freeze()
        elif args.command == "reserve": result = reserve()
        elif args.command == "run-all": result = run_all_fits()
        elif args.command == "predictions": result = build_predictions()
        elif args.command == "verify-prelabel": result = verify_prelabel()
        elif args.command == "score": result = score()
        elif args.command == "prepare-prefit-amendment": result = prepare_prefit_amendment()
        elif args.command == "rebind-prefit-amendment": result = rebind_prefit_amendment()
        else:
            if args.temp_root is None:
                raise ContrastPilotError("synthetic requires --temp-root under the system temporary directory")
            result = run_synthetic_lifecycle(args.temp_root)
        print(json.dumps(_jsonable(result), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
