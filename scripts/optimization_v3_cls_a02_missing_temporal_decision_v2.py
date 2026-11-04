"""Past-only decision-threshold transfer for A02 probabilities on missing rows.

No estimator or preprocessing is fit. Present-stream labels remain frozen at
MIX50 >= 0.50; only the missing-stream decision threshold on saved original
A02 probabilities may vary. E79 probabilities are opened only by the separate
application command after its application freeze has been committed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_mix50_temporal_decision_v1 as mix  # noqa: E402
from scripts import optimization_v3_cls_mix50_a02_missing_e79_v1 as route_audit  # noqa: E402

REPORT_DIR = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2"
PROTOCOL_PATH = ROOT / "experiments/optimization_v3/cls_a02_missing_temporal_decision_v2/protocol.json"
PREFIT_FREEZE_PATH = REPORT_DIR / "PREFIT_FREEZE.json"
SYNTHETIC_QA_PATH = REPORT_DIR / "SYNTHETIC_QA.json"
ROOT_SOURCE_PASS_PATH = REPORT_DIR / "ROOT_SOURCE_PASS.json"
LEDGER_PATH = REPORT_DIR / "SELECTOR_LEDGER.jsonl"
RECEIPT_DIR = REPORT_DIR / "receipts"
EVAL_DIR = REPORT_DIR / "evaluations"
RESULTS_PATH = REPORT_DIR / "RESULTS.json"
RUN_MANIFEST_PATH = ROOT / "experiments/runs/optimization_v3_cls_a02_missing_temporal_decision_v2/RUN_MANIFEST.json"
E79_APPLICATION_FREEZE_PATH = REPORT_DIR / "E79_APPLICATION_FREEZE.json"
MANIFEST_PATH = REPORT_DIR / "E79_MANIFEST.json"
QA_PATH = REPORT_DIR / "E79_QA.json"

TRAIN = ROOT / "data/raw/classification/Train.csv"
OOF_NPZ = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
E79_NPZ = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
OFFICIAL_TEST = ROOT / "data/raw/classification/Test.csv"
CURRENT_BEST_CSV = ROOT / "submissions/optimization_v3/cls_mix50_a02_missing_e79_v1/classification_mix50_present_a02_missing_e79.csv"
CURRENT_BEST_MANIFEST = ROOT / "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/MANIFEST.json"
CURRENT_BEST_QA = ROOT / "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/QA.json"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv"

TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
OOF_SHA256 = "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"
E79_SHA256 = "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"
TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
CURRENT_BEST_CSV_SHA256 = "3d970fa208d094e1e01c462671eeff327e05125bfacde2bacf36259b29fdf0bc"
CURRENT_BEST_MANIFEST_SHA256 = "c0ef8ce6b3116b765072fd2fe9e6dfb37dbc1577ee456ac84e8ec8f93498cfca"
CURRENT_BEST_QA_SHA256 = "36db9799ccf10a6d5dbd9842b3c10fc28780683c4171ee2aeaa01cfe2d3493ce"

POLICY_ID = "CLS_A02_MISSING_TEMPORAL_DECISION_V1"
IMPLEMENTATION_VERSION = 2
MAX_SELECTORS = 3
THRESHOLD_FIXED_PRESENT = 0.50
TIE_TOLERANCE = 1e-12
MIN_SELECTION_ROWS = 1000
MIN_SELECTION_PER_CLASS = 100
MIN_VARYING_ROWS = 200
MIN_VARYING_PER_CLASS = 30
MIN_GROUP_ROWS = 200
MIN_GROUP_PER_CLASS = 30
MAX_THREADPOOL_THREADS = 2
EXPECTED_POOLED_W5_W6_ROWS = 15953
EXPECTED_STAGES = ("D1_E67_W4_TO_W5", "D2_E67_W4W5_TO_E75_W6", "FINAL_STITCHED_E79")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return value


def _load_protocol() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("policy_id") != POLICY_ID or protocol.get("status") != "FROZEN_BEFORE_OFFICIAL_SELECTORS":
        raise RuntimeError("Frozen protocol identity/status mismatch")
    if protocol.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise RuntimeError("Frozen protocol implementation version mismatch")
    if protocol.get("stage_order") != list(EXPECTED_STAGES):
        raise RuntimeError("Frozen protocol stage order differs from the implementation")
    gates = protocol.get("support_gate", {})
    expected_gates = {
        "full_selection_rows_min": MIN_SELECTION_ROWS,
        "full_selection_each_class_min": MIN_SELECTION_PER_CLASS,
        "varying_missing_rows_min": MIN_VARYING_ROWS,
        "varying_missing_each_class_min": MIN_VARYING_PER_CLASS,
    }
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        raise RuntimeError("Frozen protocol support gate differs from the implementation")
    decision_rule = protocol.get("decision_rule", {})
    if decision_rule.get("objective") != "overall full-row Macro F1, with present-stream confusion fixed":
        raise RuntimeError("Frozen protocol objective differs from the implementation")
    if decision_rule.get("threshold_operator") != ">=" or decision_rule.get("tie_rule") != (
        "scores within 1e-12 of the maximum are tied; choose closest to 0.50, then lower threshold"
    ):
        raise RuntimeError("Frozen protocol threshold/tie rule differs from the implementation")
    inputs = protocol.get("frozen_inputs", {})
    expected_inputs = {
        "train_sha256": TRAIN_SHA256,
        "oof_bundle_sha256": OOF_SHA256,
        "e79_probability_bundle_expected_sha256_after_application_freeze": E79_SHA256,
        "official_test_expected_sha256_after_application_freeze": TEST_SHA256,
        "current_best_csv_sha256": CURRENT_BEST_CSV_SHA256,
        "current_best_manifest_sha256": CURRENT_BEST_MANIFEST_SHA256,
        "current_best_qa_sha256": CURRENT_BEST_QA_SHA256,
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise RuntimeError("Frozen protocol input hashes differ from the implementation")
    resources = protocol.get("resource_and_source_boundaries", {})
    if (
        resources.get("max_decision_selectors") != MAX_SELECTORS
        or resources.get("minimum_available_ram_bytes") != mix.MIN_AVAILABLE_RAM_BYTES
        or resources.get("estimated_incremental_peak_bytes") != mix.ESTIMATED_INCREMENTAL_PEAK_BYTES
        or resources.get("documented_safety_reserve_bytes") != mix.SAFETY_RESERVE_BYTES
        or resources.get("required_available_ram_bytes") != mix.REQUIRED_AVAILABLE_RAM_BYTES
        or resources.get("project_job_limit") != mix.MAX_PROJECT_JOBS
        or resources.get("threadpool_thread_limit") != MAX_THREADPOOL_THREADS
        or resources.get("new_estimator_fits") != 0
        or resources.get("new_preprocessing_fits") != 0
        or resources.get("new_model_inference") != 0
        or resources.get("test_labels_read") is not False
        or resources.get("SampleSubmission_read") is not False
        or resources.get("Kaggle_contacted") is not False
    ):
        raise RuntimeError("Frozen protocol resource boundary differs from the implementation")
    if (
        resources.get("resource_preflight")
        != "call the frozen parent mix._resource_preflight before reading selected Train labels or reserving each new supported selector; do not rerun for a durable consumed selector or a predeclared support fallback"
        or resources.get("single_job_lock")
        != "hold frozen parent mix._single_project_job in the V2 run namespace around selector workflow and E79 candidate build"
    ):
        raise RuntimeError("Frozen protocol resource gate or job-lock contract is missing")
    source_pass_contract = protocol.get("root_source_pass_contract", {})
    if (
        source_pass_contract.get("status") != "PASS"
        or source_pass_contract.get("source_hashes_must_match_prefit_freeze") is not True
        or source_pass_contract.get("receipt_canonical_sha256_required") is not True
        or source_pass_contract.get("receipt_must_be_committed_before_official_selectors") is not True
    ):
        raise RuntimeError("Frozen protocol root source-pass contract is missing")
    successor = protocol.get("successor_checkpoint", {})
    if (
        successor.get("v1_official_selector_outcomes_scored") is not False
        or successor.get("v2_selector_budget_cap") != MAX_SELECTORS
        or successor.get("v2_slots_used_at_prefit") != 0
    ):
        raise RuntimeError("Frozen protocol does not preserve V1 and the unused V2 selector budget")
    if protocol.get("interpretation_limits", {}).get("pooled_W5_W6_is_unique_15953_rows_from_confusion_counts") is not True:
        raise RuntimeError("Frozen protocol pooled-window denominator requirement is missing")
    return protocol


def _validate_planned_support(stage: str, support: Mapping[str, Any]) -> None:
    planned = _load_protocol()["planned_support_before_new_outcomes"][stage]
    actual_fields = {
        "rows": "rows",
        "class_0": "class_0",
        "class_1": "class_1",
        "missing_rows": "varying_missing_rows",
        "missing_class_0": "varying_missing_class_0",
        "missing_class_1": "varying_missing_class_1",
    }
    for planned_key, actual_key in actual_fields.items():
        if int(planned[planned_key]) != int(support[actual_key]):
            raise RuntimeError(f"Predeclared support differs for {stage}: {planned_key}")


def _planned_selector_supported(stage: str) -> bool:
    planned = _load_protocol()["planned_support_before_new_outcomes"][stage]
    return (
        int(planned["rows"]) >= MIN_SELECTION_ROWS
        and min(int(planned["class_0"]), int(planned["class_1"])) >= MIN_SELECTION_PER_CLASS
        and int(planned["missing_rows"]) >= MIN_VARYING_ROWS
        and min(int(planned["missing_class_0"]), int(planned["missing_class_1"])) >= MIN_VARYING_PER_CLASS
    )


def _validate_planned_evaluation(stage: str, view: Mapping[str, np.ndarray]) -> None:
    planned = _load_protocol()["planned_support_before_new_outcomes"][stage]
    if int(planned["evaluation_rows"]) != len(view["ids"]):
        raise RuntimeError(f"Predeclared evaluation denominator differs for {stage}")
    expected_cycles = list(range(int(planned["evaluation_cycles"][0]), int(planned["evaluation_cycles"][1]) + 1))
    actual_cycles = [int(value) for value in np.unique(np.asarray(view["cycles"], dtype=np.int16))]
    if actual_cycles != expected_cycles:
        raise RuntimeError(f"Predeclared evaluation cycles differ for {stage}")


def _write_once_json(path: Path, value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["canonical_sha256"] = canonical_sha256(body)
    payload = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        old = _read_json(path)
        recorded = old.pop("canonical_sha256", None)
        if recorded != canonical_sha256(old) or old != dict(value):
            raise RuntimeError(f"Immutable JSON exists with different content: {path}")
        return sha256_file(path)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _read_verified_receipt(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    signature = value.pop("canonical_sha256", None)
    if signature != canonical_sha256(value):
        raise RuntimeError(f"Receipt canonical hash mismatch: {path}")
    value["canonical_sha256"] = signature
    return value


def _append_ledger(event: Mapping[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(LEDGER_PATH, flags, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_ledger() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid selector ledger line {line_no}") from exc
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid selector ledger event at line {line_no}")
        events.append(item)
    return events


def validate_inputs(*, include_e79: bool = False) -> dict[str, str]:
    expected = {
        TRAIN.relative_to(ROOT).as_posix(): TRAIN_SHA256,
        OOF_NPZ.relative_to(ROOT).as_posix(): OOF_SHA256,
    }
    if include_e79:
        expected.update({
            E79_NPZ.relative_to(ROOT).as_posix(): E79_SHA256,
            OFFICIAL_TEST.relative_to(ROOT).as_posix(): TEST_SHA256,
            CURRENT_BEST_CSV.relative_to(ROOT).as_posix(): CURRENT_BEST_CSV_SHA256,
            CURRENT_BEST_MANIFEST.relative_to(ROOT).as_posix(): CURRENT_BEST_MANIFEST_SHA256,
            CURRENT_BEST_QA.relative_to(ROOT).as_posix(): CURRENT_BEST_QA_SHA256,
        })
    actual: dict[str, str] = {}
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        actual[rel] = sha256_file(path)
        if actual[rel] != digest:
            raise RuntimeError(f"Frozen input hash mismatch for {rel}")
    return actual


def verify_prefit_checkout() -> str:
    freeze = _read_json(PREFIT_FREEZE_PATH)
    if freeze.get("policy_id") != POLICY_ID or freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_SELECTORS":
        raise RuntimeError("Prefit freeze identity/status mismatch")
    head = _head()
    base = freeze["base_commit"]
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT, check=False)
    if ancestor.returncode != 0 or head == base:
        raise RuntimeError("Prefit source must be committed after the frozen base")
    owned = [
        "scripts/optimization_v3_cls_a02_missing_temporal_decision_v2.py",
        "tests/test_optimization_v3_cls_a02_missing_temporal_decision_v2.py",
        "experiments/optimization_v3/cls_a02_missing_temporal_decision_v2/protocol.json",
        "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/PREFIT_FREEZE.json",
        "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/SYNTHETIC_QA.json",
        "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/ROOT_SOURCE_PASS.json",
        "scripts/optimization_v3_cls_mix50_temporal_decision_v1.py",
        "scripts/optimization_v3_cls_mix50_a02_missing_e79_v1.py",
    ]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *owned], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Frozen source path has uncommitted changes")
    for rel, expected in freeze["source_hashes"].items():
        path = ROOT / rel
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Prefit source hash mismatch for {rel}")
    successor = _load_protocol()["successor_checkpoint"]
    if freeze.get("base_commit") != successor.get("preserved_v1_commit"):
        raise RuntimeError("V2 prefit base does not descend from the preserved V1 checkpoint")
    v1_freeze = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v1/PREFIT_FREEZE.json"
    if sha256_file(v1_freeze) != successor.get("preserved_v1_prefit_freeze_sha256"):
        raise RuntimeError("Preserved V1 prefit freeze hash changed")
    if not ROOT_SOURCE_PASS_PATH.exists():
        raise RuntimeError("Official selectors require an explicit committed ROOT_SOURCE_PASS.json")
    source_pass_dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", ROOT_SOURCE_PASS_PATH.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        text=True,
    ).strip()
    if source_pass_dirty:
        raise RuntimeError("ROOT source-pass receipt must be committed before official selectors")
    source_pass = _read_verified_receipt(ROOT_SOURCE_PASS_PATH)
    if (
        source_pass.get("status") != "PASS"
        or source_pass.get("policy_id") != POLICY_ID
        or source_pass.get("implementation_version") != IMPLEMENTATION_VERSION
        or source_pass.get("prefit_freeze_sha256") != sha256_file(PREFIT_FREEZE_PATH)
        or not isinstance(source_pass.get("prefit_commit"), str)
        or len(source_pass.get("prefit_commit", "")) != 40
        or source_pass.get("source_hashes") != freeze.get("source_hashes")
        or source_pass.get("decision_protocol_sha256") != sha256_file(PROTOCOL_PATH)
        or source_pass.get("synthetic_qa_sha256") != sha256_file(SYNTHETIC_QA_PATH)
        or source_pass.get("official_selector_outcomes_scored") is not False
        or source_pass.get("e79_probability_bundle_accessed") is not False
        or source_pass.get("test_accessed") is not False
    ):
        raise RuntimeError("ROOT source-pass receipt does not approve this prefit checkpoint")
    approved = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_pass["prefit_commit"], head],
        cwd=ROOT,
        check=False,
    )
    if approved.returncode != 0:
        raise RuntimeError("Current checkout is not descended from ROOT-approved prefit commit")
    validate_inputs(include_e79=False)
    return source_pass["prefit_commit"]


def _validate_probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"{name} must be a finite one-dimensional probability in [0, 1]")
    return p


def _labels(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or y.dtype.kind not in "biu" or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a one-dimensional integer/bool vector containing 0/1")
    return y.astype(np.int8, copy=False)


def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    target = _labels(y)
    decision = np.asarray(pred, dtype=bool)
    if target.shape != decision.shape:
        raise ValueError("Target/prediction shape mismatch")
    tn = int(np.sum((target == 0) & ~decision))
    fp = int(np.sum((target == 0) & decision))
    fn = int(np.sum((target == 1) & ~decision))
    tp = int(np.sum((target == 1) & decision))
    return tn, fp, fn, tp


def macro_f1_from_confusion(counts: tuple[int, int, int, int]) -> float:
    tn, fp, fn, tp = counts
    d0 = 2 * tn + fp + fn
    d1 = 2 * tp + fp + fn
    f10 = 2 * tn / d0 if d0 else 0.0
    f11 = 2 * tp / d1 if d1 else 0.0
    return float((f10 + f11) / 2.0)


def confusion_record(counts: tuple[int, int, int, int]) -> dict[str, Any]:
    tn, fp, fn, tp = counts
    d0 = 2 * tn + fp + fn
    d1 = 2 * tp + fp + fn
    f10 = 2 * tn / d0 if d0 else 0.0
    f11 = 2 * tp / d1 if d1 else 0.0
    return {
        "rows": int(sum(counts)),
        "class_0": int(tn + fp),
        "class_1": int(fn + tp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positives": fp,
        "false_negatives": fn,
        "f1_0": float(f10),
        "f1_1": float(f11),
        "macro_f1": float((f10 + f11) / 2.0),
    }


def selection_support(y: np.ndarray, missing: np.ndarray) -> dict[str, Any]:
    target = _labels(y)
    miss = np.asarray(missing, dtype=bool)
    if target.shape != miss.shape:
        raise ValueError("Selection target/missing mask mismatch")
    n0 = int(np.sum(target == 0))
    n1 = int(np.sum(target == 1))
    varying_y = target[miss]
    m0 = int(np.sum(varying_y == 0))
    m1 = int(np.sum(varying_y == 1))
    passed = (
        len(target) >= MIN_SELECTION_ROWS
        and min(n0, n1) >= MIN_SELECTION_PER_CLASS
        and int(miss.sum()) >= MIN_VARYING_ROWS
        and min(m0, m1) >= MIN_VARYING_PER_CLASS
    )
    return {
        "status": "SUPPORTED" if passed else "FALLBACK_FIXED_0_50_INSUFFICIENT_SUPPORT",
        "rows": int(len(target)),
        "class_0": n0,
        "class_1": n1,
        "varying_missing_rows": int(miss.sum()),
        "varying_missing_class_0": m0,
        "varying_missing_class_1": m1,
        "full_rows_min": MIN_SELECTION_ROWS,
        "full_each_class_min": MIN_SELECTION_PER_CLASS,
        "varying_rows_min": MIN_VARYING_ROWS,
        "varying_each_class_min": MIN_VARYING_PER_CLASS,
        "selector_consumed": bool(passed),
    }


def choose_missing_threshold(
    y: np.ndarray,
    present_probability: np.ndarray,
    p_a02: np.ndarray,
    missing: np.ndarray,
) -> dict[str, Any]:
    """Optimize overall Macro F1 while keeping every present decision fixed."""
    target = _labels(y)
    present_p = _validate_probability_vector(present_probability, "present MIX50")
    pa = _validate_probability_vector(p_a02, "missing A02 pA")
    miss = np.asarray(missing, dtype=bool)
    if target.shape != present_p.shape or target.shape != pa.shape or target.shape != miss.shape:
        raise ValueError("Selection arrays must have identical one-dimensional shape")
    support = selection_support(target, miss)
    if support["status"] != "SUPPORTED":
        return {
            "status": support["status"],
            "threshold": THRESHOLD_FIXED_PRESENT,
            "threshold_selector_consumed": False,
            "support": support,
            "selection_candidate_count": 0,
            "selection_best_macro_f1": None,
            "selection_macro_f1_at_threshold": None,
            "tie_candidate_count": 0,
            "selection_confusion": confusion_record(confusion(target, _fixed_predictions(present_p, pa, miss, THRESHOLD_FIXED_PRESENT))),
        }
    candidates = mix.threshold_candidates(pa[miss])
    scores: list[tuple[float, float, tuple[int, int, int, int]]] = []
    for threshold in candidates:
        pred = _fixed_predictions(present_p, pa, miss, float(threshold))
        counts = confusion(target, pred)
        scores.append((float(threshold), macro_f1_from_confusion(counts), counts))
    best_score = max(score for _, score, _ in scores)
    tied = [item for item in scores if best_score - item[1] <= TIE_TOLERANCE]
    threshold, selected_score, selected_counts = min(tied, key=lambda item: (abs(item[0] - 0.5), item[0]))
    return {
        "status": "SELECTED",
        "threshold": float(threshold),
        "threshold_selector_consumed": True,
        "support": support,
        "selection_candidate_count": int(len(candidates)),
        "selection_best_macro_f1": float(best_score),
        "selection_macro_f1_at_threshold": float(selected_score),
        "tie_candidate_count": int(len(tied)),
        "tie_tolerance": TIE_TOLERANCE,
        "tie_rule": "among scores within 1e-12 of best, choose closest to 0.50; then lower threshold",
        "threshold_operator": ">=",
        "selection_confusion": confusion_record(selected_counts),
    }


def _fixed_predictions(present_probability: np.ndarray, p_a02: np.ndarray,
                       missing: np.ndarray, threshold: float) -> np.ndarray:
    present_p = _validate_probability_vector(present_probability, "present MIX50")
    pa = _validate_probability_vector(p_a02, "A02 pA")
    miss = np.asarray(missing, dtype=bool)
    if present_p.shape != pa.shape or present_p.shape != miss.shape:
        raise ValueError("Policy probability/missing arrays do not align")
    pred = present_p >= THRESHOLD_FIXED_PRESENT
    pred[miss] = pa[miss] >= float(threshold)
    return pred


def _old_mix_predictions(present_probability: np.ndarray, p_p: np.ndarray,
                         missing: np.ndarray) -> np.ndarray:
    present_p = _validate_probability_vector(present_probability, "present MIX50")
    p_pos = _validate_probability_vector(p_p, "old pP missing baseline")
    miss = np.asarray(missing, dtype=bool)
    pred = present_p >= THRESHOLD_FIXED_PRESENT
    pred[miss] = p_pos[miss] >= THRESHOLD_FIXED_PRESENT
    return pred


def _safe_ids(values: np.ndarray) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind != "U" or np.any(ids == "") or len(np.unique(ids)) != len(ids):
        raise ValueError("record IDs must be nonempty, unique unicode vectors")
    return ids.astype(str, copy=False)


def _array_sha256(values: np.ndarray, dtype: str | None = None) -> str:
    arr = np.asarray(values)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _select_origin(origin: Mapping[str, np.ndarray], low: int, high: int) -> dict[str, np.ndarray]:
    mask = (origin["cycles"] >= low) & (origin["cycles"] <= high)
    names = ("ids", "rows", "cycles", "pP", "pA", "missing", "MIX50",
             "group_availability", "group_text_load", "group_entry_friction")
    selected = {name: np.asarray(origin[name])[mask] for name in names}
    if len(selected["ids"]) == 0 or len(np.unique(selected["ids"])) != len(selected["ids"]):
        raise RuntimeError(f"Empty or duplicate IDs in cycle view {low}-{high}")
    return selected


def _join_views(*parts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise ValueError("Cannot join views with different fields")
    result = {key: np.concatenate([np.asarray(part[key]) for part in parts]) for key in parts[0]}
    result["ids"] = _safe_ids(result["ids"])
    if len(np.unique(result["ids"])) != len(result["ids"]):
        raise RuntimeError("Stitched selector rows overlap by record_id")
    return result


def _load_oof() -> dict[str, dict[str, np.ndarray]]:
    return mix._load_oof_predictions(OOF_NPZ, OOF_SHA256)


def _cycle_map(ids: np.ndarray, cycles: np.ndarray) -> dict[str, int]:
    return {str(rid): int(cycle) for rid, cycle in zip(ids.tolist(), cycles.tolist())}


def _labels_by_id(view: Mapping[str, np.ndarray]) -> np.ndarray:
    return mix.load_train_labels_by_id(
        TRAIN,
        np.asarray(view["ids"], dtype=str),
        _cycle_map(np.asarray(view["ids"], dtype=str), np.asarray(view["cycles"])),
        expected_train_sha256=TRAIN_SHA256,
    )


def _selector_context(stage: str, view: Mapping[str, np.ndarray], source_commit: str) -> dict[str, Any]:
    ids = np.asarray(view["ids"], dtype=str)
    cycles = np.asarray(view["cycles"], dtype=np.int16)
    miss = np.asarray(view["missing"], dtype=np.uint8)
    present = ~np.asarray(view["missing"], dtype=bool)
    mix50 = np.asarray(view["MIX50"], dtype=np.float64)
    p_a = np.asarray(view["pA"], dtype=np.float64)
    p_p = np.asarray(view["pP"], dtype=np.float64)
    if not np.array_equal(mix50[present], (0.5 * p_p + 0.5 * p_a)[present]):
        raise RuntimeError(f"Frozen present MIX50 formula differs in {stage}")
    return {
        "stage": stage,
        "source_commit": source_commit,
        "input_hashes": {"train_sha256": TRAIN_SHA256, "oof_bundle_sha256": OOF_SHA256},
        "selection_ids_sha256": mix._ids_digest(ids),
        "selection_cycles_sha256": _array_sha256(cycles, "<i2"),
        "selection_missing_sha256": _array_sha256(miss, "u1"),
        "selection_present_mix50_probability_sha256": _array_sha256(mix50[present], "<f8"),
        "selection_missing_a02_probability_sha256": _array_sha256(p_a[~present], "<f8"),
        "selection_rows": int(len(ids)),
        "selection_cycle_values": [int(x) for x in np.unique(cycles)],
    }


def _receipt_path(stage: str) -> Path:
    return RECEIPT_DIR / f"{stage}.json"


def _eval_path(stage: str) -> Path:
    return EVAL_DIR / f"{stage}.json"


def _stage_spec() -> dict[str, dict[str, Any]]:
    return {
        "D1_E67_W4_TO_W5": {
            "selection": [("67", 68, 71)],
            "evaluation": [("67", 72, 75)],
            "evaluation_name": "E67_W5",
        },
        "D2_E67_W4W5_TO_E75_W6": {
            "selection": [("67", 68, 75)],
            "evaluation": [("75", 76, 79)],
            "evaluation_name": "E75_W6",
        },
        "FINAL_STITCHED_E79": {
            "selection": [("67", 68, 75), ("75", 76, 79)],
            "evaluation": [],
            "evaluation_name": None,
        },
    }


def _view_for_spec(oof: Mapping[str, Mapping[str, np.ndarray]], parts: list[tuple[str, int, int]]) -> dict[str, np.ndarray]:
    selected = [_select_origin(oof[origin], low, high) for origin, low, high in parts]
    return selected[0] if len(selected) == 1 else _join_views(*selected)


def _ledger_state(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for event in events:
        sid = event.get("selector_id")
        kind = event.get("event")
        if not isinstance(sid, str) or sid not in EXPECTED_STAGES:
            raise RuntimeError("Selector ledger contains an unknown selector ID")
        item = state.setdefault(sid, {"reserved": None, "saved": None, "fallback": None, "evaluation": None, "failed": None})
        if kind == "SELECTOR_RESERVED":
            if item["reserved"] is not None:
                raise RuntimeError(f"Duplicate selector reservation: {sid}")
            item["reserved"] = event
        elif kind == "SELECTOR_SAVED":
            if item["reserved"] is None or item["saved"] is not None:
                raise RuntimeError(f"Invalid selector saved event: {sid}")
            item["saved"] = event
        elif kind == "SELECTOR_FAILED":
            if item["reserved"] is None or item["failed"] is not None:
                raise RuntimeError(f"Invalid selector failed event: {sid}")
            item["failed"] = event
        elif kind == "SELECTOR_FALLBACK":
            if item["reserved"] is not None or item["fallback"] is not None:
                raise RuntimeError(f"Invalid selector fallback event: {sid}")
            item["fallback"] = event
        elif kind == "EVALUATION_SAVED":
            if (item["saved"] is None and item["fallback"] is None) or item["evaluation"] is not None:
                raise RuntimeError(f"Invalid evaluation saved event: {sid}")
            item["evaluation"] = event
        else:
            raise RuntimeError(f"Unknown selector ledger event {kind!r}")
    return state


def _validate_selection_receipt(receipt: dict[str, Any], context: dict[str, Any], y: np.ndarray) -> None:
    body = dict(receipt)
    signature = body.pop("canonical_sha256", None)
    if signature != canonical_sha256(body):
        raise RuntimeError("Selection receipt canonical hash is invalid")
    for key, expected in context.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"Selection receipt context mismatch: {key}")
    if receipt.get("selection_target_sha256") != _array_sha256(_labels(y), "i1"):
        raise RuntimeError("Selection receipt target hash mismatch")


def _reserve_selector(stage: str, context: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    resource_gate = context.get("resource_gate")
    if not isinstance(resource_gate, Mapping):
        raise RuntimeError("New supported selector reservation requires a resource preflight receipt")
    mix._validate_resource_gate_record(resource_gate)
    state = _ledger_state(events)
    stage_index = EXPECTED_STAGES.index(stage)
    for prior_stage in EXPECTED_STAGES[:stage_index]:
        prior = state.get(prior_stage)
        if prior is None or (prior["saved"] is None and prior["fallback"] is None):
            raise RuntimeError(f"Selector order requires completed prior stage {prior_stage}")
        if prior["failed"] is not None:
            raise RuntimeError(f"Prior selector failed; no later selector allowed: {prior_stage}")
    current = state.get(stage)
    if current is not None:
        if current["failed"] is not None:
            raise RuntimeError(f"Selector {stage} has a failed attempt; retry is prohibited")
        if current["fallback"] is not None:
            raise RuntimeError(f"Selector {stage} already has a fixed-threshold fallback; retry is prohibited")
        if current["reserved"] is not None:
            raise RuntimeError(f"Selector {stage} already reserved without a receipt; retry is prohibited")
    reserved_count = sum(1 for item in state.values() if item["reserved"] is not None)
    if reserved_count >= MAX_SELECTORS:
        raise RuntimeError("Three-selector decision budget is exhausted")
    event = {
        "event": "SELECTOR_RESERVED",
        "selector_id": stage,
        "reservation_key": f"{POLICY_ID}:{stage}",
        "selector_ordinal": stage_index + 1,
        "source_commit": context["source_commit"],
        "input_hashes": context["input_hashes"],
        "selection_ids_sha256": context["selection_ids_sha256"],
        "selection_rows": context["selection_rows"],
        "resource_gate": dict(resource_gate),
    }
    _append_ledger(event)
    return event


def _run_selection(stage: str, view: Mapping[str, np.ndarray], source_commit: str) -> dict[str, Any]:
    context = _selector_context(stage, view, source_commit)
    path = _receipt_path(stage)
    events = _read_ledger()
    state = _ledger_state(events)
    existing = state.get(stage)
    if existing and existing["failed"] is not None:
        raise RuntimeError(f"No retry after failed selector attempt: {stage}")
    if not path.exists() and existing is not None and (existing["reserved"] is not None or existing["fallback"] is not None):
        raise RuntimeError(f"No retry after an existing selector reservation or fallback: {stage}")
    resource_gate = None
    if not path.exists() and _planned_selector_supported(stage):
        resource_gate = mix._resource_preflight()
    y = _labels_by_id(view)
    support = selection_support(y, np.asarray(view["missing"], dtype=bool))
    _validate_planned_support(stage, support)
    if path.exists():
        receipt = _read_verified_receipt(path)
        _validate_selection_receipt(receipt, context, y)
        if existing is None or (existing["reserved"] is None and existing["fallback"] is None):
            raise RuntimeError("Durable selector receipt exists without its ledger reservation")
        decision_event = existing["saved"] or existing["fallback"]
        if decision_event is not None and decision_event.get("threshold_receipt_sha256") != sha256_file(path):
            raise RuntimeError("Selector ledger does not bind the exact durable threshold receipt file hash")
        if existing["reserved"] is not None and existing["reserved"].get("resource_gate") != receipt.get("resource_gate"):
            raise RuntimeError("Selector receipt resource gate differs from its durable reservation")
        if existing["saved"] is None and existing["fallback"] is None:
            if receipt["selection"]["threshold_selector_consumed"]:
                event = {
                    "event": "SELECTOR_SAVED",
                    "selector_id": stage,
                    "threshold_receipt_sha256": sha256_file(path),
                    "resume_recovered_receipt": True,
                    "source_commit": source_commit,
                    "input_hashes": context["input_hashes"],
                }
            else:
                event = {
                    "event": "SELECTOR_FALLBACK",
                    "selector_id": stage,
                    "threshold_receipt_sha256": sha256_file(path),
                    "reason": receipt["selection"]["status"],
                    "threshold": THRESHOLD_FIXED_PRESENT,
                    "source_commit": source_commit,
                    "input_hashes": context["input_hashes"],
                }
            _append_ledger(event)
        return receipt

    if not support["selector_consumed"]:
        chosen = {
            "status": support["status"],
            "threshold": THRESHOLD_FIXED_PRESENT,
            "threshold_selector_consumed": False,
            "support": support,
            "selection_candidate_count": 0,
            "selection_best_macro_f1": None,
            "selection_macro_f1_at_threshold": None,
            "tie_candidate_count": 0,
            "threshold_operator": ">=",
            "selection_confusion": confusion_record(confusion(
                y,
                _fixed_predictions(np.asarray(view["MIX50"], dtype=np.float64),
                                   np.asarray(view["pA"], dtype=np.float64),
                                   np.asarray(view["missing"], dtype=bool),
                                   THRESHOLD_FIXED_PRESENT),
            )),
        }
    else:
        if resource_gate is None:
            raise RuntimeError("Supported selector has no successful resource preflight")
        context["resource_gate"] = resource_gate
        _reserve_selector(stage, context, events)
        present_p = np.asarray(view["MIX50"], dtype=np.float64)
        p_a = np.asarray(view["pA"], dtype=np.float64)
        missing = np.asarray(view["missing"], dtype=bool)
        chosen = choose_missing_threshold(y, present_p, p_a, missing)
    receipt_body = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        **context,
        "selection_target_sha256": _array_sha256(y, "i1"),
        "resource_gate": resource_gate,
        "selection": chosen,
        "threshold": float(chosen["threshold"]),
        "threshold_operator": ">=",
        "present_decision_rule": "frozen MIX50 probability >= 0.50",
        "missing_decision_rule": "original A02 pA >= selected threshold",
        "new_estimator_fits": 0,
        "new_preprocessing_fits": 0,
    }
    _write_once_json(path, receipt_body)
    saved_hash = sha256_file(path)
    if chosen["threshold_selector_consumed"]:
        _append_ledger({
            "event": "SELECTOR_SAVED",
            "selector_id": stage,
            "threshold_receipt_sha256": saved_hash,
            "resume_recovered_receipt": False,
            "source_commit": source_commit,
            "input_hashes": context["input_hashes"],
        })
    else:
        _append_ledger({
            "event": "SELECTOR_FALLBACK",
            "selector_id": stage,
            "threshold_receipt_sha256": saved_hash,
            "reason": chosen["status"],
            "threshold": THRESHOLD_FIXED_PRESENT,
            "source_commit": source_commit,
            "input_hashes": context["input_hashes"],
        })
    receipt = _read_verified_receipt(path)
    _validate_selection_receipt(receipt, context, y)
    return receipt


def _paired_error_changes(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                          subset: np.ndarray) -> dict[str, int]:
    target = _labels(y)
    b = np.asarray(baseline, dtype=bool)
    c = np.asarray(candidate, dtype=bool)
    mask = np.asarray(subset, dtype=bool)
    if target.shape != b.shape or target.shape != c.shape or target.shape != mask.shape:
        raise ValueError("Paired metric arrays differ in length")
    yy, bb, cc = target[mask], b[mask], c[mask]
    return {
        "rows": int(len(yy)),
        "class_0": int(np.sum(yy == 0)),
        "class_1": int(np.sum(yy == 1)),
        "fp_removed": int(np.sum((yy == 0) & bb & ~cc)),
        "fp_added": int(np.sum((yy == 0) & ~bb & cc)),
        "fn_removed": int(np.sum((yy == 1) & ~bb & cc)),
        "fn_added": int(np.sum((yy == 1) & bb & ~cc)),
        "labels_changed": int(np.sum(bb != cc)),
    }


def _evaluate_predictions(y: np.ndarray, data: Mapping[str, np.ndarray], threshold: float) -> dict[str, Any]:
    target = _labels(y)
    missing = np.asarray(data["missing"], dtype=bool)
    p_present = np.asarray(data["MIX50"], dtype=np.float64)
    p_a = np.asarray(data["pA"], dtype=np.float64)
    p_p = np.asarray(data["pP"], dtype=np.float64)
    selected = _fixed_predictions(p_present, p_a, missing, float(threshold))
    current_best = _fixed_predictions(p_present, p_a, missing, THRESHOLD_FIXED_PRESENT)
    old_mix = _old_mix_predictions(p_present, p_p, missing)
    if not np.array_equal(selected[~missing], current_best[~missing]):
        raise RuntimeError("Selected threshold altered frozen present-stream labels")
    selected_cm = confusion(target, selected)
    current_cm = confusion(target, current_best)
    old_cm = confusion(target, old_mix)
    result: dict[str, Any] = {
        "rows": int(len(target)),
        "missing_rows": int(missing.sum()),
        "present_rows": int((~missing).sum()),
        "selected_threshold": float(threshold),
        "selected": confusion_record(selected_cm),
        "current_best_fixed_pA_0_50": confusion_record(current_cm),
        "delta_macro_f1_vs_current_best": float(macro_f1_from_confusion(selected_cm) - macro_f1_from_confusion(current_cm)),
        "old_MIX50_fixed_pP_missing_descriptive_only": confusion_record(old_cm),
        "delta_macro_f1_vs_old_MIX50_descriptive_only": float(macro_f1_from_confusion(selected_cm) - macro_f1_from_confusion(old_cm)),
        "missing_only_selected": confusion_record(confusion(target[missing], selected[missing])),
        "missing_only_current_best_fixed_0_50": confusion_record(confusion(target[missing], current_best[missing])),
        "missing_only_delta_macro_f1_vs_current_best": float(
            macro_f1_from_confusion(confusion(target[missing], selected[missing]))
            - macro_f1_from_confusion(confusion(target[missing], current_best[missing]))
        ),
        "paired_missing_errors_vs_current_best": _paired_error_changes(target, current_best, selected, missing),
        "paired_full_row_errors_vs_current_best": _paired_error_changes(target, current_best, selected, np.ones(len(target), dtype=bool)),
        "present_labels_unchanged": True,
    }
    per_cycle: dict[str, Any] = {}
    cycles = np.asarray(data["cycles"], dtype=np.int16)
    for cycle in sorted(np.unique(cycles).tolist()):
        mask = cycles == cycle
        counts = confusion(target[mask], selected[mask])
        current_counts = confusion(target[mask], current_best[mask])
        per_cycle[str(int(cycle))] = {
            "rows": int(mask.sum()),
            "missing_rows": int(np.sum(missing[mask])),
            "selected": confusion_record(counts),
            "current_best_fixed_pA_0_50": confusion_record(current_counts),
            "delta_macro_f1_vs_current_best": float(macro_f1_from_confusion(counts) - macro_f1_from_confusion(current_counts)),
        }
    result["per_cycle"] = per_cycle
    subgroup_results: dict[str, Any] = {}
    for group_name in ("group_entry_friction", "group_text_load"):
        values = np.asarray(data[group_name]).astype(str)
        subgroup_results[group_name] = {}
        for value in sorted(np.unique(values).tolist()):
            mask = values == value
            y_group = target[mask]
            counts = confusion(y_group, selected[mask])
            base_counts = confusion(y_group, current_best[mask])
            supported = len(y_group) >= MIN_GROUP_ROWS and min(int(np.sum(y_group == 0)), int(np.sum(y_group == 1))) >= MIN_GROUP_PER_CLASS
            subgroup_results[group_name][str(value)] = {
                "status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED",
                "support": {"rows": int(len(y_group)), "class_0": int(np.sum(y_group == 0)), "class_1": int(np.sum(y_group == 1))},
                "selected": confusion_record(counts) if supported else None,
                "current_best_fixed_pA_0_50": confusion_record(base_counts) if supported else None,
                "delta_macro_f1_vs_current_best": float(macro_f1_from_confusion(counts) - macro_f1_from_confusion(base_counts)) if supported else None,
            }
    result["supported_subgroups"] = subgroup_results
    return result


def _evaluate_stage(stage: str, evaluation_view: Mapping[str, np.ndarray],
                    selection_receipt: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    receipt_path = _receipt_path(stage)
    persisted = _read_verified_receipt(receipt_path)
    if persisted.get("canonical_sha256") != selection_receipt.get("canonical_sha256"):
        raise RuntimeError(f"Evaluation attempted without its durable threshold receipt: {stage}")
    y = _labels_by_id(evaluation_view)
    _validate_planned_evaluation(stage, evaluation_view)
    ids = np.asarray(evaluation_view["ids"], dtype=str)
    cycles = np.asarray(evaluation_view["cycles"], dtype=np.int16)
    context = {
        "stage": stage,
        "evaluation_name": _stage_spec()[stage]["evaluation_name"],
        "source_commit": source_commit,
        "threshold_receipt_sha256": sha256_file(receipt_path),
        "evaluation_ids_sha256": mix._ids_digest(ids),
        "evaluation_cycles_sha256": _array_sha256(cycles, "<i2"),
        "evaluation_target_sha256": _array_sha256(y, "i1"),
        "evaluation_rows": int(len(ids)),
        "evaluation_cycle_values": [int(v) for v in np.unique(cycles)],
        "evaluation_missing_sha256": _array_sha256(np.asarray(evaluation_view["missing"], dtype=np.uint8), "u1"),
        "evaluation_input_hashes": {"train_sha256": TRAIN_SHA256, "oof_bundle_sha256": OOF_SHA256},
    }
    path = _eval_path(stage)
    state = _ledger_state(_read_ledger()).get(stage)
    if state is None or (state["saved"] is None and state["fallback"] is None):
        raise RuntimeError(f"Evaluation requires a persisted selector decision: {stage}")
    if path.exists():
        saved = _read_verified_receipt(path)
        for key, expected in context.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"Evaluation receipt context mismatch for {stage}: {key}")
        replay = _evaluate_predictions(y, evaluation_view, float(selection_receipt["threshold"]))
        if saved.get("results") != replay:
            raise RuntimeError(f"Evaluation receipt metrics fail replay for {stage}")
        if state["evaluation"] is None:
            _append_ledger({
                "event": "EVALUATION_SAVED",
                "selector_id": stage,
                "evaluation_receipt_sha256": sha256_file(path),
                "evaluation_name": _stage_spec()[stage]["evaluation_name"],
                "source_commit": source_commit,
                "resume_recovered_receipt": True,
            })
        return saved
    results = _evaluate_predictions(y, evaluation_view, float(selection_receipt["threshold"]))
    body = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        **context,
        "results": results,
        "test_accessed": False,
        "new_estimator_fits": 0,
        "new_preprocessing_fits": 0,
    }
    _write_once_json(path, body)
    _append_ledger({
        "event": "EVALUATION_SAVED",
        "selector_id": stage,
        "evaluation_receipt_sha256": sha256_file(path),
        "evaluation_name": _stage_spec()[stage]["evaluation_name"],
        "source_commit": source_commit,
    })
    return _read_verified_receipt(path)


def _pooled_unique_w5_w6(d1: Mapping[str, Any], d2: Mapping[str, Any]) -> dict[str, Any]:
    a = d1["results"]
    b = d2["results"]
    cm_names = ("selected", "current_best_fixed_pA_0_50", "old_MIX50_fixed_pP_missing_descriptive_only")
    pooled_rows = int(a["rows"] + b["rows"])
    pooled: dict[str, Any] = {"rows": pooled_rows, "windows": ["E67_W5", "E75_W6"], "overlap": False}
    if pooled_rows != EXPECTED_POOLED_W5_W6_ROWS:
        raise RuntimeError(f"Unique W5/W6 denominator differs from frozen {EXPECTED_POOLED_W5_W6_ROWS} rows")
    for name in cm_names:
        ca = a[name]["confusion_matrix"]
        cb = b[name]["confusion_matrix"]
        counts = tuple(ca[i][j] + cb[i][j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
        pooled[name] = confusion_record(counts)
    pooled["delta_macro_f1_selected_vs_current_best"] = float(
        pooled["selected"]["macro_f1"] - pooled["current_best_fixed_pA_0_50"]["macro_f1"]
    )
    pooled["delta_macro_f1_selected_vs_old_MIX50_descriptive_only"] = float(
        pooled["selected"]["macro_f1"] - pooled["old_MIX50_fixed_pP_missing_descriptive_only"]["macro_f1"]
    )
    return pooled


def _run_selectors_locked() -> dict[str, Any]:
    source_commit = verify_prefit_checkout()
    oof = _load_oof()
    spec = _stage_spec()
    selectors: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for stage in EXPECTED_STAGES:
        selection_view = _view_for_spec(oof, spec[stage]["selection"])
        selectors[stage] = _run_selection(stage, selection_view, source_commit)
        if spec[stage]["evaluation"]:
            evaluation_view = _view_for_spec(oof, spec[stage]["evaluation"])
            evaluations[stage] = _evaluate_stage(stage, evaluation_view, selectors[stage], source_commit)
    if len([s for s in selectors.values() if s["selection"]["threshold_selector_consumed"]]) > MAX_SELECTORS:
        raise RuntimeError("Selector cap exceeded")
    results = {
        "schema_version": 1,
        "status": "THREE_SELECTOR_PILOT_COMPLETE",
        "policy_id": POLICY_ID,
        "source_commit": source_commit,
        "input_hashes": {"train_sha256": TRAIN_SHA256, "oof_bundle_sha256": OOF_SHA256},
        "selectors": {
            stage: {
                "threshold": receipt["threshold"],
                "selection": receipt["selection"],
                "selection_receipt_sha256": sha256_file(_receipt_path(stage)),
            }
            for stage, receipt in selectors.items()
        },
        "evaluations": {
            stage: {
                "evaluation_name": receipt["evaluation_name"],
                "results": receipt["results"],
                "evaluation_receipt_sha256": sha256_file(_eval_path(stage)),
            }
            for stage, receipt in evaluations.items()
        },
        "pooled_unique_w5_w6": _pooled_unique_w5_w6(
            evaluations["D1_E67_W4_TO_W5"], evaluations["D2_E67_W4W5_TO_E75_W6"]
        ),
        "decision_selector_budget": {
            "cap": MAX_SELECTORS,
            "reserved": sum(1 for e in _read_ledger() if e.get("event") == "SELECTOR_RESERVED"),
            "saved_selectors": sum(1 for e in _read_ledger() if e.get("event") == "SELECTOR_SAVED" and "threshold_receipt_sha256" in e),
            "new_estimator_fits": 0,
            "new_preprocessing_fits": 0,
            "retries": 0,
        },
        "development_views_adaptive_not_fresh_holdouts": True,
        "test_accessed": False,
        "kaggle_contacted": False,
    }
    _write_once_json(RESULTS_PATH, results)
    _write_once_json(RUN_MANIFEST_PATH, {
        "schema_version": 1,
        "status": "THREE_SELECTOR_PILOT_OUTPUTS_PERSISTED",
        "policy_id": POLICY_ID,
        "approved_prefit_commit": source_commit,
        "input_hashes": results["input_hashes"],
        "results_sha256": sha256_file(RESULTS_PATH),
        "selector_receipt_sha256": {
            stage: sha256_file(_receipt_path(stage))
            for stage in EXPECTED_STAGES
        },
        "evaluation_receipt_sha256": {
            stage: sha256_file(_eval_path(stage))
            for stage in evaluations
        },
        "ledger_sha256": sha256_file(LEDGER_PATH),
        "selector_count": int(results["decision_selector_budget"]["reserved"]),
        "new_estimator_fits": 0,
        "new_preprocessing_fits": 0,
        "new_model_inference": 0,
        "test_accessed": False,
        "kaggle_contacted": False,
    })
    return results


def run_selectors() -> dict[str, Any]:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to enforce the two-thread project limit") from exc
    with mix._single_project_job(RUN_MANIFEST_PATH.parent):
        with threadpool_limits(limits=MAX_THREADPOOL_THREADS):
            return _run_selectors_locked()


def write_e79_application_freeze() -> dict[str, Any]:
    source_commit = verify_prefit_checkout()
    if not RESULTS_PATH.exists():
        raise RuntimeError("Selector results are required before the E79 application freeze")
    results = _read_verified_receipt(RESULTS_PATH)
    final = results["selectors"]["FINAL_STITCHED_E79"]
    final_receipt_path = _receipt_path("FINAL_STITCHED_E79")
    final_receipt = _read_verified_receipt(final_receipt_path)
    freeze = _read_json(PREFIT_FREEZE_PATH)
    body = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_E79_PROBABILITY_BUNDLE_ACCESS",
        "policy_id": POLICY_ID,
        "source_commit": source_commit,
        "final_selector_threshold": float(final["threshold"]),
        "final_selector_receipt_sha256": sha256_file(final_receipt_path),
        "prefit_freeze_sha256": sha256_file(PREFIT_FREEZE_PATH),
        "application_input_hashes": freeze["e79_application_input_hashes"],
        "formula": "present=current best MIX50 >=0.50; missing=original A02 pA >= final selector threshold",
        "threshold_operator": ">=",
        "test_labels_read": False,
        "sample_submission_read": False,
        "new_model_inference": 0,
        "new_estimator_fits": 0,
        "kaggle_upload": False,
        "final_selection_status": final_receipt["selection"]["status"],
    }
    _write_once_json(E79_APPLICATION_FREEZE_PATH, body)
    return {**body, "application_freeze_sha256": sha256_file(E79_APPLICATION_FREEZE_PATH)}


def _verify_application_freeze_committed() -> tuple[dict[str, Any], str]:
    if not E79_APPLICATION_FREEZE_PATH.exists():
        raise RuntimeError("E79 application freeze must be written before E79 probability access")
    status = subprocess.check_output(["git", "status", "--porcelain", "--", E79_APPLICATION_FREEZE_PATH.relative_to(ROOT).as_posix()], cwd=ROOT, text=True).strip()
    if status:
        raise RuntimeError("E79 application freeze is not committed")
    source_commit = _head()
    body = _read_verified_receipt(E79_APPLICATION_FREEZE_PATH)
    final_receipt_path = _receipt_path("FINAL_STITCHED_E79")
    if body.get("final_selector_receipt_sha256") != sha256_file(final_receipt_path):
        raise RuntimeError("Application freeze does not bind the final threshold receipt")
    if body.get("status") != "FROZEN_BEFORE_E79_PROBABILITY_BUNDLE_ACCESS":
        raise RuntimeError("Application freeze status mismatch")
    return body, source_commit


def _read_test_ids_missing(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        cols = {name: i for i, name in enumerate(header)}
        if "record_id" not in cols or "signal_stream" not in cols:
            raise RuntimeError("Test lacks record_id or signal_stream")
        ids: list[str] = []
        missing: list[bool] = []
        max_index = max(cols["record_id"], cols["signal_stream"])
        for row in reader:
            if len(row) <= max_index:
                raise RuntimeError("Malformed Test CSV row")
            ids.append(row[cols["record_id"]])
            missing.append(row[cols["signal_stream"]].strip() == "")
    return mix._safe_ids(np.asarray(ids, dtype="U"), label="E79 Test IDs").tolist(), np.asarray(missing, dtype=bool)


def _read_submission(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "artifact_signal"]:
            raise RuntimeError(f"Unexpected submission schema: {reader.fieldnames}")
        rows = list(reader)
    ids = mix._safe_ids(np.asarray([row["record_id"] for row in rows], dtype="U"), label="submission IDs").tolist()
    labels = np.asarray([int(row["artifact_signal"]) for row in rows], dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Submission labels are not integer 0/1")
    return ids, labels


def _build_e79_candidate_locked() -> dict[str, Any]:
    application_freeze, application_commit = _verify_application_freeze_committed()
    # E79 bundle/Test hashes are opened only after the committed application freeze.
    input_hashes = validate_inputs(include_e79=True)
    if input_hashes != application_freeze["application_input_hashes"]:
        raise RuntimeError("E79 application inputs differ from the committed application freeze")
    with np.load(E79_NPZ, allow_pickle=False) as archive:
        required = {"record_ids", "missing_mask", "posoffset_probability", "a02_probability"}
        if not required.issubset(set(archive.files)):
            raise RuntimeError("Frozen E79 probability bundle lacks required arrays")
        bundle_ids = mix._safe_ids(np.asarray(archive["record_ids"], dtype="U"), label="E79 probability IDs")
        missing = np.asarray(archive["missing_mask"], dtype=bool)
        p_p = _validate_probability_vector(archive["posoffset_probability"], "E79 pP")
        p_a = _validate_probability_vector(archive["a02_probability"], "E79 pA")
    if len(bundle_ids) != 29421 or missing.shape != (29421,) or p_p.shape != (29421,) or p_a.shape != (29421,):
        raise RuntimeError("E79 probability arrays have unexpected row counts")
    test_ids, test_missing = _read_test_ids_missing(OFFICIAL_TEST)
    base_ids, base_labels = _read_submission(CURRENT_BEST_CSV)
    if bundle_ids.tolist() != test_ids or bundle_ids.tolist() != base_ids:
        raise RuntimeError("E79, official Test, and current-best IDs/order differ")
    if not np.array_equal(missing, test_missing):
        raise RuntimeError("E79 missing mask differs from official Test signal_stream")
    blend = 0.5 * p_p + 0.5 * p_a
    expected_current_best = np.where(missing, p_a >= THRESHOLD_FIXED_PRESENT, blend >= THRESHOLD_FIXED_PRESENT).astype(np.int8)
    if not np.array_equal(base_labels, expected_current_best):
        raise RuntimeError("Current best E79 CSV does not replay from frozen route probabilities")
    threshold = float(application_freeze["final_selector_threshold"])
    candidate_labels = np.where(missing, p_a >= threshold, blend >= THRESHOLD_FIXED_PRESENT).astype(np.int8)
    if not np.array_equal(candidate_labels[~missing], base_labels[~missing]):
        raise RuntimeError("Final threshold application changed present labels")
    missing_changes = route_audit.compare_missing_labels(base_labels, candidate_labels, missing)
    duplicate_scan = route_audit.scan_duplicate_full_vectors(
        bundle_ids.tolist(), candidate_labels,
        root=ROOT / "submissions", exclude=OUTPUT_CSV,
    )
    csv_bytes = ("record_id,artifact_signal\n" + "".join(
        f"{rid},{int(label)}\n" for rid, label in zip(bundle_ids.tolist(), candidate_labels.tolist())
    )).encode("utf-8")
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        if sha256_file(OUTPUT_CSV) != csv_sha:
            raise RuntimeError("Refusing to overwrite a different E79 candidate CSV")
    else:
        with OUTPUT_CSV.open("xb") as stream:
            stream.write(csv_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    output_ids, output_labels = _read_submission(OUTPUT_CSV)
    if output_ids != test_ids or not np.array_equal(output_labels, candidate_labels):
        raise RuntimeError("Final E79 CSV did not replay exactly")
    manifest_body = {
        "schema_version": 1,
        "status": "SKIP_UPLOAD_EXACT_DUPLICATE" if duplicate_scan["duplicate_found"] else "READY_FOR_ROOT_REVIEW_ONE_UPLOAD_MAX",
        "policy_id": POLICY_ID,
        "formula": "present=current best MIX50 >=0.50; missing=original A02 pA >= final selector threshold",
        "selected_missing_threshold": threshold,
        "threshold_operator": ">=",
        "final_selector_receipt_sha256": application_freeze["final_selector_receipt_sha256"],
        "application_freeze_sha256": sha256_file(E79_APPLICATION_FREEZE_PATH),
        "application_freeze_commit": application_commit,
        "prefit_freeze_sha256": sha256_file(PREFIT_FREEZE_PATH),
        "prefit_source_commit": _read_json(PREFIT_FREEZE_PATH)["base_commit"],
        "source_hashes": _read_json(PREFIT_FREEZE_PATH)["source_hashes"],
        "input_hashes": input_hashes,
        "current_best_csv_sha256": CURRENT_BEST_CSV_SHA256,
        "csv_path": OUTPUT_CSV.relative_to(ROOT).as_posix(),
        "csv_sha256": csv_sha,
        "rows": int(len(candidate_labels)),
        "unique_ids": int(len(np.unique(bundle_ids))),
        "columns": ["record_id", "artifact_signal"],
        "candidate_positive_count": int(candidate_labels.sum()),
        "missing_stream_rows": int(missing.sum()),
        "present_stream_rows": int((~missing).sum()),
        "missing_label_comparison_vs_current_best": missing_changes,
        "present_labels_bit_exact_to_current_best": True,
        "duplicate_full_label_scan": duplicate_scan,
        "new_decision_selectors": int(_read_json(RESULTS_PATH)["decision_selector_budget"]["reserved"]),
        "new_estimator_fits": 0,
        "new_preprocessing_fits": 0,
        "new_model_inference": 0,
        "test_fields_read": ["record_id", "signal_stream"],
        "test_labels_read": False,
        "sample_submission_read": False,
        "kaggle_contacted": False,
        "upload_performed": False,
    }
    manifest_hash = _write_once_json(MANIFEST_PATH, manifest_body)
    qa_body = {
        "schema_version": 1,
        "status": "PASS_LOCAL_QA_UPLOAD_BLOCKED_DUPLICATE" if duplicate_scan["duplicate_found"] else "PASS_LOCAL_E79_QA",
        "policy_id": POLICY_ID,
        "manifest_sha256": manifest_hash,
        "checks": {
            "application_freeze_committed_before_E79_bundle_access": True,
            "expected_E79_input_hashes_verified_after_freeze": True,
            "bundle_Test_current_best_ID_order_exact": True,
            "missing_mask_matches_Test_signal_stream": True,
            "current_best_fixed_policy_replayed": True,
            "present_labels_bit_exact_to_current_best": True,
            "CSV_29421_order_schema_binary_replayed": True,
            "full_label_duplicate_scan_completed": True,
            "duplicate_vector_found": duplicate_scan["duplicate_found"],
            "test_labels_not_read": True,
            "sample_submission_not_read": True,
            "new_estimator_fits": 0,
            "new_preprocessing_fits": 0,
            "new_model_inference": 0,
            "kaggle_not_contacted": True,
        },
    }
    qa_hash = _write_once_json(QA_PATH, qa_body)
    return {
        "csv_path": str(OUTPUT_CSV),
        "csv_sha256": csv_sha,
        "manifest_sha256": manifest_hash,
        "qa_sha256": qa_hash,
        "selected_threshold": threshold,
        "missing_label_comparison": missing_changes,
        "duplicate_full_label_scan": duplicate_scan,
    }


def build_e79_candidate() -> dict[str, Any]:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to enforce the two-thread project limit") from exc
    with mix._single_project_job(RUN_MANIFEST_PATH.parent):
        with threadpool_limits(limits=MAX_THREADPOOL_THREADS):
            return _build_e79_candidate_locked()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-selectors", action="store_true")
    group.add_argument("--write-e79-freeze", action="store_true")
    group.add_argument("--build-e79", action="store_true")
    args = parser.parse_args()
    if args.run_selectors:
        result = run_selectors()
    elif args.write_e79_freeze:
        result = write_e79_application_freeze()
    else:
        result = build_e79_candidate()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
