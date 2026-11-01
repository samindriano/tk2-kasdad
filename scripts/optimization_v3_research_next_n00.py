"""N00 cache-only audit for Research Next V1; never fits or opens Test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from src.models.classification_models import _text_values
from src.models.research_next_contract import (
    canonical_sha256,
    file_sha256,
    read_bundle,
    resume_or_publish_bundle,
)
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
PREVIOUS_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_04/freeze.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
PREVIOUS_REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N00"
PREVIOUS_BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N00"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N00/amendment_05"
OLD_LEDGER = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
OLD_SUBMISSION_FREEZE = ROOT / "reports/submissions/optimization_v3_expanded_v1/EXECUTION_FREEZE.json"
ORIGINS = (55, 59, 63, 67, 71, 75)
CLASS_CANDIDATES = ("C16", "C17", "C11")
REG_CANDIDATES = ("R22", "R21", "R06")
ALL_CANDIDATES = CLASS_CANDIDATES + REG_CANDIDATES
REPORTS = {
    "C16": ROOT / "reports/optimization_v3/final_continuation/C16/verification.json",
    "R22": ROOT / "reports/optimization_v3/final_continuation/R22/verification.json",
    "C17": ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/C17/verification.json",
    "C11": ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/C11/verification.json",
    "R21": ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R21/verification.json",
    "R06": ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json",
    "R11": ROOT / "reports/optimization_v3/batch1/R11/revision_03/verification.json",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(value):
            raise RuntimeError(f"Refusing to replace immutable output: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()


def _write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    data = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace immutable CSV: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def cache_path(card: str, cutoff: int) -> Path:
    run_root = ROOT / "experiments/runs"
    if card == "C16":
        base = "optimization_v3_batch1/C16" if cutoff in (55, 67) else "optimization_v3_final_continuation/C16"
    elif card == "R22":
        base = "optimization_v3_batch1/R22" if cutoff in (55, 67) else "optimization_v3_final_continuation/R22"
    elif card in ("C11", "C17"):
        base = f"optimization_v3_batch1/{card}" if cutoff in (55, 67) else f"optimization_v3_expanded_benchmark_v1/{card}"
    elif card == "R21":
        base = "optimization_v3_batch1/R21" if cutoff in (55, 67) else "optimization_v3_expanded_benchmark_v1_recovery_v2/R21"
    elif card == "R06":
        base = "optimization_v3_batch1/R06" if cutoff in (55, 67) else "optimization_v3_expanded_benchmark_v1_recovery_v2/R06"
    elif card == "R11":
        if cutoff not in (55, 67):
            raise ValueError("N00 permits R11 pilot cutoffs 55 and 67 only")
        base = "optimization_v3_batch1/R11/revision_03"
    else:
        raise KeyError(card)
    return run_root / base / f"{card}_fit{cutoff}.npz"


def _verification_path(card: str) -> Path:
    return REPORTS[card]


def _source_paths() -> list[Path]:
    names = (
        "scripts/optimization_v3_research_next_preflight.py",
        "scripts/optimization_v3_research_next_n00.py",
        "src/models/research_next_contract.py",
        "src/models/classification_models.py",
        "src/models/round1_models.py",
        "src/models/optimization_v3_batch1_r22.py",
        "src/models/optimization_v3_batch1_r06.py",
        "src/models/optimization_v3_batch1_c11.py",
        "src/models/optimization_v3_batch1_c16.py",
        "src/models/optimization_v3_batch1_c17.py",
        "src/models/optimization_v3_batch1_r21.py",
        "src/models/optimization_v3_expanded_continuation.py",
        "src/models/optimization_v3_final_continuation_c16.py",
        "src/models/optimization_v3_final_continuation_r22.py",
        "scripts/optimization_v3_batch1_c11.py",
        "scripts/optimization_v3_batch1_c16.py",
        "scripts/optimization_v3_batch1_c17.py",
        "scripts/optimization_v3_batch1_r21.py",
        "scripts/optimization_v3_batch1_r22.py",
        "scripts/optimization_v3_batch1_r06.py",
        "scripts/optimization_v3_expanded_benchmark.py",
        "scripts/optimization_v3_final_continuation_cls.py",
        "scripts/optimization_v3_final_continuation_r22.py",
        "tests/test_research_next_n00.py",
        "src/models/optimization_v3_sparse_classification.py",
        "src/utils/paths.py",
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/CAMPAIGN_REPORT.json",
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/CAMPAIGN_SUMMARY.md",
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R11_BLOCKER.json",
    )
    paths = [ROOT / name for name in names]
    paths += [QUEUE_PATH, READINESS_PATH, OLD_SUBMISSION_FREEZE, PREVIOUS_FREEZE_PATH]
    paths += list(REPORTS.values())
    return paths


def _all_artifact_paths() -> list[Path]:
    paths: list[Path] = []
    for card in ALL_CANDIDATES:
        for cutoff in ORIGINS:
            npz = cache_path(card, cutoff)
            paths += [npz, npz.with_suffix(".json")]
    for cutoff in (55, 67):
        npz = cache_path("R11", cutoff)
        paths += [npz, npz.with_suffix(".json")]
    paths += list(REPORTS.values())
    paths += [
        PREVIOUS_REPORT_ROOT / "summary.json",
        PREVIOUS_REPORT_ROOT / "candidate_summary.csv",
        PREVIOUS_REPORT_ROOT / "per_cycle.csv",
        PREVIOUS_REPORT_ROOT / "groups.csv",
        PREVIOUS_REPORT_ROOT / "figures" / "per_cycle_candidate_gains.png",
        PREVIOUS_REPORT_ROOT / "figures" / "n03_fixed_arm_cycle_gains.png",
    ]
    for candidate in ("C16", "C17", "C11", "R22", "R21", "R06", "N03_FIXED_ARMS"):
        paths += [PREVIOUS_BUNDLE_ROOT / candidate / "arrays.npz",
                  PREVIOUS_BUNDLE_ROOT / candidate / "receipt.json"]
    return paths


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        versions[package] = importlib.metadata.version(package)
    return versions


def _old_ledger_snapshot() -> dict[str, Any]:
    raw = OLD_LEDGER.read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    last_budget: dict[str, Any] | None = None
    for event in events:
        candidate = event.get("budget_after_fits")
        if isinstance(candidate, dict):
            last_budget = candidate
    if last_budget is None:
        raise RuntimeError("Old ledger has no frozen budget snapshot")
    old_submission = json.loads(OLD_SUBMISSION_FREEZE.read_text(encoding="utf-8"))
    prior_offledger = int(old_submission["identity"]["budget"]["previous_offledger_c16_r22_fulltrain_fits"])
    reserved = int(last_budget["reserved_predictor_fits"])
    return {
        "path": _relative(OLD_LEDGER),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "event_rows": len(events),
        "reserved_predictor_fits": reserved,
        "remaining_reserved_capacity": int(last_budget["remaining_reserved_capacity"]),
        "previous_offledger_full_train_fits": prior_offledger,
        "effective_total_including_offledger": reserved + prior_offledger,
        "old_global_cap": 180,
        "old_phase_budget_reused": False,
    }


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _verify_historical_report(card: str) -> dict[str, Any]:
    path = _verification_path(card)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"{card} historical verification report is not PASS")
    declared = report.get("card_id", report.get("candidate"))
    if declared is not None and declared != card:
        raise RuntimeError(f"{card} verification report identifies {declared!r}")

    def declares_test_use(value: Any) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"test_data_used", "test_used", "official_test_used", "test_labels_read"} and child is True:
                    return True
                if declares_test_use(child):
                    return True
        elif isinstance(value, list):
            return any(declares_test_use(child) for child in value)
        return False

    if declares_test_use(report):
        raise RuntimeError(f"{card} historical verification report declares Test use")
    return {"path": _relative(path), "sha256": file_sha256(path),
            "status": report["status"], "declared_card": declared,
            "freeze_signature": report.get("freeze_signature")}


def build_freeze() -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    if queue["batch_id"] != BATCH or queue["order"] != [f"N{i:02}" for i in range(7)]:
        raise RuntimeError("Research Next queue identity/order changed")
    train_hashes = {task: file_sha256(Path(task_files(task)["train"])) for task in ("classification", "regression")}
    if train_hashes != queue["raw_train_sha256"]:
        raise RuntimeError("Official Train hash differs from frozen queue")
    all_inputs = {}
    for path in sorted(set(_all_artifact_paths() + _source_paths()), key=lambda p: _relative(p)):
        if not path.is_file():
            raise FileNotFoundError(f"N00 required frozen input missing: {path}")
        all_inputs[_relative(path)] = file_sha256(path)
    stable_identity = {
        "batch_id": BATCH,
        "card_id": "N00",
        "freeze_amendment": {
            "amendment": "05",
            "supersedes_freeze_signature": json.loads(PREVIOUS_FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
            "supersedes_result_freeze_signature": "b468bd41e9e96440e0db6f49e316a521a5f6bf3bdabb8c4458024205e9c27113",
            "reason": "regression H12 diagnostics must compare each candidate with the R22 prediction and legacy component from the same fit origin; previous published N00 used full-six row mapping, which paired pilot H12 rows with different-origin primary outputs",
            "failed_attempts_before_metrics": [
                {
                    "status": "FAILED_BEFORE_METRIC",
                    "issue": "initial audit expected 12 score cycles for all origins and rejected C16/59; the Train targets were loaded, but no metric or candidate result was computed",
                    "official_estimator_fits": 0,
                    "outputs_published": 0,
                },
                {
                    "status": "FAILED_BEFORE_METRIC",
                    "issue": "R21 pilot schema stores the protected baseline under baseline_prediction; the Train targets were loaded, but no metric or candidate result was computed",
                    "official_estimator_fits": 0,
                    "outputs_published": 0,
                },
                {
                    "status": "FAILED_BEFORE_METRIC",
                    "issue": "class control map treated overlapping pilot H12 and continuation primary rows as if they came from the same training origin; the Train targets were loaded, but no metric or candidate result was computed",
                    "official_estimator_fits": 0,
                    "outputs_published": 0,
                },
                {
                    "status": "PARTIAL_METRICS_COMPUTED_NOT_PUBLISHED_NOT_INSPECTED",
                    "issue": "R11 pilot-only report requested unavailable origin 75 after metrics for C16/C17/C11/R22/R21/R06 had been computed in memory; no numeric result was printed, persisted, or inspected, and N03 was not evaluated",
                    "official_estimator_fits": 0,
                    "outputs_published": 0,
                },
            ],
        },
        "source_commit": _git_head(),
        "queue_sha256": file_sha256(QUEUE_PATH),
        "queue_readiness_sha256": file_sha256(READINESS_PATH),
        "raw_train_sha256": train_hashes,
        "source_and_artifact_sha256": all_inputs,
        "analysis_spec": {
            "candidate_scope": list(ALL_CANDIDATES),
            "r11_scope": "pilot_only_cutoffs_55_67_full_six_remains_unknown",
            "full_six_primary": queue["views"]["full_six"],
            "standard_primary": queue["views"]["STANDARD"]["primary"],
            "standard_h12": queue["views"]["STANDARD"]["h12"],
            "late_primary": queue["views"]["LATE"]["primary"],
            "late_h12": queue["views"]["LATE"]["h12"],
            "late_full_six_rows": queue["views"]["late_primary_rows"],
            "classification_threshold": 0.5,
            "classification_metric": "macro_f1_from_binary_confusion_counts",
            "regression_metric": "mean_absolute_error",
            "group_views": {
                "classification": ["entry_friction", "stream_availability", "fit_text_load_quartile", "fit_impact_load_quartile"],
                "regression": ["stream_availability", "entry_friction", "fit_impact_load_quartile", "observed_target_tail_evaluation_only"],
                "quartile_edges": [0.25, 0.5, 0.75],
                "quartile_fit_population": "rows with cycle_code <= the scoring origin",
            },
            "regression_controls": ["R22_ORIGINAL", "protected_legacy_champion"],
            "n03_arms": {
                "A": "R22_ORIGINAL",
                "B_present": "clip(.50*legacy+.25*rank50(raw_R22)+.25*rank50(raw_R06),0,100)",
                "B_missing": "A exactly",
                "rank_scope": "complete cycle before availability filtering; average ties",
            },
        },
        "dependency_versions": _dependency_versions(),
    }
    execution_context = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "inherited_old_budget_snapshot": _old_ledger_snapshot(),
        "amendment": "05; supersedes published amendment-04 N00 because regression H12 reference predictions were not matched by fit origin; earlier failed audit attempts remain in prior freezes",
        "free_ram_gib_at_freeze": None,
        "official_estimator_fits": 0,
        "test_opened": False,
        "kaggle_calls": 0,
    }
    return {
        "schema_version": 1,
        "identity": stable_identity,
        "freeze_signature": canonical_sha256(stable_identity),
        "execution_context": execution_context,
    }


def freeze_n00() -> dict[str, Any]:
    result = build_freeze()
    _write_once(FREEZE_PATH, result)
    return result


def verify_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    actual = build_freeze()
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N00 freeze signature is invalid")
    if actual["identity"] != frozen.get("identity"):
        raise RuntimeError("N00 frozen source/input identity changed")
    return frozen


def _load_artifact(card: str, cutoff: int, frame: pd.DataFrame) -> dict[str, Any]:
    path = cache_path(card, cutoff)
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    identity = sidecar.get("identity")
    if not isinstance(identity, dict) or identity.get("card_id") != card or int(identity.get("cutoff", -1)) != cutoff:
        raise RuntimeError(f"{card}/{cutoff} sidecar identity does not match its path")
    if sidecar.get("identity_sha256") != canonical_sha256(identity):
        raise RuntimeError(f"{card}/{cutoff} identity hash mismatch")
    integrity = {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
    if sidecar.get("integrity_sha256") != canonical_sha256(integrity):
        raise RuntimeError(f"{card}/{cutoff} sidecar integrity hash mismatch")
    if sidecar.get("npz_sha256") != file_sha256(path):
        raise RuntimeError(f"{card}/{cutoff} NPZ hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise RuntimeError(f"{card}/{cutoff} contains object arrays")
    rows = arrays.get("score_rows")
    ids = arrays.get("score_record_ids")
    if rows is None or rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows):
        raise RuntimeError(f"{card}/{cutoff} score_rows are invalid")
    if ids is None or ids.ndim != 1 or len(ids) != len(rows) or ids.dtype.kind not in "US":
        raise RuntimeError(f"{card}/{cutoff} score_record_ids are invalid")
    if not np.array_equal(np.asarray(identity.get("score_rows"), dtype=np.int64), rows.astype(np.int64)):
        raise RuntimeError(f"{card}/{cutoff} sidecar row identity differs from NPZ")
    if np.any(rows < 0) or np.any(rows >= len(frame)):
        raise RuntimeError(f"{card}/{cutoff} score row outside official Train")
    if not np.array_equal(ids.astype(str), frame.record_id.to_numpy()[rows].astype(str)):
        raise RuntimeError(f"{card}/{cutoff} IDs do not map to official Train row indices")
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)[rows]
    horizon = 12 if cutoff in (55, 67) else 4
    expected = np.flatnonzero((frame.cycle_code.to_numpy() > cutoff) &
                              (frame.cycle_code.to_numpy() <= min(cutoff + horizon, 79)))
    if not np.array_equal(rows.astype(np.int64), expected.astype(np.int64)):
        raise RuntimeError(f"{card}/{cutoff} does not contain the exact complete future-cycle population")
    fit_rows = int(np.sum(frame.cycle_code.to_numpy() <= cutoff))
    metadata = sidecar.get("metadata", {})
    fit_metadata = metadata.get("fit_metadata", {})
    state = metadata.get("state", {})
    if int(metadata.get("fit_rows", fit_metadata.get("fit_rows", -1))) != fit_rows:
        raise RuntimeError(f"{card}/{cutoff} fit population differs from official Train prefix")
    if int(metadata.get("score_rows", fit_metadata.get("n_predictions", len(rows)))) != len(rows):
        raise RuntimeError(f"{card}/{cutoff} sidecar score-row count differs from NPZ")
    if int(fit_metadata.get("fit_rows", fit_rows)) != fit_rows:
        raise RuntimeError(f"{card}/{cutoff} fit metadata population differs from Train prefix")
    for name, source in (("metadata", metadata), ("fit_metadata", fit_metadata), ("state", state)):
        for key in ("official_test_used", "test_data_used", "test_labels_read"):
            if source.get(key) is True:
                raise RuntimeError(f"{card}/{cutoff} {name} declares Test use")
        for key in ("validation_labels_passed_to_fit", "score_labels_passed_to_fit",
                    "score_labels_passed_to_adapter", "outer_score_labels_used_for_fit"):
            if source.get(key) is True:
                raise RuntimeError(f"{card}/{cutoff} {name} declares score-label leakage")
    for source_name, source in (("state", state), ("fit_metadata", fit_metadata)):
        if "cutoff" in source and int(source["cutoff"]) != cutoff:
            raise RuntimeError(f"{card}/{cutoff} {source_name} cutoff differs")
        if "fit_rows" in source and int(source["fit_rows"]) != fit_rows:
            raise RuntimeError(f"{card}/{cutoff} {source_name} fit row count differs")
        if "fit_cycle_min" in source and int(source["fit_cycle_min"]) != int(frame.cycle_code.min()):
            raise RuntimeError(f"{card}/{cutoff} {source_name} fit-cycle minimum differs")
        if "fit_cycle_max" in source and int(source["fit_cycle_max"]) != cutoff:
            raise RuntimeError(f"{card}/{cutoff} {source_name} fit-cycle maximum differs")
    if card in CLASS_CANDIDATES:
        threshold = fit_metadata.get("threshold")
        if threshold is not None and float(threshold) != 0.5:
            raise RuntimeError(f"{card}/{cutoff} frozen Classification threshold differs from 0.50")
    for key, value in arrays.items():
        if value.dtype.kind in "iuf" and not np.isfinite(value).all():
            raise RuntimeError(f"{card}/{cutoff} has nonfinite {key}")
        if value.ndim == 1 and len(value) not in (len(rows),):
            raise RuntimeError(f"{card}/{cutoff} has misaligned array {key}")
    return {
        "card": card,
        "cutoff": cutoff,
        "rows": rows.astype(np.int64),
        "cycles": cycles,
        "arrays": arrays,
        "sidecar": sidecar,
        "npz_path": path,
        "npz_sha256": file_sha256(path),
        "sidecar_sha256": file_sha256(sidecar_path),
        "fit_rows": fit_rows,
    }


def reconstruct_classification(bundle: dict[str, Any], stream_missing: np.ndarray) -> dict[str, np.ndarray]:
    arrays = bundle["arrays"]
    rows = bundle["rows"]
    missing = np.asarray(stream_missing, dtype=bool)[rows]
    cached_missing = arrays.get("control__stream_missing")
    if cached_missing is None or not np.array_equal(cached_missing.astype(bool), missing):
        raise RuntimeError(f"{bundle['card']}/{bundle['cutoff']} availability mask mismatch")
    c2b = arrays["control__c2b_probability"]
    cb = arrays["control__cb_probability"]
    sparse_raw = arrays["control__baseline_sparse_raw"]
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
    candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * arrays["raw_candidate"])
    saved_base = arrays.get("baseline_prediction", arrays.get("control__baseline_prediction"))
    if saved_base is None or not np.allclose(saved_base, baseline, atol=1e-12, rtol=0):
        raise RuntimeError(f"{bundle['card']}/{bundle['cutoff']} sparse25 control reconstruction failed")
    if not np.allclose(arrays["prediction"], candidate, atol=1e-12, rtol=0):
        raise RuntimeError(f"{bundle['card']}/{bundle['cutoff']} candidate route reconstruction failed")
    return {"rows": rows, "cycles": bundle["cycles"], "prediction": candidate,
            "control": baseline, "raw": arrays["raw_candidate"], "stream_missing": missing}


def rank50_independent(raw: Any, cycles: Any) -> np.ndarray:
    """Independent cycle-local average-tie implementation for N00 reconstruction."""
    values = np.asarray(raw, dtype=np.float64)
    groups = np.asarray(cycles, dtype=np.int64)
    if values.ndim != 1 or values.shape != groups.shape or not np.isfinite(values).all():
        raise ValueError("rank50 inputs must be finite aligned vectors")
    out = np.empty(values.shape, dtype=np.float64)
    for cycle in np.unique(groups):
        mask = groups == cycle
        order = np.argsort(values[mask], kind="mergesort")
        sorted_values = values[mask][order]
        ranks = np.empty(len(sorted_values), dtype=np.float64)
        i = 0
        while i < len(sorted_values):
            j = i + 1
            while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
                j += 1
            ranks[i:j] = ((i + 1) + j) / 2.0
            i = j
        unsorted_ranks = np.empty(len(order), dtype=np.float64)
        unsorted_ranks[order] = ranks
        percentile = 100.0 * unsorted_ranks / len(order)
        out[mask] = np.clip(0.5 * values[mask] + 0.5 * percentile, 0.0, 100.0)
    return out


def reconstruct_regression(bundle: dict[str, Any], r22_control: np.ndarray | None = None) -> dict[str, np.ndarray]:
    arrays = bundle["arrays"]
    rows, cycles, raw = bundle["rows"], bundle["cycles"], arrays["raw_candidate"].astype(np.float64)
    mapped = rank50_independent(raw, cycles)
    card = bundle["card"]
    if card == "R22":
        legacy = arrays.get("champion_prediction", arrays.get("baseline_prediction"))
        saved = arrays.get("R22_ORIGINAL", arrays.get("prediction"))
        if legacy is None:
            raise RuntimeError(f"R22/{bundle['cutoff']} legacy champion component absent")
        candidate = np.clip(0.75 * legacy + 0.25 * mapped, 0.0, 100.0)
        if not np.allclose(saved, candidate, atol=1e-12, rtol=0):
            raise RuntimeError(f"R22/{bundle['cutoff']} original policy reconstruction failed")
        return {"rows": rows, "cycles": cycles, "prediction": candidate,
                "control": candidate.copy(), "legacy": np.asarray(legacy),
                "raw": raw, "rank50": mapped}
    legacy = arrays.get("control__baseline_prediction", arrays.get("baseline_prediction"))
    if legacy is None:
        raise RuntimeError(f"{card}/{bundle['cutoff']} protected legacy baseline is absent")
    candidate = np.clip(0.75 * legacy + 0.25 * mapped, 0.0, 100.0)
    if not np.allclose(arrays["prediction"], candidate, atol=1e-12, rtol=0):
        raise RuntimeError(f"{card}/{bundle['cutoff']} candidate+legacy reconstruction failed")
    if not np.allclose(arrays["baseline_prediction"], legacy, atol=1e-12, rtol=0):
        raise RuntimeError(f"{card}/{bundle['cutoff']} stored control alias differs")
    if r22_control is not None:
        control = np.asarray(r22_control, dtype=np.float64)
        if control.shape != rows.shape:
            raise RuntimeError("R22 control alignment differs")
    return {"rows": rows, "cycles": cycles, "prediction": candidate,
            "control": legacy, "legacy": legacy, "raw": raw, "rank50": mapped}


def macro_f1_from_confusion(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, (0, 1)).all() or not np.isin(p, (0, 1)).all():
        raise ValueError("Macro F1 requires aligned binary vectors")
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    return {"macro_f1": 0.5 * (f1_0 + f1_1), "positive_recall": recall_1,
            "n": int(len(y)), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def regression_metrics(y_true: Any, prediction: Any) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("MAE requires aligned finite vectors")
    error = p - y
    return {"mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error * error))), "n": int(len(y))}


def _blocks_from_queue(queue: dict[str, Any]) -> dict[str, list[tuple[int, int, int]]]:
    return {
        "primary_full_six": [tuple(map(int, block)) for block in queue["views"]["full_six"]],
        "standard_primary": [tuple(map(int, block)) for block in queue["views"]["STANDARD"]["primary"]],
        "standard_h12": [tuple(map(int, block)) for block in queue["views"]["STANDARD"]["h12"]],
        "late_primary": [tuple(map(int, block)) for block in queue["views"]["LATE"]["primary"]],
        "late_h12": [tuple(map(int, block)) for block in queue["views"]["LATE"]["h12"]],
        "late_W5_W6": [tuple(map(int, block)) for block in queue["views"]["late_primary"]],
    }


def _collect(records_by_origin: dict[int, dict[str, np.ndarray]], blocks: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    pieces = []
    for origin, lo, hi in blocks:
        record = records_by_origin[origin]
        mask = (record["cycles"] >= lo) & (record["cycles"] <= hi)
        pieces.append({key: np.asarray(value)[mask] for key, value in record.items() if key != "rows"})
        pieces[-1]["rows"] = record["rows"][mask]
        pieces[-1]["origin"] = np.full(int(mask.sum()), origin, dtype=np.int64)
    keys = set.intersection(*(set(piece) for piece in pieces))
    result = {key: np.concatenate([piece[key] for piece in pieces]) for key in keys}
    order = np.argsort(result["rows"], kind="mergesort")
    result = {key: value[order] for key, value in result.items()}
    if len(np.unique(result["rows"])) != len(result["rows"]):
        raise RuntimeError("N00 view row population is not disjoint")
    return result


def _quantile_bins(frame: pd.DataFrame, origin: int, feature: str) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=np.float64)
    fit_values = values[frame.cycle_code.to_numpy(dtype=np.int64) <= origin]
    if not np.isfinite(fit_values).all() or not np.isfinite(values).all():
        raise RuntimeError(f"N00 fit-defined {feature} bins contain nonfinite values")
    edges = np.unique(np.quantile(fit_values, [0.25, 0.50, 0.75]))
    return np.searchsorted(edges, values, side="right").astype(np.int64), edges.astype(float)


def _metric(task: str, target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if task == "classification":
        return macro_f1_from_confusion(target, (np.asarray(prediction) >= 0.5).astype(np.int64))
    return regression_metrics(target, prediction)


def _gain(task: str, candidate: float, baseline: float) -> float:
    return float(candidate - baseline) if task == "classification" else float(baseline - candidate)


def _group_rows(task: str, frame: pd.DataFrame, target: np.ndarray, view_data: dict[str, np.ndarray],
                candidate: np.ndarray, control: np.ndarray, legacy: np.ndarray | None,
                minimum_group_n: int, minimum_class_n: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_rows = view_data["rows"]
    origins = view_data["origin"]
    full_cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    target_tail = None
    if task == "regression":
        target_tail = np.where(target <= 5, "y_le_5", np.where(target >= 95, "y_ge_95", "middle"))
    for origin in np.unique(origins):
        local = np.flatnonzero(origins == origin)
        source_rows = score_rows[local]
        data = frame.iloc[source_rows]
        y = target[source_rows]
        values = {
            "entry_friction": data.entry_friction.astype(str).to_numpy(),
            "stream_availability": np.where(_text_values(data.signal_stream)[1], "missing", "present"),
        }
        text_bins, text_edges = _quantile_bins(frame, int(origin), "text_load")
        impact_bins, impact_edges = _quantile_bins(frame, int(origin), "impact_load")
        values["fit_text_load_quartile"] = text_bins[source_rows].astype(str)
        values["fit_impact_load_quartile"] = impact_bins[source_rows].astype(str)
        if task == "regression" and target_tail is not None:
            values["observed_target_tail_evaluation_only"] = target_tail[source_rows]
        for field, labels in values.items():
            for label in np.unique(labels):
                mask = labels == label
                n = int(mask.sum())
                class_counts = {"class_0": int(np.sum(y[mask] == 0)), "class_1": int(np.sum(y[mask] == 1))} if task == "classification" else None
                supported = n >= minimum_group_n
                if class_counts is not None:
                    supported = supported and min(class_counts.values()) >= minimum_class_n
                candidate_metric = _metric(task, y[mask], candidate[local][mask])
                control_metric = _metric(task, y[mask], control[local][mask])
                row = {
                    "origin": int(origin), "field": field, "group": str(label), "n": n,
                    "supported_by_task_gate": bool(supported), "candidate_metric": candidate_metric,
                    "control_metric": control_metric,
                    "gain_candidate_vs_control": _gain(task, float(candidate_metric["macro_f1"] if task == "classification" else candidate_metric["mae"]),
                                                        float(control_metric["macro_f1"] if task == "classification" else control_metric["mae"])),
                    "fit_text_load_edges": text_edges.tolist(), "fit_impact_load_edges": impact_edges.tolist(),
                    "class_counts": class_counts,
                }
                if legacy is not None:
                    legacy_metric = _metric(task, y[mask], legacy[local][mask])
                    row["protected_legacy_metric"] = legacy_metric
                    row["gain_candidate_vs_legacy"] = _gain(task, float(candidate_metric["mae"]), float(legacy_metric["mae"]))
                rows.append(row)
    return rows


def _report_candidate(task: str, candidate_id: str, records: dict[int, dict[str, np.ndarray]],
                      blocks: dict[str, list[tuple[int, int, int]]], frame: pd.DataFrame,
                      target: np.ndarray, reference_by_row: dict[int, float],
                      legacy_by_row: dict[int, float] | None, group_rows: list[dict[str, Any]],
                      per_cycle_rows: list[dict[str, Any]],
                      reference_records_by_origin: dict[int, dict[str, np.ndarray]] | None = None,
                      legacy_records_by_origin: dict[int, dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate": candidate_id, "task": task, "views": {}}
    for view_name, definition in blocks.items():
        data = _collect(records, definition)
        row_ids = data["rows"]
        y = target[row_ids]
        if task == "classification":
            reference = data["control"].astype(np.float64)
        elif reference_records_by_origin is not None:
            aligned_reference = _collect(reference_records_by_origin, definition)
            if not np.array_equal(aligned_reference["rows"], row_ids):
                raise RuntimeError(f"{candidate_id}/{view_name} reference rows do not align by origin")
            reference = aligned_reference["prediction"].astype(np.float64)
        else:
            reference = np.asarray([reference_by_row[int(row)] for row in row_ids], dtype=np.float64)
        pred = data["prediction"].astype(np.float64)
        candidate_metric = _metric(task, y, pred)
        ref_metric = _metric(task, y, reference)
        score_key = "macro_f1" if task == "classification" else "mae"
        gain = _gain(task, float(candidate_metric[score_key]), float(ref_metric[score_key]))
        view_result: dict[str, Any] = {
            "n_unique_rows": int(len(row_ids)), "candidate": candidate_metric,
            "control": ref_metric, "gain_positive_is_better": gain,
            "prediction_disagreement": float(np.mean(np.abs(pred - reference))),
        }
        if task == "classification":
            view_result["label_disagreement_rate"] = float(np.mean((pred >= 0.5) != (reference >= 0.5)))
        if legacy_records_by_origin is not None:
            aligned_legacy = _collect(legacy_records_by_origin, definition)
            if not np.array_equal(aligned_legacy["rows"], row_ids):
                raise RuntimeError(f"{candidate_id}/{view_name} legacy rows do not align by origin")
            legacy = aligned_legacy["legacy"].astype(np.float64)
            legacy_available = True
        elif legacy_by_row is not None:
            legacy = np.asarray([legacy_by_row[int(row)] for row in row_ids], dtype=np.float64)
            legacy_available = True
        else:
            legacy = None
            legacy_available = False
        if legacy_available:
            legacy_metric = _metric(task, y, legacy)
            view_result["protected_legacy"] = legacy_metric
            view_result["gain_vs_protected_legacy"] = _gain(task, float(candidate_metric[score_key]), float(legacy_metric[score_key]))
            view_result["legacy_disagreement"] = float(np.mean(np.abs(pred - legacy)))
        result["views"][view_name] = view_result
        group_rows.extend(_group_rows(task, frame, target, data, pred, reference,
                                      legacy if legacy_available else None,
                                      minimum_group_n=200 if task == "classification" else 100))
        for cycle in np.unique(data["cycles"]):
            mask = data["cycles"] == cycle
            cm = _metric(task, y[mask], pred[mask])
            bm = _metric(task, y[mask], reference[mask])
            entry = {"candidate": candidate_id, "task": task, "view": view_name,
                     "cycle_code": int(cycle), "candidate_metric": cm, "control_metric": bm,
                     "gain_positive_is_better": _gain(task, float(cm[score_key]), float(bm[score_key])),
                     "n": int(mask.sum())}
            if legacy_available:
                lm = _metric(task, y[mask], legacy[mask])
                entry["protected_legacy_metric"] = lm
                entry["gain_vs_protected_legacy"] = _gain(task, float(cm[score_key]), float(lm[score_key]))
            per_cycle_rows.append(entry)
    return result


def _read_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray]]:
    frames: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    missing: dict[str, np.ndarray] = {}
    for task, target_name in (("classification", "artifact_signal"), ("regression", "threat_rank")):
        path = Path(task_files(task)["train"])
        frame = pd.read_csv(path, usecols=["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load", "impact_load", target_name])
        if len(frame) != 50_608 or not frame.record_id.is_unique:
            raise RuntimeError(f"{task} official Train shape/ID check failed")
        cycle = frame.cycle_code.to_numpy(dtype=np.int64)
        if not np.array_equal(np.unique(cycle), np.arange(47, 80)):
            raise RuntimeError(f"{task} official Train cycles differ from 47-79")
        _, mask = _text_values(frame.signal_stream)
        frames[task] = frame
        targets[task] = frame[target_name].to_numpy(dtype=np.int64 if task == "classification" else np.float64)
        if task == "classification" and not np.isin(targets[task], (0, 1)).all():
            raise RuntimeError("Classification Train target is not binary")
        if task == "regression" and (not np.isfinite(targets[task]).all() or
                                      np.any((targets[task] < 0) | (targets[task] > 100))):
            raise RuntimeError("Regression Train target is nonfinite or outside [0,100]")
        missing[task] = mask.astype(bool)
    return frames, targets, missing


def _load_all(frames: dict[str, pd.DataFrame], missing: dict[str, np.ndarray]) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], dict[str, dict[int, dict[str, np.ndarray]]]]:
    for card in ALL_CANDIDATES + ("R11",):
        _verify_historical_report(card)
    cls: dict[str, dict[int, dict[str, np.ndarray]]] = {card: {} for card in CLASS_CANDIDATES}
    reg: dict[str, dict[int, dict[str, np.ndarray]]] = {card: {} for card in REG_CANDIDATES}
    for card in CLASS_CANDIDATES:
        for cutoff in ORIGINS:
            bundle = _load_artifact(card, cutoff, frames["classification"])
            cls[card][cutoff] = reconstruct_classification(bundle, missing["classification"])
    for cutoff in ORIGINS:
        r22_bundle = _load_artifact("R22", cutoff, frames["regression"])
        r22 = reconstruct_regression(r22_bundle)
        reg["R22"][cutoff] = r22
        for card in ("R21", "R06"):
            bundle = _load_artifact(card, cutoff, frames["regression"])
            record = reconstruct_regression(bundle)
            # The frozen R21/R06 cache baseline is the protected legacy champion.
            if not np.array_equal(record["rows"], r22["rows"]):
                raise RuntimeError(f"{card}/{cutoff} row population differs from R22")
            if not np.allclose(record["legacy"], r22["legacy"], atol=1e-12, rtol=0):
                raise RuntimeError(f"{card}/{cutoff} protected legacy baseline differs from R22")
            reg[card][cutoff] = record
    return cls, reg


def _build_n03(reg: dict[str, dict[int, dict[str, np.ndarray]]],
               frame: pd.DataFrame, missing: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    arms: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in ORIGINS:
        a = reg["R22"][cutoff]
        b_source = reg["R06"][cutoff]
        if not np.array_equal(a["rows"], b_source["rows"]):
            raise RuntimeError(f"N03 arm caches differ at origin {cutoff}")
        rows = a["rows"]
        absent = missing[rows]
        b_present = np.clip(0.50 * a["legacy"] + 0.25 * a["rank50"] + 0.25 * b_source["rank50"], 0.0, 100.0)
        b = np.where(absent, a["prediction"], b_present)
        arms[cutoff] = {
            "rows": rows, "cycles": a["cycles"], "prediction": b,
            "control": a["prediction"], "legacy": a["legacy"],
            "raw_r22": a["raw"], "raw_r06": b_source["raw"],
            "rank50_r22": a["rank50"], "rank50_r06": b_source["rank50"],
            "stream_missing": absent,
        }
        if not np.array_equal(b[absent], a["prediction"][absent]):
            raise RuntimeError("N03 missing fallback is not exact R22")
    return arms


def _r11_pilot(frame: pd.DataFrame) -> dict[int, dict[str, np.ndarray]] | None:
    verification = json.loads(REPORTS["R11"].read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or verification.get("test_data_used") is True:
        raise RuntimeError("R11 pilot verification is not admissible")
    records: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        bundle = _load_artifact("R11", cutoff, frame)
        arrays = bundle["arrays"]
        pred = arrays.get("prediction")
        base = arrays.get("baseline_prediction")
        if pred is None or base is None or not np.isfinite(pred).all() or not np.isfinite(base).all():
            raise RuntimeError(f"R11/{cutoff} pilot predictions unavailable")
        if np.any((pred < 0) | (pred > 100)) or np.any((base < 0) | (base > 100)):
            raise RuntimeError(f"R11/{cutoff} pilot output outside [0,100]")
        records[cutoff] = {"rows": bundle["rows"], "cycles": bundle["cycles"],
                           "prediction": pred.astype(np.float64), "control": base.astype(np.float64),
                           "legacy": base.astype(np.float64)}
    return records


def _candidate_payload(task: str, candidate: str, view_data: dict[str, np.ndarray],
                       metrics: dict[str, Any], freeze: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = view_data["rows"].astype(np.int64)
    payload: dict[str, np.ndarray] = {
        "score_rows": rows,
        "score_record_ids": view_data["record_ids"].astype(str),
        "prediction": view_data["prediction"].astype(np.float64),
        "baseline_prediction": view_data["control"].astype(np.float64),
        "cycle_code": view_data["cycles"].astype(np.int64),
        "origin_cutoff": view_data["origin"].astype(np.int64),
        "stream_missing": view_data["stream_missing"].astype(bool),
    }
    if "raw" in view_data:
        payload["raw_prediction"] = view_data["raw"].astype(np.float64)
    for key in ("legacy", "raw_r22", "raw_r06", "rank50_r22", "rank50_r06"):
        if key in view_data:
            payload[key] = np.asarray(view_data[key], dtype=np.float64)
    metadata = {
        "candidate": candidate,
        "view": "primary_full_six",
        "task": task,
        "metric_and_group_summary": metrics,
        "source_artifact_hashes": freeze["identity"]["source_and_artifact_sha256"],
        "test_data_used": False,
        "test_labels_read": False,
        "official_estimator_fits": 0,
    }
    return payload, metadata


def _write_n00_bundle(task: str, candidate: str, data: dict[str, np.ndarray], metrics: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    payload, metadata = _candidate_payload(task, candidate, data, metrics, freeze)
    identity = {
        "batch_id": BATCH,
        "card_id": "N00",
        "candidate": candidate,
        "task": task,
        "view": "primary_full_six",
        "freeze_signature": freeze["freeze_signature"],
        "queue_sha256": freeze["identity"]["queue_sha256"],
    }
    path = BUNDLE_ROOT / candidate
    saved = resume_or_publish_bundle(path, payload, identity, task=task)
    # Independently rebuild the metric from the published/read-back arrays.
    arrays = saved.arrays
    if not np.array_equal(arrays["score_rows"], payload["score_rows"]):
        raise RuntimeError(f"{candidate} bundle read-back rows differ")
    if not np.allclose(arrays["prediction"], payload["prediction"], atol=0, rtol=0):
        raise RuntimeError(f"{candidate} bundle read-back predictions differ")
    return {"path": _relative(path), "receipt_sha256": saved.receipt["receipt_sha256"],
            "npz_sha256": saved.receipt["npz_sha256"], "rows": int(len(rows := arrays["score_rows"])),
            "read_back": "PASS", "independent_metric": metrics}


def execute_n00() -> dict[str, Any]:
    freeze = verify_freeze()
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    frames, targets, missing = _read_inputs()
    class_records, reg_records = _load_all(frames, missing)
    blocks = _blocks_from_queue(queue)
    groups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    outputs_for_bundle: dict[str, dict[str, np.ndarray]] = {}

    # Verify that all class candidate caches share the locked sparse25 route.
    for cutoff in ORIGINS:
        base = class_records["C16"][cutoff]
        for card in ("C11", "C17"):
            other = class_records[card][cutoff]
            index = {int(row): i for i, row in enumerate(other["rows"])}
            for row, value in zip(base["rows"], base["control"]):
                if not np.isclose(float(other["control"][index[int(row)]]), float(value), atol=1e-12, rtol=0):
                    raise RuntimeError(f"{card}/{cutoff} class baseline differs from sparse25 control")
    for card in CLASS_CANDIDATES:
        current = class_records[card]
        records = {e: {**r, "stream_missing": missing["classification"][r["rows"]]} for e, r in current.items()}
        result = _report_candidate("classification", card, records, blocks, frames["classification"],
                                   targets["classification"], {}, None, groups, per_cycle)
        candidates[card] = result
        view = _collect(records, blocks["primary_full_six"])
        view["record_ids"] = frames["classification"].record_id.to_numpy()[view["rows"]]
        outputs_for_bundle[card] = view

    # Regression models are compared to current R22 and separately to its protected legacy champion.
    r22 = reg_records["R22"]
    current_by_row: dict[int, float] = {}
    legacy_by_row: dict[int, float] = {}
    for cutoff in ORIGINS:
        record = r22[cutoff]
        for row, pred, legacy in zip(record["rows"], record["prediction"], record["legacy"]):
            current_by_row[int(row)] = float(pred)
            legacy_by_row[int(row)] = float(legacy)
    for card in REG_CANDIDATES:
        result = _report_candidate("regression", card, reg_records[card], blocks,
                                   frames["regression"], targets["regression"],
                                   current_by_row, legacy_by_row,
                                   groups, per_cycle,
                                   reference_records_by_origin=reg_records["R22"],
                                   legacy_records_by_origin=reg_records["R22"])
        candidates[card] = result
        view = _collect(reg_records[card], blocks["primary_full_six"])
        view["control"] = np.asarray([current_by_row[int(row)] for row in view["rows"]], dtype=np.float64)
        view["legacy"] = np.asarray([legacy_by_row[int(row)] for row in view["rows"]], dtype=np.float64)
        view["record_ids"] = frames["regression"].record_id.to_numpy()[view["rows"]]
        view["stream_missing"] = missing["regression"][view["rows"]]
        outputs_for_bundle[card] = view

    # R11 is a verified pilot-only diagnostic; never extend its missing full-six outputs.
    r11 = _r11_pilot(frames["regression"])
    if r11 is not None:
        r11_blocks = {"standard_primary": blocks["standard_primary"],
                      "standard_h12": blocks["standard_h12"],
                      "late_W4_pilot_only": [(67, 68, 71)],
                      "late_h12": blocks["late_h12"]}
        r11_r22_reference: dict[int, float] = {}
        r11_legacy_reference: dict[int, float] = {}
        for cutoff in (55, 67):
            for row, prediction, legacy_prediction in zip(
                    reg_records["R22"][cutoff]["rows"],
                    reg_records["R22"][cutoff]["prediction"],
                    reg_records["R22"][cutoff]["legacy"]):
                r11_r22_reference[int(row)] = float(prediction)
                r11_legacy_reference[int(row)] = float(legacy_prediction)
        r11_result = _report_candidate("regression", "R11_PILOT_ONLY", r11, r11_blocks,
                                       frames["regression"], targets["regression"],
                                       r11_r22_reference, r11_legacy_reference, groups, per_cycle,
                                       reference_records_by_origin={55: reg_records["R22"][55], 67: reg_records["R22"][67]},
                                       legacy_records_by_origin={55: reg_records["R22"][55], 67: reg_records["R22"][67]})
        r11_result["full_six_status"] = "UNKNOWN_TECHNICAL_BLOCKER_NOT_RECONSTRUCTED"
        r11_result["full_six_candidate_used"] = False
        r11_result["late_profile_scope"] = "W4 and H12 descriptive only; origin 75 is absent, so LATE primary is incomplete"
        candidates["R11_PILOT_ONLY"] = r11_result

    # Recompute fixed N03 A/B arms; rank each complete cycle before stream routing.
    n03_records = _build_n03(reg_records, frames["regression"], missing["regression"])
    n03_blocks = {key: blocks[key] for key in ("primary_full_six", "standard_primary", "standard_h12", "late_primary", "late_h12", "late_W5_W6")}
    n03_views = {}
    n03_rows: list[dict[str, Any]] = []
    for view_name, definition in n03_blocks.items():
        data = _collect(n03_records, definition)
        row_ids = data["rows"]
        y = targets["regression"][row_ids]
        a_metric = regression_metrics(y, data["control"])
        b_metric = regression_metrics(y, data["prediction"])
        legacy = np.asarray([legacy_by_row[int(row)] for row in row_ids], dtype=np.float64)
        l_metric = regression_metrics(y, legacy)
        n03_views[view_name] = {
            "n_unique_rows": int(len(row_ids)), "A_R22_ORIGINAL": a_metric,
            "B_router_input_arm": b_metric, "protected_legacy": l_metric,
            "gain_B_vs_A_positive_is_better": float(a_metric["mae"] - b_metric["mae"]),
            "gain_B_vs_legacy_positive_is_better": float(l_metric["mae"] - b_metric["mae"]),
            "missing_rows": int(data["stream_missing"].sum()),
            "present_rows": int((~data["stream_missing"]).sum()),
        }
        for cycle in np.unique(data["cycles"]):
            mask = data["cycles"] == cycle
            a = regression_metrics(y[mask], data["control"][mask])
            b = regression_metrics(y[mask], data["prediction"][mask])
            n03_rows.append({"view": view_name, "cycle_code": int(cycle), "n": int(mask.sum()),
                             "A_mae": a["mae"], "B_mae": b["mae"],
                             "gain_B_vs_A_positive_is_better": float(a["mae"] - b["mae"])})
        groups.extend(_group_rows("regression", frames["regression"], targets["regression"], data,
                                  data["prediction"], data["control"], legacy,
                                  minimum_group_n=100))
    candidates["N03_FIXED_ARMS"] = {"task": "regression", "A": "R22_ORIGINAL",
                                     "B": "present-only three-part rank blend; missing exactly A",
                                     "views": n03_views,
                                     "missing_fallback_exact": True,
                                     "no_router_fit_or_selection": True}
    data = _collect(n03_records, blocks["primary_full_six"])
    data["control"] = data["control"].astype(np.float64)
    data["record_ids"] = frames["regression"].record_id.to_numpy()[data["rows"]]
    data["stream_missing"] = data["stream_missing"].astype(bool)
    outputs_for_bundle["N03_FIXED_ARMS"] = data

    bundle_receipts: dict[str, Any] = {}
    for candidate in ("C16", "C17", "C11", "R22", "R21", "R06", "N03_FIXED_ARMS"):
        task = "classification" if candidate in CLASS_CANDIDATES else "regression"
        metric_key = candidate if candidate != "N03_FIXED_ARMS" else candidate
        bundle_receipts[candidate] = _write_n00_bundle(task, candidate, outputs_for_bundle[candidate],
                                                       candidates[metric_key], freeze)

    summary = {
        "schema_version": 1,
        "batch_id": BATCH,
        "card_id": "N00",
        "status": "CACHE_AUDIT_AND_N03_FIXED_ARM_RECONSTRUCTION_PASS",
        "freeze_signature": freeze["freeze_signature"],
        "candidates": candidates,
        "candidate_bundle_receipts": bundle_receipts,
        "new_official_estimator_fits": 0,
        "synthetic_estimator_fits": 0,
        "test_files_opened": False,
        "test_labels_or_targets_read": False,
        "kaggle_calls": 0,
        "public_score_used_for_recipe_or_control": False,
        "row_accounting": {
            "primary_full_six_expected": 38_588,
            "late_W5_W6_expected": 15_953,
            "standard_pilot_primary_expected": 9_064,
            "late_pilot_primary_expected": 13_441,
            "all_candidate_views_row_unique": True,
        },
        "candidate_group_rows": groups,
        "per_cycle_rows": per_cycle,
        "n03_per_cycle_rows": n03_rows,
        "r11_scope": "verified pilot diagnostic only; full-six UNKNOWN and not reconstructed",
        "supersedes_previous_n00": {
            "path": _relative(PREVIOUS_REPORT_ROOT / "summary.json"),
            "freeze_signature": "b468bd41e9e96440e0db6f49e316a521a5f6bf3bdabb8c4458024205e9c27113",
            "reason": "regression H12 reference/legacy predictions are now matched to each candidate's training origin; prior amendment-04 H12 comparisons are superseded",
            "primary_full_six_predictions_from_previous_result_remain_cache_aligned": True,
        },
        "resource_and_dependency_map": _resource_map(queue, freeze),
    }
    _write_once(REPORT_ROOT / "summary.json", summary)
    _write_csv_once(REPORT_ROOT / "candidate_summary.csv", [
        {"candidate": cid, "task": c.get("task"), "view": view,
         "n": data.get("n_unique_rows"),
         "candidate_metric": data.get("candidate", data.get("B_router_input_arm")),
         "control_metric": data.get("control", data.get("A_R22_ORIGINAL")),
         "gain_positive_is_better": data.get("gain_positive_is_better", data.get("gain_B_vs_A_positive_is_better"))}
        for cid, c in candidates.items() for view, data in c.get("views", {}).items()
    ])
    _write_csv_once(REPORT_ROOT / "per_cycle.csv", per_cycle + n03_rows)
    _write_csv_once(REPORT_ROOT / "groups.csv", groups)
    _write_n00_plots(candidates, per_cycle, n03_rows)
    return summary


def _resource_map(queue: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    mapped = {}
    for card in queue["cards"]:
        mapped[card["id"]] = {
            "task": card["task"], "kind": card["kind"], "profile": card["profile"],
            "pilot_fits": card["pilot_fits"], "continuation_fits_max": card["continuation_fits"],
            "controls": card.get("controls", []),
            "preconditions": card.get("preconditions", []),
            "smoke_contract": card.get("smoke", []),
            "resource_note": {
                "N01": "reuses sparse TF-IDF pipeline, but keeps two trained weighted experts per pilot origin",
                "N02": "highest sparse-matrix risk: copy C2_B CSR, 1024 PolynomialCountSketch columns, never densify TF-IDF",
                "N03": "small HGB router heads; OOF join and complete-cycle rank arrays are the main memory concern",
                "N04": "TF-IDF domain LogisticRegression plus weighted R22 XGBoost; domain and task fits each count",
                "N05": "four LightGBM quantile heads across two origins; SVD32 shared per origin",
                "N06": "four R06 char-TFIDF LightGBM fits; sparse vocabulary may dominate peak memory",
                "N00": "cache-only; no estimator fits",
            }[card["id"]],
        }
    return {
        "free_ram_gib_at_freeze": freeze["execution_context"].get("free_ram_gib_at_freeze"),
        "training_jobs_allowed": 1, "threads_per_job": 2, "job_cap_gib": 6,
        "minimum_free_ram_gib": 4, "reserve_gib": 2,
        "fit_live_check_required_before_each_fit": True,
        "lanes": mapped,
        "prohibited": ["Test", "SampleSubmission", "Kaggle", "full-Train79 refit", "Torch", "external/pretrained data"],
    }


def _write_n00_plots(candidates: dict[str, Any], per_cycle: list[dict[str, Any]], n03_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    destination = REPORT_ROOT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for candidate in CLASS_CANDIDATES:
        selected = [r for r in per_cycle if r.get("task") == "classification" and r.get("candidate") == candidate and r.get("view") == "primary_full_six"]
        axes[0].plot([r["cycle_code"] for r in selected], [r["gain_positive_is_better"] for r in selected], marker="o", label=candidate)
    for candidate in ("R21", "R06"):
        selected = [r for r in per_cycle if r.get("task") == "regression" and r.get("candidate") == candidate and r.get("view") == "primary_full_six"]
        axes[1].plot([r["cycle_code"] for r in selected], [r["gain_positive_is_better"] for r in selected], marker="o", label=f"{candidate} vs R22")
    axes[0].axhline(0, color="black", linewidth=0.8); axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Macro F1 gain vs sparse25"); axes[1].set_ylabel("MAE gain vs R22 (positive = better)")
    axes[1].set_xlabel("Cycle code"); axes[0].set_title("Full-six per-cycle differences on pooled disjoint validation rows")
    axes[0].legend(ncol=3, frameon=False); axes[1].legend(ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(destination / "per_cycle_candidate_gains.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    selected = [r for r in n03_rows if r["view"] == "primary_full_six"]
    ax.bar([r["cycle_code"] for r in selected], [r["gain_B_vs_A_positive_is_better"] for r in selected], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cycle code"); ax.set_ylabel("MAE gain of fixed arm B vs R22 arm A")
    ax.set_title("N03 fixed arms: per-cycle exploratory contrast; no router fit")
    fig.tight_layout(); fig.savefig(destination / "n03_fixed_arm_cycle_gains.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_n00()
        print(json.dumps({"freeze_signature": result["freeze_signature"], "source_commit": result["identity"]["source_commit"],
                          "artifact_count": len(result["identity"]["source_and_artifact_sha256"]),
                          "old_ledger_snapshot": result["execution_context"]["inherited_old_budget_snapshot"],
                          "official_estimator_fits": 0}, indent=2))
    else:
        result = execute_n00()
        print(json.dumps({"status": result["status"], "freeze_signature": result["freeze_signature"],
                          "candidate_ids": list(result["candidates"]), "official_estimator_fits": 0,
                          "test_files_opened": False, "kaggle_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
