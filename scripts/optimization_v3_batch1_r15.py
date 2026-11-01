"""R15 preflight, frozen missing-stream pilot, independent verification and gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r01 as r01
from scripts import optimization_v3_batch1_r18 as r18
from scripts import optimization_v3_reg_diagnostics as reg_diagnostics
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    metric,
    read_bundle,
    run_pilot,
    stream_missing_mask,
    temporal_rows,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r15 import (
    CARD_ID,
    EXPERT_WEIGHT,
    FEATURE_NAMES,
    PILOT_CUTOFFS,
    STRUCTURED_FEATURE_NAMES,
    fit_predict,
    reconstruct_r15_arrays,
)
from src.models.regression_research_tabular import RegressionResearchTabular
from src.models.round1_models import EXPANDED_FEATURES
from src.utils.paths import task_files


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R15.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R15"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R15"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R15"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
EVENT_PATH = RUN_DIR / "attempts.jsonl"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUBGROUP_PATH = REPORT_DIR / "subgroups.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
RUNTIME_NOTES_PATH = REPORT_DIR / "runtime_notes.json"

QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "d36d7e6593966a081773e262301df7597b2f15d10578bf138de23292280d9bf4"
TRAIN_SHA256 = r01.EXPECTED_TRAIN_SHA256
MANIFEST_SHA256 = r01.EXPECTED_MANIFEST_SHA256
EXPECTED_VIEWS = {
    "W4": [(67, 68, 71)],
    "W6": [(75, 76, 79)],
    "primary": [(67, 68, 71), (75, 76, 79)],
    "H12_L": [(67, 68, 79)],
}
EXPECTED_VIEW_ROWS = {"W4": 4_950, "W6": 8_491, "primary": 13_441,
                      "H12_L": 20_903, "H12_E": 17_685}
MIN_FIT_MISSING_ROWS = 1_000
MIN_VIEW_MISSING_ROWS = 100
MIN_GROUP_SUPPORT = 100
MAX_SUCCESSFUL_FITS = 180
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
STRUCTURED_WIDTH_UPPER = 807
MODEL_RUNTIME_ALLOWANCE_BYTES = int(1.5 * 1024**3)
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    if path.is_file():
        if _read_json(path) != payload:
            raise BatchIntegrityError(f"Existing R15 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _source_commit_context() -> str:
    paths = [
        "scripts/optimization_v3_batch1_r15.py",
        "src/models/optimization_v3_batch1_r15.py",
        "tests/test_optimization_v3_batch1_r15.py",
    ]
    result = subprocess.run(["git", "log", "-1", "--format=%H", "--", *paths],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise BatchIntegrityError("Cannot identify a committed R15 source revision")
    return result.stdout.strip()


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("Canonical R15 queue/card hash changed")
    canonical = next((row for row in queue.get("cards", []) if row.get("id") == CARD_ID), None)
    if canonical is None or dict(canonical) != dict(card):
        raise BatchIntegrityError("R15 dispatcher card differs from canonical queue")
    fixed = {
        "id": "R15", "task": "regression", "family": "specialist_missing",
        "policy": "R_MISSING10_RAW", "pilot_predictor_fits": 2,
        "evaluation": "LATE", "condition": "No additional dependency",
    }
    for name, expected in fixed.items():
        if card.get(name) != expected:
            raise BatchIntegrityError(f"R15 canonical queue field changed: {name}")
    if not all(text in str(card.get("change", "")) for text in (
        "Later origins67/75 only", "Structured-only LGBL1", "10% raw clipped expert+90%incumbent",
        "No subgroup percentile transformation",
    )):
        raise BatchIntegrityError("R15 queue recipe changed from its frozen missing-only policy")
    preset = queue.get("execution_preset", {})
    if (int(preset.get("concurrent_fit_jobs", -1)) != 1
            or int(preset.get("threads_per_job", -1)) != THREADS
            or float(preset.get("max_job_RSS_GiB", -1)) != 6.0
            or float(preset.get("minimum_free_RAM_GiB_before_start", -1)) != 4.0
            or int(preset.get("max_successful_new_predictor_fits", -1)) != MAX_SUCCESSFUL_FITS):
        raise BatchIntegrityError("R15 execution/resource/budget preset changed")
    gate = queue.get("evaluation", {}).get("regression_gate", {})
    expected_gate = {
        "pilot_primary_MAE_gain_min": 0.03,
        "pilot_H12_MAE_harm_max": 0.05,
        "individual_pilot_view_MAE_harm_max": 0.5,
        "supported_group_n_min": 100,
        "supported_tail_or_missing_MAE_harm_max": 0.25,
    }
    if any(float(gate.get(name, np.nan)) != value for name, value in expected_gate.items()):
        raise BatchIntegrityError("R15 regression pilot gates differ from canonical queue")
    if tuple(STRUCTURED_FEATURE_NAMES) != tuple(EXPANDED_FEATURES) or "signal_stream" in STRUCTURED_FEATURE_NAMES:
        raise BatchIntegrityError("R15 structured expert feature allowlist changed")


def _input_files(train_path: Path) -> dict[str, Path]:
    inputs = r18._input_files(train_path)
    inputs["canonical_card"] = CARD_PATH
    return inputs


def _source_files() -> list[Path]:
    paths = {
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r15.py",
        ROOT / "tests/test_optimization_v3_batch1_r15.py",
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_r01.py",
        ROOT / "src/models/optimization_v3_batch1_r01.py",
        ROOT / "scripts/optimization_v3_batch1_r18.py",
        ROOT / "src/models/optimization_v3_batch1_r18.py",
        ROOT / "scripts/optimization_v3_reg_diagnostics.py",
        ROOT / "src/models/regression_research_tabular.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/utils/paths.py",
    }
    return sorted(paths, key=str)


def _budget_preflight() -> dict[str, Any]:
    reserved: dict[str, int] = {}
    if LEDGER_PATH.is_file():
        for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid R15 fit ledger line {line_number}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key = str(event["budget_key"])
                count = int(event["predictor_fits"])
                if key in reserved and reserved[key] != count:
                    raise BatchIntegrityError(f"Conflicting fit reservation for {key}")
                reserved[key] = count
    current = reserved.get("R15:PILOT")
    total_before = sum(reserved.values())
    remaining = MAX_SUCCESSFUL_FITS - total_before
    enough = current == 2 or (current is None and remaining >= 2)
    return {
        "status": "PASS" if enough else "SKIP_BUDGET",
        "successful_fit_cap": MAX_SUCCESSFUL_FITS,
        "already_reserved": total_before,
        "remaining_before_R15": remaining,
        "R15_reserved_fits": current if current is not None else 2,
        "ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
    }


def _resource_estimate(fit_missing_rows: int, score_missing_rows: int,
                       available_bytes: int | None = None) -> dict[str, Any]:
    fit_design_bytes = int(fit_missing_rows * STRUCTURED_WIDTH_UPPER * 8)
    score_design_bytes = int(score_missing_rows * STRUCTURED_WIDTH_UPPER * 8)
    histogram_bytes = int(STRUCTURED_WIDTH_UPPER * 256 * 16 * THREADS * 3)
    estimate = int(
        MODEL_RUNTIME_ALLOWANCE_BYTES + 5 * fit_design_bytes
        + 3 * score_design_bytes + histogram_bytes
    )
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    return {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and estimate <= allowed else "BLOCKED_RESOURCE",
        "missing_fit_rows": int(fit_missing_rows),
        "missing_score_rows_upper_bound": int(score_missing_rows),
        "structured_width_upper_bound": STRUCTURED_WIDTH_UPPER,
        "fit_design_bytes_upper_bound": fit_design_bytes,
        "score_design_bytes_upper_bound": score_design_bytes,
        "histogram_bytes_upper_bound": histogram_bytes,
        "model_runtime_allowance_bytes": MODEL_RUNTIME_ALLOWANCE_BYTES,
        "estimated_additional_peak_bytes": estimate,
        "estimate_is_observed_rss": False,
        "available_ram_bytes": available,
        "minimum_free_ram_bytes": MINIMUM_FREE_BYTES,
        "reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": int(max(0, allowed)),
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "structured_only_no_text_matrix": True,
    }


def _late_support(frame: pd.DataFrame, cycles: np.ndarray,
                  views: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, np.ndarray]]:
    missing = stream_missing_mask(frame["signal_stream"].to_numpy(dtype=object))
    support: dict[str, Any] = {"minimum_fit_missing_rows": MIN_FIT_MISSING_ROWS,
                               "minimum_validation_missing_rows_per_view": MIN_VIEW_MISSING_ROWS,
                               "views": {}, "cutoffs": {}}
    fit_rows_by_cutoff: dict[int, np.ndarray] = {}
    score_rows_by_cutoff: dict[int, np.ndarray] = {}
    fit_missing_counts: dict[int, int] = {}
    for cutoff in PILOT_CUTOFFS:
        fit_rows, score_rows = temporal_rows(frame, cutoff, horizon=12)
        fit_rows_by_cutoff[cutoff] = fit_rows
        score_rows_by_cutoff[cutoff] = score_rows
        fit_missing_count = int(missing[fit_rows].sum())
        fit_missing_counts[cutoff] = fit_missing_count
        support["cutoffs"][str(cutoff)] = {
            "fit_rows": int(len(fit_rows)), "fit_missing_rows": fit_missing_count,
            "fit_missing_support_pass": fit_missing_count >= MIN_FIT_MISSING_ROWS,
            "score_rows": int(len(score_rows)),
            "score_missing_rows": int(missing[score_rows].sum()),
        }
    view_counts: dict[str, int] = {}
    for name, segments in views.items():
        rows_parts: list[np.ndarray] = []
        missing_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            score_rows = score_rows_by_cutoff[int(cutoff)]
            mask = (cycles[score_rows] >= int(low)) & (cycles[score_rows] <= int(high))
            rows = score_rows[mask]
            rows_parts.append(rows)
            missing_parts.append(missing[rows])
        rows = np.concatenate(rows_parts)
        missing_rows = int(np.concatenate(missing_parts).sum())
        if (len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows)):
            raise BatchIntegrityError(f"R15 {name} row denominator/uniqueness differs")
        view_counts[name] = missing_rows
        support["views"][name] = {
            "rows": int(len(rows)), "missing_stream_rows": missing_rows,
            "missing_stream_fraction": float(missing_rows / len(rows)),
            "support_pass": missing_rows >= MIN_VIEW_MISSING_ROWS,
        }
    support["status"] = "PASS" if (
        all(item["fit_missing_support_pass"] for item in support["cutoffs"].values())
        and all(item["support_pass"] for item in support["views"].values())
    ) else "SKIPPED_SUPPORT"
    return support, fit_rows_by_cutoff, score_rows_by_cutoff


def _control_metrics(target: np.ndarray, cycles: np.ndarray, controls: Mapping[int, Mapping[str, Any]],
                     views: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, segments in views.items():
        rows_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            block = controls[int(cutoff)]
            rows = np.asarray(block["rows"], dtype=np.int64)
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            rows_parts.append(rows[mask])
            prediction_parts.append(np.asarray(block["baseline_prediction"], dtype=np.float64)[mask])
        rows = np.concatenate(rows_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R15 row-matched control denominator differs for {name}")
        result[name] = {
            "rows": int(len(rows)),
            "mae": float(metric("regression", target[rows], np.concatenate(prediction_parts))["mae"]),
        }
    early = controls[55]
    rows = np.asarray(early["rows"], dtype=np.int64)
    mask = (cycles[rows] >= 56) & (cycles[rows] <= 67)
    rows = rows[mask]
    pred = np.asarray(early["baseline_prediction"], dtype=np.float64)[mask]
    if len(rows) != EXPECTED_VIEW_ROWS["H12_E"] or len(np.unique(rows)) != len(rows):
        raise BatchIntegrityError("R15 H12_E exact-champion fallback denominator differs")
    result["H12_E"] = {
        "rows": int(len(rows)), "mae": float(metric("regression", target[rows], pred)["mae"]),
        "policy": "exact champion; unchanged; outside the LATE pilot gate",
    }
    return result


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any],
                    support: Mapping[str, Any], controls: Mapping[str, Any],
                    resources: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    model = RegressionResearchTabular(params_id="base", features=STRUCTURED_FEATURE_NAMES)
    return {
        "card_id": CARD_ID,
        "queue_recipe": {key: card[key] for key in (
            "id", "task", "name", "family", "priority", "change", "policy",
            "pilot_predictor_fits", "evaluation", "condition",
        )},
        "model": "RegressionResearchTabular LightGBM L1 base; fit only on missing/blank/whitespace signal_stream rows",
        "structured_feature_allowlist": list(STRUCTURED_FEATURE_NAMES),
        "signal_stream_used_as_model_feature": False,
        "record_id_or_target_used_as_model_feature": False,
        "params_id": "base",
        "actual_lightgbm_parameters": dict(model.actual_parameters),
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "preprocessing": "RegressionResearchTabular fit-only structured preprocessing on missing-stream fit rows at each origin",
        "missing_stream_rule": "null, blank, or whitespace-only signal_stream",
        "prediction_policy": {
            "missing_rows": "0.90 exact incumbent + 0.10 clip(raw expert, 0, 100); final clip [0,100]",
            "present_rows": "exact incumbent, unchanged",
            "percentile_or_rank_mapping": False,
        },
        "support": dict(support),
        "row_matched_control_metrics": dict(controls),
        "views": dict(views),
        "thresholds": dict(queue["evaluation"]["regression_gate"]),
        "resource_preflight": dict(resources),
        "fit_protocol": "one structured-only LightGBM regression_l1 predictor per cutoff 67/75; no validation labels or early stopping",
        "threads": THREADS,
        "concurrent_fit_jobs": 1,
    }


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    r01._validate_manifest(dict(queue))
    train_path, frame, target, cycles = r01._check_official_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train SHA256 differs from the R15 pin")
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R15 LATE views changed: {views}")

    input_files = _input_files(train_path)
    input_sha256, evidence = r18._validate_evidence_hashes(queue, input_files)
    base_cache = r18._load_archive(r18.PRIMARY_CACHE)
    pilot_cache = r18._load_archive(r18.PILOT_CACHE)
    full_cache = r18._load_archive(r18.FULL_CACHE)
    sparse_cache = {**pilot_cache, **full_cache}
    oof = r18._load_oof_predictions_only(r18.OOF_CACHE_PATH)
    history_targets = {
        int(row): float(target[row]) for row in np.flatnonzero(np.isin(cycles, list(r18.HISTORY_CYCLES)))
    }
    source_cutoff_by_cycle, mapping_audit = r18._validate_primary_oof_mapping(
        frame=frame, cycles=cycles, history_targets=history_targets,
        diagnostics=evidence["diagnostics"], residual_freeze=evidence["residual_freeze"],
        oof=oof, sparse_cache=sparse_cache, base_cache=base_cache,
    )
    del source_cutoff_by_cycle
    control_detail = {
        67: r18._reconstruct_control(
            frame=frame, cycles=cycles, cutoff=67, low=68, high=79,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof, include_arrays=True,
        ),
        75: r18._reconstruct_control(
            frame=frame, cycles=cycles, cutoff=75, low=76, high=79,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof, include_arrays=True,
        ),
        55: r18._reconstruct_control(
            frame=frame, cycles=cycles, cutoff=55, low=56, high=67,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof, include_arrays=True,
        ),
    }
    controls_by_cutoff = {cutoff: item["_arrays"] for cutoff, item in control_detail.items()}
    control_checks = {str(cutoff): {key: value for key, value in item.items() if key != "_arrays"}
                      for cutoff, item in control_detail.items()}

    support, fit_rows_by_cutoff, score_rows_by_cutoff = _late_support(frame, cycles, views)
    missing_all = stream_missing_mask(frame["signal_stream"].to_numpy(dtype=object))
    resources: dict[str, Any] = {}
    for cutoff in PILOT_CUTOFFS:
        fit_missing = int(missing_all[fit_rows_by_cutoff[cutoff]].sum())
        late_rows = score_rows_by_cutoff[cutoff]
        score_missing = int(missing_all[late_rows].sum())
        resources[str(cutoff)] = _resource_estimate(fit_missing, score_missing)
    resources_pass = all(item["status"] == "PASS" for item in resources.values())
    budget = _budget_preflight()

    def control_loader(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        if int(cutoff) not in controls_by_cutoff:
            raise BatchIntegrityError(f"R15 has no frozen exact-champion control for cutoff {cutoff}")
        block = controls_by_cutoff[int(cutoff)]
        rows = np.asarray(block["rows"], dtype=np.int64)
        baseline = np.asarray(block["baseline_prediction"], dtype=np.float64)
        if not np.array_equal(rows, score_rows):
            raise BatchIntegrityError(f"R15 control row positions do not match cutoff {cutoff}")
        return {
            "baseline_prediction": baseline.copy(),
            "stream_missing": missing_all[rows],
            "control_rows": rows,
        }

    control_metrics = _control_metrics(target, cycles, controls_by_cutoff, views)
    recipe = _recipe_payload(card, views, support, control_metrics, resources, queue)
    source_files = _source_files()
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=source_files, evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "recipe_artifact_sha256": json_sha256(recipe),
        "source_commit_context": _source_commit_context(),
        "support_preflight": support,
        "exact_champion_control_checks": control_checks,
        "control_oof_mapping_audit": mapping_audit,
        "input_sha256": input_sha256,
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "memory_preflight": resources,
        "budget_preflight": budget,
        "fit_population": "all official regression Train rows through cutoff; estimator fit subset is missing/blank streams only",
        "score_population": "cutoff67 all cycles68-79; cutoff75 cycles76-79",
        "validation_labels_passed_to_adapter": False,
        "official_test_used": False,
    })
    all_input_hashes_match = {name: file_sha256(path) for name, path in sorted(input_files.items())} == input_sha256
    # make_freeze_identity keys source hashes with str(Path(path)); compare
    # using that exact representation so the freeze is checked against the
    # files actually hashed rather than a second, incompatible path convention.
    all_source_hashes_match = {
        str(Path(path)): file_sha256(path) for path in source_files
    } == identity["source_sha256"]
    if not all_input_hashes_match or not all_source_hashes_match:
        raise BatchIntegrityError("R15 source/input changed during preflight")
    if support["status"] != "PASS":
        status = "SKIPPED_SUPPORT"
    elif not resources_pass or budget["status"] != "PASS":
        status = "BLOCKED_RESOURCE_OR_BUDGET"
    else:
        status = "PREFLIGHT_PASS"
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "official Regression Train, frozen exact-champion caches, no Test/Sample or Kaggle access",
        "official_train_sha256": TRAIN_SHA256,
        "input_sha256": input_sha256,
        "source_sha256": identity["source_sha256"],
        "freeze_identity_sha256": json_sha256(identity),
        "accepted_manifest_sha256": MANIFEST_SHA256,
        "exact_champion_control_checks": control_checks,
        "control_oof_mapping_audit": mapping_audit,
        "row_matched_control_metrics": control_metrics,
        "support_preflight": support,
        "resource_preflight": resources,
        "budget_preflight": budget,
        "fit_predictor_count": 0,
        "candidate_fit_or_metric_computed": False,
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    return {
        "public": public, "identity": identity, "recipe": recipe,
        "frame": frame, "target": target, "cycles": cycles, "views": views,
        "controls_by_cutoff": controls_by_cutoff,
        "fit_rows_by_cutoff": fit_rows_by_cutoff,
        "score_rows_by_cutoff": score_rows_by_cutoff,
        "control_loader": control_loader,
    }


def _expected_fit_population(context: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    fit_rows = np.asarray(context["fit_rows_by_cutoff"][int(cutoff)], dtype=np.int64)
    frame = context["frame"]
    missing_mask = stream_missing_mask(
        frame.iloc[fit_rows]["signal_stream"].to_numpy(dtype=object)
    )
    missing_positions = fit_rows[missing_mask]
    cycles = np.asarray(context["cycles"], dtype=np.int64)[missing_positions]
    if not len(missing_positions):
        raise BatchIntegrityError(f"R15 cutoff {cutoff} has no missing-stream fit rows")
    return {
        "all_fit_rows": int(len(fit_rows)),
        "missing_fit_rows": int(len(missing_positions)),
        "fit_cycle_min": int(cycles.min()),
        "fit_cycle_max": int(cycles.max()),
        "fit_source_row_positions_sha256": hashlib.sha256(
            np.asarray(missing_positions, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
        "fit_missing_mask_sha256": hashlib.sha256(missing_mask.tobytes(order="C")).hexdigest(),
    }


def _read_saved_predictions(context: Mapping[str, Any], report: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    per_cutoff: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in PILOT_CUTOFFS:
        rows = np.asarray(context["score_rows_by_cutoff"][cutoff], dtype=np.int64)
        identity = {"freeze_signature": report["freeze_signature"], "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"R15_fit{cutoff}.npz", identity)
        candidate, baseline = reconstruct_r15_arrays(arrays)
        control = context["controls_by_cutoff"][cutoff]
        expected_missing = np.asarray(control["stream_missing"], dtype=bool)
        expected_fit = _expected_fit_population(context, cutoff)
        state = sidecar["metadata"].get("state", {})
        fit_metadata = sidecar["metadata"].get("fit_metadata", {})
        checks = {
            "exact_score_rows": np.array_equal(np.asarray(arrays["score_rows"], dtype=np.int64), rows),
            "exact_record_ids": np.array_equal(
                np.asarray(arrays["score_record_ids"]).astype(str),
                context["frame"].iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            ),
            "exact_champion_control": np.array_equal(baseline, np.asarray(control["baseline_prediction"])),
            "missing_mask_matches_official_train": np.array_equal(
                np.asarray(arrays["control__stream_missing"], dtype=bool), expected_missing
            ),
            "audit_mask_matches_control": np.array_equal(
                np.asarray(arrays["audit__stream_missing"], dtype=bool), expected_missing
            ),
            "expert_is_only_scored_for_missing": np.array_equal(
                np.asarray(arrays["audit__expert_missing_rows"], dtype=bool), expected_missing
            ),
            "present_rows_exact_incumbent": np.array_equal(candidate[~expected_missing], baseline[~expected_missing]),
            "all_candidate_values_finite_in_range": bool(
                np.isfinite(candidate).all() and np.all((candidate >= 0.0) & (candidate <= 100.0))
            ),
            "fit_population_exact_missing_only": (
                int(state.get("fit_rows", -1)) == expected_fit["missing_fit_rows"]
                and state.get("fit_source_row_positions_sha256") == expected_fit["fit_source_row_positions_sha256"]
                and state.get("fit_missing_mask_sha256") == expected_fit["fit_missing_mask_sha256"]
                and int(sidecar["metadata"].get("fit_rows", -1)) == expected_fit["all_fit_rows"]
                and int(fit_metadata.get("fit_missing_stream_rows", -1)) == expected_fit["missing_fit_rows"]
            ),
            "fit_population_cycle_range_exact": (
                int(state.get("fit_cycle_min", -1)) == expected_fit["fit_cycle_min"]
                and int(state.get("fit_cycle_max", -1)) == expected_fit["fit_cycle_max"]
            ),
            "structured_only_feature_allowlist": state.get("feature_names") == list(STRUCTURED_FEATURE_NAMES),
            "signal_stream_excluded_from_model": state.get("signal_stream_used_as_feature") is False,
            "rank_mapping_not_used": state.get("percentile_or_rank_mapping") is False,
        }
        per_cutoff[cutoff] = {
            "rows": rows, "prediction": candidate, "baseline_prediction": baseline,
            "missing": expected_missing,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "sidecar": sidecar,
        }
    return per_cutoff


def _view_arrays(context: Mapping[str, Any], per_cutoff: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, segments in context["views"].items():
        rows_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        baseline_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            block = per_cutoff[int(cutoff)]
            rows = np.asarray(block["rows"], dtype=np.int64)
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            rows_parts.append(rows[mask])
            prediction_parts.append(np.asarray(block["prediction"], dtype=np.float64)[mask])
            baseline_parts.append(np.asarray(block["baseline_prediction"], dtype=np.float64)[mask])
        rows = np.concatenate(rows_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R15 {name} saved rows are duplicated or wrong-length")
        output[name] = {
            "rows": rows,
            "candidate": np.concatenate(prediction_parts),
            "baseline": np.concatenate(baseline_parts),
        }
    return output


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise BatchIntegrityError("R15 subgroup MAE received invalid or unaligned vectors")
    return float(np.abs(y - p).sum(dtype=np.float64) / len(y))


def _group_metrics(context: Mapping[str, Any], views: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    frame = context["frame"]
    target = np.asarray(context["target"], dtype=np.float64)
    missing_all = stream_missing_mask(frame["signal_stream"].to_numpy(dtype=object))
    friction_all = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
    result: dict[str, Any] = {"support_minimum": MIN_GROUP_SUPPORT, "guard_max_harm": 0.25,
                              "views": {}, "guard_failures": []}
    for name, item in views.items():
        rows = np.asarray(item["rows"], dtype=np.int64)
        y = target[rows]
        missing = missing_all[rows]
        friction = friction_all[rows]
        masks = {
            "signal_stream_missing": missing,
            "signal_stream_present": ~missing,
            "target_le_5": y <= 5.0,
            "target_5_to_95": (y > 5.0) & (y < 95.0),
            "target_ge_95": y >= 95.0,
            "entry_friction_0": np.isfinite(friction) & (friction == 0.0),
            "entry_friction_1": np.isfinite(friction) & (friction == 1.0),
        }
        result["views"][name] = {}
        for label, mask in masks.items():
            n = int(mask.sum())
            candidate_mae = _mae(y[mask], np.asarray(item["candidate"])[mask]) if n else None
            baseline_mae = _mae(y[mask], np.asarray(item["baseline"])[mask]) if n else None
            harm = float(candidate_mae - baseline_mae) if n else None
            guard_applies = bool(
                name in {"W4", "W6", "H12_L"} and n >= MIN_GROUP_SUPPORT
                and label in {"signal_stream_missing", "target_le_5", "target_ge_95"}
            )
            guard_pass = bool(harm <= 0.25) if guard_applies else None
            result["views"][name][label] = {
                "n": n, "supported": n >= MIN_GROUP_SUPPORT,
                "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                "candidate_minus_baseline_mae": harm,
                "guard_applies": guard_applies, "guard_pass": guard_pass,
            }
            if guard_applies and not guard_pass:
                result["guard_failures"].append({"view": name, "group": label, "n": n, "harm": harm})
    return result


def _early_fallback(context: Mapping[str, Any]) -> dict[str, Any]:
    block = context["controls_by_cutoff"][55]
    rows = np.asarray(block["rows"], dtype=np.int64)
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    mask = (cycles[rows] >= 56) & (cycles[rows] <= 67)
    rows = rows[mask]
    baseline = np.asarray(block["baseline_prediction"], dtype=np.float64)[mask]
    target = np.asarray(context["target"], dtype=np.float64)[rows]
    if len(rows) != EXPECTED_VIEW_ROWS["H12_E"]:
        raise BatchIntegrityError("R15 exact-champion H12_E fallback row count changed")
    return {
        "rows": int(len(rows)), "baseline_mae": _mae(target, baseline),
        "candidate_mae": _mae(target, baseline), "gain_positive_is_better": 0.0,
        "fallback": "exact champion; no candidate expert; excluded from LATE gates",
    }


def _fit_accounting() -> dict[str, Any]:
    events = [json.loads(line) for line in EVENT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()] if EVENT_PATH.is_file() else []
    saved: list[dict[str, Any]] = []
    for cutoff in PILOT_CUTOFFS:
        path = RUN_DIR / f"R15_fit{cutoff}.json"
        if path.is_file():
            saved.append(_read_json(path)["metadata"])
    event_names = ("FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED")
    per_cutoff = {str(cutoff): {name: 0 for name in event_names} for cutoff in PILOT_CUTOFFS}
    complete = True
    for event in events:
        cutoff, name = str(event.get("cutoff", "")), str(event.get("event", ""))
        if cutoff not in per_cutoff or name not in per_cutoff[cutoff]:
            complete = False
            continue
        per_cutoff[cutoff][name] += 1
    for counts in per_cutoff.values():
        started = counts["FIT_STARTED"]
        outcomes = counts["FIT_SAVED"] + counts["FIT_FAILED"]
        resumed = counts["RESUMED_VERIFIED"]
        if outcomes > started or started > outcomes + resumed:
            complete = False
    return {
        "expected_predictor_fits": 2,
        "successful_saved_predictor_fits": sum(int(item["fit_predictor_count"]) for item in saved),
        "fit_seconds_total": float(sum(float(item["fit_seconds"]) for item in saved)),
        "fit_origins_saved": [int(item["cutoff"]) for item in saved],
        "fit_started_events": sum(item.get("event") == "FIT_STARTED" for item in events),
        "fit_saved_events": sum(item.get("event") == "FIT_SAVED" for item in events),
        "fit_failed_events": sum(item.get("event") == "FIT_FAILED" for item in events),
        "resumed_verified_events": sum(item.get("event") == "RESUMED_VERIFIED" for item in events),
        "attempts_by_cutoff": per_cutoff,
        "attempt_accounting_complete": complete,
        "technical_retries": sum(item.get("event") == "FIT_FAILED" for item in events),
        "control_refits": 0,
        "peak_rss_sampled": False,
    }


def _gate(report: Mapping[str, Any], groups: Mapping[str, Any], verification: Mapping[str, Any]) -> dict[str, Any]:
    views = report["views"]
    primary_gain = float(views["primary"]["gain_positive_is_better"])
    h12_gain = float(views["H12_L"]["gain_positive_is_better"])
    w4_gain = float(views["W4"]["gain_positive_is_better"])
    w6_gain = float(views["W6"]["gain_positive_is_better"])
    failures = list(groups["guard_failures"])
    checks = {
        "independent_route_and_metric_verification": verification.get("status") == "PASS",
        "primary_gain_min_0_03": primary_gain >= 0.03,
        "H12_L_harm_max_0_05": h12_gain >= -0.05,
        "W4_harm_max_0_5": w4_gain >= -0.5,
        "W6_harm_max_0_5": w6_gain >= -0.5,
        "supported_tail_or_missing_harm_max_0_25": len(failures) == 0,
    }
    return {
        "card_id": CARD_ID,
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "evaluation_profile": "LATE; primary=W4+W6; H12_L=cycles68-79; H12_E exact fallback separate",
        "checks": checks,
        "thresholds": {
            "primary_MAE_gain_min": 0.03,
            "H12_L_MAE_harm_max": 0.05,
            "individual_W4_W6_MAE_harm_max": 0.5,
            "supported_group_min_n": MIN_GROUP_SUPPORT,
            "supported_tail_or_missing_MAE_harm_max": 0.25,
        },
        "metrics": {name: {
            "rows": int(item["rows"]),
            "baseline_mae": float(item["baseline"]["mae"]),
            "candidate_mae": float(item["candidate"]["mae"]),
            "gain_positive_is_better": float(item["gain_positive_is_better"]),
        } for name, item in views.items()},
        "supported_group_guard_failures": failures,
    }


def _reconcile_budget(result: Mapping[str, Any], verification: Mapping[str, Any],
                      gate: Mapping[str, Any]) -> None:
    accounting = result["fit_accounting"]
    if (accounting["successful_saved_predictor_fits"] != 2
            or sorted(accounting["fit_origins_saved"]) != list(PILOT_CUTOFFS)
            or not accounting["attempt_accounting_complete"]
            or verification.get("status") != "PASS"):
        raise BatchIntegrityError("R15 fit/verification accounting is incomplete")
    reconciliation = {
        "event": "FIT_BUDGET_RECONCILED", "card_id": CARD_ID, "budget_key": "R15:PILOT",
        "attempt_accounting_complete": True,
        "fit_started_events": accounting["fit_started_events"],
        "fit_saved_events": accounting["fit_saved_events"],
        "fit_failed_events": accounting["fit_failed_events"],
        "resumed_verified_events": accounting["resumed_verified_events"],
        "technical_retries": accounting["technical_retries"],
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": accounting["successful_saved_predictor_fits"],
        "decision": gate["status"], "independent_verification_status": "PASS",
        "freeze_signature": result["freeze_signature"],
        "attempts_sha256": file_sha256(EVENT_PATH),
        "results_sha256": file_sha256(RESULTS_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "subgroups_sha256": file_sha256(SUBGROUP_PATH),
        "gate_sha256": file_sha256(GATE_PATH),
    }
    existing = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()] if LEDGER_PATH.is_file() else []
    matches = [event for event in existing if event.get("event") == reconciliation["event"]
               and event.get("card_id") == CARD_ID]
    if matches:
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in reconciliation.items()
                                    if key != "event"):
            raise BatchIntegrityError("Existing R15 fit-budget reconciliation differs")
        return
    r01.append_event(LEDGER_PATH, {**reconciliation, "at_utc": pd.Timestamp.now(tz="UTC").isoformat()})


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False) -> dict[str, Any]:
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_match(PREFLIGHT_PATH, context["public"])
        skipped = {
            "card_id": CARD_ID, "status": context["public"]["status"],
            "verdict": context["public"]["status"], "preflight": context["public"],
            "candidate_fit_or_metric_computed": False,
            "test_or_sample_access": False, "kaggle_access": False,
        }
        _write_or_match(RESULTS_PATH, skipped)
        return skipped

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_match(RECIPE_PATH, context["recipe"])
    signature = json_sha256(context["identity"])
    if ENGINE_REPORT_PATH.is_file():
        frozen = _read_json(FREEZE_PATH) if FREEZE_PATH.is_file() else {}
        report = _read_json(ENGINE_REPORT_PATH)
        if (frozen.get("identity") != context["identity"] or frozen.get("signature") != signature
                or report.get("freeze_signature") != signature):
            raise BatchIntegrityError("R15 existing report/freeze differs from the frozen source or inputs")
    else:
        report = run_pilot(
            frame=context["frame"], target=context["target"], card=card,
            feature_names=FEATURE_NAMES, adapter=fit_predict,
            control_loader=context["control_loader"], freeze_identity=context["identity"],
            freeze_path=FREEZE_PATH, artifact_dir=RUN_DIR, event_path=EVENT_PATH,
            evaluation_views=context["views"], threads=THREADS,
            report_path=ENGINE_REPORT_PATH, budget_ledger_path=LEDGER_PATH,
            max_successful_fits=MAX_SUCCESSFUL_FITS,
        )

    per_cutoff = _read_saved_predictions(context, report)
    candidate_reconstruction = {
        str(cutoff): {
            "status": item["status"], "checks": item["checks"],
            "rows": int(len(item["rows"])),
        } for cutoff, item in per_cutoff.items()
    }
    verification = verify_report(
        report=report, frame=context["frame"], target=context["target"],
        artifact_dir=RUN_DIR, freeze_signature=report["freeze_signature"],
        evaluation_views=context["views"],
        prediction_reconstructors={CARD_ID: reconstruct_r15_arrays},
    )
    verification["r15_candidate_control_checks"] = candidate_reconstruction
    verification["r15_status"] = "PASS" if all(
        item["status"] == "PASS" for item in per_cutoff.values()
    ) else "FAIL"
    verification["status"] = "PASS" if verification.get("status") == "PASS" and verification["r15_status"] == "PASS" else "FAIL"

    views = _view_arrays(context, per_cutoff)
    groups = _group_metrics(context, views)
    h12_e = _early_fallback(context)
    group_payload = {"card_id": CARD_ID, "status": "PASS", **groups,
                     "H12_E_exact_champion_fallback": h12_e}
    gate = _gate(report, groups, verification)
    preflight = context["public"]
    results = {
        "card_id": CARD_ID,
        "status": "PILOT_VERIFIED",
        "verdict": gate["status"],
        "source_commit_context": context["identity"]["source_commit_context"],
        "freeze_signature": report["freeze_signature"],
        "input_sha256": context["identity"]["input_sha256"],
        "source_sha256": context["identity"]["source_sha256"],
        "recipe_sha256": json_sha256(context["recipe"]),
        "fit_accounting": _fit_accounting(),
        "population": {
            "fit_rows_by_cutoff": {str(cutoff): int(len(context["fit_rows_by_cutoff"][cutoff]))
                                    for cutoff in PILOT_CUTOFFS},
            "missing_fit_rows_by_cutoff": context["public"]["support_preflight"]["cutoffs"],
            "score_rows_by_cutoff": {str(cutoff): int(len(context["score_rows_by_cutoff"][cutoff]))
                                      for cutoff in PILOT_CUTOFFS},
            "view_rows": EXPECTED_VIEW_ROWS,
            "view_missing_stream_rows": {
                name: int(groups["views"][name]["signal_stream_missing"]["n"])
                for name in context["views"]
            },
            "fit_subset_only_missing_stream": True,
        },
        "views": report["views"],
        "H12_E_exact_champion_fallback": h12_e,
        "group_diagnostics": groups,
        "support_preflight": preflight["support_preflight"],
        "fallback_coverage": {
            name: {
                "rows": int(len(item["rows"])),
                "missing_rows_expert_route": int(np.asarray(item["missing"], dtype=bool).sum()),
                "present_rows_exact_champion": int((~np.asarray(item["missing"], dtype=bool)).sum()),
                "present_prediction_changed": False,
            } for name, item in views.items()
        },
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_test_prediction_made": False,
        "validation_labels_passed_to_adapter": False,
        "percentile_or_rank_mapping_used": False,
    }
    _write_or_match(PREFLIGHT_PATH, preflight)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(SUBGROUP_PATH, group_payload)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(RESULTS_PATH, results)
    fit_metadata = [_read_json(RUN_DIR / f"R15_fit{cutoff}.json")["metadata"] for cutoff in PILOT_CUTOFFS]
    runtime_notes = {
        "card_id": CARD_ID,
        "freeze_signature": report["freeze_signature"],
        "source_commit": context["identity"]["source_commit_context"],
        "predictor_fits": {
            "started": results["fit_accounting"]["fit_started_events"],
            "saved": results["fit_accounting"]["successful_saved_predictor_fits"],
            "failed": results["fit_accounting"]["fit_failed_events"],
            "technical_retries": results["fit_accounting"]["technical_retries"],
            "cutoff_67_seconds": float(fit_metadata[0]["fit_seconds"]),
            "cutoff_75_seconds": float(fit_metadata[1]["fit_seconds"]),
            "total_seconds": float(results["fit_accounting"]["fit_seconds_total"]),
        },
        "warnings_observed": [warning for item in fit_metadata
                              for warning in item.get("fit_metadata", {}).get("fit_warnings", [])],
        "memory": {
            "preflight_estimated_peak_gib_by_cutoff": {
                key: float(value["estimated_additional_peak_bytes"] / 1024**3)
                for key, value in preflight["resource_preflight"].items()
            },
            "available_ram_gib_at_preflight_by_cutoff": {
                key: float(value["available_ram_bytes"] / 1024**3)
                for key, value in preflight["resource_preflight"].items()
            },
            "job_cap_gib": 6.0,
            "peak_process_rss_measured": False,
            "peak_process_rss_gib": None,
        },
        "official_test_read": False,
        "kaggle_accessed": False,
    }
    _write_or_match(RUNTIME_NOTES_PATH, runtime_notes)
    _reconcile_budget(results, verification, gate)
    summary = {
        "card_id": CARD_ID, "status": gate["status"], "verdict": gate["status"],
        "primary_gain": gate["metrics"]["primary"]["gain_positive_is_better"],
        "H12_L_gain": gate["metrics"]["H12_L"]["gain_positive_is_better"],
        "W4_gain": gate["metrics"]["W4"]["gain_positive_is_better"],
        "W6_gain": gate["metrics"]["W6"]["gain_positive_is_better"],
        "H12_E_policy": "exact champion fallback",
        "successful_predictor_fits": results["fit_accounting"]["successful_saved_predictor_fits"],
        "independent_verification": verification["status"],
        "test_or_sample_access": False, "kaggle_access": False,
    }
    _write_or_match(SUMMARY_PATH, summary)
    return {"results": results, "verification": verification,
            "subgroups": group_payload, "gate": gate, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
