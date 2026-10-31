"""Read-only preflight and frozen execution bridge for classification card C02."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np
import pandas as pd
import psutil

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_sparse_classification import (
    CACHE as SPARSE_PILOT_CACHE,
    CONFIG as SPARSE_PILOT_CONFIG,
    check as check_sparse_pilot,
)
from scripts.optimization_v3_sparse_classification_fullsix import (
    CB_CACHE,
    control_record,
    pilot_record,
)
from src.models.classification_models import _text_values
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c02 import (
    BLEND_C2B_WEIGHT,
    BLEND_EXPERT_WEIGHT,
    DECISION_THRESHOLD,
    FEATURE_NAMES,
    fit_predict,
)
from src.models.regression_research_text import STRUCTURED_FEATURES
from src.models.round1_models import (
    CATEGORICAL_FEATURES,
    MIXED_COUNT_FEATURES,
    NUMERIC_FEATURES,
    TELEMETRY_FEATURE,
)
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/C02/freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C02"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C02"
PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "threadpoolctl", "psutil",
]


def reconstruct_c02(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independently rebuild C02 and frozen sparse25 from stored controls."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(x) for x in (raw, missing, c2b, cb, old_sparse)}) != 1:
        raise BatchIntegrityError("C02 route inputs have inconsistent lengths")
    if not np.isfinite(raw).all() or not np.isfinite(c2b).all() or not np.isfinite(cb).all() or not np.isfinite(old_sparse).all():
        raise BatchIntegrityError("C02 route inputs contain nonfinite probabilities")
    challenger = np.where(missing, cb, BLEND_C2B_WEIGHT * c2b + BLEND_EXPERT_WEIGHT * raw)
    baseline = np.where(missing, cb, BLEND_C2B_WEIGHT * c2b + BLEND_EXPERT_WEIGHT * old_sparse)
    return challenger, baseline


def _canonical_card(queue: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    canonical_queue = load_queue(QUEUE_PATH)
    canonical = {str(item["id"]): item for item in canonical_queue["cards"]}.get("C02")
    passed = {str(item["id"]): item for item in queue.get("cards", [])}.get("C02")
    if canonical is None or card != canonical or passed != canonical:
        raise BatchIntegrityError("C02 card/queue differs from the canonical queue on disk")
    if (card.get("task") != "classification" or card.get("evaluation") != "STANDARD"
            or card.get("policy") != "C_SPARSE_REPLACE"
            or card.get("pilot_predictor_fits") != 2
            or card.get("condition") != "No additional dependency"):
        raise BatchIntegrityError("C02 canonical task, policy, support, or fit count changed")
    return canonical_queue


def _validate_baseline_manifest(queue: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = ROOT / queue["baseline"]["classification"]["manifest"]
    expected_sha = queue["baseline"]["classification"]["manifest_sha256"]
    if file_sha256(manifest_path) != expected_sha:
        raise BatchIntegrityError("Accepted V3 classification manifest hash differs from the queue")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_recipe = {
        "kind": "availability_present_blend",
        "present": [{"model": "C2_B", "weight": 0.75}, {"model": "SPARSE_BINARY", "weight": 0.25}],
        "missing": "CB_MISSING",
        "threshold": 0.5,
    }
    if manifest.get("task") != "classification" or manifest.get("recipe") != expected_recipe:
        raise BatchIntegrityError("Accepted sparse25 control manifest does not match the fixed C02 policy")
    decision_rel = manifest.get("decision_path")
    if decision_rel != "experiments/optimization_v3/decisions/v3_cls_sparse25.json":
        raise BatchIntegrityError("Accepted sparse25 decision path differs from the frozen control")
    decision_path = ROOT / decision_rel
    if file_sha256(decision_path) != manifest.get("decision_sha256"):
        raise BatchIntegrityError("Accepted sparse25 decision hash differs from its manifest")
    return manifest_path, decision_path, manifest


def _structured_nnz_upper_per_row() -> int:
    """Bound transformed structured nonzeros using the frozen parser contract."""
    upper = 0
    for feature in STRUCTURED_FEATURES:
        if feature in MIXED_COUNT_FEATURES:
            upper += 5
        elif feature in NUMERIC_FEATURES or feature == TELEMETRY_FEATURE:
            upper += 3
        elif feature in CATEGORICAL_FEATURES:
            upper += 2
        else:
            raise BatchIntegrityError(f"No C02 resource bound for structured feature {feature}")
    return upper


def _document_nnz_upper(frame: pd.DataFrame, rows: np.ndarray) -> int:
    docs, missing = _text_values(frame.iloc[rows]["signal_stream"])
    if missing.any():
        docs = [doc for doc, is_missing in zip(docs, missing) if not is_missing]
    token_counts = np.fromiter(
        (len(re.findall(r"(?u)\b\w\w+\b", value.lower())) for value in docs),
        dtype=np.int64,
        count=len(docs),
    )
    # Frozen word TF-IDF has unigrams and bigrams, no more than T+(T-1) terms.
    return int(np.sum(token_counts + np.maximum(token_counts - 1, 0), dtype=np.int64))


def _preflight(
    frame: pd.DataFrame,
    target: np.ndarray,
    train_path: Path,
    expected_sha: str,
    queue: dict[str, Any],
    views: dict[str, list[tuple[int, int, int]]],
) -> dict[str, Any]:
    digest = file_sha256(train_path)
    if digest != expected_sha:
        raise BatchIntegrityError("Classification Train hash differs from frozen V3 baseline")
    if frame.columns.has_duplicates:
        raise BatchIntegrityError("Official Classification Train has duplicate column names")
    required = set(FEATURE_NAMES) | {"record_id", "artifact_signal", "cycle_code"}
    if not required.issubset(frame.columns):
        raise BatchIntegrityError(f"Official Classification Train lacks required fields: {sorted(required - set(frame.columns))}")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    if len(frame) != 50608 or not np.array_equal(np.unique(cycles), np.arange(47, 80)):
        raise BatchIntegrityError("Expected official Classification Train cycles 47-79 and 50,608 rows")
    if frame["record_id"].isna().any() or not frame["record_id"].astype(str).is_unique:
        raise BatchIntegrityError("Train record_id is missing or duplicated")
    if target.shape != (len(frame),) or not np.isin(target, [0, 1]).all():
        raise BatchIntegrityError("Train artifact_signal is invalid")

    expected_views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    if views != expected_views:
        raise BatchIntegrityError(f"C02 STANDARD views differ from the frozen card: {views}")
    view_rows: dict[str, int] = {}
    for name, segments in views.items():
        rows = [np.flatnonzero((cycles >= low) & (cycles <= high)) for _, low, high in segments]
        pooled = np.concatenate(rows)
        if len(np.unique(pooled)) != len(pooled):
            raise BatchIntegrityError(f"C02 view {name} duplicates Train rows")
        view_rows[name] = int(len(pooled))
    if view_rows["primary"] != 9064 or view_rows["H12"] != 38588:
        raise BatchIntegrityError(f"C02 evaluation denominators differ from canonical STANDARD views: {view_rows}")

    cutoff_support: dict[str, Any] = {}
    max_cutoff = max(origin for segments in views.values() for origin, _, _ in segments)
    fit_mask = cycles <= max_cutoff
    fit_rows = np.flatnonzero(fit_mask)
    fit_missing = stream_missing_mask(frame.iloc[fit_rows])
    fit_present_positions = fit_rows[~fit_missing]
    for cutoff in sorted({origin for segments in views.values() for origin, _, _ in segments}):
        cutoff_fit_rows = np.flatnonzero(cycles <= cutoff)
        cutoff_missing = stream_missing_mask(frame.iloc[cutoff_fit_rows])
        present_rows = cutoff_fit_rows[~cutoff_missing]
        present_y = target[present_rows]
        support = {
            "fit_rows_total": int(len(cutoff_fit_rows)),
            "fit_rows_present": int(len(present_rows)),
            "fit_rows_missing": int(cutoff_missing.sum()),
            "present_class_0": int(np.sum(present_y == 0)),
            "present_class_1": int(np.sum(present_y == 1)),
        }
        if not len(present_rows) or min(support["present_class_0"], support["present_class_1"]) == 0:
            raise BatchIntegrityError(f"C02 present-only fit lacks both classes at cutoff {cutoff}: {support}")
        cutoff_support[str(cutoff)] = support

    max_fit_present = fit_present_positions
    max_fit_docs_nnz = _document_nnz_upper(frame, max_fit_present)
    structured_nnz = _structured_nnz_upper_per_row()
    fit_nnz_upper = max_fit_docs_nnz + len(max_fit_present) * (structured_nnz + 1)
    fit_sparse_bytes = int(fit_nnz_upper * 16 + (len(max_fit_present) + 1) * 8)
    max_score_sparse_bytes = 0
    for cutoff in sorted({origin for segments in views.values() for origin, _, _ in segments}):
        score_cycles = {
            cycle for segments in views.values() for origin, low, high in segments if origin == cutoff
            for cycle in range(low, high + 1)
        }
        score_rows = np.flatnonzero(np.isin(cycles, np.asarray(sorted(score_cycles), dtype=int)))
        score_missing = stream_missing_mask(frame.iloc[score_rows])
        score_present = score_rows[~score_missing]
        score_nnz_upper = _document_nnz_upper(frame, score_present) + len(score_present) * (structured_nnz + 1)
        score_bytes = int(score_nnz_upper * 16 + (len(score_present) + 1) * 8)
        max_score_sparse_bytes = max(max_score_sparse_bytes, score_bytes)

    sparse_bytes_peak_inputs = fit_sparse_bytes + max_score_sparse_bytes
    estimated_peak_gib = float(1.5 + 3.0 * sparse_bytes_peak_inputs / (1024 ** 3))
    preset = queue["execution_preset"]
    cap_gib = float(preset["max_job_RSS_GiB"])
    reserve_gib = 2.0
    minimum_free_gib = float(preset["minimum_free_RAM_GiB_before_start"])
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    allowed_gib = min(cap_gib, available_gib - reserve_gib)
    if available_gib < minimum_free_gib or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"C02 resource preflight blocked: free={available_gib:.2f}GiB, "
            f"estimate={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB"
        )
    return {
        "train_sha256": digest,
        "train_rows": int(len(frame)),
        "fit_population_at_cutoff67": {
            "total_rows": int(len(fit_rows)),
            "present_rows": int(len(max_fit_present)),
            "missing_rows": int(fit_missing.sum()),
        },
        "present_fit_support_by_cutoff": cutoff_support,
        "evaluation_view_rows": view_rows,
        "upper_estimated_nnz_fit": int(fit_nnz_upper),
        "upper_estimated_sparse_bytes_fit": fit_sparse_bytes,
        "upper_estimated_sparse_bytes_max_score": max_score_sparse_bytes,
        "estimated_peak_job_gib": estimated_peak_gib,
        "available_ram_gib_observed": available_gib,
        "allowed_job_ram_gib": allowed_gib,
        "resource_bound": {
            "text_terms": "fit-present token upper bound T + max(T-1, 0), matching frozen unigram/bigram tokenization",
            "structured_nonzeros_per_row_upper": int(structured_nnz),
            "missing_indicator_nonzeros_per_row_upper": 1,
            "csr_bytes_per_nonzero": 16,
            "peak_multiplier": 3.0,
            "fixed_overhead_gib": 1.5,
            "reserve_gib": reserve_gib,
            "cap_gib": cap_gib,
        },
        "single_training_job": True,
        "threads": int(preset["threads_per_job"]),
    }


def _control_input_files(
    train_path: Path,
    card_path: Path,
    views: dict[str, list[tuple[int, int, int]]],
    manifest_path: Path,
    decision_path: Path,
) -> dict[str, Path]:
    ends = sorted({int(cutoff) for segments in views.values() for cutoff, _, _ in segments})
    inputs: dict[str, Path] = {
        "official_classification_train": train_path,
        "queue": QUEUE_PATH,
        "card": card_path,
        "accepted_classification_manifest": manifest_path,
        "accepted_classification_decision": decision_path,
        "frozen_sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "frozen_sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
    }
    for end in ends:
        inputs[f"sparse_pilot_{end}_npz"] = SPARSE_PILOT_CACHE / f"fit{end}.npz"
        inputs[f"sparse_pilot_{end}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{end}.json"
        inputs[f"catboost_{end}_npz"] = CB_CACHE / f"fit{end}.npz"
        inputs[f"catboost_{end}_sidecar"] = CB_CACHE / f"fit{end}.json"
        inputs[f"round1_c2b_{end}"] = ROOT / f"experiments/runs/classification_round1_v1/fit{end}__C2_B.npz"
        inputs[f"round1_audit_{end}"] = ROOT / f"reports/classification_round1/fit_audits/fit{end}.json"
    return inputs


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1_c02.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "scripts/research_campaign_classification.py",
        ROOT / "scripts/optimization_v2_classification.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "src/models/campaign_classification_catboost.py",
        ROOT / "src/models/campaign_catboost.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/data/audit.py",
        ROOT / "src/models/classification_metrics.py",
        ROOT / "src/utils/paths.py",
    ]


def _load_controls(frame: pd.DataFrame, target: np.ndarray):
    check_sparse_pilot()

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        control = control_record(frame, target, int(cutoff))
        sparse25 = pilot_record(frame, control)
        np.testing.assert_array_equal(control["rows"], score_rows)
        np.testing.assert_array_equal(sparse25["rows"], score_rows)
        baseline = np.asarray(sparse25["blend25"], dtype=np.float64)
        reconstructed = np.where(
            control["missing"], control["cb_missing"],
            BLEND_C2B_WEIGHT * control["c2b"] + BLEND_EXPERT_WEIGHT * sparse25["raw_candidate"],
        )
        np.testing.assert_array_equal(baseline, reconstructed)
        return {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": np.asarray(sparse25["raw_candidate"], dtype=np.float64),
            "c2b_probability": np.asarray(control["c2b"], dtype=np.float64),
            "cb_probability": np.asarray(control["cb_missing"], dtype=np.float64),
            "stream_missing": np.asarray(control["missing"], dtype=bool),
        }

    return load


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    canonical_queue = _canonical_card(queue, card)
    train_path = task_files("classification").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Classification Train is missing")
    train_path = Path(train_path)
    frame = pd.read_csv(train_path)
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    views = evaluation_views_for_card(canonical_queue, "C02")
    expected_sha = canonical_queue["baseline"]["classification"]["Train_sha256"]
    manifest_path, decision_path, manifest = _validate_baseline_manifest(canonical_queue)
    preflight = _preflight(frame, target, train_path, expected_sha, canonical_queue, views)

    controls = _load_controls(frame, target)
    control_files = _control_input_files(
        train_path, ROOT / card["card_path"], views, manifest_path, decision_path
    )
    sources = _source_files()
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=control_files,
        source_files=sources,
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity["accepted_baseline_manifest_sha256"] = canonical_queue["baseline"]["classification"]["manifest_sha256"]
    identity["accepted_baseline_decision_sha256"] = manifest["decision_sha256"]
    identity["python_version"] = platform.python_version()
    identity["v3_sparse_pilot_freeze_signature"] = json.loads(
        (SPARSE_PILOT_CONFIG / "freeze.json").read_text(encoding="utf-8")
    )["signature"]
    identity["resource_preflight_recipe"] = preflight["resource_bound"] | {
        "estimated_peak_job_gib": preflight["estimated_peak_job_gib"],
        "minimum_free_ram_gib": canonical_queue["execution_preset"]["minimum_free_RAM_GiB_before_start"],
    }
    identity["fixed_policy"] = {
        "present": {"C2_B": BLEND_C2B_WEIGHT, "new_sparse_expert": BLEND_EXPERT_WEIGHT},
        "missing": "frozen_CatBoost_probability",
        "threshold": DECISION_THRESHOLD,
        "missing_raw_candidate_sentinel": 0.5,
    }

    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    cutoff_controls: dict[str, Any] = {}
    for cutoff in sorted({origin for segments in views.values() for origin, _, _ in segments}):
        score_cycles = {
            cycle for segments in views.values() for origin, low, high in segments if origin == cutoff
            for cycle in range(low, high + 1)
        }
        score_rows = np.flatnonzero(np.isin(cycles, np.asarray(sorted(score_cycles), dtype=int)))
        bridged = controls(int(cutoff), score_rows)
        if np.asarray(bridged["baseline_prediction"]).shape != (len(score_rows),):
            raise BatchIntegrityError(f"C02 baseline bridge shape mismatch at cutoff {cutoff}")
        if np.asarray(bridged["stream_missing"]).shape != (len(score_rows),):
            raise BatchIntegrityError(f"C02 stream route mask shape mismatch at cutoff {cutoff}")
        if not np.array_equal(
            np.asarray(bridged["baseline_prediction"], dtype=np.float64),
            np.where(
                bridged["stream_missing"], bridged["cb_probability"],
                BLEND_C2B_WEIGHT * bridged["c2b_probability"]
                + BLEND_EXPERT_WEIGHT * bridged["baseline_sparse_raw"],
            ),
        ):
            raise BatchIntegrityError(f"C02 frozen control does not reconstruct at cutoff {cutoff}")
        cutoff_controls[str(cutoff)] = {
            "score_rows": int(len(score_rows)),
            "score_present": int((~bridged["stream_missing"]).sum()),
            "score_missing": int(bridged["stream_missing"].sum()),
        }

    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT",
            "card_id": "C02",
            "preflight": preflight,
            "control_route_counts_by_cutoff": cutoff_controls,
            "source_files": len(sources),
            "control_files": len(control_files),
            "evaluation_views": views,
            "freeze_identity_sha256": json_sha256(identity),
            "official_test_read": False,
            "model_fit_performed": False,
            "freeze_written": False,
            "outputs_written": False,
        }

    append_event(REPORT_DIR / "attempts.jsonl", {"event": "PREFLIGHT_PASS", **preflight})
    report_path = REPORT_DIR / "pilot.json"
    budget_ledger_path = REPORT_DIR.parent / "batch_ledger.jsonl"
    report = run_pilot(
        frame=frame,
        target=target,
        card=card,
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=controls,
        freeze_identity=identity,
        freeze_path=FREEZE_PATH,
        artifact_dir=ARTIFACT_DIR,
        event_path=REPORT_DIR / "attempts.jsonl",
        evaluation_views=views,
        threads=int(canonical_queue["execution_preset"]["threads_per_job"]),
        report_path=report_path,
        budget_ledger_path=budget_ledger_path,
        max_successful_fits=int(canonical_queue["execution_preset"]["max_successful_new_predictor_fits"]),
    )
    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=ARTIFACT_DIR,
        freeze_signature=report["freeze_signature"],
        evaluation_views=views,
        prediction_reconstructors={"C02": reconstruct_c02},
    )
    gate_config = canonical_queue["evaluation"]["classification_gate"]
    subgroup_failures = verification["views"]["primary"]["subgroup_diagnostics"]["supported_guard_failures"]
    checks = {
        "primary_gain_at_least_0_0005": report["views"]["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_gain_min"]),
        "H12_harm_at_most_0_001": report["views"]["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_harm_max"]),
        "W1_harm_at_most_0_005": report["views"]["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "W4_harm_at_most_0_005": report["views"]["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "supported_subgroup_guards": len(subgroup_failures) == 0,
    }
    gate = {
        "card_id": "C02",
        "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": gate_config,
        "supported_subgroup_failures": subgroup_failures,
        "decision_basis": "Predeclared temporal pilot gates only; no Test or Kaggle result used.",
    }
    gate_path = REPORT_DIR / "gate.json"
    if gate_path.exists():
        existing_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if existing_gate != gate:
            raise BatchIntegrityError("Existing C02 pilot gate differs from independently verified evidence")
    else:
        write_json_exclusive(gate_path, gate)
    verification_path = REPORT_DIR / "verification.json"
    if verification_path.exists():
        existing = json.loads(verification_path.read_text(encoding="utf-8"))
        if existing != verification:
            raise BatchIntegrityError("Existing C02 independent verification differs")
    else:
        write_json_exclusive(verification_path, verification)
    return {"pilot": report, "verification": verification, "gate": gate, "preflight": preflight}


__all__ = ["reconstruct_c02", "run_card"]
