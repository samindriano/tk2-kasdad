"""Preflight, freeze, and execute the C01 cache-only route comparison.

No estimator or preprocessing object is fitted. The adapter uses verified
Train-fold caches to replace only the stream-missing branch of V3 sparse25.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_sparse_classification as sparse_pilot
from scripts import optimization_v3_sparse_classification_fullsix as sparse_fullsix
from scripts.research_campaign_classification_stage2 import _load_round1_control
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    ensure_frozen,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c01 import (
    CARD_ID,
    FEATURE_NAMES,
    FIT_ENDPOINTS,
    SPARSE_WEIGHT,
    THRESHOLD,
    fit_predict,
    reconstruct_c01,
)
from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C01.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C01"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C01"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C01"
PILOT_REPORT_DIR = sparse_pilot.OUT
FULLSIX_REPORT_DIR = sparse_fullsix.OUT
BASELINE_MANIFEST_PATH = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"

TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "3f749ec6e63f836240fb44414d93a7e80266c6898c6ff5826b29bc9cb310c13f"
MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_CARD = {
    "id": "C01",
    "task": "classification",
    "name": "Cached alternative missing expert",
    "family": "route_missing",
    "priority": "P2",
    "change": "Recompose current sparse25 present probabilities with historical C1_B missing probabilities; threshold0.50. No new model or route threshold.",
    "policy": "C_CACHE",
    "pilot_predictor_fits": 0,
    "evaluation": "CACHE_SIX",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "card_path": "reports/optimization_v3/research_backlog/cards/C01.md",
}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "threadpoolctl"]
EXPECTED_PRIMARY_ROWS = 38_588
EXPECTED_H12_E_ROWS = 17_685
EXPECTED_H12_L_ROWS = 20_903


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise BatchIntegrityError(f"Existing C01 output differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, dict(payload))


def _validate_queue_manifest(queue: Mapping[str, Any]) -> dict[str, Any]:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("C01 queue or card hash differs from the frozen protocol")
    cards = {str(card["id"]): card for card in queue["cards"]}
    canonical = cards.get(CARD_ID)
    if canonical is None or {key: canonical.get(key) for key in EXPECTED_CARD} != EXPECTED_CARD:
        raise BatchIntegrityError("C01 canonical queue card differs from its frozen definition")
    baseline = queue["baseline"]["classification"]
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest") != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or not np.isclose(float(baseline.get("primary", np.nan)), 0.8299435476765076, atol=1e-15, rtol=0)
            or not np.isclose(float(baseline.get("H12", np.nan)), 0.8029655654017788, atol=1e-15, rtol=0)):
        raise BatchIntegrityError("C01 fixed sparse25 baseline or metric differs from the canonical queue")
    if file_sha256(BASELINE_MANIFEST_PATH) != MANIFEST_SHA256:
        raise BatchIntegrityError("Accepted sparse25 manifest hash changed")
    manifest = _read_json(BASELINE_MANIFEST_PATH)
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("task") != "classification"
            or manifest.get("candidate_id") != "v3_cls_sparse25"
            or manifest.get("input_sha256", {}).get("train") != TRAIN_SHA256
            or manifest.get("recipe") != {
                "kind": "availability_present_blend",
                "present": [
                    {"model": "C2_B", "weight": 0.75},
                    {"model": "SPARSE_BINARY", "weight": 0.25},
                ],
                "missing": "CB_MISSING",
                "threshold": 0.5,
            }):
        raise BatchIntegrityError("Accepted sparse25 route manifest differs from the C01 control")
    return manifest


def _load_source_state() -> tuple[
    dict[str, Any], pd.DataFrame, np.ndarray, Path, dict[str, Any],
    dict[str, Any], dict[int, dict[str, Any]], Any,
]:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256:
        raise BatchIntegrityError("Canonical V3 queue hash differs")
    queue = load_queue(QUEUE_PATH)
    manifest = _validate_queue_manifest(queue)
    views = evaluation_views_for_card(queue, CARD_ID)
    ends = [int(value) for value in queue["evaluation"]["CACHE_SIX"]["primary_cutoffs"]]
    if tuple(ends) != FIT_ENDPOINTS:
        raise BatchIntegrityError("C01 CACHE_SIX cutoffs differ from the frozen card")
    gate = queue["evaluation"]["classification_gate"]
    if (int(gate["supported_group_n_min"]) != 200
            or int(gate["positive_and_negative_per_group_min"]) != 30
            or float(gate["supported_group_F1_harm_max"]) != 0.01
            or float(gate["positive_class_recall_harm_max"]) != 0.02
            or int(gate["full_nonnegative_windows_min"]) != 4):
        raise BatchIntegrityError("C01 full-six classification gates differ from the canonical queue")

    frame, target, train_path = sparse_pilot.load_train(None)
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train hash differs")
    if (len(frame) != 50_608 or target.shape != (50_608,)
            or not frame.record_id.astype(str).is_unique
            or frame.record_id.isna().any()
            or frame.record_id.astype(str).str.strip().eq("").any()):
        raise BatchIntegrityError("Official Classification Train row or ID contract failed")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Classification Train cycle coverage differs from 47–79")

    pilot_freeze = sparse_pilot.check()
    pilot_report = _read_json(PILOT_REPORT_DIR / "pilot.json")
    pilot_verification = _read_json(PILOT_REPORT_DIR / "verification.json")
    fullsix_freeze = sparse_fullsix.check_freeze()
    fullsix_report = _read_json(FULLSIX_REPORT_DIR / "fullsix.json")
    fullsix_verification = _read_json(FULLSIX_REPORT_DIR / "verification.json")
    if (pilot_verification.get("status") != "PASS"
            or pilot_verification.get("newfits") != 0
            or pilot_report.get("freeze_signature") != pilot_freeze["signature"]
            or pilot_report.get("Test_read") is not False):
        raise BatchIntegrityError("Sparse pilot source cache lacks a passing verification receipt")
    if (fullsix_verification.get("status") != "PASS"
            or fullsix_verification.get("newfits") != 0
            or fullsix_verification.get("test_read") is not False
            or fullsix_verification.get("kaggle_access") is not False
            or fullsix_report.get("freeze_signature") != fullsix_freeze["signature"]
            or fullsix_report.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
            or fullsix_report.get("test_read") is not False
            or fullsix_report.get("kaggle_access") is not False):
        raise BatchIntegrityError("Sparse full-six source cache lacks a passing, outcome-blind receipt")
    if (pilot_report.get("newfits") != 2 or fullsix_report.get("new_fits") != 4
            or fullsix_report.get("reused_cutoffs") != [55, 67]
            or fullsix_report.get("new_fit_cutoffs") != [59, 63, 71, 75]):
        raise BatchIntegrityError("Sparse source cache fit lineage differs from its frozen receipts")

    records: dict[int, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    for cutoff in FIT_ENDPOINTS:
        current = sparse_fullsix.control_record(frame, target, cutoff)
        if cutoff in (55, 67):
            current = sparse_fullsix.pilot_record(frame, current)
            sparse_cache = sparse_pilot.CACHE / f"fit{cutoff}.npz"
            sparse_meta = sparse_cache.with_suffix(".json")
            sparse_signature = pilot_freeze["signature"]
        else:
            sparse_cache = sparse_fullsix.CACHE / f"fit{cutoff}.npz"
            sparse_meta = sparse_cache.with_suffix(".json")
            if not sparse_cache.is_file() or not sparse_meta.is_file():
                raise BatchIntegrityError(
                    f"C01 requires the existing full-six sparse cache at cutoff {cutoff}; no fit is permitted"
                )
            current = sparse_fullsix.load_existing_prediction(
                frame, current, fullsix_freeze["signature"]
            )
            if current is None:
                raise BatchIntegrityError(f"C01 missing full-six sparse prediction at cutoff {cutoff}")
            sparse_signature = fullsix_freeze["signature"]
        horizon = 67 if cutoff == 55 else 79 if cutoff == 67 else cutoff + 4
        c1_rows, c1_probability, c1_identity = _load_round1_control(
            frame, cutoff, horizon, "C1_B"
        )
        c1_audit = _read_json(
            sparse_fullsix.ROUND1_AUDITS / f"fit{cutoff}.json"
        ).get("C1_B")
        expected_fit_rows = int(np.sum((cycles >= 47) & (cycles <= cutoff)))
        if (not isinstance(c1_audit, dict)
                or c1_audit.get("model") != "C1_B"
                or c1_audit.get("fit_cycles") != [47, cutoff]
                or c1_audit.get("validation_prediction_cycles") != [cutoff + 1, horizon]
                or int(c1_audit.get("fit_rows", -1)) != expected_fit_rows
                or int(c1_audit.get("validation_rows", -1)) != len(c1_rows)
                or c1_audit.get("outer_validation_labels_passed_to_fit_or_selection") is not False
                or c1_audit.get("threshold_selected_before_outer_refit") is not True):
            raise BatchIntegrityError(
                f"C01 C1_B fit/validation cycle audit or label boundary failed at cutoff {cutoff}"
            )
        rows = np.asarray(current["rows"], dtype=np.int64)
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= horizon))
        if not np.array_equal(rows, expected_rows) or not np.array_equal(c1_rows, rows):
            raise BatchIntegrityError(f"C01 source rows differ from exact cycle view at cutoff {cutoff}")
        sparse_receipt = _read_json(sparse_meta)
        fit_state = sparse_receipt.get("state", {})
        if (sparse_receipt.get("outer_labels_used_for_fit", False) is not False
                or fit_state.get("eval_set_used") is not False
                or fit_state.get("early_stopping_used") is not False):
            raise BatchIntegrityError(f"Sparse candidate outer-validation boundary failed at cutoff {cutoff}")
        missing = stream_missing_mask(frame.iloc[rows])
        if not np.array_equal(missing, np.asarray(current["missing"], dtype=bool)):
            raise BatchIntegrityError(f"C01 source availability mask differs at cutoff {cutoff}")
        c2b = np.asarray(current["c2b"], dtype=np.float64)
        cb = np.asarray(current["cb_missing"], dtype=np.float64)
        raw_sparse = np.asarray(current["raw_candidate"], dtype=np.float64)
        incumbent = np.asarray(current["blend25"], dtype=np.float64)
        expected_incumbent = np.where(missing, cb, 0.75 * c2b + SPARSE_WEIGHT * raw_sparse)
        if not np.array_equal(incumbent, expected_incumbent):
            raise BatchIntegrityError(f"C01 current sparse25 incumbent does not reconstruct at cutoff {cutoff}")
        if not (len(c1_probability) == len(rows) and np.isfinite(c1_probability).all()
                and np.all((c1_probability >= 0.0) & (c1_probability <= 1.0))):
            raise BatchIntegrityError(f"C01 C1_B source probability is invalid at cutoff {cutoff}")
        if (sparse_receipt.get("npz_sha256") != file_sha256(sparse_cache)
                or sparse_receipt.get("freeze_signature") != sparse_signature):
            raise BatchIntegrityError(f"Sparse25 cache receipt differs at cutoff {cutoff}")
        record = {
            **current,
            "c1b_probability": np.asarray(c1_probability, dtype=np.float64),
            "c1b_identity": c1_identity,
            "horizon": int(horizon),
            "score_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
        }
        records[int(cutoff)] = record
        lineage[str(cutoff)] = {
            "fit_cutoff": int(cutoff),
            "fit_rows": int(np.sum((cycles >= 47) & (cycles <= cutoff))),
            "score_rows": int(len(rows)),
            "score_cycles": [int(x) for x in np.unique(cycles[rows])],
            "horizon": int(horizon),
            "missing_rows": int(np.sum(missing)),
            "present_rows": int(len(rows) - np.sum(missing)),
            "row_positions_sha256": hashlib_sha256(rows),
            "record_ids_sha256": hashlib_sha256(np.asarray(record["score_ids"], dtype=str)),
            "C1_B": c1_identity,
            "C1_B_audit_fit_cycles": c1_audit["fit_cycles"],
            "C1_B_audit_validation_prediction_cycles": c1_audit["validation_prediction_cycles"],
            "C1_B_outer_validation_labels_passed_to_fit_or_selection": c1_audit[
                "outer_validation_labels_passed_to_fit_or_selection"
            ],
            "C2_B_cache_sha256": file_sha256(sparse_fullsix.ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"),
            "CatBoost_cache_sha256": file_sha256(sparse_fullsix.CB_CACHE / f"fit{cutoff}.npz"),
            "sparse25_cache_sha256": file_sha256(sparse_cache),
            "sparse25_receipt_sha256": file_sha256(sparse_meta),
            "sparse25_freeze_signature": sparse_signature,
            "current_sparse25_route_reconstructed": True,
            "C1_B_fit_scope_validated_by_Round1_audit": True,
        }

    primary_rows = []
    for cutoff, low, high in views["primary"]:
        rows = np.asarray(records[int(cutoff)]["rows"], dtype=np.int64)
        primary_rows.append(rows[(cycles[rows] >= low) & (cycles[rows] <= high)])
    primary_rows_combined = np.concatenate(primary_rows)
    if (len(np.unique(primary_rows_combined)) != len(primary_rows_combined)
            or len(primary_rows_combined) != EXPECTED_PRIMARY_ROWS):
        raise BatchIntegrityError("C01 six-window primary row count differs from 38,588")
    if (lineage["55"]["score_rows"] != EXPECTED_H12_E_ROWS
            or lineage["67"]["score_rows"] != EXPECTED_H12_L_ROWS):
        raise BatchIntegrityError("C01 early or late H12 source denominator differs from frozen expectation")

    def control_loader(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        record = records[int(cutoff)]
        rows = np.asarray(score_rows, dtype=np.int64)
        if not np.array_equal(rows, record["rows"]):
            raise BatchIntegrityError(f"C01 engine score rows differ from source cache at cutoff {cutoff}")
        return {
            "baseline_prediction": np.asarray(record["blend25"], dtype=np.float64),
            "c1b_probability": np.asarray(record["c1b_probability"], dtype=np.float64),
            "c2b_probability": np.asarray(record["c2b"], dtype=np.float64),
            "cb_missing_probability": np.asarray(record["cb_missing"], dtype=np.float64),
            "sparse_raw_probability": np.asarray(record["raw_candidate"], dtype=np.float64),
            "stream_missing": np.asarray(record["missing"], dtype=bool),
            "source_rows": rows,
            "source_record_ids": np.asarray(record["score_ids"], dtype=str),
        }

    required_controls = {
        "baseline_prediction", "c1b_probability", "c2b_probability",
        "cb_missing_probability", "sparse_raw_probability", "stream_missing",
        "source_rows", "source_record_ids",
    }
    for cutoff in FIT_ENDPOINTS:
        record = records[int(cutoff)]
        controls = control_loader(int(cutoff), np.asarray(record["rows"], dtype=np.int64))
        if set(controls) != required_controls:
            raise BatchIntegrityError(f"C01 adapter control fields differ at cutoff {cutoff}")
        expected_ids = np.asarray(record["score_ids"], dtype=str)
        if (any(np.asarray(value).shape != (len(record["rows"]),) for value in controls.values())
                or not np.array_equal(controls["source_rows"], record["rows"])
                or not np.array_equal(controls["source_record_ids"], expected_ids)
                or not np.array_equal(controls["baseline_prediction"], record["blend25"])):
            raise BatchIntegrityError(f"C01 adapter control arrays are misaligned at cutoff {cutoff}")

    control_check = {
        "baseline_manifest_sha256": file_sha256(BASELINE_MANIFEST_PATH),
        "pilot_verification_status": pilot_verification["status"],
        "fullsix_verification_status": fullsix_verification["status"],
        "fullsix_prediction_source_signature": fullsix_freeze["signature"],
        "pilot_prediction_source_signature": pilot_freeze["signature"],
        "all_six_cutoffs_aligned": True,
        "all_six_availability_masks_reconstructed": True,
        "all_six_sparse25_incumbents_reconstructed": True,
        "all_six_C1_B_lineages_validated": True,
        "all_six_adapter_control_bridges_validated": True,
        "source_lineage_by_cutoff": lineage,
        "test_data_used": False,
        "kaggle_access": False,
    }
    return queue, frame, target, train_path, views, control_check, records, control_loader


def hashlib_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.kind in "US":
        payload = "\n".join(str(value) for value in array.tolist()).encode("utf-8")
    else:
        payload = array.astype("<i8", copy=False).tobytes(order="C")
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _input_files(train_path: Path) -> dict[str, Path]:
    files: dict[str, Path] = {
        "official_classification_train": train_path,
        "queue": QUEUE_PATH,
        "card": CARD_PATH,
        "accepted_sparse25_manifest": BASELINE_MANIFEST_PATH,
        "sparse_pilot_protocol": sparse_pilot.CONFIG / "protocol.json",
        "sparse_pilot_freeze": sparse_pilot.CONFIG / "freeze.json",
        "sparse_pilot_report": sparse_pilot.OUT / "pilot.json",
        "sparse_pilot_verification": sparse_pilot.OUT / "verification.json",
        "sparse_fullsix_protocol": sparse_fullsix.CONFIG / "protocol.json",
        "sparse_fullsix_freeze": sparse_fullsix.CONFIG / "freeze.json",
        "sparse_fullsix_report": sparse_fullsix.OUT / "fullsix.json",
        "sparse_fullsix_verification": sparse_fullsix.OUT / "verification.json",
    }
    for cutoff in FIT_ENDPOINTS:
        sparse_dir = sparse_pilot.CACHE if cutoff in (55, 67) else sparse_fullsix.CACHE
        files[f"sparse25_fit{cutoff}_cache"] = sparse_dir / f"fit{cutoff}.npz"
        files[f"sparse25_fit{cutoff}_receipt"] = sparse_dir / f"fit{cutoff}.json"
        files[f"C1_B_fit{cutoff}_cache"] = sparse_fullsix.ROUND1_CACHE / f"fit{cutoff}__C1_B.npz"
        files[f"C1_B_fit{cutoff}_audit"] = sparse_fullsix.ROUND1_AUDITS / f"fit{cutoff}.json"
        files[f"C2_B_fit{cutoff}_cache"] = sparse_fullsix.ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"
        files[f"CatBoost_fit{cutoff}_cache"] = sparse_fullsix.CB_CACHE / f"fit{cutoff}.npz"
        files[f"CatBoost_fit{cutoff}_receipt"] = sparse_fullsix.CB_CACHE / f"fit{cutoff}.json"
    return files


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_c01.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v2_classification.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/utils/paths.py",
    ]


def _recipe(queue: Mapping[str, Any], control_check: Mapping[str, Any], views: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_id": CARD_ID,
        "status": "FROZEN_RECIPE_BEFORE_CACHE_RECOMPOSITION",
        "card_sha256": CARD_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "baseline_manifest_sha256": MANIFEST_SHA256,
        "baseline_manifest": "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
        "control_metrics": {"primary_macro_f1": 0.8299435476765076, "H12_macro_f1": 0.8029655654017788},
        "policy": {
            "stream_missing": "historical C1_B probability",
            "stream_present": "0.75 * C2_B probability + 0.25 * SPARSE_BINARY probability",
            "threshold": THRESHOLD,
            "learned_threshold": False,
            "new_predictor_fits": 0,
            "new_preprocessing_fits": 0,
        },
        "cache_source_cutoffs": list(FIT_ENDPOINTS),
        "fit_endpoints": list(FIT_ENDPOINTS),
        "evaluation_views": {key: [list(segment) for segment in value] for key, value in views.items()},
        "source_lineage": dict(control_check),
        "support": {
            "primary_six_rows": EXPECTED_PRIMARY_ROWS,
            "H12_E_rows": EXPECTED_H12_E_ROWS,
            "H12_L_rows": EXPECTED_H12_L_ROWS,
        },
        "gates": dict(queue["evaluation"]["classification_gate"]),
        "threshold_for_metrics": 0.50,
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
        "kaggle_access": False,
    }


def _freeze_identity(queue: Mapping[str, Any], train_path: Path, control_check: Mapping[str, Any],
                     views: Mapping[str, Any]) -> dict[str, Any]:
    identity = make_freeze_identity(
        card=EXPECTED_CARD,
        feature_names=FEATURE_NAMES,
        input_files=_input_files(train_path),
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity.update({
        "baseline_manifest_sha256": MANIFEST_SHA256,
        "source_lineage": dict(control_check),
        "recipe_sha256": json_sha256(_recipe(queue, control_check, views)),
        "python_version": platform.python_version(),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
        "kaggle_access": False,
    })
    return identity


def preflight() -> dict[str, Any]:
    queue, frame, target, train_path, views, control_check, records, _ = _load_source_state()
    del target
    identity = _freeze_identity(queue, train_path, control_check, views)
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    window_rows: dict[str, int] = {}
    for name, segments in views.items():
        selected_rows = []
        for cutoff, low, high in segments:
            rows = np.asarray(records[int(cutoff)]["rows"], dtype=np.int64)
            selected_rows.append(rows[(cycles[rows] >= low) & (cycles[rows] <= high)])
        combined = np.concatenate(selected_rows)
        if len(np.unique(combined)) != len(combined):
            raise BatchIntegrityError(f"C01 view {name} contains duplicate Train rows")
        window_rows[name] = int(len(combined))
    if window_rows.get("primary") != EXPECTED_PRIMARY_ROWS:
        raise BatchIntegrityError("C01 six-window primary denominator differs from 38,588")
    return {
        "status": "PREFLIGHT_PASS_NO_FREEZE_NO_METRICS_NO_FIT",
        "card_id": CARD_ID,
        "train_sha256": TRAIN_SHA256,
        "train_rows": int(len(frame)),
        "source_cache_cutoffs": {
            str(cutoff): {
                "rows": int(len(records[cutoff]["rows"])),
                "missing_rows": int(np.sum(records[cutoff]["missing"])),
                "present_rows": int(np.sum(~records[cutoff]["missing"])),
            }
            for cutoff in FIT_ENDPOINTS
        },
        "view_rows": window_rows,
        "source_lineage": control_check,
        "freeze_identity_sha256": json_sha256(identity),
        "predictor_fits_performed": 0,
        "validation_metrics_computed": False,
        "test_data_read": False,
        "kaggle_access": False,
        "freeze_or_output_written": False,
    }


def freeze_only() -> dict[str, Any]:
    result = preflight()
    queue, _, _, train_path, views, control_check, _, _ = _load_source_state()
    identity = _freeze_identity(queue, train_path, control_check, views)
    recipe = _recipe(queue, control_check, views)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_frozen(FREEZE_PATH, identity)
    _write_or_match(REPORT_DIR / "recipe.json", recipe)
    result["status"] = "FROZEN_BEFORE_CACHE_RECOMPOSITION"
    result["freeze_signature"] = json_sha256(identity)
    result["freeze_path"] = str(FREEZE_PATH.relative_to(ROOT))
    result["recipe_path"] = str((REPORT_DIR / "recipe.json").relative_to(ROOT))
    result["freeze_or_output_written"] = True
    return result


def _view_subgroup_failures(verification: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for view_name, view in verification["views"].items():
        diagnostics = view.get("subgroup_diagnostics", {})
        for failure in diagnostics.get("supported_guard_failures", []):
            failures.append({"view": view_name, **failure})
    return failures


def run_frozen_cache_only() -> dict[str, Any]:
    queue, frame, target, train_path, views, control_check, _, control_loader = _load_source_state()
    identity = _freeze_identity(queue, train_path, control_check, views)
    if not FREEZE_PATH.is_file():
        raise BatchIntegrityError("C01 freeze is absent; run --freeze-only first")
    frozen = _read_json(FREEZE_PATH)
    if frozen.get("identity") != identity or frozen.get("signature") != json_sha256(identity):
        raise BatchIntegrityError("C01 source/cache identity changed after freeze")
    if not ARTIFACT_DIR.exists():
        ARTIFACT_DIR.mkdir(parents=True)
    attempts_path = REPORT_DIR / "attempts.jsonl"
    if attempts_path.is_file():
        failed = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()
                  if line.strip() and json.loads(line).get("card_id") == CARD_ID
                  and json.loads(line).get("event") == "FIT_FAILED"]
        if failed:
            raise BatchIntegrityError("C01 contains a failed prior reconstruction; refusing silent retry")

    report = run_pilot(
        frame=frame,
        target=target,
        card=next(card for card in queue["cards"] if card["id"] == CARD_ID),
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=control_loader,
        freeze_identity=identity,
        freeze_path=FREEZE_PATH,
        artifact_dir=ARTIFACT_DIR,
        event_path=attempts_path,
        evaluation_views=views,
        threads=2,
        report_path=ARTIFACT_DIR / "engine_report.json",
        budget_ledger_path=REPORT_DIR.parent / "batch_ledger.jsonl",
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
    )
    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=ARTIFACT_DIR,
        freeze_signature=report["freeze_signature"],
        evaluation_views=views,
        prediction_reconstructors={CARD_ID: reconstruct_c01},
    )
    subgroup_failures = _view_subgroup_failures(verification)
    gates = queue["evaluation"]["classification_gate"]
    nonnegative_windows = sum(
        report["views"][f"W{i}"]["gain_positive_is_better"] >= 0.0 for i in range(1, 7)
    )
    checks = {
        "full_primary_gain_min": report["views"]["primary"]["gain_positive_is_better"] >= float(gates["full_primary_gain_min"]),
        "full_H12_harm_max": report["views"]["H12"]["gain_positive_is_better"] >= -float(gates["full_H12_harm_max"]),
        "full_nonnegative_windows_min": nonnegative_windows >= int(gates["full_nonnegative_windows_min"]),
        "supported_subgroup_F1_and_positive_recall_guards": len(subgroup_failures) == 0,
        "independent_metric_and_route_reconstruction": verification.get("status") == "PASS",
        "zero_predictor_fits": int(report["fit_predictor_count"]) == 0,
    }
    gate = {
        "card_id": CARD_ID,
        "profile": "CACHE_SIX",
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": {
            "primary_gain_min": gates["full_primary_gain_min"],
            "H12_harm_max": gates["full_H12_harm_max"],
            "nonnegative_windows_min": gates["full_nonnegative_windows_min"],
            "supported_group_n_min": gates["supported_group_n_min"],
            "each_class_per_group_min": gates["positive_and_negative_per_group_min"],
            "supported_group_F1_harm_max": gates["supported_group_F1_harm_max"],
            "positive_class_recall_harm_max": gates["positive_class_recall_harm_max"],
        },
        "nonnegative_windows": int(nonnegative_windows),
        "supported_group_guard_failures": subgroup_failures,
        "decision_basis": "Frozen full-six temporal metrics and supported subgroup guards; no threshold tuning, Test, or Kaggle result used.",
    }
    results = {
        "status": "CACHE_SIX_COMPLETE",
        "card_id": CARD_ID,
        "freeze_signature": report["freeze_signature"],
        "test_data_read": False,
        "kaggle_access": False,
        "predictor_fits": 0,
        "preprocessing_fits": 0,
        "recomposed_cutoffs": list(FIT_ENDPOINTS),
        "threshold": THRESHOLD,
        "views": report["views"],
        "source_lineage_by_cutoff": control_check["source_lineage_by_cutoff"],
        "verification_status": verification["status"],
        "gate_status": gate["status"],
    }
    verification_out = {
        "status": "PASS",
        "card_id": CARD_ID,
        "freeze_signature": report["freeze_signature"],
        "generic_engine_report": report,
        "source_control_reconstruction": control_check,
        "independent_metric_and_route_verification": verification,
        "predictor_fits": 0,
        "cache_recomposition_calls": int(len(FIT_ENDPOINTS)),
        "score_row_and_record_id_alignment_verified": True,
        "availability_masks_verified": True,
        "test_data_read": False,
        "kaggle_access": False,
    }
    _write_or_match(REPORT_DIR / "results.json", results)
    _write_or_match(REPORT_DIR / "verification.json", verification_out)
    _write_or_match(REPORT_DIR / "gate.json", gate)
    primary = report["views"]["primary"]
    h12 = report["views"]["H12"]
    deltas = [report["views"][f"W{i}"]["gain_positive_is_better"] for i in range(1, 7)]
    summary = (
        "# C01 Cache-only Route Summary\n\n"
        f"- Decision: **{gate['status']}**.\n"
        f"- Primary Macro F1: {primary['baseline']['macro_f1']:.6f} sparse25 → "
        f"{primary['candidate']['macro_f1']:.6f} C01; gain {primary['gain_positive_is_better']:+.6f} "
        f"on {primary['rows']:,} rows.\n"
        f"- H12 Macro F1: {h12['baseline']['macro_f1']:.6f} → {h12['candidate']['macro_f1']:.6f}; "
        f"gain {h12['gain_positive_is_better']:+.6f} on {h12['rows']:,} rows.\n"
        f"- W1–W6 gains: {', '.join(f'{value:+.6f}' for value in deltas)}; "
        f"{nonnegative_windows}/6 windows nonnegative.\n"
        f"- Supported subgroup guard failures: {len(subgroup_failures)}.\n"
        "- Six temporal windows were recomposed from frozen Train-fold caches; no predictor or preprocessing fit was run.\n"
        "- Temporal validation outcomes are reused development evidence, not untouched holdout evidence.\n"
        "- Classification Test, SampleSubmission, and Kaggle were not accessed.\n"
    )
    summary_path = REPORT_DIR / "summary.md"
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing C01 summary differs; refusing overwrite")
    else:
        with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(summary)
    return {"results": results, "verification": verification_out, "gate": gate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-cache-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        result = preflight()
    elif args.freeze_only:
        result = freeze_only()
    else:
        result = run_frozen_cache_only()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
