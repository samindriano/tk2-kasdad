"""Source preparation and synthetic lifecycle for REG_IMPACT_MONOTONE_L2_V1.

Official commands require the pinned TP2 Train and an explicit source-pass
authorization before any supervised fit. ``synthetic-smoke`` uses generated
features and labels in a temporary directory only.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from src.models.reg_impact_monotone_l2_v1 import (
    ARM_IDS,
    EXPECTED_LIGHTGBM_VERSION,
    EXPECTED_TRAIN_SHA256,
    ORIGINS,
    PrefixDesignBundle,
    apply_saved_hinge,
    arm_parameters,
    base_lgb_parameters,
    file_sha256,
    fit_arm,
    json_digest,
    json_safe,
    mae,
    prepare_prefix_design,
    rank50_map,
    reconstruct_nested_champion,
    replace_lgb_slot_present_only,
    synthetic_frame,
)
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.regression_research_text import _json_digest
from src.models.optimization_v3_batch1 import exclusive_training_slot
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "reports/optimization_v3/REG_IMPACT_MONOTONE_L2_V1/ROOT_PLAN.md"
AUDIT_FREEZE_PATH = ROOT.parent / "tk2-simplification-audit-v1/reports/simplification_v1/INPUT_FREEZE.json"
SYNTHETIC_FIT_PARAMS = {"n_estimators": 32}
SELECTED_CACHE_KEYS = (
    "reg_r06_67", "reg_direct_67", "reg_r06_67_receipt", "reg_direct_67_receipt",
    "reg_r06_75", "reg_direct_75", "reg_r06_75_receipt", "reg_direct_75_receipt",
    "reg_components67", "reg_components67_receipt", "hinge_D1", "global_D1",
    "hinge_D2", "global_D2",
)
SOURCE_FILES = (
    "src/models/reg_impact_monotone_l2_v1.py",
    "src/models/regression_research_lgb_text.py",
    "src/models/regression_research_text.py",
    "src/models/regression_round2_models.py",
    "src/models/regression_research_tabular.py",
    "src/utils/paths.py",
    "scripts/reg_impact_monotone_l2_v1.py",
    "scripts/verify_reg_impact_monotone_l2_v1.py",
    "experiments/optimization_v3/batch1/R16/freeze.json",
)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class _MemorySampler:
    def __init__(self, interval_seconds: float = 0.1):
        import psutil
        self._process = psutil.Process()
        self._psutil = psutil
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_process_rss = int(self._process.memory_info().rss)
        self.minimum_system_available = int(psutil.virtual_memory().available)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_process_rss = max(
                self.peak_process_rss, int(self._process.memory_info().rss)
            )
            self.minimum_system_available = min(
                self.minimum_system_available,
                int(self._psutil.virtual_memory().available),
            )

    def __enter__(self) -> "_MemorySampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_process_rss = max(
            self.peak_process_rss, int(self._process.memory_info().rss)
        )
        self.minimum_system_available = min(
            self.minimum_system_available,
            int(self._psutil.virtual_memory().available),
        )


def _save_bundle(bundle: PrefixDesignBundle, directory: Path) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"E{bundle.origin}"
    data_path = directory / f"{stem}_design.npz"
    transform_path = directory / f"{stem}_transforms.json"
    receipt_path = directory / f"{stem}_receipt.json"
    _write_npz_exclusive(
        data_path,
        fit_design=bundle.fit_design,
        query_design=bundle.query_design,
        fit_rows=bundle.fit_rows.astype(np.int64),
        query_rows=bundle.query_rows.astype(np.int64),
        fit_ids=bundle.fit_ids.astype(str),
        query_ids=bundle.query_ids.astype(str),
        fit_cycles=bundle.fit_cycles.astype(np.int64),
        query_cycles=bundle.query_cycles.astype(np.int64),
        tfidf_idf=bundle.tfidf_idf.astype(np.float64),
        tfidf_vocabulary_tokens=np.asarray(
            list(bundle.tfidf_vocabulary.keys()), dtype=str
        ),
        tfidf_vocabulary_indices=np.asarray(
            list(bundle.tfidf_vocabulary.values()), dtype=np.int64
        ),
        svd_components=bundle.svd_components.astype(np.float64),
        transformed_feature_names=np.asarray(bundle.feature_names, dtype=str),
        categorical_indices=np.asarray(bundle.categorical_indices, dtype=np.int64),
    )
    transform_payload = {
        "schema_version": 1,
        "origin": bundle.origin,
        "source_kind": bundle.preprocessing_state["source_kind"],
        "source_train_sha256": bundle.preprocessing_state["source_train_sha256"],
        "structured_preprocessing": bundle.preprocessing_state["structured_preprocessing"],
        "tfidf_state": bundle.preprocessing_state["tfidf"],
        "tfidf_vocabulary": bundle.tfidf_vocabulary,
        "tfidf_idf_sha256": bundle.preprocessing_state["tfidf_idf_sha256"],
        "svd_state": bundle.preprocessing_state["svd"],
        "transformed_feature_names": list(bundle.feature_names),
        "categorical_feature_indices": list(bundle.categorical_indices),
        "impact_load_value_index": bundle.impact_index,
    }
    _write_json_exclusive(transform_path, transform_payload)
    receipt = {
        "schema_version": 1,
        "bundle_id": f"REG_IMPACT_MONOTONE_L2_V1_E{bundle.origin}",
        "origin": bundle.origin,
        "source_kind": bundle.preprocessing_state["source_kind"],
        "source_train_sha256": bundle.preprocessing_state["source_train_sha256"],
        "design_path": str(data_path.resolve()),
        "design_sha256": file_sha256(str(data_path)),
        "transforms_path": str(transform_path.resolve()),
        "transforms_sha256": file_sha256(str(transform_path)),
        "preprocessing_state": bundle.preprocessing_state,
        "transform_paths": {
            "structured": "embedded in transforms JSON and design matrix",
            "tfidf_vocabulary_and_idf": "transforms JSON plus design NPZ",
            "svd32_components": "design NPZ",
        },
        "learned_transform_count": 3,
        "target_values_saved": False,
        "test_data_opened": False,
        "sample_submission_opened": False,
    }
    _write_json_exclusive(receipt_path, receipt)
    return data_path, transform_path, receipt_path


def _load_bundle(receipt_path: Path) -> PrefixDesignBundle:
    receipt = _read_json(receipt_path)
    data_path = Path(receipt["design_path"])
    transform_path = Path(receipt["transforms_path"])
    if file_sha256(str(data_path)) != receipt["design_sha256"]:
        raise ValueError("prefix design bundle hash differs from its receipt")
    if file_sha256(str(transform_path)) != receipt["transforms_sha256"]:
        raise ValueError("prefix transform state hash differs from its receipt")
    transforms = _read_json(transform_path)
    with np.load(data_path, allow_pickle=False) as z:
        return PrefixDesignBundle(
            origin=int(receipt["origin"]),
            fit_design=np.asarray(z["fit_design"], dtype=np.float64),
            query_design=np.asarray(z["query_design"], dtype=np.float64),
            fit_rows=np.asarray(z["fit_rows"], dtype=np.int64),
            query_rows=np.asarray(z["query_rows"], dtype=np.int64),
            fit_ids=np.asarray(z["fit_ids"]).astype(str),
            query_ids=np.asarray(z["query_ids"]).astype(str),
            fit_cycles=np.asarray(z["fit_cycles"], dtype=np.int64),
            query_cycles=np.asarray(z["query_cycles"], dtype=np.int64),
            feature_names=tuple(str(x) for x in z["transformed_feature_names"].tolist()),
            categorical_indices=tuple(int(x) for x in z["categorical_indices"].tolist()),
            impact_index=int(transforms["impact_load_value_index"]),
            preprocessing_state=dict(receipt["preprocessing_state"]),
            tfidf_vocabulary={str(k): int(v) for k, v in transforms["tfidf_vocabulary"].items()},
            tfidf_idf=np.asarray(z["tfidf_idf"], dtype=np.float64),
            svd_components=np.asarray(z["svd_components"], dtype=np.float64),
        )


def _save_fit_artifacts(
    bundle: PrefixDesignBundle,
    model: lightgbm.LGBMRegressor,
    raw: np.ndarray,
    meta: dict[str, Any],
    output_root: Path,
    *,
    synthetic: bool,
    receipt_metadata: dict[str, Any] | None = None,
) -> Path:
    fit_id = f"E{bundle.origin}_{meta['arm_id']}"
    directory = output_root / fit_id
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.txt"
    model.booster_.save_model(str(model_path))
    prediction_path = directory / "predictions.npz"
    mapped = rank50_map(raw, bundle.query_cycles)
    _write_npz_exclusive(
        prediction_path,
        fit_id=np.asarray([fit_id], dtype=str),
        query_ids=bundle.query_ids.astype(str),
        query_rows=bundle.query_rows.astype(np.int64),
        query_cycles=bundle.query_cycles.astype(np.int64),
        raw_prediction=np.asarray(raw, dtype=np.float64),
        mapped_prediction=mapped,
    )
    receipt = {
        "schema_version": 1,
        "fit_id": fit_id,
        "origin": bundle.origin,
        "arm_id": meta["arm_id"],
        "source_kind": bundle.preprocessing_state["source_kind"],
        "lightgbm_version": lightgbm.__version__,
        "model_path": str(model_path.resolve()),
        "model_sha256": file_sha256(str(model_path)),
        "predictions_path": str(prediction_path.resolve()),
        "predictions_sha256": file_sha256(str(prediction_path)),
        "bundle_sha256": bundle.preprocessing_state["fit_design_sha256"],
        "meta": meta,
        "synthetic_only": bool(synthetic),
        "validation_data_passed_to_fit": False,
        "test_data_used_for_fit": False,
    }
    receipt.update(receipt_metadata or {})
    _write_json_exclusive(directory / "receipt.json", receipt)
    return directory


def _resolve_train(data_dir: Path) -> Path:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"TP2 data root does not exist: {data_dir}")
    files = task_files("regression", root=data_dir)
    path = files.get("train")
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Official TP2 Regression Train is missing; set --data-dir to the directory "
            "containing regression/Train.csv. TP1 fallback is prohibited."
        )
    if file_sha256(str(path)) != EXPECTED_TRAIN_SHA256:
        raise ValueError("resolved official TP2 Train hash differs from pinned R01/R16 SHA-256")
    return Path(path)


def _load_official_features(data_dir: Path) -> tuple[Path, pd.DataFrame, str]:
    train_path = _resolve_train(data_dir)
    before = file_sha256(str(train_path))
    required = ["record_id", *HYBRID_FEATURES]
    frame = pd.read_csv(train_path, usecols=required)
    if list(frame.columns) != required and set(frame.columns) != set(required):
        raise ValueError("official Train feature allowlist differs from frozen source schema")
    if len(frame) != 50_608:
        raise ValueError(f"official Train row count differs from pinned 50,608: {len(frame)}")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise ValueError("official Train cycle coverage differs from 47..79")
    after = file_sha256(str(train_path))
    if before != after or after != EXPECTED_TRAIN_SHA256:
        raise ValueError("official Train changed while reading its target-free feature columns")
    return train_path, frame, after


def _cache_freeze(input_freeze_path: Path) -> dict[str, Any]:
    freeze = _read_json(input_freeze_path)
    files = freeze.get("files")
    if not isinstance(files, dict):
        raise ValueError("simplification audit INPUT_FREEZE lacks the expected files map")
    selected: dict[str, Any] = {}
    for key in SELECTED_CACHE_KEYS:
        item = files.get(key)
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"frozen non-target cache entry is missing: {key}")
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen non-target cache is missing: {key} ({path})")
        digest = file_sha256(str(path))
        if digest != item["sha256"]:
            raise ValueError(f"frozen non-target cache SHA-256 differs: {key}")
        selected[key] = {"path": str(path.resolve()), "sha256": digest}
    return {
        "input_freeze_path": str(input_freeze_path.resolve()),
        "input_freeze_sha256": file_sha256(str(input_freeze_path)),
        "selected_non_target_cache_files": selected,
        "target_label_cache_entries_opened": False,
    }


def cmd_freeze_source(args: argparse.Namespace) -> int:
    if lightgbm.__version__ != EXPECTED_LIGHTGBM_VERSION:
        raise RuntimeError(f"expected LightGBM {EXPECTED_LIGHTGBM_VERSION}, got {lightgbm.__version__}")
    train_path = _resolve_train(Path(args.data_dir))
    synthetic_report_path = Path(args.synthetic_report)
    synthetic_report = _read_json(synthetic_report_path)
    if synthetic_report.get("status") != "PASS":
        raise RuntimeError("synthetic lifecycle must PASS before official preprocessing is frozen")
    cache_inventory = _cache_freeze(Path(args.input_freeze))
    source_hashes = {
        name: file_sha256(str(ROOT / name))
        for name in SOURCE_FILES
        if (ROOT / name).is_file()
    }
    absent_sources = sorted(set(SOURCE_FILES) - set(source_hashes))
    if absent_sources:
        raise FileNotFoundError(f"pinned source files missing: {absent_sources}")
    train_hash = file_sha256(str(train_path))
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("TP2 Train SHA-256 changed before source freeze")
    r16_freeze_path = ROOT / "experiments/optimization_v3/batch1/R16/freeze.json"
    r16_freeze = _read_json(r16_freeze_path)
    r16_state_path = "identity/control_lineage/component_audits/0/state"
    r16_state = r16_freeze["identity"]["control_lineage"]["component_audits"][0]["state"]
    source_params = base_lgb_parameters()
    if (r16_state.get("id") != "LIGHTGBM_SVD32_BASE"
            or r16_state.get("lightgbm_version") != EXPECTED_LIGHTGBM_VERSION
            or r16_state.get("actual_hyperparameters") != source_params):
        raise RuntimeError("unweighted LGB SVD32 parameters do not match the saved R16 incumbent state")
    state_payload = {key: value for key, value in r16_state.items() if key != "state_sha256"}
    if r16_state.get("state_sha256") != _json_digest(state_payload):
        raise RuntimeError("saved R16 LGB state digest does not independently recompute")
    payload = {
        "schema_version": 1,
        "experiment": "REG_IMPACT_MONOTONE_L2_V1",
        "status": "SOURCE_RECIPE_FROZEN_PREPROCESSING_NOT_STARTED",
        "root_plan_path": str(PLAN_PATH.resolve()),
        "root_plan_sha256": file_sha256(str(PLAN_PATH)),
        "synthetic_report_path": str(synthetic_report_path.resolve()),
        "synthetic_report_sha256": file_sha256(str(synthetic_report_path)),
        "train_path": str(train_path.resolve()),
        "train_sha256": train_hash,
        "train_bytes": int(train_path.stat().st_size),
        "source_hashes": source_hashes,
        "cache_inventory": cache_inventory,
        "lightgbm_version": lightgbm.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "arm_ids": list(ARM_IDS),
        "planned_fit_ids": [f"E{o}_{arm}" for o in ORIGINS for arm in ARM_IDS],
        "planned_supervised_fit_count": 6,
        "official_supervised_fits_started": False,
        "prefixes": {
            str(origin): {
                "fit_rule": f"47 <= cycle_code <= {origin}",
                "query_rule": f"{origin} < cycle_code <= 79",
                "learned_transform_count": 3,
                "learned_transforms": ["_SelectedPreprocessor", "TF-IDF", "TruncatedSVD32"],
            }
            for origin in ORIGINS
        },
        "transform_recipe": {
            "structured_features": "STRUCTURED_FEATURES; _SelectedPreprocessor fitted on prefix rows only",
            "text_features": "signal_stream; RegressionTextModel._fit_vectorizer fitted on prefix rows only",
            "tfidf_parameters": {
                "dtype": "float64", "max_features": 20000, "min_df": 2,
                "ngram_range": [1, 2], "sublinear_tf": True,
            },
            "svd_parameters": {"algorithm": "randomized", "n_components": 32, "n_iter": 7, "random_state": 42},
            "missing_stream_indicator": True,
            "source_target_column_read": False,
            "test_or_sample_opened": False,
        },
        "original_lgb_parameters": {
            "values": source_params,
            "sha256": json_digest(source_params),
            "frozen_state_id": r16_state["id"],
            "frozen_state_path": r16_state_path,
            "frozen_state_sha256": r16_state["state_sha256"],
        },
        "resource_policy": {
            "threads": 2,
            "concurrent_heavy_jobs": 1,
            "initial_ram_budget_bytes": 1024 ** 3,
            "free_ram_reserve_bytes": 512 * 1024 ** 2,
            "shared_heavy_job_lock_required": True,
        },
    }
    _write_json_exclusive(Path(args.output), payload)
    print(json.dumps({"status": payload["status"], "output": str(Path(args.output).resolve()),
                      "train_sha256": train_hash, "selected_cache_count": len(cache_inventory["selected_non_target_cache_files"])},
                     sort_keys=True))
    return 0


def _check_source_freeze(path: Path, data_dir: Path) -> tuple[dict[str, Any], Path]:
    freeze = _read_json(path)
    if freeze.get("status") != "SOURCE_RECIPE_FROZEN_PREPROCESSING_NOT_STARTED":
        raise ValueError("source freeze status is not ready for target-free preprocessing")
    if freeze.get("root_plan_sha256") != file_sha256(str(PLAN_PATH)):
        raise ValueError("root plan changed after source freeze")
    synthetic_report = Path(freeze["synthetic_report_path"])
    if file_sha256(str(synthetic_report)) != freeze["synthetic_report_sha256"]:
        raise ValueError("synthetic lifecycle report changed after source freeze")
    for name, expected in freeze["source_hashes"].items():
        if file_sha256(str(ROOT / name)) != expected:
            raise ValueError(f"pinned implementation source changed after freeze: {name}")
    cache_inventory = _cache_freeze(Path(freeze["cache_inventory"]["input_freeze_path"]))
    if cache_inventory != freeze["cache_inventory"]:
        raise ValueError("frozen non-target cache inventory changed")
    train_path = _resolve_train(data_dir)
    if str(train_path.resolve()) != freeze["train_path"]:
        raise ValueError("resolved official Train path differs from the source freeze")
    if file_sha256(str(train_path)) != freeze["train_sha256"]:
        raise ValueError("official Train SHA-256 differs from the source freeze")
    return freeze, train_path


def _fit_authorization(
    authorization_path: Path,
    source_freeze_path: Path,
    bundle_receipt_path: Path,
    fit_id: str,
) -> dict[str, Any]:
    authorization = _read_json(authorization_path)
    source_freeze = _read_json(source_freeze_path)
    bundle_receipt = _read_json(bundle_receipt_path)
    expected_ids = [f"E{o}_{arm}" for o in ORIGINS for arm in ARM_IDS]
    required = {
        "status": "APPROVED_FOR_SUPERVISED_FITS",
        "experiment": "REG_IMPACT_MONOTONE_L2_V1",
        "fit_count": 6,
        "approved_fit_ids": expected_ids,
        "root_plan_sha256": file_sha256(str(PLAN_PATH)),
        "source_freeze_sha256": file_sha256(str(source_freeze_path)),
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "lightgbm_version": EXPECTED_LIGHTGBM_VERSION,
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise RuntimeError(f"fit authorization has no exact PASS for {key}")
    if fit_id not in authorization["approved_fit_ids"]:
        raise RuntimeError(f"fit id is not authorized: {fit_id}")
    expected_bundle_hashes = authorization.get("bundle_design_sha256")
    if not isinstance(expected_bundle_hashes, dict):
        raise RuntimeError("fit authorization lacks bundle_design_sha256")
    if expected_bundle_hashes.get(str(bundle_receipt["origin"])) != bundle_receipt["design_sha256"]:
        raise RuntimeError("fit authorization is not bound to this prefix design bundle")
    if source_freeze.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("source freeze does not pin the official TP2 Train")
    return {
        "authorization_sha256": file_sha256(str(authorization_path)),
        "source_freeze_sha256": file_sha256(str(source_freeze_path)),
        "bundle_design_sha256": bundle_receipt["design_sha256"],
        "authorization_path": str(Path(authorization_path).resolve()),
        "authorization": authorization,
    }


def _fit_lifecycle_paths(output_root: Path) -> tuple[Path, Path]:
    lifecycle_root = output_root / ".fit_lifecycle"
    return lifecycle_root / "events.jsonl", lifecycle_root / "claims"


def _append_fit_lifecycle_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_safe(event), sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_fit_lifecycle(
    ledger_path: Path,
    approved_fit_ids: list[str],
    bindings_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]], list[str]]:
    canonical_ids = [f"E{origin}_{arm}" for origin in ORIGINS for arm in ARM_IDS]
    if approved_fit_ids != canonical_ids or len(approved_fit_ids) != 6:
        raise RuntimeError("fit lifecycle requires the exact six approved fit IDs in canonical order")
    if set(bindings_by_id) != set(approved_fit_ids):
        raise RuntimeError("fit lifecycle bindings must cover exactly the six approved fit IDs")
    for fit_id in approved_fit_ids:
        if bindings_by_id[fit_id].get("fit_id") != fit_id:
            raise RuntimeError(f"fit lifecycle binding identity mismatch: {fit_id}")

    events: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    terminal_events: dict[str, dict[str, Any]] = {}
    started_ids: list[str] = []
    if ledger_path.is_file():
        with ledger_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise RuntimeError(f"fit lifecycle ledger contains a blank row at line {line_number}")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"fit lifecycle ledger is incomplete at line {line_number}") from exc
                if not isinstance(event, dict) or event.get("schema_version") != 1:
                    raise RuntimeError(f"fit lifecycle event schema mismatch at line {line_number}")
                if event.get("event_seq") != line_number or event.get("event_seq") != len(events) + 1:
                    raise RuntimeError(f"fit lifecycle event sequence mismatch at line {line_number}")
                fit_id = event.get("fit_id")
                if fit_id not in approved_fit_ids:
                    raise RuntimeError(f"fit lifecycle contains an unauthorized fit ID: {fit_id}")
                expected_binding_sha = json_digest(bindings_by_id[fit_id])
                if event.get("binding_sha256") != expected_binding_sha:
                    raise RuntimeError(f"fit lifecycle binding changed for {fit_id}")
                event_status = event.get("status")
                if event_status == "FIT_STARTED":
                    if states.get(fit_id) is not None:
                        raise RuntimeError(f"duplicate fit lifecycle start for {fit_id}")
                    next_index = len(started_ids)
                    if next_index >= len(approved_fit_ids) or fit_id != approved_fit_ids[next_index]:
                        raise RuntimeError("fit lifecycle starts are not in canonical approved order")
                    started_ids.append(fit_id)
                    states[fit_id] = "PENDING"
                elif event_status in {"FIT_SAVED", "FIT_FAILED"}:
                    if states.get(fit_id) != "PENDING":
                        raise RuntimeError(f"fit lifecycle terminal event has no unique start: {fit_id}")
                    states[fit_id] = event_status
                    terminal_events[fit_id] = event
                else:
                    raise RuntimeError(f"unknown fit lifecycle status at line {line_number}")
                events.append(event)

    return events, states, terminal_events, started_ids


def run_fit_lifecycle(
    *,
    output_root: Path,
    fit_id: str,
    approved_fit_ids: list[str],
    bindings_by_id: dict[str, dict[str, Any]],
    prepare_fn: Callable[[], Any],
    fit_and_save_fn: Callable[[Any], Path],
    readback_fn: Callable[[str, Path, dict[str, Any]], dict[str, Any]],
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run one fit through the durable, fail-closed lifecycle shared by CLI and smoke QA."""
    if fit_id not in approved_fit_ids:
        raise RuntimeError(f"fit id is not in the approved lifecycle: {fit_id}")
    output_root = Path(output_root).resolve()
    ledger_path, claims_root = _fit_lifecycle_paths(output_root)
    shared_lock = lock_path or (Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock")
    with exclusive_training_slot(shared_lock):
        events, states, terminal_events, started_ids = _read_fit_lifecycle(
            ledger_path, approved_fit_ids, bindings_by_id
        )
        for approved_id in approved_fit_ids:
            claim_path = claims_root / f"{approved_id}.json"
            artifact_dir = output_root / approved_id
            state = states.get(approved_id)
            if state is None and (claim_path.exists() or artifact_dir.exists()):
                raise RuntimeError(f"orphan fit claim/artifacts found for {approved_id}")
            if state is not None and not claim_path.is_file():
                raise RuntimeError(f"fit lifecycle claim is missing for consumed ID {approved_id}")
            if state is not None:
                claim = _read_json(claim_path)
                if (claim.get("fit_id") != approved_id
                        or claim.get("binding_sha256") != json_digest(bindings_by_id[approved_id])):
                    raise RuntimeError(f"fit lifecycle claim binding differs for {approved_id}")
            if state == "FIT_SAVED" and not artifact_dir.is_dir():
                raise RuntimeError(f"saved fit artifacts are missing for {approved_id}")

        if any(state == "PENDING" for state in states.values()):
            pending = [fit for fit, state in states.items() if state == "PENDING"]
            raise RuntimeError(f"fit lifecycle has interrupted/pending IDs with no retry allowed: {pending}")
        if any(state == "FIT_FAILED" for state in states.values()):
            failed = [fit for fit, state in states.items() if state == "FIT_FAILED"]
            raise RuntimeError(f"fit lifecycle has consumed failed IDs; no retry or later fit allowed: {failed}")

        saved_readbacks: dict[str, dict[str, Any]] = {}
        for saved_id in approved_fit_ids:
            if states.get(saved_id) != "FIT_SAVED":
                continue
            receipt_path = output_root / saved_id / "receipt.json"
            event = terminal_events[saved_id]
            readback = readback_fn(saved_id, receipt_path, bindings_by_id[saved_id])
            if event.get("receipt_path") != str(receipt_path.resolve()):
                raise RuntimeError(f"saved fit ledger receipt path differs for {saved_id}")
            if event.get("artifacts") != readback.get("artifacts"):
                raise RuntimeError(f"saved fit ledger artifact hashes/readback differ for {saved_id}")
            saved_readbacks[saved_id] = readback

        if states.get(fit_id) == "FIT_SAVED":
            return {"status": "READBACK", "fit_id": fit_id, **saved_readbacks[fit_id]}
        if states.get(fit_id) is not None:
            raise RuntimeError(f"fit ID {fit_id} is already consumed; no retry is allowed")

        next_index = len(started_ids)
        if next_index >= 6:
            raise RuntimeError("the approved six-fit lifecycle cap has been reached")
        if fit_id != approved_fit_ids[next_index]:
            raise RuntimeError(
                f"fit order violation: next approved ID is {approved_fit_ids[next_index]}, got {fit_id}"
            )

        prepared = prepare_fn()
        output_root.mkdir(parents=True, exist_ok=True)
        claims_root.mkdir(parents=True, exist_ok=True)
        claim_path = claims_root / f"{fit_id}.json"
        if claim_path.exists() or (output_root / fit_id).exists():
            raise RuntimeError(f"fit ID {fit_id} became claimed/orphaned before start")
        binding_sha = json_digest(bindings_by_id[fit_id])
        start_payload = {
            "schema_version": 1,
            "fit_id": fit_id,
            "binding_sha256": binding_sha,
            "pid": os.getpid(),
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with claim_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(start_payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        started = {
            "schema_version": 1,
            "event_seq": len(events) + 1,
            "event_time_utc": datetime.now(timezone.utc).isoformat(),
            "fit_id": fit_id,
            "status": "FIT_STARTED",
            "binding_sha256": binding_sha,
            "claim_path": str(claim_path.resolve()),
        }
        _append_fit_lifecycle_event(ledger_path, started)
        events.append(started)

        try:
            receipt_path = Path(fit_and_save_fn(prepared)).resolve()
            expected_receipt = (output_root / fit_id / "receipt.json").resolve()
            if receipt_path != expected_receipt:
                raise RuntimeError(f"fit saver returned an unexpected receipt path for {fit_id}")
            readback = readback_fn(fit_id, receipt_path, bindings_by_id[fit_id])
        except BaseException as exc:
            failed = {
                "schema_version": 1,
                "event_seq": len(events) + 1,
                "event_time_utc": datetime.now(timezone.utc).isoformat(),
                "fit_id": fit_id,
                "status": "FIT_FAILED",
                "binding_sha256": binding_sha,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
            _append_fit_lifecycle_event(ledger_path, failed)
            raise RuntimeError(f"fit ID {fit_id} failed and is now consumed; no retry is allowed") from exc

        saved = {
            "schema_version": 1,
            "event_seq": len(events) + 1,
            "event_time_utc": datetime.now(timezone.utc).isoformat(),
            "fit_id": fit_id,
            "status": "FIT_SAVED",
            "binding_sha256": binding_sha,
            "receipt_path": str(receipt_path),
            "artifacts": readback["artifacts"],
            "model_readback_max_abs_diff": readback["model_readback_max_abs_diff"],
            "mapped_prediction_max_abs_diff": readback["mapped_prediction_max_abs_diff"],
        }
        _append_fit_lifecycle_event(ledger_path, saved)
        return {"status": "FITTED", "fit_id": fit_id, **readback}


def _readback_fit_artifacts(
    fit_id: str,
    receipt_path: Path,
    bundle_receipt_path: Path,
    bundle: PrefixDesignBundle,
    binding: dict[str, Any],
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    expected_root = receipt_path.parent.resolve()
    if receipt.get("fit_id") != fit_id or receipt.get("source_kind") != binding["source_kind"]:
        raise RuntimeError(f"{fit_id} saved receipt identity/source kind mismatch")
    if receipt.get("lifecycle_binding_sha256") != json_digest(binding):
        raise RuntimeError(f"{fit_id} saved receipt has a different lifecycle binding")
    if (receipt.get("source_freeze_sha256") != binding["source_freeze_sha256"]
            or receipt.get("root_plan_sha256") != binding["root_plan_sha256"]
            or receipt.get("bundle_design_sha256") != binding["bundle_design_sha256"]
            or receipt.get("bundle_receipt_sha256") != binding["bundle_receipt_sha256"]):
        raise RuntimeError(f"{fit_id} saved receipt is detached from its frozen source/bundle")
    if receipt.get("bundle_sha256") != bundle.preprocessing_state["fit_design_sha256"]:
        raise RuntimeError(f"{fit_id} saved receipt fit-design digest differs from the prefix bundle")
    if file_sha256(str(bundle_receipt_path)) != binding["bundle_receipt_sha256"]:
        raise RuntimeError(f"{fit_id} prefix bundle receipt changed after lifecycle binding")
    if receipt.get("model_path") != str((expected_root / "model.txt").resolve()):
        raise RuntimeError(f"{fit_id} saved model path differs from its exclusive output directory")
    if receipt.get("predictions_path") != str((expected_root / "predictions.npz").resolve()):
        raise RuntimeError(f"{fit_id} saved prediction path differs from its exclusive output directory")
    model_path = Path(receipt["model_path"])
    prediction_path = Path(receipt["predictions_path"])
    if (file_sha256(str(model_path)) != receipt.get("model_sha256")
            or file_sha256(str(prediction_path)) != receipt.get("predictions_sha256")):
        raise RuntimeError(f"{fit_id} saved model/prediction hash mismatch")
    if binding["source_kind"] == "official":
        authorization_path = Path(binding["authorization_path"])
        if (receipt.get("official_fit_authorization_path") != str(authorization_path.resolve())
                or receipt.get("official_fit_authorization_sha256") != binding["authorization_sha256"]
                or file_sha256(str(authorization_path)) != binding["authorization_sha256"]):
            raise RuntimeError(f"{fit_id} saved receipt is detached from its approved authorization")

    with np.load(prediction_path, allow_pickle=False) as saved:
        if "fit_id" not in saved.files or np.asarray(saved["fit_id"]).astype(str).tolist() != [fit_id]:
            raise RuntimeError(f"{fit_id} saved predictions lack the exact fit identity")
        query_ids = np.asarray(saved["query_ids"]).astype(str)
        query_rows = np.asarray(saved["query_rows"], dtype=np.int64)
        query_cycles = np.asarray(saved["query_cycles"], dtype=np.int64)
        raw = np.asarray(saved["raw_prediction"], dtype=np.float64)
        mapped = np.asarray(saved["mapped_prediction"], dtype=np.float64)
    if (not np.array_equal(query_ids, bundle.query_ids)
            or not np.array_equal(query_rows, bundle.query_rows)
            or not np.array_equal(query_cycles, bundle.query_cycles)):
        raise RuntimeError(f"{fit_id} saved query identity/order differs from its frozen bundle")
    if raw.shape != bundle.query_cycles.shape or mapped.shape != raw.shape or not np.isfinite(raw).all():
        raise RuntimeError(f"{fit_id} saved predictions are misaligned or nonfinite")
    model = lightgbm.Booster(model_file=str(model_path))
    query_frame = pd.DataFrame(bundle.query_design, columns=list(bundle.feature_names))
    readback = np.asarray(model.predict(query_frame), dtype=np.float64)
    model_difference = float(np.max(np.abs(readback - raw))) if len(raw) else 0.0
    if not np.allclose(readback, raw, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{fit_id} saved model does not read back to its raw predictions")
    expected_mapped = rank50_map(raw, bundle.query_cycles)
    mapped_difference = float(np.max(np.abs(expected_mapped - mapped))) if len(raw) else 0.0
    if not np.allclose(expected_mapped, mapped, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{fit_id} saved rank-map output differs from the frozen mapping")
    meta = receipt.get("meta", {})
    expected_arm = fit_id[len(f"E{bundle.origin}_"):]
    if meta.get("arm_id") != expected_arm or int(meta.get("origin", -1)) != bundle.origin:
        raise RuntimeError(f"{fit_id} saved fit metadata identity differs")
    return {
        "model_readback_max_abs_diff": model_difference,
        "mapped_prediction_max_abs_diff": mapped_difference,
        "artifacts": {
            "receipt_sha256": file_sha256(str(receipt_path)),
            "model_sha256": file_sha256(str(model_path)),
            "predictions_sha256": file_sha256(str(prediction_path)),
        },
    }


def _read_prefix_target(train_path: Path, bundle: PrefixDesignBundle) -> np.ndarray:
    """Materialize threat_rank only for rows inside this learner's prefix."""
    before = file_sha256(str(train_path))
    values: list[float] = []
    rows: list[int] = []
    ids: list[str] = []
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        required = {"record_id", "cycle_code", "threat_rank"}
        if not required.issubset(set(header)):
            raise ValueError("official Train lacks record_id/cycle_code/threat_rank")
        id_index = header.index("record_id")
        cycle_index = header.index("cycle_code")
        target_index = header.index("threat_rank")
        for row_number, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"official Train row {row_number + 2} has an invalid column count")
            cycle_float = float(row[cycle_index])
            if not np.isfinite(cycle_float) or cycle_float != int(cycle_float):
                raise ValueError(f"official Train row {row_number + 2} has invalid cycle_code")
            if int(cycle_float) <= bundle.origin:
                # Future/query labels remain unopened by this branch.
                target = float(row[target_index])
                if not np.isfinite(target) or target < 0.0 or target > 100.0:
                    raise ValueError(f"prefix threat_rank is invalid at source row {row_number}")
                rows.append(row_number)
                ids.append(row[id_index])
                values.append(target)
    after = file_sha256(str(train_path))
    if before != after or before != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("official Train hash changed while reading prefix labels")
    if not np.array_equal(np.asarray(rows, dtype=np.int64), bundle.fit_rows):
        raise RuntimeError("materialized fit target rows differ from prefix design rows")
    if not np.array_equal(np.asarray(ids, dtype=str), bundle.fit_ids):
        raise RuntimeError("materialized fit target identities differ from prefix design identities")
    return np.asarray(values, dtype=np.float64)


def cmd_fit_arm(args: argparse.Namespace) -> int:
    bundle_receipt_path = Path(args.bundle_receipt)
    bundle = _load_bundle(bundle_receipt_path)
    fit_id = f"E{bundle.origin}_{args.arm}"
    source_freeze, train_path = _check_source_freeze(
        Path(args.source_freeze), Path(args.data_dir)
    )
    if source_freeze.get("source_freeze_revision") != 2:
        raise RuntimeError("official fits require the additive R2 source freeze with lifecycle guard proof")
    lifecycle_binding = source_freeze.get("fit_lifecycle_guard", {})
    lifecycle_report_path = Path(lifecycle_binding.get("report_path", ""))
    if (lifecycle_binding.get("status") != "PASS"
            or not lifecycle_report_path.is_file()
            or file_sha256(str(lifecycle_report_path)) != lifecycle_binding.get("report_sha256")):
        raise RuntimeError("R2 source freeze is missing a current passing lifecycle smoke report")
    mapping_binding = source_freeze.get("mapping_binding", {})
    mapping_proof_path = Path(mapping_binding.get("report_path", ""))
    if (mapping_binding.get("status") != "PASS"
            or mapping_binding.get("lgb_rank_weight") != 0.5
            or not mapping_proof_path.is_file()
            or file_sha256(str(mapping_proof_path)) != mapping_binding.get("report_sha256")):
        raise RuntimeError("R2 source freeze is missing the exact frozen LGB rank-mapping proof")
    gate = _fit_authorization(
        Path(args.authorization), Path(args.source_freeze), bundle_receipt_path, fit_id
    )
    authorization = gate["authorization"]
    approved_fit_ids = list(authorization["approved_fit_ids"])
    bundle_receipts: dict[int, tuple[Path, dict[str, Any]]] = {}
    bundles: dict[int, PrefixDesignBundle] = {bundle.origin: bundle}
    bindings_by_id: dict[str, dict[str, Any]] = {}
    for origin in ORIGINS:
        receipt_path = bundle_receipt_path if origin == bundle.origin else bundle_receipt_path.parent / f"E{origin}_receipt.json"
        receipt = _read_json(receipt_path)
        if (receipt.get("source_kind") != "official"
                or receipt.get("source_train_sha256") != EXPECTED_TRAIN_SHA256
                or receipt.get("design_sha256") != authorization["bundle_design_sha256"].get(str(origin))):
            raise RuntimeError(f"E{origin} bundle receipt does not match exact approval bindings")
        if file_sha256(str(Path(receipt["design_path"]))) != receipt["design_sha256"]:
            raise RuntimeError(f"E{origin} design bundle digest changed before fit lifecycle")
        if file_sha256(str(Path(receipt["transforms_path"]))) != receipt["transforms_sha256"]:
            raise RuntimeError(f"E{origin} transform bundle digest changed before fit lifecycle")
        preserved = source_freeze.get("preserved_prefix_bundles", {}).get(str(origin), {})
        if (preserved.get("design_sha256") != receipt["design_sha256"]
                or preserved.get("transforms_sha256") != receipt["transforms_sha256"]
                or preserved.get("receipt_sha256") != file_sha256(str(receipt_path))):
            raise RuntimeError(f"E{origin} existing R1 prefix bundle changed after the R2 source amendment")
        bundle_receipts[origin] = (receipt_path, receipt)
        for arm in ARM_IDS:
            planned_id = f"E{origin}_{arm}"
            bindings_by_id[planned_id] = {
                "fit_id": planned_id,
                "source_kind": "official",
                "authorization_path": gate["authorization_path"],
                "authorization_sha256": gate["authorization_sha256"],
                "source_freeze_sha256": gate["source_freeze_sha256"],
                "root_plan_sha256": source_freeze["root_plan_sha256"],
                "bundle_design_sha256": receipt["design_sha256"],
                "bundle_receipt_sha256": file_sha256(str(receipt_path)),
                "train_sha256": EXPECTED_TRAIN_SHA256,
            }

    def readback(saved_id: str, receipt_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
        origin = int(saved_id.split("_", 1)[0][1:])
        saved_receipt_path, _saved_receipt = bundle_receipts[origin]
        if origin not in bundles:
            bundles[origin] = _load_bundle(saved_receipt_path)
        return _readback_fit_artifacts(
            saved_id, receipt_path, saved_receipt_path, bundles[origin], binding
        )

    def prepare_training_target() -> np.ndarray:
        import psutil
        available = int(psutil.virtual_memory().available)
        if available < 1024 ** 3:
            raise MemoryError(f"fit RAM preflight failed: {available} available; need 1 GiB initial free RAM")
        return _read_prefix_target(train_path, bundle)

    def fit_and_save(target: np.ndarray) -> Path:
        model, raw, meta = fit_arm(bundle, target, args.arm)
        meta["official_fit_authorization_sha256"] = gate["authorization_sha256"]
        meta["source_freeze_sha256"] = gate["source_freeze_sha256"]
        meta["target_rows_materialized"] = len(target)
        meta["query_target_rows_materialized"] = 0
        output_root = Path(args.output_root)
        directory = _save_fit_artifacts(
            bundle,
            model,
            raw,
            meta,
            output_root,
            synthetic=False,
            receipt_metadata={
                "official_fit_authorization_sha256": gate["authorization_sha256"],
                "official_fit_authorization_path": str(Path(args.authorization).resolve()),
                "source_freeze_sha256": gate["source_freeze_sha256"],
                "root_plan_sha256": source_freeze["root_plan_sha256"],
                "bundle_design_sha256": bindings_by_id[fit_id]["bundle_design_sha256"],
                "bundle_receipt_sha256": bindings_by_id[fit_id]["bundle_receipt_sha256"],
                "lifecycle_binding_sha256": json_digest(bindings_by_id[fit_id]),
                "target_rows_materialized": len(target),
                "query_target_rows_materialized": 0,
            },
        )
        return directory / "receipt.json"

    result = run_fit_lifecycle(
        output_root=Path(args.output_root),
        fit_id=fit_id,
        approved_fit_ids=approved_fit_ids,
        bindings_by_id=bindings_by_id,
        prepare_fn=prepare_training_target,
        fit_and_save_fn=fit_and_save,
        readback_fn=readback,
    )
    receipt_path = Path(args.output_root).resolve() / fit_id / "receipt.json"
    print(json.dumps({"fit_id": fit_id, "status": result["status"],
                      "receipt": str(receipt_path.resolve()),
                      "model_readback_max_abs_diff": result.get("model_readback_max_abs_diff"),
                      "mapped_prediction_max_abs_diff": result.get("mapped_prediction_max_abs_diff"),
                      "artifacts": result.get("artifacts")}, sort_keys=True))
    return 0


def cmd_prelabel_qa(args: argparse.Namespace) -> int:
    command = [
        sys.executable, "-m", "scripts.verify_reg_impact_monotone_l2_v1", "official",
        "--fit-root", args.fit_root,
        "--bundle-root", args.bundle_root,
        "--ensemble-root", args.ensemble_root,
        "--input-freeze", args.input_freeze,
        "--source-freeze", args.source_freeze,
        "--output", args.output,
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"independent official prelabel QA failed: {result.stderr or result.stdout}")
    print(result.stdout.strip())
    return 0


def _query_labels_after_qa(train_path: Path, required: dict[str, int]) -> dict[str, float]:
    before = file_sha256(str(train_path))
    labels: dict[str, float] = {}
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_index = header.index("record_id")
        cycle_index = header.index("cycle_code")
        target_index = header.index("threat_rank")
        for row_number, row in enumerate(reader):
            record_id = row[id_index]
            cycle_float = float(row[cycle_index])
            cycle = int(cycle_float)
            expected_cycle = required.get(record_id)
            if expected_cycle is not None:
                if cycle_float != cycle or cycle != expected_cycle:
                    raise ValueError(f"scoring row cycle mismatch at source row {row_number}")
                target = float(row[target_index])
                if not np.isfinite(target) or target < 0.0 or target > 100.0:
                    raise ValueError(f"scoring threat_rank invalid at source row {row_number}")
                labels[record_id] = target
    after = file_sha256(str(train_path))
    if before != after or before != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("official Train changed during post-prelabel label access")
    if set(labels) != set(required):
        missing = sorted(set(required) - set(labels))[:8]
        raise ValueError(f"scoring labels do not cover the approved query IDs; missing={missing}")
    return labels


def _score_view(
    view_name: str,
    y: np.ndarray,
    frozen: np.ndarray,
    predictions: dict[str, np.ndarray],
    missing: np.ndarray,
    cycles: np.ndarray,
    raw_heads: dict[str, np.ndarray],
    mapped_slots: dict[str, np.ndarray],
    frozen_mapped_lgb: np.ndarray,
) -> dict[str, Any]:
    baseline_mae = mae(y, frozen)
    arm_rows: dict[str, Any] = {}
    for arm, pred in predictions.items():
        arm_rows[arm] = {
            "mae": mae(y, pred),
            "gain_vs_frozen": baseline_mae - mae(y, pred),
            "raw_head_mae": mae(y, raw_heads[arm]),
            "mapped_slot_mae": mae(y, mapped_slots[arm]),
            "mapped_slot_gain_vs_frozen_lgb": (
                mae(y, frozen_mapped_lgb) - mae(y, mapped_slots[arm])
            ),
        }
    supports: dict[str, np.ndarray] = {
        "stream_missing": missing,
        "stream_present": ~missing,
        "target_low_le5": y <= 5.0,
        "target_high_ge95": y >= 95.0,
    }
    subgroup_rows: dict[str, Any] = {}
    for group_name, mask in supports.items():
        count = int(np.sum(mask))
        subgroup_rows[group_name] = {
            "n": count,
            "supported": count >= 100,
            "arms": {},
        }
        for arm, pred in predictions.items():
            subgroup_rows[group_name]["arms"][arm] = {
                "candidate_mae": mae(y[mask], pred[mask]) if count else None,
                "frozen_mae": mae(y[mask], frozen[mask]) if count else None,
                "harm_vs_frozen": (
                    mae(y[mask], pred[mask]) - mae(y[mask], frozen[mask]) if count else None
                ),
            }
    by_cycle = {}
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        by_cycle[str(int(cycle))] = {
            "n": int(mask.sum()),
            "frozen_mae": mae(y[mask], frozen[mask]),
            "arms": {
                arm: {
                    "ensemble_mae": mae(y[mask], predictions[arm][mask]),
                    "raw_head_mae": mae(y[mask], raw_heads[arm][mask]),
                    "mapped_slot_mae": mae(y[mask], mapped_slots[arm][mask]),
                }
                for arm in predictions
            },
        }
    return {
        "view": view_name,
        "n": int(len(y)),
        "frozen_mae": baseline_mae,
        "arms": arm_rows,
        "C_minus_A_gain": arm_rows["A_L1_FAITHFUL"]["mae"] - arm_rows["C_L2_MONOTONE"]["mae"],
        "C_minus_B_gain": arm_rows["B_L2_CONTROL"]["mae"] - arm_rows["C_L2_MONOTONE"]["mae"],
        "per_cycle": by_cycle,
        "subgroups": subgroup_rows,
    }


def cmd_score(args: argparse.Namespace) -> int:
    prelabel_path = Path(args.prelabel_qa)
    qa = _read_json(prelabel_path)
    if qa.get("status") != "PASS" or qa.get("source_kind") != "official_prelabel_only":
        raise RuntimeError("scoring labels remain closed until official prelabel QA PASS")
    if file_sha256(str(prelabel_path)) != args.prelabel_sha256:
        raise RuntimeError("prelabel QA hash differs from the score command pin")
    train_path = _resolve_train(Path(args.data_dir))
    ensemble_root = Path(args.ensemble_root)
    loaded: dict[int, dict[str, np.ndarray]] = {}
    required: dict[str, int] = {}
    for origin in ORIGINS:
        path = ensemble_root / f"E{origin}_ensemble.npz"
        pinned_ensemble = qa.get("ensemble_outputs", {}).get(str(origin), {})
        if (Path(pinned_ensemble.get("path", "")) != path.resolve()
                or file_sha256(str(path)) != pinned_ensemble.get("sha256")):
            raise RuntimeError(f"E{origin} ensemble does not match the independent prelabel QA")
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]) for name in (
                "query_ids", "query_cycles", "missing", "frozen_final", "frozen_mapped_lgb",
                "candidate_A", "candidate_B", "candidate_C",
                "raw_A", "raw_B", "raw_C", "mapped_A", "mapped_B", "mapped_C",
            )}
        ids = arrays["query_ids"].astype(str)
        cycles = arrays["query_cycles"].astype(np.int64)
        low, high = (72, 75) if origin == 67 else (76, 79)
        mask = (cycles >= low) & (cycles <= high)
        for record_id, cycle in zip(ids[mask].tolist(), cycles[mask].tolist()):
            if record_id in required:
                raise ValueError("D1/D2 scored query IDs overlap")
            required[record_id] = int(cycle)
        arrays["score_mask"] = mask
        loaded[origin] = arrays
    labels_by_id = _query_labels_after_qa(train_path, required)

    views = []
    pooled_y: list[float] = []
    pooled_frozen: list[float] = []
    pooled_missing: list[bool] = []
    pooled_cycles: list[int] = []
    pooled_frozen_mapped: list[float] = []
    pooled_preds: dict[str, list[float]] = {arm: [] for arm in ARM_IDS}
    pooled_raw: dict[str, list[float]] = {arm: [] for arm in ARM_IDS}
    pooled_mapped: dict[str, list[float]] = {arm: [] for arm in ARM_IDS}
    for origin, view in ((67, "D1_E67_cycles72_75"), (75, "D2_E75_cycles76_79")):
        arrays = loaded[origin]
        mask = arrays["score_mask"]
        ids = arrays["query_ids"].astype(str)[mask]
        y = np.asarray([labels_by_id[str(record_id)] for record_id in ids], dtype=np.float64)
        predictions = {
            "A_L1_FAITHFUL": arrays["candidate_A"][mask].astype(np.float64),
            "B_L2_CONTROL": arrays["candidate_B"][mask].astype(np.float64),
            "C_L2_MONOTONE": arrays["candidate_C"][mask].astype(np.float64),
        }
        frozen = arrays["frozen_final"][mask].astype(np.float64)
        missing = arrays["missing"][mask].astype(bool)
        raw_heads = {
            "A_L1_FAITHFUL": arrays["raw_A"][mask].astype(np.float64),
            "B_L2_CONTROL": arrays["raw_B"][mask].astype(np.float64),
            "C_L2_MONOTONE": arrays["raw_C"][mask].astype(np.float64),
        }
        mapped_slots = {
            "A_L1_FAITHFUL": arrays["mapped_A"][mask].astype(np.float64),
            "B_L2_CONTROL": arrays["mapped_B"][mask].astype(np.float64),
            "C_L2_MONOTONE": arrays["mapped_C"][mask].astype(np.float64),
        }
        cycles = arrays["query_cycles"][mask].astype(np.int64)
        frozen_mapped_lgb = arrays["frozen_mapped_lgb"][mask].astype(np.float64)
        views.append(_score_view(view, y, frozen, predictions, missing, cycles,
                                 raw_heads, mapped_slots, frozen_mapped_lgb))
        pooled_y.extend(y.tolist())
        pooled_frozen.extend(frozen.tolist())
        pooled_missing.extend(missing.tolist())
        pooled_cycles.extend(cycles.tolist())
        pooled_frozen_mapped.extend(frozen_mapped_lgb.tolist())
        for arm in ARM_IDS:
            pooled_preds[arm].extend(predictions[arm].tolist())
            pooled_raw[arm].extend(raw_heads[arm].tolist())
            pooled_mapped[arm].extend(mapped_slots[arm].tolist())
    pooled = _score_view(
        "POOLED_D1_D2",
        np.asarray(pooled_y, dtype=np.float64),
        np.asarray(pooled_frozen, dtype=np.float64),
        {arm: np.asarray(values, dtype=np.float64) for arm, values in pooled_preds.items()},
        np.asarray(pooled_missing, dtype=bool),
        np.asarray(pooled_cycles, dtype=np.int64),
        {arm: np.asarray(values, dtype=np.float64) for arm, values in pooled_raw.items()},
        {arm: np.asarray(values, dtype=np.float64) for arm, values in pooled_mapped.items()},
        np.asarray(pooled_frozen_mapped, dtype=np.float64),
    )
    all_views = views + [pooled]
    supported_harms = [
        group["arms"]["C_L2_MONOTONE"]["harm_vs_frozen"]
        for view in all_views
        for group in view["subgroups"].values()
        if group["supported"]
    ]
    max_supported_harm = max(supported_harms) if supported_harms else None
    d1_gain = views[0]["arms"]["C_L2_MONOTONE"]["gain_vs_frozen"]
    d2_gain = views[1]["arms"]["C_L2_MONOTONE"]["gain_vs_frozen"]
    pooled_gain = pooled["arms"]["C_L2_MONOTONE"]["gain_vs_frozen"]
    pooled_gain_vs_a = pooled["C_minus_A_gain"]
    pooled_gain_vs_b = pooled["C_minus_B_gain"]
    supported_group_pass = (
        max_supported_harm is not None and max_supported_harm <= 0.25
    )
    missing_exact = qa.get("missing_rows_bitwise_exact_for_all_candidates") is True
    monotone_pass = all(
        item.get("raw_head_monotonicity") == "PASS"
        for item in qa.get("fit_checks", [])
        if item.get("fit_id", "").endswith("C_L2_MONOTONE")
    ) and sum(item.get("raw_head_monotonicity") == "PASS" for item in qa.get("fit_checks", [])) == 2
    scientific_pass = all((
        pooled_gain >= 0.02,
        pooled_gain_vs_a >= 0.02,
        d1_gain >= 0.0,
        d2_gain >= 0.0,
        pooled_gain_vs_b > 0.0,
        supported_group_pass,
        missing_exact,
        monotone_pass,
    ))
    exploratory_pass = all((
        (pooled_gain > 0.0 or (d2_gain >= 0.01 and pooled_gain >= -0.01)),
        d1_gain >= -0.05,
        -pooled_gain_vs_b <= 0.01,
        supported_group_pass,
        missing_exact,
        monotone_pass,
    ))
    report = {
        "schema_version": 1,
        "status": "SCORED_AFTER_PRELABEL_PASS",
        "experiment": "REG_IMPACT_MONOTONE_L2_V1",
        "prelabel_qa_sha256": file_sha256(str(prelabel_path)),
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "scored_query_rows": len(required),
        "test_or_sample_opened": False,
        "views": all_views,
        "supported_group_harm_max_for_C_vs_frozen": max_supported_harm,
        "scientific_gate": {
            "status": "PASS" if scientific_pass else "FAIL",
            "pooled_gain_vs_frozen": pooled_gain,
            "pooled_gain_vs_A": pooled_gain_vs_a,
            "window_gains_vs_frozen": {"D1": d1_gain, "D2": d2_gain},
            "pooled_gain_vs_B": pooled_gain_vs_b,
            "supported_group_harm_pass": supported_group_pass,
            "missing_rows_exact": missing_exact,
            "raw_C_head_monotone": monotone_pass,
        },
        "exploratory_gate": {
            "status": "PASS" if exploratory_pass else "FAIL",
            "pooled_gain_vs_frozen": pooled_gain,
            "pooled_harm_vs_B": -pooled_gain_vs_b,
            "D1_gain_vs_frozen": d1_gain,
            "D2_gain_vs_frozen": d2_gain,
            "supported_group_harm_pass": supported_group_pass,
            "missing_rows_exact": missing_exact,
            "raw_C_head_monotone": monotone_pass,
        },
        "gate_thresholds": {
            "scientific_pooled_gain_vs_frozen_and_A": 0.02,
            "scientific_window_gain_vs_frozen_min": 0.0,
            "scientific_pooled_gain_vs_B_min": 0.0,
            "max_supported_harm": 0.25,
            "exploratory_pooled_gain_vs_B_harm_max": 0.01,
            "group_support_minimum": 100,
        },
        "score_csv_written": False,
        "final_e79_or_kaggle_action": False,
    }
    _write_json_exclusive(Path(args.output), report)
    print(json.dumps({"status": report["status"], "scored_query_rows": report["scored_query_rows"],
                      "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


def cmd_prepare_prefix(args: argparse.Namespace) -> int:
    freeze, train_path = _check_source_freeze(Path(args.source_freeze), Path(args.data_dir))
    # Use the established OS advisory lock; file existence alone is not ownership.
    lock_path = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
    with exclusive_training_slot(lock_path):
        import psutil
        available = int(psutil.virtual_memory().available)
        initial = 1024 ** 3
        minimum = 512 * 1024 ** 2
        if available < initial:
            raise MemoryError(f"RAM preflight failed: {available} available; initial requirement is {initial}")
        stage_start = time.perf_counter()
        with _MemorySampler() as monitor:
            before = file_sha256(str(train_path))
            frame = pd.read_csv(train_path, usecols=["record_id", *HYBRID_FEATURES])
            after = file_sha256(str(train_path))
            if before != after or before != freeze["train_sha256"]:
                raise RuntimeError("official Train changed while reading target-free feature columns")
            output = Path(args.output_dir)
            summaries = []
            output_bytes = 0
            for origin in ORIGINS:
                origin_start = time.perf_counter()
                bundle = prepare_prefix_design(
                    frame,
                    origin,
                    source_train_sha256=before,
                    source_kind="official",
                )
                paths = _save_bundle(bundle, output)
                bundle_bytes = sum(path.stat().st_size for path in paths)
                output_bytes += bundle_bytes
                summaries.append({
                    "origin": origin,
                    "fit_rows": int(len(bundle.fit_rows)),
                    "query_rows": int(len(bundle.query_rows)),
                    "feature_width": int(bundle.fit_design.shape[1]),
                    "impact_load_value_index": bundle.impact_index,
                    "categorical_indices": list(bundle.categorical_indices),
                    "learned_transform_count": 3,
                    "fit_design_nbytes": int(bundle.fit_design.nbytes),
                    "query_design_nbytes": int(bundle.query_design.nbytes),
                    "bundle_output_bytes": int(bundle_bytes),
                    "preprocessing_seconds": float(time.perf_counter() - origin_start),
                    "design_path": str(paths[0].resolve()),
                    "receipt_path": str(paths[2].resolve()),
                })
                del bundle
            if monitor.minimum_system_available < minimum:
                raise MemoryError(
                    f"observed system available RAM fell below reserve: {monitor.minimum_system_available}"
                )
            report = {
                "status": "PASS_TARGET_FREE_PREFIX_PREPROCESSING",
                "source_freeze_sha256": file_sha256(str(Path(args.source_freeze))),
                "train_sha256": before,
                "available_ram_bytes_before_stage": available,
                "minimum_system_available_observed_bytes": monitor.minimum_system_available,
                "peak_process_rss_observed_bytes": monitor.peak_process_rss,
                "initial_free_ram_requirement_bytes": initial,
                "ram_reserve_bytes": minimum,
                "thread_count": 2,
                "heavy_job_lock": str(lock_path.resolve()),
                "target_column_read": False,
                "test_data_opened": False,
                "sample_submission_opened": False,
                "learned_transform_total": 6,
                "bundle_output_bytes_total": int(output_bytes),
                "preprocessing_seconds_total": float(time.perf_counter() - stage_start),
                "prefixes": summaries,
                "official_supervised_fits_started": False,
            }
        _write_json_exclusive(output / "PREFIX_PREPROCESSING_REPORT.json", report)
        print(json.dumps(report, sort_keys=True))
    return 0


def _synthetic_incumbent(bundle: PrefixDesignBundle, seed: int) -> dict[str, np.ndarray]:
    n = len(bundle.query_rows)
    rng = np.random.default_rng(seed)
    cycles = bundle.query_cycles.astype(np.int64)
    lgb_old = rank50_map(rng.normal(48, 14, n), cycles)
    ridge = np.clip(45.0 + 4.0 * rng.normal(size=n), 0.0, 100.0)
    word = rng.normal(50, 12, n)
    char = rng.normal(51, 11, n)
    pos = rng.normal(48, 13, n)
    missing = (np.arange(n) % 13 == 0)
    incumbent = reconstruct_nested_champion(lgb_old, ridge, word, char, pos, cycles, missing)
    return incumbent


def _exercise_fit_lifecycle_faults(
    temp: Path,
    approved_fit_ids: list[str],
    bindings_by_id: dict[str, dict[str, Any]],
    readback_fn: Callable[[str, Path, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    fit_id = approved_fit_ids[0]
    binding_sha = json_digest(bindings_by_id[fit_id])
    calls = {"prepare": 0, "fit": 0}

    def prepare() -> None:
        calls["prepare"] += 1
        return None

    def fit(_prepared: Any) -> Path:
        calls["fit"] += 1
        raise RuntimeError("synthetic-fit-lifecycle-sentinel")

    def rejected_before_callbacks(output_root: Path, ids: list[str] = approved_fit_ids,
                                  bindings: dict[str, dict[str, Any]] = bindings_by_id) -> None:
        try:
            run_fit_lifecycle(
                output_root=output_root,
                fit_id=fit_id,
                approved_fit_ids=ids,
                bindings_by_id=bindings,
                prepare_fn=prepare,
                fit_and_save_fn=fit,
                readback_fn=readback_fn,
            )
        except (RuntimeError, FileExistsError):
            if calls != {"prepare": 0, "fit": 0}:
                raise RuntimeError("lifecycle rejected an invalid state after prepare/fit callback")
            return
        raise RuntimeError("lifecycle unexpectedly accepted an invalid state")

    duplicate_root = temp / "guard_duplicate"
    ledger, claims = _fit_lifecycle_paths(duplicate_root)
    claims.mkdir(parents=True, exist_ok=True)
    claim_path = claims / f"{fit_id}.json"
    _write_json_exclusive(claim_path, {
        "schema_version": 1,
        "fit_id": fit_id,
        "binding_sha256": binding_sha,
        "pid": os.getpid(),
    })
    duplicate_event = {
        "schema_version": 1,
        "event_seq": 1,
        "event_time_utc": datetime.now(timezone.utc).isoformat(),
        "fit_id": fit_id,
        "status": "FIT_STARTED",
        "binding_sha256": binding_sha,
        "claim_path": str(claim_path.resolve()),
    }
    _append_fit_lifecycle_event(ledger, duplicate_event)
    _append_fit_lifecycle_event(ledger, {**duplicate_event, "event_seq": 2})
    rejected_before_callbacks(duplicate_root)
    duplicate_rejected = True

    orphan_root = temp / "guard_orphan"
    orphan_dir = orphan_root / fit_id
    orphan_dir.mkdir(parents=True, exist_ok=True)
    with (orphan_dir / "partial.bin").open("xb") as stream:
        stream.write(b"orphan")
    rejected_before_callbacks(orphan_root)
    orphan_rejected = True

    interrupted_root = temp / "guard_interrupted"
    interrupted_ledger, interrupted_claims = _fit_lifecycle_paths(interrupted_root)
    interrupted_claims.mkdir(parents=True, exist_ok=True)
    interrupted_claim = interrupted_claims / f"{fit_id}.json"
    _write_json_exclusive(interrupted_claim, {
        "schema_version": 1,
        "fit_id": fit_id,
        "binding_sha256": binding_sha,
        "pid": os.getpid(),
    })
    _append_fit_lifecycle_event(interrupted_ledger, {
        "schema_version": 1,
        "event_seq": 1,
        "event_time_utc": datetime.now(timezone.utc).isoformat(),
        "fit_id": fit_id,
        "status": "FIT_STARTED",
        "binding_sha256": binding_sha,
        "claim_path": str(interrupted_claim.resolve()),
    })
    rejected_before_callbacks(interrupted_root)
    interrupted_rejected = True

    failed_root = temp / "guard_failed"
    failed_attempts = 0

    def failed_fit(_prepared: Any) -> Path:
        nonlocal failed_attempts
        failed_attempts += 1
        raise RuntimeError("synthetic-fit-lifecycle-sentinel")

    try:
        run_fit_lifecycle(
            output_root=failed_root,
            fit_id=fit_id,
            approved_fit_ids=approved_fit_ids,
            bindings_by_id=bindings_by_id,
            prepare_fn=lambda: None,
            fit_and_save_fn=failed_fit,
            readback_fn=readback_fn,
        )
    except RuntimeError as exc:
        if "consumed" not in str(exc):
            raise
    else:
        raise RuntimeError("synthetic fit failure was not recorded as consumed")
    failure_events = [json.loads(line) for line in _fit_lifecycle_paths(failed_root)[0].read_text(encoding="utf-8").splitlines()]
    if [event["status"] for event in failure_events] != ["FIT_STARTED", "FIT_FAILED"]:
        raise RuntimeError("synthetic fit failure did not persist STARTED then FAILED")
    try:
        run_fit_lifecycle(
            output_root=failed_root,
            fit_id=fit_id,
            approved_fit_ids=approved_fit_ids,
            bindings_by_id=bindings_by_id,
            prepare_fn=prepare,
            fit_and_save_fn=fit,
            readback_fn=readback_fn,
        )
    except RuntimeError:
        if failed_attempts != 1 or calls != {"prepare": 0, "fit": 0}:
            raise RuntimeError("consumed failed fit was retried")
        failed_retry_rejected = True
    else:
        raise RuntimeError("consumed failed fit was unexpectedly retried")

    extra_id = "E79_FORBIDDEN"
    rejected_before_callbacks(temp / "guard_cap", approved_fit_ids + [extra_id], bindings_by_id)
    cap_rejected = True
    return {
        "duplicate_start_rejected_before_callbacks": duplicate_rejected,
        "orphan_output_rejected_before_callbacks": orphan_rejected,
        "interrupted_start_rejected_before_callbacks": interrupted_rejected,
        "failed_id_consumed_without_retry": failed_retry_rejected,
        "failure_ledger_statuses": [event["status"] for event in failure_events],
        "six_fit_cap_enforced_before_callbacks": cap_rejected,
        "invalid_state_callback_counts": calls,
    }


def _synthetic_smoke(report_path: Path) -> dict[str, Any]:
    if lightgbm.__version__ != EXPECTED_LIGHTGBM_VERSION:
        raise RuntimeError(f"expected LightGBM {EXPECTED_LIGHTGBM_VERSION}, got {lightgbm.__version__}")
    frame, target = synthetic_frame()
    canonical = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    synthetic_sha = hashlib.sha256(canonical).hexdigest()
    all_fits: list[dict[str, Any]] = []
    all_scored: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="reg_impact_monotone_l2_synthetic_") as temp_name:
        temp = Path(temp_name)
        bundle_root = temp / "bundles"
        fit_root = temp / "fits"
        ensemble_root = temp / "ensembles"
        fit_root.mkdir()
        ensemble_root.mkdir()
        fit_outputs: dict[tuple[int, str], tuple[PrefixDesignBundle, np.ndarray]] = {}
        query_truth: dict[int, np.ndarray] = {}
        bundles_by_origin: dict[int, PrefixDesignBundle] = {}
        bundle_receipts_by_origin: dict[int, Path] = {}
        for origin in ORIGINS:
            bundle = prepare_prefix_design(
                frame,
                origin,
                source_train_sha256=synthetic_sha,
                source_kind="synthetic",
            )
            bundle_receipt = _save_bundle(bundle, bundle_root)[2]
            roundtrip_bundle = _load_bundle(bundle_receipt)
            if not np.array_equal(bundle.fit_design, roundtrip_bundle.fit_design):
                raise RuntimeError("synthetic prefix bundle round-trip changed fit design bits")
            if not np.array_equal(bundle.query_design, roundtrip_bundle.query_design):
                raise RuntimeError("synthetic prefix bundle round-trip changed query design bits")
            bundles_by_origin[origin] = bundle
            bundle_receipts_by_origin[origin] = bundle_receipt

        approved_fit_ids = [f"E{origin}_{arm}" for origin in ORIGINS for arm in ARM_IDS]
        synthetic_authorization_sha = hashlib.sha256(b"REG_IMPACT_MONOTONE_L2_V1 synthetic lifecycle only").hexdigest()
        root_plan_sha = file_sha256(str(PLAN_PATH))
        bindings_by_id: dict[str, dict[str, Any]] = {}
        for origin in ORIGINS:
            bundle_receipt_path = bundle_receipts_by_origin[origin]
            bundle_receipt = _read_json(bundle_receipt_path)
            for arm in ARM_IDS:
                fit_id = f"E{origin}_{arm}"
                bindings_by_id[fit_id] = {
                    "fit_id": fit_id,
                    "source_kind": "synthetic",
                    "authorization_path": "SYNTHETIC_ONLY",
                    "authorization_sha256": synthetic_authorization_sha,
                    "source_freeze_sha256": synthetic_sha,
                    "root_plan_sha256": root_plan_sha,
                    "bundle_design_sha256": bundle_receipt["design_sha256"],
                    "bundle_receipt_sha256": file_sha256(str(bundle_receipt_path)),
                    "train_sha256": synthetic_sha,
                }

        def synthetic_readback(
            saved_id: str,
            receipt_path: Path,
            binding: dict[str, Any],
        ) -> dict[str, Any]:
            origin = int(saved_id.split("_", 1)[0][1:])
            return _readback_fit_artifacts(
                saved_id,
                receipt_path,
                bundle_receipts_by_origin[origin],
                bundles_by_origin[origin],
                binding,
            )

        fit_optimizer_call_count = 0
        lifecycle_runs: list[dict[str, Any]] = []
        for origin in ORIGINS:
            bundle = bundles_by_origin[origin]
            query_truth[origin] = target[bundle.query_rows]
            incumbent = _synthetic_incumbent(bundle, seed=5000 + origin)
            hinge = {"intercept": 1.5, "slope_low": 0.82, "slope_high": 0.94}
            candidates: dict[str, np.ndarray] = {}
            raw_predictions: dict[str, np.ndarray] = {}
            mapped_predictions: dict[str, np.ndarray] = {}
            for arm_id in ARM_IDS:
                fit_id = f"E{origin}_{arm_id}"
                binding = bindings_by_id[fit_id]

                def fit_and_save_synthetic(
                    fit_target: np.ndarray,
                    _bundle: PrefixDesignBundle = bundle,
                    _arm_id: str = arm_id,
                    _fit_id: str = fit_id,
                    _binding: dict[str, Any] = binding,
                ) -> Path:
                    nonlocal fit_optimizer_call_count
                    fit_optimizer_call_count += 1
                    model, raw_prediction, metadata = fit_arm(
                        _bundle,
                        fit_target,
                        _arm_id,
                        synthetic_params=SYNTHETIC_FIT_PARAMS,
                    )
                    directory = _save_fit_artifacts(
                        _bundle,
                        model,
                        raw_prediction,
                        metadata,
                        fit_root,
                        synthetic=True,
                        receipt_metadata={
                            "source_freeze_sha256": _binding["source_freeze_sha256"],
                            "root_plan_sha256": _binding["root_plan_sha256"],
                            "bundle_design_sha256": _binding["bundle_design_sha256"],
                            "bundle_receipt_sha256": _binding["bundle_receipt_sha256"],
                            "lifecycle_binding_sha256": json_digest(_binding),
                        },
                    )
                    return directory / "receipt.json"

                lifecycle_result = run_fit_lifecycle(
                    output_root=fit_root,
                    fit_id=fit_id,
                    approved_fit_ids=approved_fit_ids,
                    bindings_by_id=bindings_by_id,
                    prepare_fn=lambda _bundle=bundle: target[_bundle.fit_rows].copy(),
                    fit_and_save_fn=fit_and_save_synthetic,
                    readback_fn=synthetic_readback,
                )
                if lifecycle_result["status"] != "FITTED":
                    raise RuntimeError(f"fresh synthetic fit unexpectedly returned {lifecycle_result['status']}")
                lifecycle_runs.append({"fit_id": fit_id, "status": lifecycle_result["status"]})
                fit_dir = fit_root / fit_id
                saved_receipt = _read_json(fit_dir / "receipt.json")
                meta = saved_receipt["meta"]
                with np.load(fit_dir / "predictions.npz", allow_pickle=False) as saved:
                    raw = np.asarray(saved["raw_prediction"], dtype=np.float64)
                    mapped = np.asarray(saved["mapped_prediction"], dtype=np.float64)
                if not np.isfinite(mapped).all() or len(mapped) != len(bundle.query_rows):
                    raise RuntimeError("synthetic saved mapped query output failed basic QA")
                replacement = replace_lgb_slot_present_only(
                    incumbent, raw, bundle.query_cycles, hinge
                )
                candidates[arm_id] = replacement["candidate_final"]
                raw_predictions[arm_id] = raw
                mapped_predictions[arm_id] = replacement["new_lgb_mapped"]
                fit_outputs[(origin, arm_id)] = (bundle, raw)
                all_fits.append({
                    "fit_id": f"E{origin}_{arm_id}",
                    "origin": origin,
                    "arm_id": arm_id,
                    "objective": meta["objective"],
                    "feature_width": int(bundle.fit_design.shape[1]),
                    "impact_load_value_index": bundle.impact_index,
                    "categorical_indices": list(bundle.categorical_indices),
                    "constraint_vector": arm_parameters(bundle, arm_id).get("monotone_constraints"),
                    "model_roundtrip_sha256": file_sha256(str(fit_dir / "model.txt")),
                    "monotonicity": meta["monotonicity_check"],
                })
            ensemble_path = ensemble_root / f"E{origin}_ensemble.npz"
            _write_npz_exclusive(
                ensemble_path,
                query_ids=bundle.query_ids.astype(str),
                query_rows=bundle.query_rows.astype(np.int64),
                query_cycles=bundle.query_cycles.astype(np.int64),
                missing=incumbent["missing"].astype(bool),
                frozen_mapped_lgb=incumbent["mapped_lgb"],
                frozen_mapped_ridge=incumbent["mapped_ridge"],
                frozen_word_raw=incumbent["word_raw"],
                frozen_char_raw=incumbent["char_raw"],
                frozen_pos_raw=incumbent["pos_raw"],
                frozen_direct=incumbent["direct"],
                frozen_final=apply_saved_hinge(incumbent["direct"], hinge, incumbent["missing"]),
                hinge_intercept=np.asarray(hinge["intercept"], dtype=np.float64),
                hinge_slope_low=np.asarray(hinge["slope_low"], dtype=np.float64),
                hinge_slope_high=np.asarray(hinge["slope_high"], dtype=np.float64),
                candidate_A=candidates["A_L1_FAITHFUL"],
                candidate_B=candidates["B_L2_CONTROL"],
                candidate_C=candidates["C_L2_MONOTONE"],
                raw_A=raw_predictions["A_L1_FAITHFUL"],
                raw_B=raw_predictions["B_L2_CONTROL"],
                raw_C=raw_predictions["C_L2_MONOTONE"],
                mapped_A=mapped_predictions["A_L1_FAITHFUL"],
                mapped_B=mapped_predictions["B_L2_CONTROL"],
                mapped_C=mapped_predictions["C_L2_MONOTONE"],
            )

        saved_calls_before_readback = fit_optimizer_call_count
        saved_readback = run_fit_lifecycle(
            output_root=fit_root,
            fit_id=approved_fit_ids[0],
            approved_fit_ids=approved_fit_ids,
            bindings_by_id=bindings_by_id,
            prepare_fn=lambda: (_ for _ in ()).throw(RuntimeError("saved readback called prepare")),
            fit_and_save_fn=lambda _prepared: (_ for _ in ()).throw(RuntimeError("saved readback called optimizer")),
            readback_fn=synthetic_readback,
        )
        if saved_readback["status"] != "READBACK" or fit_optimizer_call_count != saved_calls_before_readback:
            raise RuntimeError("saved synthetic fit did not read back without an optimizer call")
        lifecycle_fault_checks = _exercise_fit_lifecycle_faults(
            temp, approved_fit_ids, bindings_by_id, synthetic_readback
        )
        lifecycle_fault_checks.update({
            "sequential_fit_ids": [x["fit_id"] for x in lifecycle_runs[:2]],
            "fit_count": len(lifecycle_runs),
            "actual_synthetic_optimizer_calls": fit_optimizer_call_count,
            "saved_id_readback_status": saved_readback["status"],
            "saved_id_readback_optimizer_calls": fit_optimizer_call_count - saved_calls_before_readback,
            "official_cli_and_synthetic_use_same_public_function": True,
        })

        # Independent durable pre-label QA is executed before toy query labels are scored.
        qa_path = temp / "PRELABEL_QA.json"
        verification = subprocess.run(
            [sys.executable, "-m", "scripts.verify_reg_impact_monotone_l2_v1", "synthetic",
             "--fit-root", str(fit_root),
             "--bundle-root", str(bundle_root), "--ensemble-root", str(ensemble_root),
             "--output", str(qa_path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if verification.returncode != 0:
            raise RuntimeError(f"independent synthetic prelabel verifier failed: {verification.stderr or verification.stdout}")
        qa = _read_json(qa_path)
        if qa.get("status") != "PASS":
            raise RuntimeError("independent synthetic prelabel QA did not PASS")

        pooled_ids: list[str] = []
        pooled_preds: dict[str, list[float]] = {arm: [] for arm in ARM_IDS}
        pooled_truth: list[float] = []
        for origin, lo, hi, view_name in ((67, 72, 75, "D1"), (75, 76, 79, "D2")):
            bundle = next(value[0] for key, value in fit_outputs.items() if key[0] == origin)
            with np.load(ensemble_root / f"E{origin}_ensemble.npz", allow_pickle=False) as z:
                cycles = np.asarray(z["query_cycles"], dtype=np.int64)
                mask = (cycles >= lo) & (cycles <= hi)
                ids = np.asarray(z["query_ids"]).astype(str)[mask]
                truth = query_truth[origin][mask]
                frozen = np.asarray(z["frozen_final"], dtype=np.float64)[mask]
                for arm, field in zip(ARM_IDS, ("candidate_A", "candidate_B", "candidate_C")):
                    pred = np.asarray(z[field], dtype=np.float64)[mask]
                    raw_field = "raw_" + arm[0]
                    mapped_field = "mapped_" + arm[0]
                    raw_mae = mae(truth, np.asarray(z[raw_field], dtype=np.float64)[mask])
                    mapped_mae = mae(truth, np.asarray(z[mapped_field], dtype=np.float64)[mask])
                    all_scored.append({
                        "view": view_name,
                        "arm_id": arm,
                        "n": int(len(truth)),
                        "raw_head_mae": raw_mae,
                        "mapped_slot_mae": mapped_mae,
                        "frozen_mae": mae(truth, frozen),
                        "candidate_mae": mae(truth, pred),
                        "gain_vs_frozen": mae(truth, frozen) - mae(truth, pred),
                        "synthetic_only": True,
                    })
                    pooled_preds[arm].extend(pred.tolist())
                pooled_ids.extend(ids.tolist())
                pooled_truth.extend(truth.tolist())

        # Exercise the final CSV contract on toy identifiers/predictions only.
        c_pooled = pooled_preds["C_L2_MONOTONE"]
        csv_path = temp / "synthetic_submission_contract.csv"
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["record_id", "threat_rank"])
            writer.writerows(zip(pooled_ids, c_pooled))
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        if rows[0] != ["record_id", "threat_rank"] or len(rows) != len(pooled_ids) + 1:
            raise RuntimeError("synthetic CSV header/row-count contract failed")
        if [row[0] for row in rows[1:]] != pooled_ids:
            raise RuntimeError("synthetic CSV identity order changed")
        if any(not np.isfinite(float(row[1])) for row in rows[1:]):
            raise RuntimeError("synthetic CSV predictions are nonfinite")
        result = {
            "schema_version": 1,
            "status": "PASS",
            "source_kind": "synthetic_only",
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "synthetic_source_sha256": synthetic_sha,
            "synthetic_target_labels_used_only_after_prelabel_qa": True,
            "prelabel_qa_status": qa["status"],
            "prelabel_qa_sha256": file_sha256(str(qa_path)),
            "fit_ids": [x["fit_id"] for x in all_fits],
            "fit_count": len(all_fits),
            "prefix_transform_count": 6,
            "prefix_svd_components": {
                str(o): int(next(v[0] for k, v in fit_outputs.items() if k[0] == o).svd_components.shape[0])
                for o in ORIGINS
            },
            "fit_lifecycle": lifecycle_fault_checks,
            "fits": all_fits,
            "synthetic_score_rows": all_scored,
            "pooled_synthetic_rows": len(pooled_truth),
            "synthetic_csv_contract": {
                "status": "PASS",
                "header": rows[0],
                "rows": len(rows) - 1,
                "path_was_temporary": True,
            },
            "official_fits_started": False,
            "official_labels_opened": False,
            "test_or_sample_opened": False,
            "final_submission_written": False,
        }
    _write_json_exclusive(report_path, result)
    print(json.dumps({"status": result["status"], "report": str(report_path.resolve()),
                      "fit_count": result["fit_count"], "prelabel_qa": result["prelabel_qa_status"]}, sort_keys=True))
    return result


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("synthetic-smoke", help="run the complete synthetic-only lifecycle")
    smoke.add_argument("--report", required=True)
    freeze = sub.add_parser("freeze-source", help="freeze source, input hashes, and exact prefix recipe")
    freeze.add_argument("--data-dir", required=True, help="directory containing regression/Train.csv")
    freeze.add_argument("--synthetic-report", required=True)
    freeze.add_argument("--input-freeze", default=str(AUDIT_FREEZE_PATH))
    freeze.add_argument("--output", required=True)
    prep = sub.add_parser("prepare-prefix", help="build two target-free official prefix bundles")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--source-freeze", required=True)
    prep.add_argument("--output-dir", required=True)
    fit = sub.add_parser("fit-arm", help="fit one arm only after the exact six-fit source-pass")
    fit.add_argument("--arm", choices=ARM_IDS, required=True)
    fit.add_argument("--bundle-receipt", required=True)
    fit.add_argument("--authorization", required=True)
    fit.add_argument("--source-freeze", required=True)
    fit.add_argument("--data-dir", required=True)
    fit.add_argument("--output-root", required=True)
    prelabel = sub.add_parser("prelabel-qa", help="independently integrate all six outputs without labels")
    prelabel.add_argument("--fit-root", required=True)
    prelabel.add_argument("--bundle-root", required=True)
    prelabel.add_argument("--ensemble-root", required=True)
    prelabel.add_argument("--input-freeze", default=str(AUDIT_FREEZE_PATH))
    prelabel.add_argument("--source-freeze", required=True)
    prelabel.add_argument("--output", required=True)
    score = sub.add_parser("score", help="score D1/D2 labels only after official prelabel QA PASS")
    score.add_argument("--prelabel-qa", required=True)
    score.add_argument("--prelabel-sha256", required=True)
    score.add_argument("--data-dir", required=True)
    score.add_argument("--ensemble-root", required=True)
    score.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    if args.command == "synthetic-smoke":
        _synthetic_smoke(Path(args.report))
        return 0
    if args.command == "freeze-source":
        return cmd_freeze_source(args)
    if args.command == "prepare-prefix":
        return cmd_prepare_prefix(args)
    if args.command == "fit-arm":
        return cmd_fit_arm(args)
    if args.command == "prelabel-qa":
        return cmd_prelabel_qa(args)
    if args.command == "score":
        return cmd_score(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
