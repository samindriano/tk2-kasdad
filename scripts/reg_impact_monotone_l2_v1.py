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
from pathlib import Path
from typing import Any

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
    bundle_receipt = _read_json(bundle_receipt_path)
    bundle = _load_bundle(bundle_receipt_path)
    fit_id = f"E{bundle.origin}_{args.arm}"
    source_freeze, train_path = _check_source_freeze(
        Path(args.source_freeze), Path(args.data_dir)
    )
    gate = _fit_authorization(
        Path(args.authorization), Path(args.source_freeze), bundle_receipt_path, fit_id
    )
    lock_path = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
    with exclusive_training_slot(lock_path):
        import psutil
        available = int(psutil.virtual_memory().available)
        if available < 1024 ** 3:
            raise MemoryError(f"fit RAM preflight failed: {available} available; need 1 GiB initial free RAM")
        target = _read_prefix_target(train_path, bundle)
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
                "target_rows_materialized": len(target),
                "query_target_rows_materialized": 0,
            },
        )
        print(json.dumps({"fit_id": fit_id, "status": "FITTED", "receipt": str((directory / "receipt.json").resolve()),
                          "model_sha256": file_sha256(str(directory / "model.txt")),
                          "predictions_sha256": file_sha256(str(directory / "predictions.npz"))}, sort_keys=True))
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
            query_truth[origin] = target[bundle.query_rows]
            incumbent = _synthetic_incumbent(bundle, seed=5000 + origin)
            hinge = {"intercept": 1.5, "slope_low": 0.82, "slope_high": 0.94}
            candidates: dict[str, np.ndarray] = {}
            raw_predictions: dict[str, np.ndarray] = {}
            mapped_predictions: dict[str, np.ndarray] = {}
            for arm_id in ARM_IDS:
                model, raw, meta = fit_arm(
                    bundle,
                    target[bundle.fit_rows],
                    arm_id,
                    synthetic_params=SYNTHETIC_FIT_PARAMS,
                )
                fit_dir = _save_fit_artifacts(
                    bundle,
                    model,
                    raw,
                    meta,
                    fit_root,
                    synthetic=True,
                )
                with np.load(fit_dir / "predictions.npz", allow_pickle=False) as saved:
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
