"""Durably recover the exact already-fitted C10 artifact without retraining."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_c10_trigram_e79_exploratory_v1 import (
    ARMED_PATH,
    BenchmarkError,
    CSV_PATH,
    FIT_ID,
    FIT_RECEIPT_PATH,
    INFERENCE_BUILD_PATH,
    LEDGER_PATH,
    MAX_UNITS,
    MODEL_PATH,
    PREFIT_AUDIT_PATH,
    PREFIT_FREEZE_PATH,
    PREDICTIONS_PATH,
    RECIPE_PATH,
    REPORT,
    RUN,
    SYNTHETIC_QA_PATH,
    TRAIN_PATH,
    TRAINED_FREEZE_PATH,
    _committed_source_hashes,
    _git_file_commit,
    _read_json,
    _source_hashes,
    _write_json_once,
    canonical_sha,
    file_sha,
)
from src.models.optimization_v3_batch1_c10 import (
    FEATURE_NAMES,
    SeparateTrigramSparseClassifier,
    _normalise_parameters,
)
from src.models.round1_evaluation import array_digest

RECOVERY_MANIFEST_PATH = REPORT / "FIT_RECOVERY.json"
RECOVERY_LEDGER_PATH = REPORT / "RECOVERY_LEDGER.jsonl"
PARTIAL_MODEL_PATH = RUN / "c10_model.joblib.partial"


def _canonical_array_sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def durable_copy_verified(source: Path, destination: Path) -> str:
    """Copy exact bytes via a writable handle; preserve source and verify SHA-256."""
    if destination.exists():
        raise BenchmarkError(f"Recovery destination already exists: {destination}")
    source_sha = file_sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".recovery-partial")
    if partial.exists():
        if file_sha(partial) != source_sha:
            raise BenchmarkError(f"Recovery copy temporary differs from original model: {partial}")
    else:
        with source.open("rb") as incoming, partial.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    if file_sha(partial) != source_sha:
        raise BenchmarkError("Recovered model copy digest differs before atomic promotion")
    os.replace(partial, destination)
    if file_sha(destination) != source_sha:
        raise BenchmarkError("Recovered model digest differs after atomic promotion")
    return source_sha


def _verify_model(model: Any, train: pd.DataFrame, target: np.ndarray,
                  recipe: dict[str, Any], started: dict[str, Any]) -> dict[str, Any]:
    if type(model) is not SeparateTrigramSparseClassifier:
        raise BenchmarkError("Recovered artifact is not the frozen C10 classifier class")
    state = model.audit_state()
    state_body = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != canonical_sha(state_body):
        raise BenchmarkError("Recovered C10 state digest does not reproduce")
    if state.get("fit_rows") != 50608 or len(train) != 50608:
        raise BenchmarkError("Recovered C10 state is not a full 50,608-row Train fit")
    if state.get("fit_target_sha256") != started.get("target_sha256"):
        raise BenchmarkError("Recovered C10 target hash differs from the original FIT_STARTED event")
    if not np.array_equal(model.model.classes_, np.asarray([0, 1])):
        raise BenchmarkError("Recovered C10 classifier class order differs from binary target")
    if int(model.model.n_estimators_) != 400 or int(model.model.booster_.num_trees()) != 400:
        raise BenchmarkError("Recovered C10 LightGBM iteration count is not exactly 400")
    if int(model.model.booster_.num_feature()) != int(state["design"]["shape"][1]):
        raise BenchmarkError("Recovered LightGBM feature dimension is malformed")

    expected_features = recipe["model"]["feature_allowlist"]
    if list(FEATURE_NAMES) != expected_features or state.get("input_feature_names") != expected_features:
        raise BenchmarkError("Recovered C10 model feature allowlist differs from frozen recipe")
    expected_word = recipe["model"]["word_tfidf"]
    expected_trigram = recipe["model"]["separate_trigram_tfidf"]
    word_params = _normalise_parameters(model.word.get_params())
    trigram_params = _normalise_parameters(model.trigram.get_params())
    if any(word_params.get(key) != value for key, value in expected_word.items()):
        raise BenchmarkError("Recovered C10 word vectorizer parameters differ")
    if any(trigram_params.get(key) != value for key, value in expected_trigram.items()):
        raise BenchmarkError("Recovered C10 trigram vectorizer parameters differ")
    if state.get("trigram_tfidf", {}).get("vocabulary_size") != 10000:
        raise BenchmarkError("Recovered C10 trigram vocabulary count is not the recorded 10,000")
    if state.get("word_tfidf", {}).get("vocabulary_size", 0) > 20000:
        raise BenchmarkError("Recovered C10 word vocabulary exceeds its frozen cap")
    expected_lgbm = recipe["model"]["lightgbm"]
    actual_lgbm = _normalise_parameters(model.model.get_params())
    if any(actual_lgbm.get(key) != value for key, value in expected_lgbm.items()
           if key != "early_stopping") or expected_lgbm.get("early_stopping") is not False:
        raise BenchmarkError("Recovered C10 LightGBM parameters differ from the frozen recipe")
    if state.get("classifier_parameters") != {
        key: value for key, value in expected_lgbm.items() if key != "early_stopping"
    }:
        raise BenchmarkError("Recovered C10 recorded classifier parameters differ from recipe")
    if started.get("fit_rows") != 50608 or started.get("cycle_range") != [47, 79]:
        raise BenchmarkError("Original fit start ledger does not prove full Train cycle 47-79")

    cycles = pd.to_numeric(train["cycle_code"], errors="raise").astype(np.int64).to_numpy()
    ids = train["record_id"].astype(str).tolist()
    if (set(np.unique(cycles)) != set(range(47, 80)) or train["record_id"].isna().any()
            or train["record_id"].duplicated().any()):
        raise BenchmarkError("Official Train cycle support differs from the frozen 47-79 population")
    if _canonical_array_sha(ids) != _read_json(PREFIT_AUDIT_PATH)["official_train_target_free"]["record_id_sequence_sha256"]:
        raise BenchmarkError("Recovered model Train IDs do not match the prefit identity sequence")
    if array_digest(target.astype(np.int8)) != started.get("target_sha256"):
        raise BenchmarkError("Current official Train target hash differs from original fit start")
    return state


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise BenchmarkError(f"Refusing to replace different recovery evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def recover() -> dict[str, Any]:
    if any(path.exists() for path in (INFERENCE_BUILD_PATH, PREDICTIONS_PATH, CSV_PATH)):
        raise BenchmarkError("Inference output already exists; recovery must precede Test access")
    if not PARTIAL_MODEL_PATH.is_file():
        raise BenchmarkError("Original C10 partial model artifact is missing")
    if not SYNTHETIC_QA_PATH.is_file() or _read_json(SYNTHETIC_QA_PATH).get("status") != "PASS_SYNTHETIC_C10_END_TO_END_AND_RESUME":
        raise BenchmarkError("Frozen synthetic QA evidence is unavailable")

    recipe = _read_json(RECIPE_PATH)
    prefit = _read_json(PREFIT_FREEZE_PATH)
    audit = _read_json(PREFIT_AUDIT_PATH)
    armed = _read_json(ARMED_PATH)
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    arms = [event for event in events if event.get("event") == "FIT_ARMED"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    if len(starts) != 1 or len(arms) != 1 or len(failures) != 1 or any(
        event.get("event") in {"FIT_SAVED"} for event in events
    ):
        raise BenchmarkError("Recovery requires exactly one original start and one recorded persistence failure")
    started, armed_event, failed = starts[0], arms[0], failures[0]
    if (failed.get("failure_type") != "OSError"
            or "Bad file descriptor" not in failed.get("failure_message", "")
            or failed.get("retry_count") != 0):
        raise BenchmarkError("Original failure is not the known post-joblib Windows fsync issue")
    if (started.get("fit_id") != FIT_ID or started.get("train_sha256") != prefit["train"]["train_sha256"]
            or started.get("freeze_signature") != prefit.get("freeze_signature")
            or armed.get("fit_id") != FIT_ID
            or armed_event.get("head") != armed.get("armed_at_head")
            or _git_file_commit(ARMED_PATH) != started.get("head")):
        raise BenchmarkError("Original C10 fit identity does not reconcile to its arm and freeze")
    if file_sha(TRAIN_PATH) != prefit["train"]["train_sha256"]:
        raise BenchmarkError("Official Train raw hash changed after fit")
    if _source_hashes() != prefit["source_hashes"]:
        raise BenchmarkError("Frozen C10 training sources changed since prefit")
    _committed_source_hashes(prefit["source_commit"])
    partial_sha = file_sha(PARTIAL_MODEL_PATH)
    model = joblib.load(PARTIAL_MODEL_PATH)
    frame = pd.read_csv(TRAIN_PATH, usecols=[*FEATURE_NAMES, "record_id", "artifact_signal"])
    target_float = pd.to_numeric(frame.pop("artifact_signal"), errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target_float).all() or not np.isin(target_float, (0.0, 1.0)).all():
        raise BenchmarkError("Official Train target is not finite binary data")
    target = target_float.astype(np.int8)
    state = _verify_model(model, frame, target, recipe, started)

    if MODEL_PATH.exists():
        if file_sha(MODEL_PATH) != partial_sha:
            raise BenchmarkError("Existing recovered model path is not byte-identical to the partial")
        recovered_sha = partial_sha
    else:
        recovered_sha = durable_copy_verified(PARTIAL_MODEL_PATH, MODEL_PATH)
    if recovered_sha != partial_sha:
        raise BenchmarkError("Recovered model bytes are not bit-identical to the original partial artifact")
    target_counts = {str(int(value)): int(np.count_nonzero(target == value)) for value in (0, 1)}
    receipt_body = {
        "schema_version": 1,
        "status": "RECOVERED_SAME_FIT_AFTER_POST_FIT_PERSISTENCE_ERROR",
        "fit_id": FIT_ID,
        "source_commit": prefit["source_commit"],
        "fit_start_head": started["head"],
        "recipe_sha256": prefit["recipe_sha256"],
        "prefit_freeze_signature": prefit["freeze_signature"],
        "train_sha256": prefit["train"]["train_sha256"],
        "fit_rows": int(len(target)),
        "cycle_range": [47, 79],
        "target_counts": target_counts,
        "target_sha256": started["target_sha256"],
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_file_sha256": recovered_sha,
        "original_partial_path": PARTIAL_MODEL_PATH.relative_to(ROOT).as_posix(),
        "original_partial_sha256": partial_sha,
        "recovered_model_sha256_matches_partial": True,
        "model_state": state,
        "model_state_sha256": state["state_sha256"],
        "official_predictor_fits": 1,
        "preprocessing_units": {
            "structured_selected_preprocessor": 1,
            "standard_scaler": 1,
            "one_hot_encoder": 1,
            "word_tfidf": 1,
            "trigram_tfidf": 1,
        },
        "learned_unit_total": 6,
        "duration_seconds": None,
        "duration_seconds_unavailable": True,
        "resource_snapshot_at_fit_start": started["resource_snapshot"],
        "original_fit_failed_event_preserved": True,
        "recovery_is_not_a_refit_or_retry": True,
        "test_features_opened": False,
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha(receipt_body)}
    _write_json_once(FIT_RECEIPT_PATH, receipt)
    recovery_body = {
        "schema_version": 1,
        "status": "PASS_RECOVERY_OF_SAME_CONSUMED_C10_FIT",
        "fit_id": FIT_ID,
        "recovery_source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
             Path(__file__).resolve().relative_to(ROOT).as_posix()], text=True
        ).strip(),
        "recovery_script_sha256": file_sha(Path(__file__).resolve()),
        "original_fit_source_commit": prefit["source_commit"],
        "original_fit_start_head": started["head"],
        "original_budget_ledger_sha256": file_sha(LEDGER_PATH),
        "original_fit_started_event": started,
        "original_fit_failed_event": failed,
        "original_fit_failed_event_preserved": True,
        "original_partial_path": PARTIAL_MODEL_PATH.relative_to(ROOT).as_posix(),
        "original_partial_sha256": partial_sha,
        "recovered_model_sha256": recovered_sha,
        "same_bytes_verified": recovered_sha == partial_sha,
        "loaded_model_class": f"{type(model).__module__}.{type(model).__name__}",
        "fit_rows_verified": state["fit_rows"],
        "fit_target_hash_matches_start": state["fit_target_sha256"] == started["target_sha256"],
        "trigram_vocab_verified": state["trigram_tfidf"]["vocabulary_size"],
        "lightgbm_tree_count_verified": int(model.model.booster_.num_trees()),
        "predictor_fits_total": 1,
        "learned_units_total": MAX_UNITS,
        "additional_budget_units": 0,
        "retries": 0,
        "train_sha256": prefit["train"]["train_sha256"],
        "recipe_sha256": prefit["recipe_sha256"],
        "test_features_opened": False,
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "fit_duration_seconds": None,
        "fit_duration_unavailable": True,
    }
    recovery = {**recovery_body, "recovery_sha256": canonical_sha(recovery_body)}
    _write_json_once(RECOVERY_MANIFEST_PATH, recovery)
    recovery_event = {
        "event": "FIT_RECOVERED",
        "fit_id": FIT_ID,
        "recovery_sha256": recovery["recovery_sha256"],
        "receipt_sha256": file_sha(FIT_RECEIPT_PATH),
        "original_failed_event_preserved": True,
        "additional_budget_units": 0,
        "retry_count": 0,
    }
    _write_text_once(RECOVERY_LEDGER_PATH, json.dumps(recovery_event, sort_keys=True) + "\n")
    trained_body = {
        "schema_version": 1,
        "status": "TRAINED_STATE_FROZEN_BEFORE_TEST_FEATURES",
        "fit_id": FIT_ID,
        "source_commit": prefit["source_commit"],
        "fit_start_head": started["head"],
        "train_sha256": prefit["train"]["train_sha256"],
        "fit_receipt_sha256": file_sha(FIT_RECEIPT_PATH),
        "model_file_sha256": recovered_sha,
        "model_state_sha256": state["state_sha256"],
        "recipe_sha256": prefit["recipe_sha256"],
        "prefit_freeze_signature": prefit["freeze_signature"],
        "fit_rows": 50608,
        "learned_unit_total": MAX_UNITS,
        "recovered_same_fit": True,
        "recovery_manifest_sha256": file_sha(RECOVERY_MANIFEST_PATH),
        "original_failed_budget_event_preserved": True,
        "test_features_opened": False,
        "test_labels_opened": False,
        "sample_submission_opened": False,
    }
    trained = {**trained_body, "freeze_signature": canonical_sha(trained_body)}
    _write_json_once(TRAINED_FREEZE_PATH, trained)
    return recovery


if __name__ == "__main__":
    print(json.dumps(recover(), ensure_ascii=False, sort_keys=True))
