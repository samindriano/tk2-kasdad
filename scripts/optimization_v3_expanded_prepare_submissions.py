"""Full-Train refit and official-Test inference for the five added V3 candidates.

The frozen continuation and candidate modules remain untouched. This runner
uses official Train only for fitting, opens Test only after fitting for
inference, writes one-shot CSV/manifests, and never calls Kaggle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    optimization_v3_expanded_benchmark as campaign,
    optimization_v3_expanded_recovery as recovery,
    prepare_c16_r22_fulltrain_submissions as previous,
)
from scripts import prepare_initial_submissions as initial  # noqa: E402
from src.data.audit import sha256 as file_sha256  # noqa: E402
from src.models.optimization_v3_batch1 import (  # noqa: E402
    append_event,
    exclusive_training_slot,
    file_sha256 as local_file_sha256,
    json_sha256,
)
from src.models import (  # noqa: E402
    optimization_v3_batch1_c11 as c11,
    optimization_v3_batch1_c17 as c17,
    optimization_v3_batch1_r06 as r06,
    optimization_v3_batch1_r11 as r11,
    optimization_v3_batch1_r21 as r21,
)


CAMPAIGN_ID = "optimization_v3_expanded_submissions_v1"
NEW_CANDIDATES = ("C17", "C11", "R21", "R11", "R06_ORIGINAL")
FIT_COUNTS = {"C17": 1, "C11": 1, "R21": 1, "R11": 9, "R06_ORIGINAL": 1}
TASKS = {"C17": "classification", "C11": "classification",
         "R21": "regression", "R11": "regression", "R06_ORIGINAL": "regression"}
THREADS = 2
GLOBAL_CAP = 180
NEW_RESERVATION_KEY = campaign.NEW_RESERVATION_KEY
RECOVERY_RESERVATION_KEY = recovery.RESERVATION_KEY
PREVIOUS_OFFLEDGER_FITS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
STRUCTURED_NNZ_UPPER_PER_ROW = 807
LIGHTGBM_MEMORY_OVERHEAD_BYTES = 1_500_000_000
TRAINING_LOCK = previous.TRAINING_LOCK
REPORT_ROOT = ROOT / "reports/submissions/optimization_v3_expanded_v1"
FREEZE_PATH = REPORT_ROOT / "EXECUTION_FREEZE.json"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1"
CSV_ROOT = ROOT / "submissions/optimization_v3_expanded_v1"
LEDGER_PATH = campaign.LEDGER_PATH

SOURCE_PATHS = (
    "scripts/optimization_v3_expanded_prepare_submissions.py",
    "scripts/optimization_v3_expanded_benchmark.py",
    "scripts/optimization_v3_expanded_recovery.py",
    "scripts/optimization_v3_expanded_evaluate_recovery.py",
    "scripts/prepare_initial_submissions.py",
    "scripts/prepare_c16_r22_fulltrain_submissions.py",
    "src/data/audit.py",
    "src/utils/paths.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/optimization_v2_regression.py",
    "src/models/optimization_v3_batch1.py",
    "src/models/optimization_v3_batch1_c11.py",
    "src/models/optimization_v3_batch1_c16.py",
    "src/models/optimization_v3_batch1_c17.py",
    "src/models/optimization_v3_batch1_r06.py",
    "src/models/optimization_v3_batch1_r11.py",
    "src/models/optimization_v3_batch1_r21.py",
    "src/models/optimization_v3_batch1_r22.py",
    "src/models/optimization_v3_sparse_classification.py",
    "src/models/optimization_v3_expanded_continuation.py",
    "src/models/regression_research_text.py",
    "src/models/round1_models.py",
)


class SubmissionPrepError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SubmissionPrepError(f"Refusing to overwrite differing artifact: {path}")
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _source_hashes() -> dict[str, str]:
    return {relative: local_file_sha256(ROOT / relative) for relative in SOURCE_PATHS}


def _input_paths_and_hashes() -> dict[str, Any]:
    result: dict[str, Any] = {}
    data_dir = initial.resolve_data_dir()
    for task in ("classification", "regression"):
        paths = initial.task_files(task, data_dir)
        hashes: dict[str, str] = {}
        for role, path in paths.items():
            if path is None or not Path(path).is_file():
                raise SubmissionPrepError(f"Official {task} {role} input is missing")
            hashes[role] = file_sha256(Path(path))
        if hashes != initial.EXPECTED_HASHES[task]:
            raise SubmissionPrepError(f"Official {task} raw input hashes differ from the committed receipt")
        result[task] = {
            "paths": {role: str(Path(path).resolve()) for role, path in paths.items()},
            "sha256": hashes,
            "test_values_opened": False,
        }
    return result


def _load_train_only(task: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_dir = initial.resolve_data_dir()
    paths = initial.task_files(task, data_dir)
    path = Path(paths["train"])
    observed = file_sha256(path)
    if observed != initial.EXPECTED_HASHES[task]["train"]:
        raise SubmissionPrepError(f"Official {task} Train hash differs from the expected input")
    frame = pd.read_csv(path)
    target_name = "artifact_signal" if task == "classification" else "threat_rank"
    if len(frame) != 50_608 or frame.record_id.isna().any() or not frame.record_id.is_unique:
        raise SubmissionPrepError(f"Official {task} Train row count or record_id integrity failed")
    if target_name not in frame:
        raise SubmissionPrepError(f"Official {task} Train lacks target {target_name}")
    cycles_float = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.float64)
    if (not np.isfinite(cycles_float).all() or not np.equal(cycles_float, np.floor(cycles_float)).all()
            or not np.array_equal(np.unique(cycles_float.astype(np.int64)), np.arange(47, 80))):
        raise SubmissionPrepError(f"Official {task} Train cycle population is not 47-79")
    y = frame[target_name].to_numpy()
    if task == "classification":
        if pd.isna(y).any() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise SubmissionPrepError("Classification Train target is not complete binary data")
    else:
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all() or np.any((y < 0.0) | (y > 100.0)):
            raise SubmissionPrepError("Regression Train target is incomplete or outside [0,100]")
    return frame, {"path": str(path.resolve()), "sha256": observed, "rows": len(frame),
                   "target": target_name, "cycles": [47, 79]}


def _candidate_recipe(card: str) -> dict[str, Any]:
    if card == "C17":
        return {"task": "classification", "model": "LongerRegularizedSparseClassifier",
                "feature_allowlist": list(c17.FEATURE_NAMES), "lightgbm_parameters": dict(c17.PARAMS),
                "tfidf_parameters": c17._params_json(c17._TFIDF_PARAMS),
                "preprocessing_fit_on": "all 50,608 official Train rows",
                "route": "present: 0.75*C2_B + 0.25*C17; missing: CB_MISSING",
                "threshold": c17.THRESHOLD, "fit_predictors": 1}
    if card == "C11":
        parameters = dict(c11.SPARSE_PARAMS)
        parameters.update({"n_jobs": c11.THREADS, "deterministic": True, "force_col_wise": True})
        return {"task": "classification", "model": "WordCharSparseClassifier",
                "feature_allowlist": list(c11.FEATURE_NAMES),
                "lightgbm_parameters": parameters,
                "word_tfidf_parameters": c11._safe_params(c11.WORD_TFIDF_PARAMS),
                "char_tfidf_parameters": c11._safe_params(c11.CHAR_TFIDF_PARAMS),
                "preprocessing_fit_on": "all 50,608 official Train rows",
                "route": "present: 0.75*C2_B + 0.25*C11; missing: CB_MISSING",
                "threshold": 0.50, "fit_predictors": 1}
    if card == "R21":
        return {"task": "regression", "model": "XGBoostSVD32Regressor",
                "feature_allowlist": list(r21.FEATURE_NAMES),
                "xgboost_parameters": dict(r21.XGB_PARAMETERS),
                "preprocessing": "fit structured/OHE, TF-IDF and SVD32 on all official Train",
                "mapping": "one rank50 mapping per complete Test cycle using average ties; clip mapped component [0,100]",
                "blend": "clip(0.75*three-component champion + 0.25*R21 rank50, 0, 100)",
                "fit_predictors": 1}
    if card == "R11":
        return {"task": "regression", "model": "CumulativeOrdinalSVD32",
                "feature_allowlist": list(r11.FEATURE_NAMES),
                "thresholds": list(r11.THRESHOLDS), "head_parameters": dict(r11.HEAD_PARAMETERS),
                "preprocessing": "fit structured/OHE, TF-IDF and SVD32 once on all official Train",
                "postprocessing": "row-wise PAVA; interpolate conditional median; rank50 once per complete Test cycle with average ties",
                "blend": "clip(0.75*three-component champion + 0.25*R11 rank50, 0, 100)",
                "fit_predictors": 9}
    if card == "R06_ORIGINAL":
        return {"task": "regression", "model": "R06CharacterSparseL1Hybrid",
                "feature_allowlist": list(r06.FEATURE_NAMES),
                "lightgbm_parameters": dict(r06.SPARSE_LGB_PARAMETERS),
                "char_tfidf_parameters": campaign._json_clean(r06.CHAR_TFIDF_PARAMETERS),
                "row_population": "all Train and Test rows; no present-only routing",
                "mapping": "one rank50 mapping per complete Test cycle using average ties",
                "blend": "clip(0.75*three-component champion + 0.25*R06 rank50, 0, 100)",
                "fit_predictors": 1}
    raise SubmissionPrepError(f"Candidate not in the authorized list: {card}")


def _qualification_evidence() -> dict[str, Any]:
    campaign_report = _read_json(
        ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/CAMPAIGN_REPORT.json"
    )
    if campaign_report.get("evaluated_candidates") != ["C17", "C11", "R21", "R06"]:
        raise SubmissionPrepError("Saved continuation evaluation candidate set differs from authorization")
    old = _read_json(previous.REPORT_ROOT / "SUBMISSION_STATUS.json")
    if old.get("classification", {}).get("candidate") != "C16" or old.get("regression", {}).get("candidate") != "R22_ORIGINAL":
        raise SubmissionPrepError("Existing C16/R22 submission identity differs")
    return {
        "C16": {"status": "EXPERIMENTAL_GATE_EXCEPTION", "source": "prior verified C16 continuation"},
        "C17": {"status": campaign_report["candidates"]["C17"]["qualification"],
                "result_sha256": campaign_report["candidates"]["C17"]["result_sha256"]},
        "C11": {"status": campaign_report["candidates"]["C11"]["qualification"],
                "result_sha256": campaign_report["candidates"]["C11"]["result_sha256"]},
        "R22_ORIGINAL": {"status": "QUALIFIED", "source": "prior verified R22_ORIGINAL continuation"},
        "R21": {"status": campaign_report["candidates"]["R21"]["qualification"],
                "result_sha256": campaign_report["candidates"]["R21"]["result_sha256"]},
        "R11": {"status": "EXPERIMENTAL_GATE_EXCEPTION", "performance_gate_outcome": "UNKNOWN_NOT_EVALUATED",
                "reason": "Continuation model bundles were not persisted; no R11 continuation scores were opened."},
        "R06_ORIGINAL": {"status": campaign_report["candidates"]["R06"]["qualification"],
                         "result_sha256": campaign_report["candidates"]["R06"]["result_sha256"]},
    }


def _kaggle_runtime_inventory() -> dict[str, Any]:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    env_names = sorted(name for name in os.environ if name.upper().startswith("KAGGLE_"))
    return {
        "cli_path": shutil.which("kaggle"),
        "python_module_available": importlib.util.find_spec("kaggle") is not None,
        "standard_token_file_present": token_path.is_file(),
        "kaggle_environment_variable_names": env_names,
        "credential_values_read": False,
        "live_api_checked": False,
    }


def _resource_estimate(card: str, train: pd.DataFrame) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    available = int(psutil.virtual_memory().available)
    n_rows = len(train)
    if card in {"C17", "C11", "R06_ORIGINAL"}:
        params: list[Mapping[str, Any]] = []
        if card in {"C17", "C11"}:
            params.append(c11.WORD_TFIDF_PARAMS)
        if card in {"C11", "R06_ORIGINAL"}:
            params.append(c11.CHAR_TFIDF_PARAMS if card == "C11" else r06.CHAR_TFIDF_PARAMETERS)
        unique_terms = 0
        text_width = 0
        for param in params:
            vectorizer = TfidfVectorizer(**dict(param))
            analyzer = vectorizer.build_analyzer()
            count = sum(len(set(analyzer(str(value) if not pd.isna(value) else "")))
                        for value in train.signal_stream.tolist())
            width = int(param.get("max_features") or count)
            unique_terms += min(count, n_rows * width)
            text_width += width
        nnz = unique_terms + n_rows * (STRUCTURED_NNZ_UPPER_PER_ROW + 1)
        csr_bytes = nnz * 12 + (n_rows + 1) * 8
        width = text_width + STRUCTURED_NNZ_UPPER_PER_ROW + 1
        hist_bytes = width * 256 * 16 * THREADS * 2
        estimate = 3 * csr_bytes + hist_bytes + LIGHTGBM_MEMORY_OVERHEAD_BYTES
        method = "upper-bound CSR copies plus LightGBM histogram and fixed overhead; analyzer only, no vectorizer fit"
    else:
        width = 807 + 32 + 1
        dense_bytes = n_rows * width * 8
        hist_bytes = width * 256 * 16 * THREADS * 2
        estimator_bytes = (500_000_000 if card == "R11" else 1_500_000_000)
        estimate = 4 * dense_bytes + hist_bytes + estimator_bytes + 1_000_000_000
        method = "conservative dense SVD design copies, histogram allowance, and estimator/head storage"
    allowed = min(JOB_CAP_BYTES, max(0, available - FREE_RAM_RESERVE_BYTES))
    return {"available_bytes": available, "allowed_bytes": allowed,
            "estimated_peak_bytes": int(estimate), "estimate_is_observed_peak": False,
            "threads": THREADS, "single_training_job": True,
            "free_reserve_bytes": FREE_RAM_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
            "method": method, "status": "PASS" if estimate <= allowed else "BLOCKED_RESOURCE"}


def _verify_budget_and_history() -> dict[str, Any]:
    budget = campaign._ledger_budget()
    reservations = {key: (int(value[0]), str(value[1])) for key, value in budget["reservation_keys"].items()}
    if reservations.get(NEW_RESERVATION_KEY) != (65, campaign.NEW_RESERVATION_CARD):
        raise SubmissionPrepError("Expanded benchmark's original 65-fit reservation is missing or changed")
    if reservations.get(RECOVERY_RESERVATION_KEY) != (8, recovery.RESERVATION_CARD):
        raise SubmissionPrepError("Serialization recovery's 8-fit reservation is missing or changed")
    total_with_prior = int(budget["reserved_predictor_fits"]) + PREVIOUS_OFFLEDGER_FITS
    if total_with_prior > GLOBAL_CAP:
        raise SubmissionPrepError("Global predictor-fit cap would be exceeded")
    prior_report = _read_json(previous.REPORT_ROOT / "SUBMISSION_STATUS.json")
    if prior_report.get("live_kaggle_preflight", {}).get("upload_attempted") is not False:
        raise SubmissionPrepError("C16/R22 prior upload status is ambiguous; refusing possible duplicate preparation")
    return {
        "global_cap": GLOBAL_CAP,
        "active_reserved_predictor_fits": budget["reserved_predictor_fits"],
        "original_expanded_reservation": 65,
        "recovery_reservation": 8,
        "previous_offledger_c16_r22_fulltrain_fits": PREVIOUS_OFFLEDGER_FITS,
        "current_effective_total_including_prior": total_with_prior,
        "remaining_capacity_after_reserved_and_prior": GLOBAL_CAP - total_with_prior,
        "new_fulltrain_fits_included_in_original_65": 13,
        "new_fulltrain_fit_plan": FIT_COUNTS,
        "new_fulltrain_fit_total": sum(FIT_COUNTS.values()),
        "r11_continuation_retry": "PROHIBITED; requires 36 fits, exceeds current 3-fit headroom",
        "ledger_sha256": budget["sha256"],
        "ledger_event_rows": budget["event_rows"],
        "test_or_public_score_used_for_selection": False,
    }


def _train_contract() -> dict[str, Any]:
    parent_identity, cls_context, reg_context, parent_preflight = recovery._verify_recovery_freeze()
    inputs = _input_paths_and_hashes()
    trains: dict[str, Any] = {}
    for task in ("classification", "regression"):
        frame, audit = _load_train_only(task)
        trains[task] = audit
        target_name = audit["target"]
        features = c17.FEATURE_NAMES if task == "classification" else r21.FEATURE_NAMES
        if set(features) & {"record_id", "artifact_signal", "threat_rank"}:
            raise SubmissionPrepError(f"{task} feature allowlist contains an identity or target")
        if not set(features).issubset(frame.columns):
            raise SubmissionPrepError(f"{task} official Train lacks a frozen feature")
        if task == "regression":
            target = frame[target_name].to_numpy(dtype=np.float64)
            support = r11.threshold_support(target)
            if np.any(target == 0.0) or any(not item["supported"] for item in support.values()):
                raise SubmissionPrepError("R11 full-Train boundary or threshold-support assumptions fail")
        if task == "classification" and len(cls_context["frame"]) != 50_608:
            raise SubmissionPrepError("Frozen classification Train context differs")
        if task == "regression" and len(reg_context["frame"]) != 50_608:
            raise SubmissionPrepError("Frozen regression Train context differs")
    eval_report_path = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/CAMPAIGN_REPORT.json"
    eval_report = _read_json(eval_report_path)
    reuse = campaign._submission_reuse_audit()
    return {
        "parent_freeze_signature": campaign._read_json(campaign.FREEZE_PATH)["signature"],
        "recovery_freeze_signature": recovery.json_sha256(recovery._read_json(recovery.FREEZE_PATH)["identity"]),
        "recovery_run_receipt_sha256": file_sha256(recovery.RUN_REPORT_PATH),
        "recovery_status": recovery._read_json(recovery.RUN_REPORT_PATH).get("status"),
        "final_continuation_evaluation_sha256": file_sha256(eval_report_path),
        "continuation_candidates": eval_report["evaluated_candidates"],
        "r11_continuation_blocker": eval_report["blocked_candidates"]["R11"],
        "qualification_evidence": _qualification_evidence(),
        "existing_c16_r22_reuse_audit": reuse,
        "new_continuation_fulltrain_plan": [
            {"candidate": card, "predictor_fits": FIT_COUNTS[card],
             "recipe": _candidate_recipe(card)} for card in NEW_CANDIDATES
        ],
        "official_train": trains,
        "official_input_hashes_including_test_and_sample_bytes_only": inputs,
        "test_values_opened": False,
        "sample_submission_values_opened": False,
        "budget": _verify_budget_and_history(),
        "kaggle_runtime_inventory": _kaggle_runtime_inventory(),
        "kaggle_live_rules_quota_ledger_checked": False,
        "submission_uploads_attempted": False,
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost",
            "threadpoolctl", "psutil", "catboost")},
        "source_sha256": _source_hashes(),
        "source_commit": _git("rev-parse", "HEAD"),
        "status": "FROZEN_PREPARED_FOR_SERIAL_FULLTRAIN_FITS",
    }


def freeze() -> dict[str, Any]:
    if FREEZE_PATH.exists():
        return verify_freeze()
    identity = _train_contract()
    signature = json_sha256(identity)
    output = {"campaign_id": CAMPAIGN_ID, "identity": identity, "signature": signature,
              "status": "FROZEN_BEFORE_FULLTRAIN_FITS", "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(FREEZE_PATH, output)
    return output


def verify_freeze() -> dict[str, Any]:
    frozen = _read_json(FREEZE_PATH)
    identity = frozen.get("identity", {})
    if (frozen.get("campaign_id") != CAMPAIGN_ID
            or frozen.get("status") != "FROZEN_BEFORE_FULLTRAIN_FITS"
            or frozen.get("signature") != json_sha256(identity)):
        raise SubmissionPrepError("Expanded submission execution freeze is invalid")
    if identity.get("source_sha256") != _source_hashes():
        raise SubmissionPrepError("Training/inference source code changed after freeze")
    current_head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", identity.get("source_commit", ""), current_head],
        capture_output=True, text=True, check=False,
    )
    if ancestor.returncode != 0:
        raise SubmissionPrepError("Frozen preparation commit is not an ancestor of current HEAD")
    expected_budget = identity.get("budget", {})
    observed_budget = _verify_budget_and_history()
    dynamic_ledger_fields = {"ledger_sha256", "ledger_event_rows"}
    if any(expected_budget.get(key) != value for key, value in observed_budget.items()
           if key not in dynamic_ledger_fields):
        raise SubmissionPrepError("Global fit reservation or submission history changed after freeze")
    if identity.get("qualification_evidence") != _qualification_evidence():
        raise SubmissionPrepError("Continuation results or qualification evidence changed after freeze")
    return frozen


def preflight() -> dict[str, Any]:
    frozen = verify_freeze()
    identity = frozen["identity"]
    output: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "freeze_signature": frozen["signature"],
        "execution_head": identity["source_commit"],
        "test_values_opened": False,
        "submission_uploads_attempted": False,
        "resource_preflight": {},
        "status": "READY_FOR_SERIAL_FITS",
    }
    for card in NEW_CANDIDATES:
        train, _audit = _load_train_only(TASKS[card])
        estimate = _resource_estimate(card, train)
        output["resource_preflight"][card] = estimate
    if any(item["status"] != "PASS" for item in output["resource_preflight"].values()):
        output["status"] = "CANDIDATE_SPECIFIC_RESOURCE_BLOCKS"
    _write_once(REPORT_ROOT / "PREFLIGHT.json", output)
    return output


def _load_test_only(task: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    data_dir = initial.resolve_data_dir()
    paths = initial.task_files(task, data_dir)
    hashes = {role: file_sha256(Path(path)) for role, path in paths.items()}
    if hashes != initial.EXPECTED_HASHES[task]:
        raise SubmissionPrepError(f"Official {task} input hashes changed after freeze")
    test, sample = pd.read_csv(paths["test"]), pd.read_csv(paths["sample"])
    if (len(test) != 29_421 or len(sample) != 29_421
            or test.record_id.isna().any() or sample.record_id.isna().any()
            or not test.record_id.is_unique or not sample.record_id.is_unique
            or not test.record_id.reset_index(drop=True).equals(sample.record_id.reset_index(drop=True))):
        raise SubmissionPrepError(f"Official {task} Test IDs/order do not match SampleSubmission")
    expected_columns = ["record_id", "artifact_signal" if task == "classification" else "threat_rank"]
    if list(sample.columns) != expected_columns:
        raise SubmissionPrepError(f"Official {task} SampleSubmission columns changed")
    return test, sample, hashes


def _check_fit_state(card: str, state: Mapping[str, Any], train: pd.DataFrame, target: np.ndarray) -> None:
    if int(state.get("fit_rows", 0)) != 50_608:
        raise SubmissionPrepError(f"{card} fitted-state row count is not the full official Train")
    if state.get("test_data_used") is True or state.get("test_data_used_for_fit") is True:
        raise SubmissionPrepError(f"{card} fit-state reports Test use for training")
    if state.get("validation_labels_passed_to_fit") is True or state.get("validation_data_passed_to_fit") is True:
        raise SubmissionPrepError(f"{card} fit-state reports validation leakage")
    cycles = pd.to_numeric(train.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (int(cycles.min()) != 47 or int(cycles.max()) != 79
            or int(state.get("fit_cycle_min", 47)) != 47
            or int(state.get("fit_cycle_max", 79)) != 79):
        raise SubmissionPrepError(f"{card} fit-state cycle range is not 47-79")
    if not np.isfinite(np.asarray(target, dtype=np.float64)).all():
        raise SubmissionPrepError(f"{card} fit target is nonfinite")


def _fit_model(card: str, train: pd.DataFrame, target: np.ndarray) -> tuple[Any, dict[str, Any], np.ndarray]:
    if card == "C17":
        class FullTrainC17(c17.LongerRegularizedSparseClassifier):
            def __init__(self) -> None:
                self.cutoff, self._fitted = 79, False
        model = FullTrainC17().fit(train.loc[:, list(c17.FEATURE_NAMES)], target.astype(np.int8))
        state = model.audit_state()
        return model, state, train.loc[:, list(c17.FEATURE_NAMES)].copy()
    if card == "C11":
        model = c11.WordCharSparseClassifier().fit(train.loc[:, list(c11.FEATURE_NAMES)], target.astype(np.int8))
        state = model.audit_state()
        return model, state, train.loc[:, list(c11.FEATURE_NAMES)].copy()
    if card == "R21":
        class FullTrainR21(r21.XGBoostSVD32Regressor):
            def __init__(self) -> None:
                self.cutoff, self._fitted = 79, False
                self._fit_warnings, self._state = [], {}
        features = train.loc[:, list(r21.FEATURE_NAMES)]
        model = FullTrainR21().fit(features, target.astype(np.float64))
        state = model.audit_state()
        return model, state, features
    if card == "R11":
        features = train.loc[:, list(r11.FEATURE_NAMES)]
        model = r11.CumulativeOrdinalSVD32().fit(features, target.astype(np.float64))
        state = model.audit_state()
        if int(state.get("head_fit_calls", 0)) != 9:
            raise SubmissionPrepError("R11 did not fit exactly nine frozen ordinal heads")
        return model, state, features
    if card == "R06_ORIGINAL":
        features = train.loc[:, list(r06.FEATURE_NAMES)]
        model = r06.R06CharacterSparseL1Hybrid().fit(features, target.astype(np.float64))
        state = dict(model.state)
        return model, state, features
    raise SubmissionPrepError(f"Candidate not in the authorized list: {card}")


def _predict_and_compose(card: str, model: Any, features: pd.DataFrame, train: pd.DataFrame,
                         test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if card in {"C17", "C11"}:
        base_probability, missing_probability, missing_mask, reuse = previous._verify_classification_reuse(train, test)
        feature_names = c17.FEATURE_NAMES if card == "C17" else c11.FEATURE_NAMES
        score_features = test.loc[:, list(feature_names)]
        raw = (model.predict_probability(score_features) if card == "C17"
               else model.predict_proba(score_features))
        probability = np.where(missing_mask, missing_probability,
                               0.75 * base_probability + 0.25 * raw)
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise SubmissionPrepError(f"{card} routed probabilities are invalid")
        threshold = c17.THRESHOLD if card == "C17" else 0.50
        labels = (probability >= threshold).astype(np.int64)
        if not np.array_equal(probability[missing_mask], missing_probability[missing_mask]):
            raise SubmissionPrepError(f"{card} missing-stream fallback differs from CB_MISSING")
        return (
            labels,
            {"threshold": threshold, "availability_route": "present .75*C2_B+.25*candidate; missing CB_MISSING",
             "missing_rows": int(missing_mask.sum()), "present_rows": int((~missing_mask).sum()),
             "reused_full_train_components": reuse,
             "probability_sha256": hashlib.sha256(probability.astype("<f8").tobytes()).hexdigest()},
            {"raw_candidate": raw, "c2b_probability": base_probability,
             "cb_missing_probability": missing_probability, "stream_missing": missing_mask,
             "probability": probability, "labels": labels},
        )

    champion, champion_audit = previous._verify_regression_reuse(test)
    cycles_float = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(cycles_float).all() or not np.equal(cycles_float, np.floor(cycles_float)).all():
        raise SubmissionPrepError("Official Test cycle_code must be finite integer values")
    cycles = cycles_float.astype(np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)):
        raise SubmissionPrepError("Regression rank mapping requires complete cycles 80-91")
    feature_names = {"R21": r21.FEATURE_NAMES, "R11": r11.FEATURE_NAMES,
                     "R06_ORIGINAL": r06.FEATURE_NAMES}[card]
    score_features = test.loc[:, list(feature_names)]
    if card == "R21":
        raw = model.predict(score_features)
        final, _baseline, mapped = r21.reconstruct_r21(raw, champion, cycles)
    elif card == "R11":
        raw = model.predict(score_features)
        mapped = r11.rank50_prediction(raw, cycles)
        final = r11.compose_r11_prediction(champion, raw, cycles)
    else:
        raw = model.predict(score_features)
        mapped = r06.rank50_prediction(raw, cycles)
        final = r06.compose_r06_prediction(champion, raw, cycles)
    if (final.shape != (len(test),) or not np.isfinite(final).all()
            or np.any((final < 0.0) | (final > 100.0))):
        raise SubmissionPrepError(f"{card} final regression predictions are invalid")
    rank_rebuild = (r21.rank50_prediction(raw, cycles) if card == "R21" else
                    r11.rank50_prediction(raw, cycles) if card == "R11" else
                    r06.rank50_prediction(raw, cycles))
    if not np.array_equal(mapped, rank_rebuild):
        raise SubmissionPrepError(f"{card} rank mapping reconstruction differs")
    return (
        final,
        {"champion_reuse": champion_audit,
         "mapping": _candidate_recipe(card).get("mapping", _candidate_recipe(card).get("postprocessing")),
         "cycle_rows": {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)},
         "mapped_component_range": [float(mapped.min()), float(mapped.max())],
         "prediction_range": [float(final.min()), float(final.max())],
         "prediction_sha256": hashlib.sha256(final.astype("<f8").tobytes()).hexdigest()},
        {"champion": champion, "raw_candidate": raw, "rank_mapped_candidate": mapped,
         "cycle_code": cycles, "prediction": final},
    )


def _manifest_path(card: str) -> Path:
    return REPORT_ROOT / card / "manifest.json"


def _attempt_path(card: str) -> Path:
    return RUN_ROOT / card / "attempt.json"


def _verify_existing_manifest(card: str) -> dict[str, Any]:
    manifest = _read_json(_manifest_path(card))
    csv_path = ROOT / manifest["csv"]["path_gitignored"]
    array_path = ROOT / manifest["prediction_artifact"]["path_gitignored"]
    if (file_sha256(csv_path) != manifest["csv"]["sha256"]
            or file_sha256(array_path) != manifest["prediction_artifact"]["sha256"]):
        raise SubmissionPrepError(f"Existing {card} output hashes differ from its manifest")
    return manifest


def fit_candidate(card: str) -> dict[str, Any]:
    if card not in NEW_CANDIDATES:
        raise SubmissionPrepError(f"Candidate not in the authorized new-fit list: {card}")
    frozen = verify_freeze()
    manifest_path = _manifest_path(card)
    if manifest_path.exists():
        return _verify_existing_manifest(card)
    attempt_path = _attempt_path(card)
    if attempt_path.exists():
        raise SubmissionPrepError(f"{card} has an existing fit attempt; automatic retries are disabled")
    fits_used = 0
    for item in NEW_CANDIDATES:
        path = _attempt_path(item)
        if path.exists():
            attempt = _read_json(path)
            if attempt.get("fit_started") is True:
                fits_used += FIT_COUNTS[item]
    if fits_used + FIT_COUNTS[card] > sum(FIT_COUNTS.values()):
        raise SubmissionPrepError("This fit would exceed the frozen 13-predictor full-Train allowance")

    train, train_audit = _load_train_only(TASKS[card])
    target_name = train_audit["target"]
    target = train[target_name].to_numpy(dtype=np.int8 if TASKS[card] == "classification" else np.float64)
    resources = _resource_estimate(card, train)
    if resources["status"] != "PASS":
        raise SubmissionPrepError(f"{card} full-Train fit blocked before estimator start: {resources}")

    fit_started = False
    started = None
    attempt: dict[str, Any] = {}
    try:
        with exclusive_training_slot(TRAINING_LOCK):
            available_now = int(psutil.virtual_memory().available)
            allowed_now = min(JOB_CAP_BYTES, max(0, available_now - FREE_RAM_RESERVE_BYTES))
            resources = {**resources, "available_bytes": available_now, "allowed_bytes": allowed_now,
                         "status": "PASS" if int(resources["estimated_peak_bytes"]) <= allowed_now
                         else "BLOCKED_RESOURCE"}
            if resources["status"] != "PASS":
                raise SubmissionPrepError(f"{card} resource preflight changed before locked fit: {resources}")
            attempt = {"campaign_id": CAMPAIGN_ID, "candidate": card, "fit_predictors": FIT_COUNTS[card],
                       "fit_started": True, "freeze_signature": frozen["signature"],
                       "started_at_utc": datetime.now(timezone.utc).isoformat(),
                       "source_commit": frozen["identity"]["source_commit"]}
            _write_once(attempt_path, attempt)
            append_event(LEDGER_PATH, {"event": "EXPANDED_SUBMISSION_FIT_STARTED", "campaign_id": CAMPAIGN_ID,
                                       "candidate": card, "predictor_fits": FIT_COUNTS[card],
                                       "freeze_signature": frozen["signature"],
                                       "source_commit": attempt["source_commit"]})
            fit_started = True
            started = perf_counter()
            with threadpool_limits(limits=THREADS):
                model, fit_state, fit_features = _fit_model(card, train, target)
            fit_seconds = perf_counter() - started
            _check_fit_state(card, fit_state, train, target)
            predictor_count = int(fit_state.get("head_fit_calls", 9) if card == "R11" else 1)
            if predictor_count != FIT_COUNTS[card]:
                raise SubmissionPrepError(f"{card} fitted {predictor_count} predictors, expected {FIT_COUNTS[card]}")
            # Persist the Train-only fitted-state receipt before reading Test values.
            state_audit_path = RUN_ROOT / card / "fit_state.json"
            _write_once(state_audit_path, {"candidate": card, "fit_state": fit_state,
                                           "train_audit": train_audit, "test_values_opened": False,
                                           "predictor_fits": predictor_count})
            test, sample, input_hashes = _load_test_only(TASKS[card])
            prediction, prediction_audit, arrays = _predict_and_compose(card, model, fit_features, train, test)
            target_column = "artifact_signal" if TASKS[card] == "classification" else "threat_rank"
            task = TASKS[card]
            frame = initial._submission_frame(test, sample, prediction, target_column, task)
            output_card = card.lower()
            csv_path = CSV_ROOT / f"{output_card}.csv"
            csv_info = previous._write_csv_once(csv_path, frame)
            npz_path = RUN_ROOT / card / "predictions.npz"
            npz_sha = previous._save_npz_once(npz_path, **arrays)
            manifest = {
                "campaign_id": CAMPAIGN_ID, "candidate": card, "task": task,
                "qualification": frozen["identity"]["qualification_evidence"][card],
                "status": "PREPARED_VALIDATED_NOT_SUBMITTED",
                "source_commit": attempt["source_commit"],
                "execution_head": _git("rev-parse", "HEAD"),
                "execution_freeze_sha256": file_sha256(FREEZE_PATH),
                "execution_freeze_signature": frozen["signature"],
                "recipe": _candidate_recipe(card), "fit_state": fit_state,
                "fit_audit": {"fit_rows": int(len(train)), "fit_cycle_min": 47, "fit_cycle_max": 79,
                              "predictor_fits": predictor_count, "threads": THREADS,
                              "fit_seconds": fit_seconds,
                              "resource_preflight_at_fit": resources,
                              "test_labels_passed_to_fit": False,
                              "validation_labels_passed_to_fit": False,
                              "preprocessing_fit_on_official_train_only": True,
                              "train_input": train_audit,
                              "fitted_state_path_gitignored": str(state_audit_path.relative_to(ROOT)).replace("\\", "/"),
                              "fitted_state_sha256": file_sha256(state_audit_path)},
                "prediction_audit": prediction_audit,
                "official_input_hashes": input_hashes,
                "test_rows": len(test), "sample_submission_rows": len(sample),
                "csv": csv_info,
                "prediction_artifact": {"path_gitignored": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
                                        "sha256": npz_sha},
                "submission": {"attempted": False, "submission_ref": None, "public_score": None,
                               "reason": "No authenticated Kaggle API/CLI runtime; live rules, quota and ledger not checked."},
                "test_used_only_for_inference": True,
            }
            _write_once(manifest_path, manifest)
        append_event(LEDGER_PATH, {"event": "EXPANDED_SUBMISSION_FIT_FINISHED", "campaign_id": CAMPAIGN_ID,
                                   "candidate": card, "predictor_fits": FIT_COUNTS[card],
                                   "manifest_sha256": file_sha256(manifest_path),
                                   "freeze_signature": frozen["signature"]})
        return manifest
    except Exception as exc:
        if fit_started:
            append_event(LEDGER_PATH, {"event": "EXPANDED_SUBMISSION_FIT_FAILED", "campaign_id": CAMPAIGN_ID,
                                       "candidate": card, "predictor_fits": FIT_COUNTS[card],
                                       "error_type": type(exc).__name__, "error": str(exc),
                                       "freeze_signature": frozen["signature"]})
        raise


def status() -> dict[str, Any]:
    frozen = verify_freeze()
    candidates: dict[str, Any] = {}
    for card in ("C16", "C17", "C11", "R22_ORIGINAL", "R21", "R11", "R06_ORIGINAL"):
        if card in NEW_CANDIDATES and _manifest_path(card).is_file():
            manifest = _verify_existing_manifest(card)
            candidates[card] = {"status": manifest["status"], "qualification": manifest["qualification"],
                                "csv_sha256": manifest["csv"]["sha256"], "submission_ref": None,
                                "public_score": None}
        elif card in {"C16", "R22_ORIGINAL"}:
            old = _read_json(previous.REPORT_ROOT / "SUBMISSION_STATUS.json")
            record = old["classification" if card == "C16" else "regression"]
            candidates[card] = {"status": record["status"],
                                "qualification": frozen["identity"]["qualification_evidence"][card],
                                "csv_sha256": record["csv"]["sha256"],
                                "submission_ref": None, "public_score": None}
        else:
            candidates[card] = {"status": "NOT_PREPARED", "qualification": frozen["identity"]["qualification_evidence"].get(card),
                                "submission_ref": None, "public_score": None}
    output = {"campaign_id": CAMPAIGN_ID, "status": "PREPARED_WITH_AUTH_BLOCKER",
              "candidates": candidates, "kaggle_runtime_inventory": _kaggle_runtime_inventory(),
              "live_rules_team_deadline_quota_and_shared_ledger_checked": False,
              "new_uploads_attempted": 0,
              "upload_blocker": "Authenticated Kaggle API/CLI unavailable; no upload was attempted.",
              "freeze_sha256": file_sha256(FREEZE_PATH), "freeze_signature": frozen["signature"]}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "preflight", "fit", "status"))
    parser.add_argument("candidate", nargs="?")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze()
    elif args.action == "preflight":
        result = preflight()
    elif args.action == "fit":
        if not args.candidate:
            raise SystemExit("fit requires one authorized candidate ID")
        result = fit_candidate(args.candidate)
    else:
        result = status()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
