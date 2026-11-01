"""Read-only seven-candidate CSV and prediction-recipe verification.

This verifier uses official Test only to check IDs, cycle completeness and
inference outputs after all full-Train fits have completed. It never fits or
contacts Kaggle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import optimization_v3_expanded_prepare_submissions as prep
from scripts import prepare_initial_submissions as initial
from src.data.audit import sha256 as file_sha256


ROOT = prep.ROOT
QA_PATH = prep.REPORT_ROOT / "FINAL_QA.json"
EXPECTED = ("C16", "C17", "C11", "R22_ORIGINAL", "R21", "R11", "R06_ORIGINAL")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"Prediction artifact missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _rank50(raw: Any, cycles: Any) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    groups = np.asarray(cycles, dtype=np.int64)
    if values.ndim != 1 or values.shape != groups.shape or not np.isfinite(values).all():
        raise RuntimeError("Saved rank50 inputs are misaligned or nonfinite")
    mapped = np.empty_like(values)
    for cycle in np.unique(groups):
        mask = groups == cycle
        percentile = 100.0 * rankdata(values[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * values[mask] + 0.5 * percentile
    return np.clip(mapped, 0.0, 100.0)


def _inputs(task: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = initial.task_files(task, initial.resolve_data_dir())
    hashes = {key: file_sha256(Path(path)) for key, path in paths.items()}
    if hashes != initial.EXPECTED_HASHES[task]:
        raise RuntimeError(f"{task} official raw input hashes changed")
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    target = "artifact_signal" if task == "classification" else "threat_rank"
    if (len(test) != 29_421 or len(sample) != 29_421
            or list(sample.columns) != ["record_id", target]
            or test.record_id.isna().any() or sample.record_id.isna().any()
            or not test.record_id.is_unique or not sample.record_id.is_unique
            or not test.record_id.reset_index(drop=True).equals(sample.record_id.reset_index(drop=True))):
        raise RuntimeError(f"{task} official Test/SampleSubmission IDs or schema failed")
    return test, sample, hashes


def _record(card: str, task: str, path: Path, sample: pd.DataFrame,
            predictions: np.ndarray, fit_rows: int, predictor_fits: int,
            qualification: Any, input_hashes: dict[str, str],
            arrays_path: Path, recipe_check: str) -> dict[str, Any]:
    csv = pd.read_csv(path, float_precision="round_trip")
    target = "artifact_signal" if task == "classification" else "threat_rank"
    if (list(csv.columns) != list(sample.columns) or len(csv) != 29_421
            or csv.record_id.isna().any() or not csv.record_id.is_unique
            or not csv.record_id.reset_index(drop=True).equals(sample.record_id.reset_index(drop=True))):
        raise RuntimeError(f"{card} CSV schema, count, or official ID order failed")
    values = csv[target].to_numpy()
    if task == "classification":
        if (not pd.api.types.is_integer_dtype(csv[target].dtype)
                or not np.isin(values, [0, 1]).all()
                or not np.array_equal(values.astype(np.int64), predictions.astype(np.int64))):
            raise RuntimeError(f"{card} classification labels are not exact integer 0/1 predictions")
        value_range = [int(values.min()), int(values.max())]
    else:
        values = values.astype(np.float64)
        if (not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0))
                or not np.allclose(values, predictions.astype(np.float64), atol=1e-12, rtol=0)):
            raise RuntimeError(f"{card} regression predictions are nonfinite, out of range, or differ from saved outputs")
        value_range = [float(values.min()), float(values.max())]
    if int(fit_rows) != 50_608:
        raise RuntimeError(f"{card} fitted-state population is not 50,608 Train rows")
    return {
        "candidate": card, "task": task, "status": "PASS",
        "qualification": qualification, "fit_rows": int(fit_rows),
        "predictor_fits": int(predictor_fits), "csv_path_gitignored": str(path.relative_to(ROOT)).replace("\\", "/"),
        "csv_sha256": file_sha256(path), "prediction_artifact_sha256": file_sha256(arrays_path),
        "rows": len(csv), "columns": list(csv.columns),
        "ordered_ids_sha256": prep.previous._ids_sha(csv.record_id),
        "prediction_range": value_range, "recipe_reconstruction": recipe_check,
        "official_input_sha256": input_hashes,
        "test_target_or_labels_read": False,
    }


def verify() -> dict[str, Any]:
    frozen = prep.verify_freeze()
    old = prep._read_json(prep.previous.REPORT_ROOT / "SUBMISSION_STATUS.json")
    class_test, class_sample, class_hashes = _inputs("classification")
    reg_test, reg_sample, reg_hashes = _inputs("regression")
    if not class_test.record_id.reset_index(drop=True).equals(reg_test.record_id.reset_index(drop=True)):
        raise RuntimeError("Classification and regression official Test IDs differ")
    reports: dict[str, Any] = {}

    # Prior one-time full-Train submissions, reused byte-for-byte.
    for card, task, key, output_name in (
        ("C16", "classification", "classification", "classification_c16.csv"),
        ("R22_ORIGINAL", "regression", "regression", "regression_r22_original.csv"),
    ):
        record = old[key]
        test, sample, hashes = ((class_test, class_sample, class_hashes) if task == "classification"
                                else (reg_test, reg_sample, reg_hashes))
        csv_path = ROOT / record["csv"]["path_gitignored"]
        run_dir = ROOT / "experiments/runs/c16_r22_submission"
        arrays_path = run_dir / ("classification_c16_outputs.npz" if card == "C16"
                                 else "regression_r22_original_outputs.npz")
        arrays = _load_npz(arrays_path)
        state = record["fit_state"]
        if task == "classification":
            mask = arrays["stream_missing"].astype(bool)
            official_mask = prep.previous.stream_missing_mask(class_test.signal_stream)
            if not np.array_equal(mask, official_mask):
                raise RuntimeError("C16 full-Train stream availability mask differs from official Test")
            probability = np.where(mask, arrays["missing"], 0.75 * arrays["base"] + 0.25 * arrays["c16_raw"])
            labels = (probability >= 0.50).astype(np.int64)
            if (not np.allclose(probability, arrays["probability"], atol=1e-12, rtol=0)
                    or not np.array_equal(labels, arrays["labels"])):
                raise RuntimeError("C16 availability route reconstruction failed")
            predictions = labels
            check = "C16 probability blend, exact missing fallback, threshold 0.50"
            fits = 1
        else:
            cycles = arrays["test_cycle_code"].astype(np.int64)
            official_cycles = pd.to_numeric(reg_test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
            if (not np.array_equal(cycles, official_cycles)
                    or not np.array_equal(np.unique(cycles), np.arange(80, 92))):
                raise RuntimeError("R22_ORIGINAL saved cycles differ from complete official Test cycles")
            mapped = _rank50(arrays["raw_xgboost"], cycles)
            final = np.clip(0.75 * arrays["champion"] + 0.25 * mapped, 0.0, 100.0)
            if (not np.allclose(mapped, arrays["rank50_component"], atol=1e-12, rtol=0)
                    or not np.allclose(final, arrays["prediction"], atol=1e-12, rtol=0)):
                raise RuntimeError("R22_ORIGINAL rank50 blend reconstruction failed")
            predictions = final
            check = "XGBoost rank50 once per complete cycle, champion blend, final clip"
            fits = 1
        if (record["csv"]["sha256"] != file_sha256(csv_path)
                or state.get("fit_rows") != 50_608
                or state.get("test_data_used") is True
                or state.get("test_data_used_for_fit") is True):
            raise RuntimeError(f"{card} prior CSV hash or fit-state lineage failed")
        reports[card] = _record(card, task, csv_path, sample, predictions, int(state["fit_rows"]),
                                fits, prep._qualification_evidence()[card], hashes, arrays_path, check)

    # Five newly prepared candidates.
    for card in prep.NEW_CANDIDATES:
        manifest = prep._read_json(prep._manifest_path(card))
        task = prep.TASKS[card]
        test, sample, hashes = ((class_test, class_sample, class_hashes) if task == "classification"
                                else (reg_test, reg_sample, reg_hashes))
        csv_path = ROOT / manifest["csv"]["path_gitignored"]
        arrays_path = ROOT / manifest["prediction_artifact"]["path_gitignored"]
        arrays = _load_npz(arrays_path)
        state = manifest["fit_state"]
        if manifest.get("candidate") != card or manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED":
            raise RuntimeError(f"{card} manifest identity/status failed")
        if task == "classification":
            mask = arrays["stream_missing"].astype(bool)
            official_mask = prep.previous.stream_missing_mask(class_test.signal_stream)
            if not np.array_equal(mask, official_mask):
                raise RuntimeError(f"{card} availability mask differs from official Test")
            probability = np.where(mask, arrays["cb_missing_probability"],
                                   0.75 * arrays["c2b_probability"] + 0.25 * arrays["raw_candidate"])
            labels = (probability >= 0.50).astype(np.int64)
            if (not np.allclose(probability, arrays["probability"], atol=1e-12, rtol=0)
                    or not np.array_equal(labels, arrays["labels"])):
                raise RuntimeError(f"{card} availability route reconstruction failed")
            if int(arrays["stream_missing"].sum()) != int(manifest["prediction_audit"]["missing_rows"]):
                raise RuntimeError(f"{card} stream availability count differs from manifest")
            predictions = labels
            check = "C2_B blend, exact CB_MISSING fallback, threshold 0.50"
        else:
            cycles = arrays["cycle_code"].astype(np.int64)
            observed_cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
            if not np.array_equal(cycles, observed_cycles) or not np.array_equal(np.unique(cycles), np.arange(80, 92)):
                raise RuntimeError(f"{card} full-cycle Test rank population differs")
            mapped = _rank50(arrays["raw_candidate"], cycles)
            final = np.clip(0.75 * arrays["champion"] + 0.25 * mapped, 0.0, 100.0)
            if (not np.allclose(mapped, arrays["rank_mapped_candidate"], atol=1e-12, rtol=0)
                    or not np.allclose(final, arrays["prediction"], atol=1e-12, rtol=0)):
                raise RuntimeError(f"{card} rank50 and blend reconstruction failed")
            predictions = final
            check = "rank50 average-tie mapping once per complete cycle, three-component champion blend, clip [0,100]"
        fit_audit = manifest["fit_audit"]
        if (fit_audit.get("fit_rows") != 50_608 or fit_audit.get("test_labels_passed_to_fit") is not False
                or fit_audit.get("validation_labels_passed_to_fit") is not False
                or fit_audit.get("preprocessing_fit_on_official_train_only") is not True
                or file_sha256(csv_path) != manifest["csv"]["sha256"]
                or file_sha256(arrays_path) != manifest["prediction_artifact"]["sha256"]):
            raise RuntimeError(f"{card} fitted-state/source boundary or artifact hashes failed")
        reports[card] = _record(card, task, csv_path, sample, predictions,
                                int(fit_audit["fit_rows"]), int(fit_audit["predictor_fits"]),
                                manifest["qualification"], hashes, arrays_path, check)

    if set(reports) != set(EXPECTED):
        raise RuntimeError(f"Verified candidate scope differs: {tuple(reports)}")
    reports = {card: reports[card] for card in EXPECTED}
    return {
        "campaign_id": "optimization_v3_expanded_submissions_v1",
        "status": "PASS_ALL_SEVEN_CSVS_PREPARED_NOT_SUBMITTED",
        "freeze_signature": frozen["signature"],
        "verified_candidates": list(reports),
        "candidate_count": len(reports),
        "rows_each": 29_421,
        "new_predictor_fits": sum(prep.FIT_COUNTS.values()),
        "new_uploads_attempted": 0,
        "test_used_only_for_id_cycle_and_prediction_qa_after_fit": True,
        "test_target_or_labels_read": False,
        "kaggle_runtime_inventory": prep._kaggle_runtime_inventory(),
        "live_kaggle_rules_quota_team_deadline_ledger_checked": False,
        "public_scores": {card: None for card in EXPECTED},
        "candidates": reports,
    }


if __name__ == "__main__":
    result = verify()
    prep._write_once(QA_PATH, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
