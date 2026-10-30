"""Prepare one audited all-train initial submission per TP2 task.

This command writes ignored CSVs and safe manifests; it never submits anything
to Kaggle. Train/Test inputs remain read-only and are resolved from --data-dir
or TP2_DATA_DIR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.audit import sha256, write_json
from src.models.classification_round2_models import build_round2_model, stream_missing_mask
from src.models.round1_models import build_model
from src.models.round1_evaluation import digest_json as round1_digest_json
from src.utils.paths import PROJECT_ROOT, task_files


MAIN_CHECKPOINT = "12b515c7b93ba7b6c3ff88b1c8b068e347fb0e36"
EXPECTED_ROWS = {"train": 50_608, "test": 29_421}
COMPETITIONS = {
    "classification": "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027",
    "regression": "tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027",
}
MESSAGES = {
    "classification": "Samuel C4-HGB Routing Initial Benchmark",
    "regression": "Samuel R1-D Expanded HGB Initial Benchmark",
}
EXPECTED_HASHES = {
    "classification": {
        "train": "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c",
        "test": "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c",
        "sample": "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7",
    },
    "regression": {
        "train": "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082",
        "test": "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c",
        "sample": "63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32",
    },
}


def _json_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inputs(task: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = task_files(task, data_dir)
    if not all(paths.values()):
        raise FileNotFoundError(f"Required official {task} Train/Test/Sample CSV is missing")
    hashes = {role: sha256(path) for role, path in paths.items()}
    if hashes != EXPECTED_HASHES[task]:
        raise RuntimeError(f"{task} input hashes differ from the committed official-data receipts")
    train, test, sample = (pd.read_csv(paths[k]) for k in ("train", "test", "sample"))
    if len(train) != EXPECTED_ROWS["train"] or len(test) != EXPECTED_ROWS["test"]:
        raise ValueError(f"Unexpected {task} row counts: train={len(train)}, test={len(test)}")
    if train.record_id.isna().any() or not train.record_id.is_unique:
        raise ValueError("Train record_id must be complete and unique")
    if test.record_id.isna().any() or not test.record_id.is_unique:
        raise ValueError("Test record_id must be complete and unique")
    if sample.record_id.isna().any() or not sample.record_id.is_unique:
        raise ValueError("SampleSubmission record_id must be complete and unique")
    if not test.record_id.reset_index(drop=True).equals(sample.record_id.reset_index(drop=True)):
        raise ValueError("Official test IDs/order do not exactly match the sample submission")
    return train, test, sample, hashes


def _submission_frame(test: pd.DataFrame, sample: pd.DataFrame, predictions: np.ndarray,
                      target_name: str, task: str) -> pd.DataFrame:
    values = np.asarray(predictions)
    if values.ndim != 1 or len(values) != len(test):
        raise ValueError("Prediction vector has the wrong shape")
    if list(sample.columns) != ["record_id", target_name]:
        raise ValueError("SampleSubmission columns differ from the official output contract")
    if task == "classification":
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Classification predictions must be integer labels 0/1")
        values = values.astype(np.int64, copy=False)
    else:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
            raise ValueError("Regression predictions must be finite and clipped to [0,100]")
    result = pd.DataFrame({"record_id": test.record_id.to_numpy(), target_name: values})
    if list(result.columns) != list(sample.columns):
        raise ValueError("Submission header/order mismatch")
    if not result.record_id.equals(sample.record_id):
        raise ValueError("Submission record_id/order mismatch")
    if len(result) != len(test) or result.isna().any().any():
        raise ValueError("Submission must have exactly one complete row per official test row")
    return result


def _classification(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    target = train.artifact_signal.to_numpy()
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Train artifact_signal must be binary")
    present_recipe = {"id": "C2_B", "kind": "control", "source": "C2_B"}
    missing_recipe = {"id": "C1_B", "kind": "control", "source": "C1_B"}
    present = build_round2_model(present_recipe)
    missing = build_round2_model(missing_recipe)
    present_cols = present.input_feature_names
    missing_cols = missing.input_feature_names
    missing_test = stream_missing_mask(test)
    with threadpool_limits(limits=2):
        # Each route branch uses its complete corresponding training recipe on
        # all official labeled rows. Availability chooses only at prediction.
        present.fit(train.loc[:, present_cols], target)
        missing.fit(train.loc[:, missing_cols], target)
        probabilities = np.empty(len(test), dtype=float)
        probabilities[~missing_test] = present.predict_proba(test.loc[~missing_test, present_cols])
        probabilities[missing_test] = missing.predict_proba(test.loc[missing_test, missing_cols])
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classification probabilities must be finite and within [0,1]")
    labels = (probabilities >= 0.50).astype(np.int64)
    audit = {
        "route": "C4_HGB fixed availability route; present=C2_B hybrid LR; missing=C1_B structured HGB",
        "threshold": 0.50,
        "train_rows": len(train),
        "train_stream_missing": int(stream_missing_mask(train).sum()),
        "train_stream_present": int((~stream_missing_mask(train)).sum()),
        "test_stream_missing": int(missing_test.sum()),
        "test_stream_present": int((~missing_test).sum()),
        "present_fit_state": present.audit_state(),
        "missing_fit_state": missing.audit_state(),
        "probability_sha256": hashlib.sha256(np.asarray(probabilities, dtype="<f8").tobytes()).hexdigest(),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }
    return labels, audit


def _regression(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    protocol = json.loads((PROJECT_ROOT / "experiments/round1/protocol.json").read_text(encoding="utf-8"))
    recipe = next(item for item in protocol["recipes"] if item["id"] == "R1_D")
    recipe = {**recipe, "params": {**protocol["hgb_fixed_parameters"], **recipe["params"]}}
    model = build_model(recipe)
    feature_names = model._required_feature_names()
    target = train.threat_rank.to_numpy(dtype=float)
    if not np.isfinite(target).all() or not ((0 <= target) & (target <= 100)).all():
        raise ValueError("Train threat_rank must be finite within [0,100]")
    started = perf_counter()
    with threadpool_limits(limits=protocol["thread_limit"]):
        model.fit(train.loc[:, list(feature_names)], target)
        raw = np.asarray(model.predict(test.loc[:, list(feature_names)]), dtype=float)
    clipped = np.clip(raw, 0, 100)
    if not np.isfinite(clipped).all():
        raise ValueError("Regression predictions must be finite")
    return clipped, {
        "recipe": "R1_D expanded structured HistGradientBoostingRegressor",
        "loss": "absolute_error",
        "parameters": model.hgb_params,
        "fit_rows": len(train),
        "input_features": list(feature_names),
        "fit_state": model.audit_state(),
        "fit_plus_predict_seconds": perf_counter() - started,
        "raw_prediction_range": [float(raw.min()), float(raw.max())],
        "clipped_prediction_range": [float(clipped.min()), float(clipped.max())],
        "prediction_sha256": hashlib.sha256(np.asarray(clipped, dtype="<f8").tobytes()).hexdigest(),
        "clipped_rows": int(((raw < 0) | (raw > 100)).sum()),
    }


def run(task: str, data_dir: Path) -> Path:
    train, test, sample, input_hashes = _load_inputs(task, data_dir)
    start = perf_counter()
    if task == "classification":
        predictions, audit = _classification(train, test)
        target = "artifact_signal"
    else:
        predictions, audit = _regression(train, test)
        target = "threat_rank"
    submission = _submission_frame(test, sample, predictions, target, task)
    output_dir = PROJECT_ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{task}_initial.csv"
    submission.to_csv(output, index=False, float_format="%.17g")
    # Read back the exact deliverable and independently recheck its file representation.
    reread = pd.read_csv(output, float_precision="round_trip")
    values_equal = (reread[target].equals(submission[target]) if task == "classification"
                    else np.array_equal(reread[target].to_numpy(dtype=float),
                                        submission[target].to_numpy(dtype=float)))
    if list(reread.columns) != list(submission.columns) or not reread.record_id.equals(submission.record_id) or not values_equal:
        raise RuntimeError("CSV round-trip differs from validated submission values")
    source_commit = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != MAIN_CHECKPOINT:
        raise RuntimeError(f"Source worktree commit changed from the authorized checkpoint: {source_commit}")
    source_paths = (["src/models/classification_models.py", "src/models/classification_round2_models.py"]
                    if task == "classification" else ["src/models/round1_models.py"])
    source_code_hashes = {rel: _file_sha256(PROJECT_ROOT / rel) for rel in source_paths}
    protocol_path = (PROJECT_ROOT / "experiments/classification_round2/protocol.json" if task == "classification"
                     else PROJECT_ROOT / "experiments/round1/protocol.json")
    freeze_path = (PROJECT_ROOT / "experiments/classification_round2/freeze.json" if task == "classification"
                   else PROJECT_ROOT / "experiments/round1/freeze.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if task == "classification":
        model_identity = {"protocol_sha256": freeze["identity"]["protocol_sha256"],
                          "freeze_signature": freeze["signature"],
                          "round1_freeze_signature": freeze["identity"]["round1_freeze_signature"]}
    else:
        from src.models.round1_evaluation import identity as r1_identity
        current_identity = r1_identity(data_dir)
        if current_identity != freeze["identity"]:
            raise RuntimeError("Regression Round 1 freeze verification failed")
        model_identity = {"protocol_sha256": current_identity["protocol_sha256"],
                          "freeze_signature": freeze["signature"]}
    manifest = {
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "competition": COMPETITIONS[task],
        "submission_message": MESSAGES[task],
        "source_commit": source_commit,
        "source_code_sha256": source_code_hashes,
        "model_protocol_identity": model_identity,
        "model_boundary": "refit on all official labeled Train rows; Test used only for predictions and schema/ID validation; no Test labels exist or were used",
        "input_sha256": input_hashes,
        "train_rows": len(train),
        "test_rows": len(test),
        "sample_header": list(sample.columns),
        "submission_sha256": sha256(output),
        "submission_path_gitignored": str(output.relative_to(PROJECT_ROOT)),
        "runtime": {
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
            "thread_limit": 2,
        },
        "elapsed_seconds": perf_counter() - start,
        "model": audit,
        "kaggle_submission": {
            "status": "NOT_SUBMITTED_KAGGLE_CLI_AND_STANDARD_CREDENTIALS_UNAVAILABLE",
            "one_submission_authorized": True,
            "submissions_used": 0,
            "score": None,
        },
    }
    manifest_path = PROJECT_ROOT / "reports/submissions" / f"{task}_initial_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\data\raw"))
    args = parser.parse_args()
    tasks = ["classification", "regression"] if args.task == "both" else [args.task]
    for task in tasks:
        print(f"Preparing {task} full-Train submission", flush=True)
        path = run(task, args.data_dir)
        print(f"{task}: PREPARED; manifest={path}", flush=True)


if __name__ == "__main__":
    main()
