"""C05: frozen 60k combined-word-vocabulary classification pilot.

Modes are explicit. ``--preflight-only`` is read-only; ``--freeze-only`` saves
the reviewed recipe and identity without fitting; ``--run-frozen-pilot`` requires
that committed freeze and performs exactly two temporal predictor fits.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

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
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c05 import (
    CARD_ID,
    FEATURE_NAMES,
    TFIDF_PARAMS,
    fit_predict,
)
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C05.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C05"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C05"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C05"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
ATTEMPTS_PATH = REPORT_DIR / "attempts.jsonl"

TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "bb7f6b93f3cce1466cedd91db07f80f4bd632cc788ffc2c3489f3fa1f93e9525"
MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
C04_VERIFICATION_SHA256 = "92bc181e650dcc761f51b353d12d87cec2373cafd9b5ce1649b4a1268abbc93d"
C04_RESULTS_SHA256 = "8d4c87f89f41673d68624eed0300f0d580b4869676d02c35ad2564320ae91222"
C04_GATE_SHA256 = "67eebd0f69f959dfa74d94f27c4db1fc77a48799338a3d62b4c59ceaf29baec6"
C04_ATTEMPTS_SHA256 = "670c62c7d20cd3e5a863e42f42eede7ad1f8bdc67e5837c1c4f9fb9ca74f829c"
MIN_GROUP_SUPPORT = 200
THREADS = 2
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_MEMORY_RESERVE_BYTES = 2 * 1024**3
LIGHTGBM_MEMORY_OVERHEAD_BYTES = 512 * 1024**2
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064, "H12": 38_588}
EXPECTED_CONTROL_METRICS = {
    "W1": 0.8031600548696844,
    "W4": 0.8414177014153625,
    "primary": 0.8239724981677496,
    "H12": 0.8029655654017788,
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
EXPECTED_CARD = {
    "id": "C05",
    "task": "classification",
    "name": "Rare-term retention",
    "family": "vocab_width",
    "priority": "P2",
    "policy": "C_SPARSE_REPLACE",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "C04 completed without resource failure; do not add further cap/min_df points",
    "status": "PLANNED_NOT_RUN",
}
EXPECTED_GATE = {
    "pilot_primary_gain_min": 0.0005,
    "pilot_H12_harm_max": 0.001,
    "individual_pilot_view_harm_max": 0.005,
    "supported_group_n_min": 200,
    "supported_group_F1_harm_max": 0.01,
    "positive_class_recall_harm_max": 0.02,
    "positive_and_negative_per_group_min": 30,
}
INPUTS_BASE = {
    "official_classification_train": ROOT / "data/raw/classification/Train.csv",
    "canonical_queue": QUEUE_PATH,
    "canonical_card": CARD_PATH,
    "accepted_classification_manifest": ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
    "accepted_c04_verification_reference": ROOT / "reports/optimization_v3/batch1/C04/verification.json",
    "c04_results_reference": ROOT / "reports/optimization_v3/batch1/C04/results.json",
    "c04_gate_reference": ROOT / "reports/optimization_v3/batch1/C04/gate.json",
    "c04_attempts_reference": ROOT / "reports/optimization_v3/batch1/C04/attempts.jsonl",
    "frozen_sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
    "frozen_sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
    "round1_dc_nbsvm55_cache": ROOT / "experiments/runs/dual_cls_v1a/fit55__DC_NBSVM.npz",
    "round1_dc_nbsvm67_cache": ROOT / "experiments/runs/dual_cls_v1a/fit67__DC_NBSVM.npz",
}
SOURCE_PATHS = [
    Path(__file__),
    ROOT / "src/models/optimization_v3_batch1_c05.py",
    ROOT / "tests/test_optimization_v3_batch1_c05.py",
    ROOT / "tests/test_optimization_v3_batch1_c05_runner.py",
    ROOT / "scripts/optimization_v3_batch1_c04.py",
    ROOT / "src/models/optimization_v3_batch1_c04.py",
    ROOT / "tests/test_optimization_v3_batch1_c04.py",
    ROOT / "src/models/optimization_v3_batch1.py",
    ROOT / "scripts/optimization_v3_batch1_verify.py",
    ROOT / "scripts/optimization_v3_sparse_classification.py",
    ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
    ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
    ROOT / "tests/test_optimization_v3_sparse_classification.py",
    ROOT / "scripts/research_campaign_classification_stage2.py",
    ROOT / "scripts/research_campaign_classification.py",
    ROOT / "scripts/optimization_v2_classification.py",
    ROOT / "src/models/optimization_v3_sparse_classification.py",
    ROOT / "src/models/classification_models.py",
    ROOT / "src/models/classification_round2_models.py",
    ROOT / "src/models/classification_metrics.py",
    ROOT / "src/models/round1_evaluation.py",
    ROOT / "src/models/round1_models.py",
    ROOT / "src/models/regression_research_text.py",
    ROOT / "src/models/regression_round2_models.py",
    ROOT / "src/data/audit.py",
    ROOT / "src/utils/paths.py",
]
PACKAGE_NAMES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise BatchIntegrityError("Nonfinite value in C05 JSON payload")
        return result
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, type):
        return value.__name__
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"Unsupported C05 JSON value {type(value).__name__}")


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("C05 canonical queue/card bytes differ from reviewed recipe")
    if load_queue(QUEUE_PATH) != dict(queue):
        raise BatchIntegrityError("Loaded C05 queue differs from canonical queue bytes")
    canonical = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if canonical is None or dict(card) != canonical:
        raise BatchIntegrityError("C05 queue card differs from the canonical C05 entry")
    for field, expected in EXPECTED_CARD.items():
        if card.get(field) != expected:
            raise BatchIntegrityError(f"C05 card changed in field {field}")
    change = str(card.get("change", ""))
    if ("max_features60000" not in change or "min_df2" not in change
            or int(card.get("queue_order", -1)) != 31):
        raise BatchIntegrityError("C05 frozen rare-term change/order differs")
    if (int(TFIDF_PARAMS.get("max_features", -1)) != 60_000
            or int(TFIDF_PARAMS.get("min_df", -1)) != 2
            or tuple(TFIDF_PARAMS.get("ngram_range", ())) != (1, 2)):
        raise BatchIntegrityError("C05 adapter does not implement the fixed max_features=60000/min_df=2 recipe")
    if int(card.get("pilot_predictor_fits", -1)) != 2:
        raise BatchIntegrityError("C05 must reserve exactly two candidate predictor fits")
    baseline = queue["baseline"]["classification"]
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256):
        raise BatchIntegrityError("C05 frozen classification Train/manifest lineage changed")
    gates = queue["evaluation"]["classification_gate"]
    for field, expected in EXPECTED_GATE.items():
        if float(gates.get(field, float("nan"))) != float(expected):
            raise BatchIntegrityError(f"C05 frozen classification gate changed in {field}")


def _read_official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    path = task_files("classification").get("train")
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Official Classification Train is unavailable")
    path = Path(path)
    if file_sha256(path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train hash differs from C05 freeze")
    frame = pd.read_csv(path, usecols=["record_id", *FEATURE_NAMES, "artifact_signal"])
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Expected 50,608 official Train rows, got {len(frame)}")
    if frame["record_id"].isna().any() or not frame["record_id"].astype(str).is_unique:
        raise BatchIntegrityError("Official Train record_id is missing or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Train cycles differ from the frozen range 47–79")
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    if target.shape != (len(frame),) or not np.isin(target, [0, 1]).all():
        raise BatchIntegrityError("Official Train classification target is invalid")
    if "threat_rank" in frame.columns or set(frame.columns) != {"record_id", *FEATURE_NAMES, "artifact_signal"}:
        raise BatchIntegrityError("Unexpected target or feature escaped the C05 read allowlist")
    features = frame.loc[:, list(FEATURE_NAMES)].copy()
    return path, frame, features, target


def _control_input_files(train_path: Path, views: Mapping[str, Any]) -> dict[str, Path]:
    cutoffs = sorted({int(cutoff) for segments in views.values() for cutoff, _low, _high in segments})
    inputs = dict(INPUTS_BASE)
    for cutoff in cutoffs:
        inputs[f"sparse_pilot_{cutoff}_npz"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.npz"
        inputs[f"sparse_pilot_{cutoff}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.json"
        inputs[f"catboost_{cutoff}_npz"] = CB_CACHE / f"fit{cutoff}.npz"
        inputs[f"catboost_{cutoff}_sidecar"] = CB_CACHE / f"fit{cutoff}.json"
        inputs[f"round1_c2b_{cutoff}"] = ROOT / f"experiments/runs/classification_round1_v1/fit{cutoff}__C2_B.npz"
        inputs[f"round1_audit_{cutoff}"] = ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json"
    inputs["c04_control_freeze"] = ROOT / "experiments/optimization_v3/batch1/C04/freeze.json"
    return inputs


def _resource_estimate(fit_x: pd.DataFrame) -> dict[str, Any]:
    docs, _missing = _text_values(fit_x["signal_stream"])
    analyzer = TfidfVectorizer(**dict(TFIDF_PARAMS)).build_analyzer()
    cap = int(TFIDF_PARAMS["max_features"])
    text_nnz_upper = sum(min(len(set(analyzer(doc))), cap) for doc in docs)
    n_rows = int(len(fit_x))
    categories = 3
    structured_source_count = len(FEATURE_NAMES) - categories - 1
    structured_width_upper = structured_source_count * 5 + categories * 254
    structured_nnz_upper = n_rows * structured_width_upper
    total_nnz_upper = text_nnz_upper + structured_nnz_upper + n_rows
    csr_bytes_upper = total_nnz_upper * 16 + (n_rows + 1) * 8
    features_upper = cap + structured_width_upper + 1
    histogram_bytes = features_upper * 256 * 16 * THREADS * 2
    estimated_peak = 3 * csr_bytes_upper + histogram_bytes + LIGHTGBM_MEMORY_OVERHEAD_BYTES
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_MEMORY_CAP_BYTES, max(0, available - FREE_MEMORY_RESERVE_BYTES))
    status = "PASS" if available >= 4 * 1024**3 and estimated_peak <= allowed else "BLOCKED_RESOURCE"
    return {
        "status": status,
        "fit_rows": n_rows,
        "text_nnz_upper_bound": int(text_nnz_upper),
        "structured_width_upper_bound": int(structured_width_upper),
        "csr_bytes_upper_bound": int(csr_bytes_upper),
        "histogram_bytes_upper_bound": int(histogram_bytes),
        "estimated_peak_bytes": int(estimated_peak),
        "estimated_peak_is_observed": False,
        "available_physical_bytes": available,
        "free_memory_reserve_bytes": FREE_MEMORY_RESERVE_BYTES,
        "job_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "allowed_bytes": int(allowed),
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "sparse_only": True,
    }


def _macro_f1(y: np.ndarray, probability: np.ndarray) -> float:
    if y.ndim != 1 or probability.shape != y.shape or len(y) == 0:
        raise BatchIntegrityError("C05 control metric inputs are empty or misaligned")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all():
        raise BatchIntegrityError("C05 control metric has invalid labels/probabilities")
    predicted = probability >= 0.5
    values = []
    for label in (0, 1):
        truth = y == label
        guess = predicted == label
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(sum(values) / 2.0)


def _load_controls(frame: pd.DataFrame, target: np.ndarray, views: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    check_sparse_pilot()
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    controls: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        control = control_record(frame, target, int(cutoff))
        incumbent = pilot_record(frame, control)
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if not np.array_equal(control["rows"], expected_rows) or not np.array_equal(incumbent["rows"], expected_rows):
            raise BatchIntegrityError(f"C05 frozen control rows differ at cutoff {cutoff}")
        controls[cutoff] = {
            "rows": expected_rows,
            "baseline_prediction": np.asarray(incumbent["blend25"], dtype=np.float64),
            "baseline_sparse_raw": np.asarray(incumbent["raw_candidate"], dtype=np.float64),
            "c2b_probability": np.asarray(control["c2b"], dtype=np.float64),
            "cb_probability": np.asarray(control["cb_missing"], dtype=np.float64),
            "stream_missing": np.asarray(control["missing"], dtype=bool),
        }
    return controls


def _control_metrics(frame: pd.DataFrame, target: np.ndarray, controls: Mapping[int, Any],
                     views: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    result: dict[str, dict[str, Any]] = {}
    for name, segments in views.items():
        rows_parts, prediction_parts = [], []
        for cutoff, low, high in segments:
            saved = controls[int(cutoff)]
            mask = (cycles[saved["rows"]] >= int(low)) & (cycles[saved["rows"]] <= int(high))
            rows_parts.append(saved["rows"][mask])
            prediction_parts.append(saved["baseline_prediction"][mask])
        rows = np.concatenate(rows_parts)
        probability = np.concatenate(prediction_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"C05 baseline view {name} denominator/row identity changed")
        result[name] = {"rows": int(len(rows)), "macro_f1": _macro_f1(target[rows], probability)}
    return result


def _validate_c04_reference() -> dict[str, Any]:
    base = ROOT / "reports/optimization_v3/batch1/C04"
    paths = {
        "verification": (base / "verification.json", C04_VERIFICATION_SHA256),
        "results": (base / "results.json", C04_RESULTS_SHA256),
        "gate": (base / "gate.json", C04_GATE_SHA256),
        "attempts": (base / "attempts.jsonl", C04_ATTEMPTS_SHA256),
    }
    for name, (path, expected_hash) in paths.items():
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise BatchIntegrityError(f"C04 prerequisite {name} evidence is missing or changed")
    reference = _read_json(paths["verification"][0])
    report = _read_json(paths["results"][0])
    gate = _read_json(paths["gate"][0])
    try:
        attempt_events = [
            json.loads(line)
            for line in paths["attempts"][0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, TypeError) as exc:
        raise BatchIntegrityError(f"C04 prerequisite attempt log is invalid: {exc}") from exc
    if any(event.get("event") in {"FIT_FAILED", "RESOURCE_FAILED", "RESOURCE_FAILURE"}
           for event in attempt_events):
        raise BatchIntegrityError("C04 prerequisite has a recorded failed/resource attempt")
    if (reference.get("status") != "INDEPENDENT_QA_PASS"
            or reference.get("card_id") != "C04"
            or reference.get("fit_predictor_count") != 2
            or reference.get("test_data_used") is not False):
        raise BatchIntegrityError("C04 prerequisite lacks an independently verified two-fit Train-only result")
    if (report.get("status") != "PILOT_COMPLETE_" + str(gate.get("status"))
            or gate.get("status") not in {"ADVANCE", "RETIRE"}
            or report.get("fit_accounting", {}).get("successful_predictor_fits") != 2
            or report.get("fit_accounting", {}).get("budget_reserved_predictor_fits") != 2):
        raise BatchIntegrityError("C04 prerequisite pilot is incomplete or not reconciled")
    cutoff_fits = report.get("fit_accounting", {}).get("cutoff_fits", {})
    for cutoff in ("55", "67"):
        fit = cutoff_fits.get(cutoff, {})
        if (fit.get("cache_status") != "FIT_SAVED"
                or fit.get("predictor_fit_count") != 1
                or fit.get("failed_attempt_count") != 0):
            raise BatchIntegrityError(f"C04 prerequisite fit {cutoff} is failed, missing, or unreconciled")
    memory = report.get("preflight", {}).get("memory_estimate", {})
    if any(memory.get(cutoff, {}).get("status") != "PASS" for cutoff in ("55", "67")):
        raise BatchIntegrityError("C04 prerequisite has a resource failure")
    if reference.get("decision") != gate.get("status"):
        raise BatchIntegrityError("C04 verification and gate decisions do not agree")
    result = {}
    for name, expected in EXPECTED_CONTROL_METRICS.items():
        view = reference.get("views", {}).get(name, {})
        rows = int(view.get("n", -1))
        metric = float(view.get("baseline", float("nan")))
        report_view = report.get("views", {}).get(name, {})
        report_baseline = float(report_view.get("baseline", {}).get("macro_f1", float("nan")))
        if (rows != EXPECTED_VIEW_ROWS[name]
                or not np.isclose(metric, expected, rtol=0.0, atol=1e-12)
                or not np.isclose(report_baseline, metric, rtol=0.0, atol=1e-12)):
            raise BatchIntegrityError(f"C04 verified baseline reference differs for {name}")
        result[name] = {"rows": rows, "macro_f1": metric}
    return result


def _vectorizer_recipe() -> dict[str, Any]:
    return _json_clean(dict(TFIDF_PARAMS))


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any]) -> dict[str, Any]:
    return _json_clean({
        "card_id": CARD_ID,
        "task": "classification",
        "queue_card": dict(card),
        "feature_allowlist": list(FEATURE_NAMES),
        "single_recipe_change": {"tfidf.min_df": {"baseline": 3, "candidate": 2}},
        "tfidf_parameters": _vectorizer_recipe(),
        "classifier_parameters": dict(SPARSE_PARAMS),
        "structured_preprocessing": "C04 fit-only numeric scaling and sparse OHE; unchanged",
        "fit_population": "all official Train rows with cycle_code <= the temporal cutoff",
        "prediction_policy": {
            "stream_present": "0.75*frozen C2_B probability + 0.25*C05 sparse LightGBM probability",
            "stream_missing": "frozen CatBoost probability",
            "threshold": 0.50,
            "stream_missing_rule": "null, empty, or whitespace-only signal_stream",
        },
        "evaluation": {
            "profile": "STANDARD",
            "views": {name: [list(segment) for segment in segments] for name, segments in views.items()},
            "fit_rows": EXPECTED_FIT_ROWS,
            "score_rows": EXPECTED_VIEW_ROWS,
            "minimum_group_support": MIN_GROUP_SUPPORT,
            "primary_population_note": "W1+W4 pilot, 9,064 rows; queue primary 0.8299435 is full-six and is not used as a pilot control metric",
        },
        "gate": EXPECTED_GATE,
        "resource_policy": {
            "threads": THREADS,
            "max_concurrent_training_jobs": 1,
            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
            "free_memory_reserve_bytes": FREE_MEMORY_RESERVE_BYTES,
            "sparse_only": True,
            "peak_estimate_is_not_observed": True,
        },
    })


def _write_or_require_same(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise BatchIntegrityError(f"Existing C05 artifact differs: {path}")
    else:
        write_json_exclusive(path, value)


def _validate_recipe_artifact(path: Path, recipe: Mapping[str, Any], digest: str) -> None:
    stored = _read_json(path)
    if stored != dict(recipe) or json_sha256(stored) != digest:
        raise BatchIntegrityError("Persisted C05 recipe differs from canonical freeze identity")


def _frozen_identity_matches(frozen: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """Compare canonical JSON identity, tolerating JSON stringification of map keys."""
    signature = json_sha256(identity)
    return (frozen.get("signature") == signature
            and json_sha256(frozen.get("identity", {})) == signature)


def _budget_preflight() -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        return {"status": "BLOCKED_LEDGER", "error": "Global batch ledger is missing"}
    reservations: dict[str, int] = {}
    try:
        for line_number, raw in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            event = json.loads(raw)
            if event.get("event") == "BUDGET_RESERVED":
                key = str(event["budget_key"])
                amount = int(event["predictor_fits"])
                if key in reservations and reservations[key] != amount:
                    raise ValueError(f"conflicting reservation {key}")
                reservations[key] = amount
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "BLOCKED_LEDGER", "error": str(exc)}
    total = sum(reservations.values())
    own = reservations.get("C05:PILOT")
    next_total = total if own == 2 else total + 2
    status = "PASS" if own in (None, 2) and next_total <= 180 else "BLOCKED_BUDGET"
    return {"status": status, "reserved_total": total, "next_total": next_total,
            "existing_c05_reservation": own, "cap": 180}


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    train_path, frame, features, target = _read_official_train()
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"C05 STANDARD views changed: {views}")
    manifest_path = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
    if file_sha256(manifest_path) != MANIFEST_SHA256:
        raise BatchIntegrityError("Accepted sparse25 manifest hash changed")
    controls = _load_controls(frame, target, views)
    baseline_metrics = _control_metrics(frame, target, controls, views)
    control_reference = _validate_c04_reference()
    for name, expected in EXPECTED_CONTROL_METRICS.items():
        if (baseline_metrics[name]["rows"] != EXPECTED_VIEW_ROWS[name]
                or not np.isclose(baseline_metrics[name]["macro_f1"], expected, rtol=0.0, atol=1e-12)
                or not np.isclose(baseline_metrics[name]["macro_f1"], control_reference[name]["macro_f1"],
                                  rtol=0.0, atol=1e-12)):
            raise BatchIntegrityError(
                f"C05 row-exact baseline metric differs from verified reference for {name}; freeze blocked"
            )

    memory = {}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        if len(fit_rows) != EXPECTED_FIT_ROWS[cutoff]:
            raise BatchIntegrityError(f"C05 fit row count changed at cutoff {cutoff}")
        memory[str(cutoff)] = _resource_estimate(features.iloc[fit_rows])

    input_files = _control_input_files(train_path, views)
    missing_inputs = [str(path) for path in input_files.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"C05 frozen control inputs missing: {missing_inputs}")
    source_missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if source_missing:
        raise FileNotFoundError(f"C05 frozen source files missing: {source_missing}")
    input_hashes = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in SOURCE_PATHS}
    recipe = _recipe_payload(card, views)
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=input_files,
        source_files=SOURCE_PATHS,
        evaluation_views=views,
        packages=PACKAGE_NAMES,
    )
    identity.update({
        "recipe": recipe,
        "recipe_semantic_sha256": json_sha256(recipe),
        "baseline_control_metrics": baseline_metrics,
        "baseline_control_reference_sha256": C04_VERIFICATION_SHA256,
        "resource_policy": recipe["resource_policy"],
        "expected_fit_rows": EXPECTED_FIT_ROWS,
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "test_sample_kaggle_access": False,
    })
    signature = json_sha256(identity)
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise BatchIntegrityError("Existing C05 recipe differs from canonical preflight recipe")
    if FREEZE_PATH.exists():
        frozen = _read_json(FREEZE_PATH)
        if not _frozen_identity_matches(frozen, identity):
            raise BatchIntegrityError("Existing C05 freeze differs from current source/input/recipe identity")
    budget = _budget_preflight()
    resource_pass = all(value["status"] == "PASS" for value in memory.values())
    status = "PREFLIGHT_PASS" if resource_pass and budget.get("status") == "PASS" else "BLOCKED_RESOURCE_OR_BUDGET"
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "read-only official Train/control/resource preflight; no C05 fit or candidate score",
        "official_train_sha256": input_hashes["official_classification_train"],
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "freeze_identity_sha256": signature,
        "recipe_semantic_sha256": json_sha256(recipe),
        "freeze_exists": FREEZE_PATH.exists(),
        "memory_estimate": memory,
        "baseline_control_metrics": baseline_metrics,
        "baseline_control_reference": control_reference,
        "pilot_population_warning": "Queue primary is full-six; C05 primary comparison is W1+W4 on 9,064 identical rows.",
        "budget_preflight": budget,
        "candidate_prediction_or_metric_computed": False,
        "official_test_or_sample_access": False,
        "kaggle_access": False,
        "artifacts_written": False,
        "python_version": platform.python_version(),
    }
    return {"public": public, "identity": identity, "recipe": recipe,
            "frame": frame, "features": features, "target": target,
            "cycles": cycles, "views": views, "controls": controls,
            "train_path": train_path, "input_files": input_files}


def _control_loader(context: Mapping[str, Any]):
    controls = context["controls"]

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        saved = controls[int(cutoff)]
        if not np.array_equal(saved["rows"], score_rows):
            raise BatchIntegrityError(f"C05 control rows differ from engine score rows at {cutoff}")
        return {
            "baseline_prediction": saved["baseline_prediction"],
            "baseline_sparse_raw": saved["baseline_sparse_raw"],
            "c2b_probability": saved["c2b_probability"],
            "cb_probability": saved["cb_probability"],
            "stream_missing": saved["stream_missing"],
            "control_rows": saved["rows"],
        }
    return load


def _reconstruct_c05(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(raw), len(missing), len(c2b), len(cb), len(old_sparse)}) != 1:
        raise BatchIntegrityError("C05 independent route inputs are misaligned")
    candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return candidate, baseline


def _gate_and_verify(context: Mapping[str, Any], report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_report(
        report=dict(report),
        frame=context["frame"],
        target=context["target"],
        artifact_dir=RUN_DIR,
        freeze_signature=json_sha256(context["identity"]),
        evaluation_views=context["views"],
        prediction_reconstructors={CARD_ID: _reconstruct_c05},
    )
    if verification.get("status") != "PASS":
        raise BatchIntegrityError("C05 independent prediction/metric verifier did not PASS")
    for name, expected in EXPECTED_CONTROL_METRICS.items():
        view = verification["views"][name]
        if view["n"] != EXPECTED_VIEW_ROWS[name] or not np.isclose(
                view["baseline"], expected, rtol=0.0, atol=1e-12):
            raise BatchIntegrityError(f"C05 verified baseline population/metric mismatch in {name}")
    gate_config = EXPECTED_GATE
    primary_gain = float(report["views"]["primary"]["gain_positive_is_better"])
    h12_gain = float(report["views"]["H12"]["gain_positive_is_better"])
    w1_gain = float(report["views"]["W1"]["gain_positive_is_better"])
    w4_gain = float(report["views"]["W4"]["gain_positive_is_better"])
    subgroup_failures = verification["views"]["primary"]["subgroup_diagnostics"]["supported_guard_failures"]
    checks = {
        "primary_gain_at_least_0_0005": primary_gain >= gate_config["pilot_primary_gain_min"],
        "H12_harm_at_most_0_001": h12_gain >= -gate_config["pilot_H12_harm_max"],
        "W1_harm_at_most_0_005": w1_gain >= -gate_config["individual_pilot_view_harm_max"],
        "W4_harm_at_most_0_005": w4_gain >= -gate_config["individual_pilot_view_harm_max"],
        "supported_subgroup_guards": len(subgroup_failures) == 0,
    }
    gate = {
        "card_id": CARD_ID,
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": gate_config,
        "supported_subgroup_failures": subgroup_failures,
        "baseline_scope_note": "Primary is W1+W4 (9,064 rows); queue full-six primary is not a pilot comparator.",
        "decision_basis": "Frozen temporal pilot gates and exact row-matched controls; no Test/Kaggle evidence.",
    }
    verification.update({
        "status": "INDEPENDENT_QA_PASS",
        "decision": gate["status"],
        "freeze_signature": json_sha256(context["identity"]),
        "fit_predictor_count": 2,
        "cutoffs": [55, 67],
        "gate_checks": gate["checks"],
        "threshold": 0.50,
        "test_data_used": False,
        "kaggle_access": False,
    })
    return verification, gate


def _fit_accounting(context: Mapping[str, Any]) -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        raise BatchIntegrityError("C05 fit ledger is missing after the pilot")
    ledger_events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    card_ledger_events = [event for event in ledger_events if event.get("card_id") == CARD_ID]
    reservations = [event for event in card_ledger_events if event.get("event") == "BUDGET_RESERVED"]
    if len(reservations) != 1 or int(reservations[0].get("predictor_fits", -1)) != 2:
        raise BatchIntegrityError("C05 ledger does not contain exactly one two-fit reservation")
    if not ATTEMPTS_PATH.is_file():
        raise BatchIntegrityError("C05 per-card attempt log is missing after the pilot")
    attempt_events = [json.loads(line) for line in ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    if any(event.get("card_id") != CARD_ID for event in attempt_events):
        raise BatchIntegrityError("C05 per-card attempt log contains an event for another card")

    fit_rows = {"55": EXPECTED_FIT_ROWS[55], "67": EXPECTED_FIT_ROWS[67]}
    cutoff_results: dict[str, Any] = {}
    for cutoff in (55, 67):
        rows = context["controls"][cutoff]["rows"]
        expected_identity = {
            "freeze_signature": json_sha256(context["identity"]),
            "card_id": CARD_ID,
            "cutoff": cutoff,
            "score_rows": rows.tolist(),
        }
        path = RUN_DIR / f"{CARD_ID}_fit{cutoff}.npz"
        _arrays, sidecar = read_bundle(path, expected_identity)
        metadata = sidecar.get("metadata", {})
        fit_count = int(metadata.get("fit_predictor_count", -1))
        n_fit = int(metadata.get("fit_rows", -1))
        n_score = int(metadata.get("score_rows", -1))
        seconds = float(metadata.get("fit_seconds", float("nan")))
        if (fit_count != 1 or n_fit != fit_rows[str(cutoff)] or n_score != len(rows)
                or not np.isfinite(seconds) or seconds < 0):
            raise BatchIntegrityError(f"C05 fit receipt is invalid at cutoff {cutoff}")
        by_cutoff = [event for event in attempt_events if event.get("cutoff") == cutoff]
        started = sum(event.get("event") == "FIT_STARTED" for event in by_cutoff)
        saved = sum(event.get("event") == "FIT_SAVED" for event in by_cutoff)
        failed = sum(event.get("event") == "FIT_FAILED" for event in by_cutoff)
        reused = sum(event.get("event") == "RESUMED_VERIFIED" for event in by_cutoff)
        if saved + reused < 1:
            raise BatchIntegrityError(f"C05 cutoff {cutoff} has no completed or verified-reuse event")
        if saved > 1 or reused > 1:
            raise BatchIntegrityError(f"C05 cutoff {cutoff} has duplicated success/reuse events")
        cutoff_results[str(cutoff)] = {
            "predictor_fit_count": fit_count,
            "fit_rows": n_fit,
            "score_rows": n_score,
            "fit_seconds": seconds,
            "fit_attempt_count": int(started),
            "failed_attempt_count": int(failed),
            "cache_status": "VERIFIED_REUSE" if reused else "FIT_SAVED",
            "attempt_events": [str(event.get("event")) for event in by_cutoff
                               if event.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED"}],
            "training_warnings": metadata.get("fit_metadata", {}).get("fit_warnings", []),
        }
    total_seconds = float(sum(value["fit_seconds"] for value in cutoff_results.values()))
    return {
        "budget_reserved_predictor_fits": 2,
        "successful_predictor_fits": 2,
        "cutoff_fits": cutoff_results,
        "total_fit_seconds": total_seconds,
        "peak_memory_observed": False,
        "attempt_log_path": "reports/optimization_v3/batch1/C05/attempts.jsonl",
    }


def _publish(context: Mapping[str, Any], report: Mapping[str, Any],
             verification: Mapping[str, Any], gate: Mapping[str, Any],
             fit_accounting: Mapping[str, Any]) -> None:
    result = dict(report)
    result.update({
        "status": "PILOT_COMPLETE_" + str(gate["status"]),
        "input_sha256": context["public"]["input_sha256"],
        "source_sha256": context["public"]["source_sha256"],
        "preflight": context["public"],
        "control_metric_scope": context["public"]["baseline_control_metrics"],
        "fit_accounting": dict(fit_accounting),
        "resource_note": "Memory values are conservative preflight estimates; peak process RAM was not observed.",
        "test_data_used": False,
        "kaggle_access": False,
    })
    _write_or_require_same(RESULTS_PATH, result)
    _write_or_require_same(VERIFICATION_PATH, dict(verification))
    _write_or_require_same(GATE_PATH, dict(gate))
    summary = {
        "card_id": CARD_ID,
        "decision": gate["status"],
        "fit_predictor_count": 2,
        "cutoffs": [55, 67],
        "primary_rows": EXPECTED_VIEW_ROWS["primary"],
        "primary_baseline": verification["views"]["primary"]["baseline"],
        "primary_candidate": verification["views"]["primary"]["candidate"],
        "primary_gain": report["views"]["primary"]["gain_positive_is_better"],
        "H12_rows": EXPECTED_VIEW_ROWS["H12"],
        "H12_baseline": verification["views"]["H12"]["baseline"],
        "H12_candidate": verification["views"]["H12"]["candidate"],
        "H12_gain": report["views"]["H12"]["gain_positive_is_better"],
        "total_fit_seconds": fit_accounting["total_fit_seconds"],
        "cutoff_fit_accounting": fit_accounting["cutoff_fits"],
        "fullsix_queue_primary_is_not_pilot_baseline": True,
        "test_sample_kaggle_used": False,
        "details": ["results.json", "verification.json", "gate.json"],
    }
    _write_or_require_same(SUMMARY_PATH, summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one C05 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError("C05 execution is blocked by control/resource/budget preflight")

    if freeze_only:
        _write_or_require_same(RECIPE_PATH, context["recipe"])
        _validate_recipe_artifact(RECIPE_PATH, context["recipe"], context["identity"]["recipe_semantic_sha256"])
        signature = ensure_frozen(FREEZE_PATH, context["identity"])
        return {"card_id": CARD_ID, "status": "FROZEN_NO_FIT", "freeze_signature": signature,
                "recipe_path": str(RECIPE_PATH), "freeze_path": str(FREEZE_PATH),
                "fit_performed": False, "test_or_kaggle_access": False}

    if not (RECIPE_PATH.is_file() and FREEZE_PATH.is_file()):
        raise BatchIntegrityError("C05 requires a persisted, reviewed recipe/freeze before pilot fits")
    _validate_recipe_artifact(RECIPE_PATH, context["recipe"], context["identity"]["recipe_semantic_sha256"])
    frozen = _read_json(FREEZE_PATH)
    if not _frozen_identity_matches(frozen, context["identity"]):
        raise BatchIntegrityError("C05 current source/input/recipe identity differs from committed freeze")
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train changed after C05 freeze")

        append_event(ATTEMPTS_PATH, {
        "card_id": CARD_ID,
        "event": "PREFLIGHT_PASS",
        "at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "resource_estimates": context["public"]["memory_estimate"],
        "baseline_control_metrics": context["public"]["baseline_control_metrics"],
        "fit_predictor_count_reserved": 2,
    })
    report = run_pilot(
        card=card,
        artifact_dir=RUN_DIR,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        frame=context["frame"],
        target=context["target"],
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=_control_loader(context),
        freeze_identity=context["identity"],
        freeze_path=FREEZE_PATH,
        event_path=ATTEMPTS_PATH,
        evaluation_views=context["views"],
        threads=THREADS,
        report_path=ENGINE_REPORT_PATH,
    )
    verification, gate = _gate_and_verify(context, report)
    fit_accounting = _fit_accounting(context)
    _publish(context, report, verification, gate, fit_accounting)
    return {"report": report, "verification": verification, "gate": gate}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(_json_clean(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
