"""R04 runner: stateless hashing features with a sparse LightGBM-L1 model."""
from __future__ import annotations

import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_batch1_r02 import (
    BASELINE_DIAGNOSTICS,
    FULL_CACHE,
    FULL_REPORT,
    PILOT_CACHE,
    PRIMARY_CACHE,
    SUBGROUP_DIAGNOSTICS,
    _load_control_rows,
    _validated_cache_hashes,
)
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r04 import (
    FEATURE_NAMES,
    HASHING_PARAMETERS,
    fit_predict,
    hashing_diagnostics,
)
from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import _text_documents
from src.utils.paths import task_files


CARD_ID = "R04"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R04.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R04"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R04"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R04"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUBGROUPS_PATH = REPORT_DIR / "subgroups.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
R02_REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R02"
R02_FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/R02/freeze.json"

TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "ac7fd6372ead62ee527804c9ce14657e96af209d81c9b521259629822853dea8"
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_CYCLES = np.arange(47, 80, dtype=np.int64)
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_SCORE_ROWS = {55: 17_685, 67: 20_903}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064, "H12": 38_588}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
MAX_JOB_GIB = 6.0
MIN_FREE_RAM_GIB = 4.0
RAM_RESERVE_GIB = 2.0
THREADS = 2
R02_CONDITION = "R02 completes and RAM/nnz preflight passes"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _params_json(values: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, tuple):
            output[key] = list(value)
        elif isinstance(value, type):
            output[key] = value.__name__
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(value):
            raise BatchIntegrityError(f"Existing R04 report differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, dict(value))


def _freeze_matches(frozen: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    expected = json_sha256(identity)
    try:
        return frozen.get("signature") == expected and json_sha256(frozen.get("identity")) == expected
    except (TypeError, ValueError):
        return False


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_batch1_r02.py",
        ROOT / "scripts/optimization_v3_reg_diagnostics.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1_r02.py",
        ROOT / "src/models/optimization_v3_batch1_r04.py",
        ROOT / "src/models/optimization_v2_regression.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/utils/paths.py",
        ROOT / "tests/test_optimization_v3_batch1_r04.py",
        ROOT / "tests/test_optimization_v3_batch1_r04_runner.py",
    ]


def _input_files(train_path: Path, card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Path]:
    baseline = queue["baseline"]["regression"]
    return {
        "official_regression_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_card": ROOT / str(card["card_path"]),
        "accepted_regression_manifest": ROOT / str(baseline["manifest"]),
        "cache_only_diagnostics": BASELINE_DIAGNOSTICS,
        "regression_fullsix_report": FULL_REPORT,
        "regression_subgroup_diagnostics": SUBGROUP_DIAGNOSTICS,
        "paired_lgb_ridge_cache": PRIMARY_CACHE,
        "sparse_regression_pilot_cache": PILOT_CACHE,
        "sparse_regression_fullsix_cache": FULL_CACHE,
        "r02_frozen_freeze": R02_FREEZE_PATH,
        "r02_results": R02_REPORT_DIR / "results.json",
        "r02_verification": R02_REPORT_DIR / "verification.json",
        "r02_gate": R02_REPORT_DIR / "gate.json",
    }


def _validate_r02_completion() -> dict[str, Any]:
    results = _read_json(R02_REPORT_DIR / "results.json")
    verification = _read_json(R02_REPORT_DIR / "verification.json")
    gate = _read_json(R02_REPORT_DIR / "gate.json")
    accounting = results.get("fit_accounting", {})
    fits = accounting.get("fits", {})
    if (results.get("card_id") != "R02" or results.get("status") not in {"ADVANCE", "RETIRE"}
            or verification.get("card_id") != "R02" or verification.get("status") != "PASS"
            or gate.get("card_id") != "R02" or gate.get("status") not in {"ADVANCE", "RETIRE"}
            or accounting.get("reserved_predictor_fits") != 2
            or accounting.get("successful_predictor_fits") != 2
            or accounting.get("fit_saved") != 2
            or set(fits) != {"55", "67"}
            or any(fits[key].get("fit_predictor_count") != 1 for key in ("55", "67"))):
        raise BatchIntegrityError("R04 dependency R02 is not a completed, independently verified two-fit pilot")
    freeze = _read_json(R02_FREEZE_PATH)
    identity = freeze.get("identity", {})
    r02_signature = json_sha256(identity)
    if freeze.get("signature") != r02_signature:
        raise BatchIntegrityError("R02 dependency freeze signature does not reconstruct")
    if results.get("freeze_signature") != r02_signature or verification.get("freeze_signature") != r02_signature:
        raise BatchIntegrityError("R02 dependency reports do not reference the current frozen recipe")
    r02_train_hash = identity.get("input_sha256", {}).get("official_regression_train")
    if r02_train_hash != TRAIN_SHA256:
        raise BatchIntegrityError(
            "R02 dependency was frozen against a different official Regression Train hash"
        )
    return {
        "card_id": "R02",
        "status": str(results["status"]),
        "verification_status": str(verification["status"]),
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": 2,
        "cutoffs": [55, 67],
        "freeze_signature": r02_signature,
        "results_sha256": file_sha256(R02_REPORT_DIR / "results.json"),
        "verification_sha256": file_sha256(R02_REPORT_DIR / "verification.json"),
        "gate_sha256": file_sha256(R02_REPORT_DIR / "gate.json"),
        "R02_primary_gain": results.get("views", {}).get("primary", {}).get("gain_positive_is_better"),
        "condition_met": True,
    }


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256:
        raise BatchIntegrityError("Canonical R04 queue hash differs from the reviewed queue")
    if file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("Canonical R04 card hash differs from the reviewed card")
    canonical = next((item for item in queue.get("cards", []) if item.get("id") == CARD_ID), None)
    if canonical is None or dict(canonical) != dict(card):
        raise BatchIntegrityError("R04 card object differs from the canonical queue")
    expected = {
        "id": "R04", "task": "regression", "family": "hashing",
        "policy": "R_SPARSE_REPLACE", "pilot_predictor_fits": 2,
        "evaluation": "STANDARD", "condition": R02_CONDITION,
        "change": "HashingVectorizer word(1,2),n_features262144,alternate_sign=False,norm=None; no learned vocabulary/IDF. Original sparse L1 model. Count collisions/nnz without labels; do not claim unseen-token weights are learned.",
    }
    for key, value in expected.items():
        if card.get(key) != value:
            raise BatchIntegrityError(f"R04 canonical card field changed: {key}")
    expected_hashing = {
        "analyzer": "word", "ngram_range": (1, 2), "n_features": 262_144,
        "alternate_sign": False, "norm": None,
    }
    for key, value in expected_hashing.items():
        actual = HASHING_PARAMETERS.get(key)
        if isinstance(value, tuple):
            actual = tuple(actual) if actual is not None else None
        if actual != value:
            raise BatchIntegrityError(f"R04 HashingVectorizer parameter differs from card: {key}={actual!r}")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R04 feature allowlist includes an identifier or target")
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R04 is valid only while the V3 campaign remains unfinished")
    _validate_r02_completion()


def _independent_hashing_preflight(frame: pd.DataFrame, cycles: np.ndarray,
                                  cutoff: int) -> tuple[dict[str, Any], Any, Any]:
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
    score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
    documents, _missing = _text_documents(frame.iloc[fit_rows]["signal_stream"])
    score_documents, _score_missing = _text_documents(frame.iloc[score_rows]["signal_stream"])
    vectorizer = HashingVectorizer(**dict(HASHING_PARAMETERS))
    # HashingVectorizer has no fitted vocabulary or IDF state. Transform is
    # safe on the fit and score text together; it learns no corpus statistics.
    fit_matrix = vectorizer.transform(documents).tocsr()
    score_matrix = vectorizer.transform(score_documents).tocsr()
    fit_hashing = hashing_diagnostics(documents, fit_matrix, vectorizer=vectorizer)
    score_hashing = hashing_diagnostics(score_documents, score_matrix, vectorizer=vectorizer)
    if fit_matrix.shape[0] != len(fit_rows) or score_matrix.shape[0] != len(score_rows):
        raise BatchIntegrityError(f"R04 hashing CSR rows misalign at cutoff {cutoff}")
    structured_nnz_upper = int(len(fit_rows) * (len(FEATURE_NAMES) + 8))
    nnz_upper = int(fit_matrix.nnz + structured_nnz_upper)
    csr_bytes = int(nnz_upper * 16 + (len(fit_rows) + 1) * 8)
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    estimated_peak_gib = float(1.5 + 3.0 * csr_bytes / (1024 ** 3))
    allowed_gib = float(min(MAX_JOB_GIB, available_gib - RAM_RESERVE_GIB))
    status = "PASS" if (available_gib >= MIN_FREE_RAM_GIB and allowed_gib > 0
                        and estimated_peak_gib <= allowed_gib) else "BLOCKED_RESOURCE"
    diag = {
        "cutoff": int(cutoff), "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
        "hashing_parameters": _params_json(HASHING_PARAMETERS),
        "hashing_learns_vocabulary_or_idf": False,
        "fit_hashing_diagnostics": fit_hashing,
        "score_hashing_diagnostics": score_hashing,
        "structured_nnz_upper_bound": structured_nnz_upper,
        "combined_sparse_design_nnz_upper_bound": nnz_upper,
        "combined_sparse_design_bytes_upper_bound": csr_bytes,
        "estimated_peak_gib": estimated_peak_gib,
        "estimated_peak_is_observed": False,
        "available_ram_gib": available_gib,
        "allowed_job_ram_gib": allowed_gib,
        "minimum_free_ram_gib": MIN_FREE_RAM_GIB,
        "free_ram_reserve_gib": RAM_RESERVE_GIB,
        "maximum_job_gib": MAX_JOB_GIB,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "dense_full_word_matrix": False,
        "status": status,
    }
    return diag, fit_matrix, score_matrix


def _control_metric(target: np.ndarray, cycles: np.ndarray,
                    views: Mapping[str, Any], controls: Mapping[int, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, segments in views.items():
        rows_parts: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for cutoff, low, high in segments:
            item = controls[int(cutoff)]
            rows = np.asarray(item["rows"], dtype=np.int64)
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            rows_parts.append(rows[mask])
            preds.append(np.asarray(item["baseline_prediction"], dtype=np.float64)[mask])
        rows = np.concatenate(rows_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R04 control view denominator/uniqueness differs for {name}")
        prediction = np.concatenate(preds)
        output[name] = {
            "rows": int(len(rows)),
            "mae": float(np.mean(np.abs(target[rows] - prediction))),
        }
    return output


def _hashing_recipe_summary(hashing: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Keep deterministic design diagnostics in the recipe, not live RAM readings."""
    volatile = {"available_ram_gib", "allowed_job_ram_gib", "status"}
    return {
        str(cutoff): {key: value for key, value in diagnostics.items() if key not in volatile}
        for cutoff, diagnostics in hashing.items()
    }


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any],
                    controls: Mapping[str, Any], hashing: Mapping[str, Any],
                    dependency: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(queue["evaluation"]["regression_gate"])
    return {
        "card_id": CARD_ID,
        "queue_recipe": {key: card[key] for key in (
            "id", "task", "name", "family", "priority", "change", "policy",
            "pilot_predictor_fits", "evaluation", "condition",
        )},
        "hashing_vectorizer": {
            "class": "sklearn.feature_extraction.text.HashingVectorizer",
            "parameters": _params_json(HASHING_PARAMETERS),
            "learned_vocabulary": False,
            "learned_idf": False,
            "input": "signal_stream; fit and scoring transforms are stateless",
            "diagnostics_definition": "unique analyzed ngram identities minus occupied hash buckets; CSR nnz counts nonzero document-bucket entries",
            "preflight_by_cutoff": _hashing_recipe_summary(hashing),
        },
        "estimator": {
            "class": "lightgbm.LGBMRegressor",
            "parameters": _params_json(SPARSE_LGB_PARAMETERS),
            "inherited_unchanged_from": "optimization_v2_regression.SPARSE_LGB_PARAMETERS",
            "objective": "regression_l1",
            "input_format": "CSR sparse; no full dense word matrix",
        },
        "structured_preprocessing": "R02 RegressionTextModel._structured_fit/_structured_transform; fit-only imputation/scaling/category OHE at each cutoff",
        "feature_allowlist": list(FEATURE_NAMES),
        "fit_population": "all official Regression Train rows with 47 <= cycle_code <= cutoff; no weighting/downsampling",
        "scoring_population": "STANDARD cutoff-55/67 complete 12-cycle horizons, row-aligned by Train row positions and record_id",
        "prediction_policy": {
            "baseline": "frozen Ridge25 plus 25% rank50(original sparse L1 raw prediction)",
            "candidate": "clip(0.75 * frozen Ridge25 + 0.25 * rank50(new R04 raw sparse prediction), 0, 100)",
            "rank50": "per complete scoring cycle; average-rank ties; mapped=0.5*raw+0.5*100*average_rank/n; clip[0,100]",
        },
        "evaluation_views": {name: [list(segment) for segment in segments] for name, segments in views.items()},
        "row_matched_baseline_metrics": dict(controls),
        "accepted_fullsix_control_metrics": {
            "primary_mae": float(queue["baseline"]["regression"]["primary"]),
            "H12_mae": float(queue["baseline"]["regression"]["H12"]),
        },
        "R02_completion_condition": dict(dependency),
        "pilot_gate": gates,
        "subgroup_support_rule": "n>=100 for stream-missing/tail harm guards; smaller groups are descriptive",
        "resource_policy": {
            "threads": THREADS, "concurrent_training_jobs": 1,
            "job_memory_cap_gib": MAX_JOB_GIB,
            "free_memory_reserve_gib": RAM_RESERVE_GIB,
        },
        "test_data_used": False,
        "kaggle_access": False,
    }


def _budget_preflight() -> dict[str, Any]:
    events = []
    if LEDGER_PATH.exists():
        events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    reserved = sum(int(event.get("predictor_fits", 0)) for event in events if event.get("event") == "BUDGET_RESERVED")
    own = [event for event in events if event.get("event") == "BUDGET_RESERVED"
           and event.get("budget_key") == f"{CARD_ID}:PILOT"]
    cap = int(load_queue(QUEUE_PATH)["execution_preset"]["max_successful_new_predictor_fits"])
    if len(own) > 1 or (own and int(own[0].get("predictor_fits", -1)) != 2):
        raise BatchIntegrityError("R04 budget reservation is duplicated or has the wrong fit count")
    next_total = reserved if own else reserved + 2
    return {
        "status": "PASS" if next_total <= cap else "BLOCKED_BUDGET",
        "reserved_total": reserved, "next_total": next_total,
        "existing_r04_reservation": own[0] if own else None, "cap": cap,
    }


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    dependency = _validate_r02_completion()
    train_path = task_files("regression").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    train_path = Path(train_path)
    frame = pd.read_csv(train_path)
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from the frozen queue")
    if len(frame) != EXPECTED_TRAIN_ROWS or "threat_rank" not in frame.columns:
        raise BatchIntegrityError("Official Regression Train schema or row count differs")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any() or frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Regression Train record_id is missing or duplicated")
    absent = sorted(set(FEATURE_NAMES) - set(frame.columns))
    if absent or set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError(f"R04 feature allowlist is invalid; missing={absent}")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), EXPECTED_CYCLES):
        raise BatchIntegrityError("Official Regression Train cycle distribution differs from 47-79")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise BatchIntegrityError("Official Regression Train threat_rank is invalid")

    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R04 STANDARD views differ from the frozen queue: {views}")
    input_files = _input_files(train_path, card, queue)
    cache_hashes = _validated_cache_hashes(queue, input_files)
    controls = _load_control_rows(frame, views)
    expected_cutoffs = (55, 67)
    for cutoff in expected_cutoffs:
        if len(controls[cutoff]["rows"]) != EXPECTED_SCORE_ROWS[cutoff]:
            raise BatchIntegrityError(f"R04 control score rows differ at cutoff {cutoff}")
    if any(len(np.flatnonzero(cycles <= cutoff)) != EXPECTED_FIT_ROWS[cutoff] for cutoff in expected_cutoffs):
        raise BatchIntegrityError("R04 temporal fit-row counts differ from the card")

    hashing: dict[str, Any] = {}
    for cutoff in expected_cutoffs:
        diagnostics, _fit_matrix, _score_matrix = _independent_hashing_preflight(frame, cycles, cutoff)
        if diagnostics["fit_rows"] != EXPECTED_FIT_ROWS[cutoff] or diagnostics["score_rows"] != EXPECTED_SCORE_ROWS[cutoff]:
            raise BatchIntegrityError(f"R04 hashing CSR support counts differ at cutoff {cutoff}")
        if diagnostics["status"] != "PASS":
            raise BatchIntegrityError(f"R04 RAM/nnz preflight blocked at cutoff {cutoff}: {diagnostics}")
        hashing[str(cutoff)] = diagnostics

    row_metrics: dict[str, Any] = {}
    for name, segments in views.items():
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            saved = controls[int(cutoff)]
            rows = saved["rows"]
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            row_parts.append(rows[mask])
            pred_parts.append(saved["baseline_prediction"][mask])
        rows = np.concatenate(row_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R04 baseline denominator/uniqueness differs for {name}")
        predictions = np.concatenate(pred_parts)
        row_metrics[name] = {"rows": int(len(rows)), "mae": float(np.mean(np.abs(target[rows] - predictions)))}
    if not np.isclose(row_metrics["H12"]["mae"], float(queue["baseline"]["regression"]["H12"]), atol=1e-10, rtol=0.0):
        raise BatchIntegrityError("R04 row-matched H12 baseline differs from the frozen accepted control")

    recipe = _recipe_payload(card, views, row_metrics, hashing, dependency, queue)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=_source_files(), evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "historical_verified_cache_sha256": cache_hashes,
        "R02_condition_evidence": dependency,
        "recipe": recipe,
        "recipe_artifact_sha256": json_sha256(recipe),
        "expected_fit_rows": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
        "expected_score_rows": {str(k): v for k, v in EXPECTED_SCORE_ROWS.items()},
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "python_version": platform.python_version(),
    })
    signature = json_sha256(identity)
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise BatchIntegrityError("Existing R04 recipe differs from current frozen preflight")
    if FREEZE_PATH.exists() and not _freeze_matches(_read_json(FREEZE_PATH), identity):
        raise BatchIntegrityError("Existing R04 freeze differs from current source/input/recipe identity")
    budget = _budget_preflight()
    if budget["status"] != "PASS":
        raise BatchIntegrityError(f"R04 budget preflight failed: {budget}")
    public = {
        "card_id": CARD_ID, "status": "PREFLIGHT_PASS",
        "official_train_sha256": TRAIN_SHA256,
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in _source_files()},
        "input_sha256": {key: file_sha256(path) for key, path in sorted(input_files.items())},
        "freeze_identity_sha256": signature,
        "R02_condition": dependency,
        "fit_rows": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
        "score_rows": {str(k): v for k, v in EXPECTED_SCORE_ROWS.items()},
        "view_rows": EXPECTED_VIEW_ROWS,
        "row_matched_baseline_metrics": row_metrics,
        "hashing_preflight": hashing,
        "budget": budget,
        "test_data_read": False, "kaggle_access": False,
        "candidate_fit_performed": False, "freeze_written": False,
    }
    return {
        "public": public, "identity": identity, "recipe": recipe,
        "frame": frame, "target": target, "cycles": cycles,
        "views": views, "controls": controls, "train_path": train_path,
        "hashing": hashing, "dependency": dependency,
    }


def _resource_guarded_adapter(fit_features: pd.DataFrame, fit_target: np.ndarray,
                              score_features: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    cycles = pd.to_numeric(fit_features["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    cutoff = int(cycles.max())
    diag, _fit_matrix, _score_matrix = _independent_hashing_preflight(
        pd.concat([fit_features, score_features], axis=0),
        np.concatenate([cycles, pd.to_numeric(score_features["cycle_code"], errors="raise").to_numpy(dtype=np.int64)]),
        cutoff,
    )
    if diag["status"] != "PASS":
        raise BatchIntegrityError(f"R04 per-fit resource guard failed: {diag}")
    result = fit_predict(fit_features, fit_target, score_features, **kwargs)
    result.setdefault("fit_metadata", {})["resource_preflight"] = diag
    return result


def _control_loader(controls: Mapping[int, Mapping[str, np.ndarray]]):
    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        if int(cutoff) not in controls:
            raise BatchIntegrityError(f"No frozen R04 baseline controls for cutoff {cutoff}")
        record = controls[int(cutoff)]
        if not np.array_equal(record["rows"], np.asarray(score_rows, dtype=np.int64)):
            raise BatchIntegrityError(f"R04 baseline control rows differ at cutoff {cutoff}")
        return {key: value for key, value in record.items() if key not in {
            "rows", "baseline_sparse_rank50", "ridge25_reconstruction_max_abs_error"
        }}
    return load


def _reconstruct_r04(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw_candidate = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    raw_original = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    ridge25 = np.asarray(arrays["control__ridge25_prediction"], dtype=np.float64)
    cycles = np.asarray(arrays["control__cycle_code"], dtype=np.int64)
    if not raw_candidate.shape or len({len(raw_candidate), len(raw_original), len(ridge25), len(cycles)}) != 1:
        raise BatchIntegrityError("R04 independent verifier inputs are empty or misaligned")

    def map_rank(raw: np.ndarray) -> np.ndarray:
        mapped = np.empty(len(raw), dtype=np.float64)
        for cycle in np.unique(cycles):
            mask = cycles == cycle
            n = int(mask.sum())
            percentile = 100.0 * rankdata(raw[mask], method="average") / n
            mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
        return np.clip(mapped, 0.0, 100.0)

    candidate = np.clip(0.75 * ridge25 + 0.25 * map_rank(raw_candidate), 0.0, 100.0)
    baseline = np.clip(0.75 * ridge25 + 0.25 * map_rank(raw_original), 0.0, 100.0)
    return candidate, baseline


def _subgroup_diagnostics(report: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
                          views: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    records: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in report["cutoffs"]:
        path = RUN_DIR / f"R04_fit{int(cutoff)}.npz"
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        arrays, _ = read_bundle(path, sidecar["identity"])
        records[int(cutoff)] = {
            "rows": np.asarray(arrays["score_rows"], dtype=np.int64),
            "prediction": np.asarray(arrays["prediction"], dtype=np.float64),
            "baseline": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
        }
    output: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for view_name in ("primary", "H12"):
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in views[view_name]:
            item = records[int(cutoff)]
            mask = (cycles[item["rows"]] >= int(low)) & (cycles[item["rows"]] <= int(high))
            row_parts.append(item["rows"][mask])
            pred_parts.append(item["prediction"][mask])
            base_parts.append(item["baseline"][mask])
        rows = np.concatenate(row_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[view_name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R04 subgroup rows do not match the {view_name} view")
        y = target[rows]
        candidate = np.concatenate(pred_parts)
        baseline = np.concatenate(base_parts)
        local = frame.iloc[rows]
        text = local["signal_stream"].astype("string").str.strip()
        missing = text.isna() | text.eq("")
        entry = pd.to_numeric(local["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        groups: dict[str, np.ndarray] = {
            "entry_friction_0": entry == 0.0,
            "entry_friction_1": entry == 1.0,
            "stream_present": ~missing.to_numpy(dtype=bool),
            "stream_missing": missing.to_numpy(dtype=bool),
            "target_le_5": y <= 5.0,
            "target_ge_95": y >= 95.0,
        }
        local_out: dict[str, Any] = {}
        for name, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            n = int(mask.sum())
            harm = (float(np.mean(np.abs(candidate[mask] - y[mask]))
                          - np.mean(np.abs(baseline[mask] - y[mask]))) if n else None)
            supported = n >= 100
            guard = name in {"stream_missing", "target_le_5", "target_ge_95"}
            local_out[name] = {
                "n": n, "supported": supported,
                "evidence_status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                "mae_harm_candidate_minus_control": harm,
                "numeric_guard_applies": guard,
            }
            if guard and supported and harm is not None and harm > 0.25:
                failures.append({"view": view_name, "group": name, "n": n, "mae_harm": harm})
        output[view_name] = local_out
    return {
        "minimum_support": 100,
        "maximum_supported_tail_or_missing_MAE_harm": 0.25,
        "views": output,
        "supported_guard_failures": failures,
    }


def _verify_and_gate(context: Mapping[str, Any], report: Mapping[str, Any],
                     queue: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verification = verify_report(
        report=dict(report), frame=context["frame"], target=context["target"],
        artifact_dir=RUN_DIR, freeze_signature=json_sha256(context["identity"]),
        evaluation_views=context["views"], prediction_reconstructors={"R04": _reconstruct_r04},
    )
    subgroup = _subgroup_diagnostics(report, context["frame"], context["target"], context["views"], queue)
    gate_config = queue["evaluation"]["regression_gate"]
    view = report["views"]
    checks = {
        "primary_mae_gain_at_least_0_03": bool(view["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_MAE_gain_min"])),
        "H12_harm_at_most_0_05": bool(view["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_MAE_harm_max"])),
        "W1_harm_at_most_0_5": bool(view["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"])),
        "W4_harm_at_most_0_5": bool(view["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"])),
        "supported_tail_or_missing_group_harms": not subgroup["supported_guard_failures"],
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    verification.update({
        "status": "PASS", "decision": decision,
        "gate_checks": checks, "subgroup_diagnostics": subgroup,
        "test_data_used": False, "kaggle_access": False,
    })
    gate = {
        "card_id": CARD_ID, "decision": decision,
        "status": decision,
        "checks": checks, "thresholds": gate_config,
        "supported_subgroup_failures": subgroup["supported_guard_failures"],
        "decision_basis": "Frozen STANDARD temporal metrics, exact row-matched Ridge25+sparse controls, independent route reconstruction and predeclared tail/missing guards; no Test/Kaggle evidence.",
    }
    return verification, subgroup, gate


def _fit_accounting(context: Mapping[str, Any]) -> dict[str, Any]:
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    card_events = [event for event in events if event.get("card_id") == CARD_ID]
    reservations = [event for event in card_events if event.get("event") == "BUDGET_RESERVED"]
    if len(reservations) != 1 or reservations[0].get("budget_key") != "R04:PILOT" or reservations[0].get("predictor_fits") != 2:
        raise BatchIntegrityError("R04 ledger does not contain its exact two-fit reservation")
    folds: dict[str, Any] = {}
    for cutoff in (55, 67):
        expected_rows = np.flatnonzero((context["cycles"] > cutoff) & (context["cycles"] <= cutoff + 12))
        identity = {"freeze_signature": json_sha256(context["identity"]), "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": expected_rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"R04_fit{cutoff}.npz", identity)
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        fit_meta = metadata.get("fit_metadata", {})
        if (metadata.get("fit_predictor_count") != 1
                or metadata.get("fit_rows") != EXPECTED_FIT_ROWS[cutoff]
                or metadata.get("score_rows") != EXPECTED_SCORE_ROWS[cutoff]
                or state.get("id") != "V3_BATCH1_R04_HASHING_SPARSE_L1"
                or state.get("learned_vocabulary") is not False
                or state.get("learned_idf") is not False
                or state.get("hashing_vectorizer_parameters") != _params_json(HASHING_PARAMETERS)
                or state.get("parameters") != _params_json(SPARSE_LGB_PARAMETERS)
                or fit_meta.get("validation_labels_passed_to_estimator") is not False
                or not np.isfinite(arrays["prediction"]).all()
                or np.any((arrays["prediction"] < 0) | (arrays["prediction"] > 100))):
            raise BatchIntegrityError(f"R04 saved fit receipt invalid at cutoff {cutoff}")
        fold_events = [event for event in card_events if event.get("cutoff") == cutoff]
        started = sum(event.get("event") == "FIT_STARTED" for event in fold_events)
        saved = sum(event.get("event") == "FIT_SAVED" for event in fold_events)
        resumed = sum(event.get("event") == "RESUMED_VERIFIED" for event in fold_events)
        failed = sum(event.get("event") == "FIT_FAILED" for event in fold_events)
        if saved > 1 or started < 1 or saved + resumed < 1 or failed > started:
            raise BatchIntegrityError(f"R04 fit ledger events are invalid at cutoff {cutoff}")
        folds[str(cutoff)] = {
            "fit_rows": EXPECTED_FIT_ROWS[cutoff], "score_rows": EXPECTED_SCORE_ROWS[cutoff],
            "fit_seconds": float(metadata["fit_seconds"]), "fit_attempts": int(started),
            "failed_attempts": int(failed), "fit_predictor_count": 1,
            "training_warnings": metadata.get("fit_metadata", {}).get("python_warnings", []),
            "resource_preflight": fit_meta.get("resource_preflight", {}),
            "design_nnz": state.get("training_design_nnz"),
            "model_state_sha256": state.get("state_sha256"),
            "bundle_npz_sha256": sidecar.get("sha256"),
        }
    return {
        "reserved_predictor_fits": 2, "successful_predictor_fits": 2,
        "folds": folds, "total_fit_seconds": float(sum(row["fit_seconds"] for row in folds.values())),
        "peak_memory_observed": False,
    }


def _publish(context: Mapping[str, Any], report: Mapping[str, Any], verification: Mapping[str, Any],
             subgroup: Mapping[str, Any], gate: Mapping[str, Any], accounting: Mapping[str, Any]) -> None:
    result = dict(report)
    result.update({
        "status": str(gate["decision"]),
        "input_sha256": context["public"]["input_sha256"],
        "source_sha256": context["public"]["source_sha256"],
        "preflight": context["public"],
        "R02_completion_condition": context["dependency"],
        "fit_accounting": dict(accounting),
        "resource_evidence_limit": "Per-fit memory is a pre-fit upper-bound estimate; peak process RSS was not observed.",
        "independent_verification": dict(verification),
        "subgroups": dict(subgroup),
        "gate": dict(gate),
        "test_data_used": False, "sample_submission_used": False, "kaggle_used": False,
    })
    summary = {
        "card_id": CARD_ID, "status": str(gate["decision"]),
        "primary_rows": EXPECTED_VIEW_ROWS["primary"],
        "primary_baseline_mae": float(report["views"]["primary"]["baseline"]["mae"]),
        "primary_candidate_mae": float(report["views"]["primary"]["candidate"]["mae"]),
        "primary_gain": float(report["views"]["primary"]["gain_positive_is_better"]),
        "H12_rows": EXPECTED_VIEW_ROWS["H12"],
        "H12_baseline_mae": float(report["views"]["H12"]["baseline"]["mae"]),
        "H12_candidate_mae": float(report["views"]["H12"]["candidate"]["mae"]),
        "H12_gain": float(report["views"]["H12"]["gain_positive_is_better"]),
        "W1_gain": float(report["views"]["W1"]["gain_positive_is_better"]),
        "W4_gain": float(report["views"]["W4"]["gain_positive_is_better"]),
        "supported_group_failures": len(subgroup["supported_guard_failures"]),
        "fit_predictor_count": 2, "total_fit_seconds": accounting["total_fit_seconds"],
        "test_sample_kaggle_used": False,
        "caveat": "Hash collisions can merge ngram identities; stateless hashing does not learn unseen-token weights or IDF.",
    }
    _write_or_match(RESULTS_PATH, result)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(SUBGROUPS_PATH, subgroup)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(SUMMARY_PATH, summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one R04 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if freeze_only:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_match(RECIPE_PATH, context["recipe"])
        from src.models.optimization_v3_batch1 import ensure_frozen
        ensure_frozen(FREEZE_PATH, context["identity"])
        output = dict(context["public"])
        output.update({
            "status": "FROZEN_READY",
            "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
            "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
            "freeze_sha256": file_sha256(FREEZE_PATH),
            "candidate_fit_performed": False, "freeze_written": True,
        })
        return output
    if not RECIPE_PATH.is_file() or not FREEZE_PATH.is_file():
        raise BatchIntegrityError("R04 pilot requires a committed recipe and prefit freeze")
    if _read_json(RECIPE_PATH) != context["recipe"] or not _freeze_matches(_read_json(FREEZE_PATH), context["identity"]):
        raise BatchIntegrityError("R04 freeze or recipe changed after protocol commit")
    report = run_pilot(
        frame=context["frame"], target=context["target"], card=card,
        feature_names=FEATURE_NAMES, adapter=_resource_guarded_adapter,
        control_loader=_control_loader(context["controls"]),
        freeze_identity=context["identity"], freeze_path=FREEZE_PATH,
        artifact_dir=RUN_DIR, event_path=LEDGER_PATH,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        evaluation_views=context["views"], threads=THREADS,
        report_path=ENGINE_REPORT_PATH,
    )
    verification, subgroup, gate = _verify_and_gate(context, report, queue)
    accounting = _fit_accounting(context)
    _publish(context, report, verification, subgroup, gate, accounting)
    return {
        "card_id": CARD_ID, "status": f"PILOT_COMPLETE_{gate['decision']}",
        "decision": gate["decision"], "views": verification["views"],
        "gate_checks": gate["checks"], "fit_accounting": accounting,
        "freeze_signature": json_sha256(context["identity"]),
        "test_data_used": False, "kaggle_access": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if card is None:
        raise BatchIntegrityError("R04 is missing from the canonical queue")
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_reconstruct_r04", "run_card"]
