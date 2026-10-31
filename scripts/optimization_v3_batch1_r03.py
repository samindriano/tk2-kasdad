"""R03 read-only preflight and frozen pilot runner.

Default execution is intentionally unavailable: MAIN must review this source
and explicitly invoke --run-frozen-pilot after the preflight has been accepted.
--preflight-only reads official Train and frozen caches, but writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.optimization_v3_reg_diagnostics import (
    FULL_CACHE, PILOT_CACHE, PRIMARY_CACHE,
)
from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    ensure_frozen,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r03 import (
    CARD_ID,
    EXPECTED_SCORE_CYCLES,
    FEATURE_NAMES,
    estimate_sparse_memory,
    fit_predict as r03_fit_predict,
    split_tfidf_parameters,
)
from src.models.regression_research_text import (
    CATEGORICAL_FEATURES,
)
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R03.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R03"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R03"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R03"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "c16de95d965865a6fd56bcfa249042455e4a73102a7b649cc8abda373a3d4c97"
MANIFEST_SHA256 = "dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a"
FEASIBILITY_PATH = ROOT / "reports/optimization_v3/research_backlog/feasibility.json"
FULLSIX_FREEZE_PATH = ROOT / "experiments/optimization_v2/regression_repr/fullsix_freeze.json"
REGRESSION_REPORT_PATH = ROOT / "reports/optimization_v2/regression_repr/fullsix.json"
PILOT_REPORT_PATH = ROOT / "reports/optimization_v2/regression_repr/pilot.json"
PILOT_VERIFICATION_PATH = ROOT / "reports/optimization_v2/regression_repr/verification.json"
MANIFEST_PATH = ROOT / "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json"
REGRESSION_REPORT_SHA256 = "a9686473998722b8f86a0f6a4489b4e3a75a6b6ba20c661ac564e6a5e61aea48"
MIN_GROUP_SUPPORT = 100
MAX_SUCCESSFUL_FITS = 180
THREADS = 2
JOB_CAP_BYTES = 6 * 1024 ** 3
FREE_RAM_RESERVE_BYTES = 2 * 1024 ** 3
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_CONTROL_METRICS = {"primary_mae": 11.559390078665364,
                            "H12_mae": 14.04753258299653}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064,
                      "H12_E": 17_685, "H12_L": 20_903, "H12": 38_588}
EXPECTED_CARD = {
    "id": "R03",
    "task": "regression",
    "name": "Split unigram/bigram budgets",
    "family": "vocab_allocation",
    "priority": "P1",
    "change": "Two regression TF-IDF blocks: unigram cap30000 and bigram cap30000, min_df2/sublinear, otherwise original defaults. Same structured/OHE and sparse L1 model.",
    "policy": "R_SPARSE_REPLACE",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "card_path": "reports/optimization_v3/research_backlog/cards/R03.md",
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
EXPECTED_GATE = {
    "pilot_primary_MAE_gain_min": 0.03,
    "pilot_H12_MAE_harm_max": 0.05,
    "individual_pilot_view_MAE_harm_max": 0.5,
    "supported_group_n_min": 100,
    "supported_tail_or_missing_MAE_harm_max": 0.25,
}
INPUTS = {
    "official_regression_train": ROOT / "data/raw/regression/Train.csv",
    "canonical_queue": QUEUE_PATH,
    "canonical_card": CARD_PATH,
    "accepted_v2_reg_sparse_lgb25_manifest": MANIFEST_PATH,
    "ridge25_and_paired_prediction_cache": PRIMARY_CACHE,
    "accepted_sparse_pilot_prediction_cache": PILOT_CACHE,
    "accepted_sparse_pilot_report": PILOT_REPORT_PATH,
    "accepted_sparse_pilot_verification": PILOT_VERIFICATION_PATH,
    "accepted_sparse_fullsix_prediction_cache": FULL_CACHE,
    "accepted_sparse_fullsix_report": REGRESSION_REPORT_PATH,
    "accepted_sparse_fullsix_freeze": FULLSIX_FREEZE_PATH,
    "read_only_feasibility_report": FEASIBILITY_PATH,
}
SOURCES = [
    Path(__file__),
    ROOT / "src/models/optimization_v3_batch1_r03.py",
    ROOT / "tests/test_optimization_v3_batch1_r03.py",
    ROOT / "src/models/optimization_v3_batch1.py",
    ROOT / "scripts/optimization_v3_batch1_verify.py",
    ROOT / "src/models/optimization_v2_regression.py",
    ROOT / "src/models/regression_research_text.py",
    ROOT / "src/models/regression_research_tabular.py",
    ROOT / "src/models/regression_round2_models.py",
    ROOT / "src/models/round1_models.py",
    ROOT / "scripts/optimization_v3_reg_diagnostics.py",
    ROOT / "src/utils/paths.py",
]
PACKAGE_NAMES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        if not np.isfinite(out):
            raise BatchIntegrityError("Nonfinite value in R03 JSON payload")
        return out
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, type):
        return value.__name__
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"Unsupported R03 JSON field {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_recipe_artifact(path: Path, expected_recipe: Mapping[str, Any],
                              expected_semantic_sha256: str) -> None:
    """Validate parsed recipe content against its canonical semantic digest.

    Recipe files are pretty-printed for review, while ``json_sha256`` hashes
    canonical compact JSON. Hashing the formatted file bytes would compare two
    different serialization formats and reject a valid recipe artifact.
    """
    persisted = _read_json(path)
    if (persisted != dict(expected_recipe)
            or json_sha256(persisted) != expected_semantic_sha256):
        raise BatchIntegrityError("Persisted R03 recipe differs from preflight freeze identity")


def _validate_card_and_queue(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("R03 canonical queue or card bytes differ from the reviewed recipe")
    if load_queue(QUEUE_PATH) != dict(queue):
        raise BatchIntegrityError("Loaded R03 queue differs from canonical queue bytes")
    canonical = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if canonical is None or canonical != dict(card):
        raise BatchIntegrityError("R03 card object differs from its canonical queue entry")
    for key, expected in EXPECTED_CARD.items():
        if card.get(key) != expected:
            raise BatchIntegrityError(f"R03 queue card changed in field {key}")
    if int(card.get("pilot_predictor_fits", -1)) != 2:
        raise BatchIntegrityError("R03 must reserve exactly two candidate predictor fits")
    if queue["baseline"]["regression"].get("Train_sha256") != TRAIN_SHA256:
        raise BatchIntegrityError("Canonical Regression Train hash differs")
    if queue["baseline"]["regression"].get("manifest_sha256") != MANIFEST_SHA256:
        raise BatchIntegrityError("Canonical accepted regression manifest hash differs")
    baseline = queue["baseline"]["regression"]
    if (float(baseline.get("primary", float("nan"))) != EXPECTED_CONTROL_METRICS["primary_mae"]
            or float(baseline.get("H12", float("nan"))) != EXPECTED_CONTROL_METRICS["H12_mae"]):
        raise BatchIntegrityError("Canonical accepted control metrics changed")
    gate = queue["evaluation"]["regression_gate"]
    for key in ("pilot_primary_MAE_gain_min", "pilot_H12_MAE_harm_max",
                "individual_pilot_view_MAE_harm_max", "supported_group_n_min",
                "supported_tail_or_missing_MAE_harm_max"):
        if float(gate[key]) != float(EXPECTED_GATE[key]):
            raise BatchIntegrityError(f"R03 canonical gate changed in field {key}")


def _read_official_train() -> tuple[Path, pd.DataFrame, np.ndarray]:
    """Read only official Regression Train features/IDs/cycles; never targets."""
    paths = task_files("regression")
    train_path = paths.get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Regression Train is unavailable")
    train_path = Path(train_path)
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from R03 card")
    usecols = ["record_id", *FEATURE_NAMES]
    frame = pd.read_csv(train_path, usecols=usecols)
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Expected 50,608 official Train rows, got {len(frame)}")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise BatchIntegrityError("Official Train record_id contains missing/blank values")
    record_ids = frame["record_id"].astype(str)
    if record_ids.duplicated().any():
        raise BatchIntegrityError("Official Train record_id is not unique")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Train cycles differ from expected 47–79")
    if "threat_rank" in frame.columns or "artifact_signal" in frame.columns:
        raise BatchIntegrityError("A protected target leaked into the R03 preflight feature frame")
    if (set(frame.columns) - {"record_id"}) != set(FEATURE_NAMES):
        raise BatchIntegrityError("Official Train feature allowlist changed")
    return train_path, frame, cycles


def _selected_npz(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys) - set(archive.files))
        if missing:
            raise BatchIntegrityError(f"Required cached keys are absent from {path}: {missing}")
        return {key: archive[key] for key in keys}


def _rank50_independent(raw: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Independent preflight/verifier implementation of the frozen rank policy."""
    raw = np.asarray(raw, dtype=np.float64)
    cycle = np.asarray(cycle, dtype=np.int64)
    if raw.ndim != 1 or cycle.shape != raw.shape or len(raw) == 0:
        raise BatchIntegrityError("Sparse prediction/cycle arrays are empty or misaligned")
    if not np.isfinite(raw).all():
        raise BatchIntegrityError("Cached sparse raw predictions contain nonfinite values")
    ranked = np.empty(raw.shape, dtype=np.float64)
    for current in np.unique(cycle):
        selected = cycle == current
        count = int(selected.sum())
        percentile = 100.0 * rankdata(raw[selected], method="average") / count
        ranked[selected] = 0.5 * raw[selected] + 0.5 * percentile
    ranked = np.clip(ranked, 0.0, 100.0)
    if not np.isfinite(ranked).all():
        raise BatchIntegrityError("Cached sparse rank50 component is nonfinite")
    return ranked


def _hash_ordered_ids(values: np.ndarray) -> str:
    payload = "\n".join(map(str, values.tolist())).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_and_historical_freeze() -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    if file_sha256(MANIFEST_PATH) != MANIFEST_SHA256:
        raise BatchIntegrityError("Accepted v2_reg_sparse_lgb25 manifest hash differs")
    if manifest.get("status") != "PREPARED_AND_QA_PASS" or manifest.get("candidate_id") != "v2_reg_sparse_lgb25":
        raise BatchIntegrityError("Accepted regression incumbent manifest is not QA-passed")
    components = manifest.get("recipe", {}).get("components", [])
    expected = {
        ("lgb", "base"): (0.5, 0.5625),
        ("ridge", None): (0.75, 0.1875),
        ("lgbm_sparse_l1", None): (0.5, 0.25),
    }
    observed = {}
    for item in components:
        observed[(item.get("model"), item.get("params_id"))] = (
            float(item.get("rank_weight")), float(item.get("weight")))
    if observed != expected:
        raise BatchIntegrityError("Accepted incumbent blend recipe differs from the R03 frozen control")
    freeze = _read_json(FULLSIX_FREEZE_PATH)
    if file_sha256(FULLSIX_FREEZE_PATH) != "bb38f5fb7df09f0642ad18f064373f2b7051ed64e8b5d260ccb2d31b84ef909c":
        raise BatchIntegrityError("Regression full-six control freeze hash differs")
    ridge_cache_hash = freeze.get("existing_ridge25", {}).get("cache_sha256")
    pilot_cache_hash = freeze.get("existing_pilot", {}).get("prediction_cache_sha256")
    if ridge_cache_hash != file_sha256(PRIMARY_CACHE) or pilot_cache_hash != file_sha256(PILOT_CACHE):
        raise BatchIntegrityError("Historical freeze does not bind Ridge25 and sparse pilot caches")
    if freeze.get("existing_pilot", {}).get("verified_status") != "PASS":
        raise BatchIntegrityError("Historical sparse pilot verification is not PASS")
    if (file_sha256(PILOT_REPORT_PATH) != freeze.get("existing_pilot", {}).get("report_sha256")
            or file_sha256(PILOT_VERIFICATION_PATH) != freeze.get("existing_pilot", {}).get("verification_sha256")):
        raise BatchIntegrityError("Historical sparse pilot report/verification hashes differ from freeze")
    fullsix_report = _read_json(REGRESSION_REPORT_PATH)
    if (file_sha256(REGRESSION_REPORT_PATH) != REGRESSION_REPORT_SHA256
            or fullsix_report.get("continuation_freeze_sha256") != file_sha256(FULLSIX_FREEZE_PATH)
            or fullsix_report.get("all_six_primary_windows_and_both_H12_views_completed") is not True
            or fullsix_report.get("control_fit_count") != 0
            or fullsix_report.get("new_prediction_cache_path") != "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
            or fullsix_report.get("new_prediction_cache_sha256") != file_sha256(FULL_CACHE)):
        raise BatchIntegrityError("Historical full-six report/cache lineage differs from its freeze")
    return {
        "manifest_status": manifest["status"],
        "manifest_recipe_id": manifest["candidate_id"],
        "fullsix_freeze_sha256": file_sha256(FULLSIX_FREEZE_PATH),
        "ridge25_cache_sha256": ridge_cache_hash,
        "old_sparse_pilot_cache_sha256": pilot_cache_hash,
        "old_sparse_pilot_verified": True,
        "pilot_report_sha256": file_sha256(PILOT_REPORT_PATH),
        "pilot_verification_sha256": file_sha256(PILOT_VERIFICATION_PATH),
        "fullsix_report_sha256": file_sha256(REGRESSION_REPORT_PATH),
        "fullsix_prediction_cache_sha256": file_sha256(FULL_CACHE),
    }


def _load_control_arrays(frame: pd.DataFrame, cycles: np.ndarray,
                         views: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    control: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        expected_cycles = np.asarray(EXPECTED_SCORE_CYCLES[cutoff], dtype=np.int64)
        score_rows = np.flatnonzero(np.isin(cycles, expected_cycles))
        if not np.array_equal(np.unique(cycles[score_rows]), expected_cycles):
            raise BatchIntegrityError(f"Cutoff {cutoff} score cycles are incomplete")
        expected_fit_rows = int(np.sum(cycles <= cutoff))
        if expected_fit_rows != EXPECTED_FIT_ROWS[cutoff]:
            raise BatchIntegrityError(f"Cutoff {cutoff} fit rows changed: {expected_fit_rows}")
        dual_keys = [f"rows_{cutoff}", f"blend_ridge25_{cutoff}"]
        sparse_keys = [f"rows_lgbm_sparse_l1_{cutoff}", f"raw_lgbm_sparse_l1_{cutoff}"]
        dual = _selected_npz(PRIMARY_CACHE, dual_keys)
        old_sparse = _selected_npz(PILOT_CACHE, sparse_keys)
        dual_rows = np.asarray(dual[f"rows_{cutoff}"], dtype=np.int64)
        sparse_rows = np.asarray(old_sparse[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
        ridge = np.asarray(dual[f"blend_ridge25_{cutoff}"], dtype=np.float64)
        sparse_raw = np.asarray(old_sparse[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
        if (not np.array_equal(dual_rows, score_rows)
                or not np.array_equal(sparse_rows, score_rows)
                or len(np.unique(dual_rows)) != len(dual_rows)
                or len(np.unique(sparse_rows)) != len(sparse_rows)):
            raise BatchIntegrityError(f"Cutoff {cutoff} frozen control row identities/order differ")
        if ridge.shape != score_rows.shape or sparse_raw.shape != score_rows.shape:
            raise BatchIntegrityError(f"Cutoff {cutoff} frozen control arrays are misaligned")
        if not np.isfinite(ridge).all() or np.any((ridge < 0) | (ridge > 100)):
            raise BatchIntegrityError(f"Cutoff {cutoff} Ridge25 cache is invalid")
        row_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
        if len(np.unique(row_ids)) != len(row_ids):
            raise BatchIntegrityError(f"Cutoff {cutoff} score record IDs are duplicated")
        cycles_here = cycles[score_rows]
        old_rank = _rank50_independent(sparse_raw, cycles_here)
        old_baseline = np.clip(0.75 * ridge + 0.25 * old_rank, 0.0, 100.0)
        if not np.isfinite(old_baseline).all():
            raise BatchIntegrityError(f"Cutoff {cutoff} incumbent reconstruction is nonfinite")
        control[cutoff] = {
            "score_rows": score_rows,
            "score_record_ids": row_ids,
            "score_cycle": cycles_here,
            "ridge25_prediction": ridge,
            "old_sparse_raw": sparse_raw,
            "old_sparse_rank50": old_rank,
            "baseline_prediction": old_baseline,
        }

    # Exact per-view row sets/denominators are checked without opening labels.
    expected_by_view = {"W1": (55, 56, 59), "W4": (67, 68, 71)}
    for name, (cutoff, low, high) in expected_by_view.items():
        rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        score_rows = control[cutoff]["score_rows"]
        if not np.array_equal(score_rows[(cycles[score_rows] >= low) & (cycles[score_rows] <= high)], rows):
            raise BatchIntegrityError(f"{name} row order differs from official Train")
        if len(rows) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"{name} support differs from frozen denominator")
    for name, segments in views.items():
        rows_for_view = [np.flatnonzero((cycles >= low) & (cycles <= high)) for _origin, low, high in segments]
        count = sum(len(rows) for rows in rows_for_view)
        if len(np.concatenate(rows_for_view)) != len(np.unique(np.concatenate(rows_for_view))):
            raise BatchIntegrityError(f"R03 view {name} overlaps duplicate rows")
        if count != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"R03 view {name} denominator differs: {count}")
    return control


def _read_budget_ledger() -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        return {"path": str(LEDGER_PATH), "readable": True, "reserved_total": 0,
                "card_reservation_present": False, "next_total": 2, "cap": MAX_SUCCESSFUL_FITS,
                "status": "PASS"}
    total = 0
    own = None
    try:
        for line_number, raw in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            event = json.loads(raw)
            if event.get("event") == "BUDGET_RESERVED":
                count = int(event["predictor_fits"])
                total += count
                if event.get("budget_key") == "R03:PILOT":
                    own = count
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"path": str(LEDGER_PATH), "readable": False, "error": str(exc),
                "status": "BLOCKED_LEDGER"}
    next_total = total if own == 2 else total + 2
    status = "PASS" if own in (None, 2) and next_total <= MAX_SUCCESSFUL_FITS else "BLOCKED_BUDGET"
    return {"path": str(LEDGER_PATH), "readable": True, "reserved_total": total,
            "card_reservation_present": own is not None, "existing_r03_reservation": own,
            "next_total": next_total, "cap": MAX_SUCCESSFUL_FITS, "status": status}


def _source_hashes() -> dict[str, str]:
    missing = [str(path) for path in SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R03 freeze source files missing: {missing}")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in SOURCES}


def _input_hashes() -> dict[str, str]:
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R03 frozen inputs missing: {missing}")
    hashes = {name: file_sha256(path) for name, path in sorted(INPUTS.items())}
    if hashes["official_regression_train"] != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train hash differs from R03 frozen input")
    return hashes


def _vectorizer_recipe() -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    result = {}
    for name, params in split_tfidf_parameters().items():
        vectorizer = TfidfVectorizer(**params)
        clean = {}
        for key, value in vectorizer.get_params(deep=False).items():
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, type):
                value = value.__name__
            clean[str(key)] = value
        result[name] = clean
    return result


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any]) -> dict[str, Any]:
    return _json_clean({
        "card_id": CARD_ID,
        "queue_recipe": dict(card),
        "feature_allowlist": list(FEATURE_NAMES),
        "feature_role": {"record_id": "identity_only", "threat_rank": "target_only"},
        "structured": {
            "source": "RegressionTextModel._structured_fit/_structured_transform",
            "numeric_parser": "Round2 _SelectedPreprocessor fit only on fold Train rows",
            "scaler": "sklearn StandardScaler default parameters",
            "categorical_columns": list(CATEGORICAL_FEATURES),
            "ohe": {"min_frequency": 30, "max_categories": 254,
                    "handle_unknown": "infrequent_if_exist", "sparse_output": True,
                    "dtype": "float64"},
            "missing_stream_indicator": True,
        },
        "tfidf_blocks": _vectorizer_recipe(),
        "sparse_regressor": {"class": "lightgbm.LGBMRegressor",
                             "parameters": dict(SPARSE_LGB_PARAMETERS),
                             "objective": "regression_l1", "early_stopping": False,
                             "sparse_csr_only": True},
        "prediction_policy": {
            "candidate": "clip(0.75 * frozen Ridge25 + 0.25 * rank50(new raw sparse), 0, 100)",
            "rank50": "clip(0.5 * raw + 0.5 * (100 * average_rank(raw)/n_cycle),0,100)",
            "ties": "average",
            "ranking_population": "all score rows for each cycle in complete 12-cycle horizon per cutoff",
            "ridge25_and_old_sparse_control": "row-exact frozen caches; old sparse raw receives rank50 once",
        },
        "evaluation": {
            "profile": "STANDARD",
            "views": {name: [list(x) for x in segments] for name, segments in views.items()},
            "cutoffs": [55, 67],
            "predictor_fits": 2,
            "fit_rows": EXPECTED_FIT_ROWS,
            "score_cycles": {str(k): list(v) for k, v in EXPECTED_SCORE_CYCLES.items()},
            "denominators": EXPECTED_VIEW_ROWS,
            "minimum_group_support": MIN_GROUP_SUPPORT,
        },
        "fixed_control_metrics_from_queue": EXPECTED_CONTROL_METRICS,
        "gate": EXPECTED_GATE,
        "resource_policy": {"threads": THREADS, "concurrent_jobs": 1,
                            "job_memory_cap_bytes": JOB_CAP_BYTES,
                            "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                            "no_densification": True},
    })


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card_and_queue(queue, card)
    train_path, frame, cycles = _read_official_train()
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R03 STANDARD views changed: {views}")
    input_hashes = _input_hashes()
    lineage = _validate_manifest_and_historical_freeze()
    controls = _load_control_arrays(frame, cycles, views)
    memory = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        memory[str(cutoff)] = estimate_sparse_memory(frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)])
    ledger = _read_budget_ledger()
    source_hashes = _source_hashes()
    recipe = _recipe_payload(card, views)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=INPUTS,
        source_files=SOURCES, evaluation_views=views, packages=PACKAGE_NAMES,
    )
    identity.update({
        "recipe": recipe,
        "control_lineage": lineage,
        "resource_policy": recipe["resource_policy"],
        "expected_fit_rows": EXPECTED_FIT_ROWS,
        "expected_score_rows": EXPECTED_VIEW_ROWS,
    })
    identity["recipe_artifact_sha256"] = json_sha256(recipe)
    signature = json_sha256(identity)
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise BatchIntegrityError("Existing R03 recipe differs from canonical preflight recipe")
    if FREEZE_PATH.exists():
        frozen = _read_json(FREEZE_PATH)
        if frozen.get("identity") != identity or frozen.get("signature") != signature:
            raise BatchIntegrityError("Existing R03 freeze differs from current source/input/recipe identity")
    memory_pass = all(value.get("status") == "PASS" for value in memory.values())
    preflight_status = "PREFLIGHT_PASS" if memory_pass and ledger.get("status") == "PASS" else "BLOCKED_RESOURCE_OR_BUDGET"
    public = {
        "card_id": CARD_ID,
        "status": preflight_status,
        "scope": "read-only official Train/cache preflight; no freeze, fit, candidate score, or metric computed",
        "train_path": str(train_path),
        "official_train_sha256": input_hashes["official_regression_train"],
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "freeze_identity_sha256": signature,
        "freeze_written": False,
        "recipe_payload": recipe,
        "control_lineage": lineage,
        "views": recipe["evaluation"]["views"],
        "fit_rows": EXPECTED_FIT_ROWS,
        "view_rows": EXPECTED_VIEW_ROWS,
        "control_row_id_hashes": {str(k): _hash_ordered_ids(v["score_record_ids"]) for k, v in controls.items()},
        "memory_estimate": memory,
        "budget_preflight": ledger,
        "training_lock": {"path": str(RUN_DIR.parent / ".training.lock"),
                          "acquired": False, "max_concurrent_jobs": 1},
        "candidate_prediction_or_metric_computed": False,
        "test_or_sample_access": False,
        "kaggle_access": False,
        "artifacts_written": False,
        "python_version": platform.python_version(),
    }
    return {"public": public, "identity": identity, "recipe": recipe,
            "frame": frame, "cycles": cycles, "views": views,
            "controls": controls, "train_path": train_path}


def _control_loader(context: Mapping[str, Any]):
    controls = context["controls"]

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        cutoff = int(cutoff)
        if cutoff not in controls:
            raise BatchIntegrityError(f"No frozen R03 control for cutoff {cutoff}")
        saved = controls[cutoff]
        if not np.array_equal(np.asarray(saved["score_rows"], dtype=np.int64), score_rows):
            raise BatchIntegrityError(f"R03 control rows differ from engine score rows at cutoff {cutoff}")
        return {
            "baseline_prediction": saved["baseline_prediction"],
            "ridge25_prediction": saved["ridge25_prediction"],
            "old_sparse_rank50": saved["old_sparse_rank50"],
            "control_rows": saved["score_rows"],
        }
    return load


def _independent_rank50(raw: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Verifier route intentionally does not import the model's ranking helper."""
    result = np.empty(len(raw), dtype=np.float64)
    for value in np.unique(cycle):
        mask = cycle == value
        n = int(mask.sum())
        result[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rankdata(raw[mask], method="average") / n
    return np.clip(result, 0.0, 100.0)


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    if y.ndim != 1 or pred.shape != y.shape or len(y) == 0:
        raise BatchIntegrityError("Verifier MAE arrays are empty or misaligned")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise BatchIntegrityError("Verifier MAE inputs contain nonfinite values")
    return float(np.mean(np.abs(y - pred), dtype=np.float64))


def _verify_pilot(context: Mapping[str, Any], report: Mapping[str, Any],
                  frame_with_target: pd.DataFrame) -> dict[str, Any]:
    """Independent post-freeze verifier. Not called by --preflight-only."""
    if report.get("card_id") != CARD_ID or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA":
        raise BatchIntegrityError("R03 pilot report identity/status is invalid")
    if report.get("fit_predictor_count") != 2 or report.get("cutoffs") != [55, 67]:
        raise BatchIntegrityError("R03 pilot must contain exactly two origins/fits")
    expected_signature = json_sha256(context["identity"])
    if report.get("freeze_signature") != expected_signature:
        raise BatchIntegrityError("R03 pilot report does not match current frozen identity")
    if "threat_rank" not in frame_with_target.columns:
        raise BatchIntegrityError("Post-fit verifier requires official Train target")
    y = pd.to_numeric(frame_with_target["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    cycles = pd.to_numeric(frame_with_target["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    record_ids = frame_with_target["record_id"].astype(str).to_numpy(dtype=str)
    per_cutoff = {}
    for cutoff in (55, 67):
        path = RUN_DIR / f"R03_fit{cutoff}.npz"
        identity = {"freeze_signature": expected_signature, "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": context["controls"][cutoff]["score_rows"].tolist()}
        arrays, sidecar = read_bundle(path, identity)
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        expected_rows = context["controls"][cutoff]["score_rows"]
        if not np.array_equal(rows, expected_rows) or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R03 fit bundle row identities differ at {cutoff}")
        if not np.array_equal(arrays["score_record_ids"], record_ids[rows]):
            raise BatchIntegrityError(f"R03 fit bundle record IDs differ at {cutoff}")
        raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
        ridge = np.asarray(arrays["control__ridge25_prediction"], dtype=np.float64)
        old_rank = np.asarray(arrays["control__old_sparse_rank50"], dtype=np.float64)
        baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        pred = np.asarray(arrays["prediction"], dtype=np.float64)
        cycle = cycles[rows]
        if any(value.shape != rows.shape for value in (raw, ridge, old_rank, baseline, pred)):
            raise BatchIntegrityError(f"R03 fit bundle arrays are misaligned at {cutoff}")
        metadata = sidecar["metadata"]
        if (int(metadata.get("fit_predictor_count", -1)) != 1
                or int(metadata.get("fit_rows", -1)) != EXPECTED_FIT_ROWS[cutoff]
                or int(metadata.get("score_rows", -1)) != len(rows)):
            raise BatchIntegrityError(f"R03 cutoff {cutoff} did not record one predictor fit")
        resource = metadata.get("fit_metadata", {}).get("resource_preflight", {})
        if resource.get("status") != "PASS" or resource.get("concurrent_training_jobs") != 1:
            raise BatchIntegrityError(f"R03 cutoff {cutoff} resource/slot receipt is missing")
        expected_baseline = np.clip(0.75 * ridge + 0.25 * old_rank, 0.0, 100.0)
        candidate_rank = _independent_rank50(raw, cycle)
        expected_prediction = np.clip(0.75 * ridge + 0.25 * candidate_rank, 0.0, 100.0)
        if (not np.allclose(baseline, expected_baseline, rtol=0.0, atol=1e-10)
                or not np.allclose(pred, expected_prediction, rtol=0.0, atol=1e-10)
                or not np.allclose(arrays["audit__candidate_rank50"], candidate_rank,
                                   rtol=0.0, atol=1e-10)):
            raise BatchIntegrityError(f"R03 route or rank50 reconstruction differs at cutoff {cutoff}")
        if (not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 100))
                or len(np.unique(record_ids[rows])) != len(rows)):
            raise BatchIntegrityError(f"R03 score predictions or IDs are invalid at {cutoff}")
        per_cutoff[cutoff] = {"rows": rows, "prediction": pred, "baseline": baseline}

    view_rows: dict[str, np.ndarray] = {}
    view_predictions: dict[str, np.ndarray] = {}
    view_baselines: dict[str, np.ndarray] = {}
    for name, segments in context["views"].items():
        row_parts, pred_parts, baseline_parts = [], [], []
        for cutoff, low, high in segments:
            selected = per_cutoff[cutoff]["rows"]
            mask = (cycles[selected] >= low) & (cycles[selected] <= high)
            row_parts.append(selected[mask])
            pred_parts.append(per_cutoff[cutoff]["prediction"][mask])
            baseline_parts.append(per_cutoff[cutoff]["baseline"][mask])
        rows = np.concatenate(row_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R03 verified view {name} row set/denominator differs")
        view_rows[name] = rows
        view_predictions[name] = np.concatenate(pred_parts)
        view_baselines[name] = np.concatenate(baseline_parts)

    view_metrics = {}
    for name, rows in view_rows.items():
        candidate_mae = _mae(y[rows], view_predictions[name])
        baseline_mae = _mae(y[rows], view_baselines[name])
        metric_row = {"rows": int(len(rows)), "candidate_mae": candidate_mae,
                      "baseline_mae": baseline_mae,
                      "gain_positive_is_better": baseline_mae - candidate_mae}
        reported = report.get("views", {}).get(name, {})
        if (int(reported.get("rows", -1)) != len(rows)
                or abs(float(reported.get("candidate", {}).get("mae", np.nan)) - candidate_mae) > 1e-10
                or abs(float(reported.get("baseline", {}).get("mae", np.nan)) - baseline_mae) > 1e-10):
            raise BatchIntegrityError(f"Generic and independent MAE reports differ for {name}")
        view_metrics[name] = metric_row

    group_sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        name: (rows, view_predictions[name], view_baselines[name])
        for name, rows in view_rows.items()
    }
    for name, cutoff, low, high in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
        selected = per_cutoff[cutoff]["rows"]
        mask = (cycles[selected] >= low) & (cycles[selected] <= high)
        group_sources[name] = (selected[mask], per_cutoff[cutoff]["prediction"][mask],
                               per_cutoff[cutoff]["baseline"][mask])

    group_metrics = {}
    for view, (rows, view_pred, view_base) in group_sources.items():
        missing_stream = frame_with_target.iloc[rows]["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
        masks = {
            "stream_missing": missing_stream,
            "target_le_5": y[rows] <= 5.0,
            "target_ge_95": y[rows] >= 95.0,
        }
        group_metrics[view] = {}
        for group, mask in masks.items():
            selected = np.flatnonzero(mask)
            if len(selected) < MIN_GROUP_SUPPORT:
                group_metrics[view][group] = {"n": int(len(selected)), "supported": False,
                                              "candidate_mae": None, "baseline_mae": None,
                                              "harm": None}
                continue
            candidate_mae = _mae(y[rows][selected], view_pred[selected])
            baseline_mae = _mae(y[rows][selected], view_base[selected])
            group_metrics[view][group] = {
                "n": int(len(selected)), "supported": True,
                "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                "harm": candidate_mae - baseline_mae,
            }

    gate = EXPECTED_GATE
    primary_gain = view_metrics["primary"]["gain_positive_is_better"]
    h12_harm = view_metrics["H12"]["candidate_mae"] - view_metrics["H12"]["baseline_mae"]
    individual_harms = {
        name: view_metrics[name]["candidate_mae"] - view_metrics[name]["baseline_mae"]
        for name in ("W1", "W4")
    }
    supported_harms = [
        item["harm"] for view in ("primary", "H12_E", "H12_L")
        for item in group_metrics[view].values() if item["supported"]
    ]
    max_group_harm = max(supported_harms) if supported_harms else None
    gate_results = {
        "primary_gain_at_least_0_03": primary_gain >= gate["pilot_primary_MAE_gain_min"],
        "H12_harm_at_most_0_05": h12_harm <= gate["pilot_H12_MAE_harm_max"],
        "each_pilot_window_harm_at_most_0_5": all(
            harm <= gate["individual_pilot_view_MAE_harm_max"] for harm in individual_harms.values()),
        "supported_tail_missing_harm_at_most_0_25": (
            max_group_harm is not None
            and max_group_harm <= gate["supported_tail_or_missing_MAE_harm_max"]),
    }
    decision = "ADVANCE" if all(gate_results.values()) else "RETIRE"
    return {
        "card_id": CARD_ID,
        "status": "INDEPENDENT_QA_PASS",
        "decision": decision,
        "freeze_signature": expected_signature,
        "fit_predictor_count": 2,
        "cutoffs": [55, 67],
        "views": view_metrics,
        "individual_window_harms": individual_harms,
        "groups": group_metrics,
        "maximum_supported_group_harm": max_group_harm,
        "gates": gate_results,
        "gate_definition": gate,
        "test_data_used": False,
    }


def _write_summary(verification: Mapping[str, Any]) -> None:
    summary = {
        "card_id": CARD_ID,
        "decision": verification["decision"],
        "fit_predictor_count": verification["fit_predictor_count"],
        "cutoffs": verification["cutoffs"],
        "primary_mae_gain": verification["views"]["primary"]["gain_positive_is_better"],
        "H12_mae_harm": (verification["views"]["H12"]["candidate_mae"]
                         - verification["views"]["H12"]["baseline_mae"]),
        "test_sample_kaggle_used": False,
        "details": ["results.json", "verification.json"],
    }
    write_json_exclusive(SUMMARY_PATH, summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, run_frozen_pilot: bool = False) -> dict[str, Any]:
    if preflight_only == run_frozen_pilot:
        raise BatchIntegrityError("Choose exactly one: --preflight-only or --run-frozen-pilot")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError("R03 execution is blocked by memory/budget/input preflight")
    # The freeze payload is written by the generic engine before the first fit.
    # Recipe persistence precedes freeze creation; its canonical content is also
    # inside the freeze identity, so any edit fails closed on resume.
    if RECIPE_PATH.exists():
        if _read_json(RECIPE_PATH) != context["recipe"]:
            raise BatchIntegrityError("Existing R03 recipe differs from current canonical recipe")
    else:
        write_json_exclusive(RECIPE_PATH, context["recipe"])
    _validate_recipe_artifact(
        RECIPE_PATH,
        context["recipe"],
        context["identity"]["recipe_artifact_sha256"],
    )
    identity = dict(context["identity"])
    ensure_frozen(FREEZE_PATH, identity)
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train changed after the R03 freeze was written")
    target_frame = pd.read_csv(context["train_path"], usecols=["threat_rank"])
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train changed while target values were being loaded")
    y = pd.to_numeric(target_frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if y.shape != (len(context["frame"]),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise BatchIntegrityError("Official Train target is invalid or row-misaligned")
    frame = context["frame"].copy()
    frame["threat_rank"] = y
    report = run_pilot(
        card=card,
        artifact_dir=RUN_DIR,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=MAX_SUCCESSFUL_FITS,
        frame=frame,
        target=y,
        feature_names=FEATURE_NAMES,
        adapter=r03_fit_predict,
        control_loader=_control_loader(context),
        freeze_identity=identity,
        freeze_path=FREEZE_PATH,
        event_path=LEDGER_PATH,
        evaluation_views=context["views"],
        threads=THREADS,
        report_path=RESULTS_PATH,
    )
    verified = _verify_pilot(context, report, frame)
    write_json_exclusive(VERIFICATION_PATH, verified)
    _write_summary(verified)
    return {"report": report, "verification": verified}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run-frozen-pilot", action="store_true",
                      help="Write the frozen recipe and execute exactly two pilot fits")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(_json_clean(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
