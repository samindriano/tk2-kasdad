"""R02 runner: wider word-vocabulary sparse Regression with frozen controls."""
from __future__ import annotations

import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_reg_diagnostics import (
    FULL_CACHE,
    FULL_REPORT,
    PILOT_CACHE,
    PRIMARY_CACHE,
    _make_view,
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
from src.models.optimization_v3_batch1_r02 import (
    FEATURE_NAMES,
    TFIDF_PARAMETERS,
    compose_r02_prediction,
    fit_predict,
)
from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import _text_documents
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/R02/freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R02"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R02"
BASELINE_DIAGNOSTICS = ROOT / "reports/optimization_v3/regression_diagnostics/diagnostics.json"
SUBGROUP_DIAGNOSTICS = ROOT / "reports/optimization_v2/regression_repr/fullsix_subgroup_diagnostics.json"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_CYCLES = np.arange(47, 80, dtype=np.int64)
MAX_JOB_GIB = 6.0
MIN_FREE_RAM_GIB = 4.0
RAM_RESERVE_GIB = 2.0
EXPECTED_CARD_CHANGE = (
    "Original sparse L1 hybrid model; only max_features20000"
    "\u00c3\u00a2\u00e2\u20ac\u00a0\u00e2\u20ac\u2122"
    "60000, retaining regression min_df2 and other parameters."
)


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_reg_diagnostics.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1_r02.py",
        ROOT / "src/models/optimization_v2_regression.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_research_tabular.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/utils/paths.py",
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
    }


def _validated_cache_hashes(queue: Mapping[str, Any], inputs: Mapping[str, Path]) -> dict[str, str]:
    baseline = queue["baseline"]["regression"]
    manifest = inputs["accepted_regression_manifest"]
    if file_sha256(manifest) != baseline["manifest_sha256"]:
        raise BatchIntegrityError("Accepted Regression manifest hash differs from the canonical queue")
    diagnostics = json.loads(BASELINE_DIAGNOSTICS.read_text(encoding="utf-8"))
    checks = diagnostics.get("cache_checks", {})
    if not all(checks.get(key) is True for key in (
        "primary_mae_matches_frozen_fullsix",
        "H12_pooled_mae_matches_frozen_fullsix",
        "all_Ridge25_cache_decompositions_exact",
        "frozen_subgroup_diagnostic_cache_hash_matches",
    )):
        raise BatchIntegrityError("Historical V3 Regression cache-only control audit is not PASS")
    recorded = diagnostics.get("source_and_cache_sha256", {})
    expected_paths = {
        "data/raw/regression/Train.csv": inputs["official_regression_train"],
        "experiments/runs/dual_reg_v1/stage1_paired_blends.npz": PRIMARY_CACHE,
        "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz": PILOT_CACHE,
        "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz": FULL_CACHE,
        "reports/optimization_v2/regression_repr/fullsix.json": FULL_REPORT,
        "reports/optimization_v2/regression_repr/fullsix_subgroup_diagnostics.json": SUBGROUP_DIAGNOSTICS,
    }
    observed: dict[str, str] = {}
    for relative, path in expected_paths.items():
        expected = recorded.get(relative)
        if not expected:
            raise BatchIntegrityError(f"Cache-only diagnostics omit expected SHA256 for {relative}")
        observed[relative] = file_sha256(path)
        if observed[relative] != expected:
            raise BatchIntegrityError(f"Frozen Regression cache/source hash mismatch: {relative}")
    baseline_primary = diagnostics["views"]["primary"]["overall"]
    baseline_h12 = diagnostics["views"]["H12_pooled"]["overall"]
    fullsix_primary = json.loads(FULL_REPORT.read_text(encoding="utf-8"))[
        "candidate_gates"]["lgbm_sparse_l1"]["primary_candidate"]
    if (not np.isclose(float(baseline_primary["mae"]), float(baseline["primary"]), atol=1e-10, rtol=0.0)
            or not np.isclose(float(fullsix_primary["mae"]), float(baseline["primary"]), atol=1e-10, rtol=0.0)
            or not np.isclose(float(baseline_h12["mae"]), float(baseline["H12"]), atol=1e-10, rtol=0.0)):
        raise BatchIntegrityError("Historical six-window/H12 report metrics differ from the canonical control")
    return observed


def _load_control_rows(
    frame: pd.DataFrame,
    views: Mapping[str, list[tuple[int, int, int]]],
) -> dict[int, dict[str, np.ndarray]]:
    base_cache = _load_archive(PRIMARY_CACHE)
    pilot_cache = _load_archive(PILOT_CACHE)
    full_cache = _load_archive(FULL_CACHE)
    sparse_cache = {**pilot_cache, **full_cache}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    output: dict[int, dict[str, np.ndarray]] = {}
    cutoffs = sorted({int(cutoff) for segments in views.values() for cutoff, _, _ in segments})
    for cutoff in cutoffs:
        cycles_for_cutoff = sorted({
            cycle
            for segments in views.values()
            for origin, low, high in segments if int(origin) == cutoff
            for cycle in range(int(low), int(high) + 1)
        })
        if not cycles_for_cutoff or cycles_for_cutoff != list(range(cutoff + 1, cutoff + 13)):
            raise BatchIntegrityError(f"R02 cutoff {cutoff} does not score one complete 12-cycle horizon")
        low, high = cycles_for_cutoff[0], cycles_for_cutoff[-1]
        block, audit = _make_view(frame, cutoff, low, high, sparse_cache, base_cache,
                                  view_name=f"R02 controls {low}-{high}")
        rows = np.asarray(block["row"], dtype=np.int64)
        expected_rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        if not np.array_equal(rows, expected_rows):
            raise BatchIntegrityError(f"R02 cached baseline rows are not exact for cutoff {cutoff}")

        sparse_rows = np.asarray(sparse_cache[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
        raw_all = np.asarray(sparse_cache[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
        if raw_all.shape != sparse_rows.shape:
            raise BatchIntegrityError(f"R02 sparse cache arrays are misaligned at cutoff {cutoff}")
        raw_mask = (cycles[sparse_rows] >= low) & (cycles[sparse_rows] <= high)
        if not np.array_equal(sparse_rows[raw_mask], rows):
            raise BatchIntegrityError(f"R02 sparse raw scores and Ridge25 rows differ at cutoff {cutoff}")
        baseline_sparse_raw = raw_all[raw_mask]
        ridge25 = 0.75 * np.asarray(block["lgb_svd32"], dtype=np.float64) + 0.25 * np.asarray(
            block["hybrid_ridge"], dtype=np.float64
        )
        saved_ridge25 = np.asarray(base_cache[f"blend_ridge25_{cutoff}"], dtype=np.float64)
        base_rows = np.asarray(base_cache[f"rows_{cutoff}"], dtype=np.int64)
        base_mask = (cycles[base_rows] >= low) & (cycles[base_rows] <= high)
        if not np.array_equal(base_rows[base_mask], rows):
            raise BatchIntegrityError(f"R02 Ridge25 rows differ from declared rows at cutoff {cutoff}")
        if not np.allclose(ridge25, saved_ridge25[base_mask], atol=1e-10, rtol=0.0):
            raise BatchIntegrityError(f"R02 cached Ridge25 decomposition differs at cutoff {cutoff}")

        cycle_for_rows = cycles[rows]
        baseline_prediction = np.asarray(block["prediction"], dtype=np.float64)
        baseline_reconstructed = compose_r02_prediction(ridge25, baseline_sparse_raw, cycle_for_rows)
        if not np.allclose(baseline_prediction, baseline_reconstructed, atol=1e-10, rtol=0.0):
            raise BatchIntegrityError(f"R02 frozen three-component champion does not reconstruct at cutoff {cutoff}")
        output[cutoff] = {
            "rows": rows,
            "baseline_prediction": baseline_prediction,
            "ridge25_prediction": ridge25,
            "baseline_sparse_raw": baseline_sparse_raw,
            "cycle_code": cycle_for_rows,
            "baseline_sparse_rank50": np.asarray(block["sparse_lgb_rank50"], dtype=np.float64),
            "ridge25_reconstruction_max_abs_error": np.asarray([audit["base_ridge25_reconstruction_max_abs_error"]]),
        }
    return output


def _preflight(
    frame: pd.DataFrame,
    train_path: Path,
    card: Mapping[str, Any],
    queue: Mapping[str, Any],
    views: Mapping[str, list[tuple[int, int, int]]],
    input_files: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    baseline = queue["baseline"]["regression"]
    train_hash = file_sha256(train_path)
    if train_hash != baseline["Train_sha256"]:
        raise BatchIntegrityError("Official Regression Train hash differs from the frozen queue")
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise BatchIntegrityError(f"Expected {EXPECTED_TRAIN_ROWS} official Regression Train rows")
    absent = sorted(set(FEATURE_NAMES) - set(frame.columns))
    if absent:
        raise BatchIntegrityError(f"Official Regression Train is missing R02 features: {absent}")
    forbidden = set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}
    if forbidden:
        raise BatchIntegrityError(f"R02 feature allowlist includes protected fields: {sorted(forbidden)}")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise BatchIntegrityError("Official Regression Train has missing/blank record_id values")
    if frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Regression Train record_id values are not unique")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), EXPECTED_CYCLES):
        raise BatchIntegrityError("Official Regression Train cycles differ from 47-79")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise BatchIntegrityError("Official Regression threat_rank is invalid")

    cache_hashes = _validated_cache_hashes(queue, input_files)
    controls = _load_control_rows(frame, views)
    row_predictions = {int(row): float(pred) for values in controls.values()
                       for row, pred in zip(values["rows"], values["baseline_prediction"])}
    if len(row_predictions) != sum(len(v["rows"]) for v in controls.values()):
        raise BatchIntegrityError("R02 cache control rows overlap across fit origins")

    def baseline_mae(segments: list[tuple[int, int, int]]) -> tuple[float, int]:
        row_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            origin_rows = controls[int(cutoff)]["rows"]
            origin_cycles = cycles[origin_rows]
            selected = (origin_cycles >= int(low)) & (origin_cycles <= int(high))
            row_parts.append(origin_rows[selected])
        rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
        if len(rows) == 0 or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError("R02 baseline view has empty or duplicate rows")
        predictions = np.fromiter((row_predictions[int(row)] for row in rows), dtype=np.float64, count=len(rows))
        return float(np.mean(np.abs(predictions - target[rows]))), int(len(rows))

    # The pilot primary is W1+W4, while the queue's frozen `primary` is the
    # complete six-window aggregate. Record the former but compare only views
    # with identical row definitions; do not conflate their denominators.
    primary_mae, primary_n = baseline_mae(views["primary"])
    h12_mae, h12_n = baseline_mae(views["H12"])
    if not np.isclose(h12_mae, float(baseline["H12"]), atol=1e-10, rtol=0.0):
        raise BatchIntegrityError(f"R02 reconstructed H12 control MAE differs: {h12_mae}")

    # Bound a cutoff-67 sparse fit using the inherited sklearn analyzer. The
    # token count is an upper bound on CSR nonzeros; include copies/LightGBM
    # workspace and enforce the batch RAM ceiling before any fit.
    fit_mask = cycles <= max(cutoff for segments in views.values() for cutoff, _, _ in segments)
    fit_docs, _ = _text_documents(frame.loc[fit_mask, "signal_stream"])
    analyzer = TfidfVectorizer(**TFIDF_PARAMETERS).build_analyzer()
    upper_nnz = int(sum(len(analyzer(document)) for document in fit_docs))
    fit_rows = int(fit_mask.sum())
    upper_csr_bytes = upper_nnz * 16 + (fit_rows + 1) * 8
    estimated_peak_gib = float(1.5 + 3.0 * upper_csr_bytes / (1024 ** 3))
    free_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    allowed_gib = min(MAX_JOB_GIB, free_gib - RAM_RESERVE_GIB)
    if free_gib < MIN_FREE_RAM_GIB or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"R02 resource preflight blocked: free={free_gib:.2f}GiB, "
            f"estimated_peak={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB"
        )

    manifest = json.loads(input_files["accepted_regression_manifest"].read_text(encoding="utf-8"))
    if manifest.get("task") != "regression" or manifest.get("status") != "PREPARED_AND_QA_PASS":
        raise BatchIntegrityError("Accepted regression manifest is not a QA-passed Regression artifact")
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=input_files,
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity["historical_verified_cache_sha256"] = cache_hashes
    identity["accepted_baseline_manifest_sha256"] = baseline["manifest_sha256"]
    identity["python_version"] = platform.python_version()
    identity["r02_recipe_delta"] = {
        "base_tfidf_max_features": 20_000,
        "candidate_tfidf_max_features": 60_000,
        "all_other_tfidf_parameters_unchanged": True,
        "sparse_lgb_parameters": dict(SPARSE_LGB_PARAMETERS),
        "policy": "clip(0.75*Ridge25 + 0.25*rank50(new raw sparse prediction),0,100)",
        "rank_ties": "average",
    }
    resource = {
        "cutoff": int(max(cutoff for segments in views.values() for cutoff, _, _ in segments)),
        "fit_rows": fit_rows,
        "max_features": int(TFIDF_PARAMETERS["max_features"]),
        "upper_estimated_ngram_nnz": upper_nnz,
        "upper_csr_gib": float(upper_csr_bytes / (1024 ** 3)),
        "estimated_peak_job_gib": estimated_peak_gib,
        "available_ram_gib_observed": free_gib,
        "allowed_job_ram_gib": allowed_gib,
        "maximum_job_gib": MAX_JOB_GIB,
        "minimum_free_ram_gib": MIN_FREE_RAM_GIB,
        "reserved_free_ram_gib": RAM_RESERVE_GIB,
        "single_training_job": True,
        "threads": 2,
        "dense_full_word_matrix": False,
    }
    return {
        "status": "PREFLIGHT_PASS",
        "train_sha256": train_hash,
        "train_rows": int(len(frame)),
        "primary_control_mae": primary_mae,
        "primary_control_rows": primary_n,
        "primary_control_view": "pilot W1+W4 only; canonical queue primary is full-six and not directly comparable",
        "H12_control_mae": h12_mae,
        "H12_control_rows": h12_n,
        "resource": resource,
        "control_cache_sha256": cache_hashes,
        "freeze_identity_sha256": json_sha256(identity),
        "test_data_read": False,
        "model_fit_performed": False,
        "freeze_written": False,
    }, controls, identity


def _control_loader(controls: Mapping[int, Mapping[str, np.ndarray]]):
    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        if int(cutoff) not in controls:
            raise BatchIntegrityError(f"No R02 frozen controls for cutoff {cutoff}")
        record = controls[int(cutoff)]
        if not np.array_equal(record["rows"], np.asarray(score_rows, dtype=np.int64)):
            raise BatchIntegrityError(f"R02 frozen control row positions differ at cutoff {cutoff}")
        return {key: value for key, value in record.items() if key not in {
            "rows", "baseline_sparse_rank50", "ridge25_reconstruction_max_abs_error"
        }}
    return load


def reconstruct_r02(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independently reconstruct candidate and frozen three-component control."""
    raw_candidate = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    raw_base_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    ridge25 = np.asarray(arrays["control__ridge25_prediction"], dtype=np.float64)
    cycles = np.asarray(arrays["control__cycle_code"], dtype=np.int64)
    if len({len(raw_candidate), len(raw_base_sparse), len(ridge25), len(cycles)}) != 1:
        raise BatchIntegrityError("R02 verifier inputs have inconsistent row counts")

    def independent_map(raw: np.ndarray) -> np.ndarray:
        mapped = np.empty(raw.shape, dtype=np.float64)
        for cycle in np.unique(cycles):
            mask = cycles == cycle
            mapped[mask] = 0.5 * raw[mask] + 50.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        return np.clip(mapped, 0.0, 100.0)

    candidate = np.clip(0.75 * ridge25 + 0.25 * independent_map(raw_candidate), 0.0, 100.0)
    baseline = np.clip(0.75 * ridge25 + 0.25 * independent_map(raw_base_sparse), 0.0, 100.0)
    return candidate, baseline


def _subgroup_diagnostics(
    report: Mapping[str, Any],
    frame: pd.DataFrame,
    target: np.ndarray,
    evaluation_views: Mapping[str, list[tuple[int, int, int]]],
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    support_declaration = str(queue["evaluation"].get("support_groups", "")).casefold()
    required_declarations = ("entry_friction0/1", "stream present/missing", "target tails for regression")
    if any(value not in support_declaration for value in required_declarations):
        raise BatchIntegrityError("R02 queue does not declare the subgroup diagnostics required by this adapter")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    records_by_cutoff: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in report["cutoffs"]:
        path = ARTIFACT_DIR / f"R02_fit{int(cutoff)}.npz"
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        arrays, _ = read_bundle(path, sidecar["identity"])
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        records_by_cutoff[int(cutoff)] = {
            "rows": rows,
            "prediction": np.asarray(arrays["prediction"], dtype=np.float64),
            "baseline": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
        }

    output: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for view_name in ("primary", "H12"):
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in evaluation_views[view_name]:
            stored = records_by_cutoff[int(cutoff)]
            mask = (cycles[stored["rows"]] >= int(low)) & (cycles[stored["rows"]] <= int(high))
            row_parts.append(stored["rows"][mask])
            pred_parts.append(stored["prediction"][mask])
            base_parts.append(stored["baseline"][mask])
        rows = np.concatenate(row_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R02 subgroup QA found duplicate rows in {view_name}")
        y = target[rows]
        candidate = np.concatenate(pred_parts)
        baseline = np.concatenate(base_parts)
        local = frame.iloc[rows]
        missing = local["signal_stream"].isna() | local["signal_stream"].astype(str).str.strip().eq("")
        entry_friction = pd.to_numeric(local["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        groups: dict[str, np.ndarray] = {
            "entry_friction_0": entry_friction == 0.0,
            "entry_friction_1": entry_friction == 1.0,
            "stream_present": ~missing.to_numpy(dtype=bool),
            "stream_missing": missing.to_numpy(dtype=bool),
            "target_le_5": y <= 5.0,
            "target_ge_95": y >= 95.0,
        }
        view_records: dict[str, Any] = {}
        for name, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            n = int(mask.sum())
            supported = n >= 100
            harm = float(np.mean(np.abs(candidate[mask] - y[mask])) -
                         np.mean(np.abs(baseline[mask] - y[mask]))) if n else None
            view_records[name] = {
                "n": n,
                "supported": supported,
                "evidence_status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                "mae_harm_candidate_minus_control": harm,
                "numeric_guard_applies": name in {"stream_missing", "target_le_5", "target_ge_95"},
            }
            if (name in {"stream_missing", "target_le_5", "target_ge_95"}
                    and supported and harm is not None and harm > 0.25):
                failures.append({"view": view_name, "group": name, "n": n, "mae_harm": harm})
        output[view_name] = view_records
    return {
        "minimum_support": 100,
        "maximum_supported_tail_or_missing_MAE_harm": 0.25,
        "queue_declared_groups": ["entry_friction_0", "entry_friction_1", "stream_present", "stream_missing", "target_le_5", "target_ge_95"],
        "numeric_guard_groups": ["stream_missing", "target_le_5", "target_ge_95"],
        "views": output,
        "supported_guard_failures": failures,
    }


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    if card.get("id") != "R02" or card.get("task") != "regression":
        raise BatchIntegrityError("R02 runner received a different queue card")
    card_assertions = {
        "family": "vocab_width",
        # Compare the exact serialized recipe wording in the frozen queue.
        "change": EXPECTED_CARD_CHANGE,
        "policy": "R_SPARSE_REPLACE",
        "evaluation": "STANDARD",
        "pilot_predictor_fits": 2,
        "condition": "No additional dependency",
    }
    mismatches = {key: (card.get(key), expected) for key, expected in card_assertions.items()
                  if card.get(key) != expected}
    if mismatches:
        raise BatchIntegrityError(f"R02 canonical-card assertions failed: {mismatches}")
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R02 is valid only while the V3 campaign remains unfinished")
    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    frame = pd.read_csv(train_path)
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    views = evaluation_views_for_card(queue, "R02")
    inputs = _input_files(train_path, card, queue)
    preflight, controls, identity = _preflight(frame, train_path, card, queue, views, inputs)
    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT",
            "card_id": "R02",
            "preflight": preflight,
            "evaluation_views": views,
            "source_files": len(_source_files()),
            "control_files": len(inputs),
            "official_test_read": False,
            "model_fit_performed": False,
            "freeze_written": False,
        }

    append_event(REPORT_DIR / "attempts.jsonl", {"event": "PREFLIGHT_PASS", **preflight})
    report = run_pilot(
        frame=frame,
        target=target,
        card=card,
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=_control_loader(controls),
        freeze_identity=identity,
        freeze_path=FREEZE_PATH,
        artifact_dir=ARTIFACT_DIR,
        event_path=REPORT_DIR / "attempts.jsonl",
        evaluation_views=views,
        threads=2,
        report_path=REPORT_DIR / "pilot.json",
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
        prediction_reconstructors={"R02": reconstruct_r02},
    )
    subgroup = _subgroup_diagnostics(report, frame, target, views, queue)
    gate_config = queue["evaluation"]["regression_gate"]
    gain = report["views"]
    checks = {
        "primary_mae_gain_at_least_0_03": gain["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_MAE_gain_min"]),
        "H12_harm_at_most_0_05": gain["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_MAE_harm_max"]),
        "W1_harm_at_most_0_5": gain["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "W4_harm_at_most_0_5": gain["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_group_harms": len(subgroup["supported_guard_failures"]) == 0,
    }
    gate = {
        "card_id": "R02",
        "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": gate_config,
        "subgroup_diagnostics": subgroup,
        "decision_basis": "Fixed temporal Train pilot gates only; no Test or Kaggle result used.",
    }
    for path, value, label in (
        (REPORT_DIR / "verification.json", verification, "verification"),
        (REPORT_DIR / "subgroups.json", subgroup, "subgroup diagnostics"),
        (REPORT_DIR / "gate.json", gate, "pilot gate"),
    ):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != value:
                raise BatchIntegrityError(f"Existing R02 {label} differs from recomputed evidence")
        else:
            write_json_exclusive(path, value)
    return {"pilot": report, "verification": verification, "subgroups": subgroup, "gate": gate, "preflight": preflight}


__all__ = ["reconstruct_r02", "run_card"]
