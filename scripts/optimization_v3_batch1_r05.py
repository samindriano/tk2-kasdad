"""R05 frozen binary word-presence pilot with verified V3 Regression controls."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import CountVectorizer

from scripts import optimization_v3_batch1_r02 as r02
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    make_freeze_identity,
    package_versions,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r05 import (
    COUNT_PARAMETERS,
    FEATURE_NAMES,
    compose_r05_prediction,
    fit_predict,
)
from src.models.regression_research_text import _text_documents
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R05.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R05"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R05"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R05"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUBGROUP_PATH = REPORT_DIR / "subgroups.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
ATTEMPT_PATH = REPORT_DIR / "attempts.jsonl"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"

EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
EXPECTED_QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
EXPECTED_CARD_SHA256 = "91f1a91495d3212100f3f5f1a90746bb8deda0245615eb43b71594bf19c458b0"
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_CYCLES = np.arange(47, 80, dtype=np.int64)
MAX_JOB_GIB = 6.0
MIN_FREE_RAM_GIB = 4.0
RAM_RESERVE_GIB = 2.0
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
EXPECTED_CARD = {
    "id": "R05",
    "task": "regression",
    "name": "Binary word presence",
    "family": "count_presence",
    "priority": "P2",
    "change": "CountVectorizer word(1,2)/min_df2/cap20000/binaryTrue; no IDF/norm. Original structured sparse L1 model.",
    "policy": "R_SPARSE_REPLACE",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "implementation": "NOT_IMPLEMENTED",
    "card_path": "reports/optimization_v3/research_backlog/cards/R05.md",
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}

RECIPE = {
    "card_id": "R05",
    "model": "LightGBM L1 sparse hybrid regression",
    "structured_features": list(FEATURE_NAMES),
    "text_vectorizer": {
        **{key: list(value) if isinstance(value, tuple) else value for key, value in COUNT_PARAMETERS.items()},
        "lowercase": True,
        "token_pattern": r"(?u)\b\w\w+\b",
        "strip_accents": None,
        "dtype": "int64",
        "use_idf": False,
        "row_normalization": None,
    },
    "structured_preprocessing": "RegressionTextModel fit-only numeric imputation/scaling + OneHotEncoder(min_frequency=30,max_categories=254,handle_unknown=infrequent_if_exist)",
    "missing_stream_indicator": True,
    "sparse_lgb_parameters": dict(r02.SPARSE_LGB_PARAMETERS),
    "policy": "clip(0.75*Ridge25 + 0.25*rank50(new raw R05 sparse prediction),0,100)",
    "rank50": "within each complete score cycle; average ties; 0.5*raw+0.5*100*rank/n; final clipped [0,100]",
    "views": EXPECTED_VIEWS,
    "predictor_fits": 2,
    "preprocessing_fit_calls_per_cutoff": {"count_vectorizer": 1, "structured_processor_scaler_ohe": 1},
    "fit_rows": {"55": 12020, "67": 29705},
    "score_rows": {"55": 17685, "67": 20903},
    "test_data_used": False,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _write_or_match(path: Path, payload: Any) -> None:
    if path.exists():
        if _json(path) != payload:
            raise BatchIntegrityError(f"Existing R05 output differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _source_files() -> list[Path]:
    return sorted({
        *r02._source_files(),
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r05.py",
        ROOT / "tests/test_optimization_v3_batch1_r05.py",
    }, key=str)


def _input_files(train_path: Path, card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Path]:
    files = r02._input_files(train_path, card, queue)
    files["canonical_queue"] = QUEUE_PATH
    files["canonical_card"] = CARD_PATH
    return files


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray, card: Mapping[str, Any]) -> dict[str, Any]:
    analyzer = CountVectorizer(**COUNT_PARAMETERS).build_analyzer()
    support: dict[str, Any] = {}
    maximum_occurrences = 0
    maximum_csr_bytes = 0
    for cutoff in (55, 67):
        fit_mask = cycles <= cutoff
        docs, missing = _text_documents(frame.loc[fit_mask, "signal_stream"])
        document_frequency: Counter[str] = Counter()
        total_occurrences = 0
        documents_with_ngrams = 0
        for document in docs:
            terms = analyzer(document)
            total_occurrences += len(terms)
            if terms:
                documents_with_ngrams += 1
                document_frequency.update(set(terms))
        eligible = sum(count >= int(COUNT_PARAMETERS["min_df"]) for count in document_frequency.values())
        # CountVectorizer stores int64 values; CSR assembly converts them to float64.
        # Bound both representations plus sparse indices/pointers and model workspace.
        fit_rows = int(fit_mask.sum())
        csr_bytes = total_occurrences * 24 + (fit_rows + 1) * 16
        support[str(cutoff)] = {
            "fit_rows": fit_rows,
            "fit_missing_stream_rows": int(missing.sum()),
            "nonempty_documents_with_word_ngrams": documents_with_ngrams,
            "distinct_ngrams_document_frequency_ge_min_df": int(eligible),
            "expected_vocabulary_width_after_cap": int(min(eligible, int(COUNT_PARAMETERS["max_features"]))),
            "analyzed_ngram_occurrences_upper_bound": int(total_occurrences),
            "sparse_matrix_upper_bytes": int(csr_bytes),
        }
        maximum_occurrences = max(maximum_occurrences, total_occurrences)
        maximum_csr_bytes = max(maximum_csr_bytes, csr_bytes)
    estimated_peak_gib = float(1.5 + 3.0 * maximum_csr_bytes / (1024 ** 3))
    free_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    allowed_gib = min(MAX_JOB_GIB, free_gib - RAM_RESERVE_GIB)
    if free_gib < MIN_FREE_RAM_GIB or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"R05 resource preflight blocked: free={free_gib:.2f}GiB, "
            f"estimated_peak={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB"
        )
    return {
        "status": "PASS",
        "cutoffs": support,
        "max_features": int(COUNT_PARAMETERS["max_features"]),
        "min_df": int(COUNT_PARAMETERS["min_df"]),
        "raw_analyzed_ngram_occurrences_upper_bound": maximum_occurrences,
        "estimated_csr_upper_gib": float(maximum_csr_bytes / (1024 ** 3)),
        "estimated_peak_job_gib": estimated_peak_gib,
        "available_ram_gib_observed": free_gib,
        "allowed_job_ram_gib": allowed_gib,
        "maximum_job_gib": MAX_JOB_GIB,
        "minimum_free_ram_gib": MIN_FREE_RAM_GIB,
        "reserved_free_ram_gib": RAM_RESERVE_GIB,
        "single_training_job": True,
        "threads": 2,
        "dense_full_text_matrix": False,
        "estimator_fits_reserved": int(card["pilot_predictor_fits"]),
    }


def _stable_resource_identity(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude live free-RAM observations from the immutable model identity."""
    return {key: value for key, value in resource.items() if key not in {
        "available_ram_gib_observed", "allowed_job_ram_gib"
    }}


def _append_event_once(path: Path, event: Mapping[str, Any]) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            current = json.loads(line)
            if all(current.get(key) == value for key, value in event.items()):
                return
    append_event(path, event)


def _subgroup_diagnostics(
    report: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
    evaluation_views: Mapping[str, list[tuple[int, int, int]]],
) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    bundles: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in report["cutoffs"]:
        path = RUN_DIR / f"R05_fit{int(cutoff)}.npz"
        sidecar = _json(path.with_suffix(".json"))
        from src.models.optimization_v3_batch1 import read_bundle
        arrays, _ = read_bundle(path, sidecar["identity"])
        bundles[int(cutoff)] = {
            "rows": np.asarray(arrays["score_rows"], dtype=np.int64),
            "candidate": np.asarray(arrays["prediction"], dtype=np.float64),
            "baseline": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
        }
    output: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for view in ("primary", "H12"):
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in evaluation_views[view]:
            block = bundles[int(cutoff)]
            mask = (cycles[block["rows"]] >= int(low)) & (cycles[block["rows"]] <= int(high))
            row_parts.append(block["rows"][mask])
            pred_parts.append(block["candidate"][mask])
            base_parts.append(block["baseline"][mask])
        rows = np.concatenate(row_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R05 subgroup QA found duplicate rows in {view}")
        y = target[rows]
        prediction = np.concatenate(pred_parts)
        baseline = np.concatenate(base_parts)
        local = frame.iloc[rows]
        stream = local["signal_stream"]
        missing = stream.isna().to_numpy(dtype=bool) | stream.fillna("").astype(str).str.strip().eq("").to_numpy(dtype=bool)
        friction = pd.to_numeric(local["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        groups = {
            "entry_friction_0": friction == 0.0,
            "entry_friction_1": friction == 1.0,
            "stream_present": ~missing,
            "stream_missing": missing,
            "target_le_5": y <= 5.0,
            "target_ge_95": y >= 95.0,
        }
        view_result: dict[str, Any] = {}
        for name, mask in groups.items():
            n = int(mask.sum())
            supported = n >= 100
            harm = float(np.mean(np.abs(prediction[mask] - y[mask])) -
                         np.mean(np.abs(baseline[mask] - y[mask]))) if n else None
            guard_applies = name in {"stream_missing", "target_le_5", "target_ge_95"} and supported
            record = {
                "n": n,
                "supported": supported,
                "evidence_status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                "mae_harm_candidate_minus_control": harm,
                "numeric_guard_applies": guard_applies,
                "guard_pass": None if not guard_applies else bool(harm <= 0.25),
            }
            view_result[name] = record
            if guard_applies and harm > 0.25:
                failures.append({"view": view, "group": name, **record})
        output[view] = view_result
    return {
        "minimum_support": 100,
        "maximum_supported_tail_or_missing_MAE_harm": 0.25,
        "views": output,
        "supported_guard_failures": failures,
    }


def reconstruct_r05(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent policy reconstruction from saved raw candidate/control components."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    base_raw = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    ridge = np.asarray(arrays["control__ridge25_prediction"], dtype=np.float64)
    cycles = np.asarray(arrays["control__cycle_code"], dtype=np.int64)
    if len({len(raw), len(base_raw), len(ridge), len(cycles)}) != 1:
        raise BatchIntegrityError("R05 reconstruction arrays have inconsistent lengths")

    def independent_rank50(values: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=np.float64)
        for cycle in np.unique(cycles):
            mask = cycles == cycle
            # Rank calculation is deliberately expressed independently of the model adapter.
            group = values[mask]
            order = np.argsort(group, kind="mergesort")
            ranks = np.empty(len(group), dtype=np.float64)
            start = 0
            while start < len(order):
                stop = start + 1
                while stop < len(order) and group[order[stop]] == group[order[start]]:
                    stop += 1
                ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
                start = stop
            percentile = 100.0 * ranks / len(group)
            result[mask] = 0.5 * group + 0.5 * percentile
        return np.clip(result, 0.0, 100.0)

    candidate = np.clip(0.75 * ridge + 0.25 * independent_rank50(raw), 0.0, 100.0)
    baseline = np.clip(0.75 * ridge + 0.25 * independent_rank50(base_raw), 0.0, 100.0)
    return candidate, baseline


def _read_attempt_counts() -> dict[str, Any]:
    events = [json.loads(line) for line in ATTEMPT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    own = [event for event in events if event.get("card_id") == "R05"]
    started = [event for event in own if event.get("event") == "FIT_STARTED"]
    saved = [event for event in own if event.get("event") == "FIT_SAVED"]
    failed = [event for event in own if event.get("event") == "FIT_FAILED"]
    seconds = [float(event["fit_seconds"]) for event in saved]
    vectorizer_calls = 0
    for cutoff in (55, 67):
        sidecar_path = RUN_DIR / f"R05_fit{cutoff}.json"
        if sidecar_path.is_file():
            vectorizer_calls += int(_json(sidecar_path).get("metadata", {}).get("fit_metadata", {}).get("vectorizer_fit_calls", 0))
    return {
        "fit_attempts_started": len(started),
        "successful_predictor_fits": sum(int(event.get("fit_predictor_count", 0)) for event in saved),
        "fit_failed_events": len(failed),
        "fit_saved_events": len(saved),
        "saved_fit_seconds": seconds,
        "total_saved_fit_seconds": float(sum(seconds)),
        "vectorizer_fit_calls": vectorizer_calls,
        "source": str(ATTEMPT_PATH.relative_to(ROOT)),
    }


def _render_summary(results: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    views = results["pilot"]["views"]
    lines = [
        "# R05 — Binary word presence",
        "",
        f"**Verdict: {gate['status']}** — fixed temporal Train pilot, no Test/Kaggle access.",
        "",
        "Recipe: word unigram/bigram CountVectorizer, min_df=2, cap=20,000, binary presence, no IDF/normalization; structured/OHE + missing indicator; sparse LightGBM regression_l1.",
        "",
        "| View | n | Control MAE | R05 MAE | Gain (control − R05) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("W1", "W4", "primary", "H12"):
        item = views[name]
        lines.append(f"| {name} | {item['rows']:,} | {item['baseline']['mae']:.6f} | {item['candidate']['mae']:.6f} | {item['gain_positive_is_better']:+.6f} |")
    lines.extend([
        "",
        "The comparison uses the frozen three-component Regression control on exactly the same rows. H12 includes both early and late horizons; it is not substituted for the primary W1+W4 pilot. Tokenizer and all structured transforms were fit separately using rows through each cutoff.",
        "",
        "See `results.json`, `verification.json`, `subgroups.json`, and `gate.json` for denominators, support, fit accounting, cache reconstruction, and diagnostics.",
        "",
    ])
    return "\n".join(lines)


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    mismatches = {key: (card.get(key), value) for key, value in EXPECTED_CARD.items() if card.get(key) != value}
    if mismatches:
        raise BatchIntegrityError(f"R05 canonical card assertions failed: {mismatches}")
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R05 runs only while the V3 campaign remains unfinished")
    queue_hash = file_sha256(QUEUE_PATH)
    if queue_hash != EXPECTED_QUEUE_SHA256:
        raise BatchIntegrityError(f"R05 canonical queue hash differs: {queue_hash}")
    card_hash = file_sha256(CARD_PATH)
    if EXPECTED_CARD_SHA256 and card_hash != EXPECTED_CARD_SHA256:
        raise BatchIntegrityError(f"R05 canonical card hash differs: {card_hash}")

    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    if file_sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from frozen R05 queue")
    frame = pd.read_csv(train_path)
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise BatchIntegrityError(f"R05 expected {EXPECTED_TRAIN_ROWS} official Train rows")
    missing_columns = sorted((set(FEATURE_NAMES) | {"record_id", "cycle_code", "threat_rank"}) - set(frame.columns))
    if missing_columns:
        raise BatchIntegrityError(f"R05 Train is missing required fields: {missing_columns}")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R05 feature allowlist contains a protected field")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any() or not frame["record_id"].astype(str).is_unique:
        raise BatchIntegrityError("R05 Train record_id values are blank or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), EXPECTED_CYCLES):
        raise BatchIntegrityError("R05 Train cycles differ from 47–79")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise BatchIntegrityError("R05 threat_rank is invalid")

    views = evaluation_views_for_card(queue, "R05")
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R05 temporal evaluation views changed: {views}")
    inputs = _input_files(train_path, card, queue)
    preflight_r02, controls, control_identity = r02._preflight(frame, train_path, card, queue, views, inputs)
    resource = _resource_preflight(frame, cycles, card)
    if preflight_r02.get("status") != "PREFLIGHT_PASS":
        raise BatchIntegrityError("Inherited frozen Regression control audit failed")
    # Bind the count recipe itself to the pre-score identity and the manifest/cache audit from R02.
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=inputs,
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity["candidate_recipe"] = RECIPE
    identity["candidate_recipe_sha256"] = json_sha256(RECIPE)
    identity["recipe_record_sha256"] = json_sha256({
        "recipe": RECIPE, "queue_sha256": queue_hash, "card_sha256": card_hash
    })
    identity["control_preflight_identity_sha256"] = json_sha256(control_identity)
    identity["control_cache_hashes"] = preflight_r02["control_cache_sha256"]
    identity["resource_preflight"] = _stable_resource_identity(resource)
    identity["source_commit"] = _source_commit()
    summary_preflight = {
        "status": "PREFLIGHT_PASS",
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "train_rows": len(frame),
        "fit_rows": {"55": int(np.sum(cycles <= 55)), "67": int(np.sum(cycles <= 67))},
        "score_rows": {"55": int(np.sum((cycles >= 56) & (cycles <= 67))), "67": int(np.sum((cycles >= 68) & (cycles <= 79)))},
        "view_rows": {name: int(sum(np.sum((cycles >= low) & (cycles <= high)) for _, low, high in segments)) for name, segments in views.items()},
        "pilot_control_primary_mae": preflight_r02["primary_control_mae"],
        "pilot_control_primary_rows": preflight_r02["primary_control_rows"],
        "H12_control_mae": preflight_r02["H12_control_mae"],
        "H12_control_rows": preflight_r02["H12_control_rows"],
        "resource": resource,
        "cache_sha256": preflight_r02["control_cache_sha256"],
        "freeze_identity_sha256": json_sha256(identity),
        "official_test_read": False,
        "model_fit_performed": False,
    }
    if preflight_only:
        return {"status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT", "card_id": "R05", "preflight": summary_preflight,
                "evaluation_views": views, "source_files": len(_source_files()), "official_test_read": False,
                "model_fit_performed": False, "freeze_written": False}

    # Materialize all owned output directories and the exact recipe before freeze/fit.
    for directory in (EXPERIMENT_DIR, RUN_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    recipe_record = {"recipe": RECIPE, "queue_sha256": queue_hash, "card_sha256": card_hash,
                     "source_commit": _source_commit()}
    _write_or_match(RECIPE_PATH, recipe_record)
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=inputs,
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity["candidate_recipe"] = RECIPE
    identity["candidate_recipe_sha256"] = json_sha256(RECIPE)
    identity["recipe_record_sha256"] = json_sha256(recipe_record)
    identity["control_preflight_identity_sha256"] = json_sha256(control_identity)
    identity["control_cache_hashes"] = preflight_r02["control_cache_sha256"]
    identity["resource_preflight"] = _stable_resource_identity(resource)
    identity["source_commit"] = recipe_record["source_commit"]

    if PREFLIGHT_PATH.exists():
        frozen_preflight = _json(PREFLIGHT_PATH)
        if frozen_preflight.get("freeze_identity_sha256") != json_sha256(identity):
            raise BatchIntegrityError("Existing R05 preflight is bound to a different freeze identity")
        summary_preflight = frozen_preflight
    else:
        summary_preflight["freeze_identity_sha256"] = json_sha256(identity)
        write_json_exclusive(PREFLIGHT_PATH, summary_preflight)
    append_event(ATTEMPT_PATH, {"event": "PREFLIGHT_PASS", "card_id": "R05", **summary_preflight})
    pilot = run_pilot(
        frame=frame,
        target=target,
        card=card,
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=r02._control_loader(controls),
        freeze_identity=identity,
        freeze_path=FREEZE_PATH,
        artifact_dir=RUN_DIR,
        event_path=ATTEMPT_PATH,
        evaluation_views=views,
        threads=2,
        report_path=ENGINE_REPORT_PATH,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
    )
    verification = verify_report(
        report=pilot,
        frame=frame,
        target=target,
        artifact_dir=RUN_DIR,
        freeze_signature=pilot["freeze_signature"],
        evaluation_views=views,
        prediction_reconstructors={"R05": reconstruct_r05},
    )
    subgroup = _subgroup_diagnostics(pilot, frame, target, views)
    accounting = _read_attempt_counts()
    accounting["fit_accounting_matches_card"] = (
        accounting["successful_predictor_fits"] == 2
        and accounting["fit_saved_events"] == 2
        and accounting["vectorizer_fit_calls"] == 2
        and all((RUN_DIR / f"R05_fit{cutoff}.npz").is_file() and
                (RUN_DIR / f"R05_fit{cutoff}.json").is_file() for cutoff in (55, 67))
    )
    if not accounting["fit_accounting_matches_card"]:
        raise BatchIntegrityError(f"R05 fit attempt accounting does not match two-predictor card: {accounting}")

    gate_config = queue["evaluation"]["regression_gate"]
    gains = pilot["views"]
    checks = {
        "pilot_primary_MAE_gain_at_least_0_03": gains["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_MAE_gain_min"]),
        "H12_MAE_harm_at_most_0_05": gains["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_at_most_0_5": gains["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_at_most_0_5": gains["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_group_harms": len(subgroup["supported_guard_failures"]) == 0,
        "independent_verification_pass": verification.get("status") in {"PASS", "INDEPENDENT_QA_PASS"},
        "fit_attempt_accounting_pass": accounting["fit_accounting_matches_card"],
    }
    gate = {
        "card_id": "R05",
        "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": gate_config,
        "subgroup_diagnostics": subgroup,
        "decision_basis": "Frozen W1+W4/H12 Train pilot and declared regression gates only; no Test/Kaggle information.",
    }
    results = {
        "card_id": "R05",
        "status": gate["status"],
        "source_commit": recipe_record["source_commit"],
        "freeze_signature": pilot["freeze_signature"],
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "preflight": summary_preflight,
        "pilot": pilot,
        "fit_accounting": accounting,
        "independent_verification_status": verification.get("status"),
        "subgroups": subgroup,
        "gate": gate,
        "official_test_read": False,
        "kaggle_accessed": False,
    }
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(SUBGROUP_PATH, subgroup)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(RESULTS_PATH, results)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != _render_summary(results, gate):
            raise BatchIntegrityError("Existing R05 summary differs; refusing overwrite")
    else:
        from src.models.optimization_v3_batch1 import _write_link_exclusive
        _write_link_exclusive(SUMMARY_PATH, _render_summary(results, gate).encode("utf-8"))
    ledger_event = {
        "event": "FIT_BUDGET_RECONCILED",
        "card_id": "R05",
        "budget_key": "R05:PILOT",
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": accounting["successful_predictor_fits"],
        "fit_started_events": accounting["fit_attempts_started"],
        "fit_failed_events": accounting["fit_failed_events"],
        "attempt_accounting_complete": accounting["fit_accounting_matches_card"],
        "decision": gate["status"],
        "freeze_signature": pilot["freeze_signature"],
        "results_sha256": file_sha256(RESULTS_PATH),
        "gate_sha256": file_sha256(GATE_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "independent_verification_status": verification.get("status"),
    }
    _append_event_once(LEDGER_PATH, ledger_event)
    return results


__all__ = ["reconstruct_r05", "run_card"]


if __name__ == "__main__":
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next(item for item in queue["cards"] if item["id"] == "R05")
    print(json.dumps(run_card(card, queue), indent=2, allow_nan=False))
