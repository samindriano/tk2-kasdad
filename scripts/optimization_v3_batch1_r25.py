"""R25 frozen unnormalized word-count pilot with the shared V3 harness."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_r02 as r02
from scripts import optimization_v3_batch1_r05 as r05_runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    make_freeze_identity,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r25 import COUNT_PARAMETERS, FEATURE_NAMES, fit_predict
from src.models.optimization_v3_batch1_r05 import compose_r05_prediction
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R25.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R25"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R25"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R25"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUBGROUP_PATH = REPORT_DIR / "subgroups.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
ATTEMPT_PATH = REPORT_DIR / "attempts.jsonl"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"

EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
EXPECTED_QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
EXPECTED_CARD_SHA256 = "cee89a792a0a16ab26d2545d5f77d8212e8ef0bf966f751a0ca7e0a82361f321"
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_CYCLES = np.arange(47, 80, dtype=np.int64)
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
EXPECTED_CARD = {
    "id": "R25",
    "task": "regression",
    "name": "Unnormalized word counts",
    "family": "count_presence",
    "priority": "P2",
    "change": "Original word1Ã¢â‚¬â€œ2/min_df2/cap20000 CountVectorizer(binary=False), no IDF or row norm; original structured sparse L1 model. Second count_presence recipe alongside R05.",
    "policy": "R_SPARSE_REPLACE",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "implementation": "NOT_IMPLEMENTED",
    "card_path": "reports/optimization_v3/research_backlog/cards/R25.md",
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
RECIPE = {
    "card_id": "R25",
    "model": "LightGBM L1 sparse hybrid regression",
    "structured_features": list(FEATURE_NAMES),
    "text_vectorizer": {
        "analyzer": "word",
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_features": 20_000,
        "binary": False,
        "lowercase": True,
        "token_pattern": r"(?u)\b\w\w+\b",
        "strip_accents": None,
        "dtype": "int64",
        "use_idf": False,
        "row_normalization": None,
    },
    "structured_preprocessing": "R05 inherited fit-only numeric imputation/scaling + OneHotEncoder(min_frequency=30,max_categories=254,handle_unknown=infrequent_if_exist)",
    "missing_stream_indicator": True,
    "sparse_lgb_parameters": dict(r02.SPARSE_LGB_PARAMETERS),
    "policy": "clip(0.75*Ridge25 + 0.25*rank50(new raw R25 sparse prediction),0,100)",
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
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _write_or_match(path: Path, payload: Any) -> None:
    if path.exists():
        if _json(path) != payload:
            raise BatchIntegrityError(f"Existing R25 output differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _input_files(train_path: Path, card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Path]:
    files = r02._input_files(train_path, card, queue)
    files["canonical_queue"] = QUEUE_PATH
    files["canonical_card"] = CARD_PATH
    return files


def _source_files() -> list[Path]:
    return sorted({
        *r02._source_files(),
        Path(r05_runner.__file__),
        ROOT / "src/models/optimization_v3_batch1_r05.py",
        ROOT / "tests/test_optimization_v3_batch1_r05.py",
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r25.py",
        ROOT / "tests/test_optimization_v3_batch1_r25.py",
    }, key=str)


def _stable_resource_identity(resource: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in resource.items() if key not in {
        "available_ram_gib_observed", "allowed_job_ram_gib"
    }}


def _subgroups(report: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
               views: Mapping[str, list[tuple[int, int, int]]]) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    bundles: dict[int, dict[str, np.ndarray]] = {}
    from src.models.optimization_v3_batch1 import read_bundle
    for cutoff in report["cutoffs"]:
        path = RUN_DIR / f"R25_fit{int(cutoff)}.npz"
        sidecar = _json(path.with_suffix(".json"))
        arrays, _ = read_bundle(path, sidecar["identity"])
        bundles[int(cutoff)] = {"rows": np.asarray(arrays["score_rows"], dtype=np.int64),
                                "candidate": np.asarray(arrays["prediction"], dtype=np.float64),
                                "baseline": np.asarray(arrays["baseline_prediction"], dtype=np.float64)}
    output: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for view in ("primary", "H12"):
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in views[view]:
            bundle = bundles[int(cutoff)]
            mask = (cycles[bundle["rows"]] >= int(low)) & (cycles[bundle["rows"]] <= int(high))
            row_parts.append(bundle["rows"][mask])
            pred_parts.append(bundle["candidate"][mask])
            base_parts.append(bundle["baseline"][mask])
        rows = np.concatenate(row_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R25 subgroup QA found duplicate rows in {view}")
        y = target[rows]
        candidate = np.concatenate(pred_parts)
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
        result: dict[str, Any] = {}
        for name, mask in groups.items():
            n = int(mask.sum())
            supported = n >= 100
            harm = float(np.mean(np.abs(candidate[mask] - y[mask]) - 0.0) -
                         np.mean(np.abs(baseline[mask] - y[mask]))) if n else None
            guard = name in {"stream_missing", "target_le_5", "target_ge_95"} and supported
            record = {"n": n, "supported": supported,
                      "evidence_status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                      "mae_harm_candidate_minus_control": harm,
                      "numeric_guard_applies": guard,
                      "guard_pass": None if not guard else bool(harm <= 0.25)}
            result[name] = record
            if guard and harm > 0.25:
                failures.append({"view": view, "group": name, **record})
        output[view] = result
    return {"minimum_support": 100, "maximum_supported_tail_or_missing_MAE_harm": 0.25,
            "views": output, "supported_guard_failures": failures}


def reconstruct_r25(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return r05_runner.reconstruct_r05(arrays)


def _fit_accounting() -> dict[str, Any]:
    events = [json.loads(line) for line in ATTEMPT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    own = [item for item in events if item.get("card_id") == "R25"]
    started = [item for item in own if item.get("event") == "FIT_STARTED"]
    saved = [item for item in own if item.get("event") == "FIT_SAVED"]
    failed = [item for item in own if item.get("event") == "FIT_FAILED"]
    seconds = [float(item["fit_seconds"]) for item in saved]
    vectorizer_calls = 0
    for cutoff in (55, 67):
        sidecar = RUN_DIR / f"R25_fit{cutoff}.json"
        if sidecar.is_file():
            vectorizer_calls += int(_json(sidecar).get("metadata", {}).get("fit_metadata", {}).get("vectorizer_fit_calls", 0))
    return {
        "fit_attempts_started": len(started),
        "successful_predictor_fits": sum(int(item.get("fit_predictor_count", 0)) for item in saved),
        "fit_failed_events": len(failed),
        "fit_saved_events": len(saved),
        "saved_fit_seconds": seconds,
        "total_saved_fit_seconds": float(sum(seconds)),
        "vectorizer_fit_calls": vectorizer_calls,
        "fit_accounting_matches_card": len(saved) == 2 and sum(
            int(item.get("fit_predictor_count", 0)) for item in saved
        ) == 2 and vectorizer_calls == 2 and all(
            (RUN_DIR / f"R25_fit{cutoff}.npz").is_file() and (RUN_DIR / f"R25_fit{cutoff}.json").is_file()
            for cutoff in (55, 67)
        ),
    }


def _append_once(event: Mapping[str, Any]) -> None:
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                current = json.loads(line)
                if all(current.get(key) == value for key, value in event.items()):
                    return
    append_event(LEDGER_PATH, event)


def _summary(results: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    views = results["pilot"]["views"]
    lines = [
        "# R25 — Unnormalized word counts",
        "",
        f"**Verdict: {gate['status']}** — fixed temporal Train pilot, no Test/Kaggle access.",
        "",
        "Recipe: word unigram/bigram CountVectorizer, min_df=2, cap=20,000, raw counts (`binary=False`), no IDF/normalization; inherited structured/OHE + missing indicator; sparse LightGBM regression_l1.",
        "",
        "| View | n | Control MAE | R25 MAE | Gain (control − R25) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("W1", "W4", "primary", "H12"):
        item = views[name]
        lines.append(f"| {name} | {item['rows']:,} | {item['baseline']['mae']:.6f} | {item['candidate']['mae']:.6f} | {item['gain_positive_is_better']:+.6f} |")
    lines.extend([
        "",
        "The comparator is the frozen three-component Regression champion on the exact same rows. Preprocessing is fit-only at each temporal cutoff; H12 reuses development Train windows and is not an untouched holdout.",
        "",
        "See `results.json`, `verification.json`, `subgroups.json`, `gate.json`, and `runtime_notes.json` for fit accounting, denominators, support, cache reconstruction, memory, and runtime notes.",
        "",
    ])
    return "\n".join(lines)


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    mismatches = {key: (card.get(key), value) for key, value in EXPECTED_CARD.items() if card.get(key) != value}
    if mismatches:
        raise BatchIntegrityError(f"R25 canonical card assertions failed: {mismatches}")
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R25 runs only while the V3 campaign remains unfinished")
    queue_hash = file_sha256(QUEUE_PATH)
    if queue_hash != EXPECTED_QUEUE_SHA256:
        raise BatchIntegrityError(f"R25 canonical queue hash differs: {queue_hash}")
    card_hash = file_sha256(CARD_PATH)
    if card_hash != EXPECTED_CARD_SHA256:
        raise BatchIntegrityError(f"R25 canonical card hash differs: {card_hash}")

    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file() or file_sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train is missing or differs from frozen R25 input")
    frame = pd.read_csv(train_path)
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise BatchIntegrityError(f"R25 expected {EXPECTED_TRAIN_ROWS} official Train rows")
    required = set(FEATURE_NAMES) | {"record_id", "cycle_code", "threat_rank"}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise BatchIntegrityError(f"R25 Train is missing required fields: {absent}")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R25 feature allowlist contains a protected field")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any() or not frame["record_id"].astype(str).is_unique:
        raise BatchIntegrityError("R25 Train record_id values are blank or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), EXPECTED_CYCLES):
        raise BatchIntegrityError("R25 Train cycle coverage differs from 47-79")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise BatchIntegrityError("R25 threat_rank is invalid")

    views = evaluation_views_for_card(queue, "R25")
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R25 temporal views changed: {views}")
    inputs = _input_files(train_path, card, queue)
    preflight_control, controls, control_identity = r02._preflight(frame, train_path, card, queue, views, inputs)
    # Binary=True and False have identical token support and CSR occupancy; R05's analyzer is reused.
    resource = r05_runner._resource_preflight(frame, cycles, card)
    identity = make_freeze_identity(card=card, feature_names=FEATURE_NAMES, input_files=inputs,
                                    source_files=_source_files(), evaluation_views=views, packages=PACKAGES)
    identity["candidate_recipe"] = RECIPE
    identity["candidate_recipe_sha256"] = json_sha256(RECIPE)
    identity["control_preflight_identity_sha256"] = json_sha256(control_identity)
    identity["control_cache_hashes"] = preflight_control["control_cache_sha256"]
    identity["resource_preflight"] = _stable_resource_identity(resource)
    identity["source_commit"] = _source_commit()
    summary_preflight = {
        "status": "PREFLIGHT_PASS",
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "train_rows": len(frame),
        "fit_rows": {"55": int(np.sum(cycles <= 55)), "67": int(np.sum(cycles <= 67))},
        "score_rows": {"55": int(np.sum((cycles >= 56) & (cycles <= 67))), "67": int(np.sum((cycles >= 68) & (cycles <= 79)))},
        "view_rows": {name: int(sum(np.sum((cycles >= low) & (cycles <= high)) for _, low, high in parts)) for name, parts in views.items()},
        "pilot_control_primary_mae": preflight_control["primary_control_mae"],
        "pilot_control_primary_rows": preflight_control["primary_control_rows"],
        "H12_control_mae": preflight_control["H12_control_mae"],
        "H12_control_rows": preflight_control["H12_control_rows"],
        "resource": {**resource, "binary_false_has_same_sparsity_support_as_binary_true": True},
        "cache_sha256": preflight_control["control_cache_sha256"],
        "freeze_identity_sha256": json_sha256(identity),
        "official_test_read": False,
        "model_fit_performed": False,
    }
    if preflight_only:
        return {"status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT", "card_id": "R25",
                "preflight": summary_preflight, "evaluation_views": views,
                "source_files": len(_source_files()), "official_test_read": False,
                "model_fit_performed": False, "freeze_written": False}

    for directory in (EXPERIMENT_DIR, RUN_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    recipe_record = {"recipe": RECIPE, "queue_sha256": queue_hash, "card_sha256": card_hash,
                     "source_commit": _source_commit()}
    _write_or_match(RECIPE_PATH, recipe_record)
    identity["recipe_record_sha256"] = json_sha256(recipe_record)
    # Keep preflight/resume identities stable while binding the exact recipe record.
    identity["source_commit"] = recipe_record["source_commit"]
    summary_preflight["freeze_identity_sha256"] = json_sha256(identity)
    if PREFLIGHT_PATH.exists():
        frozen_preflight = _json(PREFLIGHT_PATH)
        if frozen_preflight.get("freeze_identity_sha256") != summary_preflight["freeze_identity_sha256"]:
            raise BatchIntegrityError("Existing R25 preflight is bound to a different freeze identity")
        summary_preflight = frozen_preflight
    else:
        write_json_exclusive(PREFLIGHT_PATH, summary_preflight)
    append_event(ATTEMPT_PATH, {"event": "PREFLIGHT_PASS", "card_id": "R25", **summary_preflight})

    pilot = run_pilot(
        frame=frame, target=target, card=card, feature_names=FEATURE_NAMES,
        adapter=fit_predict, control_loader=r02._control_loader(controls),
        freeze_identity=identity, freeze_path=FREEZE_PATH, artifact_dir=RUN_DIR,
        event_path=ATTEMPT_PATH, evaluation_views=views, threads=2,
        report_path=ENGINE_REPORT_PATH, budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
    )
    verification = verify_report(report=pilot, frame=frame, target=target,
                                 artifact_dir=RUN_DIR, freeze_signature=pilot["freeze_signature"],
                                 evaluation_views=views, prediction_reconstructors={"R25": reconstruct_r25})
    subgroup = _subgroups(pilot, frame, target, views)
    accounting = _fit_accounting()
    checks = {
        "pilot_primary_MAE_gain_at_least_0_03": pilot["views"]["primary"]["gain_positive_is_better"] >= 0.03,
        "H12_MAE_harm_at_most_0_05": pilot["views"]["H12"]["gain_positive_is_better"] >= -0.05,
        "W1_MAE_harm_at_most_0_5": pilot["views"]["W1"]["gain_positive_is_better"] >= -0.5,
        "W4_MAE_harm_at_most_0_5": pilot["views"]["W4"]["gain_positive_is_better"] >= -0.5,
        "supported_tail_or_missing_group_harms": len(subgroup["supported_guard_failures"]) == 0,
        "independent_verification_pass": verification.get("status") == "PASS",
        "fit_attempt_accounting_pass": accounting["fit_accounting_matches_card"],
    }
    gate = {
        "card_id": "R25", "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": queue["evaluation"]["regression_gate"],
        "subgroup_diagnostics": subgroup,
        "decision_basis": "Frozen W1+W4/H12 Train pilot and declared regression gates only; no Test/Kaggle information.",
    }
    results = {
        "card_id": "R25", "status": gate["status"], "source_commit": recipe_record["source_commit"],
        "freeze_signature": pilot["freeze_signature"],
        "input_sha256": identity["input_sha256"], "source_sha256": identity["source_sha256"],
        "preflight": summary_preflight, "pilot": pilot, "fit_accounting": accounting,
        "independent_verification_status": verification.get("status"),
        "subgroups": subgroup, "gate": gate,
        "official_test_read": False, "kaggle_accessed": False,
    }
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(SUBGROUP_PATH, subgroup)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(RESULTS_PATH, results)
    summary_text = _summary(results, gate)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary_text:
            raise BatchIntegrityError("Existing R25 summary differs; refusing overwrite")
    else:
        from src.models.optimization_v3_batch1 import _write_link_exclusive
        _write_link_exclusive(SUMMARY_PATH, summary_text.encode("utf-8"))
    ledger_event = {
        "event": "FIT_BUDGET_RECONCILED", "card_id": "R25", "budget_key": "R25:PILOT",
        "reserved_predictor_fits": 2, "successful_predictor_fits": accounting["successful_predictor_fits"],
        "fit_started_events": accounting["fit_attempts_started"], "fit_failed_events": accounting["fit_failed_events"],
        "attempt_accounting_complete": accounting["fit_accounting_matches_card"],
        "decision": gate["status"], "freeze_signature": pilot["freeze_signature"],
        "results_sha256": file_sha256(RESULTS_PATH), "gate_sha256": file_sha256(GATE_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "independent_verification_status": verification.get("status"),
    }
    _append_once(ledger_event)
    return results


__all__ = ["reconstruct_r25", "run_card"]


if __name__ == "__main__":
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next(item for item in queue["cards"] if item["id"] == "R25")
    print(json.dumps(run_card(card, queue), indent=2, allow_nan=False))
