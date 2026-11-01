"""Freeze, run, and report the Research Next V1 N06 token-surface diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import optimization_v3_research_next_n00 as n00
from src.models.classification_models import _text_values
from src.models.optimization_v3_batch1_r06 import (
    CHAR_TFIDF_PARAMETERS, FEATURE_NAMES, R06CharacterSparseL1Hybrid, compose_r06_prediction,
)
from src.models.regression_research_text import _text_documents
from src.models.research_next_budget import (
    finish_fit, ledger_summary, read_events, release_unused, reserve_card, start_fit,
)
from src.models.research_next_contract import (
    canonical_sha256, file_sha256, publish_bundle, read_bundle, resume_or_publish_bundle,
)
from src.models import research_next_n06 as n06
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
CARD_ID = "N06"
STAGE = "PILOT"
PILOT_FITS = 4
ORIGINS = (55, 67)
SEEDS = (42, 137)
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_next_v1/cards/N06.md"
N00_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
N00_REPORT_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05/summary.json"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
R06_RESULT_PATH = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json"
R06_VERIFY_PATH = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json"
R22_VERIFY_PATH = ROOT / "reports/optimization_v3/final_continuation/R22/verification.json"
FREEZE_ROOT = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N06/amendment_01"
PREFLIGHT_PATH = FREEZE_ROOT / "preflight.json"
FREEZE_PATH = FREEZE_ROOT / "freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N06/amendment_01"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N06/amendment_01"
RECOVERY_ROOT = BUNDLE_ROOT / "recovery"

SOURCE_PATHS = (
    "scripts/optimization_v3_research_next_n06.py",
    "scripts/optimization_v3_research_next_n06_verify.py",
    "scripts/optimization_v3_research_next_n00.py",
    "src/models/research_next_n06.py",
    "src/models/research_next_contract.py",
    "src/models/research_next_budget.py",
    "src/models/optimization_v3_batch1_r06.py",
    "scripts/optimization_v3_batch1_r06.py",
    "src/models/optimization_v2_regression.py",
    "src/models/regression_research_text.py",
    "src/models/round1_models.py",
    "src/models/classification_models.py",
    "src/utils/paths.py",
    "tests/test_research_next_n06.py",
)


class ResourceDeferred(RuntimeError):
    """The fit cannot safely run within the frozen machine-resource bound."""


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _task_source_commit() -> str:
    task_paths = [name for name in SOURCE_PATHS if name.startswith(("src/models/research_next_n06", "scripts/optimization_v3_research_next_n06", "tests/test_research_next_n06"))]
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", *task_paths],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _is_ancestor(commit: str) -> bool:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                            cwd=ROOT, capture_output=True)
    return result.returncode == 0


def _dependencies() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil")
    return {name: importlib.metadata.version(name) for name in packages}


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(value):
            raise RuntimeError(f"N06 refuses to overwrite immutable artifact: {_relative(path)}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _train_frame() -> tuple[Path, pd.DataFrame, np.ndarray]:
    path = Path(task_files("regression")["train"])
    frame = pd.read_csv(path)
    required = {"record_id", "cycle_code", "threat_rank", *FEATURE_NAMES}
    if (len(frame) != 50_608 or not required.issubset(frame.columns)
            or frame.record_id.isna().any() or not frame.record_id.is_unique):
        raise RuntimeError("N06 official Regression Train shape/schema/IDs differ")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80)):
        raise RuntimeError("N06 official Regression Train cycles are not 47-79")
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise RuntimeError("N06 official Regression target is nonfinite/outside [0,100]")
    return path, frame, target


def _load_controls(frame: pd.DataFrame) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    for card in ("R22", "R06"):
        n00._verify_historical_report(card)
    sources: dict[int, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    ids = frame.record_id.to_numpy(dtype=str)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    for cutoff in ORIGINS:
        r22_bundle = n00._load_artifact("R22", cutoff, frame)
        r06_bundle = n00._load_artifact("R06", cutoff, frame)
        r22 = n00.reconstruct_regression(r22_bundle)
        r06 = n00.reconstruct_regression(r06_bundle)
        rows = r22["rows"].astype(np.int64)
        expected = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if (not np.array_equal(rows, expected) or not np.array_equal(r06["rows"], rows)
                or not np.array_equal(r22_bundle["arrays"]["score_record_ids"].astype(str), ids[rows])
                or not np.array_equal(r06_bundle["arrays"]["score_record_ids"].astype(str), ids[rows])):
            raise RuntimeError(f"N06 cached R22/R06 rows or IDs fail at cutoff {cutoff}")
        sources[cutoff] = {
            "rows": rows, "ids": ids[rows].astype("U"), "cycles": cycles[rows],
            "r22": r22["prediction"].astype(np.float64),
            "r06": r06["prediction"].astype(np.float64),
            "legacy": r06["legacy"].astype(np.float64),
        }
        audit[str(cutoff)] = {
            "fit_rows": int(r06_bundle["fit_rows"]), "score_rows": int(len(rows)),
            "R22_npz_sha256": r22_bundle["npz_sha256"],
            "R22_sidecar_sha256": r22_bundle["sidecar_sha256"],
            "R22_identity_sha256": r22_bundle["sidecar"]["identity_sha256"],
            "R06_npz_sha256": r06_bundle["npz_sha256"],
            "R06_sidecar_sha256": r06_bundle["sidecar_sha256"],
            "R06_identity_sha256": r06_bundle["sidecar"]["identity_sha256"],
            "R22_recipe_reconstruction": True, "R06_rank50_policy_reconstruction": True,
        }
    return sources, audit


def _resource_estimate(frame: pd.DataFrame, cutoff: int, seed: int,
                       available: int | None = None) -> dict[str, Any]:
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= cutoff)
    source_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
    fit_frame = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
    score_frame = frame.iloc[source_rows].loc[:, list(FEATURE_NAMES)]
    renamed_fit, fit_audit = n06.rename_features(fit_frame, seed)
    renamed_score, score_audit = n06.rename_features(score_frame, seed)
    fit_docs, _ = _text_documents(renamed_fit["signal_stream"])
    score_docs, _ = _text_documents(renamed_score["signal_stream"])
    analyzer = TfidfVectorizer(**dict(CHAR_TFIDF_PARAMETERS)).build_analyzer()

    def bounds(documents: list[str]) -> tuple[int, int, int]:
        occurrence = 0
        per_document_unique_upper = 0
        nonempty = 0
        for document in documents:
            terms = analyzer(document)
            occurrence += len(terms)
            per_document_unique_upper += len(set(terms))
            nonempty += bool(terms)
        return occurrence, per_document_unique_upper, nonempty

    fit_occ, fit_unique_upper, fit_nonempty = bounds(fit_docs)
    score_occ, score_unique_upper, score_nonempty = bounds(score_docs)
    feature_cap = int(CHAR_TFIDF_PARAMETERS["max_features"])
    fit_char_nnz = min(fit_unique_upper, len(fit_rows) * feature_cap)
    score_char_nnz = min(score_unique_upper, len(source_rows) * feature_cap)
    fit_structured_nnz = len(fit_rows) * 64
    score_structured_nnz = len(source_rows) * 64
    fit_nnz = fit_char_nnz + fit_structured_nnz + len(fit_rows)
    score_nnz = score_char_nnz + score_structured_nnz + len(source_rows)
    fit_csr = fit_nnz * 12 + (len(fit_rows) + 1) * 8
    score_csr = score_nnz * 12 + (len(source_rows) + 1) * 8
    width_upper = feature_cap + 1_000 + 1
    histogram = width_upper * 256 * 16 * THREADS * 2
    estimate = 3 * fit_csr + 2 * score_csr + histogram + 512 * 1024**2 + 512 * 1024**2
    free = int(psutil.virtual_memory().available if available is None else available)
    allowed = min(JOB_CAP_BYTES, max(0, free - FREE_RESERVE_BYTES))
    return {
        "cutoff": int(cutoff), "seed": int(seed), "fit_rows": int(len(fit_rows)),
        "score_rows": int(len(source_rows)), "fit_nonempty_documents": int(fit_nonempty),
        "score_nonempty_documents": int(score_nonempty),
        "fit_char_ngram_occurrences": int(fit_occ), "score_char_ngram_occurrences": int(score_occ),
        "fit_char_matrix_nnz_upper": int(fit_char_nnz), "score_char_matrix_nnz_upper": int(score_char_nnz),
        "fit_structured_nnz_upper": int(fit_structured_nnz), "score_structured_nnz_upper": int(score_structured_nnz),
        "fit_csr_bytes_upper": int(fit_csr), "score_csr_bytes_upper": int(score_csr),
        "estimated_peak_bytes": int(estimate), "estimate_is_observed": False,
        "available_physical_bytes": free, "allowed_bytes": int(allowed),
        "minimum_free_bytes": MINIMUM_FREE_BYTES, "reserve_bytes": FREE_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES, "threads": THREADS, "concurrent_jobs": 1,
        "renamed_fit_token_occurrences": fit_audit["token_occurrences_renamed"],
        "renamed_score_token_occurrences": score_audit["token_occurrences_renamed"],
        "status": "PASS" if free >= MINIMUM_FREE_BYTES and estimate <= allowed else "DEFERRED_RESOURCE",
    }


def _source_paths() -> list[Path]:
    paths = [ROOT / value for value in SOURCE_PATHS]
    paths += [QUEUE_PATH, READINESS_PATH, CARD_PATH, N00_FREEZE_PATH, N00_REPORT_PATH,
              R06_RESULT_PATH, R06_VERIFY_PATH, R22_VERIFY_PATH]
    return paths


def _build_identity(train_path: Path, frame: pd.DataFrame, schema: Mapping[str, Any],
                    bijections: Mapping[str, Any], cache_audit: Mapping[str, Any],
                    resources: Mapping[str, Any]) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next((item for item in queue.get("cards", []) if item.get("id") == CARD_ID), None)
    if (queue.get("batch_id") != BATCH or not card or card.get("task") != "regression"
            or card.get("kind") != "information" or card.get("profile") != "STANDARD"
            or int(card.get("pilot_fits", -1)) != PILOT_FITS
            or int(card.get("continuation_fits", -1)) != 0):
        raise RuntimeError("N06 canonical queue identity/allocation changed")
    if file_sha256(train_path) != queue["raw_train_sha256"]["regression"]:
        raise RuntimeError("N06 official Regression Train hash differs from canonical queue")
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    if readiness.get("raw_train_hashes", {}).get("regression") != file_sha256(train_path):
        raise RuntimeError("N06 readiness raw Train hash differs")
    n00_freeze = json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8"))
    if n00_freeze.get("freeze_signature") != canonical_sha256(n00_freeze.get("identity")):
        raise RuntimeError("N06 N00 freeze signature is invalid")
    n00_commit = n00_freeze.get("identity", {}).get("source_commit")
    if not n00_commit or not _is_ancestor(n00_commit):
        raise RuntimeError("N06 N00 source commit is not an ancestor of HEAD")
    inherited = n00_freeze.get("execution_context", {}).get("inherited_old_budget_snapshot", {})
    if not inherited.get("sha256") or file_sha256(OLD_LEDGER_PATH) != inherited["sha256"]:
        raise RuntimeError("N06 original campaign ledger differs from frozen N00 snapshot")
    input_paths = [train_path]
    inputs = {_relative(path): file_sha256(path) for path in input_paths}
    source_files = _source_paths()
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"N06 frozen source/report inputs are missing: {missing}")
    sources: dict[str, str] = {}
    for path in source_files:
        relative = _relative(path)
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                                  check=True, capture_output=True).stdout
        working = path.read_bytes()
        if working != committed:
            raise RuntimeError(f"N06 source is not byte-identical to committed HEAD: {relative}")
        sources[relative] = hashlib.sha256(committed).hexdigest()
    fit_rows = {str(c): int(np.sum(frame.cycle_code.to_numpy(dtype=np.int64) <= c)) for c in ORIGINS}
    resource_bounds = {
        str(key): {name: value for name, value in report.items()
                   if name not in {"available_physical_bytes", "allowed_bytes", "status"}}
        for key, report in resources.items()
    }
    return {
        "batch_id": BATCH, "card_id": CARD_ID, "stage": STAGE,
        "source_commit": _task_source_commit(),
        "lane_kind": "information_only_no_continuation_or_point_model_promotion",
        "question": "Does R06 depend on anonymous character surface beyond recurring token identity?",
        "official_train_sha256": file_sha256(train_path), "train_rows": int(len(frame)),
        "fit_population": {"rule": "all official Regression Train rows with cycle_code <= origin",
                           "fit_rows_by_cutoff": fit_rows,
                           "fit_cycle_range_by_cutoff": {str(c): [47, c] for c in ORIGINS}},
        "score_populations": {str(c): "complete next 12 official Train cycles only" for c in ORIGINS},
        "recipe": {
            "model": "exact original R06CharacterSparseL1Hybrid",
            "original_source_sha256": sources[_relative(ROOT / "src/models/optimization_v3_batch1_r06.py")],
            "features": list(FEATURE_NAMES),
            "only_input_change": "deterministic Feistel relabeling within q7/z6 token payloads",
            "seeds": list(SEEDS), "origins": list(ORIGINS),
            "mapping": {
                "domain": {"q": "q plus seven lowercase hexadecimal digits; 28-bit payload",
                           "z": "z plus six lowercase hexadecimal digits; 24-bit payload"},
                "rounds": n06.FEISTEL_ROUNDS,
                "half_bits": {"q": 14, "z": 12},
                "round_function": "low halfbits of big-endian SHA256 integer over UTF-8 research_next_v1|seed|prefix|round|right_decimal",
                "update": "(L,R)=(R,L xor F)", "preserve": ["prefix", "width", "separators", "order", "multiplicity", "repetition", "missingness"],
                "future_domain_mapping": True,
                "no_test_lexicon_or_learned_token_table": True,
            },
            "r06_character_tfidf_parameters": {key: (list(value) if isinstance(value, tuple)
                                                       else str(value) if key == "dtype" else value)
                                               for key, value in CHAR_TFIDF_PARAMETERS.items()},
            "r06_blend_policy": ".75 protected three-component legacy + .25 rank50(raw recoded R06), clip[0,100]",
            "rank50": "within complete cycle, average ties, .5 raw + .5 percentile, clip[0,100]",
            "outer_score_labels_passed_to_fit": False,
        },
        "preconditions": {"token_schema": {key: value for key, value in schema.items()
                                               if key != "unique_tokens_by_prefix"},
                          "bijection_checks": dict(bijections),
                          "unexpected_shapes_fail_closed": True},
        "controls": {"primary_mechanism_control": "R06_ORIGINAL verified cache",
                     "campaign_reference": "R22_ORIGINAL verified cache",
                     "protected_legacy": "original R06 three-component champion cache"},
        "views": {"primary": [[55, 56, 59], [67, 68, 71]],
                  "H12": [[55, 56, 67], [67, 68, 79]],
                  "windows": {f"W{i+1}": [55 if i < 3 else 67,
                                           (56 + 4 * i) if i < 3 else (68 + 4 * (i - 3)),
                                           (59 + 4 * i) if i < 3 else (71 + 4 * (i - 3))]
                              for i in range(6)},
                  "per_cutoff_H12": {"55": [55, 56, 67], "67": [67, 68, 79]}},
        "groups": ["signal_stream missing/present", "observed target <=5 (evaluation only)",
                   "observed target >=95 (evaluation only)"],
        "minimum_group_support": 100,
        "cache_lineage": dict(cache_audit),
        "queue_sha256": file_sha256(QUEUE_PATH),
        "readiness_sha256": file_sha256(READINESS_PATH),
        "source_sha256": sources, "input_sha256": inputs,
        "resource_bounds": resource_bounds, "dependencies": _dependencies(),
        "test_data_used": False, "sample_submission_used": False, "kaggle_calls": 0,
    }


def prepare_context(*, available_bytes: int | None = None,
                    ledger_context_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    train_path, frame, target = _train_frame()
    controls, cache_audit = _load_controls(frame)
    schema = n06.inspect_token_schema(frame.signal_stream)
    if schema["status"] != "PASS":
        raise RuntimeError(f"N06 fail-closed token schema: {schema}")
    bijections = {str(seed): n06.verify_observed_bijection(schema, seed) for seed in SEEDS}
    # Verify that the recoder catches every declared Train occurrence without altering separators.
    _, recoding_audit = n06.rename_series(frame.signal_stream, SEEDS[0])
    expected_occurrences = sum(schema["token_occurrences"].values())
    if recoding_audit["token_occurrences_renamed"] != expected_occurrences:
        raise RuntimeError("N06 stream renamer did not map every declared Train token occurrence")
    resources = {
        f"{cutoff}:{seed}": _resource_estimate(frame, cutoff, seed, available=available_bytes)
        for cutoff in ORIGINS for seed in SEEDS
    }
    identity = _build_identity(train_path, frame, schema, bijections, cache_audit, resources)
    ledger = {"ledger_sha256": file_sha256(LEDGER_PATH), "ledger": ledger_summary(LEDGER_PATH)}
    if ledger_context_override is not None:
        ledger = dict(ledger_context_override)
    return {
        "identity": identity, "freeze_signature": canonical_sha256(identity),
        "execution_context": {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": _git_head(),
            "python": platform.python_version(),
            "available_bytes": int(psutil.virtual_memory().available if available_bytes is None else available_bytes),
            **ledger, "resources": resources,
            "official_test_opened": False, "test_labels_read": False, "sample_submission_opened": False,
            "kaggle_calls": 0,
        },
        "train_path": train_path, "frame": frame, "target": target,
        "controls": controls, "cache_audit": cache_audit, "schema": schema,
        "bijections": bijections, "resources": resources,
    }


def _synthetic_smoke() -> dict[str, Any]:
    rng = np.random.default_rng(20261102)
    n = 240
    cycles = np.repeat(np.arange(47, 53, dtype=np.int64), 40)
    frame = pd.DataFrame(index=np.arange(n))
    frame["cycle_code"] = cycles
    for name in FEATURE_NAMES:
        if name == "signal_stream":
            frame[name] = [None if i % 17 == 0 else
                           f"q{i % 31:07x} z{i % 7:06x} q{i % 31:07x}"
                           for i in range(n)]
        elif name in {"entry_friction", "org_code", "module_code", "pattern_code"}:
            frame[name] = [f"cat_{i % 5}" if i < 160 else "unseen_score_category" for i in range(n)]
        elif name == "scope_pressure":
            frame[name] = np.nan
        else:
            frame[name] = rng.normal(size=n)
    fit_rows = np.flatnonzero(cycles <= 50)
    score_rows = np.flatnonzero((cycles > 50) & (cycles <= 52))
    fit_X = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
    score_X = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)]
    fit_y = np.clip(52 + 4 * frame.iloc[fit_rows]["impact_load"].to_numpy()
                    + rng.normal(scale=5, size=len(fit_rows)), 0, 100)
    score_y = np.clip(52 + 4 * frame.iloc[score_rows]["impact_load"].to_numpy()
                      + rng.normal(scale=5, size=len(score_rows)), 0, 100)
    seed = SEEDS[0]
    renamed_fit, fit_audit = n06.rename_features(fit_X, seed)
    renamed_score, score_audit = n06.rename_features(score_X, seed)
    baseline = np.linspace(35, 65, len(score_rows), dtype=np.float64)
    with threadpool_limits(limits=THREADS):
        output = n06.fit_predict_prepared(
            renamed_fit, fit_y, renamed_score, seed=seed,
            baseline_legacy=baseline, score_cycles=cycles[score_rows],
            fit_map_audit=fit_audit, score_map_audit=score_audit,
            original_fit_X=fit_X, original_score_X=score_X,
        )
    if output["model_state"].get("test_data_used") is not False:
        raise RuntimeError("N06 synthetic model state declares Test use")
    # The frozen output contract is exercised with Unicode IDs and immutable resume.
    payload = {
        "score_rows": score_rows.astype(np.int64),
        "score_record_ids": [f"測試-{i:04d}" for i in score_rows],
        "prediction": output["prediction"].astype(np.float64),
        "baseline_prediction": baseline,
        "raw_candidate": output["raw_candidate"].astype(np.float64),
        "diagnostics": {"model_state": output["model_state"], "mapping": output["fit_map_audit"],
                        "validation_labels_passed_to_fit": False, "windows": ("W1", "H12")},
    }
    identity = {"synthetic": True, "recipe": "N06 exact R06 synthetic smoke", "seed": seed}
    with tempfile.TemporaryDirectory(prefix="n06-synthetic-") as temporary:
        root = Path(temporary)
        failed_destination = root / "failed-publication"
        from unittest.mock import patch
        from src.models import research_next_contract as contract
        try:
            with patch.object(contract.os, "rename", side_effect=OSError("synthetic publication interruption")):
                publish_bundle(failed_destination, payload, identity, task="regression")
        except Exception:
            pass
        else:
            raise RuntimeError("N06 synthetic persistence failure injection did not fail")
        if failed_destination.exists() or not list(root.glob(".failed-publication.staging-*")):
            raise RuntimeError("N06 failed publication created a final partial bundle or lost staging evidence")
        destination = root / "bundle"
        saved = resume_or_publish_bundle(destination, payload, identity, task="regression")
        resumed = resume_or_publish_bundle(destination, payload, identity, task="regression")
        if saved.receipt != resumed.receipt:
            raise RuntimeError("N06 synthetic persistence resume receipt changed")
        arrays = resumed.arrays
        independent_rank = np.empty(len(arrays["raw_candidate"]), dtype=np.float64)
        for cycle in np.unique(cycles[score_rows]):
            mask = cycles[score_rows] == cycle
            independent_rank[mask] = np.clip(
                0.5 * arrays["raw_candidate"][mask]
                + 0.5 * 100.0 * rankdata(arrays["raw_candidate"][mask], method="average") / mask.sum(),
                0.0, 100.0,
            )
        rebuilt = np.clip(0.75 * baseline + 0.25 * independent_rank, 0.0, 100.0)
        if not np.allclose(rebuilt, arrays["prediction"], atol=1e-12, rtol=0):
            raise RuntimeError("N06 synthetic independent policy reconstruction failed")
    metrics = _metric(score_y, output["prediction"])
    extreme_raw = np.asarray([-25.0, 120.0, 52.0, 52.0], dtype=np.float64)
    extreme_cycles = np.ones(4, dtype=np.int64)
    extreme_legacy = np.asarray([0.0, 100.0, 30.0, 80.0], dtype=np.float64)
    sorted_extreme = np.sort(extreme_raw, kind="mergesort")
    rank_percentile = np.empty(4, dtype=np.float64)
    left = 0
    while left < len(sorted_extreme):
        right = left + 1
        while right < len(sorted_extreme) and sorted_extreme[right] == sorted_extreme[left]:
            right += 1
        rank_percentile[left:right] = 100.0 * (left + 1 + right) / (2.0 * len(sorted_extreme))
        left = right
    ordered_positions = np.argsort(extreme_raw, kind="mergesort")
    percentile_unsorted = np.empty(4, dtype=np.float64)
    percentile_unsorted[ordered_positions] = rank_percentile
    extreme_component = np.clip(0.5 * extreme_raw + 0.5 * percentile_unsorted, 0, 100)
    extreme_expected = np.clip(0.75 * extreme_legacy + 0.25 * extreme_component, 0, 100)
    extreme_actual = compose_r06_prediction(extreme_legacy, extreme_raw, extreme_cycles)
    clipping_probe_passed = (
        bool(np.any((extreme_raw < 0) | (extreme_raw > 100)))
        and np.allclose(extreme_actual, extreme_expected, atol=1e-12, rtol=0)
        and np.all((extreme_actual >= 0) & (extreme_actual <= 100))
    )
    if not clipping_probe_passed:
        raise RuntimeError("N06 synthetic raw-range/clipping policy check failed")
    groups = {}
    missing = _text_values(score_X["signal_stream"])[1].astype(bool)
    for name, mask in {
        "stream_missing": missing, "stream_present": ~missing,
        "target_le_5": score_y <= 5, "target_ge_95": score_y >= 95,
    }.items():
        groups[name] = {"n": int(mask.sum()), "supported_at_minimum_100": bool(mask.sum() >= 100),
                        "mae": _metric(score_y[mask], output["prediction"][mask])["mae"] if mask.any() else None}
    return {
        "status": "PASS", "official_predictor_fits": 0,
        "synthetic_predictor_fits": 1, "synthetic_fit_rows": int(len(fit_rows)),
        "synthetic_score_rows": int(len(score_rows)), "synthetic_mae": metrics,
        "unknown_category_tested": True, "all_missing_numeric_tested": True,
        "raw_out_of_range_and_r06_clip_policy_tested": True,
        "missing_and_repeated_stream_tested": True, "unicode_ids_tested": True,
        "fit_and_score_same_mapping_tested": bool(output["fit_map_audit"]["token_occurrences_renamed"] > 0
                                                   and output["score_map_audit"]["token_occurrences_renamed"] > 0),
        "outer_labels_not_passed_to_fit": True, "group_support_and_metrics": groups,
        "bundle_publish_readback_exact_resume": "PASS",
        "publication_failure_injection_and_recover_without_refit": "PASS",
        "independent_average_tie_rank50_and_blend_reconstruction": "PASS",
    }


def _metric(target: Any, prediction: Any) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("N06 metric requires aligned finite target/prediction vectors")
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(p - y))),
            "bias": float(np.mean(p - y))}


def _maybe_metric(target: Any, prediction: Any) -> dict[str, Any] | None:
    if len(target) == 0:
        return None
    return _metric(target, prediction)


def preflight() -> dict[str, Any]:
    smoke = _synthetic_smoke()
    first = prepare_context()
    second = prepare_context(
        available_bytes=max(MINIMUM_FREE_BYTES, first["execution_context"]["available_bytes"] - 1_234_567),
        ledger_context_override={"ledger_sha256": "0" * 64, "ledger": {"simulated": True}},
    )
    if first["freeze_signature"] != second["freeze_signature"]:
        raise RuntimeError("N06 stable identity depends on live RAM, timestamp, or ledger state")
    ready = all(item["status"] == "PASS" for item in first["resources"].values())
    report = {
        "status": "READY_TO_FREEZE" if ready else "DEFERRED_RESOURCE",
        "freeze_signature": first["freeze_signature"],
        "token_schema": {key: value for key, value in first["schema"].items()
                         if key != "unique_tokens_by_prefix"},
        "bijections": first["bijections"], "cache_lineage": first["cache_audit"],
        "resources": first["resources"], "synthetic_end_to_end": smoke,
        "stable_identity_repeat_check": "PASS", "official_predictor_fits_started": 0,
        "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0,
    }
    _write_once(PREFLIGHT_PATH, report)
    _write_once(REPORT_ROOT / "synthetic_smoke.json", smoke)
    return report


def freeze() -> dict[str, Any]:
    context = prepare_context()
    if any(item["status"] != "PASS" for item in context["resources"].values()):
        raise ResourceDeferred("N06 frozen resource preflight did not pass")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("N06 persisted preflight is required before freeze")
    preflight_report = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if (preflight_report.get("status") != "READY_TO_FREEZE"
            or preflight_report.get("freeze_signature") != context["freeze_signature"]):
        raise RuntimeError("N06 preflight report differs from the current stable identity")
    execution = dict(context["execution_context"])
    execution["preflight_sha256"] = file_sha256(PREFLIGHT_PATH)
    frozen = {"schema_version": 1, "identity": context["identity"],
              "freeze_signature": context["freeze_signature"], "execution_context": execution}
    _write_once(FREEZE_PATH, frozen)
    return {"status": "FROZEN", "freeze_signature": context["freeze_signature"],
            "path": _relative(FREEZE_PATH), "source_commit": execution["source_commit"]}


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("N06 freeze is required before official fits")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N06 freeze signature mismatch")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if frozen.get("execution_context", {}).get("preflight_sha256") != file_sha256(PREFLIGHT_PATH):
        raise RuntimeError("N06 preflight hash differs from frozen receipt")
    if preflight.get("status") != "READY_TO_FREEZE":
        raise RuntimeError("N06 persisted preflight is not READY_TO_FREEZE")
    context = prepare_context()
    if context["identity"] != frozen.get("identity"):
        raise RuntimeError("N06 frozen inputs, source, cache, recipe, or schema changed")
    source_commit = frozen.get("execution_context", {}).get("source_commit")
    if not source_commit or not _is_ancestor(source_commit):
        raise RuntimeError("N06 frozen source commit is not an ancestor of current HEAD")
    return frozen, context


def _fit_id(cutoff: int, seed: int) -> str:
    return f"N06:{STAGE}:{cutoff}:seed{seed}"


def _bundle_identity(signature: str, cutoff: int, seed: int) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "stage": STAGE,
            "freeze_signature": signature, "cutoff": int(cutoff), "seed": int(seed)}


def _bundle_path(cutoff: int, seed: int, *, recovery: bool = False) -> Path:
    root = RECOVERY_ROOT if recovery else BUNDLE_ROOT
    return root / f"fit{cutoff}" / f"seed{seed}"


def _resource_check(report: Mapping[str, Any]) -> dict[str, Any]:
    free = int(psutil.virtual_memory().available)
    estimate = int(report["estimated_peak_bytes"])
    allowed = min(JOB_CAP_BYTES, max(0, free - FREE_RESERVE_BYTES))
    receipt = {"available_physical_bytes": free, "estimated_peak_bytes": estimate,
               "allowed_bytes": allowed, "minimum_free_bytes": MINIMUM_FREE_BYTES,
               "reserve_bytes": FREE_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
               "threads": THREADS, "concurrent_jobs": 1,
               "frozen_shape_bound_status": report.get("status")}
    if report.get("status") != "PASS" or free < MINIMUM_FREE_BYTES or estimate > allowed:
        raise ResourceDeferred(f"N06 live resource check failed: {receipt}")
    return receipt


def _independent_bundle(bundle: Any, context: Mapping[str, Any], signature: str,
                        cutoff: int, seed: int) -> None:
    source = context["controls"][cutoff]
    arrays = bundle.arrays
    meta = bundle.receipt.get("metadata", {}).get("diagnostics", {})
    expected_rows = source["rows"]
    fit_rows = np.flatnonzero(context["frame"].cycle_code.to_numpy(dtype=np.int64) <= cutoff)
    raw = arrays.get("raw_candidate")
    if (bundle.receipt.get("task") != "regression"
            or bundle.receipt.get("identity") != _bundle_identity(signature, cutoff, seed)
            or not np.array_equal(arrays.get("score_rows"), expected_rows)
            or not np.array_equal(arrays.get("score_record_ids").astype(str), source["ids"].astype(str))
            or not np.array_equal(arrays.get("cycle_code"), source["cycles"])
            or not np.array_equal(arrays.get("baseline_prediction"), source["r22"])
            or not np.array_equal(arrays.get("legacy_prediction"), source["legacy"])
            or not np.array_equal(arrays.get("R06_original_prediction"), source["r06"])
            or raw is None or raw.shape != expected_rows.shape or not np.isfinite(raw).all()):
        raise RuntimeError(f"N06 {cutoff}/seed{seed} row/ID/control/raw alignment failed")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(source["cycles"]):
        mask = source["cycles"] == cycle
        order = np.argsort(raw[mask], kind="mergesort")
        sorted_values = raw[mask][order]
        percentile_ranks = np.empty(len(sorted_values), dtype=np.float64)
        start = 0
        while start < len(sorted_values):
            end = start + 1
            while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
                end += 1
            percentile_ranks[start:end] = 100.0 * (((start + 1) + end) / 2.0) / len(sorted_values)
            start = end
        unsorted = np.empty(len(order), dtype=np.float64)
        unsorted[order] = percentile_ranks
        mapped[mask] = np.clip(0.5 * raw[mask] + 0.5 * unsorted, 0, 100)
    expected = np.clip(0.75 * source["legacy"] + 0.25 * mapped, 0, 100)
    if not np.allclose(arrays["rank50_component"], mapped, atol=1e-12, rtol=0):
        raise RuntimeError(f"N06 {cutoff}/seed{seed} average-tie rank50 component differs")
    if not np.allclose(arrays["prediction"], expected, atol=1e-12, rtol=0):
        raise RuntimeError(f"N06 {cutoff}/seed{seed} fixed R06 .75/.25 policy differs")
    if (meta.get("cutoff") != cutoff or meta.get("seed") != seed
            or meta.get("fit_rows") != len(fit_rows)
            or meta.get("fit_row_indices_sha256") != _int64_sha256(fit_rows)
            or meta.get("fit_target_sha256") != _float64_sha256(context["target"][fit_rows])
            or meta.get("score_target_passed_to_fit") is not False
            or meta.get("test_data_used") is not False):
        raise RuntimeError(f"N06 {cutoff}/seed{seed} fit-population or leakage audit differs")
    state = meta.get("model_state", {})
    if (state.get("fit_rows") != len(fit_rows)
            or state.get("target_sha256") != _float64_sha256(context["target"][fit_rows])
            or state.get("validation_labels_passed_to_fit") is not False
            or state.get("test_data_used") is not False
            or state.get("recipe") != "char_wb TF-IDF ngrams 3-5 + structured/OHE + missing indicator; LightGBM regression_l1"):
        raise RuntimeError(f"N06 {cutoff}/seed{seed} R06 fitted-state audit differs")


def _int64_sha256(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _float64_sha256(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _fit_one(context: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int,
             seed: int) -> Any:
    signature = frozen["freeze_signature"]
    identity = _bundle_identity(signature, cutoff, seed)
    destination = _bundle_path(cutoff, seed)
    recovery = _bundle_path(cutoff, seed, recovery=True)
    fit_id = _fit_id(cutoff, seed)
    source = context["controls"][cutoff]
    if destination.is_dir():
        saved = read_bundle(destination, identity, task="regression")
        _independent_bundle(saved, context, signature, cutoff, seed)
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    if recovery.is_dir():
        recovered = read_bundle(recovery, identity, task="regression")
        _independent_bundle(recovered, context, signature, cutoff, seed)
        payload = {**recovered.arrays, **recovered.receipt["metadata"]}
        publish_bundle(destination, payload, identity, task="regression")
        saved = read_bundle(destination, identity, task="regression")
        _independent_bundle(saved, context, signature, cutoff, seed)
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    prior_started = [event for event in read_events(LEDGER_PATH)
                     if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    if prior_started:
        raise RuntimeError(f"N06 {fit_id} started with no durable bundle; refusing duplicate estimator fit")

    frame = context["frame"]
    cycles_all = frame.cycle_code.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles_all <= cutoff)
    score_rows = source["rows"].astype(np.int64)
    fit_X = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
    score_X = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)]
    renamed_fit, fit_map_audit = n06.rename_features(fit_X, seed)
    renamed_score, score_map_audit = n06.rename_features(score_X, seed)
    fit_target = context["target"][fit_rows].astype(np.float64)
    fit_ids = frame.record_id.to_numpy(dtype=str)[fit_rows].astype("U")
    resource = _resource_check(context["resources"][f"{cutoff}:{seed}"])
    start_fit(LEDGER_PATH, card_id=CARD_ID, stage=STAGE, fit_id=fit_id,
              cutoff=cutoff, component=f"recoded_R06_seed_{seed}")
    recovery_saved = False
    try:
        with threadpool_limits(limits=THREADS):
            output = n06.fit_predict_prepared(
                renamed_fit, fit_target, renamed_score, seed=seed,
                baseline_legacy=source["legacy"], score_cycles=source["cycles"],
                fit_map_audit=fit_map_audit, score_map_audit=score_map_audit,
                original_fit_X=fit_X, original_score_X=score_X,
            )
        arrays = {
            "score_rows": score_rows.astype(np.int64),
            "score_record_ids": source["ids"].astype("U"),
            "prediction": output["prediction"].astype(np.float64),
            "baseline_prediction": source["r22"].astype(np.float64),
            "raw_candidate": output["raw_candidate"].astype(np.float64),
            "legacy_prediction": source["legacy"].astype(np.float64),
            "R06_original_prediction": source["r06"].astype(np.float64),
            "cycle_code": source["cycles"].astype(np.int64),
        }
        # Store the cycle-complete component explicitly for independent audit.
        arrays["rank50_component"] = np.empty(len(score_rows), dtype=np.float64)
        for cycle in np.unique(source["cycles"]):
            mask = source["cycles"] == cycle
            arrays["rank50_component"][mask] = np.clip(
                0.5 * arrays["raw_candidate"][mask]
                + 0.5 * 100.0 * rankdata(arrays["raw_candidate"][mask], method="average") / int(mask.sum()),
                0.0, 100.0,
            )
        rebuilt_policy = np.clip(0.75 * source["legacy"] + 0.25 * arrays["rank50_component"], 0, 100)
        if not np.allclose(rebuilt_policy, arrays["prediction"], atol=1e-12, rtol=0):
            raise RuntimeError("N06 fixed R06 policy differs immediately after fit")
        metadata = {
            "diagnostics": {
                "card_id": CARD_ID, "stage": STAGE, "cutoff": cutoff, "seed": seed,
                "model_state": output["model_state"],
                "prediction_diagnostics": output["prediction_diagnostics"],
                "fit_rows": int(len(fit_rows)), "fit_cycle_range": [47, cutoff],
                "fit_row_indices_sha256": _int64_sha256(fit_rows),
                "fit_record_ids_sha256": hashlib.sha256(b"".join(
                    len(v.encode("utf-8")).to_bytes(8, "big") + v.encode("utf-8") for v in fit_ids
                )).hexdigest(),
                "fit_target_sha256": _float64_sha256(fit_target),
                "fit_map_audit": output["fit_map_audit"], "score_map_audit": output["score_map_audit"],
                "mapped_documents_sha256": output["mapped_documents_sha256"],
                "original_documents_sha256": output["original_documents_sha256"],
                "score_target_passed_to_fit": False, "outer_score_labels_read_for_fit": False,
                "test_data_used": False, "sample_submission_used": False,
                "live_resource_check": resource, "threads": THREADS, "concurrent_jobs": 1,
                "rank_mapping": "within complete score cycle, average ties, .5 raw + .5 percentile",
                "blend_policy": ".75 protected legacy + .25 rank50(raw candidate), then clip [0,100]",
                "recipe_fingerprint": output["model_state"].get("state_sha256"),
            }
        }
        payload = {**arrays, **metadata}
        try:
            publish_bundle(destination, payload, identity, task="regression")
        except Exception:
            # Keep the in-memory estimator output durable and resumable without another fit.
            publish_bundle(recovery, payload, identity, task="regression")
            recovery_saved = True
            recovered = read_bundle(recovery, identity, task="regression")
            _independent_bundle(recovered, context, signature, cutoff, seed)
            raise RuntimeError(f"N06 {fit_id} final publication failed; recovery bundle is saved; stop before next fit")
        saved = read_bundle(destination, identity, task="regression")
        _independent_bundle(saved, context, signature, cutoff, seed)
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    except Exception as exc:
        events = read_events(LEDGER_PATH)
        terminal = [event for event in events if event.get("fit_id") == fit_id
                    and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not terminal and not recovery_saved and not destination.is_dir():
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise


def _fit_records(context: Mapping[str, Any], frozen: Mapping[str, Any],
                 bundles: Mapping[int, Mapping[int, Any]]) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    records: dict[int, dict[int, dict[str, np.ndarray]]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for cutoff in ORIGINS:
            arrays = bundles[seed][cutoff].arrays
            records[seed][cutoff] = {
                "rows": arrays["score_rows"].astype(np.int64),
                "cycles": arrays["cycle_code"].astype(np.int64),
                "prediction": arrays["prediction"].astype(np.float64),
                "r22": arrays["baseline_prediction"].astype(np.float64),
                "legacy": arrays["legacy_prediction"].astype(np.float64),
                "r06": arrays["R06_original_prediction"].astype(np.float64),
            }
    return records


def _views(context: Mapping[str, Any], records: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
           seed: int) -> dict[str, Any]:
    frame = context["frame"]
    target = context["target"]
    streams_missing = _text_values(frame.signal_stream)[1].astype(bool)
    definitions: dict[str, list[tuple[int, int, int]]] = {
        "primary_W1_W4": [(55, 56, 59), (67, 68, 71)],
        "H12_pooled": [(55, 56, 67), (67, 68, 79)],
        "H12_cutoff55": [(55, 56, 67)], "H12_cutoff67": [(67, 68, 79)],
        "W1": [(55, 56, 59)], "W2": [(55, 60, 63)], "W3": [(55, 64, 67)],
        "W4": [(67, 68, 71)], "W5": [(67, 72, 75)], "W6": [(67, 76, 79)],
    }
    output: dict[str, Any] = {}
    for name, pieces in definitions.items():
        rows = np.concatenate([
            records[origin]["rows"][(records[origin]["cycles"] >= lo)
                                     & (records[origin]["cycles"] <= hi)]
            for origin, lo, hi in pieces
        ])
        if len(rows) != len(np.unique(rows)):
            raise RuntimeError(f"N06 {name} row population overlaps")
        y = target[rows]
        candidate = np.concatenate([
            records[origin]["prediction"][(records[origin]["cycles"] >= lo)
                                           & (records[origin]["cycles"] <= hi)]
            for origin, lo, hi in pieces
        ])
        controls = {}
        for key in ("r22", "legacy", "r06"):
            controls[key] = np.concatenate([
                records[origin][key][(records[origin]["cycles"] >= lo)
                                     & (records[origin]["cycles"] <= hi)]
                for origin, lo, hi in pieces
            ])
        if not (len(rows) == len(y) == len(candidate)
                and all(len(value) == len(rows) for value in controls.values())):
            raise RuntimeError(f"N06 {name} arrays do not align")
        metrics = {"candidate": _metric(y, candidate)}
        for key, prediction in controls.items():
            metrics[key] = _metric(y, prediction)
            metrics[f"gain_vs_{key}"] = float(metrics[key]["mae"] - metrics["candidate"]["mae"])
        groups: dict[str, Any] = {}
        group_masks = {
            "signal_stream_missing": streams_missing[rows],
            "signal_stream_present": ~streams_missing[rows],
            "observed_threat_rank_le_5_evaluation_only": y <= 5,
            "observed_threat_rank_ge_95_evaluation_only": y >= 95,
        }
        for group, mask in group_masks.items():
            group_y = y[mask]
            group_candidate = candidate[mask]
            group_result = {"n": int(mask.sum()), "supported_minimum_100": bool(mask.sum() >= 100),
                            "candidate": _maybe_metric(group_y, group_candidate)}
            for key, prediction in controls.items():
                group_result[key] = _maybe_metric(group_y, prediction[mask])
                group_result[f"gain_vs_{key}"] = (
                    float(group_result[key]["mae"] - group_result["candidate"]["mae"])
                    if group_result[key] is not None and group_result["candidate"] is not None else None
                )
            groups[group] = group_result
        per_cycle = {}
        view_cycles = frame.cycle_code.to_numpy(dtype=np.int64)[rows]
        for cycle in np.unique(view_cycles):
            mask = view_cycles == cycle
            per_cycle[str(int(cycle))] = {
                "candidate": _metric(y[mask], candidate[mask]),
                **{key: _metric(y[mask], prediction[mask]) for key, prediction in controls.items()},
                **{f"gain_vs_{key}": float(_metric(y[mask], prediction[mask])["mae"]
                                            - _metric(y[mask], candidate[mask])["mae"])
                   for key, prediction in controls.items()},
            }
        output[name] = {"rows": int(len(rows)), "metrics": metrics, "groups": groups,
                        "per_cycle": per_cycle}
    return {"seed": int(seed), "views": output}


def _make_plot(result: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"N06 refuses to overwrite immutable plot: {_relative(path)}")
    names = [f"W{i}" for i in range(1, 7)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for axis, seed in zip(axes, SEEDS):
        record = result["seeds"][str(seed)]["views"]
        positions = np.arange(6)
        for idx, (reference, label, color) in enumerate((
                ("r22", "R22_ORIGINAL", "#36628e"),
                ("r06", "R06_ORIGINAL", "#b36830"),
                ("legacy", "Protected legacy", "#69736e"))):
            gains = [record[f"W{i}"]["metrics"][f"gain_vs_{reference}"] for i in range(1, 7)]
            axis.bar(positions + (idx - 1) * 0.24, gains, width=0.23, label=label, color=color)
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_xticks(positions, names)
        axis.set_title(f"Mapping seed {seed}")
        axis.set_xlabel("Temporal window")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("MAE gain (positive = lower MAE)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("N06 — recoded R06 versus fixed references by window")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _independent_verify(context: Mapping[str, Any], frozen: Mapping[str, Any],
                        bundles: Mapping[int, Mapping[int, Any]], result: Mapping[str, Any]) -> dict[str, Any]:
    # Independent row-level reconstruction using raw Train targets, cache controls,
    # stable mergesort ranks, and a separate direct MAE implementation.
    checks: dict[str, bool] = {}
    for seed in SEEDS:
        for cutoff in ORIGINS:
            bundle = bundles[seed][cutoff]
            _independent_bundle(bundle, context, frozen["freeze_signature"], cutoff, seed)
    for seed in SEEDS:
        record = result["seeds"][str(seed)]["views"]
        for view_name, pieces in {
            "primary_W1_W4": [(55, 56, 59), (67, 68, 71)],
            "H12_pooled": [(55, 56, 67), (67, 68, 79)],
            "H12_cutoff55": [(55, 56, 67)], "H12_cutoff67": [(67, 68, 79)],
            **{f"W{i+1}": [(55, 56 + 4 * i, 59 + 4 * i)] for i in range(3)},
            "W4": [(67, 68, 71)], "W5": [(67, 72, 75)], "W6": [(67, 76, 79)],
        }.items():
            row_parts = []
            pred_parts = []
            ref_parts = {key: [] for key in ("r22", "legacy", "r06")}
            for cutoff, low, high in pieces:
                arrays = bundles[seed][cutoff].arrays
                mask = (arrays["cycle_code"] >= low) & (arrays["cycle_code"] <= high)
                row_parts.append(arrays["score_rows"][mask].astype(np.int64))
                pred_parts.append(arrays["prediction"][mask].astype(np.float64))
                ref_parts["r22"].append(arrays["baseline_prediction"][mask].astype(np.float64))
                ref_parts["legacy"].append(arrays["legacy_prediction"][mask].astype(np.float64))
                ref_parts["r06"].append(arrays["R06_original_prediction"][mask].astype(np.float64))
            rows = np.concatenate(row_parts)
            candidate = np.concatenate(pred_parts)
            if len(np.unique(rows)) != len(rows):
                raise RuntimeError(f"N06 independent {view_name} rows overlap")
            y = context["target"][rows]
            got = record[view_name]
            checks[f"{seed}:{view_name}:row_count"] = int(got["rows"]) == len(rows)
            checks[f"{seed}:{view_name}:candidate_mae"] = abs(
                float(got["metrics"]["candidate"]["mae"]) - float(np.mean(np.abs(candidate - y)))
            ) <= 1e-12
            for key in ref_parts:
                reference = np.concatenate(ref_parts[key])
                actual = float(np.mean(np.abs(reference - y)))
                checks[f"{seed}:{view_name}:{key}_mae"] = abs(
                    float(got["metrics"][key]["mae"]) - actual
                ) <= 1e-12
                gain = actual - float(np.mean(np.abs(candidate - y)))
                checks[f"{seed}:{view_name}:gain_vs_{key}"] = abs(
                    float(got["metrics"][f"gain_vs_{key}"]) - gain
                ) <= 1e-12
    events = read_events(LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    saves = [event for event in events if event.get("event") == "FIT_SAVED"
             and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"
                and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    expected = {_fit_id(c, s) for c in ORIGINS for s in SEEDS}
    checks["ledger_exact_four_started"] = {event.get("fit_id") for event in starts} == expected and len(starts) == 4
    checks["ledger_exact_four_saved_no_failed"] = {event.get("fit_id") for event in saves} == expected and len(saves) == 4 and not failures
    checks["no_test_or_submission_access"] = (
        result["safety"].get("test_data_used") is False
        and result["safety"].get("sample_submission_used") is False
        and result["safety"].get("kaggle_calls") == 0
    )
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL", "checks": checks,
        "failed_checks": failed, "verified_fit_count": len(saves),
        "verified_fit_ids": sorted(expected),
        "bundle_receipts": {str(seed): {str(cutoff): bundles[seed][cutoff].receipt["receipt_sha256"]
                                        for cutoff in ORIGINS} for seed in SEEDS},
        "budget": ledger_summary(LEDGER_PATH),
    }


def run_pilot() -> dict[str, Any]:
    frozen, context = _verify_freeze()
    # Ensure exact card reserve can use available slots; reserve_card is idempotent.
    reserve_card(LEDGER_PATH, card_id=CARD_ID, stage=STAGE, fits=PILOT_FITS, category="pilot")
    bundles: dict[int, dict[int, Any]] = {seed: {} for seed in SEEDS}
    for cutoff in ORIGINS:
        for seed in SEEDS:
            bundles[seed][cutoff] = _fit_one(context, frozen, cutoff, seed)
    release_unused(LEDGER_PATH, card_id=CARD_ID, stage=STAGE)
    seeds_report = {str(seed): _views(context, _fit_records(context, frozen, bundles)[seed], seed)
                    for seed in SEEDS}
    result = {
        "status": "COMPLETE_INFORMATION", "card_id": CARD_ID,
        "question": "Does R06 depend on anonymous character surface beyond recurring token identity?",
        "interpretation": "Information-only diagnostic. The two mapping seeds are descriptive replications, not a statistical sample or promotion test.",
        "freeze_signature": frozen["freeze_signature"],
        "recipe": frozen["identity"]["recipe"], "preconditions": frozen["identity"]["preconditions"],
        "cache_lineage": frozen["identity"]["cache_lineage"], "seeds": seeds_report,
        "fit_receipts": {str(seed): {str(cutoff): bundles[seed][cutoff].receipt["receipt_sha256"]
                                     for cutoff in ORIGINS} for seed in SEEDS},
        "fit_count": 4, "official_predictor_fits": 4,
        "continuation_eligible": False, "automatic_promotion": False,
        "safety": {"train_hash": frozen["identity"]["official_train_sha256"],
                   "test_data_used": False, "sample_submission_used": False,
                   "kaggle_calls": 0, "outer_score_labels_passed_to_fit": False},
        "budget": ledger_summary(LEDGER_PATH),
    }
    _write_once(REPORT_ROOT / "results.json", result)
    _make_plot(result, REPORT_ROOT / "window_gains.png")
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.optimization_v3_research_next_n06_verify"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    verification = json.loads(completed.stdout)
    if verification["status"] != "PASS":
        raise RuntimeError(f"N06 independent verification failed: {verification['failed_checks']}")
    summary = {
        "status": result["status"], "question": result["question"],
        "fit_count": result["fit_count"], "verification": verification["status"],
        "fit_receipts": result["fit_receipts"],
        "seed_primary_mae": {seed: values["views"]["primary_W1_W4"]["metrics"]
                             for seed, values in seeds_report.items()},
        "seed_H12_mae": {seed: values["views"]["H12_pooled"]["metrics"]
                          for seed, values in seeds_report.items()},
        "continuation_eligible": False, "automatic_promotion": False,
        "budget": ledger_summary(LEDGER_PATH),
    }
    _write_once(REPORT_ROOT / "SUMMARY.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run-pilot"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        output = preflight()
    elif args.command == "freeze":
        output = freeze()
    else:
        output = run_pilot()
    print(json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
