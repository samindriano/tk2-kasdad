"""Create immutable C19/A02 protocols and input/source freezes before official fits."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_a02_neighbors import (  # noqa: E402
    CORRECTION_LIMIT, NEIGHBORS_MAX, QUERY_BLOCK_ROWS, SHRINKAGE, SHUFFLE_SEED,
    TFIDF_PARAMETERS,
)
from src.models.optimization_v3_batch1_c19 import (  # noqa: E402
    BLEND_WEIGHT, FEATURE_NAMES, SVD_COMPONENTS, SVD_PARAMETERS, TFIDF_PARAMETERS as C19_TFIDF,
    THRESHOLD, XGB_PARAMETERS,
)
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from src.models.research_next_budget import read_events  # noqa: E402
from src.models.optimization_v3_budget_amendment import AMENDMENT_ID  # noqa: E402
from scripts.optimization_v3_a02_neighbors import (  # noqa: E402
    CACHE_PATHS as A02_CACHE_PATHS, EXPECTED_TRAIN_SHA256 as A02_TRAIN_SHA256,
    REPORT_ROOT as A02_REPORT_ROOT, SYNTHETIC_QA_PATH,
)


C19_ROOT = ROOT / "experiments/optimization_v3/c19_a02_v1/C19"
A02_ROOT = ROOT / "experiments/optimization_v3/c19_a02_v1/A02"
C19_REPORT = ROOT / "reports/optimization_v3/c19_benchmark_v1"
C19_TRAIN = ROOT / "data/raw/classification/Train.csv"
C19_TEST = ROOT / "data/raw/classification/Test.csv"
C19_SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
BASE_CACHE = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"
BASE_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
BASE_CSV = ROOT / "submissions/dual_campaign/optimization_v3/v3_cls_sparse25.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"

SOURCE_PATHS = [
    "src/models/classification_a02_neighbors.py",
    "src/models/classification_c19_fulltrain.py",
    "src/models/optimization_v3_batch1_c19.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/regression_research_text.py",
    "src/models/round1_models.py",
    "src/models/research_next_budget.py",
    "src/models/research_next_contract.py",
    "src/models/optimization_v3_budget_amendment.py",
    "scripts/optimization_v3_c19_benchmark.py",
    "scripts/optimization_v3_c19_benchmark_verify.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "scripts/optimization_v3_a02_synthetic.py",
    "scripts/optimization_v3_c19_a02_freeze.py",
    "tests/test_optimization_v3_a02_core.py",
    "tests/test_optimization_v3_c19_a02_prep.py",
    "tests/test_optimization_v3_budget_amendment.py",
]


def _sha(path: Path) -> str:
    return file_sha256(path)


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Frozen protocol already exists and differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "threadpoolctl", "joblib"]
    return {name: importlib.metadata.version(name) for name in names}


def _freeze(task: str, protocol: dict[str, Any], source_commit: str,
            input_paths: dict[str, Path], extra_qa: dict[str, str] | None = None) -> dict[str, Any]:
    source_hashes = {relative: _sha(ROOT / relative) for relative in SOURCE_PATHS}
    input_hashes = {relative: _sha(path) for relative, path in input_paths.items()}
    amendment_events = [event for event in read_events(LEDGER)
                        if event.get("amendment_id") == AMENDMENT_ID]
    if len(amendment_events) != 1:
        raise RuntimeError("The C19/A02 budget amendment must be recorded before freezing")
    event = amendment_events[0]
    amendment_sha = canonical_sha256(event)
    qa_hashes = extra_qa or {}
    identity = {
        "task": task,
        "campaign": "C19_FULLTRAIN_AND_A02_FOUR_FIT_PILOT",
        "source_commit": source_commit,
        "protocol_sha256": canonical_sha256(protocol),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "budget_amendment_id": AMENDMENT_ID,
        "budget_amendment_event_sha256": amendment_sha,
        "synthetic_qa_sha256": qa_hashes,
        "packages": _package_versions(),
    }
    return {
        "schema_version": 1,
        "task": task,
        "protocol_sha256": canonical_sha256(protocol),
        "freeze_signature": canonical_sha256(identity),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "budget_amendment_id": AMENDMENT_ID,
        "budget_amendment_event_sha256": amendment_sha,
        "identity": identity,
    }


def prepare() -> dict[str, Any]:
    head = _git_head()
    if len(head) != 40:
        raise RuntimeError("Could not resolve exact preparation source commit")
    for relative in SOURCE_PATHS:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"Required source not yet present: {relative}")
    if not SYNTHETIC_QA_PATH.is_file():
        raise RuntimeError("A02 synthetic end-to-end/persistence/resume QA must pass before freeze")
    synthetic = json.loads(SYNTHETIC_QA_PATH.read_text(encoding="utf-8"))
    if synthetic.get("status") != "PASS" or synthetic.get("fit_type") != "SYNTHETIC_ONLY":
        raise RuntimeError("A02 synthetic QA receipt does not pass")

    c19_recipe_path = ROOT / "experiments/optimization_v3/batch1/C19/recipe.json"
    c19_recipe = json.loads(c19_recipe_path.read_text(encoding="utf-8"))
    c19_protocol = {
        "schema_version": 1,
        "candidate_id": "C19_XGB_SVD32_FULLTRAIN_BENCHMARK",
        "task": "classification",
        "original_gate_verdict_preserved": "RETIRE / prior frozen gate failed; benchmark exception only",
        "feature_allowlist": list(FEATURE_NAMES),
        "official_train": {"rows": 50_608, "cycles": [47, 79],
                           "sha256": A02_TRAIN_SHA256},
        "fit_recipe_source": "src.models.optimization_v3_batch1_c19.XGBoostSVD32Classifier",
        "historical_recipe_sha256": _sha(c19_recipe_path),
        "historical_recipe": c19_recipe,
        "tfidf_parameters": {key: list(value) if isinstance(value, tuple) else value.__name__ if isinstance(value, type) else value
                             for key, value in C19_TFIDF.items()},
        "svd": {"components": SVD_COMPONENTS, "parameters": SVD_PARAMETERS},
        "xgb_parameters": XGB_PARAMETERS,
        "fit_population": "all official Classification Train rows cycles 47-79; no downsampling/weights; preprocessing fit only on Train",
        "prediction_route": {
            "present": "0.75 * frozen full-Train sparse25 probability + 0.25 * C19 raw probability",
            "missing": "exact frozen full-Train sparse25 CB_MISSING probability",
            "threshold": THRESHOLD,
        },
        "one_fulltrain_predictor_fit": 1,
        "test_boundary": "record_id + features only for inference/ID QA; never load target column",
        "sample_submission_boundary": "schema/ID QA only; never fit or predict",
        "resource_policy": {"threads": 2, "job_cap_bytes": 6 * 1024**3,
                            "minimum_free_ram_bytes": 4 * 1024**3,
                            "one_training_job_at_a_time": True},
        "submission": {
            "competition": "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027",
            "message": "Samuel C19 XGB SVD32 Frozen Benchmark",
            "maximum_new_uploads": 1,
        },
        "scope_boundary": "No champion replacement, no historical verdict change, no further fit from public score.",
    }
    c19_inputs = {
        "data/raw/classification/Train.csv": C19_TRAIN,
        "data/raw/classification/Test.csv": C19_TEST,
        "data/raw/classification/SampleSubmission.csv": C19_SAMPLE,
        "experiments/optimization_v3/batch1/C19/recipe.json": c19_recipe_path,
        "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz": BASE_CACHE,
        "reports/optimization_v3/kaggle/v3_cls_sparse25.json": BASE_MANIFEST,
        "submissions/dual_campaign/optimization_v3/v3_cls_sparse25.csv": BASE_CSV,
    }
    c19_freeze = _freeze("C19", c19_protocol, head, c19_inputs)
    _write_once(C19_ROOT / "protocol.json", c19_protocol)
    _write_once(C19_ROOT / "freeze.json", c19_freeze)

    a02_protocol = {
        "schema_version": 1,
        "candidate_id": "A02_LEXICAL_NEIGHBOR_RESIDUAL",
        "task": "classification",
        "question": "Within past-only residual banks, do text-neighbor residuals predict later Classification errors better than cycle × OOF-probability-decile shuffled residuals?",
        "origins": [67, 75],
        "fit_ids": {"E67_observed": FIT_IDS_LOCAL(67, "observed"),
                    "E67_shuffled": FIT_IDS_LOCAL(67, "shuffled"),
                    "E75_observed": FIT_IDS_LOCAL(75, "observed"),
                    "E75_shuffled": FIT_IDS_LOCAL(75, "shuffled")},
        "task_fit_count": 4,
        "baseline": "verified frozen sparse25 probability from C16 cache baseline_prediction; never C16 prediction or raw_candidate",
        "bank": {
            "E67": {"source_blocks": [[55, 56, 59], [59, 60, 63], [63, 64, 67]],
                    "rows": 17_685, "present_rows": 16_482},
            "E75": {"source_blocks": [[55, 56, 59], [59, 60, 63], [63, 64, 67], [67, 68, 71], [71, 72, 75]],
                    "rows": 30_097, "present_rows": 28_045},
            "residual": "artifact_signal - past-only baseline_prediction; present-stream bank only",
            "record_id_and_row_join": "exact official Train row positions plus record_id equality",
        },
        "text_space": {
            "recipe_source": "src.models.classification_models._TFIDF_PARAMS and _text_values",
            "fit_population": "all official Classification Train feature texts with cycle_code <= origin, including empty-stream rows as empty documents",
            "similarity": "L2-normalized word-TFIDF cosine only; no structured feature in similarity",
            "query_block_rows": QUERY_BLOCK_ROWS,
            "top_positive_neighbors": NEIGHBORS_MAX,
            "tie_break": "similarity descending then official Train row position ascending",
        },
        "residual_predictor": {
            "observed_arm": "fit with observed past-only OOF residual assignments",
            "matched_control": "shuffle residual values within cycle_code × min(9,floor(10*OOF probability)), stable row order, default_rng(42) reset once per origin",
            "weight": "positive cosine similarity squared",
            "shrinkage": SHRINKAGE,
            "correction_clip": CORRECTION_LIMIT,
            "present_prediction": "clip(baseline_probability + correction, 0, 1)",
            "missing_prediction": "exact baseline_probability",
            "threshold": 0.5,
        },
        "views": {
            "W4": {"origin": 67, "cycles": [68, 71], "rows": 4_950},
            "W6": {"origin": 75, "cycles": [76, 79], "rows": 8_491},
            "primary": {"components": ["W4", "W6"], "disjoint_rows": 13_441},
            "H12_L": {"origin": 67, "cycles": [68, 79], "rows": 20_903},
        },
        "scientific_gate": {
            "H12_L_gain_vs_sparse25_at_least": 0.0005,
            "H12_L_gain_vs_shuffled_at_least": 0.0002,
            "primary_gain_vs_sparse25_at_least": 0.0,
            "primary_gain_vs_shuffled_at_least": 0.0,
            "W4_and_W6_max_harm_vs_sparse25": 0.003,
            "H12_L_max_harm_vs_sparse25": 0.001,
            "integrity_and_source_boundaries": "all pass",
        },
        "subgroups": {
            "fields": ["stream_availability", "entry_friction", "fit_prefix_defined_text_load_quartile"],
            "views": ["primary", "W4", "W6", "H12_L"],
            "minimum_support": {"n": 200, "each_class": 30},
            "risk_review_only_thresholds": {"macro_f1_harm_gt": 0.03, "positive_recall_harm_gt": 0.05},
            "historical_reference_thresholds_to_report": {"macro_f1_harm": [0.01, 0.02, 0.03],
                                                            "positive_recall_harm": [0.02, 0.03, 0.05]},
        },
        "sources_boundary": {
            "only_official_classification_train_and_verified_c16_oof_cache": True,
            "test_or_sample_submission_access": False,
            "test_labels_used": False,
            "no_public_score_or_test_feedback_in_recipe": True,
        },
        "resource_policy": {"threads": 2, "job_cap_bytes": 6 * 1024**3,
                            "minimum_free_ram_bytes": 4 * 1024**3,
                            "one_training_job_at_a_time": True},
        "no_retry_no_continuation_no_submission": True,
    }
    a02_inputs = {"data/raw/classification/Train.csv": C19_TRAIN}
    for cutoff, path in A02_CACHE_PATHS.items():
        a02_inputs[f"{path.relative_to(ROOT).as_posix()}"] = path
        a02_inputs[f"{path.with_suffix('.json').relative_to(ROOT).as_posix()}"] = path.with_suffix(".json")
    a02_freeze = _freeze("A02", a02_protocol, head, a02_inputs,
                         extra_qa={"reports/optimization_v3/a02_neighbors_v1/synthetic_qa.json": _sha(SYNTHETIC_QA_PATH)})
    _write_once(A02_ROOT / "protocol.json", a02_protocol)
    _write_once(A02_ROOT / "freeze.json", a02_freeze)
    return {
        "status": "C19_A02_FROZEN_BEFORE_OFFICIAL_FITS",
        "source_commit": head,
        "C19_freeze_signature": c19_freeze["freeze_signature"],
        "A02_freeze_signature": a02_freeze["freeze_signature"],
        "C19_protocol_sha256": canonical_sha256(c19_protocol),
        "A02_protocol_sha256": canonical_sha256(a02_protocol),
    }


def FIT_IDS_LOCAL(origin: int, arm: str) -> str:
    return f"A02_E{origin}_{arm.upper()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="write immutable protocols/freeze after preparation commit and budget amendment")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("explicit --prepare is required")
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
