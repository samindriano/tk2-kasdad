"""C06 bridge for cycle-ordered category novelty and frozen controls."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_sparse_classification import CACHE as SPARSE_PILOT_CACHE
from scripts.optimization_v3_sparse_classification import CONFIG as SPARSE_PILOT_CONFIG
from scripts.optimization_v3_sparse_classification import check as check_sparse_pilot
from scripts.optimization_v3_sparse_classification import load_train
from scripts.optimization_v3_sparse_classification_fullsix import (
    CB_CACHE, PILOT_REPORT, control_record, pilot_record,
)
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
from src.models.optimization_v3_batch1_c06 import (
    BLEND_WEIGHT,
    FEATURE_NAMES,
    CATEGORY_COLUMNS,
    NOVELTY_FEATURE_NAMES,
    MINIMUM_CLASS_SUPPORT,
    THRESHOLD,
    FitChronologicalCategoryNovelty,
    CategoryNoveltySparseClassifier,
    SPARSE_PARAMS,
    fit_predict,
)
from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.classification_round2_models import stream_missing_mask
from src.models.regression_research_text import _array_digest
from src.models.round1_models import _fit_fingerprint, _is_missing
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C06.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C06"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/C06/freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C06"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C06"

EXPECTED_QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
EXPECTED_CARD_SHA256 = "9b2b16a4431950a2532b8669e5712d70e96203eb1d518be960ff7ee0cb1268dc"
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_DIAGNOSTIC_EVIDENCE_SHA256 = "c26218e0e90715a30b7b92584ddbf6c6d0eac2b3c71ecb5d4c2473c42fe4dc67"
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
ACCEPTED_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
DIAGNOSTIC_EVIDENCE = ROOT / "reports/optimization_v3/classification_diagnostics/audit.json"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]

EXPECTED_CARD_CHANGE = (
    "For org_code/module_code/pattern_code, encode a training row as novel only if category absent from strictly earlier cycles. "
    "After encoding entire cycle update feature-only vocabulary. Outer rows use all fit categories. "
    "Missing gets a separate indicator, not novel. Append flags to current sparse model."
)
EXPECTED_POLICY = ".75 frozen C2_B +.25 new sparse expert on present; frozen CB missing; threshold.50"


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required C06 input is missing: {label} ({path})")
    actual = file_sha256(path)
    if actual != expected:
        raise BatchIntegrityError(f"Frozen C06 {label} hash differs: {actual}")
    return actual


def _assert_card_identity(card: dict[str, Any], queue: dict[str, Any]) -> None:
    expected = {
        "id": "C06",
        "task": "classification",
        "name": "Learnable ordered category novelty",
        "family": "category_novelty",
        "change": EXPECTED_CARD_CHANGE,
        "policy": "C_SPARSE_REPLACE",
        "pilot_predictor_fits": 2,
        "evaluation": "STANDARD",
        "condition": "No additional dependency",
        "status": "PLANNED_NOT_RUN",
        "implementation": "NOT_IMPLEMENTED",
    }
    for key, value in expected.items():
        if card.get(key) != value:
            raise BatchIntegrityError(f"C06 queue identity mismatch for {key}: {card.get(key)!r}")
    if card.get("card_path") != "reports/optimization_v3/research_backlog/cards/C06.md":
        raise BatchIntegrityError("C06 queue card path changed")
    if queue.get("policies", {}).get("C_SPARSE_REPLACE") != EXPECTED_POLICY:
        raise BatchIntegrityError("C06 C_SPARSE_REPLACE policy differs from the frozen card")
    if queue["execution_preset"]["concurrent_fit_jobs"] != 1 or queue["execution_preset"]["threads_per_job"] != 2:
        raise BatchIntegrityError("C06 requires one fit job and two threads")
    if queue["evaluation"]["STANDARD"]["pilot_cutoffs"] != [55, 67]:
        raise BatchIntegrityError("C06 STANDARD pilot cutoffs changed")
    gate = queue["evaluation"]["classification_gate"]
    exact_gate = {
        "pilot_primary_gain_min": 0.0005,
        "pilot_H12_harm_max": 0.001,
        "individual_pilot_view_harm_max": 0.005,
        "full_primary_gain_min": 0.0005,
        "full_H12_harm_max": 0.001,
        "full_nonnegative_windows_min": 4,
        "supported_group_n_min": 200,
        "supported_group_F1_harm_max": 0.01,
        "positive_class_recall_harm_max": 0.02,
        "positive_and_negative_per_group_min": 30,
    }
    if gate != exact_gate:
        raise BatchIntegrityError("C06 classification gates changed from their frozen values")
    baseline = queue["baseline"]["classification"]
    if (baseline["manifest"] != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or baseline["manifest_sha256"] != EXPECTED_MANIFEST_SHA256
            or baseline["Train_sha256"] != EXPECTED_TRAIN_SHA256):
        raise BatchIntegrityError("C06 accepted control manifest or Train identity changed")


def _validate_manifest(queue: dict[str, Any]) -> dict[str, Any]:
    if ROOT / queue["baseline"]["classification"]["manifest"] != ACCEPTED_MANIFEST:
        raise BatchIntegrityError("C06 queue points to a different accepted control manifest")
    _require_hash(ACCEPTED_MANIFEST, EXPECTED_MANIFEST_SHA256, "accepted V3 classification manifest")
    manifest = json.loads(ACCEPTED_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("task") != "classification"
            or manifest.get("candidate_id") != "v3_cls_sparse25"
            or manifest.get("train_rows") != 50_608
            or manifest.get("input_sha256", {}).get("train") != EXPECTED_TRAIN_SHA256
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("test_labels_used") is not False):
        raise BatchIntegrityError("Accepted V3 sparse25 manifest identity or provenance changed")
    expected_recipe = {
        "kind": "availability_present_blend",
        "present": [
            {"model": "C2_B", "weight": 0.75},
            {"model": "SPARSE_BINARY", "weight": 0.25},
        ],
        "missing": "CB_MISSING",
        "threshold": 0.5,
    }
    if manifest.get("recipe") != expected_recipe:
        raise BatchIntegrityError("Accepted V3 control recipe differs from C06 policy")
    return manifest


def _official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    # load_train resolves only the official Train path through task_files; no Test/Sample is opened.
    frame, target, train_path = load_train(None)
    if not train_path.is_file():
        raise FileNotFoundError("Official Classification Train is missing")
    _require_hash(train_path, EXPECTED_TRAIN_SHA256, "official Classification Train")
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Unexpected official Classification Train row count: {len(frame)}")
    if frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any() or not frame.record_id.astype(str).is_unique:
        raise BatchIntegrityError("Classification Train record_id is blank or duplicated")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Classification Train cycle coverage differs from 47–79")
    target = np.asarray(target, dtype=np.int8)
    if target.shape != (len(frame),) or not np.isin(target, [0, 1]).all():
        raise BatchIntegrityError("Classification Train artifact_signal is invalid")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("C06 feature allowlist contains protected fields")
    return train_path.resolve(), frame, target, cycles


def _load_controls(frame: pd.DataFrame, target: np.ndarray):
    pilot_freeze = check_sparse_pilot()
    pilot_report = json.loads((PILOT_REPORT / "pilot.json").read_text(encoding="utf-8"))
    pilot_verification = json.loads((PILOT_REPORT / "verification.json").read_text(encoding="utf-8"))
    if pilot_verification.get("status") != "PASS":
        raise BatchIntegrityError("Frozen C2_B/CatBoost/sparse25 control pilot did not independently verify")
    if pilot_report.get("freeze_signature") != pilot_freeze.get("signature"):
        raise BatchIntegrityError("Sparse pilot report does not match its frozen identity")
    records: dict[int, dict[str, Any]] = {}

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        control = control_record(frame, target, int(cutoff))
        pilot = pilot_record(frame, control)
        if not np.array_equal(control["rows"], score_rows) or not np.array_equal(pilot["rows"], score_rows):
            raise BatchIntegrityError(f"C06 control row positions differ at cutoff {cutoff}")
        missing = np.asarray(control["missing"], dtype=bool)
        c2b = np.asarray(control["c2b"], dtype=np.float64)
        cb = np.asarray(control["cb_missing"], dtype=np.float64)
        old_sparse = np.asarray(pilot["raw_candidate"], dtype=np.float64)
        baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
        if not np.array_equal(baseline, np.asarray(pilot["blend25"], dtype=np.float64)):
            raise BatchIntegrityError(f"C06 accepted sparse25 route does not reconstruct at {cutoff}")
        if not np.array_equal(np.asarray(control["champion"], dtype=np.float64),
                              np.where(missing, cb, c2b)):
            raise BatchIntegrityError(f"C06 C2_B/CatBoost route does not reconstruct at {cutoff}")
        records[int(cutoff)] = {
            "score_rows": int(len(score_rows)),
            "stream_present_rows": int((~missing).sum()),
            "stream_missing_rows": int(missing.sum()),
            "c2b_control": control["c2b_identity"],
            "catboost_cache_sha256": control["cb_meta"]["npz_sha256"],
            "catboost_prediction_sha256": control["cb_meta"]["prediction_sha256"],
            "sparse_pilot_cache_sha256": pilot["pilot_cache_sha256"],
            "sparse_pilot_freeze_signature": pilot_freeze["signature"],
        }
        return {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
        }

    return load, records, pilot_freeze


def _chronological_novelty_preflight(frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        fit_x = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
        # Labels and all later-cycle category values are excluded from this proof.
        novelty = FitChronologicalCategoryNovelty(minimum_class_support=MINIMUM_CLASS_SUPPORT)
        matrix = novelty.fit_transform(fit_x, cycles[fit_rows])
        if matrix.shape != (len(fit_rows), 6):
            raise BatchIntegrityError(f"C06 chronological novelty columns are not six at cutoff {cutoff}")
        summary[str(cutoff)] = {
            "fit_rows": int(len(fit_rows)),
            "fit_cycle_min": int(np.min(cycles[fit_rows])),
            "fit_cycle_max": int(np.max(cycles[fit_rows])),
            "feature_names": list(NOVELTY_FEATURE_NAMES),
            "flag_support": novelty.audit["flag_support"],
            "cycle_audit": novelty.audit["cycle_audit"],
            "category_audit": novelty.audit["category_audit"],
            "strict_prior_cycle_rule_proven": True,
            "target_rates_used": False,
            "validation_or_test_categories_used": False,
        }
        for name, counts in summary[str(cutoff)]["flag_support"].items():
            if min(int(counts["zeros"]), int(counts["ones"])) < MINIMUM_CLASS_SUPPORT:
                raise BatchIntegrityError(
                    f"C06 feature {name} at cutoff {cutoff} has insufficient support: {counts}"
                )
    return summary


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray, card: dict[str, Any],
                        queue: dict[str, Any]) -> dict[str, Any]:
    preset = queue["execution_preset"]
    fit_rows = np.flatnonzero(cycles <= 67)
    documents, _ = _text_values(frame.iloc[fit_rows]["signal_stream"])
    analyzer = TfidfVectorizer(**dict(_TFIDF_PARAMS)).build_analyzer()
    text_nnz_upper = int(sum(len(analyzer(document)) for document in documents))
    sparse_bytes = int(text_nnz_upper * 16 + (len(fit_rows) + 1) * 8 + len(fit_rows) * 6 * 8)
    estimated_peak_gib = float(1.5 + 3.0 * sparse_bytes / (1024 ** 3))
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    cap_gib = float(preset["max_job_RSS_GiB"])
    reserve_gib = 2.0
    min_free_gib = float(preset["minimum_free_RAM_GiB_before_start"])
    allowed_gib = min(cap_gib, available_gib - reserve_gib)
    if available_gib < min_free_gib or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"C06 resource gate blocked: available={available_gib:.2f}GiB, "
            f"estimate={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB")
    if int(preset["concurrent_fit_jobs"]) != 1 or int(preset["threads_per_job"]) != 2:
        raise BatchIntegrityError("C06 requires one serialized fit job with two threads")
    if int(card["pilot_predictor_fits"]) != 2:
        raise BatchIntegrityError("C06 pilot fit count changed")
    if "no dense full-word" not in str(preset.get("RAM_policy", "")).casefold():
        raise BatchIntegrityError("Queue no longer prohibits dense full-word matrices")

    ledger = REPORT_DIR.parent / "batch_ledger.jsonl"
    reservations: dict[str, int] = {}
    if ledger.is_file():
        for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid batch budget ledger line {line_number}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key, count = str(event["budget_key"]), int(event["predictor_fits"])
                if key in reservations and reservations[key] != count:
                    raise BatchIntegrityError(f"Conflicting existing fit reservation for {key}")
                reservations[key] = count
    key = "C06:PILOT"
    if key in reservations and reservations[key] != int(card["pilot_predictor_fits"]):
        raise BatchIntegrityError("Existing C06 fit reservation differs from the canonical card")
    reserved_after = sum(reservations.values()) + (0 if key in reservations else int(card["pilot_predictor_fits"]))
    fit_cap = int(preset["max_successful_new_predictor_fits"])
    if reserved_after > fit_cap:
        raise BatchIntegrityError(f"C06 fit budget blocked: reserved_after={reserved_after}, cap={fit_cap}")
    return {
        "largest_fit_rows": int(len(fit_rows)),
        "word_ngram_nnz_upper_bound": text_nnz_upper,
        "sparse_design_upper_gib": float(sparse_bytes / (1024 ** 3)),
        "estimated_peak_job_gib": estimated_peak_gib,
        "available_ram_gib_observed": available_gib,
        "allowed_job_ram_gib": allowed_gib,
        "max_job_ram_gib": cap_gib,
        "minimum_free_ram_gib": min_free_gib,
        "reserved_fit_total_after_C06": int(reserved_after),
        "fit_budget_cap": fit_cap,
        "new_predictor_fits": int(card["pilot_predictor_fits"]),
        "concurrent_fit_jobs": 1,
        "threads": 2,
        "no_dense_full_word_matrix": True,
        "estimator_fits_performed": 0,
    }


def _control_input_files(train_path: Path, card_path: Path,
                         views: dict[str, list[tuple[int, int, int]]]) -> dict[str, Path]:
    cutoffs = sorted({int(cutoff) for segments in views.values() for cutoff, _, _ in segments})
    inputs: dict[str, Path] = {
        "official_classification_train": train_path,
        "queue": QUEUE_PATH,
        "card": card_path,
        "accepted_v3_sparse25_manifest": ACCEPTED_MANIFEST,
        "classification_diagnostic_evidence": DIAGNOSTIC_EVIDENCE,
        "frozen_sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "frozen_sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
        "sparse_pilot_report": PILOT_REPORT / "pilot.json",
        "sparse_pilot_verification": PILOT_REPORT / "verification.json",
    }
    for cutoff in cutoffs:
        inputs[f"sparse_pilot_{cutoff}_npz"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.npz"
        inputs[f"sparse_pilot_{cutoff}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.json"
        inputs[f"c2b_round1_{cutoff}_npz"] = ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"
        inputs[f"c2b_round1_{cutoff}_audit"] = ROUND1_AUDITS / f"fit{cutoff}.json"
        inputs[f"catboost_{cutoff}_npz"] = CB_CACHE / f"fit{cutoff}.npz"
        inputs[f"catboost_{cutoff}_sidecar"] = CB_CACHE / f"fit{cutoff}.json"
    return inputs


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "scripts/research_campaign_classification.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1_c06.py",
        ROOT / "tests/test_optimization_v3_batch1_c06.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "src/models/campaign_classification_catboost.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_metrics.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/utils/paths.py",
    ]


def reconstruct_c06(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild present-stream blend and unchanged CatBoost missing branch."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(x) for x in (raw, missing, c2b, cb, old_sparse)}) != 1:
        raise BatchIntegrityError("C06 saved routing arrays have inconsistent row counts")
    candidate = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise BatchIntegrityError("C06 route reconstruction contains nonfinite probabilities")
    return candidate, baseline


def _independent_novelty_support(X: pd.DataFrame, cycles: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recalculate flag support independently from the model's persisted encoder state."""
    n = len(X)
    novelty = {field: np.zeros(n, dtype=np.uint8) for field in CATEGORY_COLUMNS}
    missing = {field: np.zeros(n, dtype=np.uint8) for field in CATEGORY_COLUMNS}
    values: dict[str, list[str | None]] = {}
    for field in CATEGORY_COLUMNS:
        normalized: list[str | None] = []
        for raw in X[field].to_numpy(dtype=object):
            if _is_missing(raw):
                normalized.append(None)
            else:
                normalized.append(str(raw).strip())
        values[field] = normalized
    vocab = {field: set() for field in CATEGORY_COLUMNS}
    for cycle in sorted(np.unique(np.asarray(cycles, dtype=np.int64)).tolist()):
        positions = np.flatnonzero(np.asarray(cycles, dtype=np.int64) == int(cycle))
        for field in CATEGORY_COLUMNS:
            additions: set[str] = set()
            for position in positions:
                value = values[field][int(position)]
                if value is None:
                    missing[field][position] = 1
                elif value not in vocab[field]:
                    novelty[field][position] = 1
                    additions.add(value)
            vocab[field].update(additions)
    matrix = np.column_stack([
        col for field in CATEGORY_COLUMNS for col in (novelty[field], missing[field])
    ])
    support = {
        name: {"zeros": int(np.count_nonzero(matrix[:, index] == 0)),
               "ones": int(np.count_nonzero(matrix[:, index] == 1)),
               "minimum_per_class": MINIMUM_CLASS_SUPPORT}
        for index, name in enumerate(NOVELTY_FEATURE_NAMES)
    }
    category_audit = {}
    for field in CATEGORY_COLUMNS:
        serialized = json.dumps(sorted(vocab[field]), ensure_ascii=False,
                                separators=(",", ":")).encode("utf-8", errors="surrogatepass")
        category_audit[field] = {
            "fit_categories_seen_by_final_cycle": len(vocab[field]),
            "fit_vocabulary_sha256": hashlib.sha256(serialized).hexdigest(),
            "novel_fit_rows": int(novelty[field].sum()),
            "missing_fit_rows": int(missing[field].sum()),
        }
    return support, category_audit


def _verify_c06_fit_states(*, report: dict[str, Any], frame: pd.DataFrame,
                           target: np.ndarray, artifact_dir: Path) -> dict[str, Any]:
    """Validate saved fit-only model lineage and independently recomputed novelty support."""
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    signature = str(report["freeze_signature"])
    verified: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        identity = {"freeze_signature": signature, "card_id": "C06",
                    "cutoff": cutoff, "score_rows": score_rows.tolist()}
        arrays, sidecar = read_bundle(artifact_dir / f"C06_fit{cutoff}.npz", identity)
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        novelty_state = state.get("chronological_category_novelty_state", {})
        expected_fit_x = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
        expected_target = np.asarray(target[fit_rows], dtype=np.int8)
        expected_cycle_counts = {
            str(int(cycle)): int(np.count_nonzero(cycles[fit_rows] == cycle))
            for cycle in np.sort(np.unique(cycles[fit_rows]))
        }
        support, category_audit = _independent_novelty_support(expected_fit_x, cycles[fit_rows])
        state_without_hash = dict(state)
        actual_state_sha = state_without_hash.pop("state_sha256", None)
        expected_state_sha = hashlib.sha256(json.dumps(
            state_without_hash, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogatepass")).hexdigest()
        expected_fit_rows_sha = hashlib.sha256(np.asarray(fit_rows, dtype="<i8").tobytes(order="C")).hexdigest()
        expected_ids = ids[score_rows]
        if (not np.array_equal(arrays.get("score_rows"), score_rows)
                or not np.array_equal(arrays.get("score_record_ids"), expected_ids)
                or int(metadata.get("fit_rows", -1)) != len(fit_rows)
                or int(metadata.get("fit_predictor_count", -1)) != 1
                or metadata.get("cutoff") != cutoff
                or state.get("id") != "V3_BATCH1_C06_ORDERED_CATEGORY_NOVELTY_SPARSE_LGB"
                or state.get("fit_rows") != len(fit_rows)
                or state.get("fit_cycle_min") != int(cycles[fit_rows].min())
                or state.get("fit_cycle_max") != cutoff
                or state.get("fit_cycle_counts") != expected_cycle_counts
                or state.get("input_feature_names") != list(FEATURE_NAMES)
                or state.get("fit_row_positions_sha256") != expected_fit_rows_sha
                or state.get("fit_data_sha256") != _fit_fingerprint(expected_fit_x, FEATURE_NAMES, expected_target)
                or state.get("target_sha256") != _array_digest(expected_target, dtype="<i8")
                or state.get("sparse_classifier_parameters") != dict(SPARSE_PARAMS)
                or state.get("validation_data_passed_to_fit") is not False
                or state.get("early_stopping_used") is not False
                or state.get("test_data_used") is not False
                or state.get("target_rates_used") is not False
                or actual_state_sha != expected_state_sha
                or novelty_state.get("feature_names") != list(NOVELTY_FEATURE_NAMES)
                or novelty_state.get("minimum_class_support") != MINIMUM_CLASS_SUPPORT
                or novelty_state.get("flag_support") != support
                or novelty_state.get("category_audit") != category_audit
                or novelty_state.get("target_rates_used") is not False
                or novelty_state.get("validation_or_test_categories_used_for_fit_features") is not False):
            raise BatchIntegrityError(f"C06 fit-state/frozen-lineage verification failed at cutoff {cutoff}")
        probability = np.asarray(arrays.get("prediction"), dtype=np.float64)
        if (probability.shape != score_rows.shape or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)):
            raise BatchIntegrityError(f"C06 probability range check failed at cutoff {cutoff}")
        verified[str(cutoff)] = {
            "fit_rows": int(len(fit_rows)),
            "fit_cycle_min": int(cycles[fit_rows].min()),
            "fit_cycle_max": int(cutoff),
            "fit_row_positions_sha256": expected_fit_rows_sha,
            "fit_data_sha256": state["fit_data_sha256"],
            "target_sha256": state["target_sha256"],
            "independently_recomputed_flag_support": support,
            "independent_model_state_hash_pass": True,
        }
    return {"status": "PASS", "cutoffs": verified, "test_data_used": False}


def _validate_primary_subgroup_gates(verification: dict[str, Any],
                                     gate_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Recheck the verifier's declared support/F1/positive-recall subgroup gates."""
    diagnostics = verification["views"]["primary"].get("subgroup_diagnostics", {})
    groups = diagnostics.get("groups", {})
    failures: list[dict[str, Any]] = []
    gate_fields = {"signal_stream", "entry_friction", "text_load_q"}
    minimum_group = int(gate_config["supported_group_n_min"])
    minimum_per_class = int(gate_config["positive_and_negative_per_group_min"])
    f1_harm = float(gate_config["supported_group_F1_harm_max"])
    recall_harm = float(gate_config["positive_class_recall_harm_max"])
    for field, field_groups in groups.items():
        for label, row in field_groups.items():
            supported = (int(row["n"]) >= minimum_group
                         and int(row["class_0_n"]) >= minimum_per_class
                         and int(row["class_1_n"]) >= minimum_per_class)
            if bool(row["supported"]) != supported:
                raise BatchIntegrityError(f"Independent subgroup support rule differs for {field}/{label}")
            applies = field in gate_fields and supported
            if bool(row["guard_applies"]) != applies:
                raise BatchIntegrityError(f"Independent subgroup gate coverage differs for {field}/{label}")
            if not applies:
                if row["guard_pass"] is not None:
                    raise BatchIntegrityError(f"Unsupported/descriptive subgroup unexpectedly has a gate result: {field}/{label}")
                continue
            f1_delta = float(row["candidate_minus_baseline_macro_f1"])
            recall_delta = float(row["candidate_minus_baseline_positive_recall"])
            passed = f1_delta >= -f1_harm and recall_delta >= -recall_harm
            if row["guard_pass"] is not passed:
                raise BatchIntegrityError(f"Independent subgroup gate result differs for {field}/{label}")
            if not passed:
                failures.append({"field": field, "group": label, **row})
    recorded_failures = diagnostics.get("supported_guard_failures", [])
    recorded_keys = {(item.get("field"), item.get("group")) for item in recorded_failures}
    computed_keys = {(item.get("field"), item.get("group")) for item in failures}
    if recorded_keys != computed_keys:
        raise BatchIntegrityError("Verifier subgroup failure list differs from the canonical C06 gate")
    return failures


def _write_or_match(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise BatchIntegrityError(f"Existing C06 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _recipe(card: dict[str, Any], views: dict[str, list[tuple[int, int, int]]]) -> dict[str, Any]:
    from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS

    return {
        "card_id": "C06",
        "status": "FROZEN_RECIPE_BEFORE_PILOT",
        "card_sha256": EXPECTED_CARD_SHA256,
        "accepted_control": {
            "manifest": str(ACCEPTED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "present_model": "C2_B probability 0.75 + frozen sparse-binary probability 0.25",
            "missing_model": "frozen CatBoost probability unchanged",
            "threshold": THRESHOLD,
        },
        "input_feature_names": list(FEATURE_NAMES),
        "appended_features": list(NOVELTY_FEATURE_NAMES),
        "ordered_category_novelty": {
            "fields": list(CATEGORY_COLUMNS),
            "fit_rule": "encode every row in cycle c as novel iff nonmissing category is absent from vocabulary from cycles strictly less than c; update vocabulary only after all rows in cycle c are encoded",
            "outer_score_rule": "a nonmissing score row is novel iff its category is absent from the full fit-prefix vocabulary",
            "missing_rule": "pd.isna values set missing=1 and novel=0; category strings are stripped; blank string remains a category to match the unchanged sparse structured model",
            "features_per_field": ["novel_from_prior_cycles", "missing"],
            "minimum_fit_support_per_binary_feature": {"zeros": MINIMUM_CLASS_SUPPORT, "ones": MINIMUM_CLASS_SUPPORT},
            "target_rates_used": False,
            "later_cycle_categories_used_for_fit_flags": False,
            "validation_labels_used": False,
        },
        "unchanged_sparse_classifier": {
            "model": "LightGBM LGBMClassifier",
            "parameters": dict(SPARSE_PARAMS),
            "tfidf_parameters": {
                key: list(value) if isinstance(value, tuple) else value.__name__ if isinstance(value, type) else value
                for key, value in _TFIDF_PARAMS.items()
            },
            "structured_preprocessing": "RegressionTextModel._structured_fit/_structured_transform",
            "design_order": "original sparse structured+TFIDF+stream-missing design, then the six chronology-only novelty/missing flags",
            "threads": 2,
            "early_stopping": False,
        },
        "policy": EXPECTED_POLICY,
        "evaluation_views": views,
        "fit_predictor_count": int(card["pilot_predictor_fits"]),
        "metric": "Macro F1 at the frozen threshold 0.50",
        "gates": {
            "pilot_primary_gain_min": 0.0005,
            "pilot_H12_harm_max": 0.001,
            "individual_pilot_view_harm_max": 0.005,
            "supported_group_n_min": 200,
            "supported_group_F1_harm_max": 0.01,
            "positive_class_recall_harm_max": 0.02,
            "positive_and_negative_per_group_min": 30,
        },
        "test_data_used": False,
    }


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False,
             freeze_only: bool = False, run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one C06 mode")
    _assert_card_identity(card, queue)
    _require_hash(QUEUE_PATH, EXPECTED_QUEUE_SHA256, "canonical queue")
    _require_hash(CARD_PATH, EXPECTED_CARD_SHA256, "canonical C06 card")
    if load_queue(QUEUE_PATH) != queue:
        raise BatchIntegrityError("Dispatcher queue differs from the frozen C06 queue")
    if ROOT / card["card_path"] != CARD_PATH:
        raise BatchIntegrityError("C06 queue points to an unexpected card path")
    _require_hash(DIAGNOSTIC_EVIDENCE, EXPECTED_DIAGNOSTIC_EVIDENCE_SHA256, "classification diagnostic evidence")
    _validate_manifest(queue)
    train_path, frame, target, cycles = _official_train()
    views = evaluation_views_for_card(queue, "C06")
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"C06 evaluation views changed: {views}")

    control_loader, control_metadata, pilot_freeze = _load_controls(frame, target)
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        record = control_loader(cutoff, rows)
        for name, values in record.items():
            if np.asarray(values).shape != (len(rows),):
                raise BatchIntegrityError(f"C06 control {name} is not row-aligned at cutoff {cutoff}")
        missing = np.asarray(record["stream_missing"], dtype=bool)
        if int((~missing).sum()) != control_metadata[cutoff]["stream_present_rows"]:
            raise BatchIntegrityError(f"C06 present/missing coverage changed at cutoff {cutoff}")

    novelty_preflight = _chronological_novelty_preflight(frame, cycles)
    resources = _resource_preflight(frame, cycles, card, queue)
    inputs = _control_input_files(train_path, CARD_PATH, views)
    recipe = _recipe(card, views)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=inputs,
        source_files=_source_files(), evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "recipe_semantic_sha256": json_sha256(recipe),
        "accepted_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sparse_pilot_freeze_signature": pilot_freeze["signature"],
        "control_records": {str(key): value for key, value in sorted(control_metadata.items())},
        "ordered_category_novelty_preflight": novelty_preflight,
        "resource_preflight_recipe": {
            "estimated_peak_job_gib": resources["estimated_peak_job_gib"],
            "word_ngram_nnz_upper_bound": resources["word_ngram_nnz_upper_bound"],
            "cap_gib": resources["max_job_ram_gib"],
            "minimum_free_ram_gib": resources["minimum_free_ram_gib"],
            "reserve_gib": 2.0,
            "threads": 2,
            "concurrent_fit_jobs": 1,
            "estimate_method": "cutoff-67 fit-only word n-gram upper bound; sparse storage x3 plus six compact binary columns and 1.5GiB fixed",
        },
        "python_version": platform.python_version(),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
    })

    public = {
        "status": "PREFLIGHT_PASS",
        "card_id": "C06",
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "accepted_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "control_records": {str(key): value for key, value in sorted(control_metadata.items())},
        "ordered_category_novelty_preflight": novelty_preflight,
        "resource_preflight": resources,
        "evaluation_views": views,
        "input_file_count": len(inputs),
        "source_file_count": len(_source_files()),
        "freeze_identity_sha256": json_sha256(identity),
        "recipe_semantic_sha256": json_sha256(recipe),
        "official_test_or_sample_access": False,
        "candidate_fit_performed": False,
        "artifacts_written": False,
    }
    if preflight_only:
        public["status"] = "PREFLIGHT_PASS_NO_FREEZE_NO_FIT"
        return public

    if freeze_only:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_match(RECIPE_PATH, recipe)
        signature = ensure_frozen(FREEZE_PATH, identity)
        public.update({
            "status": "FROZEN_READY",
            "freeze_signature": signature,
            "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
            "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
            "freeze_sha256": file_sha256(FREEZE_PATH),
            "artifacts_written": True,
        })
        return public

    if not RECIPE_PATH.is_file() or not FREEZE_PATH.is_file():
        raise BatchIntegrityError("C06 pilot requires committed recipe/freeze artifacts before any fit")
    if json.loads(RECIPE_PATH.read_text(encoding="utf-8")) != recipe:
        raise BatchIntegrityError("C06 recipe differs from current preflight; no fit started")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("identity") != identity or frozen.get("signature") != json_sha256(identity):
        raise BatchIntegrityError("C06 source/input identity differs from committed freeze; no fit started")
    if file_sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train changed after C06 freeze")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    append_event(REPORT_DIR / "attempts.jsonl", {
        "event": "PREFLIGHT_PASS", "train_sha256": EXPECTED_TRAIN_SHA256,
        "resource_preflight": resources,
        "ordered_category_novelty_support_passed": True,
        "fit_predictor_count_reserved": 2,
    })
    report = run_pilot(
        frame=frame, target=target, card=card, feature_names=FEATURE_NAMES,
        adapter=fit_predict, control_loader=control_loader,
        freeze_identity=identity, freeze_path=FREEZE_PATH,
        artifact_dir=ARTIFACT_DIR, event_path=REPORT_DIR / "attempts.jsonl",
        evaluation_views=views, threads=2, report_path=REPORT_DIR / "results.json",
        budget_ledger_path=REPORT_DIR.parent / "batch_ledger.jsonl",
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
    )
    model_state_verification = _verify_c06_fit_states(
        report=report, frame=frame, target=target, artifact_dir=ARTIFACT_DIR,
    )
    verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=ARTIFACT_DIR,
        freeze_signature=report["freeze_signature"], evaluation_views=views,
        prediction_reconstructors={"C06": reconstruct_c06},
    )
    verification["c06_fit_state_verification"] = model_state_verification
    gate_config = queue["evaluation"]["classification_gate"]
    subgroup_failures = _validate_primary_subgroup_gates(verification, gate_config)
    checks = {
        "primary_gain_at_least_0_0005": report["views"]["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_gain_min"]),
        "H12_harm_at_most_0_001": report["views"]["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_harm_max"]),
        "W1_harm_at_most_0_005": report["views"]["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "W4_harm_at_most_0_005": report["views"]["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "supported_primary_subgroup_guards": len(subgroup_failures) == 0,
        "independent_metric_and_route_reconstruction": verification.get("status") == "PASS",
        "fit_only_model_state_and_ordered_novelty_reconstruction": model_state_verification.get("status") == "PASS",
    }
    gate = {
        "card_id": "C06", "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": gate_config,
        "supported_subgroup_failures": subgroup_failures,
        "decision_basis": "Predeclared temporal pilot metrics and supported subgroup guards; no Test, SampleSubmission, or Kaggle result used.",
    }
    _write_or_match(REPORT_DIR / "verification.json", verification)
    _write_or_match(REPORT_DIR / "gate.json", gate)
    summary = (
        "# C06 Pilot Summary\n\n"
        f"- Status: **{gate['status']}**\n"
        f"- Primary Macro F1 gain: {report['views']['primary']['gain_positive_is_better']:+.6f} ({report['views']['primary']['rows']:,} rows).\n"
        f"- H12 Macro F1 gain: {report['views']['H12']['gain_positive_is_better']:+.6f} ({report['views']['H12']['rows']:,} rows).\n"
        f"- W1/W4 gains: {report['views']['W1']['gain_positive_is_better']:+.6f} / {report['views']['W4']['gain_positive_is_better']:+.6f}.\n"
        "- Six category novelty/missing flags used strictly prior-cycle fit information; no target-rate encoding was used.\n"
        "- Temporal Train windows are development evidence, not an untouched holdout. Test data was not read.\n"
    )
    summary_path = REPORT_DIR / "summary.md"
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing C06 summary differs; refusing overwrite")
    else:
        with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(summary)
    return {"pilot": report, "verification": verification, "gate": gate,
            "ordered_category_novelty_preflight": novelty_preflight,
            "resource_preflight": resources}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == "C06")
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["reconstruct_c06", "run_card"]


