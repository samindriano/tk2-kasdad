from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_official as official  # noqa: E402
from scripts import optimization_v3_rnbr_continuation as ledger_prior  # noqa: E402
from src.models import research_next_budget  # noqa: E402
from src.models.classification_a02_neighbors import (  # noqa: E402
    TFIDF_PARAMETERS, _documents, fit_text_space,
)
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS,
    MAX_NEIGHBORS,
    MIN_NEIGHBORS,
    predict_from_saved_neighbors,
)
from src.models import regression_rnbr_impact_match as impact  # noqa: E402

_LIMITER = threadpool_limits(limits=2)

BASE_HEAD = "ccc787cfe1538d7426cb952d6334892b9d818e31"
BASE_FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
BASE_REPORT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1"
BASE_RECOVERY = BASE_REPORT / "SERIALIZATION_RECOVERY_V2.json"
BASE_VERIFICATION = BASE_REPORT / "INDEPENDENT_VERIFICATION.json"
OLD_BLOCKER = ROOT / "reports/optimization_v3/rnbr_impact_match_v1/SAFETY_BLOCKER.md"
OLD_PREFLIGHT = ROOT / "reports/optimization_v3/rnbr_impact_match_v1/targetfree_preflight.json"
OLD_A02_FREEZE = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
TRAIN = ROOT / "data/raw/regression/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"

CARD_ID = "REGRESSION_RNBR_IMPACT_MATCH_V2_1"
STAGE = "FOUR_FIT_EXACT_IMPACT_MEMORY_PILOT_V2_1"
REPLAY_CARD_ID = "REGRESSION_RNBR_IMPACT_MATCH_REPLAY_V2_1"
REPLAY_STAGE = "TWO_FIT_TARGET_FREE_TFIDF_REPLAY_V2_1"
AMENDMENT_ID = "RNBR_IMPACT_MATCH_CAP83_V21_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
REPLAY_RESERVATION_KEY = f"{REPLAY_CARD_ID}:{REPLAY_STAGE}"
REPLAY_FIT_IDS = (
    "RNBR_IMPACT_MATCH_REPLAY_V21_E67_TFIDF",
    "RNBR_IMPACT_MATCH_REPLAY_V21_E75_TFIDF",
)
FIT_IDS = (
    "RNBR_IMPACT_MATCH_V2_1_E67_OBSERVED",
    "RNBR_IMPACT_MATCH_V2_1_E67_SHUFFLED",
    "RNBR_IMPACT_MATCH_V2_1_E75_OBSERVED",
    "RNBR_IMPACT_MATCH_V2_1_E75_SHUFFLED",
)
ORIGINS = (67, 75)
ARMS = ("observed", "shuffled")
QUERY_CYCLES = {67: (68, 79), 75: (76, 79)}
VIEWS = {"W4": (68, 71), "W6": (76, 79), "H12": (68, 79)}
MIN_SUPPORT_COVERAGE = 0.95
MIN_SHUFFLE_MOVEMENT = 0.80
MAX_RSS_BYTES = 6 * 1024**3
MIN_FREE_RAM_BYTES = 2 * 1024**3

PROTOCOL_PATH = ROOT / "experiments/optimization_v3/rnbr_impact_match_v2_1/protocol.json"
REPLAY_PREFLIGHT_PATH = ROOT / "reports/optimization_v3/rnbr_impact_match_v2_1/replay_preflight.json"
REPLAY_FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_impact_match_v2_1/replay_freeze.json"
PREFLIGHT_PATH = ROOT / "reports/optimization_v3/rnbr_impact_match_v2_1/targetfree_preflight.json"
SYNTHETIC_PATH = ROOT / "reports/optimization_v3/rnbr_impact_match_v2_1/synthetic_qa.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_impact_match_v2_1/freeze.json"
REPORT_DIR = ROOT / "reports/optimization_v3/rnbr_impact_match_v2_1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V2_1"
TEXT_STATE_DIR = RUN_ROOT / "text_space_replays"
GEOMETRY_DIR = RUN_ROOT / "targetfree_geometry"

REPLAY_EXPECTED_AUDITS = {
    67: {
        "fit_rows": 29705,
        "fit_row_positions_sha256": "21df8b3baa7004c330d6ac403ed42e8cbd9de99081d048a15cd8b4332257108d",
        "fit_documents_sha256": "4f67c935c6a82af0b3cf7451aec60474959fca2fe0c6f83ee764be3da8d17240",
        "vocabulary_sha256": "109dbecffbacc91513f78723319cd6f03a7ae46db0a80cefab23989c8f5c1d5d",
        "idf_sha256": "76c658982e37a33b6ef2080745487689948d57c5ff7a71c04167755f050ec8d6",
    },
    75: {
        "fit_rows": 42117,
        "fit_row_positions_sha256": "e733031bb45eac2f1f55b94e45bc7161c1e80801f85564a25a289758100e2bc9",
        "fit_documents_sha256": "e4537b03354f313cdd6440f86555530e18617fc89cda2e8509cea0aeffb0d554",
        "vocabulary_sha256": "2603bf8ba90f509ccc81a4155159e7683b0a3660d7c2c0602ba49ee291266083",
        "idf_sha256": "68fc4caea39a15a0ce7e79869ddab4ae6c239b4ae86ba10c3c01dfa1770980b5",
    },
}


class ImpactMatchRunError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return impact.canonical_bytes(value)


def _sha_file(path: Path) -> str:
    return impact.sha256_file(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ImpactMatchRunError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ImpactMatchRunError(f"Refusing to overwrite immutable artifact: {path}")
        return
    staging = path.with_name(path.name + ".staging")
    with staging.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_paths() -> tuple[str, ...]:
    return (
        "scripts/optimization_v3_rnbr_impact_match_v2.py",
        "scripts/optimization_v3_rnbr_impact_match_verify_v2.py",
        "scripts/optimization_v3_rnbr_official.py",
        "scripts/optimization_v3_rnbr_official_verify.py",
        "scripts/optimization_v3_rnbr_continuation.py",
        "src/models/regression_rnbr_impact_match.py",
        "src/models/regression_neighbor_rnbr.py",
        "src/models/classification_a02_neighbors.py",
        "src/models/classification_models.py",
        "src/models/round1_models.py",
        "src/models/research_next_budget.py",
    )


def _protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "RNBR_IMPACT_MATCH_V2_1",
        "status": "FROZEN_BEFORE_REPLAY_AND_TASK_FITS",
        "authorized_base_head": BASE_HEAD,
        "origins": [67, 75],
        "replay_fit_ids": list(REPLAY_FIT_IDS),
        "replay_fit_count": 2,
        "replay_fit_kind": "TARGET_FREE_TFIDF_REPLAY",
        "replay_fit_semantics": "one fit_text_space/fit_transform per official Regression Train prefix, text/row/cycle inputs only",
        "fit_ids": list(FIT_IDS),
        "fit_count": 4,
        "fit_semantics": "one residual-memory construction for each origin x observed/shuffled arm; no estimator refit",
        "target_source": "official Regression Train only",
        "donors": "present-stream rows, cycles 56 through origin, finite exact float64 impact_load",
        "queries": {"E67_H12": [68, 79], "E67_W4": [68, 71], "E75_W6": [76, 79]},
        "retrieval": {
            "text": "replay the original A02 TF-IDF recipe from official Regression Train text prefixes only; persist/load standalone vectorizers, never load A02 predictor models",
            "tfidf_parameters": _jsonable(TFIDF_PARAMETERS),
            "normalization": "l2",
            "tokenization_and_missing_text": "exact original A02 _documents/_text_values implementation; missing/blank rows become empty strings and remain in the fit population",
            "prefix_population": "ascending source rows with cycle_code 47 through origin; includes missing/blank stream rows as empty documents exactly as A02",
            "expected_text_audits": {str(origin): REPLAY_EXPECTED_AUDITS[origin] for origin in ORIGINS},
            "matching": "exact equality of parsed float64 impact_load; no binning, rounding, or quantile edges",
            "geometry": "search all same-impact donors for positive cosine top32, stable descending cosine then source row",
            "minimum_neighbors": MIN_NEIGHBORS,
            "maximum_neighbors": MAX_NEIGHBORS,
        },
        "residual": "threat_rank minus final-mapped R06 OOF prediction using the frozen strictly-earlier source cutoff",
        "shuffle": {
            "strata": "exact donor cycle x exact impact_load x fixed R06 OOF prediction decile",
            "seed": 42,
            "rng_reset_each_origin": True,
            "traversal": "ascending cycle, impact_load, decile, and source row",
        },
        "formula": "clip(R06_origin + clip(0.25 * cosine_squared_weighted_median(residual), -5, +5), 0, 100)",
        "fallback": "exact R06 for missing stream, missing impact, or fewer than five positive-cosine same-impact neighbors",
        "preflight": {
            "positive_cosine_support_coverage_minimum": MIN_SUPPORT_COVERAGE,
            "coverage_denominator": "present-stream queries in each declared W4, W6, and E67 H12 view",
            "shuffle_source_position_movement_minimum": MIN_SHUFFLE_MOVEMENT,
        },
        "gates": {
            "W4_gain_vs_R06_min": 0.0,
            "W6_gain_vs_R06_min": 0.0,
            "pooled_gain_vs_R06_min": 0.03,
            "H12_gain_vs_R06_min": 0.0,
            "pooled_gain_vs_matched_shuffle_min": 0.01,
            "H12_gain_vs_matched_shuffle_min": 0.0,
            "pooled_gain_vs_original_RNBR_min": 0.03,
            "H12_gain_vs_original_RNBR_min": 0.0,
            "tail_subgroup_harm_vs_R06_max_if_n_ge_100": 0.25,
        },
        "source_boundaries": {
            "classification_labels_or_probabilities_or_model_pickles": "never read or deserialized in this versioned run",
            "official Test or SampleSubmission": "never read",
            "query targets": "read only after all four durable bundles and independent integrity checks",
            "Regression threat_rank during text replay": "not parsed or read; target values used only for post-start donor residuals and post-integrity query evaluation",
            "full_Train_refit": False,
            "Kaggle": "not accessed",
        },
        "amendment_context": {
            "domain_auc_was_seen_before_amendment": True,
            "outcome_blind": False,
            "research_question": "Does density weighting retain predictive benefit when domain AUC is slightly below 0.55 but ESS and clipping pass?",
            "origins": "two fixed origins only; descriptive contrast, not evidence of a general AUC-to-task-gain relationship",
            "status": "diagnostic-only; original N04 remains SKIPPED_SCIENTIFIC_PREFLIGHT",
        },
        "execution": {"jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES,
                      "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES},
        "no_rescue": "preflight failure skips all task fits; no threshold, bin, seed, or recipe search",
    }


def _check_base_freeze(*, train: Mapping[str, np.ndarray]) -> dict[str, Any]:
    base = _read_json(BASE_FREEZE_PATH)
    body = {key: value for key, value in base.items() if key != "freeze_signature"}
    if base.get("freeze_signature") != _sha256(_canonical(body)):
        raise ImpactMatchRunError("Historical RNBR official freeze signature is invalid")
    if _sha_file(TRAIN) != base["official_train"]["sha256"]:
        raise ImpactMatchRunError("Official Regression Train hash differs from the frozen source")
    for rel, expected in base.get("code_sha256", {}).items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"Historical frozen RNBR source differs: {rel}")
    r06_source = base["r06_lineage"]
    if (_sha_file(ROOT / r06_source["source_path"]) != r06_source["source_sha256"]
            or _sha_file(ROOT / r06_source["recipe_path"]) != r06_source["recipe_sha256"]):
        raise ImpactMatchRunError("R06 frozen code or recipe hash differs")
    # This loader parses only record_id, cycle_code, and signal_stream.
    recomputed = official.load_targetfree_train()
    if any(not np.array_equal(train[key], recomputed[key]) for key in ("rows", "cycles", "ids", "present")):
        raise ImpactMatchRunError("Regression target-free Train metadata changed while loading")
    r06_cache = official.load_r06_predictions(train, base)
    a02_metadata = _verify_a02_metadata(base)
    return {"freeze": base, "r06_cache": r06_cache, "a02_metadata": a02_metadata}


def _read_replay_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Read only official Train IDs, cycles, and signal_stream for TF-IDF replay."""
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "signal_stream"],
                        dtype={"record_id": str})
    rows = np.arange(len(frame), dtype=np.int64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    ids = frame["record_id"].astype(str).to_numpy()
    if len(frame) != 50608 or len(np.unique(ids)) != len(ids) or cycles.min() != 47 or cycles.max() != 79:
        raise ImpactMatchRunError("Replay Train ID/cycle boundaries differ from official frozen input")
    docs, missing = _documents(frame["signal_stream"])
    train = {"rows": rows, "cycles": cycles, "ids": ids, "present": ~missing}
    return frame, train, np.asarray(docs, dtype=object)


def _verify_a02_metadata(base_freeze: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Verify A02 code and JSON audit lineage without opening model/NPZ artifacts."""
    a02_freeze = _read_json(OLD_A02_FREEZE)
    identity = a02_freeze.get("identity", {})
    if (identity.get("task") != "A02"
            or a02_freeze.get("freeze_signature") != _sha256(_canonical(identity))
            or a02_freeze.get("source_sha256") != identity.get("source_sha256")):
        raise ImpactMatchRunError("A02 metadata freeze identity or self-signature is invalid")
    for rel, expected in a02_freeze["source_sha256"].items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"A02 TF-IDF source differs from its frozen source: {rel}")
    historical = _read_json(OLD_PREFLIGHT)
    result: dict[int, dict[str, Any]] = {}
    for origin in ORIGINS:
        audit = historical["origin_support"][str(origin)]["a02_text_audit"]
        expected = REPLAY_EXPECTED_AUDITS[origin]
        if (audit.get("origin_cycle") != origin
                or {key: audit.get(key) for key in expected} != expected
                or audit.get("tfidf_parameters") != _jsonable(TFIDF_PARAMETERS)
                or audit.get("tfidf_norm") != "l2"
                or audit.get("text_only_cosine") is not True):
            raise ImpactMatchRunError(f"E{origin} historical A02 text audit does not match the frozen replay target")
        spec = base_freeze["a02_saved_geometry"][str(origin)]["observed"]
        receipt = _read_json(ROOT / spec["receipt_json"])
        state = _read_json(ROOT / spec["state_json"])
        state_body_hash = _sha256(_canonical({key: value for key, value in state.items()
                                              if key != "state_sha256"}))
        receipt_body_hash = _sha256(_canonical({key: value for key, value in receipt.items()
                                                if key != "receipt_sha256"}))
        if (state.get("state_sha256") != state_body_hash
                or _sha256(_canonical(state)) != receipt.get("state_sha256")
                or receipt.get("receipt_sha256") != receipt_body_hash
                or receipt.get("fit_id") != f"A02_E{origin}_OBSERVED"
                or receipt.get("origin_cycle") != origin
                or receipt.get("freeze_signature") != spec.get("freeze_signature")
                or receipt.get("model_sha256") != spec.get("model_sha256")
                or receipt.get("text_prefix_rows") != expected["fit_rows"]
                or receipt.get("text_prefix_row_positions_sha256") != expected["fit_row_positions_sha256"]
                or receipt.get("query_labels_passed_to_fit") is not False
                or receipt.get("test_data_used") is not False
                or state.get("origin_cycle") != origin):
            raise ImpactMatchRunError(f"E{origin} A02 JSON metadata lineage does not verify")
        result[origin] = {**audit, "receipt_sha256": _sha_file(ROOT / spec["receipt_json"]),
                          "state_json_sha256": _sha_file(ROOT / spec["state_json"]),
                          "receipt_model_sha256_field": receipt["model_sha256"],
                          "old_fit_id": receipt["fit_id"],
                          "old_freeze_signature": spec["freeze_signature"]}
    return result


def _load_targetfree_features(train: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "signal_stream", "impact_load"],
                        dtype={"record_id": str, "impact_load": "float64"})
    if (len(frame) != len(train["rows"])
            or not np.array_equal(frame["record_id"].astype(str).to_numpy(), train["ids"])
            or not np.array_equal(pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64), train["cycles"])):
        raise ImpactMatchRunError("Target-free feature rows/IDs/cycles do not align to official Train")
    impacts = frame["impact_load"].to_numpy(dtype=np.float64)
    if np.isinf(impacts).any():
        raise ImpactMatchRunError("Official Train impact_load contains infinity")
    docs, missing = _documents(frame["signal_stream"])
    if not np.array_equal(missing, ~train["present"]):
        raise ImpactMatchRunError("Normalized text availability differs from official target-free Train")
    return frame, np.asarray(docs, dtype=object), impacts


def _replay_fit_dir(origin: int) -> Path:
    if origin not in ORIGINS:
        raise ImpactMatchRunError(f"Unknown TF-IDF replay origin: {origin}")
    return TEXT_STATE_DIR / f"E{origin}"


def _load_standalone_text_state(origin: int, *, expected_audit: Mapping[str, Any] | None = None
                                ) -> tuple[TfidfVectorizer, dict[str, Any]]:
    """Load only this pilot's receipt-bound standalone text-only vectorizer."""
    folder = _replay_fit_dir(origin)
    vectorizer_path = folder / "vectorizer.joblib"
    state_path = folder / "text_state.json"
    receipt_path = folder / "receipt.json"
    if not (vectorizer_path.is_file() and state_path.is_file() and receipt_path.is_file()):
        raise ImpactMatchRunError(f"Standalone E{origin} TF-IDF replay artifact is incomplete")
    receipt = _read_json(receipt_path)
    state = _read_json(state_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != _sha256(_canonical(body))
            or receipt.get("vectorizer_sha256") != _sha_file(vectorizer_path)
            or receipt.get("text_state_sha256") != _sha_file(state_path)
            or receipt.get("origin_cycle") != origin
            or receipt.get("fit_id") != REPLAY_FIT_IDS[ORIGINS.index(origin)]
            or receipt.get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY"
            or receipt.get("query_targets_read") is not False
            or receipt.get("classification_data_read") is not False
            or receipt.get("full_a02_model_loaded") is not False):
        raise ImpactMatchRunError(f"Standalone E{origin} TF-IDF receipt/hash/boundary differs")
    expected = dict(expected_audit or REPLAY_EXPECTED_AUDITS[origin])
    if (state.get("schema_version") != 1 or state.get("origin_cycle") != origin
            or state.get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY"
            or state.get("replay_freeze_signature") != receipt.get("replay_freeze_signature")
            or state.get("audit") != receipt.get("text_audit")
            or receipt.get("fit_population", {}).get("rows") != expected["fit_rows"]
            or receipt.get("fit_population", {}).get("cycle_bounds") != [47, origin]
            or receipt.get("fit_transform_calls") != 1):
        raise ImpactMatchRunError(f"Standalone E{origin} TF-IDF state/receipt content differs")
    vectorizer = joblib.load(vectorizer_path)
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(vectorizer, "vocabulary_"):
        raise ImpactMatchRunError(f"E{origin} replay artifact is not a fitted TfidfVectorizer")
    audit = dict(state.get("audit", {}))
    vocab = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    live_vocab_sha = _sha256(_canonical([(str(token), int(index)) for token, index in vocab]))
    live_idf_sha = impact.sha256_array(vectorizer.idf_, "<f8")
    if (audit.get("origin_cycle") != origin
            or any(audit.get(key) != value for key, value in expected.items())
            or audit.get("tfidf_parameters") != _jsonable(TFIDF_PARAMETERS)
            or audit.get("tfidf_norm") != "l2"
            or audit.get("vocabulary_sha256") != live_vocab_sha
            or audit.get("idf_sha256") != live_idf_sha
            or live_vocab_sha != expected["vocabulary_sha256"]
            or live_idf_sha != expected["idf_sha256"]):
        raise ImpactMatchRunError(f"Standalone E{origin} TF-IDF state digest mismatch")
    return vectorizer, audit


def _original_rnbr_paths(origin: int, recovery: Mapping[str, Any]) -> tuple[str, str]:
    if origin == 67:
        recovery_event = recovery["serialization_recovery_detail"]["recovery_event"]
        return recovery_event["recovery_bundle_path"], recovery_event["recovery_receipt_path"]
    fit_id = f"RNBR_PILOT_E{origin}_OBSERVED"
    folder = official._bundle_dir(fit_id)
    return (folder.relative_to(ROOT).as_posix() + "/arrays.npz",
            folder.relative_to(ROOT).as_posix() + "/receipt.json")


def _validate_comparator_alignment(origin: int, geometry: Mapping[str, np.ndarray],
                                   comparator: Mapping[str, Any]) -> None:
    if (not np.array_equal(comparator["rows"], geometry["query_rows"])
            or not np.array_equal(comparator["ids"], geometry["query_ids"])
            or not np.array_equal(comparator["cycles"], geometry["query_cycles"])
            or not np.array_equal(comparator["baseline"], geometry["baseline_prediction"])):
        raise ImpactMatchRunError(f"Original RNBR E{origin} comparator does not align to frozen geometry")


def _load_original_rnbr_predictions(origin: int, base_freeze: Mapping[str, Any],
                                    train: Mapping[str, np.ndarray]) -> dict[str, Any]:
    recovery = _read_json(BASE_RECOVERY)
    independent = _read_json(BASE_VERIFICATION)
    if (recovery.get("status") != "FOUR_FIT_PILOT_COMPLETE_DIAGNOSTIC_ONLY"
            or recovery.get("freeze_signature") != base_freeze.get("freeze_signature")
            or independent.get("status") != "PASS"
            or independent.get("freeze_signature") != base_freeze.get("freeze_signature")
            or independent.get("source_boundaries", {}).get("classification_labels_loaded") is not False
            or independent.get("source_boundaries", {}).get("test_or_sample_submission_opened") is not False):
        raise ImpactMatchRunError("Existing original-RNBR pilot recovery/verification is not PASS")
    fit_id = f"RNBR_PILOT_E{origin}_OBSERVED"
    fit_record = next((item for item in recovery["fit_receipts"] if item["fit_id"] == fit_id), None)
    if fit_record is None:
        raise ImpactMatchRunError(f"Original observed R-NBR receipt is missing for E{origin}")
    bundle_rel, receipt_rel = _original_rnbr_paths(origin, recovery)
    npz_path, receipt_path = ROOT / bundle_rel, ROOT / receipt_rel
    receipt = _read_json(receipt_path)
    receipt_body = {key: value for key, value in receipt.items() if key != "integrity_sha256"}
    if (_sha_file(npz_path) != fit_record["bundle_npz_sha256"]
            or _sha_file(receipt_path) != fit_record["receipt_sha256"]
            or receipt.get("integrity_sha256") != official.sha256_bytes(official.canonical_bytes(receipt_body))
            or receipt.get("npz_sha256") != fit_record["bundle_npz_sha256"]
            or receipt.get("identity", {}).get("fit_id") != fit_id
            or receipt.get("identity", {}).get("freeze_signature") != base_freeze["freeze_signature"]
            or receipt.get("identity", {}).get("query_labels_read") is not False
            or receipt.get("identity", {}).get("test_or_sample_submission_opened") is not False
            or receipt.get("identity", {}).get("classification_labels_used") is not False):
        raise ImpactMatchRunError(f"Accepted original RNBR E{origin} observed bundle does not verify")
    # Load only numeric query rows/predictions/baseline. Some historical string
    # members use object dtype; they are deliberately never deserialized.
    with np.load(npz_path, allow_pickle=False) as archive:
        rows = np.asarray(archive["score_rows"], dtype=np.int64).copy()
        ids = np.asarray(archive["score_record_ids"]).astype(str, copy=True)
        cycles = np.asarray(archive["score_cycles"], dtype=np.int64).copy()
        prediction = np.asarray(archive["prediction"], dtype=np.float64).copy()
        baseline = np.asarray(archive["baseline_prediction"], dtype=np.float64).copy()
    expected_low, expected_high = QUERY_CYCLES[origin]
    expected = np.flatnonzero((train["cycles"] >= expected_low) & (train["cycles"] <= expected_high))
    if (not np.array_equal(rows, expected) or not np.array_equal(ids, train["ids"][rows])
            or not np.array_equal(cycles, train["cycles"][rows])
            or prediction.shape != rows.shape or baseline.shape != rows.shape
            or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))):
        raise ImpactMatchRunError(f"Accepted original RNBR E{origin} rows/predictions are invalid")
    return {"rows": rows, "ids": ids, "cycles": cycles,
            "prediction": prediction, "baseline": baseline,
            "npz_sha256": fit_record["bundle_npz_sha256"],
            "receipt_sha256": fit_record["receipt_sha256"],
            "bundle_path": bundle_rel, "receipt_path": receipt_rel}


def _baseline_for_origin(cache: Mapping[int, Mapping[str, np.ndarray]], origin: int,
                         rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    source = cache[origin]
    lookup = {int(row): i for i, row in enumerate(source["rows"])}
    values = np.empty(len(rows), dtype=np.float64)
    for i, (row, record_id) in enumerate(zip(rows, ids)):
        j = lookup.get(int(row))
        if j is None or str(source["ids"][j]) != str(record_id):
            raise ImpactMatchRunError(f"R06 origin E{origin} query cache misses row/ID {row}")
        values[i] = float(source["prediction"][j])
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise ImpactMatchRunError(f"R06 E{origin} query baseline is invalid")
    return values


def _build_targetfree_geometry(origin: int, train: Mapping[str, np.ndarray],
                               frame: pd.DataFrame, docs: np.ndarray, impacts: np.ndarray,
                               cache: Mapping[int, Mapping[str, np.ndarray]], vectorizer: Any
                               ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cycles, present = train["cycles"], train["present"]
    donor_rows = np.flatnonzero((cycles >= 56) & (cycles <= origin) & present & np.isfinite(impacts)).astype(np.int64)
    query_low, query_high = QUERY_CYCLES[origin]
    query_rows = np.flatnonzero((cycles >= query_low) & (cycles <= query_high)).astype(np.int64)
    if not len(donor_rows) or not len(query_rows):
        raise ImpactMatchRunError(f"E{origin} donor or query population is empty")
    donor_docs = docs[donor_rows].tolist()
    query_docs = docs[query_rows].tolist()
    donor_matrix = vectorizer.transform(donor_docs).tocsr()
    query_matrix = vectorizer.transform(query_docs).tocsr()
    query_missing = ~present[query_rows]
    donor_impact = impacts[donor_rows]
    query_impact = impacts[query_rows]
    geometry = impact.exact_impact_top32(
        query_matrix, donor_matrix, query_impact=query_impact, donor_impact=donor_impact,
        donor_rows=donor_rows, query_stream_missing=query_missing,
    )
    donor_ids = train["ids"][donor_rows].astype(str)
    query_ids = train["ids"][query_rows].astype(str)
    donor_cycle = cycles[donor_rows].astype(np.int64)
    query_cycle = cycles[query_rows].astype(np.int64)
    neighbor_ids = np.full(geometry["neighbor_rows"].shape, "", dtype="U64")
    donor_lookup = {int(row): j for j, row in enumerate(donor_rows)}
    for i, count in enumerate(geometry["neighbor_count"]):
        for j, row in enumerate(geometry["neighbor_rows"][i, :int(count)]):
            neighbor_ids[i, j] = donor_ids[donor_lookup[int(row)]]
    baseline = _baseline_for_origin(cache, origin, query_rows, query_ids)
    effective_missing = query_missing | ~np.isfinite(query_impact)
    impact.validate_same_impact_geometry(
        query_rows=query_rows, query_cycles=query_cycle, query_impact=query_impact, query_ids=query_ids,
        query_stream_missing=query_missing, donor_rows=donor_rows, donor_cycles=donor_cycle,
        donor_impact=donor_impact, donor_ids=donor_ids, origin=origin,
        neighbor_rows=geometry["neighbor_rows"], neighbor_ids=neighbor_ids,
        neighbor_similarities=geometry["neighbor_similarities"], neighbor_count=geometry["neighbor_count"],
    )
    arrays = {
        "origin": np.asarray([origin], dtype=np.int16),
        "donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycle,
        "donor_impact_load": donor_impact,
        "query_rows": query_rows, "query_ids": query_ids, "query_cycles": query_cycle,
        "query_impact_load": query_impact,
        "query_stream_missing": query_missing,
        "query_impact_missing": ~np.isfinite(query_impact),
        "query_effective_missing": effective_missing,
        "baseline_prediction": baseline,
        "neighbor_rows": geometry["neighbor_rows"],
        "neighbor_ids": neighbor_ids,
        "neighbor_similarities": geometry["neighbor_similarities"],
        "neighbor_count": geometry["neighbor_count"],
    }
    present_query = ~query_missing
    view_stats: dict[str, Any] = {}
    for view, (low, high) in (("H12_E67", (68, 79)), ("W4_E67", (68, 71)), ("W6_E75", (76, 79))):
        if (origin == 67 and view == "W6_E75") or (origin == 75 and view != "W6_E75"):
            continue
        selected = (query_cycle >= low) & (query_cycle <= high)
        denominator = int(np.sum(selected & present_query))
        supported = int(np.sum(selected & present_query & (geometry["neighbor_count"] >= MIN_NEIGHBORS)))
        impact_missing = int(np.sum(selected & present_query & ~np.isfinite(query_impact)))
        view_stats[view] = {
            "total_rows": int(np.sum(selected)),
            "present_stream_denominator": denominator,
            "supported_positive_cosine_rows_ge5": supported,
            "coverage": supported / denominator if denominator else 0.0,
            "stream_missing_rows": int(np.sum(selected & query_missing)),
            "impact_missing_present_rows": impact_missing,
            "exact_r06_fallback_rows": int(np.sum(selected & (geometry["neighbor_count"] < MIN_NEIGHBORS) | selected & query_missing)),
        }
    # A unique synthetic label per source position measures the exact frozen
    # permutation's donor-position movement without opening any target values.
    _, _, shuffle_audit = impact.shuffle_residuals_within_exact_cells(
        np.arange(len(donor_rows), dtype=np.float64), donor_cycle, donor_impact,
        official.aligned_r06_predictions(donor_rows, donor_cycle, cache), donor_rows, seed=42,
    )
    meta = {
        "origin": origin,
        "donor_rows": len(donor_rows),
        "present_queries": int(present_query.sum()),
        "query_rows": len(query_rows),
        "tfidf_features": int(donor_matrix.shape[1]),
        "donor_matrix_nnz": int(donor_matrix.nnz),
        "query_matrix_nnz": int(query_matrix.nnz),
        "views": view_stats,
        "matched_shuffle_source_position_audit": shuffle_audit,
        "geometry_array_sha256": {key: impact.sha256_array(value) for key, value in arrays.items()},
    }
    return arrays, meta


def _synthetic_payload() -> dict[str, np.ndarray]:
    donor_rows = np.asarray([3, 8, 13, 21, 34, 55], dtype=np.int64)
    donor_cycles = np.asarray([56, 56, 57, 58, 59, 59], dtype=np.int64)
    donor_impact = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    donor_ids = np.asarray(["d-α", "d-東京", "d-3", "d-4", "d-5", "d-6"], dtype="U8")
    donor_matrix = sparse.csr_matrix(np.asarray([
        [1., 0., 0.], [0., 0., 1.], [1., 0., 0.],
        [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
    ]))
    query_matrix = sparse.csr_matrix(np.asarray([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
    query_rows = np.asarray([101, 102, 103, 104], dtype=np.int64)
    query_cycles = np.asarray([60, 60, 60, 60], dtype=np.int64)
    query_impact = np.asarray([1.0, 2.0, 1.0, np.nan], dtype=np.float64)
    query_ids = np.asarray(["q1", "q2", "q3", "q4"], dtype="U2")
    stream_missing = np.asarray([False, False, True, False])
    result = impact.exact_impact_top32(
        query_matrix, donor_matrix, query_impact=query_impact, donor_impact=donor_impact,
        donor_rows=donor_rows, query_stream_missing=stream_missing,
    )
    neighbor_ids = np.full(result["neighbor_rows"].shape, "", dtype="U8")
    lookup = {int(row): j for j, row in enumerate(donor_rows)}
    for i, count in enumerate(result["neighbor_count"]):
        for j, row in enumerate(result["neighbor_rows"][i, :int(count)]):
            neighbor_ids[i, j] = donor_ids[lookup[int(row)]]
    baseline = np.asarray([51.0, 62.0, 12.0, 95.0])
    residual = np.asarray([8.0, -10.0, 4.0, -2.0, 13.0, 13.0])
    donor_r06 = np.asarray([20., 20., 30., 30., 40., 40.])
    residual_shuffled, source_rows, shuffle_audit = impact.shuffle_residuals_within_exact_cells(
        residual, donor_cycles, donor_impact, donor_r06, donor_rows,
    )
    impact.validate_same_impact_geometry(
        query_rows=query_rows, query_cycles=query_cycles, query_impact=query_impact, query_ids=query_ids,
        query_stream_missing=stream_missing, donor_rows=donor_rows, donor_cycles=donor_cycles,
        donor_impact=donor_impact, donor_ids=donor_ids, origin=59,
        neighbor_rows=result["neighbor_rows"], neighbor_ids=neighbor_ids,
        neighbor_similarities=result["neighbor_similarities"], neighbor_count=result["neighbor_count"],
    )
    predicted = predict_from_saved_neighbors(
        donor_rows=donor_rows, donor_residuals=residual,
        query_r06_predictions=baseline, query_missing=stream_missing | ~np.isfinite(query_impact),
        neighbor_rows=result["neighbor_rows"], neighbor_cosines=result["neighbor_similarities"],
        neighbor_counts=result["neighbor_count"],
    )
    control = predict_from_saved_neighbors(
        donor_rows=donor_rows, donor_residuals=residual_shuffled,
        query_r06_predictions=baseline, query_missing=stream_missing | ~np.isfinite(query_impact),
        neighbor_rows=result["neighbor_rows"], neighbor_cosines=result["neighbor_similarities"],
        neighbor_counts=result["neighbor_count"],
    )
    return {
        "query_rows": query_rows, "query_ids": query_ids, "query_cycles": query_cycles,
        "query_impact_load": query_impact, "query_stream_missing": stream_missing,
        "donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycles,
        "donor_impact_load": donor_impact, "donor_residuals": residual,
        "donor_r06_predictions": donor_r06,
        "neighbor_rows": result["neighbor_rows"], "neighbor_ids": neighbor_ids,
        "neighbor_similarities": result["neighbor_similarities"],
        "neighbor_count": result["neighbor_count"], "shuffle_source_rows": source_rows,
        "shuffle_residuals": residual_shuffled,
        "observed_prediction": predicted["prediction"], "shuffled_prediction": control["prediction"],
        "baseline_prediction": baseline,
        "shuffle_source_movement": np.asarray([shuffle_audit["source_position_movement_rate"]]),
    }


def synthetic_qa() -> dict[str, Any]:
    global TEXT_STATE_DIR, REPORT_DIR
    payload = _synthetic_payload()
    vectorizer_replay = None
    old_text_root, old_report_dir = TEXT_STATE_DIR, REPORT_DIR
    with tempfile.TemporaryDirectory(prefix="rnbr-impact-match-synthetic-") as temp:
        temp_root = Path(temp)
        folder = temp_root / "fit"
        receipt = impact.persist_bundle_once(
            folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64,
            arrays=payload, metadata={"synthetic_only": True, "task_fit": False},
        )
        loaded, read_receipt = impact.load_bundle(
            folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64
        )
        if receipt != read_receipt or any(not impact.arrays_equal(loaded[key], value) for key, value in payload.items()):
            raise ImpactMatchRunError("Synthetic impact-match persistence/readback failed")
        original_hash = _sha_file(folder / "arrays.npz")
        impact.persist_bundle_once(
            folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64,
            arrays=payload, metadata={"synthetic_only": True, "task_fit": False},
        )
        if _sha_file(folder / "arrays.npz") != original_hash:
            raise ImpactMatchRunError("Synthetic impact-match resume changed the durable NPZ")
        try:
            changed = dict(payload)
            changed["observed_prediction"] = payload["observed_prediction"] + 1.0
            impact.persist_bundle_once(
                folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64,
                arrays=changed, metadata={"synthetic_only": True, "task_fit": False},
            )
        except impact.ImpactMatchError:
            reject_overwrite = True
        else:
            reject_overwrite = False
        if not reject_overwrite:
            raise ImpactMatchRunError("Synthetic altered-output resume was not rejected")

        # Exercise the exact text-only fit function on synthetic rows, then the
        # production standalone vectorizer persistence/readback/resume route.
        text_root = temp_root / "text_states"
        report_root = temp_root / "reports"
        TEXT_STATE_DIR, REPORT_DIR = text_root, report_root
        synthetic_docs = np.asarray([
            "alpha shared green", "alpha shared blue", "alpha shared gold",
            "bravo shared red", "bravo shared silver", "bravo shared white",
            "alpha bravo orange", "alpha bravo purple", "bravo alpha violet",
            "alpha shared black", "bravo shared gray", "alpha bravo yellow",
        ], dtype=object)
        synthetic_rows = np.arange(len(synthetic_docs), dtype=np.int64)
        synthetic_cycles = np.asarray([47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 66, 67], dtype=np.int64)
        space, matrix, missing = fit_text_space(
            synthetic_docs.tolist(), synthetic_rows, synthetic_cycles, origin_cycle=67,
        )
        text_audit = space.audit_state()
        expected = {key: text_audit[key] for key in (
            "fit_rows", "fit_row_positions_sha256", "fit_documents_sha256",
            "vocabulary_sha256", "idf_sha256",
        )}
        synthetic_freeze = {"freeze_signature": "s" * 64}
        text_receipt = _persist_vectorizer(
            67, space.vectorizer, text_audit, synthetic_freeze, "synthetic-only",
            expected_audit=expected,
        )
        text_hash = _sha_file(_replay_fit_dir(67) / "receipt.json")
        vectorizer, loaded_audit = _load_standalone_text_state(67, expected_audit=expected)
        if (loaded_audit != text_audit or matrix.shape[0] != len(synthetic_docs)
                or len(missing) != len(synthetic_docs)
                or vectorizer.transform(synthetic_docs.tolist()).shape != matrix.shape):
            raise ImpactMatchRunError("Synthetic text replay persistence/readback differs")
        fake_events = [
            {"event": "FIT_STARTED", "fit_id": text_receipt["fit_id"],
             "fit_kind": "TARGET_FREE_TFIDF_REPLAY", "freeze_signature": "s" * 64},
            {"event": "FIT_SAVED", "fit_id": text_receipt["fit_id"],
             "receipt_sha256": text_hash},
        ]
        if not _replay_saved(67, synthetic_freeze, fake_events, expected_audit=expected):
            raise ImpactMatchRunError("Synthetic target-free replay durable resume failed")
        if _sha_file(_replay_fit_dir(67) / "receipt.json") != text_hash:
            raise ImpactMatchRunError("Synthetic replay resume changed the persisted receipt")

        state_path = _replay_fit_dir(67) / "text_state.json"
        original_state = state_path.read_bytes()
        state_path.unlink()
        try:
            _load_standalone_text_state(67, expected_audit=expected)
        except ImpactMatchRunError:
            missing_state_rejected = True
        else:
            missing_state_rejected = False
        state_path.write_bytes(original_state)
        if not missing_state_rejected:
            raise ImpactMatchRunError("Synthetic missing standalone state was not rejected")

        vectorizer_path = _replay_fit_dir(67) / "vectorizer.joblib"
        original_vectorizer = vectorizer_path.read_bytes()
        vectorizer_path.write_bytes(original_vectorizer + b"tamper")
        try:
            _load_standalone_text_state(67, expected_audit=expected)
        except ImpactMatchRunError:
            vectorizer_tamper_rejected = True
        else:
            vectorizer_tamper_rejected = False
        vectorizer_path.write_bytes(original_vectorizer)
        if not vectorizer_tamper_rejected:
            raise ImpactMatchRunError("Synthetic replay vectorizer tamper was not rejected")

        # Nested recovery lookup must prefer the verified recovery event, even
        # when an unrelated top-level path is present.
        nested_paths = _original_rnbr_paths(67, {
            "recovery_bundle_path": "wrong/top-level.npz",
            "recovery_receipt_path": "wrong/top-level.json",
            "serialization_recovery_detail": {"recovery_event": {
                "recovery_bundle_path": "nested/recovered.npz",
                "recovery_receipt_path": "nested/recovered.json",
            }},
        })
        if nested_paths != ("nested/recovered.npz", "nested/recovered.json"):
            raise ImpactMatchRunError("Synthetic nested E67 recovery lookup selected the wrong path")
        synthetic_geometry = {key: payload[key] for key in (
            "query_rows", "query_ids", "query_cycles", "baseline_prediction",
        )}
        synthetic_comparator = {
            "rows": payload["query_rows"], "ids": payload["query_ids"],
            "cycles": payload["query_cycles"], "baseline": payload["baseline_prediction"],
        }
        _validate_comparator_alignment(67, synthetic_geometry, synthetic_comparator)
        try:
            _validate_comparator_alignment(67, synthetic_geometry,
                                           {**synthetic_comparator,
                                            "ids": synthetic_comparator["ids"][::-1]})
        except ImpactMatchRunError:
            comparator_swapped_ids_rejected = True
        else:
            comparator_swapped_ids_rejected = False
        if not comparator_swapped_ids_rejected:
            raise ImpactMatchRunError("Synthetic comparator with swapped source IDs was not rejected")

        from scripts import optimization_v3_rnbr_impact_match_verify_v2 as verify  # noqa: PLC0415
        synthetic_views = {
            67: {"query_cycles": np.arange(68, 80, dtype=np.int64)},
            75: {"query_cycles": np.arange(76, 80, dtype=np.int64)},
        }
        verify._validate_view_assignments(synthetic_views)
        try:
            verify._validate_view_assignments({67: synthetic_views[75], 75: synthetic_views[67]})
        except verify.ImpactMatchVerificationError:
            swapped_views_rejected = True
        else:
            swapped_views_rejected = False
        if not swapped_views_rejected:
            raise ImpactMatchRunError("Synthetic swapped temporal views were not rejected")

        # Run the independent verification logic against durable synthetic
        # bundle arrays, then persist/read back its result and a diagnostic-only
        # report. No official labels or result artifacts are involved.
        synthetic_verify_report = verify.verify_synthetic_fixture(loaded, payload)
        synthetic_results_report = {
            "status": "DIAGNOSTIC_ONLY", "qualification": "not eligible",
            "synthetic_only": True,
            "verification_status": synthetic_verify_report["status"],
            "observed_prediction_sha256": synthetic_verify_report["observed_prediction_sha256"],
            "shuffled_prediction_sha256": synthetic_verify_report["shuffled_prediction_sha256"],
        }
        _write_once(report_root / "verification.json", synthetic_verify_report)
        _write_once(report_root / "results.json", synthetic_results_report)
        _write_once(report_root / "SUMMARY.md", {"text": "# Synthetic diagnostic report\\n"})
        if (_read_json(report_root / "verification.json") != synthetic_verify_report
                or _read_json(report_root / "results.json") != synthetic_results_report
                or "Synthetic diagnostic report" not in (report_root / "SUMMARY.md").read_text(encoding="utf-8")):
            raise ImpactMatchRunError("Synthetic verifier/report persistence readback failed")
        vectorizer_replay = {
            "fit_transform_calls": 1, "rows": len(synthetic_docs),
            "standalone_persistence_readback": True, "durable_resume_without_refit": True,
            "missing_state_rejected": missing_state_rejected,
            "vectorizer_tamper_rejected": vectorizer_tamper_rejected,
            "nested_recovery_path": list(nested_paths), "swapped_views_rejected": swapped_views_rejected,
            "comparator_alignment_assertion": True,
            "comparator_swapped_ids_rejected": comparator_swapped_ids_rejected,
            "independent_synthetic_verifier": synthetic_verify_report,
            "verifier_and_diagnostic_report_persistence": True,
        }
        TEXT_STATE_DIR, REPORT_DIR = old_text_root, old_report_dir
    if (payload["neighbor_rows"][0, 0] not in payload["donor_rows"]
            or payload["neighbor_count"][0] < MIN_NEIGHBORS
            or payload["neighbor_count"][2] != 0 or payload["neighbor_count"][3] != 0
            or payload["observed_prediction"][2] != payload["baseline_prediction"][2]
            or payload["observed_prediction"][3] != payload["baseline_prediction"][3]):
        raise ImpactMatchRunError("Synthetic exact-impact or R06 fallback assertions failed")
    return {
        "status": "PASS",
        "fit_type": "SYNTHETIC_ONLY",
        "actual_same_impact_builder": True,
        "same_impact_full_bank_search": True,
        "strict_time_and_geometry_validation": True,
        "weighted_median_formula_observed_and_control": True,
        "missing_stream_and_missing_impact_exact_R06_fallback": True,
        "pickle_free_unicode_persistence_readback": True,
        "idempotent_resume": True,
        "changed_output_rejected": True,
        "target_free_text_replay_synthetic": vectorizer_replay,
        "synthetic_donor_rows": len(payload["donor_rows"]),
        "synthetic_query_rows": len(payload["query_rows"]),
    }


def _source_hashes() -> dict[str, str]:
    return {rel: _sha_file(ROOT / rel) for rel in _source_paths()}


def _base_artifact_hashes(base_freeze: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {
        "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json": _sha_file(BASE_FREEZE_PATH),
        "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/SERIALIZATION_RECOVERY_V2.json": _sha_file(BASE_RECOVERY),
        "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json": _sha_file(BASE_VERIFICATION),
        "reports/optimization_v3/rnbr_impact_match_v1/SAFETY_BLOCKER.md": _sha_file(OLD_BLOCKER),
        "reports/optimization_v3/rnbr_impact_match_v1/targetfree_preflight.json": _sha_file(OLD_PREFLIGHT),
        "experiments/optimization_v3/c19_a02_v2/A02/freeze.json": _sha_file(OLD_A02_FREEZE),
        "data/raw/regression/Train.csv": _sha_file(TRAIN),
        base_freeze["r06_lineage"]["source_path"]: _sha_file(ROOT / base_freeze["r06_lineage"]["source_path"]),
        base_freeze["r06_lineage"]["recipe_path"]: _sha_file(ROOT / base_freeze["r06_lineage"]["recipe_path"]),
    }
    for cutoff, item in base_freeze["r06_lineage"]["cache_inventory"].items():
        result[item["npz"]] = _sha_file(ROOT / item["npz"])
        result[item["json"]] = _sha_file(ROOT / item["json"])
    # A02 model/joblib and classifier prediction/geometry NPZ files are
    # deliberately excluded. Only frozen source and JSON audit metadata are
    # consulted for the clean text replay.
    a02_source_manifest = _read_json(OLD_A02_FREEZE)["source_sha256"]
    for rel in ("src/models/classification_a02_neighbors.py",
                "src/models/classification_models.py", "src/models/round1_models.py"):
        result[rel] = _sha_file(ROOT / rel)
        if result[rel] != a02_source_manifest.get(rel):
            raise ImpactMatchRunError(f"A02 text replay source identity differs: {rel}")
    for origin in ORIGINS:
        spec = base_freeze["a02_saved_geometry"][str(origin)]["observed"]
        for key in ("receipt_json", "state_json"):
            rel = spec[key]
            result[rel] = _sha_file(ROOT / rel)
    r06_legacy = {
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json",
        "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json",
    }
    for rel in r06_legacy:
        result[rel] = _sha_file(ROOT / rel)
    for origin in ORIGINS:
        fit_id = f"RNBR_PILOT_E{origin}_OBSERVED"
        record = _read_json(BASE_RECOVERY)["serialization_recovery_detail"]["recovery_event"] if origin == 67 else None
        if origin == 67:
            for key in ("recovery_bundle_path", "recovery_receipt_path"):
                rel = record[key]
                result[rel] = _sha_file(ROOT / rel)
        else:
            folder = official._bundle_dir(fit_id)
            result[folder.relative_to(ROOT).as_posix() + "/arrays.npz"] = _sha_file(folder / "arrays.npz")
            result[folder.relative_to(ROOT).as_posix() + "/receipt.json"] = _sha_file(folder / "receipt.json")
    return result


def _replay_prefix_report(train: Mapping[str, np.ndarray], docs: np.ndarray,
                          old_metadata: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    prefixes: dict[str, Any] = {}
    for origin in ORIGINS:
        mask = (train["cycles"] >= 47) & (train["cycles"] <= origin)
        rows = train["rows"][mask].astype(np.int64)
        cycles = train["cycles"][mask].astype(np.int64)
        prefix_docs = docs[mask].tolist()
        if (not len(rows) or np.any(np.diff(rows) <= 0) or cycles.min() != 47 or cycles.max() != origin
                or len(rows) != REPLAY_EXPECTED_AUDITS[origin]["fit_rows"]):
            raise ImpactMatchRunError(f"E{origin} replay prefix rows/cycles are not the frozen ascending 47-origin population")
        row_sha = official.sha256_array(rows, "<i8")
        doc_sha = official.sha256_bytes(official.canonical_bytes(prefix_docs))
        if (row_sha != REPLAY_EXPECTED_AUDITS[origin]["fit_row_positions_sha256"]
                or doc_sha != REPLAY_EXPECTED_AUDITS[origin]["fit_documents_sha256"]):
            raise ImpactMatchRunError(f"E{origin} target-free replay row/document digest differs before fit")
        fit_input_sha = _sha256(_canonical({
            "record_ids": train["ids"][mask].tolist(),
            "cycle_code": cycles.tolist(),
            "normalized_signal_stream_sha256": doc_sha,
        }))
        old_audit = dict(old_metadata[origin])
        prefixes[str(origin)] = {
            "origin_cycle": origin, "cycle_bounds": [47, origin],
            "fit_rows": int(len(rows)), "row_positions_sha256": row_sha,
            "documents_sha256": doc_sha, "fit_input_sha256": fit_input_sha,
            "expected_vocabulary_sha256": REPLAY_EXPECTED_AUDITS[origin]["vocabulary_sha256"],
            "expected_idf_sha256": REPLAY_EXPECTED_AUDITS[origin]["idf_sha256"],
            "expected_parameters": _jsonable(TFIDF_PARAMETERS),
            "old_json_audit_identity": {
                key: old_audit[key] for key in (
                    "receipt_sha256", "state_json_sha256", "receipt_model_sha256_field",
                    "old_fit_id", "old_freeze_signature", "vocabulary_sha256", "idf_sha256",
                    "fit_row_positions_sha256", "fit_documents_sha256",
                )
            },
        }
    return prefixes


def _replay_input_source_hashes() -> dict[str, str]:
    a02_freeze = _read_json(OLD_A02_FREEZE)
    result = {
        "data/raw/regression/Train.csv": _sha_file(TRAIN),
        "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json": _sha_file(BASE_FREEZE_PATH),
        "experiments/optimization_v3/c19_a02_v2/A02/freeze.json": _sha_file(OLD_A02_FREEZE),
        "reports/optimization_v3/rnbr_impact_match_v1/targetfree_preflight.json": _sha_file(OLD_PREFLIGHT),
        "reports/optimization_v3/rnbr_impact_match_v1/SAFETY_BLOCKER.md": _sha_file(OLD_BLOCKER),
    }
    for rel in ("src/models/classification_a02_neighbors.py",
                "src/models/classification_models.py", "src/models/round1_models.py"):
        digest = _sha_file(ROOT / rel)
        if digest != a02_freeze["source_sha256"].get(rel):
            raise ImpactMatchRunError(f"A02 source changed since its text-space audit: {rel}")
        result[rel] = digest
    for origin in ORIGINS:
        spec = _read_json(BASE_FREEZE_PATH)["a02_saved_geometry"][str(origin)]["observed"]
        for key in ("receipt_json", "state_json"):
            rel = spec[key]
            result[rel] = _sha_file(ROOT / rel)
    for rel in _source_paths():
        result[rel] = _sha_file(ROOT / rel)
    return result


def _build_replay_freeze_body(protocol: Mapping[str, Any], preflight: Mapping[str, Any],
                              synthetic: Mapping[str, Any]) -> dict[str, Any]:
    if dict(protocol) != _protocol() or synthetic.get("status") != "PASS":
        raise ImpactMatchRunError("Replay protocol or synthetic executor QA is not frozen/PASS")
    return {
        "schema_version": 1,
        "protocol_id": "RNBR_IMPACT_MATCH_V2_1_TEXT_REPLAY",
        "authorized_base_head": BASE_HEAD,
        "protocol_sha256": _sha_file(PROTOCOL_PATH),
        "replay_preflight_sha256": _sha_file(REPLAY_PREFLIGHT_PATH),
        "synthetic_qa_sha256": _sha_file(SYNTHETIC_PATH),
        "source_sha256": _replay_input_source_hashes(),
        "input_sha256": {
            "official_train_file_identity": _sha_file(TRAIN),
            "replay_prefixes": {key: item["fit_input_sha256"] for key, item in preflight["prefixes"].items()},
        },
        "prefixes": preflight["prefixes"],
        "replay_fit_ids": list(REPLAY_FIT_IDS),
        "replay_fit_kind": "TARGET_FREE_TFIDF_REPLAY",
        "replay_fit_count": 2,
        "a02_full_model_or_pickle_loaded": False,
        "classification_labels_probabilities_or_raw_train_loaded": False,
        "regression_target_values_parsed_or_used": False,
        "official_test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
        "resource_policy": {"jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES,
                            "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES},
    }


def prepare_replay() -> dict[str, Any]:
    """Freeze source, exact target-free replay inputs, and synthetic checks before TF-IDF fits."""
    protocol = _protocol()
    _write_once(PROTOCOL_PATH, protocol)
    synthetic = synthetic_qa()
    _write_once(SYNTHETIC_PATH, synthetic)
    frame, train, docs = _read_replay_inputs()
    base = _check_base_freeze(train=train)
    prefixes = _replay_prefix_report(train, docs, base["a02_metadata"])
    preflight = {
        "status": "READY_FOR_TWO_REPLAYS",
        "classification_data_read": False,
        "full_a02_model_or_pickle_loaded": False,
        "train_columns_parsed": ["record_id", "cycle_code", "signal_stream"],
        "regression_target_values_parsed_or_used": False,
        "official_test_or_sample_submission_opened": False,
        "query_targets_read": False,
        "kaggle_accessed": False,
        "train_rows": len(frame),
        "train_file_sha256": _sha_file(TRAIN),
        "prefixes": prefixes,
        "historical_preflight_used_as": "expected A02 text audit digests only; old status/clearance flags are ignored",
        "historical_blocker_path": OLD_BLOCKER.relative_to(ROOT).as_posix(),
        "a02_text_metadata_verified": True,
        "source_identity_sha256": _replay_input_source_hashes(),
    }
    _write_once(REPLAY_PREFLIGHT_PATH, preflight)
    freeze_body = _build_replay_freeze_body(protocol, preflight, synthetic)
    freeze = {**freeze_body, "freeze_signature": _sha256(_canonical(freeze_body))}
    _write_once(REPLAY_FREEZE_PATH, freeze)
    return {"status": preflight["status"], "freeze_signature": freeze["freeze_signature"],
            "preflight": preflight, "freeze_path": REPLAY_FREEZE_PATH.relative_to(ROOT).as_posix()}


def _current_replay_freeze(*, require_clean: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = _read_json(REPLAY_FREEZE_PATH)
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    protocol = _read_json(PROTOCOL_PATH)
    preflight = _read_json(REPLAY_PREFLIGHT_PATH)
    synthetic = _read_json(SYNTHETIC_PATH)
    if (freeze.get("freeze_signature") != _sha256(_canonical(body))
            or body != _build_replay_freeze_body(protocol, preflight, synthetic)):
        raise ImpactMatchRunError("Replay source/input/parameter freeze no longer matches current artifacts")
    if require_clean and _git("status", "--porcelain", "--untracked-files=all"):
        raise ImpactMatchRunError("Replay fit requires the committed clean pre-replay checkpoint")
    if _git("merge-base", "--is-ancestor", BASE_HEAD, "HEAD") is None:
        raise ImpactMatchRunError("Cannot confirm authorized base commit ancestry")
    return freeze, protocol, preflight


def _append_budget_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return ledger_prior._append_event(event)


def amend_and_reserve_all(replay_freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    events = _ledger_events()
    snapshot = _ledger_snapshot(events)
    prior_caps = {"pilot": 38, "continuation": 38, "control_replay": 0, "technical_retry": 1}
    new_caps = {"pilot": 42, "continuation": 38, "control_replay": 2, "technical_retry": 1}
    prior_hash = _sha_file(LEDGER)
    amendments = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT"
                  and event.get("amendment_id") == AMENDMENT_ID]
    if not amendments:
        expected = {"events": 201, "global_cap": 77, "caps": prior_caps,
                    "active_by_category": prior_caps, "active_reserved": 77,
                    "started": 77, "saved": 74, "failed": 3, "pending": 0}
        if snapshot != expected or prior_hash != "35ca73c9adc64df8b794db29cc8cd39109362bea865a8bc6b1d8caf0c7703ff8":
            raise ImpactMatchRunError(f"Live ledger differs from the reviewed 77-fit checkpoint: {snapshot}")
        _append_budget_event({
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sol replay amendment02; exactly two target-free replays plus four memory fits",
            "prior_global_cap": 77, "global_cap": 83,
            "prior_effective_caps": prior_caps, "effective_caps": new_caps,
            "active_reservations_before": snapshot["active_by_category"],
            "active_total_before": snapshot["active_reserved"],
            "pre_amendment_event_rows": snapshot["events"],
            "pre_amendment_ledger_sha256": prior_hash,
            "source_commit": source_commit, "freeze_signature": replay_freeze["freeze_signature"],
            "reason": "Bounded two-fit TARGET_FREE_TFIDF_REPLAY plus four exact-impact memory fits; no other fit/retry.",
        })
        events = _ledger_events()
    amendments = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT"
                  and event.get("amendment_id") == AMENDMENT_ID]
    if (len(amendments) != 1 or amendments[0].get("global_cap") != 83
            or amendments[0].get("effective_caps") != new_caps
            or amendments[0].get("freeze_signature") != replay_freeze["freeze_signature"]):
        raise ImpactMatchRunError("Existing 83-fit cap amendment differs from Sol's exact allowance")
    reservations = [
        (REPLAY_CARD_ID, REPLAY_STAGE, "control_replay", 2, list(REPLAY_FIT_IDS), REPLAY_RESERVATION_KEY),
        (CARD_ID, STAGE, "pilot", 4, list(FIT_IDS), RESERVATION_KEY),
    ]
    for card, stage, category, fits, fit_ids, key in reservations:
        events = _ledger_events()
        matches = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == key]
        if len(matches) > 1:
            raise ImpactMatchRunError(f"Duplicate reservation key: {key}")
        if not matches:
            snapshot = _ledger_snapshot(events)
            if snapshot["active_reserved"] + fits > 83:
                raise ImpactMatchRunError(f"Reservation exceeds exact cap: {snapshot}")
            _append_budget_event({
                "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": key, "card_id": card, "stage": stage,
                "category": category, "fits": fits, "fit_ids": fit_ids,
                "amendment_id": AMENDMENT_ID, "global_cap": 83,
                "effective_category_cap": new_caps[category],
                "freeze_signature": replay_freeze["freeze_signature"], "source_commit": source_commit,
            })
        elif (matches[0].get("fits") != fits or matches[0].get("fit_ids") != fit_ids
              or matches[0].get("category") != category
              or matches[0].get("freeze_signature") != replay_freeze["freeze_signature"]):
            raise ImpactMatchRunError(f"Existing reservation differs: {key}")
    final = _ledger_snapshot(_ledger_events())
    if (final["global_cap"] != 83 or final["caps"] != new_caps
            or final["active_reserved"] != 83 or final["started"] != 77 or final["pending"] != 0):
        raise ImpactMatchRunError(f"Post-reservation ledger differs from exact 83-fit cap: {final}")
    return {"snapshot": final, "ledger_sha256": _sha_file(LEDGER)}


def _replay_receipt_path(origin: int) -> Path:
    return REPORT_DIR / "replays" / f"E{origin}_receipt.json"


def _persist_vectorizer(origin: int, vectorizer: TfidfVectorizer, audit: Mapping[str, Any],
                        freeze: Mapping[str, Any], source_commit: str, *,
                        expected_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    folder = _replay_fit_dir(origin)
    folder.mkdir(parents=True, exist_ok=True)
    vectorizer_path = folder / "vectorizer.joblib"
    state_path = folder / "text_state.json"
    receipt_path = folder / "receipt.json"
    if vectorizer_path.exists() or state_path.exists() or receipt_path.exists():
        raise ImpactMatchRunError(f"Replay output already exists without a validated resume path: E{origin}")
    state = {"schema_version": 1, "origin_cycle": origin, "fit_kind": "TARGET_FREE_TFIDF_REPLAY",
             "audit": dict(audit), "source_commit": source_commit,
             "replay_freeze_signature": freeze["freeze_signature"]}
    _write_once(state_path, state)
    staging = vectorizer_path.with_name(vectorizer_path.name + ".staging")
    if staging.exists():
        raise ImpactMatchRunError(f"Unfinished vectorizer staging artifact requires review: {staging}")
    joblib.dump(vectorizer, staging, compress=3)
    with staging.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(staging, vectorizer_path)
    receipt: dict[str, Any] = {
        "schema_version": 1, "fit_id": REPLAY_FIT_IDS[ORIGINS.index(origin)],
        "fit_kind": "TARGET_FREE_TFIDF_REPLAY", "origin_cycle": origin,
        "replay_freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
        "fit_population": {"rows": audit["fit_rows"], "cycle_bounds": [47, origin],
                           "row_positions_sha256": audit["fit_row_positions_sha256"],
                           "documents_sha256": audit["fit_documents_sha256"]},
        "text_audit": dict(audit), "vectorizer_sha256": _sha_file(vectorizer_path),
        "text_state_sha256": _sha_file(state_path),
        "query_targets_read": False, "classification_data_read": False,
        "full_a02_model_loaded": False, "fit_transform_calls": 1,
    }
    receipt["receipt_sha256"] = _sha256(_canonical(receipt))
    _write_once(receipt_path, receipt)
    # Readback is restricted to the standalone vectorizer just written.
    loaded, loaded_audit = _load_standalone_text_state(origin, expected_audit=expected_audit)
    if loaded_audit != dict(audit):
        raise ImpactMatchRunError(f"Standalone E{origin} TF-IDF state differs on readback")
    del loaded
    return receipt


def _replay_saved(origin: int, freeze: Mapping[str, Any], events: list[dict[str, Any]], *,
                 expected_audit: Mapping[str, Any] | None = None) -> bool:
    fit_id = REPLAY_FIT_IDS[ORIGINS.index(origin)]
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    folder = _replay_fit_dir(origin)
    if terminal and (len(starts) != 1 or len(terminal) != 1 or terminal[0].get("event") != "FIT_SAVED"):
        raise ImpactMatchRunError(f"Replay {fit_id} has failed/ambiguous accounting; no retry")
    if not starts:
        if terminal or folder.exists():
            raise ImpactMatchRunError(f"Unstarted replay {fit_id} already has state; no fit/retry")
        return False
    # A start without a complete receipt is consumed and cannot be retried.
    vectorizer, audit = _load_standalone_text_state(origin, expected_audit=expected_audit)
    del vectorizer
    receipt = _read_json(folder / "receipt.json")
    if (starts[0].get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY"
            or starts[0].get("freeze_signature") != freeze["freeze_signature"]
            or receipt.get("replay_freeze_signature") != freeze["freeze_signature"]
            or receipt.get("fit_id") != fit_id
            or receipt.get("text_audit") != audit):
        raise ImpactMatchRunError(f"Replay {fit_id} frozen receipt identity differs")
    receipt_hash = _sha_file(folder / "receipt.json")
    if not terminal:
        _append_budget_event({
            "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": REPLAY_RESERVATION_KEY, "card_id": REPLAY_CARD_ID,
            "stage": REPLAY_STAGE, "fit_id": fit_id, "cutoff": origin,
            "receipt_sha256": receipt_hash, "fit_kind": "TARGET_FREE_TFIDF_REPLAY",
            "recovered_without_refit": True,
        })
    elif terminal[0].get("receipt_sha256") != receipt_hash:
        raise ImpactMatchRunError(f"Replay {fit_id} saved-ledger receipt hash differs")
    _write_once(_replay_receipt_path(origin), receipt)
    return True


def run_replays() -> dict[str, Any]:
    freeze, protocol, preflight = _current_replay_freeze(require_clean=True)
    source_commit = _git("rev-parse", "HEAD")
    if protocol.get("protocol_id") != "RNBR_IMPACT_MATCH_V2_1" or preflight.get("status") != "READY_FOR_TWO_REPLAYS":
        raise ImpactMatchRunError("Clean target-free replay preflight is not ready")
    resources = _resource_check()
    budget = amend_and_reserve_all(freeze, source_commit)
    frame, train, docs = _read_replay_inputs()
    receipts = []
    for origin in ORIGINS:
        fit_id = REPLAY_FIT_IDS[ORIGINS.index(origin)]
        events = _ledger_events()
        if _replay_saved(origin, freeze, events):
            receipts.append(_read_json(_replay_receipt_path(origin)))
            continue
        reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                       and event.get("reservation_key") == REPLAY_RESERVATION_KEY]
        if len(reservation) != 1 or fit_id not in reservation[0].get("fit_ids", []):
            raise ImpactMatchRunError(f"Replay fit {fit_id} is not uniquely reserved")
        if _ledger_snapshot(events)["pending"]:
            raise ImpactMatchRunError("Another Research Next fit is already pending")
        _resource_check()
        _append_budget_event({
            "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": REPLAY_RESERVATION_KEY, "card_id": REPLAY_CARD_ID,
            "stage": REPLAY_STAGE, "fit_id": fit_id, "cutoff": origin,
            "component": "original_A02_word_TFIDF_text_space_replay",
            "fit_kind": "TARGET_FREE_TFIDF_REPLAY", "fit_consumed_at_start": True,
            "amendment_id": AMENDMENT_ID, "freeze_signature": freeze["freeze_signature"],
            "source_commit": source_commit, "resource_snapshot": _resource_check(),
            "classification_data_read": False, "regression_target_values_read": False,
        })
        try:
            mask = (train["cycles"] >= 47) & (train["cycles"] <= origin)
            prefix_rows = train["rows"][mask].astype(np.int64)
            prefix_cycles = train["cycles"][mask].astype(np.int64)
            prefix_docs = docs[mask].tolist()
            text_space, fit_matrix, missing = fit_text_space(
                prefix_docs, prefix_rows, prefix_cycles, origin_cycle=origin,
            )
            audit = text_space.audit_state()
            expected = {**REPLAY_EXPECTED_AUDITS[origin],
                        "origin_cycle": origin, "fit_rows": int(len(prefix_rows)),
                        "fit_row_positions_sha256": official.sha256_array(prefix_rows, "<i8"),
                        "fit_documents_sha256": official.sha256_bytes(official.canonical_bytes(prefix_docs)),
                        "tfidf_parameters": _jsonable(TFIDF_PARAMETERS), "tfidf_norm": "l2",
                        "text_only_cosine": True}
            if any(audit.get(key) != value for key, value in expected.items()):
                raise ImpactMatchRunError(f"E{origin} replay digest/parameter mismatch; no retry is permitted")
            if fit_matrix.shape[0] != len(prefix_rows) or len(missing) != len(prefix_rows):
                raise ImpactMatchRunError(f"E{origin} replay fit output row count differs")
            receipt = _persist_vectorizer(origin, text_space.vectorizer, audit, freeze, source_commit,
                                          expected_audit=REPLAY_EXPECTED_AUDITS[origin])
            receipt_hash = _sha_file(_replay_fit_dir(origin) / "receipt.json")
            _append_budget_event({
                "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": REPLAY_RESERVATION_KEY, "card_id": REPLAY_CARD_ID,
                "stage": REPLAY_STAGE, "fit_id": fit_id, "cutoff": origin,
                "receipt_sha256": receipt_hash, "fit_kind": "TARGET_FREE_TFIDF_REPLAY",
            })
            _write_once(_replay_receipt_path(origin), receipt)
            receipts.append(receipt)
            del text_space, fit_matrix, prefix_docs, prefix_rows, prefix_cycles
        except Exception as exc:
            current = _ledger_events()
            if not any(event.get("fit_id") == fit_id and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                       for event in current):
                _append_budget_event({
                    "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                    "reservation_key": REPLAY_RESERVATION_KEY, "card_id": REPLAY_CARD_ID,
                    "stage": REPLAY_STAGE, "fit_id": fit_id, "cutoff": origin,
                    "receipt_sha256": None, "fit_kind": "TARGET_FREE_TFIDF_REPLAY",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            # Replay failure is terminal and consumes its slot. Release only the
            # four unstarted task-memory slots; never retry or refund the replay.
            try:
                research_next_budget.release_unused(LEDGER, card_id=CARD_ID, stage=STAGE)
            except Exception as release_exc:
                raise ImpactMatchRunError(
                    f"Replay failed ({type(exc).__name__}: {exc}); memory reservation release is unresolved: {release_exc}"
                ) from release_exc
            raise
    snapshot = _ledger_snapshot(_ledger_events())
    if (len(receipts) != 2 or snapshot["started"] != 79 or snapshot["saved"] != 76
            or snapshot["failed"] != 3 or snapshot["pending"] != 0):
        raise ImpactMatchRunError(f"Replay fit accounting did not reconcile exactly: {snapshot}")
    return {"status": "TWO_REPLAYS_SAVED", "fit_kind": "TARGET_FREE_TFIDF_REPLAY",
            "source_commit": source_commit, "resource_snapshot": resources,
            "fit_receipts": [{"fit_id": receipt["fit_id"], "receipt_sha256": receipt["receipt_sha256"],
                              "vectorizer_sha256": receipt["vectorizer_sha256"]} for receipt in receipts],
            "budget": snapshot, "ledger_sha256": _sha_file(LEDGER)}


def prepare_geometry() -> dict[str, Any]:
    """Rebuild target-free geometry from receipt-bound standalone TF-IDF states."""
    replay_freeze, protocol, replay_preflight = _current_replay_freeze(require_clean=False)
    synthetic = _read_json(SYNTHETIC_PATH)
    events = _ledger_events()
    replay_results = []
    for origin in ORIGINS:
        if not _replay_saved(origin, replay_freeze, events):
            raise ImpactMatchRunError(f"E{origin} replay is not durably saved; geometry is blocked")
        replay_results.append(_read_json(_replay_receipt_path(origin)))
        events = _ledger_events()
    replay_starts = [event for event in events if event.get("event") == "FIT_STARTED"
                     and event.get("reservation_key") == REPLAY_RESERVATION_KEY]
    replay_saves = [event for event in events if event.get("event") == "FIT_SAVED"
                     and event.get("reservation_key") == REPLAY_RESERVATION_KEY]
    replay_failures = [event for event in events if event.get("event") == "FIT_FAILED"
                       and event.get("reservation_key") == REPLAY_RESERVATION_KEY]
    if (len(replay_starts) != 2 or len(replay_saves) != 2 or replay_failures
            or {event.get("fit_id") for event in replay_starts} != set(REPLAY_FIT_IDS)
            or {event.get("fit_id") for event in replay_saves} != set(REPLAY_FIT_IDS)
            or any(event.get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY" for event in replay_starts + replay_saves)):
        raise ImpactMatchRunError("Two replay ledger starts/saves do not reconcile exactly")
    frame, train, docs = _read_replay_inputs()
    base = _check_base_freeze(train=train)
    frame_full, docs_full, impacts = _load_targetfree_features(train)
    if not np.array_equal(docs_full.astype(str), docs.astype(str)):
        raise ImpactMatchRunError("Geometry text normalization differs from replay prefix input")
    original_rnbr = {origin: _load_original_rnbr_predictions(origin, base["freeze"], train)
                     for origin in ORIGINS}
    origin_info: dict[str, Any] = {}
    all_views: dict[str, Any] = {}
    for origin in ORIGINS:
        vectorizer, text_audit = _load_standalone_text_state(origin)
        arrays, meta = _build_targetfree_geometry(
            origin, train, frame_full, docs_full, impacts, base["r06_cache"], vectorizer
        )
        geometry_path = GEOMETRY_DIR / f"E{origin}_geometry.npz"
        geometry_hash = impact.save_npz_once(geometry_path, **arrays)
        meta["geometry_path_ignored"] = geometry_path.relative_to(ROOT).as_posix()
        meta["geometry_file_sha256"] = geometry_hash
        meta["a02_text_audit"] = text_audit
        comparator = original_rnbr[origin]
        _validate_comparator_alignment(origin, arrays, comparator)
        meta["original_rnbr_comparator"] = {
            "status": "PASS", "fit_id": f"RNBR_PILOT_E{origin}_OBSERVED",
            "bundle_path": comparator["bundle_path"], "receipt_path": comparator["receipt_path"],
            "npz_sha256": comparator["npz_sha256"], "receipt_sha256": comparator["receipt_sha256"],
            "aligned_query_rows": int(len(comparator["rows"])),
            "query_rows_sha256": impact.sha256_array(comparator["rows"], "<i8"),
            "query_ids_sha256": impact.sha256_array(comparator["ids"].astype(str)),
            "query_cycles_sha256": impact.sha256_array(comparator["cycles"], "<i8"),
        }
        origin_info[str(origin)] = meta
        all_views.update(meta["views"])
        del vectorizer
    coverage_checks = {
        view: {"coverage": stats["coverage"], "minimum": MIN_SUPPORT_COVERAGE,
               "pass": stats["coverage"] >= MIN_SUPPORT_COVERAGE}
        for view, stats in all_views.items()
    }
    shuffle_checks = {
        str(origin): {
            "source_position_movement_rate": origin_info[str(origin)]["matched_shuffle_source_position_audit"]["source_position_movement_rate"],
            "minimum": MIN_SHUFFLE_MOVEMENT,
            "pass": origin_info[str(origin)]["matched_shuffle_source_position_audit"]["source_position_movement_rate"] >= MIN_SHUFFLE_MOVEMENT,
        } for origin in ORIGINS
    }
    support_pass = bool(coverage_checks) and all(item["pass"] for item in coverage_checks.values())
    shuffle_pass = all(item["pass"] for item in shuffle_checks.values())
    status = "PASS" if support_pass and shuffle_pass and synthetic.get("status") == "PASS" else "SKIPPED_SCIENTIFIC_PREFLIGHT"
    memory_release = None
    if status != "PASS":
        memory_release = research_next_budget.release_unused(
            LEDGER, card_id=CARD_ID, stage=STAGE,
        )
    legacy_verification_path = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json"
    legacy_results_path = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json"
    legacy_verification = _read_json(legacy_verification_path)
    legacy_results = _read_json(legacy_results_path)
    legacy_group = legacy_results["groups"]["views"]["W1"]["signal_stream_missing"]
    legacy_harm = float(legacy_group["harm"])
    legacy_n = int(legacy_group["n"])
    if (legacy_n != 392 or abs(legacy_harm - 0.3955562085105022) > 1e-12
            or legacy_verification.get("qualification", {}).get("qualified") is not False
            or ["W1", "signal_stream_missing"] not in legacy_verification.get("qualification", {}).get("supported_group_failures", [])):
        raise ImpactMatchRunError("Inherited R06 W1 missing-stream guard differs from its verified historical evidence")
    report = {
        "status": status,
        "pilot_id": "RNBR_IMPACT_MATCH_V2_1",
        "official_task_fit_started": False,
        "query_targets_read": False,
        "classification_labels_or_probabilities_read": False,
        "full_a02_model_or_pickle_loaded": False,
        "test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
        "train_sha256": _sha_file(TRAIN),
        "train_rows": len(train["rows"]),
        "impact_load_missing_rows": int(np.isnan(impacts).sum()),
        "impact_load_finite_unique_values": int(len(np.unique(impacts[np.isfinite(impacts)]))),
        "replay_lineage": {
            "replay_freeze_signature": replay_freeze["freeze_signature"],
            "replay_fit_ids": list(REPLAY_FIT_IDS),
            "replay_receipts": [{"fit_id": item["fit_id"], "receipt_sha256": item["receipt_sha256"],
                                 "vectorizer_sha256": item["vectorizer_sha256"]} for item in replay_results],
            "ledger_starts": len(replay_starts), "ledger_saves": len(replay_saves),
        },
        "unstarted_memory_reservation_release": memory_release,
        "original_rnbr_comparator_validation_before_memory_fits": True,
        "inherited_legacy_guard": {
            "source_verification": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json",
            "source_results": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json",
            "group": "W1.signal_stream_missing", "harm_vs_champion": legacy_harm,
            "n": legacy_n, "qualification": "not cleared by this diagnostic; historical failure retained",
        },
        "origin_support": origin_info,
        "support_gate_checks": coverage_checks,
        "shuffle_source_position_checks": shuffle_checks,
        "synthetic_qa_sha256": _sha_file(SYNTHETIC_PATH),
        "minimum_present_query_support_coverage": MIN_SUPPORT_COVERAGE,
        "minimum_shuffle_source_position_movement": MIN_SHUFFLE_MOVEMENT,
    }
    _write_once(PREFLIGHT_PATH, report)
    freeze_body = _build_freeze_body(base["freeze"], report)
    freeze = {**freeze_body, "freeze_signature": _sha256(_canonical(freeze_body))}
    _write_once(FREEZE_PATH, freeze)
    return {"status": status, "freeze_signature": freeze["freeze_signature"],
            "preflight": report, "freeze_path": FREEZE_PATH.relative_to(ROOT).as_posix()}


def _build_freeze_body(base_freeze: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    if _read_json(PROTOCOL_PATH) != _protocol():
        raise ImpactMatchRunError("Impact-match protocol differs from its fixed recipe")
    if _read_json(SYNTHETIC_PATH).get("status") != "PASS":
        raise ImpactMatchRunError("Synthetic end-to-end QA is not PASS")
    replay_freeze, _, replay_preflight = _current_replay_freeze(require_clean=False)
    if (replay_preflight.get("status") != "READY_FOR_TWO_REPLAYS"
            or preflight.get("replay_lineage", {}).get("replay_freeze_signature") != replay_freeze["freeze_signature"]):
        raise ImpactMatchRunError("Geometry preflight is not bound to the validated target-free replay")
    geometry_hashes = {}
    for origin in ORIGINS:
        rel = preflight["origin_support"][str(origin)]["geometry_path_ignored"]
        path = ROOT / rel
        if not path.is_file() or _sha_file(path) != preflight["origin_support"][str(origin)]["geometry_file_sha256"]:
            raise ImpactMatchRunError(f"Frozen target-free E{origin} geometry file/hash is absent or changed")
        geometry_hashes[str(origin)] = _sha_file(path)
    return {
        "schema_version": 1,
        "protocol_id": "RNBR_IMPACT_MATCH_V2_1",
        "authorized_base_head": BASE_HEAD,
        "base_rnbr_freeze_signature": base_freeze["freeze_signature"],
        "replay_freeze_signature": replay_freeze["freeze_signature"],
        "protocol_sha256": _sha_file(PROTOCOL_PATH),
        "synthetic_qa_sha256": _sha_file(SYNTHETIC_PATH),
        "targetfree_preflight_sha256": _sha_file(PREFLIGHT_PATH),
        "source_sha256": _source_hashes(),
        "input_sha256": _base_artifact_hashes(base_freeze),
        "official_train": {"path": "data/raw/regression/Train.csv", "rows": 50608,
                           "sha256": base_freeze["official_train"]["sha256"],
                           "target_policy": "donors only after their fit starts; query targets only after all four bundles verify"},
        "origins": [67, 75],
        "fit_ids": list(FIT_IDS),
        "fit_count": 4,
        "replay_fit_ids": list(REPLAY_FIT_IDS),
        "replay_fit_count": 2,
        "replay_receipts": preflight["replay_lineage"]["replay_receipts"],
        "comparator_lineage": {origin: preflight["origin_support"][origin]["original_rnbr_comparator"]
                               for origin in ("67", "75")},
        "inherited_legacy_guard": preflight["inherited_legacy_guard"],
        "geometry_file_sha256": geometry_hashes,
        "preflight_status": preflight["status"],
        "preflight_gates": {
            "support_coverage_minimum": MIN_SUPPORT_COVERAGE,
            "shuffle_source_movement_minimum": MIN_SHUFFLE_MOVEMENT,
        },
        "source_boundaries": {
            "classification_labels_or_probabilities_read": False,
            "classification_raw_data_or_full_a02_model_pickle_loaded": False,
            "regression_targets_read_for_tfidf_replay": False,
            "official_Test_or_SampleSubmission_opened": False,
            "Kaggle_accessed": False,
            "full_Train_refit": False,
            "preprocessing_refit": False,
        },
        "resource_policy": {"jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES,
                            "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES},
    }


def _current_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = _read_json(FREEZE_PATH)
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if freeze.get("freeze_signature") != _sha256(_canonical(body)):
        raise ImpactMatchRunError("Impact-match freeze self-signature is invalid")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise ImpactMatchRunError("Pilot fits require committed preparation and a clean working tree")
    if freeze.get("authorized_base_head") != BASE_HEAD:
        raise ImpactMatchRunError("Impact-match freeze is not anchored to the authorized HEAD")
    for rel, expected in freeze["source_sha256"].items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"Impact-match frozen source changed: {rel}")
    for rel, expected in freeze["input_sha256"].items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"Impact-match frozen input/lineage changed: {rel}")
    protocol = _read_json(PROTOCOL_PATH)
    preflight = _read_json(PREFLIGHT_PATH)
    if (protocol != _protocol() or _sha_file(PROTOCOL_PATH) != freeze["protocol_sha256"]
            or _sha_file(PREFLIGHT_PATH) != freeze["targetfree_preflight_sha256"]
            or _sha_file(SYNTHETIC_PATH) != freeze["synthetic_qa_sha256"]
            or _build_freeze_body(_read_json(BASE_FREEZE_PATH), preflight) != body):
        raise ImpactMatchRunError("Impact-match protocol, preflight, geometry, or freeze content changed")
    replay_freeze, _, replay_preflight = _current_replay_freeze(require_clean=False)
    events = _ledger_events()
    replay_starts = [event for event in events if event.get("event") == "FIT_STARTED"
                     and event.get("reservation_key") == REPLAY_RESERVATION_KEY]
    replay_saves = [event for event in events if event.get("event") == "FIT_SAVED"
                    and event.get("reservation_key") == REPLAY_RESERVATION_KEY]
    if (replay_freeze["freeze_signature"] != freeze.get("replay_freeze_signature")
            or replay_preflight.get("status") != "READY_FOR_TWO_REPLAYS"
            or len(replay_starts) != 2 or len(replay_saves) != 2
            or {event.get("fit_id") for event in replay_starts} != set(REPLAY_FIT_IDS)
            or {event.get("fit_id") for event in replay_saves} != set(REPLAY_FIT_IDS)
            or any(event.get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY" for event in replay_starts + replay_saves)):
        raise ImpactMatchRunError("Target-free TF-IDF replay starts/saves do not verify against the memory freeze")
    for origin in ORIGINS:
        _, audit = _load_standalone_text_state(origin)
        receipt = _read_json(_replay_receipt_path(origin))
        expected = replay_preflight["prefixes"][str(origin)]
        if (receipt.get("text_audit") != audit
                or audit.get("fit_row_positions_sha256") != expected["row_positions_sha256"]
                or audit.get("fit_documents_sha256") != expected["documents_sha256"]):
            raise ImpactMatchRunError(f"E{origin} replay receipt/audit no longer matches the frozen prefix")
    return freeze, protocol, preflight


def _resource_check() -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    rss = int(psutil.Process().memory_info().rss)
    if available < MIN_FREE_RAM_BYTES:
        raise ImpactMatchRunError(f"System free RAM is below the 2 GiB reserve: {available}")
    if rss > MAX_RSS_BYTES:
        raise ImpactMatchRunError(f"Pilot process RSS exceeds the 6 GiB limit: {rss}")
    return {"available_ram_bytes": available, "process_rss_bytes": rss,
            "jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES}


def _ledger_events() -> list[dict[str, Any]]:
    return ledger_prior._read_events()


def _ledger_snapshot(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return ledger_prior._snapshot(_ledger_events() if events is None else events)


def _append_ledger(event: Mapping[str, Any]) -> dict[str, Any]:
    return ledger_prior._append_event(event)


def _obsolete_v1_reservation_route(*args: Any, **kwargs: Any) -> None:
    raise ImpactMatchRunError("Legacy v1 budget path is disabled; use the pre-reserved v2 replay and memory accounting")


def _start_fit(fit_id: str, origin: int, arm: str, freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise ImpactMatchRunError(f"Unauthorized v2 task fit ID: {fit_id}")
    events = _ledger_events()
    existing = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if existing or terminal:
        raise ImpactMatchRunError(f"Fit {fit_id} already has ledger history; use durable resume verification, never refit")
    snapshot = _ledger_snapshot(events)
    if snapshot["pending"] != 0:
        raise ImpactMatchRunError("A fit is already pending in the shared ledger")
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == RESERVATION_KEY]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == RESERVATION_KEY]
    if len(reservation) != 1 or reservation[0].get("fit_ids") != list(FIT_IDS) or len(starts) >= 4:
        raise ImpactMatchRunError("Four-fit impact-match reservation is missing or exhausted")
    resource = _resource_check()
    event = {
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": f"exact_impact_{arm}_residual_memory",
        "fit_kind": "TARGET_MEMORY_CONSTRUCTION",
        "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
        "resource_snapshot": resource,
    }
    return _append_ledger(event)


def _target_bundle_inputs(origin: int, freeze: Mapping[str, Any],
                          geometry: Mapping[str, np.ndarray], train: Mapping[str, np.ndarray],
                          cache: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    donor_rows = geometry["donor_rows"].astype(np.int64)
    donor_cycles = geometry["donor_cycles"].astype(np.int64)
    donor_ids = geometry["donor_ids"].astype(str)
    targets = official.read_regression_targets(donor_rows, purpose=f"RNBR_IMPACT_MATCH_E{origin}_DONOR_TARGETS")
    if (not np.array_equal(targets["ids"].astype(str), donor_ids)
            or not np.array_equal(targets["cycles"].astype(np.int64), donor_cycles)):
        raise ImpactMatchRunError(f"E{origin} donor target IDs/cycles are not aligned")
    donor_y = targets["target"].astype(np.float64)
    donor_r06 = official.aligned_r06_predictions(donor_rows, donor_cycles, cache)
    cutoffs = np.asarray([EXPECTED_SOURCE_CUTOFFS[int(cycle)] for cycle in donor_cycles], dtype=np.int64)
    if np.any(cutoffs >= donor_cycles):
        raise ImpactMatchRunError(f"E{origin} donor R06 source cutoff is not strictly earlier")
    residuals = donor_y - donor_r06
    if not np.isfinite(residuals).all():
        raise ImpactMatchRunError(f"E{origin} donor residuals are nonfinite")
    return {"donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycles,
            "donor_impact_load": geometry["donor_impact_load"].astype(np.float64),
            "donor_source_cutoffs": cutoffs, "donor_targets": donor_y,
            "donor_r06_predictions": donor_r06, "donor_residuals": residuals}


def _fit_arrays(origin: int, arm: str, geometry: Mapping[str, np.ndarray],
                donor: Mapping[str, np.ndarray], source_commit: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    residuals = donor["donor_residuals"]
    if arm == "observed":
        residuals_used = residuals.copy()
        shuffle_source_rows = donor["donor_rows"].copy()
        shuffle_audit = {
            "cells": 0, "permutable_cells": 0, "permutable_donors": 0,
            "source_positions_moved": 0, "source_position_movement_rate": 0.0,
            "residual_values_moved": 0, "residual_value_movement_rate": 0.0,
        }
    elif arm == "shuffled":
        residuals_used, shuffle_source_rows, shuffle_audit = impact.shuffle_residuals_within_exact_cells(
            residuals, donor["donor_cycles"], donor["donor_impact_load"],
            donor["donor_r06_predictions"], donor["donor_rows"], seed=42,
        )
    else:
        raise ImpactMatchRunError(f"Unknown impact-match residual arm {arm}")
    if arm == "shuffled" and shuffle_audit["source_position_movement_rate"] < MIN_SHUFFLE_MOVEMENT:
        raise ImpactMatchRunError(f"E{origin} actual shuffle movement fell below the frozen 80% gate")
    predictions = predict_from_saved_neighbors(
        donor_rows=donor["donor_rows"], donor_residuals=residuals_used,
        query_r06_predictions=geometry["baseline_prediction"],
        query_missing=geometry["query_effective_missing"],
        neighbor_rows=geometry["neighbor_rows"],
        neighbor_cosines=geometry["neighbor_similarities"],
        neighbor_counts=geometry["neighbor_count"],
    )
    arrays = {
        **{key: value.copy() for key, value in donor.items() if key != "donor_targets"},
        "score_rows": geometry["query_rows"].astype(np.int64).copy(),
        "score_record_ids": geometry["query_ids"].astype(str).copy(),
        "score_cycles": geometry["query_cycles"].astype(np.int64).copy(),
        "query_impact_load": geometry["query_impact_load"].astype(np.float64).copy(),
        "query_stream_missing": geometry["query_stream_missing"].astype(bool).copy(),
        "query_impact_missing": geometry["query_impact_missing"].astype(bool).copy(),
        "query_effective_missing": geometry["query_effective_missing"].astype(bool).copy(),
        "baseline_prediction": geometry["baseline_prediction"].astype(np.float64).copy(),
        "neighbor_rows": geometry["neighbor_rows"].astype(np.int64).copy(),
        "neighbor_ids": geometry["neighbor_ids"].astype(str).copy(),
        "neighbor_similarities": geometry["neighbor_similarities"].astype(np.float64).copy(),
        "neighbor_count": geometry["neighbor_count"].astype(np.int16).copy(),
        "shuffle_source_rows": shuffle_source_rows.astype(np.int64).copy(),
        "donor_residuals_used": residuals_used.astype(np.float64).copy(),
        "delta": predictions["delta"].astype(np.float64).copy(),
        "prediction": predictions["prediction"].astype(np.float64).copy(),
        "neighbor_count_used": predictions["neighbor_count_used"].astype(np.int16).copy(),
    }
    metadata = {
        "origin": origin, "arm": arm,
        "source_commit": source_commit,
        "freeze_signature": _read_json(FREEZE_PATH)["freeze_signature"],
        "donor_rows": len(donor["donor_rows"]), "query_rows": len(geometry["query_rows"]),
        "donor_rows_sha256": impact.sha256_array(donor["donor_rows"], "<i8"),
        "donor_target_sha256": impact.sha256_array(donor["donor_targets"], "<f8"),
        "donor_r06_oof_sha256": impact.sha256_array(donor["donor_r06_predictions"], "<f8"),
        "donor_residual_sha256": impact.sha256_array(residuals, "<f8"),
        "residuals_used_sha256": impact.sha256_array(residuals_used, "<f8"),
        "shuffle_source_rows_sha256": impact.sha256_array(shuffle_source_rows, "<i8"),
        "shuffle_audit": shuffle_audit,
        "query_targets_read": False,
        "classification_labels_or_probabilities_read": False,
        "test_or_sample_submission_opened": False,
        "fit_recipe": "RNBR_IMPACT_MATCH_V2_1 frozen protocol",
        "new_estimator_fit_count": 0,
        "memory_construction_fit_count": 1,
    }
    return arrays, metadata


def _fit_dir(fit_id: str) -> Path:
    if fit_id not in FIT_IDS:
        raise ImpactMatchRunError(f"Unknown impact-match fit ID: {fit_id}")
    return RUN_ROOT / "bundles" / fit_id


def _fit_saved_event(fit_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_SAVED"]


def _resume_saved_fit(fit_id: str, freeze: Mapping[str, Any], events: list[dict[str, Any]]) -> bool:
    starts = [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_STARTED"]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    saved = _fit_saved_event(fit_id, events)
    folder = _fit_dir(fit_id)
    if saved:
        if len(starts) != 1 or len(saved) != 1 or terminal != saved:
            raise ImpactMatchRunError(f"Saved fit {fit_id} has duplicate/conflicting accounting")
        arrays, receipt = impact.load_bundle(folder, fit_id=fit_id, freeze_signature=freeze["freeze_signature"])
        digest = _sha_file(folder / "receipt.json")
        if saved[0].get("receipt_sha256") != digest:
            raise ImpactMatchRunError(f"Saved fit {fit_id} ledger receipt hash differs")
        return True
    if terminal:
        raise ImpactMatchRunError(f"Fit {fit_id} is already FIT_FAILED; retry is forbidden")
    if starts:
        if len(starts) != 1 or not folder.is_dir():
            raise ImpactMatchRunError(f"Fit {fit_id} is pending without a durable complete bundle; no retry")
        arrays, receipt = impact.load_bundle(folder, fit_id=fit_id, freeze_signature=freeze["freeze_signature"])
        receipt_hash = _sha_file(folder / "receipt.json")
        _append_ledger({
            "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": int(receipt["metadata"]["origin"]),
            "receipt_sha256": receipt_hash, "fit_kind": "TARGET_MEMORY_CONSTRUCTION",
            "recovered_without_refit": True,
        })
        return True
    if folder.exists():
        raise ImpactMatchRunError(f"Unstarted fit {fit_id} already has outputs; no reuse without lineage")
    return False


def _commit_fit_saved(fit_id: str, origin: int, receipt: Mapping[str, Any]) -> None:
    _append_ledger({
        "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin,
        "receipt_sha256": _sha_file(_fit_dir(fit_id) / "receipt.json"),
        "fit_kind": "TARGET_MEMORY_CONSTRUCTION",
    })


def execute() -> dict[str, Any]:
    freeze, protocol, preflight = _current_freeze()
    if protocol.get("protocol_id") != "RNBR_IMPACT_MATCH_V2_1" or preflight.get("status") != "PASS":
        raise ImpactMatchRunError("V2 scientific preflight did not pass; memory fits are blocked")
    source_commit = _git("rev-parse", "HEAD")
    resources = _resource_check()
    events = _ledger_events()
    if _ledger_snapshot(events)["pending"]:
        raise ImpactMatchRunError("Another Research Next fit is pending; no memory fit may start")
    memory_reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                          and event.get("reservation_key") == RESERVATION_KEY]
    if (len(memory_reservation) != 1 or memory_reservation[0].get("fit_ids") != list(FIT_IDS)
            or memory_reservation[0].get("freeze_signature") != _read_json(REPLAY_FREEZE_PATH)["freeze_signature"]):
        raise ImpactMatchRunError("Four authorized memory slots are not uniquely reserved")

    train = official.load_targetfree_train()
    frame, docs, impacts = _load_targetfree_features(train)
    base = _check_base_freeze(train=train)
    geometries: dict[int, dict[str, np.ndarray]] = {}
    vectorizers: dict[int, TfidfVectorizer] = {}
    original: dict[int, dict[str, Any]] = {}
    for origin in ORIGINS:
        geometry_path = GEOMETRY_DIR / f"E{origin}_geometry.npz"
        with np.load(geometry_path, allow_pickle=False) as archive:
            geometries[origin] = {key: np.asarray(archive[key]).copy() for key in archive.files}
        vectorizers[origin], _ = _load_standalone_text_state(origin)
        original[origin] = _load_original_rnbr_predictions(origin, base["freeze"], train)
        comparator = preflight["origin_support"][str(origin)]["original_rnbr_comparator"]
        _validate_comparator_alignment(origin, geometries[origin], original[origin])
        if (comparator.get("npz_sha256") != original[origin]["npz_sha256"]
                or comparator.get("receipt_sha256") != original[origin]["receipt_sha256"]):
            raise ImpactMatchRunError(f"Original RNBR E{origin} comparator no longer matches the preflight")

    donor_cache: dict[int, dict[str, np.ndarray]] = {}
    results: list[dict[str, Any]] = []
    for origin, arm in ((67, "observed"), (67, "shuffled"), (75, "observed"), (75, "shuffled")):
        fit_id = f"RNBR_IMPACT_MATCH_V2_1_E{origin}_{arm.upper()}"
        if fit_id not in FIT_IDS:
            raise ImpactMatchRunError(f"Internal v2 fit ID does not match frozen allowlist: {fit_id}")
        events = _ledger_events()
        if _resume_saved_fit(fit_id, freeze, events):
            results.append({"fit_id": fit_id, "status": "FIT_SAVED_VERIFIED_RESUME"})
            if arm == "observed":
                loaded, _ = impact.load_bundle(_fit_dir(fit_id), fit_id=fit_id,
                                               freeze_signature=freeze["freeze_signature"])
                donor_cache[origin] = {key: loaded[key] for key in (
                    "donor_rows", "donor_ids", "donor_cycles", "donor_impact_load",
                    "donor_source_cutoffs", "donor_r06_predictions", "donor_residuals",
                    "donor_residuals_used", "donor_targets") if key in loaded}
            continue
        _resource_check()
        _start_fit(fit_id, origin, arm, freeze, source_commit)
        try:
            if origin not in donor_cache or "donor_targets" not in donor_cache[origin]:
                donor_cache[origin] = _target_bundle_inputs(
                    origin, freeze, geometries[origin], train, base["r06_cache"]
                )
            arrays, metadata = _fit_arrays(origin, arm, geometries[origin], donor_cache[origin], source_commit)
            metadata.update({
                "fit_recipe": "RNBR_IMPACT_MATCH_V2_1 frozen protocol",
                "resource_snapshot": _resource_check(),
                "preflight_coverage": preflight["origin_support"][str(origin)]["views"],
                "full_a02_model_or_pickle_loaded": False,
                "regression_target_values_used_for_tfidf": False,
            })
            receipt = impact.persist_bundle_once(
                _fit_dir(fit_id), fit_id=fit_id, freeze_signature=freeze["freeze_signature"],
                arrays=arrays, metadata=metadata,
            )
            _commit_fit_saved(fit_id, origin, receipt)
            results.append({"fit_id": fit_id, "status": "FIT_SAVED",
                            "receipt_sha256": _sha_file(_fit_dir(fit_id) / "receipt.json"),
                            "npz_sha256": _sha_file(_fit_dir(fit_id) / "arrays.npz"),
                            "resource_snapshot": metadata["resource_snapshot"]})
        except Exception as exc:
            events = _ledger_events()
            terminal = [event for event in events if event.get("fit_id") == fit_id
                        and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
            if not terminal:
                _append_ledger({
                    "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                    "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                    "fit_id": fit_id, "cutoff": origin, "receipt_sha256": None,
                    "fit_kind": "TARGET_MEMORY_CONSTRUCTION",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            raise

    if len(results) != 4:
        raise ImpactMatchRunError(f"Expected exactly four v2 memory outputs, got {len(results)}")
    from scripts import optimization_v3_rnbr_impact_match_verify_v2 as verify  # noqa: PLC0415
    verification = verify.verify_all()
    return {"status": "FOUR_FIT_DIAGNOSTIC_COMPLETE", "preflight_status": preflight["status"],
            "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
            "pre_fit_resources": resources, "ledger": _ledger_snapshot(),
            "fit_results": results, "verification": verification}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RNBR impact-match v2 target-free replay and diagnostic pilot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-replay", action="store_true", help="Freeze target-free replay inputs and synthetic QA")
    group.add_argument("--run-replays", action="store_true", help="Run/resume the exactly two authorized text-only TF-IDF replays")
    group.add_argument("--prepare-geometry", action="store_true", help="Rebuild clean preflight geometry from replay vectorizers")
    group.add_argument("--run", action="store_true", help="Run/resume exactly four authorized memory fits")
    group.add_argument("--verify", action="store_true", help="Independently verify completed diagnostic outputs")
    group.add_argument("--synthetic", action="store_true", help="Run only isolated synthetic end-to-end QA")
    args = parser.parse_args()
    if args.synthetic:
        result = synthetic_qa()
    elif args.prepare_replay:
        result = prepare_replay()
    elif args.run_replays:
        result = run_replays()
    elif args.prepare_geometry:
        result = prepare_geometry()
    elif args.verify:
        from scripts import optimization_v3_rnbr_impact_match_verify_v2 as verify
        result = verify.verify_all()
    else:
        result = execute()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

