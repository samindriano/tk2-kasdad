"""Profile frozen known-bigram positional support without reading labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.optimization_v3_bigram_position_readiness_v1 import build_bigram_blocks


ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path("data/raw/classification/Train.csv")
CACHE = Path("experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1")
OUTPUT = Path("reports/optimization_v3/bigram_position_readiness_v1/SUPPORT_PROFILE.json")
ORIGINS = (67, 75)
SUPPORT_SCREEN = {
    "min_fit_known_pair_documents": 2_000,
    "min_query_known_pair_documents": 1_000,
    "min_known_pair_documents_per_query_cycle": 100,
    "min_query_known_pair_occurrence_coverage": 0.20,
    "min_query_descriptor_change_rate_among_known_documents": 0.30,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csr_bytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _quantiles(values: np.ndarray) -> dict[str, int | float | None]:
    if values.size == 0:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "min": int(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(np.max(values)),
    }


def _split_profile(
    name: str, indices: np.ndarray, cycles: np.ndarray, present: np.ndarray,
    matrices: dict[str, Any], metadata: dict[str, Any],
) -> dict[str, Any]:
    selected_present = indices[present[indices]]
    selected_pairs = indices[metadata["pair_count"][indices] > 0]
    selected_known = indices[metadata["known_occurrence_count"][indices] > 0]
    total_pairs = int(metadata["pair_count"][selected_present].sum())
    total_known = int(metadata["known_occurrence_count"][selected_present].sum())
    known_types = set()
    for row in selected_present:
        known_types.update(metadata["known_column_indices"][int(row)])
    changed_rows = int(metadata["descriptor_changed_under_shuffle"][selected_known].sum())
    summary: dict[str, Any] = {
        "name": name,
        "rows": int(len(indices)),
        "stream_present_rows": int(len(selected_present)),
        "stream_missing_rows": int(len(indices) - len(selected_present)),
        "present_token_count_distribution": _quantiles(metadata["token_count"][selected_present]),
        "present_adjacent_pair_occurrences": total_pairs,
        "known_bigram_occurrences": total_known,
        "known_occurrence_coverage": (total_known / total_pairs) if total_pairs else None,
        "pair_bearing_present_documents": int(np.intersect1d(selected_pairs, selected_present).size),
        "known_pair_present_documents": int(len(selected_known)),
        "known_document_coverage_of_present": (len(selected_known) / len(selected_present)) if len(selected_present) else None,
        "known_document_coverage_of_pair_bearing": (
            len(np.intersect1d(selected_known, selected_pairs)) / len(np.intersect1d(selected_pairs, selected_present))
            if len(np.intersect1d(selected_pairs, selected_present)) else None
        ),
        "positive_known_count_distribution": _quantiles(metadata["known_occurrence_count"][selected_known]),
        "observed_known_bigram_types": int(len(known_types)),
        "frozen_bigram_vocabulary_size": int(metadata["bigram_vocabulary_size"]),
        "observed_known_bigram_type_coverage": (
            len(known_types) / metadata["bigram_vocabulary_size"]
            if metadata["bigram_vocabulary_size"] else None
        ),
        "positional_descriptor_changed_rows_under_pair_shuffle": changed_rows,
        "descriptor_change_rate_among_known_documents": (
            changed_rows / len(selected_known) if len(selected_known) else None
        ),
        "csr_blocks": {},
        "cycle_support": [],
    }
    for block_name, matrix in matrices.items():
        block = matrix[indices]
        summary["csr_blocks"][block_name] = {
            "shape": [int(block.shape[0]), int(block.shape[1])],
            "nnz": int(block.nnz),
            "csr_bytes": _csr_bytes(block),
        }

    for cycle in sorted(int(c) for c in np.unique(cycles[indices])):
        cycle_rows = indices[cycles[indices] == cycle]
        cycle_present = cycle_rows[present[cycle_rows]]
        cycle_pair_rows = cycle_rows[metadata["pair_count"][cycle_rows] > 0]
        cycle_known_rows = cycle_rows[metadata["known_occurrence_count"][cycle_rows] > 0]
        cycle_pairs = int(metadata["pair_count"][cycle_present].sum())
        cycle_known = int(metadata["known_occurrence_count"][cycle_present].sum())
        cycle_terms = set()
        for row in cycle_present:
            cycle_terms.update(metadata["known_column_indices"][int(row)])
        summary["cycle_support"].append({
            "cycle_code": cycle,
            "rows": int(len(cycle_rows)),
            "stream_present_rows": int(len(cycle_present)),
            "pair_bearing_present_documents": int(len(np.intersect1d(cycle_pair_rows, cycle_present))),
            "known_pair_documents": int(len(np.intersect1d(cycle_known_rows, cycle_present))),
            "known_bigram_occurrences": cycle_known,
            "known_occurrence_coverage": (cycle_known / cycle_pairs) if cycle_pairs else None,
            "observed_known_bigram_types": int(len(cycle_terms)),
            "descriptor_changed_known_documents": int(
                metadata["descriptor_changed_under_shuffle"][np.intersect1d(cycle_known_rows, cycle_present)].sum()
            ),
        })
    return summary


def _load_frozen_vectorizer(origin: int, train_sha256: str) -> tuple[Any, dict[str, Any]]:
    unit_name = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF"
    artifact = ROOT / CACHE / "units" / f"{unit_name}.joblib"
    receipt_path = ROOT / CACHE / "receipts" / f"{unit_name}.json"
    freeze_path = ROOT / CACHE / "FREEZE.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual_artifact_sha = sha256_file(artifact)
    if actual_artifact_sha != receipt["artifact_sha256"]:
        raise RuntimeError(f"Frozen E{origin} vectorizer artifact hash mismatch")
    if receipt["input_sha256"] != train_sha256 or freeze["input_sha256"] != train_sha256:
        raise RuntimeError(f"Frozen E{origin} input identity does not match official Train")
    if receipt["origin"] != origin or receipt["fit_population"]["cycle_max"] != origin:
        raise RuntimeError(f"Frozen E{origin} vectorizer fit population has the wrong cutoff")
    vectorizer = joblib.load(artifact)
    params = vectorizer.get_params(deep=False)
    expected = {
        "analyzer": "word", "ngram_range": (1, 2), "strip_accents": "unicode",
        "lowercase": True, "token_pattern": r"(?u)\b\w\w+\b",
    }
    for key, value in expected.items():
        actual = params[key]
        if key == "ngram_range":
            actual = tuple(actual)
        if actual != value:
            raise RuntimeError(f"Frozen E{origin} tokenizer mismatch for {key}: {actual!r}")
    if len(vectorizer.vocabulary_) != len(vectorizer.idf_):
        raise RuntimeError(f"Frozen E{origin} vocabulary and IDF lengths disagree")
    artifact_record = {
        "origin": origin,
        "artifact_relative_path": str(artifact.relative_to(ROOT)),
        "artifact_sha256": actual_artifact_sha,
        "receipt_sha256": sha256_file(receipt_path),
        "freeze_signature": receipt["freeze_signature"],
        "source_commit": receipt["source_commit"],
        "fit_population": receipt["fit_population"],
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "bigram_vocabulary_size": int(sum(" " in term for term in vectorizer.vocabulary_)),
        "idf_size": int(len(vectorizer.idf_)),
        "tfidf_parameters": {
            "analyzer": params["analyzer"],
            "ngram_range": list(params["ngram_range"]),
            "strip_accents": params["strip_accents"],
            "lowercase": params["lowercase"],
            "token_pattern": params["token_pattern"],
            "max_features": params["max_features"],
        },
        "receipt_fit_metadata": receipt["fit_metadata"],
    }
    return vectorizer, artifact_record


def run_profile(train_path: Path, output_path: Path) -> dict[str, Any]:
    train_path = train_path.resolve()
    input_sha = sha256_file(train_path)
    # Intentionally reads only split metadata and the text feature. In
    # particular this usecols list excludes artifact_signal and threat_rank.
    frame = pd.read_csv(train_path, usecols=["cycle_code", "signal_stream"])
    if list(frame.columns) != ["cycle_code", "signal_stream"]:
        raise RuntimeError(f"Unexpected selected columns: {frame.columns.tolist()}")
    cycles_numeric = pd.to_numeric(frame["cycle_code"], errors="raise")
    if not np.all(np.isfinite(cycles_numeric)) or not np.all(cycles_numeric == np.floor(cycles_numeric)):
        raise RuntimeError("cycle_code must contain finite integer-valued split metadata")
    cycles = cycles_numeric.to_numpy(dtype=np.int16)
    expected_cycles = np.arange(47, 80, dtype=np.int16)
    observed_cycles = np.unique(cycles)
    if not np.array_equal(observed_cycles, expected_cycles):
        raise RuntimeError(f"Train cycles are incomplete or unexpected: {observed_cycles.tolist()}")
    raw_streams = frame["signal_stream"].tolist()
    documents = ["" if pd.isna(value) else str(value) for value in raw_streams]
    present = np.asarray([bool(document.strip()) for document in documents], dtype=bool)

    origins: list[dict[str, Any]] = []
    for origin in ORIGINS:
        vectorizer, artifact_record = _load_frozen_vectorizer(origin, input_sha)
        matrices, metadata = build_bigram_blocks(documents, vectorizer)
        fit_indices = np.flatnonzero(cycles <= origin)
        query_indices = np.flatnonzero(cycles > origin)
        fit = _split_profile("fit", fit_indices, cycles, present, matrices, metadata)
        query = _split_profile("query", query_indices, cycles, present, matrices, metadata)
        sparse_bytes = {
            name: int(_csr_bytes(matrix)) for name, matrix in matrices.items()
        }
        origin_record = {
            "origin": origin,
            "frozen_vectorizer": artifact_record,
            "fit": fit,
            "query": query,
            "resident_sparse_block_bytes_for_all_train_rows": sparse_bytes,
            "support_screen": {
                "fit_known_documents_pass": (
                    fit["known_pair_present_documents"] >= SUPPORT_SCREEN["min_fit_known_pair_documents"]
                ),
                "query_known_documents_pass": (
                    query["known_pair_present_documents"] >= SUPPORT_SCREEN["min_query_known_pair_documents"]
                ),
                "all_query_cycles_pass": all(
                    row["known_pair_documents"] >= SUPPORT_SCREEN["min_known_pair_documents_per_query_cycle"]
                    for row in query["cycle_support"]
                ),
                "query_occurrence_coverage_pass": (
                    (query["known_occurrence_coverage"] or 0.0)
                    >= SUPPORT_SCREEN["min_query_known_pair_occurrence_coverage"]
                ),
                "query_descriptor_change_pass": (
                    (query["descriptor_change_rate_among_known_documents"] or 0.0)
                    >= SUPPORT_SCREEN["min_query_descriptor_change_rate_among_known_documents"]
                ),
            },
        }
        origins.append(origin_record)

    all_pass = all(all(origin["support_screen"].values()) for origin in origins)
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    script_path = ROOT / Path(__file__).resolve().relative_to(ROOT)
    module_path = ROOT / "src/models/optimization_v3_bigram_position_readiness_v1.py"
    profile = {
        "schema_version": 1,
        "audit_id": "CLS_BIGRAM_POSITION_READINESS_V1",
        "status": "SUPPORT_READY_FOR_REVIEW" if all_pass else "SUPPORT_INSUFFICIENT_OR_MIXED",
        "source_commit_before_local_changes": source_commit,
        "input": {
            "relative_path": str(train_path.relative_to(ROOT)),
            "sha256": input_sha,
            "rows": int(len(frame)),
            "selected_columns_only": frame.columns.tolist(),
            "target_columns_read": [],
            "test_or_sample_submission_opened": False,
            "cycles": [int(c) for c in observed_cycles],
        },
        "source_files": {
            "script_sha256": sha256_file(script_path),
            "builder_sha256": sha256_file(module_path),
        },
        "profile_definition": {
            "stream_present": "non-null signal_stream with at least one non-whitespace character",
            "tokens": "frozen TfidfVectorizer Unicode strip_accents/lowercase/preprocessor/tokenizer",
            "pair_sequence": "all adjacent normalized token pairs; pair count is n_token - 1",
            "position_bin": "floor(3*j/n_pair), zero-based j, over every pair including OOV pairs",
            "known_pair": "exact phrase exists in the frozen prefix bigram vocabulary",
            "null": "deterministic_order_shuffle(pair-string sequence, seed=42); bag and bin sizes preserved",
            "feature_values": "(1 + log(count)) * frozen IDF; whole block L2-normalized once per row",
        },
        "support_screen": {
            "purpose": "target-free operational sufficiency screen for deciding whether one temporal pilot is feasible; not a power guarantee or outcome gate",
            "thresholds": SUPPORT_SCREEN,
            "both_origins_pass": bool(all_pass),
        },
        "origins": origins,
        "fit_or_scoring_performed": False,
        "labels_opened": False,
        "test_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / TRAIN)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    profile = run_profile(args.train, args.output)
    print(json.dumps({
        "audit_id": profile["audit_id"],
        "status": profile["status"],
        "input_sha256": profile["input"]["sha256"],
        "origin_screens": {str(row["origin"]): row["support_screen"] for row in profile["origins"]},
        "output": str(args.output.resolve()),
        "fit_or_scoring_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
