"""Versioned safe prelabel/score entrypoint with literal scalar W replay.

The frozen verifier is retained. This module only adapts two legacy boundaries:
the signed Unicode-safe export files and the verifier's vectorized occurrence
mean, whose last-bit reduction order differs from the frozen scalar recipe.
It independently rebuilds W from documents/vectorizer state and lets the
unchanged verifier perform every lineage, prediction, routing, and gate check.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_positional_moments_v1 as moments
from scripts import optimization_v3_positional_moments_v1_safe_export as safe_export
from scripts import verify_optimization_v3_positional_moments_v1 as verifier
from src.models.positional_text_features import deterministic_order_shuffle


ADAPTER_ID = "POS_MOMENTS_V1_SCALAR_W_REPLAY_V2"
PRELABEL_REPORT = moments.REPORT_ROOT / "SAFE_EXPORT_PRELABEL_VERIFICATION_V2.json"
PRELABEL_RUN_REPORT = moments.RUN_ROOT / "SAFE_EXPORT_PRELABEL_VERIFICATION_V2.json"
PRELABEL_FAILURE = moments.REPORT_ROOT / "SAFE_EXPORT_PRELABEL_FAILURE_V2.json"
PRELABEL_RUN_FAILURE = moments.RUN_ROOT / "SAFE_EXPORT_PRELABEL_FAILURE_V2.json"
SYNTHETIC_REPORT = moments.REPORT_ROOT / "SYNTHETIC_PRELABEL_REPLAY_V2.json"
SYNTHETIC_RUN_REPORT = moments.RUN_ROOT / "SYNTHETIC_PRELABEL_REPLAY_V2.json"


class PrelabelV2Error(RuntimeError):
    """The exact scalar-replay or safe prelabel contract failed closed."""


def rebuild_moments_scalar_order(
    documents: Sequence[str], vectorizer: Any, *, shuffled: bool,
) -> sparse.csr_matrix:
    """Independent scalar replay of the frozen W feature definition.

    Occurrence means accumulate each token's h1/h2 values in document order,
    then divide once by count. This matches the frozen recipe's specified
    scalar reduction order without calling the production matrix builder.
    """
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise PrelabelV2Error("A fitted frozen prefix vectorizer is required")
    vocabulary = {str(token): int(index) for token, index in vectorizer.vocabulary_.items()}
    unigram_original = {token: index for token, index in vocabulary.items() if " " not in token}
    unigram = {token: compact for compact, token in enumerate(
        sorted(unigram_original, key=unigram_original.__getitem__))}
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    preprocessor, tokenizer = vectorizer.build_preprocessor(), vectorizer.build_tokenizer()
    analyzer = vectorizer.build_analyzer()
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    width = len(unigram)

    for row, value in enumerate(documents):
        document = str(value).strip()
        tokens = tuple(tokenizer(preprocessor(document)))
        analyzed = analyzer(document)
        if tuple(analyzed[:len(tokens)]) != tokens:
            raise PrelabelV2Error("Analyzer/tokenizer unigram ordering differs")
        if shuffled:
            ordered = tuple(deterministic_order_shuffle(tokens, seed=42)["tokens"])
            if len(ordered) != len(tokens) or Counter(ordered) != Counter(tokens):
                raise PrelabelV2Error("Deterministic shuffle changed token length or bag")
        else:
            ordered = tokens

        n = len(ordered)
        if n <= 1:
            h1 = np.zeros(n, dtype=np.float64)
            h2 = np.zeros(n, dtype=np.float64)
        else:
            positions = np.arange(n, dtype=np.float64)
            u = 2.0 * positions / float(n - 1) - 1.0
            h1 = u
            raw_h2 = (3.0 * np.square(u) - 1.0) / 2.0
            h2 = raw_h2 - float(raw_h2.mean())

        counts = Counter(ordered)
        bag_raw: dict[str, float] = {}
        moment_sums: dict[str, list[float]] = {}
        for token, count in counts.items():
            if token in unigram:
                bag_raw[token] = (1.0 + math.log(int(count))) * float(idf[unigram_original[token]])
                moment_sums[token] = [0.0, 0.0, 0.0]
        for position, token in enumerate(ordered):
            sums = moment_sums.get(token)
            if sums is not None:
                sums[0] += float(h1[position])
                sums[1] += float(h2[position])
                sums[2] += 1.0
        bag_norm = math.sqrt(sum(value * value for value in bag_raw.values()))
        if bag_norm == 0.0:
            continue

        row_values: list[tuple[int, float]] = []
        for token, raw_value in bag_raw.items():
            base = raw_value / bag_norm
            count = moment_sums[token][2]
            mean1 = moment_sums[token][0] / count if count else 0.0
            mean2 = moment_sums[token][1] / count if count else 0.0
            token_index = unigram[token]
            row_values.extend(((token_index, base), (width + token_index, base * mean1),
                               (2 * width + token_index, base * mean2)))
        joint_norm = math.sqrt(sum(value * value for _, value in row_values))
        if not math.isfinite(joint_norm) or joint_norm <= 0.0:
            raise PrelabelV2Error("Nonempty known-token row has invalid joint norm")
        for column, value in row_values:
            normalized = value / joint_norm
            if normalized != 0.0:
                row_idx.append(row)
                col_idx.append(column)
                data.append(normalized)

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64))),
        shape=(len(documents), 3 * width), dtype=np.float64,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise PrelabelV2Error("Scalar replay generated non-finite W values")
    return matrix


@contextmanager
def _scalar_verifier_context() -> Iterator[None]:
    original = verifier._rebuild_moments
    verifier._rebuild_moments = rebuild_moments_scalar_order
    try:
        yield
    finally:
        verifier._rebuild_moments = original


def run_prelabel() -> dict[str, Any]:
    manifest, checked_exports = safe_export.validate_all_exports()
    freeze_data = moments._load_freeze()
    ready = safe_export._safe_ready(manifest)
    with safe_export._installed_read_adapters(ready, freeze_data), _scalar_verifier_context():
        result = verifier.verify_prelabel_outputs(moments.ROOT, moments.RUN_ROOT, freeze_data)
    if result.get("status") != "PASS" or result.get("query_labels_read") is not False:
        raise PrelabelV2Error("Unchanged frozen prelabel verifier did not return label-closed PASS")
    result.update({
        "verification_method": "unchanged frozen verifier with signed safe-export reader and independent scalar W replay",
        "adapter_id": ADAPTER_ID,
        "adapter_source_path": Path(__file__).resolve().relative_to(moments.ROOT).as_posix(),
        "adapter_source_sha256": moments.sha256_file(Path(__file__)),
        "frozen_verifier_source_sha256": freeze_data["source_hashes"]["verifier"],
        "safe_export_manifest_signature": manifest["manifest_signature"],
        "safe_export_manifest_sha256": moments.sha256_file(safe_export.MANIFEST_REPORT),
        "safe_export_origins": checked_exports,
        "rebuild_algorithm": "independent scalar accumulation in token occurrence order; exact frozen feature math, no call to production builder",
        "prior_failure_receipts": {
            "parent_schema_and_object_dtype": moments.sha256_file(
                moments.REPORT_ROOT / "PRELABEL_SCHEMA_ADAPTER_FAILURE_V1.json"),
            "vectorized_rebuild_hash_mismatch": moments.sha256_file(
                moments.REPORT_ROOT / "SAFE_EXPORT_PRELABEL_FAILURE_V1.json"),
        },
        "query_labels_read": False, "test_or_sample_access": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return result


def run_score() -> dict[str, Any]:
    manifest, _ = safe_export.validate_all_exports()
    freeze_data = moments._load_freeze()
    prelabel = moments._read_json(moments.RUN_ROOT / "PRELABEL_VERIFICATION.json")
    if (prelabel.get("status") != "PASS" or prelabel.get("query_labels_read") is not False
            or prelabel.get("freeze_signature") != freeze_data["freeze_signature"]
            or prelabel.get("adapter_id") != ADAPTER_ID
            or prelabel.get("adapter_source_sha256") != moments.sha256_file(Path(__file__))
            or prelabel.get("safe_export_manifest_signature") != manifest["manifest_signature"]
            or prelabel.get("safe_export_manifest_sha256") != moments.sha256_file(safe_export.MANIFEST_REPORT)
            or prelabel.get("predictions_ready_sha256") != moments.sha256_file(safe_export.PRELABEL_RUN_READY)):
        raise PrelabelV2Error("Outcome metrics remain closed until scalar-replay safe prelabel PASS")
    ready = safe_export._safe_ready(manifest)
    with safe_export._installed_read_adapters(ready, freeze_data):
        return moments.score()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _synthetic_documents() -> list[str]:
    alternating = "alpha beta " * 300
    repeated_long = "gamma " * 401
    return [
        "", "alpha", "alpha alpha", "beta alpha beta alpha beta",
        alternating + "alpha", repeated_long,
        "café alpha " * 97 + "omega_oov " * 3,
        "delta " + "alpha beta gamma delta " * 41 + "delta",
    ]


def synthetic_prelabel_rebuild_check() -> dict[str, Any]:
    """Exercise stress documents through W reconstruction and fixed prelabel policy."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=1,
                                 max_df=1.0, ngram_range=(1, 2), dtype=np.float64)
    vectorizer.fit(["alpha beta gamma delta cafe epsilon", "alpha beta delta", "gamma cafe delta"])
    docs = _synthetic_documents()
    checks = []
    for shuffled in (False, True):
        W = rebuild_moments_scalar_order(docs, vectorizer, shuffled=shuffled)
        gamma = np.linspace(-0.02, 0.03, W.shape[1], dtype=np.float64)
        z0 = np.linspace(-0.7, 0.8, W.shape[0], dtype=np.float64)
        p_a02 = np.linspace(0.15, 0.85, W.shape[0], dtype=np.float64)
        p_old = np.linspace(0.05, 0.95, W.shape[0], dtype=np.float64)
        present = np.asarray([True, True, False, True, False, True, True, False])
        q = expit(z0 + np.asarray(W @ gamma).reshape(-1))
        candidate = np.where(present, 0.5 * q + 0.5 * p_a02, p_old)
        mix50 = np.where(present, 0.5 * p_old + 0.5 * p_a02, p_old)
        check = verifier.verify_synthetic_prelabel(W, gamma, z0, p_a02, p_old,
                                                   present, q, candidate, mix50)
        checks.append({"shuffled": shuffled, **check})
    return {"status": "PASS", "documents": len(docs),
            "max_token_count": max(len(vectorizer.build_tokenizer()(vectorizer.build_preprocessor()(d.strip()))) for d in docs),
            "features": int(3 * sum(" " not in token for token in vectorizer.vocabulary_)),
            "checks": checks, "query_labels_read": False, "test_or_sample_access": False}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"synthetic", "prelabel", "score"}:
        raise SystemExit("usage: optimization_v3_positional_moments_v1_prelabel_v2.py {synthetic|prelabel|score}")
    command = sys.argv[1]
    if command == "synthetic":
        result = synthetic_prelabel_rebuild_check()
        result.update({
            "adapter_id": ADAPTER_ID,
            "adapter_source_path": Path(__file__).resolve().relative_to(moments.ROOT).as_posix(),
            "adapter_source_sha256": moments.sha256_file(Path(__file__)),
            "frozen_verifier_source_sha256": moments._load_freeze()["source_hashes"]["verifier"],
            "rebuild_algorithm": "independent scalar accumulation in token occurrence order",
        })
        _atomic_json(SYNTHETIC_REPORT, result)
        _atomic_json(SYNTHETIC_RUN_REPORT, result)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 1
    if command == "prelabel":
        try:
            result = run_prelabel()
        except Exception as exc:
            failure = {
                "schema_version": 1, "adapter_id": ADAPTER_ID,
                "status": "FAIL_PRELABEL; SCORE_REMAINS_CLOSED",
                "error_type": type(exc).__name__, "error": str(exc),
                "freeze_signature": moments._load_freeze()["freeze_signature"],
                "safe_export_manifest_signature": safe_export._load_signed_manifest().get("manifest_signature"),
                "query_labels_read": False, "test_or_sample_access": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(PRELABEL_FAILURE, failure)
            _atomic_json(PRELABEL_RUN_FAILURE, failure)
            raise
        moments._atomic_json(moments.REPORT_ROOT / "PRELABEL_VERIFICATION.json", result)
        moments._atomic_json(moments.RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
        _atomic_json(PRELABEL_REPORT, result)
        _atomic_json(PRELABEL_RUN_REPORT, result)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
        return 0
    result = run_score()
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
