"""Deterministic anonymous-token relabeling for Research Next N06.

The original R06 sparse-L1 model is imported unchanged. This adapter changes
only the characters inside declared q/z token payloads before that model sees
the signal stream; it never learns a vocabulary or lookup from future rows.
"""
from __future__ import annotations

from functools import lru_cache
from collections import Counter
import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1_r06 import (
    FEATURE_NAMES,
    R06CharacterSparseL1Hybrid,
    compose_r06_prediction,
)
from src.models.regression_research_text import _text_documents


SEEDS = (42, 137)
PREFIX_BITS = {"q": 28, "z": 24}
FEISTEL_ROUNDS = 6
ROUND_DOMAIN = "research_next_v1"
TOKEN_PATTERN = r"(?<!\w)(?P<token>q[0-9a-f]{7}|z[0-9a-f]{6})(?!\w)"
TOKEN_RE = re.compile(TOKEN_PATTERN)
WORD_RE = re.compile(r"(?u)\b\w+\b")


def _round_f(prefix: str, seed: int, round_index: int, right: int, half_bits: int) -> int:
    message = f"{ROUND_DOMAIN}|{int(seed)}|{prefix}|{int(round_index)}|{int(right)}".encode("utf-8")
    digest_integer = int.from_bytes(hashlib.sha256(message).digest(), byteorder="big", signed=False)
    return digest_integer & ((1 << half_bits) - 1)


def map_payload(prefix: str, payload: int, seed: int) -> int:
    """Six-round balanced Feistel permutation over one declared token domain."""
    if prefix not in PREFIX_BITS:
        raise ValueError(f"Unsupported N06 token prefix: {prefix!r}")
    total_bits = PREFIX_BITS[prefix]
    half_bits = total_bits // 2
    mask = (1 << half_bits) - 1
    value = int(payload)
    if value < 0 or value >= (1 << total_bits):
        raise ValueError(f"N06 {prefix} payload is outside its {total_bits}-bit domain")
    left, right = value >> half_bits, value & mask
    for round_index in range(FEISTEL_ROUNDS):
        left, right = right, left ^ _round_f(prefix, seed, round_index, right, half_bits)
    return (left << half_bits) | right


def unmap_payload(prefix: str, payload: int, seed: int) -> int:
    """Inverse of :func:`map_payload`, using the same frozen round function."""
    if prefix not in PREFIX_BITS:
        raise ValueError(f"Unsupported N06 token prefix: {prefix!r}")
    total_bits = PREFIX_BITS[prefix]
    half_bits = total_bits // 2
    mask = (1 << half_bits) - 1
    value = int(payload)
    if value < 0 or value >= (1 << total_bits):
        raise ValueError(f"N06 {prefix} payload is outside its {total_bits}-bit domain")
    left, right = value >> half_bits, value & mask
    for round_index in reversed(range(FEISTEL_ROUNDS)):
        left, right = right ^ _round_f(prefix, seed, round_index, left, half_bits), left
    return (left << half_bits) | right


@lru_cache(maxsize=250_000)
def _map_token_cached(token: str, seed: int) -> str:
    match = TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"Unexpected N06 token shape: {token!r}")
    prefix = token[0]
    payload_text = token[1:]
    payload = int(payload_text, 16)
    mapped = map_payload(prefix, payload, seed)
    return prefix + format(mapped, f"0{len(payload_text)}x")


def map_token(token: str, seed: int) -> str:
    """Map one complete token; memoization only caches a pure deterministic result."""
    if int(seed) not in SEEDS:
        raise ValueError(f"N06 mapping seed must be one of {SEEDS}")
    return _map_token_cached(str(token), int(seed))


def unmap_token(token: str, seed: int) -> str:
    if int(seed) not in SEEDS:
        raise ValueError(f"N06 mapping seed must be one of {SEEDS}")
    match = TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"Unexpected N06 token shape: {token!r}")
    prefix, payload_text = token[0], token[1:]
    payload = int(payload_text, 16)
    restored = unmap_payload(prefix, payload, seed)
    return prefix + format(restored, f"0{len(payload_text)}x")


def rename_stream(value: Any, seed: int) -> tuple[Any, int]:
    """Rename supported tokens while preserving all other characters verbatim."""
    missing = pd.isna(value)
    if isinstance(missing, (np.ndarray, pd.Series)):
        raise TypeError("N06 signal_stream values must be scalar")
    if bool(missing):
        return value, 0
    raw = str(value)
    if not raw.strip():
        return value, 0
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        original = match.group("token")
        mapped = map_token(original, seed)
        changed += 1
        return mapped

    renamed = TOKEN_RE.sub(replace, raw)
    return renamed, changed


def rename_series(series: pd.Series, seed: int) -> tuple[pd.Series, dict[str, int]]:
    if not isinstance(series, pd.Series):
        raise TypeError("N06 stream transformation requires a pandas Series")
    if int(seed) not in SEEDS:
        raise ValueError(f"N06 mapping seed must be one of {SEEDS}")
    values = []
    occurrences = 0
    present_rows = 0
    for value in series.to_numpy(dtype=object):
        renamed, count = rename_stream(value, seed)
        values.append(renamed)
        occurrences += count
        if not bool(pd.isna(value)) and bool(str(value).strip()):
            present_rows += 1
    return pd.Series(values, index=series.index, name=series.name, dtype=object), {
        "token_occurrences_renamed": int(occurrences),
        "present_rows": int(present_rows),
        "row_count": int(len(series)),
    }


def rename_features(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or set(frame.columns) != set(FEATURE_NAMES) or len(frame.columns) != len(FEATURE_NAMES)):
        raise ValueError("N06 requires the exact frozen original R06 feature allowlist")
    selected = frame.loc[:, list(FEATURE_NAMES)].copy()
    selected["signal_stream"], audit = rename_series(selected["signal_stream"], seed)
    return selected, audit


def inspect_token_schema(series: pd.Series) -> dict[str, Any]:
    """Count all word tokens and fail closed on any shape outside q7/z6."""
    counts = {"q_7hex": 0, "z_6hex": 0}
    unique: dict[str, set[str]] = {"q": set(), "z": set()}
    unexpected_sample_counts: Counter[str] = Counter()
    unexpected_occurrences = 0
    unexpected_examples: list[str] = []
    missing_rows = 0
    for value in series.to_numpy(dtype=object):
        if bool(pd.isna(value)) or not str(value).strip():
            missing_rows += 1
            continue
        text = str(value).strip()
        for match in WORD_RE.finditer(text):
            token = match.group(0)
            if TOKEN_RE.fullmatch(token):
                prefix = token[0]
                counts["q_7hex" if prefix == "q" else "z_6hex"] += 1
                unique[prefix].add(token)
            else:
                unexpected_occurrences += 1
                if token in unexpected_sample_counts or len(unexpected_sample_counts) < 20:
                    unexpected_sample_counts[token] += 1
                if len(unexpected_examples) < 20 and token not in unexpected_examples:
                    unexpected_examples.append(token)
    return {
        "status": "PASS" if not unexpected_occurrences else "SKIPPED_TOKEN_SCHEMA",
        "token_occurrences": counts,
        "unique_tokens": {prefix: len(tokens) for prefix, tokens in unique.items()},
        "missing_or_blank_rows": int(missing_rows),
        "unexpected_token_occurrences": int(unexpected_occurrences),
        "unexpected_token_shape_sample_counts": dict(sorted(unexpected_sample_counts.items())),
        "unexpected_token_shape_sample_is_capped_at_unique_20": True,
        "unexpected_examples": unexpected_examples,
        "unique_tokens_by_prefix": unique,
    }


def verify_observed_bijection(schema: dict[str, Any], seed: int) -> dict[str, Any]:
    if schema.get("status") != "PASS":
        raise ValueError("N06 cannot verify a bijection until token schema passes")
    checked = {}
    for prefix in ("q", "z"):
        tokens = sorted(schema["unique_tokens_by_prefix"][prefix])
        mapped = [map_token(token, seed) for token in tokens]
        if len(mapped) != len(set(mapped)):
            raise RuntimeError(f"N06 mapping collision among observed {prefix} tokens for seed {seed}")
        if any(mapped_token[0] != prefix or len(mapped_token) != len(token)
               for token, mapped_token in zip(sorted(tokens), mapped)):
            raise RuntimeError(f"N06 mapping changed prefix/length for seed {seed}")
        for token, mapped_token in zip(tokens, mapped):
            if unmap_token(mapped_token, seed) != token:
                raise RuntimeError(f"N06 inverse mapping failed for seed {seed}")
        checked[prefix] = {
            "unique_observed_tokens": len(tokens),
            "mapped_distinct_tokens": len(set(mapped)),
            "collision_count": len(mapped) - len(set(mapped)),
            "prefix_and_length_preserved": True,
            "inverse_roundtrip_all_observed_tokens": True,
        }
    return {"seed": int(seed), "status": "PASS", "by_prefix": checked}


def _document_sha256(documents: list[str]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        encoded = document.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def fit_predict_prepared(renamed_fit: pd.DataFrame, fit_y: Any,
                         renamed_score: pd.DataFrame, *, seed: int,
                         baseline_legacy: Any, score_cycles: Any,
                         fit_map_audit: dict[str, int],
                         score_map_audit: dict[str, int],
                         original_fit_X: pd.DataFrame,
                         original_score_X: pd.DataFrame) -> dict[str, Any]:
    """Fit exact R06 from already recoded inputs and retain mapping receipts."""
    if int(seed) not in SEEDS:
        raise ValueError(f"N06 mapping seed must be one of {SEEDS}")
    for value in (renamed_fit, renamed_score, original_fit_X, original_score_X):
        if (not isinstance(value, pd.DataFrame) or value.columns.has_duplicates
                or set(value.columns) != set(FEATURE_NAMES)
                or len(value.columns) != len(FEATURE_NAMES)):
            raise ValueError("N06 prepared model inputs must use the exact original R06 allowlist")
    target = np.asarray(fit_y, dtype=np.float64)
    if target.shape != (len(renamed_fit),) or not np.isfinite(target).all():
        raise ValueError("N06 training target is not a finite fit-row vector")
    model = R06CharacterSparseL1Hybrid().fit(renamed_fit, target)
    raw = model.predict(renamed_score)
    policy = compose_r06_prediction(baseline_legacy, raw, score_cycles)
    fit_documents, _ = _text_documents(renamed_fit["signal_stream"])
    score_documents, _ = _text_documents(renamed_score["signal_stream"])
    original_fit_documents, _ = _text_documents(original_fit_X["signal_stream"])
    original_score_documents, _ = _text_documents(original_score_X["signal_stream"])

    return {
        "raw_candidate": raw,
        "prediction": policy,
        "model_state": model.audit_state(),
        "prediction_diagnostics": model.prediction_diagnostics(),
        "fit_map_audit": fit_map_audit,
        "score_map_audit": score_map_audit,
        "mapped_documents_sha256": {
            "fit": _document_sha256(fit_documents),
            "score": _document_sha256(score_documents),
        },
        "original_documents_sha256": {
            "fit": _document_sha256(original_fit_documents),
            "score": _document_sha256(original_score_documents),
        },
        "seed": int(seed),
        "score_target_passed_to_fit": False,
        "test_data_used": False,
    }


def fit_predict(fit_X: pd.DataFrame, fit_y: Any, score_X: pd.DataFrame,
                *, seed: int, baseline_legacy: Any, score_cycles: Any) -> dict[str, Any]:
    """Fit the original R06 model after same-seed recoding of fit and score."""
    renamed_fit, fit_map_audit = rename_features(fit_X, seed)
    renamed_score, score_map_audit = rename_features(score_X, seed)
    return fit_predict_prepared(
        renamed_fit, fit_y, renamed_score, seed=seed,
        baseline_legacy=baseline_legacy, score_cycles=score_cycles,
        fit_map_audit=fit_map_audit, score_map_audit=score_map_audit,
        original_fit_X=fit_X, original_score_X=score_X,
    )


__all__ = [
    "FEISTEL_ROUNDS", "PREFIX_BITS", "SEEDS", "TOKEN_PATTERN", "TOKEN_RE",
    "fit_predict", "fit_predict_prepared", "inspect_token_schema", "map_payload", "map_token",
    "rename_features", "rename_series", "rename_stream", "unmap_payload",
    "unmap_token", "verify_observed_bijection",
]
