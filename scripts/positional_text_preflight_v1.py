"""Label-free, official-Train positional-text profiling; no vectorizer fit."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_models import _TFIDF_PARAMS, _text_values  # noqa: E402
from src.models.positional_text_features import (  # noqa: E402
    absolute_position_features,
    deterministic_order_shuffle,
)

TRAIN = ROOT / "data/raw/classification/Train.csv"
OUT = ROOT / "reports/optimization_v3/positional_text_preflight_v1"
SUMMARY_PATH = OUT / "profiling.json"
VOCAB_PLOT = OUT / "position_vocab_overlap.png"
SHUFFLE_PLOT = OUT / "order_shuffle_descriptor_change.png"
EXPECTED_ROWS = 50_608
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
READ_COLUMNS = ("record_id", "cycle_code", "signal_stream")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: list[int]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.int64)
    if not len(arr):
        return {"n": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {"n": int(len(arr)), "min": int(arr.min()), "p50": float(np.quantile(arr, .5)),
            "p90": float(np.quantile(arr, .9)), "p95": float(np.quantile(arr, .95)),
            "max": int(arr.max()), "mean": float(arr.mean())}


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    return float(len(left & right) / len(union)) if union else None


def _jsonable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, type):
            result[key] = value.__name__
        else:
            result[key] = value
    return result


def _summarize_cohort(name: str, mask: np.ndarray, tokens_by_row: list[list[str]],
                      stream_missing: np.ndarray) -> tuple[dict[str, Any], tuple[set[str], set[str], set[str]]]:
    indices = np.flatnonzero(mask)
    present_indices = [int(row) for row in indices if not stream_missing[row]]
    token_lengths = [len(tokens_by_row[row]) for row in present_indices]
    segment_vocab: list[set[str]] = [set(), set(), set()]
    segment_tokens = np.zeros(3, dtype=np.int64)
    cross_segment_repeat_rows = 0
    cross_segment_repeat_occurrences = 0
    repeated_token_types_across_segments = 0
    descriptor_changed = 0
    occurrence_movement: list[float] = []
    l1_descriptor_change: list[float] = []
    bag_preservation_checks = 0
    segment_size_checks = 0
    tokenless_present = 0
    present_rows_with_tokens = 0

    for row in present_indices:
        tokens = tokens_by_row[row]
        features = absolute_position_features(tokens)
        segment_counts = features["segment_counts"]
        if tokens:
            present_rows_with_tokens += 1
        else:
            tokenless_present += 1
        token_segment_sets: dict[str, set[int]] = {}
        for segment, block in enumerate(segment_counts):
            segment_vocab[segment].update(block)
            segment_tokens[segment] += sum(block.values())
            for token, count in block.items():
                token_segment_sets.setdefault(token, set()).add(segment)
        cross_tokens = {token for token, segments in token_segment_sets.items() if len(segments) > 1}
        if cross_tokens:
            cross_segment_repeat_rows += 1
            repeated_token_types_across_segments += len(cross_tokens)
            cross_segment_repeat_occurrences += sum(
                count for block in segment_counts for token, count in block.items() if token in cross_tokens
            )
        shuffled = deterministic_order_shuffle(tokens, seed=42)
        bag_preservation_checks += int(Counter(shuffled["tokens"]) == Counter(tokens))
        segment_size_checks += int(shuffled["segment_sizes"] == features["segment_sizes"])
        descriptor_changed += int(shuffled["segment_counts"] != segment_counts)
        occurrence_movement.append(float(shuffled["fraction_of_occurrences_moved_between_thirds"]))
        l1 = 0
        for segment in range(3):
            left, right = segment_counts[segment], shuffled["segment_counts"][segment]
            for token in left.keys() | right.keys():
                l1 += abs(int(left.get(token, 0)) - int(right.get(token, 0)))
        l1_descriptor_change.append(l1 / max(1, 2 * len(tokens)))

    present_count = int(len(present_indices))
    total_token_count = int(sum(token_lengths))
    summary = {
        "cohort": name,
        "rows": int(len(indices)),
        "stream_present_rows": present_count,
        "stream_missing_or_blank_rows": int(len(indices) - present_count),
        "present_rows_with_zero_tokens": tokenless_present,
        "token_count_present": _percentiles(token_lengths),
        "all_unigram_token_occurrences": total_token_count,
        "absolute_third_token_occurrences": [int(x) for x in segment_tokens],
        "absolute_third_vocabulary_sizes": [len(x) for x in segment_vocab],
        "fraction_present_rows_with_token_repeated_across_thirds": (
            float(cross_segment_repeat_rows / present_count) if present_count else 0.0
        ),
        "cross_third_repeated_token_types_by_row_sum": int(repeated_token_types_across_segments),
        "token_occurrence_fraction_for_types_repeated_across_thirds": (
            float(cross_segment_repeat_occurrences / total_token_count) if total_token_count else 0.0
        ),
        "order_shuffle": {
            "seed": 42,
            "normalized_text_digest": "SHA-256 over normalized token sequence; per-row digests not persisted",
            "token_bag_preserved_rows": bag_preservation_checks,
            "segment_sizes_preserved_rows": segment_size_checks,
            "rows_with_changed_segment_token_count_descriptor": descriptor_changed,
            "fraction_rows_with_changed_descriptor": float(descriptor_changed / present_count) if present_count else 0.0,
            "mean_occurrence_movement_between_thirds": float(np.mean(occurrence_movement)) if occurrence_movement else 0.0,
            "mean_normalized_segment_count_L1_change": float(np.mean(l1_descriptor_change)) if l1_descriptor_change else 0.0,
        },
    }
    return summary, tuple(segment_vocab)  # token sets stay in memory and are never serialized.


def _save_plots(comparisons: dict[str, Any], cohort_summaries: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.7), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    colors = ["#3C78A8", "#F58518"]
    for j, origin in enumerate(("E67", "E75")):
        values = comparisons[origin]["prefix_to_scored_vocab_jaccard_by_third"]
        ax.bar(x + (j - .5) * width, [v or 0.0 for v in values], width,
               label=f"{origin} prefix ↔ scored", color=colors[j])
        for pos, value in zip(x + (j - .5) * width, values):
            ax.text(pos, (value or 0.0) + .015, "n/a" if value is None else f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["First third", "Middle third", "Final third"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Jaccard overlap of distinct tokens")
    ax.set_title("Token vocabulary overlap across temporal prefix and scored cycles")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.22)
    fig.savefig(VOCAB_PLOT, dpi=160)
    plt.close(fig)

    labels = ["E67 fit", "E67 scored", "E75 fit", "E75 scored"]
    keys = ["E67_prefix_47_67", "E67_scored_68_79", "E75_prefix_47_75", "E75_scored_76_79"]
    changed = [cohort_summaries[key]["order_shuffle"]["fraction_rows_with_changed_descriptor"] for key in keys]
    moved = [cohort_summaries[key]["order_shuffle"]["mean_occurrence_movement_between_thirds"] for key in keys]
    fig, ax = plt.subplots(figsize=(8.3, 4.7), constrained_layout=True)
    x = np.arange(len(labels))
    width = .36
    ax.bar(x - width / 2, changed, width, label="Rows: descriptor changed", color="#54A24B")
    ax.bar(x + width / 2, moved, width, label="Token occurrences moved thirds", color="#B279A2")
    for positions, values in ((x - width / 2, changed), (x + width / 2, moved)):
        for position, value in zip(positions, values):
            ax.text(position, value + .018, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of present-stream rows / token occurrences")
    ax.set_title("Seed-42 order shuffle changes position while preserving each token bag")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", alpha=.22)
    fig.savefig(SHUFFLE_PLOT, dpi=160)
    plt.close(fig)


def run() -> dict[str, Any]:
    if not TRAIN.is_file():
        raise FileNotFoundError(f"Official Classification Train missing: {TRAIN}")
    train_sha = _sha_file(TRAIN)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Official Classification Train hash differs from audited source: {train_sha}")
    # Deliberately read only row identity, temporal fold, and text. No target column is requested.
    frame = pd.read_csv(TRAIN, usecols=list(READ_COLUMNS), dtype={"record_id": "string"})
    if (len(frame) != EXPECTED_ROWS or frame["record_id"].isna().any()
            or frame["record_id"].duplicated().any()):
        raise ValueError("Official Classification Train row/ID integrity failed")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if cycles.min() != 47 or cycles.max() != 79:
        raise ValueError("Unexpected cycle range in official Classification Train")
    documents, missing = _text_values(frame["signal_stream"])
    # build_analyzer applies the exact existing C2_B word normalization/token rule.
    # No fit/fit_transform method is called in this preflight.
    token_vectorizer = TfidfVectorizer(**{**_TFIDF_PARAMS, "ngram_range": (1, 1)})
    analyzer = token_vectorizer.build_analyzer()
    tokens_by_row = [list(analyzer(document)) for document in documents]
    if any(missing[row] and tokens_by_row[row] for row in range(len(frame))):
        raise AssertionError("Missing/blank stream unexpectedly produced tokens")

    cohorts = {
        "E67_prefix_47_67": (cycles >= 47) & (cycles <= 67),
        "E67_scored_68_79": (cycles >= 68) & (cycles <= 79),
        "E75_prefix_47_75": (cycles >= 47) & (cycles <= 75),
        "E75_scored_76_79": (cycles >= 76) & (cycles <= 79),
    }
    summaries: dict[str, Any] = {}
    vocabularies: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for name, mask in cohorts.items():
        summaries[name], vocabularies[name] = _summarize_cohort(name, mask, tokens_by_row, missing)

    comparisons: dict[str, Any] = {}
    for origin, prefix_key, scored_key in (
        ("E67", "E67_prefix_47_67", "E67_scored_68_79"),
        ("E75", "E75_prefix_47_75", "E75_scored_76_79"),
    ):
        prefix_vocab, scored_vocab = vocabularies[prefix_key], vocabularies[scored_key]
        comparisons[origin] = {
            "prefix_rows": summaries[prefix_key]["rows"],
            "scored_rows": summaries[scored_key]["rows"],
            "prefix_to_scored_vocab_jaccard_by_third": [
                _jaccard(prefix_vocab[j], scored_vocab[j]) for j in range(3)
            ],
            "prefix_only_vocab_types_by_third": [len(prefix_vocab[j] - scored_vocab[j]) for j in range(3)],
            "scored_only_vocab_types_by_third": [len(scored_vocab[j] - prefix_vocab[j]) for j in range(3)],
            "prefix_vocabulary_sizes_by_third": [len(v) for v in prefix_vocab],
            "scored_vocabulary_sizes_by_third": [len(v) for v in scored_vocab],
            "prefix_within_third_vocab_jaccard_1_2_3": [
                _jaccard(prefix_vocab[0], prefix_vocab[1]),
                _jaccard(prefix_vocab[0], prefix_vocab[2]),
                _jaccard(prefix_vocab[1], prefix_vocab[2]),
            ],
            "scored_within_third_vocab_jaccard_1_2_3": [
                _jaccard(scored_vocab[0], scored_vocab[1]),
                _jaccard(scored_vocab[0], scored_vocab[2]),
                _jaccard(scored_vocab[1], scored_vocab[2]),
            ],
        }

    _save_plots(comparisons, summaries)
    result = {
        "status": "PASS_LABEL_FREE_POSITIONAL_PREFLIGHT",
        "objective": "Describe absolute token-third structure and its deterministic within-document order-shuffle control; no predictive claim.",
        "source": {
            "path": TRAIN.relative_to(ROOT).as_posix(), "sha256": train_sha,
            "rows": int(len(frame)), "columns_read": list(READ_COLUMNS),
            "columns_not_read": ["artifact_signal", "threat_rank", "all other targets"],
            "record_id_used_only_for_row_identity_validation": True,
            "cycle_min_max": [int(cycles.min()), int(cycles.max())],
            "present_stream_rows": int((~missing).sum()), "missing_or_blank_stream_rows": int(missing.sum()),
        },
        "tokenization": {
            "normalizer": "src.models.classification_models._text_values",
            "analyzer": "sklearn TfidfVectorizer word analyzer; exact existing C2_B lowercase/Unicode accent-strip/default token rule",
            "existing_c2b_parameters": _jsonable_parameters(_TFIDF_PARAMS),
            "preflight_analyzer_ngram_range": [1, 1],
            "vectorizer_fit_or_fit_transform_calls": 0,
            "position_rule": "third(i)=floor(3*i/n), i zero-based, n normalized unigram tokens; empty stream kept separate",
        },
        "cohorts": summaries,
        "vocabulary_comparison": comparisons,
        "source_hashes": {
            "src/models/classification_models.py": _sha_file(ROOT / "src/models/classification_models.py"),
            "src/models/classification_a02_neighbors.py": _sha_file(ROOT / "src/models/classification_a02_neighbors.py"),
            "src/models/positional_text_features.py": _sha_file(ROOT / "src/models/positional_text_features.py"),
            "scripts/positional_text_preflight_v1.py": _sha_file(Path(__file__).resolve()),
        },
        "plots": {
            "position_vocab_overlap.png": _sha_file(VOCAB_PLOT),
            "order_shuffle_descriptor_change.png": _sha_file(SHUFFLE_PLOT),
        },
        "boundaries": {
            "target_or_query_labels_read": False, "test_or_sample_submission_opened": False,
            "kaggle_accessed": False, "model_or_vectorizer_fit": False,
            "raw_text_or_token_strings_persisted": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
