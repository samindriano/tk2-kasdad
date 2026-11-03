from __future__ import annotations

from collections import Counter
from inspect import signature

from src.models.positional_text_features import (
    absolute_position_features,
    deterministic_order_shuffle,
    token_thirds,
)


def test_empty_one_and_two_token_boundaries() -> None:
    empty = absolute_position_features([])
    assert empty == {"token_count": 0, "segment_sizes": (0, 0, 0),
                     "segment_counts": ({}, {}, {}), "empty": True}
    one = absolute_position_features(["alpha"])
    assert one["segment_sizes"] == (1, 0, 0)
    assert one["segment_counts"] == ({"alpha": 1}, {}, {})
    two = absolute_position_features(["alpha", "beta"])
    assert token_thirds(2) == (0, 1)
    assert two["segment_sizes"] == (1, 1, 0)


def test_tied_boundaries_duplicates_and_unicode_tokens() -> None:
    assert token_thirds(4) == (0, 0, 1, 2)
    features = absolute_position_features(["café", "same", "same", "naïve"])
    assert features["segment_sizes"] == (2, 1, 1)
    flattened = Counter(token for block in features["segment_counts"] for token, count in block.items()
                        for _ in range(count))
    assert flattened == Counter(["café", "same", "same", "naïve"])
    assert features["segment_counts"][1] == {"same": 1}


def test_deterministic_order_shuffle_preserves_bag_length_and_segment_sizes() -> None:
    tokens = ["café", "x7", "repeat", "repeat", "naïve", "λambda", "tail", "repeat"]
    first = deterministic_order_shuffle(tokens, seed=42)
    second = deterministic_order_shuffle(tokens, seed=42)
    assert first["tokens"] == second["tokens"]
    assert first["source_positions"] == second["source_positions"]
    assert first["stable_normalized_text_digest_sha256"] == second["stable_normalized_text_digest_sha256"]
    assert Counter(first["tokens"]) == Counter(tokens)
    assert first["token_count"] == len(tokens)
    assert first["segment_sizes"] == absolute_position_features(tokens)["segment_sizes"]
    assert sum(first["segment_sizes"]) == len(tokens)
    assert deterministic_order_shuffle(tokens, seed=43)["tokens"] != first["tokens"]


def test_empty_and_singleton_order_controls_are_well_defined() -> None:
    empty = deterministic_order_shuffle([], seed=42)
    singleton = deterministic_order_shuffle(["one"], seed=42)
    assert empty["tokens"] == () and empty["fraction_of_occurrences_moved_between_thirds"] == 0.0
    assert singleton["tokens"] == ("one",) and singleton["segment_sizes"] == (1, 0, 0)
    assert singleton["fraction_of_occurrences_moved_between_thirds"] == 0.0


def test_constructor_accepts_no_identity_or_target_arguments() -> None:
    assert list(signature(absolute_position_features).parameters) == ["tokens"]
    assert list(signature(deterministic_order_shuffle).parameters) == ["tokens", "seed"]
