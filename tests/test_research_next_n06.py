"""Mapping and end-to-end persistence checks for Research Next N06."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import research_next_n06 as n06


@pytest.mark.parametrize("seed", n06.SEEDS)
@pytest.mark.parametrize("prefix", ("q", "z"))
def test_feistel_roundtrip_and_collision_free_subset(seed: int, prefix: str) -> None:
    width = 7 if prefix == "q" else 6
    source = list(range(4096))
    mapped = [n06.map_payload(prefix, payload, seed) for payload in source]

    assert len(mapped) == len(set(mapped))
    assert all(n06.unmap_payload(prefix, value, seed) == original
               for original, value in zip(source, mapped))
    tokens = [prefix + format(value, f"0{width}x") for value in source]
    renamed = [n06.map_token(token, seed) for token in tokens]
    assert all(len(left) == len(right) and left[0] == right[0]
               for left, right in zip(tokens, renamed))
    assert all(n06.unmap_token(value, seed) == original
               for original, value in zip(tokens, renamed))


@pytest.mark.parametrize("seed", n06.SEEDS)
def test_stream_mapping_preserves_separators_order_multiplicity_and_missing(seed: int) -> None:
    source = "(q0000001|z000001) / q0000001, end"
    renamed, count = n06.rename_stream(source, seed)
    pieces = n06.TOKEN_RE.findall(source)
    mapped_pieces = n06.TOKEN_RE.findall(renamed)

    assert count == 3
    assert len(renamed) == len(source)
    assert mapped_pieces == [n06.map_token(token, seed) for token in pieces]
    assert renamed.count(mapped_pieces[0]) == 2
    restored = n06.TOKEN_RE.sub(lambda match: n06.unmap_token(match.group("token"), seed), renamed)
    assert restored == source
    missing, missing_count = n06.rename_stream(None, seed)
    assert missing is None and missing_count == 0


def test_token_schema_fails_closed_on_unrecognized_shape() -> None:
    series = pd.Series(["q0000001 z000001", "q000000g z000001", None])
    schema = n06.inspect_token_schema(series)

    assert schema["status"] == "SKIPPED_TOKEN_SCHEMA"
    assert schema["token_occurrences"] == {"q_7hex": 1, "z_6hex": 2}
    assert schema["unexpected_token_occurrences"] == 1
    assert schema["unexpected_examples"] == ["q000000g"]


def test_mapping_seeds_are_fixed_and_payload_bounds_are_checked() -> None:
    with pytest.raises(ValueError, match="mapping seed"):
        n06.map_token("q0000001", 99)
    with pytest.raises(ValueError, match="outside its 28-bit domain"):
        n06.map_payload("q", 1 << 28, 42)
    with pytest.raises(ValueError, match="outside its 24-bit domain"):
        n06.map_payload("z", 1 << 24, 137)

