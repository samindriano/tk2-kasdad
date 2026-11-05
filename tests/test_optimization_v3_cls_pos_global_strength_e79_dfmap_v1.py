from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as dfmap
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as independent


def _position_fixture() -> sparse.csr_matrix:
    # V=4, three positional blocks. Repeated features across blocks still count once per row.
    dense = np.zeros((4, 12), dtype=np.float64)
    dense[0, [0, 4, 9]] = [0.3, 0.7, 0.2]  # token 0 in two blocks, token 1 in final
    dense[1, [3, 5, 10]] = [0.4, 0.6, 0.8]  # tokens 3, 1, 2
    dense[2, [5, 6, 11]] = [0.2, 0.5, 0.9]  # token 1 repeated across blocks, token 2 final
    dense[3, [1, 6, 8]] = [0.1, 0.4, 0.3]  # token 1 first/middle, token 0 final
    return sparse.csr_matrix(dense)


def test_union_document_frequency_counts_each_token_once_per_row() -> None:
    position = _position_fixture()

    actual = dfmap.union_document_frequency(position)

    # token 0 rows 0 and 3; token 1 rows 0, 1, 2, 3; token 2 rows 1, 2, 3;
    # token 3 rows 1 and 2. Segment duplicates are deduplicated within each row.
    np.testing.assert_array_equal(actual, np.array([2, 4, 3, 2], dtype=np.int64))


def test_original_formula_seed_and_global_strength_vector_are_exact() -> None:
    position = _position_fixture()

    result = dfmap.compute_df_map(position, seed=42)
    df = np.array([2, 4, 3, 2], dtype=np.int64)
    median_df = float(np.median(df))
    raw = np.clip(np.sqrt(median_df / df.astype(np.float64)), 0.5, 2.0)
    aligned = raw / raw.mean()
    permutation = np.random.default_rng(42).permutation(len(df))
    scalar = float(np.maximum(1.0, aligned).mean())

    np.testing.assert_array_equal(result["document_frequency_int64"], df)
    np.testing.assert_array_equal(result["raw_clipped_weights_float64"], raw)
    np.testing.assert_array_equal(result["aligned_weights_float64"], aligned)
    np.testing.assert_array_equal(result["permuted_aligned_weights_float64"], aligned[permutation])
    np.testing.assert_array_equal(
        result["global_strength_position_penalty_float64"],
        np.full(12, scalar, dtype=np.float64),
    )
    assert float(result["global_strength_scalar_float64"]) == scalar
    assert np.isclose(result["statistics"]["aligned_mean"], 1.0, rtol=0.0, atol=1e-15)
    assert result["statistics"]["global_position_penalty_unique_values"] == 1


def test_rejects_zero_support_token() -> None:
    position = sparse.csr_matrix(
        np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.float64)
    )

    with pytest.raises(dfmap.DFMapError, match="frequencies must be positive"):
        dfmap.compute_df_map(position)


def test_npz_artifact_is_persistent_and_one_shot(tmp_path: Path) -> None:
    result = dfmap.compute_df_map(_position_fixture())
    arrays = {key: value for key, value in result.items() if key != "statistics"}
    path = tmp_path / "synthetic_df_map.npz"

    first_hash = dfmap._save_npz_once(path, arrays)
    assert first_hash == dfmap.sha256_file(path)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        np.testing.assert_array_equal(loaded["document_frequency_int64"], arrays["document_frequency_int64"])
        np.testing.assert_array_equal(loaded["global_strength_position_penalty_float64"], arrays["global_strength_position_penalty_float64"])
    checked = independent.verify_npz_arrays(
        path, expected_size=4, expected_rows=4, statistics=result["statistics"],
    )
    assert checked["df_min"] == 2
    assert checked["df_max"] == 4
    with pytest.raises(r3.ReplayError, match="replace an existing one-shot output"):
        dfmap._save_npz_once(path, arrays)


def _fake_attestation() -> dict[str, object]:
    return {
        "status": "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R3_PENDING_REPLAY",
        "attestation_sha256": "synthetic-attestation",
        "current_source_sha256": {"x.py": "hash-x"},
        "saved_inputs_and_fitted_state_receipts": {"immutable_file_sha256": {"Train.csv": "hash-train"}},
    }


def _fake_lineage() -> dict[str, object]:
    return {
        "head": "synthetic-head",
        "freeze_sha256": "synthetic-freeze",
        "r3_receipt_file_sha256": "synthetic-r3-receipt",
    }


def test_resource_block_writes_no_attempt_or_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(dfmap, "verify_source_freeze", lambda _root: (_fake_attestation(), _fake_lineage()))
    monkeypatch.setattr(r3, "_shared_job_lock", lambda: nullcontext({"acquired": True}))
    monkeypatch.setattr(r3, "_cgeom_local_lock", lambda _root: nullcontext({"acquired": True}))
    monkeypatch.setattr(r3, "_resource_snapshot", lambda _stage: {"gate_pass": False, "available_ram_bytes": 1})

    result = dfmap.run_df_map(root)
    attempt, artifact, receipt = dfmap._output_paths(root)

    assert result["status"] == "BLOCKED_RESOURCE_GATE"
    assert not attempt.exists() and not artifact.exists() and not receipt.exists()


def test_failure_after_marker_persists_no_retry_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(dfmap, "verify_source_freeze", lambda _root: (_fake_attestation(), _fake_lineage()))
    monkeypatch.setattr(r3, "_shared_job_lock", lambda: nullcontext({"acquired": True}))
    monkeypatch.setattr(r3, "_cgeom_local_lock", lambda _root: nullcontext({"acquired": True}))
    monkeypatch.setattr(r3, "_resource_snapshot", lambda _stage: {"gate_pass": True, "available_ram_bytes": 3_000_000_000})
    monkeypatch.setattr(dfmap, "_execute_df_map", lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic map failure")))

    result = dfmap.run_df_map(root)
    attempt, _artifact, receipt = dfmap._output_paths(root)
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    body = dict(saved)
    digest = body.pop("receipt_sha256")

    assert result["status"] == "FAILED_DF_MAP_NO_RETRY"
    assert attempt.is_file() and receipt.is_file()
    assert digest == dfmap.canonical_sha256(body)
    assert saved["error"] == "synthetic map failure"
    with pytest.raises(dfmap.DFMapError, match="refusing retry"):
        dfmap.run_df_map(root)
