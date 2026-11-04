from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_positional_moments_v1 as runner
from src.models.positional_moments_v1 import (
    build_moment_matrix, fixed_blends, sha256_csr, tokenizer_idf_signature,
)


def _small_vectorizer() -> TfidfVectorizer:
    value = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=1,
                            max_df=1.0, ngram_range=(1, 2), dtype=np.float64)
    value.fit(["alpha beta gamma", "café alpha delta", "beta beta epsilon"])
    return value


def test_empty_single_two_token_and_all_oov_rows_are_well_defined():
    vectorizer = _small_vectorizer()
    matrix, audit = build_moment_matrix(["", "alpha", "alpha beta", "unknown omega"], vectorizer)
    d = sum(" " not in token for token in vectorizer.vocabulary_)
    assert matrix.shape == (4, 3 * d)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    assert norms[0] == 0.0
    assert norms[1] == pytest.approx(1.0)
    assert norms[2] == pytest.approx(1.0)
    assert norms[3] == 0.0
    assert audit["zero_rows"] == 2
    assert audit["all_nonzero_rows_unit_norm"] is True


def test_reversal_flips_h1_and_preserves_bag_and_h2():
    vectorizer = _small_vectorizer()
    matrix, _ = build_moment_matrix(["alpha beta gamma delta", "delta gamma beta alpha"], vectorizer)
    dense = matrix.toarray()
    width = matrix.shape[1] // 3
    np.testing.assert_allclose(dense[0, :width], dense[1, :width], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(dense[0, width:2 * width], -dense[1, width:2 * width], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(dense[0, 2 * width:], dense[1, 2 * width:], rtol=0.0, atol=1e-14)


def test_shuffle_preserves_bag_and_length_and_is_deterministic():
    vectorizer = _small_vectorizer()
    docs = ["alpha beta alpha café", "gamma delta epsilon", "", "unknown token beta"]
    real, a = build_moment_matrix(docs, vectorizer, shuffled=False)
    shuffled1, b = build_moment_matrix(docs, vectorizer, shuffled=True, seed=42)
    shuffled2, c = build_moment_matrix(docs, vectorizer, shuffled=True, seed=42)
    assert a["bag_channel_sha256"] == b["bag_channel_sha256"]
    assert a["token_count_sha256"] == b["token_count_sha256"] == c["token_count_sha256"]
    assert sha256_csr(shuffled1) == sha256_csr(shuffled2)
    assert sha256_csr(real) != sha256_csr(shuffled1)
    assert tokenizer_idf_signature(vectorizer)["unigram_size"] == sum(
        " " not in token for token in vectorizer.vocabulary_
    )


def test_zero_gamma_and_fixed_policy_keep_original_missing_route_exact():
    x = sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    z0 = np.asarray([-0.2, 0.3])
    gamma = np.zeros(3, dtype=np.float64)
    q0 = 1.0 / (1.0 + np.exp(-z0))
    np.testing.assert_array_equal(1.0 / (1.0 + np.exp(-(z0 + x @ gamma))), q0)
    q = q0.copy()
    present = np.asarray([True, False])
    a02 = np.asarray([0.6, 0.9])
    old = np.asarray([0.4, 0.123456789])
    candidate, mix50 = fixed_blends(q, a02, old, present)
    assert candidate[0] == 0.5 * q[0] + 0.5 * a02[0]
    assert candidate[1] == old[1]
    assert mix50[1] == old[1]


def test_budget_amendment_recovery_and_exact_replay_are_temporary_only(tmp_path, monkeypatch):
    production_before = runner.sha256_file(runner.LEDGER)
    frozen_prefix = runner.RUN_ROOT / "ledger_pre_reservation.jsonl"
    if frozen_prefix.is_file():
        source = frozen_prefix
    else:
        source = runner.LEDGER
        assert runner.sha256_file(source) == runner.EXPECTED_LEDGER_SHA
    temp_ledger = tmp_path / "budget.jsonl"
    shutil.copyfile(source, temp_ledger)
    events, snapshot = runner._ledger_snapshot(temp_ledger)
    freeze = {
        "source_commit": runner._git("rev-parse", "HEAD"), "freeze_signature": "test-freeze-signature",
        "source_hashes": runner._source_hashes(), "input_sha256": "synthetic-input-sha",
        "pre_reservation_ledger_sha256": runner.sha256_file(temp_ledger),
        "pre_reservation_ledger_bytes": temp_ledger.stat().st_size,
        "pre_reservation_ledger_rows": len(events),
    }
    assert snapshot == runner.BASE_SNAPSHOT
    original_append = runner._append_event
    calls = 0

    def interrupt_after_amendment(path, event, *, official):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interruption after durable cap amendment")
        return original_append(path, event, official=official)

    monkeypatch.setattr(runner, "_append_event", interrupt_after_amendment)
    with pytest.raises(OSError, match="synthetic interruption"):
        runner._append_pilot_reservation(temp_ledger, freeze, official=False)
    assert runner._ledger_snapshot(temp_ledger)[1] == runner.AMENDED_PRE_RESERVATION_SNAPSHOT
    monkeypatch.setattr(runner, "_append_event", original_append)
    runner._append_pilot_reservation(temp_ledger, freeze, official=False)
    after_first = runner.sha256_file(temp_ledger)
    runner._append_pilot_reservation(temp_ledger, freeze, official=False)
    assert runner.sha256_file(temp_ledger) == after_first
    assert runner._ledger_snapshot(temp_ledger)[1] == runner.NEW_PRE_FIT_SNAPSHOT
    assert runner.sha256_file(runner.LEDGER) == production_before


def test_score_target_reader_checks_only_requested_row_identity(tmp_path, monkeypatch):
    path = tmp_path / "Train.csv"
    pd.DataFrame({"record_id": ["a", "b", "c"], "cycle_code": [47, 48, 49],
                  "artifact_signal": [0, 1, 0]}).to_csv(path, index=False)
    monkeypatch.setattr(runner, "TRAIN", path)
    target = runner._score_rows_target(np.asarray([1, 2]), np.asarray(["b", "c"]), np.asarray([48, 49]))
    np.testing.assert_array_equal(target, np.asarray([1, 0], dtype=np.int8))
    with pytest.raises(runner.MomentsPilotError, match="identity/cycle"):
        runner._score_rows_target(np.asarray([1]), np.asarray(["wrong"]), np.asarray([48]))


def test_wrong_origin_and_tampered_freeze_are_rejected():
    x = sparse.csr_matrix([[1.0, 0.0, 0.0]])
    with pytest.raises(runner.MomentsPilotError, match="outside frozen order"):
        runner._run_one_fit(runner.FIT_IDS[0], origin=75, arm="REAL", x=x,
                            z0=np.asarray([0.0]), y=np.asarray([0], dtype=np.int8),
                            fit_rows=np.asarray([0]), fit_ids=np.asarray(["id-0"]),
                            freeze_data={}, ledger_path=runner.LEDGER, run_root=runner.RUN_ROOT,
                            official=False)
    from scripts import verify_optimization_v3_positional_moments_v1 as verifier
    with pytest.raises(verifier.VerificationError, match="signature"):
        verifier._verify_freeze(ROOT, {"freeze_signature": "tampered", "source_hashes": {}})


def test_fit_memory_estimate_uses_frozen_sparse_matrix_rule():
    matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    estimate = runner._fit_memory_estimate(matrix)
    assert estimate["matrix_bytes"] == matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert estimate["estimated_incremental_bytes"] == 1024**3
