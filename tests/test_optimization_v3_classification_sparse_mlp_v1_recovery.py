from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.optimization_v3_classification_sparse_mlp_v1_recovery import (
    RESUME_AMENDMENT_HASHED_FILES,
    _atomic_json,
    _validate_pinned_source_hashes,
    _vocab_sha,
)
from scripts.optimization_v3_classification_sparse_mlp_v1_recovery_verify import _recovery_event_hashes_match


def test_recovery_vocabulary_hash_accepts_numpy_integer_indices():
    first = SimpleNamespace(vocabulary_={"alpha": np.int64(0), "beta": np.int64(1)})
    second = SimpleNamespace(vocabulary_={"beta": 1, "alpha": 0})
    assert _vocab_sha(first) == _vocab_sha(second)


def test_empty_vocabulary_recovery_hash_is_stable():
    assert _vocab_sha(None) == _vocab_sha(SimpleNamespace(vocabulary_={}))


def test_resume_amendment_requires_exact_source_set():
    with pytest.raises(RuntimeError, match="exact required recovery source set"):
        _validate_pinned_source_hashes({}, "a" * 40, lambda _: "0" * 64, lambda *_: "0" * 64)


def test_resume_amendment_binds_worktree_and_commit_hashes():
    pinned = {path: f"{index:064x}" for index, path in enumerate(RESUME_AMENDMENT_HASHED_FILES, start=1)}
    _validate_pinned_source_hashes(
        pinned, "a" * 40,
        lambda path: pinned[path],
        lambda commit, path: pinned[path] if commit == "a" * 40 else None,
    )
    with pytest.raises(RuntimeError, match="does not contain amended recovery source"):
        _validate_pinned_source_hashes(
            pinned, "a" * 40, lambda path: pinned[path], lambda *_: None
        )


def test_atomic_recovery_json_is_idempotent_and_refuses_conflict(tmp_path):
    path = tmp_path / "receipt.json"
    payload = {"fit": 55, "refit_performed": False}
    _atomic_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    _atomic_json(path, payload)
    with pytest.raises(RuntimeError, match="Refusing to overwrite nonidentical"):
        _atomic_json(path, {"fit": 67})


def test_independent_recovery_event_checks_all_durable_artifact_hashes():
    event = {
        "model_joblib_sha256": "m" * 64,
        "fit_state_sha256": "s" * 64,
        "bundle_receipt_sha256": "b" * 64,
        "recovery_code_sha256": "c" * 64,
    }
    assert _recovery_event_hashes_match(
        event, model_sha="m" * 64, state_sha="s" * 64,
        receipt_sha="b" * 64, recovery_sha="c" * 64,
    )
    mutated = {**event, "fit_state_sha256": "x" * 64}
    assert not _recovery_event_hashes_match(
        mutated, model_sha="m" * 64, state_sha="s" * 64,
        receipt_sha="b" * 64, recovery_sha="c" * 64,
    )
