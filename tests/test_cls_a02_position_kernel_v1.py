from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.a02_position_kernel_v1 import (
    MISSING_THRESHOLD,
    _top_positive,
    assemble_full_champion,
    construct_position_kernel,
    load_npz_allowlisted,
)
from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
from scripts.cls_a02_position_kernel_v1 import PilotError, persist_or_resume_unit
from scripts.score_cls_a02_position_kernel_v1 import (
    ScoreError,
    _assert_prelabel_artifact_hash,
    _targets_for,
    _view_from_arrays,
)
from scripts.verify_cls_a02_position_kernel_v1_metrics import (
    MetricVerificationError,
    _score_view,
    _target,
)


def _synthetic_case():
    width = 40
    bank_bag = sparse.eye(width, format="csr", dtype=np.float64)
    bank_position = sparse.eye(width, format="csr", dtype=np.float64)
    query_bag = sparse.csr_matrix(
        (np.asarray([1.0, 1.0]), (np.asarray([0, 1]), np.asarray([35, 2]))),
        shape=(4, width),
    )
    query_position = query_bag.copy()
    vectorizer = SimpleNamespace(vocabulary_={f"t{i}": i for i in range(width)})
    text_space = SimpleNamespace(origin_cycle=55, vectorizer=vectorizer)
    bank_ids = tuple(f"str:bank-{i}" for i in range(width))
    query_ids = ("query-0", "query-1", "query-2", "query-3")
    bank_rows = np.arange(1000, 1000 + width, dtype=np.int64)
    query_rows = np.asarray([2000, 2001, 2002, 2003], dtype=np.int64)
    bank_cycles = np.full(width, 55, dtype=np.int64)
    query_cycles = np.asarray([56, 57, 58, 59], dtype=np.int64)
    residuals = np.zeros(width, dtype=np.float64)
    residuals[35] = 0.10
    residuals[2] = -0.20
    baseline = np.asarray([0.40, 0.60, 0.30, 0.70], dtype=np.float64)
    original = np.asarray([0.43, 0.58, 0.30, 0.70], dtype=np.float64)
    old_rows = np.full((4, 32), -1, dtype=np.int64)
    old_sims = np.zeros((4, 32), dtype=np.float64)
    old_rows[0, 0] = bank_rows[0]
    old_sims[0, 0] = 0.8
    old_rows[1, 0] = bank_rows[1]
    old_sims[1, 0] = 0.6
    missing = np.asarray([False, False, False, True])
    return dict(
        text_space=text_space,
        query_bag=query_bag,
        bank_bag=bank_bag,
        query_position=query_position,
        bank_position=bank_position,
        bank_ids=bank_ids,
        bank_row_positions=bank_rows,
        bank_cycles=bank_cycles,
        bank_residuals=residuals,
        query_ids=query_ids,
        query_row_positions=query_rows,
        query_cycles=query_cycles,
        baseline_probabilities=baseline,
        original_a02_probabilities=original,
        original_a02_neighbor_rows=old_rows,
        original_a02_neighbor_similarities=old_sims,
        missing_mask=missing,
        arm="REAL",
        block_rows=2,
    )


def test_full_bank_mass_formula_and_exact_fallbacks():
    out = construct_position_kernel(**_synthetic_case())
    assert out["neighbor_source_rows"][0, 0] == 1035  # outside the cached original top-1
    assert out["neighbor_source_rows"][1, 0] == 1002
    assert abs(float(out["mass_error"][0])) <= 1e-12
    assert abs(float(out["mass_error"][1])) <= 1e-12
    assert np.isclose(out["candidate_a02_probability"][0], 0.40 + 0.64 * 0.10 / 4.64)
    assert np.isclose(out["candidate_a02_probability"][1], 0.60 - 0.36 * 0.20 / 4.36)
    assert out["candidate_a02_probability"][2] == 0.30
    assert out["candidate_a02_probability"][3] == 0.70
    assert np.array_equal(out["neighbor_source_rows"][2:], np.full((2, 32), -1, dtype=np.int64))


def test_top_k_uses_source_row_stable_ties():
    sims = np.asarray([0.8, 0.8, 0.5, 0.0], dtype=np.float64)
    source_rows = np.asarray([11, 10, 5, 1], dtype=np.int64)
    assert np.array_equal(_top_positive(sims, source_rows, 2), np.asarray([1, 0]))


def test_current_champion_routing_and_missing_threshold_are_frozen():
    p_pos = np.asarray([0.7, 0.2, 0.4], dtype=np.float64)
    p_a02 = np.asarray([0.5, 0.6, MISSING_THRESHOLD], dtype=np.float64)
    missing = np.asarray([False, True, True])
    score, predicted = assemble_full_champion(p_pos, p_a02, missing)
    assert score[0] == 0.6
    assert score[1] == 0.6
    assert score[2] == MISSING_THRESHOLD
    assert np.array_equal(predicted, np.asarray([1, 1, 1], dtype=np.int8))


def test_npz_allowlist_never_indexes_poisoned_label_member(tmp_path, monkeypatch):
    path = tmp_path / "poison.npz"
    np.savez(path, score=np.asarray([0.2, 0.8]), labels=np.asarray([object()], dtype=object))
    original_np_load = np.load
    accessed = []

    class ArchiveSpy:
        def __init__(self, inner):
            self.inner = inner
            self.files = inner.files

        def __getitem__(self, key):
            accessed.append(key)
            if key == "labels":
                raise AssertionError("query target member was accessed")
            return self.inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    def spy_load(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        return ArchiveSpy(original_np_load(*args, **kwargs))

    monkeypatch.setattr(np, "load", spy_load)
    got = load_npz_allowlisted(path, allowed_members=("score",))
    assert np.array_equal(got["score"], np.asarray([0.2, 0.8]))
    assert accessed == ["score"]


def test_pinned_npz_keeps_surplus_members_closed_and_rejects_missing_or_tampered(tmp_path, monkeypatch):
    path = tmp_path / "pinned.npz"
    np.savez(path, required=np.asarray([7]), closed_metadata=np.asarray(["unused"]),
             labels=np.asarray([object()], dtype=object))
    pinned_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    original_np_load = np.load
    accessed = []

    class ArchiveSpy:
        def __init__(self, inner):
            self.inner = inner
            self.files = inner.files

        def __getitem__(self, key):
            accessed.append(key)
            if key in {"labels", "closed_metadata"}:
                raise AssertionError(f"closed NPZ member was accessed: {key}")
            return self.inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    def spy_load(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        return ArchiveSpy(original_np_load(*args, **kwargs))

    monkeypatch.setattr(np, "load", spy_load)
    arrays = load_npz_allowlisted(
        path, allowed_members=("required",), expected_sha256=pinned_sha,
        allow_closed_extras=True,
    )
    assert np.array_equal(arrays["required"], np.asarray([7]))
    assert set(arrays.unused_member_names) == {"closed_metadata", "labels"}
    assert accessed == ["required"]

    try:
        load_npz_allowlisted(path, allowed_members=("missing",), expected_sha256=pinned_sha,
                             allow_closed_extras=True)
    except ValueError as exc:
        assert "missing allowed members" in str(exc)
    else:
        raise AssertionError("missing required NPZ member must fail closed")

    monkeypatch.setattr(np, "load", original_np_load)
    path.write_bytes(path.read_bytes() + b"tamper")
    try:
        load_npz_allowlisted(path, allowed_members=("required",), expected_sha256=pinned_sha,
                             allow_closed_extras=True)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("tampered pinned NPZ archive must fail closed")


def test_npz_target_member_cannot_be_explicitly_allowlisted(tmp_path):
    path = tmp_path / "has_label.npz"
    np.savez(path, labels=np.asarray([0, 1], dtype=np.int8))
    try:
        load_npz_allowlisted(path, allowed_members=("labels",))
    except ValueError as exc:
        assert "target-bearing" in str(exc)
    else:
        raise AssertionError("target member should be refused")


def test_score_target_is_validated_before_int8_conversion():
    bundle = {
        "score_rows": np.asarray([0], dtype=np.int64),
        "score_record_ids": np.asarray(["a"]),
        "score_cycles": np.asarray([70], dtype=np.int16),
    }
    bad = pd.DataFrame(
        {"record_id": ["a"], "cycle_code": [70], "artifact_signal": [256]}
    )
    try:
        _targets_for(bundle, bad)
    except ScoreError as exc:
        assert "0/1" in str(exc)
    else:
        raise AssertionError("nonbinary target 256 must not truncate to int8 zero")

    try:
        _target(bundle, bad)
    except MetricVerificationError as exc:
        assert "0/1" in str(exc)
    else:
        raise AssertionError("independent target verifier must reject nonbinary values")


def test_score_requires_exactly_the_prelabel_verified_bundle(tmp_path):
    path = tmp_path / "predictions.npz"
    path.write_bytes(b"frozen bundle bytes")
    unit_id = "E67_REAL"
    prelabel = {"four_unit_receipts_and_outputs": {unit_id: {"artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
    _assert_prelabel_artifact_hash(prelabel, unit_id, path)
    path.write_bytes(b"changed bundle bytes")
    try:
        _assert_prelabel_artifact_hash(prelabel, unit_id, path)
    except ScoreError as exc:
        assert "independently verified" in str(exc)
    else:
        raise AssertionError("scorer must reject a bundle changed after prelabel verification")


def test_independent_metric_verifier_checks_shuffled_gain():
    real = {
        "score_record_ids": np.asarray(["a", "b", "c", "d"]),
        "champion_score": np.asarray([0.1, 0.9, 0.4, 0.6]),
        "candidate_score": np.asarray([0.9, 0.1, 0.4, 0.6]),
        "champion_class": np.asarray([0, 1, 0, 1], dtype=np.int8),
        "candidate_class": np.asarray([1, 0, 0, 1], dtype=np.int8),
    }
    shuffled = {
        "score_record_ids": real["score_record_ids"].copy(),
        "champion_score": real["champion_score"].copy(),
        "candidate_score": np.asarray([0.2, 0.8, 0.7, 0.3]),
        "champion_class": real["champion_class"].copy(),
        "candidate_class": np.asarray([0, 1, 1, 0], dtype=np.int8),
    }
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    mask = np.ones(len(y), dtype=bool)
    scored = _view_from_arrays("synthetic", real, shuffled, y, mask)
    verified = _score_view(real, shuffled, y, mask)
    assert np.isclose(scored["gain_shuffled_vs_champion_macro_f1"], verified["gain_shuffled_vs_champion"])
    assert scored["gain_shuffled_vs_champion_macro_f1"] != 0.0


def test_synthetic_cli_persistence_and_resume_lifecycle():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/cls_a02_position_kernel_v1.py", "synthetic-self-test"],
        cwd=repo, text=True, capture_output=True, check=True,
    )
    payload = __import__("json").loads(completed.stdout.strip())
    assert payload["status"] == "PASS_SYNTHETIC_END_TO_END"
    assert payload["production_paths_written"] is False
    assert payload["query_target_in_candidate_bundle"] is False
    assert payload["first_status"] == "SAVED"
    assert payload["resume_status"] == "RESUMED_DURABLE"
    assert payload["build_calls"] == 1
    assert "PREDICTOR_SAVED" in payload["ledger_events"]


def test_existing_position_transform_shuffle_is_deterministic_and_preserves_segments():
    documents = [
        "alpha beta alpha gamma delta epsilon zeta eta theta",
        "beta gamma delta beta epsilon zeta eta theta alpha",
        "gamma alpha beta delta epsilon theta zeta eta alpha",
        "delta beta alpha gamma theta epsilon eta zeta beta",
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
    vectorizer.fit(documents)
    real, _ = build_position_block(documents, vectorizer, shuffled=False)
    first, meta = build_position_block(documents, vectorizer, shuffled=True, seed=42)
    second, _ = build_position_block(documents, vectorizer, shuffled=True, seed=42)
    assert sparse.issparse(real) and sparse.issparse(first)
    assert (first != second).nnz == 0
    assert meta["bag_preserved_rows"] == len(documents)
    assert meta["segment_sizes_preserved_rows"] == len(documents)


def test_started_without_durable_output_is_not_retried(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"event":"PREDICTOR_RESERVED","fit_id":"USED"}\n'
        '{"event":"PREDICTOR_STARTED","fit_id":"USED"}\n',
        encoding="utf-8",
    )
    built = {"n": 0}

    def should_not_build():
        built["n"] += 1
        raise AssertionError("a consumed unit must not be retried")

    try:
        persist_or_resume_unit(
            "USED", should_not_build,
            artifact_path=tmp_path / "out.npz",
            receipt_path=tmp_path / "receipt.json",
            ledger_path=ledger,
            lineage={"synthetic": True}, resource={"synthetic": True},
        )
    except PilotError as exc:
        assert "never retry" in str(exc)
    else:
        raise AssertionError("started unit without output must fail closed")
    assert built["n"] == 0
