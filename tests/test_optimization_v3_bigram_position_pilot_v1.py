import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from scripts import verify_optimization_v3_bigram_position_pilot_v1 as verifier
from scripts import optimization_v3_positional_bag_offset_v1 as offset
from src.models.optimization_v3_bigram_position_pilot_v1 import (
    reconstruct_original_pos_logit,
    route_candidate,
    zero_gamma_equivalence,
)
from scripts.optimization_v3_bigram_position_pilot_v1 import (
    FIT_IDS, PilotError, _assert_successor_reservation_untouched,
    _gate, _read_fit_labels_only,
)


def _write_migration_fixture(root, monkeypatch):
    run_root = root / "run"
    archive_root = run_root / "archive"
    identity = pd.DataFrame({"record_id": ["r0", "r1", "r2"],
                             "cycle_code": np.asarray([47, 48, 49], dtype=np.int64)})
    monkeypatch.setattr(offset, "_read_identity_frame", lambda: identity.copy())
    rows = {
        "fit_rows": np.asarray([0, 1], dtype=np.int64),
        "fit_record_ids": np.asarray(["r0", "r1"], dtype="<U2"),
        "score_rows": np.asarray([2], dtype=np.int64),
        "score_record_ids": np.asarray(["r2"], dtype="<U2"),
        "score_cycles": np.asarray([49], dtype=np.int64),
        "present_mask": np.asarray([True]),
        "no_known_bigram": np.asarray([False]),
        "z_fit_original": np.asarray([-0.1, 0.2]),
        "z_query_original": np.asarray([0.3]),
        "p_pos_original": np.asarray([0.574442516811659]),
        "p_p": np.asarray([0.4]),
        "p_a02": np.asarray([0.6]),
        "mix50": np.asarray([0.5]),
        "entry_friction": np.asarray([0.2]),
        "text_load_group": np.asarray(["Q1"], dtype="<U2"),
        "known_fit": np.asarray([2, 1], dtype=np.int64),
        "known_query": np.asarray([1], dtype=np.int64),
    }
    expected_rows = {
        "fit_rows": offset.sha256_array(rows["fit_rows"]),
        "fit_record_ids": offset.sha256_array(rows["fit_record_ids"].astype(object)),
        "score_rows": offset.sha256_array(rows["score_rows"]),
        "score_record_ids": offset.sha256_array(rows["score_record_ids"].astype(object)),
        "score_cycles": offset.sha256_array(rows["score_cycles"]),
        "present_mask": offset.sha256_array(rows["present_mask"]),
        "metadata_file": "original-row-npz-bytes-not-recovered",
    }
    old_matrices = {}
    for name, matrix in {"fit_bag": sparse.csr_matrix([[1.0], [0.0]]),
                         "query_bag": sparse.csr_matrix([[2.0]])}.items():
        old_matrices[name] = {
            "file_sha256": None,
            "logical_sha256": offset.sha256_csr(matrix),
            "shape": list(matrix.shape),
        }
    origin_freezes = {}
    for suffix in ("E67", "E75"):
        for destination in (run_root / suffix, archive_root / suffix):
            destination.mkdir(parents=True, exist_ok=True)
            np.savez(destination / "frozen_rows.npz", **rows)
            for name, matrix in {"fit_bag": sparse.csr_matrix([[1.0], [0.0]]),
                                 "query_bag": sparse.csr_matrix([[2.0]])}.items():
                sparse.save_npz(destination / f"{name}.npz", matrix)
        current = run_root / suffix
        matrices = {name: dict(item) for name, item in old_matrices.items()}
        for name, item in matrices.items():
            item["file_sha256"] = offset.sha256_file(current / f"{name}.npz")
        origin_freezes[suffix] = {
            "row_arrays_sha256": expected_rows,
            "original_z_fit_sha256": offset.sha256_array(rows["z_fit_original"]),
            "original_z_query_sha256": offset.sha256_array(rows["z_query_original"]),
            "matrix_hashes": matrices,
        }
    old = {
        "origins": origin_freezes
    }
    return run_root, archive_root, rows, old


def test_successor_cache_reconciles_unicode_ids_and_original_hash_contract(tmp_path, monkeypatch):
    run_root, archive_root, _, old = _write_migration_fixture(tmp_path, monkeypatch)
    result = verifier.verify_schema_successor_cache(tmp_path, run_root, archive_root, old)
    assert result["status"] == "PASS"
    assert result["old_row_metadata_bytes_recovered"] is False
    assert result["origins"]["E67"]["old_frozen_row_bytes_available"] is False


def test_successor_cache_rejects_row_reordering(tmp_path, monkeypatch):
    run_root, _, rows, old = _write_migration_fixture(tmp_path, monkeypatch)
    reordered = dict(rows)
    reordered["fit_rows"] = rows["fit_rows"][::-1]
    reordered["fit_record_ids"] = rows["fit_record_ids"][::-1]
    np.savez(run_root / "E67" / "frozen_rows.npz", **reordered)
    result = verifier.verify_schema_successor_cache(
        tmp_path, run_root, run_root / "archive", old)
    assert result["status"] == "FAIL"
    assert any("fit_rows" in issue or "fit_record_ids" in issue for issue in result["failures"])


def test_successor_cache_rejects_non_id_value_mutation(tmp_path, monkeypatch):
    run_root, _, rows, old = _write_migration_fixture(tmp_path, monkeypatch)
    changed = dict(rows)
    changed["p_p"] = np.asarray([0.41])
    np.savez(run_root / "E67" / "frozen_rows.npz", **changed)
    result = verifier.verify_schema_successor_cache(
        tmp_path, run_root, run_root / "archive", old)
    assert result["status"] == "FAIL"
    assert any("p_p" in issue for issue in result["failures"])


def test_successor_rebind_requires_one_original_reservation_and_zero_starts():
    snapshot = {"event_rows": 363, "global_cap": 149, "fit_starts": 138}
    reservation = {"fit_ids": list(FIT_IDS), "freeze_signature": "parent-freeze",
                   "source_commit": "parent-commit"}
    _assert_successor_reservation_untouched(
        snapshot=snapshot, expected_snapshot=snapshot,
        ledger_sha="post-reserve-ledger", expected_ledger_sha="post-reserve-ledger",
        reservations=[reservation], expected_parent_signature="parent-freeze",
        expected_parent_source="parent-commit", own_starts=[])
    with np.testing.assert_raises(PilotError):
        _assert_successor_reservation_untouched(
            snapshot=snapshot, expected_snapshot=snapshot,
            ledger_sha="post-reserve-ledger", expected_ledger_sha="post-reserve-ledger",
            reservations=[reservation], expected_parent_signature="parent-freeze",
            expected_parent_source="parent-commit", own_starts=[{"fit_id": FIT_IDS[0]}])
    with np.testing.assert_raises(PilotError):
        _assert_successor_reservation_untouched(
            snapshot=snapshot, expected_snapshot=snapshot,
            ledger_sha="post-reserve-ledger", expected_ledger_sha="post-reserve-ledger",
            reservations=[reservation, reservation], expected_parent_signature="parent-freeze",
            expected_parent_source="parent-commit", own_starts=[])


def test_reconstruct_original_logit_uses_frozen_float64_components():
    base = sparse.csr_matrix([[1.0, 0.0], [0.25, 2.0]])
    position = sparse.csr_matrix([[0.0, 1.0], [3.0, 0.0]])
    coef = np.asarray([0.2, -0.4])
    intercept = np.asarray([0.1])
    gamma = np.asarray([0.3, -0.2])

    z_bag, z_original = reconstruct_original_pos_logit(
        base, coef, intercept, position, gamma
    )

    expected_bag = intercept[0] + base @ coef
    expected = np.asarray(expected_bag + position @ gamma).reshape(-1)
    np.testing.assert_array_equal(z_bag, np.asarray(expected_bag).reshape(-1))
    np.testing.assert_array_equal(z_original, expected)


def test_zero_gamma_matches_incumbent_and_preserves_exact_routes():
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    z = np.asarray([-0.2, 0.1, 0.3, -0.4])
    present = np.asarray([True, True, False, True])
    no_known = np.asarray([False, True, True, True])
    p_a02 = np.asarray([0.8, 0.2, 0.7, 0.6])
    p_p = np.asarray([0.1, 0.3, 0.4, 0.9])
    p_pos = expit(z)
    mix = np.asarray([0.5 * p_pos[i] + 0.5 * p_a02[i] for i in range(4)])
    mix[2] = p_p[2]

    check = zero_gamma_equivalence(
        x, z, present_mask=present, p_a02=p_a02, p_p=p_p, mix50_original=mix
    )
    assert check["pass"]

    _, routed = route_candidate(
        x, z, np.asarray([0.15, -0.05]), present_mask=present,
        no_known_bigram=no_known, p_pos_original=p_pos, p_a02=p_a02,
        p_p=p_p, mix50_original=mix,
    )
    assert routed[2] == p_p[2]
    np.testing.assert_array_equal(routed[[1, 3]], mix[[1, 3]])


def test_rejects_nonfinite_and_misaligned_inputs():
    with np.testing.assert_raises(ValueError):
        reconstruct_original_pos_logit(
            sparse.csr_matrix([[np.nan]]), [1.0], [0.0],
            sparse.csr_matrix([[1.0]]), [1.0],
        )
    with np.testing.assert_raises(ValueError):
        reconstruct_original_pos_logit(
            sparse.csr_matrix([[1.0]]), [1.0], [0.0],
            sparse.csr_matrix([[1.0, 0.0]]), [1.0],
        )


def test_fit_label_reader_skips_query_rows(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text(
        "record_id,artifact_signal,signal_stream\n"
        "r0,1,alpha\n"
        "r1,0,beta\n"
        "r2,1,gamma\n"
        "r3,0,delta\n",
        encoding="utf-8",
    )
    np.testing.assert_array_equal(_read_fit_labels_only(path, np.asarray([0, 2])), [1, 1])


def test_gate_uses_frozen_tolerance_and_unknown_bound_shape():
    assert _gate("min", 0.0005 - 5e-13, minimum=0.0005)["pass"]
    assert not _gate("min", 0.0005 - 2e-12, minimum=0.0005)["pass"]
    assert _gate("max", 0.003 + 5e-13, maximum=0.003)["pass"]
    assert not _gate("max", 0.003 + 2e-12, maximum=0.003)["pass"]
