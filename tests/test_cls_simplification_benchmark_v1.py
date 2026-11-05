from __future__ import annotations

import csv

import numpy as np
import pytest

from scripts import cls_simplification_benchmark_v1 as candidate


def test_no_pos_routes_exact_threshold_ties_by_availability():
    p = np.asarray([0.499999, 0.50, candidate.MISSING_THRESHOLD,
                    candidate.MISSING_THRESHOLD - 1e-9, 0.90])
    missing = np.asarray([False, False, True, True, False])
    assert candidate.route_no_pos(p, missing).tolist() == [0, 1, 1, 0, 1]


@pytest.mark.parametrize("probability,missing", [
    ([0.5], []),
    ([float("nan")], [False]),
    ([1.1], [False]),
    ([], []),
])
def test_invalid_route_inputs_fail_closed(probability, missing):
    with pytest.raises(ValueError):
        candidate.route_no_pos(probability, missing)


def test_write_once_csv_uses_official_schema_and_refuses_changed_replay(tmp_path):
    ids = ["T-1", "T-2", "T-3"]
    labels = np.asarray([0, 1, 0], dtype=np.int8)
    path = tmp_path / "candidate.csv"
    digest = candidate.write_csv_once(path, ids, labels)
    assert candidate.sha256_file(path) == digest
    assert candidate._read_submission(path, ids).tolist() == [0, 1, 0]
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["record_id", "artifact_signal"]
    with pytest.raises(FileExistsError):
        candidate.write_csv_once(path, ids, 1 - labels)


def test_test_reader_materializes_only_id_and_stream_features(tmp_path):
    path = tmp_path / "Test.csv"
    path.write_text(
        "record_id,signal_stream,artifact_signal\n"
        "T-1,alpha,NOT_A_TARGET_INPUT\n"
        "T-2,   ,ALSO_NOT_A_TARGET_INPUT\n",
        encoding="utf-8",
    )
    ids, missing, fields_read = candidate._read_ids_missing(path)
    assert ids == ["T-1", "T-2"]
    assert missing.tolist() == [False, True]
    assert fields_read == ["record_id", "signal_stream"]


def test_duplicate_scan_matches_only_exact_official_id_vector(tmp_path, monkeypatch):
    ids = ["T-1", "T-2"]
    labels = np.asarray([0, 1], dtype=np.int8)
    root = tmp_path / "repo"
    submissions = root / "submissions"
    submissions.mkdir(parents=True)
    prior = submissions / "prior.csv"
    prior.write_text("record_id,artifact_signal\nT-1,0\nT-2,1\n", encoding="utf-8")
    other = submissions / "other.csv"
    other.write_text("record_id,artifact_signal\nT-1,1\nT-2,0\n", encoding="utf-8")
    monkeypatch.setattr(candidate, "ROOT", root)
    monkeypatch.setattr(candidate, "OUTPUT_CSV", submissions / "new.csv")
    assert candidate.scan_local_duplicate_vectors(ids, labels) == ["submissions/prior.csv"]


def test_cli_self_test_is_synthetic_only():
    assert candidate.self_test()["status"] == "PASS"
