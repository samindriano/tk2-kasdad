import csv
from pathlib import Path

import numpy as np

from scripts import optimization_v3_cls_missing_current_mix50_audit_v1 as audit


def test_fixed_candidate_keeps_present_mix50_and_uses_a02_only_when_missing():
    mix50 = np.array([0.7, 0.2, 0.5, 0.4], dtype=np.float64)
    p_a02 = np.array([0.1, 0.9, 0.75, 0.8], dtype=np.float64)
    missing = np.array([False, False, True, True])

    candidate = audit.candidate_probability(mix50, p_a02, missing)

    assert np.array_equal(candidate[~missing], mix50[~missing])
    assert np.array_equal(candidate[missing], p_a02[missing])
    assert np.array_equal(candidate, np.array([0.7, 0.2, 0.75, 0.8]))


def test_full_row_metric_uses_entire_view_denominator():
    labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    base = np.array([0.2, 0.8, 0.7, 0.4, 0.7, 0.1])
    candidate = np.array([0.2, 0.8, 0.7, 0.9, 0.8, 0.1])
    missing = np.array([False, False, False, True, True, False])
    row = audit._metric_record("synthetic", {
        "y": labels,
        "MIX50": base,
        "candidate": candidate,
        "missing": missing,
    })

    assert row["rows"] == 6
    assert row["missing_rows"] == 2
    assert row["present_rows"] == 4
    assert row["baseline"]["confusion_matrix"] == [[2, 1], [1, 2]]
    assert row["candidate"]["confusion_matrix"] == [[2, 1], [0, 3]]
    missing_base = audit._metric(labels[missing], base[missing])
    missing_candidate = audit._metric(labels[missing], candidate[missing])
    missing_delta = missing_candidate["macro_f1"] - missing_base["macro_f1"]
    assert row["delta_macro_f1"] != missing_delta


def test_train_alignment_joins_by_id_and_reconciles_cycle_label_and_strip_empty(monkeypatch, tmp_path):
    train = tmp_path / "Train.csv"
    with train.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "cycle_code", "signal_stream", "artifact_signal"])
        writer.writerow(["b", 69, "  ", 1])
        writer.writerow(["a", 68, "alpha beta", 0])
    monkeypatch.setattr(audit, "TRAIN", train)
    expected = {"a": 68, "b": 69}

    labels, missing = audit._read_train_labels_and_missing(
        np.array(["b", "a"]), expected, train_sha256=audit.sha256_file(train)
    )

    assert labels.tolist() == [1, 0]
    assert missing.tolist() == [True, False]
