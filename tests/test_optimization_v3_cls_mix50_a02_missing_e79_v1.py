import csv

import numpy as np

from scripts import optimization_v3_cls_mix50_a02_missing_e79_v1 as policy


def test_fixed_route_formula_preserves_present_and_uses_a02_on_missing():
    p_p = np.array([0.2, 0.8, 0.4, 0.6], dtype=np.float64)
    p_a = np.array([0.8, 0.4, 0.5, 0.9], dtype=np.float64)
    missing = np.array([False, False, True, True])

    baseline, candidate = policy.route_probabilities(p_p, p_a, missing)

    np.testing.assert_allclose(baseline, np.array([0.5, 0.6, 0.4, 0.6]), atol=1e-15, rtol=0)
    np.testing.assert_allclose(candidate, np.array([0.5, 0.6, 0.5, 0.9]), atol=1e-15, rtol=0)
    assert np.array_equal(candidate[~missing], baseline[~missing])
    assert np.array_equal(candidate[missing], p_a[missing])


def test_threshold_is_inclusive_and_missing_changes_are_reported():
    baseline, candidate = policy.route_probabilities(
        np.array([0.4, 0.8, 0.6, 0.2]),
        np.array([0.6, 0.2, 0.5, 0.9]),
        np.array([False, False, True, True]),
    )
    baseline_labels = policy.binary_labels(baseline)
    candidate_labels = policy.binary_labels(candidate)
    assert baseline_labels.tolist() == [1, 1, 1, 0]
    assert candidate_labels.tolist() == [1, 1, 1, 1]
    assert np.array_equal(baseline_labels[:2], candidate_labels[:2])
    assert policy.compare_missing_labels(
        baseline_labels, candidate_labels, np.array([False, False, True, True])
    ) == {
        "missing_rows": 2,
        "changed_missing_count": 1,
        "newly_positive_missing_count": 1,
        "newly_negative_missing_count": 0,
        "baseline_missing_positive_count": 1,
        "candidate_missing_positive_count": 2,
        "candidate_total_positive_count": 4,
    }


def test_full_label_duplicate_scan_aligns_by_id_not_csv_order(tmp_path):
    ids = ["id-1", "id-2", "id-3"]
    labels = np.array([1, 0, 1], dtype=np.int8)
    csv_path = tmp_path / "old_classification.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows([["id-3", 1], ["id-1", 1], ["id-2", 0]])

    result = policy.scan_duplicate_full_vectors(
        ids, labels, root=tmp_path, exclude=tmp_path / "new_candidate.csv"
    )

    assert result["full_test_classification_vectors_scanned"] == 1
    assert result["duplicate_found"] is True
    assert len(result["exact_duplicate_matches"]) == 1
