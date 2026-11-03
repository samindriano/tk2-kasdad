import numpy as np
import pytest
from scipy import sparse

from src.models.regression_rnbr_impact_match import (
    ImpactMatchError,
    exact_impact_top32,
    load_bundle,
    persist_bundle_once,
    save_npz_once,
    shuffle_residuals_within_exact_cells,
    validate_same_impact_geometry,
)


def test_exact_impact_search_restricts_full_donor_bank_and_stable_ties():
    query = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    donors = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    result = exact_impact_top32(
        query, donors,
        query_impact=np.array([7.5, 9.0, np.nan]),
        donor_impact=np.array([7.5, 7.5, 9.0, 8.0]),
        donor_rows=np.array([20, 10, 30, 5]),
        query_stream_missing=np.array([False, False, False]),
    )
    assert result["neighbor_rows"][0, :2].tolist() == [10, 20]
    assert result["neighbor_count"].tolist() == [2, 1, 0]
    assert result["neighbor_rows"][1, 0] == 30
    assert result["neighbor_rows"][2, 0] == -1
    assert result["neighbor_similarities"][0, :2].tolist() == [1.0, 1.0]


def test_exact_impact_search_rejects_infinite_or_misaligned_impact():
    matrix = sparse.eye(1, format="csr")
    with pytest.raises(ImpactMatchError, match="Infinite query impact"):
        exact_impact_top32(
            matrix, matrix, query_impact=[np.inf], donor_impact=[1.0],
            donor_rows=[3], query_stream_missing=[False],
        )
    with pytest.raises(ImpactMatchError, match="not aligned"):
        exact_impact_top32(
            matrix, matrix, query_impact=[1.0], donor_impact=[1.0, 2.0],
            donor_rows=[3], query_stream_missing=[False],
        )


def test_cell_shuffle_is_deterministic_and_preserves_exact_cell_multisets():
    residual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cycles = np.array([56, 56, 56, 56, 57, 57])
    impact = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    baseline = np.array([9.0, 9.0, 9.0, 19.0, 9.0, 9.0])
    rows = np.array([10, 11, 12, 13, 14, 15])
    shuffled, source_rows, audit = shuffle_residuals_within_exact_cells(
        residual, cycles, impact, baseline, rows
    )
    again, again_sources, again_audit = shuffle_residuals_within_exact_cells(
        residual, cycles, impact, baseline, rows
    )
    assert np.array_equal(shuffled, again)
    assert np.array_equal(source_rows, again_sources)
    assert audit == again_audit
    for key in ((56, 1.0, 0), (56, 1.0, 1), (57, 1.0, 0)):
        mask = (cycles == key[0]) & (impact == key[1]) & (np.minimum(9, baseline // 10).astype(int) == key[2])
        assert sorted(shuffled[mask].tolist()) == sorted(residual[mask].tolist())
    assert np.array_equal(rows[np.searchsorted(rows, source_rows)], source_rows)
    assert audit["source_positions_moved"] > 0


def test_geometry_validator_rejects_cross_impact_and_wrong_time_neighbor():
    args = dict(
        query_rows=np.array([90]), query_cycles=np.array([68]), query_impact=np.array([5.5]),
        query_ids=np.array(["q"], dtype="U1"), query_stream_missing=np.array([False]),
        donor_rows=np.array([30, 31]), donor_cycles=np.array([67, 66]),
        donor_impact=np.array([5.5, 6.5]), donor_ids=np.array(["d0", "d1"], dtype="U2"),
        origin=67, max_neighbors=2, neighbor_rows=np.array([[30, -1]], dtype=np.int64),
        neighbor_ids=np.array([["d0", ""]], dtype="U2"),
        neighbor_similarities=np.array([[0.9, 0.0]]), neighbor_count=np.array([1]),
    )
    validate_same_impact_geometry(**args)
    with pytest.raises(ImpactMatchError, match="exactly match"):
        changed = dict(args)
        changed["query_impact"] = np.array([5.5000000001])
        validate_same_impact_geometry(**changed)
    with pytest.raises(ImpactMatchError, match="invalid"):
        changed = dict(args)
        changed["donor_cycles"] = np.array([67, 68])
        changed["neighbor_rows"] = np.array([[31, -1]], dtype=np.int64)
        changed["neighbor_ids"] = np.array([["d1", ""]], dtype="U2")
        validate_same_impact_geometry(**changed)


def test_pickle_free_atomic_persistence_readback_and_resume(tmp_path):
    arrays = {
        "source_rows": np.array([3, 8], dtype=np.int64),
        "ids": np.array(["id-α", "id-東京"], dtype="U5"),
        "prediction": np.array([42.5, 77.25], dtype=np.float64),
    }
    path = tmp_path / "geometry.npz"
    first_hash = save_npz_once(path, **arrays)
    assert save_npz_once(path, **arrays) == first_hash
    with np.load(path, allow_pickle=False) as archive:
        assert all(np.array_equal(archive[key], value) for key, value in arrays.items())
    with pytest.raises(ImpactMatchError, match="differs"):
        save_npz_once(path, **{**arrays, "prediction": arrays["prediction"] + 1})


def test_bundle_receipt_and_durable_resume_are_exact(tmp_path):
    arrays = {"prediction": np.array([12.0, 34.0]),
              "rows": np.array([4, 9], dtype=np.int64)}
    folder = tmp_path / "fit"
    receipt = persist_bundle_once(
        folder, fit_id="RNBR_IMPACT_MATCH_E67_OBSERVED", freeze_signature="f" * 64,
        arrays=arrays, metadata={"source_commit": "c" * 40, "synthetic": True},
    )
    loaded, checked = load_bundle(
        folder, fit_id="RNBR_IMPACT_MATCH_E67_OBSERVED", freeze_signature="f" * 64
    )
    assert receipt == checked
    assert all(np.array_equal(loaded[key], value) for key, value in arrays.items())
    assert persist_bundle_once(
        folder, fit_id="RNBR_IMPACT_MATCH_E67_OBSERVED", freeze_signature="f" * 64,
        arrays=arrays, metadata={"source_commit": "c" * 40, "synthetic": True},
    ) == receipt
    with pytest.raises(ImpactMatchError, match="arrays differ"):
        persist_bundle_once(
            folder, fit_id="RNBR_IMPACT_MATCH_E67_OBSERVED", freeze_signature="f" * 64,
            arrays={**arrays, "prediction": arrays["prediction"] + 1},
            metadata={"source_commit": "c" * 40, "synthetic": True},
        )

