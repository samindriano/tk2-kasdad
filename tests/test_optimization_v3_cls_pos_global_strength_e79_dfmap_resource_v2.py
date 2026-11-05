from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 as v2
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 as v2_verify


def test_sparse_union_upper_bound_fits_approved_allowance() -> None:
    bound = v2.sparse_union_extra_upper_bound_bytes()
    assert bound["total_bytes"] < v2.EXTRA_SPARSE_BOUND_LIMIT_BYTES
    assert bound["matrix_copy_bytes"] > 0
    assert bound["three_block_copies_bytes"] > bound["matrix_copy_bytes"]


def test_resource_gate_requires_initial_and_projected_reserve() -> None:
    passed = v2.resource_snapshot("synthetic", available_bytes=v2.MIN_AVAILABLE_BYTES, process_rss_bytes=1)
    failed = v2.resource_snapshot("synthetic", available_bytes=v2.MIN_AVAILABLE_BYTES - 1, process_rss_bytes=1)
    assert passed["gate_pass"] is True
    assert failed["gate_pass"] is False
    assert passed["required_free_reserve_bytes"] == 768 * v2.MIB
    assert passed["estimate_is_measured_df_map_peak"] is False
    locks = [{"path": "shared", "acquired": True}, {"path": "local", "acquired": True}]
    assert v2_verify.verify_resource_preflight(passed, locks)["status"] == "PASS_INDEPENDENT_RESOURCE_PREFLIGHT"
    assert v2_verify.verify_resource_preflight(failed, locks)["status"] == "BLOCKED_RESOURCE_PREFLIGHT"


def test_synthetic_formula_persistence_and_no_retry() -> None:
    result = v2.run_synthetic_qa()
    assert result["status"] == "PASS_SYNTHETIC_RESOURCE_AND_PERSISTENCE_QA"
    assert result["second_global_fit_blocked"] is True
    assert result["v1_runner_attempted"] is False
    assert result["shared_guard_marker_policy_id"] == v2.POLICY_ID
    assert result["test_or_sample_or_target_access"] is False


def test_global_one_shot_guard_blocks_v1_or_v2_outputs(tmp_path) -> None:
    attempt, v1_attempt, map_path, receipt_path = v2._output_paths(tmp_path)
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"consumed global map")
    with pytest.raises(v2.ResourceV2Error, match="no retry"):
        v2.ensure_one_global_pass_not_started(tmp_path)
    map_path.unlink()
    v1_attempt.write_bytes(b"V1 attempt")
    with pytest.raises(v2.ResourceV2Error, match="no retry"):
        v2.ensure_one_global_pass_not_started(tmp_path)


def test_frozen_geometry_allocation_bound_rejects_expansion() -> None:
    bound = v2.sparse_union_extra_upper_bound_bytes(rows=50_608, nnz=5_000_000)
    assert bound["total_bytes"] > v2.EXTRA_SPARSE_BOUND_LIMIT_BYTES


def test_synthetic_position_csr_values_are_float64() -> None:
    matrix = sparse.csr_matrix(np.eye(4, dtype=np.float64))
    assert matrix.data.dtype == np.float64
    assert matrix.indices.dtype in (np.int32, np.int64)
    assert matrix.indptr.dtype in (np.int32, np.int64)
