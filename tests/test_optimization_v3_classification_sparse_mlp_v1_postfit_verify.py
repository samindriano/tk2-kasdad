from __future__ import annotations

from pathlib import Path

import pytest

from scripts.optimization_v3_classification_sparse_mlp_v1_postfit_verify import (
    _adapt_sidecar_fit_rows,
    _catboost_fit_rows_from_nested_metadata,
)


def test_nested_catboost_fit_rows_are_checked_and_adapted_in_memory():
    metadata = {
        "state": {
            "fit_rows": 12_020,
            "fit_only_preprocessor": {
                "category_counts": {
                    "org_code": {"training_rows": 12_020},
                    "module_code": {"training_rows": 12_020},
                    "pattern_code": {"training_rows": 12_020},
                }
            },
        }
    }
    adapted = _adapt_sidecar_fit_rows(
        Path("experiments/runs/dual_cls_catboost_v1/fit55.json"), metadata
    )
    assert adapted["fit_rows"] == 12_020
    assert "fit_rows" not in metadata
    assert _catboost_fit_rows_from_nested_metadata(metadata) == 12_020


def test_nested_catboost_population_mismatch_fails_closed():
    metadata = {
        "state": {
            "fit_rows": 12_020,
            "fit_only_preprocessor": {
                "category_counts": {"org_code": {"training_rows": 12_019}}
            },
        }
    }
    with pytest.raises(RuntimeError, match="preprocessor fit rows differ"):
        _adapt_sidecar_fit_rows(
            Path("experiments/runs/dual_cls_catboost_v1/fit55.json"), metadata
        )
