"""Outcome-aware schema adapter for an independent post-fit MLP verifier rerun.

The frozen historical CatBoost sidecar stores fit-row count under `state.fit_rows`,
while the MLP's independent verifier expected a top-level `fit_rows` field. This
adapter translates that already-audited metadata shape in memory only, verifies
the structured preprocessor's row counts, and delegates all other checks to the
unchanged independent verifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scripts import optimization_v3_classification_sparse_mlp_v1_recovery_verify as recovery_verify


def _catboost_fit_rows_from_nested_metadata(metadata: dict[str, Any]) -> int:
    state = metadata.get("state", {})
    fit_rows = state.get("fit_rows")
    categories = state.get("fit_only_preprocessor", {}).get("category_counts", {})
    if not isinstance(fit_rows, int) or fit_rows <= 0 or not isinstance(categories, dict) or not categories:
        raise RuntimeError("Historical CatBoost metadata lacks nested fit-population audit")
    if any(int(item.get("training_rows", -1)) != fit_rows for item in categories.values()):
        raise RuntimeError("Historical CatBoost preprocessor fit rows differ from model fit rows")
    return fit_rows


def _adapt_sidecar_fit_rows(path: str | Path, metadata: Any) -> Any:
    source = Path(path)
    if source.parent.name != "dual_cls_catboost_v1" or not isinstance(metadata, dict):
        return metadata
    adapted = dict(metadata)
    adapted["fit_rows"] = _catboost_fit_rows_from_nested_metadata(adapted)
    return adapted


def verify_recovered() -> dict[str, Any]:
    independent = recovery_verify.independent
    original_json: Callable[[str | Path], Any] = independent._json

    def _json_with_nested_fit_rows(path: str | Path) -> Any:
        return _adapt_sidecar_fit_rows(path, original_json(path))

    independent._json = _json_with_nested_fit_rows
    try:
        return recovery_verify.verify_recovered()
    finally:
        independent._json = original_json


if __name__ == "__main__":
    import json
    print(json.dumps(verify_recovered(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
