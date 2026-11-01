"""R25 unnormalized word-count sparse regression candidate.

R25 reuses the frozen R05 structured/OHE, missing-indicator, sparse LightGBM,
and rank50 blend path. Its sole recipe change is CountVectorizer(binary=False).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.models import optimization_v3_batch1_r05 as r05
from src.models.regression_research_text import _text_documents


FEATURE_NAMES = r05.FEATURE_NAMES
COUNT_PARAMETERS: dict[str, Any] = {**r05.COUNT_PARAMETERS, "binary": False}


class R25UnnormalizedWordCountHybrid(r05.R05BinaryPresenceSparseHybrid):
    """Inherited R05 pipeline with raw word counts instead of binary presence."""

    @staticmethod
    def _fit_vectorizer(documents: list[str]):
        if not any(documents):
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "all_training_text_empty"
        vectorizer = CountVectorizer(**COUNT_PARAMETERS)
        try:
            matrix = vectorizer.fit_transform(documents).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            if not any(token in message for token in ("empty vocabulary", "after pruning, no terms remain")):
                raise
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "empty_vocabulary_after_filters"
        return vectorizer, matrix, None

    def fit(self, X: pd.DataFrame, y: Any) -> "R25UnnormalizedWordCountHybrid":
        super().fit(X, y)
        self.state["id"] = "V3_BATCH1_R25_UNNORMALIZED_WORD_COUNTS_SPARSE_L1"
        self.state["recipe"] = "CountVectorizer raw word 1-2 counts + structured/OHE + missing indicator; LightGBM regression_l1"
        self.state["only_intended_change"] = "R05 binary word presence -> unnormalized CountVectorizer counts; min_df=2 and cap=20,000 retained"
        self.state["state_sha256"] = r05._json_digest({
            key: value for key, value in r05._safe_builtin(self.state).items() if key != "state_sha256"
        })
        return self


def fit_predict(
    fit_X: pd.DataFrame,
    fit_y: Any,
    score_X: pd.DataFrame,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    if card_id != "R25":
        raise ValueError(f"R25 adapter received card {card_id!r}")
    required = {"baseline_prediction", "ridge25_prediction", "baseline_sparse_raw", "cycle_code"}
    if not required.issubset(controls):
        raise ValueError(f"R25 frozen-control bridge missing {sorted(required - set(controls))}")
    model = R25UnnormalizedWordCountHybrid().fit(fit_X, fit_y)
    raw_candidate = model.predict(score_X)
    cycles = np.asarray(controls["cycle_code"], dtype=np.int64)
    baseline_reconstructed = r05.compose_r05_prediction(
        controls["ridge25_prediction"], controls["baseline_sparse_raw"], cycles
    )
    baseline_saved = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if baseline_saved.shape != baseline_reconstructed.shape or not np.allclose(
        baseline_saved, baseline_reconstructed, atol=1e-10, rtol=0.0
    ):
        raise ValueError("R25 frozen champion control does not reconstruct from row-aligned caches")
    prediction = r05.compose_r05_prediction(controls["ridge25_prediction"], raw_candidate, cycles)
    mapped = r05.rank50_prediction(raw_candidate, cycles)
    return {
        "raw_candidate": raw_candidate,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_X)),
            "score_rows": int(len(score_X)),
            "policy": "clip(0.75*Ridge25 + 0.25*rank50(raw R25 candidate),0,100)",
            "rank_mapping": "per complete future cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
            "vectorizer_fit_calls": 1,
            "predictor_fit_calls": 1,
            "validation_labels_passed_to_estimator": False,
            "test_data_used": False,
        },
        "state": model.audit_state(),
        "audit_arrays": {"candidate_rank50": mapped},
    }


__all__ = ["COUNT_PARAMETERS", "FEATURE_NAMES", "R25UnnormalizedWordCountHybrid", "fit_predict"]
