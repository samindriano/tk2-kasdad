"""Full-Train compatibility wrapper for the frozen C19 estimator recipe.

The historical C19 model constructor intentionally accepts only its temporal
pilot origins.  This wrapper reuses the unchanged implementation and permits
the one authorized full-Train cutoff (cycle 79).
"""
from __future__ import annotations

from src.models.optimization_v3_batch1_c19 import XGBoostSVD32Classifier


class C19FullTrainClassifier(XGBoostSVD32Classifier):
    """Use the original C19 implementation with all Train cycles through 79."""

    def __init__(self) -> None:
        # The base constructor initializes only recipe state; all model
        # parameters and fit/predict methods remain the historical C19 code.
        super().__init__(cutoff=67)
        self.cutoff = 79


__all__ = ["C19FullTrainClassifier"]
