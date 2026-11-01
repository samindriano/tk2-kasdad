"""N01 class-only continuation adapter with the existing pilot recipe intact."""

from __future__ import annotations

from src.models.research_next_n01 import ClassByVolumeSparseClassifier


CONTINUATION_CUTOFFS = (59, 63, 71, 75)


class N01ClassOnlyDiagnosticClassifier(ClassByVolumeSparseClassifier):
    """The frozen N01 adapter, permitting only the four authorized cutoffs.

    All fitting, preprocessing, prediction, and state-audit implementations are
    inherited unchanged. Only the constructor cutoff validation is extended.
    The caller must always pass ``weight_kind='class_only'``.
    """

    def __init__(self, cutoff: int, weight_kind: str = "class_only"):
        cutoff = int(cutoff)
        if cutoff not in CONTINUATION_CUTOFFS:
            raise ValueError(f"Diagnostic continuation cutoff must be in {CONTINUATION_CUTOFFS}")
        if weight_kind != "class_only":
            raise ValueError("N01 diagnostic continuation accepts only the frozen class_only recipe")
        self.cutoff = cutoff
        self.weight_kind = "class_only"
        self._fitted = False

