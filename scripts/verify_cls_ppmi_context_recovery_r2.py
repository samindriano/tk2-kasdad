"""Independent recovery verifier with the production root-anchor schema."""
from __future__ import annotations

from typing import Any, Mapping

from verify_cls_ppmi_context_correction_r1 import (
    IndependentVerificationError,
    verify_in_memory as _verify_views_and_gates,
)

PRODUCTION_ANCHOR_VIEWS = ("PRIMARY_UNIQUE", "E67_H12", "E75_W6")
PRODUCTION_ANCHOR_KEYS = (
    "champion", "candidate_real", "control_shuffled", "gain_real_vs_champion",
)


def _anchor_metric(scored: Mapping[str, Any], view_name: str, anchor_name: str) -> float:
    view_metrics = scored["views"][view_name]["metrics"]
    if anchor_name in {"champion", "candidate_real", "control_shuffled"}:
        return float(view_metrics[anchor_name]["macro_f1"])
    if anchor_name == "gain_real_vs_champion":
        return float(view_metrics["gains"]["candidate_real_minus_champion_macro_f1"])
    raise IndependentVerificationError(f"Unknown authorized macro-F1 anchor key {anchor_name}")


def verify_production_anchors(scored: Mapping[str, Any],
                              anchors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Check the exact frozen root schema, including its derived gain anchor."""
    if set(anchors) != set(PRODUCTION_ANCHOR_VIEWS):
        raise IndependentVerificationError("Authorized root anchor views differ from production schema")
    checked: dict[str, Any] = {}
    for view_name in PRODUCTION_ANCHOR_VIEWS:
        expected_by_name = anchors[view_name]
        if set(expected_by_name) != set(PRODUCTION_ANCHOR_KEYS):
            unknown = set(expected_by_name) - set(PRODUCTION_ANCHOR_KEYS)
            missing = set(PRODUCTION_ANCHOR_KEYS) - set(expected_by_name)
            raise IndependentVerificationError(
                f"Authorized anchor keys differ for {view_name}; unknown={sorted(unknown)}, missing={sorted(missing)}")
        checked[view_name] = {}
        for anchor_name in PRODUCTION_ANCHOR_KEYS:
            expected = float(expected_by_name[anchor_name])
            actual = _anchor_metric(scored, view_name, anchor_name)
            passed = abs(actual - expected) <= 1e-12
            checked[view_name][anchor_name] = {
                "expected": expected, "actual": actual, "pass": passed,
                "metric": ("candidate_real_minus_champion_macro_f1"
                           if anchor_name == "gain_real_vs_champion" else "macro_f1"),
            }
            if not passed:
                raise IndependentVerificationError(
                    f"Authorized macro-F1 anchor mismatch: {view_name}.{anchor_name}")
    return checked


def verify_in_memory_recovery_r2(predictions: Mapping[int, Mapping[str, Any]],
                                 labels_by_id: Mapping[str, int],
                                 scored: Mapping[str, Any],
                                 anchors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reuse the established six-view independent recomputation, then check R2 anchors."""
    # The R1 verifier independently rebuilds all metrics, row/ID identities, subgroups,
    # missing routes, overlaps, and gates. Its production-anchor loop had the sole R1 bug;
    # pass no anchors there and validate the frozen production schema below.
    rebuilt = _verify_views_and_gates(predictions, labels_by_id, scored, anchors={})
    rebuilt["macro_f1_root_anchors"] = verify_production_anchors(scored, anchors)
    rebuilt["root_anchor_schema"] = {
        "views": list(PRODUCTION_ANCHOR_VIEWS),
        "keys": list(PRODUCTION_ANCHOR_KEYS),
        "gain_real_vs_champion_metric": "candidate_real_minus_champion_macro_f1",
    }
    return rebuilt


def dual_e75_gate_readback(scored: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the original control-vs-champion gate separate from the R1 amendment."""
    gains = scored["views"]["E75_W6"]["metrics"]["gains"]
    original = float(gains["control_shuffled_minus_champion_macro_f1"])
    amended = float(gains["candidate_real_minus_control_shuffled_macro_f1"])
    return {
        "original_root_campaign": {
            "comparison": "control_shuffled_minus_champion_macro_f1",
            "gain": original,
            "nonnegative": original >= 0.0,
            "source": "scripts/cls_ppmi_context_v1.py original E75 control-gain gate",
        },
        "r1_amended_comparison": {
            "comparison": "candidate_real_minus_control_shuffled_macro_f1",
            "gain": amended,
            "nonnegative": amended >= 0.0,
            "source": "ROOT_CORRECTED_SCORING_R1.json later amendment; reported separately",
        },
    }
