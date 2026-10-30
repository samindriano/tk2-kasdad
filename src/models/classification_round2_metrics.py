"""Round 2 threshold selection and prediction evaluation.

This module consumes already-created inner/outer probability vectors. It never
fits estimators or writes artifacts. Threshold choice is derived only from the
explicit inner vectors passed to :func:`select_policies`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.classification_evaluation import window_indices
from src.models.classification_metrics import binary_metrics, reliability, subgroup_masks
from src.models.round1_evaluation import array_digest


POLICY_IDS = ("fixed_0.5", "inner_pooled", "inner_shrink50", "inner_equal_cycle")
_GRID = tuple(float(value) for value in np.round(np.arange(0.10, 0.901, 0.01), 2))


def _validated_vectors(actual, probability, cycles=None):
    y = np.asarray(actual)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned one-dimensional binary labels and probabilities required.")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Finite probabilities in [0,1] required.")
    if cycles is None:
        return y.astype(int, copy=False), p, None
    cycle = np.asarray(cycles)
    if cycle.ndim != 1 or cycle.shape != y.shape or pd.isna(cycle).any():
        raise ValueError("Inner cycles must be a nonmissing one-dimensional vector aligned to labels.")
    return y.astype(int, copy=False), p, cycle


def _macro_f1_from_counts(y, prediction):
    """Fixed-label [0,1] macro F1 using confusion counts only."""
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=bool)
    if y.shape != prediction.shape or y.ndim != 1:
        raise ValueError("Aligned one-dimensional labels and predictions required.")
    f1s = []
    for label in (0, 1):
        predicted_label = ~prediction if label == 0 else prediction
        actual_label = y == label
        tp = int(np.count_nonzero(actual_label & predicted_label))
        fp = int(np.count_nonzero(~actual_label & predicted_label))
        fn = int(np.count_nonzero(actual_label & ~predicted_label))
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return float((f1s[0] + f1s[1]) / 2)


def _score_at_threshold(y, p, cycles, threshold):
    prediction = p >= threshold
    pooled = _macro_f1_from_counts(y, prediction)
    if not len(y):
        return pooled, None
    cycle_scores = [
        _macro_f1_from_counts(y[cycles == cycle], prediction[cycles == cycle])
        for cycle in np.unique(cycles)
    ]
    return pooled, float(np.mean(cycle_scores)) if cycle_scores else None


def _threshold_config(config):
    if not isinstance(config, Mapping):
        raise ValueError("Threshold-selection config must be a mapping.")
    if "threshold_selection" in config:
        config = config["threshold_selection"]
    elif "inner_threshold" in config:
        config = config["inner_threshold"]
    grid = np.asarray(config.get("grid", _GRID), dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or not ((grid >= 0) & (grid <= 1)).all()):
        raise ValueError("Threshold grid must be a nonempty finite vector in [0,1].")
    minimum_rows = config.get("minimum_inner_validation_rows", 200)
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, (int, np.integer)) or minimum_rows < 0:
        raise ValueError("minimum_inner_validation_rows must be a nonnegative integer.")
    return config, grid, int(minimum_rows)


def _choose_grid_threshold(y, p, cycles, grid, score_kind):
    scores = []
    for threshold in grid:
        if score_kind == "pooled":
            score = _macro_f1_from_counts(y, p >= threshold)
        else:
            score = float(np.mean([
                _macro_f1_from_counts(y[cycles == cycle], p[cycles == cycle] >= threshold)
                for cycle in np.unique(cycles)
            ]))
        scores.append((float(threshold), score))
    best = max(score for _, score in scores)
    tied = [(threshold, score) for threshold, score in scores if abs(score - best) <= 1e-12]
    threshold, score = min(tied, key=lambda item: (round(abs(item[0] - 0.5), 12), item[0]))
    return threshold, float(score)


def select_policies(y_inner, p_inner, cycles_inner, config, eligible=True):
    """Select the four declared cutoffs from inner validation predictions only.

    All scoring in the grid loop is based on confusion counts. Average precision
    and other probability metrics are not recomputed for each candidate cutoff.
    """
    y, p, cycles = _validated_vectors(y_inner, p_inner, cycles_inner)
    threshold_config, grid, minimum_rows = _threshold_config(config)
    cycle_values = np.unique(cycles)
    both_classes = len(np.unique(y)) == 2
    fallback_reason = None
    if not eligible:
        fallback_reason = "Inner threshold selection is not eligible for this recipe."
    elif len(y) < minimum_rows:
        fallback_reason = "Insufficient inner validation rows."
    elif not both_classes:
        fallback_reason = "Inner validation does not contain both actual classes."
    elif len(cycle_values) < 4:
        fallback_reason = "Fewer than four inner validation cycles."

    pooled_threshold, pooled_selection_score = (0.5, None)
    equal_threshold, equal_selection_score = (0.5, None)
    if fallback_reason is None:
        pooled_threshold, pooled_selection_score = _choose_grid_threshold(y, p, cycles, grid, "pooled")
        equal_threshold, equal_selection_score = _choose_grid_threshold(y, p, cycles, grid, "equal_cycle")

    selected = {
        "fixed_0.5": 0.5,
        "inner_pooled": pooled_threshold,
        "inner_shrink50": 0.5 + 0.5 * (pooled_threshold - 0.5),
        "inner_equal_cycle": equal_threshold,
    }
    output = {}
    for policy_id, threshold in selected.items():
        pooled_f1, equal_cycle_f1 = _score_at_threshold(y, p, cycles, threshold)
        is_selected = policy_id != "fixed_0.5"
        output[policy_id] = {
            "threshold": float(threshold),
            "fallback": bool(is_selected and fallback_reason is not None),
            "reason": (
                "Prespecified fixed threshold."
                if not is_selected
                else fallback_reason
                if fallback_reason is not None
                else "Half shrink of inner pooled cutoff toward 0.5; no additional optimization or rounding."
                if policy_id == "inner_shrink50"
                else "Max inner pooled Macro F1; ties within 1e-12 closest to 0.5 then lower."
                if policy_id == "inner_pooled"
                else "Max mean inner-cycle fixed-label [0,1] Macro F1; ties within 1e-12 closest to 0.5 then lower."
            ),
            "inner_pooled_f1": pooled_f1,
            "inner_equal_cycle_f1": equal_cycle_f1,
            "selection_n": int(len(y)),
            "selection_cycles": int(len(cycle_values)),
        }
        if policy_id == "inner_pooled":
            output[policy_id]["selection_objective_f1"] = pooled_selection_score
        elif policy_id == "inner_equal_cycle":
            output[policy_id]["selection_objective_f1"] = equal_selection_score
    return output


def _recipe_ids(protocol, recipes):
    declared = protocol.get("recipes", [])
    values = declared if recipes is None else recipes
    if isinstance(values, (str, bytes)):
        raise ValueError("recipes must be a sequence of recipe IDs or recipe definitions.")
    result = []
    for value in values:
        recipe_id = value.get("id") if isinstance(value, Mapping) else value
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ValueError("Each recipe must have a nonempty string ID.")
        if recipe_id not in result:
            result.append(recipe_id)
    return result


def _policy_ids(protocol, policies):
    values = protocol.get("threshold_policies", POLICY_IDS) if policies is None else policies
    if isinstance(values, Mapping):
        values = list(values)
    if isinstance(values, (str, bytes)):
        raise ValueError("policies must be a sequence of policy IDs.")
    result = list(dict.fromkeys(values))
    unknown = [value for value in result if value not in POLICY_IDS]
    if unknown:
        raise ValueError(f"Unknown Round 2 policy IDs: {unknown}")
    return result


def _group_lookup(groups):
    lookup = {}
    for key, value in groups.items():
        try:
            normalized = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fit-group key must be an integer end cycle: {key!r}") from exc
        if normalized in lookup:
            raise ValueError(f"Duplicate fit-group key after integer normalization: {key!r}")
        if not isinstance(value, Mapping):
            raise ValueError(f"Fit group {key!r} must map recipe IDs to prediction bundles.")
        lookup[normalized] = value
    return lookup


def _aligned_group_predictions(bundle):
    if not isinstance(bundle, Mapping):
        raise ValueError("Prediction bundle must be a mapping.")
    required = ("rows", "probability", "inner_rows", "inner_probability", "thresholds", "meta")
    if any(key not in bundle for key in required):
        raise ValueError(f"Prediction bundle missing required fields: {required}")
    rows = np.asarray(bundle["rows"])
    p = np.asarray(bundle["probability"], dtype=float)
    inner_rows = np.asarray(bundle["inner_rows"])
    inner_p = np.asarray(bundle["inner_probability"], dtype=float)
    if rows.ndim != 1 or p.shape != rows.shape or rows.dtype.kind not in "iu":
        raise ValueError("Outer row IDs and probabilities must be aligned one-dimensional arrays.")
    if inner_rows.ndim != 1 or inner_p.shape != inner_rows.shape or inner_rows.dtype.kind not in "iu":
        raise ValueError("Inner row IDs and probabilities must be aligned one-dimensional arrays.")
    if len(np.unique(rows)) != len(rows) or len(np.unique(inner_rows)) != len(inner_rows):
        raise ValueError("Prediction bundles cannot contain duplicate row IDs.")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Outer probabilities must be finite and in [0,1].")
    if not np.isfinite(inner_p).all() or not ((inner_p >= 0) & (inner_p <= 1)).all():
        raise ValueError("Inner probabilities must be finite and in [0,1].")
    if not isinstance(bundle["thresholds"], Mapping) or not isinstance(bundle["meta"], Mapping):
        raise ValueError("Thresholds and metadata must be mappings.")
    return rows.astype(np.int64, copy=False), p, inner_rows.astype(np.int64, copy=False), inner_p


def _threshold(bundle, policy_id):
    try:
        value = bundle["thresholds"][policy_id]
    except KeyError as exc:
        raise ValueError(f"Prediction bundle is missing threshold policy {policy_id!r}.") from exc
    if isinstance(value, Mapping):
        value = value.get("threshold")
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid threshold for {policy_id!r}.")
    if policy_id == "fixed_0.5" and value != 0.5:
        raise ValueError("The fixed_0.5 policy must use threshold 0.5.")
    return value


def _window_prediction(frame, window, bundle, group_end):
    fit_rows, validation_rows = window_indices(frame, window)
    rows, probability, _, _ = _aligned_group_predictions(bundle)
    positions = {int(row): index for index, row in enumerate(rows)}
    missing = [int(row) for row in validation_rows if int(row) not in positions]
    if missing:
        raise ValueError(
            f"Fit group {group_end} lacks {len(missing)} required validation predictions "
            f"for window {window['id']}."
        )
    selected_positions = np.asarray([positions[int(row)] for row in validation_rows], dtype=int)
    p = probability[selected_positions]
    y = frame.iloc[validation_rows].artifact_signal.to_numpy()
    return fit_rows, validation_rows, y, p


def _cycle_records(frame, rows, y, p, threshold):
    cycle_codes = frame.iloc[rows].cycle_code.to_numpy()
    return [
        {
            "cycle": int(cycle),
            **binary_metrics(y[cycle_codes == cycle], p[cycle_codes == cycle], threshold),
        }
        for cycle in sorted(np.unique(cycle_codes))
    ]


def evaluate_predictions(frame, groups, protocol, recipes=None, policies=None):
    """Evaluate supplied outer predictions with already selected inner policies.

    The routine performs no model fitting, threshold selection, or file writing.
    Outer targets are read only for scoring after each prediction bundle has been
    aligned to its protocol validation rows.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame.")
    required_columns = {"cycle_code", "artifact_signal", "impact_load", "entry_friction",
                        "signal_stream", "org_code", "module_code", "pattern_code"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Evaluation frame is missing columns: {missing_columns}")
    if not frame.artifact_signal.isin([0, 1]).all():
        raise ValueError("Evaluation target must contain only binary labels 0 and 1.")
    recipe_ids = _recipe_ids(protocol, recipes)
    policy_ids = _policy_ids(protocol, policies)
    groups_by_end = _group_lookup(groups)
    windows = [("primary", item) for item in protocol.get("primary_windows", [])]
    windows += [("secondary", item) for item in protocol.get("secondary_windows", [])]
    jobs = []
    pooled_chunks = {}
    for family, window in windows:
        fit_range = window["fit"]
        fit_end = int(fit_range[1])
        fit_group = groups_by_end.get(fit_end)
        if fit_group is None:
            continue
        # This also rejects windows whose validation row positions are invalid.
        fit_rows, validation_rows = window_indices(frame, window)
        if len(fit_rows) == 0 or len(validation_rows) == 0:
            continue
        validation_frame = frame.iloc[validation_rows]
        y = validation_frame.artifact_signal.to_numpy()
        impact_cutpoints = frame.iloc[fit_rows].impact_load.quantile([1 / 3, 2 / 3]).to_numpy()
        masks = subgroup_masks(validation_frame, frame.iloc[fit_rows])
        for recipe_id in recipe_ids:
            bundle = fit_group.get(recipe_id)
            if bundle is None:
                continue
            _, rows, y, p = _window_prediction(frame, window, bundle, fit_end)
            threshold_map = {policy_id: _threshold(bundle, policy_id) for policy_id in policy_ids}
            policy_results = {}
            for policy_id, threshold in threshold_map.items():
                cycles = _cycle_records(frame, rows, y, p, threshold)
                metrics = binary_metrics(y, p, threshold)
                policy_results[policy_id] = {
                    "metrics": metrics,
                    "equal_cycle_macro_f1": float(np.mean([cycle["macro_f1"] for cycle in cycles])),
                    "per_cycle": cycles,
                    "subgroups": {
                        name: binary_metrics(y[mask], p[mask], threshold)
                        for name, mask in masks.items()
                    },
                }
                pooled_chunks.setdefault((family, policy_id, recipe_id), []).append(
                    (rows.copy(), y.copy(), p.copy(), np.full(len(rows), threshold), masks)
                )
            _, outer_probability, _, _ = _aligned_group_predictions(bundle)
            all_rows = np.asarray(bundle["rows"], dtype=np.int64)
            row_positions = {int(row): index for index, row in enumerate(all_rows)}
            selected_positions = np.asarray([row_positions[int(row)] for row in rows], dtype=int)
            group_meta = bundle["meta"]
            jobs.append({
                "family": family,
                "window": window["id"],
                "model": recipe_id,
                "fit_group": fit_end,
                "fit_cycles": list(fit_range),
                "validation_cycles": list(window["validation"]),
                "fit_rows": int(group_meta.get("fit_rows", len(fit_rows))),
                "validation_rows": int(len(rows)),
                "probability_sha256": array_digest(outer_probability[selected_positions]),
                "thresholds": {key: dict(bundle["thresholds"][key])
                               if isinstance(bundle["thresholds"][key], Mapping)
                               else {"threshold": float(bundle["thresholds"][key])}
                               for key in policy_ids},
                "impact_subgroup_cutpoints_from_fit": [float(value) for value in impact_cutpoints],
                "policies": policy_results,
            })

    aggregate = {
        family: {policy_id: {} for policy_id in policy_ids}
        for family in ("primary", "secondary")
    }
    for (family, policy_id, recipe_id), chunks in pooled_chunks.items():
        rows = np.concatenate([chunk[0] for chunk in chunks])
        y = np.concatenate([chunk[1] for chunk in chunks])
        p = np.concatenate([chunk[2] for chunk in chunks])
        threshold = np.concatenate([chunk[3] for chunk in chunks])
        if len(rows) != len(np.unique(rows)):
            raise ValueError(f"Duplicate validation rows within {family}/{recipe_id}.")
        related_jobs = [job for job in jobs if job["family"] == family and job["model"] == recipe_id]
        subgroup_names = list(chunks[0][4])
        subgroup_results = {}
        for name in subgroup_names:
            mask = np.concatenate([chunk[4][name] for chunk in chunks])
            subgroup_results[name] = binary_metrics(y[mask], p[mask], threshold[mask])
        aggregate[family].setdefault(policy_id, {})[recipe_id] = {
            "metrics": binary_metrics(y, p, threshold),
            "equal_cycle_macro_f1": float(np.mean([
                cycle["macro_f1"]
                for job in related_jobs
                for cycle in job["policies"][policy_id]["per_cycle"]
            ])),
            "subgroups": subgroup_results,
            "reliability": reliability(y, p),
        }
    return {"jobs": jobs, "aggregate": aggregate, "total_evaluation_jobs": len(jobs)}
