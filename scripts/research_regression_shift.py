"""Read-only feature-shift and support diagnostics for TP2 regression.

This script reads Train/Test predictor columns only and saved Round 2 metrics.
It does not fit a model, read Test labels, predict Test, or select a recipe.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.regression_research_common import (  # noqa: E402
    CONFIG, REPORT, freeze_stage, verify_stage,
)
from src.models.round1_models import (  # noqa: E402
    CATEGORICAL_FEATURES, MIXED_COUNT_FEATURES, NUMERIC_FEATURES,
    TELEMETRY_FEATURE, _category_array, _is_missing, _mixed_count_arrays,
    _number_array, _telemetry_arrays,
)
from src.utils.paths import PROJECT_ROOT, task_files  # noqa: E402


COUNT_FIELDS = tuple(MIXED_COUNT_FEATURES)
NUMERIC_FIELDS = tuple(NUMERIC_FEATURES)
BASE_FEATURES = (
    *NUMERIC_FIELDS, *COUNT_FIELDS, TELEMETRY_FEATURE, *CATEGORICAL_FEATURES,
    "signal_stream",
)
FEATURES = tuple(dict.fromkeys(BASE_FEATURES))
TEXT_FEATURE = "signal_stream"
RESULTS_PATH = PROJECT_ROOT / "reports/regression_round2/results.json"
SOURCE_AUDIT_PATH = PROJECT_ROOT / "reports/audits/source_definitions.json"
JSON_OUT = REPORT / "shift_diagnostics.json"
MD_OUT = REPORT / "SHIFT_DIAGNOSTICS.md"

PROTOCOL = {
    "purpose": "Feature-only supplied Test shift and Train support diagnostics; not model selection",
    "train_test_columns": list(FEATURES),
    "target_access": {
        "train_target_loaded": False,
        "test_target_loaded": False,
        "test_target_predictions": False,
        "test_target_based_selection": False,
    },
    "scope_zero_parser": "Use frozen Round1 numeric/mixed-count parsers; numeric zero remains distinct from missing and parse failure",
    "feature_only_relationships": {
        "correlations": "pairwise Pearson and Spearman among impact_load, entry_friction, scope_pressure, and parsed scope counts",
        "impact_bins": "ten rank-based impact_load bins; feature-only summaries of scope fields and entry_friction support",
    },
    "round2_control_comparison": "Read saved results.json only; pair R1_D with A_NO_SCOPE_SPREAD by matching validation position digest; no predictions or refits",
    "test_batch": "Report supplied rows by cycle only; do not imply real-world source-cycle completeness",
    "source_authority": "Local course-material audit plus source PDF pages 16-17; no target-generation formula or zero semantics inferred",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float | int | np.generic) -> float | int | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _count_parse(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    missing: dict[str, np.ndarray] = {}
    parse_fail: dict[str, np.ndarray] = {}
    for name in COUNT_FIELDS:
        parsed, _, _, failed, absent = _mixed_count_arrays(frame[name])
        values[name] = parsed
        missing[name] = absent.astype(bool)
        parse_fail[name] = failed.astype(bool)
    return values, missing, parse_fail


def _numeric_parse(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    missing: dict[str, np.ndarray] = {}
    invalid: dict[str, np.ndarray] = {}
    for name in NUMERIC_FIELDS:
        parsed, absent, bad = _number_array(frame[name])
        values[name] = parsed
        missing[name] = absent.astype(bool)
        invalid[name] = bad.astype(bool)
    return values, missing, invalid


def _array_summary(values: np.ndarray, missing: np.ndarray, invalid: np.ndarray | None = None) -> dict:
    finite = values[np.isfinite(values)]
    out = {
        "n_valid": int(len(finite)),
        "n_missing": int(np.sum(missing)),
        "missing_rate": float(np.mean(missing)) if len(missing) else None,
        "min": _finite(np.min(finite)) if len(finite) else None,
        "q01": _finite(np.quantile(finite, 0.01)) if len(finite) else None,
        "median": _finite(np.median(finite)) if len(finite) else None,
        "q99": _finite(np.quantile(finite, 0.99)) if len(finite) else None,
        "max": _finite(np.max(finite)) if len(finite) else None,
        "n_zero": int(np.sum(finite == 0)),
        "n_negative": int(np.sum(finite < 0)),
    }
    if invalid is not None:
        out["n_parse_invalid"] = int(np.sum(invalid))
    return out


def _feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    numeric, numeric_missing, numeric_invalid = _numeric_parse(frame)
    counts, count_missing, count_invalid = _count_parse(frame)
    telemetry_value, telemetry_unknown, telemetry_missing = _telemetry_arrays(frame[TELEMETRY_FEATURE])
    values: dict[str, np.ndarray] = {**numeric, **counts, TELEMETRY_FEATURE: telemetry_value}
    values["signal_stream_missing"] = np.array([
        _is_missing(value) or (isinstance(value, str) and not value.strip())
        for value in frame[TEXT_FEATURE].to_numpy(dtype=object)
    ], dtype=float)
    parsed = pd.DataFrame(values)
    parsed["signal_stream_missing"] = parsed["signal_stream_missing"].astype(float)
    field_audit = {}
    for name in NUMERIC_FIELDS:
        field_audit[name] = _array_summary(numeric[name], numeric_missing[name], numeric_invalid[name])
    for name in COUNT_FIELDS:
        field_audit[name] = _array_summary(counts[name], count_missing[name], count_invalid[name])
    field_audit[TELEMETRY_FEATURE] = {
        **_array_summary(telemetry_value, telemetry_missing, telemetry_unknown),
        "n_unknown_spelling": int(np.sum(telemetry_unknown)),
    }
    field_audit[TEXT_FEATURE] = {
        "n_missing_or_blank": int(np.sum(values["signal_stream_missing"])),
        "missing_or_blank_rate": float(np.mean(values["signal_stream_missing"])),
    }
    category_audit = {}
    for name in CATEGORICAL_FEATURES:
        normalized = _category_array(frame[name])
        present = np.array([not _is_missing(value) for value in normalized], dtype=bool)
        category_audit[name] = {
            "n_missing": int((~present).sum()),
            "missing_rate": float(np.mean(~present)),
            "n_nonmissing": int(present.sum()),
            # The vocabulary comparison is filled by the split-comparison step.
            "n_unknown_vs_train": None,
            "unknown_rate_of_nonmissing": None,
        }
    field_audit["categorical"] = category_audit
    field_audit[TELEMETRY_FEATURE]["unknown_rate"] = float(np.mean(telemetry_unknown))
    return parsed, {"fields": field_audit, "count_missing": count_missing,
                    "count_invalid": count_invalid, "numeric_missing": numeric_missing,
                    "numeric_invalid": numeric_invalid}


def _correlation(left: np.ndarray, right: np.ndarray) -> dict:
    mask = np.isfinite(left) & np.isfinite(right)
    n = int(mask.sum())
    if n < 2 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return {"n": n, "pearson": None, "spearman": None}
    return {
        "n": n,
        "pearson": _finite(np.corrcoef(left[mask], right[mask])[0, 1]),
        "spearman": _finite(spearmanr(left[mask], right[mask]).statistic),
    }


def _train_relationships(frame: pd.DataFrame, parsed: pd.DataFrame, audits: dict) -> dict:
    scope_names = ("scope_pressure", *COUNT_FIELDS)
    feature_names = ("impact_load", "entry_friction", *scope_names)
    matrix = {}
    for left_i, left in enumerate(feature_names):
        for right in feature_names[left_i + 1:]:
            matrix[f"{left}__{right}"] = _correlation(
                parsed[left].to_numpy(float), parsed[right].to_numpy(float)
            )

    ef_groups = {}
    entry = parsed["entry_friction"].to_numpy(float)
    for ef in (0, 1):
        rows = entry == ef
        stats = {"n": int(rows.sum())}
        for name in scope_names:
            values = parsed[name].to_numpy(float)[rows]
            finite = values[np.isfinite(values)]
            missing = audits["count_missing"].get(name, np.zeros(len(rows), dtype=bool))[rows] if name in COUNT_FIELDS else audits["numeric_missing"][name][rows]
            stats[name] = {
                "n_valid": int(len(finite)), "n_missing": int(np.sum(missing)),
                "median": _finite(np.median(finite)) if len(finite) else None,
                "q25": _finite(np.quantile(finite, .25)) if len(finite) else None,
                "q75": _finite(np.quantile(finite, .75)) if len(finite) else None,
                "zero_rate_all_rows": float(np.mean(values == 0)) if len(values) else None,
            }
        ef_groups[str(ef)] = stats

    impact = parsed["impact_load"].to_numpy(float)
    finite_positions = np.flatnonzero(np.isfinite(impact))
    impact_rank = pd.Series(impact[finite_positions]).rank(method="average").to_numpy()
    decile = np.minimum(np.floor((impact_rank - 1.0) * 10.0 / len(finite_positions)).astype(int), 9)
    bins = []
    for bin_id in range(10):
        positions = finite_positions[decile == bin_id]
        if len(positions) == 0:
            continue
        block = parsed.iloc[positions]
        item = {
            "impact_bin": bin_id + 1,
            "impact_load_min": _finite(block["impact_load"].min()),
            "impact_load_max": _finite(block["impact_load"].max()),
            "n": int(len(positions)),
            "entry_friction_0_n": int(np.sum(block["entry_friction"].to_numpy() == 0)),
            "entry_friction_1_n": int(np.sum(block["entry_friction"].to_numpy() == 1)),
            "scope": {},
        }
        for name in scope_names:
            values = block[name].to_numpy(float)
            finite = values[np.isfinite(values)]
            missing_mask = audits["count_missing"].get(name, np.zeros(len(frame), dtype=bool))[positions] if name in COUNT_FIELDS else audits["numeric_missing"][name][positions]
            item["scope"][name] = {
                "n_valid": int(len(finite)), "n_missing": int(np.sum(missing_mask)),
                "median": _finite(np.median(finite)) if len(finite) else None,
                "zero_n": int(np.sum(values == 0)),
            }
        bins.append(item)
    return {
        "description": "Train-only feature relationships; target was not loaded",
        "pairwise_correlations": matrix,
        "scope_by_entry_friction": ef_groups,
        "scope_support_by_impact_decile": bins,
    }


def _category_shift(train: pd.DataFrame, test: pd.DataFrame, test_audit: dict) -> None:
    for name in CATEGORICAL_FEATURES:
        train_values = _category_array(train[name])
        test_values = _category_array(test[name])
        train_seen = {str(value) for value in train_values if not _is_missing(value)}
        present = np.array([not _is_missing(value) for value in test_values], dtype=bool)
        unknown = np.array([
            bool(is_present and str(value) not in train_seen)
            for value, is_present in zip(test_values, present)
        ], dtype=bool)
        audit = test_audit["fields"]["categorical"][name]
        audit["n_unknown_vs_train"] = int(unknown.sum())
        audit["unknown_rate_of_nonmissing"] = float(unknown.sum() / present.sum()) if present.sum() else None


def _zero_support(train_parsed: pd.DataFrame, test_parsed: pd.DataFrame,
                  train_audit: dict, test_audit: dict) -> dict:
    train_count_values = [train_parsed[name].to_numpy(float) for name in COUNT_FIELDS]
    test_count_values = [test_parsed[name].to_numpy(float) for name in COUNT_FIELDS]
    train_any_zero = np.column_stack([values == 0 for values in train_count_values]).any(axis=1)
    test_any_zero = np.column_stack([values == 0 for values in test_count_values]).any(axis=1)
    train_all_zero = np.column_stack([values == 0 for values in train_count_values]).all(axis=1)
    test_all_zero = np.column_stack([values == 0 for values in test_count_values]).all(axis=1)
    train_any_missing = np.column_stack([train_audit["count_missing"][name] for name in COUNT_FIELDS]).any(axis=1)
    test_any_missing = np.column_stack([test_audit["count_missing"][name] for name in COUNT_FIELDS]).any(axis=1)
    train_pressure = train_parsed["scope_pressure"].to_numpy(float)
    test_pressure = test_parsed["scope_pressure"].to_numpy(float)
    train_min = float(np.nanmin(train_pressure))
    train_max = float(np.nanmax(train_pressure))
    test_below = np.isfinite(test_pressure) & (test_pressure < train_min)
    test_above = np.isfinite(test_pressure) & (test_pressure > train_max)
    return {
        "count_fields": list(COUNT_FIELDS),
        "train": {
            "rows": int(len(train_parsed)),
            "any_count_zero_n": int(train_any_zero.sum()),
            "all_three_count_zero_n": int(train_all_zero.sum()),
            "any_count_missing_n": int(train_any_missing.sum()),
            "joint_any_zero_and_pressure_below_train_min_n": int((train_any_zero & (train_pressure < train_min)).sum()),
            "scope_pressure_train_min": train_min,
            "scope_pressure_train_max": train_max,
        },
        "test": {
            "rows": int(len(test_parsed)),
            "any_count_zero_n": int(test_any_zero.sum()),
            "any_count_zero_rate": float(test_any_zero.mean()),
            "all_three_count_zero_n": int(test_all_zero.sum()),
            "all_three_count_zero_rate": float(test_all_zero.mean()),
            "any_count_missing_n": int(test_any_missing.sum()),
            "joint_any_zero_and_pressure_below_train_min_n": int((test_any_zero & test_below).sum()),
            "pressure_below_train_min_n": int(test_below.sum()),
            "pressure_above_train_max_n": int(test_above.sum()),
            "scope_pressure_min": _finite(np.nanmin(test_pressure)),
            "scope_pressure_max": _finite(np.nanmax(test_pressure)),
            "cohort_cause": "UNKNOWN: genuine low scope, missing-coded values, and source/representation shift cannot be distinguished from supplied data",
            "target_behavior": "UNKNOWN: no labeled zero-count Train analogue exists",
        },
        "test_by_supplied_cycle": [],
    }


def _test_cycles(test_raw: pd.DataFrame, test_parsed: pd.DataFrame,
                 test_audit: dict, zero_summary: dict) -> None:
    counts = np.column_stack([test_parsed[name].to_numpy(float) == 0 for name in COUNT_FIELDS])
    any_zero = counts.any(axis=1)
    all_zero = counts.all(axis=1)
    missing = np.column_stack([test_audit["count_missing"][name] for name in COUNT_FIELDS]).any(axis=1)
    cycle = test_parsed["cycle_code"].to_numpy(float)
    pressure = test_parsed["scope_pressure"].to_numpy(float)
    train_min = zero_summary["train"]["scope_pressure_train_min"]
    for code in sorted(np.unique(cycle[np.isfinite(cycle)])):
        rows = cycle == code
        zero_summary["test_by_supplied_cycle"].append({
            "cycle_code": int(code),
            "provided_rows": int(rows.sum()),
            "any_count_zero_n": int(np.sum(any_zero[rows])),
            "all_three_count_zero_n": int(np.sum(all_zero[rows])),
            "any_count_missing_n": int(np.sum(missing[rows])),
            "scope_pressure_below_train_min_n": int(np.sum(np.isfinite(pressure[rows]) & (pressure[rows] < train_min))),
        })
    zero_summary["test_cycle_completeness"] = (
        "Counts describe rows in the provided Test file per cycle; completeness of the real-world disclosure/source universe is UNKNOWN."
    )


def _paired_r2_scope(results_path: Path) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if data.get("status") != "REGRESSION_ROUND2_DEVELOPMENT_REVIEW_NO_FINAL_MODEL" or data.get("partial"):
        raise RuntimeError("Saved Round 2 results are not a complete reviewed results artifact")
    by_key = {(job["model"], job["family"], job["window"]): job for job in data["jobs"]}
    windows = []
    for family in ("primary", "secondary"):
        family_windows = sorted({job["window"] for job in data["jobs"] if job["family"] == family})
        for window in family_windows:
            full = by_key[("R1_D", family, window)]
            ablated = by_key[("A_NO_SCOPE_SPREAD", family, window)]
            same_positions = full["validation_positions_sha256"] == ablated["validation_positions_sha256"]
            if not same_positions or full["raw"]["n"] != ablated["raw"]["n"]:
                raise RuntimeError(f"Round 2 scope/full comparison is not paired: {family}/{window}")
            windows.append({
                "family": family, "window": window, "n": int(full["raw"]["n"]),
                "same_validation_positions": True,
                "validation_positions_sha256": full["validation_positions_sha256"],
                "full_r1_d_mae": float(full["raw"]["mae"]),
                "no_scope_spread_mae": float(ablated["raw"]["mae"]),
                "paired_mae_delta_no_scope_minus_full": float(ablated["raw"]["mae"] - full["raw"]["mae"]),
            })
    aggregates = {}
    for family in ("primary", "secondary"):
        full = data["aggregate"][family]["R1_D"]
        ablated = data["aggregate"][family]["A_NO_SCOPE_SPREAD"]
        if full["raw"]["n"] != ablated["raw"]["n"]:
            raise RuntimeError(f"Round 2 aggregate rows differ for {family}")
        groups = {}
        for key in ("EF0", "EF1", "tail_low_le5", "tail_high_ge95"):
            a, b = full["subgroups"].get(key), ablated["subgroups"].get(key)
            if a and b and a["n"] == b["n"]:
                groups[key] = {
                    "n": int(a["n"]), "full_r1_d_mae": float(a["mae"]),
                    "no_scope_spread_mae": float(b["mae"]),
                    "paired_mae_delta_no_scope_minus_full": float(b["mae"] - a["mae"]),
                }
        per_full = {int(row["cycle"]): row for row in full["per_cycle"]}
        per_ab = {int(row["cycle"]): row for row in ablated["per_cycle"]}
        if set(per_full) != set(per_ab) or any(per_full[c]["n"] != per_ab[c]["n"] for c in per_full):
            raise RuntimeError(f"Round 2 per-cycle scope/full comparison is not aligned: {family}")
        cycle_deltas = [float(per_ab[c]["mae"] - per_full[c]["mae"]) for c in per_full]
        aggregates[family] = {
            "n": int(full["raw"]["n"]),
            "same_rows": True,
            "full_r1_d_mae": float(full["raw"]["mae"]),
            "no_scope_spread_mae": float(ablated["raw"]["mae"]),
            "paired_mae_delta_no_scope_minus_full": float(ablated["raw"]["mae"] - full["raw"]["mae"]),
            "full_equal_cycle_mae": float(full["equal_cycle_mae"]),
            "no_scope_spread_equal_cycle_mae": float(ablated["equal_cycle_mae"]),
            "equal_cycle_mae_delta_no_scope_minus_full": float(ablated["equal_cycle_mae"] - full["equal_cycle_mae"]),
            "subgroups": groups,
            "per_cycle_delta_summary": {
                "cycles": int(len(cycle_deltas)),
                "cycles_no_scope_lower_mae": int(sum(delta < 0 for delta in cycle_deltas)),
                "cycles_no_scope_higher_mae": int(sum(delta > 0 for delta in cycle_deltas)),
                "minimum_delta": min(cycle_deltas), "maximum_delta": max(cycle_deltas),
            },
        }
    return {
        "source": "saved reports/regression_round2/results.json; no fitting or prediction was run",
        "results_sha256": _sha256(results_path),
        "comparison": "R1_D versus A_NO_SCOPE_SPREAD; same saved validation positions verified by digest at every window",
        "interpretation_limit": "Predictive ablation under prior in-support temporal validation only; not causal importance and not an estimate of Test zero-cohort error.",
        "aggregates": aggregates,
        "windows": windows,
    }


def _source_record() -> dict:
    audit = json.loads(SOURCE_AUDIT_PATH.read_text(encoding="utf-8"))
    expected_hash = audit["sha256"]
    pdf_path = PROJECT_ROOT.parent / "TK1" / "TP 1 KASDAD Ganjil 2026_2027.pdf"
    actual_hash = _sha256(pdf_path) if pdf_path.exists() else None
    return {
        "audit_file": "reports/audits/source_definitions.json",
        "audit_scope": audit["scope"],
        "audited_pdf_name": audit["source_name"],
        "audited_pdf_sha256": expected_hash,
        "audited_pdf_pages": audit["pages"],
        "local_course_pdf_path": "../TK1/TP 1 KASDAD Ganjil 2026_2027.pdf",
        "local_course_pdf_sha256": actual_hash,
        "local_course_pdf_matches_audit": bool(actual_hash == expected_hash) if actual_hash else None,
        "pages_used": [16, 17],
        "provenance_note": "The accessible sibling TP1 PDF is 44 pages and its SHA-256 differs from the tracked 57-page audit source. Its pp. 16-17 are cited as a separately identified local course copy, not as the byte-identical audited PDF.",
        "documented_features": {
            "cycle_code": "Page 16: disclosure-cycle order; larger is newer, not an absolute calendar.",
            "scope_trace": "Page 16: affected-configuration count or coverage indicator; mixed numeric/text/estimated format.",
            "org_spread": "Page 17: organization/ecosystem spread indicator; formatting is inconsistent.",
            "module_spread": "Page 17: module/component spread indicator; formatting is inconsistent.",
            "scope_pressure": "Page 17: nonlinear scope/impact intensity indicator.",
        },
        "zero_semantics": "Not specified in the cited pages or source audit.",
        "threat_rank_generation_formula": "UNKNOWN per docs/COMPETITION_INFO.md and reports/audits/source_definitions.json.",
    }


def _percent(n: int, total: int) -> str:
    return f"{100.0 * n / total:.2f}%" if total else "n/a"


def _markdown(report: dict) -> str:
    shift = report["zero_support"]
    r2 = report["round2_scope_ablation"]["aggregates"]
    lines = [
        "# Regression shift diagnostics",
        "",
        f"Generated {report['generated_at_utc']}. Feature-only Test audit; no model fitting, Test target reads, Test predictions, or recipe selection.",
        "",
        "## Zero-scope support",
        "",
        f"Train has {shift['train']['any_count_zero_n']:,}/{shift['train']['rows']:,} rows with any parsed scope count equal to zero; Test has {shift['test']['any_count_zero_n']:,}/{shift['test']['rows']:,} ({100*shift['test']['any_count_zero_rate']:.2f}%).",
        f"All-three-zero: Train {shift['train']['all_three_count_zero_n']:,}; Test {shift['test']['all_three_count_zero_n']:,}. Test any-zero rows also below Train `scope_pressure` minimum: {shift['test']['joint_any_zero_and_pressure_below_train_min_n']:,}; pressure below that minimum overall: {shift['test']['pressure_below_train_min_n']:,}.",
        "",
        "A parsed zero is distinct from a missing or parse-failed value. The course material describes scope fields but does not define zero semantics. The 6,278-row Test cohort has no labeled zero-count Train analogue; its source cause and `threat_rank` behavior are UNKNOWN. Do not impute or invent target values from this diagnostic.",
        "",
        "Per-cycle counts below refer only to the provided Test file. They do not certify that the real-world disclosure/source universe is complete within each cycle.",
        "",
        "| Test cycle | Provided rows | Any count zero | All three zero | Any count missing | Pressure below Train min |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shift["test_by_supplied_cycle"]:
        lines.append(f"| {row['cycle_code']} | {row['provided_rows']:,} | {row['any_count_zero_n']:,} | {row['all_three_count_zero_n']:,} | {row['any_count_missing_n']:,} | {row['scope_pressure_below_train_min_n']:,} |")
    lines.extend([
        "",
        "## Saved R2 scope-ablation comparison",
        "",
        "This reads the existing paired R2 result artifact only. Positive delta means removing scope/spread raised MAE on the same saved validation rows; this is conditional predictive contribution, not a causal effect and not Test-cohort error.",
        "",
        "| View | n | Full R1-D MAE | No-scope MAE | Paired delta | Equal-cycle delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for fam, label in (("primary", "Primary"), ("secondary", "H12")):
        row = r2[fam]
        lines.append(f"| {label} | {row['n']:,} | {row['full_r1_d_mae']:.6f} | {row['no_scope_spread_mae']:.6f} | {row['paired_mae_delta_no_scope_minus_full']:+.6f} | {row['equal_cycle_mae_delta_no_scope_minus_full']:+.6f} |")
    lines.extend([
        "",
        "The primary and H12 views reuse validation records, and all windows were development evidence. The ablation does not establish how either feature set behaves on the novel Test support.",
        "",
        "## Train-only feature relationships",
        "",
        "Pairwise correlations, EF summaries, and impact-bin support are computed from predictor columns only. `threat_rank` was not loaded. These are descriptive associations, not mechanisms or outcome results.",
        "",
        "## Source and limitations",
        "",
        f"Course source copy: `{report['source']['local_course_pdf_path']}`, pp. 16–17; SHA-256 `{report['source']['local_course_pdf_sha256']}`. The tracked audit identifies a different 57-page PDF (`{report['source']['audited_pdf_name']}`, SHA-256 `{report['source']['audited_pdf_sha256']}`), so the local copy is not represented as byte-identical to that audited source. Both descriptions remain course context only; neither documents zero semantics or an authoritative TP2 target-generation formula.",
        "",
        "Recommended controlled Stage 4 option: keep the frozen full R1-D and no-scope/spread representations as paired controls if MAIN elects a next stage. No-scope is only a robustness comparator; the current report does not authorize a feature decision, a test correction, or a new model choice.",
        "",
        f"Frozen diagnostic identity: `{report['freeze_signature']}`. Train/Test source hashes are recorded in the JSON artifact.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    code = ["scripts/research_regression_shift.py"]
    freeze = freeze_stage("shift_diagnostics", PROTOCOL, code)

    paths = task_files("regression")
    train_path = Path(paths["train"])
    test_path = Path(paths["test"])
    train_headers = list(pd.read_csv(train_path, nrows=0).columns)
    test_headers = list(pd.read_csv(test_path, nrows=0).columns)
    if "threat_rank" not in train_headers:
        raise RuntimeError("Train schema lacks the expected regression target column")
    if "threat_rank" in test_headers:
        raise RuntimeError("Expected official Test schema to omit threat_rank; refusing ambiguous target boundary")
    missing_train_features = sorted(set(FEATURES) - set(train_headers))
    missing_test_features = sorted(set(FEATURES) - set(test_headers))
    if missing_train_features or missing_test_features:
        raise RuntimeError(f"Required predictor schema missing: train={missing_train_features}, test={missing_test_features}")

    # Explicit usecols ensures neither Train targets nor Test labels are loaded.
    train_raw = pd.read_csv(train_path, usecols=list(FEATURES))
    test_raw = pd.read_csv(test_path, usecols=list(FEATURES))
    train_parsed, train_audit = _feature_frame(train_raw)
    test_parsed, test_audit = _feature_frame(test_raw)
    _category_shift(train_raw, test_raw, test_audit)

    zero = _zero_support(train_parsed, test_parsed, train_audit, test_audit)
    _test_cycles(test_raw, test_parsed, test_audit, zero)
    relationships = _train_relationships(train_raw, train_parsed, train_audit)
    source = _source_record()
    report = {
        "stage": "shift_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FEATURE_ONLY_DIAGNOSTIC_NO_MODEL_FIT_NO_TEST_LABEL",
        "freeze_signature": freeze["signature"],
        "protocol": PROTOCOL,
        "inputs": {
            "train_path": "official regression Train.csv",
            "train_sha256": _sha256(train_path),
            "train_rows": int(len(train_raw)),
            "test_path": "official regression Test.csv",
            "test_sha256": _sha256(test_path),
            "test_rows": int(len(test_raw)),
            "loaded_columns": list(FEATURES),
            "train_target_column_loaded": False,
            "test_target_column_loaded": False,
            "test_predictions_created": False,
        },
        "feature_summary": {"train": train_audit["fields"], "test": test_audit["fields"]},
        "zero_support": zero,
        "train_feature_relationships": relationships,
        "round2_scope_ablation": _paired_r2_scope(RESULTS_PATH),
        "source": source,
        "unknowns": [
            "Whether zero counts mean genuine zero scope or a missing-coded/source convention is unknown.",
            "There are no zero-count Train labels to validate Test zero-cohort target behavior.",
            "Provided Test rows per cycle do not prove completeness of the real-world source universe.",
            "Threat-rank generation formula and timestamp-certified availability of added features are unknown.",
        ],
    }

    REPORT.mkdir(parents=True, exist_ok=True)
    if JSON_OUT.exists() or MD_OUT.exists():
        raise FileExistsError("Shift diagnostic outputs already exist; refusing to overwrite")
    JSON_OUT.write_text(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    MD_OUT.write_text(_markdown(report), encoding="utf-8")
    verify_stage("shift_diagnostics")
    print(f"wrote {JSON_OUT.relative_to(PROJECT_ROOT)}")
    print(f"wrote {MD_OUT.relative_to(PROJECT_ROOT)}")
    print("Test target labels loaded: false; model fits: 0; freeze verified: true")


if __name__ == "__main__":
    main()
