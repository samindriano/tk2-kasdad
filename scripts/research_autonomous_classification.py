"""Frozen availability-specialist ablation for Classification.

Run ``--freeze`` before ``--run``. The run evaluates exactly three routes and
fits only two new branch experts at each of the six existing temporal cutoffs.
It reads official Classification Train only; this script never opens Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.models.classification_metrics import binary_metrics
from src.models.classification_models import build_classifier, _array_digest
from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files


PROTOCOL_DIR = ROOT / "experiments" / "autonomous_eda" / "classification"
PROTOCOL_PATH = PROTOCOL_DIR / "protocol.json"
FREEZE_PATH = PROTOCOL_DIR / "freeze.json"
REPORT_DIR = ROOT / "reports" / "autonomous_eda" / "classification"
RUN_DIR = ROOT / "experiments" / "runs" / "autonomous_cls_v1"
ROUND1_CACHE = ROOT / "experiments" / "runs" / "classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports" / "classification_round1" / "fit_audits"
ROUND2_PROTOCOL = ROOT / "experiments" / "classification_round2" / "protocol.json"
ROUND2_RESULTS = ROOT / "reports" / "classification_round2" / "results.json"
CODE_FILES = [
    "src/models/autonomous_classification.py",
    "scripts/research_autonomous_classification.py",
    "scripts/verify_autonomous_classification.py",
    "tests/test_autonomous_classification.py",
]
CANDIDATES = ["present_specialist", "missing_specialist", "both_specialist"]
MODEL_NAMES = ["C4_HGB", *CANDIDATES]
FIT_ENDS = [55, 59, 63, 67, 71, 75]
VOLUME_LABELS = ["Q1_low", "Q2", "Q3", "Q4", "Q5_high"]
PERIODS = [("cycle_47_57", 47, 57), ("cycle_58_68", 58, 68), ("cycle_69_79", 69, 79)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def train_path() -> Path:
    path = task_files("classification")["train"]
    if path is None:
        raise FileNotFoundError("Official Classification Train.csv was not found")
    return path.resolve()


def load_train() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Path]:
    path = train_path()
    frame = pd.read_csv(path)
    required = {"record_id", "cycle_code", "signal_stream", "text_load", "artifact_signal"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Train schema is missing required columns: {sorted(required - set(frame.columns))}")
    if frame["record_id"].isna().any() or not frame["record_id"].is_unique:
        raise ValueError("Train record_id must be nonmissing and unique")
    y = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("artifact_signal must contain only 0/1")
    X = frame.drop(columns=["artifact_signal"])
    return frame, y, X, path


def protocol_windows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    return protocol["validation"]["primary_windows"], protocol["validation"]["secondary_windows"]


def horizon_for_fit(fit_end: int, primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> int:
    windows = [w for w in [*primary, *secondary] if int(w["fit"][1]) == fit_end]
    if not windows:
        raise ValueError(f"No frozen evaluation window uses fit cutoff {fit_end}")
    return max(int(w["validation"][1]) for w in windows)


def expected_rows(frame: pd.DataFrame, fit_end: int, horizon: int) -> np.ndarray:
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy()
    return np.flatnonzero((cycle > fit_end) & (cycle <= horizon))


def load_control(fit_end: int, model_id: str, frame: pd.DataFrame,
                 primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = ROUND1_CACHE / f"fit{fit_end}__{model_id}.npz"
    audit_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    if not cache.exists() or not audit_path.exists():
        raise FileNotFoundError(f"Missing historical control cache/audit: {cache} / {audit_path}")
    audit = read_json(audit_path).get(model_id)
    if not isinstance(audit, dict):
        raise ValueError(f"Missing {model_id} audit for fit{fit_end}")
    with np.load(cache, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        probability = archive["probability"].astype(float, copy=True)
    expected = expected_rows(frame, fit_end, horizon_for_fit(fit_end, primary, secondary))
    if not np.array_equal(rows, expected):
        raise ValueError(f"Historical {model_id} row identity differs at fit{fit_end}")
    if probability.shape != rows.shape or not np.isfinite(probability).all():
        raise ValueError(f"Historical {model_id} probabilities are invalid at fit{fit_end}")
    expected_hash = audit.get("probability_sha256")
    if not expected_hash or _array_digest(probability) != expected_hash:
        raise ValueError(f"Historical {model_id} vector does not match its frozen audit at fit{fit_end}")
    expected_fit_rows = int((pd.to_numeric(frame["cycle_code"]) <= fit_end).sum())
    if int(audit.get("fit_rows", -1)) != expected_fit_rows:
        raise ValueError(f"Historical {model_id} fit-row count differs at fit{fit_end}")
    if audit.get("model") != model_id:
        raise ValueError(f"Historical control identity is invalid at fit{fit_end}")
    return rows, probability, {
        "fit_rows": expected_fit_rows,
        "cache_sha256": sha256_file(cache),
        "audit_sha256": sha256_file(audit_path),
        "probability_sha256": expected_hash,
        "outer_rows": int(len(rows)),
    }


def control_vectors(frame: pd.DataFrame, primary: list[dict[str, Any]],
                    secondary: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for fit_end in FIT_ENDS:
        rows_a, c1, id1 = load_control(fit_end, "C1_B", frame, primary, secondary)
        rows_b, c2, id2 = load_control(fit_end, "C2_B", frame, primary, secondary)
        if not np.array_equal(rows_a, rows_b):
            raise ValueError(f"C1_B/C2_B control rows differ at fit{fit_end}")
        missing = stream_missing_mask(frame.iloc[rows_a])
        c4 = np.where(missing, c1, c2)
        out[fit_end] = {"rows": rows_a, "C1_B": c1, "C2_B": c2, "C4_HGB": c4}
        provenance[str(fit_end)] = {"C1_B": id1, "C2_B": id2}
    return out, provenance


def raw_eda_summaries(frame: pd.DataFrame, y: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    missing = stream_missing_mask(frame)
    volume = pd.to_numeric(frame["text_load"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(volume).all():
        raise ValueError("text_load must be finite for the predeclared availability-volume description")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    q = np.quantile(volume, [0.2, 0.4, 0.6, 0.8])
    volume_bin = np.searchsorted(q, volume, side="right")
    raw_rows: list[dict[str, Any]] = []
    volume_rows: list[dict[str, Any]] = []
    for period_id, lower, upper in PERIODS:
        in_period = (cycle >= lower) & (cycle <= upper)
        for state, state_mask in (("present", ~missing), ("missing", missing)):
            mask = in_period & state_mask
            vals = volume[mask]
            raw_rows.append({
                "period": period_id, "cycle_min": lower, "cycle_max": upper,
                "availability": state, "n": int(mask.sum()),
                "positive": int(y[mask].sum()),
                "positive_rate": float(y[mask].mean()) if mask.any() else None,
                "text_load_q25": float(np.quantile(vals, 0.25)) if len(vals) else None,
                "text_load_median": float(np.median(vals)) if len(vals) else None,
                "text_load_q75": float(np.quantile(vals, 0.75)) if len(vals) else None,
            })
            for bin_id, label in enumerate(VOLUME_LABELS):
                cell = mask & (volume_bin == bin_id)
                volume_rows.append({
                    "period": period_id, "availability": state, "volume_bin": label,
                    "bin_lower": float(np.min(volume[volume_bin == bin_id])) if np.any(volume_bin == bin_id) else None,
                    "bin_upper": float(np.max(volume[volume_bin == bin_id])) if np.any(volume_bin == bin_id) else None,
                    "n": int(cell.sum()), "positive": int(y[cell].sum()),
                    "positive_rate": float(y[cell].mean()) if cell.any() else None,
                })
    return raw_rows, volume_rows, q


def fit_support_rows(frame: pd.DataFrame, y: np.ndarray) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    missing = stream_missing_mask(frame)
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    volume = pd.to_numeric(frame["text_load"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    edges_by_fit: dict[int, np.ndarray] = {}
    for fit_end in FIT_ENDS:
        fit = cycle <= fit_end
        quantiles = np.quantile(volume[fit], [0.2, 0.4, 0.6, 0.8])
        edges_by_fit[fit_end] = quantiles.astype(float)
        for state, state_mask in (("present", ~missing), ("missing", missing)):
            mask = fit & state_mask
            counts = np.bincount(y[mask], minlength=2)
            selected = volume[mask]
            rows.append({
                "fit_end_cycle": fit_end, "availability": state,
                "fit_rows": int(mask.sum()), "negative_rows": int(counts[0]),
                "positive_rows": int(counts[1]), "positive_rate": float(y[mask].mean()),
                "cycle_min": int(cycle[mask].min()), "cycle_max": int(cycle[mask].max()),
                "text_load_q25": float(np.quantile(selected, 0.25)),
                "text_load_median": float(np.median(selected)),
                "text_load_q75": float(np.quantile(selected, 0.75)),
                "quantile_edges_for_validation": [float(v) for v in quantiles],
            })
    return rows, edges_by_fit


def make_availability_figure(raw_rows: list[dict[str, Any]], path: Path) -> None:
    table = pd.DataFrame(raw_rows)
    periods = [entry[0] for entry in PERIODS]
    labels = ["47–57", "58–68", "69–79"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    colors = {"present": "#2474A6", "missing": "#D16B35"}
    for state in ("present", "missing"):
        selected = table[table["availability"] == state].set_index("period").loc[periods]
        x = np.arange(len(periods)) + (-0.08 if state == "present" else 0.08)
        axes[0].plot(x, selected["positive_rate"], marker="o", linewidth=2.0,
                     color=colors[state], label=f"Stream {state}")
        for xval, (_, row) in zip(x, selected.iterrows()):
            axes[0].annotate(f"n={int(row['n']):,}", (xval, row["positive_rate"]),
                             xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        median = selected["text_load_median"].to_numpy()
        lower = median - selected["text_load_q25"].to_numpy()
        upper = selected["text_load_q75"].to_numpy() - median
        axes[1].errorbar(np.arange(len(periods)) + (-0.08 if state == "present" else 0.08),
                         median, yerr=np.vstack([lower, upper]), fmt="o-", capsize=4,
                         color=colors[state], linewidth=2.0, label=f"Stream {state}")
    axes[0].set_xticks(np.arange(len(periods)), labels)
    axes[0].set_ylim(0, max(0.65, float(table["positive_rate"].max()) + 0.08))
    axes[0].set_ylabel("artifact_signal = 1 proportion")
    axes[0].set_title("Prevalence shifts across periods")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].set_xticks(np.arange(len(periods)), labels)
    axes[1].set_ylabel("text_load median (IQR)")
    axes[1].set_title("Text volume differs by availability")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Availability populations: label prevalence and text_load", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def expected_code_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in CODE_FILES}


def check_protocol_matches_round2(protocol: dict[str, Any]) -> None:
    source = read_json(ROUND2_PROTOCOL)
    primary = protocol["validation"]["primary_windows"]
    secondary = protocol["validation"]["secondary_windows"]
    if primary != source.get("primary_windows") or secondary != source.get("secondary_windows"):
        raise ValueError("Autonomous classification windows differ from frozen Round 2 windows")
    if source.get("primary_control_policy") != "fixed_0.5":
        raise ValueError("Frozen Round 2 primary policy is not fixed 0.50")


def _control_reference_checks(frame: pd.DataFrame, y: np.ndarray,
                              controls: dict[int, dict[str, Any]],
                              primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> dict[str, Any]:
    old_results = read_json(ROUND2_RESULTS)["aggregate"]
    checks: dict[str, Any] = {}
    for family, windows in (("primary", primary), ("secondary", secondary)):
        rows_y: list[np.ndarray] = []
        rows_p: list[np.ndarray] = []
        per_window = {}
        per_cycle = []
        for window in windows:
            fit_end = int(window["fit"][1])
            lower, upper = map(int, window["validation"])
            fold = controls[fit_end]
            selected = (frame.iloc[fold["rows"]]["cycle_code"].to_numpy() >= lower) & \
                       (frame.iloc[fold["rows"]]["cycle_code"].to_numpy() <= upper)
            idx = fold["rows"][selected]
            p = fold["C4_HGB"][selected]
            yt = y[idx]
            report = binary_metrics(yt, p, 0.50)
            per_window[window["id"]] = report
            rows_y.append(yt)
            rows_p.append(p)
            for cycle_value in sorted(np.unique(frame.iloc[idx]["cycle_code"].to_numpy())):
                cycle_mask = frame.iloc[idx]["cycle_code"].to_numpy() == cycle_value
                per_cycle.append(binary_metrics(yt[cycle_mask], p[cycle_mask], 0.50)["macro_f1"])
        y_all = np.concatenate(rows_y)
        p_all = np.concatenate(rows_p)
        pooled = binary_metrics(y_all, p_all, 0.50)
        equal_cycle = float(np.mean(per_cycle))
        target = old_results[family]["fixed_0.5"]["C4_HGB"]
        target_metrics = target["metrics"]
        if not np.isclose(pooled["macro_f1"], target_metrics["macro_f1"], atol=1e-12, rtol=0):
            raise ValueError(f"Recombined historical C4 {family} pooled score differs from Round 2 report")
        if not np.isclose(equal_cycle, target["equal_cycle_macro_f1"], atol=1e-12, rtol=0):
            raise ValueError(f"Recombined historical C4 {family} equal-cycle score differs from Round 2 report")
        checks[family] = {
            "n": int(len(y_all)), "macro_f1": pooled["macro_f1"],
            "equal_cycle_macro_f1": equal_cycle,
            "report_macro_f1": target_metrics["macro_f1"],
            "report_equal_cycle_macro_f1": target["equal_cycle_macro_f1"],
            "windows": {key: value["macro_f1"] for key, value in per_window.items()},
            "exact_reference_match_within_1e-12": True,
        }
    return checks


def freeze() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Frozen protocol already exists: {FREEZE_PATH}")
    if RUN_DIR.exists() and any(RUN_DIR.rglob("*.npz")):
        raise RuntimeError("Research prediction cache already exists; cannot freeze after fits")
    protocol = read_json(PROTOCOL_PATH)
    check_protocol_matches_round2(protocol)
    frame, y, _X, source = load_train()
    source_hash = sha256_file(source)
    if source_hash != "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c":
        raise ValueError(f"Classification Train SHA-256 differs from official frozen source: {source_hash}")
    if len(frame) != 50608 or set(frame["cycle_code"].astype(int).unique()) != set(range(47, 80)):
        raise ValueError("Official Classification Train row count/cycle coverage differs from the frozen audit")
    primary = protocol["validation"]["primary_windows"]
    secondary = protocol["validation"]["secondary_windows"]
    controls, control_identity = control_vectors(frame, primary, secondary)
    raw_rows, volume_rows, full_quantile_edges = raw_eda_summaries(frame, y)
    support_rows, edges_by_fit = fit_support_rows(frame, y)
    summary_path = REPORT_DIR / "raw_eda_by_period.csv"
    volume_path = REPORT_DIR / "raw_eda_by_period_volume.csv"
    support_path = REPORT_DIR / "fit_support.csv"
    figure_path = REPORT_DIR / "figures" / "availability_period_volume.png"
    write_csv(summary_path, raw_rows)
    write_csv(volume_path, volume_rows)
    write_csv(support_path, support_rows)
    make_availability_figure(raw_rows, figure_path)
    reference = _control_reference_checks(frame, y, controls, primary, secondary)
    support_counts = {
        str(end): {
            state: next(row for row in support_rows if row["fit_end_cycle"] == end and row["availability"] == state)
            for state in ("present", "missing")
        } for end in FIT_ENDS
    }
    for end in FIT_ENDS:
        for state in ("present", "missing"):
            row = support_counts[str(end)][state]
            if row["fit_rows"] < protocol["support_rule"]["minimum_rows"] or \
               min(row["negative_rows"], row["positive_rows"]) < protocol["support_rule"]["minimum_rows_per_class"]:
                raise ValueError(f"Predeclared fit support rule fails for fit{end} {state}: {row}")
    try:
        import sklearn
        import scipy
        import threadpoolctl as threadpoolctl_module
        import matplotlib as matplotlib_module
        packages = {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                    "scipy": scipy.__version__, "threadpoolctl": threadpoolctl_module.__version__,
                    "matplotlib": matplotlib_module.__version__}
    except Exception:
        packages = {}
    try:
        import subprocess
        source_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        source_head = "UNKNOWN"
    artifact_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in
                       [summary_path, volume_path, support_path, figure_path]}
    identity = {
        "status": "FROZEN_BEFORE_FIRST_NEW_FIT",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_head": source_head,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_code_sha256": expected_code_hashes(),
        "train": {"sha256": source_hash, "rows": int(len(frame)), "columns": list(frame.columns),
                  "cycle_min": int(frame.cycle_code.min()), "cycle_max": int(frame.cycle_code.max()),
                  "record_id_sha256": hashlib.sha256("\n".join(frame.record_id.astype(str)).encode("utf-8")).hexdigest()},
        "round2_protocol_sha256": sha256_file(ROUND2_PROTOCOL),
        "round1_controls": control_identity,
        "recombined_c4_reference": reference,
        "raw_eda": {
            "question": protocol["raw_eda_question"],
            "periods": PERIODS,
            "global_text_load_quintile_edges_target_and_availability_blind": [float(v) for v in full_quantile_edges],
            "period_summary": raw_rows,
            "fit_subset_summary": support_rows,
            "volume_table_rows": int(len(volume_rows)),
            "artifact_sha256": artifact_hashes,
            "state_separate_from_model_error_and_new_fit_results": True,
        },
        "new_fit_plan": {"cutoffs": FIT_ENDS, "new_branch_fits": 12,
                         "threshold": 0.50, "no_official_test_access": True},
        "packages": packages,
    }
    identity["freeze_signature"] = json_sha256(identity)
    write_json(FREEZE_PATH, identity)
    print(json.dumps({"status": identity["status"], "freeze_signature": identity["freeze_signature"],
                      "train_sha256": source_hash, "new_fit_budget": 12,
                      "c4_reference": reference,
                      "support": {str(end): {
                          state: {"n": support_counts[str(end)][state]["fit_rows"],
                                  "classes": [support_counts[str(end)][state]["negative_rows"],
                                              support_counts[str(end)][state]["positive_rows"]]}
                          for state in ("present", "missing")
                      } for end in FIT_ENDS}}, indent=2, ensure_ascii=False))


def verify_freeze(frame: pd.DataFrame, train_source: Path) -> dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise FileNotFoundError("Run --freeze before the first new model fit")
    freeze_data = read_json(FREEZE_PATH)
    signature = freeze_data.get("freeze_signature")
    unsigned = {key: value for key, value in freeze_data.items() if key != "freeze_signature"}
    if signature != json_sha256(unsigned):
        raise ValueError("Freeze signature does not reconcile")
    if freeze_data["protocol_sha256"] != sha256_file(PROTOCOL_PATH):
        raise ValueError("Frozen experiment protocol changed after freeze")
    if freeze_data["source_code_sha256"] != expected_code_hashes():
        raise ValueError("Research, verification, test, or model code changed after freeze")
    if freeze_data["train"]["sha256"] != sha256_file(train_source):
        raise ValueError("Official Train input changed after freeze")
    if freeze_data["round2_protocol_sha256"] != sha256_file(ROUND2_PROTOCOL):
        raise ValueError("Historical Round 2 validation protocol changed")
    if len(frame) != freeze_data["train"]["rows"]:
        raise ValueError("Train row count changed after freeze")
    return freeze_data


def fit_specialist(model_id: str, X_fit: pd.DataFrame, y_fit: np.ndarray,
                   selected: np.ndarray, branch_name: str, fit_end: int) -> tuple[Any, dict[str, Any]]:
    selected_positions = np.flatnonzero(selected)
    if not selected_positions.size:
        raise ValueError(f"No {branch_name} training rows available at fit{fit_end}")
    branch_y = y_fit[selected]
    counts = np.bincount(branch_y, minlength=2)
    protocol = read_json(PROTOCOL_PATH)
    min_rows = int(protocol["support_rule"]["minimum_rows"])
    min_class = int(protocol["support_rule"]["minimum_rows_per_class"])
    if len(branch_y) < min_rows or int(counts.min()) < min_class:
        raise ValueError(f"Support rule failed in new fit {fit_end}/{branch_name}: n={len(branch_y)}, {counts.tolist()}")
    model = build_classifier({"id": model_id})
    fit_input = X_fit.iloc[selected_positions]
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(fit_input, branch_y)
    elapsed = time.perf_counter() - start
    state = model.audit_state()
    return model, {
        "model_id": model_id, "branch": branch_name, "fit_mode": "present_only" if branch_name == "present" else "missing_only",
        "fit_rows": int(len(branch_y)), "class_counts": counts.astype(int).tolist(),
        "fit_cycle_range": [int(X_fit.iloc[selected_positions]["cycle_code"].min()),
                            int(X_fit.iloc[selected_positions]["cycle_code"].max())],
        "fit_record_ids_sha256": hashlib.sha256("\n".join(X_fit.iloc[selected_positions]["record_id"].astype(str)).encode("utf-8")).hexdigest(),
        "selected_fit_positions_sha256": hashlib.sha256(np.asarray(selected_positions, dtype="<i8").tobytes()).hexdigest(),
        "fit_data_sha256": state["fit_data_sha256"],
        "model_state_sha256": state["model_state_sha256"],
        "expert_state": state,
        "elapsed_seconds": elapsed,
    }


def _assign_volume_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="right")


def evaluate_predictions(frame: pd.DataFrame, y: np.ndarray,
                         fold_data: dict[int, dict[str, Any]],
                         edges_by_fit: dict[int, np.ndarray]) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    primary = protocol["validation"]["primary_windows"]
    secondary = protocol["validation"]["secondary_windows"]
    cycle_all = frame["cycle_code"].to_numpy(dtype=int)
    volume_all = frame["text_load"].to_numpy(dtype=float)
    missing_all = stream_missing_mask(frame)
    metrics: dict[str, dict[str, Any]] = {}
    windows_report: list[dict[str, Any]] = []
    cycles_report: list[dict[str, Any]] = []
    subgroups_report: list[dict[str, Any]] = []

    for family, windows in (("primary", primary), ("h12", secondary)):
        metrics[family] = {}
        family_vectors: dict[str, dict[str, list[np.ndarray]]] = {
            name: {"y": [], "p": [], "rows": [], "volume_bin": []} for name in MODEL_NAMES
        }
        for window in windows:
            fit_end = int(window["fit"][1])
            lo, hi = map(int, window["validation"])
            fold = fold_data[fit_end]
            rows = fold["rows"]
            selected = (cycle_all[rows] >= lo) & (cycle_all[rows] <= hi)
            eval_rows = rows[selected]
            if len(eval_rows) == 0:
                raise ValueError(f"Empty validation window {window['id']}")
            yt = y[eval_rows]
            volume_bin = _assign_volume_bins(volume_all[eval_rows], edges_by_fit[fit_end])
            for name in MODEL_NAMES:
                probability = fold[name][selected]
                report = binary_metrics(yt, probability, 0.50)
                windows_report.append({
                    "family": family, "window": window["id"], "fit_end_cycle": fit_end,
                    "validation_cycle_min": lo, "validation_cycle_max": hi,
                    "model": name, "n": int(len(eval_rows)), "macro_f1": report["macro_f1"],
                    "positive_rate": report["positive_rate"],
                    "positive_prediction_rate": report["positive_prediction_rate"],
                    "false_positive": report["false_positive"], "false_negative": report["false_negative"],
                    "per_class": report["per_class"],
                    "delta_vs_C4": None if name == "C4_HGB" else None,
                })
                bucket = family_vectors[name]
                bucket["y"].append(yt)
                bucket["p"].append(probability)
                bucket["rows"].append(eval_rows)
                bucket["volume_bin"].append(volume_bin)
                for cycle_value in range(lo, hi + 1):
                    in_cycle = cycle_all[eval_rows] == cycle_value
                    if not in_cycle.any():
                        continue
                    cy = binary_metrics(yt[in_cycle], probability[in_cycle], 0.50)
                    cycles_report.append({"family": family, "window": window["id"], "cycle_code": cycle_value,
                                          "model": name, "n": cy["n"], "macro_f1": cy["macro_f1"],
                                          "positive_rate": cy["positive_rate"],
                                          "false_positive": cy["false_positive"],
                                          "false_negative": cy["false_negative"]})

        for name, values in family_vectors.items():
            pooled_y = np.concatenate(values["y"])
            pooled_p = np.concatenate(values["p"])
            pooled_rows = np.concatenate(values["rows"])
            per_cycle_scores = []
            for cycle_value in sorted(np.unique(cycle_all[pooled_rows])):
                mask = cycle_all[pooled_rows] == cycle_value
                per_cycle_scores.append(binary_metrics(pooled_y[mask], pooled_p[mask], 0.50)["macro_f1"])
            metrics[family][name] = {
                "metrics": binary_metrics(pooled_y, pooled_p, 0.50),
                "equal_cycle_macro_f1": float(np.mean(per_cycle_scores)),
                "per_cycle_count": int(len(per_cycle_scores)),
                "subgroups": {},
            }
            for label, mask in (("stream_present", ~missing_all[pooled_rows]),
                                ("stream_missing", missing_all[pooled_rows])):
                subgroup = binary_metrics(pooled_y[mask], pooled_p[mask], 0.50)
                metrics[family][name]["subgroups"][label] = subgroup
                subgroups_report.append({"family": family, "model": name, "group": label,
                                         "n": subgroup["n"], "positive_rate": subgroup["positive_rate"],
                                         "macro_f1": subgroup["macro_f1"],
                                         "false_positive": subgroup["false_positive"],
                                         "false_negative": subgroup["false_negative"]})
            bins = np.concatenate(values["volume_bin"])
            for bin_index, bin_label in enumerate(VOLUME_LABELS):
                mask = bins == bin_index
                subgroup = binary_metrics(pooled_y[mask], pooled_p[mask], 0.50)
                metrics[family][name]["subgroups"][f"text_load_{bin_label}"] = subgroup
                subgroups_report.append({"family": family, "model": name, "group": f"text_load_{bin_label}",
                                         "n": subgroup["n"], "positive_rate": subgroup["positive_rate"],
                                         "macro_f1": subgroup["macro_f1"],
                                         "false_positive": subgroup["false_positive"],
                                         "false_negative": subgroup["false_negative"]})
        baseline_by_window = {
            row["window"]: row["macro_f1"] for row in windows_report
            if row["family"] == family and row["model"] == "C4_HGB"
        }
        for row in windows_report:
            if row["family"] == family and row["model"] != "C4_HGB":
                row["delta_vs_C4"] = row["macro_f1"] - baseline_by_window[row["window"]]

    summary = {}
    for name in MODEL_NAMES:
        pri = metrics["primary"][name]
        h12 = metrics["h12"][name]
        base_pri = metrics["primary"]["C4_HGB"]
        base_h12 = metrics["h12"]["C4_HGB"]
        primary_window_deltas = [row["delta_vs_C4"] for row in windows_report
                                 if row["family"] == "primary" and row["model"] == name] if name != "C4_HGB" else []
        class_deltas = []
        availability_deltas = []
        if name != "C4_HGB":
            for family_name in ("primary", "h12"):
                for class_index in (0, 1):
                    class_deltas.append(metrics[family_name][name]["metrics"]["per_class"][class_index]["f1"] -
                                        metrics[family_name]["C4_HGB"]["metrics"]["per_class"][class_index]["f1"])
                for group in ("stream_present", "stream_missing"):
                    availability_deltas.append(metrics[family_name][name]["subgroups"][group]["macro_f1"] -
                                               metrics[family_name]["C4_HGB"]["subgroups"][group]["macro_f1"])
        gate = read_json(PROTOCOL_PATH)["comparison_and_candidate_guard"]["external_candidate_guard"]
        guard = None
        if name != "C4_HGB":
            guard = {
                "primary": pri["metrics"]["macro_f1"] - base_pri["metrics"]["macro_f1"] >= gate["primary_delta_at_least"],
                "equal_cycle_primary": pri["equal_cycle_macro_f1"] - base_pri["equal_cycle_macro_f1"] >= gate["equal_cycle_primary_delta_at_least"],
                "h12": h12["metrics"]["macro_f1"] - base_h12["metrics"]["macro_f1"] >= gate["h12_delta_at_least"],
                "worst_primary_window": bool(primary_window_deltas) and min(primary_window_deltas) >= gate["worst_primary_window_delta_at_least"],
                "per_class_f1": bool(class_deltas) and min(class_deltas) >= gate["per_class_f1_delta_at_least"],
                "availability_subgroup_f1": bool(availability_deltas) and min(availability_deltas) >= gate["availability_subgroup_f1_delta_at_least"],
            }
            guard["all_guards_pass"] = all(guard.values())
            guard["positive_pooled_primary_or_h12"] = (
                pri["metrics"]["macro_f1"] - base_pri["metrics"]["macro_f1"] > 0 or
                h12["metrics"]["macro_f1"] - base_h12["metrics"]["macro_f1"] > 0
            )
            guard["recommend_for_parent_review"] = guard["all_guards_pass"] and guard["positive_pooled_primary_or_h12"]
        summary[name] = {
            "primary_macro_f1": pri["metrics"]["macro_f1"],
            "primary_equal_cycle_macro_f1": pri["equal_cycle_macro_f1"],
            "h12_macro_f1": h12["metrics"]["macro_f1"],
            "delta_primary": pri["metrics"]["macro_f1"] - base_pri["metrics"]["macro_f1"],
            "delta_equal_cycle_primary": pri["equal_cycle_macro_f1"] - base_pri["equal_cycle_macro_f1"],
            "delta_h12": h12["metrics"]["macro_f1"] - base_h12["metrics"]["macro_f1"],
            "primary_windows_positive": int(sum(value > 0 for value in primary_window_deltas)),
            "primary_windows_negative": int(sum(value < 0 for value in primary_window_deltas)),
            "worst_primary_window_delta": min(primary_window_deltas) if primary_window_deltas else 0.0,
            "guard": guard,
        }
    return {"aggregate": metrics, "summary": summary, "windows": windows_report,
            "cycles": cycles_report, "subgroups": subgroups_report}


def make_delta_figure(evaluation: dict[str, Any], path: Path) -> None:
    windows = pd.DataFrame(evaluation["windows"])
    names = CANDIDATES
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    primary = windows[(windows["family"] == "primary") & windows["model"].isin(names)]
    window_order = ["W1", "W2", "W3", "W4", "W5", "W6"]
    colors = {"present_specialist": "#1F77B4", "missing_specialist": "#E07A2D", "both_specialist": "#4B8B5B"}
    for name in names:
        part = primary[primary["model"] == name].set_index("window").loc[window_order]
        axes[0].plot(window_order, 100 * part["delta_vs_C4"], marker="o", linewidth=1.8,
                     label=name.replace("_", " "), color=colors[name])
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set_ylabel("Macro F1 change vs C4 (percentage points)")
    axes[0].set_title("Primary temporal windows")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    summary = evaluation["summary"]
    labels = ["Primary pooled", "Primary equal-cycle", "H12 pooled"]
    x = np.arange(len(labels))
    offsets = [-0.22, 0.0, 0.22]
    for name, offset in zip(names, offsets):
        values = [summary[name]["delta_primary"], summary[name]["delta_equal_cycle_primary"], summary[name]["delta_h12"]]
        axes[1].plot(x + offset, 100 * np.asarray(values), marker="o", linestyle="none",
                     markersize=7, label=name.replace("_", " "), color=colors[name])
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_ylabel("Macro F1 change vs C4 (percentage points)")
    axes[1].set_title("Pooled and equal-cycle comparisons")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Availability-specialist ablation: fixed C4 routing and threshold", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    protocol = read_json(PROTOCOL_PATH)
    check_protocol_matches_round2(protocol)
    frame, y, X, source = load_train()
    frozen = verify_freeze(frame, source)
    if RUN_DIR.exists() and any(RUN_DIR.rglob("*.npz")):
        raise FileExistsError(f"Prediction caches already exist; refusing overwrite: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    primary = protocol["validation"]["primary_windows"]
    secondary = protocol["validation"]["secondary_windows"]
    controls, _control_provenance = control_vectors(frame, primary, secondary)
    missing_all = stream_missing_mask(frame)
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    y_by_fold: dict[int, dict[str, Any]] = {}
    support_rows, edges_by_fit = fit_support_rows(frame, y)
    events_path = RUN_DIR / "events.jsonl"
    fit_count = 0
    with events_path.open("w", encoding="utf-8") as events:
        for fit_end in FIT_ENDS:
            fold_rows = controls[fit_end]["rows"]
            fit_mask = cycle <= fit_end
            X_fit = X.loc[fit_mask].reset_index(drop=True)
            y_fit = y[fit_mask]
            fit_missing = stream_missing_mask(X_fit)
            X_outer = X.iloc[fold_rows]
            outer_missing = missing_all[fold_rows]
            c2_model, c2_audit = fit_specialist("C2_B", X_fit, y_fit, ~fit_missing, "present", fit_end)
            c2_before = c2_model.audit_state()
            c2_rows = fold_rows[~outer_missing]
            with threadpool_limits(limits=2):
                c2_probability = c2_model.predict_proba(X.iloc[c2_rows])
            c2_unchanged = c2_model.audit_state() == c2_before
            fit_count += 1
            events.write(json.dumps({"fit_end": fit_end, "branch": "present", "fit_rows": c2_audit["fit_rows"],
                                     "elapsed_seconds": c2_audit["elapsed_seconds"], "completed_fits": fit_count},
                                    allow_nan=False) + "\n")
            events.flush()

            c1_model, c1_audit = fit_specialist("C1_B", X_fit, y_fit, fit_missing, "missing", fit_end)
            c1_before = c1_model.audit_state()
            c1_rows = fold_rows[outer_missing]
            with threadpool_limits(limits=2):
                c1_probability = c1_model.predict_proba(X.iloc[c1_rows])
            c1_unchanged = c1_model.audit_state() == c1_before
            fit_count += 1
            events.write(json.dumps({"fit_end": fit_end, "branch": "missing", "fit_rows": c1_audit["fit_rows"],
                                     "elapsed_seconds": c1_audit["elapsed_seconds"], "completed_fits": fit_count},
                                    allow_nan=False) + "\n")
            events.flush()

            baseline = controls[fit_end]
            routes = {
                "C4_HGB": np.where(outer_missing, baseline["C1_B"], baseline["C2_B"]),
                "present_specialist": np.where(outer_missing, baseline["C1_B"],
                                                _route_branch_probability(fold_rows, c2_rows, c2_probability)),
                "missing_specialist": np.where(outer_missing,
                                                _route_branch_probability(fold_rows, c1_rows, c1_probability), baseline["C2_B"]),
                "both_specialist": np.where(outer_missing,
                                              _route_branch_probability(fold_rows, c1_rows, c1_probability),
                                              _route_branch_probability(fold_rows, c2_rows, c2_probability)),
            }
            for name, probability in routes.items():
                if probability.shape != fold_rows.shape or not np.isfinite(probability).all():
                    raise ValueError(f"Invalid route probability vector {name}/fit{fit_end}")
            cache_path = RUN_DIR / f"fit{fit_end}.npz"
            np.savez_compressed(
                cache_path,
                rows=fold_rows,
                C1_B_all_probability=baseline["C1_B"],
                C2_B_all_probability=baseline["C2_B"],
                C2_B_present_rows=c2_rows,
                C2_B_present_probability=c2_probability,
                C1_B_missing_rows=c1_rows,
                C1_B_missing_probability=c1_probability,
                **{f"{name}_probability": value for name, value in routes.items()},
            )
            audit = {
                "fit_end_cycle": fit_end,
                "fit_rows_total": int(fit_mask.sum()),
                "outer_rows": int(len(fold_rows)),
                "outer_cycle_min": int(cycle[fold_rows].min()),
                "outer_cycle_max": int(cycle[fold_rows].max()),
                "baseline_control_c1_b_probability_sha256": _array_digest(baseline["C1_B"]),
                "baseline_control_c2_b_probability_sha256": _array_digest(baseline["C2_B"]),
                "new_fits": {"C2_B_present_only": c2_audit, "C1_B_missing_only": c1_audit},
                "prediction_did_not_change_fitted_state": {"C2_B_present_only": c2_unchanged,
                                                            "C1_B_missing_only": c1_unchanged},
                "branch_prediction_sha256": {"C2_B_present_only": _array_digest(c2_probability),
                                              "C1_B_missing_only": _array_digest(c1_probability)},
                "route_prediction_sha256": {name: _array_digest(value) for name, value in routes.items()},
                "cache_sha256": sha256_file(cache_path),
                "validation_labels_passed_to_fit": False,
            }
            write_json(RUN_DIR / f"fit{fit_end}_audit.json", audit)
            y_by_fold[fit_end] = {"rows": fold_rows, **routes}
            print(json.dumps({"fit_end": fit_end, "completed_fits": fit_count,
                              "present_fit_rows": c2_audit["fit_rows"],
                              "missing_fit_rows": c1_audit["fit_rows"],
                              "route_hashes": audit["route_prediction_sha256"]}, ensure_ascii=False))
    if fit_count != 12:
        raise RuntimeError(f"Expected exactly 12 new branch fits, observed {fit_count}")

    evaluation = evaluate_predictions(frame, y, y_by_fold, edges_by_fit)
    results = {
        "status": "COMPLETED_FROZEN_TEMPORAL_ABLATION",
        "protocol_id": protocol["protocol_id"],
        "freeze_signature": frozen["freeze_signature"],
        "source_head": frozen["source_head"],
        "input_sha256": frozen["train"]["sha256"],
        "threshold": 0.50,
        "primary_and_h12_overlap": True,
        "new_branch_fit_count": fit_count,
        "official_test_values_read": False,
        "official_test_target_used": False,
        "kaggle_submissions": 0,
        "adaptive_development_warning": "All cycles/folds were used in prior research; H12 reuses primary records and is not an untouched holdout.",
        "aggregate": evaluation["aggregate"],
        "summary": evaluation["summary"],
        "windows": evaluation["windows"],
        "cycles": evaluation["cycles"],
        "subgroups": evaluation["subgroups"],
        "gate": read_json(PROTOCOL_PATH)["comparison_and_candidate_guard"]["external_candidate_guard"],
    }
    results_path = REPORT_DIR / "results.json"
    write_json(results_path, results)
    summary_rows = []
    for name, item in evaluation["summary"].items():
        summary_rows.append({"model": name, **item})
    write_csv(REPORT_DIR / "summary.csv", summary_rows)
    write_csv(REPORT_DIR / "windows.csv", evaluation["windows"])
    write_csv(REPORT_DIR / "per_cycle.csv", evaluation["cycles"])
    write_csv(REPORT_DIR / "subgroups.csv", evaluation["subgroups"])
    make_delta_figure(evaluation, REPORT_DIR / "figures" / "temporal_model_deltas.png")
    report = render_report(results, frozen)
    (REPORT_DIR / "FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": results["status"], "new_branch_fit_count": fit_count,
                      "summary": results["summary"], "results": str(results_path)},
                     indent=2, ensure_ascii=False))


def _route_branch_probability(all_rows: np.ndarray, branch_rows: np.ndarray, branch_p: np.ndarray) -> np.ndarray:
    if len(branch_rows) != len(branch_p):
        raise ValueError("Branch prediction rows and probabilities are misaligned")
    lookup = {int(row): float(probability) for row, probability in zip(branch_rows, branch_p)}
    return np.asarray([lookup.get(int(row), np.nan) for row in all_rows], dtype=float)


def render_report(results: dict[str, Any], frozen: dict[str, Any]) -> str:
    summary = results["summary"]
    baseline = summary["C4_HGB"]
    lines = [
        "# Autonomous Classification: availability-specialist ablation",
        "",
        "## Pertanyaan dan desain",
        "",
        "Apakah melatih masing-masing expert C4 hanya pada populasi ketersediaan yang menerima prediksinya dapat mengurangi gangguan antar-populasi? C2_B (structured + TF-IDF Logistic Regression) tetap menjadi expert untuk stream-present; C1_B (expanded structured HGB) tetap menjadi expert untuk stream-missing. Tiga kandidat hanya mengubah baris yang masuk ke fit expert. Fitur, preprocessing, hyperparameter, routing berdasarkan availability, serta threshold 0,50 tidak diubah.",
        "",
        "Kandidat present-only, missing-only, dan both-specialist dibekukan sebelum fit. Dua expert subset baru per cutoff dipakai ulang lintas ketiga route, sehingga ada tepat 12 fit baru. Kontrol C1_B/C2_B berasal dari cache Round 1 yang probabilitasnya cocok persis dengan audit. Tidak ada threshold tuning, pencarian model, atau akses ke nilai official Test.",
        "",
        "## Konteks EDA mentah",
        "",
        "Pertanyaan raw EDA adalah apakah prevalensi artifact dan dukungan text_load dalam populasi availability berubah antarperiode. Ini deskripsi input dan label Train; tidak menyamakan availability dengan penyebab label. EDA sebelumnya mencatat gap missing-vs-present sebesar −1,84 pp pada cycle 47–57, +10,01 pp pada 58–68, dan +3,18 pp pada 69–79; volume adjustment juga dapat membalik arah gap. Jumlah serta prevalensi fit subset yang benar-benar dipakai tiap cutoff tersedia di `fit_support.csv`; rate per periode dan IQR text_load ada di `raw_eda_by_period.csv`.",
        "",
        "![Prevalence dan text_load menurut ketersediaan](figures/availability_period_volume.png)",
        "",
        "Grafik menunjukkan perubahan komposisi temporal dan skala volume antarstatus. Itu mendukung pengujian spesialisasi sebagai hipotesis tentang conditional population, tetapi tidak membuktikan bahwa fitting subset akan lebih baik.",
        "",
        "## Hasil temporal",
        "",
        f"Kontrol C4_HGB yang direkonstruksi menghasilkan primary Macro F1 **{baseline['primary_macro_f1']:.6f}**, equal-cycle **{baseline['primary_equal_cycle_macro_f1']:.6f}**, dan H12 **{baseline['h12_macro_f1']:.6f}**. Ketiga nilai sama dengan kontrol Round 2 pada toleransi 1e−12.",
        "",
        "| Model | Primary F1 | Δ primary | Equal-cycle F1 | Δ equal-cycle | H12 F1 | Δ H12 | Window positif | Guard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in MODEL_NAMES:
        item = summary[name]
        guard = "control" if name == "C4_HGB" else ("PASS" if item["guard"]["recommend_for_parent_review"] else "FAIL")
        lines.append(f"| {name} | {item['primary_macro_f1']:.6f} | {item['delta_primary']:+.6f} | "
                     f"{item['primary_equal_cycle_macro_f1']:.6f} | {item['delta_equal_cycle_primary']:+.6f} | "
                     f"{item['h12_macro_f1']:.6f} | {item['delta_h12']:+.6f} | "
                     f"{item['primary_windows_positive']}/6 | {guard} |")
    lines += [
        "",
        "![Delta temporal](figures/temporal_model_deltas.png)",
        "",
        "Per-window metrics dan delta, per-cycle metrics, per-class precision/recall/F1, availability subgroup, serta fit-defined text_load quintiles tersedia di `windows.csv`, `per_cycle.csv`, dan `subgroups.csv`. Setiap subgroup menyertakan denominator, positive rate, Macro F1, false positives, dan false negatives.",
        "",
        "## Interpretasi dan keputusan",
        "",
        "Nilai guard adalah penyaring praktis sebelum MAIN mempertimbangkan paling banyak satu kandidat eksternal; guard bukan uji signifikansi atau keputusan upload otomatis. Kandidat harus tidak jauh lebih buruk pada primary, equal-cycle, H12, window terburuk, kelas, dan availability subgroup, serta menunjukkan kenaikan pooled primary atau H12 agar direkomendasikan untuk review. Hasil tetap development evidence karena folds sudah pernah diamati.",
        "",
        "Kandidat paling layak menurut bukti ini: **" + _recommended_name(summary) + "**. Bila tertulis `NONE`, tidak ada kandidat memenuhi seluruh guard dan menunjukkan pooled gain positif. MAIN tetap memegang keputusan akhir dan semua refit full-Train/inference/submission.",
        "",
        "## Batas interpretasi",
        "",
        "Availability adalah atribut proses pengumpulan, bukan intervensi kausal. Subset training dapat mengurangi penggunaan contoh dari populasi lain, tetapi juga membuang informasi bersama. Stream availability dan text_load tidak menghilangkan confounding; H12 memakai ulang baris primary; seluruh siklus merupakan development data; hasil tidak menjamin performa official Test atau Kaggle. Hasil ini tidak mengubah model resmi, EDA resmi, atau submission.",
        "",
        f"Input Train SHA-256 `{frozen['train']['sha256']}`. Freeze signature `{frozen['freeze_signature']}`. Fit baru: {results['new_branch_fit_count']}; submission Kaggle: 0.",
        "",
    ]
    return "\n".join(lines)


def _recommended_name(summary: dict[str, Any]) -> str:
    eligible = [name for name in CANDIDATES if summary[name]["guard"]["recommend_for_parent_review"]]
    if not eligible:
        return "NONE"
    eligible.sort(key=lambda name: (
        summary[name]["primary_macro_f1"], summary[name]["h12_macro_f1"],
        summary[name]["primary_equal_cycle_macro_f1"], -len(name)
    ), reverse=True)
    return eligible[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="write the pre-fit freeze after baseline/EDA checks")
    group.add_argument("--run", action="store_true", help="run the frozen 12-fit temporal ablation")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
