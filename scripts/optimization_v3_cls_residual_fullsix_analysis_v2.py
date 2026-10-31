"""Post-fit analysis-only adapter for tied empirical subgroup quantiles.

The frozen runner and all four fitted heads remain unchanged. This adapter only
reuses saved vectors and allows tied raw quartile cutpoints to produce explicit
empty bins rather than aborting report generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_residual_fullsix as frozen
from scripts.submit_regression_research import ordered_ids_sha256
from src.data.audit import sha256 as file_sha256

_ORIGINAL_ARRAY_DIGEST = frozen.array_digest

AMENDMENT = frozen.CONFIG / "analysis_amendment_01.json"
AMENDMENT_02 = frozen.CONFIG / "analysis_amendment_02.json"
AMENDMENT_03 = frozen.CONFIG / "analysis_amendment_03.json"
ANALYSIS_SOURCE = ROOT / "scripts/optimization_v3_cls_residual_fullsix_analysis_v2.py"


def raw_train_edges(frame: pd.DataFrame, cutoff: int, feature: str) -> list[float]:
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    values = pd.to_numeric(frame.loc[(cycles >= 47) & (cycles <= cutoff), feature], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite fit-period feature values for {feature} at cutoff {cutoff}")
    edges = np.quantile(values, [0.25, 0.5, 0.75]).astype(float)
    if not np.isfinite(edges).all():
        raise ValueError(f"Nonfinite empirical quantile for {feature} at cutoff {cutoff}")
    return edges.tolist()


def quartile_codes(values: np.ndarray, edges: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    codes = np.full(values.shape, -1, dtype=np.int8)
    finite = np.isfinite(values)
    codes[finite] = np.digitize(values[finite], np.asarray(edges, dtype=float), right=True).astype(np.int8)
    return codes


def stable_identifier_digest(values: np.ndarray) -> str:
    arr = np.asarray(values)
    return ordered_ids_sha256([str(v) for v in arr.ravel().tolist()])


def stable_array_digest(values: np.ndarray) -> str:
    arr = np.asarray(values)
    if arr.dtype.kind in "USO":
        return stable_identifier_digest(arr)
    return _ORIGINAL_ARRAY_DIGEST(values)


def cached_record(frame: pd.DataFrame, cycle: np.ndarray, missing: np.ndarray, cutoff: int) -> dict:
    outer = frozen.control_window(frame, cycle, missing, cutoff)
    path = frozen.RUN_DIR / f"fit{cutoff}_{frozen.HEAD_NAME}.npz"
    sidecar = frozen.read_json(path.with_suffix(".json"))
    if file_sha256(path) != sidecar["prediction_sha256"]:
        raise AssertionError(f"Persisted head prediction hash mismatch at {cutoff}")
    with np.load(path, allow_pickle=False) as z:
        rows = z["outer_rows"].astype(np.int64, copy=True)
        candidate = z["candidate_probability"].astype(float, copy=True)
        if not np.array_equal(z["missing_mask"].astype(bool), outer["missing"]):
            raise AssertionError(f"Persisted missing mask mismatch at {cutoff}")
        for key, expected in (("base_probability", outer["base"]),
                              ("cb_probability", outer["cb"]),
                              ("champion_probability", outer["champion"]),
                              ("sparse25_probability", outer["sparse25"])):
            if not np.array_equal(z[key], expected):
                raise AssertionError(f"Persisted frozen comparator mismatch at {cutoff}/{key}")
    if not np.array_equal(rows, outer["rows"]) or not np.isfinite(candidate).all():
        raise AssertionError(f"Persisted output alignment/probability invalid at {cutoff}")
    if not np.array_equal(candidate[outer["missing"]], outer["champion"][outer["missing"]]):
        raise AssertionError(f"Persisted missing route changed at {cutoff}")
    return {**outer, "candidate": candidate, "successful_fit_sidecar": sidecar}


def evaluation_slices() -> dict[str, list[tuple[int, int, int]]]:
    slices = {name: [(end, lo, hi)] for name, (end, lo, hi) in frozen.PRIMARY_WINDOWS.items()}
    slices["primary_six_window_pooled"] = [tuple(v) for v in frozen.PRIMARY_WINDOWS.values()]
    slices["H12_E"] = [(55, 56, 67)]
    slices["H12_L"] = [(67, 68, 79)]
    slices["H12_fixed_origin_pooled"] = slices["H12_E"] + slices["H12_L"]
    return slices


def run_analysis() -> dict:
    if not AMENDMENT.is_file() or not AMENDMENT_02.is_file() or not AMENDMENT_03.is_file():
        raise FileNotFoundError("Post-fit aggregation amendments must be recorded before analysis")
    amendment = frozen.read_json(AMENDMENT)
    amendment_02 = frozen.read_json(AMENDMENT_02)
    amendment_03 = frozen.read_json(AMENDMENT_03)
    freeze = frozen.check_freeze()
    if amendment.get("fit_freeze_signature") != freeze["signature"]:
        raise AssertionError("Analysis amendment is not bound to the frozen fit inputs")
    if amendment_02.get("fit_freeze_signature") != freeze["signature"]:
        raise AssertionError("Identifier hashing amendment is not bound to the frozen fit inputs")
    if amendment_03.get("fit_freeze_signature") != freeze["signature"]:
        raise AssertionError("Canonical ID hash amendment is not bound to the frozen fit inputs")
    frame, y, cycle, missing, _ = frozen.load_train()
    # The only adapted function is subgroup edge construction; numerical raw
    # quantiles are retained even when repeated, so the frozen digitize rule
    # produces explicit empty bins for tied quartiles.
    frozen.train_edges = raw_train_edges
    frozen.array_digest = stable_array_digest
    records = {55: frozen.reuse_pilot(55), 67: frozen.reuse_pilot(67)}
    for end in frozen.NEW_ENDS:
        records[end] = cached_record(frame, cycle, missing, end)
    # The frozen aggregate casts record_id to float before hashing. Use a
    # temporary positional numeric placeholder only for that internal call,
    # then replace the placeholder hashes with hashes of original string IDs.
    aggregate_frame = frame.copy()
    aggregate_frame["record_id"] = np.arange(len(frame), dtype=np.int64)
    evaluation = frozen.aggregate(aggregate_frame, y, cycle, records)
    for view, parts in evaluation_slices().items():
        row_parts = []
        for end, lo, hi in parts:
            rr = records[end]["rows"]
            keep = (cycle[rr] >= lo) & (cycle[rr] <= hi)
            row_parts.append(rr[keep])
        exact_rows = np.concatenate(row_parts)
        evaluation[view]["row_ids_sha256"] = stable_identifier_digest(
            frame.iloc[exact_rows]["record_id"].astype(str).to_numpy()
        )
    entry_edges = {str(end): raw_train_edges(frame, end, "entry_friction") for end in frozen.ENDS}
    text_edges = {str(end): raw_train_edges(frame, end, "text_load") for end in frozen.ENDS}
    result = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "freeze_signature": freeze["signature"],
        "freeze_identity": freeze["identity"],
        "new_fits": 4,
        "new_fit_cutoffs": frozen.NEW_ENDS,
        "reused_cutoffs": frozen.REUSED_ENDS,
        "head": frozen.HEAD_NAME,
        "head_recipe": {"utility": "src.models.optimization_v3_residual.OOFResidualHead",
                        "objective": "binary LightGBM raw-margin correction",
                        "params": {"n_estimators": 200, "num_leaves": 7,
                                   "min_child_samples": 80, "reg_lambda": 10.0,
                                   "learning_rate": 0.03, "random_state": 42,
                                   "n_jobs": 2},
                        "initial_margin": "logit(C2_B OOF probability)",
                        "shrinkage": 0.5, "threshold": 0.5},
        "test_read": False, "kaggle_access": False,
        "outer_labels_used_for_fit_or_selection": False,
        "missing_route_unchanged": True,
        "execution_accounting": {
            "successful_persisted_designated_cutoff_fits": 4,
            "successful_fit_cutoffs": frozen.NEW_ENDS,
            "unsaved_execution_attempts": 1,
            "unsaved_attempt_receipt": "experiments/runs/optimization_v3_cls_residual_fullsix/attempt_01_cutoff59_failure.json",
            "recovery_retry_cutoff": 59,
            "candidate_configurations_added": 0,
            "verification_replay_fits": 0,
        },
        "fit_metadata": {str(end): (records[end]["successful_fit_sidecar"] if end in frozen.NEW_ENDS else {
            "prediction_cache_sha256": records[end]["pilot_cache_sha256"],
            "model_sha256": records[end]["head_model_sha256"], "reused": True}) for end in frozen.ENDS},
        "evaluation": evaluation,
        "fit_only_subgroup_cutpoints": {"entry_friction": entry_edges, "text_load": text_edges},
        "postfit_analysis_amendment": {
            "path": str(AMENDMENT.relative_to(ROOT)),
            "sha256": file_sha256(AMENDMENT),
            "identifier_hash_amendment_path": str(AMENDMENT_02.relative_to(ROOT)),
            "identifier_hash_amendment_sha256": file_sha256(AMENDMENT_02),
            "canonical_id_hash_amendment_path": str(AMENDMENT_03.relative_to(ROOT)),
            "canonical_id_hash_amendment_sha256": file_sha256(AMENDMENT_03),
            "canonical_identifier_hash_helper": "scripts.submit_regression_research.ordered_ids_sha256",
            "canonical_identifier_hash_helper_sha256": file_sha256(ROOT / "scripts/submit_regression_research.py"),
            "analysis_source": str(ANALYSIS_SOURCE.relative_to(ROOT)),
            "analysis_source_sha256": file_sha256(ANALYSIS_SOURCE),
        },
        "interpretation_limits": [
            "Primary six-window and H12 metrics are development evidence, not an untouched holdout.",
            "H12 fixed-origin views overlap the chronological region represented by the primary windows.",
            "Entry-friction and text-load cut points use feature values from each model's training period only.",
            "entry_friction is binary over the fit population: its raw empirical quartiles are [0,0,1] at all six cutoffs. Tied edges are retained; empty Q2/Q4 cells are shown explicitly rather than filled or recut.",
            "No model, threshold, or route is selected from these comparisons in this lane.",
        ],
    }
    report_path = frozen.REPORT_DIR / "fullsix.json"
    md_path = frozen.REPORT_DIR / "FULLSIX.md"
    if report_path.exists() or md_path.exists():
        raise FileExistsError("Refusing to overwrite existing fullsix analysis")
    frozen.write_json_new(report_path, result)
    text = frozen.report_markdown(result)
    text += ("\n## Kuartil entry_friction dengan ties\n\n"
             "`entry_friction` memiliki dua nilai pada data fit. Pada semua cutoff, raw quartile edges adalah `[0, 0, 1]`; aturan `digitize` yang sama mempertahankan observasi, sementara kuartil dengan batas sama dapat kosong. Sel kosong tercatat sebagai `n=0` pada JSON; tidak ada cutpoint yang digeser atau subgroup yang direkayasa.\n")
    md_path.write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    out = run_analysis()
    p = out["evaluation"]["primary_six_window_pooled"]
    h = out["evaluation"]["H12_fixed_origin_pooled"]
    print(json.dumps({"status": out["status"],
                      "primary_delta": p["delta_vs_champion"],
                      "h12_delta": h["delta_vs_champion"],
                      "entry_edges": out["fit_only_subgroup_cutpoints"]["entry_friction"]}, indent=2))
