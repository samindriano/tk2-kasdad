"""Versioned independent replay for the tied-edge/string-ID fullsix report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_residual_fullsix as run
from scripts import optimization_v3_verify_cls_residual_fullsix as v1
from scripts import optimization_v3_cls_residual_fullsix_analysis_v2 as analysis
from scripts.submit_regression_research import ordered_ids_sha256
from src.data.audit import sha256 as file_sha256
from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_evaluation import digest_json

VERIFY_FREEZE = run.REPORT_DIR / "verification_freeze_v2.json"
VERIFY_RESULT = run.REPORT_DIR / "verification_v2.json"


def dry_run_metric_schema() -> dict:
    value = v1.metric(np.array([0, 1, 0, 1], dtype=np.int8), np.array([.2, .8, .7, .3]))
    v1.compare(value, dict(value), "dry")
    report = run.read_json(run.REPORT_DIR / "fullsix.json")
    expected = report["evaluation"]["W1"]["models"]["residual_A"]["all"]
    if set(value) != set(expected):
        raise AssertionError("Metric schema differs from fullsix result")
    return {"status": "PASS", "fields": sorted(expected), "brier": value["brier"]}


def verification_identity() -> dict:
    freeze = run.read_json(run.FREEZE)
    current_fit_identity = run.identity()
    if current_fit_identity != freeze["identity"] or digest_json(current_fit_identity) != freeze["signature"]:
        raise AssertionError("Fit source/data identity differs from immutable prefit freeze")
    report_path = run.REPORT_DIR / "fullsix.json"
    report = run.read_json(report_path)
    if report.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION":
        raise AssertionError("Analysis result is not ready for independent replay")
    amendment = report.get("postfit_analysis_amendment", {})
    amendment_paths = [run.CONFIG / f"analysis_amendment_{i:02d}.json" for i in (1, 2, 3)]
    failed_receipt = run.RUN_DIR / "attempt_01_cutoff59_failure.json"
    missing = [str(p) for p in [*amendment_paths, failed_receipt] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Recovery or analysis amendment receipt missing: {missing}")
    if amendment.get("analysis_source_sha256") != file_sha256(ROOT / "scripts/optimization_v3_cls_residual_fullsix_analysis_v2.py"):
        raise AssertionError("Report is not bound to current post-fit analysis adapter")
    if amendment.get("canonical_identifier_hash_helper_sha256") != file_sha256(ROOT / "scripts/submit_regression_research.py"):
        raise AssertionError("Canonical ordered-ID hash helper changed after analysis")
    outputs = {}
    for end in run.NEW_ENDS:
        for suffix in ("npz", "txt", "json"):
            path = run.RUN_DIR / f"fit{end}_{run.HEAD_NAME}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Persisted fit output missing: {path}")
            outputs[path.resolve().relative_to(ROOT).as_posix()] = file_sha256(path)
    return {
        "fit_freeze_signature": freeze["signature"],
        "fit_source_data_identity_sha256": digest_json(current_fit_identity),
        "report_sha256": file_sha256(report_path),
        "fit_output_sha256": outputs,
        "failure_attempt_receipt_sha256": file_sha256(failed_receipt),
        "amendment_sha256": {p.name: file_sha256(p) for p in amendment_paths},
        "analysis_source_sha256": file_sha256(ROOT / "scripts/optimization_v3_cls_residual_fullsix_analysis_v2.py"),
        "canonical_ordered_id_helper_sha256": file_sha256(ROOT / "scripts/submit_regression_research.py"),
        "source_sha256": {
            "frozen_fit_runner": file_sha256(ROOT / "scripts/optimization_v3_cls_residual_fullsix.py"),
            "frozen_v1_verifier": file_sha256(ROOT / "scripts/optimization_v3_verify_cls_residual_fullsix.py"),
            "v2_verifier": file_sha256(Path(__file__).resolve()),
            "residual_utility": file_sha256(ROOT / "src/models/optimization_v3_residual.py"),
        },
        "metric_schema_dry_run": dry_run_metric_schema(),
        "test_read": False, "kaggle_access": False,
    }


def freeze_verification() -> dict:
    identity = verification_identity()
    sig = digest_json(identity)
    if VERIFY_FREEZE.exists():
        old = run.read_json(VERIFY_FREEZE)
        if old.get("identity") != identity or old.get("signature") != sig:
            raise RuntimeError("Existing V2 verification freeze differs; preserve and use a new version")
        return old
    receipt = {"status": "FROZEN_BEFORE_V2_DETERMINISTIC_REPLAY",
               "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "identity": identity, "signature": sig,
               "source_boundary": "Replays only four already fitted Head A outputs and independently reconstructs all row metrics; no candidate fits, Test or Kaggle access."}
    run.write_json_new(VERIFY_FREEZE, receipt)
    return receipt


def check_verification_freeze() -> dict:
    if not VERIFY_FREEZE.is_file():
        raise RuntimeError("Run --freeze-only to preflight the metric schema and bind report/artifacts before replay")
    receipt = run.read_json(VERIFY_FREEZE)
    current = verification_identity()
    if current != receipt.get("identity") or digest_json(current) != receipt.get("signature"):
        raise RuntimeError("Report, cache, source, or amendment changed after V2 verifier freeze")
    return receipt


def verify_metrics(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                   records: dict[int, dict], report: dict) -> int:
    checks = 0
    slices = v1.expected_slices()
    if set(report["evaluation"]) != set(slices):
        raise AssertionError("View list differs from the frozen six-window protocol")
    for view, parts in slices.items():
        rows_l, missing_l, cut_l = [], [], []
        vals = {name: [] for name in ("champion", "residual_A", "sparse25")}
        for end, lo, hi in parts:
            rec = records[end]
            take = (cycle[rec["rows"]] >= lo) & (cycle[rec["rows"]] <= hi)
            rr = rec["rows"][take]
            rows_l.append(rr); missing_l.append(rec["missing"][take]); cut_l.append(np.full(len(rr), end, dtype=np.int16))
            for name in vals:
                vals[name].append(rec[name][take])
        rows = np.concatenate(rows_l)
        miss = np.concatenate(missing_l).astype(bool)
        cutoff = np.concatenate(cut_l)
        if rows.size != np.unique(rows).size:
            raise AssertionError(f"Duplicate row indices in {view}")
        vectors = {k: np.concatenate(v) for k, v in vals.items()}
        ev = report["evaluation"][view]
        if ev["n"] != len(rows) or ev["row_indices_sha256"] != run.array_digest(rows):
            raise AssertionError(f"Row denominator/index hash differs in {view}")
        exact_ids = frame.iloc[rows]["record_id"].astype(str).tolist()
        if ev["row_ids_sha256"] != ordered_ids_sha256(exact_ids):
            raise AssertionError(f"Canonical ordered record_id hash differs in {view}")
        yy = y[rows]; cc = cycle[rows]
        for name, p in vectors.items():
            parts_mask = {"all": np.ones(len(rows), dtype=bool),
                          "stream_present": ~miss, "stream_missing": miss}
            for label, mask in parts_mask.items():
                if mask.any():
                    v1.compare(ev["models"][name][label], v1.metric(yy[mask], p[mask]), f"{view}.{name}.{label}")
                    checks += 1
            by_cycle = {str(int(c)): v1.metric(yy[cc == c], p[cc == c]) for c in sorted(np.unique(cc))}
            v1.compare(ev["models"][name]["by_cycle"], by_cycle, f"{view}.{name}.by_cycle")
            mean_cycle = float(np.mean([z["macro_f1"] for z in by_cycle.values()]))
            if not np.isclose(ev["models"][name]["mean_cycle_macro_f1"], mean_cycle, rtol=0, atol=1e-12):
                raise AssertionError(f"Mean-cycle metric differs in {view}/{name}")
            # Confirm redundant class summary agrees with the independently recomputed overall metric.
            cls = ev["per_class_metrics"][name]
            if not np.isclose(cls["class_0_f1"], by_cycle and v1.metric(yy, p)["f1_negative"], rtol=0, atol=1e-12):
                raise AssertionError(f"Class-0 F1 summary differs in {view}/{name}")
            if not np.isclose(cls["class_1_f1"], v1.metric(yy, p)["f1_positive"], rtol=0, atol=1e-12):
                raise AssertionError(f"Class-1 F1 summary differs in {view}/{name}")
            checks += len(by_cycle) + 3
        for feature in ("entry_friction", "text_load"):
            sub = ev["subgroups"][feature]
            bins = np.full(len(rows), -1, dtype=int)
            for end in sorted(np.unique(cutoff)):
                take = cutoff == end
                edges = v1.subgroup_edges(frame, cycle, int(end), feature)
                recorded = np.asarray(sub["fit_only_quartile_edges_by_cutoff"][str(int(end))], dtype=float)
                if not np.array_equal(edges, recorded):
                    raise AssertionError(f"Raw fit-only cutpoints differ in {view}/{feature}/{end}")
                vals_local = pd.to_numeric(frame.iloc[rows[take]][feature], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(vals_local)
                pos = np.flatnonzero(take)
                bins[pos[valid]] = np.digitize(vals_local[valid], edges, right=True)
            for idx, label in enumerate(("Q1_low", "Q2", "Q3", "Q4_high")):
                mask = bins == idx
                group = sub["groups"][label]
                if group["n"] != int(mask.sum()):
                    raise AssertionError(f"Subgroup denominator differs: {view}/{feature}/{label}")
                if mask.any():
                    for name, p in vectors.items():
                        v1.compare(group["models"][name], v1.metric(yy[mask], p[mask]), f"{view}.{feature}.{label}.{name}")
                        checks += 1
            mask = bins < 0; group = sub["groups"]["missing_feature"]
            if group["n"] != int(mask.sum()):
                raise AssertionError(f"Missing-feature denominator differs: {view}/{feature}")
            if mask.any():
                for name, p in vectors.items():
                    v1.compare(group["models"][name], v1.metric(yy[mask], p[mask]), f"{view}.{feature}.missing.{name}")
                    checks += 1
        for name in ("residual_A", "sparse25"):
            p0, p1 = vectors["champion"], vectors[name]
            e0 = (p0 >= .5) != yy; e1 = (p1 >= .5) != yy
            observed = {"champion_errors_fixed": int(np.sum(e0 & ~e1)),
                        "champion_correct_broken": int(np.sum(~e0 & e1)),
                        "candidate_errors": int(e1.sum()), "champion_errors": int(e0.sum()),
                        "prediction_disagreements": int(np.sum((p0 >= .5) != (p1 >= .5)))}
            v1.compare(ev["paired_errors"][name], observed, f"{view}.paired.{name}")
            delta = {k: ev["models"][name]["all"][k] - ev["models"]["champion"]["all"][k]
                     for k in ("macro_f1", "average_precision", "brier")}
            v1.compare(ev["delta_vs_champion"][name], delta, f"{view}.delta.{name}")
            checks += 2
        pa, ps = vectors["residual_A"], vectors["sparse25"]
        ea = (pa >= .5) != yy; es = (ps >= .5) != yy
        comp = {"probability_mae": float(np.mean(np.abs(pa - ps))),
                "probability_correlation": float(np.corrcoef(pa, ps)[0, 1]),
                "prediction_disagreement_n": int(np.sum((pa >= .5) != (ps >= .5))),
                "residual_only_correct_sparse_wrong": int(np.sum(~ea & es)),
                "sparse_only_correct_residual_wrong": int(np.sum(ea & ~es))}
        v1.compare(ev["residual_vs_sparse"], comp, f"{view}.residual_vs_sparse")
        checks += 1
    return checks


def verify() -> dict:
    vf = check_verification_freeze()
    fit_freeze = run.check_freeze()
    report = run.read_json(run.REPORT_DIR / "fullsix.json")
    if report["freeze_signature"] != fit_freeze["signature"]:
        raise AssertionError("Result references another frozen fit identity")
    frame, y, cycle, missing, _ = run.load_train()
    records = {}; replay_fits = 0
    for end in run.ENDS:
        if end in run.NEW_ENDS:
            records[end] = v1.replay_new(end, frame, y, cycle, missing, report)
            replay_fits += 1
        else:
            records[end] = v1.reuse_pilot(end, frame, y, cycle, missing)
    checks = verify_metrics(frame, y, cycle, records, report)
    if replay_fits != 4 or report["new_fits"] != 4:
        raise AssertionError("Expected four distinct persisted cutoff fits and four deterministic replay fits")
    accounting = report.get("execution_accounting", {})
    if accounting.get("successful_persisted_designated_cutoff_fits") != 4 or accounting.get("unsaved_execution_attempts") != 1:
        raise AssertionError("Candidate fit invocation accounting is incomplete")
    result = {
        "status": "PASS",
        "verification_version": 2,
        "verification_freeze_signature": vf["signature"],
        "fit_freeze_signature": fit_freeze["signature"],
        "new_successful_persisted_head_fits": 4,
        "unsaved_execution_attempts": 1,
        "deterministic_verification_replay_fits": replay_fits,
        "replay_fit_cutoffs": run.NEW_ENDS,
        "reused_cutoffs": run.REUSED_ENDS,
        "source_and_input_hashes_verified": True,
        "oof_rows_and_four_cycle_blocks_verified": True,
        "booster_state_design_raw_margin_and_probability_bitwise_verified": True,
        "champion_and_sparse25_row_exact_alignment_verified": True,
        "original_string_record_ids_hash_verified_with_repository_helper": True,
        "tied_quantile_edges_and_empty_bins_verified": True,
        "missing_stream_route_exact": True,
        "independent_metric_subgroup_and_paired_error_checks": checks,
        "primary_n": report["evaluation"]["primary_six_window_pooled"]["n"],
        "h12_n": report["evaluation"]["H12_fixed_origin_pooled"]["n"],
        "primary_macro_f1": {n: report["evaluation"]["primary_six_window_pooled"]["models"][n]["all"]["macro_f1"] for n in ("champion", "residual_A", "sparse25")},
        "primary_delta_vs_champion": report["evaluation"]["primary_six_window_pooled"]["delta_vs_champion"],
        "h12_macro_f1": {n: report["evaluation"]["H12_fixed_origin_pooled"]["models"][n]["all"]["macro_f1"] for n in ("champion", "residual_A", "sparse25")},
        "h12_delta_vs_champion": report["evaluation"]["H12_fixed_origin_pooled"]["delta_vs_champion"],
        "test_read": False, "kaggle_access": False,
        "outer_labels_used_for_fit_or_selection": False,
    }
    run.write_json_new(VERIFY_RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.run:
        parser.error("Choose exactly one of --freeze-only or --run")
    if args.freeze_only:
        rec = freeze_verification()
        print(json.dumps({"status": rec["status"], "signature": rec["signature"],
                          "metric_schema": rec["identity"]["metric_schema_dry_run"]}, indent=2))
    else:
        rec = verify()
        print(json.dumps({k: rec[k] for k in ("status", "new_successful_persisted_head_fits",
                          "unsaved_execution_attempts", "deterministic_verification_replay_fits",
                          "primary_delta_vs_champion", "h12_delta_vs_champion",
                          "independent_metric_subgroup_and_paired_error_checks")}, indent=2))


if __name__ == "__main__":
    main()
