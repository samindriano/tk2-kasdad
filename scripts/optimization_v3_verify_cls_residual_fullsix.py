"""Independent deterministic replay and row-exact metric verification for Head A fullsix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_residual_fullsix as run
from scripts import optimization_v3_cls_residual as pilot
from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_evaluation import array_digest, digest_json
from src.data.audit import sha256 as file_sha256

VERIFY_FREEZE = run.REPORT_DIR / "verification_freeze.json"
VERIFY_RESULT = run.REPORT_DIR / "verification.json"


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Invalid independent metric vectors")
    pred = p >= 0.5
    tp = int(np.count_nonzero(pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    p1 = tp / (tp + fp) if tp + fp else 0.0
    r1 = tp / (tp + fn) if tp + fn else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0
    r0 = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return {"n": int(y.size), "macro_f1": float((f0 + f1) / 2),
            "f1_negative": float(f0), "f1_positive": float(f1),
            "positive_prevalence": float(np.mean(y)),
            "positive_prediction_rate": float(np.mean(pred)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "error_count": fp + fn,
            "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
            "brier": float(np.mean((p - y) ** 2)), "threshold": 0.5}


def compare(expected: dict, actual: dict, path: str = "metric") -> None:
    if set(expected) != set(actual):
        raise AssertionError(f"{path} schema mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    for k, v in expected.items():
        a = actual[k]
        if isinstance(v, dict):
            compare(v, a, f"{path}.{k}")
        elif isinstance(v, float):
            if not np.isclose(a, v, rtol=0, atol=1e-12):
                raise AssertionError(f"{path}.{k} differs: {a} != {v}")
        elif v != a:
            raise AssertionError(f"{path}.{k} differs: {a} != {v}")


def metric_schema_dry_run() -> dict:
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    p = np.array([0.2, 0.8, 0.7, 0.3])
    sample = metric(y, p)
    compare(sample, dict(sample), "dry")
    required = {"n", "macro_f1", "f1_negative", "f1_positive", "positive_prevalence",
                "positive_prediction_rate", "tp", "tn", "fp", "fn", "error_count",
                "average_precision", "brier", "threshold"}
    if set(sample) != required or not np.isfinite(sample["brier"]):
        raise AssertionError("Independent metric schema smoke test failed")
    result_path = run.REPORT_DIR / "fullsix.json"
    report = run.read_json(result_path)
    present = report["evaluation"]["W1"]["models"]["residual_A"]["all"]
    if set(present) != required:
        raise AssertionError("Fullsix report metric schema differs from independent verifier")
    compare(sample, {k: v for k, v in sample.items()}, "actual")
    return {"status": "PASS", "metric_fields": sorted(required), "brier": sample["brier"]}


def verification_identity() -> dict:
    fit_freeze = run.read_json(run.FREEZE)
    if fit_freeze.get("status") != "FROZEN_BEFORE_FOUR_HEAD_A_FITS":
        raise AssertionError("Fullsix fit freeze status is not canonical")
    current = run.identity()
    if current != fit_freeze["identity"] or digest_json(current) != fit_freeze["signature"]:
        raise AssertionError("Fullsix pre-fit source/data identity differs from freeze")
    report_path = run.REPORT_DIR / "fullsix.json"
    if not report_path.is_file():
        raise FileNotFoundError("Fullsix result is missing")
    report = run.read_json(report_path)
    if report.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION" or report.get("new_fits") != 4:
        raise AssertionError("Fullsix output status or candidate-fit count invalid")
    artifacts = {}
    for end in run.NEW_ENDS:
        stem = f"fit{end}_{run.HEAD_NAME}"
        for suffix in ("npz", "txt", "json"):
            path = run.RUN_DIR / f"{stem}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fullsix output artifact {path}")
            artifacts[path.resolve().relative_to(ROOT).as_posix()] = file_sha256(path)
    schema = metric_schema_dry_run()
    return {"fit_freeze_signature": fit_freeze["signature"],
            "fit_identity_sha256": digest_json(current),
            "fullsix_report_sha256": file_sha256(report_path),
            "fullsix_artifact_sha256": artifacts,
            "verifier_source_sha256": file_sha256(Path(__file__).resolve()),
            "upstream_sparse_verification_sha256": file_sha256(run.SPARSE_REPORT / "verification.json"),
            "upstream_pilot_verification_sha256": file_sha256(run.PILOT_REPORT / "verification_v3.json"),
            "metric_schema_dry_run": schema,
            "test_read": False, "kaggle_access": False}


def freeze_verification() -> dict:
    identity = verification_identity()
    sig = digest_json(identity)
    if VERIFY_FREEZE.exists():
        old = run.read_json(VERIFY_FREEZE)
        if old.get("identity") != identity or old.get("signature") != sig:
            raise RuntimeError("Existing verification freeze differs; preserve it and version the verifier")
        return old
    receipt = {"status": "FROZEN_BEFORE_DETERMINISTIC_FOUR_HEAD_REPLAY",
               "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "identity": identity, "signature": sig,
               "source_boundary": "Deterministic replay of four frozen head fits and independent row/metric reconstruction; no Test or Kaggle access."}
    run.write_json_new(VERIFY_FREEZE, receipt)
    return receipt


def check_verification_freeze() -> dict:
    if not VERIFY_FREEZE.is_file():
        raise RuntimeError("Run --freeze-only and inspect metric schema before deterministic replay")
    receipt = run.read_json(VERIFY_FREEZE)
    current = verification_identity()
    if current != receipt.get("identity") or digest_json(current) != receipt.get("signature"):
        raise RuntimeError("Fullsix report, outputs, verifier, or frozen inputs changed after replay freeze")
    return receipt


def replay_new(end: int, frame: pd.DataFrame, y: np.ndarray,
               cycle: np.ndarray, missing: np.ndarray,
               report: dict) -> dict:
    fit_data = pilot.prepare_oof(frame, y, cycle, missing, end, run.HEAD_NAME)
    meta = report["fit_metadata"][str(end)]
    pred_path = run.RUN_DIR / f"fit{end}_{run.HEAD_NAME}.npz"
    model_path = run.RUN_DIR / f"fit{end}_{run.HEAD_NAME}.txt"
    sidecar = run.RUN_DIR / f"fit{end}_{run.HEAD_NAME}.json"
    if file_sha256(pred_path) != meta["prediction_sha256"] or file_sha256(model_path) != meta["model_sha256"]:
        raise AssertionError(f"Output cache/model hash mismatch at cutoff {end}")
    side = run.read_json(sidecar)
    if side["prediction_sha256"] != file_sha256(pred_path) or side["freeze_signature"] != report["freeze_signature"]:
        raise AssertionError(f"Sidecar cache/freeze binding mismatch at cutoff {end}")
    with np.load(pred_path, allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    outer = run.control_window(frame, cycle, missing, end)
    rows = outer["rows"]
    for key, value in (("outer_rows", rows), ("missing_mask", outer["missing"].astype(np.uint8)),
                       ("base_probability", outer["base"]), ("cb_probability", outer["cb"]),
                       ("champion_probability", outer["champion"]), ("sparse25_probability", outer["sparse25"]),
                       ("oof_present_rows", fit_data["present_rows"]), ("oof_scores", fit_data["scores"]),
                       ("oof_target", fit_data["y"])):
        if not np.array_equal(saved[key], value):
            raise AssertionError(f"Saved input/vector differs at cutoff {end}: {key}")
    if fit_data["raw_input_sha256"] != meta["oof_raw_input_sha256"]:
        raise AssertionError(f"Raw OOF hash differs at cutoff {end}")
    head = OOFResidualHead("classification", run.SCORE_NAMES)
    with threadpool_limits(limits=2):
        head.fit(fit_data["X"], fit_data["scores"], fit_data["y"])
    state = head.audit_state()
    for field in ("fit_data_sha256", "fit_scores_sha256", "target_sha256", "booster_sha256", "rows", "preprocessor"):
        if state.get(field) != meta["state"].get(field):
            raise AssertionError(f"Deterministic head replay differs at {end}/{field}")
    model_text = head.model.booster_.model_to_string()
    if model_text != model_path.read_text(encoding="utf-8"):
        raise AssertionError(f"Serialized booster text differs at cutoff {end}")
    present = ~outer["missing"]
    X = frame.iloc[rows[present]][list(pilot.EXPANDED_FEATURES)].copy()
    scores = outer["base"][present, None]
    design, _, _ = head._design(X, scores, False)
    raw = head.model.booster_.predict(design, raw_score=True, num_threads=2)
    expected_raw = saved["raw_correction_present"]
    if not np.array_equal(raw, expected_raw) or not np.array_equal(design.to_numpy(dtype=float), saved["outer_design_present"]):
        raise AssertionError(f"Outer design or raw-margin replay differs at cutoff {end}")
    expected = outer["champion"].copy()
    expected[present] = expit(logit(np.clip(outer["base"][present], 1e-6, 1 - 1e-6)) + 0.5 * raw)
    if not np.array_equal(expected, saved["candidate_probability"]):
        raise AssertionError(f"Raw-margin composition differs at cutoff {end}")
    if not np.array_equal(expected[outer["missing"]], outer["champion"][outer["missing"]]):
        raise AssertionError(f"Missing route changed at cutoff {end}")
    return {"rows": rows, "missing": outer["missing"], "champion": outer["champion"],
            "residual_A": expected, "sparse25": outer["sparse25"],
            "new_fit_replay": True, "oof_rows_present": int(len(fit_data["present_rows"]))}


def reuse_pilot(end: int, frame: pd.DataFrame, y: np.ndarray,
                cycle: np.ndarray, missing: np.ndarray) -> dict:
    rec = run.reuse_pilot(end)
    # Independently bind row order and route mask to official Train.
    if not np.array_equal(rec["missing"], missing[rec["rows"]]):
        raise AssertionError(f"Reused pilot missing mask differs at cutoff {end}")
    if not np.array_equal(rec["candidate"][rec["missing"]], rec["champion"][rec["missing"]]):
        raise AssertionError(f"Reused pilot changed missing route at cutoff {end}")
    return {"rows": rec["rows"], "missing": rec["missing"],
            "champion": rec["champion"], "residual_A": rec["candidate"],
            "sparse25": rec["sparse25"], "new_fit_replay": False}


def expected_slices() -> dict[str, list[tuple[int, int, int]]]:
    slices = {name: [(end, lo, hi)] for name, (end, lo, hi) in run.PRIMARY_WINDOWS.items()}
    slices["primary_six_window_pooled"] = [tuple(v) for v in run.PRIMARY_WINDOWS.values()]
    slices["H12_E"] = [(55, 56, 67)]
    slices["H12_L"] = [(67, 68, 79)]
    slices["H12_fixed_origin_pooled"] = slices["H12_E"] + slices["H12_L"]
    return slices


def subgroup_edges(frame: pd.DataFrame, cycle: np.ndarray, cutoff: int, feature: str) -> np.ndarray:
    vals = pd.to_numeric(frame.loc[(cycle >= 47) & (cycle <= cutoff), feature], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.quantile(vals, [0.25, 0.5, 0.75]).astype(float)


def verify_metrics(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                   records: dict[int, dict], report: dict) -> int:
    checks = 0
    slices = expected_slices()
    if set(report["evaluation"]) != set(slices):
        raise AssertionError("Evaluation view list differs from frozen protocol")
    for view, parts in slices.items():
        rows_l, missing_l, cutoff_l = [], [], []
        v = {name: [] for name in ("champion", "residual_A", "sparse25")}
        for end, lo, hi in parts:
            rec = records[end]
            keep = (cycle[rec["rows"]] >= lo) & (cycle[rec["rows"]] <= hi)
            rows = rec["rows"][keep]
            rows_l.append(rows); missing_l.append(rec["missing"][keep])
            cutoff_l.append(np.full(len(rows), end, dtype=np.int16))
            for name in v:
                v[name].append(rec[name][keep])
        rows = np.concatenate(rows_l); m = np.concatenate(missing_l).astype(bool)
        cuts = np.concatenate(cutoff_l)
        if len(rows) != len(np.unique(rows)):
            raise AssertionError(f"Duplicate rows in {view}")
        vectors = {name: np.concatenate(parts_) for name, parts_ in v.items()}
        ev = report["evaluation"][view]
        if ev["n"] != len(rows) or ev["row_indices_sha256"] != array_digest(rows):
            raise AssertionError(f"Row denominator/identity mismatch in {view}")
        row_ids = pd.to_numeric(frame.iloc[rows].record_id, errors="raise").to_numpy()
        if ev["row_ids_sha256"] != array_digest(row_ids):
            raise AssertionError(f"record_id hash differs in {view}")
        yy = y[rows]; cc = cycle[rows]
        for name, p in vectors.items():
            expected_parts = {"all": np.ones(len(rows), dtype=bool), "stream_present": ~m, "stream_missing": m}
            for part, mask in expected_parts.items():
                if mask.any():
                    compare(ev["models"][name][part], metric(yy[mask], p[mask]), f"{view}.{name}.{part}")
                    checks += 1
            by_cycle = {str(int(c)): metric(yy[cc == c], p[cc == c]) for c in sorted(np.unique(cc))}
            compare(ev["models"][name]["by_cycle"], by_cycle, f"{view}.{name}.by_cycle")
            mean_cycle = float(np.mean([v_["macro_f1"] for v_ in by_cycle.values()]))
            if not np.isclose(ev["models"][name]["mean_cycle_macro_f1"], mean_cycle, rtol=0, atol=1e-12):
                raise AssertionError(f"Mean per-cycle F1 differs in {view}/{name}")
            checks += len(by_cycle) + 1
        for feature in ("entry_friction", "text_load"):
            sub = ev["subgroups"][feature]
            bins = np.full(len(rows), -1, dtype=int)
            for end in sorted(np.unique(cuts)):
                take = cuts == end
                edges = subgroup_edges(frame, cycle, int(end), feature)
                recorded = np.asarray(sub["fit_only_quartile_edges_by_cutoff"][str(int(end))], dtype=float)
                if not np.array_equal(edges, recorded):
                    raise AssertionError(f"Fit-only subgroup edges mismatch for {view}/{feature}/{end}")
                vals = pd.to_numeric(frame.iloc[rows[take]][feature], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(vals)
                loc = np.flatnonzero(take)
                bins[loc[valid]] = np.digitize(vals[valid], edges, right=True)
            for idx, label in enumerate(("Q1_low", "Q2", "Q3", "Q4_high")):
                mask = bins == idx
                got = sub["groups"][label]
                if got["n"] != int(mask.sum()):
                    raise AssertionError(f"{view}/{feature}/{label} denominator differs")
                if mask.any():
                    for name, p in vectors.items():
                        compare(got["models"][name], metric(yy[mask], p[mask]), f"{view}.{feature}.{label}.{name}")
                        checks += 1
            mask = bins < 0
            got = sub["groups"]["missing_feature"]
            if got["n"] != int(mask.sum()):
                raise AssertionError(f"{view}/{feature}/missing denominator differs")
            if mask.any():
                for name, p in vectors.items():
                    compare(got["models"][name], metric(yy[mask], p[mask]), f"{view}.{feature}.missing.{name}")
                    checks += 1
        for name in ("residual_A", "sparse25"):
            p0 = vectors["champion"]; p1 = vectors[name]
            e0 = (p0 >= .5) != yy; e1 = (p1 >= .5) != yy
            observed = {"champion_errors_fixed": int(np.sum(e0 & ~e1)),
                        "champion_correct_broken": int(np.sum(~e0 & e1)),
                        "candidate_errors": int(e1.sum()), "champion_errors": int(e0.sum()),
                        "prediction_disagreements": int(np.sum((p0 >= .5) != (p1 >= .5)))}
            compare(ev["paired_errors"][name], observed, f"{view}.paired.{name}"); checks += 1
            delta = {k: ev["models"][name]["all"][k] - ev["models"]["champion"]["all"][k]
                     for k in ("macro_f1", "average_precision", "brier")}
            compare(ev["delta_vs_champion"][name], delta, f"{view}.delta.{name}"); checks += 1
        pa = vectors["residual_A"]; ps = vectors["sparse25"]
        err_a = (pa >= .5) != yy; err_s = (ps >= .5) != yy
        comp = {"probability_mae": float(np.mean(np.abs(pa - ps))),
                "probability_correlation": float(np.corrcoef(pa, ps)[0, 1]),
                "prediction_disagreement_n": int(np.sum((pa >= .5) != (ps >= .5))),
                "residual_only_correct_sparse_wrong": int(np.sum(~err_a & err_s)),
                "sparse_only_correct_residual_wrong": int(np.sum(err_a & ~err_s))}
        compare(ev["residual_vs_sparse"], comp, f"{view}.residual_vs_sparse"); checks += 1
    return checks


def verify() -> dict:
    vfreeze = check_verification_freeze()
    fit_freeze = run.check_freeze()
    report = run.read_json(run.REPORT_DIR / "fullsix.json")
    if report["freeze_signature"] != fit_freeze["signature"]:
        raise AssertionError("Output report points to a different fit freeze")
    frame, y, cycle, missing, _ = run.load_train()
    records = {}
    replay_fits = 0
    for end in run.ENDS:
        if end in run.NEW_ENDS:
            records[end] = replay_new(end, frame, y, cycle, missing, report)
            replay_fits += 1
        else:
            records[end] = reuse_pilot(end, frame, y, cycle, missing)
    checks = verify_metrics(frame, y, cycle, records, report)
    if report.get("new_fits") != 4 or replay_fits != 4:
        raise AssertionError("Expected exactly four candidate fits and four deterministic replay fits")
    result = {
        "status": "PASS",
        "verification_freeze_signature": vfreeze["signature"],
        "fit_freeze_signature": fit_freeze["signature"],
        "new_candidate_fits": 4,
        "deterministic_verification_replay_fits": replay_fits,
        "replay_fit_cutoffs": run.NEW_ENDS,
        "reused_cutoffs": run.REUSED_ENDS,
        "source_and_input_hashes_verified": True,
        "oof_rows_and_four_cycle_blocks_verified": True,
        "booster_state_design_raw_margin_and_probability_bitwise_verified": True,
        "champion_and_sparse25_row_exact_alignment_verified": True,
        "missing_stream_route_exact": True,
        "independent_metric_and_subgroup_field_checks": checks,
        "primary_n": report["evaluation"]["primary_six_window_pooled"]["n"],
        "h12_n": report["evaluation"]["H12_fixed_origin_pooled"]["n"],
        "primary_macro_f1": {name: report["evaluation"]["primary_six_window_pooled"]["models"][name]["all"]["macro_f1"]
                              for name in ("champion", "residual_A", "sparse25")},
        "primary_delta_vs_champion": report["evaluation"]["primary_six_window_pooled"]["delta_vs_champion"],
        "h12_macro_f1": {name: report["evaluation"]["H12_fixed_origin_pooled"]["models"][name]["all"]["macro_f1"]
                          for name in ("champion", "residual_A", "sparse25")},
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
        receipt = freeze_verification()
        print(json.dumps({"status": receipt["status"], "signature": receipt["signature"],
                          "metric_schema": receipt["identity"]["metric_schema_dry_run"]}, indent=2))
    else:
        result = verify()
        print(json.dumps({k: result[k] for k in ("status", "new_candidate_fits",
                          "deterministic_verification_replay_fits", "primary_delta_vs_champion",
                          "h12_delta_vs_champion", "independent_metric_and_subgroup_field_checks")}, indent=2))


if __name__ == "__main__":
    main()
