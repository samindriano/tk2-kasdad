"""Independently verify the autonomous SVD128 paired regression report.

This checker loads only official regression Train, its frozen Stage 5 control
predictions, and this lane's prediction caches. It performs no model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.regression_round2_evaluation import load_protocol, positions_digest
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT / "experiments/autonomous_eda/regression"
REPORT = PROJECT_ROOT / "reports/autonomous_eda/regression/results.json"
RUN = PROJECT_ROOT / "experiments/runs/autonomous_reg_v1"
STAGE5_RUN = PROJECT_ROOT / "experiments/runs/regression_research_v1"
AMENDMENT_PATH = ROOT / "postfit_reporting_amendment_v4.json"
PRIOR_AMENDMENT_PATH = ROOT / "postfit_reporting_amendment_v3.json"
EARLIER_AMENDMENT_PATH = ROOT / "postfit_reporting_amendment_v2.json"
INNER_PROTOCOL_PATH = ROOT / "inner_refinement_protocol.json"
INNER_FREEZE_PATH = ROOT / "inner_refinement_freeze.json"
INNER_REPORT_PATH = PROJECT_ROOT / "reports/autonomous_eda/regression/inner_refinement.json"
STAGE5_REPORT = PROJECT_ROOT / "reports/regression_research/stage5_lgb_text.json"
STAGE5_FREEZE = PROJECT_ROOT / "experiments/regression_research/stage5_lgb_text_freeze.json"
ROUND1_FREEZE = PROJECT_ROOT / "experiments/round1/freeze.json"
CODE_PATHS = [
    "src/models/autonomous_regression.py",
    "scripts/research_autonomous_regression.py",
    "scripts/verify_autonomous_regression.py",
]
REFERENCE_PATHS = [
    "experiments/regression_research/stage5_lgb_text_freeze.json",
    "reports/regression_research/stage5_lgb_text.json",
    "src/models/regression_research_lgb_text.py",
    "scripts/research_regression_lgb_text.py",
    "experiments/regression_research/stage4_refinement_freeze.json",
    "src/models/regression_research_refinement.py",
]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manual_rank_map(raw: np.ndarray, cycles: np.ndarray, weight: float) -> np.ndarray:
    out = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        out[mask] = (1.0 - weight) * raw[mask] + weight * percentile
    return np.clip(out, 0.0, 100.0)


def _expected_rows(frame: pd.DataFrame, low: int, high: int) -> np.ndarray:
    return np.flatnonzero(frame.cycle_code.between(low, high).to_numpy())


def _basic(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "signed_error": float(np.mean(error)),
        "p95": float(np.quantile(np.abs(error), .95)),
    }


def _assert_close(found: float, expected: float, label: str, atol: float = 1e-10) -> None:
    if not np.isfinite(found) or not np.isclose(found, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: computed {found} but report has {expected}")


def _identity_without_reporting_code(identity: dict) -> dict:
    value = json.loads(json.dumps(identity))
    for path in ["scripts/research_autonomous_regression.py", "scripts/verify_autonomous_regression.py"]:
        value["autonomous_code_sha256"].pop(path, None)
    return value


def _verify_postfit_amendment(receipt: dict) -> dict:
    amendment = _load(AMENDMENT_PATH)
    signed = dict(amendment)
    signature = signed.pop("signature")
    if signature != digest_json(signed):
        raise AssertionError("Post-fit reporting amendment signature does not recompute")
    if amendment["original_freeze_signature"] != receipt["signature"]:
        raise AssertionError("Amendment points to a different pre-fit freeze")
    if amendment["original_protocol_sha256"] != digest_json(_load(ROOT / "protocol.json")):
        raise AssertionError("Amendment protocol differs from the frozen protocol")
    prior_sha = _sha_file(PRIOR_AMENDMENT_PATH) if PRIOR_AMENDMENT_PATH.is_file() else None
    if amendment.get("supersedes_postfit_reporting_amendment_sha256") != prior_sha:
        raise AssertionError("Final amendment does not chain to the earlier report-only amendment")
    prior = _load(PRIOR_AMENDMENT_PATH)
    prior_signed = dict(prior)
    prior_signature = prior_signed.pop("signature")
    if prior_signature != digest_json(prior_signed):
        raise AssertionError("Prior post-fit reporting amendment signature does not recompute")
    if prior.get("supersedes_postfit_reporting_amendment_sha256") != _sha_file(EARLIER_AMENDMENT_PATH):
        raise AssertionError("Earlier post-fit amendments do not form a continuous chain")
    original_codes = receipt["identity"]["autonomous_code_sha256"]
    changed = {"scripts/research_autonomous_regression.py", "scripts/verify_autonomous_regression.py"}
    if set(amendment["pre_fit_reporting_code_sha256"]) != changed:
        raise AssertionError("Amendment does not identify exactly the two reporting scripts")
    for path in CODE_PATHS:
        current = code_digest(PROJECT_ROOT / path)
        if path in changed:
            if amendment["pre_fit_reporting_code_sha256"][path] != original_codes[path]:
                raise AssertionError(f"Amendment pre-fit code hash mismatch: {path}")
            if amendment["post_fit_reporting_code_sha256"][path] != current:
                raise AssertionError(f"Amendment post-fit code hash mismatch: {path}")
        elif original_codes.get(path) != current:
            raise AssertionError(f"Frozen model code changed: {path}")
    if amendment["unchanged_model_code_sha256"] != original_codes["src/models/autonomous_regression.py"]:
        raise AssertionError("Amendment model source reference differs")
    if amendment["stable_identity_sha256"] != digest_json(_identity_without_reporting_code(receipt["identity"])):
        raise AssertionError("Amendment stable pre-fit identity differs")
    if amendment["fit_budget"] != {
        "original_svd128_outer_fits": 6, "new_fits": 0,
        "full_train_fits": 0, "test_inference": False,
    }:
        raise AssertionError("Amendment fit budget permits an unapproved fit")
    if len(amendment["candidate_cache_manifest"]) != 6:
        raise AssertionError("Amendment does not preserve exactly six outer caches")
    for end, item in amendment["candidate_cache_manifest"].items():
        files = sorted(RUN.glob(f"svd128__{item['param']}__fit{end}__to*.npz"))
        if len(files) != 1:
            raise AssertionError(f"Amended fit cache missing or ambiguous: fit_end={end}")
        npz_path = files[0]
        meta_path = npz_path.with_suffix(".json")
        if _sha_file(npz_path) != item["npz_file_sha256"] or _sha_file(meta_path) != item["metadata_file_sha256"]:
            raise AssertionError(f"Amended fit cache file hash mismatch: fit_end={end}")
        with np.load(npz_path, allow_pickle=False) as z:
            fit_rows = z["fit_rows"].copy()
            rows = z["validation_rows"].copy()
            raw = z["raw_prediction"].copy()
            mapped = z["mapped_prediction"].copy()
            explained = z["explained_variance_ratio"].copy()
        meta = _load(meta_path)
        expected_key = digest_json({
            "freeze_signature": receipt["signature"],
            "params_id": item["param"],
            "fit_positions_sha256": positions_digest(fit_rows),
            "validation_positions_sha256": positions_digest(rows),
        })
        if meta.get("key") != expected_key or item["prediction_cache_key"] != expected_key:
            raise AssertionError(f"Amended cache key mismatch: fit_end={end}")
        if meta.get("state_sha256") != digest_json(meta.get("state")):
            raise AssertionError(f"Amended model state hash mismatch: fit_end={end}")
        if array_digest(raw) != item["raw_prediction_sha256"] or array_digest(raw) != meta["raw_prediction_sha256"]:
            raise AssertionError(f"Amended raw prediction hash mismatch: fit_end={end}")
        if array_digest(mapped) != item["mapped_prediction_sha256"] or array_digest(mapped) != meta["mapped_prediction_sha256"]:
            raise AssertionError(f"Amended mapped prediction hash mismatch: fit_end={end}")
        if array_digest(explained) != item["explained_variance_ratio_sha256"]:
            raise AssertionError(f"Amended spectrum hash mismatch: fit_end={end}")
    return amendment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-inner", action="store_true", help="Also independently verify the optional SVD128 inner-refinement stage.")
    args = parser.parse_args()
    protocol = _load(ROOT / "protocol.json")
    receipt = _load(ROOT / "freeze.json")
    result = _load(REPORT)
    train_path = task_files("regression")["train"]
    if train_path is None:
        raise FileNotFoundError("Official regression Train.csv was not found")
    train_sha = _sha_file(train_path)
    round1 = _load(ROUND1_FREEZE)
    if train_sha != round1["identity"]["input_sha256"]["train"]:
        raise AssertionError("Official Train does not match the frozen Round 1 input")
    if receipt["identity"]["train_sha256"] != train_sha:
        raise AssertionError("Run receipt Train hash differs")
    if receipt["identity"]["protocol_sha256"] != digest_json(protocol):
        raise AssertionError("Protocol hash differs from the frozen receipt")
    if receipt["signature"] != digest_json(receipt["identity"]):
        raise AssertionError("Freeze receipt signature does not recompute")
    amendment = _verify_postfit_amendment(receipt) if AMENDMENT_PATH.is_file() else None
    changed_reporting = {"scripts/research_autonomous_regression.py", "scripts/verify_autonomous_regression.py"} if amendment else set()
    for path in CODE_PATHS:
        expected = (
            amendment["post_fit_reporting_code_sha256"].get(path)
            if path in changed_reporting else receipt["identity"]["autonomous_code_sha256"].get(path)
        )
        if expected != code_digest(PROJECT_ROOT / path):
            raise AssertionError(f"Frozen or amended source changed: {path}")
    for path in REFERENCE_PATHS:
        if receipt["identity"]["reference_sha256"].get(path) != code_digest(PROJECT_ROOT / path):
            raise AssertionError(f"Historical reference changed: {path}")
    if result["freeze_signature"] != receipt["signature"]:
        raise AssertionError("Report is not tied to the active freeze")
    expected_amendment_sha = _sha_file(AMENDMENT_PATH) if amendment else None
    if result.get("postfit_reporting_amendment_sha256") != expected_amendment_sha:
        raise AssertionError("Report does not identify the post-fit amendment")
    if result["fit_budget"] != {
        "outer_svd128_fits": 6, "inner_svd128_fits": 0,
        "full_train_fits": 0, "test_inference": False,
    }:
        raise AssertionError("Fit budget/report boundary does not match the frozen scope")

    frame = pd.read_csv(train_path)
    stage5 = _load(STAGE5_REPORT)
    stage5_freeze = _load(STAGE5_FREEZE)
    if stage5["freeze_signature"] != stage5_freeze["signature"]:
        raise AssertionError("Stage 5 control report does not match its own freeze")
    selected = {int(item["fit_end"]): item["selected"] for item in stage5["inner"]}
    primary = load_protocol()["primary_windows"]
    secondary = load_protocol()["secondary_windows"]

    control_vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    candidate_vectors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for end in protocol["comparison"]["outer_fit_ends"]:
        end = int(end)
        choice = selected[end]
        frozen_choice = protocol["comparison"]["fixed_param_rank_weight_by_fit_end"][str(end)]
        if choice["param"] != frozen_choice["param"] or not np.isclose(choice["rank_weight"], frozen_choice["rank_weight"]):
            raise AssertionError(f"Stage 5 selected policy changed at fit_end={end}")
        max_cycle = max(
            int(window["validation"][1])
            for window in [*primary, *secondary]
            if int(window["fit"][1]) == end
        )
        rows_expected = _expected_rows(frame, end + 1, max_cycle)
        stem = f"lgbtext__{choice['param']}__fit{end}__to{max_cycle}"
        with np.load(STAGE5_RUN / f"{stem}.npz", allow_pickle=False) as z:
            old_fit, old_rows, raw = z["fit_rows"].copy(), z["validation_rows"].copy(), z["prediction"].copy()
        meta = _load(STAGE5_RUN / f"{stem}.json")
        if meta["state_sha256"] != digest_json(meta["state"]) or array_digest(raw) != meta["prediction_sha256"]:
            raise AssertionError(f"Stage 5 raw prediction audit failed at fit_end={end}")
        if not np.array_equal(old_rows, rows_expected):
            raise AssertionError(f"Stage 5 validation rows differ at fit_end={end}")
        fit_expected = _expected_rows(frame, 47, end)
        if not np.array_equal(old_fit, fit_expected):
            raise AssertionError(f"Stage 5 fit rows differ at fit_end={end}")
        old_mapped = _manual_rank_map(raw, frame.iloc[old_rows].cycle_code.to_numpy(), float(choice["rank_weight"]))
        with np.load(STAGE5_RUN / f"eval_lgbtext__fit{end}__inner.npz", allow_pickle=False) as z:
            saved_rows, saved_mapped = z["validation_rows"].copy(), z["prediction"].copy()
        if not np.array_equal(saved_rows, old_rows) or not np.array_equal(saved_mapped, old_mapped):
            raise AssertionError(f"Stage 5 mapped control vector mismatch at fit_end={end}")
        control_vectors[end] = (old_rows, old_mapped)

        candidate_path = RUN / f"svd128__{choice['param']}__fit{end}__to{max_cycle}.npz"
        candidate_meta = _load(RUN / f"svd128__{choice['param']}__fit{end}__to{max_cycle}.json")
        with np.load(candidate_path, allow_pickle=False) as z:
            fit_rows = z["fit_rows"].copy()
            rows = z["validation_rows"].copy()
            raw_new = z["raw_prediction"].copy()
            mapped_new = z["mapped_prediction"].copy()
            explained = z["explained_variance_ratio"].copy()
        if not np.array_equal(fit_rows, fit_expected) or not np.array_equal(rows, rows_expected):
            raise AssertionError(f"SVD128 row positions differ at fit_end={end}")
        if candidate_meta["state_sha256"] != digest_json(candidate_meta["state"]):
            raise AssertionError(f"SVD128 state hash mismatch at fit_end={end}")
        state = candidate_meta["state"]
        if state["svd"]["n_components_requested"] != 128 or state["svd"]["n_components_actual"] <= 32:
            raise AssertionError(f"SVD128 width audit failed at fit_end={end}")
        if state["validation_data_passed_to_fit"] or state["test_data_used_for_fit"]:
            raise AssertionError(f"Potential boundary leak at fit_end={end}")
        if candidate_meta["validation_labels_passed_to_fit"] or candidate_meta["test_data_read_or_used"]:
            raise AssertionError(f"Candidate fit boundary audit failed at fit_end={end}")
        if array_digest(raw_new) != candidate_meta["raw_prediction_sha256"]:
            raise AssertionError(f"SVD128 raw prediction hash mismatch at fit_end={end}")
        if array_digest(mapped_new) != candidate_meta["mapped_prediction_sha256"]:
            raise AssertionError(f"SVD128 mapped prediction hash mismatch at fit_end={end}")
        if array_digest(explained) != candidate_meta["explained_variance_ratio_sha256"]:
            raise AssertionError(f"SVD128 spectrum hash mismatch at fit_end={end}")
        manual_mapped = _manual_rank_map(raw_new, frame.iloc[rows].cycle_code.to_numpy(), float(choice["rank_weight"]))
        if not np.array_equal(mapped_new, manual_mapped):
            raise AssertionError(f"SVD128 fixed rank mapping mismatch at fit_end={end}")
        if np.any((mapped_new < 0) | (mapped_new > 100)) or not np.isfinite(mapped_new).all():
            raise AssertionError(f"SVD128 prediction range/finite check failed at fit_end={end}")
        candidate_vectors[end] = (rows, mapped_new)

    def vector_set(vectors: dict, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        row_parts, prediction_parts = [], []
        for window in specs:
            end = int(window["fit"][1])
            rows, prediction = vectors[end]
            mask = frame.iloc[rows].cycle_code.between(*window["validation"]).to_numpy()
            row_parts.append(rows[mask])
            prediction_parts.append(prediction[mask])
        rows, prediction = np.concatenate(row_parts), np.concatenate(prediction_parts)
        order = np.argsort(rows)
        return rows[order], prediction[order]

    primary_rows, primary_new = vector_set(candidate_vectors, primary)
    primary_control_rows, primary_old = vector_set(control_vectors, primary)
    h12_rows, h12_new = vector_set(candidate_vectors, secondary)
    h12_control_rows, h12_old = vector_set(control_vectors, secondary)
    if not np.array_equal(primary_rows, primary_control_rows) or not np.array_equal(h12_rows, h12_control_rows):
        raise AssertionError("Paired prediction rows differ between SVD32 and SVD128")
    if len(primary_rows) != len(np.unique(primary_rows)) or len(h12_rows) != len(np.unique(h12_rows)):
        raise AssertionError("Validation windows overlap")
    if len(primary_rows) != 38588:
        raise AssertionError(f"Canonical W1-W6 primary support changed: {len(primary_rows)}")

    computed = {}
    for key, rows, new, old in [
        ("primary", primary_rows, primary_new, primary_old),
        ("h12", h12_rows, h12_new, h12_old),
    ]:
        y = frame.iloc[rows].threat_rank.to_numpy(dtype=float)
        metric_new, metric_old = _basic(y, new), _basic(y, old)
        delta = float(np.mean(np.abs(new - y) - np.abs(old - y)))
        cycle_maes_new, cycle_maes_old = [], []
        by_cycle = []
        cycles = frame.iloc[rows].cycle_code.to_numpy()
        for cycle in sorted(pd.unique(cycles)):
            mask = cycles == cycle
            cmn, cmo = _basic(y[mask], new[mask]), _basic(y[mask], old[mask])
            cycle_maes_new.append(cmn["mae"])
            cycle_maes_old.append(cmo["mae"])
            by_cycle.append({"cycle": int(cycle), "control_mae": cmo["mae"], "candidate_mae": cmn["mae"], "delta_mae": cmn["mae"] - cmo["mae"]})
        section = result["aggregate"][key]
        _assert_close(metric_new["mae"], section["candidate"]["raw_mae"], key + " candidate MAE")
        _assert_close(metric_old["mae"], section["control"]["raw_mae"], key + " control MAE")
        _assert_close(delta, section["delta_mae"], key + " paired delta")
        _assert_close(float(np.mean(cycle_maes_new)), section["candidate"]["equal_cycle_mae"], key + " candidate equal-cycle MAE")
        _assert_close(float(np.mean(cycle_maes_old)), section["control"]["equal_cycle_mae"], key + " control equal-cycle MAE")
        computed[key] = {"control": metric_old, "candidate": metric_new, "delta_mae": delta, "cycles": by_cycle}
        view = frame.iloc[rows]
        actual_all = view.threat_rank.to_numpy(dtype=float)
        missing_stream = view.signal_stream.isna().to_numpy()
        missing_stream |= view.signal_stream.map(
            lambda value: isinstance(value, str) and not value.strip()
        ).to_numpy()
        subgroup_masks = {
            "EF0": view.entry_friction.eq(0).to_numpy(),
            "EF1": view.entry_friction.eq(1).to_numpy(),
            "tail_low_le5": actual_all <= 5,
            "tail_high_ge95": actual_all >= 95,
            "stream_missing": missing_stream,
            "stream_present": ~missing_stream,
            "region_low": actual_all < 100 / 3,
            "region_middle": (actual_all >= 100 / 3) & (actual_all < 200 / 3),
            "region_high": actual_all >= 200 / 3,
        }
        for name, mask in subgroup_masks.items():
            expected = section["subgroups"][name]
            for label, prediction in [("control", old), ("candidate", new)]:
                metric = _basic(actual_all[mask], prediction[mask])
                _assert_close(metric["n"], expected[label]["n"], f"{key} {name} {label} denominator")
                _assert_close(metric["mae"], expected[label]["mae"], f"{key} {name} {label} MAE")

    for window in [*primary, *secondary]:
        key = window["id"]
        end = int(window["fit"][1])
        rows, new = candidate_vectors[end]
        mask = frame.iloc[rows].cycle_code.between(*window["validation"]).to_numpy()
        rows = rows[mask]
        new = new[mask]
        old_rows, old = control_vectors[end]
        assert np.array_equal(rows, old_rows[mask])
        old = old[mask]
        y = frame.iloc[rows].threat_rank.to_numpy(dtype=float)
        cmn, cmo = _basic(y, new), _basic(y, old)
        delta = float(np.mean(np.abs(new - y) - np.abs(old - y)))
        section = result["windows"][key]
        _assert_close(cmn["mae"], section["candidate_mae"], f"{key} candidate MAE")
        _assert_close(cmo["mae"], section["control_mae"], f"{key} control MAE")
        _assert_close(delta, section["delta_mae"], f"{key} paired delta")

    _assert_close(computed["primary"]["control"]["mae"], 11.931600241417073, "Frozen primary SVD32 MAE")
    _assert_close(computed["h12"]["control"]["mae"], 14.704040710404005, "Frozen H12 SVD32 MAE")
    if len(result["fit_audits"]) != 6:
        raise AssertionError("Expected exactly six outer candidate fits")
    print("PASS: Train identity, source/protocol freeze, six fit boundaries, SVD128 states, row alignment and independent metrics")
    print(f"Primary: SVD32={computed['primary']['control']['mae']:.6f}; SVD128={computed['primary']['candidate']['mae']:.6f}; delta={computed['primary']['delta_mae']:+.6f}")
    print(f"H12: SVD32={computed['h12']['control']['mae']:.6f}; SVD128={computed['h12']['candidate']['mae']:.6f}; delta={computed['h12']['delta_mae']:+.6f}")
    if args.verify_inner:
        _verify_inner_refinement(frame, result)


def _verify_inner_refinement(frame: pd.DataFrame, stage1: dict) -> None:
    protocol = _load(INNER_PROTOCOL_PATH)
    freeze_path = ROOT / "inner_refinement_freeze.json"
    receipt = _load(freeze_path)
    result = _load(INNER_REPORT_PATH)
    stage1_report_path = PROJECT_ROOT / "reports/autonomous_eda/regression/results.json"
    stage1_report = _load(stage1_report_path)
    if receipt["signature"] != digest_json(receipt["identity"]):
        raise AssertionError("Inner refinement freeze signature does not recompute")
    if receipt["identity"]["protocol_sha256"] != digest_json(protocol):
        raise AssertionError("Inner protocol changed after freeze")
    if receipt["identity"]["first_stage_freeze_signature"] != stage1["freeze_signature"]:
        raise AssertionError("Inner stage points to a different first-stage freeze")
    if receipt["identity"]["first_stage_report_sha256"] != _sha_file(stage1_report_path):
        raise AssertionError("First-stage paired report changed after inner freeze")
    if receipt["identity"]["first_stage_postfit_amendment_sha256"] != _sha_file(AMENDMENT_PATH):
        raise AssertionError("First-stage report amendment changed after inner freeze")
    for path in CODE_PATHS:
        if receipt["identity"]["code_sha256"].get(path) != code_digest(PROJECT_ROOT / path):
            raise AssertionError(f"Inner stage source changed: {path}")
    for path, expected in receipt["identity"]["reference_sha256"].items():
        if expected != code_digest(PROJECT_ROOT / path):
            raise AssertionError(f"Inner stage reference changed: {path}")
    if result["freeze_signature"] != receipt["signature"]:
        raise AssertionError("Inner result does not match its own stage freeze")
    if result["first_stage_freeze_signature"] != stage1["freeze_signature"]:
        raise AssertionError("Inner result does not match the first-stage freeze")
    if result["fit_budget"]["full_train_fits"] != 0 or result["fit_budget"]["test_inference"] or result["fit_budget"]["kaggle_submissions"] != 0:
        raise AssertionError("Inner stage crossed a prohibited full-train/Test/Kaggle boundary")

    stage5 = _load(STAGE5_REPORT)
    old_selection = {int(item["fit_end"]): item["selected"] for item in stage5["inner"]}
    selected = {}
    inner_actual_fits = 0
    for entry in result["selections"]:
        end = int(entry["outer_fit_end"])
        fold = next(x for x in protocol["temporal_inner_folds"] if int(x["outer_fit_end"]) == end)
        lo_fit, hi_fit = fold["inner_fit"]
        lo_val, hi_val = fold["inner_validation"]
        fit_expected = _expected_rows(frame, lo_fit, hi_fit)
        val_expected = _expected_rows(frame, lo_val, hi_val)
        records = []
        y_val = frame.iloc[val_expected].threat_rank.to_numpy(dtype=float)
        cycle_val = frame.iloc[val_expected].cycle_code.to_numpy()
        for param in protocol["candidate_grid"]["params"]:
            fit_end = end - 4
            stem = f"inner128__{param}__fit{fit_end}__val{end}"
            npz_path, meta_path = RUN / f"{stem}.npz", RUN / f"{stem}.json"
            meta = _load(meta_path)
            with np.load(npz_path, allow_pickle=False) as z:
                fit_rows = z["fit_rows"].copy()
                val_rows = z["validation_rows"].copy()
                raw = z["raw_prediction"].copy()
                explained = z["explained_variance_ratio"].copy()
            if not np.array_equal(fit_rows, fit_expected) or not np.array_equal(val_rows, val_expected):
                raise AssertionError(f"Inner fold row coverage failed: {end}/{param}")
            if meta["state_sha256"] != digest_json(meta["state"]) or meta["state"]["svd"]["n_components_requested"] != 128:
                raise AssertionError(f"Inner fold model state invalid: {end}/{param}")
            if meta["validation_labels_passed_to_fit"] or meta["test_data_read_or_used"]:
                raise AssertionError(f"Inner fit boundary flag invalid: {end}/{param}")
            if array_digest(raw) != meta["raw_prediction_sha256"] or array_digest(explained) != meta["explained_variance_ratio_sha256"]:
                raise AssertionError(f"Inner cached output hash mismatch: {end}/{param}")
            key = digest_json({
                "inner_freeze_signature": receipt["signature"],
                "params_id": param,
                "fit_positions_sha256": positions_digest(fit_rows),
                "validation_positions_sha256": positions_digest(val_rows),
            })
            if key != meta["key"]:
                raise AssertionError(f"Inner cache key mismatch: {end}/{param}")
            inner_actual_fits += int(bool(meta.get("actual_fit_performed")))
            for weight in protocol["candidate_grid"]["rank_weights"]:
                pred = _manual_rank_map(raw, cycle_val, float(weight))
                # The weighted blend uses the same cycle percentile as the protocol.
                pred = (1.0 - float(weight)) * raw + float(weight) * _within_cycle_percentile(raw, cycle_val)
                pred = np.clip(pred, 0, 100)
                records.append({"param": param, "rank_weight": float(weight), "mae": float(np.mean(np.abs(pred - y_val)))})
        best = min(item["mae"] for item in records)
        close = [item for item in records if item["mae"] <= best + 1e-10]
        priority = {"regularized": 0, "base": 1}
        chosen = sorted(close, key=lambda item: (item["rank_weight"], priority[item["param"]]))[0]
        if chosen != entry["selected"]:
            raise AssertionError(f"Inner selection does not independently reproduce at cutoff {end}")
        if records != entry["choices"]:
            for computed, saved in zip(records, entry["choices"]):
                _assert_close(computed["mae"], saved["mae"], f"inner grid MAE {end}/{computed['param']}/{computed['rank_weight']}")
        if entry["stage5_svd32"]["param"] != old_selection[end]["param"] or not np.isclose(entry["stage5_svd32"]["rank_weight"], old_selection[end]["rank_weight"]):
            raise AssertionError(f"Stage 5 comparison policy changed at cutoff {end}")
        selected[end] = entry["selected"]
    if inner_actual_fits != 12 or result["fit_budget"]["inner_svd128_fits"] != 12:
        raise AssertionError(f"Expected twelve executed inner fits, found {inner_actual_fits}")

    primary = load_protocol()["primary_windows"]
    secondary = load_protocol()["secondary_windows"]
    stage1_vectors, control_vectors, selected_vectors = {}, {}, {}
    extra_outer_fits, reused_outer = 0, 0
    for end, choice in selected.items():
        stage5_choice = old_selection[end]
        max_cycle = max(int(w["validation"][1]) for w in [*primary, *secondary] if int(w["fit"][1]) == end)
        fit_rows = _expected_rows(frame, 47, end)
        rows_expected = _expected_rows(frame, end + 1, max_cycle)
        if choice["param"] == stage5_choice["param"]:
            stem = f"svd128__{stage5_choice['param']}__fit{end}__to{max_cycle}"
            with np.load(RUN / f"{stem}.npz", allow_pickle=False) as z:
                fit_rows_saved = z["fit_rows"].copy()
                rows_saved = z["validation_rows"].copy()
                raw, fixed = z["raw_prediction"].copy(), z["mapped_prediction"].copy()
            if not np.array_equal(fit_rows_saved, fit_rows) or not np.array_equal(rows_saved, rows_expected):
                raise AssertionError(f"Reused paired outer rows mismatch at cutoff {end}")
            entry = next(item for item in result["selections"] if item["outer_fit_end"] == end)
            if entry["outer"]["source"] != "reused_first_stage_paired_outer_fit":
                raise AssertionError(f"Reuse attribution incorrect at cutoff {end}")
            reused_outer += 1
        else:
            stem = f"outer128selected__{choice['param']}__fit{end}__to{max_cycle}"
            npz_path, meta_path = RUN / f"{stem}.npz", RUN / f"{stem}.json"
            meta = _load(meta_path)
            with np.load(npz_path, allow_pickle=False) as z:
                fit_rows_saved = z["fit_rows"].copy()
                rows_saved = z["validation_rows"].copy()
                raw = z["raw_prediction"].copy()
            if not np.array_equal(fit_rows_saved, fit_rows) or not np.array_equal(rows_saved, rows_expected):
                raise AssertionError(f"New selected outer rows mismatch at cutoff {end}")
            if meta["state_sha256"] != digest_json(meta["state"]) or meta["validation_labels_passed_to_fit"] or meta["test_data_read_or_used"]:
                raise AssertionError(f"New selected outer fit boundary invalid at cutoff {end}")
            key = digest_json({"inner_freeze_signature": receipt["signature"], "params_id": choice["param"],
                               "fit_positions_sha256": positions_digest(fit_rows_saved),
                               "validation_positions_sha256": positions_digest(rows_saved)})
            if meta["key"] != key or array_digest(raw) != meta["raw_prediction_sha256"]:
                raise AssertionError(f"New selected outer cache integrity failed at cutoff {end}")
            extra_outer_fits += int(bool(meta.get("actual_fit_performed")))
        cycle = frame.iloc[rows_expected].cycle_code.to_numpy()
        selected_pred = (1.0 - choice["rank_weight"]) * raw + choice["rank_weight"] * _within_cycle_percentile(raw, cycle)
        selected_pred = np.clip(selected_pred, 0, 100)
        if choice["param"] == stage5_choice["param"]:
            fixed_compare = (1.0 - float(stage5_choice["rank_weight"])) * raw + float(stage5_choice["rank_weight"]) * _within_cycle_percentile(raw, cycle)
            fixed_compare = np.clip(fixed_compare, 0, 100)
            if not np.array_equal(fixed_compare, fixed):
                raise AssertionError(f"Fixed SVD128 policy mapping mismatch at cutoff {end}")
        else:
            with np.load(RUN / f"svd128__{stage5_choice['param']}__fit{end}__to{max_cycle}.npz", allow_pickle=False) as z:
                old_rows = z["validation_rows"].copy()
                old_fixed = z["mapped_prediction"].copy()
            if not np.array_equal(old_rows, rows_expected):
                raise AssertionError(f"Stage 1 fixed candidate rows mismatch at cutoff {end}")
            fixed = old_fixed
        stage1_vectors[end] = (rows_expected, fixed)
        selected_vectors[end] = (rows_expected, selected_pred)
        _, control, _ = _load_stage5_control_for_verify(frame, end, old_selection[end], max_cycle)
        control_vectors[end] = (rows_expected, control)
    if extra_outer_fits != result["fit_budget"]["selected_outer_svd128_fits"] or reused_outer != result["fit_budget"]["reused_first_stage_outer_fits"]:
        raise AssertionError("Selected outer fit/reuse budget does not independently match")

    _verify_stage2_metrics(frame, result["selected_vs_stage5"], selected_vectors, control_vectors, primary, secondary, "selected_vs_stage5")
    _verify_stage2_metrics(frame, result["paired_vs_stage5"], stage1_vectors, control_vectors, primary, secondary, "paired_vs_stage5")
    _verify_stage2_metrics(frame, result["inner_vs_fixed"], selected_vectors, stage1_vectors, primary, secondary, "inner_vs_fixed", use_inner_outer=True)
    print(f"PASS: inner-only selection, twelve fit caches, {extra_outer_fits} selected outer fits, {reused_outer} reused outer fits, independent paired metrics")


def _within_cycle_percentile(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    result = np.empty_like(raw, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        result[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return result


def _load_stage5_control_for_verify(frame: pd.DataFrame, end: int, selected: dict, max_cycle: int):
    stem = f"lgbtext__{selected['param']}__fit{end}__to{max_cycle}"
    with np.load(STAGE5_RUN / f"{stem}.npz", allow_pickle=False) as z:
        rows, raw = z["validation_rows"].copy(), z["prediction"].copy()
    mapped = _manual_rank_map(raw, frame.iloc[rows].cycle_code.to_numpy(), float(selected["rank_weight"]))
    with np.load(STAGE5_RUN / f"eval_lgbtext__fit{end}__inner.npz", allow_pickle=False) as z:
        control_rows, control = z["validation_rows"].copy(), z["prediction"].copy()
    if not np.array_equal(rows, control_rows) or not np.array_equal(mapped, control):
        raise AssertionError(f"Frozen SVD32 control mapping mismatch at cutoff {end}")
    return rows, control, raw


def _collect_verification_vectors(frame: pd.DataFrame, vectors: dict, specs: list[dict]):
    row_parts, pred_parts = [], []
    for window in specs:
        end = int(window["fit"][1])
        rows, pred = vectors[end]
        mask = frame.iloc[rows].cycle_code.between(*window["validation"]).to_numpy()
        row_parts.append(rows[mask]); pred_parts.append(pred[mask])
    rows, pred = np.concatenate(row_parts), np.concatenate(pred_parts)
    order = np.argsort(rows)
    return rows[order], pred[order]


def _verify_stage2_metrics(frame: pd.DataFrame, report: dict, candidate: dict, control: dict,
                           primary: list[dict], secondary: list[dict], name: str,
                           use_inner_outer: bool = False) -> None:
    fixed_keys = ["primary", "h12"]
    for group, specs in [("primary", primary), ("h12", secondary)]:
        rows, cand = _collect_verification_vectors(frame, candidate, specs)
        old_rows, old = _collect_verification_vectors(frame, control, specs)
        if not np.array_equal(rows, old_rows) or len(rows) != 38588:
            raise AssertionError(f"{name}/{group} support differs")
        actual = frame.iloc[rows].threat_rank.to_numpy(dtype=float)
        cand_metric, old_metric = _basic(actual, cand), _basic(actual, old)
        delta = float(np.mean(np.abs(cand - actual) - np.abs(old - actual)))
        cycles = frame.iloc[rows].cycle_code.to_numpy()
        equal_cand = float(np.mean([np.mean(np.abs(cand[cycles == c] - actual[cycles == c])) for c in np.unique(cycles)]))
        equal_old = float(np.mean([np.mean(np.abs(old[cycles == c] - actual[cycles == c])) for c in np.unique(cycles)]))
        if use_inner_outer:
            section = report["aggregate"][group]
            expected_cand, expected_old = section["inner_selected"], section["fixed_policy"]
        else:
            section = report["aggregate"][group]
            expected_cand, expected_old = section["candidate"], section["control"]
        _assert_close(cand_metric["mae"], expected_cand["raw_mae"], f"{name}/{group} candidate MAE")
        _assert_close(old_metric["mae"], expected_old["raw_mae"], f"{name}/{group} control MAE")
        _assert_close(delta, section["delta_mae"], f"{name}/{group} paired delta")
        _assert_close(equal_cand, expected_cand["equal_cycle_mae"], f"{name}/{group} candidate equal-cycle MAE")
        _assert_close(equal_old, expected_old["equal_cycle_mae"], f"{name}/{group} control equal-cycle MAE")
    for window in [*primary, *secondary]:
        end = int(window["fit"][1])
        rows, cand = candidate[end]
        old_rows, old = control[end]
        mask = frame.iloc[rows].cycle_code.between(*window["validation"]).to_numpy()
        if not np.array_equal(rows[mask], old_rows[mask]):
            raise AssertionError(f"{name}/{window['id']} row alignment differs")
        y = frame.iloc[rows[mask]].threat_rank.to_numpy(dtype=float)
        delta = float(np.mean(np.abs(cand[mask] - y) - np.abs(old[mask] - y)))
        cand_mae, old_mae = float(np.mean(np.abs(cand[mask] - y))), float(np.mean(np.abs(old[mask] - y)))
        key = "inner_vs_fixed" if use_inner_outer else name
        if use_inner_outer:
            expected = report["windows"][window["id"]]
            _assert_close(cand_mae, expected["inner_selected_mae"], f"{name}/{window['id']} candidate MAE")
            _assert_close(old_mae, expected["fixed_policy_mae"], f"{name}/{window['id']} fixed MAE")
        else:
            expected = report["windows"][window["id"]]
            _assert_close(cand_mae, expected["candidate_mae"], f"{name}/{window['id']} candidate MAE")
            _assert_close(old_mae, expected["control_mae"], f"{name}/{window['id']} control MAE")
        _assert_close(delta, expected["delta_mae"], f"{name}/{window['id']} paired delta")


if __name__ == "__main__":
    main()
