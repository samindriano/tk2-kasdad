"""Read-only comparison of frozen direct-position OOF outputs and E79 B.

The script verifies all consumed artifacts before loading them. It reads only
the already-scored E67/E75 Train query rows and saved prediction/model states;
it never fits, transforms, predicts, opens Test/Sample/submission files, or
contacts Kaggle. Outputs are restricted to the forensic report namespace.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np


DIRECT_CARD = "REG_POSITION_DIRECT_L1_V1"
DIRECT_RUN = Path("experiments/runs/reg_position_direct_l1_v1")
DIRECT_REPORT = Path("reports/optimization_v3/reg_position_direct_l1_v1")
OLD_AUDIT = Path("reports/optimization_v3/reg_position_failure_mechanism_v1")
B_NAMESPACE = "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
B_REPORT = Path("reports/optimization_v3") / B_NAMESPACE
B_RUN = Path("experiments/runs") / B_NAMESPACE
OUTPUT = Path("reports/optimization_v3/public_local_forensic_v1/reg")
TRAIN_SHA = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
TARGET_SHA = "b7c22d8af92d0290ffd96615afedf5e3f6451e8bc204e9297368db33c40c9796"
REG_SNAPSHOT_SHA = "ca7d66aa2567b99460c11b0f47e7653a0d07490ecce34e29e6ad445446cdae36"
CLS_SNAPSHOT_SHA = "9ddbd945733af3c689f2024bf4598ee9a06fccf7dd372567ddd004f6b1643daa"
PUBLIC_REL = Path("reports/optimization_v3/public_local_forensic_v1")
TOL = 1e-10


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    meta = json.dumps({"shape": array.shape, "dtype": str(array.dtype)},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(meta)
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8", errors="surrogatepass"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False)
                          .encode("utf-8", errors="surrogatepass")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def verify_file(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"Missing pinned input ({label}): {path}")
    actual = sha_file(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Hash mismatch ({label}): {path}; got {actual}, expected {expected}")
    return actual


def verify_signature(value: dict[str, Any], key: str, label: str,
                     extra_excluded: tuple[str, ...] = ()) -> None:
    excluded = {key, *extra_excluded}
    expected = canonical_sha({k: v for k, v in value.items() if k not in excluded})
    if value.get(key) != expected:
        raise RuntimeError(f"{label} signature mismatch")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(target, dtype=np.float64) -
                                 np.asarray(prediction, dtype=np.float64))))


def params_for(obj: Any) -> dict[str, Any]:
    return obj.get_params(deep=False)


def params_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, np.dtype):
            a = str(a)
        if isinstance(b, np.dtype):
            b = str(b)
        if a != b:
            return False
    return True


def load_verified_joblib(path: Path, expected: str, label: str) -> Any:
    verify_file(path, expected, label)
    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--saved-artifact-root", type=Path, required=True)
    args = parser.parse_args()
    repo, source, saved = (p.resolve() for p in
                           (args.repo_root, args.source_root, args.saved_artifact_root))
    out = (repo / OUTPUT).resolve()
    if not out.is_relative_to(repo):
        raise RuntimeError("Output path escaped the owned worktree")

    # Reuse the already-reviewed query-only reader. It skips unselected Train
    # rows as raw bytes and decodes targets only for frozen scored query rows.
    sys.path.insert(0, str(repo))
    from scripts.reg_position_failure_mechanism_v1 import read_scored_query

    pins: dict[str, dict[str, str]] = {}

    def pin(path: Path, expected: str | None, label: str) -> str:
        actual = verify_file(path, expected, label)
        pins[label] = {"path": str(path), "sha256": actual}
        return actual

    old_hash_path = repo / OLD_AUDIT / "INPUT_HASHES.json"
    pin(old_hash_path, None, "prior_audit_input_hash_registry")
    pin(repo / OUTPUT / "public_local_forensic_v1.py", None, "forensic_script")
    old_hashes = read_json(old_hash_path)["input_hashes"]
    expected = lambda key: old_hashes[key]

    freeze_path = saved / DIRECT_RUN / "FREEZE_V4.json"
    completion_path = saved / DIRECT_REPORT / "CONTINUATION_COMPLETION_V1.json"
    recipe_path = saved / DIRECT_REPORT / "RECIPE.json"
    freeze = read_json(freeze_path)
    completion = read_json(completion_path)
    direct_recipe = read_json(recipe_path)
    pin(freeze_path, expected("saved_artifact_root/experiments/runs/reg_position_direct_l1_v1/FREEZE_V4.json"), "direct_freeze_v4")
    pin(completion_path, expected("saved_artifact_root/reports/optimization_v3/reg_position_direct_l1_v1/CONTINUATION_COMPLETION_V1.json"), "direct_fit_completion")
    pin(recipe_path, expected("saved_artifact_root/reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json"), "direct_recipe")

    b_freeze_path = repo / B_REPORT / "TRAINED_STATE_FREEZE_B.json"
    b_manifest_path = repo / B_REPORT / "INFERENCE_MANIFEST_B.json"
    b_protocol_path = repo / "experiments" / B_NAMESPACE / "PROTOCOL.json"
    b_freeze = read_json(b_freeze_path)
    b_manifest = read_json(b_manifest_path)
    b_protocol = read_json(b_protocol_path)
    for path, label in ((b_freeze_path, "B_trained_state_freeze"),
                        (b_manifest_path, "B_inference_manifest"),
                        (b_protocol_path, "B_protocol")):
        pin(path, None, label)
    verify_signature(b_freeze, "signature", "B trained state freeze", ("signature_method",))
    verify_signature(b_manifest, "signature", "B inference manifest", ("signature_method",))
    if b_freeze.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY":
        raise RuntimeError("Unexpected B trained-state status")
    if b_manifest.get("baseline_member_used") != "manifest-bound final mapped R06 prediction" or b_manifest.get("baseline_raw_candidate_member_used") is not False:
        raise RuntimeError("B inference did not bind to the final mapped R06 baseline")

    # Verify and load saved direct prediction bundles only.
    bundles: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for origin in (67, 75):
        for arm in ("BAG", "REAL", "SHUFFLED"):
            alias = f"DIRECT_E{origin}_{arm}"
            fit_id = f"{DIRECT_CARD}_E{origin}_{arm}"
            bundle_path = saved / DIRECT_RUN / "predictions" / f"{fit_id}.npz"
            receipt_path = bundle_path.with_suffix(".json")
            pin(bundle_path, expected(f"saved_bundles/{alias}.npz"), f"saved_{alias}_predictions")
            pin(receipt_path, expected(f"saved_bundles/{alias}.receipt"), f"saved_{alias}_receipt")
            receipt = read_json(receipt_path)
            verify_signature(receipt, "receipt_sha256", f"{alias} receipt")
            with np.load(bundle_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            required = {"query_rows", "query_record_ids", "query_cycles", "baseline",
                        "raw_auxiliary", "mapped_auxiliary", "prediction", "missing_mask"}
            if not required.issubset(arrays):
                raise RuntimeError(f"Missing arrays in {alias}")
            if receipt.get("query_targets_read") is not False:
                raise RuntimeError(f"Unexpected query-target access recorded in {alias}")
            bundles[(origin, arm)] = arrays

    e67 = bundles[(67, "REAL")]
    rows67 = np.asarray(e67["query_rows"], dtype=np.int64)
    ids67 = np.asarray(e67["query_record_ids"], dtype=str)
    cycles67 = np.asarray(e67["query_cycles"], dtype=np.int64)
    if rows67.size != 20_903 or set(np.unique(cycles67)) != set(range(68, 80)):
        raise RuntimeError("E67 support differs from frozen H12 scored query support")
    targets67, text_load67, parsed_rows, source_audit = read_scored_query(
        source, rows67, ids67, cycles67, expected_total_rows=50_608)
    if source_audit["query_target_values_sha256"] != TARGET_SHA:
        raise RuntimeError("Selected query target hash differs from prior frozen scoring support")
    pin(source / "data/raw/regression/Train.csv", TRAIN_SHA, "official_train_query_support")
    if not np.array_equal(parsed_rows, rows67):
        raise RuntimeError("Selected Train rows are misaligned")
    row_to_label = {int(row): float(y) for row, y in zip(parsed_rows, targets67)}
    row_to_load = {int(row): float(x) for row, x in zip(parsed_rows, text_load67)}

    labels: dict[int, np.ndarray] = {}
    loads: dict[int, np.ndarray] = {}
    for origin in (67, 75):
        real = bundles[(origin, "REAL")]
        rows = np.asarray(real["query_rows"], dtype=np.int64)
        ids = np.asarray(real["query_record_ids"], dtype=str)
        cycles = np.asarray(real["query_cycles"], dtype=np.int64)
        labels[origin] = np.asarray([row_to_label[int(r)] for r in rows], dtype=np.float64)
        loads[origin] = np.asarray([row_to_load[int(r)] for r in rows], dtype=np.float64)
        for arm in ("BAG", "REAL", "SHUFFLED"):
            b = bundles[(origin, arm)]
            if not (np.array_equal(b["query_rows"], rows) and
                    np.array_equal(np.asarray(b["query_record_ids"], dtype=str), ids) and
                    np.array_equal(b["query_cycles"], cycles)):
                raise RuntimeError(f"Direct arms are not aligned: E{origin} {arm}")
            if not np.array_equal(b["baseline"], real["baseline"]):
                raise RuntimeError(f"Direct arm baseline differs: E{origin} {arm}")
        if not np.isfinite(labels[origin]).all():
            raise RuntimeError(f"Nonfinite scored labels: E{origin}")

    # Bind local R06 baselines to the frozen mapped-current-R06 cache member.
    baseline_cache_pins = []
    for origin in (67, 75):
        meta = freeze["origins"][str(origin)]["baseline"]
        cache_path = source / meta["path"]
        pin(cache_path, meta["sha256"], f"R06_E{origin}_mapped_cache")
        with np.load(cache_path, allow_pickle=False) as archive:
            cache = {name: np.asarray(archive[name]) for name in archive.files}
        if not {"prediction", "score_rows", "score_record_ids"}.issubset(cache):
            raise RuntimeError(f"R06 E{origin} cache lacks identity/final prediction members")
        real = bundles[(origin, "REAL")]
        if not (np.array_equal(cache["score_rows"], real["query_rows"]) and
                np.array_equal(cache["score_record_ids"].astype(str), real["query_record_ids"].astype(str)) and
                np.array_equal(cache["prediction"], real["baseline"])):
            raise RuntimeError(f"Direct E{origin} baseline is not the exact mapped R06 cache member")
        if sha_array(cache["prediction"], dtype="float64") != meta["prediction_sha256"]:
            raise RuntimeError(f"Mapped R06 prediction member hash mismatch for E{origin}")
        baseline_cache_pins.append({"origin": origin, "archive_sha256": meta["sha256"],
                                    "prediction_member_sha256": meta["prediction_sha256"],
                                    "rows": int(cache["prediction"].size),
                                    "identity_and_prediction_equal": True})

    # Load and compare fitted states after verifying their immutable hashes.
    direct_models: dict[int, Any] = {}
    direct_structured: dict[int, dict[str, Any]] = {}
    direct_scalers: dict[int, Any] = {}
    direct_ohes: dict[int, Any] = {}
    direct_tfidfs: dict[int, Any] = {}
    for origin in (67, 75):
        units = saved / DIRECT_RUN / "units"
        def direct_unit(name: str) -> Any:
            stem = f"{DIRECT_CARD}_E{origin}_{name}"
            expected_hash = completion["models"][stem]["artifact_sha256"]
            path = units / f"{stem}.joblib"
            pin(path, expected_hash, f"direct_E{origin}_{name}_state")
            return joblib.load(path)
        direct_models[origin] = direct_unit("REAL")
        direct_structured[origin] = direct_unit("STRUCTURED")
        direct_scalers[origin] = direct_unit("SCALER")
        direct_ohes[origin] = direct_unit("OHE")
        vectorizer_path = (source / "experiments/runs/optimization_v3_research_next_v1/"
                           "POSITIONAL_TEXT_PILOT_V1/units/"
                           f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF.joblib")
        vec_hash = freeze["origins"][str(origin)]["positional_vectorizer_compatibility"]["artifact_sha256"]
        pin(vectorizer_path, vec_hash, f"direct_E{origin}_frozen_TFIDF_state")
        direct_tfidfs[origin] = joblib.load(vectorizer_path)

    b_objects: dict[str, Any] = {}
    for rec in b_freeze["fit_receipts"]:
        path = repo / rec["artifact_path"]
        b_objects[rec["fit_id"]] = load_verified_joblib(
            path, rec["artifact_sha256"], f"B_{rec['fit_id']}_state")
    b_state_path = repo / b_freeze["candidate_state_path"]
    b_state = load_verified_joblib(b_state_path, b_freeze["candidate_state_sha256"], "B_complete_candidate_state")
    if not isinstance(b_state, dict) or b_state.get("candidate") != "B":
        raise RuntimeError("Unexpected B saved-state structure")

    tfidf_params = [params_for(direct_tfidfs[67]), params_for(direct_tfidfs[75]),
                    params_for(b_objects["B_SHARED_WORD_TFIDF"])]
    tfidf_param_match = tfidf_params[0] == tfidf_params[1] == tfidf_params[2]
    vocab_stats = []
    b_vec = b_objects["B_SHARED_WORD_TFIDF"]
    for origin in (67, 75):
        dvec = direct_tfidfs[origin]
        dvocab = dvec.vocabulary_
        bvocab = b_vec.vocabulary_
        common = set(dvocab).intersection(bvocab)
        d_idf = dvec.idf_
        b_idf = b_vec.idf_
        delta = np.asarray([abs(float(d_idf[dvocab[t]]) - float(b_idf[bvocab[t]]))
                            for t in common], dtype=np.float64)
        union = set(dvocab).union(bvocab)
        vocab_stats.append({
            "origin": origin, "direct_rows": int(freeze["origins"][str(origin)]["fit_rows"]),
            "B_rows": int(b_freeze["train_rows"]), "direct_vocab_size": len(dvocab),
            "B_vocab_size": len(bvocab), "intersection_terms": len(common),
            "intersection_over_union": float(len(common) / len(union)),
            "direct_only_terms": len(set(dvocab) - set(bvocab)),
            "B_only_terms": len(set(bvocab) - set(dvocab)),
            "common_term_idf_abs_diff_mean": float(np.mean(delta)),
            "common_term_idf_abs_diff_median": float(np.median(delta)),
            "common_term_idf_abs_diff_p95": float(np.quantile(delta, 0.95)),
            "common_term_idf_abs_diff_max": float(np.max(delta)),
            "direct_idf_sha256": sha_array(d_idf, dtype="float64"),
            "B_idf_sha256": sha_array(b_idf, dtype="float64"),
        })

    b_receipt_by_id = {rec["fit_id"]: rec for rec in b_freeze["fit_receipts"]}
    b_model = b_objects["B_DIRECT_POSITION_REAL_REGRESSOR"]
    b_model_details = b_receipt_by_id["B_DIRECT_POSITION_REAL_REGRESSOR"]["details"]
    b_params = params_for(b_model)
    model_params_match = [params_equal(params_for(direct_models[o]), params_for(b_model))
                          for o in (67, 75)]
    direct_struct_names = [direct_structured[o]["feature_names"] for o in (67, 75)]
    b_struct_details = b_receipt_by_id["B_STRUCTURED"]["details"]
    b_processor_audit = b_struct_details["processor_audit"]
    b_feature_names = b_processor_audit["feature_names"]
    b_numeric_positions = b_processor_audit["numeric_positions"]
    b_categorical_positions = b_processor_audit["categorical_positions"]
    direct_struct_order_match = [direct_struct_names[o] == b_feature_names for o in (0, 1)]
    direct_numeric_positions_match = [direct_structured[o]["numeric_positions"] == b_numeric_positions
                                      for o in (67, 75)]
    direct_categorical_positions_match = [direct_structured[o]["categorical_positions"] == b_categorical_positions
                                          for o in (67, 75)]
    scaler_params_match = [params_equal(params_for(direct_scalers[o]),
                                        params_for(b_objects["B_SCALER"])) for o in (67, 75)]
    ohe_params_match = [params_equal(params_for(direct_ohes[o]),
                                     params_for(b_objects["B_OHE"])) for o in (67, 75)]

    feature_comparison = {
        "direct_recipe_status": direct_recipe["status"],
        "direct_target": direct_recipe["target"],
        "B_target": b_model_details["target"],
        "target_contract_matches": "threat_rank" in direct_recipe["target"] and "threat_rank" in b_model_details["target"],
        "direct_fit_rows": {str(o): int(freeze["origins"][str(o)]["fit_rows"]) for o in (67, 75)},
        "B_fit_rows": int(b_freeze["train_rows"]),
        "direct_fit_cycles": {str(o): freeze["origins"][str(o)]["fit_cycle_bounds"] for o in (67, 75)},
        "B_fit_cycles": b_freeze["train_cycles"],
        "direct_query_cycles": {str(o): freeze["origins"][str(o)]["query_cycle_bounds"] for o in (67, 75)},
        "B_deployment_cycles": b_manifest.get("test_cycle_counts"),
        "direct_tfidf_parameters_equal_across_states": bool(tfidf_param_match),
        "tfidf_vocab_idf_comparison": vocab_stats,
        "lgbm_class": type(b_model).__module__ + "." + type(b_model).__name__,
        "direct_lgbm_params_equal_B": model_params_match,
        "B_resolved_lgbm_parameters": b_model_details["parameters"],
        "B_model_params_match_fit_receipt": params_equal(b_params, b_model_details["parameters"]),
        "direct_model_input_widths": {str(o): int(direct_models[o].n_features_in_) for o in (67, 75)},
        "B_design_shape_from_fit_receipt": b_receipt_by_id["B_DIRECT_POSITION_REAL_REGRESSOR"]["details"]["design_shape"],
        "B_model_input_width": int(b_model.n_features_in_),
        "direct_unigram_widths": {str(o): int(freeze["origins"][str(o)]["unigram_width"]) for o in (67, 75)},
        "B_unigram_width": int(b_model_details["position_audit"]["unigram_vocabulary_size"]),
        "direct_structured_order_matches_B": direct_struct_order_match,
        "direct_structured_numeric_positions_match_B": direct_numeric_positions_match,
        "direct_structured_categorical_positions_match_B": direct_categorical_positions_match,
        "scaler_params_equal_B": scaler_params_match,
        "ohe_params_equal_B": ohe_params_match,
        "scaler_input_widths": {str(o): int(direct_scalers[o].n_features_in_) for o in (67, 75)},
        "ohe_input_widths": {str(o): int(direct_ohes[o].n_features_in_) for o in (67, 75)},
        "B_structured_output_shape": b_struct_details["structured_matrix_shape"],
        "position_bag_pipeline_contract": "[structured, unigram_bag, REAL_three_third_position_suffix, missing_text_indicator]",
        "rank_mapping_and_final_blend_contract": direct_recipe["fixed_competition_policy"],
        "B_feature_recipe": b_protocol["candidate_B"]["features"],
        "B_policy_contract": b_protocol["candidate_B"]["policy"],
        "B_baseline_member_used": b_manifest["baseline_member_used"],
        "B_raw_candidate_used_as_baseline": b_manifest["baseline_raw_candidate_member_used"],
        "B_missing_fallback_bit_exact": b_manifest["missing_fallback_bit_exact"],
        "B_test_labels_loaded": b_manifest["test_labels_loaded"],
    }

    # Recompute summary metrics from already-scored arrays only.
    metric_rows: list[dict[str, Any]] = []
    view_defs = {67: {"H12": (68, 79), "W4": (68, 71), "W6": (76, 79)},
                 75: {"W6": (76, 79)}}
    for origin, views in view_defs.items():
        for view, (lo, hi) in views.items():
            b = bundles[(origin, "REAL")]
            cycles = np.asarray(b["query_cycles"], dtype=np.int64)
            mask = (cycles >= lo) & (cycles <= hi)
            y = labels[origin][mask]
            base = np.asarray(b["baseline"], dtype=np.float64)[mask]
            arm_maes: dict[str, float] = {}
            for arm in ("BAG", "REAL", "SHUFFLED"):
                ab = bundles[(origin, arm)]
                arm_maes[arm] = mae(y, np.asarray(ab["prediction"], dtype=np.float64)[mask])
                metric_rows.append({
                    "origin": origin, "view": view, "arm": arm, "n": int(mask.sum()),
                    "r06_mae": mae(y, base),
                    "raw_auxiliary_mae": mae(y, np.asarray(ab["raw_auxiliary"], dtype=np.float64)[mask]),
                    "mapped_auxiliary_mae": mae(y, np.asarray(ab["mapped_auxiliary"], dtype=np.float64)[mask]),
                    "final_policy_mae": arm_maes[arm],
                    "gain_vs_r06": mae(y, base) - arm_maes[arm],
                    "gain_vs_BAG": "", "gain_vs_SHUFFLED": "",
                })
            for row in metric_rows[-3:]:
                if row["arm"] == "REAL":
                    row["gain_vs_BAG"] = arm_maes["BAG"] - arm_maes["REAL"]
                    row["gain_vs_SHUFFLED"] = arm_maes["SHUFFLED"] - arm_maes["REAL"]

    cycle_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    for origin in (67, 75):
        b = bundles[(origin, "REAL")]
        cycles = np.asarray(b["query_cycles"], dtype=np.int64)
        missing = np.asarray(b["missing_mask"], dtype=bool)
        y = labels[origin]
        base = np.asarray(b["baseline"], dtype=np.float64)
        pred = np.asarray(b["prediction"], dtype=np.float64)
        correction = pred - base
        residual = y - base
        improvement = np.abs(residual) - np.abs(y - pred)
        for cycle in range(76, 80):
            cm = cycles == cycle
            yy, bb, pp = y[cm], base[cm], pred[cm]
            cycle_rows.append({"origin": origin, "cycle": cycle, "n": int(cm.sum()),
                               "r06_mae": mae(yy, bb), "final_mae": mae(yy, pp),
                               "gain_vs_r06": mae(yy, bb) - mae(yy, pp)})
            for population, pmask in (("ALL", cm), ("PRESENT", cm & ~missing)):
                r, c, imp = residual[pmask], correction[pmask], improvement[pmask]
                cov = float(np.cov(c, r, ddof=0)[0, 1]) if c.size > 1 else float("nan")
                corr = float(np.corrcoef(c, r)[0, 1]) if c.size > 1 and np.std(c) > 0 and np.std(r) > 0 else float("nan")
                direction = (np.sign(c) == np.sign(r))
                correction_rows.append({
                    "origin": origin, "cycle": cycle, "population": population,
                    "n": int(c.size), "correction_mean": float(np.mean(c)) if c.size else 0.0,
                    "residual_mean": float(np.mean(r)) if r.size else 0.0,
                    "correction_residual_covariance": cov,
                    "correction_residual_correlation": corr,
                    "correction_residual_same_sign_fraction": float(np.mean(direction)) if c.size else float("nan"),
                    "fraction_rows_improved": float(np.mean(imp > 0)) if c.size else float("nan"),
                    "mean_absolute_error_improvement": float(np.mean(imp)) if c.size else float("nan"),
                })

    # Cycle-cluster bootstrap of the eight-cycle late-unique view. Resampling
    # is descriptive only: eight clusters, reused development data, adaptive
    # candidate selection, and no connection to the unknown public/private split.
    clusters: list[tuple[int, np.ndarray]] = []
    for origin, lo, hi in ((67, 68, 71), (75, 76, 79)):
        b = bundles[(origin, "REAL")]
        cyc = np.asarray(b["query_cycles"], dtype=np.int64)
        y = labels[origin]
        base = np.asarray(b["baseline"], dtype=np.float64)
        pred = np.asarray(b["prediction"], dtype=np.float64)
        d = np.abs(y - base) - np.abs(y - pred)
        for cycle in range(lo, hi + 1):
            vals = d[cyc == cycle]
            if vals.size == 0:
                raise RuntimeError(f"Empty cycle in late-unique view: E{origin} cycle {cycle}")
            clusters.append((cycle, vals))
    point = float(sum(float(np.sum(v)) for _, v in clusters) /
                  sum(int(v.size) for _, v in clusters))
    rng = np.random.default_rng(42)
    reps = np.empty(2_000, dtype=np.float64)
    for i in range(reps.size):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        total = 0.0
        count = 0
        for j in selected:
            vals = clusters[int(j)][1]
            total += float(np.sum(vals))
            count += int(vals.size)
        reps[i] = total / count
    cluster_bootstrap = {
        "view": "E67_W4_cycles_68_71_plus_E75_W6_cycles_76_79",
        "clusters": len(clusters), "cluster_cycles": [int(c) for c, _ in clusters],
        "n_rows": int(sum(v.size for _, v in clusters)),
        "paired_difference": "|R06 - target| - |final_direct_REAL - target|; positive is a gain",
        "point_row_weighted_mean_gain": point, "resamples": 2000, "seed": 42,
        "cluster_resampling": "8 cycles sampled with replacement; all rows from sampled cycles retained",
        "percentile_95_interval": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
        "bootstrap_mean": float(np.mean(reps)), "bootstrap_median": float(np.median(reps)),
        "interpretation_limit": "Descriptive development-data interval only; 8 clusters, adaptive repeated development, and not public/private uncertainty or a significance test.",
    }

    # Pull the previously frozen fixed-group table, retaining only required
    # direct REAL rows; its source hash is already pinned by the prior audit.
    group_source = repo / OLD_AUDIT / "fixed_group_metrics.csv"
    group_hash = read_json(repo / OLD_AUDIT / "AUDIT.json")["outputs"]["fixed_group_metrics.csv"]
    pin(group_source, group_hash, "prior_fixed_group_metrics")
    with group_source.open(newline="", encoding="utf-8") as stream:
        all_group_rows = list(csv.DictReader(stream))
    group_rows = [r for r in all_group_rows if r["family"] == "DIRECT" and r["arm"] == "REAL"
                  and ((r["origin"] == "67" and r["view"] == "H12") or
                       (r["origin"] == "75" and r["view"] == "W6"))
                  and r["group"].startswith(("TARGET_", "TEXT_LOAD_"))]

    # Public benchmark status is read from the supplied immutable CLI snapshot.
    public_root = source / PUBLIC_REL
    reg_snapshot = public_root / "KAGGLE_REGRESSION_SNAPSHOT_20261104.json"
    cls_snapshot = public_root / "KAGGLE_CLASSIFICATION_SNAPSHOT_20261104.json"
    pin(reg_snapshot, REG_SNAPSHOT_SHA, "Kaggle_regression_history_snapshot")
    pin(cls_snapshot, CLS_SNAPSHOT_SHA, "Kaggle_classification_history_snapshot_separate_task")
    public_records = json.loads(reg_snapshot.read_text(encoding="utf-8"))
    complete = [r for r in public_records if r.get("status") == "SubmissionStatus.COMPLETE"
                and r.get("publicScore") not in (None, "")]
    best = min(complete, key=lambda r: float(r["publicScore"]))
    b_public = next(r for r in complete if int(r["ref"]) == 57988146)
    r06_public = next(r for r in complete if int(r["ref"]) == 57874821)
    public_delta = float(r06_public["publicScore"]) - float(b_public["publicScore"])
    public_summary = {
        "snapshot_sha256": REG_SNAPSHOT_SHA, "records": len(public_records),
        "complete_with_public_score": len(complete), "B_ref": b_public["ref"],
        "B_filename": b_public["fileName"], "B_status": b_public["status"],
        "B_public_score": float(b_public["publicScore"]),
        "B_private_score": b_public["privateScore"] or None,
        "R06_ref": r06_public["ref"], "R06_public_score": float(r06_public["publicScore"]),
        "B_public_gain_vs_R06": public_delta, "B_is_current_snapshot_public_best": int(best["ref"]) == 57988146,
        "B_is_snapshot_public_best_ref": best["ref"], "B_is_snapshot_public_best_score": float(best["publicScore"]),
        "private_scores_blank_count": sum(not r.get("privateScore") for r in complete),
        "classification_snapshot_used_for_regression_inference": False,
        "public_sample_composition_or_randomness_known": False,
    }

    # Independent arithmetic check against the frozen prior metric table.
    old_metric_path = repo / OLD_AUDIT / "arm_window_metrics.csv"
    old_metric_hash = read_json(repo / OLD_AUDIT / "AUDIT.json")["outputs"]["arm_window_metrics.csv"]
    pin(old_metric_path, old_metric_hash, "prior_arm_window_metrics")
    with old_metric_path.open(newline="", encoding="utf-8") as stream:
        old_metrics = list(csv.DictReader(stream))
    checks = []
    for row in metric_rows:
        matches = [x for x in old_metrics if x["family"] == "DIRECT" and
                   int(x["origin"]) == row["origin"] and x["view"] == row["view"] and
                   x["arm"] == row["arm"] and x["subset"] == "ALL"]
        if len(matches) != 1:
            raise RuntimeError("Missing prior saved-score metric for arithmetic cross-check")
        prior = matches[0]
        diff = abs(float(prior["final_policy_gain_vs_r06"]) - float(row["gain_vs_r06"]))
        if diff > TOL:
            raise RuntimeError(f"Saved-score metric recomputation differs by {diff}")
        checks.append({"origin": row["origin"], "view": row["view"], "arm": row["arm"],
                       "gain_abs_diff": diff, "within_1e-10": True})

    write_csv(out / "public_local_metrics.csv", metric_rows,
              ["origin", "view", "arm", "n", "r06_mae", "raw_auxiliary_mae",
               "mapped_auxiliary_mae", "final_policy_mae", "gain_vs_r06", "gain_vs_BAG", "gain_vs_SHUFFLED"])
    write_csv(out / "cycle_metrics.csv", cycle_rows,
              ["origin", "cycle", "n", "r06_mae", "final_mae", "gain_vs_r06"])
    write_csv(out / "group_metrics.csv", group_rows,
              ["family", "origin", "view", "arm", "group", "n", "status", "target_scale",
               "r06_mae", "raw_auxiliary_mae_on_own_target", "final_policy_mae", "final_policy_gain_vs_r06"])
    write_csv(out / "correction_diagnostics.csv", correction_rows,
              ["origin", "cycle", "population", "n", "correction_mean", "residual_mean",
               "correction_residual_covariance", "correction_residual_correlation",
               "correction_residual_same_sign_fraction", "fraction_rows_improved",
               "mean_absolute_error_improvement"])
    write_json(out / "cluster_bootstrap.json", cluster_bootstrap)
    write_json(out / "feature_state_comparison.json", feature_comparison)
    write_json(out / "public_snapshot_summary.json", public_summary)
    write_json(out / "metric_reproduction_checks.json", {
        "source": "existing frozen scored-output metrics; no new fit or prediction",
        "n_checks": len(checks), "all_within_1e-10": all(x["within_1e-10"] for x in checks),
        "checks": checks, "R06_cache_checks": baseline_cache_pins,
        "selected_query_support": source_audit,
    })
    def metric(origin: int, view: str, arm: str = "REAL") -> dict[str, Any]:
        return next(r for r in metric_rows if r["origin"] == origin and r["view"] == view and r["arm"] == arm)

    def fmt(value: Any) -> str:
        return f"{float(value):.5f}"

    e67_h12, e67_w6, e75_w6 = metric(67, "H12"), metric(67, "W6"), metric(75, "W6")
    groups_to_print = [r for r in group_rows if r["group"].startswith("TARGET_")]
    group_md = ["| Origin/view | Group | n | Gain vs R06 (MAE) |", "|---|---:|---:|---:|"]
    for row in groups_to_print:
        group_md.append(f"| E{row['origin']} {row['view']} | {row['group']} | {row['n']} | {float(row['final_policy_gain_vs_r06']):+.5f} |")
    cycle_md = ["| Origin | Cycle | n | Gain vs R06 (MAE) |", "|---:|---:|---:|---:|"]
    for row in cycle_rows:
        cycle_md.append(f"| E{row['origin']} | {row['cycle']} | {row['n']} | {float(row['gain_vs_r06']):+.5f} |")
    correction_summary = {}
    for origin in (67, 75):
        present = [r for r in correction_rows if r["origin"] == origin and r["population"] == "PRESENT"]
        n = sum(r["n"] for r in present)
        correction_summary[str(origin)] = {
            "present_rows": n,
            "same_sign_fraction_weighted": sum(r["n"] * r["correction_residual_same_sign_fraction"] for r in present) / n,
            "cycle_correlation_min": min(r["correction_residual_correlation"] for r in present),
            "cycle_correlation_max": max(r["correction_residual_correlation"] for r in present),
            "improved_row_fraction_weighted": sum(r["n"] * r["fraction_rows_improved"] for r in present) / n,
        }
    report_lines = [
        "# Public/local forensic comparison: direct-position regression",
        "",
        "**Disposition:** keep `REG_POSITION_DIRECT_L1_V1` at FAIL / no promotion. Candidate B is the current public-score leader in the supplied regression snapshot, an observed benchmark result that does not retro-pass the local gate.",
        "",
        "## Scope and access",
        "",
        "This is a read-only comparison of hash-pinned fitted states, saved direct E67/E75 prediction bundles, already-scored Train query labels, frozen R06 OOF caches, and the supplied Kaggle history snapshots. The script made zero fit or predict calls, parsed no fit-prefix target rows, did not open Test labels, Sample Submission, or a new submission CSV, and did not contact Kaggle. Writes are limited to this report namespace.",
        "",
        "The local target support is the previously scored E67 query rows (20,903 rows, cycles 68–79); E75 W6 is the same 8,491 rows as E67 W6. These are adaptive development views, not untouched validation.",
        "",
        "## Public benchmark observation",
        "",
        f"The 2026-11-04 regression snapshot has {public_summary['complete_with_public_score']} COMPLETE scored records. B (`{public_summary['B_filename']}`, ref {public_summary['B_ref']}) scored **{public_summary['B_public_score']:.5f}**, the lowest public score in that snapshot, versus R06 ref {public_summary['R06_ref']} at **{public_summary['R06_public_score']:.5f}**: a public MAE improvement of **{public_summary['B_public_gain_vs_R06']:.5f}**. All {public_summary['private_scores_blank_count']} private scores are blank. The public sample composition/randomness and private outcome are unknown; this is one selected observed score, not a significance result or evidence of private transfer.",
        "",
        "The separate classification snapshot is pinned only to prevent cross-task mix-ups; it contributes no regression evidence.",
        "",
        "## Recipe and fitted-state comparison",
        "",
        "The available evidence shows no material recipe or baseline-definition mismatch between the frozen pilot and B:",
        "",
        "- Both use direct `threat_rank`, the same `LGBMRegressor` objective and all resolved parameters: L1 regression, GBDT, 400 trees, learning rate .05, 15 leaves, min child samples 80, lambda 3, alpha 0, max depth -1, seed 42, two jobs, deterministic and column-wise execution. The full parameter receipt is in `feature_state_comparison.json`.",
        "- The ordered structured fields, 36 numeric positions and 3 categorical positions match. Both use `StandardScaler(with_mean=True, with_std=True)` and sparse `OneHotEncoder(handle_unknown=infrequent_if_exist, min_frequency=30, max_categories=254)`. B's structured block is 426 columns; learned category sets differ with the training population.",
        "- Both use `[structured, unigram bag, REAL three-third position suffix, missing indicator]`. The word TF-IDF parameter dictionaries match exactly (word analyzer, 1–2 grams, `min_df=3`, `max_df=.995`, 20,000 max features, sublinear TF, Unicode accent stripping, float64, L2, smooth IDF). The selected unigram bag is row-L2 normalized. The position descriptor uses the same saved three-third counts weighted by `(1+log(count))*shared_IDF`, then row-L2 normalized once.",
        "- Both rank raw auxiliary scores within each complete cycle using average ties and percentile `100*rank/n`, then map `0.5*raw + 0.5*percentile`. Both use `clip(0.75*final_mapped_R06 + 0.25*mapped_aux, 0, 100)`, calculate the whole-cycle rank/blend before masking, and replace missing-text rows with exact final-mapped R06.",
        "- Direct E67 and E75 use their respective prefix-fitted states (29,705 rows through cycle 67; 42,117 through cycle 75). B uses the full 50,608 Train rows through cycle 79, as intended for deployment. Therefore the learned vocabulary, IDF, structured categories, and LGBM input width are different fitted states. That is a train-boundary/state difference, not a different recipe: the 20k-term vocabulary overlaps B by 17,143 terms for E67 (Jaccard .750) and 18,686 for E75 (Jaccard .877); mean absolute IDF differences on common terms are .2286 and .1044. Input widths are 22,985 (E67), 23,263 (E75), and 23,439 (B).",
        "- Both local R06 baselines are exact matches to the hash-pinned final mapped R06 prediction member in their origin cache. B's manifest explicitly binds the baseline to the manifest-bound final mapped R06 prediction and says the raw candidate member was not used as baseline.",
        f"- Content pins: direct freeze signature `{freeze['freeze_signature']}`; B trained-state signature `{b_freeze['signature']}`. Direct REAL model, E67/E75 and B regressor, TF-IDF, structured, scaler, OHE, prediction bundle, R06 cache, and snapshot file hashes are recorded in `evidence_pins.json`.",
        "",
        "B fits cycles 47–79 and predicts cycles 80–91 (12 cycles). E67 predicts 68–79 (12 cycles) from only 21 training cycles; E75 W6 predicts 76–79 (four cycles) from 29 training cycles. No exact 33-cycle fit → 12-future-cycle OOS evaluation exists within Train. These horizon and training-support differences are material to transfer, but are the intended full-fit vs local OOF comparison.",
        "",
        "## Saved local results",
        "",
        "| View | n | R06 MAE | REAL final MAE | Gain vs R06 | Gain vs BAG | Gain vs SHUFFLED |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| E67 H12 | {e67_h12['n']} | {fmt(e67_h12['r06_mae'])} | {fmt(e67_h12['final_policy_mae'])} | {float(e67_h12['gain_vs_r06']):+.5f} | {float(e67_h12['gain_vs_BAG']):+.5f} | {float(e67_h12['gain_vs_SHUFFLED']):+.5f} |",
        f"| E67 W6 | {e67_w6['n']} | {fmt(e67_w6['r06_mae'])} | {fmt(e67_w6['final_policy_mae'])} | {float(e67_w6['gain_vs_r06']):+.5f} | {float(e67_w6['gain_vs_BAG']):+.5f} | {float(e67_w6['gain_vs_SHUFFLED']):+.5f} |",
        f"| E75 W6 | {e75_w6['n']} | {fmt(e75_w6['r06_mae'])} | {fmt(e75_w6['final_policy_mae'])} | {float(e75_w6['gain_vs_r06']):+.5f} | {float(e75_w6['gain_vs_BAG']):+.5f} | {float(e75_w6['gain_vs_SHUFFLED']):+.5f} |",
        "",
        "E67 H12's +.02156 gain misses its frozen +.03 R06 gate. E75 W6 harms R06 by .06481; BAG and SHUFFLED also harm R06, and REAL trails them. Each of cycles 76–79 harms R06 at E75. This preserves the local FAIL.",
        "",
        "### Fixed tail groups and same-cycle changes",
        "",
        *group_md,
        "",
        "E67 H12's upper tail (target ≥95) is down .08108 while its low and middle groups are slightly up. All three E75 W6 target groups are down; the largest harm is .13675 for target ≤5. These group values are descriptive and do not override the primary gate.",
        "",
        *cycle_md,
        "",
        "On matched W6, R06 MAE falls by 1.73184 from E67 to E75; the direct raw-head MAE falls by 1.55940, so its deficit relative to R06 widens by .17244. Rank50 reduces auxiliary MAE by .29974 at E67 W6 and .09378 at E75 W6, but the mapped head remains worse than R06 in both views.",
        "",
        f"For present rows, the saved policy correction has the same sign as the R06 residual on {correction_summary['67']['same_sign_fraction_weighted']:.3f} of E67 W6 rows and {correction_summary['75']['same_sign_fraction_weighted']:.3f} of E75 W6 rows. Per-cycle correction/residual correlations range {correction_summary['67']['cycle_correlation_min']:.3f} to {correction_summary['67']['cycle_correlation_max']:.3f} (E67) and {correction_summary['75']['cycle_correlation_min']:.3f} to {correction_summary['75']['cycle_correlation_max']:.3f} (E75). This is weak directional alignment, not a stable correction mechanism.",
        "",
        f"A deterministic 2,000-resample, seed-42 cycle-cluster bootstrap over the eight-cycle late-unique view gives row-weighted gain {cluster_bootstrap['point_row_weighted_mean_gain']:+.5f}, with descriptive percentile interval [{cluster_bootstrap['percentile_95_interval'][0]:+.5f}, {cluster_bootstrap['percentile_95_interval'][1]:+.5f}]. It spans zero and is not public/private uncertainty or a significance test.",
        "",
        "## Interpretation and prior STOP language",
        "",
        "Sampling, transport, and submission selection can plausibly explain why one public score looks better while the local pilot fails, but the evidence cannot identify which mechanism did so. The observed comparison mixes different trained prefixes and different forecast windows, and public sample composition is unknown. A recipe mismatch or raw-vs-final R06 baseline mismatch is not supported by the pinned artifacts.",
        "",
        "The earlier `NONE_JUSTIFIED` / STOP conclusion is sound when read as: the frozen local gate failed, and this post-hoc decomposition does not justify another fit, retune, or promotion. It would overstate the evidence if read as proof that the recipe cannot transfer, that B's public score is false, or that the local FAIL should be replaced by PASS.",
        "",
        "## Decision by evidence layer",
        "",
        "- **Mechanism:** local E75 evidence shows baseline catch-up and weak correction alignment; it does not isolate a causal positional-text effect.",
        "- **Internal risk gate:** direct-position candidate remains **FAIL / no promotion** under the frozen pilot gate.",
        "- **Benchmark admission:** B is a completed E79 public benchmark record, explicitly marked exploratory with prior local gates failed.",
        "- **Public-best status:** B is lowest among the 18 completed regression records in this dated snapshot; private score and generalization remain unknown.",
        "",
        "No blanket PASS, significance claim, or private-score inference is warranted.",
        "",
        "## Reproduction artifacts",
        "",
        "Run `python reports/optimization_v3/public_local_forensic_v1/reg/public_local_forensic_v1.py --repo-root <owned-worktree> --source-root <TP-2-root> --saved-artifact-root <saved-artifact-root>`.",
        "",
        "- `public_local_metrics.csv`, `cycle_metrics.csv`, `group_metrics.csv`, and `correction_diagnostics.csv` contain the detailed tables.",
        "- `feature_state_comparison.json`, `cluster_bootstrap.json`, and `public_snapshot_summary.json` hold the state comparison and summary diagnostics.",
        "- `evidence_pins.json` lists every absolute input path and verified SHA-256; `output_hashes.json` pins generated files.",
        "",
    ]
    (out / "REG_PUBLIC_LOCAL_FORENSIC.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(out / "evidence_pins.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(repo), "source_root": str(source), "saved_artifact_root": str(saved),
        "write_scope": str(out), "pins": pins,
        "access_boundary": {
            "fit_calls": 0, "predict_calls": 0, "test_labels_loaded": False,
            "sample_submission_opened": False, "new_submission_or_csv_opened": False,
            "kaggle_accessed_by_script": False, "train_rows_parsed": int(len(parsed_rows)),
            "fit_prefix_target_rows_parsed": 0,
        },
    })
    outputs = {}
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name != "output_hashes.json":
            outputs[p.name] = sha_file(p)
    write_json(out / "output_hashes.json", outputs)
    print(json.dumps({"output_root": str(out), "outputs": outputs,
                      "public_summary": public_summary,
                      "bootstrap": cluster_bootstrap,
                      "metric_checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
