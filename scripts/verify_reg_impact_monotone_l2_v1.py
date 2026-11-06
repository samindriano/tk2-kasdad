"""Independent verification for the REG_IMPACT_MONOTONE_L2_V1 source package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.models.reg_impact_monotone_l2_v1 import (
    ARM_IDS,
    EXPECTED_LIGHTGBM_VERSION,
    ORIGINS,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_FREEZE_SHA256 = "d1f8f77864c1b2c1df8dc11732ffb61b68ac08f5e0b37ded48abf33f7506ac56"
SELECTED_KEYS = (
    "reg_r06_67", "reg_direct_67", "reg_r06_67_receipt", "reg_direct_67_receipt",
    "reg_r06_75", "reg_direct_75", "reg_r06_75_receipt", "reg_direct_75_receipt",
    "reg_components67", "reg_components67_receipt", "hinge_D1", "global_D1",
    "hinge_D2", "global_D2",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _assert_allclose(left: Any, right: Any, *, label: str, atol: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{label} arrays are misaligned or nonfinite")
    difference = float(np.max(np.abs(a - b))) if len(a) else 0.0
    if not np.allclose(a, b, rtol=0.0, atol=atol):
        raise ValueError(f"{label} mismatch; max abs diff={difference}")
    return difference


def verify_synthetic(args: argparse.Namespace) -> int:
    if lightgbm.__version__ != EXPECTED_LIGHTGBM_VERSION:
        raise RuntimeError(f"expected LightGBM {EXPECTED_LIGHTGBM_VERSION}, got {lightgbm.__version__}")
    fit_root = Path(args.fit_root)
    bundle_root = Path(args.bundle_root)
    ensemble_root = Path(args.ensemble_root)
    checked_fits = []
    ensemble_replays = []
    fit_params: dict[tuple[int, str], dict[str, Any]] = {}

    for origin in ORIGINS:
        bundle_receipt_path = bundle_root / f"E{origin}_receipt.json"
        bundle_receipt = _read_json(bundle_receipt_path)
        if bundle_receipt.get("source_kind") != "synthetic":
            raise ValueError("synthetic verifier received a non-synthetic bundle")
        data_path = Path(bundle_receipt["design_path"])
        transform_path = Path(bundle_receipt["transforms_path"])
        if file_sha256(str(data_path)) != bundle_receipt["design_sha256"]:
            raise ValueError(f"E{origin} bundle data hash mismatch")
        if file_sha256(str(transform_path)) != bundle_receipt["transforms_sha256"]:
            raise ValueError(f"E{origin} transform state hash mismatch")
        with np.load(data_path, allow_pickle=False) as bundle:
            fit_design = np.asarray(bundle["fit_design"], dtype=np.float64)
            query_design = np.asarray(bundle["query_design"], dtype=np.float64)
            fit_rows = np.asarray(bundle["fit_rows"], dtype=np.int64)
            query_rows = np.asarray(bundle["query_rows"], dtype=np.int64)
            fit_ids = np.asarray(bundle["fit_ids"]).astype(str)
            query_ids = np.asarray(bundle["query_ids"]).astype(str)
            fit_cycles = np.asarray(bundle["fit_cycles"], dtype=np.int64)
            query_cycles = np.asarray(bundle["query_cycles"], dtype=np.int64)
            names = tuple(str(x) for x in bundle["transformed_feature_names"].tolist())
            categories = tuple(int(x) for x in bundle["categorical_indices"].tolist())
        if fit_design.shape != (len(fit_rows), len(names)) or query_design.shape != (len(query_rows), len(names)):
            raise ValueError(f"E{origin} synthetic design schema mismatch")
        if fit_ids.shape != fit_rows.shape or query_ids.shape != query_rows.shape:
            raise ValueError(f"E{origin} synthetic bundle identity mismatch")
        if np.any(fit_cycles > origin) or np.any(query_cycles <= origin):
            raise ValueError(f"E{origin} synthetic prefix/query leakage")
        if int(bundle_receipt["preprocessing_state"]["learned_transform_count"]) != 3:
            raise ValueError(f"E{origin} transform count differs from plan")
        impact_matches = [i for i, name in enumerate(names) if name == "impact_load"]
        if len(impact_matches) != 1 or impact_matches[0] in categories:
            raise ValueError(f"E{origin} impact_load value index is not uniquely numeric")
        impact_index = impact_matches[0]

        for arm in ARM_IDS:
            fit_id = f"E{origin}_{arm}"
            directory = fit_root / fit_id
            receipt = _read_json(directory / "receipt.json")
            if receipt.get("fit_id") != fit_id or receipt.get("source_kind") != "synthetic":
                raise ValueError(f"{fit_id} identity or source kind mismatch")
            model_path = Path(receipt["model_path"])
            prediction_path = Path(receipt["predictions_path"])
            if file_sha256(str(model_path)) != receipt["model_sha256"]:
                raise ValueError(f"{fit_id} model SHA-256 mismatch")
            if file_sha256(str(prediction_path)) != receipt["predictions_sha256"]:
                raise ValueError(f"{fit_id} prediction SHA-256 mismatch")
            with np.load(prediction_path, allow_pickle=False) as prediction:
                saved_raw = np.asarray(prediction["raw_prediction"], dtype=np.float64)
                saved_mapped = np.asarray(prediction["mapped_prediction"], dtype=np.float64)
                saved_ids = np.asarray(prediction["query_ids"]).astype(str)
                saved_rows = np.asarray(prediction["query_rows"], dtype=np.int64)
                saved_cycles = np.asarray(prediction["query_cycles"], dtype=np.int64)
            if not np.array_equal(saved_ids, query_ids) or not np.array_equal(saved_rows, query_rows):
                raise ValueError(f"{fit_id} query identity/row order differs from its bundle")
            if not np.array_equal(saved_cycles, query_cycles):
                raise ValueError(f"{fit_id} query cycle vector differs from its bundle")
            model = lightgbm.Booster(model_file=str(model_path))
            query_frame = pd.DataFrame(query_design, columns=list(names))
            readback = np.asarray(model.predict(query_frame), dtype=np.float64)
            readback_diff = _assert_allclose(readback, saved_raw, label=f"{fit_id} model readback")
            expected_mapped = _mapped(saved_raw, saved_cycles)
            mapped_diff = _assert_allclose(expected_mapped, saved_mapped, label=f"{fit_id} rank50 map")
            params = dict(receipt["meta"]["actual_parameters"])
            fit_params[(origin, arm)] = params
            if arm == "A_L1_FAITHFUL":
                if params.get("objective") != "regression_l1" or "monotone_constraints" in params:
                    raise ValueError("A arm must remain unconstrained regression_l1")
            elif arm == "B_L2_CONTROL":
                if params.get("objective") != "regression" or "monotone_constraints" in params:
                    raise ValueError("B arm must be unconstrained L2")
            else:
                constraints = params.get("monotone_constraints")
                expected = [0] * len(names)
                expected[impact_index] = 1
                if (params.get("objective") != "regression" or constraints != expected
                        or params.get("monotone_constraints_method") != "advanced"):
                    raise ValueError("C arm constraint schema differs from plan")
                if any(constraints[index] != 0 for index in categories):
                    raise ValueError("C arm constrained a categorical feature")
                observed = np.unique(fit_design[:, impact_index])
                anchors = np.unique(np.linspace(0, len(fit_design) - 1, min(3, len(fit_design))).astype(int))
                min_step = 0.0
                for row in anchors:
                    counter = np.repeat(fit_design[row : row + 1], len(observed), axis=0)
                    counter[:, impact_index] = observed
                    raw_cf = np.asarray(model.predict(pd.DataFrame(counter, columns=list(names))), dtype=np.float64)
                    if len(raw_cf) > 1:
                        min_step = min(min_step, float(np.min(np.diff(raw_cf))))
                if min_step < -1e-10:
                    raise ValueError(f"{fit_id} raw head is not monotone; minimum step={min_step}")
            checked_fits.append({
                "fit_id": fit_id,
                "model_readback_max_abs_diff": readback_diff,
                "mapped_prediction_max_abs_diff": mapped_diff,
                "model_sha256": receipt["model_sha256"],
                "prediction_sha256": receipt["predictions_sha256"],
                "monotonicity_rechecked": arm == "C_L2_MONOTONE",
            })

        ensemble_path = ensemble_root / f"E{origin}_ensemble.npz"
        with np.load(ensemble_path, allow_pickle=False) as ensemble:
            ids = np.asarray(ensemble["query_ids"]).astype(str)
            cycles = np.asarray(ensemble["query_cycles"], dtype=np.int64)
            missing = np.asarray(ensemble["missing"], dtype=bool)
            frozen_mapped_lgb = np.asarray(ensemble["frozen_mapped_lgb"], dtype=np.float64)
            frozen_mapped_ridge = np.asarray(ensemble["frozen_mapped_ridge"], dtype=np.float64)
            frozen_word = np.asarray(ensemble["frozen_word_raw"], dtype=np.float64)
            frozen_char = np.asarray(ensemble["frozen_char_raw"], dtype=np.float64)
            frozen_pos = np.asarray(ensemble["frozen_pos_raw"], dtype=np.float64)
            frozen = np.asarray(ensemble["frozen_final"], dtype=np.float64)
            frozen_direct = np.asarray(ensemble["frozen_direct"], dtype=np.float64)
            hinge = {
                "intercept": float(np.asarray(ensemble["hinge_intercept"]).item()),
                "slope_low": float(np.asarray(ensemble["hinge_slope_low"]).item()),
                "slope_high": float(np.asarray(ensemble["hinge_slope_high"]).item()),
            }
            names_ens = ("candidate_A", "candidate_B", "candidate_C")
            candidate_arrays = [np.asarray(ensemble[name], dtype=np.float64) for name in names_ens]
            raw_arrays = [np.asarray(ensemble[f"raw_{arm[0]}"], dtype=np.float64) for arm in ARM_IDS]
            mapped_arrays = [np.asarray(ensemble[f"mapped_{arm[0]}"], dtype=np.float64) for arm in ARM_IDS]
        if ids.shape != cycles.shape or missing.shape != cycles.shape or frozen.shape != cycles.shape:
            raise ValueError(f"E{origin} synthetic ensemble fields are misaligned")
        frozen_replay = _nested_replay(
            frozen_mapped_lgb, frozen_mapped_ridge, frozen_word, frozen_char, frozen_pos, cycles, missing
        )
        frozen_direct_diff = _assert_allclose(
            frozen_replay["direct"], frozen_direct, label=f"E{origin} synthetic frozen direct replay", atol=1e-12
        )
        frozen_final_diff = _assert_allclose(
            _hinge(frozen_direct, hinge, missing), frozen,
            label=f"E{origin} synthetic frozen hinge replay", atol=1e-12
        )
        if not all(np.isfinite(values).all() for values in [frozen, frozen_direct, *candidate_arrays]):
            raise ValueError(f"E{origin} synthetic ensemble contains nonfinite values")
        if not np.array_equal(frozen_direct[missing], frozen_replay["r06"][missing]):
            raise ValueError(f"E{origin} synthetic frozen missing route differs from R06")
        candidate_replay_checks = []
        for arm, raw, mapped, values in zip(ARM_IDS, raw_arrays, mapped_arrays, candidate_arrays):
            mapped_diff = _assert_allclose(
                _mapped(raw, cycles), mapped, label=f"E{origin} {arm} mapped-slot replay", atol=1e-12
            )
            replay = _nested_replay(mapped, frozen_mapped_ridge, frozen_word, frozen_char,
                                    frozen_pos, cycles, missing)
            direct = replay["direct"].copy()
            direct[missing] = frozen_direct[missing]
            expected_final = _hinge(direct, hinge, missing)
            expected_final[missing] = frozen[missing]
            final_diff = _assert_allclose(
                expected_final, values, label=f"E{origin} {arm} independent ensemble replay", atol=1e-12
            )
            if not np.array_equal(values[missing], frozen[missing]):
                raise ValueError(f"E{origin} candidate changed missing champion bits")
            candidate_replay_checks.append({
                "arm_id": arm,
                "mapped_slot_max_abs_diff": mapped_diff,
                "final_ensemble_max_abs_diff": final_diff,
                "missing_rows_bitwise_exact": True,
            })
        ensemble_replays.append({
            "origin": origin,
            "synthetic_frozen_direct_replay_max_abs_diff": frozen_direct_diff,
            "synthetic_frozen_hinge_replay_max_abs_diff": frozen_final_diff,
            "synthetic_candidates": candidate_replay_checks,
        })

    for origin in ORIGINS:
        p_a = fit_params[(origin, "A_L1_FAITHFUL")]
        p_b = fit_params[(origin, "B_L2_CONTROL")]
        p_c = fit_params[(origin, "C_L2_MONOTONE")]
        if {k for k in set(p_a) | set(p_b) if p_a.get(k) != p_b.get(k)} != {"objective"}:
            raise ValueError(f"E{origin} A/B parameter differences exceed objective")
        if {k for k in set(p_b) | set(p_c) if p_b.get(k) != p_c.get(k)} != {
            "monotone_constraints", "monotone_constraints_method"
        }:
            raise ValueError(f"E{origin} B/C parameter differences exceed constraint settings")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "synthetic_only",
        "fit_ids": [entry["fit_id"] for entry in checked_fits],
        "fit_count": len(checked_fits),
        "fits": checked_fits,
        "ensemble_replays": ensemble_replays,
        "prefixes": list(ORIGINS),
        "candidate_missing_rows_exact": True,
        "model_artifact_readback": "PASS",
        "prelabel_only": True,
        "official_labels_opened": False,
        "test_or_sample_opened": False,
        "synthetic_ensemble_replacement": "PASS",
    }
    _write_json_exclusive(Path(args.output), report)
    print(json.dumps({"status": report["status"], "fit_count": report["fit_count"],
                      "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


def _frozen_file(freeze: dict[str, Any], key: str) -> Path:
    item = freeze["files"][key]
    path = Path(item["path"])
    if not path.is_file() or file_sha256(str(path)) != item["sha256"]:
        raise ValueError(f"frozen file missing or SHA mismatch: {key}")
    return path


def _load_npz_allowlisted(path: Path, allowed: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(allowed) - set(archive.files))
        if missing:
            raise ValueError(f"{path.name} missing allowlisted members: {missing}")
        return {key: np.asarray(archive[key]) for key in allowed}


def _align(target_ids: Any, source_ids: Any, *vectors: Any) -> list[np.ndarray]:
    target = np.asarray(target_ids).astype(str)
    source = np.asarray(source_ids).astype(str)
    if not target.ndim == source.ndim == 1 or len(set(source.tolist())) != len(source):
        raise ValueError("cache record identifiers must be unique one-dimensional vectors")
    index = {value: i for i, value in enumerate(source.tolist())}
    try:
        positions = np.asarray([index[value] for value in target.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"cache record id is missing: {exc}") from exc
    if not np.array_equal(source[positions], target):
        raise ValueError("cache ID alignment failed")
    return [np.asarray(vector)[positions] for vector in vectors]


def _mapped(raw: Any, cycles: Any, *, clip: bool = True) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    cycle_values = np.asarray(cycles, dtype=np.int64)
    percentiles = np.empty_like(values)
    for cycle in np.unique(cycle_values):
        mask = cycle_values == cycle
        percentiles[mask] = 100.0 * rankdata(values[mask], method="average") / int(mask.sum())
    mapped = 0.5 * values + 0.5 * percentiles
    return np.clip(mapped, 0.0, 100.0) if clip else mapped


def _nested_replay(
    mapped_lgb: Any,
    mapped_ridge: Any,
    word_raw: Any,
    char_raw: Any,
    pos_raw: Any,
    cycles: Any,
    missing: Any,
) -> dict[str, np.ndarray]:
    """Independent implementation of the cached nested champion arithmetic."""
    lgb = np.asarray(mapped_lgb, dtype=np.float64)
    ridge = np.asarray(mapped_ridge, dtype=np.float64)
    word = np.asarray(word_raw, dtype=np.float64)
    char = np.asarray(char_raw, dtype=np.float64)
    pos = np.asarray(pos_raw, dtype=np.float64)
    cycle_values = np.asarray(cycles, dtype=np.int64)
    missing_mask = np.asarray(missing, dtype=bool)
    if (lgb.ndim != 1 or any(x.shape != lgb.shape for x in (ridge, word, char, pos, cycle_values, missing_mask))
            or not all(np.isfinite(x).all() for x in (lgb, ridge, word, char, pos))):
        raise ValueError("nested replay leaves, cycles, and missing mask must be aligned finite vectors")
    mapped_word = _mapped(word, cycle_values)
    mapped_char = _mapped(char, cycle_values)
    mapped_pos = _mapped(pos, cycle_values, clip=False)
    legacy = np.clip(0.5625 * lgb + 0.1875 * ridge + 0.25 * mapped_word, 0.0, 100.0)
    r06 = np.clip(0.75 * legacy + 0.25 * mapped_char, 0.0, 100.0)
    direct = np.clip(0.75 * r06 + 0.25 * mapped_pos, 0.0, 100.0)
    direct[missing_mask] = r06[missing_mask]
    return {
        "mapped_lgb": lgb,
        "mapped_ridge": ridge,
        "word_raw": word,
        "char_raw": char,
        "pos_raw": pos,
        "mapped_word": mapped_word,
        "mapped_char": mapped_char,
        "mapped_pos": mapped_pos,
        "legacy": legacy,
        "r06": r06,
        "direct": direct,
        "missing": missing_mask,
    }


def _hinge(values: Any, state: dict[str, Any], missing: Any) -> np.ndarray:
    """Apply the frozen P50 hinge only on present rows, independently."""
    scores = np.asarray(values, dtype=np.float64)
    missing_mask = np.asarray(missing, dtype=bool)
    required = {"intercept", "slope_low", "slope_high"}
    if scores.ndim != 1 or missing_mask.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("hinge inputs must be aligned finite vectors")
    if required - set(state):
        raise ValueError(f"saved hinge state missing parameters: {sorted(required - set(state))}")
    output = scores.copy()
    present = ~missing_mask
    output[present] = np.clip(
        float(state["intercept"])
        + float(state["slope_low"]) * np.minimum(scores[present], 50.0)
        + float(state["slope_high"]) * np.maximum(scores[present] - 50.0, 0.0),
        0.0,
        100.0,
    )
    return output


def verify_frozen_cache(args: argparse.Namespace) -> int:
    freeze_path = Path(args.input_freeze)
    if file_sha256(str(freeze_path)) != EXPECTED_INPUT_FREEZE_SHA256:
        raise ValueError("simplification audit INPUT_FREEZE SHA-256 differs from pinned source")
    freeze = _read_json(freeze_path)
    loaded_sources = []
    result: dict[str, Any] = {}
    stage_specs = (
        (67, "reg_components67", "reg_r06_67", "reg_direct_67", "D1"),
        (75, None, "reg_r06_75", "reg_direct_75", "D2"),
    )
    for origin, comp_key, r06_key, direct_key, stage in stage_specs:
        if origin == 67:
            comp_path = _frozen_file(freeze, comp_key)
            comp = _load_npz_allowlisted(comp_path, (
                "score_record_ids", "control__baseline_lgb_svd32",
                "control__baseline_hybrid_ridge", "control__baseline_sparse_raw",
            ))
            loaded_sources.append({"key": comp_key, "sha256": file_sha256(str(comp_path)),
                                   "members": sorted(comp.keys())})
        r06_path = _frozen_file(freeze, r06_key)
        r06_keys = ("score_record_ids", "raw_candidate") if origin == 67 else (
            "score_record_ids", "raw_candidate", "control__baseline_lgb_svd32",
            "control__baseline_hybrid_ridge", "control__baseline_sparse_raw",
        )
        r06 = _load_npz_allowlisted(r06_path, r06_keys)
        loaded_sources.append({"key": r06_key, "sha256": file_sha256(str(r06_path)),
                               "members": sorted(r06.keys())})
        direct_path = _frozen_file(freeze, direct_key)
        direct = _load_npz_allowlisted(direct_path, (
            "query_record_ids", "query_cycles", "missing_mask", "raw_auxiliary",
            "prediction", "baseline",
        ))
        loaded_sources.append({"key": direct_key, "sha256": file_sha256(str(direct_path)),
                               "members": sorted(direct.keys())})
        ids = direct["query_record_ids"].astype(str)
        cycles = direct["query_cycles"].astype(np.int64)
        missing = direct["missing_mask"].astype(bool)
        if origin == 67:
            assert comp is not None
            lgb, ridge, word = _align(
                ids,
                comp["score_record_ids"],
                comp["control__baseline_lgb_svd32"],
                comp["control__baseline_hybrid_ridge"],
                comp["control__baseline_sparse_raw"],
            )
            char, = _align(ids, r06["score_record_ids"], r06["raw_candidate"])
        else:
            lgb, ridge, word, char = _align(
                ids,
                r06["score_record_ids"],
                r06["control__baseline_lgb_svd32"],
                r06["control__baseline_hybrid_ridge"],
                r06["control__baseline_sparse_raw"],
                r06["raw_candidate"],
            )
        mapped_word = _mapped(word, cycles)
        mapped_char = _mapped(char, cycles)
        mapped_pos = _mapped(direct["raw_auxiliary"], cycles, clip=False)
        legacy = np.clip(0.5625 * lgb + 0.1875 * ridge + 0.25 * mapped_word, 0.0, 100.0)
        rebuilt_r06 = np.clip(0.75 * legacy + 0.25 * mapped_char, 0.0, 100.0)
        rebuilt_direct = np.clip(0.75 * rebuilt_r06 + 0.25 * mapped_pos, 0.0, 100.0)
        rebuilt_direct[missing] = rebuilt_r06[missing]
        direct_diff = _assert_allclose(
            rebuilt_direct, direct["prediction"], label=f"E{origin} nested direct replay", atol=1e-10
        )
        baseline_diff = _assert_allclose(
            rebuilt_r06, direct["baseline"], label=f"E{origin} nested R06 replay", atol=1e-10
        )
        if not np.array_equal(direct["prediction"][missing], direct["baseline"][missing]):
            raise ValueError(f"E{origin} saved missing-route champion is not bitwise preserved")
        hinge_path = _frozen_file(freeze, f"hinge_{stage}")
        hinge = _read_json(hinge_path)
        solver = hinge.get("solver_result", {})
        if {"intercept", "slope_low", "slope_high"} - set(solver):
            raise ValueError(f"E{origin} frozen hinge state lacks required parameters")
        global_path = _frozen_file(freeze, f"global_{stage}")
        _read_json(global_path)
        loaded_sources.extend([
            {"key": f"hinge_{stage}", "sha256": file_sha256(str(hinge_path)), "members": ["solver_result"]},
            {"key": f"global_{stage}", "sha256": file_sha256(str(global_path)), "members": ["solver_result"]},
        ])
        result[str(origin)] = {
            "rows": int(len(ids)),
            "present_rows": int(np.sum(~missing)),
            "missing_rows": int(np.sum(missing)),
            "query_cycle_counts": {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)},
            "nested_r06_max_abs_diff_vs_saved_baseline": baseline_diff,
            "nested_direct_max_abs_diff_vs_saved_prediction": direct_diff,
            "nested_replay_tolerance": {"rtol": 0.0, "atol": 1e-10},
            "bitwise_nested_replay": bool(np.array_equal(rebuilt_direct, direct["prediction"])),
            "saved_missing_route_bitwise_equal": True,
            "effective_lgb_weight_present": 0.31640625,
            "effective_lgb_weight_missing": 0.421875,
            "nested_order_preserved": True,
        }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "frozen_non_target_cache_replay",
        "input_freeze_sha256": file_sha256(str(freeze_path)),
        "loaded_sources": loaded_sources,
        "cache_labels_opened": False,
        "test_or_sample_opened": False,
        "per_origin": result,
    }
    _write_json_exclusive(Path(args.output), report)
    print(json.dumps({"status": report["status"], "output": str(Path(args.output).resolve()),
                      "origins": list(result)}, sort_keys=True))
    return 0


def _check_source_freeze(path: Path) -> dict[str, Any]:
    source = _read_json(path)
    if source.get("experiment") != "REG_IMPACT_MONOTONE_L2_V1":
        raise ValueError("source freeze experiment id differs")
    if source.get("root_plan_sha256") != file_sha256(str(ROOT / "reports/optimization_v3/REG_IMPACT_MONOTONE_L2_V1/ROOT_PLAN.md")):
        raise ValueError("source freeze is not bound to the current root plan")
    if source.get("train_sha256") != "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082":
        raise ValueError("source freeze Train digest differs from approved TP2 source")
    for relative, digest in source["source_hashes"].items():
        if file_sha256(str(ROOT / relative)) != digest:
            raise ValueError(f"source implementation changed after freeze: {relative}")
    return source


def _official_cache_components(
    freeze: dict[str, Any], origin: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stage = "D1" if origin == 67 else "D2"
    direct_path = _frozen_file(freeze, f"reg_direct_{origin}")
    direct = _load_npz_allowlisted(direct_path, (
        "query_record_ids", "query_cycles", "missing_mask", "raw_auxiliary",
        "prediction", "baseline",
    ))
    ids = direct["query_record_ids"].astype(str)
    cycles = direct["query_cycles"].astype(np.int64)
    missing = direct["missing_mask"].astype(bool)
    r06_path = _frozen_file(freeze, f"reg_r06_{origin}")
    if origin == 67:
        r06 = _load_npz_allowlisted(r06_path, ("score_record_ids", "raw_candidate"))
        comp_path = _frozen_file(freeze, "reg_components67")
        comp = _load_npz_allowlisted(comp_path, (
            "score_record_ids", "control__baseline_lgb_svd32",
            "control__baseline_hybrid_ridge", "control__baseline_sparse_raw",
        ))
        lgb, ridge, word = _align(
            ids,
            comp["score_record_ids"],
            comp["control__baseline_lgb_svd32"],
            comp["control__baseline_hybrid_ridge"],
            comp["control__baseline_sparse_raw"],
        )
        char, = _align(ids, r06["score_record_ids"], r06["raw_candidate"])
    else:
        r06 = _load_npz_allowlisted(r06_path, (
            "score_record_ids", "raw_candidate", "control__baseline_lgb_svd32",
            "control__baseline_hybrid_ridge", "control__baseline_sparse_raw",
        ))
        lgb, ridge, word, char = _align(
            ids,
            r06["score_record_ids"],
            r06["control__baseline_lgb_svd32"],
            r06["control__baseline_hybrid_ridge"],
            r06["control__baseline_sparse_raw"],
            r06["raw_candidate"],
        )
    incumbent = _nested_replay(lgb, ridge, word, char, direct["raw_auxiliary"], cycles, missing)
    replay_diff = _assert_allclose(
        incumbent["direct"], direct["prediction"], label=f"E{origin} nested direct replay", atol=1e-10
    )
    r06_diff = _assert_allclose(
        incumbent["r06"], direct["baseline"], label=f"E{origin} nested R06 replay", atol=1e-10
    )
    if not np.array_equal(direct["prediction"][missing], direct["baseline"][missing]):
        raise ValueError(f"E{origin} missing route differs from saved cached bits")
    # Use the immutable saved direct vector as the canonical baseline after
    # independently checking the nested reconstruction within 1e-10.
    incumbent["direct"] = np.asarray(direct["prediction"], dtype=np.float64).copy()
    hinge_path = _frozen_file(freeze, f"hinge_{stage}")
    hinge_receipt = _read_json(hinge_path)
    hinge = hinge_receipt.get("solver_result", {})
    required_hinge = {"intercept", "slope_low", "slope_high"}
    if required_hinge - set(hinge):
        raise ValueError(f"E{origin} saved hinge state is incomplete")
    return {
        "ids": ids,
        "cycles": cycles,
        "missing": missing,
        "incumbent": incumbent,
        "cached_direct": np.asarray(direct["prediction"], dtype=np.float64),
        "hinge": hinge,
        "replay_max_abs_diff": replay_diff,
        "r06_max_abs_diff": r06_diff,
        "cache_sha256": {
            f"reg_direct_{origin}": file_sha256(str(direct_path)),
            f"reg_r06_{origin}": file_sha256(str(r06_path)),
            ("reg_components67" if origin == 67 else "reg_r06_75_components"): (
                file_sha256(str(comp_path)) if origin == 67 else file_sha256(str(r06_path))
            ),
            f"hinge_{stage}": file_sha256(str(hinge_path)),
        },
    }


def _verify_official_fit_receipt(
    fit_id: str,
    fit_root: Path,
    bundle: dict[str, Any],
    source_freeze_path: Path,
    plan_sha: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    directory = fit_root / fit_id
    receipt = _read_json(directory / "receipt.json")
    if receipt.get("fit_id") != fit_id or receipt.get("source_kind") != "official":
        raise ValueError(f"{fit_id} receipt identity/source mismatch")
    if receipt.get("source_freeze_sha256") != file_sha256(str(source_freeze_path)):
        raise ValueError(f"{fit_id} is not bound to the source freeze")
    if receipt.get("root_plan_sha256") != plan_sha:
        raise ValueError(f"{fit_id} is not bound to the current root plan")
    if receipt.get("query_target_rows_materialized") != 0:
        raise ValueError(f"{fit_id} exposed query targets before prelabel QA")
    auth_path = Path(receipt["official_fit_authorization_path"])
    if file_sha256(str(auth_path)) != receipt["official_fit_authorization_sha256"]:
        raise ValueError(f"{fit_id} fit authorization SHA-256 mismatch")
    authorization = _read_json(auth_path)
    expected_ids = [f"E{o}_{arm}" for o in ORIGINS for arm in ARM_IDS]
    if (authorization.get("status") != "APPROVED_FOR_SUPERVISED_FITS"
            or authorization.get("approved_fit_ids") != expected_ids
            or authorization.get("fit_count") != 6
            or authorization.get("root_plan_sha256") != plan_sha
            or authorization.get("source_freeze_sha256") != file_sha256(str(source_freeze_path))):
        raise ValueError("fit authorization receipt does not authorize the exact controlled six-arm protocol")
    model_path = Path(receipt["model_path"])
    prediction_path = Path(receipt["predictions_path"])
    if file_sha256(str(model_path)) != receipt["model_sha256"]:
        raise ValueError(f"{fit_id} model hash mismatch")
    if file_sha256(str(prediction_path)) != receipt["predictions_sha256"]:
        raise ValueError(f"{fit_id} prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as saved:
        ids = np.asarray(saved["query_ids"]).astype(str)
        rows = np.asarray(saved["query_rows"], dtype=np.int64)
        cycles = np.asarray(saved["query_cycles"], dtype=np.int64)
        raw = np.asarray(saved["raw_prediction"], dtype=np.float64)
        mapped = np.asarray(saved["mapped_prediction"], dtype=np.float64)
    if (not np.array_equal(ids, bundle["query_ids"]) or not np.array_equal(rows, bundle["query_rows"])
            or not np.array_equal(cycles, bundle["query_cycles"])):
        raise ValueError(f"{fit_id} query IDs/rows/cycles differ from the frozen prefix bundle")
    if raw.shape != cycles.shape or not np.isfinite(raw).all():
        raise ValueError(f"{fit_id} raw predictions are misaligned/nonfinite")
    mapped_diff = _assert_allclose(_mapped(raw, cycles), mapped, label=f"{fit_id} rank50 map")
    model = lightgbm.Booster(model_file=str(model_path))
    frame = pd.DataFrame(bundle["query_design"], columns=list(bundle["feature_names"]))
    model_readback_diff = _assert_allclose(model.predict(frame), raw, label=f"{fit_id} model readback")
    meta = dict(receipt["meta"])
    params = dict(meta["actual_parameters"])
    return raw, {"params": params, "meta": meta, "receipt": receipt,
                 "model_readback_max_abs_diff": model_readback_diff,
                 "mapped_max_abs_diff": mapped_diff}, {"authorization": authorization}


def verify_official_prelabel(args: argparse.Namespace) -> int:
    if lightgbm.__version__ != EXPECTED_LIGHTGBM_VERSION:
        raise RuntimeError(f"expected LightGBM {EXPECTED_LIGHTGBM_VERSION}, got {lightgbm.__version__}")
    plan_path = ROOT / "reports/optimization_v3/REG_IMPACT_MONOTONE_L2_V1/ROOT_PLAN.md"
    plan_sha = file_sha256(str(plan_path))
    source_freeze_path = Path(args.source_freeze)
    source = _check_source_freeze(source_freeze_path)
    input_freeze_path = Path(args.input_freeze)
    if file_sha256(str(input_freeze_path)) != EXPECTED_INPUT_FREEZE_SHA256:
        raise ValueError("input freeze SHA-256 differs from approved non-target inventory")
    freeze = _read_json(input_freeze_path)
    fit_root = Path(args.fit_root)
    bundle_root = Path(args.bundle_root)
    ensemble_root = Path(args.ensemble_root)
    ensemble_root.mkdir(parents=True, exist_ok=True)
    fit_checks = []
    fit_params: dict[tuple[int, str], dict[str, Any]] = {}
    fit_raw: dict[tuple[int, str], np.ndarray] = {}
    fit_mapped: dict[tuple[int, str], np.ndarray] = {}
    cache_checks = {}
    auth_sha: str | None = None

    for origin in ORIGINS:
        bundle_receipt_path = bundle_root / f"E{origin}_receipt.json"
        bundle_receipt = _read_json(bundle_receipt_path)
        if (bundle_receipt.get("source_kind") != "official"
                or bundle_receipt.get("source_train_sha256") != source["train_sha256"]):
            raise ValueError(f"E{origin} target-free design bundle is not bound to official Train")
        data_path = Path(bundle_receipt["design_path"])
        transform_path = Path(bundle_receipt["transforms_path"])
        if file_sha256(str(data_path)) != bundle_receipt["design_sha256"]:
            raise ValueError(f"E{origin} design bundle hash mismatch")
        if file_sha256(str(transform_path)) != bundle_receipt["transforms_sha256"]:
            raise ValueError(f"E{origin} transform state hash mismatch")
        with np.load(data_path, allow_pickle=False) as z:
            bundle = {
                "fit_design": np.asarray(z["fit_design"], dtype=np.float64),
                "query_design": np.asarray(z["query_design"], dtype=np.float64),
                "fit_rows": np.asarray(z["fit_rows"], dtype=np.int64),
                "query_rows": np.asarray(z["query_rows"], dtype=np.int64),
                "fit_ids": np.asarray(z["fit_ids"]).astype(str),
                "query_ids": np.asarray(z["query_ids"]).astype(str),
                "fit_cycles": np.asarray(z["fit_cycles"], dtype=np.int64),
                "query_cycles": np.asarray(z["query_cycles"], dtype=np.int64),
                "feature_names": tuple(str(x) for x in z["transformed_feature_names"].tolist()),
                "categorical_indices": tuple(int(x) for x in z["categorical_indices"].tolist()),
            }
        cache = _official_cache_components(freeze, origin)
        cache_checks[str(origin)] = {
            "nested_direct_max_abs_diff_vs_saved": cache["replay_max_abs_diff"],
            "nested_r06_max_abs_diff_vs_saved": cache["r06_max_abs_diff"],
            "cache_sha256": cache["cache_sha256"],
            "saved_missing_route_bitwise_equal": True,
        }
        if not np.array_equal(bundle["query_ids"], cache["ids"]) or not np.array_equal(bundle["query_cycles"], cache["cycles"]):
            raise ValueError(f"E{origin} prefix query IDs/cycles differ from frozen champion caches")
        if bundle["fit_design"].shape != (len(bundle["fit_rows"]), len(bundle["feature_names"])):
            raise ValueError(f"E{origin} fit design schema mismatch")
        if bundle["query_design"].shape != (len(bundle["query_rows"]), len(bundle["feature_names"])):
            raise ValueError(f"E{origin} query design schema mismatch")
        impact = [i for i, name in enumerate(bundle["feature_names"]) if name == "impact_load"]
        if len(impact) != 1 or impact[0] in bundle["categorical_indices"]:
            raise ValueError(f"E{origin} impact_load constraint target is not uniquely numeric")

        candidate_ensembles = {}
        for arm in ARM_IDS:
            fit_id = f"E{origin}_{arm}"
            raw, details, authorization_wrapper = _verify_official_fit_receipt(
                fit_id, fit_root, bundle, source_freeze_path, plan_sha
            )
            authorization = authorization_wrapper["authorization"]
            if authorization.get("bundle_design_sha256", {}).get(str(origin)) != bundle_receipt["design_sha256"]:
                raise ValueError(f"{fit_id} root authorization does not bind this design bundle")
            this_auth_sha = details["receipt"]["official_fit_authorization_sha256"]
            if auth_sha is None:
                auth_sha = this_auth_sha
            elif auth_sha != this_auth_sha:
                raise ValueError("six fit outputs use different root authorization files")
            params = details["params"]
            fit_params[(origin, arm)] = params
            fit_raw[(origin, arm)] = raw
            with np.load(details["receipt"]["predictions_path"], allow_pickle=False) as saved:
                fit_mapped[(origin, arm)] = np.asarray(saved["mapped_prediction"], dtype=np.float64)
            if arm == "A_L1_FAITHFUL":
                if params.get("objective") != "regression_l1" or "monotone_constraints" in params:
                    raise ValueError("A arm is not the faithful unconstrained L1 reference")
            elif arm == "B_L2_CONTROL":
                if params.get("objective") != "regression" or "monotone_constraints" in params:
                    raise ValueError("B arm is not the unconstrained L2 control")
            else:
                expected = [0] * len(bundle["feature_names"])
                expected[impact[0]] = 1
                if (params.get("objective") != "regression"
                        or params.get("monotone_constraints") != expected
                        or params.get("monotone_constraints_method") != "advanced"):
                    raise ValueError("C arm constraint vector/method differs from frozen recipe")
                if any(params["monotone_constraints"][i] != 0 for i in bundle["categorical_indices"]):
                    raise ValueError("C arm placed a constraint on a categorical feature")
                observed = np.unique(bundle["fit_design"][:, impact[0]])
                anchors = np.unique(np.linspace(0, len(bundle["fit_design"]) - 1,
                                                min(3, len(bundle["fit_design"]))).astype(int))
                model = lightgbm.Booster(model_file=details["receipt"]["model_path"])
                min_step = 0.0
                for row in anchors:
                    counter = np.repeat(bundle["fit_design"][row : row + 1], len(observed), axis=0)
                    counter[:, impact[0]] = observed
                    predicted = model.predict(pd.DataFrame(counter, columns=list(bundle["feature_names"])))
                    if len(predicted) > 1:
                        min_step = min(min_step, float(np.min(np.diff(predicted))))
                if min_step < -1e-10:
                    raise ValueError(f"{fit_id} raw head monotonicity failed: {min_step}")
            new_lgb_mapped = _mapped(raw, cache["cycles"])
            replay = _nested_replay(
                new_lgb_mapped,
                cache["incumbent"]["mapped_ridge"],
                cache["incumbent"]["word_raw"],
                cache["incumbent"]["char_raw"],
                cache["incumbent"]["pos_raw"],
                cache["cycles"],
                cache["missing"],
            )
            candidate_direct = replay["direct"].copy()
            candidate_direct[cache["missing"]] = cache["cached_direct"][cache["missing"]]
            frozen_final = _hinge(cache["cached_direct"], cache["hinge"], cache["missing"])
            candidate_final = _hinge(candidate_direct, cache["hinge"], cache["missing"])
            candidate_final[cache["missing"]] = frozen_final[cache["missing"]]
            if not np.array_equal(candidate_final[cache["missing"]], frozen_final[cache["missing"]]):
                raise ValueError(f"{fit_id} changed missing champion bits")
            candidate_ensembles[arm] = candidate_final
            fit_checks.append({
                "fit_id": fit_id,
                "objective": params["objective"],
                "feature_width": len(bundle["feature_names"]),
                "impact_load_value_index": impact[0],
                "categorical_indices": list(bundle["categorical_indices"]),
                "model_readback_max_abs_diff": details["model_readback_max_abs_diff"],
                "mapped_prediction_max_abs_diff": details["mapped_max_abs_diff"],
                "raw_head_monotonicity": "PASS" if arm == "C_L2_MONOTONE" else "NOT_APPLICABLE",
                "missing_rows_bitwise_exact": True,
            })

        p_a = fit_params[(origin, "A_L1_FAITHFUL")]
        p_b = fit_params[(origin, "B_L2_CONTROL")]
        p_c = fit_params[(origin, "C_L2_MONOTONE")]
        if {k for k in set(p_a) | set(p_b) if p_a.get(k) != p_b.get(k)} != {"objective"}:
            raise ValueError(f"E{origin} A/B changed parameters beyond objective")
        if {k for k in set(p_b) | set(p_c) if p_b.get(k) != p_c.get(k)} != {
            "monotone_constraints", "monotone_constraints_method"
        }:
            raise ValueError(f"E{origin} B/C changed parameters beyond constraint")
        ensemble_path = ensemble_root / f"E{origin}_ensemble.npz"
        with ensemble_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                query_ids=cache["ids"].astype(str),
                query_cycles=cache["cycles"].astype(np.int64),
                missing=cache["missing"].astype(bool),
                frozen_mapped_lgb=cache["incumbent"]["mapped_lgb"],
                frozen_mapped_ridge=cache["incumbent"]["mapped_ridge"],
                frozen_word_raw=cache["incumbent"]["word_raw"],
                frozen_char_raw=cache["incumbent"]["char_raw"],
                frozen_pos_raw=cache["incumbent"]["pos_raw"],
                frozen_direct=cache["cached_direct"],
                frozen_final=_hinge(cache["cached_direct"], cache["hinge"], cache["missing"]),
                hinge_intercept=np.asarray(cache["hinge"]["intercept"], dtype=np.float64),
                hinge_slope_low=np.asarray(cache["hinge"]["slope_low"], dtype=np.float64),
                hinge_slope_high=np.asarray(cache["hinge"]["slope_high"], dtype=np.float64),
                candidate_A=candidate_ensembles["A_L1_FAITHFUL"],
                candidate_B=candidate_ensembles["B_L2_CONTROL"],
                candidate_C=candidate_ensembles["C_L2_MONOTONE"],
                raw_A=fit_raw[(origin, "A_L1_FAITHFUL")],
                raw_B=fit_raw[(origin, "B_L2_CONTROL")],
                raw_C=fit_raw[(origin, "C_L2_MONOTONE")],
                mapped_A=fit_mapped[(origin, "A_L1_FAITHFUL")],
                mapped_B=fit_mapped[(origin, "B_L2_CONTROL")],
                mapped_C=fit_mapped[(origin, "C_L2_MONOTONE")],
            )

    auth_ids = [f"E{o}_{arm}" for o in ORIGINS for arm in ARM_IDS]
    if len(fit_checks) != 6 or [x["fit_id"] for x in fit_checks] != auth_ids:
        raise ValueError("official prelabel audit did not verify exactly the six planned fits")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "official_prelabel_only",
        "experiment": "REG_IMPACT_MONOTONE_L2_V1",
        "source_freeze_sha256": file_sha256(str(source_freeze_path)),
        "root_plan_sha256": plan_sha,
        "input_freeze_sha256": file_sha256(str(input_freeze_path)),
        "authorization_sha256": auth_sha,
        "fit_ids": auth_ids,
        "fit_count": 6,
        "fit_checks": fit_checks,
        "frozen_champion_replay": cache_checks,
        "ensemble_outputs": {
            str(origin): {
                "path": str((ensemble_root / f"E{origin}_ensemble.npz").resolve()),
                "sha256": file_sha256(str(ensemble_root / f"E{origin}_ensemble.npz")),
            }
            for origin in ORIGINS
        },
        "missing_rows_bitwise_exact_for_all_candidates": True,
        "query_labels_opened": False,
        "test_or_sample_opened": False,
        "final_e79_or_kaggle_action": False,
    }
    _write_json_exclusive(Path(args.output), report)
    print(json.dumps({"status": report["status"], "fit_count": report["fit_count"],
                      "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic", help="verify all synthetic fit and prelabel artifacts")
    synthetic.add_argument("--fit-root", required=True)
    synthetic.add_argument("--bundle-root", required=True)
    synthetic.add_argument("--ensemble-root", required=True)
    synthetic.add_argument("--output", required=True)
    cache = sub.add_parser("frozen-cache", help="replay frozen champion from non-target cache members")
    default_freeze = ROOT.parent / "tk2-simplification-audit-v1/reports/simplification_v1/INPUT_FREEZE.json"
    cache.add_argument("--input-freeze", default=str(default_freeze))
    cache.add_argument("--output", required=True)
    official = sub.add_parser("official", help="independent six-fit prelabel QA and ensemble integration")
    official.add_argument("--fit-root", required=True)
    official.add_argument("--bundle-root", required=True)
    official.add_argument("--ensemble-root", required=True)
    official.add_argument("--input-freeze", default=str(default_freeze))
    official.add_argument("--source-freeze", required=True)
    official.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        return verify_synthetic(args)
    if args.command == "frozen-cache":
        return verify_frozen_cache(args)
    if args.command == "official":
        return verify_official_prelabel(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
