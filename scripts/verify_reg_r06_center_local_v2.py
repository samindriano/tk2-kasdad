"""Independent source-ready receipt verifier for the R06 center-local transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from scripts import reg_monotone_hinge_p50_v1 as r1
from scripts import reg_r06_center_local_v2 as runner


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise runner.PilotError(f"Expected a JSON object: {path}")
    return value


def _signed(path: Path) -> tuple[dict[str, Any], str]:
    value = _json(path)
    body = dict(value)
    signature = body.pop("signature", None)
    if not isinstance(signature, str) or signature != _sha(body):
        raise runner.PilotError(f"Independent signature mismatch: {path}")
    return value, _file_sha(path)


def _load_selected(path: Path, selected: tuple[str, ...], *,
                   allowed_unloaded: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = set(selected) | set(allowed_unloaded)
        if names != required:
            raise runner.PilotError(
                f"Independent NPZ schema mismatch at {path}: expected={sorted(required)} got={sorted(names)}")
        return {name: np.asarray(archive[name]).copy() for name in selected}


def _head(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise runner.PilotError("Cannot independently resolve source commit")
    return result.stdout.strip()


def _identity_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return r1.row_identity_hashes(
        physical_rows=arrays["physical_rows"], record_ids=arrays["record_ids"],
        prediction_origins=arrays["prediction_origins"],
        original_bundle_indices=arrays["original_bundle_indices"],
        cycles=arrays["cycles"], direct_predictions=arrays["direct_predictions"],
        missing_mask=arrays["missing_mask"])


def _same(left: np.ndarray, right: np.ndarray, name: str) -> None:
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind == "f" and b.dtype.kind == "f" and a.dtype.itemsize == b.dtype.itemsize == 8:
        equal = a.shape == b.shape and np.array_equal(a.astype("<f8", copy=False).view("<u8"),
                                                       b.astype("<f8", copy=False).view("<u8"))
    else:
        equal = a.shape == b.shape and np.array_equal(a, b)
    if not equal:
        raise runner.PilotError(f"Independent R06 row replay differs: {name}")


def _verify_freeze(layout: runner.Layout) -> tuple[dict[str, Any], str]:
    freeze, digest = _signed(layout.freeze)
    body = dict(freeze)
    body.pop("signature")
    expected_keys: set[str] = set()
    for category, paths in (("source", layout.source_paths),
                            ("reference_source", layout.reference_paths),
                            ("input", layout.input_paths)):
        for name, path in paths.items():
            key = f"{category}:{name}"
            expected_keys.add(key)
            record = freeze.get("source_and_input_files", {}).get(key, {})
            resolved = Path(path).resolve()
            if (Path(record.get("path", "")).resolve() != resolved
                    or not resolved.is_file() or record.get("sha256") != _file_sha(resolved)):
                raise runner.PilotError(f"Independent frozen file pin differs: {key}")
    for name, path in runner._expected_novelty_paths(layout).items():
        key = f"novelty_summary:{name}"
        expected_keys.add(key)
        record = freeze.get("source_and_input_files", {}).get(key, {})
        resolved = Path(path).resolve()
        if (Path(record.get("path", "")).resolve() != resolved or not resolved.is_file()
                or record.get("sha256") != _file_sha(resolved)):
            raise runner.PilotError(f"Independent novelty-source pin differs: {name}")
    if set(freeze.get("source_and_input_files", {})) != expected_keys:
        raise runner.PilotError("Frozen file map contains missing or unexpected entries")
    if (freeze.get("source_commit") != _head(layout.root)
            or freeze.get("status") != "FROZEN_BEFORE_CANDIDATE_TARGET_ACCESS"
            or freeze.get("candidate_targets_loaded") is not False
            or freeze.get("score_targets_loaded") is not False
            or freeze.get("test_or_sample_accessed") is not False
            or freeze.get("e79_accessed") is not False
            or freeze.get("csv_created") is not False
            or freeze.get("kaggle_upload") is not False):
        raise runner.PilotError("Freeze source binding or access-boundary claim differs")
    policy = freeze.get("policy", {})
    if (policy.get("fit_cap") != 5 or policy.get("diagnostic_fit_cap") != 4
            or policy.get("conditional_final_fit_cap") != 1
            or policy.get("conditional_final_authorized") is not False
            or policy.get("base_model_fits") != 0 or policy.get("preprocessing_fits") != 0
            or policy.get("local_center") != 50.0 or policy.get("local_half_width") != 10.0
            or policy.get("delta_bounds") != [-2.0, 0.0]
            or policy.get("cycle_74_harm_mandatory") is not True):
        raise runner.PilotError("Frozen experiment policy differs from the source-only handoff")
    return freeze, digest


def _verify_preflight(layout: runner.Layout, freeze: Mapping[str, Any],
                      freeze_sha: str) -> tuple[dict[str, Any], str, dict[str, np.ndarray]]:
    preflight, preflight_sha = _signed(layout.preflight)
    ids_path = layout.experiment_dir / "FIT_POPULATION_IDS_R2.json"
    ids_body, ids_sha = _signed(ids_path)
    if (preflight.get("status") != "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS"
            or preflight.get("input_freeze_sha256") != freeze_sha
            or preflight.get("source_commit") != freeze.get("source_commit")
            or preflight.get("fit_population_ids_sha256") != ids_sha
            or ids_body.get("input_freeze_sha256") != freeze_sha
            or ids_body.get("source_commit") != freeze.get("source_commit")
            or ids_body.get("conditional_final_authorized") is not False
            or ids_body.get("prior_ledgers_reset") is not False):
        raise runner.PilotError("Non-target preflight or population-ID receipt is not freeze-bound")
    if (preflight.get("target_arrays_present_in_npz_but_loaded") is not False
            or preflight.get("candidate_target_values_loaded") is not False
            or preflight.get("score_target_values_loaded") is not False
            or preflight.get("calibrator_fit_starts") != 0
            or preflight.get("new_ledger_created") is not False
            or preflight.get("test_or_sample_accessed") is not False
            or preflight.get("e79_accessed") is not False
            or preflight.get("csv_created") is not False
            or preflight.get("kaggle_upload") is not False):
        raise runner.PilotError("Non-target preflight claims an out-of-bound action")

    label_path = layout.input_paths["oof_label_identity_bundle"]
    label_fields = tuple(runner.IDENTITY_FIELDS)
    labels = _load_selected(label_path, label_fields, allowed_unloaded=("targets",))
    ids = np.asarray(labels["record_ids"]).astype(str)
    physical = np.asarray(labels["physical_rows"], dtype=np.int64)
    cycles = np.asarray(labels["cycles"], dtype=np.int64)
    origins = np.asarray(labels["prediction_origins"], dtype=np.int64)
    original_direct = np.asarray(labels["direct_predictions"], dtype="<f8")
    missing = np.asarray(labels["missing_mask"], dtype=bool)
    n = len(ids)
    if (n != 20_903 or len(np.unique(ids)) != n or len(np.unique(physical)) != n
            or set(np.unique(cycles).tolist()) != set(range(68, 80))):
        raise runner.PilotError("Independent OOF identity totals or uniqueness differ")

    id_to_row = {str(record_id): index for index, record_id in enumerate(ids)}
    raw_r06 = np.full(n, np.nan, dtype="<f8")
    observed_origins = np.full(n, -1, dtype=np.int16)
    for origin, member, cache_key, allowed in (
            (67, "audit__cycle_code", "r06_fit67_npz", np.arange(68, 76)),
            (75, "score_cycles", "r06_fit75_npz", np.arange(76, 80))):
        unloaded = ("baseline_prediction", "raw_candidate", "control__ridge25_prediction",
            "control__baseline_sparse_raw", "control__cycle_code", "audit__baseline_prediction",
            "audit__candidate_rank50") if origin == 67 else (
            "raw_candidate", "baseline_prediction", "audit__candidate_rank50",
            "control__rows", "control__record_ids", "control__cycle_codes",
            "control__baseline_prediction", "control__baseline_lgb_svd32",
            "control__baseline_hybrid_ridge", "control__baseline_sparse_raw",
            "control__baseline_sparse_rank50", "control__control_rows", "control__cycle_code")
        cache = _load_selected(layout.input_paths[cache_key],
                               ("score_rows", "score_record_ids", "prediction", member),
                               allowed_unloaded=unloaded)
        cache_ids = np.asarray(cache["score_record_ids"]).astype(str)
        cache_rows = np.asarray(cache["score_rows"], dtype=np.int64)
        cache_cycles = np.asarray(cache[member], dtype=np.int64)
        values = np.asarray(cache["prediction"], dtype="<f8")
        selected = np.flatnonzero(np.isin(cache_cycles, allowed))
        actual = {(str(cache_ids[i]), int(cache_cycles[i]), int(cache_rows[i])) for i in selected}
        expected = {(str(ids[i]), int(cycles[i]), int(physical[i])) for i in range(n)
                    if int(cycles[i]) in set(allowed.tolist())}
        if (actual != expected or not np.isfinite(values[selected]).all()
                or np.any((values[selected] < 0.0) | (values[selected] > 100.0))):
            raise runner.PilotError(f"Independent R06 cache identity differs for fit{origin}")
        for i in selected:
            row = id_to_row[str(cache_ids[i])]
            if observed_origins[row] != -1:
                raise runner.PilotError("Independent cache replay found duplicate prediction ownership")
            raw_r06[row] = values[i]
            observed_origins[row] = origin
    if (not np.isfinite(raw_r06).all() or not np.array_equal(observed_origins, origins)
            or not np.array_equal(raw_r06[missing].view("<u8"), original_direct[missing].view("<u8"))):
        raise runner.PilotError("Independent raw R06 predictions or missing fallback differ")

    population_records = {(row.get("fit_id"), row.get("population")): row
                          for row in preflight.get("populations", [])}
    id_populations = ids_body.get("populations", {})
    score_ids: list[str] = []
    score_rows: list[int] = []
    observed: dict[str, np.ndarray] = {"record_ids": ids, "physical_rows": physical,
        "cycles": cycles, "prediction_origins": origins, "direct_predictions": raw_r06,
        "original_direct_predictions": original_direct, "missing_mask": missing,
        "original_bundle_indices": np.asarray(labels["original_bundle_indices"], dtype=np.int64)}
    for stage_id, fit_cycles, score_cycles, score_origin in (
            ("HINGE_D1", set(range(68, 72)), set(range(72, 76)), 67),
            ("HINGE_D2", set(range(68, 76)), set(range(76, 80)), 75)):
        fit_mask = np.isin(cycles, list(fit_cycles)) & (origins == 67)
        score_mask = np.isin(cycles, list(score_cycles)) & (origins == score_origin)
        fit_all = np.flatnonzero(fit_mask)
        fit_present = np.flatnonzero(fit_mask & ~missing)
        fit_missing = np.flatnonzero(fit_mask & missing)
        score = np.flatnonzero(score_mask)
        expected_all, expected_present, expected_score_rows = ((4_950, 4_608, 7_462)
            if stage_id == "HINGE_D1" else (12_412, 11_563, 8_491))
        if (len(fit_all) != expected_all or len(fit_present) != expected_present
                or len(fit_missing) != expected_all - expected_present
                or len(score) != expected_score_rows):
            raise runner.PilotError(f"Independent fit/score population count differs: {stage_id}")
        fit_path, score_path, _ = runner._stage_files(layout, stage_id)
        fit_arrays = _load_selected(fit_path, tuple(runner.IDENTITY_FIELDS))
        score_arrays = _load_selected(score_path, tuple(runner.SCORE_NON_TARGET_FIELDS))
        for name in runner.IDENTITY_FIELDS:
            expected_fit = observed[name][fit_all]
            _same(fit_arrays[name], expected_fit, f"{stage_id}/fit/{name}")
            expected_score = observed[name][score]
            _same(score_arrays[name], expected_score, f"{stage_id}/score/{name}")
        _same(score_arrays["original_direct_predictions"], original_direct[score],
              f"{stage_id}/score/original_direct_predictions")
        slug = "d1" if stage_id == "HINGE_D1" else "d2"
        r4_path = layout.input_paths[f"{slug}_historical_score_rows"]
        r4_fields = ("physical_rows", "record_ids", "prediction_origins",
            "original_bundle_indices", "cycles", "direct_predictions", "missing_mask",
            "hinge_present_only")
        r4_rows = _load_selected(r4_path, r4_fields, allowed_unloaded=(
            "targets", "raw_direct", "causal_global_all_rows", "causal_global_present_only"))
        for name in ("physical_rows", "record_ids", "prediction_origins",
                     "original_bundle_indices", "cycles", "missing_mask"):
            _same(r4_rows[name], score_arrays[name], f"{stage_id}/R4/{name}")
        original_score_direct = original_direct[score].astype("<f8", copy=False)
        _same(r4_rows["direct_predictions"], original_score_direct,
              f"{stage_id}/R4/direct_predictions")
        state = json.loads(layout.input_paths[f"{slug}_hinge_state"].read_text(encoding="utf-8"))
        solver = state.get("solver_result", {})
        params = {key: float(solver[key]) for key in ("intercept", "slope_low", "slope_high")}
        if (state.get("status") != "LP_COMPLETE" or solver.get("success") is not True
                or not np.isfinite(list(params.values())).all()
                or params["slope_low"] < 0.0 or params["slope_high"] < 0.0
                or params["slope_low"] - 0.2 < 0.0):
            raise runner.PilotError(f"Independent R4 monotonicity replay failed: {stage_id}")
        present_r4 = ~np.asarray(r4_rows["missing_mask"], dtype=bool)
        replay = original_score_direct.copy()
        p = original_score_direct[present_r4]
        raw_hinge = (params["intercept"] + params["slope_low"] * np.minimum(p, 50.0)
                     + params["slope_high"] * np.maximum(p - 50.0, 0.0))
        replay[present_r4] = np.clip(raw_hinge, 0.0, 100.0)
        _same(r4_rows["hinge_present_only"], replay, f"{stage_id}/R4/full_direct_hinge_replay")
        replay_record = preflight.get("r4_original_direct_hinge_replays", {}).get(stage_id, {})
        if (replay_record.get("score_rows_sha256") != _file_sha(r4_path)
                or replay_record.get("full_direct_hinge_replay") is not True
                or replay_record.get("hinge_state_sha256")
                != _file_sha(layout.input_paths[f"{slug}_hinge_state"])):
            raise runner.PilotError(f"Independent R4 replay receipt differs: {stage_id}")
        fit_summary = population_records.get((stage_id, "present_fit"))
        score_summary = population_records.get((stage_id, "score"))
        if fit_summary is None or score_summary is None:
            raise runner.PilotError(f"Independent receipt is missing population: {stage_id}")
        for path, record, arrays, expected_count, expected_npresent in (
                (fit_path, fit_summary, fit_arrays, expected_all, expected_present),
                (score_path, score_summary, score_arrays, expected_score_rows,
                 expected_score_rows - int(missing[score].sum()))):
            if (record.get("target_free_identity_artifact_sha256") != _file_sha(path)
                    or record.get("n_all") != expected_count
                    or record.get("n_present") != expected_npresent
                    or record.get("row_identity_hashes") != _identity_hashes(arrays)):
                raise runner.PilotError(f"Independent population receipt hashes differ: {stage_id}/{record.get('population')}")
        id_stage = id_populations.get(stage_id, {})
        id_present = id_stage.get("present_rows", [])
        id_missing = id_stage.get("excluded_missing_rows", [])
        id_score = id_stage.get("score_rows", [])
        expected_records = lambda indices: [
            {"record_id": str(ids[i]), "physical_row": int(physical[i]),
             "prediction_origin": int(origins[i]), "original_bundle_index": int(labels["original_bundle_indices"][i]),
             "cycle": int(cycles[i])} for i in indices]
        if id_present != expected_records(fit_present) or id_missing != expected_records(fit_missing) or id_score != expected_records(score):
            raise runner.PilotError(f"Independent explicit row-ID manifest differs: {stage_id}")
        score_ids.extend(ids[score].tolist())
        score_rows.extend(physical[score].tolist())

    if (len(score_ids) != 15_953 or len(set(score_ids)) != 15_953
            or len(set(score_rows)) != 15_953
            or ids_body.get("conditional_final_present_rows") != 19_528
            or ids_body.get("fit_counts") != runner.PRESENT_FIT_COUNTS
            or ids_body.get("excluded_missing_counts") != {"HINGE_D1": 342, "HINGE_D2": 849}
            or preflight.get("pooled_score_rows") != 15_953
            or preflight.get("cycle_74_scored_by_d1") != int(np.sum((cycles == 74) & (origins == 67)))):
        raise runner.PilotError("Independent pooled score or conditional final counts differ")

    # The source and verifier must not create an official result or consume a fit slot.
    forbidden = (layout.root_pass, layout.ledger, layout.prediction_freeze,
                 layout.evaluation, layout.verification)
    if any(path.exists() for path in forbidden) or layout.fit_dir.exists() or layout.prediction_dir.exists():
        raise runner.PilotError("Source-ready verification found an official fit/score artifact")
    return preflight, preflight_sha, observed


def verify_source_ready(layout: runner.Layout | None = None) -> dict[str, Any]:
    layout = runner.DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = _verify_freeze(layout)
    preflight, preflight_sha, identity = _verify_preflight(layout, freeze, freeze_sha)
    body = {"schema_version": 1, "experiment_id": runner.NAMESPACE.upper() + "_R2",
        "status": "PASS_SOURCE_READY_INDEPENDENT_VERIFICATION",
        "source_commit": freeze["source_commit"], "prefit_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": preflight_sha,
        "fit_population_ids_sha256": preflight["fit_population_ids_sha256"],
        "oof_rows": int(len(identity["record_ids"])),
        "fit_rows_all": runner.FIT_COUNTS,
        "fit_rows_present": runner.PRESENT_FIT_COUNTS,
        "fit_missing_excluded": {"HINGE_D1": 342, "HINGE_D2": 849},
        "score_rows": runner.SCORE_COUNTS, "pooled_score_unique_rows": 15_953,
        "conditional_final_present_rows": 19_528,
        "target_member_present_but_unloaded": True,
        "candidate_target_values_loaded": False, "score_target_values_loaded": False,
        "calibrator_fit_starts": 0, "root_source_pass_created": False,
        "ledger_created": False, "test_or_sample_accessed": False,
        "e79_accessed": False, "csv_created": False, "kaggle_upload": False}
    body["signature"] = _sha(body)
    path = layout.report_dir / "SOURCE_READY_VERIFICATION_R2.json"
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        existing, digest = _signed(path)
        if existing != body:
            raise runner.PilotError("Existing source-ready verification differs; refusing replacement")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    return {"status": body["status"], "path": str(path.resolve()),
            "sha256": _file_sha(path), "source_commit": body["source_commit"],
            "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "pooled_score_unique_rows": 15_953}


def verify_results(layout: runner.Layout | None = None) -> dict[str, Any]:
    """Compatibility entrypoint: this source-only run verifies readiness, not fit results."""
    return verify_source_ready(layout)
