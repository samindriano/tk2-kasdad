"""Independent replay verifier for the bounded center-local correction pilot."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from scripts import reg_center_local_correction_v1 as runner


FIT_IDS = ("D1_LOCAL", "D1_GLOBAL_CONTROL", "D2_LOCAL", "D2_GLOBAL_CONTROL")
STAGES = ("HINGE_D1", "HINGE_D2")
FIT_STAGE = {"D1_LOCAL": "HINGE_D1", "D1_GLOBAL_CONTROL": "HINGE_D1",
             "D2_LOCAL": "HINGE_D2", "D2_GLOBAL_CONTROL": "HINGE_D2"}
FIT_ROWS = {"HINGE_D1": 4_950, "HINGE_D2": 12_412}
FIT_PRESENT = {"HINGE_D1": 4_608, "HINGE_D2": 11_563}
SCORE_ROWS = {"HINGE_D1": 7_462, "HINGE_D2": 8_491}
ORIGIN = {("HINGE_D1", "fit"): 67, ("HINGE_D1", "score"): 67,
          ("HINGE_D2", "fit"): 67, ("HINGE_D2", "score"): 75}
CYCLES = {("HINGE_D1", "fit"): tuple(range(68, 72)),
          ("HINGE_D1", "score"): tuple(range(72, 76)),
          ("HINGE_D2", "fit"): tuple(range(68, 76)),
          ("HINGE_D2", "score"): tuple(range(76, 80))}
FIT_EPSILON = 1e-12
NOVELTY = dict(runner.NOVELTY_HASHES)


def _fail(message: str) -> None:
    raise runner.PilotError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise runner.PilotError(f"Cannot independently read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"Expected JSON object: {path}")
    return value


def _signed(path: Path, *, verdict: str | None = None) -> dict[str, Any]:
    value = _json(path)
    body = dict(value)
    signature = body.pop("signature", None)
    if not isinstance(signature, str) or signature != _canonical_sha(body):
        _fail(f"Independent canonical signature check failed: {path}")
    if verdict is not None and value.get("verdict") != verdict:
        _fail(f"Unexpected signed verdict: {path}")
    return value


def _tagged_sha(values: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    digest = hashlib.sha256(_canonical_bytes({"dtype": str(array.dtype),
                                               "shape": list(array.shape)}))
    if array.dtype.kind in "USO":
        digest.update(_canonical_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _row_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    fields = ("physical_rows", "record_ids", "prediction_origins",
              "original_bundle_indices", "cycles", "direct_predictions", "missing_mask")
    vectors = {name: np.asarray(arrays[name]) for name in fields}
    if any(value.ndim != 1 for value in vectors.values()) or len({len(v) for v in vectors.values()}) != 1:
        _fail("Independent identity replay found misaligned vectors")
    return {
        "physical_rows_sha256": _tagged_sha(vectors["physical_rows"]),
        "record_ids_sha256": _tagged_sha(vectors["record_ids"]),
        "prediction_origins_sha256": _tagged_sha(vectors["prediction_origins"]),
        "original_bundle_indices_sha256": _tagged_sha(vectors["original_bundle_indices"]),
        "cycles_sha256": _tagged_sha(vectors["cycles"]),
        "direct_prediction_sha256": _tagged_sha(vectors["direct_predictions"], dtype="<f8"),
        "missing_mask_sha256": _tagged_sha(vectors["missing_mask"]),
    }


def _load_npz(path: Path, selected: tuple[str, ...], allowed: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            required = set(selected) | set(allowed)
            if names != required:
                _fail(f"Independent NPZ schema mismatch: {path}; missing={required-names}, extra={names-required}")
            return {name: np.asarray(archive[name]).copy() for name in selected}
    except (OSError, ValueError) as exc:
        raise runner.PilotError(f"Cannot independently read NPZ artifact: {path}") from exc


IDENTITY = ("physical_rows", "record_ids", "prediction_origins",
            "original_bundle_indices", "cycles", "direct_predictions", "missing_mask")
FIT_SELECTED = IDENTITY + ("targets",)
FIT_ALLOWED = FIT_SELECTED
SCORE_NON_TARGET_SELECTED = IDENTITY + ("raw_direct", "hinge_present_only")
SCORE_FIELDS = set(SCORE_NON_TARGET_SELECTED) | {
    "targets", "causal_global_all_rows", "causal_global_present_only"}
PRED_SELECTED = IDENTITY + ("hinge_baseline", "local_prediction", "global_control_prediction")


def _validate_identity(stage: str, role: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = FIT_ROWS[stage] if role == "fit" else SCORE_ROWS[stage]
    values = [np.asarray(arrays[name]) for name in IDENTITY]
    if any(value.ndim != 1 or len(value) != n for value in values):
        _fail(f"Independent row identity length failed for {stage}/{role}")
    p = np.asarray(arrays["direct_predictions"])
    miss_raw = np.asarray(arrays["missing_mask"])
    ids = np.asarray(arrays["record_ids"]).astype(str)
    physical = np.asarray(arrays["physical_rows"], dtype=np.int64)
    origins = np.asarray(arrays["prediction_origins"], dtype=np.int64)
    cycles = np.asarray(arrays["cycles"], dtype=np.int64)
    if (p.dtype.kind != "f" or p.dtype.itemsize != 8 or miss_raw.dtype.kind != "b"
            or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 100.0))
            or len(set(ids.tolist())) != n or len(set(physical.tolist())) != n
            or np.all(miss_raw)):
        _fail(f"Independent raw prediction or uniqueness check failed for {stage}/{role}")
    if (not np.all(origins == ORIGIN[(stage, role)])
            or tuple(sorted(np.unique(cycles).tolist())) != CYCLES[(stage, role)]):
        _fail(f"Independent origin/cycle check failed for {stage}/{role}")
    missing = miss_raw.astype(bool, copy=False)
    present_n = int((~missing).sum())
    if role == "fit" and present_n != FIT_PRESENT[stage]:
        _fail(f"Independent present fit count failed for {stage}")
    b = np.maximum(0.0, 1.0 - np.abs(p - 50.0) / 10.0)
    active = int(((~missing) & (b > 0.0)).sum())
    if role == "fit" and active < 200:
        _fail(f"Independent LOCAL fit support failed for {stage}")
    return {"fit_id": stage, "population": role, "n_all": n,
            "n_present": present_n, "n_missing": int(missing.sum()),
            "local_positive_basis_present_rows": active,
            "origins": sorted(np.unique(origins).tolist()),
            "cycles": sorted(np.unique(cycles).tolist()),
            "row_identity_hashes": _row_hashes(arrays)}


def _expected_source_pins(layout: runner.Layout) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for category, files in (("source", layout.source_paths),
                            ("reference_source", layout.reference_paths),
                            ("input", layout.input_paths)):
        for name, path in files.items():
            resolved = Path(path).resolve()
            if not resolved.is_file():
                _fail(f"Frozen source/input is absent: {resolved}")
            out[f"{category}:{name}"] = {"path": str(resolved), "sha256": _sha256(resolved)}
    for name, path in runner._expected_novelty_paths(layout).items():
        digest = _sha256(path)
        if digest != NOVELTY[name]:
            _fail(f"One of the six frozen novelty summaries changed: {name}")
        out[f"novelty_summary:{name}"] = {"path": str(path.resolve()), "sha256": digest}
    return out


def _verify_freeze(layout: runner.Layout) -> tuple[dict[str, Any], str]:
    freeze = _json(layout.freeze)
    signature = freeze.pop("signature", None)
    if signature != _canonical_sha(freeze):
        _fail("Independent source-freeze signature mismatch")
    freeze["signature"] = signature
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=layout.root,
                          capture_output=True, text=True, check=False)
    if head.returncode or freeze.get("source_commit") != head.stdout.strip():
        _fail("Independent source-freeze HEAD binding mismatch")
    if freeze.get("status") != "FROZEN_BEFORE_CANDIDATE_TARGET_ACCESS":
        _fail("Independent source-freeze status mismatch")
    expected = _expected_source_pins(layout)
    if freeze.get("source_and_input_files") != expected:
        _fail("Independent source/input pins differ from the freeze inventory")
    for key in ("candidate_targets_loaded", "fit_started", "score_targets_loaded",
                "test_or_sample_accessed", "e79_accessed", "csv_created", "kaggle_upload"):
        if freeze.get(key) is not False:
            _fail(f"Source freeze boundary is not closed: {key}")
    return freeze, _sha256(layout.freeze)


def _verify_preflight(layout: runner.Layout, freeze: Mapping[str, Any],
                      freeze_sha: str) -> tuple[dict[str, Any], str, list[dict[str, np.ndarray]]]:
    receipt = _json(layout.preflight)
    signature = receipt.pop("signature", None)
    if signature != _canonical_sha(receipt):
        _fail("Independent non-target preflight signature mismatch")
    receipt["signature"] = signature
    if (receipt.get("status") != "PASS_NON_TARGET_IDENTITY_PREFLIGHT"
            or receipt.get("input_freeze_sha256") != freeze_sha
            or receipt.get("source_commit") != freeze.get("source_commit")
            or receipt.get("candidate_target_values_loaded") is not False
            or receipt.get("target_arrays_present_in_npz_but_loaded") is not False
            or receipt.get("score_target_values_loaded") is not False
            or receipt.get("calibrator_fit_starts") != 0
            or receipt.get("ledger_created") is not False):
        _fail("Independent preflight boundary/freeze binding failed")
    for key in ("test_or_sample_accessed", "e79_accessed", "csv_created", "kaggle_upload"):
        if receipt.get(key) is not False:
            _fail(f"Independent preflight prohibited-access boundary failed: {key}")
    if receipt.get("saved_hinge_present_only_replay") is not True:
        _fail("Independent preflight does not attest its saved HINGE replay")
    expected_order = [(stage, role) for stage in STAGES for role in ("fit", "score")]
    rows: list[dict[str, np.ndarray]] = []
    all_ids: list[str] = []
    all_physical: list[int] = []
    expected_population_rows = []
    for stage, role in expected_order:
        fit_path, score_path, _ = runner._stage_files(layout, stage)
        if role == "fit":
            arrays = _load_npz(fit_path, IDENTITY, ("targets",))
        else:
            arrays = _load_npz(score_path, SCORE_NON_TARGET_SELECTED,
                               tuple(SCORE_FIELDS - set(SCORE_NON_TARGET_SELECTED)))
            p = np.asarray(arrays["direct_predictions"], dtype="<f8")
            raw = np.asarray(arrays["raw_direct"], dtype="<f8")
            miss = np.asarray(arrays["missing_mask"], dtype=bool)
            if not np.array_equal(raw.view("<u8"), p.view("<u8")):
                _fail(f"Independent raw DIRECT cache equality failed: {stage}")
            hinge_cache = np.asarray(arrays["hinge_present_only"], dtype="<f8")
            if not np.array_equal(hinge_cache[miss].view("<u8"), p[miss].view("<u8")):
                _fail(f"Independent HINGE missing fallback failed: {stage}")
            state = _json(runner._stage_files(layout, stage)[2])
            params = state["solver_result"]
            replay = p.copy()
            present = ~miss
            replay[present] = np.clip(float(params["intercept"])
                + float(params["slope_low"]) * np.minimum(p[present], 50.0)
                + float(params["slope_high"]) * np.maximum(p[present] - 50.0, 0.0), 0.0, 100.0)
            if not np.array_equal(replay.view("<u8"), hinge_cache.view("<u8")):
                _fail(f"Independent saved HINGE replay failed: {stage}")
            all_ids.extend(np.asarray(arrays["record_ids"]).astype(str).tolist())
            all_physical.extend(np.asarray(arrays["physical_rows"], dtype=np.int64).tolist())
        population = _validate_identity(stage, role, arrays)
        expected_population_rows.append(population)
        rows.append(arrays)
    if receipt.get("populations") != expected_population_rows:
        _fail("Independent row identities differ from the preflight receipt")
    if (len(all_ids) != 15_953 or len(set(all_ids)) != 15_953
            or len(set(all_physical)) != 15_953
            or receipt.get("pooled_score_rows") != 15_953
            or receipt.get("pooled_score_record_ids_unique") is not True
            or receipt.get("pooled_score_physical_rows_unique") is not True):
        _fail("Independent pooled score identity uniqueness failed")
    return receipt, _sha256(layout.preflight), rows


def _verify_root_pass(layout: runner.Layout, freeze: Mapping[str, Any],
                      freeze_sha: str, preflight_sha: str) -> tuple[dict[str, Any], str]:
    passed = _signed(layout.root_pass, verdict="PASS")
    expected_fields = {"verdict", "experiment_id", "pass_kind", "source_commit",
        "prefit_freeze_sha256", "identity_preflight_sha256", "authorized_fit_ids",
        "calibrator_fit_cap", "authorized_calibrator_fits", "base_model_fits",
        "preprocessing_fits", "official_lp_starts", "prior_three_slot_budget_reset",
        "score_target_access_authorized", "test_or_sample_authorized",
        "kaggle_upload_authorized", "signature"}
    if set(passed) != expected_fields:
        _fail("Independent root source pass fields differ from the exact callback schema")
    required = {"experiment_id": runner.NAMESPACE.upper() + "_R1",
        "pass_kind": "CENTER_LOCAL_DIAGNOSTIC", "source_commit": freeze["source_commit"],
        "prefit_freeze_sha256": freeze_sha, "identity_preflight_sha256": preflight_sha,
        "authorized_fit_ids": list(FIT_IDS), "calibrator_fit_cap": 4,
        "authorized_calibrator_fits": 4, "base_model_fits": 0,
        "preprocessing_fits": 0, "official_lp_starts": 0,
        "prior_three_slot_budget_reset": False, "score_target_access_authorized": True,
        "test_or_sample_authorized": False, "kaggle_upload_authorized": False}
    if any(passed.get(key) != value for key, value in required.items()):
        _fail("Independent root source pass is not the exact bounded authorization")
    if passed.get("signature") != _canonical_sha({k: v for k, v in passed.items() if k != "signature"}):
        _fail("Independent root source pass signature mismatch")
    return passed, _sha256(layout.root_pass)


def _verify_ledger(layout: runner.Layout, freeze_sha: str,
                   pass_sha: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    try:
        raw_lines = layout.ledger.read_bytes().splitlines()
        raw_events = [json.loads(line) for line in raw_lines]
    except (OSError, ValueError) as exc:
        raise runner.PilotError("Independent verifier cannot read complete calibrator ledger") from exc
    if not raw_events:
        _fail("Independent verifier found no calibrator ledger events")
    events: dict[str, list[dict[str, Any]]] = {}
    observed_order = []
    fits: dict[str, dict[str, Any]] = {}
    for event in raw_events:
        fit_id = event.get("fit_id")
        kind = event.get("event")
        if fit_id not in FIT_IDS or kind not in {"RESERVE", "FIT_STARTED", "FIT_COMPLETE", "FIT_FAILED"}:
            _fail("Independent ledger contains unknown fit or event")
        observed_order.append((fit_id, kind))
        events.setdefault(fit_id, []).append(event)
    expected_order = [(fit_id, kind) for fit_id in FIT_IDS
                      for kind in ("RESERVE", "FIT_STARTED", "FIT_COMPLETE")]
    if observed_order != expected_order:
        _fail("Independent ledger does not show exactly four ordered completed fit slots")
    all_source_commits = set()
    for fit_id in FIT_IDS:
        records = events[fit_id]
        if [item.get("event") for item in records] != ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"]:
            _fail(f"Independent ledger fit sequence failed: {fit_id}")
        if any(item.get("schema_version") != 1 for item in records):
            _fail(f"Independent ledger schema failed: {fit_id}")
        if any(item.get("prefit_freeze_sha256") != freeze_sha
               or item.get("root_source_pass_sha256") != pass_sha for item in records):
            _fail(f"Independent ledger authorization binding failed: {fit_id}")
        stage = FIT_STAGE[fit_id]
        fit_path = runner._stage_files(layout, stage)[0]
        fit_input_sha = _sha256(fit_path)
        if any(item.get("fit_input_sha256") != fit_input_sha for item in records):
            _fail(f"Independent ledger input binding failed: {fit_id}")
        reservation = hashlib.sha256(
            f"{runner.NAMESPACE}:R1:{fit_id}:{freeze_sha}:{pass_sha}".encode("utf-8")).hexdigest()
        if any(item.get("reservation_id") != reservation for item in records):
            _fail(f"Independent ledger reservation failed: {fit_id}")
        if len({item.get("source_commit") for item in records}) != 1:
            _fail(f"Independent ledger source commit changed within fit: {fit_id}")
        all_source_commits.add(records[0].get("source_commit"))
        state_path = layout.fit_dir / f"{fit_id}.json"
        done = records[-1]
        if (Path(done.get("fit_state_path", "")).resolve() != state_path.resolve()
                or not state_path.is_file() or _sha256(state_path) != done.get("fit_state_sha256")):
            _fail(f"Independent fit state hash/path binding failed: {fit_id}")
        state = _json(state_path)
        if (state.get("fit_id") != fit_id or state.get("stage_id") != stage
                or state.get("status") != "FIT_COMPLETE"
                or state.get("source_commit") != records[0].get("source_commit")
                or state.get("fit_input_sha256") != fit_input_sha
                or state.get("prefit_freeze_sha256") != freeze_sha
                or state.get("root_source_pass_sha256") != pass_sha
                or state.get("counted_as_one_calibrator_fit") is not True
                or state.get("base_model_fit") is not False
                or state.get("preprocessing_fit") is not False):
            _fail(f"Independent fit state metadata binding failed: {fit_id}")
        fits[fit_id] = state
    frozen_commit = _json(layout.freeze).get("source_commit")
    if all_source_commits != {frozen_commit}:
        _fail("Independent ledger source commit differs across fit windows")
    return events, fits


def _parameters(layout: runner.Layout, stage: str) -> dict[str, float]:
    path = runner._stage_files(layout, stage)[2]
    state = _json(path)
    if state.get("status") != "LP_COMPLETE" or state.get("solver_result", {}).get("success") is not True:
        _fail(f"Independent saved HINGE state incomplete: {stage}")
    raw = state["solver_result"]
    result = {key: float(raw[key]) for key in ("intercept", "slope_low", "slope_high")}
    if not all(math.isfinite(value) for value in result.values()) or min(result["slope_low"], result["slope_high"]) < 0:
        _fail(f"Independent saved HINGE parameters invalid: {stage}")
    return result


def _base(p: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    return (float(params["intercept"])
            + float(params["slope_low"]) * np.minimum(p, 50.0)
            + float(params["slope_high"]) * np.maximum(p - 50.0, 0.0))


def _lower_weighted_interval(x: np.ndarray, weight: np.ndarray) -> tuple[float, float, int]:
    pairs = sorted((float(value), Fraction.from_float(float(w)))
                   for value, w in zip(x, weight, strict=True))
    grouped: list[list[Any]] = []
    for value, w in pairs:
        if grouped and value == grouped[-1][0]:
            grouped[-1][1] += w
        else:
            grouped.append([value, w])
    total = sum((item[1] for item in grouped), Fraction())
    cum = Fraction()
    for index, (value, w) in enumerate(grouped):
        cum += w
        if 2 * cum >= total:
            upper = grouped[index + 1][0] if 2 * cum == total and index + 1 < len(grouped) else value
            return value, upper, len(pairs)
    _fail("Independent weighted median accumulation failed")


def _bounded_interval(low: float, high: float) -> tuple[float, float]:
    return float(np.clip(low, -2.0, 0.0)), float(np.clip(high, -2.0, 0.0))


def _fit_replay(fit_id: str, arrays: Mapping[str, np.ndarray],
                params: Mapping[str, float]) -> dict[str, Any]:
    target = np.asarray(arrays["targets"], dtype=np.float64)
    p = np.asarray(arrays["direct_predictions"], dtype=np.float64)
    present = ~np.asarray(arrays["missing_mask"], dtype=bool)
    if (not np.isfinite(target).all() or np.any((target < 0.0) | (target > 100.0))
            or not present.any()):
        _fail(f"Independent fit target validity failed: {fit_id}")
    g = _base(p[present], params)
    residual = target[present] - g
    local = fit_id.endswith("_LOCAL")
    if local:
        b = np.maximum(0.0, 1.0 - np.abs(p[present] - 50.0) / 10.0)
        active = b > 0.0
        if int(active.sum()) < 200:
            _fail(f"Independent LOCAL support below threshold: {fit_id}")
        raw_low, raw_high, weighted = _lower_weighted_interval(residual[active] / b[active], b[active])
        bounded = _bounded_interval(raw_low, raw_high)
        delta = bounded[0]
        objective = math.fsum(float(value) for value in np.abs(residual - delta * b))
        active_rows = int(active.sum())
    else:
        ordered = np.sort(residual, kind="mergesort")
        raw_low = float(ordered[(len(ordered) - 1) // 2])
        raw_high = float(ordered[len(ordered) // 2])
        bounded = _bounded_interval(raw_low, raw_high)
        delta = bounded[0]
        objective = math.fsum(float(value) for value in np.abs(residual - delta))
        weighted = len(residual)
        active_rows = len(residual)
    fit = {"delta": delta, "unbounded_median_interval": [raw_low, raw_high],
        "bounded_optimal_interval": list(bounded),
        "solution_nonunique": bounded[0] != bounded[1],
        "unbounded_solution_nonunique": raw_low != raw_high,
        "fit_present_rows": int(present.sum()), "fit_active_rows": active_rows,
        "weighted_rows": weighted,
        "objective_unweighted_sum_absolute_unclipped": objective}
    derivatives = {"outside_left": float(params["slope_low"]),
        "local_40_to_50": float(params["slope_low"]) + delta / 10.0,
        "local_50_to_60": float(params["slope_high"]) - delta / 10.0,
        "outside_right": float(params["slope_high"])}
    monotone = min(derivatives.values()) >= 0.0
    if local and not monotone:
        _fail(f"Independent LOCAL monotonicity failed: {fit_id}")
    return {"fit": fit, "monotonicity": {"piecewise_derivatives": derivatives,
            "minimum_derivative": min(derivatives.values()),
            "monotone_non_decreasing": monotone}}


def _equal(a: Any, b: Any, path: str = "root") -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            _fail(f"Independent replay mismatch at {path}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-9):
            _fail(f"Independent numeric replay mismatch at {path}: {a} != {b}")
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            _fail(f"Independent schema replay mismatch at {path}")
        for key in a:
            _equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            _fail(f"Independent sequence replay mismatch at {path}")
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            _equal(left, right, f"{path}[{i}]")
    elif a != b:
        _fail(f"Independent replay mismatch at {path}: {a!r} != {b!r}")


def _metric(pred: np.ndarray, target: np.ndarray, minimum: int) -> dict[str, Any]:
    n = int(len(target))
    if n < minimum:
        return {"status": "UNKNOWN_UNSUPPORTED", "n": n, "mae": None,
                "mean_error": None, "p90_absolute_error": None}
    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    return {"status": "SUPPORTED", "n": n, "mae": float(np.mean(absolute)),
            "mean_error": float(np.mean(error)),
            "p90_absolute_error": float(np.quantile(absolute, 0.90))}


def _group(mask: np.ndarray, present: np.ndarray, target: np.ndarray,
           pred: Mapping[str, np.ndarray], minimum: int) -> dict[str, Any]:
    all_mask = np.asarray(mask, dtype=bool)
    present_mask = all_mask & present
    out = {"n_all": int(all_mask.sum()), "n_present": int(present_mask.sum()),
        "n_missing": int((all_mask & ~present).sum()),
        "metrics": {"all_rows": {}, "present_only": {}}}
    for name, values in pred.items():
        out["metrics"]["all_rows"][name] = _metric(values[all_mask], target[all_mask], minimum)
        out["metrics"]["present_only"][name] = _metric(values[present_mask], target[present_mask], minimum)
    out["gain_over_hinge"] = {"all_rows": {}, "present_only": {}}
    for scope in ("all_rows", "present_only"):
        hinge = out["metrics"][scope]["HINGE"]
        for name in ("LOCAL", "GLOBAL_CONTROL"):
            current = out["metrics"][scope][name]
            out["gain_over_hinge"][scope][name] = (
                hinge["mae"] - current["mae"]
                if hinge["status"] == "SUPPORTED" and current["status"] == "SUPPORTED" else None)
    return out


def _population(direct: np.ndarray, cycles: np.ndarray, missing: np.ndarray,
                target: np.ndarray, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    present = ~np.asarray(missing, dtype=bool)
    edges = np.asarray((0.0, 20.0, 40.0, 60.0, 80.0, 100.0))
    labels = ("[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]")
    index = np.clip(np.searchsorted(edges, direct, side="right") - 1, 0, 4)
    bins = [{"bin_index": i, "label": label,
        **_group(index == i, present, target, pred, 200)} for i, label in enumerate(labels)]
    by_cycle = [{"cycle": int(cycle),
        **_group(np.asarray(cycles) == cycle, present, target, pred, 50)}
        for cycle in sorted(np.unique(cycles).tolist())]
    tails = {name: _group(mask, present, target, pred, 200)
             for name, mask in (("target_le_5", target <= 5.0),
                                ("target_ge_95", target >= 95.0))}
    overall = _group(np.ones(len(target), dtype=bool), present, target, pred, 1)
    return {"n_all": int(len(target)), "n_present": int(present.sum()),
        "n_missing": int((~present).sum()), "overall": overall,
        "by_raw_direct_bin": bins, "by_cycle": by_cycle, "target_tails": tails}


def _score_gates(results: Mapping[str, Mapping[str, Any]],
                 fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gains, harms, region = {}, {}, {}
    for stage in STAGES:
        item = results[stage]
        overall = item["metrics"]["overall"]
        local_mae = overall["metrics"]["all_rows"]["LOCAL"]["mae"]
        hinge_mae = overall["metrics"]["all_rows"]["HINGE"]["mae"]
        control_mae = overall["metrics"]["all_rows"]["GLOBAL_CONTROL"]["mae"]
        gains[stage] = {"local_over_hinge": overall["gain_over_hinge"]["all_rows"]["LOCAL"],
            "local_over_global_control": control_mae - local_mae,
            "global_control_over_hinge": hinge_mae - control_mae}
        group_list = [item["metrics"]["overall"], *item["metrics"]["by_raw_direct_bin"],
                      *item["metrics"]["by_cycle"], *item["metrics"]["target_tails"].values()]
        supported_harm = []
        for group in group_list:
            hinge = group["metrics"]["all_rows"]["HINGE"]
            local = group["metrics"]["all_rows"]["LOCAL"]
            if hinge["status"] == local["status"] == "SUPPORTED":
                supported_harm.append(local["mae"] - hinge["mae"])
        harms[stage] = max(supported_harm) if supported_harm else None
        region[stage] = item["targeted_region"]["local_gain_over_hinge"]
    n = sum(SCORE_ROWS.values())
    if n != 15_953:
        _fail("Independent pooled score denominator changed")
    pooled_mae = {policy: sum(results[stage]["metrics"]["overall"]["metrics"]["all_rows"][policy]["mae"]
        * SCORE_ROWS[stage] for stage in STAGES) / n
        for policy in ("HINGE", "LOCAL", "GLOBAL_CONTROL")}
    pooled_gain = {arm: pooled_mae["HINGE"] - pooled_mae[arm]
                   for arm in ("LOCAL", "GLOBAL_CONTROL")}
    deltas = {stage: fits["D1_LOCAL" if stage == "HINGE_D1" else "D2_LOCAL"]["fit"]["delta"]
              for stage in STAGES}
    maximum_harm = max((value for value in harms.values() if value is not None), default=None)
    mechanism = (all(deltas[stage] < 0 for stage in STAGES)
        and all(gains[stage]["local_over_hinge"] > FIT_EPSILON
                and gains[stage]["local_over_global_control"] > FIT_EPSILON for stage in STAGES))
    candidate = (pooled_gain["LOCAL"] >= 0.01
        and all(gains[stage]["local_over_hinge"] >= 0 for stage in STAGES)
        and maximum_harm is not None and maximum_harm <= 0.25)
    exploratory = (pooled_gain["LOCAL"] >= 0.0
        and all(gains[stage]["local_over_hinge"] >= -0.02 for stage in STAGES)
        and maximum_harm is not None and maximum_harm <= 0.25)
    return {"score_rows_pooled_unique": n, "window_gains": gains,
        "pooled_mae": pooled_mae, "pooled_gain_over_hinge": pooled_gain,
        "supported_all_row_group_harm_vs_hinge": harms,
        "maximum_supported_all_row_group_harm_vs_hinge": maximum_harm,
        "targeted_region_gain_over_hinge": region, "local_fit_deltas": deltas,
        "mechanism_gate": {"pass": mechanism,
            "rule": "delta<0 in both fit populations; LOCAL gain >1e-12 over HINGE and GLOBAL_CONTROL in both score windows"},
        "candidate_gate": {"pass": candidate,
            "rule": "pooled gain>=0.01; each window gain>=0; max supported all-row group harm<=0.25"},
        "exploratory_review_gate": {"pass": exploratory,
            "rule": "pooled gain>=0; each window harm<=0.02; max supported all-row group harm<=0.25"}}


def _existing_verification(layout: runner.Layout, freeze_sha: str,
                           preflight_sha: str, pass_sha: str,
                           prediction_sha: str) -> dict[str, Any] | None:
    if not layout.verification.exists():
        return None
    receipt = _json(layout.verification)
    signature = receipt.pop("signature", None)
    if signature != _canonical_sha(receipt):
        _fail("Existing independent verification signature mismatch")
    receipt["signature"] = signature
    if (receipt.get("status") != "PASS_INDEPENDENT_VERIFICATION"
            or receipt.get("prefit_freeze_sha256") != freeze_sha
            or receipt.get("identity_preflight_sha256") != preflight_sha
            or receipt.get("root_source_pass_sha256") != pass_sha
            or receipt.get("prediction_freeze_sha256") != prediction_sha
            or receipt.get("evaluation_sha256") != _sha256(layout.evaluation)):
        _fail("Existing independent verification does not bind the current artifacts")
    return receipt


def verify_results(layout: runner.Layout | None = None) -> dict[str, Any]:
    layout = runner.DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = _verify_freeze(layout)
    preflight, preflight_sha, identity_rows = _verify_preflight(layout, freeze, freeze_sha)
    root_pass, pass_sha = _verify_root_pass(layout, freeze, freeze_sha, preflight_sha)
    prediction_freeze = _json(layout.prediction_freeze)
    prediction_signature = prediction_freeze.pop("signature", None)
    if prediction_signature != _canonical_sha(prediction_freeze):
        _fail("Independent prediction freeze signature mismatch")
    prediction_freeze["signature"] = prediction_signature
    if (prediction_freeze.get("status") != "SCORE_PREDICTIONS_FROZEN_BEFORE_SCORE_TARGET_LOAD"
            or prediction_freeze.get("source_commit") != freeze.get("source_commit")
            or prediction_freeze.get("prefit_freeze_sha256") != freeze_sha
            or prediction_freeze.get("identity_preflight_sha256") != preflight_sha
            or prediction_freeze.get("root_source_pass_sha256") != pass_sha
            or prediction_freeze.get("score_targets_loaded") is not False
            or any(prediction_freeze.get(key) is not False for key in
                   ("test_or_sample_accessed", "e79_accessed", "csv_created", "kaggle_upload"))):
        _fail("Independent prediction freeze authorization binding failed")
    prediction_sha = _sha256(layout.prediction_freeze)
    runner._verify_prediction_freeze(layout, freeze_sha, pass_sha)
    if not layout.evaluation.is_file():
        _fail("No evaluation exists for independent verification")
    evaluation = _json(layout.evaluation)
    eval_signature = evaluation.pop("signature", None)
    if eval_signature != _canonical_sha(evaluation):
        _fail("Independent evaluation signature mismatch")
    evaluation["signature"] = eval_signature
    if (evaluation.get("status") != "PASS_FOUR_FIT_SCORE_DIAGNOSTIC_COMPLETE"
            or evaluation.get("source_commit") != freeze.get("source_commit")
            or evaluation.get("prefit_freeze_sha256") != freeze_sha
            or evaluation.get("identity_preflight_sha256") != preflight_sha
            or evaluation.get("root_source_pass_sha256") != pass_sha
            or evaluation.get("prediction_freeze_sha256") != prediction_sha
            or evaluation.get("score_predictions_committed_before_score_target_load") is not True
            or evaluation.get("test_or_sample_accessed") is not False
            or evaluation.get("e79_accessed") is not False
            or evaluation.get("csv_created") is not False
            or evaluation.get("kaggle_upload") is not False
            or evaluation.get("calibrator_fit_starts") != 4
            or evaluation.get("base_model_fits") != 0
            or evaluation.get("preprocessing_fits") != 0):
        _fail("Independent evaluation boundary/authorization check failed")
    existing = _existing_verification(layout, freeze_sha, preflight_sha, pass_sha, prediction_sha)
    evaluation_sha = _sha256(layout.evaluation)
    if existing is not None:
        return {"status": existing["status"], "verification_path": str(layout.verification.resolve()),
                "verification_sha256": _sha256(layout.verification),
                "evaluation_sha256": evaluation_sha, "already_verified": True}

    ledger, fit_states = _verify_ledger(layout, freeze_sha, pass_sha)
    if set(prediction_freeze.get("fit_state_hashes", {})) != set(FIT_IDS):
        _fail("Independent prediction freeze missing one or more fit-state hashes")
    fit_results: dict[str, dict[str, Any]] = {}
    fit_target_hashes = {}
    for fit_id in FIT_IDS:
        state = fit_states[fit_id]
        digest = _sha256(layout.fit_dir / f"{fit_id}.json")
        if prediction_freeze["fit_state_hashes"].get(fit_id) != digest:
            _fail(f"Independent prediction freeze fit-state hash mismatch: {fit_id}")
        stage = FIT_STAGE[fit_id]
        fit_path = runner._stage_files(layout, stage)[0]
        arrays = _load_npz(fit_path, FIT_SELECTED, FIT_ALLOWED)
        population = _validate_identity(stage, "fit", arrays)
        if state.get("fit_population") != population:
            _fail(f"Independent fit population record mismatch: {fit_id}")
        target = np.asarray(arrays["targets"], dtype="<f8")
        target_hash = _tagged_sha(target, dtype="<f8")
        fit_target_hashes[fit_id] = target_hash
        if state.get("target_sha256") != target_hash:
            _fail(f"Independent fit target digest mismatch: {fit_id}")
        params = _parameters(layout, stage)
        if state.get("saved_hinge_parameters") != params:
            _fail(f"Independent saved HINGE parameter record mismatch: {fit_id}")
        replay = _fit_replay(fit_id, arrays, params)
        _equal(state.get("fit"), replay["fit"], f"fit_state.{fit_id}.fit")
        _equal(state.get("monotonicity"), replay["monotonicity"], f"fit_state.{fit_id}.monotonicity")
        fit_results[fit_id] = replay

    score_results = {}
    score_target_hashes = {}
    prediction_records = {record.get("stage_id"): record
                          for record in prediction_freeze.get("prediction_files", [])}
    if set(prediction_records) != set(STAGES) or len(prediction_records) != len(STAGES):
        _fail("Independent prediction freeze does not bind exactly two score windows")
    for stage in STAGES:
        score_path = runner._stage_files(layout, stage)[1]
        non_target = _load_npz(score_path, SCORE_NON_TARGET_SELECTED,
                               tuple(SCORE_FIELDS - set(SCORE_NON_TARGET_SELECTED)))
        stage_record = prediction_records[stage]
        prediction_path = layout.prediction_dir / f"{stage}_PREDICTIONS.npz"
        if (Path(stage_record.get("path", "")).resolve() != prediction_path.resolve()
                or _sha256(prediction_path) != stage_record.get("sha256")):
            _fail(f"Independent prediction file path/hash mismatch: {stage}")
        committed = _load_npz(prediction_path, PRED_SELECTED, PRED_SELECTED)
        pred_identity = _validate_identity(stage, "score", committed)
        if (len(committed["record_ids"]) != SCORE_ROWS[stage]
                or stage_record.get("n_rows") != SCORE_ROWS[stage]
                or stage_record.get("row_identity_sha256") != _canonical_sha(pred_identity["row_identity_hashes"])):
            _fail(f"Independent frozen prediction identity mismatch: {stage}")
        row_identity = _validate_identity(stage, "score", non_target)
        if row_identity != pred_identity:
            _fail(f"Independent prediction/non-target input identity mismatch: {stage}")
        for name in IDENTITY:
            if name == "direct_predictions":
                same = np.array_equal(np.asarray(committed[name], dtype="<f8").view("<u8"),
                                      np.asarray(non_target[name], dtype="<f8").view("<u8"))
            else:
                same = np.array_equal(committed[name], non_target[name])
            if not same:
                _fail(f"Independent frozen score row order changed: {stage}/{name}")
        p = np.asarray(non_target["direct_predictions"], dtype="<f8")
        missing = np.asarray(non_target["missing_mask"], dtype=bool)
        params = _parameters(layout, stage)
        base = _base(p[~missing], params)
        hinge = p.copy()
        hinge[~missing] = np.clip(base, 0.0, 100.0)
        cached = np.asarray(non_target["hinge_present_only"], dtype="<f8")
        if not np.array_equal(hinge.view("<u8"), cached.view("<u8")):
            _fail(f"Independent saved HINGE prediction replay failed: {stage}")
        local_fit = fit_results["D1_LOCAL" if stage == "HINGE_D1" else "D2_LOCAL"]["fit"]["delta"]
        global_fit = fit_results["D1_GLOBAL_CONTROL" if stage == "HINGE_D1" else "D2_GLOBAL_CONTROL"]["fit"]["delta"]
        b = np.maximum(0.0, 1.0 - np.abs(p[~missing] - 50.0) / 10.0)
        local_raw_present = base + local_fit * b
        global_raw_present = base + global_fit
        local = p.copy()
        global_control = p.copy()
        local[~missing] = np.clip(local_raw_present, 0.0, 100.0)
        global_control[~missing] = np.clip(global_raw_present, 0.0, 100.0)
        outside = np.abs(p[~missing] - 50.0) >= 10.0
        if np.any(b[outside] != 0.0) or np.any((local_raw_present - base)[outside] != 0.0):
            _fail(f"Independent LOCAL outside-support effect is nonzero: {stage}")
        for name, expected in (("hinge_baseline", hinge), ("local_prediction", local),
                               ("global_control_prediction", global_control)):
            actual = np.asarray(committed[name], dtype="<f8")
            if not np.array_equal(actual.view("<u8"), np.asarray(expected, dtype="<f8").view("<u8")):
                _fail(f"Independent frozen prediction replay failed: {stage}/{name}")
            if not np.array_equal(actual[missing].view("<u8"), p[missing].view("<u8")):
                _fail(f"Independent missing raw DIRECT bit check failed: {stage}/{name}")
        # The saved predictions and their hashes are checked before score targets are read.
        score_arrays = _load_npz(score_path, IDENTITY + ("targets",),
            tuple(SCORE_FIELDS - set(IDENTITY) - {"targets"}))
        if not np.array_equal(np.asarray(score_arrays["record_ids"]).astype(str),
                              np.asarray(committed["record_ids"]).astype(str)):
            _fail(f"Score target row order differs from the frozen prediction: {stage}")
        for name in ("physical_rows", "prediction_origins", "original_bundle_indices", "cycles", "missing_mask"):
            if not np.array_equal(score_arrays[name], committed[name]):
                _fail(f"Score target identity changed after prediction freeze: {stage}/{name}")
        score_direct = np.asarray(score_arrays["direct_predictions"], dtype="<f8")
        if not np.array_equal(score_direct.view("<u8"), p.view("<u8")):
            _fail(f"Score target DIRECT values differ from committed rows: {stage}")
        target = np.asarray(score_arrays["targets"], dtype="<f8")
        if not np.isfinite(target).all() or np.any((target < 0.0) | (target > 100.0)):
            _fail(f"Independent score target values invalid: {stage}")
        target_hash = _tagged_sha(target, dtype="<f8")
        score_target_hashes[stage] = target_hash
        pred_map = {"HINGE": hinge, "LOCAL": local, "GLOBAL_CONTROL": global_control}
        metrics = _population(p, np.asarray(score_arrays["cycles"]), missing, target, pred_map)
        active = (~missing) & (p > 40.0) & (p < 60.0)
        region_metrics = {name: _metric(values[active], target[active], 200)
                          for name, values in pred_map.items()}
        targeted = {"n_present": int(active.sum()),
            "support": "SUPPORTED" if int(active.sum()) >= 200 else "UNKNOWN_UNSUPPORTED",
            "metrics": region_metrics,
            "local_gain_over_hinge": (region_metrics["HINGE"]["mae"] - region_metrics["LOCAL"]["mae"]
                if region_metrics["HINGE"]["status"] == region_metrics["LOCAL"]["status"] == "SUPPORTED" else None)}
        predictions_hashes = {name: hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
                              for name, values in pred_map.items()}
        score_results[stage] = {"stage_id": stage, "score_rows": row_identity,
            "target_sha256": target_hash, "metrics": metrics, "targeted_region": targeted,
            "prediction_hashes": predictions_hashes}

    _equal(evaluation.get("fit_state_hashes"),
           {fit_id: _sha256(layout.fit_dir / f"{fit_id}.json") for fit_id in FIT_IDS},
           "evaluation.fit_state_hashes")
    _equal(evaluation.get("score_populations"), score_results, "evaluation.score_populations")
    gates = _score_gates(score_results, {key: value for key, value in fit_results.items()})
    _equal(evaluation.get("gates"), gates, "evaluation.gates")
    if evaluation.get("score_populations", {}).get("HINGE_D1", {}).get("target_sha256") != score_target_hashes["HINGE_D1"]:
        _fail("Independent score target hash is not reflected in evaluation")

    receipt = {"schema_version": 1, "experiment_id": runner.NAMESPACE.upper() + "_R1",
        "status": "PASS_INDEPENDENT_VERIFICATION", "source_commit": freeze["source_commit"],
        "prefit_freeze_sha256": freeze_sha, "identity_preflight_sha256": preflight_sha,
        "root_source_pass_sha256": pass_sha, "prediction_freeze_sha256": prediction_sha,
        "evaluation_sha256": evaluation_sha,
        "fit_state_hashes": {fit_id: _sha256(layout.fit_dir / f"{fit_id}.json") for fit_id in FIT_IDS},
        "fit_input_hashes": {fit_id: _sha256(runner._stage_files(layout, FIT_STAGE[fit_id])[0])
                              for fit_id in FIT_IDS},
        "fit_target_hashes": fit_target_hashes, "score_target_hashes": score_target_hashes,
        "prediction_hashes": {record["stage_id"]: record["sha256"]
                              for record in prediction_freeze["prediction_files"]},
        "ledger_event_count": sum(len(value) for value in ledger.values()),
        "fit_median_objectives_replayed": True, "prediction_values_replayed_bit_exact": True,
        "metrics_and_gates_replayed": True,
        "score_predictions_committed_before_score_target_load": True,
        "test_or_sample_accessed": False, "e79_accessed": False,
        "csv_created": False, "kaggle_upload": False}
    receipt["signature"] = _canonical_sha(receipt)
    digest = runner.write_json_once(layout.verification, receipt)
    return {"status": receipt["status"], "verification_path": str(layout.verification.resolve()),
            "verification_sha256": digest, "evaluation_sha256": evaluation_sha,
            "verified_fit_slots": len(FIT_IDS), "verified_score_windows": len(STAGES)}
