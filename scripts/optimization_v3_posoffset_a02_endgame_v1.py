"""Frozen zero-fit POS/A02 combination policies; no learner or preprocessing fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from src.models.classification_a02_neighbors import _array_sha256, _documents, _json_sha256  # noqa: E402

REPORT = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1"
RUN = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
RECIPE = REPORT / "RECIPE.json"
PREFLIGHT = REPORT / "PREFLIGHT.json"
FREEZE = REPORT / "FREEZE.json"
AUDIT = REPORT / "PREMETRIC_AUDIT.json"
METRICS = REPORT / "POLICY_METRICS.json"
PREDICTIONS = RUN / "policy_predictions.npz"

CACHE_PATHS = {
    55: "experiments/runs/optimization_v3_batch1/C16/C16_fit55.npz",
    59: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit59.npz",
    63: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit63.npz",
    67: "experiments/runs/optimization_v3_batch1/C16/C16_fit67.npz",
    71: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit71.npz",
    75: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit75.npz",
}
CACHE_WINDOWS = {55: (56, 67), 59: (60, 63), 63: (64, 67),
                 67: (68, 79), 71: (72, 75), 75: (76, 79)}
BANK_BLOCKS = {67: ((55, 56, 59), (59, 60, 63), (63, 64, 67)),
               75: ((55, 56, 59), (59, 60, 63), (63, 64, 67),
                    (67, 68, 71), (71, 72, 75))}
POS_PATHS = {
    67: "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/E67/predictions_E67.npz",
    75: "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/E75/predictions_E75.npz",
}
A02_ROOT = "experiments/runs/optimization_v3_a02_neighbors_v1"
A02_PATHS = {
    67: f"{A02_ROOT}/E67/observed/predictions.npz",
    75: f"{A02_ROOT}/E75/observed/predictions.npz",
}
A02_FIT_PATHS = {
    67: f"{A02_ROOT}/E67/observed",
    75: f"{A02_ROOT}/E75/observed",
}
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
A02_FREEZE = "experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
A02_PROTOCOL = "experiments/optimization_v3/c19_a02_v2/A02/protocol.json"
A02_FROZEN_SOURCES = (
    "src/models/classification_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "scripts/optimization_v3_a02_synthetic.py",
    "src/models/research_next_contract.py",
)
POS_FROZEN_SOURCES = (
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/verify_optimization_v3_positional_bag_offset_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "tests/test_optimization_v3_positional_bag_offset_v1.py",
)
ENDGAME_SOURCES = (
    "scripts/optimization_v3_posoffset_a02_endgame_v1.py",
    "scripts/verify_optimization_v3_posoffset_a02_endgame_v1.py",
    "tests/test_optimization_v3_posoffset_a02_endgame_v1.py",
    "reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json",
    "reports/optimization_v3/posoffset_a02_endgame_v1/BASELINE_CLARIFICATION_CORRECTION_20261104.md",
)
VIEWS = {
    "E67_H12": (67, 68, 79),
    "E67_W4": (67, 68, 71),
    "E67_W5": (67, 72, 75),
    "E67_W6": (67, 76, 79),
    "E75_W6": (75, 76, 79),
}
POLICIES = ("MIX50", "ADD_RESIDUAL")
COMPARATORS = ("pP", "pA", "b")
TOL = 1e-12


def _path(relative: str) -> Path:
    return ROOT / relative


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to overwrite immutable evidence: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise RuntimeError(f"Existing immutable NPZ schema differs: {path}")
            for name, value in arrays.items():
                if not np.array_equal(saved[name], value):
                    raise RuntimeError(f"Existing immutable NPZ differs at {name}: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _load_train(*, target: bool) -> pd.DataFrame:
    columns = ["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load"]
    if target:
        columns.append("artifact_signal")
    frame = pd.read_csv(TRAIN, usecols=columns, dtype={"record_id": str})
    if len(frame) != 50_608 or frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise RuntimeError("Official Train identity or row count mismatch")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise RuntimeError("Official Train cycle coverage mismatch")
    return frame


def _c16_cache(cutoff: int, train: pd.DataFrame) -> dict[str, np.ndarray]:
    path = _path(CACHE_PATHS[cutoff])
    side_path = path.with_suffix(".json")
    side = _json(side_path)
    if side.get("npz_sha256") != file_sha256(path):
        raise RuntimeError(f"C16 {cutoff} sidecar NPZ hash mismatch")
    if side.get("identity_sha256") != canonical_sha256(side.get("identity")):
        raise RuntimeError(f"C16 {cutoff} identity signature mismatch")
    if side.get("integrity_sha256") != canonical_sha256({k: v for k, v in side.items() if k != "integrity_sha256"}):
        raise RuntimeError(f"C16 {cutoff} sidecar integrity mismatch")
    if side.get("identity", {}).get("card_id") != "C16" or int(side["identity"].get("cutoff", -1)) != cutoff:
        raise RuntimeError(f"C16 {cutoff} identity/cutoff mismatch")
    data = _load_npz(path)
    required = {"score_rows", "score_record_ids", "baseline_prediction", "control__baseline_sparse_raw",
                "control__c2b_probability", "control__cb_probability", "control__stream_missing",
                "control__control_rows", "audit__stream_missing"}
    if not required.issubset(data):
        raise RuntimeError(f"Authoritative C16 {cutoff} artifact lacks required arrays")
    rows = data["score_rows"].astype(np.int64, copy=False)
    if (rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0)
            or not np.array_equal(data["score_record_ids"].astype(str), train.record_id.to_numpy()[rows])):
        raise RuntimeError(f"C16 {cutoff} rows/IDs do not align to official Train")
    missing = train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
    if (not np.array_equal(missing, data["control__stream_missing"])
            or not np.array_equal(missing, data["audit__stream_missing"])
            or not np.array_equal(rows, data["control__control_rows"])):
        raise RuntimeError(f"C16 {cutoff} stream mask/control rows mismatch")
    base = data["baseline_prediction"].astype(np.float64)
    cb = data["control__cb_probability"].astype(np.float64)
    c2b = data["control__c2b_probability"].astype(np.float64)
    sparse = data["control__baseline_sparse_raw"].astype(np.float64)
    for name, arr in (("baseline", base), ("CB", cb), ("C2_B", c2b), ("Sparse25", sparse)):
        if arr.shape != rows.shape or not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
            raise RuntimeError(f"C16 {cutoff} {name} values invalid")
    reconstructed = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse)
    if not np.array_equal(reconstructed, base):
        raise RuntimeError(f"C16 {cutoff} baseline route is not bit-exact")
    cycles = train.cycle_code.to_numpy(dtype=np.int64)[rows]
    start, stop = CACHE_WINDOWS[cutoff]
    if not np.array_equal(np.unique(cycles), np.arange(start, stop + 1)):
        raise RuntimeError(f"C16 {cutoff} is missing one or more frozen scoring cycles")
    for cycle in np.unique(cycles):
        if np.count_nonzero(cycles == cycle) != np.count_nonzero(train.cycle_code.to_numpy() == cycle):
            raise RuntimeError(f"C16 {cutoff} cycle {cycle} is incomplete")
    return {**data, "score_rows": rows, "cycles": cycles, "missing": missing,
            "baseline_prediction": base, "sidecar_sha256": np.asarray(file_sha256(side_path))}


def _all_preflight_inputs(train: pd.DataFrame) -> tuple[dict[str, str], dict[str, Any]]:
    source_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    source_files = set(A02_FROZEN_SOURCES + POS_FROZEN_SOURCES + ENDGAME_SOURCES)
    source_files.update((A02_FREEZE, A02_PROTOCOL,
                         "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
                         "reports/optimization_v3/positional_bag_offset_v1/RESULTS.json",
                         "reports/optimization_v3/a02_neighbors_v1/preflight.json",
                         "reports/optimization_v3/a02_neighbors_v1/verification.json",
                         "reports/optimization_v3/a02_neighbors_v1/technical_resume_receipt.json"))
    for relative in sorted(source_files):
        source_hashes[relative] = file_sha256(_path(relative))
    a02_freeze = _json(_path(A02_FREEZE))
    for relative, expected in a02_freeze.get("source_sha256", {}).items():
        actual = file_sha256(_path(relative))
        if actual != expected:
            raise RuntimeError(f"Frozen A02 source hash mismatch: {relative}")
    if a02_freeze.get("input_sha256", {}).get("data/raw/classification/Train.csv") != EXPECTED_TRAIN_SHA:
        raise RuntimeError("A02 freeze does not bind the expected official Train")
    if file_sha256(TRAIN) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("Official Train hash mismatch")
    input_hashes["data/raw/classification/Train.csv"] = file_sha256(TRAIN)

    counts: dict[str, Any] = {"c16": {}, "origins": {}}
    caches: dict[int, dict[str, np.ndarray]] = {}
    for cutoff, relative in sorted(CACHE_PATHS.items()):
        npz = _path(relative)
        side = npz.with_suffix(".json")
        expected_npz = a02_freeze["input_sha256"].get(relative)
        side_rel = side.relative_to(ROOT).as_posix()
        expected_side = a02_freeze["input_sha256"].get(side_rel)
        if expected_npz != file_sha256(npz) or expected_side != file_sha256(side):
            raise RuntimeError(f"C16 {cutoff} differs from A02 frozen input hash")
        cache = _c16_cache(cutoff, train)
        caches[cutoff] = cache
        input_hashes[relative] = file_sha256(npz)
        input_hashes[side_rel] = file_sha256(side)
        with np.load(npz, allow_pickle=False) as z:
            keys = sorted(z.files)
        counts["c16"][str(cutoff)] = {"npz_sha256": file_sha256(npz),
                                      "sidecar_sha256": file_sha256(side),
                                      "rows": len(cache["score_rows"]),
                                      "required_decomposition_present": True,
                                      "baseline_route_bit_exact": True,
                                      "keys": keys}

    origin_arrays: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for origin in (67, 75):
        pos_path, a_path = _path(POS_PATHS[origin]), _path(A02_PATHS[origin])
        a_side = a_path.with_suffix(".json")
        for path in (pos_path, a_path, a_side):
            rel = path.relative_to(ROOT).as_posix()
            input_hashes[rel] = file_sha256(path)
        a_sidecar = _json(a_side)
        if a_sidecar.get("npz_sha256") != file_sha256(a_path):
            raise RuntimeError(f"A02 E{origin} output sidecar hash mismatch")
        pos, a02 = _load_npz(pos_path), _load_npz(a_path)
        rows = pos.get("score_rows")
        required_pos = {"score_rows", "score_record_ids", "score_cycles", "present_mask",
                        "prob_offset_real", "prob_missing_hgb"}
        required_a02 = {"score_rows", "score_record_ids", "score_cycles", "baseline_prediction",
                        "prediction", "missing_mask", "neighbor_count", "neighbor_source_rows",
                        "neighbor_ids", "neighbor_similarities", "correction", "correction_unclipped"}
        if not required_pos.issubset(pos) or not required_a02.issubset(a02):
            raise RuntimeError(f"E{origin} frozen POS/A02 output schema incomplete")
        if not (np.array_equal(rows, a02["score_rows"])
                and np.array_equal(pos["score_record_ids"].astype(str), a02["score_record_ids"].astype(str))
                and np.array_equal(pos["score_cycles"], a02["score_cycles"])):
            raise RuntimeError(f"E{origin} POS/A02 row, ID, or cycle alignment mismatch")
        cycles = train.cycle_code.to_numpy(dtype=np.int64)[rows]
        expected_rows = np.flatnonzero((train.cycle_code.to_numpy(dtype=np.int64) > origin)
                                       & (train.cycle_code.to_numpy(dtype=np.int64) <= 79))
        if not np.array_equal(rows, expected_rows) or not np.array_equal(cycles, pos["score_cycles"].astype(np.int64)):
            raise RuntimeError(f"E{origin} query does not match the exact frozen cycle window")
        missing = train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
        if (not np.array_equal(pos["present_mask"], ~missing)
                or not np.array_equal(a02["missing_mask"], missing)):
            raise RuntimeError(f"E{origin} POS/A02 availability route mismatch")
        if not np.array_equal(a02["baseline_prediction"], caches[origin]["baseline_prediction"]):
            raise RuntimeError(f"E{origin} A02 b is not the exact matching C16 routed baseline")
        for key in ("prob_offset_real", "prob_missing_hgb", "prob_bag"):
            if key not in pos or pos[key].shape != rows.shape or not np.isfinite(pos[key]).all():
                raise RuntimeError(f"E{origin} POS {key} invalid")
            if np.any((pos[key] < 0) | (pos[key] > 1)):
                raise RuntimeError(f"E{origin} POS {key} outside [0,1]")
        if not np.array_equal(pos["prob_bag"][missing], pos["prob_missing_hgb"][missing]):
            raise RuntimeError(f"E{origin} POS missing fallback is not exact")
        p_pos = np.where(missing, pos["prob_missing_hgb"], pos["prob_offset_real"]).astype(np.float64)
        for key in ("baseline_prediction", "prediction"):
            arr = a02[key]
            if arr.shape != rows.shape or not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
                raise RuntimeError(f"E{origin} A02 {key} invalid")
        if (not np.array_equal(a02["prediction"][missing], a02["baseline_prediction"][missing])
                or not np.array_equal(a02["prediction"], np.clip(a02["baseline_prediction"] + a02["correction"], 0, 1))
                or not np.array_equal(a02["correction"][missing], np.zeros(int(missing.sum())))):
            raise RuntimeError(f"E{origin} A02 saved prediction does not reproduce its saved correction route")
        origin_arrays[origin] = {"pos": pos, "a02": a02,
                                 "data": {"rows": rows.astype(np.int64), "ids": pos["score_record_ids"].astype(str),
                                          "cycles": cycles, "missing": missing, "p_pos": p_pos,
                                          "p_a": a02["prediction"].astype(np.float64),
                                          "b": a02["baseline_prediction"].astype(np.float64)}}
        counts["origins"][str(origin)] = {"rows": len(rows), "present": int((~missing).sum()),
                                           "missing": int(missing.sum()), "row_id_cycle_mask_exact": True,
                                           "b_equals_C16_routed_baseline_bit_exact": True,
                                           "pos_sha256": file_sha256(pos_path),
                                           "a02_sha256": file_sha256(a_path)}

    # Fit artifacts and output receipts are included in the immutable input identity.
    for origin, folder_rel in A02_FIT_PATHS.items():
        folder = _path(folder_rel)
        for name in ("model.joblib", "fit_state.json", "fit_receipt.json", "predictions.json"):
            path = folder / name
            input_hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)
        receipt = _json(folder / "fit_receipt.json")
        fit_state = _json(folder / "fit_state.json")
        pred_side = _json(folder / "predictions.json")
        if (receipt.get("fit_id") != f"A02_E{origin}_OBSERVED"
                or receipt.get("freeze_signature") != _json(_path(A02_FREEZE))["freeze_signature"]
                or receipt.get("model_sha256") != file_sha256(folder / "model.joblib")
                or receipt.get("state_sha256") != canonical_sha256(fit_state)
                or pred_side.get("npz_sha256") != input_hashes[A02_PATHS[origin]]
                or pred_side.get("fit_receipt_sha256") != receipt.get("receipt_sha256")):
            raise RuntimeError(f"A02 E{origin} fit/output receipts do not bind the frozen artifacts")

    return source_hashes, input_hashes, {"counts": counts, "origins_data": origin_arrays, "caches": caches}


def preflight(*, write: bool = True) -> dict[str, Any]:
    train = _load_train(target=False)
    source_hashes, input_hashes, details = _all_preflight_inputs(train)
    value = {
        "status": "PASS_NO_TARGET_NO_TEST",
        "official_train_rows": len(train),
        "official_train_sha256": input_hashes["data/raw/classification/Train.csv"],
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "c16": details["counts"]["c16"],
        "origins": details["counts"]["origins"],
        "policy_ids": list(POLICIES),
        "query_target_read": False,
        "test_samplesubmission_read": False,
        "fit_or_ledger_event": False,
    }
    if write:
        _write_json_once(PREFLIGHT, value)
    return value


def freeze() -> dict[str, Any]:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Freeze requires a clean checkout after the source/protocol commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    observed = preflight(write=False)
    recorded = _json(PREFLIGHT)
    if recorded != observed:
        raise RuntimeError("Current no-target preflight differs from committed PREFLIGHT")
    frozen = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_POLICY_METRICS",
        "prepared_source_commit": head,
        "source_hashes": observed["source_hashes"],
        "input_hashes": observed["input_hashes"],
        "recipe_sha256": file_sha256(RECIPE),
        "preflight_sha256": file_sha256(PREFLIGHT),
        "policy_ids_in_priority_order": list(POLICIES),
        "formulas": {
            "MIX50": {"present": "0.5*pP + 0.5*pA", "missing": "pP exactly"},
            "ADD_RESIDUAL": {"present": "clip(pP + (pA - b), 0, 1)", "missing": "pP exactly",
                             "b": "A02 baseline_prediction; same-origin C16 baseline_prediction; not raw sparse"},
        },
        "gates": _json(RECIPE)["gates"],
        "adaptive_selection_disclosed": True,
        "query_targets_used": False,
        "test_or_samplesubmission_accessed": False,
        "new_fits": 0,
        "ledger_mutation": False,
    }
    frozen["freeze_signature"] = canonical_sha256(frozen)
    _write_json_once(FREEZE, frozen)
    return frozen


def _assert_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_data = _json(FREEZE)
    signature = freeze_data.get("freeze_signature")
    identity = dict(freeze_data)
    identity.pop("freeze_signature", None)
    if signature != canonical_sha256(identity):
        raise RuntimeError("Endgame freeze signature invalid")
    if freeze_data.get("status") != "FROZEN_BEFORE_POLICY_METRICS":
        raise RuntimeError("Endgame policy freeze is absent")
    current = preflight(write=False)
    if (current["source_hashes"] != freeze_data.get("source_hashes")
            or current["input_hashes"] != freeze_data.get("input_hashes")
            or file_sha256(RECIPE) != freeze_data.get("recipe_sha256")
            or file_sha256(PREFLIGHT) != freeze_data.get("preflight_sha256")):
        raise RuntimeError("A frozen source or input changed after freeze")
    return freeze_data, current


def _bank_for_origin(origin: int, caches: dict[int, dict[str, np.ndarray]], train: pd.DataFrame) -> dict[str, np.ndarray]:
    blocks = BANK_BLOCKS[origin]
    parts: list[dict[str, np.ndarray]] = []
    for cutoff, first, last in blocks:
        cache = caches[cutoff]
        use = (cache["cycles"] >= first) & (cache["cycles"] <= last)
        rows = cache["score_rows"][use]
        expected = np.flatnonzero(train.cycle_code.between(first, last).to_numpy())
        if not np.array_equal(rows, expected):
            raise RuntimeError(f"A02 OOF block {first}-{last} is incomplete or reordered")
        parts.append({"rows": rows, "ids": cache["score_record_ids"][use].astype(str),
                      "cycles": cache["cycles"][use], "baseline": cache["baseline_prediction"][use],
                      "missing": cache["missing"][use]})
    rows_all = np.concatenate([part["rows"] for part in parts])
    order = np.argsort(rows_all, kind="stable")
    result = {key: np.concatenate([part[key] for part in parts])[order]
              for key in ("rows", "ids", "cycles", "baseline", "missing")}
    expected = np.flatnonzero(train.cycle_code.between(56, origin).to_numpy())
    if not np.array_equal(result["rows"], expected) or np.any(result["missing"]):
        raise RuntimeError(f"A02 E{origin} donor row universe mismatch or missing-stream donor")
    result["present_rows"] = result["rows"].astype(np.int64)
    result["present_ids"] = result["ids"].astype(str)
    result["present_cycles"] = result["cycles"].astype(np.int64)
    result["present_baseline"] = result["baseline"].astype(np.float64)
    return result


def _verify_saved_a02(origin: int, train: pd.DataFrame, cache: dict[str, np.ndarray],
                      pos: dict[str, np.ndarray], a02: dict[str, np.ndarray]) -> dict[str, Any]:
    folder = _path(A02_FIT_PATHS[origin])
    receipt = _json(folder / "fit_receipt.json")
    state_file = _json(folder / "fit_state.json")
    model = joblib.load(folder / "model.joblib")
    state = model.audit_state()
    if state != state_file:
        raise RuntimeError(f"A02 E{origin} joblib state differs from fit_state.json")
    receipt_without_sig = dict(receipt)
    receipt_sig = receipt_without_sig.pop("receipt_sha256", None)
    if (receipt_sig != canonical_sha256(receipt_without_sig)
            or receipt.get("state_sha256") != canonical_sha256(state)
            or receipt.get("model_sha256") != file_sha256(folder / "model.joblib")
            or receipt.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False):
        raise RuntimeError(f"A02 E{origin} fitted model/receipt lineage invalid")
    state_without_hash = dict(state)
    internal_hash = state_without_hash.pop("state_sha256", None)
    if internal_hash != _json_sha256(state_without_hash):
        raise RuntimeError(f"A02 E{origin} internal model state checksum invalid")

    bank = _bank_for_origin(origin, {c: _c16_cache(c, train) for c in CACHE_PATHS}, train)
    y = pd.to_numeric(train.loc[bank["present_rows"], "artifact_signal"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isin(y, [0.0, 1.0]).all():
        raise RuntimeError("A02 OOF donor target contains non-binary values")
    residual = y - bank["present_baseline"]
    model_rows = np.asarray(model.bank_row_positions, dtype=np.int64)
    model_cycles = np.asarray(model.bank_cycles, dtype=np.int64)
    model_prob = np.asarray(model.bank_oof_probabilities, dtype=np.float64)
    model_residual = np.asarray(model.bank_residuals, dtype=np.float64)
    model_ids = tuple(model.bank_ids)
    expected_ids = tuple(f"{type(value).__name__}:{value}"
                         for value in np.asarray(bank["present_ids"]).astype(str))
    if (not np.array_equal(model_rows, bank["present_rows"])
            or not np.array_equal(model_cycles, bank["present_cycles"])
            or not np.array_equal(model_prob, bank["present_baseline"])
            or not np.array_equal(model_residual, residual)
            or model_ids != expected_ids):
        raise RuntimeError(f"A02 E{origin} donor fitted-state arrays differ from exact C16 OOF donors")
    if (state.get("bank_rows") != len(model_rows)
            or state.get("bank_ids_sha256") != _array_sha256(model_ids)
            or state.get("bank_row_positions_sha256") != _array_sha256(model_rows, "<i8")
            or state.get("bank_cycles_sha256") != _array_sha256(model_cycles, "<i8")
            or state.get("bank_oof_probability_sha256") != _array_sha256(model_prob, "<f8")
            or state.get("bank_original_residual_sha256") != _array_sha256(residual, "<f8")):
        raise RuntimeError(f"A02 E{origin} donor state hashes differ from independent OOF reconstruction")

    prefix_rows = np.flatnonzero(train.cycle_code.to_numpy(dtype=np.int64) <= origin)
    if (model.text_space.origin_cycle != origin or model.text_space.fit_rows != len(prefix_rows)
            or model.text_space.fit_row_positions_sha256 != receipt.get("text_prefix_row_positions_sha256")
            or model.text_space.fit_row_positions_sha256 != _array_sha256(prefix_rows, "<i8")):
        raise RuntimeError(f"A02 E{origin} text preprocessing prefix identity mismatch")
    prefix_docs, _ = _documents(train.loc[prefix_rows, "signal_stream"])
    if model.text_space.fit_documents_sha256 != _json_sha256(prefix_docs):
        raise RuntimeError(f"A02 E{origin} saved text-space prefix content identity mismatch")
    transformed, missing = model.text_space.transform(train.loc[model_rows, "signal_stream"])
    if (missing.any() or transformed.shape != model.bank_matrix.shape
            or not np.array_equal(transformed.indptr, model.bank_matrix.indptr)
            or not np.array_equal(transformed.indices, model.bank_matrix.indices)
            or not np.array_equal(transformed.data, model.bank_matrix.data)
            or state.get("bank_matrix_data_sha256") != _array_sha256(transformed.data, "<f8")):
        raise RuntimeError(f"A02 E{origin} saved donor TF-IDF matrix does not replay")

    rows = a02["score_rows"].astype(np.int64)
    cycles = a02["score_cycles"].astype(np.int64)
    miss = a02["missing_mask"].astype(bool)
    neighbor_rows = a02["neighbor_source_rows"].astype(np.int64)
    neighbor_ids = a02["neighbor_ids"].astype(str)
    similarities = a02["neighbor_similarities"].astype(np.float64)
    counts = a02["neighbor_count"].astype(np.int64)
    if neighbor_rows.shape != neighbor_ids.shape or neighbor_rows.shape != similarities.shape:
        raise RuntimeError(f"A02 E{origin} saved donor arrays are shape-misaligned")
    bank_index = {int(row): i for i, row in enumerate(model_rows)}
    raw = np.zeros(len(rows), dtype=np.float64)
    for i, query_row in enumerate(rows):
        count = int(counts[i])
        if count < 0 or count > neighbor_rows.shape[1]:
            raise RuntimeError("A02 saved neighbor count invalid")
        active_rows = neighbor_rows[i, :count]
        active_ids = neighbor_ids[i, :count]
        active_s = similarities[i, :count]
        padding_rows = neighbor_rows[i, count:]
        padding_ids = neighbor_ids[i, count:]
        padding_s = similarities[i, count:]
        if (np.any(padding_rows != -1) or np.any(padding_ids != "") or np.any(padding_s != 0.0)
                or not np.isfinite(active_s).all() or np.any(active_s <= 0.0)):
            raise RuntimeError(f"A02 E{origin} neighbor padding or similarity invalid")
        if miss[i] and count != 0:
            raise RuntimeError("A02 missing-stream row has lexical neighbors")
        found = []
        for donor_row, donor_id in zip(active_rows.tolist(), active_ids.tolist()):
            if donor_row not in bank_index:
                raise RuntimeError(f"A02 E{origin} neighbor row is outside saved OOF donor bank: {donor_row}")
            donor_i = bank_index[donor_row]
            if (donor_id != model_ids[donor_i]
                    or bank["present_cycles"][donor_i] >= cycles[i]
                    or donor_row == int(query_row)):
                raise RuntimeError(f"A02 E{origin} neighbor ID/cycle/self guard failed")
            found.append(donor_i)
        ordering = np.lexsort((model_rows[found], -active_s)) if found else np.asarray([], dtype=np.int64)
        if not np.array_equal(ordering, np.arange(len(found))):
            raise RuntimeError(f"A02 E{origin} neighbors are not sorted by similarity then source row")
        if count:
            weights = active_s * active_s
            raw[i] = float(np.dot(weights, model_residual[np.asarray(found, dtype=np.int64)])) / (4.0 + float(weights.sum()))
    correction = np.clip(raw, -0.05, 0.05)
    correction[miss] = 0.0
    expected_p_a = np.clip(a02["baseline_prediction"].astype(np.float64) + correction, 0.0, 1.0)
    expected_p_a[miss] = a02["baseline_prediction"][miss]
    for name, actual, expected in (("raw correction", a02["correction_unclipped"], raw),
                                   ("capped correction", a02["correction"], correction),
                                   ("prediction", a02["prediction"], expected_p_a)):
        if not np.array_equal(actual, expected):
            err = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            raise RuntimeError(f"A02 E{origin} {name} did not reconstruct (max_abs={err:.17g})")
    if not np.array_equal(a02["baseline_prediction"], cache["baseline_prediction"]):
        raise RuntimeError(f"A02 E{origin} baseline does not match same-origin C16 route")
    return {
        "status": "PASS",
        "fit_id": receipt["fit_id"],
        "model_sha256": receipt["model_sha256"],
        "fit_state_sha256": file_sha256(folder / "fit_state.json"),
        "fit_receipt_sha256": file_sha256(folder / "fit_receipt.json"),
        "bank_rows": int(len(model_rows)),
        "bank_residuals_reconstructed_from_train_targets": True,
        "text_prefix_state_replayed_without_refit": True,
        "saved_neighbor_rows_ids_cycles_order_reconstructed": int(np.sum(counts)),
        "saved_raw_capped_correction_and_prediction_bit_exact": True,
        "query_targets_used_for_audit": False,
        "test_used": False,
    }


def audit() -> dict[str, Any]:
    freeze_data, _ = _assert_frozen_inputs()
    if not FREEZE.exists():
        raise RuntimeError("Metrics/audit forbidden before committed freeze")
    train = _load_train(target=True)
    caches = {c: _c16_cache(c, train) for c in CACHE_PATHS}
    output = {"status": "PASS", "freeze_signature": freeze_data["freeze_signature"],
              "origins": {}, "query_targets_used_for_audit": False, "test_used": False}
    for origin in (67, 75):
        pos = _load_npz(_path(POS_PATHS[origin]))
        a02 = _load_npz(_path(A02_PATHS[origin]))
        output["origins"][str(origin)] = _verify_saved_a02(origin, train, caches[origin], pos, a02)
    _write_json_once(AUDIT, output)
    return output


def combine(policy: str, p_pos: Any, p_a: Any, b: Any, missing: Any) -> np.ndarray:
    if policy not in POLICIES:
        raise ValueError("Policy must be one of the two frozen choices")
    arrays = [np.asarray(v, dtype=np.float64) for v in (p_pos, p_a, b)]
    mask = np.asarray(missing, dtype=bool)
    if any(arr.ndim != 1 for arr in arrays) or mask.ndim != 1:
        raise ValueError("Probabilities and mask must be vectors")
    if any(arr.shape != arrays[0].shape for arr in arrays[1:]) or mask.shape != arrays[0].shape:
        raise ValueError("Probability and mask shapes differ")
    if any(not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)) for arr in arrays):
        raise ValueError("Probabilities must be finite in [0,1]")
    if policy == "MIX50":
        result = 0.5 * arrays[0] + 0.5 * arrays[1]
    else:
        result = np.clip(arrays[0] + (arrays[1] - arrays[2]), 0.0, 1.0)
    result[mask] = arrays[0][mask]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise RuntimeError("Frozen combination produced invalid probabilities")
    return result


def _f1_pair(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    pos_recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"n": int(len(y)), "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
            "confusion_matrix": [[tn, fp], [fn, tp]], "macro_f1": 0.5 * (f0 + f1),
            "positive_recall": pos_recall}


def _text_groups(train: pd.DataFrame, origin: int, rows: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cycles = train.cycle_code.to_numpy(dtype=np.int64)
    prefix = pd.to_numeric(train.loc[cycles <= origin, "text_load"], errors="raise").to_numpy(dtype=np.float64)
    score = pd.to_numeric(train.loc[rows, "text_load"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(prefix).all() or not np.isfinite(score).all():
        raise RuntimeError("Fit-defined text_load group values must be finite")
    edges = np.quantile(prefix, [0.25, 0.50, 0.75], method="linear")
    return np.searchsorted(edges, score, side="right").astype(np.int8), [float(x) for x in edges]


def _views_for_origin(origin: int, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for view, (view_origin, first, last) in VIEWS.items():
        if origin != view_origin:
            continue
        mask = (data["cycles"] >= first) & (data["cycles"] <= last)
        result[view] = mask
    return result


def score() -> dict[str, Any]:
    freeze_data, _ = _assert_frozen_inputs()
    audit_data = _json(AUDIT)
    if audit_data.get("status") != "PASS" or audit_data.get("freeze_signature") != freeze_data["freeze_signature"]:
        raise RuntimeError("Independent source/artifact audit must pass before outcome metrics")
    train = _load_train(target=True)
    pred_arrays: dict[str, np.ndarray] = {}
    origin_metric: dict[str, Any] = {}
    result: dict[str, Any] = {
        "status": "SCORED_FROZEN_POLICIES",
        "freeze_signature": freeze_data["freeze_signature"],
        "adaptive_post_public_selection": True,
        "E67_E75_overlap_not_replication": True,
        "views": {}, "policies": {}, "selection": {},
        "comparators": list(COMPARATORS),
    }
    all_policy_probs: dict[str, dict[int, np.ndarray]] = {p: {} for p in POLICIES}
    all_inputs: dict[int, dict[str, np.ndarray]] = {}
    all_groups: dict[int, dict[str, np.ndarray]] = {}
    edges_by_origin = {}
    for origin in (67, 75):
        pos = _load_npz(_path(POS_PATHS[origin]))
        a02 = _load_npz(_path(A02_PATHS[origin]))
        rows = pos["score_rows"].astype(np.int64)
        ids = pos["score_record_ids"].astype(str)
        y = pd.to_numeric(train.loc[rows, "artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y, [0, 1]).all() or not np.array_equal(ids, train.record_id.to_numpy()[rows]):
            raise RuntimeError(f"E{origin} target join failed after freeze")
        missing = a02["missing_mask"].astype(bool)
        p_pos = np.where(missing, pos["prob_missing_hgb"], pos["prob_offset_real"]).astype(np.float64)
        p_a = a02["prediction"].astype(np.float64)
        b = a02["baseline_prediction"].astype(np.float64)
        all_inputs[origin] = {"rows": rows, "ids": ids, "cycles": a02["score_cycles"].astype(np.int16),
                              "y": y, "p_pos": p_pos, "p_a": p_a, "b": b, "missing": missing}
        tgroup, edges = _text_groups(train, origin, rows)
        friction = pd.to_numeric(train.loc[rows, "entry_friction"], errors="raise").to_numpy(dtype=np.int8)
        groups = {"availability": np.where(missing, "missing", "present"),
                  "text_load": np.asarray([f"Q{int(x)+1}" for x in tgroup]),
                  "entry_friction": np.asarray([f"{int(x)}" for x in friction])}
        all_groups[origin] = groups
        edges_by_origin[str(origin)] = edges
        for policy in POLICIES:
            all_policy_probs[policy][origin] = combine(policy, p_pos, p_a, b, missing)
        pred_arrays[f"E{origin}_rows"] = rows.astype(np.int64)
        pred_arrays[f"E{origin}_ids"] = ids.astype(str)
        pred_arrays[f"E{origin}_cycles"] = a02["score_cycles"].astype(np.int16)
        pred_arrays[f"E{origin}_pP"] = p_pos
        pred_arrays[f"E{origin}_pA"] = p_a
        pred_arrays[f"E{origin}_b"] = b
        pred_arrays[f"E{origin}_missing"] = missing
        pred_arrays[f"E{origin}_y"] = y
        for policy in POLICIES:
            pred_arrays[f"E{origin}_{policy}"] = all_policy_probs[policy][origin]
        for family, values in groups.items():
            pred_arrays[f"E{origin}_group_{family}"] = values.astype(str)
        for view, mask in _views_for_origin(origin, all_inputs[origin]).items():
            view_data: dict[str, Any] = {"origin": f"E{origin}", "rows": int(mask.sum()),
                                         "cycle_range": [int(a02["score_cycles"][mask].min()),
                                                         int(a02["score_cycles"][mask].max())],
                                         "comparators": {}, "policies": {}, "subgroups": {}}
            comparators = {"pP": p_pos, "pA": p_a, "b": b}
            for name, prob in comparators.items():
                view_data["comparators"][name] = _f1_pair(y[mask], (prob[mask] >= 0.5).astype(np.int8))
            for policy in POLICIES:
                candidate = all_policy_probs[policy][origin]
                view_data["policies"][policy] = _f1_pair(y[mask], (candidate[mask] >= 0.5).astype(np.int8))
            for family, values in groups.items():
                group_entries = {}
                for level in np.unique(values[mask]):
                    gmask = mask & (values == level)
                    gy = y[gmask]
                    support = {"n": int(gmask.sum()), "class_0": int(np.sum(gy == 0)),
                               "class_1": int(np.sum(gy == 1))}
                    supported = support["n"] >= 200 and support["class_0"] >= 30 and support["class_1"] >= 30
                    entry: dict[str, Any] = {"support": support, "status": "SUPPORTED" if supported else "UNKNOWN_SUPPORT"}
                    if supported:
                        base_metric = _f1_pair(gy, (p_pos[gmask] >= 0.5).astype(np.int8))
                        entry["pP"] = base_metric
                        for policy in POLICIES:
                            candidate = all_policy_probs[policy][origin]
                            met = _f1_pair(gy, (candidate[gmask] >= 0.5).astype(np.int8))
                            entry[policy] = met
                            entry[f"{policy}_harm_vs_pP"] = {
                                "macro_f1": base_metric["macro_f1"] - met["macro_f1"],
                                "positive_recall": base_metric["positive_recall"] - met["positive_recall"],
                            }
                        for name, prob in (("pA", p_a), ("b", b)):
                            entry[name] = _f1_pair(gy, (prob[gmask] >= 0.5).astype(np.int8))
                    group_entries[str(level)] = entry
                view_data["subgroups"][family] = group_entries
            result["views"][view] = view_data

    result["fit_defined_text_load_quartile_edges"] = edges_by_origin
    policy_gates: dict[str, Any] = {}
    for policy in POLICIES:
        checks = {}
        gains = {}
        for view in ("E67_H12", "E75_W6"):
            gain = (result["views"][view]["policies"][policy]["macro_f1"]
                    - result["views"][view]["comparators"]["pP"]["macro_f1"])
            gains[view] = gain
            checks[f"gain_{view}_at_least_0.0005"] = gain + TOL >= 0.0005
        window_harms = {}
        for view in ("E67_W4", "E67_W5", "E67_W6"):
            harm = (result["views"][view]["comparators"]["pP"]["macro_f1"]
                    - result["views"][view]["policies"][policy]["macro_f1"])
            window_harms[view] = harm
            checks[f"harm_{view}_at_most_0.003"] = harm <= 0.003 + TOL
        subgroup_failures = []
        unsupported = []
        for view, view_data in result["views"].items():
            for family, levels in view_data["subgroups"].items():
                for level, entry in levels.items():
                    key = f"{view}/{family}/{level}"
                    if entry["status"] != "SUPPORTED":
                        unsupported.append(key)
                        continue
                    harm = entry[f"{policy}_harm_vs_pP"]
                    if harm["macro_f1"] > 0.01 + TOL or harm["positive_recall"] > 0.02 + TOL:
                        subgroup_failures.append({"group": key, **harm})
        checks["all_supported_subgroups_within_limits"] = not subgroup_failures
        checks["lineage_and_numeric_audit_pass"] = audit_data.get("status") == "PASS"
        qualifies = all(checks.values())
        policy_gates[policy] = {"qualifies": qualifies, "checks": checks,
                                "required_view_gains": gains, "E67_window_harms": window_harms,
                                "subgroup_failures": subgroup_failures,
                                "unsupported_subgroups_unknown": unsupported}
    result["policies"] = policy_gates
    selected = next((policy for policy in POLICIES if policy_gates[policy]["qualifies"]), None)
    result["selection"] = {"selected_policy_by_frozen_priority": selected,
                            "priority": list(POLICIES),
                            "upload_or_candidate_prep_authorized": False if selected is None else "selected only; no upload by Luna",
                            "internal_eligibility_only": True}
    result["target_policy"] = "Train labels read only after committed freeze and successful premetric audit; query labels used here only for scoring"
    _write_npz_once(PREDICTIONS, pred_arrays)
    result["prediction_npz_sha256"] = file_sha256(PREDICTIONS)
    _write_json_once(METRICS, result)
    return result


def build_submission(ids: Any, probabilities: Any, expected_ids: Any) -> pd.DataFrame:
    rid = np.asarray(ids).astype(str)
    exp = np.asarray(expected_ids).astype(str)
    prob = np.asarray(probabilities, dtype=np.float64)
    if rid.ndim != 1 or exp.ndim != 1 or prob.ndim != 1 or len(rid) != len(prob):
        raise ValueError("Submission ids/probabilities are not aligned vectors")
    if len(np.unique(rid)) != len(rid) or not np.array_equal(rid, exp):
        raise ValueError("Submission IDs must match all official SampleSubmission IDs in exact order")
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Submission probabilities must be finite and bounded")
    return pd.DataFrame({"record_id": rid, "artifact_signal": (prob >= 0.5).astype(np.int8)})


def prepare_e79_candidate(*, policy: str, pos_npz: Path, a02_npz: Path, c16_npz: Path,
                          sample_csv: Path, output_csv: Path) -> dict[str, Any]:
    """Conditional Test-only inference from frozen probabilities; no fit or labels."""
    frozen_metrics = _json(METRICS)
    if policy not in POLICIES or frozen_metrics.get("selection", {}).get("selected_policy_by_frozen_priority") != policy:
        raise RuntimeError("E79 access is allowed only for the frozen selected policy")
    pos, a02, c16 = _load_npz(pos_npz), _load_npz(a02_npz), _load_npz(c16_npz)
    for bundle, name in ((pos, "POS"), (a02, "A02"), (c16, "C16")):
        for field in ("score_rows", "score_record_ids"):
            if field not in bundle:
                raise RuntimeError(f"E79 {name} bundle missing {field}")
    ids = pos["score_record_ids"].astype(str)
    rows = pos["score_rows"].astype(np.int64)
    if (not np.array_equal(ids, a02["score_record_ids"].astype(str))
            or not np.array_equal(rows, a02["score_rows"].astype(np.int64))
            or not np.array_equal(ids, c16["score_record_ids"].astype(str))
            or not np.array_equal(rows, c16["score_rows"].astype(np.int64))):
        raise RuntimeError("E79 source bundle row/ID alignment mismatch")
    required = {"present_mask", "prob_offset_real", "prob_missing_hgb"}
    if not required.issubset(pos):
        raise RuntimeError("E79 POS probability fields are incomplete")
    miss = ~pos["present_mask"].astype(bool)
    p_pos = np.where(miss, pos["prob_missing_hgb"], pos["prob_offset_real"]).astype(np.float64)
    p_a, b = a02["prediction"].astype(np.float64), a02["baseline_prediction"].astype(np.float64)
    if not np.array_equal(b, c16["baseline_prediction"].astype(np.float64)):
        raise RuntimeError("E79 A02 b differs from same-origin C16 route")
    prob = combine(policy, p_pos, p_a, b, miss)
    sample = pd.read_csv(sample_csv, dtype={"record_id": str})
    if list(sample.columns) != ["record_id", "artifact_signal"] or len(sample) != 29_421:
        raise RuntimeError("SampleSubmission schema or row count mismatch")
    submission = build_submission(ids, prob, sample.record_id.to_numpy(dtype=str))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        existing = pd.read_csv(output_csv, dtype={"record_id": str})
        if not existing.equals(submission):
            raise RuntimeError("Refusing to overwrite a different candidate CSV")
    else:
        submission.to_csv(output_csv, index=False)
    label_hash = hashlib.sha256(submission.artifact_signal.to_numpy(dtype=np.int8).tobytes()).hexdigest()
    existing_matches = []
    submissions_root = ROOT / "submissions"
    for candidate in submissions_root.rglob("*.csv"):
        if candidate.resolve() == output_csv.resolve():
            continue
        try:
            old = pd.read_csv(candidate, usecols=["record_id", "artifact_signal"], dtype={"record_id": str})
        except Exception:
            continue
        if (len(old) == len(submission) and np.array_equal(old.record_id.to_numpy(dtype=str), submission.record_id.to_numpy(dtype=str))
                and np.array_equal(old.artifact_signal.to_numpy(dtype=np.int8), submission.artifact_signal.to_numpy(dtype=np.int8))):
            existing_matches.append(candidate.relative_to(ROOT).as_posix())
    if existing_matches:
        raise RuntimeError(f"Selected candidate is label-identical to an existing submission: {existing_matches}")
    manifest = {"status": "PREPARED_NOT_SUBMITTED", "policy": policy,
                "csv_path": output_csv.relative_to(ROOT).as_posix(),
                "csv_sha256": file_sha256(output_csv), "rows": len(submission),
                "columns": list(submission.columns), "unique_ids": int(submission.record_id.nunique()),
                "positive_count": int(submission.artifact_signal.sum()),
                "label_vector_sha256": label_hash, "existing_submission_identical_files": existing_matches,
                "source_npz_sha256": {"pos": file_sha256(pos_npz), "a02": file_sha256(a02_npz), "c16": file_sha256(c16_npz)},
                "sample_submission_sha256": file_sha256(sample_csv),
                "target_labels_loaded": False, "new_fits": 0, "upload_performed": False}
    _write_json_once(REPORT / "E79_SELECTED_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "freeze", "audit", "score"))
    args = parser.parse_args()
    if args.command == "preflight":
        value = preflight()
    elif args.command == "freeze":
        value = freeze()
    elif args.command == "audit":
        value = audit()
    else:
        value = score()
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
