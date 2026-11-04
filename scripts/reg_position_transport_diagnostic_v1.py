"""Zero-fit transport diagnosis of the frozen REG_POSITION_L1_V1 artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np

CARD = "REG_POSITION_L1_V1"
ORIGINS = (67, 75)
ARMS = ("BAG", "REAL", "SHUFFLED")
FITS = tuple(f"{CARD}_E{o}_{a}" for o in ORIGINS for a in ARMS)
PILOT = Path("reports/optimization_v3/reg_position_l1_v1")
RUN = Path("experiments/runs/reg_position_l1_v1")
TRAIN = Path("data/raw/regression/Train.csv")
OUT_REL = Path("reports/optimization_v3/reg_position_transport_diagnostic_v1")
MIN_N = 100


class DiagnosticError(RuntimeError):
    pass


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _canon_sha(value: Any) -> str:
    return hashlib.sha256(_canon(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(_canon({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(_canon(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _file(root: Path, rel: str | Path) -> Path:
    root = root.resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_file():
        raise DiagnosticError(f"Input escapes declared root or is missing: {rel}")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise DiagnosticError(f"Refusing to overwrite different diagnostic output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_once(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return format(value, ".12g") if math.isfinite(value) else ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows([{key: _csv_value(row.get(key)) for key in columns} for row in rows])
    _write_once(path, buffer.getvalue().encode("utf-8"))


def fixed_decile(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 100.0)
    return np.minimum(np.floor(clipped / 10.0), 9).astype(np.int8)


def distribution(values: np.ndarray, min_n: int = MIN_N) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < min_n:
        return {"n": int(len(values)), "status": "UNKNOWN_UNSUPPORTED",
                "median": None, "p10": None, "p90": None}
    p10, median, p90 = np.quantile(values, [0.1, 0.5, 0.9], method="linear")
    return {"n": int(len(values)), "status": "SUPPORTED",
            "median": float(median), "p10": float(p10), "p90": float(p90)}


def pair_by_id(ids_left: np.ndarray, ids_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(ids_left, dtype=str)
    right = np.asarray(ids_right, dtype=str)
    if (left.ndim != 1 or right.ndim != 1 or len(set(left.tolist())) != len(left)
            or len(set(right.tolist())) != len(right)):
        raise DiagnosticError("Paired record IDs must be unique one-dimensional vectors")
    lmap = {record_id: i for i, record_id in enumerate(left.tolist())}
    rmap = {record_id: i for i, record_id in enumerate(right.tolist())}
    if set(lmap) != set(rmap):
        raise DiagnosticError("W6 record-ID cohorts are not identical")
    ids = sorted(lmap)
    return (np.asarray([lmap[key] for key in ids], dtype=np.int64),
            np.asarray([rmap[key] for key in ids], dtype=np.int64))


def _verify_freeze(source: Path, worktree: Path) -> tuple[dict[str, Any], dict[str, str]]:
    freeze = _json(worktree / PILOT / "FREEZE.json")
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if (freeze.get("card_id") != CARD or freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_FITS"
            or freeze.get("freeze_signature") != _canon_sha(body)
            or freeze.get("fit_ids_order") != list(FITS)
            or freeze.get("accounting", {}).get("pilot_cap") != 6):
        raise DiagnosticError("Frozen pilot signature/order/accounting is invalid")
    paths = {}
    for key in freeze.get("source_hashes", {}):
        if not key.startswith("worktree/"):
            raise DiagnosticError("Invalid pilot worktree hash namespace")
        paths[key] = _file(worktree, key.removeprefix("worktree/"))
    for key in freeze.get("input_hashes", {}):
        if not key.startswith("main/"):
            raise DiagnosticError("Invalid pilot main-input hash namespace")
        paths[key] = _file(source, key.removeprefix("main/"))
    actual = {key: _sha(path) for key, path in sorted(paths.items())}
    expected = {**freeze["source_hashes"], **freeze["input_hashes"]}
    if actual != expected:
        raise DiagnosticError("A frozen pilot source or main input hash changed")
    return freeze, actual


def _pilot_paths(worktree: Path) -> dict[str, Path]:
    paths = {
        "FREEZE": worktree / PILOT / "FREEZE.json",
        "PRELABEL": worktree / PILOT / "PRELABEL_VERIFICATION.json",
        "SCORE": worktree / PILOT / "SCORE_METRICS.json",
        "ELIGIBILITY": worktree / PILOT / "ELIGIBILITY.json",
        "PREFIT": worktree / PILOT / "PREFIT_COMPLETE.json",
        "FINDINGS": worktree / PILOT / "FINDINGS.md",
        "LEDGER": worktree / RUN / "fit_ledger.jsonl",
    }
    for fit in FITS:
        paths[f"{fit}.model"] = worktree / RUN / f"{fit}.joblib"
        paths[f"{fit}.prediction"] = worktree / RUN / f"{fit}.npz"
        paths[f"{fit}.receipt"] = worktree / RUN / f"{fit}.receipt.json"
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise DiagnosticError(f"Missing frozen pilot artifact: {missing[:5]}")
    return paths


def _hash_map(paths: dict[str, Path]) -> dict[str, str]:
    return {key: _sha(path) for key, path in sorted(paths.items())}


def _verify_ledger(worktree: Path, freeze: dict[str, Any], expected_sha: str) -> str:
    path = worktree / RUN / "fit_ledger.jsonl"
    ledger_sha = _sha(path)
    if ledger_sha != expected_sha:
        raise DiagnosticError("The six-fit pilot ledger SHA-256 changed")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    reserve = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    if (len(reserve) != 1 or reserve[0].get("cap") != 6
            or reserve[0].get("fit_ids") != list(FITS)
            or reserve[0].get("freeze_signature") != freeze["freeze_signature"]
            or [event.get("fit_id") for event in starts] != list(FITS)
            or [event.get("fit_id") for event in saved] != list(FITS)
            or failed or len(events) != 13
            or any(e.get("shared_main_ledger_written") is not False for e in reserve + starts + saved)):
        raise DiagnosticError("Pilot ledger no longer reconciles to exactly six saved starts")
    return ledger_sha


def _read_rows(train_path: Path, wanted: dict[int, str]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    required = ("record_id", "cycle_code", "signal_stream", "threat_rank")
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(col not in reader.fieldnames for col in required):
            raise DiagnosticError("Regression Train does not match the frozen schema")
        for i, row in enumerate(reader):
            if i not in wanted:
                continue
            if row["record_id"] != wanted[i]:
                raise DiagnosticError(f"Train row and official record_id disagree at row {i}")
            cycle, y = int(float(row["cycle_code"])), float(row["threat_rank"])
            if not math.isfinite(y) or y < 0 or y > 100:
                raise DiagnosticError(f"Selected Train outcome outside [0,100] at row {i}")
            result[i] = {"record_id": row["record_id"], "cycle": cycle, "target": y,
                         "present": bool((row["signal_stream"] or "").strip())}
    if set(result) != set(wanted):
        raise DiagnosticError("A selected donor/query row is missing from Regression Train")
    return result


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _load_queries(source: Path, worktree: Path, freeze: dict[str, Any],
                  receipts: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    queries = {}
    train = _file(source, TRAIN)
    for origin in ORIGINS:
        folder = "recovered_bundles" if origin == 67 else "bundles"
        rel = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1") / folder / f"RNBR_PILOT_E{origin}_OBSERVED/arrays.npz"
        bundle = _npz(_file(source, rel))
        rows = np.asarray(bundle["score_rows"], dtype=np.int64)
        ids = np.asarray(bundle["score_record_ids"], dtype=str)
        cycles = np.asarray(bundle["score_cycles"], dtype=np.int64)
        baseline = np.asarray(bundle["baseline_prediction"], dtype=np.float64)
        missing = np.asarray(bundle["query_missing"], dtype=bool)
        if (len(set(rows.tolist())) != len(rows) or ids.shape != rows.shape
                or cycles.shape != rows.shape or baseline.shape != rows.shape
                or missing.shape != rows.shape or not np.isfinite(baseline).all()):
            raise DiagnosticError(f"E{origin} original R06 query bundle is invalid")
        arms = {}
        for arm in ARMS:
            fit = f"{CARD}_E{origin}_{arm}"
            pred_path = worktree / RUN / f"{fit}.npz"
            model_path = worktree / RUN / f"{fit}.joblib"
            receipt = receipts[fit]
            body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if (receipt.get("receipt_sha256") != _canon_sha(body)
                    or receipt.get("freeze_signature") != freeze["freeze_signature"]
                    or receipt.get("predictions_sha256") != _sha(pred_path)
                    or receipt.get("model_sha256") != _sha(model_path)
                    or receipt.get("query_targets_read_for_fit") is not False
                    or receipt.get("test_or_sample_accessed") is not False
                    or receipt.get("classification_labels_or_probabilities_accessed") is not False
                    or receipt.get("shared_main_ledger_written") is not False):
                raise DiagnosticError(f"Saved receipt/model/prediction identity fails for {fit}")
            arrays = _npz(pred_path)
            if (str(arrays["fit_id"][0]) != fit or int(arrays["origin"][0]) != origin
                    or str(arrays["arm"][0]) != arm):
                raise DiagnosticError(f"Saved query identity fails for {fit}")
            for key, expected in (("rows", rows), ("record_ids", ids), ("cycles", cycles),
                                  ("baseline", baseline), ("missing", missing)):
                if not np.array_equal(arrays[key], expected):
                    raise DiagnosticError(f"E{origin} {arm} saved {key} differs from original R06")
            head = np.asarray(arrays["head_prediction"], dtype=np.float64)
            delta = np.asarray(arrays["delta"], dtype=np.float64)
            prediction = np.asarray(arrays["prediction"], dtype=np.float64)
            expected_delta = np.clip(0.25 * head, -5.0, 5.0)
            expected_delta[missing] = 0.0
            expected_prediction = np.clip(baseline + expected_delta, 0.0, 100.0)
            expected_prediction[missing] = baseline[missing]
            if (not np.isfinite(head).all() or not np.array_equal(delta, expected_delta)
                    or not np.array_equal(prediction, expected_prediction)):
                raise DiagnosticError(f"E{origin} {arm} saved fixed delta/prediction formula differs")
            arms[arm] = arrays
        provenance = freeze["origins"][str(origin)]
        if (len(rows) != provenance["query_rows"]
                or _array_sha(rows, "<i8") != provenance["query_rows_source_positions_sha256"]
                or _array_sha(ids) != provenance["query_row_ids_sha256"]
                or _array_sha(baseline, "<f8") != provenance["query_final_r06_prediction_sha256"]):
            raise DiagnosticError(f"E{origin} query identity differs from frozen provenance")
        selected = _read_rows(train, {int(r): str(i) for r, i in zip(rows, ids)})
        y = np.asarray([selected[int(r)]["target"] for r in rows], dtype=np.float64)
        present = np.asarray([selected[int(r)]["present"] for r in rows], dtype=bool)
        train_cycles = np.asarray([selected[int(r)]["cycle"] for r in rows], dtype=np.int64)
        if not np.array_equal(train_cycles, cycles) or not np.array_equal(~present, missing):
            raise DiagnosticError(f"E{origin} official Train cycle/availability does not align")
        queries[origin] = {
            "rows": rows, "ids": ids, "cycles": cycles, "baseline": baseline,
            "missing": missing, "present": present, "y": y, "residual": y - baseline,
            "arms": arms, "bundle_sha": _sha(_file(source, rel)),
        }
    return queries


def _load_donors(source: Path, freeze: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    donors = {}
    train = _file(source, TRAIN)
    for origin in ORIGINS:
        folder = "recovered_bundles" if origin == 67 else "bundles"
        rel = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1") / folder / f"RNBR_PILOT_E{origin}_OBSERVED/arrays.npz"
        z = _npz(_file(source, rel))
        rows = np.asarray(z["donor_rows"], dtype=np.int64)
        ids = np.asarray(z["donor_record_ids"], dtype=str)
        cycles = np.asarray(z["donor_cycles"], dtype=np.int64)
        cutoffs = np.asarray(z["donor_source_cutoffs"], dtype=np.int64)
        baseline = np.asarray(z["donor_r06_predictions"], dtype=np.float64)
        residual = np.asarray(z["donor_residuals"], dtype=np.float64)
        if (len(set(rows.tolist())) != len(rows) or len(set(ids.tolist())) != len(ids)
                or ids.shape != rows.shape or cycles.shape != rows.shape or cutoffs.shape != rows.shape
                or baseline.shape != rows.shape or residual.shape != rows.shape
                or np.any(cycles < 56) or np.any(cycles > origin) or np.any(cutoffs >= cycles)
                or not np.isfinite(baseline).all() or not np.isfinite(residual).all()):
            raise DiagnosticError(f"E{origin} donor bank row/cutoff/numeric integrity failed")
        exact_cutoff = ((cycles - 56) // 4) * 4 + 55
        if not np.array_equal(cutoffs, exact_cutoff.astype(np.int64)):
            raise DiagnosticError(f"E{origin} donor cutoff is not the exact OOF four-cycle block")
        selected = _read_rows(train, {int(r): str(i) for r, i in zip(rows, ids)})
        y = np.asarray([selected[int(r)]["target"] for r in rows], dtype=np.float64)
        train_cycles = np.asarray([selected[int(r)]["cycle"] for r in rows], dtype=np.int64)
        if not np.array_equal(train_cycles, cycles) or not np.allclose(y - baseline, residual, rtol=0, atol=1e-12):
            raise DiagnosticError(f"E{origin} selected donor targets do not reproduce OOF residuals")
        provenance = freeze["origins"][str(origin)]
        if (len(rows) != provenance["fit_rows"]
                or _array_sha(rows, "<i8") != provenance["fit_rows_source_positions_sha256"]
                or _array_sha(ids) != provenance["fit_row_ids_sha256"]
                or _array_sha(baseline, "<f8") != provenance["donor_final_r06_prediction_sha256"]
                or _array_sha(residual, "<f8") != provenance["donor_residual_sha256"]):
            raise DiagnosticError(f"E{origin} donor bank differs from frozen pilot provenance")
        donors[origin] = {"rows": rows, "ids": ids, "cycles": cycles, "cutoffs": cutoffs,
                          "baseline": baseline, "residual": residual,
                          "decile": fixed_decile(baseline)}
    return donors


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction)))


def _q(values: np.ndarray) -> tuple[float, float, float]:
    low, med, high = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9], method="linear")
    return float(low), float(med), float(high)


def _paired_w6(queries: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left, right = queries[67], queries[75]
    il = np.flatnonzero((left["cycles"] >= 76) & (left["cycles"] <= 79))
    ir = np.flatnonzero((right["cycles"] >= 76) & (right["cycles"] <= 79))
    ol, oright = pair_by_id(left["ids"][il], right["ids"][ir])
    il, ir = il[ol], ir[oright]
    if (len(il) != 8491 or not np.array_equal(left["cycles"][il], right["cycles"][ir])
            or not np.array_equal(left["y"][il], right["y"][ir])):
        raise DiagnosticError("E67 and E75 W6 are not the exact same 8,491 rows/targets/cycles")
    cycle_rows = []
    for origin, query in ((67, left), (75, right)):
        for cycle in range(76, 80):
            mask = query["cycles"] == cycle
            base_mae = _mae(query["y"][mask], query["baseline"][mask])
            for arm in ARMS:
                arrays = query["arms"][arm]
                hq = _q(arrays["head_prediction"][mask])
                dq = _q(arrays["delta"][mask])
                final_mae = _mae(query["y"][mask], arrays["prediction"][mask])
                cycle_rows.append({
                    "origin": origin, "cycle": cycle, "arm": arm, "n": int(mask.sum()),
                    "r06_mae": base_mae, "candidate_mae": final_mae,
                    "gain_vs_r06": base_mae - final_mae,
                    "query_residual_mean": float(np.mean(query["residual"][mask])),
                    "query_residual_median": float(np.median(query["residual"][mask])),
                    "raw_head_mean": float(np.mean(arrays["head_prediction"][mask])),
                    "raw_head_abs_mean": float(np.mean(np.abs(arrays["head_prediction"][mask]))),
                    "raw_head_p10": hq[0], "raw_head_median": hq[1], "raw_head_p90": hq[2],
                    "delta_mean": float(np.mean(arrays["delta"][mask])),
                    "delta_abs_mean": float(np.mean(np.abs(arrays["delta"][mask]))),
                    "delta_p10": dq[0], "delta_median": dq[1], "delta_p90": dq[2],
                })
    bdiff = right["baseline"][ir] - left["baseline"][il]
    bq = _q(bdiff)
    baseline_stats = {
        "n": int(len(il)),
        "r06_mae_E67": _mae(left["y"][il], left["baseline"][il]),
        "r06_mae_E75": _mae(right["y"][ir], right["baseline"][ir]),
        "mean_signed_baseline_change_E75_minus_E67": float(np.mean(bdiff)),
        "mean_abs_baseline_change": float(np.mean(np.abs(bdiff))),
        "baseline_change_p10": bq[0], "baseline_change_median": bq[1], "baseline_change_p90": bq[2],
    }
    changes = []
    for arm in ARMS:
        a, b = left["arms"][arm], right["arms"][arm]
        h0, h1 = a["head_prediction"][il], b["head_prediction"][ir]
        d0, d1 = a["delta"][il], b["delta"][ir]
        p0, p1 = a["prediction"][il], b["prediction"][ir]
        hd, dd, pdiff = h1 - h0, d1 - d0, p1 - p0
        hnz = (h0 != 0) & (h1 != 0)
        dnz = (d0 != 0) & (d1 != 0)
        hq, dq, pq = _q(hd), _q(dd), _q(pdiff)
        changes.append({
            "arm": arm, "n": int(len(il)),
            "head_sign_agreement": float(np.mean(np.sign(h0[hnz]) == np.sign(h1[hnz]))),
            "head_sign_compared_n": int(hnz.sum()),
            "delta_sign_agreement": float(np.mean(np.sign(d0[dnz]) == np.sign(d1[dnz]))),
            "delta_sign_compared_n": int(dnz.sum()),
            "mean_head_E67": float(np.mean(h0)), "mean_head_E75": float(np.mean(h1)),
            "mean_abs_head_E67": float(np.mean(np.abs(h0))), "mean_abs_head_E75": float(np.mean(np.abs(h1))),
            "mean_signed_head_change": float(np.mean(hd)), "mean_abs_head_change": float(np.mean(np.abs(hd))),
            "head_change_p10": hq[0], "head_change_median": hq[1], "head_change_p90": hq[2],
            "mean_delta_E67": float(np.mean(d0)), "mean_delta_E75": float(np.mean(d1)),
            "mean_abs_delta_E67": float(np.mean(np.abs(d0))), "mean_abs_delta_E75": float(np.mean(np.abs(d1))),
            "mean_signed_delta_change": float(np.mean(dd)), "mean_abs_delta_change": float(np.mean(np.abs(dd))),
            "delta_change_p10": dq[0], "delta_change_median": dq[1], "delta_change_p90": dq[2],
            "mean_abs_final_prediction_change": float(np.mean(np.abs(pdiff))),
            "prediction_change_p10": pq[0], "prediction_change_median": pq[1], "prediction_change_p90": pq[2],
            "candidate_mae_E67": _mae(left["y"][il], p0),
            "candidate_mae_E75": _mae(right["y"][ir], p1),
        })
    return cycle_rows, changes, baseline_stats


def _residual_rows(donors: dict[int, dict[str, np.ndarray]],
                   queries: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    def add(origin: int, sample: str, view: str, cutoff: int | str,
            cycle: int | str, decile: int, values: np.ndarray) -> None:
        rows.append({"origin": origin, "sample": sample, "view": view,
                     "source_cutoff": cutoff, "cycle": cycle,
                     "r06_prediction_decile": decile, **distribution(values)})
    for origin in ORIGINS:
        d = donors[origin]
        for decile in range(10):
            add(origin, "donor", "DONOR_BANK", "ALL", "ALL", decile,
                d["residual"][d["decile"] == decile])
        for cutoff in sorted(set(d["cutoffs"].tolist())):
            for cycle in sorted(set(d["cycles"][d["cutoffs"] == cutoff].tolist())):
                for decile in range(10):
                    mask = ((d["cutoffs"] == cutoff) & (d["cycles"] == cycle)
                            & (d["decile"] == decile))
                    add(origin, "donor", "DONOR_CUTOFF_CYCLE", int(cutoff), int(cycle),
                        decile, d["residual"][mask])
        q = queries[origin]
        views = (("E67_W4", 68, 71), ("E67_W5", 72, 75), ("E67_W6", 76, 79)) if origin == 67 else (("E75_W6", 76, 79),)
        decile = fixed_decile(q["baseline"])
        for view, low, high in views:
            win = (q["cycles"] >= low) & (q["cycles"] <= high)
            for dec in range(10):
                mask = win & (decile == dec)
                add(origin, "query", view, "", "ALL", dec, q["residual"][mask])
            for cycle in sorted(set(q["cycles"][win].tolist())):
                for dec in range(10):
                    mask = win & (q["cycles"] == cycle) & (decile == dec)
                    add(origin, "query", view, "", int(cycle), dec, q["residual"][mask])
    return rows


def _importance_rows(worktree: Path, freeze: dict[str, Any],
                     receipts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for origin in ORIGINS:
        meta = freeze["origins"][str(origin)]["input_reconstruction"]
        unigram = int(meta["unigram_features"])
        for arm in ARMS:
            fit = f"{CARD}_E{origin}_{arm}"
            payload = joblib.load(worktree / RUN / f"{fit}.joblib")
            model = payload["model"]
            pos = 0 if arm == "BAG" else int(meta["position_features"][arm])
            n_features = unigram + pos + 1
            if (payload.get("fit_id") != fit or payload.get("freeze_signature") != freeze["freeze_signature"]
                    or payload.get("origin") != origin or payload.get("arm") != arm
                    or payload.get("parameters_sha256") != receipts[fit].get("parameters_sha256")
                    or int(model.n_features_in_) != n_features):
                raise DiagnosticError(f"{fit} model feature boundaries differ from frozen metadata")
            parameters = model.get_params(deep=False)
            if (parameters.get("objective") != "regression_l1" or parameters.get("n_estimators") != 400
                    or parameters.get("num_leaves") != 15 or parameters.get("min_child_samples") != 80
                    or parameters.get("reg_lambda") != 3.0 or parameters.get("n_jobs") != 2):
                raise DiagnosticError(f"{fit} saved model parameters differ from the frozen pilot recipe")
            split = np.asarray(model.booster_.feature_importance(importance_type="split"), dtype=np.float64)
            gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
            if split.shape != (n_features,) or gain.shape != split.shape or not np.isfinite(gain).all():
                raise DiagnosticError(f"{fit} importance vector is invalid")
            for block, start, stop in (
                ("BAG_UNIGRAM", 0, unigram),
                ("POSITION_THIRD", unigram, unigram + pos),
                ("BASELINE_SCORE", n_features - 1, n_features),
            ):
                split_sum, gain_sum = float(split[start:stop].sum()), float(gain[start:stop].sum())
                rows.append({
                    "origin": origin, "arm": arm, "block": block,
                    "feature_start_inclusive": start, "feature_stop_exclusive": stop,
                    "feature_count": stop - start, "n_features_in": n_features,
                    "split_importance": split_sum, "gain_importance": gain_sum,
                    "split_share": split_sum / float(split.sum()) if split.sum() else None,
                    "gain_share": gain_sum / float(gain.sum()) if gain.sum() else None,
                    "saved_in_sample_residual_mae_descriptive_only":
                        receipts[fit].get("in_sample_training_mae_descriptive_only"),
                    "feature_boundary_verified": True,
                })
    return rows


def _delta_rows(queries: dict[int, dict[str, Any]], receipts: dict[str, dict[str, Any]],
                score: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    effects, losses = [], []
    for origin in ORIGINS:
        q = queries[origin]
        view = "E67_H12" if origin == 67 else "E75_W6"
        base_abs = np.abs(q["y"] - q["baseline"])
        for arm in ARMS:
            fit = f"{CARD}_E{origin}_{arm}"
            arrays = q["arms"][arm]
            head = arrays["head_prediction"]
            delta = arrays["delta"]
            prediction = arrays["prediction"]
            present, missing = q["present"], q["missing"]
            residual = q["residual"]
            changed = base_abs - np.abs(q["y"] - prediction)
            raw = 0.25 * head
            hq, dq = _q(head[present]), _q(delta[present])
            sign_ok = (head[present] != 0) & (residual[present] != 0)
            metrics = score["views"][view]["metrics"][arm]
            if abs(_mae(q["y"], prediction) - metrics["mae"]) > 1e-10:
                raise DiagnosticError(f"Existing query loss does not match committed score for {fit}")
            training_mae = receipts[fit].get("in_sample_training_mae_descriptive_only")
            query_resid_mae = float(np.mean(np.abs(residual[present])))
            effect = {
                "origin": origin, "arm": arm, "n_total": len(present), "n_present": int(present.sum()),
                "n_missing": int(missing.sum()), "mean_query_residual_present": float(np.mean(residual[present])),
                "median_query_residual_present": float(np.median(residual[present])),
                "mean_raw_head_present": float(np.mean(head[present])),
                "mean_abs_raw_head_present": float(np.mean(np.abs(head[present]))),
                "raw_head_p10": hq[0], "raw_head_median": hq[1], "raw_head_p90": hq[2],
                "mean_signed_delta_present": float(np.mean(delta[present])),
                "mean_abs_delta_present": float(np.mean(np.abs(delta[present]))),
                "delta_p10": dq[0], "delta_median": dq[1], "delta_p90": dq[2],
                "alpha_head_abs_over_5_n": int(np.sum(np.abs(raw[present]) > 5)),
                "alpha_head_abs_over_5_fraction": float(np.mean(np.abs(raw[present]) > 5)),
                "final_lower_clip_n": int(np.sum((q["baseline"] + delta)[present] < 0)),
                "final_upper_clip_n": int(np.sum((q["baseline"] + delta)[present] > 100)),
                "head_residual_sign_agreement_nonzero":
                    float(np.mean(np.sign(head[present][sign_ok]) == np.sign(residual[present][sign_ok])))
                    if sign_ok.any() else None,
                "head_residual_sign_compared_n": int(sign_ok.sum()),
                "helped_present_n": int(np.sum(present & (changed > 1e-12))),
                "harmed_present_n": int(np.sum(present & (changed < -1e-12))),
                "tie_present_n": int(np.sum(present & (np.abs(changed) <= 1e-12))),
                "missing_exact_n": int(np.sum(missing)),
                "mae_change_all": float(np.mean(np.abs(q["y"] - prediction) - base_abs)),
                "mae_change_present": float(np.mean(np.abs(q["y"][present] - prediction[present]) - base_abs[present])),
                "saved_baseline_mae": score["views"][view]["metrics"]["R06"]["mae"],
                "saved_candidate_mae": metrics["mae"],
                "saved_fit_residual_mae_descriptive_only": training_mae,
                "fit_minus_query_residual_mae_descriptive_only":
                    training_mae - query_resid_mae if training_mae is not None else None,
            }
            effects.append(effect)
            losses.append({
                "origin": origin, "arm": arm, "fit_id": fit,
                "saved_fit_residual_mae_descriptive_only": training_mae,
                "existing_query_mae": metrics["mae"],
                "existing_query_gain_vs_r06": metrics["gain_vs_R06"],
                "query_residual_mae_present": query_resid_mae,
                "fit_minus_query_residual_mae_descriptive_only":
                    training_mae - query_resid_mae if training_mae is not None else None,
            })
    return effects, losses


def _paired_deciles(residual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["origin"], r["sample"], r["view"], r["r06_prediction_decile"]): r
              for r in residual_rows if r["cycle"] == "ALL"}
    rows = []
    for origin, views in ((67, ("E67_W4", "E67_W5", "E67_W6")), (75, ("E75_W6",))):
        for view in views:
            for decile in range(10):
                donor = lookup[(origin, "donor", "DONOR_BANK", decile)]
                query = lookup[(origin, "query", view, decile)]
                ok = donor["status"] == query["status"] == "SUPPORTED"
                rows.append({
                    "origin": origin, "query_view": view, "r06_prediction_decile": decile,
                    "donor_n": donor["n"], "query_n": query["n"],
                    "status": "SUPPORTED" if ok else "UNKNOWN_UNSUPPORTED",
                    "donor_residual_median": donor["median"] if ok else None,
                    "query_residual_median": query["median"] if ok else None,
                    "query_minus_donor_median": query["median"] - donor["median"] if ok else None,
                    "donor_p10": donor["p10"] if ok else None, "donor_p90": donor["p90"] if ok else None,
                    "query_p10": query["p10"] if ok else None, "query_p90": query["p90"] if ok else None,
                })
    return rows


def _plots(out: Path, cycles: list[dict[str, Any]],
           paired_deciles: list[dict[str, Any]]) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hashes = {}
    colors = {"BAG": "#2563eb", "REAL": "#16a34a", "SHUFFLED": "#9333ea"}
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    real_rows = [r for r in cycles if r["arm"] == "REAL"]
    x = np.arange(len(real_rows), dtype=float)
    width = 0.24
    labels = [f"E{r['origin']} C{r['cycle']}" for r in real_rows]
    for j, arm in enumerate(ARMS):
        selected = [r for r in cycles if r["arm"] == arm]
        ax.bar(x + (j - 1) * width, [r["gain_vs_r06"] for r in selected],
               width, label=arm, color=colors[arm])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Saved prediction MAE gain vs R06")
    ax.set_title("Same W6 record IDs, paired across E67 and E75")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170, metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(fig)
    p = out / "PAIRED_W6_CYCLE_GAINS.png"
    _write_once(p, buf.getvalue())
    hashes[p.name] = _sha(p)

    names = (("E67_W4", 67), ("E67_W5", 67), ("E67_W6", 67), ("E75_W6", 75))
    matrix = np.full((len(names), 10), np.nan)
    for row, (view, origin) in enumerate(names):
        for item in paired_deciles:
            if item["query_view"] == view and item["origin"] == origin and item["status"] == "SUPPORTED":
                matrix[row, item["r06_prediction_decile"]] = item["query_minus_donor_median"]
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="RdBu_r", vmin=-10, vmax=10)
    ax.set_xticks(np.arange(10), [str(i) for i in range(10)])
    ax.set_yticks(np.arange(len(names)), [f"E{o} {v}" for v, o in names])
    ax.set_xlabel("Fixed R06 decile: floor(clip(R06,0,100)/10), capped at 9")
    ax.set_title("Query minus donor median residual in fixed score deciles")
    for row, (view, origin) in enumerate(names):
        for item in paired_deciles:
            if item["query_view"] == view and item["origin"] == origin:
                col = item["r06_prediction_decile"]
                label = (f"{item['query_minus_donor_median']:+.1f}\nq{item['query_n']}/d{item['donor_n']}"
                         if item["status"] == "SUPPORTED" else "UNK")
                ax.text(col, row, label, ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Median residual shift: query − donor")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170, metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(fig)
    p = out / "RESIDUAL_TRANSPORT_FIXED_DECILES.png"
    _write_once(p, buf.getvalue())
    hashes[p.name] = _sha(p)
    return hashes


def run(source_root: Path, worktree_root: Path, output_root: Path) -> dict[str, Any]:
    source, worktree, out = source_root.resolve(), worktree_root.resolve(), output_root.resolve()
    if source == worktree or source in worktree.parents or worktree in source.parents:
        raise DiagnosticError("Source root and REG worktree must be disjoint")
    if source == out or source in out.parents or worktree not in out.parents:
        raise DiagnosticError("Output must be an explicit child of the REG worktree, outside main TP2")
    freeze, frozen_before = _verify_freeze(source, worktree)
    paths = _pilot_paths(worktree)
    pilot_before = _hash_map(paths)
    prelabel, score = _json(paths["PRELABEL"]), _json(paths["SCORE"])
    prefit = _json(paths["PREFIT"])
    if (prelabel.get("status") != "PASS" or prelabel.get("freeze_signature") != freeze["freeze_signature"]
            or score.get("status") != "SCORED_AFTER_PRELABEL_PASS"
            or score.get("freeze_signature") != freeze["freeze_signature"]
            or score.get("query_targets_read_after_all_six_saved") is not True
            or prefit.get("fit_starts") != 6 or prefit.get("fit_saved") != 6):
        raise DiagnosticError("Pilot prelabel/scoring/accounting evidence is incomplete")
    ledger_sha = _verify_ledger(worktree, freeze, prefit["ledger_sha256"])
    receipts = {fit: _json(paths[f"{fit}.receipt"]) for fit in FITS}
    queries = _load_queries(source, worktree, freeze, receipts)
    donors = _load_donors(source, freeze)
    importance = _importance_rows(worktree, freeze, receipts)
    effects, losses = _delta_rows(queries, receipts, score)
    cycle_rows, paired_changes, baseline_stats = _paired_w6(queries)
    residual = _residual_rows(donors, queries)
    deciles = _paired_deciles(residual)

    hash_assets = {
        "freeze_signature": freeze["freeze_signature"],
        "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"],
        "frozen_hashes_recomputed": frozen_before,
        "pilot_artifact_hashes_before": pilot_before,
        "pilot_ledger_sha256": ledger_sha,
        "pilot_audit_hashes": {key: _sha(path) for key, path in paths.items()
                               if key in ("FREEZE", "PRELABEL", "SCORE", "ELIGIBILITY", "PREFIT", "FINDINGS")},
        "diagnostic_script_sha256": _sha(Path(__file__).resolve()),
        "diagnostic_script_sha256": _sha(Path(__file__).resolve()),
        "sol_review_next_sha256": _sha(worktree / PILOT / "SOL_REVIEW_NEXT.md"),
        "main_train_sha256": _sha(_file(source, TRAIN)),
    }
    out.mkdir(parents=True, exist_ok=True)
    plots = _plots(out, cycle_rows, deciles)
    tables = {
        "paired_w6_by_cycle.csv": (cycle_rows, ["origin","cycle","arm","n","r06_mae","candidate_mae","gain_vs_r06","query_residual_mean","query_residual_median","raw_head_mean","raw_head_abs_mean","raw_head_p10","raw_head_median","raw_head_p90","delta_mean","delta_abs_mean","delta_p10","delta_median","delta_p90"]),
        "paired_w6_origin_changes.csv": (paired_changes, ["arm","n","head_sign_agreement","head_sign_compared_n","delta_sign_agreement","delta_sign_compared_n","mean_head_E67","mean_head_E75","mean_abs_head_E67","mean_abs_head_E75","mean_signed_head_change","mean_abs_head_change","head_change_p10","head_change_median","head_change_p90","mean_delta_E67","mean_delta_E75","mean_abs_delta_E67","mean_abs_delta_E75","mean_signed_delta_change","mean_abs_delta_change","delta_change_p10","delta_change_median","delta_change_p90","mean_abs_final_prediction_change","prediction_change_p10","prediction_change_median","prediction_change_p90","candidate_mae_E67","candidate_mae_E75"]),
        "residual_distribution_cells.csv": (residual, ["origin","sample","view","source_cutoff","cycle","r06_prediction_decile","n","status","median","p10","p90"]),
        "paired_decile_comparisons.csv": (deciles, ["origin","query_view","r06_prediction_decile","donor_n","query_n","status","donor_residual_median","query_residual_median","query_minus_donor_median","donor_p10","donor_p90","query_p10","query_p90"]),
        "block_importance.csv": (importance, ["origin","arm","block","feature_start_inclusive","feature_stop_exclusive","feature_count","n_features_in","split_importance","gain_importance","split_share","gain_share","saved_in_sample_residual_mae_descriptive_only","feature_boundary_verified"]),
        "delta_error_effect.csv": (effects, ["origin","arm","n_total","n_present","n_missing","mean_query_residual_present","median_query_residual_present","mean_raw_head_present","mean_abs_raw_head_present","raw_head_p10","raw_head_median","raw_head_p90","mean_signed_delta_present","mean_abs_delta_present","delta_p10","delta_median","delta_p90","alpha_head_abs_over_5_n","alpha_head_abs_over_5_fraction","final_lower_clip_n","final_upper_clip_n","head_residual_sign_agreement_nonzero","head_residual_sign_compared_n","helped_present_n","harmed_present_n","tie_present_n","missing_exact_n","mae_change_all","mae_change_present","saved_baseline_mae","saved_candidate_mae","saved_fit_residual_mae_descriptive_only","fit_minus_query_residual_mae_descriptive_only"]),
        "train_query_loss.csv": (losses, ["origin","arm","fit_id","saved_fit_residual_mae_descriptive_only","existing_query_mae","existing_query_gain_vs_r06","query_residual_mae_present","fit_minus_query_residual_mae_descriptive_only"]),
    }
    for name, (rows, columns) in tables.items():
        _write_csv(out / name, rows, columns)
    _write_json(out / "INPUT_HASHES.json", hash_assets)
    unsupported = sum(r["status"] == "UNKNOWN_UNSUPPORTED" for r in residual)
    supported_changes = {}
    for origin, view in ((67,"E67_W4"),(67,"E67_W5"),(67,"E67_W6"),(75,"E75_W6")):
        cell = [r for r in deciles if r["origin"] == origin and r["query_view"] == view and r["status"] == "SUPPORTED"]
        values = [r["query_minus_donor_median"] for r in cell]
        supported_changes[view] = {
            "supported_deciles": len(cell), "of_deciles": 10,
            "median_shift_median": float(np.median(values)) if values else None,
            "minimum_shift": float(np.min(values)) if values else None,
            "maximum_shift": float(np.max(values)) if values else None,
        }
    pilot_real = {name: score["views"][name]["metrics"]["REAL"] for name in ("E67_H12","E67_W4","E67_W5","E67_W6","E75_W6")}
    lines = [
        "# REG_POSITION_TRANSPORT_DIAGNOSTIC_V1", "",
        "Zero-fit descriptive analysis of frozen REG_POSITION_L1_V1 outputs. This report preserves the pilot's RETIRE verdict and all failed gates. It does not score a new policy or identify a causal transport effect.", "",
        f"Reproduce with: python scripts/reg_position_transport_diagnostic_v1.py --source-root \"{source}\" --worktree-root \"{worktree}\" --output-root \"{out}\"",
        f"Freeze: {freeze['freeze_signature']}. Frozen source/input files verified: {len(frozen_before)}. Pilot artifacts hashed before/after: {len(pilot_before)}. Six-fit ledger SHA-256: {ledger_sha}.", "",
        "## Paired W6 cohort", "",
        "E67 and E75 use the exact same 8,491 official record IDs, cycles 76–79, and targets, paired by record_id. The origin-specific R06 baseline differs on those same rows.", "",
        "| Quantity | Value |","|---|---:|",
        f"| E67 R06 MAE | {baseline_stats['r06_mae_E67']:.6f} |",
        f"| E75 R06 MAE | {baseline_stats['r06_mae_E75']:.6f} |",
        f"| Mean signed R06 change E75−E67 | {baseline_stats['mean_signed_baseline_change_E75_minus_E67']:+.6f} |",
        f"| Mean absolute R06 change | {baseline_stats['mean_abs_baseline_change']:.6f} |","",
        "| Arm | Head sign agreement | Mean abs head E67→E75 | Mean abs delta E67→E75 | Mean abs final-prediction change | Candidate MAE E67→E75 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in paired_changes:
        lines.append(f"| {r['arm']} | {r['head_sign_agreement']:.3f} | {r['mean_abs_head_E67']:.3f}→{r['mean_abs_head_E75']:.3f} | {r['mean_abs_delta_E67']:.3f}→{r['mean_abs_delta_E75']:.3f} | {r['mean_abs_final_prediction_change']:.3f} | {r['candidate_mae_E67']:.6f}→{r['candidate_mae_E75']:.6f} |")
    lines += [
        "", "Per-cycle R06/candidate MAE, gain, residual, raw-head, and delta distributions are in paired_w6_by_cycle.csv. Paired row-level change magnitudes and sign agreement are in paired_w6_origin_changes.csv.", "",
        "## Residual distributions", "",
        "Residual = target − origin-specific final mapped R06. Deciles are fixed as floor(clip(R06,0,100)/10), capped at 9. Donors are detailed by source cutoff and actual cycle; query cells cover E67 W4/W5/W6 and E75 W6. Every cell contains n, median, p10 and p90; cells under n=100 are UNKNOWN with quantiles withheld.", "",
        "| Query view | Supported matched deciles | Median of within-decile median shifts | Range |",
        "|---|---:|---:|---:|",
    ]
    for view, item in supported_changes.items():
        if item["supported_deciles"]:
            lines.append(f"| {view} | {item['supported_deciles']}/10 | {item['median_shift_median']:+.3f} | {item['minimum_shift']:+.3f} to {item['maximum_shift']:+.3f} |")
        else:
            lines.append(f"| {view} | 0/10 | UNKNOWN | UNKNOWN |")
    lines += [
        "", f"There are {unsupported} detailed donor/query cutoff-cycle-decile cells below n=100. They remain UNKNOWN. Marginal changes are descriptive; donor history, fit population, target horizon, and R06 baseline move jointly.", "",
        "## Saved losses and existing correction effect", "",
        "| Origin | Arm | Saved fit residual MAE* | Query MAE | Mean delta | Help/harm/tie, present rows | 0.25-head cap rate | MAE change vs R06 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in effects:
        lines.append(f"| E{r['origin']} | {r['arm']} | {r['saved_fit_residual_mae_descriptive_only']:.6f} | {r['saved_candidate_mae']:.6f} | {r['mean_signed_delta_present']:+.4f} | {r['helped_present_n']}/{r['harmed_present_n']}/{r['tie_present_n']} | {r['alpha_head_abs_over_5_fraction']:.3%} | {r['mae_change_all']:+.6f} |")
    lines += [
        "", "*Saved in-sample residual fit loss is descriptive only. Raw-head quantiles/sign agreement, final-bound clipping, missing counts, and train-query loss gaps are in delta_error_effect.csv and train_query_loss.csv. Help/harm compare each already-saved prediction to the same origin's R06 errors; no new prediction policy was evaluated.", "",
        "## LightGBM block importances", "",
        "Feature indices are verified against frozen unigram/position widths and model n_features_in_: BAG unigrams first, optional REAL/SHUFFLED position-third suffix next, normalized R06 baseline score last.", "",
        "| Origin | Arm | Block | Features | Split share | Gain share |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in importance:
        lines.append(f"| E{r['origin']} | {r['arm']} | {r['block']} | {r['feature_count']} | {float(r['split_share'] or 0):.2%} | {float(r['gain_share'] or 0):.2%} |")
    lines += [
        "", "Block importance is descriptive, not causal: correlated features and different block widths/model capacities prevent attributing gains to one block or claiming generalization.", "",
        "## Interpretation and one recommendation", "",
        "All three heads improve E75 W6, while the pilot failed E67_H12 and E67 W4 aggregate gates; actual position adds only a small E67_H12 contrast versus BAG/SHUFFLED. The paired diagnostic measures changes in baseline, head, correction and residual distributions, but cannot isolate which jointly changed history component explains the transfer pattern.", "",
        "Recommendation: ONE_SPECIFIC_NEXT_PILOT_FOR_SOL_REVIEW — use only the separately authorized, fixed REAL E79 exploratory benchmark. Keep RETIRE_REG_POSITION_L1_V1 and every pilot gate/failure unchanged. The exploratory benchmark is not a qualified model and does not authorize upload; use BAG/SHUFFLED pilot results as fixed counterevidence. Diagnostic observations do not change the benchmark recipe or selection.", "",
        "## Reproducibility assets", "",
        "INPUT_HASHES.json records freeze/source/input plus pilot model, receipt, prediction, report, and ledger hashes. Computed tables: paired_w6_by_cycle.csv, paired_w6_origin_changes.csv, residual_distribution_cells.csv, paired_decile_comparisons.csv, block_importance.csv, delta_error_effect.csv, train_query_loss.csv.", "",
        "No vectorizer/model fit, model.predict call, new candidate policy, Test/Sample/Kaggle access, dependency install, main-tree write, or pilot-ledger mutation occurred.", "",
    ]
    report = out / "TRANSPORT_DIAGNOSTIC.md"
    _write_once(report, ("\n".join(lines)).encode("utf-8"))
    frozen_after, hashes_after = _verify_freeze(source, worktree)
    pilot_after = _hash_map(paths)
    if frozen_after["freeze_signature"] != freeze["freeze_signature"] or hashes_after != frozen_before or pilot_after != pilot_before or _sha(paths["LEDGER"]) != ledger_sha:
        raise DiagnosticError("Frozen source/input or old pilot artifact changed during diagnostic")
    return {
        "status": "PASS_ZERO_FIT_TRANSPORT_DIAGNOSTIC",
        "freeze_signature": freeze["freeze_signature"],
        "pilot_ledger_sha256_before_after": [ledger_sha, _sha(paths["LEDGER"])],
        "pilot_artifact_hashes_preserved": len(pilot_before),
        "frozen_source_input_hashes_preserved": len(frozen_before),
        "new_fit_starts": 0, "new_candidate_predictions": 0,
        "paired_w6_n": baseline_stats["n"], "unsupported_residual_cells": unsupported,
        "recommendation": "ONE_SPECIFIC_NEXT_PILOT_FOR_SOL_REVIEW",
        "outputs": sorted([*tables.keys(), "INPUT_HASHES.json", "TRANSPORT_DIAGNOSTIC.md", *plots.keys()]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.source_root, args.worktree_root, args.output_root),
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
