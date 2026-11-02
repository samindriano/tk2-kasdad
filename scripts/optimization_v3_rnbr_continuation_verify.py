"""Independent verifier and Train-only evaluation for RNBR continuation v1."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_continuation as runner  # noqa: E402
from scripts import optimization_v3_rnbr_official as pilot  # noqa: E402


class VerificationError(RuntimeError):
    """Independent output, lineage, or evaluation check failed."""


def _id(value: Any) -> str:
    return f"str_:{str(value)}"


def _median(values: np.ndarray, similarities: np.ndarray, source_rows: np.ndarray) -> float:
    order = np.lexsort((source_rows.astype(np.int64), values.astype(np.float64)))
    x = values[order].astype(np.float64)
    w = np.square(similarities[order].astype(np.float64))
    grouped_x, starts = np.unique(x, return_index=True)
    grouped_w = np.add.reduceat(w, starts)
    mass = float(grouped_w.sum())
    if mass <= 0 or not np.isfinite(mass):
        raise VerificationError("Independent weighted median has invalid mass")
    acc = 0.0
    for j, (value, weight) in enumerate(zip(grouped_x, grouped_w)):
        acc += float(weight)
        if acc > mass / 2 + 1e-12 * mass:
            return float(value)
        if abs(acc - mass / 2) <= 1e-12 * mass:
            return float((value + grouped_x[j + 1]) / 2) if j + 1 < len(grouped_x) else float(value)
    return float(grouped_x[-1])


def _shuffle(residuals: np.ndarray, cycles: np.ndarray, predictions: np.ndarray,
             rows: np.ndarray, seed: int = 42) -> np.ndarray:
    bins = np.minimum(9, np.floor(np.clip(predictions, 0, 100) / 10)).astype(np.int8)
    out = residuals.copy()
    rng = np.random.default_rng(seed)
    for cycle in np.unique(cycles):
        for decile in range(10):
            ix = np.flatnonzero((cycles == cycle) & (bins == decile))
            if len(ix) > 1:
                ix = ix[np.argsort(rows[ix], kind="stable")]
                out[ix] = residuals[ix][rng.permutation(len(ix))]
    return out


def _target_rows(rows: np.ndarray, train: Mapping[str, np.ndarray], purpose: str) -> dict[str, np.ndarray]:
    requested = np.asarray(rows, dtype=np.int64)
    if len(requested) == 0 or len(np.unique(requested)) != len(requested):
        raise VerificationError("Verifier target allowlist must be unique/nonempty")
    allowed = set(map(int, requested))
    frame = pd.read_csv(
        runner.TRAIN, usecols=["record_id", "cycle_code", "threat_rank"],
        skiprows=lambda line: line > 0 and (line - 1) not in allowed)
    if len(frame) != len(requested):
        raise VerificationError(f"Target allowlist returned {len(frame)} rows; expected {len(requested)}")
    result = {"rows": requested,
              "ids": frame["record_id"].astype(str).to_numpy(),
              "cycles": pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64),
              "target": pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64),
              "purpose": purpose}
    if (not np.array_equal(result["ids"], train["ids"][requested])
            or not np.array_equal(result["cycles"], train["cycles"][requested])
            or not np.isfinite(result["target"]).all()
            or np.any((result["target"] < 0) | (result["target"] > 100))):
        raise VerificationError("Official Train target row/ID/cycle/value integrity failed")
    return result


def _cache_r06(origin: int, rows: np.ndarray, r06: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    src = r06[origin]
    lookup = {int(row): float(value) for row, value in zip(src["rows"], src["prediction"])}
    try:
        return np.asarray([lookup[int(row)] for row in rows], dtype=np.float64)
    except KeyError as exc:
        raise VerificationError(f"R06 origin-{origin} cache misses row {exc.args[0]}") from exc


def _load_geometry(origin: int, arm: str) -> dict[str, np.ndarray]:
    path = runner._geometry_paths(origin, arm)["npz"]
    with np.load(path, allow_pickle=False) as z:
        return {name: np.asarray(z[name]).copy() for name in runner.GEOMETRY_FIELDS}


def verify_fit_bundle(fit_id: str, arrays: Mapping[str, np.ndarray], receipt: Mapping[str, Any],
                      freeze: Mapping[str, Any], train: Mapping[str, np.ndarray],
                      r06: Mapping[int, Mapping[str, np.ndarray]], source_commit: str) -> dict[str, Any]:
    """Rebuild donor residuals and each prediction without calling runner builder."""
    # IDs are RNBR_CONT_E59_OBSERVED; parse without accepting alternate names.
    if not fit_id.startswith("RNBR_CONT_E") or fit_id not in runner.FIT_IDS:
        raise VerificationError(f"Fit ID is outside continuation freeze: {fit_id}")
    tail = fit_id.removeprefix("RNBR_CONT_E")
    origin = int(tail.split("_")[0]); arm = tail.split("_")[1].lower()
    ident = receipt.get("identity", {})
    if (ident.get("fit_id") != fit_id or ident.get("origin") != origin or ident.get("arm") != arm
            or ident.get("source_commit") != source_commit
            or ident.get("freeze_signature") != freeze["freeze_signature"]
            or ident.get("query_labels_read") is not False
            or ident.get("test_or_sample_submission_opened") is not False
            or ident.get("classification_labels_used") is not False
            or receipt.get("integrity_sha256") != pilot.sha256_bytes(pilot.canonical_bytes(
                {k: v for k, v in receipt.items() if k != "integrity_sha256"}))):
        raise VerificationError(f"Bundle identity/integrity failed: {fit_id}")
    if any(np.asarray(v).dtype.kind == "O" for v in arrays.values()):
        raise VerificationError("Object arrays are prohibited")

    donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin) & train["present"]).astype(np.int64)
    donor_ids = train["ids"][donor_rows]
    donor_cycles = train["cycles"][donor_rows]
    donor_y = _target_rows(donor_rows, train, f"independent donor verification {fit_id}")
    cutoffs = np.asarray([runner.EXPECTED_SOURCE_CUTOFFS[int(c)] for c in donor_cycles], dtype=np.int64)
    donor_r06 = np.empty(len(donor_rows), dtype=np.float64)
    for cutoff in np.unique(cutoffs):
        at = np.flatnonzero(cutoffs == cutoff)
        src = r06[int(cutoff)]
        lookup = {int(row): float(v) for row, v in zip(src["rows"], src["prediction"])}
        donor_r06[at] = [lookup[int(donor_rows[i])] for i in at]
    residual = donor_y["target"] - donor_r06
    assigned = residual if arm == "observed" else _shuffle(residual, donor_cycles, donor_r06, donor_rows)
    geom = _load_geometry(origin, arm)
    qrows = np.asarray(geom["score_rows"], dtype=np.int64)
    qcycles = np.asarray(geom["score_cycles"], dtype=np.int64)
    qids = np.asarray(geom["score_record_ids"]).astype(str)
    qmissing = ~train["present"][qrows]
    qbase = _cache_r06(origin, qrows, r06)
    neighbor_rows = np.asarray(geom["neighbor_source_rows"], dtype=np.int64)
    neighbor_ids = np.asarray(geom["neighbor_ids"]).astype(str)
    cosine = np.asarray(geom["neighbor_similarities"], dtype=np.float64)
    counts = np.asarray(geom["neighbor_count"], dtype=np.int64)
    donor_index = {int(row): i for i, row in enumerate(donor_rows)}
    delta = np.zeros(len(qrows), dtype=np.float64)
    used = np.zeros(len(qrows), dtype=np.int16)
    checked = 0
    for i, n0 in enumerate(counts):
        n = int(n0)
        source_rows = neighbor_rows[i, :n]
        source_ix = []
        if (np.any(neighbor_rows[i, n:] != -1) or np.any(neighbor_ids[i, n:] != "")
                or np.any(cosine[i, n:] != 0.0) or (qmissing[i] and n != 0)):
            raise VerificationError("Independent geometry padding/missing checks failed")
        if n:
            if (len(np.unique(source_rows)) != n or np.any(source_rows < 0)
                    or np.any((cosine[i, :n] <= 0) | (cosine[i, :n] > 1 + 1e-12))
                    or sorted(range(n), key=lambda j: (-cosine[i, j], source_rows[j])) != list(range(n))):
                raise VerificationError("Independent neighbor shape/order/positivity check failed")
            for j, row in enumerate(source_rows):
                pos = donor_index.get(int(row))
                if pos is None or neighbor_ids[i, j] != _id(donor_ids[pos]) or donor_cycles[pos] >= qcycles[i]:
                    raise VerificationError("Independent donor-neighbor row/typed-ID/time linkage failed")
                source_ix.append(pos)
                checked += 1
        if qmissing[i] or n < 5:
            continue
        delta[i] = float(np.clip(0.25 * _median(assigned[np.asarray(source_ix)], cosine[i, :n], source_rows), -5, 5))
        used[i] = n
    pred = np.clip(qbase + delta, 0, 100)

    expected = {
        "score_rows": qrows, "score_record_ids": qids, "score_cycles": qcycles,
        "baseline_prediction": qbase, "query_missing": qmissing, "neighbor_count": counts,
        "neighbor_source_rows": neighbor_rows, "neighbor_ids": neighbor_ids,
        "neighbor_similarities": cosine, "donor_rows": donor_rows,
        "donor_record_ids": donor_ids, "donor_cycles": donor_cycles,
        "donor_source_cutoffs": cutoffs, "donor_r06_predictions": donor_r06,
        "donor_residuals": residual, "donor_residuals_used": assigned,
        "delta": delta, "neighbor_count_used": used, "prediction": pred,
    }
    for key, value in expected.items():
        if key not in arrays or not np.array_equal(np.asarray(arrays[key]), np.asarray(value)):
            if np.asarray(arrays.get(key, [])).dtype.kind in "fc" and np.asarray(value).dtype.kind in "fc":
                if not np.allclose(arrays[key], value, rtol=0, atol=1e-12):
                    raise VerificationError(f"Independent recomputation differs for {fit_id}/{key}")
            else:
                raise VerificationError(f"Independent recomputation differs for {fit_id}/{key}")
    if not np.array_equal(pred[qmissing], qbase[qmissing]):
        raise VerificationError("Missing-stream predictions changed from exact R06 fallback")
    return {"status": "PASS", "fit_id": fit_id, "donor_rows": len(donor_rows),
            "query_rows": len(qrows), "query_cycles": [int(qcycles.min()), int(qcycles.max())],
            "independent_neighbor_values_recomputed": checked,
            "query_baseline_source": f"R06 origin cache {origin}",
            "donor_source": "R06 per-cycle OOF cache before donor cycle",
            "query_targets_read": False}


def _load_r22(freeze: Mapping[str, Any], train: Mapping[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    values = {}
    for origin_text, spec in freeze["r22_comparison_inventory"].items():
        origin = int(origin_text)
        npz = ROOT / spec["npz"]
        if pilot.sha256_file(npz) != spec["npz_sha256"]:
            raise VerificationError(f"R22 comparison input changed at {origin}")
        with np.load(npz, allow_pickle=False) as z:
            rows = np.asarray(z["score_rows"], dtype=np.int64).copy()
            ids = np.asarray(z["score_record_ids"]).astype(str, copy=True)
            cycles = np.asarray(z["score_cycles"], dtype=np.int64).copy()
            pred = np.asarray(z[spec["prediction_array"]], dtype=np.float64).copy()
        if (not np.array_equal(ids, train["ids"][rows]) or not np.array_equal(cycles, train["cycles"][rows])
                or not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 100))):
            raise VerificationError(f"R22 score row/ID/cycle/value mismatch at {origin}")
        values[origin] = {"rows": rows, "ids": ids, "cycles": cycles, "prediction": pred}
    return values


def _cache_map(cache: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    if len(np.unique(cache["rows"])) != len(cache["rows"]):
        raise VerificationError("Duplicate row in candidate prediction cache")
    lookup = {int(r): float(p) for r, p in zip(cache["rows"], cache["prediction"])}
    try:
        return np.asarray([lookup[int(r)] for r in rows], dtype=np.float64)
    except KeyError as exc:
        raise VerificationError(f"Prediction cache misses row {exc.args[0]}") from exc


def _pilot_arrays(origin: int, freeze_pilot: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit_obs = f"RNBR_PILOT_E{origin}_OBSERVED"
    fit_shuf = f"RNBR_PILOT_E{origin}_SHUFFLED"
    obs, _ = pilot.load_bundle(fit_obs, freeze_pilot)
    shuf, _ = pilot.load_bundle(fit_shuf, freeze_pilot)
    return obs, shuf


def _verify_final_ledger() -> dict[str, Any]:
    events = runner._read_events()
    snap = runner._snapshot(events)
    amendment = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                 and e.get("amendment_id") == runner.AMENDMENT_ID]
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("reservation_key") == runner.RESERVATION_KEY]
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in runner.FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in runner.FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in runner.FIT_IDS]
    expected_caps = {"pilot": 38, "continuation": 37, "control_replay": 0, "technical_retry": 1}
    if (len(amendment) != 1 or len(reservation) != 1 or reservation[0].get("fit_ids") != list(runner.FIT_IDS)
            or snap["global_cap"] != 76 or snap["caps"] != expected_caps
            or snap["active_reserved"] != 76 or snap["started"] != 76 or snap["saved"] != 73
            or snap["failed"] != 3 or snap["pending"] != 0
            or len(starts) != 6 or len(saved) != 6 or failed
            or {e.get("fit_id") for e in starts} != set(runner.FIT_IDS)
            or {e.get("fit_id") for e in saved} != set(runner.FIT_IDS)):
        raise VerificationError(f"Final six-fit ledger reconciliation failed: {snap}")
    for event in saved:
        rec_path = runner.RUN_ROOT / "bundles" / event["fit_id"] / "receipt.json"
        if pilot.sha256_file(rec_path) != event.get("receipt_sha256"):
            raise VerificationError(f"Budget receipt hash differs from durable fit bundle: {event['fit_id']}")
    return {"status": "PASS", "snapshot": snap, "amendment_id": runner.AMENDMENT_ID,
            "reservation_key": runner.RESERVATION_KEY, "new_fit_ids": list(runner.FIT_IDS),
            "new_starts": len(starts), "new_saved": len(saved), "new_failed": len(failed)}


def _view_metrics(name: str, rows: np.ndarray, y_by_row: Mapping[int, float],
                  observed: Mapping[int, float], shuffled: Mapping[int, float],
                  r06: Mapping[int, float], r22: Mapping[int, float],
                  missing_by_row: Mapping[int, bool]) -> dict[str, Any]:
    y = np.asarray([y_by_row[int(r)] for r in rows], dtype=np.float64)
    obs = np.asarray([observed[int(r)] for r in rows], dtype=np.float64)
    shuf = np.asarray([shuffled[int(r)] for r in rows], dtype=np.float64)
    base = np.asarray([r06[int(r)] for r in rows], dtype=np.float64)
    comp = np.asarray([r22[int(r)] for r in rows], dtype=np.float64)
    mae = lambda p: float(np.mean(np.abs(y - p)))
    return {"rows": int(len(rows)), "row_positions_sha256": pilot.sha256_array(rows, "<i8"),
            "mae": {"R06": mae(base), "R_NBR_observed": mae(obs), "R_NBR_shuffled": mae(shuf), "R22_ORIGINAL": mae(comp)},
            "gain_vs_R06": mae(base) - mae(obs),
            "gain_vs_matched_shuffle": mae(shuf) - mae(obs),
            "gain_vs_R22_ORIGINAL": mae(comp) - mae(obs),
            "missing_rows": int(sum(bool(missing_by_row[int(r)]) for r in rows)),
            "missing_incremental_exact_zero": bool(np.array_equal(obs[[i for i, r in enumerate(rows) if missing_by_row[int(r)]]],
                                                                    base[[i for i, r in enumerate(rows) if missing_by_row[int(r)]]]))}


def _group_metrics(name: str, rows: np.ndarray, y_by_row: Mapping[int, float],
                   pred_by_row: Mapping[int, float], r06_by_row: Mapping[int, float],
                   missing_by_row: Mapping[int, bool], minimum_n: int = 100) -> dict[str, Any]:
    result = {}
    for group, members in (("missing_stream", lambda r: bool(missing_by_row[int(r)])),
                           ("target_le_5", lambda r: float(y_by_row[int(r)]) <= 5),
                           ("target_ge_95", lambda r: float(y_by_row[int(r)]) >= 95)):
        group_rows = np.asarray([r for r in rows if members(int(r))], dtype=np.int64)
        n = len(group_rows)
        if n == 0:
            result[group] = {"n": 0, "supported": False, "harm_vs_R06": None, "gate": "NOT_EVALUATED"}
            continue
        y = np.asarray([y_by_row[int(r)] for r in group_rows], dtype=np.float64)
        p = np.asarray([pred_by_row[int(r)] for r in group_rows], dtype=np.float64)
        b = np.asarray([r06_by_row[int(r)] for r in group_rows], dtype=np.float64)
        harm = float(np.mean(np.abs(y - p)) - np.mean(np.abs(y - b)))
        supported = n >= minimum_n
        unchanged = bool(np.array_equal(p, b)) if group == "missing_stream" else None
        result[group] = {"n": n, "supported": supported, "harm_vs_R06": harm,
                         "gate": (harm <= 0.25 if supported else None),
                         "missing_incremental_exact_zero": unchanged}
    return result


def evaluate_all(freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    """Verify every new bundle first; only then access authorized Train query targets."""
    train = runner.targetfree_train()
    r06_cache = runner.load_r06(freeze, train)
    verification = {}
    new = {}
    for origin in runner.ORIGINS:
        for arm in runner.ARMS:
            fit_id = f"RNBR_CONT_E{origin}_{arm.upper()}"
            arrays, receipt = runner.load_bundle(fit_id, freeze, source_commit=source_commit)
            state_path = runner.RUN_ROOT / "bundles" / fit_id / "fit_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (state.get("fit_id") != fit_id or state.get("freeze_signature") != freeze["freeze_signature"]
                    or state.get("state") != "DURABLE_AND_INDEPENDENTLY_VERIFIED"
                    or state.get("verification", {}).get("status") != "PASS"):
                raise VerificationError(f"Fit-state persistence/resume receipt is not independently verified: {fit_id}")
            verification[fit_id] = verify_fit_bundle(fit_id, arrays, receipt, freeze, train, r06_cache, source_commit)
            new[(origin, arm)] = arrays
    if len(verification) != 6 or any(x.get("status") != "PASS" for x in verification.values()):
        raise VerificationError("All six durable independent bundle checks must pass before query target access")

    # Inherited pilot artifacts are immutable and independently verified by the prior report/freeze.
    prior_verification = json.loads((ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json").read_text(encoding="utf-8"))
    pilot_freeze = json.loads(runner.PILOT_FREEZE.read_text(encoding="utf-8"))
    if (prior_verification.get("status") != "PASS"
            or prior_verification.get("freeze_signature") != pilot_freeze.get("freeze_signature")):
        raise VerificationError("Inherited E67/E75 pilot verification is not PASS")
    p67o, p67s = _pilot_arrays(67, pilot_freeze)
    p75o, p75s = _pilot_arrays(75, pilot_freeze)
    ledger_reconciliation = _verify_final_ledger()

    # All relevant fit bundles now exist and pass; query-target access starts here.
    eval_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= 79)).astype(np.int64)
    truth = _target_rows(eval_rows, train, "authorized post-fit internal continuation evaluation")
    y_by = {int(r): float(y) for r, y in zip(eval_rows, truth["target"])}
    missing_by = {int(r): bool(not train["present"][r]) for r in eval_rows}

    def cache_maps(cache_by_origin: Mapping[int, Mapping[str, np.ndarray]], rows: np.ndarray) -> dict[int, float]:
        values = {}
        for origin, cache in cache_by_origin.items():
            values.update({int(r): float(v) for r, v in zip(cache["rows"], cache["prediction"]) if int(r) in set(map(int, rows))})
        if any(int(r) not in values for r in rows):
            raise VerificationError("Cache map did not cover all requested evaluation rows")
        return values

    observed_by: dict[int, float] = {}
    shuffled_by: dict[int, float] = {}
    r06_by: dict[int, float] = {}
    r22_by: dict[int, float] = {}
    window_rows: dict[str, np.ndarray] = {}
    origin_for = {"W1": 55, "W2": 59, "W3": 63, "W4": 67, "W5": 71, "W6": 75}
    for w, origin in origin_for.items():
        lo, hi = {"W1": (56, 59), "W2": (60, 63), "W3": (64, 67), "W4": (68, 71), "W5": (72, 75), "W6": (76, 79)}[w]
        rows = np.flatnonzero((train["cycles"] >= lo) & (train["cycles"] <= hi)).astype(np.int64)
        window_rows[w] = rows
        r06_by.update({int(r): float(v) for r, v in zip(rows, _cache_r06(origin, rows, r06_cache))})
        r22_npz = ROOT / freeze["r22_comparison_inventory"][str(origin)]["npz"]
        spec = freeze["r22_comparison_inventory"][str(origin)]
        with np.load(r22_npz, allow_pickle=False) as z:
            rrows = np.asarray(z["score_rows"], dtype=np.int64)
            rpred = np.asarray(z[spec["prediction_array"]], dtype=np.float64)
        r22_by.update({int(r): float(v) for r, v in zip(rrows, rpred) if int(r) in set(map(int, rows))})
        if w == "W1":
            observed_by.update({int(r): r06_by[int(r)] for r in rows})
            shuffled_by.update({int(r): r06_by[int(r)] for r in rows})
        elif w == "W2":
            for arm, dest in (("observed", observed_by), ("shuffled", shuffled_by)):
                a = new[(59, arm)]
                dest.update({int(r): float(v) for r, v in zip(a["score_rows"], a["prediction"])})
        elif w == "W3":
            for arm, dest in (("observed", observed_by), ("shuffled", shuffled_by)):
                a = new[(63, arm)]
                dest.update({int(r): float(v) for r, v in zip(a["score_rows"], a["prediction"])})
        elif w == "W4":
            for src, dest in ((p67o, observed_by), (p67s, shuffled_by)):
                m = (src["score_cycles"] >= lo) & (src["score_cycles"] <= hi)
                dest.update({int(r): float(v) for r, v in zip(src["score_rows"][m], src["prediction"][m])})
        elif w == "W5":
            for arm, dest in (("observed", observed_by), ("shuffled", shuffled_by)):
                a = new[(71, arm)]
                dest.update({int(r): float(v) for r, v in zip(a["score_rows"], a["prediction"])})
        elif w == "W6":
            for src, dest in ((p75o, observed_by), (p75s, shuffled_by)):
                m = (src["score_cycles"] >= lo) & (src["score_cycles"] <= hi)
                dest.update({int(r): float(v) for r, v in zip(src["score_rows"][m], src["prediction"][m])})

    views: dict[str, np.ndarray] = dict(window_rows)
    views["new_origin_aggregate"] = np.concatenate([window_rows["W2"], window_rows["W3"], window_rows["W5"]])
    views["full_primary"] = np.concatenate([window_rows[k] for k in ("W1", "W2", "W3", "W4", "W5", "W6")])
    views["late_W5_W6"] = np.concatenate([window_rows["W5"], window_rows["W6"]])
    h12_rows = np.flatnonzero((train["cycles"] >= 68) & (train["cycles"] <= 79)).astype(np.int64)
    # The inherited long horizon is exactly E67, including its already-seen W4/W5/W6 values.
    for source, dest in ((p67o, observed_by), (p67s, shuffled_by)):
        dest.update({int(r): float(v) for r, v in zip(source["score_rows"], source["prediction"])})
    views["inherited_H12_L_E67"] = h12_rows
    metrics = {}
    groups = {}
    for name, rows in views.items():
        metrics[name] = _view_metrics(name, rows, y_by, observed_by, shuffled_by, r06_by, r22_by, missing_by)
        groups[name] = _group_metrics(name, rows, y_by, observed_by, r06_by, missing_by)

    # Inherited H12-L is the unchanged E67 view, with the same origin-67 R06
    # baseline and R22 comparator on all cycles 68–79.
    h12_obs = {int(r): float(v) for r, v in zip(p67o["score_rows"], p67o["prediction"])}
    h12_shuf = {int(r): float(v) for r, v in zip(p67s["score_rows"], p67s["prediction"])}
    h12_r06 = {int(r): float(v) for r, v in zip(h12_rows, _cache_r06(67, h12_rows, r06_cache))}
    h12_r22_npz = ROOT / freeze["r22_comparison_inventory"]["67"]["npz"]
    h12_r22_spec = freeze["r22_comparison_inventory"]["67"]
    with np.load(h12_r22_npz, allow_pickle=False) as z:
        h12_r22_rows = np.asarray(z["score_rows"], dtype=np.int64)
        h12_r22_pred = np.asarray(z[h12_r22_spec["prediction_array"]], dtype=np.float64)
    h12_r22 = {int(r): float(v) for r, v in zip(h12_r22_rows, h12_r22_pred)}
    metrics["inherited_H12_L_E67"] = _view_metrics("inherited_H12_L_E67", h12_rows, y_by,
        h12_obs, h12_shuf, h12_r06, h12_r22, missing_by)
    groups["inherited_H12_L_E67"] = _group_metrics("inherited_H12_L_E67", h12_rows, y_by,
        h12_obs, h12_r06, missing_by)

    learned = ["W2", "W3", "W4", "W5", "W6"]
    gates: dict[str, Any] = {}
    primary = metrics["full_primary"]
    newagg = metrics["new_origin_aggregate"]
    late = metrics["late_W5_W6"]
    h12 = metrics["inherited_H12_L_E67"]
    gates["full_primary_gain_vs_R06_ge_0_03"] = primary["gain_vs_R06"] >= .03
    gates["full_primary_gain_vs_shuffle_ge_0_01"] = primary["gain_vs_matched_shuffle"] >= .01
    gates["new_origin_gain_vs_R06_nonnegative"] = newagg["gain_vs_R06"] >= 0
    gates["new_origin_gain_vs_shuffle_nonnegative"] = newagg["gain_vs_matched_shuffle"] >= 0
    gates["learned_primary_windows_nonnegative_at_least_3_of_5"] = sum(metrics[w]["gain_vs_R06"] >= 0 for w in learned) >= 3
    gates["late_W5_W6_gain_vs_R06_positive"] = late["gain_vs_R06"] > 0
    gates["inherited_H12_L_harm_vs_R06_le_0_05"] = -h12["gain_vs_R06"] <= .05
    gates["each_learned_window_harm_vs_R06_lt_0_25"] = all(-metrics[w]["gain_vs_R06"] < .25 for w in learned)
    subgroup_gates = {}
    for view, values in groups.items():
        subgroup_gates[view] = {}
        for group, item in values.items():
            if item["supported"]:
                subgroup_gates[view][group] = bool(item["gate"] and
                    (group != "missing_stream" or item["missing_incremental_exact_zero"]))
            else:
                subgroup_gates[view][group] = None
    gates["supported_missing_and_target_tail_group_guards"] = subgroup_gates
    gates["all_supported_subgroup_guards_pass"] = all(v is not False for d in subgroup_gates.values() for v in d.values())

    hist = json.loads(runner.R06_VERIFY.read_text(encoding="utf-8"))
    hist_results = json.loads(runner.R06_REPORT.read_text(encoding="utf-8"))
    legacy_risks = {"archived_R06_verification_status": hist.get("status"),
                    "historical_R06_legacy_guard_summary": hist_results.get("qualification"),
                    "historical_R06_W1_missing_stream": hist_results.get("groups", {}).get("views", {}).get("W1", {}).get("signal_stream_missing"),
                    "archived_R06_result_path": str(runner.R06_REPORT.relative_to(ROOT)).replace("\\", "/"),
                    "interpretation": "Historical R06-vs-legacy failures remain separate; incremental pass does not clear them."}
    return {"schema_version": 1, "status": "EVALUATED_DIAGNOSTIC_ONLY",
            "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
            "new_fit_verification": verification,
            "ledger_reconciliation": ledger_reconciliation,
            "seen_inherited_H12": {"source_fit": "E67 observed/shuffled pilot", "n": len(h12_rows),
                                   "new_origin_E71/E75_predictions_used": False,
                                   "previous_verification_status": prior_verification.get("status")},
            "view_denominators": {k: int(len(v)) for k, v in views.items()},
            "views": metrics, "subgroups_vs_R06": groups,
            "prospective_gate_checks": gates,
            "gate_pass": all(v is True for k, v in gates.items() if k != "supported_missing_and_target_tail_group_guards")
                         and gates["all_supported_subgroup_guards_pass"],
            "historical_legacy_risk": legacy_risks,
            "source_boundaries": {"target_file": "official Regression Train.csv only",
                                  "target_rows_evaluated": int(len(eval_rows)),
                                  "test_or_sample_submission_opened": False, "kaggle_accessed": False,
                                  "full_train_refit": False, "query_target_access_after_all_six_bundles_verified": True}}


def write_reports(freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    result = evaluate_all(freeze, source_commit)
    report_dir = ROOT / "reports/optimization_v3/rnbr_continuation_v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_path = report_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    verify = {"status": "PASS" if all(x["status"] == "PASS" for x in result["new_fit_verification"].values()) else "FAIL",
              "freeze_signature": result["freeze_signature"], "source_commit": source_commit,
              "all_six_independent_bundle_checks": result["new_fit_verification"],
              "query_labels_accessed_after_all_six_durable_verified": result["source_boundaries"]["query_target_access_after_all_six_bundles_verified"],
              "source_boundaries": result["source_boundaries"]}
    (report_dir / "independent_verification.json").write_text(json.dumps(verify, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# R-NBR three-origin continuation — results", "",
             "**Status:** `EVALUATED_DIAGNOSTIC_ONLY`; this tranche does not qualify a global model.", "",
             f"**Freeze:** `{result['freeze_signature']}`  ", f"**Execution commit:** `{source_commit}`", "",
             "Query targets were opened only after the six new bundles were durably saved and independently recomputed. H12-L is the inherited E67 68–79 view; it does not use E71/E75 predictions.", "",
             "## Main views", "", "| View | n | R06 MAE | R-NBR observed | Shuffled | R22_ORIGINAL | Gain vs R06 | Gain vs shuffle | Gain vs R22 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("W1", "W2", "W3", "W4", "W5", "W6", "new_origin_aggregate", "full_primary", "late_W5_W6", "inherited_H12_L_E67"):
        v = result["views"][name]; m = v["mae"]
        lines.append(f"| {name} | {v['rows']} | {m['R06']:.6f} | {m['R_NBR_observed']:.6f} | {m['R_NBR_shuffled']:.6f} | {m['R22_ORIGINAL']:.6f} | {v['gain_vs_R06']:+.6f} | {v['gain_vs_matched_shuffle']:+.6f} | {v['gain_vs_R22_ORIGINAL']:+.6f} |")
    lines += ["", "## Gate summary", ""]
    for key, value in result["prospective_gate_checks"].items():
        if key == "supported_missing_and_target_tail_group_guards":
            continue
        lines.append(f"- `{key}`: **{value}**")
    legacy_missing = result["historical_legacy_risk"].get("historical_R06_W1_missing_stream")
    if legacy_missing:
        lines += ["", "Archived R06-vs-legacy W1 missing-stream guard: "
                  f"harm **{legacy_missing.get('harm', float('nan')):+.6f} MAE** on n={legacy_missing.get('n')}; "
                  f"historical qualified={result['historical_legacy_risk'].get('historical_R06_legacy_guard_summary', {}).get('qualified')}. "
                  "This failure remains in force regardless of continuation results."]
    lines += ["", "The line item `supported_missing_and_target_tail_group_guards` in the JSON reports every supported and unsupported subgroup for every declared view. Unsupported cells are shown explicitly and are not turned into a pass claim.", "",
              "## Interpretation", "",
              "The origin comparison is descriptive because there are only three new origins and their windows are reused development cycles. All losses are reported against both exact R06 and the matched shuffle. R22_ORIGINAL uses only verified frozen prediction caches. Historical R06-versus-legacy subgroup guard failures remain unchanged and separate from this incremental check.", "",
              "## Reproduction", "",
              "Run the frozen synthetic and unit checks before official execution. Official fit bundles are under the ignored `experiments/runs/optimization_v3_research_next_v1/RNBR_CONTINUATION_OFFICIAL_V1/bundles/` directory; machine-readable verification and view metrics are in this report directory.", ""]
    (report_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] != "--evaluate":
        print("Usage: python scripts/optimization_v3_rnbr_continuation_verify.py --evaluate", file=sys.stderr)
        return 2
    try:
        freeze = runner.load_freeze()
        commit = pilot._run_git("rev-parse", "HEAD")
        result = write_reports(freeze, commit)
        print(json.dumps({"status": result["status"], "gate_pass": result["gate_pass"],
                          "results_path": "reports/optimization_v3/rnbr_continuation_v1/RESULTS.md",
                          "views": result["views"], "gates": result["prospective_gate_checks"]}, indent=2))
        return 0
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
