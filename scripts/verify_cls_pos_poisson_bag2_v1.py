"""Independent readback verifier for CLS_POS_POISSON_BAG2_V1 artifacts."""
from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path
from typing import Any

import numpy as np


FIT_RECEIPT_KEYS = {
    "schema_version", "fit_id", "origin", "replica_seed", "fit_rows",
    "position_features", "attempt_number", "retry_allowed", "synthetic_only",
    "fit_row_positions_sha256", "fit_record_ids_sha256", "fit_target_sha256",
    "input_freeze_signature", "matrix_manifest_sha256", "fit_position_sha256",
    "offset_fit_sha256", "gamma_original_sha256", "poisson_raw_counts_sha256",
    "poisson_weights_sha256", "poisson_raw_sum", "poisson_zero_count",
    "gamma_sha256", "artifact_path", "artifact_sha256", "optimizer",
    "no_test_or_sample_access", "no_query_labels_in_fit",
}
PREDICTION_KEYS = {
    "origin", "row_positions", "record_ids", "cycles", "present_mask",
    "text_load_group", "entry_friction", "p_a02", "p_pos_original",
    "p_pos_seed17", "p_pos_seed137", "p_pos_bag2", "p_champion",
    "p_candidate", "label_champion", "label_candidate",
    "matrix_manifest_sha256", "input_freeze_signature",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": list(array.shape), "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def verify_fit_artifact(run_root: Path, fit_id: str, freeze_signature: str) -> dict[str, Any]:
    artifact = run_root / "units" / f"{fit_id}.npz"
    receipt_path = run_root / "receipts" / f"{fit_id}.json"
    require(artifact.is_file() and receipt_path.is_file(), f"Fit artifact/receipt missing: {fit_id}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(set(receipt) == FIT_RECEIPT_KEYS, f"Fit receipt schema drift: {fit_id}")
    require(receipt["fit_id"] == fit_id and receipt["input_freeze_signature"] == freeze_signature,
            f"Fit receipt identity/freeze mismatch: {fit_id}")
    expected = {
        "CLS_POS_POISSON_BAG2_V1_E67_P17": (67, 17),
        "CLS_POS_POISSON_BAG2_V1_E67_P137": (67, 137),
        "CLS_POS_POISSON_BAG2_V1_E75_P17": (75, 17),
        "CLS_POS_POISSON_BAG2_V1_E75_P137": (75, 137),
    }
    require(fit_id in expected and (receipt["origin"], receipt["replica_seed"]) == expected[fit_id],
            f"Fit origin/seed differs from fixed ID: {fit_id}")
    require(receipt["attempt_number"] == 1 and receipt["retry_allowed"] is False,
            f"A fit retry is not permitted: {fit_id}")
    require(isinstance(receipt["fit_rows"], int) and receipt["fit_rows"] > 0
            and isinstance(receipt["position_features"], int) and receipt["position_features"] > 0,
            f"Fit dimensions are invalid: {fit_id}")
    if not receipt["synthetic_only"]:
        expected_rows = 29_705 if receipt["origin"] == 67 else 42_117
        require(receipt["fit_rows"] == expected_rows, f"Official fit row count mismatch: {fit_id}")
    optimizer = receipt["optimizer"]
    require(optimizer.get("optimizer") == "scipy.optimize.minimize:L-BFGS-B"
            and optimizer.get("success") is True
            and optimizer.get("ridge_c") == 1.0
            and optimizer.get("options") == {"maxiter": 2000, "ftol": 1e-12,
                                                "gtol": 1e-6, "maxls": 50}
            and optimizer.get("poisson_lambda") == 1.0
            and optimizer.get("replica_seed") == receipt["replica_seed"]
            and optimizer.get("max_threads") == 2
            and np.isclose(optimizer.get("row_weights_sum", float("nan")), receipt["fit_rows"],
                           rtol=0.0, atol=1e-10 * max(1, receipt["fit_rows"])),
            f"Fit optimizer/weight recipe mismatch: {fit_id}")
    require(receipt["no_test_or_sample_access"] is True and receipt["no_query_labels_in_fit"] is True,
            f"Fit receipt boundary flags invalid: {fit_id}")
    require(sha256_file(artifact) == receipt["artifact_sha256"], f"Fit artifact file hash mismatch: {fit_id}")
    with np.load(artifact, allow_pickle=False) as archive:
        require(archive.files == ["gamma"], f"Unexpected fit artifact members: {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64).copy()
    require(gamma.ndim == 1 and len(gamma) == receipt["position_features"]
            and np.isfinite(gamma).all(), f"Gamma schema/nonfinite mismatch: {fit_id}")
    require(sha256_array(gamma) == receipt["gamma_sha256"], f"Gamma logical hash mismatch: {fit_id}")
    return {"fit_id": fit_id, "gamma": gamma, "receipt": receipt,
            "artifact_sha256": sha256_file(artifact), "receipt_sha256": sha256_file(receipt_path)}


def verify_prediction_bundle(path: Path, *, origin: int, freeze_signature: str,
                             matrix_manifest_sha256: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"Prediction bundle missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == PREDICTION_KEYS, "Prediction bundle schema drift")
        values = {name: archive[name].copy() for name in PREDICTION_KEYS}
    require(int(values["origin"].reshape(-1)[0]) == origin, "Prediction origin mismatch")
    require(str(values["input_freeze_signature"].reshape(-1)[0]) == freeze_signature,
            "Prediction freeze signature mismatch")
    ids = values["record_ids"].astype(str)
    n_rows = len(ids)
    require(n_rows > 0 and len(set(ids.tolist())) == n_rows, "Prediction IDs empty or duplicated")
    rows = np.asarray(values["row_positions"], dtype=np.int64)
    cycles = np.asarray(values["cycles"], dtype=np.int64)
    require(np.all(rows >= 0) and len(np.unique(rows)) == n_rows,
            "Prediction row positions are invalid or duplicated")
    require(np.all((cycles >= origin + 1) & (cycles <= 79)),
            "Prediction cycles escape the frozen temporal window")
    if matrix_manifest_sha256 is not None:
        require(str(values["matrix_manifest_sha256"].reshape(-1)[0]) == matrix_manifest_sha256,
                "Prediction matrix manifest differs from frozen input hash")
    for key in ("row_positions", "cycles", "present_mask", "text_load_group", "entry_friction",
                "p_a02", "p_pos_original", "p_pos_seed17", "p_pos_seed137", "p_pos_bag2",
                "p_champion", "p_candidate", "label_champion", "label_candidate"):
        require(len(values[key]) == n_rows, f"Prediction vector length mismatch: {key}")
    present = np.asarray(values["present_mask"], dtype=bool)
    p_a = np.asarray(values["p_a02"], dtype=np.float64)
    p0 = np.asarray(values["p_pos_original"], dtype=np.float64)
    p17 = np.asarray(values["p_pos_seed17"], dtype=np.float64)
    p137 = np.asarray(values["p_pos_seed137"], dtype=np.float64)
    pbag = np.asarray(values["p_pos_bag2"], dtype=np.float64)
    pchamp = np.asarray(values["p_champion"], dtype=np.float64)
    pcandidate = np.asarray(values["p_candidate"], dtype=np.float64)
    for key, probability in (("p_a02", p_a), ("p_pos_original", p0), ("p_pos_seed17", p17),
                             ("p_pos_seed137", p137), ("p_pos_bag2", pbag),
                             ("p_champion", pchamp), ("p_candidate", pcandidate)):
        require(np.isfinite(probability).all() and np.all((probability >= 0.0) & (probability <= 1.0)),
                f"Invalid prediction probability: {key}")
    require(np.allclose(pbag, (p0 + p17 + p137) / 3.0, rtol=0.0, atol=1e-15),
            "Three-model probability average mismatch")
    expected_champion = p_a.copy()
    expected_candidate = p_a.copy()
    expected_champion[present] = 0.5 * p0[present] + 0.5 * p_a[present]
    expected_candidate[present] = 0.5 * pbag[present] + 0.5 * p_a[present]
    require(np.array_equal(pchamp.view(np.uint64), expected_champion.view(np.uint64)),
            "Champion mixture differs from frozen policy")
    require(np.array_equal(pcandidate.view(np.uint64), expected_candidate.view(np.uint64)),
            "Bag2 mixture differs from frozen policy")
    require(np.array_equal(pchamp[~present].view(np.uint64), p_a[~present].view(np.uint64))
            and np.array_equal(pcandidate[~present].view(np.uint64), p_a[~present].view(np.uint64)),
            "Missing route not exact A02 fallback")
    missing_threshold = 0.53230976129477
    label_champion = (pchamp >= np.where(present, 0.50, missing_threshold)).astype(np.int8)
    label_candidate = (pcandidate >= np.where(present, 0.50, missing_threshold)).astype(np.int8)
    require(np.array_equal(label_champion, values["label_champion"])
            and np.array_equal(label_candidate, values["label_candidate"]),
            "Frozen threshold route mismatch")
    return {"origin": origin, "rows": n_rows, "ids_sha256": sha256_array(ids),
            "bundle_sha256": sha256_file(path), "missing_rows": int((~present).sum()),
            "missing_fallback_exact": True, "has_label_values": False}


def verify_readback_pair(run_root: Path, freeze_signature: str,
                         expected_matrix_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    outputs = {}
    for origin in (67, 75):
        path = run_root / f"E{origin}" / f"predictions_E{origin}.npz"
        outputs[f"E{origin}"] = verify_prediction_bundle(path, origin=origin,
            freeze_signature=freeze_signature,
            matrix_manifest_sha256=(expected_matrix_hashes or {}).get(f"E{origin}"))
    return {"status": "PASS_INDEPENDENT_PREDICTION_READBACK", "predictions": outputs}


def _read_selected_labels(train_path: Path, rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    row_positions = np.asarray(rows, dtype=np.int64).reshape(-1)
    record_ids = np.asarray(ids).astype(str).reshape(-1)
    require(len(row_positions) == len(record_ids) and len(np.unique(row_positions)) == len(row_positions),
            "Independent score-verifier row/ID inputs are invalid")
    positions = {int(row): index for index, row in enumerate(row_positions.tolist())}
    labels = np.empty(len(row_positions), dtype=np.int8)
    seen = np.zeros(len(row_positions), dtype=bool)
    with Path(train_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header is not None and "record_id" in header and "artifact_signal" in header,
                "Independent score-verifier Train schema invalid")
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            output_index = positions.get(row_index)
            if output_index is None:
                continue
            require(max(id_col, target_col) < len(record)
                    and record[id_col] == record_ids[output_index],
                    f"Independent score-verifier ID mismatch at row {row_index}")
            target = int(record[target_col])
            require(target in (0, 1), f"Independent score-verifier target invalid at row {row_index}")
            labels[output_index] = target
            seen[output_index] = True
    require(seen.all(), "Independent score-verifier did not find every selected Train row")
    return labels


def _metric_record(y: np.ndarray, p_champion: np.ndarray, p_candidate: np.ndarray,
                   labels_champion: np.ndarray, labels_candidate: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score, recall_score

    def metric(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
        target = np.asarray(target, dtype=np.int8).reshape(-1)
        predicted = np.asarray(predicted, dtype=np.int8).reshape(-1)
        require(len(target) == len(predicted) and np.isin(target, (0, 1)).all()
                and np.isin(predicted, (0, 1)).all(), "Independent metric vectors invalid")
        return {"n": int(len(target)), "positive": int(target.sum()),
                "negative": int(len(target) - target.sum()),
                "macro_f1": float(f1_score(target, predicted, average="macro", labels=[0, 1], zero_division=0)),
                "positive_recall": float(recall_score(target, predicted, pos_label=1, zero_division=0))}

    before = metric(y, labels_champion)
    after = metric(y, labels_candidate)
    return {"champion": before, "candidate": after,
            "gain_macro_f1": after["macro_f1"] - before["macro_f1"],
            "gain_positive_recall": after["positive_recall"] - before["positive_recall"]}


def recompute_temporal_metrics(bundles: dict[int, dict[str, np.ndarray]],
                               labels_by_row: dict[int, int], *,
                               expected_primary_rows: int | None = 20_903) -> dict[str, Any]:
    """Independently recompute all paired temporal, cycle, subgroup and gate fields."""
    scored = {}
    for origin in (67, 75):
        block = bundles[origin]
        rows = np.asarray(block["row_positions"], dtype=np.int64)
        require(all(int(row) in labels_by_row for row in rows), "Score row lacks a label in frozen Train window")
        y = np.asarray([labels_by_row[int(row)] for row in rows], dtype=np.int8)
        cycles = np.asarray(block["cycles"], dtype=np.int64)
        present = np.asarray(block["present_mask"], dtype=bool)
        champ = np.asarray(block["label_champion"], dtype=np.int8)
        candidate = np.asarray(block["label_candidate"], dtype=np.int8)
        views = ({"W4": (cycles >= 68) & (cycles <= 71),
                  "W5": (cycles >= 72) & (cycles <= 75),
                  "W6": (cycles >= 76) & (cycles <= 79),
                  "H12_E67": (cycles >= 68) & (cycles <= 79)} if origin == 67 else
                 {"W6": (cycles >= 76) & (cycles <= 79)})
        view_metrics = {}
        subgroup_metrics = {}
        for view_name, view_mask in views.items():
            view_metrics[view_name] = _metric_record(
                y[view_mask], np.asarray(block["p_champion"])[view_mask],
                np.asarray(block["p_candidate"])[view_mask], champ[view_mask], candidate[view_mask])
            group_arrays = [("stream_availability", "present", present),
                            ("stream_availability", "missing", ~present)]
            for group_name, value_key in (("fit_prefix_text_load", "text_load_group"),
                                          ("entry_friction", "entry_friction")):
                raw = np.asarray(block[value_key]).astype(str).reshape(-1)
                for value in sorted(set(raw.tolist())):
                    group_arrays.append((group_name, value, raw == value))
            subgroup_metrics[view_name] = {}
            for group, value, group_mask in group_arrays:
                mask = view_mask & group_mask
                support = {"n": int(mask.sum()), "positive": int(y[mask].sum()),
                           "negative": int(mask.sum() - y[mask].sum())}
                supported = support["n"] >= 200 and support["positive"] >= 30 and support["negative"] >= 30
                metrics = (_metric_record(y[mask], np.asarray(block["p_champion"])[mask],
                                          np.asarray(block["p_candidate"])[mask], champ[mask], candidate[mask])
                           if supported else None)
                subgroup_metrics[view_name][f"{group}={value}"] = {
                    "support": support, "supported": supported, "metrics": metrics}
        per_cycle = {}
        for cycle in sorted(np.unique(cycles).tolist()):
            mask = cycles == cycle
            per_cycle[str(int(cycle))] = _metric_record(
                y[mask], np.asarray(block["p_champion"])[mask],
                np.asarray(block["p_candidate"])[mask], champ[mask], candidate[mask])
        scored[origin] = {"origin": origin, "views": view_metrics,
                          "subgroups": subgroup_metrics, "per_cycle": per_cycle}

    e67, e75 = bundles[67], bundles[75]
    e67_rows = np.asarray(e67["row_positions"], dtype=np.int64)
    e67_cycles = np.asarray(e67["cycles"], dtype=np.int64)
    e75_rows = np.asarray(e75["row_positions"], dtype=np.int64)
    primary_mask_67 = (e67_cycles >= 68) & (e67_cycles <= 75)
    primary_rows = np.concatenate([e67_rows[primary_mask_67], e75_rows])
    primary_ids = np.concatenate([np.asarray(e67["record_ids"])[primary_mask_67],
                                  np.asarray(e75["record_ids"])]).astype(str)
    require(len(primary_rows) == len(np.unique(primary_rows))
            and len(primary_ids) == len(np.unique(primary_ids)),
            "Independent unique primary pool contains duplicate rows/IDs")
    if expected_primary_rows is not None:
        require(len(primary_rows) == len(primary_ids) == expected_primary_rows,
                f"Independent unique primary pool must contain exactly {expected_primary_rows} rows/IDs")
    primary_y = np.asarray([labels_by_row[int(row)] for row in primary_rows], dtype=np.int8)
    primary_champ = np.concatenate([np.asarray(e67["label_champion"])[primary_mask_67],
                                    np.asarray(e75["label_champion"])])
    primary_candidate = np.concatenate([np.asarray(e67["label_candidate"])[primary_mask_67],
                                        np.asarray(e75["label_candidate"])])
    p_primary_champ = np.concatenate([np.asarray(e67["p_champion"])[primary_mask_67],
                                      np.asarray(e75["p_champion"])])
    p_primary_candidate = np.concatenate([np.asarray(e67["p_candidate"])[primary_mask_67],
                                          np.asarray(e75["p_candidate"])])
    unique_pool = _metric_record(primary_y, p_primary_champ, p_primary_candidate,
                                 primary_champ, primary_candidate)
    primary_cycles = np.concatenate([e67_cycles[primary_mask_67], np.asarray(e75["cycles"])])
    per_cycle = {}
    for cycle in sorted(np.unique(primary_cycles).tolist()):
        mask = primary_cycles == cycle
        per_cycle[str(int(cycle))] = _metric_record(
            primary_y[mask], p_primary_champ[mask], p_primary_candidate[mask],
            primary_champ[mask], primary_candidate[mask])

    def subgroup_guards(metrics: dict[str, Any]) -> tuple[bool, bool, list[dict[str, Any]]]:
        legacy_q4_ok = True
        exploratory_ok = True
        failures = []
        for view, groups in metrics["subgroups"].items():
            for group, detail in groups.items():
                if not detail["supported"]:
                    continue
                f1_harm = detail["metrics"]["gain_macro_f1"]
                recall_harm = detail["metrics"]["gain_positive_recall"]
                if group == "fit_prefix_text_load=Q4" and (f1_harm < -0.01 or recall_harm < -0.02):
                    legacy_q4_ok = False
                    failures.append({"view": view, "group": group, "guard": "inherited_Q4",
                                     "macro_f1_gain": f1_harm, "positive_recall_gain": recall_harm})
                if f1_harm < -0.02 or recall_harm < -0.03:
                    exploratory_ok = False
                    failures.append({"view": view, "group": group, "guard": "exploratory_supported_group",
                                     "macro_f1_gain": f1_harm, "positive_recall_gain": recall_harm})
        return legacy_q4_ok, exploratory_ok, failures

    legacy_67, exploratory_67, failures_67 = subgroup_guards(scored[67])
    legacy_75, exploratory_75, failures_75 = subgroup_guards(scored[75])
    window_guards = {name: scored[67]["views"][name]["gain_macro_f1"] >= -0.005
                     for name in ("W4", "W5", "W6")}
    pool_gain = unique_pool["gain_macro_f1"]
    e75_w6 = scored[75]["views"]["W6"]["gain_macro_f1"]
    e67_h12 = scored[67]["views"]["H12_E67"]["gain_macro_f1"]
    return {
        "unique_primary_pool": {"rows": len(primary_rows), "cycles": [68, 79],
                                "definition": "E67 cycles 68-75 + E75 cycles 76-79",
                                "metrics": unique_pool, "per_cycle": per_cycle},
        "E67": scored[67], "E75": scored[75],
        "qualification": {
            "scientific_qualified": bool(pool_gain >= .0005 and e75_w6 >= 0.0
                and e67_h12 >= -.0005 and all(window_guards.values()) and legacy_67 and legacy_75),
            "exploratory_eligible": bool((pool_gain > 0.0 or (e75_w6 >= .0005 and pool_gain >= -.0005))
                and exploratory_67 and exploratory_75),
            "unique_pool_gain_min": .0005, "E75_W6_gain_min": 0.0,
            "E67_H12_gain_min": -.0005, "E67_W4_W5_W6_harm_max": .005,
            "inherited_Q4_macro_harm_max": .01,
            "inherited_Q4_positive_recall_harm_max": .02,
            "exploratory_supported_group_f1_harm_max": .02,
            "exploratory_supported_group_recall_harm_max": .03,
            "window_guards": window_guards,
            "subgroup_guard_failures": failures_67 + failures_75,
        },
    }


def verify_score_result(result_path: Path, run_root: Path, train_path: Path, *,
                        freeze_signature: str, source_commit: str, root_pass_sha256: str,
                        train_sha256: str, expected_matrix_hashes: dict[str, str]) -> dict[str, Any]:
    """Re-read approved Train labels and independently reconstruct the full score/gate report."""
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    ready_path = Path(run_root) / "PREDICTIONS_READY.json"
    readback_path = Path(run_root) / "READBACK_VERIFICATION.json"
    ready_sha = sha256_file(ready_path)
    readback_sha = sha256_file(readback_path)
    require(sha256_file(train_path) == train_sha256,
            "Independent result verifier Train bytes differ from the frozen hash")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    require(ready.get("freeze_signature") == freeze_signature
            and ready.get("all_four_fit_receipts_durable") is True
            and ready.get("query_labels_opened") is False
            and ready.get("test_or_sample_access") is False,
            "Independent result verifier found an invalid predictions-ready manifest")
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    current_readback = verify_readback_pair(Path(run_root), freeze_signature, expected_matrix_hashes)
    require(readback.get("status") == current_readback["status"]
            and readback.get("predictions") == current_readback["predictions"]
            and readback.get("predictions_ready_sha256") == ready_sha
            and readback.get("repeat_readback_sha256") == canonical_sha256(current_readback),
            "Independent result verifier found stale prediction readback")
    bundles = {}
    for origin in (67, 75):
        suffix = f"E{origin}"
        path = Path(run_root) / suffix / f"predictions_{suffix}.npz"
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == PREDICTION_KEYS, "Independent result bundle schema mismatch")
            bundles[origin] = {name: archive[name].copy() for name in PREDICTION_KEYS}
    e67 = bundles[67]
    rows = np.asarray(e67["row_positions"], dtype=np.int64)
    ids = np.asarray(e67["record_ids"]).astype(str)
    labels = _read_selected_labels(Path(train_path), rows, ids)
    labels_by_row = {int(row): int(target) for row, target in zip(rows.tolist(), labels.tolist())}
    e75 = bundles[75]
    for row, record_id in zip(np.asarray(e75["row_positions"]).tolist(),
                              np.asarray(e75["record_ids"]).astype(str).tolist()):
        require(labels_by_row.get(int(row)) is not None and ids[np.flatnonzero(rows == int(row))[0]] == record_id,
                "Independent E75 score row/ID does not match E67 Train identity")
    stats = recompute_temporal_metrics(bundles, labels_by_row)

    query_events = [event for event in (json.loads(line) for line in
                    (Path(run_root) / "fit_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line)
                    if event.get("event") == "QUERY_LABELS_OPENED"]
    expected_query = {"event": "QUERY_LABELS_OPENED", "candidate_id": "CLS_POS_POISSON_BAG2_V1",
                      "freeze_signature": freeze_signature,
                      "root_prefit_pass_sha256": root_pass_sha256,
                      "prediction_ready_sha256": ready_sha, "readback_sha256": readback_sha,
                      "score_row_positions_sha256": sha256_array(rows),
                      "score_record_ids_sha256": sha256_array(ids), "rows": int(len(rows)),
                      "source": "official Train artifact_signal",
                      "purpose": "matched E67/E75 development scoring after prediction freeze",
                      "test_or_sample_access": False}
    require(len(query_events) == 1 and query_events[0] == expected_query,
            "Independent result verifier found an invalid query-label ledger event")
    expected = {
        "schema_version": 1, "candidate_id": "CLS_POS_POISSON_BAG2_V1",
        "status": "SCORED_EXPLORATORY_ONLY", "freeze_signature": freeze_signature,
        "source_commit": source_commit, "root_prefit_pass_sha256": root_pass_sha256,
        "candidate_role": "two fixed Poisson-weighted gamma replicas; no automatic champion change",
        "threshold_policy": {"present": .50, "missing": .53230976129477,
                             "missing_threshold_is_global_oof_not_origin_learned": True,
                             "paired_historical_development_not_fresh_validation": True},
        **stats,
        "input_lineage": {"train_sha256": train_sha256,
                          "scored_e67_target_sha256": sha256_array(labels),
                          "query_label_event_sha256": canonical_sha256(expected_query),
                          "prediction_bundles_sha256": {
                              f"E{origin}": sha256_file(Path(run_root) / f"E{origin}" / f"predictions_E{origin}.npz")
                              for origin in (67, 75)},
                          "prediction_readback_sha256": readback_sha},
        "query_labels_read_after_predictions_verified": True,
        "test_or_sample_access": False, "kaggle_upload": False,
    }
    require(canonical_sha256(result) == canonical_sha256(expected),
            "Independent score recomputation differs from persisted RESULT file")
    return {"status": "PASS_INDEPENDENT_SCORE_AND_GATE_RECOMPUTATION",
            "result_sha256": sha256_file(result_path),
            "freeze_signature": freeze_signature,
            "unique_pool_rows": stats["unique_primary_pool"]["rows"],
            "scientific_qualified": stats["qualification"]["scientific_qualified"],
            "exploratory_eligible": stats["qualification"]["exploratory_eligible"]}
