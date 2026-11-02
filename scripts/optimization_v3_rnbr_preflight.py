"""Read-only R-NBR readiness audit; never loads either task target."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import zipfile

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
REG_TRAIN = ROOT / "data/raw/regression/Train.csv"
CLS_TRAIN = ROOT / "data/raw/classification/Train.csv"
REG_SHA = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
CLS_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
RECIPE = ROOT / "experiments/optimization_v3/batch1/R06/recipe_revision_02.json"
RECIPE_SHA = "6814978b3228fb741429dbd6b8af6d2c6caa8044eff0e079bbeb50066643b788"
SOURCE_COMMIT = "be5e86750ec725e45698054a91fad299ddc24eb4"
SOURCE_RELATIVE = "src/models/optimization_v3_batch1_r06.py"
SOURCE_SHA = "37869d6125d48fc4afb2a3c439cbb19d5586f35f1cd3a501b762e42530fcd970"
OUTPUT = ROOT / "reports/optimization_v3/rnbr_readiness_v1/INVENTORY.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def npy_headers(path: pathlib.Path) -> dict[str, dict[str, object]]:
    headers: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            headers[name[:-4]] = {
                "shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran)
            }
    return headers


def fold_specs() -> list[dict[str, object]]:
    specs = []
    for cutoff, window in ((55, (56, 59)), (59, (60, 63)), (63, (64, 67)),
                           (67, (68, 71)), (71, (72, 75)), (75, (76, 79))):
        if cutoff in (55, 67):
            base = ROOT / f"experiments/runs/optimization_v3_batch1/R06/R06_fit{cutoff}"
            freeze = ROOT / "experiments/optimization_v3/batch1/R06/freeze_revision_02.json"
            signature = "02ae1b4e6e6424a8ac7f66fd0f1b955e478f6265c4f071989d9d8c5ef532c0a8"
        else:
            base = ROOT / f"experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit{cutoff}"
            freeze = ROOT / "experiments/optimization_v3/expanded_benchmark_v1_recovery_v2/freeze.json"
            signature = "6cdecf6e0729b625981f23c7b7be4152a12806011a0a1852dd978eaec30d21ae"
        specs.append({"cutoff": cutoff, "query_window": list(window), "base": base,
                      "freeze": freeze, "freeze_signature": signature})
    return specs


def verify_source_lineage() -> dict[str, object]:
    source_path = ROOT / SOURCE_RELATIVE
    current = sha256_file(source_path)
    historical = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_RELATIVE}"], cwd=ROOT
    )
    historical_sha = hashlib.sha256(historical).hexdigest()
    recipe_hash = sha256_file(RECIPE)
    if current != SOURCE_SHA or historical_sha != SOURCE_SHA or recipe_hash != RECIPE_SHA:
        raise RuntimeError("R06 source or recipe lineage hash differs")
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if recipe.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("R06 recipe source_commit differs")
    return {"source_commit": SOURCE_COMMIT, "source_path": SOURCE_RELATIVE,
            "source_sha256_current_and_at_source_commit": SOURCE_SHA,
            "recipe_path": str(RECIPE.relative_to(ROOT)).replace("\\", "/"),
            "recipe_sha256": recipe_hash, "feature_allowlist": recipe["recipe"]["feature_allowlist"],
            "model": recipe["recipe"]["model"], "rank50": recipe["recipe"]["rank50"],
            "status": "PASS"}


def compare_train_features() -> tuple[pd.DataFrame, dict[str, object]]:
    excluded = {"artifact_signal", "threat_rank", "target", "label", "y"}
    cls_cols = pd.read_csv(CLS_TRAIN, nrows=0).columns.tolist()
    reg_cols = pd.read_csv(REG_TRAIN, nrows=0).columns.tolist()
    shared = [name for name in cls_cols if name in reg_cols and name.lower() not in excluded]
    if "record_id" not in shared or "cycle_code" not in shared:
        raise RuntimeError("Task-independent shared ID/cycle fields are absent")
    cls = pd.read_csv(CLS_TRAIN, usecols=shared)
    reg = pd.read_csv(REG_TRAIN, usecols=shared)
    mismatch_counts: dict[str, int] = {}
    for name in shared:
        equal = cls[name].eq(reg[name]) | (cls[name].isna() & reg[name].isna())
        mismatch_counts[name] = int((~equal).sum())
    if len(cls) != len(reg) or any(mismatch_counts.values()):
        raise RuntimeError("Classification and Regression target-free Train rows differ")
    evidence = {
        "classification_train_sha256": sha256_file(CLS_TRAIN),
        "regression_train_sha256": sha256_file(REG_TRAIN),
        "rows_each": int(len(reg)), "shared_target_free_columns": shared,
        "columns_compared_rowwise": len(shared), "mismatch_rows_by_column": mismatch_counts,
        "record_id_order_identical": True, "cycle_code_order_identical": True,
        "target_columns_not_loaded": ["artifact_signal", "threat_rank"],
        "status": "PASS",
    }
    return reg, evidence


def verify_caches(train: pd.DataFrame) -> list[dict[str, object]]:
    cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    ids = train["record_id"].astype(str).to_numpy()
    entries = []
    for spec in fold_specs():
        cutoff = int(spec["cutoff"])
        base = spec["base"]
        npz_path = pathlib.Path(str(base) + ".npz")
        sidecar_path = pathlib.Path(str(base) + ".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        freeze = json.loads(pathlib.Path(spec["freeze"]).read_text(encoding="utf-8"))
        identity = sidecar.get("identity")
        integrity = {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
        if sidecar.get("npz_sha256") != sha256_file(npz_path):
            raise RuntimeError(f"R06 NPZ hash differs from sidecar: cutoff {cutoff}")
        if sidecar.get("identity_sha256") != canonical_sha256(identity):
            raise RuntimeError(f"R06 identity hash differs: cutoff {cutoff}")
        if sidecar.get("integrity_sha256") != canonical_sha256(integrity):
            raise RuntimeError(f"R06 sidecar integrity differs: cutoff {cutoff}")
        if (identity.get("card_id") != "R06" or int(identity.get("cutoff", -1)) != cutoff
                or identity.get("freeze_signature") != spec["freeze_signature"]
                and identity.get("campaign_freeze_signature") != spec["freeze_signature"]
                or freeze.get("signature") != spec["freeze_signature"]):
            raise RuntimeError(f"R06 sidecar/freeze lineage differs: cutoff {cutoff}")

        with np.load(npz_path, allow_pickle=False) as archive:
            required = {"score_rows", "score_record_ids", "prediction", "raw_candidate", "baseline_prediction"}
            if not required.issubset(archive.files):
                raise RuntimeError(f"R06 cache lacks required prediction/identity arrays: cutoff {cutoff}")
            rows = np.asarray(archive["score_rows"])
            score_ids = np.asarray(archive["score_record_ids"]).astype(str)
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows)
                or np.any(rows < 0) or np.any(rows >= len(train)) or not np.all(np.diff(rows) > 0)):
            raise RuntimeError(f"R06 score rows are invalid: cutoff {cutoff}")
        if not np.array_equal(score_ids, ids[rows]) or not np.array_equal(identity.get("score_rows"), rows.tolist()):
            raise RuntimeError(f"R06 row and ID lineage differs: cutoff {cutoff}")
        headers = npy_headers(npz_path)
        for name in ("prediction", "raw_candidate", "baseline_prediction"):
            if headers[name]["shape"] != [len(rows)]:
                raise RuntimeError(f"R06 array header is misaligned: {cutoff}/{name}")
        start, stop = spec["query_window"]
        expected = np.flatnonzero((cycles >= int(start)) & (cycles <= int(stop)))
        selected = rows[(cycles[rows] >= int(start)) & (cycles[rows] <= int(stop))]
        if not np.array_equal(selected, expected):
            raise RuntimeError(f"R06 first-four-cycle window is incomplete: cutoff {cutoff}")
        per_cycle = {str(c): int(np.sum(cycles[selected] == c)) for c in range(int(start), int(stop) + 1)}
        entries.append({
            "cutoff": cutoff, "cache_npz": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
            "cache_json": str(sidecar_path.relative_to(ROOT)).replace("\\", "/"),
            "npz_sha256": sha256_file(npz_path), "sidecar_sha256": sha256_file(sidecar_path),
            "freeze_manifest": str(pathlib.Path(spec["freeze"]).relative_to(ROOT)).replace("\\", "/"),
            "freeze_signature": spec["freeze_signature"], "all_scored_cycles": [int(cycles[rows].min()), int(cycles[rows].max())],
            "all_score_rows": int(len(rows)), "first_four_cycle_window": [int(start), int(stop)],
            "first_four_rows": int(len(selected)), "first_four_rows_by_cycle": per_cycle,
            "score_id_and_row_alignment": "PASS", "prediction_arrays_present_and_aligned": True,
            "npz_members": sorted(headers),
            "saved_neighbor_geometry": any(any(t in key.lower() for t in
                ("neighbor", "distance", "cosine", "embedding", "vectorizer", "bank_matrix", "feature_matrix"))
                for key in headers),
            "target_values_loaded": False,
        })
    return entries


def _array_digest(values: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    if array.dtype.kind in "OUS":
        digest = hashlib.sha256()
        for value in array.reshape(-1):
            payload = str(value).encode("utf-8", errors="surrogatepass")
            digest.update(len(payload).to_bytes(8, "little", signed=False))
            digest.update(payload)
        return digest.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def verify_saved_a02_geometry(train: pd.DataFrame, r06_entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Validate only saved row/ID/cycle/top-neighbor/cosine arrays, never class labels."""
    r06_rows_by_cutoff: dict[int, np.ndarray] = {}
    r06_row_sets: dict[int, set[int]] = {}
    for entry in r06_entries:
        with np.load(ROOT / str(entry["cache_npz"]), allow_pickle=False) as archive:
            r06_rows_by_cutoff[int(entry["cutoff"])] = np.asarray(archive["score_rows"]).copy()
        r06_row_sets[int(entry["cutoff"])] = set(map(int, r06_rows_by_cutoff[int(entry["cutoff"])]))
    cycles = train["cycle_code"].to_numpy(dtype=np.int64)
    raw_ids = train["record_id"].astype(str).to_numpy()
    signal = train["signal_stream"].fillna("").astype(str)
    stream_present = signal.str.strip().ne("").to_numpy()
    score_ranges = {55: (56, 67), 59: (60, 63), 63: (64, 67),
                    67: (68, 79), 71: (72, 75), 75: (76, 79)}
    source_cutoff = {c: f for f, lo, hi in ((55, 56, 59), (59, 60, 63), (63, 64, 67),
                                            (67, 68, 71), (71, 72, 75), (75, 76, 79))
                     for c in range(lo, hi + 1)}
    output = []
    for origin in (59, 63, 67, 71, 75):
        run_root = ROOT / ("experiments/runs/optimization_v3_a02_neighbors_v1"
                           if origin in (67, 75) else "experiments/runs/optimization_v3_a02_continuation_v1")
        freeze_path = ROOT / ("experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
                              if origin in (67, 75) else "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json")
        expected_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))["freeze_signature"]
        arms: dict[str, dict[str, np.ndarray]] = {}
        arm_meta: dict[str, dict[str, object]] = {}
        for arm in ("observed", "shuffled"):
            folder = run_root / f"E{origin}" / arm
            npz_path = folder / "predictions.npz"
            pred_meta = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
            receipt = json.loads((folder / "fit_receipt.json").read_text(encoding="utf-8"))
            state = json.loads((folder / "fit_state.json").read_text(encoding="utf-8"))
            state_body = {key: value for key, value in state.items() if key != "state_sha256"}
            receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if (pred_meta.get("npz_sha256") != sha256_file(npz_path)
                    or pred_meta.get("fit_receipt_sha256") != receipt.get("receipt_sha256")
                    or receipt.get("receipt_sha256") != canonical_sha256(receipt_body)
                    or receipt.get("state_sha256") != canonical_sha256(state)
                    or state.get("state_sha256") != canonical_sha256(state_body)
                    or receipt.get("model_sha256") != sha256_file(folder / "model.joblib")
                    or int(pred_meta.get("origin_cycle", -1)) != origin
                    or int(receipt.get("origin_cycle", -1)) != origin
                    or state.get("origin_cycle") != origin
                    or pred_meta.get("freeze_signature") != expected_freeze
                    or receipt.get("freeze_signature") != expected_freeze
                    or pred_meta.get("residual_arm") != arm
                    or state.get("residual_arm") != arm
                    or receipt.get("query_labels_passed_to_fit") is not False
                    or receipt.get("test_data_used") is not False
                    or pred_meta.get("query_labels_passed_to_prediction") is not False
                    or pred_meta.get("test_data_used", False) is not False):
                raise RuntimeError(f"A02 saved geometry provenance failed: origin {origin}/{arm}")
            with np.load(npz_path, allow_pickle=False) as archive:
                required = {"score_rows", "score_record_ids", "score_cycles", "neighbor_count",
                            "neighbor_source_rows", "neighbor_ids", "neighbor_similarities"}
                if not required.issubset(archive.files):
                    raise RuntimeError(f"A02 geometry arrays absent: origin {origin}/{arm}")
                # Deliberately do not access labels, class predictions, corrections, or residual arrays.
                arrays = {key: np.asarray(archive[key]).copy() for key in required}
            rows = arrays["score_rows"].astype(np.int64, copy=False)
            query_cycles = cycles[rows]
            if (not np.array_equal(arrays["score_record_ids"].astype(str), raw_ids[rows])
                    or not np.array_equal(arrays["score_cycles"].astype(np.int64), query_cycles)):
                raise RuntimeError(f"A02 query ID/cycle alignment differs at origin {origin}/{arm}")
            low, high = score_ranges[origin]
            expected_rows = np.flatnonzero((cycles >= low) & (cycles <= high))
            if not np.array_equal(rows, expected_rows):
                raise RuntimeError(f"A02 complete query-cycle rows differ at origin {origin}/{arm}")
            # The R06 cache has the identical first-four query rows; 55 is the deterministic cold-start fallback.
            r06_cutoff = origin
            if not np.array_equal(rows, r06_rows_by_cutoff[r06_cutoff]):
                raise RuntimeError(f"A02/R06 query rows differ at origin {origin}/{arm}")

            count = arrays["neighbor_count"].astype(np.int64, copy=False)
            source_rows = arrays["neighbor_source_rows"].astype(np.int64, copy=False)
            neighbor_ids = arrays["neighbor_ids"].astype(str)
            similarities = arrays["neighbor_similarities"].astype(np.float64, copy=False)
            if (source_rows.shape != (len(rows), 32) or neighbor_ids.shape != source_rows.shape
                    or similarities.shape != source_rows.shape or count.shape != (len(rows),)
                    or np.any(count < 0) or np.any(count > 32) or not np.isfinite(similarities).all()
                    or np.any((similarities < 0.0) | (similarities > 1.0 + 1e-12))):
                raise RuntimeError(f"A02 neighbor geometry shapes/ranges invalid: origin {origin}/{arm}")
            query_missing = ~stream_present[rows]
            if np.any(count[query_missing] != 0) or np.any(count[~query_missing] == 0):
                raise RuntimeError(f"A02 missing/present query routing differs: origin {origin}/{arm}")
            neighbor_rows_checked = 0
            donor_positions: set[int] = set()
            for i, n_neighbors in enumerate(count):
                n_neighbors = int(n_neighbors)
                active = source_rows[i, :n_neighbors]
                inactive = source_rows[i, n_neighbors:]
                if (np.any(active < 0) or np.any(active >= len(train)) or len(np.unique(active)) != len(active)
                        or np.any(inactive != -1) or np.any(similarities[i, n_neighbors:] != 0.0)
                        or np.any(neighbor_ids[i, n_neighbors:] != "")):
                    raise RuntimeError(f"A02 neighbor padding/uniqueness differs: origin {origin}/{arm}")
                if not n_neighbors:
                    continue
                source_cycles = cycles[active]
                if (np.any(source_cycles < 56) or np.any(source_cycles > origin)
                        or np.any(source_cycles >= query_cycles[i]) or np.any(~stream_present[active])):
                    raise RuntimeError(f"A02 neighbor source violates past/present bank: origin {origin}/{arm}")
                # A02 builds its typed IDs from the NumPy Unicode cache IDs (numpy.str_).
                expected_neighbor_ids = np.asarray([f"str_:{raw_ids[row]}" for row in active])
                if not np.array_equal(neighbor_ids[i, :n_neighbors], expected_neighbor_ids):
                    raise RuntimeError(f"A02 neighbor row/ID alignment differs: origin {origin}/{arm}")
                if not np.all(np.diff(similarities[i, :n_neighbors]) <= 1e-12):
                    raise RuntimeError(f"A02 neighbor similarities are not descending: origin {origin}/{arm}")
                for source_row, source_cycle in zip(active, source_cycles):
                    fold = source_cutoff[int(source_cycle)]
                    if int(source_row) not in r06_row_sets[fold]:
                        raise RuntimeError(f"A02 donor lacks an exact past-only R06 OOF prediction: {source_cycle}")
                donor_positions.update(map(int, active))
                neighbor_rows_checked += n_neighbors

            diag = pred_meta["diagnostics"]
            geometry_hashes = {
                "neighbor_ids_sha256": _array_digest(neighbor_ids),
                "neighbor_source_rows_sha256": _array_digest(source_rows, "<i8"),
                "neighbor_similarities_sha256": _array_digest(similarities, "<f8"),
            }
            if any(diag.get(name) != value for name, value in geometry_hashes.items()):
                raise RuntimeError(f"A02 geometry hashes differ from receipt: origin {origin}/{arm}")
            arrays = {key: arrays[key] for key in sorted(required)}
            arms[arm] = {key: arrays[key] for key in (
                "score_rows", "score_record_ids", "score_cycles", "neighbor_count",
                "neighbor_source_rows", "neighbor_ids", "neighbor_similarities")}
            arm_meta[arm] = {
                "fit_id": pred_meta["fit_id"], "freeze_signature": pred_meta["freeze_signature"],
                "freeze_manifest": str(freeze_path.relative_to(ROOT)).replace("\\", "/"),
                "predictions_npz_sha256": sha256_file(npz_path), "model_sha256": receipt["model_sha256"],
                "state_sha256": state["state_sha256"], "bank_rows": int(state["bank_rows"]),
                "bank_cycle_range": [int(state["bank_cycle_min"]), int(state["bank_cycle_max"])],
                "bank_matrix_shape": state["bank_matrix_shape"],
                "neighbors_max": int(state["neighbors_max"]), "query_rows": int(len(rows)),
                "first_four_rows": int(np.sum((query_cycles >= low) & (query_cycles <= min(low + 3, high)))),
                "neighbors_checked": neighbor_rows_checked, "unique_donor_rows_used": len(donor_positions),
                "geometry_hashes": geometry_hashes,
                "target_derived_prediction_arrays_loaded": False,
            }
        for key in arms["observed"]:
            if not np.array_equal(arms["observed"][key], arms["shuffled"][key]):
                raise RuntimeError(f"A02 observed/shuffled geometry mismatch: origin {origin}/{key}")
        expected_bank_rows = int(np.sum((cycles >= 56) & (cycles <= origin) & stream_present))
        if any(int(arm_meta[arm]["bank_rows"]) != expected_bank_rows for arm in arm_meta):
            raise RuntimeError(f"A02 past-only donor-bank count differs: origin {origin}")
        output.append({"origin": origin, "query_window": [score_ranges[origin][0], min(score_ranges[origin][0] + 3, score_ranges[origin][1])],
                       "observed_and_shuffled_geometry_identical": True,
                       "target_free_geometry_reuse": "PASS",
                       "arms": arm_meta,
                       "note": "Only row/ID/cycle/top-neighbor/cosine arrays were loaded from NPZ; labels, class predictions, corrections and residual arrays were not loaded."})
    return output


def budget_snapshot() -> dict[str, object]:
    events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    finished_ids = {event.get("fit_id") for event in events
                    if event.get("event") in {"FIT_SAVED", "FIT_FAILED", "FIT_INTERRUPTED"}}
    caps = [int(event["global_cap"]) for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT"]
    base = [int(event["cap"]) for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    cap = max(caps + base)
    return {"ledger_sha256": sha256_file(LEDGER), "fit_starts": len(starts),
            "fit_saved": sum(event.get("event") == "FIT_SAVED" for event in events),
            "fit_failed_or_interrupted": sum(event.get("event") in {"FIT_FAILED", "FIT_INTERRUPTED"} for event in events),
            "pending_started_fits": len({event.get("fit_id") for event in starts} - finished_ids),
            "effective_cap": cap, "remaining_capacity": max(0, cap - len(starts)),
            "status": "PASS" if len(starts) <= cap and len({event.get("fit_id") for event in starts} - finished_ids) == 0 else "FAIL"}


def main() -> None:
    source = verify_source_lineage()
    train, feature_alignment = compare_train_features()
    caches = verify_caches(train)
    a02_geometry = verify_saved_a02_geometry(train, caches)
    budget = budget_snapshot()
    donor_support = []
    present = train["signal_stream"].fillna("").astype(str).str.strip().ne("").to_numpy()
    cycles = train["cycle_code"].to_numpy(dtype=np.int64)
    for cutoff in (55, 59, 63, 67, 71, 75):
        mask = (cycles >= 56) & (cycles <= cutoff) & present
        donor_support.append({"query_cutoff": cutoff, "donor_cycles": [56, cutoff] if cutoff >= 56 else [],
                              "eligible_prior_oof_donors": int(mask.sum()),
                              "status": "NO_R06_OOF_DONOR_HISTORY" if cutoff == 55 else "OOF_DONOR_PREDICTIONS_AVAILABLE"})
    result = {
        "audit": "R-NBR readiness preflight only; no fit, prediction, metric replay, Test/Sample access, or Kaggle call",
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_recipe": source, "train_feature_alignment": feature_alignment,
        "r06_cache_origins": caches,
        "past_only_donor_support": donor_support,
        "saved_geometry": {"R06_cache_geometry": "ABSENT", "classification_A02_target_free_neighbor_geometry": "VERIFIED_REUSABLE",
                           "a02_geometry": a02_geometry,
                           "reason": "The A02 NPZ exports top-neighbor row positions, IDs and cosine similarities separately from label/prediction arrays; observed and shuffled geometry match exactly, and all selected donors have past-only R06 OOF predictions."},
        "budget": budget,
        "task_fit_accounting": {"new_RNBR_target_memory_or_index_counts_as_task_fit": True,
                                "planned_corrected_origins": 5, "planned_arms_per_origin": 2,
                                "planned_new_task_fits": 10, "new_fit_authorized": False,
                                "fits_remaining_under_current_effective_cap": budget["remaining_capacity"]},
        "verdict": "GEOMETRY_AND_OOF_READY_EXECUTION_BLOCKED_BY_EXHAUSTED_FIT_CAP",
    }
    out = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(out, encoding="utf-8", newline="\n")
    print(json.dumps({"inventory": str(OUTPUT.relative_to(ROOT)), "verdict": result["verdict"],
                      "r06_origins": len(caches), "feature_alignment": feature_alignment["status"],
                      "fit_starts": budget["fit_starts"], "effective_cap": budget["effective_cap"],
                      "remaining_capacity": budget["remaining_capacity"],
                      "verified_a02_origins": len(a02_geometry)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
