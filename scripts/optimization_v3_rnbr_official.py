"""Official Train-only runner for the frozen four-fit R-NBR diagnostic pilot.

The four "fits" are residual-memory constructions; this module fits no estimator.
The implementation keeps donor-label reads after FIT_STARTED and query-label reads
out of the runner entirely. Query labels are opened only by the independent
post-fit verifier after every prediction bundle is durable and verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS,
    PILOT_ARMS,
    PILOT_ORIGINS,
    RnbrProtocolError,
    predict_from_saved_neighbors,
    shuffle_residuals_within_cycle_decile,
    validate_donor_value_lineage,
    validate_saved_geometry,
)

REG_TRAIN = ROOT / "data/raw/regression/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
PREPARATION = ROOT / "reports/optimization_v3/rnbr_pilot_v1/PREPARATION_PREFLIGHT.json"
PROTOCOL = ROOT / "reports/optimization_v3/rnbr_pilot_v1/PROTOCOL.md"
FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1"
REPORT_ROOT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1"
LEDGER_SHA_BEFORE = "6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147"
TRAIN_SHA = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
R06_SOURCE = "src/models/optimization_v3_batch1_r06.py"
R06_SOURCE_SHA = "37869d6125d48fc4afb2a3c439cbb19d5586f35f1cd3a501b762e42530fcd970"
R06_RECIPE = "experiments/optimization_v3/batch1/R06/recipe_revision_02.json"
R06_RECIPE_SHA = "6814978b3228fb741429dbd6b8af6d2c6caa8044eff0e079bbeb50066643b788"
CARD_ID = "REGRESSION_RNBR_PILOT"
STAGE = "OFFICIAL_FOUR_FIT_MEMORY_PILOT"
AMENDMENT_ID = "RNBR_PILOT_CAP70_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = tuple(f"RNBR_PILOT_E{o}_{a.upper()}" for o in PILOT_ORIGINS for a in PILOT_ARMS)
INPUT_FIELDS = ("score_rows", "score_record_ids", "score_cycles", "neighbor_count",
                "neighbor_source_rows", "neighbor_ids", "neighbor_similarities")


class OfficialRnbrError(RuntimeError):
    """A source, budget, output, or frozen-recipe invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any, dtype: str | None = None) -> str:
    arr = np.asarray(value, dtype=dtype)
    if arr.dtype.kind in "OUS":
        h = hashlib.sha256()
        for item in arr.reshape(-1):
            data = str(item).encode("utf-8", errors="surrogatepass")
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        return h.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise OfficialRnbrError(f"Refusing to overwrite different frozen file: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _code_paths() -> tuple[str, ...]:
    return ("scripts/optimization_v3_rnbr_official.py",
            "scripts/optimization_v3_rnbr_official_verify.py",
            "src/models/regression_neighbor_rnbr.py",
            "tests/test_optimization_v3_rnbr_official.py")


def _preparation() -> dict[str, Any]:
    data = json.loads(PREPARATION.read_text(encoding="utf-8"))
    if data.get("status") != "PASS" or data.get("official_train_target_free_alignment", {}).get("status") != "PASS":
        raise OfficialRnbrError("The historical target-free preparation report is not PASS")
    if data.get("current_ledger_snapshot", {}).get("ledger_sha256") != LEDGER_SHA_BEFORE:
        raise OfficialRnbrError("Preparation report ledger checkpoint differs")
    return data


def _cache_paths(prep: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    r06: dict[int, dict[str, Any]] = {}
    for entry in prep["R06_saved_cache_inventory"]:
        cutoff = int(entry["cutoff"])
        r06[cutoff] = {
            "npz": entry["cache_npz"], "json": entry["cache_json"],
            "npz_sha256": entry["npz_sha256"], "sidecar_sha256": entry["sidecar_sha256"],
            "freeze_signature": entry["freeze_signature"],
        }
    a02: dict[int, dict[str, Any]] = {}
    for origin in PILOT_ORIGINS:
        item = prep["A02_saved_geometry"][str(origin)]
        arms = item["geometry_cache_lineage"]
        paths: dict[str, Any] = {}
        root_name = ("optimization_v3_a02_neighbors_v1" if origin in (67, 75)
                     else "optimization_v3_a02_continuation_v1")
        for arm in PILOT_ARMS:
            base = f"experiments/runs/{root_name}/E{origin}/{arm}"
            meta = arms[arm]
            paths[arm] = {
                "npz": f"{base}/predictions.npz",
                "predictions_json": f"{base}/predictions.json",
                "receipt_json": f"{base}/fit_receipt.json",
                "state_json": f"{base}/fit_state.json",
                "model": f"{base}/model.joblib",
                "npz_sha256": meta["predictions_npz_sha256"],
                "model_sha256": meta["model_sha256"],
                "state_sha256": meta["state_sha256"],
                "fit_id": meta["fit_id"],
                "freeze_signature": meta["freeze_signature"],
                "geometry_hashes": meta["geometry_hashes"],
            }
        a02[origin] = paths
    return r06, a02


def make_freeze() -> dict[str, Any]:
    """Build an immutable source/recipe manifest without reading target columns."""
    prep = _preparation()
    if sha256_file(REG_TRAIN) != TRAIN_SHA:
        raise OfficialRnbrError("Official Regression Train hash differs from the frozen source")
    if sha256_file(ROOT / R06_SOURCE) != R06_SOURCE_SHA or sha256_file(ROOT / R06_RECIPE) != R06_RECIPE_SHA:
        raise OfficialRnbrError("R06 source or recipe hash differs from verified lineage")
    r06, a02 = _cache_paths(prep)
    for cutoff, item in r06.items():
        for key, field in (("npz", "npz_sha256"), ("json", "sidecar_sha256")):
            if sha256_file(ROOT / item[key]) != item[field]:
                raise OfficialRnbrError(f"R06 saved cache changed: cutoff={cutoff}/{key}")
    a02_paths = {}
    for origin, arms in a02.items():
        a02_paths[str(origin)] = {}
        for arm, item in arms.items():
            for path_key, hash_key in (("npz", "npz_sha256"), ("model", "model_sha256")):
                if sha256_file(ROOT / item[path_key]) != item[hash_key]:
                    raise OfficialRnbrError(f"A02 cache lineage changed: E{origin}/{arm}/{path_key}")
            a02_paths[str(origin)][arm] = dict(item)
            for field in ("predictions_json", "receipt_json", "state_json"):
                a02_paths[str(origin)][arm][field + "_sha256"] = sha256_file(ROOT / item[field])
    code = {rel: sha256_file(ROOT / rel) for rel in _code_paths()}
    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "RNBR_LATE_TWO_ORIGIN_DIAGNOSTIC_B0_OFFICIAL_EXECUTION",
        "source_commit_required_clean_at_run": True,
        "official_train": {"path": "data/raw/regression/Train.csv", "rows": 50608,
                           "sha256": TRAIN_SHA, "target_used": "threat_rank",
                           "target_rows_by_fit": "only present-stream donor rows within origin history"},
        "r06_lineage": {"source_commit": "be5e86750ec725e45698054a91fad299ddc24eb4",
                        "source_path": R06_SOURCE, "source_sha256": R06_SOURCE_SHA,
                        "recipe_path": R06_RECIPE, "recipe_sha256": R06_RECIPE_SHA,
                        "cache_inventory": {str(k): v for k, v in sorted(r06.items())}},
        "a02_saved_geometry": a02_paths,
        "evaluation_references": {
            "R22_W4": {"npz": "experiments/runs/optimization_v3_batch1/R22/R22_fit67.npz",
                        "sidecar": "experiments/runs/optimization_v3_batch1/R22/R22_fit67.json",
                        "npz_sha256": sha256_file(ROOT / "experiments/runs/optimization_v3_batch1/R22/R22_fit67.npz"),
                        "sidecar_sha256": sha256_file(ROOT / "experiments/runs/optimization_v3_batch1/R22/R22_fit67.json"),
                        "array": "prediction", "cutoff": 67},
            "R22_W6": {"npz": "experiments/runs/optimization_v3_final_continuation/R22/R22_fit75.npz",
                        "sidecar": "experiments/runs/optimization_v3_final_continuation/R22/R22_fit75.json",
                        "npz_sha256": sha256_file(ROOT / "experiments/runs/optimization_v3_final_continuation/R22/R22_fit75.npz"),
                        "sidecar_sha256": sha256_file(ROOT / "experiments/runs/optimization_v3_final_continuation/R22/R22_fit75.json"),
                        "array": "R22_ORIGINAL", "cutoff": 75},
            "R06_historical_results": {"path": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json",
                                       "sha256": sha256_file(ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json")},
            "R06_historical_verification": {"path": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json",
                                            "sha256": sha256_file(ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json")},
        },
        "preparation_report": {"path": str(PREPARATION.relative_to(ROOT)).replace("\\", "/"),
                               "sha256": sha256_file(PREPARATION)},
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha256_file(PROTOCOL),
                     "formula": "clip(R06 + clip(0.25 * weighted_median(R06 OOF residual, cosine^2), -5, +5), 0, 100)",
                     "shuffle": "within exact cycle x fixed R06 prediction decile, default_rng(42), stable source row"},
        "code_sha256": code,
        "fit_ids": list(FIT_IDS),
        "fit_count": 4,
        "fit_semantics": "one persisted past-only residual memory per origin and arm; no estimator refit",
        "views": {"primary": "E67 cycles 68-71 plus E75 cycles 76-79",
                  "H12_L": "E67 predictions cycles 68-79 only"},
        "boundaries": {"classification_targets": "not loaded or used",
                       "test_or_sample_submission": "not opened",
                       "query_targets": "not opened until all four fit bundles are durable and independently verified",
                       "new_tfidf_or_neighbor_fit": False},
        "resource_policy": {"max_parallel_memory_fits": 1, "threads": 2,
                            "job_memory_gib": 6, "reserved_memory_gib": 2},
    }
    body["freeze_signature"] = sha256_bytes(canonical_bytes(body))
    return body


def write_freeze() -> dict[str, Any]:
    payload = make_freeze()
    _write_json_once(FREEZE, payload)
    return payload


def load_freeze(*, require_clean: bool = False) -> dict[str, Any]:
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    signature = data.get("freeze_signature")
    body = {k: v for k, v in data.items() if k != "freeze_signature"}
    if signature != sha256_bytes(canonical_bytes(body)):
        raise OfficialRnbrError("Official freeze signature is invalid")
    current = make_freeze()
    if data != current:
        raise OfficialRnbrError("Frozen source/cache/code identity differs from current files")
    if require_clean:
        if _run_git("status", "--porcelain", "--untracked-files=all"):
            raise OfficialRnbrError("Official fit requires the verified preparation commit and clean working tree")
        if not _run_git("rev-parse", "HEAD"):
            raise OfficialRnbrError("Cannot determine source commit")
    return data


def load_targetfree_train() -> dict[str, np.ndarray]:
    """Read only ID, cycle, and stream availability; never parse threat_rank."""
    frame = pd.read_csv(REG_TRAIN, usecols=["record_id", "cycle_code", "signal_stream"])
    if len(frame) != 50608:
        raise OfficialRnbrError(f"Unexpected official Train rows: {len(frame)}")
    rows = np.arange(len(frame), dtype=np.int64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    ids = frame["record_id"].astype(str).to_numpy()
    present = frame["signal_stream"].fillna("").astype(str).str.strip().ne("").to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise OfficialRnbrError("Train IDs are not unique")
    if cycles.min() != 47 or cycles.max() != 79:
        raise OfficialRnbrError("Official Train cycle range differs from frozen protocol")
    return {"rows": rows, "cycles": cycles, "ids": ids, "present": present}


def _verify_sidecar(sidecar_path: Path, npz_path: Path, *, expected_npz_hash: str,
                    card: str, cutoff: int) -> dict[str, Any]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    ident = sidecar.get("identity", {})
    integrity = {k: v for k, v in sidecar.items() if k != "integrity_sha256"}
    if (sha256_file(npz_path) != expected_npz_hash
            or sidecar.get("npz_sha256") != expected_npz_hash
            or sidecar.get("identity_sha256") != sha256_bytes(canonical_bytes(ident))
            or sidecar.get("integrity_sha256") != sha256_bytes(canonical_bytes(integrity))
            or ident.get("card_id") != card or int(ident.get("cutoff", -1)) != cutoff):
        raise OfficialRnbrError(f"Saved {card} cache identity/hash failed at cutoff {cutoff}")
    return sidecar


def load_r06_predictions(train: Mapping[str, np.ndarray], freeze: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for cutoff_text, spec in freeze["r06_lineage"]["cache_inventory"].items():
        cutoff = int(cutoff_text)
        npz = ROOT / spec["npz"]
        sidecar_path = ROOT / spec["json"]
        sidecar = _verify_sidecar(sidecar_path, npz, expected_npz_hash=spec["npz_sha256"],
                                  card="R06", cutoff=cutoff)
        identity = sidecar["identity"]
        if (identity.get("freeze_signature") != spec["freeze_signature"]
                and identity.get("campaign_freeze_signature") != spec["freeze_signature"]):
            raise OfficialRnbrError(f"R06 cache freeze signature differs at cutoff {cutoff}")
        with np.load(npz, allow_pickle=False) as z:
            rows = np.asarray(z["score_rows"], dtype=np.int64).copy()
            ids = np.asarray(z["score_record_ids"]).astype(str, copy=True)
            prediction = np.asarray(z["prediction"], dtype=np.float64).copy()
        if (len(rows) != len(ids) or len(rows) != len(prediction)
                or not np.array_equal(ids, train["ids"][rows])
                or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))
                or sidecar["identity"].get("score_rows") != rows.tolist()):
            raise OfficialRnbrError(f"R06 output rows, IDs, or values are invalid at cutoff {cutoff}")
        result[cutoff] = {"rows": rows, "ids": ids, "prediction": prediction,
                          "cycles": train["cycles"][rows]}
    return result


def load_geometry(origin: int, train: Mapping[str, np.ndarray], freeze: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arms = freeze["a02_saved_geometry"][str(origin)]
    values: dict[str, dict[str, np.ndarray]] = {}
    for arm in PILOT_ARMS:
        spec = arms[arm]
        path = ROOT / spec["npz"]
        if sha256_file(path) != spec["npz_sha256"] or sha256_file(ROOT / spec["model"]) != spec["model_sha256"]:
            raise OfficialRnbrError(f"A02 saved geometry/model hash differs at E{origin}/{arm}")
        for field, expected in (("predictions_json", None), ("receipt_json", None), ("state_json", None)):
            side_path = ROOT / spec[field]
            if not side_path.is_file():
                raise OfficialRnbrError(f"Missing A02 lineage artifact: {side_path}")
        pred_meta = json.loads((ROOT / spec["predictions_json"]).read_text(encoding="utf-8"))
        receipt = json.loads((ROOT / spec["receipt_json"]).read_text(encoding="utf-8"))
        state = json.loads((ROOT / spec["state_json"]).read_text(encoding="utf-8"))
        if (pred_meta.get("npz_sha256") != spec["npz_sha256"]
                or pred_meta.get("fit_id") != spec["fit_id"]
                or pred_meta.get("residual_arm") != arm
                or pred_meta.get("freeze_signature") != spec["freeze_signature"]
                or receipt.get("query_labels_passed_to_fit") is not False
                or receipt.get("test_data_used") is not False
                or state.get("state_sha256") != spec["state_sha256"]):
            raise OfficialRnbrError(f"A02 saved geometry provenance failed at E{origin}/{arm}")
        with np.load(path, allow_pickle=False) as z:
            missing = set(INPUT_FIELDS) - set(z.files)
            if missing:
                raise OfficialRnbrError(f"A02 cache lacks target-free geometry arrays: {sorted(missing)}")
            values[arm] = {key: np.asarray(z[key]).copy() for key in INPUT_FIELDS}
    for key in INPUT_FIELDS:
        if not np.array_equal(values["observed"][key], values["shuffled"][key]):
            raise OfficialRnbrError(f"A02 observed/shuffled geometry differs at E{origin}/{key}")
    data = values["observed"]
    rows = data["score_rows"].astype(np.int64)
    low, high = ((68, 79) if origin == 67 else (76, 79))
    expected_rows = np.flatnonzero((train["cycles"] >= low) & (train["cycles"] <= high))
    if (not np.array_equal(rows, expected_rows)
            or not np.array_equal(data["score_record_ids"].astype(str), train["ids"][rows])
            or not np.array_equal(data["score_cycles"].astype(np.int64), train["cycles"][rows])):
        raise OfficialRnbrError(f"A02 query rows/IDs/cycles do not match E{origin} horizon")
    if not np.array_equal(data["neighbor_count"].astype(np.int64) == 0, ~train["present"][rows]):
        raise OfficialRnbrError(f"A02 saved missing-stream geometry differs at E{origin}")
    return data


def donor_source_cutoffs(cycles: np.ndarray) -> np.ndarray:
    try:
        return np.asarray([EXPECTED_SOURCE_CUTOFFS[int(c)] for c in cycles], dtype=np.int64)
    except KeyError as exc:
        raise OfficialRnbrError(f"No R06 OOF source cutoff for donor cycle {exc.args[0]}") from exc


def aligned_r06_predictions(rows: np.ndarray, cycles: np.ndarray,
                            cache: Mapping[int, Mapping[str, np.ndarray]]) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float64)
    for cutoff in np.unique(donor_source_cutoffs(cycles)):
        pick = np.flatnonzero(donor_source_cutoffs(cycles) == cutoff)
        source = cache[int(cutoff)]
        lookup = {int(row): float(value) for row, value in zip(source["rows"], source["prediction"])}
        try:
            out[pick] = [lookup[int(rows[i])] for i in pick]
        except KeyError as exc:
            raise OfficialRnbrError(f"R06 OOF cutoff {cutoff} has no donor row {exc.args[0]}") from exc
    if not np.isfinite(out).all():
        raise OfficialRnbrError("Aligned R06 donor predictions are nonfinite")
    return out


def read_regression_targets(rows: np.ndarray, *, purpose: str) -> dict[str, np.ndarray]:
    """Parse threat_rank for an explicit row allowlist only; skips every other CSV row."""
    requested = np.asarray(rows, dtype=np.int64)
    if (requested.ndim != 1 or len(requested) == 0 or len(np.unique(requested)) != len(requested)
            or np.any(requested < 0) or np.any(requested >= 50608)):
        raise OfficialRnbrError("Target read row allowlist is invalid")
    allowed = set(map(int, requested))
    frame = pd.read_csv(
        REG_TRAIN,
        usecols=["record_id", "cycle_code", "threat_rank"],
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in allowed,
    )
    if len(frame) != len(requested):
        raise OfficialRnbrError(f"Target reader parsed {len(frame)} rows for allowlist of {len(requested)}")
    values = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise OfficialRnbrError("Official donor/query values are outside finite [0,100]")
    return {"ids": frame["record_id"].astype(str).to_numpy(),
            "cycles": pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64),
            "target": values, "purpose": np.asarray([purpose], dtype="U")}


def build_official_payload(*, origin: int, arm: str, train: Mapping[str, np.ndarray],
                           cache: Mapping[int, Mapping[str, np.ndarray]],
                           geometry: Mapping[str, np.ndarray], donor_targets: Mapping[str, np.ndarray],
                           freeze: Mapping[str, Any], source_commit: str,
                           dataset_tag: str = "OFFICIAL_REGRESSION_TRAIN") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if origin not in PILOT_ORIGINS or arm not in PILOT_ARMS:
        raise OfficialRnbrError("Only the frozen E67/E75 observed/shuffled fits are allowed")
    cycles, present = train["cycles"], train["present"]
    donor_rows = np.flatnonzero((cycles >= 56) & (cycles <= origin) & present).astype(np.int64)
    donor_ids = train["ids"][donor_rows]
    donor_cycles = cycles[donor_rows]
    donor_cutoffs = donor_source_cutoffs(donor_cycles)
    query_rows = geometry["score_rows"].astype(np.int64)
    query_ids = geometry["score_record_ids"].astype(str)
    query_cycles = geometry["score_cycles"].astype(np.int64)
    query_missing = ~present[query_rows]
    source_values = np.asarray(donor_targets["target"], dtype=np.float64)
    if (not np.array_equal(donor_targets["ids"].astype(str), donor_ids)
            or not np.array_equal(donor_targets["cycles"].astype(np.int64), donor_cycles)
            or len(source_values) != len(donor_rows)):
        raise OfficialRnbrError("Donor target values are not aligned to the frozen present-stream bank")
    donor_r06 = aligned_r06_predictions(donor_rows, donor_cycles, cache)
    validate_donor_value_lineage(
        origin=origin, donor_rows=donor_rows, donor_ids=donor_ids, donor_cycles=donor_cycles,
        donor_source_cutoffs=donor_cutoffs, target_rows=donor_rows, target_ids=donor_ids,
        target_cycles=donor_cycles, target_values=source_values, r06_rows=donor_rows,
        r06_ids=donor_ids, r06_cycles=donor_cycles, r06_predictions=donor_r06,
    )
    residuals = source_values - donor_r06
    residuals_used = (residuals if arm == "observed" else
                      shuffle_residuals_within_cycle_decile(
                          residuals, donor_cycles, donor_r06, donor_rows, seed=42))
    r06_origin = cache[origin]
    q_lookup = {int(row): float(pred) for row, pred in zip(r06_origin["rows"], r06_origin["prediction"])}
    try:
        baseline = np.asarray([q_lookup[int(row)] for row in query_rows], dtype=np.float64)
    except KeyError as exc:
        raise OfficialRnbrError(f"R06 origin cache misses query row {exc.args[0]}") from exc
    neighbor_rows = geometry["neighbor_source_rows"].astype(np.int64)
    neighbor_ids = geometry["neighbor_ids"].astype(str)
    similarities = geometry["neighbor_similarities"].astype(np.float64)
    counts = geometry["neighbor_count"].astype(np.int64)
    validate_saved_geometry(
        origin=origin, donor_rows=donor_rows, donor_ids=donor_ids, donor_cycles=donor_cycles,
        donor_source_cutoffs=donor_cutoffs, donor_present=np.ones(len(donor_rows), dtype=bool),
        query_rows=query_rows, query_ids=query_ids, query_cycles=query_cycles,
        query_missing=query_missing, neighbor_rows=neighbor_rows, neighbor_ids=neighbor_ids,
        neighbor_cosines=similarities, neighbor_counts=counts,
    )
    predicted = predict_from_saved_neighbors(
        donor_rows=donor_rows, donor_residuals=residuals_used,
        query_r06_predictions=baseline, query_missing=query_missing,
        neighbor_rows=neighbor_rows, neighbor_cosines=similarities,
        neighbor_counts=counts,
    )
    arrays = {
        "score_rows": query_rows, "score_record_ids": query_ids,
        "score_cycles": query_cycles,
        "score_window": np.where(query_cycles <= 71, "W4", np.where(query_cycles <= 75, "W5", "W6")).astype("U2"),
        "prediction": predicted["prediction"], "baseline_prediction": baseline,
        "delta": predicted["delta"], "query_missing": query_missing,
        "neighbor_count_used": predicted["neighbor_count_used"],
        "neighbor_count": counts, "neighbor_source_rows": neighbor_rows,
        "neighbor_ids": neighbor_ids, "neighbor_similarities": similarities,
        "donor_rows": donor_rows, "donor_record_ids": donor_ids,
        "donor_cycles": donor_cycles, "donor_source_cutoffs": donor_cutoffs,
        "donor_r06_predictions": donor_r06, "donor_residuals": residuals,
        "donor_residuals_used": residuals_used,
    }
    identity = {
        "fit_id": f"RNBR_PILOT_E{origin}_{arm.upper()}", "origin": int(origin), "arm": arm,
        "dataset_tag": dataset_tag,
        "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
        "formula": "clip(R06 + clip(0.25 * weighted_median(residual, cosine^2), -5, +5), 0, 100)",
        "minimum_neighbors": 5, "maximum_neighbors": 32,
        "donor_rows": int(len(donor_rows)), "query_rows": int(len(query_rows)),
        "query_cycle_range": [int(query_cycles.min()), int(query_cycles.max())],
        "query_labels_read": False, "test_or_sample_submission_opened": False,
        "classification_labels_used": False, "new_estimator_fit_count": 0,
        "memory_construction_fit_count": 1,
    }
    audit = {
        "donor_target_read": "Regression threat_rank only; present rows in cycles 56..origin",
        "donor_target_values_count": int(len(source_values)),
        "donor_target_values_sha256": sha256_array(source_values, "<f8"),
        "donor_residual_sha256": sha256_array(residuals, "<f8"),
        "arm_residual_sha256": sha256_array(residuals_used, "<f8"),
        "neighbor_rows_checked": int(np.sum(counts)),
        "supported_present_queries": int(np.sum((~query_missing) & (counts >= 5))),
        "missing_queries_unchanged": int(np.sum(query_missing)),
        "resource_snapshot": {"available_gib": round(psutil.virtual_memory().available / (1024**3), 3),
                              "max_job_gib": 6, "threads": 2, "concurrent_memory_fits": 1},
    }
    return arrays, {"identity": identity, "audit": audit}


def _bundle_dir(fit_id: str) -> Path:
    if fit_id not in FIT_IDS:
        raise OfficialRnbrError(f"Unknown official pilot fit ID: {fit_id}")
    return RUN_ROOT / "bundles" / fit_id


def persist_bundle(fit_id: str, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, Any]:
    destination = _bundle_dir(fit_id)
    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise OfficialRnbrError(f"Refusing to overwrite or auto-repair existing bundle {destination}")
    staging.mkdir(parents=True, exist_ok=False)
    npz_path = staging / "arrays.npz"
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    receipt: dict[str, Any] = {
        "schema_version": 1, "identity": dict(metadata["identity"]),
        "identity_sha256": sha256_bytes(canonical_bytes(metadata["identity"])),
        "npz_sha256": sha256_file(npz_path),
        "metadata": dict(metadata["audit"]),
        "array_shapes": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
        "array_dtypes": {k: str(np.asarray(v).dtype) for k, v in arrays.items()},
    }
    receipt["integrity_sha256"] = sha256_bytes(canonical_bytes(receipt))
    rec_path = staging / "receipt.json"
    with rec_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, destination)
    return receipt


def load_bundle(fit_id: str, freeze: Mapping[str, Any], *, source_commit: str | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folder = _bundle_dir(fit_id)
    npz_path, receipt_path = folder / "arrays.npz", folder / "receipt.json"
    if not folder.is_dir() or not npz_path.is_file() or not receipt_path.is_file():
        raise OfficialRnbrError(f"Official bundle is absent/incomplete: {folder}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt.get("identity", {})
    body = {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    if (receipt.get("integrity_sha256") != sha256_bytes(canonical_bytes(body))
            or receipt.get("npz_sha256") != sha256_file(npz_path)
            or receipt.get("identity_sha256") != sha256_bytes(canonical_bytes(identity))
            or identity.get("fit_id") != fit_id
            or identity.get("freeze_signature") != freeze["freeze_signature"]
            or (source_commit is not None and identity.get("source_commit") != source_commit)):
        raise OfficialRnbrError(f"Official bundle receipt/hash/identity mismatch: {fit_id}")
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in z.files}
    rows = arrays.get("score_rows")
    if (rows is None or rows.ndim != 1 or len(np.unique(rows)) != len(rows)
            or arrays.get("prediction", np.asarray([])).shape != rows.shape
            or arrays.get("baseline_prediction", np.asarray([])).shape != rows.shape
            or not np.isfinite(arrays["prediction"]).all()
            or np.any((arrays["prediction"] < 0) | (arrays["prediction"] > 100))):
        raise OfficialRnbrError(f"Official bundle arrays invalid: {fit_id}")
    if receipt.get("array_shapes") != {k: list(v.shape) for k, v in arrays.items()}:
        raise OfficialRnbrError(f"Official bundle shape receipt mismatch: {fit_id}")
    return arrays, receipt


def _read_ledger() -> list[dict[str, Any]]:
    events = []
    for line_no, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfficialRnbrError(f"Malformed budget ledger line {line_no}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise OfficialRnbrError(f"Malformed budget ledger event line {line_no}")
        events.append(event)
    return events


def ledger_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    auths = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(auths) != 1:
        raise OfficialRnbrError("Expected one immutable Research Next batch authorization")
    cap_events = [e for e in events if e.get("event") in {"BUDGET_CAP_AMENDMENT", "BUDGET_CATEGORY_AMENDMENT"}]
    current_global = int(cap_events[-1].get("global_cap", auths[0]["cap"])) if cap_events else int(auths[0]["cap"])
    current_categories = dict(cap_events[-1].get("effective_caps", auths[0]["category_caps"])) if cap_events else dict(auths[0]["category_caps"])
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    releases = {e.get("reservation_key"): int(e.get("unused_fits", 0))
                for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active_by_category = {category: sum(int(e["fits"]) - releases.get(e["reservation_key"], 0)
                                        for e in reservations if e.get("category") == category)
                          for category in current_categories}
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    pending = [s for s in starts if not any(t.get("fit_id") == s.get("fit_id")
                                             and t.get("event") in {"FIT_SAVED", "FIT_FAILED"} for t in events)]
    return {"event_rows": len(events), "global_cap": current_global,
            "effective_category_caps": current_categories,
            "active_reservations_by_category": active_by_category,
            "active_reservations": sum(active_by_category.values()),
            "fit_starts": len(starts), "fit_saved": len(saved),
            "fit_failed": len(failed), "fit_pending": len(pending)}


def _append_event(event: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(event)
    value.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(LEDGER, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return value


def amend_and_reserve(freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    events = _read_ledger()
    current_hash = sha256_file(LEDGER)
    snapshot = ledger_snapshot(events)
    previous_caps = {"pilot": 34, "continuation": 31, "control_replay": 0, "technical_retry": 1}
    if current_hash == LEDGER_SHA_BEFORE:
        if (snapshot["global_cap"] != 66 or snapshot["effective_category_caps"] != previous_caps
                or snapshot["active_reservations"] != 66 or snapshot["fit_starts"] != 66
                or snapshot["fit_saved"] != 64 or snapshot["fit_failed"] != 2
                or snapshot["fit_pending"] != 0):
            raise OfficialRnbrError(f"Live pre-amendment ledger accounting differs: {snapshot}")
        amendment = {
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Samuel explicit approval relayed by Sol; four pilot fits only",
            "prior_global_cap": 66, "global_cap": 70,
            "prior_effective_caps": previous_caps,
            "effective_caps": {"pilot": 38, "continuation": 31, "control_replay": 0,
                               "technical_retry": 1},
            "active_reservations_before": snapshot["active_reservations_by_category"],
            "active_total_before": snapshot["active_reservations"],
            "historical_started_saved_failed_before": {"started": 66, "saved": 64, "failed": 2},
            "pre_amendment_event_rows": len(events), "pre_amendment_ledger_sha256": current_hash,
            "source_commit": source_commit, "freeze_signature": freeze["freeze_signature"],
            "reason": "Add exactly four authorized R-NBR pilot memory fits; continuation, replay, and retry caps unchanged.",
        }
        _append_event(amendment)
        events = _read_ledger()
    else:
        prior = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                 and e.get("amendment_id") == AMENDMENT_ID]
        if len(prior) != 1:
            raise OfficialRnbrError(f"Ledger changed from authorized checkpoint without this amendment: {current_hash}")
        event = prior[0]
        if (event.get("global_cap") != 70 or event.get("effective_caps") !=
                {"pilot": 38, "continuation": 31, "control_replay": 0, "technical_retry": 1}
                or event.get("freeze_signature") != freeze["freeze_signature"]):
            raise OfficialRnbrError("Existing RNBR cap amendment differs")
    snapshot = ledger_snapshot(events)
    if snapshot["global_cap"] != 70 or snapshot["effective_category_caps"] != {
            "pilot": 38, "continuation": 31, "control_replay": 0, "technical_retry": 1}:
        raise OfficialRnbrError("Effective cap amendment was not applied exactly")

    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservations) > 1:
        raise OfficialRnbrError("Duplicate RNBR reservation key")
    if not reservations:
        if (snapshot["active_reservations"] != 66 or snapshot["active_reservations_by_category"]["pilot"] != 34
                or snapshot["fit_starts"] != 66 or snapshot["fit_pending"] != 0):
            raise OfficialRnbrError(f"Live post-amendment pre-reservation snapshot differs: {snapshot}")
        reserved = _append_event({
            "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
            "amendment_id": AMENDMENT_ID, "global_cap": 70, "effective_category_cap": 38,
            "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
        })
    else:
        reserved = reservations[0]
        if (reserved.get("fits") != 4 or reserved.get("fit_ids") != list(FIT_IDS)
                or reserved.get("amendment_id") != AMENDMENT_ID
                or reserved.get("freeze_signature") != freeze["freeze_signature"]):
            raise OfficialRnbrError("Existing RNBR reservation differs")
    after = ledger_snapshot(_read_ledger())
    events_after = _read_ledger()
    rnbr_starts = [e for e in events_after if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    rnbr_ids = [e.get("fit_id") for e in rnbr_starts]
    if (after["global_cap"] != 70 or after["active_reservations"] != 70
            or after["active_reservations_by_category"].get("pilot") != 38
            or not 66 <= after["fit_starts"] <= 70 or len(rnbr_ids) != len(set(rnbr_ids))
            or after["fit_starts"] != 66 + len(rnbr_ids)):
        raise OfficialRnbrError(f"Post-reservation ledger accounting differs: {after}")
    return {"reservation": reserved, "snapshot": after, "ledger_sha256": sha256_file(LEDGER)}


def start_fit(fit_id: str, origin: int, arm: str, freeze: Mapping[str, Any], source_commit: str) -> None:
    events = _read_ledger()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    if starts:
        if len(starts) != 1:
            raise OfficialRnbrError(f"Duplicate start event: {fit_id}")
        terminal = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if terminal:
            raise OfficialRnbrError(f"Fit {fit_id} already terminal; resume must verify its durable bundle")
        if _bundle_dir(fit_id).is_dir():
            raise OfficialRnbrError(f"Fit {fit_id} has a durable output after start; verify and reconcile before continuing")
        raise OfficialRnbrError(f"Fit {fit_id} has an unresolved prior start; no retry is authorized")
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservation) != 1 or fit_id not in reservation[0].get("fit_ids", []):
        raise OfficialRnbrError(f"Fit {fit_id} is not in the unique exact reservation")
    snap = ledger_snapshot(events)
    if snap["fit_pending"]:
        raise OfficialRnbrError("Another fit is pending in the shared ledger; refusing concurrent work")
    if (snap["fit_starts"] >= snap["global_cap"]
            or snap["active_reservations_by_category"].get("pilot", 0) > snap["effective_category_caps"]["pilot"]):
        raise OfficialRnbrError("Budget cap exhausted before official fit start")
    started_for_reservation = sum(e.get("event") == "FIT_STARTED" and e.get("reservation_key") == RESERVATION_KEY
                                  for e in events)
    if started_for_reservation >= 4:
        raise OfficialRnbrError("Four-fit R-NBR reservation already consumed")
    _append_event({
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": f"{arm}_residual_memory",
        "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
    })


def finish_fit(fit_id: str, status: str, *, receipt_sha256: str | None = None,
               error: str | None = None) -> None:
    if status not in {"FIT_SAVED", "FIT_FAILED"}:
        raise OfficialRnbrError("Invalid terminal fit status")
    events = _read_ledger()
    started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    if len(started) != 1:
        raise OfficialRnbrError(f"Fit {fit_id} lacks exactly one start event")
    terminal = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if terminal:
        expected_hash = receipt_sha256 if status == "FIT_SAVED" else None
        if len(terminal) != 1 or terminal[0].get("event") != status or terminal[0].get("receipt_sha256") != expected_hash:
            raise OfficialRnbrError(f"Fit {fit_id} already has conflicting terminal state")
        return
    _append_event({
        "event": status, "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": started[0]["cutoff"],
        "receipt_sha256": receipt_sha256, "error": error,
    })


def assert_resource_budget() -> dict[str, Any]:
    own_pid = os.getpid()
    competing = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info.get("pid") == own_pid:
            continue
        cmdline = " ".join(process.info.get("cmdline") or []).lower()
        process_name = str(process.info.get("name") or "").lower()
        if process_name.startswith("python") and ("optimization_v3" in cmdline or "rnbr_pilot" in cmdline):
            competing.append({"pid": process.info.get("pid"), "cmdline": cmdline[:240]})
    if competing:
        raise OfficialRnbrError(f"Another project training/fit process is active: {competing}")
    process_rss = int(psutil.Process().memory_info().rss)
    available = int(psutil.virtual_memory().available)
    estimate = 512 * 1024**2
    if process_rss + estimate > 6 * 1024**3 or available < 2 * 1024**3:
        raise OfficialRnbrError(
            f"Resource guard failed before FIT_STARTED: RSS={process_rss}, available={available}"
        )
    return {"process_rss_bytes_before_fit": process_rss,
            "system_available_bytes_before_fit": available,
            "reserved_memory_bytes": 2 * 1024**3,
            "estimated_additional_memory_bytes": estimate,
            "process_limit_bytes": 6 * 1024**3, "max_concurrent_memory_fits": 1}


def _run_independent_verifier(*, all_fits: bool) -> dict[str, Any]:
    script = ROOT / "scripts/optimization_v3_rnbr_official_verify.py"
    args = [sys.executable, str(script), "--verify-all" if all_fits else "--verify-pending"]
    completed = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise OfficialRnbrError(f"Independent verifier failed: {completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_official() -> dict[str, Any]:
    freeze = load_freeze(require_clean=True)
    source_commit = _run_git("rev-parse", "HEAD")
    # This preflight loads target-free features and saved predictions/geometry only.
    train = load_targetfree_train()
    r06_cache = load_r06_predictions(train, freeze)
    geo_by_origin = {o: load_geometry(o, train, freeze) for o in PILOT_ORIGINS}
    budget = amend_and_reserve(freeze, source_commit)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    fit_receipts = []
    for origin in PILOT_ORIGINS:
        donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin)
                                    & train["present"]).astype(np.int64)
        donor_values: dict[str, np.ndarray] | None = None
        for arm in PILOT_ARMS:
            fit_id = f"RNBR_PILOT_E{origin}_{arm.upper()}"
            events = _read_ledger()
            started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
            saved_terminal = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
            if saved_terminal:
                arrays, receipt = load_bundle(fit_id, freeze, source_commit=source_commit)
                if receipt.get("integrity_sha256") != saved_terminal[0].get("receipt_sha256"):
                    raise OfficialRnbrError(f"Saved ledger receipt differs for {fit_id}")
                fit_receipts.append({"fit_id": fit_id, "status": "RESUMED_VERIFIED",
                                     "receipt_sha256": receipt["integrity_sha256"]})
                continue
            if started:
                if _bundle_dir(fit_id).is_dir():
                    # Reconcile only an already durable payload; never reconstruct or refit it.
                    arrays, receipt = load_bundle(fit_id, freeze, source_commit=source_commit)
                    verify_pending = _run_independent_verifier(all_fits=False)
                    verified = {item["fit_id"] for item in verify_pending.get("fit_bundle_verification", [])
                                if item.get("status") == "PASS"}
                    if fit_id not in verified:
                        raise OfficialRnbrError(f"Durable pending bundle did not independently verify: {fit_id}")
                    finish_fit(fit_id, "FIT_SAVED", receipt_sha256=receipt["integrity_sha256"])
                    fit_receipts.append({"fit_id": fit_id, "status": "RESUMED_VERIFIED",
                                         "receipt_sha256": receipt["integrity_sha256"],
                                         "bundle_npz_sha256": receipt["npz_sha256"]})
                    continue
                raise OfficialRnbrError(f"{fit_id} has an unresolved start; no automatic retry")
            resource_guard = assert_resource_budget()
            start_fit(fit_id, origin, arm, freeze, source_commit)
            try:
                if donor_values is None:
                    # Only donor rows (present stream, 56..origin) are parsed from threat_rank.
                    donor_values = read_regression_targets(donor_rows, purpose=f"E{origin}_DONORS")
                    if (not np.array_equal(donor_values["ids"], train["ids"][donor_rows])
                            or not np.array_equal(donor_values["cycles"], train["cycles"][donor_rows])):
                        raise OfficialRnbrError(f"Donor target IDs/cycles differ at E{origin}")
                payload, metadata = build_official_payload(
                    origin=origin, arm=arm, train=train, cache=r06_cache,
                    geometry=geo_by_origin[origin], donor_targets=donor_values,
                    freeze=freeze, source_commit=source_commit,
                )
                metadata["audit"]["pre_fit_resource_guard"] = resource_guard
                receipt = persist_bundle(fit_id, payload, metadata)
                arrays, reread = load_bundle(fit_id, freeze, source_commit=source_commit)
                if reread != receipt or not all(np.array_equal(arrays[k], payload[k]) for k in payload):
                    raise OfficialRnbrError(f"Durable readback differs for {fit_id}")
                # Independent formula and lineage recomputation; no query truth is read here.
                verify_one = _run_independent_verifier(all_fits=False)
                verified = {item["fit_id"] for item in verify_one.get("fit_bundle_verification", [])
                            if item.get("status") == "PASS"}
                if fit_id not in verified:
                    raise OfficialRnbrError(f"Independent verifier did not pass {fit_id}")
                finish_fit(fit_id, "FIT_SAVED", receipt_sha256=receipt["integrity_sha256"])
                fit_receipts.append({"fit_id": fit_id, "status": "FIT_SAVED",
                                     "receipt_sha256": receipt["integrity_sha256"],
                                     "bundle_npz_sha256": receipt["npz_sha256"]})
            except Exception as exc:
                terminal = [e for e in _read_ledger() if e.get("fit_id") == fit_id
                            and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
                if not terminal:
                    finish_fit(fit_id, "FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
                raise
    all_verified = _run_independent_verifier(all_fits=True)
    if all_verified.get("status") != "PASS" or all_verified.get("query_truth_loaded_after_all_four_fit_bundles_verified") is not True:
        raise OfficialRnbrError("Post-fit independent verification/evaluation did not pass")
    final = {
        "status": "FOUR_FIT_PILOT_COMPLETE_DIAGNOSTIC_ONLY",
        "source_commit": source_commit,
        "freeze_signature": freeze["freeze_signature"],
        "fit_receipts": fit_receipts,
        "budget_before_and_after": budget,
        "ledger_final_sha256": sha256_file(LEDGER),
        "independent_verification": all_verified,
        "query_truth_used_after_fit_bundles": True,
        "test_or_sample_submission_opened": False,
        "kaggle_used": False,
        "qualification_claim": False,
    }
    _write_json_once(REPORT_ROOT / "PILOT_RESULT.json", final)
    return final


def targetfree_preflight() -> dict[str, Any]:
    freeze = load_freeze(require_clean=False)
    train = load_targetfree_train()
    r06 = load_r06_predictions(train, freeze)
    geometry = {o: load_geometry(o, train, freeze) for o in PILOT_ORIGINS}
    events = _read_ledger()
    return {
        "status": "PASS", "ledger_sha256": sha256_file(LEDGER),
        "ledger_snapshot": ledger_snapshot(events),
        "train_rows": int(len(train["rows"])), "cycle_range": [int(train["cycles"].min()), int(train["cycles"].max())],
        "target_columns_loaded": [], "test_or_sample_submission_opened": False,
        "r06_cutoffs_verified": sorted(r06),
        "a02_query_rows": {str(o): int(len(geometry[o]["score_rows"])) for o in PILOT_ORIGINS},
        "freeze_signature": freeze["freeze_signature"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = write_freeze()
    elif args.preflight:
        result = targetfree_preflight()
    else:
        result = run_official()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
