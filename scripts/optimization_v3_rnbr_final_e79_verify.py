from __future__ import annotations

"""Independent, label-safe verifier for the one-shot E79 local benchmark."""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_final_e79 as runner  # noqa: E402
from scripts import optimization_v3_rnbr_official as official  # noqa: E402
from src.models.regression_neighbor_rnbr import EXPECTED_SOURCE_CUTOFFS, MAX_NEIGHBORS, MIN_NEIGHBORS  # noqa: E402

_LIMITER = threadpool_limits(limits=2)
VERIFY = runner.REPORT / "verification.json"


def _weighted_median_independent(residuals: np.ndarray, cosines: np.ndarray, source_rows: np.ndarray) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    weights = np.square(np.asarray(cosines, dtype=np.float64))
    rows = np.asarray(source_rows, dtype=np.int64)
    order = np.lexsort((rows, values))
    values = values[order]
    weights = weights[order]
    distinct, starts = np.unique(values, return_index=True)
    mass = np.add.reduceat(weights, starts)
    total = float(mass.sum())
    cumulative = 0.0
    for index, (value, weight) in enumerate(zip(distinct, mass)):
        cumulative += float(weight)
        if cumulative > total / 2 + total * 1e-12:
            return float(value)
        if abs(cumulative - total / 2) <= total * 1e-12:
            return float((value + distinct[index + 1]) / 2) if index + 1 < len(distinct) else float(value)
    return float(distinct[-1])


def _independent_geometry(query: sparse.csr_matrix, bank: sparse.csr_matrix,
                          bank_rows: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result_rows = np.full((query.shape[0], MAX_NEIGHBORS), -1, dtype=np.int64)
    result_cosines = np.zeros((query.shape[0], MAX_NEIGHBORS), dtype=np.float64)
    counts = np.zeros(query.shape[0], dtype=np.int16)
    # Use smaller blocks than the executor and reconstruct the top-k tie rule independently.
    for start in range(0, query.shape[0], 64):
        stop = min(start + 64, query.shape[0])
        score = (query[start:stop] @ bank.T).toarray()
        for local, row_i in enumerate(range(start, stop)):
            if missing[row_i]:
                continue
            eligible = np.flatnonzero(score[local] > 0)
            if eligible.size == 0:
                continue
            by_score = np.lexsort((bank_rows[eligible], -score[local, eligible]))
            chosen = eligible[by_score[:MAX_NEIGHBORS]]
            count = len(chosen)
            counts[row_i] = count
            result_rows[row_i, :count] = bank_rows[chosen]
            result_cosines[row_i, :count] = score[local, chosen]
    return result_rows, result_cosines, counts


def verify() -> dict[str, Any]:
    freeze = runner._read_json(runner.FREEZE)
    body = {k: v for k, v in freeze.items() if k != "freeze_signature"}
    if freeze.get("freeze_signature") != runner.digest_bytes(runner.canonical_bytes(body)):
        raise runner.E79Error("E79 freeze signature is invalid")
    if freeze != runner.make_freeze():
        raise runner.E79Error("Frozen E79 sources, inputs, or preflight changed")
    receipt = runner._read_json(runner.FIT_RECEIPT)
    receipt_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt.get("fit_id") != runner.FIT_ID or receipt.get("status") != "FIT_OUTPUT_DURABLE"
            or receipt.get("receipt_sha256") != runner.digest_bytes(runner.canonical_bytes(receipt_body))
            or receipt.get("bundle_sha256") != runner.sha_file(runner.BUNDLE)
            or receipt.get("bank_matrix_file_sha256") != runner.sha_file(runner.BANK_MATRIX)
            or receipt.get("freeze_signature") != freeze["freeze_signature"]):
        raise runner.E79Error("E79 durable fit receipt does not verify")
    with np.load(runner.BUNDLE, allow_pickle=False) as archive:
        bundle = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {
        "donor_rows", "donor_ids", "donor_cycles", "donor_targets", "donor_r06_oof",
        "donor_oof_cutoffs", "donor_oof_ids", "donor_residuals", "query_ids", "query_cycles",
        "query_missing", "baseline", "neighbor_rows", "neighbor_similarities", "neighbor_counts",
        "weighted_median", "delta", "prediction",
    }
    if set(bundle) != required:
        raise runner.E79Error(f"E79 bundle keys differ: {sorted(set(bundle) ^ required)}")
    train = official.load_targetfree_train()
    expected_donors = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= 79) & train["present"]).astype(np.int64)
    donor_rows = bundle["donor_rows"].astype(np.int64)
    if not np.array_equal(donor_rows, expected_donors):
        raise runner.E79Error("E79 donor row population is not exactly present-stream cycles 56-79")
    if (len(np.unique(bundle["donor_ids"].astype(str))) != len(donor_rows)
            or not np.array_equal(bundle["donor_ids"].astype(str), train["ids"][donor_rows])
            or not np.array_equal(bundle["donor_cycles"].astype(np.int64), train["cycles"][donor_rows])
            or not np.all(train["present"][donor_rows])):
        raise runner.E79Error("E79 donor IDs/cycles/stream availability do not join to official Train")
    donor_y = runner._load_donor_targets(donor_rows, train)
    if not np.array_equal(donor_y, bundle["donor_targets"].astype(np.float64)):
        raise runner.E79Error("Persisted donor target values differ from official eligible Train rows")
    pilot_freeze = runner._read_json(runner.PILOT_FREEZE)
    r06, oof_caches = runner._load_r06_inputs(pilot_freeze, train)
    oof, cutoffs, oof_ids = runner._donor_oof(donor_rows, train["cycles"][donor_rows],
                                               train["ids"][donor_rows], oof_caches)
    residual = donor_y - oof
    for name, expected in (("donor_targets", donor_y), ("donor_r06_oof", oof),
                           ("donor_oof_cutoffs", cutoffs), ("donor_oof_ids", oof_ids),
                           ("donor_residuals", residual)):
        if not np.array_equal(bundle[name], expected):
            raise runner.E79Error(f"Independent donor lineage differs for {name}")
    if not np.array_equal(cutoffs, np.asarray([EXPECTED_SOURCE_CUTOFFS[int(c)] for c in train["cycles"][donor_rows]])):
        raise runner.E79Error("E79 donor OOF cutoffs do not strictly precede donor cycles")

    bank = sparse.load_npz(runner.BANK_MATRIX).tocsr()
    if runner._csr_digest(bank) != receipt["bank_matrix"]:
        raise runner.E79Error("Saved sparse donor geometry hash differs from fit receipt")
    train_text = pd.read_csv(runner.TRAIN, usecols=["signal_stream"]).signal_stream.iloc[donor_rows]
    donor_docs, donor_missing = runner._documents(train_text)
    if donor_missing.any():
        raise runner.E79Error("Donor text unexpectedly contains missing values")
    vectorizer = joblib.load(runner.VECTORIZER)
    reconstructed_bank = vectorizer.transform(donor_docs).tocsr()
    if runner._csr_digest(reconstructed_bank) != runner._csr_digest(bank):
        raise runner.E79Error("Saved bank matrix is not reproducible from the reused frozen A02 vectorizer")

    reg_test, _, sample = runner._targetfree_test()
    test_ids = reg_test.record_id.astype(str).to_numpy()
    test_cycles = reg_test.cycle_code.to_numpy(dtype=np.int64)
    if (not np.array_equal(bundle["query_ids"].astype(str), test_ids)
            or not np.array_equal(bundle["query_cycles"].astype(np.int64), test_cycles)
            or np.any(test_cycles <= 79)):
        raise runner.E79Error("E79 query rows/IDs/cycles differ from official target-free Test")
    baseline, r06_test_identity = runner._load_r06_test(reg_test)
    if not np.array_equal(bundle["baseline"].astype(np.float64), baseline):
        raise runner.E79Error("Persisted E79 baseline differs from manifest-bound full-Train R06 Test predictions")
    docs, missing = runner._documents(reg_test.signal_stream)
    if not np.array_equal(bundle["query_missing"].astype(bool), missing):
        raise runner.E79Error("Persisted Test stream-missing mask differs from normalized official features")
    query = vectorizer.transform(docs).tocsr()
    rows_recomputed, similarities_recomputed, counts_recomputed = _independent_geometry(query, bank, donor_rows, missing)
    for name, expected in (("neighbor_rows", rows_recomputed), ("neighbor_similarities", similarities_recomputed),
                           ("neighbor_counts", counts_recomputed)):
        if not np.array_equal(bundle[name], expected):
            raise runner.E79Error(f"Independent full Test neighbor geometry differs for {name}")
    if np.any(counts_recomputed > MAX_NEIGHBORS) or np.any((similarities_recomputed < 0) | (similarities_recomputed > 1 + 1e-12)):
        raise runner.E79Error("E79 Test cosine geometry violates the frozen positive top32 rule")

    baseline = bundle["baseline"].astype(np.float64)
    missing = bundle["query_missing"].astype(bool)
    nrows = bundle["neighbor_rows"].astype(np.int64)
    scores = bundle["neighbor_similarities"].astype(np.float64)
    counts = bundle["neighbor_counts"].astype(np.int64)
    donor_lookup = {int(row): index for index, row in enumerate(donor_rows)}
    delta = np.zeros(len(baseline), dtype=np.float64)
    medians = np.zeros(len(baseline), dtype=np.float64)
    fallback = missing | (counts < MIN_NEIGHBORS)
    for i, count in enumerate(counts):
        if fallback[i]:
            continue
        k = int(count)
        source_rows = nrows[i, :k]
        donor_ix = np.asarray([donor_lookup[int(row)] for row in source_rows], dtype=np.int64)
        median = _weighted_median_independent(residual[donor_ix], scores[i, :k], source_rows)
        medians[i] = median
        delta[i] = float(np.clip(0.25 * median, -5.0, 5.0))
    prediction = np.clip(baseline + delta, 0.0, 100.0)
    if (not np.array_equal(medians, bundle["weighted_median"])
            or not np.array_equal(delta, bundle["delta"])
            or not np.array_equal(prediction, bundle["prediction"])
            or not np.array_equal(prediction[fallback], baseline[fallback])
            or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))):
        raise runner.E79Error("Independent weighted-median/delta/fallback/prediction reconstruction failed")

    if not runner.CSV.is_file() or not runner.MANIFEST.is_file():
        raise runner.E79Error("Final E79 CSV or manifest is missing")
    csv_frame = pd.read_csv(runner.CSV, dtype={"record_id": str})
    manifest = runner._read_json(runner.MANIFEST)
    if (csv_frame.columns.tolist() != ["record_id", "threat_rank"] or len(csv_frame) != 29_421
            or not np.array_equal(csv_frame.record_id.astype(str).to_numpy(), sample.record_id.astype(str).to_numpy())
            or not np.array_equal(csv_frame.threat_rank.to_numpy(dtype=np.float64), prediction)
            or runner.sha_file(runner.CSV) != manifest.get("csv", {}).get("sha256")
            or manifest.get("csv", {}).get("prediction_raw_sha256") != runner.sha_array(prediction, "<f8")):
        raise runner.E79Error("Final E79 CSV schema/order/values/hash failed")
    events = runner._ledger_events()
    starts = [event for event in events if event.get("fit_id") == runner.FIT_ID and event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("fit_id") == runner.FIT_ID and event.get("event") == "FIT_SAVED"]
    if (len(starts) != 1 or len(saved) != 1 or saved[0].get("receipt_sha256") != runner.sha_file(runner.FIT_RECEIPT)
            or runner._ledger_snapshot()["pending"] != 0):
        raise runner.E79Error("E79 fit ledger does not reconcile to one saved slot")
    result = {
        "status": "PASS", "fit_id": runner.FIT_ID, "freeze_signature": freeze["freeze_signature"],
        "fit_receipt_sha256": runner.sha_file(runner.FIT_RECEIPT), "bundle_sha256": runner.sha_file(runner.BUNDLE),
        "donor_rows": len(donor_rows), "donor_cycles": [56, 79],
        "target_values_recomputed_for_eligible_donors_only": True,
        "r06_oof_donor_mapping": "PASS", "a02_vectorizer_fit_reused": True,
        "a02_vectorizer_refit": False, "bank_matrix_reconstruction": "PASS",
        "independent_test_geometry_rows_checked": len(test_ids),
        "independent_top32_similarity_values_recomputed": int(counts_recomputed.sum()),
        "r06_test_baseline_lineage": r06_test_identity,
        "query_missing_rows": int(missing.sum()), "supported_rows_ge5": int(np.sum(counts >= MIN_NEIGHBORS)),
        "fallback_rows": int(fallback.sum()), "r06_changed_rows": int(np.sum(prediction != baseline)),
        "prediction_range": [float(prediction.min()), float(prediction.max())],
        "csv_path": runner.CSV.relative_to(ROOT).as_posix(), "csv_sha256": runner.sha_file(runner.CSV),
        "csv_rows": len(csv_frame), "csv_columns": csv_frame.columns.tolist(),
        "test_labels_read": False, "classification_labels_or_probabilities_read": False,
        "kaggle_upload": False, "ledger_snapshot": runner._ledger_snapshot(),
    }
    runner._write_once(VERIFY, result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False, allow_nan=False))
