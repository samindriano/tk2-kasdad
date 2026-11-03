"""Exact-impact-conditioned text retrieval for the RNBR impact-match pilot.

This module contains deterministic, target-agnostic geometry and shuffle
helpers plus pickle-free, immutable array persistence. Official Train targets
are passed only to the separate execution wrapper after its source/geometry
freeze has been committed and the task slot has started.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse


MAX_NEIGHBORS = 32
MIN_NEIGHBORS = 5
SHUFFLE_SEED = 42


class ImpactMatchError(ValueError):
    """A frozen geometry, cell shuffle, prediction, or persistence check failed."""


def arrays_equal(left: Any, right: Any) -> bool:
    """Exact array equality while treating NaNs in aligned float slots equally."""
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "OUS":
        digest = hashlib.sha256()
        for item in array.reshape(-1):
            data = str(item).encode("utf-8", errors="surrogatepass")
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
        return digest.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_prediction_decile(predictions: Any) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ImpactMatchError("R06 predictions must be a finite one-dimensional vector")
    return np.minimum(9, np.floor(np.clip(values, 0.0, 100.0) / 10.0)).astype(np.int8)


def exact_impact_top32(
    query_matrix: sparse.spmatrix,
    donor_matrix: sparse.spmatrix,
    *,
    query_impact: Any,
    donor_impact: Any,
    donor_rows: Any,
    query_stream_missing: Any,
    max_neighbors: int = MAX_NEIGHBORS,
) -> dict[str, np.ndarray]:
    """Search all present donors in the exact float64 impact cell for top32.

    Missing query impact and missing query stream have zero geometry. Infinite
    impacts, nonfinite matrix values, and duplicate donor rows are errors.
    Similarity ties are ordered by the original donor source row.
    """
    query = query_matrix.tocsr().astype(np.float64, copy=False)
    donor = donor_matrix.tocsr().astype(np.float64, copy=False)
    qimpact = np.asarray(query_impact, dtype=np.float64)
    dimpact = np.asarray(donor_impact, dtype=np.float64)
    rows = np.asarray(donor_rows, dtype=np.int64)
    missing = np.asarray(query_stream_missing, dtype=bool)
    if (query.ndim != 2 or donor.ndim != 2 or query.shape[1] != donor.shape[1]
            or query.shape[0] != len(qimpact) or query.shape[0] != len(missing)
            or donor.shape[0] != len(dimpact) or donor.shape[0] != len(rows)):
        raise ImpactMatchError("Impact retrieval inputs are not aligned")
    if max_neighbors < 1 or max_neighbors > MAX_NEIGHBORS:
        raise ImpactMatchError("The frozen maximum is at most 32 neighbors")
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0):
        raise ImpactMatchError("Impact donor source rows must be unique and nonnegative")
    if np.isinf(qimpact).any() or not np.isfinite(dimpact).all():
        raise ImpactMatchError("Infinite query impact or nonfinite donor impact is invalid")
    if (not np.isfinite(query.data).all() or not np.isfinite(donor.data).all()
            or np.any(query.data < 0.0) or np.any(donor.data < 0.0)):
        raise ImpactMatchError("TF-IDF geometry must be finite and nonnegative")

    n_query = query.shape[0]
    neighbor_rows = np.full((n_query, max_neighbors), -1, dtype=np.int64)
    neighbor_similarity = np.zeros((n_query, max_neighbors), dtype=np.float64)
    neighbor_count = np.zeros(n_query, dtype=np.int16)
    donor_groups: dict[float, list[int]] = {}
    query_groups: dict[float, list[int]] = {}
    for donor_index, impact in enumerate(dimpact):
        donor_groups.setdefault(float(impact), []).append(donor_index)
    for query_index, impact in enumerate(qimpact):
        if missing[query_index] or not np.isfinite(impact):
            continue
        if float(impact) in donor_groups:
            query_groups.setdefault(float(impact), []).append(query_index)

    for impact, query_indices in query_groups.items():
        donor_indices = np.asarray(donor_groups[impact], dtype=np.int64)
        donor_order = np.argsort(rows[donor_indices], kind="stable")
        donor_indices = donor_indices[donor_order]
        donor_cell = donor[donor_indices]
        for start in range(0, len(query_indices), 128):
            qindices = np.asarray(query_indices[start:start + 128], dtype=np.int64)
            similarities = (query[qindices] @ donor_cell.T).toarray()
            if not np.isfinite(similarities).all() or np.any(similarities < -1e-12):
                raise ImpactMatchError("Computed cosine geometry is nonfinite or negative")
            similarities = np.maximum(similarities, 0.0)
            for local, query_index in enumerate(qindices):
                positive = np.flatnonzero(similarities[local] > 0.0)
                if positive.size == 0:
                    continue
                scores = similarities[local, positive]
                tied_order = np.lexsort((rows[donor_indices[positive]], -scores))
                chosen_local = positive[tied_order[:max_neighbors]]
                count = len(chosen_local)
                neighbor_rows[query_index, :count] = rows[donor_indices[chosen_local]]
                neighbor_similarity[query_index, :count] = similarities[local, chosen_local]
                neighbor_count[query_index] = count

    return {"neighbor_rows": neighbor_rows,
            "neighbor_similarities": neighbor_similarity,
            "neighbor_count": neighbor_count}


def shuffle_residuals_within_exact_cells(
    residuals: Any,
    donor_cycles: Any,
    donor_impact: Any,
    donor_r06_predictions: Any,
    donor_rows: Any,
    *,
    seed: int = SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Permute residual sources within cycle × exact impact × fixed R06 decile.

    RNG state is reset by each call. Cells are traversed in ascending
    (cycle, float64 impact, decile) order and donor source row order.
    """
    residual = np.asarray(residuals, dtype=np.float64)
    cycles = np.asarray(donor_cycles, dtype=np.int64)
    impact = np.asarray(donor_impact, dtype=np.float64)
    baseline = np.asarray(donor_r06_predictions, dtype=np.float64)
    rows = np.asarray(donor_rows, dtype=np.int64)
    n = len(rows)
    if (any(array.ndim != 1 or len(array) != n for array in
            (residual, cycles, impact, baseline))
            or len(np.unique(rows)) != n or np.any(rows < 0)):
        raise ImpactMatchError("Shuffle donor arrays are not aligned or source rows are duplicated")
    if (not np.isfinite(residual).all() or not np.isfinite(impact).all()
            or not np.isfinite(baseline).all()):
        raise ImpactMatchError("Shuffle cells require finite residuals, impacts, and R06 values")
    deciles = fixed_prediction_decile(baseline)
    cells: dict[tuple[int, float, int], list[int]] = {}
    for index in range(n):
        key = (int(cycles[index]), float(impact[index]), int(deciles[index]))
        cells.setdefault(key, []).append(index)

    shuffled = residual.copy()
    source_index = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    movable_donors = 0
    moved_positions = 0
    for key in sorted(cells):
        indices = np.asarray(cells[key], dtype=np.int64)
        order = np.argsort(rows[indices], kind="stable")
        indices = indices[order]
        if len(indices) < 2:
            continue
        movable_donors += len(indices)
        permutation = rng.permutation(len(indices))
        source_index[indices] = indices[permutation]
        shuffled[indices] = residual[source_index[indices]]
        moved_positions += int(np.sum(permutation != np.arange(len(indices))))
    moved_values = int(np.sum(shuffled != residual))
    source_rows = rows[source_index]
    return shuffled, source_rows, {
        "cells": len(cells),
        "permutable_cells": sum(len(indices) >= 2 for indices in cells.values()),
        "permutable_donors": movable_donors,
        "source_positions_moved": moved_positions,
        "source_position_movement_rate": moved_positions / n if n else 0.0,
        "residual_values_moved": moved_values,
        "residual_value_movement_rate": moved_values / n if n else 0.0,
    }


def validate_same_impact_geometry(
    *,
    query_rows: Any,
    query_cycles: Any,
    query_impact: Any,
    query_ids: Any,
    query_stream_missing: Any,
    donor_rows: Any,
    donor_cycles: Any,
    donor_impact: Any,
    donor_ids: Any,
    origin: int,
    neighbor_rows: Any,
    neighbor_ids: Any,
    neighbor_similarities: Any,
    neighbor_count: Any,
    max_neighbors: int = MAX_NEIGHBORS,
) -> None:
    """Fail closed on lineage, strict time, exact impact, padding, or order."""
    qrows = np.asarray(query_rows, dtype=np.int64)
    qcycles = np.asarray(query_cycles, dtype=np.int64)
    qimpact = np.asarray(query_impact, dtype=np.float64)
    qids = np.asarray(query_ids).astype(str)
    qmissing = np.asarray(query_stream_missing, dtype=bool)
    drows = np.asarray(donor_rows, dtype=np.int64)
    dcycles = np.asarray(donor_cycles, dtype=np.int64)
    dimpact = np.asarray(donor_impact, dtype=np.float64)
    dids = np.asarray(donor_ids).astype(str)
    nrows = np.asarray(neighbor_rows, dtype=np.int64)
    nids = np.asarray(neighbor_ids).astype(str)
    sims = np.asarray(neighbor_similarities, dtype=np.float64)
    counts = np.asarray(neighbor_count, dtype=np.int64)
    n_query = len(qrows)
    if (any(x.ndim != 1 or len(x) != n_query for x in (qcycles, qimpact, qids, qmissing))
            or len(np.unique(qrows)) != n_query or np.any(qrows < 0)
            or len(drows) != len(dcycles) or len(drows) != len(dimpact) or len(drows) != len(dids)
            or len(np.unique(drows)) != len(drows) or np.any(drows < 0)
            or np.any(dcycles < 56) or np.any(dcycles > int(origin))
            or np.any(qcycles <= int(origin))
            or not np.isfinite(dimpact).all() or np.isinf(qimpact).any()
            or len(np.unique(dids)) != len(dids) or len(np.unique(qids)) != n_query):
        raise ImpactMatchError("Impact donor/query row, ID, cycle, or value lineage is invalid")
    if (nrows.shape != (n_query, max_neighbors) or nids.shape != nrows.shape
            or sims.shape != nrows.shape or counts.shape != (n_query,)
            or np.any(counts < 0) or np.any(counts > max_neighbors)
            or not np.isfinite(sims).all() or np.any(sims < 0.0)
            or np.any(sims > 1.0 + 1e-12)):
        raise ImpactMatchError("Impact neighbor arrays have invalid shapes, counts, or values")
    donor_lookup = {int(row): j for j, row in enumerate(drows)}
    for i in range(n_query):
        count = int(counts[i])
        if qmissing[i] or not np.isfinite(qimpact[i]):
            if count != 0:
                raise ImpactMatchError("Missing stream/impact query must have no neighbors")
        active_rows = nrows[i, :count]
        active_ids = nids[i, :count]
        active_sims = sims[i, :count]
        if (len(np.unique(active_rows)) != count or np.any(active_rows < 0)
                or np.any(active_sims <= 0.0)):
            raise ImpactMatchError("Impact neighbors must be unique positive-cosine donors")
        for j, source_row in enumerate(active_rows):
            donor_index = donor_lookup.get(int(source_row))
            if donor_index is None or active_ids[j] != dids[donor_index]:
                raise ImpactMatchError("Impact neighbor row/ID is outside the frozen donor bank")
            if float(dimpact[donor_index]) != float(qimpact[i]):
                raise ImpactMatchError("Impact neighbor does not exactly match query impact_load")
        if count > 1:
            expected = np.lexsort((active_rows, -active_sims))
            if not np.array_equal(expected, np.arange(count)):
                raise ImpactMatchError("Impact neighbors are not cosine-descending/source-row-stable")
        if (np.any(nrows[i, count:] != -1) or np.any(nids[i, count:] != "")
                or np.any(sims[i, count:] != 0.0)):
            raise ImpactMatchError("Impact neighbor padding differs from the frozen sentinel values")


def save_npz_once(path: str | Path, **arrays: Any) -> str:
    """Persist pickle-free arrays atomically; resume is read-only when equal."""
    destination = Path(path)
    materialized = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind == "O" for value in materialized.values()):
        raise ImpactMatchError("Object arrays are forbidden in impact-match artifacts")
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            if set(archive.files) != set(materialized):
                raise ImpactMatchError(f"Existing impact-match array schema differs: {destination}")
            for key, value in materialized.items():
                if not arrays_equal(np.asarray(archive[key]), value):
                    raise ImpactMatchError(f"Existing impact-match array differs: {key}")
        return sha256_file(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise ImpactMatchError(f"Unfinished staging artifact requires review: {staging}")
    with staging.open("xb") as stream:
        np.savez_compressed(stream, **materialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, destination)
    return sha256_file(destination)


def persist_bundle_once(
    folder: str | Path,
    *,
    fit_id: str,
    freeze_signature: str,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one receipt-bound bundle using an immutable staging directory."""
    destination = Path(folder)
    materialized = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind == "O" for value in materialized.values()):
        raise ImpactMatchError("Object arrays are forbidden in impact-match bundles")
    if destination.is_dir():
        receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        npz_path = destination / "arrays.npz"
        if (receipt.get("receipt_sha256") != sha256_bytes(canonical_bytes(body))
                or receipt.get("npz_sha256") != sha256_file(npz_path)
                or receipt.get("fit_id") != fit_id
                or receipt.get("freeze_signature") != freeze_signature):
            raise ImpactMatchError("Existing impact-match receipt/bundle lineage differs")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(materialized) or any(
                    not arrays_equal(np.asarray(archive[key]), value)
                    for key, value in materialized.items()):
                raise ImpactMatchError("Existing impact-match bundle arrays differ; no overwrite")
        return receipt
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise ImpactMatchError(f"Unfinished bundle staging directory requires review: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    npz_path = staging / "arrays.npz"
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **materialized)
        stream.flush()
        os.fsync(stream.fileno())
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "fit_id": fit_id,
        "freeze_signature": freeze_signature,
        "metadata": dict(metadata),
        "npz_sha256": sha256_file(npz_path),
        "array_shapes": {key: list(value.shape) for key, value in materialized.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in materialized.items()},
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(receipt))
    rec_path = staging / "receipt.json"
    with rec_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, destination)
    return receipt


def load_bundle(folder: str | Path, *, fit_id: str, freeze_signature: str
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = Path(folder)
    npz_path = directory / "arrays.npz"
    receipt_path = directory / "receipt.json"
    if not directory.is_dir() or not npz_path.is_file() or not receipt_path.is_file():
        raise ImpactMatchError(f"Impact-match bundle is absent or incomplete: {directory}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != sha256_bytes(canonical_bytes(body))
            or receipt.get("npz_sha256") != sha256_file(npz_path)
            or receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze_signature):
        raise ImpactMatchError("Impact-match receipt/hash/identity mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if (receipt.get("array_shapes") != {key: list(value.shape) for key, value in arrays.items()}
            or receipt.get("array_dtypes") != {key: str(value.dtype) for key, value in arrays.items()}):
        raise ImpactMatchError("Impact-match receipt array schema mismatch")
    return arrays, receipt

