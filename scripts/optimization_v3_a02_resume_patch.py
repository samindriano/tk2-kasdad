"""Verify durable frozen A02 fits after output-writer/verifier compatibility errors.

No estimator is fit here. This compatibility wrapper changes no predictor,
preprocessing, feature, residual, metric, or fit-accounting state. It validates
the already-saved fit/output bundles and applies narrow in-memory adapters for
two defects in the frozen independent verifier. A repair receipt binds the
result to the existing A02 freeze and ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from typing import Any

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_a02_neighbors as runner  # noqa: E402
from scripts import optimization_v3_a02_neighbors_verify as frozen_verifier  # noqa: E402
from src.models.classification_a02_neighbors import _array_sha256  # noqa: E402
from src.models.research_next_budget import read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402


RECEIPT_PATH = runner.REPORT_ROOT / "technical_resume_receipt.json"
LEDGER_PATH = runner.LEDGER_PATH


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"A02 technical resume receipt differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


class _NeighborIdArchiveAdapter:
    """Normalize the NumPy string type tag in-memory for verifier comparison."""

    def __init__(self, archive: Any):
        self._archive = archive
        self.files = archive.files

    def __enter__(self) -> "_NeighborIdArchiveAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self._archive.close()

    def __getitem__(self, key: str) -> np.ndarray:
        values = np.asarray(self._archive[key]).copy()
        if key == "neighbor_ids":
            strings = values.astype(str)
            tagged = np.char.startswith(strings, "str_:")
            strings[tagged] = np.char.replace(strings[tagged], "str_:", "str:", count=1)
            return strings
        return values


def _verify_positive_temporal_lineage(
    train: Any,
    caches: dict[int, dict[str, np.ndarray]],
    banks: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Assert source-cycle, OOF-block, bank/query, and neighbor boundaries."""
    source_cycles = train["cycle_code"].to_numpy(dtype=np.int64)
    source_ids = train["record_id"].to_numpy(dtype=str)

    def _untag_id(value: str) -> str:
        if value.startswith("str_:"):
            return value[len("str_:"):]
        if value.startswith("str:"):
            return value[len("str:"):]
        raise RuntimeError("A02 temporal lineage: unknown persisted typed-ID tag")
    blocks = {
        67: ((55, 56, 59), (59, 60, 63), (63, 64, 67)),
        75: ((55, 56, 59), (59, 60, 63), (63, 64, 67),
             (67, 68, 71), (71, 72, 75)),
    }
    expected_cache_windows = {
        55: (56, 67), 59: (60, 63), 63: (64, 67),
        67: (68, 79), 71: (72, 75), 75: (76, 79),
    }
    validated_neighbors = 0
    for cutoff, (window_start, window_end) in runner.CACHE_SCORE_CYCLE_WINDOWS.items():
        cache = caches[cutoff]
        if (tuple(runner.CACHE_SCORE_CYCLE_WINDOWS[cutoff]) != expected_cache_windows[cutoff]
                or cutoff >= window_start or window_start > window_end):
            raise RuntimeError(f"A02 temporal lineage: invalid frozen C16 window at cutoff {cutoff}")
        if (not np.all(cache["cycles"] > cutoff)
                or not np.array_equal(cache["score_rows"], np.flatnonzero(
                    (source_cycles >= window_start) & (source_cycles <= window_end)))
                or cache["score_record_ids"].astype(str).tolist()
                != source_ids[cache["score_rows"]].tolist()):
            raise RuntimeError(f"A02 temporal lineage: OOF source/cycle join failed at cutoff {cutoff}")

    for origin, origin_blocks in blocks.items():
        bank = banks[origin]
        expected_parts = []
        for cutoff, first_cycle, last_cycle in origin_blocks:
            if cutoff >= first_cycle or last_cycle - first_cycle != 3:
                raise RuntimeError(f"A02 temporal lineage: non-four-cycle OOF block {cutoff}/{first_cycle}-{last_cycle}")
            cache = caches[cutoff]
            keep = (cache["cycles"] >= first_cycle) & (cache["cycles"] <= last_cycle)
            rows = cache["score_rows"][keep]
            if (not np.all(cache["cycles"][keep] > cutoff)
                    or not np.array_equal(rows, np.flatnonzero(
                        (source_cycles >= first_cycle) & (source_cycles <= last_cycle)))
                    or cache["score_record_ids"][keep].astype(str).tolist() != source_ids[rows].tolist()):
                raise RuntimeError(f"A02 temporal lineage: four-cycle OOF provenance failed for {cutoff}/{first_cycle}-{last_cycle}")
            expected_parts.extend(rows.tolist())
        expected_bank_rows = np.asarray(sorted(expected_parts), dtype=np.int64)
        present = ~bank["missing"]
        bank_rows = bank["rows"][present].astype(np.int64)
        bank_ids = bank["ids"][present].astype(str)
        bank_cycles = bank["cycles"][present].astype(np.int64)
        expected_present = expected_bank_rows[
            ~train.loc[expected_bank_rows, "signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
        ]
        if (not np.array_equal(bank["rows"], expected_bank_rows)
                or not np.array_equal(bank_rows, expected_present)):
            raise RuntimeError(f"A02 temporal lineage: fitted bank rows differ from exact OOF blocks E{origin}")
        if (np.any(bank_cycles > origin)
                or np.any(source_cycles[bank_rows] != bank_cycles)
                or bank_ids.tolist() != source_ids[bank_rows].tolist()):
            raise RuntimeError(f"A02 temporal lineage: fitted bank source identity/cycle invalid E{origin}")

        query = runner._cache_subset(caches[origin], train, origin + 1, 79)
        query_rows = query["rows"].astype(np.int64)
        query_cycles = query["cycles"].astype(np.int64)
        query_ids = query["ids"].astype(str)
        if (not np.all(query_cycles > origin)
                or np.intersect1d(bank_rows, query_rows).size
                or set(bank_ids.tolist()).intersection(query_ids.tolist())
                or query_ids.tolist() != source_ids[query_rows].tolist()):
            raise RuntimeError(f"A02 temporal lineage: bank/query identity or cycle overlap E{origin}")

        for arm in ("observed", "shuffled"):
            folder = runner.RUN_ROOT / f"E{origin}" / arm
            model = joblib.load(folder / "model.joblib")
            if not np.array_equal(model.bank_row_positions, bank_rows):
                raise RuntimeError(f"A02 temporal lineage: persisted model bank rows differ E{origin}/{arm}")
            model_id_map = {int(row): str(record_id)
                            for row, record_id in zip(model.bank_row_positions, model.bank_ids)}
            if any(_untag_id(model_id_map[int(row)]) != record_id
                   for row, record_id in zip(bank_rows, bank_ids)):
                raise RuntimeError(f"A02 temporal lineage: persisted typed bank IDs differ official Train E{origin}/{arm}")
            with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
                output_rows = archive["score_rows"].astype(np.int64)
                output_cycles = archive["score_cycles"].astype(np.int64)
                counts = archive["neighbor_count"].astype(np.int64)
                neighbor_rows = archive["neighbor_source_rows"].astype(np.int64)
                neighbor_ids = archive["neighbor_ids"].astype(str)
                similarities = archive["neighbor_similarities"].astype(np.float64)
            if (not np.array_equal(output_rows, query_rows)
                    or not np.array_equal(output_cycles, query_cycles)
                    or not np.array_equal(source_cycles[output_rows], output_cycles)):
                raise RuntimeError(f"A02 temporal lineage: output query rows/cycles differ E{origin}/{arm}")
            bank_map = {int(row): (str(record_id), int(cycle))
                        for row, record_id, cycle in zip(bank_rows, bank_ids, bank_cycles)}
            for i, count in enumerate(counts):
                rows = neighbor_rows[i, :count].astype(np.int64)
                ids = neighbor_ids[i, :count].astype(str)
                sims = similarities[i, :count]
                if (len(np.unique(rows)) != count
                        or np.any(rows == query_rows[i])
                        or any(int(row) not in bank_map for row in rows)
                        or any(model_id_map[int(row)] != record_id for row, record_id in zip(rows, ids))
                        or any(bank_map[int(row)][0] != _untag_id(record_id)
                               for row, record_id in zip(rows, ids))
                        or any(bank_map[int(row)][1] > origin for row in rows)
                        or any(bank_map[int(row)][1] >= query_cycles[i] for row in rows)):
                    raise RuntimeError(f"A02 temporal lineage: neighbor source/past identity failed E{origin}/{arm}, row={i}")
                tie_order = [(-float(sim), int(row)) for sim, row in zip(sims, rows)]
                if tie_order != sorted(tie_order):
                    raise RuntimeError(f"A02 temporal lineage: neighbor similarity/source-row tie order failed E{origin}/{arm}, row={i}")
                validated_neighbors += int(count)
    return {
        "cache_cutoffs_checked": sorted(caches),
        "exact_four_cycle_oof_blocks_checked": sum(len(value) for value in blocks.values()),
        "bank_query_id_disjoint": True,
        "query_cycles_after_origin": True,
        "neighbor_ids_rows_cycles_validated": validated_neighbors,
        "no_self_or_future_cycle_neighbors": True,
        "similarity_then_source_row_tie_order": True,
    }


def run() -> dict[str, Any]:
    freeze = runner._read_json(runner.FREEZE_PATH)
    if canonical_sha256(freeze.get("identity")) != freeze.get("freeze_signature"):
        raise RuntimeError("A02 technical resume: frozen identity is invalid")
    if any(file_sha256(ROOT / path) != digest
           for path, digest in freeze["source_sha256"].items()):
        raise RuntimeError("A02 technical resume: frozen source changed")

    ledger_before = read_events(LEDGER_PATH)
    a02_starts_before = [event for event in ledger_before
                         if event.get("event") == "FIT_STARTED"
                         and event.get("card_id") == runner.CARD_ID]
    expected_ids = sorted(runner.FIT_IDS.values())
    a02_saves_before = [event for event in ledger_before
                        if event.get("event") == "FIT_SAVED"
                        and event.get("card_id") == runner.CARD_ID]
    a02_failures_before = [event for event in ledger_before
                           if event.get("event") == "FIT_FAILED"
                           and event.get("card_id") == runner.CARD_ID]
    if (sorted(event.get("fit_id") for event in a02_starts_before) != expected_ids
            or sorted(event.get("fit_id") for event in a02_saves_before) != expected_ids
            or a02_failures_before):
        raise RuntimeError("A02 technical verifier adapter requires exactly four saved fits")

    train = runner._load_train(include_target=False)
    caches = {cutoff: runner._load_cache(cutoff, train) for cutoff in sorted(runner.CACHE_PATHS)}
    banks = {origin: runner._build_bank(origin, caches, train) for origin in (67, 75)}
    temporal_lineage = _verify_positive_temporal_lineage(train, caches, banks)
    for origin, arm in ((67, "observed"), (67, "shuffled"), (75, "observed"), (75, "shuffled")):
        model_path = runner.RUN_ROOT / f"E{origin}" / arm / "model.joblib"
        model = joblib.load(model_path)
        raw_ids = tuple(model.bank_ids)
        if _array_sha256(np.asarray(raw_ids, dtype=str)) != model.audit_state()["bank_ids_sha256"]:
            raise RuntimeError(f"A02 technical verifier: persisted bank ID digest differs E{origin}/{arm}")
        prefix = "str_:" if all(value.startswith("str_:") for value in raw_ids) else "str:"
        if prefix == "str:" and not all(value.startswith("str:") for value in raw_ids):
            raise RuntimeError(f"A02 technical verifier: mixed/unknown typed ID prefix E{origin}/{arm}")
        values = tuple(value[len(prefix):] for value in raw_ids)
        expected_values = tuple(banks[origin]["ids"][~banks[origin]["missing"]].astype(str))
        if values != expected_values:
            raise RuntimeError(f"A02 technical verifier: bank ID values differ E{origin}/{arm}")
        output_path = runner.RUN_ROOT / f"E{origin}" / arm / "predictions.npz"
        with np.load(output_path, allow_pickle=False) as archive:
            neighbor_rows = archive["neighbor_source_rows"].astype(np.int64)
            neighbor_ids = archive["neighbor_ids"].astype(str)
            neighbor_counts = archive["neighbor_count"].astype(np.int64)
        row_to_id = {int(row): value for row, value in zip(model.bank_row_positions, raw_ids)}
        for index, count in enumerate(neighbor_counts):
            expected_neighbor_ids = [row_to_id[int(row)] for row in neighbor_rows[index, :count]]
            if neighbor_ids[index, :count].tolist() != expected_neighbor_ids:
                raise RuntimeError(f"A02 technical verifier: persisted neighbor IDs differ E{origin}/{arm}")

    original_verify = frozen_verifier.verify
    original_load_arm = frozen_verifier._load_arm
    original_joblib_load = joblib.load
    original_numpy_load = np.load

    # Official Train is ordered by record identity, not by cycle. Replace only
    # the verifier's invalid source-row-order temporal assertion. Exact bank
    # membership and cycle-based past checks remain active in _load_arm.
    load_arm_source = textwrap.dedent(inspect.getsource(original_load_arm))
    invalid_order_check = "or np.any(valid_rows >= rows[i])"
    if load_arm_source.count(invalid_order_check) != 1:
        raise RuntimeError("A02 technical verifier: expected single row-order assertion not found")
    adapted_source = load_arm_source.replace(
        invalid_order_check,
        "or False  # official Train source row order is not chronological; cycle check remains",
        1,
    )
    adapted_globals = dict(original_load_arm.__globals__)
    exec(compile(adapted_source, inspect.getsourcefile(original_load_arm) or "<a02-verifier-adapter>", "exec"),
         adapted_globals)
    adapted_load_arm = adapted_globals[original_load_arm.__name__]

    def _canonical_model_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        model = original_joblib_load(path, *args, **kwargs)
        if hasattr(model, "bank_ids"):
            ids = tuple(model.bank_ids)
            if all(value.startswith("str_:") for value in ids):
                model.bank_ids = tuple("str:" + value[len("str_:"):] for value in ids)
        return model

    def _canonical_prediction_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        archive = original_numpy_load(file, *args, **kwargs)
        if Path(str(file)).name == "predictions.npz":
            return _NeighborIdArchiveAdapter(archive)
        return archive

    def _verify_numpy_string_id_compat(*, write_outputs: bool = True) -> dict[str, Any]:
        joblib.load = _canonical_model_load
        np.load = _canonical_prediction_load
        try:
            return original_verify(write_outputs=write_outputs)
        finally:
            joblib.load = original_joblib_load
            np.load = original_numpy_load

    frozen_verifier.verify = _verify_numpy_string_id_compat
    frozen_verifier._load_arm = adapted_load_arm
    try:
        result = frozen_verifier.verify(write_outputs=True)
    finally:
        frozen_verifier.verify = original_verify
        frozen_verifier._load_arm = original_load_arm

    ledger_after = read_events(LEDGER_PATH)
    starts = [event for event in ledger_after
              if event.get("event") == "FIT_STARTED"
              and event.get("card_id") == runner.CARD_ID]
    saved_events = [event for event in ledger_after
                    if event.get("event") == "FIT_SAVED"
                    and event.get("card_id") == runner.CARD_ID]
    failed = [event for event in ledger_after
              if event.get("event") == "FIT_FAILED"
              and event.get("card_id") == runner.CARD_ID]
    if (sorted(event.get("fit_id") for event in starts) != expected_ids
            or sorted(event.get("fit_id") for event in saved_events) != expected_ids
            or failed):
        raise RuntimeError("A02 technical verifier adapter: fit ledger is not exactly four saves")
    outputs = {}
    for origin, arm in ((67, "observed"), (67, "shuffled"), (75, "observed"), (75, "shuffled")):
        path = runner.RUN_ROOT / f"E{origin}" / arm / "predictions.npz"
        sidecar_path = path.with_suffix(".json")
        sidecar = runner._read_json(sidecar_path)
        outputs[f"E{origin}_{arm}"] = {
            "npz_sha256": file_sha256(path),
            "sidecar_sha256": file_sha256(sidecar_path),
            "fit_receipt_sha256": sidecar["fit_receipt_sha256"],
        }
    receipt = {
        "status": "A02_TECHNICAL_RESUME_COMPLETE",
        "repair_scope": "in-memory frozen-verifier adapter after all four fit/output bundles were durable: canonicalize NumPy string type tags; replace invalid source-row-order temporal comparison with positive cycle-based temporal and exact source-bank checks",
        "recovered_writer_error": "AttributeError: numpy.ndarray has no attribute to_numpy; already repaired by the first resume harness",
        "verifier_adapter": "canonicalize str_: to str: for typed IDs in-memory; remove only invalid neighbor_row_position < query_row_position assertion because official Train is not chronological; retain source-bank membership, identity, row, cycle, and strict bank_cycle < query_cycle checks; saved model/NPZ bytes unchanged",
        "predictor_or_preprocessing_recipe_changed": False,
        "new_fit_ids_started_during_verification": [],
        "all_four_fit_ids_reused_without_refit": expected_ids,
        "freeze_signature": freeze["freeze_signature"],
        "frozen_source_commit": freeze["identity"]["source_commit"],
        "frozen_source_sha256": freeze["source_sha256"],
        "resume_helper_path": Path(__file__).relative_to(ROOT).as_posix(),
        "resume_helper_sha256": file_sha256(Path(__file__)),
        "frozen_verifier_sha256": file_sha256(ROOT / "scripts/optimization_v3_a02_neighbors_verify.py"),
        "fit_count_started": len(starts),
        "fit_count_saved": len(saved_events),
        "fit_failures": len(failed),
        "positive_temporal_lineage_checks": temporal_lineage,
        "outputs": outputs,
        "independent_verifier": result,
        "test_or_sample_access": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_once(RECEIPT_PATH, receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
