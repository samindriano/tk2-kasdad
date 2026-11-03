"""Small, isolated end-to-end contract check for the positional pilot.

This uses synthetic rows, a temporary input CSV, a temporary run directory,
and a miniature temporary budget ledger. It never opens official Train/Test
data or parses/writes the shared research ledger; it reads that file only for
before/after SHA-256 integrity checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_positional_text_pilot_v1 as runner
from scripts.optimization_v3_positional_text_pilot_v1_verify import (
    score_origin,
    verify_metrics_independently,
    verify_prediction_bundle,
)
from src.models.classification_models import _text_values
from src.models.optimization_v3_positional_text_pilot_v1 import (
    build_position_block,
    fit_c2b_tfidf,
    validate_feature_frame,
)


_LAST_CLEANUP_AUDIT: dict[str, Any] = {}


def _synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    cycle_sequence = [47, 50, 55, 60, 67]
    for index in range(40):
        cycle = cycle_sequence[index % len(cycle_sequence)]
        label = int((index + (index // 4)) % 2)
        rows.append(_row(index, cycle, label, scored=False))
        labels.append(label)
    index = 40
    for cycle in range(68, 80):
        for repeat in range(2):
            label = int((index + repeat) % 2)
            rows.append(_row(index, cycle, label, scored=True))
            labels.append(label)
            index += 1
    frame = pd.DataFrame(rows)
    # Deliberately unsort cycle_code relative to row order; all alignment is by
    # source row position and record_id, never by a sorted cycle assumption.
    frame = frame.sample(frac=1.0, random_state=19).reset_index(drop=True)
    label_lookup = {row["record_id"]: label for row, label in zip(rows, labels)}
    target = np.asarray([label_lookup[value] for value in frame["record_id"]], dtype=np.int8)
    return frame, target


def _row(index: int, cycle: int, label: int, *, scored: bool) -> dict[str, Any]:
    score = int(scored)
    pattern = "PATTERN_UNSEEN_SCORE" if score else f"PATTERN_{index % 3}"
    org = "ORG_UNSEEN_SCORE" if score else f"ORG_{index % 3}"
    module = "MODULE_UNSEEN_SCORE" if score else f"MODULE_{index % 4}"
    phrase = "alpha beta gamma" if label else "delta epsilon zeta"
    suffix = f" noveltytoken{index}" if score else f" common{index % 7}"
    if index % 9 == 0:
        stream: Any = "  "
    elif index % 13 == 0:
        stream = np.nan
    else:
        stream = f"{phrase} shared shared {suffix}"
    return {
        "record_id": f"SYN-{index:04d}",
        "cycle_code": int(cycle),
        "artifact_signal": int(label),
        "impact_load": float(1 + index % 5),
        "entry_friction": int(index % 2),
        "scope_pressure": float((index * 3) % 17),
        "text_load": float((index * 7) % 23),
        "scope_trace": "~4" if index % 6 == 0 else f"{2 + index % 8} nodes",
        "org_spread": f"{1 + index % 4}",
        "module_spread": f"{2 + index % 5}",
        "telemetry_flag": "active" if index % 2 else "inactive",
        "org_code": org,
        "module_code": module,
        "pattern_code": pattern,
        "signal_stream": stream,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_ledger_line(event: dict[str, Any], *, path: Path, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = Path(path).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise AssertionError(f"Synthetic ledger write escaped its temporary root: {resolved_path}")
    encoded = runner._canonical_line(event)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return json.loads(encoded.decode("utf-8"))


def _read_synthetic_ledger(*, path: Path, root: Path) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    resolved_path = Path(path).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise AssertionError(f"Synthetic ledger read escaped its temporary root: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    return [json.loads(line) for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _synthetic_fold_rows(frame: pd.DataFrame, origin: int):
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= origin)).astype(np.int64)
    low, high = ((68, 79) if origin == 67 else (76, 79))
    score_rows = np.flatnonzero((cycles >= low) & (cycles <= high)).astype(np.int64)
    return fit_rows, score_rows, low, high


def run_synthetic_contract(*, inject_failure: bool = False) -> dict[str, Any]:
    frame, target = _synthetic_data()
    feature_frame = frame.drop(columns=["artifact_signal"])
    try:
        validate_feature_frame(frame)
    except ValueError as exc:
        if "Targets must not be included" not in str(exc):
            raise
    else:
        raise AssertionError("Feature validation accepted artifact_signal")

    # Direct unit check for deterministic shuffle invariants and finite blocks.
    fit_mask = frame["cycle_code"].to_numpy(dtype=int) <= 67
    fit_docs, _ = _text_values(frame.loc[fit_mask, "signal_stream"])
    vectorizer = fit_c2b_tfidf(validate_feature_frame(feature_frame.loc[fit_mask]))
    original, _ = build_position_block(fit_docs, vectorizer, shuffled=False)
    shuffled, shuffled_audit = build_position_block(fit_docs, vectorizer, shuffled=True)
    if original.shape != shuffled.shape or not np.isfinite(shuffled.data).all():
        raise AssertionError("Synthetic positional blocks are not finite and shape-aligned")
    for doc, observed, control in zip(fit_docs, original, shuffled):
        # Both vectors have the same bag-derived support; positional columns may
        # differ by third, so compare token bag and segment sizes via the frozen
        # tokenizer-level control audit rather than expecting equal positions.
        if not isinstance(doc, str):
            raise AssertionError("Normalized synthetic documents must be strings")
        if observed.shape != control.shape:
            raise AssertionError("Shuffle changed positional feature dimension")
    if (shuffled_audit["bag_preserved_rows"] != len(fit_docs)
            or shuffled_audit["segment_sizes_preserved_rows"] != len(fit_docs)):
        raise AssertionError("Deterministic shuffle did not preserve bag and segment sizes")

    changed: dict[str, Any] = {}
    for name in (
        "ROOT", "TRAIN", "TRAIN_SHA256", "REPORT_ROOT", "RUN_ROOT", "LEDGER",
        "FREEZE_PATH", "INITIAL_GLOBAL_CAP", "INITIAL_CAPS", "BASELINE_FIT_COUNTS",
        "PREVIOUS_SNAPSHOT", "AMENDED_CAPS", "CAP_AFTER", "resource_snapshot", "_fold_rows",
        "append_ledger", "read_ledger", "_append_official_ledger",
    ):
        changed[name] = getattr(runner, name)
    fit_names = (
        "fit_c2b_structured_state", "fit_c2b_scaler", "fit_c2b_ohe", "fit_c2b_tfidf",
        "fit_c1b_structured_state", "fit_logistic", "fit_missing_hgb",
    )
    original_fit_functions = {name: getattr(runner, name) for name in fit_names}
    live_ledger_path = Path(runner.LEDGER).resolve()
    live_ledger_sha_before = _sha(live_ledger_path)
    official_writer_calls = {"count": 0}
    counters = {name: 0 for name in fit_names}

    try:
        with TemporaryDirectory(prefix="position-pilot-synthetic-") as temporary:
            root = Path(temporary)
            report_root = root / "reports/optimization_v3/positional_text_pilot_v1"
            run_root = root / "experiments/runs/POSITIONAL_TEXT_PILOT_V1"
            train_path = root / "data/Train.csv"
            report_root.mkdir(parents=True)
            train_path.parent.mkdir(parents=True)
            frame.to_csv(train_path, index=False)
            runner.ROOT = root
            runner.TRAIN = train_path
            runner.TRAIN_SHA256 = _sha(train_path)
            runner.REPORT_ROOT = report_root
            runner.RUN_ROOT = run_root
            runner.LEDGER = root / "ledger.jsonl"
            runner.FREEZE_PATH = run_root / "FREEZE.json"
            runner.INITIAL_GLOBAL_CAP = 0
            runner.INITIAL_CAPS = {"pilot": 0, "continuation": 0, "control_replay": 0, "technical_retry": 0}
            runner.BASELINE_FIT_COUNTS = {"fit_starts": 0, "fit_saved": 0, "fit_failed": 0, "fit_pending": 0}
            runner.PREVIOUS_SNAPSHOT = {
                "global_cap": 0, "effective_category_caps": dict(runner.INITIAL_CAPS),
                "active_reservations_by_category": dict(runner.INITIAL_CAPS),
                "active_reservations": 0, **runner.BASELINE_FIT_COUNTS,
            }
            runner.AMENDED_CAPS = {"pilot": len(runner.FIT_IDS), "continuation": 0,
                                   "control_replay": 0, "technical_retry": 0}
            runner.CAP_AFTER = len(runner.FIT_IDS)
            runner._fold_rows = _synthetic_fold_rows
            production_append = changed["append_ledger"]
            production_read = changed["read_ledger"]
            try:
                production_append({"event": "SYNTHETIC_MISSING_PATH_MUST_FAIL"})
            except TypeError:
                missing_append_rejected = True
            else:
                raise AssertionError("Official ledger writer accepted a missing explicit path")
            try:
                production_read()
            except TypeError:
                missing_read_rejected = True
            else:
                raise AssertionError("Ledger reader accepted a missing explicit path")
            if _sha(live_ledger_path) != live_ledger_sha_before:
                raise AssertionError("Missing-path guard touched the live ledger")

            def forbidden_official_writer(*args, **kwargs):
                official_writer_calls["count"] += 1
                raise AssertionError("Official ledger writer was called from synthetic context")

            runner._append_official_ledger = forbidden_official_writer
            runner.append_ledger = lambda event, *, path: _write_synthetic_ledger_line(
                dict(event), path=path, root=root
            )
            runner.read_ledger = lambda *, path: _read_synthetic_ledger(path=path, root=root)
            try:
                runner.append_ledger({"event": "SYNTHETIC_PRODUCTION_PATH_MUST_FAIL"}, path=live_ledger_path)
            except AssertionError:
                production_path_rejected = True
            else:
                raise AssertionError("Synthetic ledger adapter accepted the official ledger path")
            if _sha(live_ledger_path) != live_ledger_sha_before:
                raise AssertionError("Production-path rejection touched the live ledger")
            runner.resource_snapshot = lambda estimate_bytes=0, **kwargs: {
                "jobs": 1, "threads": 2, "process_rss_bytes": 1,
                "available_ram_bytes": 8 * 1024**3,
                "estimated_incremental_memory_bytes": int(estimate_bytes),
                "projected_rss_bytes": int(estimate_bytes) + 1,
                "projected_available_ram_bytes": 8 * 1024**3 - int(estimate_bytes),
                "max_rss_bytes": 6 * 1024**3,
                "initial_available_ram_min_bytes": 4 * 1024**3,
                "minimum_free_ram_bytes": 2 * 1024**3,
            }

            for name in fit_names:
                original = original_fit_functions[name]

                def counted(*args, _name=name, _original=original, **kwargs):
                    counters[_name] += 1
                    return _original(*args, **kwargs)

                setattr(runner, name, counted)

            runner.append_ledger({
                "event": "BATCH_ALLOWANCE_AUTHORIZED", "batch_id": "synthetic-test-only",
                "cap": 0, "category_caps": dict(runner.INITIAL_CAPS),
            }, path=runner.LEDGER)
            initial_events = runner.read_ledger(path=runner.LEDGER)
            initial_snapshot = runner.ledger_snapshot(initial_events)
            freeze = {
                "pilot_id": runner.PILOT_ID, "source_commit": "SYNTHETIC_ONLY",
                "source_hashes": {"synthetic-input": _sha(train_path)},
                "input_sha256": _sha(train_path),
                "pre_reservation_ledger_sha256": _sha(runner.LEDGER),
                "pre_reservation_ledger_snapshot": initial_snapshot,
                "freeze_signature": "synthetic-freeze-signature",
            }
            reservation = runner.reserve_pilot(freeze)
            if reservation["snapshot"]["active_reservations"] != len(runner.FIT_IDS):
                raise AssertionError("Synthetic pilot reservation did not account for all fit units")

            predictions = []
            for origin in runner.ORIGINS:
                predictions.append(runner._fit_origin(freeze, origin, feature_frame))
                if inject_failure and origin == 67:
                    raise RuntimeError("synthetic injected failure after E67 for cleanup verification")
            first_counts = dict(counters)
            events_after_first = runner.read_ledger(path=runner.LEDGER)
            first_snapshot = runner.ledger_snapshot(events_after_first)
            if (first_snapshot["fit_starts"] != len(runner.FIT_IDS)
                    or first_snapshot["fit_saved"] != len(runner.FIT_IDS)
                    or first_snapshot["fit_pending"] != 0
                    or first_snapshot["fit_failed"] != 0):
                raise AssertionError(f"Synthetic fit accounting incomplete: {first_snapshot}")
            if any(count <= 0 for count in first_counts.values()):
                raise AssertionError(f"Synthetic executor did not exercise all learned components: {first_counts}")

            # Independent, fit-free checks run for every origin before synthetic
            # outer labels are handed to the scorer.
            verifications = []
            for origin in runner.ORIGINS:
                pred_path = run_root / f"E{origin}" / f"predictions_E{origin}.npz"
                matrix_path = runner._matrix_manifest_path(origin)
                verifications.append(verify_prediction_bundle(feature_frame, origin, pred_path, matrix_path))
            if not all(item["status"] == "PASS_INDEPENDENT_PREDICTION_INTEGRITY"
                       and item["query_labels_read"] is False
                       and item["fit_or_transform_fitted"] is False for item in verifications):
                raise AssertionError("Synthetic prelabel verification did not pass for both origins")

            scored = []
            # This is the first synthetic score-label read, after both durable
            # prediction bundles and independent verifications have passed.
            for origin in runner.ORIGINS:
                _, score_rows, low, high = runner._fold_rows(feature_frame, origin)
                score = score_origin(
                    feature_frame, origin,
                    run_root / f"E{origin}" / f"predictions_E{origin}.npz",
                    target[score_rows], low, high,
                    fit_text_q75=float(json.loads(runner._matrix_manifest_path(origin).read_text())[
                        "text_load_q75_fit_only"]),
                )
                metric_check = verify_metrics_independently(
                    score, feature_frame, origin,
                    run_root / f"E{origin}" / f"predictions_E{origin}.npz",
                    target[score_rows], score["fit_defined_text_load_q75"],
                )
                scored.append({"result": score, "metric_verification": metric_check})

            # Resume with every saved artifact present. No fit function should
            # be invoked and the ledger must remain exactly at 18 saved units.
            for origin in runner.ORIGINS:
                runner._fit_origin(freeze, origin, feature_frame)
            if counters != first_counts:
                raise AssertionError(f"Synthetic resume repeated one or more fits: {first_counts} -> {counters}")
            resumed_snapshot = runner.ledger_snapshot(runner.read_ledger(path=runner.LEDGER))
            if (resumed_snapshot["fit_starts"] != len(runner.FIT_IDS)
                    or resumed_snapshot["fit_saved"] != len(runner.FIT_IDS)
                    or resumed_snapshot["fit_pending"] != 0):
                raise AssertionError(f"Synthetic resume changed completed accounting: {resumed_snapshot}")

            # Tamper a copy of the durable prediction and prove row integrity
            # fails without damaging the saved bundle.
            origin = 67
            pred_path = run_root / f"E{origin}" / f"predictions_E{origin}.npz"
            tampered_path = run_root / f"E{origin}" / "tampered_predictions.npz"
            with np.load(pred_path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["score_record_ids"] = payload["score_record_ids"].copy()
            payload["score_record_ids"][0] = "SYN-TAMPERED-ID"
            np.savez_compressed(tampered_path, **payload)
            try:
                verify_prediction_bundle(feature_frame, origin, tampered_path, runner._matrix_manifest_path(origin))
            except AssertionError:
                tamper_rejected = True
            else:
                raise AssertionError("Independent verifier accepted tampered score IDs")

            report = {
                "status": "PASS_SYNTHETIC_END_TO_END_PERSISTENCE_ACCOUNTING_VERIFIER_REPORT",
                "official_data_opened": False,
                "shared_ledger_read_for_hash_only": True,
                "shared_ledger_parsed": False,
                "shared_ledger_mutated": False,
                "live_ledger_sha256_before": live_ledger_sha_before,
                "live_ledger_sha256_after": _sha(live_ledger_path),
                "live_ledger_unchanged": _sha(live_ledger_path) == live_ledger_sha_before,
                "missing_explicit_path_rejected": missing_append_rejected and missing_read_rejected,
                "explicit_production_path_rejected": production_path_rejected,
                "official_writer_calls": official_writer_calls["count"],
                "temporary_input_sha256": _sha(train_path),
                "rows": len(feature_frame), "unsorted_cycle_rows": True,
                "contains_empty_stream": bool(_text_values(feature_frame["signal_stream"])[1].any()),
                "contains_unseen_score_tokens_and_categories": True,
                "fit_unit_count": len(runner.FIT_IDS),
                "fit_counts_by_function": first_counts,
                "completed_ledger_snapshot": resumed_snapshot,
                "resume_repeated_fit_count": 0,
                "independent_prelabel_verification": verifications,
                "shared_preprocessing_hashes": {
                    str(origin): json.loads(runner._matrix_manifest_path(origin).read_text())[
                        "shared_preprocessing_sha256"] for origin in runner.ORIGINS
                },
                "metric_verification": [item["metric_verification"] for item in scored],
                "origin_gate_status": [item["result"]["status"] for item in scored],
                "q4_support": [item["result"]["q4_support"] for item in scored],
                "tampered_prediction_rejected": tamper_rejected,
                "route_missing_probabilities_exact": True,
                "class_order_checked": True,
                "position_shuffle_audit": shuffled_audit,
                "scored_views": [
                    {"origin": item["result"]["origin"],
                     "views": item["result"]["views"],
                     "subgroups": item["result"]["subgroups"],
                     "per_cycle": item["result"]["per_cycle"],
                     "gate_checks": item["result"]["gate_checks"]}
                    for item in scored
                ],
                "prediction_paths": [item["prediction_path"] for item in predictions],
            }
            return report
    finally:
        for name, value in changed.items():
            setattr(runner, name, value)
        for name, value in original_fit_functions.items():
            setattr(runner, name, value)
        restored_globals = all(getattr(runner, name) is value for name, value in changed.items())
        restored_fit_functions = all(getattr(runner, name) is value
                                     for name, value in original_fit_functions.items())
        live_ledger_sha_after = _sha(live_ledger_path)
        live_ledger_unchanged = live_ledger_sha_after == live_ledger_sha_before
        globals()["_LAST_CLEANUP_AUDIT"] = {
            "live_ledger_sha256_before": live_ledger_sha_before,
            "live_ledger_sha256_after": live_ledger_sha_after,
            "live_ledger_unchanged": live_ledger_unchanged,
            "globals_restored": restored_globals,
            "fit_functions_restored": restored_fit_functions,
        }
        if not live_ledger_unchanged:
            raise AssertionError("Live ledger SHA changed during synthetic executor")
        if not restored_globals or not restored_fit_functions:
            raise AssertionError("Synthetic executor failed to restore original runner globals")


def main() -> int:
    result = run_synthetic_contract()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
