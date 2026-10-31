"""Verify and publish R18 receipts left by a gate-triggered pilot stop.

This recovery path never refits a predictor or alters the frozen recipe. It
revalidates the committed R18 freeze, source/input hashes and existing receipt
bundles, then independently reconstructs the report. A failed promotion gate
is a valid RETIRE outcome and does not invalidate receipt integrity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts import optimization_v3_batch1_r18 as r18
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    exclusive_training_slot,
    file_sha256,
    json_sha256,
    load_queue,
    read_bundle,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_CUTOFFS = (55, 67, 75)
RECEIPT_FILES = tuple(
    path
    for cutoff in RECEIPT_CUTOFFS
    for path in (
        r18.RUN_DIR / f"R18_fit{cutoff}.npz",
        r18.RUN_DIR / f"R18_fit{cutoff}.json",
    )
)


def _same(left: Any, right: Any, *, label: str) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise BatchIntegrityError(f"R18 recovery receipt differs for {label}")


def _independent_results(
    *, results: Mapping[str, Any], receipt_blocks: Mapping[int, Mapping[str, Any]],
    cycles: np.ndarray, frame: Any, target_by_row: Mapping[int, float],
) -> dict[str, Any]:
    specs = {
        "W4": (receipt_blocks[67], 68, 71),
        "W6": (receipt_blocks[75], 76, 79),
        "H12_L": (receipt_blocks[67], 68, 79),
        "H12_E": (receipt_blocks[55], 56, 67),
    }
    views = {
        name: r18._slice_block(block, cycles, low, high)
        for name, (block, low, high) in specs.items()
    }
    views["primary"] = r18._join_slices(views["W4"], views["W6"])
    metrics: dict[str, Any] = {}
    for name, view in views.items():
        rows = np.asarray(view["rows"], dtype=np.int64)
        y = np.asarray([target_by_row[int(row)] for row in rows], dtype=np.float64)
        baseline_mae = r18._independent_mae(y, view["baseline"])
        candidate_mae = r18._independent_mae(y, view["candidate"])
        stored = results["views"][name]
        if (len(rows) != int(stored["rows"])
                or not np.isclose(baseline_mae, stored["baseline_mae"], atol=1e-12, rtol=0.0)
                or not np.isclose(candidate_mae, stored["candidate_mae"], atol=1e-12, rtol=0.0)):
            raise BatchIntegrityError(f"R18 recovery independently reconstructed MAE differs for {name}")
        metrics[name] = {
            "rows": int(len(rows)), "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae, "match": True,
        }

    table_checks: dict[str, Any] = {}
    guard_failures: list[dict[str, Any]] = []
    for name in ("W4", "W6", "H12_L"):
        view = views[name]
        rows = np.asarray(view["rows"], dtype=np.int64)
        y = np.asarray([target_by_row[int(row)] for row in rows], dtype=np.float64)
        group_view = {
            "rows": rows, "cycle": np.asarray(view["cycle"], dtype=np.int64),
            "cutoff": np.asarray(view["cutoff"], dtype=np.int64),
            "baseline": np.asarray(view["baseline"], dtype=np.float64),
            "candidate": np.asarray(view["candidate"], dtype=np.float64),
        }
        table_checks[name] = r18._verify_subgroup_table(
            view_name=name, view=group_view, frame=frame, target=y,
            stored=results["subgroups"][name],
            support_min=int(r18.EXPECTED_REGRESSION_GATES["supported_group_n_min"]),
        )
        labels = r18._group_labels_for_view(view=group_view, frame=frame, target=y)
        for field, group in (
            ("target_tail", "low_le5"), ("target_tail", "high_ge95"),
            ("signal_stream", "missing"),
        ):
            selected = labels[field] == group
            n = int(selected.sum())
            if n >= int(r18.EXPECTED_REGRESSION_GATES["supported_group_n_min"]):
                harm = r18._independent_mae(y[selected], group_view["candidate"][selected]) - r18._independent_mae(
                    y[selected], group_view["baseline"][selected]
                )
                if harm > r18.EXPECTED_REGRESSION_GATES["supported_tail_or_missing_MAE_harm_max"]:
                    guard_failures.append({
                        "view": name, "field": field, "group": group,
                        "n": n, "candidate_minus_baseline_mae": float(harm),
                    })

    primary = metrics["primary"]
    h12 = metrics["H12_L"]
    w4 = metrics["W4"]
    w6 = metrics["W6"]
    gate = r18.EXPECTED_REGRESSION_GATES
    gates = {
        "late_primary_gain_min_0_03": (
            primary["baseline_mae"] - primary["candidate_mae"] >= gate["pilot_primary_MAE_gain_min"]
        ),
        "H12_L_harm_max_0_05": (
            h12["candidate_mae"] - h12["baseline_mae"] <= gate["pilot_H12_MAE_harm_max"]
        ),
        "W4_individual_harm_max_0_5": (
            w4["candidate_mae"] - w4["baseline_mae"] <= gate["individual_pilot_view_MAE_harm_max"]
        ),
        "W6_individual_harm_max_0_5": (
            w6["candidate_mae"] - w6["baseline_mae"] <= gate["individual_pilot_view_MAE_harm_max"]
        ),
        "supported_tail_or_missing_harm_max_0_25": not guard_failures,
    }
    if gates != dict(results["gates"]):
        raise BatchIntegrityError("R18 recovery gate reconstruction differs from frozen result logic")
    if bool(all(gates.values())) != bool(results["all_pilot_gates_pass"]):
        raise BatchIntegrityError("R18 recovery overall decision differs from reconstructed gates")
    expected_verdict = "ADVANCE" if all(gates.values()) else "RETIRE"
    if results["verdict"] != expected_verdict:
        raise BatchIntegrityError("R18 recovery verdict differs from reconstructed gate outcome")
    reported_failures = results["subgroup_gate_failures"]
    if len(reported_failures) != len(guard_failures):
        raise BatchIntegrityError("R18 recovery subgroup gate failure count differs")
    for actual, expected in zip(reported_failures, guard_failures):
        if any(actual[key] != expected[key] for key in ("view", "field", "group", "n")) or not np.isclose(
            float(actual["candidate_minus_baseline_mae"]),
            float(expected["candidate_minus_baseline_mae"]), atol=1e-12, rtol=0.0,
        ):
            raise BatchIntegrityError("R18 recovery subgroup gate failure differs")
    return {
        "status": "PASS",
        "receipt_checks": "all three committed-freeze-bound bundles re-read and reconstructed",
        "view_metric_checks": metrics,
        "subgroup_table_checks": table_checks,
        "independent_supported_tail_or_missing_guard_failures": guard_failures,
        "independent_gate_reconstruction": gates,
        "gate_failure_is_candidate_outcome_not_receipt_integrity_failure": True,
        "decision": expected_verdict,
        "test_or_sample_used": False,
    }


def recover() -> dict[str, Any]:
    existing_reports = [
        r18.REPORT_DIR / "results.json", r18.REPORT_DIR / "verification.json",
        r18.REPORT_DIR / "summary.md", r18.PUBLICATION_PATH,
    ]
    if any(path.exists() for path in existing_reports):
        raise BatchIntegrityError("R18 recovery refuses to overwrite an existing report/publication")
    missing = [str(path) for path in RECEIPT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R18 partial receipt set is incomplete: {missing}")

    queue = load_queue(r18.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == r18.CARD_ID)
    preflight, state = r18._prepare_preflight(card, queue)
    identity = r18._freeze_identity(card, preflight)
    freeze_document = json.loads(r18.FREEZE_PATH.read_text(encoding="utf-8"))
    signature = r18._verify_freeze_document(freeze_document, identity)
    freeze_sha256 = file_sha256(r18.FREEZE_PATH)
    r18._assert_current_frozen_sources(identity, state)

    blocks = r18._frozen_candidate_blocks(state)
    target_by_row = r18._read_score_targets_after_freeze(
        state["train_path"], state["cycles"],
        identity["input_sha256"]["official_regression_train"],
    )
    evaluation = r18._evaluate_frozen_predictions(
        blocks=blocks, cycles=state["cycles"], frame=state["frame"],
        target_by_row=target_by_row,
    )
    receipt_blocks: dict[int, dict[str, Any]] = {}
    receipt_checks = []
    for cutoff in RECEIPT_CUTOFFS:
        block = blocks[cutoff]
        rows = np.asarray(block["rows"], dtype=np.int64)
        path = r18.RUN_DIR / f"R18_fit{cutoff}.npz"
        arrays, sidecar = read_bundle(path, {
            "freeze_signature": signature, "card_id": r18.CARD_ID,
            "cutoff": cutoff, "score_rows": rows.tolist(),
        })
        metadata = sidecar["metadata"]
        if (metadata.get("fit_predictor_count") != 0
                or metadata.get("preprocessing_fit_count") != 0
                or metadata.get("test_data_used") is not False
                or metadata.get("score_labels_passed_to_adapter") is not False):
            raise BatchIntegrityError(f"R18 recovery fit/test metadata differs at cutoff {cutoff}")
        for key, expected in (
            ("score_rows", rows),
            ("score_cycles", block["cycle"]),
            ("score_record_ids", block["record_id"]),
            ("baseline_prediction", block["baseline_prediction"]),
            ("prediction", block["prediction"]),
            ("delta", np.full(len(rows), float(block["state"]["delta"]))),
        ):
            if key not in arrays:
                raise BatchIntegrityError(f"R18 recovery receipt lacks {key} at cutoff {cutoff}")
            _same(arrays[key], expected, label=f"cutoff{cutoff}:{key}")
        state_meta = metadata.get("state", {})
        if not np.isclose(float(state_meta.get("delta", np.nan)),
                          float(block["state"]["delta"]), atol=0.0, rtol=0.0):
            raise BatchIntegrityError(f"R18 recovery receipt state offset differs at cutoff {cutoff}")
        history = block["history"]
        if history is None:
            if cutoff != 55 or float(state_meta.get("delta", np.nan)) != 0.0:
                raise BatchIntegrityError("R18 recovery expected an early exact-champion fallback")
        else:
            for key, expected in (
                ("history_rows", history["rows"]),
                ("history_cycles", history["cycle"]),
                ("history_source_cutoff", history["source_cutoff"]),
                ("history_prediction", history["prediction"]),
            ):
                if key not in arrays:
                    raise BatchIntegrityError(f"R18 recovery receipt lacks {key} at cutoff {cutoff}")
                _same(arrays[key], expected, label=f"cutoff{cutoff}:{key}")
            independently_estimated_delta = float(np.clip(
                0.5 * np.median(history["target"] - history["prediction"]), -2.0, 2.0
            ))
            if not np.isclose(independently_estimated_delta, float(block["state"]["delta"]), atol=0.0, rtol=0.0):
                raise BatchIntegrityError(f"R18 recovery history delta differs at cutoff {cutoff}")
        receipt_blocks[cutoff] = {
            "cutoff": cutoff, "rows": np.asarray(arrays["score_rows"], dtype=np.int64),
            "cycle": np.asarray(arrays["score_cycles"], dtype=np.int64),
            "baseline_prediction": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
            "prediction": np.asarray(arrays["prediction"], dtype=np.float64),
            "record_id": np.asarray(arrays["score_record_ids"], dtype=str),
        }
        receipt_checks.append({
            "cutoff": cutoff, "bundle_readback": True, "row_id_alignment": True,
            "history_or_fallback_reconstruction": True, "predictor_fit_count": 0,
        })

    verification = _independent_results(
        results={**evaluation, "views": evaluation["views"]},
        receipt_blocks=receipt_blocks, cycles=state["cycles"], frame=state["frame"],
        target_by_row=target_by_row,
    )
    verification["receipt_checks"] = receipt_checks
    calibration = {str(cutoff): dict(blocks[cutoff]["state"]) for cutoff in (67, 75)}
    results = {
        "schema_version": 1, "card_id": r18.CARD_ID, "task": "regression",
        "status": "PILOT_VERIFIED", "freeze_signature": signature,
        "freeze_sha256": freeze_sha256,
        "input_sha256": dict(identity["input_sha256"]),
        "source_sha256": dict(identity["source_sha256"]),
        "fit_predictor_count": 0, "preprocessing_fit_count": 0,
        "calibration": calibration, "views": evaluation["views"],
        "subgroups": evaluation["subgroups"],
        "subgroup_gate_failures": evaluation["subgroup_gate_failures"],
        "support_rule": evaluation["support_rule"], "gates": evaluation["gates"],
        "all_pilot_gates_pass": evaluation["all_pilot_gates_pass"],
        "verdict": evaluation["verdict"], "promotion_limit": evaluation["promotion_limit"],
        "fallback_coverage": {
            "H12_E": {"rows": int(len(blocks[55]["rows"])),
                       "exact_champion_rows": int(len(blocks[55]["rows"])),
                       "coverage": 1.0, "gate_applies": False},
            "late_supported_views": {
                "W4_rows": int(evaluation["views"]["W4"]["rows"]),
                "W6_rows": int(evaluation["views"]["W6"]["rows"]),
                "H12_L_rows": int(evaluation["views"]["H12_L"]["rows"]),
                "history_fallback_rows": 0,
            },
        },
        "test_or_sample_used": False,
        "future_score_labels_passed_to_adapter": False,
        "outcome_reuse_warning": "Temporal development outcomes are reused; this late-only pilot is not an untouched holdout.",
        "publication_note": "Recovered from complete frozen receipts after a valid subgroup-gate failure interrupted the original publisher.",
    }
    results["independent_verification"] = "PASS"
    if results["verdict"] != verification["decision"]:
        raise BatchIntegrityError("R18 recovery verification and result verdict differ")

    report_paths = [r18.REPORT_DIR / "results.json", r18.REPORT_DIR / "verification.json",
                    r18.REPORT_DIR / "summary.md", r18.PUBLICATION_PATH]
    if any(path.exists() for path in report_paths):
        raise BatchIntegrityError("R18 report outputs appeared during recovery; refusing overwrite")
    write_json_exclusive(report_paths[0], results)
    write_json_exclusive(report_paths[1], {
        "card_id": r18.CARD_ID, "status": "PASS", "freeze_signature": signature,
        "freeze_sha256": freeze_sha256, "recovery_script_sha256": file_sha256(Path(__file__)),
        "checks": verification,
    })
    r18._write_text_exclusive(report_paths[2], r18._render_summary(results, signature))
    manifest_files = [*RECEIPT_FILES, *report_paths[:3]]
    publication = {
        "schema_version": 1, "card_id": r18.CARD_ID, "status": "COMPLETE",
        "freeze_signature": signature,
        "recovery_script_sha256": file_sha256(Path(__file__)),
        "files": {str(path.relative_to(ROOT)): file_sha256(path)
                  for path in sorted(manifest_files, key=str)},
        "results_sha256": file_sha256(report_paths[0]),
        "verification_sha256": file_sha256(report_paths[1]),
        "summary_sha256": file_sha256(report_paths[2]),
    }
    write_json_exclusive(report_paths[3], publication)
    return {"status": "PASS", "decision": results["verdict"],
            "report": str(report_paths[0].relative_to(ROOT)),
            "verification": str(report_paths[1].relative_to(ROOT)),
            "publication": str(report_paths[3].relative_to(ROOT)),
            "failed_gates": [key for key, passed in results["gates"].items() if not passed]}


def main() -> int:
    lock = r18.RUN_DIR.parent / ".training.lock"
    with exclusive_training_slot(lock):
        result = recover()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
