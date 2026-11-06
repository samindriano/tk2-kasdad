from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cls_pos_ef_context_v1 as pilot
from scripts import recover_cls_pos_ef_context_v1_score as recovery
from scripts import verify_cls_pos_ef_context_v1 as metric_verifier
from scripts import verify_cls_pos_ef_context_v1_score_recovery as recovery_verifier


def _synthetic_bundle() -> dict[str, np.ndarray]:
    cycles = np.concatenate([
        np.full(n, cycle, dtype=np.int64)
        for cycle, n in zip(range(68, 76), [1552, 1552, 1552, 1552, 1551, 1551, 1551, 1551])
    ] + [
        np.full(n, cycle, dtype=np.int64)
        for cycle, n in zip(range(76, 80), [2123, 2123, 2123, 2122])
    ])
    n = len(cycles)
    rows = np.arange(n, dtype=np.int64)
    ids = np.asarray([f"synthetic-{i:05d}" for i in rows], dtype=str)
    y = (((rows * 37 + cycles * 11) % 11) < 4).astype(np.int8)
    y[:30] = 0
    rng = np.random.default_rng(9106)
    p_champion = np.clip(np.where(y == 1, 0.73, 0.27) + rng.normal(0, 0.17, n), 0.005, 0.995)
    present = rows % 13 != 0
    p_real = np.clip(p_champion + rng.normal(0, 0.07, n), 0.005, 0.995)
    p_shuffled = np.clip(np.where(y == 1, 0.57, 0.43) + rng.normal(0, 0.25, n), 0.005, 0.995)
    p_real[~present] = p_champion[~present]
    p_shuffled[~present] = p_champion[~present]
    label_champion = (p_champion >= 0.5).astype(np.int8)
    label_real = (p_real >= 0.5).astype(np.int8)
    label_shuffled = (p_shuffled >= 0.5).astype(np.int8)
    label_real[~present] = label_champion[~present]
    label_shuffled[~present] = label_champion[~present]
    text_group = np.where(rows < 30, "rare_unsupported", np.where(rows % 2 == 0, "low", "high"))
    return {
        "origin": np.asarray([67], dtype=np.int64),
        "row_positions": rows,
        "record_ids": ids,
        "cycles": cycles,
        "present": present.astype(bool),
        "ef1": (rows % 2).astype(np.int8),
        "text_load_group": text_group.astype(str),
        "p_a02": p_champion.astype(np.float64),
        "p_pos_original": p_champion.astype(np.float64),
        "p_champion": p_champion.astype(np.float64),
        "p_real": p_real.astype(np.float64),
        "p_shuffled": p_shuffled.astype(np.float64),
        "label_champion": label_champion,
        "label_real": label_real,
        "label_shuffled": label_shuffled,
        "delta_real": np.where(present, rng.normal(0, 0.1, n), 0.0).astype(np.float64),
        "delta_shuffled": np.where(present, rng.normal(0, 0.1, n), 0.0).astype(np.float64),
    }


def _write_bundle(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    allow = ("origin", "row_positions", "record_ids", "cycles", "present", "ef1", "text_load_group",
             "p_a02", "p_pos_original", "p_champion", "p_real", "p_shuffled",
             "label_champion", "label_real", "label_shuffled", "delta_real", "delta_shuffled")
    np.savez_compressed(path, **{key: arrays[key] for key in allow})


def test_synthetic_recovery_and_independent_metric_path(tmp_path: Path, monkeypatch) -> None:
    frozen_score = (ROOT / "scripts/cls_pos_ef_context_v1.py").read_text(encoding="utf-8")
    score_tree, score_patch = recovery.patch_score_ast(frozen_score)
    assert score_patch["prior_guard_change_count"] == 1
    assert score_patch["subgroup_key_move_count"] == 1
    assert score_patch["no_metric_formula_or_gate_changes"] is True

    report_root = tmp_path / "reports/CLS_POS_EF_CONTEXT_V1"
    run_root = tmp_path / "experiments/CLS_POS_EF_CONTEXT_V1"
    score_path = report_root / "SCORE.json"
    metric_path = report_root / "METRIC_VERIFICATION.json"
    prelabel_path = report_root / "PRELABEL_VERIFICATION.json"
    ready_path = report_root / "PREDICTIONS_READY.json"
    ledger_path = run_root / "fit_ledger.jsonl"

    e67 = _synthetic_bundle()
    e75_mask = (e67["cycles"] >= 76) & (e67["cycles"] <= 79)
    e75 = {key: value[e75_mask] if value.ndim and len(value) == len(e67["row_positions"]) else value.copy()
           for key, value in e67.items()}
    e75["origin"] = np.asarray([75], dtype=np.int64)
    e67_path = run_root / "E67/predictions_E67.npz"
    e75_path = run_root / "E75/predictions_E75.npz"
    _write_bundle(e67_path, e67)
    _write_bundle(e75_path, e75)
    assert len(e67["row_positions"]) == 20903 and len(e75["row_positions"]) == 8491

    ready = {"predictions": {
        "E67": {"path": e67_path.relative_to(tmp_path).as_posix()},
        "E75": {"path": e75_path.relative_to(tmp_path).as_posix()},
    }}
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")
    prediction_hashes = {"E67": pilot.sha256_file(e67_path), "E75": pilot.sha256_file(e75_path)}
    prelabel = {
        "status": "PASS_TARGET_FREE_PRELABEL",
        "freeze_signature": "synthetic-freeze",
        "predictions_ready_sha256": pilot.sha256_file(ready_path),
        "prediction_output_sha256s": prediction_hashes,
        "predictions": {
            "E67": {"missing_fallback_exact": True},
            "E75": {"missing_fallback_exact": True},
        },
        "query_labels_read": False,
        "test_or_sample_access": False,
    }
    prelabel_path.write_text(json.dumps(prelabel, sort_keys=True), encoding="utf-8")

    fit_ids = [f"CLS_POS_EF_CONTEXT_V1_E{o}_{arm}" for o in (67, 75) for arm in ("REAL", "SHUFFLED")]
    freeze = {"candidate_id": "CLS_POS_EF_CONTEXT_V1", "freeze_signature": "synthetic-freeze",
              "source_commit": "synthetic-source", "input_sha256": "synthetic-input", "fit_ids": fit_ids}
    (report_root / "FREEZE.json").write_text(json.dumps(freeze), encoding="utf-8")
    opened = {
        "candidate_id": freeze["candidate_id"], "event": "QUERY_LABELS_OPENED",
        "freeze_signature": freeze["freeze_signature"], "prelabel_sha256": pilot.sha256_file(prelabel_path),
        "purpose": "single Train temporal metric evaluation",
        "query_ids_sha256": pilot.sha256_array(e67["record_ids"]),
        "query_rows_sha256": pilot.sha256_array(e67["row_positions"]), "rows": 20903,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_events = ([{"event": "FIT_STARTED", "fit_id": fit_id} for fit_id in fit_ids]
                     + [{"event": "FIT_SAVED", "fit_id": fit_id} for fit_id in fit_ids]
                     + [opened])
    ledger_path.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n"
                                      for e in ledger_events), encoding="utf-8")

    score_namespace = {
        "__name__": "synthetic_cls_pos_ef_score_recovery",
        "__file__": str(ROOT / "scripts/cls_pos_ef_context_v1.py"),
        "SCORE_RECOVERY_AUTHORIZED": True,
    }
    exec(compile(score_tree, str(ROOT / "scripts/cls_pos_ef_context_v1.py"), "exec"), score_namespace)
    score_namespace.update({
        "AUDIT_ROOT": tmp_path,
        "REPORT_ROOT": report_root,
        "RUN_ROOT": run_root,
        "SCORE_PATH": score_path,
        "METRIC_VERIFY_PATH": metric_path,
        "PRELABEL_PATH": prelabel_path,
        "READY_PATH": ready_path,
        "verify_freeze": lambda _root: freeze,
        "verify_committed_prelabel": lambda: prelabel,
    })
    target_reads: list[tuple[int, ...]] = []
    ymap = dict(zip(e67["row_positions"].tolist(),
                    (((e67["row_positions"] * 37 + e67["cycles"] * 11) % 11) < 4).astype(np.int8).tolist()))
    for row in range(30):
        ymap[row] = 0

    def synthetic_targets(path: Path, rows, ids):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        assert len(rows) == len(ids) == 20903
        target_reads.append(tuple(int(row) for row in rows))
        return np.asarray([ymap[int(row)] for row in rows], dtype=np.int8)

    score_namespace["_selected_targets"] = synthetic_targets
    result = score_namespace["score_metrics"](tmp_path)
    assert len(target_reads) == 1, "Synthetic score must retrieve targets once and only once"
    assert result["status"] == "SCORED_DIAGNOSTIC_ONLY"
    assert result["test_or_sample_access"] is False
    assert result["metric_verification"]["independent_check_count"] > 0
    assert all(check["status"] == "PASS" for check in result["metric_verification"]["checks"])
    assert {"E67", "E75", "UNIQUE_PRIMARY"} == set(result["views"])
    assert {"W4", "W5", "W6", "H12_E67"}.issubset(result["views"]["E67"])
    assert "W6" in result["views"]["E75"]
    assert "per_cycle" in result["views"]["UNIQUE_PRIMARY"]
    subgroup_items = []
    for view in result["views"].values():
        for item in view.values():
            if isinstance(item, dict) and "supported_subgroups" in item:
                subgroup_items.extend(item["supported_subgroups"])
    assert any(not group["support"]["supported"] for group in subgroup_items)
    assert any(group["support"]["supported"] for group in subgroup_items)
    ids = [check["metric_id"] for check in result["metric_verification"]["checks"]]
    assert any("fit_prefix_text_load=" in metric_id for metric_id in ids)
    assert len([event for event in pilot.events(ledger_path) if event.get("event") == "FIT_STARTED"]) == 4
    assert len([event for event in pilot.events(ledger_path) if event.get("event") == "QUERY_LABELS_OPENED"]) == 2

    # Independent, label-free verification accepts the authorized two-event lineage only.
    root_authority = {
        "candidate_id": freeze["candidate_id"],
        "freeze_signature": freeze["freeze_signature"],
        "query_rows_sha256": opened["query_rows_sha256"],
        "query_ids_sha256": opened["query_ids_sha256"],
        "failed_original_query_event": opened,
        "committed_prelabel_sha256": opened["prelabel_sha256"],
    }
    start_event = {
        "event": "SCORE_RECOVERY_STARTED", "recovery_id": recovery.RECOVERY_ID,
        "candidate_id": freeze["candidate_id"], "freeze_signature": freeze["freeze_signature"],
        "same_rows_sha256": opened["query_rows_sha256"], "same_ids_sha256": opened["query_ids_sha256"],
        "prior_query_open_count": 1, "attempt_number": 2, "retry_allowed": False,
        "score_formula_changed": False, "model_or_predictions_changed": False,
    }
    pilot.append_event(ledger_path, start_event)
    pilot.append_event(ledger_path, {
        "event": "SCORE_RECOVERY_SAVED", "recovery_id": recovery.RECOVERY_ID,
        "candidate_id": freeze["candidate_id"], "freeze_signature": freeze["freeze_signature"],
        "score_sha256": pilot.sha256_file(score_path), "metric_verification_sha256": pilot.sha256_file(metric_path),
        "query_label_open_count": 2,
    })
    for attr, value in {
        "FREEZE_PATH": report_root / "FREEZE.json", "SCORE_PATH": score_path,
        "METRIC_VERIFY_PATH": metric_path, "PRELABEL_PATH": prelabel_path, "RUN_ROOT": run_root,
        "verify_freeze": lambda _root: freeze, "verify_committed_prelabel": lambda: prelabel,
    }.items():
        monkeypatch.setattr(pilot, attr, value)
    ledger_report = recovery_verifier.validate_recovery_ledger(pilot, root_authority)
    assert ledger_report["status"] == "PASS_RECOVERY_LEDGER"
    assert ledger_report["total_label_open_events"] == 2

    frozen_verifier = (ROOT / "scripts/verify_cls_pos_ef_context_v1.py").read_text(encoding="utf-8")
    verifier_tree, verifier_patch = recovery_verifier.patch_verify_score_ast(frozen_verifier)
    assert verifier_patch["ledger_guard_replacement_count"] == 1
    verifier_namespace = {"__name__": "synthetic_cls_pos_ef_metric_verify", "RECOVERY_LEDGER_VALIDATED": True}
    exec(compile(verifier_tree, str(ROOT / "scripts/verify_cls_pos_ef_context_v1.py"), "exec"), verifier_namespace)
    verified = verifier_namespace["verify_score"]()
    assert verified["status"] == "PASS_INDEPENDENT_METRIC_AND_GATE_VERIFICATION"
    assert verified["gate"]["status"] == "PASS_GATE_RECONSTRUCTION"
    assert verified["query_label_reads"] == 2
