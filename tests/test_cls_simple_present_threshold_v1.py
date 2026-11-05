from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import cls_simple_present_threshold_v1 as producer
from scripts import verify_cls_simple_present_threshold_v1 as verifier


def _balanced_fixture():
    p = np.asarray([0.45, 0.45, 0.49, 0.50, 0.50, 0.55, 0.60, 0.20] * 100, dtype=float)
    missing = np.asarray([False, False, False, False, False, False, False, True] * 100)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1] * 100, dtype=np.int8)
    return p, missing, y


def test_exact_optimizer_matches_independent_breakpoint_oracle_and_realizes_counts():
    p, missing, y = _balanced_fixture()
    actual = producer.exact_threshold_fit(p, missing, y)
    expected = verifier.independent_threshold_oracle(p, missing, y)
    assert actual["threshold"] == expected["threshold"]
    assert actual["fit_macro_f1_exact"] == expected["macro_f1_exact"]
    assert actual["threshold_hex"] == actual["threshold"].hex()
    assert producer.LOW <= actual["threshold"] <= producer.HIGH


def test_threshold_tie_prefers_centre_when_one_decision_state_spans_interval():
    p = np.full(500, 0.2, dtype=float)
    missing = np.zeros(500, dtype=bool)
    y = np.asarray([0, 1] * 250, dtype=np.int8)
    actual = producer.exact_threshold_fit(p, missing, y)
    assert actual["threshold"] == 0.5
    assert actual["decision_state_interval"] == [0.45, 0.55]


def test_missing_decisions_are_invariant_under_present_threshold():
    p, missing, y = _balanced_fixture()
    result = producer.exact_threshold_fit(p, missing, y)
    pred = np.where(missing, p >= producer.MISSING_THRESHOLD, p >= result["threshold"])
    fixed = np.where(missing, p >= producer.MISSING_THRESHOLD, p >= producer.CENTRE)
    assert np.array_equal(pred[missing], fixed[missing])


def test_present_class_support_fails_closed():
    p = np.full(220, 0.5)
    missing = np.zeros(220, dtype=bool)
    y = np.asarray([0] * 219 + [1], dtype=np.int8)
    with pytest.raises(RuntimeError, match="Unsupported calibration population"):
        producer.exact_threshold_fit(p, missing, y)


def test_selected_label_reader_does_not_parse_unselected_poison_labels(tmp_path):
    train = tmp_path / "Train.csv"
    with train.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows([["a", "0"], ["b", "1"], ["c", "POISON"], ["d", "NOT_BINARY"]])
    y = producer._read_selected_labels(train, np.asarray(["a", "b", "c", "d"]), np.asarray(["b", "a"]))
    assert y.tolist() == [1, 0]
    with pytest.raises(ValueError, match="absent"):
        producer._read_selected_labels(train, np.asarray(["a", "b", "c", "d"]), np.asarray(["missing"]))


def test_ledger_is_append_only_and_duplicate_event_ids_are_rejected(tmp_path):
    ledger = tmp_path / "LEDGER.jsonl"
    event = {"event_id": "D1:FIT_STARTED", "event": "FIT_STARTED", "stage": "D1"}
    producer.append_ledger(ledger, event)
    assert producer.read_ledger(ledger) == [event]
    with ledger.open("ab") as stream:
        stream.write((json.dumps(event) + "\n").encode())
    with pytest.raises(ValueError, match="duplicate event IDs"):
        producer.read_ledger(ledger)


def test_ambiguous_fit_started_blocks_resume_before_read_or_refit(tmp_path, monkeypatch):
    ready_path = tmp_path / "SOURCE_READY.json"
    ready_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    release_path = tmp_path / "ROOT_RELEASE.json"
    release_path.write_text(json.dumps({
        "status": "ROOT_RELEASED_FOR_D1_D2",
        "source_ready_sha256": producer.sha256_file(ready_path),
        "allowed_stages": ["D1", "D2"],
    }), encoding="utf-8")
    ledger_path = tmp_path / "LEDGER.jsonl"
    producer.append_ledger(ledger_path, {"event_id": "D1:FIT_STARTED", "event": "FIT_STARTED", "stage": "D1"})
    monkeypatch.setattr(producer, "READY", ready_path)
    monkeypatch.setattr(producer, "LEDGER", ledger_path)
    monkeypatch.setattr(producer, "_verify_runtime_lineage", lambda *_: None)
    monkeypatch.setattr(producer, "_stage_vectors", lambda *_: pytest.fail("ambiguous run must stop before vectors/labels"))
    with pytest.raises(RuntimeError, match="ambiguous FIT_STARTED"):
        producer.fit_stage("D1", tmp_path / "unused.csv", tmp_path / "unused.npz", release_path)


def _write_synthetic_source_inputs(tmp_path: Path):
    import pandas as pd

    train_path = tmp_path / "Train.csv"
    rows = []
    for cycle in range(47, 80):
        rows.append({
            "record_id": f"rec_{cycle}", "cycle_code": cycle,
            "signal_stream": "" if cycle in {68, 72, 76} else "token token2",
            "entry_friction": cycle % 2, "text_load": float(cycle),
            # Deliberately invalid: source-only preflight must never parse this target column.
            "artifact_signal": "TARGET_MUST_NOT_BE_PARSED",
        })
    pd.DataFrame(rows).to_csv(train_path, index=False)

    arrays = {}
    for origin, cycles in ((67, np.arange(68, 80)), (75, np.arange(76, 80))):
        ids = np.asarray([f"rec_{cycle}" for cycle in cycles])
        missing = np.asarray([cycle in {68, 72, 76} for cycle in cycles], dtype=bool)
        values = np.asarray(cycles, dtype=float)
        prefix = np.arange(47, origin + 1, dtype=float)
        edges = np.quantile(prefix, [0.25, 0.5, 0.75], method="linear")
        groups = np.searchsorted(edges, values, side="right")
        arrays[f"E{origin}_ids"] = ids
        arrays[f"E{origin}_cycles"] = cycles.astype(np.int16)
        arrays[f"E{origin}_missing"] = missing
        arrays[f"E{origin}_pA"] = np.linspace(0.2, 0.8, len(cycles))
        arrays[f"E{origin}_pP"] = np.linspace(0.25, 0.75, len(cycles))
        arrays[f"E{origin}_group_availability"] = np.where(missing, "missing", "present")
        arrays[f"E{origin}_group_entry_friction"] = np.asarray([str(c % 2) for c in cycles])
        arrays[f"E{origin}_group_text_load"] = np.asarray(["Q" + str(int(x) + 1) for x in groups])
    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(cache_path, **arrays)
    # Corrupt y payloads prove the implementation does not deserialize those members.
    with zipfile.ZipFile(cache_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("E67_y.npy", b"not a numpy array: target member must remain unopened")
        archive.writestr("E75_y.npy", b"not a numpy array: target member must remain unopened")
    return train_path, cache_path


def test_real_cli_freeze_preflight_and_independent_verifier_are_target_free(tmp_path, monkeypatch, capsys):
    train, cache = _write_synthetic_source_inputs(tmp_path)
    out = tmp_path / "reports"
    run = tmp_path / "run"
    source = producer
    checks = {
        "OUT": out, "RUN": run, "FREEZE": out / "SOURCE_FREEZE.json",
        "PREFLIGHT": out / "SOURCE_PREFLIGHT.json", "READY": out / "SOURCE_READY.json",
        "LEDGER": run / "LEDGER.jsonl", "FIT_DIR": run / "fits", "SCORE_DIR": run / "scores",
        "CACHE": cache, "EXPECTED_ROWS": {67: 12, 75: 4}, "EXPECTED_SCORE_ROWS": 8,
        "OFFICIAL_TRAIN_ROWS": 33,
    }
    verifier_checks = {
        "OUT": out, "FREEZE": out / "SOURCE_FREEZE.json", "PREFLIGHT": out / "SOURCE_PREFLIGHT.json",
        "CACHE": cache, "EXPECTED_ROWS": {67: 12, 75: 4}, "EXPECTED_SCORE_ROWS": 8,
        "OFFICIAL_TRAIN_ROWS": 33, "CACHE_SHA256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    for name, value in checks.items():
        monkeypatch.setattr(source, name, value)
    for name, value in verifier_checks.items():
        monkeypatch.setattr(verifier, name, value)

    assert source.main(["--train-csv", str(train), "--cache", str(cache), "freeze"]) == 0
    capsys.readouterr()
    assert source.main(["--train-csv", str(train), "--cache", str(cache), "preflight"]) == 0
    capsys.readouterr()

    ready = json.loads((out / "SOURCE_READY.json").read_text(encoding="utf-8"))
    receipt = json.loads((out / "SOURCE_PREFLIGHT.json").read_text(encoding="utf-8"))
    assert ready["status"] == "PASS"
    assert receipt["official_target_column_values_read"] is False
    assert receipt["oof_target_members_opened"] is False
    assert verifier.verify_source_preflight()["target_values_read"] is False


def test_verifier_self_check():
    result = verifier.run_synthetic_self_check()
    assert result["status"] == "PASS"
    assert result["target_accessed"] is False
