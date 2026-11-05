from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cls_simple_present_threshold_final_v1 as final
import verify_cls_simple_present_threshold_final_v1 as verifier


def make_fixture(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    train_path = base / "Train.csv"
    rows = []
    oof_ids: dict[int, list[str]] = {67: [], 75: []}
    oof_cycles: dict[int, list[int]] = {67: [], 75: []}
    oof_missing: dict[int, list[bool]] = {67: [], 75: []}
    oof_pa: dict[int, list[float]] = {67: [], 75: []}
    train_ids_by_cycle: dict[int, list[str]] = {}
    idx = 0
    for cycle in range(68, 80):
        train_ids_by_cycle[cycle] = []
        for j in range(50):
            record_id = f"tr-{idx:04d}"
            target = j % 2
            missing = j % 5 == 0
            pa = (0.56 if target else 0.44) + ((j % 7) - 3) * 0.003
            if missing:
                pa = 0.53230976129477 if j % 10 == 0 else (0.50 if target else 0.49)
            rows.append({"record_id": record_id, "cycle_code": cycle,
                         "signal_stream": "" if missing else "tok1 tok2", "artifact_signal": target})
            train_ids_by_cycle[cycle].append(record_id)
            oof_ids[67].append(record_id)
            oof_cycles[67].append(cycle)
            oof_missing[67].append(missing)
            oof_pa[67].append(pa)
            idx += 1
    pd.DataFrame(rows).to_csv(train_path, index=False)
    for cycle in range(76, 80):
        start = (cycle - 68) * 50
        for record_id in train_ids_by_cycle[cycle]:
            offset = len(oof_ids[75]) - (cycle - 76) * 50
            row = rows[start + (offset % 50)]
            oof_ids[75].append(record_id)
            oof_cycles[75].append(cycle)
            oof_missing[75].append(row["signal_stream"] == "")
            oof_pa[75].append(oof_pa[67][start + (offset % 50)])
    oof_path = base / "oof.npz"
    np.savez_compressed(
        oof_path,
        E67_ids=np.asarray(oof_ids[67], dtype="U"), E67_cycles=np.asarray(oof_cycles[67], dtype=np.int16),
        E67_missing=np.asarray(oof_missing[67], dtype=bool), E67_pA=np.asarray(oof_pa[67], dtype=np.float64),
        E75_ids=np.asarray(oof_ids[75], dtype="U"), E75_cycles=np.asarray(oof_cycles[75], dtype=np.int16),
        E75_missing=np.asarray(oof_missing[75], dtype=bool), E75_pA=np.asarray(oof_pa[75], dtype=np.float64),
        E67_y=np.full(len(oof_ids[67]), 255, dtype=np.uint8), E75_y=np.full(len(oof_ids[75]), 255, dtype=np.uint8),
    )
    test_ids = [f"te-{i:03d}" for i in range(12)]
    test_path = base / "Test.csv"
    pd.DataFrame({"record_id": test_ids,
                  "signal_stream": ["" if i in {1, 5, 9} else "tok" for i in range(12)]}).to_csv(test_path, index=False)
    sample_path = base / "SampleSubmission.csv"
    pd.DataFrame({"record_id": test_ids, "artifact_signal": np.zeros(len(test_ids), dtype=int)}).to_csv(sample_path, index=False)
    e79_path = base / "e79.npz"
    np.savez_compressed(e79_path, record_ids=np.asarray(test_ids, dtype="U"),
                        missing_mask=np.asarray([i in {1, 5, 9} for i in range(12)], dtype=bool),
                        a02_probability=np.asarray([0.45 + i * 0.025 for i in range(12)], dtype=np.float64))
    return {"train": train_path, "oof": oof_path, "e79": e79_path, "test": test_path, "sample": sample_path,
            "freeze": base / "PRE_FIT_FREEZE.json", "run": base / "run", "csv": base / "final.csv"}


def invoke(fixture: dict[str, Path], command: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(ROOT / "scripts/cls_simple_present_threshold_final_v1.py"),
           "--train-csv", str(fixture["train"]), "--oof-cache", str(fixture["oof"]),
           "--e79-cache", str(fixture["e79"]), "--test-csv", str(fixture["test"]),
           "--sample-csv", str(fixture["sample"]), "--freeze", str(fixture["freeze"]),
           "--run-dir", str(fixture["run"]), "--output-csv", str(fixture["csv"]),
           "--synthetic-fixture", command]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def test_exact_oracle_respects_tie_break_and_fixed_missing_route():
    y = np.asarray([0] * 220 + [1] * 220, dtype=np.int8)
    p = np.full(440, 0.50, dtype=np.float64)
    missing = np.zeros(440, dtype=bool)
    missing[[0, 1, 220, 221]] = True
    p[[0, 1, 220, 221]] = [0.51, 0.52, 0.54, 0.55]
    result = verifier.independent_threshold_oracle(p, missing, y)
    assert result["threshold"] == 0.50
    pred_a = np.where(missing, p >= 0.53230976129477, p >= result["threshold"])
    pred_b = np.where(missing, p >= 0.53230976129477, p >= 0.49)
    assert np.array_equal(pred_a[missing], pred_b[missing])


def test_pool_has_exact_unique_training_ids_and_cycle_partition(tmp_path):
    fixture = make_fixture(tmp_path)
    pool = final._load_oof_pool(fixture["oof"], fixture["train"], synthetic=True)
    assert len(pool["ids"]) == 600
    assert len(set(pool["ids"].tolist())) == 600
    assert set(pool["cycles"].tolist()) == set(range(68, 80))
    labels = final._read_selected_labels(fixture["train"], np.asarray(final._load_train_ids(fixture["train"])), pool["ids"])
    assert len(labels) == 600 and set(labels.tolist()) == {0, 1}


def test_cli_synthetic_freeze_fit_apply_independent_replay_and_recovery(tmp_path):
    fixture = make_fixture(tmp_path)
    for command in ("freeze", "fit"):
        result = invoke(fixture, command)
        assert result.returncode == 0, result.stderr
    fit_path = fixture["run"] / "fits/FINAL.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    assert fit["fit_rows"] == 600
    assert fit["predictor_fits"] == 0
    ledger_path = fixture["run"] / "LEDGER.jsonl"
    events = [json.loads(row) for row in ledger_path.read_text(encoding="utf-8").splitlines()]
    # Simulate an interruption after the receipt was durably written but before
    # the ledger's FIT_FROZEN completion event.
    events = [row for row in events if row["event"] != "FIT_FROZEN"]
    ledger_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in events), encoding="utf-8")
    recovered = invoke(fixture, "fit")
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout)["threshold_hex"] == fit["threshold_hex"]
    assert "FIT_RECOVERED" in ledger_path.read_text(encoding="utf-8")
    applied = invoke(fixture, "apply")
    assert applied.returncode == 0, applied.stderr
    report = verifier.verify_final_candidate(freeze_path=fixture["freeze"], train_path=fixture["train"],
                                             oof_path=fixture["oof"], e79_path=fixture["e79"],
                                             test_path=fixture["test"], sample_path=fixture["sample"],
                                             run_dir=fixture["run"], output_path=fixture["csv"], synthetic=True)
    assert report["status"] == "PASS"
    assert report["test_rows"] == 12
    assert report["test_target_read"] is False


def test_ambiguous_fit_start_blocks_retry(tmp_path):
    fixture = make_fixture(tmp_path)
    assert invoke(fixture, "freeze").returncode == 0
    events = [{"event_id": "FINAL_A02_PAST_OOF_THRESHOLD:FIT_STARTED", "event": "FIT_STARTED",
               "slot": "FINAL_A02_PAST_OOF_THRESHOLD", "fit_ids_sha256": "bad"}]
    fixture["run"].mkdir(parents=True, exist_ok=True)
    (fixture["run"] / "LEDGER.jsonl").write_text(json.dumps(events[0]) + "\n", encoding="utf-8")
    result = invoke(fixture, "fit")
    assert result.returncode != 0
    assert "retry is blocked" in result.stderr


def test_independent_verifier_rejects_tampered_csv(tmp_path):
    fixture = make_fixture(tmp_path)
    assert invoke(fixture, "freeze").returncode == 0
    assert invoke(fixture, "fit").returncode == 0
    assert invoke(fixture, "apply").returncode == 0
    frame = pd.read_csv(fixture["csv"])
    frame.loc[0, "artifact_signal"] = 1 - int(frame.loc[0, "artifact_signal"])
    frame.to_csv(fixture["csv"], index=False)
    with pytest.raises((ValueError, RuntimeError)):
        verifier.verify_final_candidate(freeze_path=fixture["freeze"], train_path=fixture["train"],
                                        oof_path=fixture["oof"], e79_path=fixture["e79"],
                                        test_path=fixture["test"], sample_path=fixture["sample"],
                                        run_dir=fixture["run"], output_path=fixture["csv"], synthetic=True)
