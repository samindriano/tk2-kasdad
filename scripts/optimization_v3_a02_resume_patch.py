"""Resume the frozen A02 run after the ndarray/Series prediction-writer error.

This compatibility wrapper does not change the frozen predictor, preprocessing,
features, residual assignments, metric, or fit accounting.  It adapts only the
already-saved query ID array to the pandas Series shape expected by the frozen
writer, then calls the original runner.  A repair receipt binds this helper to
the existing A02 freeze and ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_a02_neighbors as runner  # noqa: E402
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
    expected_saved_fit = runner.FIT_IDS[(67, "observed")]
    if [event.get("fit_id") for event in a02_starts_before] != [expected_saved_fit]:
        raise RuntimeError("A02 technical resume: unexpected task fit already started")
    saved = [event for event in ledger_before
             if event.get("event") == "FIT_SAVED" and event.get("fit_id") == expected_saved_fit]
    if len(saved) != 1:
        raise RuntimeError("A02 technical resume: E67 observed is not exactly durably saved")

    runner_pre = runner._write_prediction_once

    def _series_compat_write(origin: int, arm: str, model: Any,
                             query: dict[str, np.ndarray], train: pd.DataFrame,
                             fit_receipt: dict[str, Any]) -> Path:
        ids = np.asarray(query["ids"])
        if ids.ndim != 1 or not len(ids):
            raise RuntimeError("A02 technical resume: query IDs are not a nonempty vector")
        compatible_query = dict(query)
        compatible_query["ids"] = pd.Series(ids.astype(str), dtype=str)
        return runner_pre(origin, arm, model, compatible_query, train, fit_receipt)

    runner._write_prediction_once = _series_compat_write
    result = runner.run()

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
    expected_ids = sorted(runner.FIT_IDS.values())
    if (sorted(event.get("fit_id") for event in starts) != expected_ids
            or sorted(event.get("fit_id") for event in saved_events) != expected_ids
            or failed):
        raise RuntimeError("A02 technical resume: fit ledger is not exactly four saves")
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
        "repair_scope": "query ID array adapted to Series for the frozen prediction writer only",
        "recovered_writer_error": "AttributeError: numpy.ndarray has no attribute to_numpy",
        "predictor_or_preprocessing_recipe_changed": False,
        "new_fit_ids_started_during_resume": sorted(
            event.get("fit_id") for event in starts if event.get("fit_id") != expected_saved_fit),
        "prior_saved_fit_reused_without_refit": expected_saved_fit,
        "freeze_signature": freeze["freeze_signature"],
        "frozen_source_commit": freeze["identity"]["source_commit"],
        "frozen_source_sha256": freeze["source_sha256"],
        "resume_helper_path": Path(__file__).relative_to(ROOT).as_posix(),
        "resume_helper_sha256": file_sha256(Path(__file__)),
        "fit_count_started": len(starts),
        "fit_count_saved": len(saved_events),
        "fit_failures": len(failed),
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
