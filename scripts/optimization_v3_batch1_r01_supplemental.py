"""Reproduce the R01 entry_friction subgroup diagnostic after the frozen pilot.

This is descriptive QA only: it does not fit, score, tune, change the frozen
R01 gate, or read Test. It consumes only official Train and the R01 fit bundles.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_batch1_r01 import _view_bundle_data  # noqa: E402
from src.models.optimization_v3_batch1 import (  # noqa: E402
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    load_queue,
    write_json_exclusive,
)
from src.utils.paths import task_files  # noqa: E402


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R01"
PILOT_PATH = REPORT_DIR / "pilot.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/R01/freeze.json"
OUTPUT_PATH = REPORT_DIR / "entry_friction_diagnostics.json"
EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"


def _group_metrics(y: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    if not len(y):
        raise BatchIntegrityError("R01 entry_friction diagnostic found an empty group")
    candidate_mae = float(np.mean(np.abs(prediction - y)))
    baseline_mae = float(np.mean(np.abs(baseline - y)))
    return {
        "n": int(len(y)),
        "supported_n_ge_100": bool(len(y) >= 100),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "candidate_minus_baseline_mae": float(candidate_mae - baseline_mae),
    }


def compute() -> dict[str, Any]:
    queue = load_queue(QUEUE_PATH)
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R01 supplemental diagnostic is only valid during V3 execution")
    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    train_hash = file_sha256(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError("Regression Train hash differs from the frozen R01 source")

    frame = pd.read_csv(train_path)
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if pilot.get("freeze_signature") != freeze.get("signature"):
        raise BatchIntegrityError("R01 pilot and freeze signatures differ")
    views = evaluation_views_for_card(queue, "R01")
    output: dict[str, Any] = {
        "status": "DESCRIPTIVE_POSTFIT_QA_NOT_A_GATE",
        "card_id": "R01",
        "official_train_sha256": train_hash,
        "freeze_signature": freeze["signature"],
        "group_definition": "entry_friction is observed Train metadata, values 0 and 1; not a model feature interaction test",
        "minimum_support_for_description": 100,
        "does_not_change_predeclared_gate": True,
        "test_data_read": False,
        "views": {},
    }
    for view_name in ("primary", "H12"):
        rows, y, prediction, baseline = _view_bundle_data(
            frame, target, views, pilot["freeze_signature"], view_name
        )
        groups = pd.to_numeric(frame.iloc[rows]["entry_friction"], errors="raise").to_numpy()
        if not np.isin(groups, [0, 1]).all():
            raise BatchIntegrityError("R01 entry_friction contains values outside {0,1}")
        records = {
            str(level): _group_metrics(y[groups == level], prediction[groups == level], baseline[groups == level])
            for level in (0, 1)
        }
        weighted_candidate = sum(record["candidate_mae"] * record["n"] for record in records.values()) / len(rows)
        weighted_baseline = sum(record["baseline_mae"] * record["n"] for record in records.values()) / len(rows)
        if (sum(record["n"] for record in records.values()) != len(rows)
                or not np.isclose(weighted_candidate, float(np.mean(np.abs(prediction - y))), atol=1e-12, rtol=0.0)
                or not np.isclose(weighted_baseline, float(np.mean(np.abs(baseline - y))), atol=1e-12, rtol=0.0)):
            raise BatchIntegrityError("R01 entry_friction groups do not reconcile with their parent view")
        output["views"][view_name] = {"rows": int(len(rows)), "groups": records}
    return output


def main() -> int:
    result = compute()
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing != result:
            raise BatchIntegrityError("Existing R01 supplemental diagnostic differs; refusing overwrite")
    else:
        write_json_exclusive(OUTPUT_PATH, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
