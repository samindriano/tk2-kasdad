"""Re-verify saved R20 pilot artifacts without training another model."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r20 as runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError, file_sha256, load_queue, write_json_exclusive,
)
from src.models.regression_research_text import HYBRID_FEATURES, _json_digest


def _json_compatible(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def verify_saved_pilot() -> dict[str, Any]:
    """Recompute saved result and gates; only read Train, controls, and run receipts."""
    runner.TOKEN_ANALYZER_PARAMETERS = _json_compatible(runner.TOKEN_ANALYZER_PARAMETERS)
    queue = load_queue(runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == runner.CARD_ID)
    context = runner._prepare(card, queue)
    if context["public"].get("status") != "PREFLIGHT_PASS":
        raise BatchIntegrityError("R20 read-only QA preflight is not PASS")

    frozen = runner._read_json(runner.FREEZE_PATH)
    if (context["identity"] != frozen.get("identity")
            or runner.json_sha256(context["identity"]) != frozen.get("signature")):
        raise BatchIntegrityError("R20 QA inputs or source identity differ from the frozen fit")
    if runner._read_json(runner.RECIPE_PATH) != context["recipe"]:
        raise BatchIntegrityError("R20 QA recipe differs from the recipe used for the saved fit")
    report = runner._read_json(runner.PILOT_PATH)
    signature = report.get("freeze_signature")
    if signature != frozen["signature"]:
        raise BatchIntegrityError("R20 saved pilot receipt does not match the committed freeze")

    # Verify the selected-token digest from the exact outer-fit rows. The model
    # runner's serialized tuple becomes a list in JSON; this check is content based.
    frame = context["frame"]
    target = np.asarray(context["target"], dtype=np.float64)
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        expected = runner._selection_audit_independent(
            frame.iloc[fit_rows][list(HYBRID_FEATURES)], target[fit_rows]
        )
        sidecar = runner._read_json(runner.RUN_DIR / f"R20_fit{cutoff}.json")
        state = sidecar.get("metadata", {}).get("state", {})
        selection = state.get("selection", {})
        expected_tokens = [record["token"] for record in expected["selected_token_records"]]
        actual_tokens = [record.get("token") for record in selection.get("selected_token_records", [])]
        if (actual_tokens != expected_tokens
                or selection.get("selected_tokens_sha256") != _json_digest(expected_tokens)):
            raise BatchIntegrityError(f"R20 selected-token identity digest differs at cutoff {cutoff}")

    context["queue"] = queue
    verification = runner._verify_pilot(context, report)
    frame_with_target = frame.copy()
    frame_with_target["threat_rank"] = target
    generic = verify_report(
        report=report, frame=frame_with_target, target=target,
        artifact_dir=runner.RUN_DIR, freeze_signature=signature,
        evaluation_views=context["views"],
        prediction_reconstructors={runner.CARD_ID: runner.reconstruct_r20_arrays},
    )
    if generic.get("status") != "PASS":
        raise BatchIntegrityError("Generic independent verifier rejected the R20 saved pilot")
    verification["generic_independent_verifier"] = generic
    verification["post_fit_verifier_sha256"] = file_sha256(Path(__file__))
    verification["serialization_note"] = (
        "Frozen analyzer ngram_range is a tuple in Python and a list in JSON; QA normalized only "
        "that representation. Training recipe, fit artifacts, predictions, and freeze were unchanged."
    )
    result = {
        "status": verification["status"], "decision": verification["decision"],
        "views": verification["views"], "gate_checks": verification["gate_checks"],
        "fit_execution_accounting": verification["fit_execution_accounting"],
        "generic_independent_verifier": generic["status"],
        "post_fit_verifier_sha256": verification["post_fit_verifier_sha256"],
        "serialization_note": verification["serialization_note"],
        "selected_token_identity_digests_reconstructed": True,
        "official_test_read": False, "kaggle_access": False, "new_model_fits": 0,
    }
    report_path = runner.REPORT_DIR / "qa_recheck.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing != runner._clean(result):
            raise BatchIntegrityError(f"Existing R20 QA recheck differs; refusing overwrite: {report_path}")
    else:
        write_json_exclusive(report_path, runner._clean(result))
    return result


if __name__ == "__main__":
    print(json.dumps(verify_saved_pilot(), indent=2, allow_nan=False))
