"""Runtime compatibility adapter for the sparse MLP recovery finalizer.

The frozen training runner exposes `_release_if_reserved`; the recovery
finalizer expects its descriptive alias `release_unused_reservation`. This
adapter provides that alias in-process only, without editing the frozen runner
or recovery code that is already pinned in the cutoff-55 recovery receipt.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import optimization_v3_classification_sparse_mlp_v1_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/optimization_v3/classification_mlp_v1/FINALIZATION_ADAPTER.json"


def _git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def finalize() -> dict[str, Any]:
    runner = recovery.runner
    compatibility_alias_added = False
    if not hasattr(runner, "release_unused_reservation"):
        release = getattr(runner, "_release_if_reserved", None)
        if not callable(release):
            raise RuntimeError("Frozen runner does not expose its reservation release helper")
        runner.release_unused_reservation = release
        compatibility_alias_added = True

    result = recovery.finalize_and_verify()
    if result["verification"].get("status") != "PASS":
        raise RuntimeError("Independent recovery verification did not pass")
    frozen = runner._json(runner.FREEZE)
    adapter_receipt = {
        "status": "FINALIZATION_COMPLETE",
        "campaign": runner.CARD_ID,
        "adapter_source_commit": _git_head(),
        "adapter_source_sha256": runner._sha(Path(__file__)),
        "recovery_source_sha256": runner._sha(Path(recovery.__file__)),
        "freeze_signature": frozen["signature"],
        "runtime_compatibility_alias_added": compatibility_alias_added,
        "release_method": "runner._release_if_reserved -> research_next_budget.release_unused",
        "scientific_verdict": result["result"]["scientific_verdict"],
        "verification_status": result["verification"]["status"],
        "test_read": False,
        "sample_submission_read": False,
        "kaggle_access": False,
    }
    recovery._atomic_json(OUTPUT, adapter_receipt)
    return {"result": result["result"], "verification": result["verification"],
            "finalization_adapter": adapter_receipt}


if __name__ == "__main__":
    output = finalize()
    print(json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
