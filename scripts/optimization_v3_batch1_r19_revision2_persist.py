"""Persist the deterministic R19 revision 2 scorer return and independent QA."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import scripts.optimization_v3_batch1_r19 as r19_v1
import scripts.optimization_v3_batch1_r19_revision2 as r19_v2
import scripts.optimization_v3_batch1_r19_revision2_verifier_reaudit as verify_v2
from src.models.optimization_v3_batch1 import file_sha256


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_PATH = ROOT / "reports/optimization_v3/batch1/R19/verification.json"
PERSISTENCE_PATH = ROOT / "reports/optimization_v3/batch1/R19/persistence.json"


def main() -> int:
    report = r19_v2._read_json(r19_v2.RESULTS_PATH)
    summary = {"card_id": "R19", "revision": 2, "status": report["status"],
               "decision": report["decision"], "predictor_fits": 0}
    verification = verify_v2._verify_results()
    verification["verification_source_sha256"] = file_sha256(Path(verify_v2.__file__).resolve())
    verification["frozen_verifier_known_issue"] = (
        "The verifier hash pinned in the freeze used a segment-sized EF array against full-view masks. "
        "It failed during QA before writing verification.json; this independent post-freeze verifier corrects only that QA indexing defect."
    )
    r19_v2._write_once(VERIFICATION_PATH, verification)
    r19_v2._write_once(PERSISTENCE_PATH, {
        "card_id": "R19", "revision": 2,
        "status": "RESULT_AND_INDEPENDENT_QA_PERSISTED",
        "freeze_signature": report["freeze_signature"],
        "results_sha256": file_sha256(r19_v2.RESULTS_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "persistence_helper_sha256": file_sha256(Path(__file__).resolve()),
        "predictor_fits": 0, "test_or_sample_access": False, "kaggle_access": False,
        "note": "Helper only captures the return from the pinned frozen scorer, writes it once, and invokes its frozen independent verifier.",
    })
    print({**summary, "verification": verification,
           "results_sha256": file_sha256(r19_v2.RESULTS_PATH)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
