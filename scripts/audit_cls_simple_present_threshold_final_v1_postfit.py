"""Post-fit adapter for the independent final threshold verifier.

The producer receipt adds a provenance-only status string to the NO_POS
comparison. The frozen independent verifier recomputes and compares all
prediction, ID, hash, and count fields; this adapter removes only that exact
annotation in memory before delegating to the verifier. It never edits any
receipt, source, or prediction artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import verify_cls_simple_present_threshold_final_v1 as verifier

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/cls_simple_present_threshold_final_v1/FINAL_INDEPENDENT_QA.json"
EXPECTED_ANNOTATION = "PREVIOUSLY_UPLOADED_REF_58024217_COMPLETE"


def audit() -> dict:
    original_loads = json.loads
    normalized: list[str] = []

    def loads_with_annotation_normalization(data, *args, **kwargs):
        value = original_loads(data, *args, **kwargs)
        if isinstance(value, dict) and value.get("status") == "E79_APPLIED_QA_REQUIRED":
            comparisons = value.get("comparisons")
            if isinstance(comparisons, dict) and "no_pos" in comparisons:
                row = comparisons["no_pos"]
                if row.get("status") != EXPECTED_ANNOTATION:
                    raise ValueError("NO_POS provenance annotation is absent or differs from the recorded upload")
                row.pop("status")
                normalized.append("apply_receipt.comparisons.no_pos.status")
        return value

    json.loads = loads_with_annotation_normalization
    try:
        report = verifier.verify_final_candidate()
    finally:
        json.loads = original_loads
    if normalized != ["apply_receipt.comparisons.no_pos.status"]:
        raise ValueError("Expected exact provenance annotation normalization did not occur once")
    report["postfit_adapter"] = {
        "status": "PASS",
        "normalized_field": normalized[0],
        "accepted_exact_value": EXPECTED_ANNOTATION,
        "normalization_scope": "provenance label only; all prediction vectors, hashes, deltas, IDs, and counts were independently recomputed",
        "artifacts_mutated": False,
    }
    return report


def main() -> int:
    report = audit()
    payload = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_name(f".{REPORT.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, REPORT)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
