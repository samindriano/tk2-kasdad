"""Safely resume B01 report publication after its candidate files were saved.

The frozen benchmark runner has a post-output manifest-scope NameError. This
wrapper supplies the already-verified ledger value in the runner module, then
reruns the frozen procedure. Existing prediction/CSV bytes are immutable and
must match exactly; they are never overwritten.
"""
from __future__ import annotations

import json

from scripts import optimization_v3_b01_benchmark as benchmark


def main() -> None:
    if not benchmark.CSV_PATH.is_file() or not benchmark.PREDICTIONS_PATH.is_file():
        raise RuntimeError("Resume requires both already-materialized B01 candidate files")
    benchmark.ledger_after = benchmark.b01._verify_ledger()
    result, qa = benchmark._prepare()
    print(json.dumps({
        "status": result["status"],
        "benchmark_value": result["benchmark_value"],
        "transitions": result["label_transitions_vs_already_uploaded_A02"]["overall"],
        "csv": result["b01_predictions"]["csv_path"],
        "csv_sha256": result["b01_predictions"]["csv_sha256"],
        "QA": qa["status"],
        "checks": len(qa["checks"]),
    }, indent=2))


if __name__ == "__main__":
    main()
