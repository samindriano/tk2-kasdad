"""Run frozen Bigram E79 Test inference with the historical receipt schema.

The frozen A02-CGEOM inference receipt uses ``test_targets_loaded`` (plural),
while the new benchmark runner expects ``test_target_loaded`` (singular). This
adapter maps that verified false flag in memory only. It does not alter the
training runner, recipe, historical receipt, or fitted state.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_bigram_position_real_e79_v1 as benchmark  # noqa: E402


def _receipt_schema_compatibility():
    original_json = benchmark._json
    receipt_path = benchmark.TEST_INFERENCE_RECEIPT.resolve()

    def read_json(path: Path) -> dict[str, Any]:
        result = original_json(path)
        if path.resolve() == receipt_path:
            plural = result.get("test_targets_loaded")
            singular = result.get("test_target_loaded")
            if plural is not False or singular not in (None, False):
                raise benchmark.BenchmarkError(
                    "Frozen historical inference receipt does not attest to unloaded Test targets"
                )
            result = dict(result)
            result["test_target_loaded"] = plural
        return result

    benchmark._json = read_json


def main() -> int:
    _receipt_schema_compatibility()
    result = benchmark.infer()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
