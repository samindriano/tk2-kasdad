"""Queue-first dispatcher for the V3 batch1 execution namespace."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_batch1 import BatchIntegrityError, load_queue


QUEUE = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
def run_card(card_id: str, *, preflight_only: bool = False) -> dict[str, Any]:
    queue = load_queue(QUEUE)
    card_map = {str(item["id"]): item for item in queue["cards"]}
    if card_id not in card_map:
        raise BatchIntegrityError(f"Card is not in the frozen queue: {card_id}")
    module_name = f"scripts.optimization_v3_batch1_{card_id.lower()}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise NotImplementedError(f"{card_id} remains planned; no adapter has passed B0 yet")
    runner = getattr(module, "run_card", None)
    if not callable(runner):
        raise BatchIntegrityError(f"Registered module has no callable run_card: {module_name}")
    return runner(card_map[card_id], queue, preflight_only=preflight_only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card_id", help="ID from the frozen 45-card queue")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate official Train, source hashes, controls, views and resources; do not freeze or fit")
    args = parser.parse_args()
    result = run_card(args.card_id, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
