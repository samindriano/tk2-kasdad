"""Independent integrity/readback entry point for CLS_PPMI_CONTEXT_V1."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "_cls_ppmi_context_runner", SCRIPT_DIR / "cls_ppmi_context_v1.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load CLS_PPMI_CONTEXT_V1 runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def verify(phase: str = "prelabel") -> dict[str, object]:
    return runner.verify_artifacts(phase)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prelabel", "postlabel"), default="prelabel")
    args = parser.parse_args(argv)
    result = verify(args.phase)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
