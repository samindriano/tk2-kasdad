"""Single-pass Train temporal scoring CLI for the POS x EF context pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

AUDIT_ROOT=Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0,str(AUDIT_ROOT))

from scripts.cls_pos_ef_context_v1 import score_metrics


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root",type=Path,required=True)
    result=score_metrics(parser.parse_args().official_root)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
