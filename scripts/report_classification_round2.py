"""Render readable tables and figures after the frozen Round 2 run completes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.classification_round2_reporting import build_round2_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="Completed Round 2 results JSON path.")
    parser.add_argument("--output-dir", help="Destination directory for comparison_tables.md and figures/.")
    parser.add_argument("--data-dir", help="Reserved for interface consistency; the reporter does not read raw data.")
    args = parser.parse_args()
    result = build_round2_report(args.results, args.output_dir, args.data_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "tables"}, indent=2))


if __name__ == "__main__":
    main()
