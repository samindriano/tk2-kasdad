"""Render aggregate Round 1 comparison tables and validation figures."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.round1_reporting import build_round1_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Completed results.json (defaults to reports/round1/results.json).")
    parser.add_argument("--output-dir", type=Path, help="Directory for COMPARISON_TABLES.md and figures/.")
    parser.add_argument("--data-dir", type=Path, help="TP2 data root containing regression/Train.csv; defaults to TP2_DATA_DIR or data/raw.")
    parser.add_argument("--artifact-dir", type=Path, help="Ignored validation prediction cache directory; defaults to experiments/runs/<run_id>/.")
    args = parser.parse_args()
    outputs = build_round1_report(args.results, args.output_dir, args.data_dir, args.artifact_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
