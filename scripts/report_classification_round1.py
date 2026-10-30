"""Generate readable comparison tables and figures from completed Round 1."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.classification_reporting import build_classification_report
from src.utils.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Completed results.json; defaults to canonical report.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/classification_round1")
    parser.add_argument("--data-dir", type=Path, help="Optional TP2 root containing classification/.")
    args = parser.parse_args()
    result = build_classification_report(args.results, args.output_dir, args.data_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
