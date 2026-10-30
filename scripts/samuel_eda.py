"""Run Samuel's descriptive TP2 regression EDA."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.samuel import run_eda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data-root", dest="data_root", type=Path,
                        help="Root containing classification/ and regression/ (defaults to TP2_DATA_DIR or data/raw).")
    parser.add_argument("--output-dir", type=Path, help="Defaults to reports/eda.")
    parser.add_argument("--no-plots", action="store_true", help="Write summary JSON without figures.")
    args = parser.parse_args()
    report = run_eda(root=args.data_root, output_dir=args.output_dir, make_plots=not args.no_plots)
    output = "reports/eda" if args.output_dir is None else "<configured-output-dir>"
    print(json.dumps({"status": report["status"], "input_root": report["input_root"],
                      "missing_files": report.get("missing_files", []),
                      "report": f"{output}/samuel_regression_eda.json"}, indent=2))


if __name__ == "__main__":
    main()
