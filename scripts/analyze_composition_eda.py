"""Run Samuel's conditional/composition regression EDA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.composition_eda import run_composition_eda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        help="Root containing regression/ (defaults to TP2_DATA_DIR or data/raw).")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (defaults to reports/eda).")
    parser.add_argument("--no-plots", action="store_true", help="Write JSON only.")
    args = parser.parse_args()
    report = run_composition_eda(data_dir=args.data_dir, output_dir=args.output_dir,
                                 make_plots=not args.no_plots)
    output = args.output_dir or PROJECT_ROOT / "reports" / "eda"
    summary = {
        "status": report["status"],
        "input_rows": report["source"]["input_rows"],
        "analyzable_rows": report["source"]["analyzable_rows"],
        "periods": report["periods"],
        "common_impact_level_count": report["support_selection"]["common_impact_level_count"],
        "common_impact_x_friction_cell_count": report["support_selection"]["common_impact_x_friction_cell_count"],
        "report": str(Path(output) / "composition_robustness.json"),
        "figures": report["figures"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
