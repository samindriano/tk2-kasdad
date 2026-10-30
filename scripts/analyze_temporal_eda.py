"""Run cycle-level temporal EDA without fitting a prediction model."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.temporal_eda import run_temporal_eda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Root containing classification/ and regression/.')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports'/'eda')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    result = run_temporal_eda(data_dir=args.data_dir, output_dir=args.output_dir, make_plots=not args.no_plots)
    print(json.dumps({
        'status': result['status'], 'overall': result['overall'], 'cycle_count': result['cycle_count'],
        'marginal_vs_association': result['marginal_vs_association'],
        'partition_sensitivity': result['partition_sensitivity'],
        'primary_delta': result['primary_first_vs_last']['observed_pearson_delta_late_minus_early'],
        'bootstrap_sensitivity': result['primary_first_vs_last']['cycle_cluster_resampling'],
        'cycle_56': result['cycle_56_neighborhood_descriptive'],
        'figures': result['figures'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
