"""Reproduce regression EDA discovery; no fit, model, prediction, or submission."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.discovery import run_discovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,help='Root containing classification/ and regression/.')
    parser.add_argument('--tp1-data-dir',type=Path,help='Optional TP1 raw root for target-distribution comparison; omitted unless provided.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports'/'eda')
    parser.add_argument('--no-plots',action='store_true')
    a=parser.parse_args()
    result=run_discovery(a.data_dir,a.tp1_data_dir,a.output_dir,not a.no_plots)
    print(json.dumps({'status':result['status'],'target':result['target_tp2']['unique'],
                      'files':{k:v['name'] for k,v in result['provenance']['regression_files'].items()},
                      'overall_correlation':result['impact_threat']['overall_correlation'],
                      'within_cycle_percentile_exact':result['within_cycle_percentile_diagnostic']['within_cycle']['exact_match_fraction'],
                      'tp1_comparison_status':result['tp1_target_comparison']['status'],
                      'figures':result['figures'],'report':str(a.output_dir/'discovery_summary.json')},indent=2))


if __name__=='__main__':
    main()
