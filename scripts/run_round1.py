"""Freeze first, then run/resume bounded Regression Round1; never predicts Test.csv."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.round1_evaluation import freeze_protocol, run_round1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--run-id',default='round1_v1')
    p.add_argument('--report-dir',type=Path)
    p.add_argument('--force',action='store_true',help='Refit every frozen recipe; use separate run-id/report-dir for repeat verification.')
    a=p.parse_args()
    if a.freeze_only:
        receipt=freeze_protocol(a.data_dir)
        print('Frozen before actual training:',receipt['signature'])
    else:
        result=run_round1(a.data_dir,a.run_id,a.report_dir,a.force)
        print(result['status'],result['total_jobs'],'jobs')


if __name__=='__main__':
    main()
