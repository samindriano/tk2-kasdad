"""Run frozen Regression Round2 only; classification and submission tracks excluded."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.regression_round2_evaluation import freeze_protocol, run_round2, DEFAULT_RUN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-only', action='store_true')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--run-id', default=DEFAULT_RUN)
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--stage', choices=['base', 'all'], default='base')
    p.add_argument('--only-fit-end', type=int)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if a.freeze_only:
        print('Frozen:', freeze_protocol(a.data_dir)['signature'])
    else:
        r = run_round2(a.data_dir, a.run_id, a.report_dir, a.stage, a.only_fit_end, a.force)
        print(r['status'], len(r['jobs']), 'evaluations;', r['fresh_fits'], 'fresh fits')


if __name__ == '__main__':
    main()
