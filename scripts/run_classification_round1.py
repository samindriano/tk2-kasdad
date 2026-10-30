"""Freeze, run or resume Samuel Classification Round1; no official-test predictions."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.classification_evaluation import freeze_protocol, run_classification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-only', action='store_true')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--run-id', default='classification_round1_v1')
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--force', action='store_true')
    p.add_argument('--only-fit-end', type=int, action='append', help='Separate reproduction of a prespecified history, not new selection.')
    a = p.parse_args()
    if a.freeze_only:
        print(freeze_protocol(a.data_dir)['signature'])
    else:
        result = run_classification(a.data_dir, a.run_id, a.report_dir, a.force, a.only_fit_end)
        print(result['status'], result['total_evaluation_jobs'], 'evaluation jobs')


if __name__ == '__main__':
    main()
