"""Freeze and execute bounded Round2; never write old checkpoint or official-test predictions."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.classification_round2_evaluation import freeze_protocol, routing_diagnostic, run_round2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-only', action='store_true')
    p.add_argument('--diagnostic-only', action='store_true')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--run-id', default='classification_round2_v1')
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--force', action='store_true')
    p.add_argument('--only-fit-end', type=int, action='append')
    p.add_argument('--bootstrap-dependencies', action='store_true', help='Reconstruct missing control vectors in Round2 cache; demand exact Round1 hashes.')
    a = p.parse_args()
    if a.freeze_only: print(freeze_protocol(a.data_dir)['signature'])
    elif a.diagnostic_only: print(routing_diagnostic(a.data_dir, a.bootstrap_dependencies)['status'])
    else:
        r = run_round2(a.data_dir, a.run_id, a.report_dir, a.force, a.only_fit_end, a.bootstrap_dependencies)
        print(r['status'], r['total_evaluation_jobs'], 'evaluation jobs')


if __name__ == '__main__': main()
