"""Run from any working directory: python /path/to/scripts/audit_data.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.audit import audit_task, compare_tp1, environment, markdown_report, write_json
from src.utils.paths import PROJECT_ROOT, data_root, task_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=None, help='Root containing classification/ and regression/. Overrides TP2_DATA_DIR.')
    parser.add_argument('--tp1-data-dir', type=Path, help='Optional read-only TP1 raw root, for schema and shared-ID comparison.')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'reports' / 'audits')
    args = parser.parse_args()
    root = args.data_dir.resolve() if args.data_dir else data_root()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in ('classification', 'regression'):
        result = audit_task(task, root)
        result['environment'] = environment()
        if args.tp1_data_dir:
            result['tp1_comparison'] = compare_tp1(task, task_files(task, root), args.tp1_data_dir)
        write_json(args.output_dir / f'{task}_audit.json', result)
        (args.output_dir / f'{task}_audit.md').write_text(markdown_report(result), encoding='utf-8')
        print(f"{task}: {result['status']} | target={result['target']} | report={args.output_dir / (task+'_audit.md')}")
    # Missing inputs are an expected blocked result, not fabricated success.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
