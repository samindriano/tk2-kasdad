"""Run the complete descriptive milestone without fitting any model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.formal_eda import run_formal_eda


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    bundle = run_formal_eda(args.data_dir, args.output_dir, not args.no_plots)
    print(bundle['summary']['status'])
    print('Rows:', bundle['temporal']['overall']['n'])
    print('Question:', bundle['summary']['research_question'])


if __name__ == '__main__':
    main()
