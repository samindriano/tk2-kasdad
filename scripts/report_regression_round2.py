"""Rebuild saved-vector Regression Round2 diagnostics, tables and figures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.regression_round2_reporting import build_report

if __name__ == '__main__':
    for path in build_report():
        print(path)
