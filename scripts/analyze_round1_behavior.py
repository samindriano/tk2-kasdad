"""Analyze saved Round1 validation predictions without fitting new models."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.round1_behavior import analyze_round1_behavior

if __name__=='__main__':
    result=analyze_round1_behavior()
    print(result['status'],result['metric_audit'])
