from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "reg_position_real_e79_benchmark_v1.py"


def test_synthetic_cli_e2e_has_two_durable_units_resume_and_independent_verifier() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "synthetic-e2e"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["status"] == "PASS_SYNTHETIC_E2E"
    assert report["temp_only"] is True
    assert report["fit_invocations"] == [
        "WORD_TFIDF_FULL_TRAIN",
        "REAL_POSITION_RESIDUAL_L1_E79",
    ]
    assert report["fit_starts"] == 2
    assert report["fit_saved"] == 2
    assert report["resume_without_refit"] is True
    assert report["independent_verifier"] == "PASS_SYNTHETIC_INDEPENDENT_VERIFIER"
