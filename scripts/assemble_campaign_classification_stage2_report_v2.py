"""Corrected report-only recovery wrapper; it performs no model fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.audit import sha256 as file_sha256, write_json
from src.models.round1_evaluation import code_digest, digest_json
from src.utils.paths import task_files
from scripts import assemble_campaign_classification_stage2 as parent_assembler
from scripts.research_campaign_classification_stage2 import (
    CANDIDATES,
    NEW_FIT_ENDS,
    PRIMARY_WINDOWS,
    RUN_DIR as STAGE2_RUN_DIR,
    STAGE1_RUN_DIR,
    STAGE1_VERIFICATION_PATH,
    AMENDMENT_FREEZE_PATH,
    STAGE2_FREEZE_PATH,
    STAGE2_PROTOCOL_PATH,
    verify_freeze as verify_stage2_freeze,
)
from scripts.research_campaign_classification import (
    FAILED_ATTEMPT_PATH as STAGE1_FAILURE_PATH,
    verify_amendment,
)


REPAIR_PROTOCOL_PATH = ROOT / "experiments/dual_campaign/classification/stage2_report_recovery_amendment_01.json"
REPAIR_FREEZE_PATH = ROOT / "experiments/dual_campaign/classification/stage2_report_recovery_amendment_01_freeze.json"
PARENT_RECOVERY_NOTE = ROOT / "reports/dual_campaign/classification/stage2_report_recovery_note.json"
ATTEMPT_NOTE = ROOT / "reports/dual_campaign/classification/stage2_report_recovery_attempt_01.json"
REPORT_DIR = parent_assembler.REPORT_DIR
RESULT_PATH = parent_assembler.RESULT_PATH
STAGE1_REPORT_DIR = ROOT / "reports/dual_campaign/classification/amendment_01"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CODE_FILES = [
    "experiments/dual_campaign/classification/protocol.json",
    "experiments/dual_campaign/classification/amendment_01.json",
    "experiments/dual_campaign/classification/amendment_01_freeze.json",
    "experiments/dual_campaign/classification/stage2_protocol.json",
    "experiments/dual_campaign/classification/stage2_freeze.json",
    "experiments/dual_campaign/classification/stage2_report_recovery_protocol.json",
    "experiments/dual_campaign/classification/stage2_report_recovery_amendment_01.json",
    "scripts/research_campaign_classification.py",
    "scripts/verify_campaign_classification.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/verify_campaign_classification_stage2.py",
    "scripts/assemble_campaign_classification_stage2.py",
    "scripts/verify_campaign_classification_stage2_recovery.py",
    "scripts/assemble_campaign_classification_stage2_report_v2.py",
    "tests/test_campaign_classification.py",
    "tests/test_campaign_classification_stage2.py",
    "tests/test_campaign_classification_stage2_recovery.py",
    "tests/test_campaign_classification_stage2_recovery_v2.py",
    "src/models/campaign_classification.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/utils/paths.py",
    "src/data/audit.py",
]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_paths(data_dir: Path | None = None, *, allow_output: bool = False) -> dict[str, Path]:
    train = task_files("classification", data_dir)["train"]
    if train is None:
        raise FileNotFoundError("Official Classification Train.csv is missing")
    paths: dict[str, Path] = {
        "classification_train": train.resolve(),
        "stage1_amendment_freeze": AMENDMENT_FREEZE_PATH,
        "stage1_verification": STAGE1_VERIFICATION_PATH,
        "stage1_failure_receipt": STAGE1_FAILURE_PATH,
        "stage2_protocol": STAGE2_PROTOCOL_PATH,
        "stage2_freeze": STAGE2_FREEZE_PATH,
        "stage2_report_recovery_protocol": parent_assembler.RECOVERY_PROTOCOL_PATH,
        "stage2_report_recovery_freeze": parent_assembler.RECOVERY_FREEZE_PATH,
        "stage2_report_failure_note": PARENT_RECOVERY_NOTE,
        "stage2_report_attempt_01_note": ATTEMPT_NOTE,
    }
    for end in (55, 67):
        for recipe in CANDIDATES:
            for suffix in ("npz", "json"):
                paths[f"stage1/fit{end}/{recipe}/{suffix}"] = STAGE1_RUN_DIR / f"fit{end}__{recipe}.{suffix}"
    for end in NEW_FIT_ENDS:
        for recipe in CANDIDATES:
            for suffix in ("npz", "json"):
                paths[f"stage2/fit{end}/{recipe}/{suffix}"] = STAGE2_RUN_DIR / f"fit{end}__{recipe}.{suffix}"
    for end in (55, 59, 63, 67, 71, 75):
        for model in ("C1_B", "C2_B"):
            paths[f"round1/fit{end}/{model}"] = ROUND1_CACHE / f"fit{end}__{model}.npz"
        paths[f"round1/fit{end}/audit"] = ROUND1_AUDITS / f"fit{end}.json"
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Recovery input missing: " + "; ".join(missing))
    if RESULT_PATH.exists() and not allow_output:
        raise FileExistsError("The recovered Stage 2 report already exists; do not overwrite")
    return paths


def repair_identity(data_dir: Path | None = None) -> dict[str, Any]:
    parent_stage2 = verify_stage2_freeze(data_dir)
    parent_stage1 = verify_amendment(data_dir)
    parent_recovery = parent_assembler.verify_recovery_freeze(data_dir)
    if not (REPORT_DIR / "full_six_results.json").exists():
        pass
    else:
        raise FileExistsError("The report already exists; freeze this recovery before first assembly")
    failure_note = _read(ATTEMPT_NOTE)
    if (failure_note.get("parent_recovery_freeze_signature") != parent_recovery["signature"]
            or failure_note.get("report_json_written") is not False):
        raise RuntimeError("Second report recovery failure receipt mismatch")
    stage1_verification = _read(STAGE1_VERIFICATION_PATH)
    if stage1_verification.get("status") != "PASS" or stage1_verification.get("freeze_signature") != parent_stage1["signature"]:
        raise RuntimeError("Stage 1 reusable outputs are not verified")
    inputs = _input_paths(data_dir)
    return {
        "repair_protocol_sha256": digest_json(_read(REPAIR_PROTOCOL_PATH)),
        "code_sha256": {name: code_digest(ROOT / name) for name in CODE_FILES},
        "input_sha256": {name: file_sha256(path) for name, path in inputs.items()},
        "parent_stage2_freeze_signature": parent_stage2["signature"],
        "parent_report_recovery_freeze_signature": parent_recovery["signature"],
        "parent_report_recovery_attempt_sha256": file_sha256(ATTEMPT_NOTE),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "matplotlib"]},
    }


def freeze_repair(data_dir: Path | None = None) -> dict[str, Any]:
    current = repair_identity(data_dir)
    if REPAIR_FREEZE_PATH.exists():
        receipt = _read(REPAIR_FREEZE_PATH)
        if receipt.get("identity") != current:
            raise RuntimeError("Existing recovery amendment freeze differs; preserve it and number a new amendment")
        return receipt
    receipt = {
        "status": "PLOT_ACCESSOR_REPAIR_FROZEN_BEFORE_REPORT_OUTPUT_NO_REFITS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_stage2_freeze_signature": current["parent_stage2_freeze_signature"],
        "parent_report_recovery_freeze_signature": current["parent_report_recovery_freeze_signature"],
        "identity": current,
        "signature": digest_json(current),
        "operation_boundary": "Corrected plot rendering only; parent assembler revalidates and aggregates the same fixed metrics from the 12 existing caches. No fit or prediction.",
    }
    write_json(REPAIR_FREEZE_PATH, receipt)
    return receipt


def verify_repair_freeze(data_dir: Path | None = None) -> dict[str, Any]:
    if not REPAIR_FREEZE_PATH.is_file():
        raise RuntimeError("Freeze report correction before reassembly")
    receipt = _read(REPAIR_FREEZE_PATH)
    expected = receipt.get("identity")
    if not isinstance(expected, dict) or receipt.get("signature") != digest_json(expected):
        raise RuntimeError("Report correction freeze receipt is corrupted")
    current_code = {name: code_digest(ROOT / name) for name in CODE_FILES}
    if current_code != expected.get("code_sha256"):
        raise RuntimeError("Report correction code changed after its freeze")
    for name, path in _input_paths_allow_report(data_dir).items():
        if file_sha256(path) != expected["input_sha256"].get(name):
            raise RuntimeError(f"Report recovery input changed after freeze: {name}")
    if expected.get("parent_stage2_freeze_signature") != verify_stage2_freeze(data_dir)["signature"]:
        raise RuntimeError("Report recovery parent Stage 2 freeze changed")
    return receipt


def _input_paths_allow_report(data_dir: Path | None = None) -> dict[str, Path]:
    # Result JSON is an output and is expected after assembly, not a recovery input.
    return _input_paths(data_dir, allow_output=True)


def plot_series(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    series: dict[str, dict[str, float]] = {"macro_f1": {}, "average_precision": {}}
    for name in PRIMARY_WINDOWS:
        c4 = report["window_metrics"][name][CANDIDATES[0]]["C4_HGB"]
        series["macro_f1"][f"{name}/C4_HGB"] = float(c4["macro_f1"])
        series["average_precision"][f"{name}/C4_HGB"] = float(c4["average_precision"])
        for recipe in CANDIDATES:
            current = report["window_metrics"][name][recipe]["candidate"]
            series["macro_f1"][f"{name}/{recipe}"] = float(current["macro_f1"])
            series["average_precision"][f"{name}/{recipe}"] = float(current["average_precision"])
    return series


def _plot(report: dict[str, Any]) -> list[Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"C4_HGB": "#333333", "DC_SVC": "#4472C4", "DC_NBSVM": "#70AD47", "DC_WORD_CHAR": "#ED7D31"}
    windows = list(PRIMARY_WINDOWS)
    x = np.arange(len(windows))
    p1 = REPORT_DIR / "full_six_window_macro_f1.png"
    fig, axis = plt.subplots(figsize=(9.4, 4.7))
    for model_id in ["C4_HGB", *CANDIDATES]:
        if model_id == "C4_HGB":
            values = [report["window_metrics"][name][CANDIDATES[0]]["C4_HGB"]["macro_f1"] for name in windows]
        else:
            values = [report["window_metrics"][name][model_id]["candidate"]["macro_f1"] for name in windows]
        axis.plot(x, values, marker="o", linewidth=1.8, label=model_id, color=colors[model_id])
    axis.set_xticks(x, windows)
    axis.set_ylim(0.55, 0.90)
    axis.set_ylabel("Macro F1 · threshold 0.50")
    axis.set_title("Six expanding windows · candidate routes vs C4_HGB")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    p2 = REPORT_DIR / "paired_present_errors.png"
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    pos, width = np.arange(len(CANDIDATES)), 0.36
    candidate_only = [report["complementarity"][recipe]["candidate_only_correct"] for recipe in CANDIDATES]
    c4_only = [report["complementarity"][recipe]["C4_only_correct"] for recipe in CANDIDATES]
    axis.bar(pos - width / 2, candidate_only, width, color="#70AD47", label="Candidate-only correct")
    axis.bar(pos + width / 2, c4_only, width, color="#777777", label="C4-only correct")
    axis.set_xticks(pos, CANDIDATES, rotation=15)
    axis.set_ylabel("Stream-present validation rows")
    axis.set_title("Paired corrections on the same validation rows")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


def run_repair(data_dir: Path | None = None) -> dict[str, Any]:
    frozen = verify_repair_freeze(data_dir)
    # Runtime patch is local to this process. The frozen parent source stays unchanged.
    parent_assembler._plot = _plot
    report = parent_assembler.assemble(data_dir)
    from scripts.verify_campaign_classification_stage2 import verify as verify_stage2_report
    independent = verify_stage2_report(data_dir)
    if independent.get("status") != "PASS":
        raise RuntimeError("Frozen independent Stage 2 report verifier did not pass")
    # Do not call parent recovery verify here: it correctly required an absent output pre-assembly.
    post = verify_repair_freeze_after_output(frozen)
    verification = {
        "status": "PASS",
        "recovery_amendment_freeze_signature": frozen["signature"],
        "stage2_freeze_signature": report["freeze_signature"],
        "new_model_fits": 0,
        "report_metrics_from_saved_predictions": True,
        "independent_stage2_verifier": independent,
        "input_hashes_unchanged": post,
        "official_test_read": False,
        "kaggle_submission": False,
    }
    write_json(REPORT_DIR / "stage2_report_recovery_amendment_01_verification.json", verification)
    return report


def verify_repair_freeze_after_output(receipt: dict[str, Any]) -> dict[str, Any]:
    expected = receipt["identity"]
    current_code = {name: code_digest(ROOT / name) for name in CODE_FILES}
    if current_code != expected["code_sha256"]:
        raise RuntimeError("Report recovery source changed after freeze")
    hashes = {}
    for name, path in _input_paths_allow_report().items():
        current = file_sha256(path)
        if current != expected["input_sha256"].get(name):
            raise RuntimeError(f"Report recovery input changed after freeze: {name}")
        hashes[name] = current
    return {"verified_input_count": len(hashes), "verified_code_count": len(current_code),
            "input_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--repair-report", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.repair_report:
        parser.error("Choose exactly one of --freeze-only or --repair-report")
    if args.freeze_only:
        receipt = freeze_repair(args.data_dir)
        print(receipt["status"], receipt["signature"])
    else:
        report = run_repair(args.data_dir)
        print(report["status"], report["new_fit_count"], "new fits (all reused from immutable caches)")


if __name__ == "__main__":
    main()
