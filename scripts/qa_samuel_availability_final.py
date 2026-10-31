"""Portable fresh-kernel QA for Samuel's formal EDA and final models.

The validation copy is assembled from tracked source at HEAD plus the explicitly
owned Lane A edits. Only the two official Train CSV files are added; Test data,
ignored experiment caches, model binaries, submissions, and credentials are
excluded. The current Python environment is reused without installing packages.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "eda_availability": "notebooks/samuel/12_samuel_formal_eda_availability.ipynb",
    "classification": "notebooks/samuel/10_samuel_final_classification.ipynb",
    "regression": "notebooks/samuel/09_samuel_final_regression.ipynb",
}
TRACKED_PREFIXES = ("src/", "reports/", "experiments/")
TRACKED_ROOT_FILES = {"requirements.txt", "requirements-research.txt"}
TRACKED_NOTEBOOKS = set(NOTEBOOKS.values())
OWNED_OVERLAYS = [
    *NOTEBOOKS.values(),
    "src/eda/__init__.py",
    "src/eda/availability.py",
    "scripts/build_samuel_availability_final.py",
    "scripts/build_samuel_final_classification.py",
    "scripts/qa_samuel_availability_final.py",
    "tests/test_availability_eda.py",
    "docs/SAMUEL_EDA.md",
    "docs/SAMUEL_FINAL_COMPLIANCE.md",
    "docs/SAMUEL_DEMO.md",
    "docs/SAMUEL_FINAL_REPRODUCIBILITY.md",
    "docs/WORK_ALLOCATION.md",
    "reports/academic_availability/FINDINGS.md",
    "reports/academic_availability/QA.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(revision: str) -> list[str]:
    raw = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "-z", revision])
    return [value.decode("utf-8") for value in raw.split(b"\0") if value]


def needed_tracked(path: str) -> bool:
    if path in TRACKED_ROOT_FILES or path in TRACKED_NOTEBOOKS:
        return True
    if path.startswith(TRACKED_PREFIXES):
        suffix = Path(path).suffix.lower()
        if path.startswith("src/"):
            return suffix == ".py"
        if path.startswith("experiments/"):
            return suffix == ".json"
        return suffix in {".json", ".md", ".png", ".csv", ".txt"}
    return False


def make_source_copy(bundle: Path, revision: str) -> int:
    count = 0
    for relative in tracked_paths(revision):
        if not needed_tracked(relative):
            continue
        content = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{relative}"])
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        count += 1
    for relative in OWNED_OVERLAYS:
        source = ROOT / relative
        if not source.is_file():
            continue
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return count


def summarize_from_raw(classification_train: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from src.eda.availability import add_global_quantile_bins, prepare_availability_frame, prevalence_table, standardized_contrast
    from src.models.classification_round2_models import stream_missing_mask

    raw = pd.read_csv(classification_train)
    frame = prepare_availability_frame(raw, stream_missing_mask(raw))
    marginal = prevalence_table(frame, ["availability"]).set_index("availability")
    cycle, _ = standardized_contrast(frame, ["cycle_code"], min_support=20)
    q5frame, _ = add_global_quantile_bins(frame, 5)
    q5, _ = standardized_contrast(q5frame, ["period", "text_load_q5"], min_support=20)
    q10frame, q10_edges = add_global_quantile_bins(frame, 10)
    q10, q10cells = standardized_contrast(q10frame, ["period", "text_load_q10"], min_support=20)
    summary = {
        "train_rows": int(len(frame)),
        "missing_rows": int((frame.availability == "missing").sum()),
        "present_rows": int((frame.availability == "present").sum()),
        "prevalence_present": float(marginal.loc["present", "prevalence"]),
        "prevalence_missing": float(marginal.loc["missing", "prevalence"]),
        "marginal_gap_pp": float(100 * (marginal.loc["missing", "prevalence"] - marginal.loc["present", "prevalence"])),
        "exact_cycle_gap_pp": cycle["standardized_missing_minus_present_pp"],
        "exact_cycle_coverage": cycle["coverage_all_rows"],
        "period_x_q5_gap_pp": q5["standardized_missing_minus_present_pp"],
        "period_x_q5_coverage": q5["coverage_all_rows"],
        "period_x_q10_gap_pp": q10["standardized_missing_minus_present_pp"],
        "period_x_q10_coverage": q10["coverage_all_rows"],
        "period_x_q10_supported_cells": q10["cells_supported"],
        "period_x_q10_total_cells": q10["cells_total"],
        "period_x_q10_uncovered_missing": q10["total_rows_by_availability"]["missing"] - q10["covered_rows_by_availability"]["missing"],
        "period_x_q10_uncovered_present": q10["total_rows_by_availability"]["present"] - q10["covered_rows_by_availability"]["present"],
        "period_x_q10_quantile_bins": int(len(q10_edges) - 1),
        "period_x_q10_unsupported_cells": q10cells.loc[~q10cells.supported].to_dict(orient="records"),
    }
    expected = {
        "train_rows": 50608,
        "missing_rows": 3404,
        "present_rows": 47204,
        "marginal_gap_pp": 3.65995492624408,
        "exact_cycle_gap_pp": 3.9596125866310947,
        "period_x_q5_gap_pp": -2.2980955740857736,
        "period_x_q10_gap_pp": -2.152680597449513,
        "period_x_q10_coverage": 0.9433291179260196,
    }
    for name, value in expected.items():
        observed = summary[name]
        if isinstance(value, float):
            assert abs(observed - value) < 1e-10, (name, observed, value)
        else:
            assert observed == value, (name, observed, value)
    return summary


def main() -> int:
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(ROOT))
    from src.utils.paths import task_files

    source_inputs = {}
    for task in ("classification", "regression"):
        train = task_files(task)["train"]
        if train is None:
            raise FileNotFoundError(f"Official {task} Train.csv is required for this QA.")
        source_inputs[task] = train

    temp_parent = Path(tempfile.gettempdir()).resolve()
    temp = tempfile.TemporaryDirectory(prefix="tp2_samuel_availability_")
    bundle = Path(temp.name).resolve()
    if temp_parent not in bundle.parents:
        raise RuntimeError("Temporary validation root is outside the approved temporary directory.")
    started = time.perf_counter()
    old_data_root = os.environ.get("TP2_DATA_DIR")
    receipt = {
        "status": "RUNNING",
        "source_revision": revision,
        "source_copy": "tracked files from source revision plus explicitly scoped Lane A working files",
        "portable_copy_file_count": None,
        "environment": "fresh Jupyter kernels using the currently installed Python dependencies; no dependency installation",
        "raw_inputs_sha256": {},
        "raw_inputs_in_portable_copy": [],
        "test_files_present": False,
        "ignored_experiment_cache_present": False,
        "notebooks": {},
        "unit_tests": None,
    }
    try:
        receipt["portable_copy_file_count"] = make_source_copy(bundle, revision)
        raw_root = bundle / "official_data"
        for task, train in source_inputs.items():
            destination = raw_root / task / train.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(train, destination)
            receipt["raw_inputs_sha256"][f"{task}/Train.csv"] = sha256(train)
            receipt["raw_inputs_in_portable_copy"].append(f"{task}/Train.csv")
        os.environ["TP2_DATA_DIR"] = str(raw_root)

        csv_paths = [path.relative_to(raw_root).as_posix() for path in raw_root.rglob("*.csv")]
        assert sorted(csv_paths) == ["classification/Train.csv", "regression/Train.csv"], csv_paths
        assert not (bundle / "experiments" / "runs").exists()
        assert not (bundle / "submissions").exists()

        eda_metrics = summarize_from_raw(raw_root / "classification" / source_inputs["classification"].name)
        receipt["eda_metrics_recomputed_from_official_train"] = eda_metrics

        for key, relative in NOTEBOOKS.items():
            notebook_path = bundle / relative
            notebook = nbformat.read(notebook_path, as_version=4)
            start = time.perf_counter()
            NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(bundle)}},
            ).execute()
            nbformat.validate(notebook)
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            errors = [
                {"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")}
                for index, cell in enumerate(notebook.cells)
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            ]
            assert not errors, (relative, errors)
            assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
            notebook_figures = []
            for cell_index, cell in enumerate(notebook.cells):
                for output_index, output in enumerate(cell.get("outputs", [])):
                    png = output.get("data", {}).get("image/png")
                    if png:
                        number = len(notebook_figures) + 1
                        file_ref = (
                            f"reports/academic_availability/figures/eda_availability_{number}.png"
                            if key == "eda_availability"
                            else f"embedded in {relative}, cell {cell_index}, output {output_index}"
                        )
                        notebook_figures.append({
                            "file": file_ref,
                            "figure_index": number,
                            "bytes": len(base64.b64decode(png)),
                            "cell": cell_index,
                            "output": output_index,
                        })
            expected_figures = {"eda_availability": 4, "classification": 5, "regression": 2}[key]
            assert len(notebook_figures) == expected_figures, (key, len(notebook_figures))
            receipt["notebooks"][key] = {
                "path": relative,
                "cells": len(notebook.cells),
                "executed_code_cells": len(code_cells),
                "execution_errors": 0,
                "embedded_png_figures": notebook_figures,
                "seconds": round(time.perf_counter() - start, 2),
            }
            nbformat.write(notebook, notebook_path)

        class_receipt_path = bundle / "artifacts" / "samuel_final" / "classification_core_receipt.json"
        regression_receipt_path = bundle / "artifacts" / "samuel_final" / "regression_core_receipt.json"
        class_core = json.loads(class_receipt_path.read_text(encoding="utf-8"))
        regression_core = json.loads(regression_receipt_path.read_text(encoding="utf-8"))
        for name, core in [("classification", class_core), ("regression", regression_core)]:
            assert core["matches_historical_predictions"], (name, core)
            assert core["fit_preprocessing_only_train"], (name, core)
            assert not core["test_used_in_core"], (name, core)
        receipt["model_core_receipts"] = {
            "classification": class_core,
            "regression": regression_core,
        }

        test_run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_availability_eda.py"],
            cwd=bundle,
            text=True,
            capture_output=True,
            check=True,
        )
        receipt["unit_tests"] = {"command": "python -m pytest -q tests/test_availability_eda.py", "output": test_run.stdout.strip()}

        # Keep executed notebooks as review artifacts in the actual repository.
        for relative in NOTEBOOKS.values():
            destination = ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / relative, destination)

        figure_dir = ROOT / "reports" / "academic_availability" / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        eda_nb = nbformat.read(ROOT / NOTEBOOKS["eda_availability"], as_version=4)
        figure_paths = []
        for cell in eda_nb.cells:
            for output in cell.get("outputs", []):
                png = output.get("data", {}).get("image/png")
                if png:
                    name = f"eda_availability_{len(figure_paths) + 1}.png"
                    target = figure_dir / name
                    target.write_bytes(base64.b64decode(png))
                    figure_paths.append(target.relative_to(ROOT).as_posix())
        assert len(figure_paths) == 4
        receipt["figures_saved_for_visual_review"] = figure_paths
        receipt["visual_review"] = "PENDING_MANUAL_REVIEW_OF_ALL_EMBEDDED_FIGURES"
        receipt["status"] = "PASS_PORTABLE_RESTART_RUN_ALL"
        receipt["seconds_total"] = round(time.perf_counter() - started, 2)
        receipt["portable_copy_removed_after_check"] = True
        out = ROOT / "reports" / "academic_availability" / "QA.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": receipt["status"],
            "notebooks": {name: {
                "executed_code_cells": item["executed_code_cells"],
                "figures": len(item["embedded_png_figures"]),
                "seconds": item["seconds"],
            } for name, item in receipt["notebooks"].items()},
            "unit_tests": receipt["unit_tests"],
            "eda": {name: eda_metrics[name] for name in [
                "train_rows", "missing_rows", "present_rows", "marginal_gap_pp",
                "exact_cycle_gap_pp", "period_x_q5_gap_pp", "period_x_q10_gap_pp",
                "period_x_q10_coverage",
            ]},
            "seconds_total": receipt["seconds_total"],
        }, ensure_ascii=False, indent=2), flush=True)
    finally:
        if old_data_root is None:
            os.environ.pop("TP2_DATA_DIR", None)
        else:
            os.environ["TP2_DATA_DIR"] = old_data_root
        # TemporaryDirectory cleanup is constrained to the unique path created
        # above and guarded to remain inside the system temporary directory.
        if temp_parent not in bundle.parents:
            raise RuntimeError(f"Refusing to clean unexpected temporary path: {bundle}")
        temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

