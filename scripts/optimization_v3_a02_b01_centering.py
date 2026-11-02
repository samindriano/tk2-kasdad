"""B01 zero-fit residual-centering ablation over frozen A02 geometry.

Two commands are deliberately separate: ``--freeze`` writes the immutable recipe
and input manifest without reading candidate scores; ``--run`` verifies that
freeze and only then computes B01 scores from saved A02 neighbors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.models.classification_a02_neighbors import ResidualNeighborPredictor
from scripts import optimization_v3_a02_reliability_audit as audit
from scripts.optimization_v3_a02_next_decision import exact_top_k, matched_count, metrics as root_metrics


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/a02_b01_centering_v1"
FREEZE_PATH = REPORT / "FREEZE.json"
RESULTS_PATH = REPORT / "RESULTS.json"
QA_PATH = REPORT / "QA.json"
SUMMARY_PATH = REPORT / "SUMMARY.md"
TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
PARENT_QA_PATH = ROOT / "reports/optimization_v3/a02_decision_boundary_audit_v1/QA.json"
PARENT_GROUPS_PATH = ROOT / "reports/optimization_v3/a02_continuation_v1/groups.csv"
CARD_PATH = ROOT / "reports/optimization_v3/a02_next_decision_v1/B01_CARD.md"
DECISION_PATH = ROOT / "reports/optimization_v3/a02_next_decision_v1/DECISION.md"
PLAN_PATH = ROOT / "reports/optimization_v3/a02_next_decision_v1/PLAN.md"
ROOT_RESULTS_PATH = ROOT / "reports/optimization_v3/a02_next_decision_v1/matched_count_results.json"
ROOT_GAP_PATH = ROOT / "reports/optimization_v3/a02_next_decision_v1/confidence_gap_descriptor.json"

EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_LEDGER_SHA256 = "6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147"
ORIGINS = (59, 63, 67, 71, 75)
VIEWS = ("full_six_primary", "new_origin_aggregate", "late", "inherited_H12_L")
ROW_VIEW_NAMES = {
    "full_six_primary": "full_six_primary",
    "new_origin_aggregate": "new_origin_aggregate",
    "late": "late",
    "inherited_H12_L": "inherited_H12_L",
}
SCORE_WINDOWS = {59: (60, 63), 63: (64, 67), 67: (68, 79),
                71: (72, 75), 75: (76, 79)}
FIRST_FOUR = {o: (o + 1, o + 4) for o in ORIGINS}
SHRINKAGE = 4.0
CORRECTION_LIMIT = 0.05
THRESHOLD = 0.50
NEIGHBOR_WIDTH = 32
THREADS = 2
MIN_GROUP_ROWS = 200
MIN_CLASS_ROWS = 30

GROUP_DIMENSIONS = {
    "stream_availability": ["missing", "present"],
    "entry_friction": ["0", "1"],
    "fit_prefix_defined_text_load_quartile": ["Q1", "Q2", "Q3", "Q4"],
}

RECIPE = {
    "name": "B01_RESIDUAL_CENTERING_VS_LOCAL_LABEL_RATE",
    "scope": "adaptive Train-only zero-fit diagnostic; no qualification or upload",
    "neighbors": "the original saved top-32 row positions and cosines; no search",
    "weight": "w = cosine squared",
    "denominator": "4 + sum(w)",
    "original_A02": "d = clip(sum(w * (y_bank - p_bank_oof)) / (4 + sum(w)), -0.05, 0.05)",
    "B01": "d = clip(sum(w * (y_bank - query_baseline)) / (4 + sum(w)), -0.05, 0.05)",
    "B01_label_shuffle": "permute binary bank labels by the verified original shuffled-model residual_assignment, then use the B01 formula",
    "output": "clip(query_baseline + d, 0, 1); hard label is p >= 0.50",
    "missing_zero_neighbor_W1": "exact frozen Sparse25 baseline fallback",
    "query_labels_in_formula": False,
    "new_fit_transform_index_or_neighbor_search": False,
}

GATES = [
    {"id": "G1", "condition": "full-primary Macro F1 gain versus unchanged A02 >= 0.0005"},
    {"id": "G2", "condition": "new-origin aggregate Macro F1 gain versus unchanged A02 >= 0.0005"},
    {"id": "G3", "condition": "late and inherited H12-L Macro F1 gains versus A02 are both nonnegative"},
    {"id": "G4", "condition": "at least 4 of 5 learned primary origins have nonnegative Macro F1 gain versus A02"},
    {"id": "G5", "condition": "full-primary Macro F1 gain versus B01 label-shuffle control >= 0.0002"},
    {"id": "G6", "condition": "no supported subgroup harm versus A02 greater than 0.01 Macro F1 or 0.03 positive recall; n>=200 and both classes>=30"},
    {"id": "G7", "condition": "all source, geometry, label-separation, fallback, independent-metric, zero-fit and ledger integrity checks pass"},
]

SOURCE_PATHS = [
    "scripts/optimization_v3_a02_b01_centering.py",
    "tests/test_optimization_v3_a02_b01_centering.py",
    "scripts/optimization_v3_a02_reliability_audit.py",
    "scripts/optimization_v3_a02_next_decision.py",
    "scripts/optimization_v3_a02_continuation.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "src/models/classification_a02_neighbors.py",
]
CONTROL_PATHS = [
    "experiments/optimization_v3/c19_a02_v2/A02/protocol.json",
    "experiments/optimization_v3/c19_a02_v2/A02/freeze.json",
    "experiments/optimization_v3/a02_continuation_v1/protocol.json",
    "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json",
    "reports/optimization_v3/a02_neighbors_v1/verification.json",
    "reports/optimization_v3/a02_continuation_v1/verification.json",
    "reports/optimization_v3/a02_continuation_v1/groups.csv",
    "reports/optimization_v3/a02_decision_boundary_audit_v1/QA.json",
    "reports/optimization_v3/a02_next_decision_v1/DECISION.md",
    "reports/optimization_v3/a02_next_decision_v1/B01_CARD.md",
    "reports/optimization_v3/a02_next_decision_v1/PLAN.md",
    "reports/optimization_v3/a02_next_decision_v1/QA.json",
    "reports/optimization_v3/a02_next_decision_v1/matched_count_results.json",
    "reports/optimization_v3/a02_next_decision_v1/confidence_gap_descriptor.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n"


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json_bytes(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to overwrite nonidentical immutable output: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.relative_to(ROOT)}")
    return value


def _verify_ledger() -> dict[str, Any]:
    if sha256_file(LEDGER_PATH) != EXPECTED_LEDGER_SHA256:
        raise RuntimeError("Research fit ledger hash differs from the authorized 66-fit checkpoint")
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    terminal = {e.get("fit_id") for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    pending = [e.get("fit_id") for e in starts if e.get("fit_id") not in terminal]
    counts = {"starts": len(starts), "saved": len(saved), "failed": len(failed), "pending": pending}
    if counts != {"starts": 66, "saved": 64, "failed": 2, "pending": []}:
        raise RuntimeError(f"Unexpected frozen fit accounting: {counts}")
    return {"sha256": EXPECTED_LEDGER_SHA256, **counts, "cap": 66, "remaining": 0}


def _parent_integrity() -> dict[str, Any]:
    parent = audit.verify_freezes_and_inputs()
    parent_qa = _read_json(PARENT_QA_PATH)
    if parent_qa.get("status") != "PASS":
        raise RuntimeError("Parent A02 decision-boundary QA is not PASS")
    expected_rows = parent_qa.get("source_hashes", {}).get("row_level_inputs", {})
    verified: dict[str, str] = {}
    for name in ROW_VIEW_NAMES.values():
        relative = f"experiments/runs/optimization_v3_a02_reliability_audit_v1/row_level_{name}.npz"
        path = ROOT / relative
        expected = expected_rows.get(name)
        if not path.is_file() or not expected or sha256_file(path) != expected:
            raise RuntimeError(f"Parent row-level artifact hash mismatch: {relative}")
        verified[relative] = expected
    for relative, expected in parent.get("input_hashes_verified", {}).items():
        lower = relative.lower()
        if relative.startswith("data/raw/") and ("test" in lower or "sample" in lower):
            raise RuntimeError(f"Forbidden raw Test/Sample path is present in A02 frozen inputs: {relative}")
    return {
        "parent_a02_verification": parent,
        "parent_decision_boundary_qa_sha256": sha256_file(PARENT_QA_PATH),
        "parent_row_level_sha256": verified,
        "parent_group_summary_sha256": sha256_file(PARENT_GROUPS_PATH),
    }


def _all_input_paths() -> list[str]:
    values = {"data/raw/classification/Train.csv", LEDGER_PATH.relative_to(ROOT).as_posix()}
    values.update(SOURCE_PATHS)
    values.update(CONTROL_PATHS)
    for path in ROW_VIEW_NAMES.values():
        values.add(f"experiments/runs/optimization_v3_a02_reliability_audit_v1/row_level_{path}.npz")
    for relative in audit.CACHE_PATHS.values():
        path = ROOT / relative
        values.add(path.relative_to(ROOT).as_posix())
        values.add(path.with_suffix(".json").relative_to(ROOT).as_posix())
    for origin in ORIGINS:
        for arm in ("observed", "shuffled"):
            folder = audit._safe_pred_paths(origin, arm)
            for name in ("predictions.npz", "predictions.json", "fit_receipt.json",
                         "fit_state.json", "model.joblib"):
                values.add((folder / name).relative_to(ROOT).as_posix())
    return sorted(values)


def _check_input_scope(paths: list[str]) -> None:
    for relative in paths:
        if relative.startswith("data/raw/"):
            name = Path(relative).name.lower()
            if "test" in name or "sample" in name or "submission" in name:
                raise RuntimeError(f"Raw Test/SampleSubmission input is prohibited: {relative}")
        if "kaggle" in relative.lower() or "submission.csv" in relative.lower():
            raise RuntimeError(f"Kaggle/submission input is prohibited: {relative}")


def freeze_recipe() -> dict[str, Any]:
    if RESULTS_PATH.exists() or QA_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("B01 score outputs already exist; refusing to create a post-score freeze")
    parent = _parent_integrity()
    ledger = _verify_ledger()
    if sha256_file(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Classification Train SHA-256 differs")
    paths = _all_input_paths()
    _check_input_scope(paths)
    missing = [rel for rel in paths if not (ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Required B01 frozen inputs are missing: {missing}")
    hashes = {rel: sha256_file(ROOT / rel) for rel in paths}
    if hashes["data/raw/classification/Train.csv"] != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Train hash changed while building B01 freeze")
    code_hash = hashes["scripts/optimization_v3_a02_b01_centering.py"]
    test_hash = hashes["tests/test_optimization_v3_a02_b01_centering.py"]
    freeze = {
        "status": "B01_RECIPE_FROZEN_PRE_SCORE",
        "source_commit": _git_head(),
        "recipe": RECIPE,
        "gates": GATES,
        "subgroup_dimensions": GROUP_DIMENSIONS,
        "views": list(VIEWS),
        "eligibility_support": {"rows": MIN_GROUP_ROWS, "each_class_rows": MIN_CLASS_ROWS},
        "matched_positive_count": "descriptive only; root helper; per (origin, cycle); exact ties use ascending source row; never selects deployment K",
        "official_train_rows": 50_608,
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "fit_ledger_before_scores": ledger,
        "pre_score_parent_integrity": {
            "a02_parent_qa_status": "PASS",
            "row_level_artifacts_verified": True,
            "no_test_sample_kaggle_or_submission_paths": True,
        },
        "code_sha256": code_hash,
        "test_sha256": test_hash,
        "input_sha256": hashes,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    write_once(FREEZE_PATH, freeze)
    return freeze


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = _read_json(FREEZE_PATH)
    if freeze.get("status") != "B01_RECIPE_FROZEN_PRE_SCORE":
        raise RuntimeError("B01 freeze status is not a pre-score recipe freeze")
    if freeze.get("recipe") != RECIPE or freeze.get("gates") != GATES:
        raise RuntimeError("B01 frozen recipe or seven diagnostic gates differ from code")
    if freeze.get("subgroup_dimensions") != GROUP_DIMENSIONS or freeze.get("views") != list(VIEWS):
        raise RuntimeError("B01 subgroup/view definitions differ from the pre-score freeze")
    if freeze.get("official_train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("B01 frozen official Train hash differs")
    if sha256_file(Path(__file__)) != freeze.get("code_sha256"):
        raise RuntimeError("B01 source code changed after the pre-score freeze")
    test_path = ROOT / "tests/test_optimization_v3_a02_b01_centering.py"
    if sha256_file(test_path) != freeze.get("test_sha256"):
        raise RuntimeError("B01 tests changed after the pre-score freeze")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"], _git_head()],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode != 0:
        raise RuntimeError("B01 frozen source commit is not an ancestor of this checkout")
    paths = list(freeze.get("input_sha256", {}))
    _check_input_scope(paths)
    for relative, expected in freeze.get("input_sha256", {}).items():
        if not (ROOT / relative).is_file() or sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"B01 frozen input changed: {relative}")
    parent = _parent_integrity()
    ledger = _verify_ledger()
    if ledger != freeze.get("fit_ledger_before_scores"):
        raise RuntimeError("B01 fit ledger differs from the pre-score checkpoint")
    if sha256_file(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Classification Train SHA-256 differs")
    return freeze, {"parent": parent, "ledger_before": ledger}


@contextmanager
def forbid_training_and_inference_calls():
    """Fail closed if B01 accidentally fits, transforms, or predicts anything."""
    calls: dict[str, int] = {}
    methods = [
        (ResidualNeighborPredictor, "fit"),
        (ResidualNeighborPredictor, "predict"),
        (TfidfVectorizer, "fit"),
        (TfidfVectorizer, "fit_transform"),
        (TfidfVectorizer, "transform"),
    ]
    originals = []
    def reject(label: str):
        def _blocked(*_args: Any, **_kwargs: Any) -> Any:
            calls[label] = calls.get(label, 0) + 1
            raise RuntimeError(f"B01 zero-fit/no-inference guard blocked {label}")
        return _blocked
    try:
        for cls, name in methods:
            original = getattr(cls, name)
            originals.append((cls, name, original))
            setattr(cls, name, reject(f"{cls.__name__}.{name}"))
        yield calls
    finally:
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)


def load_train() -> pd.DataFrame:
    frame = pd.read_csv(
        TRAIN_PATH,
        usecols=["record_id", "cycle_code", "artifact_signal", "signal_stream",
                 "entry_friction", "text_load"],
        dtype={"record_id": str},
    )
    if len(frame) != 50_608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise RuntimeError("Official Train row identity/count is invalid")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int16)
    frame["artifact_signal"] = pd.to_numeric(frame["artifact_signal"], errors="raise").astype(np.int8)
    frame["entry_friction"] = pd.to_numeric(frame["entry_friction"], errors="raise").astype(np.int8)
    frame["text_load"] = pd.to_numeric(frame["text_load"], errors="raise").astype(np.float64)
    if set(frame["artifact_signal"].unique()) != {0, 1} or set(frame["entry_friction"].unique()) != {0, 1}:
        raise RuntimeError("Official Train binary target/subgroup is invalid")
    if not np.isfinite(frame["text_load"].to_numpy()).all():
        raise RuntimeError("Official Train text_load subgroup input is nonfinite")
    return frame


def fit_prefix_text_load_groups(train: pd.DataFrame, origin: int, rows: np.ndarray) -> np.ndarray:
    cycles = train["cycle_code"].to_numpy(dtype=np.int64)
    prefix = train.loc[cycles <= origin, "text_load"].to_numpy(dtype=np.float64)
    scores = train["text_load"].to_numpy(dtype=np.float64)[rows]
    if len(prefix) == 0 or not np.isfinite(prefix).all() or not np.isfinite(scores).all():
        raise RuntimeError(f"Invalid fit-prefix text_load groups at origin {origin}")
    edges = np.quantile(prefix, [0.25, 0.50, 0.75], method="linear")
    return np.searchsorted(edges, scores, side="right").astype(np.int8)


def compute_from_saved_geometry(saved: Mapping[str, np.ndarray], bank_rows: np.ndarray,
                                bank_labels: np.ndarray, bank_oof_probabilities: np.ndarray,
                                residual_assignment: np.ndarray, *,
                                shrinkage: float = SHRINKAGE,
                                correction_limit: float = CORRECTION_LIMIT) -> dict[str, np.ndarray]:
    """Build original A02 checks plus B01 and its label-shuffle control.

    This function has no query-label argument. Query truth is joined later only
    for metrics; the formula reads query baseline and saved geometry only.
    """
    rows = np.asarray(saved["score_rows"], dtype=np.int64)
    baseline = np.asarray(saved["baseline_prediction"], dtype=np.float64)
    neighbors = np.asarray(saved["neighbor_source_rows"], dtype=np.int64)
    similarities = np.asarray(saved["neighbor_similarities"], dtype=np.float64)
    counts = np.asarray(saved["neighbor_count"], dtype=np.int64)
    missing = np.asarray(saved["missing_mask"], dtype=bool)
    bank_rows = np.asarray(bank_rows, dtype=np.int64)
    bank_labels = np.asarray(bank_labels, dtype=np.int8)
    bank_p = np.asarray(bank_oof_probabilities, dtype=np.float64)
    assignment = np.asarray(residual_assignment, dtype=np.int64)
    n = len(rows)
    if (baseline.shape != (n,) or counts.shape != (n,) or missing.shape != (n,)
            or neighbors.ndim != 2 or similarities.shape != neighbors.shape or neighbors.shape[0] != n):
        raise ValueError("Saved B01 geometry arrays are not aligned")
    if (not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 1))
            or not np.isfinite(similarities).all() or np.any(similarities < 0)):
        raise ValueError("B01 query probabilities/similarities are invalid")
    if (len(np.unique(bank_rows)) != len(bank_rows) or bank_labels.shape != bank_rows.shape
            or bank_p.shape != bank_rows.shape or not np.isin(bank_labels, [0, 1]).all()
            or not np.isfinite(bank_p).all() or np.any((bank_p < 0) | (bank_p > 1))):
        raise ValueError("B01 bank rows, labels, or OOF probabilities are invalid")
    if (assignment.shape != bank_rows.shape or not np.array_equal(np.sort(assignment), np.arange(len(bank_rows)))):
        raise ValueError("B01 residual assignment is not a bank permutation")
    if np.any(counts < 0) or np.any(counts > neighbors.shape[1]):
        raise ValueError("B01 saved neighbor count is outside array width")
    if neighbors.shape[1] != NEIGHBOR_WIDTH:
        raise ValueError("B01 requires the frozen 32-neighbor geometry width")

    lookup = {int(row): index for index, row in enumerate(bank_rows)}
    original_residuals = bank_labels.astype(np.float64) - bank_p
    label_shuffle = bank_labels[assignment]
    b01 = baseline.copy()
    b01_control = baseline.copy()
    a02_rebuilt = baseline.copy()
    a02_shuffle_rebuilt = baseline.copy()
    a02_raw = np.zeros(n, dtype=np.float64)
    a02_shuffle_raw = np.zeros(n, dtype=np.float64)
    b01_raw = np.zeros(n, dtype=np.float64)
    b01_control_raw = np.zeros(n, dtype=np.float64)

    for start in range(0, n, 512):
        stop = min(n, start + 512)
        for i in range(start, stop):
            count = int(counts[i])
            if np.any(neighbors[i, count:] != -1) or np.any(similarities[i, count:] != 0):
                raise ValueError("B01 neighbor padding differs from the frozen contract")
            if missing[i] and count != 0:
                raise ValueError("A missing-stream query has saved neighbors")
            if count == 0:
                continue
            selected_rows = neighbors[i, :count]
            if len(np.unique(selected_rows)) != count or np.any(selected_rows == rows[i]):
                raise ValueError("B01 saved neighbors contain duplicates or the query itself")
            try:
                idx = np.asarray([lookup[int(row)] for row in selected_rows], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"B01 saved neighbor is outside the frozen bank: {exc}") from exc
            sim = similarities[i, :count]
            if np.any(sim <= 0) or np.any(sim > 1.0 + 1e-12) or np.any(np.diff(sim) > 1e-15):
                raise ValueError("B01 saved neighbor similarities violate frozen positive/sorted contract")
            weights = sim * sim
            denominator = shrinkage + float(weights.sum())
            if denominator <= 0:
                raise ValueError("B01 shrinkage denominator is nonpositive")
            raw_a02 = float(np.dot(weights, original_residuals[idx]) / denominator)
            raw_shuffle = float(np.dot(weights, original_residuals[assignment[idx]]) / denominator)
            raw_b01 = float(np.dot(weights, bank_labels[idx].astype(np.float64) - baseline[i]) / denominator)
            raw_control = float(np.dot(weights, label_shuffle[idx].astype(np.float64) - baseline[i]) / denominator)
            a02_raw[i] = raw_a02
            a02_shuffle_raw[i] = raw_shuffle
            b01_raw[i] = raw_b01
            b01_control_raw[i] = raw_control
            a02_rebuilt[i] = np.clip(baseline[i] + np.clip(raw_a02, -correction_limit, correction_limit), 0.0, 1.0)
            a02_shuffle_rebuilt[i] = np.clip(baseline[i] + np.clip(raw_shuffle, -correction_limit, correction_limit), 0.0, 1.0)
            b01[i] = np.clip(baseline[i] + np.clip(raw_b01, -correction_limit, correction_limit), 0.0, 1.0)
            b01_control[i] = np.clip(baseline[i] + np.clip(raw_control, -correction_limit, correction_limit), 0.0, 1.0)

    if not (np.array_equal(b01[missing], baseline[missing])
            and np.array_equal(b01_control[missing], baseline[missing])
            and np.array_equal(a02_rebuilt[missing], baseline[missing])
            and np.array_equal(a02_shuffle_rebuilt[missing], baseline[missing])):
        raise RuntimeError("B01 missing-stream rows are not exact baseline fallback")
    for values in (b01, b01_control, a02_rebuilt, a02_shuffle_rebuilt):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise RuntimeError("B01 formula produced nonfinite/out-of-range probabilities")
    return {
        "a02_rebuilt": a02_rebuilt,
        "a02_shuffle_rebuilt": a02_shuffle_rebuilt,
        "b01": b01,
        "b01_label_shuffle": b01_control,
        "a02_raw": a02_raw,
        "a02_shuffle_raw": a02_shuffle_raw,
        "b01_raw": b01_raw,
        "b01_label_shuffle_raw": b01_control_raw,
        "permuted_bank_labels": label_shuffle,
    }


def _validate_geometry(arrays: dict[str, np.ndarray], model: Any,
                       train: pd.DataFrame, origin: int) -> None:
    rows = arrays["score_rows"].astype(np.int64)
    cycles = arrays["score_cycles"].astype(np.int64)
    counts = arrays["neighbor_count"].astype(np.int64)
    neighbors = arrays["neighbor_source_rows"].astype(np.int64)
    ids = arrays["neighbor_ids"].astype(str)
    sims = arrays["neighbor_similarities"].astype(np.float64)
    missing = arrays["missing_mask"].astype(bool)
    if (neighbors.shape != (len(rows), NEIGHBOR_WIDTH) or ids.shape != neighbors.shape
            or sims.shape != neighbors.shape or counts.shape != (len(rows),)):
        raise RuntimeError(f"Saved B01 top-32 geometry has an invalid shape at E{origin}")
    bank_lookup = {int(row): index for index, row in enumerate(model.bank_row_positions)}
    for i, count in enumerate(counts):
        n = int(count)
        if n > 0:
            chosen = neighbors[i, :n]
            if np.any(cycles[i] <= model.bank_cycles[[bank_lookup[int(row)] for row in chosen]]):
                raise RuntimeError(f"B01 neighbor is not strictly historical at E{origin}")
            indices = np.asarray([bank_lookup[int(row)] for row in chosen], dtype=np.int64)
            if [audit.normalize_id(v) for v in ids[i, :n]] != [audit.normalize_id(model.bank_ids[j]) for j in indices]:
                raise RuntimeError(f"B01 saved neighbor IDs do not match bank rows at E{origin}")
            if any(sims[i, j] == sims[i, j + 1] and chosen[j] > chosen[j + 1]
                   for j in range(n - 1)):
                raise RuntimeError(f"B01 neighbor tie ordering differs from source-row tie rule at E{origin}")
        if np.any(ids[i, n:] != ""):
            raise RuntimeError(f"B01 saved neighbor ID padding differs from frozen contract at E{origin}")
        if missing[i] and n != 0:
            raise RuntimeError(f"Missing-stream row has neighbors at E{origin}")
    if not np.array_equal(arrays["entry_friction"].astype(np.int8),
                          train["entry_friction"].to_numpy(dtype=np.int8)[rows]):
        raise RuntimeError(f"Saved entry-friction subgroup differs from official Train at E{origin}")
    expected_group = fit_prefix_text_load_groups(train, origin, rows)
    if not np.array_equal(arrays["text_load_group"].astype(np.int8), expected_group):
        raise RuntimeError(f"Saved fit-prefix text-load subgroup differs from recipe at E{origin}")


def _load_origin(origin: int, train: pd.DataFrame, caches: dict[int, dict[str, np.ndarray]],
                 freeze_state: dict[str, Any]) -> dict[str, np.ndarray]:
    bank = audit.expected_bank(origin, caches, train,
                               freeze_state["pilot_protocol"],
                               freeze_state["continuation_protocol"])
    observed_path = audit._safe_pred_paths(origin, "observed")
    shuffled_path = audit._safe_pred_paths(origin, "shuffled")
    observed, _, obs_receipt, obs_state = audit.load_prediction(
        observed_path, origin, "observed", train, caches, freeze_state)
    shuffled, _, shuf_receipt, shuf_state = audit.load_prediction(
        shuffled_path, origin, "shuffled", train, caches, freeze_state)
    for key in ("score_rows", "score_record_ids", "score_cycles", "baseline_prediction",
                "missing_mask", "neighbor_count", "neighbor_source_rows", "neighbor_ids",
                "neighbor_similarities"):
        if not np.array_equal(observed[key], shuffled[key]):
            raise RuntimeError(f"Observed/shuffled B01 geometry differs at E{origin}/{key}")

    obs_model = audit.load_and_verify_model(observed_path, origin, "observed", obs_receipt,
                                            obs_state, bank, train, caches)
    bank_rows = obs_model.bank_row_positions.astype(np.int64, copy=True)
    bank_labels = train["artifact_signal"].to_numpy(dtype=np.int8)[bank_rows]
    bank_p = obs_model.bank_oof_probabilities.astype(np.float64, copy=True)
    expected_residual = bank_labels.astype(np.float64) - bank_p
    if (not np.array_equal(obs_model.bank_residuals, expected_residual)
            or not np.array_equal(obs_model.bank_residuals + bank_p, bank_labels.astype(np.float64))):
        raise RuntimeError(f"Original A02 residual plus OOF probability does not reconstruct bank labels at E{origin}")
    _validate_geometry(observed, obs_model, train, origin)
    observed_copy = {k: np.asarray(v).copy() for k, v in observed.items()}
    del obs_model
    gc.collect()

    shuf_model = audit.load_and_verify_model(shuffled_path, origin, "shuffled", shuf_receipt,
                                             shuf_state, bank, train, caches)
    assignment = shuf_model.residual_assignment.astype(np.int64, copy=True)
    if (assignment.shape != bank_labels.shape
            or not np.array_equal(np.sort(assignment), np.arange(len(bank_labels)))
            or not np.array_equal(shuf_model.bank_residuals, expected_residual[assignment])):
        raise RuntimeError(f"Frozen shuffled residual assignment is invalid at E{origin}")
    _validate_geometry(shuffled, shuf_model, train, origin)
    outputs = compute_from_saved_geometry(
        observed_copy, bank_rows, bank_labels, bank_p, assignment)
    if not np.allclose(outputs["a02_raw"], observed["correction_unclipped"], rtol=0, atol=1e-15):
        raise RuntimeError(f"Reconstructed original A02 raw correction differs at E{origin}")
    if not np.allclose(outputs["a02_shuffle_raw"], shuffled["correction_unclipped"], rtol=0, atol=1e-15):
        raise RuntimeError(f"Reconstructed original shuffled raw correction differs at E{origin}")
    if (not np.array_equal(outputs["a02_rebuilt"], observed["prediction"])
            or not np.array_equal(outputs["a02_shuffle_rebuilt"], shuffled["prediction"])):
        raise RuntimeError(f"Original A02/shuffle probabilities do not reconstruct at E{origin}")
    if (not np.array_equal(np.clip(outputs["a02_raw"], -CORRECTION_LIMIT, CORRECTION_LIMIT), observed["correction"])
            or not np.array_equal(np.clip(outputs["a02_shuffle_raw"], -CORRECTION_LIMIT, CORRECTION_LIMIT), shuffled["correction"])):
        raise RuntimeError(f"Original A02/shuffle correction cap does not reconstruct at E{origin}")
    outputs.update({
        "rows": observed["score_rows"].astype(np.int64),
        "cycles": observed["score_cycles"].astype(np.int16),
        "baseline": observed["baseline_prediction"].astype(np.float64),
        "observed": observed["prediction"].astype(np.float64),
        "shuffled": shuffled["prediction"].astype(np.float64),
        "missing": observed["missing_mask"].astype(bool),
        "entry_friction": observed["entry_friction"].astype(np.int8),
        "text_load_group": observed["text_load_group"].astype(np.int8),
        "neighbor_count": observed["neighbor_count"].astype(np.int16),
    })
    if not np.array_equal(outputs["permuted_bank_labels"], bank_labels[assignment]):
        raise RuntimeError("B01 negative control did not permute the binary bank labels by the frozen assignment")
    del shuf_model, observed, shuffled, observed_copy
    gc.collect()
    return outputs


def _load_views(train: pd.DataFrame, per_origin: dict[int, dict[str, np.ndarray]],
                parent_qa: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    expected_hashes = parent_qa["source_hashes"]["row_level_inputs"]
    all_y = train["artifact_signal"].to_numpy(dtype=np.int8)
    all_cycles = train["cycle_code"].to_numpy(dtype=np.int16)
    all_missing = train["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    for view in VIEWS:
        name = ROW_VIEW_NAMES[view]
        path = ROOT / "experiments/runs/optimization_v3_a02_reliability_audit_v1" / f"row_level_{name}.npz"
        if sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"B01 parent row-level hash changed: {view}")
        with np.load(path, allow_pickle=False) as archive:
            frame = {k: np.asarray(archive[k]).copy() for k in
                     ("rows", "cycles", "origin", "y", "missing", "baseline", "observed", "shuffled")}
        rows = frame["rows"].astype(np.int64)
        cycles = frame["cycles"].astype(np.int16)
        origins = frame["origin"].astype(np.int16)
        if (rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(rows < 0)
                or np.any(rows >= len(train)) or not np.array_equal(all_cycles[rows], cycles)
                or not np.array_equal(all_y[rows], frame["y"].astype(np.int8))
                or not np.array_equal(all_missing[rows], frame["missing"].astype(bool))):
            raise RuntimeError(f"B01 parent row/cycle/target/availability join failed: {view}")
        for arm in ("baseline", "observed", "shuffled"):
            p = frame[arm].astype(np.float64)
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise RuntimeError(f"Invalid parent {arm} probabilities in {view}")
            frame[arm] = p
        b01 = frame["baseline"].copy()
        b01_control = frame["baseline"].copy()
        text_groups = np.full(len(rows), -1, dtype=np.int8)
        for origin in ORIGINS:
            mask = origins == origin
            if not mask.any():
                continue
            record = per_origin[origin]
            r = record["rows"]
            loc = np.searchsorted(r, rows[mask])
            if np.any(loc >= len(r)) or not np.array_equal(r[loc], rows[mask]):
                raise RuntimeError(f"B01 origin prediction rows do not cover view {view}, E{origin}")
            low, high = (68, 71) if view == "full_six_primary" and origin == 67 else (
                (68, 79) if view == "inherited_H12_L" and origin == 67 else FIRST_FOUR[origin])
            use = (record["cycles"][loc] >= low) & (record["cycles"][loc] <= high)
            if not np.all(use):
                raise RuntimeError(f"Unexpected origin/cycle scope in B01 view {view}/E{origin}")
            b01[mask] = record["b01"][loc]
            b01_control[mask] = record["b01_label_shuffle"][loc]
            text_groups[mask] = record["text_load_group"][loc]
            if not np.array_equal(frame["observed"][mask], record["observed"][loc]):
                raise RuntimeError(f"Parent A02 observed probabilities do not align in {view}/E{origin}")
            if not np.array_equal(frame["shuffled"][mask], record["shuffled"][loc]):
                raise RuntimeError(f"Parent A02 shuffled probabilities do not align in {view}/E{origin}")
        w1 = origins < 0
        if w1.any():
            if (not np.array_equal(frame["observed"][w1], frame["baseline"][w1])
                    or not np.array_equal(frame["shuffled"][w1], frame["baseline"][w1])):
                raise RuntimeError("W1 is not exact Sparse25 fallback")
            text_groups[w1] = fit_prefix_text_load_groups(train, 55, rows[w1])
        if np.any(text_groups < 0) or np.any(text_groups > 3):
            raise RuntimeError(f"Text-load subgroup values are incomplete in {view}")
        if not (np.array_equal(b01[frame["missing"].astype(bool)], frame["baseline"][frame["missing"].astype(bool)])
                and np.array_equal(b01_control[frame["missing"].astype(bool)], frame["baseline"][frame["missing"].astype(bool)])
                and np.array_equal(b01[w1], frame["baseline"][w1])
                and np.array_equal(b01_control[w1], frame["baseline"][w1])):
            raise RuntimeError(f"B01 missing-stream/W1 fallback differs from Sparse25 in {view}")
        frame.update({
            "rows": rows, "cycles": cycles, "origin": origins,
            "y": frame["y"].astype(np.int8), "missing": frame["missing"].astype(bool),
            "b01": b01, "b01_label_shuffle": b01_control,
            "entry_friction": train["entry_friction"].to_numpy(dtype=np.int8)[rows],
            "stream_group": np.where(frame["missing"], "missing", "present"),
            "text_load_group": text_groups,
        })
        result[view] = frame

    # W1 subgroup mapping is the historical origin-55 fit-prefix definition.
    w1_frame = result["full_six_primary"]
    w1_mask = w1_frame["origin"] < 0
    hist = pd.read_csv(PARENT_GROUPS_PATH, dtype={"group": str})
    hist_w1 = hist[(hist["view"] == "W1") & (hist["arm"] == "observed")]
    checks = []
    for field, groups in GROUP_DIMENSIONS.items():
        for group in groups:
            if field == "stream_availability":
                mask = w1_mask & (w1_frame["stream_group"] == group)
            elif field == "entry_friction":
                mask = w1_mask & (w1_frame["entry_friction"].astype(str) == group)
            else:
                code = int(group[1:]) - 1
                mask = w1_mask & (w1_frame["text_load_group"] == code)
            record = hist_w1[(hist_w1["field"] == field) & (hist_w1["group"].astype(str) == group)]
            if len(record) != 1:
                raise RuntimeError(f"Historical W1 subgroup reference missing/duplicated: {field}/{group}")
            row = record.iloc[0]
            n = int(mask.sum())
            c0 = int(np.sum(mask & (w1_frame["y"] == 0)))
            c1 = int(np.sum(mask & (w1_frame["y"] == 1)))
            if (n, c0, c1) != (int(row["rows"]), int(row["class_0_rows"]), int(row["class_1_rows"])):
                raise RuntimeError(f"B01 W1 subgroup mapping differs from historical definition: {field}/{group}")
            checks.append(True)
    if len(checks) != 8 or not all(checks):
        raise RuntimeError("B01 W1 historical subgroup reconstruction failed")
    return result


def _manual_confusion(y: np.ndarray, labels: np.ndarray) -> dict[str, int | float]:
    y = np.asarray(y, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    if y.shape != labels.shape or not np.isin(y, [0, 1]).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Aligned binary target and hard labels required")
    tn = int(np.sum((y == 0) & (labels == 0)))
    fp = int(np.sum((y == 0) & (labels == 1)))
    fn = int(np.sum((y == 1) & (labels == 0)))
    tp = int(np.sum((y == 1) & (labels == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {"n": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "predicted_positive_count": tp + fp,
            "positive_recall": recall, "macro_f1": (f0 + f1) / 2.0}


def _probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probability metric inputs are misaligned or invalid")
    labels = (p >= THRESHOLD).astype(np.int8)
    result = _manual_confusion(y, labels)
    sklearn_cm = confusion_matrix(y, labels, labels=[0, 1])
    sklearn_f1 = float(f1_score(y, labels, labels=[0, 1], average="macro", zero_division=0))
    independent = {"tn": int(sklearn_cm[0, 0]), "fp": int(sklearn_cm[0, 1]),
                   "fn": int(sklearn_cm[1, 0]), "tp": int(sklearn_cm[1, 1]),
                   "macro_f1": sklearn_f1}
    for key in ("tn", "fp", "fn", "tp"):
        if result[key] != independent[key]:
            raise RuntimeError(f"Independent sklearn confusion replay differs at {key}")
    if not np.isclose(result["macro_f1"], independent["macro_f1"], rtol=0, atol=1e-15):
        raise RuntimeError("Independent sklearn Macro F1 replay differs")
    eps15 = 1e-15
    eps_native = np.finfo(np.float64).eps
    clip15 = np.clip(p, eps15, 1.0 - eps15)
    clipnative = np.clip(p, eps_native, 1.0 - eps_native)
    positive = y == 1
    logloss15 = -float(np.mean(positive * np.log(clip15) + (~positive) * np.log1p(-clip15)))
    logloss_native = -float(np.mean(positive * np.log(clipnative) + (~positive) * np.log1p(-clipnative)))
    result.update({
        "predicted_positive_rate": float(labels.mean()) if len(labels) else None,
        "positive_count_rule": "p >= 0.50",
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(np.mean(np.square(p - y))),
        "logloss_clip_1e-15": logloss15,
        "logloss_clip_native_float64_epsilon": logloss_native,
        "exact_zero_predictions": int(np.sum(p == 0.0)),
        "exact_one_predictions": int(np.sum(p == 1.0)),
        "wrong_exact_zero_endpoints": int(np.sum((p == 0.0) & (y == 1))),
        "wrong_exact_one_endpoints": int(np.sum((p == 1.0) & (y == 0))),
        "independent_confusion_replay_pass": True,
    })
    return result


def _arms(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"Sparse25": frame["baseline"], "A02": frame["observed"],
            "A02_original_shuffle": frame["shuffled"], "B01": frame["b01"],
            "B01_bank_label_shuffle": frame["b01_label_shuffle"]}


def _matched_count_diagnostics(frame: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reuse the verified target-free root selector; these are not gate metrics."""
    labels = frame["y"]
    policies = {
        "A02_default": (frame["observed"] >= THRESHOLD).astype(np.int8),
        "Sparse25_default": (frame["baseline"] >= THRESHOLD).astype(np.int8),
        "B01_default": (frame["b01"] >= THRESHOLD).astype(np.int8),
        "B01_shuffle_default": (frame["b01_label_shuffle"] >= THRESHOLD).astype(np.int8),
        "B01_rank_at_A02_count": matched_count(frame, "b01", "observed", "cycle"),
        "Sparse25_rank_at_A02_count": matched_count(frame, "baseline", "observed", "cycle"),
        "B01_shuffle_rank_at_A02_count": matched_count(frame, "b01_label_shuffle", "observed", "cycle"),
        "A02_rank_at_Sparse25_count": matched_count(frame, "observed", "baseline", "cycle"),
        "B01_rank_at_Sparse25_count": matched_count(frame, "b01", "baseline", "cycle"),
        "B01_shuffle_rank_at_Sparse25_count": matched_count(frame, "b01_label_shuffle", "baseline", "cycle"),
        "A02_rank_at_B01_count": matched_count(frame, "observed", "b01", "cycle"),
        "Sparse25_rank_at_B01_count": matched_count(frame, "baseline", "b01", "cycle"),
        "B01_shuffle_rank_at_B01_count": matched_count(frame, "b01_label_shuffle", "b01", "cycle"),
    }
    if (policies["B01_rank_at_A02_count"].sum() != policies["A02_default"].sum()
            or policies["Sparse25_rank_at_A02_count"].sum() != policies["A02_default"].sum()
            or policies["B01_rank_at_Sparse25_count"].sum() != policies["Sparse25_default"].sum()
            or policies["A02_rank_at_Sparse25_count"].sum() != policies["Sparse25_default"].sum()
            or policies["A02_rank_at_B01_count"].sum() != policies["B01_default"].sum()
            or policies["Sparse25_rank_at_B01_count"].sum() != policies["B01_default"].sum()):
        raise RuntimeError("B01 descriptive matched-count selector did not preserve positive counts")
    return {name: root_metrics(labels, hard) for name, hard in policies.items()}


def _scope_records(views: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    scopes = dict(views)
    primary = views["full_six_primary"]
    for origin in ORIGINS:
        mask = (primary["origin"] == origin) & (primary["cycles"] >= FIRST_FOUR[origin][0]) & (primary["cycles"] <= FIRST_FOUR[origin][1])
        if not mask.any():
            raise RuntimeError(f"Missing B01 first-four primary rows for E{origin}")
        scopes[f"E{origin}_first4"] = {key: value[mask] for key, value in primary.items()
                                         if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape}
    return scopes


def _evaluate(views: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scopes = _scope_records(views)
    metrics: dict[str, Any] = {}
    for scope, frame in scopes.items():
        y = frame["y"].astype(np.int8)
        arm_stats = {name: _probability_metrics(y, p) for name, p in _arms(frame).items()}
        deltas = {}
        for name, stats in arm_stats.items():
            deltas[name] = {
                "macro_f1_vs_A02": stats["macro_f1"] - arm_stats["A02"]["macro_f1"],
                "macro_f1_vs_Sparse25": stats["macro_f1"] - arm_stats["Sparse25"]["macro_f1"],
            }
        metrics[scope] = {"rows": len(y), "arms": arm_stats, "deltas": deltas,
                          "matched_positive_count_diagnostics": _matched_count_diagnostics(frame)}

    subgroups: dict[str, Any] = {}
    for scope, frame in scopes.items():
        fields = {
            "stream_availability": frame["stream_group"].astype(str),
            "entry_friction": frame["entry_friction"].astype(str),
            "fit_prefix_defined_text_load_quartile": np.asarray([f"Q{int(v)+1}" for v in frame["text_load_group"]]),
        }
        for field, group_values in fields.items():
            for group in GROUP_DIMENSIONS[field]:
                mask = group_values == group
                if not mask.any():
                    continue
                y = frame["y"][mask].astype(np.int8)
                stats = {arm: _probability_metrics(y, prob[mask]) for arm, prob in _arms(frame).items()}
                n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
                supported = len(y) >= MIN_GROUP_ROWS and n0 >= MIN_CLASS_ROWS and n1 >= MIN_CLASS_ROWS
                record = {
                    "scope": scope, "field": field, "group": group,
                    "n": len(y), "class_0": n0, "class_1": n1, "supported": supported,
                    "arms": stats,
                    "b01_macro_f1_delta_vs_a02": stats["B01"]["macro_f1"] - stats["A02"]["macro_f1"],
                    "b01_positive_recall_delta_vs_a02": (
                        None if stats["B01"]["positive_recall"] is None or stats["A02"]["positive_recall"] is None
                        else stats["B01"]["positive_recall"] - stats["A02"]["positive_recall"]),
                    "b01_macro_f1_delta_vs_sparse25": stats["B01"]["macro_f1"] - stats["Sparse25"]["macro_f1"],
                    "b01_positive_recall_delta_vs_sparse25": (
                        None if stats["B01"]["positive_recall"] is None or stats["Sparse25"]["positive_recall"] is None
                        else stats["B01"]["positive_recall"] - stats["Sparse25"]["positive_recall"]),
                }
                subgroups[f"{scope}|{field}|{group}"] = record

    gate_info: dict[str, Any] = {}
    full = metrics["full_six_primary"]
    new = metrics["new_origin_aggregate"]
    late = metrics["late"]
    h12 = metrics["inherited_H12_L"]
    origin_deltas = [metrics[f"E{o}_first4"]["deltas"]["B01"]["macro_f1_vs_A02"] for o in ORIGINS]
    supported = [row for row in subgroups.values() if row["supported"]]
    failing_subgroups = [row for row in supported if
                         (row["b01_macro_f1_delta_vs_a02"] < -0.01)
                         or (row["b01_positive_recall_delta_vs_a02"] is not None
                             and row["b01_positive_recall_delta_vs_a02"] < -0.03)]
    checks = {
        "G1_full_primary_gain_vs_A02_ge_0005": full["deltas"]["B01"]["macro_f1_vs_A02"] >= 0.0005,
        "G2_new_origin_gain_vs_A02_ge_0005": new["deltas"]["B01"]["macro_f1_vs_A02"] >= 0.0005,
        "G3_late_and_H12_gains_vs_A02_nonnegative": (
            late["deltas"]["B01"]["macro_f1_vs_A02"] >= 0.0
            and h12["deltas"]["B01"]["macro_f1_vs_A02"] >= 0.0),
        "G4_at_least_4_of_5_origins_nonnegative_vs_A02": sum(x >= 0 for x in origin_deltas) >= 4,
        "G5_full_primary_gain_vs_B01_label_shuffle_ge_0002": (
            full["arms"]["B01"]["macro_f1"] - full["arms"]["B01_bank_label_shuffle"]["macro_f1"] >= 0.0002),
        "G6_no_supported_subgroup_harm_over_fixed_guards": len(failing_subgroups) == 0,
    }
    gate_info = {
        "checks": checks,
        "origin_macro_f1_deltas_vs_A02": {str(o): float(x) for o, x in zip(ORIGINS, origin_deltas)},
        "nonnegative_origin_count": int(sum(x >= 0 for x in origin_deltas)),
        "supported_subgroup_records": len(supported),
        "failing_subgroups": failing_subgroups,
        "subgroup_support_rule": {"n": MIN_GROUP_ROWS, "each_class_n": MIN_CLASS_ROWS},
    }
    return metrics, subgroups, gate_info


def _all_source_hashes(freeze: dict[str, Any]) -> dict[str, str]:
    return {rel: freeze["input_sha256"][rel] for rel in sorted(freeze["input_sha256"])}


def run_b01() -> dict[str, Any]:
    freeze, pre = verify_freeze()
    before_ledger = pre["ledger_before"]
    parent = pre["parent"]
    if RESULTS_PATH.exists() or QA_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("B01 result output already exists; refusing to overwrite or score twice")
    with threadpool_limits(limits=THREADS), forbid_training_and_inference_calls() as blocked:
        train = load_train()
        freeze_state = audit.verify_freezes_and_inputs()
        caches = audit.load_c16_caches(train, freeze_state)
        per_origin: dict[int, dict[str, np.ndarray]] = {}
        for origin in ORIGINS:
            per_origin[origin] = _load_origin(origin, train, caches, freeze_state)
        del caches
        gc.collect()
        parent_qa = _read_json(PARENT_QA_PATH)
        views = _load_views(train, per_origin, parent_qa)
        metrics, subgroups, gate_info = _evaluate(views)
        if any(blocked.values()):
            raise RuntimeError(f"A fit/transform/predict call was blocked: {blocked}")

    after_ledger = _verify_ledger()
    if after_ledger != before_ledger:
        raise RuntimeError("Fit ledger changed while B01 zero-fit diagnostics were running")
    full = metrics["full_six_primary"]
    new = metrics["new_origin_aggregate"]
    late = metrics["late"]
    h12 = metrics["inherited_H12_L"]
    gate_info["checks"]["G7_source_geometry_label_separation_zero_fit_ledger_integrity_pass"] = bool(
        parent_qa.get("status") == "PASS"
        and sha256_file(TRAIN_PATH) == EXPECTED_TRAIN_SHA256
        and before_ledger["sha256"] == after_ledger["sha256"] == EXPECTED_LEDGER_SHA256
        and before_ledger["starts"] == after_ledger["starts"] == 66
        and before_ledger["saved"] == after_ledger["saved"] == 64
        and before_ledger["failed"] == after_ledger["failed"] == 2
        and not any(blocked.values())
    )
    gates_pass = all(gate_info["checks"].values())
    result = {
        "status": "ELIGIBLE_FOR_REVIEW" if gates_pass else "DIAGNOSTIC_GATE_FAILED",
        "qualification": "DIAGNOSTIC_ONLY_NOT_QUALIFIED",
        "formula": RECIPE,
        "views_and_arms": metrics,
        "subgroups": subgroups,
        "gate_evaluation": gate_info,
        "interpretation": "Adaptive reused Train diagnostics; not a new holdout, not causal, and not a deployment-count selection.",
        "provenance": {
            "source_commit": freeze["source_commit"],
            "freeze_sha256": sha256_file(FREEZE_PATH),
            "code_sha256": freeze["code_sha256"],
            "official_train_sha256": EXPECTED_TRAIN_SHA256,
            "input_sha256": _all_source_hashes(freeze),
            "ledger_before": before_ledger,
            "ledger_after": after_ledger,
            "fit_transform_predict_calls": blocked,
            "query_labels_passed_to_formula": False,
            "query_labels_used_only_for_metrics": True,
            "test_sample_kaggle_or_submission_access": False,
            "row_level_b01_predictions_written": False,
            "source_row_ids_in_tracked_outputs": False,
        },
    }
    qa = {
        "status": "PASS" if gate_info["checks"]["G7_source_geometry_label_separation_zero_fit_ledger_integrity_pass"] else "FAIL",
        "checks": {
            "pre_score_recipe_freeze_code_gate_and_input_hashes_verified": True,
            "parent_a02_freezes_and_decision_boundary_qa_pass": True,
            "official_train_hash_and_row_cycle_target_alignment_pass": True,
            "parent_row_level_artifact_hashes_verified": True,
            "all_five_observed_and_shuffled_model_receipts_and_states_verified": True,
            "observed_shuffled_saved_geometry_identical": True,
            "top32_ids_rows_cycles_padding_tie_order_and_historicality_verified": True,
            "original_bank_labels_reconstructed_from_oof_residual_plus_probability": True,
            "label_shuffle_uses_verified_bank_label_assignment_not_residual_label_mix": True,
            "original_a02_and_shuffle_predictions_reconstructed_exactly": True,
            "b01_and_label_shuffle_missing_zero_neighbor_and_w1_fallback_exact": True,
            "legacy_subgroup_mapping_reconstructed_and_supported_denominators_recorded": True,
            "independent_confusion_and_macro_f1_replay_pass": True,
            "query_labels_not_passed_to_formula": True,
            "all_seven_diagnostic_gates_reported": len(gate_info["checks"]) == 7,
            "no_fit_transform_predict_or_neighbor_search_calls": not any(blocked.values()),
            "fit_ledger_sha_and_66_of_66_accounting_unchanged": before_ledger == after_ledger,
            "no_test_sample_kaggle_ui_or_submission_access": True,
            "no_row_level_b01_prediction_artifact_written": True,
        },
        "fit_calls": blocked,
        "fit_ledger_before": before_ledger,
        "fit_ledger_after": after_ledger,
        "source_commit": freeze["source_commit"],
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "input_sha256": _all_source_hashes(freeze),
        "result_sha256": None,
    }
    if not all(qa["checks"].values()):
        raise RuntimeError(f"B01 independent QA contains a failed check: {qa['checks']}")
    summary = _make_summary(result)
    REPORT.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json_bytes(result), encoding="utf-8", newline="\n")
    SUMMARY_PATH.write_text(summary, encoding="utf-8", newline="\n")
    qa["result_sha256"] = sha256_file(RESULTS_PATH)
    qa["summary_sha256"] = sha256_file(SUMMARY_PATH)
    QA_PATH.write_text(json_bytes(qa), encoding="utf-8", newline="\n")
    return result


def _make_summary(result: dict[str, Any]) -> str:
    metrics = result["views_and_arms"]
    gates = result["gate_evaluation"]
    lines = [
        "# B01 — residual-centering versus local-label-rate ablation",
        "",
        f"**Status:** {result['status']}<br>",
        f"**Qualification:** {result['qualification']}<br>",
        "**Training and inference fits:** 0.<br>",
        "**QA:** source, formula, independent confusion replay, subgroup and ledger checks passed.",
        "",
        "B01 replaces only the bank OOF probability centering term with the query Sparse25 probability. It uses the same saved top-32 neighbors and squared-cosine weights, shrinkage 4, ±0.05 correction cap, exact fallback and 0.50 threshold. The label-shuffle control permutes binary bank labels with the verified original assignment. No query labels enter either formula.",
        "",
        "## Main results",
        "",
        "| View | n | Sparse25 F1 | A02 F1 | B01 F1 | B01 label-shuffle F1 | Δ B01 vs A02 | Δ B01 vs shuffle | B01 positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("full_six_primary", "new_origin_aggregate", "late", "inherited_H12_L"):
        view = metrics[name]
        arms = view["arms"]
        lines.append(
            f"| {name} | {view['rows']} | {arms['Sparse25']['macro_f1']:.6f} | {arms['A02']['macro_f1']:.6f} | "
            f"{arms['B01']['macro_f1']:.6f} | {arms['B01_bank_label_shuffle']['macro_f1']:.6f} | "
            f"{view['deltas']['B01']['macro_f1_vs_A02']:+.6f} | "
            f"{arms['B01']['macro_f1']-arms['B01_bank_label_shuffle']['macro_f1']:+.6f} | "
            f"{arms['B01']['predicted_positive_count']} |"
        )
    lines += [
        "",
        "## Seven diagnostic gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, passed in gates["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        "The results are adaptive Train-only diagnostics on repeatedly used development rows. H12-L overlaps primary. A `PASS` only means B01 may return for review; it does not qualify the recipe, authorize Test inference, alter the champion, or authorize Kaggle upload. Descriptive matched-positive-count tables use the frozen root utility and do not select K.",
        "",
        "Per-origin, subgroup, confusion, AUC/AP, Brier, both log-loss conventions, endpoint counts and descriptive count-matched results are in `RESULTS.json`. Hashes, zero-fit evidence and integrity checks are in `QA.json`; the recipe and inputs were frozen before score calculation in `FREEZE.json`.",
        "",
        f"Official Train SHA-256: `{result['provenance']['official_train_sha256']}`. Fit ledger SHA-256: `{result['provenance']['ledger_before']['sha256']}`; 66 starts / 64 saved / 2 historical failures before and after.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Freeze recipe, gates, code and inputs; no candidate scoring")
    mode.add_argument("--run", action="store_true", help="Verify pre-score freeze and execute the zero-fit diagnostic")
    args = parser.parse_args()
    if args.freeze:
        freeze = freeze_recipe()
        print(json.dumps({"status": freeze["status"], "source_commit": freeze["source_commit"],
                          "code_sha256": freeze["code_sha256"], "input_count": len(freeze["input_sha256"]),
                          "train_sha256": freeze["official_train_sha256"],
                          "fit_ledger": freeze["fit_ledger_before_scores"]}, indent=2))
    else:
        result = run_b01()
        print(json.dumps({"status": result["status"], "gate_evaluation": result["gate_evaluation"],
                          "fit_calls": result["provenance"]["fit_transform_predict_calls"],
                          "ledger_unchanged": result["provenance"]["ledger_before"] == result["provenance"]["ledger_after"]},
                         indent=2))


if __name__ == "__main__":
    main()
