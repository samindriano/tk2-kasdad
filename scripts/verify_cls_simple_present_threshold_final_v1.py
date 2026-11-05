"""Independent final threshold and E79 CSV replay; never fits a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT.parent / "tk2-simplification-audit-v1"
AUTH_PATH = AUDIT_ROOT / "reports/simplification_v1/CLS_SIMPLE_THRESHOLD_FINAL_AUTHORIZATION.md"
AUTH_SHA256 = "80359910f2befbf694de87fa5180b0c2c6e0c9d3ec40fc1a9c3589d569fc1375"
AUTH_COMMIT = "1facbf6b125b9aba23fc0423226f33687adaefa2"
OUT = ROOT / "reports/optimization_v3/cls_simple_present_threshold_final_v1"
FREEZE = OUT / "PRE_FIT_FREEZE.json"
RUN = ROOT / "experiments/runs/cls_simple_present_threshold_final_v1"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_simple_present_threshold_final_v1/classification_threshold_final.csv"
TRAIN = ROOT / "data/raw/classification/Train.csv"
OOF_CACHE = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
E79_CACHE = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
COMPARISONS = {
    "current_public_champion": ROOT / "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv",
    "prior_a02": ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv",
    "no_pos": ROOT / "submissions/optimization_v3/cls_simplification_benchmark_v1/classification_no_pos_e79.csv",
}
SOURCE_PATHS = (
    "scripts/cls_simple_present_threshold_final_v1.py",
    "scripts/verify_cls_simple_present_threshold_final_v1.py",
    "tests/test_cls_simple_present_threshold_final_v1.py",
    "reports/optimization_v3/cls_simple_present_threshold_final_v1/PROTOCOL.md",
)
LOW, HIGH, CENTRE = 0.45, 0.55, 0.50
MISSING_THRESHOLD = 0.53230976129477


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


def _label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return "../" + resolved.relative_to(ROOT.parent).as_posix()
        except ValueError:
            return str(resolved)


def _macro(tp: int, fp: int, tn: int, fn: int) -> Fraction:
    p = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0, 1)
    n = Fraction(2 * tn, 2 * tn + fp + fn) if (2 * tn + fp + fn) else Fraction(0, 1)
    return (p + n) / 2


def _counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.sum((y == 1) & (pred == 1))), int(np.sum((y == 0) & (pred == 1))),
        int(np.sum((y == 0) & (pred == 0))), int(np.sum((y == 1) & (pred == 0))),
    )


def independent_threshold_oracle(p_a: np.ndarray, missing: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    p_a = np.asarray(p_a, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    y = np.asarray(y, dtype=np.int8)
    if not (p_a.ndim == missing.ndim == y.ndim == 1 and len(p_a) == len(missing) == len(y) and len(y)):
        raise ValueError("Oracle received empty or misaligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
        raise ValueError("Oracle received invalid labels/probabilities")
    present = ~missing
    support = {str(c): int(np.sum(present & (y == c))) for c in (0, 1)}
    if min(support.values()) < 200:
        raise ValueError(f"Present class support below frozen minimum: {support}")
    initial = np.zeros(len(y), dtype=np.int8)
    initial[missing] = (p_a[missing] >= MISSING_THRESHOLD).astype(np.int8)
    initial[present] = (p_a[present] >= LOW).astype(np.int8)
    tp, fp, tn, fn = _counts(y, initial)
    unique = np.unique(p_a[present][(p_a[present] >= LOW) & (p_a[present] <= HIGH)])
    best: tuple[Fraction, float, tuple[int, int, int, int], tuple[float, float]] | None = None
    start = LOW
    for raw in unique:
        score = float(raw)
        if start <= score:
            threshold = min(max(CENTRE, start), score)
            objective = _macro(tp, fp, tn, fn)
            candidate = (objective, threshold, (tp, fp, tn, fn), (start, score))
            if best is None or objective > best[0] or (objective == best[0] and (abs(threshold - CENTRE), -threshold) < (abs(best[1] - CENTRE), -best[1])):
                best = candidate
        equal_score = present & (p_a == score)
        positives = int(np.sum(equal_score & (y == 1)))
        negatives = int(np.sum(equal_score & (y == 0)))
        tp -= positives
        fp -= negatives
        fn += positives
        tn += negatives
        start = float(np.nextafter(score, math.inf))
    if start <= HIGH:
        threshold = min(max(CENTRE, start), HIGH)
        objective = _macro(tp, fp, tn, fn)
        candidate = (objective, threshold, (tp, fp, tn, fn), (start, HIGH))
        if best is None or objective > best[0] or (objective == best[0] and (abs(threshold - CENTRE), -threshold) < (abs(best[1] - CENTRE), -best[1])):
            best = candidate
    if best is None:
        raise ValueError("Oracle found no representable decision state")
    objective, threshold, chosen, interval = best
    pred = np.zeros(len(y), dtype=np.int8)
    pred[missing] = (p_a[missing] >= MISSING_THRESHOLD).astype(np.int8)
    pred[present] = (p_a[present] >= threshold).astype(np.int8)
    actual = _counts(y, pred)
    if actual != chosen:
        raise AssertionError("Oracle threshold does not realize the selected state")
    return {
        "threshold": threshold, "threshold_hex": threshold.hex(), "interval": list(interval),
        "macro_f1": float(objective), "macro_f1_exact": f"{objective.numerator}/{objective.denominator}",
        "confusion": {"tp": chosen[0], "fp": chosen[1], "tn": chosen[2], "fn": chosen[3]},
        "support": support, "breakpoints": int(len(unique)), "predictions": pred,
    }


def _load_training_inputs(train_path: Path, cache_path: Path) -> dict[str, np.ndarray]:
    import pandas as pd
    meta = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"}, low_memory=False)
    if meta.isna().any().any():
        raise ValueError("Train metadata has missing IDs or cycles")
    ids_train = meta["record_id"].astype(str).to_numpy(dtype="U")
    cycles_train = pd.to_numeric(meta["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if len(ids_train) != 50_608 or len(ids_train) != len(set(ids_train.tolist())):
        raise ValueError("Train roster differs from its frozen unique 50,608 rows")
    cycle_by_id = dict(zip(ids_train.tolist(), cycles_train.tolist()))
    selected_ids, selected_cycles, selected_missing, selected_p_a = [], [], [], []
    with np.load(cache_path, allow_pickle=False) as z:
        for origin, lo, hi in ((67, 68, 75), (75, 76, 79)):
            ids = np.asarray(z[f"E{origin}_ids"], dtype="U")
            cycles = np.asarray(z[f"E{origin}_cycles"], dtype=np.int64)
            missing = np.asarray(z[f"E{origin}_missing"], dtype=bool)
            p_a = np.asarray(z[f"E{origin}_pA"], dtype=np.float64)
            if not (len(ids) == len(cycles) == len(missing) == len(p_a)) or len(ids) != len(set(ids.tolist())):
                raise ValueError(f"E{origin} OOF vectors are malformed")
            if not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
                raise ValueError(f"E{origin} OOF probabilities are invalid")
            all_allowed = np.arange(68, 80) if origin == 67 else np.arange(76, 80)
            if not np.isin(cycles, all_allowed).all():
                raise ValueError(f"E{origin} OOF cycle values are invalid")
            mask = (cycles >= lo) & (cycles <= hi)
            ids, cycles, missing, p_a = ids[mask], cycles[mask], missing[mask], p_a[mask]
            expected = 12_412 if origin == 67 else 8_491
            if len(ids) != expected:
                raise ValueError(f"E{origin} fit roster count differs from frozen plan")
            if any(item not in cycle_by_id or cycle_by_id[item] != int(c) for item, c in zip(ids.tolist(), cycles.tolist())):
                raise ValueError(f"E{origin} OOF ID/cycle mapping differs from Train")
            selected_ids.extend(ids.tolist())
            selected_cycles.extend(cycles.tolist())
            selected_missing.extend(missing.tolist())
            selected_p_a.extend(p_a.tolist())
    ids = np.asarray(selected_ids, dtype="U")
    cycles = np.asarray(selected_cycles, dtype=np.int64)
    missing = np.asarray(selected_missing, dtype=bool)
    p_a = np.asarray(selected_p_a, dtype=np.float64)
    if len(ids) != 20_903 or len(ids) != len(set(ids.tolist())):
        raise ValueError("Pooled OOF population is not the exact 20,903 unique rows")
    return {"train_ids": ids_train, "ids": ids, "cycles": cycles, "missing": missing, "pA": p_a}


def _selected_targets(train_path: Path, roster: np.ndarray, selected: np.ndarray) -> np.ndarray:
    import pandas as pd
    row_index = {value: i for i, value in enumerate(roster.tolist())}
    if len(row_index) != len(roster) or len(selected) != len(set(selected.tolist())) or not set(selected.tolist()).issubset(row_index):
        raise ValueError("Selected target IDs do not match the official Train roster")
    positions = {row_index[value] for value in selected.tolist()}
    raw = train_path.read_bytes()
    if raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1) != len(roster) + 1:
        raise ValueError("Physical Train lines do not map to its parsed roster")
    frame = pd.read_csv(train_path, usecols=["record_id", "artifact_signal"], dtype={"record_id": "string", "artifact_signal": "Int8"}, skiprows=lambda line: line > 0 and line - 1 not in positions, low_memory=False)
    ids = frame["record_id"].astype(str).tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(selected.tolist()):
        raise ValueError("Target-only selected-row read returned wrong IDs")
    mapping = dict(zip(ids, frame["artifact_signal"].astype(int).tolist()))
    y = np.asarray([mapping[key] for key in selected.tolist()], dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Selected targets are not binary")
    return y


def _read_test_inputs(test_path: Path, sample_path: Path, *, synthetic: bool = False) -> tuple[list[str], np.ndarray, list[str]]:
    import pandas as pd
    test = pd.read_csv(test_path, usecols=["record_id", "signal_stream"], dtype="string", keep_default_na=False, encoding="utf-8-sig")
    ids = test["record_id"].astype(str).tolist()
    missing = test["signal_stream"].str.strip().eq("").to_numpy(dtype=bool)
    if ((not synthetic and len(ids) != 29_421) or not ids or len(ids) != len(set(ids)) or any(not value for value in ids)):
        raise ValueError("Official Test IDs are missing, duplicated, or wrong-sized")
    header = pd.read_csv(sample_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    sample_ids = pd.read_csv(sample_path, usecols=["record_id"], dtype={"record_id": "string"}, keep_default_na=False, encoding="utf-8-sig")["record_id"].astype(str).tolist()
    if header != ["record_id", "artifact_signal"] or sample_ids != ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("SampleSubmission schema/order differs from official Test IDs")
    return ids, missing, header


def _csv_predictions(path: Path, expected_ids: list[str]) -> np.ndarray:
    ids, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["record_id", "artifact_signal"]:
            raise ValueError(f"Bad CSV header in {path}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2 or row[1] not in {"0", "1"}:
                raise ValueError(f"Bad binary CSV row in {path}:{line_no}")
            ids.append(row[0])
            labels.append(int(row[1]))
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError(f"CSV IDs/order do not match Test in {path}")
    return np.asarray(labels, dtype=np.int8)


def _mapped_inputs(freeze: dict[str, Any], train: Path, oof: Path, e79: Path, test: Path, sample: Path) -> dict[str, Path]:
    known = {
        "official_train_csv": train, "a02_oof_cache": oof, "canonical_e79_cache": e79,
        "official_test_csv": test, "sample_submission_csv": sample,
        "e79_application_freeze": ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_APPLICATION_FREEZE_V2.json",
        "e79_manifest": ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_MANIFEST.json",
        "e79_qa": ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_QA.json",
        "threshold_source_freeze": ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1/SOURCE_FREEZE.json",
        "threshold_source_ready": ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1/SOURCE_READY.json",
        "threshold_d1_d2_release": ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1/ROOT_RELEASE_D1_D2.json",
        "threshold_postfit_audit": ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1/POSTFIT_INDEPENDENT_AUDIT.json",
        "root_final_authorization": AUTH_PATH,
        "pilot_producer": ROOT / "scripts/cls_simple_present_threshold_v1.py",
        "pilot_independent_verifier": ROOT / "scripts/verify_cls_simple_present_threshold_v1.py",
        "pilot_postfit_auditor": ROOT / "scripts/audit_cls_simple_present_threshold_v1_postfit.py",
        "comparison_current_public_champion": COMPARISONS["current_public_champion"],
        "comparison_prior_a02": COMPARISONS["prior_a02"],
        "comparison_no_pos": COMPARISONS["no_pos"],
    }
    if set(known) != set(freeze.get("input_paths", {})):
        if freeze.get("synthetic_fixture_only") and set(freeze.get("input_paths", {})) == {"official_train_csv", "a02_oof_cache", "canonical_e79_cache", "official_test_csv", "sample_submission_csv"}:
            return {"official_train_csv": train, "a02_oof_cache": oof, "canonical_e79_cache": e79, "official_test_csv": test, "sample_submission_csv": sample}
        raise ValueError("Verifier input set differs from committed freeze")
    return known


def verify_final_candidate(*, freeze_path: Path = FREEZE, train_path: Path = TRAIN, oof_path: Path = OOF_CACHE,
                           e79_path: Path = E79_CACHE, test_path: Path = TEST, sample_path: Path = SAMPLE,
                           run_dir: Path = RUN, output_path: Path = OUTPUT_CSV, synthetic: bool = False) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("campaign") != "CLS_SIMPLE_PRESENT_THRESHOLD_FINAL_V1" or freeze.get("synthetic_fixture_only") != synthetic:
        raise ValueError("Freeze scope differs from verifier invocation")
    source_hashes = freeze["source_hashes"]
    if set(source_hashes) != set(SOURCE_PATHS):
        raise ValueError("Source path set differs from committed source freeze")
    for rel, expected in source_hashes.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected or _git("status", "--porcelain", "--", rel):
            raise ValueError(f"Frozen source is missing/modified/uncommitted: {rel}")
    if not synthetic:
        if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", freeze["source_commit"], "HEAD"], check=False).returncode != 0:
            raise ValueError("Frozen source commit is not an ancestor of the current branch")
        freeze_rel = freeze_path.resolve().relative_to(ROOT.resolve()).as_posix()
        if _git("status", "--porcelain", "--", freeze_rel):
            raise ValueError("Pre-fit freeze is not committed cleanly")
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{freeze_rel}"])
        if hashlib.sha256(committed).hexdigest() != sha(freeze_path):
            raise ValueError("Committed pre-fit freeze differs from the file being verified")
    if not synthetic:
        if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", freeze["source_commit"], "HEAD"], check=False).returncode != 0:
            raise ValueError("Frozen source commit is not an ancestor of the current branch")
        freeze_rel = freeze_path.resolve().relative_to(ROOT.resolve()).as_posix()
        if _git("status", "--porcelain", "--", freeze_rel):
            raise ValueError("Pre-fit freeze is not committed cleanly")
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{freeze_rel}"])
        if hashlib.sha256(committed).hexdigest() != sha(freeze_path):
            raise ValueError("Committed pre-fit freeze differs from the file being verified")
    input_paths = _mapped_inputs(freeze, train_path, oof_path, e79_path, test_path, sample_path)
    checks = ["committed_source_snapshot"]
    if set(input_paths) != set(freeze["input_paths"]):
        raise ValueError("Verifier runtime input set differs from pre-fit freeze")
    for key, path in input_paths.items():
        if _label(path) != freeze["input_paths"][key] or not path.is_file() or sha(path) != freeze["input_hashes"][key]:
            raise ValueError(f"Frozen input provenance mismatch: {key}")
    checks.append("all_source_and_input_hashes")
    if not synthetic:
        if sha(AUTH_PATH) != AUTH_SHA256 or subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "rev-parse", "HEAD"], text=True).strip() != AUTH_COMMIT:
            raise ValueError("Root authorization changed")
        checks.append("root_authorization_commit_and_hash")

    pool = _load_training_inputs(train_path, oof_path) if not synthetic else _load_synthetic_pool(train_path, oof_path)
    labels = _selected_targets(train_path, pool["train_ids"], pool["ids"])
    optimum = independent_threshold_oracle(pool["pA"], pool["missing"], labels)
    fit_path = run_dir / "fits/FINAL.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("status") != "FROZEN_FINAL_THRESHOLD" or fit.get("threshold_hex") != optimum["threshold_hex"]:
        raise ValueError("Independent final-fit threshold oracle disagrees with receipt")
    if fit.get("pre_fit_freeze_sha256") != sha(freeze_path) or fit.get("input_hashes") != freeze.get("input_hashes"):
        raise ValueError("Fit receipt is not bound to the committed pre-fit inputs")
    if fit.get("pre_fit_freeze_sha256") != sha(freeze_path) or fit.get("input_hashes") != freeze.get("input_hashes"):
        raise ValueError("Fit receipt is not bound to the committed pre-fit inputs")
    if fit.get("fit_ids_sha256") != hashlib.sha256("\n".join(pool["ids"].tolist()).encode("utf-8")).hexdigest():
        raise ValueError("Fit receipt ID identity differs from exact pooled OOF rows")
    if fit.get("fit_confusion") != optimum["confusion"] or fit.get("fit_macro_f1_exact") != optimum["macro_f1_exact"]:
        raise ValueError("Fit receipt objective or confusion differs from independent oracle")
    if fit.get("decision_state_interval") != optimum["interval"] or fit.get("present_class_support") != optimum["support"]:
        raise ValueError("Fit receipt interval/support differs from independent oracle")
    fit_ledger = [json.loads(x) for x in (run_dir / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    fit_started = [x for x in fit_ledger if x.get("event") == "FIT_STARTED" and x.get("slot") == "FINAL_A02_PAST_OOF_THRESHOLD"]
    fit_done = [x for x in fit_ledger if x.get("event") in {"FIT_FROZEN", "FIT_RECOVERED"} and x.get("slot") == "FINAL_A02_PAST_OOF_THRESHOLD"]
    if len(fit_started) != 1 or len(fit_done) != 1 or fit_done[0].get("receipt_sha256") != sha(fit_path):
        raise ValueError("Final fit ledger is not exactly one durable completed slot")
    checks.extend(["exact_unique_20903_oof_pool", "selected_train_labels_only", "exact_macro_f1_optimum", "durable_one_fit_slot"])

    e79_ids, e79_missing, p_a = _load_e79(e79_path, synthetic=synthetic)
    test_ids, test_missing, sample_header = _read_test_inputs(test_path, sample_path, synthetic=synthetic)
    if e79_ids != test_ids or not np.array_equal(e79_missing, test_missing):
        raise ValueError("Cached E79 IDs or missing mask differs from official Test")
    expected = np.where(e79_missing, p_a >= MISSING_THRESHOLD, p_a >= optimum["threshold"]).astype(np.int8)
    csv_labels = _csv_predictions(output_path, test_ids)
    if len(test_ids) != (len(e79_ids) if synthetic else 29_421) or not np.array_equal(csv_labels, expected):
        raise ValueError("CSV predictions do not independently replay from the frozen threshold/E79 inputs")
    if not np.array_equal(csv_labels[e79_missing], (p_a[e79_missing] >= MISSING_THRESHOLD).astype(np.int8)):
        raise ValueError("Missing-stream decisions differ from the fixed fallback")
    checks.extend(["E79_cache_to_Test_ID_order", "Test_missing_mask_match", "CSV_exact_prediction_replay", "missing_decision_invariance"])

    comparisons: dict[str, Any] = {}
    duplicates: list[str] = []
    if not synthetic:
        for name, path in COMPARISONS.items():
            other = _csv_predictions(path, test_ids)
            comparisons[name] = {
                "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                "changed_decisions": int(np.sum(other != csv_labels)),
                "positive_delta": int(np.sum(csv_labels) - np.sum(other)),
                "identical": bool(np.array_equal(other, csv_labels)),
            }
        submissions = ROOT / "submissions"
        candidate_hash = hashlib.sha256(np.ascontiguousarray(csv_labels, dtype=np.int8).tobytes()).hexdigest()
        for path in sorted(submissions.rglob("*.csv")):
            if path.resolve() == output_path.resolve() or path in COMPARISONS.values():
                continue
            try:
                other = _csv_predictions(path, test_ids)
            except (OSError, ValueError, UnicodeError):
                continue
            if hashlib.sha256(np.ascontiguousarray(other, dtype=np.int8).tobytes()).hexdigest() == candidate_hash:
                duplicates.append(path.relative_to(ROOT).as_posix())
        checks.extend(["champion_prior_A02_and_NO_POS_vector_comparison", "all_local_submission_duplicate_vectors_scanned"])

    apply_path = run_dir / "apply/E79_APPLIED.json"
    apply = json.loads(apply_path.read_text(encoding="utf-8"))
    if apply.get("csv_sha256") != sha(output_path) or apply.get("fit_receipt_sha256") != sha(fit_path):
        raise ValueError("E79 application receipt differs from CSV/fit receipt")
    if apply.get("rows") != len(test_ids) or apply.get("positive_count") != int(np.sum(expected)):
        raise ValueError("E79 application counts differ from replay")
    if apply.get("comparisons") != comparisons or apply.get("local_duplicate_vectors") != duplicates:
        raise ValueError("Application comparisons/duplicate scan differ from independent replay")
    apply_started = [x for x in fit_ledger if x.get("event") == "E79_APPLY_STARTED" and x.get("slot") == "FINAL_A02_PAST_OOF_THRESHOLD"]
    apply_done = [x for x in fit_ledger if x.get("event") == "E79_APPLIED" and x.get("slot") == "FINAL_A02_PAST_OOF_THRESHOLD"]
    if len(apply_started) != 1 or len(apply_done) != 1 or apply_done[0].get("receipt_sha256") != sha(apply_path):
        raise ValueError("E79 application ledger is not exactly one durable completed apply")
    if apply.get("comparisons") != comparisons or apply.get("local_duplicate_vectors") != duplicates:
        raise ValueError("Application comparisons/duplicate scan differ from independent replay")
    apply_started = [x for x in fit_ledger if x.get("event") == "E79_APPLY_STARTED" and x.get("slot") == "FINAL_A02_PAST_OOF_THRESHOLD"]
    apply_done = [x for x in fit_ledger if x.get("event") == "E79_APPLIED" and x.get("slot") == "FINAL_A02_PAST_OOF_THRESHOLD"]
    if len(apply_started) != 1 or len(apply_done) != 1 or apply_done[0].get("receipt_sha256") != sha(apply_path):
        raise ValueError("E79 application ledger is not exactly one durable completed apply")
    if not synthetic and (len(test_ids) != 29_421 or sample_header != ["record_id", "artifact_signal"]):
        raise ValueError("Official CSV dimensions/schema are not compliant")
    checks.append("application_receipt_and_sample_schema")

    return {
        "schema_version": 1, "status": "PASS", "campaign": "CLS_SIMPLE_PRESENT_THRESHOLD_FINAL_V1",
        "source_commit": freeze["source_commit"], "pre_fit_freeze_sha256": sha(freeze_path),
        "fit_receipt_sha256": sha(fit_path), "apply_receipt_sha256": sha(apply_path),
        "csv_path": _label(output_path), "csv_sha256": sha(output_path),
        "prediction_vector_sha256": hashlib.sha256(np.ascontiguousarray(csv_labels, dtype=np.int8).tobytes()).hexdigest(),
        "threshold": optimum["threshold"], "threshold_hex": optimum["threshold_hex"],
        "calibration_rows": len(pool["ids"]), "fit_macro_f1": optimum["macro_f1"],
        "fit_macro_f1_exact": optimum["macro_f1_exact"], "fit_confusion": optimum["confusion"],
        "present_class_support": optimum["support"], "decision_state_interval": optimum["interval"],
        "test_rows": len(test_ids), "positive_count": int(np.sum(csv_labels)),
        "missing_rows": int(np.sum(e79_missing)), "test_columns_read": ["record_id", "signal_stream"],
        "sample_target_values_read": False, "test_target_read": False,
        "comparisons": comparisons, "local_duplicate_vectors": duplicates,
        "upload_must_be_skipped_for_duplicate": bool(duplicates or any(row["identical"] for row in comparisons.values())),
        "checks": checks, "check_count": len(checks),
        "interpretation": "Calibration objective is a fit statistic on the selected past OOF rows, not an untouched validation estimate.",
    }


def _load_e79(path: Path, *, synthetic: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["record_ids"], dtype="U").tolist()
        missing = np.asarray(archive["missing_mask"], dtype=bool)
        p_a = np.asarray(archive["a02_probability"], dtype=np.float64)
    if not ids or len(ids) != len(set(ids)) or not (len(ids) == len(missing) == len(p_a)):
        raise ValueError("E79 vectors are malformed")
    if not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
        raise ValueError("E79 probability vector is invalid")
    if not synthetic and len(ids) != 29_421:
        raise ValueError("Canonical E79 bundle does not contain exactly 29,421 IDs")
    return ids, missing, p_a


def _load_synthetic_pool(train_path: Path, cache_path: Path) -> dict[str, np.ndarray]:
    import pandas as pd
    meta = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    roster = meta["record_id"].astype(str).to_numpy(dtype="U")
    cycles_by_id = dict(zip(roster.tolist(), pd.to_numeric(meta["cycle_code"], errors="raise").astype(int).tolist()))
    ids, cycles, missing, p_a = [], [], [], []
    with np.load(cache_path, allow_pickle=False) as z:
        for origin, lo, hi in ((67, 68, 75), (75, 76, 79)):
            rows = (np.asarray(z[f"E{origin}_cycles"], dtype=np.int64) >= lo) & (np.asarray(z[f"E{origin}_cycles"], dtype=np.int64) <= hi)
            part_ids = np.asarray(z[f"E{origin}_ids"], dtype="U")[rows]
            part_cycles = np.asarray(z[f"E{origin}_cycles"], dtype=np.int64)[rows]
            part_missing = np.asarray(z[f"E{origin}_missing"], dtype=bool)[rows]
            part_pa = np.asarray(z[f"E{origin}_pA"], dtype=np.float64)[rows]
            if any(cycles_by_id.get(record_id) != int(cycle) for record_id, cycle in zip(part_ids.tolist(), part_cycles.tolist())):
                raise ValueError("Synthetic OOF ID/cycle mapping mismatch")
            ids.extend(part_ids.tolist()); cycles.extend(part_cycles.tolist()); missing.extend(part_missing.tolist()); p_a.extend(part_pa.tolist())
    if len(ids) != len(set(ids)):
        raise ValueError("Synthetic OOF pool has duplicate IDs")
    return {"train_ids": roster, "ids": np.asarray(ids, dtype="U"), "cycles": np.asarray(cycles, dtype=np.int64),
            "missing": np.asarray(missing, dtype=bool), "pA": np.asarray(p_a, dtype=np.float64)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--train-csv", type=Path, default=TRAIN)
    parser.add_argument("--oof-cache", type=Path, default=OOF_CACHE)
    parser.add_argument("--e79-cache", type=Path, default=E79_CACHE)
    parser.add_argument("--test-csv", type=Path, default=TEST)
    parser.add_argument("--sample-csv", type=Path, default=SAMPLE)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--synthetic-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    report = verify_final_candidate(freeze_path=args.freeze, train_path=args.train_csv, oof_path=args.oof_cache,
                                    e79_path=args.e79_cache, test_path=args.test_csv, sample_path=args.sample_csv,
                                    run_dir=args.run_dir, output_path=args.output_csv, synthetic=args.synthetic_fixture)
    if not args.synthetic_fixture:
        report_path = OUT / "FINAL_INDEPENDENT_QA.json"
        data = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, report_path)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
