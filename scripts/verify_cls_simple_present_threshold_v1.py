"""Independent verifier for CLS_SIMPLE_PRESENT_THRESHOLD_V1.

This file intentionally does not import the producer. It reads only non-target
Train columns and the explicitly allowlisted prediction/group NPZ members.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / "tk2-simplification-audit-v1"
OUT = ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1"
FREEZE = OUT / "SOURCE_FREEZE.json"
PREFLIGHT = OUT / "SOURCE_PREFLIGHT.json"
DECISION = AUDIT / "reports/simplification_v1/CLS_SIMPLE_THRESHOLD_DECISION.md"
CACHE = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
DECISION_SHA256 = "e0e0ac69ce7ce0e148e8edfe2345e21e37addf55807cdcff6b2f2995a08cd0c5"
CACHE_SHA256 = "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"
ROOT_SOURCE_COMMIT = "a39721172b6d293cb68ee0d120060082d222699e"
LOW, HIGH, CENTRE = 0.45, 0.55, 0.50
MISSING_THRESHOLD = 0.53230976129477
EXPECTED_ROWS = {67: 20_903, 75: 8_491}
EXPECTED_SCORE_ROWS = 15_953
OFFICIAL_TRAIN_ROWS = 50_608
STAGE_PLAN = {
    "D1": (67, 68, 71, 67, 72, 75),
    "D2": (67, 68, 75, 75, 76, 79),
}
EXPECTED_STAGE_ROWS = {"D1": (4_950, 7_462), "D2": (12_412, 8_491)}
NPZ_MEMBERS = {
    origin: [f"E{origin}_{key}" for key in (
        "ids", "cycles", "missing", "pA", "pP", "group_availability",
        "group_entry_friction", "group_text_load",
    )]
    for origin in (67, 75)
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def id_hash(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(AUDIT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _verify_committed_source_snapshot(commit: str, hashes: dict[str, str]) -> None:
    for relative in _source_paths():
        payload = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(payload).hexdigest() != hashes.get(relative):
            raise ValueError(f"Committed source snapshot mismatch: {relative}")


def _source_paths() -> tuple[str, ...]:
    return (
        "scripts/cls_simple_present_threshold_v1.py",
        "scripts/verify_cls_simple_present_threshold_v1.py",
        "tests/test_cls_simple_present_threshold_v1.py",
        "reports/optimization_v3/cls_simple_present_threshold_v1/PROTOCOL.md",
    )


def _text_groups(train, origin: int, rows: np.ndarray) -> np.ndarray:
    values = train["text_load"].to_numpy(dtype=float)
    prefix = values[train["cycle_code"].to_numpy(dtype=int) <= origin]
    edges = np.quantile(prefix, [0.25, 0.50, 0.75], method="linear")
    bins = np.searchsorted(edges, values[rows], side="right")
    return np.asarray(["Q" + str(int(value) + 1) for value in bins])


def verify_source_preflight() -> dict[str, Any]:
    import pandas as pd

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if freeze.get("status") != "SOURCE_FROZEN_BEFORE_TARGET_FREE_PREFLIGHT":
        raise ValueError("Source freeze status is not recognized")
    if report.get("status") != "PASS_TARGET_FREE_PREFLIGHT":
        raise ValueError("Preflight is not a target-free PASS receipt")
    source_files = {
        relative: sha256_file(ROOT / relative)
        for relative in _source_paths()
    }
    if source_files != freeze.get("source_hashes"):
        raise ValueError("Independent source hash inventory mismatch")
    _verify_committed_source_snapshot(freeze["source_commit"], source_files)
    train_path = Path(freeze["inputs"]["official_train_csv"]["path_at_freeze"])
    if sha256_file(train_path) != freeze["inputs"]["official_train_csv"]["sha256"]:
        raise ValueError("Official Train hash changed after source freeze")
    if sha256_file(CACHE) != CACHE_SHA256 or sha256_file(CACHE) != freeze["inputs"]["a02_oof_cache"]["sha256"]:
        raise ValueError("A02 cache hash mismatch")
    if sha256_file(DECISION) != DECISION_SHA256:
        raise ValueError("Root decision hash mismatch")
    root_head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain"):
        raise ValueError("Root decision checkout is dirty")
    ancestor = subprocess.run(
        ["git", "-C", str(AUDIT), "merge-base", "--is-ancestor", ROOT_SOURCE_COMMIT, "HEAD"],
        capture_output=True,
    )
    if ancestor.returncode:
        raise ValueError("Required root source ancestor is missing")
    if freeze.get("root_lineage", {}).get("required_ancestor") != ROOT_SOURCE_COMMIT:
        raise ValueError("Freeze records the wrong required root source ancestor")
    if report.get("source_commit") != freeze.get("source_commit"):
        raise ValueError("Preflight and freeze producer commits differ")
    if report.get("source_freeze_sha256") != sha256_file(FREEZE):
        raise ValueError("Preflight points to a different source-freeze receipt")
    if report.get("official_train_sha256") != sha256_file(train_path):
        raise ValueError("Preflight Train hash mismatch")

    columns = pd.read_csv(train_path, nrows=0).columns
    wanted = ["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load"]
    if not set(wanted + ["artifact_signal"]).issubset(columns):
        raise ValueError("Official Train columns missing")
    train = pd.read_csv(
        train_path, usecols=wanted,
        dtype={"record_id": "string", "signal_stream": "string"},
        low_memory=False,
    )
    if len(train) != OFFICIAL_TRAIN_ROWS or train["record_id"].isna().any() or train["record_id"].duplicated().any():
        raise ValueError("Independent official Train metadata shape/IDs mismatch")
    train["record_id"] = train["record_id"].astype(str)
    train["cycle_code"] = pd.to_numeric(train["cycle_code"], errors="raise").astype(int)
    train["entry_friction"] = pd.to_numeric(train["entry_friction"], errors="raise").astype(int)
    train["text_load"] = pd.to_numeric(train["text_load"], errors="raise").astype(float)
    raw = train_path.read_bytes()
    physical_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    if physical_lines != len(train) + 1:
        raise ValueError("Official Train cannot be mapped safely by physical row for selected-label reading")

    by_id = {rid: i for i, rid in enumerate(train["record_id"].tolist())}
    cache_rows = {}
    with np.load(CACHE, allow_pickle=False) as z:
        for origin, member_names in NPZ_MEMBERS.items():
            values = {name.removeprefix(f"E{origin}_"): np.asarray(z[name]) for name in member_names}
            ids = values["ids"].astype(str)
            cycles = values["cycles"].astype(int)
            missing = values["missing"].astype(bool)
            n = len(ids)
            if n != EXPECTED_ROWS[origin] or len(set(ids.tolist())) != n:
                raise ValueError(f"E{origin} independent row count or ID uniqueness failure")
            if any(len(value) != n for value in values.values()):
                raise ValueError(f"E{origin} independent cache vector lengths differ")
            if not set(ids.tolist()).issubset(by_id):
                raise ValueError(f"E{origin} has IDs absent from Train")
            positions = np.asarray([by_id[x] for x in ids], dtype=np.int64)
            if not np.array_equal(train["cycle_code"].to_numpy()[positions], cycles):
                raise ValueError(f"E{origin} ID/cycle crosswalk mismatch")
            stream = train["signal_stream"]
            official_missing = stream.isna().to_numpy()[positions] | (stream.fillna("").astype(str).to_numpy()[positions] == "")
            if not np.array_equal(official_missing, missing):
                raise ValueError(f"E{origin} availability crosswalk mismatch")
            if not np.array_equal(train["entry_friction"].astype(int).astype(str).to_numpy()[positions], values["group_entry_friction"].astype(str)):
                raise ValueError(f"E{origin} entry_friction group mismatch")
            if not np.array_equal(_text_groups(train, origin, positions), values["group_text_load"].astype(str)):
                raise ValueError(f"E{origin} prefix text_load groups mismatch")
            if not np.array_equal(np.where(missing, "missing", "present"), values["group_availability"].astype(str)):
                raise ValueError(f"E{origin} availability group mismatch")
            p_a = values["pA"].astype(float)
            p_p = values["pP"].astype(float)
            if not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
                raise ValueError(f"E{origin} invalid pA probabilities")
            if not np.isfinite(p_p[~missing]).all() or ((p_p[~missing] < 0) | (p_p[~missing] > 1)).any():
                raise ValueError(f"E{origin} invalid present pP probabilities")
            cache_rows[origin] = {"ids": ids, "cycles": cycles}
            actual = report["origins"][str(origin)]
            if actual.get("rows") != n or actual.get("ordered_id_sha256") != id_hash(ids):
                raise ValueError(f"E{origin} preflight receipt ID/row summary differs")

    score_sets = []
    for stage, plan in STAGE_PLAN.items():
        fo, fl, fh, so, sl, sh = plan
        f = cache_rows[fo]
        s = cache_rows[so]
        fit_mask = (f["cycles"] >= fl) & (f["cycles"] <= fh)
        score_mask = (s["cycles"] >= sl) & (s["cycles"] <= sh)
        fit_ids, score_ids = set(f["ids"][fit_mask].tolist()), set(s["ids"][score_mask].tolist())
        if fit_ids & score_ids:
            raise ValueError(f"{stage} fit/score windows overlap")
        item = report["origins"][stage]
        if (item.get("fit_rows"), item.get("score_rows")) != EXPECTED_STAGE_ROWS[stage]:
            raise ValueError(f"{stage} denominator differs from frozen support")
        if item.get("fit_rows") != len(fit_ids) or item.get("score_rows") != len(score_ids):
            raise ValueError(f"{stage} fit/score denominator mismatch")
        if item.get("fit_ids_sha256") != id_hash(cache_rows[fo]["ids"][fit_mask]):
            raise ValueError(f"{stage} fit ID hash mismatch")
        if item.get("score_ids_sha256") != id_hash(cache_rows[so]["ids"][score_mask]):
            raise ValueError(f"{stage} score ID hash mismatch")
        score_sets.append(score_ids)
    unique_score = score_sets[0] | score_sets[1]
    if score_sets[0] & score_sets[1] or len(unique_score) != EXPECTED_SCORE_ROWS:
        raise ValueError("D1/D2 score rows are not disjoint with the frozen denominator")
    if report.get("combined_score_rows") != EXPECTED_SCORE_ROWS:
        raise ValueError("Combined score denominator differs")
    if report.get("official_target_column_values_read") is not False or report.get("oof_target_members_opened") is not False:
        raise ValueError("Target boundary is not recorded as closed")
    if report.get("test_or_sample_submission_accessed") is not False:
        raise ValueError("Test boundary is not recorded as closed")
    return {
        "status": "PASS",
        "checks": 34,
        "train_rows": len(train),
        "oof_rows_E67": len(cache_rows[67]["ids"]),
        "oof_rows_E75": len(cache_rows[75]["ids"]),
        "score_rows": len(unique_score),
        "target_values_read": False,
        "npz_target_members_opened": False,
        "root_artifact_head": root_head,
    }


def _macro(y: np.ndarray, pred: np.ndarray) -> Fraction:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    f1p = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
    f1n = Fraction(2 * tn, 2 * tn + fp + fn) if 2 * tn + fp + fn else Fraction(0)
    return (f1p + f1n) / 2


def independent_threshold_oracle(p_a: np.ndarray, missing: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Brute force every representable state boundary for small synthetic fixtures."""
    p_a, missing, y = np.asarray(p_a, float), np.asarray(missing, bool), np.asarray(y, np.int8)
    points = {LOW, CENTRE, HIGH}
    for score in np.unique(p_a[~missing]):
        if LOW <= score <= HIGH:
            points.add(float(score))
            above = float(np.nextafter(score, math.inf))
            if above <= HIGH:
                points.add(above)
    best = None
    for threshold in sorted(points):
        pred = np.where(missing, p_a >= MISSING_THRESHOLD, p_a >= threshold).astype(np.int8)
        objective = _macro(y, pred)
        candidate = (objective, threshold)
        if best is None or objective > best[0] or (
            objective == best[0] and (abs(threshold - CENTRE), -threshold) < (abs(best[1] - CENTRE), -best[1])
        ):
            best = candidate
    return {"threshold": best[1], "macro_f1_exact": f"{best[0].numerator}/{best[0].denominator}"}


def run_synthetic_self_check() -> dict[str, Any]:
    # Standalone oracle invariants: tied scores, bounds, inclusive >=, fixed missing.
    p_a = np.asarray([0.45, 0.45, 0.49, 0.50, 0.50, 0.55, 0.60, 0.20] * 100, dtype=float)
    missing = np.asarray([False, False, False, False, False, False, False, True] * 100)
    y = np.asarray(([0, 1, 0, 1, 0, 1, 0, 1] * 100), dtype=np.int8)
    result = independent_threshold_oracle(p_a, missing, y)
    if not LOW <= result["threshold"] <= HIGH:
        raise AssertionError("Oracle selected threshold outside inclusive bounds")
    pred1 = np.where(missing, p_a >= MISSING_THRESHOLD, p_a >= result["threshold"])
    pred2 = pred1.copy()
    pred2[missing] = p_a[missing] >= MISSING_THRESHOLD
    if not np.array_equal(pred1[missing], pred2[missing]):
        raise AssertionError("Fixed missing decisions changed")
    return {"status": "PASS", "checks": 3, "oracle_threshold": result["threshold"],
            "oracle_macro_f1_exact": result["macro_f1_exact"], "target_accessed": False}


def main() -> int:
    print(json.dumps(verify_source_preflight(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
