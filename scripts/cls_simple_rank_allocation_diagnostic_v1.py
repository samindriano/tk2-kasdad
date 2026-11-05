"""Zero-fit matched-positive-count diagnostic for POS/A02 vs A02.

--freeze hashes sources and cached inputs only; it never opens NPZ arrays.
--calculate opens only the allowlisted E67/E75 prediction members after the
pre-access freeze commit and shared/per-lane RAM/lock gates. No model is fit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / "tk2-simplification-audit-v1"
OUT = ROOT / "reports/optimization_v3/cls_simple_rank_allocation_diagnostic_v1"
FREEZE = OUT / "PRE_ACCESS_FREEZE.json"
RESULTS = OUT / "RESULTS.json"
QA = OUT / "INDEPENDENT_QA.json"
SOURCE_SELF = "scripts/cls_simple_rank_allocation_diagnostic_v1.py"
SOURCE_VERIFIER = "scripts/verify_cls_simple_rank_allocation_diagnostic_v1.py"
SOURCE_PROTOCOL = "reports/optimization_v3/cls_simple_rank_allocation_diagnostic_v1/PROTOCOL.md"

RELATIVE_FILES = {
    "audit_input_freeze": "../tk2-simplification-audit-v1/reports/simplification_v1/INPUT_FREEZE.json",
    "audit_protocol": "../tk2-simplification-audit-v1/reports/simplification_v1/PROTOCOL.md",
    "audit_script": "../tk2-simplification-audit-v1/scripts/simplification_audit_v1.py",
    "audit_decisions": "../tk2-simplification-audit-v1/reports/simplification_v1/DECISIONS.json",
    "audit_results": "../tk2-simplification-audit-v1/reports/simplification_v1/RESULTS.json",
    "audit_verification": "../tk2-simplification-audit-v1/reports/simplification_v1/VERIFICATION.json",
    "audit_findings": "../tk2-simplification-audit-v1/reports/simplification_v1/FINDINGS.md",
    "light_parallel_policy": "../tk2-simplification-audit-v1/reports/simplification_v1/LIGHT_PARALLEL_EXECUTION_POLICY.json",
    "audit_next_simple_research": "../tk2-simplification-audit-v1/reports/simplification_v1/NEXT_SIMPLE_MODEL_RESEARCH.md",
    "a02_policy_predictions": "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz",
    "pos_predictions": "experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/predictions.npz",
    "novelty_b01_summary": "reports/optimization_v3/a02_b01_centering_v1/B01_OUTPUTS/SUMMARY.md",
    "novelty_a02_neighbors": "reports/optimization_v3/a02_neighbors_v1/FINAL_SUMMARY.md",
    "novelty_a02_cgeom": "reports/optimization_v3/a02_coefficient_geometry_v1/RESULTS.md",
    "novelty_a02_continuation": "reports/optimization_v3/a02_continuation_v1/FINAL_HANDOFF.md",
}
EXPECTED_HASHES = {
    "audit_input_freeze": "d1f8f77864c1b2c1df8dc11732ffb61b68ac08f5e0b37ded48abf33f7506ac56",
    "audit_protocol": "907c2b1323ea759a057527aabb043ae2179039bc67596dcd03a385187da62b55",
    "audit_script": "b943e16f459d78fec736a801ac917f925737a266d13d3c160f6dae4b06ef3edc",
    "audit_decisions": "e9ef07ff4a364951ca13598f4601aff4d4b005fa5bc3855382e6b028f279157d",
    "audit_results": "6db199169bce9d131d530c77f63d7720cb37e9f98a5f1080a52afdbaa45882af",
    "audit_verification": "3c342d1e30a41fa882a4b6fed241b9212c533a3138980124d91a3ae6fbac35ac",
    "audit_findings": "5073d955f5d2b932bef7e18df1b8498adec3885873867d64da3f5b272e4cc000",
    "light_parallel_policy": "235de024810c602635a518851438ffd9e6e45f55e7451b3adffd749cb89f955e",
    "audit_next_simple_research": "702ea7ca9da8fa647e60d3d4b4c89a33f3b24b05bc03f43c80282a3bc2c04a07",
    "a02_policy_predictions": "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351",
    "pos_predictions": "a916a65fe6974501b3a2777a8a9bfae08049a917e5d1a0d9d3b7eb8fd4b479c1",
    "novelty_b01_summary": "3400fe21529abe96abd5a02a6191851edc366e3d5af5ace6ae29f4eadd1b516b",
    "novelty_a02_neighbors": "3d2da030b2148ebbeff2466ab61e3c65ca348ecd052e8d66181503b9a217b311",
    "novelty_a02_cgeom": "7cc8bb1049f08d515c42d1eee9bfc80e101cadb71a0783843910fd08c8f85b30",
    "novelty_a02_continuation": "39b4b0b21a48d9e269f455912d8ed34bf5e344fffaa1ae85cd085e056f853f3d",
}
FROZEN_AUDIT_SOURCE_HEAD = "a39721172b6d293cb68ee0d120060082d222699e"
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
RESOURCE_ESTIMATE_BYTES = 512 * 1024**2
RESOURCE_START_BYTES = int(1.5 * 1024**3)
RESOURCE_RESERVE_BYTES = int(0.75 * 1024**3)
LOCK_PATH = Path(tempfile.gettempdir()) / "kasdad_cls_simple_rank_allocation_diagnostic_v1.lock"
SHARED_LOCK_PATH = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
MEMBERS = {
    origin: [f"E{origin}_{suffix}" for suffix in ("ids", "cycles", "missing", "y", "pA", "pP")]
    for origin in (67, 75)
}
VIEWS = {
    "E67_W4": [(67, 68, 71)],
    "E67_W5": [(67, 72, 75)],
    "E67_W6": [(67, 76, 79)],
    "E67_H12": [(67, 68, 79)],
    "E75_W6": [(75, 76, 79)],
    "UNIQUE": [(67, 68, 75), (75, 76, 79)],
}
POLICY_NAMES = (
    "complex_threshold",
    "simple_threshold",
    "simple_topk_cycle_at_complex_count",
    "complex_topk_cycle_at_simple_count",
    "simple_topk_origin_at_complex_count",
    "complex_topk_origin_at_simple_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_path(label: str) -> Path:
    value = Path(RELATIVE_FILES[label])
    return (ROOT / value).resolve()


def git_value(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def resource_available_bytes() -> int:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for the declared RAM gate") from exc
    return int(psutil.virtual_memory().available)


@contextmanager
def lane_lock_and_ram_gate() -> Any:
    streams = []
    locked_streams = []

    def acquire(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+b")
        streams.append(stream)
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Required lock is busy: {path}") from exc
        locked_streams.append(stream)

    try:
        acquire(SHARED_LOCK_PATH)
        acquire(LOCK_PATH)
        start = resource_available_bytes()
        projected = start - RESOURCE_ESTIMATE_BYTES
        if start < RESOURCE_START_BYTES or projected < RESOURCE_RESERVE_BYTES:
            raise RuntimeError(
                f"RAM gate failed: available={start}, projected={projected}, "
                f"start_floor={RESOURCE_START_BYTES}, reserve_floor={RESOURCE_RESERVE_BYTES}"
            )
        gate_receipt = {
            "shared_lock_path": str(SHARED_LOCK_PATH), "lane_lock_path": str(LOCK_PATH),
            "available_start_bytes": start, "estimate_bytes": RESOURCE_ESTIMATE_BYTES,
            "projected_bytes": projected, "start_floor_bytes": RESOURCE_START_BYTES,
            "reserve_floor_bytes": RESOURCE_RESERVE_BYTES,
        }
        yield gate_receipt
        end = resource_available_bytes()
        if end < RESOURCE_RESERVE_BYTES:
            raise RuntimeError(f"RAM reserve failed at lane boundary: available={end}")
        gate_receipt.update({"available_end_bytes": end, "reserve_pass": True})
    finally:
        for stream in reversed(locked_streams):
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        for stream in streams:
            try:
                stream.close()
            except OSError:
                pass


def _root_audit_status() -> dict[str, Any]:
    head = git_value(AUDIT, "rev-parse", "HEAD")
    branch_status = git_value(AUDIT, "status", "--porcelain")
    ancestor = subprocess.run(
        ["git", "-C", str(AUDIT), "merge-base", "--is-ancestor", FROZEN_AUDIT_SOURCE_HEAD, "HEAD"],
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Frozen root-audit source commit is not an ancestor of the artifact checkout")
    if branch_status:
        raise RuntimeError(f"Root-audit checkout has uncommitted changes: {branch_status!r}")
    return {"source_head": FROZEN_AUDIT_SOURCE_HEAD, "artifact_head_at_freeze": head, "working_tree_clean": True}


def _hash_inventory() -> dict[str, str]:
    return {name: sha256_file(rel_path(name)) for name in RELATIVE_FILES}


def _source_hashes() -> dict[str, str]:
    return {p: sha256_file(ROOT / p) for p in (SOURCE_SELF, SOURCE_VERIFIER, SOURCE_PROTOCOL)}


def _committed_freeze_hash() -> str:
    relative = FREEZE.relative_to(ROOT).as_posix()
    payload = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
        check=True, capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def freeze_inputs() -> dict[str, Any]:
    if FREEZE.exists():
        raise FileExistsError(f"Refusing to overwrite pre-access freeze: {FREEZE}")
    audit_freeze = json.loads(rel_path("audit_input_freeze").read_text(encoding="utf-8"))
    if audit_freeze.get("head") != FROZEN_AUDIT_SOURCE_HEAD:
        raise RuntimeError("Audit INPUT_FREEZE head differs from the declared root source")
    audit_status = _root_audit_status()
    input_hashes = _hash_inventory()
    for name, expected in EXPECTED_HASHES.items():
        if input_hashes.get(name) != expected:
            raise RuntimeError(f"Frozen input hash mismatch for {name}: {input_hashes.get(name)}")
    source_hashes = _source_hashes()
    root_head = git_value(ROOT, "rev-parse", "HEAD")
    branch = git_value(ROOT, "branch", "--show-current")
    if git_value(ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked source tree must be committed before the pre-access freeze")
    versions = {}
    for package in ("numpy", "scikit-learn", "matplotlib", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    freeze = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_CACHED_VECTOR_ACCESS",
        "diagnostic_id": "CLS_SIMPLE_RANK_ALLOCATION_DIAGNOSTIC_V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": root_head,
        "branch": branch,
        "source_hashes": source_hashes,
        "root_audit": audit_status,
        "root_audit_input_freeze_sha256": input_hashes["audit_input_freeze"],
        "input_hashes": input_hashes,
        "allowed_npz_members": {str(k): v for k, v in MEMBERS.items()},
        "views": VIEWS,
        "recipe": {
            "complex_present": "0.5*pP + 0.5*pA; >= 0.50",
            "simple_present": "pA; >= 0.50",
            "missing_for_all_operational_full_rows": "pA/CB fallback; >= 0.53230976129477",
            "matched_policy": "within complete origin-cycle; compare top-K pA and top-K complex score",
            "origin_sensitivity": "within origin and selected view block; unique pool keeps E67/E75 blocks separate",
            "tie_rule": "descending score, then ascending string record_id",
            "topk_count_source": "fixed-threshold predictions only; labels are excluded from K selection",
            "auc_ap": "present rows only; descriptive, not causal",
        },
        "resource_policy": {
            "policy_source": input_hashes["audit_protocol"],
            "policy_mode": "single-job saved-vector replay; original shared-lock gate",
            "max_incremental_bytes": RESOURCE_ESTIMATE_BYTES,
            "minimum_available_at_start_bytes": RESOURCE_START_BYTES,
            "minimum_available_at_boundaries_bytes": RESOURCE_RESERVE_BYTES,
            "per_lane_lock_required": True,
            "shared_heavy_lock": str(SHARED_LOCK_PATH),
        },
        "novelty_review": {
            "documents": ["novelty_b01_summary", "novelty_a02_neighbors", "novelty_a02_cgeom", "novelty_a02_continuation"],
            "exact_duplicate_found": False,
            "note": "B01 has matched-positive-count tables but compares B01/A02/Sparse25, not the frozen POS half-blend against A02.",
        },
        "software_versions": {"python": platform.python_version(), **versions},
        "prohibited": {
            "raw_train_label_access": False,
            "test_or_sample_access": False,
            "e79_access": False,
            "new_fits": 0,
            "new_model_inference": 0,
            "kaggle_access": False,
        },
    }
    encoded = json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    with FREEZE.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": freeze["status"], "source_commit": root_head,
                      "source_hashes": source_hashes, "input_hashes": input_hashes}, indent=2))
    return freeze


def verify_frozen_inputs() -> dict[str, Any]:
    if not FREEZE.is_file():
        raise FileNotFoundError(f"Missing pre-access freeze: {FREEZE}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_CACHED_VECTOR_ACCESS":
        raise RuntimeError("Unexpected pre-access freeze status")
    if freeze.get("source_hashes") != _source_hashes():
        raise RuntimeError("Diagnostic source changed after the pre-access freeze")
    expected_views = {key: [list(segment) for segment in segments] for key, segments in VIEWS.items()}
    if freeze.get("views") != expected_views:
        raise RuntimeError("Frozen view definitions differ from the executable protocol")
    if freeze.get("allowed_npz_members") != {str(k): v for k, v in MEMBERS.items()}:
        raise RuntimeError("Frozen cached-array allowlist differs from the executable protocol")
    expected_recipe = {
        "complex_present": "0.5*pP + 0.5*pA; >= 0.50",
        "simple_present": "pA; >= 0.50",
        "missing_for_all_operational_full_rows": "pA/CB fallback; >= 0.53230976129477",
        "matched_policy": "within complete origin-cycle; compare top-K pA and top-K complex score",
        "origin_sensitivity": "within origin and selected view block; unique pool keeps E67/E75 blocks separate",
        "tie_rule": "descending score, then ascending string record_id",
        "topk_count_source": "fixed-threshold predictions only; labels are excluded from K selection",
        "auc_ap": "present rows only; descriptive, not causal",
    }
    if freeze.get("recipe") != expected_recipe:
        raise RuntimeError("Frozen recipe differs from the executable protocol")
    expected_resource_policy = {
        "policy_source": EXPECTED_HASHES["audit_protocol"],
        "policy_mode": "single-job saved-vector replay; original shared-lock gate",
        "max_incremental_bytes": RESOURCE_ESTIMATE_BYTES,
        "minimum_available_at_start_bytes": RESOURCE_START_BYTES,
        "minimum_available_at_boundaries_bytes": RESOURCE_RESERVE_BYTES,
        "per_lane_lock_required": True,
        "shared_heavy_lock": str(SHARED_LOCK_PATH),
    }
    if freeze.get("resource_policy") != expected_resource_policy:
        raise RuntimeError("Frozen resource policy differs from the authorized single-job gate")
    if _committed_freeze_hash() != sha256_file(FREEZE):
        raise RuntimeError("Pre-access freeze differs from the committed artifact")
    actual_inputs = _hash_inventory()
    if freeze.get("input_hashes") != actual_inputs:
        raise RuntimeError("An upstream source/cache/novelty document changed after freeze")
    current_audit = _root_audit_status()
    if current_audit.get("source_head") != FROZEN_AUDIT_SOURCE_HEAD:
        raise RuntimeError("Frozen root-audit source commit changed")
    if not current_audit.get("working_tree_clean"):
        raise RuntimeError("Root-audit worktree is not clean")
    return freeze


def stable_topk(indices: Any, scores: Any, ids: Any, k: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    ix = np.asarray(indices, dtype=np.int64)
    if k < 0 or k > len(ix):
        raise ValueError(f"Invalid top-K count {k} for n={len(ix)}")
    if k == 0:
        selected = ix[:0]
        boundary = None
        ties = 0
        selected_ties = 0
    elif k == len(ix):
        selected = ix[np.lexsort((ids[ix], -scores[ix]))]
        boundary = float(scores[selected[-1]])
        ties = int(np.sum(scores[ix] == boundary))
        selected_ties = ties
    else:
        order = np.lexsort((ids[ix], -scores[ix]))
        selected = ix[order[:k]]
        boundary = float(scores[selected[-1]])
        ties = int(np.sum(scores[ix] == boundary))
        selected_ties = int(np.sum(scores[selected] == boundary))
    excluded = np.setdiff1d(ix, selected, assume_unique=False)
    hash_ids = lambda vals: hashlib.sha256("\n".join(sorted(str(x) for x in vals)).encode("utf-8")).hexdigest()
    detail = {
        "n_candidates": int(len(ix)), "k": int(k), "boundary_score": boundary,
        "n_equal_boundary_score": int(ties), "selected_equal_boundary_score": int(selected_ties),
        "excluded_equal_boundary_score": int(ties - selected_ties),
        "selected_ids_sha256": hash_ids(ids[selected]), "excluded_ids_sha256": hash_ids(ids[excluded]),
    }
    return selected, detail


def confusion_metrics(y: Any, pred: Any) -> dict[str, Any]:
    import numpy as np

    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=bool)
    if y.shape != pred.shape:
        raise ValueError("Label and prediction shapes differ")
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tp = int(np.sum((y == 1) & pred))
    div = lambda num, den: float(num / den) if den else 0.0
    f1_pos = div(2 * tp, 2 * tp + fp + fn)
    f1_neg = div(2 * tn, 2 * tn + fp + fn)
    return {
        "n": int(len(y)), "actual_positive": int(np.sum(y == 1)), "actual_negative": int(np.sum(y == 0)),
        "predicted_positive": int(np.sum(pred)), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "macro_f1": float((f1_pos + f1_neg) / 2.0), "positive_f1": f1_pos, "negative_f1": f1_neg,
        "positive_precision": div(tp, tp + fp), "positive_recall": div(tp, tp + fn),
        "negative_precision": div(tn, tn + fn), "negative_recall": div(tn, tn + fp),
    }


def load_origins() -> dict[int, dict[str, Any]]:
    import numpy as np

    path = rel_path("a02_policy_predictions")
    output = {}
    with np.load(path, allow_pickle=False) as bundle:
        for origin, names in MEMBERS.items():
            required = set(names)
            if not required.issubset(bundle.files):
                raise RuntimeError(f"Origin E{origin}: required cached members absent")
            prefix = f"E{origin}_"
            ids = np.asarray(bundle[prefix + "ids"]).astype(str)
            cycles = np.asarray(bundle[prefix + "cycles"], dtype=np.int64)
            missing = np.asarray(bundle[prefix + "missing"], dtype=bool)
            y = np.asarray(bundle[prefix + "y"], dtype=np.int8)
            p_a = np.asarray(bundle[prefix + "pA"], dtype=np.float64)
            p_p = np.asarray(bundle[prefix + "pP"], dtype=np.float64)
            n = len(ids)
            if not (cycles.shape == missing.shape == y.shape == p_a.shape == p_p.shape == (n,)):
                raise RuntimeError(f"Origin E{origin}: vector shapes differ")
            if len(set(ids.tolist())) != n:
                raise RuntimeError(f"Origin E{origin}: duplicate IDs")
            if not np.isin(y, [0, 1]).all() or not np.isfinite(p_a).all():
                raise RuntimeError(f"Origin E{origin}: invalid labels or pA values")
            present = ~missing
            if not np.isfinite(p_p[present]).all():
                raise RuntimeError(f"Origin E{origin}: non-finite present pP")
            if ((p_a < 0) | (p_a > 1)).any() or ((p_p[present] < 0) | (p_p[present] > 1)).any():
                raise RuntimeError(f"Origin E{origin}: probability outside [0,1]")
            output[origin] = {"ids": ids, "cycles": cycles, "missing": missing, "y": y,
                              "pA": p_a, "pP": p_p}
    return output


def assemble_view(origins: dict[int, dict[str, Any]], segments: list[tuple[int, int, int]]) -> dict[str, Any]:
    import numpy as np

    pieces = {k: [] for k in ("ids", "origins", "cycles", "missing", "y", "pA", "pP")}
    for origin, lo, hi in segments:
        d = origins[origin]
        mask = (d["cycles"] >= lo) & (d["cycles"] <= hi)
        for key in ("ids", "missing", "y", "pA", "pP"):
            pieces[key].append(d[key][mask])
        pieces["cycles"].append(d["cycles"][mask])
        pieces["origins"].append(np.full(int(mask.sum()), origin, dtype=np.int16))
    out = {k: np.concatenate(v) for k, v in pieces.items()}
    if len(set(out["ids"].tolist())) != len(out["ids"]):
        raise RuntimeError("View contains duplicate IDs; overlapping origins are not allowed")
    return out


def set_hash(values: Any) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in values)).encode("utf-8")).hexdigest()


def ordered_ids_hash(values: Any) -> str:
    return hashlib.sha256("\n".join(str(x) for x in values).encode("utf-8")).hexdigest()


def overlap_metrics(ids: Any, present: Any, predictions: dict[str, Any]) -> dict[str, Any]:
    sets = {name: set(str(x) for x in ids[present & pred]) for name, pred in predictions.items()}
    base_complex = sets["complex_threshold"]
    base_simple = sets["simple_threshold"]
    output = {}
    for name, values in sets.items():
        union_c = values | base_complex
        union_s = values | base_simple
        output[name] = {
            "present_positive_ids_sha256": set_hash(values),
            "overlap_with_complex": int(len(values & base_complex)),
            "jaccard_with_complex": float(len(values & base_complex) / len(union_c)) if union_c else 1.0,
            "overlap_with_simple": int(len(values & base_simple)),
            "jaccard_with_simple": float(len(values & base_simple) / len(union_s)) if union_s else 1.0,
            "changed_rows_vs_complex": int(len(values ^ base_complex)),
            "changed_rows_vs_simple": int(len(values ^ base_simple)),
        }
    return output


def build_policy_predictions(view: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np

    ids, cycles, origins = view["ids"], view["cycles"], view["origins"]
    missing, y, p_a, p_p = view["missing"], view["y"], view["pA"], view["pP"]
    present = ~missing
    complex_score = 0.5 * p_p + 0.5 * p_a
    predictions = {
        "complex_threshold": np.where(missing, p_a >= MISSING_THRESHOLD, complex_score >= PRESENT_THRESHOLD),
        "simple_threshold": np.where(missing, p_a >= MISSING_THRESHOLD, p_a >= PRESENT_THRESHOLD),
        "simple_topk_cycle_at_complex_count": np.zeros(len(ids), dtype=bool),
        "complex_topk_cycle_at_simple_count": np.zeros(len(ids), dtype=bool),
        "simple_topk_origin_at_complex_count": np.zeros(len(ids), dtype=bool),
        "complex_topk_origin_at_simple_count": np.zeros(len(ids), dtype=bool),
    }
    cycle_details: list[dict[str, Any]] = []
    origin_details: list[dict[str, Any]] = []

    def apply_group(group_name: str, key: tuple[int, ...], ix: Any, matching_level: str) -> None:
        ix = np.asarray(ix, dtype=np.int64)
        pi = ix[present[ix]]
        complex_idx = pi[complex_score[pi] >= PRESENT_THRESHOLD]
        simple_idx = pi[p_a[pi] >= PRESENT_THRESHOLD]
        simple_sel, simple_ties = stable_topk(pi, p_a, ids, len(complex_idx))
        complex_sel, complex_ties = stable_topk(pi, complex_score, ids, len(simple_idx))
        # Keep public names explicit and stable.
        simple_name = "simple_topk_cycle_at_complex_count" if matching_level == "cycle" else "simple_topk_origin_at_complex_count"
        complex_name = "complex_topk_cycle_at_simple_count" if matching_level == "cycle" else "complex_topk_origin_at_simple_count"
        predictions[simple_name][simple_sel] = True
        predictions[complex_name][complex_sel] = True
        detail = {
            "group": group_name,
            "key": list(key),
            "n_rows": int(len(ix)),
            "n_present": int(len(pi)),
            "k_complex": int(len(complex_idx)),
            "k_simple": int(len(simple_idx)),
            "simple_topk_at_complex_count": simple_ties,
            "complex_topk_at_simple_count": complex_ties,
            "simple_topk_overlap_with_complex": int(len(set(ids[simple_sel]) & set(ids[complex_idx]))),
            "complex_topk_overlap_with_simple": int(len(set(ids[complex_sel]) & set(ids[simple_idx]))),
        }
        (cycle_details if matching_level == "cycle" else origin_details).append(detail)

    for origin in sorted(set(origins.tolist())):
        for cycle in sorted(set(cycles[origins == origin].tolist())):
            ix = np.flatnonzero((origins == origin) & (cycles == cycle))
            apply_group(f"E{origin}_cycle_{cycle}", (int(origin), int(cycle)), ix, "cycle")
    for origin in sorted(set(origins.tolist())):
        ix = np.flatnonzero(origins == origin)
        apply_group(f"E{origin}_origin_block", (int(origin),), ix, "origin")

    metrics = {}
    present_metrics = {}
    for name in POLICY_NAMES:
        metrics[name] = confusion_metrics(y, predictions[name])
        present_metrics[name] = confusion_metrics(y[present], predictions[name][present])
    overlaps = overlap_metrics(ids, present, predictions)
    from sklearn.metrics import average_precision_score, roc_auc_score
    ranking = {
        "present_rows": int(present.sum()),
        "actual_positive": int(np.sum(y[present] == 1)),
        "actual_negative": int(np.sum(y[present] == 0)),
        "simple_pA": {
            "roc_auc": float(roc_auc_score(y[present], p_a[present])),
            "average_precision": float(average_precision_score(y[present], p_a[present])),
        },
        "complex_half_blend": {
            "roc_auc": float(roc_auc_score(y[present], complex_score[present])),
            "average_precision": float(average_precision_score(y[present], complex_score[present])),
        },
    }
    selection = {
        name: {"present_positive_count": int(np.sum(present & pred)), "full_positive_count": int(np.sum(pred))}
        for name, pred in predictions.items()
    }
    return ({"operational_full": metrics, "present_only": present_metrics},
            {"ranking_present_only": ranking, "selection_counts": selection, "overlap_present_only": overlaps},
            {"per_cycle": cycle_details, "origin_level_sensitivity": origin_details})


def calculate() -> dict[str, Any]:
    import numpy as np

    if RESULTS.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULTS}")
    freeze = verify_frozen_inputs()
    with lane_lock_and_ram_gate() as gate_receipt:
        origins = load_origins()
        views: dict[str, Any] = {}
        for name, segments in VIEWS.items():
            view = assemble_view(origins, segments)
            if name == "UNIQUE" and len(view["ids"]) != 20_903:
                raise RuntimeError(f"Unique pool row count differs from 20,903: {len(view['ids'])}")
            present = ~view["missing"]
            base_metrics, extras, matching = build_policy_predictions(view)
            views[name] = {
                "segments": [list(s) for s in segments],
                "denominator": {
                    "rows": int(len(view["ids"])), "present": int(present.sum()),
                    "missing": int(view["missing"].sum()),
                    "actual_positive": int(np.sum(view["y"] == 1)),
                    "actual_negative": int(np.sum(view["y"] == 0)),
                    "ordered_ids_sha256": ordered_ids_hash(view["ids"]),
                    "unique_ids": int(len(set(view["ids"].tolist()))),
                },
                **base_metrics,
                **extras,
                "matching": matching,
            }
        end_available = resource_available_bytes()
        if end_available < RESOURCE_RESERVE_BYTES:
            raise RuntimeError(f"RAM reserve failed after computation: {end_available}")
        gate_receipt["available_end_bytes"] = end_available
        gate_receipt["reserve_pass"] = True
        result = {
            "schema_version": 1,
            "status": "CALCULATED_PENDING_INDEPENDENT_QA",
            "diagnostic_id": "CLS_SIMPLE_RANK_ALLOCATION_DIAGNOSTIC_V1",
            "pre_access_freeze_sha256": sha256_file(FREEZE),
            "source_commit": freeze["source_commit"],
            "input_hashes": freeze["input_hashes"],
            "recipe": freeze["recipe"],
            "views": views,
            "resource": gate_receipt,
            "prohibited_access": freeze["prohibited"],
        }
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with RESULTS.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"status": result["status"], "rows": {k: v["denominator"]["rows"] for k, v in views.items()},
                          "free_ram_start": gate_receipt["available_start_bytes"],
                          "free_ram_end": gate_receipt["available_end_bytes"], "result": str(RESULTS)}, indent=2))
        return result


def generate_plots() -> list[str]:
    if not QA.is_file() or not RESULTS.is_file():
        raise RuntimeError("Independent QA and results must exist before plotting")
    qa = json.loads(QA.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS":
        raise RuntimeError("Independent QA did not PASS; refusing plots")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    plot1 = OUT / "MATCHED_COUNT_MACRO_F1.png"
    plot2 = OUT / "PRESENT_RANKING_AUC_AP.png"
    if plot1.exists() or plot2.exists():
        raise FileExistsError("Refusing to overwrite diagnostic plots")
    with lane_lock_and_ram_gate() as plot_resource:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(VIEWS)
        display = ["E67 W4", "E67 W5", "E67 W6", "E67 H12", "E75 W6", "Unique"]
        x = list(range(len(names)))
        width = 0.2
        policies = [
            ("complex_threshold", "POS/A02 fixed threshold"),
            ("simple_threshold", "A02 fixed threshold"),
            ("simple_topk_cycle_at_complex_count", "A02 matched per cycle"),
            ("complex_topk_cycle_at_simple_count", "POS/A02 matched per cycle"),
        ]
        fig, ax = plt.subplots(figsize=(11, 5.3))
        for j, (policy, label) in enumerate(policies):
            vals = [results["views"][name]["present_only"][policy]["macro_f1"] for name in names]
            ax.bar([i + (j - 1.5) * width for i in x], vals, width, label=label)
        ax.set_xticks(x, display)
        ax.set_ylabel("Present-only Macro F1")
        ax.set_title("Fixed thresholds and matched positive counts")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot1, dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
        for ax, metric, title in zip(axes, ("roc_auc", "average_precision"), ("ROC AUC", "Average precision")):
            delta = [
                results["views"][name]["ranking_present_only"]["complex_half_blend"][metric]
                - results["views"][name]["ranking_present_only"]["simple_pA"][metric]
                for name in names
            ]
            ax.bar(x, delta, color=["#457b9d" if v >= 0 else "#c75b39" for v in delta])
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_title(f"Complex minus A02: {title}")
            ax.set_ylabel("Difference")
            ax.set_xticks(x, display, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Ranking metrics on present-stream rows only")
        fig.tight_layout()
        fig.savefig(plot2, dpi=170)
        plt.close(fig)
        plot_resource["plots_written"] = 2
    print(json.dumps({"status": "PLOTS_WRITTEN_AFTER_INDEPENDENT_QA", "plots": [str(plot1), str(plot2)]}, indent=2))
    return [str(plot1), str(plot2)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true", help="hash inputs and rules only; never loads NPZ arrays")
    modes.add_argument("--calculate", action="store_true", help="calculate frozen zero-fit diagnostic")
    modes.add_argument("--plots", action="store_true", help="write at most two plots after independent QA PASS")
    args = parser.parse_args()
    if args.freeze:
        freeze_inputs()
    elif args.calculate:
        calculate()
    else:
        generate_plots()


if __name__ == "__main__":
    main()
