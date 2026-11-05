"""Zero-fit, target-free coefficient/logit decomposition for CLS POS ridge R1."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_FREEZE.json"
PREVIOUS_FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_1_FREEZE.json"
OUTPUT_JSON_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2.json"
OUTPUT_MD_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2.md"
PREFLIGHT_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_PREFLIGHT.json"
R8_FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R8.json"
R9_ADAPTER_FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_VERIFIER_ADAPTER_R9_FREEZE.json"
PRED_MANIFEST_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/PREDICTIONS_MANIFEST_R1.json"
RESULTS_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESULTS_R1.json"
RESULTS_MD_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_REPORT_R1.md"
METRIC_VERIFY_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/METRICS_VERIFY_R1_R9.json"
INCIDENT_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_ACCESS_INCIDENT_01.json"
PROTOCOL_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/protocol.json"
POS_BAG_ROOT = "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
PILOT_ROOT = "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
RIDGE_ROOT = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1"
SOURCE_RELS = [
    "scripts/diagnose_cls_pos_frequency_ridge_mean_one_energy_r1.py",
    "scripts/evaluate_optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_frequency_ridge_eval_r9_adapter.py",
    "src/models/pos_frequency_ridge_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "tests/test_cls_pos_frequency_ridge_mean_one_energy_r1.py",
]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze_signature(freeze: dict[str, Any]) -> str:
    body = dict(freeze)
    claim = body.pop("freeze_sha256", None)
    actual = hashlib.sha256(canonical_json(body)).hexdigest()
    if claim != actual:
        raise RuntimeError("Freeze canonical self-signature mismatch")
    return str(claim)


def verify_pinned_file(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Raw file SHA256 differs from frozen input: {path.name}")
    return actual


def read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def committed_binding(rel: str) -> dict[str, str]:
    blob = git("rev-parse", f"HEAD:{rel}")
    if blob != git("hash-object", rel):
        raise RuntimeError(f"Expected committed file is modified or untracked: {rel}")
    return {"git_blob": blob, "commit": git("log", "-1", "--format=%H", "--", rel)}


def is_tracked(rel: str) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                            text=True, capture_output=True, check=False)
    return result.returncode == 0


def selected_input_paths() -> list[str]:
    paths = {
        R8_FREEZE_REL, R9_ADAPTER_FREEZE_REL, PRED_MANIFEST_REL, RESULTS_REL,
        RESULTS_MD_REL, METRIC_VERIFY_REL, INCIDENT_REL, PROTOCOL_REL,
        PREVIOUS_FREEZE_REL,
        f"{POS_BAG_ROOT}/FREEZE.json",
        f"{POS_BAG_ROOT}/PREDICTIONS_READY.json",
        f"{POS_BAG_ROOT}/PRELABEL_VERIFICATION.json",
        f"{PILOT_ROOT}/FREEZE.json",
    }
    for origin in (67, 75):
        paths.update({
            f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json",
            f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
            f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json",
            f"{RIDGE_ROOT}/maps/penalty_map_E{origin}.npz",
            f"{RIDGE_ROOT}/maps/penalty_map_E{origin}.json",
            f"{RIDGE_ROOT}/units/E{origin}_ALIGNED.npz",
            f"{RIDGE_ROOT}/units/E{origin}_PERMUTED.npz",
            f"{RIDGE_ROOT}/receipts/E{origin}_ALIGNED.json",
            f"{RIDGE_ROOT}/receipts/E{origin}_PERMUTED.json",
        })
        manifest = read_json(f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json")
        for key in ("score_base", "score_position"):
            entry = manifest.get("matrices", {}).get(key, {})
            rel = entry.get("path")
            if not isinstance(rel, str) or not rel.startswith(f"{PILOT_ROOT}/E{origin}/matrices/"):
                raise RuntimeError(f"Unexpected E{origin} matrix path for {key}")
            paths.add(rel)
    return sorted(paths)


def build_freeze() -> dict[str, Any]:
    r8 = read_json(R8_FREEZE_REL)
    read_json(R9_ADAPTER_FREEZE_REL)
    previous = read_json(PREVIOUS_FREEZE_REL)
    previous_claim = verify_freeze_signature(previous)
    if previous.get("status") != "FROZEN_ADDITIVE_ZERO_FIT_PRE_DIAGNOSTIC_R1_1":
        raise RuntimeError("Prior R1.1 diagnostic freeze is invalid")
    for rel, expected in previous.get("input_hashes", {}).items():
        if sha256_file(ROOT / rel) != expected:
            raise RuntimeError(f"Prior R1.1 bound input changed: {rel}")
    old_source_binding = previous.get("source_commits", {}).get(SOURCE_RELS[0], {})
    old_commit = old_source_binding.get("commit")
    if (not old_commit
            or git("rev-parse", f"{old_commit}:{SOURCE_RELS[0]}") != old_source_binding.get("git_blob")):
        raise RuntimeError("Prior R1 diagnostic source commit does not match its frozen blob")
    selected = selected_input_paths()
    input_hashes: dict[str, str] = {}
    input_commits: dict[str, dict[str, str]] = {}
    for rel in selected:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"Required frozen diagnostic input is missing: {rel}")
        digest = sha256_file(path)
        input_hashes[rel] = digest
        if rel in r8.get("input_hashes", {}) and r8["input_hashes"][rel] != digest:
            raise RuntimeError(f"Input differs from R8 source freeze: {rel}")
        if rel in r8.get("source_hashes", {}) and r8["source_hashes"][rel] != digest:
            raise RuntimeError(f"Source differs from R8 source freeze: {rel}")
        if is_tracked(rel):
            input_commits[rel] = committed_binding(rel)
        else:
            # Run artifacts are intentionally ignored by Git; their byte hashes
            # are bound above and checked against the upstream R8 freeze.
            if rel in r8.get("input_hashes", {}) and r8["input_hashes"][rel] != digest:
                raise
            if rel not in r8.get("input_hashes", {}):
                raise RuntimeError(f"Untracked diagnostic input lacks an upstream byte-hash binding: {rel}")
    source_hashes = {}
    source_commits = {}
    for rel in SOURCE_RELS:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"Diagnostic source missing: {rel}")
        source_hashes[rel] = sha256_file(path)
        source_commits[rel] = committed_binding(rel)
        if rel in r8.get("source_hashes", {}) and r8["source_hashes"][rel] != source_hashes[rel]:
            raise RuntimeError(f"Diagnostic dependency differs from R8 freeze: {rel}")
    results = read_json(RESULTS_REL)
    metric_verification = read_json(METRIC_VERIFY_REL)
    if (results.get("status") != "SCORED_DISCLOSED_DEVELOPMENT_EVALUATION"
            or results.get("test_or_samplesubmission_accessed") is not False
            or results.get("new_fits") != 0
            or metric_verification.get("status") != "PASS_INDEPENDENT_METRIC_AND_ROW_RECONSTRUCTION"
            or metric_verification.get("test_or_samplesubmission_accessed") is not False):
        raise RuntimeError("Current verified result boundary does not match this diagnostic scope")
    body: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_id": "CLS_POS_FREQUENCY_RIDGE_MEAN_ONE_ENERGY_R1_2",
        "status": "FROZEN_ADDITIVE_ZERO_FIT_PRE_DIAGNOSTIC_R1_2",
        "supersedes_diagnostic_freeze": PREVIOUS_FREEZE_REL,
        "supersedes_diagnostic_freeze_sha256": previous_claim,
        "supersession_reason": "R1.2 corrects parent-freeze signature verification after R1.1 stopped before loading numerical arrays or labels; measurement logic and selected inputs are unchanged.",
        "source_commit_at_freeze": git("rev-parse", "HEAD"),
        "parent_r8_freeze_sha256": sha256_file(ROOT / R8_FREEZE_REL),
        "parent_r9_adapter_freeze_sha256": sha256_file(ROOT / R9_ADAPTER_FREEZE_REL),
        "source_hashes": source_hashes,
        "source_commits": source_commits,
        "input_hashes": input_hashes,
        "input_commits": input_commits,
        "r8_source_hashes": r8.get("source_hashes", {}),
        "r8_input_hashes": r8.get("input_hashes", {}),
        "r9_adapter_result_sha256": input_hashes[METRIC_VERIFY_REL],
        "measurement_plan": {
            "origins": [67, 75],
            "candidates": ["UNIFORM", "ALIGNED", "PERMUTED"],
            "frequency_partition": "exact frozen ALIGNED token weight < 1 versus >= 1, repeated across the three positional blocks",
            "delta_logits": "score_position @ (gamma_ALIGNED - gamma_UNIFORM), decomposed by the exact frozen ALIGNED weight partition",
            "delta_summary": ["signed mean", "mean absolute", "95th percentile absolute", "max absolute"],
            "coefficient_summary": ["L2 norm", "squared L2 energy", "same-coordinate fitted penalty quadratic without objective factor 1/2"],
            "outcome_metrics": "read from committed EVALUATION_RESULTS_R1 only; no new prediction, threshold, target, or score computation",
        },
        "scope": {
            "target_labels_read": False,
            "test_or_samplesubmission_accessed": False,
            "new_model_fits": 0,
            "new_predictions_or_counterfactual_scores": 0,
            "threshold_or_bin_search": False,
            "uses_only_saved_gamma_frequency_maps_position_matrices_and_current_results": True,
        },
    }
    body["freeze_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def verify_freeze() -> dict[str, Any]:
    rel = FREEZE_REL
    committed_binding(rel)
    freeze = read_json(rel)
    verify_freeze_signature(freeze)
    for name, items in (("source_hashes", freeze.get("source_hashes", {})),
                        ("input_hashes", freeze.get("input_hashes", {}))):
        for item_rel, expected in items.items():
            verify_pinned_file(ROOT / item_rel, expected)
    if sha256_file(ROOT / R8_FREEZE_REL) != freeze["parent_r8_freeze_sha256"]:
        raise RuntimeError("Parent R8 freeze changed")
    if sha256_file(ROOT / R9_ADAPTER_FREEZE_REL) != freeze["parent_r9_adapter_freeze_sha256"]:
        raise RuntimeError("Parent R9 verifier freeze changed")
    previous = read_json(PREVIOUS_FREEZE_REL)
    previous_claim = verify_freeze_signature(previous)
    if previous_claim != freeze["supersedes_diagnostic_freeze_sha256"]:
        raise RuntimeError("Prior R1.1 diagnostic freeze signature changed")
    return freeze


def array_stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    ax = np.abs(x)
    return {
        "n_rows": int(x.size),
        "mean_signed": float(np.mean(x)) if x.size else 0.0,
        "mean_absolute": float(np.mean(ax)) if x.size else 0.0,
        "p95_absolute": float(np.quantile(ax, 0.95)) if x.size else 0.0,
        "max_absolute": float(np.max(ax)) if x.size else 0.0,
    }


def partitioned_delta_logits(x: sparse.csr_matrix, delta_gamma: np.ndarray,
                             low_weight: np.ndarray) -> dict[str, Any]:
    low_weight = np.asarray(low_weight, dtype=bool).reshape(-1)
    delta_gamma = np.asarray(delta_gamma, dtype=np.float64).reshape(-1)
    if x.shape[1] != len(delta_gamma) or len(low_weight) != len(delta_gamma):
        raise ValueError("Sparse score matrix, delta gamma, and frequency partition widths differ")
    high_weight = ~low_weight
    total = np.asarray(x @ delta_gamma, dtype=np.float64).reshape(-1)
    low = np.asarray(x[:, low_weight] @ delta_gamma[low_weight], dtype=np.float64).reshape(-1)
    high = np.asarray(x[:, high_weight] @ delta_gamma[high_weight], dtype=np.float64).reshape(-1)
    reconstruction = low + high
    error = reconstruction - total
    low_l1 = float(np.sum(np.abs(low)))
    high_l1 = float(np.sum(np.abs(high)))
    total_l1 = float(np.sum(np.abs(total)))
    denom = low_l1 + high_l1
    return {
        "total": array_stats(total),
        "w_below_1": array_stats(low),
        "w_at_least_1": array_stats(high),
        "magnitude_share": {
            "w_below_1": low_l1 / denom if denom else 0.0,
            "w_at_least_1": high_l1 / denom if denom else 0.0,
            "definition": "partition sum(abs(delta logit)) divided by combined partition L1 magnitude",
        },
        "cross_partition_cancellation_fraction": max(0.0, 1.0 - total_l1 / denom) if denom else 0.0,
        "reconstruction": {
            "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
            "mean_abs_error": float(np.mean(np.abs(error))) if error.size else 0.0,
            "allclose_atol_1e-10_rtol_1e-10": bool(np.allclose(total, reconstruction, atol=1e-10, rtol=1e-10)),
        },
    }


def candidate_coefficient_summary(gamma: np.ndarray, fitted_weights: np.ndarray,
                                  low_weight: np.ndarray, *, tokens: int) -> dict[str, Any]:
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    w = np.asarray(fitted_weights, dtype=np.float64).reshape(-1)
    low = np.asarray(low_weight, dtype=bool).reshape(-1)
    if not (len(g) == len(w) == len(low)):
        raise ValueError("Gamma, fitted weights, and frequency partition dimensions differ")
    out: dict[str, Any] = {
        "coefficient_count": int(len(g)),
        "token_count_per_positional_block": int(tokens),
        "total": {
            "l2_norm": float(np.linalg.norm(g)),
            "squared_l2_energy": float(np.dot(g, g)),
            "fitted_penalty_quadratic_without_one_half": float(np.dot(w, g * g)),
            "fitted_weight_mean": float(np.mean(w)),
        },
        "frequency_partition": {},
    }
    for name, mask in (("w_below_1", low), ("w_at_least_1", ~low)):
        block = g[mask]
        squared = float(np.dot(block, block))
        out["frequency_partition"][name] = {
            "coefficient_count": int(mask.sum()),
            "l2_norm": float(np.linalg.norm(block)),
            "squared_l2_energy": squared,
            "share_of_unweighted_coefficient_energy": squared / float(np.dot(g, g)) if np.dot(g, g) else 0.0,
            "fitted_penalty_quadratic_without_one_half": float(np.dot(w[mask], block * block)),
            "mean_fitted_weight_on_partition": float(np.mean(w[mask])) if mask.any() else 0.0,
        }
    return out


def _load_gamma(rel: str, receipt_rel: str, *, fit_id: str,
                expected_arm: str) -> np.ndarray:
    from src.models.pos_frequency_ridge_v1 import sha256_array

    receipt = read_json(receipt_rel)
    if receipt.get("fit_id") != fit_id:
        raise RuntimeError(f"Gamma receipt fit_id mismatch: {fit_id}")
    if receipt.get("arm") != expected_arm and expected_arm != "REAL":
        raise RuntimeError(f"Gamma receipt arm mismatch: {fit_id}")
    if receipt.get("artifact_sha256") != sha256_file(ROOT / rel):
        raise RuntimeError(f"Gamma artifact receipt byte hash mismatch: {fit_id}")
    with np.load(ROOT / rel, allow_pickle=False) as z:
        if set(z.files) != {"gamma"}:
            raise RuntimeError(f"Gamma artifact has unexpected members: {fit_id}")
        gamma = np.asarray(z["gamma"], dtype=np.float64).reshape(-1)
    if not np.isfinite(gamma).all() or sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise RuntimeError(f"Gamma logical hash or finiteness mismatch: {fit_id}")
    return gamma


def load_penalty_map(origin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from src.models.pos_frequency_ridge_v1 import sha256_array

    map_rel = f"{RIDGE_ROOT}/maps/penalty_map_E{origin}.npz"
    receipt_rel = f"{RIDGE_ROOT}/maps/penalty_map_E{origin}.json"
    receipt = read_json(receipt_rel)
    if receipt.get("artifact_sha256") != sha256_file(ROOT / map_rel):
        raise RuntimeError(f"E{origin} penalty map byte hash mismatch")
    with np.load(ROOT / map_rel, allow_pickle=False) as z:
        if set(z.files) != {"document_frequency", "aligned_token_weights", "permuted_token_weights", "permutation"}:
            raise RuntimeError(f"E{origin} penalty map member set mismatch")
        df = np.asarray(z["document_frequency"], dtype=np.int64).reshape(-1)
        aligned = np.asarray(z["aligned_token_weights"], dtype=np.float64).reshape(-1)
        permuted = np.asarray(z["permuted_token_weights"], dtype=np.float64).reshape(-1)
        permutation = np.asarray(z["permutation"], dtype=np.int64).reshape(-1)
    if (len(df) == 0 or not (len(df) == len(aligned) == len(permuted) == len(permutation))
            or np.any(df <= 0) or np.any(aligned <= 0) or np.any(permuted <= 0)
            or not all(np.isfinite(v).all() for v in (aligned, permuted))
            or not np.isclose(float(aligned.mean()), 1.0, rtol=0.0, atol=1e-14)
            or not np.array_equal(np.sort(aligned), np.sort(permuted))):
        raise RuntimeError(f"E{origin} penalty map array/schema constraints failed")
    expected = {
        "df_sha256": sha256_array(df), "aligned_sha256": sha256_array(aligned),
        "permuted_sha256": sha256_array(permuted), "permutation_sha256": sha256_array(permutation),
    }
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"E{origin} penalty map logical receipt hashes mismatch")
    return df, aligned, permuted, receipt


def load_position_matrix(origin: int, expected_rows: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr

    manifest_rel = f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json"
    manifest = read_json(manifest_rel)
    matrices = manifest.get("matrices", {})
    base_item, pos_item = matrices.get("score_base"), matrices.get("score_position")
    if not isinstance(base_item, dict) or not isinstance(pos_item, dict):
        raise RuntimeError(f"E{origin} matrix manifest missing score blocks")
    base_rel, augmented_rel = base_item.get("path"), pos_item.get("path")
    if not isinstance(base_rel, str) or not isinstance(augmented_rel, str):
        raise RuntimeError(f"E{origin} matrix paths malformed")
    if (sha256_file(ROOT / base_rel) != base_item.get("sha256")
            or sha256_file(ROOT / augmented_rel) != pos_item.get("sha256")):
        raise RuntimeError(f"E{origin} score matrix byte hash mismatch")
    base = sparse.load_npz(ROOT / base_rel).tocsr()
    augmented = sparse.load_npz(ROOT / augmented_rel).tocsr()
    for matrix, item, label in ((base, base_item, "score_base"), (augmented, pos_item, "score_position")):
        if (list(matrix.shape) != item.get("shape") or str(matrix.dtype) != item.get("dtype")
                or sha256_csr(matrix) != item.get("logical_sha256")):
            raise RuntimeError(f"E{origin} {label} matrix logical manifest mismatch")
    if (base.shape[0] != expected_rows or augmented.shape[0] != expected_rows
            or base.shape[1] >= augmented.shape[1]):
        raise RuntimeError(f"E{origin} saved score matrix dimensions differ from results")
    prefix_delta = (augmented[:, :base.shape[1]] - base).tocsr()
    prefix_delta.eliminate_zeros()
    if prefix_delta.nnz:
        raise RuntimeError(f"E{origin} augmented score_position base prefix changed")
    position = augmented[:, base.shape[1]:].tocsr()
    return position, {
        "matrix_manifest_sha256": sha256_file(ROOT / manifest_rel),
        "score_base_path": base_rel,
        "score_base_sha256": base_item["sha256"],
        "score_position_augmented_path": augmented_rel,
        "score_position_augmented_sha256": pos_item["sha256"],
        "score_position_augmented_shape": list(augmented.shape),
        "position_block_shape": list(position.shape),
        "position_block_nnz": int(position.nnz),
        "score_base_prefix_exact": True,
    }


def run_integration_preflight() -> dict[str, Any]:
    freeze = verify_freeze()
    results = read_json(RESULTS_REL)
    checks = {}
    for origin, view_name, expected_rows in ((67, "E67_H12", 20903), (75, "E75_W6", 8491)):
        view = results["views"][view_name]
        if int(view["n"]) != expected_rows or int(view["present_n"]) + int(view["missing_n"]) != expected_rows:
            raise RuntimeError(f"E{origin} stored result denominators do not reconcile")
        df, aligned, permuted, _ = load_penalty_map(origin)
        position, matrix_lineage = load_position_matrix(origin, expected_rows)
        width = 3 * len(df)
        gamma_paths = {
            "UNIFORM": (f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
                        f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json", "REAL"),
            "ALIGNED": (f"{RIDGE_ROOT}/units/E{origin}_ALIGNED.npz",
                        f"{RIDGE_ROOT}/receipts/E{origin}_ALIGNED.json", "ALIGNED"),
            "PERMUTED": (f"{RIDGE_ROOT}/units/E{origin}_PERMUTED.npz",
                         f"{RIDGE_ROOT}/receipts/E{origin}_PERMUTED.json", "PERMUTED"),
        }
        gamma_sizes = {}
        for candidate, (artifact, receipt, arm) in gamma_paths.items():
            gamma = _load_gamma(artifact, receipt, fit_id=(
                f"POS_BAG_OFFSET_V1_E{origin}_REAL" if candidate == "UNIFORM" else f"E{origin}_{candidate}"
            ), expected_arm=arm)
            if len(gamma) != width:
                raise RuntimeError(f"E{origin} {candidate} gamma does not match 3*token_count")
            gamma_sizes[candidate] = int(len(gamma))
        if position.shape[1] != width or not np.isclose(float(aligned.mean()), 1.0, atol=1e-14, rtol=0.0):
            raise RuntimeError(f"E{origin} score-position/map dimension or normalization mismatch")
        checks[str(origin)] = {
            "result_rows": expected_rows,
            "present_rows": int(view["present_n"]),
            "missing_rows": int(view["missing_n"]),
            "token_count": int(len(df)),
            "position_columns": int(position.shape[1]),
            "gamma_sizes": gamma_sizes,
            "partition_counts": {
                "w_below_1": int(np.count_nonzero(aligned < 1.0)),
                "w_at_least_1": int(np.count_nonzero(aligned >= 1.0)),
            },
            "permuted_multiset_matches_aligned": bool(np.array_equal(np.sort(aligned), np.sort(permuted))),
            "matrix_prefix_and_logical_hashes_pass": matrix_lineage["score_base_prefix_exact"],
            "matrix_lineage": matrix_lineage,
        }
    receipt = {
        "schema_version": 1,
        "status": "PASS_R1_2_INTEGRATION_PREFLIGHT_NO_MEASUREMENT",
        "freeze_sha256": freeze["freeze_sha256"],
        "origins": checks,
        "source_and_input_hashes_verified": True,
        "helper_imports_and_gamma_receipts_verified": True,
        "score_matrix_manifests_and_base_prefix_verified": True,
        "target_labels_read": False,
        "new_predictions_or_counterfactual_performance": 0,
        "new_fits": 0,
    }
    dest = ROOT / PREFLIGHT_REL
    if dest.exists():
        raise RuntimeError("Refusing to overwrite integration preflight receipt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def analyze() -> dict[str, Any]:
    freeze = verify_freeze()
    from src.models.pos_frequency_ridge_v1 import sha256_array

    preflight = read_json(PREFLIGHT_REL)
    if (preflight.get("status") != "PASS_R1_2_INTEGRATION_PREFLIGHT_NO_MEASUREMENT"
            or preflight.get("freeze_sha256") != freeze.get("freeze_sha256")):
        raise RuntimeError("R1.2 integration preflight is absent or bound to another freeze")
    results = read_json(RESULTS_REL)
    if results.get("status") != "SCORED_DISCLOSED_DEVELOPMENT_EVALUATION":
        raise RuntimeError("Current evaluation results are not the frozen disclosed R1 output")
    output: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_id": "CLS_POS_FREQUENCY_RIDGE_MEAN_ONE_ENERGY_R1",
        "status": "COMPLETED_ZERO_FIT_TARGET_FREE_MECHANISM_DIAGNOSTIC",
        "freeze_sha256": freeze["freeze_sha256"],
        "source_commit_at_freeze": freeze["source_commit_at_freeze"],
        "integration_preflight_sha256": sha256_file(ROOT / PREFLIGHT_REL),
        "target_labels_read": False,
        "test_or_samplesubmission_accessed": False,
        "new_fits": 0,
        "new_predictions_or_counterfactual_performance": 0,
        "frequency_partition": {
            "rule": "w < 1 versus w >= 1 using exact frozen ALIGNED mean-one token weights, repeated over all three positional blocks",
            "no_data_optimized_thresholds": True,
        },
        "origins": {},
        "previously_reported_evaluation": {
            "mechanism_gate": results["mechanism_gate"],
            "competition_gate": results["competition_gate"],
            "views": {},
        },
        "reference_provenance_correction": {
            "evaluation_code_label": "MIX50_ORIGINAL",
            "actual_primary_comparator": {
                "present": "0.5 * original uniform POS probability + 0.5 * pA",
                "missing": "pA exactly, primary threshold 0.50",
            },
            "archived_historical_A02_MIX50": {
                "present": "0.5 * pP + 0.5 * pA",
                "missing": "pP exactly",
            },
            "note": "MIX50_ORIGINAL is a code label for the current primary reference; it is not the archived historical A02 vector on missing-stream rows. Scores are unchanged.",
        },
        "interpretation_limits": [
            "The w<1 partition is selected by the exact fitted frequency map and its mean-one normalization; the decomposition is descriptive, not a causal attribution.",
            "Partition L1 shares can conceal cancellation within each sparse contribution and across the two partitions; the report includes a cross-partition cancellation fraction.",
            "The aligned, permuted, and uniform gammas are separate fitted candidates. Their coefficient or penalty quadratics are candidate-specific diagnostics, not objective values comparable across fits.",
            "The R1 target evaluation is reused development data and was not outcome-blind because the earlier label-access incident is disclosed.",
        ],
    }
    view_lookup = {
        67: ("E67_H12", "E67_W4", "E67_W5", "E67_W6"),
        75: ("E75_W6",),
    }
    for origin in (67, 75):
        baseline_view = "E67_H12" if origin == 67 else "E75_W6"
        view = results["views"][baseline_view]
        df, aligned, permuted, penalty_receipt = load_penalty_map(origin)
        if int(penalty_receipt.get("vocabulary_size", -1)) != len(df):
            raise RuntimeError(f"E{origin} penalty map vocabulary size receipt mismatch")
        row_count = int(view["n"])
        position, matrix_lineage = load_position_matrix(origin, row_count)
        token_count = len(df)
        width = 3 * token_count
        if position.shape[1] != width:
            raise RuntimeError(f"E{origin} position block width does not equal 3 * token count")
        gammas = {
            "UNIFORM": _load_gamma(
                f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
                f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json",
                fit_id=f"POS_BAG_OFFSET_V1_E{origin}_REAL", expected_arm="REAL"),
            "ALIGNED": _load_gamma(
                f"{RIDGE_ROOT}/units/E{origin}_ALIGNED.npz",
                f"{RIDGE_ROOT}/receipts/E{origin}_ALIGNED.json",
                fit_id=f"E{origin}_ALIGNED", expected_arm="ALIGNED"),
            "PERMUTED": _load_gamma(
                f"{RIDGE_ROOT}/units/E{origin}_PERMUTED.npz",
                f"{RIDGE_ROOT}/receipts/E{origin}_PERMUTED.json",
                fit_id=f"E{origin}_PERMUTED", expected_arm="PERMUTED"),
        }
        if any(len(gamma) != width for gamma in gammas.values()):
            raise RuntimeError(f"E{origin} saved gamma width mismatch")
        w_freq = np.tile(aligned, 3)
        low = w_freq < 1.0
        high = w_freq >= 1.0
        if (np.any(low & high) or np.any(~(low | high))
                or int(low.sum() + high.sum()) != width):
            raise RuntimeError(f"E{origin} exact w partition is not exhaustive")
        group_meta = {}
        for name, token_mask in (("w_below_1", aligned < 1.0), ("w_at_least_1", aligned >= 1.0)):
            group_meta[name] = {
                "token_count": int(token_mask.sum()),
                "token_share": float(np.mean(token_mask)),
                "aligned_weight_min": float(np.min(aligned[token_mask])) if token_mask.any() else None,
                "aligned_weight_median": float(np.median(aligned[token_mask])) if token_mask.any() else None,
                "aligned_weight_max": float(np.max(aligned[token_mask])) if token_mask.any() else None,
                "prefix_document_frequency_median": float(np.median(df[token_mask])) if token_mask.any() else None,
                "prefix_document_frequency_mean": float(np.mean(df[token_mask])) if token_mask.any() else None,
            }
        fitted_weights = {
            "UNIFORM": np.ones(width, dtype=np.float64),
            "ALIGNED": np.tile(aligned, 3),
            "PERMUTED": np.tile(permuted, 3),
        }
        coefficient_summary = {
            name: candidate_coefficient_summary(gammas[name], fitted_weights[name], low,
                                                tokens=token_count)
            for name in ("UNIFORM", "ALIGNED", "PERMUTED")
        }
        delta = gammas["ALIGNED"] - gammas["UNIFORM"]
        delta_summary = partitioned_delta_logits(position, delta, low)
        output["origins"][str(origin)] = {
            "reference_view": baseline_view,
            "score_rows": row_count,
            "present_rows_from_committed_results": int(view["present_n"]),
            "missing_rows_from_committed_results": int(view["missing_n"]),
            "token_count_per_block": token_count,
            "coefficient_count": width,
            "frequency_partition_tokens": group_meta,
            "gamma_artifact_lineage": {
                "UNIFORM": f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
                "ALIGNED": f"{RIDGE_ROOT}/units/E{origin}_ALIGNED.npz",
                "PERMUTED": f"{RIDGE_ROOT}/units/E{origin}_PERMUTED.npz",
            },
            "coefficient_and_candidate_penalty_energy": coefficient_summary,
            "delta_logit_aligned_minus_uniform": delta_summary,
            "position_matrix_lineage": matrix_lineage,
            "all_gamma_logical_hashes_verified": True,
            "all_map_logical_hashes_verified": True,
        }
        for view_name in view_lookup[origin]:
            v = results["views"][view_name]
            output["previously_reported_evaluation"]["views"][view_name] = {
                "rows": int(v["n"]),
                "present_rows": int(v["present_n"]),
                "missing_rows": int(v["missing_n"]),
                "competition_macro_f1_gains": v["competition_gains_macro_f1"],
                "existing_component_macro_f1": {
                    candidate: v["component_full"][candidate]["macro_f1"]
                    for candidate in ("ALIGNED", "PERMUTED", "UNIFORM")
                },
            }
    # Keep the input results exactly as observed; no metrics are recomputed here.
    out_json = ROOT / OUTPUT_JSON_REL
    out_md = ROOT / OUTPUT_MD_REL
    if out_json.exists() or out_md.exists():
        raise RuntimeError("Refusing to overwrite existing diagnostic outputs")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_report(output), encoding="utf-8")
    return output


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# POS frequency ridge: mean-one coefficient/logit diagnostic R1", "",
        "## Scope and provenance", "",
        "This is an additive zero-fit decomposition of already-frozen gamma vectors and score-position matrices. It reads no new target labels, creates no predictions, and does not compute counterfactual performance.", "",
        "The row partition is exact: frozen ALIGNED token weight `w < 1` versus `w >= 1`, repeated across the three positional blocks. No new bins or cut points were selected.", "",
        result["reference_provenance_correction"]["note"], "",
        "The primary comparison baseline uses original uniform POS probability and pA on missing-stream rows. The historical A02 MIX50 archive instead uses pP on missing-stream rows. These are distinct references; previously computed scores are unchanged.", "",
        "## Frequency partition and coefficient energy", "",
        "`w < 1` means the aligned mean-one fit applies a lower per-coordinate ridge penalty than uniform weight 1. Coefficient-energy and fitted-penalty quadratics below describe each already-fitted candidate; they are not objective values comparable across different penalties.", "",
        "| Origin | Partition | Tokens | Median prefix DF | Aligned w median |", "|---:|---|---:|---:|---:|",
    ]
    for origin, data in result["origins"].items():
        for part in ("w_below_1", "w_at_least_1"):
            p = data["frequency_partition_tokens"][part]
            lines.append(f"| {origin} | {part} | {p['token_count']:,} | {p['prefix_document_frequency_median']:.1f} | {p['aligned_weight_median']:.6f} |")
    lines += [
        "", "| Origin | Candidate | Total coefficient L2 | w<1 energy share | w<1 fitted penalty quadratic* | w>=1 fitted penalty quadratic* |", "|---:|---|---:|---:|---:|---:|",
    ]
    for origin, data in result["origins"].items():
        for candidate, summary in data["coefficient_and_candidate_penalty_energy"].items():
            low = summary["frequency_partition"]["w_below_1"]
            high = summary["frequency_partition"]["w_at_least_1"]
            total = summary["total"]
            lines.append(f"| {origin} | {candidate} | {total['l2_norm']:.6g} | {low['share_of_unweighted_coefficient_energy']:.1%} | {low['fitted_penalty_quadratic_without_one_half']:.6g} | {high['fitted_penalty_quadratic_without_one_half']:.6g} |")
    lines += [
        "", "*Candidate-specific `sum(w_j * gamma_j^2)` without the objective's factor 1/2; descriptive only.", "",
        "## Position-logit movement: ALIGNED minus UNIFORM", "",
        "For each scored row, `delta = X_position @ (gamma_ALIGNED - gamma_UNIFORM)`. The two frequency partitions sum back to total delta; the reported reconstruction error checks this numerically.", "",
        "| Origin | Contribution | Mean signed | Mean absolute | P95 absolute | L1 magnitude share |", "|---:|---|---:|---:|---:|---:|",
    ]
    for origin, data in result["origins"].items():
        delta = data["delta_logit_aligned_minus_uniform"]
        share = delta["magnitude_share"]
        for name, label in (("total", "total"), ("w_below_1", "w<1"), ("w_at_least_1", "w>=1")):
            s = delta[name]
            magnitude_share = 1.0 if name == "total" else share[name]
            lines.append(f"| {origin} | {label} | {s['mean_signed']:+.6g} | {s['mean_absolute']:.6g} | {s['p95_absolute']:.6g} | {magnitude_share:.1%} |")
    lines += [
        "", "The partition L1 shares add to one by definition. Total absolute movement can be smaller because the two signed contributions cancel across partitions; the JSON reports the cancellation fraction and reconstruction error. These are logit diagnostics, not probability or metric counterfactuals.", "",
        "## Previously observed outcome evidence", "",
        "The already-committed R1 results show that ALIGNED lost to both PERMUTED and UNIFORM on the mechanism gate at E67 H12 and E75 W6. Competition Macro F1 gains versus the recorded primary comparator were negative on E67 H12, E75 W6, and the unique pool. The per-window harm and supported-subgroup guards passed. No result is recomputed in this diagnostic.", "",
        "## Assessment and next hypothesis", "",
        "If `w<1` accounts for a majority of ALIGNED-versus-UNIFORM logit-change magnitude, that supports the narrow descriptive claim that relaxed frequent-token penalties are a major source of fitted logit movement. It does not explain why the movement is useful: the frozen outcome evidence is counterevidence because aligned weighting did not improve the saved evaluation.", "",
        "At most one prospective hypothesis is suggested: test a one-sided map that never lowers frequent-token penalties below uniform weight 1 while retaining increased penalties on the rare-token side. This changes average penalty strength, so a future protocol would need a preregistered global-strength control and the same frozen temporal comparisons. This report does not authorize any fit or continuation.", "",
        "## Limitations", "",
        "This is a post-result diagnostic chosen after observing failed R1 gates, so it is adaptive rather than confirmatory. The split `w<1`/`w>=1` is exact and inherited from the fitted map, but decomposing a linear score by feature blocks does not isolate causal effects: coefficients are jointly fitted, the partitions may cancel, and the result depends on the saved score matrix and fitted gammas. No token semantics, new target reading, Test scoring, or public leaderboard evidence is involved.", "",
        f"Freeze SHA256: `{result['freeze_sha256']}`.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-freeze", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.write_freeze:
        dest = ROOT / FREEZE_REL
        if dest.exists():
            raise RuntimeError("Refusing to overwrite diagnostic freeze")
        freeze = build_freeze()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": freeze["status"], "freeze_sha256": freeze["freeze_sha256"],
                          "sources": len(freeze["source_hashes"]), "inputs": len(freeze["input_hashes"]),
                          "target_labels_read": False}, indent=2))
    else:
        from scripts import verify_optimization_v3_cls_pos_frequency_ridge_eval_r9_adapter as adapter
        with adapter.BASE.SharedJobLock():
            result = run_integration_preflight() if args.preflight else analyze()
        if args.preflight:
            print(json.dumps({"status": result["status"], "receipt": PREFLIGHT_REL,
                              "freeze_sha256": result["freeze_sha256"],
                              "origins": sorted(result["origins"]), "target_labels_read": False}, indent=2))
        else:
            print(json.dumps({"status": result["status"], "json": OUTPUT_JSON_REL,
                              "report": OUTPUT_MD_REL, "freeze_sha256": result["freeze_sha256"],
                              "origins": sorted(result["origins"]), "target_labels_read": False,
                              "new_fits": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
