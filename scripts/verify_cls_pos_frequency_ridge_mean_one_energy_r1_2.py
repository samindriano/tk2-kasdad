"""Independent zero-fit reconstruction of the frozen R1.2 energy diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_VERIFY_FREEZE.json"
DIAGNOSTIC_FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_FREEZE.json"
DIAGNOSTIC_JSON_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2.json"
DIAGNOSTIC_MD_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2.md"
PREFLIGHT_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_PREFLIGHT.json"
RESULTS_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESULTS_R1.json"
PILOT_ROOT = "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
POS_BAG_ROOT = "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
RIDGE_ROOT = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1"
TOL = 2e-12
MIN_AVAILABLE_BYTES = int(1.5 * 1024**3)
SHARED_JOB_LOCK_PATH = str(Path(__import__("tempfile").gettempdir()) / "kasdad_single_heavy_job.lock")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or "Git command failed")
    return p.stdout.strip()


def committed(rel: str) -> dict[str, str]:
    blob = git("rev-parse", f"HEAD:{rel}")
    if blob != git("hash-object", rel):
        raise RuntimeError(f"Expected committed verifier file is modified: {rel}")
    return {"git_blob": blob, "commit": git("log", "-1", "--format=%H", "--", rel)}


def verify_signature(doc: dict[str, Any], label: str) -> str:
    body = dict(doc)
    claimed = body.pop("freeze_sha256", None)
    actual = hashlib.sha256(canonical_json(body)).hexdigest()
    if claimed != actual:
        raise RuntimeError(f"{label} canonical freeze signature mismatch")
    return str(claimed)


def selected_verification_inputs() -> list[str]:
    return sorted({DIAGNOSTIC_FREEZE_REL, DIAGNOSTIC_JSON_REL, DIAGNOSTIC_MD_REL,
                   PREFLIGHT_REL, RESULTS_REL})


def build_freeze() -> dict[str, Any]:
    parent = read_json(DIAGNOSTIC_FREEZE_REL)
    parent_sha = verify_signature(parent, "R1.2 diagnostic")
    paths = selected_verification_inputs()
    inputs = {rel: sha_file(ROOT / rel) for rel in paths}
    if inputs[DIAGNOSTIC_FREEZE_REL] != sha_file(ROOT / DIAGNOSTIC_FREEZE_REL):
        raise RuntimeError("Diagnostic freeze byte binding failed")
    diag = read_json(DIAGNOSTIC_JSON_REL)
    pf = read_json(PREFLIGHT_REL)
    if (diag.get("freeze_sha256") != parent_sha
            or diag.get("target_labels_read") is not False
            or diag.get("new_fits") != 0
            or pf.get("status") != "PASS_R1_2_INTEGRATION_PREFLIGHT_NO_MEASUREMENT"
            or pf.get("freeze_sha256") != parent_sha):
        raise RuntimeError("R1.2 diagnostic/preflight boundary mismatch")
    script_rel = "scripts/verify_cls_pos_frequency_ridge_mean_one_energy_r1_2.py"
    sources = {script_rel: sha_file(ROOT / script_rel)}
    source_commits = {script_rel: committed(script_rel)}
    body: dict[str, Any] = {
        "schema_version": 1,
        "verification_id": "CLS_POS_FREQUENCY_RIDGE_MEAN_ONE_ENERGY_R1_2_INDEPENDENT_RECONSTRUCTION",
        "status": "FROZEN_ADDITIVE_ZERO_FIT_VERIFICATION",
        "parent_diagnostic_freeze": DIAGNOSTIC_FREEZE_REL,
        "parent_diagnostic_freeze_signature": parent_sha,
        "parent_diagnostic_freeze_file_sha256": inputs[DIAGNOSTIC_FREEZE_REL],
        "source_hashes": sources,
        "source_commits": source_commits,
        "input_hashes": inputs,
        "parent_input_hashes": parent.get("input_hashes", {}),
        "measurement_scope": "independently recompute only saved gamma/map/matrix coefficient and linear logit summaries; verify reported outcomes are copied from committed R1 results",
        "resource_policy": {
            "shared_job_lock_path": SHARED_JOB_LOCK_PATH,
            "minimum_available_memory_bytes": MIN_AVAILABLE_BYTES,
            "minimum_available_memory_gib": 1.5,
            "check_timing": "fresh check after shared lock acquisition and recheck after verification",
        },
        "target_labels_read": False,
        "new_fits": 0,
        "new_predictions_or_counterfactual_performance": 0,
        "test_or_samplesubmission_accessed": False,
    }
    body["freeze_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def verify_freeze() -> dict[str, Any]:
    committed(FREEZE_REL)
    freeze = read_json(FREEZE_REL)
    verify_signature(freeze, "R1.2 verification")
    for rel, expected in freeze.get("source_hashes", {}).items():
        if sha_file(ROOT / rel) != expected:
            raise RuntimeError(f"Verifier source hash changed: {rel}")
        if committed(rel) != freeze["source_commits"][rel]:
            raise RuntimeError(f"Verifier source Git binding changed: {rel}")
    for rel, expected in freeze.get("input_hashes", {}).items():
        if sha_file(ROOT / rel) != expected:
            raise RuntimeError(f"Verification input hash changed: {rel}")
    parent = read_json(DIAGNOSTIC_FREEZE_REL)
    if verify_signature(parent, "R1.2 diagnostic") != freeze["parent_diagnostic_freeze_signature"]:
        raise RuntimeError("Parent diagnostic freeze signature changed")
    for rel, expected in freeze.get("parent_input_hashes", {}).items():
        if sha_file(ROOT / rel) != expected:
            raise RuntimeError(f"Parent diagnostic source/input changed: {rel}")
    return freeze


def independent_array_digest(a: np.ndarray) -> str:
    x = np.asarray(a)
    hmeta = hashlib.sha256(canonical_json({"shape": x.shape, "dtype": str(x.dtype)})).hexdigest()
    h = hashlib.sha256(hmeta.encode("ascii"))
    if x.dtype.kind in "OUS":
        h.update(json.dumps(x.astype(str).tolist(), ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def assert_close(actual: Any, expected: Any, label: str, diffs: list[float]) -> None:
    if isinstance(expected, bool):
        if actual is not expected:
            raise RuntimeError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        delta = abs(float(actual) - float(expected))
        diffs.append(delta)
        if not np.isfinite(float(actual)) or delta > TOL:
            raise RuntimeError(f"{label}: {actual!r} != {expected!r} within {TOL}")
        return
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def describe(x: np.ndarray) -> dict[str, Any]:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    av = np.abs(v)
    return {"n_rows": int(v.size), "mean_signed": float(np.mean(v)),
            "mean_absolute": float(np.mean(av)), "p95_absolute": float(np.quantile(av, 0.95)),
            "max_absolute": float(np.max(av))}


def load_gamma(origin: int, candidate: str) -> np.ndarray:
    if candidate == "UNIFORM":
        art = f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz"
        rec = f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json"
        fit_id = f"POS_BAG_OFFSET_V1_E{origin}_REAL"
    else:
        art = f"{RIDGE_ROOT}/units/E{origin}_{candidate}.npz"
        rec = f"{RIDGE_ROOT}/receipts/E{origin}_{candidate}.json"
        fit_id = f"E{origin}_{candidate}"
    receipt = read_json(rec)
    if receipt.get("fit_id") != fit_id or receipt.get("artifact_sha256") != sha_file(ROOT / art):
        raise RuntimeError(f"E{origin} {candidate} artifact receipt mismatch")
    with np.load(ROOT / art, allow_pickle=False) as z:
        if set(z.files) != {"gamma"}:
            raise RuntimeError(f"E{origin} {candidate} gamma archive schema mismatch")
        gamma = np.asarray(z["gamma"], dtype=np.float64).reshape(-1)
    if independent_array_digest(gamma) != receipt.get("gamma_sha256"):
        raise RuntimeError(f"E{origin} {candidate} gamma logical receipt mismatch")
    return gamma


def load_inputs(origin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix, dict[str, Any]]:
    map_art = f"{RIDGE_ROOT}/maps/penalty_map_E{origin}.npz"
    map_rec = read_json(f"{RIDGE_ROOT}/maps/penalty_map_E{origin}.json")
    if map_rec.get("artifact_sha256") != sha_file(ROOT / map_art):
        raise RuntimeError(f"E{origin} penalty map receipt mismatch")
    with np.load(ROOT / map_art, allow_pickle=False) as z:
        df = np.asarray(z["document_frequency"], dtype=np.int64).reshape(-1)
        aligned = np.asarray(z["aligned_token_weights"], dtype=np.float64).reshape(-1)
        permuted = np.asarray(z["permuted_token_weights"], dtype=np.float64).reshape(-1)
        if set(z.files) != {"document_frequency", "aligned_token_weights", "permuted_token_weights", "permutation"}:
            raise RuntimeError(f"E{origin} penalty map schema mismatch")
    if (independent_array_digest(df) != map_rec.get("df_sha256")
            or independent_array_digest(aligned) != map_rec.get("aligned_sha256")
            or independent_array_digest(permuted) != map_rec.get("permuted_sha256")):
        raise RuntimeError(f"E{origin} penalty map logical hash mismatch")
    manifest = read_json(f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json")
    base_item, score_item = manifest["matrices"]["score_base"], manifest["matrices"]["score_position"]
    base = sparse.load_npz(ROOT / base_item["path"]).tocsr()
    augmented = sparse.load_npz(ROOT / score_item["path"]).tocsr()
    if (sha_file(ROOT / base_item["path"]) != base_item["sha256"]
            or sha_file(ROOT / score_item["path"]) != score_item["sha256"]):
        raise RuntimeError(f"E{origin} score matrix byte hash mismatch")
    prefix = (augmented[:, :base.shape[1]] - base).tocsr()
    prefix.eliminate_zeros()
    if prefix.nnz:
        raise RuntimeError(f"E{origin} score_position base prefix does not reconstruct")
    return df, aligned, permuted, augmented[:, base.shape[1]:].tocsr(), manifest


def resource_snapshot() -> dict[str, int]:
    return {
        "available_memory_bytes": int(psutil.virtual_memory().available),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
        "captured_unix_time": int(time.time()),
    }


def require_memory_floor(snapshot: dict[str, int], label: str) -> None:
    if snapshot["available_memory_bytes"] < MIN_AVAILABLE_BYTES:
        available_gib = snapshot["available_memory_bytes"] / (1024**3)
        raise RuntimeError(f"{label}: available RAM {available_gib:.3f} GiB is below frozen 1.5 GiB floor")


def verify(initial_resource: dict[str, int]) -> dict[str, Any]:
    freeze = verify_freeze()
    if freeze.get("resource_policy", {}).get("minimum_available_memory_bytes") != MIN_AVAILABLE_BYTES:
        raise RuntimeError("Frozen resource floor does not match the required 1.5 GiB zero-fit floor")
    require_memory_floor(initial_resource, "Before numerical verification")
    diagnostic = read_json(DIAGNOSTIC_JSON_REL)
    preflight = read_json(PREFLIGHT_REL)
    results = read_json(RESULTS_REL)
    if (diagnostic.get("freeze_sha256") != freeze["parent_diagnostic_freeze_signature"]
            or diagnostic.get("target_labels_read") is not False
            or diagnostic.get("new_fits") != 0
            or preflight.get("target_labels_read") is not False):
        raise RuntimeError("Diagnostic output boundary is not target-free")
    if diagnostic.get("reference_provenance_correction", {}).get("actual_primary_comparator", {}).get("missing") != "pA exactly, primary threshold 0.50":
        raise RuntimeError("Primary comparison provenance is misstated")
    if diagnostic.get("reference_provenance_correction", {}).get("archived_historical_A02_MIX50", {}).get("missing") != "pP exactly":
        raise RuntimeError("Historical A02 MIX50 fallback provenance is misstated")
    diffs: list[float] = []
    comparisons = 0
    origin_checks = {}
    for origin, view_name in ((67, "E67_H12"), (75, "E75_W6")):
        saved = diagnostic["origins"][str(origin)]
        df, aligned, permuted, x, _ = load_inputs(origin)
        gammas = {name: load_gamma(origin, name) for name in ("UNIFORM", "ALIGNED", "PERMUTED")}
        low = np.tile(aligned < 1.0, 3)
        weights_by_candidate = {"UNIFORM": np.ones(3 * len(df)),
                                "ALIGNED": np.tile(aligned, 3),
                                "PERMUTED": np.tile(permuted, 3)}
        if x.shape != (int(results["views"][view_name]["n"]), 3 * len(df)):
            raise RuntimeError(f"E{origin} diagnostic matrix/gamma/map shape mismatch")
        if not np.array_equal(np.sort(aligned), np.sort(permuted)):
            raise RuntimeError(f"E{origin} ALIGNED/PERMUTED maps differ in multiset")
        checks = 0
        energy_checks = 0
        delta_checks = 0
        token_checks = 0
        for candidate, gamma in gammas.items():
            record = saved["coefficient_and_candidate_penalty_energy"][candidate]
            fitted_weights = weights_by_candidate[candidate]
            energy = float(gamma @ gamma)
            assert_close(record["total"]["l2_norm"], np.sqrt(energy), f"E{origin}/{candidate}/total norm", diffs); checks += 1; energy_checks += 1
            assert_close(record["total"]["squared_l2_energy"], energy, f"E{origin}/{candidate}/total energy", diffs); checks += 1; energy_checks += 1
            assert_close(record["total"]["fitted_penalty_quadratic_without_one_half"], float(fitted_weights @ (gamma * gamma)), f"E{origin}/{candidate}/total penalty", diffs); checks += 1; energy_checks += 1
            for part, mask in (("w_below_1", low), ("w_at_least_1", ~low)):
                gp = gamma[mask]
                e = float(gp @ gp)
                rec = record["frequency_partition"][part]
                assert_close(rec["coefficient_count"], int(mask.sum()), f"E{origin}/{candidate}/{part}/count", diffs); checks += 1; energy_checks += 1
                assert_close(rec["l2_norm"], np.sqrt(e), f"E{origin}/{candidate}/{part}/norm", diffs); checks += 1; energy_checks += 1
                assert_close(rec["squared_l2_energy"], e, f"E{origin}/{candidate}/{part}/energy", diffs); checks += 1; energy_checks += 1
                assert_close(rec["share_of_unweighted_coefficient_energy"], e / energy if energy else 0.0, f"E{origin}/{candidate}/{part}/energy share", diffs); checks += 1; energy_checks += 1
                assert_close(rec["fitted_penalty_quadratic_without_one_half"], float(fitted_weights[mask] @ (gp * gp)), f"E{origin}/{candidate}/{part}/penalty", diffs); checks += 1; energy_checks += 1
        delta_gamma = gammas["ALIGNED"] - gammas["UNIFORM"]
        total = np.asarray(x @ delta_gamma).reshape(-1)
        low_delta = np.asarray(x[:, low] @ delta_gamma[low]).reshape(-1)
        high_delta = np.asarray(x[:, ~low] @ delta_gamma[~low]).reshape(-1)
        rec = saved["delta_logit_aligned_minus_uniform"]
        for name, values in (("total", total), ("w_below_1", low_delta), ("w_at_least_1", high_delta)):
            stats = describe(values)
            for stat, expected in stats.items():
                assert_close(rec[name][stat], expected, f"E{origin}/delta/{name}/{stat}", diffs); checks += 1; delta_checks += 1
        l1_low, l1_high = float(np.abs(low_delta).sum()), float(np.abs(high_delta).sum())
        l1_total = float(np.abs(total).sum())
        denom = l1_low + l1_high
        for part, value in (("w_below_1", l1_low / denom if denom else 0.0),
                            ("w_at_least_1", l1_high / denom if denom else 0.0)):
            assert_close(rec["magnitude_share"][part], value, f"E{origin}/delta/share/{part}", diffs); checks += 1; delta_checks += 1
        cancel = max(0.0, 1.0 - l1_total / denom) if denom else 0.0
        assert_close(rec["cross_partition_cancellation_fraction"], cancel, f"E{origin}/delta/cancellation", diffs); checks += 1; delta_checks += 1
        reconstruction = low_delta + high_delta
        max_error = float(np.max(np.abs(reconstruction - total)))
        mean_error = float(np.mean(np.abs(reconstruction - total)))
        allclose = bool(np.allclose(total, reconstruction, atol=1e-10, rtol=1e-10))
        assert_close(rec["reconstruction"]["max_abs_error"], max_error, f"E{origin}/delta/reconstruct max", diffs); checks += 1; delta_checks += 1
        assert_close(rec["reconstruction"]["mean_abs_error"], mean_error, f"E{origin}/delta/reconstruct mean", diffs); checks += 1; delta_checks += 1
        assert_close(rec["reconstruction"]["allclose_atol_1e-10_rtol_1e-10"], allclose, f"E{origin}/delta/reconstruct allclose", diffs); checks += 1; delta_checks += 1
        if not allclose:
            raise RuntimeError(f"E{origin} partitioned delta logits do not reconstruct")
        for part, token_mask in (("w_below_1", aligned < 1.0), ("w_at_least_1", aligned >= 1.0)):
            tok = saved["frequency_partition_tokens"][part]
            assert_close(tok["token_count"], int(token_mask.sum()), f"E{origin}/{part}/token count", diffs); checks += 1; token_checks += 1
            assert_close(tok["prefix_document_frequency_median"], float(np.median(df[token_mask])), f"E{origin}/{part}/median DF", diffs); checks += 1; token_checks += 1
        origin_checks[str(origin)] = {
            "coefficient_energy_fields_checked": energy_checks,
            "delta_logit_fields_checked": delta_checks,
            "token_partition_fields_checked": token_checks,
            "max_absolute_numeric_difference": max(diffs) if diffs else 0.0,
            "delta_partition_reconstructs": allclose,
            "delta_reconstruction_max_abs_error": max_error,
            "w_below_1_delta_magnitude_share": l1_low / denom if denom else 0.0,
            "w_at_least_1_delta_magnitude_share": l1_high / denom if denom else 0.0,
            "cross_partition_cancellation_fraction": cancel,
        }
        comparisons += checks
    # Verify every referenced result metric is copied verbatim from committed R1 results.
    for view_name, saved_view in diagnostic["previously_reported_evaluation"]["views"].items():
        v = results["views"][view_name]
        if (saved_view["rows"] != v["n"] or saved_view["present_rows"] != v["present_n"]
                or saved_view["missing_rows"] != v["missing_n"]
                or saved_view["competition_macro_f1_gains"] != v["competition_gains_macro_f1"]):
            raise RuntimeError(f"Previously reported results differ from current R1 JSON: {view_name}")
        for candidate in ("ALIGNED", "PERMUTED", "UNIFORM"):
            if saved_view["existing_component_macro_f1"][candidate] != v["component_full"][candidate]["macro_f1"]:
                raise RuntimeError(f"Copied component metric mismatch: {view_name}/{candidate}")
        comparisons += 1
    final_resource = resource_snapshot()
    require_memory_floor(final_resource, "After numerical verification")
    return {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_ZERO_FIT_COEFFICIENT_AND_LOGIT_RECONSTRUCTION",
        "verification_freeze_sha256": freeze["freeze_sha256"],
        "parent_diagnostic_freeze_signature": freeze["parent_diagnostic_freeze_signature"],
        "origins": origin_checks,
        "metric_reference_views_checked": len(diagnostic["previously_reported_evaluation"]["views"]),
        "numeric_and_lineage_checks": comparisons,
        "maximum_numeric_difference": max(diffs) if diffs else 0.0,
        "resource": {
            "shared_job_lock_path": SHARED_JOB_LOCK_PATH,
            "shared_job_lock_held": True,
            "minimum_available_memory_bytes": MIN_AVAILABLE_BYTES,
            "before": initial_resource,
            "after": final_resource,
            "floor_passed_before_and_after": True,
        },
        "target_labels_read": False,
        "new_fits": 0,
        "new_predictions_or_counterfactual_performance": 0,
        "test_or_samplesubmission_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write_freeze:
        dest = ROOT / FREEZE_REL
        if dest.exists():
            raise RuntimeError("Refusing to overwrite verification freeze")
        freeze = build_freeze()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": freeze["status"], "freeze_sha256": freeze["freeze_sha256"],
                          "inputs": len(freeze["input_hashes"])}, indent=2))
    else:
        from scripts import verify_optimization_v3_cls_pos_frequency_ridge_eval_r9_adapter as adapter
        with adapter.BASE.SharedJobLock():
            initial_resource = resource_snapshot()
            require_memory_floor(initial_resource, "Before numerical verification")
            result = verify(initial_resource)
        dest = ROOT / "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_VERIFY.json"
        if dest.exists():
            raise RuntimeError("Refusing to overwrite independent verification receipt")
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "receipt": dest.relative_to(ROOT).as_posix(),
                          "origins": sorted(result["origins"]), "checks": result["numeric_and_lineage_checks"],
                          "max_abs_diff": result["maximum_numeric_difference"],
                          "target_labels_read": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
