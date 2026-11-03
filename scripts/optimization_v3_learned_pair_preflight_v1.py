"""Zero-fit feasibility checks for sparse rank-16 factorization interactions.

This module contains an objective/gradient prototype and a label-free, bounded
matrix benchmark. It deliberately does not import or call an optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit


def _matrix_pair(x: Any, v: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
    if sparse.issparse(x):
        x = sparse.csr_matrix(x, dtype=np.float64)
        xv = np.asarray(x @ v, dtype=np.float64)
        x2 = x.multiply(x).tocsr()
        x2v2 = np.asarray(x2 @ (v * v), dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
        xv = x @ v
        x2 = x * x
        x2v2 = x2 @ (v * v)
    return x, xv, x2v2


def fm_pair_score(x: Any, factors: np.ndarray) -> np.ndarray:
    """Return the standard second-order FM score, excluding main effects."""
    v = np.asarray(factors, dtype=np.float64)
    if v.ndim != 2 or x.shape[1] != v.shape[0]:
        raise ValueError("factor matrix must have shape (n_features, rank)")
    _, xv, x2v2 = _matrix_pair(x, v)
    return 0.5 * np.sum(xv * xv - x2v2, axis=1)


def fm_pair_score_and_v_gradient(
    x: Any,
    factors: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair scores and weighted factor gradient; residual is label-independent here."""
    v = np.asarray(factors, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    if v.ndim != 2 or x.shape[1] != v.shape[0] or x.shape[0] != r.size:
        raise ValueError("incompatible X, factors, or residual shapes")
    _, xv, x2v2 = _matrix_pair(x, v)
    scores = 0.5 * np.sum(xv * xv - x2v2, axis=1)
    weighted_xv = r[:, None] * xv
    if sparse.issparse(x):
        x_csr = sparse.csr_matrix(x, dtype=np.float64)
        x2_csr = x_csr.multiply(x_csr).tocsr()
        first = np.asarray(x_csr.T @ weighted_xv, dtype=np.float64)
        second_weight = np.asarray(x2_csr.T @ r, dtype=np.float64).reshape(-1)
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        first = x_arr.T @ weighted_xv
        second_weight = (x_arr * x_arr).T @ r
    gradient = first - second_weight[:, None] * v
    return scores, gradient


def l2_normalize_rows(x: Any) -> Any:
    """Row-wise L2 normalization; zero rows stay zero; no learned state."""
    if sparse.issparse(x):
        matrix = sparse.csr_matrix(x, dtype=np.float64, copy=True)
        norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
        inverse = np.zeros_like(norm)
        np.divide(1.0, norm, out=inverse, where=norm > 0)
        return (sparse.diags(inverse, format="csr") @ matrix).tocsr()
    matrix = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1)
    out = np.zeros_like(matrix)
    np.divide(matrix, norm[:, None], out=out, where=norm[:, None] > 0)
    return out


def logistic_l2_objective_gradient(
    x_main: Any,
    x_interaction: Any,
    target: np.ndarray,
    main_weights: np.ndarray,
    factors: np.ndarray,
    intercept: float,
    *,
    l2_main: float,
    l2_factors: float,
    l2_intercept: float,
    offset: np.ndarray | None = None,
) -> tuple[float, tuple[np.ndarray, np.ndarray, float]]:
    """Mean logistic loss plus explicit L2; supports a fixed additive offset."""
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(main_weights, dtype=np.float64).reshape(-1)
    v = np.asarray(factors, dtype=np.float64)
    if (x_main.shape[0] != y.size or x_interaction.shape[0] != y.size
            or x_main.shape[1] != w.size or x_interaction.shape[1] != v.shape[0]
            or v.ndim != 2 or not np.isin(y, [0.0, 1.0]).all()):
        raise ValueError("incompatible design, parameter, or binary target shapes")
    if not np.isfinite([l2_main, l2_factors, l2_intercept]).all() or min(
            l2_main, l2_factors, l2_intercept) < 0:
        raise ValueError("L2 coefficients must be finite and nonnegative")
    n = y.size
    if n == 0:
        raise ValueError("empty target")
    fixed_offset = np.zeros(n, dtype=np.float64) if offset is None else np.asarray(
        offset, dtype=np.float64).reshape(-1)
    if fixed_offset.size != n or not np.isfinite(fixed_offset).all():
        raise ValueError("offset must be finite and match rows")

    pair, xv, x2v2 = _matrix_pair(x_interaction, v)
    pair_score = 0.5 * np.sum(xv * xv - x2v2, axis=1)
    linear_score = np.asarray(x_main @ w, dtype=np.float64).reshape(-1)
    logits = linear_score + float(intercept) + fixed_offset + pair_score
    loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    loss += 0.5 * l2_main * float(w @ w)
    loss += 0.5 * l2_factors * float(np.sum(v * v))
    loss += 0.5 * l2_intercept * float(intercept * intercept)
    residual = (expit(logits) - y) / n

    grad_main = np.asarray(x_main.T @ residual, dtype=np.float64).reshape(-1) + l2_main * w
    weighted_xv = residual[:, None] * xv
    if sparse.issparse(pair):
        pair_csr = sparse.csr_matrix(pair, dtype=np.float64)
        pair_sq = pair_csr.multiply(pair_csr).tocsr()
        first = np.asarray(pair_csr.T @ weighted_xv, dtype=np.float64)
        second_weight = np.asarray(pair_sq.T @ residual, dtype=np.float64).reshape(-1)
    else:
        pair_arr = np.asarray(pair, dtype=np.float64)
        first = pair_arr.T @ weighted_xv
        second_weight = (pair_arr * pair_arr).T @ residual
    grad_factors = first - second_weight[:, None] * v + l2_factors * v
    grad_intercept = float(np.sum(residual) + l2_intercept * intercept)
    return float(loss), (grad_main, grad_factors, grad_intercept)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_label_free_benchmark(repo_root: Path, *, rows_total: int = 1000) -> dict[str, Any]:
    """Verify frozen C2_B matrices and time fixed rank-16 ops on <=1000 rows."""
    if rows_total != 1000:
        raise ValueError("the authorized benchmark is fixed at exactly 1,000 total rows")
    from scipy.sparse import load_npz
    from sklearn.preprocessing import normalize
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr

    run_root = repo_root / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
    freeze = json.loads((run_root / "FREEZE.json").read_text(encoding="utf-8"))
    expected_input = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
    origins: list[dict[str, Any]] = []
    for origin, sample_rows in ((67, 500), (75, 500)):
        manifest_path = run_root / f"MATRIX_MANIFEST_E{origin}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = manifest["matrices"]["fit_base"]
        matrix_path = repo_root / item["path"]
        if manifest["origin"] != origin or manifest["input_sha256"] != expected_input:
            raise RuntimeError(f"E{origin} manifest identity does not match official frozen input")
        if manifest["freeze_signature"] != freeze["freeze_signature"]:
            raise RuntimeError(f"E{origin} matrix manifest and freeze differ")
        raw_sha = _file_sha256(matrix_path)
        if raw_sha != item["sha256"]:
            raise RuntimeError(f"E{origin} fit_base archive hash mismatch")
        full = load_npz(matrix_path).tocsr()
        logical_sha = sha256_csr(full)
        if logical_sha != item["logical_sha256"]:
            raise RuntimeError(f"E{origin} fit_base logical hash mismatch")
        expected_shape = tuple(item["shape"])
        if full.shape != expected_shape or full.shape[0] != manifest["fit_row_metadata"]["rows"]:
            raise RuntimeError(f"E{origin} fit_base dimensions mismatch")

        indices = np.rint(np.linspace(0, full.shape[0] - 1, sample_rows)).astype(np.int64)
        if len(np.unique(indices)) != sample_rows:
            raise RuntimeError("deterministic sample contains duplicate rows")
        sample = full[indices, :].tocsr()
        norm_times: list[float] = []
        pair_x = None
        for _ in range(3):
            start = time.perf_counter()
            pair_x = normalize(sample, norm="l2", axis=1, copy=True).tocsr()
            norm_times.append(time.perf_counter() - start)
        assert pair_x is not None
        rank = 16
        rng = np.random.default_rng(42)
        factors = rng.normal(0.0, 0.01, size=(full.shape[1], rank)).astype(np.float64)
        main_weights = rng.normal(0.0, 0.01, size=full.shape[1]).astype(np.float64)
        residual = np.full(sample_rows, 1.0 / sample_rows, dtype=np.float64)

        candidate_elapsed: list[float] = []
        control_elapsed: list[float] = []
        score_hash = ""
        grad_hash = ""
        for _ in range(3):
            start = time.perf_counter()
            score, grad = fm_pair_score_and_v_gradient(pair_x, factors, residual)
            main_score = np.asarray(sample @ main_weights, dtype=np.float64).reshape(-1)
            main_grad = np.asarray(sample.T @ residual, dtype=np.float64).reshape(-1)
            candidate_elapsed.append(time.perf_counter() - start)
            start = time.perf_counter()
            control_score = np.asarray(sample @ main_weights, dtype=np.float64).reshape(-1)
            control_grad = np.asarray(sample.T @ residual, dtype=np.float64).reshape(-1)
            control_elapsed.append(time.perf_counter() - start)
            score_hash = hashlib.sha256(np.ascontiguousarray(score).tobytes()).hexdigest()
            grad_hash = hashlib.sha256(np.ascontiguousarray(grad).tobytes()).hexdigest()

        full_csr_bytes = int(full.indptr.nbytes + full.indices.nbytes + full.data.nbytes)
        sample_csr_bytes = int(sample.indptr.nbytes + sample.indices.nbytes + sample.data.nbytes)
        n_features = int(full.shape[1])
        n_fit = int(full.shape[0])
        n_parameters = n_features + n_features * rank + 1
        maxcor = 10
        batch_rows = 1024
        # Modeled live buffers: raw+normalized CSR, L-BFGS history, parameters and
        # gradients, V^2, a batch XV buffer, and one temporary sparse square batch.
        batch_nnz_est = int(np.ceil(sample.nnz / sample_rows * batch_rows))
        csr_batch_est = int((batch_rows + 1) * 4 + batch_nnz_est * 12)
        memory_est = (2 * full_csr_bytes + 2 * maxcor * n_parameters * 8
                      + 2 * n_parameters * 8 + n_features * rank * 8
                      + batch_rows * rank * 8 + csr_batch_est)
        candidate_median = float(np.median(candidate_elapsed))
        control_median = float(np.median(control_elapsed))
        normalization_median = float(np.median(norm_times))
        nnz_scale = float(full.nnz / max(1, sample.nnz))
        candidate_eval_full_est = candidate_median * nnz_scale
        control_eval_full_est = control_median * nnz_scale
        normalize_full_est = normalization_median * nnz_scale
        origins.append({
            "origin": origin,
            "matrix_source_commit": freeze["source_commit"],
            "manifest_sha256": _file_sha256(manifest_path),
            "fit_base_archive_sha256": raw_sha,
            "fit_base_logical_sha256": logical_sha,
            "input_sha256": manifest["input_sha256"],
            "freeze_signature": manifest["freeze_signature"],
            "matrix_shape": list(full.shape),
            "fit_rows": n_fit,
            "full_nnz": int(full.nnz),
            "sample_rows": sample_rows,
            "sample_nnz": int(sample.nnz),
            "sample_csr_bytes": sample_csr_bytes,
            "sample_candidate_main_plus_pair_score_and_gradient_median_seconds": candidate_median,
            "sample_no_pair_linear_control_score_and_gradient_median_seconds": control_median,
            "sample_pair_branch_l2_normalization_median_seconds": normalization_median,
            "nnz_linear_scale_factor": nnz_scale,
            "estimated_full_candidate_function_gradient_seconds": candidate_eval_full_est,
            "estimated_full_control_function_gradient_seconds": control_eval_full_est,
            "estimated_full_pair_normalization_seconds": normalize_full_est,
            "estimated_candidate_seconds_at_200_function_evaluations": candidate_eval_full_est * 200 + normalize_full_est,
            "estimated_control_seconds_at_200_function_evaluations": control_eval_full_est * 200,
            "estimated_two_fit_seconds_per_origin_at_200_function_evaluations":
                (candidate_eval_full_est + control_eval_full_est) * 200 + normalize_full_est,
            "rank": rank,
            "factor_init": "numpy.default_rng(42).normal(0, 0.01, (n_features, 16))",
            "score_sha256": score_hash,
            "factor_gradient_sha256": grad_hash,
            "main_score_sha256": hashlib.sha256(np.ascontiguousarray(main_score).tobytes()).hexdigest(),
            "main_gradient_sha256": hashlib.sha256(np.ascontiguousarray(main_grad).tobytes()).hexdigest(),
            "control_score_sha256": hashlib.sha256(np.ascontiguousarray(control_score).tobytes()).hexdigest(),
            "control_gradient_sha256": hashlib.sha256(np.ascontiguousarray(control_grad).tobytes()).hexdigest(),
            "estimated_sparse_csr_bytes": full_csr_bytes,
            "modeled_working_set_bytes_excluding_python_scipy_overhead": int(memory_est),
            "resource_limit_note": "estimate from sampled nonzero scaling and explicit buffers; not full-fit timing or OS peak RSS",
        })
    return {
        "status": "PASS_LABEL_FREE_RESOURCE_PREFLIGHT",
        "fit_or_optimizer_calls": 0,
        "target_or_label_files_opened": False,
        "total_distinct_matrix_rows_sampled": sum(x["sample_rows"] for x in origins),
        "repeated_timing_passes_reuse_same_rows": True,
        "matrix_sources": "saved POSITIONAL_TEXT_PILOT_V1 C2_B fit_base matrices; only deterministic row subsets processed",
        "rank": 16,
        "origins": origins,
        "environment": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_label_free_benchmark(root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
