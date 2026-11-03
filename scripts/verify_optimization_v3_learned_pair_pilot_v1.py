"""Independent, non-fitting verification for learned_pair_pilot_v1 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import optimization_v3_positional_bag_offset_v1 as offset_inputs

CARD_ID = "LEARNED_PAIR_PILOT_V1"
FIT_IDS = (
    "LEARNED_PAIR_PILOT_V1_E67_FM", "LEARNED_PAIR_PILOT_V1_E67_NOPAIR",
    "LEARNED_PAIR_PILOT_V1_E75_FM", "LEARNED_PAIR_PILOT_V1_E75_NOPAIR",
)
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
A02_REFERENCE_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
SOURCE_FILES = (
    "scripts/optimization_v3_learned_pair_pilot_v1.py",
    "scripts/verify_optimization_v3_learned_pair_pilot_v1.py",
    "src/models/learned_pair_pilot_v1.py",
    "tests/test_learned_pair_pilot_v1.py",
    "reports/optimization_v3/learned_pair_pilot_v1/PROTOCOL.md",
    "reports/optimization_v3/learned_pair_preflight_v1/REPORT.md",
    "reports/optimization_v3/learned_pair_preflight_v1/SOL_PILOT_AUTHORIZATION.md",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
)


class VerificationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _independent_fm_score(x: sparse.csr_matrix, w: np.ndarray, b: float,
                          v: np.ndarray | None, chunk_rows: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Recompute FM output independently, without importing the training module."""
    n = x.shape[0]
    logits = np.empty(n, dtype=np.float64)
    pair = np.zeros(n, dtype=np.float64)
    for begin in range(0, n, chunk_rows):
        end = min(begin + chunk_rows, n)
        block = x[begin:end].tocsr().astype(np.float64, copy=False)
        z = np.sqrt(np.asarray(block.multiply(block).sum(axis=1)).reshape(-1))
        inv = np.zeros_like(z)
        inv[z > 0] = 1.0 / z[z > 0]
        q = sparse.diags(inv, format="csr") @ block
        score = np.zeros(end - begin, dtype=np.float64)
        if v is not None:
            projection = np.asarray(q @ v, dtype=np.float64)
            correction = np.asarray(q.multiply(q) @ (v * v), dtype=np.float64)
            score = (np.square(projection) - correction).sum(axis=1) * 0.5
            pair[begin:end] = score
        logits[begin:end] = np.asarray(block @ w).reshape(-1) + b + score
    return logits, pair


def _independent_objective_gradient(theta: np.ndarray, x: sparse.csr_matrix, y: np.ndarray,
                                    rank: int, with_factors: bool,
                                    chunk_rows: int = 1024) -> tuple[float, np.ndarray, np.ndarray]:
    """Recompute SUM BCE and all analytic parameter blocks without calling the optimizer."""
    d = x.shape[1]
    p = np.asarray(theta, dtype=np.float64)
    expected = d + 1 + (d * rank if with_factors else 0)
    if p.shape != (expected,) or not np.isfinite(p).all():
        raise VerificationError("training parameter vector has invalid blocks")
    target = np.asarray(y, dtype=np.int8)
    if target.shape != (x.shape[0],) or not np.isin(target, (0, 1)).all():
        raise VerificationError("fit-prefix labels are not binary/aligned")
    w = p[:d]
    b = float(p[d])
    v = p[d + 1:].reshape(d, rank) if with_factors else None
    signed = (2 * target.astype(np.int16) - 1).astype(np.float64)
    gw = np.zeros(d, dtype=np.float64)
    gb = 0.0
    gv = np.zeros_like(v) if v is not None else None
    loss = 0.0
    logits_all = np.empty(len(target), dtype=np.float64)
    for begin in range(0, len(target), chunk_rows):
        end = min(begin + chunk_rows, len(target))
        block = x[begin:end].tocsr().astype(np.float64, copy=False)
        residual_rows = np.sqrt(np.asarray(block.multiply(block).sum(axis=1)).reshape(-1))
        inv = np.zeros_like(residual_rows)
        inv[residual_rows > 0] = 1.0 / residual_rows[residual_rows > 0]
        q = sparse.diags(inv, format="csr") @ block
        xv = np.asarray(q @ v) if v is not None else None
        q2 = q.multiply(q) if v is not None else None
        pair = (0.5 * np.sum(xv * xv - np.asarray(q2 @ (v * v)), axis=1)
                if v is not None and xv is not None and q2 is not None else 0.0)
        z = np.asarray(block @ w).reshape(-1) + b + pair
        logits_all[begin:end] = z
        t = signed[begin:end]
        loss += float(np.logaddexp(0.0, -t * z).sum(dtype=np.float64))
        residual = -t * expit(-t * z)
        gw += np.asarray(block.T @ residual).reshape(-1)
        gb += float(residual.sum())
        if gv is not None and v is not None and xv is not None and q2 is not None:
            block_grad = np.asarray(q.T @ (residual[:, None] * xv))
            correction = np.asarray(q2.T @ residual).reshape(-1)
            gv += block_grad - correction[:, None] * v
    penalty = 0.5 * (float(w @ w) + b * b + (float(np.sum(v * v)) if v is not None else 0.0))
    gw += w
    gb += b
    grad_blocks = [gw, np.asarray([gb])]
    if gv is not None and v is not None:
        grad_blocks.append((gv + v).reshape(-1))
    return loss + penalty, np.concatenate(grad_blocks), logits_all


def _verify_bundle(matrix_path: Path, manifest_path: Path, prediction_path: Path,
                   receipt_path: Path, artifact_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key].copy() for key in archive.files}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if _sha256_file(matrix_path) != manifest["matrix_archive_sha256"]:
        raise VerificationError("matrix archive file hash mismatch")
    x = sparse.load_npz(matrix_path).tocsr().astype(np.float64)
    if list(x.shape) != manifest["matrix_shape"] or offset_inputs.sha256_csr(x) != manifest["matrix_logical_sha256"]:
        raise VerificationError("matrix shape/logical hash mismatch")
    if _sha256_file(artifact_path) != receipt["artifact_sha256"]:
        raise VerificationError("parameter artifact hash mismatch")
    if _sha256_file(prediction_path) != receipt["prediction_sha256"]:
        raise VerificationError("prediction artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        theta = np.asarray(archive["parameters"], dtype=np.float64)
    with_factors = bool(manifest["with_factors"])
    d = x.shape[1]
    rank = int(manifest["rank"])
    expected = d + 1 + (d * rank if with_factors else 0)
    if theta.shape != (expected,) or not np.isfinite(theta).all():
        raise VerificationError("parameter block partition or finiteness mismatch")
    w = theta[:d]
    b = float(theta[d])
    v = theta[d + 1:].reshape(d, rank) if with_factors else None
    logits, pair = _independent_fm_score(x, w, b, v, int(manifest["chunk_rows"]))
    probability = expit(logits)
    if not np.allclose(logits, prediction["raw_score"], rtol=1e-12, atol=1e-12):
        raise VerificationError("independent raw-score calculation differs from saved output")
    if not np.allclose(probability, prediction["probability_direct"], rtol=1e-12, atol=1e-12):
        raise VerificationError("direct probabilities differ from independent raw-score calculation")
    present = np.asarray(prediction["present_mask"], dtype=bool)
    fallback = np.asarray(prediction["fallback_probability"], dtype=np.float64)
    routed = np.asarray(prediction["probability_routed"], dtype=np.float64)
    expected_route = probability.copy()
    expected_route[~present] = fallback[~present]
    if not np.array_equal(routed[~present], fallback[~present]) or not np.array_equal(routed, expected_route):
        raise VerificationError("missing-stream fallback is not exact")
    if not np.isfinite(routed).all() or (routed < 0).any() or (routed > 1).any():
        raise VerificationError("routed probabilities are invalid")
    if list(np.asarray(prediction["class_order"]).astype(int)) != [0, 1]:
        raise VerificationError("saved class order is not [0, 1]")
    if not np.array_equal(prediction["record_ids"].astype(str), np.asarray(manifest["record_ids"]).astype(str)):
        raise VerificationError("record ID order changed")
    if not np.array_equal(prediction["cycles"].astype(int), np.asarray(manifest["cycles"]).astype(int)):
        raise VerificationError("cycle order changed")
    fit_path = matrix_path.parent / manifest["fit_matrix_name"]
    fit_y_path = matrix_path.parent / manifest["fit_y_name"]
    if (_sha256_file(fit_path) != manifest["fit_matrix_archive_sha256"]
            or _sha256_file(fit_y_path) != manifest["fit_y_sha256"]):
        raise VerificationError("synthetic fit matrix/target artifact hash mismatch")
    fit_x = sparse.load_npz(fit_path).tocsr().astype(np.float64)
    if offset_inputs.sha256_csr(fit_x) != manifest["fit_matrix_logical_sha256"]:
        raise VerificationError("synthetic fit matrix logical hash mismatch")
    fit_y = np.load(fit_y_path, allow_pickle=False).astype(np.int8)
    train_loss, train_grad, fit_logits = _independent_objective_gradient(
        theta, fit_x, fit_y, rank, with_factors, int(manifest["chunk_rows"]))
    with np.load(artifact_path, allow_pickle=False) as archive:
        persisted_fit_logits = np.asarray(archive["fit_logits"], dtype=np.float64)
        persisted_fit_ids = np.asarray(archive["fit_row_ids"]).astype(str)
        persisted_fit_cycles = np.asarray(archive["fit_cycles"], dtype=np.int64)
    if (not np.array_equal(persisted_fit_ids, np.asarray(manifest["fit_record_ids"]).astype(str))
            or not np.array_equal(persisted_fit_cycles, np.asarray(manifest["fit_cycles"], dtype=np.int64))):
        raise VerificationError("synthetic fit identity order mismatch")
    opt = receipt["optimizer"]
    if (not np.allclose(fit_logits, persisted_fit_logits, rtol=1e-12, atol=1e-12)
            or abs(train_loss - opt["verified_final_loss_sum"]) > 1e-8
            or abs(float(np.max(np.abs(train_grad))) - opt["verified_final_gradient_inf_norm_sum"]) > 1e-8):
        raise VerificationError("independent fit objective/gradient does not match persisted receipt")
    return {"status": "PASS", "rows": int(x.shape[0]), "features": int(x.shape[1]),
            "raw_score_sha256": offset_inputs.sha256_array(logits),
            "routed_probability_sha256": offset_inputs.sha256_array(routed),
            "pair_score_sha256": offset_inputs.sha256_array(pair),
            "fit_gradient_inf_norm_sum": float(np.max(np.abs(train_grad))),
            "missing_rows": int((~present).sum()), "class_order": [0, 1]}


def verify_synthetic_bundle(root: Path) -> dict[str, Any]:
    root = Path(root)
    matrix_path = root / "matrix.npz"
    manifest_path = root / "manifest.json"
    prediction_path = root / "prediction.npz"
    receipt_path = root / "receipt.json"
    artifact_path = root / "parameters.npz"
    result = _verify_bundle(matrix_path, manifest_path, prediction_path, receipt_path, artifact_path)
    events_path = root / "ledger.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [event.get("event") for event in events] != ["FIT_STARTED", "FIT_SAVED"]:
        raise VerificationError("temporary completed ledger does not contain exactly one terminal synthetic fit")
    if result["rows"] != len(json.loads(manifest_path.read_text(encoding="utf-8"))["record_ids"]):
        raise VerificationError("synthetic rows/IDs mismatch")
    return result


def verify_official(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    freeze_path = run_root / "FREEZE.json"
    ready_path = run_root / "PREDICTIONS_READY.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_body = dict(frozen)
    signature = freeze_body.pop("freeze_signature", None)
    if signature != _canonical_sha256(freeze_body):
        raise VerificationError("official freeze signature mismatch")
    synthetic_qa_path = run_root / "SYNTHETIC_QA.json"
    if _sha256_file(synthetic_qa_path) != frozen.get("synthetic_qa_sha256"):
        raise VerificationError("pre-fit synthetic QA receipt hash mismatch")
    synthetic_qa = json.loads(synthetic_qa_path.read_text(encoding="utf-8"))
    if (synthetic_qa.get("status") != "PASS_SYNTHETIC_END_TO_END"
            or synthetic_qa.get("official_fit_calls") != 0
            or synthetic_qa.get("live_budget_ledger_sha256_before")
            != synthetic_qa.get("live_budget_ledger_sha256_after")):
        raise VerificationError("pre-fit synthetic QA did not demonstrate a clean no-official-fit pass")
    if _sha256_file(TRAIN) != frozen["input_sha256"]:
        raise VerificationError("official Train input hash mismatch")
    source_hashes = {rel: _sha256_file(ROOT / rel) for rel in SOURCE_FILES}
    if source_hashes != frozen["source_hashes"]:
        raise VerificationError("official source hashes changed")
    source_state = offset_inputs.verify_saved_inputs(include_current_code=True)
    if source_state["input_sha256"] != frozen["input_sha256"]:
        raise VerificationError("historical matrix source identity mismatch")
    frame = offset_inputs._read_identity_frame()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (ready.get("freeze_signature") != signature or ready.get("query_labels_opened") is not False
            or set(ready.get("predictions", {})) != {"E67", "E75"}
            or ready.get("fit_ids") != list(FIT_IDS)):
        raise VerificationError("official prediction readiness record invalid")
    ledger = run_root.parent / "budget.jsonl"
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    for fit_id in FIT_IDS:
        start = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
        if len(start) != 1 or len(saved) != 1:
            raise VerificationError(f"official ledger fit lifecycle invalid for {fit_id}")
        if start[0].get("freeze_signature") != signature:
            raise VerificationError(f"official FIT_STARTED freeze mismatch for {fit_id}")
        artifact = run_root / "units" / f"{fit_id}.npz"
        receipt_path = run_root / "receipts" / f"{fit_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("freeze_signature") != signature or not receipt.get("optimizer_converged")
                or _sha256_file(artifact) != receipt.get("artifact_sha256")
                or _sha256_file(receipt_path) != saved[0].get("receipt_sha256")):
            raise VerificationError(f"official fit artifact/receipt is not a successful durable fit: {fit_id}")
    origins: dict[str, Any] = {}
    for origin in (67, 75):
        manifest_path = PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
        source_prediction = PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        matrix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_origin = frozen["saved_inputs"]["origins"][f"E{origin}"]
        if _sha256_file(manifest_path) != frozen_origin["matrix_manifest_sha256"]:
            raise VerificationError(f"E{origin} saved matrix manifest changed")
        with np.load(source_prediction, allow_pickle=False) as source_archive:
            source = {name: source_archive[name].copy() for name in source_archive.files}
        fit_rows = np.asarray(source["fit_rows"], dtype=np.int64)
        fit_ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
        fit_matrix = sparse.csr_matrix(offset_inputs._load_matrix(matrix_manifest, "fit_base"), dtype=np.float64)
        fit_id_map = {entry["fit_id"]: entry for entry in
                      json.loads((ROOT / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json").read_text(encoding="utf-8"))["fit_units"]}
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        expected_target_sha = fit_id_map[bag_id]["fit_population"]["fit_target_sha256"]
        y_fit = offset_inputs._fit_labels(TRAIN, fit_rows, fit_ids, expected_target_sha)
        output_path = ROOT / ready["predictions"][f"E{origin}"]["path"]
        if _sha256_file(output_path) != ready["predictions"][f"E{origin}"]["sha256"]:
            raise VerificationError(f"E{origin} durable prediction hash mismatch")
        with np.load(output_path, allow_pickle=False) as archive:
            output = {name: archive[name].copy() for name in archive.files}
        rows = np.asarray(source["score_rows"], dtype=np.int64)
        expected_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
        expected_cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
        if (not np.array_equal(output["score_rows"], rows)
                or not np.array_equal(output["score_record_ids"].astype(str), expected_ids)
                or not np.array_equal(output["score_cycles"], expected_cycles)):
            raise VerificationError(f"E{origin} official prediction row order/identity mismatch")
        present = np.asarray(source["present_mask"], dtype=bool)
        fallback = np.asarray(source["prob_missing_hgb"], dtype=np.float64)
        if (not np.array_equal(output["present_mask"], present)
                or not np.array_equal(output["prob_missing_hgb"], fallback)
                or not np.array_equal(output["prob_bag"], source["prob_bag"])
                or not np.array_equal(output["prob_bag"][~present], fallback[~present])):
            raise VerificationError(f"E{origin} official stream availability/fallback source mismatch")
        a02_reference = frozen["saved_inputs"]["origins"][f"E{origin}"]["a02_current_references"]
        if a02_reference["status"] == "ALIGNED_REFERENCE_ONLY":
            a02_path = ROOT / a02_reference["path"]
            if _sha256_file(a02_path) != a02_reference["bundle_sha256"]:
                raise VerificationError(f"E{origin} cached A02/C16 reference hash changed")
            with np.load(a02_path, allow_pickle=False) as a02:
                a02_rows = np.asarray(a02["score_rows"], dtype=np.int64)
                a02_ids = np.asarray(a02["score_record_ids"]).astype(str)
                a02_cycles = np.asarray(a02["score_cycles"], dtype=np.int64)
                a02_current = np.asarray(a02["baseline_prediction"], dtype=np.float64)
                a02_observed = np.asarray(a02["prediction"], dtype=np.float64)
            if (not np.array_equal(a02_rows, rows) or not np.array_equal(a02_ids, expected_ids)
                    or not np.array_equal(a02_cycles, expected_cycles)
                    or not np.array_equal(output["prob_current_c16_reference"], a02_current)
                    or not np.array_equal(output["prob_a02_observed_reference"], a02_observed)):
                raise VerificationError(f"E{origin} cached C16/A02 references not row-aligned or not preserved")
        elif a02_reference["status"] != "UNKNOWN_REFERENCE_NOT_CACHED":
            raise VerificationError(f"E{origin} cached references have an unrecognized status")
        score_item = matrix_manifest["matrices"]["score_base"]
        matrix_path = Path(score_item["path"])
        if not matrix_path.is_absolute():
            matrix_path = ROOT / matrix_path
        if _sha256_file(matrix_path) != frozen["saved_inputs"][f"origins"][f"E{origin}"]["score_matrix"]["archive_sha256"]:
            raise VerificationError(f"E{origin} score matrix archive changed")
        x = sparse.load_npz(matrix_path).tocsr().astype(np.float64)
        if (offset_inputs.sha256_csr(x) != frozen["saved_inputs"][f"origins"][f"E{origin}"]["score_matrix"]["logical_sha256"]
                or list(x.shape) != frozen["saved_inputs"][f"origins"][f"E{origin}"]["score_matrix"]["shape"]):
            raise VerificationError(f"E{origin} score matrix logical identity mismatch")
        checks = {}
        for arm, suffix, factor in (("fm", "FM", True), ("nopair", "NOPAIR", False)):
            fit_id = f"{CARD_ID}_E{origin}_{suffix}"
            artifact = run_root / "units" / f"{fit_id}.npz"
            receipt_path = run_root / "receipts" / f"{fit_id}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt["fit_population"].get("fit_target_sha256") != expected_target_sha
                    or receipt["fit_population"].get("row_indices_sha256") != offset_inputs.sha256_array(fit_rows)
                    or receipt["fit_population"].get("record_ids_sha256") != offset_inputs.sha256_array(fit_ids)):
                raise VerificationError(f"{fit_id} fit-prefix target/row hash mismatch")
            with np.load(artifact, allow_pickle=False) as model:
                theta = np.asarray(model["parameters"], dtype=np.float64)
                persisted_fit_logits = np.asarray(model["fit_logits"], dtype=np.float64)
                persisted_fit_ids = np.asarray(model["fit_row_ids"]).astype(str)
                persisted_fit_cycles = np.asarray(model["fit_cycles"], dtype=np.int64)
            width = x.shape[1]
            expected_theta = width + 1 + (width * 16 if factor else 0)
            if theta.shape != (expected_theta,):
                raise VerificationError(f"{fit_id} parameter partition has wrong width")
            w = theta[:width]
            b = float(theta[width])
            v = theta[width + 1:].reshape(width, 16) if factor else None
            raw, pair = _independent_fm_score(x, w, b, v)
            fit_loss, fit_gradient, recomputed_fit_logits = _independent_objective_gradient(
                theta, fit_matrix, y_fit, 16, factor)
            receipt_opt = receipt["optimizer"]
            if (not np.allclose(recomputed_fit_logits, persisted_fit_logits, rtol=1e-12, atol=1e-12)
                    or not np.array_equal(persisted_fit_ids, fit_ids)
                    or not np.array_equal(persisted_fit_cycles, frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64))
                    or abs(fit_loss - receipt_opt["verified_final_loss_sum"]) > 1e-8
                    or abs(float(np.max(np.abs(fit_gradient))) - receipt_opt["verified_final_gradient_inf_norm_sum"]) > 1e-8):
                raise VerificationError(f"independent fit objective/gradient or fit-row binding mismatch: {fit_id}")
            direct = expit(raw)
            routed = direct.copy()
            routed[~present] = fallback[~present]
            output_key = "prob_fm" if factor else "prob_nopair"
            raw_key = "raw_score_fm" if factor else "raw_score_nopair"
            if (not np.allclose(raw, output[raw_key], rtol=1e-12, atol=1e-12)
                    or not np.allclose(routed, output[output_key], rtol=1e-12, atol=1e-12)
                    or not np.array_equal(routed[~present], fallback[~present])):
                raise VerificationError(f"independent official {arm} score/routing mismatch for E{origin}")
            if factor and not np.allclose(pair, output["pair_score_fm"], rtol=1e-12, atol=1e-12):
                raise VerificationError(f"independent official FM pair-score mismatch for E{origin}")
            checks[arm] = {"raw_score_sha256": offset_inputs.sha256_array(raw),
                           "probability_sha256": offset_inputs.sha256_array(routed),
                           "pair_score_sha256": offset_inputs.sha256_array(pair),
                           "fit_gradient_inf_norm_sum": float(np.max(np.abs(fit_gradient))),
                           "receipt_sha256": _sha256_file(receipt_path)}
        origins[f"E{origin}"] = {"score_rows": len(rows), "present_rows": int(present.sum()),
                                 "missing_rows": int((~present).sum()), "checks": checks}
        if a02_reference["status"] == "ALIGNED_REFERENCE_ONLY":
            origins[f"E{origin}"]["a02_current_references"] = {
                "status": "ALIGNED_REFERENCE_ONLY",
                "c16_sparse25_sha256": offset_inputs.sha256_array(a02_current),
                "a02_observed_sha256": offset_inputs.sha256_array(a02_observed),
                "score_ids_sha256": offset_inputs.sha256_array(a02_ids),
                "score_cycles_sha256": offset_inputs.sha256_array(a02_cycles),
            }
        else:
            origins[f"E{origin}"]["a02_current_references"] = {"status": "UNKNOWN_REFERENCE_NOT_CACHED"}
    return {"status": "PASS", "query_labels_opened": False,
            "source_commit": frozen["source_commit"], "freeze_signature": signature,
            "input_sha256": frozen["input_sha256"], "origins": origins,
            "verifier": "separate process; independent score formula; no fit or optimizer"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--synthetic-root", type=Path)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            if args.synthetic_root:
                result = verify_synthetic_bundle(args.synthetic_root)
            elif args.official and args.run_root:
                result = verify_official(args.run_root)
            else:
                parser.error("choose --synthetic-root or both --official and --run-root")
                return 2
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
