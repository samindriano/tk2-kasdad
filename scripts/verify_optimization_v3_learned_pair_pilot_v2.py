"""Independent, non-fitting verification for learned_pair_pilot_v2 outputs."""
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
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import optimization_v3_positional_bag_offset_v1 as offset_inputs

CARD_ID = "LEARNED_PAIR_PILOT_V2"
FIT_IDS = (
    "LEARNED_PAIR_PILOT_V2_E67_FM", "LEARNED_PAIR_PILOT_V2_E67_NOPAIR",
    "LEARNED_PAIR_PILOT_V2_E75_FM", "LEARNED_PAIR_PILOT_V2_E75_NOPAIR",
)
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
A02_REFERENCE_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
A02_PROTOCOL = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/protocol.json"
A02_FREEZE = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
TRAIN = ROOT / "data/raw/classification/Train.csv"
SOURCE_FILES = (
    "scripts/optimization_v3_learned_pair_pilot_v2.py",
    "scripts/verify_optimization_v3_learned_pair_pilot_v2.py",
    "src/models/learned_pair_pilot_v1.py",
    "tests/test_learned_pair_pilot_v1.py",
    "tests/test_learned_pair_pilot_v2.py",
    "reports/optimization_v3/learned_pair_pilot_v2/PROTOCOL.md",
    "reports/optimization_v3/learned_pair_pilot_v2/SOL_EXECUTION_HANDOFF.md",
    "reports/optimization_v3/learned_pair_pilot_v1/PROTOCOL.md",
    "reports/optimization_v3/learned_pair_pilot_v1/FAILURE_DIAGNOSTIC.md",
    "reports/optimization_v3/learned_pair_preflight_v1/REPORT.md",
    "reports/optimization_v3/learned_pair_preflight_v1/SOL_PILOT_AUTHORIZATION.md",
    "experiments/optimization_v3/c19_a02_v2/A02/protocol.json",
    "experiments/optimization_v3/c19_a02_v2/A02/freeze.json",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "src/models/classification_a02_neighbors.py",
    "src/models/research_next_contract.py",
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


def _verify_a02_freeze() -> dict[str, Any]:
    if not A02_PROTOCOL.is_file() or not A02_FREEZE.is_file():
        raise VerificationError("frozen A02 protocol/freeze is missing")
    protocol = json.loads(A02_PROTOCOL.read_text(encoding="utf-8"))
    frozen = json.loads(A02_FREEZE.read_text(encoding="utf-8"))
    if (_canonical_sha256(protocol) != frozen.get("protocol_sha256")
            or _canonical_sha256(frozen.get("identity")) != frozen.get("freeze_signature")
            or frozen.get("identity", {}).get("task") != "A02"):
        raise VerificationError("frozen A02 protocol/freeze signature is invalid")
    for relative, expected in frozen.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise VerificationError(f"frozen A02 source changed/missing: {relative}")
    for relative, expected in frozen.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise VerificationError(f"frozen A02 input changed/missing: {relative}")
    return frozen


def _verify_a02_observed(origin: int, fit_rows: np.ndarray, score_rows: np.ndarray,
                         score_ids: np.ndarray, score_cycles: np.ndarray,
                         expected_meta: dict[str, Any], a02_freeze: dict[str, Any]) -> np.ndarray:
    folder = A02_REFERENCE_ROOT / f"E{origin}" / "observed"
    required = ("predictions.npz", "fit_receipt.json", "fit_state.json", "model.joblib", "predictions.json")
    if expected_meta.get("status") != "ALIGNED_VERIFIED_REFERENCE_ONLY":
        raise VerificationError(f"E{origin} A02 exact-origin comparator was not frozen as verified")
    if any(not (folder / name).is_file() for name in required):
        raise VerificationError(f"E{origin} A02 observed bundle is incomplete")
    archive_path, receipt_path = folder / "predictions.npz", folder / "fit_receipt.json"
    state_path, model_path = folder / "fit_state.json", folder / "model.joblib"
    sidecar_path = folder / "predictions.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    receipt_body = dict(receipt)
    receipt_sig = receipt_body.pop("receipt_sha256", None)
    if (receipt_sig != _canonical_sha256(receipt_body)
            or receipt.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or receipt.get("origin_cycle") != origin or receipt.get("residual_arm") != "observed"
            or receipt.get("freeze_signature") != a02_freeze.get("freeze_signature")
            or receipt.get("model_sha256") != _sha256_file(model_path)
            or receipt.get("state_sha256") != _canonical_sha256(state)
            or state.get("card_id") != "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"
            or state.get("origin_cycle") != origin or state.get("residual_arm") != "observed"
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False
            or receipt.get("task_estimator_fit_count") != 1
            or int(receipt.get("text_prefix_rows", -1)) != len(fit_rows)
            or receipt.get("text_prefix_row_positions_sha256") != hashlib.sha256(
                fit_rows.astype("<i8").tobytes()).hexdigest()
            or sidecar.get("fit_id") != receipt.get("fit_id")
            or sidecar.get("fit_receipt_sha256") != receipt_sig
            or sidecar.get("origin_cycle") != origin or sidecar.get("residual_arm") != "observed"
            or sidecar.get("freeze_signature") != receipt.get("freeze_signature")
            or sidecar.get("npz_sha256") != _sha256_file(archive_path)
            or sidecar.get("query_labels_passed_to_prediction") is not False):
        raise VerificationError(f"E{origin} A02 fit/state/model/output receipt binding mismatch")
    for key, path in (("bundle_sha256", archive_path), ("fit_receipt_sha256", receipt_path),
                      ("fit_state_sha256", state_path), ("model_sha256", model_path),
                      ("prediction_sidecar_sha256", sidecar_path)):
        if expected_meta.get(key) != _sha256_file(path):
            raise VerificationError(f"E{origin} A02 frozen {key} changed")
    with np.load(archive_path, allow_pickle=False) as archive:
        # Deliberately read prediction/identity arrays only. The archive's labels array remains closed.
        rows = np.asarray(archive["score_rows"], dtype=np.int64)
        ids = np.asarray(archive["score_record_ids"]).astype(str)
        cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
        probability = np.asarray(archive["prediction"], dtype=np.float64)
    if (not np.array_equal(rows, score_rows) or not np.array_equal(ids, score_ids.astype(str))
            or not np.array_equal(cycles, score_cycles.astype(np.int64))
            or probability.shape != score_rows.shape or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or offset_inputs.sha256_array(probability) != expected_meta.get("observed_probability_sha256")
            or offset_inputs.sha256_array(ids) != expected_meta.get("score_ids_sha256")
            or offset_inputs.sha256_array(cycles) != expected_meta.get("score_cycles_sha256")):
        raise VerificationError(f"E{origin} A02 predictions are not exact-origin row aligned")
    sidecar_ids_sha = hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":"),
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    if (int(sidecar.get("score_rows", -1)) != len(score_rows)
            or sidecar.get("score_row_positions_sha256") != hashlib.sha256(score_rows.astype("<i8").tobytes()).hexdigest()
            or sidecar.get("score_record_ids_sha256") != sidecar_ids_sha):
        raise VerificationError(f"E{origin} A02 prediction sidecar row identity mismatch")
    return probability


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
    if "prob_bag_raw" in prediction or "prob_bag" in prediction:
        if "prob_bag_raw" not in prediction or "prob_bag" not in prediction:
            raise VerificationError("synthetic cached bag route lacks raw or routed probabilities")
        bag_raw = np.asarray(prediction["prob_bag_raw"], dtype=np.float64)
        bag_routed = np.asarray(prediction["prob_bag"], dtype=np.float64)
        if (bag_raw.shape != present.shape or bag_routed.shape != present.shape
                or not np.isfinite(bag_raw).all() or not np.isfinite(bag_routed).all()
                or (bag_raw < 0).any() or (bag_raw > 1).any()
                or (bag_routed < 0).any() or (bag_routed > 1).any()):
            raise VerificationError("synthetic cached bag probabilities are invalid")
        expected_bag = bag_raw.copy()
        expected_bag[~present] = fallback[~present]
        if not np.array_equal(bag_routed, expected_bag):
            raise VerificationError("cached bag raw/routed availability mapping is invalid")
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
    gradient_inf = float(np.max(np.abs(train_grad)))
    normalized_gradient_inf = gradient_inf / len(fit_y)
    if (not np.allclose(fit_logits, persisted_fit_logits, rtol=1e-12, atol=1e-12)
            or abs(train_loss - opt["verified_final_loss_sum"]) > 1e-8
            or abs(gradient_inf - opt["verified_final_gradient_inf_norm_sum"]) > 1e-8
            or abs(normalized_gradient_inf - opt.get("verified_normalized_gradient_inf_norm", float("inf"))) > 1e-12
            or normalized_gradient_inf > 1e-4
            or abs(float(len(fit_y) * 1e-4) - opt.get("gtol", float("nan"))) > 1e-12
            or not opt.get("success") or not opt.get("finite_state")
            or not opt.get("stationarity_guard") or not opt.get("loss_nonincrease_guard")
            or opt.get("final_loss_sum", float("inf")) > opt.get("initial_loss_sum", float("nan")) + 1e-8):
        raise VerificationError("independent fit objective/gradient does not match persisted receipt")
    return {"status": "PASS", "rows": int(x.shape[0]), "features": int(x.shape[1]),
            "raw_score_sha256": offset_inputs.sha256_array(logits),
            "routed_probability_sha256": offset_inputs.sha256_array(routed),
            "pair_score_sha256": offset_inputs.sha256_array(pair),
            "fit_gradient_inf_norm_sum": gradient_inf,
            "fit_gradient_inf_norm_mean": normalized_gradient_inf,
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
    a02_freeze = _verify_a02_freeze()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (ready.get("freeze_signature") != signature or ready.get("query_labels_opened") is not False
            or set(ready.get("predictions", {})) != {"E67", "E75"}
            or ready.get("fit_ids") != list(FIT_IDS)):
        raise VerificationError("official prediction readiness record invalid")
    ledger = run_root.parent / "budget.jsonl"
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    budget = offset_inputs.ledger_snapshot(events)
    expected_budget = {
        "global_cap": 117,
        "effective_category_caps": {"continuation": 38, "control_replay": 2, "pilot": 76, "technical_retry": 1},
        "active_reservations_by_category": {"continuation": 38, "control_replay": 2, "pilot": 73, "technical_retry": 1},
        "active_reservations": 114, "fit_starts": 114, "fit_saved": 110,
        "fit_failed": 4, "fit_pending": 0, "event_rows": len(events),
    }
    if budget != expected_budget:
        raise VerificationError(f"V2 exact budget snapshot mismatch before query labels: {budget}")
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
        bag_raw = np.asarray(source["prob_bag"], dtype=np.float64)
        bag_route = np.asarray(source["prob_route_bag"], dtype=np.float64)
        if (present.shape != rows.shape or fallback.shape != rows.shape
                or bag_raw.shape != rows.shape or bag_route.shape != rows.shape):
            raise VerificationError(f"E{origin} official stream availability arrays are misaligned")
        expected_bag_route = bag_raw.copy()
        expected_bag_route[~present] = fallback[~present]
        if (not np.array_equal(output["present_mask"], present)
                or not np.array_equal(output["prob_missing_hgb"], fallback)
                or not np.array_equal(bag_route, expected_bag_route)
                or not np.array_equal(output["prob_bag"], bag_route)
                or not np.array_equal(output["prob_bag_raw"], bag_raw)):
            raise VerificationError(f"E{origin} official stream availability/fallback source mismatch")
        a02_reference = frozen["saved_inputs"]["origins"][f"E{origin}"]["a02_observed_reference"]
        a02_observed = _verify_a02_observed(
            origin, fit_rows, rows, expected_ids, expected_cycles, a02_reference, a02_freeze)
        if ("prob_a02_observed" not in output
                or not np.array_equal(output["prob_a02_observed"], a02_observed)):
            raise VerificationError(f"E{origin} exact-origin A02 comparator was not preserved")
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
            normalized_grad = float(np.max(np.abs(fit_gradient)) / len(y_fit))
            expected_gtol = float(len(y_fit) * 1e-4)
            if (not np.allclose(recomputed_fit_logits, persisted_fit_logits, rtol=1e-12, atol=1e-12)
                    or not np.array_equal(persisted_fit_ids, fit_ids)
                    or not np.array_equal(persisted_fit_cycles, frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64))
                    or abs(fit_loss - receipt_opt["verified_final_loss_sum"]) > 1e-8
                    or abs(float(np.max(np.abs(fit_gradient))) - receipt_opt["verified_final_gradient_inf_norm_sum"]) > 1e-8
                    or abs(normalized_grad - receipt_opt.get("verified_normalized_gradient_inf_norm", float("inf"))) > 1e-12
                    or normalized_grad > 1e-4
                    or abs(expected_gtol - receipt_opt.get("gtol", float("nan"))) > 1e-12
                    or not receipt_opt.get("success") or not receipt_opt.get("finite_state")
                    or not receipt_opt.get("stationarity_guard")
                    or not receipt_opt.get("loss_nonincrease_guard")
                    or receipt_opt.get("final_loss_sum", float("inf"))
                       > receipt_opt.get("initial_loss_sum", float("nan")) + 1e-8):
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
                           "fit_gradient_inf_norm_mean": normalized_grad,
                           "required_gtol_sum_scale": expected_gtol,
                           "receipt_sha256": _sha256_file(receipt_path)}
        origins[f"E{origin}"] = {"score_rows": len(rows), "present_rows": int(present.sum()),
                                 "missing_rows": int((~present).sum()), "checks": checks}
        origins[f"E{origin}"]["a02_reference"] = {
                "status": "ALIGNED_VERIFIED_REFERENCE_ONLY",
                "a02_observed_sha256": offset_inputs.sha256_array(a02_observed),
                "score_ids_sha256": offset_inputs.sha256_array(expected_ids),
                "score_cycles_sha256": offset_inputs.sha256_array(expected_cycles),
            }
    return {"status": "PASS", "query_labels_opened": False,
            "source_commit": frozen["source_commit"], "freeze_signature": signature,
            "input_sha256": frozen["input_sha256"], "origins": origins,
            "verifier": "separate process; independent score formula; no fit or optimizer"}


def _metric_independent(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise VerificationError("independent metric input is not a finite aligned probability vector")
    if len(y) == 0:
        return {"n": 0, "positive": 0, "negative": 0, "macro_f1": None,
                "positive_f1": None, "precision": None, "positive_recall": None,
                "roc_auc": None, "average_precision": None, "log_loss": None,
                "brier": None, "confusion_matrix": [[0, 0], [0, 0]]}
    pred = (p >= 0.5).astype(np.int8)
    both = np.unique(y).size == 2
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def _label_independent(value: Any) -> str:
    return "<missing>" if pd.isna(value) else str(value)


def _views_independent(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {"W4": (cycles >= 68) & (cycles <= 71),
                "W5": (cycles >= 72) & (cycles <= 75),
                "W6": (cycles >= 76) & (cycles <= 79),
                "H12_E67": (cycles >= 68) & (cycles <= 79)}
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def _supported_independent(metric: dict[str, Any]) -> bool:
    return metric["n"] >= 200 and metric["positive"] >= 30 and metric["negative"] >= 30


def _gates_independent(origin: int, views: list[dict[str, Any]],
                       groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_view = {(item["view"], item["arm"]): item for item in views}
    result: list[dict[str, Any]] = []

    def difference(view: str, comparator: str, metric: str) -> float:
        return float(by_view[(view, "fm")][metric] - by_view[(view, comparator)][metric])

    def emit(track: str, gate: str, view: str, comparator: str, metric: str,
             threshold: float, value: float) -> None:
        result.append({"track": track, "gate": gate, "origin": origin, "view": view,
                       "arm": "fm", "comparator": comparator, "metric": metric,
                       "value": value, "threshold": threshold,
                       "pass": bool(value >= threshold - 1e-12)})

    if origin == 67:
        for control, minimum in (("bag", 0.0005), ("nopair", 0.0002)):
            value = difference("H12_E67", control, "macro_f1")
            emit("mechanism", f"E67_H12_gain_min_{minimum}_vs_{control}", "H12_E67",
                 control, "macro_f1", minimum, value)
        value = difference("H12_E67", "a02_observed", "macro_f1")
        emit("competition", "E67_H12_gain_min_0.0005_vs_a02_observed", "H12_E67",
             "a02_observed", "macro_f1", 0.0005, value)
        for view in ("W4", "W5", "W6"):
            for control in ("bag", "nopair"):
                value = difference(view, control, "macro_f1")
                emit("mechanism", f"E67_{view}_harm_max_0.005_vs_{control}", view,
                     control, "macro_f1", -0.005, value)
            value = difference(view, "a02_observed", "macro_f1")
            emit("competition", f"E67_{view}_harm_max_0.005_vs_a02_observed", view,
                 "a02_observed", "macro_f1", -0.005, value)
    else:
        for control in ("bag", "nopair"):
            value = difference("W6", control, "macro_f1")
            emit("mechanism", f"E75_W6_gain_nonnegative_vs_{control}", "W6",
                 control, "macro_f1", 0.0, value)
        value = difference("W6", "a02_observed", "macro_f1")
        emit("competition", "E75_W6_gain_min_0.0005_vs_a02_observed", "W6",
             "a02_observed", "macro_f1", 0.0005, value)
    by_group = {(g["view"], g["group"], g["group_value"], g["arm"]): g for g in groups}
    for item in groups:
        if item["arm"] != "fm":
            continue
        supported = _supported_independent(item)
        result.append({"track": "both", "gate": f"{item['view']}_{item['group']}_{item['group_value']}_support",
                       "origin": origin, "view": item["view"], "group": item["group"],
                       "group_value": item["group_value"], "arm": "fm",
                       "value": {"n": item["n"], "positive": item["positive"], "negative": item["negative"]},
                       "threshold": {"n": 200, "each_class": 30},
                       "pass": True if supported else None,
                       "status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED"})
        if not supported:
            continue
        for comparator in ("bag", "nopair", "a02_observed"):
            track = "competition" if comparator == "a02_observed" else "mechanism"
            reference = by_group[(item["view"], item["group"], item["group_value"], comparator)]
            for metric, max_harm in (("macro_f1", 0.01), ("positive_recall", 0.02)):
                value = float(item[metric] - reference[metric])
                result.append({"track": track,
                               "gate": f"{item['view']}_{item['group']}_{item['group_value']}_{metric}_harm_vs_{comparator}",
                               "origin": origin, "view": item["view"], "group": item["group"],
                               "group_value": item["group_value"], "arm": "fm", "comparator": comparator,
                               "metric": metric, "value": value, "threshold": -max_harm,
                               "pass": bool(value >= -max_harm - 1e-12)})
    return result


def verify_scored(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    result_path = run_root / "RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ready_path = run_root / "PREDICTIONS_READY.json"
    access_path, read_path = run_root / "QUERY_LABEL_ACCESS.json", run_root / "QUERY_LABEL_READ.json"
    prelabel_path = run_root / "PRELABEL_VERIFICATION.json"
    if any(not path.is_file() for path in (ready_path, access_path, read_path, prelabel_path)):
        raise VerificationError("score/access/read/prelabel receipts are incomplete")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    read_receipt = json.loads(read_path.read_text(encoding="utf-8"))
    prelabel = json.loads(prelabel_path.read_text(encoding="utf-8"))
    prelabel_body = dict(prelabel)
    prelabel_signature = prelabel_body.pop("verification_signature", None)
    if (result.get("status") != "COMPLETED_DIAGNOSTIC_ONLY"
            or result.get("query_labels_opened_after_prelabel_verification") is not True
            or result.get("query_label_access_receipt_sha256") != _sha256_file(access_path)
            or result.get("query_label_read_receipt_sha256") != _sha256_file(read_path)
            or result.get("predictions_ready_sha256") != _sha256_file(ready_path)
            or access.get("prelabel_verification_sha256") != _sha256_file(prelabel_path)
            or access.get("prediction_ready_sha256") != _sha256_file(ready_path)
            or read_receipt.get("query_label_access_sha256") != _sha256_file(access_path)
            or prelabel.get("status") != "PASS"
            or prelabel.get("freeze_signature") != result.get("freeze_signature")
            or prelabel_signature != _canonical_sha256(prelabel_body)
            or ready.get("query_labels_opened") is not False
            or ready.get("freeze_signature") != result.get("freeze_signature")
            or read_receipt.get("train_sha256") != result.get("input_sha256")
            or read_receipt.get("rows_read") != 50_608
            or read_receipt.get("columns_read") != ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"]):
        raise VerificationError("saved scoring receipts do not prove the query-label barrier")
    if _sha256_file(TRAIN) != result.get("input_sha256"):
        raise VerificationError("official Train hash changed before independent metric verification")
    train = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
                        dtype={"record_id": "string"})
    y_all = pd.to_numeric(train["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    cycles_all = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.isin(y_all, (0, 1)).all():
        raise VerificationError("official Train target is not binary")
    actual_origins: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for origin in (67, 75):
        key = f"E{origin}"
        item = ready.get("predictions", {}).get(key)
        if item is None:
            raise VerificationError(f"durable predictions missing for {key}")
        prediction_path = ROOT / item["path"]
        if _sha256_file(prediction_path) != item.get("sha256"):
            raise VerificationError(f"durable prediction hash changed for {key}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            pred = {name: archive[name].copy() for name in archive.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        if (not np.array_equal(train.iloc[rows]["record_id"].astype(str).to_numpy(), pred["score_record_ids"].astype(str))
                or not np.array_equal(cycles_all[rows], pred["score_cycles"].astype(np.int64))):
            raise VerificationError(f"saved score rows/IDs/cycles do not match official Train for {key}")
        y = y_all[rows]
        cycles = cycles_all[rows]
        arms = {"bag": np.asarray(pred["prob_bag"], dtype=np.float64),
                "nopair": np.asarray(pred["prob_nopair"], dtype=np.float64),
                "a02_observed": np.asarray(pred["prob_a02_observed"], dtype=np.float64),
                "fm": np.asarray(pred["prob_fm"], dtype=np.float64)}
        if any(prob.shape != y.shape or not np.isfinite(prob).all()
               or np.any((prob < 0) | (prob > 1)) for prob in arms.values()):
            raise VerificationError(f"metric probability output is invalid for {key}")
        view_masks = _views_independent(origin, cycles)
        present = np.asarray(pred["present_mask"], dtype=bool)
        sub = train.iloc[rows]
        text_load = pd.to_numeric(sub["text_load"], errors="coerce").to_numpy(dtype=np.float64)
        matrix_manifest = json.loads((PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json").read_text(encoding="utf-8"))
        q75 = float(matrix_manifest["text_load_q75_fit_only"])
        if result["origins"][key]["fit_defined_text_load_q75"] != q75:
            raise VerificationError(f"fit-only text-load threshold changed for {key}")
        q4 = np.isfinite(text_load) & (text_load >= q75)
        group_masks: list[tuple[str, str, np.ndarray]] = [
            ("availability", "present", present),
            ("availability", "missing_or_blank", ~present),
            ("text_load", "Q4", q4),
            ("text_load", "below_Q4", np.isfinite(text_load) & ~q4),
            ("text_load", "missing", ~np.isfinite(text_load)),
        ]
        friction = sub["entry_friction"].to_numpy(dtype=object)
        for value in sorted({_label_independent(v) for v in friction}):
            group_masks.append(("entry_friction", value,
                                np.asarray([_label_independent(v) == value for v in friction], dtype=bool)))
        views: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        cycles_report: list[dict[str, Any]] = []
        for view, mask in view_masks.items():
            for arm, probability in arms.items():
                views.append({"view": view, "arm": arm, **_metric_independent(y[mask], probability[mask])})
            for group, value, group_mask in group_masks:
                for arm, probability in arms.items():
                    groups.append({"view": view, "group": group, "group_value": value,
                                   "arm": arm, **_metric_independent(y[mask & group_mask], probability[mask & group_mask])})
        for cycle in sorted(np.unique(cycles).tolist()):
            mask = cycles == cycle
            metrics = {arm: _metric_independent(y[mask], probability[mask]) for arm, probability in arms.items()}
            cycles_report.append({
                "cycle": int(cycle), "n": int(mask.sum()), "metrics": metrics,
                "fm_minus_bag_macro_f1": metrics["fm"]["macro_f1"] - metrics["bag"]["macro_f1"],
                "fm_minus_nopair_macro_f1": metrics["fm"]["macro_f1"] - metrics["nopair"]["macro_f1"],
                "fm_minus_a02_observed_macro_f1": metrics["fm"]["macro_f1"] - metrics["a02_observed"]["macro_f1"],
            })
        gates = _gates_independent(origin, views, groups)
        all_gates.extend(gates)
        expected_origin = result["origins"][key]
        if (_canonical_sha256(views) != _canonical_sha256(expected_origin["views"])
                or _canonical_sha256(groups) != _canonical_sha256(expected_origin["subgroups"])
                or _canonical_sha256(cycles_report) != _canonical_sha256(expected_origin["per_cycle"])
                or _canonical_sha256(gates) != _canonical_sha256(expected_origin["gates"])
                or _canonical_sha256({
                    comparator: {
                        "positive": int(sum(row[f"fm_minus_{comparator}_macro_f1"] > 0 for row in cycles_report)),
                        "negative": int(sum(row[f"fm_minus_{comparator}_macro_f1"] < 0 for row in cycles_report)),
                        "tie": int(sum(row[f"fm_minus_{comparator}_macro_f1"] == 0 for row in cycles_report)),
                    } for comparator in ("bag", "nopair", "a02_observed")
                }) != _canonical_sha256(expected_origin.get("per_cycle_gain_sign_counts"))):
            raise VerificationError(f"independent metric, subgroup, cycle, or gate rows differ for {key}")
        actual_origins[key] = {
            "score_rows": len(rows), "positive": int(y.sum()), "negative": int((1-y).sum()),
            "views": views, "subgroups": groups, "per_cycle": cycles_report, "gates": gates,
            "per_cycle_sign_counts": {
                comparator: {
                    "positive": int(sum(row[f"fm_minus_{comparator}_macro_f1"] > 0 for row in cycles_report)),
                    "negative": int(sum(row[f"fm_minus_{comparator}_macro_f1"] < 0 for row in cycles_report)),
                    "tie": int(sum(row[f"fm_minus_{comparator}_macro_f1"] == 0 for row in cycles_report)),
                }
                for comparator in ("bag", "nopair", "a02_observed")
            },
        }
    mechanism = [gate for gate in all_gates if gate.get("track") == "mechanism"]
    competition = [gate for gate in all_gates if gate.get("track") == "competition"]
    optimizer_pass = len(result.get("fit_receipts", {})) == 4 and result.get("optimizer_all_success") is True
    mechanism_pass = optimizer_pass and all(gate["pass"] is True for gate in mechanism)
    competition_pass = mechanism_pass and all(gate["pass"] is True for gate in competition)
    expected_verdicts = {
        "mechanism": "MECHANISM_PASS" if mechanism_pass else "MECHANISM_FAIL",
        "competition": "COMPETITION_PASS" if competition_pass else "COMPETITION_FAIL",
    }
    if (_canonical_sha256(all_gates) != _canonical_sha256(result.get("gate_checks"))
            or result.get("verdicts") != expected_verdicts
            or result.get("mechanism_gates_pass") is not mechanism_pass
            or result.get("competition_gates_pass") is not competition_pass):
        raise VerificationError("independent overall gate/verdict recomputation differs")
    body = {
        "status": "PASS_INDEPENDENT_METRICS_AND_GATES",
        "result_sha256": _sha256_file(result_path),
        "train_sha256": _sha256_file(TRAIN),
        "query_label_read_receipt_sha256": _sha256_file(read_path),
        "prelabel_verification_sha256": _sha256_file(prelabel_path),
        "prediction_sha256s": {key: _sha256_file(ROOT / item["path"])
                                for key, item in ready["predictions"].items()},
        "verdicts": expected_verdicts,
        "mechanism_gate_count": len(mechanism),
        "competition_gate_count": len(competition),
        "unsupported_group_count": sum(g.get("status") == "UNKNOWN_UNSUPPORTED" for g in all_gates),
        "independently_recomputed_origins": actual_origins,
        "labels_read_only_for_post-fit_evaluation": True,
        "no_fit_or_model_selection_performed": True,
    }
    body["verification_signature"] = _canonical_sha256(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--scored", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--synthetic-root", type=Path)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            if args.synthetic_root:
                result = verify_synthetic_bundle(args.synthetic_root)
            elif args.scored and args.run_root:
                result = verify_scored(args.run_root)
            elif args.official and args.run_root:
                result = verify_official(args.run_root)
            else:
                parser.error("choose --synthetic-root, --scored with --run-root, or --official with --run-root")
                return 2
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
