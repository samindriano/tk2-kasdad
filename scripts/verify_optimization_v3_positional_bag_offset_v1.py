"""Independent, no-refit verification for POS_BAG_OFFSET_V1 receipts."""
from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha(value: Any) -> str:
    a = np.asarray(value)
    header = _canonical_sha({"shape": a.shape, "dtype": str(a.dtype)})
    h = hashlib.sha256(header.encode("ascii"))
    if a.dtype.kind in "OUS":
        h.update(json.dumps(a.astype(str).tolist(), ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csr_sha(value: sparse.spmatrix) -> str:
    x = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    x.sum_duplicates()
    x.sort_indices()
    h = hashlib.sha256(_canonical_sha({"shape": x.shape, "dtype": str(x.dtype)}).encode("ascii"))
    h.update(np.asarray(x.indptr, dtype="<i8").tobytes())
    h.update(np.asarray(x.indices, dtype="<i8").tobytes())
    h.update(np.asarray(x.data, dtype="<f8").tobytes())
    return h.hexdigest()


def _objective_gradient(x: sparse.spmatrix, z0: np.ndarray, y: np.ndarray,
                        gamma: np.ndarray, c: float = 1.0) -> tuple[float, np.ndarray]:
    """Independent expression; intentionally does not import the optimizer."""
    x = sparse.csr_matrix(x, dtype=np.float64)
    z = np.asarray(z0, dtype=np.float64).reshape(-1) + np.asarray(x @ gamma, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    beta = np.asarray(gamma, dtype=np.float64).reshape(-1)
    objective = np.logaddexp(0.0, -(2.0 * target - 1.0) * z).sum() + (beta @ beta) / (2.0 * c)
    grad = np.asarray(x.T @ (expit(z) - target), dtype=np.float64).reshape(-1) + beta / c
    return float(objective), grad


def _load_csr(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()


def _select_fit_labels(path: Path, fit_rows: np.ndarray,
                       expected_ids: np.ndarray) -> np.ndarray:
    rows = np.asarray(fit_rows, dtype=np.int64)
    ids = np.asarray(expected_ids).astype(str)
    if len(rows) != len(ids) or len(np.unique(rows)) != len(rows):
        raise AssertionError("frozen fit row ID vector is invalid")
    wanted = {int(row): pos for pos, row in enumerate(rows.tolist())}
    target = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or "record_id" not in header or "artifact_signal" not in header:
            raise AssertionError("Train CSV header is invalid")
        id_column, target_column = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            pos = wanted.get(row_index)
            if pos is None:
                continue
            if max(id_column, target_column) >= len(record):
                raise AssertionError("frozen fit CSV row is malformed")
            if record[id_column] != ids[pos]:
                raise AssertionError("frozen fit target ID mismatch")
            target[pos] = int(record[target_column])
            seen[pos] = True
    if not seen.all() or not np.isin(target, (0, 1)).all():
        raise AssertionError("frozen fit target values are absent or invalid")
    return target


def verify_synthetic_bundle(artifact_path: Path, receipt_path: Path) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("artifact_sha256") != _file_sha(artifact_path):
        raise AssertionError("synthetic artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as data:
        gamma = np.asarray(data["gamma"], dtype=np.float64)
        x = sparse.csr_matrix((data["x_data"], data["x_indices"], data["x_indptr"]),
                              shape=tuple(data["x_shape"].astype(int).tolist()))
        base = sparse.csr_matrix((data["base_data"], data["base_indices"], data["base_indptr"]),
                                 shape=tuple(data["base_shape"].astype(int).tolist()))
        y = np.asarray(data["y"], dtype=np.int8)
        z0 = np.asarray(data["z0"], dtype=np.float64)
        saved_probability = np.asarray(data["probability"], dtype=np.float64)
        fit_ids = np.asarray(data["fit_ids"]).astype(str)
        score_ids = np.asarray(data["score_ids"]).astype(str)
        classes = np.asarray(data["classes"], dtype=np.int8)
        present = np.asarray(data["present"], dtype=bool)
        fallback = np.asarray(data["fallback"], dtype=np.float64)
        base_coef = np.asarray(data["base_coef"], dtype=np.float64)
        base_intercept = np.asarray(data["base_intercept"], dtype=np.float64)
    if receipt.get("gamma_sha256") != _array_sha(gamma):
        raise AssertionError("synthetic gamma hash mismatch")
    if (receipt.get("fit_ids_sha256") != _array_sha(fit_ids)
            or receipt.get("score_ids_sha256") != _array_sha(score_ids)
            or receipt.get("base_coef_sha256") != _array_sha(base_coef)
            or receipt.get("base_intercept_sha256") != _array_sha(base_intercept)):
        raise AssertionError("synthetic row ID or frozen bag parameter hash mismatch")
    if receipt.get("base_matrix_sha256") != _csr_sha(base):
        raise AssertionError("synthetic frozen bag matrix hash mismatch")
    reconstructed_offset = base_intercept[0] + np.asarray(base @ base_coef).reshape(-1)
    if not np.array_equal(reconstructed_offset, z0):
        raise AssertionError("synthetic fixed bag offset was changed")
    if not np.array_equal(classes, np.asarray([0, 1], dtype=np.int8)):
        raise AssertionError("synthetic class order mismatch")
    loss, gradient = _objective_gradient(x, z0, y, gamma)
    zero_loss, _ = _objective_gradient(x, z0, y, np.zeros_like(gamma))
    probability = expit(z0 + x @ gamma)
    routed = probability.copy()
    routed[~present] = fallback[~present]
    if not np.allclose(saved_probability, routed, atol=1e-14, rtol=0):
        raise AssertionError("synthetic probability formula mismatch")
    if not np.array_equal(saved_probability[~present], fallback[~present]):
        raise AssertionError("synthetic exact missing fallback mismatch")
    if not np.isfinite(loss) or not np.isfinite(gradient).all() or loss > zero_loss + 1e-8:
        raise AssertionError("synthetic independent objective check failed")
    meta = receipt["optimizer"]
    if not np.isclose(loss, meta["loss_final"], atol=1e-8, rtol=0):
        raise AssertionError("synthetic receipt loss mismatch")
    return {"status": "PASS", "loss": loss,
            "gradient_inf_norm": float(np.max(np.abs(gradient))),
            "zero_gamma_loss": float(zero_loss), "rows": int(x.shape[0]),
            "features": int(x.shape[1]), "optimizer_called": False}


def verify_prelabel_outputs(root: Path, run_root: Path,
                            freeze_data: Mapping[str, Any]) -> dict[str, Any]:
    """Verify saved fit math and query scores without accessing query labels."""
    root = Path(root)
    run_root = Path(run_root)
    train_path = root / "data/raw/classification/Train.csv"
    pilot = root / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
    accounting = json.loads((root / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json").read_text(encoding="utf-8"))
    accounting_by_id = {x["fit_id"]: x for x in accounting["fit_units"]}
    ready = json.loads((run_root / "PREDICTIONS_READY.json").read_text(encoding="utf-8"))
    if ready.get("freeze_signature") != freeze_data["freeze_signature"] or ready.get("query_labels_opened") is not False:
        raise AssertionError("prelabel prediction manifest does not match freeze or labels were opened")
    if set(ready.get("predictions", {})) != {"E67", "E75"}:
        raise AssertionError("prediction readiness manifest must contain exactly E67 and E75")
    identity = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    origins: dict[str, Any] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        frozen_origin = freeze_data["origins"][key]
        legacy_manifest_path = pilot / f"MATRIX_MANIFEST_E{origin}.json"
        if _file_sha(legacy_manifest_path) != frozen_origin["matrix_manifest_sha256"]:
            raise AssertionError(f"E{origin} pilot manifest differs from amendment freeze")
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        legacy_prediction_path = pilot / key / f"predictions_E{origin}.npz"
        if _file_sha(legacy_prediction_path) != frozen_origin["prediction_bundle_sha256"]:
            raise AssertionError(f"E{origin} pilot prediction bundle differs from amendment freeze")
        with np.load(legacy_prediction_path, allow_pickle=False) as a:
            legacy = {k: a[k].copy() for k in a.files}
        output_path = run_root / key / f"predictions_E{origin}.npz"
        if _file_sha(output_path) != ready["predictions"][key]["sha256"]:
            raise AssertionError(f"E{origin} prediction output hash mismatch")
        with np.load(output_path, allow_pickle=False) as a:
            output = {k: a[k].copy() for k in a.files}
        fit_rows = np.asarray(output["fit_rows"], dtype=np.int64)
        score_rows = np.asarray(output["score_rows"], dtype=np.int64)
        if not np.array_equal(fit_rows, legacy["fit_rows"]) or not np.array_equal(score_rows, legacy["score_rows"]):
            raise AssertionError(f"E{origin} frozen row indices changed")
        ids = identity.iloc[score_rows]["record_id"].astype(str).to_numpy()
        cycles = pd.to_numeric(identity.iloc[score_rows]["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if not np.array_equal(ids, output["score_record_ids"].astype(str)) or not np.array_equal(cycles, output["score_cycles"]):
            raise AssertionError(f"E{origin} score row IDs/cycles mismatch")
        if not np.array_equal(output["present_mask"], legacy["present_mask"]):
            raise AssertionError(f"E{origin} availability mask mismatch")
        matrix_items = legacy_manifest["matrices"]
        matrices: dict[str, sparse.csr_matrix] = {}
        matrices: dict[str, sparse.csr_matrix] = {}
        needed = {"fit_base", "fit_position", "fit_shuffled_position", "score_base", "score_position", "score_shuffled_position"}
        for name, item in matrix_items.items():
            path = Path(item["path"])
            if not path.is_absolute():
                path = root / path
            if (_file_sha(path) != item["sha256"]
                    or _file_sha(path) != frozen_origin["matrix_file_hashes"].get(name)
                    or item["logical_sha256"] != frozen_origin["matrix_logical_hashes"].get(name)):
                raise AssertionError(f"E{origin} matrix byte hash mismatch: {name}")
            if path.suffix == ".npz":
                matrix: Any = _load_csr(path)
                logical_hash = _csr_sha(matrix)
            else:
                matrix = np.load(path, allow_pickle=False)
                logical_hash = _array_sha(matrix)
            if list(matrix.shape) != item["shape"] or logical_hash != item["logical_sha256"]:
                raise AssertionError(f"E{origin} matrix logical hash mismatch: {name}")
            if name in needed:
                matrices[name] = sparse.csr_matrix(matrix, dtype=np.float64)
        for component, frozen_state in frozen_origin["preprocessing"]["units"].items():
            unit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_{component}"
            accounting_row = accounting_by_id[unit_id]
            unit_path = pilot / "units" / f"{unit_id}.joblib"
            receipt_path = pilot / "receipts" / f"{unit_id}.json"
            if (_file_sha(unit_path) != frozen_state["artifact_sha256"]
                    or _file_sha(unit_path) != accounting_row["artifact_sha256"]
                    or _file_sha(receipt_path) != frozen_state["receipt_sha256"]
                    or _file_sha(receipt_path) != accounting_row["receipt_sha256"]):
                raise AssertionError(f"E{origin} {component} preprocessing artifact differs from amendment freeze")
        base_fit, base_score = matrices["fit_base"], matrices["score_base"]
        if _csr_sha(matrices["fit_position"][:, :base_fit.shape[1]]) != _csr_sha(base_fit):
            raise AssertionError(f"E{origin} real positional matrix base prefix mismatch")
        if _csr_sha(matrices["fit_shuffled_position"][:, :base_fit.shape[1]]) != _csr_sha(base_fit):
            raise AssertionError(f"E{origin} shuffled positional matrix base prefix mismatch")
        if _csr_sha(matrices["score_position"][:, :base_score.shape[1]]) != _csr_sha(base_score):
            raise AssertionError(f"E{origin} real score base prefix mismatch")
        if _csr_sha(matrices["score_shuffled_position"][:, :base_score.shape[1]]) != _csr_sha(base_score):
            raise AssertionError(f"E{origin} shuffled score base prefix mismatch")
        bag_path = pilot / "units" / f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR.joblib"
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        if (_file_sha(bag_path) != accounting_by_id[bag_id]["artifact_sha256"]
                or _file_sha(bag_path) != frozen_origin["bag_artifact_sha256"]
                or accounting_by_id[bag_id]["receipt_sha256"] != frozen_origin["bag_receipt_sha256"]):
            raise AssertionError(f"E{origin} saved bag artifact differs from pilot accounting")
        bag_unit = joblib.load(bag_path)
        bag = bag_unit["estimator"]
        if list(np.asarray(bag.classes_).tolist()) != [0, 1] or bag.coef_.shape[1] != base_fit.shape[1]:
            raise AssertionError(f"E{origin} saved bag feature/class order mismatch")
        zfit = np.asarray(bag.intercept_[0] + base_fit @ bag.coef_[0], dtype=np.float64).reshape(-1)
        zscore = np.asarray(bag.intercept_[0] + base_score @ bag.coef_[0], dtype=np.float64).reshape(-1)
        fit_ids = identity.iloc[fit_rows]["record_id"].astype(str).to_numpy()
        fit_y = _select_fit_labels(train_path, fit_rows, fit_ids)
        if not np.isin(fit_y, (0, 1)).all() or len(fit_ids) != len(zfit):
            raise AssertionError(f"E{origin} fit labels/rows invalid")
        if _array_sha(fit_y) != accounting_by_id[bag_id]["fit_population"]["fit_target_sha256"]:
            raise AssertionError(f"E{origin} prefix labels differ from saved pilot fit population")
        saved_bag_probability = np.asarray(bag_unit["probability"], dtype=np.float64)
        if (accounting_by_id[bag_id]["fit_metadata"]["coef_sha256"] != _array_sha(bag.coef_)
                or accounting_by_id[bag_id]["fit_metadata"]["intercept_sha256"] != _array_sha(bag.intercept_)
                or accounting_by_id[bag_id]["fit_metadata"]["probability_sha256"] != _array_sha(saved_bag_probability)
                or not np.allclose(saved_bag_probability, legacy["prob_bag"], rtol=0, atol=1e-14)):
            raise AssertionError(f"E{origin} saved C2_B coefficients/intercept/probabilities mismatch")
        fit_checks = {}
        for arm, fit_name, score_name in (
            ("REAL", "fit_position", "score_position"),
            ("SHUFFLED", "fit_shuffled_position", "score_shuffled_position"),
        ):
            fit_id = f"POS_BAG_OFFSET_V1_E{origin}_{arm}"
            artifact_path = run_root / "units" / f"{fit_id}.npz"
            receipt_path = run_root / "receipts" / f"{fit_id}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != freeze_data["freeze_signature"]
                    or receipt.get("artifact_sha256") != _file_sha(artifact_path)):
                raise AssertionError(f"E{origin} {arm} durable fit receipt mismatch")
            with np.load(artifact_path, allow_pickle=False) as saved:
                gamma = np.asarray(saved["gamma"], dtype=np.float64)
            xfit = matrices[fit_name][:, base_fit.shape[1]:].tocsr()
            xscore = matrices[score_name][:, base_score.shape[1]:].tocsr()
            if len(gamma) != xfit.shape[1] or receipt.get("gamma_sha256") != _array_sha(gamma):
                raise AssertionError(f"E{origin} {arm} gamma width/hash mismatch")
            if receipt["fit_population"]["fit_ids_sha256"] != _array_sha(fit_ids):
                raise AssertionError(f"E{origin} {arm} fit IDs mismatch")
            if receipt["fit_population"]["fit_target_sha256"] != _array_sha(fit_y):
                raise AssertionError(f"E{origin} {arm} fit label hash mismatch")
            loss, grad = _objective_gradient(xfit, zfit, fit_y, gamma, c=1.0)
            zero_loss, _ = _objective_gradient(xfit, zfit, fit_y, np.zeros_like(gamma), c=1.0)
            optimizer = receipt["optimizer"]
            ginf = float(np.max(np.abs(grad))) if len(grad) else 0.0
            if (optimizer.get("success") is not True or not np.isfinite(loss)
                    or not np.isfinite(grad).all() or loss > zero_loss + 1e-8
                    or not np.isclose(loss, optimizer["loss_final"], rtol=0, atol=1e-7)
                    or not np.isclose(zero_loss, optimizer["loss_gamma0"], rtol=0, atol=1e-7)
                    or not np.isclose(ginf, optimizer["gradient_inf_norm"], rtol=1e-7, atol=1e-10)):
                raise AssertionError(f"E{origin} {arm} independent objective/gradient receipt mismatch")
            raw = expit(zscore + np.asarray(xscore @ gamma).reshape(-1))
            routed = raw.copy()
            missing = ~np.asarray(output["present_mask"], dtype=bool)
            fallback = np.asarray(legacy["prob_missing_hgb"], dtype=np.float64)
            routed[missing] = fallback[missing]
            saved_name = "prob_offset_real" if arm == "REAL" else "prob_offset_shuffled"
            if not np.allclose(routed, output[saved_name], rtol=0, atol=1e-13):
                raise AssertionError(f"E{origin} {arm} prediction does not reproduce frozen offset formula")
            if not np.array_equal(output[saved_name][missing], fallback[missing]):
                raise AssertionError(f"E{origin} {arm} missing/blank HGB fallback is not exact")
            fit_checks[arm] = {
                "fit_id": fit_id, "receipt_sha256": _file_sha(receipt_path),
                "gamma_sha256": receipt["gamma_sha256"], "loss": loss,
                "zero_gamma_loss": zero_loss, "gradient_inf_norm": ginf,
                "optimizer_called": False,
            }
        if not np.array_equal(output["prob_bag"][~output["present_mask"]], legacy["prob_route_bag"][~legacy["present_mask"]]):
            raise AssertionError(f"E{origin} saved bag fallback differs on missing/blank rows")
        hgb_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_MISSING_HGB"
        hgb_path = pilot / "units" / f"{hgb_id}.joblib"
        if (_file_sha(hgb_path) != frozen_origin["missing_hgb_artifact_sha256"]
                or accounting_by_id[hgb_id]["receipt_sha256"] != frozen_origin["missing_hgb_receipt_sha256"]):
            raise AssertionError(f"E{origin} saved missing-stream HGB differs from amendment freeze")
        hgb_unit = joblib.load(hgb_path)
        fallback = np.asarray(hgb_unit["probability"], dtype=np.float64)
        if (accounting_by_id[hgb_id]["fit_metadata"]["probability_sha256"] != _array_sha(fallback)
                or not np.array_equal(fallback, legacy["prob_missing_hgb"])):
            raise AssertionError(f"E{origin} exact saved HGB probabilities differ from the pilot bundle")
        origins[key] = {
            "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
            "score_rows_sha256": _array_sha(score_rows), "fits": fit_checks,
            "query_labels_read": False,
        }
    return {"status": "PASS", "freeze_signature": freeze_data["freeze_signature"],
            "origins": origins, "query_labels_read": False,
            "optimizer_refits": 0, "preprocessing_fits": 0}
