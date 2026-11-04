"""Independent prelabel and post-score verifier for POS_CONTRAST_V1.

Prelabel verification reads only official Train identity columns and fit-prefix
labels. The mixed A02 cache contains a historical query-label field; this
verifier selects only explicitly named prediction and identity arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

_ROOT_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_IMPORT))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)

from scripts import optimization_v3_positional_bag_offset_v1 as offset


CARD_ID = "POS_CONTRAST_V1"
FIT_IDS = (
    "POS_CONTRAST_V1_E67_REAL", "POS_CONTRAST_V1_E67_SHUFFLED",
    "POS_CONTRAST_V1_E75_REAL", "POS_CONTRAST_V1_E75_SHUFFLED",
)
E79_ID = "POS_CONTRAST_V1_E79_REAL"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
PILOT_ROOT_REL = Path("experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1")
BAG_ROOT_REL = Path("experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1")
A02_ROOT_REL = Path("experiments/runs/optimization_v3_a02_neighbors_v1")
REPORT_ROOT_REL = Path("reports/optimization_v3/positional_contrast_v1")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object at {path}")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _independent_contrast(augmented: sparse.spmatrix, base_width: int) -> sparse.csr_matrix:
    x = sparse.csr_matrix(augmented, dtype=np.float64)
    position_width = x.shape[1] - int(base_width)
    if base_width < 0 or position_width < 0 or position_width % 3:
        raise AssertionError("invalid augmented three-block feature matrix")
    width = position_width // 3
    f = x[:, base_width:base_width + width]
    m = x[:, base_width + width:base_width + 2 * width]
    l = x[:, base_width + 2 * width:]
    projected = sparse.hstack(((f - l) / np.sqrt(2.0),
                               (f - 2.0 * m + l) / np.sqrt(6.0)), format="csr")
    projected.sum_duplicates()
    projected.eliminate_zeros()
    projected.sort_indices()
    return projected


def _safe_a02(root: Path, origin: int, *, groups: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    folder = root / A02_ROOT_REL / f"E{origin}" / "observed"
    meta = _read(folder / "predictions.json")
    receipt = _read(folder / "fit_receipt.json")
    keys = ["score_rows", "score_record_ids", "score_cycles", "prediction", "missing_mask"]
    if groups:
        keys.extend(["entry_friction", "text_load_group"])
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in keys}
    if (meta.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or meta.get("query_labels_passed_to_prediction") is not False
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False):
        raise AssertionError(f"A02 E{origin} source-boundary receipt mismatch")
    expected = meta["diagnostics"]["candidate_probability_sha256"]
    prediction_sha = hashlib.sha256(np.asarray(arrays["prediction"], dtype="<f8").tobytes(order="C")).hexdigest()
    rows_sha = hashlib.sha256(np.asarray(arrays["score_rows"], dtype="<i8").tobytes(order="C")).hexdigest()
    ids_payload = json.dumps(arrays["score_record_ids"].astype(str).tolist(), separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
    ids_sha = hashlib.sha256(ids_payload).hexdigest()
    if prediction_sha != expected:
        raise AssertionError(f"A02 E{origin} candidate probability array differs")
    if rows_sha != meta["score_row_positions_sha256"]:
        raise AssertionError(f"A02 E{origin} row-index array differs")
    if ids_sha != meta["score_record_ids_sha256"]:
        raise AssertionError(f"A02 E{origin} row-ID array differs")
    return {"meta": meta, "receipt": receipt}, arrays


def independent_metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) != len(p) or not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise AssertionError("metric inputs are invalid")
    if len(y) == 0:
        return {"n": 0, "positive": 0, "negative": 0, "macro_f1": None,
                "positive_f1": None, "precision": None, "positive_recall": None,
                "roc_auc": None, "average_precision": None, "log_loss": None,
                "brier": None, "confusion_matrix": [[0, 0], [0, 0]]}
    pred = (p >= 0.5).astype(np.int8)
    binary = np.unique(y).size == 2
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if binary else None,
        "average_precision": float(average_precision_score(y, p)) if binary else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def _verify_receipt_binding(receipt: Mapping[str, Any], *, fit_id: str,
                            freeze_signature: str, artifact_sha256: str) -> None:
    if (receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze_signature
            or receipt.get("artifact_sha256") != artifact_sha256):
        raise AssertionError(f"fit receipt binding failure for {fit_id}")


def verify_receipt_binding(receipt: Mapping[str, Any], *, fit_id: str,
                           freeze_signature: str, artifact_sha256: str) -> None:
    """Public corruption guard exercised by production prelabel QA and tests."""
    _verify_receipt_binding(receipt, fit_id=fit_id, freeze_signature=freeze_signature,
                            artifact_sha256=artifact_sha256)


def assert_ids_match(actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(np.asarray(actual).astype(str), np.asarray(expected).astype(str)):
        raise AssertionError("record_id alignment mismatch")


def verify_file_hash(path: Path, expected_sha256: str) -> None:
    if offset.sha256_file(Path(path)) != expected_sha256:
        raise AssertionError("bundle hash mismatch")


def _read_fit_labels(train_path: Path, fit_rows: np.ndarray, fit_ids: np.ndarray,
                     expected_target_sha: str) -> np.ndarray:
    # Only selected prefix rows' target cells are converted/used. Query target
    # fields are never requested from the CSV row objects.
    import csv
    rows = np.asarray(fit_rows, dtype=np.int64)
    ids = np.asarray(fit_ids).astype(str)
    positions = {int(row): i for i, row in enumerate(rows.tolist())}
    y = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            out = positions.get(row_index)
            if out is None:
                continue
            if record[id_col] != ids[out]:
                raise AssertionError("fit-prefix label ID mismatch")
            y[out] = int(record[target_col])
            seen[out] = True
    if not seen.all() or not np.isin(y, (0, 1)).all() or offset.sha256_array(y) != expected_target_sha:
        raise AssertionError("fit-prefix target hash mismatch")
    return y


def _parent_components(root: Path, origin: int, parent_freeze: Mapping[str, Any]):
    # Use the already frozen and independently verified POS_BAG matrix/model
    # loader. The contrast calculation and objective below are reconstructed
    # independently from its raw saved augmented matrices.
    return offset._load_origin_payload(origin, parent_freeze)


def _verify_source_files(root: Path, freeze: Mapping[str, Any]) -> None:
    source_paths = {
        "runner": root / "scripts/optimization_v3_positional_contrast_v1.py",
        "verifier": root / "scripts/verify_optimization_v3_positional_contrast_v1.py",
        "contrast_model": root / "src/models/positional_contrast_v1.py",
        "offset_model": root / "src/models/positional_bag_offset_v1.py",
        "offset_runner": root / "scripts/optimization_v3_positional_bag_offset_v1.py",
        "contrast_tests": root / "tests/test_positional_contrast_v1.py",
        "pilot_tests": root / "tests/test_optimization_v3_positional_contrast_v1.py",
        "recipe": root / "reports/optimization_v3/positional_contrast_v1/RECIPE.json",
        "protocol": root / "reports/optimization_v3/posmix_followup_v1/SOL_NEXT_RESEARCH.md",
    }
    current = {key: offset.sha256_file(path) for key, path in source_paths.items()}
    if current != freeze["source_hashes"]:
        raise AssertionError("frozen source hashes changed")
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", str(freeze["source_commit"]), "HEAD"],
                        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if rc.returncode != 0:
        raise AssertionError("frozen implementation commit is not an ancestor of current HEAD")
    if freeze.get("schema_version") == 2:
        report_root = root / REPORT_ROOT_REL
        base = _read(report_root / "FREEZE.json")
        base_body = {key: value for key, value in base.items() if key != "freeze_signature"}
        if base.get("freeze_signature") != _sha_obj(base_body):
            raise AssertionError("prefit successor base freeze signature mismatch")
        if (freeze.get("prior_freeze_signature") != base.get("freeze_signature")
                or freeze.get("prior_source_commit") != base.get("source_commit")):
            raise AssertionError("prefit successor does not identify its exact predecessor")
        attempt = freeze.get("prefit_dispatch_attempt", {})
        if (freeze.get("prefit_amendment_id") != "POS_CONTRAST_V1_PREFIT_SOURCE_REBIND_1"
                or attempt.get("attempt_source_commit") != "ba79ec30fd6a8b17f66cfde8186e79282ca1a494"
                or attempt.get("result") != "PRE_DISPATCH_REJECTED_ZERO_FITS"
                or attempt.get("error") != "TypeError: _check_ledger() got an unexpected keyword argument 'freeze_signature'"
                or attempt.get("estimator_fit_calls") != 0 or attempt.get("fit_started_ids") != []
                or attempt.get("terminal_fit_ids") != []
                or attempt.get("ledger_sha256_after_attempt") != freeze.get("reservation_checkpoint_ledger_sha256")):
            raise AssertionError("prefit successor lacks exact zero-fit recovery evidence")
        for key, value in base.items():
            if key in {"schema_version", "source_commit", "source_hashes", "freeze_signature"}:
                continue
            if freeze.get(key) != value:
                raise AssertionError(f"prefit successor altered frozen recipe/input field {key}")
        record = _read(report_root / "PREFIT_AMENDMENT.json")
        event = record.get("event")
        if not isinstance(event, dict) or record.get("event_sha256") != _sha_obj(event):
            raise AssertionError("prefit rebind report is missing or has an invalid event hash")
        events = offset._ledger_events(root / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl")
        ledger_bytes = (root / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl").read_bytes()
        prefix_size = int(freeze["reservation_checkpoint_ledger_bytes"])
        prefix = ledger_bytes[:prefix_size]
        first_tail_line = ledger_bytes[prefix_size:].splitlines()[0]
        if (hashlib.sha256(prefix).hexdigest() != freeze["reservation_checkpoint_ledger_sha256"]
                or json.loads(first_tail_line.decode("utf-8")) != event):
            raise AssertionError("prefit source rebind is not immediately appended after the frozen reservation checkpoint")
        rebinds = [item for item in events if item.get("event") == "PREFIT_SOURCE_REBIND"
                   and item.get("card_id") == CARD_ID]
        if len(rebinds) != 1 or rebinds[0] != event:
            raise AssertionError("append-only prefit source rebind differs from its report")
        required = {
            "prior_freeze_signature": base["freeze_signature"],
            "freeze_signature": freeze["freeze_signature"],
            "prior_source_commit": base["source_commit"],
            "source_commit": freeze["source_commit"],
            "fit_ids": list(FIT_IDS), "conditional_fit_id": E79_ID,
            "reservation_keys": ["POS_CONTRAST_V1:PILOT_4", "POS_CONTRAST_V1:CONDITIONAL_E79_AFTER_SOL_REVIEW"],
            "added_fit_slots": 0, "estimator_fit_calls_before_rebind": 0,
            "fit_starts_before_rebind": [], "terminal_fit_ids_before_rebind": [],
            "ledger_sha256_before_rebind": freeze["reservation_checkpoint_ledger_sha256"],
            "ledger_snapshot_before_rebind": freeze["reservation_checkpoint_snapshot"],
            "caps_unchanged": True, "recipe_gates_populations_unchanged": True,
        }
        if any(event.get(key) != value for key, value in required.items()):
            raise AssertionError("prefit rebind changed authorization or lacks zero-start evidence")
        if record.get("status") != "PASS_REBOUND_NO_NEW_SLOTS":
            raise AssertionError("prefit rebind report status is not PASS")
        amendments = [item for item in events if item.get("event") == "BUDGET_CAP_AMENDMENT"
                      and item.get("amendment_id") == "POS_CONTRAST_V1_CAP137_2026_11_04"]
        reservations = [item for item in events if item.get("event") == "CARD_FITS_RESERVED"
                        and item.get("reservation_key") in required["reservation_keys"]]
        if (len(amendments) != 1 or len(reservations) != 2
                or amendments[0].get("freeze_signature") != base["freeze_signature"]
                or any(item.get("freeze_signature") != base["freeze_signature"] for item in reservations)):
            raise AssertionError("original budget reservations are not preserved by the rebind")


def verify_prelabel_outputs(root: Path, run_root: Path,
                            freeze_data: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    run_root = Path(run_root).resolve()
    train_path = root / "data/raw/classification/Train.csv"
    if offset.sha256_file(train_path) != EXPECTED_TRAIN_SHA:
        raise AssertionError("Train source hash mismatch")
    body = {k: v for k, v in freeze_data.items() if k != "freeze_signature"}
    if freeze_data.get("freeze_signature") != _sha_obj(body):
        raise AssertionError("freeze signature mismatch")
    _verify_source_files(root, freeze_data)
    ready_path = run_root / "PREDICTIONS_READY.json"
    ready = _read(ready_path)
    if (ready.get("freeze_signature") != freeze_data["freeze_signature"]
            or ready.get("query_labels_opened") is not False
            or ready.get("test_or_sample_access") is not False
            or set(ready.get("predictions", {})) != {"E67", "E75"}):
        raise AssertionError("prediction-ready manifest is not closed and freeze-bound")
    ledger_path = root / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
    events = offset._ledger_events(ledger_path)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    if len(starts) != 4 or len(saved) != 4 or failed or {e.get("fit_id") for e in starts} != set(FIT_IDS):
        raise AssertionError("all four exact fits must be durably saved before prelabel QA")
    parent_freeze = _read(root / BAG_ROOT_REL / "FREEZE.json")
    if parent_freeze.get("freeze_signature") != freeze_data["parent_freeze_signature"]:
        raise AssertionError("POS_BAG parent freeze changed")
    bag_prelabel = _read(root / BAG_ROOT_REL / "PRELABEL_VERIFICATION.json")
    if bag_prelabel.get("status") != "PASS" or bag_prelabel.get("query_labels_read") is not False:
        raise AssertionError("historical pP prelabel boundary is not valid")
    identity = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    if len(identity) != 50608 or identity["record_id"].isna().any() or identity["record_id"].duplicated().any():
        raise AssertionError("official Train identity frame is invalid")
    origins_verified: dict[str, Any] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        frozen_origin = freeze_data["origins"][key]
        manifest, legacy, bag, payload = _parent_components(root, origin, parent_freeze)
        matrices = payload["matrices"]
        base_fit = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
        base_score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
        fit_rows = np.asarray(legacy["fit_rows"], dtype=np.int64)
        score_rows = np.asarray(legacy["score_rows"], dtype=np.int64)
        if (offset.sha256_array(fit_rows) != frozen_origin["fit_rows_sha256"]
                or offset.sha256_array(score_rows) != frozen_origin["score_rows_sha256"]):
            raise AssertionError(f"E{origin} frozen fit/query row positions changed")
        if len(fit_rows) != (29705 if origin == 67 else 42117) or len(score_rows) != (20903 if origin == 67 else 8491):
            raise AssertionError(f"E{origin} fit/query population size mismatch")
        expected_fit_cycle = identity.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64)
        expected_score_cycle = identity.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
        fit_hi = 67 if origin == 67 else 75
        if not np.all((expected_fit_cycle >= 47) & (expected_fit_cycle <= fit_hi)):
            raise AssertionError(f"E{origin} fit cycle boundary mismatch")
        if not np.all((expected_score_cycle >= (68 if origin == 67 else 76)) & (expected_score_cycle <= 79)):
            raise AssertionError(f"E{origin} query cycle boundary mismatch")
        ids = identity.iloc[score_rows]["record_id"].astype(str).to_numpy()
        a02_info, a02 = _safe_a02(root, origin)
        expected_a02_hashes = freeze_data["parent_verification"]["origins"][key]["a02_safe_array_hashes"]
        if {name: offset.sha256_array(value) for name, value in a02.items()} != expected_a02_hashes:
            raise AssertionError(f"E{origin} A02 safe arrays changed after freeze")
        if not np.array_equal(a02["score_rows"], score_rows):
            raise AssertionError(f"E{origin} A02 row positions differ")
        if not np.array_equal(a02["score_record_ids"].astype(str), ids):
            raise AssertionError(f"E{origin} A02 IDs differ from Train query rows")
        if not np.array_equal(a02["score_cycles"].astype(np.int64), expected_score_cycle):
            raise AssertionError(f"E{origin} A02 cycles differ from Train query rows")
        old_pred_path = root / BAG_ROOT_REL / key / f"predictions_E{origin}.npz"
        old_ready = _read(root / BAG_ROOT_REL / "PREDICTIONS_READY.json")
        if offset.sha256_file(old_pred_path) != old_ready["predictions"][key]["sha256"]:
            raise AssertionError(f"E{origin} original POS_BAG prediction hash mismatch")
        with np.load(old_pred_path, allow_pickle=False) as a:
            old = {k: np.asarray(a[k]).copy() for k in ["fit_rows", "score_rows", "score_record_ids", "score_cycles",
                                                          "present_mask", "prob_offset_real", "prob_missing_hgb", "prob_bag"]}
        if not np.array_equal(old["score_rows"], score_rows) or not np.array_equal(old["score_record_ids"].astype(str), ids):
            raise AssertionError(f"E{origin} pP row/ID lineage mismatch")
        present = np.asarray(old["present_mask"], dtype=bool)
        if not np.array_equal(a02["missing_mask"], ~present):
            raise AssertionError(f"E{origin} A02/POS availability mismatch")
        if not np.array_equal(old["prob_offset_real"][~present], old["prob_missing_hgb"][~present]):
            raise AssertionError(f"E{origin} exact original pP missing fallback mismatch")
        out_path = run_root / key / f"predictions_E{origin}.npz"
        entry = ready["predictions"][key]
        if offset.sha256_file(out_path) != entry["sha256"]:
            raise AssertionError(f"E{origin} candidate prediction bundle hash mismatch")
        with np.load(out_path, allow_pickle=False) as a:
            output = {k: np.asarray(a[k]).copy() for k in a.files}
        required = {"origin", "fit_rows", "score_rows", "score_record_ids", "score_cycles", "present_mask",
                    "prob_q_real", "prob_q_shuffled", "prob_pP", "prob_a02", "prob_mix50",
                    "prob_real_blend", "prob_shuffled_blend", "prob_bag"}
        if set(output) != required | {"matrix_manifest_sha256", "a02_safe_probability_sha256"}:
            raise AssertionError(f"E{origin} candidate output keys do not match the frozen contract")
        if (not np.array_equal(output["fit_rows"], fit_rows)
                or not np.array_equal(output["score_rows"], score_rows)
                or not np.array_equal(output["score_record_ids"].astype(str), ids)
                or not np.array_equal(output["score_cycles"].astype(np.int64), expected_score_cycle)
                or not np.array_equal(output["present_mask"], present)):
            raise AssertionError(f"E{origin} prediction identity/availability alignment failed")
        base_width = base_fit.shape[1]
        arm_checks: dict[str, Any] = {}
        for arm, matrix_name in (("REAL", "position"), ("SHUFFLED", "shuffled_position")):
            x_fit_aug = sparse.csr_matrix(matrices[f"fit_{matrix_name}"], dtype=np.float64)
            x_score_aug = sparse.csr_matrix(matrices[f"score_{matrix_name}"], dtype=np.float64)
            z_fit = _independent_contrast(x_fit_aug, base_width)
            z_score = _independent_contrast(x_score_aug, base_width)
            mf = frozen_origin["contrast_matrices"][f"fit_{matrix_name}"]
            ms = frozen_origin["contrast_matrices"][f"score_{matrix_name}"]
            if (offset.sha256_csr(z_fit) != mf["sha256"] or list(z_fit.shape) != mf["shape"]
                    or offset.sha256_csr(z_score) != ms["sha256"] or list(z_score.shape) != ms["shape"]):
                raise AssertionError(f"E{origin} {arm} independent contrast reconstruction mismatch")
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            artifact = run_root / "units" / f"{fit_id}.npz"
            receipt_path = run_root / "receipts" / f"{fit_id}.json"
            receipt = _read(receipt_path)
            artifact_sha = offset.sha256_file(artifact)
            verify_receipt_binding(receipt, fit_id=fit_id, freeze_signature=str(freeze_data["freeze_signature"]),
                                   artifact_sha256=artifact_sha)
            if (receipt.get("source_commit") != freeze_data["source_commit"]
                    or receipt.get("source_hashes") != freeze_data["source_hashes"]
                    or receipt.get("input_sha256") != freeze_data["input_sha256"]
                    or receipt.get("origin") != origin or receipt.get("arm") != arm
                    or receipt.get("fit_population", {}).get("target_read_purpose") != "fit-prefix-only"):
                raise AssertionError(f"E{origin} {arm} source/population receipt mismatch")
            with np.load(artifact, allow_pickle=False) as a:
                theta = np.asarray(a["gamma"], dtype=np.float64).copy()
            if offset.sha256_array(theta) != receipt.get("gamma_sha256") or len(theta) != z_fit.shape[1]:
                raise AssertionError(f"E{origin} {arm} theta shape/hash mismatch")
            bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
            bag_acc = {v["fit_id"]: v for v in _read(root / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json")["fit_units"]}[bag_id]
            fit_ids = identity.iloc[fit_rows]["record_id"].astype(str).to_numpy()
            yfit = _read_fit_labels(train_path, fit_rows, fit_ids, bag_acc["fit_population"]["fit_target_sha256"])
            z0_fit = np.asarray(bag.intercept_[0] + base_fit @ bag.coef_[0], dtype=np.float64).reshape(-1)
            z0_score = np.asarray(bag.intercept_[0] + base_score @ bag.coef_[0], dtype=np.float64).reshape(-1)
            logits = z0_fit + np.asarray(z_fit @ theta, dtype=np.float64).reshape(-1)
            signed_y = 2.0 * yfit.astype(np.float64) - 1.0
            objective = float(np.logaddexp(0.0, -signed_y * logits).sum() + 0.5 * np.dot(theta, theta))
            gradient = np.asarray(z_fit.T @ (expit(logits) - yfit) + theta, dtype=np.float64).reshape(-1)
            optimizer = receipt.get("optimizer", {})
            if (not optimizer.get("success") or not np.isfinite(objective)
                    or objective > float(optimizer.get("loss_gamma0", np.inf)) + 1e-8
                    or abs(objective - float(optimizer.get("loss_final", np.inf))) > 1e-6
                    or np.max(np.abs(gradient), initial=0.0) > max(1.1e-6, float(optimizer.get("gradient_inf_norm", 0.0)) * 1.05)):
                raise AssertionError(f"E{origin} {arm} independent objective/gradient verification failed")
            q = expit(z0_score + np.asarray(z_score @ theta, dtype=np.float64).reshape(-1))
            saved_q = output["prob_q_real" if arm == "REAL" else "prob_q_shuffled"]
            if not np.allclose(q, saved_q, rtol=0, atol=1e-12):
                raise AssertionError(f"E{origin} {arm} q probability replay mismatch")
            arm_checks[arm] = {"fit_id": fit_id, "theta_sha256": offset.sha256_array(theta),
                               "fit_contrast_sha256": offset.sha256_csr(z_fit),
                               "score_contrast_sha256": offset.sha256_csr(z_score),
                               "objective": objective, "gradient_inf_norm": float(np.max(np.abs(gradient), initial=0.0)),
                               "optimizer_called": False}
        p_p = np.asarray(old["prob_offset_real"], dtype=np.float64)
        p_a = np.asarray(a02["prediction"], dtype=np.float64)
        expected_mix = p_p.copy()
        expected_real = p_p.copy()
        expected_shuf = p_p.copy()
        expected_mix[present] = 0.5 * p_p[present] + 0.5 * p_a[present]
        expected_real[present] = 0.5 * output["prob_q_real"][present] + 0.5 * p_a[present]
        expected_shuf[present] = 0.5 * output["prob_q_shuffled"][present] + 0.5 * p_a[present]
        formulas = (("prob_pP", p_p), ("prob_a02", p_a), ("prob_mix50", expected_mix),
                    ("prob_real_blend", expected_real), ("prob_shuffled_blend", expected_shuf),
                    ("prob_bag", old["prob_bag"]))
        for name, expected in formulas:
            if not np.allclose(output[name], expected, rtol=0, atol=1e-12):
                raise AssertionError(f"E{origin} {name} fixed formula mismatch")
        if (not np.array_equal(output["prob_mix50"][~present], p_p[~present])
                or not np.array_equal(output["prob_real_blend"][~present], p_p[~present])
                or not np.array_equal(output["prob_shuffled_blend"][~present], p_p[~present])):
            raise AssertionError(f"E{origin} fixed missing route is not exact original pP")
        if not np.array_equal(p_a, a02["prediction"]):
            raise AssertionError(f"E{origin} A02 safe probability values differ after metadata lineage check")
        origins_verified[key] = {
            "score_rows": int(len(score_rows)), "score_rows_sha256": offset.sha256_array(score_rows),
            "score_record_ids_sha256": offset.sha256_array(ids), "query_labels_read": False,
            "missing_route_exact": True, "fixed_formula_reconstructed": True,
            "arms": arm_checks,
        }
    result = {
        "status": "PASS", "freeze_signature": freeze_data["freeze_signature"],
        "predictions_ready_sha256": offset.sha256_file(ready_path),
        "prediction_output_sha256s": {k: ready["predictions"][k]["sha256"] for k in ("E67", "E75")},
        "origins": origins_verified, "fit_count": 4, "optimizer_refits": 0,
        "query_labels_read": False, "test_or_sample_access": False,
        "checks": ["train/input identity", "source and freeze hashes", "four durable fit receipts",
                   "independent contrast matrices", "objective, gradient and L2 penalty",
                   "q and fixed blend formula", "exact pP missing route", "A02 safe probability lineage"],
    }
    result["verification_signature"] = _sha_obj(result)
    return result


def verify_scored_results(root: Path, run_root: Path,
                          freeze_data: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    run_root = Path(run_root).resolve()
    prelabel = _read(run_root / "PRELABEL_VERIFICATION.json")
    ready = _read(run_root / "PREDICTIONS_READY.json")
    if (prelabel.get("status") != "PASS" or prelabel.get("query_labels_read") is not False
            or prelabel.get("predictions_ready_sha256") != offset.sha256_file(run_root / "PREDICTIONS_READY.json")):
        raise AssertionError("scoring began without current independent prelabel PASS")
    if result.get("freeze_signature") != freeze_data.get("freeze_signature") or result.get("query_labels_opened") is not True:
        raise AssertionError("score result freeze/query-label marker mismatch")
    train_path = root / "data/raw/classification/Train.csv"
    # This label read is deliberately after durable independent prelabel QA.
    frame = pd.read_csv(train_path, usecols=["record_id", "cycle_code", "artifact_signal"], dtype={"record_id": "string"})
    origins_replayed: dict[str, Any] = {}
    gate_replay: list[dict[str, Any]] = []
    for origin in (67, 75):
        key = f"E{origin}"
        out_path = run_root / key / f"predictions_E{origin}.npz"
        if offset.sha256_file(out_path) != ready["predictions"][key]["sha256"]:
            raise AssertionError(f"E{origin} scored prediction bundle changed")
        with np.load(out_path, allow_pickle=False) as z:
            pred = {name: np.asarray(z[name]).copy() for name in z.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        sub = frame.iloc[rows]
        ids = sub["record_id"].astype(str).to_numpy()
        if not np.array_equal(ids, pred["score_record_ids"].astype(str)):
            raise AssertionError(f"E{origin} score labels joined to the wrong IDs")
        y = pd.to_numeric(sub["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        cycles = pd.to_numeric(sub["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        _, a02 = _safe_a02(root, origin, groups=True)
        views = ({"H12_E67": (cycles >= 68) & (cycles <= 79),
                  "W4": (cycles >= 68) & (cycles <= 71),
                  "W5": (cycles >= 72) & (cycles <= 75),
                  "W6": (cycles >= 76) & (cycles <= 79)} if origin == 67
                 else {"W6": (cycles >= 76) & (cycles <= 79)})
        policies = {"contrast_real": pred["prob_real_blend"], "contrast_shuffled": pred["prob_shuffled_blend"],
                    "MIX50": pred["prob_mix50"], "pP_oldPOS": pred["prob_pP"], "bag": pred["prob_bag"]}
        view_replayed: dict[str, Any] = {}
        subgroup_replayed: list[dict[str, Any]] = []
        cycle_replayed: list[dict[str, Any]] = []
        for view, mask in views.items():
            metrics = {name: independent_metric(y[mask], np.asarray(prob)[mask]) for name, prob in policies.items()}
            present_view = mask & np.asarray(pred["present_mask"], dtype=bool)
            metrics["contrast_q_real_present"] = independent_metric(y[present_view], pred["prob_q_real"][present_view])
            metrics["contrast_q_shuffled_present"] = independent_metric(y[present_view], pred["prob_q_shuffled"][present_view])
            metrics["A02_observed_present"] = independent_metric(y[present_view], pred["prob_a02"][present_view])
            stored = result["origins"][key]["views"][view]["metrics"]
            for policy, metric in metrics.items():
                for metric_name, value in metric.items():
                    saved_value = stored[policy][metric_name]
                    if isinstance(value, float):
                        if not np.isclose(value, float(saved_value), rtol=0, atol=1e-12):
                            raise AssertionError(f"{key}/{view}/{policy}/{metric_name} metric mismatch")
                    elif value != saved_value:
                        raise AssertionError(f"{key}/{view}/{policy}/{metric_name} support/count mismatch")
            deltas = {
                "real_minus_mix50": metrics["contrast_real"]["macro_f1"] - metrics["MIX50"]["macro_f1"],
                "shuffled_minus_mix50": metrics["contrast_shuffled"]["macro_f1"] - metrics["MIX50"]["macro_f1"],
                "real_minus_shuffled": metrics["contrast_real"]["macro_f1"] - metrics["contrast_shuffled"]["macro_f1"],
            }
            if any(not np.isclose(deltas[name], result["origins"][key]["views"][view]["deltas"][name], rtol=0, atol=1e-12) for name in deltas):
                raise AssertionError(f"{key}/{view} policy deltas mismatch")
            view_replayed[view] = {"metrics": metrics, "deltas": deltas}
            if origin == 67 and view in {"W4", "W5", "W6"}:
                gate_replay.append({"gate": f"E67_{view}_harm_vs_MIX50_max_0.003", "delta": deltas["real_minus_mix50"],
                                    "limit": -0.003, "pass": bool(deltas["real_minus_mix50"] >= -0.003)})
            if (origin == 67 and view == "H12_E67") or (origin == 75 and view == "W6"):
                gate_replay.extend([
                    {"gate": f"E{origin}_{view}_real_gain_vs_MIX50_min_0.0005", "delta": deltas["real_minus_mix50"],
                     "limit": 0.0005, "pass": bool(deltas["real_minus_mix50"] >= 0.0005)},
                    {"gate": f"E{origin}_{view}_real_gain_vs_shuffled_min_0", "delta": deltas["real_minus_shuffled"],
                     "limit": 0.0, "pass": bool(deltas["real_minus_shuffled"] >= 0.0)},
                ])
            groups = [("availability_present", pred["present_mask"]), ("availability_missing", ~pred["present_mask"]),
                      ("entry_friction_0", a02["entry_friction"] == 0), ("entry_friction_1", a02["entry_friction"] == 1)]
            groups.extend((f"text_load_Q{i+1}", a02["text_load_group"] == i) for i in range(4))
            for group, group_mask in groups:
                use = mask & np.asarray(group_mask, dtype=bool)
                n = int(use.sum())
                pos = int(y[use].sum())
                neg = n - pos
                supported = n >= 200 and pos >= 30 and neg >= 30
                saved_rows = [row for row in result["origins"][key]["subgroups"] if row["view"] == view and row["group"] == group]
                if len(saved_rows) != 1:
                    raise AssertionError(f"{key}/{view}/{group} subgroup missing or duplicated")
                saved = saved_rows[0]
                if supported != (saved["status"] == "SUPPORTED") or (n, pos, neg) != (saved["n"], saved["positive"], saved["negative"]):
                    raise AssertionError(f"{key}/{view}/{group} subgroup support mismatch")
                if supported:
                    mr = independent_metric(y[use], pred["prob_real_blend"][use])
                    mb = independent_metric(y[use], pred["prob_mix50"][use])
                    df1 = mr["macro_f1"] - mb["macro_f1"]
                    dr = mr["positive_recall"] - mb["positive_recall"]
                    if not np.isclose(df1, saved["delta_macro_f1"], rtol=0, atol=1e-12) or not np.isclose(dr, saved["delta_positive_recall"], rtol=0, atol=1e-12):
                        raise AssertionError(f"{key}/{view}/{group} subgroup metric mismatch")
                    gate_replay.append({"gate": f"E{origin}_{view}_{group}_subgroup_guard", "delta_macro_f1": df1,
                                        "delta_positive_recall": dr, "support": n, "positive": pos, "negative": neg,
                                        "pass": bool(df1 >= -0.01 and dr >= -0.02)})
                subgroup_replayed.append({"view": view, "group": group, "support": n, "status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED"})
        for cycle in sorted(np.unique(cycles)):
            cm = cycles == cycle
            mr = independent_metric(y[cm], pred["prob_real_blend"][cm])
            mb = independent_metric(y[cm], pred["prob_mix50"][cm])
            cycle_replayed.append({"cycle": int(cycle), "n": int(cm.sum()),
                                   "real_macro_f1": mr["macro_f1"], "mix50_macro_f1": mb["macro_f1"],
                                   "delta": mr["macro_f1"] - mb["macro_f1"]})
        stored_cycles = result["origins"][key]["per_cycle"]
        if len(stored_cycles) != len(cycle_replayed):
            raise AssertionError(f"{key} per-cycle result count mismatch")
        for expected, stored in zip(cycle_replayed, stored_cycles):
            if expected["cycle"] != stored.get("cycle") or expected["n"] != stored.get("n"):
                raise AssertionError(f"{key} per-cycle identity/support mismatch at cycle {expected['cycle']}")
            for field in ("real_macro_f1", "mix50_macro_f1", "delta"):
                if not np.isclose(expected[field], float(stored[field]), rtol=0, atol=1e-12):
                    raise AssertionError(f"{key} per-cycle {field} mismatch at cycle {expected['cycle']}")
        origins_replayed[key] = {"views": view_replayed, "subgroups": subgroup_replayed, "per_cycle": cycle_replayed}
    if len(gate_replay) != len(result["gate_checks"]):
        raise AssertionError("independent gate reconstruction count mismatch")
    for expected, stored in zip(gate_replay, result["gate_checks"]):
        if expected["gate"] != stored["gate"] or expected["pass"] != stored.get("pass"):
            raise AssertionError(f"independent gate mismatch: {expected['gate']}")
        for field in ("delta", "delta_macro_f1", "delta_positive_recall", "limit"):
            if field in expected and not np.isclose(float(expected[field]), float(stored[field]), rtol=0, atol=1e-12):
                raise AssertionError(f"independent gate numeric mismatch: {expected['gate']} {field}")
    all_pass = all(x.get("pass") is True for x in gate_replay)
    if bool(result.get("all_gates_pass")) != all_pass or bool(result.get("qualified_candidate")) != all_pass:
        raise AssertionError("independent overall gate verdict mismatch")
    return {
        "status": "PASS", "freeze_signature": freeze_data["freeze_signature"],
        "prelabel_verification_sha256": offset.sha256_file(run_root / "PRELABEL_VERIFICATION.json"),
        "results_gate_reconstruction": gate_replay, "origins": origins_replayed,
        "query_labels_read_after_prelabel": True, "test_or_sample_access": False,
        "optimizer_refits": 0, "independent_metric_reconstruction": True,
        "gate_count": len(gate_replay), "all_gates_pass": all_pass,
    }
