"""Independent, target-blind verifier for CLS_BIGRAM_POSITION_PILOT_V1 outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from src.models.optimization_v3_bigram_position_pilot_v1 import (
    reconstruct_original_pos_logit, route_candidate,
)
from src.models.optimization_v3_bigram_position_readiness_v1 import build_bigram_blocks
from scripts import optimization_v3_positional_bag_offset_v1 as offset
from scripts import optimization_v3_positional_contrast_v1 as contrast


CARD = "CLS_BIGRAM_POSITION_PILOT_V1"
FIT_IDS = tuple(f"{CARD}_E{o}_{a}" for o in (67, 75) for a in ("BAG", "REAL", "SHUFFLED"))
TRAIN = Path("data/raw/classification/Train.csv")
POS_BAG_ROOT = Path("experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1")
POS_CONTRAST_ROOT = Path("experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1")
TEXT_ROOT = Path("experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1")
EXPECTED_PARENT_FREEZE = "ba58b1e446a6be8546789c5a56596fbdbf075efc0ee19fcc1f41e02af7b419d7"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def verify_prelabel_outputs(root: Path, run_root: Path,
                            freeze: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    run_root = Path(run_root)
    errors: list[str] = []
    origin_checks: dict[str, Any] = {}
    try:
        train_path = root / TRAIN
        if offset.sha256_file(train_path) != freeze["input_sha256"]:
            raise AssertionError("official Train file hash differs from freeze")
        preflight_path = root / "reports/optimization_v3/bigram_position_pilot_v1/PREFLIGHT.json"
        preflight = _read_json(preflight_path)
        if (offset.sha256_file(preflight_path) != freeze.get("preflight_sha256")
                or preflight.get("status") != "PREFIT_PASS"
                or preflight.get("labels_opened") is not False
                or preflight.get("fit_label_reader", {}).get("embedded_CR_or_LF_rows") != 0):
            raise AssertionError("frozen target-free preflight/fit-only label reader proof is invalid")
        if offset._git("rev-parse", "HEAD") != freeze["source_commit"] and offset._git(
                "merge-base", "--is-ancestor", freeze["source_commit"], "HEAD") is None:
            raise AssertionError("frozen source commit is not current or ancestor")
        for rel, digest in freeze["source_hashes"].items():
            if offset.sha256_file(root / rel) != digest:
                raise AssertionError(f"frozen source hash changed: {rel}")
        parent_freeze = _read_json(root / POS_BAG_ROOT / "FREEZE.json")
        if parent_freeze.get("freeze_signature") != EXPECTED_PARENT_FREEZE:
            raise AssertionError("POS_BAG parent freeze mismatch")
        parent_integrity = offset.verify_saved_inputs(include_current_code=False)
        if parent_integrity.get("input_sha256") != freeze["input_sha256"]:
            raise AssertionError("historical POS_BAG source identity mismatch")
        parent_prelabel = _read_json(root / POS_BAG_ROOT / "PRELABEL_VERIFICATION.json")
        if parent_prelabel.get("status") != "PASS" or parent_prelabel.get("query_labels_read") is not False:
            raise AssertionError("historical POS_BAG query labels were not kept closed")

        identity = offset._read_identity_frame()
        text = pd.read_csv(train_path, usecols=["signal_stream"])["signal_stream"]
        documents = ["" if pd.isna(x) else str(x) for x in text.tolist()]
        raw_present = np.asarray([bool(x.strip()) for x in documents], dtype=bool)
        del text
        events = offset._ledger_events(root / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl")
        for origin in (67, 75):
            key = f"E{origin}"
            fz = freeze["origins"][key]
            manifest_path = root / TEXT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
            manifest = _read_json(manifest_path)
            if offset.sha256_file(manifest_path) != fz["parent_manifest_sha256"]:
                raise AssertionError(f"E{origin} parent matrix manifest hash mismatch")
            old_pred_path = root / POS_BAG_ROOT / f"E{origin}/predictions_E{origin}.npz"
            if offset.sha256_file(old_pred_path) != fz["parent_prediction_sha256"]:
                raise AssertionError(f"E{origin} parent prediction file hash mismatch")
            old_gamma_id = f"POS_BAG_OFFSET_V1_E{origin}_REAL"
            old_rec_path = root / POS_BAG_ROOT / f"receipts/{old_gamma_id}.json"
            old_rec = _read_json(old_rec_path)
            old_gamma_path = root / old_rec["artifact_path"]
            if offset.sha256_file(old_gamma_path) != old_rec["artifact_sha256"]:
                raise AssertionError(f"E{origin} original POS gamma artifact hash mismatch")
            with np.load(old_gamma_path, allow_pickle=False) as a:
                old_gamma = np.asarray(a["gamma"], dtype=np.float64).copy()
            if offset.sha256_array(old_gamma) != old_rec["gamma_sha256"]:
                raise AssertionError(f"E{origin} original POS gamma logical hash mismatch")

            c2b_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF"
            vector_path = root / TEXT_ROOT / "units" / f"{c2b_id}.joblib"
            vector_receipt_path = root / TEXT_ROOT / "receipts" / f"{c2b_id}.json"
            vector_receipt = _read_json(vector_receipt_path)
            if offset.sha256_file(vector_path) != vector_receipt["artifact_sha256"]:
                raise AssertionError(f"E{origin} frozen TF-IDF state hash mismatch")
            vectorizer = joblib.load(vector_path)
            params = vectorizer.get_params(deep=False)
            if (params.get("token_pattern") != r"(?u)\b\w\w+\b"
                    or tuple(params.get("ngram_range", ())) != (1, 2)
                    or params.get("strip_accents") != "unicode" or params.get("lowercase") is not True):
                raise AssertionError(f"E{origin} frozen text normalization/tokenization changed")

            # The independent feature rebuild consumes text only, never artifact_signal.
            rebuilt, metadata = build_bigram_blocks(documents, vectorizer)
            directory = run_root / f"E{origin}"
            rows_path = directory / "frozen_rows.npz"
            if offset.sha256_file(rows_path) != fz["row_arrays_sha256"]["metadata_file"]:
                raise AssertionError(f"E{origin} frozen row metadata file hash mismatch")
            with np.load(rows_path, allow_pickle=False) as a:
                rows = {name: a[name].copy() for name in a.files}
            fit_rows = np.asarray(rows["fit_rows"], dtype=np.int64)
            score_rows = np.asarray(rows["score_rows"], dtype=np.int64)
            if (not np.array_equal(identity.iloc[fit_rows]["record_id"].astype(str).to_numpy(), rows["fit_record_ids"].astype(str))
                    or not np.array_equal(identity.iloc[score_rows]["record_id"].astype(str).to_numpy(), rows["score_record_ids"].astype(str))
                    or not np.array_equal(identity.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64), rows["score_cycles"].astype(np.int64))):
                raise AssertionError(f"E{origin} row index/ID/cycle alignment failed")
            if not np.array_equal(raw_present[score_rows], rows["present_mask"].astype(bool)):
                raise AssertionError(f"E{origin} present/missing mask changed")
            actual_hashes = {}
            for arm in ("bag", "real", "shuffled"):
                for split, row_idx in (("fit", fit_rows), ("query", score_rows)):
                    matrix = rebuilt[arm][row_idx].tocsr()
                    cache = sparse.load_npz(directory / f"{split}_{arm}.npz").tocsr()
                    item = fz["matrix_hashes"][f"{split}_{arm}"]
                    if (list(cache.shape) != item["shape"]
                            or offset.sha256_csr(cache) != item["logical_sha256"]
                            or offset.sha256_file(directory / f"{split}_{arm}.npz") != item["file_sha256"]
                            or offset.sha256_csr(matrix) != item["logical_sha256"]):
                        raise AssertionError(f"E{origin} {split}/{arm} feature matrix rebuild/hash mismatch")
                    actual_hashes[f"{split}_{arm}"] = item["logical_sha256"]
            if not np.array_equal(metadata["pair_bag_preserved"], np.ones(len(documents), dtype=bool)) or not np.array_equal(
                    metadata["segment_sizes_preserved"], np.ones(len(documents), dtype=bool)):
                raise AssertionError(f"E{origin} shuffled pair multiset/bin-size control failed")

            # Independently reconstruct the frozen C2B + original unigram-POS logit.
            _, parent_pred, bag_estimator, payload = offset._load_origin_payload(origin, parent_freeze)
            mats = payload["matrices"]
            base_fit = sparse.csr_matrix(mats["fit_base"], dtype=np.float64)
            base_query = sparse.csr_matrix(mats["score_base"], dtype=np.float64)
            full_fit_pos = sparse.csr_matrix(mats["fit_position"], dtype=np.float64)
            full_query_pos = sparse.csr_matrix(mats["score_position"], dtype=np.float64)
            xfit_old = full_fit_pos[:, base_fit.shape[1]:].tocsr()
            xquery_old = full_query_pos[:, base_query.shape[1]:].tocsr()
            zbag_fit, zfit = reconstruct_original_pos_logit(base_fit, bag_estimator.coef_[0],
                                                            bag_estimator.intercept_, xfit_old, old_gamma)
            zbag_query, zquery = reconstruct_original_pos_logit(base_query, bag_estimator.coef_[0],
                                                                bag_estimator.intercept_, xquery_old, old_gamma)
            if (offset.sha256_array(zfit) != fz["original_z_fit_sha256"]
                    or offset.sha256_array(zquery) != fz["original_z_query_sha256"]):
                raise AssertionError(f"E{origin} frozen original logit hashes differ")
            if (not np.array_equal(rows["z_fit_original"], zfit)
                    or not np.array_equal(rows["z_query_original"], zquery)
                    or not np.array_equal(rows["p_pos_original"], expit(zquery))):
                raise AssertionError(f"E{origin} saved original logit/probability cache differs from independent reconstruction")
            if (offset.sha256_array(zbag_fit) != offset.sha256_array(
                    np.asarray(bag_estimator.intercept_[0] + base_fit @ bag_estimator.coef_[0], dtype=np.float64).reshape(-1))
                    or offset.sha256_array(zbag_query) != offset.sha256_array(
                        np.asarray(bag_estimator.intercept_[0] + base_query @ bag_estimator.coef_[0], dtype=np.float64).reshape(-1))):
                raise AssertionError(f"E{origin} independent frozen bag-logit reconstruction failed")

            contrast_path = root / POS_CONTRAST_ROOT / f"E{origin}/predictions_E{origin}.npz"
            ready = _read_json(root / POS_CONTRAST_ROOT / "PREDICTIONS_READY.json")
            if ready.get("query_labels_opened") is not False or offset.sha256_file(contrast_path) != ready["predictions"][f"E{origin}"]["sha256"]:
                raise AssertionError(f"E{origin} POS_CONTRAST anchor bundle hash/target boundary failed")
            safe_keys = ["score_rows", "score_record_ids", "score_cycles", "present_mask", "prob_q_real",
                         "prob_pP", "prob_a02", "prob_mix50"]
            anchor = contrast._load_named_npz(contrast_path, safe_keys)
            p_a02_meta, a02 = contrast._load_a02(origin, include_groups=True)
            present = np.asarray(anchor["present_mask"], dtype=bool)
            if not np.array_equal(anchor["score_rows"], score_rows) or not np.array_equal(anchor["score_record_ids"].astype(str), rows["score_record_ids"].astype(str)):
                raise AssertionError(f"E{origin} query anchor rows/IDs do not match frozen cache")
            if (not np.array_equal(present, np.asarray(rows["present_mask"], dtype=bool))
                    or not np.array_equal(np.asarray(a02["missing_mask"], dtype=bool), ~present)):
                raise AssertionError(f"E{origin} anchor/A02/frozen availability masks do not agree")
            replay = expit(zquery)
            max_replay = float(np.max(np.abs(replay[present] - anchor["prob_pP"][present]), initial=0.0))
            if max_replay > 1e-12:
                raise AssertionError(f"E{origin} original POS gamma probability replay mismatch")
            if not np.array_equal(anchor["prob_mix50"][~present], anchor["prob_pP"][~present]):
                raise AssertionError(f"E{origin} missing stream no longer routes exactly to pP")
            if not np.array_equal(np.asarray(rows["entry_friction"]).astype(str), a02["entry_friction"].astype(str)):
                raise AssertionError(f"E{origin} frozen entry-friction guard differs from A02 groups")
            if not np.array_equal(rows["text_load_group"], a02["text_load_group"]):
                raise AssertionError(f"E{origin} frozen text-load guard differs from A02 groups")
            zero = 0.5 * anchor["prob_pP"] + 0.5 * anchor["prob_a02"]
            zero[~present] = anchor["prob_pP"][~present]
            if float(np.max(np.abs(zero[present] - anchor["prob_mix50"][present]), initial=0.0)) > 1e-12:
                raise AssertionError(f"E{origin} zero-gamma MIX50 equivalence failed")

            arm_checks = {}
            for arm in ("BAG", "REAL", "SHUFFLED"):
                fit_id = f"{CARD}_{'E'+str(origin)}_{arm}"
                gamma_path = run_root / "fits" / f"{fit_id}.npz"
                receipt_path = run_root / "receipts" / f"{fit_id}.json"
                receipt = _read_json(receipt_path)
                if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != freeze["freeze_signature"]
                        or receipt.get("source_commit") != freeze["source_commit"]
                        or receipt.get("artifact_sha256") != offset.sha256_file(gamma_path)):
                    raise AssertionError(f"{fit_id} receipt does not bind to artifact/freeze")
                with np.load(gamma_path, allow_pickle=False) as g:
                    gamma = np.asarray(g["gamma"], dtype=np.float64).copy()
                xfit = sparse.load_npz(directory / f"fit_{arm.lower()}.npz").tocsr()
                xq = sparse.load_npz(directory / f"query_{arm.lower()}.npz").tocsr()
                optimizer = receipt.get("optimizer", {})
                saved_event = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
                if len(saved_event) != 1:
                    raise AssertionError(f"{fit_id} must have exactly one FIT_SAVED ledger event")
                fit_receipt_bound = (
                    receipt.get("input_sha256") == freeze["input_sha256"]
                    and receipt.get("fit_rows") == len(fit_rows)
                    and receipt.get("fit_rows_sha256") == offset.sha256_array(fit_rows)
                    and receipt.get("fit_record_ids_sha256") == offset.sha256_array(rows["fit_record_ids"].astype(str))
                    and receipt.get("feature_matrix_sha256") == offset.sha256_csr(xfit)
                    and receipt.get("offset_sha256") == offset.sha256_array(zfit)
                    and receipt.get("fit_label_count_0", 0) + receipt.get("fit_label_count_1", 0) == len(fit_rows)
                    and isinstance(receipt.get("fit_label_sha256"), str)
                    and len(receipt.get("fit_label_sha256", "")) == 64
                    and saved_event[0].get("artifact_sha256") == receipt.get("artifact_sha256")
                    and saved_event[0].get("receipt_sha256") == offset.sha256_file(receipt_path)
                )
                optimizer_bound = (
                    optimizer.get("success") is True and optimizer.get("c") == 1.0
                    and optimizer.get("options") == {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
                    and np.isfinite(optimizer.get("loss_final", np.nan))
                    and optimizer.get("loss_final", np.inf) <= optimizer.get("loss_gamma0", -np.inf) + 1e-8
                )
                if (not fit_receipt_bound or not optimizer_bound
                        or offset.sha256_array(gamma) != receipt.get("gamma_sha256")
                        or len(gamma) != xq.shape[1]
                        or len(gamma) != xfit.shape[1]
                        or receipt.get("source_commit") != freeze["source_commit"]
                        or receipt.get("freeze_signature") != freeze["freeze_signature"]
                        or offset.sha256_csr(xq) != freeze["origins"][f"E{origin}"]["matrix_hashes"][f"query_{arm.lower()}"]["logical_sha256"]):
                    raise AssertionError(f"{fit_id} fit/query receipt binding, optimizer, gamma, or feature hashes mismatch")
                q = expit(zquery + np.asarray(xq @ gamma, dtype=np.float64).reshape(-1))
                routed = 0.5 * q + 0.5 * anchor["prob_a02"]
                routed[~present] = anchor["prob_pP"][~present]
                no_known = np.asarray(rows["no_known_bigram"], dtype=bool)
                routed[present & no_known] = anchor["prob_mix50"][present & no_known]
                if not np.allclose(q, np.asarray(
                        offset.offset_probability(xq, zquery, gamma), dtype=np.float64), rtol=0, atol=0):
                    raise AssertionError(f"{fit_id} independent logit probability replay mismatch")
                arm_checks[arm] = {"fit_id": fit_id, "artifact_sha256": offset.sha256_file(gamma_path),
                                   "gamma_sha256": offset.sha256_array(gamma),
                                   "receipt_sha256": offset.sha256_file(receipt_path),
                                   "fit_receipt_binding_pass": fit_receipt_bound,
                                   "optimizer_binding_pass": optimizer_bound,
                                   "query_probability_sha256": offset.sha256_array(q),
                                   "route_probability_sha256": offset.sha256_array(routed),
                                   "missing_route_exact": bool(np.array_equal(routed[~present], anchor["prob_pP"][~present])),
                                   "zero_known_route_exact": bool(np.array_equal(routed[present & no_known], anchor["prob_mix50"][present & no_known]))}
            bundle_path = run_root / f"E{origin}/predictions_E{origin}.npz"
            ready_pilot = _read_json(run_root / "PREDICTIONS_READY.json")
            if offset.sha256_file(bundle_path) != ready_pilot["predictions"][f"E{origin}"]["sha256"]:
                raise AssertionError(f"E{origin} new prediction bundle file hash mismatch")
            with np.load(bundle_path, allow_pickle=False) as bundle:
                new = {k: bundle[k].copy() for k in bundle.files}
            for arm in ("BAG", "REAL", "SHUFFLED"):
                if offset.sha256_array(new[f"q_{arm.lower()}"].astype(np.float64)) != arm_checks[arm]["query_probability_sha256"]:
                    raise AssertionError(f"E{origin} saved {arm} query probabilities differ from independent replay")
                if offset.sha256_array(new[f"route_{arm.lower()}"].astype(np.float64)) != arm_checks[arm]["route_probability_sha256"]:
                    raise AssertionError(f"E{origin} saved {arm} route differs from independent replay")
            origin_checks[key] = {
                "fit_rows": int(len(fit_rows)), "query_rows": int(len(score_rows)),
                "fit_rows_sha256": offset.sha256_array(fit_rows), "query_rows_sha256": offset.sha256_array(score_rows),
                "rebuilt_feature_hashes": actual_hashes,
                "old_original_logit_replay_max_abs": max_replay,
                "all_three_arms": arm_checks,
                "bundle_sha256": offset.sha256_file(bundle_path),
                "a02_safe_hashes": p_a02_meta["safe_array_hashes"],
                "query_labels_read": False,
            }
            del rebuilt, metadata, parent_pred, payload
        own_events = [e for e in events if e.get("fit_id") in FIT_IDS]
        for fit_id in FIT_IDS:
            started = [e for e in own_events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
            if len(started) != 1:
                raise AssertionError(f"ledger does not have one FIT_STARTED event for {fit_id}")
            if (started[0].get("reservation_key") != f"{CARD}:SIX_FIT_TEMPORAL_PILOT"
                    or started[0].get("freeze_signature") != freeze["freeze_signature"]
                    or started[0].get("source_commit") != freeze["source_commit"]):
                raise AssertionError(f"FIT_STARTED event does not bind to authorized freeze for {fit_id}")
            if sum(e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id for e in own_events) != 1:
                raise AssertionError(f"ledger does not have one FIT_SAVED event for {fit_id}")
            if any(e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id for e in own_events):
                raise AssertionError(f"ledger records a failed fit for {fit_id}")
        return {"status": "PASS", "card_id": CARD, "freeze_signature": freeze["freeze_signature"],
                "origins": origin_checks, "fit_ids": list(FIT_IDS),
                "query_labels_read": False, "test_opened": False,
                "sample_submission_opened": False, "kaggle_accessed": False}
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return {"status": "FAIL", "errors": errors, "origins": origin_checks,
                "query_labels_read": False, "test_opened": False}


__all__ = ["verify_prelabel_outputs"]
