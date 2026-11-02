"""Independent, zero-fit verification of the frozen A01 official pilot."""
from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)

from scripts.research_campaign_classification import load_train
from src.models.classification_models import _number_array
from src.models.classification_round2_models import stream_missing_mask
from src.models.classification_recency_a01 import (
    CARD_ID, FEATURE_NAMES, ORIGINS, PARAMS, THRESHOLD, _TFIDF_PARAMS,
)
from src.models.optimization_v3_batch1 import json_sha256, read_bundle as read_c16_bundle
from src.models.research_next_budget import read_events
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
CONFIG = ROOT / "experiments/optimization_v3/classification_recency_a01"
PROTOCOL = CONFIG / "protocol.json"
FREEZE = CONFIG / "freeze.json"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_classification_recency_a01"
REPORT_ROOT = ROOT / "reports/optimization_v3/classification_recency_a01"
RESULTS = REPORT_ROOT / "results.json"
VERIFICATION = REPORT_ROOT / "independent_verification.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
C16_FREEZE = ROOT / "experiments/optimization_v3/batch1/C16/freeze.json"
C16_RECIPE = ROOT / "experiments/optimization_v3/batch1/C16/recipe.json"
C16_CACHE = ROOT / "experiments/runs/optimization_v3_batch1/C16"
C16_DRIFT_REPORT = ROOT / "reports/optimization_v3/batch1/RECONCILIATION_01.md"
CARD_ID = "CLASSIFICATION_RECENCY_A01"
STAGE = "A01_WEIGHTED_CLASSIFICATION_PILOT"
AMENDMENT_ID = "A01_PILOT_REALLOCATION_2026_11_02"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
ORIGINS = (55, 67)
ARMS = ("RECENCY", "SHUFFLED_WITHIN_CLASS")
ORDER = ((55, "RECENCY"), (55, "SHUFFLED_WITHIN_CLASS"),
         (67, "RECENCY"), (67, "SHUFFLED_WITHIN_CLASS"))
VIEWS = {
    "primary": [(55, 56, 59), (67, 68, 71)],
    "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
    "H12_55": [(55, 56, 67)], "H12_67": [(67, 68, 79)],
    "H12_pooled": [(55, 56, 67), (67, 68, 79)],
}
GATES = {
    "primary_gain_vs_sparse25_min": 0.0, "primary_gain_vs_shuffled_min": 0.0,
    "H12_gain_vs_sparse25_min": 0.0005, "H12_gain_vs_shuffled_min": 0.0002,
    "H12_55_max_harm_vs_sparse25": 0.001, "H12_67_max_harm_vs_sparse25": 0.001,
    "W1_max_harm_vs_sparse25": 0.003, "W4_max_harm_vs_sparse25": 0.003,
}
GROUP_SUPPORT = {"minimum_n": 200, "minimum_each_class_n": 30}
GROUP_GUARDS = {"macro_f1_harm_max": 0.01, "positive_recall_harm_max": 0.03}
RISK = {"macro_f1_harm_gt": 0.03, "positive_recall_harm_gt": 0.05}
SOURCE_FILES = (
    "src/models/classification_recency_a01.py",
    "scripts/optimization_v3_classification_recency_a01.py",
    "scripts/optimization_v3_classification_recency_a01_verify.py",
    "scripts/optimization_v3_a01_preflight.py",
    "scripts/optimization_v3_a01_preflight_verify.py",
    "tests/test_optimization_v3_a01_preflight.py",
    "scripts/research_campaign_classification.py",
    "scripts/optimization_v3_batch1_c16.py",
    "scripts/optimization_v3_batch1_verify.py",
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/optimization_v3_sparse_classification.py",
    "src/models/optimization_v3_batch1_c16.py",
    "src/models/optimization_v3_batch1.py",
    "src/models/research_next_budget.py",
    "src/models/research_next_contract.py",
    "src/models/regression_research_text.py",
    "src/models/round1_models.py",
    "src/data/audit.py",
    "src/utils/paths.py",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_int64(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= THRESHOLD).astype(np.int8)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "positive_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "negative_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "positive_rate": float(y.mean()), "predicted_positive_rate": float(pred.mean()),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def _weights(cycles: np.ndarray, target: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.maximum(0.25, np.power(2.0, -(cutoff - cycles) / 8.0))
    recency = np.empty(len(raw), dtype=np.float64)
    rng = np.random.default_rng(42)
    shuffled = np.empty_like(recency)
    source = np.arange(len(raw), dtype=np.int64)
    indices: dict[str, list[int]] = {}
    by_class = {}
    for cls in (0, 1):
        take = np.flatnonzero(target == cls)
        recency[take] = raw[take] / raw[take].mean()
        permutation = rng.permutation(len(take)).astype(np.int64)
        indices[str(cls)] = permutation.tolist()
        source[take] = take[permutation]
        shuffled[take] = recency[take[permutation]]
        for arm, weights in (("recency", recency), ("shuffled", shuffled)):
            current = weights[take]
            ess = float(current.sum() ** 2 / np.square(current).sum())
            by_class.setdefault(str(cls), {})[f"{arm}_ess_ratio"] = ess / len(take)
            by_class[str(cls)][f"{arm}_mass"] = float(current.sum())
    diag = {
        "cutoff": int(cutoff), "fit_rows": int(len(cycles)),
        "recency_sha256": hashlib.sha256(recency.tobytes(order="C")).hexdigest(),
        "shuffled_sha256": hashlib.sha256(shuffled.tobytes(order="C")).hexdigest(),
        "permutation_source_rows_sha256": _sha_int64(source),
        "permutation_indices_by_class": indices,
        "by_class": by_class,
        "all_class_ess_ge_0_50": all(value["recency_ess_ratio"] >= 0.5
                                      and value["shuffled_ess_ratio"] >= 0.5
                                      for value in by_class.values()),
    }
    return recency, shuffled, diag


def _expected_inputs(train_path: Path) -> dict[str, Path]:
    static = (
        "experiments/optimization_v3/batch1/C16/recipe.json",
        "experiments/optimization_v3/batch1/C16/freeze.json",
        "reports/optimization_v3/batch1/C16/results.json",
        "reports/optimization_v3/batch1/C16/verification.json",
        "reports/optimization_v3/batch1/C16/gate.json",
        "reports/optimization_v3/batch1/C16/summary.json",
        "reports/optimization_v3/batch1/RECONCILIATION_01.md",
        "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
        "experiments/optimization_v3/classification_sparse/freeze.json",
        "reports/classification_round1/fit_audits/fit55.json",
        "reports/classification_round1/fit_audits/fit67.json",
        "reports/optimization_v3/classification_recency_a01/synthetic_preflight_v2.json",
        "experiments/runs/optimization_v3_classification_recency_a01/synthetic_preflight_v2/independent_verification.json",
    )
    paths = {"official_classification_train": train_path}
    paths.update({name.replace("/", "__"): ROOT / name for name in static})
    for cutoff in ORIGINS:
        paths[f"C16_fit{cutoff}_npz"] = C16_CACHE / f"C16_fit{cutoff}.npz"
        paths[f"C16_fit{cutoff}_json"] = C16_CACHE / f"C16_fit{cutoff}.json"
    return paths


def _verify_controls(frame: pd.DataFrame, cycles: np.ndarray, protocol: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    freeze = _json(C16_FREEZE)
    recipe = _json(C16_RECIPE)
    if freeze.get("signature") != json_sha256(freeze.get("identity", {})):
        raise RuntimeError("C16 freeze identity checksum failed")
    if freeze["identity"].get("recipe_artifact_sha256") != json_sha256(recipe):
        raise RuntimeError("C16 recipe artifact does not match frozen hash")
    if _verify_c16_source_drift(freeze) != protocol.get("c16_source_drift_reconciliation"):
        raise RuntimeError("A01 C16 source-drift reconciliation differs from its frozen provenance")
    results = {}
    for cutoff in ORIGINS:
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        expected = {"freeze_signature": freeze["signature"], "card_id": "C16",
                    "cutoff": cutoff, "score_rows": rows.tolist()}
        archive, sidecar = read_c16_bundle(C16_CACHE / f"C16_fit{cutoff}.npz", expected)
        state = sidecar.get("metadata", {}).get("state", {})
        meta = sidecar.get("metadata", {})
        ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
        missing = stream_missing_mask(frame.iloc[rows]).astype(bool)
        if (int(meta.get("fit_rows", -1)) != int(np.sum(cycles <= cutoff))
                or int(meta.get("score_rows", -1)) != len(rows)
                or meta.get("fit_metadata", {}).get("score_labels_passed_to_adapter") is not False
                or state.get("sample_weight_used") is not False
                or state.get("id") != "C16_MODERATE_SPARSE_CAPACITY_LGBM"
                or not np.array_equal(archive["score_rows"], rows)
                or not np.array_equal(archive["control__control_rows"], rows)
                or not np.array_equal(archive["score_record_ids"].astype(str), ids)
                or not np.array_equal(archive["control__stream_missing"].astype(bool), missing)):
            raise RuntimeError(f"Independent C16 cache identity failed at cutoff {cutoff}")
        base = np.where(missing, archive["control__cb_probability"],
                        0.75 * archive["control__c2b_probability"]
                        + 0.25 * archive["control__baseline_sparse_raw"])
        np.testing.assert_allclose(base, archive["baseline_prediction"], atol=1e-12, rtol=0)
        results[cutoff] = {"rows": rows, "ids": ids, "cycles": cycles[rows], "missing": missing,
                           "baseline": base, "sparse_raw": archive["control__baseline_sparse_raw"].astype(float),
                           "c2b": archive["control__c2b_probability"].astype(float),
                           "cb": archive["control__cb_probability"].astype(float)}
    return results


def _verify_c16_source_drift(frozen: dict[str, Any]) -> dict[str, Any]:
    source_hashes = frozen.get("identity", {}).get("source_sha256", {})
    relative = "src/models/optimization_v3_batch1.py"
    source_path = ROOT / relative
    matches = [(Path(path), digest) for path, digest in source_hashes.items()
               if Path(path).resolve() == source_path.resolve()]
    if len(matches) != 1:
        raise RuntimeError("C16 frozen source manifest is missing its shared helper identity")
    pinned = str(matches[0][1])
    current_sha = file_sha256(source_path)
    other_mismatches = []
    for path_text, expected in source_hashes.items():
        path = Path(path_text)
        if not path.is_file():
            other_mismatches.append(str(path))
        elif file_sha256(path) != expected and path.resolve() != source_path.resolve():
            other_mismatches.append(str(path))
    if (other_mismatches or pinned != "d21ac914d0d1f516a93e431116556dabdbe7313a21fd2b1b04a73371bd56f420"
            or current_sha != "746e12d408e74608e5b559bc7dcf4c10174df9f367859750987310adb6ae8ffd"):
        raise RuntimeError(f"C16 source drift is outside the reconciled budget helper change: {other_mismatches}")
    old = subprocess.check_output(["git", "-C", str(ROOT), "show", f"9e9b7fb:{relative}"])
    if hashlib.sha256(old).hexdigest() != pinned:
        raise RuntimeError("C16 historical helper blob does not match its frozen source hash")
    def without_reservation_function(payload: bytes) -> str:
        module = ast.parse(payload.decode("utf-8"))
        module.body = [node for node in module.body
                       if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       or node.name != "reserve_card_budget"]
        return ast.dump(module, include_attributes=False)
    if without_reservation_function(old) != without_reservation_function(source_path.read_bytes()):
        raise RuntimeError("C16 helper drift includes a change outside reserve_card_budget")
    report = C16_DRIFT_REPORT.read_text(encoding="utf-8")
    required = ("only later helper diff is budget-reservation bookkeeping in `reserve_card_budget`",
                pinned, "current SHA `746e12d4")
    if not all(token in report for token in required):
        raise RuntimeError("C16 source-drift reconciliation report lacks the metadata-only conclusion")
    return {"status": "METADATA_ONLY_RESERVATION_BOOKKEEPING_ONLY", "path": relative,
            "pinned_sha256": pinned, "current_sha256": current_sha,
            "historical_source_commit": "9e9b7fb", "changed_function": "reserve_card_budget",
            "ast_outside_changed_function_identical": True,
            "reconciliation_report": "reports/optimization_v3/batch1/RECONCILIATION_01.md",
            "reconciliation_report_sha256": file_sha256(C16_DRIFT_REPORT)}


def _load_outputs(frame: pd.DataFrame, cycles: np.ndarray, protocol: dict[str, Any],
                  signature: str) -> dict[int, dict[str, Any]]:
    controls = _verify_controls(frame, cycles, protocol)
    outputs = {}
    for cutoff in ORIGINS:
        per_arm = {}
        for arm in ARMS:
            fit_id = f"{CARD_ID}_{cutoff}_{arm}"
            fit_dir = RUN_ROOT / f"fit{cutoff}_{arm.lower()}"
            model_path, state_path, bundle_path = fit_dir / "fitted_state.joblib", fit_dir / "fit_state.json", fit_dir / "bundle"
            state = _json(state_path)
            model_sha = file_sha256(model_path)
            state_payload = {k: v for k, v in state.items() if k != "fit_state_sha256"}
            if (state.get("fit_id") != fit_id or state.get("model_joblib_sha256") != model_sha
                    or state.get("fit_state_sha256") != canonical_sha256(state_payload)
                    or state.get("freeze_signature") != signature
                    or state.get("score_labels_passed_to_fit") is not False
                    or state.get("test_or_sample_data_used") is not False
                    or state.get("kaggle_access") is not False):
                raise RuntimeError(f"A01 independent model state failed for {fit_id}")
            bundle_identity = {"campaign": CARD_ID, "stage": STAGE, "cutoff": cutoff, "arm": arm,
                               "source_commit": protocol["source_commit"],
                               "protocol_sha256": file_sha256(PROTOCOL), "freeze_signature": signature,
                               "model_joblib_sha256": model_sha, "fit_state_sha256": file_sha256(state_path),
                               "threshold": THRESHOLD}
            bundle = read_bundle(bundle_path, bundle_identity, task="classification")
            arrays = bundle.arrays
            ctl = controls[cutoff]
            if (not np.array_equal(arrays["score_rows"], ctl["rows"])
                    or not np.array_equal(arrays["score_record_ids"].astype(str), ctl["ids"])
                    or not np.array_equal(arrays["cycle_code"].astype(np.int64), ctl["cycles"])
                    or not np.array_equal(arrays["stream_missing"].astype(bool), ctl["missing"])
                    or not np.array_equal(arrays["baseline_prediction"], ctl["baseline"])
                    or bundle.receipt["receipt_sha256"] != next(e["receipt_sha256"] for e in read_events(LEDGER)
                        if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id)):
                raise RuntimeError(f"A01 independent bundle identity failed for {fit_id}")
            stored = joblib.load(model_path)
            model = stored["model"]
            actual_probability = model.predict_probability(frame.iloc[ctl["rows"]].loc[:, list(FEATURE_NAMES)])
            if not np.array_equal(actual_probability, arrays["raw_probability"]):
                raise RuntimeError(f"A01 raw probability does not reproduce from fitted model: {fit_id}")
            weight_recency, weight_shuffle, diagnostics = _weights(
                cycles[(cycles >= 47) & (cycles <= cutoff)],
                pd.to_numeric(frame.loc[(cycles >= 47) & (cycles <= cutoff), "artifact_signal"], errors="raise").to_numpy(dtype=np.int8), cutoff)
            expected_weight = weight_recency if arm == "RECENCY" else weight_shuffle
            audit = model.audit_state()
            if (audit.get("lightgbm_parameters") != dict(PARAMS)
                    or audit.get("tfidf_parameters") != {k: (list(v) if isinstance(v, tuple)
                        else (v.__name__ if isinstance(v, type) else v)) for k, v in _TFIDF_PARAMS.items()}
                    or audit.get("sample_weight_sha256") != hashlib.sha256(expected_weight.tobytes(order="C")).hexdigest()
                    or not diagnostics["all_class_ess_ge_0_50"]
                    or diagnostics["permutation_source_rows_sha256"] != state["sample_weight_diagnostics"]["permutation_source_rows_sha256"]):
                raise RuntimeError(f"A01 independent recipe/weight audit failed for {fit_id}")
            per_arm[arm] = {key: np.asarray(value) for key, value in arrays.items()}
            per_arm[arm]["state"] = state
            per_arm[arm]["receipt"] = bundle.receipt
        rec = per_arm["RECENCY"]
        shuf = per_arm["SHUFFLED_WITHIN_CLASS"]
        if not np.array_equal(rec["score_rows"], shuf["score_rows"]):
            raise RuntimeError(f"A01 arm row mismatch at cutoff {cutoff}")
        # Independently reconstruct the frozen output routes.
        candidate = np.where(ctl["missing"], ctl["cb"], 0.75 * ctl["c2b"] + 0.25 * rec["raw_probability"])
        matched = np.where(ctl["missing"], ctl["cb"], 0.75 * ctl["c2b"] + 0.25 * shuf["raw_probability"])
        np.testing.assert_allclose(candidate, rec["prediction"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(matched, shuf["prediction"], atol=1e-12, rtol=0)
        outputs[cutoff] = {"rows": ctl["rows"], "cycles": ctl["cycles"], "ids": ctl["ids"],
                           "missing": ctl["missing"], "candidate": candidate, "baseline": ctl["baseline"],
                           "matched": matched, "c2b": np.where(ctl["missing"], ctl["cb"], ctl["c2b"]),
                           "origin": np.full(len(ctl["rows"]), cutoff, dtype=np.int64),
                           "states": {arm: per_arm[arm]["state"] for arm in ARMS},
                           "receipts": {arm: per_arm[arm]["receipt"] for arm in ARMS}}
    return outputs


def _view_arrays(outputs: dict[int, dict[str, Any]], blocks: list[list[int]] | list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    pieces = []
    for cutoff, lo, hi in blocks:
        result = outputs[int(cutoff)]
        take = (result["cycles"] >= int(lo)) & (result["cycles"] <= int(hi))
        pieces.append({key: result[key][take] for key in ("rows", "candidate", "baseline", "matched", "c2b", "origin", "cycles")})
    merged = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    order = np.argsort(merged["rows"], kind="stable")
    return {key: value[order] for key, value in merged.items()}


def _independent_eval(outputs: dict[int, dict[str, Any]], frame: pd.DataFrame,
                      target: np.ndarray, expected_result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    computed_metrics: dict[str, Any] = {}
    computed_groups: list[dict[str, Any]] = []
    computed_cycles: list[dict[str, Any]] = []
    bins = {}
    parsed, _missing_values, _invalid = _number_array(frame["text_load"])
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    for cutoff in ORIGINS:
        finite = parsed[(cycles >= 47) & (cycles <= cutoff) & np.isfinite(parsed)]
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
        labels = np.full(len(frame), -1, dtype=np.int64)
        valid = np.isfinite(parsed)
        labels[valid] = np.searchsorted(edges, parsed[valid], side="right")
        bins[cutoff] = labels
    risk_groups = []
    for name, blocks in VIEWS.items():
        data = _view_arrays(outputs, blocks)
        rows = data["rows"]
        y = target[rows]
        model_probability = {"recency": data["candidate"], "sparse25": data["baseline"],
                             "shuffled": data["matched"], "C2_B_diagnostic": data["c2b"]}
        metrics = {key: _metric(y, probability) for key, probability in model_probability.items()}
        gains = {key: {field: (None if metrics["recency"][field] is None or metrics[key][field] is None
                               else metrics["recency"][field] - metrics[key][field])
                       for field in ("macro_f1", "positive_recall", "positive_precision",
                                     "predicted_positive_rate", "log_loss")}
                  for key in ("sparse25", "shuffled", "C2_B_diagnostic")}
        computed_metrics[name] = {"n": len(rows), "models": metrics, "gains": gains,
                                  "origins": sorted(set(data["origin"].tolist())),
                                  "score_rows_sha256": _sha_int64(rows)}
        for cycle_value in sorted(np.unique(data["cycles"]).tolist()):
            take_cycle = data["cycles"] == cycle_value
            computed_cycles.append({"view": name, "cycle": int(cycle_value), "n": int(take_cycle.sum()),
                                    **{f"{model}_macro_f1": _metric(y[take_cycle], probability[take_cycle])["macro_f1"]
                                       for model, probability in model_probability.items()}})
        row_frame = frame.iloc[rows]
        label_groups = {
            "fit_defined_text_load_quartile": np.asarray(["missing" if bins[int(o)][int(r)] < 0
                else f"Q{min(int(bins[int(o)][int(r)]) + 1, 4)}" for r, o in zip(rows, data["origin"])]),
            "stream_availability": np.where(stream_missing_mask(row_frame), "missing", "present"),
            "entry_friction": row_frame["entry_friction"].astype(str).to_numpy(),
        }
        for field, labels_group in label_groups.items():
            for label in sorted(np.unique(labels_group).tolist()):
                take = labels_group == label
                gy = y[take]
                n0, n1 = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                supported = len(gy) >= GROUP_SUPPORT["minimum_n"] and min(n0, n1) >= GROUP_SUPPORT["minimum_each_class_n"]
                a, b, c = (_metric(gy, model_probability[k][take]) for k in ("recency", "sparse25", "shuffled"))
                f1_harm = b["macro_f1"] - a["macro_f1"]
                recall_harm = b["positive_recall"] - a["positive_recall"]
                row = {"view": name, "field": field, "group": str(label), "n": len(gy),
                       "class_0_n": n0, "class_1_n": n1, "supported": bool(supported),
                       "recency_macro_f1": a["macro_f1"], "sparse25_macro_f1": b["macro_f1"],
                       "shuffled_macro_f1": c["macro_f1"], "f1_harm_vs_sparse25": f1_harm,
                       "recency_positive_recall": a["positive_recall"],
                       "sparse25_positive_recall": b["positive_recall"],
                       "positive_recall_harm_vs_sparse25": recall_harm,
                       "fails_legacy_f1_guard": bool(supported and f1_harm > GROUP_GUARDS["macro_f1_harm_max"]),
                       "fails_legacy_recall_guard": bool(supported and recall_harm > GROUP_GUARDS["positive_recall_harm_max"]),
                       "risk_review_hold": bool(supported and (f1_harm > RISK["macro_f1_harm_gt"]
                                                                 or recall_harm > RISK["positive_recall_harm_gt"]))}
                computed_groups.append(row)
                if row["risk_review_hold"]:
                    risk_groups.append(row)
    # Compare the independently calculated primary outputs with the persisted report.
    if set(expected_result.get("metrics_by_view", {})) != set(computed_metrics):
        raise RuntimeError("A01 report view set differs from independent reconstruction")
    for view, values in computed_metrics.items():
        saved = expected_result["metrics_by_view"][view]
        if saved.get("n") != values["n"]:
            raise RuntimeError(f"A01 denominator mismatch in {view}")
        for model in values["models"]:
            for field, expected_value in values["models"][model].items():
                saved_value = saved["models"][model].get(field)
                if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                    if not np.isclose(saved_value, expected_value, atol=1e-14, rtol=0):
                        raise RuntimeError(f"A01 metric mismatch: {view}/{model}/{field}")
                elif saved_value != expected_value:
                    raise RuntimeError(f"A01 metric mismatch: {view}/{model}/{field}")
        for comparator, fields in values["gains"].items():
            for field, expected_value in fields.items():
                saved_value = saved["gains"].get(comparator, {}).get(field)
                if expected_value is None:
                    if saved_value is not None:
                        raise RuntimeError(f"A01 metric gain mismatch: {view}/{comparator}/{field}")
                elif not np.isclose(saved_value, expected_value, atol=1e-14, rtol=0):
                    raise RuntimeError(f"A01 metric gain mismatch: {view}/{comparator}/{field}")
    if canonical_sha256(computed_groups) != canonical_sha256(expected_result.get("protected_groups", [])):
        raise RuntimeError("A01 protected subgroup table differs from independent reconstruction")
    if canonical_sha256(computed_cycles) != canonical_sha256(expected_result.get("per_cycle", [])):
        raise RuntimeError("A01 per-cycle diagnostics differ from independent reconstruction")
    gains = {name: computed_metrics[name]["gains"]["sparse25"]["macro_f1"]
             for name in ("primary", "W1", "W4", "H12_55", "H12_67", "H12_pooled")}
    checks = {
        "primary_nonnegative_vs_sparse25": gains["primary"] >= GATES["primary_gain_vs_sparse25_min"],
        "primary_nonnegative_vs_shuffled": computed_metrics["primary"]["gains"]["shuffled"]["macro_f1"] >= GATES["primary_gain_vs_shuffled_min"],
        "H12_min_gain_vs_sparse25": gains["H12_pooled"] >= GATES["H12_gain_vs_sparse25_min"],
        "H12_min_gain_vs_shuffled": computed_metrics["H12_pooled"]["gains"]["shuffled"]["macro_f1"] >= GATES["H12_gain_vs_shuffled_min"],
        "H12_55_harm_guard": gains["H12_55"] >= -GATES["H12_55_max_harm_vs_sparse25"],
        "H12_67_harm_guard": gains["H12_67"] >= -GATES["H12_67_max_harm_vs_sparse25"],
        "W1_harm_guard": gains["W1"] >= -GATES["W1_max_harm_vs_sparse25"],
        "W4_harm_guard": gains["W4"] >= -GATES["W4_max_harm_vs_sparse25"],
        "protected_legacy_subgroup_guards": not any(g["supported"] and
            (g["fails_legacy_f1_guard"] or g["fails_legacy_recall_guard"]) for g in computed_groups),
    }
    scientific = "RISK_REVIEW_HOLD" if all(checks.values()) and risk_groups else (
        "QUALIFIED_FOR_REVIEW" if all(checks.values()) else "NO_GO_GATE_FAILED")
    saved_gate = expected_result.get("frozen_gate", {})
    if (saved_gate.get("checks") != checks or saved_gate.get("scientific_verdict") != scientific
            or saved_gate.get("gains_vs_sparse25") != gains):
        raise RuntimeError("A01 frozen gate verdict differs from independent reconstruction")
    return {"metrics_by_view": computed_metrics, "per_cycle": computed_cycles,
            "gates": {"checks": checks,
            "gains_vs_sparse25": gains, "scientific_verdict": scientific,
            "risk_review_hold_groups": risk_groups}}, computed_groups


def _verify_budget(protocol: dict[str, Any]) -> dict[str, Any]:
    raw_lines = LEDGER.read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in raw_lines if line.strip()]
    amendment = [e for e in events if e.get("event") == "BUDGET_CATEGORY_AMENDMENT"
                 and e.get("amendment_id") == AMENDMENT_ID]
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("reservation_key") == RESERVATION_KEY]
    prefix_count = protocol["budget_pre_amendment"]["event_rows"]
    prefix_hash = hashlib.sha256(b"".join(raw_lines[:prefix_count])).hexdigest()
    auth = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if (len(auth) != 1 or auth[0].get("category_caps") != {
            "pilot": 26, "continuation": 24, "control_replay": 6, "technical_retry": 4}
            or int(auth[0].get("cap", -1)) != 60
            or len(amendment) != 1 or len(reservation) != 1
            or prefix_hash != protocol["budget_pre_amendment"]["ledger_sha256"]
            or amendment[0].get("pre_amendment_ledger_sha256") != prefix_hash
            or amendment[0].get("pre_amendment_event_rows") != prefix_count
            or amendment[0].get("original_caps") != {"continuation": 24, "control_replay": 6,
                                                       "pilot": 26, "technical_retry": 4}
            or amendment[0].get("original_authorization_sha256") != json_sha256(auth[0])
            or amendment[0].get("source_category") != "control_replay"
            or amendment[0].get("destination_category") != "pilot"
            or int(amendment[0].get("fits", -1)) != 4
            or amendment[0].get("effective_caps") != {"continuation": 24, "control_replay": 2,
                                                       "pilot": 30, "technical_retry": 4}
            or int(reservation[0].get("fits", -1)) != 4
            or reservation[0].get("category") != "pilot"
            or reservation[0].get("amendment_id") != AMENDMENT_ID):
        raise RuntimeError("A01 append-only budget amendment or reservation did not verify")
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    a01_started = [e for e in starts if e.get("card_id") == CARD_ID]
    a01_saved = [e for e in saved if e.get("card_id") == CARD_ID]
    a01_failed = [e for e in failed if e.get("card_id") == CARD_ID]
    expected_ids = {f"{CARD_ID}_{cutoff}_{arm}" for cutoff, arm in ORDER}
    if ({e.get("fit_id") for e in a01_started} != expected_ids
            or {e.get("fit_id") for e in a01_saved} != expected_ids
            or a01_failed or len(a01_started) != 4 or len(a01_saved) != 4
            or len(starts) != 54 or len(saved) != 52 or len(failed) != 2):
        raise RuntimeError("A01 global or card fit accounting does not equal exactly four successful fits")
    release = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"
               and e.get("reservation_key") == RESERVATION_KEY]
    if len(release) != 1 or int(release[0].get("unused_fits", -1)) != 0:
        raise RuntimeError("A01 reservation lacks exactly one zero-unused release")
    return {"amendment": amendment[0], "reservation": reservation[0], "release": release[0],
            "global_starts": len(starts), "global_saved": len(saved), "global_failed": len(failed),
            "a01_started": len(a01_started), "a01_saved": len(a01_saved), "a01_failed": len(a01_failed),
            "pre_amendment_prefix_sha256": prefix_hash, "ledger_sha256": file_sha256(LEDGER),
            "ledger_event_rows": len(events)}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Immutable verification exists and differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def verify() -> dict[str, Any]:
    protocol = _json(PROTOCOL)
    frozen = _json(FREEZE)
    signature = frozen.get("freeze_signature")
    if (frozen.get("identity") != protocol or signature != canonical_sha256(protocol)
            or frozen.get("protocol_sha256") != file_sha256(PROTOCOL)):
        raise RuntimeError("A01 official freeze/protocol checksum does not verify")
    if not RESULTS.is_file():
        raise FileNotFoundError("A01 results.json is missing")
    report = _json(RESULTS)
    train_path = load_train(None)[2]
    frame, target = load_train(None)[:2]
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if file_sha256(train_path) != protocol["official_train_sha256"] or len(frame) != 50_608:
        raise RuntimeError("Independent A01 Train hash or size differs")
    source_ok = {name: file_sha256(ROOT / name) == digest
                 for name, digest in protocol["code_sha256"].items()}
    if set(protocol["code_sha256"]) != set(SOURCE_FILES):
        raise RuntimeError("A01 frozen source manifest keys differ from the independent verifier's allowlist")
    input_paths = _expected_inputs(Path(train_path))
    input_ok = {key: Path(input_paths[key]).is_file()
                and file_sha256(Path(input_paths[key])) == digest
                for key, digest in protocol["input_sha256"].items()
                if key in input_paths}
    if set(input_ok) != set(protocol["input_sha256"]):
        raise RuntimeError("A01 independent source/input manifest keys differ")
    if not all(source_ok.values()) or not all(input_ok.values()):
        raise RuntimeError(f"A01 code/input hash mismatch: sources={source_ok}, inputs={input_ok}")
    if (report.get("official_estimator_fits") != 4 or report.get("official_train_rows") != 50_608
            or report.get("official_train_sha256") != protocol["official_train_sha256"]
            or report.get("test_read") is not False or report.get("sample_submission_read") is not False
            or report.get("kaggle_access") is not False):
        raise RuntimeError("A01 report scope or fit count differs from frozen protocol")
    budget = _verify_budget(protocol)
    outputs = _load_outputs(frame, cycles, protocol, signature)
    eval_values, groups = _independent_eval(outputs, frame, target, report)
    checks = {
        "freeze_and_protocol_identity": True,
        "source_and_input_hashes": True,
        "official_train_only_boundary": True,
        "C16_control_cache_row_and_route_integrity": True,
        "exact_four_fit_budget_and_append_only_amendment": True,
        "fit_only_weights_and_class_ESS": True,
        "fitted_model_raw_prediction_reproduction": True,
        "bundle_rows_route_and_receipts": True,
        "primary_h12_windows_subgroups_and_gate_reconstructed": True,
    }
    result = {"schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
              "campaign": CARD_ID, "freeze_signature": signature,
              "source_commit": protocol["source_commit"], "checks": checks,
              "source_hash_checks": source_ok, "input_hash_checks": input_ok,
              "independent_metrics_and_gates": eval_values,
              "independent_protected_groups": groups, "budget_reconciliation": budget,
              "official_estimator_fits_performed_by_verifier": 0,
              "test_read": False, "sample_submission_read": False, "kaggle_access": False}
    _write_once(VERIFICATION, result)
    return result


if __name__ == "__main__":
    result = verify()
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "scientific_verdict": result["independent_metrics_and_gates"]["gates"]["scientific_verdict"]},
                     sort_keys=True, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
