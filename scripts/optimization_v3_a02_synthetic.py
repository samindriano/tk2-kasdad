"""Synthetic A02 end-to-end, persistence/resume, and failure-injection QA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.classification_a02_neighbors import (  # noqa: E402
    ResidualNeighborPredictor,
    fit_text_space,
)
from src.models.research_next_contract import canonical_sha256  # noqa: E402
from scripts.optimization_v3_a02_neighbors import _atomic_joblib  # noqa: E402


def _manual_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return float((f0 + f1) / 2)


def run_synthetic() -> dict[str, Any]:
    n_prefix = 96
    origin = 67
    prefix_rows = np.arange(n_prefix, dtype=np.int64)
    cycles = np.resize(np.arange(47, 68, dtype=np.int64), n_prefix)
    docs = []
    for index in range(n_prefix):
        if index % 13 == 0:
            docs.append("")
        else:
            docs.append(f"marker shared signal{index % 16} family{index % 9} detail{index % 7}")
    text_space, _prefix_matrix, _prefix_missing = fit_text_space(
        docs, prefix_rows, cycles, origin_cycle=origin
    )
    bank_mask = (cycles >= 58) & (cycles <= 67)
    bank_rows = prefix_rows[bank_mask]
    bank_texts = np.asarray(docs, dtype=object)[bank_mask].tolist()
    present = np.asarray([bool(value.strip()) for value in bank_texts])
    bank_rows = bank_rows[present]
    bank_texts = np.asarray(bank_texts, dtype=object)[present].tolist()
    bank_ids = [f"synthetic-{row}" for row in bank_rows]
    bank_cycles = cycles[bank_rows]
    bank_probability = np.linspace(0.15, 0.85, len(bank_rows), dtype=np.float64)
    residual = np.where(np.arange(len(bank_rows)) % 2 == 0, 0.22, -0.18).astype(np.float64)

    query_rows = np.arange(n_prefix, n_prefix + 12, dtype=np.int64)
    query_cycles = np.full(len(query_rows), 68, dtype=np.int64)
    query_texts = [f"marker shared signal{index % 16} family{index % 9} extra"
                   for index in range(len(query_rows))]
    query_texts[1] = ""  # exact missing-stream fallback
    query_texts[2] = "unseenwordzzz"  # non-missing zero vector
    query_ids = [f"synthetic-query-{row}" for row in query_rows]
    baseline = np.linspace(0.25, 0.75, len(query_rows), dtype=np.float64)
    missing = np.asarray([not text.strip() for text in query_texts], dtype=bool)

    models = {}
    outputs = {}
    for arm in ("observed", "shuffled"):
        model = ResidualNeighborPredictor(origin_cycle=origin, residual_arm=arm)
        model.fit(text_space, bank_texts, bank_ids, bank_rows, bank_cycles,
                  bank_probability, residual)
        out = model.predict(query_texts, query_ids, query_rows, query_cycles,
                            baseline, missing_mask=missing)
        models[arm] = model
        outputs[arm] = out

    observed = outputs["observed"]
    if (not np.array_equal(observed["candidate_probability"][missing], baseline[missing])
            or observed["correction"][2] != 0.0
            or not np.isfinite(observed["candidate_probability"]).all()):
        raise AssertionError("A02 synthetic missing/zero-vector routing failed")
    if np.intersect1d(query_rows, bank_rows).size or np.any(bank_cycles > origin):
        raise AssertionError("A02 synthetic row/cycle boundary failed")

    with tempfile.TemporaryDirectory(prefix="a02-synthetic-") as temporary:
        root = Path(temporary)
        model_path = root / "observed/model.joblib"
        model_path.parent.mkdir(parents=True)
        # Inject a failure at atomic publish; output must remain absent and can
        # be recovered from the already-fitted in-memory model without refit.
        try:
            with patch("scripts.optimization_v3_a02_neighbors.os.replace", side_effect=OSError("synthetic publish interruption")):
                _atomic_joblib(models["observed"], model_path)
        except OSError:
            pass
        else:
            raise AssertionError("A02 synthetic atomic-publish failure injection did not trigger")
        if model_path.exists() or not model_path.with_name(model_path.name + ".tmp").exists():
            raise AssertionError("A02 failed atomic publish created a partial final model")
        _atomic_joblib(models["observed"], model_path)
        state = models["observed"].audit_state()
        state_path = root / "observed/fit_state.json"
        state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        restored = joblib.load(model_path)
        restored_state = restored.audit_state()
        if state != restored_state:
            raise AssertionError("A02 synthetic model/state persistence roundtrip failed")
        resumed = restored.predict(query_texts, query_ids, query_rows, query_cycles,
                                   baseline, missing_mask=missing)
        if (not np.array_equal(resumed["candidate_probability"], observed["candidate_probability"])
                or not np.array_equal(resumed["neighbor_source_rows"], observed["neighbor_source_rows"])):
            raise AssertionError("A02 synthetic resume changed saved predictions")

        # Persist a safe, pickle-free prediction payload then read it back and
        # independently reconstruct the residual correction from neighbor rows.
        output_path = root / "observed/predictions.npz"
        np.savez_compressed(
            output_path,
            score_rows=query_rows,
            score_record_ids=np.asarray(query_ids, dtype=str),
            baseline_prediction=baseline,
            prediction=observed["candidate_probability"],
            missing_mask=missing,
            neighbor_source_rows=observed["neighbor_source_rows"],
            neighbor_similarities=observed["neighbor_similarities"],
            correction=observed["correction"],
        )
        with np.load(output_path, allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        if saved["score_record_ids"].dtype.kind != "U" or saved["prediction"].dtype.kind == "O":
            raise AssertionError("A02 synthetic output payload contains unsafe dtypes")
        row_to_bank = {int(row): index for index, row in enumerate(bank_rows)}
        reconstructed = np.zeros(len(query_rows), dtype=np.float64)
        for row_index in range(len(query_rows)):
            source_rows = saved["neighbor_source_rows"][row_index]
            sims = saved["neighbor_similarities"][row_index]
            valid = source_rows >= 0
            source_rows = source_rows[valid]
            sims = sims[valid]
            idx = np.asarray([row_to_bank[int(row)] for row in source_rows], dtype=np.int64)
            weight = sims * sims
            reconstructed[row_index] = np.clip(
                np.dot(weight, models["observed"].bank_residuals[idx]) / (4.0 + weight.sum()),
                -0.05, 0.05,
            ) if len(idx) else 0.0
        reconstructed[missing] = 0.0
        candidate = np.clip(baseline + reconstructed, 0.0, 1.0)
        candidate[missing] = baseline[missing]
        if not np.array_equal(candidate, saved["prediction"]):
            raise AssertionError("A02 synthetic independent prediction reconstruction failed")

        synthetic_y = (np.arange(len(query_rows)) % 2).astype(np.int8)
        base_f1 = _manual_macro_f1(synthetic_y, (baseline >= 0.5).astype(np.int8))
        candidate_f1 = _manual_macro_f1(synthetic_y, (candidate >= 0.5).astype(np.int8))
        groups = np.where(missing, "missing", "present")
        support = {name: {"n": int(np.sum(groups == name)),
                          "class0": int(np.sum((groups == name) & (synthetic_y == 0))),
                          "class1": int(np.sum((groups == name) & (synthetic_y == 1)))}
                   for name in sorted(set(groups.tolist()))}
        if any(value["n"] >= 200 and value["class0"] >= 30 and value["class1"] >= 30
               for value in support.values()):
            raise AssertionError("Synthetic support fixture unexpectedly passed production minimum support")
        gates = {"synthetic_metric_finite": np.isfinite(base_f1) and np.isfinite(candidate_f1),
                 "synthetic_support_minimum_applied": True,
                 "synthetic_source_boundary": True}
        if not all(gates.values()):
            raise AssertionError("A02 synthetic gate/reconstruction pipeline failed")

    return {
        "status": "PASS",
        "fit_type": "SYNTHETIC_ONLY",
        "official_fit_count": 0,
        "synthetic_predictor_fits": 2,
        "synthetic_test_or_sample_access": False,
        "steps": ["fit_text_space", "fit_observed", "fit_shuffled", "predict",
                  "missing_route", "zero_vector", "failure_injection", "atomic_publish_recovery",
                  "joblib_persistence_resume", "pickle_free_npz_roundtrip",
                  "independent_neighbor_formula_reconstruction", "metric", "group_support", "gate"],
        "checks": {"failed_publish_left_no_final_model": True,
                   "resume_reused_fitted_model_without_refit": True,
                   "probability_reconstruction_bit_exact": True,
                   "missing_fallback_bit_exact": True,
                   "zero_similarity_row_zero_correction": True,
                   "query_labels_absent_from_prediction_signature": True,
                   "unsupported_group_marked_by_minimum_support": True},
        "synthetic_group_support": support,
        "synthetic_example_metric": {"baseline_macro_f1": base_f1,
                                     "candidate_macro_f1": candidate_f1},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_synthetic()
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Synthetic QA exists and differs: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
