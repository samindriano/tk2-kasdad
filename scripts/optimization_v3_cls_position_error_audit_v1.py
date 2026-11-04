"""Zero-fit paired error audit for frozen positional classification predictions.

This script reads only already-saved E67/E75 Train validation predictions,
official Train identity columns, and the Train validation targets after all
prediction receipts and row alignment checks pass. It never opens Test or
SampleSubmission and never fits an estimator or changes a threshold.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN = ROOT / "data/raw/classification/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
OUT = ROOT / "reports/optimization_v3/cls_position_error_audit_v1"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
THRESHOLD = 0.50
MARGIN_EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50)
MARGIN_LABELS = (
    "[0, 0.02)", "[0.02, 0.05)", "[0.05, 0.10)",
    "[0.10, 0.20)", "[0.20, 0.50]",
)
SIGN_TOLERANCE = 1e-12

RUNS = {
    "BIGRAM": ROOT / "experiments/runs/optimization_v3_research_next_v1/BIGRAM_POSITION_PILOT_V1",
    "CONTRAST": ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1",
    "MOMENTS": ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1",
}
FREEZES = {
    # Run roots retain predecessor freezes; select the accepted freeze that
    # actually binds each E67/E75 prediction bundle.
    "BIGRAM": ROOT / "experiments/runs/optimization_v3_research_next_v1/BIGRAM_POSITION_PILOT_V1/FREEZE_ACTIVE.json",
    "CONTRAST": ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1/PREFIT_FREEZE.json",
    "MOMENTS": ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1/FREEZE.json",
}

CANDIDATES = {
    "BIGRAM_BAG": ("BIGRAM", "route_bag"),
    "BIGRAM_REAL": ("BIGRAM", "route_real"),
    "BIGRAM_SHUFFLED": ("BIGRAM", "route_shuffled"),
    "CONTRAST_REAL": ("CONTRAST", "prob_real_blend"),
    "CONTRAST_SHUFFLED": ("CONTRAST", "prob_shuffled_blend"),
    "MOMENTS_REAL": ("MOMENTS", "prob_candidate_real"),
    "MOMENTS_SHUFFLED": ("MOMENTS", "prob_candidate_shuffled"),
}

SAFE_KEYS = {
    "BIGRAM": ("score_rows", "score_record_ids", "score_cycles", "present_mask",
               "no_known_bigram", "mix50", "q_bag", "route_bag", "q_real", "route_real",
               "q_shuffled", "route_shuffled"),
    "CONTRAST": ("score_rows", "score_record_ids", "score_cycles", "present_mask",
                 "prob_mix50", "prob_real_blend", "prob_shuffled_blend"),
    # Moments score_record_ids is object dtype in the frozen NPZ. Do not unpickle it.
    # Its exact identity is established by matching score_rows/cycles to Train and
    # to the safe Unicode IDs in the Bigram and Contrast bundles.
    "MOMENTS": ("score_rows", "score_cycles", "present_mask", "prob_mix50",
                "prob_candidate_real", "prob_candidate_shuffled"),
}

VIEWS = {
    67: {"H12_E67": (68, 79), "W4": (68, 71), "W5": (72, 75), "W6": (76, 79)},
    75: {"W6": (76, 79)},
}


class AuditError(RuntimeError):
    """Raised when frozen prediction inputs or row identity fail closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header_payload = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)},
                                sort_keys=True, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")).encode("utf-8")
    header_sha = hashlib.sha256(header_payload).hexdigest()
    digest = hashlib.sha256(header_sha.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object: {path}")
    return value


def _false_flag(data: Mapping[str, Any], candidates: Sequence[str], *, label: str) -> None:
    present = [key for key in candidates if key in data]
    if not present or any(data[key] is not False for key in present):
        raise AuditError(f"Required closed boundary is absent or false check failed: {label}")


def verify_source(method: str) -> dict[str, Any]:
    run_root = RUNS[method]
    ready_path = run_root / "PREDICTIONS_READY.json"
    prelabel_path = run_root / "PRELABEL_VERIFICATION.json"
    freeze_path = FREEZES[method]
    ready = read_json(ready_path)
    prelabel = read_json(prelabel_path)
    freeze = read_json(freeze_path)
    expected_card = {
        "BIGRAM": "CLS_BIGRAM_POSITION_PILOT_V1",
        "CONTRAST": "POS_CONTRAST_V1",
        "MOMENTS": "POS_MOMENTS_V1",
    }[method]
    if ready.get("card_id") != expected_card:
        raise AuditError(f"Unexpected frozen candidate identity for {method}")
    # POS_CONTRAST_V1's independent verifier schema intentionally binds to the
    # ready receipt by hash and omits card_id. Require that binding explicitly;
    # all other current verifier receipts carry their card_id directly.
    prelabel_card = prelabel.get("card_id")
    if prelabel_card is not None and prelabel_card != expected_card:
        raise AuditError(f"Unexpected prelabel verifier identity for {method}")
    if prelabel_card is None:
        if method != "CONTRAST":
            raise AuditError(f"Prelabel verifier does not bind to the frozen ready receipt for {method}")
    if prelabel.get("status") != "PASS":
        raise AuditError(f"Independent prelabel verification is not PASS for {method}")
    _false_flag(ready, ("query_labels_opened", "query_labels_read"), label=f"{method} predictions remain prelabel-closed")
    _false_flag(prelabel, ("query_labels_read",), label=f"{method} prelabel verifier did not read target")
    _false_flag(ready, ("test_opened", "test_or_sample_access"), label=f"{method} readiness has no Test/Sample access")
    _false_flag(prelabel, ("test_opened", "test_or_sample_access"), label=f"{method} verifier has no Test/Sample access")
    if (freeze.get("card_id") != expected_card
            or freeze.get("freeze_signature") != ready.get("freeze_signature")
            or ready.get("freeze_signature") != prelabel.get("freeze_signature")):
        raise AuditError(f"Ready, verifier, and accepted freeze signature mismatch for {method}")
    if EXPECTED_TRAIN_SHA not in json.dumps(freeze.get("input_sha256", {}), sort_keys=True):
        raise AuditError(f"Frozen {method} input does not attest to official Train")
    if (ready.get("input_sha256") is not None
            and freeze.get("input_sha256") != ready.get("input_sha256")):
        raise AuditError(f"Ready/freeze Train input hash mismatch for {method}")
    ready_sha = sha256_file(ready_path)
    bound_ready_sha = prelabel.get("predictions_ready_sha256")
    if bound_ready_sha is not None and bound_ready_sha != ready_sha:
        raise AuditError(f"Prelabel verifier is not bound to the exact ready receipt for {method}")
    if method == "BIGRAM":
        provenance = prelabel.get("verifier_provenance", {})
        successor = provenance.get("successor_verifier", {})
        verifier_path = ROOT / str(successor.get("path", ""))
        if (provenance.get("status") != "PASS"
                or provenance.get("freeze_signature") != ready.get("freeze_signature")
                or provenance.get("input_sha256") != ready.get("input_sha256")
                or not verifier_path.is_file()
                or sha256_file(verifier_path) != successor.get("sha256")):
            raise AuditError("Bigram independent verifier provenance does not bind the active freeze/input")
    elif bound_ready_sha != ready_sha:
        raise AuditError(f"Prelabel verifier is not bound to the exact ready receipt for {method}")
    _false_flag(freeze, ("query_labels_read", "query_labels_opened"), label=f"{method} fit freeze kept query labels closed")
    _false_flag(freeze, ("test_opened", "test_or_sample_access"), label=f"{method} fit freeze kept Test/Sample closed")

    predictions: dict[int, dict[str, np.ndarray]] = {}
    prediction_records: dict[str, Any] = {}
    for origin in (67, 75):
        entry = ready.get("predictions", {}).get(f"E{origin}")
        if not isinstance(entry, dict):
            raise AuditError(f"Missing {method} E{origin} prediction receipt")
        expected_name = f"predictions_E{origin}.npz"
        expected_dir = (run_root / f"E{origin}").resolve()
        path = (ROOT / entry["path"]).resolve()
        if path.parent != expected_dir or path.name != expected_name or not path.is_file():
            raise AuditError(f"{method} E{origin} prediction path escapes its frozen run location")
        artifact_sha = sha256_file(path)
        if artifact_sha != entry.get("sha256"):
            raise AuditError(f"Frozen {method} E{origin} prediction SHA mismatch")
        with np.load(path, allow_pickle=False) as archive:
            missing_keys = sorted(set(SAFE_KEYS[method]) - set(archive.files))
            if missing_keys:
                raise AuditError(f"{method} E{origin} safe NPZ keys missing: {missing_keys}")
            arrays = {key: np.asarray(archive[key]).copy() for key in SAFE_KEYS[method]}
        predictions[origin] = arrays
        prediction_records[f"E{origin}"] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": artifact_sha,
            "rows": int(len(arrays["score_rows"])),
            "score_rows_sha256": sha256_array(arrays["score_rows"]),
            "score_cycles_sha256": sha256_array(arrays["score_cycles"]),
            "record_ids_sha256": sha256_array(arrays["score_record_ids"].astype(str)) if "score_record_ids" in arrays else None,
            "moments_object_ids_unpickled": False if method == "MOMENTS" else None,
        }
        if int(entry.get("rows", len(arrays["score_rows"]))) != len(arrays["score_rows"]):
            raise AuditError(f"{method} E{origin} row count differs from ready receipt")
        if (entry.get("score_cycles_sha256") is not None
                and sha256_array(arrays["score_cycles"]) != entry["score_cycles_sha256"]):
            raise AuditError(f"{method} E{origin} cycles differ from the ready receipt")
        if (entry.get("score_rows_sha256") is not None
                and sha256_array(arrays["score_rows"]) != entry["score_rows_sha256"]):
            raise AuditError(f"{method} E{origin} rows differ from the ready receipt")
        declared_ids_sha = entry.get("record_ids_sha256", entry.get("score_record_ids_sha256"))
        # Moments' saved ID digest uses an object-array serialization. Keep its
        # IDs unopened; row positions/cycles/availability are checked to Train.
        if (declared_ids_sha is not None and method != "MOMENTS"
                and sha256_array(arrays["score_record_ids"].astype(str)) != declared_ids_sha):
            raise AuditError(f"{method} E{origin} IDs differ from the ready receipt")
        if method == "BIGRAM":
            origin_receipt = prelabel.get("origins", {}).get(f"E{origin}", {})
            if sha256_array(arrays["score_rows"]) != origin_receipt.get("query_rows_sha256"):
                raise AuditError(f"Bigram E{origin} row positions differ from the independent verifier")
            arms = {"BAG": "bag", "REAL": "real", "SHUFFLED": "shuffled"}
            for arm, suffix in arms.items():
                arm_receipt = origin_receipt.get("all_three_arms", {}).get(arm, {})
                if (sha256_array(arrays[f"q_{suffix}"]) != arm_receipt.get("query_probability_sha256")
                        or sha256_array(arrays[f"route_{suffix}"]) != arm_receipt.get("route_probability_sha256")):
                    raise AuditError(f"Bigram E{origin} {arm} probabilities differ from verifier receipt")
    return {
        "method": method,
        "card_id": expected_card,
        "freeze_signature": ready["freeze_signature"],
        "ready_sha256": sha256_file(ready_path),
        "prelabel_sha256": sha256_file(prelabel_path),
        "freeze_sha256": sha256_file(freeze_path),
            "freeze_path": freeze_path.relative_to(ROOT).as_posix(),
        "source_commit": ready.get("source_commit"),
        "predictions": predictions,
        "prediction_receipts": prediction_records,
    }


def validate_alignment(sources: Mapping[str, Mapping[str, Any]], train_sha: str) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    identity = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code"], dtype={"record_id": str})
    if (len(identity) != 50_608 or identity.record_id.isna().any()
            or identity.record_id.duplicated().any() or sha256_file(TRAIN) != train_sha):
        raise AuditError("Official Train identity frame failed its frozen row/hash/ID checks")
    train_ids = identity.record_id.astype(str).to_numpy(dtype=str)
    train_cycles = pd.to_numeric(identity.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(train_cycles), np.arange(47, 80)):
        raise AuditError("Official Train cycle support changed")

    aligned: dict[int, dict[str, Any]] = {}
    for origin in (67, 75):
        bigram = sources["BIGRAM"]["predictions"][origin]
        contrast = sources["CONTRAST"]["predictions"][origin]
        moments = sources["MOMENTS"]["predictions"][origin]
        rows = np.asarray(bigram["score_rows"], dtype=np.int64)
        ids = bigram["score_record_ids"].astype(str)
        cycles = np.asarray(bigram["score_cycles"], dtype=np.int64)
        present = np.asarray(bigram["present_mask"], dtype=bool)
        if (rows.ndim != 1 or len(np.unique(rows)) != len(rows)
                or np.any(rows < 0) or np.any(rows >= len(train_ids))
                or not np.all(np.diff(rows) > 0)):
            raise AuditError(f"Bigram E{origin} score-row positions are invalid")
        if not np.array_equal(ids, train_ids[rows]) or not np.array_equal(cycles, train_cycles[rows]):
            raise AuditError(f"Bigram E{origin} saved identity does not match official Train row positions")

        for method, bundle in (("CONTRAST", contrast), ("MOMENTS", moments)):
            if (not np.array_equal(rows, np.asarray(bundle["score_rows"], dtype=np.int64))
                    or not np.array_equal(cycles, np.asarray(bundle["score_cycles"], dtype=np.int64))
                    or not np.array_equal(present, np.asarray(bundle["present_mask"], dtype=bool))):
                raise AuditError(f"{method} E{origin} is not exactly row/cycle/availability matched")
            if "score_record_ids" in bundle and not np.array_equal(ids, bundle["score_record_ids"].astype(str)):
                raise AuditError(f"{method} E{origin} saved IDs differ from Bigram safe Unicode IDs")
        if len(rows) != (20_903 if origin == 67 else 8_491):
            raise AuditError(f"Unexpected E{origin} score population")
        expected_cycles = np.arange(68, 80) if origin == 67 else np.arange(76, 80)
        if not np.array_equal(np.unique(cycles), expected_cycles):
            raise AuditError(f"Unexpected E{origin} score-cycle support")
        aligned[origin] = {
            "score_rows": rows,
            "record_ids": ids,
            "cycles": cycles,
            "present": present,
            "missing": ~present,
            "positions_sha256": sha256_array(rows),
            "record_ids_sha256": sha256_array(ids),
            "cycles_sha256": sha256_array(cycles),
            "present_mask_sha256": sha256_array(present),
            "present_rows": int(present.sum()),
            "missing_rows": int((~present).sum()),
        }

        baseline_arrays = {
            "BIGRAM": bigram["mix50"],
            "CONTRAST": contrast["prob_mix50"],
            "MOMENTS": moments["prob_mix50"],
        }
        max_diffs = {name: float(np.max(np.abs(np.asarray(value, dtype=np.float64)
                                                 - np.asarray(baseline_arrays["BIGRAM"], dtype=np.float64)), initial=0.0))
                     for name, value in baseline_arrays.items()}
        if any(value > 1e-12 for value in max_diffs.values()):
            raise AuditError(f"E{origin} frozen MIX50 values are not identical across sources: {max_diffs}")
        aligned[origin]["mix50_max_abs_diffs_by_source"] = max_diffs

        mix = np.asarray(baseline_arrays["BIGRAM"], dtype=np.float64)
        candidate_probabilities = {
            "BIGRAM_BAG": bigram["route_bag"],
            "BIGRAM_REAL": bigram["route_real"],
            "BIGRAM_SHUFFLED": bigram["route_shuffled"],
            "CONTRAST_REAL": contrast["prob_real_blend"],
            "CONTRAST_SHUFFLED": contrast["prob_shuffled_blend"],
            "MOMENTS_REAL": moments["prob_candidate_real"],
            "MOMENTS_SHUFFLED": moments["prob_candidate_shuffled"],
        }
        for name, value in candidate_probabilities.items():
            probability = np.asarray(value, dtype=np.float64)
            if (probability.shape != mix.shape or not np.isfinite(probability).all()
                    or np.any((probability < 0.0) | (probability > 1.0))):
                raise AuditError(f"E{origin} {name} probability array is invalid")
            if not np.array_equal(probability[~present], mix[~present]):
                raise AuditError(f"E{origin} {name} missing-stream fallback differs from MIX50")
        no_known = np.asarray(bigram["no_known_bigram"], dtype=bool)
        for name in ("BIGRAM_BAG", "BIGRAM_REAL", "BIGRAM_SHUFFLED"):
            probability = np.asarray(candidate_probabilities[name], dtype=np.float64)
            if not np.array_equal(probability[present & no_known], mix[present & no_known]):
                raise AuditError(f"E{origin} {name} no-known-bigram fallback differs from MIX50")
        aligned[origin]["missing_fallback_exact_all_candidates"] = True
        aligned[origin]["bigram_no_known_fallback_exact"] = True
        aligned[origin]["mix50"] = mix
        aligned[origin]["candidates"] = {name: np.asarray(value, dtype=np.float64)
                                           for name, value in candidate_probabilities.items()}

    e67, e75 = aligned[67], aligned[75]
    e67_w6 = e67["cycles"] >= 76
    if (int(e67_w6.sum()) != 8_491
            or not np.array_equal(e67["record_ids"][e67_w6], e75["record_ids"])
            or not np.array_equal(e67["score_rows"][e67_w6], e75["score_rows"])
            or not np.array_equal(e67["cycles"][e67_w6], e75["cycles"])
            or not np.array_equal(e67["present"][e67_w6], e75["present"])):
        raise AuditError("E67 W6 and E75 W6 are not the same 8,491 score rows")
    return aligned, identity


def margin_bin_indices(margin: Any) -> np.ndarray:
    values = np.asarray(margin, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 0.5)):
        raise ValueError("Margins must be finite and in [0, 0.5]")
    return np.searchsorted(np.asarray(MARGIN_EDGES[1:-1]), values, side="right")


def validate_aligned_rows(ids_left: Any, ids_right: Any, cycles_left: Any, cycles_right: Any) -> None:
    if (not np.array_equal(np.asarray(ids_left).astype(str), np.asarray(ids_right).astype(str))
            or not np.array_equal(np.asarray(cycles_left), np.asarray(cycles_right))):
        raise ValueError("Prediction rows must match by exact ID and cycle order")


def probability_metrics(y: Any, probability: Any) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if (len(target) != len(p) or len(target) == 0 or not np.isin(target, (0, 1)).all()
            or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("Metric inputs must be aligned finite binary targets and probabilities")
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(p, epsilon, 1.0 - epsilon)
    logloss = -np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    two_classes = len(np.unique(target)) == 2
    auc = float(roc_auc_score(target, p)) if two_classes else None
    return {
        "n": int(len(target)),
        "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "logloss": float(logloss),
        "brier": float(np.mean((p - target) ** 2)),
        "roc_auc": auc,
        "logloss_clip_epsilon": float(epsilon),
    }


def paired_flip_counts(y: Any, base_probability: Any, candidate_probability: Any) -> dict[str, int]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    base = np.asarray(base_probability, dtype=np.float64).reshape(-1) >= THRESHOLD
    candidate = np.asarray(candidate_probability, dtype=np.float64).reshape(-1) >= THRESHOLD
    if len(target) != len(base) or len(target) != len(candidate):
        raise ValueError("Paired labels and prediction arrays must align")
    base_correct = base == target
    candidate_correct = candidate == target
    helpful = ~base_correct & candidate_correct
    harmful = base_correct & ~candidate_correct
    false_positive_base = (target == 0) & base
    false_positive_candidate = (target == 0) & candidate
    false_negative_base = (target == 1) & ~base
    false_negative_candidate = (target == 1) & ~candidate
    return {
        "n": int(len(target)),
        "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "unchanged_labels": int(np.sum(base == candidate)),
        "flipped_labels": int(np.sum(base != candidate)),
        "wrong_to_correct": int(helpful.sum()),
        "correct_to_wrong": int(harmful.sum()),
        "wrong_to_correct_y0": int(np.sum(helpful & (target == 0))),
        "wrong_to_correct_y1": int(np.sum(helpful & (target == 1))),
        "correct_to_wrong_y0": int(np.sum(harmful & (target == 0))),
        "correct_to_wrong_y1": int(np.sum(harmful & (target == 1))),
        "false_positive_base": int(false_positive_base.sum()),
        "false_positive_candidate": int(false_positive_candidate.sum()),
        "false_positive_delta": int(false_positive_candidate.sum() - false_positive_base.sum()),
        "false_negative_base": int(false_negative_base.sum()),
        "false_negative_candidate": int(false_negative_candidate.sum()),
        "false_negative_delta": int(false_negative_candidate.sum() - false_negative_base.sum()),
    }


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "logloss_candidate_minus_mix50": float(candidate["logloss"] - baseline["logloss"]),
        "brier_candidate_minus_mix50": float(candidate["brier"] - baseline["brier"]),
        "roc_auc_candidate_minus_mix50": (
            None if candidate["roc_auc"] is None or baseline["roc_auc"] is None
            else float(candidate["roc_auc"] - baseline["roc_auc"])
        ),
    }


def _append_jsonable(out: list[dict[str, Any]], value: Mapping[str, Any]) -> None:
    out.append(dict(value))


def score_audit(aligned: Mapping[int, Mapping[str, Any]], target_by_origin: Mapping[int, np.ndarray]) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    margins: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    sign_summaries: list[dict[str, Any]] = []

    for origin in (67, 75):
        data = aligned[origin]
        y_all = np.asarray(target_by_origin[origin], dtype=np.int8)
        cycles = np.asarray(data["cycles"], dtype=np.int64)
        present = np.asarray(data["present"], dtype=bool)
        group_masks = {"all": np.ones(len(y_all), dtype=bool),
                       "present": present, "missing": ~present}
        view_masks = {
            name: (cycles >= low) & (cycles <= high)
            for name, (low, high) in VIEWS[origin].items()
        }
        baseline_probability = np.asarray(data["mix50"], dtype=np.float64)
        methods = data["candidates"]

        for view_name, view_mask in view_masks.items():
            for group_name, group_mask in group_masks.items():
                use = view_mask & group_mask
                if not use.any():
                    continue
                baseline_metric = probability_metrics(y_all[use], baseline_probability[use])
                for method, candidate_probability in methods.items():
                    candidate_metric = probability_metrics(y_all[use], candidate_probability[use])
                    _append_jsonable(scores, {
                        "origin": origin, "view": view_name, "availability": group_name,
                        "method": method, "baseline": baseline_metric,
                        "candidate": candidate_metric,
                        "delta": _metric_delta(candidate_metric, baseline_metric),
                    })

            for method, candidate_probability in methods.items():
                for group_name, group_mask in group_masks.items():
                    use = view_mask & group_mask
                    if not use.any():
                        continue
                    base_y, base_p, cand_p = y_all[use], baseline_probability[use], candidate_probability[use]
                    all_counts = paired_flip_counts(base_y, base_p, cand_p)
                    _append_jsonable(flips, {
                        "origin": origin, "view": view_name, "availability": group_name,
                        "method": method, "true_class": "all", **all_counts,
                    })
                    for cls in (0, 1):
                        class_rows = base_y == cls
                        class_counts = paired_flip_counts(base_y[class_rows], base_p[class_rows], cand_p[class_rows])
                        _append_jsonable(flips, {
                            "origin": origin, "view": view_name, "availability": group_name,
                            "method": method, "true_class": cls, **class_counts,
                        })

                    base_margin = np.abs(base_p - THRESHOLD)
                    bins = margin_bin_indices(base_margin)
                    for index, label in enumerate(MARGIN_LABELS):
                        selected = bins == index
                        if not selected.any():
                            continue
                        margin_counts = paired_flip_counts(base_y[selected], base_p[selected], cand_p[selected])
                        _append_jsonable(margins, {
                            "origin": origin, "view": view_name, "availability": group_name,
                            "method": method, "incumbent_margin_bin": label,
                            "margin_lower_inclusive": MARGIN_EDGES[index],
                            "margin_upper_inclusive": bool(index == len(MARGIN_LABELS) - 1),
                            "margin_upper": MARGIN_EDGES[index + 1],
                            **margin_counts,
                        })

        for cycle in sorted(np.unique(cycles).tolist()):
            mask = cycles == cycle
            baseline = probability_metrics(y_all[mask], baseline_probability[mask])
            for method, candidate_probability in methods.items():
                candidate = probability_metrics(y_all[mask], candidate_probability[mask])
                _append_jsonable(per_cycle, {
                    "origin": origin, "cycle": int(cycle), "method": method,
                    "baseline": baseline, "candidate": candidate,
                    "delta": _metric_delta(candidate, baseline),
                })

        for view_name, (low, high) in VIEWS[origin].items():
            cycles_in_view = [cycle for cycle in sorted(np.unique(cycles).tolist()) if low <= cycle <= high]
            for method in methods:
                rows = [r for r in per_cycle if r["origin"] == origin and r["method"] == method
                        and low <= r["cycle"] <= high]
                item: dict[str, Any] = {"origin": origin, "view": view_name, "method": method,
                                        "cycle_count": len(cycles_in_view)}
                for metric, favorable_sign in (("logloss", -1), ("brier", -1), ("roc_auc", 1)):
                    key = {"logloss": "logloss_candidate_minus_mix50",
                           "brier": "brier_candidate_minus_mix50",
                           "roc_auc": "roc_auc_candidate_minus_mix50"}[metric]
                    deltas = [row["delta"][key] for row in rows if row["delta"][key] is not None]
                    favorable = sum(value * favorable_sign > SIGN_TOLERANCE for value in deltas)
                    unfavorable = sum(value * favorable_sign < -SIGN_TOLERANCE for value in deltas)
                    tied = len(deltas) - favorable - unfavorable
                    item[metric] = {"available_cycles": len(deltas), "favorable": int(favorable),
                                    "unfavorable": int(unfavorable), "ties": int(tied),
                                    "direction": "lower_is_better" if favorable_sign == -1 else "higher_is_better"}
                sign_summaries.append(item)

    history: list[dict[str, Any]] = []
    e67, e75 = aligned[67], aligned[75]
    y67, y75 = target_by_origin[67], target_by_origin[75]
    if not np.array_equal(y67[e67["cycles"] >= 76], y75):
        raise AuditError("E67 W6 and E75 W6 targets differ after exact row alignment")
    for method in ["MIX50", *CANDIDATES]:
        p67 = e67["mix50"] if method == "MIX50" else e67["candidates"][method]
        p75 = e75["mix50"] if method == "MIX50" else e75["candidates"][method]
        mask67 = e67["cycles"] >= 76
        for group_name, group_mask in (("all", np.ones(len(y75), dtype=bool)),
                                       ("present", e75["present"]), ("missing", e75["missing"])):
            if not group_mask.any():
                continue
            abs_shift = np.abs(p75[group_mask] - p67[mask67][group_mask])
            history_flips = paired_flip_counts(y75[group_mask], p67[mask67][group_mask], p75[group_mask])
            history.append({
                "method": method, "availability": group_name,
                "n": int(group_mask.sum()),
                "mean_abs_probability_change_E67_to_E75": float(abs_shift.mean()),
                "median_abs_probability_change_E67_to_E75": float(np.median(abs_shift)),
                "threshold_label_changes": history_flips["flipped_labels"],
                "E67_correct_to_E75_wrong": history_flips["correct_to_wrong"],
                "E67_wrong_to_E75_correct": history_flips["wrong_to_correct"],
                "E67_to_E75_label_unchanged": history_flips["unchanged_labels"],
            })

    return {
        "schema_version": 1,
        "status": "PASS_ZERO_FIT_TRAIN_TEMPORAL_ERROR_AUDIT",
        "adaptive_development_analysis": True,
        "threshold": THRESHOLD,
        "fixed_incumbent_margin_edges": list(MARGIN_EDGES),
        "fixed_incumbent_margin_labels": list(MARGIN_LABELS),
        "scores": scores,
        "paired_flips": flips,
        "incumbent_margin_flips": margins,
        "per_cycle": per_cycle,
        "per_cycle_sign_summaries": sign_summaries,
        "matched_history_E67W6_to_E75W6": history,
        "counts": {
            "score_comparisons": len(scores), "paired_flip_rows": len(flips),
            "margin_rows": len(margins), "per_cycle_rows": len(per_cycle),
        },
    }


def _lookup(items: Sequence[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    matches = [item for item in items if all(item.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise AuditError(f"Expected one metric row for {keys}, found {len(matches)}")
    return matches[0]


def _format_delta(value: float | None, *, positive_good: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.5f}"


def _make_plots(result: Mapping[str, Any]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    flip_rows = result["paired_flips"]
    methods = list(CANDIDATES)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained", sharey=True)
    for ax, origin in zip(axes, (67, 75)):
        labels, values = [], []
        for method in methods:
            row = _lookup(flip_rows, origin=origin, view="W6", availability="all",
                          method=method, true_class="all")
            labels.append(method.replace("_", " "))
            values.append((row["wrong_to_correct"], row["correct_to_wrong"]))
        y = np.arange(len(labels))
        helpful = np.asarray([row[0] for row in values])
        harmful = np.asarray([row[1] for row in values])
        ax.barh(y, helpful, color="#287a57", label="Wrong → correct")
        ax.barh(y, -harmful, color="#b34d4d", label="Correct → wrong")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(f"E{origin} W6; n=8,491, threshold=0.50")
        ax.set_xlabel("Paired label flips vs MIX50 (negative = harmful)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Correctness changes against the same MIX50 incumbent")
    plot1 = OUT / "w6_correctness_flips.png"
    fig.savefig(plot1, dpi=160)
    plt.close(fig)

    per_cycle = result["per_cycle"]
    plot_methods = ["BIGRAM_REAL", "BIGRAM_BAG", "BIGRAM_SHUFFLED",
                    "CONTRAST_REAL", "MOMENTS_REAL"]
    colors = {name: color for name, color in zip(plot_methods,
                                                 ("#1b7837", "#2166ac", "#762a83", "#e08214", "#5e3c99"))}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained", sharey=True)
    for ax, origin in zip(axes, (67, 75)):
        for method in plot_methods:
            rows = sorted((item for item in per_cycle if item["origin"] == origin
                           and item["method"] == method and 76 <= item["cycle"] <= 79),
                          key=lambda item: item["cycle"])
            x = [row["cycle"] for row in rows]
            y = [row["delta"]["logloss_candidate_minus_mix50"] for row in rows]
            ax.plot(x, y, marker="o", linewidth=1.7, label=method.replace("_", " "), color=colors[method])
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks([76, 77, 78, 79])
        ax.set_title(f"E{origin} W6")
        ax.set_xlabel("Cycle")
    axes[0].set_ylabel("Within-cycle logloss delta (candidate − MIX50)\nnegative is favorable")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("Per-cycle logloss on the exact same W6 score rows")
    plot2 = OUT / "w6_per_cycle_logloss_deltas.png"
    fig.savefig(plot2, dpi=160)
    plt.close(fig)
    return [plot1.relative_to(ROOT).as_posix(), plot2.relative_to(ROOT).as_posix()]


def _render_report(result: Mapping[str, Any], *, provenance: Mapping[str, Any],
                   aligned: Mapping[int, Mapping[str, Any]], input_train_sha: str,
                   ledger_sha: str, plots: Sequence[str]) -> str:
    lines = [
        "# Zero-fit positional classification error audit",
        "",
        "**Status:** completed on frozen E67/E75 temporal Train predictions. No estimator fit, threshold change, Test scoring, SampleSubmission access, or Kaggle action occurred.",
        "",
        "This is adaptive development analysis: the Bigram pilot verdicts and previous POS/Contrast/Moments results were already visible. E67 and E75 overlap; their W6 comparison uses the identical 8,491 rows and is descriptive, not an independent replication.",
        "",
        "## Provenance and row support",
        "",
        f"Official Train SHA-256: `{input_train_sha}`. Prediction bundles, accepted freeze receipts, and prelabel-verifier bindings are hashed in `AUDIT.json`; all three methods passed source checks before this audit read targets.",
        "",
        "| Evaluation view | Rows | Present | Missing | Positive / negative targets |",
        "|---|---:|---:|---:|---:|",
    ]
    for origin in (67, 75):
        data = aligned[origin]
        y = np.asarray(data["y"], dtype=np.int8)
        cycles = np.asarray(data["cycles"], dtype=np.int64)
        for name, (low, high) in VIEWS[origin].items():
            use = (cycles >= low) & (cycles <= high)
            present = use & data["present"]
            missing = use & ~data["present"]
            lines.append(f"| E{origin} {name} | {int(use.sum()):,} | {int(present.sum()):,} | {int(missing.sum()):,} | {int(y[use].sum()):,} / {int(use.sum()-y[use].sum()):,} |")
    lines.extend([
        "",
        f"E67 W6 and E75 W6 exact row-position SHA: `{sha256_array(aligned[67]['score_rows'][aligned[67]['cycles'] >= 76])}`; ID order, cycles, availability, and targets match for all 8,491 rows.",
        "Moments' frozen NPZ stores `score_record_ids` as object dtype; this audit did not unpickle it. Exact alignment was verified with integer `score_rows`, cycle, availability, and official Train identity columns against Bigram and Contrast Unicode-ID bundles.",
        "",
        "## Paired W6 probability metrics",
        "",
        "Deltas are candidate minus MIX50 for logloss and Brier (negative is favorable), and candidate minus MIX50 for ROC-AUC (positive is favorable). Each comparison uses exactly the same rows; the present-only table isolates rows with available text.",
        "",
        "| Method | E67 W6 Δ logloss | Δ Brier | Δ AUC | E75 W6 Δ logloss | Δ Brier | Δ AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in CANDIDATES:
        e67 = _lookup(result["scores"], origin=67, view="W6", availability="all", method=method)
        e75 = _lookup(result["scores"], origin=75, view="W6", availability="all", method=method)
        d67, d75 = e67["delta"], e75["delta"]
        lines.append(f"| {method} | {_format_delta(d67['logloss_candidate_minus_mix50'], positive_good=False)} | {_format_delta(d67['brier_candidate_minus_mix50'], positive_good=False)} | {_format_delta(d67['roc_auc_candidate_minus_mix50'], positive_good=True)} | {_format_delta(d75['logloss_candidate_minus_mix50'], positive_good=False)} | {_format_delta(d75['brier_candidate_minus_mix50'], positive_good=False)} | {_format_delta(d75['roc_auc_candidate_minus_mix50'], positive_good=True)} |")
    lines.extend([
        "",
        "Present-only results, all E67/E75 views, and score denominators are in `AUDIT.json`. On missing-stream rows the frozen policy routes every candidate exactly to MIX50; these rows therefore contribute zero paired flips and zero probability-score delta.",
        "",
        "## W6 threshold flips and errors",
        "",
        "At the frozen threshold 0.50, `wrong → correct` counts are useful flips and `correct → wrong` counts are harmful flips relative to MIX50. FP/FN deltas are candidate minus MIX50.",
        "",
        "| Origin | Method | n | Unchanged | Flipped | Wrong→correct (y=0 / y=1) | Correct→wrong (y=0 / y=1) | ΔFP | ΔFN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for origin in (67, 75):
        for method in CANDIDATES:
            item = _lookup(result["paired_flips"], origin=origin, view="W6", availability="all",
                           method=method, true_class="all")
            lines.append(f"| E{origin} | {method} | {item['n']:,} | {item['unchanged_labels']:,} | {item['flipped_labels']:,} | {item['wrong_to_correct_y0']:,} / {item['wrong_to_correct_y1']:,} | {item['correct_to_wrong_y0']:,} / {item['correct_to_wrong_y1']:,} | {item['false_positive_delta']:+,} | {item['false_negative_delta']:+,} |")
    real = _lookup(result["paired_flips"], origin=75, view="W6", availability="present",
                   method="BIGRAM_REAL", true_class="all")
    lines.extend([
        "",
        f"For E75 W6 Bigram REAL, present rows contribute {real['n']:,} of 8,491; {real['flipped_labels']:,} labels change, with {real['wrong_to_correct']:,} useful and {real['correct_to_wrong']:,} harmful flips. Full true-class and availability splits for every method/view are in `AUDIT.json`.",
        "",
        "## Incumbent-margin bins and cycle signs",
        "",
        "Margins are binned only from `abs(MIX50 - 0.50)` using the fixed bins `[0,0.02)`, `[0.02,0.05)`, `[0.05,0.10)`, `[0.10,0.20)`, `[0.20,0.50]`. Candidate probabilities never determine bin membership. The complete denominators, useful/harmful flips, class counts, and unchanged/flipped rows are in `AUDIT.json`.",
        "",
        "Per-cycle logloss, Brier, and AUC deltas plus favorable/unfavorable/tied cycle counts are also in `AUDIT.json`. Cycles are descriptive slices, not independent tests.",
        "",
        "## What was already tried",
        "",
        "The earlier POS/A02/MIX50 zero-fit audit (`scripts/analyze_posmix_frontier_v1.py`, `reports/optimization_v3/posmix_followup_v1/ERROR_AUDIT.json`) already reported aggregate/per-cycle scores and agreement/fix-break counts against POS and A02. The A02 decision-boundary audit used different fixed probability bands. Neither decomposed Bigram REAL/BAG/SHUFFLED flips against MIX50 by true class, availability, and incumbent margin. Contrast and Moments had prior aggregate pilot gates, preserved unchanged here.",
        "",
        "The Bigram pilot verdict remains **FAIL / DIAGNOSTIC_ONLY**. Its original Macro F1 values are unchanged: E67 H12 REAL 0.812143 vs MIX50 0.811937 (Δ +0.000206; required ≥ +0.000500), and E75 W6 REAL 0.814617 vs MIX50 0.816099 (Δ −0.001482; required ≥ +0.000500 and REAL is below BAG/SHUFFLED). See `reports/optimization_v3/bigram_position_pilot_v1/RESULTS.md`. This audit does not reopen or relax any gate, and makes no causal, significance, or Test-generalization claim.",
        "",
        "## Next lane recommendation",
        "",
        result["next_lane_recommendation"],
        "",
        "## Visuals",
        "",
    ])
    for plot in plots:
        lines.extend([f"![{Path(plot).stem}]({Path(plot).name})", ""])
    lines.extend([
        "## Reproducibility and boundaries",
        "",
        "Run `python scripts/optimization_v3_cls_position_error_audit_v1.py`. It verifies accepted freeze signatures/input hashes, verifier bindings, prediction-file hashes and paths, exact row alignment, then reads only `record_id`, `cycle_code`, and query `artifact_signal` from official Train. It never opens Test/SampleSubmission. Existing models, recipes, thresholds, and historical gate records remain unchanged.",
        f"Ledger SHA-256 before/after: `{ledger_sha}` (unchanged). New fits: 0. Kaggle accessed: no.",
        "",
        f"Audit script SHA-256: `{result['source_script_sha256']}`. Latest repo HEAD at analysis: `{result['source_commit']}`.",
    ])
    return "\n".join(lines) + "\n"


def _choose_next_lane(result: Mapping[str, Any]) -> str:
    """Record the conservative post-audit lane decision without authorizing work."""
    e67 = _lookup(result["scores"], origin=67, view="W6", availability="all", method="BIGRAM_REAL")
    e75 = _lookup(result["scores"], origin=75, view="W6", availability="all", method="BIGRAM_REAL")
    e67_flips = _lookup(result["paired_flips"], origin=67, view="W6", availability="all",
                        method="BIGRAM_REAL", true_class="all")
    e75_flips = _lookup(result["paired_flips"], origin=75, view="W6", availability="all",
                        method="BIGRAM_REAL", true_class="all")
    signs67 = _lookup(result["per_cycle_sign_summaries"], origin=67, view="W6",
                      method="BIGRAM_REAL")
    signs75 = _lookup(result["per_cycle_sign_summaries"], origin=75, view="W6",
                      method="BIGRAM_REAL")
    ll_key = "logloss_candidate_minus_mix50"
    delta67 = e67["delta"][ll_key]
    delta75 = e75["delta"][ll_key]
    if delta67 < 0 and delta75 < 0:
        loss_summary = f"logloss improves slightly at both origins (Δ {delta67:+.6f} at E67; {delta75:+.6f} at E75)"
    elif delta67 > 0 and delta75 > 0:
        loss_summary = f"logloss worsens at both origins (Δ {delta67:+.6f} at E67; {delta75:+.6f} at E75)"
    else:
        loss_summary = f"logloss is mixed across origins (Δ {delta67:+.6f} at E67; {delta75:+.6f} at E75)"

    def flip_balance(row: Mapping[str, Any]) -> str:
        useful = int(row["wrong_to_correct"])
        harmful = int(row["correct_to_wrong"])
        if harmful > useful:
            return f"harmful exceed helpful by {harmful - useful} ({harmful} vs {useful})"
        if useful > harmful:
            return f"helpful exceed harmful by {useful - harmful} ({useful} vs {harmful})"
        return f"helpful and harmful flips tie at {useful} each"

    cycle_summary = (
        f"within-cycle logloss improves in {signs67['logloss']['favorable']}/"
        f"{signs67['logloss']['available_cycles']} E67 cycles and "
        f"{signs75['logloss']['favorable']}/{signs75['logloss']['available_cycles']} E75 cycles"
    )
    return (
        "**No additional fit lane is justified by this audit.** Bigram REAL's "
        f"{loss_summary}, but {cycle_summary}. At the unchanged 0.50 threshold, "
        f"E67 W6 flips are {flip_balance(e67_flips)} and E75 W6 flips are {flip_balance(e75_flips)}. "
        "Those local diagnostics do not overturn the existing Macro F1 gate failures. The views "
        "are adaptive and overlap, so preserve historical verdicts and wait for review; any later "
        "fit needs a separately predeclared and authorized question. This audit starts no follow-up work."
    )


def main() -> int:
    ledger_before = sha256_file(LEDGER)
    train_sha = sha256_file(TRAIN)
    if train_sha != EXPECTED_TRAIN_SHA:
        raise AuditError("Official Classification Train SHA differs from frozen pilot inputs")
    sources = {name: verify_source(name) for name in RUNS}
    aligned, identity = validate_alignment(sources, train_sha)

    # Query targets are read only after all source receipts, bundle hashes, IDs,
    # row positions, cycles, and availability masks pass.
    target_frame = pd.read_csv(TRAIN, usecols=["artifact_signal"])
    targets = pd.to_numeric(target_frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    if len(targets) != 50_608 or not np.isin(targets, (0, 1)).all():
        raise AuditError("Official Train target is not a 50,608-row binary vector")
    target_by_origin: dict[int, np.ndarray] = {}
    for origin, data in aligned.items():
        target_by_origin[origin] = targets[np.asarray(data["score_rows"], dtype=np.int64)]
        data["y"] = target_by_origin[origin]
    result = score_audit(aligned, target_by_origin)
    result.update({
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          check=True, capture_output=True, text=True).stdout.strip(),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "input_train_sha256": train_sha,
        "input_train_rows": len(identity),
        "input_train_target_sha256": sha256_array(targets),
        "sources": {name: {key: value for key, value in source.items() if key != "predictions"}
                    for name, source in sources.items()},
        "alignment": {
            str(origin): {key: value for key, value in data.items()
                          if key not in {"mix50", "candidates", "y", "score_rows",
                                         "record_ids", "cycles", "present", "missing"}}
            for origin, data in aligned.items()
        },
        "fit_count": 0,
        "test_accessed": False,
        "sample_submission_accessed": False,
        "kaggle_accessed": False,
        "threshold_changed": False,
        "historical_gate_verdicts_changed": False,
        "ledger_sha256_before": ledger_before,
    })
    # This text is a conservative synthesis of the fixed diagnostics, not an
    # authorization for another fit or score.
    result["next_lane_recommendation"] = _choose_next_lane(result)
    plots = _make_plots(result)
    result["plots"] = plots
    if sha256_file(LEDGER) != ledger_before:
        raise AuditError("Shared fit ledger changed during this zero-fit audit")
    result["ledger_sha256_after"] = ledger_before
    result["ledger_unchanged"] = True

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                ensure_ascii=False, allow_nan=False) + "\n",
                                     encoding="utf-8")
    report = _render_report(result, provenance=sources, aligned=aligned,
                            input_train_sha=train_sha, ledger_sha=ledger_before, plots=plots)
    (OUT / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": result["status"], "train_rows": result["input_train_rows"],
                      "views": result["counts"], "ledger_unchanged": result["ledger_unchanged"],
                      "output": str((OUT / "RESULTS.md").relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
