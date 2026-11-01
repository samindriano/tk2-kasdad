"""Independent read-only verification for the completed N04 domain-only pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
TRAIN_PATH = Path(task_files("regression")["train"])
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
CURRENT_FREEZE = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_02/freeze.json"
CURRENT_PREFLIGHT = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_02/preflight.json"
PREVIOUS_FREEZE = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_01/freeze.json"
RESULTS_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N04/amendment_02/results.json"
REPORT_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N04/amendment_02/verification.json"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
CURRENT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N04/amendment_02"
PREVIOUS_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N04/amendment_01"
RECOVERY_NAME = {
    (55, 0): ("previous", "N04_PILOT_55_domain_even"),
    (55, 1): ("current", "N04_PILOT_55_domain_odd"),
    (67, 0): ("current", "N04_PILOT_67_domain_even"),
    (67, 1): ("current", "N04_PILOT_67_domain_odd"),
}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_hash(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype else np.asarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def pairwise_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive = probabilities[labels == 1]
    negative = np.sort(probabilities[labels == 0], kind="mergesort")
    if not len(positive) or not len(negative):
        raise ValueError("AUC needs both domain classes")
    wins = 0.0
    for score in positive:
        left = int(np.searchsorted(negative, score, side="left"))
        right = int(np.searchsorted(negative, score, side="right"))
        wins += left + .5 * (right - left)
    return float(wins / (len(positive) * len(negative)))


def row_hash(cycles: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(cycles <= cutoff)
    fit_cycles = cycles[rows]
    distinct = np.unique(fit_cycles)
    recent = set(distinct[-4:].tolist())
    labels = np.isin(fit_cycles, list(recent)).astype(np.int8)
    return rows, labels


def load_receipt_bundle(path: Path, expected_identity: dict[str, Any], task: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path, receipt_path = path / "arrays.npz", path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    integrity = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_hash(integrity):
        raise RuntimeError(f"Receipt hash mismatch: {path}")
    if receipt.get("identity") != expected_identity or receipt.get("identity_sha256") != canonical_hash(expected_identity):
        raise RuntimeError(f"Bundle identity mismatch: {path}")
    if receipt.get("task") != task or receipt.get("npz_sha256") != digest_file(npz_path):
        raise RuntimeError(f"Bundle task or NPZ hash mismatch: {path}")
    if receipt.get("metadata_sha256") != canonical_hash(receipt["metadata"]):
        raise RuntimeError(f"Bundle metadata hash mismatch: {path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if sorted(arrays) != receipt.get("array_names"):
        raise RuntimeError(f"Bundle array names differ from receipt: {path}")
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise RuntimeError(f"Object array found in bundle: {path}")
    return arrays, receipt


def verify() -> dict[str, Any]:
    frozen = json.loads(CURRENT_FREEZE.read_text(encoding="utf-8"))
    preflight = json.loads(CURRENT_PREFLIGHT.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_FREEZE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    identity = frozen["identity"]
    if frozen.get("freeze_signature") != canonical_hash(identity):
        raise RuntimeError("N04 amendment_02 freeze signature mismatch")
    if (frozen["execution_context"].get("preflight_sha256") != digest_file(CURRENT_PREFLIGHT)
            or preflight.get("freeze_signature") != frozen["freeze_signature"]):
        raise RuntimeError("N04 amendment_02 preflight hash/signature mismatch")
    if previous.get("freeze_signature") != identity["inherited_fit_audit"]["previous_freeze_signature"]:
        raise RuntimeError("N04 inherited amendment_01 signature mismatch")
    source_commit = frozen["execution_context"]["source_commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                      cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("N04 frozen source commit is not an ancestor of current HEAD")

    hashes_checked = {}
    for relative, expected in identity["input_and_source_sha256"].items():
        path = ROOT / Path(relative)
        actual = digest_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input/source hash changed: {relative}")
        hashes_checked[relative] = actual
    if digest_file(TRAIN_PATH) != identity["official_train_sha256"]:
        raise RuntimeError("Official Train hash differs from N04 freeze")
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    if digest_file(QUEUE_PATH) != identity["queue_sha256"] or queue["raw_train_sha256"]["regression"] != identity["official_train_sha256"]:
        raise RuntimeError("N04 queue or raw Train lineage differs from freeze")

    columns = ["record_id", "cycle_code", "signal_stream"]
    frame = pd.read_csv(TRAIN_PATH, usecols=columns)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    if len(frame) != 50_608 or not frame.record_id.is_unique:
        raise RuntimeError("Official Train row count or record_id uniqueness changed")
    all_folds = []
    per_cutoff: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows, full_labels = row_hash(cycles, cutoff)
        fold_parts = []
        fold_reports = []
        for parity in (0, 1):
            current_signature = frozen["freeze_signature"]
            bundle_identity = {
                "batch_id": "optimization_v3_research_next_v1", "card_id": "N04",
                "freeze_signature": current_signature, "stage": "PILOT",
                "cutoff": cutoff, "component": "domain",
                "heldout_cycle_parity": "odd" if parity else "even",
            }
            old = (cutoff, parity) == (55, 0)
            if old:
                bundle_identity["freeze_signature"] = previous["freeze_signature"]
                bundle_root = PREVIOUS_ROOT
            else:
                bundle_root = CURRENT_ROOT
            component = "domain_even" if parity == 0 else "domain_odd"
            path = bundle_root / f"fit{cutoff}" / component
            arrays, receipt = load_receipt_bundle(path, bundle_identity, "classification")
            rows = arrays["score_rows"].astype(np.int64)
            fit_pos = np.searchsorted(fit_rows, rows)
            if np.any(fit_pos >= len(fit_rows)) or not np.array_equal(fit_rows[fit_pos], rows):
                raise RuntimeError("Domain heldout rows escaped its Train fit prefix")
            expected_labels = full_labels[fit_pos]
            held_cycles = cycles[rows]
            labels = arrays["domain_label"].astype(np.int8)
            probability = arrays["prediction"].astype(np.float64)
            if (not np.array_equal(arrays["score_record_ids"].astype(str),
                                   frame.record_id.to_numpy(dtype=str)[rows])
                    or not np.array_equal(arrays["cycle_code"].astype(np.int64), held_cycles)
                    or not np.array_equal(labels, expected_labels)
                    or np.any(held_cycles % 2 != parity)
                    or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise RuntimeError(f"Domain row/ID/cycle/label audit failed for {cutoff}/{parity}")

            diagnostics = receipt["metadata"]["diagnostics"]
            train_mask = cycles[fit_rows] % 2 != parity
            train_rows = fit_rows[train_mask]
            train_labels = full_labels[train_mask]
            if (len(train_rows) != diagnostics["fit_rows"]
                    or array_hash(train_rows, "<i8") != diagnostics["fit_row_indices_sha256"]
                    or array_hash(rows, "<i8") != diagnostics["heldout_row_indices_sha256"]
                    or array_hash(train_labels, "<i8") != diagnostics["classifier_state"]["fit_domain_label_sha256"]
                    or array_hash(frame.record_id.to_numpy(dtype=str)[train_rows].astype("U"))
                    != diagnostics["classifier_state"]["fit_record_id_sha256"]):
                raise RuntimeError(f"Domain fit population audit failed for {cutoff}/{parity}")
            prep = diagnostics["preprocessor_state"]
            if (prep["cycle_code_used"] or prep["record_id_used"] or prep["task_target_used"]
                    or diagnostics["cycle_code_used_as_feature"]
                    or diagnostics["record_id_used_as_feature"]
                    or diagnostics["task_target_used"]):
                raise RuntimeError("N04 domain learner used a prohibited feature or target")
            expected_parameters = {
                key: value for key, value in identity["recipe"]["domain_model"].items()
                if key != "class"
            }
            if diagnostics["classifier_state"]["parameters"] != expected_parameters:
                raise RuntimeError("N04 domain LogisticRegression parameters differ from frozen recipe")
            auc = pairwise_auc(labels, probability)
            if not np.isclose(auc, diagnostics["independent_heldout_auc"], rtol=0, atol=1e-14):
                raise RuntimeError(f"Independent pairwise AUC mismatch for {cutoff}/{parity}")
            if old and not np.isclose(
                    auc, preflight.get("inherited_fit_audit", {}).get("independent_heldout_auc", auc),
                    rtol=0, atol=1e-14):
                raise RuntimeError("Inherited fold AUC differs from amendment_02 preflight audit")
            recovery_area, recovery_stem = RECOVERY_NAME[(cutoff, parity)]
            recovery_root = PREVIOUS_ROOT / "recovery" if recovery_area == "previous" else CURRENT_ROOT / "recovery"
            recovery_npz = recovery_root / f"{recovery_stem}.npz"
            recovery_json = recovery_root / f"{recovery_stem}.json"
            recovery = json.loads(recovery_json.read_text(encoding="utf-8"))
            expected_recovery_identity = bundle_identity
            if (recovery.get("identity") != expected_recovery_identity
                    or recovery.get("npz_sha256") != digest_file(recovery_npz)):
                raise RuntimeError(f"Recovery receipt failed for {cutoff}/{parity}")
            with np.load(recovery_npz, allow_pickle=False) as archive:
                recovery_arrays = {key: archive[key].copy() for key in archive.files}
            if set(recovery_arrays) != set(arrays) or any(
                    not np.array_equal(recovery_arrays[key], arrays[key]) for key in arrays):
                raise RuntimeError(f"Recovery arrays differ from published bundle for {cutoff}/{parity}")
            fold_parts.append((rows, labels, probability))
            fold_reports.append({
                "parity": "odd" if parity else "even", "score_rows": int(len(rows)),
                "fit_rows": int(len(train_rows)), "auc_pairwise": auc,
                "bundle_receipt_sha256": receipt["receipt_sha256"],
                "bundle_npz_sha256": digest_file(path / "arrays.npz"),
                "recovery_matches_bundle": True, "inherited_output_reused_without_refit": old,
            })
            all_folds.append((cutoff, parity, receipt["receipt_sha256"]))

        joined_rows = np.concatenate([part[0] for part in fold_parts])
        order = np.argsort(joined_rows, kind="mergesort")
        joined_rows = joined_rows[order]
        if len(joined_rows) != len(np.unique(joined_rows)) or not np.array_equal(joined_rows, fit_rows):
            raise RuntimeError(f"OOF domain rows do not cover exact fit prefix at cutoff {cutoff}")
        labels = np.concatenate([part[1] for part in fold_parts])[order]
        probability = np.concatenate([part[2] for part in fold_parts])[order]
        source = labels == 0
        clipped_probability = np.clip(probability, 1e-6, 1 - 1e-6)
        odds = clipped_probability / (1 - clipped_probability)
        source_raw = odds[source]
        source_weight = np.clip(source_raw, .25, 4.0)
        weights = np.ones(len(labels), dtype=np.float64)
        weights[source] = source_weight
        weights /= weights.mean()
        ess = float(weights.sum() ** 2 / np.square(weights).sum())
        clip_fraction = float(np.mean((source_raw <= .25) | (source_raw >= 4.0)))
        fold_auc = [item["auc_pairwise"] for item in fold_reports]
        report_cutoff = results["pre_task_gates"]["by_cutoff"][str(cutoff)]
        report_weights = report_cutoff["weight_diagnostics"]
        if (array_hash(weights, "<f8") != report_weights["weight_sha256"]
                or not np.isclose(ess / len(weights), report_weights["effective_sample_size_fraction"],
                                 rtol=0, atol=1e-12)
                or not np.isclose(clip_fraction, report_weights["source_fraction_at_or_beyond_clip_bounds"],
                                 rtol=0, atol=1e-12)
                or not np.isclose(weights.mean(), report_weights["weight_mean"], rtol=0, atol=1e-12)
                or fold_auc != report_cutoff["heldout_fold_aucs"]):
            raise RuntimeError(f"Independent overlap/weight diagnostics differ at cutoff {cutoff}")
        passed = all(.55 <= value <= .95 for value in fold_auc) \
            and ess / len(weights) >= .5 and clip_fraction <= .2 \
            and np.isfinite(weights).all() and np.all(weights > 0)
        if passed != report_cutoff["pass"]:
            raise RuntimeError(f"Independent pre-task gate verdict differs at cutoff {cutoff}")
        per_cutoff[str(cutoff)] = {
            "fit_rows": int(len(fit_rows)), "source_rows": int(source.sum()),
            "recent_rows": int((~source).sum()), "overall_oof_auc_pairwise": pairwise_auc(labels, probability),
            "folds": fold_reports, "fold_aucs": fold_auc,
            "effective_sample_size_fraction": ess / len(weights),
            "source_clip_fraction": clip_fraction, "weight_mean": float(weights.mean()),
            "pre_task_gates_pass": bool(passed),
        }

    ledger_events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    n04_events = [event for event in ledger_events if event.get("card_id") == "N04"]
    reservation = [event for event in n04_events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == "N04:PILOT"]
    starts = [event for event in n04_events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == "N04:PILOT"]
    saved = [event for event in n04_events if event.get("event") == "FIT_SAVED"
             and event.get("reservation_key") == "N04:PILOT"]
    failed = [event for event in n04_events if event.get("event") == "FIT_FAILED"
              and event.get("reservation_key") == "N04:PILOT"]
    released = [event for event in n04_events if event.get("event") == "RESERVATION_UNUSED_RELEASED"
                and event.get("reservation_key") == "N04:PILOT"]
    if (len(reservation) != 1 or reservation[0]["fits"] != 8
            or len(starts) != 4 or len(saved) != 4 or failed
            or len(released) != 1 or released[0]["unused_fits"] != 4):
        raise RuntimeError("N04 budget ledger does not reconcile to four saved domain fits and four unused task fits")
    if any(event.get("component") not in {"domain_lr_even", "domain_lr_odd"} for event in starts):
        raise RuntimeError("Unexpected N04 task estimator was started")
    task_paths = [CURRENT_ROOT / f"fit{cutoff}" / component
                  for cutoff in (55, 67) for component in ("weighted", "within_cycle_shuffle")]
    if any(path.exists() for path in task_paths):
        raise RuntimeError("A prohibited N04 task bundle exists despite failed pre-task gate")
    if results.get("status") != "SKIPPED_SCIENTIFIC_PREFLIGHT" \
            or results["pre_task_gates"]["all_cutoffs_pass"] is not False:
        raise RuntimeError("N04 report status does not match the failed frozen pre-task gate")
    if results.get("official_test_used") is not False or results.get("kaggle_calls") != 0:
        raise RuntimeError("N04 report declares prohibited Test/Kaggle use")
    if not any(not value["pre_task_gates_pass"] for value in per_cutoff.values()):
        raise RuntimeError("N04 independent audit did not reproduce a pre-task failure")
    return {
        "schema_version": 1, "status": "PASS",
        "verification_method": "separate read-only script; pairwise AUC and direct full-prefix label/weight reconstruction",
        "freeze_signature": frozen["freeze_signature"], "source_commit": source_commit,
        "input_source_hashes_verified": len(hashes_checked),
        "official_train_rows": int(len(frame)), "official_train_sha256": digest_file(TRAIN_PATH),
        "inherited_fit_audit": identity["inherited_fit_audit"],
        "cutoffs": per_cutoff,
        "ledger": {
            "N04_reserved": int(reservation[0]["fits"]), "N04_started": len(starts),
            "N04_saved": len(saved), "N04_failed": len(failed),
            "N04_released_unused": int(released[0]["unused_fits"]),
            "N04_task_fits_started": 0,
        },
        "verdict": "SKIPPED_SCIENTIFIC_PREFLIGHT; the cutoff-55 heldout domain AUC gate failed. No task model was fit.",
        "test_or_kaggle_used": False,
    }


def main() -> int:
    result = verify()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if REPORT_PATH.exists():
        existing = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(result):
            raise RuntimeError("Refusing to overwrite N04 independent verification report")
    else:
        with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
