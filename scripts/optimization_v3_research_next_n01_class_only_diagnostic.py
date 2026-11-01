"""Four-fit, diagnostic-only N01 class-only continuation amendment."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from scripts import optimization_v3_research_next_n00 as n00
from scripts import optimization_v3_research_next_n01 as n01
from src.models.classification_models import _text_values
from src.models.research_next_budget import (
    DEFAULT_CAPS, authorize_new_phase, finish_fit, ledger_summary, read_events,
    release_unused, reserve_card, start_fit,
)
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle, resume_or_publish_bundle
from src.models.research_next_n01 import (
    FEATURE_NAMES, MINIMUM_CLASS_BIN_N, PARAMS, THRESHOLD,
    class_by_volume_weight_plan, route_expert_probabilities,
)
from src.models.research_next_n01_class_only_diagnostic import (
    CONTINUATION_CUTOFFS, N01ClassOnlyDiagnosticClassifier,
)
from src.models.round1_models import _number_array
from src.models import research_next_n01 as n01_model
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
CARD_ID = "N01_CLASS_ONLY_DIAGNOSTIC"
STAGE = "CONTINUATION"
AMENDMENT = "amendment_01"
PILOT_ORIGINS = (55, 67)
ALL_ORIGINS = (55, 59, 63, 67, 71, 75)
N_NEW_FITS = 4
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
THREADS = 2
MIN_GROUP_N = 200
MIN_EACH_CLASS_N = 30
MAX_F1_HARM = 0.01
MAX_RECALL_HARM = 0.02
FREEZE_ROOT = ROOT / f"experiments/optimization_v3/research_next_v1/frozen/{CARD_ID}/{AMENDMENT}"
FREEZE_PATH = FREEZE_ROOT / "freeze.json"
REPORT_ROOT = ROOT / f"reports/optimization_v3/research_next_v1/results/{CARD_ID}/{AMENDMENT}"
BUNDLE_ROOT = ROOT / f"experiments/runs/optimization_v3_research_next_v1/{CARD_ID}/{AMENDMENT}"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
AUDIT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N01_N02_GUARD_AUDIT/amendment_01"
SYNTHETIC_RECEIPT_PATH = REPORT_ROOT / "synthetic_preflight.json"


class ResourceDeferred(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_once(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite immutable diagnostic evidence: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import io

    columns = list(rows[0]) if rows else []
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite immutable CSV evidence: {path}")
        return
    path.write_bytes(payload)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT,
                          check=False, capture_output=True).returncode == 0


def _hash_array(values: Any, *, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def _dependency_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "psutil")
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _source_and_input_paths(train_path: Path) -> list[Path]:
    paths = [
        ROOT / "experiments/optimization_v3/research_next_v1/queue.json",
        ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json",
        n00.FREEZE_PATH, n00.REPORT_ROOT / "summary.json",
        n01.FREEZE_PATH,
        AUDIT_ROOT / "results.json", AUDIT_ROOT / "groups.csv", AUDIT_ROOT / "verification.json",
        ROOT / "reports/optimization_v3/research_next_v1/results/N01/gate.json",
        ROOT / "reports/optimization_v3/research_next_v1/results/N01/results.json",
        ROOT / "reports/optimization_v3/research_next_v1/results/N02/amendment_02/gate.json",
        ROOT / "src/models/research_next_n01.py",
        ROOT / "src/models/research_next_n01_class_only_diagnostic.py",
        ROOT / "src/models/research_next_contract.py",
        ROOT / "src/models/research_next_budget.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/optimization_v3_batch1_c16.py",
        ROOT / "scripts/optimization_v3_research_next_n00.py",
        ROOT / "scripts/optimization_v3_research_next_n01.py",
        ROOT / "scripts/optimization_v3_research_next_n01_class_only_diagnostic.py",
        ROOT / "scripts/optimization_v3_research_next_n01_class_only_diagnostic_verify.py",
        ROOT / "tests/test_research_next_n01_class_only_diagnostic.py",
        ROOT / "scripts/optimization_v3_batch1_c16.py",
        ROOT / "src/utils/paths.py",
        train_path,
    ]
    for cutoff in PILOT_ORIGINS:
        pilot = n01._bundle_path(cutoff, "class_only")
        paths.extend((pilot / "arrays.npz", pilot / "receipt.json"))
    for cutoff in ALL_ORIGINS:
        cache = n00.cache_path("C16", cutoff)
        paths.extend((cache, cache.with_suffix(".json")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"N01 class-only diagnostic lineage inputs are missing: {missing}")
    return sorted(set(paths), key=lambda path: _relative(path))


def _resource_estimate(fit_features: pd.DataFrame, score_features: pd.DataFrame,
                       *, available_bytes: int | None = None) -> dict[str, Any]:
    def sparse_upper(frame: pd.DataFrame) -> tuple[int, int]:
        docs, _missing = _text_values(frame["signal_stream"])
        tokens = np.fromiter((len(re.findall(r"(?u)\b\w\w+\b", doc.lower())) for doc in docs),
                             dtype=np.int64, count=len(docs))
        nnz = int(np.sum(tokens + np.maximum(tokens - 1, 0), dtype=np.int64))
        return nnz, int(nnz * 16 + (len(docs) + 1) * 8)

    fit_nnz, fit_sparse_bytes = sparse_upper(fit_features)
    score_nnz, score_sparse_bytes = sparse_upper(score_features)
    peak = int(1.5 * 1024**3 + 3 * fit_sparse_bytes + 2 * score_sparse_bytes)
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    return {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)), "score_rows": int(len(score_features)),
        "fit_text_nnz_upper": fit_nnz, "score_text_nnz_upper": score_nnz,
        "fit_sparse_bytes_upper": fit_sparse_bytes, "score_sparse_bytes_upper": score_sparse_bytes,
        "estimated_peak_bytes": peak, "estimated_peak_is_observed": False,
        "available_physical_bytes": available, "allowed_bytes": max(0, int(allowed)),
        "minimum_free_bytes": MINIMUM_FREE_BYTES, "reserve_bytes": FREE_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES, "threads": THREADS, "concurrent_training_jobs": 1,
    }


def _train_context() -> dict[str, Any]:
    n00_frozen = n00.verify_freeze()
    n01_frozen, parent = n01._verify_freeze()
    frame = parent["frame"]
    target = parent["target"].astype(np.int8)
    cycles = parent["cycles"].astype(np.int64)
    if not np.array_equal(cycles, frame["cycle_code"].to_numpy(dtype=np.int64)):
        raise RuntimeError("N01 diagnostic cycle array differs from official Train source rows")
    audit_report = json.loads((AUDIT_ROOT / "results.json").read_text(encoding="utf-8"))
    audit_verification = json.loads((AUDIT_ROOT / "verification.json").read_text(encoding="utf-8"))
    if (audit_verification.get("status") != "PASS_INDEPENDENT_ZERO_FIT_VERIFICATION"
            or audit_report.get("status") != "PASS_ZERO_FIT_AUDIT"
            or audit_report["continuation_gates"]["N01_class_only_diagnostic_candidate"].get("eligible_for_authorized_continuation") is not True
            or audit_report["continuation_gates"]["N02_sketch_candidate"].get("eligible_for_authorized_continuation") is not False):
        raise RuntimeError("Cached all-view gate audit does not authorize only the N01 class-only diagnostic")
    if audit_report.get("historical_verdicts_changed") is not False:
        raise RuntimeError("Gate audit does not preserve the historical N01/N02 verdicts")

    queue = parent["queue"]
    card = queue["cards"]
    queue_n01 = next(item for item in card if item["id"] == "N01")
    if queue_n01["recipe"]["policy"] != ".75 frozen C2_B + .25 weighted sparse expert on present; frozen CB on missing":
        raise RuntimeError("N01 frozen availability route changed")
    _docs, missing = _text_values(frame["signal_stream"])
    controls: dict[int, dict[str, Any]] = {}
    plans: dict[int, dict[str, Any]] = {}
    fit_rows: dict[int, np.ndarray] = {}
    resources: dict[int, dict[str, Any]] = {}
    pilot_bundles: dict[int, dict[str, np.ndarray]] = {}
    pilot_receipts: dict[int, dict[str, Any]] = {}

    for cutoff in ALL_ORIGINS:
        c16 = n00._load_artifact("C16", cutoff, frame)
        reconstructed = n00.reconstruct_classification(c16, missing)
        arrays = c16["arrays"]
        rows = c16["rows"].astype(np.int64)
        if not np.array_equal(rows, np.flatnonzero((cycles > cutoff) & (cycles <= min(cutoff + (12 if cutoff in PILOT_ORIGINS else 4), 79)))):
            raise RuntimeError(f"C16/{cutoff} control cache has unexpected continuation score rows")
        controls[cutoff] = {
            "rows": rows,
            "record_ids": arrays["score_record_ids"].astype(str),
            "cycles": c16["cycles"].astype(np.int64),
            "c2b": arrays["control__c2b_probability"].astype(np.float64),
            "cb": arrays["control__cb_probability"].astype(np.float64),
            "sparse_raw": arrays["control__baseline_sparse_raw"].astype(np.float64),
            "current": reconstructed["control"].astype(np.float64),
            "missing": reconstructed["stream_missing"].astype(bool),
            "npz_sha256": c16["npz_sha256"],
            "sidecar_sha256": c16["sidecar_sha256"],
        }
        if not np.array_equal(controls[cutoff]["record_ids"], frame.iloc[rows]["record_id"].astype(str).to_numpy()):
            raise RuntimeError(f"C16/{cutoff} record IDs do not match the Train row index")
        if cutoff in CONTINUATION_CUTOFFS:
            fit = np.flatnonzero(cycles <= cutoff).astype(np.int64)
            if not len(fit) or int(cycles[fit].max()) != cutoff:
                raise RuntimeError(f"N01 diagnostic fit prefix does not end at cutoff {cutoff}")
            fit_features = frame.iloc[fit].loc[:, list(FEATURE_NAMES)]
            plan = class_by_volume_weight_plan(fit_features, target[fit], minimum_cell_n=MINIMUM_CLASS_BIN_N)
            if plan["diagnostics"]["score_rows_used_for_edges_or_weights"] is not False:
                raise RuntimeError("N01 diagnostic weight plan used score rows")
            score_mask = (controls[cutoff]["cycles"] > cutoff) & (controls[cutoff]["cycles"] <= min(cutoff + 4, 79))
            score_rows = controls[cutoff]["rows"][score_mask]
            score_features = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)]
            fit_rows[cutoff] = fit
            plans[cutoff] = plan
            resources[cutoff] = _resource_estimate(fit_features, score_features)
        else:
            loaded = n01._load_existing_fit(parent, cutoff, "class_only")
            if loaded is None:
                raise RuntimeError(f"N01 pilot class-only bundle missing for origin {cutoff}")
            arrays_pilot = loaded[0]
            expected = controls[cutoff]
            if (not np.array_equal(arrays_pilot["score_rows"], expected["rows"])
                    or not np.array_equal(arrays_pilot["score_record_ids"].astype(str), expected["record_ids"])
                    or not np.allclose(arrays_pilot["baseline_prediction"], expected["current"], rtol=0, atol=1e-12)):
                raise RuntimeError(f"N01 pilot class-only bundle differs from official C16 controls at {cutoff}")
            pilot_bundles[cutoff] = arrays_pilot
            pilot_receipts[cutoff] = loaded[1]

    paths = _source_and_input_paths(parent["train_path"])
    source_hashes = {_relative(path): file_sha256(path) for path in paths}
    return {
        "n00_freeze": n00_frozen, "n01_freeze": n01_frozen,
        "parent_context": parent, "frame": frame, "target": target, "cycles": cycles,
        "train_path": parent["train_path"], "queue": queue, "audit_report": audit_report,
        "audit_verification": audit_verification, "controls": controls,
        "plans": plans, "fit_rows": fit_rows, "resources": resources,
        "pilot_bundles": pilot_bundles, "source_hashes": source_hashes,
        "pilot_receipts": pilot_receipts,
    }


def _stable_identity(context: Mapping[str, Any]) -> dict[str, Any]:
    c = context["controls"]
    plans = context["plans"]
    fit_rows = context["fit_rows"]
    score_rows: dict[str, np.ndarray] = {}
    for cutoff in CONTINUATION_CUTOFFS:
        control = c[cutoff]
        mask = (control["cycles"] > cutoff) & (control["cycles"] <= min(cutoff + 4, 79))
        score_rows[str(cutoff)] = control["rows"][mask]
    recipe = {
        "model_source": "N01 ClassByVolumeSparseClassifier inherited fit/predict/state methods; only constructor permits 59/63/71/75",
        "weight_kind": "class_only; per-class mean of frozen class-by-volume weights; mean-one; equal class total mass",
        "weight_plan": "fit-only text_load quantiles [.25,.50,.75], collapse duplicate edges, side=right; cell weights sqrt(n_fit/(2*G*n_class_bin)), clip [.5,2], normalize mean-one; class-only is per-class candidate-weight mean, renormalized mean-one",
        "minimum_class_bin_n": MINIMUM_CLASS_BIN_N,
        "fit_features": list(FEATURE_NAMES), "estimator_parameters": dict(PARAMS),
        "structured_transform": "existing RegressionTextModel structured preprocessing, fit on cutoff prefix only",
        "text_transform": "existing TF-IDF parameters and fitted vocabulary/IDF, fit on cutoff prefix only",
        "policy": ".75 frozen C2_B + .25 class-only sparse expert when signal_stream is present; exact frozen CB on missing",
        "threshold": float(THRESHOLD),
        "score_population": "only the next four complete cycles per newly fitted origin; existing pilot 55/67 bundles reused",
        "official_new_fits": N_NEW_FITS,
        "matched_control_refits": 0,
        "test_sample_submission_kaggle_used": False,
    }
    fit_info = {}
    for cutoff in CONTINUATION_CUTOFFS:
        fit = fit_rows[cutoff]
        plan = plans[cutoff]
        fit_info[str(cutoff)] = {
            "fit_rows": int(len(fit)),
            "fit_rows_sha256": _hash_array(fit, dtype="<i8"),
            "fit_target_sha256": _hash_array(context["target"][fit], dtype="i1"),
            "score_rows": int(len(score_rows[str(cutoff)])),
            "score_rows_sha256": _hash_array(score_rows[str(cutoff)], dtype="<i8"),
            "class_only_weights_sha256": _hash_array(plan["class_only_weights"], dtype="<f8"),
            "weight_diagnostics": n01._stable_bin_summary(plan),
        }
    return {
        "batch_id": BATCH, "card_id": CARD_ID, "amendment": AMENDMENT,
        "stage": STAGE, "source_commit": _head(),
        "parent_N00_freeze_signature": context["n00_freeze"]["freeze_signature"],
        "parent_N01_freeze_signature": context["n01_freeze"]["freeze_signature"],
        "cached_gate_audit_signature": context["audit_report"]["audit_signature"],
        "official_train_sha256": file_sha256(context["train_path"]),
        "source_and_input_sha256": context["source_hashes"],
        "dependencies": _dependency_versions(), "recipe": recipe,
        "fit_and_score_rows": fit_info,
        "reused_pilot_origins": {
            str(cutoff): {
                "fit_receipt_sha256": context["pilot_receipts"][cutoff]["receipt_sha256"],
                "score_rows_sha256": _hash_array(context["pilot_bundles"][cutoff]["score_rows"], dtype="<i8"),
                "fit_refit": False,
            } for cutoff in PILOT_ORIGINS
        },
        "C16_control_cache_hashes": {
            str(cutoff): {"npz": c[cutoff]["npz_sha256"], "sidecar": c[cutoff]["sidecar_sha256"]}
            for cutoff in ALL_ORIGINS
        },
        "views": context["queue"]["views"]["full_six"],
        "evaluation_gates": context["queue"]["gates"]["classification"],
        "diagnostic_only": True,
    }


def _budget_preflight(*, require_unreserved: bool) -> dict[str, Any]:
    events = read_events(LEDGER_PATH)
    summary = ledger_summary(LEDGER_PATH)
    authorizations = [event for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(authorizations) != 1 or int(authorizations[0].get("cap", 0)) != 60:
        raise RuntimeError("Research Next fit allowance is missing or no longer matches the authorized cap")
    category_caps = authorizations[0].get("category_caps", {})
    if int(category_caps.get("continuation", 0)) != 24:
        raise RuntimeError("Research Next continuation fit allowance changed")
    key = f"{CARD_ID}:{STAGE}"
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == key]
    if len(reservations) > 1:
        raise RuntimeError("Duplicate N01 diagnostic fit reservation exists")
    if require_unreserved and reservations:
        raise RuntimeError("N01 diagnostic reservation exists before the frozen prefit checkpoint")
    if reservations and (int(reservations[0].get("fits", -1)) != N_NEW_FITS
                         or reservations[0].get("category") != "continuation"):
        raise RuntimeError("Existing N01 diagnostic reservation differs from the four-fit authorization")
    released_by_key = {event.get("reservation_key"): int(event.get("unused_fits", 0))
                       for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"}
    reserved_total = sum(int(event["fits"]) for event in events if event.get("event") == "CARD_FITS_RESERVED")
    active_total = reserved_total - sum(released_by_key.values())
    active_continuation = sum(
        int(event["fits"]) - released_by_key.get(event["reservation_key"], 0)
        for event in events if event.get("event") == "CARD_FITS_RESERVED"
        and event.get("category") == "continuation"
    )
    added_total = 0 if reservations else N_NEW_FITS
    added_category = 0 if reservations else N_NEW_FITS
    if active_total + added_total > 60 or active_continuation + added_category > 24:
        raise RuntimeError("The four N01 diagnostic fits do not fit the live shared allowance")
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    pending = [start for start in starts if not any(
        terminal.get("fit_id") == start.get("fit_id")
        and terminal.get("event") in {"FIT_SAVED", "FIT_FAILED"}
        for terminal in events
    )]
    own_ids = {event.get("fit_id") for event in starts if event.get("card_id") == CARD_ID}
    allowed_ids = {f"{CARD_ID}:{STAGE}:{cutoff}:class_only" for cutoff in CONTINUATION_CUTOFFS}
    if not own_ids.issubset(allowed_ids):
        raise RuntimeError("Unexpected N01 diagnostic estimator fit exists in the ledger")
    if require_unreserved and (len(starts) != 26 or len(saved) != 26 or failed or pending):
        raise RuntimeError("Research Next ledger is not at the expected 26-saved-fit pre-amendment checkpoint")
    if len(starts) > 26 + N_NEW_FITS or len(saved) > 26 + N_NEW_FITS or failed:
        raise RuntimeError("Research Next ledger fit count/status exceeds the authorized diagnostic scope")
    return {
        "ledger_sha256": file_sha256(LEDGER_PATH), "event_rows": summary["event_rows"],
        "started": len(starts), "saved": len(saved), "failed": len(failed),
        "pending": len(pending), "reserved_total": reserved_total,
        "released_unused": summary["released_unused"], "active_reserved": active_total,
        "active_continuation_reserved": active_continuation,
        "future_reserved_total_if_new": active_total + added_total,
        "future_continuation_if_new": active_continuation + added_category,
        "authorized_cap": 60, "continuation_cap": 24,
        "this_amendment_fits": N_NEW_FITS,
        "existing_matching_reservation": bool(reservations),
    }


def _execution_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_physical_bytes": int(psutil.virtual_memory().available),
        "resource_estimates_by_cutoff": context["resources"],
        "budget_ledger": _budget_preflight(require_unreserved=True),
        "official_estimator_fits": 0,
        "synthetic_fit_required_before_official_run": True,
        "test_files_opened": 0, "sample_submission_opened": 0, "kaggle_calls": 0,
    }


def _synthetic_contract_probe(root: Path) -> dict[str, Any]:
    """Exercise one synthetic fit, durable publish, recovery, route, metric, and guard."""
    from unittest.mock import patch

    root.mkdir(parents=True, exist_ok=True)
    fit_n, score_n = 2_000, 80
    total = fit_n + score_n
    index = np.arange(total, dtype=np.int64)
    cycles = np.concatenate((np.repeat(np.arange(47, 60, dtype=np.int64), 154)[:fit_n],
                             np.repeat(np.arange(60, 64, dtype=np.int64), score_n // 4)))
    if cycles.shape != (total,) or cycles[fit_n - 1] != 59:
        raise RuntimeError("Synthetic N01 cycles do not reach the permitted cutoff")
    frame = pd.DataFrame({
        "impact_load": (index % 7 + 1).astype(float),
        "entry_friction": (index % 5 + 1).astype(float),
        "cycle_code": cycles,
        "scope_pressure": (index % 9 + 1).astype(float),
        "text_load": (index % 4 + 1).astype(float),
        "scope_trace": [f"{value % 8 + 1} scopes" for value in index],
        "org_spread": [f"~{value % 5 + 1}" for value in index],
        "module_spread": (index % 6 + 1).astype(int),
        "telemetry_flag": np.where(index % 2 == 0, "active", "inactive"),
        "org_code": np.where((index >= fit_n) & (index % 3 == 0), "org_new",
                             np.where(index % 3 == 0, "org_alpha", "org_beta")),
        "module_code": np.where(index % 4 == 0, "module_alpha", "module_beta"),
        "pattern_code": np.where(index % 5 == 0, "pattern_alpha", "pattern_beta"),
        "signal_stream": [
            (f"unicode café common token{value % 7} signal" if value % 11 else None)
            for value in index
        ],
    }, index=index).loc[:, list(FEATURE_NAMES)]
    target = ((index // 4) % 2).astype(np.int8)
    fit_rows = np.arange(fit_n, dtype=np.int64)
    score_rows = np.arange(fit_n, total, dtype=np.int64)
    fit_features = frame.iloc[fit_rows]
    plan = class_by_volume_weight_plan(fit_features, target[fit_rows], minimum_cell_n=MINIMUM_CLASS_BIN_N)
    score_cycles = cycles[score_rows]
    c2b = np.linspace(0.15, 0.85, score_n, dtype=np.float64)
    cb = np.linspace(0.80, 0.25, score_n, dtype=np.float64)
    sparse_raw = np.linspace(0.25, 0.75, score_n, dtype=np.float64)
    _docs, missing = _text_values(frame.iloc[score_rows]["signal_stream"])
    current = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
    control = {
        "rows": score_rows.copy(),
        "record_ids": np.asarray([f"SYN-{row:05d}" for row in score_rows]),
        "cycles": score_cycles.copy(), "c2b": c2b, "cb": cb,
        "sparse_raw": sparse_raw, "current": current,
        "missing": missing.astype(bool),
    }
    context = {
        "frame": frame, "target": target,
        "fit_rows": {59: fit_rows}, "controls": {59: control},
        "plans": {59: plan},
    }
    ledger = root / "synthetic_budget.jsonl"
    bundles = root / "bundles"
    reports = root / "reports"
    authorize_new_phase(ledger, cap=1,
                        category_caps={"pilot": 0, "continuation": 1,
                                       "control_replay": 0, "technical_retry": 0},
                        inherited_old_budget_snapshot={"sha256": "synthetic", "effective_total": 0})
    reserve_card(ledger, card_id=CARD_ID, stage=STAGE, fits=1, category="continuation")
    frozen = {"freeze_signature": "synthetic-diagnostic-freeze"}
    original_estimator = n01_model.LGBMClassifier
    original_params = dict(n01_model.PARAMS)
    calls = {"fit": 0}

    class CountingEstimator:
        def __init__(self, **kwargs: Any):
            self.inner = original_estimator(**kwargs)

        def fit(self, features: Any, labels: Any, *, sample_weight: Any = None) -> Any:
            calls["fit"] += 1
            self.inner.fit(features, labels, sample_weight=sample_weight)
            return self

        def predict_proba(self, features: Any) -> np.ndarray:
            return self.inner.predict_proba(features)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    forced_resource = {
        "status": "PASS", "fit_rows": fit_n, "score_rows": score_n,
        "estimated_peak_bytes": 1, "threads": THREADS, "concurrent_training_jobs": 1,
    }
    with patch.object(n01_model, "PARAMS", {**original_params, "n_estimators": 3}), \
            patch.object(n01_model, "LGBMClassifier", CountingEstimator), \
            patch.object(sys.modules[__name__], "LEDGER_PATH", ledger), \
            patch.object(sys.modules[__name__], "BUNDLE_ROOT", bundles), \
            patch.object(sys.modules[__name__], "REPORT_ROOT", reports), \
            patch.object(sys.modules[__name__], "_resource_estimate", lambda *_a, **_k: forced_resource):
        publisher = resume_or_publish_bundle
        interrupted = {"once": False}

        def publish_then_interrupt(*args: Any, **kwargs: Any) -> Any:
            result = publisher(*args, **kwargs)
            if not interrupted["once"]:
                interrupted["once"] = True
                raise OSError("synthetic interruption after immutable publish")
            return result

        with patch.object(sys.modules[__name__], "resume_or_publish_bundle", publish_then_interrupt):
            try:
                _fit_one(context, frozen, 59)
            except OSError as exc:
                if "synthetic interruption" not in str(exc):
                    raise
            else:
                raise RuntimeError("Synthetic persistence probe did not inject its interruption")

        pending = read_events(ledger)
        started = [event for event in pending if event.get("event") == "FIT_STARTED"]
        terminal = [event for event in pending if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if len(started) != 1 or terminal or calls["fit"] != 1:
            raise RuntimeError("Synthetic interrupted publish did not leave one resumable fit")
        arrays, receipt = _fit_one(context, frozen, 59)
        if calls["fit"] != 1:
            raise RuntimeError("Synthetic resume refit the estimator instead of reading its bundle")
        bundle = read_bundle(_bundle_path(59), _bundle_identity(frozen["freeze_signature"], 59),
                             task="classification")
        if bundle.receipt["receipt_sha256"] != receipt["receipt_sha256"]:
            raise RuntimeError("Synthetic receipt changed after durable read-back")
        routed = _independent_route(arrays)
        if not np.array_equal(arrays["score_rows"], score_rows):
            raise RuntimeError("Synthetic score row order changed")
        if not np.array_equal(routed[arrays["stream_missing"]], arrays["control__cb_probability"][arrays["stream_missing"]]):
            raise RuntimeError("Synthetic missing-stream fallback changed")
        metric_candidate = _metric(target[score_rows], routed)
        metric_baseline = _metric(target[score_rows], arrays["baseline_prediction"])
        quartile_edges = np.unique(np.quantile(frame.iloc[fit_rows]["text_load"], [0.25, 0.5, 0.75]))
        score_bins = np.searchsorted(quartile_edges, frame.iloc[score_rows]["text_load"], side="right")
        group_rows = []
        for group_name, labels in (("availability", np.where(missing, "missing", "present")),
                                   ("text_load_bin", score_bins.astype(str))):
            for label in sorted(set(labels.tolist())):
                selected = labels == label
                cm = _metric(target[score_rows][selected], routed[selected])
                bm = _metric(target[score_rows][selected], arrays["baseline_prediction"][selected])
                group_rows.append({
                    "field": group_name, "group": str(label), "n": int(selected.sum()),
                    "candidate_macro_f1": cm["macro_f1"], "baseline_macro_f1": bm["macro_f1"],
                    "f1_harm": bm["macro_f1"] - cm["macro_f1"],
                    "positive_recall_harm": bm["positive_recall"] - cm["positive_recall"],
                    "supported": bool(selected.sum() >= 1 and np.unique(target[score_rows][selected]).size == 2),
                })
        report = {
            "status": "PASS_SYNTHETIC_E2E", "fit_rows": fit_n, "score_rows": score_n,
            "candidate_metric": metric_candidate, "baseline_metric": metric_baseline,
            "subgroup_guard_rows": group_rows, "estimator_fit_calls": calls["fit"],
            "atomic_publish": True, "interruption_recovery_without_refit": True,
            "read_back_route_reconstruction": True, "fit_only_preprocessing": True,
            "unicode_and_missing_stream_covered": True, "unknown_category_covered": True,
        }
        report_path = reports / "synthetic_report.json"
        _write_once(report_path, report)
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise RuntimeError("Synthetic report persistence/read-back changed")
        events = read_events(ledger)
        if (sum(event.get("event") == "FIT_STARTED" for event in events) != 1
                or sum(event.get("event") == "FIT_SAVED" for event in events) != 1
                or sum(event.get("event") == "FIT_FAILED" for event in events) != 0):
            raise RuntimeError("Synthetic persistence/resume accounting is invalid")
        release_unused(ledger, card_id=CARD_ID, stage=STAGE)
    return {
        "status": "PASS_SYNTHETIC_E2E", "estimator_fit_calls": 1,
        "resumed_existing_bundle_without_refit": True,
        "bundle_receipt_sha256": receipt["receipt_sha256"],
        "bundle_npz_sha256": receipt["npz_sha256"],
        "fit_only_preprocessing": True, "exact_missing_route": True,
        "metric_rows": score_n, "subgroup_guard_rows": len(group_rows),
        "synthetic_only": True, "official_fits": 0, "test_or_kaggle_used": False,
    }


def synthetic_preflight() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="n01_class_only_diagnostic_") as scratch:
        result = _synthetic_contract_probe(Path(scratch) / "probe")
    source_paths = [
        ROOT / "src/models/research_next_n01.py",
        ROOT / "src/models/research_next_n01_class_only_diagnostic.py",
        ROOT / "src/models/research_next_contract.py",
        ROOT / "src/models/research_next_budget.py",
        ROOT / "scripts/optimization_v3_research_next_n01_class_only_diagnostic.py",
        ROOT / "scripts/optimization_v3_research_next_n01_class_only_diagnostic_verify.py",
        ROOT / "tests/test_research_next_n01_class_only_diagnostic.py",
    ]
    receipt = {
        **result,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m scripts.optimization_v3_research_next_n01_class_only_diagnostic synthetic-check",
        "source_sha256": {_relative(path): file_sha256(path) for path in source_paths},
    }
    _write_once(SYNTHETIC_RECEIPT_PATH, receipt)
    return receipt


def _read_synthetic_receipt(context: Mapping[str, Any]) -> dict[str, Any]:
    if not SYNTHETIC_RECEIPT_PATH.is_file():
        raise RuntimeError("Run the synthetic-check stage before preflight or freeze")
    receipt = json.loads(SYNTHETIC_RECEIPT_PATH.read_text(encoding="utf-8"))
    expected_sources = receipt.get("source_sha256", {})
    if (receipt.get("status") != "PASS_SYNTHETIC_E2E"
            or receipt.get("official_fits") != 0
            or receipt.get("test_or_kaggle_used") is not False
            or receipt.get("resumed_existing_bundle_without_refit") is not True):
        raise RuntimeError("Synthetic N01 diagnostic end-to-end preflight receipt failed")
    for relative, digest in expected_sources.items():
        if context["source_hashes"].get(relative) != digest:
            raise RuntimeError(f"Synthetic preflight receipt source changed: {relative}")
    return receipt


def preflight() -> dict[str, Any]:
    context = _train_context()
    synthetic = _read_synthetic_receipt(context)
    budget = _budget_preflight(require_unreserved=True)
    identity = _stable_identity(context)
    identity["synthetic_preflight_receipt_sha256"] = file_sha256(SYNTHETIC_RECEIPT_PATH)
    repeat = json.loads(json.dumps(identity, ensure_ascii=False, allow_nan=False))
    if canonical_sha256(identity) != canonical_sha256(repeat):
        raise RuntimeError("N01 class-only diagnostic identity is not stable")
    result = {
        "status": "PREFLIGHT_PASS" if all(item["status"] == "PASS" for item in context["resources"].values()) else "DEFERRED_RESOURCE",
        "identity": identity, "freeze_signature": canonical_sha256(identity),
        "execution_context": _execution_context(context),
        "budget_preflight": budget,
        "dynamic_context_sensitivity": {
            "identity_based_on_explicit_stable_fields": True,
            "excluded_fields": ["prepared_at_utc", "available_physical_bytes", "budget_ledger"],
        },
    }
    _write_once(REPORT_ROOT / "preflight.json", result)
    return result


def freeze() -> dict[str, Any]:
    context = _train_context()
    synthetic = _read_synthetic_receipt(context)
    budget = _budget_preflight(require_unreserved=True)
    identity_a = _stable_identity(context)
    identity_a["synthetic_preflight_receipt_sha256"] = file_sha256(SYNTHETIC_RECEIPT_PATH)
    # Recompute the explicit stable-field identity; execution-only context is
    # stored separately and never passed into this identity function.
    dynamic_context_a = _execution_context(context)
    identity_b = _stable_identity(context)
    identity_b["synthetic_preflight_receipt_sha256"] = file_sha256(SYNTHETIC_RECEIPT_PATH)
    if canonical_sha256(identity_a) != canonical_sha256(identity_b):
        raise RuntimeError("N01 diagnostic stable identity depends on RAM or budget context")
    resource_failures = {k: v for k, v in context["resources"].items() if v["status"] != "PASS"}
    payload = {
        "schema_version": 1, "identity": identity_a,
        "freeze_signature": canonical_sha256(identity_a),
        "execution_context": {
            **dynamic_context_a,
            "budget_prefit_snapshot": budget,
            "dynamic_context_sensitivity": {
                "identity_based_on_explicit_stable_fields": True,
                "excluded_fields": ["prepared_at_utc", "available_physical_bytes", "budget_ledger"],
            },
            "resource_failures": resource_failures,
            "synthetic_preflight": "REQUIRED_BEFORE_RUN",
            "synthetic_preflight_receipt": synthetic,
        },
    }
    if resource_failures:
        raise ResourceDeferred(json.dumps(resource_failures, sort_keys=True))
    _write_once(FREEZE_PATH, payload)
    return {"status": "FROZEN_READY", "path": _relative(FREEZE_PATH),
            "freeze_signature": payload["freeze_signature"], "fit_count": N_NEW_FITS,
            "resource_estimates": context["resources"]}


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N01 class-only diagnostic freeze signature is invalid")
    if not _is_ancestor(str(frozen["identity"].get("source_commit", ""))):
        raise RuntimeError("N01 class-only diagnostic source commit is not an ancestor of HEAD")
    context = _train_context()
    current_budget = _budget_preflight(require_unreserved=False)
    frozen_budget = frozen.get("execution_context", {}).get("budget_prefit_snapshot", {})
    if (current_budget["started"] < frozen_budget.get("started", 26)
            or current_budget["started"] > frozen_budget.get("started", 26) + N_NEW_FITS
            or current_budget["saved"] < frozen_budget.get("saved", 26)
            or current_budget["saved"] > frozen_budget.get("saved", 26) + N_NEW_FITS):
        raise RuntimeError("Live budget ledger no longer matches the frozen four-fit N01 diagnostic scope")
    actual = _stable_identity(context)
    actual["synthetic_preflight_receipt_sha256"] = file_sha256(SYNTHETIC_RECEIPT_PATH)
    expected = dict(frozen["identity"])
    actual.pop("source_commit", None)
    expected.pop("source_commit", None)
    if actual != expected:
        raise RuntimeError("N01 class-only diagnostic source/input/recipe identity changed after freeze")
    resource_failures = {k: v for k, v in context["resources"].items() if v["status"] != "PASS"}
    if resource_failures:
        raise ResourceDeferred(json.dumps(resource_failures, sort_keys=True))
    return frozen, context


def _bundle_identity(signature: str, cutoff: int) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "amendment": AMENDMENT,
            "stage": STAGE, "freeze_signature": signature, "cutoff": int(cutoff),
            "component": "class_only"}


def _bundle_path(cutoff: int) -> Path:
    return BUNDLE_ROOT / f"fit{cutoff}" / "class_only"


def _independent_route(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = arrays["stream_missing"].astype(bool)
    c2b = arrays["control__c2b_probability"].astype(np.float64)
    cb = arrays["control__cb_probability"].astype(np.float64)
    raw = arrays["raw_prediction"].astype(np.float64)
    expected = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * arrays["control__sparse_baseline_raw"])
    if not np.array_equal(expected, arrays["prediction"]):
        if not np.allclose(expected, arrays["prediction"], rtol=0, atol=1e-12):
            raise RuntimeError("Independent candidate routing reconstruction differs")
    if not np.allclose(baseline, arrays["baseline_prediction"], rtol=0, atol=1e-12):
        raise RuntimeError("Independent sparse25 routing reconstruction differs")
    if not np.array_equal(expected[missing], cb[missing]):
        raise RuntimeError("Missing-stream prediction is not exact frozen CB fallback")
    return expected


def _load_existing(context: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = _bundle_path(cutoff)
    events = read_events(LEDGER_PATH)
    fit_id = f"{CARD_ID}:{STAGE}:{cutoff}:class_only"
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if not path.exists():
        if starts:
            raise RuntimeError(f"{fit_id} started without a durable bundle; preserve and recover, do not refit")
        return None
    if (len(starts) != 1 or len(terminal) > 1
            or (terminal and terminal[0].get("event") != "FIT_SAVED")):
        raise RuntimeError(f"Existing continuation bundle lacks valid start/saved ledger lineage: {fit_id}")
    bundle = read_bundle(path, _bundle_identity(frozen["freeze_signature"], cutoff), task="classification")
    arrays = bundle.arrays
    _independent_route(arrays)
    fit_rows = context["fit_rows"][cutoff]
    expected = context["controls"][cutoff]
    if (arrays["fit_rows"].shape != fit_rows.shape
            or not np.array_equal(arrays["fit_rows"], fit_rows)
            or not np.array_equal(arrays["score_rows"], expected["rows"])
            or not np.array_equal(arrays["score_record_ids"].astype(str), expected["record_ids"])
            or not np.array_equal(arrays["cycle_code"], expected["cycles"])
            or (terminal and bundle.receipt["receipt_sha256"] != terminal[0].get("receipt_sha256"))):
        raise RuntimeError(f"Existing continuation bundle alignment/accounting failed at {cutoff}")
    metadata = bundle.receipt.get("metadata", {}).get("metadata", {})
    plan = context["plans"][cutoff]
    fit_state = metadata.get("fit_state", {})
    if (metadata.get("card_id") != CARD_ID or metadata.get("cutoff") != cutoff
            or metadata.get("weight_kind") != "class_only"
            or metadata.get("fit_rows") != len(fit_rows)
            or metadata.get("score_rows") != len(expected["rows"])
            or metadata.get("test_data_used") is not False
            or metadata.get("score_labels_passed_to_fit") is not False
            or metadata.get("sample_submission_used") is not False
            or metadata.get("kaggle_calls") != 0
            or fit_state.get("sample_weight_used") is not True
            or fit_state.get("sample_weight_sha256") != _hash_array(plan["class_only_weights"], dtype="<f8")
            or fit_state.get("fit_rows") != len(fit_rows)
            or metadata.get("fit_row_sha256") != _hash_array(fit_rows, dtype="<i8")
            or metadata.get("fit_target_sha256") != _hash_array(context["target"][fit_rows], dtype="i1")):
        raise RuntimeError(f"Existing continuation fit-state receipt is inconsistent at {cutoff}")
    if not terminal:
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
    _write_once(REPORT_ROOT / "progress" / f"fit{cutoff}.json", {
        "fit_id": fit_id, "status": "FIT_SAVED", "receipt_sha256": bundle.receipt["receipt_sha256"],
        "bundle_npz_sha256": bundle.receipt["npz_sha256"], "fit_rows": int(len(fit_rows)),
        "score_rows": int(len(expected["rows"])), "resumed_existing_bundle": True,
    })
    return {k: v.copy() for k, v in arrays.items()}, bundle.receipt


def _fit_one(context: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    existing = _load_existing(context, frozen, cutoff)
    if existing is not None:
        return existing
    plan = context["plans"][cutoff]
    fit_rows = context["fit_rows"][cutoff]
    control = context["controls"][cutoff]
    score_mask = (control["cycles"] > cutoff) & (control["cycles"] <= min(cutoff + 4, 79))
    score_rows = control["rows"][score_mask]
    fit_features = context["frame"].iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
    score_features = context["frame"].iloc[score_rows].loc[:, list(FEATURE_NAMES)]
    resource = _resource_estimate(fit_features, score_features)
    if resource["status"] != "PASS":
        raise ResourceDeferred(json.dumps(resource, sort_keys=True))
    fit_id = f"{CARD_ID}:{STAGE}:{cutoff}:class_only"
    events = read_events(LEDGER_PATH)
    if any(e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id for e in events):
        raise RuntimeError(f"Refusing duplicate estimator start for {fit_id}")
    start_fit(LEDGER_PATH, fit_id=fit_id, card_id=CARD_ID, stage=STAGE,
              cutoff=cutoff, component="class_only")
    try:
        model = N01ClassOnlyDiagnosticClassifier(cutoff=cutoff, weight_kind="class_only")
        model.fit(fit_features, context["target"][fit_rows], plan["class_only_weights"])
        raw = model.predict_probability(score_features)
        # Controls are aligned by the same score rows; no target is passed here.
        all_mask = np.isin(control["rows"], score_rows)
        selected = {key: np.asarray(value)[all_mask] for key, value in control.items()
                    if key in {"rows", "record_ids", "cycles", "c2b", "cb", "sparse_raw", "current", "missing"}}
        if (not np.array_equal(selected["rows"], score_rows)
                or not np.array_equal(selected["cycles"], control["cycles"][score_mask])
                or not np.array_equal(selected["record_ids"], control["record_ids"][score_mask])):
            raise RuntimeError(f"N01 C16 controls are not in exact score-row order at {cutoff}")
        routes = route_expert_probabilities(
            raw_candidate=np.full_like(raw, 0.5), raw_class_only=raw,
            c2b_probability=selected["c2b"], cb_probability=selected["cb"],
            sparse_baseline_raw=selected["sparse_raw"], stream_missing=selected["missing"],
        )
        candidate = routes["matched_control"]
        if len(raw) != len(score_rows) or not np.isfinite(raw).all():
            raise RuntimeError(f"N01 class-only raw predictions invalid at {cutoff}")
        payload: dict[str, Any] = {
            "fit_rows": fit_rows.astype(np.int64),
            "score_rows": score_rows.astype(np.int64),
            "score_record_ids": selected["record_ids"].astype(str),
            "cycle_code": selected["cycles"].astype(np.int64),
            "origin_cutoff": np.full(len(score_rows), cutoff, dtype=np.int64),
            "prediction": candidate.astype(np.float64),
            "baseline_prediction": selected["current"].astype(np.float64),
            "raw_prediction": raw.astype(np.float64),
            "control__c2b_probability": selected["c2b"].astype(np.float64),
            "control__cb_probability": selected["cb"].astype(np.float64),
            "control__sparse_baseline_raw": selected["sparse_raw"].astype(np.float64),
            "stream_missing": selected["missing"].astype(bool),
            "metadata": {
                "card_id": CARD_ID, "task": "classification", "cutoff": cutoff,
                "weight_kind": "class_only", "fit_rows": int(len(fit_rows)),
                "score_rows": int(len(score_rows)),
                "fit_row_sha256": _hash_array(fit_rows, dtype="<i8"),
                "fit_target_sha256": _hash_array(context["target"][fit_rows], dtype="i1"),
                "class_only_weight_sha256": _hash_array(plan["class_only_weights"], dtype="<f8"),
                "fit_state": model.audit_state(), "weight_diagnostics": n01._stable_bin_summary(plan),
                "resource_preflight": resource,
                "score_labels_passed_to_fit": False, "test_data_used": False,
                "sample_submission_used": False, "kaggle_calls": 0,
            },
        }
        bundle = resume_or_publish_bundle(
            _bundle_path(cutoff), payload, _bundle_identity(frozen["freeze_signature"], cutoff),
            task="classification",
        )
        arrays = {key: bundle.arrays[key].copy() for key in bundle.arrays}
        _independent_route(arrays)
        if not np.array_equal(arrays["score_rows"], score_rows):
            raise RuntimeError(f"Read-back score rows differ for {fit_id}")
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
        _write_once(REPORT_ROOT / "progress" / f"fit{cutoff}.json", {
            "fit_id": fit_id, "status": "FIT_SAVED", "receipt_sha256": bundle.receipt["receipt_sha256"],
            "bundle_npz_sha256": bundle.receipt["npz_sha256"], "fit_rows": int(len(fit_rows)),
            "score_rows": int(len(score_rows)), "resource_preflight": resource,
        })
        return arrays, bundle.receipt
    except Exception as exc:
        events_after = read_events(LEDGER_PATH)
        terminal = [e for e in events_after if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not terminal and not _bundle_path(cutoff).exists():
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def _metric(target: Any, probabilities: Any) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int64)
    pred = (np.asarray(probabilities, dtype=np.float64) >= 0.50).astype(np.int64)
    if y.shape != pred.shape or y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError("N01 diagnostic metric inputs are not aligned binary vectors")
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "macro_f1": float((f0 + f1) / 2), "positive_recall": float(recall)}


def _pilot_record(context: Mapping[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    arrays = context["pilot_bundles"][cutoff]
    control = context["controls"][cutoff]
    raw = arrays["raw_prediction"].astype(np.float64)
    missing = arrays["stream_missing"].astype(bool)
    candidate = np.where(missing, arrays["control__cb_probability"],
                         0.75 * arrays["control__c2b_probability"] + 0.25 * raw)
    if not np.allclose(candidate, arrays["prediction"], rtol=0, atol=1e-12):
        raise RuntimeError(f"N01 cached class-only pilot route reconstruction failed at {cutoff}")
    return {
        "rows": arrays["score_rows"].astype(np.int64),
        "record_ids": arrays["score_record_ids"].astype(str),
        "cycles": arrays["cycle_code"].astype(np.int64),
        "origin": arrays["origin_cutoff"].astype(np.int64),
        "candidate": candidate,
        "current": arrays["baseline_prediction"].astype(np.float64),
        "missing": missing,
    }


def _continuation_record(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    _independent_route(arrays)
    return {"rows": arrays["score_rows"].astype(np.int64),
            "record_ids": arrays["score_record_ids"].astype(str),
            "cycles": arrays["cycle_code"].astype(np.int64),
            "origin": arrays["origin_cutoff"].astype(np.int64),
            "candidate": arrays["prediction"].astype(np.float64),
            "current": arrays["baseline_prediction"].astype(np.float64),
            "missing": arrays["stream_missing"].astype(bool)}


def _collect(records: Mapping[int, Mapping[str, np.ndarray]], blocks: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    parts = []
    for cutoff, lo, hi in blocks:
        row = records[int(cutoff)]
        mask = (row["cycles"] >= int(lo)) & (row["cycles"] <= int(hi))
        parts.append({key: np.asarray(value)[mask] for key, value in row.items()})
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(merged["rows"], kind="mergesort")
    merged = {key: value[order] for key, value in merged.items()}
    if len(np.unique(merged["rows"])) != len(merged["rows"]):
        raise RuntimeError("N01 full-six view contains duplicate source rows")
    return merged


def _fit_bins(frame: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, list[float]]:
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    fit_mask = frame["cycle_code"].to_numpy(dtype=np.int64) <= cutoff
    finite = parsed[fit_mask & np.isfinite(parsed)]
    bins = np.full(len(frame), -1, dtype=np.int64)
    if not len(finite):
        return bins, []
    edges = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
    valid = np.isfinite(parsed)
    bins[valid] = np.searchsorted(edges, parsed[valid], side="right").astype(np.int64)
    return bins, edges.astype(float).tolist()


def _analyze(context: Mapping[str, Any], records: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    queue_views = context["queue"]["views"]
    blocks = {"primary_full_six": queue_views["full_six"],
              "late_primary_W5_W6": queue_views["late_primary"]}
    for idx, block in enumerate(queue_views["full_six"], start=1):
        blocks[f"W{idx}"] = [block]
    view_results: dict[str, Any] = {}
    group_rows: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    bins_by_origin = {cutoff: _fit_bins(context["frame"], cutoff) for cutoff in ALL_ORIGINS}
    for view_name, view_blocks in blocks.items():
        data = _collect(records, view_blocks)
        rows = data["rows"].astype(np.int64)
        y = context["target"][rows]
        actual_cycles = context["cycles"][rows]
        actual_ids = context["frame"].iloc[rows]["record_id"].astype(str).to_numpy()
        if not np.array_equal(actual_cycles, data["cycles"]) or not np.array_equal(actual_ids, data["record_ids"]):
            raise RuntimeError(f"N01 {view_name} source-row alignment failed")
        candidate = _metric(y, data["candidate"])
        current = _metric(y, data["current"])
        view_results[view_name] = {
            "n_unique_rows": int(len(rows)), "source_rows_sha256": _hash_array(rows, dtype="<i8"),
            "candidate": candidate, "sparse25": current,
            "gain_macro_f1_vs_sparse25": float(candidate["macro_f1"] - current["macro_f1"]),
            "positive_recall_gain_vs_sparse25": float(candidate["positive_recall"] - current["positive_recall"]),
            "cycle_counts": {str(int(c)): int(np.sum(actual_cycles == c)) for c in np.unique(actual_cycles)},
        }
        for cycle in np.unique(actual_cycles):
            mask = actual_cycles == cycle
            cm, bm = _metric(y[mask], data["candidate"][mask]), _metric(y[mask], data["current"][mask])
            per_cycle.append({"view": view_name, "cycle_code": int(cycle), "n": int(mask.sum()),
                              "candidate_macro_f1": cm["macro_f1"], "sparse25_macro_f1": bm["macro_f1"],
                              "gain_macro_f1_vs_sparse25": cm["macro_f1"] - bm["macro_f1"],
                              "candidate_positive_recall": cm["positive_recall"],
                              "sparse25_positive_recall": bm["positive_recall"]})
        local = context["frame"].iloc[rows]
        labels = {
            "entry_friction": local["entry_friction"].astype(str).to_numpy(),
            "stream_availability": np.where(data["missing"], "missing", "present"),
            "fit_defined_text_load_quartile": np.asarray([
                str(int(bins_by_origin[int(origin)][0][int(row)]))
                for row, origin in zip(rows, data["origin"].astype(np.int64))
            ], dtype=str),
        }
        for field, group_labels in labels.items():
            for label in sorted(set(group_labels.tolist())):
                mask = group_labels == label
                subset_rows = rows[mask]
                subset_y = context["target"][subset_rows]
                c0 = int(np.sum(subset_y == 0)); c1 = int(np.sum(subset_y == 1))
                supported = len(subset_rows) >= MIN_GROUP_N and min(c0, c1) >= MIN_EACH_CLASS_N
                cm, bm = _metric(subset_y, data["candidate"][mask]), _metric(subset_y, data["current"][mask])
                f1_harm = float(bm["macro_f1"] - cm["macro_f1"])
                recall_harm = float(bm["positive_recall"] - cm["positive_recall"])
                group_rows.append({
                    "view": view_name, "field": field, "group": str(label), "n": int(len(subset_rows)),
                    "class_0": c0, "class_1": c1, "supported": bool(supported),
                    "candidate_macro_f1": cm["macro_f1"], "sparse25_macro_f1": bm["macro_f1"],
                    "f1_harm_vs_sparse25": f1_harm,
                    "candidate_positive_recall": cm["positive_recall"],
                    "sparse25_positive_recall": bm["positive_recall"],
                    "positive_recall_harm_vs_sparse25": recall_harm,
                    "guard_failure_descriptive": bool(supported and (f1_harm > MAX_F1_HARM or recall_harm > MAX_RECALL_HARM)),
                    "source_rows_sha256": _hash_array(subset_rows, dtype="<i8"),
                    "fit_text_edges": {str(k): bins_by_origin[k][1] for k in ALL_ORIGINS}
                    if field == "fit_defined_text_load_quartile" else None,
                })

    # H12 is reported only for the two already-fitted pilot origins. New
    # continuation fit caches score their next four cycles only, matching C16.
    inherited_h12: dict[str, Any] = {}
    for cutoff, block in ((55, [55, 56, 67]), (67, [67, 68, 79])):
        data = _collect(records, [block])
        y = context["target"][data["rows"]]
        cm, bm = _metric(y, data["candidate"]), _metric(y, data["current"])
        inherited_h12[str(cutoff)] = {
            "source": "reused_N01_class_only_pilot_bundle; not a newly fitted continuation origin",
            "n": len(data["rows"]), "candidate": cm, "sparse25": bm,
            "gain_macro_f1_vs_sparse25": cm["macro_f1"] - bm["macro_f1"],
            "positive_recall_gain_vs_sparse25": cm["positive_recall"] - bm["positive_recall"],
        }

    supported = [row for row in group_rows if row["supported"]]
    report = {
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "card_id": CARD_ID, "amendment": AMENDMENT,
        "question": "Does the already-gated N01 class-only pilot recipe retain diagnostic value across the six non-overlapping primary windows?",
        "decision_scope": "diagnostic only; no model qualification or automatic continuation",
        "views": view_results, "inherited_pilot_H12": inherited_h12,
        "per_cycle": per_cycle,
        "subgroup_summary": {
            "supported_group_rows": len(supported),
            "unsupported_group_rows": len(group_rows) - len(supported),
            "supported_guard_failures": sum(bool(row["guard_failure_descriptive"]) for row in supported),
            "thresholds": {"minimum_n": MIN_GROUP_N, "minimum_each_class": MIN_EACH_CLASS_N,
                           "maximum_macro_f1_harm": MAX_F1_HARM,
                           "maximum_positive_recall_harm": MAX_RECALL_HARM},
        },
        "full_six_unique_primary_rows": view_results["primary_full_six"]["n_unique_rows"],
        "late_W5_W6_rows": view_results["late_primary_W5_W6"]["n_unique_rows"],
        "no_full_six_h12_claim": "Only cutoff 55/67 H12 bundles exist for the frozen current route; these are reused pilot diagnostics and overlap primary. New origin fits score the same next-four-cycle rows as the C16 control cache.",
    }
    return {"report": report, "groups": group_rows, "per_cycle": per_cycle}


def _write_window_plot(report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    names = [f"W{i}" for i in range(1, 7)]
    gains = [report["views"][name]["gain_macro_f1_vs_sparse25"] for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#3274a1" if value >= 0 else "#b0433a" for value in gains]
    ax.bar(names, gains, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_ylabel("Macro F1 gain vs sparse25")
    ax.set_title("N01 class-only diagnostic — six primary windows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = REPORT_ROOT / "window_gains.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    fig.savefig(temporary, format="png", dpi=150)
    plt.close(fig)
    if path.exists():
        if file_sha256(path) != hashlib.sha256(temporary.read_bytes()).hexdigest():
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Refusing to overwrite immutable diagnostic plot: {path}")
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(path)


def _receipt_map() -> dict[str, Any]:
    events = read_events(LEDGER_PATH)
    receipt_map = {}
    for cutoff in CONTINUATION_CUTOFFS:
        fit_id = f"{CARD_ID}:{STAGE}:{cutoff}:class_only"
        starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
        if len(starts) != 1 or len(saved) != 1:
            raise RuntimeError(f"Fit accounting does not contain one saved receipt for {fit_id}")
        bundle = read_bundle(_bundle_path(cutoff), _bundle_identity(_read_frozen_signature(), cutoff), task="classification")
        if bundle.receipt["receipt_sha256"] != saved[0].get("receipt_sha256"):
            raise RuntimeError(f"Ledger receipt differs from saved bundle for {fit_id}")
        receipt_map[fit_id] = {
            "status": "FIT_SAVED", "receipt_sha256": bundle.receipt["receipt_sha256"],
            "npz_sha256": bundle.receipt["npz_sha256"],
            "fit_rows": int(bundle.arrays["fit_rows"].size),
            "score_rows": int(bundle.arrays["score_rows"].size),
        }
    return receipt_map


def _read_frozen_signature() -> str:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"]


def run() -> dict[str, Any]:
    frozen, context = verify_freeze()
    reservation = reserve_card(LEDGER_PATH, card_id=CARD_ID, stage=STAGE,
                               fits=N_NEW_FITS, category="continuation")
    output_arrays: dict[int, dict[str, np.ndarray]] = {}
    done: list[int] = []
    try:
        for cutoff in CONTINUATION_CUTOFFS:
            arrays, _receipt = _fit_one(context, frozen, cutoff)
            output_arrays[cutoff] = arrays
            done.append(cutoff)
        release_unused(LEDGER_PATH, card_id=CARD_ID, stage=STAGE)
    except ResourceDeferred as exc:
        result = {"status": "DEFERRED_RESOURCE", "completed_cutoffs": done,
                  "message": str(exc), "budget": ledger_summary(LEDGER_PATH),
                  "reservation_preserved_for_same_frozen_resume": True,
                  "test_data_used": False, "kaggle_calls": 0}
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _write_once(REPORT_ROOT / "attempts" / f"{attempt}_resource_deferred.json", result)
        return result
    except Exception as exc:
        # Preserve any started/saved outputs and stop before the next origin.
        result = {"status": "TECHNICAL_REWORK", "completed_cutoffs": done,
                  "failed_at_or_after": CONTINUATION_CUTOFFS[len(done)] if len(done) < len(CONTINUATION_CUTOFFS) else None,
                  "error": f"{type(exc).__name__}: {exc}",
                  "budget": ledger_summary(LEDGER_PATH), "test_data_used": False, "kaggle_calls": 0}
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _write_once(REPORT_ROOT / "attempts" / f"{attempt}_technical_rework.json", result)
        raise

    records: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in PILOT_ORIGINS:
        records[cutoff] = _pilot_record(context, cutoff)
    for cutoff in CONTINUATION_CUTOFFS:
        records[cutoff] = _continuation_record(output_arrays[cutoff])
    analysis = _analyze(context, records)
    accounting = {
        "reserved_fit_count": int(reservation["fits"]),
        "official_estimator_fits": N_NEW_FITS,
        "fit_receipts": _receipt_map(),
        "budget": ledger_summary(LEDGER_PATH),
        "new_phase_fit_count_before_this_amendment": 26,
        "this_amendment_fits": N_NEW_FITS,
        "new_phase_fit_count_after_this_amendment": ledger_summary(LEDGER_PATH)["started_and_consumed"],
        "old_historical_budget_reset": False,
        "test_data_used": False, "sample_submission_used": False, "kaggle_calls": 0,
    }
    report = {
        **analysis["report"],
        "freeze_signature": frozen["freeze_signature"],
        "source_commit": frozen["identity"]["source_commit"],
        "input_and_source_sha256": frozen["identity"]["source_and_input_sha256"],
        "fit_populations": frozen["identity"]["fit_and_score_rows"],
        "fit_accounting": accounting,
        "reused_pilot_H12_source": "N01 class_only pilot bundles at cutoff 55 and 67; no refit",
        "no_test_sample_submission_or_kaggle": True,
        "champion_or_model_selection_changed": False,
    }
    _write_once(REPORT_ROOT / "results.json", report)
    _write_csv_once(REPORT_ROOT / "groups.csv", analysis["groups"])
    _write_csv_once(REPORT_ROOT / "per_cycle.csv", analysis["per_cycle"])
    _write_window_plot(report)
    _write_once(REPORT_ROOT / "run_status.json", {
        "status": "COMPLETE_DIAGNOSTIC_ONLY", "completed_cutoffs": list(CONTINUATION_CUTOFFS),
        "official_estimator_fits": N_NEW_FITS, "test_data_used": False, "kaggle_calls": 0,
    })
    return {"status": report["status"], "fit_accounting": accounting,
            "primary": report["views"]["primary_full_six"],
            "late": report["views"]["late_primary_W5_W6"],
            "subgroup_summary": report["subgroup_summary"],
            "output": str(REPORT_ROOT)}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("synthetic-check", "preflight", "freeze", "run"))
    args = parser.parse_args()
    if args.stage == "synthetic-check":
        result = synthetic_preflight()
    elif args.stage == "preflight":
        result = preflight()
    elif args.stage == "freeze":
        result = freeze()
    else:
        result = run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
