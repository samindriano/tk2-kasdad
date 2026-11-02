"""Run the frozen, measured Classification benchmark exception (no Kaggle upload)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_research_next_n00 as n00  # noqa: E402
from scripts import optimization_v3_research_next_n01 as n01  # noqa: E402
from scripts import optimization_v3_research_next_n02 as n02  # noqa: E402
from scripts import prepare_c16_r22_fulltrain_submissions as fulltrain  # noqa: E402
from src.models.classification_models import ClassificationModel, _text_values  # noqa: E402
from src.models.optimization_v3_batch1 import exclusive_training_slot  # noqa: E402
from src.models.research_next_budget import (  # noqa: E402
    DEFAULT_CAPS, finish_fit, ledger_summary, read_events, release_unused,
    reserve_card, start_fit,
)
from src.models.research_next_contract import (  # noqa: E402
    canonical_sha256, file_sha256, read_bundle, resume_or_publish_bundle,
)
from src.models.research_next_n01 import (  # noqa: E402
    ClassByVolumeSparseClassifier, FEATURE_NAMES, THRESHOLD, class_by_volume_weight_plan,
)
from src.utils.paths import task_files  # noqa: E402

TASK = "MEASURED_CLASSIFICATION_BENCHMARK_V1"
PROTOCOL = ROOT / "experiments/optimization_v3/measured_benchmark_v1/class_benchmark_protocol.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION"
FREEZE_PATH = ROOT / "experiments/optimization_v3/measured_benchmark_v1/CLASSIFICATION/freeze.json"
FIT_ROOT = ROOT / "experiments/runs/optimization_v3_measured_benchmark_v1/CLASSIFICATION"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
LOCK = ROOT / "experiments/runs/optimization_v3_batch1/.training.lock"
ORIGINS = (55, 59, 63, 67, 71, 75)
CONTINUATION_ORIGINS = (59, 63, 71, 75)
FIT_KINDS = ("sketch", "no_sketch")
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
RESERVE_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3
MIN_GROUP_N = 200
MIN_EACH_CLASS_N = 30
MAX_F1_HARM = 0.01
ORIGINAL_MAX_RECALL_HARM = 0.02
BENCHMARK_MAX_RECALL_HARM = 0.03
FEATURES = tuple(ClassificationModel({"id": "C2_B"}).input_feature_names)


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _array_sha(values: Any, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _write_once(path: Path, value: Any) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite immutable measured evidence: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _dependencies() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "psutil", "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in packages}}


def _source_paths() -> list[Path]:
    names = (
        "scripts/optimization_v3_measured_classification.py",
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n01.py",
        "scripts/optimization_v3_research_next_n02.py",
        "scripts/prepare_c16_r22_fulltrain_submissions.py",
        "src/models/classification_models.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n01.py",
        "src/models/research_next_n02.py",
        "src/models/optimization_v3_batch1.py",
        "tests/test_optimization_v3_measured_classification.py",
    )
    return [ROOT / name for name in names]


def _input_paths(train_path: Path) -> list[Path]:
    paths = [PROTOCOL, ROOT / "reports/optimization_v3/measured_benchmark_v1/DECISION.md",
             n02.QUEUE_PATH, n02.READINESS_PATH, n00.FREEZE_PATH,
             n00.REPORT_ROOT / "summary.json", n01.FREEZE_PATH, n02.FREEZE_PATH,
             n02.N00_SUMMARY_PATH, n02.REPORT_ROOT / "results.json",
             ROOT / "reports/optimization_v3/research_next_v1/results/N01_CLASS_ONLY_DIAGNOSTIC/amendment_01/results.json",
             ROOT / "reports/optimization_v3/research_next_v1/results/N01_CLASS_ONLY_DIAGNOSTIC/amendment_01/verification.json",
             ROOT / "reports/optimization_v3/research_next_v1/results/N01_N02_GUARD_AUDIT/amendment_01/results.json",
             ROOT / "reports/optimization_v3/research_next_v1/results/N01_N02_GUARD_AUDIT/amendment_01/verification.json",
             train_path]
    for cutoff in ORIGINS:
        cache = n00.cache_path("C16", cutoff)
        paths.extend((cache, cache.with_suffix(".json")))
    for cutoff in (55, 67):
        for kind in FIT_KINDS:
            bundle = n02._bundle_path(cutoff, kind)
            paths.extend((bundle / "arrays.npz", bundle / "receipt.json"))
    return sorted(set(paths), key=_relative)


def _verify_n00_parent_freeze() -> dict[str, Any]:
    """Verify N00 immutable inputs while allowing this measured work as a descendant commit."""
    frozen = json.loads(n00.FREEZE_PATH.read_text(encoding="utf-8"))
    expected = dict(frozen.get("identity", {}))
    if frozen.get("freeze_signature") != canonical_sha256(expected):
        raise RuntimeError("N00 parent freeze signature is invalid")
    source_commit = str(expected.get("source_commit", ""))
    if not source_commit or not n02._is_ancestor(source_commit):
        raise RuntimeError("N00 frozen source commit is not an ancestor of measured source")
    actual = dict(n00.build_freeze().get("identity", {}))
    expected.pop("source_commit", None)
    actual.pop("source_commit", None)
    if actual != expected:
        raise RuntimeError("N00 frozen non-code inputs changed; preserve and investigate before fitting")
    return frozen


def _context() -> dict[str, Any]:
    _verify_n00_parent_freeze()
    frozen, parent = n02._verify_freeze()
    frame = parent["frame"]
    target = parent["target"].astype(np.int8)
    cycles = parent["cycles"].astype(np.int64)
    if len(frame) != 50_608 or set(np.unique(target)) != {0, 1}:
        raise RuntimeError("Official classification Train row count/target validation failed")
    data: dict[int, dict[str, Any]] = {}
    missing_all = _text_values(frame["signal_stream"])[1].astype(bool)
    for cutoff in ORIGINS:
        cache = n00._load_artifact("C16", cutoff, frame)
        reconstructed = n00.reconstruct_classification(cache, missing_all)
        if cutoff in (55, 67):
            score_mask = (cache["cycles"] > cutoff) & (cache["cycles"] <= min(cutoff + 4, 79))
        else:
            score_mask = np.ones(len(cache["rows"]), dtype=bool)
        rows = cache["rows"][score_mask].astype(np.int64)
        score_cycles = cache["cycles"][score_mask].astype(np.int64)
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= min(cutoff + 4, 79)))
        if not np.array_equal(rows, expected_rows):
            raise RuntimeError(f"C16/{cutoff} cached score rows do not equal the four-cycle target window")
        if not np.array_equal(score_cycles, cycles[rows]):
            raise RuntimeError(f"C16/{cutoff} cycle alignment failed")
        arrays = cache["arrays"]
        data[cutoff] = {
            "fit_rows": np.flatnonzero(cycles <= cutoff).astype(np.int64),
            "rows": rows, "cycles": score_cycles,
            "record_ids": frame.record_id.astype(str).to_numpy()[rows],
            "current": reconstructed["control"][score_mask].astype(np.float64),
            "cb": arrays["control__cb_probability"][score_mask].astype(np.float64),
            "missing": reconstructed["stream_missing"][score_mask].astype(bool),
            "cache_npz_sha256": cache["npz_sha256"], "cache_sidecar_sha256": cache["sidecar_sha256"],
        }
        if int(cycles[data[cutoff]["fit_rows"]].max()) != cutoff:
            raise RuntimeError(f"Fit prefix for cutoff {cutoff} does not end at its cutoff")
    return {"frame": frame, "target": target, "cycles": cycles, "train_path": parent["train_path"],
            "n02_frozen": frozen, "n02_parent": parent, "missing_all": missing_all, "data": data}


def _synthetic_preflight() -> dict[str, Any]:
    names = FEATURES
    n = 48
    X = pd.DataFrame({
        "impact_load": np.arange(n, dtype=float), "entry_friction": np.arange(n) % 7,
        "cycle_code": np.full(n, 55), "scope_pressure": (np.arange(n) % 11).astype(float),
        "text_load": np.arange(n) % 20, "scope_trace": np.ones(n),
        "org_spread": np.ones(n), "module_spread": np.ones(n),
        "telemetry_flag": np.arange(n) % 2, "org_code": [f"org-{i%4}" for i in range(n)],
        "module_code": [f"mod-{i%5}" for i in range(n)],
        "pattern_code": [f"pat-{i%3}" for i in range(n)],
        "signal_stream": [None if i % 9 == 0 else f"token{i%6} shared pattern{i%3}" for i in range(n)],
    }).loc[:, list(names)]
    y = (np.arange(n) % 2).astype(np.int8)
    model = n02.C2BInteractionClassifier(include_sketch=True)
    with threadpool_limits(limits=THREADS):
        model.fit(X, y)
        raw = model.predict_probability(X.iloc[:12].copy())
    routes = n02.route_expert_probabilities(raw, np.full(12, 0.5), np.full(12, 0.4),
                                            np.full(12, 0.7), np.zeros(12, dtype=bool))
    if raw.shape != (12,) or not np.isfinite(raw).all() or not np.isfinite(routes["candidate"]).all():
        raise RuntimeError("Synthetic N02 end-to-end classifier/routing check failed")
    payload = {
        "score_rows": np.arange(12, dtype=np.int64),
        "score_record_ids": np.asarray([f"SYN-{i}" for i in range(12)], dtype="U"),
        "cycle_code": np.full(12, 56, dtype=np.int64),
        "origin_cutoff": np.full(12, 55, dtype=np.int64),
        "prediction": routes["candidate"].astype(np.float64),
        "baseline_prediction": routes["current_route"].astype(np.float64),
        "raw_prediction": raw.astype(np.float64),
        "metadata": {"synthetic_only": True, "fit_rows": n, "test_data_used": False},
    }
    identity = {"batch_id": TASK, "card_id": "SYNTHETIC", "stage": "QA"}
    with tempfile.TemporaryDirectory(prefix="kasdad-measured-cls-") as temporary:
        location = Path(temporary) / "bundle"
        first = resume_or_publish_bundle(location, payload, identity, task="classification")
        second = resume_or_publish_bundle(location, payload, identity, task="classification")
        if first.receipt["receipt_sha256"] != second.receipt["receipt_sha256"]:
            raise RuntimeError("Synthetic bundle persistence/readback/resume hash changed")
        reopened = read_bundle(location, identity, task="classification")
        if not np.array_equal(reopened.arrays["prediction"], payload["prediction"]):
            raise RuntimeError("Synthetic bundle resume returned different predictions")
    sketch_smoke = n02._sketch_api_smoke()
    return {"status": "PASS", "synthetic_estimator_fits": 1,
            "end_to_end_fit_predict_route": "PASS", "bundle_persist_readback_resume": "PASS",
            "bundle_receipt_sha256": first.receipt["receipt_sha256"],
            "sketch_api_smoke": sketch_smoke, "official_data_used": False,
            "test_data_used": False, "kaggle_called": False}


def _freeze(context: Mapping[str, Any], synthetic: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = _source_paths()
    input_paths = _input_paths(context["train_path"])
    missing = [str(path) for path in [*source_paths, *input_paths] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Measured Classification pre-fit inputs missing: {missing}")
    inputs = {_relative(path): file_sha256(path) for path in input_paths}
    sources = {_relative(path): file_sha256(path) for path in source_paths}
    records: dict[int, dict[str, Any]] = {}
    estimates: dict[int, dict[str, Any]] = {}
    for cutoff in CONTINUATION_ORIGINS:
        datum = context["data"][cutoff]
        train_features = context["frame"].iloc[datum["fit_rows"]].loc[:, list(FEATURES)]
        score_features = context["frame"].iloc[datum["rows"]].loc[:, list(FEATURES)]
        estimate = n02._resource_estimate(train_features, score_features)
        if estimate["status"] != "PASS":
            raise RuntimeError(f"N02/{cutoff} failed the fixed fit resource preflight: {estimate}")
        estimates[cutoff] = estimate
        records[cutoff] = {
            "fit_rows": int(len(datum["fit_rows"])), "score_rows": int(len(datum["rows"])),
            "fit_rows_sha256": hashlib.sha256(datum["fit_rows"].astype("<i8").tobytes()).hexdigest(),
            "score_rows_sha256": hashlib.sha256(datum["rows"].astype("<i8").tobytes()).hexdigest(),
            "control_cache_hashes": {"npz": datum["cache_npz_sha256"], "sidecar": datum["cache_sidecar_sha256"]},
        }
    ledger_state = ledger_summary(LEDGER)
    if (ledger_state["started_and_consumed"] != 30 or ledger_state["saved"] != 30
            or ledger_state["failed_or_interrupted_started"] != 0
            or ledger_state["authorized_cap"] != 60):
        raise RuntimeError(f"Research fit budget no longer matches the authorized preflight state: {ledger_state}")
    recipe = {
        "candidate": "N02_C2B_SKETCH", "control": "N02_C2B_NO_SKETCH",
        "cutoffs": list(CONTINUATION_ORIGINS), "feature_allowlist": list(FEATURES),
        "sketch_parameters": dict(n02.SKETCH_PARAMS), "threshold": 0.50,
        "route": ".75 * v3_cls_sparse25 route + .25 * new expert on present; frozen CB_MISSING on missing",
        "matched_control": "same C2_B LR fit and route without polynomial count sketch",
        "fit_scope": "official Classification Train cycles 47..cutoff; score next four complete cycles",
        "test_labels_used": False, "test_data_used": False, "kaggle_contacted": False,
    }
    identity = {
        "task_id": TASK, "source_commit": _head(), "protocol_sha256": file_sha256(PROTOCOL),
        "research_next_freeze_sha256": file_sha256(n02.FREEZE_PATH),
        "n02_pilot_freeze_signature": context["n02_frozen"]["freeze_signature"],
        "official_train_sha256": file_sha256(context["train_path"]),
        "source_sha256": sources, "input_sha256": inputs,
        "fit_plan": {str(key): value for key, value in records.items()}, "recipe": recipe,
        "resource_estimates": {str(key): value for key, value in estimates.items()},
        "budget_before": ledger_state, "dependencies": _dependencies(),
        "synthetic_preflight": dict(synthetic),
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
    }
    return {"schema_version": 1, "status": "FROZEN_BEFORE_OFFICIAL_FIT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity, "freeze_signature": canonical_sha256(identity)}


def _verify_freeze(context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    identity = frozen.get("identity")
    if frozen.get("freeze_signature") != canonical_sha256(identity):
        raise RuntimeError("Measured Classification freeze signature mismatch")
    if not n02._is_ancestor(str(identity.get("source_commit", ""))):
        raise RuntimeError("Measured Classification source commit is not an ancestor of current HEAD")
    for collection in ("source_sha256", "input_sha256"):
        for relative, digest in identity[collection].items():
            path = ROOT / relative
            if not path.is_file() or file_sha256(path) != digest:
                raise RuntimeError(f"Measured Classification frozen hash changed: {relative}")
    if context is not None and file_sha256(context["train_path"]) != identity["official_train_sha256"]:
        raise RuntimeError("Official Train hash differs from measured freeze")
    return frozen


def _bundle_identity(signature: str, cutoff: int, kind: str) -> dict[str, Any]:
    return {"batch_id": TASK, "card_id": "MEASURED_N02", "stage": "CONTINUATION",
            "freeze_signature": signature, "cutoff": int(cutoff), "fit_kind": kind}


def _bundle_path(cutoff: int, kind: str) -> Path:
    return FIT_ROOT / "N02_CONTINUATION" / f"cutoff_{cutoff}" / kind


def _fit_id(cutoff: int, kind: str) -> str:
    return f"MEASURED_N02_CONT:{cutoff}:{kind}"


def _load_fit(context: Mapping[str, Any], freeze: Mapping[str, Any], cutoff: int,
              kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = _bundle_path(cutoff, kind)
    fit_id = _fit_id(cutoff, kind)
    events = read_events(LEDGER)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if not path.exists():
        if starts:
            raise RuntimeError(f"Started-unsaved measured fit {fit_id}; recover without refitting")
        return None
    if len(starts) != 1 or any(e["event"] == "FIT_FAILED" for e in terminals):
        raise RuntimeError(f"Measured fit bundle lacks unique started lineage: {fit_id}")
    bundle = read_bundle(path, _bundle_identity(freeze["freeze_signature"], cutoff, kind), task="classification")
    datum = context["data"][cutoff]
    for key, expected in (("score_rows", datum["rows"]), ("score_record_ids", datum["record_ids"]),
                          ("cycle_code", datum["cycles"])):
        if not np.array_equal(bundle.arrays[key], expected):
            raise RuntimeError(f"Saved measured fit {fit_id} has changed row alignment: {key}")
    metadata = bundle.receipt.get("metadata", {}).get("metadata", {})
    if (metadata.get("fit_kind") != kind or metadata.get("cutoff") != cutoff
            or metadata.get("test_data_used") is not False or metadata.get("kaggle_called") is not False):
        raise RuntimeError(f"Saved measured fit {fit_id} receipt failed boundary checks")
    if terminals:
        if len(terminals) != 1 or terminals[0].get("event") != "FIT_SAVED" or terminals[0].get("receipt_sha256") != bundle.receipt["receipt_sha256"]:
            raise RuntimeError(f"Saved fit receipt differs from ledger for {fit_id}")
    else:
        finish_fit(LEDGER, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=bundle.receipt["receipt_sha256"])
    return {key: value.copy() for key, value in bundle.arrays.items()}, bundle.receipt


def _fit_one(context: Mapping[str, Any], freeze: Mapping[str, Any], cutoff: int,
             kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    existing = _load_fit(context, freeze, cutoff, kind)
    if existing is not None:
        return existing
    datum = context["data"][cutoff]
    fit_rows = datum["fit_rows"]
    rows = datum["rows"]
    frame = context["frame"]
    fit_x = frame.iloc[fit_rows].loc[:, list(FEATURES)]
    score_x = frame.iloc[rows].loc[:, list(FEATURES)]
    estimate = n02._resource_estimate(fit_x, score_x)
    if estimate["status"] != "PASS":
        raise RuntimeError(f"Fresh N02/{cutoff} resource check failed: {estimate}")
    fit_id = _fit_id(cutoff, kind)
    start_fit(LEDGER, fit_id=fit_id, card_id="MEASURED_N02", stage="CONTINUATION",
              cutoff=cutoff, component=kind)
    try:
        with exclusive_training_slot(LOCK):
            fresh_ram = int(psutil.virtual_memory().available)
            estimate = n02._resource_estimate(fit_x, score_x, available_bytes=fresh_ram)
            if estimate["status"] != "PASS":
                raise RuntimeError(f"In-lock N02/{cutoff} resource check failed: {estimate}")
            with threadpool_limits(limits=THREADS):
                model = n02.C2BInteractionClassifier(include_sketch=(kind == "sketch"))
                model.fit(fit_x, context["target"][fit_rows])
                raw = model.predict_probability(score_x)
            routes = n02.route_expert_probabilities(
                raw_candidate=raw if kind == "sketch" else np.full_like(raw, 0.5),
                raw_no_sketch=raw if kind == "no_sketch" else np.full_like(raw, 0.5),
                current_route=datum["current"], cb_probability=datum["cb"],
                stream_missing=datum["missing"],
            )
            prediction = routes["candidate"] if kind == "sketch" else routes["no_sketch_control"]
            payload = {
                "score_rows": rows.astype(np.int64), "score_record_ids": datum["record_ids"].astype(str),
                "cycle_code": datum["cycles"].astype(np.int64),
                "origin_cutoff": np.full(len(rows), cutoff, dtype=np.int64),
                "prediction": prediction.astype(np.float64),
                "baseline_prediction": routes["current_route"].astype(np.float64),
                "raw_prediction": raw.astype(np.float64),
                "control__current_route": datum["current"].astype(np.float64),
                "control__cb_probability": datum["cb"].astype(np.float64),
                "stream_missing": datum["missing"].astype(bool),
                "metadata": {"fit_kind": kind, "cutoff": cutoff, "fit_rows": int(len(fit_rows)),
                             "score_rows": int(len(rows)), "fit_state": model.audit_state(),
                             "resource_preflight": estimate, "threshold": 0.50,
                             "test_data_used": False, "test_labels_passed_to_fit": False,
                             "kaggle_called": False, "protocol_sha256": file_sha256(PROTOCOL)},
            }
            identity = _bundle_identity(freeze["freeze_signature"], cutoff, kind)
            bundle = resume_or_publish_bundle(_bundle_path(cutoff, kind), payload, identity, task="classification")
            expected_candidate = np.where(datum["missing"], datum["cb"],
                                         0.75 * datum["current"] + 0.25 * raw)
            expected_base = np.where(datum["missing"], datum["cb"], datum["current"])
            if (not np.allclose(bundle.arrays["prediction"], expected_candidate if kind == "sketch" else
                                np.where(datum["missing"], datum["cb"], 0.75 * datum["current"] + 0.25 * raw),
                                rtol=0, atol=1e-12)
                    or not np.allclose(bundle.arrays["baseline_prediction"], expected_base, rtol=0, atol=1e-12)):
                raise RuntimeError(f"Measured N02 route reconstruction failed for {fit_id}")
            finish_fit(LEDGER, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=bundle.receipt["receipt_sha256"])
            return {key: value.copy() for key, value in bundle.arrays.items()}, bundle.receipt
    except Exception as exc:
        events = read_events(LEDGER)
        done = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not done:
            finish_fit(LEDGER, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def _pack_new(context: Mapping[str, Any], cutoff: int, arrays: Mapping[str, np.ndarray],
              raw_key: str = "prediction") -> dict[str, np.ndarray]:
    datum = context["data"][cutoff]
    pred = np.asarray(arrays[raw_key], dtype=np.float64)
    raw = np.asarray(arrays["raw_prediction"], dtype=np.float64)
    return {"rows": datum["rows"], "record_ids": datum["record_ids"], "cycles": datum["cycles"],
            "origin": np.full(len(pred), cutoff, dtype=np.int64), "candidate": pred,
            "current": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
            "matched": np.where(datum["missing"], datum["cb"], 0.75 * datum["current"] + 0.25 * raw),
            "missing": datum["missing"]}


def _pilot_output(context: Mapping[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    old_context = context["n02_parent"]
    sk, _ = n02._load_existing_fit(old_context, cutoff, "sketch") or (None, None)
    ns, _ = n02._load_existing_fit(old_context, cutoff, "no_sketch") or (None, None)
    if sk is None or ns is None:
        raise RuntimeError(f"Frozen N02 pilot output missing at cutoff {cutoff}")
    merged = n02._merge_origin(old_context, cutoff, {"sketch": sk, "no_sketch": ns})
    return {"rows": merged["rows"], "record_ids": merged["record_ids"], "cycles": merged["cycles"],
            "origin": merged["origin"], "candidate": merged["candidate"],
            "current": merged["current"], "matched": merged["matched"], "missing": merged["missing"]}


def _collect(outputs: Mapping[int, Mapping[str, np.ndarray]], origins: tuple[int, ...],
             lohi: Mapping[int, tuple[int, int]] | None = None) -> dict[str, np.ndarray]:
    chunks = []
    for cutoff in origins:
        record = outputs[cutoff]
        if lohi is None:
            mask = np.ones(len(record["rows"]), dtype=bool)
        else:
            lo, hi = lohi[cutoff]
            mask = (record["cycles"] >= lo) & (record["cycles"] <= hi)
        chunks.append({key: np.asarray(value)[mask] for key, value in record.items()})
    result = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    if len(np.unique(result["rows"])) != len(result["rows"]):
        raise RuntimeError("A measured Classification scoring view has duplicate Train row IDs")
    order = np.argsort(result["rows"], kind="mergesort")
    return {key: value[order] for key, value in result.items()}


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    labels = (np.asarray(probability, dtype=np.float64) >= THRESHOLD).astype(np.int8)
    target = np.asarray(y, dtype=np.int8)
    if labels.shape != target.shape or not np.isin(target, (0, 1)).all():
        raise ValueError("Classification metric arrays do not align")
    tn = int(np.sum((target == 0) & (labels == 0)))
    fp = int(np.sum((target == 0) & (labels == 1)))
    fn = int(np.sum((target == 1) & (labels == 0)))
    tp = int(np.sum((target == 1) & (labels == 1)))
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(target)), "macro_f1": 0.5 * (f1_0 + f1_1),
            "positive_recall": recall, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _views(outputs: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    windows = {f"W{i}": (ORIGINS[i - 1],) for i in range(1, 7)}
    definitions: dict[str, tuple[tuple[int, ...], Mapping[int, tuple[int, int]] | None]] = {
        **{name: (origins, None) for name, origins in windows.items()},
        "PRIMARY_FULL_SIX": (ORIGINS, None),
        "LATE_W5_W6": ((71, 75), None),
        "H12_CUTOFF_55": ((55,), None),
        "H12_CUTOFF_67": ((67,), None),
        "H12_POOLED": ((55, 67), None),
    }
    primary_ranges = {
        55: (56, 59), 59: (60, 63), 63: (64, 67),
        67: (68, 71), 71: (72, 75), 75: (76, 79),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (origins, _unused) in definitions.items():
        records: dict[int, Mapping[str, np.ndarray]] = {}
        for cutoff in origins:
            records[cutoff] = outputs[cutoff]
        if name.startswith("H12"):
            lohi = {cutoff: (cutoff + 1, min(cutoff + 12, 79)) for cutoff in origins}
            # Pilot records already contain their complete H12 horizon; new-origin records contain W2/W3/W5/W6 only.
        else:
            lohi = {cutoff: primary_ranges[cutoff] for cutoff in origins}
        data = _collect(records, tuple(origins), lohi)
        result[name] = data
    return result


def _group_rows(context: Mapping[str, Any], views: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    frame, target = context["frame"], context["target"]
    rows: list[dict[str, Any]] = []
    for view_name, data in views.items():
        for cutoff in np.unique(data["origin"]):
            origin_mask = data["origin"] == cutoff
            source_rows = data["rows"][origin_mask].astype(np.int64)
            y = target[source_rows]
            local = frame.iloc[source_rows]
            quartiles, edges = n01._text_quartiles(frame, int(cutoff))
            group_fields = {
                "entry_friction": local["entry_friction"].astype(str).to_numpy(),
                "stream_availability": np.where(data["missing"][origin_mask], "missing", "present"),
                "fit_defined_text_load_quartile": quartiles[source_rows].astype(str),
            }
            for field, labels in group_fields.items():
                for label in sorted(np.unique(labels)):
                    mask = labels == label
                    yy = y[mask]
                    n = int(mask.sum())
                    class0 = int(np.sum(yy == 0))
                    class1 = int(np.sum(yy == 1))
                    supported = n >= MIN_GROUP_N and min(class0, class1) >= MIN_EACH_CLASS_N
                    candidate = _metric(yy, data["candidate"][origin_mask][mask])
                    current = _metric(yy, data["current"][origin_mask][mask])
                    matched = _metric(yy, data["matched"][origin_mask][mask])
                    f1_harm = current["macro_f1"] - candidate["macro_f1"]
                    recall_harm = current["positive_recall"] - candidate["positive_recall"]
                    matched_f1_harm = current["macro_f1"] - matched["macro_f1"]
                    matched_recall_harm = current["positive_recall"] - matched["positive_recall"]
                    rows.append({
                        "view": view_name, "origin": int(cutoff), "field": field, "group": str(label),
                        "n": n, "class_0_n": class0, "class_1_n": class1, "supported": supported,
                        "candidate_macro_f1": candidate["macro_f1"], "current_macro_f1": current["macro_f1"],
                        "candidate_f1_harm_vs_current": f1_harm,
                        "candidate_positive_recall_harm_vs_current": recall_harm,
                        "no_sketch_macro_f1": matched["macro_f1"],
                        "no_sketch_positive_recall": matched["positive_recall"],
                        "no_sketch_f1_harm_vs_current": matched_f1_harm,
                        "no_sketch_positive_recall_harm_vs_current": matched_recall_harm,
                        "original_0_02_recall_guard_fail": bool(supported and recall_harm > ORIGINAL_MAX_RECALL_HARM),
                        "benchmark_0_03_recall_guard_fail": bool(supported and recall_harm > BENCHMARK_MAX_RECALL_HARM),
                        "f1_guard_fail": bool(supported and f1_harm > MAX_F1_HARM),
                        "fit_only_text_load_edges": edges if field == "fit_defined_text_load_quartile" else None,
                    })
    return rows


def _supported_groups_pass(groups: list[Mapping[str, Any]]) -> bool:
    supported = [group for group in groups if group["supported"]]
    return all(not (group["f1_guard_fail"] or group["benchmark_0_03_recall_guard_fail"])
               for group in supported)


def _evaluate(context: Mapping[str, Any], outputs: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    views = _views(outputs)
    metrics: dict[str, Any] = {}
    for name, data in views.items():
        y = context["target"][data["rows"]]
        cand, base, matched = (_metric(y, data[k]) for k in ("candidate", "current", "matched"))
        metrics[name] = {"n": int(len(y)), "candidate": cand, "sparse25_current": base,
                         "no_sketch_matched_control": matched,
                         "gain_vs_sparse25": float(cand["macro_f1"] - base["macro_f1"]),
                         "gain_vs_no_sketch": float(cand["macro_f1"] - matched["macro_f1"])}
    groups = _group_rows(context, views)
    supported = [g for g in groups if g["supported"]]
    guard_ok = _supported_groups_pass(groups)
    original_recall_violations = [g for g in supported if g["original_0_02_recall_guard_fail"]]
    gates = {
        "primary_gain_vs_sparse25_ge_0_0002": metrics["PRIMARY_FULL_SIX"]["gain_vs_sparse25"] >= 0.0002,
        "pooled_h12_harm_vs_sparse25_le_0_001": metrics["H12_POOLED"]["gain_vs_sparse25"] >= -0.001,
        "late_w5_w6_harm_vs_sparse25_le_0_001": metrics["LATE_W5_W6"]["gain_vs_sparse25"] >= -0.001,
        "w6_harm_vs_sparse25_le_0_002": metrics["W6"]["gain_vs_sparse25"] >= -0.002,
        "at_least_3_of_6_windows_nonnegative": sum(metrics[f"W{i}"]["gain_vs_sparse25"] >= 0 for i in range(1, 7)) >= 3,
        "primary_gain_vs_no_sketch_nonnegative": metrics["PRIMARY_FULL_SIX"]["gain_vs_no_sketch"] >= 0,
        "all_supported_group_f1_harm_le_0_01_and_recall_harm_le_0_03": guard_ok,
    }
    return {"metrics": metrics, "groups": groups, "gates": gates,
            "decision": "BENCHMARK_GATES_PASS" if all(gates.values()) else "BENCHMARK_GATES_FAIL",
            "original_0_02_recall_guard_violations": original_recall_violations,
            "supported_group_count": len(supported), "unsupported_group_count": len(groups) - len(supported),
            "guard_note": "0.03 recall ceiling is a benchmark-only risk amendment. Original 0.02 violations remain listed and do not become PASS."}


def _build_outputs(context: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
    outputs = {55: _pilot_output(context, 55), 67: _pilot_output(context, 67)}
    for cutoff in CONTINUATION_ORIGINS:
        pair = {kind: results[(cutoff, kind)][0] for kind in FIT_KINDS}
        sk, ns = pair["sketch"], pair["no_sketch"]
        if not np.array_equal(sk["score_rows"], ns["score_rows"]):
            raise RuntimeError(f"N02/{cutoff} matched pair score rows differ")
        datum = context["data"][cutoff]
        for key, expected in (("score_rows", datum["rows"]), ("cycle_code", datum["cycles"]),
                              ("score_record_ids", datum["record_ids"])):
            if not np.array_equal(sk[key], expected) or not np.array_equal(ns[key], expected):
                raise RuntimeError(f"N02/{cutoff} continuation output alignment failed: {key}")
        outputs[cutoff] = {
            "rows": datum["rows"], "record_ids": datum["record_ids"], "cycles": datum["cycles"],
            "origin": np.full(len(datum["rows"]), cutoff, dtype=np.int64),
            "candidate": sk["prediction"], "current": sk["baseline_prediction"],
            "matched": ns["prediction"], "missing": datum["missing"],
        }
    return outputs


def run_temporal() -> dict[str, Any]:
    context = _context()
    frozen = _verify_freeze(context)
    budget = ledger_summary(LEDGER)
    if budget["started_and_consumed"] < 30 or budget["failed_or_interrupted_started"]:
        raise RuntimeError(f"Unexpected measured fit ledger before continuation: {budget}")
    reserve_card(LEDGER, card_id="MEASURED_N02", stage="CONTINUATION", fits=8, category="continuation")
    results: dict[tuple[int, str], tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    try:
        for cutoff in CONTINUATION_ORIGINS:
            for kind in FIT_KINDS:
                results[(cutoff, kind)] = _fit_one(context, frozen, cutoff, kind)
    finally:
        release_unused(LEDGER, card_id="MEASURED_N02", stage="CONTINUATION")
    outputs = _build_outputs(context, results)
    analysis = _evaluate(context, outputs)
    fit_index = {f"{cutoff}:{kind}": {"bundle": _relative(_bundle_path(cutoff, kind)),
                                       "receipt_sha256": results[(cutoff, kind)][1]["receipt_sha256"]}
                 for cutoff in CONTINUATION_ORIGINS for kind in FIT_KINDS}
    report = {
        "schema_version": 1, "task_id": TASK, "freeze_signature": frozen["freeze_signature"],
        "status": analysis["decision"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_fits_in_this_stage": 8, "fit_receipts": fit_index,
        "fit_ledger_after_stage": ledger_summary(LEDGER),
        "scope": {"official_train_rows": len(context["frame"]), "test_data_used": False,
                  "sample_submission_used": False, "kaggle_called": False,
                  "candidate_and_matched_control_fits": 4, "temporal_origins": list(CONTINUATION_ORIGINS)},
        **analysis,
    }
    _write_once(REPORT_ROOT / "N02_CONTINUATION_RESULTS.json", report)
    pd.DataFrame(analysis["groups"]).to_csv(REPORT_ROOT / "N02_CONTINUATION_GROUPS.csv", index=False,
                                             lineterminator="\n", float_format="%.12g")
    return report


class FullTrainClassOnly(ClassByVolumeSparseClassifier):
    """N01 exact estimator implementation, with only the preauthorized cutoff 79."""

    def __init__(self) -> None:
        self.cutoff = 79
        self.weight_kind = "class_only"
        self._fitted = False


def _submission_frame(test: pd.DataFrame, sample: pd.DataFrame, labels: np.ndarray,
                      target_name: str) -> pd.DataFrame:
    if list(sample.columns) != ["record_id", target_name] or "record_id" not in test:
        raise RuntimeError("Official SampleSubmission schema differs from the expected competition contract")
    ids = test["record_id"].astype(str).reset_index(drop=True)
    if ids.duplicated().any() or len(ids) != 29_421:
        raise RuntimeError("Official Test IDs have duplicates or unexpected row count")
    frame = pd.DataFrame({"record_id": ids, target_name: np.asarray(labels, dtype=np.int8)})
    if not frame.record_id.equals(ids) or set(frame[target_name].unique()) - {0, 1}:
        raise RuntimeError("Classification submission labels or order failed validation")
    return frame


def _fit_fulltrain_candidate(*, candidate: str, train: pd.DataFrame, test: pd.DataFrame,
                             current: np.ndarray, c2b: np.ndarray, cb: np.ndarray,
                             missing: np.ndarray) -> dict[str, Any]:
    y = train.artifact_signal.to_numpy(dtype=np.int8)
    if len(train) != 50_608 or not np.array_equal(np.unique(y), np.array([0, 1])):
        raise RuntimeError("Full-Train classification fit population/target failed validation")
    features = train.loc[:, list(FEATURES)]
    test_features = test.loc[:, list(FEATURES)]
    if candidate == "CLASS_ONLY":
        plan = class_by_volume_weight_plan(features, y)
        model: Any = FullTrainClassOnly()
        fit_id, card_id = "MEASURED_CLASS_ONLY_FULLTRAIN", "MEASURED_CLASS_ONLY"
        stage = "BENCHMARK_FULLTRAIN"
        weights = plan["class_only_weights"]
        raw_resource = {"status": "PASS" if psutil.virtual_memory().available >= MIN_FREE_BYTES else "BLOCKED_RESOURCE",
                        "available_physical_bytes": int(psutil.virtual_memory().available),
                        "estimated_peak_bytes": 2 * 1024**3, "job_cap_bytes": JOB_CAP_BYTES,
                        "reserve_bytes": RESERVE_BYTES, "threads": THREADS}
    elif candidate == "N02_SKETCH":
        model = n02.C2BInteractionClassifier(include_sketch=True)
        fit_id, card_id = "MEASURED_N02_FULLTRAIN", "MEASURED_N02"
        stage = "BENCHMARK_FULLTRAIN"
        weights = None
        raw_resource = n02._resource_estimate(features, test_features)
    else:
        raise ValueError(candidate)
    if raw_resource["status"] != "PASS":
        raise RuntimeError(f"{candidate} full-Train fit failed RAM preflight: {raw_resource}")
    bundle_path = FIT_ROOT / "FULLTRAIN" / candidate
    events = read_events(LEDGER)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if bundle_path.exists():
        if len(starts) != 1 or len(terminals) != 1 or terminals[0].get("event") != "FIT_SAVED":
            raise RuntimeError(f"Existing {candidate} full-Train bundle has invalid ledger lineage")
        identity = {"batch_id": TASK, "card_id": card_id, "stage": stage,
                    "freeze_signature": json.loads(FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
                    "candidate": candidate, "cutoff": 79}
        saved = read_bundle(bundle_path, identity, task="classification")
        if terminals[0].get("receipt_sha256") != saved.receipt["receipt_sha256"]:
            raise RuntimeError(f"Existing {candidate} bundle differs from fit ledger receipt")
        metadata = saved.receipt.get("metadata", {}).get("metadata", {})
        return {"candidate": candidate, "fit_id": fit_id, "bundle": _relative(bundle_path),
                "receipt_sha256": saved.receipt["receipt_sha256"], "fit_state": metadata["fit_state"],
                "probability": saved.arrays["prediction"].astype(np.float64),
                "labels": saved.arrays["label"].astype(np.int8),
                "raw": saved.arrays["raw_prediction"].astype(np.float64),
                "resource_preflight": metadata["resource_preflight"], "train_rows": len(train)}
    if starts:
        raise RuntimeError(f"Started-unsaved {candidate} full-Train fit; do not refit")
    reserve_card(LEDGER, card_id=card_id, stage=stage, fits=1, category="continuation")
    start_fit(LEDGER, fit_id=fit_id, card_id=card_id, stage=stage, cutoff=79, component=candidate)
    try:
        with exclusive_training_slot(LOCK):
            fresh = int(psutil.virtual_memory().available)
            if candidate == "N02_SKETCH":
                raw_resource = n02._resource_estimate(features, test_features, available_bytes=fresh)
                if raw_resource["status"] != "PASS":
                    raise RuntimeError(f"N02 full-Train in-lock resource preflight failed: {raw_resource}")
            elif fresh < MIN_FREE_BYTES + RESERVE_BYTES:
                raise RuntimeError(f"Class-only full-Train in-lock memory below guard: {fresh}")
            with threadpool_limits(limits=THREADS):
                if weights is None:
                    model.fit(features, y)
                else:
                    model.fit(features, y, sample_weight=weights)
                raw = model.predict_probability(test_features)
            if candidate == "CLASS_ONLY":
                probability = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
            else:
                probability = np.where(missing, cb, 0.75 * current + 0.25 * raw)
            if probability.shape != (len(test),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise RuntimeError(f"{candidate} full-Train inference returned invalid probabilities")
            labels = (probability >= 0.50).astype(np.int8)
            arrays = {"score_rows": np.arange(len(test), dtype=np.int64),
                      "score_record_ids": test.record_id.astype(str).to_numpy(dtype="U"),
                      "cycle_code": pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64),
                      "raw_prediction": raw.astype(np.float64), "prediction": probability.astype(np.float64),
                      "label": labels.astype(np.int8),
                      "metadata": {"candidate": candidate, "fit_rows": 50_608, "cutoff": 79,
                                   "fit_state": model.audit_state(), "resource_preflight": raw_resource,
                                   "preprocessing_fit_only_official_train": True,
                                   "test_data_used_only_for_inference": True, "test_labels_used": False,
                                   "threshold": 0.50, "kaggle_called": False}}
            identity = {"batch_id": TASK, "card_id": card_id, "stage": stage,
                        "freeze_signature": json.loads(FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
                        "candidate": candidate, "cutoff": 79}
            bundle = resume_or_publish_bundle(bundle_path, arrays, identity, task="classification")
            finish_fit(LEDGER, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=bundle.receipt["receipt_sha256"])
            return {"candidate": candidate, "fit_id": fit_id, "bundle": _relative(bundle_path),
                    "receipt_sha256": bundle.receipt["receipt_sha256"], "fit_state": model.audit_state(),
                    "probability": probability, "labels": labels, "raw": raw,
                    "resource_preflight": raw_resource, "train_rows": len(train)}
    except Exception as exc:
        events = read_events(LEDGER)
        done = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not done:
            finish_fit(LEDGER, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        release_unused(LEDGER, card_id=card_id, stage=stage)


def _prepare_one_submission(candidate: str, fit: Mapping[str, Any], train: pd.DataFrame,
                            test: pd.DataFrame, sample: pd.DataFrame, inputs: Mapping[str, str],
                            reuse: Mapping[str, Any]) -> dict[str, Any]:
    frame = _submission_frame(test, sample, fit["labels"], "artifact_signal")
    csv_path = ROOT / "submissions/optimization_v3/measured_benchmark_v1" / f"{candidate.lower()}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if csv_path.exists():
        if csv_path.read_bytes() != expected_bytes:
            raise FileExistsError(f"Existing measured Classification CSV differs; refusing overwrite: {csv_path}")
    else:
        with csv_path.open("xb") as stream:
            stream.write(expected_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    reread = pd.read_csv(csv_path)
    if (len(reread) != 29_421 or list(reread.columns) != ["record_id", "artifact_signal"]
            or reread.record_id.astype(str).duplicated().any()
            or not reread.record_id.astype(str).equals(test.record_id.astype(str).reset_index(drop=True))
            or not np.isin(reread.artifact_signal.to_numpy(), [0, 1]).all()):
        raise RuntimeError(f"{candidate} submission CSV failed round-trip QA")
    csv_hash = file_sha256(csv_path)
    manifest = {
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED", "task": "classification",
        "candidate": candidate, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _head(), "freeze_signature": json.loads(FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
        "recipe": {"CLASS_ONLY": "N01 exact class_only sparse expert; .75 C2_B + .25 expert on present; frozen CB_MISSING on missing; threshold .50",
                   "N02_SKETCH": "C2_B+PolynomialCountSketch degree2/1024 LR; .75 v3 sparse25 route + .25 expert on present; frozen CB_MISSING on missing; threshold .50"}[candidate],
        "official_input_sha256": dict(inputs), "fit_rows": 50_608, "test_rows": 29_421,
        "fit_state_sha256": fit["fit_state"].get("state_sha256"),
        "fit_bundle": fit["bundle"], "fit_receipt_sha256": fit["receipt_sha256"],
        "fit_ledger": ledger_summary(LEDGER), "reuse_audit": dict(reuse),
        "test_labels_used": False, "preprocessing_fit_only_train": True,
        "csv": {"path_gitignored": csv_path.relative_to(ROOT).as_posix(), "sha256": csv_hash,
                "rows": len(frame), "columns": list(frame.columns),
                "ordered_ids_sha256": hashlib.sha256(("\n".join(frame.record_id.astype(str)) + "\n").encode()).hexdigest(),
                "labels": {str(int(k)): int(v) for k, v in zip(*np.unique(fit["labels"], return_counts=True))}},
        "kaggle_called": False, "submission_ref": None,
    }
    _write_once(REPORT_ROOT / f"{candidate}_SUBMISSION_MANIFEST.json", manifest)
    return manifest


def run_fulltrain() -> dict[str, Any]:
    context = _context()
    frozen = _verify_freeze(context)
    temporal_path = REPORT_ROOT / "N02_CONTINUATION_RESULTS.json"
    if not temporal_path.is_file():
        raise RuntimeError("N02 temporal selection must finish before full-Train inference is opened")
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    if temporal.get("freeze_signature") != frozen["freeze_signature"]:
        raise RuntimeError("N02 temporal results are not bound to the current measured freeze")
    allow_n02 = temporal.get("decision") == "BENCHMARK_GATES_PASS"
    files = task_files("classification")
    train_path, test_path, sample_path = files["train"], files["test"], files["sample"]
    if train_path != context["train_path"]:
        raise RuntimeError("Classification Train path changed before submission preparation")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    sample = pd.read_csv(sample_path, low_memory=False)
    # Revalidate official Train after opening Test; never read a Test target.
    if len(train) != 50_608 or len(test) != 29_421 or len(sample) != 29_421:
        raise RuntimeError("Official full-Train/Test/Sample row counts changed")
    if not test.record_id.astype(str).equals(sample.record_id.astype(str)):
        raise RuntimeError("Official Test IDs differ from SampleSubmission order")
    c2b, cb, missing, reuse = fulltrain._verify_classification_reuse(train, test)
    with np.load(fulltrain.CLASS_CACHE, allow_pickle=False) as cache:
        current = np.asarray(cache["prediction"], dtype=np.float64)
    c_manifest = json.loads(fulltrain.CLASS_MANIFEST.read_text(encoding="utf-8"))
    if (_array_sha(current) != c_manifest.get("prediction_sha256") or current.shape != (len(test),)
            or not np.isfinite(current).all() or np.any((current < 0) | (current > 1))):
        raise RuntimeError("Attested current sparse25 prediction vector is invalid")
    input_hashes = {"train": file_sha256(train_path), "test": file_sha256(test_path),
                    "sample": file_sha256(sample_path)}
    manifests: list[dict[str, Any]] = []
    class_fit = _fit_fulltrain_candidate(candidate="CLASS_ONLY", train=train, test=test, current=current,
                                         c2b=c2b, cb=cb, missing=missing)
    manifests.append(_prepare_one_submission("CLASS_ONLY", class_fit, train, test, sample, input_hashes, reuse))
    if allow_n02:
        sketch_fit = _fit_fulltrain_candidate(candidate="N02_SKETCH", train=train, test=test, current=current,
                                              c2b=c2b, cb=cb, missing=missing)
        manifests.append(_prepare_one_submission("N02_SKETCH", sketch_fit, train, test, sample, input_hashes, reuse))
    else:
        manifests.append({"candidate": "N02_SKETCH", "status": "NO_CSV_BENCHMARK_GATES_FAILED",
                          "gate_result": temporal["gates"], "kaggle_called": False})
    result = {"status": "FULLTRAIN_PREPARATION_COMPLETE", "candidates": manifests,
              "input_sha256": input_hashes, "fit_ledger": ledger_summary(LEDGER),
              "test_labels_used": False, "model_selection_on_test": False, "kaggle_called": False}
    _write_once(REPORT_ROOT / "FULLTRAIN_PREPARATION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "freeze", "temporal", "fulltrain"))
    args = parser.parse_args()
    if args.action == "preflight":
        context = _context()
        synthetic = _synthetic_preflight()
        report = {"status": "PASS", "synthetic": synthetic,
                  "ledger": ledger_summary(LEDGER), "official_fits": 0,
                  "resource_estimates": {str(c): n02._resource_estimate(
                      context["frame"].iloc[context["data"][c]["fit_rows"]].loc[:, list(FEATURES)],
                      context["frame"].iloc[context["data"][c]["rows"]].loc[:, list(FEATURES)])
                      for c in CONTINUATION_ORIGINS}}
        _write_once(REPORT_ROOT / "synthetic_preflight.json", report)
        print(json.dumps({"status": "PASS", "official_fits": 0,
                          "synthetic_fits": synthetic["synthetic_estimator_fits"]}, sort_keys=True))
    elif args.action == "freeze":
        context = _context()
        synthetic = json.loads((REPORT_ROOT / "synthetic_preflight.json").read_text(encoding="utf-8"))["synthetic"]
        if synthetic.get("status") != "PASS":
            raise RuntimeError("Synthetic end-to-end/persistence preflight has not passed")
        frozen = _freeze(context, synthetic)
        _write_once(FREEZE_PATH, frozen)
        print(json.dumps({"status": frozen["status"], "freeze_signature": frozen["freeze_signature"],
                          "source_commit": frozen["identity"]["source_commit"]}, sort_keys=True))
    elif args.action == "temporal":
        result = run_temporal()
        print(json.dumps({"status": result["status"], "decision": result["decision"],
                          "gates": result["gates"], "ledger": result["fit_ledger_after_stage"]}, sort_keys=True))
    else:
        result = run_fulltrain()
        print(json.dumps({"status": result["status"], "candidates": [x.get("candidate") for x in result["candidates"]],
                          "ledger": result["fit_ledger"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
