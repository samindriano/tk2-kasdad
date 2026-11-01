"""R11 cumulative ordinal regression preflight and generic pilot adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    package_versions,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r11 import (
    CARD_ID,
    FEATURE_NAMES,
    HEAD_PARAMETERS,
    MIN_CLASS_SUPPORT,
    SVD_PARAMETERS,
    TFIDF_PARAMETERS,
    fit_predict,
    threshold_support,
    validated_cycle_codes,
)
from src.models.regression_research_text import RegressionTextModel, _text_documents
from src.utils.paths import task_files
from scripts import optimization_v3_batch1_r02 as r02
from scripts import optimization_v3_batch1_r07 as r07


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R11.md"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
TRAIN_ROWS = 50_608
CUTOFFS = (55, 67)
THREADS = 2
MIN_FREE_GIB = 4.0
RESERVE_GIB = 2.0
JOB_CAP_GIB = 6.0
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R11"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R11"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R11"
ATTEMPT_PATH = REPORT_DIR / "attempts.jsonl"
PILOT_REPORT_PATH = ARTIFACT_DIR / "pilot.json"
LEDGER_PATH = ARTIFACT_DIR.parent / "batch_ledger.jsonl"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _require_committed_sources() -> None:
    sources = [path.resolve() for path in _source_files()]
    relative_sources: list[Path] = []
    for source in sources:
        try:
            relative = source.relative_to(ROOT)
        except ValueError as exc:
            raise BatchIntegrityError(f"R11 frozen source is outside the repository: {source}") from exc
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=ROOT, capture_output=True, text=True,
        )
        if tracked.returncode != 0:
            raise BatchIntegrityError(f"R11 source must be committed before freezing: {relative.as_posix()}")
        relative_sources.append(relative)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *[path.as_posix() for path in relative_sources]],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise BatchIntegrityError(f"R11 frozen sources have uncommitted changes; refusing to freeze: {status}")


def _require_committed_artifacts(paths: Sequence[Path]) -> None:
    relative_paths: list[str] = []
    for path in paths:
        try:
            relative = path.resolve().relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise BatchIntegrityError(f"R11 frozen artifact is outside the repository: {path}") from exc
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT, capture_output=True, text=True,
        )
        if tracked.returncode != 0:
            raise BatchIntegrityError(f"R11 frozen artifact must be committed before pilot: {relative}")
        relative_paths.append(relative)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise BatchIntegrityError(f"R11 frozen artifacts have uncommitted changes: {status}")


def _verify_live_freeze_hashes(freeze_identity: Mapping[str, Any]) -> None:
    queue = load_queue(QUEUE_PATH)
    cards = {str(item["id"]): item for item in queue.get("cards", [])}
    card = cards.get(CARD_ID)
    train_path = task_files("regression").get("train")
    if not isinstance(card, dict) or train_path is None or not train_path.is_file():
        raise BatchIntegrityError("R11 cannot verify current frozen inputs")
    views = evaluation_views_for_card(queue, CARD_ID)
    inputs = _input_files(train_path, card, queue)
    current = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=inputs,
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    for key in ("card", "feature_names", "input_sha256", "source_sha256",
                "package_versions", "evaluation_views"):
        if current.get(key) != freeze_identity.get(key):
            raise BatchIntegrityError(f"R11 current {key} differs from the saved freeze")


def _check_runtime_memory(preflight_path: Path) -> dict[str, float]:
    try:
        preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
        resource = preflight_record["resource"]
        cutoff_records = resource["cutoffs"].values()
        estimates = [
            float(item["estimated_peak_gib"])
            for item in cutoff_records
            if item.get("status") == "PASS"
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"R11 cannot read frozen memory estimates: {exc}") from exc
    if len(estimates) != len(CUTOFFS) or not np.isfinite(estimates).all():
        raise BatchIntegrityError("R11 frozen memory estimates do not cover both cutoffs")
    available = float(psutil.virtual_memory().available / (1024 ** 3))
    allowed = min(JOB_CAP_GIB, available - RESERVE_GIB)
    estimated_peak = max(estimates)
    if available < MIN_FREE_GIB:
        raise BatchIntegrityError(
            f"R11 runtime memory gate blocked: available={available:.2f} GiB is below {MIN_FREE_GIB:.2f} GiB"
        )
    if estimated_peak > allowed:
        raise BatchIntegrityError(
            f"R11 runtime memory gate blocked: estimated_peak={estimated_peak:.2f} GiB "
            f"exceeds current allowance={allowed:.2f} GiB"
        )
    return {
        "available_ram_gib_at_pilot_start": available,
        "current_allowed_estimated_peak_gib": allowed,
        "frozen_max_estimated_peak_gib": estimated_peak,
    }


def _validated_train_cycles(values: Any) -> np.ndarray:
    try:
        return validated_cycle_codes(values, name="Official Regression Train cycle_code")
    except (TypeError, ValueError) as exc:
        raise BatchIntegrityError(str(exc)) from exc


def _write_json_or_match(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchIntegrityError(f"Existing R11 artifact is unreadable: {path}: {exc}") from exc
        if existing != dict(payload):
            raise BatchIntegrityError(f"Existing R11 artifact differs; refusing overwrite: {path}")
        return
    write_json_exclusive(path, dict(payload))


def _assert_json_match_or_absent(path: Path, payload: Mapping[str, Any]) -> None:
    """Reject an existing conflicting artifact before publishing any sibling."""
    if not path.is_file():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Existing R11 artifact is unreadable: {path}: {exc}") from exc
    if existing != dict(payload):
        raise BatchIntegrityError(f"Existing R11 artifact differs; refusing overwrite: {path}")


def _assert_freeze_match_or_absent(path: Path, identity: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Existing R11 freeze is unreadable: {path}: {exc}") from exc
    if (existing.get("signature") != json_sha256(identity)
            or existing.get("identity") != json.loads(json.dumps(dict(identity), allow_nan=False))):
        raise BatchIntegrityError(f"Existing R11 freeze differs; refusing overwrite: {path}")


def _load_official_train() -> tuple[Path, dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    queue = load_queue(QUEUE_PATH)
    cards = {str(item["id"]): item for item in queue.get("cards", [])}
    card = cards.get(CARD_ID)
    if not isinstance(card, dict) or card.get("pilot_predictor_fits") != 18:
        raise BatchIntegrityError("Canonical R11 card or 18-fit reservation differs")
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R11 only runs while the V3 campaign remains active")
    if cards.get("R10", {}).get("status") != "SKIPPED_PREREQUISITE":
        raise BatchIntegrityError("R11 waits for the R10 condition lane to be terminal")
    if cards.get("R13", {}).get("status") != "SKIPPED_PREREQUISITE":
        raise BatchIntegrityError("R11 waits for the R13 condition lane to be terminal")
    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    if _sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from the R11 card")
    frame = pd.read_csv(train_path)
    if len(frame) != TRAIN_ROWS:
        raise BatchIntegrityError(f"Expected {TRAIN_ROWS} official Regression Train rows")
    missing = sorted((set(FEATURE_NAMES) | {"record_id", "cycle_code", "threat_rank"}) - set(frame.columns))
    if missing:
        raise BatchIntegrityError(f"Official Train is missing R11 fields: {missing}")
    record_ids = frame["record_id"].astype(str)
    if (frame["record_id"].isna().any() or record_ids.str.strip().eq("").any()
            or not record_ids.is_unique):
        raise BatchIntegrityError("Regression Train IDs are missing or duplicated")
    if set(FEATURE_NAMES) & {"record_id", "threat_rank", "artifact_signal"}:
        raise BatchIntegrityError("R11 feature allowlist contains a protected field")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    cycles = _validated_train_cycles(frame["cycle_code"])
    if (target.shape != (len(frame),) or not np.isfinite(target).all()
            or np.any((target < 0) | (target > 100))
            or not np.array_equal(np.unique(cycles), np.arange(47, 80))):
        raise BatchIntegrityError("Regression target or cycle support differs from the frozen protocol")
    return train_path, queue, frame, target, cycles


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    available_gib = available / (1024 ** 3)
    allowed_gib = min(JOB_CAP_GIB, available_gib - RESERVE_GIB)
    output: dict[str, Any] = {
        "available_ram_gib_observed": available_gib,
        "minimum_free_ram_gib": MIN_FREE_GIB,
        "reserve_gib": RESERVE_GIB,
        "job_cap_gib": JOB_CAP_GIB,
        "allowed_estimated_peak_gib": allowed_gib,
        "threads": THREADS,
        "concurrent_fit_jobs": 1,
        "candidate_head_fits_performed": 0,
        "preprocessing_probe": {
            "performed": True,
            "fit_only_tfidf_fit_calls": 0,
            "fit_only_svd_fit_calls": 0,
            "validation_labels_used": False,
            "lightgbm_head_fit_calls": 0,
        },
        "status": "PASS" if available_gib >= MIN_FREE_GIB else "BLOCKED_RESOURCE",
        "cutoffs": {},
    }
    if output["status"] != "PASS":
        output["resource_block_reason"] = "available memory below the frozen 4 GiB start floor"
        return output
    for cutoff in CUTOFFS:
        fit_mask = cycles <= cutoff
        score_mask = (cycles > cutoff) & (cycles <= cutoff + 12)
        fit_docs, _ = _text_documents(frame.loc[fit_mask, "signal_stream"])
        score_docs, _ = _text_documents(frame.loc[score_mask, "signal_stream"])
        vectorizer, fit_sparse, text_fallback = RegressionTextModel._fit_vectorizer(fit_docs)
        output["preprocessing_probe"]["fit_only_tfidf_fit_calls"] += 1
        if vectorizer is None or fit_sparse.shape[1] <= 32:
            output["status"] = "SKIPPED_SUPPORT"
            output["cutoffs"][str(cutoff)] = {
                "status": "SKIPPED_SUPPORT",
                "fit_rows": int(fit_mask.sum()),
                "score_rows": int(score_mask.sum()),
                "reason": "fit-only TF-IDF does not support 32 SVD components",
                "candidate_head_fits_performed": 0,
            }
            continue
        score_sparse = vectorizer.transform(score_docs).tocsr()
        svd = TruncatedSVD(n_components=32, **dict(SVD_PARAMETERS))
        fit_projection = svd.fit_transform(fit_sparse)
        output["preprocessing_probe"]["fit_only_svd_fit_calls"] += 1
        score_projection = svd.transform(score_sparse)
        sparse_bytes = int(fit_sparse.data.nbytes + fit_sparse.indices.nbytes + fit_sparse.indptr.nbytes
                           + score_sparse.data.nbytes + score_sparse.indices.nbytes + score_sparse.indptr.nbytes)
        projection_bytes = int(fit_projection.nbytes + score_projection.nbytes)
        # The estimate includes two dense design copies, the retained SVD
        # matrices, nine sequential LightGBM head workspaces, and process overhead.
        structured_width = 48  # Conservative upper bound from the expanded/OHE recipe.
        design_bytes = int((int(fit_mask.sum()) + int(score_mask.sum()))
                           * (structured_width + 32 + 1) * 8)
        estimated_peak = 1.5 + (2 * sparse_bytes + 2 * projection_bytes + 2 * design_bytes) / (1024 ** 3)
        cutoff_result = {
            "status": "PASS" if estimated_peak <= allowed_gib else "BLOCKED_RESOURCE",
            "fit_rows": int(fit_mask.sum()),
            "score_rows": int(score_mask.sum()),
            "tfidf_features": int(fit_sparse.shape[1]),
            "tfidf_fit_nnz": int(fit_sparse.nnz),
            "tfidf_score_nnz": int(score_sparse.nnz),
            "text_fallback": text_fallback,
            "sparse_bytes_observed": sparse_bytes,
            "svd_projection_bytes_observed": projection_bytes,
            "estimated_peak_gib": estimated_peak,
            "estimate_is_observed_peak": False,
            "head_count": 9,
            "candidate_head_fits_performed": 0,
            "fit_only_tfidf_fit_calls": 1,
            "fit_only_svd_fit_calls": 1,
            "validation_labels_used": False,
        }
        output["cutoffs"][str(cutoff)] = cutoff_result
        if cutoff_result["status"] != "PASS":
            output["status"] = "BLOCKED_RESOURCE"
    return output


def _input_files(train_path: Path, card: Mapping[str, Any],
                 queue: Mapping[str, Any]) -> dict[str, Path]:
    inputs = r02._input_files(train_path, card, queue)
    inputs["canonical_queue"] = QUEUE_PATH
    inputs["canonical_card"] = CARD_PATH
    return inputs


def _source_files() -> list[Path]:
    return sorted(set([
        *r07._source_files(),
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r11.py",
        ROOT / "tests/test_optimization_v3_batch1_r11.py",
    ]), key=str)


def _stable_resource_identity(resource: Mapping[str, Any]) -> dict[str, Any]:
    volatile = {"available_ram_gib_observed", "allowed_estimated_peak_gib"}
    result = {key: value for key, value in resource.items() if key not in volatile}
    if isinstance(result.get("cutoffs"), Mapping):
        result["cutoffs"] = {
            key: {name: value for name, value in entry.items() if name not in volatile}
            for key, entry in result["cutoffs"].items()
        }
    return result


def _budget_preflight(queue: Mapping[str, Any]) -> dict[str, int]:
    reservations: dict[str, int] = {}
    if LEDGER_PATH.is_file():
        for line_no, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid global fit ledger line {line_no}") from exc
            if item.get("event") == "BUDGET_RESERVED":
                key = str(item["budget_key"])
                count = int(item["predictor_fits"])
                if key in reservations:
                    raise BatchIntegrityError(f"Duplicate budget reservation for {key}")
                reservations[key] = count
    if "R11:PILOT" in reservations:
        raise BatchIntegrityError("R11 already has a fit reservation; inspect its existing checkpoint")
    cap = int(queue["execution_preset"]["max_successful_new_predictor_fits"])
    already_reserved = int(sum(reservations.values()))
    if already_reserved + 18 > cap:
        raise BatchIntegrityError(f"R11 fit budget blocked: {already_reserved}+18 exceeds {cap}")
    return {"cap": cap, "reserved_before": already_reserved, "R11_predictor_fits": 18,
            "reserved_after": already_reserved + 18}


def _prepare_context() -> dict[str, Any]:
    train_path, queue, frame, target, cycles = _load_official_train()
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R11 temporal views differ from the frozen protocol: {views}")
    support = {
        str(cutoff): threshold_support(target[cycles <= cutoff])
        for cutoff in CUTOFFS
    }
    unsupported = {
        cutoff: {threshold: counts for threshold, counts in values.items() if not counts["supported"]}
        for cutoff, values in support.items()
    }
    if any(unsupported.values()):
        raise BatchIntegrityError(f"R11 has unsupported cumulative targets: {unsupported}")
    zero_counts = {
        str(cutoff): int(np.sum((cycles <= cutoff) & (target == 0.0)))
        for cutoff in CUTOFFS
    }
    if any(zero_counts.values()):
        raise BatchIntegrityError(f"R11 median boundary assumption requires no zero target in fit folds: {zero_counts}")
    cards = {str(item["id"]): item for item in queue["cards"]}
    card = cards[CARD_ID]
    inputs = _input_files(train_path, card, queue)
    controls, control_identity = r07._control_preflight(
        frame, queue, inputs, views, target, cycles
    )
    cache_hashes = control_identity["cache_hashes"]
    control_metrics = control_identity["views"]
    resource = _resource_preflight(frame, cycles)
    if resource["status"] != "PASS":
        raise BatchIntegrityError(f"R11 resource preflight blocked: {resource}")
    if resource["preprocessing_probe"]["lightgbm_head_fit_calls"] != 0:
        raise BatchIntegrityError("R11 preflight unexpectedly fitted candidate heads")
    budget = _budget_preflight(queue)
    recipe = candidate_recipe()
    queue_hash = file_sha256(QUEUE_PATH)
    card_hash = file_sha256(CARD_PATH)
    source_commit = _git_head()
    return {
        "card": card,
        "queue": queue,
        "frame": frame,
        "target": target,
        "cycles": cycles,
        "train_path": train_path,
        "views": views,
        "controls": controls,
        "inputs": inputs,
        "cache_hashes": cache_hashes,
        "control_metrics": control_metrics,
        "control_identity": control_identity,
        "recipe": recipe,
        "resource": resource,
        "budget": budget,
        "source_commit": source_commit,
        "queue_hash": queue_hash,
        "card_hash": card_hash,
        "preflight": {
        "card_id": CARD_ID,
        "status": "PREFLIGHT_PASS",
        "source_commit": source_commit,
        "queue_sha256": queue_hash,
        "card_sha256": card_hash,
        "official_train_sha256": _sha256(train_path),
        "train_rows": int(len(frame)),
        "fit_rows": {str(c): int(np.sum(cycles <= c)) for c in CUTOFFS},
        "score_rows": {str(c): int(np.sum((cycles > c) & (cycles <= c + 12))) for c in CUTOFFS},
        "views": views,
        "threshold_support_by_cutoff": support,
        "fit_budget": budget,
        "target_boundary_audit": {
            "minimum_target": float(target.min()),
            "exact_zero_count_train": int(np.sum(target == 0.0)),
            "exact_zero_count_by_fit_cutoff": zero_counts,
            "S0_equals_1_assumption": True,
            "S100_equals_0_strict_y_gt_100": True,
        },
        "control_metrics": control_metrics,
        "control_cache_sha256": cache_hashes,
        "resource": resource,
        "official_test_read": False,
        "candidate_head_fits_performed": 0,
        },
    }


def preflight() -> dict[str, Any]:
    return _prepare_context()["preflight"]


def candidate_recipe() -> dict[str, Any]:
    def json_params(params: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in params.items():
            if isinstance(value, tuple):
                result[name] = list(value)
            elif isinstance(value, type):
                result[name] = value.__name__
            else:
                result[name] = value
        return result

    return {
        "card_id": CARD_ID,
        "model": "Nine cumulative binary LightGBM heads for threat_rank > 10,...,90; row-wise nonincreasing PAVA; linearly interpolated median crossing",
        "feature_allowlist": list(FEATURE_NAMES),
        "tfidf_parameters": json_params(TFIDF_PARAMETERS),
        "svd_parameters": {**json_params(SVD_PARAMETERS), "n_components": 32},
        "head_parameters": json_params(HEAD_PARAMETERS),
        "thresholds": list(range(10, 100, 10)),
        "minimum_support_per_class": MIN_CLASS_SUPPORT,
        "median_assumption": "Linear interpolation between decile thresholds approximates a continuous conditional median. S(0)=1 is valid only when each fit fold has no exact target zero; R11 fails closed otherwise. Official Train audit found minimum target 0.0428 and zero exact zeros; freeze preflight rechecks both folds. S(100)=0 follows strict y>100 being false.",
        "policy": "clip(0.75*current_three_component_champion + 0.25*rank50(raw target-scale R11 median), 0, 100)",
        "predictor_fits": 18,
        "test_data_used": False,
    }


def prepare_freeze() -> dict[str, Any]:
    """Read official Train, verify row-exact controls, then write immutable pre-fit artifacts."""
    _require_committed_sources()
    context = _prepare_context()
    recipe_record: dict[str, Any] = {
        "recipe": context["recipe"],
        "queue_sha256": context["queue_hash"],
        "card_sha256": context["card_hash"],
        "official_train_sha256": context["preflight"]["official_train_sha256"],
        "control_cache_sha256": context["cache_hashes"],
        "control_preflight_identity_sha256": json_sha256(context["control_identity"]),
        "source_commit": context["source_commit"],
    }
    identity = make_freeze_identity(
        card=context["card"],
        feature_names=FEATURE_NAMES,
        input_files=context["inputs"],
        source_files=_source_files(),
        evaluation_views=context["views"],
        packages=PACKAGES,
    )
    recipe_record.update({
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "package_versions": identity["package_versions"],
    })
    preflight_record = {
        **context["preflight"],
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "package_versions": identity["package_versions"],
        "candidate_head_fits_performed": 0,
        "freeze_written": True,
    }
    identity.update({
        "candidate_recipe": context["recipe"],
        "candidate_recipe_sha256": json_sha256(context["recipe"]),
        "recipe_record_sha256": json_sha256(recipe_record),
        "preflight_record_sha256": json_sha256(preflight_record),
        "source_commit": context["source_commit"],
        "control_preflight_identity_sha256": json_sha256(context["control_identity"]),
        "control_cache_sha256": context["cache_hashes"],
        "threshold_support_by_cutoff": context["preflight"]["threshold_support_by_cutoff"],
        "target_boundary_audit": context["preflight"]["target_boundary_audit"],
        "resource_preflight": _stable_resource_identity(context["resource"]),
        "python_version": platform.python_version(),
    })
    signature = json_sha256(identity)
    preflight_record["freeze_identity_sha256"] = signature
    # Catch stale outputs before any write can leave a partial artifact set.
    _assert_json_match_or_absent(RECIPE_PATH, recipe_record)
    _assert_json_match_or_absent(PREFLIGHT_PATH, preflight_record)
    _assert_freeze_match_or_absent(FREEZE_PATH, identity)
    _write_json_or_match(RECIPE_PATH, recipe_record)
    written_signature = ensure_frozen(FREEZE_PATH, identity)
    if written_signature != signature:
        raise BatchIntegrityError("R11 written freeze signature differs from prepared identity")
    _write_json_or_match(PREFLIGHT_PATH, preflight_record)
    return {
        "card_id": CARD_ID,
        "status": "FROZEN_NO_HEAD_FITS",
        "freeze_signature": signature,
        "recipe_path": str(RECIPE_PATH),
        "preflight_path": str(PREFLIGHT_PATH),
        "freeze_path": str(FREEZE_PATH),
        "candidate_head_fits_performed": 0,
        "budget_reserved": False,
        "official_test_read": False,
    }


def _independent_rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    cycle_values = validated_cycle_codes(cycles, name="R11 verifier cycle_code")
    if values.ndim != 1 or values.shape != cycle_values.shape or not len(values) or not np.isfinite(values).all():
        raise BatchIntegrityError("R11 verifier rank inputs must be nonempty aligned finite vectors")
    mapped = np.empty(len(values), dtype=np.float64)
    for cycle in np.unique(cycle_values):
        mask = cycle_values == cycle
        percentile = 100.0 * rankdata(values[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * values[mask] + 0.5 * percentile
    return np.clip(mapped, 0.0, 100.0)


def reconstruct_r11(arrays: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    sparse_raw = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    ridge25 = np.asarray(arrays["control__ridge25_prediction"], dtype=np.float64)
    cycles = validated_cycle_codes(arrays["control__cycle_code"], name="R11 verifier control cycle_code")
    saved_baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    baseline = np.clip(0.75 * ridge25 + 0.25 * _independent_rank50(sparse_raw, cycles), 0, 100)
    if not np.array_equal(saved_baseline, baseline):
        raise BatchIntegrityError("R11 saved frozen three-component champion does not reconstruct")
    candidate = np.clip(0.75 * baseline + 0.25 * _independent_rank50(raw, cycles), 0, 100)
    return candidate, baseline


def run_frozen_pilot(*, card: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
                     controls: Mapping[int, Mapping[str, np.ndarray]], freeze_identity: Mapping[str, Any],
                     freeze_path: Path, artifact_dir: Path, budget_ledger_path: Path,
                     event_path: Path, report_path: Path, evaluation_views: Mapping[str, Any],
                     max_successful_fits: int, recipe_path: Path = RECIPE_PATH,
                     preflight_path: Path = PREFLIGHT_PATH) -> dict[str, Any]:
    """Use the shared runner only after exact recipe/preflight/freeze identity checks."""
    _require_frozen_artifacts(
        freeze_path=freeze_path,
        freeze_identity=freeze_identity,
        recipe_path=recipe_path,
        preflight_path=preflight_path,
    )
    _require_committed_sources()
    _require_committed_artifacts((freeze_path, recipe_path, preflight_path))
    _verify_live_freeze_hashes(freeze_identity)
    memory_gate = _check_runtime_memory(preflight_path)
    append_event(event_path, {
        "card_id": CARD_ID,
        "event": "RUNTIME_MEMORY_GATE_PASS",
        **memory_gate,
        "candidate_head_fits_performed": 0,
    })
    return run_pilot(
        card=card,
        artifact_dir=artifact_dir,
        budget_ledger_path=budget_ledger_path,
        max_successful_fits=max_successful_fits,
        frame=frame,
        target=target,
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=r02._control_loader(controls),
        freeze_identity=freeze_identity,
        freeze_path=freeze_path,
        event_path=event_path,
        evaluation_views=evaluation_views,
        threads=THREADS,
        report_path=report_path,
    )


def _require_frozen_artifacts(*, freeze_path: Path, freeze_identity: Mapping[str, Any],
                              recipe_path: Path, preflight_path: Path) -> str:
    signature = json_sha256(freeze_identity)
    if not freeze_path.is_file():
        raise BatchIntegrityError(f"R11 frozen pilot requires an existing freeze file: {freeze_path}")
    try:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"R11 freeze file is unreadable: {exc}") from exc
    normalized_identity = json.loads(json.dumps(dict(freeze_identity), allow_nan=False))
    if frozen.get("signature") != signature or frozen.get("identity") != normalized_identity:
        raise BatchIntegrityError("R11 freeze signature and identity do not exactly match the supplied identity")
    if not recipe_path.is_file() or not preflight_path.is_file():
        raise BatchIntegrityError("R11 frozen pilot requires persisted recipe and preflight files")
    recipe_record = json.loads(recipe_path.read_text(encoding="utf-8"))
    preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if json_sha256(recipe_record) != freeze_identity.get("recipe_record_sha256"):
        raise BatchIntegrityError("R11 persisted recipe differs from the frozen identity")
    preflight_signature = preflight_record.pop("freeze_identity_sha256", None)
    if (preflight_signature != signature
            or json_sha256(preflight_record) != freeze_identity.get("preflight_record_sha256")):
        raise BatchIntegrityError("R11 persisted preflight differs from the frozen identity")
    return signature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true",
                       help="read official Train, controls, and memory; write no freeze and fit no heads")
    modes.add_argument("--freeze-only", action="store_true",
                       help="after code commit, run preflight and write exclusive recipe/preflight/freeze artifacts; fit no heads")
    args = parser.parse_args(argv)
    report = preflight() if args.preflight_only else prepare_freeze()
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "candidate_recipe", "preflight", "prepare_freeze", "reconstruct_r11",
    "run_frozen_pilot", "_require_frozen_artifacts",
]
