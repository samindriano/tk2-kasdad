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
    canonical_json,
    ensure_frozen,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    package_versions,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from scripts.optimization_v3_batch1_verify import verify_report
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
LEDGER_PATH = ARTIFACT_DIR.parent / "batch_ledger.jsonl"
FREEZE_REVISION = 3
EXPERIMENT_REVISION_02_DIR = EXPERIMENT_DIR / "revision_02"
REPORT_REVISION_02_DIR = REPORT_DIR / "revision_02"
ARTIFACT_REVISION_02_DIR = ARTIFACT_DIR / "revision_02"
RECIPE_REVISION_02_PATH = EXPERIMENT_REVISION_02_DIR / "recipe.json"
FREEZE_REVISION_02_PATH = EXPERIMENT_REVISION_02_DIR / "freeze.json"
PREFLIGHT_REVISION_02_PATH = REPORT_REVISION_02_DIR / "preflight.json"
ATTEMPT_REVISION_02_PATH = REPORT_REVISION_02_DIR / "attempts.jsonl"
EXPERIMENT_REVISION_03_DIR = EXPERIMENT_DIR / "revision_03"
REPORT_REVISION_03_DIR = REPORT_DIR / "revision_03"
ARTIFACT_REVISION_03_DIR = ARTIFACT_DIR / "revision_03"
RECIPE_REVISION_03_PATH = EXPERIMENT_REVISION_03_DIR / "recipe.json"
FREEZE_REVISION_03_PATH = EXPERIMENT_REVISION_03_DIR / "freeze.json"
PREFLIGHT_REVISION_03_PATH = REPORT_REVISION_03_DIR / "preflight.json"
ATTEMPT_REVISION_03_PATH = REPORT_REVISION_03_DIR / "attempts.jsonl"
PILOT_REPORT_PATH = ARTIFACT_REVISION_03_DIR / "pilot.json"
RESULTS_PATH = REPORT_REVISION_03_DIR / "results.json"
VERIFICATION_PATH = REPORT_REVISION_03_DIR / "verification.json"
SUBGROUP_PATH = REPORT_REVISION_03_DIR / "subgroups.json"
GATE_PATH = REPORT_REVISION_03_DIR / "gate.json"
SUMMARY_PATH = REPORT_REVISION_03_DIR / "summary.md"
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


def _freeze_payloads(
    context: Mapping[str, Any], *, preflight_base: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    recipe_record: dict[str, Any] = {
        "freeze_revision": FREEZE_REVISION,
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
    preflight_record = dict(preflight_base) if preflight_base is not None else {
        **context["preflight"],
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "package_versions": identity["package_versions"],
        "candidate_head_fits_performed": 0,
        "freeze_written": True,
        "freeze_revision": FREEZE_REVISION,
    }
    for name, current in (
        ("input_sha256", identity["input_sha256"]),
        ("source_sha256", identity["source_sha256"]),
        ("package_versions", identity["package_versions"]),
    ):
        if preflight_record.get(name) != current:
            raise BatchIntegrityError(f"R11 frozen preflight {name} differs from current inputs")
    identity.update({
        "freeze_revision": FREEZE_REVISION,
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
    return recipe_record, identity, preflight_record, signature


def prepare_freeze() -> dict[str, Any]:
    """Read official Train, verify row-exact controls, then write revision-03 artifacts."""
    _require_committed_sources()
    context = _prepare_context()
    recipe_record, identity, preflight_record, signature = _freeze_payloads(context)
    # Catch stale outputs before any write can leave a partial artifact set.
    _assert_json_match_or_absent(RECIPE_REVISION_03_PATH, recipe_record)
    _assert_json_match_or_absent(PREFLIGHT_REVISION_03_PATH, preflight_record)
    _assert_freeze_match_or_absent(FREEZE_REVISION_03_PATH, identity)
    _write_json_or_match(RECIPE_REVISION_03_PATH, recipe_record)
    written_signature = ensure_frozen(FREEZE_REVISION_03_PATH, identity)
    if written_signature != signature:
        raise BatchIntegrityError("R11 written freeze signature differs from prepared identity")
    _write_json_or_match(PREFLIGHT_REVISION_03_PATH, preflight_record)
    return {
        "card_id": CARD_ID,
        "freeze_revision": FREEZE_REVISION,
        "status": "FROZEN_NO_HEAD_FITS",
        "freeze_signature": signature,
        "recipe_path": str(RECIPE_REVISION_03_PATH),
        "preflight_path": str(PREFLIGHT_REVISION_03_PATH),
        "freeze_path": str(FREEZE_REVISION_03_PATH),
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
                     max_successful_fits: int, recipe_path: Path = RECIPE_REVISION_03_PATH,
                     preflight_path: Path = PREFLIGHT_REVISION_03_PATH) -> dict[str, Any]:
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
        "freeze_signature": json_sha256(freeze_identity),
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
    if int(freeze_identity.get("freeze_revision", -1)) != FREEZE_REVISION:
        raise BatchIntegrityError("R11 pilot requires freeze revision 3; earlier revisions are historical")
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


def _append_event_once(path: Path, event: Mapping[str, Any]) -> None:
    if path.is_file():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                current = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid R11 attempt event at line {line_no}") from exc
            if (current.get("card_id") == event.get("card_id")
                    and current.get("event") == event.get("event")
                    and current.get("freeze_signature") == event.get("freeze_signature")):
                if current != dict(event):
                    raise BatchIntegrityError(f"Conflicting existing R11 event: {event['event']}")
                return
    append_event(path, event)


def _assert_no_prior_pilot_attempt(event_path: Path, artifact_dir: Path) -> None:
    fit_events = {
        "FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED",
        "RUNTIME_MEMORY_GATE_PASS",
    }
    if event_path.is_file():
        for line_no, line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid R11 attempt event at line {line_no}") from exc
            if event.get("card_id") == CARD_ID and event.get("event") in fit_events:
                raise BatchIntegrityError(
                    "R11 revision-03 has prior adapter attempt evidence; preserve and review partial outputs before any retry"
                )
    if any((artifact_dir / f"{CARD_ID}_fit{cutoff}.{suffix}").exists()
           for cutoff in CUTOFFS for suffix in ("npz", "json")):
        raise BatchIntegrityError(
            "R11 revision-03 contains prior fit bundles; preserve and review them before any retry"
        )
    if (artifact_dir / "pilot.json").exists():
        raise BatchIntegrityError("R11 revision-03 pilot report already exists; refusing a second shared run")


def _attempt_accounting(*, event_path: Path, artifact_dir: Path,
                        freeze_signature: str) -> dict[str, Any]:
    if not event_path.is_file():
        raise BatchIntegrityError("R11 attempt event log is missing")
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchIntegrityError(f"Invalid R11 attempt event at line {line_no}") from exc
        if event.get("card_id") == CARD_ID:
            events.append(event)
    preflight_events = [event for event in events if event.get("event") == "PREFLIGHT_PASS"]
    memory_events = [event for event in events if event.get("event") == "RUNTIME_MEMORY_GATE_PASS"]
    started = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    resumed = [event for event in events if event.get("event") == "RESUMED_VERIFIED"]
    memory_gate = dict(memory_events[0]) if len(memory_events) == 1 else {}
    for key in ("card_id", "event", "freeze_signature", "candidate_head_fits_performed"):
        memory_gate.pop(key, None)
    memory_values = [
        memory_gate.get("available_ram_gib_at_pilot_start"),
        memory_gate.get("current_allowed_estimated_peak_gib"),
        memory_gate.get("frozen_max_estimated_peak_gib"),
    ]
    memory_gate_valid = (
        len(memory_events) == 1
        and memory_events[0].get("freeze_signature") == freeze_signature
        and all(isinstance(value, (int, float)) and np.isfinite(value) for value in memory_values)
        and int(memory_events[0].get("candidate_head_fits_performed", -1)) == 0
    )
    preflight_valid = (
        len(preflight_events) == 1
        and preflight_events[0].get("freeze_signature") == freeze_signature
        and int(preflight_events[0].get("freeze_revision", -1)) == FREEZE_REVISION
    )

    head_calls = 0
    state_head_calls = 0
    vectorizer_calls = 0
    svd_calls = 0
    head_calls_by_cutoff: dict[str, int] = {}
    vectorizer_calls_by_cutoff: dict[str, int] = {}
    svd_calls_by_cutoff: dict[str, int] = {}
    fit_seconds_by_cutoff: dict[str, float] = {}
    for event in saved:
        cutoff = int(event.get("cutoff", -1))
        fit_seconds = float(event.get("fit_seconds", float("nan")))
        if cutoff not in CUTOFFS or not np.isfinite(fit_seconds) or fit_seconds <= 0.0:
            raise BatchIntegrityError("R11 saved adapter event has an invalid cutoff or runtime")
        fit_seconds_by_cutoff[str(cutoff)] = fit_seconds
    for cutoff in CUTOFFS:
        bundle_path = artifact_dir / f"{CARD_ID}_fit{cutoff}.npz"
        sidecar_path = bundle_path.with_suffix(".json")
        if not bundle_path.is_file() or not sidecar_path.is_file():
            raise BatchIntegrityError(f"R11 cutoff {cutoff} lacks its complete verified bundle")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        identity = sidecar.get("identity", {})
        if (identity.get("card_id") != CARD_ID or int(identity.get("cutoff", -1)) != cutoff
                or identity.get("freeze_signature") != freeze_signature):
            raise BatchIntegrityError(f"R11 cutoff {cutoff} sidecar identity is invalid")
        _, verified_sidecar = read_bundle(bundle_path, identity)
        metadata = verified_sidecar.get("metadata", {})
        fit_metadata = metadata.get("fit_metadata", {})
        state = metadata.get("state", {})
        current_head_calls = int(fit_metadata.get("predictor_fit_calls", -1))
        current_state_calls = int(state.get("head_fit_calls", -1))
        if current_head_calls < 0 or current_state_calls < 0:
            raise BatchIntegrityError(f"R11 cutoff {cutoff} is missing head-fit accounting")
        head_calls += current_head_calls
        state_head_calls += current_state_calls
        vectorizer_calls += int(fit_metadata.get("vectorizer_fit_calls", 0))
        svd_calls += int(fit_metadata.get("svd_fit_calls", 0))
        head_calls_by_cutoff[str(cutoff)] = current_head_calls
        vectorizer_calls_by_cutoff[str(cutoff)] = int(fit_metadata.get("vectorizer_fit_calls", 0))
        svd_calls_by_cutoff[str(cutoff)] = int(fit_metadata.get("svd_fit_calls", 0))

    saved_predictor_fits = sum(int(event.get("fit_predictor_count", 0)) for event in saved)
    expected_cutoffs = {int(cutoff) for cutoff in CUTOFFS}
    started_cutoffs = [int(event.get("cutoff", -1)) for event in started]
    saved_cutoffs = [int(event.get("cutoff", -1)) for event in saved]
    saved_counts_by_cutoff = {
        int(event.get("cutoff", -1)): int(event.get("fit_predictor_count", 0))
        for event in saved
    }
    complete = (
        preflight_valid
        and memory_gate_valid
        and len(started) == len(CUTOFFS)
        and len(saved) == len(CUTOFFS)
        and set(started_cutoffs) == expected_cutoffs
        and len(set(started_cutoffs)) == len(CUTOFFS)
        and all(event.get("freeze_signature") == freeze_signature for event in started)
        and set(saved_cutoffs) == expected_cutoffs
        and len(set(saved_cutoffs)) == len(CUTOFFS)
        and not failed and not resumed
        and saved_predictor_fits == 18
        and saved_counts_by_cutoff == {int(cutoff): 9 for cutoff in CUTOFFS}
        and head_calls == 18 and state_head_calls == 18
        and vectorizer_calls == len(CUTOFFS)
        and svd_calls == len(CUTOFFS)
        and set(head_calls_by_cutoff) == {str(cutoff) for cutoff in CUTOFFS}
        and all(count == 9 for count in head_calls_by_cutoff.values())
        and all(count == 1 for count in vectorizer_calls_by_cutoff.values())
        and all(count == 1 for count in svd_calls_by_cutoff.values())
    )
    return {
        "temporal_adapter_attempts_started": len(started),
        "temporal_adapter_attempts_saved": len(saved),
        "temporal_adapter_attempts_failed": len(failed),
        "resumed_adapter_bundles": len(resumed),
        "successful_predictor_fits_from_receipts": saved_predictor_fits,
        "candidate_head_fit_calls_from_metadata": head_calls,
        "candidate_head_fit_calls_from_state": state_head_calls,
        "head_fit_calls_by_cutoff": head_calls_by_cutoff,
        "vectorizer_fit_calls": vectorizer_calls,
        "vectorizer_fit_calls_by_cutoff": vectorizer_calls_by_cutoff,
        "svd_fit_calls": svd_calls,
        "svd_fit_calls_by_cutoff": svd_calls_by_cutoff,
        "fit_seconds_by_cutoff": fit_seconds_by_cutoff,
        "total_fit_seconds": float(sum(fit_seconds_by_cutoff.values())),
        "preflight_events": len(preflight_events),
        "runtime_memory_gate_events": len(memory_events),
        "runtime_memory_gate": memory_gate,
        "attempt_accounting_complete": bool(complete),
    }


def _subgroup_diagnostics(report: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
                          views: Mapping[str, Any], artifact_dir: Path,
                          minimum_support: int, harm_limit: float) -> dict[str, Any]:
    cycles = _validated_train_cycles(frame["cycle_code"])
    y_all = np.asarray(target, dtype=np.float64)
    if (y_all.shape != (len(frame),) or not np.isfinite(y_all).all()
            or np.any((y_all < 0.0) | (y_all > 100.0)) or int(minimum_support) < 1
            or not np.isfinite(float(harm_limit))):
        raise BatchIntegrityError("R11 subgroup inputs or support gate are invalid")
    blocks: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in CUTOFFS:
        bundle_path = artifact_dir / f"{CARD_ID}_fit{cutoff}.npz"
        sidecar_path = bundle_path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise BatchIntegrityError(f"R11 subgroup diagnostics missing sidecar at cutoff {cutoff}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        identity = sidecar.get("identity", {})
        if (identity.get("card_id") != CARD_ID
                or int(identity.get("cutoff", -1)) != cutoff
                or identity.get("freeze_signature") != report.get("freeze_signature")):
            raise BatchIntegrityError(f"R11 subgroup bundle identity differs at cutoff {cutoff}")
        arrays, _ = read_bundle(bundle_path, identity)
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        prediction = np.asarray(arrays["prediction"], dtype=np.float64)
        baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        if (rows.ndim != 1 or prediction.shape != rows.shape or baseline.shape != rows.shape
                or np.any((rows < 0) | (rows >= len(frame)))
                or not np.isfinite(prediction).all() or not np.isfinite(baseline).all()):
            raise BatchIntegrityError(f"R11 subgroup arrays are invalid at cutoff {cutoff}")
        expected_cycles = sorted({
            cycle for segments in views.values()
            for origin, low, high in segments if int(origin) == cutoff
            for cycle in range(int(low), int(high) + 1)
        })
        expected_rows = np.flatnonzero(np.isin(cycles, np.asarray(expected_cycles, dtype=np.int64)))
        if not np.array_equal(rows, expected_rows):
            raise BatchIntegrityError(f"R11 subgroup rows differ from the declared views at cutoff {cutoff}")
        blocks[cutoff] = {"rows": rows, "candidate": prediction, "baseline": baseline}

    output: dict[str, Any] = {}
    guard_failures: list[dict[str, Any]] = []
    for view_name in ("primary", "H12"):
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in views[view_name]:
            block = blocks[int(cutoff)]
            mask = (cycles[block["rows"]] >= int(low)) & (cycles[block["rows"]] <= int(high))
            row_parts.append(block["rows"][mask])
            pred_parts.append(block["candidate"][mask])
            base_parts.append(block["baseline"][mask])
        rows = np.concatenate(row_parts)
        prediction = np.concatenate(pred_parts)
        baseline = np.concatenate(base_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R11 subgroup view {view_name} includes duplicate Train rows")
        y = y_all[rows]
        local = frame.iloc[rows]
        stream = local["signal_stream"]
        missing = stream.isna().to_numpy(dtype=bool) | stream.fillna("").astype(str).str.strip().eq("").to_numpy(dtype=bool)
        friction = pd.to_numeric(local["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        groups = {
            "entry_friction_0": friction == 0.0,
            "entry_friction_1": friction == 1.0,
            "stream_present": ~missing,
            "stream_missing": missing,
            "target_le_5": y <= 5.0,
            "target_ge_95": y >= 95.0,
        }
        view_result: dict[str, Any] = {}
        for group_name, mask in groups.items():
            n = int(mask.sum())
            supported = n >= int(minimum_support)
            candidate_mae = float(np.mean(np.abs(prediction[mask] - y[mask]))) if n else None
            control_mae = float(np.mean(np.abs(baseline[mask] - y[mask]))) if n else None
            harm = None if n == 0 else float(candidate_mae - control_mae)
            guard_applies = group_name in {"stream_missing", "target_le_5", "target_ge_95"} and supported
            guard_pass = None if not guard_applies else bool(harm <= harm_limit)
            record = {
                "n": n,
                "supported": bool(supported),
                "evidence_status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                "candidate_mae": candidate_mae,
                "control_mae": control_mae,
                "mae_harm_candidate_minus_control": harm,
                "guard_applies": bool(guard_applies),
                "guard_pass": guard_pass,
            }
            view_result[group_name] = record
            if guard_applies and not guard_pass:
                guard_failures.append({"view": view_name, "group": group_name, **record})
        output[view_name] = view_result
    return {
        "minimum_support": int(minimum_support),
        "supported_tail_or_missing_MAE_harm_max": float(harm_limit),
        "views": output,
        "supported_guard_failures": guard_failures,
    }


def _build_gate(queue: Mapping[str, Any], pilot: Mapping[str, Any], verification: Mapping[str, Any],
                subgroups: Mapping[str, Any], accounting: Mapping[str, Any],
                card: Mapping[str, Any]) -> dict[str, Any]:
    if verification.get("status") != "PASS":
        raise BatchIntegrityError("R11 independent verification did not pass; gate is not computed")
    if not accounting.get("attempt_accounting_complete"):
        raise BatchIntegrityError("R11 attempt accounting is incomplete; gate is not computed")
    gate_config = queue["evaluation"]["regression_gate"]
    gains = pilot["views"]
    checks = {
        "pilot_primary_MAE_gain_at_least_threshold": gains["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_MAE_gain_min"]),
        "pilot_H12_MAE_harm_at_most_threshold": gains["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_at_most_threshold": gains["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_at_most_threshold": gains["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_group_harms": len(subgroups["supported_guard_failures"]) == 0,
        "independent_verification_pass": True,
        "fit_attempt_accounting_pass": True,
    }
    return {
        "card_id": CARD_ID,
        "freeze_revision": FREEZE_REVISION,
        "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": gate_config,
        "subgroup_diagnostics": subgroups,
        "decision_basis": "Revision-03 frozen W1+W4/H12 Train pilot, independent reconstruction, and declared regression gates only; no Test/Kaggle evidence.",
    }


def _render_summary(results: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    pilot = results["pilot"]
    lines = [
        "# R11 — Cumulative ordinal median pilot",
        "",
        f"**Verdict: {gate['status']}** — revision-03 W1+W4/H12 Train pilot; no Test/Kaggle access.",
        "",
        "Recipe: nine binary LightGBM heads estimate strict survival thresholds y>10,...,90; row-wise PAVA enforces nonincreasing probabilities; the median is linearly interpolated at the 0.5 crossing and rank50-mapped within cycle before the frozen 0.75/0.25 blend.",
        "",
        "| View | n | Control MAE | R11 MAE | Gain (control − R11) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("W1", "W4", "primary", "H12"):
        row = pilot["views"][name]
        lines.append(
            f"| {name} | {row['rows']:,} | {row['baseline']['mae']:.6f} | "
            f"{row['candidate']['mae']:.6f} | {row['gain_positive_is_better']:+.6f} |"
        )
    accounting = results["fit_accounting"]
    lines.extend([
        "",
        f"Fit accounting: {accounting['temporal_adapter_attempts_started']} temporal adapter attempts; {accounting['successful_predictor_fits_from_receipts']} successful LightGBM head fits; {accounting['total_fit_seconds']:.2f} fit seconds.",
        "SVD components receive no semantic interpretation. Validation windows are development data and are not an untouched holdout.",
        "",
        "See `results.json`, `verification.json`, `subgroups.json`, `gate.json`, `attempts.jsonl`, and `experiments/optimization_v3/batch1/R11/revision_03/freeze.json` for provenance.",
        "",
    ])
    return "\n".join(lines)


def _write_text_or_match(path: Path, value: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != value:
            raise BatchIntegrityError(f"Existing R11 summary differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def _append_reconciliation(results: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        raise BatchIntegrityError("Shared fit ledger is missing; cannot reconcile R11 budget")
    ledger: list[dict[str, Any]] = []
    for line_no, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ledger.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BatchIntegrityError(f"Invalid shared fit ledger at line {line_no}") from exc
    reservations = [item for item in ledger if item.get("event") == "BUDGET_RESERVED"
                    and item.get("budget_key") == "R11:PILOT"]
    accounting = results["fit_accounting"]
    if (len(reservations) != 1 or reservations[0].get("card_id") != CARD_ID
            or int(reservations[0].get("predictor_fits", 0)) != 18
            or int(accounting["successful_predictor_fits_from_receipts"]) != 18
            or int(accounting["temporal_adapter_attempts_started"]) != 2
            or int(accounting["temporal_adapter_attempts_saved"]) != 2
            or int(accounting["temporal_adapter_attempts_failed"]) != 0
            or not accounting["attempt_accounting_complete"]):
        raise BatchIntegrityError("R11 fit budget reconciliation requires exactly 18 reserved and successful head fits")
    totals = [int(item["reserved_total_after"]) for item in ledger
              if item.get("event") == "BUDGET_RESERVED" and "reserved_total_after" in item]
    event = {
        "event": "FIT_BUDGET_RECONCILED",
        "card_id": CARD_ID,
        "budget_key": "R11:PILOT",
        "freeze_revision": FREEZE_REVISION,
        "reserved_predictor_fits": 18,
        "successful_predictor_fits": 18,
        "temporal_adapter_attempts": 2,
        "fit_saved_events": 2,
        "fit_failed_events": 0,
        "attempt_accounting_complete": True,
        "decision": gate["status"],
        "freeze_signature": results["freeze_signature"],
        "reserved_global_total_after": max(totals) if totals else None,
        "results_sha256": file_sha256(RESULTS_PATH),
        "gate_sha256": file_sha256(GATE_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
    }
    _append_event_once(LEDGER_PATH, event)
    return event


def _require_source_commit_ancestor(source_commit: str) -> None:
    if not source_commit or not isinstance(source_commit, str):
        raise BatchIntegrityError(f"R11 revision-{FREEZE_REVISION:02d} recipe lacks its source commit")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise BatchIntegrityError("R11 frozen source commit is not an ancestor of the current checkout")


def _assert_live_preflight_compatible(context: Mapping[str, Any],
                                      frozen_preflight: Mapping[str, Any],
                                      identity: Mapping[str, Any]) -> None:
    current = context["preflight"]
    fields = (
        "card_id", "status", "source_commit", "queue_sha256", "card_sha256",
        "official_train_sha256", "train_rows", "fit_rows", "score_rows", "views",
        "threshold_support_by_cutoff", "target_boundary_audit", "control_metrics",
        "control_cache_sha256", "official_test_read", "candidate_head_fits_performed",
    )
    for field in fields:
        if canonical_json(current.get(field)) != canonical_json(frozen_preflight.get(field)):
            raise BatchIntegrityError(f"R11 live preflight field {field} differs from revision-{FREEZE_REVISION:02d} snapshot")
    if (frozen_preflight.get("freeze_revision") != FREEZE_REVISION
            or frozen_preflight.get("input_sha256") != identity.get("input_sha256")
            or frozen_preflight.get("source_sha256") != identity.get("source_sha256")
            or frozen_preflight.get("package_versions") != identity.get("package_versions")):
        raise BatchIntegrityError(f"R11 revision-{FREEZE_REVISION:02d} preflight hashes or revision differ from live identity")
    frozen_resource = frozen_preflight.get("resource")
    if not isinstance(frozen_resource, Mapping) or _stable_resource_identity(frozen_resource) != _stable_resource_identity(context["resource"]):
        raise BatchIntegrityError(f"R11 stable resource estimates differ from revision-{FREEZE_REVISION:02d} preflight")
    frozen_budget = frozen_preflight.get("fit_budget", {})
    live_budget = current.get("fit_budget", {})
    if (int(frozen_budget.get("R11_predictor_fits", -1)) != 18
            or int(live_budget.get("R11_predictor_fits", -1)) != 18):
        raise BatchIntegrityError(f"R11 revision-{FREEZE_REVISION:02d} pilot budget must remain exactly 18 head fits")


def run_card(*, mode: str) -> dict[str, Any]:
    """Run a single explicit R11 mode; the pilot mode is deliberately one-shot."""
    if mode == "preflight":
        return preflight()
    if mode == "freeze":
        return prepare_freeze()
    if mode != "pilot":
        raise BatchIntegrityError("R11 mode must be preflight, freeze, or pilot")

    _require_committed_sources()
    _require_committed_artifacts((FREEZE_REVISION_03_PATH, RECIPE_REVISION_03_PATH,
                                  PREFLIGHT_REVISION_03_PATH))
    frozen_recipe = json.loads(RECIPE_REVISION_03_PATH.read_text(encoding="utf-8"))
    frozen_preflight = json.loads(PREFLIGHT_REVISION_03_PATH.read_text(encoding="utf-8"))
    source_commit = frozen_recipe.get("source_commit")
    _require_source_commit_ancestor(source_commit)
    context = _prepare_context()
    # The freeze commit advances HEAD after it records the source-code commit.
    # Reuse that committed source commit while recomputing all live file hashes.
    context["source_commit"] = source_commit
    context["preflight"]["source_commit"] = source_commit
    preflight_base = dict(frozen_preflight)
    preflight_base.pop("freeze_identity_sha256", None)
    recipe_record, identity, expected_preflight, signature = _freeze_payloads(
        context, preflight_base=preflight_base
    )
    _assert_live_preflight_compatible(context, frozen_preflight, identity)
    _require_frozen_artifacts(
        freeze_path=FREEZE_REVISION_03_PATH,
        freeze_identity=identity,
        recipe_path=RECIPE_REVISION_03_PATH,
        preflight_path=PREFLIGHT_REVISION_03_PATH,
    )
    _verify_live_freeze_hashes(identity)
    if json.loads(RECIPE_REVISION_03_PATH.read_text(encoding="utf-8")) != recipe_record:
        raise BatchIntegrityError("R11 revision-03 recipe differs from the live frozen recipe")
    if json.loads(PREFLIGHT_REVISION_03_PATH.read_text(encoding="utf-8")) != expected_preflight:
        raise BatchIntegrityError("R11 revision-03 preflight differs from the live preflight")
    _assert_no_prior_pilot_attempt(ATTEMPT_REVISION_03_PATH, ARTIFACT_REVISION_03_DIR)
    _append_event_once(ATTEMPT_REVISION_03_PATH, {
        "card_id": CARD_ID,
        "event": "PREFLIGHT_PASS",
        "freeze_signature": signature,
        "freeze_revision": FREEZE_REVISION,
        "resource": _stable_resource_identity(context["resource"]),
        "control_metrics": context["control_metrics"],
        "fit_predictor_count_reserved": 18,
        "candidate_head_fits_performed": 0,
    })
    pilot = run_frozen_pilot(
        card=context["card"],
        frame=context["frame"],
        target=context["target"],
        controls=context["controls"],
        freeze_identity=identity,
        freeze_path=FREEZE_REVISION_03_PATH,
        artifact_dir=ARTIFACT_REVISION_03_DIR,
        budget_ledger_path=LEDGER_PATH,
        event_path=ATTEMPT_REVISION_03_PATH,
        report_path=PILOT_REPORT_PATH,
        evaluation_views=context["views"],
        max_successful_fits=int(context["queue"]["execution_preset"]["max_successful_new_predictor_fits"]),
        recipe_path=RECIPE_REVISION_03_PATH,
        preflight_path=PREFLIGHT_REVISION_03_PATH,
    )
    verification = verify_report(
        report=pilot,
        frame=context["frame"],
        target=context["target"],
        artifact_dir=ARTIFACT_REVISION_03_DIR,
        freeze_signature=signature,
        evaluation_views=context["views"],
        prediction_reconstructors={CARD_ID: reconstruct_r11},
    )
    if verification.get("status") != "PASS":
        raise BatchIntegrityError("R11 independent verification failed; preserving pilot output without a gate")
    subgroup = _subgroup_diagnostics(
        pilot, context["frame"], context["target"], context["views"],
        ARTIFACT_REVISION_03_DIR,
        int(context["queue"]["evaluation"]["regression_gate"]["supported_group_n_min"]),
        float(context["queue"]["evaluation"]["regression_gate"]["supported_tail_or_missing_MAE_harm_max"]),
    )
    accounting = _attempt_accounting(event_path=ATTEMPT_REVISION_03_PATH,
                                     artifact_dir=ARTIFACT_REVISION_03_DIR,
                                     freeze_signature=signature)
    if not accounting["attempt_accounting_complete"]:
        raise BatchIntegrityError(f"R11 18-head fit accounting is incomplete; preserving outputs: {accounting}")
    gate = _build_gate(context["queue"], pilot, verification, subgroup, accounting, context["card"])
    results = {
        "card_id": CARD_ID,
        "freeze_revision": FREEZE_REVISION,
        "status": f"PILOT_COMPLETE_{gate['status']}",
        "source_commit": context["source_commit"],
        "freeze_signature": signature,
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "preflight": context["preflight"],
        "pilot": pilot,
        "fit_accounting": accounting,
        "independent_verification_status": verification["status"],
        "subgroups": subgroup,
        "gate": gate,
        "official_test_read": False,
        "kaggle_accessed": False,
    }
    result_files = ((VERIFICATION_PATH, verification), (SUBGROUP_PATH, subgroup),
                    (GATE_PATH, gate), (RESULTS_PATH, results))
    summary = _render_summary(results, gate)
    for path, payload in result_files:
        _assert_json_match_or_absent(path, payload)
    if SUMMARY_PATH.is_file() and SUMMARY_PATH.read_text(encoding="utf-8") != summary:
        raise BatchIntegrityError(f"Existing R11 summary differs; refusing overwrite: {SUMMARY_PATH}")
    for path, payload in result_files:
        _write_json_or_match(path, payload)
    _write_text_or_match(SUMMARY_PATH, summary)
    reconciliation = _append_reconciliation(results, gate)
    return {"results": results, "verification": verification, "gate": gate,
            "budget_reconciliation": reconciliation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true",
                       help="read official Train, controls, and memory; write no freeze and fit no heads")
    modes.add_argument("--freeze-only", action="store_true",
                       help="after code commit, run preflight and write exclusive recipe/preflight/freeze artifacts; fit no heads")
    modes.add_argument("--pilot", action="store_true",
                       help="run the one-shot revision-03 frozen pilot, verification, gate, and append-only reconciliation")
    args = parser.parse_args(argv)
    mode = "preflight" if args.preflight_only else "freeze" if args.freeze_only else "pilot"
    report = run_card(mode=mode)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "candidate_recipe", "preflight", "prepare_freeze", "reconstruct_r11",
    "run_card", "run_frozen_pilot", "_attempt_accounting", "_build_gate",
    "_require_frozen_artifacts", "_subgroup_diagnostics",
]
