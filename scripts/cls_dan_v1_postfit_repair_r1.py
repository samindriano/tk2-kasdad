"""Authorized post-fit adapter for the CLS_DAN_V1 validation bundle.

The original fit runner remains frozen. This adapter repairs one invalid
cross-origin equality guard and permits an ignored, hash-bound local prelabel
NPZ while keeping every other committed-file requirement intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))

from scripts import cls_dan_v1 as runner


ROOT_REPAIR_PATH = runner.REPORT_ROOT / "ROOT_PRELABEL_REPAIR_R1.json"
POSTFIT_RECEIPT_PATH = runner.REPORT_ROOT / "POSTFIT_PRELABEL_R1.json"
ALIGNMENT_ERROR = "E67/E75 champion probabilities differ on overlapping query IDs"
ADAPTER_FILES = (
    "scripts/cls_dan_v1_postfit_repair_r1.py",
    "tests/test_cls_dan_v1_postfit_repair_r1.py",
    "reports/simplification_v1/CLS_DAN_V1/POSTFIT_PRELABEL_REPAIR_R1.md",
    "experiments/CLS_DAN_V1/.gitignore",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    return runner.sha256_array(value)


def _committed_bytes(path: Path) -> bytes:
    relative = Path(path).resolve().relative_to(runner.AUDIT.resolve()).as_posix()
    return subprocess.check_output(["git", "-C", str(runner.AUDIT), "show", f"HEAD:{relative}"])


def _assert_committed_adapter() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in ADAPTER_FILES:
        path = runner.AUDIT / relative
        committed = _committed_bytes(path)
        on_disk = path.read_bytes()
        _require(committed.replace(b"\r\n", b"\n") == on_disk.replace(b"\r\n", b"\n"),
                 f"Adapter source/evidence is not committed exactly: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _assert_binary_ignored_and_untracked(path: Path) -> dict[str, Any]:
    relative = Path(path).resolve().relative_to(runner.AUDIT.resolve()).as_posix()
    ignored = subprocess.run(
        ["git", "-C", str(runner.AUDIT), "check-ignore", "--quiet", "--", relative],
        check=False,
    ).returncode == 0
    tracked = subprocess.run(
        ["git", "-C", str(runner.AUDIT), "ls-files", "--error-unmatch", "--", relative],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    _require(ignored and not tracked,
             f"Prelabel binary must remain ignored and untracked: {relative}")
    return {"path": relative, "git_ignored": ignored, "git_tracked": tracked}


def validate_origin_alignment(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Require shared source-row identity while allowing origin-specific scores."""
    ids_first = np.asarray(first["score_record_ids"]).astype(str).reshape(-1)
    ids_second = np.asarray(second["score_record_ids"]).astype(str).reshape(-1)
    rows_first = np.asarray(first["score_rows"], dtype=np.int64).reshape(-1)
    rows_second = np.asarray(second["score_rows"], dtype=np.int64).reshape(-1)
    cycles_first = np.asarray(first["score_cycles"], dtype=np.int64).reshape(-1)
    cycles_second = np.asarray(second["score_cycles"], dtype=np.int64).reshape(-1)
    present_first = np.asarray(first["present"], dtype=bool).reshape(-1)
    present_second = np.asarray(second["present"], dtype=bool).reshape(-1)
    champion_first = np.asarray(first["champion"], dtype=np.float64).reshape(-1)
    champion_second = np.asarray(second["champion"], dtype=np.float64).reshape(-1)
    first_sizes = {len(ids_first), len(rows_first), len(cycles_first), len(present_first),
                   len(champion_first)}
    second_sizes = {len(ids_second), len(rows_second), len(cycles_second), len(present_second),
                    len(champion_second)}
    _require(len(first_sizes) == 1 and len(second_sizes) == 1,
             "Each origin's query identity/availability/probability arrays must align")
    _require(len(np.unique(ids_first)) == len(ids_first)
             and len(np.unique(ids_second)) == len(ids_second),
             "Origin query IDs must be unique within each source view")
    _require(len(np.unique(rows_first)) == len(rows_first)
             and len(np.unique(rows_second)) == len(rows_second),
             "Origin source-row positions must be unique within each source view")
    _require(np.isfinite(champion_first).all() and np.isfinite(champion_second).all()
             and np.all((champion_first >= 0.0) & (champion_first <= 1.0))
             and np.all((champion_second >= 0.0) & (champion_second <= 1.0)),
             "Origin champion probabilities must be finite values in [0, 1]")
    first_by_id = {record_id: index for index, record_id in enumerate(ids_first)}
    mapped = np.asarray([first_by_id.get(record_id, -1) for record_id in ids_second], dtype=np.int64)
    _require(np.all(mapped >= 0), "A later-origin query ID is absent from the earlier query view")
    _require(np.array_equal(rows_first[mapped], rows_second),
             "Shared query IDs map to different official Train source-row positions")
    _require(np.array_equal(cycles_first[mapped], cycles_second),
             "Shared query IDs map to different cycle values")
    _require(np.array_equal(present_first[mapped], present_second),
             "Shared query IDs map to different availability values")
    _require(np.all((cycles_second >= 76) & (cycles_second <= 79)),
             "Later-origin query view escaped cycles 76-79")
    equal = champion_first[mapped].view(np.uint64) == champion_second.view(np.uint64)
    return {
        "shared_query_rows": int(len(mapped)),
        "record_ids_exact": True,
        "source_rows_exact": True,
        "cycles_exact": True,
        "availability_exact": True,
        "champion_probabilities_required_equal": False,
        "champion_probabilities_exact_equal_rows": int(equal.sum()),
        "champion_probabilities_different_rows": int((~equal).sum()),
        "earlier_rows_sha256": _sha256_array(rows_first[mapped]),
        "later_rows_sha256": _sha256_array(rows_second),
    }


def validate_prelabel_binding(
    prelabel_path: Path,
    ready: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_ready_sha256: str | None = None,
    enforce_git_policy: bool = False,
) -> dict[str, Any]:
    """Validate the committed JSON-to-ignored-NPZ binding without reading labels."""
    path = Path(prelabel_path).resolve()
    _require(path.is_file(), "Hash-bound prelabel NPZ is missing")
    relative = path.relative_to(runner.AUDIT.resolve()).as_posix()
    actual_sha = _sha256_file(path)
    _require(ready.get("status") == "PASS_PRELABEL_READY_NO_QUERY_LABELS"
             and ready.get("prelabel_path") == relative
             and ready.get("prelabel_sha256") == actual_sha
             and ready.get("evaluation_query_targets_read") is False
             and ready.get("metrics_computed") is False
             and ready.get("test_opened") is False
             and ready.get("sample_submission_opened") is False
             and ready.get("kaggle_contacted") is False,
             "Committed target-free prediction manifest does not bind the local prelabel NPZ")
    _require(receipt.get("status") in {
                 "PASS_POSTFIT_PRELABEL_REPAIR_R1",
                 "PASS_SYNTHETIC_POSTFIT_PRELABEL_REPAIR_R1",
             }
             and receipt.get("prelabel_path") == relative
             and receipt.get("prelabel_sha256") == actual_sha
             and receipt.get("prediction_ready_sha256") == _sha256_file(
                 runner.PREDICTIONS_READY_PATH
             ),
             "Committed post-fit adapter receipt does not bind the local prelabel NPZ/readiness")
    if expected_ready_sha256 is not None:
        _require(receipt.get("prediction_ready_sha256") == expected_ready_sha256,
                 "Post-fit receipt prediction manifest hash changed")
    if enforce_git_policy:
        binary_policy = _assert_binary_ignored_and_untracked(path)
    else:
        binary_policy = {"path": relative, "git_policy_checked": False}
    return {"prelabel_sha256": actual_sha, "prediction_ready_sha256": _sha256_file(
        runner.PREDICTIONS_READY_PATH), "binary_policy": binary_policy}


@contextmanager
def _ignored_prelabel_commit_shim(*, enforce_git_policy: bool) -> Iterator[None]:
    """Substitute only PRELABEL's binary-commit guard with its committed JSON hash binding."""
    original = runner._require_committed_file

    def guarded(path: Path) -> None:
        if Path(path).resolve() == Path(runner.PRELABEL_PATH).resolve():
            ready = runner._require_committed_json(runner.PREDICTIONS_READY_PATH)
            receipt = runner._require_committed_json(POSTFIT_RECEIPT_PATH)
            validate_prelabel_binding(
                runner.PRELABEL_PATH, ready, receipt,
                expected_ready_sha256=receipt.get("prediction_ready_sha256"),
                enforce_git_policy=enforce_git_policy,
            )
            return
        original(path)

    runner._require_committed_file = guarded
    try:
        yield
    finally:
        runner._require_committed_file = original


def _validate_root_repair(official_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, _, reservation = runner.verify_root_prefit_and_reservation(official_root)
    authorization = runner._require_committed_json(ROOT_REPAIR_PATH)
    _require(authorization.get("status") == "APPROVED_TARGET_FREE_ORIGIN_SPECIFIC_PRELABEL_REPAIR"
             and authorization.get("candidate_id") == "CLS_DAN_V1"
             and authorization.get("freeze_signature") == freeze["freeze_signature"]
             and authorization.get("original_source_commit") == freeze["source_commit"]
             and authorization.get("original_runner_file_sha256") == _sha256_file(
                 Path(runner.__file__)
             )
             and authorization.get("additional_fit_starts_allowed") == 0
             and authorization.get("evaluation_labels_opened_before_repair") is False
             and authorization.get("prelabel_output_exists_before_repair") is False,
             "Committed root repair authorization does not match the frozen fit source/state")
    fit_ids = set(runner.FIT_IDS)
    _require(set(authorization.get("saved_fits", {})) == fit_ids,
             "Root repair authorization does not bind exactly four saved fits")
    ledger_sha = _sha256_file(runner.FIT_LEDGER_PATH)
    _require(ledger_sha == authorization.get("ledger_sha256_before_repair"),
             "Fit ledger changed after root repair approval")
    events = runner._ledger_events()
    starts = [event["fit_id"] for event in events if event["event"] == "FIT_STARTED"]
    saves = [event["fit_id"] for event in events if event["event"] == "FIT_SAVED"]
    failures = [event for event in events if event["event"] == "FIT_FAILED_CONSUMED"]
    _require(len(events) == 8 and set(starts) == fit_ids and len(starts) == 4
             and set(saves) == fit_ids and len(saves) == 4 and not failures,
             "Four-fit ledger is incomplete, duplicated, or contains a consumed failure")
    for fit_id in runner.FIT_IDS:
        saved = authorization["saved_fits"][fit_id]
        receipt_path = runner.RUN_ROOT / "units" / "receipts" / f"{fit_id}.json"
        model_path = runner.RUN_ROOT / "units" / f"{fit_id}.npz"
        prediction_path = runner.RUN_ROOT / "predictions" / f"{fit_id}.npz"
        receipt = runner.read_json(receipt_path)
        _require(_sha256_file(receipt_path) == saved["receipt_sha256"]
                 and _sha256_file(model_path) == saved["model_artifact_sha256"]
                 and _sha256_file(prediction_path) == saved["prediction_artifact_sha256"]
                 and receipt.get("status") == "FIT_SAVED"
                 and receipt.get("no_retry") is True,
                 f"Saved fit no longer matches root-bound repair evidence: {fit_id}")
    return freeze, authorization


def _origin_cache(ready: Mapping[str, Any], origin: int) -> dict[str, np.ndarray]:
    record = ready["origins"][str(origin)]
    path = runner.AUDIT / record["path"]
    _require(_sha256_file(path) == record["sha256"],
             f"E{origin} target-free feature cache hash differs from its committed manifest")
    with np.load(path, allow_pickle=False) as archive:
        return {
            "score_rows": np.asarray(archive["score_rows"], dtype=np.int64).copy(),
            "score_record_ids": np.asarray(archive["score_record_ids"]).astype(str),
            "score_cycles": np.asarray(archive["score_cycles"], dtype=np.int64).copy(),
            "present": np.asarray(archive["score_present_mask"], dtype=bool).copy(),
            "champion": np.asarray(archive["score_champion_probability"], dtype=np.float64).copy(),
            "p_pos_original": np.asarray(archive["score_pos_original_probability"], dtype=np.float64).copy(),
            "p_a02": np.asarray(archive["score_a02_probability"], dtype=np.float64).copy(),
            "entry_friction": np.asarray(archive["score_entry_friction"]).astype(str),
            "text_load_group": np.asarray(archive["score_text_load_group"]).astype(str),
            "coverage": np.asarray(archive["score_coverage"], dtype=np.float64).copy(),
        }


def _verify_prelabel_against_caches(
    prelabel_path: Path,
    ready: Mapping[str, Any],
    caches: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    per_origin: dict[str, Any] = {}
    values: dict[int, dict[str, np.ndarray]] = {}
    with np.load(prelabel_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["prelabel_metadata_json"].item()))
        _require(metadata.get("candidate_id") == "CLS_DAN_V1"
                 and metadata.get("evaluation_query_targets_read") is False
                 and metadata.get("test_opened") is False
                 and metadata.get("sample_submission_opened") is False
                 and metadata.get("kaggle_contacted") is False,
                 "Prelabel payload metadata is not target-free")
        for origin in runner.ORIGINS:
            prefix = f"E{origin}_"
            view = {
                "score_rows": np.asarray(archive[prefix + "score_rows"], dtype=np.int64),
                "score_record_ids": np.asarray(archive[prefix + "score_record_ids"]).astype(str),
                "score_cycles": np.asarray(archive[prefix + "score_cycles"], dtype=np.int64),
                "present": np.asarray(archive[prefix + "present"], dtype=bool),
                "champion": np.asarray(archive[prefix + "champion"], dtype=np.float64),
                "dan": np.asarray(archive[prefix + "dan"], dtype=np.float64),
                "linear": np.asarray(archive[prefix + "linear"], dtype=np.float64),
            }
            values[origin] = view
            for key in ("score_rows", "score_record_ids", "score_cycles", "present", "champion",
                        "p_pos_original", "p_a02", "entry_friction", "text_load_group", "coverage"):
                cache_key = "champion" if key == "champion" else key
                payload_key = "champion" if key == "champion" else key
                if key in {"p_pos_original", "p_a02", "entry_friction", "text_load_group", "coverage"}:
                    payload_key = key
                stored = np.asarray(archive[prefix + payload_key])
                expected = np.asarray(caches[origin][cache_key])
                equal = (np.array_equal(stored.view(np.uint64), expected.view(np.uint64))
                         if stored.dtype.kind == "f" else np.array_equal(stored, expected))
                _require(equal, f"E{origin} prelabel {key} differs from its hash-bound feature cache")
            for model in ("dan", "linear"):
                probability = view[model]
                _require(np.isfinite(probability).all()
                         and np.all((probability >= 0.0) & (probability <= 1.0)),
                         f"E{origin} {model} prelabel probabilities are invalid")
                _require(np.array_equal(probability[~view["present"]].view(np.uint64),
                                        view["champion"][~view["present"]].view(np.uint64)),
                         f"E{origin} {model} missing-stream fallback differs from its own champion")
            per_origin[str(origin)] = {
                "rows": int(len(view["score_rows"])),
                "feature_cache_sha256": ready["origins"][str(origin)]["sha256"],
                "row_positions_sha256": _sha256_array(view["score_rows"]),
                "record_ids_sha256": _sha256_array(view["score_record_ids"]),
                "cycles_sha256": _sha256_array(view["score_cycles"]),
                "champion_sha256": _sha256_array(view["champion"]),
                "missing_route_exact": True,
            }
    overlap = validate_origin_alignment(values[67], values[75])
    return {"origins": per_origin, "overlap": overlap,
            "target_free_payload_metadata": True}


def build_prelabel(official_root: Path) -> dict[str, Any]:
    """Build the original prelabel with one authorized origin-specific guard repair."""
    freeze, authorization = _validate_root_repair(official_root)
    _require(not runner.PRELABEL_PATH.exists()
             and not runner.PREDICTIONS_READY_PATH.exists()
             and not (runner.RUN_ROOT / "EVALUATION_ARM.json").exists()
             and not (runner.RUN_ROOT / "EVALUATION_STARTED.json").exists()
             and not (runner.RUN_ROOT / "EVALUATION_CONSUMED.json").exists()
             and not runner.RESULTS_PATH.exists(),
             "Prelabel repair requires the original unscored state")
    _assert_committed_adapter()
    _assert_binary_ignored_and_untracked(runner.PRELABEL_PATH)
    ready, _ = runner.verify_features_ready(official_root, freeze=freeze)
    caches = {origin: _origin_cache(ready, origin) for origin in runner.ORIGINS}
    overlap = validate_origin_alignment(caches[67], caches[75])

    original_require = runner.require
    bypassed: list[str] = []

    def scoped_require(condition: bool, message: str) -> None:
        if message == ALIGNMENT_ERROR:
            bypassed.append(message)
            return
        original_require(condition, message)

    runner.require = scoped_require
    try:
        prediction_ready = runner.build_prelabel_predictions(official_root)
    finally:
        runner.require = original_require
    _require(bypassed == [ALIGNMENT_ERROR],
             "Expected exactly one authorized cross-origin champion equality guard")
    audit = _verify_prelabel_against_caches(runner.PRELABEL_PATH, ready, caches)
    _assert_binary_ignored_and_untracked(runner.PRELABEL_PATH)
    prelabel_sha = _sha256_file(runner.PRELABEL_PATH)
    prediction_ready_sha = _sha256_file(runner.PREDICTIONS_READY_PATH)
    _require(prediction_ready.get("status") == "PASS_PRELABEL_READY_NO_QUERY_LABELS"
             and prediction_ready.get("prelabel_sha256") == prelabel_sha
             and prediction_ready.get("evaluation_query_targets_read") is False
             and prediction_ready.get("metrics_computed") is False
             and prediction_ready.get("test_opened") is False
             and prediction_ready.get("sample_submission_opened") is False
             and prediction_ready.get("kaggle_contacted") is False,
             "Prelabel builder did not preserve target-free readiness flags")
    expected_fit_receipts = {
        fit_id: saved["receipt_sha256"]
        for fit_id, saved in authorization["saved_fits"].items()
    }
    _require(prediction_ready["fit_receipt_hashes"] == expected_fit_receipts,
             "Prelabel manifest fit receipts differ from the four root-bound saved fits")
    receipt = {
        "status": "PASS_POSTFIT_PRELABEL_REPAIR_R1",
        "candidate_id": "CLS_DAN_V1",
        "repair_authorization_sha256": _sha256_file(ROOT_REPAIR_PATH),
        "freeze_signature": freeze["freeze_signature"],
        "original_fit_source_commit": freeze["source_commit"],
        "original_frozen_runner_sha256": authorization["original_runner_file_sha256"],
        "adapter_commit": runner.git(runner.AUDIT, "rev-parse", "HEAD"),
        "adapter_file_hashes": _assert_committed_adapter(),
        "reservation_sha256": _sha256_file(runner.RESERVATION_PATH),
        "features_ready_sha256": _sha256_file(runner.FEATURES_READY_PATH),
        "fit_ledger_sha256": _sha256_file(runner.FIT_LEDGER_PATH),
        "fit_receipt_hashes": expected_fit_receipts,
        "saved_models_and_predictions_unchanged": True,
        "cross_origin_repair": {
            "only_bypassed_guard": ALIGNMENT_ERROR,
            "bypass_count": len(bypassed),
            **overlap,
            "per_origin_champions_verified_against_own_hash_bound_cache": True,
            "per_origin_missing_fallback_exact": True,
        },
        "prelabel_path": runner.PRELABEL_PATH.relative_to(runner.AUDIT).as_posix(),
        "prelabel_sha256": prelabel_sha,
        "prediction_ready_sha256": prediction_ready_sha,
        "prediction_ready_status": prediction_ready["status"],
        "prelabel_cache_verification": audit,
        "binary_policy": _assert_binary_ignored_and_untracked(runner.PRELABEL_PATH),
        "evaluation_labels_read": False,
        "test_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "metrics_computed": False,
    }
    runner.atomic_json(POSTFIT_RECEIPT_PATH, receipt, write_once=True)
    return receipt


def arm_evaluation(official_root: Path) -> dict[str, Any]:
    _validate_root_repair(official_root)
    _assert_committed_adapter()
    _prelabel_committed_guard(strict_git=True)
    with _ignored_prelabel_commit_shim(enforce_git_policy=True):
        return runner.arm_evaluation(official_root)


def begin_evaluation(official_root: Path) -> dict[str, Any]:
    _validate_root_repair(official_root)
    _assert_committed_adapter()
    _prelabel_committed_guard(strict_git=True)
    with _ignored_prelabel_commit_shim(enforce_git_policy=True):
        return runner.begin_evaluation(official_root)


def evaluate_once(official_root: Path) -> dict[str, Any]:
    _validate_root_repair(official_root)
    _assert_committed_adapter()
    _prelabel_committed_guard(strict_git=True)
    with _ignored_prelabel_commit_shim(enforce_git_policy=True):
        return runner.evaluate_once(official_root)


def _prelabel_committed_guard(*, strict_git: bool) -> dict[str, Any]:
    ready = runner._require_committed_json(runner.PREDICTIONS_READY_PATH)
    receipt = runner._require_committed_json(POSTFIT_RECEIPT_PATH)
    return validate_prelabel_binding(
        runner.PRELABEL_PATH, ready, receipt,
        expected_ready_sha256=receipt.get("prediction_ready_sha256"),
        enforce_git_policy=strict_git,
    )


@contextmanager
def _patch_attributes(target: Any, values: Mapping[str, Any]) -> Iterator[None]:
    old = {name: getattr(target, name) for name in values}
    try:
        for name, value in values.items():
            setattr(target, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(target, name, value)


def _synthetic_prelabel_arrays() -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    early_counts = [1553, 1553, 1553, 1553, 1550, 1550, 1550, 1550]
    late_counts = [2123, 2123, 2123, 2122]
    cycles = np.concatenate([
        np.concatenate([np.full(count, cycle, dtype=np.int64)
                        for cycle, count in zip(range(68, 76), early_counts)]),
        np.concatenate([np.full(count, cycle, dtype=np.int64)
                        for cycle, count in zip(range(76, 80), late_counts)]),
    ])
    n = len(cycles)
    _require(n == 20903 and int(np.sum(cycles >= 76)) == 8491,
             "Synthetic EDA fixture sizes are invalid")
    rows = np.arange(n, dtype=np.int64)
    ids = np.asarray([f"SYN{value:013d}" for value in range(n)], dtype=str)
    present = rows % 11 != 0
    champion_67 = 0.15 + 0.70 * (((rows * 17) % 101) / 100.0)
    champion_75 = 0.13 + 0.72 * (((rows * 29 + 7) % 97) / 96.0)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for origin, select, champion in (
        (67, np.ones(n, dtype=bool), champion_67),
        (75, cycles >= 76, champion_75),
    ):
        idx = rows[select]
        local_present = present[select]
        local_champion = champion[select].astype(np.float64)
        aux_dan = np.clip(local_champion + 0.08 * np.sin(idx * 0.017), 0.02, 0.98)
        aux_linear = np.clip(local_champion + 0.05 * np.cos(idx * 0.023), 0.02, 0.98)
        dan = local_champion.copy()
        linear = local_champion.copy()
        dan[local_present] = 0.75 * local_champion[local_present] + 0.25 * aux_dan[local_present]
        linear[local_present] = 0.75 * local_champion[local_present] + 0.25 * aux_linear[local_present]
        outputs[origin] = {
            "score_rows": idx,
            "score_record_ids": ids[select],
            "score_cycles": cycles[select],
            "present": local_present,
            "champion": local_champion,
            "dan": dan,
            "linear": linear,
            "entry_friction": np.where(idx % 2 == 0, "low", "high").astype(str),
            "text_load_group": np.where(idx % 4 < 2, "Q1", "Q2").astype(str),
            "coverage": ((idx % 100) / 100.0).astype(np.float64),
        }
    return outputs, (rows % 2).astype(np.int8)


def run_synthetic_downstream_qa(output_dir: Path) -> dict[str, Any]:
    """Exercise target-free prelabel binding, real arm/begin/evaluate code on synthetic rows."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _require(not any(root.iterdir()), "Synthetic output directory must be empty")
    audit = root / "audit"
    run = audit / "experiments/CLS_DAN_V1"
    report = audit / "reports/simplification_v1/CLS_DAN_V1"
    official = root / "official"
    train = official / "data/raw/classification/Train.csv"
    prelabel = run / "predictions/PRELABEL.npz"
    prediction_ready_path = run / "PREDICTIONS_READY.json"
    postfit_receipt_path = report / "POSTFIT_PRELABEL_R1.json"
    arrays, labels = _synthetic_prelabel_arrays()
    overlap = validate_origin_alignment(arrays[67], arrays[75])
    _require(overlap["champion_probabilities_different_rows"] > 0,
             "Synthetic overlapping origins must have different champion probabilities")
    run.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    prelabel.parent.mkdir(parents=True, exist_ok=True)
    train.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    origins: dict[str, Any] = {}
    for origin, values in arrays.items():
        prefix = f"E{origin}_"
        payload.update({prefix + key: value for key, value in values.items()})
        missing = ~values["present"]
        origins[str(origin)] = {
            "rows": len(values["score_rows"]),
            "record_ids_sha256": _sha256_array(values["score_record_ids"]),
            "row_positions_sha256": _sha256_array(values["score_rows"]),
            "cycles_sha256": _sha256_array(values["score_cycles"]),
            "present_rows": int(values["present"].sum()),
            "missing_rows": int(missing.sum()),
            "missing_route_exact_for_dan": bool(np.array_equal(
                values["dan"][missing].view(np.uint64), values["champion"][missing].view(np.uint64)
            )),
            "missing_route_exact_for_linear": bool(np.array_equal(
                values["linear"][missing].view(np.uint64), values["champion"][missing].view(np.uint64)
            )),
        }
    payload["prelabel_metadata_json"] = np.asarray(json.dumps({
        "candidate_id": "CLS_DAN_V1", "freeze_signature": "SYNTHETIC_FREEZE",
        "fit_receipt_hashes": {}, "fit_artifact_hashes": {},
        "evaluation_query_targets_read": False, "test_opened": False,
        "sample_submission_opened": False, "kaggle_contacted": False,
    }, sort_keys=True, ensure_ascii=False, allow_nan=False))
    runner.atomic_npz(prelabel, **payload)
    ready = {
        "candidate_id": "CLS_DAN_V1", "status": "PASS_PRELABEL_READY_NO_QUERY_LABELS",
        "freeze_signature": "SYNTHETIC_FREEZE", "source_commit": "synthetic",
        "prelabel_path": prelabel.relative_to(audit).as_posix(),
        "prelabel_sha256": _sha256_file(prelabel), "origins": origins,
        "unique_eval_query_rows": 20903, "primary_unique_expected_rows": 20903,
        "evaluation_query_targets_read": False, "metrics_computed": False,
        "test_opened": False, "sample_submission_opened": False, "kaggle_contacted": False,
    }
    runner.atomic_json(prediction_ready_path, ready, write_once=True)
    repair_receipt = {
        "status": "PASS_SYNTHETIC_POSTFIT_PRELABEL_REPAIR_R1",
        "prelabel_path": prelabel.relative_to(audit).as_posix(),
        "prelabel_sha256": _sha256_file(prelabel),
        "prediction_ready_sha256": _sha256_file(prediction_ready_path),
        "evaluation_labels_read": False,
    }
    runner.atomic_json(postfit_receipt_path, repair_receipt, write_once=True)
    for fit_id in runner.FIT_IDS:
        fit_receipt = run / "units/receipts" / f"{fit_id}.json"
        runner.atomic_json(fit_receipt, {"fit_target_values_read": 0}, write_once=True)
    with train.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows(zip(arrays[67]["score_record_ids"].astype(str), labels.tolist()))

    ready_features = {"origins": {"75": {"fit_label_overlap_with_E67_query_cycles_68_75_rows": 0}}}
    freeze = {"freeze_signature": "SYNTHETIC_FREEZE", "source_commit": "synthetic"}
    original_state = {
        "AUDIT": audit,
        "RUN_ROOT": run,
        "REPORT_ROOT": report,
        "FEATURES_READY_PATH": run / "FEATURES_READY.json",
        "PREDICTIONS_READY_PATH": prediction_ready_path,
        "PRELABEL_PATH": prelabel,
        "RESULTS_PATH": run / "RESULTS.json",
    }
    runner.atomic_json(original_state["FEATURES_READY_PATH"], ready_features, write_once=True)
    original_calls = {"target_reads": 0}
    original_read_targets = runner._read_exact_targets

    def tracked_target_read(*args: Any, **kwargs: Any) -> np.ndarray:
        original_calls["target_reads"] += 1
        return original_read_targets(*args, **kwargs)

    def local_committed_json(path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def local_committed_file(path: Path) -> None:
        _require(Path(path).is_file(), f"Synthetic committed file is missing: {path}")

    patched = {
        **original_state,
        "verify_root_prefit_and_reservation": lambda _root: (freeze, {}, {}),
        "verify_features_ready": lambda _root, freeze=None: (ready_features, {}),
        "_require_committed_json": local_committed_json,
        "_require_committed_file": local_committed_file,
        "_read_exact_targets": tracked_target_read,
    }
    with _patch_attributes(runner, patched), _patch_attributes(
        sys.modules[__name__],
        {"POSTFIT_RECEIPT_PATH": postfit_receipt_path},
    ):
        _prelabel_committed_guard(strict_git=False)
        with _ignored_prelabel_commit_shim(enforce_git_policy=False):
            arm = runner.arm_evaluation(official)
        _prelabel_committed_guard(strict_git=False)
        with _ignored_prelabel_commit_shim(enforce_git_policy=False):
            started = runner.begin_evaluation(official)
        _prelabel_committed_guard(strict_git=False)
        with _ignored_prelabel_commit_shim(enforce_git_policy=False):
            result = runner.evaluate_once(official)
        binding = _prelabel_committed_guard(strict_git=False)
    _require(original_calls["target_reads"] == 1,
             "Synthetic evaluator did not read its query-label vector exactly once")
    _require(arm["query_rows"] == 20903 and started["one_query_label_read_attempt"] is True
             and result["qa"]["query_labels_read_once"] is True
             and result["qa"]["independent_metrics_match"] is True,
             "Synthetic prelabel/arm/begin/score path failed its expected QA")
    return {
        "status": "PASS_SYNTHETIC_PRELABEL_ARM_BEGIN_SCORE",
        "candidate_id": "CLS_DAN_V1",
        "overlap": overlap,
        "prelabel_binding": binding,
        "target_read_calls": original_calls["target_reads"],
        "primary_rows": result["views"]["PRIMARY_UNIQUE"]["n"],
        "evaluation_status": result["status"],
        "independent_metrics_match": result["qa"]["independent_metrics_match"],
        "test_opened": result["query_target_read"]["test_opened"],
        "sample_submission_opened": result["query_target_read"]["sample_submission_opened"],
        "kaggle_contacted": result["query_target_read"]["kaggle_contacted"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorized post-fit CLS DAN R1 adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic", help="Run target-free full downstream synthetic QA")
    synthetic.add_argument("--output-dir", type=Path, required=True)
    for name in ("build-prelabel", "arm", "begin", "score"):
        command = sub.add_parser(name)
        if name == "build-prelabel":
            command.add_argument("--official-root", type=Path, required=True)
        else:
            command.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = run_synthetic_downstream_qa(args.output_dir)
    elif args.command == "build-prelabel":
        result = build_prelabel(args.official_root)
    elif args.command == "arm":
        result = arm_evaluation(args.official_root)
    elif args.command == "begin":
        result = begin_evaluation(args.official_root)
    elif args.command == "score":
        result = evaluate_once(args.official_root)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
