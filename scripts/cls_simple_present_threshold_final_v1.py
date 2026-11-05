"""One final, past-OOF threshold calibration and cached E79 application.

This wrapper adds no predictor or preprocessing fit. Its production source and
all inputs are frozen before the one calibration fit. E79 probabilities are
opened only after the durable final-threshold receipt exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import cls_simple_present_threshold_v1 as pilot

AUDIT_ROOT = ROOT.parent / "tk2-simplification-audit-v1"
AUTH_REL = "reports/simplification_v1/CLS_SIMPLE_THRESHOLD_FINAL_AUTHORIZATION.md"
AUTH_PATH = AUDIT_ROOT / AUTH_REL
AUTH_SHA256 = "80359910f2bebf694de87fa5180b0c2c6e0c9d3ec40fc1a9c3589d569fc1375"
AUTH_COMMIT = "1facbf6b125b9aba23fc0423226f33687adaefa2"

OUT = ROOT / "reports/optimization_v3/cls_simple_present_threshold_final_v1"
RUN = ROOT / "experiments/runs/cls_simple_present_threshold_final_v1"
FREEZE = OUT / "PRE_FIT_FREEZE.json"
FIT_RECEIPT = RUN / "fits/FINAL.json"
APPLY_RECEIPT = RUN / "apply/E79_APPLIED.json"
LEDGER = RUN / "LEDGER.jsonl"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_simple_present_threshold_final_v1/classification_threshold_final.csv"

TRAIN = ROOT / "data/raw/classification/Train.csv"
OOF_CACHE = pilot.CACHE
E79_CACHE = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
E79_APPLICATION_FREEZE = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_APPLICATION_FREEZE_V2.json"
E79_MANIFEST = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_MANIFEST.json"
E79_QA = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/E79_QA.json"

COMPARISONS = {
    "current_public_champion": ROOT / "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv",
    "prior_a02": ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv",
    "no_pos": ROOT / "submissions/optimization_v3/cls_simplification_benchmark_v1/classification_no_pos_e79.csv",
}

SOURCE_PATHS = (
    "scripts/cls_simple_present_threshold_final_v1.py",
    "scripts/verify_cls_simple_present_threshold_final_v1.py",
    "tests/test_cls_simple_present_threshold_final_v1.py",
    "reports/optimization_v3/cls_simple_present_threshold_final_v1/PROTOCOL.md",
)
INPUTS = {
    "official_train_csv": TRAIN,
    "a02_oof_cache": OOF_CACHE,
    "canonical_e79_cache": E79_CACHE,
    "official_test_csv": TEST,
    "sample_submission_csv": SAMPLE,
    "e79_application_freeze": E79_APPLICATION_FREEZE,
    "e79_manifest": E79_MANIFEST,
    "e79_qa": E79_QA,
    "threshold_source_freeze": pilot.FREEZE,
    "threshold_source_ready": pilot.READY,
    "threshold_d1_d2_release": pilot.OUT / "ROOT_RELEASE_D1_D2.json",
    "threshold_postfit_audit": pilot.OUT / "POSTFIT_INDEPENDENT_AUDIT.json",
    "root_final_authorization": AUTH_PATH,
    "pilot_producer": ROOT / "scripts/cls_simple_present_threshold_v1.py",
    "pilot_independent_verifier": ROOT / "scripts/verify_cls_simple_present_threshold_v1.py",
    "pilot_postfit_auditor": ROOT / "scripts/audit_cls_simple_present_threshold_v1_postfit.py",
    **{f"comparison_{key}": path for key, path in COMPARISONS.items()},
}

EXPECTED_OFFICIAL_INPUTS = {
    "official_train_csv": "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c",
    "a02_oof_cache": "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351",
    "canonical_e79_cache": "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271",
    "official_test_csv": "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c",
    "sample_submission_csv": "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7",
    "root_final_authorization": AUTH_SHA256,
}

EXPECTED_ROWS = {"E67_68_75": 12_412, "E75_76_79": 8_491, "pooled": 20_903}
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_TEST_ROWS = 29_421
LOW, HIGH, CENTRE = 0.45, 0.55, 0.50
MISSING_THRESHOLD = 0.53230976129477
MIN_PRESENT_PER_CLASS = 200
FIT_CAP = 1
FINAL_SLOT = "FINAL_A02_PAST_OOF_THRESHOLD"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) or "event_id" not in row for row in events):
        raise ValueError("Malformed final-fit ledger")
    ids = [row["event_id"] for row in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event ID in final-fit ledger")
    return events


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8").strip()


def _auth_check() -> None:
    if not AUTH_PATH.is_file() or sha256_file(AUTH_PATH) != AUTH_SHA256:
        raise RuntimeError("Root final authorization is missing or changed")
    if _git(AUDIT_ROOT, "rev-parse", "HEAD") != AUTH_COMMIT:
        raise RuntimeError("Root authorization checkout HEAD changed; verify the new checkpoint before work")


def _validate_upstream_e79() -> None:
    application = json.loads(E79_APPLICATION_FREEZE.read_text(encoding="utf-8"))
    manifest = json.loads(E79_MANIFEST.read_text(encoding="utf-8"))
    qa = json.loads(E79_QA.read_text(encoding="utf-8"))
    if application.get("status") != "FROZEN_BEFORE_E79_PROBABILITY_BUNDLE_ACCESS" or application.get("test_labels_read") is not False:
        raise RuntimeError("Canonical E79 application freeze is not admissible")
    expected_inputs = application.get("application_input_hashes", {})
    if expected_inputs.get("experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz") != EXPECTED_OFFICIAL_INPUTS["canonical_e79_cache"]:
        raise RuntimeError("E79 application freeze is bound to another probability bundle")
    if expected_inputs.get("data/raw/classification/Test.csv") != EXPECTED_OFFICIAL_INPUTS["official_test_csv"]:
        raise RuntimeError("E79 application freeze is bound to another Test file")
    if manifest.get("status") != "READY_FOR_ROOT_REVIEW_ONE_UPLOAD_MAX" or manifest.get("test_labels_read") is not False:
        raise RuntimeError("Canonical E79 manifest is not admissible")
    if manifest.get("selected_missing_threshold") != MISSING_THRESHOLD or manifest.get("threshold_operator") != ">=":
        raise RuntimeError("Canonical E79 missing route differs from the authorized route")
    if qa.get("status") != "PASS_LOCAL_E79_QA":
        raise RuntimeError("Canonical E79 application QA did not pass")


def _validate_pilot_lineage() -> None:
    source_freeze = json.loads(pilot.FREEZE.read_text(encoding="utf-8"))
    source_ready = json.loads(pilot.READY.read_text(encoding="utf-8"))
    release = json.loads((pilot.OUT / "ROOT_RELEASE_D1_D2.json").read_text(encoding="utf-8"))
    postfit = json.loads((pilot.OUT / "POSTFIT_INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))
    if source_freeze.get("source_commit") != "aa0bf8a7cb2a7dfec3e41f557921bce31f88f09b":
        raise RuntimeError("Historical D1/D2 frozen producer commit changed")
    for relative, expected in source_freeze.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Historical D1/D2 source hash mismatch: {relative}")
    if source_ready.get("status") != "PASS" or source_ready.get("source_commit") != source_freeze["source_commit"]:
        raise RuntimeError("Historical D1/D2 source-ready receipt is not valid")
    if (release.get("status") != "ROOT_RELEASED_FOR_D1_D2" or release.get("final_fit_authorized") is not False
            or release.get("Kaggle_access") is not False or release.get("Test_access") is not False):
        raise RuntimeError("Historical D1/D2 release scope has been changed")
    if (postfit.get("status") != "PASS" or postfit.get("check_count") != 20
            or postfit.get("qualification") != "PASS" or postfit.get("exploratory_review") != "PASS"):
        raise RuntimeError("Historical D1/D2 independent audit no longer passes")


def _source_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return "../" + resolved.relative_to(ROOT.parent).as_posix()
        except ValueError:
            return str(resolved)


def _require_synthetic_temp_paths(paths: list[Path]) -> None:
    temp_root = Path(tempfile.gettempdir()).resolve()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(temp_root) or resolved.is_relative_to(ROOT.resolve()):
            raise ValueError("Synthetic fixture mode is restricted to isolated temporary files outside the repository")


@contextmanager
def _calibration_resource_gate(*, synthetic: bool):
    if not synthetic:
        with pilot.single_job_resource_gate():
            yield
        return
    # Synthetic tests are tiny, isolated fixtures. Keep the shared and lane
    # locks, while using a proportionate memory floor; official calibration
    # always uses the unchanged frozen project thresholds above.
    saved = (pilot.RESOURCE_START_BYTES, pilot.RESOURCE_RESERVE_BYTES, pilot.RESOURCE_ESTIMATE_BYTES)
    pilot.RESOURCE_START_BYTES = 512 * 1024**2
    pilot.RESOURCE_RESERVE_BYTES = 256 * 1024**2
    pilot.RESOURCE_ESTIMATE_BYTES = 64 * 1024**2
    try:
        with pilot.single_job_resource_gate():
            yield
    finally:
        pilot.RESOURCE_START_BYTES, pilot.RESOURCE_RESERVE_BYTES, pilot.RESOURCE_ESTIMATE_BYTES = saved


def _input_paths(train_path: Path, oof_path: Path, e79_path: Path, test_path: Path,
                 sample_path: Path, *, synthetic: bool) -> dict[str, Path]:
    paths = {
        "official_train_csv": train_path,
        "a02_oof_cache": oof_path,
        "canonical_e79_cache": e79_path,
        "official_test_csv": test_path,
        "sample_submission_csv": sample_path,
    }
    if not synthetic:
        paths.update({key: path for key, path in INPUTS.items() if key not in paths})
    return paths


def create_pre_fit_freeze(*, train_path: Path = TRAIN, oof_path: Path = OOF_CACHE,
                          e79_path: Path = E79_CACHE, test_path: Path = TEST,
                          sample_path: Path = SAMPLE, freeze_path: Path = FREEZE,
                          synthetic: bool = False) -> dict[str, Any]:
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite pre-fit freeze: {freeze_path}")
    if not synthetic:
        _auth_check()
        _validate_pilot_lineage()
        _validate_upstream_e79()
    input_paths = _input_paths(train_path, oof_path, e79_path, test_path, sample_path, synthetic=synthetic)
    if synthetic:
        _require_synthetic_temp_paths(list(input_paths.values()) + [freeze_path])
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen inputs are missing: {missing}")
    input_hashes = {key: sha256_file(path) for key, path in input_paths.items()}
    if not synthetic:
        for key, expected in EXPECTED_OFFICIAL_INPUTS.items():
            if input_hashes[key] != expected:
                raise RuntimeError(f"Official input hash mismatch for {key}")
    source_status = _git(ROOT, "status", "--porcelain", "--", *SOURCE_PATHS)
    if source_status:
        raise RuntimeError("Final source files must be committed and clean before the pre-fit freeze")
    freeze = {
        "schema_version": 1,
        "campaign": "CLS_SIMPLE_PRESENT_THRESHOLD_FINAL_V1",
        "status": "SYNTHETIC_TEST_FREEZE" if synthetic else "FROZEN_BEFORE_FINAL_CALIBRATION",
        "synthetic_fixture_only": synthetic,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {"path": str(AUTH_PATH), "sha256": AUTH_SHA256, "source_commit": AUTH_COMMIT},
        "source_commit": _git(ROOT, "rev-parse", "HEAD"),
        "source_hashes": _source_hashes(),
        "input_hashes": input_hashes,
        "input_paths": {key: _path_label(path) for key, path in input_paths.items()},
        "fit_cap": FIT_CAP,
        "historical_diagnostic_fit_count_separate": 2,
        "fit_population": {
            "E67": {"cycles": [68, 75], "rows": EXPECTED_ROWS["E67_68_75"]},
            "E75": {"cycles": [76, 79], "rows": EXPECTED_ROWS["E75_76_79"]},
            "pooled_unique_rows": EXPECTED_ROWS["pooled"],
            "target_source": "official Train artifact_signal for exact cached OOF IDs only",
            "oof_target_members_opened": False,
        },
        "recipe": {
            "score": "canonical E67/E75 OOF pA",
            "objective": "full-row binary Macro F1, exact rational comparisons, labels [0,1], zero_division=0",
            "present_threshold_interval_inclusive": [LOW, HIGH],
            "present_decision": "pA >= final threshold",
            "missing_decision": f"pA >= {MISSING_THRESHOLD}",
            "present_minimum_class_support": MIN_PRESENT_PER_CLASS,
            "tie_break": ["maximum exact Macro F1", "minimum distance to 0.50", "higher threshold"],
            "predictor_fits": 0,
            "preprocessing_fits": 0,
            "test_label_access": False,
        },
    }
    write_json_exclusive(freeze_path, freeze)
    return freeze


def _verify_source_and_inputs(freeze_path: Path, expected_inputs: dict[str, Path], *, synthetic: bool = False) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("campaign") != "CLS_SIMPLE_PRESENT_THRESHOLD_FINAL_V1" or freeze.get("synthetic_fixture_only") != synthetic:
        raise RuntimeError("Pre-fit freeze scope does not match this invocation")
    if freeze.get("fit_cap") != FIT_CAP or freeze.get("recipe", {}).get("predictor_fits") != 0:
        raise RuntimeError("Frozen cap or no-predictor-fit boundary is invalid")
    if _source_hashes() != freeze.get("source_hashes"):
        raise RuntimeError("Final wrapper/verifier/tests/protocol hashes changed after freeze")
    if _git(ROOT, "status", "--porcelain", "--", *SOURCE_PATHS):
        raise RuntimeError("Frozen source files or pre-fit freeze have uncommitted changes")
    if freeze_path.is_relative_to(ROOT):
        freeze_relative = freeze_path.relative_to(ROOT).as_posix()
        if _git(ROOT, "status", "--porcelain", "--", freeze_relative):
            raise RuntimeError("Frozen source files or pre-fit freeze have uncommitted changes")
        committed_freeze = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{freeze_relative}"])
        if hashlib.sha256(committed_freeze).hexdigest() != sha256_file(freeze_path):
            raise RuntimeError("Pre-fit freeze differs from the committed copy")
    source_commit = freeze["source_commit"]
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", source_commit, "HEAD"], check=False).returncode != 0:
        raise RuntimeError("Frozen source commit is not an ancestor of the current checkout")
    if set(expected_inputs) != set(freeze.get("input_paths", {})):
        raise RuntimeError("Runtime input path set differs from the pre-fit freeze")
    for key, path in expected_inputs.items():
        if _path_label(path) != freeze["input_paths"].get(key):
            raise RuntimeError(f"Runtime input path differs from the pre-fit freeze: {key}")
        if not path.is_file() or sha256_file(path) != freeze["input_hashes"].get(key):
            raise RuntimeError(f"Frozen input hash mismatch: {key}")
    if not synthetic:
        _auth_check()
        _validate_pilot_lineage()
        _validate_upstream_e79()
        for key, expected in EXPECTED_OFFICIAL_INPUTS.items():
            if freeze["input_hashes"].get(key) != expected:
                raise RuntimeError(f"Frozen official input differs from authorized identity: {key}")
    return freeze


def _load_oof_pool(cache_path: Path, train_path: Path, *, synthetic: bool = False) -> dict[str, np.ndarray]:
    import pandas as pd

    train_meta = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"}, low_memory=False)
    if train_meta["record_id"].isna().any() or train_meta["cycle_code"].isna().any():
        raise ValueError("Official Train metadata IDs/cycles contain missing values")
    train_meta["record_id"] = train_meta["record_id"].astype(str)
    train_meta["cycle_code"] = pd.to_numeric(train_meta["cycle_code"], errors="raise").astype(int)
    if train_meta["record_id"].isna().any() or train_meta["record_id"].duplicated().any():
        raise ValueError("Official Train metadata IDs are missing or duplicated")
    if not synthetic and (len(train_meta) != EXPECTED_TRAIN_ROWS or (int(train_meta["cycle_code"].min()), int(train_meta["cycle_code"].max())) != (47, 79)):
        raise ValueError("Official Train row/cycle population differs from frozen identity")
    cycle_by_id = dict(zip(train_meta["record_id"], train_meta["cycle_code"]))
    ids_parts, cycles_parts, pa_parts, missing_parts = [], [], [], []
    with np.load(cache_path, allow_pickle=False) as archive:
        for origin, low_cycle, high_cycle in ((67, 68, 75), (75, 76, 79)):
            ids = np.asarray(archive[f"E{origin}_ids"], dtype="U")
            cycles = np.asarray(archive[f"E{origin}_cycles"], dtype=np.int64)
            missing = np.asarray(archive[f"E{origin}_missing"], dtype=bool)
            p_a = np.asarray(archive[f"E{origin}_pA"], dtype=np.float64)
            if ids.ndim != 1 or len(ids) != len(set(ids.tolist())):
                raise ValueError(f"E{origin} OOF IDs are not unique one-dimensional strings")
            if not (len(ids) == len(cycles) == len(missing) == len(p_a)):
                raise ValueError(f"E{origin} OOF vectors have inconsistent row counts")
            if not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
                raise ValueError(f"E{origin} OOF pA contains invalid probabilities")
            allowed = np.arange(68, 80) if origin == 67 else np.arange(76, 80)
            if not np.isin(cycles, allowed).all():
                raise ValueError(f"E{origin} OOF cycle values are outside the frozen archive range")
            keep = (cycles >= low_cycle) & (cycles <= high_cycle)
            ids, cycles, missing, p_a = ids[keep], cycles[keep], missing[keep], p_a[keep]
            if not synthetic and len(ids) != EXPECTED_ROWS[f"E{origin}_{low_cycle}_{high_cycle}"]:
                raise ValueError(f"E{origin} OOF selected row count changed")
            if any(item not in cycle_by_id or cycle_by_id[item] != int(cycle) for item, cycle in zip(ids.tolist(), cycles.tolist())):
                raise ValueError(f"E{origin} OOF ID/cycle mapping differs from official Train")
            ids_parts.append(ids)
            cycles_parts.append(cycles)
            pa_parts.append(p_a)
            missing_parts.append(missing)
    ids = np.concatenate(ids_parts)
    cycles = np.concatenate(cycles_parts)
    p_a = np.concatenate(pa_parts)
    missing = np.concatenate(missing_parts)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("E67/E75 pooled OOF calibration IDs overlap")
    if not synthetic and len(ids) != EXPECTED_ROWS["pooled"]:
        raise ValueError("Pooled OOF calibration population differs from 20,903 rows")
    return {"ids": ids, "cycles": cycles, "pA": p_a, "missing": missing}


def _read_selected_labels(train_path: Path, roster_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    import pandas as pd

    roster_ids = np.asarray(roster_ids).astype(str)
    selected_ids = np.asarray(selected_ids).astype(str)
    row_by_id = {value: i for i, value in enumerate(roster_ids.tolist())}
    if len(row_by_id) != len(roster_ids) or len(selected_ids) != len(set(selected_ids.tolist())):
        raise ValueError("Train roster or selected calibration IDs contain duplicates")
    if not set(selected_ids.tolist()).issubset(row_by_id):
        raise ValueError("OOF calibration IDs are absent from official Train")
    selected_positions = {row_by_id[value] for value in selected_ids.tolist()}
    raw = train_path.read_bytes()
    line_count = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    if line_count != len(roster_ids) + 1:
        raise ValueError("Train physical lines do not map one-to-one to parsed records")
    frame = pd.read_csv(
        train_path,
        usecols=["record_id", "artifact_signal"],
        dtype={"record_id": "string", "artifact_signal": "Int8"},
        skiprows=lambda line: line > 0 and (line - 1) not in selected_positions,
        low_memory=False,
    )
    frame["record_id"] = frame["record_id"].astype(str)
    if frame["record_id"].duplicated().any() or set(frame["record_id"].tolist()) != set(selected_ids.tolist()):
        raise ValueError("Selected-only target reader returned unexpected calibration IDs")
    mapping = dict(zip(frame["record_id"], frame["artifact_signal"].astype(int)))
    labels = np.asarray([mapping[value] for value in selected_ids.tolist()], dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Selected Train targets are not binary")
    return labels


def _fit_receipt_valid(path: Path, event: dict[str, Any], freeze_path: Path) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != event.get("receipt_sha256"):
        raise RuntimeError("Fit completion receipt does not match its durable ledger event")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_FINAL_THRESHOLD" or receipt.get("fit_ids_sha256") != event.get("fit_ids_sha256"):
        raise RuntimeError("Existing final threshold receipt is inconsistent with the started fit")
    if receipt.get("pre_fit_freeze_sha256") != sha256_file(freeze_path):
        raise RuntimeError("Existing final threshold receipt is bound to another freeze")
    if not LOW <= float(receipt["threshold"]) <= HIGH or receipt.get("threshold_hex") != float(receipt["threshold"]).hex():
        raise RuntimeError("Existing final threshold receipt has an invalid threshold")
    return receipt


def fit_final_threshold(*, train_path: Path = TRAIN, cache_path: Path = OOF_CACHE, freeze_path: Path = FREEZE,
                        e79_path: Path = E79_CACHE, test_path: Path = TEST, sample_path: Path = SAMPLE,
                        run_dir: Path = RUN, synthetic: bool = False) -> dict[str, Any]:
    # The old pilot's resource gate coordinates all heavy KASDAD jobs. This is
    # only a small calibrator, but retaining the shared/lane lock is mandatory.
    if synthetic:
        _require_synthetic_temp_paths([train_path, cache_path, e79_path, test_path, sample_path, freeze_path, run_dir])
    with _calibration_resource_gate(synthetic=synthetic):
        input_paths = _input_paths(train_path, cache_path, e79_path, test_path, sample_path, synthetic=synthetic)
        freeze = _verify_source_and_inputs(freeze_path, input_paths, synthetic=synthetic)
        ledger_path = run_dir / "LEDGER.jsonl"
        receipt_path = run_dir / "fits/FINAL.json"
        events = _read_ledger(ledger_path)
        starts = [row for row in events if row.get("slot") == FINAL_SLOT and row.get("event") == "FIT_STARTED"]
        completions = [row for row in events if row.get("slot") == FINAL_SLOT and row.get("event") in {"FIT_FROZEN", "FIT_RECOVERED", "FIT_UNSUPPORTED"}]
        if len(starts) > FIT_CAP or len(completions) > 1:
            raise RuntimeError("Final fit ledger exceeds its one-slot cap")
        if completions:
            if completions[-1].get("event") == "FIT_UNSUPPORTED":
                if not receipt_path.is_file() or sha256_file(receipt_path) != completions[-1].get("receipt_sha256"):
                    raise RuntimeError("Unsupported-fit receipt does not match its final ledger event")
                return json.loads(receipt_path.read_text(encoding="utf-8"))
            return _fit_receipt_valid(receipt_path, completions[-1], freeze_path)
        if starts:
            if receipt_path.exists():
                recovered = json.loads(receipt_path.read_text(encoding="utf-8"))
                start = starts[-1]
                if (recovered.get("status") == "FROZEN_FINAL_THRESHOLD"
                        and recovered.get("fit_ids_sha256") == start.get("fit_ids_sha256")
                        and recovered.get("pre_fit_freeze_sha256") == sha256_file(freeze_path)
                        and LOW <= float(recovered.get("threshold", float("nan"))) <= HIGH
                        and recovered.get("threshold_hex") == float(recovered["threshold"]).hex()):
                    digest = sha256_file(receipt_path)
                    _append_event(ledger_path, {"event_id": f"{FINAL_SLOT}:FIT_RECOVERED", "event": "FIT_RECOVERED",
                                                "slot": FINAL_SLOT, "receipt_sha256": digest,
                                                "fit_ids_sha256": recovered["fit_ids_sha256"],
                                                "recovered_after_durable_receipt": True,
                                                "at_utc": datetime.now(timezone.utc).isoformat()})
                    return recovered
            raise RuntimeError("FIT_STARTED has no valid durable receipt; retry is blocked")
        if receipt_path.exists():
            raise RuntimeError("Orphan fit receipt exists without a start event; refusing overwrite")
        pool = _load_oof_pool(cache_path, train_path, synthetic=synthetic)
        ids = pool["ids"]
        fit_ids_sha256 = hashlib.sha256("\n".join(ids.tolist()).encode("utf-8")).hexdigest()
        _append_event(ledger_path, {
            "event_id": f"{FINAL_SLOT}:FIT_STARTED", "event": "FIT_STARTED", "slot": FINAL_SLOT,
            "fit_ids_sha256": fit_ids_sha256, "fit_rows": len(ids),
            "official_train_sha256": sha256_file(train_path), "oof_cache_sha256": sha256_file(cache_path),
            "pre_fit_freeze_sha256": sha256_file(freeze_path), "at_utc": datetime.now(timezone.utc).isoformat(),
        })
        train_roster = np.asarray(_load_train_ids(train_path), dtype="U")
        labels = _read_selected_labels(train_path, train_roster, ids)
        try:
            optimized = pilot.exact_threshold_fit(pool["pA"], pool["missing"], labels)
        except RuntimeError as exc:
            if not str(exc).startswith("Unsupported calibration population:"):
                raise
            unsupported = {
                "schema_version": 1, "status": "UNSUPPORTED_CALIBRATION_POPULATION", "slot": FINAL_SLOT,
                "fit_rows": len(ids), "fit_ids_sha256": fit_ids_sha256,
                "present_class_support": {str(c): int(np.sum((~pool["missing"]) & (labels == c))) for c in (0, 1)},
                "pre_fit_freeze_sha256": sha256_file(freeze_path),
            }
            digest = write_json_exclusive(receipt_path, unsupported)
            _append_event(ledger_path, {"event_id": f"{FINAL_SLOT}:FIT_UNSUPPORTED", "event": "FIT_UNSUPPORTED",
                                        "slot": FINAL_SLOT, "fit_ids_sha256": fit_ids_sha256, "receipt_sha256": digest,
                                        "at_utc": datetime.now(timezone.utc).isoformat()})
            return unsupported
        receipt = {
            "schema_version": 1, "status": "FROZEN_FINAL_THRESHOLD", "slot": FINAL_SLOT,
            "fit_rows": len(ids), "fit_ids_sha256": fit_ids_sha256,
            "fit_cycle_counts": {str(cycle): int(np.sum(pool["cycles"] == cycle)) for cycle in np.unique(pool["cycles"])},
            "fit_cycles": [68, 79], "origin_composition": {"E67_cycles_68_75": 12_412, "E75_cycles_76_79": 8_491},
            "threshold": optimized["threshold"], "threshold_hex": optimized["threshold_hex"],
            "decision_state_interval": optimized["decision_state_interval"],
            "fit_macro_f1": optimized["fit_macro_f1"], "fit_macro_f1_exact": optimized["fit_macro_f1_exact"],
            "fit_confusion": optimized["fit_confusion"], "present_class_support": optimized["present_class_support"],
            "breakpoints_considered": optimized["present_score_breakpoints_considered"],
            "official_train_sha256": sha256_file(train_path), "oof_cache_sha256": sha256_file(cache_path),
            "pre_fit_freeze_sha256": sha256_file(freeze_path), "source_commit": freeze["source_commit"],
            "source_hashes": freeze["source_hashes"], "input_hashes": freeze["input_hashes"],
            "target_source": "official Train artifact_signal for selected cached OOF IDs only",
            "oof_target_members_opened": False, "predictor_fits": 0, "preprocessing_fits": 0,
            "test_or_sample_submission_accessed": False,
            "recipe": "CLS_SIMPLE_PRESENT_THRESHOLD_FINAL_V1; exact frozen pilot objective and tie rules",
        }
        digest = write_json_exclusive(receipt_path, receipt)
        _append_event(ledger_path, {"event_id": f"{FINAL_SLOT}:FIT_FROZEN", "event": "FIT_FROZEN", "slot": FINAL_SLOT,
                                    "fit_ids_sha256": fit_ids_sha256, "receipt_sha256": digest,
                                    "at_utc": datetime.now(timezone.utc).isoformat()})
        return receipt


def _load_train_ids(train_path: Path) -> list[str]:
    import pandas as pd
    frame = pd.read_csv(train_path, usecols=["record_id"], dtype={"record_id": "string"}, low_memory=False)
    if frame["record_id"].isna().any():
        raise ValueError("Official Train ID roster contains missing values")
    ids = frame["record_id"].astype(str).tolist()
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Official Train ID roster contains missing or duplicated values")
    return ids


def _read_test_ids_missing(path: Path, *, synthetic: bool = False) -> tuple[list[str], np.ndarray]:
    import pandas as pd
    frame = pd.read_csv(path, usecols=["record_id", "signal_stream"], dtype="string", keep_default_na=False, encoding="utf-8-sig")
    ids = frame["record_id"].astype(str).tolist()
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Test IDs contain empty or duplicate values")
    missing = frame["signal_stream"].str.strip().eq("").to_numpy(dtype=bool)
    if not synthetic and len(ids) != EXPECTED_TEST_ROWS:
        raise ValueError("Official Test row count differs from 29,421")
    return ids, missing


def _read_sample(path: Path, *, synthetic: bool = False) -> tuple[list[str], list[str]]:
    import pandas as pd
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if header != ["record_id", "artifact_signal"]:
        raise ValueError(f"Unexpected SampleSubmission columns: {header}")
    ids = pd.read_csv(path, usecols=["record_id"], dtype={"record_id": "string"}, keep_default_na=False, encoding="utf-8-sig")["record_id"].astype(str).tolist()
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("SampleSubmission IDs contain empty or duplicate values")
    if not synthetic and len(ids) != EXPECTED_TEST_ROWS:
        raise ValueError("SampleSubmission row count differs from 29,421")
    return ids, header


def _read_submission(path: Path, expected_ids: list[str]) -> np.ndarray:
    ids, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["record_id", "artifact_signal"]:
            raise ValueError(f"Unexpected CSV header: {path}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2 or row[1] not in {"0", "1"}:
                raise ValueError(f"Malformed or nonbinary CSV row at {path}:{line_no}")
            ids.append(row[0])
            labels.append(int(row[1]))
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError(f"CSV IDs/order differ from official Test: {path}")
    return np.asarray(labels, dtype=np.int8)


def _write_csv_once(path: Path, ids: list[str], labels: np.ndarray) -> str:
    values = np.asarray(labels)
    if values.shape != (len(ids),) or not np.isin(values, [0, 1]).all():
        raise ValueError("CSV predictions must be aligned integer 0/1 values")
    payload = ("record_id,artifact_signal\n" + "".join(f"{record_id},{int(label)}\n" for record_id, label in zip(ids, values))).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != digest:
            raise FileExistsError("Refusing to overwrite an existing different final CSV")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return digest


def _load_e79_vectors(path: Path, *, synthetic: bool = False) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["record_ids"], dtype="U").tolist()
        missing = np.asarray(archive["missing_mask"], dtype=bool)
        p_a = np.asarray(archive["a02_probability"], dtype=np.float64)
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("E79 probability archive IDs are empty or duplicated")
    if not (len(ids) == len(missing) == len(p_a)) or not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
        raise ValueError("E79 probability archive vectors are malformed")
    if not synthetic and len(ids) != EXPECTED_TEST_ROWS:
        raise ValueError("Canonical E79 probability archive has the wrong row count")
    return ids, missing, p_a


def apply_frozen_threshold(*, test_path: Path = TEST, sample_path: Path = SAMPLE, e79_path: Path = E79_CACHE,
                           freeze_path: Path = FREEZE, run_dir: Path = RUN, output_path: Path = OUTPUT_CSV,
                           train_path: Path = TRAIN, oof_path: Path = OOF_CACHE,
                           synthetic: bool = False) -> dict[str, Any]:
    input_paths = _input_paths(train_path, oof_path, e79_path, test_path, sample_path, synthetic=synthetic)
    if synthetic:
        _require_synthetic_temp_paths(list(input_paths.values()) + [freeze_path, run_dir, output_path])
    freeze = _verify_source_and_inputs(freeze_path, input_paths, synthetic=synthetic)
    fit_path = run_dir / "fits/FINAL.json"
    ledger_path = run_dir / "LEDGER.jsonl"
    if not fit_path.is_file():
        raise FileNotFoundError("Final learned threshold receipt must exist before E79 access")
    fit_digest = sha256_file(fit_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("status") != "FROZEN_FINAL_THRESHOLD" or fit.get("pre_fit_freeze_sha256") != sha256_file(freeze_path):
        raise RuntimeError("Final threshold receipt is not frozen for this input freeze")
    existing = [row for row in _read_ledger(ledger_path) if row.get("event") == "E79_APPLIED"]
    receipt_path = run_dir / "apply/E79_APPLIED.json"
    if existing:
        if len(existing) != 1 or not receipt_path.is_file() or sha256_file(receipt_path) != existing[-1].get("receipt_sha256"):
            raise RuntimeError("Existing E79 application ledger is inconsistent; refusing repeat")
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    events = _read_ledger(ledger_path)
    apply_starts = [row for row in events if row.get("event") == "E79_APPLY_STARTED"]
    if apply_starts:
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            csv_path = ROOT / prior["csv_path"] if not synthetic else output_path
            if (prior.get("status") == "E79_APPLIED_QA_REQUIRED"
                    and prior.get("fit_receipt_sha256") == fit_digest
                    and prior.get("pre_fit_freeze_sha256") == sha256_file(freeze_path)
                    and csv_path.is_file() and prior.get("csv_sha256") == sha256_file(csv_path)):
                digest = sha256_file(receipt_path)
                _append_event(ledger_path, {"event_id": f"{FINAL_SLOT}:E79_APPLIED", "event": "E79_APPLIED",
                                            "slot": FINAL_SLOT, "csv_sha256": prior["csv_sha256"],
                                            "receipt_sha256": digest, "recovered_after_durable_receipt": True,
                                            "at_utc": datetime.now(timezone.utc).isoformat()})
                return prior
        raise RuntimeError("E79 application has an ambiguous start without a completion receipt; refusing repeat")
    _append_event(ledger_path, {"event_id": f"{FINAL_SLOT}:E79_APPLY_STARTED", "event": "E79_APPLY_STARTED",
                                "slot": FINAL_SLOT, "fit_receipt_sha256": fit_digest,
                                "e79_cache_sha256": sha256_file(e79_path),
                                "at_utc": datetime.now(timezone.utc).isoformat()})
    e79_ids, e79_missing, p_a = _load_e79_vectors(e79_path, synthetic=synthetic)
    test_ids, test_missing = _read_test_ids_missing(test_path, synthetic=synthetic)
    sample_ids, sample_header = _read_sample(sample_path, synthetic=synthetic)
    if e79_ids != test_ids or sample_ids != test_ids:
        raise ValueError("E79/Test/SampleSubmission ID order does not match exactly")
    if not np.array_equal(e79_missing, test_missing):
        raise ValueError("E79 missing mask differs from Test stream availability")
    labels = np.where(e79_missing, p_a >= MISSING_THRESHOLD, p_a >= float(fit["threshold"])).astype(np.int8)
    csv_hash = _write_csv_once(output_path, e79_ids, labels)
    replay = _read_submission(output_path, test_ids)
    if not np.array_equal(replay, labels):
        raise AssertionError("CSV replay differs from frozen E79 decisions")
    comparisons: dict[str, Any] = {}
    for name, path in (COMPARISONS.items() if not synthetic else []):
        if path.is_file():
            comparison = _read_submission(path, test_ids)
            diff = int(np.sum(comparison != labels))
            comparisons[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path),
                                 "changed_decisions": diff, "positive_delta": int(np.sum(labels) - np.sum(comparison)),
                                 "identical": diff == 0}
            if name == "no_pos":
                comparisons[name]["status"] = "PREVIOUSLY_UPLOADED_REF_58024217_COMPLETE"
        else:
            comparisons[name] = {"path": str(path), "status": "MISSING"}
    duplicates = []
    submissions_root = ROOT / "submissions"
    candidate_vector_hash = hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int8).tobytes()).hexdigest()
    for path in sorted(submissions_root.rglob("*.csv")) if not synthetic else []:
        if path.resolve() == output_path.resolve() or path in COMPARISONS.values():
            continue
        try:
            other = _read_submission(path, test_ids)
        except (OSError, UnicodeError, ValueError):
            continue
        if hashlib.sha256(np.ascontiguousarray(other, dtype=np.int8).tobytes()).hexdigest() == candidate_vector_hash:
            duplicates.append(path.relative_to(ROOT).as_posix())
    if duplicates:
        duplicates.append("STOP_UPLOAD_DUPLICATE_LOCAL_VECTOR")
    receipt = {
        "schema_version": 1, "status": "E79_APPLIED_QA_REQUIRED", "slot": FINAL_SLOT,
        "fit_receipt_sha256": fit_digest, "pre_fit_freeze_sha256": sha256_file(freeze_path),
        "source_commit": freeze["source_commit"], "input_hashes": freeze["input_hashes"],
        "e79_cache_sha256": sha256_file(e79_path), "csv_path": _path_label(output_path),
        "csv_sha256": csv_hash, "rows": len(labels), "prediction_vector_sha256": candidate_vector_hash,
        "positive_count": int(np.sum(labels)), "negative_count": int(np.sum(labels == 0)),
        "missing_rows": int(np.sum(e79_missing)), "present_rows": int(np.sum(~e79_missing)),
        "missing_rows_decision_unchanged_from_frozen_threshold": True,
        "threshold": fit["threshold"], "missing_threshold": MISSING_THRESHOLD,
        "sample_submission_header": sample_header, "comparisons": comparisons,
        "local_duplicate_vectors": duplicates, "test_columns_read": ["record_id", "signal_stream"],
        "test_target_read": False,
    }
    digest = write_json_exclusive(receipt_path, receipt)
    _append_event(ledger_path, {"event_id": f"{FINAL_SLOT}:E79_APPLIED", "event": "E79_APPLIED", "slot": FINAL_SLOT,
                                "csv_sha256": csv_hash, "receipt_sha256": digest,
                                "at_utc": datetime.now(timezone.utc).isoformat()})
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=TRAIN)
    parser.add_argument("--oof-cache", type=Path, default=OOF_CACHE)
    parser.add_argument("--e79-cache", type=Path, default=E79_CACHE)
    parser.add_argument("--test-csv", type=Path, default=TEST)
    parser.add_argument("--sample-csv", type=Path, default=SAMPLE)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--synthetic-fixture", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("fit")
    sub.add_parser("apply")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        value = create_pre_fit_freeze(train_path=args.train_csv, oof_path=args.oof_cache,
                                      e79_path=args.e79_cache, test_path=args.test_csv,
                                      sample_path=args.sample_csv, freeze_path=args.freeze,
                                      synthetic=args.synthetic_fixture)
    elif args.command == "fit":
        value = fit_final_threshold(train_path=args.train_csv, cache_path=args.oof_cache,
                                    freeze_path=args.freeze, run_dir=args.run_dir,
                                    e79_path=args.e79_cache, test_path=args.test_csv, sample_path=args.sample_csv,
                                    synthetic=args.synthetic_fixture)
    else:
        value = apply_frozen_threshold(test_path=args.test_csv, sample_path=args.sample_csv,
                                       e79_path=args.e79_cache, freeze_path=args.freeze,
                                       run_dir=args.run_dir, output_path=args.output_csv,
                                       train_path=args.train_csv, oof_path=args.oof_cache,
                                       synthetic=args.synthetic_fixture)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
