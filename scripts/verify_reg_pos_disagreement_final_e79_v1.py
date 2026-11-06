"""Independent source and post-pass artifact verification for final E79 shrinkage."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_POS_DISAGREEMENT_FINAL_E79_V1"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
SOURCE_REPORT = REPORT_ROOT / "SOURCE_PREPARATION_R1.json"
SOURCE_QA = REPORT_ROOT / "SOURCE_QA_R3.json"
VERIFIER_AMENDMENT_R2 = REPORT_ROOT / "SOURCE_VERIFIER_AMENDMENT_R2.json"
VERIFIER_AMENDMENT = REPORT_ROOT / "SOURCE_VERIFIER_AMENDMENT_R3.json"
SOURCE_ARCHIVE = ARTIFACT_ROOT / "FINAL_E79_SOURCE_INPUTS_R1.npz"
FINAL_PASS = ROOT / "experiments" / LANE / "ROOT_FINAL_SOURCE_PASS_R1.json"
FINAL_RUN = ROOT / "experiments" / "runs" / LANE
FINAL_CSV = ROOT / "submissions" / "reg_pos_disagreement_final_e79_v1.csv"
RUNNER_PATH = ROOT / "scripts" / "reg_pos_disagreement_final_e79_v1.py"
PARENT_RUNNER = ROOT / "scripts" / "optimization_v3_reg_current_r06_e79_exploratory_v1.py"
HINGE_RUNNER = ROOT / "scripts" / "reg_monotone_hinge_p50_v1.py"
PILOT_REPORT = ROOT / "reports" / "optimization_v3" / "REG_POS_DISAGREEMENT_GATE_V1"
PILOT_ARTIFACT = ROOT / "experiments" / "runs" / "REG_POS_DISAGREEMENT_GATE_V1"
PARENT_MANIFEST = ROOT / "reports" / "optimization_v3" / "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1" / "INFERENCE_MANIFEST_B.json"
PARENT_ARCHIVE = ROOT / "experiments" / "runs" / "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1" / "inference_B.npz"
HINGE_STATE = ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_R1" / "fits" / "HINGE_FINAL.json"
HINGE_REPORT = ROOT / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_REPORT_R1.json"
CHAMPION_CSV = ROOT / "submissions" / "reg_monotone_hinge_p50_v1" / "hinge_p50_final_exploratory_e79.csv"
PILOT_SOURCE_R1 = PILOT_ARTIFACT / "SOURCE_FREEZE_R1.json"
PILOT_SOURCE_R2 = PILOT_ARTIFACT / "SOURCE_FREEZE_R2.json"
PILOT_PREDICTION_R2 = PILOT_ARTIFACT / "PREDICTION_FREEZE_R2.json"
PILOT_PRELABEL_R2 = PILOT_REPORT / "PRELABEL_QA_R2.json"
PILOT_SCORE_R2 = PILOT_REPORT / "SCORE_R2.json"
PILOT_METRIC_QA_R3 = PILOT_REPORT / "METRIC_QA_R3.json"
ROOT_HEADLINE = PILOT_REPORT / "ROOT_HEADLINE_REPLAY_R1.json"
LEDGER = PILOT_ARTIFACT / "lane_ledger.jsonl"
EXPECTED = {
    "parent_manifest": "d2bd7dea17feab1740ca4630bb5cf7ace62caf8746051b3322cc38d168674484",
    "parent_archive": "e29b1eabb1692729749b25d88b7cf86d1bdf44c34ce74ab964b124074f5817f2",
    "parent_raw_score": "4483e6bf2ff96e8baf65435c59d5eddb6348bf2e797a672afa38585e8a2ea338",
    "hinge_state": "a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a",
    "hinge_report": "fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73",
    "champion_csv": "a5813fdd384da31611db1e945c7db33199d8ef9044fb71dd2941ef7af6422da9",
    "root_headline": "b3e60d52b576577b3c18e48007e5d1e9b36f375f217947734fffc01c508d468c",
}
PARENT_ARCHIVE_MEMBERS = {"record_id", "cycle_code", "missing", "prediction", "baseline", "raw"}
PARENT_READ_MEMBERS = ("record_id", "cycle_code", "missing", "baseline", "raw")
OOF_EXPECTED_ROWS = 20903
E79_EXPECTED_ROWS = 29421


class VerifyError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_array(value: Any) -> str:
    array = np.asarray(value)
    header = _canonical({"dtype": array.dtype.str, "shape": list(array.shape)})
    if array.dtype.kind in "OUS":
        payload = bytearray()
        for item in array.reshape(-1).tolist():
            raw = str(item).encode("utf-8", errors="surrogatepass")
            payload.extend(len(raw).to_bytes(8, "little")); payload.extend(raw)
        body = bytes(payload)
    else:
        body = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(len(header).to_bytes(8, "little") + header + body).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerifyError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerifyError(f"Expected JSON object: {path}")
    return value


def _verify_signed(value: Mapping[str, Any], key: str, label: str) -> None:
    body = {name: item for name, item in value.items() if name != key}
    if value.get(key) != _sha_json(body):
        raise VerifyError(f"Signature mismatch: {label}")


def _read_npz(path: Path, members: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise VerifyError(f"Required NPZ missing: {path}")
    with zipfile.ZipFile(path) as archive:
        names = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
    if names != set(members):
        raise VerifyError(f"NPZ allowlist mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in members}


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise VerifyError(f"Required source module missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerifyError(f"Cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pilot_inputs() -> dict[str, Any]:
    files = {
        "pilot_source_r1": PILOT_SOURCE_R1,
        "pilot_source_r2": PILOT_SOURCE_R2,
        "pilot_prediction_freeze_r2": PILOT_PREDICTION_R2,
        "pilot_prelabel_r2": PILOT_PRELABEL_R2,
        "pilot_score_r2": PILOT_SCORE_R2,
        "pilot_metric_qa_r3": PILOT_METRIC_QA_R3,
        "pilot_verifier_amendment_r3": PILOT_REPORT / "METRIC_QA_VERIFIER_AMENDMENT_R3.json",
        "root_headline_replay_r1": ROOT_HEADLINE,
        "lane_ledger": LEDGER,
        "parent_manifest": PARENT_MANIFEST,
        "parent_archive": PARENT_ARCHIVE,
        "hinge_state": HINGE_STATE,
        "hinge_report": HINGE_REPORT,
        "champion_csv": CHAMPION_CSV,
    }
    hashes = {name: _sha_file(path) for name, path in files.items()}
    for name in ("parent_manifest", "parent_archive", "hinge_state", "hinge_report", "champion_csv", "root_headline_replay_r1"):
        if hashes[name] != EXPECTED[name if name != "root_headline_replay_r1" else "root_headline"]:
            raise VerifyError(f"Pinned input hash mismatch: {name}")
    source_r1, source_r2 = _read_json(PILOT_SOURCE_R1), _read_json(PILOT_SOURCE_R2)
    prediction, prelabel = _read_json(PILOT_PREDICTION_R2), _read_json(PILOT_PRELABEL_R2)
    score, metric = _read_json(PILOT_SCORE_R2), _read_json(PILOT_METRIC_QA_R3)
    amendment, headline = _read_json(files["pilot_verifier_amendment_r3"]), _read_json(ROOT_HEADLINE)
    for value, key, label in ((source_r1, "source_signature", "pilot source R1"),
                              (source_r2, "source_signature", "pilot source R2"),
                              (prediction, "prediction_signature", "pilot prediction freeze"),
                              (prelabel, "qa_signature", "pilot prelabel"),
                              (score, "score_signature", "pilot score"),
                              (metric, "qa_signature", "pilot metric QA R3"),
                              (amendment, "amendment_signature", "pilot verifier amendment")):
        _verify_signed(value, key, label)
    if (source_r2.get("status") != "SOURCE_AMENDED_R2"
            or source_r2.get("source_signature_r1") != source_r1.get("source_signature")
            or source_r2.get("source_freeze_r1_sha256") != hashes["pilot_source_r1"]
            or prediction.get("source_signature") != source_r2.get("source_signature")
            or prelabel.get("status") != "PASS" or prelabel.get("query_targets_read") is not False
            or prelabel.get("source_signature") != source_r2.get("source_signature")
            or prelabel.get("prediction_signature") != prediction.get("prediction_signature")
            or score.get("status") != "SCORED"
            or score.get("source_signature") != source_r2.get("source_signature")
            or score.get("prediction_signature") != prediction.get("prediction_signature")
            or score.get("prelabel_qa_sha256") != hashes["pilot_prelabel_r2"]
            or metric.get("status") != "PASS"
            or metric.get("source_signature") != source_r2.get("source_signature")
            or metric.get("prediction_signature") != prediction.get("prediction_signature")
            or metric.get("score_signature") != score.get("score_signature")
            or metric.get("target_values_persisted") is not False
            or metric.get("test_or_sample_accessed") is not False
            or metric.get("fixed_groups_known") is not True
            or metric.get("scientific_qualification") is not True
            or metric.get("exploratory_benchmark_eligible") is not True):
        raise VerifyError("Pilot frozen source/score/metric QA linkage failed")
    if (headline.get("status") != "PASS_ROOT_INDEPENDENT_HEADLINE_METRIC_REPLAY"
            or headline.get("score_sha256") != hashes["pilot_score_r2"]
            or headline.get("test_access") is not False
            or headline.get("new_fits") != 0
            or headline.get("new_candidate_policies") != 0
            or headline.get("pooled_REAL_MAE_gain") != metric.get("pooled_REAL_MAE_gain")
            or headline.get("pooled_REAL_over_CONTROL_gain") != metric.get("pooled_REAL_over_CONTROL_gain")):
        raise VerifyError("Root headline replay does not match metric QA R3")
    return {"hashes": hashes, "source": source_r2, "prediction": prediction,
            "score": score, "metric": metric, "headline": headline}


def _verify_source_report() -> dict[str, Any]:
    report = _read_json(SOURCE_REPORT)
    _verify_signed(report, "source_manifest_signature", "source preparation R1")
    if report.get("status") != "SOURCE_READY_PENDING_ROOT_SOURCE_PASS":
        raise VerifyError("Source preparation is not in the expected review state")
    deps = _pilot_inputs()
    if report.get("source_dependencies") != deps["hashes"]:
        raise VerifyError("Source preparation dependency hashes changed")
    code_paths = {
        "runner": RUNNER_PATH,
        "independent_verifier": Path(__file__),
        "source_regression_test": ROOT / "tests" / "test_reg_pos_disagreement_final_e79_source_v1.py",
        "pilot_runner": ROOT / "scripts" / "reg_pos_disagreement_gate_v1.py",
        "pilot_verifier": ROOT / "scripts" / "verify_reg_pos_disagreement_gate_v1.py",
        "parent_runner": PARENT_RUNNER,
        "hinge_runner": HINGE_RUNNER,
    }
    amendment_r2 = _read_json(VERIFIER_AMENDMENT_R2)
    _verify_signed(amendment_r2, "amendment_signature", "source verifier amendment R2")
    amendment = _read_json(VERIFIER_AMENDMENT)
    _verify_signed(amendment, "amendment_signature", "source verifier amendment R3")
    old_hash = report.get("source_code_hashes", {}).get("independent_verifier")
    if (amendment_r2.get("status") != "VERIFIER_AMENDED_R2"
            or amendment_r2.get("source_report_sha256") != _sha_file(SOURCE_REPORT)
            or amendment_r2.get("source_archive_sha256") != _sha_file(SOURCE_ARCHIVE)
            or amendment_r2.get("old_verifier_sha256") != old_hash
            or amendment_r2.get("expected_hash_fix") != {
                "field": "EXPECTED.hinge_state",
                "old_value": "a81b2e8e709a33e9e15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a",
                "new_value": "a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a",
            }
            or amendment.get("status") != "VERIFIER_AMENDED_R3"
            or amendment.get("source_report_sha256") != _sha_file(SOURCE_REPORT)
            or amendment.get("source_archive_sha256") != _sha_file(SOURCE_ARCHIVE)
            or amendment.get("previous_verifier_amendment_sha256") != _sha_file(VERIFIER_AMENDMENT_R2)
            or amendment.get("old_verifier_sha256") != amendment_r2.get("new_verifier_sha256")
            or amendment.get("new_verifier_sha256") != _sha_file(Path(__file__))
            or amendment.get("targeted_regression_test_sha256") != report.get("source_code_hashes", {}).get("source_regression_test")
            or amendment.get("source_order_check_fix") != {
                "removed_assumption": "oof_cycles are sorted by physical query row",
                "required_order": "oof_query_rows sorted and unique; cycle values aligned to their physical rows",
                "cycle_domain": [68, 79],
            }):
        raise VerifyError("Source verifier amendment chain does not bind the two exact QA corrections")
    active_hashes = {key: _sha_file(path) for key, path in code_paths.items()}
    active_hashes.pop("independent_verifier")
    frozen_hashes = dict(report.get("source_code_hashes", {}))
    frozen_hashes.pop("independent_verifier", None)
    if active_hashes != frozen_hashes:
        raise VerifyError("Final E79 source runner/verifier/test code changed after source freeze")
    archive_receipt = report.get("source_archive", {})
    if archive_receipt.get("path") != SOURCE_ARCHIVE.relative_to(ROOT).as_posix():
        raise VerifyError("Final source archive path differs from allowlist")
    source_arrays = _read_npz(SOURCE_ARCHIVE, tuple(archive_receipt.get("members", {})))
    if (_sha_file(SOURCE_ARCHIVE) != archive_receipt.get("sha256")
            or {key: _sha_array(value) for key, value in source_arrays.items()} != archive_receipt.get("members")):
        raise VerifyError("Final source archive file/member receipt mismatch")
    expected_members = {
        "oof_query_rows", "oof_query_ids", "oof_cycles", "oof_missing", "oof_baseline",
        "oof_pos_t50", "oof_disagreement", "e79_record_id", "e79_cycle_code", "e79_missing",
        "e79_baseline", "e79_raw_pos", "e79_pos_t50", "e79_champion_direct", "e79_champion_final",
    }
    if set(source_arrays) != expected_members:
        raise VerifyError("Final source archive member allowlist changed")
    oof_rows = source_arrays["oof_query_rows"].astype(np.int64)
    oof_ids = source_arrays["oof_query_ids"].astype(str)
    oof_cycles = source_arrays["oof_cycles"].astype(np.int64)
    oof_missing = source_arrays["oof_missing"].astype(bool)
    oof_baseline = source_arrays["oof_baseline"].astype(np.float64)
    oof_pos = source_arrays["oof_pos_t50"].astype(np.float64)
    oof_disagreement = source_arrays["oof_disagreement"].astype(np.float64)
    positive = (~oof_missing) & (oof_disagreement > 0.0)
    if (len(oof_rows) != OOF_EXPECTED_ROWS or len(np.unique(oof_rows)) != OOF_EXPECTED_ROWS
            or len(np.unique(oof_ids)) != OOF_EXPECTED_ROWS
            or not np.array_equal(oof_rows, np.sort(oof_rows))
            or not np.array_equal(np.unique(oof_cycles), np.arange(68, 80, dtype=np.int64))
            or not np.array_equal(oof_disagreement, np.abs(oof_pos - oof_baseline))
            or int(positive.sum()) != report["final_scale_source"]["positive_present_disagreements"]
            or int(positive.sum()) < 200):
        raise VerifyError("Final OOF scalar source identity or formula check failed")

    parent_manifest = _read_json(PARENT_MANIFEST)
    if (parent_manifest.get("raw_prediction_archive_sha256") != EXPECTED["parent_archive"]
            or parent_manifest.get("raw_score_sha256") != EXPECTED["parent_raw_score"]
            or parent_manifest.get("baseline_raw_candidate_member_used") is not False):
        raise VerifyError("Approved parent input pin semantics changed")
    with zipfile.ZipFile(PARENT_ARCHIVE) as parent_zip:
        parent_members = {name[:-4] for name in parent_zip.namelist() if name.endswith(".npy")}
    if parent_members != PARENT_ARCHIVE_MEMBERS:
        raise VerifyError("Parent archive member inventory differs")
    with np.load(PARENT_ARCHIVE, allow_pickle=False) as parent_zip:
        parent = {name: np.asarray(parent_zip[name]).copy() for name in PARENT_READ_MEMBERS}
    parent_runner = _load_module(PARENT_RUNNER, "reg_current_r06_e79_parent_for_final_qa")
    raw = parent["raw"].astype(np.float64)
    if parent_runner.sha_array(raw, dtype="<f8") != parent_manifest["raw_score_sha256"]:
        raise VerifyError("Parent raw POS digest does not use the producer's exact format")
    ids = parent["record_id"].astype(str)
    cycles = parent["cycle_code"].astype(np.int64)
    missing = parent["missing"].astype(bool)
    baseline = parent["baseline"].astype(np.float64)
    raw_t50 = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        raw_t50[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    direct = np.clip(0.75 * baseline + 0.25 * raw_t50, 0.0, 100.0)
    direct[missing] = baseline[missing]
    producer_direct = parent_runner._candidate_output_prediction("B", raw, cycles, missing, baseline)
    if not np.array_equal(direct, producer_direct):
        raise VerifyError("Original single-stream DIRECT arithmetic differs from its producer")
    hinge_state = _read_json(HINGE_STATE)
    solver = hinge_state.get("solver_result", {})
    params = {"intercept": solver.get("intercept"), "slope_low": solver.get("slope_low"),
              "slope_high": solver.get("slope_high"), "knot": 50.0}
    hinge = _load_module(HINGE_RUNNER, "reg_monotone_hinge_for_final_e79_qa")
    champion_final = hinge.apply_hinge_present_only(direct, missing, params)
    champion = pd.read_csv(CHAMPION_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    champion_ids = champion["record_id"].astype(str).to_numpy(dtype=str)
    champion_values = pd.to_numeric(champion["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.array_equal(ids, champion_ids)
            or not np.array_equal(champion_final.view("<u8"), champion_values.view("<u8"))):
        raise VerifyError("Original champion row order or final hinge bits differ")
    for key, expected in (("e79_record_id", ids), ("e79_cycle_code", cycles),
                          ("e79_missing", missing), ("e79_baseline", baseline),
                          ("e79_raw_pos", raw), ("e79_pos_t50", raw_t50),
                          ("e79_champion_direct", direct), ("e79_champion_final", champion_final)):
        if not np.array_equal(source_arrays[key], expected):
            raise VerifyError(f"E79 source package member differs from independent replay: {key}")

    ledger_events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line]
    policy = [item for item in ledger_events if item.get("operation_type") == "deterministic_target_free_policy_construction"]
    starts = [item for item in policy if item.get("event") == "OPERATION_STARTED"]
    saves = [item for item in policy if item.get("event") == "OPERATION_SAVED"]
    failures = [item for item in policy if item.get("event") == "OPERATION_FAILED"]
    if (len(starts) != 6 or len(saves) != 4 or len(failures) != 2
            or {item.get("operation_id") for item in failures} != {"REAL_D1", "CONTROL_D1"}):
        raise VerifyError("Pilot lane ledger policy budget/failure retention differs")
    metric = deps["metric"]
    if (metric.get("fixed_groups_known") is not True
            or metric.get("maximum_supported_REAL_group_harm") > 0.25
            or metric.get("scientific_qualification") is not True
            or metric.get("exploratory_benchmark_eligible") is not True):
        raise VerifyError("Supported-group pilot qualification is not PASS")
    boundary = report.get("execution_boundary", {})
    final_contract = report.get("final_candidate_contract", {})
    if (boundary.get("test_file_opened") is not False or boundary.get("sample_file_opened") is not False
            or boundary.get("test_labels_read") is not False or boundary.get("sample_rows_read") is not False
            or boundary.get("real_e79_candidate_application") is not False
            or final_contract.get("final_scale_state_fit_started") is not False
            or final_contract.get("real_e79_application_started") is not False
            or report.get("final_scale_source", {}).get("scale_s_computed") is not False):
        raise VerifyError("Source preparation crossed the final fit/application boundary")
    return {"report": report, "source_arrays": source_arrays, "deps": deps,
            "verifier_amendment": amendment,
            "champion_replay": {"ids_exact": True, "direct_exact": True, "hinge_bit_exact": True}}


def source_stage() -> dict[str, Any]:
    verified = _verify_source_report()
    report = verified["report"]
    body = {
        "schema_version": 1, "namespace": LANE, "status": "PASS",
        "source_manifest_signature": report["source_manifest_signature"],
        "verifier_amendment_sha256": _sha_file(VERIFIER_AMENDMENT),
        "active_verifier_sha256": _sha_file(Path(__file__)),
        "source_report_sha256": _sha_file(SOURCE_REPORT),
        "source_archive_sha256": _sha_file(SOURCE_ARCHIVE),
        "pilot_metric_qa_sha256": report["source_dependencies"]["pilot_metric_qa_r3"],
        "root_headline_sha256": report["source_dependencies"]["root_headline_replay_r1"],
        "prediction_freeze_sha256": report["source_dependencies"]["pilot_prediction_freeze_r2"],
        "oof_rows": OOF_EXPECTED_ROWS, "e79_parent_rows": E79_EXPECTED_ROWS,
        "parent_raw_digest_producer_format_pass": True,
        "parent_prediction_member_loaded": False,
        "champion_replay": verified["champion_replay"],
        "policy_ledger_pass": True,
        "all_supported_group_qa_pass": True,
        "pilot_scientific_qualification": True,
        "pilot_exploratory_eligible": True,
        "final_scale_state_fit_started": False,
        "real_e79_application_started": False,
        "test_labels_read": False, "sample_accessed": False,
        "target_values_persisted": False, "kaggle_or_upload": False,
    }
    result = dict(body)
    result["qa_signature"] = _sha_json(result)
    payload = _canonical(result) + b"\n"
    SOURCE_QA.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_QA.exists() and SOURCE_QA.read_bytes() != payload:
        raise VerifyError("Refusing to overwrite changed source QA report")
    if not SOURCE_QA.exists():
        SOURCE_QA.write_bytes(payload)
    return {"status": result["status"], "qa_signature": result["qa_signature"],
            "source_report_sha256": result["source_report_sha256"],
            "parent_raw_digest_producer_format_pass": True,
            "champion_replay_bit_exact": True, "real_e79_application_started": False}


def output_stage() -> dict[str, Any]:
    if not FINAL_PASS.is_file():
        raise VerifyError("Output verification requires the root-signed final source pass")
    approval = _read_json(FINAL_PASS)
    _verify_signed(approval, "approval_signature", "root final pass")
    manifest_path = FINAL_RUN / "FINAL_MANIFEST_R1.json"
    manifest = _read_json(manifest_path)
    _verify_signed(manifest, "manifest_signature", "final prediction manifest")
    if (manifest.get("status") != "CSV_READY_PENDING_INDEPENDENT_QA"
            or manifest.get("final_pass_sha256") != _sha_file(FINAL_PASS)
            or manifest.get("csv_path") != FINAL_CSV.relative_to(ROOT).as_posix()
            or _sha_file(FINAL_CSV) != manifest.get("csv_sha256")):
        raise VerifyError("Final E79 manifest/CSV binding failed")
    source_report = _read_json(SOURCE_REPORT)
    _verify_signed(source_report, "source_manifest_signature", "source preparation")
    archive_path = FINAL_RUN / "predictions" / "E79.npz"
    receipt_path = FINAL_RUN / "predictions" / "E79.json"
    receipt = _read_json(receipt_path)
    _verify_signed(receipt, "receipt_signature", "final prediction receipt")
    members = tuple(receipt.get("members", {}))
    arrays = _read_npz(archive_path, members)
    if (_sha_file(archive_path) != receipt.get("prediction_sha256")
            or {key: _sha_array(value) for key, value in arrays.items()} != receipt.get("members")):
        raise VerifyError("Final prediction archive/receipt readback failed")
    source_arrays = _read_npz(SOURCE_ARCHIVE, tuple(source_report["source_archive"]["members"]))
    scale = _read_json(FINAL_RUN / "final_scale_state_R1.json")
    _verify_signed(scale, "state_signature", "final scale state")
    positive = (~source_arrays["oof_missing"].astype(bool)) & (source_arrays["oof_disagreement"] > 0.0)
    scale_values = source_arrays["oof_disagreement"][positive]
    expected_s = float(np.median(scale_values))
    if scale.get("scale_s") != expected_s or scale.get("labels_used") is not False:
        raise VerifyError("Final target-free scalar state differs from its frozen source")
    baseline = arrays["baseline"].astype(np.float64)
    p = arrays["pos_t50"].astype(np.float64)
    missing = arrays["missing"].astype(bool)
    disagreement = np.abs(p - baseline)
    weight = 0.25 * np.exp(-disagreement / expected_s)
    direct = np.clip(baseline + weight * (p - baseline), 0.0, 100.0)
    direct[missing] = baseline[missing]
    hinge_state = _read_json(HINGE_STATE)
    solver = hinge_state["solver_result"]
    hinge = _load_module(HINGE_RUNNER, "reg_monotone_hinge_for_final_output_qa")
    final = hinge.apply_hinge_present_only(direct, missing, {
        "intercept": solver["intercept"], "slope_low": solver["slope_low"],
        "slope_high": solver["slope_high"], "knot": 50.0,
    })
    if (not np.array_equal(arrays["disagreement"], disagreement)
            or not np.array_equal(arrays["weight"], weight)
            or not np.array_equal(arrays["candidate_direct"], direct)
            or not np.array_equal(arrays["candidate_final"], final)
            or not np.array_equal(final[missing].view("<u8"), baseline[missing].view("<u8"))):
        raise VerifyError("Final candidate formula or missing fallback differs")
    csv = pd.read_csv(FINAL_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    ids = csv["record_id"].astype(str).to_numpy(dtype=str)
    values = pd.to_numeric(csv["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.array_equal(ids, arrays["record_id"].astype(str))
            or not np.array_equal(values, final)
            or manifest.get("scale_state_fits") != 1
            or manifest.get("candidate_applications") != 1
            or manifest.get("predictor_fits") != 0
            or manifest.get("preprocessor_fits") != 0
            or manifest.get("hinge_fits") != 0
            or manifest.get("test_labels_read") is not False
            or manifest.get("sample_accessed") is not False):
        raise VerifyError("Final E79 CSV or one-fit/one-application budget failed")
    return {"status": "PASS", "csv_sha256": _sha_file(FINAL_CSV),
            "rows": int(len(ids)), "scale_state_fits": 1,
            "candidate_applications": 1, "test_labels_read": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", action="store_true")
    group.add_argument("--output", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = source_stage() if args.source else output_stage()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
