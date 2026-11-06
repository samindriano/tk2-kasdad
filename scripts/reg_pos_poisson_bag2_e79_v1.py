"""Source-only preparation and gated E79 inference for two POS Poisson replicas.

The default ``--source-only`` mode validates frozen source artifacts without
reading official Train/Test feature rows or starting an official fit. It also
runs a complete, isolated synthetic lifecycle. ``--execute`` is blocked until
an exact root-signed final source pass binds the source preparation report.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

NAMESPACE = "REG_POS_POISSON_BAG2_E79_V1"
PARENT = "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
PARENT_RUN = "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
PARENT_MANIFEST = ROOT / PARENT / "INFERENCE_MANIFEST_B.json"
PARENT_ARCHIVE = ROOT / PARENT_RUN / "inference_B.npz"
TRAINED_FREEZE = ROOT / PARENT / "TRAINED_STATE_FREEZE_B.json"
PRELABEL_PASS = ROOT / PARENT / "PRELABEL_PASS_B.json"
PARENT_CSV_QA = ROOT / PARENT / "CSV_QA_B.json"
STATE_BUNDLE = ROOT / PARENT_RUN / "state_B.joblib"
HINGE_STATE = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fits/HINGE_FINAL.json"
HINGE_REPORT = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_REPORT_R1.json"
CHAMPION_CSV = ROOT / "submissions/reg_monotone_hinge_p50_v1/hinge_p50_final_exploratory_e79.csv"
R06_BASELINE_CSV = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\submissions\optimization_v3_expanded_v1\r06_original.csv")
SCORE_REPLAY = ROOT / "reports/optimization_v3/reg_pos_poisson_bag2_v1/ROOT_INDEPENDENT_SCORE_REPLAY_R1.json"
SCORE_REPORT = ROOT / "reports/optimization_v3/reg_pos_poisson_bag2_v1/SCORE_R1.json"
POS_FREEZE = ROOT / "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1.json"
SOURCE_REPORT = ROOT / "reports/optimization_v3/REG_POS_POISSON_BAG2_E79_V1/SOURCE_PREPARATION_R3.json"
SUPERSEDED_SOURCE_REPORT_R1_SHA256 = "5f07200715f08733abd2cd810ca9cf5dcf6d01efcc68c84f2fe02b2eb95b8fd2"
SUPERSEDED_SOURCE_REPORT_R2_SHA256 = "a60d6989b213196fb21dcb5eb0d08d5ac3f598bb3bb521f2198afdc38dcb68de"
FINAL_PASS = ROOT / "experiments/reg_pos_poisson_bag2_e79_v1/ROOT_FINAL_SOURCE_PASS_R1.json"
RUN_ROOT = ROOT / "experiments/runs/REG_POS_POISSON_BAG2_E79_V1"
CSV_PATH = ROOT / "submissions/reg_pos_poisson_bag2_e79_v1/pos_poisson_bag2_final_e79.csv"
FIT_IDS = (
    "REG_POS_POISSON_BAG2_E79_V1_P17",
    "REG_POS_POISSON_BAG2_E79_V1_P137",
)
SEEDS = (17, 137)
EXPECTED = {
    "parent_manifest": "d2bd7dea17feab1740ca4630bb5cf7ace62caf8746051b3322cc38d168674484",
    "parent_archive": "e29b1eabb1692729749b25d88b7cf86d1bdf44c34ce74ab964b124074f5817f2",
    "trained_freeze": "3364e75a8c791cebbd9fd4e2693149fa1429e7e5ef19f68bb1cadb6828b51218",
    "prelabel_pass": "a4771eacfe808e5645977008827ebc9552c288dbed5910606649e3d4670716c3",
    "parent_csv_qa": "bc19efcbbf57ae0e621fe1b338f00e1726380ed9a736947f6bf760b8a9fa67c8",
    "hinge_state": "a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a",
    "hinge_report": "fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73",
    "champion_csv": "a5813fdd384da31611db1e945c7db33199d8ef9044fb71dd2941ef7af6422da9",
    "score_replay": "a3f1ede82846f39e57d9ecf9254f64990c631cc65e5713c0cedad1df54f5fb34",
    "score_report": "e1b6260bb6a9868a98f7633b47ce12ce6e17637dc933e45461943103d31dcc59",
    "pos_freeze": "ba347046ba25d2f0a4fc1ec2ef35dd38f69bef0999ed1b3d75856a6b149a41e8",
    "pos_freeze_signature": "aeacf1bf9d9b9c20af4823a3dcb65e7329d38f62c052127ed58f6637d2f80b83",
}
FINAL_HINGE = {
    "intercept": 1.7288079036674198,
    "slope_low": 0.9933285599956824,
    "slope_high": 0.957778072288735,
    "knot": 50.0,
}
SCORE_INPUT_SHA = {
    "67": "b996d09f18446eff7f8723c76b5d44d83e0a6d684672c98d38839a2ea81e2643",
    "75": "8c950bd8fd27b00f89c71a4e87eed0e0270cf385b9aece9cca33caeee91040e6",
}


class SourceError(RuntimeError):
    """Fail-closed source or authorization error."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    header = _json_bytes({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header)
    if array.dtype.kind in "OUS":
        digest.update(_json_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _sha_csr(value: Any) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(_json_bytes({"shape": matrix.shape, "dtype": str(matrix.dtype)}))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceError(f"Unable to read JSON authority: {path}") from exc
    if not isinstance(value, dict):
        raise SourceError(f"Expected JSON object: {path}")
    return value


def _verify_signed(path: Path, signature_key: str = "signature") -> dict[str, Any]:
    value = _read_json(path)
    body = {k: v for k, v in value.items() if k not in {signature_key, "signature_method"}}
    if (value.get("signature_method") != "SHA-256 canonical JSON"
            or value.get(signature_key) != _canonical_sha(body)):
        raise SourceError(f"Canonical signature mismatch: {path}")
    return value


def _verify_hashed(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise SourceError(f"Missing frozen input: {label} ({path})")
    actual = _sha_file(path)
    if actual != expected:
        raise SourceError(f"Frozen input hash mismatch: {label}")
    return actual


def _verify_manifest_sources(manifest: Mapping[str, Any]) -> dict[str, str]:
    files: dict[str, str] = {}
    for rel, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / Path(rel)
        files[rel] = _verify_hashed(path, str(expected), f"source:{rel}")
    if not files:
        raise SourceError("Parent manifest has no source lineage")
    return files


def _check_source_contract(*, run_synthetic: bool) -> dict[str, Any]:
    file_hashes = {
        "parent_manifest": _verify_hashed(PARENT_MANIFEST, EXPECTED["parent_manifest"], "R06 B inference manifest"),
        "parent_archive": _verify_hashed(PARENT_ARCHIVE, EXPECTED["parent_archive"], "R06 B inference archive"),
        "trained_freeze": _verify_hashed(TRAINED_FREEZE, EXPECTED["trained_freeze"], "R06 B trained-state freeze"),
        "prelabel_pass": _verify_hashed(PRELABEL_PASS, EXPECTED["prelabel_pass"], "R06 B prelabel pass"),
        "parent_csv_qa": _verify_hashed(PARENT_CSV_QA, EXPECTED["parent_csv_qa"], "R06 B CSV QA"),
        "hinge_state": _verify_hashed(HINGE_STATE, EXPECTED["hinge_state"], "final hinge state"),
        "hinge_report": _verify_hashed(HINGE_REPORT, EXPECTED["hinge_report"], "final hinge report"),
        "champion_csv": _verify_hashed(CHAMPION_CSV, EXPECTED["champion_csv"], "frozen final champion CSV"),
        "score_replay": _verify_hashed(SCORE_REPLAY, EXPECTED["score_replay"], "independent score replay"),
        "score_report": _verify_hashed(SCORE_REPORT, EXPECTED["score_report"], "official Poisson score report"),
        "pos_freeze": _verify_hashed(POS_FREEZE, EXPECTED["pos_freeze"], "POS Poisson fit/parameter freeze"),
    }
    parent = _verify_signed(PARENT_MANIFEST)
    trained = _verify_signed(TRAINED_FREEZE)
    prelabel = _verify_signed(PRELABEL_PASS)
    qa = _verify_signed(PARENT_CSV_QA)
    pos_freeze = _read_json(POS_FREEZE)
    pos_body = {k: v for k, v in pos_freeze.items() if k != "freeze_signature"}
    if (pos_freeze.get("freeze_signature") != _canonical_sha(pos_body)
            or pos_freeze.get("freeze_signature") != EXPECTED["pos_freeze_signature"]):
        raise SourceError("POS Poisson source freeze signature mismatch")

    if (parent.get("namespace") != "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
            or parent.get("candidate") != "B"
            or parent.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or parent.get("baseline_member_used") != "manifest-bound final mapped R06 prediction"
            or parent.get("baseline_raw_candidate_member_used") is not False
            or parent.get("raw_prediction_archive") != PARENT_ARCHIVE.relative_to(ROOT).as_posix()
            or parent.get("raw_prediction_archive_sha256") != EXPECTED["parent_archive"]
            or parent.get("test_labels_loaded") is not False
            or parent.get("upload_performed") is not False):
        raise SourceError("R06 B parent semantics or label/upload boundary differs")
    if trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY" or trained.get("candidate") != "B":
        raise SourceError("R06 B trained state is not the expected frozen candidate")
    if (trained.get("candidate_state_path") != STATE_BUNDLE.relative_to(ROOT).as_posix()
            or trained.get("candidate_state_sha256") != _sha_file(STATE_BUNDLE)
            or trained.get("train_rows") != 50608
            or trained.get("train_cycles") != [47, 79]
            or trained.get("test_features_loaded") is not False
            or trained.get("test_labels_loaded") is not False):
        raise SourceError("R06 B state path, fit population, or prelabel boundary differs")
    if prelabel.get("status") != "PRELABEL_PASS" or prelabel.get("candidate") != "B":
        raise SourceError("R06 B prelabel verification is not PASS")
    if (qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or qa.get("csv_sha256") != parent.get("csv_sha256")
            or qa.get("candidate") != "B"):
        raise SourceError("R06 B independent CSV QA does not bind the exact parent")
    parent_candidate_csv = ROOT / Path(parent["csv_path"])
    _verify_hashed(parent_candidate_csv, str(parent.get("csv_sha256")), "R06 B candidate CSV")
    baseline_csv_hash = str(parent.get("baseline_csv_sha256"))
    if (parent.get("input_sha256", {}).get("current_r06_e79_final_csv") != baseline_csv_hash
            or _sha_file(R06_BASELINE_CSV) != baseline_csv_hash):
        raise SourceError("Parent final mapped R06 four-learner baseline identity differs")

    state_fit = next((r for r in trained.get("fit_receipts", [])
                      if r.get("fit_id") == "B_DIRECT_POSITION_REAL_REGRESSOR"), None)
    if state_fit is None:
        raise SourceError("Frozen B predictor fit receipt is absent")
    unit_hashes: dict[str, str] = {}
    for receipt in trained.get("fit_receipts", []):
        rel = receipt.get("artifact_path")
        if not rel or not receipt.get("artifact_sha256"):
            raise SourceError("R06 B fit receipt lacks an artifact identity")
        unit_hashes[rel] = _verify_hashed(ROOT / Path(rel), receipt["artifact_sha256"], f"R06 B unit:{rel}")

    bundle = joblib.load(STATE_BUNDLE)
    if (bundle.get("candidate") != "B"
            or bundle.get("model_id") != "B_DIRECT_POSITION_REAL_REGRESSOR"
            or not isinstance(bundle.get("transform"), dict)
            or not {"vectorizer", "processor", "scaler", "encoder", "numeric_positions"}.issubset(bundle["transform"])):
        raise SourceError("Saved B model id or cached feature transform differs")
    model_path = ROOT / Path(state_fit["artifact_path"])
    parent_model = joblib.load(model_path)
    model_params = parent_model.get_params(deep=False)
    frozen_params = pos_freeze.get("resolved_parameters")
    receipt_params = state_fit.get("details", {}).get("parameters")
    if (not isinstance(frozen_params, dict) or receipt_params != frozen_params
            or model_params != frozen_params):
        raise SourceError("Frozen POS parameters differ from the saved R06 B learner")

    source_hashes = _verify_manifest_sources(parent)
    trained_source_hashes = _verify_manifest_sources(trained)
    if source_hashes != trained_source_hashes:
        raise SourceError("R06 B parent and trained-state source maps differ")
    hinge = _read_json(HINGE_STATE)
    solver = hinge.get("solver_result", {})
    actual_hinge = {"intercept": solver.get("intercept"), "slope_low": solver.get("slope_low"),
                    "slope_high": solver.get("slope_high"), "knot": solver.get("knot")}
    if hinge.get("status") != "LP_COMPLETE" or actual_hinge != FINAL_HINGE:
        raise SourceError("Frozen final hinge coefficients differ from the root contract")
    hinge_report = _read_json(HINGE_REPORT)
    if (hinge_report.get("csv_sha256") != EXPECTED["champion_csv"]
            or _sha_file(CHAMPION_CSV) != EXPECTED["champion_csv"]):
        raise SourceError("Final champion CSV is not the report-bound baseline")

    score_replay = _read_json(SCORE_REPLAY)
    score = _read_json(SCORE_REPORT)
    if (score_replay.get("status") != "PASS_ROOT_INDEPENDENT_SCORE_REPLAY"
            or score_replay.get("score_sha256") != EXPECTED["score_report"]
            or score_replay.get("exploratory_eligible") is not True
            or score_replay.get("scientific_qualified") is not False
            or score_replay.get("new_fits") != 0
            or score_replay.get("test_access") is not False
            or score.get("exploratory_benchmark_eligible") is not True
            or score.get("scientific_qualification") is not False):
        raise SourceError("Exploratory-only source eligibility evidence is absent or inconsistent")
    for origin, expected in SCORE_INPUT_SHA.items():
        if score.get("windows", {}).get(origin, {}).get("score_input_sha256") != expected:
            raise SourceError(f"E{origin} score input hash is not frozen")

    if RUN_ROOT.exists() or CSV_PATH.exists():
        raise SourceError("E79 final output or candidate CSV already exists")
    if run_synthetic and FINAL_PASS.exists():
        raise SourceError("Source-only preparation cannot run after final root pass issuance")

    synthetic = _synthetic_e2e() if run_synthetic else None
    return {
        "inputs": file_hashes,
        "parent_source_hashes": source_hashes,
        "parent_fit_artifact_hashes": unit_hashes,
        "parent_manifest_status": parent["status"],
        "parent_csv_qa_status": qa["status"],
        "saved_model_id": bundle["model_id"],
        "saved_transform_keys": sorted(bundle["transform"]),
        "pos_fit_parameters": frozen_params,
        "official_train_rows": 50608,
        "official_train_cycles": [47, 79],
        "pos_replica_seeds": list(SEEDS),
        "additional_preprocessing_fits_authorized": 0,
        "additional_predictor_fits_authorized": 2,
        "full_champion_csv_sha256": EXPECTED["champion_csv"],
        "final_hinge_parameters": FINAL_HINGE,
        "score_status": {
            "pooled_mae_gain": score_replay["pooled_gain"],
            "E67_gain": score_replay["windows"]["67"]["gain"],
            "E75_gain": score_replay["windows"]["75"]["gain"],
            "exploratory_eligible": True,
            "scientific_qualified": False,
        },
        "synthetic_e2e": synthetic,
        "official_fit_units_started": 0,
        "official_test_features_loaded": False,
        "official_test_labels_loaded": False,
        "official_candidate_csv_created": False,
    }


def _weight_draw(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.random.default_rng(seed).poisson(1.0, size=n).astype(np.uint8)
    total = int(counts.sum())
    if total <= 0:
        raise SourceError(f"Poisson bootstrap has zero total counts for seed {seed}")
    weights = counts.astype(np.float64) * (n / float(total))
    if (len(weights) != n or not np.isfinite(weights).all()
            or abs(float(weights.sum()) - n) > 1e-8 * n):
        raise SourceError("Normalized Poisson bootstrap weights failed")
    return counts, weights


def _rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    if raw.ndim != 1 or cycles.ndim != 1 or len(raw) != len(cycles):
        raise SourceError("Raw score and cycle vectors are misaligned")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        mapped[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return mapped


def _direct_candidate(mean_raw: np.ndarray, baseline: np.ndarray, cycles: np.ndarray,
                      missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not (mean_raw.shape == baseline.shape == cycles.shape == missing.shape):
        raise SourceError("E79 candidate vectors are misaligned")
    # The direct T50 blend intentionally has no pre-blend clipping.
    t50 = 0.5 * mean_raw + 0.5 * _rank50(mean_raw, cycles)
    direct = np.clip(0.75 * baseline + 0.25 * t50, 0.0, 100.0)
    direct = direct.astype(np.float64, copy=False)
    direct[np.asarray(missing, dtype=bool)] = baseline[np.asarray(missing, dtype=bool)]
    return t50, direct


def _csv_bytes(ids: np.ndarray, predictions: np.ndarray) -> bytes:
    if ids.ndim != 1 or predictions.ndim != 1 or len(ids) != len(predictions):
        raise SourceError("CSV row IDs and predictions are misaligned")
    frame = pd.DataFrame({"record_id": ids.astype(str), "threat_rank": predictions.astype(np.float64)})
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _read_csv_round_trip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={"record_id": "string"}, float_precision="round_trip")
    if list(frame.columns) != ["record_id", "threat_rank"]:
        raise SourceError(f"Unexpected output CSV columns: {path}")
    return (frame["record_id"].astype(str).to_numpy(dtype=str),
            pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64))


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SourceError(f"Refusing to overwrite existing output: {path}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_new(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_new(path, buffer.getvalue())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    body = dict(value)
    body["receipt_sha256"] = _canonical_sha(body)
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         indent=2).encode("utf-8") + b"\n"
    _write_new(path, payload)


def _verify_receipt(path: Path, signature_key: str = "receipt_sha256") -> dict[str, Any]:
    value = _read_json(path)
    body = {k: v for k, v in value.items() if k != signature_key}
    if value.get(signature_key) != _canonical_sha(body):
        raise SourceError(f"Canonical receipt signature mismatch: {path}")
    return value


def _apply_hinge(direct: np.ndarray, missing: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    from scripts.reg_monotone_hinge_p50_v1 import apply_hinge_present_only
    return apply_hinge_present_only(np.asarray(direct, dtype=np.float64), missing, params)


def _synthetic_e2e() -> dict[str, Any]:
    """Run a real miniature feature-to-fit-to-receipt-to-CSV flow in temp storage."""
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(79137)
    x_train = rng.normal(size=(96, 7)).astype(np.float64)
    y_train = np.clip(50.0 + 13.0 * x_train[:, 0] - 7.0 * x_train[:, 2], 0.0, 100.0)
    x_query = rng.normal(size=(32, 7)).astype(np.float64)
    x_train_sparse = sparse.csr_matrix(x_train)
    x_query_sparse = sparse.csr_matrix(x_query)
    params = {
        "boosting_type": "gbdt", "deterministic": True, "force_col_wise": True,
        "learning_rate": 0.05, "max_depth": -1, "min_child_samples": 4,
        "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 8,
        "n_jobs": 2, "num_leaves": 7, "objective": "regression_l1",
        "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 3.0,
        "subsample": 1.0, "subsample_for_bin": 2000, "subsample_freq": 0,
        "verbosity": -1,
    }
    ids = np.asarray([f"SYN-{i:04d}" for i in range(len(x_query))], dtype=str)
    cycles = np.repeat(np.asarray([80, 81], dtype=np.int64), 16)
    missing = (np.arange(len(ids)) % 9) == 0
    baseline = np.linspace(4.0, 96.0, len(ids), dtype=np.float64)
    original_raw = np.linspace(8.0, 92.0, len(ids), dtype=np.float64) + 0.17 * x_query[:, 0]
    original_direct = _direct_candidate(original_raw, baseline, cycles, missing)[1]
    synthetic_champion = _apply_hinge(original_direct, missing, FINAL_HINGE)

    with tempfile.TemporaryDirectory(prefix="reg-pos-e79-synthetic-") as temp_name:
        temp = Path(temp_name)
        unit_dir = temp / "units"
        unit_dir.mkdir(parents=True)
        fit_events: list[dict[str, Any]] = []
        member_raw: dict[int, np.ndarray] = {}
        for seed, fit_id in zip(SEEDS, FIT_IDS):
            counts, weights = _weight_draw(len(y_train), seed)
            model = LGBMRegressor(**params)
            fit_events.append({"event": "FIT_STARTED", "fit_id": fit_id, "seed": seed,
                               "feature_sha256": _sha_csr(x_train_sparse), "synthetic": True})
            with threadpool_limits(limits=2):
                model.fit(x_train_sparse, y_train, sample_weight=weights)
            artifact = unit_dir / f"{fit_id}.joblib"
            fd = os.open(artifact, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                joblib.dump(model, stream, compress=0)
                stream.flush()
                os.fsync(stream.fileno())
            receipt_path = unit_dir / f"{fit_id}.json"
            receipt = {
                "schema_version": 1, "namespace": NAMESPACE, "fit_id": fit_id,
                "seed": seed, "synthetic": True, "fit_rows": int(len(y_train)),
                "feature_count": int(x_train_sparse.shape[1]),
                "fit_design_sha256": _sha_csr(x_train_sparse),
                "fit_target_sha256": _sha_array(y_train, dtype="<f8"),
                "poisson_counts_sha256": _sha_array(counts),
                "sample_weight_sha256": _sha_array(weights, dtype="<f8"),
                "sample_weight_sum": float(weights.sum()),
                "parameters": params,
                "underlying_fit_calls": ["LGBMRegressor.fit x1 with sample_weight"],
                "artifact_sha256": _sha_file(artifact),
            }
            _write_json_new(receipt_path, receipt)
            with threadpool_limits(limits=2):
                member_raw[seed] = np.asarray(model.predict(x_query_sparse), dtype=np.float64)
            fit_events.append({"event": "FIT_SAVED", "fit_id": fit_id, "seed": seed,
                               "artifact_sha256": _sha_file(artifact),
                               "receipt_sha256": _sha_file(receipt_path), "synthetic": True})

        if len([e for e in fit_events if e["event"] == "FIT_STARTED"]) != 2:
            raise SourceError("Synthetic fit lifecycle did not start exactly two units")
        for seed, fit_id in zip(SEEDS, FIT_IDS):
            artifact = unit_dir / f"{fit_id}.joblib"
            receipt = _verify_receipt(unit_dir / f"{fit_id}.json")
            if receipt.get("artifact_sha256") != _sha_file(artifact):
                raise SourceError("Synthetic fit receipt failed artifact hash readback")
            restored = joblib.load(artifact)
            with threadpool_limits(limits=2):
                readback = np.asarray(restored.predict(x_query_sparse), dtype=np.float64)
            if not np.array_equal(readback, member_raw[seed]):
                raise SourceError("Synthetic fit model readback predictions differ")

        mean_raw = (original_raw + member_raw[17] + member_raw[137]) / 3.0
        t50, direct = _direct_candidate(mean_raw, baseline, cycles, missing)
        final = _apply_hinge(direct, missing, FINAL_HINGE)
        if not np.array_equal(final[missing].view("<u8"), baseline[missing].view("<u8")):
            raise SourceError("Synthetic final hinge changed a missing fallback")
        arrays = {
            "record_id": ids, "cycle_code": cycles, "missing": missing,
            "baseline": baseline, "original_raw": original_raw,
            "parent_direct_prediction": original_direct,
            "replica17_raw": member_raw[17], "replica137_raw": member_raw[137],
            "mean_raw": mean_raw, "mapped_t50": t50, "direct_prediction": direct,
            "final_prediction": final,
        }
        archive_path = temp / "prediction.npz"
        _write_npz_new(archive_path, **arrays)
        receipt_body = {"schema_version": 1, "synthetic": True,
                        "archive_sha256": _sha_file(archive_path),
                        "members": {k: _sha_array(v) for k, v in arrays.items()},
                        "fit_ids": list(FIT_IDS), "fit_units_started": 2,
                        "fit_units_saved": 2, "test_labels_loaded": False}
        receipt_path = temp / "prediction.json"
        _write_json_new(receipt_path, receipt_body)
        receipt_read = _verify_receipt(receipt_path)
        if receipt_read.get("archive_sha256") != _sha_file(archive_path):
            raise SourceError("Synthetic prediction receipt archive readback failed")
        with np.load(archive_path, allow_pickle=False) as saved:
            restored_arrays = {key: saved[key] for key in saved.files}
        if set(restored_arrays) != set(arrays):
            raise SourceError("Synthetic prediction archive member set differs")
        for key, expected in arrays.items():
            if not np.array_equal(restored_arrays[key], expected):
                raise SourceError(f"Synthetic prediction archive readback differs: {key}")
            if receipt_read["members"].get(key) != _sha_array(restored_arrays[key]):
                raise SourceError(f"Synthetic prediction member receipt differs: {key}")
        replay_parent_direct = _direct_candidate(restored_arrays["original_raw"],
                                                 restored_arrays["baseline"],
                                                 restored_arrays["cycle_code"],
                                                 restored_arrays["missing"])[1]
        if not np.array_equal(replay_parent_direct, restored_arrays["parent_direct_prediction"]):
            raise SourceError("Synthetic original direct replay differs from frozen parent prediction")

        champion_path = temp / "champion.csv"
        _write_new(champion_path, _csv_bytes(ids, synthetic_champion))
        champion_ids, champion_values = _read_csv_round_trip(champion_path)
        if not np.array_equal(champion_ids, ids) or not np.array_equal(champion_values, synthetic_champion):
            raise SourceError("Synthetic champion CSV parity/readback failed")
        # Recompute candidate application from archive readback before writing its CSV.
        readback_mean = (restored_arrays["original_raw"] + restored_arrays["replica17_raw"]
                         + restored_arrays["replica137_raw"]) / 3.0
        _, readback_direct = _direct_candidate(readback_mean, restored_arrays["baseline"],
                                              restored_arrays["cycle_code"], restored_arrays["missing"])
        readback_final = _apply_hinge(readback_direct, restored_arrays["missing"], FINAL_HINGE)
        if not np.array_equal(readback_final, restored_arrays["final_prediction"]):
            raise SourceError("Synthetic application replay differs from saved final prediction")
        candidate_path = temp / "candidate.csv"
        _write_new(candidate_path, _csv_bytes(ids, readback_final))
        candidate_ids, candidate_values = _read_csv_round_trip(candidate_path)
        if not np.array_equal(candidate_ids, ids) or not np.array_equal(candidate_values, readback_final):
            raise SourceError("Synthetic candidate CSV parity/readback failed")

        return {
            "status": "PASS_SYNTHETIC_FULL_CLI",
            "features_to_fit_artifact": True,
            "two_fit_artifacts": 2,
            "fit_receipts_written_and_read_back": True,
            "prediction_archive_receipt_and_member_readback": True,
            "original_direct_replay": True,
            "final_application_readback": True,
            "champion_csv_parity": True,
            "candidate_csv_round_trip": True,
            "synthetic_candidate_csv_count": 1,
            "official_fit_units_started": 0,
            "official_test_features_loaded": False,
            "official_test_labels_loaded": False,
        }


def _write_source_report(contract: Mapping[str, Any]) -> dict[str, Any]:
    script_sha = _sha_file(Path(__file__))
    body = {
        "schema_version": 1,
        "namespace": NAMESPACE,
        "status": "SOURCE_READY_PENDING_ROOT_FINAL_SOURCE_PASS",
        "source_script": Path(__file__).relative_to(ROOT).as_posix(),
        "source_script_sha256": script_sha,
        "supersedes_source_preparation_r1_sha256": SUPERSEDED_SOURCE_REPORT_R1_SHA256,
        "supersedes_source_preparation_r2_sha256": SUPERSEDED_SOURCE_REPORT_R2_SHA256,
        "parent_namespace": "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1",
        "parent_candidate_id": "DIRECT_POSITION_REAL_E79",
        "parent_baseline_member_used": "manifest-bound final mapped R06 four-learner prediction",
        "parent_baseline_raw_candidate_member_used": False,
        "source_inputs": contract["inputs"],
        "parent_source_hashes": contract["parent_source_hashes"],
        "parent_fit_artifact_hashes": contract["parent_fit_artifact_hashes"],
        "pos_model_id": contract["saved_model_id"],
        "cached_transform_keys": contract["saved_transform_keys"],
        "official_train_population": {
            "rows": contract["official_train_rows"], "cycles": contract["official_train_cycles"],
            "training_csv_sha256": "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082",
        },
        "replica_plan": {
            "fit_ids": list(FIT_IDS), "poisson_seeds": list(SEEDS),
            "resolved_parameters": contract["pos_fit_parameters"],
            "preprocessing_fits": 0, "predictor_fit_cap": 2,
        },
        "candidate_formula": {
            "mean_raw": "(parent.original_POS_raw + replica17_raw + replica137_raw) / 3",
            "T50": "0.5 * mean_raw + 0.5 * 100 * average_rank(mean_raw) / cycle_rows",
            "direct_present": "clip(0.75 * frozen_R06_four_learner_baseline + 0.25 * T50, 0, 100)",
            "missing": "frozen_R06_four_learner_baseline",
            "final_hinge": contract["final_hinge_parameters"],
            "final_hinge_application": "frozen final hinge on present rows; missing rows retain baseline bits",
        },
        "champion_parity_gate": {
            "path": CHAMPION_CSV.relative_to(ROOT).as_posix(),
            "sha256": contract["full_champion_csv_sha256"],
            "hinge_input": "reconstructed original full DIRECT from parent raw POS scores plus frozen R06 four-learner baseline",
            "required_before_candidate_csv": True,
            "row_ids_and_float_values": "exact round-trip comparison",
        },
        "score_basis": {"independent_replay_sha256": EXPECTED["score_replay"], **contract["score_status"]},
        "synthetic_e2e": contract["synthetic_e2e"],
        "execution_boundary": {
            "final_source_pass_required": FINAL_PASS.relative_to(ROOT).as_posix(),
            "final_fits_started": 0,
            "test_features_loaded": False,
            "test_labels_loaded": False,
            "sample_loaded": False,
            "candidate_csv_created": False,
            "kaggle_uploaded": False,
            "final_output_csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
            "final_output_archive_path": (RUN_ROOT / "predictions" / "E79.npz").relative_to(ROOT).as_posix(),
        },
    }
    body["source_preparation_sha256"] = _canonical_sha(body)
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         indent=2).encode("utf-8") + b"\n"
    SOURCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_REPORT.exists():
        if SOURCE_REPORT.read_bytes() != payload:
            raise SourceError("Source preparation report exists with different bytes; no overwrite")
    else:
        _write_new(SOURCE_REPORT, payload)
    return body


def _validate_final_pass() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _check_source_contract(run_synthetic=False)
    if not SOURCE_REPORT.is_file():
        raise SourceError("SOURCE_PREPARATION_R1.json is absent; --source-only must pass first")
    source_report = _read_json(SOURCE_REPORT)
    source_body = {k: v for k, v in source_report.items() if k != "source_preparation_sha256"}
    if source_report.get("source_preparation_sha256") != _canonical_sha(source_body):
        raise SourceError("Source preparation report signature mismatch")
    if source_report.get("source_script_sha256") != _sha_file(Path(__file__)):
        raise SourceError("Source script changed after source pass preparation")
    if not FINAL_PASS.is_file():
        raise SourceError("Final root source pass is absent; no final fit/Test/CSV action is authorized")
    approval = _read_json(FINAL_PASS)
    approval_body = {k: v for k, v in approval.items() if k != "approval_sha256"}
    if approval.get("approval_sha256") != _canonical_sha(approval_body):
        raise SourceError("Final root source pass signature mismatch")
    scope = approval.get("scope", {})
    if (approval.get("status") != "PASS"
            or approval.get("execute_final_pipeline") is not True
            or approval.get("source_preparation_sha256") != source_report["source_preparation_sha256"]
            or approval.get("source_report_sha256") != _sha_file(SOURCE_REPORT)
            or approval.get("source_script_sha256") != _sha_file(Path(__file__))
            or scope.get("fit_ids") != list(FIT_IDS)
            or scope.get("additional_predictor_fit_cap") != 2
            or scope.get("preprocessing_fits") != 0
            or scope.get("test_labels_access_allowed") is not False
            or scope.get("csv_path") != CSV_PATH.relative_to(ROOT).as_posix()
            or scope.get("upload_allowed") is not False):
        raise SourceError("Final root source pass does not bind this exact two-fit E79 scope")
    if RUN_ROOT.exists() or CSV_PATH.exists():
        raise SourceError("Final E79 output path exists; retries/overwrites are prohibited")
    return contract, approval


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fit_replica(fit_id: str, seed: int, design: Any, target: np.ndarray,
                 ids: np.ndarray, params: Mapping[str, Any], run_dir: Path) -> tuple[Any, dict[str, Any]]:
    from lightgbm import LGBMRegressor

    if fit_id not in FIT_IDS or seed not in SEEDS:
        raise SourceError("Unexpected E79 fit identity")
    ledger = run_dir / "fit_ledger.jsonl"
    events = []
    if ledger.exists():
        events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    if any(e.get("fit_id") == fit_id for e in events):
        raise SourceError(f"Fit already started or saved; retry forbidden: {fit_id}")
    prior_starts = [e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"]
    if prior_starts != list(FIT_IDS[:len(prior_starts)]) or len(prior_starts) >= 2:
        raise SourceError("Fit ledger order/cap violation")
    counts, weights = _weight_draw(len(target), seed)
    artifact = run_dir / "units" / f"{fit_id}.joblib"
    receipt_path = run_dir / "units" / f"{fit_id}.json"
    if artifact.exists() or receipt_path.exists():
        raise SourceError(f"Partial fit artifact exists; retry forbidden: {fit_id}")
    if not np.isfinite(target).all() or design.shape[0] != len(target) or len(ids) != len(target):
        raise SourceError("E79 training design, targets, and IDs are misaligned")
    import psutil
    ram_before = int(psutil.virtual_memory().available)
    if ram_before < 512 * 1024**2:
        raise SourceError("E79 fit blocked by the 512 MiB available-RAM floor")
    _append_jsonl(ledger, {"event": "FIT_STARTED", "fit_id": fit_id, "seed": seed,
                          "namespace": NAMESPACE, "fit_design_sha256": _sha_csr(design),
                          "fit_ids_sha256": _sha_array(ids), "target_sha256": _sha_array(target, dtype="<f8"),
                          "poisson_counts_sha256": _sha_array(counts), "synthetic": False})
    try:
        model = LGBMRegressor(**dict(params))
        with threadpool_limits(limits=2):
            model.fit(design, target, sample_weight=weights)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(artifact, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            joblib.dump(model, stream, compress=0)
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "schema_version": 1, "namespace": NAMESPACE, "fit_id": fit_id, "seed": seed,
            "synthetic": False, "train_rows": int(len(target)), "fit_feature_count": int(design.shape[1]),
            "fit_design_sha256": _sha_csr(design), "train_ids_sha256": _sha_array(ids),
            "fit_target_sha256": _sha_array(target, dtype="<f8"),
            "poisson_counts_sha256": _sha_array(counts),
            "sample_weight_sha256": _sha_array(weights, dtype="<f8"),
            "sample_weight_sum": float(weights.sum()), "parameters": dict(params),
            "underlying_fit_calls": ["LGBMRegressor.fit x1 with sample_weight"],
            "preprocessing_fit_calls": 0, "artifact_sha256": _sha_file(artifact),
            "available_ram_before_bytes": ram_before,
        }
        _write_json_new(receipt_path, receipt)
        _append_jsonl(ledger, {"event": "FIT_SAVED", "fit_id": fit_id, "seed": seed,
                              "artifact_sha256": _sha_file(artifact),
                              "receipt_sha256": _sha_file(receipt_path), "synthetic": False})
        return model, _read_json(receipt_path)
    except Exception as exc:
        _append_jsonl(ledger, {"event": "FIT_FAILED", "fit_id": fit_id, "seed": seed,
                              "exception_type": type(exc).__name__, "retry_performed": False,
                              "synthetic": False})
        raise


def _execute_final() -> dict[str, Any]:
    contract, approval = _validate_final_pass()
    import scripts.optimization_v3_reg_current_r06_e79_exploratory_v1 as parent_runner

    # This is the first official Train/Test feature access in this lane.
    train_path = parent_runner.TRAIN_PATH
    train_sha = _sha_file(train_path)
    parent_manifest = _read_json(PARENT_MANIFEST)
    expected_train_sha = parent_manifest.get("input_sha256", {}).get("official_regression_train")
    if (train_sha != expected_train_sha
            or train_sha != "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"):
        raise SourceError("Official Train bytes differ from the frozen 50,608-row source")

    trained = _verify_signed(TRAINED_FREEZE)
    state_bundle = joblib.load(STATE_BUNDLE)
    model_receipt = next(r for r in trained["fit_receipts"]
                         if r.get("fit_id") == "B_DIRECT_POSITION_REAL_REGRESSOR")
    original_model_path = ROOT / Path(model_receipt["artifact_path"])
    original_model = joblib.load(original_model_path)
    transform = state_bundle["transform"]

    # Reconstruct the original R06 B direct array and final champion before any new replica fit.
    test_columns = ["record_id", "cycle_code", "signal_stream", "impact_load", "entry_friction",
                    "scope_pressure", "text_load", "scope_trace", "org_spread", "module_spread",
                    "telemetry_flag", "org_code", "module_code", "pattern_code"]
    test = pd.read_csv(parent_runner.TEST_PATH, usecols=test_columns, dtype={"record_id": "string"})
    sample = pd.read_csv(parent_runner.SAMPLE_PATH, usecols=["record_id"], dtype={"record_id": "string"})
    anchor = pd.read_csv(parent_runner.ANCHOR_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    champion = pd.read_csv(CHAMPION_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    test_ids = test["record_id"].astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(test["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    baseline_ids = anchor["record_id"].astype(str).to_numpy(dtype=str)
    champion_ids = champion["record_id"].astype(str).to_numpy(dtype=str)
    if (len(test) != 29421 or not np.array_equal(test_ids, sample["record_id"].astype(str).to_numpy(dtype=str))
            or not np.array_equal(test_ids, baseline_ids) or not np.array_equal(test_ids, champion_ids)
            or not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64))):
        raise SourceError("Official Test, Sample, R06 baseline, or champion IDs/cycles differ")
    baseline = pd.to_numeric(anchor["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    champion_values = pd.to_numeric(champion["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    test_design, missing = parent_runner._infer_design("B", test, transform)
    with threadpool_limits(limits=2):
        original_raw = np.asarray(original_model.predict(test_design), dtype=np.float64)
    with np.load(PARENT_ARCHIVE, allow_pickle=False) as saved:
        archive = {key: saved[key] for key in saved.files}
    if (not np.array_equal(archive["record_id"].astype(str), test_ids)
            or not np.array_equal(archive["cycle_code"], cycles)
            or not np.array_equal(archive["missing"].astype(bool), missing)
            or not np.array_equal(archive["baseline"].astype(np.float64), baseline)
            or not np.array_equal(original_raw, archive["raw"].astype(np.float64))
            or _sha_array(original_raw, dtype="<f8") != parent_manifest.get("raw_score_sha256")):
        raise SourceError("Cached-transform original POS raw replay differs from frozen R06 E79 parent")
    original_direct = parent_runner._candidate_output_prediction("B", original_raw, cycles, missing, baseline)
    if not np.array_equal(original_direct, archive["prediction"].astype(np.float64)):
        raise SourceError("Reconstructed original R06 DIRECT differs from the frozen parent prediction")
    frozen_champion = _apply_hinge(original_direct, missing, FINAL_HINGE)
    if (not np.array_equal(frozen_champion, champion_values)
            or not np.array_equal(frozen_champion.view("<u8"), champion_values.view("<u8"))):
        raise SourceError("Original final champion hinge parity failed; no replicas were fitted")
    original_checks = {
        "parent_raw_sha256": _sha_array(original_raw, dtype="<f8"),
        "parent_direct_sha256": _sha_array(original_direct, dtype="<f8"),
        "champion_ids_sha256": _sha_array(champion_ids),
        "champion_values_sha256": _sha_array(champion_values, dtype="<f8"),
        "raw_replay_bit_exact": True,
        "original_direct_replay_bit_exact": True,
        "champion_hinge_input": "reconstructed original full DIRECT from parent archive raw plus frozen R06 four-learner baseline",
        "champion_final_hinge_parity_bit_exact": True,
    }

    # Only the already fitted B feature transform is reused; this lane performs no preprocessing fit.
    train, target = parent_runner._load_training_frame()
    train_design, train_missing = parent_runner._infer_design("B", train, transform)
    train_cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    train_ids = train["record_id"].astype(str).to_numpy(dtype=str)
    if (len(train) != 50608 or len(target) != 50608
            or not np.array_equal(np.unique(train_cycles), np.arange(47, 80, dtype=np.int64))
            or not np.isfinite(target).all()):
        raise SourceError("Official Train fit population differs from the frozen 50,608 rows")
    del train_missing

    params = contract["pos_fit_parameters"]
    if RUN_ROOT.exists():
        raise SourceError("E79 run directory exists before fit starts")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    models: dict[int, Any] = {}
    receipts: dict[str, Any] = {}
    for seed, fit_id in zip(SEEDS, FIT_IDS):
        model, receipt = _fit_replica(fit_id, seed, train_design, target, train_ids, params, RUN_ROOT)
        models[seed] = model
        receipts[fit_id] = receipt
    starts = [json.loads(line) for line in (RUN_ROOT / "fit_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = [e for e in starts if e.get("event") == "FIT_STARTED"]
    if [e.get("fit_id") for e in starts] != list(FIT_IDS):
        raise SourceError("Official final fit ledger does not contain exactly the approved two fit units")

    raw_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        with threadpool_limits(limits=2):
            raw_by_seed[seed] = np.asarray(models[seed].predict(test_design), dtype=np.float64)
        if raw_by_seed[seed].shape != original_raw.shape or not np.isfinite(raw_by_seed[seed]).all():
            raise SourceError(f"E79 replica output is invalid for seed {seed}")
    mean_raw = (original_raw + raw_by_seed[17] + raw_by_seed[137]) / 3.0
    t50, direct = _direct_candidate(mean_raw, baseline, cycles, missing)
    final = _apply_hinge(direct, missing, FINAL_HINGE)
    if (not np.isfinite(final).all() or np.any((final < 0.0) | (final > 100.0))
            or not np.array_equal(final[missing].view("<u8"), baseline[missing].view("<u8"))):
        raise SourceError("Final E79 predictions violate range or missing fallback")

    prediction_arrays = {
        "record_id": test_ids, "cycle_code": cycles, "missing": missing.astype(bool),
        "baseline": baseline, "original_raw": original_raw,
        "replica17_raw": raw_by_seed[17], "replica137_raw": raw_by_seed[137],
        "mean_raw": mean_raw, "mapped_t50": t50,
        "direct_prediction": direct, "final_prediction": final,
    }
    prediction_path = RUN_ROOT / "predictions" / "E79.npz"
    _write_npz_new(prediction_path, **prediction_arrays)
    prediction_receipt = {
        "schema_version": 1, "namespace": NAMESPACE, "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha_file(prediction_path),
        "members": {k: _sha_array(v) for k, v in prediction_arrays.items()},
        "source_preparation_sha256": _read_json(SOURCE_REPORT)["source_preparation_sha256"],
        "final_source_pass_sha256": _sha_file(FINAL_PASS),
        "fit_ids": list(FIT_IDS), "fit_units_started": 2, "fit_units_saved": 2,
        "test_labels_loaded": False, "sample_ids_used_for_order_check": True,
        "missing_fallback_bit_exact": True,
    }
    _write_json_new(RUN_ROOT / "predictions" / "E79.json", prediction_receipt)
    # Read back the archive and receipt, then replay the complete candidate application.
    saved_receipt = _verify_receipt(RUN_ROOT / "predictions" / "E79.json")
    if saved_receipt.get("prediction_sha256") != _sha_file(prediction_path):
        raise SourceError("Final E79 prediction receipt readback failed")
    with np.load(prediction_path, allow_pickle=False) as saved:
        readback = {key: saved[key] for key in saved.files}
    for key, expected in prediction_arrays.items():
        if not np.array_equal(readback[key], expected) or saved_receipt["members"].get(key) != _sha_array(readback[key]):
            raise SourceError(f"Final E79 prediction member readback failed: {key}")
    readback_mean = (readback["original_raw"] + readback["replica17_raw"] + readback["replica137_raw"]) / 3.0
    _, readback_direct = _direct_candidate(readback_mean, readback["baseline"], readback["cycle_code"], readback["missing"])
    readback_final = _apply_hinge(readback_direct, readback["missing"], FINAL_HINGE)
    if not np.array_equal(readback_final, readback["final_prediction"]):
        raise SourceError("Final E79 hinge application does not replay from saved raw members")
    if CSV_PATH.exists():
        raise SourceError("Final candidate CSV path exists; overwrite forbidden")
    _write_new(CSV_PATH, _csv_bytes(test_ids, readback_final))
    csv_ids, csv_values = _read_csv_round_trip(CSV_PATH)
    if not np.array_equal(csv_ids, test_ids) or not np.array_equal(csv_values, readback_final):
        raise SourceError("Final E79 CSV readback/row/value parity failed")

    manifest = {
        "schema_version": 1, "namespace": NAMESPACE,
        "status": "CSV_READY_PENDING_INDEPENDENT_QA",
        "source_preparation_sha256": _read_json(SOURCE_REPORT)["source_preparation_sha256"],
        "final_source_pass_sha256": _sha_file(FINAL_PASS),
        "parent_manifest_sha256": EXPECTED["parent_manifest"],
        "parent_archive_sha256": EXPECTED["parent_archive"],
        "training_rows": int(len(target)), "training_cycles": [47, 79],
        "fit_ids": list(FIT_IDS), "predictor_fits_started": 2,
        "preprocessing_fits": 0, "test_features_loaded": True,
        "test_labels_loaded": False, "sample_labels_loaded": False,
        "kaggle_accessed": False, "upload_performed": False,
        "baseline_member_used": "manifest-bound final mapped R06 four-learner prediction",
        "baseline_raw_candidate_member_used": False,
        "original_replay_checks": original_checks,
        "fit_receipts": receipts,
        "prediction_archive": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha_file(prediction_path),
        "prediction_receipt_sha256": _sha_file(RUN_ROOT / "predictions" / "E79.json"),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(), "csv_sha256": _sha_file(CSV_PATH),
        "csv_rows": int(len(csv_ids)), "csv_ids_sha256": _sha_array(csv_ids),
        "csv_values_sha256": _sha_array(csv_values, dtype="<f8"),
        "champion_parity_passed_before_candidate_csv": True,
        "candidate_csv_round_trip_passed": True,
        "final_hinge": FINAL_HINGE,
        "scientific_status": "EXPLORATORY_ONLY_NO_E79_LABELS_ACCESSED",
    }
    manifest_path = RUN_ROOT / "FINAL_MANIFEST_R1.json"
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    _write_new(manifest_path, json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False, indent=2).encode("utf-8") + b"\n")
    return {"status": manifest["status"], "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": _sha_file(manifest_path), "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
            "csv_sha256": _sha_file(CSV_PATH), "fit_ids": list(FIT_IDS),
            "test_labels_loaded": False, "upload_performed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-only", action="store_true")
    group.add_argument("--synthetic-e2e", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.synthetic_e2e:
        print(json.dumps(_synthetic_e2e(), sort_keys=True))
        return 0
    if args.source_only:
        contract = _check_source_contract(run_synthetic=True)
        report = _write_source_report(contract)
        print(json.dumps({"status": report["status"],
                          "source_report": SOURCE_REPORT.relative_to(ROOT).as_posix(),
                          "source_report_sha256": _sha_file(SOURCE_REPORT),
                          "source_preparation_sha256": report["source_preparation_sha256"],
                          "synthetic_e2e": report["synthetic_e2e"],
                          "official_fit_units_started": 0,
                          "official_test_features_loaded": False,
                          "official_candidate_csv_created": False}, sort_keys=True))
        return 0
    print(json.dumps(_execute_final(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
