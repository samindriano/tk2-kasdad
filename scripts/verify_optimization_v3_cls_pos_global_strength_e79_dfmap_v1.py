"""Independent verifier for the one-shot E79 Train-only DF-map artifact."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts import verify_optimization_v3_cls_pos_global_strength_e79_v3 as r3_verifier


ROOT = Path(__file__).resolve().parents[1]
DFMAP_ID = "CLS_POS_GLOBAL_STRENGTH_E79_DF_MAP_V1"
FREEZE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DFMAP_FREEZE.json"
ATTESTATION_REL = r3_verifier.ATTESTATION_REL
R3_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/descendant_replay_v1/REPLAY_RECEIPT.json"
OUTPUT_ROOT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/df_map_v1"
MAP_REL = f"{OUTPUT_ROOT_REL}/E79_DF_MAP.npz"
ATTEMPT_REL = f"{OUTPUT_ROOT_REL}/ATTEMPT.started.json"
RECEIPT_REL = f"{OUTPUT_ROOT_REL}/DFMAP_RECEIPT.json"
TRAIN_REL = "data/raw/classification/Train.csv"
A02_MODEL_REL = "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/model.joblib"
A02_MODEL_SHA256 = "2791eddf410c7e81b119bde2a0b1d33a694eed4de84027c61cee97f6668dcb6e"
EXPECTED_ROWS = 50_608
EXPECTED_UNIGRAM_VOCABULARY_SIZE = 5_753
EXPECTED_TRAIN_COLUMNS = [
    "record_id", "cycle_code", "impact_load", "entry_friction", "scope_pressure",
    "text_load", "scope_trace", "org_spread", "module_spread", "telemetry_flag",
    "org_code", "module_code", "pattern_code", "signal_stream",
]
EXPECTED_POSITION = {
    "shape": [50_608, 17_259],
    "nnz": 1_638_979,
    "logical_sha256": "32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942",
}
CONTINUATION_FILE_PATHS = {
    ".gitattributes",
    "reports/optimization_v3/cls_pos_global_strength_e79_v3/DFMAP_CONTINUATION.md",
    "scripts/optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py",
    "scripts/verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py",
    "tests/test_optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py",
    "src/models/pos_one_sided_ridge_v1.py",
    "src/models/pos_frequency_ridge_v1.py",
}
EXPECTED_FORMULA = {
    "df": "count each retained unigram at most once per Train row if present in any positional segment block",
    "raw": "clip(sqrt(median(df)/df), 0.5, 2.0)",
    "aligned": "raw / mean(raw)",
    "permutation": "numpy.default_rng(42).permutation(V); permuted_aligned=aligned[permutation]",
    "global_strength": "v=max(1,aligned); c=mean(v); penalty=full(3*V,c)",
}


class VerificationError(RuntimeError):
    """Raised when the DF-map artifact or evidence is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_sha256(path: Path, relative: str) -> str:
    if Path(relative).suffix.lower() in {".md", ".json"}:
        return _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return _sha256_file(path)


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        raise VerificationError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _tree_sha256(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise VerificationError(f"Committed frozen path missing: {relative}")
    return _sha256_bytes(completed.stdout)


def _verify_path_manifest(root: Path, commit: str, manifest: dict[str, str]) -> None:
    for relative, expected in sorted(manifest.items()):
        path = root / relative
        if not path.is_file() or _stable_file_sha256(path, relative) != expected or _tree_sha256(root, commit, relative) != expected:
            raise VerificationError(f"Frozen file hash mismatch: {relative}")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def verify_npz_arrays(map_path: Path, *, expected_size: int, expected_rows: int,
                      statistics: dict[str, Any]) -> dict[str, Any]:
    """Independently check persisted arrays and the fixed recipe without source data."""
    with np.load(map_path, allow_pickle=False) as arrays:
        required = {
            "document_frequency_int64", "raw_clipped_weights_float64", "aligned_weights_float64",
            "permuted_aligned_weights_float64", "global_strength_position_penalty_float64",
            "global_strength_scalar_float64",
        }
        if set(arrays.files) != required:
            raise VerificationError("DF-map archive has missing or unexpected arrays")
        df = arrays["document_frequency_int64"]
        raw = arrays["raw_clipped_weights_float64"]
        aligned = arrays["aligned_weights_float64"]
        permuted = arrays["permuted_aligned_weights_float64"]
        penalty = arrays["global_strength_position_penalty_float64"]
        scalar_array = arrays["global_strength_scalar_float64"]
    if (
        df.dtype != np.dtype(np.int64) or df.shape != (expected_size,) or np.any(df <= 0)
        or np.any(df > expected_rows) or not np.isfinite(raw).all()
        or not np.isfinite(aligned).all() or not np.isfinite(permuted).all()
        or raw.dtype != np.dtype(np.float64) or aligned.dtype != np.dtype(np.float64)
        or permuted.dtype != np.dtype(np.float64) or penalty.dtype != np.dtype(np.float64)
        or raw.shape != (expected_size,) or aligned.shape != (expected_size,)
        or permuted.shape != (expected_size,) or penalty.shape != (3 * expected_size,)
        or scalar_array.shape != ()
    ):
        raise VerificationError("DF-map arrays have invalid shape, dtype, range, or finiteness")

    median_df = float(np.median(df))
    expected_raw = np.clip(np.sqrt(median_df / df.astype(np.float64)), 0.5, 2.0)
    expected_aligned = expected_raw / float(np.mean(expected_raw, dtype=np.float64))
    expected_permutation = np.random.default_rng(42).permutation(expected_size)
    expected_permuted = expected_aligned[expected_permutation]
    expected_scalar = float(np.mean(np.maximum(1.0, expected_aligned), dtype=np.float64))
    expected_penalty = np.full(3 * expected_size, expected_scalar, dtype=np.float64)
    if (
        not np.array_equal(raw, expected_raw)
        or not np.array_equal(aligned, expected_aligned)
        or not np.array_equal(permuted, expected_permuted)
        or float(scalar_array) != expected_scalar
        or not np.array_equal(penalty, expected_penalty)
    ):
        raise VerificationError("DF-map artifact does not match the frozen original formula")

    expected_statistics = {
        "retained_unigram_vocabulary_size": expected_size,
        "train_rows": expected_rows,
        "median_df": median_df,
        "df_min": int(df.min()),
        "df_max": int(df.max()),
        "aligned_mean": float(np.mean(expected_aligned, dtype=np.float64)),
        "aligned_min": float(expected_aligned.min()),
        "aligned_max": float(expected_aligned.max()),
        "raw_clip_low_count": int(np.count_nonzero(expected_raw <= 0.5)),
        "raw_clip_high_count": int(np.count_nonzero(expected_raw >= 2.0)),
        "permuted_aligned_sha256": _sha256_bytes(np.ascontiguousarray(expected_permuted, dtype="<f8").tobytes()),
        "global_strength_scalar": expected_scalar,
        "global_position_penalty_size": int(expected_penalty.size),
        "global_position_penalty_unique_values": int(np.unique(expected_penalty).size),
    }
    for key, expected_value in expected_statistics.items():
        actual_value = statistics.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (int, float)) or not np.isclose(actual_value, expected_value, rtol=0.0, atol=1e-14):
                raise VerificationError(f"DF-map statistics field differs: {key}")
        elif actual_value != expected_value:
            raise VerificationError(f"DF-map statistics field differs: {key}")
    return {
        "document_frequency": df,
        "aligned_weights": aligned,
        "global_strength_scalar": expected_scalar,
        "median_df": median_df,
        "df_min": int(df.min()),
        "df_max": int(df.max()),
    }


def verify_dfmap(root: Path = ROOT) -> dict[str, Any]:
    freeze_path = root / FREEZE_REL
    freeze = _json(freeze_path)
    freeze_body = dict(freeze)
    freeze_digest = freeze_body.pop("freeze_sha256", None)
    if freeze_digest != _canonical_sha256(freeze_body):
        raise VerificationError("DF-map freeze self-digest failed")
    if freeze.get("dfmap_id") != DFMAP_ID or freeze.get("status") != "FROZEN_FOR_ONE_TRAIN_ONLY_DF_MAP_FIT":
        raise VerificationError("Unexpected DF-map freeze identity or status")
    if freeze.get("formula") != EXPECTED_FORMULA:
        raise VerificationError("DF-map freeze formula differs from the original E79 recipe")
    head = _git(root, "rev-parse", "HEAD")
    prepared_head = freeze.get("prepared_from_head")
    ancestry = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", str(prepared_head), head],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if ancestry.returncode != 0:
        raise VerificationError("DF-map freeze source is not an ancestor of current HEAD")
    manifest = freeze.get("continuation_file_sha256")
    if not isinstance(manifest, dict) or set(manifest) != CONTINUATION_FILE_PATHS:
        raise VerificationError("DF-map freeze source manifest is incomplete")
    _verify_path_manifest(root, head, manifest)
    _verify_path_manifest(root, str(prepared_head), manifest)
    _verify_path_manifest(root, head, {FREEZE_REL: _stable_file_sha256(freeze_path, FREEZE_REL)})

    attestation_result = r3_verifier.verify_attestation(root)
    attestation = _json(root / ATTESTATION_REL)
    if (
        attestation_result.get("attestation_sha256") != freeze.get("r3_attestation_sha256")
        or _sha256_file(root / ATTESTATION_REL) != freeze.get("r3_attestation_file_sha256")
        or freeze.get("r3_current_source_sha256") != attestation.get("current_source_sha256")
        or freeze.get("immutable_input_sha256") != attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"]
    ):
        raise VerificationError("R3 frozen source/input manifest differs")

    r3_receipt_path = root / R3_RECEIPT_REL
    r3_result = r3_verifier.verify_replay_receipt(root, r3_receipt_path)
    r3_receipt = _json(r3_receipt_path)
    if (
        r3_result.get("status") != "PASS_INDEPENDENT_DATA_FREE_REPLAY_VERIFICATION"
        or _sha256_file(r3_receipt_path) != freeze.get("r3_receipt_file_sha256")
        or r3_receipt.get("receipt_sha256") != freeze.get("r3_receipt_sha256")
    ):
        raise VerificationError("Frozen prerequisite R3 receipt failed independent verification")

    train_path = root / TRAIN_REL
    if _sha256_file(train_path) != freeze.get("official_train_sha256"):
        raise VerificationError("Official Train bytes differ from DF-map freeze")
    if _sha256_file(root / A02_MODEL_REL) != A02_MODEL_SHA256:
        raise VerificationError("Saved A02 vectorizer model hash differs")

    attempt = _json(root / ATTEMPT_REL)
    if attempt.get("dfmap_id") != DFMAP_ID or attempt.get("no_retry_if_interrupted") is not True:
        raise VerificationError("DF-map attempt marker is invalid")
    receipt_path = root / RECEIPT_REL
    receipt = _json(receipt_path)
    receipt_body = dict(receipt)
    receipt_digest = receipt_body.pop("receipt_sha256", None)
    if receipt_digest != _canonical_sha256(receipt_body):
        raise VerificationError("DF-map receipt self-digest failed")
    if receipt.get("status") != "PASS_E79_DF_MAP_CREATED_TRAIN_ONLY" or receipt.get("dfmap_id") != DFMAP_ID:
        raise VerificationError(f"DF-map receipt status failed: {receipt.get('status')}")
    if (
        receipt.get("target_column_requested_or_loaded") is not False
        or receipt.get("test_or_sample_loaded") is not False
        or receipt.get("estimator_fit_calls") != 0
        or receipt.get("prediction_or_inference_calls") != 0
        or receipt.get("vectorizer_fit_calls") != 0
        or receipt.get("structured_preprocessor_fit_calls") != 0
        or receipt.get("newly_learned_preprocessing_fit_calls") != 1
        or receipt.get("df_map_formula_fit_calls") != 1
        or receipt.get("csv_created") is not False
        or receipt.get("kaggle_contacted") is not False
        or receipt.get("train_path") != TRAIN_REL
        or receipt.get("train_sha256") != freeze.get("official_train_sha256")
        or receipt.get("train_rows") != EXPECTED_ROWS
        or receipt.get("ordered_train_ids_sha256") != r3_receipt.get("ordered_train_ids_sha256")
        or receipt.get("r3_receipt_file_sha256") != _sha256_file(r3_receipt_path)
        or receipt.get("dfmap_freeze_sha256") != freeze_digest
    ):
        raise VerificationError("DF-map receipt violates frozen fit population or source boundaries")
    if receipt.get("position_matrix") != EXPECTED_POSITION:
        raise VerificationError("Rebuilt position matrix metadata differs from R3 frozen reference")
    if receipt.get("train_columns_requested") != EXPECTED_TRAIN_COLUMNS:
        raise VerificationError("DF-map feature read columns differ from the exact official feature allowlist")

    artifact_rel = receipt.get("artifact_path")
    if artifact_rel != MAP_REL:
        raise VerificationError("DF-map artifact path escaped its frozen output namespace")
    map_path = root / artifact_rel
    artifact_sha = _sha256_file(map_path)
    if artifact_sha != receipt.get("artifact_sha256"):
        raise VerificationError("DF-map compressed artifact file hash differs")
    size = int(receipt["statistics"]["retained_unigram_vocabulary_size"])
    if size != EXPECTED_UNIGRAM_VOCABULARY_SIZE:
        raise VerificationError("Retained unigram vocabulary size differs from frozen R3 geometry")
    array_result = verify_npz_arrays(
        map_path, expected_size=size, expected_rows=EXPECTED_ROWS,
        statistics=receipt.get("statistics", {}),
    )

    model = joblib.load(root / A02_MODEL_REL)
    vectorizer = model.text_space.vectorizer
    ordered_unigrams = [
        [term, int(index)]
        for term, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
        if " " not in term
    ]
    expected_order_hash = _canonical_sha256(ordered_unigrams)
    if expected_order_hash != receipt.get("unigram_vocabulary_order_sha256"):
        raise VerificationError("DF-map token ordering differs from saved A02 vectorizer order")
    if receipt.get("statistics", {}).get("retained_unigram_vocabulary_size") != len(ordered_unigrams):
        raise VerificationError("DF-map vocabulary size differs from saved A02 vectorizer")

    return {
        "status": "PASS_INDEPENDENT_E79_DF_MAP_VERIFICATION",
        "dfmap_id": DFMAP_ID,
        "freeze_sha256": freeze_digest,
        "r3_replay_receipt_sha256": r3_result["replay_receipt_sha256"],
        "dfmap_receipt_sha256": receipt_digest,
        "dfmap_receipt_file_sha256": _sha256_file(receipt_path),
        "dfmap_artifact_sha256": artifact_sha,
        "retained_unigram_vocabulary_size": size,
        "train_rows": EXPECTED_ROWS,
        "df_min": array_result["df_min"],
        "df_median": array_result["median_df"],
        "df_max": array_result["df_max"],
        "aligned_mean": float(array_result["aligned_weights"].mean()),
        "global_strength_scalar": array_result["global_strength_scalar"],
        "position_matrix": EXPECTED_POSITION,
        "target_test_sample_or_kaggle_used": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.parse_args()
    try:
        result = verify_dfmap()
    except (VerificationError, r3_verifier.VerificationError) as exc:
        print(json.dumps({"status": "FAIL_DF_MAP_VERIFICATION", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
