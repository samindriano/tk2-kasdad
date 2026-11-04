"""Compatibility checks for the legacy R06 raw-byte prediction digest."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


SOURCE_ROOT = r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
WORKTREE_ROOT = r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier"
ORIGINAL_COMMIT = "0ca0255f85f8c8248c1aa4fa369456ed9b638180"
EXPECTED_PREFLIGHT_SHA256 = "6f2604a1813d65d5a684df549253a0ac9f56acadc286f9dc63e29fcb88114562"
EXPECTED_FREEZE_SHA256 = "fc44d92aaf9285af3680082649153ac88f77c917e282a9883ceb70a1770f8027"
EXPECTED_FREEZE_SIGNATURE = "03977900247e92e3a715985e37a6097d5425febda4e6d7ab0fc204c26018dce5"
EXPECTED_LEGACY_PREDICTION_SHA256 = "eb032101ec3b363e5e2f12d7f34f5b272b7e39cb9cad8fca001350384b857c3e"
EXPECTED_STRUCTURED_PREDICTION_SHA256 = "c9e2f8b5abb4a68acc8845191d3e550cb6787c22cce09070def1ed8bea2f25b6"

EXPECTED_ORIGINAL_WORKTREE_SOURCE_HASHES = {
    "reports/optimization_v3/reg_position_real_e79_benchmark_v1/BENCHMARK_AUTHORIZATION.md": "117e55a50f2135fffc9b0a4f9573e6e9a9d05d086d682487f66ceea19b88db81",
    "scripts/reg_position_real_e79_benchmark_v1.py": "6c32309c3d88720212592f851d9120b5bb389e681f7bfb33361cbdd998fd6aeb",
    "scripts/verify_reg_position_real_e79_benchmark_v1.py": "f690f760276f10382db8298c8667043a4b8c83470beba254715c2c3d7becf15a",
    "src/models/classification_models.py": "6fa123f5da93c2d1d1fc2d2eae0a04ead2988848123ece9b206e97ae5021c646",
    "src/models/optimization_v3_positional_text_pilot_v1.py": "390d9585a91440cc83314d2fa58acfebf374371032308937489019c2bd242c36",
    "src/models/reg_position_l1_v1.py": "d340217ed2527209407f0b97e8ac25de1c83c9e3f702fca60ecdefb9d5379302",
    "src/models/regression_research_tabular.py": "674d7e52c485583729b8fe999dc133c39aae8dd85abd005a598e2caea9af99f9",
}

EXPECTED_FIT_ARTIFACTS = {
    "WORD_TFIDF_FULL_TRAIN": {
        "artifact_sha256": "458d7929f644f8250a263061d1f2ee12ac336074dfd50984ddc6a42a9749f552",
        "receipt_sha256": "d1f47c5c39d7fbbcbdc181c2d7750813fc4059da439701142235f3f6e3adb14e",
        "receipt_file_sha256": "d244878470fbd1c04beb218c5511c9511ed1f8e1d821e7b56ee026ebe1a8ce2e",
    },
    "REAL_POSITION_RESIDUAL_L1_E79": {
        "artifact_sha256": "5cd67bd85a3bec8493fe274f5336f1bc739f92e11b224ac7cdee54a194f092ac",
        "receipt_sha256": "bb230851ade898795e799cd8fcf41c9e0cf63364c8a542912a3321e796a5dd0d",
        "receipt_file_sha256": "81613bfd0671a8bb52d6de9378c61dadf3ec3306571edd28156359e4c71404f2",
    },
}

EXPECTED_R06_FILE_HASHES = {
    "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json": "3f77e4ac148ada5fb79e5e2c16e7975e77726f4bfcc42a29f9234f801c697d67",
    "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz": "ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba",
    "submissions/optimization_v3_expanded_v1/r06_original.csv": "4a72b13e0d6ac93e18adee23010bb15ddf15fcb2a9543dc903154d5af60ee64f",
}


def legacy_raw_le_f8_sha256(values: Any) -> str:
    """Match the original producer: SHA256(final.astype('<f8').tobytes())."""
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1:
        raise ValueError("Legacy prediction digest requires a one-dimensional array")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def structured_f8_sha256(values: Any) -> str:
    """Match the newer structured array digest, including shape and dtype."""
    array = np.asarray(values, dtype="<f8")
    header = json.dumps(
        {"shape": array.shape, "dtype": str(array.dtype)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def validate_legacy_prediction_manifest_digest(manifest_sha256: Any, values: Any) -> dict[str, Any]:
    """Validate the old manifest field under its producer's raw-byte contract."""
    raw_sha = legacy_raw_le_f8_sha256(values)
    structured_sha = structured_f8_sha256(values)
    checks = {
        "manifest_digest_is_string": isinstance(manifest_sha256, str),
        "manifest_matches_legacy_raw_le_f8": manifest_sha256 == raw_sha,
        "legacy_raw_differs_from_structured_digest": raw_sha != structured_sha,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("Legacy R06 prediction digest validation failed: " + ", ".join(failed))
    return {
        "status": "PASS_LEGACY_RAW_LE_F8_CONTRACT",
        "manifest_prediction_sha256": manifest_sha256,
        "legacy_raw_le_f8_sha256": raw_sha,
        "structured_array_sha256": structured_sha,
        "manifest_contract": "sha256(final.astype('<f8').tobytes())",
        "structured_contract": "sha256(canonical({shape,dtype}) + contiguous_array_bytes)",
        "checks": checks,
    }
