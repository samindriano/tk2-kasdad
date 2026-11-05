"""Independent read-only QA for the frozen NO_POS classification CSV.

This verifier deliberately does not import the candidate builder. It recomputes
the threshold route from the pre-existing E79 pA and availability vectors,
then checks only Test identifiers/availability and SampleSubmission schema.
It never reads a Test target or fits a model.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/optimization_v3/cls_simplification_benchmark_v1"
E79 = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
OUTPUT = ROOT / "submissions/optimization_v3/cls_simplification_benchmark_v1/classification_no_pos_e79.csv"
CHAMPION = ROOT / "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv"
HISTORICAL_A02 = ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv"
MANIFEST = REPORT_DIR / "RUN_MANIFEST.json"
QA_OUTPUT = REPORT_DIR / "INDEPENDENT_QA.json"
EXPECTED = {
    "e79": "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271",
    "test": "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c",
    "sample": "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7",
}
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_items(items: list[str]) -> str:
    digest = hashlib.sha256()
    for value in items:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    if QA_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {QA_OUTPUT}")
    for path in (E79, TEST, SAMPLE, OUTPUT, CHAMPION, HISTORICAL_A02, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {
        "e79_npz": sha256_file(E79),
        "official_test": sha256_file(TEST),
        "sample_submission": sha256_file(SAMPLE),
    }
    if input_hashes["e79_npz"] != EXPECTED["e79"]:
        raise ValueError("E79 archive hash differs from frozen value")
    if input_hashes["official_test"] != EXPECTED["test"]:
        raise ValueError("Official Test hash differs from frozen value")
    if input_hashes["sample_submission"] != EXPECTED["sample"]:
        raise ValueError("SampleSubmission hash differs from frozen value")

    with np.load(E79, allow_pickle=False) as bundle:
        ids = np.asarray(bundle["record_ids"], dtype="U")
        missing = np.asarray(bundle["missing_mask"], dtype=bool)
        probability = np.asarray(bundle["a02_probability"], dtype=np.float64)
    if not (ids.shape == missing.shape == probability.shape == (29_421,)):
        raise ValueError("Frozen E79 vectors do not all have 29,421 rows")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("E79 IDs contain duplicates")
    if not np.isfinite(probability).all():
        raise ValueError("E79 probability vector contains non-finite values")

    test = pd.read_csv(TEST, usecols=["record_id", "signal_stream"])
    test_ids = test["record_id"].astype(str).tolist()
    test_missing = test["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    sample = pd.read_csv(SAMPLE)
    sample_ids = sample["record_id"].astype(str).tolist()
    output = pd.read_csv(OUTPUT)
    output_ids = output["record_id"].astype(str).tolist()
    if not (ids.tolist() == test_ids == sample_ids == output_ids):
        raise ValueError("E79, Test, SampleSubmission, and output ID order differs")
    if not np.array_equal(missing, test_missing):
        raise ValueError("E79 availability differs from signal_stream in Test")
    if output.columns.tolist() != sample.columns.tolist():
        raise ValueError("Output columns differ from SampleSubmission")
    if len(output) != 29_421 or output["record_id"].nunique() != 29_421:
        raise ValueError("Output row count or ID uniqueness failed")
    labels = output["artifact_signal"].to_numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Output labels are not finite binary integers")
    if not np.equal(labels, labels.astype(np.int64)).all():
        raise ValueError("Output labels are not integer-valued")

    expected_labels = np.where(
        missing,
        probability >= MISSING_THRESHOLD,
        probability >= PRESENT_THRESHOLD,
    ).astype(np.int64)
    if not np.array_equal(labels.astype(np.int64), expected_labels):
        raise ValueError("Output labels differ from independently recomputed route")

    champion = pd.read_csv(CHAMPION)
    historic = pd.read_csv(HISTORICAL_A02)
    for name, frame in (("champion", champion), ("historical_a02", historic)):
        if frame.columns.tolist() != sample.columns.tolist() or frame["record_id"].astype(str).tolist() != ids.tolist():
            raise ValueError(f"{name} schema or ID order differs")
    champion_labels = champion["artifact_signal"].to_numpy(dtype=np.int64)
    historic_labels = historic["artifact_signal"].to_numpy(dtype=np.int64)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_manifest = manifest["csv"]
    output_hash = sha256_file(OUTPUT)
    label_hash = hashlib.sha256(expected_labels.astype(np.int8).tobytes()).hexdigest()
    if output_hash != expected_manifest["sha256"]:
        raise ValueError("Output CSV hash differs from run manifest")
    if label_hash != expected_manifest["label_vector_sha256"]:
        raise ValueError("Independent label vector hash differs from run manifest")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "verifier": "standalone; does not import candidate builder",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["source_commit"],
        "benchmark_id": manifest["benchmark_id"],
        "inputs": input_hashes,
        "test_fields_read": ["record_id", "signal_stream"],
        "test_target_read": False,
        "fits": 0,
        "new_model_inference": 0,
        "checks": {
            "row_count": 29_421,
            "unique_ids": int(len(set(output_ids))),
            "ordered_id_sha256": sha256_text_items(ids.tolist()),
            "sample_columns_match": True,
            "labels_integer_binary": True,
            "independent_route_replay": True,
            "availability_matches_test": True,
            "output_sha256": output_hash,
            "label_vector_sha256": label_hash,
            "label_counts": {
                "0": int(np.sum(expected_labels == 0)),
                "1": int(np.sum(expected_labels == 1)),
            },
            "changed_rows_vs_current_public_champion": int(np.sum(expected_labels != champion_labels)),
            "changed_rows_vs_historical_a02": int(np.sum(expected_labels != historic_labels)),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    QA_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
