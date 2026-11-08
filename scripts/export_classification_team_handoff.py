"""Export/replay saved Samuel champion probabilities; never fits or uploads.

The bundled validation predictions are temporal model predictions, but the
operational missing threshold was chosen after viewing development outcomes.
Use probabilities with correctly aligned teammate OOF, not these hard labels
as an independent validation result. Raw Train labels are never exported.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "handoff/classification_team_20261108"
MISSING_THRESHOLD = 0.53230976129477
SOURCES = {
    "validation": ("experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz", "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"),
    "test": ("experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz", "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"),
}
BEST_CSV = ROOT / "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv"
BEST_SHA = "189a779b6100333a2fecfe443689fe7d340a331e828167221c81281a7ece4a4a"
TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
TEST_SHA = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy(p_pos, p_a02, missing):
    p_pos, p_a02 = np.asarray(p_pos, float), np.asarray(p_a02, float)
    missing = np.asarray(missing, bool)
    if p_pos.ndim != 1 or p_pos.shape != p_a02.shape or missing.shape != p_pos.shape:
        raise ValueError("Probability/missing vectors must have the same 1D shape")
    if not all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in [p_pos, p_a02]):
        raise ValueError("Invalid probability domain")
    blend = 0.5 * p_pos + 0.5 * p_a02
    probability = np.where(missing, p_a02, blend)
    threshold = np.where(missing, MISSING_THRESHOLD, 0.5)
    return probability, threshold, (probability >= threshold).astype(np.int8)


def frame(ids, cycles, missing, pp, pa, origin, cutoff, rows):
    p, threshold, label = policy(pp, pa, missing)
    result = pd.DataFrame({"origin": origin, "fit_cycle_start": 47,
        "fit_cycle_end": cutoff, "record_id": np.asarray(ids).astype(str),
        "source_row": rows, "cycle_code": cycles, "stream_missing": np.asarray(missing, np.int8),
        "p_pos": pp, "p_a02": pa, "p_samuel": p,
        "operational_threshold": threshold, "samuel_prediction": label,
        "prediction_at_050": (p >= 0.5).astype(np.int8)})
    if result.record_id.duplicated().any() or not result.cycle_code.gt(cutoff).all():
        raise ValueError("Bad identity or temporal boundary")
    return result


def validate_tables(val, test):
    if len(val) != 29394 or len(test) != 29421:
        raise ValueError("Unexpected validation/Test counts")
    if val.duplicated(["origin", "record_id"]).any() or test.record_id.duplicated().any():
        raise ValueError("Duplicate origin+ID or Test ID")
    if {"artifact_signal", "threat_rank", "y", "target"} & (set(val) | set(test)):
        raise ValueError("Raw labels must not be exported")
    for t in [val, test]:
        if t.record_id.isna().any() or not t.stream_missing.isin([0, 1]).all() or not t.fit_cycle_start.eq(47).all():
            raise ValueError("Invalid IDs, availability flag or fit start")
        p, threshold, labels = policy(t.p_pos, t.p_a02, t.stream_missing)
        if not np.allclose(t.p_samuel, p, rtol=0, atol=1e-15):
            raise ValueError("Probability policy replay failed")
        if not np.array_equal(t.operational_threshold, threshold) or not np.array_equal(t.samuel_prediction, labels):
            raise ValueError("Operational decision replay failed")
        if not t.cycle_code.gt(t.fit_cycle_end).all():
            raise ValueError("Prediction cycle overlaps fit prefix")
    expected = {"E67": (20903, 67, 68, 79), "E75": (8491, 75, 76, 79)}
    for origin, (n, cutoff, lo, hi) in expected.items():
        part = val[val.origin.eq(origin)]
        if len(part) != n or not part.fit_cycle_end.eq(cutoff).all() or not part.cycle_code.between(lo, hi).all():
            raise ValueError("Invalid origin view")
    expected_primary = val.origin.eq("E75") | val.cycle_code.le(75)
    if not np.array_equal(val.is_primary_unique, expected_primary.astype(np.int8)):
        raise ValueError("Primary unique assignment differs")
    primary = val[val.is_primary_unique.eq(1)]
    if len(primary) != 20903 or primary.record_id.duplicated().any():
        raise ValueError("Primary unique support differs")
    if not test.origin.eq("E79").all() or not test.fit_cycle_end.eq(79).all() or not test.cycle_code.between(80, 91).all():
        raise ValueError("Invalid Test scope")
    if int(test.stream_missing.sum()) != 1062 or int(test.samuel_prediction.sum()) != 6759:
        raise ValueError("Champion Test count replay differs")
    return {"validation_rows": len(val), "test_rows": len(test), "primary_unique_rows": len(primary),
            "overlapping_validation_IDs_across_origins": int(val.record_id.duplicated().sum()),
            "test_missing_rows": int(test.stream_missing.sum()), "test_predicted_positive": int(test.samuel_prediction.sum())}


def export(destination):
    destination = Path(destination); folder = destination / "artifacts"
    if (folder / "MANIFEST.json").exists():
        raise FileExistsError("Bundle already exists; verify it instead of overwriting")
    folder.mkdir(parents=True, exist_ok=True)
    for rel, expected in SOURCES.values():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Frozen source hash mismatch: {rel}")
    raw = ROOT / "data/raw/classification"
    if sha(raw / "Train.csv") != TRAIN_SHA or sha(raw / "Test.csv") != TEST_SHA or sha(BEST_CSV) != BEST_SHA:
        raise ValueError("Official input or confirmed champion CSV hash differs")
    train = pd.read_csv(raw / "Train.csv", usecols=["record_id", "cycle_code", "signal_stream"])
    official_test = pd.read_csv(raw / "Test.csv", usecols=["record_id", "cycle_code", "signal_stream"])
    parts = []
    with np.load(ROOT / SOURCES["validation"][0], allow_pickle=False) as z:
        for origin, cutoff in [("E67", 67), ("E75", 75)]:
            rows = z[origin + "_rows"].astype(int)
            t = frame(z[origin + "_ids"], z[origin + "_cycles"], z[origin + "_missing"], z[origin + "_pP"], z[origin + "_pA"], origin, cutoff, rows)
            selected = train.iloc[rows]
            if not np.array_equal(t.record_id, selected.record_id) or not np.array_equal(t.cycle_code, selected.cycle_code):
                raise ValueError("Validation source IDs/cycles differ")
            mask = selected.signal_stream.fillna("").str.strip().eq("").to_numpy()
            if not np.array_equal(t.stream_missing.astype(bool), mask):
                raise ValueError("Validation missing mask differs")
            t["is_primary_unique"] = ((origin == "E75") | (t.cycle_code <= 75)).astype(np.int8)
            parts.append(t)
    val = pd.concat(parts, ignore_index=True)
    with np.load(ROOT / SOURCES["test"][0], allow_pickle=False) as z:
        test = frame(z["record_ids"], official_test.cycle_code.to_numpy(), z["missing_mask"], z["posoffset_probability"], z["a02_probability"], "E79", 79, np.arange(len(official_test)))
    if not np.array_equal(test.record_id, official_test.record_id):
        raise ValueError("Test order differs from official source")
    if not np.array_equal(test.stream_missing.astype(bool), official_test.signal_stream.fillna("").str.strip().eq("")):
        raise ValueError("Test missing mask differs")
    confirmed = pd.read_csv(BEST_CSV)
    if not np.array_equal(test.record_id, confirmed.record_id) or not np.array_equal(test.samuel_prediction, confirmed.artifact_signal):
        raise ValueError("Export does not reproduce confirmed0.83225 submission")
    counts = validate_tables(val, test)
    files = {}
    for name, data in [("validation_probabilities.csv.gz", val), ("test_probabilities.csv.gz", test)]:
        path = folder / name
        data.to_csv(path, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
        files[name] = {"sha256": sha(path), "rows": len(data), "columns": data.columns.tolist(), "bytes": path.stat().st_size}
    manifest = {"schema_version": 1, "policy": "Samuel0.83225 public benchmark (confirmed submission58001282)",
        "export_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_artifacts": {key: {"path": rel, "sha256": digest} for key, (rel, digest) in SOURCES.items()},
        "raw_source_hashes": {"classification/Train.csv": TRAIN_SHA, "classification/Test.csv": TEST_SHA},
        "confirmed_submission_csv_sha256": BEST_SHA, "missing_threshold": MISSING_THRESHOLD,
        "formula": "present: 0.5*p_pos+0.5*p_a02 at threshold0.50; missing: p_a02 at threshold0.53230976129477",
        "validation_warning": "Model probabilities are forward predictions. Operational missing threshold was selected on reused adaptive development outcomes, not independently validated. E67/E75 overlap8491 IDs; do not pool both as independent rows.",
        "labels": "No official Train labels or Test labels exported. samuel_prediction is a model prediction, not ground truth. Join Train labels locally by record_id when evaluating.",
        "calibration_warning": "p_samuel is a score in[0,1], not proven calibrated or directly commensurate with teammate scores.",
        "new_fits": 0, "new_inference": 0, "new_metric_selection": 0, "kaggle_access": False,
        "counts": counts, "files": files}
    (folder / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify(destination)


def verify(destination):
    folder = Path(destination) / "artifacts"
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, item in manifest["files"].items():
        if sha(folder / name) != item["sha256"]:
            raise ValueError(f"Bundle file hash differs: {name}")
    val, test = [pd.read_csv(folder / name, float_precision="round_trip") for name in ["validation_probabilities.csv.gz", "test_probabilities.csv.gz"]]
    return {"status": "PASS", **validate_tables(val, test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export-local", "verify"])
    parser.add_argument("--destination", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = export(args.destination) if args.action == "export-local" else verify(args.destination)
    print(json.dumps(result, indent=2))
