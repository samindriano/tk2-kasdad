"""Prepare one frozen E79 CSV using already saved Train-only model predictions.

This is arithmetic/recomposition only: no model fit, preprocessing fit, Test
prediction, label read, threshold selection, SampleSubmission access, or Kaggle
operation is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/PREFIT_FREEZE.json"
REPORT_DIR = ROOT / "reports/optimization_v3/cls_mix50_a02_missing_e79_v1"
BUNDLE_PATH = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
BASELINE_CSV = ROOT / "submissions/optimization_v3/posoffset_a02_endgame_v1/classification_mix50_e79.csv"
OFFICIAL_TEST = ROOT / "data/raw/classification/Test.csv"
SUBMISSION_ROOT = ROOT / "submissions"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_mix50_a02_missing_e79_v1/classification_mix50_present_a02_missing_e79.csv"

POLICY_ID = "MIX50_PRESENT_A02_MISSING"
THRESHOLD = 0.50
REQUIRED_BUNDLE_KEYS = {"record_ids", "missing_mask", "posoffset_probability", "a02_probability"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_ids(ids: Iterable[str]) -> list[str]:
    result = [str(value) for value in ids]
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError("record_ids must be nonempty, unique, and ordered")
    return result


def route_probabilities(
    p_p: np.ndarray, p_a: np.ndarray, missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline MIX50 and the frozen A02-on-missing candidate."""
    pos = np.asarray(p_p, dtype=np.float64)
    a02 = np.asarray(p_a, dtype=np.float64)
    miss = np.asarray(missing, dtype=bool)
    if pos.ndim != 1 or pos.shape != a02.shape or pos.shape != miss.shape:
        raise ValueError("probabilities and missing mask must be aligned one-dimensional arrays")
    for name, probability in (("pP", pos), ("pA", a02)):
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    present_blend = 0.5 * pos + 0.5 * a02
    baseline = np.where(miss, pos, present_blend)
    candidate = np.where(miss, a02, present_blend)
    return baseline, candidate


def binary_labels(probability: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be a finite vector in [0, 1]")
    return (p >= float(threshold)).astype(np.int8)


def compare_missing_labels(
    baseline: np.ndarray, candidate: np.ndarray, missing: np.ndarray
) -> dict[str, int]:
    base = np.asarray(baseline, dtype=np.int8)
    new = np.asarray(candidate, dtype=np.int8)
    miss = np.asarray(missing, dtype=bool)
    if base.ndim != 1 or base.shape != new.shape or base.shape != miss.shape:
        raise ValueError("label vectors and missing mask must align")
    if not np.isin(base, [0, 1]).all() or not np.isin(new, [0, 1]).all():
        raise ValueError("labels must be integer 0/1")
    return {
        "missing_rows": int(miss.sum()),
        "changed_missing_count": int(np.sum(base[miss] != new[miss])),
        "newly_positive_missing_count": int(np.sum((base[miss] == 0) & (new[miss] == 1))),
        "newly_negative_missing_count": int(np.sum((base[miss] == 1) & (new[miss] == 0))),
        "baseline_missing_positive_count": int(base[miss].sum()),
        "candidate_missing_positive_count": int(new[miss].sum()),
        "candidate_total_positive_count": int(new.sum()),
    }


def _read_csv_ids_labels(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if fields != ["record_id", "artifact_signal"]:
            raise RuntimeError(f"Unexpected submission columns in {path}: {fields}")
        ids: list[str] = []
        labels: list[int] = []
        for row in reader:
            ids.append(row["record_id"])
            value = row["artifact_signal"]
            if value not in {"0", "1"}:
                raise RuntimeError(f"Non-binary label in {path}")
            labels.append(int(value))
    validate_ids(ids)
    return ids, np.asarray(labels, dtype=np.int8), fields


def _read_test_ids_and_missing(path: Path) -> tuple[list[str], np.ndarray]:
    """Read only Test identity and signal availability; never read target labels."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError("Official Test CSV is empty") from exc
        columns = {name: i for i, name in enumerate(header)}
        if "record_id" not in columns or "signal_stream" not in columns:
            raise RuntimeError("Official Test lacks record_id or signal_stream")
        ids: list[str] = []
        missing: list[bool] = []
        max_col = max(columns["record_id"], columns["signal_stream"])
        for row in reader:
            if len(row) <= max_col:
                raise RuntimeError("Malformed official Test row")
            ids.append(row[columns["record_id"]])
            missing.append(row[columns["signal_stream"]].strip() == "")
    validate_ids(ids)
    return ids, np.asarray(missing, dtype=bool)


def _load_bundle(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        absent = REQUIRED_BUNDLE_KEYS - set(archive.files)
        if absent:
            raise RuntimeError(f"Frozen E79 probability bundle lacks keys: {sorted(absent)}")
        ids = validate_ids(archive["record_ids"].astype(str).tolist())
        missing = np.asarray(archive["missing_mask"], dtype=bool)
        p_p = np.asarray(archive["posoffset_probability"], dtype=np.float64)
        p_a = np.asarray(archive["a02_probability"], dtype=np.float64)
    n = len(ids)
    if missing.shape != (n,) or p_p.shape != (n,) or p_a.shape != (n,):
        raise RuntimeError("Frozen E79 bundle arrays do not match the ID vector")
    return ids, missing, p_p, p_a


def _canonical_vector(ids: list[str], labels: np.ndarray) -> tuple[str, dict[str, int]]:
    id_order = validate_ids(ids)
    y = np.asarray(labels, dtype=np.int8)
    if y.shape != (len(id_order),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Full submission vector must be aligned binary labels")
    by_id = dict(zip(id_order, y.tolist()))
    return "\n".join(id_order), by_id


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def scan_duplicate_full_vectors(
    candidate_ids: list[str], candidate_labels: np.ndarray, *, root: Path, exclude: Path
) -> dict[str, Any]:
    """Scan local submission CSVs for exact full-Test label vectors, order independent."""
    official_order, candidate_by_id = _canonical_vector(candidate_ids, candidate_labels)
    canonical_ids = official_order.splitlines()
    matches: list[dict[str, str]] = []
    eligible = 0
    ignored_nonclassification = 0
    ignored_nonfull_or_other_ids = 0
    invalid_full_vectors: list[str] = []
    for path in sorted(root.rglob("*.csv")):
        if path.resolve() == exclude.resolve():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            if not {"record_id", "artifact_signal"}.issubset(fields):
                ignored_nonclassification += 1
                continue
            rows = list(reader)
        ids = [row.get("record_id", "") for row in rows]
        if len(ids) != len(canonical_ids) or len(set(ids)) != len(ids) or set(ids) != set(canonical_ids):
            ignored_nonfull_or_other_ids += 1
            continue
        labels: list[int] = []
        for row in rows:
            value = row.get("artifact_signal", "")
            if value not in {"0", "1"}:
                invalid_full_vectors.append(_display_path(path))
                labels = []
                break
            labels.append(int(value))
        if not labels:
            continue
        eligible += 1
        existing_by_id = dict(zip(ids, labels))
        if all(existing_by_id[rid] == candidate_by_id[rid] for rid in canonical_ids):
            matches.append({"path": _display_path(path), "sha256": sha256_file(path)})
    if invalid_full_vectors:
        raise RuntimeError("Full-size classification submission(s) had invalid labels: " + ", ".join(invalid_full_vectors))
    return {
        "scanned_root": _display_path(root),
        "full_test_classification_vectors_scanned": eligible,
        "ignored_nonclassification_csvs": ignored_nonclassification,
        "ignored_nonfull_or_other_id_csvs": ignored_nonfull_or_other_ids,
        "exact_duplicate_matches": matches,
        "duplicate_found": bool(matches),
    }


def _verify_frozen_checkout(freeze: dict[str, Any]) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base = freeze["base_commit"]
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT, check=False)
    if ancestor.returncode != 0 or head == base:
        raise RuntimeError("Prefit source/freeze must be committed after the frozen base commit")
    owned_paths = [
        "scripts/optimization_v3_cls_mix50_a02_missing_e79_v1.py",
        "tests/test_optimization_v3_cls_mix50_a02_missing_e79_v1.py",
        "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/PREFIT_FREEZE.json",
        "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/SYNTHETIC_QA.json",
    ]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *owned_paths], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Frozen source paths have uncommitted changes")
    for relative, expected in freeze["source_hashes"].items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    return head


def _validate_frozen_inputs(freeze: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in freeze["input_hashes"].items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {relative}")
    return actual


def _validate_baseline_provenance(freeze: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/E79_SELECTED_MANIFEST.json"
    qa_path = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/E79_INDEPENDENT_QA.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    recorded = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if recorded != canonical_sha256(body):
        raise RuntimeError("Frozen MIX50 E79 manifest canonical hash is invalid")
    if qa.get("status") != "PASS" or qa.get("candidate") != "MIX50":
        raise RuntimeError("Frozen MIX50 E79 independent QA is not PASS")
    if manifest.get("candidate") != "MIX50" or manifest.get("csv_path") != BASELINE_CSV.relative_to(ROOT).as_posix():
        raise RuntimeError("Frozen baseline manifest does not identify the expected MIX50 CSV")
    if manifest.get("formula") != "missing: pP exactly; present: 0.5*pP + 0.5*pA; threshold >= 0.50":
        raise RuntimeError("Frozen baseline manifest formula differs from current MIX50")
    if manifest.get("source_hashes", {}).get("combined_e79_probability_bundle") != freeze["input_hashes"][BUNDLE_PATH.relative_to(ROOT).as_posix()]:
        raise RuntimeError("MIX50 manifest E79 bundle hash differs from the prefit freeze")
    if manifest.get("source_hashes", {}).get("official_test") != freeze["input_hashes"][OFFICIAL_TEST.relative_to(ROOT).as_posix()]:
        raise RuntimeError("MIX50 manifest official Test hash differs from the prefit freeze")
    if manifest.get("recipe_sha256") != freeze["source_hashes"]["reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json"]:
        raise RuntimeError("MIX50 manifest recipe hash differs from the prefit freeze")
    return {
        "manifest_sha256": recorded,
        "independent_qa_status": qa["status"],
        "baseline_csv_sha256": manifest["csv_sha256"],
        "baseline_recipe_sha256": manifest["recipe_sha256"],
    }


def _write_csv_once(path: Path, ids: list[str], labels: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["record_id,artifact_signal\n"]
    lines.extend(f"{rid},{int(value)}\n" for rid, value in zip(ids, labels.tolist()))
    payload = "".join(lines).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Refusing to overwrite a different prepared CSV: {path}")
        return digest
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return digest


def _write_json_once(path: Path, value: dict[str, Any]) -> str:
    body = dict(value)
    body["canonical_sha256"] = canonical_sha256(body)
    payload = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Refusing to overwrite a different immutable report: {path}")
        return digest
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return digest


def build_candidate() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("policy_id") != POLICY_ID or freeze.get("threshold") != THRESHOLD:
        raise RuntimeError("Prefit freeze policy identity differs from the authorized fixed policy")
    prefit_head = _verify_frozen_checkout(freeze)
    input_hashes = _validate_frozen_inputs(freeze)
    baseline_provenance = _validate_baseline_provenance(freeze)

    bundle_ids, bundle_missing, p_p, p_a = _load_bundle(BUNDLE_PATH)
    test_ids, test_missing = _read_test_ids_and_missing(OFFICIAL_TEST)
    base_ids, baseline_labels, _ = _read_csv_ids_labels(BASELINE_CSV)
    if bundle_ids != test_ids or bundle_ids != base_ids:
        raise RuntimeError("E79 bundle, official Test, and frozen baseline IDs/order differ")
    if not np.array_equal(bundle_missing, test_missing):
        raise RuntimeError("Frozen E79 availability mask differs from official Test signal_stream")

    baseline_probability, candidate_probability = route_probabilities(p_p, p_a, bundle_missing)
    replay_baseline = binary_labels(baseline_probability)
    candidate_labels = binary_labels(candidate_probability)
    if not np.array_equal(replay_baseline, baseline_labels):
        raise RuntimeError("Saved MIX50 E79 baseline CSV does not replay from the frozen probability bundle")
    baseline_label_sha256 = hashlib.sha256(np.asarray(baseline_labels, dtype=np.int8).tobytes()).hexdigest()
    baseline_manifest = json.loads((ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/E79_SELECTED_MANIFEST.json").read_text(encoding="utf-8"))
    if baseline_label_sha256 != baseline_manifest.get("label_vector_sha256"):
        raise RuntimeError("Frozen MIX50 baseline label vector hash does not match its manifest")
    present = ~bundle_missing
    if not np.array_equal(candidate_probability[present], baseline_probability[present]):
        raise RuntimeError("Candidate changed present-stream probabilities")
    if not np.array_equal(candidate_labels[present], baseline_labels[present]):
        raise RuntimeError("Candidate changed present-stream labels")
    if not np.array_equal(candidate_probability[bundle_missing], p_a[bundle_missing]):
        raise RuntimeError("Candidate missing branch differs from exact saved A02 pA")

    duplicate_scan = scan_duplicate_full_vectors(
        bundle_ids, candidate_labels, root=SUBMISSION_ROOT, exclude=OUTPUT_CSV
    )
    csv_sha = _write_csv_once(OUTPUT_CSV, bundle_ids, candidate_labels)
    output_ids, output_labels, output_columns = _read_csv_ids_labels(OUTPUT_CSV)
    if output_ids != test_ids or not np.array_equal(output_labels, candidate_labels):
        raise RuntimeError("Prepared CSV failed independent schema/order/label replay")
    metrics = compare_missing_labels(baseline_labels, candidate_labels, bundle_missing)
    manifest = {
        "schema_version": 1,
        "status": "PREPARED_FOR_ROOT_REVIEW_NOT_UPLOADED",
        "policy_id": POLICY_ID,
        "formula": "present=0.5*pP+0.5*pA; missing=pA; threshold>=0.50",
        "threshold": THRESHOLD,
        "source_base_commit": freeze["base_commit"],
        "prefit_source_commit": prefit_head,
        "prefit_freeze_sha256": sha256_file(FREEZE_PATH),
        "source_hashes": freeze["source_hashes"],
        "input_hashes": input_hashes,
        "baseline_provenance": baseline_provenance,
        "baseline_label_vector_sha256": baseline_label_sha256,
        "bundle_path": BUNDLE_PATH.relative_to(ROOT).as_posix(),
        "baseline_csv_path": BASELINE_CSV.relative_to(ROOT).as_posix(),
        "official_test_path": OFFICIAL_TEST.relative_to(ROOT).as_posix(),
        "csv_path": OUTPUT_CSV.relative_to(ROOT).as_posix(),
        "csv_sha256": csv_sha,
        "rows": len(bundle_ids),
        "unique_record_ids": len(set(bundle_ids)),
        "columns": output_columns,
        "candidate_positive_count": int(candidate_labels.sum()),
        "missing_stream_rows": int(bundle_missing.sum()),
        "present_stream_rows": int(present.sum()),
        "missing_label_comparison": metrics,
        "present_probabilities_bit_exact_to_mix50": True,
        "present_labels_bit_exact_to_mix50": True,
        "duplicate_full_label_scan": duplicate_scan,
        "new_fits": 0,
        "new_model_inference": 0,
        "test_features_read_for_ids_and_availability_only": True,
        "test_labels_read": False,
        "sample_submission_read": False,
        "kaggle_contacted": False,
        "upload_performed": False,
    }
    manifest_sha = _write_json_once(REPORT_DIR / "MANIFEST.json", manifest)
    qa = {
        "schema_version": 1,
        "status": "PASS_LOCAL_CSV_QA",
        "policy_id": POLICY_ID,
        "prefit_source_commit": prefit_head,
        "prefit_freeze_sha256": sha256_file(FREEZE_PATH),
        "manifest_sha256": manifest_sha,
        "checks": {
            "frozen_source_commit_and_hashes_verified": True,
            "frozen_input_hashes_verified": True,
            "E79_bundle_ids_match_official_Test_order": True,
            "baseline_ids_match_official_Test_order": True,
            "missing_mask_matches_Test_signal_stream_strip_empty": True,
            "MIX50_baseline_replayed_from_saved_probabilities": True,
            "present_candidate_probabilities_unchanged": True,
            "present_candidate_labels_unchanged": True,
            "missing_candidate_probability_equals_A02_pA": True,
            "threshold_is_fixed_inclusive_0_50": True,
            "CSV_columns_order_rows_ids_and_integer_labels_verified": True,
            "CSV_sha256_verified": sha256_file(OUTPUT_CSV) == csv_sha,
            "full_label_duplicate_scan_completed": True,
            "duplicate_vector_found": duplicate_scan["duplicate_found"],
            "test_labels_not_read": True,
            "SampleSubmission_not_read": True,
            "new_fits": 0,
            "new_model_inference": 0,
            "Kaggle_not_contacted": True,
        },
    }
    qa_sha = _write_json_once(REPORT_DIR / "QA.json", qa)
    return {"manifest": manifest, "manifest_sha256": manifest_sha, "qa": qa,
            "qa_sha256": qa_sha, "csv_path": str(OUTPUT_CSV), "csv_sha256": csv_sha}


def main() -> None:
    print(json.dumps(build_candidate(), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
