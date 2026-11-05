"""One recovery-aware Test application of the certified, already fitted gamma."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import psutil
from scipy.special import expit

AUDIT = Path(__file__).resolve().parents[1]
REPORT = AUDIT / "reports/simplification_v1/CLS_GAMMA_ROOT_APPLICATION_R2"
CSV = AUDIT / "submissions/root_cls_gamma_recovered_v2/classification_global_strength_recovered.csv"
CERT = AUDIT / "reports/simplification_v1/CLS_GAMMA_ROOT_CERTIFICATE_R2.json"
POLICY = AUDIT / "reports/simplification_v1/RESOURCE_POLICY_USER_REVISED_20261106.json"
CERT_SHA = "ba154fa5198aefffc59f928464c0eb168bfc11206f43f5d4a78d8a5d9d3064ed"
GAMMA_SHA = "4bc240bb6b53eb623f5f099246b70572f987b8c3b0f8dfc123f49029d3c079ab"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def route(offset, x, gamma, a02, present):
    result = a02.copy()
    result[present] = 0.5 * expit(offset[present] + np.asarray(x[present] @ gamma).reshape(-1)) + 0.5 * a02[present]
    labels = (result >= np.where(present, 0.50, 0.53230976129477)).astype(np.int8)
    assert np.isfinite(result).all() and np.all((result >= 0) & (result <= 1))
    assert np.array_equal(result[~present].view(np.uint64), a02[~present].view(np.uint64))
    return result, labels


def self_test():
    from scipy import sparse
    x = sparse.csr_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    offset = np.array([0., 0., 0.]); coef = np.zeros(3)
    a02 = np.array([.5, .53230976129477, .4]); present = np.array([True, False, True])
    p, labels = route(offset, x, coef, a02, present)
    np.testing.assert_array_equal(labels, [1, 1, 0])
    assert p[1].tobytes() == a02[1].tobytes()
    text = io.StringIO(newline=""); writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["record_id", "artifact_signal"]); writer.writerows(zip(["a", "b", "c"], map(int, labels)))
    assert list(csv.reader(io.StringIO(text.getvalue())))[1:] == [["a", "1"], ["b", "1"], ["c", "0"]]
    print(json.dumps({"status": "PASS_SYNTHETIC_ROUTE_THRESHOLD_CSV"}))


def load_context(repository):
    sys.path.insert(0, str(repository))
    from scripts import optimization_v3_cls_pos_global_strength_e79_gamma_v1 as g
    assert sha(CERT) == CERT_SHA
    cert = read(CERT)
    assert cert["status"] == "RECOVERED_ARTIFACT_NUMERICALLY_VERIFIED" and cert["certificate_pass"] is True
    assert cert["new_predictor_fits"] == cert["new_preprocessing_fits"] == 0
    assert cert["objective_gap_upper_bound"] <= cert["fixed_acceptance_threshold"]
    g.verify_source_freeze(repository)
    return g, cert


def prepare(repository):
    g, cert = load_context(repository)
    _, gamma_path, failed = g._fit_paths(repository)
    paths = {"gamma": gamma_path, "original_failed_receipt": failed, "test_features": g.cgeom.TEST,
             "a02_probability_cache": g.cgeom.A02_TEST_NPZ, "a02_test_receipt": g.cgeom.A02_TEST_RECEIPT,
             "a02_fit_receipt": repository / g.r3.A02_RECEIPT_REL, "certificate": CERT, "ram_policy": POLICY,
             "application_script": Path(__file__)}
    pins = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in paths.items()}
    assert pins["gamma"]["sha256"] == GAMMA_SHA
    assert pins["test_features"]["sha256"] == g.cgeom.EXPECTED_TEST_SHA
    assert pins["original_failed_receipt"]["sha256"] == cert["preserved_original_hashes"]["fit_receipt.json"]
    commit = subprocess.check_output(["git", "-C", str(AUDIT), "rev-parse", "HEAD"], text=True).strip()
    source = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"{commit}:scripts/{Path(__file__).name}"])
    assert source.replace(b"\r\n", b"\n") == Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    manifest = {"status": "ROOT_AUTHORIZED_ONE_CERTIFIED_ARTIFACT_TEST_APPLICATION", "source_commit": commit,
                "repository": str(repository), "inputs": pins, "output_csv": str(CSV),
                "new_model_or_preprocessing_fits": 0, "test_features_authorized": True,
                "test_labels_or_sample_authorized": False, "kaggle_upload_authorized": False,
                "original_failed_receipts_not_rewritten": True}
    write_once(REPORT / "APPLICATION_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "manifest_sha256": sha(REPORT / "APPLICATION_MANIFEST.json")}))


def apply(repository):
    g, cert = load_context(repository)
    manifest = read(REPORT / "APPLICATION_MANIFEST.json")
    assert manifest["repository"] == str(repository) and manifest["new_model_or_preprocessing_fits"] == 0
    for item in manifest["inputs"].values():
        assert sha(item["path"]) == item["sha256"]
    marker = REPORT / "ATTEMPT.json"
    assert not marker.exists() and not CSV.exists(), "No repeated application or overwrite"
    reserve = read(POLICY)["feature_rebuild_or_gamma_application"]
    with ExitStack() as stack:
        locks = [stack.enter_context(g.r3._shared_job_lock()), stack.enter_context(g.r3._cgeom_local_lock(repository))]
        before = psutil.virtual_memory().available
        assert before >= reserve["minimum_available_bytes"], "Before feature read RAM gate"
        receipt = read(g.cgeom.A02_TEST_RECEIPT); fit = read(repository / g.r3.A02_RECEIPT_REL)
        assert receipt["official_test_targets_loaded"] is False and fit["test_data_used"] is False
        assert fit["query_labels_passed_to_fit"] is False and fit["model_sha256"] == g.r3.A02_MODEL_SHA256
        assert receipt["npz_sha256"] == sha(g.cgeom.A02_TEST_NPZ)
        assert receipt["fit_receipt_sha256"] == fit["receipt_sha256"] and receipt["freeze_signature"] == fit["freeze_signature"]
        test, columns = g.r3.read_feature_csv(g.cgeom.TEST, g.cgeom.CLASSIFICATION_FEATURES)
        ids = test.record_id.astype(str).to_numpy(dtype=str)
        assert len(ids) == len(set(ids)) == 29421 and not {"artifact_signal", "threat_rank"}.intersection(test.columns)
        assert set(test.cycle_code.astype(int)) == set(range(80, 92))
        with np.load(g.cgeom.A02_TEST_NPZ, allow_pickle=False) as archive:
            assert np.array_equal(archive["record_ids"].astype(str), ids)
            a02 = np.asarray(archive["candidate_probability"], dtype=np.float64).copy()
        assert a02.shape == (29421,) and np.isfinite(a02).all() and np.all((a02 >= 0) & (a02 <= 1))
        built = g._build_feature_matrices_for_rows(repository, test)
        present = ~built["missing_mask"]
        assert np.sum(~present) == 1062
        with np.load(manifest["inputs"]["gamma"]["path"], allow_pickle=False) as archive:
            gamma = archive["gamma_float64"].copy()
        assert gamma.shape == (built["position"].shape[1],) and np.isfinite(gamma).all()
        write_once(marker, {"status": "ONE_CERTIFIED_ARTIFACT_INFERENCE_STARTED", "fit_count": 0,
                            "manifest_sha256": sha(REPORT / "APPLICATION_MANIFEST.json")})
        probability, labels = route(built["offset"], built["position"], gamma, a02, present)
        # Independent application replay via the unchanged, previously tested recipe.
        replay = g.recipe.route_and_combine(built["position"], built["offset"], gamma, present, a02)
        np.testing.assert_array_equal(probability, replay)
        np.testing.assert_array_equal(labels, g.recipe.apply_frozen_thresholds(replay, present))
        assert psutil.virtual_memory().available >= reserve["minimum_actual_reserve_bytes"]
        CSV.parent.mkdir(parents=True, exist_ok=True)
        with CSV.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n"); writer.writerow(["record_id", "artifact_signal"])
            writer.writerows(zip(ids.tolist(), labels.astype(int).tolist()))
        rows = list(csv.reader(io.StringIO(CSV.read_text())))
        assert rows[0] == ["record_id", "artifact_signal"] and [r[0] for r in rows[1:]] == ids.tolist()
        np.testing.assert_array_equal([int(r[1]) for r in rows[1:]], labels)
        after = psutil.virtual_memory().available
    for item in manifest["inputs"].values():
        assert sha(item["path"]) == item["sha256"]
    result = {"status": "PASS_CERTIFIED_ARTIFACT_TEST_APPLICATION_CSV", "csv": str(CSV), "csv_sha256": sha(CSV),
              "rows": len(ids), "positive_labels": int(labels.sum()), "missing_rows": int((~present).sum()),
              "ordered_ids_sha256": g.positional.sha256_array(ids), "missing_fallback_bit_exact": True,
              "independent_route_and_threshold_replay": True, "test_labels_or_sample_read": False,
              "new_model_or_preprocessing_fits": 0, "kaggle_upload": False, "locks": locks,
              "available_ram_before": before, "available_ram_after": after, "feature_columns": columns,
              "certificate_sha256": CERT_SHA, "application_manifest_sha256": sha(REPORT / "APPLICATION_MANIFEST.json")}
    write_once(REPORT / "APPLICATION_RESULT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["self-test", "prepare", "apply"])
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test": self_test()
    else:
        assert args.repository
        repository = args.repository.resolve()
        prepare(repository) if args.mode == "prepare" else apply(repository)
