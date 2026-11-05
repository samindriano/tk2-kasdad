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
REPORT = AUDIT / "reports/simplification_v1/CLS_GAMMA_ROOT_APPLICATION_R4"
CSV = AUDIT / "submissions/root_cls_gamma_recovered_v2/classification_global_strength_recovered.csv"
CERT = AUDIT / "reports/simplification_v1/CLS_GAMMA_ROOT_CERTIFICATE_R2.json"
POLICY = AUDIT / "reports/simplification_v1/RESOURCE_POLICY_USER_REVISED_20261106.json"
CERT_SHA = "ba154fa5198aefffc59f928464c0eb168bfc11206f43f5d4a78d8a5d9d3064ed"
GAMMA_SHA = "4bc240bb6b53eb623f5f099246b70572f987b8c3b0f8dfc123f49029d3c079ab"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message="Application invariant failed"):
    if not condition:
        raise RuntimeError(message)


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
    require(np.isfinite(result).all() and np.all((result >= 0) & (result <= 1)), 'Invariant at application source line 55')
    require(np.array_equal(result[~present].view(np.uint64), a02[~present].view(np.uint64)), 'Invariant at application source line 56')
    return result, labels


def self_test():
    from scipy import sparse
    x = sparse.csr_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    offset = np.array([0., 0., 0.]); coef = np.zeros(3)
    a02 = np.array([.5, .53230976129477, .4]); present = np.array([True, False, True])
    p, labels = route(offset, x, coef, a02, present)
    np.testing.assert_array_equal(labels, [1, 1, 0])
    require(p[1].tobytes() == a02[1].tobytes(), 'Invariant at application source line 67')
    text = io.StringIO(newline=""); writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["record_id", "artifact_signal"]); writer.writerows(zip(["a", "b", "c"], map(int, labels)))
    require(list(csv.reader(io.StringIO(text.getvalue())))[1:] == [['a', '1'], ['b', '1'], ['c', '0']], 'Invariant at application source line 70')
    print(json.dumps({"status": "PASS_SYNTHETIC_ROUTE_THRESHOLD_CSV"}))


def load_context(repository):
    sys.path.insert(0, str(repository))
    from scripts import optimization_v3_cls_pos_global_strength_e79_gamma_v1 as g
    require(sha(CERT) == CERT_SHA, 'Invariant at application source line 77')
    cert = read(CERT)
    require(cert['status'] == 'RECOVERED_ARTIFACT_NUMERICALLY_VERIFIED' and cert['certificate_pass'] is True, 'Invariant at application source line 79')
    require(cert['new_predictor_fits'] == cert['new_preprocessing_fits'] == 0, 'Invariant at application source line 80')
    require(cert['objective_gap_upper_bound'] <= cert['fixed_acceptance_threshold'], 'Invariant at application source line 81')
    g.verify_source_freeze(repository)
    original_run = repository / "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1"
    for relative, expected in cert["preserved_original_hashes"].items():
        require(sha(original_run / relative) == expected, f"Preserved failure-chain hash changed: {relative}")
    with np.load(original_run / "gamma.npz", allow_pickle=False) as archive:
        require(archive.files == ["gamma_float64"], "Gamma artifact member mismatch")
        coef = archive["gamma_float64"].copy()
    require(coef.shape == (17259,) and coef.dtype == np.float64 and np.isfinite(coef).all(), "Gamma schema failed")
    require(g.positional.sha256_array(coef) == cert["gamma_logical_sha256"], "Gamma logical digest differs from certificate")
    return g, cert


def prepare(repository):
    g, cert = load_context(repository)
    _, gamma_path, failed = g._fit_paths(repository)
    original_run = gamma_path.parent
    paths = {"gamma": gamma_path, "original_failed_receipt": failed,
             "original_attempt": original_run / "ATTEMPT.started.json",
             "failed_recovery_receipt": original_run / "recovery_v1/RECOVERY_RECEIPT.json", "test_features": g.cgeom.TEST,
             "a02_probability_cache": g.cgeom.A02_TEST_NPZ, "a02_test_receipt": g.cgeom.A02_TEST_RECEIPT,
             "a02_fit_receipt": repository / g.r3.A02_RECEIPT_REL, "certificate": CERT, "ram_policy": POLICY,
             "application_script": Path(__file__)}
    pins = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in paths.items()}
    require(pins['gamma']['sha256'] == GAMMA_SHA, 'Invariant at application source line 105')
    require(pins['test_features']['sha256'] == g.cgeom.EXPECTED_TEST_SHA, 'Invariant at application source line 106')
    require(pins['original_failed_receipt']['sha256'] == cert['preserved_original_hashes']['fit_receipt.json'], 'Invariant at application source line 107')
    commit = subprocess.check_output(["git", "-C", str(AUDIT), "rev-parse", "HEAD"], text=True).strip()
    source = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"{commit}:scripts/{Path(__file__).name}"])
    require(source.replace(b'\r\n', b'\n') == Path(__file__).read_bytes().replace(b'\r\n', b'\n'), 'Invariant at application source line 110')
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
    require(manifest['repository'] == str(repository) and manifest['new_model_or_preprocessing_fits'] == 0, 'Invariant at application source line 123')
    manifest_blob = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"HEAD:{(REPORT / 'APPLICATION_MANIFEST.json').relative_to(AUDIT).as_posix()}"])
    require(manifest_blob.replace(b"\r\n", b"\n") == (REPORT / "APPLICATION_MANIFEST.json").read_bytes().replace(b"\r\n", b"\n"), "Manifest is not committed exactly")
    marker = REPORT / "ATTEMPT.json"
    require(not marker.exists() and (not CSV.exists()), 'No repeated application or overwrite')
    reserve = read(POLICY)["feature_rebuild_or_gamma_application"]
    with ExitStack() as stack:
        locks = [stack.enter_context(g.r3._shared_job_lock()), stack.enter_context(g.r3._cgeom_local_lock(repository))]
        require(not marker.exists() and not CSV.exists(), "Prior application appeared before locked start")
        before = psutil.virtual_memory().available
        require(before >= reserve['minimum_available_bytes'], 'Before feature read RAM gate')
        for item in manifest["inputs"].values():
            require(sha(item["path"]) == item["sha256"], f"Frozen input changed: {item['path']}")
        receipt = read(g.cgeom.A02_TEST_RECEIPT); fit = read(repository / g.r3.A02_RECEIPT_REL)
        require(receipt['official_test_targets_loaded'] is False and fit['test_data_used'] is False, 'Invariant at application source line 136')
        require(fit['query_labels_passed_to_fit'] is False and fit['model_sha256'] == g.r3.A02_MODEL_SHA256, 'Invariant at application source line 137')
        require(receipt['npz_sha256'] == sha(g.cgeom.A02_TEST_NPZ), 'Invariant at application source line 138')
        require(receipt['fit_receipt_sha256'] == fit['receipt_sha256'] and receipt['freeze_signature'] == fit['freeze_signature'], 'Invariant at application source line 139')
        test, columns = g.r3.read_feature_csv(g.cgeom.TEST, g.cgeom.CLASSIFICATION_FEATURES)
        ids = test.record_id.astype(str).to_numpy(dtype=str)
        require(len(ids) == len(set(ids)) == 29421 and (not {'artifact_signal', 'threat_rank'}.intersection(test.columns)), 'Invariant at application source line 142')
        require(set(test.cycle_code.astype(int)) == set(range(80, 92)), 'Invariant at application source line 143')
        with np.load(g.cgeom.A02_TEST_NPZ, allow_pickle=False) as archive:
            require(np.array_equal(archive['record_ids'].astype(str), ids), 'Invariant at application source line 145')
            a02 = np.asarray(archive["candidate_probability"], dtype=np.float64).copy()
        require(a02.shape == (29421,) and np.isfinite(a02).all() and np.all((a02 >= 0) & (a02 <= 1)), 'Invariant at application source line 147')
        built = g._build_feature_matrices_for_rows(repository, test)
        present = ~built["missing_mask"]
        require(np.sum(~present) == 1062, 'Invariant at application source line 150')
        with np.load(manifest["inputs"]["gamma"]["path"], allow_pickle=False) as archive:
            gamma = archive["gamma_float64"].copy()
        require(gamma.shape == (built['position'].shape[1],) and np.isfinite(gamma).all(), 'Invariant at application source line 153')
        write_once(marker, {"status": "ONE_CERTIFIED_ARTIFACT_INFERENCE_STARTED", "fit_count": 0,
                            "manifest_sha256": sha(REPORT / "APPLICATION_MANIFEST.json")})
        probability, labels = route(built["offset"], built["position"], gamma, a02, present)
        # Independent application replay via the unchanged, previously tested recipe.
        replay = g.recipe.route_and_combine(built["position"], built["offset"], gamma, present, a02)
        np.testing.assert_array_equal(probability, replay)
        np.testing.assert_array_equal(labels, g.recipe.apply_frozen_thresholds(replay, present))
        require(psutil.virtual_memory().available >= reserve['minimum_actual_reserve_bytes'], 'Invariant at application source line 161')
        CSV.parent.mkdir(parents=True, exist_ok=True)
        with CSV.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n"); writer.writerow(["record_id", "artifact_signal"])
            writer.writerows(zip(ids.tolist(), labels.astype(int).tolist()))
        rows = list(csv.reader(io.StringIO(CSV.read_text())))
        require(rows[0] == ['record_id', 'artifact_signal'] and [r[0] for r in rows[1:]] == ids.tolist(), 'Invariant at application source line 167')
        np.testing.assert_array_equal([int(r[1]) for r in rows[1:]], labels)
        after = psutil.virtual_memory().available
        for item in manifest["inputs"].values():
            require(sha(item["path"]) == item["sha256"], "Input changed during application")
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
        require(args.repository, 'Invariant at application source line 190')
        repository = args.repository.resolve()
        prepare(repository) if args.mode == "prepare" else apply(repository)
