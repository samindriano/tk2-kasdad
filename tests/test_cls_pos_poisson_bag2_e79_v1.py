from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scripts import cls_pos_poisson_bag2_e79_v1 as runner
from scripts import verify_cls_pos_poisson_bag2_e79_v1 as verifier
from src.models import cls_pos_poisson_bag2_e79_v1 as model


def test_fixed_identity_matrix_bundle_and_policy() -> None:
    assert model.CANDIDATE_ID == "CLS_POS_POISSON_BAG2_E79_V1"
    with pytest.raises(RuntimeError, match="Unfrozen E79 Poisson seed"):
        model.fit_id_for_seed(23)
    base = sparse.csr_matrix((50_608, 20_427), dtype=np.float64)
    position = sparse.csr_matrix((50_608, 17_259), dtype=np.float64)
    built = {"base": base, "position": position, "offset": np.zeros(50_608),
             "ids_sha256": model.EXPECTED_ORDERED_TRAIN_IDS_SHA256,
             "matrices": {
                 "base": {"shape": list(model.EXPECTED_BASE_SHAPE), "nnz": model.EXPECTED_BASE_NNZ,
                          "logical_sha256": model.EXPECTED_BASE_SHA256},
                 "position": {"shape": list(model.EXPECTED_POSITION_SHAPE),
                              "nnz": model.EXPECTED_POSITION_NNZ,
                              "logical_sha256": model.EXPECTED_POSITION_SHA256},
             }}
    # Bundle validation checks the frozen receipt, not the sparse contents; the
    # official legacy builder performs byte-level matrix parity before returning.
    b, p, offset = model.validate_matrix_bundle(built)
    assert b.shape == model.EXPECTED_BASE_SHAPE
    assert p.shape == model.EXPECTED_POSITION_SHAPE
    assert len(offset) == model.TRAIN_ROWS
    with pytest.raises(RuntimeError, match="Unexpected legacy matrix metadata keys"):
        model.validate_matrix_bundle({**built, "matrices": {**built["matrices"], "checks": {}}})

    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    offset_small = np.asarray([-0.1, 0.2, 0.4])
    p_a = np.asarray([0.31, 0.72, 0.63])
    present = np.asarray([True, False, True])
    policies = model.build_probabilities(x, offset_small, [0.2, -0.1], [0.3, -0.15],
                                         [0.1, -0.05], p_a, present)
    assert policies["probability_candidate"][1] == p_a[1]
    assert policies["probability_champion"][1] == p_a[1]
    labels = model.apply_thresholds(policies["probability_candidate"], present)
    assert set(labels.tolist()).issubset({0, 1})


def test_original_cgeom_recipe_requires_symbolic_objective_and_uniform_c_one() -> None:
    recipe = {
        "model": "C2_B logit as fixed offset plus real-position gamma only",
        "objective": "sum(logaddexp(0, -(2*y-1)*(z0 + X_position@gamma))) + dot(gamma,gamma)/(2*C)",
        "optimizer": "L-BFGS-B, zero initialization, C=1, maxiter=2000, ftol=1e-12, gtol=1e-6, maxls=50; no retries",
    }
    runner._verify_cgeom_recipe(recipe)
    with pytest.raises(RuntimeError, match="uniform C=1"):
        runner._verify_cgeom_recipe({**recipe, "optimizer": recipe["optimizer"].replace("C=1", "C=2")})


def test_root_pass_accepts_ancestor_review_and_requires_current_head_blob(tmp_path: Path) -> None:
    root = tmp_path / "audit"
    root.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "e79-test@example.invalid")
    git("config", "user.name", "E79 test")
    source_hashes = {}
    for relative in runner.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen source {relative}\n".encode("utf-8"))
        source_hashes[relative] = runner.stable_file_sha256(path, relative)
    freeze_path = root / runner.FREEZE_REL
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_bytes(b'{"freeze_revision":2}\n')
    freeze_sha = runner.stable_file_sha256(freeze_path, runner.FREEZE_REL)
    git("add", "--all")
    git("commit", "-m", "source and R2 freeze")
    reviewed_head = git("rev-parse", "HEAD")

    unrelated = root / "review_notes.txt"
    unrelated.write_bytes(b"unrelated committed note\n")
    git("add", "review_notes.txt")
    git("commit", "-m", "unrelated commit between review and pass")
    pass_path = root / runner.ROOT_PASS_REL
    pass_path.parent.mkdir(parents=True, exist_ok=True)
    pass_path.write_bytes(b'{"pass_sha256":"synthetic"}\n')
    git("add", runner.ROOT_PASS_REL)
    git("commit", "-m", "commit root pass")
    current_head = git("rev-parse", "HEAD")
    stray = root / "untracked_note.txt"
    stray.write_bytes(b"unrelated untracked worktree note\n")

    head, reviewed = runner._verify_root_pass_commit_binding(
        root, root_pass_rel=runner.ROOT_PASS_REL, reviewed_head=reviewed_head,
        freeze_rel=runner.FREEZE_REL, freeze_file_sha256=freeze_sha,
        source_hashes=source_hashes)
    assert head == current_head
    assert reviewed == reviewed_head
    assert head != reviewed

    altered_sources = dict(source_hashes)
    first_source = runner.SOURCE_FILES[0]
    altered_sources[first_source] = "0" * 64
    with pytest.raises(RuntimeError, match="does not contain the frozen candidate source"):
        runner._verify_root_pass_commit_binding(
            root, root_pass_rel=runner.ROOT_PASS_REL, reviewed_head=reviewed_head,
            freeze_rel=runner.FREEZE_REL, freeze_file_sha256=freeze_sha,
            source_hashes=altered_sources)
    with pytest.raises(RuntimeError, match="exact frozen R2 file"):
        runner._verify_root_pass_commit_binding(
            root, root_pass_rel=runner.ROOT_PASS_REL, reviewed_head=reviewed_head,
            freeze_rel=runner.FREEZE_REL, freeze_file_sha256="0" * 64,
            source_hashes=source_hashes)

    pass_path.write_bytes(b'{"pass_sha256":"tampered"}\n')
    with pytest.raises(RuntimeError, match="not committed exactly at current HEAD"):
        runner._verify_root_pass_commit_binding(
            root, root_pass_rel=runner.ROOT_PASS_REL, reviewed_head=reviewed_head,
            freeze_rel=runner.FREEZE_REL, freeze_file_sha256=freeze_sha,
            source_hashes=source_hashes)


def test_r2_resource_gates_and_r1_freeze_are_pinned() -> None:
    assert runner.MIN_FIT_START_BYTES == 1 * runner.GIB
    assert runner.POST_FEATURE_RESERVE_BYTES == 512 * 1024 ** 2
    assert runner.POST_FIT_RESERVE_BYTES == 512 * 1024 ** 2
    assert runner.MAX_THREADS == 2
    r1 = runner.AUDIT_ROOT / runner.FREEZE_R1_REL
    assert runner.sha256_file(r1) == runner.FREEZE_R1_SHA256


def test_synthetic_cli_end_to_end_and_resume_without_additional_fits(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    command = [sys.executable, "-O", str(runner.AUDIT_ROOT / runner.RUNNER_REL),
               "--command", "synthetic", "--output-dir", str(output)]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    first_json = json.loads(first.stdout)
    second_json = json.loads(second.stdout)
    assert first_json == second_json
    assert first_json["status"] == "PASS_SYNTHETIC_E79_END_TO_END"
    assert first_json["fit_count"] == 2 and first_json["application_count"] == 1
    assert first_json["ledger_fit_starts"] == first_json["ledger_fit_saves"] == 2
    assert first_json["test_or_sample_access"] is False
    assert first_json["kaggle_contacted"] is False

    fit_qa = verifier.verify_fit_set(output / "fits", freeze_signature="SYNTHETIC_FREEZE")
    assert fit_qa["status"] == "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED"
    app_qa = verifier.verify_application(
        output / "fits", receipt_path=output / "synthetic_application_receipt.json",
        bundle_path=output / "synthetic_application_predictions.npz",
        csv_path=output / "synthetic_submission.csv",
        baseline_csv_path=output / "SYNTHETIC_BASELINE.csv",
        expected_baseline_sha256=verifier.sha256_file(output / "SYNTHETIC_BASELINE.csv"))
    assert app_qa["status"] == "PASS_INDEPENDENT_E79_APPLICATION_READBACK"
    assert app_qa["baseline_champion_decision_parity"] is True
    assert app_qa["missing_fallback_exact"] is True


def test_synthetic_fit_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    runner.run_synthetic(output)
    fit_id = model.FIT_IDS[0]
    tampered = tmp_path / "tampered"
    shutil.copytree(output, tampered)
    artifact = tampered / "fits" / fit_id / "gamma.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        gamma = archive["gamma"].copy()
    gamma[0] += 0.01
    np.savez_compressed(artifact, gamma=gamma)
    with pytest.raises(RuntimeError, match="artifact file hash mismatch"):
        verifier.verify_fit_artifact(tampered / "fits", fit_id,
                                     freeze_signature="SYNTHETIC_FREEZE")


def test_synthetic_fit_receipt_tampering_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    runner.run_synthetic(output)
    fit_id = model.FIT_IDS[0]
    receipt_path = output / "fits" / fit_id / "FIT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["ridge_c"] = 2.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Receipt self-hash mismatch"):
        verifier.verify_fit_artifact(output / "fits", fit_id,
                                     freeze_signature="SYNTHETIC_FREEZE")


def test_synthetic_application_csv_and_receipt_tampering_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    runner.run_synthetic(output)
    csv_path = output / "synthetic_submission.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace("SYNTH-0000", "SYNTH-X000", 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CSV file hash mismatch"):
        verifier.verify_application(
            output / "fits", receipt_path=output / "synthetic_application_receipt.json",
            bundle_path=output / "synthetic_application_predictions.npz", csv_path=csv_path,
            baseline_csv_path=output / "SYNTHETIC_BASELINE.csv",
            expected_baseline_sha256=verifier.sha256_file(output / "SYNTHETIC_BASELINE.csv"))


def test_production_gates_do_not_depend_on_bare_asserts() -> None:
    paths = [runner.AUDIT_ROOT / runner.RUNNER_REL,
             runner.AUDIT_ROOT / runner.VERIFIER_REL,
             runner.AUDIT_ROOT / runner.HELPER_REL]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), path
