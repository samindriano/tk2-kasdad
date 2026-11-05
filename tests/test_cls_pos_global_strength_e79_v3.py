from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path
import shutil
import symtable
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as runner
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_v3 as independent_verifier
from scripts.optimization_v3_cls_pos_global_strength_e79_v3 import (
    CGEOM_SCRIPT_REL,
    CGEOM_TEST_REL,
    EXPLICIT_SOURCE_VARIANCES,
    ROOT,
    ReplayError,
    _hash_map,
    _receipt_source_audit,
    assemble_and_validate_matrices,
    canonical_feature_columns,
    read_feature_csv,
    sha256_file,
    validate_feature_frame_columns,
)
from src.models import classification_models, optimization_v3_positional_text_pilot_v1 as positional
from src.models.round1_models import EXPANDED_FEATURES, MIXED_COUNT_FEATURES, NUMERIC_FEATURES
from src.models.optimization_v3_positional_text_pilot_v1 import sha256_array, sha256_csr


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _read_csv_usecols(tree: ast.AST) -> list[ast.AST]:
    calls = [
        call for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "read_csv"
    ]
    return [kw.value for call in calls for kw in call.keywords if kw.arg == "usecols"]


class E79R3CompatibilityReplayTests(unittest.TestCase):
    def _synthetic_csv_frame(self, columns: list[str]) -> pd.DataFrame:
        rows = []
        for row_id in range(3):
            row = {}
            for column in columns:
                if column == "record_id":
                    row[column] = f"synthetic-{row_id}"
                elif column == "cycle_code":
                    row[column] = 47 + row_id
                elif column == "signal_stream":
                    row[column] = f"token{row_id} repeated token{row_id}"
                else:
                    row[column] = row_id + 1
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _synthetic_training_frame(self) -> pd.DataFrame:
        rows = []
        streams = ["alpha beta gamma", "alpha beta delta", "alpha beta gamma", None]
        for row_id, stream in enumerate(streams):
            row = {"record_id": f"r3-synthetic-{row_id}", "signal_stream": stream}
            for feature in EXPANDED_FEATURES:
                if feature == "cycle_code":
                    row[feature] = 47 + row_id
                elif feature in NUMERIC_FEATURES:
                    row[feature] = float(row_id + 1)
                elif feature in MIXED_COUNT_FEATURES:
                    row[feature] = str(row_id + 1)
                elif feature == "telemetry_flag":
                    row[feature] = "active" if row_id % 2 == 0 else "inactive"
                else:
                    row[feature] = f"fixed-{feature}"
            rows.append(row)
        return pd.DataFrame(rows, columns=["record_id", *EXPANDED_FEATURES, "signal_stream"])

    def test_execute_payload_receipt_readback_and_independent_verification(self) -> None:
        feature_frame = self._synthetic_training_frame()
        with tempfile.TemporaryDirectory(prefix="e79-r3-execute-e2e-") as temp:
            root = Path(temp)
            train_path = root / runner.TRAIN_REL
            train_path.parent.mkdir(parents=True)
            feature_frame.to_csv(train_path, index=False)
            train_sha = sha256_file(train_path)

            # Synthetic fitted preprocessing states exercise the real transforms;
            # these are not official or estimator training runs.
            structured_state = positional.fit_c2b_structured_state(feature_frame.loc[:, list(cgeom.CLASSIFICATION_FEATURES)])
            scaler = positional.fit_c2b_scaler(structured_state, feature_frame)
            encoder = positional.fit_c2b_ohe(structured_state, feature_frame)
            vectorizer = positional.fit_c2b_tfidf(feature_frame)
            states = {"c2b_structured": structured_state, "c2b_scaler": scaler, "c2b_ohe": encoder}
            model = SimpleNamespace(text_space=SimpleNamespace(vectorizer=vectorizer))

            features = feature_frame.loc[:, list(cgeom.CLASSIFICATION_FEATURES)].copy()
            structured = positional.transform_c2b_structured(structured_state, scaler, encoder, features)
            text, missing = positional.transform_c2b_text(vectorizer, features)
            documents, _ = classification_models._text_values(features["signal_stream"])
            position, _ = positional.build_position_block(documents, vectorizer, shuffled=False)
            base = sparse.hstack(
                (structured, text, sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))),
                format="csr", dtype=np.float64,
            )
            ids = feature_frame["record_id"].astype(str).to_numpy(dtype=str)
            expected = {
                "base_shape": list(base.shape),
                "base_nnz": int(base.nnz),
                "base_logical_sha256": sha256_csr(base),
                "position_shape": list(position.shape),
                "position_nnz": int(position.nnz),
                "position_logical_sha256": sha256_csr(position),
                "ordered_train_ids_sha256": sha256_array(ids),
            }
            expected_compatibility = {
                "status": "PASS_A02_FULLTRAIN_TFIDF_FEATURE_COMPATIBILITY",
                "train_sha256": train_sha,
                "train_rows": len(feature_frame),
                "vocabulary_sha256": "synthetic-vocabulary",
                "idf_sha256": "synthetic-idf",
                "fit_documents_sha256": "synthetic-documents",
                "fit_row_positions_sha256": "synthetic-row-positions",
                "target_column_requested_or_loaded": False,
                "uses_test_or_sample": False,
            }
            saved_units = {}
            for key, state in states.items():
                summary = cgeom._state_summary(state)
                saved_units[key] = {
                    "artifact_path": f"synthetic_saved_states/{key}.joblib",
                    "state_sha256": runner.canonical_sha256(summary),
                }
            attestation = {
                "attestation_sha256": "synthetic-attestation-digest-is-replaced-below",
                "current_source_sha256": {},
                "saved_inputs_and_fitted_state_receipts": {
                    "immutable_file_sha256": {runner.TRAIN_REL: train_sha},
                    "saved_units": saved_units,
                    "matrix_reference": expected,
                },
                "prepared_from_commit": "a" * 40,
            }
            attestation["attestation_sha256"] = runner.canonical_sha256({
                key: value for key, value in attestation.items() if key != "attestation_sha256"
            })
            resource = {
                "stage": "synthetic-only",
                "available_ram_bytes": 3 * 1024**3,
                "incremental_memory_estimate_bytes": runner.INCREMENTAL_ESTIMATE_BYTES,
                "required_free_reserve_bytes": runner.FREE_RESERVE_BYTES,
                "projected_reserve_bytes": 2 * 1024**3,
                "gate_pass": True,
            }

            def fake_load(path: Path) -> Any:
                candidate = Path(path)
                if candidate == root / runner.A02_MODEL_REL:
                    return model
                for key, record in saved_units.items():
                    if candidate == root / record["artifact_path"]:
                        return states[key]
                raise AssertionError(f"Unexpected synthetic saved-state read: {candidate}")

            with patch.multiple(
                runner,
                EXPECTED_ROWS=len(feature_frame),
                EXPECTED_CYCLES=(47, 48, 49, 50),
                TRAIN_SHA256=train_sha,
                BASE_SHAPE=tuple(expected["base_shape"]),
                BASE_NNZ=expected["base_nnz"],
                BASE_SHA256=expected["base_logical_sha256"],
                POSITION_SHAPE=tuple(expected["position_shape"]),
                POSITION_NNZ=expected["position_nnz"],
                POSITION_SHA256=expected["position_logical_sha256"],
                ORDERED_TRAIN_IDS_SHA256=expected["ordered_train_ids_sha256"],
            ), patch.object(runner.joblib, "load", side_effect=fake_load), \
                 patch.object(runner, "read_json", return_value={"a02_vectorizer_compatibility": expected_compatibility}), \
                 patch.object(cgeom, "audit_c2b_vectorizer_compatibility", return_value=expected_compatibility), \
                 patch.object(runner, "_resource_snapshot", return_value=resource):
                payload = runner._execute_feature_only_replay(
                    root,
                    attestation,
                    {"head": "b" * 40, "parent": "a" * 40},
                    [{"path": "synthetic-lock", "acquired": True}],
                    resource,
                )

            self.assertEqual(payload["status"], "PASS_SAVED_STATE_BASE_AND_POSITION_MATRIX_PARITY")
            self.assertTrue(all(payload["matrices"]["checks"].values()))
            self.assertEqual(payload["matrices"]["base"]["logical_sha256"], expected["base_logical_sha256"])
            self.assertEqual(payload["matrices"]["position"]["logical_sha256"], expected["position_logical_sha256"])
            self.assertEqual(payload["train_columns_requested"], canonical_feature_columns(cgeom.CLASSIFICATION_FEATURES))
            self.assertFalse(payload["target_column_requested_or_loaded"])
            self.assertFalse(payload["test_or_sample_loaded"])
            self.assertEqual(payload["estimator_fit_calls"], 0)

            # The independent verifier reads a disk-backed synthetic attestation,
            # exact source/input manifests, and the persisted JSON receipt.
            source_manifest = {}
            for relative in (
                "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py",
                "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v3.py",
                "experiments/optimization_v3/cls_pos_global_strength_e79_v3/protocol.json",
                "reports/optimization_v3/cls_pos_global_strength_e79_v3/CORRECTION_NOTE.md",
            ):
                source_path = ROOT / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
                source_manifest[relative] = sha256_file(destination)
            auth_source = ROOT / runner.AUTH_PATH
            auth_destination = root / runner.AUTH_PATH
            auth_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(auth_source, auth_destination)
            marker = root / "synthetic_inputs" / "state_marker.json"
            marker.parent.mkdir(parents=True)
            marker.write_text('{"synthetic":true}\n', encoding="utf-8")
            immutable_manifest = {
                runner.TRAIN_REL: train_sha,
                "synthetic_inputs/state_marker.json": sha256_file(marker),
            }
            attestation_body = {
                "schema_version": 1,
                "attestation_id": "SYNTHETIC_E79_R3_E2E",
                "status": "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R3_PENDING_REPLAY",
                "authorization": {"path": runner.AUTH_PATH, "sha256": runner.AUTH_SHA256},
                "prepared_from_commit": "a" * 40,
                "current_source_sha256": source_manifest,
                "saved_inputs_and_fitted_state_receipts": {
                    "immutable_file_sha256": immutable_manifest,
                    "matrix_reference": expected,
                    "preserved_r2_failed_attempt": {
                        "status": "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY",
                        "replay_receipt_file_sha256": "1a001f7843dfa19356adab62b0eadd21aecf1b0be8fa83f3db63a9a354c68ff9",
                        "replay_receipt_internal_digest_present": True,
                        "replay_receipt_internal_sha256": "7245c94a097f3440fa43938b2d798646f7d31d2f1324354144867cae54762031",
                        "matrix_digests_persisted": False,
                    },
                },
            }
            attestation_body["attestation_sha256"] = independent_verifier._canonical_sha(attestation_body)
            attestation_path = root / independent_verifier.ATTESTATION_REL
            attestation_path.parent.mkdir(parents=True, exist_ok=True)
            attestation_path.write_text(
                json.dumps(attestation_body, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            payload["source_compatibility_attestation_sha256"] = attestation_body["attestation_sha256"]
            payload["source_sha256"] = source_manifest
            payload["input_sha256"] = immutable_manifest
            payload["source_commit_parent"] = attestation_body["prepared_from_commit"]
            receipt_path = root / independent_verifier.REPLAY_RECEIPT_REL
            persisted = runner.write_and_readback_receipt(receipt_path, payload)
            verified = independent_verifier.verify_replay_receipt(
                root, expected_matrix=expected,
            )
            self.assertEqual(persisted["receipt_sha256"], verified["replay_receipt_sha256"])
            self.assertEqual(verified["status"], "PASS_INDEPENDENT_DATA_FREE_REPLAY_VERIFICATION")
            with self.assertRaises(ReplayError):
                runner.write_and_readback_receipt(receipt_path, payload)
            corrupted = dict(persisted)
            corrupted["train_rows"] += 1
            receipt_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaises(independent_verifier.VerificationError):
                independent_verifier.verify_replay_receipt(root, expected_matrix=expected)

    def test_recursive_symtable_scan_handles_comprehension_locals_and_catches_old_digest_names(self) -> None:
        for relative in (
            "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py",
            "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v3.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertEqual(runner.unresolved_global_references(source, filename=relative), [])
        self.assertEqual(
            runner.unresolved_global_references("def okay(rows): return [value for value in rows if value > 0]"),
            [],
        )
        missing = runner.unresolved_global_references("def broken(): return base_sha, position_sha")
        self.assertEqual({item["name"] for item in missing}, {"base_sha", "position_sha"})

    def test_actual_classification_feature_list_is_canonicalized_before_csv_read(self) -> None:
        columns = canonical_feature_columns(cgeom.CLASSIFICATION_FEATURES)
        self.assertIn("cycle_code", cgeom.CLASSIFICATION_FEATURES)
        self.assertEqual(len(columns), len(set(columns)))
        self.assertEqual(columns.count("cycle_code"), 1)
        frame_to_write = self._synthetic_csv_frame(columns)
        with tempfile.TemporaryDirectory(prefix="e79-r2-synthetic-csv-") as temp:
            path = Path(temp) / "synthetic_train.csv"
            frame_to_write.to_csv(path, index=False)
            loaded, requested = read_feature_csv(path, cgeom.CLASSIFICATION_FEATURES)
        self.assertEqual(len(requested), len(set(requested)))
        self.assertEqual(set(loaded.columns), set(columns))
        self.assertEqual(len(loaded.columns), len(columns))
        self.assertNotIn("artifact_signal", loaded.columns)
        self.assertNotIn("threat_rank", loaded.columns)

    def test_feature_csv_rejects_missing_extra_and_target_columns(self) -> None:
        columns = canonical_feature_columns(cgeom.CLASSIFICATION_FEATURES)
        frame = self._synthetic_csv_frame(columns)
        with self.assertRaises(ReplayError):
            validate_feature_frame_columns(frame.drop(columns=["cycle_code"]), columns)
        with self.assertRaises(ReplayError):
            validate_feature_frame_columns(frame.assign(unexpected_extra=1), columns)
        with self.assertRaises(ReplayError):
            validate_feature_frame_columns(frame.assign(artifact_signal=1), columns)
        with self.assertRaises(ReplayError):
            validate_feature_frame_columns(frame.assign(threat_rank=1), columns)
        with self.assertRaises(ReplayError):
            canonical_feature_columns([*cgeom.CLASSIFICATION_FEATURES, "artifact_signal"])
        with self.assertRaises(ReplayError):
            canonical_feature_columns([*cgeom.CLASSIFICATION_FEATURES, "threat_rank"])
        with tempfile.TemporaryDirectory(prefix="e79-r2-missing-csv-column-") as temp:
            path = Path(temp) / "missing.csv"
            frame.drop(columns=["cycle_code"]).to_csv(path, index=False)
            with self.assertRaises(ReplayError):
                read_feature_csv(path, cgeom.CLASSIFICATION_FEATURES)

    def test_synthetic_top_level_matrix_assembly_matches_expected_hashes(self) -> None:
        structured = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        text = sparse.csr_matrix(np.asarray([[0.5], [0.0], [0.25]]))
        missing = np.asarray([False, True, False])
        position = sparse.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0], [0.25, 0.75]]))
        ids = np.asarray(["syn-a", "syn-b", "syn-c"], dtype=str)
        expected_base = sparse.hstack(
            (structured, text, sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
            format="csr", dtype=np.float64,
        )
        expected = {
            "base_shape": list(expected_base.shape),
            "base_nnz": int(expected_base.nnz),
            "base_logical_sha256": sha256_csr(expected_base),
            "position_shape": list(position.shape),
            "position_nnz": int(position.nnz),
            "position_logical_sha256": sha256_csr(position),
            "ordered_train_ids_sha256": sha256_array(ids),
        }
        base, rebuilt_position, ids_sha, checks, actual = assemble_and_validate_matrices(
            structured, text, missing, position, ids, expected,
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(sha256_csr(base), expected["base_logical_sha256"])
        self.assertEqual(sha256_csr(rebuilt_position), expected["position_logical_sha256"])
        self.assertEqual(ids_sha, expected["ordered_train_ids_sha256"])
        self.assertEqual(actual["base"]["logical_sha256"], expected["base_logical_sha256"])
        self.assertEqual(actual["position"]["logical_sha256"], expected["position_logical_sha256"])
        corrupted = dict(expected)
        corrupted["base_logical_sha256"] = "0" * 64
        with self.assertRaises(ReplayError):
            assemble_and_validate_matrices(structured, text, missing, position, ids, corrupted)

    def test_changed_helper_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e79-r2-helper-hash-") as temp:
            root = Path(temp)
            helper = root / "helper.py"
            helper.write_text("def transform(x): return x + 1\n", encoding="utf-8")
            expected = sha256_file(helper)
            _hash_map(root, {"helper.py": expected}, "fixture source")
            helper.write_text("def transform(x): return x + 2\n", encoding="utf-8")
            with self.assertRaises(ReplayError):
                _hash_map(root, {"helper.py": expected}, "fixture source")

    def test_changed_saved_artifact_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e79-r2-artifact-hash-") as temp:
            root = Path(temp)
            artifact = root / "saved.joblib"
            artifact.write_bytes(b"frozen artifact bytes")
            expected = sha256_file(artifact)
            _hash_map(root, {"saved.joblib": expected}, "fixture artifact")
            artifact.write_bytes(b"changed artifact bytes")
            with self.assertRaises(ReplayError):
                _hash_map(root, {"saved.joblib": expected}, "fixture artifact")

    def test_only_exact_receipt_source_variances_are_allowed(self) -> None:
        for key in ("c2b_structured", "c2b_scaler", "c2b_ohe"):
            expected = EXPLICIT_SOURCE_VARIANCES[key]
            receipt = {"source_sha256": {path: item["receipt_expected"] for path, item in expected.items()}}
            audit = _receipt_source_audit(ROOT, key, receipt)
            self.assertTrue(all(item["authorized_variance"] for item in audit.values()))
        receipt = {"source_sha256": {
            CGEOM_SCRIPT_REL: EXPLICIT_SOURCE_VARIANCES["c2b_structured"][CGEOM_SCRIPT_REL]["receipt_expected"],
            CGEOM_TEST_REL: EXPLICIT_SOURCE_VARIANCES["c2b_structured"][CGEOM_TEST_REL]["receipt_expected"],
            "scripts/optimization_v3_a02_coefficient_geometry_v1.py": "0" * 64,
        }}
        with self.assertRaises(ReplayError):
            _receipt_source_audit(ROOT, "c2b_structured", receipt)

    def test_replay_source_has_no_estimator_fit_or_prediction_calls(self) -> None:
        script = ROOT / "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py"
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_methods = {"fit", "fit_transform", "predict", "predict_proba", "decision_function"}
        observed = {
            call.func.attr for call in _calls(tree)
            if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden_methods
        }
        self.assertEqual(observed, set())
        forbidden_names = {"union_document_frequency", "make_penalty_maps", "fit_global_strength_gamma"}
        called_names = {call.func.id for call in _calls(tree) if isinstance(call.func, ast.Name)}
        self.assertFalse(forbidden_names & called_names)
        self.assertNotIn("Test.csv", source)
        self.assertNotIn("SampleSubmission", source)

    def test_official_read_uses_canonical_request_and_blocks_target_names(self) -> None:
        script = ROOT / "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        read_calls = [
            call for call in _calls(tree)
            if isinstance(call.func, ast.Attribute) and call.func.attr == "read_csv"
        ]
        self.assertEqual(len(read_calls), 1)
        usecols = next(kw.value for kw in read_calls[0].keywords if kw.arg == "usecols")
        self.assertIsInstance(usecols, ast.Name)
        self.assertEqual(usecols.id, "columns")
        source = script.read_text(encoding="utf-8")
        self.assertIn('columns = canonical_feature_columns(feature_columns)', source)
        self.assertIn('columns = list(dict.fromkeys(raw))', source)
        self.assertIn('"artifact_signal", "threat_rank"', source)

    def test_compatibility_audit_has_no_fit_or_target_read(self) -> None:
        cgeom_path = ROOT / CGEOM_SCRIPT_REL
        source = cgeom_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            item for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "audit_c2b_vectorizer_compatibility"
        )
        self.assertFalse(any(
            isinstance(call.func, ast.Attribute) and call.func.attr in {"fit", "fit_transform"}
            for call in _calls(node)
        ))
        audit_source = ast.get_source_segment(source, node) or ""
        self.assertNotIn('"artifact_signal"', audit_source)
        self.assertIn('"record_id", "cycle_code", "signal_stream"', audit_source)


if __name__ == "__main__":
    unittest.main()
