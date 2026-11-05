from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
from scripts.optimization_v3_cls_pos_global_strength_e79_v2 import (
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
from src.models.optimization_v3_positional_text_pilot_v1 import sha256_array, sha256_csr


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _read_csv_usecols(tree: ast.AST) -> list[ast.AST]:
    calls = [
        call for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "read_csv"
    ]
    return [kw.value for call in calls for kw in call.keywords if kw.arg == "usecols"]


class E79R2CompatibilityReplayTests(unittest.TestCase):
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
        base, rebuilt_position, ids_sha, checks = assemble_and_validate_matrices(
            structured, text, missing, position, ids, expected,
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(sha256_csr(base), expected["base_logical_sha256"])
        self.assertEqual(sha256_csr(rebuilt_position), expected["position_logical_sha256"])
        self.assertEqual(ids_sha, expected["ordered_train_ids_sha256"])
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
        script = ROOT / "scripts/optimization_v3_cls_pos_global_strength_e79_v2.py"
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
        script = ROOT / "scripts/optimization_v3_cls_pos_global_strength_e79_v2.py"
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
