from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from scripts.optimization_v3_cls_pos_global_strength_e79_v1 import (
    CGEOM_SCRIPT_REL,
    CGEOM_TEST_REL,
    EXPLICIT_SOURCE_VARIANCES,
    ROOT,
    ReplayError,
    _hash_map,
    _receipt_source_audit,
    sha256_file,
)


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _read_csv_usecols(tree: ast.AST) -> list[ast.AST]:
    calls = [
        call for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "read_csv"
    ]
    return [
        kw.value for call in calls for kw in call.keywords if kw.arg == "usecols"
    ]


class E79CompatibilityReplayTests(unittest.TestCase):
    def test_changed_helper_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e79-helper-hash-") as temp:
            root = Path(temp)
            helper = root / "helper.py"
            helper.write_text("def transform(x): return x + 1\n", encoding="utf-8")
            expected = sha256_file(helper)
            _hash_map(root, {"helper.py": expected}, "fixture source")
            helper.write_text("def transform(x): return x + 2\n", encoding="utf-8")
            with self.assertRaises(ReplayError):
                _hash_map(root, {"helper.py": expected}, "fixture source")

    def test_changed_saved_artifact_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e79-artifact-hash-") as temp:
            root = Path(temp)
            artifact = root / "saved.joblib"
            artifact.write_bytes(b"frozen artifact bytes")
            expected = sha256_file(artifact)
            _hash_map(root, {"saved.joblib": expected}, "fixture artifact")
            artifact.write_bytes(b"changed artifact bytes")
            with self.assertRaises(ReplayError):
                _hash_map(root, {"saved.joblib": expected}, "fixture artifact")

    def test_only_three_explicit_receipt_source_variances_are_allowed(self) -> None:
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

    def test_replay_script_has_no_estimator_fit_or_prediction_calls(self) -> None:
        script = ROOT / "scripts/optimization_v3_cls_pos_global_strength_e79_v1.py"
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_methods = {"fit", "fit_transform", "predict", "predict_proba", "decision_function"}
        observed = {
            call.func.attr for call in _calls(tree)
            if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden_methods
        }
        self.assertEqual(observed, set())
        forbidden_names = {"union_document_frequency", "make_penalty_maps", "fit_global_strength_gamma"}
        called_names = {
            call.func.id for call in _calls(tree) if isinstance(call.func, ast.Name)
        }
        self.assertFalse(forbidden_names & called_names)
        self.assertNotIn("Test.csv", source)
        self.assertNotIn("SampleSubmission", source)

    def test_official_read_is_feature_only_and_never_names_target(self) -> None:
        script = ROOT / "scripts/optimization_v3_cls_pos_global_strength_e79_v1.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        usecols = _read_csv_usecols(tree)
        self.assertEqual(len(usecols), 1)
        value = usecols[0]
        if isinstance(value, ast.Name) and value.id == "train_columns":
            assignment = next(
                item for item in ast.walk(tree)
                if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == value.id for target in item.targets)
            )
            value = assignment.value
        self.assertIsInstance(value, ast.List)
        literal_values = [
            item.value for item in value.elts if isinstance(item, ast.Constant)
        ]
        self.assertEqual(literal_values[:2], ["record_id", "cycle_code"])
        self.assertFalse({"artifact_signal", "threat_rank"} & set(literal_values))
        self.assertTrue(any(
            isinstance(item, ast.Starred)
            and isinstance(item.value, ast.Attribute)
            and item.value.attr == "CLASSIFICATION_FEATURES"
            for item in value.elts
        ))

    def test_compatibility_audit_does_not_fit_or_read_target(self) -> None:
        cgeom = ROOT / "scripts/optimization_v3_a02_cgeom_benchmark_v1.py"
        source = cgeom.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            item for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "audit_c2b_vectorizer_compatibility"
        )
        calls = _calls(node)
        self.assertFalse(any(
            isinstance(call.func, ast.Attribute) and call.func.attr in {"fit", "fit_transform"}
            for call in calls
        ))
        self.assertNotIn('"artifact_signal"', ast.get_source_segment(source, node) or "")
        self.assertIn('"record_id", "cycle_code", "signal_stream"', ast.get_source_segment(source, node) or "")


if __name__ == "__main__":
    unittest.main()
