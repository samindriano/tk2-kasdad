"""Read-only integration checks for the complete Round2 artifacts.

These checks become active once MAIN has produced the frozen results bundle.
The verifier may write only its canonical Round2 verification receipt.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import nbformat

from src.utils.paths import PROJECT_ROOT
from scripts.verify_classification_round2 import (
    REPORT_DIR,
    RESULT_PATH,
    _can_preserve_representative_refit,
    verify_results,
)


R1_ROOT = PROJECT_ROOT / "experiments/runs/classification_round1_v1"
R1_REPORT = PROJECT_ROOT / "reports/classification_round1"
R2_ARTIFACTS_READY = (
    RESULT_PATH.is_file()
    and (REPORT_DIR / "routing_diagnostic.json").is_file()
    and (PROJECT_ROOT / "experiments/classification_round2/freeze.json").is_file()
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_round1_paths():
    paths = [
        PROJECT_ROOT / "experiments/classification_round1/freeze.json",
        PROJECT_ROOT / "experiments/classification_round1/protocol.json",
        R1_REPORT / "results.json",
    ]
    paths.extend(R1_REPORT / "fit_audits" / f"fit{end}.json" for end in (55, 59, 63, 67, 71, 75))
    paths.extend(
        R1_ROOT / f"fit{end}__{model}.npz"
        for end in (55, 59, 63, 67, 71, 75)
        for model in ("C1_A", "C1_B", "C2_A", "C2_B")
    )
    return paths


class RepresentativeRefitReceiptPredicateTests(unittest.TestCase):
    def test_stale_round2_vector_hash_drops_old_pass_receipt(self):
        old = {
            "freeze_signature": "frozen",
            "original_vector_set_sha256": "round1-vectors",
            "round2_vector_set_sha256": "stale-round2-vectors",
            "representative_refit": {"status": "PASS", "fresh_model_fits": 12},
        }
        self.assertFalse(_can_preserve_representative_refit(
            old, "frozen", "round1-vectors", "current-round2-vectors"
        ))
        self.assertTrue(_can_preserve_representative_refit(
            old, "frozen", "round1-vectors", "stale-round2-vectors"
        ))
        self.assertFalse(_can_preserve_representative_refit(
            {**old, "freeze_signature": "another-freeze"},
            "frozen", "round1-vectors", "stale-round2-vectors",
        ))


@unittest.skipUnless(R2_ARTIFACTS_READY, "complete frozen Round2 results are not present yet")
class ClassificationRound2ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = verify_results()

    def test_freeze_vectors_thresholds_routes_and_metrics_are_complete(self):
        self.assertEqual(self.receipt["status"], "PASS")
        counts = self.receipt["counts"]
        expected = {
            "fresh_round2_fits_attested": 72,
            "model_vector_groups": 48,
            "derived_route_groups": 12,
            "candidate_probability_groups": 60,
            "linear_component_groups": 42,
            "round1_outer_control_probability_and_state_matches": 24,
            "round1_control_inner_probability_and_state_reuses": 24,
            "routing_diagnostic_window_hash_matches": 24,
            "threshold_groups_recomputed": 60,
            "policy_cut_records_recomputed": 240,
            "window_policy_cut_records_verified": 320,
            "window_jobs_verified": 80,
            "job_policy_metric_records_verified": 320,
            "per_class_records_verified": 640,
            "per_cycle_metric_records_verified": 1920,
            "pooled_and_equal_cycle_aggregates_verified": 80,
        }
        for key, value in expected.items():
            with self.subTest(counter=key):
                self.assertEqual(counts[key], value)
        self.assertEqual(counts["pooled_cycle_rows_per_family_model"], 38588)
        self.assertEqual(self.receipt["modeling_or_refits_performed_by_verifier"], 0)
        self.assertLessEqual(self.receipt["maximum_logit_probability_reconstruction_error"], 1e-12)

    def test_independent_scientific_subgroups_and_transform_fingerprints(self):
        counts = self.receipt["counts"]
        self.assertEqual(counts["scientific_subgroups_per_job_policy"], 20)
        self.assertEqual(counts["job_policy_subgroup_metrics_verified"], 320 * 20)
        self.assertEqual(counts["aggregate_subgroup_metrics_verified"], 80 * 20)
        self.assertEqual(counts["structured_transform_comparisons"], 6 * 5)
        self.assertEqual(counts["text_transform_comparisons"], 6 * 4)
        self.assertIn("raw", self.receipt["subgroup_masks_source"])
        self.assertIn("sklearn", self.receipt["threshold_method"])

    def test_round1_checkpoint_stays_byte_identical_and_receipt_repeats(self):
        before = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in protected_round1_paths()}
        first_receipt = dict(self.receipt)
        second_receipt = verify_results()
        after = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in protected_round1_paths()}
        self.assertEqual(before, after)
        self.assertEqual(first_receipt, second_receipt)
        self.assertGreaterEqual(self.receipt["freeze_manifest_files_verified"], 1)
        self.assertEqual(self.receipt["round1_freeze_signature"], self.receipt["identity"]["round1_freeze_signature"])

    def test_existing_representative_refit_receipt_is_not_fabricated(self):
        repeat_report = Path(tempfile.gettempdir()) / "tp2-cls2-repeat" / "results.json"
        receipt_path = REPORT_DIR / "verification.json"
        if not repeat_report.is_file():
            self.skipTest("representative 12-fit repeat bundle has not arrived yet")
        self.assertTrue(receipt_path.is_file())
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["original_vector_set_sha256"], self.receipt["original_vector_set_sha256"])
        self.assertEqual(stored["round2_vector_set_sha256"], self.receipt["round2_vector_set_sha256"])
        self.assertEqual(stored["representative_refit"]["status"], "PASS")
        self.assertEqual(stored["representative_refit"]["fresh_model_fits"], 12)
        self.assertEqual(stored["representative_refit"]["outer_probability_vectors_bit_identical"], 8)
        self.assertEqual(stored["representative_refit"]["variant_inner_probability_vectors_bit_identical"], 4)
        self.assertEqual(stored["representative_refit"]["control_inner_vector_reuses_bit_identical"], 4)
        self.assertEqual(stored["representative_refit"]["logit_component_arrays_bit_identical"], 35)
        self.assertEqual(stored["representative_refit"]["all_recipe_window_policy_cut_records_identical"], 80)
        self.assertEqual(stored["representative_refit"]["component_array_max_absolute_delta"], 0.0)


class ClassificationRound2NotebookTests(unittest.TestCase):
    def test_new_executed_notebook_contains_at_least_six_rendered_figures(self):
        notebook_path = PROJECT_ROOT / "notebooks/samuel/06_samuel_classification_round2.ipynb"
        if not notebook_path.is_file():
            self.skipTest("Round2 notebook has not been built yet")
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        execution_counts = [cell.execution_count for cell in code_cells]
        self.assertTrue(code_cells, "Round2 notebook has no executable cells")
        self.assertTrue(all(value is not None for value in execution_counts), "Not all code cells were executed")
        self.assertEqual(execution_counts, list(range(1, len(code_cells) + 1)),
                         "Round2 notebook execution counts are not sequential")
        figures = 0
        errors = []
        for cell in code_cells:
            for output in cell.outputs:
                if output.output_type == "error":
                    errors.append(output.ename + ": " + output.evalue)
                if output.output_type in {"display_data", "execute_result"} and (
                    "image/png" in output.data or "image/svg+xml" in output.data
                ):
                    figures += 1
        self.assertFalse(errors, f"Round2 notebook contains execution errors: {errors[:2]}")
        self.assertGreaterEqual(figures, 6, "Round2 notebook needs at least six rendered figures")


if __name__ == "__main__":
    unittest.main()
