from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "src" / "models"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from cls_ppmi_context_correction_r1 import (
    CorrectionError,
    PRIMARY_ROWS,
    VIEW_NAMES,
    labels_to_map,
    score_correction,
    selection_binding,
    synthetic_prediction_fixture,
    synthetic_qa,
)
_SCRIPT_DIR = _ROOT / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from verify_cls_ppmi_context_correction_r1 import (
    synthetic_gate_verification,
    verify_in_memory,
)

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cls_ppmi_context_correction_r1.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("cls_ppmi_context_correction_r1_runner_module",
                                                      _RUNNER_PATH)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise RuntimeError("Could not load corrected scoring runner for selected-read fixture")
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)
read_selected_labels_fixture = _RUNNER._read_selected_labels_once


def make_scored(predictions=None, labels=None):
    if predictions is None or labels is None:
        predictions, labels = synthetic_prediction_fixture()
    result = score_correction(predictions, labels,
                              prelabel_verification_sha256="synthetic-prelabel")
    return predictions, labels, result


class MacroF1CorrectionTests(unittest.TestCase):
    def test_imbalanced_asymmetric_confusion_distinguishes_macro_from_binary_f1(self):
        y = np.asarray([0] * 100 + [1] * 10, dtype=np.int64)
        decision = np.asarray([0] * 90 + [1] * 10 + [0] * 5 + [1] * 5, dtype=np.int64)
        probability = np.where(decision == 1, 0.8, 0.2)
        scored = score_correction.__globals__["_metric_manual_confusion"](
            y, probability, np.full(len(y), 0.5))
        binary_f1 = float(f1_score(y, decision, zero_division=0))
        expected_macro = (180 / 195 + 10 / 25) / 2
        self.assertEqual(scored["confusion_matrix_labels_0_1"], [[90, 10], [5, 5]])
        self.assertAlmostEqual(scored["macro_f1"], expected_macro, places=14)
        self.assertAlmostEqual(scored["macro_f1"], f1_score(
            y, decision, labels=[0, 1], average="macro", zero_division=0), places=14)
        self.assertNotAlmostEqual(scored["macro_f1"], binary_f1, places=3)

    def test_perfect_constant_and_threshold_tie_fixtures(self):
        qa = synthetic_qa()
        self.assertEqual(qa["status"], "PASS")
        self.assertIn("perfect_prediction_macro_f1", qa["cases"])
        self.assertIn("constant_class_and_threshold_tie", qa["cases"])


class CorrectedViewTests(unittest.TestCase):
    def test_exact_six_views_source_rows_and_independent_metric_recompute(self):
        predictions, labels, scored = make_scored()
        self.assertEqual(set(scored["views"]), set(VIEW_NAMES))
        expected = {"E67_W4": 4_950, "E67_W5": 7_462, "E67_W6": 8_491,
                    "E67_H12": 20_903, "E75_W6": 8_491, "PRIMARY_UNIQUE": 20_903}
        self.assertEqual({name: view["rows"] for name, view in scored["views"].items()}, expected)
        self.assertEqual(scored["views"]["E67_W4"]["prediction_source"], "E67")
        self.assertEqual(scored["views"]["E67_W5"]["prediction_source"], "E67")
        self.assertEqual(scored["views"]["E67_W6"]["prediction_source"], "E67")
        self.assertEqual(scored["views"]["E75_W6"]["prediction_source"], "E75")
        self.assertEqual(scored["views"]["PRIMARY_UNIQUE"]["prediction_source"],
                         "E67_cycles_68_75 + E75_cycles_76_79")
        self.assertEqual(set(scored["subgroups_by_view"]), set(VIEW_NAMES))
        self.assertEqual(scored["window_overlap_checks"]["E67_H12_and_PRIMARY_UNIQUE_shared_ids"], 20_903)
        verified = verify_in_memory(predictions, labels, scored, anchors={})
        self.assertEqual(verified["status"], "PASS")
        self.assertEqual(verified["views_recomputed"], 6)
        self.assertEqual(verified["subgroup_views_recomputed"], 6)
        self.assertEqual(verified["additional_target_reads"], 0)

    def test_e75_gate_uses_real_minus_shuffled_not_shuffled_minus_champion(self):
        predictions, labels = synthetic_prediction_fixture()
        e75 = predictions[75]
        positive_present = np.asarray([
            i for i, record_id in enumerate(e75["score_record_ids"].astype(str))
            if e75["present_mask"][i] and labels[record_id] == 1
        ], dtype=np.int64)
        self.assertGreater(len(positive_present), 300)
        e75["p_candidate_real"][positive_present[:100]] = 0.1
        e75["p_control_shuffled"][positive_present[:220]] = 0.1
        scored = score_correction(predictions, labels,
                                  prelabel_verification_sha256="synthetic-prelabel")
        e75_metrics = scored["views"]["E75_W6"]["metrics"]
        self.assertLess(e75_metrics["gains"]["candidate_real_minus_champion_macro_f1"], 0.0)
        self.assertGreater(e75_metrics["gains"]["candidate_real_minus_control_shuffled_macro_f1"], 0.0)
        self.assertLess(e75_metrics["gains"]["control_shuffled_minus_champion_macro_f1"], 0.0)
        self.assertTrue(scored["gates"]["mechanism"]["checks"]["e75_real_minus_shuffled_nonnegative"])
        self.assertEqual(scored["gates"]["mechanism"]["status"], "SEPARATED_FROM_SHUFFLED_CONTROL")
        direction = synthetic_gate_verification()
        self.assertTrue(direction["signs_differ"])
        self.assertEqual(direction["intended_gate_source"], "real_minus_shuffled")

    def test_subgroup_guard_aggregates_all_six_views(self):
        predictions, labels = synthetic_prediction_fixture()
        e67 = predictions[67]
        selected: list[int] = []
        for i, record_id in enumerate(e67["score_record_ids"].astype(str)):
            if (68 <= int(e67["score_cycles"][i]) <= 71
                    and e67["present_mask"][i]
                    and str(e67["entry_friction"][i]) == "0"
                    and labels[record_id] == 1):
                selected.append(i)
                if len(selected) == 80:
                    break
        self.assertEqual(len(selected), 80)
        e67["p_candidate_real"][selected] = 0.1
        scored = score_correction(predictions, labels,
                                  prelabel_verification_sha256="synthetic-prelabel")
        w4_group = scored["subgroups_by_view"]["E67_W4"]["entry_friction=0"]
        self.assertTrue(w4_group["supported_for_gate"])
        self.assertGreater(w4_group["candidate_macro_f1_harm"], 0.01)
        self.assertFalse(scored["gates"]["scientific"]["checks"][
            "all_view_supported_subgroup_macro_f1_harm_le_0_01"])
        self.assertEqual(set(scored["gates"]["scientific"]["supported_groups_by_view"]), set(VIEW_NAMES))
        self.assertEqual(len(scored["subgroups_by_view"]), 6)

    def test_poisoned_label_id_and_duplicate_id_fail_closed(self):
        predictions, labels = synthetic_prediction_fixture()
        selection = selection_binding(predictions)
        ids = selection["selected_ids"]
        values = np.asarray([labels[record_id] for record_id in ids], dtype=np.int64)
        with self.assertRaises(CorrectionError):
            labels_to_map(np.asarray([ids[0], ids[0]]), np.asarray([0, 1]))
        poisoned = dict(labels)
        removed = next(iter(poisoned))
        poisoned.pop(removed)
        poisoned["POISONED-ID"] = 0
        with self.assertRaises(CorrectionError):
            score_correction(predictions, poisoned,
                             prelabel_verification_sha256="synthetic-prelabel")
        self.assertEqual(len(values), PRIMARY_ROWS)


class SelectedLabelReadTests(unittest.TestCase):
    def test_reader_materializes_only_selected_csv_rows_and_identity_checks_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.csv"
            rows = [
                ["record_id", "artifact_signal", "other"],
                ["r0", "not-binary", "x"],
                ["r1", "1", "x"],
                ["r2", "not-binary", "x"],
                ["r3", "not-binary", "x"],
                ["r4", "0", "x"],
                ["r5", "not-binary", "x"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            selected = {"selected_rows": np.asarray([1, 4], dtype=np.int64),
                        "selected_ids": np.asarray(["r1", "r4"], dtype=str)}
            ids, labels = read_selected_labels_fixture(path, selected, expected_rows=2)
            self.assertEqual(ids.tolist(), ["r1", "r4"])
            self.assertEqual(labels.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
