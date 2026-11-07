from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cls_ppmi_context_recovery_r2 as recovery  # noqa: E402
from verify_cls_ppmi_context_recovery_r2 import (  # noqa: E402
    IndependentVerificationError,
    dual_e75_gate_readback,
    verify_in_memory_recovery_r2,
)


class RecoveryAnchorTests(unittest.TestCase):
    def setUp(self):
        self.predictions, self.labels = recovery.r1.core.synthetic_prediction_fixture()
        self.scored = recovery.r1.core.score_correction(
            self.predictions, self.labels, prelabel_verification_sha256="synthetic-r2-test")
        self.anchors = recovery._production_schema_anchors(self.scored)

    def test_exact_production_anchor_schema_and_gain_mapping(self):
        self.assertEqual(set(self.anchors), {"PRIMARY_UNIQUE", "E67_H12", "E75_W6"})
        for values in self.anchors.values():
            self.assertEqual(set(values), {
                "champion", "candidate_real", "control_shuffled", "gain_real_vs_champion",
            })
        verification = verify_in_memory_recovery_r2(
            self.predictions, self.labels, self.scored, self.anchors)
        gain_anchor = verification["macro_f1_root_anchors"]["PRIMARY_UNIQUE"]["gain_real_vs_champion"]
        self.assertTrue(gain_anchor["pass"])
        self.assertEqual(gain_anchor["metric"], "candidate_real_minus_champion_macro_f1")

    def test_mismatched_gain_and_unknown_anchor_keys_fail_closed(self):
        mismatched = {view: dict(values) for view, values in self.anchors.items()}
        mismatched["E75_W6"]["gain_real_vs_champion"] += 0.01
        with self.assertRaises(IndependentVerificationError):
            verify_in_memory_recovery_r2(self.predictions, self.labels, self.scored, mismatched)
        unknown = {view: dict(values) for view, values in self.anchors.items()}
        unknown["E75_W6"]["surprise"] = 0.0
        with self.assertRaises(IndependentVerificationError):
            verify_in_memory_recovery_r2(self.predictions, self.labels, self.scored, unknown)

    def test_original_and_r1_amended_e75_gates_are_separate(self):
        e75 = self.predictions[75]
        positive_present = np.asarray([
            index for index, record_id in enumerate(e75["score_record_ids"].astype(str))
            if e75["present_mask"][index] and self.labels[record_id] == 1
        ], dtype=np.int64)
        e75["p_candidate_real"][positive_present[:100]] = 0.1
        e75["p_control_shuffled"][positive_present[:220]] = 0.1
        scored = recovery.r1.core.score_correction(
            self.predictions, self.labels, prelabel_verification_sha256="synthetic-r2-gates")
        result = dual_e75_gate_readback(scored)
        self.assertEqual(result["original_root_campaign"]["comparison"],
                         "control_shuffled_minus_champion_macro_f1")
        self.assertEqual(result["r1_amended_comparison"]["comparison"],
                         "candidate_real_minus_control_shuffled_macro_f1")
        self.assertNotEqual(result["original_root_campaign"]["nonnegative"],
                            result["r1_amended_comparison"]["nonnegative"])

    def test_commit_read_guard_blocks_before_reader(self):
        calls: list[str] = []

        def reject_uncommitted(_intent):
            raise recovery.RecoveryRunnerError("intent/source set not committed")

        def forbidden_reader(*_args, **_kwargs):
            calls.append("read")
            return None

        with self.assertRaises(recovery.RecoveryRunnerError):
            recovery._run_read_guard(
                {}, Path("unused.csv"), {},
                committed_check=reject_uncommitted, reader=forbidden_reader)
        self.assertEqual(calls, [])

    def test_full_synthetic_score_verify_serialization_and_six_view_guards(self):
        qa = recovery.synthetic_recovery_lifecycle()
        lifecycle = qa["complete_synthetic_lifecycle"]
        self.assertEqual(qa["status"], "PASS")
        self.assertEqual(lifecycle["score_views"], 6)
        self.assertTrue(lifecycle["all_view_guards_recomputed"])
        self.assertEqual(lifecycle["serialization_roundtrip_signature"], "PASS")
        self.assertEqual(lifecycle["report_generation"], "PASS")
        self.assertTrue(lifecycle["mismatched_anchor_rejected"])
        self.assertTrue(lifecycle["unknown_anchor_rejected"])
        self.assertEqual(qa["pre_read_guard_case"], "PASS")


if __name__ == "__main__":
    unittest.main()
