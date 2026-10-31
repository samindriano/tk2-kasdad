"""Synthetic-only tests for the C09 sparse interaction adapter."""
from __future__ import annotations

import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.optimization_v3_batch1_c09 import (
    _fit_quantile_labels,
    _independent_group_guard_audit,
    _reserve_c09_budget,
)
from scripts import optimization_v3_batch1_c09 as c09_runner
from src.models.optimization_v3_batch1 import BatchIntegrityError
from src.models.optimization_v3_batch1_c09 import (
    C09TextEntryFrictionClassifier,
    append_reference_interaction,
    entry_friction_one_mask,
)


class C09SyntheticTests(unittest.TestCase):
    def test_interaction_is_sparse_reference_coded_and_appended(self) -> None:
        base = sparse.csr_matrix(np.asarray([[3.0, 0.0], [0.0, 4.0], [5.0, 6.0]]))
        text = sparse.csr_matrix(np.asarray([[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]]))
        ef1 = np.asarray([False, True, False])
        extended, interaction = append_reference_interaction(base, text, ef1)
        self.assertTrue(sparse.isspmatrix_csr(extended))
        self.assertTrue(sparse.isspmatrix_csr(interaction))
        self.assertEqual(extended.shape, (3, 4))
        np.testing.assert_array_equal(extended[:, :2].toarray(), base.toarray())
        np.testing.assert_array_equal(interaction.toarray(), [[0, 0], [3, 0], [0, 0]])
        self.assertEqual(interaction.nnz, 1)

    def test_entry_friction_parser_uses_exact_numeric_one(self) -> None:
        frame = pd.DataFrame({"entry_friction": [0, 1, "1.0", None, "", 2, "bad"]})
        mask, audit = entry_friction_one_mask(frame)
        np.testing.assert_array_equal(mask, [False, True, True, False, False, False, False])
        self.assertEqual(audit["ef1_rows"], 2)
        self.assertEqual(audit["ef0_rows"], 1)
        self.assertEqual(audit["missing_rows"], 2)
        self.assertEqual(audit["invalid_rows"], 1)
        self.assertEqual(audit["other_finite_rows"], 1)

    def test_fit_defined_quartiles_handle_ties_and_missing(self) -> None:
        labels, edges = _fit_quantile_labels(
            np.asarray([2.0, 2.0, 2.0, 2.0]),
            np.asarray([2.0, np.nan, 9.0]),
        )
        self.assertEqual(edges, [2.0])
        np.testing.assert_array_equal(labels, ["Q1", "missing", "Q1"])

    def test_independent_group_guard_recomputes_support_and_rejects_bad_denominator(self) -> None:
        actual = np.asarray([0] * 120 + [1] * 120, dtype=np.int8)
        baseline = np.where(actual == 1, 0.9, 0.1)
        candidate = baseline.copy()
        labels = {"entry_friction": np.full(240, "EF0", dtype=object)}
        group = {
            "n": 240,
            "negative_support": 120,
            "positive_support": 120,
            "supported": True,
            "baseline_macro_f1": 1.0,
            "candidate_macro_f1": 1.0,
            "macro_f1_harm": 0.0,
            "baseline_positive_recall": 1.0,
            "candidate_positive_recall": 1.0,
            "positive_recall_harm": 0.0,
            "gate_pass": True,
        }
        gate = {
            "supported_group_n_min": 200,
            "positive_and_negative_per_group_min": 30,
            "supported_group_F1_harm_max": 0.01,
            "positive_class_recall_harm_max": 0.02,
        }
        _, checks = _independent_group_guard_audit(
            actual, baseline, candidate, labels,
            {"entry_friction": {"EF0": group}}, gate,
        )
        self.assertTrue(checks["entry_friction:EF0 direct denominator and guard match"])
        malformed = {**group, "n": 239}
        _, bad_checks = _independent_group_guard_audit(
            actual, baseline, candidate, labels,
            {"entry_friction": {"EF0": malformed}}, gate,
        )
        self.assertFalse(bad_checks["entry_friction:EF0 direct denominator and guard match"])

    def test_freeze_check_rejects_blocked_preflight_even_when_identity_matches(self) -> None:
        blocked = {
            "status": "PREFLIGHT_BLOCKED_MEMORY_UNKNOWN_OR_OVER_LIMIT",
            "memory_forecast": {"status": "UNKNOWN_OR_OVER_LIMIT"},
            "identity": {"source": "same"},
            "frozen_identity_sha256": "same-signature",
        }
        with patch.object(c09_runner, "preflight", return_value=blocked), patch.object(
            c09_runner, "_read_json", return_value={"identity": {"source": "same"}, "signature": "same-signature"}
        ) as read_json:
            with self.assertRaisesRegex(ValueError, "preflight or memory forecast is not passing"):
                c09_runner._check_freeze()
        read_json.assert_not_called()

    def test_freeze_check_rejects_unknown_memory_even_when_preflight_status_passes(self) -> None:
        unknown_memory = {
            "status": "PREFLIGHT_PASS_READ_ONLY_NO_FIT_NO_SCORE",
            "memory_forecast": {"status": "UNKNOWN_OR_OVER_LIMIT"},
            "identity": {"source": "same"},
            "frozen_identity_sha256": "same-signature",
        }
        with patch.object(c09_runner, "preflight", return_value=unknown_memory), patch.object(
            c09_runner, "_read_json", return_value={"identity": {"source": "same"}, "signature": "same-signature"}
        ) as read_json:
            with self.assertRaisesRegex(ValueError, "preflight or memory forecast is not passing"):
                c09_runner._check_freeze()
        read_json.assert_not_called()

    def test_c09_budget_reservation_uses_two_fit_card_count_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "batch_ledger.jsonl"
            first = _reserve_c09_budget(ledger)
            second = _reserve_c09_budget(ledger)
            self.assertEqual(first["status"], "RESERVED")
            self.assertEqual(first["card_reservation"], 2)
            self.assertEqual(first["budget_key"], "C09:PILOT")
            self.assertEqual(second["status"], "EXISTING_RESERVATION")
            entries = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(entries), 1)

    def test_dispatcher_bridge_only_allows_canonical_read_only_preflight(self) -> None:
        queue, card = c09_runner._canonical_card()
        expected = {"status": "SYNTHETIC_PREFLIGHT_ONLY"}
        with patch.object(c09_runner, "preflight", return_value=expected) as preflight_mock:
            self.assertEqual(c09_runner.run_card(card, queue, preflight_only=True), expected)
        preflight_mock.assert_called_once_with()

        with patch.object(c09_runner, "preflight") as preflight_mock:
            with self.assertRaisesRegex(BatchIntegrityError, "explicit --write-freeze followed by --run"):
                c09_runner.run_card(card, queue, preflight_only=False)
        preflight_mock.assert_not_called()

    def test_dispatcher_bridge_rejects_noncanonical_card_before_preflight(self) -> None:
        queue, card = c09_runner._canonical_card()
        changed = {**card, "policy": "C_SPARSE_REPLACE"}
        with patch.object(c09_runner, "preflight") as preflight_mock:
            with self.assertRaisesRegex(BatchIntegrityError, "noncanonical card or queue"):
                c09_runner.run_card(changed, queue, preflight_only=True)
        preflight_mock.assert_not_called()

    def test_model_fit_predict_on_synthetic_rows_only(self) -> None:
        n = 48
        ef = np.arange(n) % 2
        token_side = (np.arange(n) // 2) % 2
        text = np.where(token_side == 0, "alpha beacon", "beta beacon").astype(object)
        text[np.arange(n) % 7 == 0] = ""
        frame = pd.DataFrame({
            "impact_load": np.linspace(0.1, 0.9, n),
            "entry_friction": ef,
            "cycle_code": 47 + (np.arange(n) % 4),
            "scope_pressure": np.linspace(1.0, 8.0, n),
            "text_load": np.arange(n) % 12,
            "scope_trace": [f"{1 + (i % 5)} nodes" for i in range(n)],
            "org_spread": [f"{1 + (i % 3)} nodes" for i in range(n)],
            "module_spread": [f"{1 + (i % 4)} nodes" for i in range(n)],
            "telemetry_flag": np.arange(n) % 2,
            "org_code": [f"org_{i % 3}" for i in range(n)],
            "module_code": [f"module_{i % 4}" for i in range(n)],
            "pattern_code": [f"pattern_{i % 2}" for i in range(n)],
            "signal_stream": text,
        })
        y = (((ef == 1) & (token_side == 0)) | ((ef == 0) & (token_side == 1))).astype(np.int8)
        model = C09TextEntryFrictionClassifier().fit(frame, y)
        probabilities = model.predict_proba(frame.iloc[:12].copy())
        state = model.audit_state()
        self.assertEqual(state["id"], "C09")
        self.assertEqual(state["c09_interaction"]["recipe"], "append TF-IDF * I(parsed entry_friction == 1)")
        self.assertEqual(state["c09_interaction"]["design"]["interaction_width"],
                         state["c09_interaction"]["design"]["main_tfidf_width"])
        self.assertGreater(state["c09_interaction"]["design"]["interaction_nnz"], 0)
        self.assertEqual(probabilities.shape, (12,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
