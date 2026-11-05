import unittest
import hashlib

import numpy as np

from scripts.reg_final_wave_followup_v2_b_only import (
    GateError,
    compose_present_only,
    same_float_bits,
    sha_raw_f64,
    sha_vector,
)
from scripts.verify_reg_final_wave_followup_v2_b_only import (
    VerificationError,
    verify_row_aligned_output,
)


class PresentOnlyCompositionTests(unittest.TestCase):
    def setUp(self):
        self.archive_ids = np.asarray(["d", "b", "a", "c"])
        self.global_ids = ["a", "b", "c", "d"]
        self.global_values = np.asarray([10.0, 20.0, 30.0, 40.0])
        self.direct_ids = ["c", "a", "d", "b"]
        self.direct_values = np.asarray([300.0, 100.0, 400.0, 200.0])
        self.missing = np.asarray([False, True, False, True])

    def test_aligns_by_id_and_restores_only_missing(self):
        ids, global_aligned, direct_aligned, output = compose_present_only(
            self.archive_ids, self.global_ids, self.global_values,
            self.direct_ids, self.direct_values, self.missing)
        self.assertEqual(ids.tolist(), ["d", "b", "a", "c"])
        self.assertTrue(same_float_bits(global_aligned, np.asarray([40.0, 20.0, 10.0, 30.0])))
        self.assertTrue(same_float_bits(direct_aligned, np.asarray([400.0, 200.0, 100.0, 300.0])))
        self.assertTrue(same_float_bits(output, np.asarray([40.0, 200.0, 10.0, 300.0])))
        self.assertTrue(same_float_bits(output[~self.missing], global_aligned[~self.missing]))
        self.assertTrue(same_float_bits(output[self.missing], direct_aligned[self.missing]))

    def test_duplicate_id_fails_closed(self):
        with self.assertRaises(GateError):
            compose_present_only(self.archive_ids, ["a", "a", "c", "d"],
                                 self.global_values, self.direct_ids,
                                 self.direct_values, self.missing)

    def test_id_set_mismatch_fails_closed(self):
        with self.assertRaises(GateError):
            compose_present_only(self.archive_ids, self.global_ids,
                                 self.global_values, ["c", "a", "d", "x"],
                                 self.direct_values, self.missing)

    def test_mask_misalignment_fails_closed(self):
        with self.assertRaises(GateError):
            compose_present_only(self.archive_ids, self.global_ids,
                                 self.global_values, self.direct_ids,
                                 self.direct_values, np.asarray([False, True]))

    def test_source_specific_prediction_hashes_are_distinct(self):
        values = np.asarray([1.0, 2.0, 3.5], dtype=np.float64)
        raw_digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
        self.assertEqual(sha_raw_f64(values), raw_digest)
        self.assertNotEqual(sha_raw_f64(values), sha_vector(values))

    def test_independent_replay_rejects_wrong_value_and_reordered_output(self):
        archive_ids = ["d", "b", "a", "c"]
        missing = np.asarray([False, True, False, True])
        global_values = np.asarray([40.0, 20.0, 10.0, 30.0])
        direct_values = np.asarray([400.0, 200.0, 100.0, 300.0])
        expected = np.asarray([40.0, 200.0, 10.0, 300.0])
        verify_row_aligned_output(archive_ids, archive_ids, expected, expected,
                                  missing, global_values, direct_values)
        wrong_value = expected.copy()
        wrong_value[1] = np.nextafter(wrong_value[1], np.inf)
        with self.assertRaises(VerificationError):
            verify_row_aligned_output(archive_ids, archive_ids, wrong_value, expected,
                                      missing, global_values, direct_values)
        reordered_ids = archive_ids[::-1]
        with self.assertRaises(VerificationError):
            verify_row_aligned_output(reordered_ids, archive_ids, expected[::-1], expected,
                                      missing, global_values, direct_values)


if __name__ == "__main__":
    unittest.main()
