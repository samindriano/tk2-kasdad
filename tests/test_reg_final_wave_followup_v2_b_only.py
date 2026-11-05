import unittest

import numpy as np

from scripts.reg_final_wave_followup_v2_b_only import (
    GateError,
    compose_present_only,
    same_float_bits,
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


if __name__ == "__main__":
    unittest.main()
