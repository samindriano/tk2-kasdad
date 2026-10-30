"""Presentation must bind to the completed, correctly selected experiment stage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.models import regression_round2_reporting as reporting


class ReportingBindingTests(unittest.TestCase):
    def test_rejects_base_or_stale_interaction_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            valid = {'freeze_signature': 'f', 'partial': False, 'stage': 'all', 'interaction_signature': 's'}
            with patch.object(reporting, 'REPORT', root), patch.object(reporting, 'verify_freeze', return_value={'signature': 'f'}), patch.object(reporting, 'load_interaction_selection', return_value={'signature': 's'}):
                for changed in [{'stage': 'base'}, {'interaction_signature': 'old'}, {'partial': True}, {'freeze_signature': 'old'}]:
                    (root/'results.json').write_text(json.dumps({**valid, **changed}), encoding='utf-8')
                    with self.assertRaises(RuntimeError):
                        reporting.load_results()
                (root/'results.json').write_text(json.dumps(valid), encoding='utf-8')
                self.assertEqual(reporting.load_results(), valid)


if __name__ == '__main__':
    unittest.main()
