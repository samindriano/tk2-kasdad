"""Meaningful checks of discovery, blocked inputs, and actual local audit evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.data.audit import audit_task, profile, sha256
from src.data.samuel import _infer_target
from src.utils.paths import PROJECT_ROOT, data_root, task_files


class AuditTests(unittest.TestCase):
    def test_missing_inputs_remain_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_task('regression', Path(folder))
            self.assertEqual(result['status'], 'BLOCKED_MISSING_DATA')
            self.assertIsNone(result['target'])
            self.assertEqual(len(result['missing_roles']), 3)

    def test_external_path_and_ambiguous_names(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'TP2_DATA_DIR': folder}):
                self.assertEqual(data_root(), Path(folder).resolve())
            path = Path(folder) / 'classification'
            path.mkdir()
            # Empty filename fixtures only; no fabricated observations.
            (path / 'Train.csv').touch()
            (path / 'TRAIN.csv').touch()
            # On case-insensitive Windows these names identify one file.
            if len(list(path.glob('*.csv'))) > 1:
                with self.assertRaises(ValueError):
                    task_files('classification', folder)
            else:
                self.assertIsNotNone(task_files('classification', folder)['train'])

    def test_empty_numeric_profile_serializes_strict_json(self):
        value = profile(pd.DataFrame({'measurement': pd.Series(dtype=float)}))
        json.dumps(value, allow_nan=False)
        self.assertEqual(value['shape'], [0, 1])

    def test_regression_target_requires_sample_id_and_exact_schema(self):
        # Header-only frames have no fabricated rows or analysis evidence.
        train = pd.DataFrame(columns=['record_id', 'impact_load', 'cycle_code', 'threat_rank'])
        test = train.drop(columns=['threat_rank'])
        malformed = pd.DataFrame(columns=['threat_rank'])
        official_headers = pd.DataFrame(columns=['record_id', 'threat_rank'])
        self.assertIsNone(_infer_target(train, test, malformed)['confirmed_target'])
        self.assertEqual(_infer_target(train, test, official_headers)['confirmed_target'], 'threat_rank')

    def test_saved_report_matches_immutable_real_inputs(self):
        report = PROJECT_ROOT / 'reports/audits/classification_audit.json'
        paths = task_files('classification', PROJECT_ROOT / 'data/raw')
        if not report.is_file() or not all(paths.values()):
            self.skipTest('Actual classification data and executed report required; no synthetic substitute.')
        saved = json.loads(report.read_text(encoding='utf-8'))
        for role, path in paths.items():
            frame = pd.read_csv(path)
            self.assertEqual(saved['files'][role]['sha256'], sha256(path))
            self.assertEqual(saved['files'][role]['shape'], list(frame.shape))
        train = pd.read_csv(paths['train'])
        test = pd.read_csv(paths['test'])
        sample = pd.read_csv(paths['sample'])
        self.assertEqual(set(train)-set(test), {saved['target']})
        self.assertEqual(set(sample)-{'record_id'}, {saved['target']})
        self.assertEqual(train[saved['target']].value_counts().to_dict(), {int(k): v for k,v in saved['target_summary']['counts'].items()})
        self.assertTrue(sample.record_id.equals(test.record_id))


if __name__ == '__main__':
    unittest.main()
