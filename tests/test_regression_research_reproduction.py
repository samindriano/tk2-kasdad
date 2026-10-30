from pathlib import Path
import tempfile
import unittest
from scripts.reproduce_regression_research import preserve_reports


class ReportPreservationTests(unittest.TestCase):
    def test_success_preserves_original_and_separates_new_measurement(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);path=root/'historical.json';path.write_bytes(b'{"runtime":237}\r\n')
            with preserve_reports([path],root/'newrun'):
                path.write_bytes(b'{"runtime":4}\n')
            self.assertEqual(path.read_bytes(),b'{"runtime":237}\r\n')
            self.assertEqual((root/'newrun/historical.json').read_bytes(),b'{"runtime":4}\n')

    def test_failure_restores_original_even_if_writer_removed_output(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);path=root/'historical.json';path.write_bytes(b'original')
            with self.assertRaises(RuntimeError):
                with preserve_reports([path],root/'newrun'):
                    path.unlink();raise RuntimeError('software fixture')
            self.assertEqual(path.read_bytes(),b'original')


if __name__=='__main__':unittest.main()
