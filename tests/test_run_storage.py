"""New-run routing must not redirect recovery or explicit test destinations."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_storage import DEFAULT_RESULTS_ROOT, LEGACY_RESULTS_ROOT, new_run_path


class RunStorageTests(unittest.TestCase):
    def test_new_paths_route_to_standalone_results(self):
        with patch.dict(os.environ, {}, clear=True):
            for path in ('results/cifar/future_run', 'cifar/future_run',
                         LEGACY_RESULTS_ROOT / 'cifar/future_run'):
                self.assertEqual(new_run_path(path), DEFAULT_RESULTS_ROOT / 'cifar/future_run')

    def test_existing_and_explicit_absolute_paths_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(new_run_path(root), root)
            self.assertEqual(new_run_path(root / 'new'), root / 'new')
            with patch('run_storage.Path.exists', return_value=True):
                old = LEGACY_RESULTS_ROOT / 'cifar/old_run'
                self.assertEqual(new_run_path(old), old)

    def test_override_and_no_relative_escape(self):
        with patch.dict(os.environ, {'MMD_RESULTS_ROOT': '/tmp/custom_results'}):
            self.assertEqual(new_run_path('results/cifar/new'), Path('/tmp/custom_results/cifar/new'))
            with self.assertRaises(ValueError):
                new_run_path('results/../escape')
        with patch.dict(os.environ, {'MMD_RESULTS_ROOT': 'relative'}):
            with self.assertRaises(ValueError):
                new_run_path('results/cifar/new')
