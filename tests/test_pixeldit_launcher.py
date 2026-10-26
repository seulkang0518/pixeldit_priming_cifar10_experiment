"""Freeze the recipe, keep reference flags, and reject altered frozen source."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import main

script = Path(__file__).resolve().parents[1] / 'scripts/cs_cluster/pixeldit_study.py'
spec = importlib.util.spec_from_file_location('pixeldit_study', script)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_reference_recipe_survives_backbone_swap_and_source_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            for name in ('assets', 'tests', 'third_party', 'scripts/cs_cluster'):
                (repo / name).mkdir(parents=True)
            (repo / 'main.py').write_text('# synthetic source for snapshot test\n')
            weight = repo / 'assets/pt_inception-2015-12-05-6726825d.pth'
            weight.write_bytes(b'synthetic test weight; not used for inference')
            (repo / 'assets/inception.sha256').write_text(hashlib.sha256(weight.read_bytes()).hexdigest())
            pre, post = launcher.default_commands('python', Path('/old/source'), Path('/old/study'), Path('/old/data'))
            pre += ['--architecture', 'dit', '--lpips-weight', '.1', '--lpips-cache', '/old/cache']
            post += ['--architecture', 'dit']
            # Deliberately use a nondefault recipe to catch accidental replacement with defaults.
            launcher.set_option(post, '--learning-rate', '.00017')
            launcher.set_option(post, '--train-particles', '96')
            reference = root / 'reference.json'
            reference.write_text(json.dumps(dict(stages=[dict(stage='reconstruction', command=pre),
                                                        dict(stage='posttraining', command=post)])))
            args = SimpleNamespace(study=root / 'study', python='python', data_root=root / 'data',
                                   backbone='pixeldit', data_parallel=True, reference=reference)
            with patch.object(launcher, 'REPO', repo), contextlib.redirect_stdout(io.StringIO()):
                launcher.prepare(args)
            manifest = json.loads((args.study / 'manifest.json').read_text())
            allowed = {'output', 'data_root', 'architecture', 'data_parallel', 'lpips_cache', 'init_checkpoint'}
            for original, stage in zip((pre, post), manifest['stages']):
                parsed = []
                for command in (original, stage['command']):
                    with patch('sys.argv', command[2:]):
                        parsed.append(vars(main.parse_args()))
                for key in parsed[0].keys() - allowed:
                    self.assertEqual(parsed[0][key], parsed[1][key], key)
                self.assertEqual(parsed[1]['architecture'], 'pixeldit')
                self.assertTrue(parsed[1]['data_parallel'])
            self.assertIn('--lpips-weight', manifest['smoke_command'])
            self.assertEqual(manifest['reference_sha256'], hashlib.sha256(reference.read_bytes()).hexdigest())
            # Editing the checkout cannot change the frozen source, and editing the frozen copy is detected.
            (repo / 'main.py').write_text('# changed checkout\n')
            frozen = args.study / 'source/main.py'
            self.assertIn('synthetic source', frozen.read_text())
            frozen.write_text('# changed frozen source\n')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                launcher.run(SimpleNamespace(manifest=args.study / 'manifest.json', dry_run=False, smoke_only=True))


if __name__ == '__main__':
    unittest.main()
