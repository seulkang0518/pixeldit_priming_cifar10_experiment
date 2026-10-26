"""DiT spatial layout, optimization, stage handoff, and exact recovery checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import main
from backbone import CIFAR10DiTParticleGenerator, build_backbone


class DiTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_patch_layout_and_batch_independence(self):
        model = CIFAR10DiTParticleGenerator(width=24, depth=2, heads=3)
        image = torch.arange(3072).reshape(1, 3, 32, 32)
        patches = image.reshape(1, 3, 16, 2, 16, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 256, 12)
        self.assertTrue(torch.equal(model.unpatchify(patches), image))
        noise = torch.randn(2, 3072)
        together = model(noise)
        torch.testing.assert_close(together[:1], model(noise[:1]))
        self.assertEqual(together.shape, (2, 3072))
        self.assertTrue(((together > 0) & (together < 1)).all())
        self.assertFalse(torch.equal(together[0], together[1]))
        with self.assertRaises(ValueError):
            build_backbone('mnist', 'dit', 784, 32, [128], 4)
        with self.assertRaises(ValueError):
            CIFAR10DiTParticleGenerator(width=25, heads=3)

    def test_corrected_multiscale_field_trains_attention_and_mlp(self):
        model = CIFAR10DiTParticleGenerator(width=24, depth=2, heads=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        before = copy.deepcopy(model.state_dict())
        for _ in range(3):
            info = main.generator_step(model, torch.randn(3, 3072), torch.rand(5, 3072),
                optimizer, 8, objective='softmax-riesz', normalize_per_temperature=False,
                no_field_rms_normalization=True, max_grad_norm=1.,
                multiscale_options=dict(image_shape=(3, 32, 32), coarse_sizes=(8, 16),
                    patch_sizes=(7, 14), crops_per_size=1, global_weight=2.,
                    coarse_weight=1., patch_weight=1., patch_rng=torch.Generator().manual_seed(44)))
            self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values()))
        for name in ('patch_embed.weight', 'blocks.0.qkv.weight', 'blocks.1.mlp.0.weight',
                     'blocks.0.modulation', 'head.weight'):
            self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)

    def run_training(self, output, extra=()):
        data = torch.arange(6 * 3072).reshape(6, 3072).remainder(256).byte()
        labels = torch.zeros(6, dtype=torch.long)
        argv = ['main.py', '--dataset', 'cifar10', '--architecture', 'dit',
                '--dit-width', '24', '--dit-depth', '2', '--dit-heads', '3',
                '--steps', '3', '--particles', '3', '--eval-size', '4',
                '--target-size', '0', '--target-batch-size', '2',
                '--save-every', '1', '--checkpoint-every', '1', '--ema-decay', '.9',
                '--threads', '1', '--device', 'cpu', '--output', str(output), *extra]
        with patch('sys.argv', argv), patch.object(main, 'load_dataset', return_value=(
                Path('/unused/cifar-10-python.tar.gz'), (data, labels), (data, labels))), \
                patch('fid.CIFAR10FID') as evaluator, patch.object(main, 'save_grid'), \
                patch.object(main, 'save_trajectory'), patch.object(main, 'plot_mmd'):
            evaluator.return_value.evaluate_generator.return_value = 100.
            main.main()
        return torch.load(output / 'checkpoint_latest.pt', weights_only=False)

    def test_perceptual_priming_resume_and_handoff(self):
        class Perceptual(torch.nn.Module):
            metadata = {'weights_sha256': 'test-frozen-weights', 'frozen': True}
            def forward(self, prediction, clean):
                return (prediction - clean).abs().mean()

        with tempfile.TemporaryDirectory() as directory, \
                patch('perceptual.VGGPerceptualLoss', return_value=Perceptual()) as loader:
            root = Path(directory)
            common = ['--training-stage', 'reconstruction', '--lpips-weight', '.1',
                      '--lpips-cache', str(root / 'cache')]
            full = self.run_training(root / 'full', common)
            original = main.reconstruction_step
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('perceptual interruption')
                return original(*args, **kwargs)
            with patch.object(main, 'reconstruction_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'perceptual interruption'):
                    self.run_training(root / 'resume', common)
            recovered = self.run_training(root / 'resume', common + [
                '--resume', str(root / 'resume/checkpoint_latest.pt')])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], recovered[key][name], rtol=0, atol=0)
            self.assertIn('train_reconstruction_lpips', (root / 'full/metrics.csv').read_text())
            with self.assertRaisesRegex(ValueError, 'mismatch for lpips_weight'):
                self.run_training(root / 'resume', common + ['--lpips-weight', '.2',
                    '--resume', str(root / 'resume/checkpoint_latest.pt')])
            previous_calls = loader.call_count
            post = self.run_training(root / 'post', [
                '--init-checkpoint', str(root / 'full/checkpoint_latest.pt'), '--steps', '0'])
            self.assertEqual(loader.call_count, previous_calls)
            self.assertEqual(post['optimizer']['state'], {})
            self.assertEqual(post['config']['priming_initialization']['source_perceptual']['weight'], .1)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], post[key][name], rtol=0, atol=0)

    def test_priming_handoff_resume_and_architecture_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primed = self.run_training(root / 'priming', ['--training-stage', 'reconstruction'])
            common = ['--init-checkpoint', str(root / 'priming/checkpoint_latest.pt'),
                      '--no-normalize-per-temperature', '--no-field-rms-normalization']
            initial = self.run_training(root / 'initial', common + ['--steps', '0'])
            for key in ('model', 'ema_model'):
                for name in primed[key]:
                    torch.testing.assert_close(primed[key][name], initial[key][name], rtol=0, atol=0)
            self.assertEqual(initial['optimizer']['state'], {})
            full = self.run_training(root / 'full', common)
            original = main.generator_step
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('test interruption')
                return original(*args, **kwargs)

            with patch.object(main, 'generator_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    self.run_training(root / 'resume', common)
            recovered = self.run_training(root / 'resume', common + [
                '--resume', str(root / 'resume/checkpoint_latest.pt')])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], recovered[key][name], rtol=0, atol=0)
            # Changing head count preserves tensor shapes, but must still be rejected.
            with self.assertRaisesRegex(ValueError, 'mismatch for dit_heads'):
                self.run_training(root / 'bad', common + ['--dit-heads', '6'])


if __name__ == '__main__':
    unittest.main()
