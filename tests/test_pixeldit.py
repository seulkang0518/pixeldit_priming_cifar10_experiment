"""PixelDiT interface, real objectives, stage transfer, and deterministic recovery."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import main
from backbone import build_backbone
from pixeldit_backbone import CIFAR10PixelDiTParticleGenerator, GeneratorDataParallel


class PixelDiTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def small(self):
        return CIFAR10PixelDiTParticleGenerator(width=24, patch_depth=2, pixel_depth=1,
                                               heads=3, patch_size=4, pixel_width=4)

    def test_vendor_checksums_and_native_interface(self):
        vendor = Path(__file__).resolve().parents[1] / 'third_party/pixeldit'
        record = json.loads((vendor / 'UPSTREAM.json').read_text())
        for name, digest in record['files'].items():
            self.assertEqual(hashlib.sha256((vendor / name).read_bytes()).hexdigest(), digest)
        model = self.small()
        # Make the otherwise zero output head nonzero to test layout and sample independence.
        torch.nn.init.normal_(model.network.final_layer.linear.weight, std=.01)
        inputs = torch.randn(3, 3072)
        actual = model(inputs)
        native = model.network(inputs.reshape(3, 3, 32, 32), torch.zeros(3), torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(actual, native.sigmoid().flatten(1))
        torch.testing.assert_close(actual[:1], model(inputs[:1]))
        self.assertFalse(torch.equal(actual[0], actual[1]))
        self.assertEqual(actual.shape, (3, 3072))
        self.assertTrue(((actual > 0) & (actual < 1)).all())
        with self.assertRaises(ValueError):
            model(torch.randn(3, 1024))
        with self.assertRaises(ValueError):
            build_backbone('mnist', 'pixeldit', 784, 32, [8], 1)
        with self.assertRaises(ValueError):
            CIFAR10PixelDiTParticleGenerator(width=24, heads=4)

    def test_corrected_riesz_updates_both_pathways(self):
        model = self.small()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        before = copy.deepcopy(model.state_dict())
        for _ in range(4):
            info = main.generator_step(model, torch.randn(3, 3072), torch.rand(5, 3072), optimizer, 8,
                objective='softmax-riesz', normalize_per_temperature=False,
                no_field_rms_normalization=True, max_grad_norm=1.,
                multiscale_options=dict(image_shape=(3, 32, 32), coarse_sizes=(8, 16),
                    patch_sizes=(7, 14), crops_per_size=1, global_weight=2., coarse_weight=1.,
                    patch_weight=1., patch_rng=torch.Generator().manual_seed(44)))
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values()))
        for name in ('network.s_embedder.proj.weight', 'network.patch_blocks.0.attn.qkv.weight',
                     'network.pixel_embedder.proj.weight', 'network.pixel_blocks.0.compress_to_attn.weight',
                     'network.final_layer.linear.weight'):
            self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)

    def test_parallel_checkpoint_names_and_ema(self):
        model = self.small()
        wrapped = GeneratorDataParallel(model)
        ema = copy.deepcopy(wrapped)
        inputs = torch.randn(2, 3072)
        torch.testing.assert_close(wrapped(inputs), model(inputs))
        self.assertEqual(set(wrapped.state_dict()), set(model.state_dict()))
        wrapped.load_state_dict(model.state_dict(), strict=True)
        main.update_ema(wrapped, ema, .9)
        model.load_state_dict(ema.state_dict(), strict=True)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires at least two CUDA GPUs')
    def test_cuda_parallel_matches_full_population_update(self):
        serial = self.small().cuda()
        # Open the native zero output gate before checking both pathways' updates.
        torch.nn.init.normal_(serial.network.final_layer.linear.weight, std=.01)
        parallel = GeneratorDataParallel(copy.deepcopy(serial), device_ids=[0, 1])
        noise, real = torch.randn(8, 3072, device='cuda'), torch.rand(11, 3072, device='cuda')
        for model in (serial, parallel):
            main.generator_step(model, noise, real, torch.optim.SGD(model.parameters(), lr=.01), 16,
                                objective='softmax-riesz', normalize_per_temperature=False,
                                no_field_rms_normalization=True)
        for name, value in serial.state_dict().items():
            torch.testing.assert_close(value, parallel.state_dict()[name], rtol=3e-4, atol=3e-6)

    def run_training(self, output, extra=()):
        data = torch.arange(6 * 3072).reshape(6, 3072).remainder(256).byte()
        labels = torch.zeros(6, dtype=torch.long)
        argv = ['main.py', '--dataset', 'cifar10', '--architecture', 'pixeldit',
                '--pixeldit-width', '24', '--pixeldit-patch-depth', '2', '--pixeldit-pixel-depth', '1',
                '--pixeldit-heads', '3', '--pixeldit-pixel-width', '4',
                '--steps', '3', '--particles', '3', '--eval-size', '4', '--target-size', '0',
                '--target-batch-size', '2', '--save-every', '1', '--checkpoint-every', '1',
                '--ema-decay', '.9', '--threads', '1', '--device', 'cpu', '--output', str(output), *extra]
        with patch('sys.argv', argv), patch.object(main, 'load_dataset', return_value=(
                Path('/unused/cifar-10-python.tar.gz'), (data, labels), (data, labels))), \
                patch('fid.CIFAR10FID') as evaluator, patch('fid.WEIGHTS', Path(__file__)), \
                patch.object(main, 'save_grid'), patch.object(main, 'save_trajectory'), \
                patch.object(main, 'plot_mmd'), contextlib.redirect_stdout(io.StringIO()):
            evaluator.return_value.evaluate_generator.return_value = 100.
            main.main()
        return torch.load(output / 'checkpoint_latest.pt', weights_only=False)

    def test_checkpoint_handoff_resume_and_shape_preserving_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pre = self.run_training(root / 'pre', ['--training-stage', 'reconstruction'])
            common = ['--init-checkpoint', str(root / 'pre/checkpoint_latest.pt'),
                      '--no-normalize-per-temperature', '--no-field-rms-normalization', '--multiscale-loss']
            initial = self.run_training(root / 'initial', common + ['--steps', '0'])
            for key in ('model', 'ema_model'):
                for name in pre[key]:
                    torch.testing.assert_close(pre[key][name], initial[key][name], rtol=0, atol=0)
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
                    self.run_training(root / 'resumed', common)
            resumed = self.run_training(root / 'resumed', common + [
                '--resume', str(root / 'resumed/checkpoint_latest.pt')])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            for key in ('torch_rng', 'data_rng', 'latent_rng', 'patch_rng'):
                self.assertTrue(torch.equal(full[key], resumed[key]))
            with self.assertRaisesRegex(ValueError, 'mismatch for pixeldit_heads'):
                self.run_training(root / 'bad', common + ['--pixeldit-heads', '1'])


if __name__ == '__main__':
    unittest.main()
