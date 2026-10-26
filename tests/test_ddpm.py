"""DDPM math, time conditioning, legacy backbone compatibility, and resume."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from backbone import CIFAR10UNetParticleGenerator
import ddpm


class DDPMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_corruption_and_reverse_posterior(self):
        diffusion = ddpm.Diffusion()
        clean = torch.rand(3, 3, 32, 32) * 2 - 1
        noise = torch.randn_like(clean)
        t = torch.tensor([0, 400, 999])
        noisy = diffusion.corrupt(clean, t, noise)
        beta = torch.linspace(.0001, .02, 1000, dtype=torch.float64)
        abar = (1 - beta).cumprod(0)
        expected = abar[t].sqrt()[:, None, None, None] * clean + (1 - abar[t]).sqrt()[:, None, None, None] * noise
        torch.testing.assert_close(noisy.double(), expected, atol=2e-7, rtol=1e-5)
        # Oracle epsilon at t=0 must recover x0 with no stochastic final step.
        recovered = diffusion.reverse_mean(noisy[:1], noise[:1], 0)
        torch.testing.assert_close(recovered, clean[:1], atol=1e-6, rtol=1e-5)
        step = 400
        alpha = 1 - beta[step]
        mean = (noisy[1] - beta[step] / (1 - abar[step]).sqrt() * noise[1]) / alpha.sqrt()
        torch.testing.assert_close(diffusion.reverse_mean(noisy[1], noise[1], step), mean.float(),
                                   atol=1e-6, rtol=1e-5)

    def test_time_conditioning_and_unrestricted_head(self):
        model = CIFAR10UNetParticleGenerator(8, 1, time_conditioned=True).eval()
        # Make residual/head branches visible so timestep dependence is measurable.
        for name, parameter in model.named_parameters():
            if 'conv2.weight' in name or name == 'output.2.weight':
                torch.nn.init.normal_(parameter, std=.03)
        x = torch.randn(2, 3, 32, 32)
        first = model(x, torch.zeros(2, dtype=torch.long))
        second = model(x, torch.full((2,), 999))
        self.assertEqual(first.shape, x.shape)
        self.assertFalse(torch.allclose(first, second))
        self.assertTrue((first < 0).any())
        (first.square().mean()).backward()
        self.assertGreater(model.time_mlp[0].weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(x)

    def test_original_generator_matches_existing_source_snapshot(self):
        # Immutable pre-DDPM snapshot checks both checkpoint keys and execution.
        source = Path(__file__).resolve().parents[1] / 'results/cifar/priming_seed42_20261025/source/backbone.py'
        if not source.exists():
            self.skipTest('Pre-DDPM source snapshot is not available')
        spec = importlib.util.spec_from_file_location('legacy_backbone', source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        legacy = old.CIFAR10UNetParticleGenerator(8, 1).eval()
        current = CIFAR10UNetParticleGenerator(8, 1).eval()
        current.load_state_dict(legacy.state_dict(), strict=True)
        x = torch.randn(2, 3072)
        torch.testing.assert_close(current(x), legacy(x), atol=0, rtol=0)

    def test_sampler_is_reproducible_and_preserves_training_rng(self):
        class Zero(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        model = Zero().train()
        diffusion = ddpm.Diffusion(timesteps=8)
        state = torch.get_rng_state().clone()
        first = diffusion.sample(model, 2, generator=torch.Generator().manual_seed(7))
        second = diffusion.sample(model, 2, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(((first >= 0) & (first <= 1)).all())

    def test_training_reduces_fixed_batch_loss(self):
        model = CIFAR10UNetParticleGenerator(8, 1, time_conditioned=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        diffusion = ddpm.Diffusion()
        clean, noise = torch.rand(2, 3, 32, 32) * 2 - 1, torch.randn(2, 3, 32, 32)
        t = torch.tensor([400, 800])
        model.eval()
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            loss = (model(diffusion.corrupt(clean, t, noise), t) - noise).square().mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], .8 * losses[0])

    def test_interrupted_resume_matches_uninterrupted(self):
        data = torch.arange(5 * 3072).reshape(5, 3072).remainder(256).byte()
        def run(output, resume=False, steps=4):
            argv = ['ddpm.py', '--output', str(output), '--device', 'cpu', '--steps', str(steps),
                    '--batch-size', '2', '--base-channels', '8', '--num-res-blocks', '1',
                    '--checkpoint-every', '2', '--preview-every', '4', '--fid-samples', '0', '--threads', '1']
            if resume:
                argv += ['--resume', str(output / 'checkpoint_latest.pt')]
            with (patch('sys.argv', argv), patch.object(ddpm, 'download_cifar10', return_value=Path('/unused')),
                 patch.object(ddpm, 'load_cifar10_split', return_value=(data, torch.zeros(5))),
                 patch.object(ddpm.Diffusion, 'sample', return_value=torch.zeros(64, 3, 32, 32)),
                 patch.object(ddpm, 'save_grid')):
                ddpm.main()
            return torch.load(output / 'checkpoint_latest.pt', weights_only=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = run(root / 'full')
            original = ddpm.train_step
            def interrupted(*args, **kwargs):
                if args[5] == 3:
                    raise RuntimeError('interrupted')
                return original(*args, **kwargs)
            with patch.object(ddpm, 'train_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run(root / 'resumed')
            resumed = run(root / 'resumed', True)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], atol=0, rtol=0)
            self.assertTrue(torch.equal(full['torch_rng'], resumed['torch_rng']))
            run(root / 'extended', steps=2)
            extended = run(root / 'extended', resume=True)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], extended[key][name], atol=0, rtol=0)
            self.assertTrue(torch.equal(full['torch_rng'], extended['torch_rng']))
            with self.assertRaisesRegex(ValueError, 'cannot shorten'):
                run(root / 'extended', resume=True, steps=2)


if __name__ == '__main__':
    unittest.main()
