"""Raw softmax field math, clipping telemetry, and trainer resume coverage."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

import torch

from loss import _riesz_topk_softmax_loss_impl, _softmax_toward
from main import generator_step
from multiscale import multiscale_loss
import test_priming


class UnnormalizedSoftmaxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_near_collisions_match_double_precision_unit_directions(self):
        # More than 25 points exercises the sizes where cdist used the unstable
        # squared-norm matrix identity; more than 32 queries crosses a chunk.
        x = .5 + torch.randn(2, 37, 147) * 1e-4
        for self_interaction in (False, True):
            y = x.clone() if self_interaction else .5 + torch.randn(2, 41, 147) * 1e-4
            if self_interaction:
                y[:, 33] = y[:, 0]
                x[:, 33] = x[:, 0]
            tau = .02 * 147**.5
            actual, ess = _softmax_toward(x, y, torch.ones(y.shape[:2]), tau, 1e-8,
                                          exclude_self=self_interaction)
            delta = y.double()[:, None] - x.double()[:, :, None]
            distances = delta.norm(dim=-1)
            logits = -distances / tau
            if self_interaction:
                logits.masked_fill_(torch.eye(37, dtype=torch.bool)[None], -torch.inf)
            weights = logits.softmax(-1)
            expected = (weights[..., None] * delta / distances.clamp_min(1e-8)[..., None]).sum(2)
            torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(ess.double(), 1 / weights.square().sum(-1), atol=1e-5, rtol=1e-5)
            self.assertLessEqual(actual.norm(dim=-1).max().item(), 1 + 1e-6)

    def test_exact_collisions_have_zero_field_and_gradient(self):
        x = torch.full((1, 37, 147), .5, requires_grad=True)
        y = torch.full((1, 41, 147), .5)
        value, info = _riesz_topk_softmax_loss_impl(
            x, y, tau_list=(.2, .05, .02), no_field_rms_normalization=True, return_field=True)
        value.mean().backward()
        torch.testing.assert_close(info['frozen_velocity'], torch.zeros_like(x), atol=0, rtol=0)
        torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)

    def test_weighted_directions_preserve_support_coefficients(self):
        x, y = torch.randn(2, 35, 5), torch.randn(2, 43, 5)
        coefficients = torch.rand(2, 43) * 2
        actual, _ = _softmax_toward(x, y, coefficients, .2, 1e-8)
        delta = y[:, None] - x[:, :, None]
        distances = delta.norm(dim=-1)
        expected = ((-distances / .2).softmax(-1)[..., None] * coefficients[:, None, :, None]
                    * delta / distances.clamp_min(1e-8)[..., None]).sum(2)
        torch.testing.assert_close(actual, expected)

    def test_raw_field_and_gradient_against_pairwise_formula(self):
        x = torch.rand(1, 4, 5, requires_grad=True)
        y = torch.rand(1, 7, 5, requires_grad=True)
        taus = (.2, .05, .02)
        value, info = _riesz_topk_softmax_loss_impl(
            x, y, tau_list=taus, epsilon=1e-6, normalize_per_temperature=True,
            no_field_rms_normalization=True, return_field=True)
        with torch.no_grad():
            scale = torch.cdist(x, torch.cat([x, y], dim=1)).mean() / (1 + 1e-6)
            coordinate_scale = (scale / 5**.5).clamp(min=1e-3)
            a, b = x / coordinate_scale, y / coordinate_scale
            fields = []
            for tau in taus:
                def toward(target, exclude_self=False):
                    delta = target[:, None, :, :] - a[:, :, None, :]
                    distance = delta.norm(dim=-1)
                    logits = -distance / (tau * 5**.5)
                    if exclude_self:
                        logits.masked_fill_(torch.eye(4, dtype=torch.bool)[None], -torch.inf)
                    return (logits.softmax(-1)[..., None] * delta /
                            distance.clamp(min=1e-8)[..., None]).sum(2)
                fields.append((toward(b) - toward(a, True)) / tau)
            raw = sum(fields)
        torch.testing.assert_close(info['scale_inputs'], coordinate_scale)
        torch.testing.assert_close(info['frozen_velocity'], raw, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(info['riesz_topk_softmax_force_rms'], raw.square().mean().sqrt())
        value.mean().backward()
        torch.testing.assert_close(x.grad, -2 * raw / (x.numel() * coordinate_scale), rtol=2e-5, atol=2e-5)
        self.assertIsNone(y.grad)
        # The old switch still yields combined RMS normalization, not the raw field.
        _, combined = _riesz_topk_softmax_loss_impl(
            x, y, tau_list=taus, epsilon=1e-6, normalize_per_temperature=False, return_field=True)
        torch.testing.assert_close(combined['frozen_velocity'], raw / raw.square().mean().sqrt(),
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(combined['scale_inputs'], coordinate_scale)
        _, raw_false = _riesz_topk_softmax_loss_impl(
            x, y, tau_list=taus, epsilon=1e-6, normalize_per_temperature=False,
            no_field_rms_normalization=True, return_field=True)
        torch.testing.assert_close(raw_false['frozen_velocity'], info['frozen_velocity'])

    def test_multiscale_and_clipping_diagnostics(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Sigmoid())
        initial = copy.deepcopy(model)
        z, y = torch.randn(4, 3), torch.rand(7, 16)
        options = dict(image_shape=(1, 4, 4), coarse_sizes=(2,), patch_sizes=(2,),
                       patch_rng=torch.Generator().manual_seed(4))
        for threshold, expected_clipping in ((1e-6, 1.), (1e6, 0.), (0., 0.)):
            model.load_state_dict(initial.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            result = generator_step(model, z, y, optimizer, 8, objective='softmax-riesz',
                                    multiscale_options=options, no_field_rms_normalization=True,
                                    max_grad_norm=threshold)
            self.assertEqual(result['clip_fraction'].item(), expected_clipping)
            self.assertGreater(result['grad_norm_before_clip'].item(), 0)
            for name in ('raw_force_rms', 'raw_attraction_rms', 'raw_self_repulsion_rms',
                         'raw_force_rms_tau_0.02'):
                self.assertTrue(torch.isfinite(result[name]))
            post_clip = torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
            if expected_clipping:
                self.assertLessEqual(post_clip.item(), threshold * 1.0001)
        # Optional target-update route must preserve the same raw-field loss and gradient.
        x = torch.rand(4, 16, requires_grad=True)
        kwargs = dict(objective='softmax-riesz', no_field_rms_normalization=True,
                      coarse_weight=0, patch_weight=0, coarse_sizes=(), patch_sizes=())
        direct, _ = multiscale_loss(x, y, (1, 4, 4), 8, **kwargs)
        alternate, telemetry = multiscale_loss(x, y, (1, 4, 4), 8,
                                             target_options={'field_step': 1.}, **kwargs)
        torch.testing.assert_close(alternate, direct)
        torch.testing.assert_close(torch.autograd.grad(alternate, x)[0], torch.autograd.grad(direct, x)[0])
        self.assertIn('raw_force_rms', telemetry)

    def test_logged_diagnostics_and_resume(self):
        helper = test_priming.PrimingTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = ['--training-stage', 'distributional', '--objective', 'softmax-riesz',
                      '--no-field-rms-normalization', '--multiscale-loss',
                      '--max-grad-norm', '0.000001']
            full = helper.run_training(root / 'full', common + ['--steps', '5'])
            helper.run_training(root / 'resume', common + ['--steps', '3'])
            resumed = helper.run_training(root / 'resume', common + ['--steps', '5', '--save-every', '5', '--resume',
                                           str(root / 'resume/checkpoint_latest.pt')])
            self.assertEqual(resumed['config']['snapshot_iterations'], [0, 5])
            self.assertEqual(resumed['config']['softmax_field_normalization'], 'none')
            self.assertFalse(resumed['config']['normalize_per_temperature'])
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=0, atol=0)
            with (root / 'resume/training_diagnostics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['update']) for row in rows], [1, 2, 3, 4, 5])
            self.assertTrue(all(float(row['clip_fraction']) == 1 for row in rows))
            self.assertTrue(all(float(row['raw_force_rms']) > 0 for row in rows))


if __name__ == '__main__':
    unittest.main()
