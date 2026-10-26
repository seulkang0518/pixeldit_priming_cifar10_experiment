"""Official-reference parity and training integration checks (CPU, no JAX needed)."""
import json
from pathlib import Path
import unittest

import torch

from loss import drifting_loss, sinkhorn_plan, sinkhorn_target_loss
from main import generator_step, training_learning_rate
from multiscale import multiscale_loss


class LossTests(unittest.TestCase):
    def test_official_warmup_schedule(self):
        self.assertEqual(training_learning_rate(0, 4e-4, 10000), 1e-6)
        self.assertAlmostEqual(training_learning_rate(5000, 4e-4, 10000), 0.0002005)
        self.assertEqual(training_learning_rate(10000, 4e-4, 10000), 4e-4)
        self.assertEqual(training_learning_rate(200000, 4e-4, 10000), 4e-4)
        self.assertEqual(training_learning_rate(0, 2e-4, 0), 2e-4)

    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_official_drifting_loss_and_gradient(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/drifting_official.json').read_text())
        for case in fixture['cases']:
            b, n, m, k, d = case['shape']
            arrays = [torch.tensor(a).reshape(b, c, d).requires_grad_()
                      for a, c in zip(case['arrays'], (n, m, k))]
            weights = [torch.tensor(w).reshape(b, c).requires_grad_()
                       for w, c in zip(case['weights'], (n, m, k))]
            value, _ = drifting_loss(*arrays, *weights, tau_list=fixture['temperatures'])
            torch.testing.assert_close(value, torch.tensor(case['loss']), rtol=2e-4, atol=2e-5)
            value.mean().backward()
            torch.testing.assert_close(arrays[0].grad, torch.tensor(case['gradient']),
                                       rtol=3e-4, atol=3e-5)
            self.assertTrue(all(a.grad is None for a in arrays[1:] + weights))

    def test_collisions_and_invalid_temperatures(self):
        for initial in (0., .5):
            x = torch.full((1, 4, 6), initial, requires_grad=True)
            value, _ = drifting_loss(x, torch.full((1, 7, 6), initial))
            value.mean().backward()
            self.assertTrue(torch.isfinite(value).all() and torch.isfinite(x.grad).all())
            if initial == 0:
                self.assertEqual(value.item(), 0)
        for temperatures in ((), (-1.,), (float('nan'),)):
            with self.assertRaises(ValueError):
                drifting_loss(x, x.detach(), tau_list=temperatures)

    def test_training_steps_and_multiscale(self):
        for objective in ('drifting', 'sinkhorn', 'softmax-riesz', 'riesz'):
            for multiscale in (False, True):
                model = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Sigmoid())
                before = model[0].weight.detach().clone()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                options = dict(image_shape=(1, 4, 4), coarse_sizes=(2,), patch_sizes=(2,),
                               patch_rng=torch.Generator().manual_seed(0)) if multiscale else None
                generator_step(model, torch.randn(4, 3), torch.rand(7, 16), optimizer, 8,
                               objective=objective, multiscale_options=options,
                               sinkhorn_latent=torch.randn(4, 3) if objective == 'sinkhorn' else None)
                self.assertTrue(torch.isfinite(model[0].weight).all())
                self.assertFalse(torch.equal(before, model[0].weight))
        # A global-only view must give the same loss/gradient as the direct API.
        x = torch.rand(4, 16, requires_grad=True)
        y = torch.rand(7, 16)
        direct, _ = drifting_loss(x[None], y[None])
        grouped, _ = multiscale_loss(x, y, (1, 4, 4), 8, objective='drifting',
                                    coarse_weight=0, patch_weight=0, coarse_sizes=(), patch_sizes=())
        torch.testing.assert_close(grouped, direct.mean())
        torch.testing.assert_close(torch.autograd.grad(grouped, x)[0],
                                   torch.autograd.grad(direct.mean(), x)[0])

    def test_sinkhorn_balanced_plan_and_detached_supports(self):
        x, y, r = [torch.randn(n, 3, requires_grad=True) for n in (4, 7, 5)]
        plan = sinkhorn_plan(x, y, epsilon=1., iterations=200)
        torch.testing.assert_close(plan.sum(1), torch.full((4,), 1/4))
        torch.testing.assert_close(plan.sum(0), torch.full((7,), 1/7))
        value, _ = sinkhorn_target_loss(x, y, r)
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNone(y.grad)
        self.assertIsNone(r.grad)

    def test_official_sinkhorn_loss_plan_and_gradient(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/sinkhorn_official.json').read_text())
        for case in fixture['cases']:
            x = torch.tensor(case['x'], requires_grad=True)
            y = torch.tensor(case['y'])
            reference = torch.tensor(case['reference'])
            value, _ = sinkhorn_target_loss(x, y, reference)
            gradient, = torch.autograd.grad(value, x)
            torch.testing.assert_close(value, torch.tensor(case['loss']), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(gradient, torch.tensor(case['gradient']), rtol=3e-4, atol=3e-6)
            plan = sinkhorn_plan(x, y, epsilon=.7, iterations=10)
            torch.testing.assert_close(plan, torch.tensor(case['plan']), rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
