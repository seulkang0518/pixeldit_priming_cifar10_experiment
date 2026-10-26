"""Direct detached-plan OT math, integration, and checkpoint compatibility."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from loss import sinkhorn_cost_plan, sinkhorn_ot_loss
from main import generator_step
from multiscale import multiscale_loss
import test_priming


class SinkhornOTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    def test_uniform_marginals_and_detached_plan(self):
        cost = torch.rand(5, 7, requires_grad=True)
        plan, error, iterations = sinkhorn_cost_plan(cost, epsilon=.1, iterations=1000, tolerance=1e-6)
        self.assertFalse(plan.requires_grad)
        torch.testing.assert_close(plan.sum(1), torch.full((5,), 1/5), atol=1e-6, rtol=0)
        torch.testing.assert_close(plan.sum(0), torch.full((7,), 1/7), atol=1e-6, rtol=0)
        self.assertLess(error, 1e-6)
        self.assertLess(iterations, 1000)
        _, _, count = sinkhorn_cost_plan(cost, epsilon=.1, iterations=13, tolerance=0)
        self.assertEqual(count, 13)

    def test_one_query_analytic_gradient_and_detached_supports(self):
        x = torch.tensor([[.2, .7, .4]], requires_grad=True)
        y = torch.tensor([[.9, .1, .3], [.4, .2, .8]], requires_grad=True)
        ref = torch.tensor([[.1, .4, .5], [.7, .6, .1]], requires_grad=True)
        value, info = sinkhorn_ot_loss(x, y, ref, iterations=100, tolerance=1e-7)
        with torch.no_grad():
            support = torch.cat((x, ref, y))
            scale = (torch.cdist(x, support).square().mean().sqrt()/3**.5).clamp_min(1e-3)
            expected = (ref.mean(0) - y.mean(0))[None] / (scale.square() * 3)
            expected_value = .5 * ((x-y).square().mean() - (x-ref).square().mean()) / scale.square()
        value.backward()
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(value, expected_value)
        self.assertIsNone(y.grad)
        self.assertIsNone(ref.grad)
        self.assertTrue(all(torch.isfinite(v) for v in info.values()))

    def test_fixed_iterations_do_not_check_convergence(self):
        # Even an already balanced problem must run exactly the requested count.
        cost = torch.zeros(5, 7, requires_grad=True)
        with mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('host convergence check')):
            with mock.patch('torch.logsumexp', wraps=torch.logsumexp) as reductions:
                plan, error, count = sinkhorn_cost_plan(cost, epsilon=.001, iterations=10, tolerance=0)
                self.assertEqual(count, 10)
                self.assertEqual(reductions.call_count, 20)
        self.assertFalse(plan.requires_grad)
        torch.testing.assert_close(plan, torch.full_like(plan, 1/35))
        self.assertLess(float(error), 1e-6)

    def test_explicit_epsilon_reaches_every_multiscale_transport_solve(self):
        x = torch.rand(4, 16, requires_grad=True)
        y, ref = torch.rand(7, 16), torch.rand(4, 16)
        with mock.patch('loss.sinkhorn_cost_plan', wraps=sinkhorn_cost_plan) as solver:
            value, info = multiscale_loss(x, y, (1, 4, 4), 8, objective='sinkhorn',
                reference_images=ref, sinkhorn_mode='ot', sinkhorn_iters=10,
                sinkhorn_tolerance=0, sinkhorn_ot_epsilon=.05,
                coarse_sizes=(2,), patch_sizes=(2,), crops_per_size=1)
            self.assertEqual(solver.call_count, 6)  # Three views, attraction and repulsion.
            self.assertTrue(all(call.kwargs['epsilon'] == .05 for call in solver.call_args_list))
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(info['sinkhorn_solver_iterations']), 10)

    def test_equal_supports_cancel_and_multiscale_updates(self):
        x = torch.rand(4, 16, requires_grad=True)
        y = torch.rand(7, 16)
        value, _ = sinkhorn_ot_loss(x, y, y, iterations=100)
        self.assertEqual(value.item(), 0)
        torch.testing.assert_close(torch.autograd.grad(value, x)[0], torch.zeros_like(x), rtol=0, atol=0)
        model = torch.nn.Sequential(torch.nn.Linear(3,16), torch.nn.Sigmoid())
        before=model[0].weight.detach().clone()
        options=dict(image_shape=(1,4,4),coarse_sizes=(2,),patch_sizes=(2,),
                     patch_rng=torch.Generator().manual_seed(4))
        metrics=generator_step(model,torch.randn(4,3),y,torch.optim.Adam(model.parameters(),lr=2e-4),8,
                               objective='sinkhorn',sinkhorn_latent=torch.randn(4,3),
                               sinkhorn_mode='ot',sinkhorn_iters=100,multiscale_options=options)
        self.assertFalse(torch.equal(model[0].weight,before))
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        self.assertIn('sinkhorn_ot_attraction',metrics)

    def test_training_and_resume(self):
        helper=test_priming.PrimingTests()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            common=['--training-stage','distributional','--objective','sinkhorn',
                    '--sinkhorn-mode','ot','--sinkhorn-iters','10','--sinkhorn-tolerance','0',
                    '--sinkhorn-ot-epsilon','.05','--multiscale-loss']
            full=helper.run_training(root/'full',common+['--steps','3'])
            helper.run_training(root/'resumed',common+['--steps','1'])
            resumed=helper.run_training(root/'resumed',common+['--steps','3','--resume',
                                         str(root/'resumed/checkpoint_latest.pt')])
            for key in full['model']:
                torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
            self.assertEqual(resumed['config']['sinkhorn_velocity_normalization'],'none')
            self.assertEqual(resumed['config']['sinkhorn_epsilon_mixture'], {'0.05': 1.})
            with (root/'resumed/metrics.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(rows[-1]['sinkhorn_marginal_max_error'])
            self.assertTrue(rows[-1]['sinkhorn_ot_attraction'])
            with (root/'resumed/training_diagnostics.csv').open() as stream:
                diagnostic_rows = list(csv.DictReader(stream))
            self.assertGreater(float(diagnostic_rows[-1]['parameter_update_relative_norm']), 0)
            self.assertEqual(float(diagnostic_rows[-1]['sinkhorn_solver_iterations']), 10)


if __name__=='__main__':
    unittest.main()
