"""Check perceptual gradients, frozen teachers, normalization, and CLI boundaries."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

import main
from perceptual import VGGPerceptualLoss
from priming import reconstruction_step


class TinyLPIPS(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 2, 1)
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, prediction, target, normalize=False):
        if not normalize:
            raise AssertionError('LPIPS must convert [0,1] inputs')
        a = self.features(prediction * 2 - 1)
        b = self.features(target * 2 - 1)
        return self.dropout((a - b).square()).mean((1, 2, 3), keepdim=True)


class PerceptualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_frozen_encoder_preserves_rng_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as cache, patch('lpips.LPIPS', TinyLPIPS):
            before = torch.get_rng_state().clone()
            hub = torch.hub.get_dir()
            perceptual = VGGPerceptualLoss(cache)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(hub, torch.hub.get_dir())
            self.assertFalse(perceptual.metric.training)
            prediction = torch.rand(2, 3072, requires_grad=True)
            target = torch.rand(2, 3072, requires_grad=True)
            value = perceptual(prediction, target)
            value.backward()
            self.assertGreater(prediction.grad.abs().sum().item(), 0)
            self.assertIsNone(target.grad)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in perceptual.parameters()))
            self.assertEqual(perceptual(prediction.detach(), prediction.detach()).item(), 0.)

    def test_weighted_update_and_zero_weight_compatibility(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sigmoid())
        expected = copy.deepcopy(model)
        clean, noise = torch.rand(3, 4), torch.randn(3, 4)
        def perceptual(a, b):
            return (a - b).abs().mean()
        prediction = expected(.2 * (2 * clean - 1) + .8 * noise)
        mse = (prediction - clean).square().mean()
        loss = mse + .1 * perceptual(prediction, clean)
        reference = torch.optim.SGD(expected.parameters(), lr=.1)
        loss.backward()
        reference.step()
        actual = reconstruction_step(model, clean, noise,
                                     torch.optim.SGD(model.parameters(), lr=.1),
                                     perceptual_loss=perceptual, perceptual_weight=.1)
        self.assertAlmostEqual(actual['reconstruction_total'], loss.item())
        self.assertAlmostEqual(actual['reconstruction_mse'], mse.item())
        for a, b in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)
        with patch('torch.nn.Module.forward', side_effect=AssertionError('unused')):
            info = reconstruction_step(model, clean, noise,
                                       torch.optim.SGD(model.parameters(), lr=.1),
                                       perceptual_loss=torch.nn.Module(), perceptual_weight=0.)
        self.assertEqual(set(info), {'reconstruction_mse'})

    def test_invalid_perceptual_stage_rejected(self):
        for extra in [[], ['--training-stage', 'reconstruction', '--dataset', 'mnist']]:
            with patch('sys.argv', ['main.py', '--lpips-weight', '.1', *extra]):
                with self.assertRaises(SystemExit):
                    main.parse_args()


if __name__ == '__main__':
    unittest.main()
