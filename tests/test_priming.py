"""Paired objective, stage handoff, and interrupted-run equivalence tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import main
from priming import reconstruction_step, reconstruction_error


class PrimingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_paired_corruption_and_update(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sigmoid())
        expected = copy.deepcopy(model)
        clean, noise = torch.rand(3, 4), torch.randn(3, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        reference = torch.optim.SGD(expected.parameters(), lr=.1)
        loss = (expected(.2 * (2 * clean - 1) + .8 * noise) - clean).square().mean()
        loss.backward()
        reference.step()
        actual = reconstruction_step(model, clean, noise, optimizer, .8)
        self.assertAlmostEqual(actual['reconstruction_mse'], loss.item())
        for p, q in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(p, q)

    def run_training(self, output, extra=()):
        # Small MNIST MLP exercises the real trainer without datasets or FID.
        data = torch.arange(6 * 784).reshape(6, 784).remainder(256).byte()
        labels = torch.zeros(6, dtype=torch.long)
        argv = ['main.py', '--dataset', 'mnist', '--architecture', 'mlp',
                '--latent-dim', '784', '--hidden-dims', '8', '--target-size', '0',
                '--target-batch-size', '2', '--particles', '3', '--eval-size', '4',
                '--epochs', '2', '--save-every', '1', '--checkpoint-every', '1',
                '--training-stage', 'reconstruction', '--ema-decay', '.9',
                '--threads', '1', '--device', 'cpu', '--output', str(output), *extra]
        if '--steps' in extra:
            index = argv.index('--epochs')
            del argv[index:index + 2]
        with patch('sys.argv', argv), patch.object(main, 'load_dataset', return_value=(
                Path('/unused'), (data, labels), (data, labels))), \
                patch.object(main, 'save_grid'), patch.object(main, 'save_trajectory'), \
                patch.object(main, 'plot_mmd'):
            main.main()
        return torch.load(output / 'checkpoint_latest.pt', weights_only=False)

    def test_heldout_diagnostic_preserves_rng_and_training_mode(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.5), torch.nn.Sigmoid())
        clean = torch.rand(7, 4)
        state = torch.get_rng_state().clone()
        first = reconstruction_error(model, clean, .8, 123, 3)
        second = reconstruction_error(model, clean, .8, 123, 3)
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_resume_and_stage_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.run_training(root / 'full')
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:  # First update after the epoch-one checkpoint.
                    raise RuntimeError('simulated interruption')
                return reconstruction_step(*args, **kwargs)

            with patch.object(main, 'reconstruction_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    self.run_training(root / 'resumed')
            resumed = self.run_training(root / 'resumed',
                                        ['--resume', str(root / 'resumed/checkpoint_latest.pt')])
            self.assertEqual(resumed['update_count'], 6)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            for key in ('torch_rng', 'data_rng', 'latent_rng'):
                self.assertTrue(torch.equal(full[key], resumed[key]))

            handoff = self.run_training(root / 'post', [
                '--training-stage', 'distributional', '--epochs', '0',
                '--init-checkpoint', str(root / 'full/checkpoint_latest.pt')])
            self.assertEqual(handoff['optimizer']['state'], {})
            self.assertEqual(handoff['update_count'], 0)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], handoff[key][name], rtol=0, atol=0)
            trained = self.run_training(root / 'post_trained', [
                '--training-stage', 'distributional', '--epochs', '1',
                '--init-checkpoint', str(root / 'full/checkpoint_latest.pt')])
            self.assertEqual(trained['update_count'], 3)
            self.assertTrue(any(not torch.equal(full['model'][name], trained['model'][name])
                                for name in full['model']))
            with self.assertRaisesRegex(ValueError, 'mismatch for latent_dim'):
                self.run_training(root / 'bad', ['--training-stage', 'distributional',
                    '--latent-dim', '32', '--init-checkpoint', str(root / 'full/checkpoint_latest.pt')])

    def test_extend_completed_budget_matches_uninterrupted(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage, duration in [('reconstruction', '--epochs'), ('distributional', '--steps')]:
                common = ['--training-stage', stage, duration, '4']
                full = self.run_training(root / f'{stage}_full', common)
                output = root / stage
                self.run_training(output, ['--training-stage', stage, duration, '2'])
                resumed = self.run_training(output, common + [
                    '--resume', str(output / 'checkpoint_latest.pt')])
                for key in ('model', 'ema_model'):
                    for name in full[key]:
                        torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
                self.assertEqual(json.loads((output / 'config.json').read_text())['total_updates'],
                                 full['update_count'])
                with self.assertRaisesRegex(ValueError, 'cannot shorten'):
                    self.run_training(output, ['--training-stage', stage, duration, '2',
                        '--resume', str(output / 'checkpoint_latest.pt')])

    def test_tuned_fork_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ['--training-stage', 'distributional', '--steps', '2', '--multiscale-loss']
            source = self.run_training(root / 'source', base)
            tuned = base + ['--steps', '4', '--fork-checkpoint',
                           str(root / 'source/checkpoint_latest.pt'), '--schedule-start-update', '2',
                           '--learning-rate', '.0001', '--min-learning-rate', '.00005',
                           '--lr-schedule', 'cosine', '--warmup-steps', '1', '--train-particles', '6',
                           '--random-flip', '--max-grad-norm', '1', '--ema-decay', '.99',
                           '--crops-per-size', '2', '--global-loss-weight', '2',
                           '--keep-checkpoint-every', '3']
            full = self.run_training(root / 'full', tuned)
            self.assertEqual(full['config']['training_generated_batch'], 6)
            self.assertEqual(full['arguments']['particles'], 3)
            self.assertEqual(full['config']['tuning_initialization']['source_update'], 2)
            retained = torch.load(root / 'full/checkpoint_step_000003.pt', weights_only=False)
            self.assertEqual(retained['update_count'], 3)
            self.assertTrue((root / 'full/checkpoint_step_000004.pt').exists())
            self.assertEqual(full['optimizer']['state'][0]['step'].item(), 4)
            calls = 0
            original = main.generator_step
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                self.assertEqual(len(args[1]), 6)
                if calls == 2:
                    raise RuntimeError('tuned interruption')
                return original(*args, **kwargs)
            with patch.object(main, 'generator_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'tuned interruption'):
                    self.run_training(root / 'resumed', tuned)
            resume = list(tuned)
            i = resume.index('--fork-checkpoint')
            del resume[i:i + 2]
            resume += ['--resume', str(root / 'resumed/checkpoint_latest.pt')]
            resumed = self.run_training(root / 'resumed', resume)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            for key in ('torch_rng', 'data_rng', 'latent_rng', 'patch_rng'):
                self.assertTrue(torch.equal(full[key], resumed[key]))
            with self.assertRaisesRegex(ValueError, 'mismatch for learning_rate'):
                self.run_training(root / 'source', base + [
                    '--resume', str(root / 'source/checkpoint_latest.pt'), '--learning-rate', '.0001'])
            unchanged = torch.load(root / 'source/checkpoint_latest.pt', weights_only=False)
            self.assertEqual(unchanged['update_count'], source['update_count'])

    def test_continuation_schedule_and_flip(self):
        schedule = dict(schedule='cosine', minimum=5e-5, start=10, total=30)
        self.assertEqual(main.training_learning_rate(10, 1e-4, 2, **schedule), 1e-6)
        self.assertAlmostEqual(main.training_learning_rate(12, 1e-4, 2, **schedule), 1e-4)
        self.assertAlmostEqual(main.training_learning_rate(30, 1e-4, 2, **schedule), 5e-5)
        batch = torch.arange(24).reshape(2, 12)
        with patch.object(torch, 'rand', return_value=torch.tensor([.1, .9])):
            result = main.horizontal_flip_batch(batch, (1, 3, 4))
        torch.testing.assert_close(result[0], batch[0].reshape(3, 4).flip(-1).flatten())
        torch.testing.assert_close(result[1], batch[1])


if __name__ == '__main__':
    unittest.main()
