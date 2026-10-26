#!/usr/bin/env python3
"""Standard epsilon-prediction DDPM sanity check using the existing CIFAR backbone.

Recipe reference: https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py
Uses 1000 linear-beta steps, fixed-large reverse variance, clipped x0, Adam,
5000-step warmup, gradient clipping, random horizontal flips, and EMA.
The spatial network is this repository's DDPM++ adaptation, not Ho's U-Net.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import time

import torch

from backbone import CIFAR10UNetParticleGenerator
from run_storage import new_run_path
from data import download_cifar10, load_cifar10_split
from visualization import save_grid


class Diffusion:
    def __init__(self, device='cpu', timesteps=1000):
        self.timesteps = timesteps
        beta = torch.linspace(1e-4, .02, timesteps, dtype=torch.float64, device=device)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat([torch.ones_like(abar[:1]), abar[:-1]])
        self.beta = beta.float()
        self.signal = abar.sqrt().float()
        self.noise = (1 - abar).sqrt().float()
        self.coef_x0 = (beta * previous.sqrt() / (1 - abar)).float()
        self.coef_xt = ((1 - previous) * alpha.sqrt() / (1 - abar)).float()

    def corrupt(self, clean, timesteps, noise):
        return (self.signal[timesteps, None, None, None] * clean
                + self.noise[timesteps, None, None, None] * noise)

    def reverse_mean(self, noisy, predicted_noise, timestep):
        clean = ((noisy - self.noise[timestep] * predicted_noise)
                 / self.signal[timestep]).clamp(-1, 1)
        return self.coef_x0[timestep] * clean + self.coef_xt[timestep] * noisy

    @torch.inference_mode()
    def sample(self, model, count, *, generator):
        device = self.beta.device
        was_training = model.training
        model.eval()
        try:
            x = torch.randn(count, 3, 32, 32, device=device, generator=generator)
            for step in reversed(range(self.timesteps)):
                t = torch.full((count,), step, device=device, dtype=torch.long)
                x = self.reverse_mean(x, model(x, t), step)
                if step:
                    x = x + self.beta[step].sqrt() * torch.randn(
                        x.shape, device=device, generator=generator)
            return ((x + 1) / 2).clamp(0, 1)
        finally:
            model.train(was_training)


@torch.no_grad()
def update_ema(model, ema, decay):
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.lerp_(current, 1 - decay)


def train_step(model, ema, optimizer, diffusion, clean, step, *, lr, warmup, decay):
    rate = lr * min(step / max(warmup, 1), 1)
    for group in optimizer.param_groups:
        group['lr'] = rate
    flips = torch.rand(len(clean), device=clean.device) < .5
    clean = torch.where(flips[:, None, None, None], clean.flip(-1), clean)
    t = torch.randint(diffusion.timesteps, (len(clean),), device=clean.device)
    noise = torch.randn_like(clean)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(diffusion.corrupt(clean, t, noise), t)
    loss = (prediction - noise).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f'Nonfinite DDPM loss at update {step}')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    update_ema(model, ema, decay)
    return loss.item(), grad.item(), rate


@torch.inference_mode()
def validation_loss(model, diffusion, clean):
    rng = torch.Generator(device=clean.device).manual_seed(9876)
    t = torch.randint(diffusion.timesteps, (len(clean),), device=clean.device, generator=rng)
    noise = torch.randn(clean.shape, device=clean.device, generator=rng)
    prediction = model(diffusion.corrupt(clean, t, noise), t)
    return (prediction - noise).square().mean().item()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--base-channels', type=int, default=128)
    parser.add_argument('--num-res-blocks', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--preview-every', type=int, default=5000)
    parser.add_argument('--fid-samples', type=int, default=10000)
    parser.add_argument('--sample-batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not args.resume:
        args.output = new_run_path(args.output)
    if min(args.steps, args.batch_size, args.checkpoint_every, args.preview_every,
           args.sample_batch_size, args.base_channels, args.num_res_blocks) < 1:
        parser.error('Training counts and intervals must be positive')
    if args.fid_samples < 0 or args.fid_samples == 1:
        parser.error('FID sample count must be zero (smoke test) or at least two')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists() and args.resume is None:
        raise ValueError('Existing run: use --resume or a fresh output directory')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    archive = download_cifar10(args.data_root)
    raw, _ = load_cifar10_split(archive, True)
    heldout, _ = load_cifar10_split(archive, False)
    # Dataset fits on the allocated H200; shuffled passes retain the final short batch.
    clean = raw.to(device=device, dtype=torch.float32).reshape(-1, 3, 32, 32) / 127.5 - 1
    heldout = heldout[:128].to(device=device, dtype=torch.float32).reshape(-1, 3, 32, 32) / 127.5 - 1
    model = CIFAR10UNetParticleGenerator(args.base_channels, args.num_res_blocks,
                                        time_conditioned=True).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    diffusion = Diffusion(device)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                 if k != 'resume'}
    config = dict(arguments=arguments, parameters=sum(p.numel() for p in model.parameters()),
                  objective='epsilon_prediction_mse', diffusion_steps=1000,
                  beta_schedule='linear', beta_start=.0001, beta_end=.02,
                  sampling='ancestral_ddpm_fixed_large_variance_clip_x0',
                  time_conditioning=True, output_activation='identity',
                  backbone='existing CIFAR DDPM++ spatial topology',
                  fid_reference='all 50000 CIFAR10 training images', fid_seed=12345,
                  validation='fixed 128 test images, noise, and timesteps; EMA model',
                  purpose='architecture sanity check; training budget does not imply convergence')
    start, elapsed_offset, history = 0, 0., []
    data_rng = torch.Generator().manual_seed(args.seed + 1)
    permutation, cursor = torch.randperm(len(clean), generator=data_rng), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        previous_arguments = dict(checkpoint['arguments'])
        if args.steps < previous_arguments['steps']:
            raise ValueError('Resume cannot shorten steps')
        previous_arguments['steps'] = args.steps
        if previous_arguments != arguments:
            raise ValueError('Resume arguments differ from the checkpoint')
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, elapsed_offset = checkpoint['update_count'], checkpoint['elapsed_seconds']
        history = checkpoint['history']
        permutation, cursor = checkpoint['permutation'], checkpoint['cursor']
        data_rng.set_state(checkpoint['data_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        del checkpoint
    write_json(args.output / 'config.json', config)
    print(f'DDPM parameters={config["parameters"]} steps={args.steps} device={device}', flush=True)
    started = time.monotonic()
    def elapsed():
        return elapsed_offset + time.monotonic() - started

    def save_checkpoint(step):
        value = dict(arguments=arguments, model=model.state_dict(), ema_model=ema.state_dict(),
                     optimizer=optimizer.state_dict(), update_count=step, elapsed_seconds=elapsed(),
                     history=history, permutation=permutation, cursor=cursor,
                     data_rng=data_rng.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        path = args.output / 'checkpoint_latest.pt'
        torch.save(value, path.with_suffix('.pt.tmp'))
        path.with_suffix('.pt.tmp').replace(path)

    fields = ['update', 'noise_mse', 'grad_norm', 'learning_rate', 'validation_noise_mse', 'elapsed_seconds']
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
        total_loss, total_grad, count = 0., 0., 0
        for step in range(start + 1, args.steps + 1):
            if cursor == len(clean):
                permutation, cursor = torch.randperm(len(clean), generator=data_rng), 0
            indices = permutation[cursor:cursor + args.batch_size].to(device)
            cursor += len(indices)
            loss, grad, lr = train_step(model, ema, optimizer, diffusion, clean[indices], step,
                                       lr=args.learning_rate, warmup=args.warmup, decay=args.ema_decay)
            total_loss += loss
            total_grad += grad
            count += 1
            if step % 100 == 0 or step % args.checkpoint_every == 0 or step == args.steps:
                validation = (validation_loss(ema, diffusion, heldout)
                              if step % args.checkpoint_every == 0 or step == args.steps else '')
                row = dict(update=step, noise_mse=total_loss / count, grad_norm=total_grad / count,
                           learning_rate=lr, validation_noise_mse=validation, elapsed_seconds=elapsed())
                writer.writerow(row)
                stream.flush()
                history.append(row)
                print(f'step={step} noise_MSE={row["noise_mse"]:.6f} '
                      f'validation={validation} elapsed={elapsed():.1f}s', flush=True)
                total_loss, total_grad, count = 0., 0., 0
            if step % args.checkpoint_every == 0 or step == args.steps:
                save_checkpoint(step)
            if step % args.preview_every == 0 or step == args.steps:
                print(f'Sampling 64 preview images at step {step} (1000 DDPM steps)', flush=True)
                rng = torch.Generator(device=device).manual_seed(12345)
                images = diffusion.sample(ema, 64, generator=rng)
                save_grid(images, args.output / f'samples_step_{step:06d}.png', image_shape=(3, 32, 32))

    fid_path = args.output / 'fid_metrics.json'
    fid_history = json.loads(fid_path.read_text()) if fid_path.exists() else []
    if args.fid_samples and not any(row['iteration'] == args.steps for row in fid_history):
        from fid import CIFAR10FID, feature_statistics
        evaluator = CIFAR10FID(raw, device=device, batch_size=args.sample_batch_size)
        rng = torch.Generator(device=device).manual_seed(12345)
        def batches():
            for offset in range(0, args.fid_samples, args.sample_batch_size):
                n = min(args.sample_batch_size, args.fid_samples - offset)
                images = diffusion.sample(ema, n, generator=rng)
                print(f'FID sampling {offset + n}/{args.fid_samples} elapsed={elapsed():.1f}s', flush=True)
                yield images
        statistics = feature_statistics(batches(), evaluator.inception, device)
        fid = evaluator.score(statistics)
        fid_history.append(dict(iteration=args.steps, fid=fid, generated_count=statistics[2],
                                reference_count=len(raw), sampling_steps=1000))
        write_json(fid_path, fid_history)
        print(f'step={args.steps} FID={fid:.6f} ({args.fid_samples} generated images)', flush=True)
    print(f'Finished DDPM sanity check. Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
