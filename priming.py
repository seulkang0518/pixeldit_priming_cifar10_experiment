"""Reconstruction priming for the existing one-pass generator.

Adapted from the two-stage idea in https://arxiv.org/abs/2610.20331,
Eq. (1). No upstream implementation is copied. The generator retains its
[0,1] sigmoid output; corruption uses centered [-1,1] clean inputs.
"""
from pathlib import Path
import hashlib
import math
import torch


def reconstruction_step(model, clean, noise, optimizer, corruption_level=0.8,
                        max_grad_norm=0.0, *, perceptual_loss=None, perceptual_weight=0.0):
    """One paired denoising update; no distributional field or timestep input."""
    if clean.shape != noise.shape:
        raise ValueError('Reconstruction noise and clean images must have matching shapes')
    if not 0 <= corruption_level <= 1:
        raise ValueError('corruption_level must be in [0, 1]')
    if not math.isfinite(perceptual_weight) or perceptual_weight < 0:
        raise ValueError('Perceptual weight must be finite and nonnegative')
    if perceptual_weight and perceptual_loss is None:
        raise ValueError('Positive perceptual weight requires a perceptual loss')
    optimizer.zero_grad(set_to_none=True)
    corrupted = (1 - corruption_level) * (2 * clean - 1) + corruption_level * noise
    prediction = model(corrupted)
    mse = torch.nn.functional.mse_loss(prediction, clean)
    loss = mse
    diagnostics = {'reconstruction_mse': mse.detach().item()}
    if perceptual_weight:
        perceptual = perceptual_loss(prediction, clean)
        loss = mse + perceptual_weight * perceptual
        diagnostics.update(reconstruction_lpips=perceptual.detach().item(),
                           reconstruction_total=loss.detach().item())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite reconstruction loss')
    loss.backward()
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return diagnostics


def initialize_from_checkpoint(model, ema_model, path, args):
    """Transfer online and EMA weights, deliberately leaving optimizer/RNG fresh."""
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = checkpoint['arguments']
    if source.get('training_stage') != 'reconstruction':
        raise ValueError('Priming initialization requires a reconstruction checkpoint')
    for key in ('dataset', 'architecture', 'latent_dim', 'base_channels', 'num_res_blocks',
                'hidden_dims', 'class_id'):
        if source.get(key) != getattr(args, key):
            raise ValueError(f'Priming checkpoint mismatch for {key}')
    if args.architecture == 'dit':
        for key in ('dit_width', 'dit_depth', 'dit_heads', 'dit_patch_size'):
            if source.get(key) != getattr(args, key):
                raise ValueError(f'Priming checkpoint mismatch for {key}')
    if args.architecture == 'pixeldit':
        for key in ('pixeldit_width', 'pixeldit_patch_depth', 'pixeldit_pixel_depth',
                    'pixeldit_heads', 'pixeldit_patch_size', 'pixeldit_pixel_width'):
            if source.get(key) != getattr(args, key):
                raise ValueError(f'Priming checkpoint mismatch for {key}')
    duration = source['epochs'] if source['epochs'] is not None else source['steps']
    if checkpoint['iteration'] != duration:
        raise ValueError('Priming initialization requires a completed reconstruction stage')
    model.load_state_dict(checkpoint['model'], strict=True)
    if ema_model is not None:
        shadow = checkpoint.get('ema_model')
        ema_model.load_state_dict(shadow if shadow is not None else checkpoint['model'], strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(checkpoint=str(path.resolve()), sha256=digest,
                source_updates=checkpoint['update_count'], source_iteration=checkpoint['iteration'],
                source_perceptual=checkpoint.get('config', {}).get('reconstruction_perceptual'),
                online_weights=True, inherited_ema=checkpoint.get('ema_model') is not None,
                optimizer='fresh', random_streams='fresh_post_training_seed')


@torch.no_grad()
def reconstruction_error(model, clean, corruption_level, seed, batch_size):
    """Fixed held-out corruption; preserves model mode and all training RNGs."""
    training = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    squared_error = 0.0
    try:
        for batch in clean.split(batch_size):
            noise = torch.randn(batch.shape, generator=rng).to(batch.device)
            inputs = (1 - corruption_level) * (2 * batch - 1) + corruption_level * noise
            squared_error += (model(inputs) - batch).square().sum().item()
    finally:
        model.train(training)
    return squared_error / clean.numel()
