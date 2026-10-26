#!/usr/bin/env python3
"""Train one-pass image generators with Riesz, Drifting, or Sinkhorn objectives on MNIST or CIFAR-10."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch

from backbone import build_backbone, backbone_metadata
from run_storage import new_run_path
from data import (FILES, CIFAR10_ARCHIVES, CIFAR10_CLASSES, download_mnist,
                  download_cifar10, load_dataset, image_shape as dataset_image_shape)
from loss import (backpropagate_objective, mean_distance, mmd_squared, velocity,
                  _riesz_topk_softmax_loss_impl, drifting_loss, sinkhorn_target_loss,
                  raw_field_diagnostics, sinkhorn_ot_loss)
from visualization import save_grid, save_trajectory, plot_mmd
from multiscale import multiscale_loss
from priming import reconstruction_step, initialize_from_checkpoint, reconstruction_error


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    """Average post-update parameters; copy non-parameter state from the live model."""
    for averaged, current in zip(ema_model.parameters(), model.parameters()):
        averaged.lerp_(current.detach(), 1 - decay)
    for averaged, current in zip(ema_model.buffers(), model.buffers()):
        averaged.copy_(current)


@torch.no_grad()
def evaluation_images(model: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Disable dropout for diagnostics without changing training mode or its RNG."""
    training = model.training
    model.eval()
    try:
        return model(latent)
    finally:
        model.train(training)


def generator_step(model: torch.nn.Module, latent: torch.Tensor, target: torch.Tensor,
                   optimizer: torch.optim.Optimizer, chunk_size: int, *,
                   objective: str = "riesz", tau_list: tuple[float, ...] = (0.2, 0.05, 0.02),
                   normalize_per_temperature: bool = True,
                   multiscale_options: dict | None = None,
                   target_options: dict | None = None,
                   sinkhorn_latent: torch.Tensor | None = None,
                   sinkhorn_epsilon: float = 0.05, sinkhorn_iters: int = 10,
                   max_grad_norm: float = 0.0,
                   no_field_rms_normalization: bool = False,
                   sinkhorn_mode: str = 'wflow', sinkhorn_tolerance: float = 1e-6,
                   sinkhorn_ot_epsilon: float | None = None) -> dict:
    """Backpropagate the chosen particle-field rule through the generator.

    Ordinary Riesz uses the exact half-MMD image gradient -velocity/N.
    Softmax Riesz uses upstream's scaled, detached-goal regression loss.
    Only model parameters are optimized, never latents or output particles.
    """
    optimizer.zero_grad(set_to_none=True)
    images = model(latent)
    reference_images = None
    if objective == 'sinkhorn':
        if sinkhorn_latent is None:
            raise ValueError('Sinkhorn training needs a second independently sampled latent batch')
        with torch.no_grad():
            reference_images = model(sinkhorn_latent)
    diagnostics = {}
    if multiscale_options is None and objective == 'sinkhorn':
        if sinkhorn_mode == 'ot':
            loss, diagnostics = sinkhorn_ot_loss(images, target, reference_images,
                iterations=sinkhorn_iters, tolerance=sinkhorn_tolerance, epsilon=sinkhorn_ot_epsilon)
        else:
            loss, diagnostics = sinkhorn_target_loss(
                images, target, reference_images, epsilon=sinkhorn_epsilon, iterations=sinkhorn_iters)
        loss.backward()
    elif multiscale_options is None and target_options:
        from target_updates import softmax_target_loss
        loss, diagnostics = softmax_target_loss(
            images, target, tau_list=tau_list,
            normalize_per_temperature=normalize_per_temperature,
            no_field_rms_normalization=no_field_rms_normalization, **target_options)
        loss.backward()
    elif multiscale_options is None and objective == 'softmax-riesz' and no_field_rms_normalization:
        loss, info = _riesz_topk_softmax_loss_impl(
            images.unsqueeze(0), target.unsqueeze(0), tau_list=tau_list,
            normalize_per_temperature=normalize_per_temperature,
            no_field_rms_normalization=True, epsilon=1e-6)
        loss.mean().backward()
        diagnostics = raw_field_diagnostics(info)
    elif multiscale_options is None:
        backpropagate_objective(images, target, chunk_size, objective=objective,
                                tau_list=tau_list, normalize_per_temperature=normalize_per_temperature)
    else:
        loss, diagnostics = multiscale_loss(
            images, target, chunk_size=chunk_size, objective=objective,
            tau_list=tau_list, normalize_per_temperature=normalize_per_temperature,
            target_options=target_options, reference_images=reference_images,
            sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=sinkhorn_iters,
            no_field_rms_normalization=no_field_rms_normalization,
            sinkhorn_mode=sinkhorn_mode, sinkhorn_tolerance=sinkhorn_tolerance,
            sinkhorn_ot_epsilon=sinkhorn_ot_epsilon,
            **multiscale_options)
        loss.backward()
    record_sinkhorn_diagnostics = objective == 'sinkhorn'
    if max_grad_norm or no_field_rms_normalization or record_sinkhorn_diagnostics:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm or float('inf'),
            error_if_nonfinite=no_field_rms_normalization or record_sinkhorn_diagnostics)
        if no_field_rms_normalization or record_sinkhorn_diagnostics:
            diagnostics.update(grad_norm_before_clip=norm.detach(),
                               clip_fraction=(norm > max_grad_norm).float() if max_grad_norm
                               else norm.new_zeros(()))
    before_step = [p.detach().clone() for p in model.parameters()] if record_sinkhorn_diagnostics else None
    optimizer.step()
    if before_step is not None:
        with torch.no_grad():
            delta_squared = torch.stack([(p - old).square().sum()
                                         for p, old in zip(model.parameters(), before_step)]).sum()
            weight_squared = torch.stack([old.square().sum() for old in before_step]).sum()
            diagnostics['parameter_update_relative_norm'] = (delta_squared / weight_squared.clamp_min(1e-30)).sqrt()
    return diagnostics



def snapshot_steps(steps: int, intervals: int) -> set[int]:
    return {0, *(math.ceil(steps * index / intervals) for index in range(1, intervals + 1))}


def training_learning_rate(update: int, peak: float, warmup_steps: int, *,
                           schedule='constant', minimum=0.0, start=0, total=None) -> float:
    """Released W-Flow/Drifting schedule, indexed by completed updates."""
    age = max(0, update - start)
    if age < warmup_steps:
        return 1e-6 + (peak - 1e-6) * age / warmup_steps
    if schedule == 'cosine':
        progress = min(1.0, (age - warmup_steps) / max(1, total - start - warmup_steps))
        return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * progress))
    return peak


def horizontal_flip_batch(batch, image_shape):
    images = batch.reshape(-1, *image_shape)
    flips = torch.rand(len(images), device=batch.device) < .5
    return torch.where(flips[:, None, None, None], images.flip(-1), images).reshape_as(batch)



def epoch_batches(size: int, batch_size: int, rng: torch.Generator):
    """Visit every target exactly once in a freshly shuffled epoch, without dropping the tail."""
    order = torch.randperm(size, generator=rng)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]



def parse_args(default_dataset: str = "cifar10") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/zongchen/data"))
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), default=default_dataset)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--target-size", type=int, default=2048,
                        help="Fixed training subset; 0 uses all eligible training images")
    parser.add_argument("--target-batch-size", type=int, default=0,
                        help="0: full-target field; positive: minibatch size (exhaustive shuffled batches in epoch mode)")
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--fid-samples", type=int, default=10000,
                        help="CIFAR-10 generated images for FID at saved snapshots")
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--digit", type=int, choices=range(10), default=None)
    parser.add_argument("--class-id", type=int, choices=range(10), default=None)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, default=None, help="Number of individual updates")
    duration.add_argument("--epochs", type=int, default=None, help="Number of full passes over the target dataset")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--min-learning-rate', type=float, default=0.0)
    parser.add_argument('--schedule-start-update', type=int, default=0)
    parser.add_argument('--train-particles', type=int,
                        help='Generated training batch; defaults to --particles, which also controls diagnostics')
    parser.add_argument('--random-flip', action='store_true', help='Horizontal flips on training images only')
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="Adam")
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear warmup from 1e-6, then constant LR; counted in optimizer updates")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="Clip parameter gradient norm; 0 disables")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay per optimizer update; default 0.999 for CIFAR, 0 for MNIST; 0 disables")
    parser.add_argument("--architecture", choices=("unet", "mlp", "dit", "pixeldit"), default="unet")
    parser.add_argument('--data-parallel', action='store_true',
                        help='Split generator forwards across visible CUDA GPUs; retain one full-population loss')
    parser.add_argument('--pixeldit-width', type=int, default=768)
    parser.add_argument('--pixeldit-patch-depth', type=int, default=12)
    parser.add_argument('--pixeldit-pixel-depth', type=int, default=2)
    parser.add_argument('--pixeldit-heads', type=int, default=12)
    parser.add_argument('--pixeldit-patch-size', type=int, default=4)
    parser.add_argument('--pixeldit-pixel-width', type=int, default=16)
    parser.add_argument('--dit-width', type=int, default=384)
    parser.add_argument('--dit-depth', type=int, default=12)
    parser.add_argument('--dit-heads', type=int, default=6)
    parser.add_argument('--dit-patch-size', type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-res-blocks", type=int, default=4, help="Residual blocks per CIFAR U-Net stage")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Spatial U-Net noise: 784 for MNIST, 3072 for CIFAR-10; MLP: 32")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--objective", choices=("softmax-riesz", "riesz", "sinkhorn", "drifting"), default="softmax-riesz")
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05,
                        help="Dimension-relative entropic regularization for quadratic Sinkhorn")
    parser.add_argument("--sinkhorn-iters", type=int, default=10)
    parser.add_argument('--sinkhorn-mode', choices=('wflow', 'ot'), default='wflow',
                        help='Normalized W-Flow regression or direct detached-plan OT epsilon mixture')
    parser.add_argument('--sinkhorn-tolerance', type=float, default=1e-6,
                        help='Absolute marginal tolerance for direct OT; zero uses all iterations')
    parser.add_argument('--sinkhorn-ot-epsilon', type=float, default=None,
                        help='Single direct-OT regularization; omitted retains the historical .001/.01 mixture')
    parser.add_argument("--tau-list", type=float, nargs="+", default=[0.2, 0.05, 0.02])
    parser.add_argument("--normalize-per-temperature", action=argparse.BooleanOptionalAction,
                        default=True, help="Normalize each softmax field before summing")
    parser.add_argument('--no-field-rms-normalization', action='store_true',
                        help='Bypass both softmax field RMS normalizations; retain batch-distance scaling')
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--multiscale-loss", action="store_true",
                        help="Combine full-image, coarse-image and aligned patch objectives")
    parser.add_argument("--coarse-sizes", type=int, nargs='+', default=[8, 16])
    parser.add_argument("--patch-sizes", type=int, nargs='+', default=[7, 14])
    parser.add_argument("--crops-per-size", type=int, default=1)
    parser.add_argument("--global-loss-weight", type=float, default=1.0)
    parser.add_argument("--coarse-loss-weight", type=float, default=1.0)
    parser.add_argument("--patch-loss-weight", type=float, default=1.0)
    parser.add_argument("--field-step", type=float, default=1.0,
                        help="Step size in the detached softmax regression target")
    parser.add_argument("--noise-levels", type=float, nargs='+', default=[0.0],
                        help="Gaussian standard deviations in each pixel view; losses are averaged")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save a resumable latest checkpoint every N iterations; 0 disables")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume the same output directory from its latest checkpoint")
    parser.add_argument('--fork-checkpoint', type=Path,
                        help='Continue in a new output directory with explicitly changed tuning settings')
    parser.add_argument('--keep-checkpoint-every', type=int, default=0,
                        help='Retain numbered checkpoints at this interval, plus the final checkpoint')
    parser.add_argument("--training-stage", choices=("distributional", "reconstruction"),
                        default="distributional")
    parser.add_argument("--corruption-level", type=float, default=0.8,
                        help="Fixed reconstruction corruption: (1-s)*(2*image-1) + s*noise")
    parser.add_argument('--lpips-weight', type=float, default=0.0,
                        help='Frozen VGG LPIPS weight for CIFAR reconstruction only')
    parser.add_argument('--lpips-cache', type=Path, default=None,
                        help='Explicit pretrained-weight cache for perceptual priming')
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Start a new distributional run from online weights and inherited EMA; fresh optimizer")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--save-snapshots", type=int, default=None,
                        help="Save this many evenly spaced intervals plus initialization; overrides save-every")
    parser.add_argument("--trajectory-particles", type=int, default=8,
                        help="Number of fixed particle identities shown in trajectory.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.resume and args.fork_checkpoint:
        parser.error('--resume and --fork-checkpoint are mutually exclusive')
    if (args.train_particles is not None and args.train_particles < 1
            or args.keep_checkpoint_every < 0 or args.schedule_start_update < 0):
        parser.error('Invalid training batch, checkpoint retention, or schedule start')
    if (not 0 <= args.adam_beta2 < 1 or args.warmup_steps < 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm < 0):
        parser.error('Invalid optimizer, warmup, or gradient clipping settings')
    if not math.isfinite(args.sinkhorn_epsilon) or args.sinkhorn_epsilon <= 0 or args.sinkhorn_iters < 1:
        parser.error('Sinkhorn epsilon and iterations must be positive')
    if not math.isfinite(args.sinkhorn_tolerance) or args.sinkhorn_tolerance < 0:
        parser.error('Sinkhorn tolerance must be finite and nonnegative')
    if args.sinkhorn_mode == 'ot' and args.objective != 'sinkhorn':
        parser.error('--sinkhorn-mode ot requires --objective sinkhorn')
    if args.sinkhorn_ot_epsilon is not None:
        if not math.isfinite(args.sinkhorn_ot_epsilon) or args.sinkhorn_ot_epsilon <= 0:
            parser.error('Direct OT epsilon must be finite and positive')
        if args.objective != 'sinkhorn' or args.sinkhorn_mode != 'ot':
            parser.error('--sinkhorn-ot-epsilon requires --objective sinkhorn --sinkhorn-mode ot')
    if (not math.isfinite(args.field_step) or args.field_step <= 0 or
            args.checkpoint_every < 0 or
            any(not math.isfinite(s) or s < 0 for s in args.noise_levels)):
        parser.error("invalid field step, noise levels, or checkpoint interval")
    if args.objective == 'drifting' and not args.normalize_per_temperature:
        parser.error('The official Drifting loss requires per-temperature normalization')
    if args.no_field_rms_normalization:
        if args.objective != 'softmax-riesz' or args.training_stage != 'distributional':
            parser.error('--no-field-rms-normalization requires distributional softmax-riesz')
        args.normalize_per_temperature = False
    if args.objective != 'softmax-riesz' and (
            args.field_step != 1 or args.noise_levels != [0.0]):
        parser.error("target-update experiments require --objective softmax-riesz")
    if args.fid_samples < 2 or args.fid_batch_size < 1:
        parser.error("FID needs at least two samples and a positive batch size")
    if args.learning_rate is None:
        args.learning_rate = 1e-3 if args.dataset == "mnist" else 2e-4
    if not math.isfinite(args.min_learning_rate) or not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error('Minimum learning rate must be between zero and the peak learning rate')
    if args.ema_decay is None:
        args.ema_decay = 0.999 if args.dataset == "cifar10" else 0.0
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be finite and in [0, 1)")
    image_dim = 784 if args.dataset == "mnist" else 3072
    if not math.isfinite(args.corruption_level) or not 0 <= args.corruption_level <= 1:
        parser.error('--corruption-level must be finite and in [0, 1]')
    if not math.isfinite(args.lpips_weight) or args.lpips_weight < 0:
        parser.error('--lpips-weight must be finite and nonnegative')
    if args.lpips_weight and (args.training_stage != 'reconstruction' or args.dataset != 'cifar10'):
        parser.error('LPIPS is only supported for CIFAR reconstruction pretraining')
    if args.lpips_weight and args.lpips_cache is None:
        parser.error('--lpips-cache is required with positive --lpips-weight')
    if args.init_checkpoint and args.training_stage != 'distributional':
        parser.error('--init-checkpoint is for distributional post-training')
    if args.output is None:
        args.output = Path("results") / ("mnist" if args.dataset == "mnist" else "cifar") / f"{args.dataset}_riesz_p1"
    if not args.resume:
        args.output = new_run_path(args.output)
    if args.base_channels is None:
        args.base_channels = 32 if args.dataset == "mnist" else 128
    if args.digit is not None:
        if args.dataset != "mnist":
            parser.error("--digit is MNIST-only; use --class-id for CIFAR-10")
        if args.class_id is not None:
            parser.error("use only one of --digit and --class-id")
        args.class_id = args.digit
    if args.latent_dim is None:
        args.latent_dim = image_dim if args.architecture in ('unet', 'dit', 'pixeldit') else 32
    if args.architecture in ('unet', 'dit', 'pixeldit') and args.latent_dim != image_dim:
        parser.error(f"the {args.dataset} {args.architecture} requires --latent-dim {image_dim}")
    if args.architecture == 'dit':
        if args.dataset != 'cifar10':
            parser.error('DiT currently supports CIFAR-10 only')
        if (min(args.dit_width, args.dit_depth, args.dit_heads, args.dit_patch_size) <= 0
                or args.dit_width % args.dit_heads or args.dit_width % 4 or 32 % args.dit_patch_size):
            parser.error('DiT needs positive sizes, width divisible by heads and 4, and patch size dividing 32')
    if args.architecture == 'pixeldit':
        from pixeldit_backbone import validate_pixeldit
        if args.dataset != 'cifar10':
            parser.error('PixelDiT currently supports CIFAR-10 only')
        try:
            validate_pixeldit(args.pixeldit_width, args.pixeldit_patch_depth, args.pixeldit_pixel_depth,
                             args.pixeldit_heads, args.pixeldit_patch_size, args.pixeldit_pixel_width)
        except ValueError as error:
            parser.error(str(error))
    if args.training_stage == 'reconstruction' and args.latent_dim != image_dim:
        parser.error('Reconstruction requires image-dimensional inputs')
    if args.training_stage == 'reconstruction' and (args.multiscale_loss or args.noise_levels != [0.0]):
        parser.error('Distributional view/noise options do not apply to reconstruction')
    if args.steps is None and args.epochs is None:
        args.steps = 2000
    for key in ("particles", "eval_size", "chunk_size", "save_every", "threads", "trajectory_particles", "latent_dim", "base_channels", "num_res_blocks"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if ((args.steps is not None and args.steps < 0) or
            (args.epochs is not None and args.epochs < 0) or
            args.target_size < 0 or args.target_batch_size < 0):
        parser.error("steps and target sizes must be nonnegative")
    if args.save_snapshots is not None and args.save_snapshots <= 0:
        parser.error("--save-snapshots must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if any(width <= 0 for width in args.hidden_dims):
        parser.error("--hidden-dims must be positive")
    if any(not math.isfinite(tau) or tau <= 0 for tau in args.tau_list):
        parser.error("--tau-list must contain finite positive temperatures")
    view_weights = [args.global_loss_weight, args.coarse_loss_weight, args.patch_loss_weight]
    if any(not math.isfinite(w) or w < 0 for w in view_weights) or sum(view_weights) <= 0:
        parser.error("loss weights must be finite, nonnegative, and have positive sum")
    side = 28 if args.dataset == 'mnist' else 32
    if args.crops_per_size < 1 or any(s < 1 or s > side for s in args.coarse_sizes + args.patch_sizes):
        parser.error("view sizes must fit the image and crops-per-size must be positive")
    return args



def main(default_dataset: str = "cifar10") -> None:
    args = parse_args(default_dataset)
    restore_path = args.resume or args.fork_checkpoint
    if args.download_only:
        raw = download_mnist(args.data_root) if args.dataset == "mnist" else download_cifar10(args.data_root)
        print(f"{args.dataset} downloaded and checksums verified: {raw}")
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    if args.data_parallel and device != 'cuda':
        raise RuntimeError('--data-parallel requires CUDA')
    rng = torch.Generator().manual_seed(args.seed)
    raw, (train, train_labels), (test, test_labels) = load_dataset(args.dataset, args.data_root)
    image_shape = dataset_image_shape(args.dataset)
    image_dim = math.prod(image_shape)
    train_ids = torch.arange(len(train))
    test_ids = torch.arange(len(test))
    if args.class_id is not None:
        train_ids = train_ids[train_labels == args.class_id]
        test_ids = test_ids[test_labels == args.class_id]
    train_ids = train_ids[torch.randperm(len(train_ids), generator=rng)]
    if args.target_size:
        train_ids = train_ids[:args.target_size]
    test_ids = test_ids[torch.randperm(len(test_ids), generator=rng)[:args.eval_size]]
    target = train[train_ids].to(device=device, dtype=torch.float32) / 255
    heldout = test[test_ids].to(device=device, dtype=torch.float32) / 255
    # Evaluation identities stay fixed; training uses independent fresh latents.
    eval_latent = torch.randn((args.particles, args.latent_dim),
                             generator=torch.Generator().manual_seed(args.seed + 4)).to(device)
    train_latent_rng = torch.Generator().manual_seed(args.seed + 1)
    sinkhorn_rng = torch.Generator().manual_seed(args.seed + 5)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 3)
    target_options = None
    if args.field_step != 1 or args.noise_levels != [0.0]:
        target_options = dict(field_step=args.field_step,
                              noise_levels=args.noise_levels, noise_rng=noise_rng)
    multiscale_options = None
    if args.multiscale_loss:
        multiscale_options = dict(
            image_shape=image_shape, coarse_sizes=args.coarse_sizes, patch_sizes=args.patch_sizes,
            crops_per_size=args.crops_per_size, patch_rng=torch.Generator().manual_seed(args.seed + 2),
            global_weight=args.global_loss_weight, coarse_weight=args.coarse_loss_weight,
            patch_weight=args.patch_loss_weight)
    dit_options = {key: getattr(args, key) for key in ('dit_width', 'dit_depth', 'dit_heads', 'dit_patch_size')}
    dit_options.update({key: value for key, value in vars(args).items() if key.startswith('pixeldit_')})
    model = build_backbone(args.dataset, args.architecture, args.latent_dim, args.base_channels,
                           args.hidden_dims, args.num_res_blocks, **dit_options).to(device)
    if args.data_parallel:
        from pixeldit_backbone import GeneratorDataParallel
        model = GeneratorDataParallel(model)
        print(f'Generator forwards on {torch.cuda.device_count()} GPUs; full particle loss on cuda:0', flush=True)
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.learning_rate, betas=(0.9, args.adam_beta2),
        weight_decay=args.weight_decay)
    ema_model = copy.deepcopy(model).requires_grad_(False).eval() if args.ema_decay else None
    initialization = None
    if args.init_checkpoint and restore_path is None:
        initialization = initialize_from_checkpoint(model, ema_model, args.init_checkpoint, args)
    evaluation_model = ema_model if ema_model is not None else model
    perceptual_loss = None
    if args.lpips_weight:
        from perceptual import VGGPerceptualLoss
        perceptual_loss = VGGPerceptualLoss(args.lpips_cache, device)
    x = evaluation_images(evaluation_model, eval_latent)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists() and args.resume is None:
        raise FileExistsError(f"Existing run in {args.output}; choose a fresh --output")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(evaluation_latent_seed=args.seed + 4,
                  target_update="drifting_style_stop_gradient_fixed_point_regression",
                  noise_sampling_seed=args.seed + 3,
                  noise_geometry="independent_gaussian_per_view_no_clipping",
                  target_regression_scaling="anchor_batch_scale")
    unit = "epoch" if args.epochs is not None else "step"
    total_iterations = args.epochs if unit == "epoch" else args.steps
    batch_size = args.target_batch_size or len(target)
    updates_per_iteration = math.ceil(len(target) / batch_size) if unit == "epoch" else 1
    config.update(kernel="-euclidean_distance", power=1, device=device,
                  actual_target_size=len(target), actual_eval_size=len(heldout),
                  torch_version=str(torch.__version__),
                  initialization="ddpm_fan_average_uniform_near_zero_output_branches"
                  if args.dataset == "cifar10" and args.architecture == "unet" else
                  'xavier_linear_zero_residual_gates_pixel_head_std_1e-3' if args.architecture == 'dit' else
                  'upstream_zero_output_and_adaln_modulation' if args.architecture == 'pixeldit' else 'pytorch_default',
                  parameterization="neural_generator", latent_distribution="standard_normal",
                  training_latent_sampling="resample_every_update",
                  training_latent_seed=args.seed + 1,
                  evaluation_latent_sampling="fixed_independent_standard_normal",
                  architecture=backbone_metadata(args.dataset, args.architecture, args.latent_dim,
                                                 args.base_channels, args.hidden_dims, args.num_res_blocks, **dit_options),
                  hidden_activation='GELU' if args.architecture == 'dit' else
                  'patch_SwiGLU_pixel_GELU' if args.architecture == 'pixeldit' else 'SiLU', output_activation="Sigmoid",
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  training_rule="detached_scaled_velocity_regression" if args.objective in ("softmax-riesz", "sinkhorn", "drifting")
                  else "exact_half_mmd_gradient",
                  evaluation_metric="fid" if args.dataset == "cifar10" else "biased_empirical_riesz_mmd_squared",
                  energy={"drifting": "drifting_velocity_regression_surrogate", "softmax-riesz": "softmax_field_surrogate", "sinkhorn": "sinkhorn_velocity_regression_surrogate",
                          "riesz": "0.5 * biased_empirical_mmd_squared"}[args.objective], optimizer=args.optimizer,
                  adam_betas=[0.9, args.adam_beta2], adam_eps=1e-8, weight_decay=args.weight_decay,
                  lr_schedule="linear_warmup_then_constant", warmup_start_lr=1e-6,
                  dataset_md5=FILES if args.dataset == "mnist" else {raw.name: CIFAR10_ARCHIVES[raw.name]},
                  data_source=str(raw.resolve()),
                  image_shape=image_shape, pixel_scaling="uint8_div_255", iteration_unit=unit,
                  updates_per_iteration=updates_per_iteration,
                  total_updates=total_iterations * updates_per_iteration,
                  evaluation_weights="ema" if ema_model is not None else "raw",
                  ema_update_unit="optimizer_step" if ema_model is not None else None,
                  ema_initialization="copy_initial_model" if ema_model is not None else None,
                  snapshot_iterations=sorted(snapshot_steps(total_iterations, args.save_snapshots))
                  if args.save_snapshots else sorted(set(range(0, total_iterations + 1, args.save_every)) | {total_iterations}))
    config.update(lr_schedule=f'linear_warmup_then_{args.lr_schedule}',
                  training_generated_batch=args.train_particles or args.particles,
                  training_random_horizontal_flip=args.random_flip)
    if args.dataset == "cifar10":
        config["class_names"] = CIFAR10_CLASSES
    if initialization is not None:
        config['priming_initialization'] = initialization
    if args.training_stage == 'reconstruction':
        config.update(training_rule='paired_denoising_mse', energy='mean_pixel_squared_error',
                      priming_paper='https://arxiv.org/abs/2610.20331',
                      reconstruction_input='(1-s)*(2*x_01-1) + s*N(0,I)',
                      reconstruction_target='x_01; sigmoid output; mean MSE in [0,1]',
                      reconstruction_adaptation='plain MSE; no perceptual loss; no class conditioning',
                      reconstruction_eval_seed=args.seed + 7,
                      training_latent_sampling='one_independent_noise_image_per_real_target')
        if perceptual_loss is not None:
            config.update(training_rule='paired_denoising_mse_plus_lpips',
                          energy='mean_pixel_squared_error_plus_weighted_lpips',
                          reconstruction_adaptation='MSE plus frozen VGG LPIPS; no class conditioning',
                          reconstruction_perceptual={**perceptual_loss.metadata, 'weight': args.lpips_weight})
    if args.multiscale_loss:
        config.update(training_view_geometry="full_image_and_average_pooled_images_and_aligned_crops",
                      patch_sampling_seed=args.seed + 2,
                      patch_alignment="shared_coordinates_across_generated_and_real_populations",
                      view_weighting="mean_within_group_then_normalized_group_weights",
                      view_distance_scaling="upstream_batch_distance_scale" if args.objective in ('softmax-riesz', 'sinkhorn', 'drifting')
                      else "euclidean_distance_divided_by_sqrt_view_dimension",
                      training_loss_diagnostic="mean_over_updates_in_iteration_gradient_surrogates_not_mmd_energy")
    if args.objective == "softmax-riesz":
        source_path = Path(__file__).with_name("loss.py")
        config.update(softmax_source_repository="https://github.com/hudsonchen/riesz_flow_hudson",
                      softmax_source_revision="777d696f31340a1370f5b39a2e7d000464b10526",
                      softmax_module_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      softmax_epsilon=1e-6, softmax_direction_epsilon=1e-8,
                      softmax_rms_epsilon=1e-8, softmax_feature_space="raw_pixels",
                      softmax_numerics='direct_pairwise_differences', softmax_query_chunk=32,
                      softmax_fixed_negatives=False,
                      softmax_field_normalization=('none' if args.no_field_rms_normalization else
                                                   'per_temperature' if args.normalize_per_temperature else 'combined'),
                      velocity_diagnostic=('raw_field_in_scaled_pixel_space' if args.no_field_rms_normalization
                                           else 'normalized_field_in_scaled_pixel_space'))
        if args.no_field_rms_normalization:
            config.update(training_diagnostics_file='training_diagnostics.csv',
                          training_diagnostics_aggregation='arithmetic mean across active views, then updates in each logged interval',
                          training_diagnostics_interval=100,
                          clipping_diagnostic='fraction of updates whose pre-clip parameter gradient L2 norm exceeds max_grad_norm')
    elif args.objective == 'drifting':
        config.update(kernel='joint_row_column_softmax_geometric_mean',
                      drifting_reference_code='https://github.com/lambertae/drifting/blob/main/drift_loss.py',
                      drifting_module_sha256=hashlib.sha256(Path(__file__).with_name('loss.py').read_bytes()).hexdigest(),
                      drifting_source_revision='8861066430db11d2fb0858d042c7845b99df49f1',
                      drifting_source_sha256='445ba0485e1d81be81aee4e1d5458d85548db171ecd310c1fc855ea6242bc782',
                      drifting_feature_space='raw_pixel_views_no_encoder',
                      drifting_affinity_floor=1e-6, drifting_diagonal_distance_mask=100.0,
                      drifting_normalization='per_temperature_rms_floor_squared_1e-8',
                      velocity_diagnostic='drifting_normalized_field_in_scaled_pixel_space')
    elif args.objective == 'sinkhorn':
        config.update(kernel="entropic_quadratic_transport", power=2,
                      sinkhorn_paper="https://arxiv.org/abs/2605.11755v2",
                      sinkhorn_reference_code="https://github.com/hanjq17/W-Flow/blob/main/drift_loss_ot.py",
                      sinkhorn_reference_code_sha256="c987285caf69cb1085e3e50450936368373087d6066bb6fe15020bae8796b58e",
                      sinkhorn_module_sha256=hashlib.sha256(Path(__file__).with_name('loss.py').read_bytes()).hexdigest(),
                      sinkhorn_self_transport="independent_generated_batch_no_diagonal_mask",
                      sinkhorn_reference_seed=args.seed + 5, sinkhorn_eval_reference_seed=args.seed + 6,
                      sinkhorn_cost="0.5 * squared_euclidean_in_scaled_pixel_views",
                      sinkhorn_regularization="epsilon_times_view_dimension",
                      sinkhorn_scale="rms_pair_distance_over_sqrt_view_dimension_floor_1e-3",
                      sinkhorn_velocity_normalization="per_view_rms_floor_squared_1e-8",
                      sinkhorn_source_space="raw_pixel_views_no_encoder",
                      sinkhorn_marginal_diagnostic="max_absolute_plan_mass_error_per_view_then_mean_over_updates",
                      velocity_diagnostic="sinkhorn_raw_velocity_rms_in_scaled_pixel_space")
        if args.sinkhorn_mode == 'ot':
            config.update(sinkhorn_paper='https://arxiv.org/abs/2610.20331',
                          sinkhorn_reference_code=None, sinkhorn_reference_code_sha256=None,
                          sinkhorn_epsilon_mixture=({'0.001': .75, '0.01': .25}
                              if args.sinkhorn_ot_epsilon is None else {f'{args.sinkhorn_ot_epsilon:g}': 1.}),
                          sinkhorn_cost='0.5 * mean_squared_distance_in_batch_scaled_pixel_view',
                          sinkhorn_regularization='epsilon_in_dimension_averaged_cost_units',
                          sinkhorn_velocity_normalization='none',
                          training_rule='direct_detached_plan_ot_attraction_minus_repulsion',
                          target_update='none_direct_ot', target_regression_scaling='not_applicable',
                          energy='detached_plan_ot_gradient_surrogate',
                          velocity_diagnostic='negative_ot_gradient_times_query_count_in_scaled_dimension_normalized_coordinates')
    else:
        config["velocity_diagnostic"] = "ordinary_riesz_field_in_pixel_space"
    fid_evaluator = None
    if args.dataset == "cifar10":
        from fid import CIFAR10FID, WEIGHTS
        real = train if args.class_id is None else train[train_labels == args.class_id]
        fid_evaluator = CIFAR10FID(real, device=device, batch_size=args.fid_batch_size)
        config.update(fid_reference_split="train", fid_reference_count=len(real),
                      fid_feature_dim=2048, fid_latent_seed=args.seed + 12345,
                      fid_evaluation_iterations=config["snapshot_iterations"],
                      fid_inception="pytorch-fid-v0.3.0",
                      fid_weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                      fid_preprocessing="float [0,1]; bilinear 299x299 align_corners=False; scale to [-1,1]",
                      diagnostic_metric="biased_empirical_riesz_mmd_squared")
    if restore_path is None:
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "data_indices.json").write_text(json.dumps(
        {"train_indices": train_ids.tolist(), "test_indices": test_ids.tolist()}) + "\n")
    save_grid(target, args.output / "target.png", image_shape=image_shape)
    save_grid(heldout, args.output / "heldout.png", image_shape=image_shape)
    eval_train = target[:args.eval_size]
    train_yy = mean_distance(eval_train, eval_train, args.chunk_size)
    test_yy = mean_distance(heldout, heldout, args.chunk_size)
    save_iterations = set(config["snapshot_iterations"])
    trajectory_states, trajectory_iterations, history = [], [], []
    update_count = 0
    fid_history = []
    training_stats, training_updates = {}, 0
    loss_names = ('global', 'coarse', 'patch', 'combined') if args.multiscale_loss else ()
    if args.training_stage == 'reconstruction':
        loss_names = ('reconstruction_mse',)
        if perceptual_loss is not None:
            loss_names += ('reconstruction_lpips', 'reconstruction_total')
    elif args.objective == 'sinkhorn':
        loss_names += ('sinkhorn_velocity_raw_rms', 'sinkhorn_marginal_max_error')
        if args.sinkhorn_mode == 'ot':
            loss_names += ('sinkhorn_ot_attraction', 'sinkhorn_ot_repulsion', 'sinkhorn_solver_iterations')
    start_iteration, elapsed_offset = 0, 0.0
    if restore_path is not None:
        if args.resume and not (args.output / 'config.json').exists():
            raise ValueError('Resume requires the original output directory and config.json')
        checkpoint = torch.load(restore_path, map_location='cpu', weights_only=False)
        if perceptual_loss is not None:
            previous_perceptual = checkpoint['config'].get('reconstruction_perceptual', {})
            if previous_perceptual.get('weights_sha256') != perceptual_loss.metadata['weights_sha256']:
                raise ValueError('Resume perceptual weights mismatch')
        if 'priming_initialization' in checkpoint['config']:
            config['priming_initialization'] = checkpoint['config']['priming_initialization']
        ignored = {'resume', 'fork_checkpoint', 'threads', 'device', 'checkpoint_every', 'keep_checkpoint_every',
                   'save_every'}  # Evaluation cadence does not change training or optimizer state.
        tuning_keys = {'output', 'learning_rate', 'lr_schedule', 'min_learning_rate',
                       'schedule_start_update', 'warmup_steps', 'train_particles', 'random_flip',
                       'max_grad_norm', 'ema_decay', 'crops_per_size', 'global_loss_weight',
                       'coarse_loss_weight', 'patch_loss_weight'}
        if args.fork_checkpoint:
            ignored |= tuning_keys
            if checkpoint['arguments'].get('training_stage', 'distributional') != 'distributional':
                raise ValueError('Tuned continuation requires a distributional checkpoint')
            if args.schedule_start_update != checkpoint['update_count']:
                raise ValueError('Fork schedule must start at the source checkpoint update')
            config['tuning_initialization'] = dict(
                checkpoint=str(restore_path.resolve()), source_update=checkpoint['update_count'],
                inherited_optimizer=True, inherited_ema=True,
                changes={k: {'before': checkpoint['arguments'].get(k),
                             'after': str(getattr(args, k)) if isinstance(getattr(args, k), Path)
                             else getattr(args, k)} for k in sorted(tuning_keys)
                         if checkpoint['arguments'].get(k) != (str(getattr(args, k))
                             if isinstance(getattr(args, k), Path) else getattr(args, k))})
        elif 'tuning_initialization' in checkpoint['config']:
            config['tuning_initialization'] = checkpoint['config']['tuning_initialization']
        legacy_defaults = dict(optimizer='Adam', adam_beta2=0.999, weight_decay=0.0,
                               warmup_steps=0, max_grad_norm=0.0, training_stage='distributional',
                               corruption_level=0.8, init_checkpoint=None, train_particles=None,
                               random_flip=False, lr_schedule='constant', min_learning_rate=0.0,
                               schedule_start_update=0, no_field_rms_normalization=False,
                               sinkhorn_mode='wflow', sinkhorn_tolerance=1e-6,
                               dit_width=384, dit_depth=12, dit_heads=6, dit_patch_size=2,
                               lpips_weight=0.0, lpips_cache=None, data_parallel=False,
                               pixeldit_width=768, pixeldit_patch_depth=12, pixeldit_pixel_depth=2,
                               pixeldit_heads=12, pixeldit_patch_size=4, pixeldit_pixel_width=16)
        for key, value in vars(args).items():
            value = str(value) if isinstance(value, Path) else value
            if key == ('epochs' if unit == 'epoch' else 'steps'):
                previous_duration = checkpoint['arguments'][key]
                if previous_duration is None:
                    raise ValueError('Resume cannot change between steps and epochs')
                if value < previous_duration:
                    raise ValueError(f'Resume cannot shorten {key}')
                continue
            if key not in ignored and checkpoint['arguments'].get(key, legacy_defaults.get(key)) != value:
                raise ValueError(f'Resume configuration mismatch for {key}')
        model.load_state_dict(checkpoint['model'])
        if ema_model is not None:
            if checkpoint['ema_model'] is None:
                raise ValueError('Continuation requires the source EMA state')
            ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        rng.set_state(checkpoint['data_rng'])
        train_latent_rng.set_state(checkpoint['latent_rng'])
        if 'sinkhorn_rng' in checkpoint:
            sinkhorn_rng.set_state(checkpoint['sinkhorn_rng'])
        noise_rng.set_state(checkpoint['noise_rng'])
        if multiscale_options is not None:
            multiscale_options['patch_rng'].set_state(checkpoint['patch_rng'])
        start_iteration = checkpoint['iteration'] + 1
        update_count = checkpoint['update_count']
        history, fid_history = checkpoint['history'], checkpoint['fid_history']
        trajectory_states = checkpoint['trajectory_states']
        trajectory_iterations = checkpoint['trajectory_iterations']
        elapsed_offset = checkpoint['elapsed_seconds']
        # Remove rows written after the last durable checkpoint before appending.
        metrics_source = (Path(checkpoint['arguments']['output']) if args.fork_checkpoint else args.output)
        with (metrics_source / 'metrics.csv').open(newline='') as previous:
            rows = list(csv.reader(previous))
        with (args.output / 'metrics.csv').open('w', newline='') as previous:
            csv.writer(previous).writerows(
                [rows[0]] + [r for r in rows[1:] if int(r[0]) < start_iteration])
        if fid_evaluator is not None:
            (args.output / 'fid_metrics.json').write_text(json.dumps(
                [{'iteration': i, 'fid': f} for i, f in fid_history], indent=2) + '\n')
        # The loop below starts with evaluation; perform the pending epoch first.
        resume_training = start_iteration <= total_iterations
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        if args.fork_checkpoint:
            fork_state = dict(checkpoint, config=config, arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
            temporary = args.output / 'checkpoint_latest.pt.tmp'
            torch.save(fork_state, temporary)
            temporary.replace(args.output / 'checkpoint_latest.pt')
            if args.keep_checkpoint_every:
                os.link(args.output / 'checkpoint_latest.pt',
                        args.output / f'checkpoint_{unit}_{checkpoint["iteration"]:06d}.pt')
            del fork_state
    else:
        resume_training = False

    diagnostic_totals, diagnostic_count = {}, 0
    diagnostic_path = args.output / 'training_diagnostics.csv'
    record_training_diagnostics = args.no_field_rms_normalization or args.objective == 'sinkhorn'
    if record_training_diagnostics and diagnostic_path.exists():
        with diagnostic_path.open(newline='') as previous:
            diagnostic_rows = list(csv.reader(previous))
        with diagnostic_path.open('w', newline='') as previous:
            csv.writer(previous).writerows([diagnostic_rows[0]] +
                [row for row in diagnostic_rows[1:] if int(row[0]) <= update_count])

    def train_iteration():
        nonlocal update_count, training_stats, training_updates
        nonlocal diagnostic_totals, diagnostic_count
        batches = (epoch_batches(len(target), batch_size, rng) if unit == 'epoch'
                   else [torch.randperm(len(target), generator=rng)[:batch_size]])
        training_stats, training_updates = {}, 0
        for ids in batches:
            learning_rate = training_learning_rate(
                update_count, args.learning_rate, args.warmup_steps, schedule=args.lr_schedule,
                minimum=args.min_learning_rate, start=args.schedule_start_update,
                total=config['total_updates'])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            batch = target[ids.to(device)]
            if args.random_flip:
                batch = horizontal_flip_batch(batch, image_shape)
            if args.training_stage == 'reconstruction':
                noise = torch.randn(batch.shape, generator=train_latent_rng).to(device)
                step_diagnostics = reconstruction_step(
                    model, batch, noise, optimizer, args.corruption_level, args.max_grad_norm,
                    perceptual_loss=perceptual_loss, perceptual_weight=args.lpips_weight)
                for name, value in step_diagnostics.items():
                    training_stats[name] = training_stats.get(name, 0) + value
                training_updates += 1
                if ema_model is not None:
                    update_ema(model, ema_model, args.ema_decay)
                update_count += 1
                continue
            train_latent = torch.randn((args.train_particles or args.particles, args.latent_dim),
                                       generator=train_latent_rng).to(device)
            sinkhorn_latent = (torch.randn((args.train_particles or args.particles, args.latent_dim), generator=sinkhorn_rng).to(device)
                               if args.objective == 'sinkhorn' else None)
            step_diagnostics = generator_step(
                model, train_latent, batch, optimizer, args.chunk_size,
                objective=args.objective, tau_list=tuple(args.tau_list),
                normalize_per_temperature=args.normalize_per_temperature,
                multiscale_options=multiscale_options, target_options=target_options,
                sinkhorn_latent=sinkhorn_latent, sinkhorn_epsilon=args.sinkhorn_epsilon,
                sinkhorn_iters=args.sinkhorn_iters, max_grad_norm=args.max_grad_norm,
                no_field_rms_normalization=args.no_field_rms_normalization,
                sinkhorn_mode=args.sinkhorn_mode, sinkhorn_tolerance=args.sinkhorn_tolerance,
                sinkhorn_ot_epsilon=args.sinkhorn_ot_epsilon)
            for name, value in step_diagnostics.items():
                training_stats[name] = training_stats.get(name, 0) + float(value)
            training_updates += 1
            if ema_model is not None:
                update_ema(model, ema_model, args.ema_decay)
            update_count += 1
            if record_training_diagnostics:
                for name, value in step_diagnostics.items():
                    if name.startswith(('raw_', 'sinkhorn_')) or name in ('grad_norm_before_clip', 'clip_fraction', 'parameter_update_relative_norm'):
                        diagnostic_totals[name] = diagnostic_totals.get(name, 0.) + float(value)
                diagnostic_count += 1
                if (update_count % 100 == 0 or update_count == config['total_updates']
                        or (args.checkpoint_every and
                            (update_count % args.checkpoint_every == 0 if unit == 'step'
                             else update_count % updates_per_iteration == 0))):
                    row = dict(update=update_count, interval_updates=diagnostic_count,
                               learning_rate=learning_rate,
                               **{k: v / diagnostic_count for k, v in diagnostic_totals.items()})
                    with diagnostic_path.open('a', newline='') as diagnostic_stream:
                        diagnostic_writer = csv.DictWriter(diagnostic_stream, fieldnames=list(row))
                        if diagnostic_stream.tell() == 0:
                            diagnostic_writer.writeheader()
                        diagnostic_writer.writerow(row)
                    diagnostic_totals, diagnostic_count = {}, 0

    started = time.monotonic()
    if resume_training:
        train_iteration()
    print(f"Stage={args.training_stage}, neural {args.objective}: {config['architecture']}, {config['parameter_count']} parameters, "
          f"{len(x)} generated particles, {len(target)} targets, {device}", flush=True)
    with (args.output / "metrics.csv").open("a" if restore_path else "w", newline="") as stream:
        writer = csv.writer(stream)
        if restore_path is None:
            writer.writerow(["iteration", "epoch", "update", "train_subset_mmd_squared", "heldout_mmd_squared",
                             "velocity_eval_subset_rms", "pixel_min", "pixel_max", "elapsed_seconds"]
                            + (["fid"] if fid_evaluator is not None else [])
                            + (["heldout_reconstruction_mse"] if args.training_stage == 'reconstruction' else [])
                            + [name if name.startswith('sinkhorn_') else
                               (f'train_{name}' if name.startswith('reconstruction_') else f'train_{name}_surrogate')
                               for name in loss_names])
        for iteration in range(start_iteration, total_iterations + 1):
            if iteration > 0:
                x = evaluation_images(evaluation_model, eval_latent)
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"Nonfinite generated images after update {update_count}")
            # Evaluate every full epoch, even when particle grids are saved less often.
            if unit == "epoch" or iteration in save_iterations:
                fid_value = ""
                if fid_evaluator is not None and iteration in save_iterations:
                    fid_value = fid_evaluator.evaluate_generator(
                        evaluation_model, args.latent_dim, count=args.fid_samples, seed=args.seed + 12345)
                    fid_history.append((iteration, fid_value))
                    (args.output / "fid_metrics.json").write_text(json.dumps(
                        [{"iteration": i, "fid": f} for i, f in fid_history], indent=2) + "\n")
                    print(f"{unit}={iteration} FID={fid_value:.6f} ({args.fid_samples} generated images)", flush=True)
                train_mmd = mmd_squared(x, eval_train, train_yy, args.chunk_size)
                test_mmd = mmd_squared(x, heldout, test_yy, args.chunk_size)
                # Report the velocity on the fixed evaluation target, separately from MMD.
                if args.objective == "softmax-riesz":
                    with torch.no_grad():
                        _, info = _riesz_topk_softmax_loss_impl(
                            x.unsqueeze(0), eval_train.unsqueeze(0), tau_list=args.tau_list,
                            normalize_per_temperature=args.normalize_per_temperature, epsilon=1e-6,
                            no_field_rms_normalization=args.no_field_rms_normalization)
                    speed = info["riesz_topk_softmax_velocity_rms"].item()
                elif args.objective == 'drifting':
                    with torch.no_grad():
                        _, info = drifting_loss(x.unsqueeze(0), eval_train.unsqueeze(0), tau_list=args.tau_list)
                    speed = info['drifting_velocity_rms'].item()
                elif args.objective == 'sinkhorn':
                    ref_latent = torch.randn((args.particles, args.latent_dim),
                                             generator=torch.Generator().manual_seed(args.seed + 6)).to(device)
                    reference = evaluation_images(evaluation_model, ref_latent)
                    if args.sinkhorn_mode == 'ot':
                        _, info = sinkhorn_ot_loss(x, eval_train, reference,
                            iterations=args.sinkhorn_iters, tolerance=args.sinkhorn_tolerance,
                            epsilon=args.sinkhorn_ot_epsilon)
                    else:
                        _, info = sinkhorn_target_loss(
                            x, eval_train, reference, epsilon=args.sinkhorn_epsilon, iterations=args.sinkhorn_iters)
                    speed = info['sinkhorn_velocity_raw_rms'].item()
                else:
                    speed = velocity(x, eval_train, args.chunk_size).square().mean().sqrt().item()
                writer.writerow([iteration, iteration if unit == "epoch" else "", update_count,
                                 train_mmd, test_mmd, speed,
                                 x.min().item(), x.max().item(), elapsed_offset + time.monotonic() - started]
                                + ([fid_value] if fid_evaluator is not None else [])
                                + ([reconstruction_error(evaluation_model, heldout, args.corruption_level,
                                                         args.seed + 7, batch_size)
                                    if iteration in save_iterations else '']
                                   if args.training_stage == 'reconstruction' else [])
                                + [float(training_stats[name]) / training_updates
                                   if name in training_stats else '' for name in loss_names])
                stream.flush()
                history.append((iteration, train_mmd, test_mmd))
                print(f"{unit}={iteration:6d} updates={update_count} train MMD²={train_mmd:.6f} heldout MMD²={test_mmd:.6f} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
            if iteration in save_iterations:
                save_grid(x, args.output / f"particles_{unit}_{iteration:06d}.png", image_shape=image_shape)
                trajectory_states.append(x.detach().cpu().reshape(-1, *image_shape).clone())
                trajectory_iterations.append(iteration)
                save_trajectory(trajectory_states, trajectory_iterations, args.output,
                                args.trajectory_particles, unit)
                plot_mmd(history, args.output, unit, fid_history=fid_history)
            retain_checkpoint = args.keep_checkpoint_every and (
                iteration % args.keep_checkpoint_every == 0 or iteration == total_iterations)
            if retain_checkpoint or args.checkpoint_every and (iteration % args.checkpoint_every == 0 or iteration == total_iterations):
                checkpoint = dict(
                    config=config,
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    iteration=iteration, update_count=update_count,
                    model=model.state_dict(), ema_model=ema_model.state_dict() if ema_model else None,
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                    data_rng=rng.get_state(), latent_rng=train_latent_rng.get_state(),
                    sinkhorn_rng=sinkhorn_rng.get_state(),
                    noise_rng=noise_rng.get_state(),
                    patch_rng=multiscale_options['patch_rng'].get_state() if multiscale_options else None,
                    history=history, fid_history=fid_history, trajectory_states=trajectory_states,
                    trajectory_iterations=trajectory_iterations,
                    elapsed_seconds=elapsed_offset + time.monotonic() - started)
                temporary = args.output / 'checkpoint_latest.pt.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(args.output / 'checkpoint_latest.pt')
                if retain_checkpoint:
                    retained = args.output / f'checkpoint_{unit}_{iteration:06d}.pt'
                    if not retained.exists():
                        # Latest checkpoints are atomically replaced, so this inode stays immutable.
                        os.link(args.output / 'checkpoint_latest.pt', retained)
            if iteration == total_iterations:
                break
            train_iteration()
    print(f"Finished. Results: {args.output.resolve()}")



if __name__ == "__main__":
    main()
