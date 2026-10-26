#!/usr/bin/env python3
"""Evaluate a saved DDPM EMA with the MMD runs' fixed Riesz MMD² protocol.

Reads checkpoints only. Saves scalar metrics separately from DDPM training.
Use 64 generated images, the exact recorded 512-image train/test subsets,
unquantized [0,1] pixels, and the same biased -Euclidean-distance MMD².
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('DRIFT_COMPILE', '0')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import CIFAR10UNetParticleGenerator
from data import download_cifar10, load_cifar10_split
from ddpm import Diffusion
from loss import mean_distance, mmd_squared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    config = json.loads((args.reference_run / 'config.json').read_text())
    indices_path = args.reference_run / 'data_indices.json'
    indices = json.loads(indices_path.read_text())
    assert config['dataset'] == 'cifar10' and config['objective'] == 'softmax-riesz'
    assert config['diagnostic_metric'] == 'biased_empirical_riesz_mmd_squared'
    # mmap pins this atomically replaced checkpoint version for the entire read.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    step = checkpoint['update_count']
    options = checkpoint['arguments']
    model = CIFAR10UNetParticleGenerator(options['base_channels'], options['num_res_blocks'],
                                        time_conditioned=True).to(device).eval()
    model.load_state_dict(checkpoint['ema_model'])
    del checkpoint
    count = config['particles']
    seed = config['evaluation_latent_seed']
    print(f'Evaluating DDPM EMA update={step}; generated={count}; diffusion steps=1000', flush=True)
    images = Diffusion(device).sample(model, count,
                generator=torch.Generator(device=device).manual_seed(seed)).flatten(1)
    assert torch.isfinite(images).all()
    archive = download_cifar10(args.data_root)
    train, _ = load_cifar10_split(archive, True)
    test, _ = load_cifar10_split(archive, False)
    train = train[indices['train_indices'][:config['eval_size']]].to(device).float() / 255
    test = test[indices['test_indices']].to(device).float() / 255
    chunk = config['chunk_size']
    row = dict(iteration=step, update=step,
               train_subset_mmd_squared=mmd_squared(images, train, mean_distance(train, train, chunk), chunk),
               heldout_mmd_squared=mmd_squared(images, test, mean_distance(test, test, chunk), chunk),
               generated_count=count, train_reference_count=len(train), test_reference_count=len(test),
               kernel='-euclidean_distance', estimator='biased_empirical_riesz_mmd_squared',
               pixel_scaling='unquantized_float_[0,1]', evaluation_weights='ema',
               sampling_steps=1000, sampling_seed=seed, sampling_rng='cuda',
               reference_run=str(args.reference_run.resolve()),
               reference_indices_sha256=hashlib.sha256(indices_path.read_bytes()).hexdigest(),
               checkpoint=str(args.checkpoint.resolve()),
               evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    rows = json.loads(args.output.read_text()) if args.output.exists() else []
    rows = [r for r in rows if r['update'] != step] + [row]
    rows.sort(key=lambda r: r['update'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(rows, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
