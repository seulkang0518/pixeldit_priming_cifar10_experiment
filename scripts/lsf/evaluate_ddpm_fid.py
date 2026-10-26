#!/usr/bin/env python3
"""Evaluate a preserved DDPM EMA checkpoint without interrupting training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import CIFAR10UNetParticleGenerator
from data import download_cifar10, load_cifar10_split
from ddpm import Diffusion, write_json
from fid import CIFAR10FID, WEIGHTS, feature_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress = dict(status='starting', job_id=os.environ.get('LSB_JOBID'),
                    generated_count=0, target_count=10000,
                    checkpoint=str(args.checkpoint.resolve()))

    def status(state, **updates):
        progress.update(status=state, **updates,
                        elapsed_seconds=time.monotonic() - started,
                        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(args.output / 'job_status.json', progress)

    status('loading_checkpoint')
    try:
        with args.checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        step, options = checkpoint['update_count'], checkpoint['arguments']
        model = CIFAR10UNetParticleGenerator(options['base_channels'], options['num_res_blocks'],
                                             time_conditioned=True).to(device).eval()
        model.load_state_dict(checkpoint['ema_model'])
        del checkpoint
        status('computing_reference', iteration=step, checkpoint_sha256=digest)
        print(f'Evaluating DDPM EMA update={step}: FID-10K, 1000 reverse steps', flush=True)
        archive = download_cifar10(args.data_root)
        real, _ = load_cifar10_split(archive, True)
        evaluator = CIFAR10FID(real, device=device, batch_size=64)
        diffusion = Diffusion(device)
        rng = torch.Generator(device=device).manual_seed(12345)

        def batches():
            for offset in range(0, 10000, 64):
                count = min(64, 10000 - offset)
                images = diffusion.sample(model, count, generator=rng)
                yield images
                status('sampling', generated_count=offset + count)
                print(f'FID sampling {offset + count}/10000 '
                      f'elapsed={time.monotonic() - started:.1f}s', flush=True)

        status('sampling')
        statistics = feature_statistics(batches(), evaluator.inception, device)
        assert statistics[2] == 10000
        score = evaluator.score(statistics)
        row = dict(iteration=step, fid=score, generated_count=statistics[2],
                   reference_count=len(real), reference_split='train', sampling_steps=1000,
                   sampling_seed=12345, sampling_rng='cuda', sample_batch_size=64,
                   evaluation_weights='ema', checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=digest, feature_dim=2048,
                   inception='pytorch-fid-v0.3.0',
                   inception_weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                   evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(args.output / 'fid_metrics.json', [row])
        status('evaluation_complete', fid=score)
        print(f'step={step} FID={score:.6f} (10000 generated images)', flush=True)
    except Exception as error:
        status('failed', error=str(error))
        raise
    try:
        subprocess.run([sys.executable, str(Path(__file__).with_name('finalize_cifar_comparison.py')),
                        '--notebook', str(args.notebook)], check=True)
    except Exception as error:
        status('notebook_refresh_failed', error=str(error))
        raise
    status('complete')


if __name__ == '__main__':
    main()
