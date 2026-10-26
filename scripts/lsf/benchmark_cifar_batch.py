#!/usr/bin/env python3
"""Measure primed CIFAR batch memory and speed without altering training runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def measure(args):
    sys.path.insert(0, str(args.source))
    import copy
    import statistics
    import torch
    from backbone import build_backbone
    from data import load_dataset
    from fid import load_inception
    from main import generator_step, horizontal_flip_batch, update_ema

    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    result = dict(objective=args.objective, generated_batch=args.batch, real_batch=256,
                  gpu=torch.cuda.get_device_name(), job_id=os.environ.get('LSB_JOBID'),
                  total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  checkpoint=str(args.checkpoint), warmup_updates=2, timed_updates=8,
                  learning_rate=2e-4, precision='float32', status='running')
    phase = 'initialization'
    try:
        # Retain the full real dataset and FID network, as the production trainer does.
        _, (train, _), _ = load_dataset('cifar10', args.data_root)
        target = train.to(device='cuda', dtype=torch.float32).div_(255)
        inception = load_inception('cuda')
        model = build_backbone('cifar10', 'unet', 3072, 128, [128, 256], 4).cuda()
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema_model'])
        del checkpoint
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        options = dict(image_shape=(3, 32, 32), coarse_sizes=[8, 16], patch_sizes=[7, 14],
                       crops_per_size=4, patch_rng=torch.Generator().manual_seed(44),
                       global_weight=2., coarse_weight=1., patch_weight=1.)
        objective = 'sinkhorn' if args.objective == 'sinkhorn_ot' else 'softmax-riesz'
        rng = torch.Generator().manual_seed(43)
        times = []
        phase = 'training'
        torch.cuda.reset_peak_memory_stats()
        for step in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            real = horizontal_flip_batch(target[torch.randint(len(target), (256,), generator=rng)], (3, 32, 32))
            latent = torch.randn(args.batch, 3072, generator=rng).cuda()
            reference = torch.randn(args.batch, 3072, generator=rng).cuda() if objective == 'sinkhorn' else None
            diagnostics = generator_step(
                model, latent, real, optimizer, 64, objective=objective,
                no_field_rms_normalization=args.objective == 'softmax_no_rms',
                normalize_per_temperature=args.objective != 'softmax_no_rms',
                sinkhorn_latent=reference, sinkhorn_mode='ot' if objective == 'sinkhorn' else 'wflow',
                sinkhorn_iters=1000, sinkhorn_tolerance=1e-6,
                multiscale_options=options, max_grad_norm=1.)
            update_ema(model, ema, .9999)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            assert all(torch.isfinite(v) for v in diagnostics.values()), 'Nonfinite training diagnostics'
            if step >= 2:
                times.append(elapsed)
            print(json.dumps(dict(phase=phase, update=step + 1, seconds=elapsed)), flush=True)
        assert all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite model parameters'
        result.update(seconds_per_update=statistics.mean(times),
                      median_seconds_per_update=statistics.median(times),
                      generated_images_per_second=args.batch/statistics.mean(times),
                      training_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      training_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      diagnostics={k:float(v) for k,v in diagnostics.items()})
        # Keep Adam state and gradients resident during the normal evaluation batch.
        phase = 'evaluation_batch_64'
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            images = ema(torch.randn(64, 3072, generator=rng).cuda())
            features = inception(images.reshape(-1, 3, 32, 32))[0]
            assert torch.isfinite(features).all()
        torch.cuda.synchronize()
        result.update(status='passed', evaluation_batch=64,
                      evaluation_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      evaluation_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='out_of_memory', phase=phase, error=str(error))
    except Exception as error:
        result.update(status='failed', phase=phase, error=repr(error))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objective', choices=('softmax', 'softmax_no_rms', 'sinkhorn_ot'))
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    if args.objective:
        if not args.batch or args.batch < 1:
            parser.error('--batch must be positive')
        measure(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for objective in ('softmax', 'softmax_no_rms', 'sinkhorn_ot'):
        # Each case has a fresh CUDA process, including after an out-of-memory failure.
        batches = [64, 96] if objective != 'softmax_no_rms' else [64]
        for batch in batches:
            path = args.output / f'{objective}_batch{batch}.json'
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--source', str(args.source), '--data-root', str(args.data_root),
                       '--checkpoint', str(args.checkpoint), '--output', str(path),
                       '--objective', objective, '--batch', str(batch)]
            print('Starting', objective, batch, flush=True)
            with path.with_suffix('.log').open('w') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            record = json.loads(path.read_text()) if path.exists() else dict(
                objective=objective, generated_batch=batch, status='process_failed',
                exit_code=completed.returncode, log=str(path.with_suffix('.log')))
            results.append(record)
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            if batch == 64 and record['status'] == 'out_of_memory':
                batches[:] = [64, 32]
            elif record['status'] not in ('passed', 'out_of_memory'):
                raise RuntimeError(f'Benchmark failed: {record}')


if __name__ == '__main__':
    main()
