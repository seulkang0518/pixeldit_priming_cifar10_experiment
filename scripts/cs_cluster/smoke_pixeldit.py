#!/usr/bin/env python3
"""Full-size GPU capacity check; synthetic images, production particle counts."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import build_backbone
from main import generator_step, update_ema
from priming import reconstruction_step
from pixeldit_backbone import GeneratorDataParallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=('dit', 'pixeldit'), default='pixeldit')
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lpips-weight', type=float, default=0.)
    parser.add_argument('--lpips-cache', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires CUDA')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    devices = torch.cuda.device_count() if args.data_parallel else 1
    model = build_backbone('cifar10', args.backbone, 3072, 128, [128, 256], 4).cuda()
    if args.data_parallel:
        model = GeneratorDataParallel(model)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    # Include the dataset-sized allocation and FID model, both resident in main.py.
    data = torch.rand(50000, 3072, device='cuda')
    from fid import load_inception
    inception = load_inception('cuda')
    for device in range(devices):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    pre_optimizer = torch.optim.Adam(model.parameters(), lr=.0002, betas=(.9, .999))
    perceptual = None
    if args.lpips_weight:
        from perceptual import VGGPerceptualLoss
        perceptual = VGGPerceptualLoss(args.lpips_cache, 'cuda')
    pre = []
    for _ in range(3):
        pre.append(reconstruction_step(model, data[:64], torch.randn(64, 3072, device='cuda'), pre_optimizer,
                                       perceptual_loss=perceptual, perceptual_weight=args.lpips_weight))
        update_ema(model, ema, .9999)
    del pre_optimizer, perceptual
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002, betas=(.9, .999))
    post = []
    for _ in range(3):
        info = generator_step(model, torch.randn(128, 3072, device='cuda'), data[:256], optimizer, 64,
            objective='softmax-riesz', normalize_per_temperature=False,
            no_field_rms_normalization=True, max_grad_norm=1.,
            multiscale_options=dict(image_shape=(3, 32, 32), coarse_sizes=(8, 16),
                patch_sizes=(7, 14), crops_per_size=4, global_weight=2., coarse_weight=1.,
                patch_weight=1., patch_rng=torch.Generator().manual_seed(44)))
        update_ema(model, ema, .9999)
        post.append({key: float(value) for key, value in info.items()})
    with torch.no_grad():
        images = ema(torch.randn(64, 3072, device='cuda'))
        features = inception(images.reshape(-1, 3, 32, 32))[0]
    assert torch.isfinite(images).all() and torch.isfinite(features).all()
    assert images.shape == (64, 3072) and (images >= 0).all() and (images <= 1).all()
    for device in range(devices):
        torch.cuda.synchronize(device)
    result = dict(status='passed', backbone=args.backbone, parameters=sum(p.numel() for p in model.parameters()),
        visible_devices=[torch.cuda.get_device_name(index) for index in range(devices)],
        elapsed_seconds=time.monotonic() - started, reconstruction_batch=64,
        post_generated_batch=128, post_real_batch=256, precision='float32; no AMP changes',
        full_population_loss=True, reconstruction=pre, posttraining=post,
        memory=[dict(device=index, allocated_gib=torch.cuda.max_memory_allocated(index) / 2**30,
                     reserved_gib=torch.cuda.max_memory_reserved(index) / 2**30) for index in range(devices)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
