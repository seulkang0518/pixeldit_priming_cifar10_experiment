"""Full-size DiT GPU check for priming, corrected Riesz, EMA, and FID features."""
import copy
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import build_backbone
from data import load_dataset
from fid import load_inception
from loss import _softmax_toward
from main import generator_step, update_ema
from priming import reconstruction_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--lpips-weight', type=float, default=0.0)
    parser.add_argument('--lpips-cache', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke check requires CUDA')
    _, (images, _), _ = load_dataset('cifar10', args.data)
    targets = images.cuda().float() / 255
    model = build_backbone('cifar10', 'dit', 3072, 128, [128, 256], 4).cuda()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    inception = load_inception('cuda')
    perceptual_loss = None
    if args.lpips_weight:
        from perceptual import VGGPerceptualLoss
        perceptual_loss = VGGPerceptualLoss(args.lpips_cache, 'cuda')
    timings = {}
    stage_diagnostics = {}
    for stage in ('reconstruction', 'distributional'):
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(3):
            if stage == 'reconstruction':
                info = reconstruction_step(model, targets[:64], torch.randn(64, 3072, device='cuda'), optimizer,
                                           perceptual_loss=perceptual_loss, perceptual_weight=args.lpips_weight)
            else:
                info = generator_step(model, torch.randn(128, 3072, device='cuda'), targets[:256],
                    optimizer, 64, objective='softmax-riesz', normalize_per_temperature=False,
                    no_field_rms_normalization=True, max_grad_norm=1.,
                    multiscale_options=dict(image_shape=(3, 32, 32), coarse_sizes=(8, 16),
                        patch_sizes=(7, 14), crops_per_size=4, global_weight=2.,
                        coarse_weight=1., patch_weight=1., patch_rng=torch.Generator().manual_seed(44)))
            update_ema(model, ema, .9999)
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values())
        torch.cuda.synchronize()
        timings[stage] = (time.monotonic() - started) / 3
        stage_diagnostics[stage] = {name: float(torch.as_tensor(value).detach().cpu())
                                    for name, value in info.items()}
        if perceptual_loss is not None:
            assert all(p.grad is None and not p.requires_grad for p in perceptual_loss.parameters())
        del optimizer
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.blocks[0].qkv.weight.grad.abs().sum() > 0
    with torch.no_grad():
        features = inception(ema(torch.randn(64, 3072, device='cuda')).reshape(-1, 3, 32, 32))[0]
    assert features.shape == (64, 2048, 1, 1) and torch.isfinite(features).all()
    x = .5 + 1e-4 * torch.randn(1, 128, 147, device='cuda')
    y = .5 + 1e-4 * torch.randn(1, 256, 147, device='cuda')
    field, _ = _softmax_toward(x, y, torch.ones(1, 256, device='cuda'), .02 * 147**.5, 1e-8)
    bound = field.norm(dim=-1).max().item()
    assert bound <= 1.00001
    result = dict(status='passed', gpu=torch.cuda.get_device_name(),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  seconds_per_update=timings, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  generated_batch=128, real_batch=256, fid_batch=64,
                  objective='softmax-riesz', field_rms_normalization='none',
                  pretraining_lpips_weight=args.lpips_weight, stage_diagnostics=stage_diagnostics,
                  perceptual=perceptual_loss.metadata if perceptual_loss is not None else None,
                  softmax_numerics='direct_pairwise_differences', unit_average_norm=bound)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
