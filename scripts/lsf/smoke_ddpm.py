#!/usr/bin/env python3
"""Allocated-GPU check of full-width DDPM updates, EMA, and reverse sampling."""
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import CIFAR10UNetParticleGenerator
from ddpm import Diffusion, train_step
from data import download_cifar10, load_cifar10_split

torch.set_num_threads(4)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
assert torch.cuda.is_available()
raw, _ = load_cifar10_split(download_cifar10(Path(sys.argv[1])), True)
clean = raw[:128].cuda().float().reshape(-1, 3, 32, 32) / 127.5 - 1
model = CIFAR10UNetParticleGenerator(time_conditioned=True).cuda()
ema = copy.deepcopy(model).eval().requires_grad_(False)
optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
diffusion = Diffusion('cuda')
start = time.monotonic()
losses = []
for step in range(1, 4):
    loss, grad, lr = train_step(model, ema, optimizer, diffusion, clean, step,
                               lr=.0002, warmup=5000, decay=.9999)
    losses.append(loss)
torch.cuda.synchronize()
seconds = time.monotonic() - start
# Short reverse chain exercises sampling mechanics without producing quality claims.
samples = Diffusion('cuda', timesteps=8).sample(
    ema, 64, generator=torch.Generator(device='cuda').manual_seed(12345))
assert torch.isfinite(samples).all() and samples.shape == (64, 3, 32, 32)
result = dict(gpu=torch.cuda.get_device_name(), losses=losses, three_update_seconds=seconds,
              peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
              parameters=sum(p.numel() for p in model.parameters()), sampling_shape=list(samples.shape))
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
