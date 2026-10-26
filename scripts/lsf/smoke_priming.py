"""Full-size CIFAR GPU memory, gradient, and FID smoke test (not an experiment)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import build_backbone
from data import load_dataset
from fid import load_inception
from main import generator_step, update_ema
from priming import reconstruction_step
import copy

torch.set_num_threads(4)
torch.manual_seed(42)
assert torch.cuda.is_available()
device = 'cuda'
print('GPU:', torch.cuda.get_device_name(), flush=True)
deadline = time.monotonic() + 1200
while not any((Path(sys.argv[1]) / 'CIFAR10/raw' / name).exists()
              for name in ('cifar-10-binary.tar.gz', 'cifar-10-python.tar.gz')):
    if time.monotonic() > deadline:
        raise RuntimeError('Dataset preparation did not finish within 20 minutes')
    time.sleep(5)
_, (images, _), _ = load_dataset('cifar10', Path(sys.argv[1]))
targets = images.to(device=device, dtype=torch.float32) / 255
model = build_backbone('cifar10', 'unet', 3072, 128, [128, 256], 4).to(device)
ema = copy.deepcopy(model).requires_grad_(False).eval()
inception = load_inception(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
timings = {}
for stage in ('reconstruction', 'distributional'):
    if stage == 'distributional':
        del optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(3):
        noise = torch.randn(64, 3072, device=device)
        if stage == 'reconstruction':
            info = reconstruction_step(model, targets[:64], noise, optimizer)
        else:
            info = generator_step(model, noise, targets[:256], optimizer, 64,
                objective='softmax-riesz', multiscale_options=dict(
                    image_shape=(3,32,32), coarse_sizes=[8,16], patch_sizes=[7,14],
                    crops_per_size=1, patch_rng=torch.Generator().manual_seed(44),
                    global_weight=1., coarse_weight=1., patch_weight=1.))
        update_ema(model, ema, .999)
        assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.cuda.synchronize()
    timings[stage] = (time.monotonic() - start) / 3
    print(stage, timings[stage], info, flush=True)
with torch.no_grad():
    features = inception(ema(torch.randn(4,3072,device=device)).reshape(-1,3,32,32))[0]
assert features.shape == (4,2048,1,1) and torch.isfinite(features).all()
result = dict(status='passed', gpu=torch.cuda.get_device_name(), seconds_per_update=timings,
              peak_gib=torch.cuda.max_memory_allocated()/2**30)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
