"""Check the full-width tuned batch on its allocated GPU before training."""
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import build_backbone
from main import generator_step, horizontal_flip_batch, update_ema

torch.set_num_threads(4)
torch.manual_seed(42)
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = build_backbone('cifar10', 'unet', 3072, 128, [128, 256], 4).cuda()
ema = copy.deepcopy(model).eval().requires_grad_(False)
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
objective = sys.argv[3] if len(sys.argv) > 3 else 'softmax-riesz'
no_field_rms_normalization = len(sys.argv) > 4 and sys.argv[4] == 'none'
sinkhorn_mode = sys.argv[5] if len(sys.argv) > 5 else 'wflow'
sinkhorn_iters = int(sys.argv[6]) if len(sys.argv) > 6 else 10
sinkhorn_tolerance = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-6
sinkhorn_ot_epsilon = float(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != 'none' else None
if len(sys.argv) > 9:
    checkpoint = torch.load(sys.argv[9], map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(checkpoint['model'])
    ema.load_state_dict(checkpoint['ema_model'])
    del checkpoint
    from data import load_dataset
    _, (training_images, _), _ = load_dataset('cifar10', Path(sys.argv[10]))
else:
    training_images = None
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
options = dict(image_shape=(3, 32, 32), coarse_sizes=[8, 16], patch_sizes=[7, 14],
               crops_per_size=4, patch_rng=torch.Generator().manual_seed(44),
               global_weight=2., coarse_weight=1., patch_weight=1.)
started = time.monotonic()
for _ in range(2):
    real = (torch.rand(256, 3072, device='cuda') if training_images is None else
            training_images[torch.randint(len(training_images), (256,))].cuda().float() / 255)
    real = horizontal_flip_batch(real, (3, 32, 32))
    reference_latent = torch.randn(128, 3072, device='cuda') if objective == 'sinkhorn' else None
    diagnostics = generator_step(model, torch.randn(128, 3072, device='cuda'), real,
                                 optimizer, 64, objective=objective,
                                 no_field_rms_normalization=no_field_rms_normalization,
                                 sinkhorn_latent=reference_latent, sinkhorn_mode=sinkhorn_mode,
                                 sinkhorn_iters=sinkhorn_iters, sinkhorn_tolerance=sinkhorn_tolerance,
                                 sinkhorn_ot_epsilon=sinkhorn_ot_epsilon,
                                 multiscale_options=options, max_grad_norm=1.)
    update_ema(model, ema, .9999)
    assert all(torch.isfinite(value) for value in diagnostics.values())
assert all(torch.isfinite(p).all() for p in model.parameters())
torch.cuda.synchronize()
seconds_per_update = (time.monotonic() - started) / 2
numerical_regression = None
if objective == 'softmax-riesz':
    from loss import _softmax_toward
    source = .5 + torch.randn(1, 128, 147, device='cuda') * 1e-4
    support = .5 + torch.randn(1, 256, 147, device='cuda') * 1e-4
    field, _ = _softmax_toward(source, support, torch.ones(1, 256, device='cuda'),
                              .02 * 147**.5, 1e-8)
    maximum = float(field.norm(dim=-1).max())
    assert maximum <= 1.00001, maximum
    numerical_regression = dict(near_collision_max_unit_average_norm=maximum, bound=1.)
    if no_field_rms_normalization:
        for tau in (.2, .05, .02):
            assert float(diagnostics[f'raw_force_rms_tau_{tau:g}']) <= 2 / (tau * 147**.5) * 1.00001
result = dict(status='passed', training_generated_batch=128, real_batch=256,
              learning_rate=learning_rate, objective=objective,
              field_rms_normalization=('none' if no_field_rms_normalization or
                                       objective == 'sinkhorn' and sinkhorn_mode == 'ot'
                                       else 'per_view' if objective == 'sinkhorn' else 'per_temperature'),
              sinkhorn_mode=sinkhorn_mode if objective == 'sinkhorn' else None,
              sinkhorn_iters=sinkhorn_iters, sinkhorn_tolerance=sinkhorn_tolerance,
              sinkhorn_ot_epsilon=sinkhorn_ot_epsilon,
              initialization='reconstruction' if training_images is not None else 'random',
              diagnostics={key: float(value) for key, value in diagnostics.items()},
              crops_per_size=4, gpu=torch.cuda.get_device_name(),
              peak_gib=torch.cuda.max_memory_allocated() / 2**30,
              seconds_per_update=seconds_per_update, numerical_regression=numerical_regression)
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
