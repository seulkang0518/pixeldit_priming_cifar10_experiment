"""Frozen VGG LPIPS for paired CIFAR reconstruction, never distributional matching."""
import hashlib
from importlib.metadata import version
from pathlib import Path

import torch


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, cache, device='cpu'):
        super().__init__()
        import lpips

        if cache is None:
            raise ValueError('LPIPS requires an explicit cache directory')
        cache = Path(cache).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        previous_hub = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(cache / 'hub'))
            # Torchvision initializes layers before loading weights. Preserve RNG.
            with torch.random.fork_rng(devices=[]):
                self.metric = lpips.LPIPS(net='vgg', version='0.1', pretrained=True,
                                          pnet_rand=False, pnet_tune=False,
                                          eval_mode=True, verbose=False)
        finally:
            torch.hub.set_dir(previous_hub)
        self.requires_grad_(False).eval()
        digest = hashlib.sha256()
        for name, tensor in sorted(self.metric.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.numpy().tobytes())
        self.metadata = dict(package='lpips', package_version=version('lpips'),
                             torchvision_version=version('torchvision'),
                             network='vgg16', lpips_version='0.1',
                             weights_sha256=digest.hexdigest(), frozen=True,
                             input='native CIFAR RGB 32x32; [0,1] converted to [-1,1]',
                             resize=False, cache=str(cache))
        self.to(device)

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction[0].numel() != 3072:
            raise ValueError('Perceptual priming requires matching CIFAR RGB images')
        # Frozen weights preserve the differentiable path to generated images.
        return self.metric(prediction.reshape(-1, 3, 32, 32),
                           target.detach().reshape(-1, 3, 32, 32), normalize=True).mean()
