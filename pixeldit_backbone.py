"""Original NVIDIA PixelDiT adapted only at the unconditional image interface."""
from pathlib import Path
import sys

import torch

# Keep the upstream package unchanged, including its absolute package imports.
_vendor = Path(__file__).resolve().parent / 'third_party' / 'pixeldit'
sys.path.insert(0, str(_vendor))
from pixdit_core.pixeldit_c2i import PixDiT


PIXELDIT_DEFAULTS = dict(pixeldit_width=768, pixeldit_patch_depth=12,
                        pixeldit_pixel_depth=2, pixeldit_heads=12,
                        pixeldit_patch_size=4, pixeldit_pixel_width=16)


def validate_pixeldit(width, patch_depth, pixel_depth, heads, patch_size, pixel_width):
    if min(width, patch_depth, pixel_depth, heads, patch_size, pixel_width) <= 0:
        raise ValueError('PixelDiT sizes must be positive')
    if width % heads or (width // heads) % 4 or pixel_width % 4 or 32 % patch_size:
        raise ValueError('PixelDiT requires width/head divisible by 4, pixel width divisible by 4, and patch size dividing 32')


class CIFAR10PixelDiTParticleGenerator(torch.nn.Module):
    """Flat Gaussian/corrupted RGB -> flat [0,1] RGB, in one network call.

    No class labels or noise levels are supplied by training. The upstream
    conditioning inputs are constants, so their embeddings act as learned
    constants. This preserves Hudson's unconditional, fixed-corruption recipe.
    The upstream dual pathway blocks and initialization remain unchanged.
    """

    def __init__(self, width=768, patch_depth=12, pixel_depth=2, heads=12,
                 patch_size=4, pixel_width=16):
        super().__init__()
        validate_pixeldit(width, patch_depth, pixel_depth, heads, patch_size, pixel_width)
        self.network = PixDiT(in_channels=3, num_groups=heads, hidden_size=width,
                             pixel_hidden_size=pixel_width, patch_depth=patch_depth,
                             pixel_depth=pixel_depth, patch_size=patch_size,
                             num_classes=1, use_pixel_abs_pos=True,
                             pit_adaln_post_modulation=False)

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[1] != 3072:
            raise ValueError('CIFAR PixelDiT expects flattened inputs with shape (batch, 3072)')
        batch = latent.shape[0]
        fixed_time = latent.new_zeros(batch)
        fixed_label = torch.zeros(batch, dtype=torch.long, device=latent.device)
        logits = self.network(latent.reshape(batch, 3, 32, 32), fixed_time, fixed_label)
        return logits.sigmoid().flatten(1)


class GeneratorDataParallel(torch.nn.DataParallel):
    """Split generator forwards; gather ALL particles before evaluating the loss.

    Checkpoints use the underlying model's original names. EMA and Adam keep
    their existing semantics, and checkpoints do not depend on replica count.
    This is single-process data parallelism, not independent per-GPU MMD losses.
    """

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)
