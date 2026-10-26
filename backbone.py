"""Shared backbones for one-pass MNIST and CIFAR-10 image generation.

CIFAR DDPM++ blocks/topology adapted from Google Research's
yang-song/score_sde_pytorch (Copyright 2020 The Google Research Authors,
Apache-2.0; see assets/SCORE_SDE_LICENSE). Modifications omit time embedding,
use a sigmoid image head, and implement attention with PyTorch SDPA.
"""

import math
import torch


class ParticleGenerator(torch.nn.Module):
    """Small shared MLP mapping Gaussian latents to flattened image pixels."""

    def __init__(self, latent_dim: int = 32, hidden_dims: tuple[int, ...] = (128, 256), output_dim: int = 784):
        super().__init__()
        layers = []
        width = latent_dim
        for hidden in hidden_dims:
            layers.extend([torch.nn.Linear(width, hidden), torch.nn.SiLU()])
            width = hidden
        layers.extend([torch.nn.Linear(width, output_dim), torch.nn.Sigmoid()])
        self.network = torch.nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)



class ResidualConvBlock(torch.nn.Module):
    """Per-image normalization and residual convolutions for spatial features."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.GroupNorm(math.gcd(8, in_channels), in_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(math.gcd(8, out_channels), out_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (torch.nn.Identity() if in_channels == out_channels
                     else torch.nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (self.layers(features) + self.skip(features)) / math.sqrt(2)



class UNetParticleGenerator(torch.nn.Module):
    """Compact U-Net adaptation for one-pass generation, without diffusion time."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        width = base_channels
        self.input_conv = torch.nn.Conv2d(1, width, 3, padding=1)
        self.encoder28 = ResidualConvBlock(width, width)
        self.down14 = torch.nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder14 = ResidualConvBlock(2 * width, 2 * width)
        self.down7 = torch.nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.bottleneck = torch.nn.Sequential(
            ResidualConvBlock(4 * width, 4 * width),
            ResidualConvBlock(4 * width, 4 * width),
        )
        self.up14 = torch.nn.Conv2d(4 * width, 2 * width, 3, padding=1)
        self.decoder14 = ResidualConvBlock(4 * width, 2 * width)
        self.up28 = torch.nn.Conv2d(2 * width, width, 3, padding=1)
        self.decoder28 = ResidualConvBlock(2 * width, width)
        self.output = torch.nn.Sequential(
            torch.nn.GroupNorm(math.gcd(8, width), width),
            torch.nn.SiLU(), torch.nn.Conv2d(width, 1, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        noise = latent.reshape(-1, 1, 28, 28)
        skip28 = self.encoder28(self.input_conv(noise))
        skip14 = self.encoder14(self.down14(skip28))
        features = self.bottleneck(self.down7(skip14))
        features = self.up14(torch.nn.functional.interpolate(
            features, size=skip14.shape[-2:], mode="bilinear", align_corners=False))
        features = self.decoder14(torch.cat([features, skip14], dim=1))
        features = self.up28(torch.nn.functional.interpolate(
            features, size=skip28.shape[-2:], mode="bilinear", align_corners=False))
        features = self.decoder28(torch.cat([features, skip28], dim=1))
        return self.output(features).flatten(1)



def ddpm_conv(in_channels, out_channels, kernel_size=3, init_scale=1.0):
    """DDPM fan-average uniform initialization, including its near-zero branches."""
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=kernel_size // 2)
    scale = 1e-10 if init_scale == 0 else init_scale
    bound = math.sqrt(3 * scale / ((in_channels + out_channels) * kernel_size ** 2 / 2))
    torch.nn.init.uniform_(conv.weight, -bound, bound)
    torch.nn.init.zeros_(conv.bias)
    return conv


def ddpm_norm(channels):
    return torch.nn.GroupNorm(math.gcd(min(channels // 4, 32), channels), channels, eps=1e-6)


class ResidualBlock(torch.nn.Module):
    """DDPM++ BigGAN block with optional resampling and timestep conditioning.

    Adapted from score_sde_pytorch/models/layerspp.py (Apache-2.0).
    """
    def __init__(self, in_channels, out_channels=None, *, up=False, down=False, dropout=0.1,
                 time_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.up, self.down = up, down
        self.norm1 = ddpm_norm(in_channels)
        self.conv1 = ddpm_conv(in_channels, out_channels)
        self.norm2 = ddpm_norm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ddpm_conv(out_channels, out_channels, init_scale=0)
        self.skip = (ddpm_conv(in_channels, out_channels, 1)
                     if in_channels != out_channels or up or down else torch.nn.Identity())
        self.time_projection = (torch.nn.Linear(time_channels, out_channels)
                                if time_channels is not None else None)

    def forward(self, x, time_embedding=None):
        h = torch.nn.functional.silu(self.norm1(x))
        if self.up:
            h = torch.nn.functional.interpolate(h, scale_factor=2, mode='nearest')
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        elif self.down:
            h = torch.nn.functional.avg_pool2d(h, 2)
            x = torch.nn.functional.avg_pool2d(x, 2)
        h = self.conv1(h)
        if self.time_projection is not None:
            if time_embedding is None:
                raise ValueError('A timestep embedding is required')
            h = h + self.time_projection(torch.nn.functional.silu(time_embedding))[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return (self.skip(x) + h) / math.sqrt(2)


class SpatialAttention(torch.nn.Module):
    """Single-head DDPM++ attention with near-zero output projection."""
    def __init__(self, channels):
        super().__init__()
        self.norm = ddpm_norm(channels)
        self.q = ddpm_conv(channels, channels, 1, init_scale=0.1)
        self.k = ddpm_conv(channels, channels, 1, init_scale=0.1)
        self.v = ddpm_conv(channels, channels, 1, init_scale=0.1)
        self.projection = ddpm_conv(channels, channels, 1, init_scale=0)

    def forward(self, x):
        batch, channels, height, width = x.shape
        normalized = self.norm(x)
        q, k, v = [layer(normalized).flatten(2) for layer in (self.q, self.k, self.v)]
        attended = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).unsqueeze(1), k.transpose(1, 2).unsqueeze(1),
            v.transpose(1, 2).unsqueeze(1))
        attended = attended.squeeze(1).transpose(1, 2).reshape(batch, channels, height, width)
        return (x + self.projection(attended)) / math.sqrt(2)


class CIFAR10UNetParticleGenerator(torch.nn.Module):
    """Official CIFAR DDPM++ spatial topology adapted to one-pass image generation.

    Reference: yang-song/score_sde_pytorch models/ncsnpp.py and
    configs/vp/cifar10_ddpmpp.py. Default: no time embedding, sigmoid image output.
    The DDPM variant adds timestep embeddings and an unrestricted noise head.
    """
    def __init__(self, base_channels: int = 128, num_res_blocks: int = 4, *,
                 time_conditioned: bool = False):
        super().__init__()
        self.time_conditioned = time_conditioned
        self.base_channels = base_channels
        time_channels = 4 * base_channels if time_conditioned else None
        self.time_mlp = (torch.nn.Sequential(
            torch.nn.Linear(base_channels, time_channels), torch.nn.SiLU(),
            torch.nn.Linear(time_channels, time_channels)) if time_conditioned else None)
        widths = [base_channels * multiplier for multiplier in (1, 2, 2, 2)]
        self.input_conv = ddpm_conv(3, widths[0])
        self.encoders = torch.nn.ModuleList()
        self.downsamples = torch.nn.ModuleList()
        skip_channels = [widths[0]]
        channels = widths[0]
        for level, width in enumerate(widths):
            stage = torch.nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(torch.nn.Sequential(
                    ResidualBlock(channels, width, time_channels=time_channels),
                    SpatialAttention(width) if level == 1 else torch.nn.Identity()))
                channels = width
                skip_channels.append(channels)
            self.encoders.append(stage)
            if level < 3:
                self.downsamples.append(ResidualBlock(channels, down=True, time_channels=time_channels))
                skip_channels.append(channels)
        self.middle = torch.nn.Sequential(ResidualBlock(channels, time_channels=time_channels), SpatialAttention(channels),
                                          ResidualBlock(channels, time_channels=time_channels))
        self.decoders = torch.nn.ModuleList()
        self.decoder_attention = torch.nn.ModuleList()
        self.upsamples = torch.nn.ModuleList()
        for level in (3, 2, 1, 0):
            stage = torch.nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(ResidualBlock(channels + skip_channels.pop(), widths[level], time_channels=time_channels))
                channels = widths[level]
            self.decoders.append(stage)
            self.decoder_attention.append(SpatialAttention(channels) if level == 1 else torch.nn.Identity())
            if level > 0:
                self.upsamples.append(ResidualBlock(channels, up=True, time_channels=time_channels))
        assert not skip_channels
        self.output = torch.nn.Sequential(ddpm_norm(channels), torch.nn.SiLU(),
                                          ddpm_conv(channels, 3, init_scale=0),
                                          torch.nn.Identity() if time_conditioned else torch.nn.Sigmoid())

    def forward(self, latent, timesteps=None):
        embedding = None
        if self.time_conditioned:
            if timesteps is None or timesteps.shape != (len(latent),):
                raise ValueError('Expected one timestep per image')
            half = self.base_channels // 2
            frequencies = torch.exp(-math.log(10000) * torch.arange(
                half, device=latent.device, dtype=torch.float32) / max(half - 1, 1))
            angles = timesteps.float()[:, None] * frequencies[None]
            positional = torch.cat([angles.sin(), angles.cos()], dim=1)
            positional = torch.nn.functional.pad(positional, (0, self.base_channels - positional.shape[1]))
            embedding = self.time_mlp(positional)
        elif timesteps is not None:
            raise ValueError('This generator has no timestep conditioning')
        x = self.input_conv(latent.reshape(-1, 3, 32, 32))
        skips = [x]
        for level, stage in enumerate(self.encoders):
            for block in stage:
                x = block[1](block[0](x, embedding))
                skips.append(x)
            if level < 3:
                x = self.downsamples[level](x, embedding)
                skips.append(x)
        x = self.middle[2](self.middle[1](self.middle[0](x, embedding)), embedding)
        for level, stage in enumerate(self.decoders):
            for block in stage:
                x = block(torch.cat([x, skips.pop()], dim=1), embedding)
            x = self.decoder_attention[level](x)
            if level < 3:
                x = self.upsamples[level](x, embedding)
        assert not skips
        output = self.output(x)
        return output if self.time_conditioned else output.flatten(1)



class DiTBlock(torch.nn.Module):
    """Unconditional DiT-style block with learned constant adaLN-Zero modulation.

    A time/class embedding is unnecessary for the one-pass distributional model.
    Its six modulation vectors are therefore learned directly, independently per
    block. Zero gates initialize each attention/MLP residual branch to identity.
    """

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = torch.nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.projection = torch.nn.Linear(width, width)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 4 * width),
                                       torch.nn.GELU(approximate='tanh'),
                                       torch.nn.Linear(4 * width, width))
        self.modulation = torch.nn.Parameter(torch.zeros(6, width))

    def forward(self, tokens):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation.unbind(0)
        batch, count, width = tokens.shape
        normalized = self.norm1(tokens) * (1 + scale1) + shift1
        qkv = self.qkv(normalized).reshape(batch, count, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        tokens = tokens + gate1 * self.projection(attended.transpose(1, 2).reshape(batch, count, width))
        return tokens + gate2 * self.mlp(self.norm2(tokens) * (1 + scale2) + shift2)


class CIFAR10DiTParticleGenerator(torch.nn.Module):
    """Compact unconditional patch transformer producing pixels in one pass.

    DiT-style architecture, not a diffusion/MeanFlow objective. Fixed 2D sine/
    cosine positions and global attention operate on nonoverlapping noise patches.
    A small nonzero pixel head lets residual gates receive gradients immediately.
    """

    def __init__(self, width=384, depth=12, heads=6, patch_size=2):
        super().__init__()
        if (width <= 0 or depth <= 0 or heads <= 0 or patch_size <= 0
                or width % heads or width % 4 or 32 % patch_size):
            raise ValueError('DiT requires positive sizes, width divisible by heads and 4, and patch size dividing 32')
        self.patch_size = patch_size
        self.grid_size = 32 // patch_size
        self.patch_embed = torch.nn.Conv2d(3, width, patch_size, stride=patch_size)
        coordinates = torch.arange(self.grid_size, dtype=torch.float32)
        y, x = torch.meshgrid(coordinates, coordinates, indexing='ij')
        frequencies = 10000 ** (-torch.arange(width // 4, dtype=torch.float32) / (width // 4))
        phases = [axis.flatten()[:, None] * frequencies for axis in (y, x)]
        positions = torch.cat([part for phase in phases for part in (phase.sin(), phase.cos())], dim=1)
        self.register_buffer('positions', positions.unsqueeze(0))
        self.blocks = torch.nn.ModuleList([DiTBlock(width, heads) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(width, eps=1e-6)
        self.head = torch.nn.Linear(width, 3 * patch_size ** 2)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        torch.nn.init.zeros_(self.patch_embed.bias)
        torch.nn.init.normal_(self.head.weight, std=1e-3)

    def unpatchify(self, patches):
        batch = patches.shape[0]
        grid, patch = self.grid_size, self.patch_size
        # Patch vectors store row, column, channel; output uses CHW ordering.
        pixels = patches.reshape(batch, grid, grid, patch, patch, 3)
        return pixels.permute(0, 5, 1, 3, 2, 4).reshape(batch, 3, 32, 32)

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[1] != 3072:
            raise ValueError('CIFAR DiT expects flattened noise with shape (batch, 3072)')
        tokens = self.patch_embed(latent.reshape(-1, 3, 32, 32)).flatten(2).transpose(1, 2)
        tokens = tokens + self.positions
        for block in self.blocks:
            tokens = block(tokens)
        return self.unpatchify(self.head(self.norm(tokens))).sigmoid().flatten(1)


def build_backbone(dataset, architecture, latent_dim, base_channels, hidden_dims, num_res_blocks,
                   *, dit_width=384, dit_depth=12, dit_heads=6, dit_patch_size=2,
                   pixeldit_width=768, pixeldit_patch_depth=12, pixeldit_pixel_depth=2,
                   pixeldit_heads=12, pixeldit_patch_size=4, pixeldit_pixel_width=16):
    """Construct an architecture without dataset-specific trainer entry points."""
    if architecture == "mlp":
        return ParticleGenerator(latent_dim, tuple(hidden_dims), 784 if dataset == "mnist" else 3072)
    if architecture == 'pixeldit':
        if dataset != 'cifar10' or latent_dim != 3072:
            raise ValueError('PixelDiT currently supports CIFAR-10 with latent_dim=3072')
        from pixeldit_backbone import CIFAR10PixelDiTParticleGenerator
        return CIFAR10PixelDiTParticleGenerator(pixeldit_width, pixeldit_patch_depth,
            pixeldit_pixel_depth, pixeldit_heads, pixeldit_patch_size, pixeldit_pixel_width)
    if architecture == 'dit':
        if dataset != 'cifar10' or latent_dim != 3072:
            raise ValueError('DiT currently supports CIFAR-10 with latent_dim=3072')
        return CIFAR10DiTParticleGenerator(dit_width, dit_depth, dit_heads, dit_patch_size)
    if architecture != "unet":
        raise ValueError(f"Unknown architecture: {architecture}")
    if dataset == "mnist":
        return UNetParticleGenerator(base_channels)
    if dataset == "cifar10":
        return CIFAR10UNetParticleGenerator(base_channels, num_res_blocks)
    raise ValueError(f"Unknown dataset: {dataset}")


def backbone_metadata(dataset, architecture, latent_dim, base_channels, hidden_dims, num_res_blocks,
                      *, dit_width=384, dit_depth=12, dit_heads=6, dit_patch_size=2,
                      pixeldit_width=768, pixeldit_patch_depth=12, pixeldit_pixel_depth=2,
                      pixeldit_heads=12, pixeldit_patch_size=4, pixeldit_pixel_width=16):
    if architecture == "mlp":
        return [latent_dim, *hidden_dims, 784 if dataset == "mnist" else 3072]
    if architecture == 'pixeldit':
        return dict(name='cifar10_original_pixeldit_onepass', input_shape=[3, 32, 32],
                    width=pixeldit_width, patch_depth=pixeldit_patch_depth,
                    pixel_depth=pixeldit_pixel_depth, heads=pixeldit_heads,
                    patch_size=pixeldit_patch_size, pixel_width=pixeldit_pixel_width,
                    patch_tokens=(32 // pixeldit_patch_size) ** 2, pixel_tokens=1024,
                    normalization='RMSNorm', hidden_activation='patch_SwiGLU_pixel_GELU',
                    positional_embedding='patch_RoPE_and_pixel_fixed_2d_sincos',
                    time_conditioning=False, class_conditioning=False,
                    upstream_conditioning_inputs='constant t=0, dummy label=0',
                    sampling_network_evaluations=1, output_activation='sigmoid',
                    initialization='upstream_zero_output_and_adaln_modulation',
                    upstream='NVlabs/PixelDiT@f7d7c35f71893d81c6e71d2576ea80601a1e8529')
    if architecture == 'dit':
        return dict(name='cifar10_compact_dit_onepass', input_shape=[3, 32, 32],
                    width=dit_width, depth=dit_depth, heads=dit_heads, patch_size=dit_patch_size,
                    tokens=(32 // dit_patch_size) ** 2, mlp_ratio=4,
                    normalization='LayerNorm', positional_embedding='fixed_2d_sincos',
                    modulation='learned_constant_adaln_zero', dropout=0.,
                    time_conditioning=False, class_conditioning=False,
                    sampling_network_evaluations=1, output_activation='sigmoid',
                    initialization='xavier_linear_zero_residual_gates_pixel_head_std_1e-3')
    metadata = {
        "name": "compact_residual_unet" if dataset == "mnist" else "cifar10_ddpmpp_onepass_unet",
        "channels": [base_channels, 2 * base_channels, 4 * base_channels] if dataset == "mnist"
                    else [base_channels, 2 * base_channels, 2 * base_channels, 2 * base_channels],
        "resolutions": [28, 14, 7] if dataset == "mnist" else [32, 16, 8, 4],
        "skip_connections": "concatenation", "normalization": "GroupNorm",
        "upsampling": "bilinear_then_conv", "input_shape": [1, 28, 28] if dataset == "mnist" else [3, 32, 32],
        "time_conditioning": False, "sampling_network_evaluations": 1,
    }
    if dataset == "cifar10":
        metadata.update(residual_blocks_per_stage=num_res_blocks,
                        decoder_residual_blocks_per_stage=num_res_blocks + 1,
                        decoder_resolutions=[4, 8, 16, 32],
                        skip_connections="concatenation_after_each_encoder_block_and_downsample",
                        upsampling="biggan_residual_nearest", downsampling="biggan_residual_average_pool",
                        attention_resolutions=[16], bottleneck_attention=True,
                        encoder_attention="after_each_residual_block_at_16",
                        decoder_attention="after_stage_at_16", residual_block_type="biggan",
                        dropout=0.1, groupnorm_eps=1e-6, skip_rescale=True, fir=False,
                        initialization="ddpm_fan_average_uniform_near_zero_output_branches",
                        output_activation="sigmoid",
                        adaptations=["omit_time_embedding", "sigmoid_image_output"],
                        scale_reference="https://github.com/yang-song/score_sde_pytorch/blob/main/configs/vp/cifar10_ddpmpp.py")
    return metadata
