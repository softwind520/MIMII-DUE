"""Timestep-conditioned 2D U-Net used as the DDPM noise predictor."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 32) -> int:
    """Return the largest useful GroupNorm divisor up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer diffusion timesteps with sinusoidal features."""

    def __init__(self, dimension: int):
        super().__init__()
        if dimension < 4:
            raise ValueError("Time embedding dimension must be at least 4")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """Residual convolutional block with scale-shift timestep modulation."""

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.embedding = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, out_channels * 2))
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        scale, shift = self.embedding(embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return hidden + self.skip(inputs)


class AttentionBlock(nn.Module):
    """Spatial self-attention backed by PyTorch's efficient SDPA kernel."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError(f"Attention channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        query, key, value = self.qkv(self.norm(inputs)).chunk(3, dim=1)

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, self.head_dim, height * width).transpose(2, 3)

        attended = F.scaled_dot_product_attention(
            reshape(query), reshape(key), reshape(value), dropout_p=0.0
        )
        attended = attended.transpose(2, 3).reshape(batch, channels, height, width)
        return inputs + self.projection(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(inputs, scale_factor=2.0, mode="nearest"))


class UNetStage(nn.Module):
    """A stack of residual blocks followed by optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        residual_blocks: int,
        dropout: float,
        use_attention: bool,
        attention_heads: int,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(residual_blocks):
            blocks.append(ResidualBlock(current_channels, out_channels, embedding_dim, dropout))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels, attention_heads) if use_attention else nn.Identity()

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.attention(hidden)


class UNetDenoiser(nn.Module):
    """Predict noise for a spectrogram patch at a diffusion timestep.

    Stage 3 is deliberately unconditional: ``section_id`` and ``domain_id``
    are accepted for a stable future interface but are not used until stage 5.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]
        data_config = config["data"]
        in_channels = int(model_config["in_channels"])
        base_channels = int(model_config["base_channels"])
        multipliers = [int(value) for value in model_config["channel_multipliers"]]
        residual_blocks = int(model_config["residual_blocks"])
        attention_resolutions = {int(value) for value in model_config["attention_resolutions"]}
        attention_heads = int(model_config["attention_heads"])
        dropout = float(model_config["dropout"])
        embedding_dim = int(model_config["condition_dimension"])
        input_resolution = int(data_config["patch_frames"])

        if not multipliers:
            raise ValueError("channel_multipliers cannot be empty")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        divisor = 2 ** (len(multipliers) - 1)
        if input_resolution % divisor:
            raise ValueError(f"Patch resolution {input_resolution} must be divisible by {divisor}")

        channels = [base_channels * multiplier for multiplier in multipliers]
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.input_projection = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        down_stages: list[nn.Module] = []
        downsamplers: list[nn.Module] = []
        current_channels = channels[0]
        resolution = input_resolution
        for index, out_channels in enumerate(channels):
            down_stages.append(
                UNetStage(
                    current_channels,
                    out_channels,
                    embedding_dim,
                    residual_blocks,
                    dropout,
                    resolution in attention_resolutions,
                    attention_heads,
                )
            )
            current_channels = out_channels
            if index < len(channels) - 1:
                downsamplers.append(Downsample(current_channels))
                resolution //= 2
        self.down_stages = nn.ModuleList(down_stages)
        self.downsamplers = nn.ModuleList(downsamplers)

        self.middle_block1 = ResidualBlock(current_channels, current_channels, embedding_dim, dropout)
        self.middle_attention = AttentionBlock(current_channels, attention_heads)
        self.middle_block2 = ResidualBlock(current_channels, current_channels, embedding_dim, dropout)

        upsamplers: list[nn.Module] = []
        up_stages: list[nn.Module] = []
        reversed_indices = list(reversed(range(len(channels))))
        for position, channel_index in enumerate(reversed_indices):
            stage_channels = channels[channel_index]
            up_stages.append(
                UNetStage(
                    current_channels + stage_channels,
                    stage_channels,
                    embedding_dim,
                    residual_blocks,
                    dropout,
                    (input_resolution // (2**channel_index)) in attention_resolutions,
                    attention_heads,
                )
            )
            current_channels = stage_channels
            if position < len(reversed_indices) - 1:
                next_channels = channels[channel_index - 1]
                upsamplers.append(Upsample(current_channels, next_channels))
                current_channels = next_channels
        self.up_stages = nn.ModuleList(up_stages)
        self.upsamplers = nn.ModuleList(upsamplers)

        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.output_projection = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_patch: torch.Tensor,
        timestep: torch.Tensor,
        section_id: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del section_id, domain_id
        if noisy_patch.ndim != 4:
            raise ValueError(f"Expected BCHW input, received shape {tuple(noisy_patch.shape)}")
        if timestep.ndim == 0:
            timestep = timestep.expand(noisy_patch.shape[0])
        if timestep.shape != (noisy_patch.shape[0],):
            raise ValueError("One diffusion timestep is required per batch element")

        embedding = self.time_embedding(timestep)
        hidden = self.input_projection(noisy_patch)
        skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.down_stages):
            hidden = stage(hidden, embedding)
            skips.append(hidden)
            if index < len(self.downsamplers):
                hidden = self.downsamplers[index](hidden)

        hidden = self.middle_block1(hidden, embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle_block2(hidden, embedding)

        for index, stage in enumerate(self.up_stages):
            skip = skips.pop()
            if hidden.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    f"U-Net skip resolution mismatch: {hidden.shape[-2:]} and {skip.shape[-2:]}"
                )
            hidden = stage(torch.cat((hidden, skip), dim=1), embedding)
            if index < len(self.upsamplers):
                hidden = self.upsamplers[index](hidden)

        return self.output_projection(F.silu(self.output_norm(hidden)))


# Keep the stage-1 public name valid while stage 5 adds metadata conditioning.
ConditionalUNet = UNetDenoiser
