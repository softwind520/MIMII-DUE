"""Forward DDPM process and noise-prediction training objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def make_beta_schedule(name: str, steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Create a float64 schedule before registering float32 model buffers."""
    if steps < 2:
        raise ValueError("Diffusion training_steps must be at least 2")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("Expected 0 < beta_start < beta_end < 1")

    if name == "linear":
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    elif name == "sigmoid":
        values = torch.linspace(-6.0, 6.0, steps, dtype=torch.float64).sigmoid()
        values = (values - values[0]) / (values[-1] - values[0])
        betas = beta_start + values * (beta_end - beta_start)
    elif name == "cosine":
        offset = 0.008
        times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        cumulative = torch.cos((times + offset) / (1 + offset) * torch.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = 1.0 - cumulative[1:] / cumulative[:-1]
        betas = betas.clamp(min=beta_start, max=0.999)
    else:
        raise ValueError(f"Unsupported beta schedule: {name!r}")
    return betas


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather one schedule value per sample and make it broadcastable."""
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM forward process with the epsilon-prediction MSE objective."""

    def __init__(self, config: dict):
        super().__init__()
        diffusion_config = config["diffusion"]
        self.training_steps = int(diffusion_config["training_steps"])
        self.prediction_type = diffusion_config.get("prediction_type", "epsilon")
        if self.prediction_type != "epsilon":
            raise ValueError("Stage 3 currently supports prediction_type='epsilon' only")

        betas = make_beta_schedule(
            diffusion_config["beta_schedule"],
            self.training_steps,
            float(diffusion_config.get("beta_start", 1e-4)),
            float(diffusion_config.get("beta_end", 2e-2)),
        ).float()
        alphas = 1.0 - betas
        cumulative_alphas = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("cumulative_alphas", cumulative_alphas)
        self.register_buffer("sqrt_cumulative_alphas", cumulative_alphas.sqrt())
        self.register_buffer(
            "sqrt_one_minus_cumulative_alphas",
            (1.0 - cumulative_alphas).sqrt(),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.training_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = (
            extract(self.sqrt_cumulative_alphas, timesteps, clean.shape) * clean
            + extract(self.sqrt_one_minus_cumulative_alphas, timesteps, clean.shape) * noise
        )
        return noisy, noise

    def training_loss(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the standard simplified DDPM epsilon-prediction loss."""
        if timesteps is None:
            timesteps = self.sample_timesteps(clean.shape[0], clean.device)
        noisy, target_noise = self.q_sample(clean, timesteps, noise)
        predicted_noise = model(noisy, timesteps)
        return F.mse_loss(predicted_noise, target_noise)

    def ddim_reconstruct(self, *args, **kwargs):
        """Partial-noise DDIM reconstruction is introduced in stage 4."""
        del args, kwargs
        raise NotImplementedError("DDIM reconstruction is implemented in stage 4.")
