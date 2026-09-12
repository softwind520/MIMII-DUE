"""Unit tests for the stage-3 U-Net and DDPM objective."""

from copy import deepcopy
from pathlib import Path
import unittest

import torch

from diffusion.config import load_config
from diffusion.diffusion import GaussianDiffusion, make_beta_schedule
from diffusion.unet import UNetDenoiser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> dict:
    config = deepcopy(load_config(PROJECT_ROOT / "diffusion.yaml"))
    config["data"]["n_mels"] = 32
    config["data"]["patch_frames"] = 32
    config["model"]["base_channels"] = 8
    config["model"]["channel_multipliers"] = [1, 2, 4]
    config["model"]["residual_blocks"] = 1
    config["model"]["attention_resolutions"] = [8]
    config["model"]["attention_heads"] = 4
    config["model"]["condition_dimension"] = 32
    config["diffusion"]["training_steps"] = 20
    return config


class DiffusionTest(unittest.TestCase):
    def test_sigmoid_schedule_is_valid_and_increasing(self) -> None:
        betas = make_beta_schedule("sigmoid", 20, 1e-4, 2e-2)
        self.assertEqual(tuple(betas.shape), (20,))
        self.assertTrue(torch.all(betas[1:] >= betas[:-1]))
        self.assertAlmostEqual(float(betas[0]), 1e-4)
        self.assertAlmostEqual(float(betas[-1]), 2e-2)

    def test_unet_preserves_patch_shape(self) -> None:
        model = UNetDenoiser(tiny_config())
        inputs = torch.randn(2, 1, 32, 32)
        timesteps = torch.tensor([0, 19])
        outputs = model(inputs, timesteps)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(torch.isfinite(outputs).all())

    def test_ddpm_loss_supports_backward(self) -> None:
        config = tiny_config()
        model = UNetDenoiser(config)
        diffusion = GaussianDiffusion(config)
        inputs = torch.rand(2, 1, 32, 32).mul(2.0).sub(1.0)
        loss = diffusion.training_loss(model, inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.output_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
