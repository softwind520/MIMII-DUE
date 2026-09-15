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


def tiny_conditional_config() -> dict:
    config = tiny_config()
    config["conditioning"]["use_section"] = True
    config["conditioning"]["use_domain"] = True
    config["conditioning"]["num_sections"] = 3
    config["conditioning"]["condition_dropout"] = 0.0
    return config


def tiny_section_only_config() -> dict:
    config = tiny_conditional_config()
    config["conditioning"]["use_domain"] = False
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

    def test_conditional_ddpm_loss_supports_backward(self) -> None:
        config = tiny_conditional_config()
        model = UNetDenoiser(config)
        diffusion = GaussianDiffusion(config)
        inputs = torch.rand(2, 1, 32, 32).mul(2.0).sub(1.0)
        loss = diffusion.training_loss(
            model,
            inputs,
            section_id=torch.tensor([0, 2]),
            domain_id=torch.tensor([0, 1]),
        )
        loss.backward()
        self.assertIsNotNone(model.metadata_conditioner.section_embedding.weight.grad)
        self.assertIsNotNone(model.metadata_conditioner.domain_embedding.weight.grad)

    def test_metadata_changes_conditional_embedding(self) -> None:
        model = UNetDenoiser(tiny_conditional_config()).eval()
        first = model.metadata_conditioner(
            torch.tensor([0]), torch.tensor([0]), 1, torch.device("cpu")
        )
        second = model.metadata_conditioner(
            torch.tensor([1]), torch.tensor([1]), 1, torch.device("cpu")
        )
        self.assertFalse(torch.equal(first, second))

    def test_section_only_conditioner_ignores_domain(self) -> None:
        model = UNetDenoiser(tiny_section_only_config()).eval()
        conditioner = model.metadata_conditioner
        self.assertIsNotNone(conditioner.section_embedding)
        self.assertIsNone(conditioner.domain_embedding)
        source = conditioner(
            torch.tensor([1]), torch.tensor([0]), 1, torch.device("cpu")
        )
        target = conditioner(
            torch.tensor([1]), torch.tensor([1]), 1, torch.device("cpu")
        )
        torch.testing.assert_close(source, target)

    def test_conditional_model_supports_unknown_metadata(self) -> None:
        model = UNetDenoiser(tiny_conditional_config()).eval()
        inputs = torch.randn(2, 1, 32, 32)
        outputs = model(inputs, torch.tensor([1, 2]))
        self.assertEqual(outputs.shape, inputs.shape)
        with self.assertRaises(ValueError):
            model(
                inputs,
                torch.tensor([1, 2]),
                section_id=torch.tensor([0, 4]),
                domain_id=torch.tensor([0, 1]),
            )

    def test_ddim_reconstruction_is_finite_and_deterministic(self) -> None:
        config = tiny_config()
        diffusion = GaussianDiffusion(config)
        model = UNetDenoiser(config).eval()
        clean = torch.rand(2, 1, 32, 32).mul(2.0).sub(1.0)
        noise = torch.randn_like(clean)
        first = diffusion.ddim_reconstruct(
            model, clean, start_step=10, stride=3, noise=noise
        )
        second = diffusion.ddim_reconstruct(
            model, clean, start_step=10, stride=3, noise=noise
        )
        self.assertEqual(first.shape, clean.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
