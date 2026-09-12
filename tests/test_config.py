"""Smoke tests for the diffusion project scaffold."""

from pathlib import Path
import unittest

from diffusion.config import REQUIRED_SECTIONS, load_config


class ConfigTest(unittest.TestCase):
    def test_default_config_has_required_sections(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "diffusion.yaml")
        self.assertTrue(REQUIRED_SECTIONS.issubset(config))
        self.assertEqual(config["data"]["n_mels"], 128)
        self.assertEqual(config["data"]["patch_frames"], 128)


if __name__ == "__main__":
    unittest.main()
