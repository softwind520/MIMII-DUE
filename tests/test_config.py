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
        self.assertFalse(config["conditioning"]["use_section"])
        self.assertFalse(config["conditioning"]["use_domain"])

    def test_conditional_experiment_uses_separate_checkpoints(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        baseline = load_config(project_root / "diffusion.yaml")
        conditional = load_config(project_root / "conditional.yaml")
        self.assertTrue(conditional["conditioning"]["use_section"])
        self.assertTrue(conditional["conditioning"]["use_domain"])
        self.assertEqual(conditional["conditioning"]["num_sections"], 3)
        self.assertNotEqual(
            baseline["project"]["checkpoint_directory"],
            conditional["project"]["checkpoint_directory"],
        )

    def test_section_only_experiment_is_isolated(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        baseline = load_config(project_root / "diffusion.yaml")
        conditional = load_config(project_root / "conditional.yaml")
        section_only = load_config(project_root / "section_only.yaml")

        self.assertTrue(section_only["conditioning"]["use_section"])
        self.assertFalse(section_only["conditioning"]["use_domain"])
        self.assertTrue(section_only["conditioning"]["domain_balance"])
        self.assertEqual(section_only["conditioning"]["num_sections"], 3)
        checkpoint_directories = {
            baseline["project"]["checkpoint_directory"],
            conditional["project"]["checkpoint_directory"],
            section_only["project"]["checkpoint_directory"],
        }
        output_directories = {
            baseline["project"]["output_directory"],
            conditional["project"]["output_directory"],
            section_only["project"]["output_directory"],
        }
        self.assertEqual(len(checkpoint_directories), 3)
        self.assertEqual(len(output_directories), 3)


if __name__ == "__main__":
    unittest.main()
