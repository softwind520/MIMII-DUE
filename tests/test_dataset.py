"""Unit tests for MIMII DUE indexing and log-FBank patch extraction."""

from pathlib import Path
import unittest

import torch

from diffusion.config import load_config, resolve_config_path
from diffusion.dataset import (
    MIMIIDUEPatchDataset,
    build_domain_balanced_sampler,
    discover_audio_records,
    make_patch_starts,
    parse_audio_record,
)


class FilenameParsingTest(unittest.TestCase):
    def test_train_filename_with_attributes(self) -> None:
        record = parse_audio_record(
            "section_02_target_train_normal_0001_strength_2_ambient.wav",
            machine_type="fan",
            directory_split="train",
        )
        self.assertEqual(record.section, "section_02")
        self.assertEqual(record.domain, "target")
        self.assertEqual(record.label, 0)

    def test_test_filename_with_anomaly_label(self) -> None:
        record = parse_audio_record(
            "section_01_source_test_anomaly_0042.wav",
            machine_type="pump",
            directory_split="source_test",
        )
        self.assertEqual(record.section, "section_01")
        self.assertEqual(record.domain, "source")
        self.assertEqual(record.label, 1)


class PatchingTest(unittest.TestCase):
    def test_asd_diffusion_training_stride(self) -> None:
        starts = make_patch_starts(1001, patch_frames=128, patch_hop=128)
        self.assertEqual(starts, [0, 128, 256, 384, 512, 640, 768])

    def test_asd_diffusion_test_stride(self) -> None:
        starts = make_patch_starts(1001, patch_frames=128, patch_hop=5)
        self.assertEqual(len(starts), 175)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 870)

    def test_real_audio_produces_normalized_patch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "diffusion.yaml")
        config["data"]["memory_cache_files"] = 0
        data_root = resolve_config_path(config, config["data"]["root"])
        if not data_root.is_dir():
            self.skipTest(f"MIMII DUE data not available: {data_root}")

        record = discover_audio_records(data_root, ["fan"], "train")[0]
        dataset = MIMIIDUEPatchDataset([record], config, training=True)
        item = dataset[0]

        self.assertEqual(tuple(item["patch"].shape), (1, 128, 128))
        self.assertTrue(torch.isfinite(item["patch"]).all())
        self.assertGreaterEqual(float(item["patch"].min()), 0.0)
        self.assertLessEqual(float(item["patch"].max()), 1.0)
        self.assertEqual(item["domain_id"], 0)

    def test_domain_balanced_sampler_has_equal_expected_mass(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "diffusion.yaml")
        data_root = resolve_config_path(config, config["data"]["root"])
        if not data_root.is_dir():
            self.skipTest(f"MIMII DUE data not available: {data_root}")

        records = discover_audio_records(data_root, ["fan"], "train")
        source_record = next(record for record in records if record.domain == "source")
        target_record = next(record for record in records if record.domain == "target")
        dataset = MIMIIDUEPatchDataset([source_record, target_record], config, training=True)
        sampler = build_domain_balanced_sampler(dataset)

        domain_mass = {"source": 0.0, "target": 0.0}
        for patch_index, weight in zip(dataset.index, sampler.weights.tolist()):
            domain = dataset.records[patch_index.record_index].domain
            domain_mass[domain] += weight
        self.assertAlmostEqual(domain_mass["source"], domain_mass["target"])


if __name__ == "__main__":
    unittest.main()
