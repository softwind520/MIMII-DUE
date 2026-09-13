"""Tests for patch scoring, audio aggregation, and DCASE metrics."""

import unittest

import torch

from diffusion.metrics import evaluate_audio_scores, harmonic_mean
from diffusion.scoring import aggregate_patch_scores, score_reconstruction


class ScoringTest(unittest.TestCase):
    def test_topk_absolute_residual(self) -> None:
        original = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        reconstructed = torch.zeros_like(original)
        scores = score_reconstruction(
            original,
            reconstructed,
            {"residual": "absolute", "topk_ratio": 0.5, "positive_only": False},
        )
        self.assertAlmostEqual(float(scores[0]), 2.5)

    def test_audio_level_mean_aggregation(self) -> None:
        result = aggregate_patch_scores(
            [1.0, 3.0, 5.0],
            ["a.wav", "a.wav", "b.wav"],
            method="mean",
        )
        self.assertEqual(result, {"a.wav": 2.0, "b.wav": 5.0})

    def test_perfect_auc_and_harmonic_mean(self) -> None:
        result = evaluate_audio_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.1)
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["pauc"], 1.0)
        self.assertAlmostEqual(harmonic_mean([0.5, 1.0]), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
