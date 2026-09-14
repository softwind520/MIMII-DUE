"""Tests for patch scoring, audio aggregation, and DCASE metrics."""

import unittest

import torch

from diffusion.metrics import evaluate_audio_scores, harmonic_mean
from diffusion.scoring import (
    aggregate_patch_scores,
    score_reconstruction,
    score_reconstruction_sweep,
)


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

    def test_score_sweep_matches_individual_scoring(self) -> None:
        original = torch.tensor([[[[-1.0, 1.0], [2.0, 3.0]]]])
        reconstructed = torch.zeros_like(original)
        ratios = [0.25, 0.5, 1.0]
        sweep = score_reconstruction_sweep(original, reconstructed, ratios)
        for positive_only in (False, True):
            for ratio in ratios:
                expected = score_reconstruction(
                    original,
                    reconstructed,
                    {
                        "residual": "absolute",
                        "topk_ratio": ratio,
                        "positive_only": positive_only,
                    },
                )
                torch.testing.assert_close(sweep[(positive_only, ratio)], expected)

    def test_audio_level_topk_mean_aggregation(self) -> None:
        result = aggregate_patch_scores(
            [1.0, 4.0, 2.0, 3.0],
            ["a.wav"] * 4,
            method="topk_mean",
            topk_ratio=0.5,
        )
        self.assertEqual(result, {"a.wav": 3.5})

    def test_perfect_auc_and_harmonic_mean(self) -> None:
        result = evaluate_audio_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.1)
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["pauc"], 1.0)
        self.assertAlmostEqual(harmonic_mean([0.5, 1.0]), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
