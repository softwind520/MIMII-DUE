"""Tests for residual pooling and GMM anomaly likelihoods."""

import unittest

import numpy as np
import torch

from diffusion.gmm_scoring import fit_gmm_anomaly_scores, pool_residual_features


class GMMScoringTest(unittest.TestCase):
    def test_pool_residual_features(self) -> None:
        original = torch.tensor([[[[-1.0, 1.0], [2.0, 4.0]]]])
        reconstructed = torch.zeros_like(original)
        result = pool_residual_features(original, reconstructed)
        torch.testing.assert_close(result["signed"], torch.tensor([[0.0, 3.0]]))
        torch.testing.assert_close(result["absolute"], torch.tensor([[1.0, 3.0]]))
        torch.testing.assert_close(result["relu"], torch.tensor([[0.5, 3.0]]))

    def test_gmm_scores_far_sample_as_more_anomalous(self) -> None:
        rng = np.random.default_rng(42)
        train = np.concatenate(
            [
                rng.normal(-1.0, 0.1, size=(100, 2)),
                rng.normal(1.0, 0.1, size=(100, 2)),
            ]
        )
        test = np.asarray([[1.0, 1.0], [6.0, 6.0]])
        train_sections = np.asarray(["section_00"] * len(train))
        test_sections = np.asarray(["section_00"] * len(test))
        scores, diagnostics = fit_gmm_anomaly_scores(
            train,
            test,
            train_sections,
            test_sections,
            scope="section",
            covariance_type="full",
            components=2,
            reg_covar=1e-5,
            seed=42,
        )
        self.assertGreater(scores[1], scores[0])
        self.assertTrue(diagnostics["all_converged"])


if __name__ == "__main__":
    unittest.main()

