"""Tests for residual pooling and GMM anomaly likelihoods."""

import unittest

import numpy as np
import torch

from diffusion.gmm_scoring import (
    _aggregate_machine_results,
    fit_gmm_anomaly_scores,
    pool_residual_features,
)


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

    def test_machine_results_are_aggregated_under_one_protocol(self) -> None:
        common = {
            "residual_mode": "signed",
            "scope": "section",
            "covariance_type": "diag",
            "components": 2,
            "reg_covar": 1e-5,
        }

        def result(machine_type, aucs, paucs, iterations):
            groups = [
                {
                    **common,
                    "section": f"section_{index:02d}",
                    "domain": "source",
                    "auc": auc,
                    "pauc": pauc,
                    "normal_files": 100,
                    "anomaly_files": 100,
                }
                for index, (auc, pauc) in enumerate(zip(aucs, paucs))
            ]
            return {
                "machine_type": machine_type,
                "group_rows": groups,
                "summary_rows": [
                    {
                        **common,
                        "gmm_groups": 2,
                        "all_converged": True,
                        "max_iterations": iterations,
                    }
                ],
            }

        summary, groups = _aggregate_machine_results(
            [
                result("fan", [0.8, 0.6], [0.6, 0.5], 10),
                result("pump", [0.9, 0.7], [0.7, 0.6], 14),
            ]
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["machine_count"], 2)
        self.assertEqual(summary[0]["metric_groups"], 4)
        self.assertEqual(summary[0]["gmm_groups"], 4)
        self.assertEqual(summary[0]["max_iterations"], 14)
        self.assertAlmostEqual(summary[0]["auc_mean"], 0.75)
        self.assertAlmostEqual(summary[0]["pauc_mean"], 0.6)
        self.assertEqual({row["machine_type"] for row in groups}, {"fan", "pump"})


if __name__ == "__main__":
    unittest.main()
