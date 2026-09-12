"""DCASE-compatible AUC and partial-AUC evaluation."""

from __future__ import annotations


def evaluate_audio_scores(labels, scores, max_fpr: float) -> dict[str, float]:
    """Compute AUC and standardized partial AUC for one evaluation group."""
    raise NotImplementedError("Evaluation metrics are implemented in stage 4.")

