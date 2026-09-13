"""DCASE-compatible AUC and partial-AUC evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_audio_scores(
    labels: Sequence[int],
    scores: Sequence[float],
    max_fpr: float,
) -> dict[str, float]:
    """Compute AUC and standardized partial AUC for one evaluation group."""
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape != score_array.shape or label_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if not np.isfinite(score_array).all():
        raise ValueError("Anomaly scores contain NaN or infinity")
    if set(np.unique(label_array).tolist()) != {0, 1}:
        raise ValueError("AUC requires at least one normal and one anomalous file")
    return {
        "auc": float(roc_auc_score(label_array, score_array)),
        "pauc": float(roc_auc_score(label_array, score_array, max_fpr=max_fpr)),
    }


def harmonic_mean(values: Sequence[float]) -> float:
    """Return a stable harmonic mean for non-negative DCASE metrics."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("At least one value is required")
    if (array < 0.0).any():
        raise ValueError("Harmonic mean values must be non-negative")
    if (array == 0.0).any():
        return 0.0
    return float(array.size / np.reciprocal(array).sum())
