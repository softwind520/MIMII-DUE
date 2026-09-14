"""Patch residual scoring and audio-level aggregation."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch


def score_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Return one anomaly score per reconstructed spectrogram patch."""
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Reconstruction shape {tuple(reconstructed.shape)} does not match "
            f"input shape {tuple(original.shape)}"
        )
    scoring_config = config.get("scoring", config)
    residual_name = scoring_config.get("residual", "absolute")
    difference = original - reconstructed
    if bool(scoring_config.get("positive_only", False)):
        difference = difference.clamp_min(0.0)
    if residual_name == "absolute":
        residual = difference.abs()
    elif residual_name == "squared":
        residual = difference.square()
    else:
        raise ValueError(f"Unsupported residual score: {residual_name!r}")

    ratio = float(scoring_config.get("topk_ratio", 1.0))
    if not 0.0 < ratio <= 1.0:
        raise ValueError("topk_ratio must satisfy 0 < topk_ratio <= 1")
    flattened = residual.flatten(start_dim=1)
    count = max(1, math.ceil(flattened.shape[1] * ratio))
    return flattened.topk(count, dim=1, largest=True, sorted=False).values.mean(dim=1)


def score_reconstruction_sweep(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    topk_ratios: list[float],
    positive_only_modes: tuple[bool, ...] = (False, True),
) -> dict[tuple[bool, float], torch.Tensor]:
    """Calculate many AF variants while sorting each residual map only once.

    ``positive_only=False`` implements TopK(|x - x_hat|), while ``True``
    implements the ReLU(x - x_hat) variant from ASD-Diffusion.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Reconstruction shape {tuple(reconstructed.shape)} does not match "
            f"input shape {tuple(original.shape)}"
        )
    if not topk_ratios:
        raise ValueError("At least one topk ratio is required")
    ratios = [float(ratio) for ratio in topk_ratios]
    if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("All topk ratios must satisfy 0 < ratio <= 1")

    difference = (original - reconstructed).flatten(start_dim=1)
    results: dict[tuple[bool, float], torch.Tensor] = {}
    for positive_only in positive_only_modes:
        residual = difference.clamp_min(0.0) if positive_only else difference.abs()
        descending = residual.sort(dim=1, descending=True).values
        cumulative = descending.cumsum(dim=1)
        for ratio in ratios:
            count = max(1, math.ceil(descending.shape[1] * ratio))
            results[(positive_only, ratio)] = cumulative[:, count - 1] / count
    return results


def aggregate_patch_scores(
    patch_scores: torch.Tensor | list[float],
    audio_paths: list[str],
    method: str,
    topk_ratio: float = 0.1,
) -> dict[str, float]:
    """Aggregate patch scores into one anomaly score for each audio file."""
    values = (
        patch_scores.detach().cpu().tolist()
        if isinstance(patch_scores, torch.Tensor)
        else list(patch_scores)
    )
    if len(values) != len(audio_paths):
        raise ValueError("patch_scores and audio_paths must have equal length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for path, score in zip(audio_paths, values):
        grouped[str(path)].append(float(score))

    if method == "topk_mean":
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("Patch topk_ratio must satisfy 0 < topk_ratio <= 1")

        def reduce_topk(scores: list[float]) -> float:
            count = max(1, math.ceil(len(scores) * topk_ratio))
            return float(np.mean(np.partition(scores, len(scores) - count)[-count:]))

        return {path: reduce_topk(scores) for path, scores in grouped.items()}

    reducers = {"mean": np.mean, "max": np.max, "median": np.median}
    if method not in reducers:
        raise ValueError(f"Unsupported patch aggregation: {method!r}")
    reducer = reducers[method]
    return {path: float(reducer(scores)) for path, scores in grouped.items()}
