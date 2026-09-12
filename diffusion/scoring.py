"""Patch residual scoring and audio-level aggregation."""

from __future__ import annotations


def score_reconstruction(original, reconstructed, config: dict):
    """Return one anomaly score per reconstructed spectrogram patch."""
    raise NotImplementedError("Residual scoring is implemented in stage 4.")


def aggregate_patch_scores(patch_scores, audio_paths, method: str):
    """Aggregate patch scores into one anomaly score for each audio file."""
    raise NotImplementedError("Audio-level aggregation is implemented in stage 4.")

