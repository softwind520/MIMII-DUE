"""Shared data records used by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioRecord:
    """One MIMII DUE audio file and the metadata encoded in its name."""

    path: Path
    machine_type: str
    section: str
    domain: str
    split: str
    label: int | None


@dataclass(frozen=True)
class PatchMetadata:
    """Metadata required to map a spectrogram patch back to its audio file."""

    audio_path: Path
    patch_index: int
    start_frame: int
    section_id: int
    domain_id: int


@dataclass(frozen=True)
class PatchIndex:
    """A lightweight index entry; the spectrogram is loaded on demand."""

    record_index: int
    patch_index: int
    start_frame: int
