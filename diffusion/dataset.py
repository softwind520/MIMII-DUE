"""MIMII DUE discovery, log-Mel extraction, patching, and sampling."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Sequence
import hashlib
import json
from math import gcd
import os
from pathlib import Path
import re
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .conditioning import DOMAIN_TO_ID
from .records import AudioRecord, PatchIndex


FILENAME_PATTERN = re.compile(
    r"^(?P<section>section_\d{2})_"
    r"(?P<domain>source|target)_"
    r"(?P<split>train|test)_"
    r"(?P<label>normal|anomaly)_"
    r"(?P<index>\d+)"
    r"(?:_.*)?\.wav$",
    flags=re.IGNORECASE,
)


def parse_audio_record(path: str | Path, machine_type: str, directory_split: str) -> AudioRecord:
    """Parse metadata encoded by the official MIMII DUE naming scheme."""
    audio_path = Path(path).resolve()
    match = FILENAME_PATTERN.match(audio_path.name)
    if match is None:
        raise ValueError(f"Unrecognized MIMII DUE filename: {audio_path.name}")

    values = {key: value.lower() for key, value in match.groupdict().items()}
    expected_split = "train" if directory_split == "train" else "test"
    if values["split"] != expected_split:
        raise ValueError(
            f"Filename split {values['split']!r} does not match directory "
            f"{directory_split!r}: {audio_path.name}"
        )
    if directory_split in {"source_test", "target_test"}:
        expected_domain = directory_split.split("_", maxsplit=1)[0]
        if values["domain"] != expected_domain:
            raise ValueError(
                f"Filename domain {values['domain']!r} does not match directory "
                f"{directory_split!r}: {audio_path.name}"
            )

    return AudioRecord(
        path=audio_path,
        machine_type=machine_type,
        section=values["section"],
        domain=values["domain"],
        split=directory_split,
        label=0 if values["label"] == "normal" else 1,
    )


def discover_audio_records(
    data_root: str | Path,
    machine_types: Sequence[str],
    split: str,
) -> list[AudioRecord]:
    """Discover files for train, source_test, target_test, or both tests."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if split == "test":
        directory_splits = ("source_test", "target_test")
    elif split in {"train", "source_test", "target_test"}:
        directory_splits = (split,)
    else:
        raise ValueError(f"Unsupported split {split!r}")

    records: list[AudioRecord] = []
    for machine_type in machine_types:
        machine_dir = root / machine_type
        if not machine_dir.is_dir():
            raise FileNotFoundError(f"Machine directory does not exist: {machine_dir}")
        for directory_split in directory_splits:
            split_dir = machine_dir / directory_split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")
            for path in sorted(split_dir.glob("*.wav")):
                records.append(parse_audio_record(path, machine_type, directory_split))
    return records


def _pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert integer or floating-point PCM samples to mono float32."""
    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    if np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        limit = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / limit
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2.0
        audio = (audio.astype(np.float32) - midpoint) / midpoint
    else:
        audio = audio.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio)


def read_audio(path: str | Path, target_sample_rate: int) -> torch.Tensor:
    """Read a WAV file, convert it to mono, and resample when required."""
    sample_rate, audio = wavfile.read(Path(path), mmap=False)
    audio = _pcm_to_float32(audio)
    if sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            up=target_sample_rate // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(audio))


def _hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def create_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Create an HTK-style triangular Mel filter bank."""
    min_mel = _hz_to_mel(torch.tensor(0.0))
    max_mel = _hz_to_mel(torch.tensor(sample_rate / 2.0))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)

    lower = hz_points[:-2].unsqueeze(1)
    center = hz_points[1:-1].unsqueeze(1)
    upper = hz_points[2:].unsqueeze(1)
    rising = (frequencies.unsqueeze(0) - lower) / (center - lower).clamp_min(1e-12)
    falling = (upper - frequencies.unsqueeze(0)) / (upper - center).clamp_min(1e-12)
    return torch.maximum(torch.zeros_like(rising), torch.minimum(rising, falling))


class LogMelExtractor:
    """Convert one waveform into a normalized [n_mels, time] log-FBank."""

    def __init__(self, data_config: dict):
        self.sample_rate = int(data_config["sampling_rate"])
        self.n_fft = int(data_config["n_fft"])
        self.win_length = int(data_config["win_length"])
        self.hop_length = int(data_config["hop_length"])
        self.n_mels = int(data_config["n_mels"])
        self.power = float(data_config.get("power", 2.0))
        self.top_db = float(data_config.get("top_db", 80.0))
        self.normalization = data_config.get("normalization", "log_db_minmax")
        self.window = torch.hann_window(self.win_length)
        self.mel_filter = create_mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)

    def __call__(self, path: str | Path) -> torch.Tensor:
        waveform = read_audio(path, self.sample_rate)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitude = spectrum.abs().pow(self.power)
        mel_power = self.mel_filter @ magnitude
        log_mel = 10.0 * torch.log10(mel_power.clamp_min(torch.finfo(torch.float32).eps))

        if self.normalization == "log_db_minmax":
            log_mel = log_mel - log_mel.max()
            log_mel = log_mel.clamp(min=-self.top_db, max=0.0)
            log_mel = (log_mel + self.top_db) / self.top_db
        elif self.normalization != "none":
            raise ValueError(f"Unsupported normalization: {self.normalization!r}")
        return log_mel.to(dtype=torch.float32).contiguous()


def estimate_spectrogram_frames(path: str | Path, target_sample_rate: int, hop_length: int) -> int:
    """Estimate torch.stft(center=True) frame count without decoding the audio."""
    with wave.open(str(path), "rb") as stream:
        source_frames = stream.getnframes()
        source_rate = stream.getframerate()
    target_frames = int(round(source_frames * target_sample_rate / source_rate))
    return 1 + target_frames // hop_length


def make_patch_starts(
    frame_count: int,
    patch_frames: int,
    patch_hop: int,
    cover_tail: bool = False,
) -> list[int]:
    """Return deterministic starts, padding only recordings shorter than a patch."""
    if frame_count <= patch_frames:
        return [0]
    starts = list(range(0, frame_count - patch_frames + 1, patch_hop))
    tail_start = frame_count - patch_frames
    if cover_tail and starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


class MIMIIDUEPatchDataset(Dataset):
    """Map audio files to fixed log-FBank patches and retain file metadata."""

    def __init__(
        self,
        records: Sequence[AudioRecord],
        config: dict,
        training: bool,
        feature_cache: str | Path | None = None,
    ):
        if not records:
            raise ValueError("At least one audio record is required")
        self.records = list(records)
        self.config = config
        self.training = training
        self.data_config = config["data"]
        self.extractor = LogMelExtractor(self.data_config)
        self.patch_frames = int(self.data_config["patch_frames"])
        hop_key = "train_patch_hop" if training else "test_patch_hop"
        self.patch_hop = int(self.data_config[hop_key])
        self.cover_tail = bool(self.data_config.get("cover_tail", False))
        self.section_to_id = {
            section: index
            for index, section in enumerate(sorted({record.section for record in self.records}))
        }
        self.memory_cache_files = int(self.data_config.get("memory_cache_files", 0))
        self._memory_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self.feature_cache = Path(feature_cache).resolve() if feature_cache else None
        signature_keys = (
            "sampling_rate",
            "n_fft",
            "win_length",
            "hop_length",
            "n_mels",
            "power",
            "normalization",
            "top_db",
        )
        signature_values = {key: self.data_config.get(key) for key in signature_keys}
        signature_json = json.dumps(signature_values, sort_keys=True, separators=(",", ":"))
        self.feature_signature = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:12]
        self.index = self._build_index()

    def _build_index(self) -> list[PatchIndex]:
        index: list[PatchIndex] = []
        for record_index, record in enumerate(self.records):
            frame_count = estimate_spectrogram_frames(
                record.path,
                target_sample_rate=self.extractor.sample_rate,
                hop_length=self.extractor.hop_length,
            )
            starts = make_patch_starts(
                frame_count,
                patch_frames=self.patch_frames,
                patch_hop=self.patch_hop,
                cover_tail=self.cover_tail,
            )
            index.extend(
                PatchIndex(record_index, patch_index, start)
                for patch_index, start in enumerate(starts)
            )
        return index

    def _cache_path(self, record: AudioRecord) -> Path | None:
        if self.feature_cache is None:
            return None
        return (
            self.feature_cache
            / self.feature_signature
            / record.machine_type
            / record.split
            / f"{record.path.stem}.npy"
        )

    @staticmethod
    def _read_cached_feature(cache_path: Path) -> torch.Tensor | None:
        if not cache_path.is_file():
            return None
        feature = np.load(cache_path, allow_pickle=False)
        return torch.from_numpy(np.array(feature, dtype=np.float32, copy=True))

    def _write_cached_feature(self, cache_path: Path, feature: torch.Tensor) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dtype_name = self.data_config.get("cache_dtype", "float16")
        dtype = np.float16 if dtype_name == "float16" else np.float32
        # DataLoader workers can request the same uncached file concurrently.
        temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
        with temporary_path.open("wb") as stream:
            np.save(stream, feature.cpu().numpy().astype(dtype, copy=False), allow_pickle=False)
        try:
            temporary_path.replace(cache_path)
        except FileNotFoundError:
            # Another request in this worker may already have promoted it.
            if not cache_path.is_file():
                raise

    def _load_feature(self, record: AudioRecord) -> torch.Tensor:
        cached = self._memory_cache.get(record.path)
        if cached is not None:
            self._memory_cache.move_to_end(record.path)
            return cached

        cache_path = self._cache_path(record)
        feature = self._read_cached_feature(cache_path) if cache_path else None
        if feature is None:
            feature = self.extractor(record.path)
            if cache_path:
                self._write_cached_feature(cache_path, feature)

        if self.memory_cache_files > 0:
            self._memory_cache[record.path] = feature
            self._memory_cache.move_to_end(record.path)
            while len(self._memory_cache) > self.memory_cache_files:
                self._memory_cache.popitem(last=False)
        return feature

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        patch_index = self.index[item]
        record = self.records[patch_index.record_index]
        feature = self._load_feature(record)
        start = patch_index.start_frame
        patch = feature[:, start : start + self.patch_frames]
        if patch.shape[1] < self.patch_frames:
            patch = torch.nn.functional.pad(patch, (0, self.patch_frames - patch.shape[1]))
        return {
            "patch": patch.unsqueeze(0).contiguous(),
            "label": -1 if record.label is None else record.label,
            "section_id": self.section_to_id[record.section],
            "domain_id": DOMAIN_TO_ID[record.domain],
            "machine_type": record.machine_type,
            "section": record.section,
            "domain": record.domain,
            "audio_path": str(record.path),
            "patch_index": patch_index.patch_index,
            "start_frame": start,
        }


def build_domain_balanced_sampler(dataset: MIMIIDUEPatchDataset) -> WeightedRandomSampler:
    """Sample source and target patches with equal expected domain mass."""
    domain_ids = [DOMAIN_TO_ID[dataset.records[item.record_index].domain] for item in dataset.index]
    counts = Counter(domain_ids)
    weights = torch.tensor([1.0 / counts[domain_id] for domain_id in domain_ids], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def count_records(records: Sequence[AudioRecord]) -> Counter:
    """Count records by machine, split, domain, and label."""
    return Counter(
        (record.machine_type, record.split, record.domain, record.label)
        for record in records
    )
