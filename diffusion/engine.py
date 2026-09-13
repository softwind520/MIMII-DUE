"""Training, checkpoint, EMA, and inference orchestration."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import resolve_config_path
from .dataset import (
    MIMIIDUEPatchDataset,
    build_domain_balanced_sampler,
    discover_audio_records,
)
from .diffusion import GaussianDiffusion
from .metrics import evaluate_audio_scores, harmonic_mean
from .records import AudioRecord
from .scoring import aggregate_patch_scores, score_reconstruction
from .unet import UNetDenoiser


class ExponentialMovingAverage:
    """Maintain a non-trainable moving-average copy for later reconstruction."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1")
        self.decay = decay
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_parameters = dict(model.named_parameters())
        for name, averaged in self.model.named_parameters():
            averaged.lerp_(source_parameters[name], 1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, averaged in self.model.named_buffers():
            averaged.copy_(source_buffers[name])

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(config: dict) -> torch.device:
    requested = str(config["project"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    return device


def _checkpoint_payload(
    machine_type: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
) -> dict:
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "format_version": 1,
        "machine_type": machine_type,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": public_config,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _ema_checkpoint_payload(
    machine_type: str,
    epoch: int,
    global_step: int,
    ema: ExponentialMovingAverage,
    config: dict,
) -> dict:
    """Create the compact checkpoint needed by reconstruction only."""
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "format_version": 1,
        "machine_type": machine_type,
        "epoch": epoch,
        "global_step": global_step,
        "ema": ema.state_dict(),
        "config": public_config,
    }


def _load_checkpoint(
    path: Path,
    machine_type: str,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("machine_type") != machine_type:
        raise ValueError(
            f"Checkpoint belongs to {checkpoint.get('machine_type')!r}, not {machine_type!r}"
        )
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def _make_loader(config: dict, machine_type: str, device: torch.device) -> DataLoader:
    data_config = config["data"]
    data_root = resolve_config_path(config, data_config["root"])
    records = discover_audio_records(data_root, [machine_type], "train")
    feature_cache = resolve_config_path(config, data_config["feature_cache"])
    dataset = MIMIIDUEPatchDataset(
        records,
        config,
        training=True,
        feature_cache=feature_cache,
    )
    sampler = build_domain_balanced_sampler(dataset)
    workers = int(config["training"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )


def _train_machine(
    config: dict,
    machine_type: str,
    device: torch.device,
    max_steps: int | None,
    resume: bool,
) -> None:
    training_config = config["training"]
    loader = _make_loader(config, machine_type, device)
    model = UNetDenoiser(config).to(device)
    diffusion = GaussianDiffusion(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    use_amp = bool(training_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = ExponentialMovingAverage(model, float(training_config["ema_decay"]))

    checkpoint_root = resolve_config_path(config, config["project"]["checkpoint_directory"])
    machine_checkpoint_dir = checkpoint_root / machine_type
    last_checkpoint = machine_checkpoint_dir / "last.pt"
    start_epoch = 0
    global_step = 0
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Resume requested but checkpoint does not exist: {last_checkpoint}")
        start_epoch, global_step = _load_checkpoint(
            last_checkpoint, machine_type, model, ema, optimizer, scaler, device
        )
        print(f"resume: machine={machine_type} epoch={start_epoch} step={global_step}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    print(
        f"train_start: machine={machine_type} device={device} "
        f"patches={len(loader.dataset)} batches={len(loader)} parameters={parameter_count:,} "
        f"batch={loader.batch_size} accumulation={accumulation_steps} "
        f"effective_batch={loader.batch_size * accumulation_steps} amp={use_amp}"
    )

    epochs = int(training_config["epochs"])
    log_interval = int(training_config.get("log_interval", 50))
    gradient_clip = float(training_config.get("gradient_clip_norm", 0.0))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    stop_training = False

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            clean = batch["patch"].to(device, non_blocking=True)
            clean = clean.mul(2.0).sub(1.0)
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(loader) - group_start)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = diffusion.training_loss(model, clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {machine_type} at epoch={epoch + 1}, step={global_step + 1}"
                )

            scaler.scale(loss / group_size).backward()
            epoch_batches += 1
            epoch_loss += float(loss.detach())
            should_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_update:
                if gradient_clip > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

                global_step += 1
                if global_step == 1 or global_step % log_interval == 0:
                    print(
                        f"train_step: machine={machine_type} epoch={epoch + 1}/{epochs} "
                        f"step={global_step} loss={float(loss.detach()):.6f}"
                    )
                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break

        if epoch_batches == 0:
            raise RuntimeError("Training DataLoader yielded no batches; reduce training.batch_size")
        elapsed = time.perf_counter() - epoch_start
        print(
            f"epoch_end: machine={machine_type} epoch={epoch + 1}/{epochs} "
            f"loss={epoch_loss / epoch_batches:.6f} seconds={elapsed:.1f}"
        )

        payload = _checkpoint_payload(
            machine_type, epoch, global_step, model, ema, optimizer, scaler, config
        )
        _save_checkpoint(last_checkpoint, payload)
        _save_checkpoint(
            machine_checkpoint_dir / "ema.pt",
            _ema_checkpoint_payload(machine_type, epoch, global_step, ema, config),
        )
        print(f"checkpoint: {last_checkpoint}")
        if stop_training:
            break

    print(f"train_complete: machine={machine_type} steps={global_step}")


def train(
    config: dict,
    machine_types: list[str] | None = None,
    max_steps: int | None = None,
    resume: bool = False,
) -> None:
    """Train one unconditional DDPM per configured machine type."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    configured_machines = list(config["data"]["machine_types"])
    machines = machine_types or configured_machines
    unknown = sorted(set(machines).difference(configured_machines))
    if unknown:
        raise ValueError(f"Unknown machine types: {', '.join(unknown)}")

    seed = int(config["project"]["seed"])
    _set_seed(seed)
    device = _resolve_device(config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")
    for machine_type in machines:
        _train_machine(config, machine_type, device, max_steps, resume)


def _checkpoint_for_evaluation(config: dict, machine_type: str) -> Path:
    checkpoint_root = resolve_config_path(config, config["project"]["checkpoint_directory"])
    machine_dir = checkpoint_root / machine_type
    compact = machine_dir / "ema.pt"
    full = machine_dir / "last.pt"
    if compact.is_file():
        return compact
    if full.is_file():
        return full
    raise FileNotFoundError(f"No EMA or training checkpoint found under {machine_dir}")


def _load_evaluation_model(
    config: dict,
    machine_type: str,
    device: torch.device,
) -> tuple[UNetDenoiser, Path, dict]:
    checkpoint_path = _checkpoint_for_evaluation(config, machine_type)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("machine_type") != machine_type:
        raise ValueError(
            f"Checkpoint belongs to {checkpoint.get('machine_type')!r}, not {machine_type!r}"
        )
    if "ema" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain EMA weights: {checkpoint_path}")
    model = UNetDenoiser(config).to(device).eval()
    model.load_state_dict(checkpoint["ema"])
    return model, checkpoint_path, checkpoint


def _select_evaluation_records(
    records: list[AudioRecord],
    files_per_group: int | None,
) -> list[AudioRecord]:
    if files_per_group is None:
        return records
    if files_per_group < 1:
        raise ValueError("max_files_per_group must be positive")
    grouped: dict[tuple[str, str, int | None], list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.domain, record.label)].append(record)
    selected: list[AudioRecord] = []
    for key in sorted(grouped):
        selected.extend(grouped[key][:files_per_group])
    return sorted(selected, key=lambda record: str(record.path))


def _evaluation_loader(
    config: dict,
    records: list[AudioRecord],
    device: torch.device,
) -> DataLoader:
    data_config = config["data"]
    feature_cache = resolve_config_path(config, data_config["feature_cache"])
    dataset = MIMIIDUEPatchDataset(
        records,
        config,
        training=False,
        feature_cache=feature_cache,
    )
    workers = int(config["evaluation"].get("num_workers", 1))
    return DataLoader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 32)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _write_machine_results(
    output_root: Path,
    machine_type: str,
    records: list[AudioRecord],
    audio_scores: dict[str, float],
    max_fpr: float,
) -> list[dict]:
    machine_output = output_root / machine_type
    machine_output.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.domain)].append(record)

    metric_rows: list[dict] = []
    for (section, domain), group_records in sorted(grouped.items()):
        score_path = (
            machine_output
            / f"anomaly_score_{machine_type}_{section}_{domain}_test.csv"
        )
        with score_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for record in sorted(group_records, key=lambda item: item.path.name):
                writer.writerow([record.path.name, f"{audio_scores[str(record.path)]:.10f}"])

        labels = [int(record.label) for record in group_records if record.label is not None]
        scores = [
            audio_scores[str(record.path)]
            for record in group_records
            if record.label is not None
        ]
        result = evaluate_audio_scores(labels, scores, max_fpr)
        row = {
            "machine_type": machine_type,
            "section": section,
            "domain": domain,
            "auc": result["auc"],
            "pauc": result["pauc"],
            "normal_files": labels.count(0),
            "anomaly_files": labels.count(1),
        }
        metric_rows.append(row)
        print(
            f"metric: machine={machine_type} section={section} domain={domain} "
            f"auc={result['auc']:.6f} pauc={result['pauc']:.6f}"
        )
    return metric_rows


def _write_metric_summary(output_root: Path, metric_rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "machine_type",
        "section",
        "domain",
        "auc",
        "pauc",
        "normal_files",
        "anomaly_files",
    ]
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    auc_values = [float(row["auc"]) for row in metric_rows]
    pauc_values = [float(row["pauc"]) for row in metric_rows]
    summary = {
        "groups": len(metric_rows),
        "auc_arithmetic_mean": float(np.mean(auc_values)),
        "auc_harmonic_mean": harmonic_mean(auc_values),
        "pauc_arithmetic_mean": float(np.mean(pauc_values)),
        "pauc_harmonic_mean": harmonic_mean(pauc_values),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(
        "summary: "
        f"auc_mean={summary['auc_arithmetic_mean']:.6f} "
        f"auc_hmean={summary['auc_harmonic_mean']:.6f} "
        f"pauc_mean={summary['pauc_arithmetic_mean']:.6f} "
        f"pauc_hmean={summary['pauc_harmonic_mean']:.6f}"
    )
    print(f"results: {output_root}")


def evaluate(
    config: dict,
    machine_types: list[str] | None = None,
    max_files_per_group: int | None = None,
) -> None:
    """Run partial-DDIM reconstruction and DCASE metrics for trained machines."""
    seed = int(config["project"]["seed"])
    _set_seed(seed)
    device = _resolve_device(config)
    configured_machines = list(config["data"]["machine_types"])
    if machine_types is None:
        machines = [
            machine
            for machine in configured_machines
            if (
                resolve_config_path(config, config["project"]["checkpoint_directory"])
                / machine
                / "ema.pt"
            ).is_file()
            or (
                resolve_config_path(config, config["project"]["checkpoint_directory"])
                / machine
                / "last.pt"
            ).is_file()
        ]
    else:
        machines = machine_types
    unknown = sorted(set(machines).difference(configured_machines))
    if unknown:
        raise ValueError(f"Unknown machine types: {', '.join(unknown)}")
    if not machines:
        raise FileNotFoundError("No trained machine checkpoints were found")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")
    evaluation_config = config["evaluation"]
    reconstruction_samples = int(evaluation_config.get("reconstruction_samples", 1))
    if reconstruction_samples < 1:
        raise ValueError("reconstruction_samples must be positive")
    log_interval = int(evaluation_config.get("log_interval_batches", 50))
    output_root = resolve_config_path(config, config["project"]["output_directory"])
    all_metric_rows: list[dict] = []

    for machine_type in machines:
        model, checkpoint_path, checkpoint = _load_evaluation_model(
            config, machine_type, device
        )
        diffusion = GaussianDiffusion(config).to(device)
        data_root = resolve_config_path(config, config["data"]["root"])
        records = discover_audio_records(data_root, [machine_type], "test")
        records = _select_evaluation_records(records, max_files_per_group)
        loader = _evaluation_loader(config, records, device)
        patch_scores_by_audio: dict[str, list[float]] = defaultdict(list)
        use_amp = bool(evaluation_config.get("mixed_precision", True)) and device.type == "cuda"
        print(
            f"evaluate_start: machine={machine_type} device={device} "
            f"checkpoint={checkpoint_path} epoch={int(checkpoint['epoch']) + 1} "
            f"files={len(records)} patches={len(loader.dataset)} batches={len(loader)} "
            f"ddim_start={config['diffusion']['inference_start_step']} "
            f"ddim_stride={config['diffusion']['ddim_stride']} samples={reconstruction_samples}"
        )
        start_time = time.perf_counter()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                clean = batch["patch"].to(device, non_blocking=True).mul(2.0).sub(1.0)
                reconstructed = torch.zeros_like(clean)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    for _ in range(reconstruction_samples):
                        reconstructed.add_(diffusion.ddim_reconstruct(model, clean))
                reconstructed.div_(reconstruction_samples)
                scores = score_reconstruction(clean, reconstructed, config)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError(
                        f"Non-finite anomaly score in evaluation batch {batch_index}"
                    )
                for path, score in zip(batch["audio_path"], scores.cpu().tolist()):
                    patch_scores_by_audio[str(path)].append(float(score))
                if batch_index == 1 or batch_index % log_interval == 0:
                    print(
                        f"evaluate_batch: machine={machine_type} "
                        f"batch={batch_index}/{len(loader)}"
                    )

        aggregation = config["scoring"].get("patch_aggregation", "mean")
        audio_scores: dict[str, float] = {}
        for path, scores in patch_scores_by_audio.items():
            audio_scores.update(
                aggregate_patch_scores(scores, [path] * len(scores), aggregation)
            )
        if len(audio_scores) != len(records):
            raise RuntimeError(
                f"Expected {len(records)} audio scores, produced {len(audio_scores)}"
            )
        elapsed = time.perf_counter() - start_time
        print(
            f"evaluate_complete: machine={machine_type} files={len(audio_scores)} "
            f"seconds={elapsed:.1f}"
        )
        all_metric_rows.extend(
            _write_machine_results(
                output_root,
                machine_type,
                records,
                audio_scores,
                float(evaluation_config["max_fpr"]),
            )
        )
        del model, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_metric_summary(output_root, all_metric_rows)
