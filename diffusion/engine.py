"""Training, checkpoint, EMA, and inference orchestration."""

from __future__ import annotations

from copy import deepcopy
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
    checkpoint_interval = int(training_config.get("checkpoint_every_epochs", 10))
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
        if (epoch + 1) % checkpoint_interval == 0 and not stop_training:
            _save_checkpoint(machine_checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", payload)
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


def evaluate(config: dict) -> None:
    """Reconstruct test audio and write DCASE-compatible result files."""
    del config
    raise NotImplementedError("The evaluation engine is implemented in stage 4.")
