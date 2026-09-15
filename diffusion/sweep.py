"""Fan-only DDIM and anomaly-filter ablation without retraining."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import resolve_config_path
from .dataset import discover_audio_records
from .diffusion import GaussianDiffusion
from .engine import (
    _evaluation_loader,
    _load_evaluation_model,
    _resolve_device,
    _select_evaluation_records,
    _set_seed,
)
from .metrics import evaluate_audio_scores, harmonic_mean
from .records import AudioRecord
from .scoring import aggregate_patch_scores, score_reconstruction_sweep


@dataclass(frozen=True)
class SweepSpec:
    start_step: int
    positive_only: bool
    topk_ratio: float
    aggregation: str

    @property
    def af_mode(self) -> str:
        return "relu" if self.positive_only else "absolute"


def _validate_sweep(
    config: dict,
    start_steps: list[int],
    topk_ratios: list[float],
    aggregations: list[str],
    patch_topk_ratio: float,
) -> None:
    training_steps = int(config["diffusion"]["training_steps"])
    if not start_steps:
        raise ValueError("At least one diffusion start step is required")
    if any(step < 0 or step >= training_steps for step in start_steps):
        raise ValueError(f"Sweep start steps must be in [0, {training_steps - 1}]")
    if not topk_ratios or any(not 0.0 < ratio <= 1.0 for ratio in topk_ratios):
        raise ValueError("Sweep TopK ratios must satisfy 0 < ratio <= 1")
    supported = {"mean", "max", "median", "topk_mean"}
    unknown = sorted(set(aggregations).difference(supported))
    if not aggregations or unknown:
        raise ValueError(f"Unsupported patch aggregations: {', '.join(unknown)}")
    if not 0.0 < patch_topk_ratio <= 1.0:
        raise ValueError("Patch TopK ratio must satisfy 0 < ratio <= 1")


def _group_metrics(
    records: list[AudioRecord],
    audio_scores: dict[str, float],
    max_fpr: float,
) -> list[dict]:
    grouped: dict[tuple[str, str], list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.domain)].append(record)
    rows: list[dict] = []
    for (section, domain), group_records in sorted(grouped.items()):
        labelled = [record for record in group_records if record.label is not None]
        labels = [int(record.label) for record in labelled]
        scores = [audio_scores[str(record.path)] for record in labelled]
        result = evaluate_audio_scores(labels, scores, max_fpr)
        rows.append(
            {
                "section": section,
                "domain": domain,
                "auc": result["auc"],
                "pauc": result["pauc"],
                "normal_files": labels.count(0),
                "anomaly_files": labels.count(1),
            }
        )
    return rows


def _summarize(rows: list[dict]) -> dict:
    auc_values = [float(row["auc"]) for row in rows]
    pauc_values = [float(row["pauc"]) for row in rows]
    all_values = auc_values + pauc_values
    return {
        "auc_mean": float(np.mean(auc_values)),
        "auc_hmean": harmonic_mean(auc_values),
        "pauc_mean": float(np.mean(pauc_values)),
        "pauc_hmean": harmonic_mean(pauc_values),
        "overall_hmean": harmonic_mean(all_values),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty sweep table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fan_sweep(
    config: dict,
    start_steps: list[int],
    topk_ratios: list[float],
    aggregations: list[str],
    patch_topk_ratio: float = 0.1,
    max_files_per_group: int | None = None,
) -> None:
    """Evaluate many fan scoring settings with one reconstruction per start step."""
    start_steps = list(dict.fromkeys(int(step) for step in start_steps))
    topk_ratios = list(dict.fromkeys(float(ratio) for ratio in topk_ratios))
    aggregations = list(dict.fromkeys(aggregations))
    _validate_sweep(
        config, start_steps, topk_ratios, aggregations, patch_topk_ratio
    )

    machine_type = "fan"
    seed = int(config["project"]["seed"])
    _set_seed(seed)
    device = _resolve_device(config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")

    model, checkpoint_path, checkpoint = _load_evaluation_model(
        config, machine_type, device
    )
    data_root = resolve_config_path(config, config["data"]["root"])
    records = discover_audio_records(data_root, [machine_type], "test")
    records = _select_evaluation_records(records, max_files_per_group)
    loader = _evaluation_loader(config, records, device)
    evaluation_config = config["evaluation"]
    reconstruction_samples = int(evaluation_config.get("reconstruction_samples", 1))
    if reconstruction_samples < 1:
        raise ValueError("reconstruction_samples must be positive")
    log_interval = int(evaluation_config.get("log_interval_batches", 50))
    use_amp = bool(evaluation_config.get("mixed_precision", True)) and device.type == "cuda"
    max_fpr = float(evaluation_config["max_fpr"])

    output_root = (
        resolve_config_path(config, config["project"]["output_directory"])
        / "fan_sweep"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "machine_type": machine_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "seed": seed,
        "files": len(records),
        "patches": len(loader.dataset),
        "test_patch_hop": int(config["data"]["test_patch_hop"]),
        "ddim_stride": int(config["diffusion"]["ddim_stride"]),
        "reconstruction_samples": reconstruction_samples,
        "start_steps": start_steps,
        "pixel_topk_ratios": topk_ratios,
        "patch_aggregations": aggregations,
        "patch_topk_ratio": patch_topk_ratio,
    }
    with (output_root / "fan_sweep_run.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
    summary_rows: list[dict] = []
    group_rows: list[dict] = []

    print(
        f"fan_sweep_start: checkpoint={checkpoint_path} "
        f"epoch={int(checkpoint['epoch']) + 1} files={len(records)} "
        f"patches={len(loader.dataset)} batches={len(loader)} "
        f"start_steps={start_steps} topk_ratios={topk_ratios} "
        f"aggregations={aggregations}"
    )
    sweep_start_time = time.perf_counter()

    for start_index, start_step in enumerate(start_steps, start=1):
        # Reusing the same seed makes the forward noise comparable across start steps.
        _set_seed(seed)
        diffusion = GaussianDiffusion(config).to(device)
        patch_scores: dict[
            tuple[bool, float], dict[str, list[float]]
        ] = {
            (positive_only, ratio): defaultdict(list)
            for positive_only in (False, True)
            for ratio in topk_ratios
        }
        step_start_time = time.perf_counter()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                clean = batch["patch"].to(device, non_blocking=True).mul(2.0).sub(1.0)
                section_id = batch["section_id"].to(device, non_blocking=True)
                domain_id = batch["domain_id"].to(device, non_blocking=True)
                reconstructed = torch.zeros_like(clean)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    for _ in range(reconstruction_samples):
                        reconstructed.add_(
                            diffusion.ddim_reconstruct(
                                model,
                                clean,
                                start_step=start_step,
                                section_id=section_id,
                                domain_id=domain_id,
                            )
                        )
                reconstructed.div_(reconstruction_samples)
                batch_scores = score_reconstruction_sweep(
                    clean, reconstructed, topk_ratios
                )
                paths = [str(path) for path in batch["audio_path"]]
                for key, values in batch_scores.items():
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(
                            f"Non-finite score at start step {start_step}, "
                            f"batch {batch_index}, AF setting {key}"
                        )
                    for path, value in zip(paths, values.cpu().tolist()):
                        patch_scores[key][path].append(float(value))
                if batch_index == 1 or batch_index % log_interval == 0:
                    print(
                        f"fan_sweep_batch: start={start_step} "
                        f"batch={batch_index}/{len(loader)}"
                    )

        for positive_only in (False, True):
            for ratio in topk_ratios:
                scores_by_audio = patch_scores[(positive_only, ratio)]
                for aggregation in aggregations:
                    audio_scores: dict[str, float] = {}
                    for path, scores in scores_by_audio.items():
                        audio_scores.update(
                            aggregate_patch_scores(
                                scores,
                                [path] * len(scores),
                                aggregation,
                                topk_ratio=patch_topk_ratio,
                            )
                        )
                    if len(audio_scores) != len(records):
                        raise RuntimeError(
                            f"Expected {len(records)} audio scores, "
                            f"produced {len(audio_scores)}"
                        )
                    spec = SweepSpec(start_step, positive_only, ratio, aggregation)
                    metrics = _group_metrics(records, audio_scores, max_fpr)
                    common = {
                        "start_step": spec.start_step,
                        "af_mode": spec.af_mode,
                        "topk_ratio": spec.topk_ratio,
                        "patch_aggregation": spec.aggregation,
                        "patch_topk_ratio": (
                            patch_topk_ratio if aggregation == "topk_mean" else ""
                        ),
                    }
                    for metric in metrics:
                        group_rows.append({**common, **metric})
                    summary_rows.append({**common, **_summarize(metrics)})

        elapsed = time.perf_counter() - step_start_time
        print(
            f"fan_sweep_step_complete: start={start_step} "
            f"index={start_index}/{len(start_steps)} seconds={elapsed:.1f}"
        )
        del diffusion, patch_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows.sort(key=lambda row: float(row["overall_hmean"]), reverse=True)
    _write_rows(output_root / "fan_sweep_summary.csv", summary_rows)
    _write_rows(output_root / "fan_sweep_groups.csv", group_rows)
    best = summary_rows[0]
    with (output_root / "fan_sweep_best.json").open("w", encoding="utf-8") as stream:
        json.dump(best, stream, indent=2)
    elapsed = time.perf_counter() - sweep_start_time
    print(
        "fan_sweep_best: "
        f"start={best['start_step']} af={best['af_mode']} "
        f"topk={best['topk_ratio']} aggregation={best['patch_aggregation']} "
        f"auc_mean={best['auc_mean']:.6f} pauc_mean={best['pauc_mean']:.6f} "
        f"overall_hmean={best['overall_hmean']:.6f}"
    )
    print(f"fan_sweep_complete: seconds={elapsed:.1f} results={output_root}")
    print(
        "warning: the best setting is selected on labelled development-test data; "
        "freeze it before reporting results on other machines or evaluation data"
    )
