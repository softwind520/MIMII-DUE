"""Residual-distribution GMM scoring for a trained fan diffusion model."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.mixture import GaussianMixture
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


RESIDUAL_MODES = ("signed", "absolute", "relu")
GMM_SCOPES = ("global", "section")
GMM_COVARIANCES = ("full", "diag")


def pool_residual_features(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pool a [B, 1, F, T] residual over time into [B, F] vectors."""
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Reconstruction shape {tuple(reconstructed.shape)} does not match "
            f"input shape {tuple(original.shape)}"
        )
    if original.ndim != 4 or original.shape[1] != 1:
        raise ValueError("Residual features require tensors shaped [B, 1, F, T]")
    difference = (original - reconstructed).squeeze(1)
    return {
        "signed": difference.mean(dim=-1),
        "absolute": difference.abs().mean(dim=-1),
        "relu": difference.clamp_min(0.0).mean(dim=-1),
    }


def fit_gmm_anomaly_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_sections: np.ndarray,
    test_sections: np.ndarray,
    scope: str,
    covariance_type: str,
    components: int,
    reg_covar: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit normal residual GMMs and return test negative log likelihoods."""
    if scope not in GMM_SCOPES:
        raise ValueError(f"Unsupported GMM scope: {scope!r}")
    if covariance_type not in GMM_COVARIANCES:
        raise ValueError(f"Unsupported GMM covariance: {covariance_type!r}")
    if components < 1:
        raise ValueError("GMM components must be positive")
    if reg_covar <= 0.0:
        raise ValueError("GMM reg_covar must be positive")
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("GMM features must be two-dimensional")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Train and test feature dimensions must match")
    if len(train_features) != len(train_sections):
        raise ValueError("Train features and section labels must have equal length")
    if len(test_features) != len(test_sections):
        raise ValueError("Test features and section labels must have equal length")

    scores = np.empty(len(test_features), dtype=np.float64)
    group_names = ["global"] if scope == "global" else sorted(set(test_sections.tolist()))
    convergence: list[bool] = []
    iterations: list[int] = []
    for group_name in group_names:
        if scope == "global":
            train_mask = np.ones(len(train_features), dtype=bool)
            test_mask = np.ones(len(test_features), dtype=bool)
        else:
            train_mask = train_sections == group_name
            test_mask = test_sections == group_name
        group_train = train_features[train_mask]
        if len(group_train) < components:
            raise ValueError(
                f"GMM group {group_name!r} has {len(group_train)} samples, "
                f"fewer than {components} components"
            )
        if not test_mask.any():
            continue
        model = GaussianMixture(
            n_components=components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=200,
            n_init=1,
            random_state=seed,
        )
        model.fit(group_train)
        scores[test_mask] = -model.score_samples(test_features[test_mask])
        convergence.append(bool(model.converged_))
        iterations.append(int(model.n_iter_))

    if not np.isfinite(scores).all():
        raise FloatingPointError("GMM produced a non-finite anomaly score")
    return scores, {
        "gmm_groups": len(group_names),
        "all_converged": all(convergence),
        "max_iterations": max(iterations),
    }


def _select_train_records(
    records: list[AudioRecord],
    files_per_section_domain: int | None,
) -> list[AudioRecord]:
    if files_per_section_domain is None:
        return records
    if files_per_section_domain < 1:
        raise ValueError("Training files per section/domain must be positive")
    grouped: dict[tuple[str, str], list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.domain)].append(record)
    selected: list[AudioRecord] = []
    for key in sorted(grouped):
        selected.extend(grouped[key][:files_per_section_domain])
    return sorted(selected, key=lambda record: str(record.path))


def _extract_audio_features(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: torch.utils.data.DataLoader,
    records: list[AudioRecord],
    device: torch.device,
    start_step: int,
    reconstruction_samples: int,
    use_amp: bool,
    log_interval: int,
    seed: int,
    split_name: str,
) -> dict[str, np.ndarray]:
    """Reconstruct patches and average their time-pooled residuals per audio."""
    _set_seed(seed)
    feature_sums: dict[str, dict[str, np.ndarray]] = {
        mode: {} for mode in RESIDUAL_MODES
    }
    patch_counts: dict[str, int] = defaultdict(int)
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
                    reconstructed.add_(
                        diffusion.ddim_reconstruct(
                            model, clean, start_step=start_step
                        )
                    )
            reconstructed.div_(reconstruction_samples)
            pooled = pool_residual_features(clean, reconstructed)
            paths = [str(path) for path in batch["audio_path"]]
            for mode, values in pooled.items():
                array = values.float().cpu().numpy()
                if not np.isfinite(array).all():
                    raise FloatingPointError(
                        f"Non-finite {mode} residual feature in {split_name} "
                        f"batch {batch_index}"
                    )
                for path, vector in zip(paths, array):
                    if path not in feature_sums[mode]:
                        feature_sums[mode][path] = vector.astype(np.float64)
                    else:
                        feature_sums[mode][path] += vector
            for path in paths:
                patch_counts[path] += 1
            if batch_index == 1 or batch_index % log_interval == 0:
                print(
                    f"gmm_feature_batch: split={split_name} "
                    f"batch={batch_index}/{len(loader)}"
                )

    ordered_paths = [str(record.path) for record in records]
    missing = [path for path in ordered_paths if patch_counts[path] == 0]
    if missing:
        raise RuntimeError(f"No residual features were produced for {len(missing)} files")
    return {
        mode: np.stack(
            [feature_sums[mode][path] / patch_counts[path] for path in ordered_paths]
        ).astype(np.float32)
        for mode in RESIDUAL_MODES
    }


def _group_metrics(
    records: list[AudioRecord],
    scores: np.ndarray,
    max_fpr: float,
) -> list[dict]:
    score_by_path = {
        str(record.path): float(score) for record, score in zip(records, scores)
    }
    grouped: dict[tuple[str, str], list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.domain)].append(record)
    rows: list[dict] = []
    for (section, domain), group_records in sorted(grouped.items()):
        labelled = [record for record in group_records if record.label is not None]
        labels = [int(record.label) for record in labelled]
        group_scores = [score_by_path[str(record.path)] for record in labelled]
        result = evaluate_audio_scores(labels, group_scores, max_fpr)
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
    return {
        "auc_mean": float(np.mean(auc_values)),
        "auc_hmean": harmonic_mean(auc_values),
        "pauc_mean": float(np.mean(pauc_values)),
        "pauc_hmean": harmonic_mean(pauc_values),
        "overall_hmean": harmonic_mean(auc_values + pauc_values),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty GMM table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fan_gmm(
    config: dict,
    start_step: int = 400,
    patch_hop: int = 32,
    residual_modes: list[str] | None = None,
    scopes: list[str] | None = None,
    covariance_types: list[str] | None = None,
    components: int = 2,
    reg_covar: float = 1e-5,
    max_test_files_per_group: int | None = None,
    max_train_files_per_section_domain: int | None = None,
) -> None:
    """Fit GMMs on normal fan residuals and evaluate their likelihood scores."""
    residual_modes = list(dict.fromkeys(residual_modes or RESIDUAL_MODES))
    scopes = list(dict.fromkeys(scopes or GMM_SCOPES))
    covariance_types = list(
        dict.fromkeys(covariance_types or GMM_COVARIANCES)
    )
    unknown_modes = sorted(set(residual_modes).difference(RESIDUAL_MODES))
    unknown_scopes = sorted(set(scopes).difference(GMM_SCOPES))
    unknown_covariances = sorted(
        set(covariance_types).difference(GMM_COVARIANCES)
    )
    if unknown_modes or unknown_scopes or unknown_covariances:
        raise ValueError(
            "Unsupported GMM options: "
            f"modes={unknown_modes}, scopes={unknown_scopes}, "
            f"covariances={unknown_covariances}"
        )
    training_steps = int(config["diffusion"]["training_steps"])
    if not 0 <= start_step < training_steps:
        raise ValueError(f"GMM start step must be in [0, {training_steps - 1}]")
    if patch_hop < 1:
        raise ValueError("GMM patch hop must be positive")

    gmm_config = deepcopy(config)
    gmm_config["data"]["test_patch_hop"] = patch_hop
    gmm_config["data"]["cover_tail"] = True
    machine_type = "fan"
    seed = int(gmm_config["project"]["seed"])
    _set_seed(seed)
    device = _resolve_device(gmm_config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")

    model, checkpoint_path, checkpoint = _load_evaluation_model(
        gmm_config, machine_type, device
    )
    diffusion = GaussianDiffusion(gmm_config).to(device)
    data_root = resolve_config_path(gmm_config, gmm_config["data"]["root"])
    train_records = discover_audio_records(data_root, [machine_type], "train")
    if any(record.label not in (None, 0) for record in train_records):
        raise ValueError("GMM fitting data must contain only normal training audio")
    train_records = _select_train_records(
        train_records, max_train_files_per_section_domain
    )
    test_records = discover_audio_records(data_root, [machine_type], "test")
    test_records = _select_evaluation_records(
        test_records, max_test_files_per_group
    )
    train_loader = _evaluation_loader(gmm_config, train_records, device)
    test_loader = _evaluation_loader(gmm_config, test_records, device)
    evaluation_config = gmm_config["evaluation"]
    reconstruction_samples = int(evaluation_config.get("reconstruction_samples", 1))
    if reconstruction_samples < 1:
        raise ValueError("reconstruction_samples must be positive")
    log_interval = int(evaluation_config.get("log_interval_batches", 50))
    use_amp = bool(evaluation_config.get("mixed_precision", True)) and device.type == "cuda"
    max_fpr = float(evaluation_config["max_fpr"])

    output_root = (
        resolve_config_path(gmm_config, gmm_config["project"]["output_directory"])
        / "fan_gmm"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "machine_type": machine_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "seed": seed,
        "train_files": len(train_records),
        "test_files": len(test_records),
        "train_patches": len(train_loader.dataset),
        "test_patches": len(test_loader.dataset),
        "start_step": start_step,
        "ddim_stride": int(gmm_config["diffusion"]["ddim_stride"]),
        "patch_hop": patch_hop,
        "cover_tail": True,
        "reconstruction_samples": reconstruction_samples,
        "residual_modes": residual_modes,
        "scopes": scopes,
        "covariance_types": covariance_types,
        "components": components,
        "reg_covar": reg_covar,
    }
    with (output_root / "fan_gmm_run.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)

    print(
        f"fan_gmm_start: checkpoint={checkpoint_path} "
        f"epoch={int(checkpoint['epoch']) + 1} start={start_step} "
        f"patch_hop={patch_hop} train_files={len(train_records)} "
        f"train_patches={len(train_loader.dataset)} test_files={len(test_records)} "
        f"test_patches={len(test_loader.dataset)}"
    )
    start_time = time.perf_counter()
    train_features = _extract_audio_features(
        model,
        diffusion,
        train_loader,
        train_records,
        device,
        start_step,
        reconstruction_samples,
        use_amp,
        log_interval,
        seed,
        "train",
    )
    test_features = _extract_audio_features(
        model,
        diffusion,
        test_loader,
        test_records,
        device,
        start_step,
        reconstruction_samples,
        use_amp,
        log_interval,
        seed + 1,
        "test",
    )

    train_paths = np.asarray([str(record.path) for record in train_records])
    test_paths = np.asarray([str(record.path) for record in test_records])
    train_sections = np.asarray([record.section for record in train_records])
    test_sections = np.asarray([record.section for record in test_records])
    train_domains = np.asarray([record.domain for record in train_records])
    test_domains = np.asarray([record.domain for record in test_records])
    test_labels = np.asarray([int(record.label) for record in test_records])
    np.savez_compressed(
        output_root / "fan_residual_features.npz",
        train_paths=train_paths,
        test_paths=test_paths,
        train_sections=train_sections,
        test_sections=test_sections,
        train_domains=train_domains,
        test_domains=test_domains,
        test_labels=test_labels,
        **{f"train_{mode}": train_features[mode] for mode in RESIDUAL_MODES},
        **{f"test_{mode}": test_features[mode] for mode in RESIDUAL_MODES},
    )

    summary_rows: list[dict] = []
    group_rows: list[dict] = []
    score_rows: list[dict] = []
    for residual_mode in residual_modes:
        for scope in scopes:
            for covariance_type in covariance_types:
                scores, diagnostics = fit_gmm_anomaly_scores(
                    train_features[residual_mode],
                    test_features[residual_mode],
                    train_sections,
                    test_sections,
                    scope,
                    covariance_type,
                    components,
                    reg_covar,
                    seed,
                )
                common = {
                    "residual_mode": residual_mode,
                    "scope": scope,
                    "covariance_type": covariance_type,
                    "components": components,
                    "reg_covar": reg_covar,
                }
                metrics = _group_metrics(test_records, scores, max_fpr)
                for metric in metrics:
                    group_rows.append({**common, **metric})
                summary_rows.append(
                    {**common, **diagnostics, **_summarize(metrics)}
                )
                for record, score in zip(test_records, scores):
                    score_rows.append(
                        {
                            **common,
                            "filename": record.path.name,
                            "section": record.section,
                            "domain": record.domain,
                            "label": int(record.label),
                            "score": float(score),
                        }
                    )
                print(
                    f"gmm_result: residual={residual_mode} scope={scope} "
                    f"covariance={covariance_type} "
                    f"overall_hmean={summary_rows[-1]['overall_hmean']:.6f}"
                )

    summary_rows.sort(key=lambda row: float(row["overall_hmean"]), reverse=True)
    _write_rows(output_root / "fan_gmm_summary.csv", summary_rows)
    _write_rows(output_root / "fan_gmm_groups.csv", group_rows)
    _write_rows(output_root / "fan_gmm_audio_scores.csv", score_rows)
    best = summary_rows[0]
    with (output_root / "fan_gmm_best.json").open("w", encoding="utf-8") as stream:
        json.dump(best, stream, indent=2)
    elapsed = time.perf_counter() - start_time
    print(
        "fan_gmm_best: "
        f"residual={best['residual_mode']} scope={best['scope']} "
        f"covariance={best['covariance_type']} "
        f"auc_mean={best['auc_mean']:.6f} pauc_mean={best['pauc_mean']:.6f} "
        f"overall_hmean={best['overall_hmean']:.6f}"
    )
    print(f"fan_gmm_complete: seconds={elapsed:.1f} results={output_root}")
    print(
        "warning: the best GMM is selected on labelled development-test data; "
        "freeze it before evaluating other machines or final evaluation data"
    )

