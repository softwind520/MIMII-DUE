"""Human-readable validation for the stage-2 data pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import resolve_config_path
from .dataset import (
    MIMIIDUEPatchDataset,
    build_domain_balanced_sampler,
    count_records,
    discover_audio_records,
)
from .records import AudioRecord


def _representative_records(records: list[AudioRecord], limit: int) -> list[AudioRecord]:
    grouped: dict[tuple[str, str, str], list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.machine_type, record.section, record.domain)].append(record)
    selected = [grouped[key][0] for key in sorted(grouped)]
    if limit > len(selected):
        selected_paths = {record.path for record in selected}
        selected.extend(record for record in records if record.path not in selected_paths)
    return selected[:limit]


def _print_inventory(records: list[AudioRecord]) -> None:
    counts = count_records(records)
    print("machine      split         domain  label     files")
    print("-----------  ------------  ------  --------  -----")
    for (machine, split, domain, label), count in sorted(counts.items()):
        label_name = "unknown" if label is None else ("normal" if label == 0 else "anomaly")
        print(f"{machine:11}  {split:12}  {domain:6}  {label_name:8}  {count:5d}")


def check_data_pipeline(
    config: dict,
    machine_types: list[str] | None = None,
    max_files: int = 12,
    build_cache: bool = False,
) -> None:
    """Validate discovery and feature extraction without starting model training."""
    if max_files < 1:
        raise ValueError("max_files must be positive")
    data_config = config["data"]
    machines = machine_types or list(data_config["machine_types"])
    unknown = sorted(set(machines).difference(data_config["machine_types"]))
    if unknown:
        raise ValueError(f"Unknown machine types: {', '.join(unknown)}")

    data_root = resolve_config_path(config, data_config["root"])
    train_records = discover_audio_records(data_root, machines, "train")
    test_records = discover_audio_records(data_root, machines, "test")

    print(f"dataset_root: {data_root}")
    print(f"machines: {', '.join(machines)}")
    print(f"train_files: {len(train_records)}")
    print(f"test_files: {len(test_records)}")
    _print_inventory(train_records + test_records)

    examples = _representative_records(train_records, min(max_files, len(train_records)))
    cache_root = resolve_config_path(config, data_config["feature_cache"]) if build_cache else None
    dataset = MIMIIDUEPatchDataset(examples, config, training=True, feature_cache=cache_root)

    first_item_for_record: dict[int, int] = {}
    for item_index, patch_index in enumerate(dataset.index):
        first_item_for_record.setdefault(patch_index.record_index, item_index)

    observed_shapes: set[tuple[int, ...]] = set()
    minimum = float("inf")
    maximum = float("-inf")
    for item_index in first_item_for_record.values():
        item = dataset[item_index]
        patch = item["patch"]
        if not torch.isfinite(patch).all():
            raise ValueError(f"Non-finite feature values: {item['audio_path']}")
        observed_shapes.add(tuple(patch.shape))
        minimum = min(minimum, float(patch.min()))
        maximum = max(maximum, float(patch.max()))

    expected_shape = (1, int(data_config["n_mels"]), int(data_config["patch_frames"]))
    if observed_shapes != {expected_shape}:
        raise ValueError(f"Unexpected patch shapes: {sorted(observed_shapes)}")
    if data_config["normalization"] == "log_db_minmax" and not (0.0 <= minimum <= maximum <= 1.0):
        raise ValueError(f"Normalized features outside [0, 1]: min={minimum}, max={maximum}")

    sampler = build_domain_balanced_sampler(dataset)
    domain_mass = defaultdict(float)
    for patch_index, weight in zip(dataset.index, sampler.weights.tolist()):
        domain = dataset.records[patch_index.record_index].domain
        domain_mass[domain] += weight
    if len(domain_mass) > 1 and max(domain_mass.values()) - min(domain_mass.values()) > 1e-9:
        raise ValueError(f"Domain sampler is not balanced: {dict(domain_mass)}")

    batch_size = min(int(config["training"]["batch_size"]), len(dataset))
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)))
    expected_batch_shape = (batch_size, *expected_shape)
    if tuple(batch["patch"].shape) != expected_batch_shape:
        raise ValueError(
            f"Unexpected batch shape {tuple(batch['patch'].shape)}; expected {expected_batch_shape}"
        )
    conditioning_config = config["conditioning"]
    if bool(conditioning_config.get("use_section", False)):
        num_sections = int(conditioning_config["num_sections"])
        if bool(((batch["section_id"] < 0) | (batch["section_id"] >= num_sections)).any()):
            raise ValueError(f"Section ids fall outside [0, {num_sections - 1}]")
    if bool(conditioning_config.get("use_domain", False)):
        if bool(((batch["domain_id"] < 0) | (batch["domain_id"] > 1)).any()):
            raise ValueError("Known domain ids must be source=0 or target=1")

    print(f"checked_audio_files: {len(first_item_for_record)}")
    print(f"indexed_training_patches: {len(dataset)}")
    print(f"patch_shape: {expected_shape}")
    print(f"batch_shape: {tuple(batch['patch'].shape)}")
    print(f"balanced_domain_mass: {dict(sorted(domain_mass.items()))}")
    print(f"feature_range: [{minimum:.6f}, {maximum:.6f}]")
    print(f"feature_cache: {cache_root if cache_root else 'disabled for this check'}")
    print(
        "conditioning: "
        f"section={bool(conditioning_config.get('use_section', False))} "
        f"domain={bool(conditioning_config.get('use_domain', False))} "
        f"dropout={float(conditioning_config.get('condition_dropout', 0.0))}"
    )
    print("data_pipeline_check: PASS")
