"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "project",
    "data",
    "model",
    "conditioning",
    "diffusion",
    "training",
    "scoring",
    "evaluation",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and reject incomplete experiment files."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing configuration sections: {missing_text}")

    conditioning = config["conditioning"]
    if not isinstance(conditioning, dict):
        raise ValueError("conditioning must be a mapping")
    uses_section = bool(conditioning.get("use_section", False))
    uses_domain = bool(conditioning.get("use_domain", False))
    if uses_section and int(conditioning.get("num_sections", 0)) < 1:
        raise ValueError("conditioning.num_sections must be positive")
    dropout = float(conditioning.get("condition_dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("conditioning.condition_dropout must satisfy 0 <= p < 1")
    if (uses_section or uses_domain) and "condition_dimension" not in config["model"]:
        raise ValueError("model.condition_dimension is required for metadata conditioning")

    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent)
    return config


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the directory containing the YAML file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()
