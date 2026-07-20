from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    """Load configuration from ``configs/config.yaml``.

    The loader locates the project root automatically based on the location of
    this file, then reads and parses the YAML configuration file stored in
    ``configs/config.yaml``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If ``configs/config.yaml`` does not exist.
        ValueError: If the YAML content cannot be parsed.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in configuration file {config_path}: {exc}"
        ) from exc

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file {config_path} must contain a YAML mapping"
        )

    return config
