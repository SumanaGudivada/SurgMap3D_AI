"""Application entry point for the SurgMap3D-AI dataset layer."""

from __future__ import annotations

from pathlib import Path

import yaml

from datasets.dataset_loader import DatasetLoader
from datasets.dataset_statistics import DatasetStatistics
from datasets.dataset_validator import DatasetValidator
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Load, validate, and summarize the configured dataset.

    The function coordinates configuration loading and dataset services only.
    It does not read image contents or run any model or preprocessing logic.

    Raises:
        FileNotFoundError: If the project configuration file is missing.
        KeyError: If required configuration keys are missing.
        yaml.YAMLError: If the configuration contains invalid YAML.
    """
    # Project root
    project_root = Path(__file__).resolve().parent

    # Configuration file
    config_path = project_root / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    # Load configuration
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Dataset configuration
    datasets_config = config["datasets"]
    dataset_name = datasets_config["default_dataset"]
    dataset_path = Path(
        datasets_config["available_datasets"][dataset_name]
)

    dataset_root = (
    dataset_path
    if dataset_path.is_absolute()
    else project_root / dataset_path
)

    LOGGER.info(
        "Loading configured dataset '%s' from %s",
        dataset_name,
        dataset_root,
    )

    # Create dataset adapter
    dataset = DatasetLoader.load_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )

    # Validate dataset
    if not DatasetValidator.validate(dataset):
        LOGGER.error("Dataset validation failed. Exiting application.")
        return

    LOGGER.info("Dataset validation completed successfully.")

    # Retrieve statistics
    statistics = DatasetStatistics.get_statistics(dataset)

    LOGGER.info("Dataset statistics retrieved successfully.")

    for key, value in statistics.items():
        LOGGER.info("%s: %s", key, value)

    # Console output
    print("\n" + "=" * 50)
    print("        DATASET SUMMARY")
    print("=" * 50)

    for key, value in statistics.items():
        print(f"{key.replace('_', ' ').title():25}: {value}")

    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "Unexpected error while running SurgMap3D-AI"
        )