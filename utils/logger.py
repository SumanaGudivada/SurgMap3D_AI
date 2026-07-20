from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path


def get_logger(name: str) -> Logger:
    """Create and return a configured logger.

    The logger writes INFO-level and above records to both the console and
    to ``outputs/logs/app.log``. The output directory is created automatically.

    Args:
        name: The logger name, typically ``__name__`` from the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == str(log_file)
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
