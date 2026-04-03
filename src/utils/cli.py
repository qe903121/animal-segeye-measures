"""Shared CLI utilities for entry-point scripts.

Provides common helpers (config loading, logging setup) used by
multiple ``run_*.py`` entry points, avoiding code duplication.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with console output.

    Args:
        verbose: If True, set level to DEBUG; else INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to the ``config.yaml`` file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: Config file does not exist.
        yaml.YAMLError: Config file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"設定檔不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.getLogger(__name__).info("已載入設定檔: %s", path)
    return config
