"""logging.py - Auto-generated placeholder."""

"""Centralized logging configuration for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "llm_finetune",
    level: int = logging.INFO,
    log_dir: Optional[str | Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_dir: Directory to save log files. Defaults to outputs/logs.
        log_to_file: Whether to write logs to a file.
        log_to_console: Whether to print logs to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Avoid duplicate handlers on re-import
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path("outputs/logs")
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str = "llm_finetune") -> logging.Logger:
    """Get an existing logger or create a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Fallback basic config if setup_logger was never called
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    return logger