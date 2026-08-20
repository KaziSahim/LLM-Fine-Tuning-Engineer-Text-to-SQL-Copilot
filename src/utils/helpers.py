"""General helper utilities for the LLM fine-tuning project."""

import json
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .exceptions import ConfigError, DataError
from .logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return config or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file: {path}", details=str(e))


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a dictionary as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Saved YAML to {path}")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL (JSON Lines) file."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"JSONL file not found: {path}")

    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DataError(
                        f"Invalid JSON on line {line_num} in {path}",
                        details=str(e),
                    )
        logger.info(f"Loaded {len(data)} records from {path}")
        return data
    except Exception as e:
        raise DataError(f"Failed to load JSONL: {path}", details=str(e))


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save a list of dictionaries as a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(data)} records to {path}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it does not exist and return the Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current timestamp as string."""
    return datetime.now().strftime(fmt)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """Format large numbers into human-readable form (e.g. 1.2M, 3.4B)."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def print_model_size(model: torch.nn.Module) -> None:
    """Log total and trainable parameter counts."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    logger.info(
        f"Model parameters → Total: {format_number(total)} | "
        f"Trainable: {format_number(trainable)} "
        f"({100 * trainable / total:.2f}%)"
    )


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device