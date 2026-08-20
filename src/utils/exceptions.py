"""Custom exceptions for the LLM fine-tuning project."""

from typing import Optional, Any


class LLMProjectError(Exception):
    """Base exception for all project-related errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigError(LLMProjectError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(LLMProjectError):
    """Raised for dataset-related problems (loading, formatting, validation)."""
    pass


class ModelError(LLMProjectError):
    """Raised for model loading, quantization, or adapter issues."""
    pass


class TrainingError(LLMProjectError):
    """Raised during the training process."""
    pass


class EvaluationError(LLMProjectError):
    """Raised during evaluation or metric calculation."""
    pass


class InferenceError(LLMProjectError):
    """Raised during inference / generation."""
    pass


class ExportError(LLMProjectError):
    """Raised when merging adapters or exporting the model fails."""
    pass