"""Unified Evaluation Framework.

Provides a modular, registry-based validation engine for evaluating
pipeline outputs (eye localization, measurements, etc.) in a
standardized manner.

Public API:
    - :class:`BaseValidator`: Abstract base class for all validators.
    - :class:`EvaluationEngine`: Registry + executor for validators.
    - :class:`LocalizationValidator`: Eye localization quality validator.
    - :class:`MeasurementValidator`: Phase 3 baseline measurement validator.
"""

from .base import BaseValidator
from .engine import EvaluationEngine
from .valid_loc import LocalizationValidator
from .valid_measure import MeasurementValidator

__all__ = [
    "BaseValidator",
    "EvaluationEngine",
    "LocalizationValidator",
    "MeasurementValidator",
]
