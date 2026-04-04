"""Unified Evaluation Framework.

Provides a modular, registry-based validation engine for evaluating
pipeline outputs (eye localization, measurements, etc.) in a
standardized manner.

Public API:
    - :class:`BaseValidator`: Abstract base class for all validators.
    - :class:`EvaluationEngine`: Registry + executor for validators.
    - :class:`LocalizationValidator`: Eye localization quality validator.
    - :class:`MeasurementValidator`: Phase 3 baseline measurement validator.
    - :class:`AccuracyValidator`: Ground truth accuracy validator.
"""

from .base import BaseValidator
from .engine import EvaluationEngine
from .localization import LocalizationValidator
from .measurement import MeasurementValidator
from .accuracy import AccuracyValidator

__all__ = [
    "BaseValidator",
    "EvaluationEngine",
    "LocalizationValidator",
    "MeasurementValidator",
    "AccuracyValidator",
]
