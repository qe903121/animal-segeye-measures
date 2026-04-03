"""Landmark localization module for animal eye detection.

Provides:
    BaseEyeDetector: Abstract interface for eye detection strategies.
    HeuristicCVDetector: Traditional CV pipeline (blob detection + geometry).
    DeepPoseDetector: Placeholder for future AI-based detection.
    create_detector: Factory function for strategy selection.
"""

from .base import BaseEyeDetector
from .detector_ai import DeepPoseDetector
from .detector_cv import HeuristicCVDetector
from .factory import create_detector

__all__ = [
    "BaseEyeDetector",
    "HeuristicCVDetector",
    "DeepPoseDetector",
    "create_detector",
]
