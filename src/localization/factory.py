"""Factory function for creating eye detector instances.

Provides :func:`create_detector` which maps a method name string
to the corresponding detector class, supporting the Strategy Pattern.
"""

from __future__ import annotations

import logging

from .base import BaseEyeDetector
from .detector_ai import DeepPoseDetector
from .detector_cv import HeuristicCVDetector

logger = logging.getLogger(__name__)


def create_detector(method: str, config: dict) -> BaseEyeDetector:
    """Create an eye detector instance based on the specified method.

    Args:
        method: Detection strategy name. Supported values:
            - ``"cv"``: Traditional computer vision pipeline.
            - ``"ai"``: Deep learning (currently stub, falls back to CV).
        config: Full parsed ``config.yaml`` dictionary.

    Returns:
        An instance of a :class:`BaseEyeDetector` subclass.

    Raises:
        ValueError: Unknown method name.
    """
    method = method.lower().strip()

    if method == "cv":
        logger.info("使用偵測策略: HeuristicCVDetector (傳統 CV)")
        return HeuristicCVDetector(config)
    elif method == "ai":
        logger.info("使用偵測策略: DeepPoseDetector (MMPose AP-10K)")
        return DeepPoseDetector(config)
    else:
        raise ValueError(
            f"未知的偵測方法: '{method}'。支援的方法: 'cv', 'ai'"
        )
