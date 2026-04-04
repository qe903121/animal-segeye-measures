"""Factory function for creating eye detector instances.

Provides :func:`create_detector` which maps a method name string
to the corresponding detector class, supporting the Strategy Pattern.

For ``method="ai"``, the repo now supports multiple pose runtimes behind the
same eye-detector contract:

- ``runtime="pytorch"`` -> MMPose top-down inference
- ``runtime="onnx"`` -> ONNX Runtime top-down inference
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
            - ``"ai"``: Pose-based keypoint localization. The concrete AI
              runtime is selected from configuration.
        config: Full parsed ``config.yaml`` dictionary.

    Returns:
        An instance of a :class:`BaseEyeDetector` subclass.

    Raises:
        ValueError: Unknown method name.
    """
    method = method.lower().strip()

    if method == "cv":
        logger.info("Using detection strategy: HeuristicCVDetector (Traditional CV)")
        return HeuristicCVDetector(config)
    elif method == "ai":
        ai_cfg = config.get("eye_detection", {}).get("ai_model", {})
        runtime = str(ai_cfg.get("runtime", "pytorch")).strip().lower()
        if runtime == "pytorch":
            logger.info(
                "Using detection strategy: DeepPoseDetector (MMPose AP-10K, runtime=pytorch)"
            )
            return DeepPoseDetector(config)
        if runtime == "onnx":
            from .detector_ai_onnx import OnnxPoseDetector

            logger.info(
                "Using detection strategy: OnnxPoseDetector (RTMPose AP-10K, runtime=onnx)"
            )
            return OnnxPoseDetector(config)
        raise ValueError(
            f"Unknown AI runtime: '{runtime}'. Supported runtimes: 'pytorch', 'onnx'"
        )
    else:
        raise ValueError(
            f"Unknown detection method: '{method}'. Supported methods: 'cv', 'ai'"
        )
