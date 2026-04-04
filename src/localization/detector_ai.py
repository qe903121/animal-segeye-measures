"""AI-based eye detector using OpenMMLab MMPose.

Implements :class:`DeepPoseDetector` with an AP-10K pose model
(e.g. RTMPose-m for animals). The detector is initialized via
``MMPoseInferencer`` for alias/checkpoint resolution, but inference is
executed with the lower-level ``inference_topdown`` API so each call is
strictly anchored to the current annotation's GT bounding box.

Indices ``0`` and ``1`` correspond to the left and right eyes in the
AP-10K dataset format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.data.loader import EyeResult

from .base import BaseEyeDetector

logger = logging.getLogger(__name__)


class DeepPoseDetector(BaseEyeDetector):
    """MMPose-based animal eye detector using AP-10K dataset weights.

    Attributes:
        _inferencer: Instance of MMPoseInferencer used for model loading.
        _pose_model: Loaded pose estimator model.
        _inference_topdown: Low-level bbox-driven inference function.
        _threshold: Minimum confidence score to accept a keypoint.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the MMPose inferencer.

        Args:
            config: Full parsed config dictionary.
        """
        # Delay import to avoid loading heavy PyTorch/MMPose if not used.
        try:
            from mmpose.apis import MMPoseInferencer, inference_topdown
        except ImportError:
            raise ImportError(
                "MMPose is not installed. Please install the packages specified in "
                "`requirements-ai.txt` to use the AI method."
            )

        ai_cfg = config.get("eye_detection", {}).get("ai_model", {})
        alias = ai_cfg.get("alias", "animal")
        device = ai_cfg.get("device", "cpu")
        self._threshold = ai_cfg.get("score_threshold", 0.3)

        logger.info(
            "Initializing DeepPoseDetector... (MMPoseInferencer, alias='%s', device='%s')",
            alias, device
        )

        # Build the pose model via model alias. We intentionally keep the
        # inferencer object for alias/checkpoint resolution, but do not use its
        # high-level __call__ path because that path re-runs whole-image
        # detection and can ignore the caller-provided bbox for top-down models.
        self._inferencer = MMPoseInferencer(pose2d=alias, device=device)
        self._pose_model = self._inferencer.inferencer.model
        self._inference_topdown = inference_topdown
        logger.info("DeepPoseDetector initialized.")

    def detect(
        self,
        image: np.ndarray,
        bbox: list[float],
        mask: np.ndarray,
        category: str,
    ) -> EyeResult:
        """Detect eyes using a GT-bbox-anchored top-down pose model.

        Args:
            image: Original BGR image, shape ``(H, W, 3)``.
            bbox: Bounding box ``[x, y, w, h]``.
            mask: Context GT mask, unused by the current AI path.
            category: Animal category string.

        Returns:
            :class:`EyeResult` containing extracted eye coordinates.
        """
        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h
        bbox_xyxy = [[x_min, y_min, x_max, y_max]]

        # Run low-level bbox-driven top-down inference so the current
        # annotation bbox is the only instance sent to the pose model.
        data_samples = self._inference_topdown(
            self._pose_model,
            image,
            bboxes=bbox_xyxy,
            bbox_format="xyxy",
        )

        if not data_samples:
            logger.warning("MMPose inference did not return any data samples.")
            return _make_result("FAILED_NOT_FOUND")

        # With one bbox input, inference_topdown should return one data sample.
        data_sample = data_samples[0]
        if (
            not hasattr(data_sample, "pred_instances")
            or len(data_sample.pred_instances) == 0
        ):
            return _make_result("FAILED_NOT_FOUND")

        pred_instances = data_sample.pred_instances

        # Each bbox produces one pose instance.
        keypoints = pred_instances.keypoints[0]
        scores = pred_instances.keypoint_scores[0]

        # AP-10K keypoint mapping:
        # 0: 'L_Eye', 1: 'R_Eye', 2: 'Nose', 3: 'Neck', 4: 'Root_of_Tail',
        # 5: 'L_Shoulder', 6: 'L_Elbow', 7: 'L_F_Paw',
        # 8: 'R_Shoulder', 9: 'R_Elbow', 10: 'R_F_Paw',
        # 11: 'L_Hip', 12: 'L_Knee', 13: 'L_B_Paw',
        # 14: 'R_Hip', 15: 'R_Knee', 16: 'R_B_Paw'
        if len(scores) < 2:
            logger.warning(
                "Insufficient predicted keypoints (<2). Verify the model uses AP-10K format."
            )
            return _make_result("FAILED_NOT_FOUND")

        # Store all keypoints for full-skeleton visualization
        all_keypoints = [[float(pt[0]), float(pt[1])] for pt in keypoints]
        all_scores = [float(s) for s in scores]

        l_pt, l_score = keypoints[0], float(scores[0])
        r_pt, r_score = keypoints[1], float(scores[1])

        left_eye = [float(l_pt[0]), float(l_pt[1])] if l_score >= self._threshold else None
        right_eye = [float(r_pt[0]), float(r_pt[1])] if r_score >= self._threshold else None

        # Determine overall status and confidence
        if left_eye is not None and right_eye is not None:
            status = "SUCCESS"
        elif left_eye is not None or right_eye is not None:
            status = "SINGLE_EYE"
        else:
            status = "FAILED_NOT_FOUND"

        valid_scores = [s for s in (l_score, r_score) if s >= self._threshold]
        confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        logger.warning(
            "AI Detection (bbox=[%.1f,%.1f,%.1f,%.1f], conf=%.2f): "
            "L=(%.1f,%.1f/s:%.2f) R=(%.1f,%.1f/s:%.2f) -> %s",
            x_min, y_min, x_max, y_max,
            confidence,
            l_pt[0], l_pt[1], l_score,
            r_pt[0], r_pt[1], r_score,
            status,
        )

        return {
            "status": status,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "confidence": round(confidence, 3),
            "all_keypoints": all_keypoints,
            "all_scores": all_scores,
        }

def _make_result(status: str) -> EyeResult:
    """Create a failed EyeResult with the given status."""
    return {
        "status": status,
        "left_eye": None,
        "right_eye": None,
        "confidence": 0.0,
    }
