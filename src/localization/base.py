"""Abstract base class for eye detection strategies.

Defines the unified interface that all eye detectors must implement,
enabling seamless strategy switching between CV and AI approaches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.data.loader import AnnotationEntry, EyeResult, ImageRecord

logger = logging.getLogger(__name__)


class BaseEyeDetector(ABC):
    """Abstract base class for eye detection strategies.

    All detectors must implement :meth:`detect` with identical input/output
    signatures. The :meth:`process_dataset` method provides batch processing
    logic shared across all strategies.

    Subclasses:
        - :class:`HeuristicCVDetector`: Traditional CV pipeline.
        - :class:`DeepPoseDetector`: AI keypoint localization via MMPose.
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        bbox: list[float],
        mask: np.ndarray,
        category: str,
    ) -> EyeResult:
        """Detect eyes for a single animal instance.

        Args:
            image: Original BGR image, shape ``(H, W, 3)``.
            bbox: Bounding box ``[x, y, width, height]``.
            mask: Binary mask, shape ``(H, W)``, dtype ``uint8``,
                values 0 or 255.
            category: Animal category name (e.g., ``"dog"``).

        Returns:
            :class:`EyeResult` with status, coordinates, and confidence.
        """

    def process_dataset(
        self, dataset: list[ImageRecord]
    ) -> list[ImageRecord]:
        """Batch-process a dataset, populating ``eyes`` for each annotation.

        Loads each image once from disk and runs :meth:`detect` on all
        annotations within that image. Results are written in-place to
        each annotation's ``eyes`` field.

        Args:
            dataset: Phase 1 output (list of :class:`ImageRecord`).

        Returns:
            The same dataset list with ``eyes`` fields populated.
        """
        total = sum(len(r["annotations"]) for r in dataset)
        processed = 0
        stats: dict[str, int] = {}

        for record in dataset:
            image_path = Path(record["image_path"])
            image = cv2.imread(str(image_path))

            if image is None:
                logger.error("無法讀取圖片: %s", image_path)
                for ann in record["annotations"]:
                    ann["eyes"] = {
                        "status": "FAILED_NOT_FOUND",
                        "left_eye": None,
                        "right_eye": None,
                        "confidence": 0.0,
                    }
                continue

            for ann in record["annotations"]:
                result = self.detect(
                    image=image,
                    bbox=ann["bbox"],
                    mask=ann["mask"],
                    category=ann["category"],
                )
                ann["eyes"] = result

                status = result["status"]
                stats[status] = stats.get(status, 0) + 1
                processed += 1

        logger.info(
            "眼睛偵測完成: %d/%d 個標註已處理", processed, total
        )
        for status, count in sorted(stats.items()):
            logger.info("  %s: %d (%.1f%%)", status, count, count / max(total, 1) * 100)

        return dataset
