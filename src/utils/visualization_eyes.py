"""Visualization tools for eye detection results (Phase 2).

Overlays GT contours, detected eye positions, connecting lines, status
labels, and confidence scores onto original images for human visual
inspection.

Typical usage:
    >>> from src.utils.visualization_eyes import debug_visualize_eyes
    >>> saved = debug_visualize_eyes(dataset, output_dir="output/debug_eyes")
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.data.loader import ImageRecord

logger = logging.getLogger(__name__)

# Color constants (BGR)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)
_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_WHITE = (255, 255, 255)
_GRAY = (180, 180, 180)

# AP-10K keypoint names (17 points)
_AP10K_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Tail_Root",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]

# AP-10K skeleton connectivity (index pairs)
_AP10K_SKELETON = [
    (0, 2), (1, 2),        # Eyes → Nose
    (2, 3),                # Nose → Neck
    (3, 5), (5, 6), (6, 7),   # Neck → L_Shoulder → L_Elbow → L_F_Paw
    (3, 8), (8, 9), (9, 10),  # Neck → R_Shoulder → R_Elbow → R_F_Paw
    (3, 4),                # Neck → Tail_Root
    (4, 11), (11, 12), (12, 13),  # Tail → L_Hip → L_Knee → L_B_Paw
    (4, 14), (14, 15), (15, 16),  # Tail → R_Hip → R_Knee → R_B_Paw
]

# Subtle pastel colors for non-eye keypoints (BGR)
_SKELETON_LINE_COLOR = (100, 180, 160)   # Light warm gray
_KEYPOINT_DOT_COLOR = (90, 160, 140)    # Slightly darker
_SKELETON_ALPHA = 0.7                   # Transparency for overlay
_CONTOUR_ALPHA = 0.28
_CONTOUR_LINE_THICKNESS = 2

_CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {}


def _get_category_color(category: str) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a given category name."""
    if category not in _CATEGORY_COLORS:
        hue = hash(category) % 180
        hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        _CATEGORY_COLORS[category] = tuple(int(c) for c in bgr[0, 0])
    return _CATEGORY_COLORS[category]


def debug_visualize_eyes(
    dataset: list[ImageRecord],
    output_dir: str = "output/debug_eyes",
    sample_count: int = 10,
    seed: int = 42,
) -> list[str]:
    """Overlay eye detection results on original images.

    For each annotation, draws:
        - Semi-transparent GT contours from Phase 1.
        - Bounding box and category label.
        - **SUCCESS**: Red circle (left eye) + Blue circle (right eye)
          + Green connecting line + confidence score.
        - **SINGLE_EYE**: Single red circle + yellow status label.
        - **FAILED_***: Yellow "X" marker at head ROI center + red
          status label.

    Args:
        dataset: Processed dataset with ``eyes`` field populated.
        output_dir: Directory to save annotated images.
        sample_count: Number of images to visualize. Clamped to
            dataset size. Set to ``len(dataset)`` for all.
        seed: Random seed for reproducible sampling.

    Returns:
        List of absolute paths to saved debug images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_count = min(sample_count, len(dataset))
    if sample_count == 0:
        logger.warning("資料集為空，無法產生眼睛偵測視覺化圖片。")
        return []

    rng = random.Random(seed)
    samples = rng.sample(dataset, sample_count)

    saved_paths: list[str] = []

    for record in samples:
        image_path = Path(record["image_path"])
        if not image_path.is_file():
            logger.error("原圖不存在: %s", image_path)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error("無法讀取圖片: %s", image_path)
            continue

        contour_overlay = image.copy()
        for ann in record["annotations"]:
            color = _get_category_color(ann["category"])
            cv2.drawContours(
                contour_overlay,
                ann["contours"],
                -1,
                color,
                cv2.FILLED,
            )

        image = cv2.addWeighted(
            contour_overlay,
            _CONTOUR_ALPHA,
            image,
            1 - _CONTOUR_ALPHA,
            0,
        )

        for ann in record["annotations"]:
            color = _get_category_color(ann["category"])

            cv2.drawContours(
                image,
                ann["contours"],
                -1,
                color,
                _CONTOUR_LINE_THICKNESS,
                cv2.LINE_AA,
            )

            # Draw bounding box
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

            # Category label
            label = f"{ann['category']} #{ann['id']}"
            cv2.putText(
                image, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
            )

            # Eye detection results
            eyes = ann.get("eyes")
            if eyes is None:
                continue

            # Draw full skeleton first (subtle, behind eye markers)
            _draw_skeleton(image, eyes)

            status = eyes["status"]

            if status == "SUCCESS":
                _draw_success(image, eyes)
            elif status == "SINGLE_EYE":
                _draw_single_eye(image, eyes, x, y, w)
            else:
                _draw_failed(image, eyes, x, y, w, h)

        # Image ID watermark
        cv2.putText(
            image,
            f"ID: {record['image_id']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _YELLOW,
            2,
            cv2.LINE_AA,
        )

        out_file = output_path / f"eyes_{record['image_id']}.jpg"
        cv2.imwrite(str(out_file), image)
        saved_paths.append(str(out_file))

    logger.info(
        "眼睛偵測視覺化完成: %d 張圖片已儲存至 %s",
        len(saved_paths),
        output_path,
    )
    return saved_paths


def _draw_success(
    image: np.ndarray, eyes: dict,
) -> None:
    """Draw successfully detected eye pair on image.

    Args:
        image: BGR image to draw on (modified in-place).
        eyes: EyeResult dict with ``left_eye`` and ``right_eye``.
    """
    left = eyes["left_eye"]
    right = eyes["right_eye"]
    confidence = eyes.get("confidence", 0.0)

    if left is None or right is None:
        return

    lx, ly = int(left[0]), int(left[1])
    rx, ry = int(right[0]), int(right[1])

    # Eye markers
    radius = 5
    cv2.circle(image, (lx, ly), radius, _RED, -1, cv2.LINE_AA)
    cv2.circle(image, (rx, ry), radius, _BLUE, -1, cv2.LINE_AA)

    # Connecting line
    cv2.line(image, (lx, ly), (rx, ry), _GREEN, 2, cv2.LINE_AA)

    # Confidence label
    mid_x = (lx + rx) // 2
    mid_y = (ly + ry) // 2
    conf_text = f"{confidence:.2f}"
    cv2.putText(
        image, conf_text, (mid_x, mid_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GREEN, 1, cv2.LINE_AA,
    )


def _draw_single_eye(
    image: np.ndarray, eyes: dict,
    bbox_x: int, bbox_y: int, bbox_w: int,
) -> None:
    """Draw single-eye detection result.

    Args:
        image: BGR image to draw on (modified in-place).
        eyes: EyeResult dict with at least ``left_eye``.
        bbox_x: Bounding box x-offset.
        bbox_y: Bounding box y-offset.
        bbox_w: Bounding box width.
    """
    eye = eyes.get("left_eye") or eyes.get("right_eye")
    if eye is None:
        return

    ex, ey = int(eye[0]), int(eye[1])
    cv2.circle(image, (ex, ey), 5, _RED, -1, cv2.LINE_AA)

    # Status label
    cv2.putText(
        image, "SINGLE_EYE", (bbox_x, bbox_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _YELLOW, 1, cv2.LINE_AA,
    )


def _draw_failed(
    image: np.ndarray, eyes: dict,
    bbox_x: int, bbox_y: int, bbox_w: int, bbox_h: int,
) -> None:
    """Draw failed detection indicator.

    Draws a yellow X at the head ROI center and a red status label.

    Args:
        image: BGR image to draw on (modified in-place).
        eyes: EyeResult dict with failure status.
        bbox_x: Bounding box x-offset.
        bbox_y: Bounding box y-offset.
        bbox_w: Bounding box width.
        bbox_h: Bounding box height.
    """
    # X marker at approximate head center
    cx = bbox_x + bbox_w // 2
    cy = bbox_y + bbox_h // 5  # top 20% of bbox
    cross_size = 8
    cv2.line(
        image,
        (cx - cross_size, cy - cross_size),
        (cx + cross_size, cy + cross_size),
        _YELLOW, 2, cv2.LINE_AA,
    )
    cv2.line(
        image,
        (cx - cross_size, cy + cross_size),
        (cx + cross_size, cy - cross_size),
        _YELLOW, 2, cv2.LINE_AA,
    )

    # Status label
    status = eyes.get("status", "FAILED")
    cv2.putText(
        image, status, (bbox_x, bbox_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _RED, 1, cv2.LINE_AA,
    )


def _draw_skeleton(
    image: np.ndarray, eyes: dict,
    score_threshold: float = 0.3,
) -> None:
    """Draw all AP-10K keypoints and skeleton edges with subtle styling.

    Uses semi-transparent overlay so the skeleton doesn't dominate the
    image. Eye keypoints (indices 0, 1) are drawn as tiny dots since
    they are already rendered prominently by :func:`_draw_success`.

    Args:
        image: BGR image to draw on (modified in-place).
        eyes: EyeResult dict, must contain ``all_keypoints`` and
            ``all_scores`` fields (only present for AI method).
        score_threshold: Minimum score to draw a keypoint.
    """
    all_kps = eyes.get("all_keypoints")
    all_scores = eyes.get("all_scores")

    if all_kps is None or all_scores is None:
        return  # CV method doesn't have full keypoints

    n_kps = len(all_kps)
    overlay = image.copy()

    # Draw skeleton edges first (behind the dots)
    for i, j in _AP10K_SKELETON:
        if i >= n_kps or j >= n_kps:
            continue
        if all_scores[i] < score_threshold or all_scores[j] < score_threshold:
            continue

        pt1 = (int(all_kps[i][0]), int(all_kps[i][1]))
        pt2 = (int(all_kps[j][0]), int(all_kps[j][1]))
        cv2.line(overlay, pt1, pt2, _SKELETON_LINE_COLOR, 1, cv2.LINE_AA)

    # Draw keypoint dots (skip eyes — they have their own prominent markers)
    for idx in range(n_kps):
        if all_scores[idx] < score_threshold:
            continue

        px, py = int(all_kps[idx][0]), int(all_kps[idx][1])

        if idx <= 1:
            # Eyes: tiny dot only (prominent markers drawn elsewhere)
            cv2.circle(overlay, (px, py), 2, _KEYPOINT_DOT_COLOR, -1, cv2.LINE_AA)
        else:
            # Other keypoints: small dot + optional tiny label
            cv2.circle(overlay, (px, py), 3, _KEYPOINT_DOT_COLOR, -1, cv2.LINE_AA)

    # Blend the overlay onto the original image
    cv2.addWeighted(overlay, _SKELETON_ALPHA, image, 1 - _SKELETON_ALPHA, 0, image)
