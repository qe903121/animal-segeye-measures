"""Heuristic eye detector using Mask-Bounded Global Cascade.

Implements a 3-stage detection pipeline that scans the ENTIRE animal
mask region (no head-position guessing) using OpenCV Haar Cascades,
with blob detection as a final fallback.

Pipeline (3-Stage Cascade Fallback):
    Stage 1: Cat Face Cascade → Eye Cascade within detected face.
    Stage 2: Global Eye Cascade → scan full mask region.
    Stage 3: Blob Detection Fallback → CLAHE + SimpleBlobDetector.

Each stage passes results through geometric validation (angle + spacing)
before accepting an eye pair.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.data.loader import EyeResult

from .base import BaseEyeDetector

logger = logging.getLogger(__name__)


class HeuristicCVDetector(BaseEyeDetector):
    """Mask-Bounded Global Cascade eye detector.

    Uses the animal's GT segmentation mask to strictly bound the search
    area, then applies a 3-stage detection cascade:

    1. **Cat Face Cascade**: ``haarcascade_frontalcatface_extended.xml``
       detects the face, then ``haarcascade_eye.xml`` finds eyes within.
    2. **Global Eye Cascade**: ``haarcascade_eye.xml`` scans the entire
       mask region at multiple scales.
    3. **Blob Fallback**: ``SimpleBlobDetector`` on CLAHE-enhanced
       grayscale as a last resort.

    All detected candidates are validated with geometric constraints
    (angle, spacing) before being accepted as an eye pair.

    Attributes:
        _cat_face_cascade: Haar cascade for cat frontal face detection.
        _eye_cascade: Haar cascade for eye detection.
    """

    def __init__(self, config: dict) -> None:
        """Initialize cascades and parameters from config.

        Args:
            config: Full parsed config dictionary with ``eye_detection``
                section containing cascade, clahe, blob, and validation
                parameters.
        """
        eye_cfg = config.get("eye_detection", {})

        # --- Load Haar Cascades (one-time) ---
        cat_face_path = (
            cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml"
        )
        eye_path = (
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self._cat_face_cascade = cv2.CascadeClassifier(cat_face_path)
        self._eye_cascade = cv2.CascadeClassifier(eye_path)

        if self._cat_face_cascade.empty():
            logger.warning("Failed to load cat face Cascade: %s", cat_face_path)
        if self._eye_cascade.empty():
            logger.warning("Failed to load eye Cascade: %s", eye_path)

        # --- Cascade parameters ---
        cascade_cfg = eye_cfg.get("cascade", {})
        self._face_scale: float = cascade_cfg.get("cat_face_scale", 1.05)
        self._face_neighbors: int = cascade_cfg.get("cat_face_neighbors", 3)
        self._eye_scale: float = cascade_cfg.get("eye_scale", 1.05)
        self._eye_neighbors: int = cascade_cfg.get("eye_neighbors", 2)
        self._eye_min_ratio: float = cascade_cfg.get(
            "eye_min_size_ratio", 0.03
        )
        self._eye_max_ratio: float = cascade_cfg.get(
            "eye_max_size_ratio", 0.25
        )

        # --- CLAHE parameters ---
        clahe_cfg = eye_cfg.get("clahe", {})
        self._clahe_clip: float = clahe_cfg.get("clip_limit", 3.0)
        self._clahe_tile: int = clahe_cfg.get("tile_size", 8)

        # --- Blob fallback parameters ---
        blob_cfg = eye_cfg.get("blob", {})
        self._min_circularity: float = blob_cfg.get("min_circularity", 0.3)
        self._min_convexity: float = blob_cfg.get("min_convexity", 0.5)
        self._area_ratio_range: list[float] = blob_cfg.get(
            "area_ratio_range", [0.002, 0.08]
        )

        # --- Geometric validation ---
        val_cfg = eye_cfg.get("validation", {})
        self._max_angle_deg: float = val_cfg.get("max_eye_angle_deg", 30.0)
        self._spacing_range: list[float] = val_cfg.get(
            "spacing_ratio_range", [0.10, 0.60]
        )

        # --- Minimum mask area for detection ---
        self._min_mask_pixels: int = eye_cfg.get("min_head_pixels", 400)

        logger.info(
            "HeuristicCVDetector initialization (Mask-Bounded Global Cascade): "
            "CLAHE(clip=%.1f, tile=%d), angle_max=%.0f°",
            self._clahe_clip,
            self._clahe_tile,
            self._max_angle_deg,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        bbox: list[float],
        mask: np.ndarray,
        category: str,
    ) -> EyeResult:
        """Detect eyes using 3-stage cascade within mask bounds.

        Args:
            image: Original BGR image, shape ``(H, W, 3)``.
            bbox: Bounding box ``[x, y, w, h]``.
            mask: Binary mask, shape ``(H, W)``, dtype ``uint8``, 0/255.
            category: Animal category name.

        Returns:
            :class:`EyeResult` with detection status and coordinates.
        """
        # Step 1: Mask-bounded crop
        crop_result = self._mask_crop(image, mask)
        if crop_result is None:
            return _make_result("FAILED_LOW_RES")

        roi_bgr, roi_mask, offset_x, offset_y = crop_result
        roi_h, roi_w = roi_bgr.shape[:2]

        # Step 2: Preprocess
        gray = self._preprocess(roi_bgr, roi_mask)

        # Step 3: 3-Stage cascade
        eye_pair: tuple[list[float], list[float]] | None = None
        stage_used = ""

        # --- Stage 1: Cat face → eyes within face ---
        eye_pair = self._stage_face_then_eyes(gray, roi_mask, roi_w)
        if eye_pair is not None:
            stage_used = "face_cascade"

        # --- Stage 2: Global eye cascade ---
        if eye_pair is None:
            eye_pair = self._stage_global_eyes(gray, roi_mask, roi_w)
            if eye_pair is not None:
                stage_used = "global_eye_cascade"

        # --- Stage 3: Blob fallback ---
        if eye_pair is None:
            eye_pair = self._stage_blob_fallback(gray, roi_mask, roi_w)
            if eye_pair is not None:
                stage_used = "blob_fallback"

        # Step 4: Build result
        if eye_pair is None:
            return _make_result("FAILED_NOT_FOUND")

        left_local, right_local = eye_pair
        left_eye = [left_local[0] + offset_x, left_local[1] + offset_y]
        right_eye = [right_local[0] + offset_x, right_local[1] + offset_y]

        confidence = self._compute_confidence(
            left_local, right_local, roi_w
        )

        logger.debug(
            "Detection successful (stage=%s, conf=%.2f): L=%s R=%s",
            stage_used, confidence, left_eye, right_eye,
        )

        return {
            "status": "SUCCESS",
            "left_eye": left_eye,
            "right_eye": right_eye,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Mask crop & preprocess
    # ------------------------------------------------------------------

    def _mask_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, int] | None:
        """Crop image to mask bounding rectangle with background removal.

        Uses the tight bounding box of the mask (not the COCO bbox)
        to extract the minimal region containing the animal. Non-animal
        pixels are set to white to reduce false positives.

        Args:
            image: Original BGR image.
            mask: Binary mask for the entire image.

        Returns:
            ``(roi_bgr, roi_mask, offset_x, offset_y)`` or ``None``
            if the mask area is below ``min_mask_pixels``.
        """
        mask_area = int(np.count_nonzero(mask))
        if mask_area < self._min_mask_pixels:
            return None

        # Get tight bounding rect of the mask
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)

        # Clip to image bounds
        img_h, img_w = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        roi_bgr = image[y1:y2, x1:x2].copy()
        roi_mask = mask[y1:y2, x1:x2].copy()

        # Set non-animal pixels to white
        bg = roi_mask == 0
        roi_bgr[bg] = [255, 255, 255]

        return roi_bgr, roi_mask, x1, y1

    def _preprocess(
        self, roi_bgr: np.ndarray, roi_mask: np.ndarray
    ) -> np.ndarray:
        """Convert to grayscale and apply CLAHE + histogram equalization.

        Args:
            roi_bgr: BGR image of the mask-cropped region.
            roi_mask: Binary mask for the region.

        Returns:
            Preprocessed grayscale image.
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip,
            tileGridSize=(self._clahe_tile, self._clahe_tile),
        )
        gray = clahe.apply(gray)

        # Mask out non-animal pixels
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

        return gray

    # ------------------------------------------------------------------
    # Stage 1: Cat Face Cascade → Eyes within face
    # ------------------------------------------------------------------

    def _stage_face_then_eyes(
        self,
        gray: np.ndarray,
        roi_mask: np.ndarray,
        roi_w: int,
    ) -> tuple[list[float], list[float]] | None:
        """Stage 1: Detect cat face, then find eyes within face region.

        Uses ``haarcascade_frontalcatface_extended.xml`` to locate the
        face, then runs ``haarcascade_eye.xml`` within the detected
        face bounding box.

        Args:
            gray: Preprocessed grayscale ROI.
            roi_mask: Binary mask for ROI.
            roi_w: Width of the ROI.

        Returns:
            ``(left_eye, right_eye)`` in ROI-local coords, or ``None``.
        """
        if self._cat_face_cascade.empty():
            return None

        roi_h, roi_w_actual = gray.shape[:2]
        min_face_size = max(20, int(min(roi_h, roi_w_actual) * 0.15))

        faces = self._cat_face_cascade.detectMultiScale(
            gray,
            scaleFactor=self._face_scale,
            minNeighbors=self._face_neighbors,
            minSize=(min_face_size, min_face_size),
        )

        if len(faces) == 0:
            return None

        # Use the largest detected face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        for fx, fy, fw, fh in faces_sorted:
            face_gray = gray[fy : fy + fh, fx : fx + fw]

            min_eye = max(5, int(fw * self._eye_min_ratio * 3))
            max_eye = max(min_eye + 1, int(fw * self._eye_max_ratio * 3))

            eyes = self._eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=self._eye_scale,
                minNeighbors=self._eye_neighbors,
                minSize=(min_eye, min_eye),
                maxSize=(max_eye, max_eye),
            )

            if len(eyes) >= 2:
                # Convert eye rects to center points (face-local → ROI-local)
                candidates = []
                for ex, ey, ew, eh in eyes:
                    cx = float(fx + ex + ew / 2)
                    cy = float(fy + ey + eh / 2)
                    candidates.append([cx, cy])

                pair = self._validate_and_select(candidates, roi_w)
                if pair is not None:
                    return pair

        return None

    # ------------------------------------------------------------------
    # Stage 2: Global Eye Cascade
    # ------------------------------------------------------------------

    def _stage_global_eyes(
        self,
        gray: np.ndarray,
        roi_mask: np.ndarray,
        roi_w: int,
    ) -> tuple[list[float], list[float]] | None:
        """Stage 2: Run eye cascade on the entire mask region.

        Scans the full mask-bounded ROI with ``haarcascade_eye.xml``
        at multiple scales. All detections are filtered to ensure they
        lie within the animal mask, then validated geometrically.

        Args:
            gray: Preprocessed grayscale ROI.
            roi_mask: Binary mask for ROI.
            roi_w: Width of the ROI.

        Returns:
            ``(left_eye, right_eye)`` in ROI-local coords, or ``None``.
        """
        if self._eye_cascade.empty():
            return None

        roi_h, roi_w_actual = gray.shape[:2]
        min_eye = max(5, int(min(roi_h, roi_w_actual) * self._eye_min_ratio))
        max_eye = max(
            min_eye + 1,
            int(min(roi_h, roi_w_actual) * self._eye_max_ratio),
        )

        eyes = self._eye_cascade.detectMultiScale(
            gray,
            scaleFactor=self._eye_scale,
            minNeighbors=self._eye_neighbors,
            minSize=(min_eye, min_eye),
            maxSize=(max_eye, max_eye),
        )

        if len(eyes) < 2:
            return None

        # Convert to center points, filter by mask
        candidates: list[list[float]] = []
        for ex, ey, ew, eh in eyes:
            cx = float(ex + ew / 2)
            cy = float(ey + eh / 2)
            # Check if center is within animal mask
            ix, iy = int(cx), int(cy)
            if 0 <= iy < roi_mask.shape[0] and 0 <= ix < roi_mask.shape[1]:
                if roi_mask[iy, ix] > 0:
                    candidates.append([cx, cy])

        if len(candidates) < 2:
            return None

        return self._validate_and_select(candidates, roi_w)

    # ------------------------------------------------------------------
    # Stage 3: Blob Detection Fallback
    # ------------------------------------------------------------------

    def _stage_blob_fallback(
        self,
        gray: np.ndarray,
        roi_mask: np.ndarray,
        roi_w: int,
    ) -> tuple[list[float], list[float]] | None:
        """Stage 3: Blob detection on full mask region as last resort.

        Uses ``SimpleBlobDetector`` to find dark circular features in
        the CLAHE-enhanced grayscale image, constrained to the mask.

        Args:
            gray: Preprocessed grayscale ROI.
            roi_mask: Binary mask for ROI.
            roi_w: Width of the ROI.

        Returns:
            ``(left_eye, right_eye)`` in ROI-local coords, or ``None``.
        """
        roi_area = max(1, int(np.count_nonzero(roi_mask)))
        min_blob = max(4, int(roi_area * self._area_ratio_range[0]))
        max_blob = max(min_blob + 1, int(roi_area * self._area_ratio_range[1]))

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = float(min_blob)
        params.maxArea = float(max_blob)
        params.filterByCircularity = True
        params.minCircularity = self._min_circularity
        params.filterByConvexity = True
        params.minConvexity = self._min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = 0.2

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        if len(keypoints) < 2:
            return None

        # Filter by mask and convert to center points
        candidates: list[list[float]] = []
        for kp in sorted(keypoints, key=lambda k: k.size, reverse=True):
            cx, cy = kp.pt
            ix, iy = int(cx), int(cy)
            if 0 <= iy < roi_mask.shape[0] and 0 <= ix < roi_mask.shape[1]:
                if roi_mask[iy, ix] > 0:
                    candidates.append([cx, cy])

        if len(candidates) < 2:
            return None

        return self._validate_and_select(candidates, roi_w)

    # ------------------------------------------------------------------
    # Geometric validation (shared across all stages)
    # ------------------------------------------------------------------

    def _validate_and_select(
        self,
        candidates: list[list[float]],
        roi_w: int,
    ) -> tuple[list[float], list[float]] | None:
        """Select the best eye pair from candidate points.

        Evaluates all pairs against geometric constraints:
            1. Angle between pair ≤ ``max_eye_angle_deg``.
            2. Spacing within ``spacing_ratio_range`` of ROI width.
            3. Best combined score wins.

        Args:
            candidates: List of ``[x, y]`` candidate eye positions.
            roi_w: Width of the ROI for spacing normalization.

        Returns:
            ``(left_eye, right_eye)`` where left has smaller x,
            or ``None`` if no valid pair found.
        """
        if len(candidates) < 2:
            return None

        best_pair: tuple[list[float], list[float]] | None = None
        best_score = float("inf")

        top_n = min(len(candidates), 10)
        for pt_a, pt_b in combinations(candidates[:top_n], 2):
            ax, ay = pt_a
            bx, by = pt_b

            dx = abs(ax - bx)
            dy = abs(ay - by)
            distance = math.hypot(dx, dy)

            # Spacing constraint
            spacing_ratio = distance / max(roi_w, 1)
            if not (
                self._spacing_range[0]
                <= spacing_ratio
                <= self._spacing_range[1]
            ):
                continue

            # Angle constraint
            angle_deg = abs(math.degrees(math.atan2(dy, max(dx, 1e-6))))
            if angle_deg > self._max_angle_deg:
                continue

            # Score (lower is better)
            ideal_spacing = 0.25
            spacing_penalty = abs(spacing_ratio - ideal_spacing)
            angle_penalty = angle_deg / self._max_angle_deg
            score = angle_penalty + spacing_penalty

            if score < best_score:
                best_score = score
                if ax < bx:
                    best_pair = ([ax, ay], [bx, by])
                else:
                    best_pair = ([bx, by], [ax, ay])

        return best_pair

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        left: list[float],
        right: list[float],
        roi_w: int,
    ) -> float:
        """Compute heuristic confidence for the detected eye pair.

        Based on angle deviation from horizontal and spacing proximity
        to the ideal ratio (~25% of ROI width).

        Args:
            left: Left eye ``[x, y]`` in ROI-local coordinates.
            right: Right eye ``[x, y]`` in ROI-local coordinates.
            roi_w: Width of the ROI.

        Returns:
            Confidence in ``[0.0, 1.0]``.
        """
        dx = abs(right[0] - left[0])
        dy = abs(right[1] - left[1])

        angle_deg = abs(math.degrees(math.atan2(dy, max(dx, 1e-6))))
        angle_score = max(0.0, 1.0 - angle_deg / self._max_angle_deg)

        distance = math.hypot(dx, dy)
        spacing_ratio = distance / max(roi_w, 1)
        ideal = 0.25
        spacing_score = max(0.0, 1.0 - abs(spacing_ratio - ideal) / ideal)

        confidence = 0.6 * angle_score + 0.4 * spacing_score
        return round(min(1.0, max(0.0, confidence)), 3)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _make_result(status: str) -> EyeResult:
    """Create a failed EyeResult with the given status.

    Args:
        status: Failure status string.

    Returns:
        EyeResult dict with ``None`` coordinates and zero confidence.
    """
    return {
        "status": status,
        "left_eye": None,
        "right_eye": None,
        "confidence": 0.0,
    }
