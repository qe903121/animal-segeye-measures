"""COCO Ground Truth data loader with multi-animal filtering.

Loads COCO val2017 annotations, applies configurable filtering rules
(multi-instance, multi-category, min area, crowd exclusion, mask overlap), and produces
standardized ImageRecord dicts with unified binary masks and contours.

Typical usage:
    >>> from src.data.loader import COCODataLoader
    >>> loader = COCODataLoader("data/coco", target_categories=["dog", "cat"])
    >>> dataset = loader.load_filtered_dataset()
    >>> print(f"Found {len(dataset)} qualifying images")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from typing_extensions import NotRequired

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from .downloader import AutoDownloader

logger = logging.getLogger(__name__)


# ======================================================================
# Standardized output types
# ======================================================================


class EyeResult(TypedDict):
    """Eye detection result for a single animal instance.

    Attributes:
        status: Detection outcome. One of:
            - ``"SUCCESS"``: Both eyes located.
            - ``"SINGLE_EYE"``: Only one eye detected.
            - ``"FAILED_NOT_FOUND"``: No eyes detected.
            - ``"FAILED_LOW_RES"``: Head ROI resolution too low.
        left_eye: ``[x, y]`` pixel coordinates, or ``None`` if not found.
        right_eye: ``[x, y]`` pixel coordinates, or ``None`` if not found.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
            CV method uses a heuristic geometric score;
            AI method (future) will use model confidence.
    """

    status: str
    left_eye: list[float] | None
    right_eye: list[float] | None
    confidence: float
    all_keypoints: NotRequired[list[list[float]]]
    all_scores: NotRequired[list[float]]


class AnnotationEntry(TypedDict):
    """Standardized structure for a single animal annotation.

    Attributes:
        id: COCO annotation ID.
        category: Human-readable category name (e.g., ``"dog"``).
        bbox: Bounding box as ``[x, y, width, height]``.
        mask: Binary mask, shape ``(H, W)``, dtype ``uint8``, values 0 or 255.
        contours: OpenCV-format contour arrays from ``cv2.findContours``.
        eyes: Eye detection result (populated by prediction-side localization). Optional.
    """

    id: int
    category: str
    bbox: list[float]
    mask: np.ndarray
    contours: list[np.ndarray]
    eyes: NotRequired[EyeResult]


class ImageRecord(TypedDict):
    """Standardized data structure for a single qualifying image.

    Attributes:
        image_id: COCO image ID.
        image_path: Filesystem path to the JPEG image.
        image_size: ``(width, height)`` tuple.
        annotations: List of animal annotations that passed all filters.
    """

    image_id: int
    image_path: str
    image_size: tuple[int, int]
    annotations: list[AnnotationEntry]


# ======================================================================
# Main loader class
# ======================================================================


class COCODataLoader:
    """COCO Ground-Truth data loader and multi-animal filter.

    Encapsulates ``pycocotools.coco.COCO`` and applies business-rule
    filtering to identify images containing multiple animals of different
    species. All segmentation masks are unified to binary ``np.ndarray``
    format regardless of the original COCO encoding (polygon or RLE).

    Attributes:
        coco: Underlying ``pycocotools`` COCO API instance.
        target_categories: List of target animal category names.
        data_root: Path to the COCO data root directory.
    """

    def __init__(
        self,
        data_root: str,
        target_categories: list[str] | None = None,
        config: dict | None = None,
        auto_download: bool = True,
    ) -> None:
        """Initialize the loader, triggering download check if needed.

        Args:
            data_root: Root directory containing COCO data (e.g.,
                ``"data/coco"``).
            target_categories: Animal category names to filter for. When
                ``None``, uses the default list from ``config.yaml``.
            config: Full parsed config dictionary. When ``None``, a
                minimal default config is used.
            auto_download: If True, call :class:`AutoDownloader` before
                opening the COCO annotation file. Set to False when the
                caller has already handled download readiness explicitly.

        Raises:
            FileNotFoundError: Annotation file missing and download failed.
            ValueError: A category name doesn't exist in COCO.
        """
        self.data_root = Path(data_root)
        self._config = config or {}
        filter_cfg = self._config.get("filtering", {})

        # Resolve target categories
        if target_categories is not None:
            self.target_categories = target_categories
        else:
            self.target_categories = filter_cfg.get(
                "default_animal_categories",
                ["dog", "cat", "bird", "horse", "sheep",
                 "cow", "elephant", "bear", "zebra", "giraffe"],
            )

        # Filtering thresholds
        self._min_instances: int = filter_cfg.get("min_instances", 2)
        self._min_categories: int = filter_cfg.get("min_categories", 2)
        self._min_area: int = filter_cfg.get("min_area", 2000)
        self._exclude_crowd: bool = filter_cfg.get("exclude_crowd", True)
        self._max_overlap_ratio: float = filter_cfg.get(
            "max_overlap_ratio", 0.8
        )

        # Resolved paths
        coco_cfg = self._config.get("coco", {})
        self._images_dir = coco_cfg.get("images_dir", "val2017")
        self._ann_file = coco_cfg.get(
            "annotations_file", "instances_val2017.json"
        )

        # Ensure data is downloaded before loading unless the caller
        # already handled readiness (for example via CLI --skip-download).
        if auto_download:
            downloader = AutoDownloader(self._config)
            downloader.ensure_ready()

        # Initialize pycocotools COCO API
        ann_path = self.data_root / "annotations" / self._ann_file
        if not ann_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}. Verify download succeeded."
            )
        logger.info("Loading COCO annotations: %s", ann_path)
        self.coco = COCO(str(ann_path))

        # Build category ID ↔ name mapping
        self._cat_id_to_name = self._resolve_category_ids(
            self.target_categories
        )
        self._target_cat_ids = list(self._cat_id_to_name.keys())

        logger.info(
            "Target categories parsed: %s (total %d classes)",
            self._cat_id_to_name,
            len(self._cat_id_to_name),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_filtered_dataset(self) -> list[ImageRecord]:
        """Execute the full filtering pipeline and return standardized data.

        Pipeline steps:
            1. Collect all image IDs containing any target category.
            2. For each image, retrieve annotations and apply filters:
               a. Exclude ``iscrowd=1`` annotations.
               b. Exclude annotations below ``min_area`` (bbox area).
               c. Require ``>= min_instances`` qualifying annotations.
               d. Require ``>= min_categories`` distinct categories.
            3. Decode segmentation masks (polygon/RLE → binary ndarray).
            4. Exclude annotations with excessive pairwise mask overlap.
            5. Re-validate instance/category counts after overlap removal.
            6. Extract contours via ``cv2.findContours``.

        Returns:
            List of :class:`ImageRecord` dicts that passed all filters.
        """
        # Step 1: Collect candidate image IDs
        candidate_img_ids: set[int] = set()
        for cat_id in self._target_cat_ids:
            candidate_img_ids.update(self.coco.getImgIds(catIds=[cat_id]))

        logger.info(
            "Candidate images (contain any target category): %d", len(candidate_img_ids)
        )

        # Step 2-6: Filter, decode, overlap check, and build records
        dataset: list[ImageRecord] = []
        skipped = 0
        overlap_removed_total = 0

        for img_id in sorted(candidate_img_ids):
            img_info = self.coco.loadImgs(img_id)[0]
            height: int = img_info["height"]
            width: int = img_info["width"]

            ann_ids = self.coco.getAnnIds(
                imgIds=img_id, catIds=self._target_cat_ids
            )
            raw_anns = self.coco.loadAnns(ann_ids)

            # Basic filters (crowd, area, counts)
            filtered = self._filter_image(img_id, raw_anns)
            if filtered is None:
                skipped += 1
                continue

            # Decode masks for all passing annotations
            decoded: list[tuple[dict, np.ndarray]] = []
            for ann in filtered:
                mask = self._decode_mask(ann, height, width)
                decoded.append((ann, mask))

            # Overlap filter: remove heavily occluded annotations
            decoded, n_removed = self._filter_overlap(decoded, img_id)
            overlap_removed_total += n_removed

            # Re-validate instance/category counts after overlap removal
            if len(decoded) < self._min_instances:
                skipped += 1
                continue
            remaining_cats = {
                ann["category_id"] for ann, _ in decoded
            } & set(self._target_cat_ids)
            if len(remaining_cats) < self._min_categories:
                skipped += 1
                continue

            # Build final annotation entries with contours
            annotations: list[AnnotationEntry] = []
            for ann, mask in decoded:
                contours = self._extract_contours(mask)
                annotations.append(
                    AnnotationEntry(
                        id=ann["id"],
                        category=self._cat_id_to_name[ann["category_id"]],
                        bbox=ann["bbox"],
                        mask=mask,
                        contours=contours,
                    )
                )

            image_path = str(
                self.data_root / self._images_dir / img_info["file_name"]
            )

            dataset.append(
                ImageRecord(
                    image_id=img_id,
                    image_path=image_path,
                    image_size=(width, height),
                    annotations=annotations,
                )
            )

        logger.info(
            "Filtering complete: %d images passed / %d excluded (out of %d candidates)",
            len(dataset),
            skipped,
            len(candidate_img_ids),
        )
        if overlap_removed_total > 0:
            logger.info(
                "Overlap filter: Removed %d highly occluded annotations (threshold: %.0f%%)",
                overlap_removed_total,
                self._max_overlap_ratio * 100,
            )
        return dataset

    def export_csv(
        self, dataset: list[ImageRecord], output_path: str
    ) -> None:
        """Export a baseline index CSV for downstream evaluation.

        CSV columns:
            - ``image_id``: COCO image ID.
            - ``image_path``: Filesystem path to the image.
            - ``num_instances``: Number of animal instances.
            - ``num_categories``: Number of distinct animal categories.
            - ``categories``: Comma-separated category names.

        Args:
            dataset: Output from :meth:`load_filtered_dataset`.
            output_path: Destination CSV file path.
        """
        rows: list[dict[str, Any]] = []
        for record in dataset:
            cats = sorted(
                {ann["category"] for ann in record["annotations"]}
            )
            rows.append(
                {
                    "image_id": record["image_id"],
                    "image_path": record["image_path"],
                    "num_instances": len(record["annotations"]),
                    "num_categories": len(cats),
                    "categories": ",".join(cats),
                }
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)
        df.to_csv(output, index=False, encoding="utf-8")
        logger.info("Exported baseline index: %s (%d records)", output, len(df))

    def load_dataset_from_instances(
        self,
        instances: pd.DataFrame,
    ) -> list[ImageRecord]:
        """Rebuild a standardized dataset from exported instance rows.

        This path preserves a frozen Dataset Asset membership by using
        the exact ``image_id`` / ``annotation_id`` pairs from
        ``instances.csv`` instead of re-running dataset filtering logic.

        Args:
            instances: Dataset asset per-instance table.

        Returns:
            Reconstructed :class:`ImageRecord` list with masks and
            contours restored from COCO annotations.
        """
        required_columns = {"image_id", "annotation_id"}
        missing = required_columns - set(instances.columns)
        if missing:
            raise ValueError(
                "instances.csv is missing necessary columns: "
                f"{sorted(missing)}"
            )

        if instances.empty:
            logger.warning("instances.csv is empty, returning empty dataset.")
            return []

        dataset: list[ImageRecord] = []
        grouped = instances.sort_values(
            by=["image_id", "annotation_id"],
            kind="stable",
        ).groupby("image_id", sort=True)

        for image_id, group in grouped:
            image_id_int = int(image_id)
            img_infos = self.coco.loadImgs([image_id_int])
            if not img_infos:
                logger.warning("image_id=%d not found in COCO labels, skipping.", image_id_int)
                continue

            img_info = img_infos[0]
            height: int = img_info["height"]
            width: int = img_info["width"]

            requested_ann_ids = [int(v) for v in group["annotation_id"].tolist()]
            raw_anns = self.coco.loadAnns(requested_ann_ids)
            raw_ann_by_id = {int(ann["id"]): ann for ann in raw_anns}

            missing_ann_ids = [
                ann_id for ann_id in requested_ann_ids
                if ann_id not in raw_ann_by_id
            ]
            if missing_ann_ids:
                logger.warning(
                    "image_id=%d has %d annotations absent from COCO labels: %s",
                    image_id_int,
                    len(missing_ann_ids),
                    missing_ann_ids,
                )

            annotations: list[AnnotationEntry] = []
            for ann_id in requested_ann_ids:
                ann = raw_ann_by_id.get(ann_id)
                if ann is None:
                    continue

                mask = self._decode_mask(ann, height, width)
                contours = self._extract_contours(mask)
                category_name = self._cat_id_to_name.get(
                    ann["category_id"],
                    self.coco.loadCats([ann["category_id"]])[0]["name"],
                )
                annotations.append(
                    AnnotationEntry(
                        id=ann["id"],
                        category=category_name,
                        bbox=ann["bbox"],
                        mask=mask,
                        contours=contours,
                    )
                )

            if not annotations:
                logger.warning("image_id=%d has no usable annotations, skipping.", image_id_int)
                continue

            image_path = str(
                self.data_root / self._images_dir / img_info["file_name"]
            )
            dataset.append(
                ImageRecord(
                    image_id=image_id_int,
                    image_path=image_path,
                    image_size=(width, height),
                    annotations=annotations,
                )
            )

        total_annotations = sum(len(record["annotations"]) for record in dataset)
        logger.info(
            "Reconstructed dataset from Dataset Asset: %d images / %d annotations",
            len(dataset),
            total_annotations,
        )
        return dataset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_category_ids(
        self, names: list[str]
    ) -> dict[int, str]:
        """Map category names to COCO category IDs.

        Args:
            names: List of category names (e.g., ``["dog", "cat"]``).

        Returns:
            Mapping of ``{category_id: category_name}``.

        Raises:
            ValueError: One or more names not found in COCO categories.
        """
        cat_id_map: dict[int, str] = {}
        missing: list[str] = []

        for name in names:
            ids = self.coco.getCatIds(catNms=[name])
            if not ids:
                missing.append(name)
            else:
                cat_id_map[ids[0]] = name

        if missing:
            all_cats = [c["name"] for c in self.coco.loadCats(self.coco.getCatIds())]
            raise ValueError(
                f"The following categories do not exist in the COCO dataset: {missing}."
                f" Available classes: {all_cats}"
            )

        return cat_id_map

    def _filter_image(
        self, image_id: int, annotations: list[dict]
    ) -> list[dict] | None:
        """Apply all filtering rules to a single image's annotations.

        Args:
            image_id: COCO image ID (for logging).
            annotations: Raw COCO annotation dicts for this image.

        Returns:
            List of annotations passing all filters, or ``None`` if the
            image should be excluded entirely.
        """
        # (a) Exclude crowd annotations
        if self._exclude_crowd:
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0
            ]

        # (b) Exclude by min bbox area (w * h)
        annotations = [
            ann
            for ann in annotations
            if ann["bbox"][2] * ann["bbox"][3] >= self._min_area
        ]

        # (c) Check minimum instance count
        if len(annotations) < self._min_instances:
            return None

        # (d) Check minimum category diversity
        unique_cats = {ann["category_id"] for ann in annotations}
        target_cats_in_image = unique_cats & set(self._target_cat_ids)
        if len(target_cats_in_image) < self._min_categories:
            return None

        return annotations

    def _filter_overlap(
        self,
        decoded: list[tuple[dict, np.ndarray]],
        image_id: int,
    ) -> tuple[list[tuple[dict, np.ndarray]], int]:
        """Remove annotations whose masks are heavily overlapped by others.

        For each pair of annotations, computes the overlap ratio as:
        ``intersection_pixels / smaller_mask_pixels``. If this exceeds
        ``max_overlap_ratio``, the annotation with the smaller mask area
        is removed (it is likely heavily occluded).

        Args:
            decoded: List of ``(raw_annotation, binary_mask)`` tuples.
            image_id: COCO image ID (for logging).

        Returns:
            Tuple of (filtered list, number of removed annotations).
        """
        if self._max_overlap_ratio >= 1.0 or len(decoded) < 2:
            return decoded, 0

        # Pre-compute mask areas (count of non-zero pixels)
        areas = [int(np.count_nonzero(mask)) for _, mask in decoded]
        remove_indices: set[int] = set()

        for i in range(len(decoded)):
            if i in remove_indices:
                continue
            for j in range(i + 1, len(decoded)):
                if j in remove_indices:
                    continue

                # Compute pixel-level intersection
                mask_i = decoded[i][1]
                mask_j = decoded[j][1]
                intersection = int(np.count_nonzero(
                    cv2.bitwise_and(mask_i, mask_j)
                ))

                if intersection == 0:
                    continue

                # Overlap ratio = intersection / smaller mask area
                smaller_area = min(areas[i], areas[j])
                if smaller_area == 0:
                    continue

                overlap_ratio = intersection / smaller_area

                if overlap_ratio > self._max_overlap_ratio:
                    # Remove the smaller (more occluded) annotation
                    victim = i if areas[i] <= areas[j] else j
                    remove_indices.add(victim)
                    victim_ann = decoded[victim][0]
                    logger.debug(
                        "Image %d: Removing highly occluded annotation #%d (%s), "
                        "Overlap ratio %.1f%% > %.0f%%",
                        image_id,
                        victim_ann["id"],
                        self._cat_id_to_name.get(
                            victim_ann["category_id"], "unknown"
                        ),
                        overlap_ratio * 100,
                        self._max_overlap_ratio * 100,
                    )

        if not remove_indices:
            return decoded, 0

        filtered = [
            item for idx, item in enumerate(decoded)
            if idx not in remove_indices
        ]
        return filtered, len(remove_indices)

    def _decode_mask(
        self, annotation: dict, height: int, width: int
    ) -> np.ndarray:
        """Decode a COCO segmentation annotation into a binary mask.

        Handles both polygon and RLE (Run-Length Encoding) formats.

        Args:
            annotation: Single COCO annotation dict containing
                ``"segmentation"`` field.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Binary mask of shape ``(height, width)`` with dtype ``uint8``.
            Foreground pixels are 255, background pixels are 0.
        """
        seg = annotation["segmentation"]

        if isinstance(seg, list):
            # Polygon format → convert to RLE first
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            # Already in RLE format
            rle = seg
        else:
            raise TypeError(
                f"Unsupported segmentation format: {type(seg)} "
                f"(annotation ID: {annotation['id']})"
            )

        binary_mask = mask_utils.decode(rle)  # shape: (H, W), uint8, 0/1
        return (binary_mask * 255).astype(np.uint8)

    def _extract_contours(
        self, mask: np.ndarray
    ) -> list[np.ndarray]:
        """Extract external contours from a binary mask.

        Uses ``cv2.RETR_EXTERNAL`` to retrieve only the outermost contours.

        Args:
            mask: Binary mask, shape ``(H, W)``, dtype ``uint8``,
                values 0 or 255.

        Returns:
            List of contour arrays in OpenCV format.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)
