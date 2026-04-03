"""Dataset asset exporter for standardized Phase 1 outputs.

Transforms the in-memory filtered dataset into a persistent dataset
asset directory that can be reused by later ground-truth labeling,
prediction, and evaluation stages.

Exported artefacts:
    - ``manifest.json``: dataset-level metadata and filtering context
    - ``instances.csv``: one row per animal instance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .loader import ImageRecord

logger = logging.getLogger(__name__)

INSTANCE_COLUMNS = [
    "dataset_id",
    "image_id",
    "image_path",
    "image_rel_path",
    "image_width",
    "image_height",
    "annotation_id",
    "category",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_area_px",
]


@dataclass(frozen=True)
class DatasetAssetInfo:
    """Summary information for an exported dataset asset."""

    dataset_id: str
    asset_dir: Path
    manifest_path: Path
    instances_path: Path


class DatasetAssetExporter:
    """Export a Phase 1 dataset into a reusable asset directory."""

    def __init__(self, config: dict) -> None:
        """Initialize the exporter from global config."""
        self._config = config
        asset_cfg = config.get("assets", {})
        self._dataset_root = Path(
            asset_cfg.get("dataset_root", "assets/datasets")
        )
        self._schema_version = int(asset_cfg.get("schema_version", 1))

    def export(
        self,
        dataset: list[ImageRecord],
        target_categories: list[str],
        dataset_id: str | None = None,
    ) -> DatasetAssetInfo:
        """Persist manifest and per-instance table for the dataset.

        Args:
            dataset: Filtered Phase 1 dataset.
            target_categories: Requested target categories for this run.
            dataset_id: Optional fixed dataset id. When omitted, a
                deterministic id is derived from source/filter settings
                plus the exact exported dataset membership.

        Returns:
            A :class:`DatasetAssetInfo` describing the exported paths.
        """
        rows = self._build_instance_rows(dataset)
        resolved_dataset_id = dataset_id or self._build_dataset_id(
            target_categories,
            rows,
        )
        for row in rows:
            row["dataset_id"] = resolved_dataset_id

        asset_dir = self._dataset_root / resolved_dataset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = asset_dir / "manifest.json"
        instances_path = asset_dir / "instances.csv"

        manifest = self._build_manifest(
            dataset,
            resolved_dataset_id,
            target_categories,
            rows,
        )

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(rows, columns=INSTANCE_COLUMNS).to_csv(
            instances_path, index=False, encoding="utf-8"
        )

        logger.info("已匯出 Dataset Asset: %s", asset_dir)
        logger.info("  manifest:  %s", manifest_path)
        logger.info("  instances: %s (%d 筆)", instances_path, len(rows))

        return DatasetAssetInfo(
            dataset_id=resolved_dataset_id,
            asset_dir=asset_dir,
            manifest_path=manifest_path,
            instances_path=instances_path,
        )

    def _build_dataset_id(
        self,
        target_categories: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        """Build a deterministic dataset id from config and dataset membership."""
        coco_cfg = self._config.get("coco", {})
        filter_cfg = self._config.get("filtering", {})

        source_slug = _slugify(coco_cfg.get("images_dir", "coco"))
        cats_slug = "-".join(_slugify(cat) for cat in sorted(target_categories))
        if not cats_slug:
            cats_slug = "uncategorized"

        payload = {
            "images_dir": coco_cfg.get("images_dir", "val2017"),
            "annotations_file": coco_cfg.get(
                "annotations_file", "instances_val2017.json"
            ),
            "target_categories": sorted(target_categories),
            "min_instances": filter_cfg.get("min_instances", 2),
            "min_categories": filter_cfg.get("min_categories", 2),
            "min_area": filter_cfg.get("min_area", 2000),
            "exclude_crowd": filter_cfg.get("exclude_crowd", True),
            "max_overlap_ratio": filter_cfg.get("max_overlap_ratio", 0.8),
            "membership": self._build_membership_signature(rows),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]

        return f"coco_{source_slug}_{cats_slug}_{digest}"

    def _build_manifest(
        self,
        dataset: list[ImageRecord],
        dataset_id: str,
        target_categories: list[str],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble dataset-level metadata for manifest.json."""
        filter_cfg = self._config.get("filtering", {})
        coco_cfg = self._config.get("coco", {})

        present_categories = sorted({
            ann["category"]
            for record in dataset
            for ann in record["annotations"]
        })
        n_annotations = sum(len(record["annotations"]) for record in dataset)

        return {
            "dataset_id": dataset_id,
            "source": "COCO val2017",
            "created_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "schema_version": self._schema_version,
            "requested_categories": sorted(target_categories),
            "present_categories": present_categories,
            "membership_digest": self._build_membership_digest(rows),
            "filtering": {
                "min_instances": filter_cfg.get("min_instances", 2),
                "min_categories": filter_cfg.get("min_categories", 2),
                "min_area": filter_cfg.get("min_area", 2000),
                "exclude_crowd": filter_cfg.get("exclude_crowd", True),
                "max_overlap_ratio": filter_cfg.get("max_overlap_ratio", 0.8),
            },
            "coco": {
                "data_root": coco_cfg.get("data_root", "data/coco"),
                "images_dir": coco_cfg.get("images_dir", "val2017"),
                "annotations_file": coco_cfg.get(
                    "annotations_file", "instances_val2017.json"
                ),
            },
            "n_images": len(dataset),
            "n_annotations": n_annotations,
        }

    def _build_instance_rows(
        self,
        dataset: list[ImageRecord],
    ) -> list[dict[str, Any]]:
        """Flatten the dataset into one CSV row per animal instance."""
        rows: list[dict[str, Any]] = []

        for record in dataset:
            image_width, image_height = record["image_size"]
            image_path = str(record["image_path"])
            image_rel_path = self._resolve_image_rel_path(Path(image_path))
            for ann in record["annotations"]:
                bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
                rows.append({
                    "dataset_id": "",
                    "image_id": record["image_id"],
                    "image_path": image_path,
                    "image_rel_path": image_rel_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "annotation_id": ann["id"],
                    "category": ann["category"],
                    "bbox_x": round(float(bbox_x), 3),
                    "bbox_y": round(float(bbox_y), 3),
                    "bbox_w": round(float(bbox_w), 3),
                    "bbox_h": round(float(bbox_h), 3),
                    "bbox_area_px": round(float(bbox_w) * float(bbox_h), 3),
                })

        return rows

    def _build_membership_signature(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, int]]:
        """Return a stable per-instance signature for dataset identity."""
        return [
            {
                "image_id": int(row["image_id"]),
                "annotation_id": int(row["annotation_id"]),
            }
            for row in sorted(
                rows,
                key=lambda row: (int(row["image_id"]), int(row["annotation_id"])),
            )
        ]

    def _build_membership_digest(self, rows: list[dict[str, Any]]) -> str:
        """Return a compact digest of exported dataset membership."""
        signature = self._build_membership_signature(rows)
        return hashlib.sha1(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    def _resolve_image_rel_path(self, image_path: Path) -> str:
        """Resolve image path relative to COCO ``data_root`` when possible."""
        coco_cfg = self._config.get("coco", {})
        data_root = Path(coco_cfg.get("data_root", "data/coco"))
        try:
            return str(image_path.resolve().relative_to(data_root.resolve()))
        except ValueError:
            try:
                return str(image_path.relative_to(data_root))
            except ValueError:
                return image_path.name


def _slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "na"
