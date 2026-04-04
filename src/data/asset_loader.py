"""Dataset asset loader for Phase 1 exported assets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .loader import ImageRecord, AnnotationEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetAsset:
    """Loaded dataset asset bundle."""

    dataset_id: str
    asset_dir: Path
    manifest: dict[str, Any]
    instances: pd.DataFrame


class DatasetAssetLoader:
    """Read exported dataset assets from ``assets/datasets``."""

    def __init__(self, config: dict) -> None:
        """Initialize from global config."""
        asset_cfg = config.get("assets", {})
        self._dataset_root = Path(
            asset_cfg.get("dataset_root", "assets/datasets")
        )

    @property
    def dataset_root(self) -> Path:
        """Return root directory containing dataset assets."""
        return self._dataset_root

    def list_dataset_ids(self) -> list[str]:
        """Return available dataset ids sorted by name."""
        if not self._dataset_root.is_dir():
            return []

        dataset_ids: list[str] = []
        for child in sorted(self._dataset_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "manifest.json").is_file() and (child / "instances.csv").is_file():
                dataset_ids.append(child.name)
        return dataset_ids

    def load(self, dataset_id: str) -> DatasetAsset:
        """Load one dataset asset by id."""
        asset_dir = self._dataset_root / dataset_id
        manifest_path = asset_dir / "manifest.json"
        instances_path = asset_dir / "instances.csv"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.json 不存在: {manifest_path}")
        if not instances_path.is_file():
            raise FileNotFoundError(f"instances.csv 不存在: {instances_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        instances = pd.read_csv(instances_path)
        if "dataset_id" not in instances.columns:
            instances["dataset_id"] = dataset_id
        if "image_rel_path" not in instances.columns:
            instances["image_rel_path"] = instances["image_path"].map(
                lambda value: _derive_image_rel_path(
                    str(value),
                    Path(str(manifest.get("coco", {}).get("data_root", "data/coco"))),
                )
            )
        instances = instances.sort_values(
            by=["image_id", "annotation_id"],
            kind="stable",
        ).reset_index(drop=True)
        logger.info(
            "已載入 Dataset Asset: %s (%d images / %d instances)",
            dataset_id,
            manifest.get("n_images", -1),
            len(instances),
        )

        return DatasetAsset(
            dataset_id=dataset_id,
            asset_dir=asset_dir,
            manifest=manifest,
            instances=instances,
        )


def build_lightweight_dataset_from_asset(asset: DatasetAsset) -> list["ImageRecord"]:
    """Build a lightweight runtime dataset directly from one Dataset Asset.

    This path intentionally avoids reloading raw COCO annotations. It is meant
    for asset-driven flows such as GT validation where annotation metadata
    (image id, category, bbox) is sufficient and segmentation masks /
    contours are not required.
    """
    if asset.instances.empty:
        return []

    dataset: list["ImageRecord"] = []
    grouped = asset.instances.sort_values(
        by=["image_id", "annotation_id"],
        kind="stable",
    ).groupby("image_id", sort=True)

    for image_id, group in grouped:
        first = group.iloc[0]
        image_path = _resolve_asset_image_path(asset.manifest, first)
        image_width = int(first.get("image_width", 0) or 0)
        image_height = int(first.get("image_height", 0) or 0)

        annotations: list["AnnotationEntry"] = []
        for _, row in group.iterrows():
            bbox = [
                float(row["bbox_x"]),
                float(row["bbox_y"]),
                float(row["bbox_w"]),
                float(row["bbox_h"]),
            ]
            annotations.append({
                "id": int(row["annotation_id"]),
                "category": str(row["category"]),
                "bbox": bbox,
                # Lightweight placeholder: validate path does not require
                # full segmentation reconstruction.
                "mask": np.zeros((1, 1), dtype=np.uint8),
                "contours": [],
            })

        dataset.append({
            "image_id": int(image_id),
            "image_path": str(image_path),
            "image_size": (image_width, image_height),
            "annotations": annotations,
        })

    logger.info(
        "已從 Dataset Asset 建立 lightweight runtime dataset: %d 張圖片 / %d 個標註",
        len(dataset),
        sum(len(record["annotations"]) for record in dataset),
    )
    return dataset


def _derive_image_rel_path(image_path: str, data_root: Path) -> str:
    """Fallback conversion from stored absolute image path to relative path."""
    path = Path(image_path)
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        try:
            return str(path.relative_to(data_root))
        except ValueError:
            return path.name


def _resolve_asset_image_path(manifest: dict[str, Any], row: pd.Series) -> Path:
    """Resolve a best-effort image path from one dataset-asset row."""
    stored_path = Path(str(row.get("image_path", "")))
    if stored_path.is_file():
        return stored_path

    rel_path = str(row.get("image_rel_path", "")).strip()
    data_root = Path(str(manifest.get("coco", {}).get("data_root", "data/coco")))
    if rel_path:
        return data_root / rel_path
    return stored_path
