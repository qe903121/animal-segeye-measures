"""Dataset asset loader for Phase 1 exported assets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

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
