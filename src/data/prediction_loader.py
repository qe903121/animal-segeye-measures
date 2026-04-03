"""Prediction asset loader and runtime merge helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from .prediction_store import (
    LOCALIZATION_COLUMNS,
    MEASUREMENT_INSTANCE_COLUMNS,
    MEASUREMENT_PAIR_COLUMNS,
)

if TYPE_CHECKING:
    from .loader import ImageRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionAsset:
    """Loaded prediction asset bundle."""

    run_id: str
    asset_dir: Path
    meta: dict[str, Any]
    localization: pd.DataFrame
    measurement_instances: pd.DataFrame
    measurement_pairs: pd.DataFrame


class PredictionAssetLoader:
    """Read exported prediction assets from ``assets/predictions``."""

    def __init__(self, config: dict) -> None:
        """Initialize from global config."""
        asset_cfg = config.get("assets", {})
        self._prediction_root = Path(
            asset_cfg.get("prediction_root", "assets/predictions")
        )

    @property
    def prediction_root(self) -> Path:
        """Return root directory containing prediction assets."""
        return self._prediction_root

    def list_run_ids(self) -> list[str]:
        """Return available prediction run ids sorted by name."""
        if not self._prediction_root.is_dir():
            return []

        run_ids: list[str] = []
        for child in sorted(self._prediction_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "run_meta.json").is_file():
                run_ids.append(child.name)
        return run_ids

    def load(self, run_id: str) -> PredictionAsset:
        """Load one prediction asset by run id."""
        asset_dir = self._prediction_root / run_id
        meta_path = asset_dir / "run_meta.json"
        localization_path = asset_dir / "localization.csv"
        measurement_instances_path = asset_dir / "measurement_instances.csv"
        measurement_pairs_path = asset_dir / "measurement_pairs.csv"

        if not meta_path.is_file():
            raise FileNotFoundError(f"run_meta.json 不存在: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        localization = _load_optional_csv(
            localization_path,
            LOCALIZATION_COLUMNS,
        )
        measurement_instances = _load_optional_csv(
            measurement_instances_path,
            MEASUREMENT_INSTANCE_COLUMNS,
        )
        measurement_pairs = _load_optional_csv(
            measurement_pairs_path,
            MEASUREMENT_PAIR_COLUMNS,
        )

        logger.info(
            "已載入 Prediction Asset: %s (localization=%d, measurement_instances=%d, measurement_pairs=%d)",
            run_id,
            len(localization),
            len(measurement_instances),
            len(measurement_pairs),
        )

        return PredictionAsset(
            run_id=run_id,
            asset_dir=asset_dir,
            meta=meta,
            localization=localization,
            measurement_instances=measurement_instances,
            measurement_pairs=measurement_pairs,
        )


def apply_localization_predictions(
    dataset: list[ImageRecord],
    localization: pd.DataFrame,
) -> list[ImageRecord]:
    """Populate ``annotation['eyes']`` from saved localization predictions."""
    if localization.empty:
        raise ValueError("localization.csv 為空，無法回填 saved prediction。")

    required = {"annotation_id", "status"}
    missing = required.difference(localization.columns)
    if missing:
        raise ValueError(f"localization.csv 缺少必要欄位: {sorted(missing)}")

    lookup = (
        localization.sort_values(by=["image_id", "annotation_id"], kind="stable")
        .drop_duplicates(subset=["annotation_id"], keep="last")
        .set_index("annotation_id")
        .to_dict("index")
    )

    applied = 0
    missing_annotations = 0

    for record in dataset:
        for ann in record["annotations"]:
            ann_id = ann["id"]
            row = lookup.get(ann_id)
            if not row:
                missing_annotations += 1
                ann["eyes"] = {
                    "status": "FAILED_NOT_FOUND",
                    "left_eye": None,
                    "right_eye": None,
                    "confidence": 0.0,
                }
                continue

            ann["eyes"] = {
                "status": str(row.get("status", "UNKNOWN")),
                "left_eye": _maybe_point(
                    row.get("pred_left_eye_x"),
                    row.get("pred_left_eye_y"),
                ),
                "right_eye": _maybe_point(
                    row.get("pred_right_eye_x"),
                    row.get("pred_right_eye_y"),
                ),
                "confidence": _safe_float(row.get("confidence")) or 0.0,
            }
            applied += 1

    logger.info(
        "已將 saved localization prediction 回填至 runtime dataset: %d 筆，缺少 %d 筆",
        applied,
        missing_annotations,
    )
    return dataset


def _load_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Load one optional CSV and normalize canonical columns."""
    if not path.is_file():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns]


def _safe_float(value: Any) -> float | None:
    """Convert scalar to float, preserving blanks as ``None``."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_point(x: Any, y: Any) -> list[float] | None:
    """Build a point when both x/y are present."""
    x_val = _safe_float(x)
    y_val = _safe_float(y)
    if x_val is None or y_val is None:
        return None
    return [x_val, y_val]
