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
        self._schema_version = int(
            asset_cfg.get(
                "prediction_schema_version",
                asset_cfg.get("schema_version", 1),
            )
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
            raise FileNotFoundError(f"run_meta.json not found: {meta_path}")
        if not localization_path.is_file():
            raise FileNotFoundError(
                f"localization.csv not found: {localization_path}"
            )
        if not measurement_instances_path.is_file():
            raise FileNotFoundError(
                "measurement_instances.csv not found: "
                f"{measurement_instances_path}"
            )
        if not measurement_pairs_path.is_file():
            raise FileNotFoundError(
                f"measurement_pairs.csv not found: {measurement_pairs_path}"
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._validate_meta(run_id, meta)

        localization = _load_required_csv(
            localization_path,
            LOCALIZATION_COLUMNS,
        )
        measurement_instances = _load_required_csv(
            measurement_instances_path,
            MEASUREMENT_INSTANCE_COLUMNS,
        )
        measurement_pairs = _load_required_csv(
            measurement_pairs_path,
            MEASUREMENT_PAIR_COLUMNS,
        )

        logger.info(
            "Loaded Prediction Asset: %s (localization=%d, measurement_instances=%d, measurement_pairs=%d)",
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

    def _validate_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        """Validate metadata contract for one prediction asset."""
        required_keys = {
            "run_id",
            "dataset_id",
            "created_at",
            "schema_version",
            "method",
            "task_scope",
        }
        missing = sorted(
            key for key in required_keys
            if not str(meta.get(key, "")).strip()
            and key not in {"schema_version"}
        )
        if "schema_version" not in meta:
            missing.append("schema_version")
        if missing:
            raise ValueError(
                "Prediction Asset metadata is missing required fields: "
                f"{missing}"
            )

        meta_run_id = str(meta.get("run_id", "")).strip()
        if meta_run_id != run_id:
            raise ValueError(
                "Prediction Asset metadata directory mismatch with directory name: "
                f"{meta_run_id} != {run_id}"
            )

        try:
            schema_version = int(meta.get("schema_version"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Illegal schema_version in Prediction Asset metadata."
            ) from exc

        if schema_version != self._schema_version:
            raise ValueError(
                "Incompatible Prediction Asset schema_version: "
                f"{schema_version} != {self._schema_version}"
            )


def apply_localization_predictions(
    dataset: list[ImageRecord],
    localization: pd.DataFrame,
) -> list[ImageRecord]:
    """Populate ``annotation['eyes']`` from saved localization predictions."""
    if localization.empty:
        raise ValueError("localization.csv is empty, cannot backfill saved predictions.")

    required = {"annotation_id", "status"}
    missing = required.difference(localization.columns)
    if missing:
        raise ValueError(f"localization.csv is missing required fields: {sorted(missing)}")

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
        "Backfilled saved localization prediction to runtime dataset: %d records, missing %d records",
        applied,
        missing_annotations,
    )
    return dataset


def _load_required_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Load one required CSV and validate canonical columns."""
    df = pd.read_csv(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing necessary columns: {missing}"
        )
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
