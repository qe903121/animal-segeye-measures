"""Baseline measurement validator for Phase 3 metrology outputs.

Computes two measurement families from the already-populated ``eyes``
results in the dataset:

1. Per-animal eye distance in pixels
2. Pairwise front/back separation proxy derived from apparent eye size

The second metric is intentionally reported as a **proxy**, not as a
physical 3D distance. In a single uncalibrated COCO RGB image, we do
not have enough information to recover real-world depth. Instead, this
validator uses inter-eye pixel distance as a relative depth cue:
animals with larger apparent eye distance are treated as closer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from .base import BaseValidator
from src.prediction import (
    build_measurement_prediction,
    summarize_measurement_asset_frames,
    summarize_measurement_prediction,
)

if TYPE_CHECKING:
    from src.data.loader import ImageRecord
    from src.data.prediction_loader import PredictionAsset

logger = logging.getLogger(__name__)


class MeasurementValidator(BaseValidator):
    """Phase 3 baseline measurement validator.

    This validator does not run inference. It consumes the dataset after
    Phase 2 eye localization and derives measurement tables plus summary
    statistics for:

    - ``eye_distance_px`` per valid animal instance
    - ``front_back_proxy_gap_px`` for every valid pair in an image
    """

    def evaluate(
        self,
        dataset: list[ImageRecord],
        prediction_asset: PredictionAsset | None = None,
    ) -> dict[str, Any]:
        """Compute baseline eye-distance and front/back proxy metrics."""
        if prediction_asset is not None:
            measurement_instances = prediction_asset.measurement_instances
            measurement_pairs = prediction_asset.measurement_pairs
            if not measurement_instances.empty or not measurement_pairs.empty:
                logger.info(
                    "MeasurementValidator: 直接使用 saved measurement prediction assets。"
                )
                return summarize_measurement_asset_frames(
                    measurement_instances,
                    measurement_pairs,
                )

        tables = build_measurement_prediction(dataset)
        return summarize_measurement_prediction(tables)

    def generate_report(
        self,
        metrics: dict[str, Any],
        dataset: list[ImageRecord],
        output_dir: Path,
        prediction_asset: PredictionAsset | None = None,
    ) -> None:
        """Persist measurement CSVs and log summary statistics."""
        del prediction_asset  # Reports currently use only prepared metrics.
        del dataset  # Not needed for this report yet.

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._export_csvs(metrics, output_dir)
        self._log_statistics(metrics)

    def _export_csvs(self, metrics: dict[str, Any], output_dir: Path) -> None:
        """Write per-animal and per-pair measurement tables."""
        annotation_rows = metrics.get("annotation_rows", [])
        if annotation_rows:
            ann_path = output_dir / "measurement_eye_distances.csv"
            pd.DataFrame(annotation_rows).to_csv(
                ann_path, index=False, encoding="utf-8"
            )
            logger.info(
                "已匯出雙眼距離 CSV: %s (%d 筆)",
                ann_path,
                len(annotation_rows),
            )

        pair_rows = metrics.get("pair_rows", [])
        if pair_rows:
            pair_path = output_dir / "measurement_front_back_pairs.csv"
            pd.DataFrame(pair_rows).to_csv(
                pair_path, index=False, encoding="utf-8"
            )
            logger.info(
                "已匯出前後距離 proxy CSV: %s (%d 筆)",
                pair_path,
                len(pair_rows),
            )

    def _log_statistics(self, metrics: dict[str, Any]) -> None:
        """Print a concise summary of measurement outputs."""
        logger.info("=" * 60)
        logger.info("Measurement Evaluation 結果")
        logger.info("=" * 60)
        logger.info("  動物實體總數:           %d", metrics["total_instances"])
        logger.info(
            "  可量測雙眼距離:         %d (%.1f%%)",
            metrics["valid_eye_measurements"],
            metrics["eye_measurement_rate"],
        )
        logger.info(
            "  可量測前後 pair:        %d/%d (%.1f%%)",
            metrics["valid_pairs"],
            metrics["total_pairs"],
            metrics["valid_pair_rate"],
        )

        eye_stats = metrics.get("eye_distance_stats", {})
        if eye_stats:
            logger.info("  --- 雙眼距離統計 (px) ---")
            logger.info(
                "    min=%.3f  max=%.3f  mean=%.3f  median=%.3f",
                eye_stats["min"],
                eye_stats["max"],
                eye_stats["mean"],
                eye_stats["median"],
            )

        pair_stats = metrics.get("front_back_gap_stats", {})
        if pair_stats:
            logger.info("  --- 前後距離 proxy gap 統計 (px) ---")
            logger.info(
                "    min=%.3f  max=%.3f  mean=%.3f  median=%.3f",
                pair_stats["min"],
                pair_stats["max"],
                pair_stats["mean"],
                pair_stats["median"],
            )

        if metrics.get("per_category"):
            logger.info("  --- 按類別的雙眼距離 ---")
            for category, cat_data in metrics["per_category"].items():
                logger.info(
                    "    %s: %d 筆, mean=%.3f px, median=%.3f px",
                    category,
                    cat_data["count"],
                    cat_data["mean_eye_distance_px"],
                    cat_data["median_eye_distance_px"],
                )

        logger.info("  --- ⚠ 量測侷限 ---")
        logger.info(
            "    前後距離目前為 monocular relative-depth proxy，"
            "以眼距大小差異表示，不是實體世界距離。"
        )
        logger.info("=" * 60)
