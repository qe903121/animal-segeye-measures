"""Baseline measurement validator for prediction-side metrology outputs.

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
    """Prediction-side baseline measurement validator.

    This validator does not run inference. It consumes the dataset after
    eye localization and derives measurement tables plus summary
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
                    "MeasurementValidator: Using saved measurement prediction assets directly."
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
                "Exported inter-eye distance CSV: %s (%d records)",
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
                "Exported front/back relative-depth proxy CSV: %s (%d records)",
                pair_path,
                len(pair_rows),
            )

    def _log_statistics(self, metrics: dict[str, Any]) -> None:
        """Print a concise summary of measurement outputs."""
        logger.info("=" * 60)
        logger.info("Measurement Evaluation Results")
        logger.info("=" * 60)
        logger.info("  Total Animal Instances: %d", metrics["total_instances"])
        logger.info(
            "  Measurable inter-eye distances: %d (%.1f%%)",
            metrics["valid_eye_measurements"],
            metrics["eye_measurement_rate"],
        )
        logger.info(
            "  Measurable front/back pair: %d/%d (%.1f%%)",
            metrics["valid_pairs"],
            metrics["total_pairs"],
            metrics["valid_pair_rate"],
        )

        eye_stats = metrics.get("eye_distance_stats", {})
        if eye_stats:
            logger.info("  --- Inter-eye Distance Stats (px) ---")
            logger.info(
                "    min=%.3f  max=%.3f  mean=%.3f  median=%.3f",
                eye_stats["min"],
                eye_stats["max"],
                eye_stats["mean"],
                eye_stats["median"],
            )

        pair_stats = metrics.get("front_back_gap_stats", {})
        if pair_stats:
            logger.info("  --- Front/Back Depth Proxy Gap Stats (px) ---")
            logger.info(
                "    min=%.3f  max=%.3f  mean=%.3f  median=%.3f",
                pair_stats["min"],
                pair_stats["max"],
                pair_stats["mean"],
                pair_stats["median"],
            )

        if metrics.get("per_category"):
            logger.info("  --- Inter-eye Distance by Category ---")
            for category, cat_data in metrics["per_category"].items():
                logger.info(
                    "    %s: %d records, mean=%.3f px, median=%.3f px",
                    category,
                    cat_data["count"],
                    cat_data["mean_eye_distance_px"],
                    cat_data["median_eye_distance_px"],
                )

        logger.info("  --- ⚠ Measurement Limitations ---")
        logger.info(
            "    Front/back distance is currently a monocular relative-depth proxy,"
            "Represented by eye distance difference; this is not real-world distance."
        )
        logger.info("=" * 60)
