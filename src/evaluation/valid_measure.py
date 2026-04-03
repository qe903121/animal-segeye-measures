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
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from .base import BaseValidator

if TYPE_CHECKING:
    from src.data.loader import ImageRecord

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
    ) -> dict[str, Any]:
        """Compute baseline eye-distance and front/back proxy metrics."""
        annotation_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        eye_distances: list[float] = []
        pair_gaps: list[float] = []
        per_category_values: dict[str, list[float]] = {}

        total_instances = 0
        valid_eye_measurements = 0
        total_pairs = 0
        valid_pairs = 0

        for record in dataset:
            image_id = record["image_id"]
            measured_annotations: list[dict[str, Any]] = []

            for ann in record["annotations"]:
                total_instances += 1

                eyes = ann.get("eyes", {})
                status = eyes.get("status", "UNKNOWN")
                category = ann.get("category", "unknown")
                left_eye = eyes.get("left_eye")
                right_eye = eyes.get("right_eye")
                eye_distance_px: float | None = None

                if (
                    status == "SUCCESS"
                    and left_eye is not None
                    and right_eye is not None
                ):
                    dx = float(right_eye[0]) - float(left_eye[0])
                    dy = float(right_eye[1]) - float(left_eye[1])
                    eye_distance_px = round(math.hypot(dx, dy), 3)
                    eye_distances.append(eye_distance_px)
                    valid_eye_measurements += 1
                    per_category_values.setdefault(category, []).append(
                        eye_distance_px
                    )

                annotation_info = {
                    "image_id": image_id,
                    "annotation_id": ann.get("id", -1),
                    "category": category,
                    "status": status,
                    "left_eye": _fmt_coords(left_eye),
                    "right_eye": _fmt_coords(right_eye),
                    "eye_distance_px": eye_distance_px,
                    "depth_proxy_px": eye_distance_px,
                    "measurement_valid": eye_distance_px is not None,
                }
                annotation_rows.append(annotation_info)
                measured_annotations.append(annotation_info)

            for ann_a, ann_b in combinations(measured_annotations, 2):
                total_pairs += 1
                dist_a = ann_a["eye_distance_px"]
                dist_b = ann_b["eye_distance_px"]
                valid_pair = dist_a is not None and dist_b is not None

                row: dict[str, Any] = {
                    "image_id": image_id,
                    "annotation_a_id": ann_a["annotation_id"],
                    "category_a": ann_a["category"],
                    "eye_distance_a_px": dist_a,
                    "annotation_b_id": ann_b["annotation_id"],
                    "category_b": ann_b["category"],
                    "eye_distance_b_px": dist_b,
                    "valid_pair": valid_pair,
                    "front_back_proxy_gap_px": None,
                    "front_back_proxy_ratio": None,
                    "closer_annotation_id": None,
                    "farther_annotation_id": None,
                    "relation": "UNAVAILABLE",
                    "skip_reason": "",
                }

                if valid_pair:
                    valid_pairs += 1
                    gap_px = round(abs(dist_a - dist_b), 3)
                    max_dist = max(dist_a, dist_b)
                    ratio = round(gap_px / max_dist, 4) if max_dist > 0 else 0.0
                    pair_gaps.append(gap_px)

                    row["front_back_proxy_gap_px"] = gap_px
                    row["front_back_proxy_ratio"] = ratio

                    if math.isclose(dist_a, dist_b, rel_tol=0.0, abs_tol=1e-6):
                        row["relation"] = "TIE"
                    elif dist_a > dist_b:
                        row["relation"] = "A_CLOSER"
                        row["closer_annotation_id"] = ann_a["annotation_id"]
                        row["farther_annotation_id"] = ann_b["annotation_id"]
                    else:
                        row["relation"] = "B_CLOSER"
                        row["closer_annotation_id"] = ann_b["annotation_id"]
                        row["farther_annotation_id"] = ann_a["annotation_id"]
                else:
                    row["skip_reason"] = (
                        f"{ann_a['status']}|{ann_b['status']}"
                    )

                pair_rows.append(row)

        skipped_eye_measurements = total_instances - valid_eye_measurements
        skipped_pairs = total_pairs - valid_pairs
        eye_measurement_rate = (
            valid_eye_measurements / total_instances * 100
            if total_instances > 0 else 0.0
        )
        valid_pair_rate = (
            valid_pairs / total_pairs * 100 if total_pairs > 0 else 0.0
        )

        per_category = {
            category: {
                "count": len(values),
                "mean_eye_distance_px": round(statistics.mean(values), 3),
                "median_eye_distance_px": round(statistics.median(values), 3),
            }
            for category, values in sorted(per_category_values.items())
        }

        return {
            "total_instances": total_instances,
            "valid_eye_measurements": valid_eye_measurements,
            "skipped_eye_measurements": skipped_eye_measurements,
            "eye_measurement_rate": round(eye_measurement_rate, 1),
            "total_pairs": total_pairs,
            "valid_pairs": valid_pairs,
            "skipped_pairs": skipped_pairs,
            "valid_pair_rate": round(valid_pair_rate, 1),
            "eye_distance_stats": _summarize(eye_distances),
            "front_back_gap_stats": _summarize(pair_gaps),
            "per_category": per_category,
            "annotation_rows": annotation_rows,
            "pair_rows": pair_rows,
        }

    def generate_report(
        self,
        metrics: dict[str, Any],
        dataset: list[ImageRecord],
        output_dir: Path,
    ) -> None:
        """Persist measurement CSVs and log summary statistics."""
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


def _summarize(values: list[float]) -> dict[str, float]:
    """Return a standard numeric summary dictionary."""
    if not values:
        return {}

    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def _fmt_coords(coords: list[float] | None) -> str:
    """Format coordinate list as a CSV-friendly string."""
    if coords is None:
        return ""
    return f"({coords[0]:.1f}, {coords[1]:.1f})"
