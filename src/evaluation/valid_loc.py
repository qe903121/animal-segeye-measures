"""Localization quality validator for eye detection results.

Evaluates and reports on the quality of Phase 2 eye localization by
computing detection rates, confidence statistics, and per-category
breakdowns. Reuses :func:`~src.utils.visualization_eyes.debug_visualize_eyes`
for debug image generation to avoid code duplication.

Metrics computed:
    - Overall success / single-eye / failure counts and rates
    - Per-category (cat, dog, …) breakdown
    - Confidence score statistics (min, max, mean, median)
    - Per-image summary table

Reports generated:
    - ``eval_localization.csv`` — per-annotation detail table
    - ``debug/`` — annotated debug images (delegated to visualisation utils)
    - Structured log output with formatted statistics

Note:
    COCO does not provide ground-truth animal eye keypoints. Therefore
    this validator measures **detection rate (recall)** — whether the
    model returned two points above the confidence threshold — but
    cannot assess **positional accuracy (precision / PCK)** without
    supplementary GT keypoint data.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from .base import BaseValidator

if TYPE_CHECKING:
    from src.data.loader import ImageRecord

logger = logging.getLogger(__name__)


class LocalizationValidator(BaseValidator):
    """Eye localization quality validator.

    Inspects the ``eyes`` field of each annotation in the dataset
    (already populated by the detection pipeline) and computes
    aggregate quality metrics without performing any inference.

    Uses the existing ``visualization_eyes`` module to render debug
    images, keeping the drawing logic in a single location.

    Attributes:
        _score_threshold: Minimum confidence to consider a detection
            valid (used only for statistics, not for re-filtering).
    """

    def __init__(self, config: dict) -> None:
        """Initialize with global configuration.

        Args:
            config: Full parsed ``config.yaml`` dictionary.
        """
        super().__init__(config)
        ai_cfg = config.get("eye_detection", {}).get("ai_model", {})
        self._score_threshold: float = ai_cfg.get("score_threshold", 0.3)

    # ------------------------------------------------------------------
    # Lifecycle: evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: list[ImageRecord],
    ) -> dict[str, Any]:
        """Compute localization quality metrics from the dataset.

        Iterates over every annotation, inspects the ``eyes`` field,
        and aggregates counts, rates, and confidence statistics.

        Args:
            dataset: Phase 1/2 output with ``eyes`` populated.

        Returns:
            Metrics dictionary with keys documented in the class
            docstring.
        """
        status_counts: dict[str, int] = {}
        per_category: dict[str, dict[str, int]] = {}
        confidences: list[float] = []
        per_image_summary: list[dict[str, Any]] = []
        annotation_rows: list[dict[str, Any]] = []
        total_instances = 0

        for record in dataset:
            image_id = record["image_id"]
            img_success = 0
            img_fail = 0

            for ann in record["annotations"]:
                total_instances += 1
                eyes = ann.get("eyes", {})
                status: str = eyes.get("status", "UNKNOWN")
                confidence: float = eyes.get("confidence", 0.0)
                category: str = ann.get("category", "unknown")

                # Global status counts
                status_counts[status] = status_counts.get(status, 0) + 1

                # Per-category counts
                if category not in per_category:
                    per_category[category] = {}
                cat_counts = per_category[category]
                cat_counts[status] = cat_counts.get(status, 0) + 1

                # Confidence tracking (only for non-failure)
                if status in ("SUCCESS", "SINGLE_EYE"):
                    confidences.append(confidence)

                # Per-annotation detail row (for CSV)
                annotation_rows.append({
                    "image_id": image_id,
                    "annotation_id": ann.get("id", -1),
                    "category": category,
                    "status": status,
                    "confidence": round(confidence, 4),
                    "left_eye": _fmt_coords(eyes.get("left_eye")),
                    "right_eye": _fmt_coords(eyes.get("right_eye")),
                })

                if status == "SUCCESS":
                    img_success += 1
                else:
                    img_fail += 1

            per_image_summary.append({
                "image_id": image_id,
                "n_annotations": len(record["annotations"]),
                "n_success": img_success,
                "n_fail": img_fail,
            })

        # Derived metrics
        success_count = status_counts.get("SUCCESS", 0)
        single_count = status_counts.get("SINGLE_EYE", 0)
        failed_count = total_instances - success_count - single_count
        success_rate = (
            success_count / total_instances * 100
            if total_instances > 0 else 0.0
        )

        confidence_stats: dict[str, float] = {}
        if confidences:
            confidence_stats = {
                "min": round(min(confidences), 4),
                "max": round(max(confidences), 4),
                "mean": round(statistics.mean(confidences), 4),
                "median": round(statistics.median(confidences), 4),
            }

        # Per-category rates
        per_category_rates: dict[str, dict[str, Any]] = {}
        for cat, counts in per_category.items():
            cat_total = sum(counts.values())
            cat_success = counts.get("SUCCESS", 0)
            per_category_rates[cat] = {
                "total": cat_total,
                "success": cat_success,
                "success_rate": round(
                    cat_success / cat_total * 100 if cat_total > 0 else 0.0,
                    1,
                ),
                "status_distribution": dict(counts),
            }

        return {
            "total_instances": total_instances,
            "success_count": success_count,
            "single_eye_count": single_count,
            "failed_count": failed_count,
            "success_rate": round(success_rate, 1),
            "status_distribution": dict(status_counts),
            "per_category": per_category_rates,
            "confidence_stats": confidence_stats,
            "per_image_summary": per_image_summary,
            "annotation_rows": annotation_rows,
        }

    # ------------------------------------------------------------------
    # Lifecycle: generate_report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        metrics: dict[str, Any],
        dataset: list[ImageRecord],
        output_dir: Path,
    ) -> None:
        """Persist evaluation artefacts to *output_dir*.

        Generates:
            1. ``eval_localization.csv`` — per-annotation results.
            2. ``debug/`` — annotated debug images via the existing
               :func:`~src.utils.visualization_eyes.debug_visualize_eyes`.
            3. Formatted log output with full statistics.

        Args:
            metrics: Output from :meth:`evaluate`.
            dataset: Same dataset (for debug visualisation).
            output_dir: Target directory for all output files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. CSV export ---
        self._export_csv(metrics, output_dir)

        # --- 2. Debug visualisation (reuse existing tool) ---
        self._generate_debug_images(dataset, output_dir)

        # --- 3. Log formatted statistics ---
        self._log_statistics(metrics)

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    def _export_csv(
        self, metrics: dict[str, Any], output_dir: Path,
    ) -> None:
        """Export per-annotation detail CSV.

        Args:
            metrics: Must contain ``annotation_rows`` key.
            output_dir: Target directory.
        """
        rows = metrics.get("annotation_rows", [])
        if not rows:
            logger.warning("沒有標註資料可供匯出 CSV。")
            return

        csv_path = output_dir / "eval_localization.csv"
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("已匯出驗證 CSV: %s (%d 筆)", csv_path, len(df))

    def _generate_debug_images(
        self, dataset: list[ImageRecord], output_dir: Path,
    ) -> None:
        """Delegate debug image generation to visualisation utilities.

        Args:
            dataset: Dataset with ``eyes`` populated.
            output_dir: Root output directory. Debug images are saved
                into a ``debug/`` subdirectory.
        """
        # Lazy import to avoid circular dependency
        from src.utils.visualization_eyes import debug_visualize_eyes

        debug_dir = str(output_dir / "debug")
        saved = debug_visualize_eyes(
            dataset,
            output_dir=debug_dir,
            sample_count=len(dataset),  # Visualise all for evaluation
        )
        logger.info("已產生 debug 視覺化: %d 張 → %s", len(saved), debug_dir)

    def _log_statistics(self, metrics: dict[str, Any]) -> None:
        """Print formatted evaluation statistics to logger.

        Args:
            metrics: Full metrics dictionary from :meth:`evaluate`.
        """
        logger.info("=" * 60)
        logger.info("Localization Evaluation 結果")
        logger.info("=" * 60)

        total = metrics["total_instances"]
        logger.info("  動物實體總數:     %d", total)
        logger.info("  雙眼成功 (SUCCESS): %d (%.1f%%)",
                     metrics["success_count"], metrics["success_rate"])
        logger.info("  單眼 (SINGLE_EYE):  %d", metrics["single_eye_count"])
        logger.info("  失敗 (FAILED):      %d", metrics["failed_count"])

        # Status distribution
        logger.info("  --- 狀態分佈 ---")
        for status, count in sorted(metrics["status_distribution"].items()):
            pct = count / max(total, 1) * 100
            logger.info("    %s: %d (%.1f%%)", status, count, pct)

        # Per-category
        logger.info("  --- 按類別分組 ---")
        for cat, cat_data in sorted(metrics["per_category"].items()):
            logger.info(
                "    %s: %d 隻, SUCCESS %d (%.1f%%)",
                cat, cat_data["total"],
                cat_data["success"], cat_data["success_rate"],
            )

        # Confidence
        conf = metrics.get("confidence_stats", {})
        if conf:
            logger.info("  --- 信心分數統計 ---")
            logger.info(
                "    min=%.4f  max=%.4f  mean=%.4f  median=%.4f",
                conf["min"], conf["max"], conf["mean"], conf["median"],
            )

        # Limitation notice
        logger.info("  --- ⚠ 評估侷限 ---")
        logger.info(
            "    本指標為偵測率 (Recall)，非精度 (PCK/OKS)。"
            " COCO 不提供動物眼睛 GT keypoint。"
        )
        logger.info("=" * 60)


def _fmt_coords(coords: list[float] | None) -> str:
    """Format coordinate list as a string for CSV output.

    Args:
        coords: ``[x, y]`` or ``None``.

    Returns:
        Formatted string like ``"(123.4, 567.8)"`` or ``""`` if None.
    """
    if coords is None:
        return ""
    return f"({coords[0]:.1f}, {coords[1]:.1f})"
