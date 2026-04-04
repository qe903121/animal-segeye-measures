"""Accuracy Validator for comparing module predictions against Human GT.

Calculates NME (Normalized Mean Error), RDE (Relative Distance Error) for
Localization, and Pairwise Accuracy for relative depth ordinal ranking.
"""

from __future__ import annotations

import csv
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from src.data.loader import ImageRecord
from .base import BaseValidator

if TYPE_CHECKING:
    from src.data.prediction_loader import PredictionAsset

logger = logging.getLogger(__name__)


def _euclidean_distance(pt1: list[float] | tuple[float, float], pt2: list[float] | tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def _normalize_label_status(value: Any) -> str:
    """Normalize GT label status to a stable uppercase token."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _safe_float(value: Any) -> float | None:
    """Convert a scalar to float, returning None for blank / NaN values."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_gt_point(row: dict[str, Any], prefix: str) -> tuple[float, float] | None:
    """Extract one GT point from a row dictionary."""
    x = _safe_float(row.get(f"{prefix}_x"))
    y = _safe_float(row.get(f"{prefix}_y"))
    if x is None or y is None:
        return None
    return (x, y)


def _safe_rank(value: Any) -> float | None:
    """Convert depth rank to float when valid."""
    rank = _safe_float(value)
    if rank is None or rank < 1:
        return None
    return rank


def _build_measurement_instance_lookup(
    measurement_instances: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    """Build annotation_id -> saved measurement row lookup."""
    if measurement_instances.empty or "annotation_id" not in measurement_instances.columns:
        return {}

    prepared = (
        measurement_instances
        .sort_values(by=["image_id", "annotation_id"], kind="stable")
        .drop_duplicates(subset=["annotation_id"], keep="last")
        .set_index("annotation_id")
    )
    return prepared.to_dict("index")


def _build_measurement_pair_lookup(
    measurement_pairs: pd.DataFrame,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Build an unordered pair lookup keyed by image_id + sorted ann ids."""
    if measurement_pairs.empty:
        return {}

    required = {"image_id", "annotation_a_id", "annotation_b_id"}
    if not required.issubset(measurement_pairs.columns):
        return {}

    lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in measurement_pairs.to_dict("records"):
        key = _pair_key(
            row.get("image_id"),
            row.get("annotation_a_id"),
            row.get("annotation_b_id"),
        )
        if key is None:
            continue
        lookup[key] = row
    return lookup


def _pair_key(
    image_id: Any,
    ann_a_id: Any,
    ann_b_id: Any,
) -> tuple[int, int, int] | None:
    """Build a normalized unordered pair key."""
    try:
        img = int(image_id)
        a = int(ann_a_id)
        b = int(ann_b_id)
    except (TypeError, ValueError):
        return None
    if a <= b:
        return (img, a, b)
    return (img, b, a)


def _saved_pair_pred_closer(
    saved_row: dict[str, Any],
    ann_a_id: int,
    ann_b_id: int,
) -> str | None:
    """Map saved measurement pair relation to local A/B/TIE label."""
    relation = str(saved_row.get("relation", "UNAVAILABLE"))
    if relation == "TIE":
        return "TIE"
    if relation not in {"A_CLOSER", "B_CLOSER"}:
        return None

    closer_id = _safe_float(saved_row.get("closer_annotation_id"))
    if closer_id is None:
        return None
    closer_int = int(closer_id)
    if closer_int == ann_a_id:
        return "A"
    if closer_int == ann_b_id:
        return "B"
    return None


def _as_bool(value: Any) -> bool:
    """Normalize bool-ish values loaded from CSV."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _build_localization_lookup(
    localization: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    """Build annotation_id -> saved localization row lookup."""
    if localization.empty or "annotation_id" not in localization.columns:
        return {}

    prepared = (
        localization
        .sort_values(by=["image_id", "annotation_id"], kind="stable")
        .drop_duplicates(subset=["annotation_id"], keep="last")
        .set_index("annotation_id")
    )
    return prepared.to_dict("index")


def _extract_pred_point(
    row: dict[str, Any],
    prefix: str,
) -> list[float] | None:
    """Extract one saved predicted point from a localization asset row."""
    x = _safe_float(row.get(f"{prefix}_x"))
    y = _safe_float(row.get(f"{prefix}_y"))
    if x is None or y is None:
        return None
    return [x, y]


def _compute_unordered_nme(
    pred_left: list[float],
    pred_right: list[float],
    gt_left: tuple[float, float],
    gt_right: tuple[float, float],
) -> tuple[float, float, float, str]:
    """Compute NME with unordered eye-pair matching.

    The assignment task requires locating the two eyes and measuring their
    distance; it does not require anatomical laterality classification.
    Therefore we evaluate both pairings:

    - direct:  pred_left -> gt_left, pred_right -> gt_right
    - swapped: pred_left -> gt_right, pred_right -> gt_left

    and keep the smaller normalized error.
    """
    dist_gt = _euclidean_distance(gt_left, gt_right)
    if dist_gt <= 0:
        return (float("inf"), float("inf"), float("inf"), "INVALID_GT")

    direct = (
        _euclidean_distance(pred_left, gt_left)
        + _euclidean_distance(pred_right, gt_right)
    ) / (2.0 * dist_gt)
    swapped = (
        _euclidean_distance(pred_left, gt_right)
        + _euclidean_distance(pred_right, gt_left)
    ) / (2.0 * dist_gt)

    if swapped < direct:
        return (swapped, direct, swapped, "SWAPPED")
    return (direct, direct, swapped, "DIRECT")


class AccuracyValidator(BaseValidator):
    """Validator computing NME, RDE, and Pairwise Depth Accuracy against Ground Truth.

    Prefers resolving ``dataset_id`` from the formal Prediction Asset metadata.
    Falls back to ``config["runtime_dataset_id"]`` for older internal paths.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the AccuracyValidator."""
        super().__init__(config)
        self.gt_root = Path(config.get("assets", {}).get("ground_truth_root", "assets/ground_truth"))
        self.abs_tol = 1e-6  # Tolerance for TIE in prediction proxy

    def evaluate(
        self,
        dataset: list[ImageRecord],
        prediction_asset: PredictionAsset | None = None,
    ) -> dict[str, Any]:
        """Compute accuracy metrics against Human GT."""
        prediction_asset = prediction_asset or self._config.get("runtime_prediction_asset")
        dataset_id = self._resolve_dataset_id(prediction_asset)

        if not dataset_id:
            logger.error(
                "AccuracyValidator: 未提供 dataset_id，也無法從 Prediction Asset metadata 推得。"
            )
            return {"error": True}

        self.dataset_id = dataset_id

        gt_path = self.gt_root / self.dataset_id / "human_labels.csv"
        if not gt_path.exists():
            logger.error("AccuracyValidator: 找不到對應的 GT 檔案: %s", gt_path)
            return {"error": True}

        logger.info("載入 Human GT: %s", gt_path)
        gt_df = pd.read_csv(gt_path)
        gt_df, gt_stats = self._prepare_gt_frame(gt_df)
        localization_lookup: dict[int, dict[str, Any]] = {}
        measurement_instance_lookup: dict[int, dict[str, Any]] = {}
        measurement_pair_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
        if prediction_asset is not None:
            localization_lookup = _build_localization_lookup(
                prediction_asset.localization
            )
            measurement_instance_lookup = _build_measurement_instance_lookup(
                prediction_asset.measurement_instances
            )
            measurement_pair_lookup = _build_measurement_pair_lookup(
                prediction_asset.measurement_pairs
            )
            if measurement_instance_lookup or measurement_pair_lookup:
                logger.info(
                    "AccuracyValidator: 優先使用 saved measurement prediction assets 做 RDE / pairwise 評估。"
                )

        try:
            gt_lookup = gt_df.set_index("annotation_id").to_dict("index")
        except ValueError as exc:
            logger.error("AccuracyValidator: GT annotation_id 無法建立唯一索引: %s", exc)
            return {"error": True}

        instance_results: list[dict[str, Any]] = []
        pair_results: list[dict[str, Any]] = []

        total_anns = 0
        annotations_with_gt = 0
        gt_usable_annotations = 0
        comparable_anns = 0
        excluded_no_gt = 0
        excluded_gt_skipped = 0
        excluded_gt_invalid = 0
        excluded_pred_not_success = 0
        pair_total_candidates = 0
        pair_tie_count = 0
        pair_excluded_pred_not_success = 0

        # Data collection
        for record in dataset:
            image_id = record["image_id"]

            pair_candidates: list[dict[str, Any]] = []

            for ann in record["annotations"]:
                ann_id = ann["id"]
                category = ann["category"]
                eyes = ann.get("eyes", {})
                localization_row = localization_lookup.get(ann_id)
                if localization_row is not None:
                    pred_status = str(localization_row.get("status", "UNKNOWN"))
                    pred_left = _extract_pred_point(localization_row, "pred_left_eye")
                    pred_right = _extract_pred_point(localization_row, "pred_right_eye")
                else:
                    pred_status = eyes.get("status", "UNKNOWN")
                    pred_left = eyes.get("left_eye")
                    pred_right = eyes.get("right_eye")
                measurement_row = measurement_instance_lookup.get(ann_id)
                pred_status_measurement = (
                    str(measurement_row.get("status", pred_status))
                    if measurement_row is not None else pred_status
                )
                dist_pred_measurement = (
                    _safe_float(measurement_row.get("eye_distance_px"))
                    if measurement_row is not None else None
                )
                total_anns += 1

                instance_row: dict[str, Any] = {
                    "image_id": image_id,
                    "annotation_id": ann_id,
                    "category": category,
                    "gt_label_status": "",
                    "pred_status": pred_status_measurement,
                    "gt_left_eye": "",
                    "gt_right_eye": "",
                    "pred_left_eye": self.fmt_coords(pred_left),
                    "pred_right_eye": self.fmt_coords(pred_right),
                    "depth_rank_gt": None,
                    "include_in_accuracy": False,
                    "exclude_reason": "",
                    "nme": None,
                    "nme_direct": None,
                    "nme_swapped": None,
                    "nme_match_mode": "",
                    "rde": None,
                    "dist_pred": None,
                    "dist_gt": None,
                }

                gt_row = gt_lookup.get(ann_id)
                if not gt_row:
                    excluded_no_gt += 1
                    instance_row["exclude_reason"] = "NO_GT"
                    instance_results.append(instance_row)
                    continue

                annotations_with_gt += 1
                label_status = _normalize_label_status(gt_row.get("label_status"))
                instance_row["gt_label_status"] = label_status

                if label_status == "SKIPPED":
                    excluded_gt_skipped += 1
                    instance_row["exclude_reason"] = "GT_SKIPPED"
                    instance_results.append(instance_row)
                    continue

                if label_status != "LABELED":
                    excluded_gt_invalid += 1
                    instance_row["exclude_reason"] = "GT_INVALID_STATUS"
                    instance_results.append(instance_row)
                    continue

                gt_left = _extract_gt_point(gt_row, "left_eye")
                gt_right = _extract_gt_point(gt_row, "right_eye")
                if gt_left is None or gt_right is None:
                    excluded_gt_invalid += 1
                    instance_row["exclude_reason"] = "GT_MISSING_COORDS"
                    instance_results.append(instance_row)
                    continue

                gt_usable_annotations += 1
                instance_row["gt_left_eye"] = self.fmt_coords(gt_left)
                instance_row["gt_right_eye"] = self.fmt_coords(gt_right)

                gt_rank = _safe_rank(gt_row.get("depth_rank"))
                instance_row["depth_rank_gt"] = gt_rank

                if gt_rank is not None:
                    pair_candidates.append({
                        "ann_id": ann_id,
                        "category": category,
                        "depth_rank_gt": gt_rank,
                        "pred_status": pred_status_measurement,
                        "dist_pred": dist_pred_measurement,
                    })

                if (
                    pred_status_measurement != "SUCCESS"
                    or pred_left is None
                    or pred_right is None
                ):
                    excluded_pred_not_success += 1
                    instance_row["exclude_reason"] = f"PRED_{pred_status_measurement}"
                    instance_results.append(instance_row)
                    continue

                comparable_anns += 1
                instance_row["include_in_accuracy"] = True

                dist_gt = _euclidean_distance(gt_left, gt_right)
                nme, nme_direct, nme_swapped, nme_match_mode = _compute_unordered_nme(
                    pred_left,
                    pred_right,
                    gt_left,
                    gt_right,
                )

                dist_pred = (
                    dist_pred_measurement
                    if dist_pred_measurement is not None
                    else _euclidean_distance(pred_left, pred_right)
                )
                rde = abs(dist_pred - dist_gt) / dist_gt if dist_gt > 0 else float("inf")

                instance_row["nme"] = nme
                instance_row["nme_direct"] = nme_direct
                instance_row["nme_swapped"] = nme_swapped
                instance_row["nme_match_mode"] = nme_match_mode
                instance_row["rde"] = rde
                instance_row["dist_pred"] = dist_pred
                instance_row["dist_gt"] = dist_gt
                instance_results.append(instance_row)

            if len(pair_candidates) >= 2:
                for a, b in combinations(pair_candidates, 2):
                    pair_total_candidates += 1
                    rank_a = a["depth_rank_gt"]
                    rank_b = b["depth_rank_gt"]
                    pair_row: dict[str, Any] = {
                        "image_id": image_id,
                        "ann_a_id": a["ann_id"],
                        "ann_b_id": b["ann_id"],
                        "cat_a": a["category"],
                        "cat_b": b["category"],
                        "gt_rank_a": rank_a,
                        "gt_rank_b": rank_b,
                        "pred_status_a": a["pred_status"],
                        "pred_status_b": b["pred_status"],
                        "gt_closer": None,
                        "pred_closer": None,
                        "include_in_accuracy": False,
                        "exclude_reason": "",
                        "correct": None,
                    }

                    if rank_a < rank_b:
                        gt_closer = "A"
                    elif rank_b < rank_a:
                        gt_closer = "B"
                    else:
                        gt_closer = "TIE"

                    pair_row["gt_closer"] = gt_closer

                    if gt_closer == "TIE":
                        pair_tie_count += 1
                        pair_row["pred_closer"] = "N/A"
                        pair_row["exclude_reason"] = "GT_TIE"
                        pair_results.append(pair_row)
                        continue

                    saved_pair = measurement_pair_lookup.get(
                        _pair_key(image_id, a["ann_id"], b["ann_id"])
                    )
                    pred_closer: str | None = None
                    if saved_pair is not None:
                        if not _as_bool(saved_pair.get("valid_pair")):
                            pair_excluded_pred_not_success += 1
                            pair_row["exclude_reason"] = "PRED_NOT_SUCCESS"
                            pair_results.append(pair_row)
                            continue
                        pred_closer = _saved_pair_pred_closer(
                            saved_pair,
                            a["ann_id"],
                            b["ann_id"],
                        )

                    if pred_closer is None:
                        dist_a = a["dist_pred"]
                        dist_b = b["dist_pred"]
                        if (
                            a["pred_status"] != "SUCCESS"
                            or b["pred_status"] != "SUCCESS"
                            or dist_a is None
                            or dist_b is None
                        ):
                            pair_excluded_pred_not_success += 1
                            pair_row["exclude_reason"] = "PRED_NOT_SUCCESS"
                            pair_results.append(pair_row)
                            continue

                        if dist_a > dist_b + self.abs_tol:
                            pred_closer = "A"
                        elif dist_b > dist_a + self.abs_tol:
                            pred_closer = "B"
                        else:
                            pred_closer = "TIE"

                    pair_row["include_in_accuracy"] = True
                    pair_row["pred_closer"] = pred_closer
                    is_correct = (gt_closer == pred_closer)
                    pair_row["correct"] = is_correct
                    pair_results.append(pair_row)

        pair_eligible_non_tie = pair_total_candidates - pair_tie_count
        pair_valid = [
            row for row in pair_results
            if row.get("include_in_accuracy")
        ]
        instance_valid = [
            row for row in instance_results
            if row.get("include_in_accuracy")
        ]

        metrics: dict[str, Any] = {
            "total_annotations": total_anns,
            "annotations_with_gt": annotations_with_gt,
            "gt_total_rows": gt_stats["total_rows"],
            "gt_duplicate_rows_removed": gt_stats["duplicate_rows_removed"],
            "gt_labeled_rows": gt_stats["labeled_rows"],
            "gt_skipped_rows": gt_stats["skipped_rows"],
            "gt_invalid_labeled_rows": gt_stats["invalid_labeled_rows"],
            "gt_usable_annotations": gt_usable_annotations,
            "comparable_annotations": comparable_anns,
            "excluded_no_gt": excluded_no_gt,
            "excluded_gt_skipped": excluded_gt_skipped,
            "excluded_gt_invalid": excluded_gt_invalid,
            "excluded_pred_not_success": excluded_pred_not_success,
            "instance_coverage_percent": (
                comparable_anns / gt_usable_annotations * 100
                if gt_usable_annotations > 0 else 0.0
            ),
            "instance_rows": instance_results,
            "pair_rows": pair_results,
            "total_pairs_all": pair_total_candidates,
            "total_pairs_tie": pair_tie_count,
            "total_pairs_valid": len(pair_valid),
            "pair_excluded_pred_not_success": pair_excluded_pred_not_success,
            "pair_coverage_percent": (
                len(pair_valid) / pair_eligible_non_tie * 100
                if pair_eligible_non_tie > 0 else 0.0
            ),
        }

        if instance_valid:
            df_inst = pd.DataFrame(instance_valid)
            metrics["nme_mean"] = df_inst["nme"].mean()
            metrics["nme_median"] = df_inst["nme"].median()
            metrics["nme_exc_pct"] = (df_inst["nme"] < 0.05).mean() * 100
            metrics["nme_acc_pct"] = (df_inst["nme"] < 0.10).mean() * 100

            metrics["rde_mean_percent"] = df_inst["rde"].mean() * 100
            metrics["rde_median_percent"] = df_inst["rde"].median() * 100
            metrics["rde_hlrel_pct"] = (df_inst["rde"] < 0.05).mean() * 100
            metrics["rde_acc_pct"] = (df_inst["rde"] < 0.15).mean() * 100
            match_mode_counts = (
                df_inst["nme_match_mode"]
                .fillna("")
                .value_counts()
                .to_dict()
            )
            metrics["nme_match_mode_counts"] = {
                str(key): int(value)
                for key, value in match_mode_counts.items()
                if str(key)
            }
            
            # Per Category
            cat_stats = df_inst.groupby("category").agg({
                "nme": ["mean", "count"],
                "rde": "mean"
            }).reset_index()
            cat_dict = {}
            for _, row in cat_stats.iterrows():
                c = row[("category", "")]
                cat_dict[c] = {
                    "count": int(row[("nme", "count")]),
                    "nme_mean": row[("nme", "mean")],
                    "rde_mean_pct": row[("rde", "mean")] * 100
                }
            metrics["per_category_instances"] = cat_dict

        if pair_valid:
            df_pair = pd.DataFrame(pair_valid)
            metrics["pairwise_acc_percent"] = df_pair["correct"].mean() * 100
            metrics["pairwise_correct_count"] = int(df_pair["correct"].sum())
            
            # Per category combination (e.g., cat-cat, cat-dog)
            def _normalize_pair_cat(row):
                cats = sorted([row["cat_a"], row["cat_b"]])
                return f"{cats[0]}-{cats[1]}"
                
            df_pair["cat_pair"] = df_pair.apply(_normalize_pair_cat, axis=1)
            pair_cat_stats = df_pair.groupby("cat_pair").agg({"correct": ["mean", "count"]}).reset_index()
            p_cat_dict = {}
            for _, row in pair_cat_stats.iterrows():
                p_cat_dict[row[("cat_pair", "")]] = {
                    "count": int(row[("correct", "count")]),
                    "acc_pct": row[("correct", "mean")] * 100
                }
            metrics["per_category_pairs"] = p_cat_dict

        return metrics

    def _resolve_dataset_id(
        self,
        prediction_asset: PredictionAsset | None,
    ) -> str:
        """Resolve dataset_id from the most reliable available source."""
        if prediction_asset is not None:
            dataset_id = str(prediction_asset.meta.get("dataset_id", "")).strip()
            if dataset_id:
                return dataset_id
        return str(self._config.get("runtime_dataset_id", "")).strip()

    def generate_report(
        self,
        metrics: dict[str, Any],
        dataset: list[ImageRecord],
        output_dir: Path,
        prediction_asset: PredictionAsset | None = None,
    ) -> None:
        """Persist CSV files and emit summary logs."""
        del dataset, prediction_asset
        if metrics.get("error"):
            return

        instance_csv = output_dir / "eval_accuracy_instances.csv"
        pair_csv = output_dir / "eval_accuracy_pairs.csv"

        self._export_csv(instance_csv, metrics.get("instance_rows", []))
        self._export_csv(pair_csv, metrics.get("pair_rows", []))

        # Output Summary
        logger.info("=" * 60)
        logger.info("=== Eye Localization Accuracy ===")
        logger.info(
            "  標註數:          %d\n"
            "  有 GT 數:        %d\n"
            "  GT 可用數:       %d (GT=LABELED 且座標完整)\n"
            "  可比對數:        %d (pred=SUCCESS ∩ GT usable)\n"
            "  Coverage:        %5.1f%%",
            metrics.get("total_annotations", 0),
            metrics.get("annotations_with_gt", 0),
            metrics.get("gt_usable_annotations", 0),
            metrics.get("comparable_annotations", 0),
            metrics.get("instance_coverage_percent", 0.0),
        )
        logger.info(
            "  排除統計:        NO_GT=%d, GT_SKIPPED=%d, GT_INVALID=%d, PRED_NOT_SUCCESS=%d",
            metrics.get("excluded_no_gt", 0),
            metrics.get("excluded_gt_skipped", 0),
            metrics.get("excluded_gt_invalid", 0),
            metrics.get("excluded_pred_not_success", 0),
        )
        if metrics.get("gt_duplicate_rows_removed", 0):
            logger.info(
                "  GT 清理:         duplicate_rows_removed=%d, invalid_labeled_rows=%d",
                metrics.get("gt_duplicate_rows_removed", 0),
                metrics.get("gt_invalid_labeled_rows", 0),
            )

        nme_m = metrics.get("nme_mean")
        if nme_m is not None:
            logger.info("  --- L1: 定位品質 (NME, unordered eye pair aware) ---")
            logger.info("    NME_mean:     %.4f", nme_m)
            logger.info("    NME_median:   %.4f", metrics.get("nme_median", 0))
            logger.info("    NME < 0.05:   %5.1f%%   ← Excellent", metrics.get("nme_exc_pct", 0))
            logger.info("    NME < 0.10:   %5.1f%%   ← Acceptable", metrics.get("nme_acc_pct", 0))
            match_counts = metrics.get("nme_match_mode_counts", {})
            if match_counts:
                logger.info(
                    "    Eye-pair matching: DIRECT=%d, SWAPPED=%d",
                    int(match_counts.get("DIRECT", 0)),
                    int(match_counts.get("SWAPPED", 0)),
                )

            logger.info("  --- L2: 量測品質 (RDE) ---")
            logger.info("    RDE_mean:     %5.1f%%", metrics.get("rde_mean_percent", 0))
            logger.info("    RDE_median:   %5.1f%%", metrics.get("rde_median_percent", 0))
            logger.info("    RDE < 5%%:     %5.1f%%   ← Highly Reliable", metrics.get("rde_hlrel_pct", 0))
            logger.info("    RDE < 15%%:    %5.1f%%   ← Acceptable", metrics.get("rde_acc_pct", 0))

            logger.info("  --- Per-Category Instances ---")
            cat_stats = metrics.get("per_category_instances", {})
            for c, st in cat_stats.items():
                logger.info(
                    "    %-5s: NME=%.4f, RDE=%5.1f%%  (n=%d)",
                    c, st["nme_mean"], st["rde_mean_pct"], st["count"]
                )

        p_acc = metrics.get("pairwise_acc_percent")
        if p_acc is not None or metrics.get("total_pairs_all", 0) > 0:
            total_v = metrics.get("total_pairs_valid", 0)
            logger.info("\n=== Front-Back Ordering Accuracy ===")
            logger.info("  Proxy:          eye_distance_px")
            logger.info("  GT pair 候選數:  %d", metrics.get("total_pairs_all", 0))
            logger.info("  TIE 排除數:      %d", metrics.get("total_pairs_tie", 0))
            logger.info(
                "  pred 排除數:     %d",
                metrics.get("pair_excluded_pred_not_success", 0),
            )
            logger.info(
                "  Comparable pair: %d (%5.1f%% coverage)",
                total_v,
                metrics.get("pair_coverage_percent", 0.0),
            )
            if p_acc is not None:
                logger.info(
                    "  --- Pairwise Accuracy ---\n"
                    "    正確:         %d/%d (%5.1f%%)\n"
                    "    Random base:  50.0%%",
                    metrics.get("pairwise_correct_count", 0),
                    total_v,
                    p_acc,
                )
                logger.info("  --- Per-Category Pair ---")
                p_stats = metrics.get("per_category_pairs", {})
                for cp, st in p_stats.items():
                    logger.info(
                        "    %-7s: %3.1f%% (n=%d)",
                        cp, st["acc_pct"], st["count"]
                    )

    def _export_csv(self, file_path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def _prepare_gt_frame(
        self,
        gt_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Validate and normalize GT rows before lookup construction."""
        required_columns = {
            "annotation_id",
            "label_status",
            "left_eye_x",
            "left_eye_y",
            "right_eye_x",
            "right_eye_y",
            "depth_rank",
        }
        missing_columns = sorted(required_columns - set(gt_df.columns))
        if missing_columns:
            raise ValueError(f"Human GT 缺少必要欄位: {missing_columns}")

        prepared = gt_df.copy()
        prepared["label_status"] = prepared["label_status"].map(_normalize_label_status)

        if "labeled_at" in prepared.columns:
            prepared = prepared.sort_values(
                by=["annotation_id", "labeled_at"],
                kind="stable",
                na_position="last",
            )

        duplicate_rows_removed = int(
            prepared["annotation_id"].duplicated(keep="last").sum()
        )
        if duplicate_rows_removed:
            logger.warning(
                "Human GT 偵測到 %d 筆重複 annotation_id，將保留最新一筆。",
                duplicate_rows_removed,
            )
        prepared = prepared.drop_duplicates(
            subset=["annotation_id"],
            keep="last",
        ).reset_index(drop=True)

        invalid_labeled_rows = int(
            (
                (prepared["label_status"] == "LABELED")
                & (
                    prepared["left_eye_x"].isna()
                    | prepared["left_eye_y"].isna()
                    | prepared["right_eye_x"].isna()
                    | prepared["right_eye_y"].isna()
                )
            ).sum()
        )
        if invalid_labeled_rows:
            logger.warning(
                "Human GT 有 %d 筆 LABELED 缺少完整眼睛座標，將在 accuracy 中排除。",
                invalid_labeled_rows,
            )

        stats = {
            "total_rows": len(gt_df),
            "duplicate_rows_removed": duplicate_rows_removed,
            "labeled_rows": int((prepared["label_status"] == "LABELED").sum()),
            "skipped_rows": int((prepared["label_status"] == "SKIPPED").sum()),
            "invalid_labeled_rows": invalid_labeled_rows,
        }
        return prepared, stats
