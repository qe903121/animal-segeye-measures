"""Prediction generation builders for the formal Prediction Layer.

This module separates ``C``-layer construction from evaluation/reporting.
It can be used by:

- the runtime export path (`main.py evaluate --save-predictions`)
- validators that need prediction-derived tables

The goal is to keep validators focused on ``D`` (evaluation and reporting)
while centralizing the generation of reusable prediction tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from itertools import combinations
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.data.loader import ImageRecord


@dataclass(frozen=True)
class MeasurementPredictionTables:
    """Prediction-layer measurement tables derived from localized eyes."""

    annotation_rows: list[dict[str, Any]]
    pair_rows: list[dict[str, Any]]


def build_localization_prediction_rows(
    *,
    dataset: list[ImageRecord],
    run_id: str,
    dataset_id: str,
    method: str,
    model_name: str,
) -> list[dict[str, object]]:
    """Build canonical localization prediction rows from the runtime dataset."""
    rows: list[dict[str, object]] = []

    for record in dataset:
        image_id = record["image_id"]
        for ann in record["annotations"]:
            eyes = ann.get("eyes", {})
            left_eye = eyes.get("left_eye")
            right_eye = eyes.get("right_eye")
            status = str(eyes.get("status", "UNKNOWN"))

            rows.append({
                "run_id": run_id,
                "dataset_id": dataset_id,
                "image_id": image_id,
                "annotation_id": ann.get("id", -1),
                "category": ann.get("category", "unknown"),
                "method": method,
                "model_name": model_name,
                "status": status,
                "pred_left_eye_x": _coord_value(left_eye, 0),
                "pred_left_eye_y": _coord_value(left_eye, 1),
                "pred_right_eye_x": _coord_value(right_eye, 0),
                "pred_right_eye_y": _coord_value(right_eye, 1),
                "confidence": float(eyes.get("confidence", 0.0)),
                "prediction_valid": (
                    status == "SUCCESS"
                    and left_eye is not None
                    and right_eye is not None
                ),
            })

    return rows


def build_measurement_prediction(
    dataset: list[ImageRecord],
) -> MeasurementPredictionTables:
    """Generate measurement-layer prediction tables from localized eyes."""
    annotation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for record in dataset:
        image_id = record["image_id"]
        measured_annotations: list[dict[str, Any]] = []

        for ann in record["annotations"]:
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
                eye_distance_px = round(_euclidean_distance(left_eye, right_eye), 3)

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
                gap_px = round(abs(dist_a - dist_b), 3)
                max_dist = max(dist_a, dist_b)
                ratio = round(gap_px / max_dist, 4) if max_dist > 0 else 0.0

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
                row["skip_reason"] = f"{ann_a['status']}|{ann_b['status']}"

            pair_rows.append(row)

    return MeasurementPredictionTables(
        annotation_rows=annotation_rows,
        pair_rows=pair_rows,
    )


def summarize_measurement_prediction(
    tables: MeasurementPredictionTables,
) -> dict[str, Any]:
    """Summarize measurement prediction tables into validator metrics."""
    annotation_rows = tables.annotation_rows
    pair_rows = tables.pair_rows

    total_instances = len(annotation_rows)
    valid_eye_rows = [
        row for row in annotation_rows if row.get("measurement_valid")
    ]
    valid_eye_measurements = len(valid_eye_rows)
    eye_distances = [
        float(row["eye_distance_px"])
        for row in valid_eye_rows
        if row.get("eye_distance_px") is not None
    ]

    total_pairs = len(pair_rows)
    valid_pair_rows = [row for row in pair_rows if row.get("valid_pair")]
    valid_pairs = len(valid_pair_rows)
    pair_gaps = [
        float(row["front_back_proxy_gap_px"])
        for row in valid_pair_rows
        if row.get("front_back_proxy_gap_px") is not None
    ]

    per_category_values: dict[str, list[float]] = {}
    for row in valid_eye_rows:
        category = str(row.get("category", "unknown"))
        eye_distance = row.get("eye_distance_px")
        if eye_distance is None:
            continue
        per_category_values.setdefault(category, []).append(float(eye_distance))

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


def summarize_measurement_asset_frames(
    measurement_instances: pd.DataFrame,
    measurement_pairs: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize saved measurement asset tables into validator metrics.

    This lets evaluators consume saved ``measurement_instances.csv`` and
    ``measurement_pairs.csv`` directly, instead of recomputing measurement
    tables from runtime-localized eyes.
    """
    annotation_rows = measurement_instances.to_dict("records")
    pair_rows = measurement_pairs.to_dict("records")

    total_instances = len(annotation_rows)
    valid_eye_rows = [
        row for row in annotation_rows
        if _as_bool(row.get("measurement_valid"))
    ]
    valid_eye_measurements = len(valid_eye_rows)
    eye_distances = [
        float(row["eye_distance_px"])
        for row in valid_eye_rows
        if row.get("eye_distance_px") is not None and not pd.isna(row.get("eye_distance_px"))
    ]

    total_pairs = len(pair_rows)
    valid_pair_rows = [
        row for row in pair_rows
        if _as_bool(row.get("valid_pair"))
    ]
    valid_pairs = len(valid_pair_rows)
    pair_gaps = [
        float(row["front_back_proxy_gap_px"])
        for row in valid_pair_rows
        if row.get("front_back_proxy_gap_px") is not None and not pd.isna(row.get("front_back_proxy_gap_px"))
    ]

    per_category_values: dict[str, list[float]] = {}
    for row in valid_eye_rows:
        category = str(row.get("category", "unknown"))
        eye_distance = row.get("eye_distance_px")
        if eye_distance is None or pd.isna(eye_distance):
            continue
        per_category_values.setdefault(category, []).append(float(eye_distance))

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


def build_measurement_instance_asset_rows(
    *,
    tables: MeasurementPredictionTables,
    run_id: str,
    dataset_id: str,
    method: str,
    model_name: str,
) -> list[dict[str, object]]:
    """Build canonical per-instance measurement asset rows."""
    output: list[dict[str, object]] = []
    for row in tables.annotation_rows:
        output.append({
            "run_id": run_id,
            "dataset_id": dataset_id,
            "image_id": row.get("image_id"),
            "annotation_id": row.get("annotation_id"),
            "category": row.get("category"),
            "method": method,
            "model_name": model_name,
            "status": row.get("status"),
            "eye_distance_px": row.get("eye_distance_px"),
            "depth_proxy_px": row.get("depth_proxy_px"),
            "measurement_valid": row.get("measurement_valid"),
        })
    return output


def build_measurement_pair_asset_rows(
    *,
    tables: MeasurementPredictionTables,
    run_id: str,
    dataset_id: str,
    method: str,
    model_name: str,
) -> list[dict[str, object]]:
    """Build canonical pairwise measurement asset rows."""
    output: list[dict[str, object]] = []
    for row in tables.pair_rows:
        output.append({
            "run_id": run_id,
            "dataset_id": dataset_id,
            "image_id": row.get("image_id"),
            "annotation_a_id": row.get("annotation_a_id"),
            "category_a": row.get("category_a"),
            "annotation_b_id": row.get("annotation_b_id"),
            "category_b": row.get("category_b"),
            "method": method,
            "model_name": model_name,
            "valid_pair": row.get("valid_pair"),
            "front_back_proxy_gap_px": row.get("front_back_proxy_gap_px"),
            "front_back_proxy_ratio": row.get("front_back_proxy_ratio"),
            "relation": row.get("relation"),
            "closer_annotation_id": row.get("closer_annotation_id"),
            "farther_annotation_id": row.get("farther_annotation_id"),
            "skip_reason": row.get("skip_reason"),
        })
    return output


def _coord_value(
    coords: list[float] | tuple[float, float] | None,
    index: int,
) -> float | None:
    """Safely extract one coordinate component."""
    if coords is None:
        return None
    return float(coords[index])


def _fmt_coords(coords: list[float] | tuple[float, float] | None) -> str:
    """Format coordinate list as a CSV-friendly string."""
    if coords is None:
        return ""
    return f"({coords[0]:.1f}, {coords[1]:.1f})"


def _euclidean_distance(
    pt1: list[float] | tuple[float, float],
    pt2: list[float] | tuple[float, float],
) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


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


def _as_bool(value: Any) -> bool:
    """Normalize bool-ish CSV values loaded from pandas."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}
