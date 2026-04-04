"""Prediction asset schema and storage helpers.

This module defines the current MVP contract for the Prediction Asset Layer.
It establishes the filesystem schema, canonical column definitions, and
metadata structure used by export/load flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

LOCALIZATION_COLUMNS = [
    "run_id",
    "dataset_id",
    "image_id",
    "annotation_id",
    "category",
    "method",
    "model_name",
    "status",
    "pred_left_eye_x",
    "pred_left_eye_y",
    "pred_right_eye_x",
    "pred_right_eye_y",
    "confidence",
    "prediction_valid",
]

MEASUREMENT_INSTANCE_COLUMNS = [
    "run_id",
    "dataset_id",
    "image_id",
    "annotation_id",
    "category",
    "method",
    "model_name",
    "status",
    "eye_distance_px",
    "depth_proxy_px",
    "measurement_valid",
]

MEASUREMENT_PAIR_COLUMNS = [
    "run_id",
    "dataset_id",
    "image_id",
    "annotation_a_id",
    "category_a",
    "annotation_b_id",
    "category_b",
    "method",
    "model_name",
    "valid_pair",
    "front_back_proxy_gap_px",
    "front_back_proxy_ratio",
    "relation",
    "closer_annotation_id",
    "farther_annotation_id",
    "skip_reason",
]


@dataclass(frozen=True)
class PredictionAssetPaths:
    """Filesystem paths for one prediction run asset."""

    run_id: str
    asset_dir: Path
    meta_path: Path
    localization_path: Path
    measurement_instances_path: Path
    measurement_pairs_path: Path


class PredictionAssetStore:
    """Manage Prediction Asset Layer paths, metadata, and CSV schemas."""

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
        """Return the root directory containing prediction assets."""
        return self._prediction_root

    def build_run_id(
        self,
        dataset_id: str,
        method: str,
        model_name: str | None = None,
        task_scope: str = "pipeline",
        created_at: str | None = None,
    ) -> str:
        """Build a readable, collision-resistant prediction run id."""
        created = created_at or _now_iso()
        timestamp = _timestamp_slug(created)
        payload = {
            "dataset_id": dataset_id,
            "method": method,
            "model_name": model_name or "",
            "task_scope": task_scope,
            "created_at": created,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        base = "_".join(
            part
            for part in [
                "pred",
                _slugify(dataset_id),
                _slugify(method),
                _slugify(model_name or ""),
                _slugify(task_scope),
                timestamp,
            ]
            if part
        )
        return f"{base}_{digest}"

    def get_paths(self, run_id: str) -> PredictionAssetPaths:
        """Resolve all filesystem paths for a prediction run asset."""
        asset_dir = self._prediction_root / run_id
        return PredictionAssetPaths(
            run_id=run_id,
            asset_dir=asset_dir,
            meta_path=asset_dir / "run_meta.json",
            localization_path=asset_dir / "localization.csv",
            measurement_instances_path=asset_dir / "measurement_instances.csv",
            measurement_pairs_path=asset_dir / "measurement_pairs.csv",
        )

    def run_exists(self, run_id: str) -> bool:
        """Return whether a prediction asset already exists for ``run_id``."""
        paths = self.get_paths(run_id)
        return any(
            path.exists()
            for path in [
                paths.meta_path,
                paths.localization_path,
                paths.measurement_instances_path,
                paths.measurement_pairs_path,
            ]
        )

    def build_run_meta(
        self,
        *,
        dataset_id: str,
        method: str,
        task_scope: str,
        model_name: str | None = None,
        run_id: str | None = None,
        created_at: str | None = None,
        git_commit: str | None = None,
        config_fingerprint_source: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build canonical metadata for one prediction run."""
        created = created_at or _now_iso()
        resolved_run_id = run_id or self.build_run_id(
            dataset_id=dataset_id,
            method=method,
            model_name=model_name,
            task_scope=task_scope,
            created_at=created,
        )

        meta: dict[str, Any] = {
            "run_id": resolved_run_id,
            "dataset_id": dataset_id,
            "created_at": created,
            "schema_version": self._schema_version,
            "method": method,
            "model_name": model_name or "",
            "task_scope": task_scope,
            "git_commit": git_commit or "",
            "config_fingerprint": self._fingerprint_config(
                config_fingerprint_source
            ),
        }

        if extra_meta:
            meta["extra_meta"] = extra_meta

        return meta

    def initialize_run(
        self,
        meta: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> PredictionAssetPaths:
        """Create the run directory and persist ``run_meta.json``.

        By default, existing prediction assets are treated as immutable.
        Callers must pass ``overwrite=True`` explicitly to replace a run
        with the same ``run_id``.
        """
        run_id = str(meta["run_id"])
        paths = self.get_paths(run_id)
        existing_files = [
            path for path in [
                paths.meta_path,
                paths.localization_path,
                paths.measurement_instances_path,
                paths.measurement_pairs_path,
            ] if path.exists()
        ]

        if existing_files and not overwrite:
            raise FileExistsError(
                "Prediction Asset 已存在，為避免覆蓋既有實驗結果，"
                f"請改用新的 --run-id 或明確指定覆寫。run_id={run_id}"
            )

        paths.asset_dir.mkdir(parents=True, exist_ok=True)
        if existing_files and overwrite:
            logger.warning(
                "Prediction Asset 將被覆寫: %s", paths.asset_dir
            )

        with open(paths.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")

        logger.info("已初始化 Prediction Asset: %s", paths.asset_dir)
        logger.info("  run_meta: %s", paths.meta_path)
        return paths

    def write_localization(
        self,
        paths: PredictionAssetPaths,
        rows: Iterable[dict[str, Any]] | pd.DataFrame,
    ) -> None:
        """Write canonical localization prediction rows."""
        df = _to_frame(rows, LOCALIZATION_COLUMNS)
        df.to_csv(paths.localization_path, index=False, encoding="utf-8")
        logger.info(
            "已匯出 localization prediction: %s (%d 筆)",
            paths.localization_path,
            len(df),
        )

    def write_measurement_instances(
        self,
        paths: PredictionAssetPaths,
        rows: Iterable[dict[str, Any]] | pd.DataFrame,
    ) -> None:
        """Write canonical per-instance measurement rows."""
        df = _to_frame(rows, MEASUREMENT_INSTANCE_COLUMNS)
        df.to_csv(paths.measurement_instances_path, index=False, encoding="utf-8")
        logger.info(
            "已匯出 measurement instances: %s (%d 筆)",
            paths.measurement_instances_path,
            len(df),
        )

    def write_measurement_pairs(
        self,
        paths: PredictionAssetPaths,
        rows: Iterable[dict[str, Any]] | pd.DataFrame,
    ) -> None:
        """Write canonical pairwise measurement rows."""
        df = _to_frame(rows, MEASUREMENT_PAIR_COLUMNS)
        df.to_csv(paths.measurement_pairs_path, index=False, encoding="utf-8")
        logger.info(
            "已匯出 measurement pairs: %s (%d 筆)",
            paths.measurement_pairs_path,
            len(df),
        )

    def empty_localization_frame(self) -> pd.DataFrame:
        """Return an empty localization table with canonical columns."""
        return pd.DataFrame(columns=LOCALIZATION_COLUMNS)

    def empty_measurement_instances_frame(self) -> pd.DataFrame:
        """Return an empty per-instance measurement table."""
        return pd.DataFrame(columns=MEASUREMENT_INSTANCE_COLUMNS)

    def empty_measurement_pairs_frame(self) -> pd.DataFrame:
        """Return an empty pairwise measurement table."""
        return pd.DataFrame(columns=MEASUREMENT_PAIR_COLUMNS)

    def _fingerprint_config(self, config: dict[str, Any] | None) -> str:
        """Return a short stable digest for a config snapshot."""
        if not config:
            return ""
        return hashlib.sha1(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]


def _to_frame(
    rows: Iterable[dict[str, Any]] | pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Normalize rows into a DataFrame with canonical column order."""
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(list(rows))

    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns]


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_slug(value: str) -> str:
    """Convert ISO time into a filename-safe timestamp slug."""
    slug = value.replace("-", "").replace(":", "").replace("T", "_")
    return slug.replace("Z", "Z")


def _slugify(value: str) -> str:
    """Return a simple lowercase filesystem slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value
