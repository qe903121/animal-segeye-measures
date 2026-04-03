"""Human ground-truth storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

GT_COLUMNS = [
    "dataset_id",
    "image_id",
    "annotation_id",
    "category",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",
    "depth_rank",
    "label_status",
    "annotator",
    "labeled_at",
    "notes",
]


@dataclass(frozen=True)
class GroundTruthPaths:
    """Filesystem paths for one dataset's GT asset."""

    dataset_id: str
    asset_dir: Path
    labels_path: Path
    meta_path: Path


class HumanLabelStore:
    """Load and persist human labels under ``assets/ground_truth``."""

    def __init__(self, config: dict) -> None:
        """Initialize from global config."""
        asset_cfg = config.get("assets", {})
        self._ground_truth_root = Path(
            asset_cfg.get("ground_truth_root", "assets/ground_truth")
        )
        self._schema_version = int(asset_cfg.get("schema_version", 1))

    def get_paths(self, dataset_id: str) -> GroundTruthPaths:
        """Resolve ground-truth asset paths for one dataset."""
        asset_dir = self._ground_truth_root / dataset_id
        return GroundTruthPaths(
            dataset_id=dataset_id,
            asset_dir=asset_dir,
            labels_path=asset_dir / "human_labels.csv",
            meta_path=asset_dir / "meta.json",
        )

    def load_labels(self, dataset_id: str) -> pd.DataFrame:
        """Load existing labels or return an empty table."""
        paths = self.get_paths(dataset_id)
        if not paths.labels_path.is_file():
            return pd.DataFrame(columns=GT_COLUMNS)

        df = pd.read_csv(paths.labels_path)
        for column in GT_COLUMNS:
            if column not in df.columns:
                df[column] = None
        return df[GT_COLUMNS]

    def upsert_label(
        self,
        dataset_id: str,
        row: dict[str, Any],
    ) -> None:
        """Insert or replace one annotation-level human label row."""
        paths = self.get_paths(dataset_id)
        paths.asset_dir.mkdir(parents=True, exist_ok=True)

        df = self.load_labels(dataset_id)
        df = df.copy()

        mask = (
            (df["dataset_id"] == row["dataset_id"])
            & (df["annotation_id"] == row["annotation_id"])
        )
        if mask.any():
            df = df.loc[~mask].copy()

        records = df.to_dict(orient="records")
        records.append(row)
        df = pd.DataFrame(records, columns=GT_COLUMNS)
        df = df.sort_values(
            by=["image_id", "annotation_id"], kind="stable"
        ).reset_index(drop=True)
        df.to_csv(paths.labels_path, index=False, encoding="utf-8")
        self._write_meta(paths, row.get("annotator", "unknown"))

    def _write_meta(self, paths: GroundTruthPaths, annotator: str) -> None:
        """Create or update meta.json."""
        now = _now_iso()
        meta: dict[str, Any]
        if paths.meta_path.is_file():
            with open(paths.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {
                "dataset_id": paths.dataset_id,
                "created_at": now,
                "schema_version": self._schema_version,
                "depth_rank_rule": "1 = closest to camera",
                "annotators": [],
            }

        annotators = set(meta.get("annotators", []))
        if annotator:
            annotators.add(str(annotator))

        meta["annotators"] = sorted(annotators)
        meta["updated_at"] = now

        with open(paths.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
