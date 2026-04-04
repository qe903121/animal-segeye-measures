"""Shared helpers for annotation sub-commands (annotate / review).

These helpers keep rendering, path resolution, and interactive prompt logic
shared between ``cmd_annotate`` and ``cmd_review`` without duplicating code.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
WINDOW_NAME = "Annotation Preview"
ACTION_QUIT = "__quit__"
ACTION_REDO = "__redo__"
ACTION_SKIP = "__skip__"
COMPLETED_LABEL_STATUSES = {"LABELED"}

# ── Image / path helpers ─────────────────────────────────────────────


def resolve_image_path(
    manifest: dict[str, Any],
    row: pd.Series,
) -> Path:
    """Resolve a previewable image path from asset row + manifest."""
    stored_path = Path(str(row.get("image_path", "")))
    if stored_path.is_file():
        return stored_path

    rel_path = str(row.get("image_rel_path", "")).strip()
    data_root = Path(str(manifest.get("coco", {}).get("data_root", "data/coco")))
    if rel_path:
        candidate = data_root / rel_path
        if candidate.is_file():
            return candidate

    return stored_path


def can_attempt_imshow() -> bool:
    """Return True when the current environment likely supports GUI windows."""
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
    return True


def resolve_imshow_usage(
    no_imshow: bool,
    mode: str,
    default_enabled: bool = True,
) -> bool:
    """Resolve imshow usage once per run."""
    if no_imshow:
        return False
    if not can_attempt_imshow():
        print(
            "[INFO] 目前環境未偵測到可用圖形顯示（DISPLAY / WAYLAND_DISPLAY）。"
        )
        print("[INFO] 已自動停用 cv2.imshow()，改為純終端模式。")
        return False

    prompt = (
        "是否使用 cv2.imshow() 逐張檢查標註結果？"
        if mode == "review"
        else "是否使用 cv2.imshow() 顯示圖片提示？"
    )
    return prompt_yes_no(prompt, default=default_enabled)


def color_for_category(category: str) -> tuple[int, int, int]:
    """Return deterministic BGR color for one category."""
    hue = hash(category) % 180
    hsv_arr = np.array([[[hue, 220, 230]]], dtype=np.uint8)
    bgr_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2BGR)
    return tuple(int(v) for v in bgr_arr[0, 0])


# ── Image preview ────────────────────────────────────────────────────


def show_image_preview(
    image_path: Path,
    image_rows: pd.DataFrame,
    enabled: bool,
) -> bool:
    """Optionally display the image with bbox / id hints."""
    if not enabled:
        return False
    if not image_path.is_file():
        print(f"[WARN] 原圖不存在，無法顯示: {image_path}")
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] 原圖讀取失敗，無法顯示: {image_path}")
        return False

    canvas = image.copy()
    for _, row in image_rows.iterrows():
        color = color_for_category(str(row["category"]))
        x = int(round(float(row["bbox_x"])))
        y = int(round(float(row["bbox_y"])))
        w = int(round(float(row["bbox_w"])))
        h = int(round(float(row["bbox_h"])))
        ann_id = int(row["annotation_id"])
        label = f"{row['category']} #{ann_id}"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"Image ID: {int(image_rows.iloc[0]['image_id'])}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)
        print("[INFO] 圖片視窗已顯示，可一邊看圖一邊在終端機輸入。")
        return True
    except cv2.error as exc:
        print(f"[WARN] cv2.imshow() 失敗，改為純終端模式: {exc}")
        return False


# ── Review rendering ─────────────────────────────────────────────────


def build_review_rows(
    instances: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join latest human labels onto dataset instances for review rendering."""
    if labels_df.empty:
        review = instances.copy()
        review["label_status"] = None
        return review

    latest_labels = labels_df.sort_values(
        by=["image_id", "annotation_id", "labeled_at"],
        kind="stable",
    ).drop_duplicates(
        subset=["dataset_id", "annotation_id"],
        keep="last",
    )

    label_columns = [
        "dataset_id",
        "image_id",
        "annotation_id",
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
    review = instances.merge(
        latest_labels[label_columns],
        on=["dataset_id", "image_id", "annotation_id"],
        how="left",
        suffixes=("", "_gt"),
    )
    return review.sort_values(
        by=["image_id", "annotation_id"],
        kind="stable",
    ).reset_index(drop=True)


def render_review_canvas(
    image_path: Path,
    image_rows: pd.DataFrame,
) -> np.ndarray | None:
    """Render one review image with human labels overlaid on the original."""
    if not image_path.is_file():
        print(f"[WARN] 原圖不存在，無法產生 review: {image_path}")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] 原圖讀取失敗，無法產生 review: {image_path}")
        return None

    canvas = image.copy()
    for _, row in image_rows.iterrows():
        color = color_for_category(str(row["category"]))
        x = int(round(float(row["bbox_x"])))
        y = int(round(float(row["bbox_y"])))
        w = int(round(float(row["bbox_w"])))
        h = int(round(float(row["bbox_h"])))
        ann_id = int(row["annotation_id"])
        status = str(row.get("label_status", "")).strip().upper() or "UNLABELED"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas,
            f"{row['category']} #{ann_id}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        if status == "LABELED":
            _draw_review_labeled(canvas, row, color)
        elif status == "SKIPPED":
            _draw_review_skipped(canvas, x, y, w, h)
        else:
            _draw_review_unlabeled(canvas, x, y, w)

    cv2.putText(
        canvas,
        f"ID: {int(image_rows.iloc[0]['image_id'])}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_review_labeled(
    canvas: np.ndarray,
    row: pd.Series,
    color: tuple[int, int, int],
) -> None:
    """Draw human-labeled eye points and depth rank."""
    left_eye = _point_from_row(row, "left_eye")
    right_eye = _point_from_row(row, "right_eye")
    depth_rank = row.get("depth_rank")
    if left_eye is None or right_eye is None:
        _draw_review_skipped(
            canvas,
            int(round(float(row["bbox_x"]))),
            int(round(float(row["bbox_y"]))),
            int(round(float(row["bbox_w"]))),
            int(round(float(row["bbox_h"]))),
        )
        return

    cv2.circle(canvas, left_eye, 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, right_eye, 5, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.line(canvas, left_eye, right_eye, (0, 255, 0), 2, cv2.LINE_AA)

    mid_x = (left_eye[0] + right_eye[0]) // 2
    mid_y = (left_eye[1] + right_eye[1]) // 2
    rank_text = f"rank={int(depth_rank)}" if pd.notna(depth_rank) else "rank=?"
    cv2.putText(
        canvas,
        rank_text,
        (mid_x, max(20, mid_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_review_skipped(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw an obvious SKIPPED marker for one bbox."""
    cx = x + w // 2
    cy = y + h // 2
    cv2.line(canvas, (cx - 12, cy - 12), (cx + 12, cy + 12), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, (cx - 12, cy + 12), (cx + 12, cy - 12), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "SKIPPED",
        (x, y + h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_review_unlabeled(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
) -> None:
    """Mark a bbox as not yet labeled."""
    del w  # unused
    cv2.putText(
        canvas,
        "UNLABELED",
        (x, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def _point_from_row(
    row: pd.Series,
    prefix: str,
) -> tuple[int, int] | None:
    """Extract one integer point from merged review row."""
    x = row.get(f"{prefix}_x")
    y = row.get(f"{prefix}_y")
    if pd.isna(x) or pd.isna(y):
        return None
    return int(round(float(x))), int(round(float(y)))


# ── Image ID resolution ──────────────────────────────────────────────


def resolve_image_ids(
    instances: pd.DataFrame,
    labels_df: pd.DataFrame,
    explicit_ids: list[int] | None,
    from_csv: str | None,
    skip_labeled: bool,
) -> list[int]:
    """Resolve target image ids from args or interactive prompt."""
    if explicit_ids:
        return sorted(dict.fromkeys(int(v) for v in explicit_ids))

    if from_csv:
        return load_image_ids_from_csv(from_csv)

    all_image_ids = sorted(int(v) for v in instances["image_id"].unique())
    if skip_labeled:
        pending_ids = []
        labeled_ann_ids = completed_annotation_ids(labels_df)
        for image_id in all_image_ids:
            image_rows = instances.loc[instances["image_id"] == image_id]
            ann_ids = set(int(v) for v in image_rows["annotation_id"].tolist())
            if not ann_ids.issubset(labeled_ann_ids):
                pending_ids.append(image_id)
        print(f"目前尚有 {len(pending_ids)} 張影像含未完成標註。")
        return pending_ids
    else:
        pending_ids = all_image_ids

    print(
        "可直接輸入單一 image_id 或多個 image_id（用逗號分隔）；"
        "直接按 Enter 則依序處理目前候選清單。"
    )
    raw = input("Image ID 清單: ").strip()
    if not raw:
        return pending_ids

    parsed = parse_int_list(raw)
    if not parsed:
        print("輸入無法解析，改用預設候選清單。")
        return pending_ids
    return parsed


def resolve_review_image_ids(
    review_rows: pd.DataFrame,
    explicit_ids: list[int] | None,
    from_csv: str | None,
) -> list[int]:
    """Resolve image ids for review mode."""
    if explicit_ids:
        return sorted(dict.fromkeys(int(v) for v in explicit_ids))

    if from_csv:
        return load_image_ids_from_csv(from_csv)

    labeled = review_rows.loc[review_rows["label_status"].notna()]
    image_ids = sorted(int(v) for v in labeled["image_id"].unique())
    print(f"目前共有 {len(image_ids)} 張影像含既有 human labels 可供檢查。")
    return image_ids


def load_image_ids_from_csv(csv_path: str) -> list[int]:
    """Load image ids from a CSV file."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 不存在: {path}")

    df = pd.read_csv(path)
    if "image_id" in df.columns:
        values = df["image_id"].tolist()
    else:
        values = df.iloc[:, 0].tolist()
    return sorted(dict.fromkeys(int(v) for v in values))


# ── Label-status helpers ─────────────────────────────────────────────


def completed_annotation_ids(labels_df: pd.DataFrame) -> set[int]:
    """Return annotation ids considered fully labeled."""
    if labels_df.empty or "label_status" not in labels_df.columns:
        return set()

    completed = labels_df.loc[
        labels_df["label_status"].isin(COMPLETED_LABEL_STATUSES)
    ]
    return set(int(value) for value in completed["annotation_id"].tolist())


def is_completed_label(existing: dict[str, Any]) -> bool:
    """Return True when an existing label should count as completed."""
    status = str(existing.get("label_status", "")).strip().upper()
    return status in COMPLETED_LABEL_STATUSES


def get_existing_label(
    labels_df: pd.DataFrame,
    annotation_id: int,
) -> dict[str, Any] | None:
    """Return existing label row for one annotation, if present."""
    matched = labels_df.loc[labels_df["annotation_id"] == annotation_id]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def build_used_ranks(
    labels_df: pd.DataFrame,
    image_id: int,
) -> dict[int, int]:
    """Collect existing depth ranks for one image."""
    subset = labels_df.loc[
        (labels_df["image_id"] == image_id)
        & (labels_df["label_status"] == "LABELED")
        & labels_df["depth_rank"].notna()
    ]
    return {
        int(row["annotation_id"]): int(row["depth_rank"])
        for _, row in subset.iterrows()
    }


# ── Interactive prompts ──────────────────────────────────────────────


def prompt_yes_no(prompt: str, default: bool) -> bool:
    """Prompt for yes/no answer with default fallback."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("請輸入 y 或 n。")


def prompt_coordinate(prompt: str) -> tuple[float, float] | str:
    """Prompt for one ``x,y`` coordinate pair with robust parsing."""
    while True:
        raw = input(f"{prompt}: ").strip()
        lowered = raw.lower()
        if lowered in {"quit", "skip", "redo"}:
            return f"__{lowered}__"
        if not raw:
            print("輸入不能為空，請用 x,y 格式。")
            continue

        try:
            x_str, y_str = [part.strip() for part in raw.split(",", 1)]
            x = float(x_str)
            y = float(y_str)
            return round(x, 3), round(y, 3)
        except ValueError:
            print("格式錯誤，請輸入 x,y，例如 123,45")


def prompt_depth_rank(
    used_ranks: dict[int, int],
    current_annotation_id: int,
) -> int | str:
    """Prompt for positive integer depth rank."""
    while True:
        raw = input("深度排序 rank（1=最靠前）: ").strip()
        lowered = raw.lower()
        if lowered in {"quit", "skip", "redo"}:
            return f"__{lowered}__"
        if not raw:
            print("depth_rank 不能為空。")
            continue

        try:
            rank = int(raw)
            if rank <= 0:
                raise ValueError
        except ValueError:
            print("請輸入正整數，例如 1")
            continue

        collision = [
            ann_id for ann_id, used_rank in used_ranks.items()
            if ann_id != current_annotation_id and used_rank == rank
        ]
        if collision:
            print(f"[WARN] 目前此 image 已有 annotation 使用相同 rank: {collision}")
            if not prompt_yes_no("仍要使用這個 rank 嗎？", default=False):
                continue

        return rank


def prompt_dataset_id(asset_loader: Any) -> str:
    """Prompt user to choose a dataset asset id."""
    dataset_ids = asset_loader.list_dataset_ids()
    if not dataset_ids:
        raise FileNotFoundError(
            f"找不到 dataset assets。請先執行 `python main.py data`。\n"
            f"搜尋路徑: {asset_loader.dataset_root}"
        )

    print("可用的 dataset assets:")
    for idx, dataset_id in enumerate(dataset_ids, start=1):
        print(f"  {idx}. {dataset_id}")

    while True:
        raw = input("請輸入 dataset_id 或序號: ").strip()
        if not raw:
            print("dataset_id 不能為空。")
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(dataset_ids):
                return dataset_ids[index]
            print("序號超出範圍，請重新輸入。")
            continue
        if raw in dataset_ids:
            return raw
        print("找不到對應的 dataset_id，請重新輸入。")


# ── Misc ─────────────────────────────────────────────────────────────


def parse_int_list(raw: str) -> list[int]:
    """Parse comma-separated integers; ignore invalid tokens."""
    values: list[int] = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            logger.warning("忽略無法解析的 image_id: %s", token)
    return sorted(dict.fromkeys(values))


def now_iso() -> str:
    """Return current UTC timestamp string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def special_result(
    row: pd.Series,
    annotator: str,
    action: str,
) -> dict[str, Any] | str:
    """Convert skip/quit action into saveable row or quit signal."""
    if action == ACTION_QUIT:
        return ACTION_QUIT
    if action == ACTION_REDO:
        return ACTION_REDO

    return {
        "dataset_id": row["dataset_id"],
        "image_id": int(row["image_id"]),
        "annotation_id": int(row["annotation_id"]),
        "category": str(row["category"]),
        "bbox_x": float(row["bbox_x"]),
        "bbox_y": float(row["bbox_y"]),
        "bbox_w": float(row["bbox_w"]),
        "bbox_h": float(row["bbox_h"]),
        "left_eye_x": None,
        "left_eye_y": None,
        "right_eye_x": None,
        "right_eye_y": None,
        "depth_rank": None,
        "label_status": "SKIPPED",
        "annotator": annotator,
        "labeled_at": now_iso(),
        "notes": "",
    }
