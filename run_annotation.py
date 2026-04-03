"""Human ground-truth annotation CLI.

Builds reusable human labels from exported dataset assets using a
lightweight terminal workflow. Interaction is strictly text-based via
``print()`` and ``input()``; image display is optional via
``cv2.imshow()`` for visual reference.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.data import DatasetAssetLoader, HumanLabelStore
from src.utils.cli import load_config, setup_logging

logger = logging.getLogger(__name__)

WINDOW_NAME = "Annotation Preview"
ACTION_QUIT = "__quit__"
ACTION_REDO = "__redo__"
ACTION_SKIP = "__skip__"
COMPLETED_LABEL_STATUSES = {"LABELED"}


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Human GT annotation tool (terminal-only workflow)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="設定檔路徑 (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="要標註的 dataset asset id",
    )
    parser.add_argument(
        "--image-id",
        nargs="+",
        type=int,
        default=None,
        help="僅標註指定 image_id，可傳多個",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="從 CSV 讀取 image_id 清單",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="anonymous",
        help="標註者名稱",
    )
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        help="跳過 label_status=LABELED 的 annotation（SKIPPED 不算完成）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 annotation 已有標註，直接覆寫",
    )
    parser.add_argument(
        "--no-imshow",
        action="store_true",
        help="不使用 cv2.imshow 顯示圖片",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="啟用詳細日誌 (DEBUG level)",
    )
    return parser


def main() -> None:
    """Run the interactive annotation tool."""
    args = _build_parser().parse_args()
    setup_logging(verbose=args.verbose)

    config = load_config(args.config)
    asset_loader = DatasetAssetLoader(config)
    gt_store = HumanLabelStore(config)

    dataset_id = args.dataset_id or _prompt_dataset_id(asset_loader)
    asset = asset_loader.load(dataset_id)
    labels_df = gt_store.load_labels(dataset_id)

    if args.no_imshow:
        use_imshow = False
    elif not _can_attempt_imshow():
        use_imshow = False
        print(
            "[INFO] 目前環境未偵測到可用圖形顯示（DISPLAY / WAYLAND_DISPLAY）。"
        )
        print("[INFO] 已自動停用 cv2.imshow()，改為純終端標註模式。")
    else:
        use_imshow = _prompt_yes_no(
            "是否使用 cv2.imshow() 顯示圖片提示？", default=True
        )

    selected_image_ids = _resolve_image_ids(
        asset.instances,
        labels_df,
        explicit_ids=args.image_id,
        from_csv=args.from_csv,
        skip_labeled=args.skip_labeled,
    )
    if not selected_image_ids:
        print("沒有需要標註的圖片，結束。")
        return

    print(f"\nDataset ID: {dataset_id}")
    print(f"待處理影像數: {len(selected_image_ids)}")
    if args.image_id or args.from_csv or not args.skip_labeled:
        print(f"待處理 image_ids: {selected_image_ids}")
    else:
        print("將依序自動處理尚未完成標註的圖片。")
    print()

    try:
        for image_id in selected_image_ids:
            image_rows = asset.instances.loc[
                asset.instances["image_id"] == image_id
            ].sort_values(by=["annotation_id"], kind="stable")

            if image_rows.empty:
                logger.warning("image_id=%s 不在 dataset asset 中，跳過。", image_id)
                continue

            image_path = _resolve_image_path(
                asset.manifest,
                image_rows.iloc[0],
            )
            preview_ok = _show_image_preview(
                image_path=image_path,
                image_rows=image_rows,
                enabled=use_imshow,
            )
            if use_imshow and not preview_ok:
                use_imshow = False

            print("=" * 72)
            print(f"Image ID: {image_id} | Path: {image_path}")
            print(f"Annotations: {len(image_rows)}")
            print("=" * 72)

            used_ranks = _build_used_ranks(labels_df, image_id)

            for _, row in image_rows.iterrows():
                annotation_id = int(row["annotation_id"])
                existing = _get_existing_label(labels_df, annotation_id)

                if (
                    existing is not None
                    and args.skip_labeled
                    and _is_completed_label(existing)
                ):
                    print(f"[SKIP] annotation_id={annotation_id} 已有既有標註。")
                    continue

                if existing is not None and not args.overwrite:
                    print(
                        f"[INFO] annotation_id={annotation_id} 已存在標註 "
                        f"(status={existing.get('label_status', '')})."
                    )
                    if not _prompt_yes_no("是否覆寫這筆標註？", default=False):
                        print("[KEEP] 保留既有標註，略過。\n")
                        continue

                result = _annotate_one(
                    row=row,
                    annotator=args.annotator,
                    used_ranks=used_ranks,
                )
                if result == ACTION_QUIT:
                    print("已收到 quit 指令，安全結束。")
                    return
                if result is None:
                    continue

                gt_store.upsert_label(dataset_id, result)
                labels_df = gt_store.load_labels(dataset_id)

                if result["label_status"] == "LABELED":
                    used_ranks[result["annotation_id"]] = int(result["depth_rank"])
                    print(
                        f"[SAVED] annotation_id={annotation_id} "
                        f"depth_rank={result['depth_rank']}\n"
                    )
                else:
                    print(f"[SAVED] annotation_id={annotation_id} 已標記為 SKIPPED\n")
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _prompt_dataset_id(asset_loader: DatasetAssetLoader) -> str:
    """Prompt user to choose a dataset asset id."""
    dataset_ids = asset_loader.list_dataset_ids()
    if not dataset_ids:
        raise FileNotFoundError(
            f"找不到 dataset assets。請先執行 run_data_pipeline.py。\n"
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


def _resolve_image_ids(
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
        return _load_image_ids_from_csv(from_csv)

    all_image_ids = sorted(int(v) for v in instances["image_id"].unique())
    if skip_labeled:
        pending_ids = []
        labeled_ann_ids = _completed_annotation_ids(labels_df)
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

    parsed = _parse_int_list(raw)
    if not parsed:
        print("輸入無法解析，改用預設候選清單。")
        return pending_ids
    return parsed


def _load_image_ids_from_csv(csv_path: str) -> list[int]:
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


def _show_image_preview(
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
        color = _color_for_category(str(row["category"]))
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


def _can_attempt_imshow() -> bool:
    """Return True when the current environment likely supports GUI windows."""
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
    return True


def _resolve_image_path(
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


def _annotate_one(
    row: pd.Series,
    annotator: str,
    used_ranks: dict[int, int],
) -> dict[str, Any] | str | None:
    """Interactively annotate one object."""
    annotation_id = int(row["annotation_id"])
    category = str(row["category"])
    bbox = (
        float(row["bbox_x"]),
        float(row["bbox_y"]),
        float(row["bbox_w"]),
        float(row["bbox_h"]),
    )

    while True:
        print(f"\n[Annotation #{annotation_id}] {category} bbox={bbox}")
        print("輸入格式: x,y")
        print("特殊指令: skip / redo / quit")

        left_eye = _prompt_coordinate("左眼座標")
        if left_eye in (ACTION_QUIT, ACTION_SKIP):
            return _special_result(row, annotator, left_eye)
        if left_eye == ACTION_REDO:
            continue

        right_eye = _prompt_coordinate("右眼座標")
        if right_eye in (ACTION_QUIT, ACTION_SKIP):
            return _special_result(row, annotator, right_eye)
        if right_eye == ACTION_REDO:
            continue

        depth_rank = _prompt_depth_rank(used_ranks, annotation_id)
        if depth_rank in (ACTION_QUIT, ACTION_SKIP):
            return _special_result(row, annotator, depth_rank)
        if depth_rank == ACTION_REDO:
            continue

        print(
            "確認標註: "
            f"L={left_eye}, R={right_eye}, depth_rank={depth_rank}"
        )
        confirm = input("按 Enter 儲存，或輸入 redo / skip / quit: ").strip().lower()
        if confirm == "":
            return {
                "dataset_id": row["dataset_id"],
                "image_id": int(row["image_id"]),
                "annotation_id": annotation_id,
                "category": category,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "left_eye_x": left_eye[0],
                "left_eye_y": left_eye[1],
                "right_eye_x": right_eye[0],
                "right_eye_y": right_eye[1],
                "depth_rank": int(depth_rank),
                "label_status": "LABELED",
                "annotator": annotator,
                "labeled_at": _now_iso(),
                "notes": "",
            }
        if confirm == "redo":
            continue
        if confirm == "skip":
            return _special_result(row, annotator, ACTION_SKIP)
        if confirm == "quit":
            return ACTION_QUIT
        print("無法辨識的指令，重新輸入目前 annotation。")


def _prompt_coordinate(prompt: str) -> tuple[float, float] | str:
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


def _prompt_depth_rank(
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
            if not _prompt_yes_no("仍要使用這個 rank 嗎？", default=False):
                continue

        return rank


def _special_result(
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
        "labeled_at": _now_iso(),
        "notes": "",
    }


def _get_existing_label(
    labels_df: pd.DataFrame,
    annotation_id: int,
) -> dict[str, Any] | None:
    """Return existing label row for one annotation, if present."""
    matched = labels_df.loc[labels_df["annotation_id"] == annotation_id]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def _completed_annotation_ids(labels_df: pd.DataFrame) -> set[int]:
    """Return annotation ids considered fully labeled."""
    if labels_df.empty or "label_status" not in labels_df.columns:
        return set()

    completed = labels_df.loc[
        labels_df["label_status"].isin(COMPLETED_LABEL_STATUSES)
    ]
    return set(int(value) for value in completed["annotation_id"].tolist())


def _is_completed_label(existing: dict[str, Any]) -> bool:
    """Return True when an existing label should count as completed."""
    status = str(existing.get("label_status", "")).strip().upper()
    return status in COMPLETED_LABEL_STATUSES


def _build_used_ranks(
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


def _prompt_yes_no(prompt: str, default: bool) -> bool:
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


def _parse_int_list(raw: str) -> list[int]:
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


def _color_for_category(category: str) -> tuple[int, int, int]:
    """Return deterministic BGR color for one category."""
    hue = hash(category) % 180
    hsv_arr = np.array([[[hue, 220, 230]]], dtype=np.uint8)
    bgr_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2BGR)
    return tuple(int(v) for v in bgr_arr[0, 0])


def _now_iso() -> str:
    """Return current UTC timestamp string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中止。")
        sys.exit(130)
