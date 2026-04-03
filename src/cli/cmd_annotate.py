"""Sub-command: annotate — interactive human GT annotation."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from src.cli._shared_parsers import dataset_id_parser

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``annotate`` sub-command."""
    parser = subparsers.add_parser(
        "annotate",
        help="Human GT 人工標註 (terminal-only workflow)",
        parents=[dataset_id_parser()],
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
    parser.set_defaults(func=main)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the interactive annotation workflow."""
    import cv2

    from src.cli._annotation_helpers import (
        ACTION_QUIT,
        resolve_imshow_usage,
        resolve_image_ids,
        resolve_image_path,
        show_image_preview,
        get_existing_label,
        is_completed_label,
        build_used_ranks,
        prompt_dataset_id,
        prompt_coordinate,
        prompt_depth_rank,
        prompt_yes_no,
        special_result,
        now_iso,
        ACTION_REDO,
        ACTION_SKIP,
    )
    from src.data import DatasetAssetLoader, HumanLabelStore

    asset_loader = DatasetAssetLoader(config)
    gt_store = HumanLabelStore(config)

    dataset_id = args.dataset_id or prompt_dataset_id(asset_loader)
    asset = asset_loader.load(dataset_id)
    labels_df = gt_store.load_labels(dataset_id)

    use_imshow = resolve_imshow_usage(
        no_imshow=args.no_imshow,
        mode="annotate",
    )

    try:
        _run_annotate(
            asset=asset,
            gt_store=gt_store,
            labels_df=labels_df,
            explicit_ids=args.image_id,
            from_csv=args.from_csv,
            skip_labeled=args.skip_labeled,
            overwrite=args.overwrite,
            annotator=args.annotator,
            use_imshow=use_imshow,
        )
    except KeyboardInterrupt:
        print("\n已中止。")
        sys.exit(130)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _run_annotate(
    asset: Any,
    gt_store: Any,
    labels_df: Any,
    explicit_ids: list[int] | None,
    from_csv: str | None,
    skip_labeled: bool,
    overwrite: bool,
    annotator: str,
    use_imshow: bool,
) -> None:
    """Core annotation loop."""
    selected_image_ids = resolve_image_ids(
        asset.instances,
        labels_df,
        explicit_ids=explicit_ids,
        from_csv=from_csv,
        skip_labeled=skip_labeled,
    )
    if not selected_image_ids:
        print("沒有需要標註的圖片，結束。")
        return

    print(f"\nDataset ID: {asset.dataset_id}")
    print(f"待處理影像數: {len(selected_image_ids)}")
    if explicit_ids or from_csv or not skip_labeled:
        print(f"待處理 image_ids: {selected_image_ids}")
    else:
        print("將依序自動處理尚未完成標註的圖片。")
    print()

    for image_id in selected_image_ids:
        image_rows = asset.instances.loc[
            asset.instances["image_id"] == image_id
        ].sort_values(by=["annotation_id"], kind="stable")

        if image_rows.empty:
            logger.warning("image_id=%s 不在 dataset asset 中，跳過。", image_id)
            continue

        image_path = resolve_image_path(
            asset.manifest,
            image_rows.iloc[0],
        )
        preview_ok = show_image_preview(
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

        used_ranks = build_used_ranks(labels_df, image_id)

        for _, row in image_rows.iterrows():
            annotation_id = int(row["annotation_id"])
            existing = get_existing_label(labels_df, annotation_id)

            if (
                existing is not None
                and skip_labeled
                and is_completed_label(existing)
            ):
                print(f"[SKIP] annotation_id={annotation_id} 已有既有標註。")
                continue

            if existing is not None and not overwrite:
                print(
                    f"[INFO] annotation_id={annotation_id} 已存在標註 "
                    f"(status={existing.get('label_status', '')})."
                )
                if not prompt_yes_no("是否覆寫這筆標註？", default=False):
                    print("[KEEP] 保留既有標註，略過。\n")
                    continue

            result = _annotate_one(
                row=row,
                annotator=annotator,
                used_ranks=used_ranks,
            )
            if result == ACTION_QUIT:
                print("已收到 quit 指令，安全結束。")
                return
            if result is None:
                continue

            gt_store.upsert_label(asset.dataset_id, result)
            labels_df = gt_store.load_labels(asset.dataset_id)

            if result["label_status"] == "LABELED":
                used_ranks[result["annotation_id"]] = int(result["depth_rank"])
                print(
                    f"[SAVED] annotation_id={annotation_id} "
                    f"depth_rank={result['depth_rank']}\n"
                )
            else:
                print(f"[SAVED] annotation_id={annotation_id} 已標記為 SKIPPED\n")


def _annotate_one(
    row: Any,
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

        left_eye = prompt_coordinate("左眼座標")
        if left_eye in (ACTION_QUIT, ACTION_SKIP):
            return special_result(row, annotator, left_eye)
        if left_eye == ACTION_REDO:
            continue

        right_eye = prompt_coordinate("右眼座標")
        if right_eye in (ACTION_QUIT, ACTION_SKIP):
            return special_result(row, annotator, right_eye)
        if right_eye == ACTION_REDO:
            continue

        depth_rank = prompt_depth_rank(used_ranks, annotation_id)
        if depth_rank in (ACTION_QUIT, ACTION_SKIP):
            return special_result(row, annotator, depth_rank)
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
                "labeled_at": now_iso(),
                "notes": "",
            }
        if confirm == "redo":
            continue
        if confirm == "skip":
            return special_result(row, annotator, ACTION_SKIP)
        if confirm == "quit":
            return ACTION_QUIT
        print("無法辨識的指令，重新輸入目前 annotation。")
