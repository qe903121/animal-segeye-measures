"""Sub-command: review — GT visual inspection and overlay export."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.cli._shared_parsers import dataset_id_parser

logger = logging.getLogger(__name__)

REVIEW_DEFAULT_OUTPUT_DIR = "output/debug_labels"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``review`` sub-command."""
    parser = subparsers.add_parser(
        "review",
        help="Human GT 標註視覺檢查",
        parents=[dataset_id_parser()],
    )
    parser.add_argument(
        "--image-id",
        nargs="+",
        type=int,
        default=None,
        help="僅檢查指定 image_id，可傳多個",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="從 CSV 讀取 image_id 清單",
    )
    parser.add_argument(
        "--no-imshow",
        action="store_true",
        help="不使用 cv2.imshow 顯示圖片",
    )
    parser.add_argument(
        "--review-output-dir",
        type=str,
        default=REVIEW_DEFAULT_OUTPUT_DIR,
        help=f"review 模式輸出圖片目錄 (default: {REVIEW_DEFAULT_OUTPUT_DIR})",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the GT review workflow."""
    import cv2

    from src.cli._annotation_helpers import (
        WINDOW_NAME,
        resolve_image_path,
        resolve_imshow_usage,
        resolve_review_image_ids,
        build_review_rows,
        render_review_canvas,
        prompt_dataset_id,
    )
    from src.data import DatasetAssetLoader, HumanLabelStore

    asset_loader = DatasetAssetLoader(config)
    gt_store = HumanLabelStore(config)

    dataset_id = args.dataset_id or prompt_dataset_id(asset_loader)
    asset = asset_loader.load(dataset_id)
    labels_df = gt_store.load_labels(dataset_id)

    use_imshow = resolve_imshow_usage(
        no_imshow=args.no_imshow,
        mode="review",
    )

    try:
        _run_review(
            asset=asset,
            labels_df=labels_df,
            explicit_ids=args.image_id,
            from_csv=args.from_csv,
            output_dir=args.review_output_dir,
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


def _run_review(
    asset: Any,
    labels_df: Any,
    explicit_ids: list[int] | None,
    from_csv: str | None,
    output_dir: str,
    use_imshow: bool,
) -> None:
    """Core review loop."""
    if labels_df.empty:
        print("目前沒有 human labels 可供檢查，結束。")
        return

    review_rows = build_review_rows(asset.instances, labels_df)
    selected_image_ids = resolve_review_image_ids(
        review_rows,
        explicit_ids=explicit_ids,
        from_csv=from_csv,
    )
    if not selected_image_ids:
        print("沒有可供 review 的圖片，結束。")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset ID: {asset.dataset_id}")
    print(f"待檢查影像數: {len(selected_image_ids)}")
    print(f"Review 輸出目錄: {output_path}\n")

    saved_paths: list[Path] = []

    for index, image_id in enumerate(selected_image_ids, start=1):
        image_rows = review_rows.loc[
            review_rows["image_id"] == image_id
        ].sort_values(by=["annotation_id"], kind="stable")
        if image_rows.empty:
            continue

        image_path = resolve_image_path(asset.manifest, image_rows.iloc[0])
        canvas = render_review_canvas(image_path, image_rows)
        if canvas is None:
            continue

        out_file = output_path / f"labels_{image_id}.jpg"
        cv2.imwrite(str(out_file), canvas)
        saved_paths.append(out_file)

        print(
            f"[REVIEW {index}/{len(selected_image_ids)}] "
            f"image_id={image_id} -> {out_file}"
        )

        if use_imshow:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1)
            except cv2.error as exc:
                print(f"[WARN] cv2.imshow() 失敗，改為純輸出模式: {exc}")
                use_imshow = False
            else:
                action = input("按 Enter 看下一張，或輸入 q 結束 review: ").strip().lower()
                if action == "q":
                    break

    print(f"\nReview 完成，已輸出 {len(saved_paths)} 張圖片到 {output_path}")
