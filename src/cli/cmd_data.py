"""Sub-command: data — Phase 1 COCO download, filter, and asset export."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from src.cli._shared_parsers import coco_data_parser

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``data`` sub-command."""
    parser = subparsers.add_parser(
        "data",
        help="Phase 1: COCO 下載 → 過濾 → Dataset Asset 匯出",
        parents=[coco_data_parser()],
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=10,
        help="視覺化抽檢數量，0=跳過 (default: 10)",
    )
    parser.add_argument(
        "--visualize-all",
        action="store_true",
        help="輸出所有通過篩選的圖片的 debug 視覺化（覆蓋 --visualize）",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the Phase 1 data pipeline."""
    from src.data import AutoDownloader, COCODataLoader, DatasetAssetExporter
    from src.utils.visualization import debug_visualize

    # ---- Step 1: Download (unless skipped) ----
    if not args.skip_download:
        logger.info("=" * 60)
        logger.info("Step 1/5: 檢查並下載 COCO val2017 資料集")
        logger.info("=" * 60)
        downloader = AutoDownloader(config)
        downloader.ensure_ready()
    else:
        logger.info("已跳過下載檢查 (--skip-download)")

    # ---- Step 2: Load and filter ----
    logger.info("=" * 60)
    logger.info("Step 2/5: 載入標註並執行過濾")
    logger.info("=" * 60)

    t_start = time.time()

    loader = COCODataLoader(
        data_root=config["coco"]["data_root"],
        target_categories=args.categories,
        config=config,
        auto_download=False,
    )
    dataset = loader.load_filtered_dataset()

    t_filter = time.time() - t_start

    # ---- Step 3: Export CSV ----
    logger.info("=" * 60)
    logger.info("Step 3/5: 匯出基準索引 CSV")
    logger.info("=" * 60)

    csv_path = config.get("output", {}).get("csv_path", "output/test_samples.csv")
    loader.export_csv(dataset, csv_path)

    # ---- Step 4: Export dataset asset ----
    logger.info("=" * 60)
    logger.info("Step 4/5: 匯出 Dataset Asset")
    logger.info("=" * 60)

    exporter = DatasetAssetExporter(config)
    asset_info = exporter.export(
        dataset=dataset,
        target_categories=loader.target_categories,
    )

    # ---- Step 5: Visualization ----
    vis_count = len(dataset) if args.visualize_all else args.visualize
    if vis_count > 0:
        mode_label = "全部" if args.visualize_all else f"抽檢 {vis_count}"
        logger.info("=" * 60)
        logger.info("Step 5/5: 視覺化 — %s (%d 張)", mode_label, vis_count)
        logger.info("=" * 60)

        debug_dir = config.get("output", {}).get("debug_dir", "output/debug")
        saved = debug_visualize(
            dataset,
            output_dir=debug_dir,
            sample_count=vis_count,
        )
        logger.info("圖片儲存至: %s (%d 張)", debug_dir, len(saved))
    else:
        logger.info("已跳過視覺化 (--visualize 0)")

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Pipeline 完成！摘要如下：")
    logger.info("=" * 60)
    logger.info("  通過篩選的圖片數: %d", len(dataset))
    total_instances = 0
    if dataset:
        total_instances = sum(len(r["annotations"]) for r in dataset)
        all_cats = set()
        for r in dataset:
            for ann in r["annotations"]:
                all_cats.add(ann["category"])
        logger.info("  動物實體總數:     %d", total_instances)
        logger.info("  涵蓋類別:         %s", sorted(all_cats))
        avg_instances = total_instances / len(dataset)
    else:
        avg_instances = 0.0
        logger.info("  動物實體總數:     0")
        logger.info("  涵蓋類別:         []")
    logger.info("  平均每圖實體數:   %.1f", avg_instances)
    logger.info("  過濾耗時:         %.2f 秒", t_filter)
    logger.info("  CSV 匯出路徑:     %s", csv_path)
    logger.info("  Dataset ID:       %s", asset_info.dataset_id)
    logger.info("  Dataset Asset:    %s", asset_info.asset_dir)
    logger.info("=" * 60)
