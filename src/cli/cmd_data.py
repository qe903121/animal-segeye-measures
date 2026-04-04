"""Sub-command: data — COCO download, filtering, and Dataset Asset export."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from src.cli._shared_parsers import coco_data_parser
from src.utils.cli import BaseCLICommand, CommandContext

logger = logging.getLogger(__name__)


class DataCommand(BaseCLICommand):
    """User-facing dataset-building command."""

    name = "data"
    help = "COCO 下載 → 過濾 → Dataset Asset 匯出"

    def get_parser_kwargs(self) -> dict[str, Any]:
        return {"parents": [coco_data_parser()]}

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
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

    def execute(
        self,
        args: argparse.Namespace,
        context: CommandContext,
    ) -> None:
        main(args, context.config)


COMMAND = DataCommand()


def register(subparsers: argparse._SubParsersAction) -> None:
    """Compatibility wrapper for legacy register-style imports."""
    COMMAND.register(subparsers)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the dataset-building pipeline."""
    from src.data import AutoDownloader, COCODataLoader, DatasetAssetExporter
    from src.utils.cli import SummaryItem, log_banner, log_summary
    from src.utils.visualization import debug_visualize

    # ---- Step 1: Download (unless skipped) ----
    if not args.skip_download:
        log_banner(logger, "Step 1/5: 檢查並下載 COCO val2017 資料集")
        downloader = AutoDownloader(config)
        downloader.ensure_ready()
    else:
        logger.info("已跳過下載檢查 (--skip-download)")

    # ---- Step 2: Load and filter ----
    log_banner(logger, "Step 2/5: 載入標註並執行過濾")

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
    log_banner(logger, "Step 3/5: 匯出基準索引 CSV")

    csv_path = config.get("output", {}).get("csv_path", "output/test_samples.csv")
    loader.export_csv(dataset, csv_path)

    # ---- Step 4: Export dataset asset ----
    log_banner(logger, "Step 4/5: 匯出 Dataset Asset")

    exporter = DatasetAssetExporter(config)
    asset_info = exporter.export(
        dataset=dataset,
        target_categories=loader.target_categories,
    )

    # ---- Step 5: Visualization ----
    vis_count = len(dataset) if args.visualize_all else args.visualize
    if vis_count > 0:
        mode_label = "全部" if args.visualize_all else f"抽檢 {vis_count}"
        log_banner(logger, f"Step 5/5: 視覺化 — {mode_label} ({vis_count} 張)")

        data_dir = config.get("output", {}).get(
            "data_dir",
            config.get("output", {}).get("debug_dir", "output/data"),
        )
        saved = debug_visualize(
            dataset,
            output_dir=data_dir,
            sample_count=vis_count,
        )
        logger.info("圖片儲存至: %s (%d 張)", data_dir, len(saved))
    else:
        logger.info("已跳過視覺化 (--visualize 0)")

    # ---- Summary ----
    total_instances = 0
    if dataset:
        total_instances = sum(len(r["annotations"]) for r in dataset)
        all_cats = set()
        for r in dataset:
            for ann in r["annotations"]:
                all_cats.add(ann["category"])
        avg_instances = total_instances / len(dataset)
        categories_summary = sorted(all_cats)
    else:
        avg_instances = 0.0
        categories_summary = []
    log_summary(
        logger,
        "Pipeline 完成！摘要如下：",
        [
            SummaryItem("通過篩選的圖片數", len(dataset)),
            SummaryItem("動物實體總數", total_instances),
            SummaryItem("涵蓋類別", categories_summary),
            SummaryItem("平均每圖實體數", f"{avg_instances:.1f}"),
            SummaryItem("過濾耗時", f"{t_filter:.2f} 秒"),
            SummaryItem("CSV 匯出路徑", csv_path),
            SummaryItem("Dataset ID", asset_info.dataset_id),
            SummaryItem("Dataset Asset", asset_info.asset_dir),
        ],
    )
