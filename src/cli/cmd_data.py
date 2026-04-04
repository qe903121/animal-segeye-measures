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
    help = "COCO Download -> Filter -> Export Dataset Asset"

    def get_parser_kwargs(self) -> dict[str, Any]:
        return {"parents": [coco_data_parser()]}

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--visualize",
            type=int,
            default=10,
            help="Number of samples to visualize, 0 to skip (default: 10)",
        )
        parser.add_argument(
            "--visualize-all",
            action="store_true",
            help="Output debug visualization for all filtered images (overrides --visualize)",
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
        log_banner(logger, "Step 1/5: Checking and downloading COCO val2017 dataset")
        downloader = AutoDownloader(config)
        downloader.ensure_ready()
    else:
        logger.info("Skipped download check (--skip-download)")

    # ---- Step 2: Load and filter ----
    log_banner(logger, "Step 2/5: Loading annotations and filtering")

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
    log_banner(logger, "Step 3/5: Exporting baseline index CSV")

    csv_path = config.get("output", {}).get("csv_path", "output/test_samples.csv")
    loader.export_csv(dataset, csv_path)

    # ---- Step 4: Export dataset asset ----
    log_banner(logger, "Step 4/5: Exporting Dataset Asset")

    exporter = DatasetAssetExporter(config)
    asset_info = exporter.export(
        dataset=dataset,
        target_categories=loader.target_categories,
    )

    # ---- Step 5: Visualization ----
    vis_count = len(dataset) if args.visualize_all else args.visualize
    if vis_count > 0:
        mode_label = "All" if args.visualize_all else f"Sample {vis_count}"
        log_banner(logger, f"Step 5/5: Visualization — {mode_label} ({vis_count} images)")

        data_dir = config.get("output", {}).get(
            "data_dir",
            config.get("output", {}).get("debug_dir", "output/data"),
        )
        saved = debug_visualize(
            dataset,
            output_dir=data_dir,
            sample_count=vis_count,
        )
        logger.info("Images saved to: %s (%d images)", data_dir, len(saved))
    else:
        logger.info("Skipped visualization (--visualize 0)")

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
        "Pipeline Finished! Summary:",
        [
            SummaryItem("Filtered Images", len(dataset)),
            SummaryItem("Total Animal Instances", total_instances),
            SummaryItem("Categories Covered", categories_summary),
            SummaryItem("Avg Instances / Image", f"{avg_instances:.1f}"),
            SummaryItem("Filtering Time", f"{t_filter:.2f} s"),
            SummaryItem("CSV Export Path", csv_path),
            SummaryItem("Dataset ID", asset_info.dataset_id),
            SummaryItem("Dataset Asset", asset_info.asset_dir),
        ],
    )
