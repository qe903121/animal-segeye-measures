"""Phase 1 Data Pipeline — One-Shot Entry Point.

Downloads COCO val2017 (if needed), applies multi-animal filtering,
exports a baseline CSV index, and optionally generates debug visualizations.

Usage examples:
    # Use default config
    python run_data_pipeline.py

    # Specify target categories
    python run_data_pipeline.py --categories dog cat horse

    # Increase debug visualization count
    python run_data_pipeline.py --visualize 20

    # Skip download check (when data is already cached)
    python run_data_pipeline.py --skip-download

    # Custom config file
    python run_data_pipeline.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from src.data.downloader import AutoDownloader
from src.data.loader import COCODataLoader
from src.utils.visualization import debug_visualize

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger with console output.

    Args:
        verbose: If True, set level to DEBUG; else INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def _load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to the ``config.yaml`` file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: Config file does not exist.
        yaml.YAMLError: Config file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"設定檔不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.getLogger(__name__).info("已載入設定檔: %s", path)
    return config


# ------------------------------------------------------------------
# CLI argument parser
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Phase 1 Data Pipeline: COCO val2017 下載 → 過濾 → 匯出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="設定檔路徑 (default: config/config.yaml)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="目標動物類別 (default: 使用設定檔中的預設清單)",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=10,
        help="視覺化抽檢數量，0=跳過 (default: 10)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳過下載檢查 (已有資料時加速)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="啟用詳細日誌 (DEBUG level)",
    )
    return parser


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def main() -> None:
    """Execute the full Phase 1 data pipeline.

    Steps:
        1. Load ``config.yaml``.
        2. (Optional) Run ``AutoDownloader.ensure_ready()``.
        3. Initialize ``COCODataLoader`` and run ``load_filtered_dataset()``.
        4. Export ``test_samples.csv``.
        5. (Optional) Generate debug visualizations.
        6. Print summary statistics to stdout.
    """
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # ---- Step 1: Load config ----
    config = _load_config(args.config)

    # ---- Step 2: Download (unless skipped) ----
    if not args.skip_download:
        logger.info("=" * 60)
        logger.info("Step 1/4: 檢查並下載 COCO val2017 資料集")
        logger.info("=" * 60)
        downloader = AutoDownloader(config)
        downloader.ensure_ready()
    else:
        logger.info("已跳過下載檢查 (--skip-download)")

    # ---- Step 3: Load and filter ----
    logger.info("=" * 60)
    logger.info("Step 2/4: 載入標註並執行過濾")
    logger.info("=" * 60)

    t_start = time.time()

    loader = COCODataLoader(
        data_root=config["coco"]["data_root"],
        target_categories=args.categories,
        config=config,
    )
    dataset = loader.load_filtered_dataset()

    t_filter = time.time() - t_start

    # ---- Step 4: Export CSV ----
    logger.info("=" * 60)
    logger.info("Step 3/4: 匯出基準索引 CSV")
    logger.info("=" * 60)

    csv_path = config.get("output", {}).get("csv_path", "output/test_samples.csv")
    loader.export_csv(dataset, csv_path)

    # ---- Step 5: Visualization ----
    if args.visualize > 0:
        logger.info("=" * 60)
        logger.info("Step 4/4: 視覺化抽檢 (%d 張)", args.visualize)
        logger.info("=" * 60)

        debug_dir = config.get("output", {}).get("debug_dir", "output/debug")
        saved = debug_visualize(
            dataset,
            output_dir=debug_dir,
            sample_count=args.visualize,
        )
        logger.info("抽檢圖片儲存至: %s (%d 張)", debug_dir, len(saved))
    else:
        logger.info("已跳過視覺化抽檢 (--visualize 0)")

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Pipeline 完成！摘要如下：")
    logger.info("=" * 60)
    logger.info("  通過篩選的圖片數: %d", len(dataset))
    if dataset:
        total_instances = sum(len(r["annotations"]) for r in dataset)
        all_cats = set()
        for r in dataset:
            for ann in r["annotations"]:
                all_cats.add(ann["category"])
        logger.info("  動物實體總數:     %d", total_instances)
        logger.info("  涵蓋類別:         %s", sorted(all_cats))
        logger.info("  平均每圖實體數:   %.1f", total_instances / len(dataset))
    logger.info("  過濾耗時:         %.2f 秒", t_filter)
    logger.info("  CSV 匯出路徑:     %s", csv_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
