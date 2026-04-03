"""Phase 2 Eye Detection — One-Shot Entry Point.

Loads the Phase 1 filtered dataset, runs eye detection on all animal
instances, then delegates evaluation (statistics + debug visualisation)
to the Unified Evaluation Framework.

Usage examples:
    # Default: CV method
    python run_eye_detection.py

    # AI method (MMPose AP-10K)
    python run_eye_detection.py --method ai

    # Skip data download, verbose logging
    python run_eye_detection.py --skip-download --verbose

    # Custom config
    python run_eye_detection.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.data.downloader import AutoDownloader
from src.data.loader import COCODataLoader
from src.localization import create_detector
from src.evaluation import EvaluationEngine, LocalizationValidator
from src.utils.cli import setup_logging, load_config

# ------------------------------------------------------------------
# CLI argument parser
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2 Eye Detection: 偵測動物雙眼座標",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="設定檔路徑 (default: config/config.yaml)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cv",
        choices=["cv", "ai"],
        help="偵測策略: cv=傳統視覺, ai=深度學習 (default: cv)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="目標動物類別 (default: 使用設定檔中的預設清單)",
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
    """Execute the Phase 2 eye detection pipeline.

    Steps:
        1. Load ``config.yaml``.
        2. Run Phase 1 data pipeline (download + filter).
        3. Create eye detector via factory.
        4. Process all annotations (detect eyes).
        5. Print detection statistics.
        6. (Optional) Generate debug visualizations.
    """
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # ---- Step 1: Load config ----
    config = load_config(args.config)

    # ---- Step 2: Phase 1 — Load filtered dataset ----
    logger.info("=" * 60)
    logger.info("Step 1/3: 載入 Phase 1 篩選後的資料集")
    logger.info("=" * 60)

    if not args.skip_download:
        AutoDownloader(config).ensure_ready()
    else:
        logger.info("已跳過下載檢查 (--skip-download)")

    loader = COCODataLoader(
        data_root=config["coco"]["data_root"],
        target_categories=args.categories,
        config=config,
        auto_download=False,
    )
    dataset = loader.load_filtered_dataset()

    if not dataset:
        logger.error("Phase 1 篩選結果為空，無法進行眼睛偵測。")
        sys.exit(1)

    total_annotations = sum(len(r["annotations"]) for r in dataset)
    logger.info(
        "已載入 %d 張圖片 / %d 個動物實體",
        len(dataset), total_annotations,
    )

    # ---- Step 3: Eye detection ----
    logger.info("=" * 60)
    logger.info("Step 2/3: 執行眼睛偵測 (method=%s)", args.method)
    logger.info("=" * 60)

    t_start = time.time()
    detector = create_detector(args.method, config)
    dataset = detector.process_dataset(dataset)
    t_detect = time.time() - t_start

    # ---- Step 4: Evaluate via Unified Evaluation Framework ----
    logger.info("=" * 60)
    logger.info("Step 3/3: 統一驗證 (Localization Evaluation)")
    logger.info("=" * 60)

    engine = EvaluationEngine(config)
    engine.register("localization", LocalizationValidator)

    eval_output_dir = config.get("output", {}).get(
        "debug_eyes_dir", "output/debug_eyes"
    )
    metrics = engine.run("localization", dataset, Path(eval_output_dir))

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Phase 2 完成！摘要如下：")
    logger.info("=" * 60)
    logger.info("  偵測方法:         %s", args.method.upper())
    logger.info("  處理圖片數:       %d", len(dataset))
    logger.info("  動物實體數:       %d", total_annotations)
    logger.info("  偵測耗時:         %.2f 秒", t_detect)
    logger.info(
        "  雙眼成功率:       %.1f%%", metrics.get("success_rate", 0.0)
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
