"""Unified Evaluation CLI — Entry Point.

Loads a dataset (fresh Phase 1 pipeline, frozen dataset asset, or mock),
registers validators with the :class:`~src.evaluation.EvaluationEngine`,
and executes the requested evaluation task(s).

Usage examples::

    # Run the full real pipeline: download/check → filter → eye detection
    # → localization report → measurement report
    python run_evaluate.py --task pipeline --method ai --output-dir output/eval

    # Reuse an exported dataset asset to keep evaluation aligned with GT
    python run_evaluate.py --task localization --method ai --dataset-id <id>

    # Run only the measurement stage
    python run_evaluate.py --task measurement --method ai --output-dir output/eval

    # Quick test with mock data (no COCO download, no inference)
    python run_evaluate.py --task pipeline --mock --output-dir /tmp/eval_test

    # Verbose logging
    python run_evaluate.py --task pipeline --method ai --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.data import AutoDownloader, COCODataLoader, DatasetAssetLoader
from src.data.loader import ImageRecord
from src.evaluation import (
    EvaluationEngine,
    LocalizationValidator,
    MeasurementValidator,
)
from src.localization import create_detector
from src.utils.cli import setup_logging, load_config

TASK_PIPELINE = "pipeline"
TASK_ALL = "all"
TASK_LOCALIZATION = "localization"
TASK_MEASUREMENT = "measurement"
TASK_CHOICES = [
    TASK_PIPELINE,
    TASK_ALL,
    TASK_LOCALIZATION,
    TASK_MEASUREMENT,
]

# ------------------------------------------------------------------
# Dataset preparation
# ------------------------------------------------------------------


def _load_real_dataset(
    config: dict,
    method: str,
    categories: list[str] | None = None,
    skip_download: bool = False,
    dataset_id: str | None = None,
) -> list[ImageRecord]:
    """Run Phase 2 on either a fresh Phase 1 dataset or a frozen asset.

    Args:
        config: Full parsed config dictionary.
        method: Detection method (``"cv"`` or ``"ai"``).
        categories: Optional target animal categories override.
        skip_download: If True, do not run :class:`AutoDownloader`.
        dataset_id: Optional Dataset Asset id. When provided, Phase 1
            membership is reconstructed from the exported asset instead
            of re-running filter selection.

    Returns:
        Dataset with ``eyes`` fields populated by the detector.

    Raises:
        SystemExit: If Phase 1 filtering yields zero results.
    """
    logger = logging.getLogger(__name__)

    if not skip_download:
        logger.info("檢查 COCO 資料是否已就緒...")
        AutoDownloader(config).ensure_ready()
    else:
        logger.info("已跳過下載檢查 (--skip-download)")

    if dataset_id:
        if categories:
            logger.warning(
                "已指定 --dataset-id，將忽略 --categories，改用 asset 固定成員。"
            )

        logger.info("載入固定 Dataset Asset: %s", dataset_id)
        asset = DatasetAssetLoader(config).load(dataset_id)
        asset_categories = asset.manifest.get("requested_categories") or categories
        loader = COCODataLoader(
            data_root=config["coco"]["data_root"],
            target_categories=asset_categories,
            config=config,
            auto_download=False,
        )
        dataset = loader.load_dataset_from_instances(asset.instances)
    else:
        # Phase 1: Load and filter
        logger.info("載入 Phase 1 篩選資料集...")
        loader = COCODataLoader(
            data_root=config["coco"]["data_root"],
            target_categories=categories,
            config=config,
            auto_download=False,
        )
        dataset = loader.load_filtered_dataset()

    if not dataset:
        logger.error("資料集為空，無法進行驗證。")
        sys.exit(1)

    # Phase 2: Eye detection
    logger.info("執行 Phase 2 眼睛偵測 (method=%s)...", method)
    detector = create_detector(method, config)
    dataset = detector.process_dataset(dataset)

    return dataset


def _build_mock_dataset() -> list[ImageRecord]:
    """Generate a small mock dataset for testing validator logic.

    Returns a minimal dataset with synthetic ``eyes`` results so that
    the evaluation framework can be tested without COCO data or model
    inference. **Images are not real files**, so debug visualisation
    will be skipped gracefully.

    Returns:
        A list of 3 mock :class:`ImageRecord` entries containing
        6 annotations with varying detection statuses.
    """
    import numpy as np

    logger = logging.getLogger(__name__)
    logger.info("使用 mock 資料集（不執行真實推論）")

    # Create a tiny blank mask for type consistency
    mock_mask = np.zeros((100, 100), dtype=np.uint8)

    dataset: list[ImageRecord] = [
        {
            "image_id": 99901,
            "image_path": "/mock/image_99901.jpg",
            "image_size": (640, 480),
            "annotations": [
                {
                    "id": 1001,
                    "category": "cat",
                    "bbox": [100.0, 50.0, 200.0, 180.0],
                    "mask": mock_mask,
                    "contours": [],
                    "eyes": {
                        "status": "SUCCESS",
                        "left_eye": [150.0, 100.0],
                        "right_eye": [220.0, 105.0],
                        "confidence": 0.92,
                    },
                },
                {
                    "id": 1002,
                    "category": "dog",
                    "bbox": [350.0, 80.0, 180.0, 160.0],
                    "mask": mock_mask,
                    "contours": [],
                    "eyes": {
                        "status": "SUCCESS",
                        "left_eye": [390.0, 130.0],
                        "right_eye": [460.0, 135.0],
                        "confidence": 0.88,
                    },
                },
            ],
        },
        {
            "image_id": 99902,
            "image_path": "/mock/image_99902.jpg",
            "image_size": (640, 480),
            "annotations": [
                {
                    "id": 1003,
                    "category": "cat",
                    "bbox": [50.0, 30.0, 150.0, 130.0],
                    "mask": mock_mask,
                    "contours": [],
                    "eyes": {
                        "status": "SINGLE_EYE",
                        "left_eye": [100.0, 70.0],
                        "right_eye": None,
                        "confidence": 0.45,
                    },
                },
                {
                    "id": 1004,
                    "category": "dog",
                    "bbox": [300.0, 60.0, 200.0, 170.0],
                    "mask": mock_mask,
                    "contours": [],
                    "eyes": {
                        "status": "FAILED_NOT_FOUND",
                        "left_eye": None,
                        "right_eye": None,
                        "confidence": 0.0,
                    },
                },
            ],
        },
        {
            "image_id": 99903,
            "image_path": "/mock/image_99903.jpg",
            "image_size": (640, 480),
            "annotations": [
                {
                    "id": 1005,
                    "category": "dog",
                    "bbox": [80.0, 40.0, 250.0, 200.0],
                    "mask": mock_mask,
                    "contours": [],
                    "eyes": {
                        "status": "SUCCESS",
                        "left_eye": [160.0, 110.0],
                        "right_eye": [250.0, 115.0],
                        "confidence": 0.95,
                    },
                },
                {
                    "id": 1006,
                    "category": "cat",
                    "bbox": [400.0, 100.0, 140.0, 120.0],
                    "mask": mock_mask,
                    "contours": [],
                    "eyes": {
                        "status": "SUCCESS",
                        "left_eye": [430.0, 140.0],
                        "right_eye": [490.0, 145.0],
                        "confidence": 0.87,
                    },
                },
            ],
        },
    ]

    total = sum(len(r["annotations"]) for r in dataset)
    logger.info(
        "Mock 資料集: %d 張圖片 / %d 個標註 (4 SUCCESS, 1 SINGLE, 1 FAILED)",
        len(dataset), total,
    )
    return dataset


# ------------------------------------------------------------------
# CLI argument parser
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Unified Evaluation: 全 pipeline CLI（資料 → 偵測 → 量測 → 驗證）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python run_evaluate.py --task pipeline --method ai\n"
            "  python run_evaluate.py --task measurement --method ai\n"
            "  python run_evaluate.py --task pipeline --mock\n"
            "  python run_evaluate.py --task localization --method ai --dataset-id <id>\n"
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_CHOICES,
        default=TASK_PIPELINE,
        help=(
            "執行任務: pipeline=完整流程, all=所有已註冊驗證器, "
            "localization=眼睛定位, measurement=量測層"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/eval",
        help="報告輸出目錄 (default: output/eval)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ai",
        choices=["cv", "ai"],
        help="偵測方法，僅真實模式有效 (default: ai)",
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
        "--dataset-id",
        type=str,
        default=None,
        help="使用既有 Dataset Asset id 重建 Phase 1 成員，而非重新過濾",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 mock 資料（不執行下載或推論，純測試 validator 邏輯）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳過 COCO 下載檢查 (已有資料時加速)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="啟用詳細日誌 (DEBUG level)",
    )
    return parser


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Execute the unified evaluation pipeline.

    Steps:
        1. Parse CLI arguments and load config.
        2. Build :class:`EvaluationEngine` and register validators.
        3. Prepare dataset (real pipeline or mock).
        4. Run the requested evaluation task(s).
        5. Print summary.
    """
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # ---- Step 1: Load config ----
    config = load_config(args.config)

    # ---- Step 2: Build engine and register validators ----
    engine = EvaluationEngine(config)
    engine.register(TASK_LOCALIZATION, LocalizationValidator)
    engine.register(TASK_MEASUREMENT, MeasurementValidator)

    logger.info(
        "已註冊 %d 個驗證器: %s",
        len(engine.registered_tasks),
        engine.registered_tasks,
    )

    # ---- Step 3: Prepare dataset ----
    logger.info("=" * 60)
    if args.mock:
        logger.info("模式: MOCK（不執行真實推論）")
        dataset = _build_mock_dataset()
    else:
        logger.info("模式: REAL PIPELINE (method=%s)", args.method)
        dataset = _load_real_dataset(
            config,
            args.method,
            categories=args.categories,
            skip_download=args.skip_download,
            dataset_id=args.dataset_id,
        )
    logger.info("=" * 60)

    total_annotations = sum(len(r["annotations"]) for r in dataset)
    logger.info(
        "資料集: %d 張圖片 / %d 個動物實體",
        len(dataset), total_annotations,
    )

    # ---- Step 4: Run evaluation ----
    output_dir = Path(args.output_dir)
    t_start = time.time()

    requested_task = TASK_ALL if args.task == TASK_PIPELINE else args.task

    if requested_task == TASK_ALL:
        all_metrics = engine.run_all(dataset, output_dir)
    else:
        all_metrics = {
            requested_task: engine.run(requested_task, dataset, output_dir)
        }

    t_eval = time.time() - t_start

    # ---- Step 5: Summary ----
    logger.info("=" * 60)
    logger.info("驗證完成！摘要如下：")
    logger.info("=" * 60)
    logger.info("  執行模式:     %s", "MOCK" if args.mock else "REAL")
    logger.info("  執行任務:     %s", list(all_metrics.keys()))
    logger.info("  輸出目錄:     %s", output_dir)
    logger.info("  驗證耗時:     %.2f 秒", t_eval)

    for task_name, metrics in all_metrics.items():
        if metrics.get("error"):
            logger.error("  [%s] 執行失敗", task_name)
        elif task_name == TASK_LOCALIZATION:
            sr = metrics.get("success_rate", "N/A")
            logger.info("  [%s] Success Rate: %s%%", task_name, sr)
        elif task_name == TASK_MEASUREMENT:
            valid_eye = metrics.get("valid_eye_measurements", "N/A")
            total_eye = metrics.get("total_instances", "N/A")
            valid_pair = metrics.get("valid_pairs", "N/A")
            total_pair = metrics.get("total_pairs", "N/A")
            logger.info(
                "  [%s] Eye Distances: %s/%s, Pairwise Proxy: %s/%s",
                task_name,
                valid_eye,
                total_eye,
                valid_pair,
                total_pair,
            )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
