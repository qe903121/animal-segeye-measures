"""Advanced internal task-centric evaluation backend.

This module still powers shared helpers used by newer user-facing
commands, but it is no longer the primary operator mental model.

User-facing workflow should prefer:

- ``main.py predict`` for prediction-side work
- ``main.py validate`` for GT-based validation
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.cli._shared_parsers import coco_data_parser, dataset_id_parser
from src.utils.cli import SummaryItem, log_summary

logger = logging.getLogger(__name__)

# ── Task constants ───────────────────────────────────────────────────
TASK_PIPELINE = "pipeline"
TASK_ALL = "all"
TASK_LOCALIZATION = "localization"
TASK_MEASUREMENT = "measurement"
TASK_ACCURACY = "accuracy"
PIPELINE_TASKS = [TASK_LOCALIZATION, TASK_MEASUREMENT]
TASK_CHOICES = [
    TASK_PIPELINE,
    TASK_ALL,
    TASK_LOCALIZATION,
    TASK_MEASUREMENT,
    TASK_ACCURACY,
]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``evaluate`` sub-command."""
    parser = subparsers.add_parser(
        "evaluate",
        help="internal backend: localization → measurement → validation",
        parents=[coco_data_parser(), dataset_id_parser()],
        epilog=(
            "範例:\n"
            "  python main.py evaluate --task pipeline --method ai\n"
            "  python main.py evaluate --task accuracy --dataset-id <id>\n"
            "  python main.py evaluate --task pipeline --mock\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_CHOICES,
        default=TASK_PIPELINE,
        help=(
            "執行任務: pipeline=完整流程, all=所有已註冊驗證器, "
            "localization=眼睛定位, measurement=量測層, accuracy=GT精度評估"
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
        "--prediction-run-id",
        type=str,
        default=None,
        help=(
            "使用既有 Prediction Asset id 回填 saved localization prediction，"
            "跳過 detector inference"
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "將當前 runtime flow 產生的 prediction 匯出為 "
            "assets/predictions/<run_id>/...（正式 Prediction Asset）"
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="指定 Prediction Asset 的 run_id；未提供時自動生成",
    )
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="允許覆寫同名 Prediction Asset run_id（進階/internal 用途）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 mock 資料（不執行下載或推論，純測試 validator 邏輯）",
    )
    parser.set_defaults(func=main)


# ── Main entry ───────────────────────────────────────────────────────


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the unified evaluation pipeline."""
    from src.data import (
        AutoDownloader,
        COCODataLoader,
        DatasetAssetLoader,
        PredictionAssetLoader,
        PredictionAssetStore,
    )
    from src.data.loader import ImageRecord
    from src.evaluation import (
        EvaluationEngine,
        LocalizationValidator,
        MeasurementValidator,
    )
    from src.localization import create_detector
    from src.prediction import (
        build_localization_prediction_rows,
        build_measurement_instance_asset_rows,
        build_measurement_pair_asset_rows,
        build_measurement_prediction,
    )

    gt_path: Path | None = None
    prediction_asset = None

    # ---- Validation ----
    if args.prediction_run_id and args.mock:
        print("錯誤: --prediction-run-id 不能與 --mock 同時使用。", file=sys.stderr)
        sys.exit(2)
    if args.save_predictions and args.prediction_run_id:
        print("錯誤: --save-predictions 不能與 --prediction-run-id 同時使用。", file=sys.stderr)
        sys.exit(2)

    if args.prediction_run_id:
        prediction_asset = PredictionAssetLoader(config).load(args.prediction_run_id)
        prediction_dataset_id = str(prediction_asset.meta.get("dataset_id", "")).strip()
        prediction_method = str(prediction_asset.meta.get("method", "")).strip()
        if not prediction_dataset_id:
            print(
                f"錯誤: Prediction Asset {args.prediction_run_id} 缺少 dataset_id。",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.dataset_id and args.dataset_id != prediction_dataset_id:
            print(
                "錯誤: --dataset-id 與 --prediction-run-id 所屬 dataset_id 不一致："
                f" {args.dataset_id} != {prediction_dataset_id}",
                file=sys.stderr,
            )
            sys.exit(2)
        args.dataset_id = prediction_dataset_id
        if prediction_method:
            args.method = prediction_method
        config["runtime_prediction_run_id"] = args.prediction_run_id
        config["runtime_prediction_asset"] = prediction_asset

    if args.dataset_id:
        config["runtime_dataset_id"] = args.dataset_id
        gt_root = Path(
            config.get("assets", {}).get(
                "ground_truth_root",
                "assets/ground_truth",
            )
        )
        gt_path = gt_root / args.dataset_id / "human_labels.csv"

    if args.task == TASK_ACCURACY:
        if not args.dataset_id:
            print("錯誤: --task accuracy 需要搭配 --dataset-id", file=sys.stderr)
            sys.exit(2)
        if gt_path is None or not gt_path.is_file():
            print(
                f"錯誤: --task accuracy 找不到對應的 Human GT: {gt_path}",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.save_predictions and not args.mock and not args.dataset_id:
        print(
            "錯誤: --save-predictions 需要搭配 --dataset-id",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- Build engine ----
    engine = EvaluationEngine(config)
    engine.register(TASK_LOCALIZATION, LocalizationValidator)
    engine.register(TASK_MEASUREMENT, MeasurementValidator)

    accuracy_enabled = bool(gt_path and gt_path.is_file())
    if accuracy_enabled:
        from src.evaluation.accuracy import AccuracyValidator
        engine.register(TASK_ACCURACY, AccuracyValidator)
    else:
        logger.info(
            "本次未註冊 accuracy validator：未提供有效的 --dataset-id / Human GT。"
        )

    logger.info(
        "已註冊 %d 個驗證器: %s",
        len(engine.registered_tasks),
        engine.registered_tasks,
    )

    # ---- Prepare dataset ----
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
            prediction_run_id=args.prediction_run_id,
            prediction_asset=prediction_asset,
        )
    logger.info("=" * 60)

    total_annotations = sum(len(r["annotations"]) for r in dataset)
    logger.info(
        "資料集: %d 張圖片 / %d 個動物實體",
        len(dataset), total_annotations,
    )

    # ---- Run evaluation ----
    output_dir = Path(args.output_dir)
    t_start = time.time()

    requested_task = args.task

    if requested_task == TASK_PIPELINE:
        all_metrics: dict[str, dict[str, object]] = {}
        for task_name in PIPELINE_TASKS:
            all_metrics[task_name] = engine.run(
                task_name,
                dataset,
                output_dir,
                prediction_asset=prediction_asset,
            )
    elif requested_task == TASK_ALL:
        all_metrics = engine.run_all(
            dataset,
            output_dir,
            prediction_asset=prediction_asset,
        )
    else:
        all_metrics = {
            requested_task: engine.run(
                requested_task,
                dataset,
                output_dir,
                prediction_asset=prediction_asset,
            )
        }

    t_eval = time.time() - t_start

    # ---- Summary ----
    _log_summary(args, all_metrics, output_dir, t_eval)

    if args.save_predictions:
        _export_prediction_assets(
            config=config,
            dataset=dataset,
            requested_task=requested_task,
            dataset_id=args.dataset_id or ("mock_dataset" if args.mock else ""),
            method=args.method,
            run_id=args.run_id,
            overwrite=args.overwrite_predictions,
        )

    logger.info("=" * 60)


# ── Dataset helpers ──────────────────────────────────────────────────


def _load_real_dataset(
    config: dict,
    method: str,
    categories: list[str] | None = None,
    skip_download: bool = False,
    dataset_id: str | None = None,
    prediction_run_id: str | None = None,
    prediction_asset: Any = None,
) -> list:
    """Prepare a runtime dataset from either fresh COCO filtering or a frozen asset."""
    from src.data import AutoDownloader, COCODataLoader, DatasetAssetLoader

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
        logger.info("載入即時過濾資料集...")
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

    if prediction_run_id:
        from src.data import PredictionAssetLoader
        prediction_asset = prediction_asset or PredictionAssetLoader(config).load(
            prediction_run_id
        )
        dataset = _apply_prediction_asset_to_dataset(
            dataset,
            prediction_asset.localization,
        )
        logger.info(
            "已載入 saved prediction，跳過 detector inference: %s",
            prediction_run_id,
        )
    else:
        from src.localization import create_detector

        logger.info("執行眼睛偵測 (method=%s)...", method)
        detector = create_detector(method, config)
        dataset = detector.process_dataset(dataset)

    return dataset


def _build_mock_dataset() -> list:
    """Generate a small mock dataset for testing validator logic."""
    import numpy as np

    logger.info("使用 mock 資料集（不執行真實推論）")
    mock_mask = np.zeros((100, 100), dtype=np.uint8)

    dataset = [
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


def _apply_prediction_asset_to_dataset(dataset: list, localization_df: Any) -> list:
    """Apply saved localization predictions to the runtime dataset."""
    from src.data.prediction_loader import apply_localization_predictions

    return apply_localization_predictions(dataset, localization_df)


# ── Summary / export helpers ─────────────────────────────────────────


def _log_summary(
    args: argparse.Namespace,
    all_metrics: dict[str, dict[str, Any]],
    output_dir: Path,
    t_eval: float,
) -> None:
    """Log a formatted summary of evaluation results."""
    log_summary(
        logger,
        "驗證完成！摘要如下：",
        [
            SummaryItem("執行模式", "MOCK" if args.mock else "REAL"),
            SummaryItem("執行任務", ", ".join(all_metrics.keys())),
            SummaryItem("輸出目錄", output_dir),
            SummaryItem("驗證耗時", f"{t_eval:.2f} 秒"),
        ],
    )

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
        elif task_name == TASK_ACCURACY:
            nme = metrics.get("nme_mean", "N/A")
            rde = metrics.get("rde_mean_percent", "N/A")
            acc = metrics.get("pairwise_acc_percent", "N/A")
            if isinstance(nme, float):
                nme = f"{nme:.4f}"
            if isinstance(rde, float):
                rde = f"{rde:.1f}%"
            if isinstance(acc, float):
                acc = f"{acc:.1f}%"
            logger.info(
                "  [%s] NME: %s, RDE: %s, Pairwise Acc: %s",
                task_name, nme, rde, acc,
            )


def _export_prediction_assets(
    *,
    config: dict,
    dataset: list,
    requested_task: str,
    dataset_id: str,
    method: str,
    run_id: str | None,
    overwrite: bool = False,
) -> Any:
    """Export canonical prediction assets from the current runtime flow."""
    from src.data import PredictionAssetStore
    from src.prediction import (
        build_localization_prediction_rows,
        build_measurement_instance_asset_rows,
        build_measurement_pair_asset_rows,
        build_measurement_prediction,
    )

    store = PredictionAssetStore(config)
    model_name = _resolve_model_name(method, config)
    task_scope = requested_task

    meta = store.build_run_meta(
        dataset_id=dataset_id,
        method=method,
        model_name=model_name,
        task_scope=task_scope,
        run_id=run_id,
        git_commit=_get_git_commit(),
        config_fingerprint_source=config,
    )
    paths = store.initialize_run(meta, overwrite=overwrite)

    localization_rows = build_localization_prediction_rows(
        dataset=dataset,
        run_id=meta["run_id"],
        dataset_id=dataset_id,
        method=method,
        model_name=model_name,
    )
    store.write_localization(paths, localization_rows)

    measurement_tables = build_measurement_prediction(dataset)
    instance_rows = build_measurement_instance_asset_rows(
        tables=measurement_tables,
        run_id=meta["run_id"],
        dataset_id=dataset_id,
        method=method,
        model_name=model_name,
    )
    pair_rows = build_measurement_pair_asset_rows(
        tables=measurement_tables,
        run_id=meta["run_id"],
        dataset_id=dataset_id,
        method=method,
        model_name=model_name,
    )

    store.write_measurement_instances(paths, instance_rows)
    store.write_measurement_pairs(paths, pair_rows)

    logger.info("Prediction Asset 匯出完成: %s", paths.asset_dir)
    return paths


def _resolve_model_name(method: str, config: dict) -> str:
    """Return a stable model label for prediction asset metadata."""
    if method == "ai":
        alias = (
            config.get("eye_detection", {})
            .get("ai_model", {})
            .get("alias", "animal")
        )
        return f"mmpose:{alias}"
    if method == "cv":
        return "heuristic_cv"
    return method


def _get_git_commit() -> str:
    """Best-effort lookup of current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except Exception:
        return ""
