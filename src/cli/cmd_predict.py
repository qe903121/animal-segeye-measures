"""Sub-command: predict — generate prediction assets from a frozen dataset.

This user-facing command owns all prediction-side work:

- Phase 2 eye localization
- Phase 3 measurement generation
- formal Prediction Asset export

Conceptual contract:
    Dataset Asset (A) -> Prediction Asset (C)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.cli._shared_parsers import dataset_id_parser
from src.utils.cli import BaseCLICommand, CommandContext

logger = logging.getLogger(__name__)


class PredictCommand(BaseCLICommand):
    """Prediction-side command producing formal Prediction Assets."""

    name = "predict"
    help = "Prediction side: 眼睛定位 + 量測 + Prediction Asset 匯出"

    def get_parser_kwargs(self) -> dict[str, Any]:
        return {"parents": [dataset_id_parser()]}

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--skip-download",
            action="store_true",
            help="跳過 COCO 下載檢查 (已有資料時加速)",
        )
        parser.add_argument(
            "--method",
            type=str,
            default="ai",
            choices=["cv", "ai"],
            help="偵測方法 (default: ai)",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="output/predict",
            help="prediction-side 報告輸出目錄 (default: output/predict)",
        )
        parser.add_argument(
            "--run-id",
            type=str,
            default=None,
            help="指定 Prediction Asset 的 run_id；未提供時自動生成",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="允許覆寫同名 Prediction Asset run_id（預設禁止）",
        )

    def execute(
        self,
        args: argparse.Namespace,
        context: CommandContext,
    ) -> None:
        main(args, context.config)


COMMAND = PredictCommand()


def register(subparsers: argparse._SubParsersAction) -> None:
    """Compatibility wrapper for legacy register-style imports."""
    COMMAND.register(subparsers)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the prediction-side workflow."""
    from src.cli.cmd_evaluate import _export_prediction_assets, _load_real_dataset
    from src.data import PredictionAssetStore
    from src.evaluation import (
        EvaluationEngine,
        LocalizationValidator,
        MeasurementValidator,
    )
    from src.utils.cli import SummaryItem, log_banner, log_summary
    t_start = time.time()

    if not args.dataset_id:
        print("錯誤: predict 需要搭配 --dataset-id", file=sys.stderr)
        sys.exit(2)

    if args.run_id:
        store = PredictionAssetStore(config)
        if store.run_exists(args.run_id) and not args.overwrite:
            _fail_cli(
                "錯誤: Prediction Asset run_id 已存在。"
                " 請改用新的 --run-id，或明確加入 --overwrite。"
            )

    log_banner(logger, "Prediction 模式")
    logger.info(
        "Prediction 模式: dataset_id=%s, method=%s",
        args.dataset_id,
        args.method,
    )

    try:
        dataset = _load_real_dataset(
            config,
            args.method,
            categories=None,
            skip_download=args.skip_download,
            dataset_id=args.dataset_id,
            prediction_run_id=None,
            prediction_asset=None,
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail_cli(
            f"錯誤: 無法載入 Dataset Asset '{args.dataset_id}': {exc}"
        )

    engine = EvaluationEngine(config)
    engine.register("localization", LocalizationValidator)
    engine.register("measurement", MeasurementValidator)

    output_dir = Path(args.output_dir)

    localization_metrics = engine.run(
        "localization",
        dataset,
        output_dir,
        prediction_asset=None,
    )
    measurement_metrics = engine.run(
        "measurement",
        dataset,
        output_dir,
        prediction_asset=None,
    )

    try:
        paths = _export_prediction_assets(
            config=config,
            dataset=dataset,
            requested_task="predict",
            dataset_id=args.dataset_id,
            method=args.method,
            run_id=args.run_id,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        _fail_cli(f"錯誤: {exc}")

    t_total = time.time() - t_start

    log_summary(
        logger,
        "Prediction 完成！摘要如下：",
        [
            SummaryItem("Dataset ID", args.dataset_id),
            SummaryItem("Method", args.method),
            SummaryItem("Output Dir", output_dir),
            SummaryItem("Prediction Asset", paths.asset_dir),
            SummaryItem("Prediction Run ID", paths.run_id),
            SummaryItem(
                "Localization SR",
                f"{localization_metrics.get('success_rate', 'N/A')}%",
            ),
            SummaryItem(
                "Eye Distances",
                (
                    f"{measurement_metrics.get('valid_eye_measurements', 'N/A')}/"
                    f"{measurement_metrics.get('total_instances', 'N/A')}"
                ),
            ),
            SummaryItem(
                "Pairwise Proxy",
                (
                    f"{measurement_metrics.get('valid_pairs', 'N/A')}/"
                    f"{measurement_metrics.get('total_pairs', 'N/A')}"
                ),
            ),
            SummaryItem("Total Time", f"{t_total:.2f} 秒"),
        ],
    )


def _fail_cli(message: str) -> None:
    """Exit the current CLI command with a concise user-facing error."""
    print(message, file=sys.stderr)
    sys.exit(2)
