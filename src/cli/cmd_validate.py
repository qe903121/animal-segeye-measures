"""Sub-command: validate — compare GT against a saved Prediction Asset.

This user-facing command owns GT-side validation work.

Conceptual contract:
    Dataset Asset (A) + Human GT (B) + Prediction Asset (C) -> Report (D)
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


class ValidateCommand(BaseCLICommand):
    """GT-side validation command consuming saved prediction assets."""

    name = "validate"
    help = "GT Validation: Human GT + Prediction Asset -> accuracy report"

    def get_parser_kwargs(self) -> dict[str, Any]:
        return {"parents": [dataset_id_parser()]}

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--prediction-run-id",
            type=str,
            default=None,
            help="Use existing Prediction Asset ID",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="output/validate",
            help="validation report output directory (default: output/validate)",
        )

    def execute(
        self,
        args: argparse.Namespace,
        context: CommandContext,
    ) -> None:
        main(args, context.config)


COMMAND = ValidateCommand()


def register(subparsers: argparse._SubParsersAction) -> None:
    """Compatibility wrapper for legacy register-style imports."""
    COMMAND.register(subparsers)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute GT-based validation against one saved Prediction Asset.

    Long-term contract:
        Dataset Asset (A) + Human GT (B) + Prediction Asset (C) -> Report (D)
    """
    from src.data import (
        DatasetAssetLoader,
        PredictionAssetLoader,
        build_lightweight_dataset_from_asset,
    )
    from src.evaluation import AccuracyValidator, EvaluationEngine
    from src.utils.cli import SummaryItem, log_banner, log_summary

    if not args.dataset_id:
        print("Error: validate requires --dataset-id", file=sys.stderr)
        sys.exit(2)
    if not args.prediction_run_id:
        print("Error: validate requires --prediction-run-id", file=sys.stderr)
        sys.exit(2)

    try:
        prediction_asset = PredictionAssetLoader(config).load(
            args.prediction_run_id
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail_cli(
            "Error: Cannot load Prediction Asset "
            f"'{args.prediction_run_id}': {exc}"
        )

    prediction_dataset_id = str(
        prediction_asset.meta.get("dataset_id", "")
    ).strip()
    if not prediction_dataset_id:
        _fail_cli(
            f"Error: Prediction Asset {args.prediction_run_id} is missing dataset_id."
        )
    if args.dataset_id != prediction_dataset_id:
        _fail_cli(
            "Error: mismatch between --dataset-id and --prediction-run-id dataset_ids:"
            f" {args.dataset_id} != {prediction_dataset_id}",
        )

    try:
        dataset_asset = DatasetAssetLoader(config).load(args.dataset_id)
    except (FileNotFoundError, ValueError) as exc:
        _fail_cli(f"Error: Cannot load Dataset Asset '{args.dataset_id}': {exc}")

    gt_root = Path(
        config.get("assets", {}).get(
            "ground_truth_root",
            "assets/ground_truth",
        )
    )
    gt_path = gt_root / args.dataset_id / "human_labels.csv"
    if not gt_path.is_file():
        _fail_cli(
            f"Error: validate cannot find corresponding Human GT: {gt_path}"
        )

    log_banner(logger, "Validation Mode")
    logger.info(
        "Validation Mode: dataset_id=%s, prediction_run_id=%s",
        args.dataset_id,
        args.prediction_run_id,
    )
    logger.info("Validation Contract: Dataset Asset + Human GT + Prediction Asset -> Report")

    dataset = build_lightweight_dataset_from_asset(dataset_asset)

    engine = EvaluationEngine(config)
    engine.register("accuracy", AccuracyValidator)

    output_dir = Path(args.output_dir)
    t_start = time.time()
    metrics = engine.run(
        "accuracy",
        dataset,
        output_dir,
        prediction_asset=prediction_asset,
    )
    t_total = time.time() - t_start

    nme = metrics.get("nme_mean", "N/A")
    rde = metrics.get("rde_mean_percent", "N/A")
    pair_acc = metrics.get("pairwise_acc_percent", "N/A")
    if isinstance(nme, float):
        nme = f"{nme:.4f}"
    if isinstance(rde, float):
        rde = f"{rde:.1f}%"
    if isinstance(pair_acc, float):
        pair_acc = f"{pair_acc:.1f}%"

    log_summary(
        logger,
        "Validation Finished! Summary:",
        [
            SummaryItem("Dataset ID", args.dataset_id),
            SummaryItem("Prediction Run ID", args.prediction_run_id),
            SummaryItem("Output Dir", output_dir),
            SummaryItem("NME", nme),
            SummaryItem("RDE", rde),
            SummaryItem("Pairwise Accuracy", pair_acc),
            SummaryItem("Total Time", f"{t_total:.2f} s"),
        ],
    )


def _fail_cli(message: str) -> None:
    """Exit the current CLI command with a concise user-facing error."""
    print(message, file=sys.stderr)
    sys.exit(2)
