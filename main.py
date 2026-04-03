"""Unified CLI entry point for the Animal SegEye Measures pipeline.

Routes all commands through a single ``argparse`` sub-command router,
replacing the legacy ``run_*.py`` scripts.

Usage::

    python main.py [--config CONFIG] [--verbose] <command> [command-args]

Available commands::

    data       Phase 1: COCO download, filter, and dataset asset export
    annotate   Human GT annotation (terminal-only interactive workflow)
    review     Human GT visual review and overlay export
    evaluate   Phase 2-4: eye detection, measurement, and validation

Examples::

    python main.py data --categories cat dog --skip-download
    python main.py annotate --dataset-id <id> --annotator hsien
    python main.py review --dataset-id <id> --no-imshow
    python main.py evaluate --task pipeline --method ai --dataset-id <id>
    python main.py --verbose evaluate --task accuracy --dataset-id <id>
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Parse global args, bootstrap logging/config, and dispatch."""
    parser = argparse.ArgumentParser(
        prog="animal-segeye-measures",
        description="Animal Segmentation & Eye Measurement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Global arguments ─────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="設定檔路徑 (default: config/config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="啟用詳細日誌 (DEBUG level)",
    )

    # ── Sub-commands (lazy import to keep --help fast) ─────────────────
    from src.cli import cmd_data, cmd_annotate, cmd_review, cmd_evaluate

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="可用的子命令",
        metavar="<command>",
    )
    subparsers.required = True

    cmd_data.register(subparsers)
    cmd_annotate.register(subparsers)
    cmd_review.register(subparsers)
    cmd_evaluate.register(subparsers)

    # ── Parse, bootstrap, dispatch ───────────────────────────────────
    args = parser.parse_args()

    from src.utils.cli import setup_logging, load_config

    setup_logging(verbose=args.verbose)
    config = load_config(args.config)

    args.func(args, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中止。")
        sys.exit(130)
