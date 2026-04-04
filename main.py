"""Unified CLI entry point for the Animal SegEye Measures pipeline.

Routes all commands through one OOP-style application lifecycle:

- ``CLIApplication`` owns parser bootstrap and dispatch
- concrete command objects own their parser contracts and execution
- ``CommandContext`` encapsulates shared runtime state

This keeps the router thin while preserving a single canonical operator
entry point.

Usage::

    python main.py [--config CONFIG] [--verbose] <command> [command-args]

Available commands::

    data       Phase 1: COCO download, filter, and dataset asset export
    annotate   Human GT annotation (terminal-only interactive workflow)
    review     Human GT visual review and overlay export
    predict    Phase 2-3: localization + measurement + Prediction Asset export
    validate   GT-based validation from Dataset Asset + Prediction Asset

Examples::

    python main.py data --categories cat dog --skip-download
    python main.py annotate --dataset-id <id> --annotator hsien
    python main.py review --dataset-id <id> --no-imshow
    python main.py predict --dataset-id <id> --method ai --skip-download
    python main.py validate --dataset-id <id> --prediction-run-id <run_id>
"""

from __future__ import annotations

import sys


def main() -> None:
    """Bootstrap the CLI application and run one command lifecycle."""
    from src.cli import build_commands
    from src.utils.cli import CLIApplication

    app = CLIApplication(
        prog="animal-segeye-measures",
        description="Animal Segmentation & Eye Measurement Pipeline",
        commands=build_commands(),
    )
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中止。")
        sys.exit(130)
