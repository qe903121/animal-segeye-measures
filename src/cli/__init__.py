"""CLI command package for the user-facing router.

The canonical operator entry point is ``main.py``. This package exports a
stable ordered command list so the application lifecycle can register and
dispatch sub-commands polymorphically.
"""

from __future__ import annotations

from src.utils.cli import BaseCLICommand

from .cmd_annotate import COMMAND as ANNOTATE_COMMAND
from .cmd_data import COMMAND as DATA_COMMAND
from .cmd_predict import COMMAND as PREDICT_COMMAND
from .cmd_review import COMMAND as REVIEW_COMMAND
from .cmd_validate import COMMAND as VALIDATE_COMMAND


def build_commands() -> list[BaseCLICommand]:
    """Return the canonical ordered user-facing command set."""
    return [
        DATA_COMMAND,
        ANNOTATE_COMMAND,
        REVIEW_COMMAND,
        PREDICT_COMMAND,
        VALIDATE_COMMAND,
    ]


__all__ = ["build_commands"]
