"""Shared CLI abstractions, lifecycle management, and bootstrap helpers.

This module centralizes the user-facing CLI lifecycle behind a small OOP
surface:

- :class:`CommandContext` encapsulates runtime configuration state.
- :class:`BaseCLICommand` abstracts one sub-command contract.
- :class:`CommandRegistry` owns command registration and uniqueness.
- :class:`CLIApplication` owns parser creation, bootstrap, and dispatch.

Heavy domain logic still lives in dedicated command modules; this module keeps
the entry-point orchestration cohesive without pushing business logic into the
top-level router.
"""

from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


@dataclass(slots=True, frozen=True)
class CommandContext:
    """Immutable runtime context shared across sub-commands.

    Attributes:
        config_path: Resolved config path used for this process.
        verbose: Whether DEBUG logging was requested.
        config: Parsed YAML configuration payload.
    """

    config_path: Path
    verbose: bool
    config: dict[str, Any]


class BaseCLICommand(ABC):
    """Abstract polymorphic contract for one CLI sub-command.

    Concrete commands encapsulate:
    - parser configuration
    - execution behavior
    - command-specific help text
    """

    name: str
    help: str

    def register(
        self,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """Attach this command to the shared subparser router."""
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            **self.get_parser_kwargs(),
        )
        self.configure_parser(parser)
        parser.set_defaults(_command=self)

    def get_parser_kwargs(self) -> dict[str, Any]:
        """Return command-specific parser construction kwargs."""
        return {}

    @abstractmethod
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Register command-specific CLI arguments."""

    @abstractmethod
    def execute(
        self,
        args: argparse.Namespace,
        context: CommandContext,
    ) -> None:
        """Run the command with one bootstrapped runtime context."""


class CommandRegistry:
    """Encapsulated ordered registry for CLI command objects."""

    def __init__(self, commands: Iterable[BaseCLICommand] | None = None) -> None:
        self._commands: dict[str, BaseCLICommand] = {}
        if commands is not None:
            for command in commands:
                self.add(command)

    def add(self, command: BaseCLICommand) -> None:
        """Register one command, enforcing unique command names."""
        if command.name in self._commands:
            raise ValueError(f"Duplicate CLI command name: {command.name}")
        self._commands[command.name] = command

    def register_all(
        self,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """Attach every registered command to argparse subparsers."""
        for command in self._commands.values():
            command.register(subparsers)

    def names(self) -> list[str]:
        """Return the registered command names in declaration order."""
        return list(self._commands.keys())


class CLIApplication:
    """Own the CLI process lifecycle from parse to dispatch."""

    def __init__(
        self,
        *,
        prog: str,
        description: str,
        commands: Iterable[BaseCLICommand],
        default_config_path: str = "config/config.yaml",
    ) -> None:
        self._prog = prog
        self._description = description
        self._default_config_path = default_config_path
        self._registry = CommandRegistry(commands)

    def build_parser(self) -> argparse.ArgumentParser:
        """Build the root parser and attach all sub-commands."""
        parser = argparse.ArgumentParser(
            prog=self._prog,
            description=self._description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--config",
            type=str,
            default=self._default_config_path,
            help=f"Config file path (default: {self._default_config_path})",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable detailed logging (DEBUG level)",
        )

        subparsers = parser.add_subparsers(
            dest="command",
            title="commands",
            description="Available sub-commands",
            metavar="<command>",
        )
        subparsers.required = True
        self._registry.register_all(subparsers)
        return parser

    def bootstrap_context(self, args: argparse.Namespace) -> CommandContext:
        """Set up logging, load config, and create one immutable context."""
        setup_logging(verbose=bool(args.verbose))
        config_path = Path(str(args.config))
        config = load_config(str(config_path))
        return CommandContext(
            config_path=config_path,
            verbose=bool(args.verbose),
            config=config,
        )

    def run(self, argv: list[str] | None = None) -> None:
        """Parse args, bootstrap shared state, and dispatch polymorphically."""
        parser = self.build_parser()
        args = parser.parse_args(argv)
        context = self.bootstrap_context(args)

        command = getattr(args, "_command", None)
        if not isinstance(command, BaseCLICommand):
            parser.error("No valid sub-command resolved.")
        command.execute(args, context)


@dataclass(slots=True, frozen=True)
class SummaryItem:
    """One key-value line in a CLI summary block."""

    label: str
    value: Any


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.getLogger(__name__).info("Loaded configuration file: %s", path)
    return config


def log_banner(
    logger: logging.Logger,
    title: str,
    *,
    width: int = 60,
) -> None:
    """Emit one visually stable banner block."""
    logger.info("=" * width)
    logger.info("%s", title)
    logger.info("=" * width)


def log_summary(
    logger: logging.Logger,
    title: str,
    items: Iterable[SummaryItem | tuple[str, Any]],
    *,
    width: int = 60,
    key_width: int = 20,
) -> None:
    """Emit a consistent end-of-command summary block.

    Callers remain responsible for domain-specific value formatting; this
    helper standardizes the surrounding structure and key/value alignment.
    """
    log_banner(logger, title, width=width)
    for item in items:
        if isinstance(item, SummaryItem):
            label = item.label
            value = item.value
        else:
            label, value = item
        logger.info("  %-*s %s", key_width, f"{label}:", value)
    logger.info("=" * width)
