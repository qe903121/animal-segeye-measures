"""Evaluation engine with validator registry.

Manages a registry mapping task names to :class:`BaseValidator`
subclasses. At execution time, the engine instantiates the requested
validator, then drives its full lifecycle (evaluate → generate_report).

Typical usage::

    engine = EvaluationEngine(config)
    engine.register("localization", LocalizationValidator)
    results = engine.run("localization", dataset, output_dir)

    # Or run every registered validator:
    all_results = engine.run_all(dataset, output_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .base import BaseValidator

if TYPE_CHECKING:
    from src.data.loader import ImageRecord

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Registry-based evaluation orchestrator.

    Maintains an internal dictionary of ``{task_name: ValidatorClass}``
    pairs. When :meth:`run` is called, the engine:

    1. Looks up the validator class by *task_name*.
    2. Instantiates it with the shared ``config`` dict.
    3. Calls ``validator.evaluate(dataset)`` to compute metrics.
    4. Calls ``validator.generate_report(metrics, dataset, output_dir)``
       to persist artefacts.
    5. Returns the metrics dictionary.

    Attributes:
        _config: Full parsed configuration dictionary.
        _registry: Mapping of task names to validator classes.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the engine.

        Args:
            config: Full parsed ``config.yaml`` dictionary. Passed
                to each validator at instantiation time.
        """
        self._config = config
        self._registry: dict[str, type[BaseValidator]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        task_name: str,
        validator_cls: type[BaseValidator],
    ) -> None:
        """Register a validator class under a task name.

        Args:
            task_name: Unique string identifier (e.g. ``"localization"``).
            validator_cls: A **class** (not instance) that is a subclass
                of :class:`BaseValidator`.

        Raises:
            TypeError: *validator_cls* is not a subclass of
                :class:`BaseValidator`.
            ValueError: *task_name* is already registered.
        """
        if not (isinstance(validator_cls, type)
                and issubclass(validator_cls, BaseValidator)):
            raise TypeError(
                f"'{validator_cls}' 不是 BaseValidator 的子類別。"
            )
        if task_name in self._registry:
            raise ValueError(
                f"任務名稱 '{task_name}' 已被註冊。"
            )
        self._registry[task_name] = validator_cls
        logger.debug("已註冊驗證器: %s → %s", task_name, validator_cls.__name__)

    @property
    def registered_tasks(self) -> list[str]:
        """Return a sorted list of all registered task names."""
        return sorted(self._registry.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        task_name: str,
        dataset: list[ImageRecord],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Execute a single validator's full lifecycle.

        Args:
            task_name: Registered task name.
            dataset: Standardised dataset with ``eyes`` populated.
            output_dir: Root directory for this validator's output.
                A subdirectory named after *task_name* will be created
                automatically.

        Returns:
            The metrics dictionary produced by the validator.

        Raises:
            KeyError: *task_name* is not registered.
        """
        if task_name not in self._registry:
            available = ", ".join(self.registered_tasks) or "(none)"
            raise KeyError(
                f"未知的任務名稱: '{task_name}'。"
                f" 已註冊: {available}"
            )

        validator_cls = self._registry[task_name]
        task_output_dir = Path(output_dir) / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("開始驗證: %s (%s)", task_name, validator_cls.__name__)
        logger.info("=" * 60)

        # Step 1: Instantiate
        validator = validator_cls(self._config)

        # Step 2: Evaluate
        metrics = validator.evaluate(dataset)

        # Step 3: Generate report
        validator.generate_report(metrics, dataset, task_output_dir)

        logger.info(
            "驗證完成: %s → 報告輸出至 %s", task_name, task_output_dir
        )
        return metrics

    def run_all(
        self,
        dataset: list[ImageRecord],
        output_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        """Execute all registered validators sequentially.

        Args:
            dataset: Standardised dataset.
            output_dir: Root output directory. Each validator gets a
                subdirectory named after its task name.

        Returns:
            Mapping of ``{task_name: metrics_dict}`` for every
            registered validator.
        """
        if not self._registry:
            logger.warning("沒有已註冊的驗證器，跳過。")
            return {}

        all_metrics: dict[str, dict[str, Any]] = {}

        for task_name in self.registered_tasks:
            try:
                metrics = self.run(task_name, dataset, output_dir)
                all_metrics[task_name] = metrics
            except Exception:
                logger.exception("驗證器 '%s' 執行失敗。", task_name)
                all_metrics[task_name] = {"error": True}

        return all_metrics
