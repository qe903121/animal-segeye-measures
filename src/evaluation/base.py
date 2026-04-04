"""Abstract base class for all evaluation validators.

Defines the unified lifecycle interface that every validator must
implement: ``evaluate()`` → ``generate_report()``.

Design constraints:
    - No hardcoded file paths — all paths are passed via arguments.
    - Validators only process data and produce reports; they never
      execute neural network inference.
    - Strict type hints throughout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.loader import ImageRecord
    from src.data.prediction_loader import PredictionAsset


class BaseValidator(ABC):
    """Abstract base class for pipeline evaluation validators.

    Subclasses implement two lifecycle methods:

    1. :meth:`evaluate` — Receive a standardised dataset and compute
       quality metrics, returning them as a flat dictionary.
    2. :meth:`generate_report` — Receive the computed metrics and
       persist human-readable artefacts (charts, CSV, debug images)
       to a specified output directory.

    The :class:`EvaluationEngine` orchestrates these two steps in
    sequence, ensuring a consistent workflow across all validators.

    Attributes:
        _config: Full parsed ``config.yaml`` dictionary, injected by
            the engine at instantiation time.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the validator with global configuration.

        Args:
            config: Full parsed ``config.yaml`` dictionary.
        """
        self._config = config

    @abstractmethod
    def evaluate(
        self,
        dataset: list[ImageRecord],
        prediction_asset: PredictionAsset | None = None,
    ) -> dict[str, Any]:
        """Compute evaluation metrics from a standardised dataset.

        This method **must not** perform any neural network inference.
        It should only inspect the already-populated fields (e.g.
        ``annotation["eyes"]``) and derive aggregate statistics.

        Args:
            dataset: Runtime dataset — list of :class:`ImageRecord`
                dicts with ``eyes`` fields already populated.
            prediction_asset: Optional formal Prediction Asset bundle.
                When provided, validators should prefer consuming it
                directly instead of relying only on runtime dataset
                mutation.

        Returns:
            A dictionary of metric names to values. The structure is
            validator-specific; the engine treats it as opaque data
            and passes it unchanged to :meth:`generate_report`.
        """

    @abstractmethod
    def generate_report(
        self,
        metrics: dict[str, Any],
        dataset: list[ImageRecord],
        output_dir: Path,
        prediction_asset: PredictionAsset | None = None,
    ) -> None:
        """Persist evaluation artefacts to *output_dir*.

        Typical artefacts include CSV summaries, debug images, and
        formatted log output.

        Args:
            metrics: The dictionary returned by :meth:`evaluate`.
            dataset: The same dataset passed to :meth:`evaluate`
                (needed for debug visualisation).
            output_dir: Target directory for all output files.
                Must **not** be hardcoded — always use this argument.
            prediction_asset: Optional formal Prediction Asset bundle
                associated with the current evaluation run.
        """
        
    @staticmethod
    def fmt_coords(coords: list[float] | tuple[float, float] | None) -> str:
        """Format coordinate list as a CSV-friendly string.

        Args:
            coords: ``[x, y]`` or ``None``.

        Returns:
            Formatted string like ``"(123.4, 567.8)"`` or ``""`` if None.
        """
        if coords is None:
            return ""
        return f"({coords[0]:.1f}, {coords[1]:.1f})"
