"""Data infrastructure module for COCO dataset management.

Provides:
    AutoDownloader: Idempotent COCO val2017 downloader with retry logic.
    DatasetAssetLoader: Reader for exported Phase 1 dataset assets.
    DatasetAssetExporter: Export standardized Phase 1 dataset assets.
    COCODataLoader: GT annotation loader with multi-animal filtering.
    HumanLabelStore: Persist reusable human ground-truth labels.
    PredictionAssetLoader: Reader for exported prediction assets.
    PredictionAssetStore: Prediction asset schema and storage helper.
"""

from .asset_loader import DatasetAssetLoader
from .asset_exporter import DatasetAssetExporter
from .downloader import AutoDownloader
from .gt_store import HumanLabelStore
from .loader import COCODataLoader
from .prediction_loader import PredictionAssetLoader
from .prediction_store import PredictionAssetStore

__all__ = [
    "AutoDownloader",
    "COCODataLoader",
    "DatasetAssetExporter",
    "DatasetAssetLoader",
    "HumanLabelStore",
    "PredictionAssetLoader",
    "PredictionAssetStore",
]
