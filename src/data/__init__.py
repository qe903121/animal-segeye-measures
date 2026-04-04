"""Data infrastructure module for COCO dataset and asset management.

Provides:
    AutoDownloader: Idempotent COCO val2017 downloader with retry logic.
    DatasetAssetLoader: Reader for exported dataset assets.
    DatasetAssetExporter: Export standardized dataset assets.
    build_lightweight_dataset_from_asset: Asset-only runtime dataset builder.
    COCODataLoader: GT annotation loader with multi-animal filtering.
    HumanLabelStore: Persist reusable human ground-truth labels.
    PredictionAssetLoader: Reader for exported prediction assets.
    PredictionAssetStore: Prediction asset schema and storage helper.
"""

from .asset_loader import DatasetAssetLoader, build_lightweight_dataset_from_asset
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
    "build_lightweight_dataset_from_asset",
    "HumanLabelStore",
    "PredictionAssetLoader",
    "PredictionAssetStore",
]
