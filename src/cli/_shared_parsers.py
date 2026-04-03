"""Shared parent parsers for cross-sub-command arguments.

Defines reusable ``argparse.ArgumentParser`` fragments that multiple
sub-commands can inherit via ``parents=[...]``, following DRY.
"""

from __future__ import annotations

import argparse


def coco_data_parser() -> argparse.ArgumentParser:
    """Parent parser for COCO data-related flags.

    Provides:
        --skip-download, --categories
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳過 COCO 下載檢查 (已有資料時加速)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="目標動物類別 (default: 使用設定檔中的預設清單)",
    )
    return parser


def dataset_id_parser() -> argparse.ArgumentParser:
    """Parent parser for ``--dataset-id``.

    Provides:
        --dataset-id
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="使用既有 Dataset Asset id",
    )
    return parser
