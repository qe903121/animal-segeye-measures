"""Sub-command: annotate — interactive human GT annotation."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from src.cli._shared_parsers import dataset_id_parser
from src.utils.cli import BaseCLICommand, CommandContext

logger = logging.getLogger(__name__)


class AnnotateCommand(BaseCLICommand):
    """Interactive GT annotation command."""

    name = "annotate"
    help = "Human GT manual annotation (terminal-only workflow)"

    def get_parser_kwargs(self) -> dict[str, Any]:
        return {"parents": [dataset_id_parser()]}

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--image-id",
            nargs="+",
            type=int,
            default=None,
            help="Only annotate specified image_id(s) (multiple allowed)",
        )
        parser.add_argument(
            "--from-csv",
            type=str,
            default=None,
            help="Read image_id list from CSV",
        )
        parser.add_argument(
            "--annotator",
            type=str,
            default=None,
            help="Annotator name (default: use config.annotation.default_annotator)",
        )
        parser.add_argument(
            "--skip-labeled",
            action="store_true",
            help="Skip annotation if label_status=LABELED (SKIPPED does not count as completed)",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Directly overwrite if annotation exists",
        )
        parser.add_argument(
            "--no-imshow",
            action="store_true",
            help="Do not use cv2.imshow for image display",
        )

    def execute(
        self,
        args: argparse.Namespace,
        context: CommandContext,
    ) -> None:
        main(args, context.config)


COMMAND = AnnotateCommand()


def register(subparsers: argparse._SubParsersAction) -> None:
    """Compatibility wrapper for legacy register-style imports."""
    COMMAND.register(subparsers)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the interactive annotation workflow."""
    import cv2

    from src.cli._annotation_helpers import (
        ACTION_QUIT,
        resolve_imshow_usage,
        resolve_image_ids,
        resolve_image_path,
        show_image_preview,
        get_existing_label,
        is_completed_label,
        build_used_ranks,
        prompt_dataset_id,
        prompt_coordinate,
        prompt_depth_rank,
        prompt_yes_no,
        special_result,
        now_iso,
        ACTION_REDO,
        ACTION_SKIP,
    )
    from src.data import DatasetAssetLoader, HumanLabelStore

    asset_loader = DatasetAssetLoader(config)
    gt_store = HumanLabelStore(config)

    dataset_id = args.dataset_id or prompt_dataset_id(asset_loader)
    asset = asset_loader.load(dataset_id)
    labels_df = gt_store.load_labels(dataset_id)
    annotation_cfg = config.get("annotation", {})
    annotator = (
        args.annotator
        or str(annotation_cfg.get("default_annotator", "anonymous"))
    )

    use_imshow = resolve_imshow_usage(
        no_imshow=args.no_imshow,
        mode="annotate",
        default_enabled=bool(
            annotation_cfg.get("use_imshow_by_default", True)
        ),
    )

    try:
        _run_annotate(
            asset=asset,
            gt_store=gt_store,
            labels_df=labels_df,
            explicit_ids=args.image_id,
            from_csv=args.from_csv,
            skip_labeled=args.skip_labeled,
            overwrite=args.overwrite,
            annotator=annotator,
            use_imshow=use_imshow,
        )
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _run_annotate(
    asset: Any,
    gt_store: Any,
    labels_df: Any,
    explicit_ids: list[int] | None,
    from_csv: str | None,
    skip_labeled: bool,
    overwrite: bool,
    annotator: str,
    use_imshow: bool,
) -> None:
    """Core annotation loop."""
    from src.cli._annotation_helpers import (
        ACTION_QUIT,
        build_used_ranks,
        get_existing_label,
        is_completed_label,
        prompt_yes_no,
        resolve_image_ids,
        resolve_image_path,
        show_image_preview,
    )

    selected_image_ids = resolve_image_ids(
        asset.instances,
        labels_df,
        explicit_ids=explicit_ids,
        from_csv=from_csv,
        skip_labeled=skip_labeled,
    )
    if not selected_image_ids:
        print("No images need annotation, exiting.")
        return

    print(f"\nDataset ID: {asset.dataset_id}")
    print(f"Pending images to process: {len(selected_image_ids)}")
    if explicit_ids or from_csv or not skip_labeled:
        print(f"Pending image_ids: {selected_image_ids}")
    else:
        print("Will automatically process unannotated images sequentially.")
    print()

    for image_id in selected_image_ids:
        image_rows = asset.instances.loc[
            asset.instances["image_id"] == image_id
        ].sort_values(by=["annotation_id"], kind="stable")

        if image_rows.empty:
            logger.warning("image_id=%s not found in dataset asset, skipping.", image_id)
            continue

        image_path = resolve_image_path(
            asset.manifest,
            image_rows.iloc[0],
        )
        preview_ok = show_image_preview(
            image_path=image_path,
            image_rows=image_rows,
            enabled=use_imshow,
        )
        if use_imshow and not preview_ok:
            use_imshow = False

        print("=" * 72)
        print(f"Image ID: {image_id} | Path: {image_path}")
        print(f"Annotations: {len(image_rows)}")
        print("=" * 72)

        used_ranks = build_used_ranks(labels_df, image_id)

        for _, row in image_rows.iterrows():
            annotation_id = int(row["annotation_id"])
            existing = get_existing_label(labels_df, annotation_id)

            if (
                existing is not None
                and skip_labeled
                and is_completed_label(existing)
            ):
                print(f"[SKIP] annotation_id={annotation_id} already has an existing annotation.")
                continue

            if existing is not None and not overwrite:
                print(
                    f"[INFO] annotation_id={annotation_id} already contains an annotation "
                    f"(status={existing.get('label_status', '')})."
                )
                if not prompt_yes_no("Overwrite this annotation?", default=False):
                    print("[KEEP] Retained existing annotation, skipped.\n")
                    continue

            result = _annotate_one(
                row=row,
                annotator=annotator,
                used_ranks=used_ranks,
            )
            if result == ACTION_QUIT:
                print("Quit instruction received, exiting safely.")
                return
            if result is None:
                continue

            gt_store.upsert_label(asset.dataset_id, result)
            labels_df = gt_store.load_labels(asset.dataset_id)

            if result["label_status"] == "LABELED":
                used_ranks[result["annotation_id"]] = int(result["depth_rank"])
                print(
                    f"[SAVED] annotation_id={annotation_id} "
                    f"depth_rank={result['depth_rank']}\n"
                )
            else:
                print(f"[SAVED] annotation_id={annotation_id} marked as SKIPPED\n")


def _annotate_one(
    row: Any,
    annotator: str,
    used_ranks: dict[int, int],
) -> dict[str, Any] | str | None:
    """Interactively annotate one object."""
    from src.cli._annotation_helpers import (
        ACTION_QUIT,
        ACTION_REDO,
        ACTION_SKIP,
        now_iso,
        prompt_coordinate,
        prompt_depth_rank,
        special_result,
    )

    annotation_id = int(row["annotation_id"])
    category = str(row["category"])
    bbox = (
        float(row["bbox_x"]),
        float(row["bbox_y"]),
        float(row["bbox_w"]),
        float(row["bbox_h"]),
    )

    while True:
        print(f"\n[Annotation #{annotation_id}] {category} bbox={bbox}")
        print("Input format: x,y")
        print("Special commands: skip / redo / quit")

        left_eye = prompt_coordinate("Left eye coordinates")
        if left_eye in (ACTION_QUIT, ACTION_SKIP):
            return special_result(row, annotator, left_eye)
        if left_eye == ACTION_REDO:
            continue

        right_eye = prompt_coordinate("Right eye coordinates")
        if right_eye in (ACTION_QUIT, ACTION_SKIP):
            return special_result(row, annotator, right_eye)
        if right_eye == ACTION_REDO:
            continue

        depth_rank = prompt_depth_rank(used_ranks, annotation_id)
        if depth_rank in (ACTION_QUIT, ACTION_SKIP):
            return special_result(row, annotator, depth_rank)
        if depth_rank == ACTION_REDO:
            continue

        print(
            "Confirm annotation: "
            f"L={left_eye}, R={right_eye}, depth_rank={depth_rank}"
        )
        confirm = input("Press Enter to save, or input redo / skip / quit: ").strip().lower()
        if confirm == "":
            return {
                "dataset_id": row["dataset_id"],
                "image_id": int(row["image_id"]),
                "annotation_id": annotation_id,
                "category": category,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "left_eye_x": left_eye[0],
                "left_eye_y": left_eye[1],
                "right_eye_x": right_eye[0],
                "right_eye_y": right_eye[1],
                "depth_rank": int(depth_rank),
                "label_status": "LABELED",
                "annotator": annotator,
                "labeled_at": now_iso(),
                "notes": "",
            }
        if confirm == "redo":
            continue
        if confirm == "skip":
            return special_result(row, annotator, ACTION_SKIP)
        if confirm == "quit":
            return ACTION_QUIT
        print("Unrecognized command, please re-enter current annotation.")
