"""Sub-command: review — GT visual inspection and overlay export."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.cli._shared_parsers import dataset_id_parser
from src.utils.cli import BaseCLICommand, CommandContext

logger = logging.getLogger(__name__)

REVIEW_DEFAULT_OUTPUT_DIR = "output/review_labels"


class ReviewCommand(BaseCLICommand):
    """GT review and overlay export command."""

    name = "review"
    help = "Visual Review of Human GT Annotations"

    def get_parser_kwargs(self) -> dict[str, Any]:
        return {"parents": [dataset_id_parser()]}

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--image-id",
            nargs="+",
            type=int,
            default=None,
            help="Only check specified image_id(s) (multiple allowed)",
        )
        parser.add_argument(
            "--from-csv",
            type=str,
            default=None,
            help="Read image_id list from CSV",
        )
        parser.add_argument(
            "--no-imshow",
            action="store_true",
            help="Do not use cv2.imshow for image display",
        )
        parser.add_argument(
            "--review-output-dir",
            type=str,
            default=None,
            help=(
                "Output image directory for review mode"
                "(default: use config.annotation.review_output_dir)"
            ),
        )

    def execute(
        self,
        args: argparse.Namespace,
        context: CommandContext,
    ) -> None:
        main(args, context.config)


COMMAND = ReviewCommand()


def register(subparsers: argparse._SubParsersAction) -> None:
    """Compatibility wrapper for legacy register-style imports."""
    COMMAND.register(subparsers)


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute the GT review workflow."""
    import cv2

    from src.cli._annotation_helpers import (
        WINDOW_NAME,
        resolve_image_path,
        resolve_imshow_usage,
        resolve_review_image_ids,
        build_review_rows,
        render_review_canvas,
        prompt_dataset_id,
    )
    from src.data import DatasetAssetLoader, HumanLabelStore

    asset_loader = DatasetAssetLoader(config)
    gt_store = HumanLabelStore(config)

    dataset_id = args.dataset_id or prompt_dataset_id(asset_loader)
    asset = asset_loader.load(dataset_id)
    labels_df = gt_store.load_labels(dataset_id)
    annotation_cfg = config.get("annotation", {})
    review_output_dir = str(
        args.review_output_dir
        or annotation_cfg.get("review_output_dir", REVIEW_DEFAULT_OUTPUT_DIR)
    )

    use_imshow = resolve_imshow_usage(
        no_imshow=args.no_imshow,
        mode="review",
        default_enabled=bool(
            annotation_cfg.get("use_imshow_by_default", True)
        ),
    )

    try:
        _run_review(
            asset=asset,
            labels_df=labels_df,
            explicit_ids=args.image_id,
            from_csv=args.from_csv,
            output_dir=review_output_dir,
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


def _run_review(
    asset: Any,
    labels_df: Any,
    explicit_ids: list[int] | None,
    from_csv: str | None,
    output_dir: str,
    use_imshow: bool,
) -> None:
    """Core review loop."""
    import cv2

    from src.cli._annotation_helpers import (
        WINDOW_NAME,
        build_review_rows,
        render_review_canvas,
        resolve_image_path,
        resolve_review_image_ids,
    )

    if labels_df.empty:
        print("No human labels available to review, exiting.")
        return

    review_rows = build_review_rows(asset.instances, labels_df)
    selected_image_ids = resolve_review_image_ids(
        review_rows,
        explicit_ids=explicit_ids,
        from_csv=from_csv,
    )
    if not selected_image_ids:
        print("No images available for review, exiting.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset ID: {asset.dataset_id}")
    print(f"Pending images to check: {len(selected_image_ids)}")
    print(f"Review output directory: {output_path}\n")

    saved_paths: list[Path] = []

    for index, image_id in enumerate(selected_image_ids, start=1):
        image_rows = review_rows.loc[
            review_rows["image_id"] == image_id
        ].sort_values(by=["annotation_id"], kind="stable")
        if image_rows.empty:
            continue

        image_path = resolve_image_path(asset.manifest, image_rows.iloc[0])
        canvas = render_review_canvas(image_path, image_rows)
        if canvas is None:
            continue

        out_file = output_path / f"labels_{image_id}.jpg"
        cv2.imwrite(str(out_file), canvas)
        saved_paths.append(out_file)

        print(
            f"[REVIEW {index}/{len(selected_image_ids)}] "
            f"image_id={image_id} -> {out_file}"
        )

        if use_imshow:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1)
            except cv2.error as exc:
                print(f"[WARN] cv2.imshow() failed, falling back to output-only mode: {exc}")
                use_imshow = False
            else:
                action = input("Press Enter for next, or input 'q' to quit review: ").strip().lower()
                if action == "q":
                    break

    print(f"\nReview completed. Output {len(saved_paths)} images to {output_path}")
