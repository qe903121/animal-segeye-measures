#!/usr/bin/env python3
"""Fetch the official RTMPose AP-10K ONNX artifact for this repo.

The script intentionally downloads the model outside normal git history and
extracts only the runtime files needed by the ONNX backend:

- ``end2end.onnx``
- ``detail.json``
- ``pipeline.json``
- ``deploy.json``
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

_RUNTIME_FILES = {
    "end2end.onnx",
    "detail.json",
    "pipeline.json",
    "deploy.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the official RTMPose AP-10K ONNX artifact.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file path (default: config/config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    ai_cfg = config.get("eye_detection", {}).get("ai_model", {})

    artifact_url = str(ai_cfg.get("onnx_artifact_url", "")).strip()
    artifact_sha256 = str(ai_cfg.get("onnx_artifact_sha256", "")).strip().lower()
    model_path_raw = str(ai_cfg.get("onnx_model_path", "")).strip()
    if not artifact_url:
        raise SystemExit("Missing eye_detection.ai_model.onnx_artifact_url in config.")
    if not model_path_raw:
        raise SystemExit("Missing eye_detection.ai_model.onnx_model_path in config.")
    model_path = Path(model_path_raw)

    dest_dir = model_path.parent
    if model_path.is_file() and not args.force:
        print(f"ONNX model already exists: {model_path}")
        print("Use --force to re-download and overwrite.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rtmpose_onnx_") as tmpdir:
        archive_path = Path(tmpdir) / "artifact.zip"
        print(f"Downloading ONNX artifact from:\n  {artifact_url}")
        download_file(artifact_url, archive_path)

        actual_sha256 = sha256_file(archive_path)
        print(f"Downloaded archive SHA256:\n  {actual_sha256}")
        if artifact_sha256 and actual_sha256 != artifact_sha256:
            raise SystemExit(
                "Archive checksum mismatch.\n"
                f"Expected: {artifact_sha256}\n"
                f"Actual:   {actual_sha256}"
            )

        extract_runtime_files(archive_path, model_path, force=args.force)

    print("ONNX runtime files ready:")
    print(f"  model:   {model_path}")
    for companion in ("detail.json", "pipeline.json", "deploy.json"):
        companion_path = dest_dir / companion
        if companion_path.is_file():
            print(f"  {companion}: {companion_path}")


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url) as response, output_path.open("wb") as f:
        shutil.copyfileobj(response, f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_runtime_files(archive_path: Path, model_path: Path, *, force: bool) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        model_members = [name for name in names if name.endswith("/end2end.onnx")]
        if len(model_members) != 1:
            raise SystemExit(
                "Unable to locate a unique ONNX model inside the archive. "
                f"Found candidates: {model_members}"
            )

        artifact_root = model_members[0].rsplit("/", 1)[0]
        target_dir = model_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in sorted(_RUNTIME_FILES):
            member = f"{artifact_root}/{filename}"
            if member not in names:
                continue

            output_path = model_path if filename == "end2end.onnx" else target_dir / filename
            if output_path.exists() and not force:
                raise SystemExit(
                    f"Refusing to overwrite existing file without --force: {output_path}"
                )

            with zf.open(member) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        raise SystemExit(130)
