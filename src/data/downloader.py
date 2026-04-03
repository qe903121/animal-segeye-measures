"""Idempotent COCO val2017 dataset downloader.

Downloads COCO val2017 images and annotations if not already present.
Provides progress tracking via tqdm and exponential-backoff retry logic.

Typical usage:
    >>> from src.data.downloader import AutoDownloader
    >>> downloader = AutoDownloader(config)
    >>> downloader.ensure_ready()
"""

from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AutoDownloader:
    """Idempotent COCO dataset downloader with streaming and retry support.

    Checks local filesystem for existing data before initiating any network
    requests. Downloads use chunked streaming to avoid excessive memory usage,
    and display real-time progress via tqdm.

    Attributes:
        data_root: Root directory for COCO data (e.g., ``data/coco``).
        images_url: URL for val2017 images ZIP.
        annotations_url: URL for annotations ZIP.
        images_dir: Name of the images subdirectory (e.g., ``val2017``).
        annotations_file: Name of the target annotation JSON file.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the downloader from a parsed config dictionary.

        Args:
            config: Full parsed ``config.yaml`` dictionary. Expected keys:
                - ``coco.data_root``
                - ``coco.urls.images``
                - ``coco.urls.annotations``
                - ``coco.images_dir``
                - ``coco.annotations_file``
                - ``download.*`` (chunk_size, max_retries, etc.)
        """
        coco_cfg = config["coco"]
        dl_cfg = config.get("download", {})

        self.data_root = Path(coco_cfg["data_root"])
        self.images_url: str = coco_cfg["urls"]["images"]
        self.annotations_url: str = coco_cfg["urls"]["annotations"]
        self.images_dir: str = coco_cfg["images_dir"]
        self.annotations_file: str = coco_cfg["annotations_file"]

        # Download tuning parameters
        self._chunk_size: int = dl_cfg.get("chunk_size", 8192)
        self._max_retries: int = dl_cfg.get("max_retries", 3)
        self._retry_backoff: float = dl_cfg.get("retry_backoff", 2.0)
        self._connect_timeout: int = dl_cfg.get("connect_timeout", 30)
        self._read_timeout: int = dl_cfg.get("read_timeout", 300)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_ready(self) -> None:
        """Ensure both images and annotations are available locally.

        This is the main entry point. It is **idempotent**: calling it multiple
        times will only download data that is genuinely missing.

        Raises:
            ConnectionError: All retry attempts exhausted for a download.
            zipfile.BadZipFile: A downloaded ZIP file is corrupted.
        """
        self.data_root.mkdir(parents=True, exist_ok=True)

        if self._check_images_exist():
            logger.info("影像目錄已存在，跳過下載: %s", self._images_path)
        else:
            logger.info("開始下載 COCO val2017 影像...")
            self._download_and_extract(
                url=self.images_url,
                extract_to=self.data_root,
                desc="下載 val2017 影像",
            )

        if self._check_annotations_exist():
            logger.info("標註檔已存在，跳過下載: %s", self._annotations_path)
        else:
            logger.info("開始下載 COCO annotations...")
            self._download_and_extract(
                url=self.annotations_url,
                extract_to=self.data_root,
                desc="下載 annotations",
                target_member=self.annotations_file,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _images_path(self) -> Path:
        """Resolved path to the images directory."""
        return self.data_root / self.images_dir

    @property
    def _annotations_path(self) -> Path:
        """Resolved path to the target annotation JSON."""
        return self.data_root / "annotations" / self.annotations_file

    def _check_images_exist(self) -> bool:
        """Return True if the images directory exists and is non-empty."""
        images_dir = self._images_path
        if not images_dir.is_dir():
            return False
        # Quick sanity: at least some .jpg files present
        return any(images_dir.glob("*.jpg"))

    def _check_annotations_exist(self) -> bool:
        """Return True if the target annotation JSON exists."""
        return self._annotations_path.is_file()

    def _download_and_extract(
        self,
        url: str,
        extract_to: Path,
        desc: str,
        target_member: str | None = None,
    ) -> None:
        """Download a ZIP and extract it (or a single member) to *extract_to*.

        Args:
            url: Remote URL of the ZIP file.
            extract_to: Local directory to extract contents into.
            desc: Human-readable description for the progress bar.
            target_member: If set, only extract the ZIP member whose filename
                matches this value (searched across all paths in the archive).
                Everything else is ignored to save disk space.

        Raises:
            ConnectionError: Download failed after all retries.
            zipfile.BadZipFile: The downloaded file is not a valid ZIP.
        """
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self._download_with_retry(url, tmp_path, desc)

            logger.info("解壓縮中: %s", tmp_path.name)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                if target_member is not None:
                    self._extract_single_member(zf, target_member, extract_to)
                else:
                    zf.extractall(extract_to)
            logger.info("解壓縮完成 → %s", extract_to)
        finally:
            # Always clean up the temp ZIP regardless of success/failure
            if tmp_path.exists():
                tmp_path.unlink()
                logger.debug("已刪除暫存檔: %s", tmp_path)

    def _extract_single_member(
        self,
        zf: zipfile.ZipFile,
        target_filename: str,
        extract_to: Path,
    ) -> None:
        """Extract only the ZIP member whose basename matches *target_filename*.

        The member is extracted preserving its internal directory structure
        (e.g., ``annotations/instances_val2017.json``).

        Args:
            zf: An open ZipFile object.
            target_filename: The basename to search for (e.g.
                ``instances_val2017.json``).
            extract_to: Destination root directory.

        Raises:
            FileNotFoundError: No member with matching basename found.
        """
        for member in zf.namelist():
            if Path(member).name == target_filename:
                zf.extract(member, extract_to)
                logger.info("已解壓目標檔案: %s", member)
                return
        raise FileNotFoundError(
            f"ZIP 內找不到目標檔案 '{target_filename}'。"
            f" 可用成員: {zf.namelist()[:10]}..."
        )

    def _download_with_retry(self, url: str, dest: Path, desc: str) -> None:
        """Stream-download *url* to *dest* with exponential-backoff retry.

        Uses ``requests.get(stream=True)`` combined with tqdm for real-time
        progress reporting. On failure, retries up to ``max_retries`` times
        with exponentially increasing wait intervals.

        Args:
            url: Remote file URL.
            dest: Local file path to write to.
            desc: Progress bar description text.

        Raises:
            ConnectionError: All retries exhausted.
        """
        timeout = (self._connect_timeout, self._read_timeout)

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    "下載嘗試 %d/%d: %s", attempt, self._max_retries, url
                )
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with (
                    open(dest, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=desc,
                        ncols=80,
                    ) as pbar,
                ):
                    for chunk in response.iter_content(
                        chunk_size=self._chunk_size
                    ):
                        written = f.write(chunk)
                        pbar.update(written)

                logger.info("下載完成: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
                return  # Success — exit retry loop

            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                if attempt == self._max_retries:
                    raise ConnectionError(
                        f"下載失敗（已重試 {self._max_retries} 次）: {url}"
                    ) from exc

                wait = self._retry_backoff ** attempt
                logger.warning(
                    "下載失敗 (嘗試 %d/%d): %s — %s 秒後重試",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
