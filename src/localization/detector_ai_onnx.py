"""ONNX Runtime-based eye detector for the AP-10K RTMPose backend.

This detector preserves the repo's current AI contract:

- input: image + caller-provided GT bbox
- runtime: top-down pose inference
- output: the same ``EyeResult`` structure used by the PyTorch path

It intentionally does **not** reintroduce detector-driven full-scene
inference. The goal is to swap only the pose backend while keeping the rest of
the prediction/evaluation stack unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.data.loader import EyeResult

from .base import BaseEyeDetector

logger = logging.getLogger(__name__)

_DEFAULT_ALIAS = "animal"
_DEFAULT_PADDING = 1.25
_DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_DEFAULT_INPUT_SIZE = (256, 256)  # (w, h)


class OnnxPoseDetector(BaseEyeDetector):
    """RTMPose AP-10K eye detector backed by ONNX Runtime."""

    def __init__(self, config: dict) -> None:
        """Initialize the ONNX session and decode helpers."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed. Please install the packages "
                "specified in `requirements-ai.txt` to use runtime=onnx."
            ) from exc

        try:
            from mmpose.codecs import SimCCLabel
            from mmpose.datasets.transforms import GetBBoxCenterScale, TopdownAffine
        except ImportError as exc:
            raise ImportError(
                "MMPose runtime helpers are unavailable. The ONNX backend still "
                "relies on MMPose preprocessing/codec utilities."
            ) from exc

        ai_cfg = config.get("eye_detection", {}).get("ai_model", {})
        alias = str(ai_cfg.get("alias", _DEFAULT_ALIAS)).strip()
        if alias != _DEFAULT_ALIAS:
            raise ValueError(
                "The current ONNX backend only supports alias='animal'. "
                f"Received alias='{alias}'."
            )

        model_path_raw = str(ai_cfg.get("onnx_model_path", "")).strip()
        if not model_path_raw:
            raise ValueError(
                "Missing eye_detection.ai_model.onnx_model_path in config."
            )
        model_path = Path(model_path_raw)
        if not model_path.is_file():
            raise FileNotFoundError(
                "ONNX model not found: "
                f"{model_path}. Run `python tools/fetch_rtmpose_onnx.py` first."
            )

        metadata = _load_companion_metadata(model_path)
        self._threshold = float(ai_cfg.get("score_threshold", 0.3))
        self._model_path = model_path
        requested_providers = _normalize_providers(ai_cfg.get("providers"))
        self._providers = _select_providers(
            requested_providers=requested_providers,
            available_providers=ort.get_available_providers(),
        )
        self._session = ort.InferenceSession(
            str(model_path),
            providers=self._providers,
        )
        self._input_name = _resolve_input_name(self._session, metadata)
        self._output_names = _resolve_output_names(self._session, metadata)
        self._input_size = _resolve_input_size(self._session, metadata)
        self._mean = np.array(metadata.get("mean", _DEFAULT_MEAN), dtype=np.float32)
        self._std = np.array(metadata.get("std", _DEFAULT_STD), dtype=np.float32)
        self._to_rgb = bool(metadata.get("to_rgb", True))
        self._padding = float(metadata.get("padding", _DEFAULT_PADDING))

        codec_cfg = metadata.get("codec", {})
        codec_input_size = tuple(codec_cfg.get("input_size", self._input_size))
        self._codec = SimCCLabel(
            input_size=codec_input_size,
            sigma=tuple(codec_cfg.get("sigma", (5.66, 5.66))),
            simcc_split_ratio=float(codec_cfg.get("simcc_split_ratio", 2.0)),
            normalize=bool(codec_cfg.get("normalize", False)),
            use_dark=bool(codec_cfg.get("use_dark", False)),
        )
        self._bbox_to_center_scale = GetBBoxCenterScale(padding=self._padding)
        self._affine = TopdownAffine(input_size=self._input_size)

        logger.info(
            "Initialized OnnxPoseDetector: model=%s, requested_providers=%s, active_providers=%s, input_size=%s",
            self._model_path,
            requested_providers,
            self._session.get_providers(),
            self._input_size,
        )

    def detect(
        self,
        image: np.ndarray,
        bbox: list[float],
        mask: np.ndarray,
        category: str,
    ) -> EyeResult:
        """Detect eyes using bbox-anchored ONNX top-down pose inference."""
        del mask, category

        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h
        bbox_xyxy = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

        warped_image, input_center, input_scale, input_size = self._prepare_input(
            image=image,
            bbox_xyxy=bbox_xyxy,
        )
        input_tensor = _image_to_tensor(
            warped_image,
            mean=self._mean,
            std=self._std,
            to_rgb=self._to_rgb,
        )

        try:
            simcc_x, simcc_y = self._session.run(
                self._output_names,
                {self._input_name: input_tensor},
            )
        except Exception as exc:
            logger.warning("ONNX inference failed for bbox=%s: %s", bbox, exc)
            return _make_result("FAILED_NOT_FOUND")

        keypoints, scores = self._codec.decode(simcc_x, simcc_y)
        if keypoints.size == 0 or scores.size == 0:
            return _make_result("FAILED_NOT_FOUND")

        keypoints = _map_keypoints_to_image(
            keypoints=keypoints,
            input_center=input_center,
            input_scale=input_scale,
            input_size=np.asarray(input_size, dtype=np.float32),
        )[0]
        scores = np.asarray(scores, dtype=np.float32)[0]

        if len(scores) < 2:
            logger.warning(
                "Insufficient predicted keypoints (<2). Verify the ONNX model uses AP-10K format."
            )
            return _make_result("FAILED_NOT_FOUND")

        all_keypoints = [[float(pt[0]), float(pt[1])] for pt in keypoints]
        all_scores = [float(score) for score in scores]

        l_pt, l_score = keypoints[0], float(scores[0])
        r_pt, r_score = keypoints[1], float(scores[1])

        left_eye = [float(l_pt[0]), float(l_pt[1])] if l_score >= self._threshold else None
        right_eye = [float(r_pt[0]), float(r_pt[1])] if r_score >= self._threshold else None

        if left_eye is not None and right_eye is not None:
            status = "SUCCESS"
        elif left_eye is not None or right_eye is not None:
            status = "SINGLE_EYE"
        else:
            status = "FAILED_NOT_FOUND"

        valid_scores = [score for score in (l_score, r_score) if score >= self._threshold]
        confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        logger.warning(
            "ONNX Detection (bbox=[%.1f,%.1f,%.1f,%.1f], conf=%.2f): "
            "L=(%.1f,%.1f/s:%.2f) R=(%.1f,%.1f/s:%.2f) -> %s",
            x_min,
            y_min,
            x_max,
            y_max,
            confidence,
            l_pt[0],
            l_pt[1],
            l_score,
            r_pt[0],
            r_pt[1],
            r_score,
            status,
        )

        return {
            "status": status,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "confidence": round(confidence, 3),
            "all_keypoints": all_keypoints,
            "all_scores": all_scores,
        }

    def _prepare_input(
        self,
        *,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        """Apply the same bbox-to-input preprocessing used by the PyTorch path."""
        results: dict[str, Any] = {
            "img": image.copy(),
            "bbox": bbox_xyxy,
            "bbox_score": np.ones(1, dtype=np.float32),
        }
        results = self._bbox_to_center_scale.transform(results)
        results = self._affine.transform(results)
        return (
            np.asarray(results["img"], dtype=np.float32),
            np.asarray(results["input_center"], dtype=np.float32),
            np.asarray(results["input_scale"], dtype=np.float32),
            tuple(int(v) for v in results["input_size"]),
        )


def _image_to_tensor(
    image: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
) -> np.ndarray:
    """Normalize one warped image and convert it into NCHW float32."""
    processed = image.astype(np.float32, copy=False)
    if to_rgb:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = (processed - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    processed = np.transpose(processed, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(processed, dtype=np.float32)


def _map_keypoints_to_image(
    *,
    keypoints: np.ndarray,
    input_center: np.ndarray,
    input_scale: np.ndarray,
    input_size: np.ndarray,
) -> np.ndarray:
    """Map keypoints from model-input space back to original image space."""
    restored = np.asarray(keypoints, dtype=np.float32).copy()
    restored[..., :2] = (
        restored[..., :2] / input_size.reshape(1, 1, 2) * input_scale.reshape(1, 1, 2)
        + input_center.reshape(1, 1, 2)
        - 0.5 * input_scale.reshape(1, 1, 2)
    )
    return restored


def _load_companion_metadata(model_path: Path) -> dict[str, Any]:
    """Load optional JSON metadata placed alongside the ONNX model."""
    metadata: dict[str, Any] = {}

    detail_path = model_path.with_name("detail.json")
    if detail_path.is_file():
        with detail_path.open("r", encoding="utf-8") as f:
            detail = json.load(f)
        onnx_cfg = detail.get("onnx_config", {})
        metadata["input_name"] = _first_item(onnx_cfg.get("input_names"))
        output_names = onnx_cfg.get("output_names")
        if isinstance(output_names, list) and output_names:
            metadata["output_names"] = [str(name) for name in output_names]

    pipeline_path = model_path.with_name("pipeline.json")
    if pipeline_path.is_file():
        with pipeline_path.open("r", encoding="utf-8") as f:
            pipeline = json.load(f)
        tasks = pipeline.get("pipeline", {}).get("tasks", [])
        preprocess = next((task for task in tasks if task.get("name") == "Preprocess"), None)
        postprocess = next((task for task in tasks if task.get("name") == "postprocess"), None)

        if preprocess is not None:
            for transform in preprocess.get("transforms", []):
                transform_type = str(transform.get("type", ""))
                if transform_type == "TopDownGetBboxCenterScale":
                    metadata["padding"] = float(transform.get("padding", _DEFAULT_PADDING))
                elif transform_type == "Normalize":
                    metadata["mean"] = [
                        float(value) for value in transform.get("mean", _DEFAULT_MEAN.tolist())
                    ]
                    metadata["std"] = [
                        float(value) for value in transform.get("std", _DEFAULT_STD.tolist())
                    ]
                    metadata["to_rgb"] = bool(transform.get("to_rgb", True))

        if postprocess is not None:
            params = postprocess.get("params", {})
            codec_cfg: dict[str, Any] = {}
            if "input_size" in params:
                codec_cfg["input_size"] = tuple(int(v) for v in params["input_size"])
            if "sigma" in params:
                codec_cfg["sigma"] = tuple(float(v) for v in params["sigma"])
            if "simcc_split_ratio" in params:
                codec_cfg["simcc_split_ratio"] = float(params["simcc_split_ratio"])
            if "normalize" in params:
                codec_cfg["normalize"] = bool(params["normalize"])
            if "use_dark" in params:
                codec_cfg["use_dark"] = bool(params["use_dark"])
            if codec_cfg:
                metadata["codec"] = codec_cfg

    return metadata


def _resolve_input_name(session: Any, metadata: dict[str, Any]) -> str:
    """Resolve the ONNX input tensor name."""
    input_name = str(metadata.get("input_name", "")).strip()
    if input_name:
        return input_name
    return str(session.get_inputs()[0].name)


def _resolve_output_names(session: Any, metadata: dict[str, Any]) -> list[str]:
    """Resolve ONNX output tensor names in the expected order."""
    session_names = [str(output.name) for output in session.get_outputs()]
    configured = metadata.get("output_names")
    if isinstance(configured, list) and configured and all(name in session_names for name in configured):
        return [str(name) for name in configured]
    return session_names[:2]


def _resolve_input_size(session: Any, metadata: dict[str, Any]) -> tuple[int, int]:
    """Resolve model input size as ``(w, h)``."""
    codec_cfg = metadata.get("codec", {})
    codec_input_size = codec_cfg.get("input_size")
    if codec_input_size:
        return tuple(int(v) for v in codec_input_size)

    shape = list(session.get_inputs()[0].shape)
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return (int(shape[3]), int(shape[2]))
    return _DEFAULT_INPUT_SIZE


def _normalize_providers(raw_providers: Any) -> list[str]:
    """Normalize provider configuration into a non-empty string list."""
    if isinstance(raw_providers, list):
        providers = [str(provider).strip() for provider in raw_providers if str(provider).strip()]
        if providers:
            return providers
    return ["CPUExecutionProvider"]


def _select_providers(
    *,
    requested_providers: list[str],
    available_providers: list[str],
) -> list[str]:
    """Return the usable providers in request order, with a safe CPU fallback."""
    available_lookup = {str(provider).strip() for provider in available_providers if str(provider).strip()}
    selected = [provider for provider in requested_providers if provider in available_lookup]
    missing = [provider for provider in requested_providers if provider not in available_lookup]

    if missing:
        logger.warning(
            "Requested ONNX providers are unavailable in this environment: %s. "
            "Available providers: %s",
            missing,
            available_providers,
        )

    if selected:
        return selected

    if "CPUExecutionProvider" in available_lookup:
        logger.warning(
            "Falling back to CPUExecutionProvider because none of the requested "
            "providers are currently available."
        )
        return ["CPUExecutionProvider"]

    raise RuntimeError(
        "No usable ONNX Runtime execution provider is available. "
        f"Requested={requested_providers}, available={available_providers}"
    )


def _first_item(values: Any) -> str:
    """Return the first string-like item from a sequence."""
    if isinstance(values, list) and values:
        return str(values[0])
    return ""


def _make_result(status: str) -> EyeResult:
    """Create a failed EyeResult with the given status."""
    return {
        "status": status,
        "left_eye": None,
        "right_eye": None,
        "confidence": 0.0,
    }
