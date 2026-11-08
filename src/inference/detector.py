"""Обёртка над Ultralytics YOLO для одиночной и пакетной детекции."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .types import Detection

logger = logging.getLogger(__name__)


class Detector:
    """Детектор объектов на базе Ultralytics YOLO/ONNX."""

    def __init__(self, model_path: str | Path, confidence: float = 0.5):
        self.model_path = str(model_path)
        self.confidence = confidence
        logger.info("Loading YOLO model from %s", self.model_path)
        try:
            self.model = YOLO(self.model_path, task="detect")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Cannot load YOLO model: {self.model_path}") from exc

    @staticmethod
    def _convert_result(result: Any) -> list[Detection]:
        detections: list[Detection] = []
        if result is None or result.boxes is None:
            return detections
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=str(names[cls_id]),
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def predict(self, frame) -> list[Detection]:
        """Запускает детекцию на одном BGR-кадре."""
        results = self.model.predict(
            frame, conf=self.confidence, imgsz=480, verbose=False
        )
        return self._convert_result(results[0]) if results else []

    def predict_batch(self, frames: list) -> list[list[Detection]]:
        """Пакетная детекция. При несовместимом статическом ONNX batch делает fallback."""
        if not frames:
            return []
        try:
            results = self.model.predict(
                frames, conf=self.confidence, imgsz=480, verbose=False
            )
            if len(results) != len(frames):
                raise RuntimeError(
                    f"Model returned {len(results)} results for {len(frames)} images"
                )
            return [self._convert_result(result) for result in results]
        except Exception as exc:
            logger.warning(
                "Batch inference is not supported by %s; using per-image inference: %s",
                self.model_path,
                exc,
            )
            return [self.predict(frame) for frame in frames]
