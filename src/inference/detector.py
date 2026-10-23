"""Обертка над YOLO для получения списка детекций."""
from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

from .types import Detection

logger = logging.getLogger(__name__)


class Detector:
    """Детектор объектов на базе Ultralytics YOLO."""

    def __init__(self, model_path: str | Path, confidence: float = 0.5):
        self.model_path = str(model_path)
        self.confidence = confidence
        logger.info("Loading YOLO model from %s", self.model_path)
        
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Cannot load YOLO model: {self.model_path}") from exc

    def predict(self, frame) -> list[Detection]:
        """Запускает детекцию на кадре (BGR numpy array)."""
        results = self.model.predict(
            frame, 
            conf=self.confidence, 
            verbose=False
        )
        
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
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
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )
        return detections
