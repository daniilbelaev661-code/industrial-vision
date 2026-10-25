"""Модуль трекинга объектов (ByteTrack)."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.inference.types import Detection


class Tracker:
    """Обертка над Ultralytics YOLO для трекинга (присвоения Track ID)."""

    def __init__(self, model_path: str, confidence: float = 0.3):
        # Используем ту же модель, что и для детекции
        self.model = YOLO(model_path)
        self.confidence = confidence

    def update(self, frame: np.ndarray) -> list[Detection]:
        """Обрабатывает кадр, возвращает список детекций с Track ID."""
        # persist=True сохраняет состояние трекера между кадрами!
        # Если этого не сделать, трекер будет сбрасываться на каждом кадре.
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        detections: list[Detection] = []
        if not results:
            return detections
            
        result = results[0]
        names = result.names
        
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Извлекаем Track ID (может быть None, если объект только появился)
            track_id = int(box.id.item()) if box.id is not None else None

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=str(names[cls_id]),
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    track_id=track_id
                )
            )
        return detections