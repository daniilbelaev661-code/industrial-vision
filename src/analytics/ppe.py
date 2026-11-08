"""Аналитика наличия СИЗ (каски и жилета) на человеке.

Поддерживает пакетную обработку person-crops. Если ONNX-модель экспортирована
со статическим batch=1 и не принимает несколько изображений, автоматически
используется безопасный последовательный fallback.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.inference.types import Detection

logger = logging.getLogger(__name__)


def is_inside(inner_bbox: tuple[int, int, int, int], outer_bbox: tuple[int, int, int, int]) -> bool:
    """Проверяет, находится ли центр inner_bbox внутри outer_bbox."""
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    cx = (ix1 + ix2) // 2
    cy = (iy1 + iy2) // 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _prepare_person_crop(
    person: Detection,
    frame: np.ndarray,
    min_person_height: int,
    padding_y_ratio: float,
    padding_x_ratio: float,
) -> tuple[np.ndarray, int, int] | None:
    person_height = person.y2 - person.y1
    person_width = person.x2 - person.x1
    if person_height < min_person_height or person_width <= 0:
        return None

    h, w = frame.shape[:2]
    padding_y = int(person_height * padding_y_ratio)
    padding_x = int(person_width * padding_x_ratio)
    x1 = max(0, person.x1 - padding_x)
    y1 = max(0, person.y1 - padding_y)
    x2 = min(w, person.x2 + padding_x)
    y2 = min(h, person.y2 + padding_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, x1, y1


def _to_full_frame_detections(
    detections: list[Detection], offset_x: int, offset_y: int
) -> list[Detection]:
    return [
        Detection(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            x1=d.x1 + offset_x,
            y1=d.y1 + offset_y,
            x2=d.x2 + offset_x,
            y2=d.y2 + offset_y,
        )
        for d in detections
    ]


def _summarize_ppe(detections: list[Detection]) -> dict[str, bool]:
    has_helmet = any(d.class_name == "Hardhat" for d in detections)
    has_vest = any(d.class_name == "Safety Vest" for d in detections)
    explicit_no_helmet = any(d.class_name == "NO-Hardhat" for d in detections)
    explicit_no_vest = any(d.class_name == "NO-Safety Vest" for d in detections)
    return {
        "has_helmet": has_helmet and not explicit_no_helmet,
        "has_vest": has_vest and not explicit_no_vest,
    }


def check_person_ppe_crops(
    persons: list[Detection],
    ppe_detector: Any,
    frame: np.ndarray,
    min_person_height: int = 120,
    padding_y_ratio: float = 0.5,
    padding_x_ratio: float = 0.2,
) -> dict[int, tuple[dict[str, bool], list[Detection]]]:
    """Проверяет СИЗ сразу у нескольких людей.

    Возвращает словарь track_id -> (результат СИЗ, детекции в координатах кадра).
    Люди без track_id пропускаются, поскольку результат нельзя надёжно связать
    с конкретным треком.
    """
    prepared: list[tuple[int, np.ndarray, int, int]] = []
    for person in persons:
        if person.track_id is None:
            continue
        crop_data = _prepare_person_crop(
            person, frame, min_person_height, padding_y_ratio, padding_x_ratio
        )
        if crop_data is not None:
            crop, offset_x, offset_y = crop_data
            prepared.append((person.track_id, crop, offset_x, offset_y))

    if not prepared:
        return {}

    crops = [item[1] for item in prepared]
    # Detector.predict_batch сам делает fallback на одиночные вызовы, если
    # конкретная ONNX-модель не поддерживает batch > 1.
    if hasattr(ppe_detector, "predict_batch"):
        batch_detections = ppe_detector.predict_batch(crops)
    else:
        batch_detections = [ppe_detector.predict(crop) for crop in crops]

    output: dict[int, tuple[dict[str, bool], list[Detection]]] = {}
    for (track_id, _crop, offset_x, offset_y), detections in zip(prepared, batch_detections):
        output[track_id] = (
            _summarize_ppe(detections),
            _to_full_frame_detections(detections, offset_x, offset_y),
        )
    return output


def check_person_ppe_crop(
    person: Detection,
    ppe_detector: Any,
    frame: np.ndarray,
    min_person_height: int = 120,
) -> tuple[dict[str, bool], list[Detection]] | None:
    """Обратнос совместимая функция для проверки одного человека."""
    if person.track_id is None:
        # Для старых вызывающих участков допускаем одиночную проверку без ID.
        crop_data = _prepare_person_crop(person, frame, min_person_height, 0.5, 0.2)
        if crop_data is None:
            return None
        crop, offset_x, offset_y = crop_data
        detections = ppe_detector.predict(crop)
        return _summarize_ppe(detections), _to_full_frame_detections(detections, offset_x, offset_y)

    result = check_person_ppe_crops([person], ppe_detector, frame, min_person_height)
    return result.get(person.track_id)
