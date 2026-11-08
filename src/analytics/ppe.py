"""Аналитика наличия СИЗ (каски и жилета) на человеке."""
from __future__ import annotations

from src.inference.types import Detection


def is_inside(inner_bbox: tuple[int, int, int, int], outer_bbox: tuple[int, int, int, int]) -> bool:
    """Проверяет, находится ли центр inner_bbox внутри outer_bbox."""
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    
    cx = (ix1 + ix2) // 2
    cy = (iy1 + iy2) // 2
    
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def check_person_ppe_crop(
    person: Detection, 
    ppe_detector, 
    frame: np.ndarray,
    min_person_height: int = 120
) -> tuple[dict[str, bool], list[Detection]] | None:
    person_height = person.y2 - person.y1
    if person_height < min_person_height:
        return None

    h, w = frame.shape[:2]
    
    # УВЕЛИЧИВАЕМ ОТСТУПЫ: 50% сверху и снизу, 20% по бокам
    padding_y = int(person_height * 0.5)
    padding_x = int((person.x2 - person.x1) * 0.2)

    x1 = max(0, person.x1 - padding_x)
    y1 = max(0, person.y1 - padding_y)
    x2 = min(w, person.x2 + padding_x)
    y2 = min(h, person.y2 + padding_y)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Прогоняем кроп через модель
    ppe_detections = ppe_detector.predict(crop)

    # Пересчитываем координаты обратно на полный кадр
    shifted_detections = []
    for d in ppe_detections:
        shifted_detections.append(Detection(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            x1=d.x1 + x1, y1=d.y1 + y1,
            x2=d.x2 + x1, y2=d.y2 + y1
        ))

    has_helmet = any(d.class_name == "Hardhat" for d in ppe_detections)
    has_vest = any(d.class_name == "Safety Vest" for d in ppe_detections)
    
    explicit_no_helmet = any(d.class_name == "NO-Hardhat" for d in ppe_detections)
    explicit_no_vest = any(d.class_name == "NO-Safety Vest" for d in ppe_detections)

    result = {
        "has_helmet": has_helmet and not explicit_no_helmet,
        "has_vest": has_vest and not explicit_no_vest
    }
    
    return result, shifted_detections