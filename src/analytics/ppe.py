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


def check_person_ppe(
    person: Detection, 
    ppe_detections: list[Detection], 
    min_person_height: int = 120
) -> dict[str, bool] | None:
    """Проверяет наличие каски и жилета у человека.
    
    Возвращает словарь {'has_helmet': bool, 'has_vest': bool}.
    Возвращает None, если человек слишком мал для анализа.
    """
    person_height = person.y2 - person.y1
    if person_height < min_person_height:
        return None  # Объект слишком мал для надежной проверки

    # Ищем каски и жилеты, которые находятся в пределах рамки этого человека
    helmets = [d for d in ppe_detections if d.class_name == "helmet" and is_inside(d.bbox, person.bbox)]
    vests = [d for d in ppe_detections if d.class_name == "vest" and is_inside(d.bbox, person.bbox)]

    return {
        "has_helmet": len(helmets) > 0,
        "has_vest": len(vests) > 0
    }