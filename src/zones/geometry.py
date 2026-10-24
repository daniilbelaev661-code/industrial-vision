"""Геометрия для работы с контролируемыми зонами (ROI)."""
from __future__ import annotations

import cv2
import numpy as np

from src.inference.types import Detection


def bottom_center(det: Detection) -> tuple[int, int]:
    """Возвращает точку контакта объекта с землей (нижний центр bbox)."""
    return (det.x1 + det.x2) // 2, det.y2


def point_in_polygon(point: tuple[int, int], polygon_px: list[tuple[int, int]]) -> bool:
    """Проверяет, находится ли точка внутри полигона (в пикселях).
    Использует cv2.pointPolygonTest. Возвращает True, если точка внутри 
    или на границе полигона.
    """
    if not polygon_px:
        return False
    contour = np.array(polygon_px, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0


def polygon_norm_to_px(
    polygon_norm: list[list[float]], 
    frame_width: int, 
    frame_height: int
) -> list[tuple[int, int]]:
    """Переводит нормализованные координаты [0..1] в пиксели."""
    return [
        (int(x * frame_width), int(y * frame_height))
        for x, y in polygon_norm
    ]