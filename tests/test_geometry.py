"""Unit-тесты для геометрии ROI."""
from __future__ import annotations

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon, polygon_norm_to_px


def test_bottom_center():
    det = Detection(
        class_id=0, class_name="person", confidence=0.9,
        x1=100, y1=200, x2=300, y2=400
    )
    # (100+300)//2 = 200, y2 = 400
    assert bottom_center(det) == (200, 400)


def test_point_in_polygon_inside():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), polygon) is True


def test_point_in_polygon_outside():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((150, 50), polygon) is False
    assert point_in_polygon((50, 150), polygon) is False


def test_point_on_boundary():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((0, 0), polygon) is True
    assert point_in_polygon((100, 100), polygon) is True


def test_polygon_norm_to_px():
    polygon_norm = [[0.1, 0.4], [0.9, 0.4], [0.9, 0.95], [0.1, 0.95]]
    px = polygon_norm_to_px(polygon_norm, 1280, 720)
    assert px == [(128, 288), (1152, 288), (1152, 684), (128, 684)]