"""Унифицированные типы данных для модуля детекции."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Одна детекция объекта на кадре."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2
