"""Измерение FPS по скользящему окну."""
from __future__ import annotations

import time
from collections import deque


class FPSMeter:
    """Считает FPS по времени между последними N кадрами.

    Использование:
        meter = FPSMeter(window=30)
        ...
        fps = meter.tick()
    """

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Регистрирует кадр и возвращает текущий FPS."""
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0:
            return 0.0
        return (len(self._times) - 1) / dt

    def reset(self) -> None:
        self._times.clear()

    @property
    def samples(self) -> int:
        return len(self._times)
