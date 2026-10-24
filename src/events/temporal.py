"""Временное подтверждение событий (N из M)."""
from __future__ import annotations

from collections import deque


class TemporalConfirm:
    """Подтверждает событие, если оно наблюдалось N раз за последние M кадров."""
    
    def __init__(self, n: int = 3, m: int = 5):
        self.n = n
        self.window = deque(maxlen=m)
        
    def update(self, detected: bool) -> bool:
        """Добавляет кадр в окно и возвращает True, если событие подтверждено."""
        self.window.append(bool(detected))
        # Подтверждаем только если окно заполнено и сумма >= N
        return len(self.window) == self.window.maxlen and sum(self.window) >= self.n