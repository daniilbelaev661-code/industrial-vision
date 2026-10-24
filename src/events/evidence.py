"""Сохранение доказательств событий (snapshot + JSON)."""
from __future__ import annotations

import json
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.events.event_engine import Event


class EvidenceManager:
    def __init__(self, base_dir: str = "events", snapshot_format: str = "jpg"):
        self.base_dir = Path(base_dir)
        self.snapshot_format = snapshot_format
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, event: Event, frame: np.ndarray) -> str:
        """Сохраняет snapshot и event.json. Возвращает путь к папке события."""
        # 1. Создаем папку по дате и UUID
        now = datetime.now(timezone.utc)
        date_path = self.base_dir / now.strftime("%Y/%m/%d")
        event_dir = date_path / event.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        # 2. Сохраняем снимок
        snapshot_name = f"snapshot.{self.snapshot_format}"
        snapshot_path = event_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), frame)

        # 3. Обновляем путь к снимку в событии и сохраняем JSON
        event.snapshot = str(snapshot_path.as_posix())
        
        json_path = event_dir / "event.json"
        with open(json_path, "w", encoding="utf-8") as f:
            # Превращаем dataclass в словарь и сохраняем
            json.dump(event.__dict__, f, indent=4, ensure_ascii=False)

        return str(event_dir)