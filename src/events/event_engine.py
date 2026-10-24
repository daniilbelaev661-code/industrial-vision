"""Движок событий: анализ детекций и генерация событий."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon, polygon_norm_to_px
from src.events.temporal import TemporalConfirm

logger = logging.getLogger(__name__)

@dataclass
class Event:
    event_type: str
    timestamp: str
    camera_id: str
    track_id: int | None
    confidence: float

class EventEngine:
    def __init__(self, config: dict):
        self.camera_id = config["camera"]["id"]
        self.zones_cfg = config["zones"]
        self.events_cfg = config["events"]
        
        # Настройки временного подтверждения и кулдауна
        tc_cfg = self.events_cfg["temporal_confirmation"]
        self.n = tc_cfg["n"]
        self.m = tc_cfg["m"]
        self.cooldown_sec = self.events_cfg["cooldown_sec"]
        
        # Хранилища состояний для каждого Track ID
        self._temporal_states: dict[int, TemporalConfirm] = {}
        self._last_event_time: dict[int, float] = {}

    def _check_cooldown(self, track_id: int) -> bool:
        """Проверяет, прошло ли достаточно времени с последнего события для этого ID."""
        now = time.time()
        last_time = self._last_event_time.get(track_id, 0.0)
        if now - last_time >= self.cooldown_sec:
            self._last_event_time[track_id] = now
            return True
        return False

    def process_frame(self, detections: list[Detection], frame_shape: tuple[int, int]) -> list[Event]:
        events = []
        h, w = frame_shape[:2]

        zone_name = list(self.zones_cfg.keys())[0]
        zone_norm = self.zones_cfg[zone_name]["polygon"]
        zone_px = polygon_norm_to_px(zone_norm, w, h)

        for det in detections:
            if det.class_name != "person" or det.track_id is None:
                continue

            pt = bottom_center(det)
            in_zone = point_in_polygon(pt, zone_px)
            
            # Инициализируем состояние для нового трека
            if det.track_id not in self._temporal_states:
                self._temporal_states[det.track_id] = TemporalConfirm(self.n, self.m)
                
            # Обновляем состояние
            confirmed = self._temporal_states[det.track_id].update(in_zone)
            
            # Если подтвердилось и прошел кулдаун
            if confirmed and self._check_cooldown(det.track_id):
                event = Event(
                    event_type=f"person_in_{zone_name}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    camera_id=self.camera_id,
                    track_id=det.track_id,
                    confidence=det.confidence
                )
                events.append(event)
                logger.warning(f"Событие подтверждено (N={self.n}): {event.event_type}, ID:{event.track_id}")

        return events