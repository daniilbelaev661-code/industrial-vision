"""Движок событий: анализ детекций и генерация событий."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon, polygon_norm_to_px
from src.events.temporal import TemporalConfirm
from src.analytics.ppe import check_person_ppe

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
        
        tc_cfg = self.events_cfg["temporal_confirmation"]
        self.n = tc_cfg["n"]
        self.m = tc_cfg["m"]
        self.cooldown_sec = self.events_cfg["cooldown_sec"]
        
        self._temporal_states: dict[str, TemporalConfirm] = {} # Ключ: "тип_id"
        self._last_event_time: dict[str, float] = {}

    def _check_cooldown(self, event_key: str) -> bool:
        now = time.time()
        last_time = self._last_event_time.get(event_key, 0.0)
        if now - last_time >= self.cooldown_sec:
            self._last_event_time[event_key] = now
            return True
        return False

    def _process_event(self, event_type: str, det: Detection, events_list: list) -> None:
        """Вспомогательный метод для обработки события с учетом cooldown."""
        key = f"{event_type}_{det.track_id}"
        if self._check_cooldown(key):
            events_list.append(Event(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                camera_id=self.camera_id,
                track_id=det.track_id,
                confidence=det.confidence
            ))
            logger.warning(f"Событие: {event_type}, ID:{det.track_id}")

    def process_frame(
        self, 
        person_detections: list[Detection], 
        ppe_detections: list[Detection],
        frame_shape: tuple[int, int]
    ) -> list[Event]:
        events = []
        h, w = frame_shape[:2]

        zone_name = list(self.zones_cfg.keys())[0]
        zone_norm = self.zones_cfg[zone_name]["polygon"]
        zone_px = polygon_norm_to_px(zone_norm, w, h)

        for det in person_detections:
            if det.class_name != "person" or det.track_id is None:
                continue

            # 1. Проверка ROI (Зона)
            pt = bottom_center(det)
            in_zone = point_in_polygon(pt, zone_px)
            zone_key = f"zone_{det.track_id}"
            
            if zone_key not in self._temporal_states:
                self._temporal_states[zone_key] = TemporalConfirm(self.n, self.m)
            
            if self._temporal_states[zone_key].update(in_zone):
                self._process_event(f"person_in_{zone_name}", det, events)

            # 2. Проверка СИЗ (PPE)
            ppe_result = check_person_ppe(det, ppe_detections)
            if ppe_result is not None:
                # Проверяем каску
                helmet_key = f"helmet_{det.track_id}"
                if helmet_key not in self._temporal_states:
                    self._temporal_states[helmet_key] = TemporalConfirm(self.n, self.m)
                if self._temporal_states[helmet_key].update(not ppe_result["has_helmet"]):
                    self._process_event("ppe_helmet_missing", det, events)

                # Проверяем жилет
                vest_key = f"vest_{det.track_id}"
                if vest_key not in self._temporal_states:
                    self._temporal_states[vest_key] = TemporalConfirm(self.n, self.m)
                if self._temporal_states[vest_key].update(not ppe_result["has_vest"]):
                    self._process_event("ppe_vest_missing", det, events)

        return events